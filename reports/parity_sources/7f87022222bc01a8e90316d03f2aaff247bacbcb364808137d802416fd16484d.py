"""Bounded cached/native parity gate; never interpret this as task accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import torch
from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV


@torch.no_grad()
def check(adapter, ids, tokens=3):
    adapter.prefill_kv_cache(ids[:, :-1])
    rows = []
    for _ in range(tokens):
        native = adapter.model(input_ids=ids, use_cache=False, exit_at_step=adapter.total_steps - 1).logits[:, -1:, :]
        result = run_scheduled_trajectory(adapter, ids[:, -1:], method='S0', use_cached_step=True, record_snapshots=False, record_pca=False)
        actual = result.primary_logits[:, -1:, :]
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(native).all())
        difference = float((actual.float() - native.float()).abs().max())
        top1 = bool(torch.equal(actual.argmax(-1), native.argmax(-1)))
        lengths = [k.shape[2] for k in adapter._active_kv_cache.key_cache]
        rows.append({'prefix_length': ids.shape[1], 'max_abs_difference': difference, 'top1_match': top1, 'finite': finite, 'cache_lengths_match': all(n == ids.shape[1] for n in lengths)})
        ids = torch.cat([ids, native.argmax(-1)], dim=1)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='ByteDance/Ouro-1.4B-Thinking')
    parser.add_argument('--revision', default=PINNED_OURO_THINKING_REV)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--attention', choices=['eager', 'sdpa'], default='sdpa')
    parser.add_argument('--prefix-lengths', type=int, nargs='+', default=[1, 5])
    parser.add_argument('--tokens', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or any(length < 1 for length in args.prefix_lengths):
        parser.error('Prefix lengths and token count must be positive')
    if not re.fullmatch('[0-9a-f]{40}', args.revision) or args.output.exists():
        parser.error('Require immutable revision and a new output path')
    if args.model != 'ByteDance/Ouro-1.4B-Thinking' and args.revision == PINNED_OURO_THINKING_REV:
        parser.error('Supply the selected model revision')
    if not torch.cuda.is_available():
        parser.error('Real checkpoint parity targets a CUDA runtime')
    adapter = OuroRecurrentAdapter.from_pretrained(repo_id=args.model, revision=args.revision, device='cuda', dtype=getattr(torch, args.dtype), cache_mode='branched_kv', attn_implementation=args.attention)
    rows = []
    for length in args.prefix_lengths:
        rows.extend(check(adapter, torch.arange(1, length + 1, device='cuda').unsqueeze(0), tokens=args.tokens))
    tolerance = 1e-4 if args.dtype == 'float32' else .25
    passed = all(r['finite'] and r['top1_match'] and r['cache_lengths_match'] and r['max_abs_difference'] <= tolerance for r in rows)
    report = {'model': args.model, 'revision': args.revision, 'device': torch.cuda.get_device_name(0), 'dtype': args.dtype, 'attention': args.attention, 'torch': torch.__version__, 'absolute_logit_tolerance': tolerance, 'rows': rows, 'passed': passed, 'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'prefix_lengths': args.prefix_lengths, 'steps_per_prefix': args.tokens, 'scope': 'synthetic-prefix S0 checks; not accuracy or intervention parity', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
