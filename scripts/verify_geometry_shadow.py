"""Bounded native/shadow parity probe. No steering and no efficacy claims."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import torch
from transformers import AutoTokenizer
from tebdil.core.scheduler import generate_tebdil_cached
from tebdil.core.state import hash_tensor
from tebdil.evaluation.reasoning import encode_reasoning_prompt
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'A train travels 120 kilometers in 2 hours at constant speed. How far does it travel in 5 hours? Explain your reasoning.'


def cache_hashes(adapter):
    cache = adapter._active_kv_cache
    return [[hash_tensor(k, 64), hash_tensor(v, 64)] for k, v in zip(cache.key_cache, cache.value_cache)]


@torch.no_grad()
def probe(adapter, tokenizer, ids, budget):
    final_logits = []
    original = adapter.readout_logits
    def record(state, last_token_only=False):
        logits = original(state, last_token_only=last_token_only)
        if state.recurrent_step == adapter.total_steps:
            final_logits.append(hash_tensor(logits, 64))
        return logits
    adapter.readout_logits = record
    runs, caches, logits_by_run = [], [], []
    try:
        for shadow in (False, True):
            final_logits.clear()
            result = generate_tebdil_cached(adapter, tokenizer, ids, method='S0',
                                            max_new_tokens=budget, geometry_shadow=shadow)
            runs.append(result)
            caches.append(cache_hashes(adapter))
            logits_by_run.append(list(final_logits))
    finally:
        adapter.readout_logits = original
    baseline, shadow = runs
    checks = {
        'tokens_exact': baseline['generated_token_ids'] == shadow['generated_token_ids'],
        'final_logits_every_token_exact': logits_by_run[0] == logits_by_run[1] and len(logits_by_run[0]) == len(baseline['generated_token_ids']),
        'all_committed_kv_slots_exact': caches[0] == caches[1],
        'recurrent_calls_equal': baseline['recurrent_block_calls'] == shadow['recurrent_block_calls'],
        'no_excursions': baseline['excursion_count'] == shadow['excursion_count'] == 0,
        'one_extra_readout_per_token': shadow['geometry_shadow']['extra_readout_projections'] == len(shadow['generated_token_ids']),
        'strictly_causal_bank': all(all(p < row['token_position'] for p in row['bank_positions']) for row in shadow['geometry_shadow']['records']),
    }
    # Retraction is a separate tensor-only check; its candidate is never passed to transition.
    calls = adapter.recurrent_block_calls
    state = adapter.initialize(ids)
    for _ in range(3):
        state = adapter.transition(state)
    metric = adapter.geometry_coordinates(state.hidden)
    q = torch.nn.functional.normalize(metric[0, -1], dim=-1)
    direction = torch.zeros_like(q)
    direction[q.abs().argmin()] = 1
    direction -= direction.dot(q) * q
    direction /= direction.norm()
    before = hash_tensor(state.hidden, 64)
    candidate = adapter.geometry_retract_hidden(state.hidden, direction, .1)
    zero = adapter.geometry_retract_hidden(state.hidden, direction, 0)
    raw = state.raw_hidden.float().clone()
    raw[0, -1] = torch.nn.functional.normalize(q + .1 * direction, dim=-1) * raw[0, -1].norm()
    expected = adapter.normalize_hidden(raw)
    max_error = float((candidate - expected).abs().max())
    checks.update({
        'zero_radius_exact': torch.equal(zero, state.hidden),
        'inactive_tokens_exact': torch.equal(candidate[:, :-1], state.hidden[:, :-1]),
        'retraction_input_unchanged': before == hash_tensor(state.hidden, 64),
        'retraction_matches_rotated_raw_norm': bool(torch.allclose(candidate, expected, atol=2e-5, rtol=2e-5)),
        'only_three_reference_transitions': adapter.recurrent_block_calls - calls == 3,
    })
    return {'checks': checks, 'all_passed': all(checks.values()), 'runs': runs,
            'final_logits_sha256': logits_by_run, 'committed_kv_sha256': caches,
            'retraction_max_abs_error': max_error, 'retraction_reference_calls': 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-model', action='store_true', help='Pinned Ouro-1.4B-Thinking on CUDA; otherwise mini CPU')
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens < 1:
        parser.error('Output must be new and token budget positive')
    torch.manual_seed(42)
    if args.real_model:
        if not torch.cuda.is_available():
            parser.error('Real-model verification requires CUDA')
        model_id = 'ByteDance/Ouro-1.4B-Thinking'
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_OURO_THINKING_REV)
        adapter = OuroRecurrentAdapter.from_pretrained(repo_id=model_id, dtype=torch.float32,
                    device='cuda', total_ut_steps=4, attn_implementation='sdpa')
        ids = encode_reasoning_prompt(tokenizer, PROMPT).to(adapter.device)
    else:
        tokenizer = None
        adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=32, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1)
        ids = torch.tensor([[1, 2, 3]])
    files = sorted(set(ROOT.glob('src/**/*.py')) | set(ROOT.glob('vendor/**/*.py'))
                   | {Path(__file__).resolve(), ROOT / 'uv.lock', ROOT / 'docs/GEOMETRIC_TEBDIL_PROPOSAL.md'})
    report = {'purpose': 'Shadow non-interference and metric checks; not answer-quality evidence',
              'utc': datetime.now(timezone.utc).isoformat(), 'model_id': adapter.model_id,
              'model_revision': adapter.model_revision, 'device': str(adapter.device), 'dtype': str(adapter.dtype),
              'attention': adapter.model.config._attn_implementation,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'huggingface-hub')},
              'gpu': torch.cuda.get_device_name() if adapter.device.type == 'cuda' else None,
              'prompt': PROMPT if args.real_model else 'synthetic token IDs', 'prompt_ids': ids.tolist(), 'budget': args.tokens,
              'timing_note': 'Diagnostic hashing and readouts included; not an overhead benchmark',
              **probe(adapter, tokenizer, ids, args.tokens)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'all_passed': report['all_passed'], 'checks': report['checks'], 'output': str(args.output)}))
    if not report['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
