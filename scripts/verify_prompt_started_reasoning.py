"""One new True-mode native smoke on an already-observed development question."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from verify_completed_answer import MODEL, ROOT, SOURCES, native_generate, sha256, synchronize
from tebdil.core.state import hash_tensor
from tebdil.evaluation.reasoning import encode_reasoning_prompt, evaluate_reasoning_integer, validate_reasoning_completion
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

REASONING_SOURCES = list(dict.fromkeys(SOURCES + [
    'scripts/verify_prompt_started_reasoning.py', 'src/tebdil/evaluation/reasoning.py',
    'src/tebdil/tasks/completion_v2.py', 'tests/test_prompt_started_reasoning.py',
    'docs/prompt_started_reasoning_protocol.md', 'reports/ouro_completed_answer_smoke.json',
]))


@torch.no_grad()
def run_reasoning_completion(adapter, tokenizer, prompt, budget, on_forward):
    """Observe native True-mode generation without a second decoding engine."""
    if adapter.total_steps != 4 or adapter.readout_mode != 'last_step':
        raise ValueError('Four-depth last-step contract required')
    ids = encode_reasoning_prompt(tokenizer, prompt).to(adapter.device)
    prompt_ids, proposals, observations = ids.tolist()[0], [], []
    physical_calls = 0

    def boundary(module, inputs, output):
        nonlocal physical_calls
        physical_calls += 1

    def observe(module, args, kwargs, output):
        prefix, logits = kwargs['input_ids'], output.logits[:, -1]
        if prefix.tolist()[0] != prompt_ids + proposals or not torch.isfinite(logits).all():
            raise ValueError('Native prefix/readout disagrees with independent observer')
        entry = dict(token_index=len(proposals), greedy_readout_token_id=int(logits.argmax().item()),
                     prefix_length=prefix.shape[1], prefix_token_ids_sha256=hash_tensor(prefix, 64),
                     calls_before=4*len(proposals), recurrent_block_calls=physical_calls-4*len(proposals))
        if entry['recurrent_block_calls'] != 4:
            raise ValueError('Independent recurrent boundary count mismatch')
        proposals.append(entry['greedy_readout_token_id'])
        observations.append(entry)
        on_forward(entry)

    norm_hook = adapter.model.model.norm.register_forward_hook(boundary)
    forward_hook = adapter.model.register_forward_hook(observe, with_kwargs=True)
    try:
        tokens, ledger, calls = native_generate(adapter, tokenizer, ids, budget)
    finally:
        forward_hook.remove()
        norm_hook.remove()
    if tokens != proposals or calls != physical_calls:
        raise ValueError('Actual generated tokens/calls disagree with observer')
    record = validate_reasoning_completion(tokenizer, prompt_ids, tokens, ledger, calls, budget=budget, depth=4)
    for observed, native in zip(observations, record['token_ledger'], strict=True):
        if any(observed[k] != native[k] for k in ('prefix_length', 'prefix_token_ids_sha256', 'calls_before', 'recurrent_block_calls')):
            raise ValueError('Independent/native ledgers disagree')
    return {**record, 'independent_forward_checks_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    journal = args.output.with_suffix('.jsonl')
    if args.output == journal:
        parser.error('Choose a report path distinct from the .jsonl journal')
    if args.output.exists() or journal.exists():
        parser.error('Output or journal exists; preserve prior evidence')
    reference = json.loads((ROOT / 'reports/ouro_completed_answer_smoke.json').read_text())
    task = json.loads((ROOT / 'configs/completion_v1_tasks.json').read_text())['partitions']['development'][0]
    if (not reference['all_checks_passed'] or task['problem_id'] != 'completion_dev_000'
            or task['prompt'] != reference['records'][0]['prompt'] or task['gold_answer'] != reference['records'][0]['gold_answer']):
        raise ValueError('Expected unchanged already-observed development smoke')
    sources = {p: sha256(ROOT / p) for p in REASONING_SOURCES}
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint = {p: sha256(snapshot / p) for p in reference['checkpoint_sha256']}
    if checkpoint != reference['checkpoint_sha256'] or torch.__version__ != reference['torch_version']:
        raise ValueError('Checkpoint/software identity differs from frozen contract')
    identity = dict(schema_version=1, protocol='prompt_started_reasoning_v1', enable_thinking=True,
                    recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256=sources, checkpoint_sha256=checkpoint, model_id=MODEL,
                    model_revision=PINNED_OURO_THINKING_REV, device=reference['device'], dtype=reference['dtype'],
                    torch_version=torch.__version__, recurrent_depth=4, readout_mode='last_step', cache_mode='disabled',
                    max_output_tokens=2048, eos_token=reference['eos_token'], eos_token_id=reference['eos_token_id'],
                    problem_ids=[task['problem_id']], scope='True-mode recording smoke, not reasoning utility or a False-mode comparison')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': 'start', **identity}) + '\n')
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
        if tokenizer.eos_token != reference['eos_token'] or tokenizer.eos_token_id != reference['eos_token_id']:
            raise ValueError('Tokenizer EOS differs from reference')
        adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16, device=reference['device'], local_files_only=True)
        synchronize(adapter.device)
        started = time.monotonic()
        record = run_reasoning_completion(adapter, tokenizer, task['prompt'], 2048,
                                         lambda entry: append_trace_record({'event': 'forward', **entry}, journal))
        synchronize(adapter.device)
        record.update(problem_id=task['problem_id'], split='development', method='S0', prompt=task['prompt'],
                      gold_answer=task['gold_answer'], wall_seconds=round(time.monotonic()-started, 3))
        record.update(evaluate_reasoning_integer(record, task['gold_answer']))
        if sources != {p: sha256(ROOT / p) for p in REASONING_SOURCES} or checkpoint != {p: sha256(snapshot / p) for p in checkpoint}:
            raise ValueError('Source/checkpoint changed during smoke')
        report = {**identity, 'records': [record], 'actual_new_recurrent_calls': record['recurrent_block_calls'],
                  'all_checks_passed': True,
                  'limitations': ['One already-observed development question, no new task difficulty evidence',
                                  'No stall monitor, hidden-state progress claim, intervention, controller or reserved-test evaluation',
                                  'Explicit thinking prefix does not establish faithful or improved reasoning',
                                  'Instrumented pipeline runtime, not optimized latency; positions are not FLOPs']}
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        append_trace_record({'event': 'finish', 'actual_new_recurrent_calls': record['recurrent_block_calls'], 'record': record}, journal)
        print(f"{record['status']}, {len(record['generated_token_ids'])} tokens, {record['reasoning_status']}, {record['wall_seconds']}s", flush=True)
    except Exception as exc:
        append_trace_record({'event': 'failure', 'type': type(exc).__name__, 'message': str(exc)}, journal)
        raise


if __name__ == '__main__':
    main()
