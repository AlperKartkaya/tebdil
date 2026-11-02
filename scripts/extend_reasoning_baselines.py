"""Frozen 512-token follow-up of both unfinished reasoning baselines, no TEBDIL."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from complete_ouro_baseline import native_generate
from compare_completed_answers import write_new_json
from record_passive_reasoning import PASSIVE_SOURCES
from verify_completed_answer import MODEL, ROOT, sha256, synchronize
from tebdil.core.state import hash_tensor
from tebdil.evaluation.reasoning import split_prompt_started_reasoning
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.reasoning import evaluate_reasoning_task

SOURCES = list(dict.fromkeys(PASSIVE_SOURCES + [
    'scripts/extend_reasoning_baselines.py', 'docs/reasoning_budget_protocol.md',
    'tests/test_reasoning_budget.py',
]))
EXTRA_TOKENS = 512


@torch.no_grad()
def resume_baseline(adapter, tokenizer, original, budget, on_entry):
    """Resume actual unedited tokens; no task/answer enters generation."""
    if adapter.total_steps != 4 or adapter.readout_mode != 'last_step':
        raise ValueError('Four-depth last-step contract required')
    old = original['generated_token_ids']
    if (not old or tokenizer.eos_token_id in old or original['completed']
            or original['reasoning_status'] != 'open' or budget <= 0):
        raise ValueError('An unfinished open-reasoning baseline and positive budget required')
    prefix = original['prompt_token_ids'] + old
    proposals, observations = [], []
    physical = [0]
    def boundary(module, args, output):
        physical[0] += 1
    def observe(module, args, kwargs, output):
        ids, logits = kwargs['input_ids'], output.logits[:, -1]
        if ids.tolist()[0] != prefix + proposals or not torch.isfinite(logits).all():
            raise ValueError('Continuation prefix/readout mismatch')
        entry = dict(token_index=len(old)+len(proposals), token_id=int(logits.argmax()),
                     prefix_length=ids.shape[1], prefix_token_ids_sha256=hash_tensor(ids,64),
                     calls_before=4*len(proposals), recurrent_block_calls=physical[0]-4*len(proposals))
        if entry['recurrent_block_calls'] != 4:
            raise ValueError('Independent recurrent boundary count mismatch')
        proposals.append(entry['token_id'])
        observations.append(entry)
        on_entry(entry)
    norm = adapter.model.model.norm.register_forward_hook(boundary)
    handle = adapter.model.register_forward_hook(observe,with_kwargs=True)
    try:
        tokens, ledger, calls = native_generate(adapter,tokenizer,
            torch.tensor([prefix],device=adapter.device),budget)
    finally:
        handle.remove()
        norm.remove()
    if tokens != proposals or len(ledger) != len(tokens):
        raise ValueError('Continuation observed/native token mismatch')
    compact = []
    for k, (observed, entry) in enumerate(zip(observations,ledger,strict=True)):
        expected = prefix + tokens[:k]
        if (entry['processed_input_ids'] != expected or entry['prefix_length'] != len(expected)
                or entry['processed_sequence_length'] != len(expected)
                or entry['prefix_token_ids_sha256'] != observed['prefix_token_ids_sha256']
                or entry['calls_before'] != 4*k or entry['recurrent_block_calls'] != 4):
            raise ValueError('Actual native full-prefix ledger mismatch')
        compact.append({**observed, 'calls_before':entry['calls_before'], 'recurrent_block_calls':4})
    if not tokens or calls != physical[0] or calls != 4*len(tokens) or (tokenizer.eos_token_id in tokens[:-1]):
        raise ValueError('Continuation stopping/cost mismatch')
    combined = old + tokens
    completed = tokens[-1] == tokenizer.eos_token_id
    if not completed and len(tokens) != budget:
        raise ValueError('Unexplained short native continuation')
    text = tokenizer.decode(combined,skip_special_tokens=False)
    return dict(prompt_token_ids=original['prompt_token_ids'], generated_token_ids=combined,
        decoded_text_with_special_tokens=text, continuation_token_ledger=compact,
        prior_generated_tokens=len(old), new_generated_tokens=len(tokens), total_generated_tokens=len(combined),
        new_generation_calls=calls, new_recurrent_token_positions=sum(e['prefix_length']*4 for e in compact),
        reused_baseline_calls=original['recurrent_block_calls'], completed=completed,
        stop_reason='eos_token' if completed else 'max_output_tokens',
        **split_prompt_started_reasoning(text,tokenizer.eos_token,completed=completed))


@torch.no_grad()
def verify_last_proposal(adapter, tokenizer, case):
    old = case['record']
    ids = old['prompt_token_ids'] + old['generated_token_ids'][:-1]
    states = []
    handle = adapter.model.model.norm.register_forward_hook(lambda m,a,o: states.append(hash_tensor(o[0,-1],64)))
    try:
        tokens, ledger, calls = native_generate(adapter,tokenizer,torch.tensor([ids],device=adapter.device),1)
    finally:
        handle.remove()
    expected = case['observations'][-1]['depth_states'][-1]['normalized_sha256']
    if tokens != old['generated_token_ids'][-1:] or len(states) != 4 or states[-1] != expected or calls != 4:
        raise ValueError('Original terminal token/state parity failed')
    return dict(token_index=len(old['generated_token_ids'])-1, token_id=tokens[0],
        final_last_position_hash=states[-1], prefix_token_ids_sha256=ledger[0]['prefix_token_ids_sha256'],
        recurrent_calls=calls, recurrent_token_positions=calls*len(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new directory; preserve completed and partial evidence')
    folder = ROOT/'reports/passive_reasoning_v1'
    reference = json.loads((folder/'report.json').read_text())
    sources = {p:sha256(ROOT/p) for p in SOURCES}
    inputs = {p.relative_to(ROOT).as_posix():sha256(p) for p in folder.iterdir()}
    if (not reference['all_checks_passed'] or not reference['enable_thinking']
            or reference['dtype'] != 'torch.bfloat16'
            or reference['source_sha256'] != {p:sha256(ROOT/p) for p in reference['source_sha256']}):
        raise ValueError('Original baseline/source contract mismatch')
    snapshot = Path(snapshot_download(MODEL,revision=PINNED_OURO_THINKING_REV,local_files_only=True))
    checkpoint = {p:sha256(snapshot/p) for p in reference['checkpoint_sha256']}
    if checkpoint != reference['checkpoint_sha256'] or torch.__version__ != reference['torch_version']:
        raise ValueError('Checkpoint/software mismatch')
    tasks = json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())['partitions']['development'][:2]
    identity = dict(protocol='reasoning_budget_v1',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sources,input_sha256=inputs,checkpoint_sha256=checkpoint,
        model_id=MODEL,model_revision=PINNED_OURO_THINKING_REV,device=reference['device'],dtype=reference['dtype'],
        torch_version=torch.__version__,enable_thinking=True,recurrent_depth=4,readout_mode='last_step',
        cache_mode='disabled',additional_token_cap=EXTRA_TOKENS,total_output_cap=2048+EXTRA_TOKENS,
        selected_problem_ids=[t['problem_id'] for t in tasks],scope='Ordinary baseline budget diagnostic; no TEBDIL')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    journal = args.output_dir/'journal.jsonl'
    append_trace_record(dict(event='start',**identity),journal)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=PINNED_OURO_THINKING_REV,local_files_only=True)
        if tokenizer.eos_token_id != reference['eos_token_id'] or tokenizer.eos_token != reference['eos_token']:
            raise ValueError('Tokenizer EOS mismatch')
        adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16,device=reference['device'],local_files_only=True)
        records = []
        for task, summary in zip(tasks,reference['records'],strict=True):
            path = folder/summary['raw_file']
            if sha256(path) != summary['raw_sha256']:
                raise ValueError('Original recording checksum mismatch')
            case = json.loads(path.read_text())
            old = case['record']
            if task['problem_id'] != old['problem_id'] or task['prompt'] != old['prompt'] or len(old['generated_token_ids']) != 2048:
                raise ValueError('Unexpected selected case/prompt/budget')
            problem_id = task['problem_id']
            append_trace_record(dict(event='case_start',problem_id=problem_id),journal)
            synchronize(adapter.device)
            started = time.monotonic()
            replay = verify_last_proposal(adapter,tokenizer,case)
            append_trace_record(dict(event='parity',problem_id=problem_id,**replay),journal)
            print(problem_id+'; terminal parity passed; resuming for at most 512 tokens',flush=True)
            record = resume_baseline(adapter,tokenizer,old,EXTRA_TOKENS,
                lambda e:append_trace_record(dict(event='token',problem_id=problem_id,**e),journal))
            synchronize(adapter.device)
            record.update(problem_id=problem_id,method='S0',family=task['family'],split='development',
                original_raw_file=path.relative_to(ROOT).as_posix(),original_raw_sha256=summary['raw_sha256'],
                terminal_parity=replay,wall_seconds=round(time.monotonic()-started,3))
            record.update(evaluate_reasoning_task(task,record))
            records.append(record)
            append_trace_record(dict(event='case_finish',record=record),journal)
            print(problem_id+'; '+record['status']+'; total tokens '+str(record['total_generated_tokens']),flush=True)
        if sources != {p:sha256(ROOT/p) for p in SOURCES} or inputs != {p.relative_to(ROOT).as_posix():sha256(p) for p in folder.iterdir()} or checkpoint != {p:sha256(snapshot/p) for p in checkpoint}:
            raise ValueError('Source/input/checkpoint changed during run')
        report = dict(**identity,records=records,all_checks_passed=True,
            actual_new_recurrent_calls=sum(r['new_generation_calls']+r['terminal_parity']['recurrent_calls'] for r in records),
            recurrent_token_positions=sum(r['new_recurrent_token_positions']+r['terminal_parity']['recurrent_token_positions'] for r in records),
            reused_baseline_calls=sum(r['reused_baseline_calls'] for r in records))
        write_new_json(args.output_dir/'report.json',report)
        append_trace_record(dict(event='finish',report_sha256=sha256(args.output_dir/'report.json')),journal)
    except Exception as exc:
        append_trace_record(dict(event='failure',type=type(exc).__name__,message=str(exc)),journal)
        raise


if __name__ == '__main__':
    main()
