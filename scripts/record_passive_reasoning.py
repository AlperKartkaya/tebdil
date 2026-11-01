"""Frozen passive recording pilot on the first two development reasoning tasks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from compare_completed_answers import write_new_json
from verify_prompt_started_reasoning import REASONING_SOURCES, run_reasoning_completion
from verify_completed_answer import MODEL, ROOT, sha256, synchronize
from tebdil.evaluation.monitoring import passive_observer
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.reasoning import evaluate_reasoning_task, validate_manifest

PASSIVE_SOURCES = list(dict.fromkeys(REASONING_SOURCES + [
    'src/tebdil/evaluation/monitoring.py', 'scripts/record_passive_reasoning.py',
    'scripts/compare_completed_answers.py', 'src/tebdil/tasks/reasoning.py',
    'configs/reasoning_v1_tasks.json', 'docs/reasoning_tasks_protocol.md',
    'docs/passive_reasoning_protocol.md', 'tests/test_passive_reasoning.py',
    'tests/test_reasoning_tasks.py', 'reports/ouro_prompt_started_reasoning.json',
]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new directory; preserve all partial evidence')
    manifest = json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())
    validate_manifest(manifest)
    tasks = manifest['partitions']['development'][:2]
    if [t['structural_level'] for t in tasks] != [0,0] or [t['family'] for t in tasks] != ['cnf','key_planning']:
        raise ValueError('Unexpected frozen first development pair')
    reference = json.loads((ROOT/'reports/ouro_prompt_started_reasoning.json').read_text())
    sources = {p: sha256(ROOT/p) for p in PASSIVE_SOURCES}
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint = {p: sha256(snapshot/p) for p in reference['checkpoint_sha256']}
    if checkpoint != reference['checkpoint_sha256'] or torch.__version__ != reference['torch_version']:
        raise ValueError('Checkpoint/software identity mismatch')
    identity = dict(protocol='passive_reasoning_v1', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                    base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    source_sha256=sources, checkpoint_sha256=checkpoint, model_id=MODEL,
                    model_revision=PINNED_OURO_THINKING_REV, torch_version=torch.__version__,
                    device=reference['device'], dtype=reference['dtype'], recurrent_depth=4,
                    readout_mode='last_step', cache_mode='disabled', enable_thinking=True,
                    max_output_tokens=2048, eos_token=reference['eos_token'], eos_token_id=reference['eos_token_id'],
                    selected_problem_ids=[t['problem_id'] for t in tasks],
                    scope='First development pair, passive recording pilot; no intervention or semantic stall classifier')
    args.output_dir.mkdir(parents=True)
    journal = args.output_dir/'run.jsonl'
    append_trace_record({'event':'start',**identity},journal)
    summaries = []
    try:
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_OURO_THINKING_REV,local_files_only=True)
        if tokenizer.eos_token_id != identity['eos_token_id'] or tokenizer.eos_token != identity['eos_token']:
            raise ValueError('Tokenizer EOS mismatch')
        adapter=OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16,device=identity['device'],local_files_only=True)
        for task in tasks:
            problem_id=task['problem_id']
            observations=[]
            trace=args.output_dir/(problem_id+'.jsonl')
            append_trace_record({'event':'case_start','problem_id':problem_id,**identity},trace)
            append_trace_record({'event':'case_start','problem_id':problem_id},journal)
            print('Passive observation '+problem_id+'; budget 2048',flush=True)
            def save(entry):
                observations.append(entry)
                append_trace_record({'event':'observation',**entry},trace)
            synchronize(adapter.device)
            started=time.monotonic()
            with passive_observer(adapter,tokenizer,save) as observer:
                record=run_reasoning_completion(adapter,tokenizer,task['prompt'],2048,observer)
            synchronize(adapter.device)
            elapsed=round(time.monotonic()-started,3)
            record.update(problem_id=problem_id,split='development',method='S0',prompt=task['prompt'],
                          family=task['family'],structural_level=task['structural_level'],wall_seconds=elapsed)
            record.update(evaluate_reasoning_task(task,record))
            if (len(observations) != len(record['generated_token_ids']) or
                    [e['greedy_readout_token_id'] for e in observations] != record['generated_token_ids']):
                raise ValueError('Observed/returned token ledger mismatch')
            if sources != {p:sha256(ROOT/p) for p in PASSIVE_SOURCES} or checkpoint != {p:sha256(snapshot/p) for p in checkpoint}:
                raise ValueError('Source/checkpoint changed during recording')
            append_trace_record({'event':'case_finish','record':record},trace)
            case=args.output_dir/(problem_id+'.json')
            write_new_json(case,{**identity,'record':record,'observations':observations,'all_checks_passed':True})
            flags=[e['token_index'] for e in observations if e['candidate_repeat_flag']]
            summary=dict(problem_id=problem_id,family=task['family'],status=record['status'],
                         completed=record['completed'],reasoning_status=record['reasoning_status'],
                         generated_tokens=len(record['generated_token_ids']),recurrent_block_calls=record['recurrent_block_calls'],
                         recurrent_token_positions=record['recurrent_token_positions'],wall_seconds=elapsed,
                         candidate_repeat_flags=len(flags),first_candidate_flag_token_index=flags[0] if flags else None,
                         raw_file=case.name,raw_sha256=sha256(case),journal_file=trace.name,journal_sha256=sha256(trace))
            summaries.append(summary)
            append_trace_record({'event':'case_finish',**summary},journal)
            print(f"{record['status']}; {len(observations)} tokens; {len(flags)} candidate repeat flags; {elapsed}s",flush=True)
        report={**identity,'records':summaries,'actual_new_recurrent_calls':sum(r['recurrent_block_calls'] for r in summaries),
                'all_checks_passed':True,'limitations':['Two lowest-structural-level development cases; no difficulty or population estimate',
                'Exact repeats and within-prefix normalized updates do not establish semantic stagnation',
                'No intervention, controller, reserved-test observation or causal benefit',
                'Instrumented CPU-copy/journal runtime, not optimized latency; token positions are not FLOPs']}
        write_new_json(args.output_dir/'report.json',report)
        append_trace_record({'event':'finish','report_sha256':sha256(args.output_dir/'report.json')},journal)
    except Exception as exc:
        append_trace_record({'event':'failure','type':type(exc).__name__,'message':str(exc)},journal)
        raise


if __name__=='__main__':
    main()
