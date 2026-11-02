"""Fixed-prefix reasoning sensitivity and conditional short CoT windows, not rescue."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from complete_ouro_baseline import native_generate
from record_passive_reasoning import PASSIVE_SOURCES
from verify_completed_answer import MODEL, ROOT, sha256, synchronize
from compare_completed_answers import write_new_json
from tebdil.core.branching import fork_branches
from tebdil.core.controls import (build_output_matched_random_state,
    execute_direct_state_mixing_step, execute_update_mixing_step)
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.monitoring import SNAPSHOT_TOKEN_INDICES
from tebdil.evaluation.reasoning import split_prompt_started_reasoning
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.reasoning import evaluate_reasoning_task

SOURCES = list(dict.fromkeys(PASSIVE_SOURCES + [
    'scripts/probe_reasoning_cot.py', 'docs/reasoning_probe_protocol.md',
    'tests/test_reasoning_probe.py', 'scripts/compare_completed_answers.py',
    'src/tebdil/evaluation/comparison.py', 'src/tebdil/core/controls.py',
    'src/tebdil/core/excursions.py', 'src/tebdil/core/branching.py',
    'src/tebdil/core/perturbations.py', 'src/tebdil/core/scheduler.py',
    'src/tebdil/core/policies.py',
]))
METHODS = ('S0', 'M0', 'M1', 'E1', 'R_output')


@torch.no_grad()
def probe_prefix(adapter, ids, config, seed):
    """Only IDs, fixed config and seed enter model computation; no task or gold."""
    if adapter.total_steps != 4 or adapter.readout_mode != 'last_step':
        raise ValueError('Four-depth last-step contract required')
    physical, stages = [0], []
    before = adapter.recurrent_block_calls
    handle = adapter.model.model.norm.register_forward_hook(lambda *args: physical.__setitem__(0, physical[0]+1))
    def measure(label, fn):
        start = physical[0]
        result = fn()
        stages.append(dict(stage=label, recurrent_calls=physical[0]-start))
        return result
    try:
        root = measure('shared_depth_1', lambda: adapter.transition(adapter.initialize(ids)))
        branches, _ = fork_branches(root, count=2, relative_norm=.25, seed=seed)
        a, b = [measure('branch_'+s.branch_id+'_depth_2', lambda s=s: adapter.transition(s)) for s in branches]
        originals = [hash_tensor(s.hidden,64) for s in (a,b)]
        normal = measure('ordinary_depth_3', lambda: adapter.transition(a))
        e1, _, telemetry = measure('E1_depth_3', lambda: execute_excursion_step(adapter,a,b,config))
        m0, _, _ = measure('M0_depth_3', lambda: execute_direct_state_mixing_step(adapter,a,b,config))
        m1, _, _ = measure('M1_depth_3', lambda: execute_update_mixing_step(adapter,a,b,config))
        random = build_output_matched_random_state(adapter,normal,e1,config,seed+17)
        if originals != [hash_tensor(s.hidden,64) for s in (a,b)] or telemetry.skipped_unstable or not telemetry.post_finite:
            raise ValueError('Input mutation or unstable intervention')
        target = float((e1.hidden.float()-normal.hidden.float()).norm())
        realized = float((random.hidden.float()-normal.hidden.float()).norm())
        measurements, baseline = [], None
        for method, state in zip(METHODS,(normal,m0,m1,e1,random),strict=True):
            final = measure(method+'_depth_4', lambda state=state: adapter.transition(state))
            logits = adapter.readout_logits(final)[0,-1].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite readout')
            if baseline is None:
                baseline = logits.clone()
            logp, base_logp = logits.log_softmax(-1), baseline.log_softmax(-1)
            top = logits.topk(2)
            measurements.append(dict(method=method,next_token_id=int(top.indices[0]),
                logit_margin=float(top.values[0]-top.values[1]),
                logit_l2_vs_S0=float((logits-baseline).norm()),
                kl_method_to_S0=max(0.,float((logp.exp()*(logp-base_logp)).sum())),
                post_intervention_hash=hash_tensor(state.hidden,64),
                final_last_position_hash=hash_tensor(final.hidden[0,-1],64),
                shift_vs_ordinary=float((state.hidden.float()-normal.hidden.float()).norm())))
        if physical[0] != adapter.recurrent_block_calls-before or physical[0] != sum(e['recurrent_calls'] for e in stages):
            raise ValueError('Physical/adapter/stage call mismatch')
        return dict(prefix_length=ids.shape[1],prefix_sha256=hash_tensor(ids,64),
            positions_sha256=hash_tensor(torch.arange(ids.shape[1])[None],64),
            branch_seed=seed,intervention=asdict(telemetry),measurements=measurements,
            random_target_norm=target,random_realized_norm=realized,
            random_relative_norm_error=abs(realized-target)/max(target,1e-12),
            stages=stages,recurrent_calls=physical[0],
            recurrent_token_positions=physical[0]*ids.shape[1],inputs_unchanged=True)
    finally:
        handle.remove()


@torch.no_grad()
def continue_window(adapter, tokenizer, row, k, first_token, budget, on_entry):
    """Native continuation after one actual intervened token; no edited CoT/hints."""
    prefix = row['prompt_token_ids']+row['generated_token_ids'][:k]+[first_token]
    proposals = []
    def observe(module,args,kwargs,output):
        ids, logits = kwargs['input_ids'], output.logits[:,-1]
        if ids.tolist()[0] != prefix+proposals or not torch.isfinite(logits).all():
            raise ValueError('Native continuation prefix/readout mismatch')
        token = int(logits.argmax())
        on_entry(dict(token_index=k+1+len(proposals),token_id=token,
                      prefix_length=ids.shape[1],prefix_sha256=hash_tensor(ids,64)))
        proposals.append(token)
    tokens, ledger, calls = [], [], 0
    if first_token != tokenizer.eos_token_id and budget > 1:
        handle = adapter.model.register_forward_hook(observe,with_kwargs=True)
        try:
            tokens,ledger,calls = native_generate(adapter,tokenizer,torch.tensor([prefix],device=adapter.device),budget-1)
        finally:
            handle.remove()
        if tokens != proposals:
            raise ValueError('Generated/observed continuation mismatch')
    generated = [first_token]+tokens
    completed = generated[-1] == tokenizer.eos_token_id
    text = tokenizer.decode(row['generated_token_ids'][:k]+generated,skip_special_tokens=False)
    return dict(generated_token_ids=generated,token_ledger=ledger,new_recurrent_calls=calls,
        recurrent_token_positions=sum(e['prefix_length']*e['recurrent_block_calls'] for e in ledger),
        completed=completed,stop_reason='eos_token' if completed else 'window_exhausted',
        decoded_generated_text=text,**split_prompt_started_reasoning(text,tokenizer.eos_token,completed=completed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new directory; preserve completed and partial evidence')
    folder=ROOT/'reports/passive_reasoning_v1'
    reference=json.loads((folder/'report.json').read_text())
    if (not reference['all_checks_passed'] or not reference['enable_thinking'] or
        reference['dtype'] != 'torch.bfloat16' or reference['source_sha256'] != {p:sha256(ROOT/p) for p in reference['source_sha256']}):
        raise ValueError('Frozen baseline/source contract mismatch')
    tasks=json.loads((ROOT/'configs/reasoning_v1_tasks.json').read_text())['partitions']['development'][:2]
    source_hashes={p:sha256(ROOT/p) for p in SOURCES}
    input_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in folder.iterdir()}
    snapshot=Path(snapshot_download(MODEL,revision=PINNED_OURO_THINKING_REV,local_files_only=True))
    checkpoint={p:sha256(snapshot/p) for p in reference['checkpoint_sha256']}
    if checkpoint != reference['checkpoint_sha256'] or torch.__version__ != reference['torch_version']:
        raise ValueError('Checkpoint/software mismatch')
    config=ExcursionConfig(alpha=.65,beta=.65,max_relative_norm=.1,visit_step=2,token_scope='full')
    report=dict(protocol='reasoning_cot_probe_v1',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=source_hashes,input_sha256=input_hashes,checkpoint_sha256=checkpoint,
        model_id=MODEL,model_revision=PINNED_OURO_THINKING_REV,dtype=reference['dtype'],
        device=reference['device'],torch_version=torch.__version__,enable_thinking=True,
        cache_mode='disabled',readout_mode='last_step',recurrent_depth=4,
        excursion_config=asdict(config),probe_indices=list(SNAPSHOT_TOKEN_INDICES),
        scope='Two reused development cases; fixed-prefix sensitivity and short process windows, not stall recovery',
        reused_baseline_calls=reference['actual_new_recurrent_calls'],probes=[],windows=[])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    journal=args.output_dir/'journal.jsonl'
    append_trace_record(dict(event='start',**report),journal)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_OURO_THINKING_REV,local_files_only=True)
    adapter=OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16,device=reference['device'],local_files_only=True)
    started=time.monotonic()
    try:
        cases=[]
        for task,summary in zip(tasks,reference['records'],strict=True):
            case_path=folder/summary['raw_file']
            if sha256(case_path) != summary['raw_sha256']:
                raise ValueError('Raw baseline checksum mismatch')
            case=json.loads(case_path.read_text())
            row=case['record']
            if task['problem_id'] != row['problem_id'] or task['prompt'] != row['prompt'] or not row['prompt_text_with_special_tokens'].endswith('<|im_start|>assistant\n<think>\n'):
                raise ValueError('Unexpected task/prompt identity')
            cases.append((task,row))
            for k in SNAPSHOT_TOKEN_INDICES:
                ids=torch.tensor([row['prompt_token_ids']+row['generated_token_ids'][:k]],device=adapter.device)
                result=probe_prefix(adapter,ids,config,20261102+131*k)
                expected=case['observations'][k]
                baseline=result['measurements'][0]
                if (result['prefix_sha256'] != expected['prefix_token_ids_sha256'] or
                    baseline['next_token_id'] != row['generated_token_ids'][k] or
                    baseline['final_last_position_hash'] != expected['depth_states'][-1]['normalized_sha256']):
                    raise ValueError('S0 native recorded token/state parity failed')
                result.update(problem_id=row['problem_id'],token_index=k,
                    reasoning_prefix=tokenizer.decode(row['generated_token_ids'][:k],skip_special_tokens=False))
                for m in result['measurements']:
                    m['next_token_text']=tokenizer.decode([m['next_token_id']],skip_special_tokens=False)
                    m['text_with_proposal']=tokenizer.decode(row['generated_token_ids'][:k]+[m['next_token_id']],skip_special_tokens=False)
                    m['changed_vs_S0']=m['next_token_id'] != baseline['next_token_id']
                report['probes'].append(result)
                append_trace_record(dict(event='probe',**result),journal)
                print(row['problem_id'],k,'E1 changed:',result['measurements'][3]['changed_vs_S0'],flush=True)
        for task,row in cases:
            divergent=next((p for p in report['probes'] if p['problem_id']==row['problem_id'] and p['measurements'][3]['changed_vs_S0']),None)
            if divergent is None:
                continue
            k=divergent['token_index']
            budget=min(128,2048-k)
            for proposal in divergent['measurements']:
                method=proposal['method']
                window=dict(problem_id=row['problem_id'],method=method,token_index=k,budget=budget)
                append_trace_record(dict(event='window_start',**window),journal)
                if method=='S0':
                    tokens=row['generated_token_ids'][k:k+budget]
                    text=tokenizer.decode(row['generated_token_ids'][:k]+tokens,skip_special_tokens=False)
                    window.update(execution='previously_recorded_S0',generated_token_ids=tokens,
                        decoded_generated_text=text,new_recurrent_calls=0,reused_window_calls=4*len(tokens),
                        recurrent_token_positions=0,completed=False,stop_reason='window_exhausted',
                        **split_prompt_started_reasoning(text,tokenizer.eos_token,completed=False))
                elif not proposal['changed_vs_S0']:
                    window.update(execution='not_executed_same_first_token',generated_token_ids=[proposal['next_token_id']],
                        new_recurrent_calls=0,recurrent_token_positions=0,completed=False,stop_reason='not_executed')
                else:
                    window.update(execution='new_native_continuation',**continue_window(adapter,tokenizer,row,k,
                        proposal['next_token_id'],budget,lambda e:append_trace_record(dict(event='window_token',method=method,problem_id=row['problem_id'],**e),journal)))
                window['final_evaluation']=evaluate_reasoning_task(task,window) if window['completed'] else None
                report['windows'].append(window)
                append_trace_record(dict(event='window_finish',**window),journal)
                print('window',row['problem_id'],method,window['stop_reason'],flush=True)
        if source_hashes != {p:sha256(ROOT/p) for p in SOURCES} or input_hashes != {p.relative_to(ROOT).as_posix():sha256(p) for p in folder.iterdir()} or checkpoint != {p:sha256(snapshot/p) for p in checkpoint}:
            raise ValueError('Source/input/checkpoint changed')
        synchronize(adapter.device)
        report.update(actual_new_recurrent_calls=sum(p['recurrent_calls'] for p in report['probes'])+sum(w['new_recurrent_calls'] for w in report['windows']),
            recurrent_token_positions=sum(p['recurrent_token_positions'] for p in report['probes'])+sum(w['recurrent_token_positions'] for w in report['windows']),
            wall_seconds=round(time.monotonic()-started,3),all_checks_passed=True)
        write_new_json(args.output_dir/'report.json',report)
        append_trace_record(dict(event='finish',report_sha256=sha256(args.output_dir/'report.json')),journal)
    except Exception as exc:
        append_trace_record(dict(event='failure',type=type(exc).__name__,message=str(exc)),journal)
        raise


if __name__=='__main__':
    main()
