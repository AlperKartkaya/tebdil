"""Fixed-prefix Ouro operator diagnostic, not generation or task accuracy.

uv run python scripts/verify_operator_controls.py --output reports/ouro_operator_controls.json
Outputs are exclusive; preserve previous observations rather than overwrite them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from tebdil.core.branching import fork_branches
from tebdil.core.controls import (
    build_output_matched_random_state,
    execute_norm_matched_random_perturbation_step,
    execute_update_mixing_step,
)
from tebdil.core.excursions import build_visit_state, execute_excursion_step
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'ByteDance/Ouro-1.4B-Thinking'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.float()).item())


@torch.no_grad()
def probe_pair(adapter, a, b, config, answer_token):
    """Measure one paired condition; all branches share a fixed input prefix."""
    start = adapter.recurrent_block_calls
    input_hashes = [hash_tensor(s.hidden, 64) for s in (a, b)]
    normal, donor = adapter.transition(a), adapter.transition(b)
    visit, alpha, clipped, _, _ = build_visit_state(a, b, config)
    visited = adapter.transition(visit)
    ha, hb, hv = [s.hidden.float() for s in (a, b, visit)]
    va, vb, vv = normal.hidden.float() - ha, donor.hidden.float() - hb, visited.hidden.float() - hv
    mixed_update = (1 - alpha) * va + alpha * vb
    residual = vv - mixed_update
    pre_m1 = ha + ((1 - config.beta) * va + config.beta * mixed_update)
    pre_e1 = ha + ((1 - config.beta) * va + config.beta * vv)
    e1, e1_donor, telemetry = execute_excursion_step(adapter, a, b, config)
    m1, _, _ = execute_update_mixing_step(adapter, a, b, config)
    r0, _, _ = execute_norm_matched_random_perturbation_step(adapter, a, b, config, seed=42)
    checks = {
        'e1_matches_return_formula_after_cast': torch.equal(e1.hidden, pre_e1.to(a.hidden.dtype)),
        'm1_matches_mixing_formula_after_cast': torch.equal(m1.hidden, pre_m1.to(a.hidden.dtype)),
        'donor_is_ordinary_transition': torch.equal(e1_donor.hidden, donor.hidden),
        'input_states_unchanged': input_hashes == [hash_tensor(s.hidden, 64) for s in (a, b)],
        'no_instability_fallback': not telemetry.skipped_unstable,
    }
    candidates = {'ordinary': normal, 'M1': m1, 'E1': e1, 'R0_input_noise': r0}
    matches = {}
    for label, base in [('output_noise', normal), ('residual_noise', m1)]:
        for seed in (42, 43, 44):
            key = f'{label}_{seed}'
            state = build_output_matched_random_state(adapter, base, e1, config, seed)
            candidates[key] = state
            target = norm(e1.hidden.float() - base.hidden.float())
            realized = norm(state.hidden.float() - base.hidden.float())
            matches[key] = {'seed': seed, 'target_output_norm': target, 'realized_output_norm': realized,
                            'relative_norm_error': abs(realized - target) / max(target, 1e-12)}
    for scale in (0., .5, 1., 2.):
        state = m1.clone()
        state.hidden = (pre_m1 + scale * config.beta * residual).to(a.hidden.dtype)
        state.hidden_history = state.hidden_history[:-1] + (state.hidden,)
        state.gate_history = state.gate_history[:-1] + (adapter.compute_gate(state.hidden),)
        candidates[f'residual_scale_{scale:g}'] = state
    construction_calls = adapter.recurrent_block_calls - start
    measurements = []
    baseline_logits = None
    for name, state in candidates.items():
        state_hash = hash_tensor(state.hidden, 64)
        before = adapter.recurrent_block_calls
        final = adapter.transition(state)
        logits = adapter.readout_logits(final)[0, -1].float()
        logp = logits.log_softmax(-1)
        if baseline_logits is None:
            baseline_logits = logits
        baseline_logp = baseline_logits.log_softmax(-1)
        measurements.append({
            'method': name, 'post_intervention_hash': state_hash,
            'final_hidden_hash': hash_tensor(final.hidden, 64),
            'shift_from_ordinary_at_intervention': norm(state.hidden.float() - normal.hidden.float()),
            'shift_from_M1_at_intervention': norm(state.hidden.float() - m1.hidden.float()),
            'next_token_id': int(logits.argmax().item()),
            'first_answer_token_logprob': float(logp[answer_token].item()),
            'logit_l2_vs_ordinary': norm(logits - baseline_logits),
            'kl_method_to_ordinary': max(0., float((logp.exp() * (logp - baseline_logp)).sum().item())),
            'continuation_recurrent_calls': adapter.recurrent_block_calls - before,
            **matches.get(name, {}),
        })
    return {
        'alpha': config.alpha, 'effective_alpha': alpha, 'beta': config.beta,
        'max_relative_norm': config.max_relative_norm, 'clipped': clipped,
        'input_state_hashes': input_hashes, 'visit_state_hash': hash_tensor(visit.hidden, 64),
        'normal_output_hash': hash_tensor(normal.hidden, 64),
        'residual_hash': hash_tensor(residual, 64), 'residual_norm': norm(residual),
        'residual_relative_to_normal_update': norm(residual) / max(norm(va), 1e-12),
        'visit_cast_offset_norm': norm(hv - (ha + alpha * (hb - ha))),
        'precast_algebra_error_norm': norm(pre_e1 - pre_m1 - config.beta * residual),
        'postcast_algebra_error_norm': norm(e1.hidden.float() - m1.hidden.float() - config.beta * residual),
        'construction_and_reference_recurrent_calls': construction_calls,
        'total_recurrent_calls': adapter.recurrent_block_calls - start,
        'checks': checks, 'measurements': measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    journal = args.output.with_suffix('.jsonl')
    if args.output.exists() or journal.exists():
        parser.error('output or journal exists; choose a new evidence path')
    sources = ['scripts/verify_operator_controls.py', 'docs/operator_controls_protocol.md',
               'src/tebdil/core/controls.py', 'src/tebdil/core/excursions.py',
               'src/tebdil/core/branching.py', 'src/tebdil/core/perturbations.py',
               'src/tebdil/core/state.py', 'src/tebdil/models/ouro_adapter.py',
               'vendor/ouro_1_4b_thinking/modeling_ouro.py',
               'vendor/ouro_1_4b_thinking/configuration_ouro.py', 'uv.lock']
    inputs = ['reports/ouro_baseline_calibration.json', 'site/static/evidence/model_provenance.json']
    report = {
        'schema_version': 1, 'purpose': 'Exploratory fixed-prefix operator diagnostic, not accuracy',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: sha256(ROOT / p) for p in sources},
        'input_sha256': {p: sha256(ROOT / p) for p in inputs},
        'model_id': MODEL, 'model_revision': PINNED_OURO_THINKING_REV,
        'torch_version': torch.__version__, 'cache_mode': 'disabled', 'readout_mode': 'last_step',
        'total_depth': 4, 'fork_depth': 1, 'intervention_input_depth': 2,
        'token_scope': 'full', 'branch_seed': 20261031, 'donor_relative_norm': .25,
        'independent_prefixes': 3, 'conditions': [], 'preparation': [],
        'limitations': [
            'Three reused pilot prefixes, not held-out examples; twelve dependent conditions',
            'No generation, answer accuracy, statistical test, learned controller or efficacy claim',
            'Gold is used only for offline first-answer-token likelihood, not action selection',
            'Shared reference construction is charged; not independent compute-matched method runs',
            'Output-noise controls require the already computed reference E1 displacement',
            'Residual scales are cast(precast_M1 + lambda * beta * C); rounding may alter endpoints',
            'FP32 matching before dtype cast does not guarantee realized BF16 output norm equality',
            'No cache, long-context or CUDA evidence unless explicitly recorded',
        ],
    }
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    provenance = json.loads((ROOT / inputs[1]).read_text())
    report['checkpoint_sha256'] = {p: sha256(snapshot / p) for p in provenance['weight_hashes']}
    if not all(report['checkpoint_sha256'][p] == v['sha256'] for p, v in provenance['weight_hashes'].items()):
        raise ValueError('Checkpoint provenance mismatch')
    pilots = [r for r in json.loads((ROOT / inputs[0]).read_text())['records'] if r['prompt_format'] == 'chat']
    if len(pilots) != 3:
        raise ValueError('Expected three recorded pilot chat prefixes')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with journal.open('x') as stream:
        stream.write(json.dumps({'event': 'start', **report}, allow_nan=False) + '\n')
        stream.flush()
        for dtype in (torch.float32, torch.bfloat16):
            adapter = OuroRecurrentAdapter.from_pretrained(dtype=dtype, local_files_only=True, total_ut_steps=4)
            calls = [0]

            def count_norm(module, args, output):
                calls[0] += 1

            handle = adapter.model.model.norm.register_forward_hook(count_norm)
            try:
                for pilot in pilots:
                    before = adapter.recurrent_block_calls
                    state = adapter.transition(adapter.initialize(torch.tensor([pilot['prompt_token_ids']])))
                    branches, _ = fork_branches(state, count=2, relative_norm=.25, seed=20261031)
                    a, b = [adapter.transition(s) for s in branches]
                    report['preparation'].append({'problem_id': pilot['problem_id'], 'dtype': str(dtype),
                                                  'recurrent_calls': adapter.recurrent_block_calls - before})
                    answer_ids = tokenizer.encode(str(pilot['gold_answer']), add_special_tokens=False)
                    for radius in (.1, .005):
                        print(f"{dtype} {pilot['problem_id']} radius={radius}", flush=True)
                        config = ExcursionConfig(alpha=.65, beta=.65, max_relative_norm=radius, visit_step=2)
                        record = probe_pair(adapter, a, b, config, answer_ids[0])
                        record.update({'problem_id': pilot['problem_id'], 'family': pilot['family'],
                                       'dtype': str(dtype), 'device': str(adapter.device),
                                       'prompt_token_ids': pilot['prompt_token_ids'],
                                       'prefix_sha256': hash_tensor(torch.tensor([pilot['prompt_token_ids']]), 64),
                                       'first_answer_token_id': answer_ids[0], 'answer_token_ids': answer_ids})
                        report['conditions'].append(record)
                        stream.write(json.dumps({'event': 'condition', **record}, allow_nan=False) + '\n')
                        stream.flush()
                if calls[0] != adapter.recurrent_block_calls:
                    raise ValueError('Physical recurrent norm hook and adapter counter disagree')
                stream.write(json.dumps({'event': 'dtype_calls', 'dtype': str(dtype), 'physical_calls': calls[0]}) + '\n')
            finally:
                handle.remove()
            del adapter, a, b, branches, state
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    report['actual_recurrent_calls'] = sum(p['recurrent_calls'] for p in report['preparation']) + sum(c['total_recurrent_calls'] for c in report['conditions'])
    report['all_checks_passed'] = all(all(c['checks'].values()) for c in report['conditions'])
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    if report['source_sha256'] != {p: sha256(ROOT / p) for p in sources}:
        raise ValueError('Source files changed while the diagnostic ran')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Checks passed: {report['all_checks_passed']}; calls={report['actual_recurrent_calls']}; {args.output}", flush=True)
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
