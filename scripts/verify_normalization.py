"""Real Ouro fixed-prefix radial/directional and norm-preserving diagnostics.

uv run python scripts/verify_normalization.py --output reports/ouro_normalization.json
No generation, gold answers, controller training or production operator change.
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

from tebdil.analysis.normalization import match_token_norm, radial_decomposition
from tebdil.core.branching import fork_branches
from tebdil.core.excursions import build_visit_state, execute_excursion_step
from tebdil.core.state import ExcursionConfig, frobenius_norm, hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'ByteDance/Ouro-1.4B-Thinking'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.no_grad()
def boundary_observation(module, args, output):
    x = args[0]
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + module.variance_epsilon)
    expected = module.weight * normalized.to(x.dtype)
    return {
        'input_hash': hash_tensor(x, 64), 'output_hash': hash_tensor(output, 64),
        'input_rms_by_token': xf.square().mean(-1).sqrt().tolist(),
        'output_rms_by_token': output.float().square().mean(-1).sqrt().tolist(),
        'deweighted_output_rms_by_token': (output.float() / module.weight.float()).square().mean(-1).sqrt().tolist(),
        'native_formula_max_abs_error': float((output.float() - expected.float()).abs().max().item()),
    }


def changed_state(adapter, base, hidden):
    state = base.clone()
    state.hidden = hidden.to(base.hidden.dtype)
    if not torch.isfinite(state.hidden).all():
        raise ValueError('Diagnostic state became nonfinite')
    state.hidden_history = state.hidden_history[:-1] + (state.hidden,)
    state.gate_history = state.gate_history[:-1] + (adapter.compute_gate(state.hidden),)
    return state


def relative_norm_error(candidate, reference, weight=None):
    divisor = 1 if weight is None else weight.float()
    a, b = candidate.float() / divisor, reference.float() / divisor
    target = b.norm(dim=-1)
    return float(((a.norm(dim=-1) - target).abs() / target.clamp_min(1e-12)).max().item())


@torch.no_grad()
def probe_norms(adapter, a, b, config, measure):
    start = adapter.recurrent_block_calls
    old_hashes = [hash_tensor(s.hidden, 64) for s in (a, b)]
    normal = measure('ordinary_reference', lambda: adapter.transition(a))
    donor = measure('donor_reference', lambda: adapter.transition(b))
    visit, alpha, clipped, _, _ = build_visit_state(a, b, config)
    delta = visit.hidden.float() - a.hidden.float()
    radial, tangent = radial_decomposition(a.hidden, delta)
    weight = adapter.model.model.norm.weight.detach()
    visit_states = {
        'E1': visit,
        'radial_visit': changed_state(adapter, a, a.hidden.float() + radial),
        'tangential_visit': changed_state(adapter, a, a.hidden.float() + tangent),
        'visit_norm_l2': changed_state(adapter, a, match_token_norm(visit.hidden, a.hidden)),
        'visit_norm_weighted': changed_state(adapter, a, match_token_norm(visit.hidden, a.hidden, weight)),
    }
    va = normal.hidden.float() - a.hidden.float()
    vb = donor.hidden.float() - b.hidden.float()
    precast_m1 = a.hidden.float() + ((1 - config.beta) * va + config.beta * ((1 - alpha) * va + alpha * vb))
    candidates = {'ordinary': normal, 'M1': changed_state(adapter, normal, precast_m1)}
    visits = []
    for name, state in visit_states.items():
        out = measure(name + '_visit_computation', lambda: adapter.transition(state))
        update = out.hidden.float() - state.hidden.float()
        returned = a.hidden.float() + ((1 - config.beta) * va + config.beta * update)
        candidates[name] = changed_state(adapter, normal, returned)
        visits.append({
            'method': name, 'visit_hash': hash_tensor(state.hidden, 64),
            'visited_output_hash': hash_tensor(out.hidden, 64),
            'relative_offset_from_A': frobenius_norm(state.hidden.float() - a.hidden.float()) / frobenius_norm(a.hidden),
            'max_l2_norm_error_vs_A': relative_norm_error(state.hidden, a.hidden),
            'max_weighted_norm_error_vs_A': relative_norm_error(state.hidden, a.hidden, weight),
        })
    e1 = candidates['E1']
    candidates['output_norm_l2'] = changed_state(adapter, e1, match_token_norm(e1.hidden, normal.hidden))
    candidates['output_norm_weighted'] = changed_state(adapter, e1, match_token_norm(e1.hidden, normal.hidden, weight))
    core_e1, core_donor, telemetry = measure('core_E1_implementation_check', lambda: execute_excursion_step(adapter, a, b, config))
    construction = adapter.recurrent_block_calls - start
    checks = {
        'manual_E1_matches_core_after_cast': torch.equal(e1.hidden, core_e1.hidden),
        'donor_is_ordinary_transition': torch.equal(core_donor.hidden, donor.hidden),
        'input_states_unchanged': old_hashes == [hash_tensor(s.hidden, 64) for s in (a, b)],
        'no_instability_fallback': not telemetry.skipped_unstable,
    }
    measurements, baseline_logits = [], None
    for name, state in candidates.items():
        before = adapter.recurrent_block_calls
        final = measure(name + '_continuation', lambda: adapter.transition(state))
        logits = adapter.readout_logits(final)[0, -1].float()
        if baseline_logits is None:
            baseline_logits = logits
        logp, base_logp = logits.log_softmax(-1), baseline_logits.log_softmax(-1)
        measurements.append({
            'method': name, 'intervention_output_hash': hash_tensor(state.hidden, 64),
            'final_hidden_hash': hash_tensor(final.hidden, 64),
            'output_shift_from_ordinary': frobenius_norm(state.hidden.float() - normal.hidden.float()),
            'mean_token_cosine_to_ordinary': float(torch.nn.functional.cosine_similarity(state.hidden.float(), normal.hidden.float(), dim=-1).mean().item()),
            'max_l2_norm_error_vs_ordinary': relative_norm_error(state.hidden, normal.hidden),
            'max_weighted_norm_error_vs_ordinary': relative_norm_error(state.hidden, normal.hidden, weight),
            'next_token_id': int(logits.argmax().item()),
            'logit_l2_vs_ordinary': frobenius_norm(logits - baseline_logits),
            'kl_method_to_ordinary': max(0., float((logp.exp() * (logp - base_logp)).sum().item())),
            'continuation_recurrent_calls': adapter.recurrent_block_calls - before,
        })
    return {
        'alpha': config.alpha, 'effective_alpha': alpha, 'beta': config.beta,
        'max_relative_norm': config.max_relative_norm, 'clipped': clipped,
        'input_state_hashes': old_hashes,
        'radial_norm': frobenius_norm(radial), 'tangential_norm': frobenius_norm(tangent),
        'radial_tangent_max_token_inner_product': float((radial * tangent).sum(-1).abs().max().item()),
        'decomposition_error_norm': frobenius_norm(delta - radial - tangent),
        'construction_and_reference_recurrent_calls': construction,
        'total_recurrent_calls': adapter.recurrent_block_calls - start,
        'visits': visits, 'measurements': measurements, 'checks': checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    journal = args.output.with_suffix('.jsonl')
    if args.output.exists() or journal.exists():
        parser.error('output or journal exists; preserve earlier evidence')
    sources = ['scripts/verify_normalization.py', 'docs/normalization_protocol.md',
               'src/tebdil/analysis/normalization.py', 'src/tebdil/core/excursions.py',
               'src/tebdil/core/branching.py', 'src/tebdil/core/perturbations.py',
               'src/tebdil/core/state.py', 'src/tebdil/models/ouro_adapter.py',
               'vendor/ouro_1_4b_thinking/modeling_ouro.py',
               'vendor/ouro_1_4b_thinking/configuration_ouro.py', 'uv.lock']
    inputs = ['reports/ouro_baseline_calibration.json', 'site/static/evidence/model_provenance.json']
    report = {
        'schema_version': 1, 'purpose': 'Exploratory normalization sensitivity, not answer accuracy',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: sha256(ROOT / p) for p in sources},
        'input_sha256': {p: sha256(ROOT / p) for p in inputs},
        'model_id': MODEL, 'model_revision': PINNED_OURO_THINKING_REV,
        'torch_version': torch.__version__, 'cache_mode': 'disabled', 'readout_mode': 'last_step',
        'total_depth': 4, 'fork_depth': 1, 'intervention_input_depth': 2,
        'token_scope': 'full', 'branch_seed': 20261031, 'donor_relative_norm': .25,
        'independent_prefixes': 3, 'conditions': [], 'preparation': [], 'boundary_records': [],
        'normalization': [],
        'limitations': [
            'Three reused pilot prefixes; twelve dependent conditions; no generated answers or gold inputs',
            'No efficacy, statistical, learned-controller, semantic-manifold or compute-matched claim',
            'Norm rescaling changes realized displacement and may exceed the initial trust radius',
            'Euclidean and learned-weight-divided coordinate norms are distinct diagnostic choices',
            'Casting can undo exact norm preservation; realized errors are retained',
            'Native RMSNorm retains epsilon and learned weights; production computation is unchanged',
            'No cache, long-context or CUDA verification unless recorded',
        ],
    }
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    provenance = json.loads((ROOT / inputs[1]).read_text())
    report['checkpoint_sha256'] = {p: sha256(snapshot / p) for p in provenance['weight_hashes']}
    if not all(report['checkpoint_sha256'][p] == v['sha256'] for p, v in provenance['weight_hashes'].items()):
        raise ValueError('Checkpoint provenance mismatch')
    pilots = [r for r in json.loads((ROOT / inputs[0]).read_text())['records'] if r['prompt_format'] == 'chat']
    if len(pilots) != 3:
        raise ValueError('Expected three pilot chat prefixes')
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x') as stream:
        stream.write(json.dumps({'event': 'start', **report}, allow_nan=False) + '\n')
        stream.flush()
        for dtype in (torch.float32, torch.bfloat16):
            adapter = OuroRecurrentAdapter.from_pretrained(dtype=dtype, local_files_only=True, total_ut_steps=4)
            module = adapter.model.model.norm
            if (module.weight.abs() <= 1e-8).any() or not torch.isfinite(module.weight).all():
                raise ValueError('Native norm weights unsafe for metric division')
            report['normalization'].append({'dtype': str(dtype), 'epsilon': module.variance_epsilon,
                                            'weight_hash': hash_tensor(module.weight, 64),
                                            'min_abs_weight': float(module.weight.abs().min().item()),
                                            'max_abs_weight': float(module.weight.abs().max().item())})
            metadata = {'dtype': str(dtype), 'device': str(adapter.device)}
            before_boundaries = len(report['boundary_records'])

            def hook(module, args, output):
                report['boundary_records'].append({**metadata, **boundary_observation(module, args, output)})

            def measure(label, operation):
                metadata['operation'] = label
                return operation()

            handle = module.register_forward_hook(hook)
            try:
                for pilot in pilots:
                    metadata.update({'problem_id': pilot['problem_id'], 'radius': None})
                    before = adapter.recurrent_block_calls
                    state = measure('prepare_depth_one', lambda adapter=adapter: adapter.transition(adapter.initialize(torch.tensor([pilot['prompt_token_ids']]))))
                    branches, _ = fork_branches(state, count=2, relative_norm=.25, seed=20261031)
                    a = measure('prepare_A_depth_two', lambda adapter=adapter, branches=branches: adapter.transition(branches[0]))
                    b = measure('prepare_B_depth_two', lambda adapter=adapter, branches=branches: adapter.transition(branches[1]))
                    report['preparation'].append({'problem_id': pilot['problem_id'], 'dtype': str(dtype),
                                                  'recurrent_calls': adapter.recurrent_block_calls - before})
                    for radius in (.1, .005):
                        metadata['radius'] = radius
                        print(f'{dtype} {pilot["problem_id"]} radius={radius}', flush=True)
                        record = probe_norms(adapter, a, b, ExcursionConfig(alpha=.65, beta=.65, max_relative_norm=radius), measure)
                        record.update({'dtype': str(dtype), 'device': str(adapter.device),
                                       'problem_id': pilot['problem_id'], 'family': pilot['family'],
                                       'prompt_token_ids': pilot['prompt_token_ids'],
                                       'prefix_sha256': hash_tensor(torch.tensor([pilot['prompt_token_ids']]), 64)})
                        report['conditions'].append(record)
                        stream.write(json.dumps({'event': 'condition', **record}, allow_nan=False) + '\n')
                        stream.flush()
                observed_calls = len(report['boundary_records']) - before_boundaries
                if observed_calls != adapter.recurrent_block_calls:
                    raise ValueError('Native boundary hooks disagree with adapter call counter')
                stream.write(json.dumps({'event': 'boundaries', 'dtype': str(dtype), 'records': report['boundary_records'][before_boundaries:]}, allow_nan=False) + '\n')
            finally:
                handle.remove()
            del adapter, a, b, state, branches, module
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    report['actual_recurrent_calls'] = sum(p['recurrent_calls'] for p in report['preparation']) + sum(c['total_recurrent_calls'] for c in report['conditions'])
    report['all_checks_passed'] = all(all(c['checks'].values()) for c in report['conditions']) and all(r['native_formula_max_abs_error'] == 0. for r in report['boundary_records'])
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    if report['source_sha256'] != {p: sha256(ROOT / p) for p in sources}:
        raise ValueError('Sources changed during diagnostic')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Checks passed: {report['all_checks_passed']}; calls={report['actual_recurrent_calls']}", flush=True)
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
