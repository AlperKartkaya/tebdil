"""Synthetic algebra and control checks; these are not real-model evidence."""

from dataclasses import replace

import pytest
import torch

from tebdil.core.controls import (
    build_output_matched_random_state,
    execute_norm_matched_random_perturbation_step,
    execute_update_mixing_step,
)
from tebdil.core.excursions import build_visit_state, execute_excursion_step
from tebdil.core.state import BranchState, ExcursionConfig, TransitionContext, validate_state_compatibility
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


class PolynomialAdapter:
    validate_compatibility = staticmethod(validate_state_compatibility)

    def __init__(self, nonlinear=False):
        self.nonlinear = nonlinear
        self.inputs = []

    def transition(self, state):
        self.inputs.append(state.hidden.clone())
        hidden = 1.25 * state.hidden + 0.2
        if self.nonlinear:
            hidden = hidden + 0.15 * state.hidden.square()
        return replace(state, hidden=hidden, recurrent_step=state.recurrent_step + 1,
                       context=replace(state.context, step=state.recurrent_step + 1),
                       hidden_history=state.hidden_history + (hidden,))


def pair():
    a = BranchState('A', 1, torch.tensor([[[1., 2.], [3., 4.]]]),
                    TransitionContext(1, 'tokens', 'mask', 'positions'))
    b = a.clone('B')
    b.hidden += torch.tensor([[[2., -1.], [1., 3.]]])
    return a, b


@pytest.mark.parametrize('radius', [0.005, 1.])
@pytest.mark.parametrize('nonlinear', [False, True])
def test_residual_identity_uses_effective_clipped_alpha(radius, nonlinear):
    adapter = PolynomialAdapter(nonlinear)
    a, b = pair()
    cfg = ExcursionConfig(alpha=.65, beta=.65, max_relative_norm=radius)
    visit, alpha, clipped, _, _ = build_visit_state(a, b, cfg)
    fa, fb, fv = [adapter.transition(s).hidden for s in (a, b, visit)]
    va, vb, vv = fa - a.hidden, fb - b.hidden, fv - visit.hidden
    residual = vv - ((1 - alpha) * va + alpha * vb)
    e1, donor, telemetry = execute_excursion_step(adapter, a, b, cfg)
    m1, _, _ = execute_update_mixing_step(adapter, a, b, cfg)
    assert clipped == (radius == .005)
    assert telemetry.effective_alpha == alpha
    torch.testing.assert_close(e1.hidden - m1.hidden, cfg.beta * residual, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(donor.hidden, fb)
    if not nonlinear:
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-6, rtol=0)
        torch.testing.assert_close(e1.hidden, m1.hidden)
    else:
        assert residual.norm() > 1e-5


def test_r0_is_input_noise_not_random_output_correction():
    adapter = PolynomialAdapter(nonlinear=True)
    a, b = pair()
    cfg = ExcursionConfig(alpha=.65, beta=.65, max_relative_norm=.1)
    original = a.hidden.clone()
    _, _, _, visit_norm, _ = build_visit_state(a, b, cfg)
    result, _, _ = execute_norm_matched_random_perturbation_step(adapter, a, b, cfg, seed=42)
    noisy_input = adapter.inputs[0]
    torch.testing.assert_close((noisy_input - a.hidden).norm(), torch.tensor(cfg.beta * visit_norm))
    torch.testing.assert_close(result.hidden, 1.25 * noisy_input + .2 + .15 * noisy_input.square())
    assert not torch.allclose((result.hidden - (1.25 * a.hidden + .2 + .15 * a.hidden.square())).norm(),
                              torch.tensor(cfg.beta * visit_norm))
    assert torch.equal(a.hidden, original)


@pytest.mark.parametrize('scope,positions', [('full', ()), ('last_token', ()), ('selected', (0,))])
def test_output_matching_norm_determinism_scope_and_isolation(scope, positions):
    adapter = PolynomialAdapter()
    a, reference = pair()
    original = a.hidden.clone()
    cfg = ExcursionConfig(token_scope=scope, selected_positions=positions)
    result = build_output_matched_random_state(adapter, a, reference, cfg, seed=42)
    repeat = build_output_matched_random_state(adapter, a, reference, cfg, seed=42)
    different = build_output_matched_random_state(adapter, a, reference, cfg, seed=43)
    torch.testing.assert_close((result.hidden - a.hidden).norm(), (reference.hidden - a.hidden).norm())
    assert torch.equal(result.hidden, repeat.hidden)
    assert not torch.equal(result.hidden, different.hidden)
    assert result.context == a.context and result.recurrent_step == a.recurrent_step
    if scope == 'last_token':
        assert torch.equal(result.hidden[:, :-1], a.hidden[:, :-1])
    if scope == 'selected':
        assert torch.equal(result.hidden[:, 1:], a.hidden[:, 1:])
    assert not adapter.inputs, 'output construction must not execute a recurrent transition'
    result.hidden.zero_()
    assert torch.equal(a.hidden, original)


def test_output_matching_zero_invalid_context_nonfinite_and_empty_scope():
    adapter = PolynomialAdapter()
    a, reference = pair()
    zero = build_output_matched_random_state(adapter, a, a, ExcursionConfig())
    assert torch.equal(zero.hidden, a.hidden) and zero.hidden.data_ptr() != a.hidden.data_ptr()
    with pytest.raises(ValueError, match='scope'):
        build_output_matched_random_state(adapter, a, reference, ExcursionConfig(token_scope='selected', selected_positions=(99,)))
    reference.context = replace(reference.context, token_ids_hash='different')
    with pytest.raises(ValueError, match='Token IDs'):
        build_output_matched_random_state(adapter, a, reference, ExcursionConfig())
    reference.context = a.context
    reference.hidden.fill_(1e38)
    with pytest.raises(ValueError, match='finite'):
        build_output_matched_random_state(adapter, a, reference, ExcursionConfig())
    reference.hidden.fill_(float('nan'))
    with pytest.raises(ValueError, match='finite'):
        build_output_matched_random_state(adapter, a, reference, ExcursionConfig())


def test_output_matching_updates_readout_history_and_gate():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=12)
    normal = adapter.transition(adapter.initialize(torch.tensor([[1, 2, 3]])))
    reference = normal.clone()
    reference.hidden += .1
    before = adapter.recurrent_block_calls
    result = build_output_matched_random_state(adapter, normal, reference, ExcursionConfig())
    assert adapter.recurrent_block_calls == before
    assert torch.equal(result.hidden_history[-1], result.hidden)
    torch.testing.assert_close(result.gate_history[-1], adapter.compute_gate(result.hidden))
    assert not torch.equal(normal.hidden, result.hidden)


def test_m1_gate_matches_the_returned_mixed_state():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=12)
    a = adapter.transition(adapter.initialize(torch.tensor([[1, 2, 3]])))
    b = a.clone('B')
    b.hidden += .4
    m1, _, _ = execute_update_mixing_step(adapter, a, b, ExcursionConfig())
    torch.testing.assert_close(m1.gate_history[-1], adapter.compute_gate(m1.hidden), atol=0, rtol=0)


def test_fixed_prefix_probe_accounts_for_all_reference_and_continuation_calls():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('operator_probe', Path(__file__).parents[1] / 'scripts/verify_operator_controls.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=12)
    a = adapter.transition(adapter.transition(adapter.initialize(torch.tensor([[1, 2, 3]]))))
    b = a.clone('B')
    b.hidden += .1
    record = module.probe_pair(adapter, a, b, ExcursionConfig(alpha=.65, beta=.65), answer_token=4)
    assert all(record['checks'].values())
    assert record['construction_and_reference_recurrent_calls'] == 10
    assert record['total_recurrent_calls'] == 24
    assert len(record['measurements']) == 14
    assert all(m['continuation_recurrent_calls'] == 1 for m in record['measurements'])
    assert record['precast_algebra_error_norm'] < 1e-5
    assert all(m['relative_norm_error'] < 1e-5 for m in record['measurements'] if 'relative_norm_error' in m)
