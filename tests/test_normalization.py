"""Coordinate diagnostics on synthetic tensors; not model or efficacy evidence."""

import pytest
import torch

from tebdil.analysis.normalization import match_token_norm, radial_decomposition


def test_radial_decomposition_is_per_token_orthogonal_and_reconstructs():
    anchor = torch.tensor([[[1., 2., 3.], [3., -1., 2.]]])
    delta = torch.tensor([[[.2, -.4, .5], [.1, .5, -.3]]])
    radial, tangent = radial_decomposition(anchor, delta)
    torch.testing.assert_close(radial + tangent, delta)
    torch.testing.assert_close((tangent * anchor).sum(-1), torch.zeros((1, 2)), atol=1e-6, rtol=0)
    assert radial.norm() > 0 and tangent.norm() > 0


@pytest.mark.parametrize('weight', [None, torch.tensor([.5, -2., 3.])])
def test_token_norm_matching_preserves_direction_and_requested_metric(weight):
    candidate = torch.tensor([[[1., 2., 3.], [3., -1., 2.]]])
    reference = torch.tensor([[[.2, -.4, .5], [.1, .5, -.3]]])
    result = match_token_norm(candidate, reference, weight)
    scale = 1 if weight is None else weight
    torch.testing.assert_close((result / scale).norm(dim=-1), (reference / scale).norm(dim=-1))
    torch.testing.assert_close(torch.nn.functional.cosine_similarity(result, candidate, dim=-1), torch.ones((1, 2)))
    assert result.data_ptr() != candidate.data_ptr()


def test_norm_matching_zero_nonfinite_shape_and_weight_boundaries():
    zero = torch.zeros((1, 2, 3))
    one = torch.ones_like(zero)
    assert torch.equal(match_token_norm(zero, zero), zero)
    assert torch.equal(match_token_norm(one, zero), zero)
    with pytest.raises(ValueError, match='zero'):
        match_token_norm(zero, one)
    with pytest.raises(ValueError, match='finite'):
        match_token_norm(one * float('nan'), one)
    with pytest.raises(ValueError, match='shape'):
        match_token_norm(one, one[:, :1])
    with pytest.raises(ValueError, match='weight'):
        match_token_norm(one, one, torch.zeros(3))
    with pytest.raises(ValueError, match='zero'):
        radial_decomposition(zero, one)


def test_native_boundary_observer_and_probe_on_mini_model():
    import importlib.util
    from pathlib import Path
    from tebdil.core.state import ExcursionConfig
    from tebdil.models.ouro_adapter import OuroRecurrentAdapter
    spec = importlib.util.spec_from_file_location('norm_probe', Path(__file__).parents[1] / 'scripts/verify_normalization.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=12)
    observed = []
    handle = adapter.model.model.norm.register_forward_hook(lambda m, a, o: observed.append(module.boundary_observation(m, a, o)))
    a = adapter.transition(adapter.transition(adapter.initialize(torch.tensor([[1, 2, 3]]))))
    b = a.clone('B')
    b.hidden += .1
    try:
        result = module.probe_norms(adapter, a, b, ExcursionConfig(alpha=.65, beta=.65), lambda name, operation: operation())
    finally:
        handle.remove()
    assert all(result['checks'].values())
    assert result['construction_and_reference_recurrent_calls'] == 10
    assert result['total_recurrent_calls'] == 19
    assert len(result['measurements']) == 9 and len(result['visits']) == 5
    assert len(observed) == adapter.recurrent_block_calls == 21
    assert all(r['native_formula_max_abs_error'] == 0 for r in observed)
    for m in result['measurements']:
        if m['method'] == 'output_norm_l2':
            assert m['max_l2_norm_error_vs_ordinary'] < 1e-6
        if m['method'] == 'output_norm_weighted':
            assert m['max_weighted_norm_error_vs_ordinary'] < 1e-6
