import torch
import pytest
from tebdil.core import policies
from tebdil.core.branching import fork_branches
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.state import ExcursionConfig
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


@pytest.mark.parametrize('candidate,accepted', [([2., 0., 0.], True), ([1., 0., 0.], False), ([0.5, 0., 0.], False), ([float('nan'), 0., 0.], False)])
def test_receiver_acceptance(candidate, accepted):
    class Adapter:
        def readout_logits(self, state, **kwargs):
            return torch.tensor([[state]])
    assert policies.accept_returned_excursion(Adapter(), [1., 0., 0.], candidate) is accepted


@pytest.mark.parametrize('accept', [True, False])
def test_e3_returns_exact_selected_state_and_cache(monkeypatch, accept):
    adapter = OuroRecurrentAdapter.from_mini_config(seed=21)
    adapter.prefill_kv_cache(torch.tensor([[2, 4]]))
    root = adapter.initialize_cached_step(torch.tensor([[8]]))
    state = adapter.transition(root)
    a, b = fork_branches(state, count=2, relative_norm=0.1, seed=42)[0]
    normal = adapter.transition(a.clone())
    cfg = ExcursionConfig(alpha=0.65, beta=0.65, visit_step=1)
    candidate, _, _ = execute_excursion_step(adapter, a.clone(), b.clone(), cfg)
    before = adapter.recurrent_block_calls
    monkeypatch.setattr(policies, 'accept_returned_excursion', lambda *args: accept)
    selected, _, telemetry = execute_excursion_step(adapter, a.clone(), b.clone(), cfg, require_return_improvement=True)
    expected = candidate if accept else normal
    assert adapter.recurrent_block_calls - before == 3
    assert torch.equal(selected.hidden, expected.hidden)
    assert telemetry.return_rejected is (not accept)
    assert (telemetry.effective_alpha != 0) is accept
    for selected_layer, expected_layer in zip(selected.step_kv, expected.step_kv):
        for selected_tensor, expected_tensor in zip(selected_layer, expected_layer):
            assert torch.equal(selected_tensor, expected_tensor)


def test_rejected_e3_generations_match_s0_with_real_cache(monkeypatch):
    from types import SimpleNamespace
    from tebdil.core.scheduler import generate_tebdil_cached
    monkeypatch.setattr(policies.AdaptiveExcursionPolicy, 'decide', lambda *a, **k: SimpleNamespace(action='visit', donor_index=0, adaptive_alpha=0.65, adaptive_beta=0.65))
    monkeypatch.setattr(policies, 'accept_returned_excursion', lambda *a: False)
    adapter = OuroRecurrentAdapter.from_mini_config(seed=21)
    ids = torch.tensor([[2, 4, 8]])
    expected = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=3)
    expected_cache = [k.clone() for k in adapter._active_kv_cache.key_cache]
    result = generate_tebdil_cached(adapter, None, ids, method='E3', max_new_tokens=3)
    assert result['generated_token_ids'] == expected['generated_token_ids']
    assert result['excursion_count'] == 0
    assert result['rejected_return_token_indices'] == [0, 1, 2]
    assert result['recurrent_block_calls'] > expected['recurrent_block_calls']
    for actual, expected in zip(adapter._active_kv_cache.key_cache, expected_cache):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
