"""Native full-prefix reference for cached S0 and selected-branch storage."""
import pytest
import torch
from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


@pytest.mark.parametrize('length', [1, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cached_s0_matches_native_across_tokens(length, dtype):
    adapter = OuroRecurrentAdapter.from_mini_config(dtype=dtype)
    ids = torch.arange(1, length + 1).unsqueeze(0)
    adapter.prefill_kv_cache(ids[:, :-1])
    for _ in range(3):
        with torch.no_grad():
            native = adapter.model(input_ids=ids, use_cache=False, exit_at_step=3).logits[:, -1:, :]
        result = run_scheduled_trajectory(adapter, ids[:, -1:], method='S0', use_cached_step=True, record_snapshots=False, record_pca=False)
        actual = result.primary_logits[:, -1:, :]
        torch.testing.assert_close(actual, native, atol=1e-5 if dtype == torch.float32 else .02, rtol=1e-5 if dtype == torch.float32 else .02)
        assert torch.equal(actual.argmax(-1), native.argmax(-1))
        assert all(k.shape[2] == ids.shape[1] for k in adapter._active_kv_cache.key_cache)
        ids = torch.cat([ids, native.argmax(-1)], dim=1)


@pytest.mark.parametrize('ids,budget,steps', [(torch.tensor([[]],dtype=torch.long), 2, 4), (torch.tensor([[1]]), 0, 4), (torch.tensor([[1],[2]]), 2, 4), (torch.tensor([[1]]), 2, 3)])
def test_cached_generation_rejects_unsupported_inputs(ids, budget, steps):
    from tebdil.core.scheduler import generate_tebdil_cached
    adapter = OuroRecurrentAdapter.from_mini_config()
    with pytest.raises(ValueError):
        generate_tebdil_cached(adapter, None, ids, max_new_tokens=budget, recurrent_steps=steps)


def test_selected_donor_cache_contains_its_actual_depth_slices(monkeypatch):
    from tebdil.core import scheduler
    adapter = OuroRecurrentAdapter.from_mini_config()
    adapter.prefill_kv_cache(torch.tensor([[1, 2]]))
    seen = []
    transition = adapter.transition
    def capture(state):
        result = transition(state)
        seen.append(result)
        return result
    monkeypatch.setattr(adapter, 'transition', capture)
    def choose_donor(adapter, state):
        logits = torch.zeros(1, 1, 128)
        logits[..., 1] = 1 if state.branch_id == 'A' else 10
        return logits
    monkeypatch.setattr(scheduler, '_fast_readout', choose_donor)
    result = scheduler.run_scheduled_trajectory(adapter, torch.tensor([[3]]), method='P0', use_cached_step=True, record_snapshots=False, record_pca=False)
    chosen = result.primary_state.branch_id
    assert chosen != 'A'
    for state in seen:
        if state.branch_id != chosen:
            continue
        for layer, (key, value) in enumerate(state.step_kv):
            slot = (state.recurrent_step - 1) * adapter.num_layers + layer
            torch.testing.assert_close(adapter._active_kv_cache.key_cache[slot][:, :, 2:, :], key, rtol=0, atol=0)
            torch.testing.assert_close(adapter._active_kv_cache.value_cache[slot][:, :, 2:, :], value, rtol=0, atol=0)


@pytest.mark.parametrize('parameter', ['alpha', 'beta'])
def test_cached_noop_excursion_matches_s0_and_is_not_counted(parameter):
    from dataclasses import replace
    from tebdil.core.scheduler import generate_tebdil_cached
    from tebdil.core.state import ExcursionConfig
    adapter = OuroRecurrentAdapter.from_mini_config()
    ids = torch.tensor([[1, 2, 3]])
    baseline = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=3)
    config = replace(ExcursionConfig(update_space='pre_norm', token_scope='last_token'), **{parameter: 0.0})
    result = generate_tebdil_cached(adapter, None, ids, method='E1', max_new_tokens=3, excursion_config=config)
    assert result['generated_token_ids'] == baseline['generated_token_ids']
    assert result['excursion_count'] == 0


def test_unknown_cached_method_is_not_silently_an_excursion():
    from tebdil.core.scheduler import generate_tebdil_cached
    with pytest.raises(ValueError, match='Unsupported cached-generation method'):
        generate_tebdil_cached(None, None, torch.tensor([[1]]), method='typo')
