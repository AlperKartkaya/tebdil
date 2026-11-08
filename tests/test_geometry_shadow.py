"""Shadow mechanics and exact non-interference on vendored mini Ouro, not efficacy."""
import json
import pytest
import torch
from tebdil.core.geometry_shadow import GeometryShadow
from tebdil.core.scheduler import generate_tebdil_cached, run_scheduled_trajectory
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def mini(**kwargs):
    return OuroRecurrentAdapter.from_mini_config(vocab_size=32, hidden_size=16,
                                                intermediate_size=32, num_hidden_layers=1, **kwargs)


def test_metric_removes_anisotropic_signed_gains_and_retraction_is_scoped():
    adapter = mini()
    with torch.no_grad():
        adapter.model.model.norm.weight.copy_(torch.linspace(-2., 3., 16))
    state = adapter.transition(adapter.initialize(torch.tensor([[1, 2, 3]])))
    hidden = state.hidden.clone()
    metric = adapter.geometry_coordinates(hidden)
    raw = state.raw_hidden.float()
    torch.testing.assert_close(torch.nn.functional.normalize(metric, dim=-1),
                               torch.nn.functional.normalize(raw, dim=-1), atol=1e-6, rtol=1e-6)
    q = torch.nn.functional.normalize(metric[0, -1], dim=-1)
    tangent = torch.arange(16).float()
    tangent -= tangent.dot(q) * q
    tangent /= tangent.norm()
    calls = adapter.recurrent_block_calls
    result = adapter.geometry_retract_hidden(hidden, tangent, .1)
    assert torch.equal(result[:, :-1], hidden[:, :-1])
    assert torch.equal(state.hidden, hidden)
    assert result.data_ptr() != hidden.data_ptr()
    assert torch.equal(adapter.geometry_retract_hidden(hidden, tangent, 0), hidden)
    actual = adapter.geometry_coordinates(result)
    torch.testing.assert_close(actual.norm(dim=-1), metric.norm(dim=-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.nn.functional.normalize(actual[0, -1], dim=-1),
                               torch.nn.functional.normalize(q + .1 * tangent, dim=-1))
    # Same output as rotating the PRE-norm residual at unchanged raw norm, then native normalization.
    rotated_raw = raw.clone()
    rotated_raw[0, -1] = torch.nn.functional.normalize(q + .1 * tangent, dim=-1) * raw[0, -1].norm()
    torch.testing.assert_close(result, adapter.normalize_hidden(rotated_raw), atol=2e-6, rtol=2e-6)
    assert adapter.recurrent_block_calls == calls
    with pytest.raises(ValueError, match='tangent'):
        adapter.geometry_retract_hidden(hidden, q, .1)
    with pytest.raises(ValueError):
        adapter.geometry_retract_hidden(hidden, tangent, float('nan'))
    with torch.no_grad():
        adapter.model.model.norm.weight[0] = 0
    with pytest.raises(ValueError, match='invertible'):
        adapter.geometry_coordinates(hidden)


def test_causal_reference_bank_warmup_bounded_memory_and_no_rng():
    observer = GeometryShadow()
    path = torch.tensor([[1., 0., 0.], [.8, .6, 0.], [1., 0., 0.]])
    rng = torch.random.get_rng_state().clone()
    for i in range(100):
        observer.observe(path, torch.zeros(8), i)
        row = observer.records[-1]
        assert all(p < i for p in row['bank_positions'])
        assert row['reference_count'] == min(i, observer.reference_capacity)
        assert row['attempted'] is False
    assert len(observer.anchors) == observer.bank_capacity
    assert len(observer.movements) == len(observer.entropies) == observer.reference_capacity
    assert observer.records[7]['reference_scale'] == 0
    assert observer.records[8]['reference_scale'] == pytest.approx(torch.tensor([.2, -.6, 0.]).norm().item())
    assert not observer.records[9]['eligible'] and observer.records[10]['eligible']
    assert torch.equal(rng, torch.random.get_rng_state())
    json.dumps(observer.report(), allow_nan=False)
    with pytest.raises(ValueError, match='causal'):
        observer.observe(path, torch.zeros(8), 99)


def test_current_entropy_does_not_leak_into_threshold_and_invalids_do_not_pollute():
    observer = GeometryShadow()
    path = torch.tensor([[1., 0., 0.], [.8, .6, 0.], [1., 0., 0.]])
    for i in range(8):
        observer.observe(path, torch.zeros(8), i)
    observer.observe(path, torch.tensor([100., 0., 0., 0.]), 8)
    assert observer.records[-1]['ambiguity_threshold'] == pytest.approx(1.)
    assert not observer.records[-1]['ambiguous']
    count = len(observer.movements)
    observer.observe(path * float('nan'), torch.zeros(8), 9)
    observer.observe(path, torch.full((8,), float('inf')), 10)
    assert len(observer.movements) == count
    assert not observer.records[-1]['eligible']
    observer.gate.close()
    observer.observe(path, torch.zeros(8), 11)
    assert not observer.records[-1]['eligible'] and observer.records[-1]['closed']
    assert len(observer.movements) == count
    json.dumps(observer.report(), allow_nan=False)


def test_shadow_generation_preserves_exact_tokens_and_every_cache_slot():
    adapter = mini()
    ids = torch.tensor([[1, 2, 3]])
    before = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=12)
    cache_before = [(k.clone(), v.clone()) for k, v in zip(adapter._active_kv_cache.key_cache,
                                                          adapter._active_kv_cache.value_cache)]
    rng = torch.random.get_rng_state().clone()
    after = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=12, geometry_shadow=True)
    assert 'geometry_shadow' not in before
    for key in before.keys() - {'wall_seconds'}:
        assert after[key] == before[key]
    for (k, v), actual_k, actual_v in zip(cache_before, adapter._active_kv_cache.key_cache, adapter._active_kv_cache.value_cache):
        assert torch.equal(k, actual_k) and torch.equal(v, actual_v)
    assert torch.equal(rng, torch.random.get_rng_state())
    report = after['geometry_shadow']
    assert report['extra_readout_projections'] == 12
    assert [r['token_position'] for r in report['records']] == list(range(2, 14))
    assert report['records'][0]['reference_count'] == 0
    assert report['records'][0]['bank_positions'] == []
    again = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=12, geometry_shadow=True)
    assert again['geometry_shadow'] == report  # A new request cannot inherit the previous bank.
    json.dumps(after, allow_nan=False)


def test_observer_runs_before_fourth_boundary(monkeypatch):
    adapter = mini()
    observer = GeometryShadow()
    transition = adapter.transition
    seen = []
    def step(state):
        seen.append((state.recurrent_step, len(observer.records)))
        return transition(state)
    monkeypatch.setattr(adapter, 'transition', step)
    run_scheduled_trajectory(adapter, torch.tensor([[1]]), method='S0', record_snapshots=False,
                             record_pca=False, geometry_observer=observer)
    assert seen == [(0, 0), (1, 0), (2, 0), (3, 1)]


def test_shadow_closes_after_think_token_without_stopping_native_generation(monkeypatch):
    adapter = mini()
    class Tokenizer:
        eos_token_id = None
        def convert_tokens_to_ids(self, token): return 7
        def decode(self, ids, **kwargs): return 'fixture'
    def readout(state, last_token_only=False):
        logits = torch.zeros(1, 1, 32)
        logits[..., 7] = 1
        return logits
    monkeypatch.setattr(adapter, 'readout_logits', readout)
    result = generate_tebdil_cached(adapter, Tokenizer(), torch.tensor([[1]]), method='S0',
                                   max_new_tokens=3, geometry_shadow=True)
    assert result['generated_token_ids'] == [7, 7, 7]
    assert [r['closed'] for r in result['geometry_shadow']['records']] == [False, True, True]


@pytest.mark.parametrize('method,steps', [('E2', 4), ('S0', 3)])
def test_shadow_rejects_unsupported_execution_before_model_calls(method, steps):
    adapter = mini(total_ut_steps=steps)
    with pytest.raises(ValueError, match='shadow'):
        generate_tebdil_cached(adapter, None, torch.tensor([[1]]), method=method, geometry_shadow=True)
    assert adapter.recurrent_block_calls == 0


def test_overflowing_entropy_abstains_without_nonfinite_json():
    observer = GeometryShadow()
    path = torch.tensor([[1., 0.], [.8, .6], [1., 0.]])
    observer.observe(path, torch.tensor([3e38, -3e38]), 0)
    assert observer.records[0]['normalized_entropy'] is None
    assert len(observer.movements) == 0
    json.dumps(observer.report(), allow_nan=False)


def test_probe_reports_all_noninterference_checks():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('geometry_probe', Path(__file__).parents[1] / 'scripts/verify_geometry_shadow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.probe(mini(), None, torch.tensor([[1, 2]]), 3)
    assert result['all_passed'], result['checks']
    assert len(result['final_logits_sha256'][0]) == 3
    json.dumps(result, allow_nan=False)
