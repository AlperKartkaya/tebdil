"""Forced waypoints test plumbing only; natural triggers remain frozen for evaluation."""
import pytest
import torch
from tebdil.core.geometry import GeometricWaypoint
from tebdil.core.geometry_controller import GeometryController
from tebdil.core.scheduler import generate_tebdil_cached, run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def mini():
    return OuroRecurrentAdapter.from_mini_config(vocab_size=32, hidden_size=16,
                                                intermediate_size=32, num_hidden_layers=1)


def force_one_waypoint(observer, monkeypatch):
    observe = observer.observe
    def forced(path, logits, position):
        observe(path, logits, position)
        if len(observer.records) != 1:
            return None
        q = path[-1].float() / path[-1].float().norm()
        direction = torch.zeros_like(q)
        direction[q.abs().argmin()] = 1
        direction -= direction.dot(q) * q
        direction /= direction.norm()
        observer.gate.consecutive = observer.gate.persistence
        return GeometricWaypoint(0, direction, .1, .8, 1., .9)
    monkeypatch.setattr(observer, 'observe', forced)


def caches(adapter):
    return [(k.clone(), v.clone()) for k, v in zip(adapter._active_kv_cache.key_cache,
                                                  adapter._active_kv_cache.value_cache)]


def check_selected_visit(adapter, monkeypatch, rollback):
    prompt = torch.tensor([[1, 2, 3]])
    adapter.prefill_kv_cache(prompt[:, :-1])
    normal = run_scheduled_trajectory(adapter, prompt[:, -1:], method='S0', use_cached_step=True,
                                      record_snapshots=False, record_pca=False)
    normal_cache = caches(adapter)
    observer = GeometryController()
    force_one_waypoint(observer, monkeypatch)
    adapter.prefill_kv_cache(prompt[:, :-1])
    cfg = ExcursionConfig(alpha=1., beta=.65, rms_Instability_threshold=0. if rollback else 50.)
    result = run_scheduled_trajectory(adapter, prompt[:, -1:], method='geometry', use_cached_step=True,
                                      record_snapshots=False, record_pca=False, geometry_observer=observer,
                                      geometry_position=2, excursion_config=cfg)
    assert result.recurrent_block_calls == 5  # Three native + normal fourth + selected visit.
    assert result.excursion_telemetry.skipped_unstable == rollback
    assert result.excursion_telemetry.curvature_frob == 0
    assert result.excursion_telemetry.effective_alpha == 1
    assert observer.records[-1]['attempted'] and observer.records[-1]['rolled_back'] == rollback
    assert observer.gate.blocked_through == 20
    actual_cache = caches(adapter)
    for i, ((k, v), (nk, nv)) in enumerate(zip(actual_cache, normal_cache)):
        assert k.shape[2] == v.shape[2] == 3
        if i < 3 or rollback:
            assert torch.equal(k, nk) and torch.equal(v, nv)
    if rollback:
        assert torch.equal(result.primary_state.hidden, normal.primary_state.hidden)
        assert torch.equal(result.primary_logits, normal.primary_logits)
    else:
        assert not torch.equal(result.primary_state.hidden, normal.primary_state.hidden)
        assert any(not torch.equal(k, nk) for (k, _), (nk, _) in zip(actual_cache, normal_cache))
    next_token = result.primary_logits.argmax(-1)[:, -1:]
    following = run_scheduled_trajectory(adapter, next_token, method='geometry', use_cached_step=True,
                                         record_snapshots=False, record_pca=False, geometry_observer=observer,
                                         geometry_position=3, excursion_config=cfg)
    assert following.recurrent_block_calls == 4
    assert not observer.records[-1]['eligible']
    for (k, v), (before_k, before_v) in zip(caches(adapter), actual_cache):
        assert k.shape[2] == 4
        assert torch.equal(k[:, :, :3], before_k) and torch.equal(v[:, :, :3], before_v)


def test_abstaining_geometry_is_native_not_random_or_extra_recurrence():
    adapter = mini()
    ids = torch.tensor([[1, 2]])
    native = generate_tebdil_cached(adapter, None, ids, method='S0', max_new_tokens=8)
    before = caches(adapter)
    rng = torch.random.get_rng_state().clone()
    result = generate_tebdil_cached(adapter, None, ids, method='geometry', max_new_tokens=8)
    assert result['generated_token_ids'] == native['generated_token_ids']
    assert result['recurrent_block_calls'] == native['recurrent_block_calls']
    assert result['geometry']['mode'] == 'active'
    assert result['geometry']['extra_readout_projections'] == 8
    assert not any(r['attempted'] for r in result['geometry']['records'])
    assert torch.equal(rng, torch.random.get_rng_state())
    for (k, v), (nk, nv) in zip(caches(adapter), before):
        assert torch.equal(k, nk) and torch.equal(v, nv)


def test_native_trust_bound_abstains_before_attempt(monkeypatch):
    adapter = mini()
    observer = GeometryController()
    force_one_waypoint(observer, monkeypatch)
    result = run_scheduled_trajectory(adapter, torch.tensor([[1, 2]]), method='geometry',
                                     geometry_observer=observer, geometry_position=1,
                                     excursion_config=ExcursionConfig(max_relative_norm=1e-5),
                                     record_snapshots=False, record_pca=False)
    assert result.recurrent_block_calls == 4
    assert result.excursion_telemetry is None
    assert observer.records[-1]['abstention'] == 'native_trust_bound'
    assert not observer.records[-1]['attempted']


@pytest.mark.parametrize('rollback', [False, True])
def test_selected_visit_commit_or_rollback_and_following_token(monkeypatch, rollback):
    check_selected_visit(mini(), monkeypatch, rollback)


@pytest.mark.real_model
def test_real_selected_visit_commit_and_rollback(monkeypatch):
    adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.float32, device='cuda',
                                                   total_ut_steps=4, attn_implementation='sdpa')
    for rollback in (False, True):
        check_selected_visit(adapter, monkeypatch, rollback)
