"""Synthetic geometry properties, not evidence of reasoning improvement."""
from dataclasses import replace

import pytest
import torch

from tebdil.core.geometry import (
    GeometryAnchor, GeometryGate, GeometryProfile, diagnose_motion, select_waypoint,
)

CYCLE = torch.tensor([[1., 0., 0.], [.8, .6, 0.], [1., 0., 0.]], dtype=torch.float64)
PROGRESS = torch.tensor([[1., 0., 0.], [.8, .6, 0.], [.6, .8, 0.]], dtype=torch.float64)


def bank():
    return [GeometryAnchor('request', 0, 2, torch.tensor([.8, .6, 0.], dtype=torch.float64)),
            GeometryAnchor('request', 1, 2, torch.tensor([.8, 0., .6], dtype=torch.float64))]


def choose(path=CYCLE, anchors=None, **kwargs):
    return select_waypoint(path, bank() if anchors is None else anchors,
                           request_id='request', token_position=3, boundary=2,
                           reference_scale=0.2, **kwargs)


def test_progress_cycle_and_healthy_convergence():
    cycle = diagnose_motion(CYCLE, 0.2)
    assert cycle.reason == 'cycling' and cycle.efficiency == 0
    assert diagnose_motion(PROGRESS, 0.2).reason == 'progress'
    flat = diagnose_motion(CYCLE[-1:].repeat(3, 1), 0.2)
    assert flat.reason == 'flattening'
    gate = GeometryGate()
    for pos in range(5):
        assert not gate.observe(pos, flat, ambiguous=False)


def test_candidate_is_local_but_not_in_repeated_direction():
    selected = choose()
    assert selected.anchor_position == 1
    assert selected.cosine == pytest.approx(.8)
    assert selected.novelty == pytest.approx(1)
    assert selected.score == pytest.approx(.9)
    assert selected.radius == pytest.approx(.1)
    assert torch.dot(selected.direction, CYCLE[-1]) == 0
    assert torch.linalg.vector_norm(selected.direction) == 1


def test_no_random_fallback_or_input_mutation():
    before = torch.random.get_rng_state().clone()
    path = CYCLE.clone()
    anchors = bank()
    originals = [a.vector.clone() for a in anchors]
    assert choose(anchors=[]) is None
    assert choose(anchors=anchors[:1]) is None  # Only the repeated direction exists.
    assert choose(CYCLE[-1:].repeat(3, 1)) is None
    first, second = choose(path, anchors), choose(path, list(reversed(anchors)))
    assert first.anchor_position == second.anchor_position
    assert torch.equal(first.direction, second.direction)
    assert torch.equal(path, CYCLE)
    assert all(torch.equal(a.vector, b) for a, b in zip(anchors, originals))
    assert torch.equal(before, torch.random.get_rng_state())


def test_rotation_and_uniform_scale_equivariance():
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64))
    expected = choose()
    for scale in (0.01, 1., 100.):
        path = CYCLE @ rotation * scale
        anchors = [replace(a, vector=a.vector @ rotation * scale) for a in bank()]
        actual = choose(path, anchors)
        assert actual.anchor_position == expected.anchor_position
        assert actual.score == pytest.approx(expected.score)
        assert actual.radius == expected.radius
        torch.testing.assert_close(actual.direction, expected.direction @ rotation)
        assert diagnose_motion(path, .2).reason == 'cycling'


def test_ties_use_earliest_position_not_list_order():
    anchors = [replace(bank()[1], token_position=2), bank()[1]]
    assert choose(anchors=anchors).anchor_position == 1
    assert choose(anchors=list(reversed(anchors))).anchor_position == 1


@pytest.mark.parametrize('path,scale', [(CYCLE[:2], .2), (CYCLE * 0, .2), (CYCLE * float('nan'), .2), (CYCLE, 0.), (CYCLE, float('nan'))])
def test_invalid_or_insufficient_geometry_abstains(path, scale):
    assert not diagnose_motion(path, scale).warning
    assert select_waypoint(path, bank(), request_id='request', token_position=3, boundary=2, reference_scale=scale) is None


@pytest.mark.parametrize('change', [dict(request_id='other'), dict(boundary=3), dict(token_position=3), dict(token_position=-1), dict(token_position=True)])
def test_mixed_request_phase_and_future_anchors_fail_closed(change):
    with pytest.raises(ValueError, match='causal anchors'):
        choose(anchors=[replace(bank()[0], **change)])


def test_duplicates_dimension_errors_and_nonfinite_anchor():
    with pytest.raises(ValueError):
        choose(anchors=[bank()[0], bank()[0]])
    with pytest.raises(ValueError):
        choose(anchors=[replace(bank()[0], vector=torch.ones(5))])
    with pytest.raises(ValueError):
        choose(path=torch.tensor(1.))
    assert choose(anchors=[replace(bank()[0], vector=torch.full((3,), float('inf')))]) is None


@pytest.mark.parametrize('changes', [dict(min_cosine=.99, max_cosine=.9), dict(max_radius=0), dict(epsilon=float('nan')), dict(min_score=-1)])
def test_invalid_profile_rejected(changes):
    with pytest.raises(ValueError):
        GeometryProfile(**changes)


def test_persistence_attempt_cooldown_gap_reset_and_close():
    motion = diagnose_motion(CYCLE, .2)
    gate = GeometryGate(persistence=2, cooldown_tokens=2)
    assert not gate.observe(0, motion, ambiguous=True)
    assert gate.observe(1, motion, ambiguous=True)
    gate.attempted()  # Same cooldown for success and rollback.
    assert not gate.observe(2, motion, ambiguous=True)
    assert not gate.observe(3, motion, ambiguous=True)
    assert not gate.observe(4, motion, ambiguous=True)
    assert gate.observe(5, motion, ambiguous=True)
    assert not gate.observe(7, motion, ambiguous=True)  # Missing observation resets persistence.
    assert not gate.observe(8, diagnose_motion(PROGRESS, .2), ambiguous=True)
    assert not gate.observe(9, motion, ambiguous=True)
    assert gate.observe(10, motion, ambiguous=True)
    gate.close()
    assert not gate.observe(11, motion, ambiguous=True)
    with pytest.raises(ValueError):
        gate.observe(11, motion, ambiguous=True)
    with pytest.raises(ValueError):
        GeometryGate().attempted()
