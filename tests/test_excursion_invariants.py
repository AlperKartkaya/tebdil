"""Tests for TEBDIL excursion operator mathematical invariants and stability safeguards."""

import torch

from tebdil.core.branching import fork_branches
from tebdil.core.controls import execute_update_mixing_step
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.state import ExcursionConfig
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def test_excursion_identity_when_alpha_or_beta_zero_or_equal_states():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=21)
    s1 = adapter.transition(adapter.initialize(torch.tensor([[2, 4, 8, 16]])))
    branches, _ = fork_branches(s1, count=2, relative_norm=0.05, seed=42)
    a, b = branches[0], branches[1]

    expected_a = adapter.transition(a.clone())

    # Case 1: alpha = 0
    cfg_a0 = ExcursionConfig(alpha=0.0, beta=0.5, visit_step=1)
    na_a0, _, tel_a0 = execute_excursion_step(adapter, a.clone(), b.clone(), cfg_a0)
    assert torch.equal(na_a0.hidden, expected_a.hidden)
    assert tel_a0.curvature_frob == 0.0

    # Case 2: beta = 0
    cfg_b0 = ExcursionConfig(alpha=0.5, beta=0.0, visit_step=1)
    na_b0, _, tel_b0 = execute_excursion_step(adapter, a.clone(), b.clone(), cfg_b0)
    assert torch.equal(na_b0.hidden, expected_a.hidden)
    assert tel_b0.curvature_frob == 0.0

    # Case 3: A == B
    cfg_eq = ExcursionConfig(alpha=0.5, beta=0.5, visit_step=1)
    na_eq, _, tel_eq = execute_excursion_step(adapter, a.clone(), a.clone("B"), cfg_eq)
    assert torch.equal(na_eq.hidden, expected_a.hidden)
    assert tel_eq.curvature_frob == 0.0


def test_excursion_curvature_ablation_matches_update_mixing_m1():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=33)
    s1 = adapter.transition(adapter.initialize(torch.tensor([[7, 14, 21, 28]])))
    branches, _ = fork_branches(s1, count=2, relative_norm=0.08, seed=99)
    a, b = branches[0], branches[1]

    cfg = ExcursionConfig(alpha=0.5, beta=0.5, max_relative_norm=0.20, visit_step=1)
    na_full, _, tel_full = execute_excursion_step(adapter, a.clone(), b.clone(), cfg, ablate_curvature=False)
    na_no_c, _, _ = execute_excursion_step(adapter, a.clone(), b.clone(), cfg, ablate_curvature=True)
    na_m1, _, _ = execute_update_mixing_step(adapter, a.clone(), b.clone(), cfg)

    assert tel_full.curvature_frob > 0.0
    assert torch.allclose(na_no_c.hidden, na_m1.hidden, atol=1e-6)
    assert not torch.allclose(na_full.hidden, na_m1.hidden, atol=1e-6)


def test_excursion_trust_region_clipping_and_nan_fallback():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=44)
    s1 = adapter.transition(adapter.initialize(torch.tensor([[1, 3, 5, 7]])))
    branches, _ = fork_branches(s1, count=2, relative_norm=0.50, seed=55)
    a, b = branches[0], branches[1]

    cfg_clip = ExcursionConfig(alpha=1.0, beta=0.5, max_relative_norm=0.05, visit_step=1)
    _, _, tel_clip = execute_excursion_step(adapter, a.clone(), b.clone(), cfg_clip)
    assert tel_clip.clipped is True
    assert abs(tel_clip.visit_relative_offset - 0.05) <= 1e-4

    # Corrupt B with NaN and verify fallback_if_unstable='skip_intervention' preserves finite A
    b_nan = b.clone()
    b_nan.hidden[0, -1, 0] = float("nan")
    expected_a = adapter.transition(a.clone())
    na_safe, _, tel_safe = execute_excursion_step(adapter, a.clone(), b_nan, cfg_clip)
    assert tel_safe.skipped_unstable is True
    assert tel_safe.post_finite is True
    assert torch.equal(na_safe.hidden, expected_a.hidden)
