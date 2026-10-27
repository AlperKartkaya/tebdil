"""Tests for all baseline and control systems and recurrent call accounting."""

import pytest
import torch

from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


@pytest.mark.parametrize(
    "method,expected_calls",
    [
        ("S0", 4),
        ("S1", 7),
        ("P0", 7),
        ("P1", 7),
        ("M0", 7),
        ("M1", 7),
        ("R0", 7),
        ("C0", 7),
        ("E0", 8),
        ("E1", 8),
        ("E1_no_curv", 8),
        ("E1_rand_curv", 8),
    ],
)
def test_control_methods_compute_call_accounting(method: str, expected_calls: int):
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=50)
    input_ids = torch.tensor([[4, 8, 15, 16, 23, 42]])
    cfg = ExcursionConfig(alpha=0.5, beta=0.5, visit_step=2, token_scope="last_token")

    res = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=input_ids,
        method=method,
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        perturbation_relative_norm=0.05,
        excursion_config=cfg,
        seed=42,
        extra_steps_for_s1=3,
    )
    assert res.recurrent_block_calls == expected_calls
    assert torch.isfinite(res.primary_logits).all()


def test_adaptive_e2_budget_and_selective_execution():
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=60)
    input_ids = torch.tensor([[10, 20, 30, 40]])
    cfg = ExcursionConfig(alpha=0.5, beta=0.5, visit_step=2, token_scope="full")

    res = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=input_ids,
        method="E2",
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        perturbation_relative_norm=0.05,
        excursion_config=cfg,
        seed=42,
    )
    assert res.recurrent_block_calls in (7, 8)
    assert res.policy_decision is not None
    assert res.policy_decision.action in ("visit", "no_visit")
