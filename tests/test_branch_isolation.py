"""Tests for branch isolation, storage non-aliasing, and deterministic perturbations."""

import torch

from tebdil.core.branching import fork_branches
from tebdil.core.perturbations import apply_perturbation
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def test_branch_storage_isolation_and_aliasing_safety():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=10)
    state0 = adapter.initialize(torch.tensor([[3, 6, 9, 12]]))
    state1 = adapter.transition(state0)

    branches, reports = fork_branches(
        base_state=state1,
        count=2,
        perturbation_method="normalized_gaussian",
        relative_norm=0.05,
        seed=42,
        token_scope="full",
    )
    branch_a, branch_b = branches[0], branches[1]
    assert branch_a.hidden.data_ptr() != branch_b.hidden.data_ptr()
    assert branch_a.hidden.data_ptr() != state1.hidden.data_ptr()
    assert reports[0].actual_relative_norm == 0.0
    assert abs(reports[1].actual_relative_norm - 0.05) < 1e-4

    # Mutating B in-place must never alter A
    a_snapshot = branch_a.hidden.clone()
    branch_b.hidden.add_(999.0)
    assert torch.equal(branch_a.hidden, a_snapshot)


def test_zero_perturbation_reproduces_identical_branches():
    adapter = OuroRecurrentAdapter.from_mini_config(seed=11)
    state1 = adapter.transition(adapter.initialize(torch.tensor([[5, 10, 15]])))
    branches, _ = fork_branches(
        base_state=state1,
        count=2,
        perturbation_method="normalized_gaussian",
        relative_norm=0.0,
        seed=42,
    )
    next_a = adapter.transition(branches[0])
    next_b = adapter.transition(branches[1])
    assert torch.equal(next_a.hidden, next_b.hidden)


def test_perturbation_seed_determinism_and_token_scopes():
    base = torch.ones((1, 5, 16), dtype=torch.float32)
    p1, _ = apply_perturbation(base, method="normalized_gaussian", relative_norm=0.1, seed=77, token_scope="last_token")
    p2, _ = apply_perturbation(base, method="normalized_gaussian", relative_norm=0.1, seed=77, token_scope="last_token")
    p3, _ = apply_perturbation(base, method="normalized_gaussian", relative_norm=0.1, seed=78, token_scope="last_token")

    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)
    # Prefix tokens (0..3) must remain untouched under token_scope='last_token'
    assert torch.equal(p1[:, :-1, :], base[:, :-1, :])
    assert not torch.equal(p1[:, -1:, :], base[:, -1:, :])
