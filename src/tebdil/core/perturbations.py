"""Seeded, norm-bounded hidden-state perturbations and scope masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch
from torch import Tensor

from tebdil.core.state import frobenius_norm


@dataclass(frozen=True)
class PerturbationReport:
    method: str
    seed: int
    token_scope: str
    target_relative_norm: float
    actual_relative_norm: float
    delta_frobenius_norm: float
    base_frobenius_norm: float


def build_scope_mask(
    shape: Tuple[int, ...],
    token_scope: str,
    device: torch.device,
    dtype: torch.dtype,
    selected_positions: Tuple[int, ...] = (),
) -> Tensor:
    """Create a multiplicative mask of shape [batch, seq_len, 1] for token_scope."""
    batch_size, seq_len, _ = shape
    if token_scope == "full":
        return torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, seq_len, 1), device=device, dtype=dtype)
    if token_scope == "last_token":
        mask[:, -1:, :] = 1.0
        return mask
    if token_scope == "selected":
        if not selected_positions:
            mask[:, -1:, :] = 1.0
        else:
            for pos in selected_positions:
                idx = pos if pos >= 0 else seq_len + pos
                if 0 <= idx < seq_len:
                    mask[:, idx : idx + 1, :] = 1.0
        return mask
    raise ValueError(f"Unsupported token_scope: {token_scope}")


def apply_perturbation(
    hidden: Tensor,
    method: str = "normalized_gaussian",
    relative_norm: float = 0.01,
    seed: int = 42,
    token_scope: str = "full",
    selected_positions: Tuple[int, ...] = (),
    rank: int = 4,
) -> tuple[Tensor, PerturbationReport]:
    """Return a newly allocated perturbed tensor and its PerturbationReport.

    Never mutates `hidden` in place.
    """
    base = hidden.detach().clone()
    base_f32 = base.float()
    scope_mask = build_scope_mask(
        base.shape,
        token_scope=token_scope,
        device=base.device,
        dtype=torch.float32,
        selected_positions=selected_positions,
    )
    scoped_base_norm = float(torch.linalg.vector_norm(base_f32 * scope_mask).item())
    full_base_norm = frobenius_norm(base)
    reference_norm = scoped_base_norm if scoped_base_norm > 0 else full_base_norm

    if method == "none" or relative_norm <= 0.0:
        return base, PerturbationReport(
            method=method,
            seed=seed,
            token_scope=token_scope,
            target_relative_norm=0.0,
            actual_relative_norm=0.0,
            delta_frobenius_norm=0.0,
            base_frobenius_norm=full_base_norm,
        )

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if method in ("gaussian", "normalized_gaussian"):
        noise = torch.randn(base.shape, generator=gen, dtype=torch.float32).to(base.device)
        noise = noise * scope_mask
        if method == "normalized_gaussian":
            noise_norm = float(torch.linalg.vector_norm(noise).item())
            target_norm = relative_norm * reference_norm
            if noise_norm > 1e-12:
                noise = noise * (target_norm / noise_norm)
        else:
            noise = noise * relative_norm
    elif method == "structured_lowrank":
        b, s, d = base.shape
        r = max(1, min(rank, s, d))
        u = torch.randn((b, s, r), generator=gen, dtype=torch.float32).to(base.device)
        v = torch.randn((b, r, d), generator=gen, dtype=torch.float32).to(base.device)
        noise = torch.bmm(u, v) * scope_mask
        noise_norm = float(torch.linalg.vector_norm(noise).item())
        target_norm = relative_norm * reference_norm
        if noise_norm > 1e-12:
            noise = noise * (target_norm / noise_norm)
    else:
        raise ValueError(f"Unsupported perturbation method: {method}")

    perturbed = (base_f32 + noise).to(dtype=base.dtype)
    actual_delta = (perturbed.float() - base_f32)
    actual_delta_norm = float(torch.linalg.vector_norm(actual_delta).item())
    actual_rel = actual_delta_norm / (reference_norm + 1e-12)

    return perturbed, PerturbationReport(
        method=method,
        seed=seed,
        token_scope=token_scope,
        target_relative_norm=relative_norm,
        actual_relative_norm=actual_rel,
        delta_frobenius_norm=actual_delta_norm,
        base_frobenius_norm=full_base_norm,
    )
