"""Core state and configuration types for TEBDIL recurrent interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor


def hash_tensor(tensor: Optional[Tensor], length: int = 16) -> str:
    """Compute a deterministic SHA-256 hex digest of tensor shape, dtype, and bytes."""
    if tensor is None:
        return "none"
    cpu_t = tensor.detach().to(device="cpu").contiguous()
    if cpu_t.dtype == torch.bfloat16:
        raw = cpu_t.view(torch.int16).numpy().tobytes()
    else:
        raw = cpu_t.numpy().tobytes()
    meta = f"{tuple(cpu_t.shape)}:{cpu_t.dtype}:".encode("utf-8")
    return hashlib.sha256(meta + raw).hexdigest()[:length]


@dataclass(frozen=True)
class TransitionContext:
    step: int
    token_ids_hash: str
    mask_hash: str
    positions_hash: str
    cache_mode: str = "disabled"


@dataclass
class BranchState:
    branch_id: str
    recurrent_step: int
    hidden: Tensor  # [batch, tokens, hidden]
    context: TransitionContext
    gate_history: Tuple[Tensor, ...] = field(default_factory=tuple)
    hidden_history: Tuple[Tensor, ...] = field(default_factory=tuple)

    def clone(self, new_branch_id: Optional[str] = None) -> BranchState:
        """Deep-clone the branch state with isolated tensor storage."""
        return BranchState(
            branch_id=new_branch_id if new_branch_id is not None else self.branch_id,
            recurrent_step=self.recurrent_step,
            hidden=self.hidden.detach().clone(),
            context=self.context,
            gate_history=tuple(g.detach().clone() for g in self.gate_history),
            hidden_history=tuple(h.detach().clone() for h in self.hidden_history),
        )


@dataclass(frozen=True)
class ExcursionConfig:
    alpha: float = 0.5
    beta: float = 0.5
    max_relative_norm: float = 0.10
    visit_step: int = 2
    token_scope: str = "full"  # full | last_token | selected
    donor_strategy: str = "fixed"  # fixed | random | diversity | adaptive | oracle
    selected_positions: Tuple[int, ...] = ()
    fallback_if_unstable: str = "skip_intervention"
    rms_Instability_threshold: float = 50.0
    subspace_fraction: float = 1.0
    coordinate_topk_fraction: float = 1.0
    gate_modulated: bool = False
    eligible_visit_steps: Tuple[int, ...] = ()


def check_tensor_finite(tensor: Tensor, name: str = "hidden") -> bool:
    """Return True if tensor contains only finite values and non-zero norm."""
    if not torch.isfinite(tensor).all().item():
        return False
    return True


def compute_rms(tensor: Tensor) -> float:
    """Compute root-mean-square over all elements in float32."""
    return float(tensor.detach().float().pow(2).mean().sqrt().item())


def frobenius_norm(tensor: Tensor) -> float:
    """Compute Frobenius (L2) norm over all elements in float32."""
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def validate_state_compatibility(a: BranchState, b: BranchState) -> None:
    """Standard compatibility validation across two BranchStates."""
    if a.recurrent_step != b.recurrent_step:
        raise ValueError(
            f"Recurrent step mismatch: {a.recurrent_step} vs {b.recurrent_step}"
        )
    if a.context.step != b.context.step or a.context.step != a.recurrent_step:
        raise ValueError(
            f"TransitionContext step mismatch: {a.context.step} vs {b.context.step} (state step {a.recurrent_step})"
        )
    if a.context.token_ids_hash != b.context.token_ids_hash:
        raise ValueError(
            f"Token IDs hash mismatch: {a.context.token_ids_hash} vs {b.context.token_ids_hash}"
        )
    if a.context.mask_hash != b.context.mask_hash:
        raise ValueError(
            f"Mask hash mismatch: {a.context.mask_hash} vs {b.context.mask_hash}"
        )
    if a.context.positions_hash != b.context.positions_hash:
        raise ValueError(
            f"Positions hash mismatch: {a.context.positions_hash} vs {b.context.positions_hash}"
        )
    if a.context.cache_mode != b.context.cache_mode:
        raise ValueError(
            f"Cache mode mismatch: {a.context.cache_mode} vs {b.context.cache_mode}"
        )
    if a.hidden.shape != b.hidden.shape:
        raise ValueError(f"Hidden shape mismatch: {a.hidden.shape} vs {b.hidden.shape}")
    if a.hidden.dtype != b.hidden.dtype:
        raise ValueError(f"Hidden dtype mismatch: {a.hidden.dtype} vs {b.hidden.dtype}")
    if a.hidden.device != b.hidden.device:
        raise ValueError(
            f"Hidden device mismatch: {a.hidden.device} vs {b.hidden.device}"
        )


@runtime_checkable
class RecurrentAdapter(Protocol):
    """Protocol decoupling TEBDIL latent operators from specific LoopLM implementations."""

    @property
    def total_steps(self) -> int:
        ...

    @property
    def hidden_size(self) -> int:
        ...

    @property
    def num_layers(self) -> int:
        ...

    @property
    def recurrent_block_calls(self) -> int:
        ...

    def reset_call_counter(self) -> None:
        ...

    def initialize(self, input_ids: Tensor) -> BranchState:
        ...

    def transition(self, state: BranchState) -> BranchState:
        ...

    def readout_logits(self, state: BranchState) -> Tensor:
        ...

    def validate_compatibility(self, a: BranchState, b: BranchState) -> None:
        ...

