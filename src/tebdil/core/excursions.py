"""TEBDIL Cooperative Latent Excursion operator and curvature decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor

from tebdil.core.perturbations import build_scope_mask
from tebdil.core.state import (
    BranchState,
    ExcursionConfig,
    RecurrentAdapter,
    check_tensor_finite,
    compute_rms,
    frobenius_norm,
    hash_tensor,
)


@dataclass(frozen=True)
class ExcursionTelemetry:
    step: int
    alpha: float
    effective_alpha: float
    beta: float
    token_scope: str
    clipped: bool
    skipped_unstable: bool
    branch_distance_frob: float
    branch_relative_distance: float
    branch_cosine_similarity: float
    visit_offset_frob: float
    visit_relative_offset: float
    update_a_frob: float
    update_b_frob: float
    update_visit_frob: float
    curvature_frob: float
    curvature_relative_to_update_a: float
    post_rms_a: float
    post_rms_b: float
    post_finite: bool
    donor_branch_id: str = "B"
    receiving_state_hash: str = ""
    donor_state_hash: str = ""
    visit_state_hash: str = ""
    normal_update_hash: str = ""
    visit_update_hash: str = ""
    applied_update_hash: str = ""
    post_receiving_state_hash: str = ""
    applied_update_frob: float = 0.0


def compute_branch_similarity(a_hidden: Tensor, b_hidden: Tensor) -> tuple[float, float, float]:
    """Return (frobenius_distance, relative_distance_to_a, mean_token_cosine_similarity)."""
    a_f = a_hidden.detach().float()
    b_f = b_hidden.detach().float()
    diff = b_f - a_f
    dist = float(torch.linalg.vector_norm(diff).item())
    norm_a = float(torch.linalg.vector_norm(a_f).item())
    rel_dist = dist / (norm_a + 1e-12)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=-1).mean().item()
    return dist, rel_dist, float(cos)


def build_visit_state(
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
    direction_override: Optional[Tensor] = None,
) -> tuple[BranchState, float, bool, float, float]:
    """Construct H_t^{visit} from state_a toward state_b (or direction_override).

    Returns (visit_state, effective_alpha, was_clipped, visit_offset_frob, visit_relative_offset).
    """
    h_a = state_a.hidden.detach().float()
    h_b = state_b.hidden.detach().float()
    scope_mask = build_scope_mask(
        h_a.shape,
        token_scope=config.token_scope,
        device=h_a.device,
        dtype=torch.float32,
        selected_positions=config.selected_positions,
    )

    if direction_override is not None:
        d_vec = direction_override.detach().float() * scope_mask
    else:
        d_vec = (h_b - h_a) * scope_mask

    raw_delta = float(config.alpha) * d_vec
    raw_norm = float(torch.linalg.vector_norm(raw_delta).item())
    norm_a = float(torch.linalg.vector_norm(h_a).item())

    clipped = False
    clip_scale = 1.0
    if config.max_relative_norm > 0.0 and raw_norm > config.max_relative_norm * norm_a:
        clip_scale = (config.max_relative_norm * norm_a) / (raw_norm + 1e-12)
        clipped = True

    effective_alpha = float(config.alpha) * clip_scale
    delta = raw_delta * clip_scale
    visit_norm = float(torch.linalg.vector_norm(delta).item())
    visit_rel = visit_norm / (norm_a + 1e-12)

    h_visit = (h_a + delta).to(dtype=state_a.hidden.dtype)
    visit_state = state_a.clone(new_branch_id=f"{state_a.branch_id}_visit")
    visit_state.hidden = h_visit
    return visit_state, effective_alpha, clipped, visit_norm, visit_rel


def execute_excursion_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
    ablate_curvature: bool = False,
    random_curvature_seed: Optional[int] = None,
    random_direction_seed: Optional[int] = None,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """Execute one recurrent step with cooperative latent excursion from A toward B.

    - Evaluates F_t(H_t^A) and F_t(H_t^B).
    - Constructs H_t^{visit} and evaluates F_t(H_t^{visit}).
    - Transports update V_t^{visit} back to H_t^A with weight beta.
    - Leaves B's trajectory undisturbed: H_{t+1}^B = F_t(H_t^B).
    """
    adapter.validate_compatibility(state_a, state_b)
    step = state_a.recurrent_step
    dist, rel_dist, cos_sim = compute_branch_similarity(state_a.hidden, state_b.hidden)

    # Normal transitions for A and B
    next_a_normal = adapter.transition(state_a)
    next_b = adapter.transition(state_b)

    h_a_f = state_a.hidden.detach().float()
    h_b_f = state_b.hidden.detach().float()
    f_a_f = next_a_normal.hidden.detach().float()
    f_b_f = next_b.hidden.detach().float()

    v_a = f_a_f - h_a_f
    v_b = f_b_f - h_b_f
    norm_v_a = float(torch.linalg.vector_norm(v_a).item())
    norm_v_b = float(torch.linalg.vector_norm(v_b).item())

    # Identity short-circuit when alpha=0, beta=0, or A==B (with no random direction override)
    if (
        config.alpha == 0.0
        or config.beta == 0.0
        or (dist <= 1e-12 and random_direction_seed is None)
    ):
        h_a_hash = hash_tensor(state_a.hidden)
        h_b_hash = hash_tensor(state_b.hidden)
        v_a_hash = hash_tensor(v_a)
        post_a_hash = hash_tensor(next_a_normal.hidden)
        telemetry = ExcursionTelemetry(
            step=step,
            alpha=config.alpha,
            effective_alpha=0.0 if config.alpha == 0.0 or dist <= 1e-12 else config.alpha,
            beta=config.beta,
            token_scope=config.token_scope,
            clipped=False,
            skipped_unstable=False,
            branch_distance_frob=dist,
            branch_relative_distance=rel_dist,
            branch_cosine_similarity=cos_sim,
            visit_offset_frob=0.0,
            visit_relative_offset=0.0,
            update_a_frob=norm_v_a,
            update_b_frob=norm_v_b,
            update_visit_frob=norm_v_a,
            curvature_frob=0.0,
            curvature_relative_to_update_a=0.0,
            post_rms_a=compute_rms(next_a_normal.hidden),
            post_rms_b=compute_rms(next_b.hidden),
            post_finite=check_tensor_finite(next_a_normal.hidden),
            donor_branch_id=state_b.branch_id,
            receiving_state_hash=h_a_hash,
            donor_state_hash=h_b_hash,
            visit_state_hash=h_a_hash,
            normal_update_hash=v_a_hash,
            visit_update_hash=v_a_hash,
            applied_update_hash=v_a_hash,
            post_receiving_state_hash=post_a_hash,
            applied_update_frob=norm_v_a,
        )
        return next_a_normal, next_b, telemetry

    direction_override: Optional[Tensor] = None
    if random_direction_seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(random_direction_seed))
        rand_dir = torch.randn(h_a_f.shape, generator=gen, dtype=torch.float32).to(h_a_f.device)
        rand_norm = float(torch.linalg.vector_norm(rand_dir).item())
        target_d_norm = dist if dist > 1e-12 else 0.05 * float(torch.linalg.vector_norm(h_a_f).item())
        if rand_norm > 1e-12:
            direction_override = rand_dir * (target_d_norm / rand_norm)

    visit_state, eff_alpha, clipped, visit_norm, visit_rel = build_visit_state(
        state_a, state_b, config, direction_override=direction_override
    )
    next_visit = adapter.transition(visit_state)

    h_visit_f = visit_state.hidden.detach().float()
    f_visit_f = next_visit.hidden.detach().float()
    v_visit = f_visit_f - h_visit_f
    norm_v_visit = float(torch.linalg.vector_norm(v_visit).item())

    # Linear update mix vs nonlinear excursion update
    v_mix = (1.0 - eff_alpha) * v_a + eff_alpha * v_b
    curvature = v_visit - v_mix
    norm_curvature = float(torch.linalg.vector_norm(curvature).item())
    rel_curvature = norm_curvature / (norm_v_a + 1e-12)

    scope_mask = build_scope_mask(
        h_a_f.shape,
        token_scope=config.token_scope,
        device=h_a_f.device,
        dtype=torch.float32,
        selected_positions=config.selected_positions,
    )

    if ablate_curvature:
        effective_v_visit = v_mix
    elif random_curvature_seed is not None:
        gen_c = torch.Generator(device="cpu").manual_seed(int(random_curvature_seed))
        rand_c = torch.randn(h_a_f.shape, generator=gen_c, dtype=torch.float32).to(h_a_f.device) * scope_mask
        rand_c_norm = float(torch.linalg.vector_norm(rand_c).item())
        if rand_c_norm > 1e-12:
            rand_c = rand_c * (norm_curvature / rand_c_norm)
        effective_v_visit = v_mix + rand_c
    else:
        effective_v_visit = v_visit

    delta_v = (effective_v_visit - v_a) * scope_mask
    if 0.0 < config.subspace_fraction < 1.0:
        d_dim = delta_v.shape[-1]
        k_keep = max(1, int(round(d_dim * config.subspace_fraction)))
        topk_vals = torch.topk(delta_v.abs(), k=k_keep, dim=-1).values
        thresh = topk_vals[..., -1:]
        subspace_mask = (delta_v.abs() >= thresh).to(dtype= torch.float32)
        delta_v = delta_v * subspace_mask

    gate_scale: Tensor | float = 1.0
    if config.gate_modulated and next_a_normal.gate_history:
        g_prob = torch.sigmoid(next_a_normal.gate_history[-1].detach().float())
        gate_scale = 1.0 - 0.5 * g_prob

    if config.subspace_fraction >= 1.0 and not config.gate_modulated:
        combined_update = (1.0 - config.beta) * v_a + config.beta * (
            v_a * (1.0 - scope_mask) + effective_v_visit * scope_mask
        )
    else:
        combined_update = v_a + config.beta * gate_scale * delta_v
    h_a_prime = (h_a_f + combined_update).to(dtype=state_a.hidden.dtype)

    is_finite = check_tensor_finite(h_a_prime)
    post_rms = compute_rms(h_a_prime) if is_finite else float("inf")
    skipped_unstable = False

    if (not is_finite) or (post_rms > config.rms_Instability_threshold):
        if config.fallback_if_unstable == "skip_intervention":
            h_a_prime = next_a_normal.hidden.detach().clone()
            is_finite = check_tensor_finite(h_a_prime)
            post_rms = compute_rms(h_a_prime)
            skipped_unstable = True

    next_a = BranchState(
        branch_id=state_a.branch_id,
        recurrent_step=next_a_normal.recurrent_step,
        hidden=h_a_prime,
        context=next_a_normal.context,
        gate_history=next_a_normal.gate_history[:-1] + (next_visit.gate_history[-1],)
        if next_a_normal.gate_history and next_visit.gate_history
        else next_a_normal.gate_history,
        hidden_history=state_a.hidden_history + (h_a_prime,),
    )

    telemetry = ExcursionTelemetry(
        step=step,
        alpha=config.alpha,
        effective_alpha=eff_alpha,
        beta=config.beta,
        token_scope=config.token_scope,
        clipped=clipped,
        skipped_unstable=skipped_unstable,
        branch_distance_frob=dist,
        branch_relative_distance=rel_dist,
        branch_cosine_similarity=cos_sim,
        visit_offset_frob=visit_norm,
        visit_relative_offset=visit_rel,
        update_a_frob=norm_v_a,
        update_b_frob=norm_v_b,
        update_visit_frob=norm_v_visit,
        curvature_frob=norm_curvature,
        curvature_relative_to_update_a=rel_curvature,
        post_rms_a=post_rms,
        post_rms_b=compute_rms(next_b.hidden),
        post_finite=is_finite,
        donor_branch_id=state_b.branch_id,
        receiving_state_hash=hash_tensor(state_a.hidden),
        donor_state_hash=hash_tensor(state_b.hidden),
        visit_state_hash=hash_tensor(visit_state.hidden),
        normal_update_hash=hash_tensor(v_a),
        visit_update_hash=hash_tensor(v_visit),
        applied_update_hash=hash_tensor(combined_update),
        post_receiving_state_hash=hash_tensor(h_a_prime),
        applied_update_frob=float(torch.linalg.vector_norm(combined_update).item()),
    )
    return next_a, next_b, telemetry
