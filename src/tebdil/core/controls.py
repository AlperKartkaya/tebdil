"""Baseline and negative control operators sharing the RecurrentAdapter API."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional
import torch

from tebdil.core.excursions import (
    ExcursionTelemetry,
    build_visit_state,
    compute_branch_similarity,
    execute_excursion_step,
)
from tebdil.core.perturbations import build_scope_mask
from tebdil.core.state import (
    BranchState,
    ExcursionConfig,
    RecurrentAdapter,
    check_tensor_finite,
    compute_rms,
)


def execute_direct_state_mixing_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """M0: Direct hidden-state averaging prior to recurrent transition.

    Mixes H_t^{mix} = (1 - alpha_eff * beta) H_t^A + (alpha_eff * beta) H_t^B
    (using build_visit_state with alpha * beta) and runs a single F_t(H_t^{mix}).
    Uses 2 recurrent calls at step t (one for mixed A', one for B).
    """
    adapter.validate_compatibility(state_a, state_b)
    step = state_a.recurrent_step
    dist, rel_dist, cos_sim = compute_branch_similarity(state_a.hidden, state_b.hidden)

    mix_cfg = replace(config, alpha=config.alpha * config.beta)
    mixed_state, eff_alpha, clipped, visit_norm, visit_rel = build_visit_state(
        state_a, state_b, mix_cfg
    )
    next_a = adapter.transition(mixed_state)
    next_a.branch_id = state_a.branch_id
    next_b = adapter.transition(state_b)

    v_a_mix = (next_a.hidden.detach().float() - state_a.hidden.detach().float())
    v_b = (next_b.hidden.detach().float() - state_b.hidden.detach().float())
    norm_v_a = float(torch.linalg.vector_norm(v_a_mix).item())
    norm_v_b = float(torch.linalg.vector_norm(v_b).item())

    telemetry = ExcursionTelemetry(
        step=step,
        alpha=config.alpha,
        effective_alpha=eff_alpha,
        beta=config.beta,
        token_scope=config.token_scope,
        clipped=clipped,
        skipped_unstable=False,
        branch_distance_frob=dist,
        branch_relative_distance=rel_dist,
        branch_cosine_similarity=cos_sim,
        visit_offset_frob=visit_norm,
        visit_relative_offset=visit_rel,
        update_a_frob=norm_v_a,
        update_b_frob=norm_v_b,
        update_visit_frob=norm_v_a,
        curvature_frob=0.0,
        curvature_relative_to_update_a=0.0,
        post_rms_a=compute_rms(next_a.hidden),
        post_rms_b=compute_rms(next_b.hidden),
        post_finite=check_tensor_finite(next_a.hidden),
    )
    return next_a, next_b, telemetry


def execute_update_mixing_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """M1: Direct update-vector mixing without evaluating F_t(H_t^{visit}).

    H_{t+1}^{A'} = H_t^A + (1 - beta) V_t^A + beta * [(1 - alpha_eff) V_t^A + alpha_eff V_t^B].
    Uses 2 recurrent calls at step t (F_t(H_t^A) and F_t(H_t^B)).
    """
    adapter.validate_compatibility(state_a, state_b)
    step = state_a.recurrent_step
    dist, rel_dist, cos_sim = compute_branch_similarity(state_a.hidden, state_b.hidden)

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

    _, eff_alpha, clipped, visit_norm, visit_rel = build_visit_state(state_a, state_b, config)
    scope_mask = build_scope_mask(
        h_a_f.shape,
        token_scope=config.token_scope,
        device=h_a_f.device,
        dtype=torch.float32,
        selected_positions=config.selected_positions,
    )

    v_mix = (1.0 - eff_alpha) * v_a + eff_alpha * v_b
    combined_update = (1.0 - config.beta) * v_a + config.beta * (
        v_a * (1.0 - scope_mask) + v_mix * scope_mask
    )
    h_a_prime = (h_a_f + combined_update).to(dtype=state_a.hidden.dtype)
    gate_history = next_a_normal.gate_history
    if gate_history and hasattr(adapter, "compute_gate"):
        gate_history = gate_history[:-1] + (adapter.compute_gate(h_a_prime),)

    next_a = BranchState(
        branch_id=state_a.branch_id,
        recurrent_step=next_a_normal.recurrent_step,
        hidden=h_a_prime,
        context=next_a_normal.context,
        gate_history=gate_history,
        hidden_history=state_a.hidden_history + (h_a_prime,),
    )
    telemetry = ExcursionTelemetry(
        step=step,
        alpha=config.alpha,
        effective_alpha=eff_alpha,
        beta=config.beta,
        token_scope=config.token_scope,
        clipped=clipped,
        skipped_unstable=False,
        branch_distance_frob=dist,
        branch_relative_distance=rel_dist,
        branch_cosine_similarity=cos_sim,
        visit_offset_frob=visit_norm,
        visit_relative_offset=visit_rel,
        update_a_frob=norm_v_a,
        update_b_frob=norm_v_b,
        update_visit_frob=float(torch.linalg.vector_norm(v_mix).item()),
        curvature_frob=0.0,
        curvature_relative_to_update_a=0.0,
        post_rms_a=compute_rms(h_a_prime),
        post_rms_b=compute_rms(next_b.hidden),
        post_finite=check_tensor_finite(h_a_prime),
    )
    return next_a, next_b, telemetry


def execute_visit_without_return_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """C0: Jump to H_t^{visit} and continue from F_t(H_t^{visit}) without returning to H_t^A."""
    adapter.validate_compatibility(state_a, state_b)
    step = state_a.recurrent_step
    dist, rel_dist, cos_sim = compute_branch_similarity(state_a.hidden, state_b.hidden)

    visit_state, eff_alpha, clipped, visit_norm, visit_rel = build_visit_state(
        state_a, state_b, config
    )
    next_visit = adapter.transition(visit_state)
    next_b = adapter.transition(state_b)

    next_a = BranchState(
        branch_id=state_a.branch_id,
        recurrent_step=next_visit.recurrent_step,
        hidden=next_visit.hidden,
        context=next_visit.context,
        gate_history=next_visit.gate_history,
        hidden_history=state_a.hidden_history + (next_visit.hidden,),
    )
    v_visit = next_visit.hidden.detach().float() - visit_state.hidden.detach().float()
    v_b = next_b.hidden.detach().float() - state_b.hidden.detach().float()

    telemetry = ExcursionTelemetry(
        step=step,
        alpha=config.alpha,
        effective_alpha=eff_alpha,
        beta=1.0,
        token_scope=config.token_scope,
        clipped=clipped,
        skipped_unstable=False,
        branch_distance_frob=dist,
        branch_relative_distance=rel_dist,
        branch_cosine_similarity=cos_sim,
        visit_offset_frob=visit_norm,
        visit_relative_offset=visit_rel,
        update_a_frob=float(torch.linalg.vector_norm(v_visit).item()),
        update_b_frob=float(torch.linalg.vector_norm(v_b).item()),
        update_visit_frob=float(torch.linalg.vector_norm(v_visit).item()),
        curvature_frob=0.0,
        curvature_relative_to_update_a=0.0,
        post_rms_a=compute_rms(next_a.hidden),
        post_rms_b=compute_rms(next_b.hidden),
        post_finite=check_tensor_finite(next_a.hidden),
    )
    return next_a, next_b, telemetry


def execute_norm_matched_random_perturbation_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
    seed: int = 42,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """R0: Input noise of norm beta * clipped_visit_norm, then F_t(A + noise).

    Historical name retained for compatibility. This does not match the output
    correction norm of an excursion and does not transport a visited update.
    """
    adapter.validate_compatibility(state_a, state_b)
    step = state_a.recurrent_step
    dist, rel_dist, cos_sim = compute_branch_similarity(state_a.hidden, state_b.hidden)

    _, eff_alpha, clipped, visit_norm, visit_rel = build_visit_state(state_a, state_b, config)
    h_a_f = state_a.hidden.detach().float()
    scope_mask = build_scope_mask(
        h_a_f.shape,
        token_scope=config.token_scope,
        device=h_a_f.device,
        dtype=torch.float32,
        selected_positions=config.selected_positions,
    )
    gen = torch.Generator(device="cpu").manual_seed(int(seed + step * 997))
    noise = torch.randn(h_a_f.shape, generator=gen, dtype=torch.float32).to(h_a_f.device) * scope_mask
    noise_norm = float(torch.linalg.vector_norm(noise).item())
    target_norm = visit_norm * config.beta
    if noise_norm > 1e-12:
        noise = noise * (target_norm / noise_norm)

    pert_state = state_a.clone(new_branch_id=state_a.branch_id)
    pert_state.hidden = (h_a_f + noise).to(dtype=state_a.hidden.dtype)
    next_a = adapter.transition(pert_state)
    next_b = adapter.transition(state_b)

    v_a = next_a.hidden.detach().float() - pert_state.hidden.detach().float()
    v_b = next_b.hidden.detach().float() - state_b.hidden.detach().float()

    telemetry = ExcursionTelemetry(
        step=step,
        alpha=config.alpha,
        effective_alpha=eff_alpha,
        beta=config.beta,
        token_scope=config.token_scope,
        clipped=clipped,
        skipped_unstable=False,
        branch_distance_frob=dist,
        branch_relative_distance=rel_dist,
        branch_cosine_similarity=cos_sim,
        visit_offset_frob=target_norm,
        visit_relative_offset=target_norm / (float(torch.linalg.vector_norm(h_a_f).item()) + 1e-12),
        update_a_frob=float(torch.linalg.vector_norm(v_a).item()),
        update_b_frob=float(torch.linalg.vector_norm(v_b).item()),
        update_visit_frob=float(torch.linalg.vector_norm(v_a).item()),
        curvature_frob=0.0,
        curvature_relative_to_update_a=0.0,
        post_rms_a=compute_rms(next_a.hidden),
        post_rms_b=compute_rms(next_b.hidden),
        post_finite=check_tensor_finite(next_a.hidden),
    )
    return next_a, next_b, telemetry


@torch.no_grad()
def build_output_matched_random_state(
    adapter: RecurrentAdapter,
    normal_state: BranchState,
    reference_state: BranchState,
    config: ExcursionConfig,
    seed: int = 42,
) -> BranchState:
    """Analysis-only random output displacement matching reference-minus-normal.

    Requires an already computed reference intervention at the same depth/context.
    No recurrent call is made: reference cost must be reported separately. Norms
    match in FP32 before casting; callers must measure realized low-precision error.
    """
    adapter.validate_compatibility(normal_state, reference_state)
    if not all(check_tensor_finite(s.hidden) for s in (normal_state, reference_state)):
        raise ValueError("Output matching requires finite hidden states")
    base = normal_state.hidden.detach().float()
    target_norm = torch.linalg.vector_norm(reference_state.hidden.detach().float() - base)
    if not torch.isfinite(target_norm).item():
        raise ValueError("Output matching requires a finite displacement norm")
    result = normal_state.clone()
    if target_norm.item() == 0.0:
        return result
    mask = build_scope_mask(
        base.shape, config.token_scope, base.device, torch.float32, config.selected_positions,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(base.shape, generator=generator, dtype=torch.float32).to(base.device) * mask
    noise_norm = torch.linalg.vector_norm(noise)
    if noise_norm.item() == 0.0:
        raise ValueError("Output matching requires a nonempty token scope")
    result.hidden = (base + noise * (target_norm / noise_norm)).to(normal_state.hidden.dtype)
    if not check_tensor_finite(result.hidden):
        raise ValueError("Output matching produced a nonfinite state after casting")
    if result.hidden_history:
        result.hidden_history = result.hidden_history[:-1] + (result.hidden,)
    if result.gate_history and hasattr(adapter, "compute_gate"):
        result.gate_history = result.gate_history[:-1] + (adapter.compute_gate(result.hidden),)
    return result


def execute_random_direction_excursion_step(
    adapter: RecurrentAdapter,
    state_a: BranchState,
    state_b: BranchState,
    config: ExcursionConfig,
    seed: int = 42,
) -> tuple[BranchState, BranchState, ExcursionTelemetry]:
    """E0: Excursion-and-return along a random direction of matched distance ||H_t^B - H_t^A||_F."""
    return execute_excursion_step(
        adapter=adapter,
        state_a=state_a,
        state_b=state_b,
        config=config,
        random_direction_seed=seed + state_a.recurrent_step * 7919,
    )
