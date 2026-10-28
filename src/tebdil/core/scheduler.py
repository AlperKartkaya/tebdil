"""Multi-branch time-sliced execution scheduler for recurrent interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from tebdil.controller.learned_policy import (
        ControllerDecision,
        LearnedControllerCheckpoint,
    )

from tebdil.core.branching import fork_branches
from tebdil.core.controls import (
    execute_direct_state_mixing_step,
    execute_norm_matched_random_perturbation_step,
    execute_random_direction_excursion_step,
    execute_update_mixing_step,
    execute_visit_without_return_step,
)
from tebdil.core.excursions import (
    ExcursionTelemetry,
    compute_branch_similarity,
    execute_excursion_step,
)
from tebdil.core.perturbations import PerturbationReport
from tebdil.core.policies import (
    AdaptiveExcursionPolicy,
    AdaptivePolicyConfig,
    PolicyDecision,
    compute_last_token_uncertainty,
)
from tebdil.core.state import (
    BranchState,
    ExcursionConfig,
    RecurrentAdapter,
    check_tensor_finite,
    compute_rms,
    frobenius_norm,
)


@dataclass
class StepSnapshot:
    step: int
    branch_rms: Dict[str, float]
    branch_frob: Dict[str, float]
    branch_entropy: Dict[str, float]
    branch_margin: Dict[str, float]
    branch_top1_token: Dict[str, int]
    ab_distance_frob: float
    ab_relative_distance: float
    ab_cosine_similarity: float
    intervention_applied: bool = False
    excursion_telemetry: Optional[Dict[str, Any]] = None
    policy_decision: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryScheduleResult:
    method: str
    seed: int
    recurrent_steps: int
    fork_after_step: int
    branch_count: int
    primary_state: BranchState
    branch_states: Dict[str, BranchState]
    primary_logits: Tensor
    branch_logits: Dict[str, Tensor]
    recurrent_block_calls: int
    wall_time_ms: float
    perturbation_reports: List[Dict[str, Any]]
    step_snapshots: List[StepSnapshot]
    excursion_telemetry: Optional[ExcursionTelemetry] = None
    policy_decision: Optional[PolicyDecision | ControllerDecision] = None
    controller_calls: int = 0
    controller_overhead_ms: float = 0.0
    exited_early_step: Optional[int] = None
    pca_coords: Dict[str, List[List[float]]] = field(default_factory=dict)
    pca_explained_variance: List[float] = field(default_factory=list)


def _compute_pca_projection(
    branch_vectors: Dict[str, List[np.ndarray]],
) -> tuple[Dict[str, List[List[float]]], List[float]]:
    """Project last-token hidden state trajectories onto 2D PCA for visualization only."""
    all_pts = []
    keys = []
    for b_id, vecs in branch_vectors.items():
        for step_idx, v in enumerate(vecs):
            all_pts.append(v)
            keys.append((b_id, step_idx))
    if len(all_pts) < 2:
        return {b_id: [[0.0, 0.0] for _ in vecs] for b_id, vecs in branch_vectors.items()}, [1.0, 0.0]

    mat = np.stack(all_pts, axis=0).astype(np.float64)
    centered = mat - mat.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        total_var = float(var.sum())
        if total_var > 1e-12:
            evr = [
                float(var[0] / total_var),
                float(var[1] / total_var) if len(var) > 1 else 0.0,
            ]
        else:
            evr = [1.0, 0.0]
        coords_2d = np.zeros((len(all_pts), 2), dtype=np.float64)
        coords_2d[:, 0] = u[:, 0] * s[0]
        if u.shape[1] > 1:
            coords_2d[:, 1] = u[:, 1] * s[1]
    except np.linalg.LinAlgError:
        coords_2d = np.zeros((len(all_pts), 2), dtype=np.float64)
        evr = [0.0, 0.0]

    out: Dict[str, List[List[float]]] = {b_id: [] for b_id in branch_vectors}
    for (b_id, _), pt in zip(keys, coords_2d):
        out[b_id].append([round(float(pt[0]), 5), round(float(pt[1]), 5)])
    return out, [round(x, 4) for x in evr]


def run_scheduled_trajectory(
    adapter: RecurrentAdapter,
    input_ids: Tensor,
    method: str = "E1",
    recurrent_steps: Optional[int] = None,
    fork_after_step: int = 1,
    branch_count: int = 2,
    perturbation_method: str = "normalized_gaussian",
    perturbation_relative_norm: float = 0.02,
    excursion_config: Optional[ExcursionConfig] = None,
    policy_config: Optional[AdaptivePolicyConfig] = None,
    controller_checkpoint: Optional[LearnedControllerCheckpoint] = None,
    seed: int = 42,
    extra_steps_for_s1: int = 3,
    record_pca: bool = True,
) -> TrajectoryScheduleResult:
    """Execute a complete recurrent trajectory under the specified intervention method.

    Supported methods:
      - S0: Standard single trajectory (T steps)
      - S1: Extended single trajectory (T + extra_steps_for_s1 steps)
      - P0: Independent N branches, label-free highest-margin selection
      - P1: Independent N branches (returns all branch logits; caller may apply oracle check)
      - M0: Direct state averaging at visit_step
      - M1: Direct update mixing at visit_step
      - R0: Norm-matched random perturbation at visit_step
      - C0: Visit without return at visit_step
      - E0: Excursion toward random direction at visit_step
      - E1: Fixed-schedule cooperative latent excursion toward donor B
      - E1_no_curv: E1 with curvature residual C_t ablated
      - E1_rand_curv: E1 with curvature residual C_t randomized in direction
      - E2: Adaptive label-free cooperative latent excursion
    """
    if excursion_config is None:
        excursion_config = ExcursionConfig()
    if policy_config is None:
        policy_config = AdaptivePolicyConfig()

    base_steps = recurrent_steps if recurrent_steps is not None else adapter.total_steps
    total_steps = base_steps + extra_steps_for_s1 if method == "S1" else base_steps

    calls_before = adapter.recurrent_block_calls
    t0 = time.perf_counter()

    root = adapter.initialize(input_ids)
    step_snapshots: List[StepSnapshot] = []
    pert_reports: List[PerturbationReport] = []
    exc_telemetry: Optional[ExcursionTelemetry] = None
    pol_decision: Optional[PolicyDecision | ControllerDecision] = None
    ctrl_calls = 0
    ctrl_overhead_ms = 0.0
    exited_step: Optional[int] = None

    branch_vec_history: Dict[str, List[np.ndarray]] = {}

    def _record_vec(state: BranchState) -> None:
        if not record_pca:
            return
        vec = state.hidden[:, -1, :].detach().float().cpu().numpy().reshape(-1)
        branch_vec_history.setdefault(state.branch_id, []).append(vec)

    def _snapshot(
        step_idx: int,
        states: List[BranchState],
        intervened: bool = False,
        telem: Optional[ExcursionTelemetry] = None,
        dec: Optional[PolicyDecision] = None,
    ) -> StepSnapshot:
        rms_map: Dict[str, float] = {}
        frob_map: Dict[str, float] = {}
        ent_map: Dict[str, float] = {}
        mar_map: Dict[str, float] = {}
        top1_map: Dict[str, int] = {}
        for st in states:
            rms_map[st.branch_id] = compute_rms(st.hidden)
            frob_map[st.branch_id] = frobenius_norm(st.hidden)
            logits = adapter.readout_logits(st)
            ent, mar = compute_last_token_uncertainty(logits)
            ent_map[st.branch_id] = ent
            mar_map[st.branch_id] = mar
            top1_map[st.branch_id] = int(logits[:, -1, :].argmax(dim=-1).item())
            _record_vec(st)

        if len(states) >= 2:
            dist, rel_dist, cos_sim = compute_branch_similarity(states[0].hidden, states[1].hidden)
        else:
            dist, rel_dist, cos_sim = 0.0, 0.0, 1.0

        return StepSnapshot(
            step=step_idx,
            branch_rms=rms_map,
            branch_frob=frob_map,
            branch_entropy=ent_map,
            branch_margin=mar_map,
            branch_top1_token=top1_map,
            ab_distance_frob=dist,
            ab_relative_distance=rel_dist,
            ab_cosine_similarity=cos_sim,
            intervention_applied=intervened,
            excursion_telemetry=asdict(telem) if telem is not None else None,
            policy_decision=asdict(dec) if dec is not None else None,
        )

    # Single-branch methods: S0, S1
    if method in ("S0", "S1") or branch_count == 1:
        state_a = root.clone(new_branch_id="A")
        for t in range(total_steps):
            state_a = adapter.transition(state_a)
            step_snapshots.append(_snapshot(t + 1, [state_a]))

        logits_a = adapter.readout_logits(state_a)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        pca_coords, pca_evr = _compute_pca_projection(branch_vec_history) if record_pca else ({}, [])
        return TrajectoryScheduleResult(
            method=method,
            seed=seed,
            recurrent_steps=total_steps,
            fork_after_step=0,
            branch_count=1,
            primary_state=state_a,
            branch_states={"A": state_a},
            primary_logits=logits_a,
            branch_logits={"A": logits_a},
            recurrent_block_calls=adapter.recurrent_block_calls - calls_before,
            wall_time_ms=wall_ms,
            perturbation_reports=[],
            step_snapshots=step_snapshots,
            pca_coords=pca_coords,
            pca_explained_variance=pca_evr,
        )

    # Multi-branch execution: run shared pre-fork steps (1 .. fork_after_step)
    shared_state = root.clone(new_branch_id="A")
    actual_fork = max(0, min(fork_after_step, total_steps - 1))
    for t in range(actual_fork):
        shared_state = adapter.transition(shared_state)

    branches, reports = fork_branches(
        base_state=shared_state,
        count=branch_count,
        perturbation_method=perturbation_method,
        relative_norm=perturbation_relative_norm,
        seed=seed,
        token_scope=excursion_config.token_scope,
        selected_positions=excursion_config.selected_positions,
    )
    pert_reports = [asdict(r) for r in reports]
    if actual_fork > 0:
        step_snapshots.append(_snapshot(actual_fork, branches))

    # Map controller method names to SelfDirectedController mode
    ctrl_mode_map = {
        "EXIT_2": "fixed_exit",
        "RAND_CTRL": "random_controller",
        "H_budget": "budgeted_heuristic",
        "E3_learned": "learned_ridge",
        "E3_ridge": "learned_ridge",
        "E3_mlp": "learned_mlp",
        "E3_no_exit": "learned_ridge",
        "D_random": "donor_random",
        "D_nearest": "donor_nearest",
        "D_farthest": "donor_farthest",
        "D_confidence": "donor_confidence",
        "D_learned": "donor_learned",
    }

    prev_exc_count = 0
    prev_exc_margin_delta = 0.0
    prev_top1_prob_a: Optional[float] = None
    prev_margin_a: Optional[float] = None
    if step_snapshots and "A" in step_snapshots[-1].branch_margin:
        prev_margin_a = step_snapshots[-1].branch_margin["A"]

    # Advance branches through remaining steps (actual_fork .. total_steps - 1)
    for t in range(actual_fork, total_steps):
        is_visit_step = (t == excursion_config.visit_step) or (
            method == "E1_multi_visit" and t in (excursion_config.visit_step, excursion_config.visit_step + 1)
        )
        if is_visit_step and method not in ("P0", "P1"):
            state_a = branches[0]
            donors = branches[1:]
            donor_idx = 0
            intervened = True
            active_cfg = excursion_config

            if method == "E1_last_token":
                active_cfg = replace(excursion_config, token_scope="last_token")
            elif method == "E1_norm_tight":
                active_cfg = replace(excursion_config, max_relative_norm=0.10)
            elif method == "E1_subspace":
                active_cfg = replace(excursion_config, subspace_fraction=0.50)
            elif method == "E1_gated":
                active_cfg = replace(excursion_config, gate_modulated=True)

            if method in ctrl_mode_map:
                from tebdil.controller.learned_policy import SelfDirectedController

                allow_exit_flag = method not in ("E3_no_exit", "D_random", "D_nearest", "D_farthest", "D_confidence", "D_learned")
                sd_ctrl = SelfDirectedController(
                    mode=ctrl_mode_map[method],
                    checkpoint=controller_checkpoint,
                    min_relative_divergence=policy_config.min_relative_divergence,
                    max_relative_divergence=policy_config.max_relative_divergence,
                    rms_instability_limit=policy_config.rms_Instability_limit,
                    e2_max_margin_trigger=policy_config.max_margin_trigger,
                    e2_min_entropy_trigger=policy_config.min_entropy_trigger,
                    allow_exit=allow_exit_flag,
                )
                budget_rem = max(1, (total_steps - t) * len(branches))
                c_dec = sd_ctrl.decide(
                    adapter=adapter,
                    state_a=state_a,
                    donors=donors,
                    total_steps=total_steps,
                    budget_remaining=budget_rem,
                    base_alpha=active_cfg.alpha,
                    base_beta=active_cfg.beta,
                    base_scope=active_cfg.token_scope,
                    max_relative_norm=active_cfg.max_relative_norm,
                    prev_excursions=prev_exc_count,
                    prev_excursion_margin_delta=prev_exc_margin_delta,
                    prev_top1_prob_a=prev_top1_prob_a,
                    prev_margin_a=prev_margin_a,
                    seed=seed,
                )
                pol_decision = c_dec
                ctrl_calls += sd_ctrl.controller_calls
                ctrl_overhead_ms += c_dec.controller_overhead_ms
                donor_idx = c_dec.donor_index

                if c_dec.action_type == "EXIT":
                    exited_step = t
                    if step_snapshots:
                        step_snapshots[-1].policy_decision = asdict(c_dec)
                    else:
                        step_snapshots.append(_snapshot(t, branches, intervened=False, dec=c_dec))
                    break
                elif c_dec.action_type != "EXCURSION":
                    intervened = False
                else:
                    active_cfg = replace(
                        active_cfg,
                        alpha=c_dec.adaptive_alpha,
                        beta=c_dec.adaptive_beta,
                        token_scope=c_dec.token_scope,
                    )
            elif method == "E2":
                policy = AdaptiveExcursionPolicy(policy_config)
                budget_rem = max(1, (total_steps - t) * len(branches))
                t_pol0 = time.perf_counter()
                pol_decision = policy.decide(
                    adapter=adapter,
                    state_a=state_a,
                    donors=donors,
                    budget_remaining=budget_rem,
                    base_alpha=excursion_config.alpha,
                    base_beta=excursion_config.beta,
                    seed=seed,
                )
                ctrl_calls += 1
                ctrl_overhead_ms += (time.perf_counter() - t_pol0) * 1000.0
                donor_idx = pol_decision.donor_index
                if pol_decision.action != "visit":
                    intervened = False
                else:
                    active_cfg = replace(
                        excursion_config,
                        alpha=pol_decision.adaptive_alpha,
                        beta=pol_decision.adaptive_beta,
                    )
            elif excursion_config.donor_strategy == "random" and len(donors) > 1:
                gen = torch.Generator(device="cpu").manual_seed(int(seed + t))
                donor_idx = int(torch.randint(0, len(donors), (1,), generator=gen).item())
            elif excursion_config.donor_strategy == "diversity" and len(donors) > 1:
                dists = [compute_branch_similarity(state_a.hidden, d.hidden)[1] for d in donors]
                donor_idx = int(np.argmax(dists))

            if not intervened:
                branches = [adapter.transition(b) for b in branches]
                snap_now = _snapshot(t + 1, branches, intervened=False, dec=pol_decision)
                step_snapshots.append(snap_now)
                prev_margin_a = snap_now.branch_margin.get("A", prev_margin_a)
                continue

            pre_mar_a = step_snapshots[-1].branch_margin.get("A", 0.0) if step_snapshots else 0.0
            selected_donor = donors[donor_idx]
            if method == "M0":
                next_a, next_donor, exc_telemetry = execute_direct_state_mixing_step(
                    adapter, state_a, selected_donor, active_cfg
                )
            elif method in ("M1", "E1_no_curv"):
                if method == "M1":
                    next_a, next_donor, exc_telemetry = execute_update_mixing_step(
                        adapter, state_a, selected_donor, active_cfg
                    )
                else:
                    next_a, next_donor, exc_telemetry = execute_excursion_step(
                        adapter, state_a, selected_donor, active_cfg, ablate_curvature=True
                    )
            elif method == "E1_rand_curv":
                next_a, next_donor, exc_telemetry = execute_excursion_step(
                    adapter,
                    state_a,
                    selected_donor,
                    active_cfg,
                    random_curvature_seed=seed + 31337,
                )
            elif method == "R0":
                next_a, next_donor, exc_telemetry = execute_norm_matched_random_perturbation_step(
                    adapter, state_a, selected_donor, active_cfg, seed=seed
                )
            elif method == "C0":
                next_a, next_donor, exc_telemetry = execute_visit_without_return_step(
                    adapter, state_a, selected_donor, active_cfg
                )
            elif method == "E0":
                next_a, next_donor, exc_telemetry = execute_random_direction_excursion_step(
                    adapter, state_a, selected_donor, active_cfg, seed=seed
                )
            else:  # E1, E2, E3_*, D_*, E1_*
                next_a, next_donor, exc_telemetry = execute_excursion_step(
                    adapter, state_a, selected_donor, active_cfg
                )

            new_branches = [next_a]
            for idx, d_state in enumerate(donors):
                if idx == donor_idx:
                    new_branches.append(next_donor)
                else:
                    new_branches.append(adapter.transition(d_state))
            branches = new_branches
            snap_now = _snapshot(
                t + 1, branches, intervened=True, telem=exc_telemetry, dec=pol_decision
            )
            step_snapshots.append(snap_now)
            prev_exc_count += 1
            post_mar_a = snap_now.branch_margin.get("A", pre_mar_a)
            prev_exc_margin_delta = post_mar_a - pre_mar_a
            prev_margin_a = post_mar_a
        else:
            branches = [adapter.transition(b) for b in branches]
            snap_now = _snapshot(t + 1, branches)
            step_snapshots.append(snap_now)
            prev_margin_a = snap_now.branch_margin.get("A", prev_margin_a)

    branch_states = {b.branch_id: b for b in branches}
    branch_logits = {b.branch_id: adapter.readout_logits(b) for b in branches}

    # For P0, select the branch with highest top1-top2 margin (label-free confidence selection)
    if method == "P0":
        best_b = max(
            branches,
            key=lambda b: compute_last_token_uncertainty(branch_logits[b.branch_id])[1],
        )
        primary_state = best_b
        primary_logits = branch_logits[best_b.branch_id]
    else:
        primary_state = branches[0]
        primary_logits = branch_logits[branches[0].branch_id]

    wall_ms = (time.perf_counter() - t0) * 1000.0
    pca_coords, pca_evr = _compute_pca_projection(branch_vec_history) if record_pca else ({}, [])

    return TrajectoryScheduleResult(
        method=method,
        seed=seed,
        recurrent_steps=total_steps,
        fork_after_step=actual_fork,
        branch_count=branch_count,
        primary_state=primary_state,
        branch_states=branch_states,
        primary_logits=primary_logits,
        branch_logits=branch_logits,
        recurrent_block_calls=adapter.recurrent_block_calls - calls_before,
        wall_time_ms=wall_ms,
        perturbation_reports=pert_reports,
        step_snapshots=step_snapshots,
        excursion_telemetry=exc_telemetry,
        policy_decision=pol_decision,
        controller_calls=ctrl_calls,
        controller_overhead_ms=round(ctrl_overhead_ms, 4),
        exited_early_step=exited_step,
        pca_coords=pca_coords,
        pca_explained_variance=pca_evr,
    )
