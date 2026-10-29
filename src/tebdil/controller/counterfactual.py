"""Counterfactual trajectory collection and offline controller training for TEBDIL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor

from tebdil.controller.action_space import (
    ControllerAction,
    build_discrete_action_space,
    make_continue_action,
    make_excursion_action,
    make_exit_action,
)
from tebdil.controller.features import (
    FEATURE_SCHEMA,
    StateFeatures,
    extract_state_features,
    featurize_state_action,
)
from tebdil.controller.learned_policy import LearnedControllerCheckpoint
from tebdil.core.branching import fork_branches
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.policies import compute_last_token_uncertainty
from tebdil.core.state import BranchState, ExcursionConfig, RecurrentAdapter
from tebdil.evaluation.metrics import compute_token_logprob
from tebdil.tasks import TaskProblem, generate_task_suite, validate_problem_response


@dataclass
class ActionOutcomeRecord:
    action_id: str
    action_type: str
    donor_index: int
    alpha: float
    beta: float
    token_scope: str
    decoded_text: str
    extracted_answer: str
    is_correct: bool
    gold_logprob: float
    delta_gold_logprob_vs_continue: float
    recurrent_block_calls: int
    delta_calls_vs_continue: int
    curvature_frob: float
    task_reward: float
    compute_cost_normalized: float
    utility: float
    advantage_vs_continue: float


def compute_action_utility(
    is_correct: bool,
    gold_logprob: float,
    continue_is_correct: bool,
    continue_gold_logprob: float,
    recurrent_calls: int,
    continue_calls: int,
    action_type: str = "CONTINUE",
    compute_penalty: float = 0.05,
    logprob_weight: float = 0.30,
    degradation_penalty: float = 0.50,
    ref_calls: float = 7.0,
    margin_a: float = 0.50,
    entropy_a: float = 1.50,
    alpha: float = 0.65,
) -> Tuple[float, float, float, float]:
    """Compute (task_reward, compute_cost_norm, utility, advantage_vs_continue).

    Uses verifiable task accuracy, gold log-probability change, state convergence/uncertainty,
    and exact recurrent call cost.
    """
    acc_val = 1.0 if is_correct else 0.0
    cont_acc = 1.0 if continue_is_correct else 0.0
    delta_lp = float(gold_logprob) - float(continue_gold_logprob)
    shaped_lp = math.tanh(max(-4.0, min(4.0, delta_lp)))

    # Penalize any degradation where CONTINUE was right and intervention became wrong
    deg_pen = degradation_penalty if (continue_is_correct and not is_correct) else 0.0

    state_shaping = 0.0
    if action_type == "EXIT":
        if margin_a < 0.72:
            exit_lp_pen = 0.20 * abs(min(0.0, delta_lp))
            state_shaping -= 0.45 * (0.72 - margin_a) + exit_lp_pen
        else:
            # Converged top-1 token at step 2: do not penalize step-2 vs step-4 logit sharpening
            if is_correct == continue_is_correct:
                shaped_lp = max(0.0, shaped_lp)
            state_shaping += 0.12 * min(2.0, margin_a)
    elif action_type == "EXCURSION":
        if margin_a > 0.70 or entropy_a < 1.20:
            state_shaping -= 0.25
        else:
            step_pref = max(0.0, 1.0 - 2.5 * abs(alpha - 0.65))
            state_shaping += 0.18 * step_pref
            if (not continue_is_correct) and is_correct:
                state_shaping += 0.35

    task_reward = acc_val + logprob_weight * shaped_lp - deg_pen + state_shaping
    cont_task_reward = cont_acc

    cost_norm = float(recurrent_calls) / max(1.0, ref_calls)
    cont_cost_norm = float(continue_calls) / max(1.0, ref_calls)

    utility = task_reward - compute_penalty * cost_norm
    cont_utility = cont_task_reward - compute_penalty * cont_cost_norm
    advantage = utility - cont_utility
    return round(task_reward, 6), round(cost_norm, 6), round(utility, 6), round(advantage, 6)


def _decode_continuation_from_step0_logits(
    adapter: RecurrentAdapter,
    tokenizer: Any,
    problem: TaskProblem,
    prompt_ids: Tensor,
    step0_logits: Tensor,
    step0_calls: int,
    recurrent_steps: int,
    max_output_tokens: int = 2,
    early_exit_step: Optional[int] = None,
    excursion_config: Optional[ExcursionConfig] = None,
    donor_index: int = 0,
    fork_after_step: int = 1,
    branch_count: int = 2,
    perturbation_method: str = "normalized_gaussian",
    perturbation_relative_norm: float = 0.25,
    seed: int = 42,
    gold_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[str, int, float]:
    """Decode continuation tokens starting from step0_logits and account for all recurrent calls."""
    device = getattr(adapter, "device", torch.device("cpu"))
    tok_0 = int(step0_logits[:, -1, :].argmax(dim=-1).item())
    generated_tokens = [tok_0]
    total_calls = int(step0_calls)
    g_ids = list(gold_token_ids) if gold_token_ids else [tok_0]
    lp_list = [compute_token_logprob(step0_logits, g_ids[0])]

    first_decoded = tokenizer.decode([tok_0])
    needs_more = (
        max_output_tokens > 1
        and "\n" not in first_decoded
        and problem.template_id not in ("logic_bool_xor_not_v1", "logic_bool_or_xor_test")
    )

    if needs_more:
        cur_ids = torch.cat(
            [prompt_ids, torch.tensor([[tok_0]], dtype=torch.long, device=device)],
            dim=1,
        )
        actual_fork = max(0, min(fork_after_step, recurrent_steps - 1))
        for tok_step in range(1, max_output_tokens):
            root_k = adapter.initialize(cur_ids)
            shared_k = root_k.clone(new_branch_id="A")
            for _ in range(actual_fork):
                shared_k = adapter.transition(shared_k)

            branches_k, _ = fork_branches(
                base_state=shared_k,
                count=branch_count,
                perturbation_method=perturbation_method,
                relative_norm=perturbation_relative_norm,
                seed=seed + tok_step * 131,
                token_scope="full",
            )
            visit_step = (
                excursion_config.visit_step
                if excursion_config is not None
                else (early_exit_step if early_exit_step is not None else 2)
            )
            steps_before = max(0, visit_step - actual_fork)
            for _ in range(steps_before):
                branches_k = [adapter.transition(b) for b in branches_k]

            st_a_k = branches_k[0]
            logits_visit_k = adapter.readout_logits(st_a_k)
            ent_k, mar_k = compute_last_token_uncertainty(logits_visit_k)

            if early_exit_step is not None and mar_k >= 0.72:
                logits_k = logits_visit_k
                total_calls += actual_fork + steps_before * branch_count
            elif (
                excursion_config is not None
                and mar_k <= 0.70
                and ent_k >= 1.20
                and len(branches_k) > 1
            ):
                d_idx = min(max(0, donor_index), len(branches_k) - 2)
                next_a_k, _, _ = execute_excursion_step(
                    adapter=adapter,
                    state_a=st_a_k,
                    state_b=branches_k[1 + d_idx],
                    config=excursion_config,
                )
                rem_k = max(0, recurrent_steps - visit_step - 1)
                for _ in range(rem_k):
                    next_a_k = adapter.transition(next_a_k)
                logits_k = adapter.readout_logits(next_a_k)
                total_calls += actual_fork + (recurrent_steps - actual_fork) * branch_count + 1
            else:
                st_fin_k = st_a_k
                for _ in range(max(0, recurrent_steps - visit_step)):
                    st_fin_k = adapter.transition(st_fin_k)
                logits_k = adapter.readout_logits(st_fin_k)
                total_calls += actual_fork + (recurrent_steps - actual_fork) * branch_count

            if tok_step < len(g_ids):
                lp_list.append(compute_token_logprob(logits_k, g_ids[tok_step]))
            tok_k = int(logits_k[:, -1, :].argmax(dim=-1).item())
            generated_tokens.append(tok_k)
            if "\n" in tokenizer.decode(generated_tokens):
                break
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[tok_k]], dtype=torch.long, device=device)],
                dim=1,
            )

    decoded_text = tokenizer.decode(generated_tokens).strip()
    mean_lp = float(sum(lp_list) / len(lp_list))
    return decoded_text, total_calls, mean_lp


def evaluate_counterfactual_state(
    adapter: RecurrentAdapter,
    tokenizer: Any,
    problem: TaskProblem,
    seed: int = 42,
    recurrent_steps: int = 4,
    fork_after_step: int = 1,
    visit_step: int = 2,
    branch_count: int = 2,
    perturbation_method: str = "normalized_gaussian",
    perturbation_relative_norm: float = 0.25,
    alpha_beta_grid: Sequence[Tuple[float, float]] = (
        (0.65, 0.65),
        (0.35, 0.50),
        (0.85, 0.85),
    ),
    token_scopes: Sequence[str] = ("full",),
    max_relative_norm: float = 0.35,
    max_output_tokens: int = 2,
    compute_penalty: float = 0.05,
    logprob_weight: float = 0.30,
) -> Dict[str, Any]:
    """Advance branches to `visit_step`, extract features z_t, and evaluate all actions in A."""
    device = getattr(adapter, "device", torch.device("cpu"))
    prompt_ids = tokenizer.encode(problem.prompt, return_tensors="pt").to(device)
    gold_token_ids = tokenizer.encode(problem.gold_answer, add_special_tokens=False)
    if not gold_token_ids:
        gold_token_ids = [0]

    # 1. Shared pre-fork steps (0 .. fork_after_step)
    root = adapter.initialize(prompt_ids)
    shared = root.clone(new_branch_id="A")
    actual_fork = max(0, min(fork_after_step, recurrent_steps - 1))
    for _ in range(actual_fork):
        shared = adapter.transition(shared)

    logits_prev = adapter.readout_logits(shared)
    _, prev_margin_a = compute_last_token_uncertainty(logits_prev)
    prev_top1_prob_a = float(torch.softmax(logits_prev[:, -1, :].float(), dim=-1).max().item())

    branches, _ = fork_branches(
        base_state=shared,
        count=branch_count,
        perturbation_method=perturbation_method,
        relative_norm=perturbation_relative_norm,
        seed=seed,
        token_scope="full",
    )

    # 2. Advance all branches to visit_step
    steps_before_visit = max(0, visit_step - actual_fork)
    for _ in range(steps_before_visit):
        branches = [adapter.transition(b) for b in branches]

    state_a_at_visit = branches[0]
    donors_at_visit = branches[1:]
    calls_to_reach_visit = actual_fork + steps_before_visit * len(branches)
    budget_rem = max(1, (recurrent_steps - visit_step) * len(branches))

    features: StateFeatures = extract_state_features(
        adapter=adapter,
        state_a=state_a_at_visit,
        donors=donors_at_visit,
        total_steps=recurrent_steps,
        budget_remaining=budget_rem,
        prev_excursions=0,
        prev_excursion_margin_delta=0.0,
        prev_top1_prob_a=prev_top1_prob_a,
        prev_margin_a=prev_margin_a,
    )

    candidate_actions = build_discrete_action_space(
        num_donors=len(donors_at_visit),
        alpha_beta_grid=alpha_beta_grid,
        token_scopes=token_scopes,
        max_relative_norm=max_relative_norm,
        allow_exit=True,
    )

    # 3. First evaluate CONTINUE (standard continuation from state_a_at_visit to recurrent_steps)
    st_cont = state_a_at_visit.clone()
    rem_steps = max(0, recurrent_steps - visit_step)
    for _ in range(rem_steps):
        st_cont = adapter.transition(st_cont)
    cont_logits = adapter.readout_logits(st_cont)
    cont_step0_calls = calls_to_reach_visit + rem_steps * len(branches)
    cont_text, cont_total_calls, cont_lp = _decode_continuation_from_step0_logits(
        adapter=adapter,
        tokenizer=tokenizer,
        problem=problem,
        prompt_ids=prompt_ids,
        step0_logits=cont_logits,
        step0_calls=cont_step0_calls,
        recurrent_steps=recurrent_steps,
        max_output_tokens=max_output_tokens,
        fork_after_step=fork_after_step,
        branch_count=branch_count,
        perturbation_method=perturbation_method,
        perturbation_relative_norm=perturbation_relative_norm,
        seed=seed,
        gold_token_ids=gold_token_ids,
    )
    cont_val = validate_problem_response(problem, cont_text)

    action_outcomes: Dict[str, ActionOutcomeRecord] = {}
    pairwise_preferences: List[Dict[str, Any]] = []

    for act in candidate_actions:
        if act.action_type == "CONTINUE":
            tr, cn, util, adv = compute_action_utility(
                is_correct=cont_val.is_valid,
                gold_logprob=cont_lp,
                continue_is_correct=cont_val.is_valid,
                continue_gold_logprob=cont_lp,
                recurrent_calls=cont_total_calls,
                continue_calls=cont_total_calls,
                action_type="CONTINUE",
                compute_penalty=compute_penalty,
                logprob_weight=logprob_weight,
                margin_a=features.margin_a,
                entropy_a=features.entropy_a,
            )
            action_outcomes[act.action_id] = ActionOutcomeRecord(
                action_id=act.action_id,
                action_type=act.action_type,
                donor_index=0,
                alpha=0.0,
                beta=0.0,
                token_scope="full",
                decoded_text=cont_text,
                extracted_answer=cont_val.extracted_answer,
                is_correct=cont_val.is_valid,
                gold_logprob=round(cont_lp, 6),
                delta_gold_logprob_vs_continue=0.0,
                recurrent_block_calls=cont_total_calls,
                delta_calls_vs_continue=0,
                curvature_frob=0.0,
                task_reward=tr,
                compute_cost_normalized=cn,
                utility=util,
                advantage_vs_continue=adv,
            )
        elif act.action_type == "EXIT":
            exit_logits = adapter.readout_logits(state_a_at_visit)
            exit_step0_calls = calls_to_reach_visit
            exit_text, exit_total_calls, exit_lp = _decode_continuation_from_step0_logits(
                adapter=adapter,
                tokenizer=tokenizer,
                problem=problem,
                prompt_ids=prompt_ids,
                step0_logits=exit_logits,
                step0_calls=exit_step0_calls,
                recurrent_steps=recurrent_steps,
                max_output_tokens=max_output_tokens,
                early_exit_step=visit_step,
                fork_after_step=fork_after_step,
                branch_count=branch_count,
                perturbation_method=perturbation_method,
                perturbation_relative_norm=perturbation_relative_norm,
                seed=seed,
                gold_token_ids=gold_token_ids,
            )
            exit_val = validate_problem_response(problem, exit_text)
            tr, cn, util, adv = compute_action_utility(
                is_correct=exit_val.is_valid,
                gold_logprob=exit_lp,
                continue_is_correct=cont_val.is_valid,
                continue_gold_logprob=cont_lp,
                recurrent_calls=exit_total_calls,
                continue_calls=cont_total_calls,
                action_type="EXIT",
                compute_penalty=compute_penalty,
                logprob_weight=logprob_weight,
                margin_a=features.margin_a,
                entropy_a=features.entropy_a,
            )
            action_outcomes[act.action_id] = ActionOutcomeRecord(
                action_id=act.action_id,
                action_type=act.action_type,
                donor_index=0,
                alpha=0.0,
                beta=0.0,
                token_scope="full",
                decoded_text=exit_text,
                extracted_answer=exit_val.extracted_answer,
                is_correct=exit_val.is_valid,
                gold_logprob=round(exit_lp, 6),
                delta_gold_logprob_vs_continue=round(exit_lp - cont_lp, 6),
                recurrent_block_calls=exit_total_calls,
                delta_calls_vs_continue=exit_total_calls - cont_total_calls,
                curvature_frob=0.0,
                task_reward=tr,
                compute_cost_normalized=cn,
                utility=util,
                advantage_vs_continue=adv,
            )
        elif act.action_type == "EXCURSION":
            donor_st = donors_at_visit[act.donor_index].clone()
            exc_cfg = ExcursionConfig(
                alpha=act.alpha,
                beta=act.beta,
                max_relative_norm=act.max_relative_norm,
                visit_step=visit_step,
                token_scope=act.token_scope,
            )
            next_a, _, telem = execute_excursion_step(
                adapter=adapter,
                state_a=state_a_at_visit.clone(),
                state_b=donor_st,
                config=exc_cfg,
            )
            st_exc = next_a
            for _ in range(max(0, rem_steps - 1)):
                st_exc = adapter.transition(st_exc)
            exc_logits = adapter.readout_logits(st_exc)
            exc_step0_calls = cont_step0_calls + 1
            exc_text, exc_total_calls, exc_lp = _decode_continuation_from_step0_logits(
                adapter=adapter,
                tokenizer=tokenizer,
                problem=problem,
                prompt_ids=prompt_ids,
                step0_logits=exc_logits,
                step0_calls=exc_step0_calls,
                recurrent_steps=recurrent_steps,
                max_output_tokens=max_output_tokens,
                excursion_config=exc_cfg,
                donor_index=act.donor_index,
                fork_after_step=fork_after_step,
                branch_count=branch_count,
                perturbation_method=perturbation_method,
                perturbation_relative_norm=perturbation_relative_norm,
                seed=seed,
                gold_token_ids=gold_token_ids,
            )
            exc_val = validate_problem_response(problem, exc_text)
            tr, cn, util, adv = compute_action_utility(
                is_correct=exc_val.is_valid,
                gold_logprob=exc_lp,
                continue_is_correct=cont_val.is_valid,
                continue_gold_logprob=cont_lp,
                recurrent_calls=exc_total_calls,
                continue_calls=cont_total_calls,
                action_type="EXCURSION",
                compute_penalty=compute_penalty,
                logprob_weight=logprob_weight,
                margin_a=features.margin_a,
                entropy_a=features.entropy_a,
                alpha=act.alpha,
            )
            action_outcomes[act.action_id] = ActionOutcomeRecord(
                action_id=act.action_id,
                action_type=act.action_type,
                donor_index=act.donor_index,
                alpha=act.alpha,
                beta=act.beta,
                token_scope=act.token_scope,
                decoded_text=exc_text,
                extracted_answer=exc_val.extracted_answer,
                is_correct=exc_val.is_valid,
                gold_logprob=round(exc_lp, 6),
                delta_gold_logprob_vs_continue=round(exc_lp - cont_lp, 6),
                recurrent_block_calls=exc_total_calls,
                delta_calls_vs_continue=exc_total_calls - cont_total_calls,
                curvature_frob=round(telem.curvature_frob, 6),
                task_reward=tr,
                compute_cost_normalized=cn,
                utility=util,
                advantage_vs_continue=adv,
            )

    best_act_id = max(action_outcomes.keys(), key=lambda k: action_outcomes[k].utility)
    for aid_1, out_1 in action_outcomes.items():
        for aid_2, out_2 in action_outcomes.items():
            if out_1.utility - out_2.utility > 0.02:
                pairwise_preferences.append(
                    {
                        "preferred": aid_1,
                        "rejected": aid_2,
                        "utility_margin": round(out_1.utility - out_2.utility, 6),
                    }
                )

    return {
        "problem_id": problem.problem_id,
        "family": problem.family,
        "split": problem.split,
        "template_id": problem.template_id,
        "seed": seed,
        "visit_step": visit_step,
        "branch_count": branch_count,
        "state_features": features.to_dict(),
        "actions": {k: asdict(v) for k, v in action_outcomes.items()},
        "best_action_id": best_act_id,
        "best_action_type": action_outcomes[best_act_id].action_type,
        "continue_is_correct": cont_val.is_valid,
        "any_excursion_rescued": (not cont_val.is_valid)
        and any(v.is_correct for v in action_outcomes.values() if v.action_type == "EXCURSION"),
        "exit_preserves_correctness": (exit_rec := action_outcomes.get("EXIT")) is not None
        and (exit_rec.is_correct == cont_val.is_valid),
        "pairwise_preferences": pairwise_preferences,
    }


def collect_counterfactual_dataset(
    adapter: RecurrentAdapter,
    tokenizer: Any,
    split: str = "train",
    task_family: str = "all",
    num_problems: int = 24,
    seeds: Sequence[int] = (42,),
    recurrent_steps: int = 4,
    fork_after_step: int = 1,
    visit_step: int = 2,
    branch_count: int = 2,
    perturbation_relative_norm: float = 0.25,
    alpha_beta_grid: Sequence[Tuple[float, float]] = (
        (0.65, 0.65),
        (0.35, 0.50),
        (0.85, 0.85),
    ),
    token_scopes: Sequence[str] = ("full",),
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect real model counterfactual trajectories on `split` (must not be `test`)."""
    if split == "test":
        raise ValueError("Counterfactual training dataset collection is prohibited on the held-out 'test' split.")

    t0 = time.perf_counter()
    calls_start = adapter.recurrent_block_calls
    problems = generate_task_suite(
        family=task_family,
        split=split,
        num_problems=num_problems,
        seed=seeds[0],
    )

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        for prob in problems:
            rec = evaluate_counterfactual_state(
                adapter=adapter,
                tokenizer=tokenizer,
                problem=prob,
                seed=seed,
                recurrent_steps=recurrent_steps,
                fork_after_step=fork_after_step,
                visit_step=visit_step,
                branch_count=branch_count,
                perturbation_relative_norm=perturbation_relative_norm,
                alpha_beta_grid=alpha_beta_grid,
                token_scopes=token_scopes,
            )
            records.append(rec)

    elapsed_s = time.perf_counter() - t0
    total_calls = adapter.recurrent_block_calls - calls_start

    dataset_payload = {
        "split": split,
        "task_family": task_family,
        "num_problems": num_problems,
        "seeds": list(seeds),
        "num_states": len(records),
        "branch_count": branch_count,
        "alpha_beta_grid": [list(p) for p in alpha_beta_grid],
        "token_scopes": list(token_scopes),
        "total_recurrent_calls": total_calls,
        "runtime_seconds": round(elapsed_s, 3),
        "template_ids": sorted({r["template_id"] for r in records}),
        "records": records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(dataset_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return dataset_payload


def _build_training_matrices(
    records: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
    """Build (X, y_advantage, sample_weights, preference_pairs) from counterfactual state records."""
    rows_x: List[List[float]] = []
    rows_y: List[float] = []
    rows_w: List[float] = []
    pref_pairs: List[Tuple[int, int, float]] = []

    for rec in records:
        sf = rec["state_features"]
        actions_map = rec["actions"]
        state_row_indices: Dict[str, int] = {}

        for aid, a_data in actions_map.items():
            vec = featurize_state_action(sf, a_data)
            adv = float(a_data["advantage_vs_continue"])
            # Weight rescues and degradations higher so the controller focuses on decision-critical states
            w = 1.0
            if a_data["action_type"] == "EXCURSION":
                if (not rec["continue_is_correct"]) and a_data["is_correct"]:
                    w = 4.0  # Rescue state
                elif rec["continue_is_correct"] and (not a_data["is_correct"]):
                    w = 3.5  # Degradation state
            elif a_data["action_type"] == "EXIT":
                if rec["continue_is_correct"] != a_data["is_correct"]:
                    w = 3.0
            idx = len(rows_x)
            state_row_indices[aid] = idx
            rows_x.append(vec)
            rows_y.append(adv)
            rows_w.append(w)

        for p in rec.get("pairwise_preferences", []):
            i_pref = state_row_indices.get(p["preferred"])
            i_rej = state_row_indices.get(p["rejected"])
            if i_pref is not None and i_rej is not None:
                pref_pairs.append((i_pref, i_rej, float(p["utility_margin"])))

    return (
        np.asarray(rows_x, dtype=np.float64),
        np.asarray(rows_y, dtype=np.float64),
        np.asarray(rows_w, dtype=np.float64),
        pref_pairs,
    )


def _calibrate_thresholds_on_records(
    ckpt: LearnedControllerCheckpoint,
    records: Sequence[Dict[str, Any]],
) -> Tuple[float, float, Dict[str, Any]]:
    """Calibrate (exit_min_advantage, excursion_min_advantage) on training/validation counterfactuals."""
    exit_raw_vals: List[float] = []
    exc_raw_vals: List[float] = []
    for rec in records:
        sf = rec["state_features"]
        acts = rec["actions"]
        cont_q = ckpt.predict_raw_score(featurize_state_action(sf, acts["CONTINUE"]))
        for aid, a_data in acts.items():
            if aid == "CONTINUE":
                continue
            adv = ckpt.predict_raw_score(featurize_state_action(sf, a_data)) - cont_q
            if a_data["action_type"] == "EXIT" and float(sf.get("margin_a", 0.0)) >= 0.72:
                exit_raw_vals.append(adv)
            elif (
                a_data["action_type"] == "EXCURSION"
                and float(sf.get("margin_a", 1.0)) <= 0.70
                and float(sf.get("entropy_a", 0.0)) >= 1.20
            ):
                exc_raw_vals.append(adv)

    quantiles = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    exit_cands = sorted(
        set(
            [-0.02, -0.01, 0.0, 0.01]
            + ([float(np.quantile(exit_raw_vals, q)) for q in quantiles] if exit_raw_vals else [])
        )
    )
    exc_cands = sorted(
        set(
            [-0.02, -0.01, 0.0, 0.01]
            + ([float(np.quantile(exc_raw_vals, q)) for q in quantiles] if exc_raw_vals else [])
        )
    )

    best_score = -1e9
    best_pair = (0.0, 0.0)
    best_stats: Dict[str, Any] = {}

    for ex_th in exit_cands:
        for exc_th in exc_cands:
            total_u = 0.0
            total_acc = 0
            total_calls = 0
            counts = {"CONTINUE": 0, "EXCURSION": 0, "EXIT": 0}
            for rec in records:
                sf = rec["state_features"]
                acts = rec["actions"]
                cont_q = ckpt.predict_raw_score(featurize_state_action(sf, acts["CONTINUE"]))
                allow_exit_rec = float(sf.get("margin_a", 0.0)) >= 0.72
                allow_exc_rec = (
                    float(sf.get("margin_a", 1.0)) <= 0.70
                    and float(sf.get("entropy_a", 0.0)) >= 1.20
                )

                chosen_id = "CONTINUE"
                chosen_adj_adv = 0.0
                for aid, a_data in acts.items():
                    if aid == "CONTINUE":
                        continue
                    q_val = ckpt.predict_raw_score(featurize_state_action(sf, a_data))
                    raw_adv = q_val - cont_q
                    if a_data["action_type"] == "EXIT" and allow_exit_rec:
                        adj = raw_adv - ex_th
                        if adj > chosen_adj_adv:
                            chosen_adj_adv = adj
                            chosen_id = aid
                    elif a_data["action_type"] == "EXCURSION" and allow_exc_rec:
                        adj = raw_adv - exc_th
                        if adj > chosen_adj_adv:
                            chosen_adj_adv = adj
                            chosen_id = aid

                chosen_out = acts[chosen_id]
                counts[chosen_out["action_type"]] += 1
                total_u += float(chosen_out["utility"])
                total_acc += int(bool(chosen_out["is_correct"]))
                total_calls += int(chosen_out["recurrent_block_calls"])

            n = max(1, len(records))
            mean_u = total_u / n
            mean_acc = total_acc / n
            active_types = sum(1 for v in counts.values() if v >= max(1, int(0.08 * n)))
            diversity_bonus = 0.02 * active_types
            score = mean_u + 0.30 * mean_acc + diversity_bonus
            if score > best_score:
                best_score = score
                best_pair = (round(float(ex_th), 6), round(float(exc_th), 6))
                best_stats = {
                    "mean_counterfactual_utility": round(mean_u, 6),
                    "counterfactual_accuracy": round(mean_acc, 4),
                    "mean_recurrent_calls": round(total_calls / n, 2),
                    "action_counts": counts,
                }

    return best_pair[0], best_pair[1], best_stats


def train_learned_controller(
    train_records: Sequence[Dict[str, Any]],
    val_records: Optional[Sequence[Dict[str, Any]]] = None,
    model_type: str = "ridge",
    controller_id: str = "tebdil_ridge_v1",
    l2_reg: float = 1.5,
    mlp_hidden_dim: int = 16,
    mlp_epochs: int = 300,
    mlp_lr: float = 0.015,
    alpha_beta_grid: Sequence[Tuple[float, float]] = (
        (0.65, 0.65),
        (0.35, 0.50),
        (0.85, 0.85),
    ),
    token_scopes: Sequence[str] = ("full",),
    allow_exit: bool = True,
    seed: int = 42,
) -> LearnedControllerCheckpoint:
    """Train a Ridge or 2-layer MLP action-advantage controller on counterfactual data."""
    for r in train_records:
        if r.get("split") == "test":
            raise ValueError("Data leakage guard triggered: 'test' split record found in train_records!")
    if val_records:
        for r in val_records:
            if r.get("split") == "test":
                raise ValueError("Data leakage guard triggered: 'test' split record found in val_records!")

    x_mat, y_vec, w_vec, pref_pairs = _build_training_matrices(train_records)
    d_dim = x_mat.shape[1]

    # Scale features by root-mean-square without shifting 0.0 so CONTINUE remains anchored at 0.0
    feat_mean = np.zeros(d_dim, dtype=np.float64)
    feat_std = np.ones(d_dim, dtype=np.float64)
    for col in range(4, d_dim):
        col_vals = x_mat[:, col]
        rms_col = float(np.sqrt(np.mean(col_vals ** 2)))
        feat_std[col] = rms_col if rms_col > 1e-6 else 1.0

    x_norm = (x_mat - feat_mean) / feat_std

    if model_type == "ridge":
        # Weighted Ridge regression + pairwise preference gradient refinement
        sqrt_w = np.sqrt(w_vec)[:, None]
        xw = x_norm * sqrt_w
        yw = y_vec * np.sqrt(w_vec)
        reg_eye = np.eye(d_dim, dtype=np.float64) * l2_reg
        reg_eye[0, 0] = 1e-6  # Do not heavily penalize bias
        weights = np.linalg.solve(xw.T @ xw + reg_eye, xw.T @ yw)

        # Refine with 80 steps of pairwise Bradley-Terry ranking loss on preference pairs
        if pref_pairs:
            for _ in range(80):
                scores = x_norm @ weights
                grad = ((scores - y_vec) * w_vec) @ x_norm / len(y_vec) + (l2_reg / len(y_vec)) * weights
                for i_pref, i_rej, margin in pref_pairs:
                    diff = scores[i_pref] - scores[i_rej]
                    sig = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, diff))))
                    weight_pair = min(2.0, margin * 4.0)
                    grad -= (0.25 * weight_pair * sig / len(pref_pairs)) * (x_norm[i_pref] - x_norm[i_rej])
                weights -= 0.05 * grad

        ckpt = LearnedControllerCheckpoint(
            controller_id=controller_id,
            model_type="ridge",
            feature_schema=list(FEATURE_SCHEMA),
            feature_mean=[round(float(v), 6) for v in feat_mean],
            feature_std=[round(float(v), 6) for v in feat_std],
            weights=[round(float(v), 6) for v in weights],
            bias=0.0,
            alpha_beta_grid=[list(p) for p in alpha_beta_grid],
            token_scopes=list(token_scopes),
            allow_exit=allow_exit,
        )
    elif model_type == "mlp":
        torch.manual_seed(seed)
        x_t = torch.tensor(x_norm, dtype=torch.float32)
        y_t = torch.tensor(y_vec, dtype=torch.float32)
        w_t = torch.tensor(w_vec, dtype=torch.float32)

        lin1 = torch.nn.Linear(d_dim, mlp_hidden_dim)
        lin2 = torch.nn.Linear(mlp_hidden_dim, 1)
        opt = torch.optim.AdamW(
            list(lin1.parameters()) + list(lin2.parameters()),
            lr=mlp_lr,
            weight_decay=l2_reg * 1e-2,
        )

        if pref_pairs:
            idx_pref = torch.tensor([p[0] for p in pref_pairs], dtype=torch.long)
            idx_rej = torch.tensor([p[1] for p in pref_pairs], dtype=torch.long)
        else:
            idx_pref = None
            idx_rej = None

        for _ in range(mlp_epochs):
            opt.zero_grad()
            h = torch.tanh(lin1(x_t))
            pred = lin2(h).squeeze(-1)
            mse_loss = torch.mean(w_t * (pred - y_t) ** 2)
            if idx_pref is not None and idx_rej is not None and len(idx_pref) > 0:
                diff = pred[idx_pref] - pred[idx_rej]
                rank_loss = torch.mean(torch.nn.functional.softplus(-diff))
                loss = mse_loss + 0.15 * rank_loss
            else:
                loss = mse_loss
            loss.backward()
            opt.step()

        w1_np = lin1.weight.detach().cpu().numpy().T  # [D, H]
        b1_np = lin1.bias.detach().cpu().numpy()  # [H]
        w2_np = lin2.weight.detach().cpu().numpy().reshape(-1)  # [H]
        b2_val = float(lin2.bias.detach().cpu().item())

        ckpt = LearnedControllerCheckpoint(
            controller_id=controller_id,
            model_type="mlp",
            feature_schema=list(FEATURE_SCHEMA),
            feature_mean=[round(float(v), 6) for v in feat_mean],
            feature_std=[round(float(v), 6) for v in feat_std],
            mlp_w1=[[round(float(v), 6) for v in row] for row in w1_np],
            mlp_b1=[round(float(v), 6) for v in b1_np],
            mlp_w2=[round(float(v), 6) for v in w2_np],
            mlp_b2=round(b2_val, 6),
            alpha_beta_grid=[list(p) for p in alpha_beta_grid],
            token_scopes=list(token_scopes),
            allow_exit=allow_exit,
        )
    else:
        raise ValueError(f"Unsupported controller model_type: {model_type}")

    calib_Pool = list(train_records) + (list(val_records) if val_records else [])
    exit_th, exc_th, calib_stats = _calibrate_thresholds_on_records(ckpt, calib_Pool)
    ckpt.exit_min_advantage = exit_th
    ckpt.excursion_min_advantage = exc_th

    # Compute training fit diagnostics
    preds = np.asarray([ckpt.predict_raw_score(row) for row in x_mat], dtype=np.float64)
    mse = float(np.mean((preds - y_vec) ** 2))
    pair_acc = 1.0
    if pref_pairs:
        correct_pairs = sum(1 for i_p, i_r, _ in pref_pairs if preds[i_p] > preds[i_r])
        pair_acc = correct_pairs / float(len(pref_pairs))

    ckpt.training_metadata = {
        "num_train_states": len(train_records),
        "num_val_states": len(val_records) if val_records else 0,
        "num_state_action_rows": int(len(y_vec)),
        "num_preference_pairs": len(pref_pairs),
        "train_mse": round(mse, 6),
        "pairwise_ranking_accuracy": round(pair_acc, 4),
        "calibration_summary": calib_stats,
    }
    return ckpt
