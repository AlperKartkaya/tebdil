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
    unshaped_utility: float = 0.0
    unshaped_advantage_vs_continue: float = 0.0
    outcome_hash: str = ""


def compute_outcome_hash(
    problem_id: str,
    seed: int,
    visit_step: int,
    action_id: str,
    decoded_text: str,
    extracted_answer: str,
    is_correct: bool,
    gold_logprob: float,
    recurrent_block_calls: int,
) -> str:
    """Compute immutable SHA-256 hash over raw counterfactual execution outcome fields."""
    payload = {
        "problem_id": str(problem_id),
        "seed": int(seed),
        "visit_step": int(visit_step),
        "action_id": str(action_id),
        "decoded_text": str(decoded_text),
        "extracted_answer": str(extracted_answer),
        "is_correct": bool(is_correct),
        "gold_logprob": round(float(gold_logprob), 6),
        "recurrent_block_calls": int(recurrent_block_calls),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_counterfactual_outcomes_unmodified(records: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Verify that every action outcome in `records` matches its immutable `outcome_hash`."""
    errors: List[str] = []
    for idx, rec in enumerate(records):
        pid = str(rec.get("problem_id", f"rec_{idx}"))
        seed = int(rec.get("seed", 42))
        vstep = int(rec.get("visit_step", 2))
        actions = rec.get("actions", {})
        for aid, a_data in actions.items():
            stored_hash = a_data.get("outcome_hash", "")
            if not stored_hash:
                errors.append(f"{pid}@seed={seed}@step={vstep}:{aid} missing outcome_hash")
                continue
            expected_hash = compute_outcome_hash(
                problem_id=pid,
                seed=seed,
                visit_step=vstep,
                action_id=str(a_data.get("action_id", aid)),
                decoded_text=str(a_data.get("decoded_text", "")),
                extracted_answer=str(a_data.get("extracted_answer", "")),
                is_correct=bool(a_data.get("is_correct", False)),
                gold_logprob=float(a_data.get("gold_logprob", 0.0)),
                recurrent_block_calls=int(a_data.get("recurrent_block_calls", 0)),
            )
            if stored_hash != expected_hash:
                errors.append(
                    f"{pid}@seed={seed}@step={vstep}:{aid} outcome_hash mismatch "
                    f"(stored={stored_hash[:12]} != expected={expected_hash[:12]})"
                )
    return len(errors) == 0, errors


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
    shaped: bool = True,
) -> Tuple[float, float, float, float]:
    """Compute (task_reward, compute_cost_norm, utility, advantage_vs_continue).

    Uses verifiable task accuracy, gold log-probability change, state convergence/uncertainty
    (when `shaped=True`), and exact recurrent call cost.
    """
    acc_val = 1.0 if is_correct else 0.0
    cont_acc = 1.0 if continue_is_correct else 0.0
    delta_lp = float(gold_logprob) - float(continue_gold_logprob)
    shaped_lp = math.tanh(max(-4.0, min(4.0, delta_lp)))

    # Penalize any degradation where CONTINUE was right and intervention became wrong
    deg_pen = degradation_penalty if (continue_is_correct and not is_correct) else 0.0

    state_shaping = 0.0
    if shaped:
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
        and problem.template_id not in (
            "logic_bool_xor_not_v1",
            "logic_bool_or_xor_test",
            "logic_mux_branch_fresh",
        )
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
                shaped=True,
            )
            _, _, u_unshaped, adv_unshaped = compute_action_utility(
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
                shaped=False,
            )
            ohash = compute_outcome_hash(
                problem_id=problem.problem_id,
                seed=seed,
                visit_step=visit_step,
                action_id=act.action_id,
                decoded_text=cont_text,
                extracted_answer=cont_val.extracted_answer,
                is_correct=cont_val.is_valid,
                gold_logprob=round(cont_lp, 6),
                recurrent_block_calls=cont_total_calls,
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
                unshaped_utility=u_unshaped,
                unshaped_advantage_vs_continue=adv_unshaped,
                outcome_hash=ohash,
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
                shaped=True,
            )
            _, _, u_unshaped, adv_unshaped = compute_action_utility(
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
                shaped=False,
            )
            ohash = compute_outcome_hash(
                problem_id=problem.problem_id,
                seed=seed,
                visit_step=visit_step,
                action_id=act.action_id,
                decoded_text=exit_text,
                extracted_answer=exit_val.extracted_answer,
                is_correct=exit_val.is_valid,
                gold_logprob=round(exit_lp, 6),
                recurrent_block_calls=exit_total_calls,
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
                unshaped_utility=u_unshaped,
                unshaped_advantage_vs_continue=adv_unshaped,
                outcome_hash=ohash,
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
                shaped=True,
            )
            _, _, u_unshaped, adv_unshaped = compute_action_utility(
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
                shaped=False,
            )
            ohash = compute_outcome_hash(
                problem_id=problem.problem_id,
                seed=seed,
                visit_step=visit_step,
                action_id=act.action_id,
                decoded_text=exc_text,
                extracted_answer=exc_val.extracted_answer,
                is_correct=exc_val.is_valid,
                gold_logprob=round(exc_lp, 6),
                recurrent_block_calls=exc_total_calls,
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
                unshaped_utility=u_unshaped,
                unshaped_advantage_vs_continue=adv_unshaped,
                outcome_hash=ohash,
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
        "utility_version": "v2_shaped_and_unshaped",
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
    visit_steps: Optional[Sequence[int]] = None,
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
    """Collect real model counterfactual trajectories on `split` (must not be `test` or `fresh_test`)."""
    if split in ("test", "fresh_test"):
        raise ValueError(
            f"Counterfactual training dataset collection is prohibited on held-out split '{split}'."
        )

    active_visit_steps = [int(v) for v in visit_steps] if visit_steps else [int(visit_step)]

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
            for vstep in active_visit_steps:
                rec = evaluate_counterfactual_state(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    problem=prob,
                    seed=seed,
                    recurrent_steps=recurrent_steps,
                    fork_after_step=fork_after_step,
                    visit_step=vstep,
                    branch_count=branch_count,
                    perturbation_relative_norm=perturbation_relative_norm,
                    alpha_beta_grid=alpha_beta_grid,
                    token_scopes=token_scopes,
                )
                records.append(rec)

    elapsed_s = time.perf_counter() - t0
    total_calls = adapter.recurrent_block_calls - calls_start
    records_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    dataset_payload = {
        "split": split,
        "task_family": task_family,
        "num_problems": num_problems,
        "seeds": list(seeds),
        "visit_steps": active_visit_steps,
        "num_states": len(records),
        "branch_count": branch_count,
        "alpha_beta_grid": [list(p) for p in alpha_beta_grid],
        "token_scopes": list(token_scopes),
        "utility_version": "v2_shaped_and_unshaped",
        "records_sha256": records_sha256,
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


def _get_action_utility_and_advantage(
    rec: Dict[str, Any],
    a_data: Dict[str, Any],
    use_shaped_utility: bool = True,
) -> Tuple[float, float]:
    """Retrieve (utility, advantage_vs_continue) for shaped or unshaped objective without mutating raw outcomes."""
    if use_shaped_utility:
        return float(a_data["utility"]), float(a_data["advantage_vs_continue"])
    if "unshaped_advantage_vs_continue" in a_data and "unshaped_utility" in a_data:
        u_val = float(a_data["unshaped_utility"])
        adv_val = float(a_data["unshaped_advantage_vs_continue"])
        if u_val != 0.0 or adv_val != 0.0 or a_data.get("action_type") == "CONTINUE":
            return u_val, adv_val
    cont_data = rec["actions"]["CONTINUE"]
    sf = rec.get("state_features", {})
    _, _, u_un, adv_un = compute_action_utility(
        is_correct=bool(a_data["is_correct"]),
        gold_logprob=float(a_data["gold_logprob"]),
        continue_is_correct=bool(cont_data["is_correct"]),
        continue_gold_logprob=float(cont_data["gold_logprob"]),
        recurrent_calls=int(a_data["recurrent_block_calls"]),
        continue_calls=int(cont_data["recurrent_block_calls"]),
        action_type=str(a_data["action_type"]),
        margin_a=float(sf.get("margin_a", 0.5)),
        entropy_a=float(sf.get("entropy_a", 1.5)),
        alpha=float(a_data.get("alpha", 0.65)),
        shaped=False,
    )
    return u_un, adv_un


def _build_training_matrices(
    records: Sequence[Dict[str, Any]],
    use_shaped_utility: bool = True,
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
        state_utils: Dict[str, float] = {}

        for aid, a_data in actions_map.items():
            vec = featurize_state_action(sf, a_data)
            u_val, adv = _get_action_utility_and_advantage(
                rec, a_data, use_shaped_utility=use_shaped_utility
            )
            state_utils[aid] = u_val
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

        if use_shaped_utility and "pairwise_preferences" in rec:
            for p in rec.get("pairwise_preferences", []):
                i_pref = state_row_indices.get(p["preferred"])
                i_rej = state_row_indices.get(p["rejected"])
                if i_pref is not None and i_rej is not None:
                    pref_pairs.append((i_pref, i_rej, float(p["utility_margin"])))
        else:
            for aid_1, u_1 in state_utils.items():
                for aid_2, u_2 in state_utils.items():
                    if u_1 - u_2 > 0.02:
                        pref_pairs.append(
                            (state_row_indices[aid_1], state_row_indices[aid_2], round(u_1 - u_2, 6))
                        )

    return (
        np.asarray(rows_x, dtype=np.float64),
        np.asarray(rows_y, dtype=np.float64),
        np.asarray(rows_w, dtype=np.float64),
        pref_pairs,
    )


def _calibrate_thresholds_on_records(
    ckpt: LearnedControllerCheckpoint,
    records: Sequence[Dict[str, Any]],
    use_heuristic_gates: bool = True,
    use_shaped_utility: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """Calibrate (exit_min_advantage, excursion_min_advantage) strictly on validation counterfactuals.

    Does NOT apply any artificial action-diversity bonus.
    """
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
            if a_data["action_type"] == "EXIT":
                if (not use_heuristic_gates) or float(sf.get("margin_a", 0.0)) >= 0.72:
                    exit_raw_vals.append(adv)
            elif a_data["action_type"] == "EXCURSION":
                if (not use_heuristic_gates) or (
                    float(sf.get("margin_a", 1.0)) <= 0.70
                    and float(sf.get("entropy_a", 0.0)) >= 1.20
                ):
                    exc_raw_vals.append(adv)

    quantiles = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    exit_cands = sorted(
        set(
            [-0.02, -0.01, 0.0, 0.01, 0.05]
            + ([float(np.quantile(exit_raw_vals, q)) for q in quantiles] if exit_raw_vals else [])
        )
    )
    exc_cands = sorted(
        set(
            [-0.02, -0.01, 0.0, 0.01, 0.05]
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
                if use_heuristic_gates:
                    allow_exit_rec = float(sf.get("margin_a", 0.0)) >= 0.72
                    allow_exc_rec = (
                        float(sf.get("margin_a", 1.0)) <= 0.70
                        and float(sf.get("entropy_a", 0.0)) >= 1.20
                    )
                else:
                    allow_exit_rec = True
                    allow_exc_rec = True

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
                u_val, _ = _get_action_utility_and_advantage(
                    rec, chosen_out, use_shaped_utility=use_shaped_utility
                )
                total_u += u_val
                total_acc += int(bool(chosen_out["is_correct"]))
                total_calls += int(chosen_out["recurrent_block_calls"])

            n = max(1, len(records))
            mean_u = total_u / n
            mean_acc = total_acc / n
            score = mean_u + 0.30 * mean_acc
            if score > best_score:
                best_score = score
                best_pair = (round(float(ex_th), 6), round(float(exc_th), 6))
                best_stats = {
                    "mean_counterfactual_utility": round(mean_u, 6),
                    "counterfactual_accuracy": round(mean_acc, 4),
                    "mean_recurrent_calls": round(total_calls / n, 2),
                    "action_counts": counts,
                    "diversity_bonus_applied": False,
                }

    return best_pair[0], best_pair[1], best_stats


def _hash_arrays(arrays: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a64 = np.ascontiguousarray(np.round(arr.astype(np.float64), 6))
        h.update(a64.tobytes())
    return h.hexdigest()


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
    use_heuristic_gates: bool = True,
    use_shaped_utility: bool = True,
    seed: int = 42,
) -> LearnedControllerCheckpoint:
    """Train a Ridge or 2-layer MLP action-advantage controller on counterfactual data."""
    for r in train_records:
        if r.get("split") in ("test", "fresh_test"):
            raise ValueError(
                f"Data leakage guard triggered: '{r.get('split')}' split record found in train_records!"
            )
    if val_records:
        for r in val_records:
            if r.get("split") in ("test", "fresh_test"):
                raise ValueError(
                    f"Data leakage guard triggered: '{r.get('split')}' split record found in val_records!"
                )

    # Verify outcome hashes if present on records
    has_hashes = any(
        bool(a.get("outcome_hash"))
        for r in train_records
        for a in r.get("actions", {}).values()
    )
    if has_hashes:
        ok_tr, err_tr = verify_counterfactual_outcomes_unmodified(train_records)
        if not ok_tr:
            raise ValueError(f"Counterfactual train_records failed outcome hash check: {err_tr[:3]}")
        if val_records:
            ok_va, err_va = verify_counterfactual_outcomes_unmodified(val_records)
            if not ok_va:
                raise ValueError(f"Counterfactual val_records failed outcome hash check: {err_va[:3]}")

    utility_version = "v2_shaped" if use_shaped_utility else "v2_unshaped"
    x_mat, y_vec, w_vec, pref_pairs = _build_training_matrices(
        train_records, use_shaped_utility=use_shaped_utility
    )
    d_dim = x_mat.shape[1]

    # Scale features by root-mean-square without shifting 0.0 so CONTINUE remains anchored at 0.0
    feat_mean = np.zeros(d_dim, dtype=np.float64)
    feat_std = np.ones(d_dim, dtype=np.float64)
    for col in range(4, d_dim):
        col_vals = x_mat[:, col]
        rms_col = float(np.sqrt(np.mean(col_vals ** 2)))
        feat_std[col] = rms_col if rms_col > 1e-6 else 1.0

    x_norm = (x_mat - feat_mean) / feat_std

    if val_records:
        x_val_mat, y_val_vec, w_val_vec, _ = _build_training_matrices(
            val_records, use_shaped_utility=use_shaped_utility
        )
        x_val_norm = (x_val_mat - feat_mean) / feat_std
    else:
        x_val_mat = None
        y_val_vec = None
        w_val_vec = None
        x_val_norm = None

    train_loss_curve: List[float] = []
    val_loss_curve: List[float] = []
    init_param_hash = ""
    final_param_hash = ""
    optimizer_config: Dict[str, Any] = {}

    if model_type == "ridge":
        # Weighted Ridge regression + pairwise preference gradient refinement
        sqrt_w = np.sqrt(w_vec)[:, None]
        xw = x_norm * sqrt_w
        yw = y_vec * np.sqrt(w_vec)
        reg_eye = np.eye(d_dim, dtype=np.float64) * l2_reg
        reg_eye[0, 0] = 1e-6  # Do not heavily penalize bias
        weights = np.linalg.solve(xw.T @ xw + reg_eye, xw.T @ yw)
        init_param_hash = _hash_arrays([weights])

        def _ridge_loss(x_in: np.ndarray, y_in: np.ndarray, w_in: np.ndarray, w_param: np.ndarray) -> float:
            sc = x_in @ w_param
            return float(np.mean(w_in * (sc - y_in) ** 2))

        train_loss_curve.append(round(_ridge_loss(x_norm, y_vec, w_vec, weights), 6))
        if x_val_norm is not None and y_val_vec is not None and w_val_vec is not None:
            val_loss_curve.append(round(_ridge_loss(x_val_norm, y_val_vec, w_val_vec, weights), 6))

        # Refine with 80 steps of pairwise Bradley-Terry ranking loss on preference pairs
        if pref_pairs:
            for step_i in range(80):
                scores = x_norm @ weights
                grad = ((scores - y_vec) * w_vec) @ x_norm / len(y_vec) + (l2_reg / len(y_vec)) * weights
                for i_pref, i_rej, margin in pref_pairs:
                    diff = scores[i_pref] - scores[i_rej]
                    sig = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, diff))))
                    weight_pair = min(2.0, margin * 4.0)
                    grad -= (0.25 * weight_pair * sig / len(pref_pairs)) * (x_norm[i_pref] - x_norm[i_rej])
                weights -= 0.05 * grad
                if (step_i + 1) % 20 == 0 or step_i == 79:
                    train_loss_curve.append(round(_ridge_loss(x_norm, y_vec, w_vec, weights), 6))
                    if x_val_norm is not None and y_val_vec is not None and w_val_vec is not None:
                        val_loss_curve.append(
                            round(_ridge_loss(x_val_norm, y_val_vec, w_val_vec, weights), 6)
                        )

        final_param_hash = _hash_arrays([weights])
        optimizer_config = {
            "solver": "closed_form_weighted_ridge_plus_bt_gd",
            "l2_reg": l2_reg,
            "bt_steps": 80 if pref_pairs else 0,
            "bt_lr": 0.05,
        }

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
            use_heuristic_gates=use_heuristic_gates,
            utility_version=utility_version,
        )
    elif model_type == "mlp":
        torch.manual_seed(seed)
        x_t = torch.tensor(x_norm, dtype=torch.float32)
        y_t = torch.tensor(y_vec, dtype=torch.float32)
        w_t = torch.tensor(w_vec, dtype=torch.float32)

        lin1 = torch.nn.Linear(d_dim, mlp_hidden_dim)
        lin2 = torch.nn.Linear(mlp_hidden_dim, 1)
        init_param_hash = _hash_arrays(
            [
                lin1.weight.detach().cpu().numpy().T,
                lin1.bias.detach().cpu().numpy(),
                lin2.weight.detach().cpu().numpy().reshape(-1),
                lin2.bias.detach().cpu().numpy(),
            ]
        )
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

        if x_val_norm is not None and y_val_vec is not None and w_val_vec is not None:
            x_val_t = torch.tensor(x_val_norm, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_vec, dtype=torch.float32)
            w_val_t = torch.tensor(w_val_vec, dtype=torch.float32)
        else:
            x_val_t = None
            y_val_t = None
            w_val_t = None

        for ep in range(mlp_epochs):
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

            if ep == 0 or (ep + 1) % 50 == 0 or ep == mlp_epochs - 1:
                train_loss_curve.append(round(float(loss.detach().cpu().item()), 6))
                if x_val_t is not None and y_val_t is not None and w_val_t is not None:
                    with torch.no_grad():
                        val_pred = lin2(torch.tanh(lin1(x_val_t))).squeeze(-1)
                        val_mse = torch.mean(w_val_t * (val_pred - y_val_t) ** 2)
                        val_loss_curve.append(round(float(val_mse.cpu().item()), 6))

        w1_np = lin1.weight.detach().cpu().numpy().T  # [D, H]
        b1_np = lin1.bias.detach().cpu().numpy()  # [H]
        w2_np = lin2.weight.detach().cpu().numpy().reshape(-1)  # [H]
        b2_val = float(lin2.bias.detach().cpu().item())
        final_param_hash = _hash_arrays([w1_np, b1_np, w2_np, np.asarray([b2_val])])
        optimizer_config = {
            "optimizer": "AdamW",
            "lr": mlp_lr,
            "weight_decay": l2_reg * 1e-2,
            "epochs": mlp_epochs,
            "hidden_dim": mlp_hidden_dim,
            "seed": seed,
        }

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
            use_heuristic_gates=use_heuristic_gates,
            utility_version=utility_version,
        )
    else:
        raise ValueError(f"Unsupported controller model_type: {model_type}")

    # Strictly calibrate thresholds on validation records when provided (no train+val pooling)
    if val_records and len(val_records) > 0:
        calib_pool = list(val_records)
        calib_split_name = "validation"
    else:
        calib_pool = list(train_records)
        calib_split_name = "train_fallback"

    exit_th, exc_th, calib_stats = _calibrate_thresholds_on_records(
        ckpt,
        calib_pool,
        use_heuristic_gates=use_heuristic_gates,
        use_shaped_utility=use_shaped_utility,
    )
    ckpt.exit_min_advantage = exit_th
    ckpt.excursion_min_advantage = exc_th

    # Compute training and validation fit diagnostics
    preds = np.asarray([ckpt.predict_raw_score(row) for row in x_mat], dtype=np.float64)
    mse = float(np.mean((preds - y_vec) ** 2))
    pair_acc = 1.0
    if pref_pairs:
        correct_pairs = sum(1 for i_p, i_r, _ in pref_pairs if preds[i_p] > preds[i_r])
        pair_acc = correct_pairs / float(len(pref_pairs))

    val_mse_final = None
    if x_val_mat is not None and y_val_vec is not None and len(y_val_vec) > 0:
        val_preds = np.asarray([ckpt.predict_raw_score(row) for row in x_val_mat], dtype=np.float64)
        val_mse_final = round(float(np.mean((val_preds - y_val_vec) ** 2)), 6)

    # Verify JSON round-trip reload equivalence
    reloaded = LearnedControllerCheckpoint(**json.loads(json.dumps(asdict(ckpt))))
    reloaded_preds = np.asarray([reloaded.predict_raw_score(row) for row in x_mat], dtype=np.float64)
    reload_max_abs_diff = float(np.max(np.abs(preds - reloaded_preds))) if len(preds) > 0 else 0.0

    dataset_sha256 = hashlib.sha256(
        json.dumps(
            {
                "train": list(train_records),
                "val": list(val_records) if val_records else [],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    ckpt.training_metadata = {
        "num_train_states": len(train_records),
        "num_val_states": len(val_records) if val_records else 0,
        "calibration_split": calib_split_name,
        "num_state_action_rows": int(len(y_vec)),
        "num_preference_pairs": len(pref_pairs),
        "utility_version": utility_version,
        "use_heuristic_gates": use_heuristic_gates,
        "init_param_hash": init_param_hash,
        "final_param_hash": final_param_hash,
        "param_changed_from_init": init_param_hash != final_param_hash,
        "optimizer_config": optimizer_config,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "train_mse": round(mse, 6),
        "val_mse": val_mse_final,
        "pairwise_ranking_accuracy": round(pair_acc, 4),
        "reload_max_abs_diff": round(reload_max_abs_diff, 10),
        "dataset_sha256": dataset_sha256,
        "code_sha256": code_sha256,
        "calibration_summary": calib_stats,
    }
    return ckpt
