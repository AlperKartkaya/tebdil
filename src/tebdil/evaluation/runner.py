"""Deterministic experiment runner for TEBDIL phases and comparative benchmarks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor
import yaml

from tebdil.controller.learned_policy import LearnedControllerCheckpoint
from tebdil.core.policies import AdaptivePolicyConfig, compute_last_token_uncertainty
from tebdil.core.scheduler import TrajectoryScheduleResult, run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig, RecurrentAdapter, hash_tensor
from tebdil.evaluation.metrics import compare_logits, compute_token_logprob
from tebdil.evaluation.trace import (
    create_run_manifest,
    save_run_manifest,
    write_trace_records,
)
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.models.toy_adapter import ToyRecurrentAdapter
from tebdil.tasks import TaskProblem, generate_task_suite, validate_problem_response


class ByteLevelToyTokenizer:
    """Deterministic ASCII tokenizer used when running with ToyRecurrentAdapter."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Any:
        ids = [min(ord(ch), self.vocab_size - 1) for ch in text]
        if not ids:
            ids = [0]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        chars = []
        for tid in token_ids:
            if 32 <= int(tid) < 127 or int(tid) == 10:
                chars.append(chr(int(tid)))
        return "".join(chars)


def load_adapter_and_tokenizer(
    model_cfg: Dict[str, Any],
) -> tuple[RecurrentAdapter, Any, str, str]:
    """Instantiate RecurrentAdapter and tokenizer from config dictionary."""
    repo = str(model_cfg.get("repo", "ByteDance/Ouro-1.4B-Thinking"))
    revision = str(model_cfg.get("revision", PINNED_OURO_THINKING_REV))
    if revision == "PIN_BEFORE_RUN":
        revision = PINNED_OURO_THINKING_REV

    precision = str(model_cfg.get("precision", "bf16")).lower()
    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else torch.float32
    steps = int(model_cfg.get("recurrent_steps", 4))
    readout_mode = str(model_cfg.get("readout_mode", "last_step"))

    if repo == "toy":
        adapter = ToyRecurrentAdapter(seed=int(model_cfg.get("seed", 42)))
        tokenizer = ByteLevelToyTokenizer(vocab_size=128)
        return adapter, tokenizer, "toy-looped-lm", "local-toy-v1"

    if repo == "ouro-mini":
        adapter = OuroRecurrentAdapter.from_mini_config(
            vocab_size=128,
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            intermediate_size=int(model_cfg.get("intermediate_size", 128)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 2)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 4)),
            total_ut_steps=steps,
            dtype=torch.float32,
            device="cpu",
            readout_mode=readout_mode,
        )
        tokenizer = ByteLevelToyTokenizer(vocab_size=128)
        return adapter, tokenizer, "ouro-mini-verification", "local-vendored-cb8c980b"

    from transformers import AutoTokenizer

    local_only = bool(model_cfg.get("local_files_only", True))
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo, revision=revision, local_files_only=local_only
        )
        adapter = OuroRecurrentAdapter.from_pretrained(
            repo_id=repo,
            revision=revision,
            dtype=dtype,
            device=model_cfg.get("device"),
            readout_mode=readout_mode,
            total_ut_steps=steps,
            local_files_only=local_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=False)
        adapter = OuroRecurrentAdapter.from_pretrained(
            repo_id=repo,
            revision=revision,
            dtype=dtype,
            device=model_cfg.get("device"),
            readout_mode=readout_mode,
            total_ut_steps=steps,
            local_files_only=False,
        )
    return adapter, tokenizer, repo, revision


def _sync_device(device: torch.device) -> None:
    """Synchronize accelerator device before/after wall-clock timing measurements."""
    try:
        if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _get_memory_mb(device: torch.device) -> float:
    """Return current allocated accelerator memory (or process RSS fallback) in MB."""
    try:
        if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return round(float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0), 2)
        if device.type == "cuda" and torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    import psutil

    return round(float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0), 2)


def _resolve_decision_summary(
    method: str,
    sched_res: TrajectoryScheduleResult,
    excursion_config: ExcursionConfig,
) -> Dict[str, Any]:
    """Extract canonical decision fields without falsely labeling S0/S1/P0/P1 as EXCURSION."""
    telem = sched_res.excursion_telemetry
    pol_dec = sched_res.policy_decision
    if method in ("S0", "S1"):
        return {
            "action_type": "CONTINUE",
            "action_id": "CONTINUE",
            "policy_action": "none",
            "policy_reason": "single_branch_baseline",
            "donor_index": 0,
            "selected_donor": "none",
            "effective_alpha": 0.0,
            "effective_beta": 0.0,
            "chosen_depth": int(sched_res.recurrent_steps),
            "predicted_advantage": 0.0,
            "action_scores": {},
            "rejected_actions": {},
            "token_scope": "none",
        }
    if method in ("P0", "P1"):
        return {
            "action_type": "CONTINUE",
            "action_id": "CONTINUE",
            "policy_action": "parallel_selection",
            "policy_reason": "independent_branches",
            "donor_index": 0,
            "selected_donor": "none",
            "effective_alpha": 0.0,
            "effective_beta": 0.0,
            "chosen_depth": int(sched_res.recurrent_steps),
            "predicted_advantage": 0.0,
            "action_scores": {},
            "rejected_actions": {},
            "token_scope": excursion_config.token_scope,
        }
    if pol_dec is not None:
        act_type = getattr(
            pol_dec,
            "action_type",
            "EXCURSION"
            if pol_dec.action == "visit"
            else ("EXIT" if pol_dec.action == "exit" else "CONTINUE"),
        )
        act_id = getattr(pol_dec, "action_id", pol_dec.action)
        donor_idx = int(getattr(pol_dec, "donor_index", 0))
        if telem is not None:
            sel_donor = telem.donor_branch_id
            eff_a = float(telem.effective_alpha)
            eff_b = float(telem.beta)
        elif act_type == "EXCURSION":
            sel_donor = getattr(pol_dec, "selected_donor", "B" if donor_idx == 0 else "C")
            eff_a = float(getattr(pol_dec, "effective_alpha", getattr(pol_dec, "adaptive_alpha", excursion_config.alpha)))
            eff_b = float(getattr(pol_dec, "effective_beta", getattr(pol_dec, "adaptive_beta", excursion_config.beta)))
        else:
            sel_donor = "none"
            eff_a = 0.0
            eff_b = 0.0
        chosen_depth = int(
            getattr(
                pol_dec,
                "chosen_depth",
                sched_res.exited_early_step
                if sched_res.exited_early_step is not None
                else excursion_config.visit_step,
            )
        )
        return {
            "action_type": act_type,
            "action_id": act_id,
            "policy_action": pol_dec.action,
            "policy_reason": pol_dec.reason,
            "donor_index": donor_idx,
            "selected_donor": sel_donor,
            "effective_alpha": round(eff_a, 4),
            "effective_beta": round(eff_b, 4),
            "chosen_depth": chosen_depth,
            "predicted_advantage": float(getattr(pol_dec, "predicted_advantage", 0.0)),
            "action_scores": dict(getattr(pol_dec, "action_scores", {}) or {}),
            "rejected_actions": dict(getattr(pol_dec, "rejected_actions", {}) or {}),
            "token_scope": getattr(pol_dec, "token_scope", excursion_config.token_scope),
        }
    # Fixed-schedule intervention methods (E1, M0, M1, R0, C0, E0, E1_*)
    sel_donor = telem.donor_branch_id if telem is not None else "B"
    eff_a = float(telem.effective_alpha) if telem is not None else float(excursion_config.alpha)
    eff_b = float(telem.beta) if telem is not None else float(excursion_config.beta)
    return {
        "action_type": "EXCURSION",
        "action_id": f"FIXED_{method}",
        "policy_action": "always",
        "policy_reason": "fixed_schedule",
        "donor_index": 0,
        "selected_donor": sel_donor,
        "effective_alpha": round(eff_a, 4),
        "effective_beta": round(eff_b, 4),
        "chosen_depth": int(excursion_config.visit_step),
        "predicted_advantage": 0.0,
        "action_scores": {},
        "rejected_actions": {},
        "token_scope": excursion_config.token_scope,
    }


def _build_token_ledger_entry(
    token_index: int,
    token_id: int,
    decoded_token: str,
    sched_res: TrajectoryScheduleResult,
    method: str,
    excursion_config: ExcursionConfig,
    prefix_ids: Tensor,
    branch_seed: int,
) -> Dict[str, Any]:
    """Construct an exact per-token call and decision ledger entry."""
    dec_info = _resolve_decision_summary(method, sched_res, excursion_config)
    total_c = int(sched_res.recurrent_block_calls)
    b_cnt = int(sched_res.branch_count)
    f_steps = int(sched_res.fork_after_step)
    if b_cnt <= 1 or method in ("S0", "S1"):
        shared_c = total_c
        primary_post_c = 0
        donor_c = 0
        visit_c = 0
        cont_c = 0
    else:
        shared_c = f_steps
        post_fork_snaps = max(0, len(sched_res.step_snapshots) - (1 if f_steps > 0 else 0))
        if sched_res.exited_early_step is not None and post_fork_snaps == 0:
            post_fork_snaps = max(0, int(sched_res.exited_early_step) - f_steps)
        base_branched = shared_c + post_fork_snaps * b_cnt
        visit_c = max(0, total_c - base_branched)
        donor_c = post_fork_snaps * max(0, b_cnt - 1)
        primary_post_c = max(0, total_c - shared_c - donor_c - visit_c)
        if dec_info["action_type"] == "EXCURSION":
            v_depth = int(dec_info["chosen_depth"])
            steps_up_to_exc = max(0, v_depth - f_steps + 1)
            cont_c = max(0, primary_post_c - steps_up_to_exc)
        else:
            cont_c = 0

    return {
        "token_index": int(token_index),
        "token_id": int(token_id),
        "decoded_token": str(decoded_token),
        "prefix_token_ids_sha256": hash_tensor(prefix_ids, length=64),
        "prefix_length": int(prefix_ids.shape[1]),
        "branch_seed": int(branch_seed),
        "recurrent_block_calls": total_c,
        "shared_calls": int(shared_c),
        "primary_branch_calls": int(primary_post_c),
        "donor_calls": int(donor_c),
        "visit_calls": int(visit_c),
        "continuation_calls": int(cont_c),
        "chosen_depth": int(dec_info["chosen_depth"]),
        "exited_early_step": sched_res.exited_early_step,
        "controller_action_type": dec_info["action_type"],
        "controller_action_id": dec_info["action_id"],
        "policy_action": dec_info["policy_action"],
        "policy_reason": dec_info["policy_reason"],
        "selected_donor": dec_info["selected_donor"],
        "effective_alpha": dec_info["effective_alpha"],
        "effective_beta": dec_info["effective_beta"],
        "predicted_advantage": round(float(dec_info["predicted_advantage"]), 6),
        "action_scores": dec_info["action_scores"],
        "rejected_actions": dec_info["rejected_actions"],
        "step_snapshots": [asdict(s) for s in sched_res.step_snapshots],
    }


_PARALLEL_COST_FIELDS = (
    "token_index",
    "recurrent_block_calls",
    "shared_calls",
    "primary_branch_calls",
    "donor_calls",
    "visit_calls",
    "continuation_calls",
)


def _attach_parallel_compute_provenance(
    selected_record: Dict[str, Any],
    *,
    compute_source: Dict[str, Any],
    method: str,
    selected_branch: str,
    consulted_sources: Sequence[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Attach an idealized reference budget without calling it executed selection cost."""
    source_ledger = compute_source.get("token_ledger", [])
    compute_ledger = [
        {key: entry.get(key, 0) for key in _PARALLEL_COST_FIELDS}
        for entry in source_ledger
    ]
    charged_calls = sum(int(entry["recurrent_block_calls"]) for entry in compute_ledger)
    source_calls = int(compute_source.get("recurrent_block_calls", 0))
    if not compute_ledger or charged_calls != source_calls:
        raise ValueError(
            f"Parallel compute source ledger mismatch: {charged_calls} vs {source_calls}"
        )

    consulted = [
        {
            "source_id": f"{branch}:{record.get('method', 'unknown')}:{record.get('problem_id', 'unknown')}:{record.get('seed', 'na')}",
            "branch": branch,
            "method": str(record.get("method", "unknown")),
            "problem_id": record.get("problem_id"),
            "seed": record.get("seed"),
            "recurrent_block_calls": int(record.get("recurrent_block_calls", 0)),
        }
        for branch, record in consulted_sources
    ]
    reference_source = next(
        (source for source in consulted if source["branch"] == "B"),
        None,
    )
    if reference_source is None:
        raise ValueError("Parallel reference budget requires the independent B source")

    output_source_method = str(selected_record.get("method", "unknown"))
    result = dict(selected_record)
    result.update(
        {
            "method": method,
            "derived_parallel_selection": True,
            "reference_parallel_budget_ledger": compute_ledger,
            "reference_parallel_budget_source": reference_source,
            "recurrent_block_calls_semantics": "idealized_reference_parallel_budget",
            "executed_selection_compute_verified": False,
            "selection_provenance": {
                "selected_branch": selected_branch,
                "output_source_method": output_source_method,
                "consulted_sources": consulted,
            },
            "recurrent_block_calls": charged_calls,
            "estimated_flops": compute_source.get("estimated_flops", 0.0),
            "estimated_flops_semantics": "idealized_reference_parallel_budget",
            "wall_time_ms": compute_source.get("wall_time_ms", 0.0),
            "wall_time_ms_semantics": "reference_source_only_not_total_selection_execution",
        }
    )
    return result


def _decode_method_on_problem(
    adapter: RecurrentAdapter,
    tokenizer: Any,
    problem: TaskProblem,
    method: str,
    seed: int,
    recurrent_steps: int,
    fork_after_step: int,
    branch_count: int,
    perturbation_method: str,
    perturbation_relative_norm: float,
    excursion_config: ExcursionConfig,
    policy_config: AdaptivePolicyConfig,
    controller_checkpoint: Optional[LearnedControllerCheckpoint] = None,
    max_output_tokens: int = 2,
    extra_steps_for_s1: int = 3,
    select_branch_b: bool = False,
    select_branch_id: Optional[str] = None,
    prompt_format: str = "raw",
) -> Dict[str, Any]:
    """Run a single method on a single problem and decode its short continuation."""
    device = getattr(adapter, "device", torch.device("cpu"))
    _sync_device(device)
    t_wall_start = time.perf_counter()
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if prompt_format == "raw":
        prompt_ids = tokenizer.encode(problem.prompt, return_tensors="pt").to(device)
    elif prompt_format == "chat":
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
        ).to(device)
    else:
        raise ValueError(f"Unsupported prompt_format: {prompt_format}")
    gold_token_ids = tokenizer.encode(problem.gold_answer, add_special_tokens=False)
    if not gold_token_ids:
        gold_token_ids = [0]

    target_branch = select_branch_id or ("B" if select_branch_b else None)

    # Step 0: Prefill trajectory
    sched_0: TrajectoryScheduleResult = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=prompt_ids,
        method=method,
        recurrent_steps=recurrent_steps,
        fork_after_step=fork_after_step,
        branch_count=branch_count,
        perturbation_method=perturbation_method,
        perturbation_relative_norm=perturbation_relative_norm,
        excursion_config=excursion_config,
        policy_config=policy_config,
        controller_checkpoint=controller_checkpoint,
        seed=seed,
        extra_steps_for_s1=extra_steps_for_s1,
        record_pca=True,
    )

    if target_branch and target_branch in sched_0.branch_logits:
        step0_logits = sched_0.branch_logits[target_branch]
    else:
        step0_logits = sched_0.primary_logits

    first_gold_lp = compute_token_logprob(step0_logits, gold_token_ids[0])
    ent_0, margin_0 = compute_last_token_uncertainty(step0_logits)
    tok_0 = int(step0_logits[:, -1, :].argmax(dim=-1).item())
    generated_tokens = [tok_0]
    first_decoded = tokenizer.decode([tok_0])

    token_ledger: List[Dict[str, Any]] = [
        _build_token_ledger_entry(0, tok_0, first_decoded, sched_0, method, excursion_config, prompt_ids, seed)
    ]
    step_k_logits_by_prefix: Dict[Tuple[int, ...], Tensor] = {}

    total_calls = sched_0.recurrent_block_calls
    total_ctrl_calls = sched_0.controller_calls
    total_ctrl_ms = sched_0.controller_overhead_ms
    margins = [margin_0]

    # Legacy raw prompts use short-answer stops; chat may contain reasoning/newlines.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_reason = "max_output_tokens"
    if tok_0 == eos_id:
        stop_reason = "eos_token"
    elif prompt_format == "raw" and "\n" in first_decoded:
        stop_reason = "newline"
    elif prompt_format == "raw" and problem.template_id in (
        "logic_bool_xor_not_v1", "logic_bool_or_xor_test", "logic_mux_branch_fresh",
    ):
        stop_reason = "boolean_single_token_rule"
    needs_more = (
        max_output_tokens > 1
        and stop_reason == "max_output_tokens"
    )

    if needs_more:
        cur_ids = torch.cat(
            [prompt_ids, torch.tensor([[tok_0]], dtype=torch.long, device=device)],
            dim=1,
        )
        for tok_step in range(1, max_output_tokens):
            sched_k = run_scheduled_trajectory(
                adapter=adapter,
                input_ids=cur_ids,
                method=method,
                recurrent_steps=recurrent_steps,
                fork_after_step=fork_after_step,
                branch_count=branch_count,
                perturbation_method=perturbation_method,
                perturbation_relative_norm=perturbation_relative_norm,
                excursion_config=excursion_config,
                policy_config=policy_config,
                controller_checkpoint=controller_checkpoint,
                seed=seed + tok_step * 131,
                extra_steps_for_s1=extra_steps_for_s1,
                record_pca=False,
            )
            logits_k = (
                sched_k.branch_logits[target_branch]
                if (target_branch and target_branch in sched_k.branch_logits)
                else sched_k.primary_logits
            )
            step_k_logits_by_prefix[tuple(generated_tokens)] = logits_k
            _, mar_k = compute_last_token_uncertainty(logits_k)
            margins.append(mar_k)
            tok_k = int(logits_k[:, -1, :].argmax(dim=-1).item())
            tok_k_str = tokenizer.decode([tok_k])
            token_ledger.append(
                _build_token_ledger_entry(tok_step, tok_k, tok_k_str, sched_k, method, excursion_config, cur_ids, seed + tok_step * 131)
            )
            generated_tokens.append(tok_k)
            total_calls += sched_k.recurrent_block_calls
            total_ctrl_calls += sched_k.controller_calls
            total_ctrl_ms += sched_k.controller_overhead_ms
            dec_k = tokenizer.decode(generated_tokens)
            if tok_k == eos_id:
                stop_reason = "eos_token"
                break
            if prompt_format == "raw" and "\n" in dec_k:
                stop_reason = "newline"
                break
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[tok_k]], dtype=torch.long, device=device)],
                dim=1,
            )

    _sync_device(device)
    total_wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
    peak_mem_mb = _get_memory_mb(device)

    # Compute teacher-forced sequence_gold_logprob across all gold answer tokens
    # Any teacher-forced scoring calls (when greedy token != gold prefix) are tracked separately
    # in `scoring_recurrent_calls` and NOT added to inference `recurrent_block_calls`.
    gold_token_lps: List[float] = [first_gold_lp]
    scoring_calls = 0
    if len(gold_token_ids) > 1:
        for g_idx in range(1, min(len(gold_token_ids), max_output_tokens)):
            prefix_tuple = tuple(gold_token_ids[:g_idx])
            if prefix_tuple in step_k_logits_by_prefix:
                g_logits = step_k_logits_by_prefix[prefix_tuple]
            else:
                calls_pre_score = adapter.recurrent_block_calls
                tf_ids = torch.cat(
                    [
                        prompt_ids,
                        torch.tensor([list(prefix_tuple)], dtype=torch.long, device=device),
                    ],
                    dim=1,
                )
                sched_tf = run_scheduled_trajectory(
                    adapter=adapter,
                    input_ids=tf_ids,
                    method=method,
                    recurrent_steps=recurrent_steps,
                    fork_after_step=fork_after_step,
                    branch_count=branch_count,
                    perturbation_method=perturbation_method,
                    perturbation_relative_norm=perturbation_relative_norm,
                    excursion_config=excursion_config,
                    policy_config=policy_config,
                    controller_checkpoint=controller_checkpoint,
                    seed=seed + g_idx * 131,
                    extra_steps_for_s1=extra_steps_for_s1,
                    record_pca=False,
                )
                scoring_calls += int(adapter.recurrent_block_calls - calls_pre_score)
                g_logits = (
                    sched_tf.branch_logits[target_branch]
                    if (target_branch and target_branch in sched_tf.branch_logits)
                    else sched_tf.primary_logits
                )
            gold_token_lps.append(compute_token_logprob(g_logits, gold_token_ids[g_idx]))

    sequence_gold_lp = float(sum(gold_token_lps))
    mean_gold_token_lp = float(sum(gold_token_lps) / len(gold_token_lps))

    decoded_text = tokenizer.decode(generated_tokens).strip()
    val_outcome = validate_problem_response(problem, decoded_text)

    estimated_flops = 0.0
    for entry in token_ledger:
        seq_len = entry["prefix_length"]
        if hasattr(adapter, "estimate_step_flops"):
            flops_per_call = float(adapter.estimate_step_flops(seq_len=seq_len, batch_size=1))
        else:
            flops_per_call = float(seq_len * adapter.hidden_size * adapter.hidden_size * 12)
        estimated_flops += entry["recurrent_block_calls"] * flops_per_call

    telem = sched_0.excursion_telemetry
    pol_dec = sched_0.policy_decision
    dec_summary = _resolve_decision_summary(method, sched_0, excursion_config)

    return {
        "problem_id": problem.problem_id,
        "family": problem.family,
        "split": problem.split,
        "template_id": problem.template_id,
        "difficulty": problem.difficulty,
        "prompt": problem.prompt,
        "prompt_hash": hashlib.sha256(problem.prompt.encode("utf-8")).hexdigest()[:16],
        "prompt_format": prompt_format,
        "prompt_token_ids": prompt_ids.detach().cpu().tolist()[0],
        "generated_token_ids": generated_tokens,
        "trajectory_semantics": "full_prefix_refork_per_generated_token",
        "max_output_tokens": max_output_tokens,
        "stop_reason": stop_reason,
        "gold_answer": problem.gold_answer,
        "method": method,
        "seed": seed,
        "decoded_text": decoded_text,
        "extracted_answer": val_outcome.extracted_answer,
        "is_correct": val_outcome.is_valid,
        "validation_reason": val_outcome.reason,
        "gold_logprob": round(first_gold_lp, 6),
        "first_token_gold_logprob": round(first_gold_lp, 6),
        "sequence_gold_logprob": round(sequence_gold_lp, 6),
        "mean_token_gold_logprob": round(mean_gold_token_lp, 6),
        "gold_token_logprobs": [round(x, 6) for x in gold_token_lps],
        "first_token_entropy": round(ent_0, 6),
        "first_token_margin": round(margin_0, 6),
        "mean_sequence_margin": round(float(sum(margins) / len(margins)), 6),
        "recurrent_block_calls": total_calls,
        "scoring_recurrent_calls": scoring_calls,
        "controller_calls": total_ctrl_calls,
        "controller_overhead_ms": round(total_ctrl_ms, 4),
        "exited_early_step": sched_0.exited_early_step,
        "chosen_depth": dec_summary["chosen_depth"],
        "estimated_flops": estimated_flops,
        "wall_time_ms": round(total_wall_ms, 3),
        "peak_memory_mb": peak_mem_mb,
        "alpha": dec_summary["effective_alpha"],
        "beta": dec_summary["effective_beta"],
        "visit_step": dec_summary["chosen_depth"] if dec_summary["action_type"] == "EXCURSION" else excursion_config.visit_step,
        "token_scope": dec_summary["token_scope"],
        "perturbation_relative_norm": perturbation_relative_norm,
        "curvature_frob": round(telem.curvature_frob, 6) if telem is not None else 0.0,
        "curvature_relative": round(telem.curvature_relative_to_update_a, 6) if telem is not None else 0.0,
        "returned_shift_vs_normal_frob": round(getattr(telem, "returned_shift_vs_normal_frob", 0.0), 6)
        if telem is not None
        else 0.0,
        "branch_distance_frob": round(telem.branch_distance_frob, 6) if telem is not None else 0.0,
        "branch_relative_distance": round(telem.branch_relative_distance, 6) if telem is not None else 0.0,
        "branch_cosine_similarity": round(telem.branch_cosine_similarity, 6) if telem is not None else 1.0,
        "visit_offset_frob": round(telem.visit_offset_frob, 6) if telem is not None else 0.0,
        "clipped": bool(telem.clipped) if telem is not None else False,
        "skipped_unstable": bool(telem.skipped_unstable) if telem is not None else False,
        "post_rms_a": round(telem.post_rms_a, 6)
        if telem is not None
        else (round(sched_0.step_snapshots[-1].branch_rms.get("A", 0.0), 6) if sched_0.step_snapshots else 0.0),
        "post_finite": bool(telem.post_finite) if telem is not None else True,
        "policy_action": dec_summary["policy_action"],
        "policy_reason": dec_summary["policy_reason"],
        "controller_action_type": dec_summary["action_type"],
        "controller_action_id": dec_summary["action_id"],
        "controller_donor_index": dec_summary["donor_index"],
        "selected_donor": dec_summary["selected_donor"],
        "early_exit_step": sched_0.exited_early_step,
        "controller_predicted_advantage": round(dec_summary["predicted_advantage"], 6),
        "controller_advantage": round(dec_summary["predicted_advantage"], 6),
        "controller_action_scores": dec_summary["action_scores"],
        "controller_rejected_actions": dec_summary["rejected_actions"],
        "controller_state_features": getattr(pol_dec, "state_features", None) if pol_dec is not None else None,
        "token_ledger": token_ledger,
        "step_snapshots": [asdict(s) for s in sched_0.step_snapshots],
        "pca_coords": sched_0.pca_coords,
        "pca_explained_variance": sched_0.pca_explained_variance,
        "_step0_logits": step0_logits.detach().cpu(),
    }


def run_experiment_from_config(
    config_path: Path,
    repo_root: Path,
    output_root: Optional[Path] = None,
    adapter_override: Optional[tuple[RecurrentAdapter, Any, str, str]] = None,
    shared_eval_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute an experiment defined by a YAML config file and write full artifacts."""
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = raw_cfg.get("phase", 1)
    run_id = str(
        raw_cfg.get(
            "run_id",
            f"phase{phase}_{config_path.stem}_{int(time.time())}",
        )
    )
    out_dir = (output_root if output_root is not None else (repo_root / "results")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = raw_cfg.get("model", {})
    branch_cfg = raw_cfg.get("branches", {})
    interv_cfg = raw_cfg.get("intervention", {})
    exec_cfg = raw_cfg.get("execution", {})
    eval_cfg = raw_cfg.get("evaluation", {})
    ctrl_raw = raw_cfg.get("controller", {})
    pol_raw = raw_cfg.get("policy", {})
    def _load_ckpt_opt(key_name: str, default_rel: str) -> Optional[LearnedControllerCheckpoint]:
        rel_val = ctrl_raw.get(key_name, default_rel)
        if not rel_val:
            return None
        cp = Path(str(rel_val))
        if not cp.is_absolute():
            cp = repo_root / cp
        if cp.exists():
            return LearnedControllerCheckpoint.load(cp)
        return None

    ctrl_ckpt = _load_ckpt_opt("checkpoint_path", "")
    mlp_ckpt = _load_ckpt_opt("mlp_checkpoint_path", "artifacts/controllers/tebdil_mlp_v1.json")
    mlp_ungated_ckpt = _load_ckpt_opt(
        "mlp_ungated_checkpoint_path", "artifacts/controllers/tebdil_mlp_ungated_v1.json"
    )
    mlp_unshaped_ckpt = _load_ckpt_opt(
        "mlp_unshaped_checkpoint_path", "artifacts/controllers/tebdil_mlp_unshaped_v1.json"
    )
    mlp_t2_1donor_ckpt = _load_ckpt_opt(
        "mlp_t2_1donor_checkpoint_path", "artifacts/controllers/tebdil_mlp_t2_1donor_v1.json"
    )

    if adapter_override is not None:
        adapter, tokenizer, model_id, model_rev = adapter_override
    else:
        adapter, tokenizer, model_id, model_rev = load_adapter_and_tokenizer(model_cfg)

    recurrent_steps = int(model_cfg.get("recurrent_steps", adapter.total_steps))
    fork_after_step = int(branch_cfg.get("fork_after_step", 1))
    branch_count = int(branch_cfg.get("count", 2))
    pert_method = str(branch_cfg.get("perturbation", "normalized_gaussian"))
    pert_norm = float(branch_cfg.get("perturbation_relative_norm", 0.02))
    raw_elig = interv_cfg.get("eligible_visit_steps", [])
    elig_tuple = tuple(int(s) for s in raw_elig) if raw_elig else ()

    base_exc_cfg = ExcursionConfig(
        alpha=float(interv_cfg.get("alpha", 0.5)),
        beta=float(interv_cfg.get("beta", 0.5)),
        max_relative_norm=float(interv_cfg.get("max_relative_norm", 0.15)),
        visit_step=int(interv_cfg.get("visit_step", 2)),
        token_scope=str(interv_cfg.get("token_scope", "full")),
        donor_strategy=str(interv_cfg.get("donor_strategy", "fixed")),
        fallback_if_unstable=str(interv_cfg.get("fallback_if_unstable", "skip_intervention")),
        coordinate_topk_fraction=float(interv_cfg.get("coordinate_topk_fraction", 1.0)),
        eligible_visit_steps=elig_tuple,
    )
    policy_cfg = AdaptivePolicyConfig(
        min_relative_divergence=float(pol_raw.get("min_relative_divergence", 0.002)),
        max_relative_divergence=float(pol_raw.get("max_relative_divergence", 0.65)),
        min_entropy_trigger=float(pol_raw.get("min_entropy_trigger", 1.20)),
        max_margin_trigger=float(pol_raw.get("max_margin_trigger", 0.70)),
        stagnation_ratio_threshold=float(pol_raw.get("stagnation_ratio_threshold", 0.85)),
        donor_selection=str(pol_raw.get("donor_selection", "confidence_gain")),
    )

    seeds: List[int] = exec_cfg.get("seeds", [int(exec_cfg.get("seed", 42))])
    max_output_tokens = int(exec_cfg.get("max_output_tokens", 2))
    extra_s1 = int(interv_cfg.get("extra_steps_for_s1", 3))
    task_family = str(eval_cfg.get("task_family", "arithmetic"))
    split = str(eval_cfg.get("split", "pilot"))
    num_problems = int(eval_cfg.get("num_problems", 20))

    methods: List[str] = raw_cfg.get(
        "methods",
        ["S0", str(interv_cfg.get("method", "E1"))],
    )
    # Map descriptive method names to canonical IDs if needed
    method_alias = {
        "excursion_return": "E1",
        "targeted_excursion": "E1",
        "adaptive_excursion": "E2",
        "direct_state_mixing": "M0",
        "update_mixing": "M1",
        "random_perturbation": "R0",
    }
    canonical_methods = []
    for m in methods:
        cm = method_alias.get(m, m)
        if cm not in canonical_methods:
            canonical_methods.append(cm)
    if "S0" not in canonical_methods:
        canonical_methods.insert(0, "S0")

    t_exp_start = time.perf_counter()
    calls_exp_start = adapter.recurrent_block_calls
    manifest = create_run_manifest(
        run_id=run_id,
        repo_root=repo_root,
        model_id=model_id,
        model_revision=model_rev,
        seed=seeds[0],
        task_family=task_family,
        split=split,
        decode_policy=str(exec_cfg.get("decoding", "greedy")),
        recurrent_steps=recurrent_steps,
        branch_count=branch_count,
        intervention=",".join(canonical_methods),
        intervention_step=base_exc_cfg.visit_step,
        alpha=base_exc_cfg.alpha,
        beta=base_exc_cfg.beta,
        norm_clip=base_exc_cfg.max_relative_norm,
        cache_policy="disabled",
        precision=str(model_cfg.get("precision", "bf16")),
        config_path=str(config_path),
        output_dir=str(out_dir),
    )
    save_run_manifest(manifest, out_dir / "manifest.yaml")

    problems = generate_task_suite(
        family=task_family,
        split=split,
        num_problems=num_problems,
        seed=seeds[0],
    )

    all_records: List[Dict[str, Any]] = []
    if shared_eval_cache is None:
        shared_eval_cache = {}

    def _select_ckpt_for_method(m_name: str) -> Optional[LearnedControllerCheckpoint]:
        if m_name == "E3_mlp":
            return mlp_ckpt or ctrl_ckpt
        if m_name == "E3_mlp_ungated":
            return mlp_ungated_ckpt or mlp_ckpt or ctrl_ckpt
        if m_name == "E3_mlp_unshaped":
            return mlp_unshaped_ckpt or mlp_ckpt or ctrl_ckpt
        if m_name == "E3_t2_1donor":
            return mlp_t2_1donor_ckpt or mlp_ckpt or ctrl_ckpt
        return ctrl_ckpt

    def _cache_key(prob: TaskProblem, m_name: str, sd: int) -> str:
        active_ck = _select_ckpt_for_method(m_name)
        ck_sig = "none"
        if active_ck is not None:
            t_meta = active_ck.training_metadata or {}
            ck_sig = (
                f"{active_ck.controller_id}:{t_meta.get('final_param_hash', '')}:"
                f"{active_ck.exit_min_advantage}:{active_ck.excursion_min_advantage}:"
                f"{active_ck.use_heuristic_gates}:{active_ck.utility_version}"
            )
        prompt_h = hashlib.sha256(prob.prompt.encode("utf-8")).hexdigest()[:16]
        eff_payload = {
            "model_id": model_id,
            "model_rev": model_rev,
            "prob_id": prob.problem_id,
            "prompt_hash": prompt_h,
            "method": m_name,
            "seed": sd,
            "recurrent_steps": recurrent_steps,
            "fork_after_step": fork_after_step,
            "branch_count": 1 if m_name == "S0" else (2 if m_name == "E3_t2_1donor" else branch_count),
            "pert_method": "none" if m_name in ("S0", "S1") else pert_method,
            "pert_norm": 0.0 if m_name in ("S0", "S1") else pert_norm,
            "excursion_cfg": asdict(base_exc_cfg),
            "policy_cfg": asdict(policy_cfg),
            "max_output_tokens": max_output_tokens,
            "extra_steps_for_s1": extra_s1 if m_name == "S1" else 0,
            "ckpt_sig": ck_sig,
        }
        return hashlib.sha256(
            json.dumps(eff_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    for seed in seeds:
        for prob in problems:
            # 1. Always evaluate S0 baseline first for paired reference
            s0_key = _cache_key(prob, "S0", seed)
            if s0_key in shared_eval_cache:
                s0_cached = shared_eval_cache[s0_key]
                s0_res = dict(s0_cached["res"])
                s0_logits = s0_cached["logits"]
            else:
                s0_res = _decode_method_on_problem(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    problem=prob,
                    method="S0",
                    seed=seed,
                    recurrent_steps=recurrent_steps,
                    fork_after_step=fork_after_step,
                    branch_count=1,
                    perturbation_method="none",
                    perturbation_relative_norm=0.0,
                    excursion_config=base_exc_cfg,
                    policy_config=policy_cfg,
                    max_output_tokens=max_output_tokens,
                )
                s0_logits = s0_res.pop("_step0_logits")
                shared_eval_cache[s0_key] = {"res": dict(s0_res), "logits": s0_logits}

            # 2. Evaluate independent donor branch B (and C if branch_count >= 3) once per (problem, seed)
            b_key = _cache_key(prob, "_B_INDEP", seed)
            if b_key in shared_eval_cache:
                b_cached = shared_eval_cache[b_key]
                b_res = dict(b_cached["res"])
                b_logits = b_cached["logits"]
            else:
                b_res = _decode_method_on_problem(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    problem=prob,
                    method="P1",
                    seed=seed,
                    recurrent_steps=recurrent_steps,
                    fork_after_step=fork_after_step,
                    branch_count=branch_count,
                    perturbation_method=pert_method,
                    perturbation_relative_norm=pert_norm,
                    excursion_config=base_exc_cfg,
                    policy_config=policy_cfg,
                    max_output_tokens=max_output_tokens,
                    select_branch_b=True,
                    select_branch_id="B",
                )
                b_logits = b_res.pop("_step0_logits")
                shared_eval_cache[b_key] = {"res": dict(b_res), "logits": b_logits}

            c_res = None
            c_logits = None
            if branch_count >= 3:
                c_key = _cache_key(prob, "_C_INDEP", seed)
                if c_key in shared_eval_cache:
                    c_cached = shared_eval_cache[c_key]
                    c_res = dict(c_cached["res"])
                    c_logits = c_cached["logits"]
                else:
                    c_res = _decode_method_on_problem(
                        adapter=adapter,
                        tokenizer=tokenizer,
                        problem=prob,
                        method="P1",
                        seed=seed,
                        recurrent_steps=recurrent_steps,
                        fork_after_step=fork_after_step,
                        branch_count=branch_count,
                        perturbation_method=pert_method,
                        perturbation_relative_norm=pert_norm,
                        excursion_config=base_exc_cfg,
                        policy_config=policy_cfg,
                        max_output_tokens=max_output_tokens,
                        select_branch_id="C",
                    )
                    c_logits = c_res.pop("_step0_logits")
                    shared_eval_cache[c_key] = {"res": dict(c_res), "logits": c_logits}

            base_a_correct = bool(s0_res["is_correct"])
            donor_b_correct = bool(b_res["is_correct"])
            donor_c_correct = bool(c_res["is_correct"]) if c_res is not None else False
            any_donor_correct = donor_b_correct or donor_c_correct
            base_a_lp = float(s0_res["gold_logprob"])
            base_a_seq_lp = float(s0_res.get("sequence_gold_logprob", base_a_lp))
            parallel_sources = [("A", s0_res), ("B", b_res)]
            if c_res is not None:
                parallel_sources.append(("C", c_res))

            for method in canonical_methods:
                if method == "S0":
                    rec = dict(s0_res)
                    rec["base_a_correct"] = base_a_correct
                    rec["donor_b_correct"] = any_donor_correct
                    rec["donor_b_answer"] = b_res["extracted_answer"]
                    rec["donor_c_answer"] = c_res["extracted_answer"] if c_res is not None else None
                    rec["delta_gold_logprob"] = 0.0
                    rec["delta_sequence_gold_logprob"] = 0.0
                    rec["jsd_vs_s0"] = 0.0
                    all_records.append(rec)
                    continue

                if method == "P0":
                    # Label-free parallel selection across independent branches (A, B, and C if present)
                    cand_branches = [("A", s0_res, s0_logits), ("B", b_res, b_logits)]
                    if c_res is not None and c_logits is not None:
                        cand_branches.append(("C", c_res, c_logits))
                    best_b_id, best_b_res, best_b_logits = max(
                        cand_branches, key=lambda item: float(item[1]["mean_sequence_margin"])
                    )
                    chosen = _attach_parallel_compute_provenance(
                        best_b_res,
                        compute_source=b_res,
                        method="P0",
                        selected_branch=best_b_id,
                        consulted_sources=parallel_sources,
                    )
                    chosen["selected_donor"] = best_b_id
                    chosen["base_a_correct"] = base_a_correct
                    chosen["donor_b_correct"] = any_donor_correct
                    chosen["donor_b_answer"] = b_res["extracted_answer"]
                    chosen["donor_c_answer"] = c_res["extracted_answer"] if c_res is not None else None
                    chosen["delta_gold_logprob"] = round(float(chosen["gold_logprob"]) - base_a_lp, 6)
                    chosen["delta_sequence_gold_logprob"] = round(
                        float(chosen.get("sequence_gold_logprob", chosen["gold_logprob"])) - base_a_seq_lp, 6
                    )
                    cmp_b = compare_logits(s0_logits, best_b_logits)
                    chosen["jsd_vs_s0"] = round(cmp_b.js_divergence, 6)
                    all_records.append(chosen)
                    continue

                if method == "P1":
                    # Oracle best-of-N upper bound (analysis only)
                    if base_a_correct:
                        best_b_id, best_b_res, best_b_logits = ("A", s0_res, s0_logits)
                    elif donor_b_correct:
                        best_b_id, best_b_res, best_b_logits = ("B", b_res, b_logits)
                    elif c_res is not None and c_logits is not None and donor_c_correct:
                        best_b_id, best_b_res, best_b_logits = ("C", c_res, c_logits)
                    else:
                        best_b_id, best_b_res, best_b_logits = ("A", s0_res, s0_logits)
                    chosen = _attach_parallel_compute_provenance(
                        best_b_res,
                        compute_source=b_res,
                        method="P1",
                        selected_branch=best_b_id,
                        consulted_sources=parallel_sources,
                    )
                    chosen["selected_donor"] = best_b_id
                    chosen["base_a_correct"] = base_a_correct
                    chosen["donor_b_correct"] = any_donor_correct
                    chosen["donor_b_answer"] = b_res["extracted_answer"]
                    chosen["donor_c_answer"] = c_res["extracted_answer"] if c_res is not None else None
                    chosen["delta_gold_logprob"] = round(float(chosen["gold_logprob"]) - base_a_lp, 6)
                    chosen["delta_sequence_gold_logprob"] = round(
                        float(chosen.get("sequence_gold_logprob", chosen["gold_logprob"])) - base_a_seq_lp, 6
                    )
                    cmp_b = compare_logits(s0_logits, best_b_logits)
                    chosen["jsd_vs_s0"] = round(cmp_b.js_divergence, 6)
                    all_records.append(chosen)
                    continue

                m_key = _cache_key(prob, method, seed)
                if m_key in shared_eval_cache:
                    m_cached = shared_eval_cache[m_key]
                    m_res = dict(m_cached["res"])
                    m_logits = m_cached["logits"]
                elif method == "D_oracle":
                    # Diagnostic Oracle Donor Selection (analytical upper bound only)
                    d_near_key = _cache_key(prob, "D_nearest", seed)
                    d_far_key = _cache_key(prob, "D_farthest", seed)
                    for d_m, d_k in (("D_nearest", d_near_key), ("D_farthest", d_far_key)):
                        if d_k not in shared_eval_cache:
                            tmp_res = _decode_method_on_problem(
                                adapter=adapter,
                                tokenizer=tokenizer,
                                problem=prob,
                                method=d_m,
                                seed=seed,
                                recurrent_steps=recurrent_steps,
                                fork_after_step=fork_after_step,
                                branch_count=branch_count,
                                perturbation_method=pert_method,
                                perturbation_relative_norm=pert_norm,
                                excursion_config=base_exc_cfg,
                                policy_config=policy_cfg,
                                controller_checkpoint=ctrl_ckpt,
                                max_output_tokens=max_output_tokens,
                                extra_steps_for_s1=extra_s1,
                            )
                            tmp_logits = tmp_res.pop("_step0_logits")
                            shared_eval_cache[d_k] = {"res": dict(tmp_res), "logits": tmp_logits}
                    near_c = shared_eval_cache[d_near_key]
                    far_c = shared_eval_cache[d_far_key]
                    pick_far = (not near_c["res"]["is_correct"]) and (
                        far_c["res"]["is_correct"] or far_c["res"]["gold_logprob"] > near_c["res"]["gold_logprob"]
                    )
                    chosen_c = far_c if pick_far else near_c
                    m_res = dict(chosen_c["res"])
                    m_res["method"] = "D_oracle"
                    m_res["policy_reason"] = "diagnostic_oracle_donor_upper_bound"
                    m_logits = chosen_c["logits"]
                else:
                    active_ckpt = _select_ckpt_for_method(method)
                    m_res = _decode_method_on_problem(
                        adapter=adapter,
                        tokenizer=tokenizer,
                        problem=prob,
                        method=method,
                        seed=seed,
                        recurrent_steps=recurrent_steps,
                        fork_after_step=fork_after_step,
                        branch_count=branch_count,
                        perturbation_method=pert_method,
                        perturbation_relative_norm=pert_norm,
                        excursion_config=base_exc_cfg,
                        policy_config=policy_cfg,
                        controller_checkpoint=active_ckpt,
                        max_output_tokens=max_output_tokens,
                        extra_steps_for_s1=extra_s1,
                    )
                    m_logits = m_res.pop("_step0_logits")
                    shared_eval_cache[m_key] = {"res": dict(m_res), "logits": m_logits}

                cmp_m = compare_logits(s0_logits, m_logits)
                m_res["base_a_correct"] = base_a_correct
                m_res["donor_b_correct"] = any_donor_correct
                m_res["donor_b_answer"] = b_res["extracted_answer"]
                m_res["donor_c_answer"] = c_res["extracted_answer"] if c_res is not None else None
                m_res["delta_gold_logprob"] = round(float(m_res["gold_logprob"]) - base_a_lp, 6)
                m_res["delta_sequence_gold_logprob"] = round(
                    float(m_res.get("sequence_gold_logprob", m_res["gold_logprob"])) - base_a_seq_lp, 6
                )
                m_res["jsd_vs_s0"] = round(cmp_m.js_divergence, 6)
                all_records.append(m_res)

    write_trace_records(all_records, out_dir / "traces.jsonl")

    total_scoring_calls = int(sum(int(r.get("scoring_recurrent_calls", 0)) for r in all_records))
    manifest.tokenizer_revision = model_rev
    manifest.controller_checkpoint = str(ctrl_raw.get("checkpoint_path", "none"))
    manifest.recurrent_forward_calls = max(
        0, int(adapter.recurrent_block_calls - calls_exp_start - total_scoring_calls)
    )
    manifest.controller_calls = int(sum(int(r.get("controller_calls", 0)) for r in all_records))
    manifest.runtime_seconds = round(time.perf_counter() - t_exp_start, 3)
    save_run_manifest(manifest, out_dir / "manifest.yaml")

    from tebdil.analysis.compute_frontier import compute_pareto_frontier
    from tebdil.analysis.curvature import analyze_curvature
    from tebdil.analysis.rescue import compute_rescue_matrix

    rescue_matrix = compute_rescue_matrix(all_records, baseline_method="S0")
    curvature_summary = analyze_curvature(all_records)
    pareto_summary = compute_pareto_frontier(rescue_matrix["methods"])

    summary_payload = {
        "run_id": run_id,
        "manifest": asdict(manifest),
        "num_total_records": len(all_records),
        "rescue_matrix": rescue_matrix,
        "curvature_analysis": curvature_summary,
        "pareto_frontier": pareto_summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload
