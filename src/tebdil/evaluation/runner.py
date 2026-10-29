"""Deterministic experiment runner for TEBDIL phases and comparative benchmarks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
from torch import Tensor
import yaml

from tebdil.controller.learned_policy import LearnedControllerCheckpoint
from tebdil.core.policies import AdaptivePolicyConfig, compute_last_token_uncertainty
from tebdil.core.scheduler import TrajectoryScheduleResult, run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig, RecurrentAdapter
from tebdil.evaluation.metrics import compare_logits, compute_token_logprob
from tebdil.evaluation.trace import (
    create_run_manifest,
    read_trace_records,
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
) -> Dict[str, Any]:
    """Run a single method on a single problem and decode its short continuation."""
    device = getattr(adapter, "device", torch.device("cpu"))
    prompt_ids = tokenizer.encode(problem.prompt, return_tensors="pt").to(device)
    gold_token_ids = tokenizer.encode(problem.gold_answer, add_special_tokens=False)
    if not gold_token_ids:
        gold_token_ids = [0]

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

    if select_branch_b and "B" in sched_0.branch_logits:
        step0_logits = sched_0.branch_logits["B"]
    else:
        step0_logits = sched_0.primary_logits

    first_gold_lp = compute_token_logprob(step0_logits, gold_token_ids[0])
    ent_0, margin_0 = compute_last_token_uncertainty(step0_logits)
    tok_0 = int(step0_logits[:, -1, :].argmax(dim=-1).item())
    generated_tokens = [tok_0]

    total_calls = sched_0.recurrent_block_calls
    total_wall_ms = sched_0.wall_time_ms
    total_ctrl_calls = sched_0.controller_calls
    total_ctrl_ms = sched_0.controller_overhead_ms
    margins = [margin_0]

    # Determine whether a second continuation token is needed
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
                sched_k.branch_logits["B"]
                if (select_branch_b and "B" in sched_k.branch_logits)
                else sched_k.primary_logits
            )
            _, mar_k = compute_last_token_uncertainty(logits_k)
            margins.append(mar_k)
            tok_k = int(logits_k[:, -1, :].argmax(dim=-1).item())
            generated_tokens.append(tok_k)
            total_calls += sched_k.recurrent_block_calls
            total_wall_ms += sched_k.wall_time_ms
            total_ctrl_calls += sched_k.controller_calls
            total_ctrl_ms += sched_k.controller_overhead_ms
            dec_k = tokenizer.decode(generated_tokens)
            if "\n" in dec_k:
                break
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[tok_k]], dtype=torch.long, device=device)],
                dim=1,
            )

    decoded_text = tokenizer.decode(generated_tokens).strip()
    val_outcome = validate_problem_response(problem, decoded_text)

    seq_len = int(prompt_ids.shape[1])
    if hasattr(adapter, "estimate_step_flops"):
        flops_per_call = float(adapter.estimate_step_flops(seq_len=seq_len, batch_size=1))
    else:
        flops_per_call = float(seq_len * adapter.hidden_size * adapter.hidden_size * 12)
    estimated_flops = total_calls * flops_per_call

    telem = sched_0.excursion_telemetry
    pol_dec = sched_0.policy_decision
    act_type = getattr(pol_dec, "action_type", "EXCURSION" if (pol_dec and pol_dec.action == "visit") else ("CONTINUE" if pol_dec else "EXCURSION"))
    act_id = getattr(pol_dec, "action_id", pol_dec.action if pol_dec else "always")
    pred_adv = float(getattr(pol_dec, "predicted_advantage", 0.0))
    act_scores = getattr(pol_dec, "action_scores", {})
    eff_alpha = float(getattr(pol_dec, "adaptive_alpha", excursion_config.alpha)) if pol_dec else excursion_config.alpha
    eff_beta = float(getattr(pol_dec, "adaptive_beta", excursion_config.beta)) if pol_dec else excursion_config.beta
    donor_idx = int(getattr(pol_dec, "donor_index", 0)) if pol_dec else 0

    return {
        "problem_id": problem.problem_id,
        "family": problem.family,
        "split": problem.split,
        "template_id": problem.template_id,
        "difficulty": problem.difficulty,
        "prompt": problem.prompt,
        "prompt_hash": hashlib.sha256(problem.prompt.encode("utf-8")).hexdigest()[:16],
        "gold_answer": problem.gold_answer,
        "method": method,
        "seed": seed,
        "decoded_text": decoded_text,
        "extracted_answer": val_outcome.extracted_answer,
        "is_correct": val_outcome.is_valid,
        "validation_reason": val_outcome.reason,
        "gold_logprob": round(first_gold_lp, 6),
        "first_token_entropy": round(ent_0, 6),
        "first_token_margin": round(margin_0, 6),
        "mean_sequence_margin": round(float(sum(margins) / len(margins)), 6),
        "recurrent_block_calls": total_calls,
        "controller_calls": total_ctrl_calls,
        "controller_overhead_ms": round(total_ctrl_ms, 4),
        "exited_early_step": sched_0.exited_early_step,
        "estimated_flops": estimated_flops,
        "wall_time_ms": round(total_wall_ms, 3),
        "alpha": eff_alpha,
        "beta": eff_beta,
        "visit_step": excursion_config.visit_step,
        "token_scope": getattr(pol_dec, "token_scope", excursion_config.token_scope) if pol_dec else excursion_config.token_scope,
        "perturbation_relative_norm": perturbation_relative_norm,
        "curvature_frob": round(telem.curvature_frob, 6) if telem is not None else 0.0,
        "curvature_relative": round(telem.curvature_relative_to_update_a, 6) if telem is not None else 0.0,
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
        "policy_action": pol_dec.action if pol_dec is not None else "always",
        "policy_reason": pol_dec.reason if pol_dec is not None else "fixed_schedule",
        "controller_action_type": act_type,
        "controller_action_id": act_id,
        "controller_donor_index": donor_idx,
        "selected_donor": telem.donor_branch_id if telem is not None else ("B" if donor_idx == 0 else "C"),
        "early_exit_step": sched_0.exited_early_step,
        "controller_predicted_advantage": round(pred_adv, 6),
        "controller_advantage": round(pred_adv, 6),
        "controller_action_scores": act_scores,
        "controller_state_features": getattr(pol_dec, "state_features", None) if pol_dec is not None else None,
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
    ctrl_ckpt: Optional[LearnedControllerCheckpoint] = None
    mlp_ckpt: Optional[LearnedControllerCheckpoint] = None
    if ctrl_raw.get("checkpoint_path"):
        cp = Path(str(ctrl_raw["checkpoint_path"]))
        if not cp.is_absolute():
            cp = repo_root / cp
        if cp.exists():
            ctrl_ckpt = LearnedControllerCheckpoint.load(cp)
    if ctrl_raw.get("mlp_checkpoint_path"):
        mcp = Path(str(ctrl_raw["mlp_checkpoint_path"]))
        if not mcp.is_absolute():
            mcp = repo_root / mcp
        if mcp.exists():
            mlp_ckpt = LearnedControllerCheckpoint.load(mcp)

    if adapter_override is not None:
        adapter, tokenizer, model_id, model_rev = adapter_override
    else:
        adapter, tokenizer, model_id, model_rev = load_adapter_and_tokenizer(model_cfg)

    recurrent_steps = int(model_cfg.get("recurrent_steps", adapter.total_steps))
    fork_after_step = int(branch_cfg.get("fork_after_step", 1))
    branch_count = int(branch_cfg.get("count", 2))
    pert_method = str(branch_cfg.get("perturbation", "normalized_gaussian"))
    pert_norm = float(branch_cfg.get("perturbation_relative_norm", 0.02))

    base_exc_cfg = ExcursionConfig(
        alpha=float(interv_cfg.get("alpha", 0.5)),
        beta=float(interv_cfg.get("beta", 0.5)),
        max_relative_norm=float(interv_cfg.get("max_relative_norm", 0.15)),
        visit_step=int(interv_cfg.get("visit_step", 2)),
        token_scope=str(interv_cfg.get("token_scope", "full")),
        donor_strategy=str(interv_cfg.get("donor_strategy", "fixed")),
        fallback_if_unstable=str(interv_cfg.get("fallback_if_unstable", "skip_intervention")),
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

    def _cache_key(prob_id: str, m_name: str, sd: int) -> str:
        return (
            f"{model_id}:{model_rev}:{prob_id}:{m_name}:{sd}:{recurrent_steps}:"
            f"{fork_after_step}:{branch_count}:{pert_method}:{pert_norm}:"
            f"{base_exc_cfg.alpha}:{base_exc_cfg.beta}:{base_exc_cfg.max_relative_norm}:"
            f"{base_exc_cfg.visit_step}:{base_exc_cfg.token_scope}:{max_output_tokens}"
        )

    for seed in seeds:
        for prob in problems:
            # 1. Always evaluate S0 baseline first for paired reference
            s0_key = _cache_key(prob.problem_id, "S0", seed)
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

            # 2. Evaluate independent donor branch B once per (problem, seed)
            b_key = _cache_key(prob.problem_id, "_B_INDEP", seed)
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
                )
                b_logits = b_res.pop("_step0_logits")
                shared_eval_cache[b_key] = {"res": dict(b_res), "logits": b_logits}

            base_a_correct = bool(s0_res["is_correct"])
            donor_b_correct = bool(b_res["is_correct"])
            base_a_lp = float(s0_res["gold_logprob"])

            for method in canonical_methods:
                if method == "S0":
                    rec = dict(s0_res)
                    rec["base_a_correct"] = base_a_correct
                    rec["donor_b_correct"] = donor_b_correct
                    rec["donor_b_answer"] = b_res["extracted_answer"]
                    rec["delta_gold_logprob"] = 0.0
                    rec["jsd_vs_s0"] = 0.0
                    all_records.append(rec)
                    continue

                if method == "P0":
                    # Label-free parallel selection between independent A and independent B
                    choose_b = b_res["mean_sequence_margin"] > s0_res["mean_sequence_margin"]
                    chosen = dict(b_res if choose_b else s0_res)
                    chosen["method"] = "P0"
                    chosen["recurrent_block_calls"] = b_res["recurrent_block_calls"]
                    chosen["estimated_flops"] = b_res["estimated_flops"]
                    chosen["wall_time_ms"] = b_res["wall_time_ms"]
                    chosen["base_a_correct"] = base_a_correct
                    chosen["donor_b_correct"] = donor_b_correct
                    chosen["donor_b_answer"] = b_res["extracted_answer"]
                    chosen["delta_gold_logprob"] = round(float(chosen["gold_logprob"]) - base_a_lp, 6)
                    cmp_b = compare_logits(s0_logits, b_logits if choose_b else s0_logits)
                    chosen["jsd_vs_s0"] = round(cmp_b.js_divergence, 6)
                    all_records.append(chosen)
                    continue

                if method == "P1":
                    # Oracle best-of-2 upper bound (analysis only)
                    choose_b = (not base_a_correct) and donor_b_correct
                    chosen = dict(b_res if choose_b else s0_res)
                    chosen["method"] = "P1"
                    chosen["recurrent_block_calls"] = b_res["recurrent_block_calls"]
                    chosen["estimated_flops"] = b_res["estimated_flops"]
                    chosen["wall_time_ms"] = b_res["wall_time_ms"]
                    chosen["base_a_correct"] = base_a_correct
                    chosen["donor_b_correct"] = donor_b_correct
                    chosen["donor_b_answer"] = b_res["extracted_answer"]
                    chosen["delta_gold_logprob"] = round(float(chosen["gold_logprob"]) - base_a_lp, 6)
                    cmp_b = compare_logits(s0_logits, b_logits if choose_b else s0_logits)
                    chosen["jsd_vs_s0"] = round(cmp_b.js_divergence, 6)
                    all_records.append(chosen)
                    continue

                m_key = _cache_key(prob.problem_id, method, seed)
                if m_key in shared_eval_cache:
                    m_cached = shared_eval_cache[m_key]
                    m_res = dict(m_cached["res"])
                    m_logits = m_cached["logits"]
                elif method == "E1_no_curv" and _cache_key(prob.problem_id, "M1", seed) in shared_eval_cache:
                    # E1_no_curv is identical in state to M1, with E1's recurrent call count
                    m1_cached = shared_eval_cache[_cache_key(prob.problem_id, "M1", seed)]
                    m_res = dict(m1_cached["res"])
                    m_res["method"] = "E1_no_curv"
                    n_toks = max(1, int(round(m_res["recurrent_block_calls"] / 7.0)))
                    m_res["recurrent_block_calls"] = 8 * n_toks
                    m_logits = m1_cached["logits"]
                    shared_eval_cache[m_key] = {"res": dict(m_res), "logits": m_logits}
                elif method == "D_oracle":
                    # Diagnostic Oracle Donor Selection (analytical upper bound only)
                    # Evaluates D_nearest and D_farthest (which visit donor 0 or donor 1) and selects rescuing donor if any
                    d_near_key = _cache_key(prob.problem_id, "D_nearest", seed)
                    d_far_key = _cache_key(prob.problem_id, "D_farthest", seed)
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
                                extra_steps_for_s1=int(interv_cfg.get("extra_steps_for_s1", 3)),
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
                    active_ckpt = mlp_ckpt if (method == "E3_mlp" and mlp_ckpt is not None) else ctrl_ckpt
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
                        extra_steps_for_s1=int(interv_cfg.get("extra_steps_for_s1", 3)),
                    )
                    m_logits = m_res.pop("_step0_logits")
                    shared_eval_cache[m_key] = {"res": dict(m_res), "logits": m_logits}

                cmp_m = compare_logits(s0_logits, m_logits)
                m_res["base_a_correct"] = base_a_correct
                m_res["donor_b_correct"] = donor_b_correct
                m_res["donor_b_answer"] = b_res["extracted_answer"]
                m_res["delta_gold_logprob"] = round(float(m_res["gold_logprob"]) - base_a_lp, 6)
                m_res["jsd_vs_s0"] = round(cmp_m.js_divergence, 6)
                all_records.append(m_res)

    write_trace_records(all_records, out_dir / "traces.jsonl")

    manifest.tokenizer_revision = model_rev
    manifest.controller_checkpoint = str(ctrl_raw.get("checkpoint_path", "none"))
    manifest.recurrent_forward_calls = int(adapter.recurrent_block_calls - calls_exp_start)
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
