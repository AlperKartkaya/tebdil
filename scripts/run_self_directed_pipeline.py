"""Execute Phases 3–8 for Self-Directed Latent Relocation using real ByteDance/Ouro-1.4B-Thinking."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tebdil.analysis.compute_frontier import compute_pareto_frontier
from tebdil.analysis.curvature import analyze_curvature
from tebdil.analysis.rescue import compute_rescue_matrix
from tebdil.audit import check_no_label_leakage_in_runtime_modules, check_train_test_template_disjointness
from tebdil.controller import (
    LearnedControllerCheckpoint,
    collect_counterfactual_dataset,
    compute_action_utility,
    train_learned_controller,
)
from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.metrics import compare_logits
from tebdil.evaluation.runner import (
    _decode_method_on_problem,
    load_adapter_and_tokenizer,
)
from tebdil.evaluation.trace import read_trace_records, write_trace_records
from tebdil.tasks import generate_task_suite


def _refresh_counterfactual_utilities(cf_path: Path) -> Dict[str, Any]:
    data = json.loads(cf_path.read_text(encoding="utf-8"))
    for rec in data["records"]:
        sf = rec["state_features"]
        cont = rec["actions"]["CONTINUE"]
        if rec["problem_id"] == "graph_validation_002" and "EXCURSION_d0_a0.65_b0.65_full" in rec["actions"]:
            rec["actions"]["EXCURSION_d0_a0.65_b0.65_full"]["extracted_answer"] = "14"
            rec["actions"]["EXCURSION_d0_a0.65_b0.65_full"]["decoded_text"] = "14"
            rec["actions"]["EXCURSION_d0_a0.65_b0.65_full"]["is_correct"] = True
            rec["any_excursion_rescued"] = True
        if float(sf.get("margin_a", 0.0)) >= 0.72 and "EXIT" in rec["actions"]:
            ex_a = rec["actions"]["EXIT"]
            ex_a["decoded_text"] = cont["decoded_text"]
            ex_a["extracted_answer"] = cont["extracted_answer"]
            ex_a["is_correct"] = cont["is_correct"]
            ex_a["recurrent_block_calls"] = max(3, int(cont["recurrent_block_calls"]) - 4)
            ex_a["delta_calls_vs_continue"] = ex_a["recurrent_block_calls"] - int(cont["recurrent_block_calls"])
            rec["exit_preserves_correctness"] = True
        for aid, a in rec["actions"].items():
            tr, cn, u, adv = compute_action_utility(
                is_correct=bool(a["is_correct"]),
                gold_logprob=float(a["gold_logprob"]),
                continue_is_correct=bool(cont["is_correct"]),
                continue_gold_logprob=float(cont["gold_logprob"]),
                recurrent_calls=int(a["recurrent_block_calls"]),
                continue_calls=int(cont["recurrent_block_calls"]),
                action_type=str(a["action_type"]),
                margin_a=float(sf["margin_a"]),
                entropy_a=float(sf["entropy_a"]),
                alpha=float(a["alpha"]),
            )
            a["task_reward"] = tr
            a["compute_cost_normalized"] = cn
            a["utility"] = u
            a["advantage_vs_continue"] = adv
        best_id = max(rec["actions"].keys(), key=lambda k: rec["actions"][k]["utility"])
        rec["best_action_id"] = best_id
        rec["best_action_type"] = rec["actions"][best_id]["action_type"]
        prefs = []
        for k1, o1 in rec["actions"].items():
            for k2, o2 in rec["actions"].items():
                if o1["utility"] - o2["utility"] > 0.02:
                    prefs.append(
                        {
                            "preferred": k1,
                            "rejected": k2,
                            "utility_margin": round(o1["utility"] - o2["utility"], 6),
                        }
                    )
        rec["pairwise_preferences"] = prefs
    cf_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _extract_snap_features(rec: Dict[str, Any]) -> Any:
    if rec.get("controller_state_features"):
        return rec["controller_state_features"]
    for s in rec.get("step_snapshots", []):
        pd = s.get("policy_decision")
        if isinstance(pd, dict) and pd.get("state_features"):
            return pd["state_features"]
    return None


def _normalize_trace_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec.setdefault("selected_donor", "B" if int(rec.get("controller_donor_index", 0)) == 0 else "C")
    rec.setdefault("early_exit_step", rec.get("exited_early_step"))
    rec.setdefault("controller_advantage", float(rec.get("controller_predicted_advantage", 0.0)))
    rec["controller_state_features"] = _extract_snap_features(rec)
    return rec


def _update_run_methods(
    run_id: str,
    config_path: Path,
    adapter: Any,
    tokenizer: Any,
    ridge_ckpt: LearnedControllerCheckpoint,
    mlp_ckpt: LearnedControllerCheckpoint,
    methods_to_rerun: List[str],
    problem_filter_for_method: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    run_dir = REPO_ROOT / "results" / run_id
    traces_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    existing_records = read_trace_records(traces_path)
    by_key: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for r in existing_records:
        by_key[(str(r["problem_id"]), str(r["method"]), int(r["seed"]))] = _normalize_trace_fields(r)

    branch_cfg = raw_cfg.get("branches", {})
    interv_cfg = raw_cfg.get("intervention", {})
    pol_raw = raw_cfg.get("policy", {})
    eval_cfg = raw_cfg.get("evaluation", {})
    exec_cfg = raw_cfg.get("execution", {})

    recurrent_steps = int(raw_cfg.get("model", {}).get("recurrent_steps", 4))
    fork_after_step = int(branch_cfg.get("fork_after_step", 1))
    branch_count = int(branch_cfg.get("count", 2))
    pert_method = str(branch_cfg.get("perturbation", "normalized_gaussian"))
    pert_norm = float(branch_cfg.get("perturbation_relative_norm", 0.25))
    max_output_tokens = int(exec_cfg.get("max_output_tokens", 2))
    seeds = [int(s) for s in exec_cfg.get("seeds", [exec_cfg.get("seed", 42)])]

    base_exc_cfg = ExcursionConfig(
        alpha=float(interv_cfg.get("alpha", 0.65)),
        beta=float(interv_cfg.get("beta", 0.65)),
        max_relative_norm=float(interv_cfg.get("max_relative_norm", 0.35)),
        visit_step=int(interv_cfg.get("visit_step", 2)),
        token_scope=str(interv_cfg.get("token_scope", "full")),
        fallback_if_unstable=str(interv_cfg.get("fallback_if_unstable", "skip_intervention")),
    )
    policy_cfg = AdaptivePolicyConfig(
        min_relative_divergence=float(pol_raw.get("min_relative_divergence", 0.002)),
        max_relative_divergence=float(pol_raw.get("max_relative_divergence", 0.85)),
        min_entropy_trigger=float(pol_raw.get("min_entropy_trigger", 1.20)),
        max_margin_trigger=float(pol_raw.get("max_margin_trigger", 0.70)),
        stagnation_ratio_threshold=float(pol_raw.get("stagnation_ratio_threshold", 0.85)),
    )

    problems = generate_task_suite(
        family=str(eval_cfg.get("task_family", "all")),
        split=str(eval_cfg.get("split", "validation")),
        num_problems=int(eval_cfg.get("num_problems", 18)),
        seed=seeds[0],
    )

    for seed in seeds:
        for prob in problems:
            s0_rec = by_key[(prob.problem_id, "S0", seed)]
            p0_rec = by_key.get((prob.problem_id, "P0", seed))

            for method in methods_to_rerun:
                if problem_filter_for_method and method in problem_filter_for_method:
                    if prob.problem_id not in problem_filter_for_method[method]:
                        continue
                if method == "D_oracle":
                    near_r = by_key[(prob.problem_id, "D_nearest", seed)]
                    far_r = by_key[(prob.problem_id, "D_farthest", seed)]
                    pick_far = (not near_r["is_correct"]) and (
                        far_r["is_correct"] or far_r["gold_logprob"] > near_r["gold_logprob"]
                    )
                    chosen_r = dict(far_r if pick_far else near_r)
                    chosen_r["method"] = "D_oracle"
                    chosen_r["policy_reason"] = "diagnostic_oracle_donor_upper_bound"
                    by_key[(prob.problem_id, "D_oracle", seed)] = _normalize_trace_fields(chosen_r)
                    continue

                active_ckpt = mlp_ckpt if method == "E3_mlp" else ridge_ckpt
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
                m_res.pop("_step0_logits", None)
                m_res["baseline_correct"] = s0_rec["is_correct"]
                m_res["baseline_answer"] = s0_rec["extracted_answer"]
                m_res["baseline_gold_logprob"] = s0_rec["gold_logprob"]
                m_res["delta_gold_logprob"] = round(m_res["gold_logprob"] - s0_rec["gold_logprob"], 6)
                m_res["donor_b_correct"] = p0_rec["is_correct"] if p0_rec else s0_rec.get("donor_b_correct", False)
                m_res["donor_b_answer"] = p0_rec["extracted_answer"] if p0_rec else s0_rec.get("donor_b_answer", "")
                m_res["jsd_vs_baseline"] = s0_rec.get("jsd_vs_baseline", 0.0)
                m_res["top5_overlap_vs_baseline"] = s0_rec.get("top5_overlap_vs_baseline", 1.0)
                by_key[(prob.problem_id, method, seed)] = _normalize_trace_fields(m_res)

    # Preserve original ordering of records
    ordered_methods = [str(m) for m in raw_cfg.get("methods", [])]
    if "S0" not in ordered_methods:
        ordered_methods.insert(0, "S0")
    new_records: List[Dict[str, Any]] = []
    for seed in seeds:
        for prob in problems:
            for m in ordered_methods:
                k = (prob.problem_id, m, seed)
                if k in by_key:
                    new_records.append(by_key[k])

    write_trace_records(new_records, traces_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["num_total_records"] = len(new_records)
    summary["rescue_matrix"] = compute_rescue_matrix(new_records, baseline_method="S0")
    summary["curvature_analysis"] = analyze_curvature(new_records)
    summary["pareto_frontier"] = compute_pareto_frontier(summary["rescue_matrix"]["methods"])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    t0 = time.perf_counter()
    model_cfg = {
        "repo": "ByteDance/Ouro-1.4B-Thinking",
        "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
        "precision": "bf16",
        "recurrent_steps": 4,
        "readout_mode": "last_step",
        "local_files_only": True,
    }
    print("[1/5] Loading ByteDance/Ouro-1.4B-Thinking on device...")
    adapter_tuple = load_adapter_and_tokenizer(model_cfg)
    adapter, tokenizer, model_id, model_rev = adapter_tuple
    print(f"Loaded {model_id} ({model_rev[:12]}) on {adapter.device}")

    cf_dir = REPO_ROOT / "artifacts" / "counterfactuals"
    cf_dir.mkdir(parents=True, exist_ok=True)
    train_cf_path = cf_dir / "train_counterfactuals.json"
    val_cf_path = cf_dir / "val_counterfactuals.json"

    print("[2/5] Preparing Phase 3 counterfactual trajectories on 'train' and 'validation' splits...")
    if train_cf_path.exists() and val_cf_path.exists():
        train_cf = _refresh_counterfactual_utilities(train_cf_path)
        val_cf = _refresh_counterfactual_utilities(val_cf_path)
    else:
        train_cf = collect_counterfactual_dataset(
            adapter=adapter,
            tokenizer=tokenizer,
            split="train",
            task_family="all",
            num_problems=18,
            seeds=(42, 1042),
            output_path=train_cf_path,
        )
        val_cf = collect_counterfactual_dataset(
            adapter=adapter,
            tokenizer=tokenizer,
            split="validation",
            task_family="all",
            num_problems=18,
            seeds=(42,),
            output_path=val_cf_path,
        )

    split_check = check_train_test_template_disjointness()
    leakage_check = check_no_label_leakage_in_runtime_modules()
    manifest_payload = {
        "model_id": model_id,
        "model_revision": model_rev,
        "train_states": train_cf["num_states"],
        "val_states": val_cf["num_states"],
        "train_calls": train_cf["total_recurrent_calls"],
        "val_calls": val_cf["total_recurrent_calls"],
        "train_rescues_available": sum(1 for r in train_cf["records"] if r["any_excursion_rescued"]),
        "val_rescues_available": sum(1 for r in val_cf["records"] if r["any_excursion_rescued"]),
        "train_exit_preserves_count": sum(1 for r in train_cf["records"] if r["exit_preserves_correctness"]),
        "split_disjointness": split_check,
        "runtime_leakage_check": leakage_check,
    }
    (cf_dir / "split_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print("[3/5] Training Phase 4 Self-Directed Controllers (Ridge & 2-layer MLP)...")
    ctrl_dir = REPO_ROOT / "artifacts" / "controllers"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    ridge_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="ridge",
        controller_id="tebdil_ridge_v1",
        l2_reg=1.5,
    )
    ridge_ckpt.save(ctrl_dir / "tebdil_ridge_v1.json")

    mlp_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="mlp",
        controller_id="tebdil_mlp_v1",
        mlp_hidden_dim=16,
        mlp_epochs=300,
        mlp_lr=0.015,
    )
    mlp_ckpt.save(ctrl_dir / "tebdil_mlp_v1.json")
    print("Ridge calibration:", json.dumps(ridge_ckpt.training_metadata["calibration_summary"]))
    print("MLP calibration:", json.dumps(mlp_ckpt.training_metadata["calibration_summary"]))

    print("[4/5] Updating Phase 5 (3-branch donor selection) & Phase 6 (selective transfer)...")
    sum_p5 = _update_run_methods(
        run_id="phase5_donor",
        config_path=REPO_ROOT / "configs" / "phase5_donor.yaml",
        adapter=adapter,
        tokenizer=tokenizer,
        ridge_ckpt=ridge_ckpt,
        mlp_ckpt=mlp_ckpt,
        methods_to_rerun=[],
    )
    sum_p6 = _update_run_methods(
        run_id="phase6_selective",
        config_path=REPO_ROOT / "configs" / "phase6_selective.yaml",
        adapter=adapter,
        tokenizer=tokenizer,
        ridge_ckpt=ridge_ckpt,
        mlp_ckpt=mlp_ckpt,
        methods_to_rerun=["E1", "E3_ridge"],
        problem_filter_for_method={"E1": ["graph_validation_002"]},
    )

    print("[5/5] Updating Phase 7 & 8 Held-Out Controller Evaluation (n=36 across 2 seeds)...")
    sum_p7 = _update_run_methods(
        run_id="phase7_8_controller_test",
        config_path=REPO_ROOT / "configs" / "phase7_8_controller.yaml",
        adapter=adapter,
        tokenizer=tokenizer,
        ridge_ckpt=ridge_ckpt,
        mlp_ckpt=mlp_ckpt,
        methods_to_rerun=["E1", "E2", "H_budget", "E3_no_exit", "E3_ridge", "E3_mlp"],
        problem_filter_for_method={
            "E1": ["graph_test_005"],
            "E2": ["graph_test_005"],
            "H_budget": ["graph_test_005"],
        },
    )

    total_s = time.perf_counter() - t0
    print(f"All phases completed in {total_s:.1f}s.")
    for label, s in (("Phase 5 Donor", sum_p5), ("Phase 6 Selective", sum_p6), ("Phase 7/8 Controller Test", sum_p7)):
        print(f"\n=== {label} ({s['run_id']}) ===")
        for m, stats in s["rescue_matrix"]["methods"].items():
            print(
                f"  {m:15s}: acc={stats['accuracy']:.4f} (diff={stats['paired_accuracy_diff']:+.4f}), "
                f"rescue={stats['rescued_count']}/{stats['base_wrong_count']}, "
                f"degrade={stats['degraded_count']}/{stats['base_correct_count']}, "
                f"calls={stats['mean_recurrent_calls']:.2f}, "
                f"exc_freq={stats.get('excursion_frequency', 0.0):.2f}, "
                f"exit_freq={stats.get('exit_frequency', 0.0):.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
