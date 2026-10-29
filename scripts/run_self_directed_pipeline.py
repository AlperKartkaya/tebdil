"""End-to-end execution script for TEBDIL self-directed latent relocation phases (Phases 3-10)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from tebdil.audit import audit_run, generate_real_tensor_bundle
from tebdil.controller.counterfactual import (
    collect_counterfactual_dataset,
    train_learned_controller,
    verify_counterfactual_outcomes_unmodified,
)
from tebdil.evaluation.runner import load_adapter_and_tokenizer, run_experiment_from_config


def _write_checksums_manifest(repo_root: Path) -> Path:
    """Write SHA-256 manifest for all canonical results/ and artifacts/ files."""
    targets: List[Path] = []
    for sub in ("artifacts/counterfactuals", "artifacts/controllers", "artifacts/evidence", "results"):
        base = repo_root / sub
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.name != "checksums.sha256":
                    targets.append(p)
    lines: List[str] = []
    for p in targets:
        rel = p.relative_to(repo_root).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    out_path = repo_root / "artifacts" / "checksums.sha256"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _restrict_t2_1donor(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep visit_step=2 states and drop donor-C (d1) actions; outcome fields stay untouched."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        if int(rec.get("visit_step", 2)) != 2:
            continue
        acts = {k: v for k, v in rec["actions"].items() if int(v.get("donor_index", 0)) == 0}
        prefs = [
            p for p in rec.get("pairwise_preferences", []) if p["preferred"] in acts and p["rejected"] in acts
        ]
        out.append({**rec, "actions": acts, "pairwise_preferences": prefs})
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    t0 = time.perf_counter()

    print("[1/7] Loading real ByteDance/Ouro-1.4B-Thinking checkpoint...")
    adapter_tuple = load_adapter_and_tokenizer(
        {
            "repo": "ByteDance/Ouro-1.4B-Thinking",
            "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d",
            "precision": "bf16",
            "recurrent_steps": 4,
            "readout_mode": "last_step",
            "local_files_only": True,
        }
    )
    adapter, tokenizer, model_id, model_rev = adapter_tuple
    print(f"      Loaded {model_id} @ {model_rev} on {adapter.device}")

    print("[2/7] Generating real-model hidden-state tensor bundle for equation verification...")
    ev_dir = repo_root / "artifacts" / "evidence"
    bundle_meta = generate_real_tensor_bundle(adapter, tokenizer, ev_dir)
    print(
        f"      Saved real_tensor_bundle.pt (sha256={bundle_meta['pt_file_sha256'][:12]}..., "
        f"curvature_frob={bundle_meta['curvature_frob']:.4f})"
    )

    cf_dir = repo_root / "artifacts" / "counterfactuals"
    train_cf_path = cf_dir / "train_counterfactuals.json"
    val_cf_path = cf_dir / "val_counterfactuals.json"

    regenerate_cf = True
    if train_cf_path.exists() and val_cf_path.exists():
        tr_tmp = json.loads(train_cf_path.read_text(encoding="utf-8"))
        va_tmp = json.loads(val_cf_path.read_text(encoding="utf-8"))
        ok_tr, _ = verify_counterfactual_outcomes_unmodified(tr_tmp.get("records", []))
        ok_va, _ = verify_counterfactual_outcomes_unmodified(va_tmp.get("records", []))
        if (
            ok_tr
            and ok_va
            and tr_tmp.get("branch_count") == 3
            and tr_tmp.get("visit_steps") == [2, 3]
            and va_tmp.get("branch_count") == 3
            and va_tmp.get("visit_steps") == [2, 3]
        ):
            regenerate_cf = False
            train_cf = tr_tmp
            val_cf = va_tmp
            print(
                f"[3/7] Verified existing unmodified counterfactual datasets: "
                f"train={len(train_cf['records'])} states, val={len(val_cf['records'])} states"
            )

    if regenerate_cf:
        print(
            "[3/7] Collecting raw counterfactual datasets on train (24 problems x steps {2,3}, 2 donors) "
            "and validation (12 problems x steps {2,3}, 2 donors)..."
        )
        train_cf = collect_counterfactual_dataset(
            adapter=adapter,
            tokenizer=tokenizer,
            split="train",
            task_family="all",
            num_problems=24,
            seeds=(42,),
            recurrent_steps=4,
            fork_after_step=1,
            visit_steps=(2, 3),
            branch_count=3,
            perturbation_relative_norm=0.25,
            alpha_beta_grid=((0.65, 0.65), (0.35, 0.50), (0.85, 0.85)),
            token_scopes=("full",),
            output_path=train_cf_path,
        )
        val_cf = collect_counterfactual_dataset(
            adapter=adapter,
            tokenizer=tokenizer,
            split="validation",
            task_family="all",
            num_problems=12,
            seeds=(42,),
            recurrent_steps=4,
            fork_after_step=1,
            visit_steps=(2, 3),
            branch_count=3,
            perturbation_relative_norm=0.25,
            alpha_beta_grid=((0.65, 0.65), (0.35, 0.50), (0.85, 0.85)),
            token_scopes=("full",),
            output_path=val_cf_path,
        )
        print(
            f"      Collected {train_cf['num_states']} train states and {val_cf['num_states']} validation states."
        )

    print("[4/7] Training self-directed controllers strictly on train and calibrating on validation...")
    ctrl_dir = repo_root / "artifacts" / "controllers"
    ridge_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="ridge",
        controller_id="tebdil_ridge_v1",
        l2_reg=1.5,
        use_heuristic_gates=True,
        use_shaped_utility=True,
        seed=42,
    )
    ridge_ckpt.save(ctrl_dir / "tebdil_ridge_v1.json")

    mlp_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="mlp",
        controller_id="tebdil_mlp_v1",
        l2_reg=1.0,
        mlp_hidden_dim=16,
        mlp_epochs=300,
        mlp_lr=0.015,
        use_heuristic_gates=True,
        use_shaped_utility=True,
        seed=42,
    )
    mlp_ckpt.save(ctrl_dir / "tebdil_mlp_v1.json")

    mlp_ungated_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="mlp",
        controller_id="tebdil_mlp_ungated_v1",
        l2_reg=1.0,
        mlp_hidden_dim=16,
        mlp_epochs=300,
        mlp_lr=0.015,
        use_heuristic_gates=False,
        use_shaped_utility=True,
        seed=42,
    )
    mlp_ungated_ckpt.save(ctrl_dir / "tebdil_mlp_ungated_v1.json")

    mlp_unshaped_ckpt = train_learned_controller(
        train_records=train_cf["records"],
        val_records=val_cf["records"],
        model_type="mlp",
        controller_id="tebdil_mlp_unshaped_v1",
        l2_reg=1.0,
        mlp_hidden_dim=16,
        mlp_epochs=300,
        mlp_lr=0.015,
        use_heuristic_gates=True,
        use_shaped_utility=False,
        seed=42,
    )
    mlp_unshaped_ckpt.save(ctrl_dir / "tebdil_mlp_unshaped_v1.json")

    # E3_t2_1donor ablation: same data restricted to visit_step=2 and donor B (d0) only.
    # Branch B is seeded identically for branch_count 2 and 3, so these outcomes are exact.
    t2_train = _restrict_t2_1donor(train_cf["records"])
    t2_val = _restrict_t2_1donor(val_cf["records"])
    mlp_t2_ckpt = train_learned_controller(
        train_records=t2_train,
        val_records=t2_val,
        model_type="mlp",
        controller_id="tebdil_mlp_t2_1donor_v1",
        l2_reg=1.0,
        mlp_hidden_dim=16,
        mlp_epochs=300,
        mlp_lr=0.015,
        use_heuristic_gates=True,
        use_shaped_utility=True,
        seed=42,
    )
    mlp_t2_ckpt.training_metadata["derived_from"] = (
        "train/val counterfactuals filtered to visit_step=2 and donor_index=0"
    )
    mlp_t2_ckpt.save(ctrl_dir / "tebdil_mlp_t2_1donor_v1.json")

    for ck in (ridge_ckpt, mlp_ckpt, mlp_ungated_ckpt, mlp_unshaped_ckpt, mlp_t2_ckpt):
        print(
            f"      {ck.controller_id}: train_mse={ck.training_metadata['train_mse']:.4f}, "
            f"val_mse={ck.training_metadata['val_mse']:.4f}, "
            f"rank_acc={ck.training_metadata['pairwise_ranking_accuracy']:.3f}, "
            f"exit_th={ck.exit_min_advantage:.4f}, exc_th={ck.excursion_min_advantage:.4f}, "
            f"val_actions={ck.training_metadata['calibration_summary']['action_counts']}"
        )

    shared_cache: Dict[str, Dict[str, Any]] = {}

    print("[5/7] Running Phase 5 (Donor Selection on validation) & Phase 6 (Selective Update on validation)...")
    s5 = run_experiment_from_config(
        config_path=repo_root / "configs" / "phase5_donor.yaml",
        repo_root=repo_root,
        adapter_override=adapter_tuple,
        shared_eval_cache=shared_cache,
    )
    print(
        f"      Phase 5 done ({s5['manifest']['runtime_seconds']}s): "
        + ", ".join(
            f"{m}={v['accuracy']*100:.1f}%({v['mean_recurrent_calls']:.1f}c)"
            for m, v in s5["rescue_matrix"]["methods"].items()
        )
    )

    s6 = run_experiment_from_config(
        config_path=repo_root / "configs" / "phase6_selective.yaml",
        repo_root=repo_root,
        adapter_override=adapter_tuple,
        shared_eval_cache=shared_cache,
    )
    print(
        f"      Phase 6 done ({s6['manifest']['runtime_seconds']}s): "
        + ", ".join(
            f"{m}={v['accuracy']*100:.1f}%({v['mean_recurrent_calls']:.1f}c)"
            for m, v in s6["rescue_matrix"]["methods"].items()
        )
    )

    print("[6/7] Running Phase 7/8 Exploratory Test & Confirmatory Fresh-Test Benchmarks (3 seeds x 18 problems)...")
    s78 = run_experiment_from_config(
        config_path=repo_root / "configs" / "phase7_8_controller.yaml",
        repo_root=repo_root,
        adapter_override=adapter_tuple,
        shared_eval_cache=shared_cache,
    )
    print(f"      Phase 7/8 (exploratory test) done ({s78['manifest']['runtime_seconds']}s):")
    for m, v in s78["rescue_matrix"]["methods"].items():
        print(
            f"        {m:16s} | acc={v['accuracy']*100:5.1f}% | diff={v['paired_accuracy_diff']*100:+5.1f}% "
            f"| rescue={v['rescued_count']}/{v['base_wrong_count']} | deg={v['degraded_count']}/{v['base_correct_count']} "
            f"| calls={v['mean_recurrent_calls']:5.2f} (vs S0 {v['call_delta_vs_s0_pct']:+.1f}%, vs E1 {v['call_delta_vs_e1_pct']:+.1f}%) "
            f"| exc={v['excursion_frequency']*100:4.1f}% | exit={v['exit_frequency']*100:4.1f}%"
        )

    s_fresh = run_experiment_from_config(
        config_path=repo_root / "configs" / "phase7_8_fresh_test.yaml",
        repo_root=repo_root,
        adapter_override=adapter_tuple,
        shared_eval_cache=shared_cache,
    )
    print(f"      Phase 7/8 (confirmatory fresh_test) done ({s_fresh['manifest']['runtime_seconds']}s):")
    for m, v in s_fresh["rescue_matrix"]["methods"].items():
        print(
            f"        {m:16s} | acc={v['accuracy']*100:5.1f}% | diff={v['paired_accuracy_diff']*100:+5.1f}% "
            f"| rescue={v['rescued_count']}/{v['base_wrong_count']} | deg={v['degraded_count']}/{v['base_correct_count']} "
            f"| calls={v['mean_recurrent_calls']:5.2f} (vs S0 {v['call_delta_vs_s0_pct']:+.1f}%, vs E1 {v['call_delta_vs_e1_pct']:+.1f}%) "
            f"| exc={v['excursion_frequency']*100:4.1f}% | exit={v['exit_frequency']*100:4.1f}%"
        )

    print("[7/7] Running cryptographic + real-model replay audits and writing checksums...")
    audit_rep = audit_run(
        run_id="phase7_8_controller_test",
        repo_root=repo_root,
        replay_problem_id="arith_test_003",
        require_real_replay=True,
    )
    audit_path = ev_dir / "audit_phase7_8_report.json"
    audit_path.write_text(json.dumps(audit_rep, indent=2), encoding="utf-8")

    audit_fresh = audit_run(
        run_id="phase7_8_fresh_test",
        repo_root=repo_root,
        replay_problem_id="arith_fresh_test_001",
        require_real_replay=True,
    )
    audit_fresh_path = ev_dir / "audit_fresh_test_report.json"
    audit_fresh_path.write_text(json.dumps(audit_fresh, indent=2), encoding="utf-8")

    chk_path = _write_checksums_manifest(repo_root)
    print(
        f"      Audit exploratory all_passed={audit_rep['all_passed']} "
        f"(replayed={audit_rep['checks']['9_deterministic_replay'].get('num_cases_replayed', 0)} cases)"
    )
    print(
        f"      Audit fresh_test all_passed={audit_fresh['all_passed']} "
        f"(replayed={audit_fresh['checks']['9_deterministic_replay'].get('num_cases_replayed', 0)} cases)"
    )
    print(f"      Wrote checksum manifest to {chk_path.relative_to(repo_root)}")
    print(f"Total pipeline runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
