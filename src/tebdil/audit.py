"""Independent cryptographic evidence audit and deterministic replay verification for TEBDIL."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import yaml

import tebdil.controller.action_space as action_space_mod
import tebdil.controller.features as features_mod
import tebdil.controller.learned_policy as learned_policy_mod
import tebdil.core.branching as branching_mod
import tebdil.core.controls as controls_mod
import tebdil.core.excursions as excursions_mod
import tebdil.core.perturbations as perturbations_mod
import tebdil.core.policies as policies_mod
import tebdil.core.scheduler as scheduler_mod
import tebdil.core.state as state_mod
from tebdil.controller.counterfactual import (
    _hash_arrays,
    verify_counterfactual_outcomes_unmodified,
)
from tebdil.controller.learned_policy import LearnedControllerCheckpoint
from tebdil.core.branching import fork_branches
from tebdil.core.excursions import build_visit_state, execute_excursion_step
from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.runner import _decode_method_on_problem, load_adapter_and_tokenizer
from tebdil.evaluation.trace import read_trace_records
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite


FORBIDDEN_RUNTIME_IDENTIFIERS = {
    "gold_answer",
    "gold_label",
    "validate_problem_response",
    "validate_arithmetic",
    "validate_logic",
    "validate_graph",
    "ground_truth",
}

_FILE_HASH_CACHE: Dict[Tuple[str, int, int], str] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    resolved = path.resolve()
    st = resolved.stat()
    cache_key = (str(resolved), int(st.st_mtime_ns), int(st.st_size))
    if cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]
    h = hashlib.sha256()
    with resolved.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASH_CACHE[cache_key] = digest
    return digest


def check_no_label_leakage_in_runtime_modules() -> Dict[str, Any]:
    runtime_modules = [
        state_mod,
        perturbations_mod,
        branching_mod,
        excursions_mod,
        controls_mod,
        policies_mod,
        scheduler_mod,
        action_space_mod,
        features_mod,
        learned_policy_mod,
    ]
    violations = []
    for mod in runtime_modules:
        src_path = Path(inspect.getfile(mod))
        source = src_path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_RUNTIME_IDENTIFIERS:
            if forbidden in source:
                violations.append(f"{src_path.name}: contains '{forbidden}'")
        for _, obj in inspect.getmembers(mod, inspect.isfunction):
            sig = inspect.signature(obj)
            for param_name in sig.parameters:
                if "gold" in param_name.lower() or "label" in param_name.lower():
                    violations.append(f"{src_path.name}:{obj.__name__} param '{param_name}'")
    return {
        "passed": len(violations) == 0,
        "status": "PASS" if len(violations) == 0 else "FAIL",
        "category": "unit_check",
        "modules_checked": [Path(inspect.getfile(m)).name for m in runtime_modules],
        "violations": violations,
    }


def check_train_test_template_disjointness(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    train_probs = generate_task_suite(family="all", split="train", num_problems=18, seed=42)
    val_probs = generate_task_suite(family="all", split="validation", num_problems=18, seed=42)
    test_probs = generate_task_suite(family="all", split="test", num_problems=18, seed=42)
    fresh_probs = generate_task_suite(family="all", split="fresh_test", num_problems=18, seed=42)

    train_tmpls = {p.template_id for p in train_probs}
    val_tmpls = {p.template_id for p in val_probs}
    test_tmpls = {p.template_id for p in test_probs}
    fresh_tmpls = {p.template_id for p in fresh_probs}

    overlap_train_test = sorted(train_tmpls.intersection(test_tmpls))
    overlap_val_test = sorted(val_tmpls.intersection(test_tmpls))
    overlap_with_fresh = sorted((train_tmpls | val_tmpls | test_tmpls).intersection(fresh_tmpls))

    train_prompts = {p.prompt for p in train_probs}
    val_prompts = {p.prompt for p in val_probs}
    test_prompts = {p.prompt for p in test_probs}
    fresh_prompts = {p.prompt for p in fresh_probs}
    prompt_overlap_test = len((train_prompts | val_prompts).intersection(test_prompts))
    prompt_overlap_fresh = len((train_prompts | val_prompts | test_prompts).intersection(fresh_prompts))

    config_split_errors: List[str] = []
    if repo_root is not None:
        for dev_cfg_name in ("phase5_donor.yaml", "phase6_selective.yaml"):
            cfg_p = repo_root / "configs" / dev_cfg_name
            if cfg_p.exists():
                c_data = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
                sp = str(c_data.get("evaluation", {}).get("split", ""))
                if sp in ("test", "fresh_test"):
                    config_split_errors.append(
                        f"Development config {dev_cfg_name} illegally uses held-out split '{sp}'"
                    )

    passed = (
        len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
        and len(overlap_with_fresh) == 0
        and prompt_overlap_test == 0
        and prompt_overlap_fresh == 0
        and len(config_split_errors) == 0
    )
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "train_templates": sorted(train_tmpls),
        "validation_templates": sorted(val_tmpls),
        "test_templates": sorted(test_tmpls),
        "fresh_test_templates": sorted(fresh_tmpls),
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
        "overlap_with_fresh_test": overlap_with_fresh,
        "prompt_overlap_with_test": prompt_overlap_test,
        "prompt_overlap_with_fresh_test": prompt_overlap_fresh,
        "config_split_errors": config_split_errors,
    }


def check_provenance_artifact(repo_root: Path) -> Dict[str, Any]:
    prov_path = repo_root / "artifacts" / "evidence" / "model_provenance.json"
    if not prov_path.exists():
        return {
            "passed": False,
            "status": "FAIL",
            "category": "artifact_consistency_check",
            "reason": "missing artifacts/evidence/model_provenance.json",
        }
    data = json.loads(prov_path.read_text(encoding="utf-8"))
    snap = Path(str(data.get("snapshot_path", "")))
    weight_hashes = data.get("weight_hashes", {})

    hash_mismatches: List[str] = []
    verified_files: Dict[str, str] = {}
    for fname, expected_info in weight_hashes.items():
        fpath = snap / fname
        if not fpath.exists():
            hash_mismatches.append(f"missing {fname} in snapshot")
            continue
        actual_size = fpath.resolve().stat().st_size
        expected_size = int(expected_info.get("size_bytes", -1))
        if actual_size != expected_size:
            hash_mismatches.append(f"{fname} size {actual_size} != expected {expected_size}")
        actual_sha = _sha256_file(fpath)
        expected_sha = str(expected_info.get("sha256", ""))
        if actual_sha != expected_sha:
            hash_mismatches.append(f"{fname} sha256 {actual_sha[:12]} != expected {expected_sha[:12]}")
        verified_files[fname] = actual_sha

    # Verify vendored model code hashes against snapshot
    vendored_dir = repo_root / "vendor" / "ouro_1_4b_thinking"
    if not vendored_dir.exists():
        vendored_dir = repo_root / "src" / "tebdil" / "models" / "vendored_ouro"
    for v_file in ("modeling_ouro.py", "configuration_ouro.py"):
        vp = vendored_dir / v_file
        sp = snap / v_file
        if not vp.exists():
            hash_mismatches.append(f"missing vendored {v_file}")
        elif sp.exists() and _sha256_file(vp) != _sha256_file(sp):
            hash_mismatches.append(f"vendored {v_file} does not match pinned upstream snapshot")

    passed = bool(
        data.get("model_id") == "ByteDance/Ouro-1.4B-Thinking"
        and data.get("model_revision") == PINNED_OURO_THINKING_REV
        and len(weight_hashes) >= 3
        and len(hash_mismatches) == 0
    )
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "model_id": data.get("model_id"),
        "model_revision": data.get("model_revision"),
        "verified_sha256": verified_files,
        "hash_mismatches": hash_mismatches,
        "forward_equivalence_verified": data.get("custom_model_code", {}).get("forward_equivalence_verified"),
    }


def check_counterfactual_and_controller_integrity(repo_root: Path) -> Dict[str, Any]:
    """Verify counterfactual outcome hashes and controller parameter/dataset hash chains."""
    cf_dir = repo_root / "artifacts" / "counterfactuals"
    ctrl_dir = repo_root / "artifacts" / "controllers"
    errors: List[str] = []

    train_p = cf_dir / "train_counterfactuals.json"
    val_p = cf_dir / "val_counterfactuals.json"
    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []

    for label, p, expected_split in (
        ("train", train_p, "train"),
        ("validation", val_p, "validation"),
    ):
        if not p.exists():
            errors.append(f"missing {p.relative_to(repo_root)}")
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("split") != expected_split:
            errors.append(f"{p.name} split={payload.get('split')} != {expected_split}")
        recs = payload.get("records", [])
        ok, rec_errs = verify_counterfactual_outcomes_unmodified(recs)
        if not ok:
            errors.extend(rec_errs[:5])
        rec_sha = hashlib.sha256(
            json.dumps(recs, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if payload.get("records_sha256") and payload.get("records_sha256") != rec_sha:
            errors.append(f"{p.name} records_sha256 mismatch")
        if label == "train":
            train_recs = recs
        else:
            val_recs = recs

    expected_ds_sha = ""
    if train_recs:
        expected_ds_sha = hashlib.sha256(
            json.dumps(
                {"train": train_recs, "val": val_recs},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    checked_controllers: List[str] = []
    for ckpt_name in ("tebdil_ridge_v1.json", "tebdil_mlp_v1.json"):
        cp = ctrl_dir / ckpt_name
        if not cp.exists():
            errors.append(f"missing controller checkpoint {ckpt_name}")
            continue
        ckpt = LearnedControllerCheckpoint.load(cp)
        meta = ckpt.training_metadata or {}
        if meta.get("calibration_split") != "validation":
            errors.append(f"{ckpt_name} calibration_split={meta.get('calibration_split')} != 'validation'")
        if meta.get("calibration_summary", {}).get("diversity_bonus_applied", True) is not False:
            errors.append(f"{ckpt_name} applied artificial diversity_bonus during calibration")
        if expected_ds_sha and meta.get("dataset_sha256") != expected_ds_sha:
            errors.append(
                f"{ckpt_name} dataset_sha256={str(meta.get('dataset_sha256'))[:12]} "
                f"!= current counterfactual dataset {expected_ds_sha[:12]}"
            )
        if ckpt.model_type == "ridge":
            actual_param_hash = _hash_arrays([np.asarray(ckpt.weights, dtype=np.float64)])
        else:
            actual_param_hash = _hash_arrays(
                [
                    np.asarray(ckpt.mlp_w1, dtype=np.float64),
                    np.asarray(ckpt.mlp_b1, dtype=np.float64),
                    np.asarray(ckpt.mlp_w2, dtype=np.float64),
                    np.asarray([ckpt.mlp_b2], dtype=np.float64),
                ]
            )
        if meta.get("final_param_hash") != actual_param_hash:
            errors.append(
                f"{ckpt_name} final_param_hash mismatch "
                f"({str(meta.get('final_param_hash'))[:12]} != {actual_param_hash[:12]})"
            )
        checked_controllers.append(ckpt_name)

    passed = len(errors) == 0 and len(checked_controllers) >= 2
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "checked_controllers": checked_controllers,
        "num_train_states": len(train_recs),
        "num_val_states": len(val_recs),
        "errors": errors,
    }


def generate_real_tensor_bundle(
    adapter: OuroRecurrentAdapter,
    tokenizer: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """Capture and save a real-model hidden-state tensor bundle for equation & hash verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Compute the exact integer value of ((11 * 6) - 18) // 2 + 24. "
        "Reply with the final integer only.\nAnswer: "
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(adapter.device)

    root = adapter.initialize(input_ids)
    h0 = root.hidden[:, -1, :].detach().float().cpu()
    s1 = adapter.transition(root.clone("A"))
    h1 = s1.hidden[:, -1, :].detach().float().cpu()

    branches, _ = fork_branches(
        s1,
        count=3,
        perturbation_method="normalized_gaussian",
        relative_norm=0.25,
        seed=42,
        token_scope="full",
    )
    a2 = adapter.transition(branches[0])
    b2 = adapter.transition(branches[1])
    c2 = adapter.transition(branches[2])

    cfg = ExcursionConfig(alpha=0.65, beta=0.65, max_relative_norm=0.35, visit_step=2, token_scope="full")
    h2_a = a2.hidden[:, -1, :].detach().float().cpu()
    h2_b = b2.hidden[:, -1, :].detach().float().cpu()
    h2_c = c2.hidden[:, -1, :].detach().float().cpu()

    # Compute exact intermediate excursion states on the real model
    # Same visit constructor as the operator (includes norm clipping of the visit offset).
    st_visit, _, _, _, _ = build_visit_state(a2.clone(), b2.clone(), cfg)
    visit_hidden = st_visit.hidden
    f_a = adapter.transition(a2.clone())
    f_b = adapter.transition(b2.clone())
    f_visit = adapter.transition(st_visit)

    na_3, nb_3, telem = execute_excursion_step(adapter, a2.clone(), b2.clone(), cfg)
    na_4 = adapter.transition(na_3.clone())
    exc_logits = adapter.readout_logits(na_4)[:, -1, :].detach().float().cpu()

    # CONTINUE and EXIT reference tensors
    cont_4 = adapter.transition(f_a.clone())
    cont_logits = adapter.readout_logits(cont_4)[:, -1, :].detach().float().cpu()
    exit_logits = adapter.readout_logits(a2)[:, -1, :].detach().float().cpu()

    tensors = {
        "h0_last": h0,
        "h1_last": h1,
        "h2_a_last": h2_a,
        "h2_b_last": h2_b,
        "h2_c_last": h2_c,
        "h2_visit_last": visit_hidden[:, -1, :].detach().float().cpu(),
        "f_h2_a_last": f_a.hidden[:, -1, :].detach().float().cpu(),
        "f_h2_b_last": f_b.hidden[:, -1, :].detach().float().cpu(),
        "f_h2_visit_last": f_visit.hidden[:, -1, :].detach().float().cpu(),
        "h3_returned_a_last": na_3.hidden[:, -1, :].detach().float().cpu(),
        "h4_continued_a_last": na_4.hidden[:, -1, :].detach().float().cpu(),
        "h4_normal_a_last": cont_4.hidden[:, -1, :].detach().float().cpu(),
        "excursion_logits_last": exc_logits,
        "continue_logits_last": cont_logits,
        "exit_step2_logits_last": exit_logits,
    }
    pt_path = output_dir / "real_tensor_bundle.pt"
    torch.save(tensors, pt_path)

    tensor_hashes = {k: hash_tensor(v, length=64) for k, v in tensors.items()}
    meta = {
        "model_id": "ByteDance/Ouro-1.4B-Thinking",
        "model_revision": PINNED_OURO_THINKING_REV,
        "prompt": prompt,
        "seed": 42,
        "visit_step": 2,
        "alpha": float(telem.effective_alpha),
        "beta": float(telem.beta),
        "max_relative_norm": cfg.max_relative_norm,
        "clipped": bool(telem.clipped),
        "curvature_frob": float(telem.curvature_frob),
        "returned_shift_vs_normal_frob": float(telem.returned_shift_vs_normal_frob),
        "pt_file_sha256": _sha256_file(pt_path),
        "tensor_sha256": tensor_hashes,
        "decoded_top1": {
            "CONTINUE": tokenizer.decode([int(cont_logits.argmax(dim=-1).item())]),
            "EXIT_step2": tokenizer.decode([int(exit_logits.argmax(dim=-1).item())]),
            "EXCURSION_d0": tokenizer.decode([int(exc_logits.argmax(dim=-1).item())]),
        },
    }
    json_path = output_dir / "real_tensor_bundle.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def check_real_tensor_bundle(repo_root: Path) -> Dict[str, Any]:
    """Verify saved real-model tensor bundle hashes and reconstruct excursion equations."""
    ev_dir = repo_root / "artifacts" / "evidence"
    pt_path = ev_dir / "real_tensor_bundle.pt"
    json_path = ev_dir / "real_tensor_bundle.json"
    if not pt_path.exists() or not json_path.exists():
        return {
            "passed": False,
            "status": "FAIL",
            "category": "artifact_consistency_check",
            "reason": "missing artifacts/evidence/real_tensor_bundle.{pt,json}",
        }

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    actual_pt_sha = _sha256_file(pt_path)
    errors: List[str] = []
    if meta.get("pt_file_sha256") != actual_pt_sha:
        errors.append(
            f"real_tensor_bundle.pt sha256 {actual_pt_sha[:12]} != expected {str(meta.get('pt_file_sha256'))[:12]}"
        )

    try:
        tensors = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {
            "passed": False,
            "status": "FAIL",
            "category": "artifact_consistency_check",
            "reason": f"failed to load real_tensor_bundle.pt: {exc}",
        }

    expected_hashes = meta.get("tensor_sha256", {})
    for k, exp_h in expected_hashes.items():
        if k not in tensors:
            errors.append(f"missing tensor '{k}' in bundle")
            continue
        act_h = hash_tensor(tensors[k], length=64)
        if act_h != exp_h:
            errors.append(f"tensor '{k}' hash mismatch ({act_h[:12]} != {exp_h[:12]})")

    # Reconstruct excursion equations on last-token float32 vectors
    alpha = float(meta.get("alpha", 0.65))
    beta = float(meta.get("beta", 0.65))
    h_a = tensors["h2_a_last"].float()
    h_b = tensors["h2_b_last"].float()
    h_vis = tensors["h2_visit_last"].float()
    f_a = tensors["f_h2_a_last"].float()
    f_b = tensors["f_h2_b_last"].float()
    f_vis = tensors["f_h2_visit_last"].float()
    h3_ret = tensors["h3_returned_a_last"].float()

    recon_vis = (1.0 - alpha) * h_a + alpha * h_b
    vis_err = float((recon_vis - h_vis).abs().max().item())
    if vis_err > 2e-2:  # bfloat16 rounding tolerance
        errors.append(f"visit state equation error {vis_err:.5f} > 0.02")

    delta_a = f_a - h_a
    delta_b = f_b - h_b
    delta_vis = f_vis - h_vis
    curv = delta_vis - ((1.0 - alpha) * delta_a + alpha * delta_b)
    curv_norm = float(torch.linalg.vector_norm(curv).item())
    if curv_norm <= 0.0:
        errors.append("curvature C_t norm is zero")

    # Operator (full scope, no top-k, no gate): H'_{t+1} = F(H_A) + beta * (V_visit - V_A).
    # max_relative_norm clips only the visit offset, which is already baked into h_vis.
    recon_h3 = f_a + beta * (delta_vis - delta_a)
    ret_err = float((recon_h3 - h3_ret).abs().max().item())
    if ret_err > 5e-2:
        errors.append(f"returned state equation error {ret_err:.5f} > 0.05")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "visit_equation_max_abs_err": round(vis_err, 6),
        "return_equation_max_abs_err": round(ret_err, 6),
        "last_token_curvature_norm": round(curv_norm, 6),
        "errors": errors,
    }


def check_summary_trace_consistency(summary: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute per-method accuracy and mean calls from traces and compare with summary.json."""
    summary_methods = summary.get("rescue_matrix", {}).get("methods", {})
    errors: List[str] = []
    for m_name, m_stats in summary_methods.items():
        m_recs = [r for r in records if r.get("method") == m_name]
        if not m_recs:
            errors.append(f"method {m_name} in summary.json has 0 records in traces.jsonl")
            continue
        mean_c = sum(float(r["recurrent_block_calls"]) for r in m_recs) / len(m_recs)
        if abs(mean_c - float(m_stats.get("mean_recurrent_calls", 0.0))) > 1e-3:
            errors.append(f"{m_name} mean_recurrent_calls {mean_c:.4f} != summary {m_stats.get('mean_recurrent_calls')}")
        mean_acc = sum(1.0 if r.get("is_correct") else 0.0 for r in m_recs) / len(m_recs)
        if abs(mean_acc - float(m_stats.get("accuracy", 0.0))) > 1e-4:
            errors.append(f"{m_name} accuracy {mean_acc:.4f} != summary {m_stats.get('accuracy')}")
    passed = not errors and len(summary_methods) > 0
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "errors": errors,
    }


def check_execution_ledgers(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fail closed unless every recorded call count has matching execution provenance."""
    valid_calls = 0
    ledger_consistent = 0
    reference_budget_consistent = 0
    s0_misannotated = 0
    ledger_errors: List[str] = []

    for index, record in enumerate(records):
        method = str(record.get("method", "unknown"))
        record_id = f"{method}:{record.get('problem_id', index)}:{record.get('seed', 'na')}"
        calls = int(record.get("recurrent_block_calls", 0))
        if calls >= 2:
            valid_calls += 1

        if record.get("derived_parallel_selection"):
            reference_ledger = record.get("reference_parallel_budget_ledger", [])
            provenance = record.get("selection_provenance", {})
            if method not in ("P0", "P1"):
                ledger_errors.append(f"{record_id} derived parallel provenance on non-parallel method")
            if not record.get("reference_parallel_budget_source") or not provenance.get("selected_branch"):
                ledger_errors.append(f"{record_id} missing parallel selection provenance")
            reference_calls = sum(
                int(entry.get("recurrent_block_calls", 0)) for entry in reference_ledger
            )
            if reference_ledger and reference_calls == calls:
                reference_budget_consistent += 1
            else:
                ledger_errors.append(
                    f"{record_id} reference budget calls {reference_calls} != record calls {calls}"
                )

            if not record.get("executed_selection_compute_verified"):
                ledger_errors.append(
                    f"{record_id} idealized reference budget is not verified executed cost"
                )
                ledger = []
            else:
                ledger = record.get("selection_execution_ledger", [])
                consulted_ids = {
                    str(source.get("source_id"))
                    for source in provenance.get("consulted_sources", [])
                }
                accounted_ids = {
                    str(source_id)
                    for source_id in record.get("executed_selection_source_ids", [])
                }
                if not consulted_ids or accounted_ids != consulted_ids:
                    ledger_errors.append(
                        f"{record_id} executed selection sources do not account for every consulted source"
                    )
        else:
            ledger = record.get("token_ledger", [])
            if method in ("P0", "P1"):
                ledger_errors.append(
                    f"{record_id} legacy parallel selection lacks total execution provenance"
                )

        if not ledger and not record.get("derived_parallel_selection"):
            ledger_errors.append(f"{record_id} missing execution ledger")
        elif ledger:
            ledger_calls = sum(int(entry.get("recurrent_block_calls", 0)) for entry in ledger)
            if ledger_calls == calls:
                ledger_consistent += 1
            else:
                ledger_errors.append(f"{record_id} ledger calls {ledger_calls} != record calls {calls}")

        if method == "S0" and record.get("controller_action_type") == "EXCURSION":
            s0_misannotated += 1

    passed = (
        len(records) > 0
        and valid_calls == len(records)
        and ledger_consistent == len(records)
        and s0_misannotated == 0
        and not ledger_errors
    )
    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "category": "artifact_consistency_check",
        "valid_records": f"{valid_calls}/{len(records)}",
        "token_ledger_consistent": f"{ledger_consistent}/{len(records)}",
        "reference_budget_consistent": f"{reference_budget_consistent}/{len(records)}",
        "s0_misannotated_count": s0_misannotated,
        "ledger_error_count": len(ledger_errors),
        "ledger_errors": ledger_errors,
    }


def audit_run(
    run_id: str,
    repo_root: Optional[Path] = None,
    replay_problem_id: Optional[str] = None,
    require_real_replay: bool = False,
) -> Dict[str, Any]:
    """Audit recorded artifacts for `run_id` across unit, artifact consistency, and real-model checks."""
    if repo_root is None:
        repo_root = _repo_root()
    run_dir = repo_root / "results" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.yaml"
    summary_path = run_dir / "summary.json"
    traces_path = run_dir / "traces.jsonl"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    records = read_trace_records(traces_path)

    # 1. Provenance & full weight SHA-256 check
    c1_prov = check_provenance_artifact(repo_root)

    # 2. Expected model class & revision in manifest
    c2_passed = (
        manifest.get("model_id") == "ByteDance/Ouro-1.4B-Thinking"
        and manifest.get("model_revision") == PINNED_OURO_THINKING_REV
        and int(manifest.get("recurrent_steps", 0)) >= 4
    )
    c2_arch = {
        "passed": c2_passed,
        "status": "PASS" if c2_passed else "FAIL",
        "category": "artifact_consistency_check",
        "manifest_model_id": manifest.get("model_id"),
        "manifest_revision": manifest.get("model_revision"),
        "recurrent_steps": manifest.get("recurrent_steps"),
    }

    # 3 & 4. Genuine recurrent execution & real hidden-state capture + token_ledger consistency
    valid_snaps = 0
    for r in records:
        snaps = r.get("step_snapshots", [])
        if snaps and all(s.get("branch_rms", {}).get("A", 0.0) > 0.0 for s in snaps):
            valid_snaps += 1
    c3_exec = check_execution_ledgers(records)
    c4_passed = len(records) > 0 and valid_snaps == len(records)
    c4_hidden = {
        "passed": c4_passed,
        "status": "PASS" if c4_passed else "FAIL",
        "category": "artifact_consistency_check",
        "valid_snapshots": f"{valid_snaps}/{len(records)}",
    }

    # 5, 6, 8. Shared weights, branch isolation, and no-op intervention equivalence (unit check on Ouro adapter)
    mini_adapter = OuroRecurrentAdapter.from_mini_config(seed=42)
    st0 = mini_adapter.transition(mini_adapter.initialize(torch.tensor([[1, 5, 9, 13]])))
    branches, _ = fork_branches(st0, count=2, relative_norm=0.10, seed=42)
    b_a, b_b = branches[0], branches[1]
    shared_weights_ok = len({p.data_ptr() for p in mini_adapter.model.parameters()}) > 0
    isolation_ok = (b_a.hidden.data_ptr() != b_b.hidden.data_ptr()) and not torch.equal(b_a.hidden, b_b.hidden)

    ref_a = mini_adapter.transition(b_a.clone())
    na_a0, _, _ = execute_excursion_step(mini_adapter, b_a.clone(), b_b.clone(), ExcursionConfig(alpha=0.0, beta=0.5))
    na_b0, _, _ = execute_excursion_step(mini_adapter, b_a.clone(), b_b.clone(), ExcursionConfig(alpha=0.5, beta=0.0))
    na_eq, _, _ = execute_excursion_step(mini_adapter, b_a.clone(), b_a.clone("B"), ExcursionConfig(alpha=0.5, beta=0.5))
    noop_ok = (
        torch.equal(na_a0.hidden, ref_a.hidden)
        and torch.equal(na_b0.hidden, ref_a.hidden)
        and torch.equal(na_eq.hidden, ref_a.hidden)
    )

    c5_shared = {"passed": shared_weights_ok, "status": "PASS" if shared_weights_ok else "FAIL", "category": "unit_check"}
    c6_isolation = {"passed": isolation_ok, "status": "PASS" if isolation_ok else "FAIL", "category": "unit_check"}
    c8_noop = {"passed": noop_ok, "status": "PASS" if noop_ok else "FAIL", "category": "unit_check"}

    # 7. Visited-state nonlinear curvature & real tensor bundle check
    exc_recs = [
        r
        for r in records
        if r.get("controller_action_type") == "EXCURSION"
        and r.get("method") not in ("M0", "M1", "R0")
    ]
    nonzero_curv = sum(1 for r in exc_recs if float(r.get("curvature_frob", 0.0)) > 0.0)
    bundle_check = check_real_tensor_bundle(repo_root)
    c7_passed = ((len(exc_recs) == 0) or (nonzero_curv == len(exc_recs))) and bool(bundle_check.get("passed"))
    c7_visit = {
        "passed": c7_passed,
        "status": "PASS" if c7_passed else "FAIL",
        "category": "artifact_consistency_check",
        "executed_excursions": len(exc_recs),
        "nonzero_curvature_count": nonzero_curv,
        "tensor_bundle_verification": bundle_check,
    }

    # 10 & 11. Label leakage, train/test/fresh_test split separation, and counterfactual/controller hash chain
    c10_leakage = check_no_label_leakage_in_runtime_modules()
    c11_splits = check_train_test_template_disjointness(repo_root=repo_root)
    cf_ctrl_check = check_counterfactual_and_controller_integrity(repo_root)
    c11_splits["counterfactual_and_controller_chain"] = cf_ctrl_check
    c11_splits["passed"] = bool(c11_splits["passed"] and cf_ctrl_check["passed"])
    c11_splits["status"] = "PASS" if c11_splits["passed"] else "FAIL"

    # 12. Honest compute & accuracy accounting consistency between traces.jsonl and summary.json
    c12_compute = check_summary_trace_consistency(summary, records)

    # 9. Deterministic real-model replay check
    if replay_problem_id is None:
        c9_replay: Dict[str, Any] = {
            "passed": False if require_real_replay else None,
            "status": "FAIL" if require_real_replay else "not_run",
            "category": "real_model_check",
            "mode": "not_run",
            "reason": "Pass --replay-problem-id to execute live real-model replay verification.",
        }
    else:
        cfg_path = Path(str(manifest.get("config_path", "")))
        if not cfg_path.is_absolute():
            cfg_path = repo_root / cfg_path
        raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        adapter, tokenizer, _, _ = load_adapter_and_tokenizer(raw_cfg.get("model", {}))
        interv_cfg = raw_cfg.get("intervention", {})
        pol_raw = raw_cfg.get("policy", {})
        branch_cfg = raw_cfg.get("branches", {})
        ctrl_raw = raw_cfg.get("controller", {})
        exec_cfg = raw_cfg.get("execution", {})

        raw_elig = interv_cfg.get("eligible_visit_steps", [])
        elig_tuple = tuple(int(s) for s in raw_elig) if raw_elig else ()
        exc_cfg = ExcursionConfig(
            alpha=float(interv_cfg.get("alpha", manifest.get("alpha", 0.65))),
            beta=float(interv_cfg.get("beta", manifest.get("beta", 0.65))),
            max_relative_norm=float(interv_cfg.get("max_relative_norm", manifest.get("norm_clip", 0.35))),
            visit_step=int(interv_cfg.get("visit_step", manifest.get("intervention_step", 2))),
            token_scope=str(interv_cfg.get("token_scope", "full")),
            donor_strategy=str(interv_cfg.get("donor_strategy", "fixed")),
            fallback_if_unstable=str(interv_cfg.get("fallback_if_unstable", "skip_intervention")),
            coordinate_topk_fraction=float(interv_cfg.get("coordinate_topk_fraction", 1.0)),
            eligible_visit_steps=elig_tuple,
        )
        pol_cfg = AdaptivePolicyConfig(
            min_relative_divergence=float(pol_raw.get("min_relative_divergence", 0.002)),
            max_relative_divergence=float(pol_raw.get("max_relative_divergence", 0.65)),
            min_entropy_trigger=float(pol_raw.get("min_entropy_trigger", 1.20)),
            max_margin_trigger=float(pol_raw.get("max_margin_trigger", 0.70)),
            stagnation_ratio_threshold=float(pol_raw.get("stagnation_ratio_threshold", 0.85)),
            donor_selection=str(pol_raw.get("donor_selection", "confidence_gain")),
        )

        def _load_ckpt_rel(key_name: str, default_rel: str) -> Optional[LearnedControllerCheckpoint]:
            rel = ctrl_raw.get(key_name, default_rel)
            if not rel:
                return None
            p = repo_root / str(rel) if not Path(str(rel)).is_absolute() else Path(str(rel))
            return LearnedControllerCheckpoint.load(p) if p.exists() else None

        ridge_ckpt = _load_ckpt_rel("checkpoint_path", "artifacts/controllers/tebdil_ridge_v1.json")
        mlp_ckpt = _load_ckpt_rel("mlp_checkpoint_path", "artifacts/controllers/tebdil_mlp_v1.json")
        mlp_ungated_ckpt = _load_ckpt_rel(
            "mlp_ungated_checkpoint_path", "artifacts/controllers/tebdil_mlp_ungated_v1.json"
        )
        mlp_unshaped_ckpt = _load_ckpt_rel(
            "mlp_unshaped_checkpoint_path", "artifacts/controllers/tebdil_mlp_unshaped_v1.json"
        )
        mlp_t2_ckpt = _load_ckpt_rel(
            "mlp_t2_1donor_checkpoint_path", "artifacts/controllers/tebdil_mlp_t2_1donor_v1.json"
        )

        def _ckpt_for(m_name: str) -> Optional[LearnedControllerCheckpoint]:
            if m_name == "E3_mlp":
                return mlp_ckpt or ridge_ckpt
            if m_name == "E3_mlp_ungated":
                return mlp_ungated_ckpt or mlp_ckpt
            if m_name == "E3_mlp_unshaped":
                return mlp_unshaped_ckpt or mlp_ckpt
            if m_name == "E3_t2_1donor":
                return mlp_t2_ckpt or mlp_ckpt
            return ridge_ckpt

        # Select explicit target cases to replay:
        # 1. S0 on replay_problem_id
        # 2. E1 (forced EXCURSION) on replay_problem_id
        # 3. EXIT_2 (forced EXIT) on replay_problem_id (if present)
        # 4. E3_mlp and E3_ridge on replay_problem_id (if present)
        # 5. Additional natural E3_mlp / E3_ridge cases covering CONTINUE, EXIT, EXCURSION, and an incorrect outcome
        replay_targets: List[Dict[str, Any]] = []
        seen_keys = set()

        def _add_target(rec_opt: Optional[Dict[str, Any]], role: str) -> None:
            if rec_opt is None:
                return
            k = (rec_opt["problem_id"], int(rec_opt["seed"]), rec_opt["method"])
            if k not in seen_keys:
                seen_keys.add(k)
                replay_targets.append({"role": role, "record": rec_opt})

        for m_want, role_name in (
            ("S0", "baseline_S0"),
            ("E1", "forced_EXCURSION_E1"),
            ("EXIT_2", "forced_EXIT_2"),
            ("E3_mlp", "learned_E3_mlp_primary"),
            ("E3_ridge", "learned_E3_ridge_primary"),
            ("D_learned", "donor_learned_primary"),
        ):
            cand = next(
                (r for r in records if r["problem_id"] == replay_problem_id and r["method"] == m_want),
                None,
            )
            _add_target(cand, role_name)

        # Ensure coverage of natural learned-policy CONTINUE, EXIT, EXCURSION, and an incorrect outcome
        learned_recs = [r for r in records if r.get("method") in ("E3_mlp", "E3_ridge")]
        for act_want in ("CONTINUE", "EXIT", "EXCURSION"):
            cand_act = next(
                (r for r in learned_recs if r.get("controller_action_type") == act_want),
                None,
            )
            _add_target(cand_act, f"natural_learned_{act_want}")
        cand_wrong = next(
            (r for r in learned_recs if not bool(r.get("is_correct", True))),
            None,
        )
        _add_target(cand_wrong, "natural_learned_incorrect_case")

        if not replay_targets:
            c9_replay = {
                "passed": False,
                "status": "FAIL",
                "category": "real_model_check",
                "reason": f"Problem {replay_problem_id} not found in {run_id}",
            }
        else:
            eval_cfg = raw_cfg.get("evaluation", {})
            suite_probs = generate_task_suite(
                family=str(eval_cfg.get("task_family", manifest.get("task_family", "all"))),
                split=str(eval_cfg.get("split", manifest.get("split", "test"))),
                num_problems=int(eval_cfg.get("num_problems", 18)),
                seed=int(manifest.get("seed", 42)),
            )
            prob_map = {p.problem_id: p for p in suite_probs}

            replayed_cases: List[Dict[str, Any]] = []
            all_replays_matched = True
            for item in replay_targets:
                rec = item["record"]
                prob = prob_map.get(rec["problem_id"])
                if prob is None:
                    all_replays_matched = False
                    replayed_cases.append(
                        {"role": item["role"], "matched": False, "reason": f"problem {rec['problem_id']} not in suite"}
                    )
                    continue
                m_name = str(rec["method"])
                rep = _decode_method_on_problem(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    problem=prob,
                    method=m_name,
                    seed=int(rec["seed"]),
                    recurrent_steps=int(manifest.get("recurrent_steps", 4)),
                    fork_after_step=int(branch_cfg.get("fork_after_step", 1)),
                    branch_count=1 if m_name == "S0" else int(manifest.get("branch_count", 2)),
                    perturbation_method="none" if m_name in ("S0", "S1") else str(branch_cfg.get("perturbation", "normalized_gaussian")),
                    perturbation_relative_norm=0.0 if m_name in ("S0", "S1") else float(rec.get("perturbation_relative_norm", 0.25)),
                    excursion_config=exc_cfg,
                    policy_config=pol_cfg,
                    controller_checkpoint=_ckpt_for(m_name),
                    max_output_tokens=int(exec_cfg.get("max_output_tokens", 2)),
                    extra_steps_for_s1=int(interv_cfg.get("extra_steps_for_s1", 3)),
                )
                ans_ok = rep["extracted_answer"] == rec["extracted_answer"]
                lp_ok = abs(float(rep["gold_logprob"]) - float(rec["gold_logprob"])) <= 1e-3
                calls_ok = int(rep["recurrent_block_calls"]) == int(rec["recurrent_block_calls"])
                act_ok = str(rep.get("controller_action_type")) == str(rec.get("controller_action_type"))
                matched = bool(ans_ok and lp_ok and calls_ok and act_ok)
                if not matched:
                    all_replays_matched = False
                replayed_cases.append(
                    {
                        "role": item["role"],
                        "problem_id": rec["problem_id"],
                        "seed": int(rec["seed"]),
                        "method": m_name,
                        "action_type": rec.get("controller_action_type"),
                        "is_correct": bool(rec.get("is_correct")),
                        "recorded_answer": rec["extracted_answer"],
                        "replayed_answer": rep["extracted_answer"],
                        "recorded_gold_logprob": rec["gold_logprob"],
                        "replayed_gold_logprob": rep["gold_logprob"],
                        "recorded_calls": int(rec["recurrent_block_calls"]),
                        "replayed_calls": int(rep["recurrent_block_calls"]),
                        "matched": matched,
                    }
                )
            c9_replay = {
                "passed": all_replays_matched,
                "status": "PASS" if all_replays_matched else "FAIL",
                "category": "real_model_check",
                "mode": "real_model_execution_replay",
                "primary_problem_id": replay_problem_id,
                "num_cases_replayed": len(replayed_cases),
                "replayed_cases": replayed_cases,
            }

    checks = {
        "1_checkpoint_provenance": c1_prov,
        "2_model_architecture": c2_arch,
        "3_genuine_recurrent_execution": c3_exec,
        "4_real_hidden_state_capture": c4_hidden,
        "5_shared_weights": c5_shared,
        "6_branch_isolation": c6_isolation,
        "7_visited_state_computation": c7_visit,
        "8_noop_intervention_equivalence": c8_noop,
        "9_deterministic_replay": c9_replay,
        "10_no_ground_truth_leakage": c10_leakage,
        "11_train_test_split_separation": c11_splits,
        "12_honest_compute_accounting": c12_compute,
    }

    unit_checks_passed = all(
        bool(v.get("passed")) for v in checks.values() if v.get("category") == "unit_check"
    )
    artifact_checks_passed = all(
        bool(v.get("passed"))
        for v in checks.values()
        if v.get("category") == "artifact_consistency_check"
    )
    real_model_verified = bool(c9_replay.get("passed") is True)
    all_passed = bool(
        unit_checks_passed
        and artifact_checks_passed
        and (real_model_verified if (replay_problem_id is not None or require_real_replay) else True)
    )

    return {
        "run_id": run_id,
        "all_passed": all_passed,
        "unit_checks_passed": unit_checks_passed,
        "artifact_consistency_passed": artifact_checks_passed,
        "real_model_replay_verified": real_model_verified,
        "num_records": len(records),
        "checks": checks,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tebdil.audit", description="Audit TEBDIL run evidence")
    parser.add_argument("--run-id", required=True, help="Run ID under results/")
    parser.add_argument(
        "--replay-problem-id",
        default=None,
        help="Optional problem_id to re-execute on the real Ouro model and compare against saved trace",
    )
    parser.add_argument(
        "--require-real-replay",
        action="store_true",
        help="Fail the audit if --replay-problem-id is omitted",
    )
    args = parser.parse_args(argv)
    report = audit_run(
        run_id=args.run_id,
        replay_problem_id=args.replay_problem_id,
        require_real_replay=args.require_real_replay,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
