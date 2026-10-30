"""Regression tests: feature history, tamper detection, audit failure modes, call accounting."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tebdil.audit import (
    check_counterfactual_and_controller_integrity,
    check_execution_ledgers,
    check_provenance_artifact,
    check_real_tensor_bundle,
    check_summary_trace_consistency,
)
from tebdil.controller.counterfactual import (
    _hash_arrays,
    compute_outcome_hash,
    verify_counterfactual_outcomes_unmodified,
)
from tebdil.controller.features import _compute_recent_update_cosine
from tebdil.controller.learned_policy import LearnedControllerCheckpoint
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.bootstrap import cluster_paired_bootstrap_ci
from tebdil.evaluation.runner import ByteLevelToyTokenizer, _decode_method_on_problem
from tebdil.evaluation.runner import _attach_parallel_compute_provenance
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite


# ---------------------------------------------------------------- update cosine
def test_update_cosine_uses_h0_and_never_substitutes_h1():
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=3)
    s0 = adapter.initialize(torch.tensor([[1, 2, 3, 4]]))
    assert len(s0.hidden_history) == 1  # H_0 stored
    s1 = adapter.transition(s0)
    cos1, avail1 = _compute_recent_update_cosine(s1)
    assert avail1 is False and cos1 == 0.0

    s2 = adapter.transition(s1)
    cos2, avail2 = _compute_recent_update_cosine(s2)
    assert avail2 is True
    h0, h1, h2 = (h.float().reshape(-1) for h in s2.hidden_history[-3:])
    v1, v2 = h1 - h0, h2 - h1
    expected = float(torch.dot(v2, v1) / (v1.norm() * v2.norm()))
    assert cos2 == pytest.approx(expected, abs=1e-5)
    # Must differ from the old buggy substitution cos(H2 - H1, H1).
    buggy = float(torch.dot(v2, h1) / (v2.norm() * h1.norm()))
    assert abs(cos2 - buggy) > 1e-6


# ---------------------------------------------------------- counterfactual tamper
def _make_record():
    a = {
        "action_id": "CONTINUE",
        "decoded_text": "42",
        "extracted_answer": "42",
        "is_correct": True,
        "gold_logprob": -0.25,
        "recurrent_block_calls": 8,
    }
    a["outcome_hash"] = compute_outcome_hash(problem_id="p1", seed=42, visit_step=2, **a)
    return {"problem_id": "p1", "seed": 42, "visit_step": 2, "actions": {"CONTINUE": a}}


@pytest.mark.parametrize(
    "field,value",
    [("is_correct", False), ("extracted_answer", "41"), ("gold_logprob", -0.26), ("recurrent_block_calls", 7)],
)
def test_counterfactual_tamper_detected(field, value):
    rec = _make_record()
    ok, errs = verify_counterfactual_outcomes_unmodified([rec])
    assert ok and not errs
    bad = copy.deepcopy(rec)
    bad["actions"]["CONTINUE"][field] = value
    ok, errs = verify_counterfactual_outcomes_unmodified([bad])
    assert not ok and "outcome_hash mismatch" in errs[0]


# --------------------------------------------------------------- audit: provenance
def _fake_provenance_repo(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    vend = tmp_path / "src" / "tebdil" / "models" / "vendored_ouro"
    vend.mkdir(parents=True)
    hashes = {}
    for name, body in (("model.safetensors", b"w" * 64), ("config.json", b"{}"), ("tokenizer.json", b"[]")):
        (snap / name).write_bytes(body)
        hashes[name] = {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}
    for v in ("modeling_ouro.py", "configuration_ouro.py"):
        (vend / v).write_text("# code\n")
        (snap / v).write_text("# code\n")
    ev = tmp_path / "artifacts" / "evidence"
    ev.mkdir(parents=True)
    (ev / "model_provenance.json").write_text(
        json.dumps(
            {
                "model_id": "ByteDance/Ouro-1.4B-Thinking",
                "model_revision": PINNED_OURO_THINKING_REV,
                "snapshot_path": str(snap),
                "weight_hashes": hashes,
            }
        )
    )
    return tmp_path


def test_audit_fails_on_one_char_provenance_hash_corruption(tmp_path: Path):
    repo = _fake_provenance_repo(tmp_path)
    assert check_provenance_artifact(repo)["passed"] is True
    p = repo / "artifacts" / "evidence" / "model_provenance.json"
    data = json.loads(p.read_text())
    h = data["weight_hashes"]["model.safetensors"]["sha256"]
    data["weight_hashes"]["model.safetensors"]["sha256"] = ("0" if h[0] != "0" else "1") + h[1:]
    p.write_text(json.dumps(data))
    res = check_provenance_artifact(repo)
    assert res["passed"] is False and res["hash_mismatches"]


def test_audit_fails_when_vendored_code_differs(tmp_path: Path):
    repo = _fake_provenance_repo(tmp_path)
    (repo / "src" / "tebdil" / "models" / "vendored_ouro" / "modeling_ouro.py").write_text("# edited\n")
    assert check_provenance_artifact(repo)["passed"] is False


# ------------------------------------------------- audit: controller param hashes
def _fake_controller_repo(tmp_path: Path) -> Path:
    rec = _make_record()
    cf = tmp_path / "artifacts" / "counterfactuals"
    cf.mkdir(parents=True)
    for split, name in (("train", "train_counterfactuals.json"), ("validation", "val_counterfactuals.json")):
        recs = [rec]
        sha = hashlib.sha256(json.dumps(recs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        (cf / name).write_text(json.dumps({"split": split, "records": recs, "records_sha256": sha}))
    ds_sha = hashlib.sha256(
        json.dumps({"train": [rec], "val": [rec]}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    meta_base = {
        "calibration_split": "validation",
        "calibration_summary": {"diversity_bonus_applied": False},
        "dataset_sha256": ds_sha,
    }
    ctrl = tmp_path / "artifacts" / "controllers"
    w = [0.1, -0.2, 0.3]
    ridge = LearnedControllerCheckpoint(
        controller_id="r", model_type="ridge", feature_schema=["a", "b", "c"],
        feature_mean=[0.0] * 3, feature_std=[1.0] * 3, weights=w,
        training_metadata={**meta_base, "final_param_hash": _hash_arrays([np.asarray(w, dtype=np.float64)])},
    )
    w1, b1, w2, b2 = [[0.1, 0.2, 0.3]], [0.0], [0.5], 0.1
    mlp_hash = _hash_arrays([np.asarray(w1), np.asarray(b1), np.asarray(w2), np.asarray([b2])])
    mlp = LearnedControllerCheckpoint(
        controller_id="m", model_type="mlp", feature_schema=["a", "b", "c"],
        feature_mean=[0.0] * 3, feature_std=[1.0] * 3, mlp_w1=w1, mlp_b1=b1, mlp_w2=w2, mlp_b2=b2,
        training_metadata={**meta_base, "final_param_hash": mlp_hash},
    )
    ridge.save(ctrl / "tebdil_ridge_v1.json")
    mlp.save(ctrl / "tebdil_mlp_v1.json")
    return tmp_path


def test_audit_fails_when_controller_weight_modified(tmp_path: Path):
    repo = _fake_controller_repo(tmp_path)
    assert check_counterfactual_and_controller_integrity(repo)["passed"] is True
    p = repo / "artifacts" / "controllers" / "tebdil_mlp_v1.json"
    d = json.loads(p.read_text())
    d["mlp_w2"][0] = 0.6
    p.write_text(json.dumps(d))
    res = check_counterfactual_and_controller_integrity(repo)
    assert res["passed"] is False
    assert any("final_param_hash mismatch" in e for e in res["errors"])


def test_audit_fails_when_counterfactual_outcome_edited(tmp_path: Path):
    repo = _fake_controller_repo(tmp_path)
    p = repo / "artifacts" / "counterfactuals" / "train_counterfactuals.json"
    d = json.loads(p.read_text())
    d["records"][0]["actions"]["CONTINUE"]["is_correct"] = False
    p.write_text(json.dumps(d))
    assert check_counterfactual_and_controller_integrity(repo)["passed"] is False


# ---------------------------------------------------------- audit: tensor bundle
def _fake_bundle_repo(tmp_path: Path) -> Path:
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=11)
    s1 = adapter.transition(adapter.initialize(torch.tensor([[3, 1, 4, 1, 5]])))
    a2 = adapter.transition(s1.clone("A"))
    b_state = s1.clone("B")
    b_state.hidden = s1.hidden + 0.05 * torch.randn_like(s1.hidden)
    b2 = adapter.transition(b_state)
    cfg = ExcursionConfig(alpha=0.65, beta=0.65, max_relative_norm=10.0, visit_step=2)
    na, _, telem = execute_excursion_step(adapter, a2.clone(), b2.clone(), cfg)
    vis = a2.clone("V")
    vis.hidden = (1 - telem.effective_alpha) * a2.hidden + telem.effective_alpha * b2.hidden
    last = lambda s: s.hidden[:, -1, :].detach().float()  # noqa: E731
    tensors = {
        "h2_a_last": last(a2),
        "h2_b_last": last(b2),
        "h2_visit_last": last(vis),
        "f_h2_a_last": last(adapter.transition(a2.clone())),
        "f_h2_b_last": last(adapter.transition(b2.clone())),
        "f_h2_visit_last": last(adapter.transition(vis)),
        "h3_returned_a_last": last(na),
    }
    ev = tmp_path / "artifacts" / "evidence"
    ev.mkdir(parents=True)
    pt = ev / "real_tensor_bundle.pt"
    torch.save(tensors, pt)
    meta = {
        "alpha": telem.effective_alpha,
        "beta": telem.beta,
        "pt_file_sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
        "tensor_sha256": {k: hash_tensor(v, length=64) for k, v in tensors.items()},
    }
    (ev / "real_tensor_bundle.json").write_text(json.dumps(meta))
    return tmp_path


def test_tensor_bundle_check_passes_then_fails_on_corruption(tmp_path: Path):
    repo = _fake_bundle_repo(tmp_path)
    ok = check_real_tensor_bundle(repo)
    assert ok["passed"] is True, ok
    pt = repo / "artifacts" / "evidence" / "real_tensor_bundle.pt"
    raw = bytearray(pt.read_bytes())
    raw[-10] ^= 0xFF
    pt.write_bytes(bytes(raw))
    assert check_real_tensor_bundle(repo)["passed"] is False


def test_tensor_bundle_check_fails_on_json_corruption(tmp_path: Path):
    repo = _fake_bundle_repo(tmp_path)
    jp = repo / "artifacts" / "evidence" / "real_tensor_bundle.json"
    meta = json.loads(jp.read_text())
    meta["tensor_sha256"]["h2_a_last"] = "0" * 64
    jp.write_text(json.dumps(meta))
    assert check_real_tensor_bundle(repo)["passed"] is False


# ---------------------------------------------------- audit: summary vs traces
def test_summary_trace_disagreement_detected():
    records = [
        {"method": "S0", "recurrent_block_calls": 8, "is_correct": True},
        {"method": "S0", "recurrent_block_calls": 8, "is_correct": False},
    ]
    good = {"rescue_matrix": {"methods": {"S0": {"mean_recurrent_calls": 8.0, "accuracy": 0.5}}}}
    assert check_summary_trace_consistency(good, records)["passed"] is True
    bad = {"rescue_matrix": {"methods": {"S0": {"mean_recurrent_calls": 8.0, "accuracy": 1.0}}}}
    assert check_summary_trace_consistency(bad, records)["passed"] is False
    missing = {"rescue_matrix": {"methods": {"E1": {"mean_recurrent_calls": 8.0, "accuracy": 0.5}}}}
    assert check_summary_trace_consistency(missing, records)["passed"] is False


# ------------------------------------------- S0 labels + token ledger accounting
@pytest.mark.parametrize("method,branch_count", [("S0", 1), ("E1", 2)])
def test_decode_labels_and_token_ledger_identity(method, branch_count):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128, total_ut_steps=4, seed=5)
    tok = ByteLevelToyTokenizer(vocab_size=128)
    prob = generate_task_suite(family="arithmetic", split="train", num_problems=1, seed=42)[0]
    rec = _decode_method_on_problem(
        adapter=adapter,
        tokenizer=tok,
        problem=prob,
        method=method,
        seed=42,
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=branch_count,
        perturbation_method="none" if method == "S0" else "normalized_gaussian",
        perturbation_relative_norm=0.0 if method == "S0" else 0.25,
        excursion_config=ExcursionConfig(visit_step=2),
        policy_config=AdaptivePolicyConfig(),
        max_output_tokens=3,
    )
    ledger = rec["token_ledger"]
    assert ledger
    assert sum(int(t["recurrent_block_calls"]) for t in ledger) == int(rec["recurrent_block_calls"])
    if method == "S0":
        assert rec["controller_action_type"] == "CONTINUE"
        assert rec.get("selected_donor", "none") == "none"
    else:
        assert rec["controller_action_type"] == "EXCURSION"


def test_parallel_reference_budget_is_not_verified_execution_cost():
    selected = {
        "problem_id": "p1",
        "seed": 42,
        "method": "S0",
        "recurrent_block_calls": 4,
        "token_ledger": [
            {"token_index": 0, "token_id": 7, "decoded_token": "7", "recurrent_block_calls": 4}
        ],
    }
    compute_source = {
        "problem_id": "p1",
        "seed": 42,
        "method": "P1",
        "recurrent_block_calls": 10,
        "estimated_flops": 123.0,
        "wall_time_ms": 45.0,
        "token_ledger": [
            {
                "token_index": 0,
                "token_id": 9,
                "decoded_token": "9",
                "recurrent_block_calls": 10,
                "shared_calls": 1,
                "primary_branch_calls": 3,
                "donor_calls": 6,
                "visit_calls": 0,
                "continuation_calls": 0,
            }
        ],
    }

    derived = _attach_parallel_compute_provenance(
        selected,
        compute_source=compute_source,
        method="P0",
        selected_branch="A",
        consulted_sources=(("A", selected), ("B", compute_source)),
    )

    assert derived["method"] == "P0"
    assert derived["token_ledger"][0]["token_id"] == 7
    assert "token_id" not in derived["reference_parallel_budget_ledger"][0]
    assert derived["recurrent_block_calls"] == 10
    assert derived["recurrent_block_calls_semantics"] == "idealized_reference_parallel_budget"
    assert derived["executed_selection_compute_verified"] is False
    assert [source["branch"] for source in derived["selection_provenance"]["consulted_sources"]] == ["A", "B"]
    checked = check_execution_ledgers([derived])
    assert checked["passed"] is False
    assert checked["reference_budget_consistent"] == "1/1"
    assert "not verified executed cost" in checked["ledger_errors"][0]

    hidden_mismatch = copy.deepcopy(derived)
    hidden_mismatch.pop("reference_parallel_budget_ledger")
    assert check_execution_ledgers([hidden_mismatch])["passed"] is False

    tampered = copy.deepcopy(derived)
    tampered["reference_parallel_budget_ledger"][0]["recurrent_block_calls"] = 9
    tampered_check = check_execution_ledgers([tampered])
    assert tampered_check["passed"] is False
    assert tampered_check["reference_budget_consistent"] == "0/1"


def test_historical_parallel_token_ledger_mismatch_fails_closed():
    historical = {
        "problem_id": "p1",
        "seed": 42,
        "method": "P1",
        "recurrent_block_calls": 10,
        "token_ledger": [{"recurrent_block_calls": 4}],
        "step_snapshots": [{"branch_rms": {"A": 1.0}}],
    }
    result = check_execution_ledgers([historical])
    assert result["passed"] is False
    assert "P1" in result["ledger_errors"][0]


def test_e0_does_not_publish_invented_visit_projection():
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=13)
    result = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        method="E0",
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        perturbation_method="normalized_gaussian",
        perturbation_relative_norm=0.25,
        excursion_config=ExcursionConfig(visit_step=2, max_relative_norm=0.35),
        record_pca=True,
    )
    assert "VISIT" not in result.pca_coords


# ------------------------------------------------------- cluster bootstrap
def test_cluster_bootstrap_resamples_problems_not_rows():
    clusters = ["p1"] * 3 + ["p2"] * 3
    base = [0, 0, 0, 1, 1, 1]
    meth = [1, 1, 1, 1, 1, 1]
    diff, lo, hi = cluster_paired_bootstrap_ci(clusters, base, meth, n_resamples=500, seed=0)
    assert diff == pytest.approx(0.5)
    # Only two clusters: resampled means can only be 0, 0.5, or 1.
    assert lo in (0.0, 0.5) and hi in (0.5, 1.0)
    with pytest.raises(ValueError):
        cluster_paired_bootstrap_ci(clusters, base, meth[:-1])


def test_historical_parallel_equal_sums_do_not_verify_selection_execution():
    record = {"problem_id": "p1", "seed": 42, "method": "P0",
              "recurrent_block_calls": 4, "token_ledger": [{"recurrent_block_calls": 4}]}
    result = check_execution_ledgers([record])
    assert result["token_ledger_consistent"] == "1/1"
    assert result["passed"] is False
    assert "lacks total execution provenance" in result["ledger_errors"][0]
