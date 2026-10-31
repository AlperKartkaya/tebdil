"""Export an evidence-bound static observatory; verify it without local model/results files.

Build: python scripts/build_landing_data.py
Check committed export: python scripts/build_landing_data.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_LABELS = {
    "phase7_8_fresh_test": "Fresh-template evaluation",
    "phase7_8_controller_test": "Exploratory controller evaluation",
    "phase4_adaptive": "Adaptive vs fixed excursions",
    "phase3_ablations": "Mechanism and curvature controls",
    "phase5_donor": "Multiple-donor comparison",
    "phase6_selective": "Selective-transfer comparison",
    "phase2_recovery": "Recovery pilot",
    "phase1_smoke": "Initial feasibility pilot",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty recording: {path}")
    return rows


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ledger_matches(row: dict[str, Any]) -> bool:
    if row.get("derived_parallel_selection"):
        return False  # Reference budget is not total execution across selected candidates.
    ledger = row.get("token_ledger")
    return bool(ledger) and sum(t["recurrent_block_calls"] for t in ledger) == row["recurrent_block_calls"]


def validate_rows(rows: list[dict[str, Any]]) -> None:
    keys = [(r["problem_id"], r["seed"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate problem/seed/method recording")
    baseline = {(r["problem_id"], r["seed"]) for r in rows if r["method"] == "S0"}
    if not baseline:
        raise ValueError("Recording has no S0 baseline")
    methods = {r["method"] for r in rows}
    for method in methods:
        pairs = {(r["problem_id"], r["seed"]) for r in rows if r["method"] == method}
        if pairs != baseline:
            raise ValueError(f"Unpaired recording for {method}")
    for row in rows:
        if type(row["is_correct"]) is not bool:
            raise ValueError("Correctness must be an explicit boolean")
        if not isinstance(row["recurrent_block_calls"], int) or row["recurrent_block_calls"] <= 0:
            raise ValueError("Missing or invalid measured recurrent-call count")


def build(root: Path = ROOT) -> None:
    from tebdil.evaluation.bootstrap import mcnemar_exact_test, paired_bootstrap_ci
    from tebdil.evaluation.metrics import summarize_paired_records

    site = root / "site"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "meta": {"project": "TEBDİL", "mode": "Recorded experiment", "model_id": "ByteDance/Ouro-1.4B-Thinking",
                 "model_revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"},
        "runs": {}, "evidence": [], "files": {},
    }

    def write_file(relative: str, body: bytes, source: str) -> str:
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        manifest["files"][relative] = {"sha256": digest(body), "bytes": len(body), "source": source}
        return relative

    for run_id, label in RUN_LABELS.items():
        base = root / "results" / run_id
        rows = load_jsonl(base / "traces.jsonl")
        validate_rows(rows)
        baseline = [r for r in rows if r["method"] == "S0"]
        e1 = [r for r in rows if r["method"] == "E1"]
        methods = {}
        mismatches = Counter(r["method"] for r in rows if not ledger_matches(r))
        for method in dict.fromkeys(r["method"] for r in rows):
            selected = [r for r in rows if r["method"] == method]
            stats = asdict(summarize_paired_records(
                method, selected, baseline_records=baseline, e1_records=e1 or None,
                bootstrap_fn=paired_bootstrap_ci, mcnemar_fn=mcnemar_exact_test,
            ))
            stats = {key: stats[key] for key in (
                "num_instances", "accuracy", "baseline_accuracy", "paired_accuracy_diff",
                "rescued_count", "base_wrong_count", "degraded_count", "base_correct_count",
                "mean_recurrent_calls", "cluster_ci_95_low", "cluster_ci_95_high",
                "cluster_mcnemar_p_value", "num_unique_problems",
            )}
            stats["ledger_mismatches"] = mismatches[method]
            methods[method] = stats
        body = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        trace_url = write_file(f"static/data/{run_id}.json", body, f"results/{run_id}/traces.jsonl")
        run_manifest = write_file(f"static/evidence/{run_id}.yaml", (base / "manifest.yaml").read_bytes(), f"results/{run_id}/manifest.yaml")
        splits = {r["split"] for r in rows}
        if len(splits) != 1:
            raise ValueError(f"Mixed splits in {run_id}")
        manifest["runs"][run_id] = {
            "run_id": run_id, "label": label, "split": next(iter(splits)), "record_count": len(rows),
            "problem_ids": sorted({r["problem_id"] for r in baseline}),
            "num_problems": len({r["problem_id"] for r in baseline}), "seeds": sorted({r["seed"] for r in baseline}),
            "methods": methods, "trace_url": trace_url, "trace_sha256": digest(body),
            "source_trace_sha256": digest((base / "traces.jsonl").read_bytes()),
            "manifest_url": run_manifest, "ledger_mismatches": sum(mismatches.values()),
            "status": "Exploratory; historical recording, not a preregistered confirmatory study",
        }

    sources = [
        "reports/mvp_audit_fresh_test.json",
        "artifacts/evidence/model_provenance.json",
        "artifacts/evidence/phase1_excursion_verification.json",
        "artifacts/evidence/real_tensor_bundle.json", "artifacts/evidence/real_tensor_bundle.pt",
        "artifacts/evidence/audit_phase7_8_report.json", "artifacts/evidence/audit_fresh_test_report.json",
        "artifacts/counterfactuals/train_counterfactuals.json", "artifacts/counterfactuals/val_counterfactuals.json",
        "artifacts/counterfactuals/split_manifest.json",
        "results/phase1_grid/grid_summary.json",
        "results/phase3_ablations/dimension_ablations.json",
    ]
    sources += [p.relative_to(root).as_posix() for p in sorted((root / "artifacts/controllers").glob("*.json"))]
    for source in sources:
        path = root / source
        url = write_file(f"static/evidence/{path.name}", path.read_bytes(), source)
        manifest["evidence"].append({"label": path.stem.replace("_", " "), "source": source, "url": url, "sha256": digest(path.read_bytes()),
                                     "historical": path.name.startswith("audit_")})
    bundle = "window.TEBDIL_DATA = " + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n"
    (site / "static/tebdil_data.js").write_text(bundle, encoding="utf-8")
    check(root)
    print(f"Built {len(manifest['runs'])} recorded runs; startup manifest {len(bundle.encode()):,} bytes.")


def check(root: Path = ROOT) -> dict[str, Any]:
    site = root / "site"
    bundle = (site / "static/tebdil_data.js").read_text(encoding="utf-8")
    manifest = json.loads(bundle.removeprefix("window.TEBDIL_DATA = ").removesuffix(";\n"))
    if manifest["schema_version"] != 2 or not manifest["runs"]:
        raise ValueError("Unsupported or empty observatory export")
    for relative, expected in manifest["files"].items():
        path = (site / relative).resolve()
        if not path.is_relative_to(site.resolve()):
            raise ValueError(f"Unsafe export path: {relative}")
        data = path.read_bytes()
        if digest(data) != expected["sha256"] or len(data) != expected["bytes"]:
            raise ValueError(f"Export checksum mismatch: {relative}")
    for run in manifest["runs"].values():
        body = (site / run["trace_url"]).read_bytes()
        if digest(body) != run["trace_sha256"]:
            raise ValueError("Trace does not match recording manifest")
        rows = json.loads(body)
        validate_rows(rows)
        if len(rows) != run["record_count"]:
            raise ValueError("Trace record count mismatch")
        mismatches = sum(not ledger_matches(r) for r in rows)
        if mismatches != run["ledger_mismatches"]:
            raise ValueError("Unreported compute-ledger mismatch")
    print(f"Verified export checksums and pairing for {len(manifest['runs'])} runs.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed export without model or raw local results")
    args = parser.parse_args()
    check() if args.check else build()


if __name__ == "__main__":
    main()
