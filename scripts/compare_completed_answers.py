"""Serial gate-2 technical smoke; preserves per-token partial evidence on failure."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from verify_completed_answer import MODEL, ROOT, SOURCES, sha256, synchronize
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.comparison import generate_fixed_completion
from tebdil.evaluation.completion import score_completion
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

METHODS = ("B_only", "M0", "M1", "E0", "E1")
COMPARISON_SOURCES = list(dict.fromkeys(SOURCES + [
    "scripts/compare_completed_answers.py", "src/tebdil/evaluation/comparison.py",
    "src/tebdil/core/scheduler.py", "src/tebdil/core/branching.py",
    "src/tebdil/core/perturbations.py", "src/tebdil/core/controls.py",
    "src/tebdil/core/excursions.py", "src/tebdil/core/policies.py",
    "docs/completed_answer_comparison_protocol.md", "tests/test_fixed_completion_comparison.py",
    "reports/ouro_completed_answer_smoke.json",
]))


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new directory; preserve all completed and partial evidence")
    sources = {p: sha256(ROOT / p) for p in COMPARISON_SOURCES}
    reference = json.loads((ROOT / "reports/ouro_completed_answer_smoke.json").read_text())
    baseline = reference["records"][0]
    if not reference["all_checks_passed"] or baseline["problem_id"] != "completion_dev_000":
        raise ValueError("Verified preselected native baseline required")
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint = {name: sha256(snapshot / name) for name in reference["checkpoint_sha256"]}
    if checkpoint != reference["checkpoint_sha256"]:
        raise ValueError("Checkpoint/tokenizer/config provenance mismatch")
    config = ExcursionConfig(alpha=.65, beta=.65, max_relative_norm=.1, visit_step=2, token_scope="full")
    identity = {
        "schema_version": 1, "protocol": "completed_answer_comparison_smoke_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources, "checkpoint_sha256": checkpoint,
        "model_id": MODEL, "model_revision": PINNED_OURO_THINKING_REV,
        "device": reference["device"], "dtype": reference["dtype"],
        "torch_version": torch.__version__, "methods": list(METHODS),
        "problem_id": baseline["problem_id"], "split": "development",
        "seed": 20261101, "max_output_tokens": 2048,
        "excursion_config": asdict(config), "cache_mode": "disabled",
        "purpose": "One already-correct development comparison smoke, not intervention efficacy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    journal = args.output_dir / "journal.jsonl"
    with journal.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "start", **identity}) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16, local_files_only=True)
    if str(adapter.device) != reference["device"] or torch.__version__ != reference["torch_version"]:
        raise ValueError("Reused native baseline requires the same device/software contract")
    records = []
    total_calls = adapter.recurrent_block_calls
    try:
        parity = generate_fixed_completion(adapter, tokenizer, baseline["prompt"], "S0", config, seed=20261101, budget=8)
        if parity["generated_token_ids"] != baseline["generated_token_ids"][:8]:
            raise ValueError("Native/adapter eight-token prefix parity failed")
        write_new_json(args.output_dir / "parity.json", parity)
        append_trace_record({"event": "parity", "record": parity}, journal)
        for method in METHODS:
            print(f"{method}: full completion, max 2048 tokens", flush=True)
            append_trace_record({"event": "method_start", "method": method}, journal)

            def record_token(entry, current_method=method):
                append_trace_record({"event": "token", "method": current_method, **entry}, journal)
                if (entry["token_index"] + 1) % 64 == 0:
                    print(f"{current_method}: {entry['token_index'] + 1} tokens", flush=True)

            synchronize(adapter.device)
            started = time.monotonic()
            row = generate_fixed_completion(
                adapter, tokenizer, baseline["prompt"], method, config,
                seed=20261101, budget=2048, on_token=record_token,
            )
            synchronize(adapter.device)
            row.update({
                "problem_id": baseline["problem_id"], "split": "development",
                "prompt": baseline["prompt"], "gold_answer": baseline["gold_answer"],
                **score_completion(row, baseline["gold_answer"]),
                "wall_seconds": round(time.monotonic() - started, 3),
            })
            if sources != {p: sha256(ROOT / p) for p in COMPARISON_SOURCES}:
                raise ValueError("Source/input identities changed during comparison")
            write_new_json(args.output_dir / f"{method}.json", row)
            append_trace_record({"event": "method_finish", "record": row}, journal)
            records.append(row)
            print(f"{method}: {row['status']}, {len(row['generated_token_ids'])} tokens, {row['wall_seconds']}s", flush=True)
        if checkpoint != {name: sha256(snapshot / name) for name in checkpoint}:
            raise ValueError("Checkpoint changed during comparison")
        new_calls = adapter.recurrent_block_calls - total_calls
        if new_calls != parity["recurrent_block_calls"] + sum(r["recurrent_block_calls"] for r in records):
            raise ValueError("Campaign call accounting mismatch")
        report = {
            **identity, "baseline_reference": "reports/ouro_completed_answer_smoke.json",
            "baseline_record": baseline, "parity": parity, "records": records,
            "actual_new_recurrent_calls": new_calls,
            "reused_baseline_recurrent_calls": baseline["recurrent_block_calls"],
            "all_checks_passed": True,
            "limitations": [
                "One already-correct development problem; no rescue opportunity or general accuracy estimate",
                "E0 matches input direction distance, not returned output norm",
                "B_only charges both branches; not optimized independent sampling or best-of-N",
                "Equal E0/E1 calls at the same prefix do not imply matched total sequence compute",
                "No held-out questions, efficacy pilot, statistical test or controller evaluated",
                "Partial tokens and every method outcome retained; no favorable subset selected",
            ],
        }
        write_new_json(args.output_dir / "report.json", report)
        append_trace_record({"event": "finish", "actual_new_recurrent_calls": new_calls}, journal)
    except Exception as exc:
        append_trace_record({"event": "failure", "type": type(exc).__name__, "message": str(exc)}, journal)
        raise


if __name__ == "__main__":
    main()
