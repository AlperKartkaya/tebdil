"""Gate 1: frozen, gold-free native generation; offline completed-answer scoring.

Runs only the manifest's development smoke selection, never its reserved test set.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

from huggingface_hub import snapshot_download
import psutil
import torch
from transformers import AutoTokenizer

from complete_ouro_baseline import native_generate, sha256
from tebdil.evaluation.completion import (
    encode_completion_prompt, score_completion, summarize_completions,
    validate_completion, validate_task_manifest,
)
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

ROOT = Path(__file__).resolve().parents[1]
MODEL = "ByteDance/Ouro-1.4B-Thinking"
SOURCES = [
    "scripts/verify_completed_answer.py", "scripts/complete_ouro_baseline.py",
    "src/tebdil/evaluation/completion.py", "src/tebdil/evaluation/trace.py",
    "src/tebdil/tasks/completion.py", "src/tebdil/tasks/arithmetic.py",
    "src/tebdil/tasks/validators.py", "src/tebdil/models/ouro_adapter.py",
    "src/tebdil/core/state.py", "vendor/ouro_1_4b_thinking/modeling_ouro.py",
    "vendor/ouro_1_4b_thinking/configuration_ouro.py", "uv.lock",
    "configs/completion_v1_tasks.json", "docs/completed_answer_protocol.md",
    "tests/test_completed_answer_protocol.py",
]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or args.output == journal:
        parser.error("Output/journal must be distinct new paths; preserve earlier evidence")
    manifest = json.loads((ROOT / "configs/completion_v1_tasks.json").read_text())
    validate_task_manifest(manifest)
    rows = {r["problem_id"]: r for r in manifest["partitions"]["development"]}
    sources = {p: sha256(ROOT / p) for p in SOURCES}
    provenance = ROOT / "reports/ouro_baseline_completion.json"
    reference = json.loads(provenance.read_text())
    if reference["model_id"] != MODEL or reference["model_revision"] != PINNED_OURO_THINKING_REV:
        raise ValueError("Checkpoint reference has wrong identity")
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint = {name: sha256(snapshot / name) for name in reference["checkpoint_sha256"]}
    if checkpoint != reference["checkpoint_sha256"]:
        raise ValueError("Checkpoint/tokenizer/config provenance mismatch")
    identity = {
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources, "checkpoint_sha256": checkpoint,
        "checkpoint_reference": provenance.relative_to(ROOT).as_posix(),
        "checkpoint_reference_sha256": sha256(provenance),
        "model_id": MODEL, "model_revision": PINNED_OURO_THINKING_REV,
        "protocol": "completed_answer_v1", "max_output_tokens": 2048,
        "recurrent_depth": 4, "dtype": "torch.bfloat16", "cache_mode": "disabled",
        "engine": "native generate, greedy, full prefix, forced last-step",
        "scope": "development smoke only; not a held-out accuracy benchmark",
        "smoke_problem_ids": manifest["smoke_problem_ids"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "start", **identity}) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16, device=args.device, local_files_only=True)
    if adapter.total_steps != 4 or adapter.readout_mode != "last_step":
        raise ValueError("Unsupported recurrent/readout contract")
    records = []
    physical_calls = 0

    def count_boundary(module, inputs, output):
        nonlocal physical_calls
        physical_calls += 1

    handle = adapter.model.model.norm.register_forward_hook(count_boundary)
    try:
        for problem_id in manifest["smoke_problem_ids"]:
            problem = rows[problem_id]
            # Only this string enters generation; gold and metadata stay offline.
            prompt_ids = encode_completion_prompt(tokenizer, problem["prompt"]).to(adapter.device)
            synchronize(adapter.device)
            before, started = physical_calls, time.monotonic()
            print(f"Frozen smoke {problem_id}: budget 2048; {adapter.device}", flush=True)
            tokens, ledger, calls = native_generate(adapter, tokenizer, prompt_ids, 2048)
            synchronize(adapter.device)
            elapsed = time.monotonic() - started
            if physical_calls - before != calls:
                raise ValueError("Independent physical-call counters disagree")
            record = validate_completion(
                tokenizer, prompt_ids.tolist()[0], tokens, ledger, calls, budget=2048, depth=4,
            )
            record.update({
                "problem_id": problem_id, "split": "development", "method": "S0",
                "prompt": problem["prompt"], "template_id": problem["template_id"],
                "gold_answer": problem["gold_answer"],
                **score_completion(record, problem["gold_answer"]),
                "wall_seconds": round(elapsed, 3),
                "process_rss_after_bytes": psutil.Process().memory_info().rss,
                "mps_allocated_after_bytes": torch.mps.current_allocated_memory() if adapter.device.type == "mps" else None,
            })
            records.append(record)
            append_trace_record({"event": "completion", "record": record}, journal)
            print(f"{len(tokens)} tokens; {record['status']}; {calls} calls; {elapsed:.1f}s", flush=True)
    finally:
        handle.remove()
    if sources != {p: sha256(ROOT / p) for p in SOURCES}:
        raise ValueError("Inputs or sources changed during execution")
    if checkpoint != {name: sha256(snapshot / name) for name in checkpoint}:
        raise ValueError("Checkpoint changed during execution")
    summary = summarize_completions(records)
    checks = {
        "source_inputs_unchanged": True, "checkpoint_unchanged": True,
        "physical_calls_match": physical_calls == summary["recurrent_block_calls"],
        "exact_smoke_selection": [r["problem_id"] for r in records] == manifest["smoke_problem_ids"],
        "no_test_questions_executed": all(r["split"] == "development" for r in records),
    }
    report = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **identity, "device": str(adapter.device), "torch_version": torch.__version__,
        "hardware": platform.platform(), "system_memory_bytes": psutil.virtual_memory().total,
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
        "records": records, "summary": summary, "checks": checks,
        "all_checks_passed": all(checks.values()), "actual_recurrent_calls": physical_calls,
        "limitations": [
            "One preselected development smoke, not accuracy, difficulty or intervention-utility evidence",
            "Existing parser was developed on earlier pilots; frozen before this observation, not a novel preregistration",
            "Reserved test questions were not generated; no controller or intervention evaluated",
            "All attempted cases retained; budget exhaustion is not completed-answer success",
            "Call counts and recurrent token positions are not FLOPs or compute-matched comparisons",
            "RSS and MPS allocation are after-run snapshots, not peak memory measurements",
            "Full-prefix uncached generation; no persistent cross-token hidden-state branches",
        ],
    }
    append_trace_record({"event": "finish", "checks": checks, "summary": summary}, journal)
    if not report["all_checks_passed"]:
        raise ValueError("Completion evidence checks failed")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
