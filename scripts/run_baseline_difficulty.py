"""Serial development-only native S0 pilot; no interventions or reserved tests."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from verify_completed_answer import MODEL, ROOT, SOURCES, native_generate, sha256, synchronize
from tebdil.core.state import hash_tensor
from tebdil.evaluation.completion import (
    encode_completion_prompt, score_completion, summarize_completions,
    validate_completion, validate_task_manifest,
)
from tebdil.evaluation.trace import append_trace_record
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks.completion_v2 import evaluate_completion_v2

PILOT_SOURCES = list(dict.fromkeys(SOURCES + [
    "scripts/run_baseline_difficulty.py", "tests/test_baseline_difficulty.py",
    "src/tebdil/tasks/completion_v2.py", "tests/test_completion_v2.py",
    "docs/completed_answer_v2_protocol.md", "docs/baseline_difficulty_protocol.md",
    "reports/ouro_completed_answer_smoke.json", "reports/ouro_completion_v2_audit.json",
]))


@torch.no_grad()
def run_native_completion(adapter, tokenizer, prompt, budget, on_forward):
    """Observe the existing native engine independently without giving it gold."""
    if adapter.total_steps != 4 or adapter.readout_mode != "last_step":
        raise ValueError("Frozen four-depth last-step contract required")
    ids = encode_completion_prompt(tokenizer, prompt).to(adapter.device)
    prompt_ids = ids.tolist()[0]
    physical_calls, previous_calls = 0, 0
    proposals, observations = [], []

    def boundary(module, inputs, output):
        nonlocal physical_calls
        physical_calls += 1

    def observe(module, args, kwargs, output):
        nonlocal previous_calls
        prefix = kwargs["input_ids"]
        last_logits = output.logits[:, -1]
        if prefix.tolist()[0] != prompt_ids + proposals or not torch.isfinite(last_logits).all():
            raise ValueError("Actual native prefix/readout disagrees with greedy observer")
        token = int(last_logits.argmax().item())
        entry = {
            "token_index": len(proposals), "greedy_readout_token_id": token,
            "prefix_length": prefix.shape[1], "prefix_token_ids_sha256": hash_tensor(prefix, 64),
            "calls_before": previous_calls, "recurrent_block_calls": physical_calls - previous_calls,
        }
        if entry["recurrent_block_calls"] != 4:
            raise ValueError("Independent physical depth mismatch")
        proposals.append(token)
        observations.append(entry)
        previous_calls = physical_calls
        on_forward(entry)

    norm_hook = adapter.model.model.norm.register_forward_hook(boundary)
    forward_hook = adapter.model.register_forward_hook(observe, with_kwargs=True)
    try:
        tokens, ledger, calls = native_generate(adapter, tokenizer, ids, budget)
    finally:
        forward_hook.remove()
        norm_hook.remove()
    if tokens != proposals or physical_calls != calls:
        raise ValueError("Native outputs/physical calls disagree with independent observer")
    record = validate_completion(tokenizer, prompt_ids, tokens, ledger, calls, budget=budget, depth=4)
    for entry, native in zip(observations, record["token_ledger"], strict=True):
        if any(entry[k] != native[k] for k in ("prefix_length", "prefix_token_ids_sha256", "calls_before", "recurrent_block_calls")):
            raise ValueError("Independent/native prefix ledgers disagree")
    return {**record, "independent_forward_checks_passed": True}


def difficulty_decision(counts):
    if counts["invalid_format"] or counts["budget_exhausted"]:
        return "headroom_unresolved_format_or_truncation"
    return "recognized_baseline_failures" if counts["incorrect"] else "no_observed_rescue_opportunity"


def write_new(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new directory; preserve partial evidence")
    manifest = json.loads((ROOT / "configs/completion_v1_tasks.json").read_text())
    validate_task_manifest(manifest)
    tasks = manifest["partitions"]["development"]
    expected = [f"completion_dev_{k:03d}" for k in range(12)]
    if [p["problem_id"] for p in tasks] != expected:
        raise ValueError("Frozen development identities/order changed")
    source_hashes = {p: sha256(ROOT / p) for p in PILOT_SOURCES}
    reference = json.loads((ROOT / "reports/ouro_completed_answer_smoke.json").read_text())
    if (not reference["all_checks_passed"] or reference["records"][0]["problem_id"] != expected[0]
            or reference["records"][0]["prompt"] != tasks[0]["prompt"]
            or reference["records"][0]["gold_answer"] != tasks[0]["gold_answer"]):
        raise ValueError("Verified reusable baseline required")
    snapshot = Path(snapshot_download(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    checkpoint = {name: sha256(snapshot / name) for name in reference["checkpoint_sha256"]}
    if checkpoint != reference["checkpoint_sha256"] or torch.__version__ != reference["torch_version"]:
        raise ValueError("Checkpoint/software contract differs from reused S0")
    identity = {
        "schema_version": 1, "protocol": "development_baseline_difficulty_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes, "checkpoint_sha256": checkpoint,
        "model_id": MODEL, "model_revision": PINNED_OURO_THINKING_REV,
        "device": reference["device"], "dtype": reference["dtype"], "torch_version": torch.__version__,
        "cache_mode": "disabled", "recurrent_depth": 4, "readout_mode": "last_step",
        "max_output_tokens": 2048, "answer_extraction": "completed_final_integer_v2",
        "eos_token": reference["eos_token"], "eos_token_id": reference["eos_token_id"],
        "problem_ids": expected, "new_problem_ids": expected[1:], "reused_problem_ids": expected[:1],
        "reused_reference": "reports/ouro_completed_answer_smoke.json",
        "scope": "Development baseline headroom; no intervention or held-out accuracy claim",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    journal = args.output_dir / "journal.jsonl"
    with journal.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "start", **identity}) + "\n")
    campaign_calls = 0
    campaign_hook = None
    records = []

    def count_campaign(module, inputs, output):
        nonlocal campaign_calls
        campaign_calls += 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
        if tokenizer.eos_token != reference["eos_token"] or tokenizer.eos_token_id != reference["eos_token_id"]:
            raise ValueError("Tokenizer EOS differs from the reused native contract")
        adapter = OuroRecurrentAdapter.from_pretrained(dtype=torch.bfloat16, device=reference["device"], local_files_only=True)
        campaign_hook = adapter.model.model.norm.register_forward_hook(count_campaign)
        for k, task in enumerate(tasks):
            problem_id = task["problem_id"]
            if k == 0:
                record = dict(reference["records"][0])
                origin = "reused_gate_1"
            else:
                print(f"{problem_id}: native S0, max 2048 tokens", flush=True)
                append_trace_record({"event": "attempt_start", "problem_id": problem_id}, journal)

                def forward(entry, current=problem_id):
                    append_trace_record({"event": "forward", "problem_id": current, **entry}, journal)

                synchronize(adapter.device)
                started, before = time.monotonic(), campaign_calls
                record = run_native_completion(adapter, tokenizer, task["prompt"], 2048, forward)
                synchronize(adapter.device)
                if campaign_calls - before != record["recurrent_block_calls"]:
                    raise ValueError("Campaign/native physical counts disagree")
                record.update({
                    "problem_id": problem_id, "split": "development", "method": "S0",
                    "prompt": task["prompt"], "template_id": task["template_id"], "gold_answer": task["gold_answer"],
                    "wall_seconds": round(time.monotonic() - started, 3),
                    **score_completion(record, task["gold_answer"]),
                })
                origin = "new_native_generation"
            old_fields = {key: record[key] for key in ("answer_extraction", "format_valid", "extracted_answer", "status", "is_correct")}
            record = {**record, "origin": origin, "v1_evaluation": old_fields,
                      **evaluate_completion_v2(record, tokenizer.eos_token, task["gold_answer"])}
            if source_hashes != {p: sha256(ROOT / p) for p in PILOT_SOURCES}:
                raise ValueError("Frozen sources/tasks changed during pilot")
            write_new(args.output_dir / f"{problem_id}.json", record)
            append_trace_record({"event": "reuse" if k == 0 else "completion", "record": record}, journal)
            records.append(record)
            print(f"{problem_id}: {record['status']}, {len(record['generated_token_ids'])} tokens, {record['wall_seconds']}s ({origin})", flush=True)
        if checkpoint != {name: sha256(snapshot / name) for name in checkpoint}:
            raise ValueError("Checkpoint changed during pilot")
        if campaign_calls != sum(r["recurrent_block_calls"] for r in records[1:]):
            raise ValueError("Campaign recurrent accounting mismatch")
        summary = summarize_completions(records)
        report = {
            **identity, "records": records, "summary": summary,
            "by_template": {t: summarize_completions([r for r in records if r["template_id"] == t]) for t in sorted({r["template_id"] for r in records})},
            "actual_new_recurrent_calls": campaign_calls,
            "reused_recurrent_calls": records[0]["recurrent_block_calls"],
            "difficulty_decision": difficulty_decision(summary["outcome_counts"]),
            "all_checks_passed": True,
            "limitations": ["Twelve convenience development questions, three known templates, one previously observed S0 completion",
                           "No intervention, rescue, significance test, controller or held-out generalization evaluated",
                           "Formatting/truncation outcomes do not establish wrong arithmetic",
                           "Instrumented pipeline runtime; recurrent token positions are not FLOPs"],
        }
        write_new(args.output_dir / "report.json", report)
        append_trace_record({"event": "finish", "actual_new_recurrent_calls": campaign_calls, "summary": summary}, journal)
    except Exception as exc:
        append_trace_record({"event": "failure", "type": type(exc).__name__, "message": str(exc)}, journal)
        raise
    finally:
        if campaign_hook is not None:
            campaign_hook.remove()


if __name__ == "__main__":
    main()
