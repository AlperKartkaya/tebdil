"""Exploratory prompt/budget diagnostic on three pilot problems, never held-out claims.

uv run python scripts/calibrate_ouro_baseline.py --output reports/ouro_baseline_calibration.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer

from tebdil.core.policies import AdaptivePolicyConfig
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.runner import _decode_method_on_problem
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import generate_task_suite

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier evidence and choose a new path")
    repo = "ByteDance/Ouro-1.4B-Thinking"
    snapshot = Path(snapshot_download(repo, revision=PINNED_OURO_THINKING_REV, local_files_only=True))
    provenance = json.loads((ROOT / "site/static/evidence/model_provenance.json").read_text())
    identities = {name: sha256(snapshot / name) for name in provenance["weight_hashes"]}
    assert all(identities[n] == h["sha256"] for n, h in provenance["weight_hashes"].items())
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    adapter = OuroRecurrentAdapter.from_pretrained(local_files_only=True, readout_mode="last_step")
    seed = 20261031
    problems = [generate_task_suite(family=f, split="pilot", num_problems=1, seed=seed)[0]
                for f in ("arithmetic", "logic", "graph")]
    records = []
    for problem in problems:
        for prompt_format, budget in (("raw", 2), ("raw", 64), ("chat", 128)):
            print(f"{problem.problem_id}: {prompt_format}, max {budget} tokens", flush=True)
            before = adapter.recurrent_block_calls
            record = _decode_method_on_problem(
                adapter, tokenizer, problem, "S0", seed, 4, 1, 1, "none", 0.0,
                ExcursionConfig(), AdaptivePolicyConfig(), max_output_tokens=budget,
                prompt_format=prompt_format,
            )
            del record["_step0_logits"]
            record["executed_recurrent_calls_including_scoring"] = adapter.recurrent_block_calls - before
            assert record["executed_recurrent_calls_including_scoring"] == record["recurrent_block_calls"] + record["scoring_recurrent_calls"]
            record["strict_integer_format_valid"] = bool(re.fullmatch(r"[+-]?\d+", record["decoded_text"]))
            record["strict_correct"] = record["strict_integer_format_valid"] and record["is_correct"]
            record["encoded_prompt_text"] = tokenizer.decode(record["prompt_token_ids"], skip_special_tokens=False)
            records.append(record)
            print(f"  {record['stop_reason']}, {len(record['generated_token_ids'])} tokens; strict_correct={record['strict_correct']}", flush=True)

    for offset in range(0, len(records), 3):
        short, long = records[offset:offset + 2]
        n = min(len(short["generated_token_ids"]), len(long["generated_token_ids"]))
        assert short["generated_token_ids"][:n] == long["generated_token_ids"][:n]
    sources = ["scripts/calibrate_ouro_baseline.py", "src/tebdil/evaluation/runner.py",
               "src/tebdil/core/scheduler.py", "src/tebdil/tasks/validators.py", "uv.lock"]
    report = {
        "schema_version": 1, "purpose": "Exploratory baseline-format diagnostic, not a held-out benchmark",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: sha256(ROOT / p) for p in sources},
        "model_id": repo, "model_revision": PINNED_OURO_THINKING_REV,
        "checkpoint_sha256": identities, "device": str(adapter.device), "dtype": str(adapter.dtype),
        "readout_mode": "last_step", "cache_mode": "disabled", "torch_version": torch.__version__,
        "seed": seed, "independent_problems": len(problems),
        "chat_template": tokenizer.chat_template, "chat_enable_thinking": False,
        "records": records,
        "limitations": [
            "Three pilot problems selected before observing outputs; no held-out or statistical efficacy claim",
            "Chat uses the pinned tokenizer template with enable_thinking=False; not a claim of optimal prompting",
            "Raw retains legacy first-newline/one-token Boolean stops; chat stops at EOS or its token budget",
            "Legacy is_correct extracts the first integer; strict_correct additionally requires an integer-only whole response",
            "Budget exhaustion is not an EOS or proof that a reasoning answer completed",
            "No intervention, controller training or historical result replacement",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Evidence: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
