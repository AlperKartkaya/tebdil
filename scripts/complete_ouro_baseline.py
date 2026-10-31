"""Follow up truncated pilot chat responses with native, cache-free generation.

uv run python scripts/complete_ouro_baseline.py --output reports/ouro_baseline_completion.json
This is a completion diagnostic, not a new intervention benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import torch
from transformers import AutoTokenizer

from tebdil.core.state import hash_tensor
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.tasks import TaskProblem, validate_problem_response

ROOT = Path(__file__).resolve().parents[1]


def final_integer(text: str, eos_text: str) -> str | None:
    """Read only the completed final region; never score integers inside thought."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    text = text.strip()
    if eos_text and text.endswith(eos_text):
        text = text[:-len(eos_text)].strip()
    return text if re.fullmatch(r"[+-]?\d+", text) else None


@torch.no_grad()
def native_generate(adapter, tokenizer, prompt_ids, budget):
    """Count real boundary executions and preserve each actual input prefix."""
    calls = 0
    ledger = []

    def norm_hook(module, args, output):
        nonlocal calls
        calls += 1

    def prefix_hook(module, args, kwargs):
        ids = kwargs["input_ids"]
        ledger.append({
            "prefix_length": int(ids.shape[1]),
            "prefix_token_ids_sha256": hash_tensor(ids, length=64),
            "calls_before": calls,
        })

    norm_handle = adapter.model.model.norm.register_forward_hook(norm_hook)
    prefix_handle = adapter.model.register_forward_pre_hook(prefix_hook, with_kwargs=True)
    try:
        outputs = adapter.model.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            use_cache=False, do_sample=False, num_beams=1,
            max_new_tokens=budget, exit_at_step=adapter.total_steps - 1,
            logits_to_keep=0, repetition_penalty=1.0, no_repeat_ngram_size=0,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        norm_handle.remove()
        prefix_handle.remove()
    tokens = outputs[0, prompt_ids.shape[1]:].tolist()
    for k, entry in enumerate(ledger):
        after = ledger[k + 1]["calls_before"] if k + 1 < len(ledger) else calls
        entry["recurrent_block_calls"] = after - entry["calls_before"]
    if len(tokens) != len(ledger) or any(e["recurrent_block_calls"] != adapter.total_steps for e in ledger):
        raise ValueError("Native generation ledger does not match executed depth")
    return tokens, ledger, calls


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier evidence")
    reference_path = ROOT / "reports/ouro_baseline_calibration.json"
    reference = json.loads(reference_path.read_text())
    repo = "ByteDance/Ouro-1.4B-Thinking"
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=PINNED_OURO_THINKING_REV, local_files_only=True)
    adapter = OuroRecurrentAdapter.from_pretrained(local_files_only=True)
    budget = 2048
    records = []
    # Selection is completion-driven, not correctness-driven: every chat case.
    for initial in (r for r in reference["records"] if r["prompt_format"] == "chat"):
        print(f"Native completion: {initial['problem_id']}, max {budget} tokens", flush=True)
        ids = torch.tensor([initial["prompt_token_ids"]], device=adapter.device)
        started = time.monotonic()
        tokens, ledger, calls = native_generate(adapter, tokenizer, ids, budget)
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        complete = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
        answer = final_integer(raw, tokenizer.eos_token) if complete else None
        problem = TaskProblem(
            initial["problem_id"], initial["family"], initial["split"], initial["template_id"],
            initial["difficulty"], initial["prompt"], initial["gold_answer"],
        )
        outcome = validate_problem_response(problem, answer) if answer is not None else None
        old_tokens = initial["generated_token_ids"]
        compared = min(len(old_tokens), len(tokens))
        divergence = next((i for i in range(compared) if tokens[i] != old_tokens[i]), None)
        row = {
            "problem_id": problem.problem_id, "family": problem.family, "split": problem.split,
            "prompt": problem.prompt, "gold_answer": problem.gold_answer,
            "prompt_token_ids": initial["prompt_token_ids"], "generated_token_ids": tokens,
            "decoded_text_with_special_tokens": raw,
            "stop_reason": "eos_token" if complete else "max_output_tokens",
            "final_integer": answer, "strict_correct": bool(outcome and outcome.is_valid),
            "reference_prefix_tokens_compared": compared,
            "first_divergent_token_vs_adapter": divergence,
            "recurrent_block_calls": calls, "token_ledger": ledger,
            "wall_seconds": round(time.monotonic() - started, 3),
        }
        records.append(row)
        print(f"  {len(tokens)} tokens; {row['stop_reason']}; final={answer}; strict_correct={row['strict_correct']}; divergence={divergence}", flush=True)

    sources = ["scripts/complete_ouro_baseline.py", "src/tebdil/models/ouro_adapter.py",
               "src/tebdil/tasks/validators.py", "vendor/ouro_1_4b_thinking/modeling_ouro.py", "uv.lock"]
    report = {
        "schema_version": 1, "purpose": "Native completion follow-up to truncated pilot calibration",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: sha256(ROOT / p) for p in sources},
        "reference_calibration_sha256": sha256(reference_path),
        "model_id": repo, "model_revision": PINNED_OURO_THINKING_REV,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "device": str(adapter.device), "dtype": str(adapter.dtype), "torch_version": torch.__version__,
        "engine": "native generate, cache disabled, forced depth 4, greedy, full-prefix logits",
        "max_output_tokens": budget, "independent_problems": len(records), "records": records,
        "limitations": [
            "Adaptive budget follow-up after all three pilot chat responses were unfinished at 128 tokens",
            "Same three pilot problems, not additional independent observations or a held-out study",
            "Final integer is scored only after EOS and outside a completed thought region",
            "Native ledger counts actual RMSNorm-boundary executions, not inferred compute savings",
            "No intervention efficacy, optimal-prompt or general model-competence claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
