"""Shared completed-answer contract; generation never receives gold answers."""

import re

import torch

from tebdil.core.state import hash_tensor
from tebdil.tasks.completion import extract_completed_integer


def validate_task_manifest(manifest):
    """Reject duplicate identities/prompts and overlapping development/test templates."""
    if manifest.get("schema_version") != 1 or set(manifest.get("partitions", {})) != {"development", "test"}:
        raise ValueError("Expected v1 development/test partitions")
    ids, prompts, templates = set(), set(), {}
    for split, rows in manifest["partitions"].items():
        if not rows:
            raise ValueError("Empty partition")
        templates[split] = set()
        for row in rows:
            if (not isinstance(row["problem_id"], str) or not row["problem_id"]
                    or row["problem_id"] in ids or row["prompt"] in prompts):
                raise ValueError("Duplicate or empty problem identity/prompt")
            if (row["split"] != split or row["family"] != "arithmetic"
                    or not row["prompt"].strip() or not row["template_id"]
                    or not re.fullmatch(r"[+-]?\d+", row["gold_answer"])):
                raise ValueError("Invalid integer-task record")
            ids.add(row["problem_id"])
            prompts.add(row["prompt"])
            templates[split].add(row["template_id"])
    if not templates["development"].isdisjoint(templates["test"]):
        raise ValueError("Development/test templates overlap")
    smoke = manifest["smoke_problem_ids"]
    dev_ids = {r["problem_id"] for r in manifest["partitions"]["development"]}
    if not smoke or len(set(smoke)) != len(smoke) or not set(smoke) <= dev_ids:
        raise ValueError("Smoke selection must contain unique development identities")


def encode_completion_prompt(tokenizer, prompt):
    """No gold, candidates or task metadata enter the model's chat input."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )


def validate_completion(tokenizer, prompt_ids, tokens, ledger, calls, *, budget, depth):
    """Validate native evidence before parsing; fail closed on inconsistent stops/costs."""
    if type(budget) is not int or budget <= 0 or type(depth) is not int or depth <= 0:
        raise ValueError("Positive integer budget/depth required")
    if (not prompt_ids or not tokens or len(tokens) > budget
            or any(type(t) is not int or t < 0 or t >= len(tokenizer) for t in prompt_ids + tokens)):
        raise ValueError("Invalid prompt/generated token IDs or output length")
    if len(ledger) != len(tokens) or type(calls) is not int or calls != depth * len(tokens):
        raise ValueError("Measured recurrent calls do not match generation")
    eos = tokenizer.eos_token_id
    if eos in tokens[:-1]:
        raise ValueError("Generation continued after EOS")
    completed = tokens[-1] == eos
    if not completed and len(tokens) != budget:
        raise ValueError("Generation ended without EOS or budget exhaustion")
    compact = []
    for k, entry in enumerate(ledger):
        prefix = prompt_ids + tokens[:k]
        if (entry["processed_input_ids"] != prefix
                or entry["prefix_length"] != len(prefix)
                or entry["processed_sequence_length"] != len(prefix)
                or entry["prefix_token_ids_sha256"] != hash_tensor(torch.tensor([prefix]), 64)
                or entry["calls_before"] != k * depth
                or entry["recurrent_block_calls"] != depth):
            raise ValueError("Actual prefix/call ledger mismatch")
        compact.append({key: value for key, value in entry.items() if key != "processed_input_ids"})
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    answer = extract_completed_integer(text, tokenizer.eos_token, completed=completed)
    return {
        "prompt_token_ids": prompt_ids, "generated_token_ids": tokens,
        "decoded_text_with_special_tokens": text,
        "stop_reason": "eos_token" if completed else "max_output_tokens",
        "completed": completed, "format_valid": answer is not None,
        "extracted_answer": answer, "answer_extraction": "completed_final_integer_v1",
        "recurrent_block_calls": calls, "token_ledger": compact,
        "recurrent_token_positions": sum(e["prefix_length"] * depth for e in compact),
    }


def score_completion(record, gold_answer):
    """Offline only: unfinished and malformed responses remain in the denominator."""
    if not isinstance(gold_answer, str) or not re.fullmatch(r"[+-]?\d+", gold_answer):
        raise ValueError("Gold must be an integer string")
    if not record["completed"]:
        return {"status": "budget_exhausted", "is_correct": False}
    if record["extracted_answer"] is None:
        return {"status": "invalid_format", "is_correct": False}
    correct = int(record["extracted_answer"]) == int(gold_answer)
    return {"status": "correct" if correct else "incorrect", "is_correct": correct}


def summarize_completions(records):
    if not records or len({r["problem_id"] for r in records}) != len(records):
        raise ValueError("Expected nonempty unique problem attempts")
    statuses = ("correct", "incorrect", "invalid_format", "budget_exhausted")
    if any(r["status"] not in statuses for r in records):
        raise ValueError("Unknown completion outcome")
    counts = {s: sum(r["status"] == s for r in records) for s in statuses}
    return {
        "attempted": len(records), "outcome_counts": counts,
        "completed": len(records) - counts["budget_exhausted"],
        "valid_format": counts["correct"] + counts["incorrect"],
        "accuracy_all_attempts": counts["correct"] / len(records),
        "generated_tokens": sum(len(r["generated_token_ids"]) for r in records),
        "recurrent_block_calls": sum(r["recurrent_block_calls"] for r in records),
        "recurrent_token_positions": sum(r["recurrent_token_positions"] for r in records),
    }
