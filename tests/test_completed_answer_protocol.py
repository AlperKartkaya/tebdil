"""Synthetic boundary fixtures; manifest checks do not execute reserved test inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tebdil.core.state import hash_tensor
from tebdil.evaluation.completion import (
    encode_completion_prompt, score_completion, summarize_completions,
    validate_completion, validate_task_manifest,
)
from tebdil.tasks import generate_arithmetic_problems
from tebdil.models.ouro_adapter import OuroRecurrentAdapter

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token_id = 2
    eos_token = "<eos>"
    text = "<think>999</think>Answer: 7<eos>"

    def __len__(self):
        return 128

    def decode(self, tokens, **kwargs):
        return self.text


def fixture(tokens=None, budget=2):
    tokens = [7, 2] if tokens is None else tokens
    prompt = [1, 4]
    ledger = []
    for k in range(len(tokens)):
        prefix = prompt + tokens[:k]
        ledger.append({
            "processed_input_ids": prefix, "prefix_length": len(prefix),
            "processed_sequence_length": len(prefix),
            "prefix_token_ids_sha256": hash_tensor(torch.tensor([prefix]), 64),
            "calls_before": k * 4, "recurrent_block_calls": 4,
        })
    return dict(tokenizer=Tokenizer(), prompt_ids=prompt, tokens=tokens, ledger=ledger,
                calls=4 * len(tokens), budget=budget, depth=4)


def test_prompt_only_receives_user_question():
    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == dict(tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt")
            return torch.tensor([[1, 4]])
    assert encode_completion_prompt(ChatTokenizer(), "question").tolist() == [[1, 4]]


def test_eos_on_budget_boundary_is_complete_and_compact_ledger_reconstructs():
    record = validate_completion(**fixture())
    assert record["completed"] and record["format_valid"]
    assert record["extracted_answer"] == "7"
    assert record["recurrent_token_positions"] == 20
    assert "processed_input_ids" not in record["token_ledger"][0]
    assert score_completion(record, "7") == {"status": "correct", "is_correct": True}
    assert score_completion(record, "8") == {"status": "incorrect", "is_correct": False}


def test_truncated_correct_looking_output_is_never_scored_correct():
    args = fixture(tokens=[7, 7])
    args["tokenizer"].text = "Answer: 7"
    record = validate_completion(**args)
    assert not record["completed"] and record["extracted_answer"] is None
    assert score_completion(record, "7")["status"] == "budget_exhausted"


def test_completed_unrecognized_output_is_invalid_not_first_integer_match():
    args = fixture()
    args["tokenizer"].text = "<think>7</think>Some possibilities are 7 and 8.<eos>"
    assert score_completion(validate_completion(**args), "7")["status"] == "invalid_format"


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(tokens=[]),
    lambda a: a.update(budget=0),
    lambda a: a.update(budget=True),
    lambda a: a.update(calls=4),
    lambda a: a.update(tokens=[2, 7]),
    lambda a: a.update(tokens=[7, 128]),
    lambda a: a.update(budget=3, tokens=[7, 7]),
    lambda a: a["ledger"][0].update(processed_input_ids=[1, 5]),
    lambda a: a["ledger"][0].update(prefix_token_ids_sha256="wrong"),
    lambda a: a["ledger"][1].update(calls_before=0),
    lambda a: a["ledger"][1].update(recurrent_block_calls=3),
])
def test_inconsistent_or_invalid_native_evidence_fails_closed(mutation):
    args = fixture()
    mutation(args)
    with pytest.raises(ValueError):
        validate_completion(**args)


def test_all_attempts_including_malformed_and_unfinished_remain_in_denominator():
    base = validate_completion(**fixture())
    rows = []
    for i, status in enumerate(("correct", "incorrect", "invalid_format", "budget_exhausted")):
        rows.append({**base, "problem_id": str(i), "status": status})
    summary = summarize_completions(rows)
    assert summary["attempted"] == 4 and summary["accuracy_all_attempts"] == 0.25
    assert summary["completed"] == 3 and summary["valid_format"] == 2
    assert summary["recurrent_block_calls"] == 32
    assert all(v == 1 for v in summary["outcome_counts"].values())
    with pytest.raises(ValueError):
        summarize_completions(rows + [rows[0]])


def test_frozen_manifest_is_reproducible_and_gold_is_mathematically_consistent():
    manifest = json.loads((ROOT / "configs/completion_v1_tasks.json").read_text())
    validate_task_manifest(manifest)
    for split, source_split, count in [("development", "validation", 12), ("test", "test", 60)]:
        expected = generate_arithmetic_problems(
            split=source_split, num_problems=count, seed=manifest["generator_seeds"][split],
        )
        for row, problem in zip(manifest["partitions"][split], expected, strict=True):
            assert row["prompt"] == problem.prompt and row["gold_answer"] == problem.gold_answer
            m = row["metadata"]
            if "linear_eq" in m:
                a, b, c = m["linear_eq"]
                assert a * int(row["gold_answer"]) + b == c
            else:
                op = m["op"]
                a, b, c, *rest = m["operands"]
                if op == "a*b+c-d":
                    value = a*b+c-rest[0]
                elif op == "(a*b+c)%m":
                    value = (a*b+c) % rest[0]
                elif op == "a*b-c*d":
                    value = a*b-c*rest[0]
                elif op == "w1*x1+w2*x2-bias":
                    value = a*b+c*rest[0]-rest[1]
                else:
                    raise AssertionError(f"Unverified arithmetic template: {op}")
                assert value == int(row["gold_answer"])


@pytest.mark.parametrize("change", ["id", "prompt", "template", "smoke", "gold"])
def test_manifest_rejects_leakage_and_invalid_selection(change):
    manifest = json.loads((ROOT / "configs/completion_v1_tasks.json").read_text())
    dev, test = manifest["partitions"]["development"][0], manifest["partitions"]["test"][0]
    if change in ("id", "prompt", "template"):
        key = {"id": "problem_id", "prompt": "prompt", "template": "template_id"}[change]
        test[key] = dev[key]
    elif change == "smoke":
        manifest["smoke_problem_ids"] = [test["problem_id"]]
    else:
        test["gold_answer"] = "not an integer"
    with pytest.raises(ValueError):
        validate_task_manifest(deepcopy(manifest))


def test_mini_native_engine_and_shared_validator_reconcile_real_boundaries(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("verify_completed_answer", ROOT / "scripts/verify_completed_answer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = 2
    ids = torch.tensor([[1, 4]])
    tokens, ledger, calls = module.native_generate(adapter, tokenizer, ids, 3)
    record = validate_completion(tokenizer, [1, 4], tokens, ledger, calls, budget=3, depth=4)
    assert calls == 4 * len(tokens)
    assert len(record["token_ledger"]) == len(tokens)
