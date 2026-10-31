"""Mini-model fixtures only; no accuracy evidence or reserved test inference."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.comparison import METHOD_CALLS, generate_fixed_completion
from tebdil.evaluation.completion import score_completion
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


class Tokenizer:
    eos_token_id = 2
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "question"}]
        assert kwargs["enable_thinking"] is False
        return torch.tensor([[1, 4]])

    def decode(self, tokens, **kwargs):
        return "<think>99</think>Answer: 7<eos>"


def adapter():
    model = OuroRecurrentAdapter.from_mini_config(vocab_size=128)

    def readout(state):
        logits = torch.zeros((*state.hidden.shape[:2], 128))
        logits[..., 7] = 10
        return logits

    model.readout_logits = readout
    return model


@pytest.mark.parametrize("method", list(METHOD_CALLS))
def test_fixed_method_calls_seeds_prefixes_and_budget_retention(method):
    model, tokenizer = adapter(), Tokenizer()
    events = []
    record = generate_fixed_completion(
        model, tokenizer, "question", method, ExcursionConfig(alpha=.65, beta=.65),
        seed=20261101, budget=2, on_token=events.append,
    )
    assert record["generated_token_ids"] == [7, 7]
    assert record["token_ledger"] == events
    assert record["recurrent_block_calls"] == 2 * METHOD_CALLS[method]
    assert record["recurrent_token_positions"] == 5 * METHOD_CALLS[method]
    assert not record["completed"] and not record["format_valid"]
    assert score_completion(record, "7")["status"] == "budget_exhausted"
    assert "gold_answer" not in record and "is_correct" not in record
    for k, entry in enumerate(events):
        assert entry["prefix_token_ids_sha256"] == hash_tensor(torch.tensor([[1, 4] + [7] * k]), 64)
        assert entry["branch_seed"] == 20261101 + 131 * k
        assert entry["calls_before"] == k * METHOD_CALLS[method]
        assert entry["selected_branch"] == ("B" if method == "B_only" else "A")
    assert len(model.model.model.norm._forward_hooks) == 0


def test_eos_is_scored_only_after_gold_free_generation():
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 7
    record = generate_fixed_completion(adapter(), tokenizer, "question", "E1", ExcursionConfig(), seed=1, budget=3)
    assert record["completed"] and record["generated_token_ids"] == [7]
    assert record["extracted_answer"] == "7"
    assert score_completion(record, "8")["status"] == "incorrect"
    assert score_completion(record, "7")["status"] == "correct"


def test_callback_failure_preserves_evidence_and_removes_hook():
    model = adapter()
    events = []

    def fail(entry):
        events.append(entry)
        raise RuntimeError("synthetic journal failure")

    with pytest.raises(RuntimeError, match="journal"):
        generate_fixed_completion(model, Tokenizer(), "question", "E1", ExcursionConfig(), seed=1, budget=2, on_token=fail)
    assert len(events) == 1 and events[0]["recurrent_block_calls"] == 8
    assert not model.model.model.norm._forward_hooks


def test_invalid_method_and_nonfrozen_scope_are_rejected():
    for method, config in [("E3_mlp", ExcursionConfig()), ("E1", ExcursionConfig(token_scope="last_token"))]:
        with pytest.raises(ValueError):
            generate_fixed_completion(adapter(), Tokenizer(), "question", method, config, seed=1, budget=1)


def test_runner_refuses_to_overwrite_partial_evidence(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text('{"event":"partial"}\n')
    result = subprocess.run(
        [sys.executable, "scripts/compare_completed_answers.py", "--output-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "preserve all completed and partial evidence" in result.stderr
    assert journal.read_text() == '{"event":"partial"}\n'
    assert list(tmp_path.iterdir()) == [journal]
