"""Native generation and final-region parsing, with mini/synthetic fixtures only."""

import pytest
import torch
import importlib.util
from pathlib import Path

from tebdil.models.ouro_adapter import OuroRecurrentAdapter

SPEC = importlib.util.spec_from_file_location(
    "complete_ouro_baseline", Path(__file__).resolve().parents[1] / "scripts/complete_ouro_baseline.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
final_integer = MODULE.final_integer
native_generate = MODULE.native_generate


@pytest.mark.parametrize("text,expected", [
    ("<think>8 * 6 is 48.</think>50<eos>", "50"),
    ("<think>50 is tempting but unfinished", None),
    ("<think>50</think>not an answer<eos>", None),
    ("-12<eos>", "-12"),
    ("There are 12 edges and 15 is the answer<eos>", None),
    (r"<think>8 * 6 is 48.</think>\boxed{50}<eos>", "50"),
    (r"<think>8 * 6 is 48.</think>$\boxed{-12}$<eos>", "-12"),
])
def test_final_integer_excludes_thought_and_requires_complete_integer_region(text, expected):
    assert final_integer(text, "<eos>") == expected


def test_native_greedy_prefix_matches_adapter_and_actual_calls():
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

    ids = torch.tensor([[1, 14, 29]])
    tokens, ledger, calls = native_generate(adapter, Tokenizer(), ids, 3)
    for token in tokens:
        state = adapter.initialize(ids)
        for _ in range(4):
            state = adapter.transition(state)
        assert int(adapter.readout_logits(state)[:, -1].argmax()) == token
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    assert calls == len(tokens) * 4
    assert [e["prefix_length"] for e in ledger] == list(range(3, 3 + len(tokens)))
    assert [e["processed_sequence_length"] for e in ledger] == [e["prefix_length"] for e in ledger]
