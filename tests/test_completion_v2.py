"""Synthetic parser boundaries; no model runs or reserved-test inputs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tebdil.tasks.completion import extract_completed_integer
from tebdil.tasks.completion_v2 import evaluate_completion_v2, extract_completed_integer_v2


@pytest.mark.parametrize("expression", [
    "29", "29.", "+29", "-29", r"\boxed{29}", r"\boxed{ 29 }",
    "$29$", "$$29$$", r"\(29\)", r"\[29\]",
    r"$\boxed{29}$", r"$$\boxed{29}$$", r"\(\boxed{29}\)", r"\[\boxed{29}\]",
    "**Answer:** 29", r"Final answer: \(29\)", r"Result: $\boxed{29}$",
    r"We compute \(30-1=29\). Thus, \(\boxed{29}\).",
    "<think>999 and 8</think>Answer: 29",
    "Working: 30 minus 1.\n  Answer: 29",
])
def test_supported_unambiguous_completed_formats(expression):
    expected = "-29" if expression == "-29" else "29"
    answer = extract_completed_integer_v2(expression + "<eos>", "<eos>", completed=True)
    assert answer is not None and int(answer) == int(expected)


@pytest.mark.parametrize("text", [
    "", "29 or 30", "29 kg", "29.0", "2/3", "٢٩", r"\boxed{29/1}",
    "<think>29", "<think>29</think>", "29</think>",
    "<think><think>1</think></think>29", "<think>1</think><think>2</think>29",
    "Answer: 29\nAnswer: 30", "Answer: 29\nResult: 29",
    "Working.\n  Answer: 29\n  Result: 30", r"Working." "\n  Answer: 28\n" r"\boxed{29}",
    r"Answer: 29 then \boxed{30}", r"\boxed{29} \boxed{29}",
    r"\boxed{x} then \boxed{29}", r"\(\boxed{29}", r"\boxed{29}\)",
    r"\[\boxed{29}\)", "$29$$", "$$29$", r"\($29$\)",
    "Answer: 29<eos>junk", "29<eos><eos>",
    "<think>29<eos></think>29<eos>",
])
def test_ambiguous_malformed_or_unsupported_output_is_not_guessed(text):
    assert extract_completed_integer_v2(text, "<eos>", completed=True) is None


@pytest.mark.parametrize("text", ["29", r"\(\boxed{29}\).", "Answer: 29"])
def test_unfinished_output_never_yields_an_answer(text):
    assert extract_completed_integer_v2(text, "<eos>", completed=False) is None


def test_v1_is_not_silently_changed_and_v2_scoring_never_mutates_record():
    raw = r"<think>999</think>Thus, \(\boxed{29}\).<eos>"
    assert extract_completed_integer(raw, "<eos>", completed=True) is None
    assert extract_completed_integer(r"\boxed{29}", "<eos>", completed=True) == "29"
    record = {"completed": True, "decoded_text_with_special_tokens": raw,
              "extracted_answer": None, "status": "invalid_format", "is_correct": False}
    original = deepcopy(record)
    result = evaluate_completion_v2(record, "<eos>", "29")
    assert result == {"answer_extraction": "completed_final_integer_v2", "format_valid": True,
                      "extracted_answer": "29", "status": "correct", "is_correct": True}
    assert evaluate_completion_v2(record, "<eos>", "30")["status"] == "incorrect"
    assert record == original
    unfinished = {**record, "completed": False}
    assert evaluate_completion_v2(unfinished, "<eos>", "29")["status"] == "budget_exhausted"


def test_offline_audit_preserves_sources_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'reports' / name for name in ('ouro_completed_answer_smoke.json', 'ouro_completed_answer_comparison.json')]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    output = tmp_path / 'audit.json'
    command = [sys.executable, 'scripts/verify_completion_v2.py', '--output', str(output)]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report['posthoc'] and report['new_model_executions'] == 0
    assert len(report['records']) == 6 and report['independent_problems'] == 1
    changed = [r for r in report['records'] if r['original_v1'] != r['v2_diagnostic'] and r['original_v1']['status'] != r['v2_diagnostic']['status']]
    assert len(changed) == 1 and changed[0]['method'] == 'B_only'
    assert changed[0]['original_v1']['status'] == 'invalid_format'
    assert changed[0]['v2_diagnostic']['status'] == 'correct'
    second = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert second.returncode == 2 and 'preserve prior evidence' in second.stderr
    assert output.read_bytes() == raw
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
