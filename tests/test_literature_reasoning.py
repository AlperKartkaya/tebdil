"""Unit tests for the literature-grounded LLM reasoning benchmark suite."""

from pathlib import Path

from tebdil.tasks.literature_reasoning import (
    build_literature_reasoning_dataset,
    evaluate_literature_reasoning,
    extract_literature_integer,
)


def test_literature_dataset_verifies_all_families():
    data = build_literature_reasoning_dataset()
    assert data["suite"] == "literature_reasoning_v1"
    dev = data["partitions"]["development"]
    test = data["partitions"]["reserved_test"]
    assert len(dev) == 12
    assert len(test) == 6
    dev_families = {t["family"] for t in dev}
    assert dev_families == {
        "crt_counter_trap",
        "knights_and_knaves",
        "gsm8k_multistep",
        "math_number_theory",
        "zebra_constraint",
        "graph_greedy_trap",
    }
    cfg_path = Path(__file__).resolve().parents[1] / "configs/literature_reasoning_v1_tasks.json"
    assert cfg_path.exists()


def test_extract_literature_integer_formats():
    assert extract_literature_integer("Answer: 18", completed=True) == "18"
    assert extract_literature_integer("**Answer:** 42513", completed=True) == "42513"
    assert extract_literature_integer(r"Thus the answer is \boxed{79}.", completed=True) == "79"
    assert extract_literature_integer("x = 23.", completed=True) == "23"
    assert extract_literature_integer("Answer: 18", completed=False) is None


def test_evaluate_literature_reasoning_status():
    ok = evaluate_literature_reasoning(
        {"completed": True, "reasoning_status": "closed", "final_text": "\n\nAnswer: 101"},
        "101",
    )
    assert ok["status"] == "correct"
    assert ok["is_correct"] is True

    wrong = evaluate_literature_reasoning(
        {"completed": True, "reasoning_status": "closed", "final_text": "\n\nAnswer: 110"},
        "101",
    )
    assert wrong["status"] == "incorrect"
    assert wrong["is_correct"] is False
