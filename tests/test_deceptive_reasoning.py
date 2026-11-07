"""Unit tests for deceptive reasoning benchmark suite and verifiers."""

import json
from pathlib import Path

from tebdil.tasks.deceptive_reasoning import (
    build_deceptive_reasoning_dataset,
    evaluate_deceptive_reasoning,
    write_deceptive_tasks_config,
)


def test_deceptive_reasoning_dataset_construction():
    data = build_deceptive_reasoning_dataset()
    assert data["suite"] == "deceptive_traps_v1"
    assert len(data["tasks"]) == 38
    assert len(data["families"]) == 4

    families = {t["family"] for t in data["tasks"]}
    assert families == {
        "reasoning_rigidity",
        "gsm_noop_distractor",
        "cognitive_illusion",
        "rate_and_combinatorics",
    }


def test_deceptive_reasoning_gold_and_traps():
    data = build_deceptive_reasoning_dataset()
    task_map = {t["problem_id"]: t for t in data["tasks"]}

    # 1. Infertile rabbits
    assert task_map["trap_infertile_rabbits"]["gold_answer"] == "1"
    assert "8" in task_map["trap_infertile_rabbits"]["trap_answers"]

    # 2. Direct jug
    assert task_map["trap_simple_jug_measure"]["gold_answer"] == "3"
    assert "5" in task_map["trap_simple_jug_measure"]["trap_answers"]

    # 3. Harmonic speed
    assert task_map["trap_speed_harmonic_mean"]["gold_answer"] == "40"
    assert "45" in task_map["trap_speed_harmonic_mean"]["trap_answers"]

    # 4. GSM-NoOp kiwi
    assert task_map["trap_gsm_kiwi_size"]["gold_answer"] == "190"
    assert "185" in task_map["trap_gsm_kiwi_size"]["trap_answers"]

    # 5. Monty Hall 4-door
    assert task_map["trap_monty_hall_4doors"]["gold_answer"] == "75"
    assert "50" in task_map["trap_monty_hall_4doors"]["trap_answers"]

    # 6. Newly added XERJ / ReasoningTrap / GSM-NoOp / MATH-500 / BBEH tasks
    assert task_map["trap_potted_tree_soil"]["gold_answer"] == "10"
    assert task_map["trap_aime_alice_bob_nim_take_all"]["gold_answer"] == "0"
    assert task_map["trap_gsm_noop_andre_raise"]["gold_answer"] == "4272"
    assert task_map["trap_bbeh_retroactive_inventory"]["gold_answer"] == "35"
    assert task_map["trap_mtbench_executive_parking_zebra"]["gold_answer"] == "32514"
    assert task_map["trap_math500_amicable_284"]["gold_answer"] == "284"
    assert task_map["trap_math500_monic_factors_10"]["gold_answer"] == "10"


def test_evaluate_deceptive_reasoning_statuses():
    row_correct = {
        "final_text": "The answer is \\boxed{40}.",
        "completed": True,
        "trap_answers": ["45", "50"],
    }
    res = evaluate_deceptive_reasoning(row_correct, "40")
    assert res["status"] == "correct"
    assert res["is_correct"] is True
    assert res["hit_trap_answer"] is False

    row_trap = {
        "final_text": "Therefore, the average speed is 45.",
        "completed": True,
        "trap_answers": ["45", "50"],
    }
    res_trap = evaluate_deceptive_reasoning(row_trap, "40")
    assert res_trap["status"] == "trap_failure"
    assert res_trap["is_correct"] is False
    assert res_trap["hit_trap_answer"] is True

    row_exhausted = {
        "final_text": "",
        "completed": False,
        "trap_answers": ["45"],
    }
    res_exh = evaluate_deceptive_reasoning(row_exhausted, "40")
    assert res_exh["status"] == "budget_exhausted"


def test_write_deceptive_tasks_config(tmp_path: Path):
    cfg_file = tmp_path / "deceptive_tasks.json"
    data = write_deceptive_tasks_config(cfg_file)
    assert cfg_file.exists()
    loaded = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert len(loaded["tasks"]) == len(data["tasks"]) == 38
