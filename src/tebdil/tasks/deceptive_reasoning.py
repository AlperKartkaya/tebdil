"""Deceptive and trap-laden reasoning benchmark suite.

Targets cognitive illusions, arithmetic distractors (Apple GSM-NoOp),
reasoning rigidity / instruction overriding (KAIST/AITRICS ReasoningTrap),
and counterfactual traps where standard Looped LM / CoT pattern-matching defaults
to a familiar incorrect answer (S0 trap), testing if latent excursion-and-return (E2)
re-anchors the model to the conditioned constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tebdil.tasks.literature_reasoning import extract_literature_integer


def build_deceptive_reasoning_dataset() -> Dict[str, Any]:
    """Construct the 16-task deceptive reasoning benchmark."""
    suffix = "Be concise. Output only the integer result.\nAnswer: "

    tasks = [
        # --- Family 1: reasoning_rigidity_puzzles (ReasoningTrap / Counterfactuals) ---
        {
            "problem_id": "trap_infertile_rabbits",
            "family": "reasoning_rigidity",
            "template_id": "trap_rabbit_infertility_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Fibonacci Inversion",
            "reasoning_pathology": "habitual_fibonacci_overriding",
            "prompt": (
                "A pair of permanently infertile adult rabbits is placed in an enclosure on January 1st. "
                "Every month, each fertile mature pair produces one new pair of offspring. "
                "Rabbits take one month to mature. If no rabbits die or leave, how many total pairs "
                f"of rabbits will be in the enclosure on June 1st (after 5 months)? {suffix}"
            ),
            "gold_answer": "1",
            "trap_answers": ["8", "5", "13", "6"],
        },
        {
            "problem_id": "trap_simple_jug_measure",
            "family": "reasoning_rigidity",
            "template_id": "trap_direct_jug_capacity_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Water Jug Inversion",
            "reasoning_pathology": "die_hard_pouring_overcomplication",
            "prompt": (
                "A gardener has four empty water jugs: a 2-gallon jug, a 3-gallon jug, a 4-gallon jug, "
                "and a 5-gallon jug. None of the jugs have intermediate measurement markings. "
                "To obtain exactly 3 gallons of water, what is the capacity in gallons of the single "
                f"jug that can be filled directly to the brim? {suffix}"
            ),
            "gold_answer": "3",
            "trap_answers": ["5", "4", "2", "6"],
        },
        {
            "problem_id": "trap_hotel_empty_rooms",
            "family": "reasoning_rigidity",
            "template_id": "trap_hilbert_unoccupied_v1",
            "difficulty": "hard",
            "source_benchmark": "KAIST/AITRICS ReasoningTrap / Hilbert Hotel Inversion",
            "reasoning_pathology": "unnecessary_hilbert_shift",
            "prompt": (
                "A grand hotel has an infinite number of rooms, but only 10 rooms are currently occupied "
                "and all other rooms are completely empty. A new guest arrives. "
                "What is the minimum number of currently registered guests that must be relocated "
                f"to another room in order to accommodate the new guest? {suffix}"
            ),
            "gold_answer": "0",
            "trap_answers": ["10", "1", "11"],
        },
        {
            "problem_id": "trap_base10_reversal",
            "family": "reasoning_rigidity",
            "template_id": "trap_aime_base10_cycle_v1",
            "difficulty": "hard",
            "source_benchmark": "AIME 2022-I-2 Base Inversion",
            "reasoning_pathology": "memorized_aime_base9_recall",
            "prompt": (
                "Find the smallest three-digit positive integer abc (where a, b, and c are decimal digits "
                "with a > 0) such that the three-digit integer abc equals the three-digit integer bca "
                f"in base ten. {suffix}"
            ),
            "gold_answer": "111",
            "trap_answers": ["227", "100", "101"],
        },
        # --- Family 2: gsm_noop_distractors (Apple AIML GSM-NoOp) ---
        {
            "problem_id": "trap_gsm_kiwi_size",
            "family": "gsm_noop_distractor",
            "template_id": "noop_kiwi_under_average_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp (Mirzadeh et al., 2024)",
            "reasoning_pathology": "distractor_subtraction_bias",
            "prompt": (
                "Oliver picks 44 kiwis on Friday and 58 kiwis on Saturday. On Sunday, he picks double "
                "the number of kiwis he picked on Friday. On Sunday, 5 of the kiwis he picked were "
                "a bit smaller than average. How many total kiwis did Oliver pick across all three "
                f"days? {suffix}"
            ),
            "gold_answer": "190",
            "trap_answers": ["185", "180", "195"],
        },
        {
            "problem_id": "trap_gsm_school_inflation",
            "family": "gsm_noop_distractor",
            "template_id": "noop_school_historical_inflation_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp (Mirzadeh et al., 2024)",
            "reasoning_pathology": "historical_rate_application",
            "prompt": (
                "Liam buys 20 notebooks that cost $5 each and 15 pens that cost $2 each today. "
                "Due to general economic inflation, prices of stationery were 10 percent lower last year. "
                f"How many dollars does Liam pay in total for his purchase today? {suffix}"
            ),
            "gold_answer": "130",
            "trap_answers": ["117", "143", "120"],
        },
        {
            "problem_id": "trap_gsm_apple_crates",
            "family": "gsm_noop_distractor",
            "template_id": "noop_crate_shelf_placement_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp / Inventory Placement",
            "reasoning_pathology": "spatial_subset_exclusion",
            "prompt": (
                "A warehouse stores 15 crates of apples, and each crate contains exactly 20 apples. "
                "During routine morning restocking, 3 of the crates were placed on the top shelf. "
                f"How many total apples are stored in the warehouse? {suffix}"
            ),
            "gold_answer": "300",
            "trap_answers": ["240", "60", "260"],
        },
        {
            "problem_id": "trap_gsm_bakery_shapes",
            "family": "gsm_noop_distractor",
            "template_id": "noop_bakery_cookie_shape_v1",
            "difficulty": "medium",
            "source_benchmark": "Apple GSM-NoOp / Attribute Irrelevance",
            "reasoning_pathology": "shape_filter_deduction",
            "prompt": (
                "A baker bakes 60 chocolate cookies and 40 butter cookies. Of all the cookies baked, "
                "15 cookies were shaped like stars and the rest were shaped like circles. "
                f"How many total cookies did the baker bake? {suffix}"
            ),
            "gold_answer": "100",
            "trap_answers": ["85", "75", "115"],
        },
        # --- Family 3: cognitive_illusions (Cognitive Reflection & Speed Traps) ---
        {
            "problem_id": "trap_speed_harmonic_mean",
            "family": "cognitive_illusion",
            "template_id": "trap_roundtrip_harmonic_speed_v1",
            "difficulty": "medium",
            "source_benchmark": "Classic Physics / SimpleBench Cognitive Traps",
            "reasoning_pathology": "arithmetic_mean_heuristic",
            "prompt": (
                "A delivery van travels from Warehouse A to Warehouse B at an average speed of 30 km/h, "
                "and immediately returns along the exact same road from Warehouse B to Warehouse A "
                "at an average speed of 60 km/h. What is the average speed of the van for the entire "
                f"round trip in km/h? {suffix}"
            ),
            "gold_answer": "40",
            "trap_answers": ["45", "50", "90"],
        },
        {
            "problem_id": "trap_bat_ball_glove",
            "family": "cognitive_illusion",
            "template_id": "trap_three_item_shifted_crt_v1",
            "difficulty": "medium",
            "source_benchmark": "Modified CRT / Frederick Cognitive Reflection Test",
            "reasoning_pathology": "superficial_subtraction_heuristic",
            "prompt": (
                "A baseball bat, a ball, and a leather glove cost $150 in total. The glove costs $40. "
                "The bat costs $100 more than the ball. How many dollars does the ball cost? "
                f"{suffix}"
            ),
            "gold_answer": "5",
            "trap_answers": ["10", "25", "50"],
        },
        {
            "problem_id": "trap_lily_pad_halving",
            "family": "cognitive_illusion",
            "template_id": "trap_exponential_halving_day_v1",
            "difficulty": "medium",
            "source_benchmark": "Cognitive Reflection Test / Doubling Rate",
            "reasoning_pathology": "linear_halving_heuristic",
            "prompt": (
                "A patch of water lilies in a lake doubles in size every single day. If it takes exactly "
                "48 days for the patch to completely cover the entire surface of the lake, "
                f"on what day was the lake exactly half covered? {suffix}"
            ),
            "gold_answer": "47",
            "trap_answers": ["24", "46", "12"],
        },
        {
            "problem_id": "trap_monty_hall_4doors",
            "family": "cognitive_illusion",
            "template_id": "trap_monty_hall_4door_switch_v1",
            "difficulty": "hard",
            "source_benchmark": "4-Door Monty Hall Probability Problem",
            "reasoning_pathology": "equal_split_fallacy",
            "prompt": (
                "In a game show, there are 4 closed doors. Behind 1 door is a luxury car, and behind "
                "each of the other 3 doors is a goat. You initially choose Door 1. "
                "The host, who knows where the car is, opens Door 2 and Door 3, revealing goats behind both. "
                "The host then offers you the option to switch to Door 4. "
                "What is the probability (expressed as an integer percentage from 0 to 100) of winning "
                f"the car if you switch to Door 4? {suffix}"
            ),
            "gold_answer": "75",
            "trap_answers": ["50", "25", "33"],
        },
        # --- Family 4: rate_and_combinatoric_traps (SimpleBench & Parity Traps) ---
        {
            "problem_id": "trap_widget_machines",
            "family": "rate_and_combinatorics",
            "template_id": "trap_unit_rate_scaling_v1",
            "difficulty": "medium",
            "source_benchmark": "Cognitive Reflection Test / Machine Scaling",
            "reasoning_pathology": "superficial_proportionality_scaling",
            "prompt": (
                "If 5 identical automated machines take 5 minutes to manufacture 5 widgets, "
                "how many minutes does it take 100 identical automated machines to manufacture "
                f"100 widgets? {suffix}"
            ),
            "gold_answer": "5",
            "trap_answers": ["100", "20", "1"],
        },
        {
            "problem_id": "trap_clock_strikes",
            "family": "rate_and_combinatorics",
            "template_id": "trap_fencepost_interval_chimes_v1",
            "difficulty": "hard",
            "source_benchmark": "Classic Arithmetic Fencepost Riddle / SimpleBench",
            "reasoning_pathology": "fencepost_interval_neglect",
            "prompt": (
                "An antique tower clock takes 6 seconds to complete striking 4 o'clock (4 chimes separated "
                "by equal silent intervals). How many seconds does the clock take to complete striking "
                f"10 o'clock (10 chimes) at the exact same chiming rate? {suffix}"
            ),
            "gold_answer": "18",
            "trap_answers": ["15", "20", "12"],
        },
        {
            "problem_id": "trap_couple_handshakes",
            "family": "rate_and_combinatorics",
            "template_id": "trap_spouse_exclusion_handshakes_v1",
            "difficulty": "hard",
            "source_benchmark": "Discrete Combinatorics / SimpleBench Social Traps",
            "reasoning_pathology": "spouse_exclusion_neglect",
            "prompt": (
                "Three married couples (6 people in total) attend a dinner party. Each person shakes hands "
                "with every other person in the room exactly once, except that no one shakes hands "
                f"with their own spouse. How many total handshakes occur? {suffix}"
            ),
            "gold_answer": "12",
            "trap_answers": ["15", "9", "18"],
        },
        {
            "problem_id": "trap_rope_burn_timing",
            "family": "rate_and_combinatorics",
            "template_id": "trap_nonuniform_rope_burning_v1",
            "difficulty": "hard",
            "source_benchmark": "Non-uniform Rope Burning Puzzle",
            "reasoning_pathology": "halving_interpolation_error",
            "prompt": (
                "You have two identical ropes. Each rope burns completely in 60 minutes from end to end, "
                "but the ropes burn non-uniformly (different sections burn at different speeds). "
                "By lighting Rope 1 at both ends and Rope 2 at one end simultaneously, and lighting the "
                "second end of Rope 2 the moment Rope 1 finishes burning, you measure an exact duration. "
                f"How many total minutes of time elapse from the start until Rope 2 burns out completely? {suffix}"
            ),
            "gold_answer": "45",
            "trap_answers": ["60", "30", "15"],
        },
    ]

    _verify_tasks(tasks)
    return {
        "schema_version": 1,
        "suite": "deceptive_traps_v1",
        "purpose": (
            "Deceptive reasoning benchmark targeting cognitive illusions, arithmetic distractors (GSM-NoOp), "
            "reasoning rigidity (ReasoningTrap), and fencepost/scaling traps where S0 habitually outputs trap answers."
        ),
        "families": [
            "reasoning_rigidity",
            "gsm_noop_distractor",
            "cognitive_illusion",
            "rate_and_combinatorics",
        ],
        "tasks": tasks,
    }


def _verify_tasks(tasks: List[Dict[str, Any]]) -> None:
    seen_ids = set()
    for t in tasks:
        tid = t["problem_id"]
        assert tid not in seen_ids, f"Duplicate problem_id: {tid}"
        seen_ids.add(tid)
        gold = t["gold_answer"]
        assert isinstance(gold, str) and gold.strip(), f"Invalid gold_answer for {tid}"
        int(gold)  # must be integer
        traps = t.get("trap_answers", [])
        assert isinstance(traps, list) and len(traps) >= 1, f"Missing trap_answers for {tid}"
        for trap in traps:
            assert trap != gold, f"Trap answer '{trap}' cannot equal gold '{gold}' in {tid}"


def evaluate_deceptive_reasoning(row: Dict[str, Any], gold_answer: str) -> Dict[str, Any]:
    """Evaluate a generated deceptive reasoning completion against gold and trap answers."""
    final_text = row.get("final_text")
    completed = bool(row.get("completed", False))
    extracted = extract_literature_integer(final_text, completed=completed)

    trap_answers = set(row.get("trap_answers", []))
    hit_trap = extracted in trap_answers if extracted is not None else False

    if not completed and extracted is None:
        return {
            "status": "budget_exhausted",
            "extracted_answer": None,
            "is_correct": False,
            "hit_trap_answer": False,
            "gold_answer": gold_answer,
        }

    if extracted is None:
        return {
            "status": "invalid_format",
            "extracted_answer": None,
            "is_correct": False,
            "hit_trap_answer": False,
            "gold_answer": gold_answer,
        }

    is_correct = str(extracted).strip() == str(gold_answer).strip()
    return {
        "status": "correct" if is_correct else ("trap_failure" if hit_trap else "incorrect"),
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "hit_trap_answer": hit_trap,
        "gold_answer": gold_answer,
    }


def write_deceptive_tasks_config(path: Path) -> Dict[str, Any]:
    data = build_deceptive_reasoning_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
