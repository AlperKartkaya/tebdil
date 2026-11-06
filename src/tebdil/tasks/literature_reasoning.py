"""Literature-grounded LLM reasoning benchmark suite and programmatic verifiers.

Covers six research-established reasoning families designed to probe latent trajectory
revision, backtracking, and overthinking in Looped Language Models (Ouro-1.4B-Thinking):
1. crt_counter_trap: Modified Cognitive Reflection Test (CRT) counter-memorization traps
2. knights_and_knaves: Smullyan Boolean truth-teller/liar deduction requiring assumption backtracking
3. gsm8k_multistep: Multi-step word problems with retroactive constraints and rate changes
4. math_number_theory: Chinese Remainder Theorem (CRT) and simultaneous modular congruences
5. zebra_constraint: Linear ordering and position constraint satisfaction puzzles
6. graph_greedy_trap: Directed weighted shortest-path networks with greedy first-hop traps
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from tebdil.evaluation.completion import score_completion
from tebdil.tasks.completion_v2 import extract_completed_integer_v2
from tebdil.tasks.graph import dijkstra_shortest_path


def extract_literature_integer(final_text: str | None, *, completed: bool) -> str | None:
    """Extract the final integer answer from a closed reasoning completion."""
    if not completed or not isinstance(final_text, str) or not final_text.strip():
        return None
    strict = extract_completed_integer_v2(final_text, "", completed=True)
    if strict is not None:
        return strict

    cleaned = final_text.replace("**", "").strip()
    boxed = re.findall(r"\\boxed\{\s*([+-]?[0-9]+)\s*\}", cleaned)
    if len(boxed) == 1:
        return boxed[0]

    labeled = list(
        re.finditer(
            r"(?:Final\s+Answer|Answer|Result)\s*(?:is|=|:)\s*\$?\s*([+-]?[0-9]+)",
            cleaned,
            re.IGNORECASE,
        )
    )
    if labeled:
        return labeled[-1].group(1)

    eq_match = re.fullmatch(r"\s*[a-zA-Z_0-9]+\s*=\s*\$?\s*([+-]?[0-9]+)\s*\.?\s*", cleaned)
    if eq_match:
        return eq_match.group(1)

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        m_last = re.search(r"(?:is|=|:)\s*\$?\s*([+-]?[0-9]+)\s*\.?$", last_line, re.IGNORECASE)
        if m_last:
            return m_last.group(1)
        m_only = re.fullmatch(r"\$?\s*([+-]?[0-9]+)\s*\.?", last_line)
        if m_only:
            return m_only.group(1)
    return None


def evaluate_literature_reasoning(record: Dict[str, Any], gold_answer: str) -> Dict[str, Any]:
    """Evaluate a prompt-started reasoning record against its exact integer gold answer."""
    answer = None
    if record.get("completed") and record.get("reasoning_status") == "closed":
        answer = extract_literature_integer(record.get("final_text"), completed=True)
    return {
        "answer_extraction": "literature_reasoning_integer_v1",
        "extracted_answer": answer,
        "format_valid": answer is not None,
        **score_completion(
            {"completed": bool(record.get("completed")), "extracted_answer": answer},
            str(gold_answer),
        ),
    }


def _verify_all_tasks(partitions: Dict[str, List[Dict[str, Any]]]) -> None:
    """Programmatically verify every task's mathematical or logical gold answer."""
    dev = {t["problem_id"]: t for t in partitions["development"]}
    test = {t["problem_id"]: t for t in partitions["reserved_test"]}

    # 1. CRT Counter-Traps
    # lit_dev_crt_000: 6 machines, 6 min -> 6 gears => 1 gear / (machine * 6 min).
    # 18 machines in m minutes -> 18 * (m / 6) = 54 => m = 18.
    assert int(dev["lit_dev_crt_000"]["gold_answer"]) == (54 * 6) // 18 == 18
    # lit_dev_crt_001: triples daily, full on day 30 -> 1/9 on day 30 - log_3(9) = 28.
    assert int(dev["lit_dev_crt_001"]["gold_answer"]) == 30 - 2 == 28
    # lit_test_crt_000: jacket + hat = 150, jacket = hat + 120 => 2*hat + 120 = 150 => hat = 15.
    assert int(test["lit_test_crt_000"]["gold_answer"]) == (150 - 120) // 2 == 15

    # 2. Knights & Knaves (exhaustive truth-table verification)
    kk_sol_0 = [
        (a, b, c)
        for a, b, c in itertools.product([0, 1], repeat=3)
        if (a == int(b == 0 or c == 0)) and (b == int(a == 0)) and (c == int(b == 0))
    ]
    assert kk_sol_0 == [(1, 0, 1)]
    assert dev["lit_dev_kk_000"]["gold_answer"] == "101"

    kk_sol_1 = [
        (a, b, c)
        for a, b, c in itertools.product([0, 1], repeat=3)
        if (a == int(a == b)) and (b == int((a == 0) + (c == 0) == 1)) and (c == int(a == 0 and b == 0))
    ]
    assert kk_sol_1 == [(1, 1, 0)]
    assert dev["lit_dev_kk_001"]["gold_answer"] == "110"

    kk_sol_test = [
        (a, b, c)
        for a, b, c in itertools.product([0, 1], repeat=3)
        if (a == int(b == 0)) and (b == int(c == 0)) and (c == int(not (a == 1 and b == 1)))
    ]
    assert kk_sol_test == [(1, 0, 1)]
    assert test["lit_test_kk_000"]["gold_answer"] == "101"

    # 3. GSM8K Multi-Step
    assert int(dev["lit_dev_gsm_000"]["gold_answer"]) == (15 * 30 + (40 - 15 - 5) * (30 - 6) - 450) == 480
    assert int(dev["lit_dev_gsm_001"]["gold_answer"]) == int((120 + 4 * (35 - 20)) * (2 / 3)) == 120
    assert int(test["lit_test_gsm_000"]["gold_answer"]) == (24 * 12 - 18) // 9 * 5 == 150

    # 4. MATH / AIME Number Theory
    n0 = next(n for n in range(11, 500) if n % 3 == 2 and n % 5 == 3 and n % 7 == 2)
    assert n0 == 23 and dev["lit_dev_nt_000"]["gold_answer"] == "23"
    n1 = next(n for n in range(10, 500) if (4 * n + 3) % 11 == 0 and (3 * n + 1) % 7 == 0)
    assert n1 == 79 and dev["lit_dev_nt_001"]["gold_answer"] == "79"
    nt_test = next(n for n in range(10, 500) if n % 4 == 3 and n % 5 == 4 and n % 6 == 5)
    assert nt_test == 59 and test["lit_test_nt_000"]["gold_answer"] == "59"

    # 5. Zebra / Linear Ordering Constraints
    z0 = [
        p
        for p in itertools.permutations([1, 2, 3, 4, 5])
        if (p[0] == 4 or p[4] == 4)
        and abs(p.index(2) - p.index(4)) == 1
        and p.index(1) < p.index(3)
        and abs(p.index(5) - p.index(2)) == 1
        and abs(p.index(5) - p.index(1)) == 1
        and (p.index(2) + p.index(1)) == 2 * p.index(5)
        and p[4] == 3
    ]
    assert z0 == [(4, 2, 5, 1, 3)]
    assert dev["lit_dev_zebra_000"]["gold_answer"] == "42513"

    z1 = [
        p
        for p in itertools.permutations([1, 2, 3, 4])
        if p.index(2) - p.index(1) == 2
        and p[0] != 3
        and abs(p.index(4) - p.index(1)) == 1
        and p.index(3) > p.index(2)
    ]
    assert z1 == [(1, 4, 2, 3)]
    assert dev["lit_dev_zebra_001"]["gold_answer"] == "1423"

    z_test = [
        p
        for p in itertools.permutations([1, 2, 3, 4])
        if p.index(3) - p.index(1) == 2
        and abs(p.index(4) - p.index(3)) == 1
        and p.index(2) < p.index(1)
    ]
    assert z_test == [(2, 1, 4, 3)]
    assert test["lit_test_zebra_000"]["gold_answer"] == "2143"

    # 6. Graph Greedy Trap
    g0_edges = [(0, 1, 2), (0, 2, 5), (1, 3, 14), (1, 4, 12), (2, 3, 3), (2, 4, 4), (3, 5, 6), (4, 5, 3)]
    assert dijkstra_shortest_path(6, g0_edges, 0, 5) == 12
    assert dev["lit_dev_graph_000"]["gold_answer"] == "12"

    g1_edges = [(0, 1, 3), (0, 2, 6), (1, 2, 2), (1, 3, 11), (2, 3, 3), (2, 4, 5), (3, 5, 4), (4, 5, 3)]
    assert dijkstra_shortest_path(6, g1_edges, 0, 5) == 12
    assert dev["lit_dev_graph_001"]["gold_answer"] == "12"

    gt_edges = [(0, 1, 2), (0, 2, 6), (1, 3, 13), (1, 2, 3), (2, 3, 4), (2, 4, 5), (3, 5, 5), (4, 5, 3)]
    assert dijkstra_shortest_path(6, gt_edges, 0, 5) == 13
    assert test["lit_test_graph_000"]["gold_answer"] == "13"


def build_literature_reasoning_dataset() -> Dict[str, Any]:
    """Construct the verified literature reasoning benchmark suite."""
    suffix = "Be concise. Output only the integer result.\nAnswer: "
    development = [
        {
            "problem_id": "lit_dev_crt_000",
            "family": "crt_counter_trap",
            "split": "development",
            "template_id": "crt_widget_scaling_v1",
            "difficulty": "medium",
            "source_benchmark": "Modified Cognitive Reflection Test (Frederick 2005 / Modified-CRT)",
            "reasoning_pathology": "pattern_matching_vs_rate_decomposition",
            "prompt": (
                "In a precision factory, 6 identical machines working simultaneously take 6 minutes "
                "to manufacture 6 gears. At the exact same individual production rate, how many minutes "
                "does it take 18 identical machines working simultaneously to manufacture 54 gears? "
                f"{suffix}"
            ),
            "gold_answer": "18",
            "trap_answers": ["6", "54", "9"],
        },
        {
            "problem_id": "lit_dev_crt_001",
            "family": "crt_counter_trap",
            "split": "development",
            "template_id": "crt_exponential_tripling_v1",
            "difficulty": "medium",
            "source_benchmark": "Modified Cognitive Reflection Test (Exponential Tripling Counter-Trap)",
            "reasoning_pathology": "memorized_lilypad_heuristic_override",
            "prompt": (
                "An algal bloom in a test basin triples in surface area every day. If the bloom "
                "completely covers the entire basin on day 30, on which day did the bloom cover "
                f"exactly one-ninth (1/9) of the basin? {suffix}"
            ),
            "gold_answer": "28",
            "trap_answers": ["29", "10", "27"],
        },
        {
            "problem_id": "lit_dev_kk_000",
            "family": "knights_and_knaves",
            "split": "development",
            "template_id": "kk_3islanders_disjunction_v1",
            "difficulty": "hard",
            "source_benchmark": "Smullyan Knights & Knaves / LLM Logic Reasoning Benchmark",
            "reasoning_pathology": "assumption_contradiction_backtracking",
            "prompt": (
                "On an island, every inhabitant is either a Knight (who always tells the truth, "
                "digit 1) or a Knave (who always lies, digit 0). You meet three inhabitants: "
                "Alice (A), Bob (B), and Carol (C).\n"
                "- Alice says: 'At least one of Bob or Carol is a Knave.'\n"
                "- Bob says: 'Alice is a Knave.'\n"
                "- Carol says: 'Bob is a Knave.'\n"
                "Determine the truth values of A, B, and C, and express your answer as the 3-digit "
                f"integer ABC (for example, 101 means A=1, B=0, C=1). {suffix}"
            ),
            "gold_answer": "101",
            "trap_answers": ["110", "100", "010"],
        },
        {
            "problem_id": "lit_dev_kk_001",
            "family": "knights_and_knaves",
            "split": "development",
            "template_id": "kk_3islanders_parity_v1",
            "difficulty": "hard",
            "source_benchmark": "Smullyan Knights & Knaves / LLM Logic Reasoning Benchmark",
            "reasoning_pathology": "biconditional_collapse_and_parity",
            "prompt": (
                "On an island, every inhabitant is either a Knight (always tells the truth, digit 1) "
                "or a Knave (always lies, digit 0). You meet three inhabitants: Alice (A), Bob (B), "
                "and Carol (C).\n"
                "- Alice says: 'Bob and I are of the same type.'\n"
                "- Bob says: 'Exactly one of Alice or Carol is a Knave.'\n"
                "- Carol says: 'Alice and Bob are both Knaves.'\n"
                "Determine the truth values of A, B, and C, and express your answer as the 3-digit "
                f"integer ABC. {suffix}"
            ),
            "gold_answer": "110",
            "trap_answers": ["101", "111", "010"],
        },
        {
            "problem_id": "lit_dev_gsm_000",
            "family": "gsm8k_multistep",
            "split": "development",
            "template_id": "gsm_retroactive_defect_profit_v1",
            "difficulty": "medium",
            "source_benchmark": "GSM8K / GSM-Symbolic Multi-Step Arithmetic Reasoning",
            "reasoning_pathology": "retroactive_inventory_subtraction",
            "prompt": (
                "A workshop produces 40 wooden chairs. It sells the first 15 chairs at $30 each. "
                "Of the remaining chairs, 5 are damaged and cannot be sold, and the rest are sold "
                "at a $6 discount off the original $30 price. If the workshop spent $450 on materials "
                f"for all 40 chairs, what is the workshop's net profit in dollars? {suffix}"
            ),
            "gold_answer": "480",
            "trap_answers": ["600", "510", "930"],
        },
        {
            "problem_id": "lit_dev_gsm_001",
            "family": "gsm8k_multistep",
            "split": "development",
            "template_id": "gsm_coupled_daily_flow_v1",
            "difficulty": "medium",
            "source_benchmark": "GSM8K / Multi-Step Rate Accumulation",
            "reasoning_pathology": "multi_day_net_flow_and_fractional_depletion",
            "prompt": (
                "A warehouse starts Monday morning with 120 crates. From Monday through Thursday "
                "(4 days), it receives 35 new crates each morning and ships out 20 crates each evening. "
                "On Friday, it receives no new crates and ships out one-third of all crates currently "
                "in the warehouse. How many crates remain in the warehouse at the end of Friday? "
                f"{suffix}"
            ),
            "gold_answer": "120",
            "trap_answers": ["60", "180", "130"],
        },
        {
            "problem_id": "lit_dev_nt_000",
            "family": "math_number_theory",
            "split": "development",
            "template_id": "math_crt_sunzi_v1",
            "difficulty": "hard",
            "source_benchmark": "MATH / AIME Number Theory (Chinese Remainder Theorem)",
            "reasoning_pathology": "modular_congruence_search_loop",
            "prompt": (
                "Find the smallest positive integer n greater than 10 such that n leaves a remainder "
                "of 2 when divided by 3, a remainder of 3 when divided by 5, and a remainder of 2 "
                f"when divided by 7. {suffix}"
            ),
            "gold_answer": "23",
            "trap_answers": ["38", "53", "128"],
        },
        {
            "problem_id": "lit_dev_nt_001",
            "family": "math_number_theory",
            "split": "development",
            "template_id": "math_linear_congruence_pair_v1",
            "difficulty": "hard",
            "source_benchmark": "MATH / AIME Modular Diophantine Equations",
            "reasoning_pathology": "single_digit_boundary_constraint",
            "prompt": (
                "What is the smallest positive two-digit integer n such that 4n + 3 is divisible "
                f"by 11 and 3n + 1 is divisible by 7? {suffix}"
            ),
            "gold_answer": "79",
            "trap_answers": ["2", "13", "68"],
        },
        {
            "problem_id": "lit_dev_zebra_000",
            "family": "zebra_constraint",
            "split": "development",
            "template_id": "zebra_5box_ordering_v1",
            "difficulty": "hard",
            "source_benchmark": "ZebraLogic / Linear Constraint Satisfaction Benchmark",
            "reasoning_pathology": "spatial_adjacency_and_endpoint_elimination",
            "prompt": (
                "Five boxes (labeled 1, 2, 3, 4, 5) are arranged in a row from left (position 1) "
                "to right (position 5).\n"
                "- Box 4 is at one of the two ends (position 1 or position 5), and Box 2 is immediately adjacent to Box 4.\n"
                "- Box 1 is somewhere to the left of Box 3.\n"
                "- Box 5 is directly between Box 2 and Box 1 (so Box 2, Box 5, and Box 1 occupy three consecutive positions).\n"
                "- Box 3 is at position 5.\n"
                "Compute the 5-digit integer formed by the box labels from left to right "
                f"(positions 1 to 5). {suffix}"
            ),
            "gold_answer": "42513",
            "trap_answers": ["42153", "31524", "45213"],
        },
        {
            "problem_id": "lit_dev_zebra_001",
            "family": "zebra_constraint",
            "split": "development",
            "template_id": "zebra_4floor_assignment_v1",
            "difficulty": "hard",
            "source_benchmark": "ZebraLogic / Floor Assignment Constraint Satisfaction",
            "reasoning_pathology": "two_branch_case_split_elimination",
            "prompt": (
                "Four researchers (numbered 1, 2, 3, 4) occupy four different floors (floors 1, 2, 3, 4 "
                "from bottom to top) of an institute.\n"
                "- Researcher 2 is on a higher floor than Researcher 1, with exactly one floor between them.\n"
                "- Researcher 3 is on a higher floor than Researcher 2.\n"
                "- Researcher 4 is on a floor immediately adjacent to Researcher 1.\n"
                "Determine the 4-digit integer formed by the researcher numbers from floor 1 to floor 4. "
                f"{suffix}"
            ),
            "gold_answer": "1423",
            "trap_answers": ["4132", "1324", "2413"],
        },
        {
            "problem_id": "lit_dev_graph_000",
            "family": "graph_greedy_trap",
            "split": "development",
            "template_id": "graph_6node_bottleneck_trap_v1",
            "difficulty": "medium",
            "source_benchmark": "CLRS / Graph Reasoning Benchmark (Dijkstra Greedy Trap)",
            "reasoning_pathology": "greedy_first_hop_tunnel_vision",
            "prompt": (
                "Consider a directed network with 6 nodes (0, 1, 2, 3, 4, 5) and weighted directed edges: "
                "0->1 (cost 2), 0->2 (cost 5), 1->3 (cost 14), 1->4 (cost 12), 2->3 (cost 3), "
                "2->4 (cost 4), 3->5 (cost 6), 4->5 (cost 3). Find the minimum total path cost "
                f"from node 0 to node 5. {suffix}"
            ),
            "gold_answer": "12",
            "trap_answers": ["14", "17", "22"],
        },
        {
            "problem_id": "lit_dev_graph_001",
            "family": "graph_greedy_trap",
            "split": "development",
            "template_id": "graph_6node_relay_shortcut_v1",
            "difficulty": "hard",
            "source_benchmark": "CLRS / Multi-Hop Relay Shortest-Path Benchmark",
            "reasoning_pathology": "missing_cross_edge_relaxation",
            "prompt": (
                "Consider a directed network with 6 nodes (0, 1, 2, 3, 4, 5) and weighted directed edges: "
                "0->1 (cost 3), 0->2 (cost 6), 1->2 (cost 2), 1->3 (cost 11), 2->3 (cost 3), "
                "2->4 (cost 5), 3->5 (cost 4), 4->5 (cost 3). Find the minimum total path cost "
                f"from node 0 to node 5. {suffix}"
            ),
            "gold_answer": "12",
            "trap_answers": ["13", "14", "18"],
        },
    ]

    reserved_test = [
        {
            "problem_id": "lit_test_crt_000",
            "family": "crt_counter_trap",
            "split": "reserved_test",
            "template_id": "crt_bat_ball_shifted_v1",
            "difficulty": "medium",
            "source_benchmark": "Modified Cognitive Reflection Test",
            "reasoning_pathology": "impulsive_subtraction_trap",
            "prompt": (
                "A winter jacket and a wool hat cost $150 in total. The jacket costs $120 more "
                f"than the hat. How many dollars does the wool hat cost? {suffix}"
            ),
            "gold_answer": "15",
            "trap_answers": ["30", "25", "20"],
        },
        {
            "problem_id": "lit_test_kk_000",
            "family": "knights_and_knaves",
            "split": "reserved_test",
            "template_id": "kk_3islanders_chain_v1",
            "difficulty": "hard",
            "source_benchmark": "Smullyan Knights & Knaves",
            "reasoning_pathology": "contradiction_backtracking",
            "prompt": (
                "On an island of Knights (always tell the truth, digit 1) and Knaves (always lie, digit 0), "
                "you meet Alice (A), Bob (B), and Carol (C).\n"
                "- Alice says: 'Bob is a Knave.'\n"
                "- Bob says: 'Carol is a Knave.'\n"
                "- Carol says: 'Alice and Bob are not both Knights.'\n"
                f"Express the truth values of A, B, and C as the 3-digit integer ABC. {suffix}"
            ),
            "gold_answer": "101",
            "trap_answers": ["010", "100", "110"],
        },
        {
            "problem_id": "lit_test_gsm_000",
            "family": "gsm8k_multistep",
            "split": "reserved_test",
            "template_id": "gsm_repackaging_loss_v1",
            "difficulty": "medium",
            "source_benchmark": "GSM8K Multi-Step Word Problems",
            "reasoning_pathology": "multi_stage_unit_conversion",
            "prompt": (
                "A bakery bakes 24 trays of muffins, with 12 muffins on each tray. After cooling, "
                "18 muffins are set aside for staff, and the remaining muffins are packed into boxes "
                "of 9 muffins each. If every full box is sold for $5, how many dollars does the bakery "
                f"collect from selling all the boxes? {suffix}"
            ),
            "gold_answer": "150",
            "trap_answers": ["160", "145", "135"],
        },
        {
            "problem_id": "lit_test_nt_000",
            "family": "math_number_theory",
            "split": "reserved_test",
            "template_id": "math_lcm_minus_one_crt_v1",
            "difficulty": "hard",
            "source_benchmark": "MATH / AIME Number Theory",
            "reasoning_pathology": "common_negative_remainder_recognition",
            "prompt": (
                "Find the smallest positive integer n greater than 10 such that n leaves a remainder "
                "of 3 when divided by 4, a remainder of 4 when divided by 5, and a remainder of 5 "
                f"when divided by 6. {suffix}"
            ),
            "gold_answer": "59",
            "trap_answers": ["119", "29", "49"],
        },
        {
            "problem_id": "lit_test_zebra_000",
            "family": "zebra_constraint",
            "split": "reserved_test",
            "template_id": "zebra_4slot_schedule_v1",
            "difficulty": "hard",
            "source_benchmark": "ZebraLogic Constraint Satisfaction",
            "reasoning_pathology": "relative_offset_constraint_propagation",
            "prompt": (
                "Four speakers (numbered 1, 2, 3, 4) are scheduled in four consecutive slots "
                "(slots 1, 2, 3, 4 from first to last).\n"
                "- Speaker 3 speaks exactly two slots after Speaker 1.\n"
                "- Speaker 4 speaks in a slot immediately adjacent to Speaker 3.\n"
                "- Speaker 2 speaks in an earlier slot than Speaker 1.\n"
                "Determine the 4-digit integer formed by the speaker numbers from slot 1 to slot 4. "
                f"{suffix}"
            ),
            "gold_answer": "2143",
            "trap_answers": ["1432", "2134", "4123"],
        },
        {
            "problem_id": "lit_test_graph_000",
            "family": "graph_greedy_trap",
            "split": "reserved_test",
            "template_id": "graph_6node_bridge_v1",
            "difficulty": "hard",
            "source_benchmark": "CLRS / Shortest-Path Benchmark",
            "reasoning_pathology": "multi_hop_bridge_relaxation",
            "prompt": (
                "Consider a directed network with 6 nodes (0, 1, 2, 3, 4, 5) and weighted directed edges: "
                "0->1 (cost 2), 0->2 (cost 6), 1->3 (cost 13), 1->2 (cost 3), 2->3 (cost 4), "
                "2->4 (cost 5), 3->5 (cost 5), 4->5 (cost 3). Find the minimum total path cost "
                f"from node 0 to node 5. {suffix}"
            ),
            "gold_answer": "13",
            "trap_answers": ["14", "15", "20"],
        },
    ]

    partitions = {"development": development, "reserved_test": reserved_test}
    _verify_all_tasks(partitions)
    return {
        "schema_version": 1,
        "suite": "literature_reasoning_v1",
        "purpose": (
            "Literature-grounded LLM reasoning benchmark covering Modified CRT traps, "
            "Smullyan Knights & Knaves Boolean backtracking, GSM8K multi-step word problems, "
            "MATH/AIME number theory, ZebraLogic linear constraints, and CLRS shortest-path greedy traps."
        ),
        "families": [
            "crt_counter_trap",
            "knights_and_knaves",
            "gsm8k_multistep",
            "math_number_theory",
            "zebra_constraint",
            "graph_greedy_trap",
        ],
        "partitions": partitions,
    }


def write_literature_reasoning_config(path: Path) -> Dict[str, Any]:
    data = build_literature_reasoning_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
