"""Boolean logic evaluation and small CNF constraint satisfaction task generator."""

from __future__ import annotations

import itertools
import random
from typing import Dict, List

from tebdil.tasks.validators import TaskProblem, evaluate_cnf_clauses


_SPLIT_SEED_OFFSETS = {
    "train": 2000,
    "pilot": 6000,
    "validation": 12000,
    "test": 22000,
    "fresh_test": 42000,
}


def _solve_cnf_exhaustive(clauses: List[List[int]], num_vars: int) -> List[Dict[int, int]]:
    solutions = []
    for bits in itertools.product([0, 1], repeat=num_vars):
        assign = {i + 1: bits[i] for i in range(num_vars)}
        if evaluate_cnf_clauses(clauses, assign):
            solutions.append(assign)
    return solutions


def generate_logic_problems(
    split: str = "pilot",
    num_problems: int = 20,
    seed: int = 42,
) -> List[TaskProblem]:
    """Generate Boolean formula evaluation and small 2-SAT/3-SAT constraint tasks."""
    offset = _SPLIT_SEED_OFFSETS.get(split, 32000)
    rng = random.Random(seed + offset)
    problems: List[TaskProblem] = []

    for idx in range(num_problems):
        pid = f"logic_{split}_{idx:03d}"
        if idx % 2 == 0:
            # Boolean circuit evaluation with 3 variables A, B, C in {0, 1}
            a = rng.randint(0, 1)
            b = rng.randint(0, 1)
            c = rng.randint(0, 1)
            if split == "fresh_test":
                # Fresh unseen test template: (((NOT A) AND B) OR (A AND (NOT C)))
                val = int((((1 - a) and b) or (a and (1 - c))))
                expr_str = f"(((NOT {a}) AND {b}) OR ({a} AND (NOT {c})))"
                tmpl = "logic_mux_branch_fresh"
                prompt = (
                    f"Compute the truth value in {{0, 1}} of {expr_str} where 1=True and 0=False. "
                    f"Output only 0 or 1.\nAnswer: "
                )
            elif split != "test":
                # Validation/Pilot template: ((A AND B) XOR (NOT C))
                val = int(((a and b) ^ (1 - c)))
                expr_str = f"(({a} AND {b}) XOR (NOT {c}))"
                tmpl = "logic_bool_xor_not_v1"
                prompt = (
                    f"Evaluate the Boolean expression {expr_str} where 1 is True and 0 is False. "
                    f"Output only 0 or 1.\nAnswer: "
                )
            else:
                # Test template: ((A OR (NOT B)) AND (B XOR C))
                val = int(((a or (1 - b)) and (b ^ c)))
                expr_str = f"(({a} OR (NOT {b})) AND ({b} XOR {c}))"
                tmpl = "logic_bool_or_xor_test"
                prompt = (
                    f"Evaluate the Boolean expression {expr_str} where 1 is True and 0 is False. "
                    f"Output only 0 or 1.\nAnswer: "
                )
            problems.append(
                TaskProblem(
                    problem_id=pid,
                    family="logic",
                    split=split,
                    template_id=tmpl,
                    difficulty="medium",
                    prompt=prompt,
                    gold_answer=str(val),
                    candidate_answers=(str(val), str(1 - val)),
                    metadata={"logic_mode": "boolean_eval", "vars": [a, b, c]},
                )
            )
        else:
            # Small uniquely satisfiable 2-variable CNF where validator checks clause satisfaction
            target_a = rng.randint(0, 1)
            target_b = rng.randint(0, 1)
            lit_a = 1 if target_a == 1 else -1
            lit_b = 2 if target_b == 1 else -2
            # Construct 3 clauses uniquely satisfied by (target_a, target_b)
            clauses = [
                [lit_a, lit_b],
                [lit_a, -lit_b],
                [-lit_a, lit_b],
            ]
            sols = _solve_cnf_exhaustive(clauses, 2)
            assert len(sols) == 1
            sol = sols[0]
            gold_str = f"{sol[1]}{sol[2]}"

            def _fmt_lit(lit: int) -> str:
                name = "A" if abs(lit) == 1 else "B"
                return name if lit > 0 else f"NOT {name}"

            clause_strs = [f"({_fmt_lit(c[0])} OR {_fmt_lit(c[1])})" for c in clauses]
            formula_str = " AND ".join(clause_strs)
            if split == "fresh_test":
                prompt = (
                    f"Determine the unique binary assignment AB in {{00, 01, 10, 11}} making every clause true: {formula_str}. "
                    f"Output only the 2-character bitstring AB.\nAnswer: "
                )
                tmpl = "logic_2sat_horn_fresh"
            elif split != "test":
                prompt = (
                    f"Find binary values in {{0, 1}} for A and B that satisfy: {formula_str}. "
                    f"Output only the 2-digit bitstring AB (for example 10 means A=1, B=0).\nAnswer: "
                )
                tmpl = "logic_2sat_unique_v1"
            else:
                prompt = (
                    f"Satisfy all clauses for binary variables A, B in {{0, 1}}: {formula_str}. "
                    f"Output only the 2-digit bitstring AB.\nAnswer: "
                )
                tmpl = "logic_2sat_unique_test"

            distractors = [gold_str]
            for ba, bb in itertools.product([0, 1], repeat=2):
                cand = f"{ba}{bb}"
                if cand != gold_str:
                    distractors.append(cand)

            problems.append(
                TaskProblem(
                    problem_id=pid,
                    family="logic",
                    split=split,
                    template_id=tmpl,
                    difficulty="hard",
                    prompt=prompt,
                    gold_answer=gold_str,
                    candidate_answers=tuple(distractors),
                    metadata={
                        "logic_mode": "cnf_assignment",
                        "clauses": clauses,
                        "num_vars": 2,
                    },
                )
            )

    return problems
