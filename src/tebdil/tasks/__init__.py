"""Task generation and formal validation suite for TEBDIL."""

from __future__ import annotations

from typing import List

from tebdil.tasks.arithmetic import generate_arithmetic_problems
from tebdil.tasks.graph import dijkstra_shortest_path, generate_graph_problems
from tebdil.tasks.logic import generate_logic_problems
from tebdil.tasks.validators import (
    TaskProblem,
    ValidationOutcome,
    evaluate_cnf_clauses,
    extract_first_int,
    validate_arithmetic,
    validate_graph,
    validate_logic,
    validate_problem_response,
)


def generate_task_suite(
    family: str = "arithmetic",
    split: str = "pilot",
    num_problems: int = 20,
    seed: int = 42,
) -> List[TaskProblem]:
    """Generate problems for `arithmetic`, `logic`, `graph`, or `all` (mixed balanced)."""
    if family == "arithmetic":
        return generate_arithmetic_problems(split=split, num_problems=num_problems, seed=seed)
    if family == "logic":
        return generate_logic_problems(split=split, num_problems=num_problems, seed=seed)
    if family == "graph":
        return generate_graph_problems(split=split, num_problems=num_problems, seed=seed)
    if family == "all":
        per_fam = max(1, num_problems // 3)
        rem = num_problems - 3 * per_fam
        probs = []
        probs.extend(generate_arithmetic_problems(split=split, num_problems=per_fam + (1 if rem > 0 else 0), seed=seed))
        probs.extend(generate_logic_problems(split=split, num_problems=per_fam + (1 if rem > 1 else 0), seed=seed))
        probs.extend(generate_graph_problems(split=split, num_problems=per_fam, seed=seed))
        return probs[:num_problems]
    raise ValueError(f"Unsupported task family: {family}")


__all__ = [
    "TaskProblem",
    "ValidationOutcome",
    "dijkstra_shortest_path",
    "evaluate_cnf_clauses",
    "extract_first_int",
    "generate_arithmetic_problems",
    "generate_graph_problems",
    "generate_logic_problems",
    "generate_task_suite",
    "validate_arithmetic",
    "validate_graph",
    "validate_logic",
    "validate_problem_response",
]
