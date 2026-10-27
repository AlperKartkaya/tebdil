"""Independent formal validators and problem schemas for TEBDIL tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskProblem:
    problem_id: str
    family: str  # "arithmetic" | "logic" | "graph"
    split: str   # "pilot" | "validation" | "test" | "train"
    template_id: str
    difficulty: str
    prompt: str
    gold_answer: str
    candidate_answers: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationOutcome:
    is_valid: bool
    extracted_answer: str
    reason: str


_INT_PATTERN = re.compile(r"[-+]?\d+")
_ASSIGN_PATTERN = re.compile(r"([A-C])\s*=\s*([01]|true|false|T|F)", re.IGNORECASE)


def extract_first_int(text: str) -> Optional[int]:
    """Extract the first signed integer from decoded continuation text."""
    cleaned = text.strip()
    if not cleaned:
        return None
    first_line = cleaned.splitlines()[0].strip()
    match = _INT_PATTERN.search(first_line)
    if match is None:
        match = _INT_PATTERN.search(cleaned)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def validate_arithmetic(problem: TaskProblem, response: str) -> ValidationOutcome:
    """Validate an arithmetic problem against exact mathematical solution."""
    val = extract_first_int(response)
    if val is None:
        return ValidationOutcome(
            is_valid=False,
            extracted_answer="",
            reason="no_integer_found",
        )
    expected = int(problem.gold_answer)
    # Also verify equation constraint directly if present in metadata
    if "linear_eq" in problem.metadata:
        a, b, c = problem.metadata["linear_eq"]  # a * x + b = c
        is_ok = (a * val + b == c)
    else:
        is_ok = (val == expected)
    return ValidationOutcome(
        is_valid=is_ok,
        extracted_answer=str(val),
        reason="exact_match" if is_ok else f"expected_{expected}_got_{val}",
    )


def evaluate_cnf_clauses(
    clauses: List[List[int]],
    assignment: Dict[int, int],
) -> bool:
    """Evaluate a CNF formula where literal +k means x_k=1 and -k means x_k=0."""
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var_idx = abs(lit)
            if var_idx not in assignment:
                return False
            val = assignment[var_idx]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                clause_sat = True
                break
        if not clause_sat:
            return False
    return True


def validate_logic(problem: TaskProblem, response: str) -> ValidationOutcome:
    """Validate Boolean deduction or CNF satisfaction problems using formal evaluator."""
    cleaned = response.strip()
    mode = problem.metadata.get("logic_mode", "boolean_eval")

    if mode == "cnf_assignment":
        clauses: List[List[int]] = problem.metadata["clauses"]
        num_vars: int = problem.metadata.get("num_vars", 2)
        assign: Dict[int, int] = {}
        for var_name, raw_val in _ASSIGN_PATTERN.findall(cleaned):
            idx = ord(var_name.upper()) - ord("A") + 1
            assign[idx] = 1 if raw_val.lower() in ("1", "true", "t") else 0
        if len(assign) < num_vars:
            bits = re.findall(r"[01]", cleaned)
            if len(bits) >= num_vars:
                for i in range(num_vars):
                    assign[i + 1] = int(bits[i])
        if len(assign) < num_vars:
            return ValidationOutcome(False, "", "incomplete_assignment")
        sat = evaluate_cnf_clauses(clauses, assign)
        formatted = ",".join(f"{chr(ord('A')+i-1)}={assign[i]}" for i in range(1, num_vars + 1))
        return ValidationOutcome(
            is_valid=sat,
            extracted_answer=formatted,
            reason="clauses_satisfied" if sat else "clause_violation",
        )

    # Boolean formula evaluation (0 or 1)
    val = extract_first_int(cleaned)
    if val is None or val not in (0, 1):
        low = cleaned.lower()
        if low.startswith("true") or low.startswith("yes"):
            val = 1
        elif low.startswith("false") or low.startswith("no"):
            val = 0
    if val is None or val not in (0, 1):
        return ValidationOutcome(False, "", "invalid_boolean_output")
    expected = int(problem.gold_answer)
    is_ok = (val == expected)
    return ValidationOutcome(
        is_valid=is_ok,
        extracted_answer=str(val),
        reason="exact_boolean_match" if is_ok else f"expected_{expected}_got_{val}",
    )


def validate_graph(problem: TaskProblem, response: str) -> ValidationOutcome:
    """Validate shortest-path or reachability answer against exact Dijkstra/BFS solution."""
    val = extract_first_int(response)
    if val is None:
        return ValidationOutcome(False, "", "no_integer_distance_found")
    expected = int(problem.gold_answer)
    is_ok = (val == expected)
    return ValidationOutcome(
        is_valid=is_ok,
        extracted_answer=str(val),
        reason="exact_graph_distance" if is_ok else f"expected_{expected}_got_{val}",
    )


def validate_problem_response(problem: TaskProblem, response: str) -> ValidationOutcome:
    """Dispatch validation to the appropriate formal family validator."""
    if problem.family == "arithmetic":
        return validate_arithmetic(problem, response)
    if problem.family == "logic":
        return validate_logic(problem, response)
    if problem.family == "graph":
        return validate_graph(problem, response)
    raise ValueError(f"Unknown problem family: {problem.family}")
