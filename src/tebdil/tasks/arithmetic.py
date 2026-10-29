"""Arithmetic and symbolic algebraic task generator with split-disjoint templates."""

from __future__ import annotations

import random
from typing import List

from tebdil.tasks.validators import TaskProblem


_SPLIT_SEED_OFFSETS = {
    "train": 1000,
    "pilot": 5000,
    "validation": 10000,
    "test": 20000,
    "fresh_test": 40000,
}


def generate_arithmetic_problems(
    split: str = "pilot",
    num_problems: int = 20,
    seed: int = 42,
) -> List[TaskProblem]:
    """Generate split-isolated arithmetic and symbolic reasoning problems.

    Templates are separated between validation/pilot and test to prevent template overfitting.
    """
    offset = _SPLIT_SEED_OFFSETS.get(split, 30000)
    rng = random.Random(seed + offset)
    problems: List[TaskProblem] = []

    for idx in range(num_problems):
        pid = f"arith_{split}_{idx:03d}"
        if split in ("pilot", "validation", "train"):
            kind = idx % 3
            if kind == 0:
                # Template A1: Two-step affine cascade (a * b + c - d) in [11, 97]
                a = rng.randint(3, 9)
                b = rng.randint(3, 9)
                c = rng.randint(4, 18)
                d = rng.randint(2, 12)
                ans = a * b + c - d
                distractors = (str(ans), str(a * b + c), str(a * (b + c) - d), str(ans + 2))
                prompt = (
                    f"Compute the exact value of ({a} * {b}) + {c} - {d}. "
                    f"Output only the integer result.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_affine_cascade_v1",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [a, b, c, d], "op": "a*b+c-d"},
                    )
                )
            elif kind == 1:
                # Template A2: Linear equation solving a*x + b = c
                a = rng.randint(2, 9)
                x_val = rng.randint(3, 16)
                b = rng.randint(4, 29)
                c = a * x_val + b
                distractors = (str(x_val), str(x_val + 1), str(x_val - 1), str((c + b) // a))
                prompt = (
                    f"Solve for x: {a} * x + {b} = {c}. "
                    f"Output only the integer value of x.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_linear_eq_v1",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(x_val),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"linear_eq": [a, b, c]},
                    )
                )
            else:
                # Template A3: Modular remainder after product-sum
                a = rng.randint(7, 19)
                b = rng.randint(5, 13)
                c = rng.randint(3, 11)
                m = rng.choice([7, 9, 11, 13])
                ans = (a * b + c) % m
                distractors = (str(ans), str((a + b * c) % m), str((a * b) % m), str((ans + 1) % m))
                prompt = (
                    f"Find the remainder when ({a} * {b} + {c}) is divided by {m}. "
                    f"Output only the integer remainder.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_mod_product_v1",
                        difficulty="hard",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [a, b, c, m], "op": "(a*b+c)%m"},
                    )
                )
        elif split == "fresh_test":
            # Fresh unseen test templates (never used during development or controller calibration)
            kind = idx % 3
            if kind == 0:
                # Fresh Template F1: Exact quotient plus quadratic offset ((a * b) // a + c * d - e)
                q = rng.randint(4, 12)
                m = rng.randint(3, 8)
                c = rng.randint(3, 9)
                d = rng.randint(2, 6)
                e = rng.randint(3, 11)
                dividend = q * m
                ans = q + (c * d) - e
                distractors = (str(ans), str(q + c * d), str(dividend + c * d - e), str(ans + 2))
                prompt = (
                    f"Determine the integer result of ({dividend} / {m}) + ({c} * {d}) - {e}. "
                    f"Output only the integer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_quotient_offset_fresh",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [dividend, m, c, d, e], "op": "(dividend/m)+(c*d)-e"},
                    )
                )
            elif kind == 1:
                # Fresh Template F2: Two-stage coupled substitution u = a + b, v = c * u - d
                a = rng.randint(3, 9)
                b = rng.randint(2, 8)
                c = rng.randint(3, 6)
                d = rng.randint(4, 15)
                u_val = a + b
                ans = c * u_val - d
                distractors = (str(ans), str(c * a + b - d), str(c * u_val + d), str(ans - c))
                prompt = (
                    f"Let u = {a} + {b} and v = {c} * u - {d}. What is the integer value of v? "
                    f"Output only the integer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_two_var_system_fresh",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [a, b, c, d], "op": "c*(a+b)-d"},
                    )
                )
            else:
                # Fresh Template F3: Nested difference of scaled sums a*(b + c) - d*(e + f)
                a = rng.randint(4, 8)
                b = rng.randint(3, 7)
                c = rng.randint(2, 6)
                d = rng.randint(2, 3)
                e = rng.randint(2, 4)
                f = rng.randint(1, 3)
                ans = a * (b + c) - d * (e + f)
                distractors = (str(ans), str(a * b + c - d * e + f), str(a * (b + c)), str(ans + d))
                prompt = (
                    f"Compute {a} * ({b} + {c}) - {d} * ({e} + {f}). "
                    f"Output only the final integer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_nested_diff_fresh",
                        difficulty="hard",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [a, b, c, d, e, f], "op": "a*(b+c)-d*(e+f)"},
                    )
                )
        else:
            # Held-out test templates with distinct structural forms
            kind = idx % 3
            if kind == 0:
                # Test Template T1: Difference of products (a * b) - (c * d) in [5, 95]
                a = rng.randint(5, 11)
                b = rng.randint(5, 9)
                c = rng.randint(2, 4)
                d = rng.randint(2, 5)
                ans = (a * b) - (c * d)
                distractors = (str(ans), str(a * b + c * d), str((a - c) * (b - d)), str(ans - c))
                prompt = (
                    f"Evaluate ({a} * {b}) - ({c} * {d}). "
                    f"Output only the final integer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_diff_of_products_test",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [a, b, c, d], "op": "a*b-c*d"},
                    )
                )
            elif kind == 1:
                # Test Template T2: Subtracted linear equation a*x - b = c
                a = rng.randint(3, 11)
                x_val = rng.randint(4, 18)
                b = rng.randint(5, 27)
                c = a * x_val - b
                distractors = (str(x_val), str(max(1, (c - b) // a)), str(x_val + 2), str(x_val - 1))
                prompt = (
                    f"If {a} * x - {b} = {c}, what is the integer value of x? "
                    f"Output only the integer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_linear_sub_test",
                        difficulty="medium",
                        prompt=prompt,
                        gold_answer=str(x_val),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"linear_eq": [a, -b, c]},
                    )
                )
            else:
                # Test Template T3: Three-term weighted combination in [8, 81]
                x1 = rng.randint(4, 11)
                x2 = rng.randint(3, 10)
                w1 = rng.randint(2, 4)
                w2 = rng.randint(2, 4)
                bias = rng.randint(3, 12)
                ans = w1 * x1 + w2 * x2 - bias
                distractors = (str(ans), str(w1 * x1 + w2 * x2), str((w1 + w2) * (x1 + x2) - bias), str(ans + w1))
                prompt = (
                    f"Calculate {w1}*{x1} + {w2}*{x2} - {bias}. "
                    f"Output only the integer answer.\nAnswer: "
                )
                problems.append(
                    TaskProblem(
                        problem_id=pid,
                        family="arithmetic",
                        split=split,
                        template_id="arith_weighted_comb_test",
                        difficulty="hard",
                        prompt=prompt,
                        gold_answer=str(ans),
                        candidate_answers=tuple(dict.fromkeys(distractors)),
                        metadata={"operands": [w1, x1, w2, x2, bias], "op": "w1*x1+w2*x2-bias"},
                    )
                )

    return problems
