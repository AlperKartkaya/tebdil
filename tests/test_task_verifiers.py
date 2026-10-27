"""Tests for formal task generators, validators, and split template isolation."""

import pytest

from tebdil.tasks import (
    generate_arithmetic_problems,
    generate_graph_problems,
    generate_logic_problems,
    generate_task_suite,
    validate_problem_response,
)


@pytest.mark.parametrize("family", ["arithmetic", "logic", "graph"])
def test_task_generators_and_gold_self_consistency(family: str):
    for split in ("pilot", "validation", "test"):
        probs = generate_task_suite(family=family, split=split, num_problems=10, seed=42)
        assert len(probs) == 10
        for p in probs:
            outcome = validate_problem_response(p, p.gold_answer)
            assert outcome.is_valid is True, f"Gold failed validation for {p}"
            # Verify distractor candidates (beyond index 0) fail validation
            for wrong_cand in p.candidate_answers[1:]:
                wrong_out = validate_problem_response(p, wrong_cand)
                assert wrong_out.is_valid is False, f"Distractor {wrong_cand} passed for {p}"


def test_validation_and_test_splits_have_disjoint_templates_and_prompts():
    for gen_fn in (generate_arithmetic_problems, generate_logic_problems, generate_graph_problems):
        val_probs = gen_fn(split="validation", num_problems=12, seed=42)
        test_probs = gen_fn(split="test", num_problems=12, seed=42)

        val_templates = {p.template_id for p in val_probs}
        test_templates = {p.template_id for p in test_probs}
        assert val_templates.isdisjoint(test_templates)

        val_prompts = {p.prompt for p in val_probs}
        test_prompts = {p.prompt for p in test_probs}
        assert val_prompts.isdisjoint(test_prompts)
