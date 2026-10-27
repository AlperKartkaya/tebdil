"""Static and runtime verification that no gold labels leak into tebdil.core."""

import inspect
from pathlib import Path

import tebdil.core.branching as branching_mod
import tebdil.core.controls as controls_mod
import tebdil.core.excursions as excursions_mod
import tebdil.core.perturbations as perturbations_mod
import tebdil.core.policies as policies_mod
import tebdil.core.scheduler as scheduler_mod
import tebdil.core.state as state_mod


FORBIDDEN_IDENTIFIERS = {
    "gold_answer",
    "gold_label",
    "validate_problem_response",
    "validate_arithmetic",
    "validate_logic",
    "validate_graph",
    "ground_truth",
}


def test_core_modules_never_reference_gold_labels_or_validators():
    core_modules = [
        state_mod,
        perturbations_mod,
        branching_mod,
        excursions_mod,
        controls_mod,
        policies_mod,
        scheduler_mod,
    ]
    for mod in core_modules:
        src_path = Path(inspect.getfile(mod))
        source = src_path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IDENTIFIERS:
            assert forbidden not in source, (
                f"Forbidden identifier '{forbidden}' found in {src_path.name}"
            )
        for _, obj in inspect.getmembers(mod, inspect.isfunction):
            sig = inspect.signature(obj)
            for param_name in sig.parameters:
                assert "gold" not in param_name.lower()
                assert "label" not in param_name.lower()
