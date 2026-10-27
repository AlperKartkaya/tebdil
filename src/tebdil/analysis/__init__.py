"""Scientific analysis modules for TEBDIL experimental traces."""

from tebdil.analysis.accuracy_depth import analyze_step_progression
from tebdil.analysis.compute_frontier import compute_pareto_frontier
from tebdil.analysis.curvature import analyze_curvature
from tebdil.analysis.rescue import compute_rescue_matrix

__all__ = [
    "analyze_curvature",
    "analyze_step_progression",
    "compute_pareto_frontier",
    "compute_rescue_matrix",
]
