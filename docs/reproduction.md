# Reproduction Guide

## 1. Environment Setup

TEBDIL uses Python 3.12 and `uv` with a locked `uv.lock` dependency graph:

```bash
uv sync
```

## 2. Pinned Checkpoint Verification

The primary model is `ByteDance/Ouro-1.4B-Thinking` pinned at revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Audited model definitions (`configuration_ouro.py` and `modeling_ouro.py`) are vendored in `vendor/ouro_1_4b_thinking/`.

Inspect hardware and checkpoint configuration:
```bash
.venv/bin/python -m tebdil.cli inspect-model --model ByteDance/Ouro-1.4B-Thinking
```

Verify exact numerical equivalence (`max_abs_diff == 0.0`) between standard `OuroForCausalLM.forward` and stepwise `OuroRecurrentAdapter`:
```bash
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml
```

## 3. Running Unit and Invariant Tests

```bash
.venv/bin/pytest -q
```

## 4. Reproducing Experimental Phases

Run the complete experimental pipeline via `reproduce.sh` or individual CLI commands:

```bash
# Phase 1: Feasibility & smoke pilot
.venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml

# Phase 1C: Alpha/Beta parameter grid & identity verification
.venv/bin/python scripts/run_phase1_grid.py

# Phase 2: Trajectory recovery & degradation benchmark (validation split)
.venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml

# Phase 3: Mechanism & curvature ablations (validation split)
.venv/bin/python -m tebdil.cli run --config configs/phase3_ablations.yaml

# Phase 4 & 5: Label-free adaptive excursion on held-out test split
.venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml

# Aggregate paired statistics & Pareto frontier into reports/final_results.json
.venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/
```

## 5. Launching the TEBDIL Observatory UI

```bash
.venv/bin/streamlit run ui/app.py
```
