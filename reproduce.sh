#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/6] Running unit and invariant test suite..."
.venv/bin/pytest -q

echo "[2/6] Validating Ouro-1.4B-Thinking stepwise adapter equivalence..."
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

echo "[3/6] Running Phase 1 feasibility experiment..."
.venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml
.venv/bin/python scripts/run_phase1_grid.py

echo "[4/6] Running Phase 2 recovery & control comparison..."
.venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml

echo "[5/6] Running Phase 3 ablations & Phase 4 held-out adaptive evaluation..."
.venv/bin/python -m tebdil.cli run --config configs/phase3_ablations.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml

echo "[6/6] Generating aggregated analysis in reports/final_results.json..."
.venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/
echo "Reproduction complete."
