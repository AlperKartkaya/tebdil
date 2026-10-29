#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:---default}"

if [[ "$MODE" == "--verify-restored" ]]; then
  echo "[1/5] Verifying SHA-256 checksums in artifacts/checksums.sha256..."
  shasum -a 256 -c artifacts/checksums.sha256

  echo "[2/5] Running non-real-model unit and invariant test suite..."
  .venv/bin/pytest -q -m "not real_model"

  echo "[3/5] Running cryptographic and artifact consistency audits..."
  .venv/bin/python -m tebdil.audit --run-id phase7_8_controller_test > /dev/null
  .venv/bin/python -m tebdil.audit --run-id phase7_8_fresh_test > /dev/null

  echo "[4/5] Re-aggregating reports/final_results.json..."
  .venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/

  echo "[5/5] Rebuilding static observatory data bundle..."
  .venv/bin/python scripts/build_landing_data.py
  echo "Restored-artifact verification complete."
  exit 0
fi

if [[ "$MODE" == "--full-compute" ]]; then
  echo "[1/7] Running unit and real-model integration test suites..."
  .venv/bin/pytest -q -m "not real_model"
  .venv/bin/pytest -q -m "real_model"

  echo "[2/7] Validating Ouro-1.4B-Thinking stepwise adapter equivalence..."
  .venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

  echo "[3/7] Running Phase 1-4 foundational experiments..."
  .venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml
  .venv/bin/python scripts/run_phase1_grid.py
  .venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml
  .venv/bin/python -m tebdil.cli run --config configs/phase3_ablations.yaml
  .venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml

  echo "[4/7] Running self-directed pipeline (counterfactuals, controllers, Phase 5/6/7/8, fresh_test, audits)..."
  .venv/bin/python scripts/run_self_directed_pipeline.py

  echo "[5/7] Generating aggregated analysis in reports/final_results.json..."
  .venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/

  echo "[6/7] Building static observatory data bundle..."
  .venv/bin/python scripts/build_landing_data.py

  echo "[7/7] Verifying SHA-256 checksums..."
  shasum -a 256 -c artifacts/checksums.sha256
  echo "Full compute reproduction complete."
  exit 0
fi

echo "[1/6] Verifying SHA-256 checksums in artifacts/checksums.sha256..."
shasum -a 256 -c artifacts/checksums.sha256

echo "[2/6] Running unit and real-model integration test suites..."
.venv/bin/pytest -q -m "not real_model"
.venv/bin/pytest -q -m "real_model"

echo "[3/6] Validating Ouro-1.4B-Thinking stepwise adapter equivalence..."
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

echo "[4/6] Executing live real-model replay audits on exploratory and fresh_test runs..."
.venv/bin/python -m tebdil.audit --run-id phase7_8_controller_test --replay-problem-id arith_test_003 --require-real-replay > /dev/null
.venv/bin/python -m tebdil.audit --run-id phase7_8_fresh_test --replay-problem-id arith_fresh_001 --require-real-replay > /dev/null

echo "[5/6] Generating aggregated analysis in reports/final_results.json..."
.venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/

echo "[6/6] Rebuilding static observatory data bundle..."
.venv/bin/python scripts/build_landing_data.py
echo "Default reproduction and live real-model replay verification complete."
