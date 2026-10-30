# Reproduction guide

## 1. Inspect the committed static export

This path works without a model checkpoint or local raw `results/` directory:

```bash
python3 scripts/build_landing_data.py --check
python3 -m http.server 8080 --directory site
```

Open `http://localhost:8080`. Relative assets also support the GitHub Pages `/tebdil/` base path. Public controls replay recorded artifacts; they do not execute inference. The [Pages workflow](../.github/workflows/pages.yml) verifies and packages `site/`.

`--check` verifies file hashes, record pairing, counts and disclosure of ledger gaps. It does not assert that historical ledgers all match or that model execution was repeated.

## 2. Install the locked research environment

```bash
uv sync --locked
.venv/bin/pytest -q -m "not real_model"
```

The project supports Python 3.11–3.13; the local verification environment uses Python 3.12. A clean clone includes the static JSON export and selected provenance evidence, but **not the ignored raw `results/` directory or checkpoint weights**.

## 3. Verify real Ouro execution

Model: `ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Vendored model definitions live in [`vendor/ouro_1_4b_thinking/`](../vendor/ouro_1_4b_thinking/). Checkpoint identity is recorded in [model provenance](../site/static/evidence/model_provenance.json).

The real-model integration test and supplied experiment configurations use `local_files_only=True`. Populate the local Hugging Face cache with this exact checkpoint revision before executing them; the static export is not a substitute for model weights.

```bash
.venv/bin/python -m tebdil.cli inspect-model --model ByteDance/Ouro-1.4B-Thinking
.venv/bin/pytest -q -m real_model
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml
```

The latest local full suite passed 68 tests, including real Ouro integration. Selected real replays ran on Apple MPS; CUDA was unavailable. The [verification record](../reports/mvp_scientific_verification.md) separates those checks from unresolved historical audit failures.

## 4. Regenerate experiments deliberately

These commands perform real model computation and write local experiment outputs. Preserve any existing recordings before rerunning the same run IDs.

```bash
.venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml
.venv/bin/python scripts/run_phase1_grid.py
.venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase3_ablations.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml
# Trains controllers and executes the remaining research phases:
.venv/bin/python scripts/run_self_directed_pipeline.py
.venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/
```

A complete campaign was not rerun during the MVP refactor. Historical accuracy values must not be labeled fresh results from the refactored code.

## 5. Audit recordings and rebuild the export

With local raw results and research artifacts available:

```bash
.venv/bin/python -m tebdil.audit --run-id phase7_8_fresh_test
.venv/bin/python -m tebdil.audit --run-id phase7_8_fresh_test \
  --replay-problem-id arith_fresh_test_001 --require-real-replay
.venv/bin/python scripts/build_landing_data.py
.venv/bin/python scripts/build_landing_data.py --check
```

**Historical audit failures are expected and must remain visible.** The fresh evaluation has 71 parallel-ledger mismatches; the exploratory controller run has 81. Earlier runs have missing token ledgers. Real replay passing does not override those failures. `reproduce.sh` remains a research pipeline, not a clean-clone demo launcher; its fail-fast historical audits may stop it. Do not treat an old `all_passed` field as current verification.

Building without `--check` requires all eight raw run directories plus controller/counterfactual/provenance inputs. The committed static export can be checked independently, but does not reconstruct every omitted research artifact automatically.

## 6. Optional local Streamlit inspection

```bash
.venv/bin/streamlit run ui/app.py
```

This advanced tool expects local raw results. It is separate from the static public observatory; no public live inference service is configured.
