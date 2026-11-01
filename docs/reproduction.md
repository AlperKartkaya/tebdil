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

The [execution contract](ouro_execution_contract.md) defines the exact recurrence,
positions, masks and readout modes. Its [fresh machine-readable checks](../reports/ouro_execution_contract.json)
cover every depth on FP32 and BF16; these checks do not rerun the task benchmark.

Model: `ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Vendored model definitions live in [`vendor/ouro_1_4b_thinking/`](../vendor/ouro_1_4b_thinking/). Checkpoint identity is recorded in [model provenance](../site/static/evidence/model_provenance.json).

The real-model integration test and supplied experiment configurations use `local_files_only=True`. Populate the local Hugging Face cache with this exact checkpoint revision before executing them; the static export is not a substitute for model weights.

```bash
.venv/bin/python -m tebdil.cli inspect-model --model ByteDance/Ouro-1.4B-Thinking
.venv/bin/pytest -q -m real_model
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml
uv run python scripts/verify_ouro_execution.py --output /tmp/ouro-execution-new.json
```

See the dated [execution verification](../reports/ouro_execution_verification.md) and
[generation verification](../reports/trajectory_verification.md) for current checks;
the earlier [MVP record](../reports/mvp_scientific_verification.md) is historical.
Real checks use Apple MPS; CUDA was unavailable. None resolve the historical
benchmark audit failures.

Pilot generation diagnostics (not a task campaign; these can take minutes
to hours on MPS and refuse to overwrite existing evidence):

```bash
uv run pytest -q tests/test_generation_semantics.py tests/test_native_generation.py
uv run python scripts/calibrate_ouro_baseline.py --output /tmp/ouro-calibration-new.json
uv run python scripts/complete_ouro_baseline.py --output /tmp/ouro-completion-new.json
# Separate post-hoc scoring of the committed native completion artifact:
uv run python scripts/score_ouro_completion.py --output /tmp/ouro-format-diagnostic-new.json
```

The completion script uses the committed initial pilot calibration as its input,
not `/tmp/ouro-calibration-new.json`; the scoring script likewise reads the committed
native artifact, not the temporary rerun. See the [generation contract](trajectory_semantics.md)
for exact prefix/branch lifetime, chat formatting, EOS and final-answer rules.
Each finished follow-up case is saved to a sibling JSONL journal; the final JSON
is only written when all three cases finish. A journal alone is not a complete run.

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

## 7. Fixed-prefix operator controls (gate C)

Uses the committed gate B pilot prefixes, pinned cached checkpoint and tokenizer.
The [frozen protocol](operator_controls_protocol.md) defines the complete paired
design. This does not run free generation or train/evaluate a controller.

```bash
uv run python scripts/verify_operator_controls.py --output /tmp/ouro-controls-new.json
```

The output and adjacent `.jsonl` must not exist. The script hashes the checkpoint
and source before execution, journals every condition, counts real boundary-norm
calls independently, and rejects source changes during the run. The recorded MPS
run took about one minute; other hardware can differ. FP32/BF16 results are device-
dependent diagnostics, not bitwise cross-device guarantees. Both reference
construction and continuation costs are reported; do not treat shared-work probes
as independently executed compute-matched benchmarks. Historical results are untouched.

## 8. Native normalization and norm-preserving diagnostics (gate D)

```bash
uv run python scripts/verify_normalization.py --output /tmp/ouro-normalization-new.json
```

Requires the same pinned cached checkpoint and committed pilot prefixes. Both the
output and adjacent journal must be absent. The [protocol](normalization_protocol.md)
was frozen before observations. Native boundary input/output RMS and exact formula
parity are recorded for each call; all reference and continuation costs reconcile.
The completed Apple MPS run took about 49 seconds. The output is a sensitivity
diagnostic, not a generation benchmark. Native normalization and historical results
are unchanged; BF16 rescaling is only approximately norm-preserving after casting.

## 9. Shared completed-answer evaluation (gate 1)

```bash
uv run pytest -q tests/test_completed_answer_protocol.py
uv run python scripts/verify_completed_answer.py --output /tmp/completed-answer-new.json
```

The [frozen v1 protocol](completed_answer_protocol.md) uses the existing native
cache-free generator and final-integer parser. It executes only the preselected
first development arithmetic question, not the reserved test set. A 2048-token
ceiling is not a guaranteed completion or a compute-matched intervention budget.
Both output and sibling journal must be new. All checkpoint/tokenizer/config and
source hashes are checked, and the physical call counters must agree. RSS/MPS
memory values are after-run snapshots, not peak measurements. See the
[verification record](../reports/completed_answer_verification.md) before making
any claim about completion or deployment. This does not rerun historical phases.

## 10. Fixed-method completed-answer comparison (gate 2)

```bash
uv run pytest -q tests/test_fixed_completion_comparison.py
uv run python scripts/compare_completed_answers.py --output-dir /tmp/comparison-new
```

Uses only completion_dev_000 and the committed native S0 record; first verifies
an eight-token adapter/native prefix. All five new methods run serially to EOS
or 2048 tokens. This can take hours on the local MPS machine; do not mistake a
partial journal for a completed campaign. The directory must not exist. Each
method has an exclusive output and every token is journaled. An interruption
leaves partial evidence; there is intentionally no silent restart/overwrite.
No reserved test inference or controller evaluation. Check the
[verification status](../reports/completed_answer_comparison_verification.md)
before claiming any result. See the [frozen scope and cost caveats](completed_answer_comparison_protocol.md).

## 11. Measurement v2 (offline only)

```bash
uv run pytest -q tests/test_completion_v2.py
uv run python scripts/verify_completion_v2.py --output /tmp/completion-v2-new.json
```

Requires no checkpoint or inference server. Reads only the two committed development
smoke reports, preserves all originals and refuses an existing output. The new
report contains original v1 fields and separate post-hoc v2 diagnostics, not new
model outcomes. [Contract](completed_answer_v2_protocol.md) and
[verification](../reports/completion_v2_verification.md). Future pilot generation
must explicitly apply v2 after validating its actual token/call ledger.

## 12. Development baseline difficulty pilot

```bash
uv run pytest -q tests/test_baseline_difficulty.py
uv run python scripts/run_baseline_difficulty.py --output-dir /tmp/baseline-difficulty-new
```

Requires the exact cached checkpoint and the frozen MPS/software contract from
native S0. Reuses question 000's prior output; runs questions 001–011 serially.
No intervention or reserved-test inference. A new output directory is mandatory;
there is no silent resume/overwrite. Independent forward proposals are validated
against actual returned tokens before a question is marked finished. Partial
journals are diagnostic progress, not completed outcomes. This may take hours.
[Protocol](baseline_difficulty_protocol.md) and
[current verification](../reports/baseline_difficulty_verification.md).

The committed [raw pilot](../reports/ouro_baseline_difficulty.json) and
[append-only journal](../reports/ouro_baseline_difficulty.jsonl) retain all twelve
attempts (eleven new, one reused). Audit without running the checkpoint:

```bash
uv run pytest -q tests/test_site_release.py -k baseline_difficulty_retains
```

This reconstructs every token-prefix hash and reconciles independent forward
proposals, physical calls, per-template summaries and frozen source hashes.
It audits recorded evidence, not a clean-clone or cross-device generation rerun.
The outcome is 10 recognized correct, 0 recognized incorrect, 2 invalid format,
0 budget exhausted. Do not repair these scores or infer arithmetic accuracy from
format failures. No intervention or reserved-test execution is implied.

## 13. Prompt-started thinking-mode recording

```bash
uv run pytest -q tests/test_prompt_started_reasoning.py
uv run python scripts/verify_prompt_started_reasoning.py --output /tmp/thinking-smoke-new.json
```

Requires the exact cached checkpoint and frozen BF16 MPS/software contract. Uses
only already-observed development question 000, not the new reasoning task pool.
The new encoder uses True; old runners still use False. Existing JSON or sibling
journal paths are refused. [Protocol](prompt_started_reasoning_protocol.md) and
[verification](../reports/prompt_started_reasoning_verification.md). An unfinished
or malformed outcome is retained, not silently retried or scored from its thoughts.

## 14. Frozen reasoning task/verifier pool (no inference)

```bash
uv run python scripts/freeze_reasoning_tasks.py --check
uv run pytest -q tests/test_reasoning_tasks.py
```

[Protocol and exact grammar](reasoning_tasks_protocol.md). Six development and
twelve reserved instances cover CNF satisfaction and stateful key/door planning.
Exhaustive Boolean verification and state-space BFS accept alternate valid answers;
no model is run by these commands. The pool is instance-held-out, not template-OOD.
The old 12/60 arithmetic split and its recorded scores are untouched. The public
prompt download is development only, explicitly not a recorded model experiment.

## 15. Passive reasoning recorder

```bash
uv run pytest -q tests/test_passive_reasoning.py
uv run python scripts/record_passive_reasoning.py --output-dir /tmp/passive-reasoning-new
```

[Pre-observation protocol](passive_reasoning_protocol.md). The first level-0
CNF/planning development pair is fixed before observations; no reserved tasks or
interventions. Reuses native True-mode generation, records actual aligned depth
states and raw CoT, then applies the frozen task verifier offline. The exact-repeat
screen is heuristic, not a validated semantic-stall detector. Existing/partial
output directories are refused. MPS execution can take hours. Preserve all files,
including failures; a start/partial journal is not a completed experiment.
