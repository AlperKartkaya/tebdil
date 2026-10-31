# TEBDİL — Self-Directed Latent Relocation for Recurrent Reasoning

**Visit another latent region. Compute there. Bring the update back.**

TEBDİL investigates whether a recurrent language model can recover from an unproductive reasoning trajectory through temporary latent excursions. It uses the real, open-weight `ByteDance/Ouro-1.4B-Thinking` model with shared weights and instrumented hidden-state access.

[Open the research observatory](https://alperkartkaya.github.io/tebdil/) · [Findings](docs/findings.md) · [Reproduce](docs/reproduction.md)

## What changes inside the model?

A receiving trajectory A temporarily visits a bounded neighborhood toward donor B. The model computes its genuine recurrent update at that visit state, then transfers the update to A. A continues; it is **not replaced by B**.

$$V_t^{\mathrm{visit}} = F_{\theta,t}(H_t^{\mathrm{visit}})-H_t^{\mathrm{visit}},\qquad H_{t+1}^{A'}=H_t^A+(1-\beta)V_t^A+\beta V_t^{\mathrm{visit}}$$

This is the full-scope, unprojected return formula. Scoped and coordinate-projected variants restrict the returned correction; see [methods](docs/methods.md) and the [implementation](src/tebdil/core/excursions.py).

- **Mechanism:** real Ouro recurrence, independent branch states, bounded visit and return, no-op and isolation tests.
- **Decision:** E1 uses a fixed schedule; E2 uses heuristic rules. E3 uses a trained action scorer, with hand-written feasibility gates in gated variants. These are not interchangeable claims of autonomy.
- **Evaluation:** paired arithmetic, logic and graph tasks; baseline recurrence, parallel trajectories, state/update mixing and random controls.
- **Public experience:** recorded experiments and explicitly illustrative diagrams. GitHub Pages does not run a GPU inference backend.

## What does the evidence show?

The static export contains **2,866 trace records across eight runs**, not 2,866 independent problems. Each controller/fresh evaluation uses **18 problems × 3 seeds = 54 method-instance pairs**. Repeated seeds are not independent problem samples.

On fresh templates, S0 scores **27.78%**, fixed excursions E1 **33.33%**, gated E3_mlp **27.78%**, and ungated E3_mlp **31.48%**. Ungated E3 rescues **2/39** failed baseline instances with **0/15** degradations. Its problem-cluster 95% difference interval is **[0.00, +9.26] percentage points**, with problem-level exact sign test **p = 0.5**. These small exploratory results do not establish generalization or superiority over mixing baselines.

Compute is reported, but comparisons are not uniformly compute-matched. Historical parallel records have ledger mismatches; several earlier runs lack token ledgers. The current audit **fails closed**, rather than reusing historical `all_passed` flags. [Full findings and limitations](docs/findings.md).

**Protocol limit:** these historical results use raw prompts and a two-token
short-answer budget, not completed chat-format thinking. Branches are re-forked
within each generated token's recurrent computation; they are not persistent
reasoning agents. The [generation contract](docs/trajectory_semantics.md) separates
these semantics from a fresh pilot prompt/budget calibration. Calibration does not
replace the historical results or establish intervention efficacy. The native
follow-up now preserves three EOS-completed pilot continuations and real call
ledgers; its separate post-hoc scoring distinguishes answer-format failures from
mathematical errors. See [point B verification](reports/trajectory_verification.md).

A separate [fixed-prefix operator diagnostic](reports/operator_controls_verification.md)
now checks E1 against update mixing and output-matched random directions in FP32/BF16.
Its 306 real recurrent calls verify the bounded mechanism, not a reasoning gain.
Historical R0 is input-offset-matched noise, not an output-matched control.

The [normalization diagnostic](reports/normalization_verification.md) adds radial,
tangential and norm-preserved controls with 246 real boundary observations. Equal
norms do not eliminate logit sensitivity; no completed-answer benefit follows.

The [completed-answer gate](docs/completed_answer_protocol.md) freezes one
prompt/stopping/parser contract and separate development/reserved test questions.
Its [verification record](reports/completed_answer_verification.md) distinguishes
pipeline smoke validation from a future intervention-accuracy comparison.
Gate 1 is complete. The [fixed-method comparison](docs/completed_answer_comparison_protocol.md)
now records five real EOS completions on the same already-correct development
question: mixing and excursion methods retain the recognized correct answer;
donor-only uses a boxed answer rejected by the frozen parser, not demonstrated
wrong arithmetic. No rescue or held-out benefit is established.
[Gate-2 findings and actual costs](reports/completed_answer_comparison_verification.md).

## Run the static observatory

No checkpoint, backend, or raw `results/` directory is needed to serve the committed export:

```bash
python3 scripts/build_landing_data.py --check
python3 -m http.server 8080 --directory site
# Open http://localhost:8080
```

The export check validates checksums, record pairing and disclosed ledger gaps. It does not rerun model inference.

For research development:

```bash
uv sync --locked
.venv/bin/pytest -q -m "not real_model"
# Requires the pinned checkpoint already cached locally:
.venv/bin/pytest -q -m real_model
```

The [Ouro execution contract](docs/ouro_execution_contract.md) now includes fresh
per-depth FP32/BF16 real-checkpoint checks. The earlier [MVP scientific verification](reports/mvp_scientific_verification.md)
documents sampled replays and historical-audit failures. Neither proves efficacy,
CUDA behavior or a full campaign rerun.

Rebuilding the export with `python scripts/build_landing_data.py` requires local raw experiment inputs. Those `results/` files are ignored by Git and are not supplied by a clean clone. See the [reproduction guide](docs/reproduction.md) before running research scripts.

## Research layers

- [Architecture](docs/architecture.md), [methods](docs/methods.md), [methodology](docs/methodology.md), [limitations](docs/limitations.md)
- [Current findings](docs/findings.md), [final report](reports/final_report.md), [aggregated historical results](reports/final_results.json)
- [Model provenance](site/static/evidence/model_provenance.json), [controller checkpoint](site/static/evidence/tebdil_mlp_v1.json), [split manifest](site/static/evidence/split_manifest.json)
- [Release verification](reports/mvp_release_verification.md)
- [References and installed skills](reports/references_and_skills.md), [design contract](DESIGN.md)
- Local advanced inspection: `.venv/bin/streamlit run ui/app.py` (requires local raw results; not the public Pages application).

Licensed under [Apache-2.0](LICENSE). Ouro checkpoint terms are published by ByteDance under Apache-2.0.

Parallel selection caveat: all historical P0/P1 total-selection costs are unverified, including rows whose selected-output ledger sums happen to match. The 71/81 counts describe arithmetic mismatches, not the full scope of uncertain parallel execution.
