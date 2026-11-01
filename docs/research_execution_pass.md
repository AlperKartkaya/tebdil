# Sequential research pass

The public page's original design and historical recordings stay unchanged. New
verification is separate from historical task-performance evidence. Existing
uncommitted controller, counterfactual and report artifacts are not inputs to this pass.

| Order | Gate before moving on | Status |
|---|---|---|
| A. Ouro execution contract | Per-depth real-checkpoint state/gate/logit parity; masks/positions; both readouts; FP32/BF16 diagnostic; documentation and Pages evidence | Complete: `e0728b3`; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41219360287) and public browser checks passed |
| B. Trajectory semantics | Explicit token/depth and prefix contracts; baseline-generation calibration | Complete: `22cc752`, 122 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41239947652) and full public browser checks passed |
| C. Operator controls | Correctly named input noise; output-matched controls; algebraic invariants; fixed-prefix residual study | Complete: `65cd09b`, 135 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41241237400), public hashes and full browser checks passed |
| D. Normalization diagnostics | Explicit radial/directional and norm-preserving ablations | Complete: `43e807d`, 142 tests including real Ouro; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41242354541), public hashes and full browser checks passed |
| Completed-answer gate 1 | Frozen prompt/stopping/scoring; real baseline smoke; docs and Pages | Complete: `54b3381`, `d74a61f`; 167 tests, real EOS completion, [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41245538443), public browser/hash checks passed |
| Completed-answer gate 2 | Fixed intervention/donor definitions, appropriate controls and measured completed-answer costs | Technical smoke complete: `c60dbaf`, `0dcbdc3`; five EOS completions, 14,402 new calls; 178 tests, [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41254056115), public browser/hash checks passed; boxed-format limit retained, no efficacy claim |
| Measurement v2 prerequisite | Versioned format repair, immutable v1 scores, offline diagnostic and Pages verification | Code/protocol frozen at `b315d22`; 232 tests and local browser checks passed; public verification pending; no new model completions |
| Completed-answer gate 3 | Development timing/utility pilot, then frozen independent test with uncertainty | Not started; all 60 reserved test questions remain unexecuted |
| E. Controller evaluation | Unshaped and heuristic controls only after intervention utility is established | Prerequisite unmet: C/D establish sensitivity, not completed-answer utility; no new controller evaluation |
| Publication package | Frozen held-out campaign, complete costs and immutable reproducible evidence | Not started |

No later stage is considered complete because an earlier stage's tests pass.
Real-model execution is not proof of efficacy, statistical support or complete
historical provenance. Each completed stage gets its own validation record and commit.

C/D do not clear the efficacy gate. Before controller evaluation, a new held-out
intervention comparison needs one frozen completed-answer prompt/stopping/scoring
protocol, matched controls, measured costs and immutable outputs. The historical
raw two-token tables and these three reused pilot prefixes cannot supply that proof.
