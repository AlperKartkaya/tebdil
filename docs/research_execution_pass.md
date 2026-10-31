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
| E. Controller evaluation | Unshaped and heuristic controls only after intervention utility is established | Prerequisite unmet: C/D establish sensitivity, not completed-answer utility; no new controller evaluation |
| Publication package | Frozen held-out campaign, complete costs and immutable reproducible evidence | Not started |

No later stage is considered complete because an earlier stage's tests pass.
Real-model execution is not proof of efficacy, statistical support or complete
historical provenance. Each completed stage gets its own validation record and commit.

C/D do not clear the efficacy gate. Before controller evaluation, a new held-out
intervention comparison needs one frozen completed-answer prompt/stopping/scoring
protocol, matched controls, measured costs and immutable outputs. The historical
raw two-token tables and these three reused pilot prefixes cannot supply that proof.
