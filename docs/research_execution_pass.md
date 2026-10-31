# Sequential research pass

The public page's original design and historical recordings stay unchanged. New
verification is separate from historical task-performance evidence. Existing
uncommitted controller, counterfactual and report artifacts are not inputs to this pass.

| Order | Gate before moving on | Status |
|---|---|---|
| A. Ouro execution contract | Per-depth real-checkpoint state/gate/logit parity; masks/positions; both readouts; FP32/BF16 diagnostic; documentation and Pages evidence | Complete: `e0728b3`; [CI/deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41219360287) and public browser checks passed |
| B. Trajectory semantics | Explicit token/depth and prefix contracts; baseline-generation calibration | In progress |
| C. Operator controls | Correctly named input noise; output-matched controls; algebraic invariants; fixed-prefix residual study | Not started |
| D. Normalization diagnostics | Explicit radial/directional and norm-preserving ablations | Not started |
| E. Controller evaluation | Unshaped and heuristic controls only after intervention utility is established | Not started |
| Publication package | Frozen held-out campaign, complete costs and immutable reproducible evidence | Not started |

No later stage is considered complete because an earlier stage's tests pass.
Real-model execution is not proof of efficacy, statistical support or complete
historical provenance. Each completed stage gets its own validation record and commit.
