# Gate C: fixed-prefix operator controls

Status: protocol frozen before the new model observations; implementation in progress.
This is an exploratory mechanism check, not a benchmark or a controller evaluation.

## Question and scope

Does the implemented excursion differ from linear update mixing by the stated
nonlinear response residual, and how does that direction affect a fixed-prefix
readout compared with random directions of the same **output** displacement?

1. Lock R0's existing numerical behavior and affine/nonlinear invariants with tests.
2. Keep historical R0 unchanged: it matches input noise to beta times the clipped
   visit offset, then evaluates F(A + noise). Correct misleading names/formulas.
3. Add an analysis-only output-matched random-state builder, including finite,
   scope, context, isolation and history checks. No scheduler method or dependency.
4. Execute the pinned Ouro checkpoint at fixed prefixes. Preserve an exclusive
   journal, input/model/source hashes, observed call counts and per-condition data.
5. Publish separate evidence and documentation; preserve original page styling
   and all historical results. Verify tests, exports, browser and Pages deployment.

## Frozen diagnostic design

- Same three pilot chat prefixes as gate B, identified by their exact token IDs;
  no generated continuation, answer extraction or held-out claim.
- FP32 and BF16; forced depth four, last-step readout, eager attention, no cache.
- Fork after depth one, Gaussian donor at relative norm 0.25, seed 20261031;
  advance A and B to depth two; intervene into depth three; continue to depth four.
- Full token scope, no projection or gate modulation; alpha = beta = 0.65.
- Trust radii 0.1 and 0.005. Twelve dependent conditions, three independent prefixes.
- Ordinary A, M1, E1, historical input-noise R0; random output displacements about
  ordinary A matched to E1-minus-ordinary, and about M1 matched to E1-minus-M1.
  Random seeds 42, 43, 44. Residual scale lambda in {0, 0.5, 1, 2} about M1.
- Shared reference transitions are measured and charged as diagnostic overhead;
  continuation calls are explicit. These are not independent compute-matched runs.
- Report actual states/updates by hash, pre-cast algebra error, cast/visit rounding,
  target and realized random norms, next-token ID, logit distance/KL to ordinary,
  and first-answer-token log probability. This last quantity is a diagnostic only,
  not a full answer score. Gold is read only for offline evaluation, never controls.

## Interpretation constraints

The full-scope ungated identity E1 − M1 = beta C is an exact-arithmetic statement.
Use the **effective clipped alpha**, and measure finite-precision deviations.
Affinity implies zero residual only when the actual visit lies on the full endpoint
segment. Masking, coordinate selection, gate modulation and quantization require
additional qualification. C is not a geometric curvature tensor or proof of utility.
Output matching uses an already computed reference intervention; it is an ablation,
not a cheap deployable policy. Matching before casting does not guarantee equal
realized BF16 norm. No p-values or accuracy estimate from these dependent probes.

## Reference use

XERJ searches of `tebdilcontract` recovered the existing visit/control abstractions;
that index predates gate B, so changed files were inspected directly. Searches of
`mvpouro` and `mvpaim` confirmed the pinned recurrent norm boundary and explicit
run/context/step tracking pattern. No upstream code copied. See
[reference revisions and licenses](../reports/references_and_skills.md).
