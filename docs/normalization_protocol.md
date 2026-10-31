# Gate D: normalization diagnostic protocol

Frozen diagnostic design before gate D model observations. Current execution status
belongs in [verification](../reports/normalization_verification.md), not this snapshot.
No production recurrence, normalization, historical score or controller is changed.

## Question

Do observed fixed-prefix effects survive removing visit/output norm changes, and
what does the native recurrent-boundary RMSNorm actually do? Neither a change in
residual norm nor a two-dimensional plot identifies semantic reasoning geometry.

## Design

Reuse gate C's three pilot chat prefixes, branch seed 20261031, Gaussian donor
relative norm 0.25, fork after depth one and intervention at depth two. FP32/BF16,
forced depth four, full scope, alpha = beta = 0.65, radii 0.1/0.005, no projection,
no gate modulation, eager attention, no cache, last-step readout. Twelve dependent
conditions; no new held-out questions or generated answers.

For the **actual cast** visit delta, decompose per token into its Euclidean radial
component along A and orthogonal remainder. Evaluate genuine F at the original,
radial-only, tangential-only and two norm-preserved visited states. The two norm
metrics are Euclidean L2 and L2 after division by the native boundary's learned
RMSNorm weight. Match visit norm to A, or match returned output norm to ordinary
F(A), separately. The weighted coordinate metric is a diagnostic choice, not a
claim that hidden states occupy a semantic manifold. Fail if the native weights
are nonfinite or too close to zero for safe division.

Compare ordinary A, M1, E1, radial visit, tangential visit, visit-norm-preserved L2,
visit-norm-preserved weighted L2, output-norm-preserved L2, and output-norm-preserved
weighted L2. Continue all nine output states through the final real recurrent step.
Verify the manual E1 formula against the existing core implementation. Preserve
state hashes, realized norm errors after casting, radial/tangent decomposition
error, cosine/distance to ordinary output and next-token logit sensitivity. No
answer likelihood or gold-answer input is needed in this study.

Hook the **actual native boundary** before/after every transition. Record per-token
input/output RMS, output RMS after division by learned weight, epsilon and exact
formula parity including cast order. Native RMSNorm scales by learned coordinate
weights; it does not impose equal Euclidean norm. Reconcile physical hook counts
with adapter counters, including preparation and duplicated implementation checks.

## Interpretation limits

These are paired diagnostic ablations, not equal-displacement or independent
compute-matched benchmarks. Norm rescaling may move a visit outside the original
trust radius; record the realized displacement, do not silently reclip and destroy
norm preservation. Cast-induced norm mismatch remains visible. Radial/tangential
parts depend on the chosen per-token metric. Do not infer causal reasoning benefit,
accuracy, significance, optimal normalization or a trained controller from them.
Archive exclusive raw reports and journals with checkpoint/source/input hashes.

## Reference use

XERJ searches recovered upstream `OuroRMSNorm.forward` and existing curvature
analysis. Direct inspection of pinned `modeling_ouro.py:379–384` confirms FP32 RMS,
epsilon, cast back **before** learned-weight multiplication. The older source index
predates gates B/C; new source is inspected directly. No upstream code is copied
into production and no extra runtime dependency is installed.
