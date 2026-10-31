# Gate D: normalization diagnostics

Real study and local verification complete; public deployment checks pending. Gate C was closed and publicly
verified (`65cd09b`, closeout `beff3cb`) before this study began.

## Scope and evidence

[Raw report](ouro_normalization.json), [append-only journal](ouro_normalization.jsonl),
[frozen protocol](../docs/normalization_protocol.md). Same three pilot chat prefixes,
two precisions and two trust radii as C: twelve dependent conditions, not twelve
independent examples. No gold answers, generation or controller inputs are used.
The native production transition, checkpoint and historical recordings are unchanged.

Pinned Ouro-1.4B-Thinking on Apple MPS, forced four depths, full prefix/scope,
last-step readout, no cache, alpha = beta = 0.65. **246 measured recurrent calls**,
49.118 seconds in the completed run. Physical boundary hooks agree with adapter
counters. Counts include preparation, reference work, duplicated implementation
checks and all continuations; they are not per-arm compute-matched benchmark costs.
An earlier FP32 attempt was interrupted to fix static closure-capture lint, not
because of outcomes; its partial journal remains locally under `.research/diagnostics/`.
It is excluded from this completed-run count, which is not total development cost.

## Implemented diagnostic ablations

Per-token Euclidean radial/orthogonal decomposition of the **actual cast visit**;
radial-only and tangential-only visits; visit-norm and returned-output-norm matching
in Euclidean and learned-weight-divided coordinates. Each visited state receives
real recurrent computation, and all nine output variants continue through depth four.
The return formula matches the core operator, donor progression is ordinary, input
branches are unchanged and no unstable fallback occurs in all twelve conditions.

Two small analysis helpers validate shapes, finite values, nonzero directions and
safe metric weights. They are not new production normalizers or scheduler options.
Synthetic tests establish reconstruction, per-token orthogonality, norm/direction
preservation, invalid cases and native hook/call accounting before model execution.

## Native normalization is not a unit Euclidean shell

Every observed boundary matches the exact pinned formula, including epsilon = 1e-6,
FP32 RMS calculation, cast back, then learned-weight multiplication. There are zero
formula differences across all 246 observations. Absolute learned weights range
from 0.00479126 to 1.8828125.

| Observed per-token RMS over all recorded transitions | FP32 | BF16 |
|---|---:|---:|
| Before native boundary norm | 0.302–4.794 | 0.299–4.784 |
| After native boundary norm | 0.436–1.214 | 0.436–1.215 |
| After dividing out learned weights | 0.9999945–1.0000000 | 0.9960551–1.0036753 |

These ranges include preparation and every diagnostic transition. Learned-weight
coordinates are an explicitly chosen metric, not evidence of a semantic manifold.
Do not interpret these absolute RMS ratios as donor-distance contraction estimates.

## Norm-preserved outputs still change logits

| Max realized per-token relative norm error | FP32 | BF16 |
|---|---:|---:|
| L2-preserved returned output | 0.0000118% | 0.1951% |
| Weighted-norm-preserved returned output | 0.0000169% | 0.2420% |
| L2-preserved visit | 0.0000156% | 0.1933% |
| Weighted-norm-preserved visit | 0.0000169% | 0.2235% |

After L2-preserving the returned output, the final vocabulary-logit distance from
ordinary A remains **0.593–15.410 in FP32** (six conditions). Thus equal output
norm does not eliminate fixed-prefix logit sensitivity here. Weighted-norm preservation
also leaves nonzero differences (0.577–15.393). This does **not** establish a useful
direction, answer correctness or reasoning recovery. Radial-only and tangential-only
visits both change outputs; their complete contrasts remain in the report.

BF16 small-radius probes have much larger logit distances than FP32 in these cases.
Do not treat very small BF16 interventions as precision-robust geometric effects.
Rescaling changes displacement, and post-cast preservation is approximate. This
study neither removes native RMSNorm nor identifies which of the stack's many
normalization/attention operations caused historical scope differences.

The public historical scope caption is corrected: two versus one correct cases
among nine, with changed scope/perturbation settings, is not causal normalization
evidence. No historical number, chart geometry, styling or result was changed.

## Release checks

- Fresh full suite: **142 passed**, including real Ouro integration.
- Targeted Ruff E4/E7/E9/F, TypeScript, JS syntax, export and diff checks pass.
- Full local browser pass covers eight datasets, three demo cases, evidence JSON,
  keyboard/downloads, failure/retry/checksum handling, MathJax and widths
  320/390/768/1024/1440. No page errors, failed normal requests, external runtime
  requests or body overflow. Screenshots inspected; original style/CSS/IDs/order
  regression checks pass.
- Manifest is append-only relative to gate C: every historical metric, recording
  and file identity is unchanged. Seven pre-existing dirty artifacts are untouched.
- Public Pages deployment and public browser/hash verification remain pending.
Controller evaluation remains gated on demonstrated intervention utility under a
common completed-answer protocol, not on these sensitivity diagnostics.
