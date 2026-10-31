# TEBDİL: recorded findings and evidence limits

Updated 2026-10-31. Values below come from the current [static manifest](../site/static/tebdil_data.js), recomputed from exported raw records by `scripts/build_landing_data.py`. No model outputs were changed for presentation.

## Scope

These are **historical raw-prompt, two-token short-answer experiments**, not
completed chat-format Thinking evaluations. Branches re-fork within each generated
token's recurrent unroll; no hidden branch memory crosses output positions.
See the [generation contract](trajectory_semantics.md) for new prefix/termination
metadata and the separate pilot calibration. Changing prompt/budget settings for
S0 alone would not produce a fair intervention comparison: every method needs the
same frozen completion protocol in a new campaign.

There are **2,866 trace records across eight runs**. A trace is one method/problem/seed result, not an independent problem. Both controller evaluations contain **18 distinct problems × 3 seeds = 54 paired instances per method**. The label `fresh_test` identifies a separate generated split; it does not establish preregistration, distributional generalization, or a confirmatory study.

## Separate baseline protocol diagnostic

A new three-problem pilot is separate from every table below. Raw/2 and raw/64
each produced one correct answer; chat/128 always truncated thinking. Longer
native cache-free continuations reached EOS in 437/483/1,514 tokens, with
1,748/1,932/6,056 measured recurrent calls. All three match the adapter's first
128 tokens. The original integer-only format rejects their explanatory prose.
A separately labeled post-hoc final parser recognizes arithmetic and logic but
not the graph conclusion, despite its mathematically correct answer of 15.

This exposes prompt, completion and extraction confounds, not a TEBDİL benefit.
Neither the parser nor these pilot outcomes provide a held-out accuracy result.
See the [generation contract](trajectory_semantics.md),
[native raw evidence](../reports/ouro_baseline_completion.json), and
[post-hoc annotation](../reports/ouro_completion_scoring.json).

## Fresh-template evaluation

Source: [records](../site/static/data/phase7_8_fresh_test.json), [configuration](../site/static/evidence/phase7_8_fresh_test.yaml).

| Method | Accuracy | Difference vs S0 | Problem-cluster 95% CI | Rescued | Degraded | Mean calls | Ledger gaps |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `S0` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 7.33 | 0 |
| `S1` | 11.11% | -16.67 pp | [-33.33, +0.00] pp | 0/39 | 9/15 | 12.83 | 0 |
| `EXIT_2` | 22.22% | -5.56 pp | [-16.67, +0.00] pp | 0/39 | 3/15 | 5.50 | 0 |
| `P0` | 20.37% | -7.41 pp | [-16.67, +0.00] pp | 0/39 | 4/15 | 17.96 | 21 |
| `P1` | 35.19% | +7.41 pp | [+0.00, +18.52] pp | 4/39 | 0/15 | 17.96 | 50 |
| `M0` | 29.63% | +1.85 pp | [+0.00, +5.56] pp | 1/39 | 0/15 | 12.83 | 0 |
| `M1` | 35.19% | +7.41 pp | [+0.00, +18.52] pp | 4/39 | 0/15 | 12.83 | 0 |
| `RAND_CTRL` | 35.19% | +7.41 pp | [+0.00, +18.52] pp | 4/39 | 0/15 | 17.83 | 0 |
| `E1` | 33.33% | +5.56 pp | [-3.70, +16.67] pp | 4/39 | 1/15 | 14.67 | 0 |
| `E2` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 13.83 | 0 |
| `G_gate_only` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 15.33 | 0 |
| `E3_t2_1donor` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 12.02 | 0 |
| `E3_mlp` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 16.83 | 0 |
| `E3_mlp_ungated` | 31.48% | +3.70 pp | [+0.00, +9.26] pp | 2/39 | 0/15 | 17.24 | 0 |
| `E3_mlp_unshaped` | 27.78% | +0.00 pp | [+0.00, +0.00] pp | 0/39 | 0/15 | 15.63 | 0 |

## Exploratory controller evaluation

Source: [records](../site/static/data/phase7_8_controller_test.json), [configuration](../site/static/evidence/phase7_8_controller_test.yaml).

| Method | Accuracy | Difference vs S0 | Problem-cluster 95% CI | Rescued | Degraded | Mean calls | Ledger gaps |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `S0` | 22.22% | +0.00 pp | [+0.00, +0.00] pp | 0/42 | 0/12 | 7.33 | 0 |
| `S1` | 22.22% | +0.00 pp | [+0.00, +0.00] pp | 0/42 | 0/12 | 12.83 | 0 |
| `EXIT_2` | 27.78% | +5.56 pp | [-11.11, +27.78] pp | 6/42 | 3/12 | 5.50 | 0 |
| `P0` | 22.22% | +0.00 pp | [+0.00, +0.00] pp | 0/42 | 0/12 | 18.33 | 30 |
| `P1` | 27.78% | +5.56 pp | [+0.00, +14.81] pp | 3/42 | 0/12 | 18.33 | 51 |
| `M0` | 31.48% | +9.26 pp | [+0.00, +20.37] pp | 5/42 | 0/12 | 12.83 | 0 |
| `M1` | 25.93% | +3.70 pp | [+0.00, +9.26] pp | 2/42 | 0/12 | 12.83 | 0 |
| `RAND_CTRL` | 29.63% | +7.41 pp | [+0.00, +16.67] pp | 4/42 | 0/12 | 17.83 | 0 |
| `E1` | 27.78% | +5.56 pp | [+0.00, +14.81] pp | 3/42 | 0/12 | 14.67 | 0 |
| `E2` | 27.78% | +5.56 pp | [+0.00, +14.81] pp | 3/42 | 0/12 | 13.78 | 0 |
| `G_gate_only` | 27.78% | +5.56 pp | [+0.00, +14.81] pp | 3/42 | 0/12 | 14.61 | 0 |
| `E3_t2_1donor` | 27.78% | +5.56 pp | [+0.00, +14.81] pp | 3/42 | 0/12 | 11.06 | 0 |
| `E3_mlp` | 31.48% | +9.26 pp | [+0.00, +22.22] pp | 5/42 | 0/12 | 15.94 | 0 |
| `E3_mlp_ungated` | 31.48% | +9.26 pp | [+0.00, +22.22] pp | 5/42 | 0/12 | 15.41 | 0 |
| `E3_mlp_unshaped` | 22.22% | +0.00 pp | [+0.00, +0.00] pp | 0/42 | 0/12 | 16.89 | 0 |

## Interpretation

- The real excursion-and-return operator is implemented and tested. Its implementation does not establish a general reasoning benefit.
- On fresh templates, gated E3_mlp matches S0 accuracy; ungated E3_mlp improves the observed average by 3.70 percentage points. The latter interval includes zero and problem-level exact sign test p = 0.5.
- E1 records both rescues and a degradation on fresh templates. Zero observed degradations for another method is not a safety guarantee.
- Direct update mixing M1 and random control score 35.19% on fresh templates, above both E3 variants. TEBDİL does not claim superiority to these controls.
- S1 loses 16.67 percentage points on fresh templates. Its problem-cluster interval is [-33.33, 0.00] points and problem-level exact sign test p = 0.25. The earlier instance-level p-value treated repeated seeds differently and must not support an independent-problem significance claim.
- Curvature and reward-shaping ablations are local exploratory observations. They do not prove the mechanism is universally necessary or isolate every possible explanation.
- P1 is an answer-informed oracle upper bound, not an executable label-free deployment policy. E3 is a learned action scorer with heuristic gates in gated variants; E2 is heuristic, not learned.

## Compute and provenance limitations

Mean calls are historical recorded aggregates, not wall-clock speed or universally matched budgets. A nonzero ledger-gap count means the displayed total cannot be reconciled with the available per-token ledger; do not use those rows for validated efficiency claims. There are **71** such records in fresh evaluation and **81** in exploratory controller evaluation, confined to historical P0/P1 selection records. Initial feasibility, recovery, curvature and adaptive runs also lack token ledgers for their recorded rows.

Current audit code reports these gaps as failures. Older exported audit JSON is retained as historical evidence, not a current all-checks-passed certificate. Future parallel records preserve selected-output provenance and an explicitly idealized reference parallel budget, not verified total execution cost; this does not retroactively repair old evidence.

The pinned checkpoint and seven selected replay cases were checked on Apple MPS. See [MVP scientific verification](../reports/mvp_scientific_verification.md). No full campaign was rerun. Curvature telemetry measures the full pre-projection residual, not necessarily the applied correction norm.

## Earlier phases and deeper evidence

The observatory retains all eight experiments, including donor selection and selective-transfer ablations. Their full method statistics and original records remain in the [manifest](../site/static/tebdil_data.js) and [data directory](../site/static/data/). Older prose reports describe historical snapshots and may use different record counts or inference units; the current manifest and explicit source records govern this summary.

See [reproduction](reproduction.md), [methods](methods.md), [limitations](limitations.md), and [final report](../reports/final_report.md).

Parallel selection caveat: all historical P0/P1 total-selection costs are unverified, including rows whose selected-output ledger sums happen to match. The 71/81 counts describe arithmetic mismatches, not the full scope of uncertain parallel execution.

## Fixed-prefix mechanism follow-up (gate C)

Twelve dependent conditions on three reused pilot prefixes verify the return formula,
branch isolation and actual checkpoint execution (306 recurrent calls). FP32 E1 − M1
matches beta C within 0.000016 Frobenius norm; BF16 output casting increases the
identity error to 0.597–0.900. Random-output matching reports realized norm error,
not an assumption of exact BF16 equality. Residual magnitude/direction has no
consistent first-answer-token likelihood advantage over matched random controls.
This surrogate is not completed-answer accuracy or evidence of a reasoning gain.
See [raw observations and limits](../reports/operator_controls_verification.md).

## Normalization follow-up (gate D)

All 246 observed native boundary calls match the pinned formula, including cast order.
Learned weights produce per-token Euclidean RMS of 0.436–1.215 across these probes,
not a universal unit shell. In FP32, preserving returned output L2 norms to within
0.0000118% still leaves final logit distances of 0.593–15.410 from ordinary A. Norm
alone therefore does not eliminate sensitivity in these fixed-prefix cases. This
is not completed-answer utility, a semantic-manifold result or a causal explanation
of historical token-scope outcomes. BF16 preservation errors reach 0.2420% under
the weighted metric; small-radius effects are not precision-robust by default.
[Raw observations, full ablations and limitations](../reports/normalization_verification.md).
