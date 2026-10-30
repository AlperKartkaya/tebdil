# TEBDIL Final Scientific Report

**Title:** Self-Directed Cooperative Latent Excursions for Recurrent Reasoning in Looped Language Models
**Date:** 2026-10-30
**Model Backbone:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`)
**Recorded scope:** 2,866 method/problem/seed trace records across eight experiment runs; not 2,866 independent problems. Each controller evaluation has 18 problems × 3 seeds.

---

## 1. Problem Statement and Research Question

Looped Language Models (LoopLMs) such as `ByteDance/Ouro-1.4B-Thinking` refine token representations by applying a shared 24-layer Transformer block $F_{\theta, t}$ across $T=4$ recurrent steps (`total_ut_steps=4`). Standard inference advances a single hidden-state trajectory $H_t \in \mathbb{R}^{B \times L \times d}$ deterministically.

**Core Research Question:**
> Can a recurrent language model learn **when**, **where**, and **how** to relocate its intermediate latent computation during inference—choosing among continuing normally (`CONTINUE`), executing a cross-trajectory excursion (`EXCURSION(d, \alpha, \beta)`), or halting early (`EXIT`)—to recover from unproductive reasoning trajectories while improving performance under controlled compute budgets?

---

## 2. Novelty Scope and Literature Positioning

The [literature matrix](literature_matrix.md) records the project’s prior-work positioning, not an independently established novelty claim. TEBDIL specifically isolates and tests **intra-model excursion-and-return update transport** (`E1`/`E2`) and **learned inference-time meta-control** (`E3_ridge`, `E3_mlp`) against linear update mixing (`M1`), direct state mixing (`M0`), fixed early exit (`EXIT_2`), and random control (`RAND_CTRL`), explicitly quantifying the **nonlinear curvature residual** $C_t = V_t^{\mathrm{visit}} - [(1-\alpha_{\mathrm{eff}})V_t^A + \alpha_{\mathrm{eff}}V_t^B]$. See [`literature_matrix.md`](literature_matrix.md), [`docs/methodology.md`](../docs/methodology.md), and [`docs/findings.md`](../docs/findings.md).

---

## 3. Mathematical Operator, Meta-Controller, and Compute Cost

At recurrent step $t = t_{\mathrm{visit}}$ (`visit_step \in \{2, 3\}` after forking at $t_{\mathrm{fork}} = 1$):
1. **Visited Neighborhood:**
   $$H_t^{\mathrm{visit}} = H_t^A + \alpha_{\mathrm{eff}} (H_t^{(d)} - H_t^A) \odot M_{\mathrm{scope}}$$
   bounded by relative Frobenius norm trust region $\rho_{\max} = 0.35$.
2. **Local Transition & Return:**
   $$V_t^A = F_{\theta, t}(H_t^A) - H_t^A, \qquad V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$$
   $$H_{t+1}^{A'} = H_t^A + (1 - \beta) V_t^A + \beta V_t^{\mathrm{visit}}$$
3. **Curvature Decomposition:**
   $$H_{t+1}^{A'} = \underbrace{H_t^A + (1 - \beta) V_t^A + \beta \big[(1 - \alpha_{\mathrm{eff}}) V_t^A + \alpha_{\mathrm{eff}} V_t^{(d)}\big]}_{\text{Linear Update Mixing (M1 / E1\_no\_curv)}} + \beta C_t$$
4. **Learned Inference-Time Controller $\pi_\phi(z_t)$:**
   Observes label-free state features $z_t \in \mathbb{R}^{29}$ (recipient entropy $H(p_t^A)$, top-1/top-2 margin $m_t^A$, step-to-step velocity cosine, native exit gate $g_t^A$, and donor compatibility $\|H_t^{(d)} - H_t^A\|_F / \|H_t^A\|_F$, $\Delta m_{t, d}$, $\mathrm{JSD}(p_t^A \| p_t^{(d)})$) and predicts relative counterfactual advantage $\hat{A}_\phi(z_t, a)$ over $\mathcal{A} = \{\text{CONTINUE}, \text{EXIT}, \text{EXCURSION}(d, \alpha, \beta)\}$.

The return/decomposition above describes the unprojected full-scope operator. Token masks, coordinate projections and clipping require the implemented correction formula; the recorded full curvature norm is not the applied correction norm. Gated E3 variants combine a learned scorer with hand-written feasibility gates; controller scores are not calibrated probabilities.

---

## 4. Ouro Integration, Provenance, and Exact Equivalence Verification

- Audited and pinned `ByteDance/Ouro-1.4B-Thinking` (`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, `model.safetensors` SHA-256 `6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb` in [`model_provenance.json`](../site/static/evidence/model_provenance.json)).
- Stepwise unrolling via `OuroRecurrentAdapter` (`use_cache=False`) achieved **`0.0` maximum absolute logit difference** (`1.0` top-5 overlap, `0.0` JSD) against native `OuroForCausalLM.forward` in the documented adapter checks ([`phase0_compatibility_report.md`](phase0_compatibility_report.md)).
- Current audits fail closed on historical parallel or missing ledgers. Fresh evaluation has 71 mismatches; exploratory controller evaluation has 81. Older `all_passed` audit JSON is historical, not a current certification. See [MVP scientific verification](mvp_scientific_verification.md).

---

## 5. Recorded results and uncertainty

The current [static manifest](../site/static/tebdil_data.js) derives statistics from exported records. These observations remain exploratory; fresh templates do not by themselves make a study confirmatory. Problem-cluster intervals keep repeated seeds together. Methods are compute-accounted where ledgers support the totals, not uniformly compute-matched.

### Fresh-template evaluation: 18 problems × 3 seeds

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

### Exploratory controller evaluation: 18 problems × 3 seeds

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

The [findings](../docs/findings.md) explain denominators and inference limits. All eight phases, including donor and transfer ablations, remain selectable in the observatory; full statistics and records are linked from the manifest. A ledger-gap count is a failed reconciliation, not a repaired cost estimate. P1 is an answer-informed oracle upper bound.

## 6. What is supported

- **Implemented and tested:** genuine pinned Ouro recurrence, hidden-state access, isolated branches, visited-state computation and return; trained scorers with explicit gates; fixed and heuristic controls.
- **Observed:** exploratory E3_mlp accuracy is 31.48%, versus S0 22.22%. On fresh templates, gated E3_mlp equals S0 at 27.78%; ungated E3_mlp reaches 31.48%, while M1 and random control reach 35.19%.
- **Not established:** broad generalization, superiority to the strongest baseline, universal safety, a causal necessity claim for reward shaping/curvature, or a validated compute Pareto frontier across rows with ledger gaps.

Fresh ungated E3's difference is +3.70 percentage points, problem-cluster 95% CI [0.00, +9.26], problem-level exact sign test p = 0.5. Fresh S1's difference is -16.67 points, interval [-33.33, 0.00], problem-level exact sign test p = 0.25. Neither claim should use repeated-seed instance counts as independent problem counts. Multiple exploratory comparisons add further uncertainty.

## 7. Release verification and remaining limits

The local full test suite passed 65 tests including real Ouro integration. Seven selected replay cases matched recorded answers, log-probabilities, costs and action types on Apple MPS. This does not mean every historical trace was replayed or every audit passed. The [scientific verification record](mvp_scientific_verification.md) records the failing historical-ledger audit and scope of real-model checks.

The public product is a static recorded-evidence explorer, not live inference. A clean clone can verify the committed export with `python3 scripts/build_landing_data.py --check`; rebuilding from research inputs requires ignored local `results/` and other artifacts. See [reproduction](../docs/reproduction.md).

No complete experiment campaign was rerun during this refactor. CUDA-specific behavior, larger independent benchmarks, preregistered comparisons and generalization remain unverified. Zero observed degradations are not a safety guarantee.

Parallel selection caveat: all historical P0/P1 total-selection costs are unverified, including rows whose selected-output ledger sums happen to match. The 71/81 counts describe arithmetic mismatches, not the full scope of uncertain parallel execution.
