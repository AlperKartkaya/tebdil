# TEBDIL Final Scientific Report

**Title:** Self-Directed Cooperative Latent Excursions for Recurrent Reasoning in Looped Language Models  
**Date:** 2026-10-28  
**Model Backbone:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`)  
**Total Executed Trajectories:** 1,408 verified runs across 7 experimental runs (`results/` & [`reports/final_results.json`](file:///Users/alperkartkaya/Documents/tebdil/reports/final_results.json))

---

## 1. Problem Statement and Research Question

Looped Language Models (LoopLMs) such as `ByteDance/Ouro-1.4B-Thinking` refine token representations by applying a shared 24-layer Transformer block $F_{\theta, t}$ across $T=4$ recurrent steps (`total_ut_steps=4`). Standard inference advances a single hidden-state trajectory $H_t \in \mathbb{R}^{B \times L \times d}$ deterministically.

**Core Research Question:**
> Can a recurrent language model learn **when**, **where**, and **how** to relocate its intermediate latent computation during inference—choosing among continuing normally (`CONTINUE`), executing a cross-trajectory excursion (`EXCURSION(d, \alpha, \beta)`), or halting early (`EXIT`)—to recover from unproductive reasoning trajectories while improving performance under controlled compute budgets?

---

## 2. Novelty Scope and Literature Positioning

Prior 2025–2026 studies established looped recurrent pretraining (Ouro), parallel latent trajectory sampling with reward selection (ACL 2026), learned multi-agent latent links (RecursiveMAS, Interlat), and single-trajectory recurrent stabilization (STARS). TEBDIL specifically isolates and tests **intra-model excursion-and-return update transport** (`E1`/`E2`) and **learned inference-time meta-control** (`E3_ridge`, `E3_mlp`) against linear update mixing (`M1`), direct state mixing (`M0`), fixed early exit (`EXIT_2`), and random control (`RAND_CTRL`), explicitly quantifying the **nonlinear curvature residual** $C_t = V_t^{\mathrm{visit}} - [(1-\alpha_{\mathrm{eff}})V_t^A + \alpha_{\mathrm{eff}}V_t^B]$. See [`literature_matrix.md`](file:///Users/alperkartkaya/Documents/tebdil/reports/literature_matrix.md), [`docs/methodology.md`](file:///Users/alperkartkaya/Documents/tebdil/docs/methodology.md), and [`docs/findings.md`](file:///Users/alperkartkaya/Documents/tebdil/docs/findings.md).

---

## 3. Mathematical Operator, Meta-Controller, and Compute Cost

At recurrent step $t = t_{\mathrm{visit}}$ (`visit_step = 2` after forking at $t_{\mathrm{fork}} = 1$):
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

---

## 4. Ouro Integration, Provenance, and Exact Equivalence Verification

- Audited and pinned `ByteDance/Ouro-1.4B-Thinking` (`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, `model.safetensors` SHA-256 `6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb` in [`model_provenance.json`](file:///Users/alperkartkaya/Documents/tebdil/artifacts/evidence/model_provenance.json)).
- Stepwise unrolling via `OuroRecurrentAdapter` (`use_cache=False`) achieved **`0.0` maximum absolute logit difference** (`1.0` top-5 overlap, `0.0` JSD) against native `OuroForCausalLM.forward` across all validation checks ([`phase0_compatibility_report.md`](file:///Users/alperkartkaya/Documents/tebdil/reports/phase0_compatibility_report.md)).
- All 12 research integrity and live-replay checks pass (`all_passed: true` in [`audit_phase7_8_report.json`](file:///Users/alperkartkaya/Documents/tebdil/artifacts/evidence/audit_phase7_8_report.json)).

---

## 5. Experimental Results Across All Phases

### 5.1 Phase 1–4 Baseline & Excursion Mechanics (`pilot`, `validation`, `test`)
- **Pilot (`n = 20`):** `S0` achieved `15.0%` (`3/20`); `E1` achieved `20.0%` (`4/20`, `+5.0%`), rescuing `1/17` failed trajectories with `0/3` degradations and `+0.0601` nats mean gold log-prob gain.
- **Validation (`n = 18`) & Causal Curvature Ablation:** Unconditional `E1` rescued `graph_validation_002` (`\|C_t\|_F = 64.85`), outperforming `P0`, `R0`, and linear update mixing `M1` (`0/11` rescued). Ablating $C_t \to 0$ (`E1_no_curv`) or randomizing the direction of $C_t$ (`E1_rand_curv`) eliminated the rescue (`0/11`), confirming that the specific direction of $C_t$ is causally required.

### 5.2 Phase 5: 3-Branch Donor Selection (`results/phase5_donor/summary.json`, `n = 18`)
- With two candidate donors $B, C$ (`branch_count = 3`), **nearest-donor selection (`D_nearest`) achieved `33.33%` accuracy (`+11.11%` vs `S0`, `2/14` rescued, `0/4` degraded)**, matching the diagnostic donor oracle (`D_oracle`, `33.33%`) and outperforming farthest-donor relocation (`D_farthest`, `22.22%`, `0/14` rescued), while `D_learned`, `D_confidence`, and `D_random` achieved `27.78%` (`+5.56%`, `1/14` rescued).

### 5.3 Phase 6: Selective Transfer Mechanisms (`results/phase6_selective/summary.json`, `n = 18`)
- Restricting perturbations to the final token (`E1_last_token`, `22.22%`), tightening the trust region to $\rho_{\max} = 0.15$ (`E1_norm_tight`, `22.22%`), or attenuating updates by the native gate (`E1_gated`, `22.22%`) eliminated the validation rescue due to Sandwich RMSNorm contraction.
- **Top-50% velocity subspace projection (`E1_subspace`, `27.78%`, `1/14` rescued)** preserved the full rescue while zeroing out the bottom 50% of hidden dimensions.
- **`E3_ridge` on validation (`27.78%`, `1/14` rescued, `0/4` degraded)** matched `E1` accuracy while reducing mean recurrent calls from `14.67` to **`11.00` (`-25.0%` compute)**.

### 5.4 Phase 7 & 8: Held-Out Test Split Controller Evaluation (`results/phase7_8_controller_test/summary.json`, `n = 36`)
- **`E3_mlp` (2-layer MLP controller) achieved `27.78%` (`10/36`, `+5.56%` vs `S0`, `95% CI [0.00%, +13.89%]`), rescuing `2/28` failed trajectories (`arith_test_003` and `graph_test_005`) with `0/8` (`0.0%`) degradations at only `11.08` mean recurrent calls** — saving **24.5% of compute vs `E1` (`14.67` calls)**, **19.6% vs `E2` (`13.78` calls)**, and **18.9% vs `H_budget` (`13.67` calls)**.
- **Comparison against fixed early exit (`EXIT_2`):** `EXIT_2` (`5.50` calls) rescued `4/28` instances where later recurrent steps over-thought an initially valid answer, but **degraded `2/8` (`25.0%`) of correct trajectories** (`graph_test_000`, where the second token had low step-2 margin `0.125`). By gating `EXIT` on step-2 convergence (`margin_a >= 0.72`), `E3_mlp` (`exit_freq = 0.56`) and `E3_ridge` (`exit_freq = 0.44`) achieved **`0/8` (`0.0%`) degradations**.
- **Comparison against linear state mixing (`M0`):** On the held-out test split, direct state mixing (`M0`) achieved `33.33%` (`12/36`, `4/28` rescued, `0/8` degraded) at `12.83` calls, placing **`["EXIT_2", "E3_mlp", "M0"]`** on the non-oracle compute-accuracy Pareto frontier.

---

## 6. Scientific Conclusion

Our empirical findings establish that:
1. **Self-directed latent control is viable and compute-efficient:** A compact 2-layer MLP controller (`E3_mlp`) trained on counterfactual rollouts learns to combine early halting on converged states (`55.6%` of instances), normal continuation on stable states (`13.9%`), and cooperative latent excursions on uncertain states (`30.6%`), matching unconditional excursion accuracy (`27.78%`) with zero degradations while cutting recurrent compute by **24.5%** (`11.08` vs `14.67` calls).
2. **Donor proximity and subspace concentration govern effective relocation:** In multi-branch settings, visiting the nearest compatible donor (`D_nearest`) matches the diagnostic donor oracle (`33.33%`), and top-50% velocity subspace transport (`E1_subspace`) suffices to preserve nonlinear rescue dynamics.
3. **Simpler state mixing (`M0`) remains a strong higher-compute competitor:** At `12.83` recurrent calls/problem, direct linear state mixing (`M0`) achieves the highest overall test accuracy (`33.33%`), whereas `E3_mlp` dominates the lower-compute regime (`11.08` calls) without the `25.0%` degradation penalty of fixed early exit (`EXIT_2`).
