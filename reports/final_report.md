# TEBDIL Final Scientific Report

**Title:** Cooperative Latent Excursions for Recurrent Reasoning in Looped Language Models  
**Date:** 2026-10-28  
**Model Backbone:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`)  
**Total Executed Trajectories:** 634 verified runs across 4 experimental phases (`results/` & `reports/final_results.json`)

---

## 1. Problem Statement and Research Question

Looped Language Models (LoopLMs) such as `ByteDance/Ouro-1.4B-Thinking` refine token representations by applying a shared 24-layer Transformer block $F_{\theta, t}$ across $T=4$ recurrent steps (`total_ut_steps=4`). Standard inference advances a single hidden-state trajectory $H_t \in \mathbb{R}^{B \times L \times d}$ deterministically.

**Core Research Question:**
> Can a recurrent language model recover from an unsuccessful latent reasoning trajectory by temporarily computing the recurrent transition $F_{\theta, t}$ in another co-evolving trajectory's latent neighborhood, then transporting the resulting local update vector back to its own state, more effectively than independent parallel sampling, direct linear mixing, random perturbation, or extra recurrent steps under matched compute budgets?

---

## 2. Novelty Scope and Literature Positioning

Prior 2025–2026 studies have established looped recurrent pretraining (Ouro), parallel latent trajectory sampling with reward selection (ACL 2026), learned multi-agent latent links (RecursiveMAS, Interlat), and single-trajectory recurrent stabilization (STARS). TEBDIL specifically isolates and tests **training-free, intra-model excursion-and-return update transport** (`E1`/`E2`) against linear update mixing (`M1`) and state mixing (`M0`), explicitly quantifying the **nonlinear curvature residual** $C_t = V_t^{\mathrm{visit}} - [(1-\alpha_{\mathrm{eff}})V_t^A + \alpha_{\mathrm{eff}}V_t^B]$. See [literature_matrix.md](file:///Users/alperkartkaya/Documents/tebdil/reports/literature_matrix.md) for the full comparative table.

---

## 3. Mathematical Operator, Invariants, and Compute Cost

At recurrent step $t = t_{\mathrm{visit}}$ (`visit_step = 2` after forking at $t_{\mathrm{fork}} = 1$):
1. **Visited Neighborhood:**
   $$H_t^{\mathrm{visit}} = H_t^A + \alpha_{\mathrm{eff}} (H_t^B - H_t^A) \odot M_{\mathrm{scope}}$$
   bounded by relative Frobenius norm trust region $\rho_{\max} = 0.35$.
2. **Local Transition & Return:**
   $$V_t^A = F_{\theta, t}(H_t^A) - H_t^A, \qquad V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$$
   $$H_{t+1}^{A'} = H_t^A + (1 - \beta) V_t^A + \beta V_t^{\mathrm{visit}}$$
3. **Curvature Decomposition:**
   $$H_{t+1}^{A'} = \underbrace{H_t^A + (1 - \beta) V_t^A + \beta \big[(1 - \alpha_{\mathrm{eff}}) V_t^A + \alpha_{\mathrm{eff}} V_t^B\big]}_{\text{Linear Update Mixing (M1 / E1\_no\_curv)}} + \beta C_t$$
4. **Compute Accounting:**
   For $T=4$ and fork after step 1, `S0` requires 4 recurrent core calls/token; 2-branch non-excursion controls (`P0`, `P1`, `M0`, `M1`, `R0`, `C0`) and `S1` ($T=7$) require 7 calls/token; unconditional excursion (`E0`, `E1`) requires 8 calls/token; selective excursion (`E2`) requires $7 + \mathbb{I}(\text{visit})$ calls/token.

---

## 4. Ouro Integration and Exact Equivalence Verification

- Audited and vendored `configuration_ouro.py` and `modeling_ouro.py` from `ByteDance/Ouro-1.4B-Thinking` (`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`).
- Stepwise unrolling via `OuroRecurrentAdapter` (`use_cache=False`) achieved **`0.0` maximum absolute logit difference** (`1.0` top-5 overlap, `0.0` JSD) against native `OuroForCausalLM.forward` across all validation checks ([phase0_compatibility_report.md](file:///Users/alperkartkaya/Documents/tebdil/reports/phase0_compatibility_report.md)).

---

## 5. Main Experimental Results Across Phases

### 5.1 Phase 1 Pilot (`n = 20` Arithmetic Problems, `pilot` Split)
- `S0` achieved `15.0%` accuracy (`3/20`); `E1` achieved `20.0%` (`4/20`, paired diff `+5.0%`, `95% CI [0.0%, +15.0%]`), rescuing `1/17` failed trajectories with `0/3` degradations and increasing mean gold token log-probability by `+0.0601` nats ([phase1_report.md](file:///Users/alperkartkaya/Documents/tebdil/reports/phase1_report.md)).

### 5.2 Phase 2 & 3 Validation Benchmark and Ablations (`n = 18` Balanced Problems, `validation` Split)
- On the validation split (`6` arithmetic, `6` logic/2-SAT, `6` graph shortest-path), baseline `S0` achieved `38.89%` (`7/18`) and Oracle Best-of-2 (`P1`) achieved `50.00%` (`9/18`).
- Unconditional `E1` rescued `1/11` failed trajectories (`graph_validation_002`, where $\|C_t\|_F = 64.85$), outperforming `P0`, `R0`, and linear update mixing `M1` (`0/11` rescued), but degraded `1/7` already-correct trajectories (`arith_validation_003`, where branch $A$ was already confident at step 2 with margin `0.875`), yielding net `38.89%` accuracy.
- **Causal Curvature Ablation:** Ablating $C_t \to 0$ (`E1_no_curv` / `M1`) dropped accuracy from `38.89%` to `33.33%` (`0/11` rescued). Replacing $C_t$ with a norm-matched random direction (`E1_rand_curv`) also eliminated the rescue (`0/11`), demonstrating that the **specific direction** of the nonlinear update residual $C_t$ is causally required for that rescue.
- **Random-Direction Excursion (`E0`):** Excursion along a random direction of matched distance induced $4\times$ larger curvature distortion ($\|C_t\|_F = 168.41$ vs `42.06` for donor-directed `E1`) and degraded gold log-probability by `-0.1254` nats.
- **Strong Simpler Competitors on Validation:** Extra recurrent depth (`S1`, `44.44%`), direct state mixing (`M0`, `44.44%`), and visit-without-return (`C0`, `44.44%`) matched or exceeded unconditional `E1` on the validation set at lower compute (`12.8` vs `14.7` calls).

### 5.3 Phase 4 & 5 Held-Out Test Split (`n = 36` Paired Instances across 3 Families and 2 Seeds)
- On held-out test templates (`results/phase4_adaptive/summary.json`), `S0`, `S1` ($T=7$), and `P0` each achieved `22.22%` (`8/36`).
- Linear update mixing (`M1`) and Oracle Best-of-2 (`P1`) achieved `25.00%` (`9/36`, `+2.78%`, `1/28` rescued).
- Both `E1` and the validation-calibrated label-free adaptive policy `E2` achieved **`27.78%` (`10/36`, `+5.56%`, `95% CI [0.00%, +13.89%]`)**, rescuing `2/28` (`7.14%`) failed trajectories with **`0/8` (`0.0%`) degradations**.
- Crucially, `E2` skipped unnecessary excursions on confident trajectories, reducing mean recurrent calls from `14.67` (`E1`) to `14.03` and placing **`["S0", "M1", "E2"]`** on the non-oracle Pareto-optimal frontier ([phase4_report.md](file:///Users/alperkartkaya/Documents/tebdil/reports/phase4_report.md)).

---

## 6. Scientific Conclusion (Open Science Commitment)

In accordance with Section 28 of the specification, our empirical findings support **Outcome 2 (Conditionally advantageous under specific mechanistic conditions, with strong competition from simpler baselines)**:
1. **When TEBDIL helps:** When full-sequence perturbations ($\eta \approx 0.25$) are used to overcome `Ouro-1.4B-Thinking`'s strong Sandwich RMSNorm contraction, visiting a co-evolving donor branch at an intermediate step (`visit_step = 2`) produces a directional nonlinear update component $C_t$ that rescues specific failed arithmetic and graph trajectories—including cases (`arith_test_003`) where neither branch $A$ nor donor $B$ succeeded independently (`P1` failed, `E1`/`E2` succeeded).
2. **When TEBDIL hurts or is unnecessary:** Unconditional excursion (`E1`) wastes compute and can degrade trajectories where branch $A$ already has a high intermediate logit margin (`arith_validation_003`), and on several validation instances simpler state mixing (`M0`) or extra recurrent steps (`S1`) perform equally well at lower compute cost (`12.8` vs `14.7` calls).
3. **Role of Selective Triggering (`E2`):** Filtering excursions via label-free intermediate logit uncertainty ($m_A \le 0.70, H(p_A) \ge 1.20$) reduces compute cost while preserving observed test-split rescues.
