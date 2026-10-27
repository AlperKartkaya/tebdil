# Experiment Registry and Pre-Registration Protocol

**Project:** TEBDIL — Cooperative Latent Excursions for Recurrent Reasoning  
**Created:** 2026-10-27  
**Status:** Active Pre-Registration (locked prior to main test runs)

---

## 1. Pre-Registered Hypotheses

- **H1 (Feasibility):** A single pretrained LoopLM (`ByteDance/Ouro-1.4B-Thinking`) can be unrolled step-by-step via `RecurrentAdapter` with exact numerical equivalence to standard forward execution, and forked into isolated latent trajectories $A, B$ with bounded perturbations.
- **H2 (Recovery):** On reasoning instances where reference trajectory $A$ fails, targeted excursion-and-return (`E1`/`E2`) achieves a positive net paired accuracy gain ($\mathrm{Acc}(A') - \mathrm{Acc}(A) > 0$) and higher conditional rescue rate than norm-matched random perturbation (`R0`) and direct state mixing (`M0`).
- **H3 (Nonlinear Mechanism):** The excursion update differs from linear update mixing (`M1`) by a non-zero curvature component $C_t = V_t^{\mathrm{visit}} - [(1-\alpha)V_t^A + \alpha V_t^B]$, and ablating $C_t$ (reducing `E1` to `M1`) alters recovery behavior.
- **H4 (Selectivity):** A label-free adaptive policy (`E2`) triggering excursions only under high branch stagnation/uncertainty and sufficient donor divergence reduces degradation rate compared to unconditional excursion (`E1`).
- **H5 (Compute Efficiency):** Under matched recurrent block calls (`num_recurrent_block_calls`) and estimated FLOPs, selective excursion (`E2`) competes favorably on the accuracy-compute Pareto frontier against extra recurrent depth (`S1`) and parallel independent branches (`P0`).

---

## 2. Primary and Secondary Metrics

### Primary Evaluation Metrics
1. **Task Accuracy (`accuracy`):** Fraction of problems where the decoded continuation satisfies the exact formal validator.
2. **Paired Accuracy Difference (`paired_acc_diff`):** $\mathrm{Acc}(\text{Method}) - \mathrm{Acc}(\text{Baseline } A)$ paired per `(problem_id, seed)`, with 95% paired bootstrap confidence intervals (1,000 resamples) and exact McNemar test $p$-values.
3. **Conditional Rescue Rate (`rescue_rate`):** $\Pr(A' \text{ correct} \mid A \text{ wrong})$.
4. **Conditional Degradation Rate (`degradation_rate`):** $\Pr(A' \text{ wrong} \mid A \text{ correct})$.
5. **Compute Cost (`recurrent_calls`, `estimated_flops`, `wall_time_ms`):** Exact count of recurrent core transitions $F_{\theta,t}$ executed per problem.

### Mechanistic Metrics
1. **Branch Divergence:** Frobenius distance $\|H_t^B - H_t^A\|_F$, relative distance $\|H_t^B - H_t^A\|_F / \|H_t^A\|_F$, and mean token cosine similarity.
2. **Update Norms:** $\|V_t^A\|_F$, $\|V_t^{\mathrm{visit}}\|_F$.
3. **Curvature Residual:** $\|C_t\|_F$ and relative nonlinearity ratio $\|C_t\|_F / (\|V_t^A\|_F + \varepsilon)$.
4. **Distributional Shift:** Final readout logits Jensen-Shannon divergence (JSD), KL divergence $\mathrm{KL}(P_A \parallel P_{A'})$, and gold token log-probability delta $\Delta \log p(\text{gold})$.
5. **Numerical Stability:** Finite checks (`isfinite`), state RMS, and max relative perturbation clipping indicator.

---

## 3. Dataset Splits and Anti-Leakage Protocol

Across all three verifiable task families (**Arithmetic**, **Logic/CSP**, **Graph/Path**):
- **`pilot` split:** 20–30 problems per family used in Phase 1 to verify numerical stability, logit shifts, and finite norms.
- **`validation` split:** Used exclusively to tune hyperparameters ($\alpha, \beta$, relative norm clip $\rho$, `visit_step`, and adaptive controller thresholds). Template parameters and random generator seeds (`seed_offset = 10000`) are disjoint from `test`.
- **`test` split:** Held-out problem templates and instances (`seed_offset = 20000`) evaluated **only once** after hyperparameters and controller rules are frozen in config files.
- **No Label Leakage Rule:** Gold answers and validators are strictly isolated in `tasks/validators.py` and `evaluation/metrics.py`. No method in `core/` (`excursions.py`, `controls.py`, `policies.py`) accepts or inspects gold labels. Oracle donor selection (`P1`, `E_oracle`) is computed post-hoc or via an explicitly flagged analysis-only wrapper that cannot be invoked by non-oracle policies.

---

## 4. Comparison Systems Registry

| Condition ID | Description | Recurrent Calls ($T=4$, fork at $t=1$, visit at $t=2$) | Oracle Access |
|---|---|---|---|
| `S0` | Standard single trajectory ($T=4$) | 4 | No |
| `S1` | Extended single trajectory ($T=7$, compute-matched to 2-branch excursion) | 7 | No |
| `P0` | Independent 2 branches ($A, B$), label-free confidence/entropy selection | 7 (1 shared + 3×2) | No |
| `P1` | Independent 2 branches, **oracle best-of-2** (upper bound only) | 7 | **Yes (Analysis Only)** |
| `M0` | Direct hidden-state averaging $H_t^{A'} = (1-\alpha)H_t^A + \alpha H_t^B$ | 7 | No |
| `M1` | Direct update-vector mixing $H_{t+1}^{A'} = H_t^A + (1-\beta)V_t^A + \beta[(1-\alpha)V_t^A + \alpha V_t^B]$ | 7 | No |
| `R0` | Norm-matched random perturbation at `visit_step` | 7 | No |
| `C0` | Visit without return: jump to $H_t^{\mathrm{visit}}$ and continue ($H_{t+1}^{A'} = F_{\theta,t}(H_t^{\mathrm{visit}})$) | 7 | No |
| `E0` | Excursion toward random direction with matched visit norm | 8 (1 shared + 3×2 + 1 visit) | No |
| `E1` | Fixed-schedule cooperative latent excursion toward donor $B$ | 8 | No |
| `E2` | Adaptive label-free cooperative latent excursion (triggered selectively) | $7 + \mathbb{I}(\text{visit})$ | No |
| `S1_8` | Extended single trajectory ($T=8$, exact call-match to `E1`) | 8 | No |

---

## 5. Phase Gate Status Log

| Gate | Requirement | Status | Evidence Artifact |
|---|---|---|---|
| **G0** | Stepwise adapter reproduces standard Ouro forward within numerical tolerance | **PASSED (`max_abs_diff = 0.0`)** | `reports/phase0_compatibility_report.md` |
| **G1** | Branching & excursion produce finite, norm-bounded states and measurable logit shifts | **PASSED (`100% finite`, `JSD > 0`, `rescue 1/17`)** | `reports/phase1_report.md` |
| **G2** | Sequence-level task accuracy and rescue/degradation measured with formal validators | **PASSED (`n=18 validation`, rescue & degradation quantified)** | `reports/phase2_report.md` |
| **G3** | Matched-compute ablations and curvature $C_t$ causal analysis completed | **PASSED (`12 methods` + timing/scope ablations)** | `reports/phase3_report.md` |
| **G4** | Label-free adaptive controller (`E2`) evaluated on held-out test split | **PASSED (`+5.56%` vs `S0`, `0%` degradation, Pareto-optimal)** | `reports/phase4_report.md` |
| **G5** | Cross-family generalization, Observatory UI, and reproduction suite verified | **PASSED (`634 records`, UI & `reproduce.sh` verified)** | `reports/final_report.md` |
