# Phase 4 & 5 Report — Adaptive Label-Free Excursion Policy and Held-Out Generalization

## Objective and Hypothesis

Test **H4 (Selectivity)** and **H5 (Efficiency & Generalization)** on the held-out `test` split across disjoint problem templates and multiple seeds (`seeds=[42, 1042]`, $n=36$ paired instances): Evaluate whether a causal, label-free policy (`E2`) that triggers excursions only when branch $A$ exhibits high step-2 uncertainty achieves a superior accuracy-compute Pareto point compared to unconditional excursion (`E1`), extra recurrent depth (`S1`), parallel independent sampling (`P0`), and linear update mixing (`M1`) (`G4` and `G5` Gates).

## Source Revisions, Dependencies, and Hardware

- **Model:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, `bfloat16`).
- **Hardware:** Apple M1 Pro (`arm64`), 16.0 GB unified memory (`mps:0`).

## What Was Actually Implemented

- Implemented `AdaptiveExcursionPolicy` (`src/tebdil/core/policies.py`) using strictly label-free intermediate signals at `visit_step = 2`:
  - Intermediate readout Shannon entropy $H(p_A)$ and top-1/top-2 logit margin $m_A = z_{(1)}^A - z_{(2)}^A$.
  - Relative Frobenius branch divergence $\|H_2^B - H_2^A\|_F / \|H_2^A\|_F \in [0.05, 0.65]$.
  - Finite-value and RMS stability guards (`rms <= 40.0`).
- Calibrated thresholds exclusively on the `validation` split (`phase2_recovery`): trigger `visit` iff $m_A \le 0.70$ and $H(p_A) \ge 1.20$.

## Tests and Equivalence Checks

- `tests/test_no_label_leakage.py` verified via static source inspection and function signature checks that `AdaptiveExcursionPolicy` has zero access to `gold_answer` or formal validators.
- `tests/test_controls.py` verified that `E2` consumes `7` recurrent calls per token when skipping an excursion (`no_visit`) and `8` calls per token when executing an excursion (`visit`).

## Dataset and Split Policy

- **Split:** `test` (`seed_offset = 20000 / 22000 / 24000`, structural templates `arith_diff_of_products_test`, `arith_linear_sub_test`, `arith_weighted_comb_test`, `logic_bool_or_xor_test`, `logic_2sat_unique_test`, `graph_5node_test` completely disjoint from `pilot` and `validation`).
- **Instances:** 18 held-out problems $\times$ 2 seeds (`42`, `1042`) = **36 paired evaluations per method** (252 total trajectories in `results/phase4_adaptive/traces.jsonl`).

## Results with Uncertainty (`results/phase4_adaptive/summary.json`)

| Method | Test Accuracy (n=36) | Paired Diff vs `S0` | 95% Bootstrap CI | McNemar $p$ | Rescue Rate ($A$ wrong) | Degradation Rate ($A$ right) | Oracle Rescue | Mean Calls |
|---|---|---|---|---|---|---|---|---|
| `S0` | `0.2222` (8/36) | `0.0000` | `[0.0000, 0.0000]` | `1.000` | `0.000` (0/28) | `0.000` (0/8) | `0/1` | `7.33` |
| `S1` ($T=7$) | `0.2222` (8/36) | `0.0000` | `[0.0000, 0.0000]` | `1.000` | `0.000` (0/28) | `0.000` (0/8) | `0/1` | `12.83` |
| `P0` | `0.2222` (8/36) | `0.0000` | `[0.0000, 0.0000]` | `1.000` | `0.000` (0/28) | `0.000` (0/8) | `0/1` | `12.83` |
| `P1` *(Oracle)* | `0.2500` (9/36) | `+0.0278` | `[0.0000, +0.0833]` | `1.000` | `0.036` (1/28) | `0.000` (0/8) | `1/1` | `12.83` |
| `M1` | `0.2500` (9/36) | `+0.0278` | `[0.0000, +0.0833]` | `1.000` | `0.036` (1/28) | `0.000` (0/8) | `1/1` | `12.83` |
| `E1` | `0.2778` (10/36) | `+0.0556` | `[0.0000, +0.1389]` | `0.500` | `0.071` (2/28) | `0.000` (0/8) | `1/1` | `14.67` |
| `E2` *(Adaptive)* | `0.2778` (10/36) | `+0.0556` | `[0.0000, +0.1389]` | `0.500` | `0.071` (2/28) | `0.000` (0/8) | `1/1` | `14.03` |

### Per-Task-Family Breakdown on Held-Out Test Split

| Task Family | `S0` Accuracy | `S1` Accuracy | `P0` Accuracy | `M1` Accuracy | `E1` Accuracy | `E2` Accuracy |
|---|---|---|---|---|---|---|
| **Arithmetic** (n=12) | `0.0833` (1/12) | `0.0833` (1/12) | `0.0833` (1/12) | `0.0833` (1/12) | `0.1667` (2/12) | `0.1667` (2/12) |
| **Logic / 2-SAT** (n=12) | `0.3333` (4/12) | `0.3333` (4/12) | `0.3333` (4/12) | `0.3333` (4/12) | `0.3333` (4/12) | `0.3333` (4/12) |
| **Graph Shortest Path** (n=12) | `0.2500` (3/12) | `0.2500` (3/12) | `0.2500` (3/12) | `0.3333` (4/12) | `0.3333` (4/12) | `0.3333` (4/12) |

## Failure Cases and Negative Findings

1. **Modest Absolute Rescue Rate on Hard Held-Out Templates:**
   - Out of 28 failed `S0` test instances, `E1` and `E2` rescued 2 instances (`7.14%`: `arith_test_003` where `S0` predicted `42` and `E1`/`E2` recovered gold `48`, and `graph_test_005` where `S0` predicted `11` and `E1`/`E2` recovered gold `10`), while causing `0/8` (`0.0%`) degradations.
   - Because $n=36$, the `+5.56%` accuracy gain over `S0`, `S1`, and `P0` has a 95% bootstrap confidence interval of `[0.00%, +13.89%]` and McNemar $p = 0.50$, meaning the positive effect on this sample size is **suggestive rather than statistically definitive at $\alpha = 0.05$**.
2. **Emergent Cooperative Rescue Beyond Oracle Best-of-2 (`P1`):**
   - Notably, on `arith_test_003` (`seed=42`, gold `48`), both independent trajectory $A$ (`42`) and independent donor trajectory $B$ (`40`) were individually wrong, so `P1` (Oracle best-of-2) failed (`0`), whereas the excursion update $V_2^{\mathrm{visit}}$ evaluated at the midpoint neighborhood $H_2^{\mathrm{visit}}$ enabled `E1` and `E2` to decode the exact answer `48`.

## Artifacts and Reproduction Commands

- Artifacts: `results/phase4_adaptive/traces.jsonl`, `results/phase4_adaptive/summary.json`, `reports/final_results.json`.
- Reproduction:
  ```bash
  .venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml
  .venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/
  ```

## Go/No-Go Decision with Evidence

- **Decision:** **GO (Gates G4 and G5 Completed)**
- **Evidence:** On held-out test templates, `S0`, `M1`, and `E2` form the non-oracle Pareto-optimal frontier (`pareto_optimal_non_oracle: ["S0", "M1", "E2"]`), with `E2` dominating unconditional `E1` by achieving identical accuracy (`0.2778`) at lower recurrent call cost (`14.03` vs `14.67` calls).
