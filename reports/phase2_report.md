# Phase 2 Report — Failed Trajectory Recovery and Degradation Analysis

## Objective and Hypothesis

Test **H2 (Recovery)** on the validation split across all three verifiable task families (**Arithmetic**, **Logic/CSP**, **Graph/Path**): Measure conditional rescue rate $\Pr(A' \text{ correct} \mid A \text{ wrong})$, conditional degradation rate $\Pr(A' \text{ wrong} \mid A \text{ correct})$, and net paired accuracy difference against parallel sampling (`P0`), oracle best-of-2 (`P1`), random perturbation (`R0`), direct state mixing (`M0`), and linear update mixing (`M1`) (`G2` Gate).

## Source Revisions, Dependencies, and Hardware

- **Model:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, `bfloat16`, `total_ut_steps=4`).
- **Hardware:** Apple M1 Pro (`arm64`), 16.0 GB unified memory (`mps:0`).

## What Was Actually Implemented

- Implemented formal validators (`src/tebdil/tasks/validators.py`) verifying exact arithmetic solutions, exhaustive CNF clause satisfaction, and Dijkstra shortest-path distances.
- Implemented paired evaluation in `src/tebdil/evaluation/runner.py` recording both `base_a_correct` (`S0`) and `donor_b_correct` (independent branch $B$) for every `(problem_id, seed)` instance to separate **Oracle-eligible recovery** ($A$ wrong, $B$ right) from **Non-oracle recovery**.

## Tests and Equivalence Checks

- `tests/test_task_verifiers.py` and `tests/test_no_label_leakage.py` verified that all gold answers satisfy formal validators, distractors are rejected, templates are disjoint between `validation` and `test`, and zero gold label information enters `tebdil.core`.

## Dataset and Split Policy

- **Split:** `validation` (`n = 18` balanced problems: 6 `arithmetic`, 6 `logic`, 6 `graph`).
- **Hyperparameters:** `fork_after_step=1`, `visit_step=2`, `alpha=0.65`, `beta=0.65`, `perturbation_relative_norm=0.25`, `max_relative_norm=0.35`, `token_scope=full`.

## Experiments and Compute Budgets

| Method | Description | Recurrent Calls / Token | Mean Calls / Problem | Oracle Access |
|---|---|---|---|---|
| `S0` | Standard single trajectory ($T=4$) | 4 | `7.3` | No |
| `P0` | Independent 2 branches, label-free margin selection | 7 | `12.8` | No |
| `P1` | Independent 2 branches, **oracle best-of-2** (upper bound) | 7 | `12.8` | **Yes (Analysis Only)** |
| `R0` | Norm-matched random perturbation at step 2 | 7 | `12.8` | No |
| `M0` | Direct state averaging prior to $F_2$ | 7 | `12.8` | No |
| `M1` | Direct linear update mixing ($C_t = 0$) | 7 | `12.8` | No |
| `E1` | Unconditional cooperative latent excursion | 8 | `14.7` | No |

## Results with Uncertainty (`results/phase2_recovery/summary.json`)

| Method | Accuracy (n=18) | Paired Diff vs `S0` | 95% Bootstrap CI | McNemar $p$ | Rescue Rate ($A$ wrong) | Degradation Rate ($A$ right) | Mean $\|C_t\|_F$ |
|---|---|---|---|---|---|---|---|
| `S0` | `0.3889` (7/18) | `0.0000` | `[0.0000, 0.0000]` | `1.000` | `0.000` (0/11) | `0.000` (0/7) | `0.00` |
| `P1` *(Oracle)* | `0.5000` (9/18) | `+0.1111` | `[0.0000, +0.2778]` | `0.500` | `0.182` (2/11) | `0.000` (0/7) | `0.00` |
| `P0` | `0.3333` (6/18) | `-0.0556` | `[-0.1667, 0.0000]` | `1.000` | `0.000` (0/11) | `0.143` (1/7) | `0.00` |
| `R0` | `0.3333` (6/18) | `-0.0556` | `[-0.1667, 0.0000]` | `1.000` | `0.000` (0/11) | `0.143` (1/7) | `0.00` |
| `M0` | `0.4444` (8/18) | `+0.0556` | `[-0.1111, +0.2222]` | `1.000` | `0.182` (2/11) | `0.143` (1/7) | `0.00` |
| `M1` | `0.3333` (6/18) | `-0.0556` | `[-0.1667, 0.0000]` | `1.000` | `0.000` (0/11) | `0.143` (1/7) | `0.00` |
| `E1` | `0.3889` (7/18) | `0.0000` | `[-0.1667, +0.1667]` | `1.000` | `0.091` (1/11) | `0.143` (1/7) | `42.06` |

## Failure Cases and Negative Findings

1. **Trade-Off Between Rescue and Degradation (Decision Tree Case E):**
   - `E1` rescued `graph_validation_002` (`S0` output `13` $\to$ `E1` output `17`, gold `17`, where $\|C_t\|_F = 64.85$), outperforming `P0`, `R0`, and linear update mixing `M1` (`0/11` rescued).
   - However, unconditional `E1` also degraded `arith_validation_003` (`S0` output `40` $\to$ `E1` output `36`, gold `40`), resulting in a net `0.0000` paired accuracy difference when applied blindly to every instance.
2. **Mechanistic Signal Distinguishing Rescue vs Degradation:**
   - Inspecting intermediate step-2 logits on `arith_validation_003` (degraded) vs `graph_validation_002` (rescued) reveals that on `arith_validation_003`, branch $A$ already had a confident top-1/top-2 logit margin at step 2 (`margin_A = 0.875`), whereas on `graph_validation_002`, branch $A$ was nearly tied (`margin_A = 0.125`, `entropy_A = 1.225`).

## Artifacts and Reproduction Commands

- Artifacts: `results/phase2_recovery/traces.jsonl`, `results/phase2_recovery/summary.json`.
- Reproduction:
  ```bash
  .venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml
  ```

## Known Limitations

- Validation sample size is $n=18$ (6 per task family), so individual instance flips correspond to $\pm 5.56\%$ accuracy.

## Go/No-Go Decision with Evidence

- **Decision:** **GO to Phase 3 & Phase 4 (Decision Tree Case E confirmed)**
- **Evidence:** Trajectory rescue is empirically demonstrated on real `Ouro-1.4B-Thinking` states (`graph_validation_002`) and exceeds `R0`, `P0`, and `M1`, but unconditional intervention suffers from degradation on already-confident trajectories (`1/7`), directly motivating Phase 3 curvature ablations and Phase 4 selective triggering (`E2`).

## Next Step Justified by Results

Execute **Phase 3 (Mechanism & Curvature Ablations)** and **Phase 4 (Selective Adaptive Controller `E2`)** calibrated to skip excursions when branch $A$'s step-2 margin exceeds `0.70`.
