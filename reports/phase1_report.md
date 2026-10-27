# Phase 1 Report — Technical Feasibility and Excursion Grid

## Objective and Hypothesis

Test **H1 (Feasibility)**: Verify that a single pretrained LoopLM (`ByteDance/Ouro-1.4B-Thinking`) can be forked after recurrent step 1 into two isolated latent trajectories ($A, B$), execute a norm-bounded excursion-and-return update at step 2 across an $(\alpha, \beta)$ grid, satisfy exact identity limits at $\alpha=0, \beta=0, A=B$, and produce finite, measurable distributional shifts in output logits and decoded answers (`G1` Gate).

## Source Revisions, Dependencies, and Hardware

- **Model:** `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, `bfloat16`, `total_ut_steps=4`, `use_cache=False`).
- **Software Lock Hash:** `uv.lock` verified with PyTorch `2.14.1`, `transformers==4.55.0`.
- **Hardware:** Apple M1 Pro (`arm64`), 16.0 GB unified memory (`mps:0`).

## What Was Actually Implemented

1. **Branching & Seeded Perturbations (`src/tebdil/core/branching.py`, `perturbations.py`):**
   - Clones step-1 hidden state $H_1$ into isolated tensors (`data_ptr()` verified disjoint) for branch $A$ (unperturbed) and branch $B$ (normalized Gaussian perturbation $\epsilon$).
2. **Excursion-and-Return Operator (`src/tebdil/core/excursions.py`):**
   - Computes $H_t^{\mathrm{visit}} = H_t^A + \alpha_{\mathrm{eff}}(H_t^B - H_t^A)$ subject to relative trust-region clipping $\rho_{\max} = 0.35$.
   - Evaluates $V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$ and returns $H_{t+1}^{A'} = H_t^A + (1-\beta)V_t^A + \beta V_t^{\mathrm{visit}}$.
   - Extracts nonlinear curvature residual $C_t = V_t^{\mathrm{visit}} - [(1-\alpha_{\mathrm{eff}})V_t^A + \alpha_{\mathrm{eff}}V_t^B]$.

## Tests and Equivalence Checks

All 10 mandatory Phase 1 checks in `tests/` passed (`28 passed in 4.79s`):
- `[x]` Adapter disabled (`S0`) == original `OuroForCausalLM.forward` (`max_abs_diff = 0.0`).
- `[x]` $\alpha=0$ or $\beta=0$ exact identity behavior (`max_abs_diff = 0.0`, $\|C_t\|_F = 0.0$).
- `[x]` $A = B$ exact identity behavior (`max_abs_diff = 0.0`, $\|C_t\|_F = 0.0$).
- `[x]` In-place mutation of $B$ leaves $A$ untouched (`test_branch_isolation.py`).
- `[x]` Same-step / same-token-position / hash assertions enforced (`validate_compatibility`).
- `[x]` Zero-noise and seeded perturbation reproducibility verified.
- `[x]` Tensor shape, dtype, and device assertions active.
- `[x]` NaN/Inf and RMS threshold fallback (`skip_intervention`) verified.
- `[x]` All interventions written to JSONL trace logs (`results/phase1_smoke/traces.jsonl`).
- `[x]` `batch=1` time-sliced execution requires no cross-batch padding.

## Dataset and Split Policy

- **Split:** `pilot` (`seed_offset = 5000`, disjoint from `validation` and `test`).
- **Task Family:** 20 arithmetic & algebraic problems (`arith_pilot_000` .. `arith_pilot_019`).

## Experiments and Compute Budgets

1. **Calibration Sweep (`last_token` vs `full` scope):**
   Because `OuroDecoderLayer` applies four Sandwich `OuroRMSNorm` normalizations per layer across 24 layers plus step-end `self.norm`, small last-token-only perturbations ($\eta = 0.03$) are contracted by self-attention over the 29 unperturbed prefix tokens (`rel_dist = 0.0049`, `0` decision flips). Applying full-sequence scope (`token_scope: full`, $\eta = 0.25$, $\rho_{\max} = 0.35$) preserves branch separation across steps (`rel_dist` $\approx 0.25 \to 0.38$) while remaining 100% finite.
2. **Phase 1C $(\alpha, \beta)$ Grid (`results/phase1_grid/grid_summary.json`):**

| $\alpha$ | $\beta$ | Mean Max Abs Logit Diff | Mean Logit JSD | Mean $\|C_t\|_F$ | Mean $\Delta \log p(\text{gold})$ | Top-1 Flips (of 6) | All Finite |
|---|---|---|---|---|---|---|---|
| `0.00` | `0.00`–`1.00` | `0.000000` | `0.000000` | `0.0000` | `0.000000` | `0` | `True` |
| `0.25`–`1.00` | `0.00` | `0.000000` | `0.000000` | `0.0000` | `0.000000` | `0` | `True` |
| `0.25` | `0.25` | `0.179688` | `0.000062` | `18.2550` | `-0.003400` | `0` | `True` |
| `0.50` | `0.50` | `0.364583` | `0.000237` | `29.7402` | `+0.009612` | `0` | `True` |
| `0.75` | `0.50` | `0.486979` | `0.000383` | `35.1258` | `+0.015524` | `1` | `True` |
| `1.00` | `1.00` | `1.125000` | `0.001809` | `35.6761` | `+0.023502` | `2` | `True` |

## Results with Uncertainty (`results/phase1_smoke/summary.json`)

| Method | Accuracy (n=20) | Paired Diff vs `S0` | 95% Bootstrap CI | Rescue Rate | Degradation Rate | Mean $\Delta \log p(\text{gold})$ | Mean $\|C_t\|_F$ | Mean Recurrent Calls |
|---|---|---|---|---|---|---|---|---|
| `S0` | `0.1500` (3/20) | `0.0000` | `[0.0000, 0.0000]` | `0.000` (0/17) | `0.000` (0/3) | `0.00000` | `0.00` | `8.0` |
| `E1` | `0.2000` (4/20) | `+0.0500` | `[0.0000, +0.1500]` | `0.059` (1/17) | `0.000` (0/3) | `+0.06013` | `27.95` | `16.0` |

## Failure Cases and Negative Findings

- Under `token_scope: last_token` with $\eta \le 0.12$, Sandwich RMSNorm + prefix attention contracts the perturbation to $< 0.5\%$ relative distance, yielding zero decision changes. Full-sequence perturbation (`token_scope: full`, $\eta \ge 0.20$) is required to induce distinct branch trajectories in `Ouro-1.4B-Thinking`.

## Artifacts and Reproduction Commands

- Artifacts: `results/phase1_smoke/traces.jsonl`, `results/phase1_smoke/summary.json`, `results/phase1_grid/grid_summary.json`.
- Reproduction:
  ```bash
  .venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml
  .venv/bin/python scripts/run_phase1_grid.py
  ```

## Known Limitations

- Phase 1 evaluates only a 20-problem arithmetic pilot and compares `S0` vs `E1` without compute-matched controls (`P0`, `M0`, `M1`, `R0`), which are deferred to Phase 2 and Phase 3.

## Go/No-Go Decision with Evidence

- **Decision:** **GO (Gate G1 Passed)**
- **Evidence:** All identity invariants hold (`0.0` error), 100% of excursion trajectories remain finite, nonlinear curvature $\|C_t\|_F \approx 27.95$ (`35.3%` of $\|V_t^A\|_F$) is non-trivial, and `E1` increases mean gold log-probability by `+0.0601` nats and rescues `1/17` failed pilot instances with `0/3` degradations.

## Next Step Justified by Results

Proceed to **Phase 2 (Trajectory Recovery & Control Comparison)** across all three verifiable task families (`arithmetic`, `logic`, `graph`) on the validation split.
