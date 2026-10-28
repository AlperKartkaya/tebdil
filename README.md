# TEBDIL — Cooperative Latent Excursions for Recurrent Reasoning

Open-source research platform investigating inference-time **cooperative latent excursions** in recurrent/looped language models (`ByteDance/Ouro-1.4B-Thinking`).

---

## 1. Overview

When a looped language model iterates a shared recurrent core $F_{\theta, t}$ across $T$ latent steps, an individual trajectory $A$ may progress along an ineffective reasoning route. **TEBDIL** forks a co-evolving latent trajectory $B$ after the first recurrent step, temporarily visits $B$'s bounded neighborhood $H_t^{\mathrm{visit}} = H_t^A + \alpha_{\mathrm{eff}}(H_t^B - H_t^A)$ at an intermediate step $t_{\mathrm{visit}}$, evaluates the nonlinear recurrent transition $V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$ there, and transports the local update direction back to $A$:

$$H_{t+1}^{A'} = H_t^A + (1 - \beta) V_t^A + \beta V_t^{\mathrm{visit}}$$

This repository provides:
1. **Instrumented LoopLM Runtime (`tebdil.models`):** Stepwise `OuroRecurrentAdapter` with verified `0.0` logit discrepancy against `ByteDance/Ouro-1.4B-Thinking` (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`), plus a fast pure-PyTorch `ToyRecurrentAdapter` for CPU CI tests.
2. **Cooperative Latent Excursion & Control Engine (`tebdil.core`):** Branch forking, norm-bounded perturbations, excursion-and-return, curvature residual decomposition ($C_t$), label-free adaptive triggering (`E2`), and matched-compute controls (`S0`, `S1`, `P0`, `P1`, `M0`, `M1`, `R0`, `C0`, `E0`, `E1`, `E2`).
3. **Verifiable Evaluation Suite (`tebdil.tasks`, `tebdil.evaluation`, `tebdil.analysis`):** Split-isolated problem generators and formal validators for **Arithmetic**, **Boolean Logic / 2-SAT**, and **Weighted Graph Shortest Path**, with paired bootstrap confidence intervals, exact McNemar tests, and Pareto frontier analysis.
4. **TEBDIL Observatory (`ui/app.py`):** Interactive Streamlit + Plotly interface inspecting real trajectory traces, full-dimensional step divergence, curvature $\|C_t\|_F$, rescue/degradation outcomes, and 2D PCA projections.

---

## 2. Key Empirical Results (`ByteDance/Ouro-1.4B-Thinking`)

### Held-Out Test Split (`configs/phase4_adaptive.yaml`, $n=36$ paired instances across 3 task families & 2 seeds)

| Method | Description | Accuracy | Paired Diff vs `S0` | 95% Bootstrap CI | Rescue Rate | Degradation Rate | Mean $\|C_t\|_F$ | Mean Calls |
|---|---|---|---|---|---|---|---|---|
| `S0` | Standard single trajectory ($T=4$) | `22.22%` | `0.00%` | `[0.00%, 0.00%]` | `0.00%` (0/28) | `0.00%` (0/8) | `0.00` | `7.33` |
| `S1` | Extended single trajectory ($T=7$) | `22.22%` | `0.00%` | `[0.00%, 0.00%]` | `0.00%` (0/28) | `0.00%` (0/8) | `0.00` | `12.83` |
| `P0` | Parallel 2-branch, label-free margin selection | `22.22%` | `0.00%` | `[0.00%, 0.00%]` | `0.00%` (0/28) | `0.00%` (0/8) | `0.00` | `12.83` |
| `P1` | Parallel 2-branch, **oracle best-of-2** (upper bound) | `25.00%` | `+2.78%` | `[0.00%, +8.33%]` | `3.57%` (1/28) | `0.00%` (0/8) | `0.00` | `12.83` |
| `M1` | Direct linear update mixing ($C_t = 0$) | `25.00%` | `+2.78%` | `[0.00%, +8.33%]` | `3.57%` (1/28) | `0.00%` (0/8) | `0.00` | `12.83` |
| `E1` | Fixed-schedule cooperative latent excursion | `27.78%` | `+5.56%` | `[0.00%, +13.89%]` | `7.14%` (2/28) | `0.00%` (0/8) | `45.83` | `14.67` |
| `E2` | **Adaptive label-free excursion (Pareto-optimal)** | **`27.78%`** | **`+5.56%`** | `[0.00%, +13.89%]` | **`7.14%` (2/28)** | **`0.00%` (0/8)** | `20.22` | **`14.03`** |

---

## 3. Quick Start & CLI Usage

```bash
# 1. Install locked environment
uv sync

# 2. Run unit and invariant tests (no GPU/checkpoint download required)
.venv/bin/pytest -q

# 3. Inspect hardware and pinned Ouro checkpoint configuration
.venv/bin/python -m tebdil.cli inspect-model --model ByteDance/Ouro-1.4B-Thinking

# 4. Verify exact stepwise adapter equivalence (max_abs_diff == 0.0)
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

# 5. Run individual experimental phases
.venv/bin/python -m tebdil.cli run --config configs/phase1_smoke.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase2_recovery.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase3_ablations.yaml
.venv/bin/python -m tebdil.cli run --config configs/phase4_adaptive.yaml

# 6. Aggregate paired statistics and Pareto frontier into reports/final_results.json
.venv/bin/python -m tebdil.cli analyze --runs results/ --output reports/

# 7. Launch Interactive Academic Landing Page UI (http://127.0.0.1:8080)
.venv/bin/python scripts/build_landing_data.py
.venv/bin/python -m tebdil.cli serve-site --host 127.0.0.1 --port 8080

# 8. Launch Streamlit Observatory UI
.venv/bin/streamlit run ui/app.py
```

---

## 4. Documentation and Reports

- **Architecture & Invariants:** [docs/architecture.md](file:///Users/alperkartkaya/Documents/tebdil/docs/architecture.md)
- **Methodology & Controls:** [docs/methods.md](file:///Users/alperkartkaya/Documents/tebdil/docs/methods.md)
- **Reproduction Guide:** [docs/reproduction.md](file:///Users/alperkartkaya/Documents/tebdil/docs/reproduction.md)
- **Limitations & Scope:** [docs/limitations.md](file:///Users/alperkartkaya/Documents/tebdil/docs/limitations.md)
- **Literature Matrix:** [reports/literature_matrix.md](file:///Users/alperkartkaya/Documents/tebdil/reports/literature_matrix.md)
- **Decision Log:** [reports/decision_log.md](file:///Users/alperkartkaya/Documents/tebdil/reports/decision_log.md)
- **Experiment Registry:** [reports/experiment_registry.md](file:///Users/alperkartkaya/Documents/tebdil/reports/experiment_registry.md)
- **Phase Reports:** [Phase 0](file:///Users/alperkartkaya/Documents/tebdil/reports/phase0_compatibility_report.md) · [Phase 1](file:///Users/alperkartkaya/Documents/tebdil/reports/phase1_report.md) · [Phase 2](file:///Users/alperkartkaya/Documents/tebdil/reports/phase2_report.md) · [Phase 3](file:///Users/alperkartkaya/Documents/tebdil/reports/phase3_report.md) · [Phase 4 & 5](file:///Users/alperkartkaya/Documents/tebdil/reports/phase4_report.md) · [Final Report](file:///Users/alperkartkaya/Documents/tebdil/reports/final_report.md)
- **Aggregated Metrics JSON:** [reports/final_results.json](file:///Users/alperkartkaya/Documents/tebdil/reports/final_results.json)

---

## 5. License

Licensed under [Apache-2.0](file:///Users/alperkartkaya/Documents/tebdil/LICENSE). The underlying `ByteDance/Ouro-1.4B-Thinking` model is licensed under Apache-2.0 by ByteDance.
