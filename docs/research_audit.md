# Phase 0 Research Integrity & Baseline Audit

**Date:** 2026-10-28  
**Repository Commit Audited:** `9e74c323709f`  
**Verified Provenance Artifact:** [`artifacts/evidence/model_provenance.json`](file:///Users/alperkartkaya/Documents/tebdil/artifacts/evidence/model_provenance.json)

---

## 1. Model, Checkpoint, and Environment Verification

1. **Real Open-Weight Recurrent Model:**
   - Model ID: `ByteDance/Ouro-1.4B-Thinking`
   - Pinned Revision: `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`
   - Cached Snapshot: `/Users/alperkartkaya/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B-Thinking/snapshots/3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`
   - Primary Weight File: `model.safetensors` (`2,869,336,434` bytes, SHA-256: `6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb`).
   - Architecture: 24-layer Universal Transformer (`OuroForCausalLM`), `hidden_size=2048`, `intermediate_size=5504`, `num_attention_heads=16`, `total_ut_steps=4`, `vocab_size=49152`.
2. **Vendored Model Code & Instrumentation:**
   - `vendor/ouro_1_4b_thinking/configuration_ouro.py` (SHA-256: `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`) and `modeling_ouro.py` (SHA-256: `cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30`) match the official HuggingFace snapshot byte-for-byte.
   - `OuroRecurrentAdapter` (`src/tebdil/models/ouro_adapter.py`) executes genuine recurrent steps (`decoder_layer` 0..23 + `ouro_base.norm` + `ouro_base.early_exit_gate`) with `use_cache=False` to prevent KV-cache aliasing across forked trajectories.
3. **Hardware & Runtime Environment:**
   - OS / Hardware: `macOS-26.2-arm64` (Apple Silicon 16.0 GB unified memory, `mps:0`).
   - Python: `3.12.13`, PyTorch: `2.14.1`, Transformers: `4.55.0`.
   - Stepwise equivalence check (`.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml`) verified `max_abs_diff = 0.0`, `js_divergence = 0.0`, `top1_match = true`, and `topk_overlap = 1.0` across all 5 pilot prompts on `mps:0`.

---

## 2. Audit of Existing Baseline & Intervention Results

We inspected all existing artifacts in `results/` (`phase1_smoke`, `phase1_grid`, `phase2_recovery`, `phase3_ablations`, `phase4_adaptive`):

1. **Phase 1 (`results/phase1_smoke/summary.json`, `n=20` pilot arithmetic):**
   - `S0` accuracy: `15.0%` (`3/20`), `E1` accuracy: `20.0%` (`4/20`, `1/17` rescued, `0/3` degraded).
2. **Phase 2 (`results/phase2_recovery/summary.json`, `n=18` validation across arithmetic, logic, graph):**
   - `S0` accuracy: `38.89%` (`7/18`, `7.33` mean recurrent calls).
   - `P0` (parallel confidence selection): `33.33%` (`6/18`, `0/11` rescued, `1/7` degraded, `12.83` calls).
   - `P1` (Oracle best-of-2): `50.00%` (`9/18`, `2/11` rescued).
   - `M0` (direct state mixing): `44.44%` (`8/18`, `2/11` rescued, `1/7` degraded, `12.83` calls).
   - `M1` (linear update mixing): `33.33%` (`6/18`, `0/11` rescued, `1/7` degraded, `12.83` calls).
   - `R0` (norm-matched random perturbation): `33.33%` (`6/18`, `0/11` rescued, `1/7` degraded).
   - `E1` (fixed excursion at `visit_step=2`, $\alpha=\beta=0.65$): `38.89%` (`7/18`, `1/11` rescued on `graph_validation_002` with $\|C_t\|_F = 64.85$, `1/7` degraded on `arith_validation_003` where step-2 margin was already `0.875`).
3. **Phase 3 (`results/phase3_ablations/summary.json`, `n=18` validation):**
   - `E1_no_curv` ($C_t \to 0$) and `E1_rand_curv` (randomized $C_t$ direction) both lost the rescue (`0/11` rescued, `33.33%` accuracy).
   - `S1` ($T=7$ single branch) and `C0` (visit without return) achieved `44.44%` (`8/18`) on validation.
4. **Phase 4 (`results/phase4_adaptive/summary.json`, `n=36` held-out test across 2 seeds):**
   - `S0`, `S1`, `P0`: `22.22%` (`8/36`).
   - `P1`, `M1`: `25.00%` (`9/36`, `1/28` rescued).
   - `E1`, `E2`: `27.78%` (`10/36`, `2/28` rescued, `0/8` degraded; `E1` used `14.67` mean calls, `E2` used `13.78` mean calls in `summary.json`).
   - *Minor reporting discrepancy identified:* `reports/phase4_report.md` stated `14.03` mean calls for `E2`, whereas the actual artifact `results/phase4_adaptive/summary.json` recorded `13.7778` (`13.78`) mean calls.

---

## 3. Audit of Existing `E1`/`E2` Controller & Identified Gaps

Inspecting `src/tebdil/core/policies.py` and `src/tebdil/core/scheduler.py` reveals the exact gaps between the existing system and **Self-Directed Latent Relocation**:

1. **Action Space Gap:**
   - Existing `PolicyDecision.action` only supports `"visit"` vs `"no_visit"` at a single fixed step (`visit_step = 2`).
   - Missing explicit 3-action space $\mathcal{A} = \{\text{CONTINUE}, \text{EXCURSION}(\text{donor}, \alpha, \beta), \text{EXIT}\}$, including genuine early recurrent termination (`EXIT` at step $t < T$ to save recurrent block evaluations).
2. **Heuristic vs. Learned Controller Gap:**
   - Existing `E2` (`AdaptiveExcursionPolicy`) is a hand-tuned threshold rule (`margin_a <= 0.70 and entropy_a >= 1.20`).
   - There is no counterfactual data collector, no `train`-split trajectory dataset $(z_t, a_t, U_t)$, and no learned policy $\pi_\phi(z_t)$ trained on action utilities $U(s, a) = R_{\mathrm{task}}(s, a) - \lambda_{\mathrm{compute}} C(s, a)$.
3. **Donor & Parameter Selection Gap:**
   - Existing `E2` uses fixed $\alpha, \beta$ (`adaptive_alpha=base_alpha`, `adaptive_beta=base_beta`) and heuristic donor rules (`confidence_gain`, `diversity`, `random`, `fixed`).
   - Missing learned state-dependent selection over donor branches (e.g., 3-branch settings with nearest, farthest, highest-confidence, learned, and diagnostic oracle donor selection) and discrete $(\alpha, \beta)$ action choices.
4. **Verification & Audit Infrastructure Gap:**
   - Existing unit tests (`tests/*.py`) run exclusively on `OuroRecurrentAdapter.from_mini_config` (random CPU weights); no `@pytest.mark.real_model` integration test executes the real `ByteDance/Ouro-1.4B-Thinking` checkpoint in `pytest`.
   - No `python -m tebdil.audit --run-id <run_id>` CLI/module or deterministic replay command exists to independently audit run manifests, weight hashes, recurrent call accounting, and trace consistency.

---

## 4. Reproducible Baseline Commands

```bash
# 1. Inspect hardware and checkpoint configuration
.venv/bin/python -m tebdil.cli inspect-model

# 2. Verify exact 0.0 logit equivalence of OuroRecurrentAdapter on real checkpoint
.venv/bin/python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

# 3. Run existing unit test suite
.venv/bin/pytest -q
```
