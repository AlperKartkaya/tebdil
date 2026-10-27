# Architectural and Engineering Decision Log

**Project:** TEBDIL — Cooperative Latent Excursions for Recurrent Reasoning  
**Maintainer:** Autonomous Engineering Agent

---

## ADR-001: Pinned Ouro Checkpoint and Vendored Modeling Code

- **Date:** 2026-10-27
- **Context:** `ByteDance/Ouro-1.4B-Thinking` uses custom Hugging Face modeling code (`configuration_ouro.py`, `modeling_ouro.py`) via `auto_map`. Running remote code blindly with `trust_remote_code=True` poses both security and reproducibility risks.
- **Decision:**
  1. Pin `ByteDance/Ouro-1.4B-Thinking` to commit SHA `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` (and `ByteDance/Ouro-1.4B` to `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`).
  2. Vendor and audit `configuration_ouro.py` (SHA256: `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`) and `modeling_ouro.py` (SHA256: `cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30`) under `vendor/ouro_1_4b_thinking/`.
  3. Load `OuroConfig` and `OuroForCausalLM` directly from the audited local module rather than executing unreviewed remote code at runtime.

---

## ADR-002: Definition of Recurrent State $H_t$ in Ouro

- **Date:** 2026-10-27
- **Context:** In `modeling_ouro.py` (`OuroModel.forward`, lines 633–657), the recurrent loop executes:
  ```python
  hidden_states = inputs_embeds  # step 0 input
  for current_ut in range(self.total_ut_steps):
      for decoder_layer in self.layers[: self.config.num_hidden_layers]:
          hidden_states = decoder_layer(hidden_states, ..., current_ut=current_ut)
      hidden_states = self.norm(hidden_states)
      hidden_states_list.append(hidden_states)
      gate_list.append(self.early_exit_gate(hidden_states))
  ```
- **Decision:**
  - Define $H_0 = \mathrm{embed\_tokens}(\mathrm{input\_ids}) \in \mathbb{R}^{B \times L \times d}$ as the initial pre-recurrent state (`recurrent_step = 0`).
  - Define each recurrent transition $F_{\theta, t}(H_t)$ for $t \in \{0, \dots, T-1\}$ as executing all `num_hidden_layers` (24) `OuroDecoderLayer` blocks followed by `self.norm(hidden_states)`.
  - Thus $H_{t+1} = \mathrm{norm}(\mathrm{Layers}_{0..23}(H_t))$ is the exact state stored in `hidden_states_list[t]` and fed into step $t+1$.
- **Consequence:** Stepwise unrolling in `OuroRecurrentAdapter` matches `OuroModel.forward` with zero mathematical discrepancy when `use_cache=False`.

---

## ADR-003: Cache Policy (`use_cache=False`) for Recurrent Interventions

- **Date:** 2026-10-27
- **Context:** `OuroAttention.forward` indexes `UniversalTransformerCache` by `current_ut * num_hidden_layers + layer_idx`. During a forward pass over a full prompt (prefill) or stateless multi-step continuation, enabling KV caching while branching or perturbing hidden states would require cloning and mutating $24 \times 4 = 96$ KV cache slots or risking cross-branch cache contamination. Furthermore, when `use_cache=False`, a single forward pass over `input_ids` does not use `past_key_values` at all—each UT step attends over the current sequence's keys and values computed from that UT step's `hidden_states`.
- **Decision:** Enforce `use_cache=False` across all Phase 1–4 recurrent state interventions. For multi-token continuation (Level 1 evaluation), each generated token appends to `input_ids` and runs the instrumented recurrent loop cleanly without stale KV state.

---

## ADR-004: Early-Exit Gate Handling and Readout Determinism

- **Date:** 2026-10-27
- **Context:** `OuroForCausalLM.forward` computes an exit probability distribution across steps using `early_exit_gate` when `early_exit_threshold` is set (`1.0` in `config.json`). In `bfloat16`, if `sigmoid(gate)` rounds to `1.0` at an early step $t < T-1$, `cumulative_probs >= 1.0` can select step $t$ for specific tokens, whereas `exit_at_step = T - 1` (or `outputs.last_hidden_state`) always reads out from the final recurrent step $H_T$.
- **Decision:**
  - Support both readout modes in `OuroRecurrentAdapter`:
    1. `last_step` (`lm_head(H_T)`): deterministic fixed-depth readout required by Phase 1A ("Early-exit sabitlemesi yap; testin recurrent step sayısı konusunda değişken olmamalı").
    2. `native_gate`: tracks `hidden_states_list` and `gate_list` across steps to reproduce `OuroForCausalLM.forward` bit-for-bit when validating against default model forward calls.
  - Verify exact equivalence of both modes against `OuroForCausalLM.forward`.

---

## ADR-005: Hardware Execution Environment (Apple M1 Pro 16 GB)

- **Date:** 2026-10-27
- **Context:** The host machine is an Apple M1 Pro with 16 GB unified memory running macOS (`arm64`). `ByteDance/Ouro-1.4B-Thinking` weights in `bfloat16`/`float16` occupy ~2.87 GB on disk and in unified memory, fitting comfortably in 16 GB RAM on `mps` or `cpu`.
- **Decision:**
  - Use a dual-tier testing and execution strategy:
    1. **Fast deterministic `ToyRecurrentAdapter`**: A small, self-contained shared-weight looped transformer in pure PyTorch (`d=64`, `layers=2`, `steps=4`) for rapid unit testing, CI verification, and high-sample statistical sweeps.
    2. **Real `OuroRecurrentAdapter` (`ByteDance/Ouro-1.4B-Thinking`)**: Full integration, adapter equivalence verification, and real-model Phase 1–4 experiments running on local unified memory (`mps` / `cpu` in `bfloat16` / `float32`).
