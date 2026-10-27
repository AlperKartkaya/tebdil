# Literature Matrix and Novelty Scope

**Project:** TEBDIL — Cooperative Latent Excursions for Recurrent Reasoning  
**Date:** 2026-10-27  
**Status:** Pre-experiment literature synthesis and falsifiable scope boundary

---

## 1. Overview and Positioning

Looped Language Models (LoopLMs) and Universal Transformer variants iterate a shared set of transformer parameters across multiple recurrent steps (`total_ut_steps`) before projecting hidden states to vocabulary logits. Recent 2025–2026 work has explored scaling recurrent depth, stabilizing recurrent dynamics, parallel latent sampling, and inter-agent latent communication.

TEBDIL investigates a specific, falsifiable mechanism within a **single shared-weight LoopLM**:
> When a recurrent trajectory $A$ is progressing along an ineffective latent route, does temporarily computing the recurrent transition $F_{\theta,t}$ at a bounded excursion point $H_t^{\mathrm{visit}}$ toward a co-evolving trajectory $B$, and transporting only the resulting local update vector $V_t^{\mathrm{visit}} = F_{\theta,t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$ back to $A$, recover failed trajectories more effectively than matched-compute independent sampling, extra recurrent steps, direct state mixing, or direct update mixing?

---

## 2. Comparative Matrix of Prior and Concurrent Work

| Reference | Core Mechanism & Contribution | Relation to TEBDIL | Falsifiable Distinction Tested in TEBDIL |
|---|---|---|---|
| **Ouro / Looped Language Models** ([Zhu et al., 2025](https://huggingface.co/papers/2510.25741); [`ByteDance/Ouro-1.4B-Thinking`](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)) | Shared-weight 24-layer Universal Transformer with 4 recurrent steps (`total_ut_steps=4`), Sandwich RMSNorm, per-step `self.norm`, and adaptive exit gate (`early_exit_gate`). | Provides the primary open-weight recurrent reasoning backbone (`rev: 3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`). | Ouro executes a single deterministic trajectory per prompt. TEBDIL instruments the inter-step boundary to fork trajectories and execute cross-branch excursion-and-return updates. |
| **Parallel Test-Time Scaling for Latent Reasoning Models** ([ACL 2026](https://aclanthology.org/2026.acl-long.2069/)) | Generates multiple parallel latent trajectories via latent noise or dropout and selects among completed trajectories using a latent reward / scoring model. | Establishes that controlled latent perturbations produce diverse trajectories in LoopLMs (`P0` / `P1` baselines). | Parallel scaling keeps branches isolated until readout. TEBDIL tests whether **online mid-recurrence interaction** via excursion transition rescues failing branches beyond independent parallel selection (`P0`). |
| **GRAM — Generative Recursive Reasoning** ([2026](https://ahn-ml.github.io/gram-website/)) | Explores recursive multi-path latent generation and structured latent reasoning transitions. | Demonstrates value of multi-path exploration in recursive latent spaces. | TEBDIL isolates the specific **excursion-and-return update transport operator** and tests its causal rescue minus degradation effect against linear state/update mixing (`M0`, `M1`). |
| **RecursiveMAS** ([2026](https://recursivemas.github.io/); [GitHub](https://github.com/RecursiveMAS/RecursiveMAS)) | Learned latent links connecting distinct multi-agent models across recurrent reasoning rounds. | Shares the high-level theme of cooperative recurrent computation in latent space. | RecursiveMAS relies on learned projections across multi-agent systems. TEBDIL operates **training-free** inside a **single shared-weight LoopLM**, exploiting identical coordinates across branches. |
| **Interlat — Enabling Agents to Communicate Entirely in Latent Space** ([ACL 2026](https://aclanthology.org/2026.acl-long.1248/)) | Replaces token-level inter-agent messages with continuous hidden-state communication. | Motivates direct hidden-state transfer (`M0` / `C0` controls). | Interlat transfers state representations directly. TEBDIL tests whether **evaluating the nonlinear transition $F_{\theta,t}$ at the visited neighborhood** contributes non-trivial curvature $C_t \neq 0$ beyond state sharing. |
| **STARS — Stabilizing Recurrent Dynamics** ([ICML 2026](https://proceedings.mlr.press/v306/yang26t.html)) | Stabilizes single-trajectory recurrent iteration to prevent representation drift or collapse at higher depth. | Informs our numerical stability invariants (RMS tracking, relative norm trust regions) and extra-step baseline (`S1`). | STARS stabilizes single-branch recurrence; TEBDIL tests cross-branch trajectory rescue when a single stable trajectory still converges to an incorrect answer. |
| **What Makes Recurrence Effective in Looped Language Models?** ([arXiv:2609.36636](https://arxiv.org/abs/2609.36636)) | Analyzes recurrent block placement, effective horizon, and history-state conditioning in LoopLMs. | Directly informs our choice of intervention step (`visit_step`) and update-mixing controls (`M1`). | Rather than architectural retraining or static history conditioning, TEBDIL performs online cross-trajectory update transport on a frozen checkpoint. |
| **GTS — Gaussian Thought Sampler** ([EMNLP 2026](https://aclanthology.org/2026.emnlp-main.341/)) | Learned sampler for drawing diverse, in-distribution latent trajectories. | Highlights the risk that naive Gaussian noise (`R0`) may go out-of-distribution if uncalibrated. | Even given diverse branches, TEBDIL tests whether excursion-and-return adds value beyond sampling (`P0`) or random norm-matched perturbation (`R0`). |
| **Open LoopLM Ecosystem** ([`rkstgr/LoopLM`](https://github.com/rkstgr/LoopLM)) | Clean reference implementations and experimentation utilities for looped transformer architectures. | Reference for clean stepwise unrolling and recurrent state inspection. | TEBDIL builds a formal `RecurrentAdapter` protocol supporting both exact Ouro weights and lightweight deterministic toy models for CI verification. |

---

## 3. Strict Claim Boundaries

1. **No claim of "reading model thoughts":** Hidden states $H_t \in \mathbb{R}^{B \times L \times d}$ are intermediate activation tensors of a recurrent computation, not human-interpretable mental states.
2. **No claim of novel multi-branch sampling:** Generating multiple trajectories via noise (`P0`) or averaging states (`M0`) is prior art and treated strictly as baseline control groups.
3. **Core falsifiable claim:** Any advantage claimed for TEBDIL must come from:
   - Comparing `E1`/`E2` against `S0`, `S1`, `P0`, `M0`, `M1`, `R0`, and `C0` under **explicitly accounted recurrent block evaluations (`num_recurrent_block_calls`) and FLOPs**.
   - Measuring **net paired accuracy** ($\mathrm{Acc}(A') - \mathrm{Acc}(A)$), reporting both **Conditional Rescue Rate** and **Degradation Rate**.
   - Separating **Oracle donor selection** (mechanism capacity upper bound) from **Non-oracle label-free selection** (deployable algorithm).
   - Quantifying the **nonlinear curvature residual** $C_t = V_t^{\mathrm{visit}} - [(1-\alpha)V_t^A + \alpha V_t^B]$ and testing whether removing $C_t$ (reducing to `M1`) eliminates any observed gain.
