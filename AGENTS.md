# TEBDİL — Computation & Architecture Guidelines

Computation hold lifted by explicit user instruction on 2026-11-05 following core
architectural fixes:

1. **Branched KV-Cache (`UniversalTransformerCache`) & Single-Token Readout:**
   `OuroRecurrentAdapter` supports `prefill_kv_cache`, `initialize_cached_step`, and
   `commit_cached_step` so excursion-modified K/V states persist across generated tokens
   without $O(L^2)$ full-prefix recomputation.
2. **Pre-RMSNorm Residual Update Transport (`update_space="pre_norm"`):**
   Excursion updates $V_t = \tilde{F}(H_t) - H_t$ are computed in pre-RMSNorm residual
   space and normalized back onto the native `OuroRMSNorm` manifold after combination.
3. **Scoped Trust-Region & Competitive Logit Donors (`competitive_logit`):**
   Perturbations and trust-region clipping operate on the active `last_token` scope
   using competitive unembedding directions ($w_{y_j} - w_{y_1}$) rather than full-prefix
   isotropic Gaussian noise.
4. **Unshaped Counterfactual Controller & Multi-Token Excursion Windows:**
   Reward shaping (`state_shaping`) and hard-coded heuristic gating (`use_heuristic_gates`)
   are disabled by default; excursions persist over `excursion_window_tokens`.
5. **Execution Target:**
   Heavy Ouro-1.4B-Thinking runs target Google Colab Pro **NVIDIA L4 (24GB)** or **A100**
   GPUs via `scripts/run_colab_l4.py`. Local unit/mini-model verification is unrestricted.

## User verdict required for approach changes (2026-11-08)

Before implementing a new method variant, changing the research approach, or
launching an experiment for such an improvement, present its hypothesis, intended
benefit, mechanism, comparison and risks, then wait for the user's verdict.
Do not treat a bridge connection or general autonomy instructions as approval for
approach changes. Read-only analysis, existing-test verification, and explicitly
authorized infrastructure maintenance may continue. The user approved review
and commit/push of the updated Colab bridge cell separately.
