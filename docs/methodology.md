# TEBDİL Methodology: Self-Directed Latent Relocation in Recurrent Language Models

## 1. Overview and Scientific Objective

**TEBDİL** investigates whether a looped recurrent language model (`ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`) can learn **when**, **where**, and **how** to relocate its intermediate hidden-state trajectory during inference to recover from unproductive latent attractors under controlled compute budgets.

Unlike prompt-level multi-agent frameworks or fixed-schedule latent interventions (`E1`), Self-Directed TEBDİL (`E3_ridge`, `E3_mlp`) introduces a lightweight, label-free inference-time meta-controller $\pi_\phi(z_t)$ that observes internal recurrent dynamics at decision step $t_{\mathrm{visit}} = 2$ and selects among three computational regimes:
1. **`CONTINUE`**: Proceed with standard recurrent refinement on branch $A$ without paying excursion evaluation cost.
2. **`EXCURSION(d, \alpha, \beta)`**: Temporarily relocate to the neighborhood of co-evolving donor branch $d \in \{B, C, \dots\}$ with step size $\alpha$, evaluate the shared 24-layer recurrent core $F_{\theta, t}$ at the visited state, and transport the nonlinear update vector back to branch $A$ with mixing weight $\beta$.
3. **`EXIT`**: Halt recurrent loop execution early at step $t$ (`early_exit_step = t`) and decode immediately, saving remaining recurrent core evaluations on already-converged trajectories.

---

## 2. Backbone Recurrent Architecture (`ByteDance/Ouro-1.4B-Thinking`)

`ByteDance/Ouro-1.4B-Thinking` applies a shared 24-layer Universal Transformer stack $F_{\theta, t}$ across $T = 4$ recurrent steps (`total_ut_steps = 4`, `hidden_size = 2048`, `num_hidden_layers = 24`, `vocab_size = 49152`). Each recurrent step passes through 96 RMSNorm operations (4 per layer in a Sandwich-Norm layout).

In `OuroRecurrentAdapter` ([`ouro_adapter.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/models/ouro_adapter.py)):
- Step 1 executes over the input prompt embeddings to produce $H_1 \in \mathbb{R}^{1 \times L \times 2048}$.
- At fork step $t_{\mathrm{fork}} = 1$, donor branches $d \in \{B, C\}$ are initialized via norm-matched, zero-mean Gaussian perturbations calibrated to relative RMS scale $\eta$ (`branch_noise_eta = 0.25`):
  $$H_1^{(d)} = H_1^{(A)} + \eta \cdot \frac{\|H_1^{(A)}\|_F}{\|\epsilon^{(d)}\|_F} \epsilon^{(d)}$$
- All branches advance through $F_{\theta, 2}$ to reach step $t = 2$, where the meta-controller observes label-free trajectory features $z_2$.

---

## 3. Excursion Operator and Nonlinear Curvature Decomposition

When an excursion to donor $d$ with parameters $(\alpha, \beta)$ is executed at step $t$ ([`excursions.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/core/excursions.py)):

1. **Scoped & Subspace-Projected Direction:**
   Let $\Delta_t^{(d)} = (H_t^{(d)} - H_t^{(A)}) \odot M_{\mathrm{scope}}$.
   - Under `token_scope = "all"`, $M_{\mathrm{scope}} = \mathbf{1}$.
   - Under `token_scope = "last_token"` (`E1_last_token`), $M_{\mathrm{scope}}$ is non-zero only on the final sequence position.
   - Under `subspace_fraction = \kappa < 1` (`E1_subspace`), $\Delta_t^{(d)}$ is restricted to the top-$\kappa d_{\mathrm{model}}$ hidden dimensions ranked by donor-recipient divergence magnitude.

2. **Trust-Region Bounded Visit State:**
   With relative Frobenius bound $\rho_{\max}$ (`max_relative_jump = 0.35`, or `0.15` for `E1_norm_tight`):
   $$\alpha_{\mathrm{eff}} = \min\!\left(\alpha,\; \frac{\rho_{\max} \|H_t^{(A)}\|_F}{\|\Delta_t^{(d)}\|_F + \varepsilon}\right), \qquad H_t^{\mathrm{visit}} = H_t^{(A)} + \alpha_{\mathrm{eff}} \Delta_t^{(d)}$$

3. **Genuine Recurrent Evaluation & Update Transport:**
   $$V_t^{(A)} = F_{\theta, t}(H_t^{(A)}) - H_t^{(A)}, \qquad V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$$
   - Standard return (`E1`, `E2`, `E3`):
     $$H_{t+1}^{(A')} = H_t^{(A)} + (1 - \beta) V_t^{(A)} + \beta V_t^{\mathrm{visit}}$$
   - Gate-modulated return (`E1_gated`): weights per-token return strength $\beta_i \propto H(p_t^{(A)}(i)) / \overline{H(p_t^{(A)})}$ so tokens with higher predictive entropy incorporate a stronger donor update.

4. **Nonlinear Curvature Residual $C_t$:**
   Given donor velocity $V_t^{(d)} = F_{\theta, t}(H_t^{(d)}) - H_t^{(d)}$, the linear velocity interpolation is $L_t = (1 - \alpha_{\mathrm{eff}}) V_t^{(A)} + \alpha_{\mathrm{eff}} V_t^{(d)}$, and the exact nonlinear excursion residual is:
   $$C_t = V_t^{\mathrm{visit}} - L_t$$
   When $\alpha = 0$ or $\beta = 0$ or $H_t^{(A)} = H_t^{(d)}$, $H_{t+1}^{(A')} \equiv H_{t+1}^{(A)}$ and $\|C_t\|_F = 0$.

---

## 4. Label-Free Controller Feature Space $z_t$

At decision step $t$, [`extract_state_features`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/controller/features.py) computes strictly label-free features from branch $A$ and candidate donors $\{d\}$:
- **Recipient Uncertainty & Confidence:** Last-token logit entropy $H(p_t^{(A)})$, top-1 probability $p_{1, t}^{(A)}$, top-1 vs top-2 margin $m_t^{(A)}$, and Ouro native early-exit gate probability $g_t^{(A)} = \sigma(\text{Linear}(H_t^{(A)}))$.
- **Temporal Stability Dynamics:** Entropy change $\Delta H_t^{(A)} = H_t^{(A)} - H_{t-1}^{(A)}$, margin change $\Delta m_t^{(A)}$, last-token cosine velocity $\cos(H_t^{(A)}, H_{t-1}^{(A)})$, relative update norm $\|H_t^{(A)} - H_{t-1}^{(A)}\|_F / \|H_{t-1}^{(A)}\|_F$, top-1 token stability $\mathbb{I}[\arg\max p_t^{(A)} = \arg\max p_{t-1}^{(A)}]$, and step-to-step predictive Jensen-Shannon divergence $\mathrm{JSD}(p_t^{(A)} \| p_{t-1}^{(A)})$.
- **Pairwise Donor Compatibility $z_{t, d}$:** Full-dimensional Frobenius distance $\|H_t^{(d)} - H_t^{(A)}\|_F$, relative distance, last-token cosine similarity, donor entropy/margin/top-1 confidence, top-1 disagreement $\mathbb{I}[\arg\max p_t^{(d)} \ne \arg\max p_t^{(A)}]$, and predictive $\mathrm{JSD}(p_t^{(A)} \| p_t^{(d)})$.

An AST-level static and runtime test ([`test_no_label_leakage.py`](file:///Users/alperkartkaya/Documents/tebdil/tests/test_no_label_leakage.py)) verifies that no ground-truth label (`gold_answer`, `is_correct`, `validator`, etc.) is ever imported or referenced in [`action_space.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/controller/action_space.py), [`features.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/controller/features.py), [`learned_policy.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/controller/learned_policy.py), or [`adaptive_policy.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/core/adaptive_policy.py).

---

## 5. Counterfactual Rollout Collection and Controller Training

In [`counterfactual.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/controller/counterfactual.py):
1. **Strict Split Hygiene:** Counterfactual trajectories are collected on disjoint `train` (`n=18`) and `validation` (`n=18`) problem templates across seeds `[42, 43]`. Held-out `test` (`n=18` problems $\times$ 2 seeds $= 36$ paired instances) is never touched during counterfactual generation, feature normalization, or hyperparameter selection.
2. **Complete Action Rollouts:** For each decision state $z_2$, every candidate action $a \in \mathcal{A}$ (`CONTINUE`, `EXIT`, and `EXCURSION(d, \alpha, \beta)` across donors $d \in \{B, C\}$ and $(\alpha, \beta) \in \{(0.35, 0.50), (0.50, 0.50), (0.50, 0.75)\}$) is genuinely unrolled through `Ouro-1.4B-Thinking` to termination.
3. **Compute-Aware Counterfactual Utility:**
   $$U(z_t, a) = \mathbb{I}[\text{correct}(a)] + \lambda_{\mathrm{lp}} \log p_{\mathrm{gold}}(a) - \lambda_{\mathrm{cost}} \cdot \text{extra\_calls}(a) - \lambda_{\mathrm{deg}} \mathbb{I}[\text{correct}(\mathrm{CONTINUE}) \land \neg\text{correct}(a)]$$
   Relative advantage is defined against `CONTINUE`:
   $$A(z_t, a) = U(z_t, a) - U(z_t, \mathrm{CONTINUE})$$
4. **Anchored State-Action Featurization:**
   `featurize_state_action(z_t, CONTINUE)` is identically $\mathbf{0} \in \mathbb{R}^{28}$, and feature RMS scaling preserves $0.0$ without mean-shifting, guaranteeing $\hat{A}_\phi(z_t, \mathrm{CONTINUE}) \equiv 0.0$ for both linear Ridge (`E3_ridge`) and zero-bias 2-layer ReLU MLP (`E3_mlp`) controllers.
5. **Validation Threshold Calibration:**
   Excursion threshold $\tau_{\mathrm{exc}}$ and early-exit threshold $\tau_{\mathrm{exit}}$ are calibrated via quantile grid search over predicted advantages on the validation split to maximize net counterfactual utility.

---

## 6. Verification & Reproducibility Audit (`tebdil.audit`)

The standalone CLI verifier (`python -m tebdil.audit --run-id <run_id> [--replay-problem-id <pid>]`, implemented in [`audit.py`](file:///Users/alperkartkaya/Documents/tebdil/src/tebdil/audit.py)) performs six automated integrity checks:
1. **Provenance Verification:** Confirms `model_provenance.json` SHA-256 weight digests match the local `ByteDance/Ouro-1.4B-Thinking` snapshot.
2. **Split Disjointness:** Verifies zero overlap in `problem_id` or prompt text across `train`, `validation`, and `test`.
3. **Trace Tensor & Hash Integrity:** Audits every record in `traces.jsonl` for finite norms, valid recurrent call counts, and distinct SHA-256 hidden-state hashes (`receiving_state_hash != visit_state_hash`, `normal_update_hash != visit_update_hash`, `receiving_state_hash != post_receiving_state_hash`) on every active excursion.
4. **Recomputed Paired Bootstrap Statistics:** Independently recomputes task accuracy, rescue/degradation counts, paired accuracy differences vs `S0`, and 10,000-sample paired bootstrap 95% confidence intervals from raw `traces.jsonl`, asserting exact agreement with `summary.json`.
5. **Controller Checkpoint Integrity:** Verifies finite weights and zero leakage in `tebdil_ridge_v1.json` and `tebdil_mlp_v1.json`.
6. **Live Hardware Replay:** Optionally re-executes a problem ID on `ByteDance/Ouro-1.4B-Thinking` and verifies exact step-by-step hidden-state RMS and output equivalence against the recorded trace.
