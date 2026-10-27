# Experimental Methodology

## 1. Cooperative Latent Excursion Operator

Given a pretrained looped language model with shared recurrent transition $F_{\theta, t}: \mathbb{R}^{B \times L \times d} \to \mathbb{R}^{B \times L \times d}$ across $t \in \{0, \dots, T-1\}$:

1. **Initialization ($t=0$):**
   Token ids are embedded to $H_0 = \mathrm{embed\_tokens}(\mathrm{input\_ids})$.
2. **Shared Recurrent Prefix ($t=0 \to 1$):**
   Both reference branch $A$ and donor branch(es) $B$ share the first recurrent transition $H_1 = F_{\theta, 0}(H_0)$.
3. **Branch Forking ($t=1$):**
   - $H_1^A = H_1$ (unperturbed reference trajectory).
   - $H_1^B = H_1 + \epsilon$, where $\epsilon$ is a seeded Gaussian perturbation restricted to `token_scope` (`full`, `last_token`, or `selected`) and normalized so $\|\epsilon\|_F = \eta \|H_1 \odot M_{\mathrm{scope}}\|_F$.
4. **Excursion-and-Return ($t = t_{\mathrm{visit}}$):**
   - Direction vector: $D_t = (H_t^B - H_t^A) \odot M_{\mathrm{scope}}$.
   - Trust-region bounded visit offset:
     $$\Delta_t = \alpha_{\mathrm{eff}} D_t, \qquad \alpha_{\mathrm{eff}} = \alpha \cdot \min\left(1, \frac{\rho_{\max} \|H_t^A\|_F}{\|\alpha D_t\|_F + \varepsilon}\right)$$
   - Visited state: $H_t^{\mathrm{visit}} = H_t^A + \Delta_t$.
   - Local update evaluations:
     $$V_t^A = F_{\theta, t}(H_t^A) - H_t^A, \quad V_t^B = F_{\theta, t}(H_t^B) - H_t^B, \quad V_t^{\mathrm{visit}} = F_{\theta, t}(H_t^{\mathrm{visit}}) - H_t^{\mathrm{visit}}$$
   - Return transport to trajectory $A$:
     $$H_{t+1}^{A'} = H_t^A + (1 - \beta) V_t^A + \beta V_t^{\mathrm{visit}}$$
5. **Nonlinear Curvature Decomposition:**
   The excursion update decomposes into a linear update mixture plus a nonlinear curvature residual:
   $$V_t^{\mathrm{visit}} = \underbrace{(1 - \alpha_{\mathrm{eff}}) V_t^A + \alpha_{\mathrm{eff}} V_t^B}_{V_t^{\mathrm{mix}}\text{ (Method M1)}} + C_t$$
   Therefore:
   $$H_{t+1}^{A'} = H_{t+1}^{\mathrm{M1}} + \beta C_t$$
   Comparing `E1` against `M1` (`E1_no_curv`) and `E1_rand_curv` (where $C_t$ is replaced by a random vector of identical Frobenius norm $\|C_t\|_F$) directly tests whether the magnitude and direction of the nonlinear curvature $C_t$ contribute causally to trajectory recovery.

## 2. Control Groups and Compute Accounting

Every call to $F_{\theta, t}$ is counted in `recurrent_block_calls` and converted to estimated FLOPs:
- `S0`: 1 branch, $T=4$ steps (4 calls/token).
- `S1`: 1 branch, $T=7$ steps (7 calls/token, matching 2-branch compute).
- `P0`: 2 independent branches, label-free selection via maximum sequence logit margin (7 calls/token).
- `P1`: 2 independent branches, **oracle best-of-2** using ground-truth validator post-hoc (upper bound only).
- `M0`: Direct hidden-state averaging prior to $F_{\theta, t}$ (7 calls/token).
- `M1`: Linear update-vector mixing without evaluating $F_{\theta, t}(H_t^{\mathrm{visit}})$ (7 calls/token).
- `R0`: Norm-matched random perturbation to $H_t^A$ at $t_{\mathrm{visit}}$ (7 calls/token).
- `C0`: Jump to $H_t^{\mathrm{visit}}$ and continue without returning to $H_t^A$ (7 calls/token).
- `E0`: Excursion-and-return along a random direction of norm $\|D_t\|_F$ (8 calls/token).
- `E1`: Fixed-schedule cooperative latent excursion toward donor $B$ (8 calls/token).
- `E2`: Adaptive label-free excursion triggered only when branch $A$ exhibits high predictive uncertainty or stagnation and donor $B$ exhibits higher logit margin within the divergence trust window ($7 + \mathbb{I}(\text{visit})$ calls/token).
