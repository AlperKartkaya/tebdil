# Methodological and Technical Limitations

1. **Inference-Time Perturbation vs Semantic Reasoning Diversity:**
   Forking trajectory $B$ via norm-bounded Gaussian perturbation ($\epsilon$) creates a distinct hidden-state trajectory in $\mathbb{R}^{B \times L \times d}$, but does not guarantee a semantically distinct high-level reasoning strategy. When $A$ and $B$ fail in the same way, excursion cannot recover the answer without an alternative branching mechanism.
2. **Trade-Off Between Rescue and Degradation:**
   Unconditional excursion (`E1`) can perturb trajectories where branch $A$ was already on a valid path, causing non-zero conditional degradation ($\Pr(A' \text{ wrong} \mid A \text{ correct}) > 0$). Reporting rescue rate in isolation is scientifically invalid; net paired accuracy difference ($\mathrm{Acc}(A') - \mathrm{Acc}(A)$), problem-cluster uncertainty and discordant outcomes belong together. Repeated seeds are not independent problems.
3. **Extra Recurrent Evaluation Cost:**
   Each executed excursion at $t_{\mathrm{visit}}$ requires one additional recurrent block evaluation $F_{\theta, t}(H_t^{\mathrm{visit}})$ (increasing cost from 7 to 8 recurrent calls for a 2-branch, 4-step schedule with fork at step 1). This schedule illustration is not a verified cost for every recorded arm. Historical parallel selection costs have ledger gaps; no strict compute-matched conclusion follows.
4. **Stateless `use_cache=False` Execution:**
   To guarantee zero KV-cache contamination across forked and perturbed hidden states, `OuroRecurrentAdapter` runs with `use_cache=False`. Current tests cover short verifiable prompts and answers. Efficient long-generation cache support would require a separately validated branch-isolated cache design; it is not implemented or benchmarked here.
5. **2D PCA Projection Caveat:**
   PCA trajectory plots in the Observatory capture only the top 2 principal components of last-token hidden states ($d = 2048$). Visual separation or proximity in 2D is purely illustrative and does not constitute proof of dynamical attractor basins.
6. **No Claim of Internal Intent Reading or Alignment Guarantees:**
   Intervening on intermediate hidden states measures functional sensitivity of the recurrent transition $F_{\theta, t}$ on verifiable tasks. It does not read "true model thoughts" or provide general AI safety/alignment guarantees.
