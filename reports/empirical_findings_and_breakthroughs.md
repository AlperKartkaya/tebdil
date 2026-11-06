# Empirical Findings and Breakthroughs of Latent Excursion (E2)
**Project TEBDİL: Counterfactual Latent Dynamics on Looped Language Models (`ByteDance/Ouro-1.4B-Thinking`)**
*Date: November 2026*  
*Hardware Target: Google Colab Pro NVIDIA L4 (24GB VRAM) / A100 Tensor Core GPUs*

---

## Executive Summary

Autoregressive reasoning models (specifically Chain-of-Thought / RL-tuned reasoning models like `ByteDance/Ouro-1.4B-Thinking`) suffer from two prominent cognitive pathologies:
1. **Reasoning Rigidity & Deceptive Trap Entrapment:** When a prompt structurally resembles a well-known benchmark puzzle but contains an inverted condition or negation, the model's learned prior overrides the literal text, trapping the CoT generation in circular backtracking loops until token budgets exhaust.
2. **Arithmetic Hallucination & Deadlock Cascades:** An early off-by-one or carry addition mistake creates an impossible downstream state (e.g., non-integer division of discrete objects leading to $56.666\dots$), plunging the model into infinite rationalization loops.

**TEBDİL's Latent Excursion Mechanism ($E_2$)** intervenes directly on the recurrent hidden state manifold ($H_t$) using competitive logit donor vectors ($w_{y_j} - w_{y_1}$) transported in pre-RMSNorm residual space. This report documents key empirical breakthroughs where $E_2$ successfully rescued the model from reasoning traps, broken deadlocks, and catastrophic arithmetic cascades that completely defeated the unperturbed baseline ($S_0$), while simultaneously providing massive token compression on complex reasoning trajectories.

---

## Full 32-Job Deceptive Traps Benchmark (3000 Token Budget, 3 Workers, NVIDIA L4)

A complete 32-job evaluation (16 tasks $\times$ 2 methods, $S_0$ vs $E_2$) was conducted under an expanded **3000-token generation budget** across 3 concurrent GPU workers using PyTorch memory fragmentation optimization (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`):

| Task ID | Family | $S_0$ Answer (Status, Tokens) | $E_2$ Answer (Status, Tokens, Exc) | Gold | Outcome / Significance |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `trap_hotel_empty_rooms` | `reasoning_rigidity` | `None` (`budget_exhausted`, 3000t) | **`0` (`correct`, 940t, 26e)** | `0` | **Major E2 Breakthrough** (-68.7% tokens, -66.4% compute) |
| `trap_rope_burn_timing` | `rate_and_combinatorics` | `None` (`budget_exhausted`, 3000t) | **`45` (`correct`, 2815t, 69e)** | `45` | **Major E2 Breakthrough** (Solved non-uniform deadlock) |
| `trap_gsm_school_inflation` | `gsm_noop_distractor` | `130` (`correct`, 1722t) | **`130` (`correct`, 680t, 18e)** | `130` | **Huge CoT Compression** (-1042t / -60.5% tokens, 2.1x faster) |
| `trap_base10_reversal` | `reasoning_rigidity` | `111` (`correct`, 2482t) | **`111` (`correct`, 1832t, 27e)** | `111` | **CoT Compression** (-650t / -26.2% tokens, 64.5s faster) |
| `trap_clock_strikes` | `rate_and_combinatorics` | `18` (`correct`, 1592t) | **`18` (`correct`, 1487t, 23e)** | `18` | **CoT Compression** (-105t savings, avoided fencepost trap 15) |
| `trap_lily_pad_halving` | `cognitive_illusion` | `47` (`correct`, 429t) | **`47` (`correct`, 382t, 11e)** | `47` | **CoT Compression** (-47t savings, avoided linear 24) |
| `trap_gsm_apple_crates` | `gsm_noop_distractor` | `300` (`correct`, 341t) | **`300` (`correct`, 323t, 6e)** | `300` | Both correct, avoided shelf distractor |
| `trap_speed_harmonic_mean` | `cognitive_illusion` | `40` (`correct`, 546t) | **`40` (`correct`, 535t, 4e)** | `40` | Both correct, avoided arithmetic mean trap 45 |
| `trap_gsm_kiwi_size` | `gsm_noop_distractor` | `190` (`correct`, 778t) | **`190` (`correct`, 774t, 12e)** | `190` | Both correct, avoided fruit size distractor |
| `trap_widget_machines` | `rate_and_combinatorics` | `5` (`correct`, 395t) | **`5` (`correct`, 415t, 9e)** | `5` | Both correct, avoided scaling trap 100 |
| `trap_gsm_bakery_shapes` | `gsm_noop_distractor` | `100` (`correct`, 312t) | **`100` (`correct`, 393t, 10e)** | `100` | Both correct, avoided cookie shape distractor |
| `trap_couple_handshakes` | `rate_and_combinatorics` | `12` (`correct`, 518t) | **`12` (`correct`, 782t, 16e)** | `12` | Both correct, avoided handshake trap 15 |
| `trap_bat_ball_glove` | `cognitive_illusion` | `None` (`invalid_format`, 309t) | `None` (`invalid_format`, 309t, 2e) | `5` | Both deduced 5 correctly in CoT, regex missed text |
| `trap_monty_hall_4doors` | `cognitive_illusion` | `75` (`correct`, 1301t) | `None` (`budget_exhausted`, 3000t, 41e) | `75` | S0 converged to 75%; E2 debated between 50% and 75% |
| `trap_infertile_rabbits` | `reasoning_rigidity` | `None` (`budget_exhausted`, 3000t) | `None` (`budget_exhausted`, 3000t, 39e) | `1` | Both trapped in Fibonacci recurrence derivations |
| `trap_simple_jug_measure` | `reasoning_rigidity` | `None` (`budget_exhausted`, 3000t) | `None` (`budget_exhausted`, 3000t, 36e) | `3` | Both trapped in Die Hard water jug pouring sequences |

### Benchmark Aggregate Metrics

* **Accuracy:**
  - **Baseline ($S_0$):** 11 / 16 (**68.8%**)
  - **Latent Excursion ($E_2$):** 12 / 16 (**75.0%**) $\longrightarrow$ **+6.2% absolute gain**
* **Total Token Consumption:**
  - **Baseline ($S_0$):** 22,725 tokens
  - **Latent Excursion ($E_2$):** 20,667 tokens $\longrightarrow$ **-2,058 tokens saved overall (-9.1% net token savings)**
* **Zero Failure on Simple Distractors:**
  Both methods resisted superficial distractors on GSM-NoOp tasks (`trap_gsm_apple_crates`, `trap_gsm_bakery_shapes`, `trap_gsm_kiwi_size`).
* **Massive Circuit-Breaking on Rigidity Traps:**
  $E_2$ broke the infinite loops on `trap_hotel_empty_rooms` and `trap_rope_burn_timing` where $S_0$ burned all 3000 tokens with zero output.

---

## Breakthrough Case 1: Hilbert Hotel Rigidity Escape (`trap_hotel_empty_rooms`)

### 1.1 Problem Specification
* **Source:** KAIST / AITRICS ReasoningTrap Suite & Inversion Logic
* **Family:** `reasoning_rigidity` / Habitual Pattern Overriding
* **Task ID:** `trap_hotel_empty_rooms`
* **Prompt:**
  ```text
  A grand hotel has an infinite number of rooms, but only 10 rooms are currently occupied and all other rooms are completely empty. A new guest arrives. What is the minimum number of currently registered guests that must be relocated to another room in order to accommodate the new guest? Be concise. Output only the integer result.
  Answer: 
  ```
* **Gold Target Answer:** `0`
* **Trap Lure Answers:** `1`, `10`, `11` (arising from reflexively applying Hilbert's Paradox shift: "move guest $k \to k+1$")

### 1.2 Quantitative Comparison

| Metric | Baseline ($S_0$) at 1500 Tokens | Baseline ($S_0$) at 3000 Tokens | Latent Excursion ($E_2$) | Delta / Effect ($E_2$ vs $S_0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Status** | `budget_exhausted` | `budget_exhausted` | **`correct` (SOLVED)** | **Complete Recovery from Trap** |
| **Extracted Answer** | `None` | `None` | **`0`** | Exact Gold Match |
| **Generated Tokens** | `1500` | `3000` | **`940`** | **-2060 tokens (-68.7% savings)** |
| **Recurrent Block Calls** | `6004` calls | `12004` calls | **`4032` calls** | **-7972 calls (-66.4% compute)** |
| **Wall Clock Time (L4)** | `363.9s` | `601.7s` | **`210.1s`** | **Almost 3x faster termination** |
| **Excursion Count** | `0` | `0` | **`26`** | Active latent interventions |

### 1.3 Behavioral Breakdown & Mechanics
* **$S_0$ Failure Mode:**  
  The prompt mentions "grand hotel", "infinite number of rooms", and "accommodate new guest". The model's associative prior fires the classic Hilbert Hotel schema, insisting that guests must shift $1 \to 2, 2 \to 3 \dots$. However, reading *"only 10 rooms are currently occupied and all other rooms are completely empty"* triggers a logical contradiction: empty rooms are immediately available!  
  Under standard greedy generation ($S_0$), the attention heads fixate in a cyclic deadlock:
  > *"Wait, if there are empty rooms, why shift anyone? But Hilbert's paradox requires shifting. Does the hotel rule enforce shifting? But it asks for the minimum relocations... If room 11 is empty, just put the guest there! That's 0 moves. But wait, what if the manager insists on moving? No, minimum number... let me reconsider... wait, if all rooms are empty except 10, then 0. But why is it an infinite hotel problem? Maybe 1? But why move 1?..."*  
  Even when doubled to **3000 tokens** (12,004 recurrent calls, 10 minutes of GPU time), $S_0$ was completely incapable of breaking this loop.
* **$E_2$ Recovery Mechanism:**  
  At token positions where top-2 logit entropy spiked due to the competition between candidate "0" and candidate "1", $E_2$'s competitive donor vector pushed the recurrent state along the $(w_{\text{"0"}} - w_{\text{"1"}})$ manifold.  
  The counterfactual update ($V_t = \tilde{F}(H_t) - H_t$) transported in pre-RMSNorm space evicted the hidden state from the cyclic attractor basin. The model concluded decisively:
  > *"Since only 10 rooms are occupied and there are infinitely many empty rooms, the new guest can simply be assigned to any unoccupied room (e.g., room 11) without relocating any existing guest. Therefore, the minimum number of guests relocated is 0."*  
  It outputted `0`, closed the thinking block, and terminated cleanly at token 940.

---

## Breakthrough Case 2: Non-Uniform Rope Burning Puzzle (`trap_rope_burn_timing`)

### 2.1 Problem Specification
* **Source:** Classic Lateral Thinking / Non-uniform Rate Invariance
* **Family:** `rate_and_combinatorics` / Halving Interpolation Deadlock
* **Task ID:** `trap_rope_burn_timing`
* **Prompt:**
  ```text
  You have two identical ropes. Each rope burns completely in 60 minutes from end to end, but the ropes burn non-uniformly (different sections burn at different speeds). By lighting Rope 1 at both ends and Rope 2 at one end simultaneously, and lighting the second end of Rope 2 the moment Rope 1 finishes burning, you measure an exact duration. How many total minutes of time elapse from the start until Rope 2 burns out completely? Be concise. Output only the integer result.
  Answer: 
  ```
* **Gold Target Answer:** `45`
* **Trap Lure Answers:** `60`, `30`, `15`

### 2.2 Quantitative Comparison

| Metric | Baseline ($S_0$) | Latent Excursion ($E_2$) | Delta / Effect |
| :--- | :--- | :--- | :--- |
| **Status** | `budget_exhausted` (FAIL) | **`correct` (SOLVED)** | **Breakthrough Solution** |
| **Extracted Answer** | `None` (deadlock loop) | **`45`** | Exact Gold Match |
| **Generated Tokens** | `3000` (budget exhausted) | **`2815`** | Solved before cutoff |
| **Recurrent Block Calls** | `12004` calls | **`12200` calls** | Full constraint verification |
| **Wall Clock Time (L4)** | `465.3s` | **`448.9s`** | Complete solution |
| **Excursion Count** | `0` | **`69`** | Extensive saccadic steering |

### 2.3 Behavioral Breakdown & Mechanics
* **$S_0$ Failure Mode:**  
  $S_0$ quickly recognized that Rope 1 lit at both ends burns in 30 minutes. However, calculating the state of Rope 2 caused severe CoT paralysis:
  > *"Rope 2 has burned for 30 minutes from one end. But because it burns non-uniformly, does 30 minutes mean half of the rope burned? No! Non-uniform means length doesn't equal time! But wait, how much time is left in Rope 2? 30 minutes of burn time! But now both ends are lit! If 30 minutes of burn time is left, does lighting both ends take 15 minutes? Yes, 30 / 2 = 15 minutes! Wait, but is that mathematically rigorous if the burn rate is non-uniform? Let me prove it by integrating the density function $\int v(x) dx$... Wait, if the density is variable, what if the two flames meet at a point with infinite density?..."*  
  $S_0$ entered an infinite loop trying to write and verify differential equations for non-uniform flame propagation, burning all 3000 tokens without an answer.
* **$E_2$ Recovery Mechanism:**  
  With 69 latent excursions injected during hesitation spikes, $E_2$ repeatedly redirected the model away from continuous density modeling back toward discrete invariant time accounting:
  > *"The total burn time capacity of Rope 2 is 60 minutes. After 30 minutes of single-end burning, 30 minutes of burn capacity remains. When both ends of the remaining section are lit, the burning rate doubles, consuming the remaining 30 minutes of capacity in exactly 15 minutes. Total elapsed time = 30 + 15 = 45 minutes."*  
  It outputted `\boxed{45}` and terminated successfully.

---

## Breakthrough Case 3: Arithmetic Carry Correction & Repeating Decimal Escape (`lit_dev_gsm_001`)

### 3.1 Problem Specification
* **Source:** Literature Reasoning Suite / GSM Multi-Step Inventory Tracking
* **Family:** Multi-Step State Tracking with Discrete Constraints
* **Task ID:** `lit_dev_gsm_001`
* **Prompt:**
  ```text
  A distribution warehouse starts Monday morning with 100 crates.
  Monday: receives 40 crates, ships out 25.
  Tuesday: receives 50 crates, ships out 30.
  Wednesday: receives 60 crates, ships out 30.
  Thursday: receives 35 crates, ships out 20.
  Friday: receives 0 crates, ships out one-third of all crates currently in the warehouse.
  How many crates remain in the warehouse at the end of Friday?
  ```
* **Gold Target Answer:** `120`
* **Arithmetic Ground Truth Steps:**
  - End of Wednesday: $100 + (40-25) + (50-30) + (60-30) = 165$ crates.
  - Thursday morning: $165 + 35 = \mathbf{200}$ crates.
  - Thursday evening: $200 - 20 = \mathbf{180}$ crates.
  - Friday: ships out $\frac{1}{3} \times 180 = 60$ crates $\implies 180 - 60 = \mathbf{120}$ crates remaining.

### 3.2 Quantitative Comparison

| Metric | Baseline ($S_0$) | Latent Excursion ($E_2$) | Delta / Effect |
| :--- | :--- | :--- | :--- |
| **Status** | `budget_exhausted` (FAIL) | **`correct` (SOLVED)** | **Carry Error Corrected** |
| **Extracted Answer** | `None` (trapped in $56.666\dots$ loop) | **`120`** | Exact Gold Match |
| **Generated Tokens** | `1088` (budget exhausted) | **`963`** | **-125 tokens (-11.5% savings)** |
| **Wall Clock Time (L4)** | `147.3s` | **`139.7s`** | **Faster termination** |

### 3.3 Behavioral Breakdown & Mechanics
* **$S_0$ Arithmetic Hallucination:**  
  On Thursday morning, $S_0$ performed the addition $165 + 35$. The model dropped the carry bit ($5+5=10, 6+3=9 \implies 190$), hallucinating:
  > *"Thursday morning: 165 + 35 = 190 crates. Thursday evening: ships 20, leaving 170 crates."*  
  On Friday, the model attempted $\frac{1}{3} \times 170 = \mathbf{56.666\dots}$ crates. Because crates are discrete objects, the model entered a repetitive loop:
  > *"170 divided by 3 is 56.666... But since you can't ship a fraction of a crate, do we round? But the problem says one-third... Maybe 56.666? But crates are integers..."*  
  It exhausted its token budget debating fractional crates.
* **$E_2$ Saccadic Latent Correction:**  
  At tokens $k=509$ and $k=544$, where hesitation arose during the arithmetic sequence, $E_2$'s competitive donor evaluated the alternative token subspace. The excursion update shifted the hidden state toward the correct carry digit.  
  The model revised its addition to $165 + 35 = 200$, subtracted 20 to get $180$, and cleanly computed $\frac{180}{3} = 60 \implies 180 - 60 = \mathbf{120}$.

---

## Breakthrough Case 4: CoT Distractor Pruning & Token Compression

On problems where both $S_0$ and $E_2$ arrive at the correct answer, $E_2$ systematically prunes redundant deliberation and ignores deceptive distractors:

### 4.1 GSM Distractor Filtering (`trap_gsm_school_inflation`)
* **Prompt:** Liam buys 20 notebooks ($5 each) and 15 pens ($2 each) today. The problem adds a distractor: *"Due to general economic inflation, prices of stationery were 10 percent lower last year."* How many dollars does Liam pay today?
* **Gold Answer:** $20 \times 5 + 15 \times 2 = 100 + 30 = \mathbf{130}$.
* **Comparison:**
  - **$S_0$:** Generated **1722 tokens** (322.1s, 6892 recurrent calls). It spent 1200+ tokens calculating last year's prices, adjusting for inflation backwards and forwards, before finally noticing the question asked for *today's* price.
  - **$E_2$:** Generated only **680 tokens** (150.6s, 2906 recurrent calls, 18 excursions). It pruned the inflation distractor on step 2, achieving a **60.5% token reduction (1042 tokens saved) and running 2.1x faster**!

### 4.2 Base Inversion Trajectory Compression (`trap_base10_reversal`)
* **Prompt:** Smallest 3-digit $abc$ such that $abc = bca$ in base 10.
* **Gold Answer:** $\mathbf{111}$.
* **Comparison:**
  - **$S_0$:** Generated **2482 tokens** (497.4s, 9932 calls). It derived all modular congruences $100a + 10b + c = 100b + 10c + a \implies 99a - 90b - 9c = 0$, verified all combinations multiple times.
  - **$E_2$:** Generated **1832 tokens** (432.9s, 7825 calls, 27 excursions). It reached the exact same correct answer $111$ with **650 fewer tokens (-26.2% savings)** and 64.5s faster.

---

## Reproducibility Protocol & Specifications

To reproduce every finding reported above, execute using the exact pinned hyperparameters below:

### Model & Weights
* **Model ID:** `ByteDance/Ouro-1.4B-Thinking`
* **Pinned Revision:** `846cff8ee9f1ec52c9ae25c3fe8f3523fcebe188`
* **Precision:** `torch.bfloat16`
* **Attention Kernel:** `sdpa` (Scaled Dot-Product Attention)

### $E_2$ Excursion Configuration
```python
ExcursionConfig(
    alpha=0.65,                         # Blending rate: H_t + alpha * (F_tilde - H_t)
    beta=0.65,                          # Return projection factor
    visit_step=2,                       # Recurrent sub-step to launch excursion
    token_scope="last_token",           # Apply perturbation strictly to active token
    donor_strategy="competitive_logit", # Perturbation direction: (w_y2 - w_y1)
    update_space="pre_norm",            # Compute updates before OuroRMSNorm
    excursion_window_tokens=1,          # Window of persistent excursion influence
    state_shaping=False,                # No artificial reward shaping
    use_heuristic_gates=False           # Driven purely by intrinsic logit hesitation
)
```

### Exact Execution Command on NVIDIA L4 (Google Colab Pro)
```bash
python3 scripts/run_deceptive_parallel.py \
    --config configs/deceptive_traps_v1_tasks.json \
    --workers 3 \
    --max-new-tokens 3000 \
    --seed 42 \
    --output reports/colab_l4_deceptive_traps_3000_32jobs.json
```
