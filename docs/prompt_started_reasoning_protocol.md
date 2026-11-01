# Prompt-started reasoning recording, v1

## Scope and freeze

First sequential point after the reasoning-stall design correction. Freeze this
protocol, new implementation and tests before the real smoke. Do not change the
hash-bound v1/v2 generators, parsers, protocols, old outputs or task partitions.
This gate verifies explicit thinking-mode input and recording, not a new task set,
stall monitor, controller, intervention utility or superiority over thinking=False.

## Prompt and execution

Use the exact cached ByteDance/Ouro-1.4B-Thinking revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, tokenizer and BF16 MPS contract from the
prior native smoke. Four recurrent depths, forced last-step readout, eager
attention, uncached greedy native generation and 2048 new tokens. Stop only at EOS
or budget. The generation function accepts the question, not gold or task metadata.

The new encoder explicitly sets `enable_thinking=True`. Require the actual decoded
input to end with `<|im_start|>assistant\n<think>\n`. Record full prompt token IDs
and prompt text separately from generated token IDs and generated text. The
opening tag belongs to the prompt; never insert a fabricated opening tag into
raw generated text. The flag prepends the tag, not a guarantee of useful or faithful
reasoning. Old encoders remain False; this gate does not switch historical runners.

## Reasoning/final boundary

The new recording contract assumes the prompt has opened reasoning. A generated
`</think>` closes it. Preserve the exact generated text before and after that tag
as `reasoning_text` and `final_text`, with whitespace intact and terminal EOS
removed only from these derived fields. The original generated text is untouched.
An empty reasoning section is possible and is not evidence of a long CoT.

- One closing tag and no generated opening tag: `closed`.
- No closing tag before budget: `open`; no final answer can be assessed.
- EOS before a closing tag: `missing_close`; completed generation, invalid format.
- Generated opening/repeated closing tags or inconsistent EOS: `malformed`.
- A closed section followed by an unfinished final answer stays budget-exhausted.

Keep generation completion, reasoning boundary and answer validity separate. For
this technical integer smoke only, apply the unchanged v2 integer grammar to the
final section, offline, then reuse existing outcome definitions. Never parse an
integer from reasoning or repair tags according to gold. Other task types will
need their own tested final-answer verifier; this is not their task contract.

## Real smoke and evidence

Select only the already-observed `completion_dev_000` by ID, before observations.
Run one new native S0 completion with the exact existing question, not a new
reasoning benchmark or a paired thinking-mode effectiveness comparison. None of
the sixty reserved test questions, donors or controllers execute. All stopping
outcomes are retained; a malformed or budget result is not silently retried.

Reuse the native generator and ledger validator. Independently observe each
forward's prefix length/hash and greedy token proposal, and physical recurrent
boundaries; reconcile these against actual returned tokens and native ledgers.
Remove hooks on failure too. Store exclusive raw JSON and an append-only sibling
journal, source/checkpoint identities before and after, actual calls, weighted
recurrent token positions and instrumented wall time. No old cost is counted anew.

Tests must cover prompt provenance, opening-in-prompt vs opening-in-output,
malformed tags/EOS, unfinished sections, gold-free extraction, actual native
mini-model tokens/calls, failure cleanup and overwrite refusal. Synthetic fixtures
are not checkpoint evidence. Only after tests, real-smoke inspection, documentation
and public Pages verification pass may the task/verifier point begin. No confidence
interval, stall recovery or reasoning-benefit claim follows from this smoke.
