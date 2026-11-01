# Completed-answer measurement v2

## Scope and implementation boundary

Fix measurement before the development baseline pilot. Add a versioned parser,
test its positive and negative boundaries, and publish a separate offline audit
of already-recorded outputs. Keep v1 parser, callers, scores and artifacts intact.
Only after tests, documentation and public Pages verification may the pilot begin.
No new dependency, controller, model completion experiment or interface redesign
in this step. Existing real-checkpoint integration tests remain validation checks.

This amendment follows a **known development format failure**: v1 accepts
`\boxed{29}` but rejects `\(\boxed{29}\).`. The audit of known outputs is post-hoc,
not a new benchmark, rescue or improvement in model reasoning. Use v2 prospectively
only after this implementation is frozen; identify both versions explicitly.

## Frozen extraction contract: completed_final_integer_v2

- Parse only responses known to have reached EOS, never budget-exhausted output.
  The existing generation/evidence validator remains responsible for verifying
  actual token IDs, EOS/budget stops and physical recurrent calls.
- Permit no thought region, or exactly one complete leading `<think>...</think>`
  region. Ignore its contents; reject malformed, nested, repeated or stray markers.
- Remove at most one trailing EOS text marker and Markdown bold markers. Reject
  embedded/repeated EOS markers. The parser never receives the gold answer.
- Accept ASCII signed integers as the whole final region, a single terminal integer
  `\boxed{...}` with optional preceding prose, or one terminal Answer/Result label
  (optionally prefixed Final). Allow whitespace inside a box and a terminal period.
- Accept balanced, non-nested `$...$`, `$$...$$`, `\(...\)` and `\[...\]` around
  **only an integer or integer box**. Do not evaluate arithmetic, fractions,
  decimals, units or equations, and do not take the first/last arbitrary number.
- Reject multiple boxes or multiple explicit Answer/Result labels, even if their
  values agree. A label followed by conflicting prose/another answer is invalid.
  Balanced worked mathematics may precede one unambiguous final answer.
- Unsupported/malformed responses stay `invalid_format`; unfinished responses
  stay `budget_exhausted`; recognized final integers are compared to gold offline.
  Every attempt remains in the denominator. Correctness, completion and formatting
  are separate outcomes. No gold-based repair, stopping or output selection.

## Verification and evidence

Synthetic tests cover delimiter pairs, signs, whitespace, thought isolation,
conflicting answers, truncation, malformed delimiters and old-v1 behavior.
An offline audit references the immutable native S0 and five-method reports by
SHA-256 and record identity. It preserves original fields alongside v2 fields,
records source hashes, and creates a new exclusive output; it never overwrites
or silently replaces the original scores. Known-output recognition is a regression
check, not proof of coverage for every possible mathematical response.

The baseline pilot must separately freeze task selection, prompt, generation
budget, reuse of the prior S0 output, costs and difficulty decisions before any
new model observations. Reserved-test inference remains prohibited at this step.
