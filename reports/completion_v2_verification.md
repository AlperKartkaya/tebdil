# Measurement v2 verification

Status: implementation, offline audit and local release checks complete; public
deployment verification pending.
The baseline difficulty pilot has **not** started. No reserved test or controller
is evaluated here. Original page design, generation source and historical scores
stay unchanged.

## Change and scope

The [v2 contract](../docs/completed_answer_v2_protocol.md) and implementation were
committed at `b315d22`. The v1 parser already accepts a bare integer box; the known
failure was its surrounding `\(...\)` delimiter. The separately versioned v2
parser handles balanced integer math wrappers and rejects malformed thought
regions, unfinished responses and competing explicit answers. It receives no gold.
Offline scoring reuses the existing status definitions, without editing records.

Fifty-three synthetic/CLI tests cover supported and unsupported syntax, signs,
thought isolation, ambiguity, EOS, budget exhaustion, unchanged v1 behavior,
immutable source artifacts and overwrite refusal. This bounds supported syntax;
it is not proof of universal mathematical-language understanding.

## Separate post-hoc audit

[Raw audit](ouro_completion_v2_audit.json), SHA-256
`fa32bda597ab45f8156a138e04d140043675419fc72ccdae2008f843211a5836`.
It contains six **already-observed** development outputs (one S0 and five fixed
methods on one problem), each linked to its unchanged source report and text hash.
V1 and v2 fields are explicitly separate. Only B_only changes status under v2:
`invalid_format` → `correct`, recognizing its existing `\(\boxed{29}\).` answer.
The original B_only record remains invalid under v1. No output, token, cost or old
score was regenerated or overwritten. **Zero new model completions** were executed
for this diagnostic. It is not six independent problems or a model improvement.

The amendment was motivated by the known format miss, so this audit is post-hoc.
Future development inference must explicitly name v2, use the frozen contract and
retain incorrect, malformed and unfinished attempts. Model execution provenance
is still enforced by the existing generator/ledger validator, not the text parser.

## Release gate

**232 tests passed**, including the available cached real-model integration.
Scoped Ruff, JavaScript syntax, TypeScript, static export and diff checks passed.
The local browser matrix passed eight recordings, three demos, new raw download,
keyboard/retry/checksum checks and 320–1440 px layouts without console, network or
overflow failures. The original layout/CSS/section identities remain protected by
release tests. Historical manifest entries and run data are unchanged.

Before baseline inference, verify the public download/hash and original browser
flows after Pages deployment. Source/input hashes bind the offline audit; no
clean-clone or cross-device model rerun is claimed by parser tests.
