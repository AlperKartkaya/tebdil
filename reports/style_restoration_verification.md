# Original visual identity restoration — 2026-10-30

User correction: retain the RecursiveMAS-inspired TEBDIL style rather than the dark observatory theme.

## Scope

Reused TEBDIL's own `fd074b8:site/index.html` visual vocabulary, checked against https://recursivemas.github.io/: white/slate surfaces, indigo–sky centered academic heading, softly gridded hero, dark resource pills, rounded diagrams. No upstream source was copied. No dependencies added. `DESIGN.md` now records this identity as a constraint.

The five-step flow, lazy recording selection, controller labels, scientific limitations and artifacts remain unchanged. SVG diagrams now use CSS theme tokens rather than duplicate dark-theme colors. Fixed missing spaces around hidden heading line breaks on mobile.

## Fresh checks

- Before edits: 18 site release tests passed.
- After edits: 68 non-model tests passed; one real-model test deliberately deselected (presentation-only scope).
- JavaScript syntax, TypeScript, changed-file Ruff E4/E7/E9/F checks, export verification for all eight runs and whitespace checks passed.
- Local `/tebdil/` browser checks at 1280×900 and 390×844: inspected restored problem, mechanism, experiment and evidence views; no horizontal page overflow or browser errors observed.
- Switched all three mechanism diagrams. Selected fixed excursion E1 and changed the recording from fresh-template evaluation (972 traces) to curvature controls (216 traces); displayed data and browser-verified hashes changed accordingly.
- Keyboard ArrowLeft moved Experiment → Decision, End moved to Evidence with focus/selection synchronized. Evidence displayed 18 methods in the fresh-template recording.
- Confirmed projected trajectory colors resolve from CSS tokens on the light background. No metric or research-artifact edits in this change.

## Limits

No new model execution, full screen-reader audit or WCAG certification. Browser screenshots are local verification; deployment is verified separately through the Pages workflow and public asset checksums.
