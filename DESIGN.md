# Design

## Source of truth
Active, 2026-10-30. Public static observatory in `site/`; Python research engine remains independent. Evidence: README, architecture/methodology/limitations, current HTML/JS, recorded summaries and token ledgers, pinned Ouro source. Streamlit remains an advanced local tool, not a live backend for Pages.

## Brand
A restrained research instrument: precise, curious, transparent. No artificial latent landscapes, promotional metrics, glow, or decorative inference controls.

## Product goals
Explain visit–compute–return in 30 seconds; inspect a real recorded decision and paired outcome; expose provenance and uncertainty. Do not launch GPU inference or add experiments from the website.

## Personas and jobs
New visitors understand the invention. Researchers inspect actual steps, methods, costs, configs, and source records.

## Information architecture
One continuous selectable sequence: Problem, Mechanism, Decision, Experiment, Evidence. Keep current recording selection across sections. Technical detail lives in native expandable sections. Preserve existing anchor/query links where practical through explicit aliases.

## Design principles
Mechanism before metrics. One primary message per panel. Recorded facts and illustrative diagrams are different layers. Negative results remain visible. No absent telemetry becomes a fabricated zero.

## Visual language
Preserve the original RecursiveMAS-inspired academic project-page identity, restored at the user's request (2026-10-30). Reference: https://recursivemas.github.io/ and TEBDIL's own pre-refactor `fd074b8:site/index.html`. White `#ffffff`, slate surface `#f8fafc`, line `#e2e8f0`, ink `#0f172a`, secondary text `#475569`. Indigo `#4338ca` and sky accents define the centered project title; a soft tinted grid hero and dark resource pills retain the existing identity. No dark observatory redesign.

Recipient indigo `#4338ca`, donor dark amber `#92400e`, excursion violet `#7c3aed`; darker tones keep diagram text readable on light surfaces. Names and line patterns duplicate color meaning. SVG uses the same CSS tokens as the interface. Platform sans and monospace avoid remote font dependencies. 4/8/16/24/40px spacing, rounded 12–18px panels, subtle diagram shadows. Centered academic introduction, left-aligned research content and responsive split comparisons. Retain the five-step flow and scientific corrections; do not restore unsupported old claims or decorative controls. Reduced motion respected.

## Components
Native links/buttons/selects/details, selectable journey, schematic SVG, per-step telemetry, paired outputs, results table, provenance downloads. CSS owns tokens; one JS state object owns selection; generated manifest owns dataset labels/statistics.

## Accessibility
WCAG 2.2 AA target, not certification. Skip link, landmarks, logical headings, labeled selects, native controls, visible focus, 44px touch targets. Step buttons use pressed state; tables have captions. SVG has text alternatives. Never color-only outcomes.

## Responsive behavior
Desktop split content at 900px; mobile one column, wrapped navigation, scrollable labeled tables/diagrams only where needed. No hidden body overflow masking layout defects.

## Interaction states
No JavaScript: explanatory content plus source links remain usable. Trace loading: explicit status. Failure: error and retry, no stale metrics. Missing snapshots/fields: unavailable, not borrowed from another token/method. Changing datasets cancels stale render work.

## Content voice
Short, literal English. Name purpose first, method ID second. Say recorded experiment, learned controller with heuristic gate, and exploratory evidence. Never label recorded playback live inference or imply probabilities from uncalibrated action scores.

## Implementation constraints
Existing HTML/CSS/vanilla JS, Python exporter, no new framework/runtime dependencies. Local relative assets work under `/tebdil/`. Small startup manifest and lazy trace shards. Test scientific invariants, export consistency, browser flows, mobile, keyboard, error handling, and base path. Build must fail rather than invent records.

## Open questions
- [ ] Larger preregistered independent benchmark and genuinely compute-matched research conclusions remain future research, not an MVP UI obligation.
