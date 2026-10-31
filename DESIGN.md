# Design

## Source of truth
Active, 2026-10-31. Public static observatory in `site/`; Python research engine remains independent. Evidence: README, architecture/methodology/limitations, current HTML/JS, recorded summaries and token ledgers, pinned Ouro source. Streamlit remains an advanced local tool, not a live backend for Pages.

## Brand
A restrained research instrument: precise, curious, transparent. No artificial latent landscapes, promotional metrics, glow, or decorative inference controls.

## Product goals
Explain visit–compute–return in 30 seconds; inspect a real recorded decision and paired outcome; expose provenance and uncertainty. Do not launch GPU inference or add experiments from the website.

## Personas and jobs
New visitors understand the invention. Researchers inspect actual steps, methods, costs, configs, and source records.

## Information architecture
The actual `fd074b8` academic page is the baseline, not the rejected five-panel rewrite. Preserve the original sections and anchors in order: top, demo, abstract, architecture, theory, patterns, results, scaling, case-study, examples, roadmap, bibtex. Preserve the floating contents menu, centered publication header, four metrics, four-column experiment stage, adjacent PCA/telemetry panels, dark output terminal, research tables, heatmap, dark case cards, trace explorer, and citation.

## Design principles
Correct the original page rather than reinterpret it. The original inline CSS is frozen by a SHA-256 regression test. `static/style.css` contains only targeted accessibility, loading, and responsive repairs. Keep negative results visible and distinguish measured evidence from schematic diagrams. Never turn absent telemetry into a fabricated zero.

## Visual language
Keep the original RecursiveMAS-inspired layout, fonts, spacing, indigo/sky hero, resource pills, light research sections, and dark terminal/case studies. Original Inter and JetBrains Mono fonts, Bulma, icons, and MathJax are pinned and hosted locally. Asset sources and licenses are documented in `reports/original_asset_sources.md`. No new palette or page-wide reskin.

## Components
Existing HTML/CSS/vanilla JS. One recorded-data selection drives the original stage, exact snapshot telemetry, final terminal output, and provenance downloads. All eight recording sets are selectable; methods and token/step options derive from their actual traces. Original case-card buttons and explorer rows open the matching recorded example. The replay also offers the same three cases as concise walkthrough shortcuts. Method selects and terminal chips share one context-preserving handler: no token/step clamping or neighboring-snapshot substitution. Missing comparisons are explicit; a token index denotes generation position, not necessarily the same prefix. Selected-step explanation and expandable decision/compute details reuse the original narrative box.

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
