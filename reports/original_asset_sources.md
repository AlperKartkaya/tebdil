# Original page runtime assets

Restored locally for the `fd074b8` academic page structure. The old page loaded these families from public CDNs; local copies preserve the original appearance while removing runtime network dependencies. Academicons was not restored because the original markup contains no Academicons class.

| Asset | Version/source | Local files |
|---|---|---|
| Bulma | 0.9.4, `https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css` | `site/static/vendor/bulma/bulma.min.css`, MIT `LICENSE` |
| Font Awesome Free | 6.5.1, `https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/` | `fontawesome/fontawesome.min.css`, `fontawesome/solid.min.css`, `webfonts/fa-solid-900.woff2`, upstream `LICENSE.txt` |

| Inter | Google Fonts API family requested at weights 400–800; returned `s/inter/v20/` variable WOFF2 Latin and Latin Extended subsets | `fonts/inter-latin*.woff2`, SIL OFL `INTER-OFL.txt` |
| JetBrains Mono | Google Fonts API family requested at weights 400–700; returned `s/jetbrainsmono/v24/` variable WOFF2 Latin and Latin Extended subsets | `fonts/jetbrains-mono-latin*.woff2`, SIL OFL `JETBRAINS-MONO-OFL.txt` |
| MathJax | 3.2.2 combined TeX-to-SVG component, `https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js` | `mathjax/tex-svg.js`, Apache-2.0 `LICENSE` |

The vendored `solid.min.css` differs from upstream only by removing its unused `.ttf` fallback URL; the included and browser-verified WOFF2 remains the sole font source, so the local stylesheet has no dangling runtime URL.

Font API request: `https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap`. Retrieved with a WOFF2-capable browser user agent on 2026-10-30. Only Latin/Latin Extended subsets are retained; these include Turkish text used by TEBDİL.

## SHA-256

```text
10ecf152e6235bd8d3fdeb88918272b9b6bab957750069db9af3012cc3383f7e  site/static/vendor/bulma/LICENSE
ad3a5d3b41d7042369ade00772eead0763e9839d79568fb91ad612b2734bcfef  site/static/vendor/bulma/bulma.min.css
0aa8f86525273b2efa4f40f4272a188e187704252170e979dc06879adf68d43c  site/static/vendor/fontawesome/LICENSE.txt
8eb5b44ce5971a501e5a17954c464b80aba0c4604675b5603a7f456459af88a1  site/static/vendor/fontawesome/fontawesome.min.css
ad139e0384a6c2eccd18968a8572bac5e3efd214050a7552e941c597791d9697  site/static/vendor/fontawesome/solid.min.css
5b9321a4298cfeb6b34354164a1c3afc3db114569984c502b9b35d988fd58c57  site/static/vendor/fonts/INTER-OFL.txt
b2fe5e8987594e9ffd1d2ca52a2f5d73eb8335243893c5d6254b5ad69269591d  site/static/vendor/fonts/JETBRAINS-MONO-OFL.txt
a2928c9151a41423c00eead6ad10c1047996493c21d5c3a867de9766e22722aa  site/static/vendor/fonts/fonts.css
a28eb6d3ccb534ae0c94ca999371df024aab60b08c3c8a5720ee9e32fa0faaa2  site/static/vendor/fonts/inter-latin-ext.woff2
c940764593d0fe5d596be327ca7558855e018039fb78509aa21921fd3644c3e4  site/static/vendor/fonts/inter-latin.woff2
9c38cb2d0d2d93c1ee6e21fa78db76f13ea7e15e15cc64214c7ca89b6aaa35c4  site/static/vendor/fonts/jetbrains-mono-latin-ext.woff2
2c32b9b3ee358c119e210f6f5195f9bd34894d78a785ff2e95d60e718e400af4  site/static/vendor/fonts/jetbrains-mono-latin.woff2
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30  site/static/vendor/mathjax/LICENSE
d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882  site/static/vendor/mathjax/tex-svg.js
9fc85f3a4544ab0d570c7f8f9bbb88db8d92c359b2707580ea8b07c75673eae2  site/static/vendor/webfonts/fa-solid-900.woff2
```

## HTML paths

Load before the original inline page styles:

```html
<link rel="stylesheet" href="static/vendor/bulma/bulma.min.css">
<link rel="stylesheet" href="static/vendor/fontawesome/fontawesome.min.css">
<link rel="stylesheet" href="static/vendor/fontawesome/solid.min.css">
<link rel="stylesheet" href="static/vendor/fonts/fonts.css">
```

Keep the original `window.MathJax` configuration before:

```html
<script src="static/vendor/mathjax/tex-svg.js" defer></script>
```

Exact font binaries:

- Inter Latin Extended: `https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa25L7W0Q5n-wU.woff2`
- Inter Latin: `https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7W0Q5nw.woff2`
- JetBrains Mono Latin Extended: `https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbv2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKwBNntkaToggR7BYRbKPx7cwgknk-6nFg.woff2`
- JetBrains Mono Latin: `https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbv2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKwBNntkaToggR7BYRbKPxDcwgknk-4.woff2`

The browser smoke test loaded all four local stylesheets, the solid icon WOFF2, both font families, and MathJax TeX-to-SVG successfully. No runtime asset request failed. A second offline smoke rendered the 171 unique TeX expressions extracted from the original page with zero MathJax errors or unprocessed expressions; the combined `tex-svg.js` therefore covers every extension used by that page without additional component downloads.
