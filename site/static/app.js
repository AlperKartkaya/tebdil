/// <reference path="./schema.d.ts" />
/* Recorded evidence only. Schematic layout never represents measured latent coordinates. */
(() => {
  "use strict";
  /** @type {Observatory} */
  const DATA = window.TEBDIL_DATA;
  const hasData = DATA?.schema_version === 2 && Object.keys(DATA.runs || {}).length > 0;
  /** @overload @param {"runSelect"|"caseSelect"|"methodSelect"|"tokenSelect"|"stepSelect"} id @returns {HTMLSelectElement} */
  /** @overload @param {"previousView"|"nextView"|"retryLoad"} id @returns {HTMLButtonElement} */
  /** @overload @param {"downloadTrace"|"downloadRun"|"downloadConfig"} id @returns {HTMLAnchorElement} */
  /** @overload @param {string} id @returns {HTMLElement} */
  /** @param {string} id */
  function $(id) {
    const element = document.getElementById(id);
    if (!element) throw new Error(`Missing observatory element: ${id}`);
    return element;
  }
  /** @param {string} selector */
  const all = (selector) => /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll(selector));
  const VIEWS = ["problem", "mechanism", "decision", "experiment", "evidence"];
  const VIEW_NAMES = ["The problem", "The mechanism", "The decision", "The experiment", "The evidence"];
  const METHODS = {
    S0: ["Ordinary recurrence", "No intervention"], S1: ["Extra recurrent depth", "Fixed schedule"],
    P0: ["Independent branches", "Confidence-based selection"], P1: ["Oracle best branch", "Uses gold outcomes; upper bound only"],
    M0: ["Direct state mixing", "Fixed schedule"], M1: ["Linear update mixing", "Fixed schedule; no visited-state computation"],
    E1: ["Fixed excursion", "Fixed schedule"], E2: ["Adaptive excursion", "Heuristic adaptive rules; not learned"],
    E3_mlp: ["Learned relocation", "Offline-trained MLP + hand-set margin/entropy gates"],
    E3_mlp_ungated: ["Ungated learned relocation", "Offline-trained MLP; heuristic eligibility gates disabled"],
    E3_ridge: ["Learned linear control", "Offline-trained ridge + hand-set margin/entropy gates"],
    E3_no_exit: ["Learned control without exit", "Offline-trained scorer + heuristic gates; EXIT disabled"],
    E3_mlp_unshaped: ["Unshaped learned control", "Offline-trained MLP + heuristic gates; no utility shaping"],
    E3_t2_1donor: ["Budgeted learned control", "Offline-trained MLP + heuristic gates; step 2, one donor"],
    EXIT_2: ["Fixed early exit", "Fixed schedule; stop at step 2"],
    RAND_CTRL: ["Random control", "Random actions; not learned"], H_budget: ["Budget heuristic", "Hand-set adaptive rules; not learned"],
    G_gate_only: ["Gate-only controller", "Hand-set margin/entropy EXIT and EXCURSION rules; no learned scorer"],
    R0: ["Norm-matched random noise", "Negative control"], E0: ["Random-direction excursion", "Negative control"],
    C0: ["Visit without return", "Visited state replaces receiving trajectory"],
    E1_no_curv: ["Excursion without residual", "Curvature ablation"], E1_rand_curv: ["Randomized residual", "Negative control"],
    D_nearest: ["Nearest donor", "Heuristic donor selection"], D_farthest: ["Farthest donor", "Heuristic donor selection"],
    D_confidence: ["Highest-margin donor", "Heuristic donor selection"], D_random: ["Random donor", "Random donor selection"],
    D_oracle: ["Oracle donor", "Uses gold outcomes; upper bound only"], D_learned: ["Learned donor", "Offline-trained donor scorer + hand-set margin/entropy gates"],
    E1_last_token: ["Last-token transfer", "Fixed scoped excursion"], E1_norm_tight: ["Tight trust region", "Fixed clipped excursion"],
    E1_coord_top50: ["Coordinate-restricted transfer", "Fixed masked excursion"], E1_subspace: ["Coordinate-limited transfer", "Fixed correction-coordinate restriction; not a learned subspace"],
    E1_multi_visit: ["Multiple visits", "Fixed schedule"], E1_gated: ["Native-gate weighted transfer", "Fixed schedule; return scaled by Ouro gate, not a trained controller"],
  };
  const name = (m) => METHODS[m]?.[0] || m;
  const policy = (m) => METHODS[m]?.[1] || "See execution manifest";
  const esc = (v) => String(v ?? "Unavailable").replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c]);
  const num = (v, digits = 3) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "Unavailable";
  const pct = (v) => typeof v === "number" ? `${(100 * v).toFixed(1)}%` : "Unavailable";
  const pp = (v) => typeof v === "number" ? `${v > 0 ? "+" : ""}${(100 * v).toFixed(1)}` : "Unavailable";
  const options = (select, rows, value) => {
    select.replaceChildren(...rows.map(([key, label]) => new Option(label, String(key))));
    select.value = String(value);
    select.disabled = !rows.length;
  };
  const params = new URLSearchParams(location.search);
  const oldRuns = {fresh_test_phase7_8:"phase7_8_fresh_test", test_phase7_8:"phase7_8_controller_test", test_phase4:"phase4_adaptive", validation_phase3:"phase3_ablations", validation_phase5:"phase5_donor", validation_phase6:"phase6_selective"};
  const aliases = {"live-stage":"experiment", "demo":"experiment", "simulator":"experiment", "architecture":"mechanism", "theory":"mechanism", "controls":"mechanism", "abstract":"problem", "results":"evidence", "main-results":"evidence", "grid":"evidence", "patterns":"mechanism", "scaling":"evidence", "case-study":"experiment", "examples":"experiment", "reproduction":"evidence", "case-studies":"experiment", "trace-explorer":"experiment", "cases":"experiment", "roadmap":"evidence", "bibtex":"evidence", "cite":"evidence"};
  const workflow = {understand:"problem", explore:"experiment", reproduce:"evidence"};
  const hashView = () => aliases[location.hash.slice(1)] || (VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : null);
  const state = {
    view: params.get("view") || hashView() || workflow[params.get("workflow")] || "problem",
    run: oldRuns[params.get("run")] || params.get("run") || "phase7_8_fresh_test",
    problem: params.get("problem"), seed: params.get("seed"), method: params.get("method") || "E3_mlp",
    token: Number(params.get("token")) || 0, step: Number(params.get("step")) || 2, mechanism: "excursion",
  };
  /** @type {RecordedTrace[]} */
  let rows = [];
  /** @type {Map<string, Record<string, RecordedTrace>>} */
  let pairs = new Map();
  let loadVersion = 0, downloadUrl;
  const cache = new Map();
  const run = () => DATA.runs[state.run];
  const trace = () => pairs.get(`${state.problem}|${state.seed}`)?.[state.method];
  const baseline = () => pairs.get(`${state.problem}|${state.seed}`)?.S0;
  const token = () => trace()?.token_ledger?.[state.token];
  const snapshots = () => token()?.step_snapshots || (state.token === 0 ? trace()?.step_snapshots || [] : []);
  const snapshot = () => snapshots().find((s) => s.step === state.step);
  const telemetry = () => snapshots().find((s) => s.excursion_telemetry?.step === state.step)?.excursion_telemetry;

  function syncUrl() {
    const query = new URLSearchParams();
    ["view", "run", "problem", "seed", "method", "token", "step"].forEach((key) => {
      if (state[key] !== null && state[key] !== undefined) query.set(key, state[key]);
    });
    history.replaceState(null, "", `${location.pathname}?${query}`);
  }

  function setView(view, focus = false) {
    state.view = VIEWS.includes(view) ? view : "problem";
    VIEWS.forEach((v) => {
      const selected = v === state.view;
      $(`tab-${v}`).setAttribute("aria-selected", String(selected));
      $(`tab-${v}`).tabIndex = selected ? 0 : -1;
      $(`view-${v}`).hidden = !selected;
    });
    if (focus) $(`tab-${state.view}`).focus();
    const index = VIEWS.indexOf(state.view);
    $("previousView").disabled = index === 0;
    $("nextView").disabled = index === VIEWS.length - 1;
    $("nextView").textContent = index === VIEWS.length - 1 ? "End of journey" : `Next: ${VIEW_NAMES[index + 1]}`;
    $("journeyPosition").textContent = `${index + 1} of ${VIEWS.length}`;
    $("dataControls").hidden = index < 2;
    $("tokenControls").hidden = !["decision", "experiment"].includes(state.view) || !trace();
    const caseLabel = $("caseSelect").closest("label");
    if (caseLabel instanceof HTMLElement) caseLabel.hidden = state.view === "evidence";
    if (["decision", "experiment"].includes(state.view)) {
      if (rows.length) { syncMethods(); renderRecording(); }
      else loadRun();
    }
    if (state.view === "evidence") {
      renderEvidence();
      if (hasData && !rows.length) $("loadStatus").textContent = "Recorded summary · choose a method, or open its table row to inspect the underlying traces.";
    }
    syncUrl();
  }

  async function loadRun() {
    if (!hasData) return;
    const version = ++loadVersion;
    const selectedRun = state.run;
    const recording = run();
    rows = []; pairs = new Map();
    $("loadStatus").textContent = `Loading ${recording.label.toLowerCase()}… Recorded data, not a new model run.`;
    $("retryLoad").hidden = true;
    all("#caseSelect, #methodSelect, #tokenSelect, #stepSelect").forEach((select) => { if (select instanceof HTMLSelectElement) select.disabled = true; });
    all(".data-dependent").forEach((el) => { el.hidden = true; });
    $("tokenControls").hidden = true;
    try {
      let recorded = cache.get(selectedRun);
      if (!recorded) {
        const response = await fetch(recording.trace_url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const bytes = await response.arrayBuffer();
        if (!globalThis.crypto?.subtle) throw new Error("Secure browser context required to verify recording checksums");
        const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("");
        if (hash !== recording.trace_sha256) throw new Error("Recording checksum mismatch");
        recorded = JSON.parse(new TextDecoder().decode(bytes));
        if (!Array.isArray(recorded) || recorded.length !== recording.record_count) throw new Error("Recording count mismatch");
        cache.set(selectedRun, recorded);
      }
      if (version !== loadVersion) return;
      rows = recorded;
      rows.forEach((r) => {
        const key = `${r.problem_id}|${r.seed}`;
        if (!pairs.has(key)) pairs.set(key, {});
        pairs.get(key)[r.method] = r;
      });
      let key = `${state.problem}|${state.seed}`;
      if (!pairs.has(key)) {
        const candidate = rows.find((r) => r.method === state.method && r.token_ledger?.some((t) => t.controller_action_type === "EXCURSION")) || rows[0];
        state.problem = candidate.problem_id; state.seed = String(candidate.seed); key = `${state.problem}|${state.seed}`;
      }
      options($("caseSelect"), [...pairs].map(([k, values]) => [k, `${values.S0.problem_id} · seed ${values.S0.seed}`]), key);
      syncMethods();
      $("loadStatus").textContent = `${recording.record_count.toLocaleString()} recorded traces · ${recording.num_problems} distinct problems · ${recording.seeds.length} seed(s) · SHA-256 verified in this browser.`;
      renderRecording();
    } catch (error) {
      if (version !== loadVersion) return;
      $("loadStatus").textContent = `Recording unavailable: ${error.message}. Check your connection, then retry. No results have been simulated.`;
      $("retryLoad").hidden = false;
    }
  }

  function availableMethod(methods) {
    if (methods[state.method]) return state.method;
    return ["E3_mlp", "E1", "S0"].find((method) => methods[method]) || Object.keys(methods)[0];
  }

  function syncMethods() {
    const selected = pairs.get(`${state.problem}|${state.seed}`) || {};
    const available = Object.keys(selected);
    state.method = availableMethod(selected);
    options($("methodSelect"), available.map((m) => [m, `${name(m)} · ${m}`]), state.method);
    const ledger = trace()?.token_ledger || [];
    state.token = Math.max(0, Math.min(ledger.length - 1, state.token));
    options($("tokenSelect"), ledger.length ? ledger.map((t, i) => [i, `Token ${i + 1}: ${JSON.stringify(t.decoded_token ?? t.token_id)}`]) : [[0, "Final output · per-token data unavailable"]], state.token);
    $("tokenSelect").disabled = !ledger.length;
    syncSteps();
  }

  function syncSteps() {
    const steps = snapshots().map((s) => s.step);
    if (!steps.includes(state.step)) state.step = steps[0];
    options($("stepSelect"), steps.map((s) => [s, `Step ${s}`]), state.step);
  }

  function decisionAtStep() {
    for (const s of snapshots()) {
      const d = s.policy_decision;
      if (!d) continue;
      const step = d.state_features?.step ?? s.excursion_telemetry?.step ?? (s.step - (d.action_type === "EXIT" ? 0 : 1));
      if (step === state.step) return d;
    }
    return null;
  }

  function renderDecision() {
    const r = trace(), t = token(), d = decisionAtStep(), tel = telemetry();
    $("policyKind").textContent = policy(r.method);
    const controlActions = {M0: "STATE MIX", M1: "UPDATE MIX", C0: "VISIT WITHOUT RETURN", R0: "RANDOM NOISE"};
    const action = d?.action_type || (tel ? controlActions[r.method] || "EXCURSION" : null);
    all("[data-action]").forEach((el) => { el.dataset.selected = String(el.dataset.action === action); });
    const actionLabel = action || "No decision recorded at this step";
    $("decisionReadout").innerHTML = `<p><strong>${esc(actionLabel)}</strong> · ${t ? `token ${state.token + 1}` : "final-output snapshot; token identity unavailable"}, recurrent step ${esc(state.step)}</p><p class="muted">${esc(d?.reason || (tel ? "Scheduled intervention with recorded transition telemetry." : "The token summary below is not a decision at every recurrent step."))}</p><p class="small">Token-level summary: ${esc(t?.controller_action_type)} · donor ${esc(t?.selected_donor)} · α ${num(t?.effective_alpha, 2)} · β ${num(t?.effective_beta, 2)}</p>`;
    $("decisionDetails").textContent = JSON.stringify({selected_step_decision: d, outgoing_excursion: tel || null, token_summary: {action: t?.controller_action_type, reason: t?.policy_reason, action_scores: t?.action_scores, rejected_actions: t?.rejected_actions}}, null, 2);
  }

  function ledgerMatches(r) {
    if (r.derived_parallel_selection || ["P0", "P1"].includes(r.method)) return false; // Reference budget omits separately decoded selection candidates.
    const ledger = r.token_ledger;
    return !!ledger?.length && ledger.reduce((sum, t) => sum + t.recurrent_block_calls, 0) === r.recurrent_block_calls;
  }

  function output(r) {
    if (!r) return "<p>Paired baseline unavailable.</p>";
    return `<p class="output-answer">${esc(r.extracted_answer ?? "No valid answer")}</p><div class="output-meta"><span class="outcome ${r.is_correct ? "correct" : "incorrect"}">${r.is_correct ? "Correct" : "Incorrect"}</span><span>${esc(r.recurrent_block_calls)} calls${ledgerMatches(r) ? "" : " · ledger unverified"}</span><span>${num(r.wall_time_ms, 0)} ms ${["P0", "P1"].includes(r.method) ? "reference timing, not total" : "recorded"}</span></div><p class="small muted" style="margin:14px 0 0">Decoded: ${esc(JSON.stringify(r.decoded_text))}</p>`;
  }

  function mechanismCopy(method) {
    const controls = {
      E0: {stages: ["Receiving A", "Visit in random direction", "Compute F(visit) − visit", "Weighted update returns to A"], note: "Random-direction excursion: this visit does not move toward the donor state.", residualMeasured: true},
      M0: {stages: ["Receiving A + donor B", "Mix hidden states", "F(mixed state)", "Continue from mix"], note: "Direct state mixing: no visited update returns to the original A.", residualMeasured: false},
      M1: {stages: ["F(A) and F(B)", "Mix update vectors", "Apply linear update to A"], note: "No transition is evaluated at a visited state; the zero residual field is a placeholder.", residualMeasured: false},
      C0: {stages: ["Receiving A", "Visit toward B", "F(visit)", "Continue from visit"], note: "Visit-without-return control: the visited state replaces the receiving anchor.", residualMeasured: false},
      R0: {stages: ["Receiving A", "Norm-matched noise", "Perturbed normal update"], note: "Random-noise control, not a latent excursion.", residualMeasured: false},
      E1_no_curv: {stages: ["Receiving A", "Compute visited response", "Remove nonlinear residual", "Linear update returns to A"], note: "Residual ablation: visit computation is measured but its nonlinear residual is not transferred.", residualMeasured: true},
      E1_rand_curv: {stages: ["Receiving A", "Compute visited response", "Randomize residual", "Modified update returns to A"], note: "Randomized-residual control: the actual visited residual is replaced before transfer.", residualMeasured: true},
    };
    return controls[method] || {stages: ["Receiving A", "Temporary visit toward donor", "Compute F(visit) − visit", "Weighted update returns to A"], note: "A genuine visited-state transition is recorded; the returned update remains anchored at A.", residualMeasured: true};
  }

  function renderRecording() {
    const r = trace(), b = baseline(), t = token(), s = snapshot(), tel = telemetry();
    if (!r) return;
    all(".data-dependent").forEach((el) => { el.hidden = false; });
    $("tokenControls").hidden = !["decision", "experiment"].includes(state.view);
    $("problemPrompt").textContent = r.prompt;
    $("goldAnswer").textContent = `Verified target: ${r.gold_answer}`;
    $("baselineOutput").innerHTML = output(b);
    $("methodOutput").innerHTML = output(r);
    $("methodHeading").textContent = `${name(r.method)} · ${r.method}`;
    let result = "Baseline unavailable.";
    if (b) {
      if (!b.is_correct && r.is_correct) result = "Recovered in this recorded instance.";
      else if (b.is_correct && !r.is_correct) result = "Degraded in this recorded instance.";
      else result = r.is_correct ? "Both methods answered correctly." : "Neither method recovered the correct answer.";
    }
    $("outcomeNote").textContent = `${result} One selected example is not evidence of a general benefit.`;
    $("stepContext").textContent = `${t ? `Token ${state.token + 1}` : "Final-output snapshot; token identity unavailable"} · step ${state.step ?? "unavailable"} · snapshots are measured states`;
    const copy = mechanismCopy(r.method);
    const vals = [["Receiving A · entropy (nats)", s?.branch_entropy?.A], ["Receiving A · logit margin", s?.branch_margin?.A], ["A–B relative distance", s?.ab_relative_distance], ["Full-response residual ‖C‖", copy.residualMeasured ? tel?.curvature_frob : undefined]];
    $("stepTelemetry").innerHTML = vals.map(([label, value]) => `<div><small>${esc(label)}</small><strong>${num(value)}</strong></div>`).join("");
    renderDecision();
    const d = decisionAtStep();
    $("recordedStepTitle").textContent = `${t ? `Token ${state.token + 1}` : "Final-output snapshot"}, step ${state.step ?? "unavailable"}`;
    if (tel) {
      $("recordedMechanism").innerHTML = copy.stages.map((label, i) => `<span class="${i === 1 ? "visit" : "return"}">${esc(label)}</span>`).join("<b>→</b>");
      $("recordedStepNote").textContent = `Outgoing step ${tel.step} → ${tel.step + 1}. ${tel.skipped_unstable ? "Intervention rejected as unstable; no relocation applied." : copy.note} Donor ${tel.donor_branch_id ?? "unavailable"}; α_eff=${num(tel.effective_alpha, 2)}, β=${num(tel.beta, 2)}.${copy.residualMeasured ? ` Returned shift vs normal ${num(tel.returned_shift_vs_normal_frob)}. Full-response residual is not necessarily the scoped, applied update.` : " Placeholder response-residual and return-shift fields are not displayed as measurements for this control."}`;
      if (tel.receiving_state_hash && tel.visit_state_hash === tel.receiving_state_hash && (tel.beta === 0 || tel.effective_alpha === 0)) {
        $("recordedMechanism").innerHTML = '<span class="return">Receiving A</span><b>→</b><span>Identity/no-op: normal update</span>';
        $("recordedStepNote").textContent = "Recorded identity short-circuit: no visited-state transition is required. A advances normally.";
      }
      if (tel.skipped_unstable) $("recordedMechanism").innerHTML = '<span class="return">Receiving A</span><b>→</b><span>Normal update; intervention skipped</span>';
    } else if (d?.action_type === "EXIT") {
      $("recordedMechanism").innerHTML = '<span class="return">Receiving A</span><b>EXIT</b><span>Readout at this depth</span>';
      $("recordedStepNote").textContent = "Later steps are not recorded for this token. No excursion is implied.";
    } else {
      $("recordedMechanism").innerHTML = `<span class="return">Receiving A · step ${esc(state.step)}</span><span>${esc(d?.action_type || "Recorded state snapshot")}</span>`;
      $("recordedStepNote").textContent = "No outgoing excursion telemetry at the selected step. Absence of telemetry is not a fabricated visit or a zero residual.";
    }
    const ledger = r.token_ledger || [];
    $("ledgerRows").innerHTML = ledger.map((item, i) => `<tr data-selected="${i === state.token}"><td>${i + 1}: ${esc(JSON.stringify(item.decoded_token ?? item.token_id))}</td><td>${esc(item.controller_action_type)}</td><td>${esc(item.selected_donor)}</td><td>${num(item.effective_alpha, 2)} / ${num(item.effective_beta, 2)}</td>${["shared_calls", "primary_branch_calls", "donor_calls", "visit_calls", "continuation_calls", "recurrent_block_calls"].map((key) => `<td>${esc(item[key])}</td>`).join("")}</tr>`).join("");
    $("ledgerNote").textContent = ledgerMatches(r) ? `Ledger totals match ${r.recurrent_block_calls} recorded calls. Calls describe execution, not accuracy or equal FLOPs.` : "Missing or inconsistent execution ledger. Cost is unverified. Raw values are preserved, not repaired in the UI.";
    if (["P0", "P1"].includes(r.method)) $("ledgerNote").textContent = "Parallel selection cost is unverified, even if ledger sums coincide. These rows describe the selected output, not the total work of decoding all consulted candidates. Raw records preserve the reference budget separately when available.";
    if (r.derived_parallel_selection) $("ledgerNote").textContent += " Rows above describe the selected output only. reference_parallel_budget_ledger is a reference budget, not the total executed cost of decoding all selection candidates; inspect raw JSON.";
    renderProjection(r);
    $("rawTrace").textContent = JSON.stringify(r, null, 2);
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], {type:"application/json"}));
    $("downloadTrace").href = downloadUrl;
    $("downloadTrace").download = `${r.problem_id}-${r.seed}-${r.method}.json`;
    $("downloadRun").href = run().trace_url;
    $("downloadConfig").href = run().manifest_url;
    renderEvidence(); syncUrl();
  }

  function renderProjection(r) {
    const svg = $("projection");
    const branches = Object.entries(r.pca_coords || {}).filter(([branch, points]) => branch !== "VISIT" && Array.isArray(points) && points.length);
    if (state.token !== 0 || r.method === "E0" || !branches.length) {
      svg.replaceChildren();
      $("projectionNote").textContent = state.token !== 0 ? "No per-token projection recorded here. Token 1 coordinates are not borrowed for later tokens." : r.method === "E0" ? "Historical random-direction visit coordinates were reconstructed, not captured. Projection withheld." : "No measured projection available.";
      return;
    }
    const pts = branches.flatMap(([, points]) => points).filter((p) => p.length >= 2 && p.every(Number.isFinite));
    if (!pts.length) { svg.replaceChildren(); return; }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const x = (v) => 48 + (v - minX) / (maxX - minX || 1) * 600;
    const y = (v) => 215 - (v - minY) / (maxY - minY || 1) * 175;
    const colors = {A:"var(--a)", B:"var(--b)", C:"var(--visit)"};
    svg.innerHTML = '<text x="10" y="20" fill="var(--muted)" font-size="12">PC2</text><text x="660" y="247" fill="var(--muted)" font-size="12">PC1</text>' + branches.map(([branch, points]) => {
      const valid = points.filter((p) => p.length >= 2 && p.every(Number.isFinite));
      const color = colors[branch] || "var(--text)";
      return `<polyline points="${valid.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2" ${branch === "A" ? "" : 'stroke-dasharray="5 5"'}/>` + valid.map((p, i) => `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${i + 1 === state.step ? 6 : 4}" fill="${color}"/><text x="${x(p[0]) + 8}" y="${y(p[1]) - 8}" fill="${color}" font-size="11">${esc(branch)} #${i + 1}</text>`).join("");
    }).join("");
    $("projectionNote").textContent = `Recorded token 1 only. ${r.pca_explained_variance?.length >= 2 ? `PC1 ${pct(r.pca_explained_variance[0])}, PC2 ${pct(r.pca_explained_variance[1])} explained variance.` : "Explained variance unavailable."} Points are recorded branch samples; visit geometry is deliberately omitted.`;
  }

  function renderEvidence() {
    if (!hasData) { $("evidenceConclusion").textContent = "Recording manifest unavailable. No metrics can be verified or displayed."; return; }
    const current = run(), method = availableMethod(current.methods), stats = current.methods[method];
    if (state.view === "evidence") {
      state.method = method;
      options($("methodSelect"), Object.keys(current.methods).map((m) => [m, `${name(m)} · ${m}`]), method);
    }
    $("evidenceConclusion").textContent = `${name(method)}: ${pct(stats.accuracy)} recorded accuracy vs ${pct(stats.baseline_accuracy)} for ordinary recurrence (${pp(stats.paired_accuracy_diff)} percentage points). ${stats.cluster_ci_95_low <= 0 && stats.cluster_ci_95_high >= 0 ? "The problem-cluster interval includes zero; a general accuracy gain is not established." : "This exploratory contrast is not corrected for multiple comparisons."}`;
    $("sampleNote").textContent = `${current.label} · split ${current.split} · ${current.num_problems} distinct problems × ${current.seeds.length} seed(s) · ${stats.num_instances} paired evaluations per method · ${current.record_count} method traces. Seeds are repeated measurements, not independent problems.`;
    $("integrityNote").textContent = current.ledger_mismatches ? `${current.ledger_mismatches} execution ledgers are missing, inconsistent, or reference-only. Affected methods show “Unverified” costs below. Output accuracy is descriptive; old audit pass flags are not current verification.` : "Export hashes and paired keys are checkable. This recording is exploratory; no claim of strict compute matching or independent replication.";
    $("resultRows").innerHTML = Object.entries(current.methods).map(([m, s]) => `<tr data-selected="${m === state.method}"><td><button data-result-method="${esc(m)}" title="Inspect this recorded method"><span class="method-name">${esc(name(m))}</span><span class="method-code">${esc(m)}${m === "P1" || m === "D_oracle" ? " · gold-label oracle" : ""}</span></button></td><td>${pct(s.accuracy)}</td><td>${pp(s.paired_accuracy_diff)} pp</td><td>[${pp(s.cluster_ci_95_low)}, ${pp(s.cluster_ci_95_high)}] pp</td><td>${s.rescued_count} / ${s.base_wrong_count}</td><td>${s.degraded_count} / ${s.base_correct_count}</td><td>${(s.ledger_mismatches || ["P0", "P1"].includes(m)) ? "Unverified" : num(s.mean_recurrent_calls, 2)}</td></tr>`).join("");
    $("evidenceLinks").innerHTML = DATA.evidence.map((e) => `<li><a href="${esc(e.url)}">${esc(e.label)}${e.historical ? " · historical audit" : ""}</a><small>SHA-256 ${esc(e.sha256)}</small></li>`).join("");
    $("runHash").textContent = `Original JSONL SHA-256: ${current.source_trace_sha256}\nExported recording SHA-256: ${current.trace_sha256}`;
  }

  function renderMechanism() {
    const mode = state.mechanism;
    const descriptions = {
      excursion: ["Excursion-and-return", "A temporary visit moves toward B. The shared recurrent transition computes an update there. That update returns to A; A does not become B."],
      switch: ["Trajectory switching", "Select B as the new trajectory. A is abandoned. No update computed during a temporary visit returns to A."],
      mix: ["Direct state mixing", "Replace A with a blend of A and B, then continue from that blended state. There is no return to A’s original anchor."],
    };
    const [title, description] = descriptions[mode];
    const node = (x, y, w, label, sub, color = "var(--a)") => `<rect x="${x}" y="${y}" width="${w}" height="64" rx="6" class="node" style="stroke:${color}"/><text x="${x + w / 2}" y="${y + 27}" text-anchor="middle" class="label" style="fill:${color}">${label}</text><text x="${x + w / 2}" y="${y + 47}" text-anchor="middle" class="sub-label">${sub}</text>`;
    const line = (path, color, dashed = false) => `<path d="${path}" fill="none" stroke="${color}" stroke-width="2" ${dashed ? 'stroke-dasharray="6 5"' : ""} marker-end="url(#arrow)"/>`;
    let diagram = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="var(--muted)"/></marker></defs><title id="mechanismTitle">${title}</title><desc id="mechanismDesc">${description}</desc>`;
    diagram += node(24, 34, 150, "Receiving A", "original anchor") + node(24, 210, 150, "Donor B", "same model weights", "var(--b)");
    if (mode === "excursion") {
      diagram += node(280, 34, 180, "Normal update", "V(A) = F(A) − A") + node(280, 185, 180, "Temporary visit", "H_visit = A + α_eff(B−A)", "var(--visit)") + node(540, 185, 150, "Compute there", "V_visit = F(visit) − visit", "var(--visit)") + node(694, 34, 165, "Return to A", "A′ = A + weighted update");
      diagram += line("M174 66 H280", "var(--a)") + line("M174 242 H230 V217 H280", "var(--b)", true) + line("M99 98 V160 H370 V185", "var(--visit)", true) + line("M460 217 H540", "var(--visit)") + line("M615 185 V130 H776 V98", "var(--visit)", true) + line("M460 66 H694", "var(--a)");
      diagram += '<text x="540" y="49" class="sub-label">(1−β) V(A)</text><text x="672" y="120" class="sub-label">β V_visit returns</text>';
    } else if (mode === "switch") {
      diagram += node(620, 210, 210, "Continue as B", "A is not the receiving anchor", "var(--b)") + line("M174 242 H620", "var(--b)");
      diagram += '<text x="290" y="72" class="sub-label">A stops here. No excursion-and-return.</text>';
    } else {
      diagram += node(330, 120, 190, "Mixed state", "H_mix = (1−α)A + αB", "var(--visit)") + node(660, 120, 170, "Continue from mix", "F(H_mix)", "var(--visit)");
      diagram += line("M174 66 H250 V142 H330", "var(--a)") + line("M174 242 H250 V164 H330", "var(--b)", true) + line("M520 152 H660", "var(--visit)");
    }
    $("mechanismDiagram").innerHTML = diagram;
    $("mechanismExplanation").textContent = description;
    $("mechanismCaption").textContent = `${title}. Schematic positions encode computation order, not latent distance. Recipient A is indigo; donor B amber; temporary computation violet and dashed.`;
    all("[data-mechanism]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mechanism === mode)));
  }

  if (!hasData) {
    $("loadStatus").textContent = "Recording manifest unavailable. Source and methodology links remain available.";
    $("dataControls").hidden = false;
  }
  if (hasData && !DATA.runs[state.run]) state.run = Object.keys(DATA.runs)[0];
  if (hasData) options($("runSelect"), Object.entries(DATA.runs).map(([id, r]) => [id, r.label]), state.run);
  $("dataControls").after($("tokenControls"));
  all("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $("main").addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const go = event.target.closest("[data-go]");
    if (go instanceof HTMLElement) setView(go.dataset.go, true);
    const mechanism = event.target.closest("[data-mechanism]");
    if (mechanism instanceof HTMLElement) { state.mechanism = mechanism.dataset.mechanism; renderMechanism(); }
    const method = event.target.closest("[data-result-method]");
    if (method instanceof HTMLElement) { state.method = method.dataset.resultMethod; setView("experiment", true); }
  });
  /** @type {HTMLElement} */ (document.querySelector(".journey")).addEventListener("keydown", (event) => {
    const index = VIEWS.indexOf(state.view);
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % VIEWS.length;
    if (event.key === "ArrowLeft") next = (index + VIEWS.length - 1) % VIEWS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = VIEWS.length - 1;
    if (next !== undefined) { event.preventDefault(); setView(VIEWS[next], true); }
  });
  $("previousView").addEventListener("click", () => setView(VIEWS[VIEWS.indexOf(state.view) - 1], true));
  $("nextView").addEventListener("click", () => setView(VIEWS[VIEWS.indexOf(state.view) + 1], true));
  $("runSelect").addEventListener("change", () => {
    state.run = $("runSelect").value; state.problem = null; state.seed = null; state.token = 0;
    ++loadVersion; rows = []; pairs = new Map();
    options($("caseSelect"), [], "");
    setView(state.view);
  });
  $("caseSelect").addEventListener("change", () => { [state.problem, state.seed] = $("caseSelect").value.split("|"); state.token = 0; syncMethods(); renderRecording(); });
  $("methodSelect").addEventListener("change", () => {
    state.method = $("methodSelect").value;
    if (state.view === "evidence") { renderEvidence(); syncUrl(); }
    else { syncMethods(); renderRecording(); }
  });
  $("tokenSelect").addEventListener("change", () => { state.token = Number($("tokenSelect").value); syncSteps(); renderRecording(); });
  $("stepSelect").addEventListener("change", () => { state.step = Number($("stepSelect").value); renderRecording(); });
  $("retryLoad").addEventListener("click", loadRun);
  window.addEventListener("hashchange", () => { if (hashView()) setView(hashView(), true); });
  renderMechanism(); setView(state.view);
})();
