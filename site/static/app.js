/// <reference path="./schema.d.ts" />
/* Recorded evidence replay. The four-column stage is schematic, never latent geometry. */
(() => {
  "use strict";
  /** @type {Observatory | undefined} */
  const DATA = window.TEBDIL_DATA;
  if (!DATA || DATA.schema_version !== 2) return;

  const METHODS = {
    S0:["Ordinary recurrence","No intervention"], S1:["Extra recurrent depth","Fixed schedule"],
    P0:["Independent branches","Confidence-based selection; total selection cost unverified"], P1:["Oracle best branch","Uses gold outcomes; upper bound only; total selection cost unverified"],
    M0:["Direct state mixing","No return to receiving anchor"], M1:["Linear update mixing","No visited-state transition"],
    E1:["Fixed excursion","Fixed schedule"], E2:["Adaptive excursion","Heuristic adaptive rules; not learned"],
    E3_mlp:["Learned relocation","Offline-trained MLP + hand-set margin/entropy eligibility gates"],
    E3_mlp_ungated:["Ungated learned relocation","Offline-trained MLP; eligibility gates disabled"],
    E3_ridge:["Learned linear control","Offline-trained ridge + hand-set eligibility gates"],
    E3_no_exit:["Learned control without exit","Offline-trained scorer + hand-set eligibility gates"],
    E3_mlp_unshaped:["Unshaped learned control","Offline-trained MLP + heuristic gates"],
    E3_t2_1donor:["Budgeted learned control","Offline-trained MLP + heuristic gates"],
    EXIT_2:["Fixed early exit","Fixed schedule"], RAND_CTRL:["Random control","Random actions; not learned"],
    H_budget:["Budget heuristic","Hand-set adaptive rules; not learned"], G_gate_only:["Gate-only controller","Hand-set rules; no learned scorer"],
    R0:["Norm-matched random noise","Negative control"], E0:["Random-direction excursion","Negative control"],
    C0:["Visit without return","Visited state replaces receiving anchor"],
    E1_no_curv:["Excursion without residual","Visited response measured; nonlinear residual removed"],
    E1_rand_curv:["Randomized residual","Visited residual replaced before transfer"],
    D_nearest:["Nearest donor","Heuristic donor selection"], D_farthest:["Farthest donor","Heuristic donor selection"],
    D_confidence:["Highest-margin donor","Heuristic donor selection"], D_random:["Random donor","Random donor selection"],
    D_oracle:["Oracle donor","Uses gold outcomes; upper bound only"], D_learned:["Learned donor","Offline-trained scorer + hand-set gates"],
    E1_last_token:["Last-token transfer","Fixed scoped excursion"], E1_norm_tight:["Tight trust region","Fixed clipped excursion"],
    E1_coord_top50:["Coordinate-restricted transfer","Fixed mask"], E1_subspace:["Coordinate-limited transfer","Fixed correction-coordinate restriction; not a learned subspace"],
    E1_multi_visit:["Multiple visits","Fixed schedule"], E1_gated:["Native-gate weighted transfer","Return scaled by Ouro gate; not a trained controller"],
  };
  const ORDER = Object.keys(METHODS);
  const name = (method) => METHODS[method]?.[0] || method;
  const policy = (method) => METHODS[method]?.[1] || "See execution manifest";
  const esc = (value) => String(value ?? "Unavailable").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  const num = (value, digits=3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "Unavailable";
  const pct = (value) => typeof value === "number" ? `${(100 * value).toFixed(1)}%` : "Unavailable";
  const pp = (value) => typeof value === "number" ? `${value > 0 ? "+" : ""}${(100 * value).toFixed(1)} pp` : "Unavailable";
  /** @param {string} id */
  const el = (id) => document.getElementById(id);
  /** @param {string} id */
  const select = (id) => /** @type {HTMLSelectElement | null} */ (el(id));
  /** @param {string} id */
  const button = (id) => /** @type {HTMLButtonElement | null} */ (el(id));
  /** @param {HTMLSelectElement | null} target @param {Array<[string|number,string]>} items @param {string|number|null} value */
  function setOptions(target, items, value) {
    if (!target) return;
    target.replaceChildren(...items.map(([key,label]) => new Option(label,String(key))));
    target.value = String(value ?? ""); target.disabled = !items.length;
  }

  const aliases = {fresh_test_phase7_8:"phase7_8_fresh_test",test_phase7_8:"phase7_8_controller_test",test_phase4:"phase4_adaptive",validation_phase3:"phase3_ablations",validation_phase5:"phase5_donor",validation_phase6:"phase6_selective"};
  const params = new URLSearchParams(location.search);
  const requestedRun = aliases[params.get("run")] || params.get("run");
  const defaultRun = DATA.runs.phase7_8_controller_test ? "phase7_8_controller_test" : Object.keys(DATA.runs)[0];
  const state = {
    run: DATA.runs[requestedRun] ? requestedRun : defaultRun,
    problem: params.get("problem") || "arith_test_003",
    seed: params.get("seed") || "42",
    method: params.get("method") || "E3_mlp",
    token: Math.max(0, Number(params.get("token")) || 0),
    step: Math.max(1, Number(params.get("step")) || 2),
  };
  /** @type {RecordedTrace[]} */ let rows=[];
  /** @type {Map<string,Record<string,RecordedTrace>>} */ let pairs=new Map();
  const cache = new Map(); let loadVersion=0; let autoLoopTimer=0; let selectedDownloadUrl="";
  const run = () => DATA.runs[state.run];
  const pair = () => pairs.get(`${state.problem}|${state.seed}`) || {};
  const trace = () => pair()[state.method];
  const baseline = () => pair().S0;
  const ledger = () => trace()?.token_ledger || [];
  const token = () => ledger()[state.token];
  const snapshots = () => token()?.step_snapshots || (state.token === 0 ? trace()?.step_snapshots || [] : []);
  const snapshot = () => snapshots().find((item) => item.step === state.step);
  const telemetry = () => snapshots().find((item) => item.excursion_telemetry?.step === state.step)?.excursion_telemetry || null;
  function decision() {
    for (const item of snapshots()) {
      const d=item.policy_decision; if (!d) continue;
      const actionType=d.action_type||(d.action==="visit"?"EXCURSION":d.action==="no_visit"?"CONTINUE":"");
      const decisionStep=d.state_features?.step ?? item.excursion_telemetry?.step ?? (item.step-(actionType === "EXIT" ? 0 : 1));
      if (decisionStep === state.step) return {...d,action_type:actionType};
    }
    return null;
  }
  function ledgerMatches(row) {
    if (row.derived_parallel_selection || ["P0","P1"].includes(row.method)) return false;
    return !!row.token_ledger?.length && row.token_ledger.reduce((sum,item)=>sum+item.recurrent_block_calls,0) === row.recurrent_block_calls;
  }
  function syncUrl() {
    const query=new URLSearchParams();
    Object.entries(state).forEach(([key,value])=>query.set(key,String(value)));
    history.replaceState(null,"",`${location.pathname}?${query}${location.hash}`);
  }
  function typeset(...elements) { window.MathJax?.typesetPromise?.(elements.filter(Boolean)).catch(()=>{}); }

  function initControls() {
    setOptions(select("runSelect"),Object.entries(DATA.runs).map(([id,item])=>[id,item.label]),state.run);
    select("runSelect")?.addEventListener("change",()=>{ state.run=select("runSelect").value; state.problem=""; state.seed=""; loadRun(); });
    select("caseSelect")?.addEventListener("change",()=>{ const [problem,seed]=select("caseSelect").value.split("|"); stopAutoLoop();state.problem=problem;state.seed=seed;syncMethods();renderAll(); });
    select("methodSelect")?.addEventListener("change",()=>changeMethod(select("methodSelect").value));
    select("tokenSelect")?.addEventListener("change",()=>{ stopAutoLoop();state.token=Number(select("tokenSelect").value);syncSteps();renderAll(); });
    select("stepSelect")?.addEventListener("change",()=>{ state.step=Number(select("stepSelect").value);renderAll(); });
    document.querySelectorAll(".step-pill").forEach((item)=>item.addEventListener("click",()=>{ stopAutoLoop(); state.step=Number(item.getAttribute("data-step"))||1; if(select("stepSelect")) select("stepSelect").value=String(state.step); renderAll(); }));
    button("playLoopBtn")?.addEventListener("click",()=>autoLoopTimer?stopAutoLoop():startAutoLoop());
    button("retryLoad")?.addEventListener("click",loadRun);
    select("explorerSplitFilter")?.addEventListener("change",()=>{ state.run=select("explorerSplitFilter").value; if(select("runSelect")) select("runSelect").value=state.run; state.problem="";state.seed="";loadRun(); });
    select("explorerFamilyFilter")?.addEventListener("change",renderExplorer);
    select("explorerMethodFilter")?.addEventListener("change",renderExplorer);
    select("gridMetricSelect")?.addEventListener("change",()=>renderHeatmap(select("gridMetricSelect")?.value || "mean_curvature_frob"));
    initTocAndCopy();
  }
  function startAutoLoop() {
    stopAutoLoop();
    if(availableSteps().length<2)return;
    if(el("playLoopIcon")) el("playLoopIcon").className="fas fa-pause";
    if(el("playLoopText")) el("playLoopText").textContent="Pause";
    autoLoopTimer=window.setInterval(()=>{ const steps=availableSteps(); const index=Math.max(0,steps.indexOf(state.step)); state.step=steps[(index+1)%steps.length]; if(select("stepSelect"))select("stepSelect").value=String(state.step);renderAll(); },1800);
  }
  function stopAutoLoop() {
    if(autoLoopTimer) window.clearInterval(autoLoopTimer); autoLoopTimer=0;
    if(el("playLoopIcon")) el("playLoopIcon").className="fas fa-play";
    if(el("playLoopText")) el("playLoopText").textContent="Auto-Loop";
  }
  async function loadRun() {
    stopAutoLoop(); const version=++loadVersion; const selectedRun=state.run; const recording=run();
    rows=[];pairs=new Map(); renderResultTables(); renderExplorer(); setLoading(true,`Loading ${recording.label.toLowerCase()}… Recorded data, not a new model run.`);
    try {
      let recorded=cache.get(selectedRun);
      if(!recorded) {
        const response=await fetch(recording.trace_url); if(!response.ok) throw new Error(`HTTP ${response.status}`);
        const bytes=await response.arrayBuffer(); if(!crypto?.subtle) throw new Error("Secure browser context required for checksum verification");
        const hash=[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map((b)=>b.toString(16).padStart(2,"0")).join("");
        if(hash!==recording.trace_sha256) throw new Error("Recording checksum mismatch");
        recorded=JSON.parse(new TextDecoder().decode(bytes));
        if(!Array.isArray(recorded)||recorded.length!==recording.record_count) throw new Error("Recording count mismatch");
        cache.set(selectedRun,recorded);
      }
      if(version!==loadVersion)return;
      rows=recorded;
      for(const row of rows) { const key=`${row.problem_id}|${row.seed}`; if(!pairs.has(key))pairs.set(key,{});pairs.get(key)[row.method]=row; }
      let key=`${state.problem}|${state.seed}`;
      if(!pairs.has(key)) {
        const featured=rows.find((item)=>item.method==="E3_mlp"&&item.problem_id==="arith_test_003"&&item.seed===42)
          || rows.find((item)=>item.method==="E3_mlp"&&item.token_ledger?.some((entry)=>entry.controller_action_type==="EXCURSION")) || rows[0];
        state.problem=featured.problem_id;state.seed=String(featured.seed);key=`${state.problem}|${state.seed}`;
      }
      setOptions(select("caseSelect"),[...pairs].map(([pairKey,methods])=>{const first=methods.S0||Object.values(methods)[0];return [pairKey,`${first.problem_id} · seed ${first.seed} · ${first.family}`];}),key);
      if(select("explorerSplitFilter")) setOptions(select("explorerSplitFilter"),Object.entries(DATA.runs).map(([id,item])=>[id,`${item.label} · ${item.record_count.toLocaleString()} traces`]),state.run);
      setLoading(false,`${recording.record_count.toLocaleString()} recorded traces · ${recording.num_problems} distinct problems · ${recording.seeds.length} seed(s) · SHA-256 verified in this browser.`);
      syncMethods();renderAll(); renderResultTables(); renderExplorer(); loadAuxiliaryEvidence();
    } catch(error) {
      if(version!==loadVersion)return;
      const message=error instanceof Error?error.message:String(error);
      setLoading(false,`Recording unavailable: ${message}. No results were simulated.`,true);
    }
  }
  function setLoading(loading,message,failed=false) {
    if(el("loadStatus")){el("loadStatus").textContent=message;el("loadStatus").setAttribute("aria-busy",String(loading));}
    if(button("retryLoad"))button("retryLoad").hidden=!failed;
    ["caseSelect","methodSelect","tokenSelect","stepSelect"].forEach((id)=>{const target=select(id);if(target)target.disabled=loading||failed;});
    document.querySelectorAll(".stage-grid,.output-terminal,.recording-evidence:not(#ouroExecutionContract),.question-box-live").forEach((item)=>{if(item instanceof HTMLElement)item.hidden=loading||failed;});
    document.querySelectorAll(".step-pill,#playLoopBtn").forEach((item)=>{if(item instanceof HTMLButtonElement)item.disabled=loading||failed;});
    if(loading||failed){if(el("comparisonContext"))el("comparisonContext").textContent="";if(el("runSummary"))el("runSummary").textContent=message;}
    if(loading||failed)document.querySelectorAll(".metric-num").forEach((item)=>{item.textContent=loading?"Loading…":"Unavailable";});
  }
  function syncMethods() {
    const methods=pair(); const available=Object.keys(methods); if(!available.includes(state.method))state.method=available.includes("E3_mlp")?"E3_mlp":available.includes("E1")?"E1":available[0];
    available.sort((a,b)=>{const ai=ORDER.indexOf(a),bi=ORDER.indexOf(b);return(ai<0?999:ai)-(bi<0?999:bi);});
    setOptions(select("methodSelect"),available.map((method)=>[method,`${name(method)} · ${method}`]),state.method); syncTokenAndSteps();
  }
  function changeMethod(method) {
    stopAutoLoop();state.method=method;
    if(select("methodSelect"))select("methodSelect").value=method;
    syncTokenAndSteps();renderAll();
  }
  function syncTokenAndSteps() {
    const entries=ledger();
    /** @type {Array<[number,string]>} */
    const items=entries.map((item,index)=>[index,`Token ${index+1}: ${JSON.stringify(item.decoded_token??item.token_id)}`]);
    if(!entries.length&&trace()?.step_snapshots?.length)items.push([0,"Token 1 · legacy snapshots; token cost unavailable"]);
    if(!items.some(([index])=>index===state.token))items.push([state.token,`Token ${state.token+1} · not recorded for this method`]);
    setOptions(select("tokenSelect"),items,state.token);
    if(select("tokenSelect"))select("tokenSelect").disabled=items.length<2;syncSteps();
  }
  function availableSteps() {
    return [...new Set(snapshots().map((item)=>item.step))].sort((a,b)=>a-b);
  }
  function syncSteps() {
    const steps=availableSteps();
    /** @type {Array<[number,string]>} */
    const items=steps.map((step)=>[step,`Step ${step}`]);
    if(!steps.includes(state.step))items.push([state.step,`Step ${state.step} · not recorded for this method`]);
    setOptions(select("stepSelect"),items,state.step);
    if(select("stepSelect"))select("stepSelect").disabled=!steps.length;
  }
  function renderAll() {
    const row=trace();if(!row)return;
    document.querySelectorAll(".step-pill").forEach((item)=>{const selected=Number(item.getAttribute("data-step"))===state.step;item.classList.toggle("is-active",selected);item.setAttribute("aria-pressed",String(selected));if(item instanceof HTMLButtonElement)item.disabled=!availableSteps().includes(Number(item.getAttribute("data-step")));});
    if(button("playLoopBtn"))button("playLoopBtn").disabled=availableSteps().length<2;
    const context=`Same case / seed / token position / step across methods. ${snapshot()?"Exact snapshot recorded.":"Snapshot unavailable; no neighboring token or step substituted."}`;
    if(el("comparisonContext")&&el("comparisonContext").textContent!==context)el("comparisonContext").textContent=context;
    if(el("stageStepLabel"))el("stageStepLabel").textContent=`Token ${state.token+1} · recurrent step t = ${state.step} · ${name(row.method)}`;
    if(el("liveProblemHeader"))el("liveProblemHeader").textContent=`Problem: ${row.problem_id} (seed=${row.seed}) · ${row.split} · ${row.family}`;
    if(el("liveGoldBadge"))el("liveGoldBadge").textContent=`Verified target: ${row.gold_answer}`;
    if(el("livePromptBody"))el("livePromptBody").textContent=row.prompt;
    if(el("policyKind"))el("policyKind").textContent=policy(row.method);
    renderStageSvg(row);renderPca(row);renderTelemetry(row);renderTerminal(row);renderTopMetrics();renderEvidence(row);syncUrl();
  }
  function renderTopMetrics() {
    const summary=run().methods[state.method]; const cards=Array.from(document.querySelectorAll(".metric-card")); if(cards.length<4||!summary)return;
    const values=[pct(summary.accuracy),!summary.ledger_mismatches&&ledgerMatches(trace())?num(summary.mean_recurrent_calls,2):"Unverified",`${summary.rescued_count} / ${summary.degraded_count}`,String(Object.keys(DATA.runs).length)];
    const labels=[`${name(state.method)} accuracy`,`Mean recurrent calls`,`Rescued / degraded`,`Recorded experiment sets`];
    const subs=[`${pp(summary.paired_accuracy_diff)} vs S0 · 95% cluster CI [${pp(summary.cluster_ci_95_low)}, ${pp(summary.cluster_ci_95_high)}]`,!summary.ledger_mismatches&&ledgerMatches(trace())?"Selected recording summary":"Cost ledger is missing, reference-only, or incomplete",`${summary.base_wrong_count} baseline-wrong / ${summary.base_correct_count} baseline-correct`,"Exploratory artifacts with explicit provenance"];
    cards.forEach((card,index)=>{const value=card.querySelector(".metric-num"),label=card.querySelector(".metric-label"),sub=card.querySelector(".metric-sub");if(value)value.textContent=values[index];if(label)label.textContent=labels[index];if(sub)sub.textContent=subs[index];});
  }
  function mechanism(row, tel=telemetry()) {
    const d=decision(),entry=token();
    const action=(d?.action_type||entry?.controller_action_type||"").toUpperCase();
    if(action==="EXIT")return {kind:"exit",labels:["Receiving A","Controller: EXIT","Later recurrence bypassed","Readout at this depth"],note:"Recorded EXIT decision. Later recurrent states are not fabricated.",formula:"EXIT → readout"};
    if(row.method==="M0")return {kind:"mix",labels:["Receiving A + donor","Direct hidden-state mix","Compute F(mixed state)","Continue from mixture"],note:"Direct state mixing permanently changes the anchor; it is not excursion-and-return.",formula:"H_mix = (1−α)A + αB"};
    if(row.method==="M1")return {kind:"mix",labels:["Endpoint A and donor","Mix endpoint updates","Apply linear update to A","Continue from A"],note:"Linear update mixing does not evaluate the recurrent transition at a visited state.",formula:"A′ = A + (1−αβ)V(A) + αβV(B)"};
    if(row.method==="C0")return {kind:"switch",labels:["Receiving A","Visit toward donor","Compute F(visit)","Continue from visit"],note:"Visit-without-return replaces the receiving anchor. No update returns to A.",formula:"H′ = F(H_visit)"};
    if(row.method==="R0")return {kind:"noise",labels:["Receiving A","Norm-matched noise","Perturbed normal update","Continue perturbed"],note:"Random-noise control; no donor visit or return is implied.",formula:"A′ = F(A) + ε"};
    if(["P0","P1"].includes(row.method))return {kind:"parallel",labels:["Independent A and B","Run both branches","Select an output","Selected readout"],note:`${row.method} selects between independent trajectories. It does not transfer an update; total selection cost is unverified.`,formula:"independent branches → selection"};
    if(["S0","S1","EXIT_2"].includes(row.method)||action==="CONTINUE"&&!tel)return {kind:"normal",labels:["Receiving A","Normal recurrent update","Continue A","Readout"],note:action==="CONTINUE"?"Recorded CONTINUE decision. No outgoing excursion is recorded at this step.":"Ordinary recurrence; no intervention is implied.",formula:"A′ = F(A)"};
    if(!tel)return {kind:"unavailable",labels:["Receiving A",`Step ${state.step}`,"No outgoing visit recorded","No return claimed"],note:"No outgoing excursion telemetry exists at the selected token and step. Absence is not shown as a zero-valued visit.",formula:"telemetry unavailable"};
    if(tel.skipped_unstable)return {kind:"normal",labels:["Receiving A","Intervention rejected","Normal update","Continue A"],note:"Recorded intervention skipped as unstable; no excursion applied.",formula:"A′ = F(A)"};
    if(tel.visit_state_hash===tel.receiving_state_hash&&(tel.effective_alpha===0||tel.beta===0))return {kind:"normal",labels:["Receiving A","Identity/no-op","Normal update","Continue A"],note:"Identity short-circuit; no visited-state transition is required.",formula:"A′ = F(A)"};
    const alpha=tel.effective_alpha;const beta=tel.beta;
    const note=row.method==="E0"?"Random-direction visited-state transition; the visit does not point toward the donor state."
      :row.method==="E1_no_curv"?"Visited response measured; the nonlinear residual is removed before transfer."
      :row.method==="E1_rand_curv"?"Visited response measured; its residual is replaced with a norm-matched random direction."
      :"Recorded visited-state transition. The computed update returns to receiving trajectory A.";
    return {kind:"excursion",labels:["Receiving A",`Temporary visit · α_eff=${num(alpha,2)}`,"Compute F(visit) − visit",`Return update · β=${num(beta,2)}`],note:`${note}${tel.clipped?" Trust-region clipping changed the effective visit strength.":""}`,formula:"A′ = A + (1−β)V(A) + βV(visit)"};
  }
  function renderStageSvg(row) {
    const svg=el("excursionStageSvg");if(!(svg instanceof SVGElement))return;
    if(!snapshot()) {
      svg.innerHTML='<text x="340" y="145" text-anchor="middle" font-size="15" fill="#475569">Selected snapshot not recorded</text><text x="340" y="172" text-anchor="middle" font-size="12" fill="#64748b">Choose an available token or step; final output below remains recorded.</text>';
      if(el("liveFormulaPill"))el("liveFormulaPill").textContent="No snapshot substituted";
      if(el("stageStepNarrative"))el("stageStepNarrative").textContent=`Token ${state.token+1}, step ${state.step} is unavailable for ${name(row.method)}. Comparison context is preserved; no visit, return or controller action is inferred.`;
      return;
    }
    // Retain the original fork/visit/return diagram; the path covers the selected token, not just one snapshot.
    const states=snapshots(),event=states.find((item)=>item.excursion_telemetry)?.excursion_telemetry;
    const entry=token(),model=mechanism(row,event),activeStep=state.step;
    const sourceStep=event?.step??row.visit_step??2,source=states.find((item)=>item.step===sourceStep);
    const selectedDonor=event?.donor_branch_id||entry?.selected_donor||"B";
    const donorCount=Math.max(0,Object.keys(states[0]?.branch_rms||{}).length-1);
    const hasBranchB=donorCount>0,hasReturn=model.kind==="excursion";
    const hasMidpointEval=!!event&&(hasReturn||model.kind==="switch");
    const earlyExited=entry?.controller_action_type==="EXIT";
    const continuation=model.kind==="mix"?"State / update mix":model.kind==="switch"?"Visit becomes state":model.kind==="parallel"?"Independent branches":model.kind==="noise"?"Noise intervention":hasReturn?"Update anchored at A":"Normal recurrence";
    const trace={...row,alpha:event?.effective_alpha,beta:event?.beta};
    const fmtNum=num,curvFrob=num(event?.curvature_frob,2);
    const relDist2=num(event?.branch_relative_distance??source?.ab_relative_distance,3);
    const marginA2=num(source?.branch_margin?.A,3),entA2=num(source?.branch_entropy?.A,3);
    const s1Active=activeStep===1,s2Active=activeStep===2,s3Active=activeStep===3,s4Active=activeStep===4;
    const colActive="#4338ca",colDonor="#0ea5e9",colVisit="#7c3aed",colReturn="#059669";
    const donorLabel=`Donor H₂${selectedDonor === "C" ? "ᶜ" : "ᴮ"}`;
    svg.innerHTML = `
      <defs>
        <marker id="arrIndigo" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#4338ca"/>
        </marker>
        <marker id="arrSky" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#0ea5e9"/>
        </marker>
        <marker id="arrViolet" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/>
        </marker>
        <marker id="arrEmerald" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669"/>
        </marker>
      </defs>

      <!-- Step column background bands -->
      <rect x="18" y="10" width="135" height="298" rx="10" fill="${s1Active ? 'rgba(99,102,241,0.09)' : '#f8fafc'}" stroke="${s1Active ? '#6366f1' : '#e2e8f0'}" stroke-width="${s1Active ? 2 : 1}"/>
      <rect x="168" y="10" width="185" height="298" rx="10" fill="${s2Active ? 'rgba(124,58,237,0.09)' : '#f8fafc'}" stroke="${s2Active ? '#7c3aed' : '#e2e8f0'}" stroke-width="${s2Active ? 2 : 1}"/>
      <rect x="368" y="10" width="140" height="298" rx="10" fill="${s3Active ? 'rgba(5,150,105,0.09)' : '#f8fafc'}" stroke="${s3Active ? '#059669' : '#e2e8f0'}" stroke-width="${s3Active ? 2 : 1}"/>
      <rect x="522" y="10" width="144" height="298" rx="10" fill="${s4Active ? 'rgba(14,165,233,0.09)' : '#f8fafc'}" stroke="${s4Active ? '#0ea5e9' : '#e2e8f0'}" stroke-width="${s4Active ? 2 : 1}"/>

      <text x="85" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#475569">STEP t = 1 (FORK)</text>
      <text x="260" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#475569">STEP t = 2 (STATE / ACTION)</text>
      <text x="438" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#475569">STEP t = 3 (UPDATE)</text>
      <text x="594" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#475569">${earlyExited ? "EARLY READOUT" : "CONTINUE / READOUT"}</text>

      <!-- ================= PRIMARY BRANCH A (TOP RAIL) ================= -->
      <rect x="34" y="44" width="104" height="52" rx="10" fill="#fff" stroke="${colActive}" stroke-width="${s1Active ? 3 : 2}"/>
      <text x="86" y="65" text-anchor="middle" font-size="12" font-weight="800" fill="#1e1b4b">Primary H₁ᴬ</text>
      <text x="86" y="82" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono,monospace" fill="#475569">Anchor State</text>

      <line x1="138" y1="70" x2="196" y2="70" stroke="${colActive}" stroke-width="2.5" marker-end="url(#arrIndigo)"/>
      <text x="167" y="62" text-anchor="middle" font-size="9" font-weight="700" fill="#4338ca">F_θ,1</text>

      <rect x="202" y="44" width="116" height="52" rx="10" fill="#fff" stroke="${colActive}" stroke-width="${s2Active ? 3 : 2}"/>
      <text x="260" y="64" text-anchor="middle" font-size="12" font-weight="800" fill="#1e1b4b">State H₂ᴬ</text>
      <text x="260" y="81" text-anchor="middle" font-size="9" font-family="JetBrains Mono,monospace" fill="#475569">m=${marginA2} | H=${entA2}</text>

      <!-- Baseline V2^A arrow -->
      <line x1="318" y1="70" x2="376" y2="70" stroke="${colActive}" stroke-width="2.5" marker-end="url(#arrIndigo)" opacity="${earlyExited ? '0.3' : '1.0'}"/>
      <text x="347" y="61" text-anchor="middle" font-size="9" font-weight="700" fill="#4338ca">${earlyExited ? 'EXIT' : hasReturn ? '(1−β)V₂ᴬ' : 'shared F / update'}</text>

      <!-- Returned state H3^{A'} -->
      <rect x="382" y="44" width="112" height="54" rx="10" fill="${s3Active ? '#ecfdf5' : '#fff'}" stroke="${colReturn}" stroke-width="${s3Active ? 3 : 2}" opacity="${earlyExited ? '0.45' : '1.0'}"/>
      <text x="438" y="64" text-anchor="middle" font-size="12" font-weight="800" fill="#065f46">${earlyExited ? 'Bypassed' : hasReturn ? 'Return H₃ᴬ′' : 'Next state H₃'}</text>
      <text x="438" y="81" text-anchor="middle" font-size="9" font-family="JetBrains Mono,monospace" fill="#047857">${earlyExited ? 'Readout at exit depth' : continuation}</text>

      <line x1="494" y1="70" x2="532" y2="70" stroke="${colReturn}" stroke-width="2.5" marker-end="url(#arrEmerald)"/>
      <text x="513" y="61" text-anchor="middle" font-size="9" font-weight="700" fill="#059669">${earlyExited ? 'readout' : 'shared F'}</text>

      <!-- Final Readout Node -->
      <rect x="538" y="42" width="116" height="58" rx="10" fill="#ffffff" stroke="#4338ca" stroke-width="${s4Active ? 3 : 2}"/>
      <text x="596" y="62" text-anchor="middle" font-size="11" font-weight="800" fill="#1e1b4b">Token readout</text>
      <text x="596" y="80" text-anchor="middle" font-size="11" font-family="JetBrains Mono,monospace" font-weight="800" fill="#4338ca">${esc(JSON.stringify(entry?.decoded_token ?? "unavailable"))}</text>
      <text x="596" y="93" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono,monospace" fill="#64748b">Final answer in terminal</text>

      <!-- ================= MIDPOINT EXCURSION ZONE (CENTER RAIL) ================= -->
      <g opacity="1.0" style="display:${hasMidpointEval ? 'inline' : 'none'}">
        <path d="M 245 96 L 245 126" stroke="${colVisit}" stroke-width="2" stroke-dasharray="4 2" marker-end="url(#arrViolet)"/>
        <path d="M 245 236 L 245 200" stroke="${colVisit}" stroke-width="2" stroke-dasharray="4 2" marker-end="url(#arrViolet)"/>

        <rect x="182" y="132" width="156" height="62" rx="11" fill="${s2Active ? '#f5f3ff' : '#fff'}" stroke="${colVisit}" stroke-width="${s2Active ? 3 : 2}"/>
        <text x="260" y="151" text-anchor="middle" font-size="11" font-weight="800" fill="#5b21b6">Visit H₂ᵛⁱˢⁱᵗ (α_eff=${fmtNum(trace.alpha,2)})</text>
        <text x="260" y="167" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono,monospace" fill="#6d28d9">${hasReturn ? "V_visit = F(visit) − visit" : "F(visit): state replaces A"}</text>
        <text x="260" y="182" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono,monospace" font-weight="700" fill="#4c1d95">${hasReturn ? `||C₂||_F = ${curvFrob}` : "No anchor return"}</text>

        <path d="M 338 162 Q 415 162 432 104" fill="none" stroke="${colReturn}" stroke-width="2.8" marker-end="url(#arrEmerald)"/>
        <rect x="355" y="130" width="96" height="22" rx="6" fill="#ecfdf5" stroke="#6ee7b7"/>
        <text x="403" y="145" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono,monospace" font-weight="700" fill="#047857">${hasReturn ? `β V_visit (β=${fmtNum(trace.beta,2)})` : "Continue from visit"}</text>
      </g>

      <!-- ================= PERTURBED DONOR BRANCH B/C (BOTTOM RAIL) ================= -->
      <g opacity="1.0" style="display:${hasBranchB ? 'inline' : 'none'}">
        <path d="M 86 96 L 86 230" stroke="${colDonor}" stroke-width="2" stroke-dasharray="4 2" marker-end="url(#arrSky)"/>
        <rect x="38" y="148" width="96" height="22" rx="6" fill="#e0f2fe" stroke="#7dd3fc"/>
        <text x="86" y="163" text-anchor="middle" font-size="9" font-family="JetBrains Mono,monospace" font-weight="700" fill="#0369a1">Norm-bounded fork</text>

        <rect x="34" y="236" width="104" height="52" rx="10" fill="#fff" stroke="${colDonor}" stroke-width="${s1Active ? 3 : 2}"/>
        <text x="86" y="257" text-anchor="middle" font-size="12" font-weight="800" fill="#0c4a6e">Donor H₁${selectedDonor === "C" ? "ᶜ" : "ᴮ"}</text>
        <text x="86" y="274" text-anchor="middle" font-size="9.5" font-family="JetBrains Mono,monospace" fill="#0369a1">${donorCount > 1 ? "Donor pool B, C" : "Perturbed fork"}</text>

        <line x1="138" y1="262" x2="196" y2="262" stroke="${colDonor}" stroke-width="2.5" marker-end="url(#arrSky)"/>
        <text x="167" y="254" text-anchor="middle" font-size="9" font-weight="700" fill="#0284c7">F_θ,1</text>

        <rect x="202" y="236" width="116" height="52" rx="10" fill="#fff" stroke="${colDonor}" stroke-width="${s2Active ? 3 : 2}"/>
        <text x="260" y="256" text-anchor="middle" font-size="11.5" font-weight="800" fill="#0c4a6e">${donorLabel}</text>
        <text x="260" y="273" text-anchor="middle" font-size="9" font-family="JetBrains Mono,monospace" fill="#0369a1">RelDist=${relDist2}</text>

        <rect x="378" y="242" width="124" height="40" rx="8" fill="#f1f5f9" stroke="#cbd5e1" stroke-dasharray="4 2"/>
        <text x="440" y="259" text-anchor="middle" font-size="9.5" font-weight="700" fill="#475569">${model.kind === "parallel" ? "Independent selection" : "Donor ≠ receiver"}</text>
        <text x="440" y="273" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono,monospace" fill="#64748b">${model.kind === "parallel" ? `B: ${esc(row.donor_b_answer)}` : "Shared model weights"}</text>
      </g>
    `;
    if(el("liveFormulaPill"))el("liveFormulaPill").textContent=model.formula;
    renderStepNarrative(row,mechanism(row),states);
  }
  function renderStepNarrative(row,model,states) {
    const target=el("stageStepNarrative"),tel=telemetry(),d=decision(),entry=token();if(!target)return;
    const previous=states.find((item)=>item.excursion_telemetry?.step===state.step-1)?.excursion_telemetry;
    let selected=tel?"Recorded intervention.":"No outgoing controller decision or visit recorded at this step.";
    if(d)selected=`Controller: ${d.action_type}.`;
    if(tel&&!tel.skipped_unstable&&model.kind==="excursion")selected+=` ${row.method==="E0"?"Random-direction visit":`Visit toward ${tel.donor_branch_id||"donor (ID not captured)"}`} with α_eff=${num(tel.effective_alpha,3)}. Compute V(visit)=F(visit)−visit, ‖V(visit)‖=${num(tel.update_visit_frob,3)}. Transfer with β=${num(tel.beta,3)}; keep A as anchor. Returned shift vs ordinary F(A): ${num(tel.returned_shift_vs_normal_frob,3)} (not the residual ‖C‖).`;
    else if(tel)selected+=` ${model.note}`;
    else if(previous&&!previous.skipped_unstable)selected+=" This snapshot follows the recorded intervention at the preceding step; it is not another excursion.";
    const reason=d?.reason||(!d&&tel?entry?.policy_reason:null);
    target.innerHTML=`<b>Selected step ${state.step} · token ${state.token+1}:</b> ${esc(selected)}<details><summary>Decision reason &amp; token compute</summary><p>Reason: <code>${esc(reason||"not captured at this step")}</code>. ${d?.predicted_advantage!==undefined?`Recorded action score: ${num(d.predicted_advantage,4)} (not a probability). `:""}${entry?`Token total: ${entry.recurrent_block_calls} recurrent calls; donor ${entry.donor_calls??"unavailable"}, visit ${entry.visit_calls??"unavailable"}.`:"Legacy first-token snapshots only; token-level compute unavailable."}</p>${tel?`<p>Receiving / visited state hashes: <code>${esc(tel.receiving_state_hash)}</code> / <code>${esc(tel.visit_state_hash)}</code>. Post-intervention state: <code>${esc(tel.post_receiving_state_hash)}</code>. Missing hashes or norms remain unavailable; see the selected raw trace below.</p>`:""}</details><small>${esc(model.note)} Whole-token schematic, not latent coordinates.${states.filter((item)=>item.excursion_telemetry).length>1?" Only the first visit is drawn.":""}${availableSteps().some((step)=>step>4)?" Use the step selector beyond the four-column schematic.":""}</small>`;
  }
  function renderPca(row) {
    const svg=el("pcaPlotSvg");if(!(svg instanceof SVGElement))return;
    const note=el("pcaNote");if(el("pcaVarLabel"))el("pcaVarLabel").textContent="Projection unavailable";
    if(!snapshot()||state.token!==0||row.method==="E0") {svg.innerHTML=`<text x="190" y="86" text-anchor="middle" font-size="11" fill="#64748b">Projection withheld</text>`;if(note)note.textContent=!snapshot()?"No projection highlighted: the selected snapshot is not recorded.":state.token!==0?"Only token 1 coordinates were recorded; they are not borrowed for later tokens.":"Random-direction visit coordinates were reconstructed historically, not captured; projection withheld.";return;}
    const branches=Object.entries(row.pca_coords||{}).filter(([branch,points])=>branch!=="VISIT"&&Array.isArray(points)&&points.length);
    const points=branches.flatMap(([,values])=>values).filter((point)=>point.length>=2&&point.every(Number.isFinite));
    if(!points.length){svg.innerHTML=`<text x="190" y="86" text-anchor="middle" font-size="11" fill="#64748b">No recorded projection</text>`;if(note)note.textContent="No coordinates are available for the selected trace.";return;}
    const xs=points.map((p)=>p[0]),ys=points.map((p)=>p[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const x=(value)=>28+(value-minX)/(maxX-minX||1)*324,y=(value)=>151-(value-minY)/(maxY-minY||1)*126;
    const palette={A:"#4338ca",B:"#0284c7",C:"#7c3aed"};
    svg.innerHTML=`<line x1="20" y1="157" x2="364" y2="157" stroke="#e2e8f0"/><line x1="24" y1="12" x2="24" y2="161" stroke="#e2e8f0"/>`+branches.map(([branch,values])=>{const color=palette[branch]||"#475569",valid=values.filter((point)=>point.length>=2&&point.every(Number.isFinite));return `<polyline points="${valid.map((p)=>`${x(p[0])},${y(p[1])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2" ${branch==="A"?"":'stroke-dasharray="5 4"'}/>`+valid.map((p,index)=>`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${index+1===state.step?6:4}" fill="${color}"/><text x="${x(p[0])+6}" y="${y(p[1])-5}" font-size="8.5" fill="${color}">${esc(branch)}${index+1}</text>`).join("");}).join("");
    const variance=row.pca_explained_variance||[];if(el("pcaVarLabel"))el("pcaVarLabel").textContent=variance.length>=2?`PC1 ${pct(variance[0])} · PC2 ${pct(variance[1])}`:"Explained variance unavailable";
    if(note)note.textContent="Recorded token 1 branch samples. PCA is fit separately per method; axes are not comparable. No visit point is synthesized.";
  }
  function renderTelemetry(row) {
    const grid=el("telemetryGrid"),snap=snapshot(),tel=telemetry(),d=decision();if(!grid)return;
    const measured=tel&&!["M0","M1","C0","R0"].includes(row.method)&&!tel.skipped_unstable;
    if(el("telemetryStepTag"))el("telemetryStepTag").textContent=`Token ${state.token+1} · step ${state.step}`;
    const items=[
      ["Branch A entropy / margin",`${num(snap?.branch_entropy?.A,3)} / ${num(snap?.branch_margin?.A,3)}`],
      ["Controller decision",d?.action_type||"No decision at this step"],
      ["Policy type",row.method==="E3_mlp_ungated"?"Learned scorer; ungated":row.method.startsWith("E3")||row.method==="D_learned"?"Learned scorer + gates":row.method==="E2"||row.method==="H_budget"?"Heuristic adaptive":"Fixed schedule / control"],
      ["A–B relative distance",num(snap?.ab_relative_distance,4)],
      ["Visited residual ‖C‖",measured?num(tel.curvature_frob,3):"Not measured at this step"],
      ["α_eff / β",tel?`${num(tel.effective_alpha,2)} / ${num(tel.beta,2)}`:"Not recorded"],
    ];
    grid.innerHTML=items.map(([key,value])=>`<div class="t-kv"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div></div>`).join("");
  }
  function renderTerminal(row) {
    const main=el("terminalMainText"),bar=el("terminalMethodBadges"),cost=el("terminalCallCost");if(!main||!bar)return;
    const verified=ledgerMatches(row),base=baseline(),entry=token(),d=decision();
    if(cost)cost.textContent=verified?`Recorded calls: ${row.recurrent_block_calls} · scoring: ${row.scoring_recurrent_calls||0} separate`:`Execution cost unverified · raw reference calls: ${row.recurrent_block_calls}`;
    const outcome=!base?"No paired S0 output":row.method==="S0"?"Ordinary recurrence reference":!base.is_correct&&row.is_correct?"Recovery vs S0":base.is_correct&&!row.is_correct?"Degradation vs S0":row.is_correct?"Both correct":"Both incorrect";
    const mixing=pair().M1;
    main.innerHTML=`<div><b>${esc(name(row.method))} [${esc(row.method)}]</b> · decoded <code>${esc(JSON.stringify(row.decoded_text))}</code> → extracted <b>${esc(row.extracted_answer)}</b> · target <b>${esc(row.gold_answer)}</b> <span style="color:${row.is_correct?'#34d399':'#fda4af'};font-weight:800">${row.is_correct?'✓ CORRECT':'✗ INCORRECT'}</span></div><div style="font-size:.76rem;color:#94a3b8;margin-top:.3rem">Selected token action: <code>${esc(d?.action_type||entry?.controller_action_type||"not recorded")}</code> · donor <code>${esc(entry?.selected_donor)}</code> · policy <code>${esc(policy(row.method))}</code>${base?` · S0 answer <code>${esc(base.extracted_answer)}</code> (${base.is_correct?"correct":"incorrect"}${ledgerMatches(base)?`, ${base.recurrent_block_calls} calls`:", cost unverified"})`:""}</div><div>${esc(outcome)}.${base&&!base.is_correct&&row.is_correct&&mixing?.is_correct&&(row.method.startsWith("E1")||row.method.startsWith("E3")||row.method==="E2")?" Linear update mixing (M1) also succeeds here; this is not an exclusive TEBDİL benefit.":""}</div>`;
    const methods=ORDER.filter((method)=>pair()[method]).concat(Object.keys(pair()).filter((method)=>!ORDER.includes(method)));
    bar.innerHTML=methods.map((method)=>{const item=pair()[method];return `<button type="button" class="m-chip ${item.is_correct?'pass':'fail'} ${method===row.method?'active-method':''}" data-method="${esc(method)}" title="Inspect ${esc(name(method))}">${esc(method)}: ${esc(item.extracted_answer)} ${item.is_correct?'✓':'✗'} ${ledgerMatches(item)?`(${item.recurrent_block_calls}c)`:'(cost unverified)'}</button>`;}).join("");
    bar.querySelectorAll("button[data-method]").forEach((item)=>item.addEventListener("click",()=>changeMethod(item.getAttribute("data-method")||state.method)));
  }
  function renderEvidence(row) {
    if(el("runSummary")){const summary=run().methods[row.method];el("runSummary").textContent=`${run().label} · ${run().num_problems} distinct problems × ${run().seeds.length} seed(s) · ${summary.num_instances} paired rows/method · cluster CI [${pp(summary.cluster_ci_95_low)}, ${pp(summary.cluster_ci_95_high)}] · ${run().status||"exploratory"}`;}
    if(el("selectedTraceJson"))el("selectedTraceJson").textContent=JSON.stringify(row,null,2);
    if(selectedDownloadUrl)URL.revokeObjectURL(selectedDownloadUrl);
    selectedDownloadUrl=URL.createObjectURL(new Blob([JSON.stringify(row,null,2)],{type:"application/json"}));
    const traceLink=/** @type {HTMLAnchorElement|null} */(el("downloadTrace")),runLink=/** @type {HTMLAnchorElement|null} */(el("downloadRun")),configLink=/** @type {HTMLAnchorElement|null} */(el("downloadConfig"));
    if(traceLink){traceLink.href=selectedDownloadUrl;traceLink.download=`${row.problem_id}-${row.seed}-${row.method}.json`;}
    if(runLink)runLink.href=run().trace_url;if(configLink)configLink.href=run().manifest_url;
    if(el("evidenceLinks"))el("evidenceLinks").innerHTML=DATA.evidence.map((item)=>`<li><a href="${esc(item.url)}">${esc(item.label)}${item.historical?' · historical audit':''}</a><small> SHA-256 ${esc(item.sha256)}</small></li>`).join("");
  }

  function summaryRows(recording,methods) {
    return methods.filter((method)=>recording.methods[method]).map((method)=>[method,recording.methods[method]]);
  }
  function fillSummaryTable(tableId,recording,methods) {
    const table=el(tableId);if(!(table instanceof HTMLTableElement)||!recording)return;
    const items=summaryRows(recording,methods);
    table.innerHTML=`<caption>${esc(recording.label)} · ${recording.num_problems} distinct problems; intervals resample problem clusters, not individual seeds.</caption><thead><tr><th>Method</th><th>Accuracy</th><th>Δ vs S0</th><th>95% cluster CI</th><th>Rescued</th><th>Degraded</th><th>Mean calls</th></tr></thead><tbody>${items.map(([method,item])=>`<tr class="${method.startsWith("E")||method.startsWith("D_")?'row-ours':''}"><td><b>${esc(name(method))}</b><br><code>${esc(method)}</code></td><td><b>${pct(item.accuracy)}</b></td><td>${pp(item.paired_accuracy_diff)}</td><td>[${pp(item.cluster_ci_95_low)}, ${pp(item.cluster_ci_95_high)}]</td><td>${item.rescued_count} / ${item.base_wrong_count}</td><td>${item.degraded_count} / ${item.base_correct_count}</td><td>${item.ledger_mismatches||["P0","P1"].includes(method)?"Unverified":num(item.mean_recurrent_calls,2)}</td></tr>`).join("")}</tbody>`;
  }
  function renderResultTables() {
    const controller=DATA.runs.phase7_8_controller_test;
    if(el("controllerComparisonSummary")&&controller)el("controllerComparisonSummary").innerHTML=`<div><b>Controller recording: ${controller.num_problems} problems × ${controller.seeds.length} seeds</b></div>`+["S0","E1","E2","E3_mlp"].map((method)=>{const item=controller.methods[method];return `<div>• ${esc(name(method))} (${method}): ${pct(item.accuracy)} · ${item.rescued_count} rescued / ${item.degraded_count} degraded · ${item.ledger_mismatches?"unverified cost":num(item.mean_recurrent_calls,2)+" mean calls"}</div>`;}).join("");
    const current=run();if(el("selectedResultsTitle"))el("selectedResultsTitle").textContent=`1. Selected recording — ${current.label}`;
    fillSummaryTable("phase78TestTable",current,Object.keys(current.methods));
    fillSummaryTable("phase5DonorTable",DATA.runs.phase5_donor,["S0","D_farthest","D_random","D_confidence","D_learned","D_nearest","D_oracle"]);
    fillSummaryTable("phase6SelectiveTable",DATA.runs.phase6_selective,["S0","E1","E1_subspace","E1_coord_top50","E1_norm_tight","E1_last_token","E1_gated","E3_ridge"]);
    fillSummaryTable("phase3ValidationTable",DATA.runs.phase3_ablations,["S0","S1","P0","P1","M0","M1","C0","R0","E0","E1_no_curv","E1_rand_curv","E1"]);
    renderFamilyTable();renderCostPlot(current);
  }
  function renderFamilyTable() {
    const table=el("familyBreakdownTable");if(!(table instanceof HTMLTableElement))return;
    const families=[...new Set(rows.map((row)=>row.family))].sort(); const methods=Object.keys(run().methods);
    table.innerHTML=`<caption>Descriptive accuracy by task family in the selected recording. Seeds are repeated measurements.</caption><thead><tr><th>Method</th>${families.map((family)=>`<th>${esc(family)}</th>`).join("")}<th>Overall</th></tr></thead><tbody>${methods.map((method)=>`<tr><td><b>${esc(method)}</b></td>${families.map((family)=>{const selected=rows.filter((row)=>row.method===method&&row.family===family);return `<td>${selected.length?pct(selected.filter((row)=>row.is_correct).length/selected.length):"—"}</td>`;}).join("")}<td>${pct(run().methods[method].accuracy)}</td></tr>`).join("")}</tbody>`;
    if(el("familyNote"))el("familyNote").textContent="Descriptive family slices are small and exploratory; no family-level generalization claim follows.";
  }
  function renderCostPlot(recording) {
    const svg=el("paretoChartSvg");if(!(svg instanceof SVGElement))return;
    const points=Object.entries(recording.methods).filter(([method,item])=>!item.ledger_mismatches&&!["P0","P1"].includes(method)&&Number.isFinite(item.mean_recurrent_calls)&&Number.isFinite(item.accuracy));
    if(!points.length){svg.replaceChildren();return;}
    const calls=points.map(([,item])=>item.mean_recurrent_calls),accuracies=points.map(([,item])=>item.accuracy),minX=Math.min(...calls),maxX=Math.max(...calls),minY=Math.min(...accuracies),maxY=Math.max(...accuracies);
    const x=(value)=>55+(value-minX)/(maxX-minX||1)*350,y=(value)=>215-(value-minY)/(maxY-minY||1)*170;
    svg.innerHTML=`<rect x="55" y="25" width="350" height="190" fill="#f8fafc" stroke="#e2e8f0"/>${[0,.5,1].map((fraction)=>`<text x="47" y="${215-170*fraction+3}" text-anchor="end" font-size="9" fill="#475569">${pct(minY+(maxY-minY)*fraction)}</text><text x="${55+350*fraction}" y="230" text-anchor="middle" font-size="9" fill="#475569">${num(minX+(maxX-minX)*fraction,1)}</text>`).join("")}<text x="55" y="16" font-size="10" fill="#334155">Accuracy</text><text x="230" y="250" text-anchor="middle" font-size="10" fill="#334155">Verified mean recurrent calls</text>`+points.map(([method,item])=>`<circle cx="${x(item.mean_recurrent_calls)}" cy="${y(item.accuracy)}" r="5" fill="${method.startsWith("E")?'#059669':'#4338ca'}"/><text x="${x(item.mean_recurrent_calls)+(x(item.mean_recurrent_calls)>350?-7:7)}" text-anchor="${x(item.mean_recurrent_calls)>350?'end':'start'}" y="${y(item.accuracy)-6}" font-size="9" fill="#334155">${esc(method)}</text>`).join("");
    if(el("costPlotNote"))el("costPlotNote").textContent="Scatter of methods whose execution ledgers are verified. No hard-coded frontier is drawn; parallel-selection reference budgets are excluded.";
  }

  function renderExplorer() {
    const family=select("explorerFamilyFilter")?.value||"all";
    const previous=select("explorerMethodFilter")?.value||"all";
    const available=[...new Set(rows.map((row)=>row.method))];
    setOptions(select("explorerMethodFilter"),[["all","All methods"],...available.map((method)=>/** @type {[string,string]} */([method,`${name(method)} · ${method}`]))],available.includes(previous)?previous:"all");
    const method=select("explorerMethodFilter")?.value||"all";
    const filtered=rows.filter((row)=>(family==="all"||row.family===family)&&(method==="all"||row.method===method));
    if(el("explorerCountBadge"))el("explorerCountBadge").textContent=`Showing ${filtered.length} of ${rows.length} recorded traces`;
    const body=el("traceExplorerBody");if(!body)return;
    body.innerHTML=filtered.map((row)=>{
      const base=pairs.get(`${row.problem_id}|${row.seed}`)?.S0;
      const status=base&&!base.is_correct&&row.is_correct?"RESCUED":base?.is_correct&&!row.is_correct?"DEGRADED":row.is_correct?"Correct":"Incorrect";
      return `<tr><td><code>${esc(row.problem_id)}</code></td><td>${esc(row.family)}</td><td>${row.seed}</td><td><b>${esc(row.method)}</b></td><td>${esc(row.gold_answer)}</td><td>${esc(row.extracted_answer)}</td><td>${status}</td><td>${["M0","M1","C0","R0"].includes(row.method)?"Not measured":num(row.curvature_frob,2)}</td><td>${num(row.delta_gold_logprob,4)}</td><td>${ledgerMatches(row)?row.recurrent_block_calls:"Unverified"}</td><td><button type="button" class="stage-btn load-trace-btn" data-problem="${esc(row.problem_id)}" data-seed="${row.seed}" data-method="${esc(row.method)}">Inspect ↑</button></td></tr>`;
    }).join("");
    body.querySelectorAll(".load-trace-btn").forEach((item)=>item.addEventListener("click",()=>inspectCase(state.run,item.getAttribute("data-problem"),item.getAttribute("data-seed"),item.getAttribute("data-method"))));
  }
  async function inspectCase(runId,problem,seed,method) {
    stopAutoLoop();state.problem=problem;state.seed=seed;state.method=method;state.token=0;state.step=2;
    if(state.run!==runId||!rows.length){state.run=runId;if(select("runSelect"))select("runSelect").value=runId;await loadRun();}
    else{if(select("caseSelect"))select("caseSelect").value=`${problem}|${seed}`;syncMethods();renderAll();}
    el("demo")?.scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth"});
    select("caseSelect")?.focus({preventScroll:true});
  }

  let grid=[],dimensions=[],auxiliaryPromise;
  async function fetchVerifiedJson(url,expectedHash) {
    const response=await fetch(url);if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const bytes=await response.arrayBuffer();
    if(!globalThis.crypto?.subtle)throw new Error("Secure browser context required for checksum verification");
    const hash=[...new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))].map((b)=>b.toString(16).padStart(2,"0")).join("");
    if(hash!==expectedHash)throw new Error("Recording checksum mismatch");
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  async function loadAuxiliaryEvidence() {
    if(!auxiliaryPromise)auxiliaryPromise=Promise.all(["grid_summary.json","dimension_ablations.json"].map(async(filename)=>{
      const item=DATA.evidence.find((entry)=>entry.url.endsWith("/"+filename));
      if(!item)throw new Error(`Evidence unavailable: ${filename}`);
      return fetchVerifiedJson(item.url,item.sha256);
    })).then(([sweep,ablation])=>{grid=sweep.grid;dimensions=ablation.ablations;});
    try{
      await auxiliaryPromise;
      renderHeatmap(select("gridMetricSelect")?.value||"mean_curvature_frob");
      const table=el("dimAblationTable");if(table)table.innerHTML=`<caption>Historical nine-problem schedule/scope subset. Descriptive, not an optimality claim.</caption><thead><tr><th>Visit step</th><th>Token scope</th><th>Mean ‖C‖</th><th>Accuracy</th><th>Problems</th></tr></thead><tbody>${dimensions.map((row)=>`<tr><td>${row.visit_step}</td><td>${esc(row.token_scope)}</td><td>${num(row.mean_curvature_frob,2)}</td><td>${pct(row.accuracy)}</td><td>${row.num_problems}</td></tr>`).join("")}</tbody>`;
    }catch(error){auxiliaryPromise=null;const message=error instanceof Error?error.message:String(error);if(el("alphaBetaHeatmap"))el("alphaBetaHeatmap").textContent=`Grid unavailable: ${message}`;if(el("dimAblationTable"))el("dimAblationTable").innerHTML='<caption>Ablation recording unavailable. No values have been substituted.</caption>';}
  }
  function renderHeatmap(metricKey) {
    const target=el("alphaBetaHeatmap");if(!target||!grid.length)return;
    const alphas=[...new Set(grid.map((item)=>item.alpha))].sort((a,b)=>a-b),betas=[...new Set(grid.map((item)=>item.beta))].sort((a,b)=>a-b);
    const values=grid.map((item)=>item[metricKey]).filter(Number.isFinite),max=Math.max(...values.map(Math.abs),1e-9);
    target.style.gridTemplateColumns=`repeat(${betas.length+1},minmax(0,1fr))`;
    target.innerHTML='<div class="hm-head">α / β</div>'+betas.map((b)=>`<div class="hm-head">β=${num(b,2)}</div>`).join("")+alphas.map((a)=>`<div class="hm-head">α=${num(a,2)}</div>`+betas.map((b)=>{
      const value=grid.find((item)=>item.alpha===a&&item.beta===b)?.[metricKey],ratio=Number.isFinite(value)?Math.min(Math.abs(value)/max,1):0;
      const background=!Number.isFinite(value)||value===0?"#f8fafc":value>0?`rgba(67,56,202,${.1+.72*ratio})`:`rgba(225,29,72,${.1+.65*ratio})`;
      return `<div class="hm-cell" style="background:${background};color:${ratio>.52?'#fff':'#0f172a'}" title="α=${a}, β=${b}; ${esc(metricKey)}=${num(value,6)}">${Number.isFinite(value)?num(value,metricKey==='mean_curvature_frob'?2:4):'Not recorded'}</div>`;
    }).join("")).join("");
  }
  function initTocAndCopy() {
    const rail=el("tocRail"),toggle=button("tocToggle");
    if(rail&&toggle){
      // Keep mobile navigation compact; the original floating rail remains the navigation surface.
      if(innerWidth<1180){rail.classList.add("is-collapsed");toggle.setAttribute("aria-expanded","false");el("tocLinks")?.setAttribute("inert","");}
      toggle.addEventListener("click",()=>{const collapsed=rail.classList.toggle("is-collapsed");toggle.setAttribute("aria-expanded",String(!collapsed));el("tocLinks")?.toggleAttribute("inert",collapsed);});
    }
    const links=Array.from(document.querySelectorAll(".toc-link[href^='#']"));
    const sections=links.map((link)=>document.querySelector(link.getAttribute("href"))).filter(Boolean);
    if("IntersectionObserver"in window){const observer=new IntersectionObserver((entries)=>entries.forEach((entry)=>{if(entry.isIntersecting)links.forEach((link)=>link.classList.toggle("is-active",link.getAttribute("href")==="#"+entry.target.id));}),{rootMargin:"-25% 0px -60% 0px",threshold:.01});sections.forEach((section)=>observer.observe(section));}
    button("copyBibBtn")?.addEventListener("click",async()=>{const target=button("copyBibBtn");try{await navigator.clipboard.writeText(el("bibtexCode")?.textContent||"");target.textContent="Copied!";}catch{target.textContent="Copy unavailable — select the text below";} });
    document.querySelectorAll(".load-case-btn").forEach((item)=>item.addEventListener("click",()=>inspectCase(item.getAttribute("data-run"),item.getAttribute("data-problem"),item.getAttribute("data-seed"),item.getAttribute("data-method"))));
  }
  initControls();
  renderResultTables();
  loadRun();
})();
