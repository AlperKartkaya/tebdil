// Optional browser verification; uses an existing Playwright installation, not a site runtime dependency.
const {chromium}=require(process.env.TEBDIL_PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs');
const base=process.env.SITE_URL||'http://127.0.0.1:8093/';
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
 const errors=[],external=[],failed=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(!r.url().startsWith(new URL(base).origin)&&!r.url().startsWith('data:'))external.push(r.url())});
 page.on('response',r=>{if(r.status()>=400)failed.push([r.status(),r.url()])});
 await page.goto(base,{waitUntil:'networkidle'});
 await page.waitForFunction(()=>document.querySelector('#loadStatus')?.textContent.includes('SHA-256 verified'));
 await page.screenshot({path:'/tmp/tebdil-final-desktop-hero.png'});
 await page.locator('#excursionStageSvg').scrollIntoViewIfNeeded();
 await page.screenshot({path:'/tmp/tebdil-final-desktop-stage.png'});
 const manifest=await page.evaluate(()=>window.TEBDIL_DATA);
 const checks=[];
 for(const [id,run]of Object.entries(manifest.runs)){
  await page.selectOption('#runSelect',id);
  await page.waitForFunction(count=>document.querySelector('#loadStatus').textContent.startsWith(count.toLocaleString()+' recorded traces'),run.record_count);
  assert.equal(await page.locator('#explorerSplitFilter').inputValue(),id);
  assert.equal(await page.locator('#traceExplorerBody tr').count(),run.record_count);
  const selected=await page.locator('#methodSelect').inputValue();
  const text=await page.locator('#terminalMainText').innerText();
  const raw=JSON.parse(await page.locator('#selectedTraceJson').textContent());
  assert.equal(raw.method,selected); assert(text.includes(raw.extracted_answer));
  assert.equal(await page.locator('#caseSelect').inputValue(),raw.problem_id+'|'+raw.seed);
  assert((await page.locator('#phase78TestTable').innerText()).includes(run.label));
  checks.push(id+': '+run.record_count+' rows');
 }
 await page.selectOption('#runSelect','phase4_adaptive');
 await page.waitForFunction(()=>document.querySelector('#loadStatus').textContent.startsWith('252 recorded'));
 await page.selectOption('#methodSelect','E2');await page.selectOption('#stepSelect','2');
 assert(/CONTINUE|EXCURSION/.test(await page.locator('#telemetryGrid').innerText()));
 await page.selectOption('#runSelect','phase3_ablations');
 await page.waitForFunction(()=>document.querySelector('#traceExplorerBody tr')?.textContent.includes('validation'));
 for(const method of ['M0','M1','C0','R0','S0','E1']){
  await page.selectOption('#methodSelect',method);
  assert.equal(JSON.parse(await page.locator('#selectedTraceJson').textContent()).method,method);
  const narrative=await page.locator('#stageStepNarrative').innerText();
  if(method==='C0')assert(narrative.includes('No update returns to A'));
  if(method==='M1')assert(narrative.includes('does not evaluate'));
  if(['M0','M1','C0','R0'].includes(method))assert((await page.locator('#telemetryGrid').innerText()).includes('Not measured'));
 }
 await page.selectOption('#methodSelect','S1');
 assert((await page.locator('#stepSelect option').allTextContents()).includes('Step 5'));
 await page.selectOption('#stepSelect','5');
 assert((await page.locator('#telemetryStepTag').innerText()).toLowerCase().includes('step 5'));
 await page.selectOption('#methodSelect','E1');
 const tokenValues=await page.locator('#tokenSelect option').evaluateAll(xs=>xs.map(x=>x.value));
 if(tokenValues.length>1){await page.selectOption('#tokenSelect',tokenValues[1]);assert((await page.locator('#pcaNote').innerText()).includes('not borrowed'));}
 if(await page.locator('#tokenSelect').isEnabled())await page.selectOption('#tokenSelect','0');
 await page.selectOption('#stepSelect','1');
 assert((await page.locator('#telemetryGrid').innerText()).includes('No decision at this step'));
 await page.locator('#playLoopBtn').click(); await page.waitForTimeout(1900);assert.notEqual(await page.locator('#stepSelect').inputValue(),'1');await page.locator('#playLoopBtn').click();
 await page.selectOption('#gridMetricSelect','mean_delta_gold_logprob');
 assert.equal(await page.locator('#alphaBetaHeatmap .hm-cell').count(),20);
 assert.equal(await page.locator('#dimAblationTable tbody tr').count(),4);
 await page.locator('.load-case-btn').first().click();
 await page.waitForFunction(()=>JSON.parse(document.querySelector('#selectedTraceJson').textContent).method==='E3_mlp');
 assert.equal(await page.locator('#runSelect').inputValue(),'phase7_8_controller_test');
 assert.equal(await page.locator('#caseSelect').inputValue(),'arith_test_003|42');
 await page.selectOption('#tokenSelect','1');assert((await page.locator('#pcaNote').innerText()).includes('not borrowed'));await page.selectOption('#tokenSelect','0');
 await page.selectOption('#explorerFamilyFilter','graph');
 assert(await page.locator('#traceExplorerBody tr').count()>0);
 assert((await page.locator('#traceExplorerBody').innerText()).includes('graph'));
 assert(!(await page.locator('#traceExplorerBody').innerText()).includes('arithmetic'));
 for(const id of ['architecture','results','scaling','case-study']){
  await page.locator('#'+id).scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/tebdil-final-'+id+'.png'});
 }
 const widths=[];
 for(const width of [390,320,768,1024,1440]){
  await page.setViewportSize({width,height:844});await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth),0,'body overflow at '+width);
  widths.push(width);
  if(width===390)await page.screenshot({path:'/tmp/tebdil-final-mobile-hero.png'});
 }
 await page.reload({waitUntil:'networkidle'});
 await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.className),'skip-link');
 await page.keyboard.press('Enter');assert.equal(await page.evaluate(()=>document.activeElement.id),'main-content');
 await page.waitForFunction(()=>document.querySelectorAll('mjx-container').length>100);assert.equal(await page.locator('mjx-merror').count(),0);
 // A failed or tampered recording must not leave the previous recording visible.
 const isolated=await browser.newPage();
 await isolated.route('**/static/data/**',r=>r.fulfill({status:503,body:'unavailable'}));
 await isolated.goto(base,{waitUntil:'networkidle'});
 await isolated.waitForFunction(()=>document.querySelector('#loadStatus').textContent.includes('Recording unavailable'));
 assert(await isolated.locator('.stage-grid').isHidden());assert(await isolated.locator('#retryLoad').isVisible());
 assert(await isolated.locator('#caseSelect').isDisabled());
 await isolated.unroute('**/static/data/**');await isolated.click('#retryLoad');
 await isolated.waitForFunction(()=>document.querySelector('#loadStatus').textContent.includes('SHA-256 verified'));
 assert(await isolated.locator('.stage-grid').isVisible());
 await isolated.route('**/static/data/**',r=>r.fulfill({status:200,body:'[]'}));
 await isolated.selectOption('#runSelect','phase1_smoke');
 await isolated.waitForFunction(()=>document.querySelector('#loadStatus').textContent.includes('checksum mismatch'));
 assert(await isolated.locator('.output-terminal').isHidden());assert((await isolated.locator('#runSummary').innerText()).includes('checksum mismatch'));
 assert((await isolated.locator('.metric-num').first().innerText()).includes('Unavailable'));
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);assert.deepEqual(failed,[]);
 const result={base,checks,widths,errors,external,failed,math:await page.locator('mjx-container').count(),failureRetryAndChecksum:'pass',keyboard:'pass',controls:'pass'};
 fs.writeFileSync('/tmp/tebdil-original-restoration-browser.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
