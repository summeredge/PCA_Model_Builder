"""Opt-in screening workflow check using the existing browser runtime settings."""

import hashlib
import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web_model_results as ui


def test_screening_layout_and_unchanged_calculation_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    rng = np.random.default_rng(42)
    n = 480
    x = np.r_[rng.normal(-3, .2, n // 2), rng.normal(3, .2, n // 2)]
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="5min"), "TAG_A": x,
                          "TAG_B": 2 * x + rng.normal(0, .1, n), "TAG_C": rng.normal(0, 1, n),
                          "PERFORMANCE": rng.normal(50, 5, n)})
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[],reports=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 try {
  const response=await page.goto(process.env.SCREENING_TEST_URL);await page.evaluate(()=>document.fonts.ready);
  assert.equal(crypto.createHash('sha256').update(await response.body()).digest('hex'),process.env.SCREENING_HTML_SHA256);
  const nav=page.locator('#candidateSectionNav');
  assert(!await nav.isVisible());
  assert.equal(await page.locator('meta[name="pca-model-builder-build"]').getAttribute('content'),process.env.SCREENING_BUILD_ID);
  async function action(id,endpoint) {
   const pending=page.waitForResponse(r=>r.url().endsWith(endpoint));
   await page.locator('#'+id).click();const response=await pending;
   const data=await response.json();assert(response.ok(),JSON.stringify(data));return data;
  }
  async function autoSummary(change) {
   const pending=page.waitForResponse(r=>r.url().endsWith('/api/modeling-eligibility'));
   await change();const response=await pending;assert(response.ok(),await response.text());
   await page.waitForFunction(()=>document.querySelector('#eligibilitySummary').textContent.startsWith('分析时间范围：'));
  }
  await page.locator('#fileInput').setInputFiles(process.env.SCREENING_TEST_CSV);
  await action('uploadButton','/api/upload');await page.locator('#timestampColumn').selectOption('time');
  await action('inspectButton','/api/inspect');
  await page.locator('[data-panel="candidatePanel"]').click();
  await page.locator('[data-panel="stateExplorationPanel"]').click();
  assert(await nav.isVisible());assert.equal(await nav.locator('a').count(),4);
  await page.locator('#addEligibilityKeep').click();
  const keep=page.locator('#eligibilityKeepConditions .eligibility-condition');
  await keep.locator('select').selectOption('TAG_A');await keep.locator('[data-field="minimum"]').fill('-10');
  assert.equal(await page.locator('#refreshEligibilitySummary').count(),0);
  await autoSummary(()=>keep.locator('[data-field="maximum"]').fill('10'));
  assert((await page.locator('#eligibilitySummary').textContent()).includes('480'));
  await page.locator('#addEligibilityExcludeGroup').click();
  const exclude=page.locator('#eligibilityExcludeGroups .eligibility-condition');
  await exclude.locator('select').selectOption('TAG_A');await exclude.locator('[data-field="minimum"]').fill('-10');
  await autoSummary(()=>exclude.locator('[data-field="maximum"]').fill('10'));
  assert(await page.locator('#eligibilitySummary').evaluate(e=>e.classList.contains('warning')));
  await autoSummary(()=>exclude.getByRole('button',{name:'删除条件',exact:true}).click());
  await autoSummary(()=>keep.getByRole('button',{name:'删除条件',exact:true}).click());
  assert.equal(await page.locator('.eligibility-condition').count(),0);
  assert(!await page.locator('#eligibilitySummary').evaluate(e=>e.classList.contains('warning')||e.classList.contains('notice')));
  const start=await page.locator('#analysisStart').inputValue(),end=await page.locator('#analysisEnd').inputValue();
  await autoSummary(()=>page.locator('#analysisStart').fill('2026-01-01T01:00'));
  assert((await page.locator('#eligibilitySummary').innerText()).includes('468'));
  await autoSummary(()=>page.locator('#analysisEnd').fill('2026-01-02T12:00'));
  assert((await page.locator('#eligibilitySummary').innerText()).includes('421'));
  await autoSummary(()=>page.locator('#analysisStart').fill(start));await autoSummary(()=>page.locator('#analysisEnd').fill(end));
  await page.route('**/api/modeling-eligibility',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({error:'摘要测试失败'})}));
  await page.locator('#analysisStart').fill('2026-01-01T01:00');
  await page.waitForFunction(()=>document.querySelector('#eligibilitySummary').classList.contains('warning')&&document.querySelector('#eligibilitySummary').textContent.includes('摘要测试失败'));
  await page.unroute('**/api/modeling-eligibility');await autoSummary(()=>page.locator('#analysisStart').fill(start));
  await page.locator('#sampleInterval').fill('10');await page.locator('#resamplingMethod').selectOption('mean');
  await page.locator('#filterMethod').selectOption('first_order');await page.locator('#firstOrderAlpha').fill('0.2');
  assert(await page.locator('#firstOrderAlpha').isEnabled());
  await page.locator('#filterMethod').selectOption('trailing_mean');await page.locator('#smoothingWindow').fill('15');
  assert(await page.locator('#smoothingWindow').isEnabled());
  await page.locator('#smoothingWindow').fill('10');await page.locator('#filterMethod').selectOption('none');
  await page.locator('#sampleInterval').fill('5');await page.locator('#resamplingMethod').selectOption('none');
  await page.locator('#maxLag').fill('10');await page.locator('#lagStep').fill('5');
  await page.locator('#explorationClusterCount').fill('2');await page.locator('#explorationMinimumDuration').fill('30');
  await page.locator('#explorationCandidateCount').fill('3');await page.locator('#explorationMaximumPlotPoints').fill('1200');
  await page.locator('#explorationPerformanceTag').selectOption('PERFORMANCE');
  await page.locator('#explorationPerformanceDirection').selectOption('target_range');
  await page.locator('#explorationTargetMin').fill('45');await page.locator('#explorationTargetMax').fill('55');
  const data=await action('stateExplorationButton','/api/state-exploration/run');
  await page.waitForFunction(()=>!document.querySelector('#explorationContent').hidden);
  const comparisonKeys=['exploration_config','performance_config','preprocessing_summary','exploratory_model_summary',
    'cluster_quality','variable_diagnostics','cluster_centers','full_point_count','returned_point_count','cluster_summaries','cluster_series_full','data_usage'];
  if(process.env.SCREENING_BASELINE_JSON) {
   const baseline=JSON.parse(fs.readFileSync(process.env.SCREENING_BASELINE_JSON,'utf8'));
   for(const key of comparisonKeys)assert.deepEqual(data[key],baseline[key],key);
  }
  assert.equal(await page.locator('.screening-kpis .metric').count(),6);
  assert.equal(await page.locator('#explorationOverview .metric').count(),6);
  assert.equal(await page.locator('#explorationClusterQuality .screening-kpis h3').count(),0);
  assert.equal(await page.locator('#explorationEvidence').innerText(),'结果预览');
  const expectedKpis=[
   ['完整样本',String(data.full_point_count)],
   ['有效覆盖率',`${(data.preprocessing_summary.effective_coverage_ratio*100).toFixed(1)}%`],
   ['工况组数量',String(data.cluster_quality.cluster_count)],
   ['Silhouette Score',typeof data.cluster_quality.silhouette_score==='number'?data.cluster_quality.silhouette_score.toFixed(3):'—'],
   ['保留主元数',String(data.cluster_quality.n_components??'—')],
   ['累计解释率',typeof data.cluster_quality.cumulative_explained_variance==='number'?`${(data.cluster_quality.cumulative_explained_variance*100).toFixed(1)}%`:'—']
  ];
  assert.deepEqual(await page.locator('#explorationOverview .metric').evaluateAll(nodes=>nodes.map(node=>[node.querySelector('span').innerText,node.querySelector('strong').innerText])),expectedKpis);
  assert((await page.locator('.screening-kpis').innerText()).includes('人工确认'));
  if(data.cluster_quality.silhouette_score<.25) assert((await page.locator('.screening-kpis').innerText()).includes('分离度较弱'));
  assert.equal(await page.locator('#explorationClusterDetails').evaluate(node=>node.open),false);
  assert.equal(await page.locator('#explorationClusterDetails summary span').innerText(),String(data.cluster_summaries.length));
  assert.equal(await page.locator('.exploration-candidate-details').count(),3);
  assert.deepEqual(await page.locator('.exploration-candidate-details').evaluateAll(nodes=>nodes.map(node=>node.open)),[false,false,false]);
  assert.equal(await page.locator('.exploration-candidate-details').nth(0).locator('summary span').innerText(),String(data.cluster_candidates.length));
  assert.equal(await page.locator('.exploration-candidate-details').nth(1).locator('summary span').innerText(),String(data.performance_candidates.length));
  assert.equal(await page.locator('.exploration-candidate-details').nth(2).locator('summary span').innerText(),String(data.preferred_region_candidates.length));
  assert.equal(await page.locator('.variable-diagnostics > details').evaluate(node=>node.open),false);
  const diagnosticTitle=await page.locator('#variableDiagnosticsTitle').innerText();
  for(const count of [data.variable_diagnostics.summary.tag_count,data.variable_diagnostics.summary.high_correlation_pair_count,data.variable_diagnostics.summary.attention_tag_count]) assert(diagnosticTitle.includes(String(count)));
  const summaryGeometry=await page.locator('.screening-kpis').evaluate(node=>({gap:parseFloat(getComputedStyle(node).rowGap),paragraphMargins:[...node.querySelectorAll(':scope > p')].map(p=>[getComputedStyle(p).marginTop,getComputedStyle(p).marginBottom])}));
  assert(summaryGeometry.gap<=8,JSON.stringify(summaryGeometry));
  assert(summaryGeometry.paragraphMargins.flat().every(value=>value==='0px'),JSON.stringify(summaryGeometry));
  await page.evaluate(async()=>Plotly.relayout(state.preferredRegionPlot,{shapes:[{type:'circle',xref:'x',yref:'y',x0:-.5,x1:.5,y0:-.5,y1:.5,fillcolor:'rgba(23,107,135,.08)',line:{color:'#176b87',width:2}}]}));
  await page.locator('#explorationQualityDetails').evaluate(node=>node.closest('details').open=true);
  const technicalText=await page.locator('#explorationQualityDetails').innerText();
  assert(technicalText.includes('样本与绘图详情'));
  assert(await page.locator('#explorationClusterQuality .screening-center-heading').isVisible());
  for(const label of ['时间连续性','平均持续时间','最长连续时间','状态切换次数']) assert(!technicalText.includes(label));
  assert.equal(await page.locator('#explorationQualityDetails .screening-evidence-compact .metric').count(),0);
  assert.equal(await page.locator('.screening-temporal-summary').count(),0);
  assert(await page.locator('#explorationClusterTable tr').count()>0);
  const artifacts=process.env.SCREENING_ARTIFACT_DIR||process.env.SCREENING_TEST_DIR;
  const snapshot=()=>page.evaluate(()=>JSON.stringify({config:commonPayload(),selectedModelTags:[...state.selectedModelTags],
    explorationRun:state.exploration?.exploration_run_id,quality:state.exploration?.cluster_quality,
    summaries:state.exploration?.cluster_summaries,decisions:state.exploration?.candidate_decisions,
    windows:[state.candidateWindows,state.excludedWindows,state.trainingWindows],registry:state.registry,
    stages:[...document.querySelectorAll('.workflow-step')].map(e=>[e.dataset.panel,e.className,e.getAttribute('aria-selected'),e.innerText]),
    tools:[...document.querySelectorAll('.candidate-tool-tab')].map(e=>[e.dataset.panel,e.getAttribute('aria-selected')]),
    fields:[...document.querySelectorAll('#candidatePanel input,#candidatePanel select')].map(e=>[e.id,e.value,e.checked])}));
  async function checkNavigation() {
   const before=await snapshot(),requestCount=requests.length;
   for(let index=0;index<4;index++) {
    const link=nav.locator('a').nth(index),id=(await link.getAttribute('href')).slice(1);
    // Native anchors retain keyboard activation. No tool/stage selection accompanies the jump.
    if(index===2) {await link.focus();await link.press('Enter');} else await link.click();
    await page.waitForFunction(({id,index})=>{
     const target=document.getElementById(id).getBoundingClientRect();
     const current=document.querySelector('#candidateSectionNav a[aria-current="location"]');
     const expected=document.querySelectorAll('#candidateSectionNav a')[index];
     return current===expected&&target.top>=0&&target.bottom<=innerHeight&&
       (target.top<=48||Math.abs(scrollY+innerHeight-document.documentElement.scrollHeight)<=2);
    },{id,index});
    assert.equal(await snapshot(),before);assert.equal(requests.length,requestCount);
   }
   // Manual scrolling selects the area; a 48–80px dead band prevents boundary jitter.
   async function position(top) {
    await page.evaluate(top=>window.scrollTo({top:scrollY+document.getElementById('candidateAnalysis').getBoundingClientRect().top-top,behavior:'instant'}),top);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   }
   await position(40);
   assert.equal(await nav.locator('[aria-current="location"]').innerText(),'状态分析');
   for(const top of [60,68,62,70]) {await position(top);assert.equal(await nav.locator('[aria-current="location"]').innerText(),'状态分析');}
   await position(90);assert.equal(await nav.locator('[aria-current="location"]').innerText(),'筛选准备');
   assert.equal(await snapshot(),before);assert.equal(requests.length,requestCount);
   const geometry=await nav.evaluate(e=>{
    const items=[...e.querySelectorAll('a')].map(a=>a.getBoundingClientRect()),r=e.getBoundingClientRect();
    return {outside:items.some(a=>a.left<r.left||a.right>r.right),overlap:items.some((a,i)=>items.slice(i+1).some(b=>Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)))};
   });
   assert.deepEqual(geometry,{outside:false,overlap:false});
  }
  for(const [width,zoom] of [[1440,1],[900,1],[390,1],[1440,2]]) {
   await page.setViewportSize({width,height:1000});
   await page.evaluate(zoom=>{document.body.style.zoom=zoom;},zoom);
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   await page.waitForFunction(()=>{
    const timeline=document.querySelector('#explorationTimeline'),svg=timeline?.querySelector('svg'),plot=state.preferredRegionPlot;
    const viewBox=svg?.getAttribute('viewBox')?.trim().split(/\s+/).map(Number);
    return plot?._fullLayout?._size&&viewBox&&Math.abs(viewBox[2]-Math.round(timeline.clientWidth))<=1;
   });
   if(zoom===1&&[1440,900,390].includes(width)) {
    await checkNavigation();
    await page.locator('.workflow-sidebar').screenshot({path:path.join(artifacts,`candidate-nav-${width}.png`)});
   }
   const report=await page.evaluate(()=>{
    const rect=e=>e.getBoundingClientRect(),spread=a=>Math.max(...a)-Math.min(...a);
    const fields=[...document.querySelectorAll('#modelingEligibility input,#candidatePanel .shared-preprocessing input,#candidatePanel .shared-preprocessing select,#stateExplorationPanel .exploration-controls input,#stateExplorationPanel .exploration-controls select')].filter(e=>e.getClientRects().length);
    const controls=fields.map(rect),metrics=[...document.querySelectorAll('#explorationOverview .metric')],kpis=metrics.map(rect);
    const dates=['analysisStart','analysisEnd'].map(id=>rect(document.getElementById(id)));
    const evidence=[...document.querySelectorAll('#explorationClusterQuality .screening-evidence-compact')],table=document.querySelector('.screening-center-table');
    const grid=document.querySelector('.exploration-result-grid'),pc=document.querySelector('.exploration-pc-card'),region=document.querySelector('.exploration-region-card'),timelineCard=document.querySelector('.exploration-timeline-card');
    const plotContainer=document.querySelector('#explorationPcChart'),plot=state.preferredRegionPlot,plotNode=plotContainer.querySelector('.js-plotly-plot'),overlay=plotContainer.querySelector('.exploration-pc-overlay');
    const modebar=plotContainer.querySelector('.modebar-container'),legend=plotContainer.querySelector('.legend'),modebarRect=modebar?.getBoundingClientRect(),legendRect=legend?.getBoundingClientRect();
    const overlaps=modebarRect&&legendRect&&Math.min(modebarRect.right,legendRect.right)>Math.max(modebarRect.left,legendRect.left)&&Math.min(modebarRect.bottom,legendRect.bottom)>Math.max(modebarRect.top,legendRect.top);
    const timeline=document.querySelector('#explorationTimeline'),timelineSvg=timeline.querySelector('svg'),viewBox=timelineSvg.getAttribute('viewBox').trim().split(/\s+/).map(Number);
    const kpiRows=[...new Set(kpis.map(r=>Math.round(r.top)))],kpiColumns=Math.max(...kpiRows.map(top=>kpis.filter(r=>Math.round(r.top)===top).length)),sameRow=Math.abs(rect(pc).top-rect(region).top)<1;
    return {width:innerWidth,zoom:document.body.style.zoom,
     evidenceHeight:evidence.map(e=>rect(e).height),centerTableWidth:rect(table).width,
     evidenceWidth:rect(evidence[0]).width,centerTableOffset:rect(table).left-rect(evidence[0]).left,
     configHeight:rect(document.getElementById('stateExplorationButton')).bottom-rect(document.getElementById('modelingEligibility')).top,
     overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
     dateWidthSpread:spread(dates.map(r=>r.width)),
     summaryStyle:metrics.map(node=>{const css=getComputedStyle(node);return {minHeight:css.minHeight,padding:css.padding,border:css.borderWidth,background:css.backgroundColor,font:parseFloat(getComputedStyle(node.querySelector('strong')).fontSize)};}),
     summaryOverlap:kpis.some((a,i)=>kpis.slice(i+1).some(b=>Math.min(a.right,b.right)>Math.max(a.left,b.left)+1&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+1)),
     summaryEscapes:kpis.some(r=>r.left<rect(document.querySelector('#explorationOverview')).left-1||r.right>rect(document.querySelector('#explorationOverview')).right+1),
     kpiCount:kpis.length,kpiRows:kpiRows.length,kpiColumns,
     resultGridWidth:rect(grid).width,pcWidth:rect(pc).width,pcHeight:rect(plotContainer).height,plotNodeWidth:rect(plotNode).width,plotNodeHeight:rect(plotNode).height,
     plotWidth:plot._fullLayout.width,plotHeight:plot._fullLayout.height,plotAreaHeight:plot._fullLayout._size.h,shapeCount:plot._fullLayout.shapes.length,overlayWidth:rect(overlay).width,overlayHeight:rect(overlay).height,
     regionWidth:rect(region).width,regionHeight:rect(region).height,sameRow,heightSpread:Math.abs(rect(pc).height-rect(region).height),
     timelineWidth:rect(timeline).width,timelineClientWidth:timeline.clientWidth,timelineCardWidth:rect(timelineCard).width,timelineHeight:rect(timelineSvg).height,timelineViewBoxWidth:viewBox[2],timelineColumnStart:getComputedStyle(timelineCard).gridColumnStart,
     timelineBelow:rect(timelineCard).top>=Math.max(rect(pc).bottom,rect(region).bottom)-1,
     modebarCount:modebar?.querySelectorAll('.modebar-btn').length||0,legendPresent:Boolean(legend),modebarLegendOverlap:overlaps,
     clipped:fields.filter(e=>{const r=rect(e),p=rect(e.parentElement);return r.left<p.left-1||r.right>p.right+1||r.width<100;}).map(e=>e.id),
     overlap:controls.some((a,i)=>controls.slice(i+1).some(b=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1))};
   });
   assert.equal(report.overflow,0,JSON.stringify(report));assert(!report.overlap,JSON.stringify(report));assert.deepEqual(report.clipped,[],JSON.stringify(report));
   assert(report.centerTableWidth<report.evidenceWidth,JSON.stringify(report));assert.equal(report.centerTableOffset,0);
   assert(report.dateWidthSpread<=1,JSON.stringify(report));
   assert(!report.summaryOverlap&&!report.summaryEscapes,JSON.stringify(report));
   assert(report.summaryStyle.every(css=>css.minHeight==='0px'&&css.padding==='0px'&&css.border==='0px'&&css.background==='rgba(0, 0, 0, 0)'&&css.font<=14),JSON.stringify(report));
   assert.equal(report.kpiCount,6,JSON.stringify(report));assert(report.timelineBelow,JSON.stringify(report));
   assert(Math.abs(report.timelineWidth-report.resultGridWidth)<=1,JSON.stringify(report));
   assert(Math.abs(report.timelineHeight-112*Number(zoom))<=1,JSON.stringify(report));assert(Math.abs(report.timelineViewBoxWidth-report.timelineClientWidth)<=1,JSON.stringify(report));
   assert(report.plotNodeWidth>0&&report.plotNodeHeight>0&&report.plotAreaHeight>200,JSON.stringify(report));
   assert.equal(report.shapeCount,1,JSON.stringify(report));
   assert(Math.abs(report.overlayWidth-report.pcWidth)<=2*Number(zoom)+1&&Math.abs(report.overlayHeight-report.pcHeight)<=2*Number(zoom)+1,JSON.stringify(report));
   assert(report.modebarCount>=3&&report.legendPresent&&!report.modebarLegendOverlap,JSON.stringify(report));
   assert(report.kpiRows>=1&&report.kpiRows<=report.kpiCount,JSON.stringify(report));
   if(width>=761&&zoom===1) {
    assert(report.sameRow&&report.heightSpread<=1,JSON.stringify(report));
    assert(report.pcHeight>=420&&report.pcHeight<=480,JSON.stringify(report));
   }
   if(width===390&&zoom===1) assert.equal(report.pcHeight,340,JSON.stringify(report));
   reports.push(report);
   if(width===1440&&zoom===1) {
    await page.screenshot({path:path.join(artifacts,'screening-after.png'),fullPage:true});
    await page.locator('#modelingEligibility').screenshot({path:path.join(artifacts,'screening-conditions.png')});
    await page.locator('#explorationClusterQuality').screenshot({path:path.join(artifacts,'screening-results.png')});
   }
   if(zoom===1&&[900,390].includes(width)) await page.locator('#explorationContent').screenshot({path:path.join(artifacts,`screening-results-${width}.png`)});
  }
  await page.locator('.variable-diagnostics > details > summary').click();
  assert(await page.locator('#explorationVariableDiagnostics .metrics').isVisible());
  await page.locator('.variable-diagnostics > details > summary').click();
  await page.evaluate(()=>{document.body.style.zoom=1;});
  await page.setViewportSize({width:1440,height:1000});
  const centerReports=[];
  const centerCases=[
   {variance:[8,2,1,1],columns:['PC1','PC2'],orientation:'多主元分布明显'},
   {variance:[1,0,9,0],columns:['PC1','PC2','PC3'],orientation:'主要沿 PC3'},
   {variance:[1,1,4,4],columns:['PC1','PC2','PC3','PC4'],orientation:'多主元分布明显'},
   {variance:[1,1,0,9],columns:['PC1','PC2','PC4'],orientation:'主要沿 PC4'},
   {variance:[1,1,2,3,93],columns:['PC1','PC2','PC3','PC4'],orientation:'主要沿 PC5'},
  ];
  for(const width of [1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   for(const example of centerCases) {
    const report=await page.evaluate(({example,quality})=>{
     const coordinates=example.variance.map(Math.sqrt),centers={'cluster_001':coordinates.map(v=>-v),'cluster_002':coordinates};
     const synthetic={...quality,centers:Object.entries(centers).map(([cluster,row])=>({cluster,pc1:row[0],pc2:row[1]}))};
     renderClusterQuality(document.getElementById('explorationClusterQuality'),synthetic,'state_exploration',centers);
     const table=document.querySelector('.screening-center-table'),heading=document.querySelector('.screening-center-heading');
     const cellStyles=[...table.querySelectorAll('tbody tr:first-child td')].map(e=>getComputedStyle(e).textAlign);
     return {width:innerWidth,columns:[...table.querySelectorAll('th')].slice(1).map(e=>e.textContent),
      orientation:heading.querySelector('span').textContent,align:cellStyles,
      overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
      localOverflow:table.scrollWidth-table.clientWidth,tableWidth:table.getBoundingClientRect().width,
      availableWidth:table.parentElement.getBoundingClientRect().width,
      temporal:[...document.querySelectorAll('.screening-temporal-summary dd')].map(e=>e.textContent)};
    },{example,quality:data.cluster_quality});
    assert.deepEqual(report.columns,example.columns);assert.equal(report.orientation,example.orientation);
    assert.deepEqual(report.align,['left',...example.columns.map(()=>'right')]);
    assert.equal(report.overflow,0,JSON.stringify(report));assert(report.tableWidth<=report.availableWidth,JSON.stringify(report));
    assert.deepEqual(report.temporal,[]);
    centerReports.push(report);
    if(width===1440&&example.columns.length===4&&example.variance.length===4) {
     await page.locator('#explorationQualityDetails').screenshot({path:path.join(artifacts,'screening-centers-pc4.png')});
    }
   }
   // A narrow evidence slot must scroll locally, with the keyboard able to reach later columns.
   await page.evaluate(()=>{document.querySelector('.screening-center-table').parentElement.style.width='160px';});
   const local=await page.locator('.screening-center-table').evaluate(e=>({overflow:getComputedStyle(e).overflowX,scroll:e.scrollWidth-e.clientWidth,
    page:document.documentElement.scrollWidth-document.documentElement.clientWidth}));
   assert.equal(local.overflow,'auto');assert(local.scroll>0);assert.equal(local.page,0);
   await page.locator('.screening-center-table').focus();await page.locator('.screening-center-table').press('ArrowRight');
   await page.waitForFunction(()=>document.querySelector('.screening-center-table').scrollLeft>0);
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(quality=>renderClusterQuality(document.getElementById('explorationClusterQuality'),quality,'state_exploration',state.exploration.cluster_centers),data.cluster_quality);
  // Each tool retains its own evidence anchor, even before results are available.
  for(const [tool,evidence] of [['trendPanel','trendEvidence'],['clusterPanel','clusterEvidence'],['performancePanel','performanceEvidence'],['stateExplorationPanel','explorationEvidence']]) {
   await page.locator(`[data-panel="${tool}"]`).click();
   assert.equal(await nav.locator('a').nth(2).getAttribute('href'),'#'+evidence);
   const before=await snapshot(),count=requests.length;
   await nav.locator('a').nth(2).click();
   await page.waitForFunction(id=>{const r=document.getElementById(id).getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&(r.top<=48||Math.abs(scrollY+innerHeight-document.documentElement.scrollHeight)<=2);},evidence);
   assert.equal(await nav.locator('[aria-current="location"]').innerText(),'结果证据');
   assert.equal(await snapshot(),before);assert.equal(requests.length,count);
  }
  // Existing candidate IDs and explicit human confirmation remain the only training path.
  await page.evaluate(()=>renderStateExploration(state.exploration));
  assert.equal(await page.locator('[id="explorationOverview"]').count(),1);
  assert.deepEqual(await page.locator('#explorationOverview .metric').evaluateAll(nodes=>nodes.map(node=>[node.querySelector('span').innerText,node.querySelector('strong').innerText])),expectedKpis);
  assert(data.cluster_candidates.length>0);
  const candidate=data.cluster_candidates[0];
  await page.locator('#explorationClusterQuality [data-profile]').first().click();
  const profileCandidateButton=page.locator('#explorationClusterQuality [data-profile-candidates]').first();
  if(await profileCandidateButton.count()) { await profileCandidateButton.click(); assert(await page.locator('#explorationClusterCandidates').evaluate(node=>node.closest('details').open)); }
  const performanceProfile=data.cluster_quality.group_profiles[0];
  const performanceSummary=data.cluster_summaries.find(item=>item.cluster_id===performanceProfile.cluster_id);
  for(const key of ['performance_valid_count','performance_target_count','performance_target_ratio','performance_median']) assert.equal(performanceProfile[key],performanceSummary[key]);
  const performanceText=await page.locator('[data-profile-performance]').innerText();
  assert(performanceText.includes('性能后验评价')&&performanceText.includes('不参与 PCA 聚类'));
  for(const value of [performanceSummary.performance_valid_count,performanceSummary.performance_target_count,`${(performanceSummary.performance_target_ratio*100).toFixed(1)}%`,String(Number(performanceSummary.performance_median.toPrecision(5)))]) assert(performanceText.includes(String(value)));
  for(const width of [900,1440]) {
   await page.setViewportSize({width,height:1000});
   const geometry=await page.locator('[data-profile-performance]').evaluate(node=>{const r=node.getBoundingClientRect();return {width:r.width,right:r.right,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};});
   assert(geometry.width>0&&geometry.right<=width+1,JSON.stringify(geometry));assert.equal(geometry.overflow,0);
  }
  if(!await page.locator('.exploration-candidate-details').first().evaluate(node=>node.open)) await page.locator('.exploration-candidate-details').first().locator('summary').click();
  assert.deepEqual(await page.locator('#explorationClusterCandidates thead th').allTextContents(),['选择','状态','备注','工况组','时间范围','覆盖时长','样本数','中心距离','稳定性','排名']);
  await page.locator('#explorationClusterCandidates .exploration-candidate-select').first().check();
  await page.locator('#convertExplorationCandidates').click();
  const sourceRef=`state-exploration-${data.exploration_run_id}-${candidate.candidate_id}`;
  assert(await page.evaluate(ref=>state.candidateWindows.some(item=>item.source_ref===ref&&item.source==='cluster'),sourceRef));
  const pending=page.waitForResponse(r=>r.url().endsWith('/api/training-windows'));
  await page.locator('#candidateWindows').getByRole('button',{name:'确认作为训练窗口',exact:true}).last().click();
  assert((await pending).ok());await page.waitForFunction(ref=>state.trainingWindows.some(item=>item.source_ref===ref),sourceRef);
  await page.evaluate(()=>showWorkflowStage('modelPanel'));
  const quality=await action('qualityButton','/api/quality');assert(quality.training_readiness.normal_state.can_train);
  const training=await action('trainButton','/api/train');assert(training.model_quality.training_condition_diagnostic.available);
  const modelReports=[];
  for(const width of [1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   await page.evaluate(()=>document.fonts.ready);
   const report=await page.evaluate(()=>{
    const summary=document.querySelector('#modelMetrics'),items=[...summary.querySelectorAll('.metric')],rect=e=>e.getBoundingClientRect(),bounds=rect(summary);
    const table=document.querySelector('#modelQualitySummary table');
    return {width:innerWidth,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
     metricCount:items.length,summaryHeight:bounds.height,
     escaped:items.some(e=>rect(e).left<bounds.left-1||rect(e).right>bounds.right+1),
     headings:[...table.querySelectorAll('thead th')].map(e=>e.textContent),rows:table.querySelectorAll('tbody tr').length,
     nestedPadding:[...document.querySelectorAll('#modelContent > section,.model-overview-grid > section')].map(e=>getComputedStyle(e).padding)};
   });
   assert.equal(report.overflow,0,JSON.stringify(report));assert(!report.escaped,JSON.stringify(report));
   assert.equal(report.metricCount,8);assert.deepEqual(report.headings,['指标','T²统计','SPE/Q统计']);assert.equal(report.rows,8);
   assert(report.nestedPadding.every(value=>value==='0px'),JSON.stringify(report));
   modelReports.push(report);
   await page.locator('#modelContent').screenshot({path:path.join(artifacts,`model-results-${width}.png`)});
  }
  fs.writeFileSync(path.join(artifacts,'model-summary-geometry.json'),JSON.stringify(modelReports,null,2));
  await page.setViewportSize({width:1440,height:1000});
  const diagnostic=await page.evaluate(()=>{
   const diagnostic=state.training.model_quality.training_condition_diagnostic,group=diagnostic.groups[0];
   focusTrainingDiagnostic(diagnostic,group);return {id:state.trainingDiagnosticFocus?.clusterId,windows:JSON.stringify(state.trainingWindows)};
  });
  assert.equal(diagnostic.id,candidate.cluster_id);
  assert(await page.locator('#explorationClusterDetails').evaluate(node=>node.open));
  await page.locator('#explorationClusterQuality [data-profile]').last().click();
  assert.equal(await page.evaluate(()=>state.trainingDiagnosticFocus.clusterId),candidate.cluster_id);
  assert.equal(await page.evaluate(()=>JSON.stringify(state.trainingWindows)),diagnostic.windows);
  await page.locator('.variable-diagnostics > details > summary').click();
  await page.locator('#diagnosticsTagConfig').click();
  assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
  assert(!await nav.isVisible());assert.equal(await nav.evaluate(e=>e.getBoundingClientRect().height),0);
  await page.locator('.workflow-step[data-panel="modelPanel"]').click();assert(!await nav.isVisible());
  assert.equal(await page.locator('.workflow-step[aria-selected="true"]').getAttribute('data-panel'),'modelPanel');
  await page.locator('[data-panel="candidatePanel"]').click();
  assert(await nav.isVisible());assert.equal(await page.locator('.candidate-tool-tab[aria-selected="true"]').getAttribute('data-panel'),'stateExplorationPanel');
  await page.locator('#explorationClusterCount').fill('3');await page.locator('#explorationClusterCount').blur();
  assert(await page.locator('#explorationContent').evaluate(e=>e.hidden));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(artifacts,'screening-geometry.json'),JSON.stringify(reports,null,2));
  fs.writeFileSync(path.join(artifacts,'screening-center-geometry.json'),JSON.stringify(centerReports,null,2));
  console.log(JSON.stringify({status:'PASS',reports,baselineCompared:Boolean(process.env.SCREENING_BASELINE_JSON)}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    env = {**os.environ, "SCREENING_TEST_URL": f"http://127.0.0.1:{server.server_port}",
           "SCREENING_TEST_CSV": str(csv), "SCREENING_TEST_DIR": str(tmp_path), "SCREENING_BUILD_ID": ui.WEB_BUILD_ID,
           "SCREENING_HTML_SHA256": hashlib.sha256(ui.INDEX_HTML.encode("utf-8")).hexdigest()}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
