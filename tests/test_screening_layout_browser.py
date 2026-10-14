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
    monkeypatch.setattr(ui._BASE_WEB, "RUNS_DIR", tmp_path / "runs")
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
  const resultStyles=await page.evaluate(()=>['explorationContent','clusterContent','performanceContent','modelQualityResults','modelContent','validationContent','releaseContent'].map(id=>{
   const node=document.getElementById(id),hidden=node.hidden;
   node.hidden=true;const hiddenDisplay=getComputedStyle(node).display;
   node.hidden=false;const style=getComputedStyle(node),expanded={display:style.display,gap:style.rowGap,minWidth:style.minWidth};
   node.hidden=hidden;return {id,hiddenDisplay,...expanded,expectedGap:getComputedStyle(document.documentElement).getPropertyValue('--space-3').trim()};
  }));
  for(const style of resultStyles){assert.equal(style.hiddenDisplay,'none',JSON.stringify(style));assert.equal(style.display,'grid',JSON.stringify(style));assert.equal(style.gap,style.expectedGap,JSON.stringify(style));assert.equal(style.minWidth,'0px');}
  const profileDialog=page.locator('#groupProfileDialog'),profileBody=page.locator('#groupProfileDialogBody');
  async function openProfile() {
   await page.locator('#explorationClusterQuality .group-profile-entry [data-profile-open]').click();
   await profileDialog.waitFor({state:'visible'});
   assert.equal(await profileDialog.evaluate(node=>node.open),true);
   assert.equal(await page.locator('#explorationClusterQuality [data-profile]').count(),0);
  }
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
  assert(await page.locator('#uploadButton').evaluate(e=>e.classList.contains('secondary')));
  assert(!await page.locator('#inspectButton').evaluate(e=>e.classList.contains('secondary')));
  await page.locator('[data-panel="configPanel"]').click();
  const qualityText=await page.evaluate(()=>{
   const quality=state.quality,tag=state.selectedTag;state.selectedTag='TAG_A';
   state.quality={tags:[{tag:'TAG_A',role:'continuous_input',status:'review',suggested_action:'use',issues:[],full:{sample_count:480},reference:{sample_count:480}}]};
   renderCurrentTagQuality();const text=document.querySelector('#currentTagQuality').textContent;
   state.quality=quality;state.selectedTag=tag;renderCurrentTagQuality();return text;
  });
  assert(qualityText.includes('连续输入')&&qualityText.includes('需确认')&&qualityText.includes('建议操作：使用'));
  assert(!/continuous_input|review|use|480\.00/.test(qualityText));
  const workspaceReports=[];
  for(const width of [1280,1366,1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   const report=await page.evaluate(()=>{
    const workspace=document.querySelector('.tag-workspace'),file=document.querySelector('#tagConfigFile');
    return {width:innerWidth,columns:getComputedStyle(workspace).gridTemplateColumns.split(' ').length,
     overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
     fileWidth:file.getBoundingClientRect().width,available:document.querySelector('.results').getBoundingClientRect().width};
   });
   assert.equal(report.columns,width>=900?2:1,JSON.stringify(report));assert.equal(report.overflow,0,JSON.stringify(report));
   assert(report.fileWidth>200||width===390,JSON.stringify(report));workspaceReports.push(report);
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('[data-panel="candidatePanel"]').click();
  assert(await page.locator('#trendPanel').isVisible());
  assert(!await page.locator('#modelingEligibility').isVisible());
  await page.locator('#candidateConfigSummary button').click();
  assert(await page.locator('#candidatePreparationPanel').isVisible());
  assert(await nav.isVisible());assert.equal(await nav.locator('a').count(),3);
  await page.locator('#addEligibilityKeep').click();
  const keep=page.locator('#eligibilityKeepConditions .eligibility-condition');
  await keep.locator('[data-field="column"]').selectOption('TAG_A');await keep.locator('[data-field="mode"]').selectOption('interval');await keep.locator('[data-field="minimum"]').fill('-10');
  assert.equal(await page.locator('#refreshEligibilitySummary').count(),0);
  await autoSummary(()=>keep.locator('[data-field="maximum"]').fill('10'));
  assert((await page.locator('#eligibilitySummary').textContent()).includes('480'));
  await page.locator('#addEligibilityExcludeGroup').click();
  const exclude=page.locator('#eligibilityExcludeGroups .eligibility-condition');
  await exclude.locator('[data-field="column"]').selectOption('TAG_A');await exclude.locator('[data-field="mode"]').selectOption('interval');await exclude.locator('[data-field="minimum"]').fill('-10');
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
  await page.locator('#addEligibilityKeep').click();
  const comparison=page.locator('#eligibilityKeepConditions .eligibility-condition');
  await comparison.locator('[data-field="column"]').selectOption('TAG_A');
  await comparison.locator('[data-field="operator"]').selectOption('>');
  await comparison.locator('[data-field="value"]').fill('0.73');
  await page.waitForFunction(()=>document.querySelector('#eligibilitySummary').textContent.startsWith('分析时间范围：'));
  assert.equal(await comparison.locator('.eligibility-preview').textContent(),'保留 TAG_A > 0.73 的数据');
  await page.setViewportSize({width:390,height:1000});
  const eligibilityGeometry=await comparison.evaluate(node=>{const r=node.getBoundingClientRect();return {right:r.right,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};});
  assert(eligibilityGeometry.right<=391,JSON.stringify(eligibilityGeometry));assert.equal(eligibilityGeometry.overflow,0);
  await page.setViewportSize({width:1440,height:1000});
  await autoSummary(()=>comparison.getByRole('button',{name:'删除条件',exact:true}).click());
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
  await page.locator('[data-panel="stateExplorationPanel"]').click();
  assert((await page.locator('#candidateConfigSummary').innerText()).includes('最大 Lag：10'));
  assert(!await page.locator('#modelingEligibility').isVisible());
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
  // 优选区域统计：指标表是四列紧凑表，工况分布表按内容定宽、标签左对齐、数值右对齐。
  const readRegionSummary=()=>page.locator('#explorationRegionSummary').evaluate(node=>{
   const rect=e=>e.getBoundingClientRect();
   const metricTable=node.querySelector('.region-metric-table'),clusterTable=node.querySelector('.region-cluster-table');
   const metricCells=[...metricTable.querySelectorAll('tr')].map(row=>[...row.children].map(cell=>cell.textContent));
   const clusterRows=[...clusterTable.querySelectorAll('tbody tr')].map(row=>[...row.children].map(cell=>cell.textContent));
   const card=node.closest('.exploration-region-card'),heading=card.querySelector('h3');
   return {metricRows:metricCells.length,metricCells,metricLabels:metricCells.flatMap(row=>[row[0],row[2]]).filter(Boolean),
    clusterHeaders:[...clusterTable.querySelectorAll('thead th')].map(th=>th.textContent),clusterRows,
    alignments:[...clusterTable.querySelectorAll('tbody tr:first-child td')].map(cell=>getComputedStyle(cell).textAlign),
    metricAlignments:[...metricTable.querySelector('tr').children].map(cell=>getComputedStyle(cell).textAlign),
    rowHeight:rect(clusterTable.querySelector('tbody tr')).height,tableWidth:rect(clusterTable).width,cardWidth:rect(card).width,
    topGap:rect(metricTable).top-rect(heading).bottom,nodeText:node.innerText,
    overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};
  });
  async function applyRegion(ellipses) {
   const pending=page.waitForResponse(r=>r.url().endsWith('/preferred-region'));
   await page.evaluate(async list=>{await updateExplorationPreferredRegion(list);},ellipses);
   const payload=await (await pending).json();
  await page.waitForFunction(seq=>state.preferredRegionUpdateSeq>=seq,payload.preferred_region_update_seq);
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   return payload;
  }
  const regionClusterIds=new Set((data.cluster_series_full||data.cluster_series).map(row=>String(row.cluster_id))).size;
  const fullRegionData=await applyRegion([{center_pc1:0,center_pc2:0,radius_pc1:1000000,radius_pc2:1000000}]);
  const regionSummary=await readRegionSummary();
  assert(regionSummary.metricRows>=3&&regionSummary.metricCells.every(row=>row.length===4),JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricLabels,['优选区域样本','完整有效样本占比','最大 工况组占比','区域稳定性','连续候选数','性能有效样本','性能达标样本','性能达标率','性能中位数'],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricCells[0],['优选区域样本',String(fullRegionData.selected_sample_count),'完整有效样本占比',`${(fullRegionData.selected_sample_ratio*100).toFixed(1)}%`],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricCells[1],['最大 工况组占比',`${(fullRegionData.max_cluster_share*100).toFixed(1)}%`,'区域稳定性',fullRegionData.stability_score.toFixed(4)],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricCells[2],['连续候选数',String(fullRegionData.candidate_count),'性能有效样本',String(fullRegionData.performance_valid_count)],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricCells[3],['性能达标样本',String(fullRegionData.performance_target_count),'性能达标率',`${(fullRegionData.performance_target_ratio*100).toFixed(1)}%`],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricCells[4],['性能中位数',fullRegionData.performance_median.toFixed(3),'',''],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.metricAlignments,['left','right','left','right'],JSON.stringify(regionSummary));
  assert.deepEqual(regionSummary.clusterHeaders,['工况组','区域样本数','区域占比']);
  assert.deepEqual(regionSummary.alignments,['left','right','right'],JSON.stringify(regionSummary));
  assert.equal(regionSummary.clusterRows.length,regionClusterIds,JSON.stringify(regionSummary));
  assert(regionSummary.clusterRows.length>=2&&regionSummary.clusterRows.every(row=>row.length===3),JSON.stringify(regionSummary));
  assert.equal(regionSummary.clusterRows.reduce((sum,row)=>sum+Number(row[1]),0),fullRegionData.selected_sample_count,JSON.stringify(regionSummary));
  assert(regionSummary.rowHeight>=28&&regionSummary.rowHeight<=30,JSON.stringify(regionSummary));
  assert(regionSummary.tableWidth<regionSummary.cardWidth-1,JSON.stringify(regionSummary));
  assert(regionSummary.topGap<=13,JSON.stringify(regionSummary));
  assert(regionSummary.nodeText.includes('工况分布')&&regionSummary.nodeText.includes('重叠样本只计一次'),JSON.stringify(regionSummary));
  assert.equal(regionSummary.overflow,0,JSON.stringify(regionSummary));
  assert.equal(await page.locator('#explorationRegionSummary .metrics').count(),0);
  // 零样本区域：工况分布保留全部工况组（含 0 样本），占比与缺失值沿用原格式。
  const emptyRegionData=await applyRegion([{center_pc1:0,center_pc2:0,radius_pc1:1e-9,radius_pc2:1e-9}]);
  assert.equal(emptyRegionData.selected_sample_count,0,JSON.stringify(emptyRegionData));
  const emptyRegion=await readRegionSummary();
  assert.equal(emptyRegion.clusterRows.length,regionClusterIds,JSON.stringify(emptyRegion));
  assert(emptyRegion.clusterRows.every(row=>row[1]==='0'&&row[2]==='0.0%'),JSON.stringify(emptyRegion));
  assert.deepEqual(emptyRegion.metricCells[0],['优选区域样本','0','完整有效样本占比','0.0%'],JSON.stringify(emptyRegion));
  assert.deepEqual(emptyRegion.metricCells[1],['最大 工况组占比','0.0%','区域稳定性','—'],JSON.stringify(emptyRegion));
  assert.deepEqual(emptyRegion.metricCells[2],['连续候选数','0','性能有效样本','0'],JSON.stringify(emptyRegion));
  assert.deepEqual(emptyRegion.metricCells[3],['性能达标样本','0','性能达标率','—'],JSON.stringify(emptyRegion));
  assert.deepEqual(emptyRegion.metricCells[4],['性能中位数','—','',''],JSON.stringify(emptyRegion));
  assert.equal(emptyRegion.overflow,0,JSON.stringify(emptyRegion));
  // 未定义椭圆时保留原提示（同一渲染入口，不额外发起区域请求）。
  await page.evaluate(()=>{state.preferredRegion={ellipses:[]};renderExplorationRegionSummary(state.exploration);});
  assert.equal(await page.locator('#explorationRegionSummary').innerText(),'尚未定义优选运行区域；点击“椭圆选择”后在 PC1/PC2 图中拖动绘制。');
  // 用已收到的完整区域响应恢复状态，后续几何用例仍在真实数据上运行。
  await page.evaluate(payload=>{state.preferredRegion=payload.preferred_region;state.exploration={...state.exploration,preferred_region_candidates:payload.preferred_region_candidates};renderExplorationRegionSummary(state.exploration);},fullRegionData);
  assert.equal(await page.locator('#explorationRegionSummary .region-cluster-table tbody tr').count(),regionClusterIds);
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
  const retained = await page.evaluate(()=>JSON.stringify([commonPayload(),state.exploration,state.candidateWindows,state.trainingWindows]));
  for(const width of [1280,1366,1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   await page.locator('#candidateConfigSummary button').click();
   assert(await page.locator('#modelingEligibility').isVisible());
   assert(!await nav.locator('a').nth(1).isVisible());
   const preparation=await page.locator('#candidatePreparationPanel').evaluate(node=>({
    overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
    clipped:[...node.querySelectorAll('input,select')].filter(e=>e.getClientRects().length&&e.getBoundingClientRect().right>innerWidth+1).map(e=>e.id),
    dates:['analysisStart','analysisEnd'].map(id=>document.getElementById(id).getBoundingClientRect().width)
   }));
   assert.equal(preparation.overflow,0,JSON.stringify(preparation));
   assert.deepEqual(preparation.clipped,[],JSON.stringify(preparation));
   assert(Math.abs(preparation.dates[0]-preparation.dates[1])<=1,JSON.stringify(preparation));
   for(const tool of ['trendPanel','clusterPanel','performancePanel','stateExplorationPanel']) {
    await page.locator(`[data-panel="${tool}"]`).click();
    assert(await page.locator('#'+tool).isVisible());
    assert(!await page.locator('#modelingEligibility').isVisible());
    assert.equal(await page.locator('.candidate-tool-tab[aria-selected="true"]').getAttribute('data-panel'),tool);
    assert(await page.locator('#candidateConfigSummary').isVisible());
   }
   assert.equal(await page.evaluate(()=>JSON.stringify([commonPayload(),state.exploration,state.candidateWindows,state.trainingWindows])),retained);
   const summaryBar=await page.locator('#candidateConfigSummary').evaluate(node=>{
    const span=node.querySelector('span'),button=node.querySelector('button'),rect=e=>e.getBoundingClientRect();
    const lineHeight=parseFloat(getComputedStyle(span).lineHeight);
    return {height:rect(node).height,lines:Math.round(rect(span).height/lineHeight),text:span.textContent,
     scroll:span.scrollHeight,client:span.clientHeight,buttonVisible:rect(button).width>0&&rect(button).height>0,
     buttonLabel:button.textContent,buttonRight:rect(button).right,
     overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};
   });
   // 配置摘要固定单行紧凑信息栏：长文本局部省略、按钮始终可见、不产生页面级横向溢出。
   assert(summaryBar.lines===1,JSON.stringify(summaryBar));
   assert(summaryBar.height<=48,JSON.stringify(summaryBar));
   assert(summaryBar.scroll<=summaryBar.client+1,JSON.stringify(summaryBar));
   assert(summaryBar.buttonVisible&&summaryBar.buttonLabel==='修改配置',JSON.stringify(summaryBar));
   assert(summaryBar.buttonRight<=width+1,JSON.stringify(summaryBar));
   assert(summaryBar.text.includes('资格：筛选后 480 点'),JSON.stringify(summaryBar));
   assert.equal(summaryBar.overflow,0,JSON.stringify(summaryBar));
  }
  async function checkNavigation() {
   const before=await snapshot(),requestCount=requests.length;
   for(let index=0;index<3;index++) {
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
   await position(90);assert.equal(await nav.locator('[aria-current="location"]').innerText(),'状态分析');
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
    const regionMetric=document.querySelector('#explorationRegionSummary .region-metric-table'),regionCluster=document.querySelector('#explorationRegionSummary .region-cluster-table');
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
     regionTableWidth:rect(regionMetric).width,regionClusterTableWidth:rect(regionCluster).width,regionContentWidth:rect(region).width,
     regionTopGap:rect(regionMetric).top-rect(region.querySelector('h3')).bottom,
     regionRowHeight:rect(regionCluster.querySelector('tbody tr')).height,
     regionTableScroll:regionMetric.parentElement.scrollWidth-regionMetric.parentElement.clientWidth,
     regionWrapperOverflow:getComputedStyle(regionMetric.parentElement).overflowX,
     regionPageOverflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
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
   assert.equal(report.regionPageOverflow,0,JSON.stringify(report));
   // 两张表都按内容定宽：放得下就不撑满，放不下时由 .table-wrap 在本区域内滚动而不撑开整页。
   const fitsOrScrolls=width=>width<=report.regionContentWidth+1||(report.regionTableScroll>0&&report.regionWrapperOverflow==='auto');
   assert(fitsOrScrolls(report.regionTableWidth),JSON.stringify(report));
   assert(fitsOrScrolls(report.regionClusterTableWidth),JSON.stringify(report));
   if(Number(zoom)===1) {
    assert(report.regionClusterTableWidth<report.regionContentWidth,JSON.stringify(report));
    assert(report.regionRowHeight>=28&&report.regionRowHeight<=30,JSON.stringify(report));
   }
   assert(report.regionTopGap<=13*Number(zoom)+1,JSON.stringify(report));
   if(width>=761&&zoom===1) {
    assert(report.sameRow&&report.heightSpread<=1,JSON.stringify(report));
    assert(report.pcHeight>=420&&report.pcHeight<=480,JSON.stringify(report));
   }
   if(width===390&&zoom===1) assert.equal(report.pcHeight,340,JSON.stringify(report));
   reports.push(report);
   if(width===1440&&zoom===1) {
    await page.screenshot({path:path.join(artifacts,'screening-after.png'),fullPage:true});
    await page.locator('#candidateConfigSummary').screenshot({path:path.join(artifacts,'screening-config-summary.png')});
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
  for(const width of [1280,1366,1440,900,390]) {
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
   assert.equal(await nav.locator('a').nth(1).getAttribute('href'),'#'+evidence);
   const before=await snapshot(),count=requests.length;
   await nav.locator('a').nth(1).click();
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
  const colorInvariant=()=>page.evaluate(()=>JSON.stringify({points:state.exploration.cluster_series_full,candidates:state.exploration.cluster_candidates,region:state.preferredRegion}));
  const colorBaseline=await colorInvariant();
  for(const mode of ['time','performance']) {
   await page.locator('#explorationPcColorMode').selectOption(mode);
   await page.waitForFunction(mode=>state.preferredRegionPlot&&
    (mode==='time'?state.preferredRegionPlot.layout.coloraxis?.colorbar.title.text==='时间':state.preferredRegionPlot.data.some(trace=>trace.name.includes('达标'))),mode);
   const colors=await page.evaluate(()=>state.preferredRegionPlot.data.filter(trace=>trace.type==='scattergl').reduce((n,trace)=>n+trace.x.length,0));
   assert.equal(colors,data.full_point_count);assert.equal(await colorInvariant(),colorBaseline);
  }
  await page.locator('#explorationPcColorMode').selectOption('variable:TAG_A');
  await page.waitForFunction(()=>state.preferredRegionPlot.data.some(trace=>trace.marker?.coloraxis==='coloraxis'));
  const legendGroups=Object.keys(data.cluster_centers);
  const legendSnapshot=()=>page.evaluate(()=>JSON.stringify([state.exploration.cluster_summaries,state.exploration.cluster_candidates,state.exploration.preferred_region_candidates,state.candidateWindows,state.trainingWindows,state.preferredRegion]));
  const beforeLegend=await legendSnapshot();
  const savedColors=await page.evaluate(groups=>{
   const values=state.explorationColorValues.get('TAG_A');
   state.explorationColorValues.set('TAG_A',state.exploration.cluster_series_full.map((row,index)=>row.cluster_id===groups[0]&&index%2===0?index:null));
   renderExplorationPcChart(state.exploration);return values;
  },legendGroups);
  await page.waitForFunction(()=>state.preferredRegionPlot?._fullLayout);
  assert.equal(await page.locator('#explorationPcChart .legendtoggle').count(),legendGroups.length);
  for(const visible of [false,true]) {
   for(const cluster of legendGroups) {
    const label=await page.evaluate(id=>clusterUiLabel(id),cluster);
    await page.locator('#explorationPcChart .legend .traces').filter({hasText:label}).locator('.legendtoggle').click();
    await page.waitForFunction(({cluster,visible})=>state.preferredRegionPlot._fullData.filter(trace=>trace.legendgroup===`cluster:${cluster}`).every(trace=>trace.visible===(visible?true:'legendonly')),{cluster,visible});
   }
   const visiblePoints=await page.evaluate(()=>state.preferredRegionPlot._fullData.filter(trace=>trace.type==='scattergl'&&trace.visible===true).reduce((count,trace)=>count+trace.x.length,0));
   assert.equal(visiblePoints,visible?data.full_point_count:0);
   assert.equal(await legendSnapshot(),beforeLegend);
  }
  await page.evaluate(values=>{state.explorationColorValues.set('TAG_A',values);renderExplorationPcChart(state.exploration);},savedColors);
  await page.waitForFunction(()=>state.preferredRegionPlot?._fullLayout);
  await page.locator('#explorationPcColorMode').selectOption('cluster');
  await page.waitForFunction(()=>state.preferredRegionPlot?.data.every(trace=>!trace.marker?.coloraxis));
  const firstLabel=await page.evaluate(id=>clusterUiLabel(id),legendGroups[0]);
  await page.locator('#explorationPcChart .legend .traces').filter({hasText:firstLabel}).locator('.legendtoggle').click();
  await page.waitForFunction(cluster=>state.preferredRegionPlot._fullData.filter(trace=>trace.legendgroup===`cluster:${cluster}`).every(trace=>trace.visible==='legendonly'),legendGroups[0]);
  assert.equal(await page.evaluate(cluster=>state.preferredRegionPlot._fullData.filter(trace=>trace.meta==='center'&&trace.legendgroup===`cluster:${cluster}`).length,legendGroups[0]),1);
  await page.locator('#explorationPcColorMode').selectOption('variable:TAG_A');
  await page.waitForFunction(()=>state.preferredRegionPlot?.data.some(trace=>trace.marker?.coloraxis==='coloraxis'));
  await page.evaluate(()=>{globalThis.continuousProfilePlot=state.preferredRegionPlot;});
  await page.evaluate(()=>Plotly.relayout(state.preferredRegionPlot,{'xaxis.range':[-2,2],'yaxis.range':[-3,3]}));
  const continuousView=()=>page.evaluate(()=>JSON.stringify({mode:document.getElementById('explorationPcColorMode').value,
   x:state.preferredRegionPlot.layout.xaxis.range,y:state.preferredRegionPlot.layout.yaxis.range,
   shapes:state.preferredRegionPlot.layout.shapes,dragmode:state.preferredRegionPlot.layout.dragmode,coloraxis:state.preferredRegionPlot.layout.coloraxis,
   colors:state.preferredRegionPlot.data.map(trace=>trace.marker?.color)}));
  const continuousBefore=await continuousView();
  const profileWindows=await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion]));
  await openProfile();
  await profileBody.locator('.group-profiles [data-profile]').first().click();
  await page.waitForFunction(()=>state.preferredRegionPlot.data.filter(trace=>trace.type==='scattergl').every(trace=>trace.marker.opacity===((trace.legendgroup===`cluster:${state.groupProfileSelection.state_exploration}`||trace.name===clusterUiLabel(state.groupProfileSelection.state_exploration))?.85:.12)));
  assert(await page.evaluate(()=>state.preferredRegionPlot===globalThis.continuousProfilePlot));
  assert.equal(await continuousView(),continuousBefore);
  await profileBody.locator('[data-profile-candidates]').click();
  await profileDialog.waitFor({state:'hidden'});
  assert(await page.locator('#explorationClusterCandidates').evaluate(node=>node.closest('details').open));
  assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),profileWindows);
  await openProfile();
  const performanceProfile=data.cluster_quality.group_profiles[0];
  const performanceSummary=data.cluster_summaries.find(item=>item.cluster_id===performanceProfile.cluster_id);
  for(const key of ['performance_valid_count','performance_target_count','performance_target_ratio','performance_median']) assert.equal(performanceProfile[key],performanceSummary[key]);
  const performanceText=await profileBody.locator('[data-profile-performance]').innerText();
  assert(performanceText.includes('性能后验评价')&&performanceText.includes('不参与 PCA 聚类'));
  for(const value of [performanceSummary.performance_valid_count,performanceSummary.performance_target_count,`${(performanceSummary.performance_target_ratio*100).toFixed(1)}%`,String(Number(performanceSummary.performance_median.toPrecision(5)))]) assert(performanceText.includes(String(value)));
  for(const width of [1280,1366,1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   const geometry=await profileBody.locator('[data-profile-performance]').evaluate(node=>{const r=node.getBoundingClientRect();return {width:r.width,right:r.right,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};});
   assert(geometry.width>0&&geometry.right<=width+1,JSON.stringify(geometry));assert.equal(geometry.overflow,0);
   assert(await page.evaluate(()=>state.preferredRegionPlot===globalThis.continuousProfilePlot));
   assert.equal(await continuousView(),continuousBefore);
  }
  await page.locator('#groupProfileDialogClose').click();await profileDialog.waitFor({state:'hidden'});
  assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),profileWindows);
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#explorationPcColorMode').selectOption('cluster');
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
  const trainingExploration=await page.evaluate(()=>[state.exploration,state.preferredRegion]);
  await page.route('**/api/train',async route=>{
   const response=await route.fetch();
   await page.evaluate(()=>{el('maxLag').value='15';el('maxLag').dispatchEvent(new Event('change',{bubbles:true}));});
   await route.fulfill({response});
  });
  await action('trainButton','/api/train');
  await page.waitForFunction(()=>document.getElementById('trainButton').textContent===document.getElementById('trainButton').dataset.label);
  assert.deepEqual(await page.evaluate(()=>({training:state.training,runId:state.runId,quality:state.quality,trainDisabled:el('trainButton').disabled,validateDisabled:el('validateButton').disabled})),{training:null,runId:null,quality:null,trainDisabled:true,validateDisabled:true});
  await page.unroute('**/api/train');
  await page.evaluate(()=>{el('maxLag').value='10';el('maxLag').dispatchEvent(new Event('change',{bubbles:true}));});
  await page.evaluate(([exploration,region])=>{state.exploration=exploration;state.preferredRegion=region;renderStateExploration(exploration);},trainingExploration);
  assert((await action('qualityButton','/api/quality')).training_readiness.normal_state.can_train);
  const training=await action('trainButton','/api/train');assert(training.model_quality.training_condition_diagnostic.available);
  const trainingInvariant=()=>page.evaluate(()=>JSON.stringify({scores:state.training.scores.map(({color_column,color_value,...row})=>row),windows:state.trainingWindows,limits:[state.training.t2_limits,state.training.q_limits]}));
  const trainingBaseline=await trainingInvariant();
  const scorePositions=()=>page.locator('#scoreChart circle').evaluateAll(nodes=>nodes.map(node=>[node.getAttribute('cx'),node.getAttribute('cy')]));
  const positions=await scorePositions();
  for(const column of ['TAG_A','PERFORMANCE']) {
   const response=page.waitForResponse(r=>r.url().endsWith('/api/training-color-values'));
   await page.locator('#scoreColorMode').selectOption(`variable:${column}`);
   const values=await (await response).json();assert.equal(values.column,column);
   assert.deepEqual(values.timestamps,training.scores.map(row=>row.timestamp));
   await page.waitForFunction(column=>el('scoreChartNote').textContent.startsWith(`${column} 工程值范围`),column);
   assert.deepEqual(await scorePositions(),positions);assert.equal(await trainingInvariant(),trainingBaseline);
  }
  const clusterDisabled=await page.locator('#scoreColorMode option[value="cluster"]').isDisabled();
  assert.equal(clusterDisabled,!training.scores.some(row=>row.cluster_id));
  if(!clusterDisabled) {await page.locator('#scoreColorMode').selectOption('cluster');assert.deepEqual(await scorePositions(),positions);}
  await page.locator('#scoreColorMode').selectOption('default');
  const modelReports=[];
  for(const width of [1280,1366,1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   await page.evaluate(()=>document.fonts.ready);
   const report=await page.evaluate(()=>{
    const summary=document.querySelector('#modelMetrics'),items=[...summary.querySelectorAll('.metric')],rect=e=>e.getBoundingClientRect(),bounds=rect(summary);
    const table=document.querySelector('#modelQualitySummary table');
    return {width:innerWidth,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
     metricCount:items.length,summaryHeight:bounds.height,overviewColumns:getComputedStyle(document.querySelector('.model-overview-grid')).gridTemplateColumns.split(' ').length,
     escaped:items.some(e=>rect(e).left<bounds.left-1||rect(e).right>bounds.right+1),
     headings:[...table.querySelectorAll('thead th')].map(e=>e.textContent),rows:table.querySelectorAll('tbody tr').length,
     nestedPadding:[...document.querySelectorAll('#modelContent > section,.model-overview-grid > section')].map(e=>getComputedStyle(e).padding)};
   });
   assert.equal(report.overflow,0,JSON.stringify(report));assert(!report.escaped,JSON.stringify(report));
   assert.equal(report.metricCount,8);assert.deepEqual(report.headings,['指标','T²统计','SPE/Q统计']);assert.equal(report.rows,8);
   assert.equal(report.overviewColumns,width>=900?2:1,JSON.stringify(report));
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
  await openProfile();
  await profileBody.locator('.group-profiles [data-profile]').last().click();
  assert.equal(await page.evaluate(()=>state.trainingDiagnosticFocus.clusterId),candidate.cluster_id);
  assert.equal(await page.evaluate(()=>JSON.stringify(state.trainingWindows)),diagnostic.windows);
  await page.keyboard.press('Escape');await profileDialog.waitFor({state:'hidden'});
  await page.locator('.variable-diagnostics > details > summary').click();
  await page.locator('#diagnosticsTagConfig').click();
  assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
  assert(!await nav.isVisible());assert.equal(await nav.evaluate(e=>e.getBoundingClientRect().height),0);
  await page.locator('.workflow-step[data-panel="modelPanel"]').click();assert(!await nav.isVisible());
  assert.equal(await page.locator('.workflow-step[aria-selected="true"]').getAttribute('data-panel'),'modelPanel');
  await page.locator('[data-panel="candidatePanel"]').click();
  assert(await nav.isVisible());assert.equal(await page.locator('.candidate-tool-tab[aria-selected="true"]').getAttribute('data-panel'),'stateExplorationPanel');
  await openProfile();
  await page.evaluate(()=>{document.getElementById('explorationClusterCount').value='3';document.getElementById('explorationClusterCount').dispatchEvent(new Event('change',{bubbles:true}));});
  await profileDialog.waitFor({state:'hidden'});
  await page.locator('#explorationClusterCount').fill('3');await page.locator('#explorationClusterCount').blur();
  assert(await page.locator('#explorationContent').evaluate(e=>e.hidden));
  // 趋势浏览范围不同于候选分析范围时，配置摘要仍使用候选分析范围资格结果。
  // 放在几何用例之后：绘制趋势会让后续视口变更重绘 Plotly，拖慢用例。
  await page.locator('[data-panel="trendPanel"]').click();
  // 切页签后数据项目趋势会在下一帧把旧版时间控件同步到 dpTrendStart/End；
  // 不等这一帧就填值，刚填入的窗口会被旧范围覆盖（基线同样存在此竞态）。
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await page.locator('#dpTrendVar1').selectOption('TAG_A');
  await page.locator('#dpTrendStart').fill('2026-01-01T01:00');
  // 短趋势窗口：只需与候选分析范围不同，避免大范围 Plotly 重绘拖慢用例。
  await page.locator('#dpTrendEnd').fill('2026-01-01T02:00');
  const trendPending=page.waitForResponse(r=>r.url().endsWith('/api/trend'));
  await page.locator('#dpDrawTrend').click();
  assert((await trendPending).ok());
  await page.waitForFunction(()=>document.querySelector('#trendEligibilitySummary').textContent.startsWith('当前趋势窗口'));
  const scopes=await page.evaluate(()=>({trend:document.querySelector('#trendEligibilitySummary').textContent,
   preparation:document.querySelector('#eligibilitySummary').textContent,
   bar:document.querySelector('#candidateConfigSummary span').textContent,
   overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth}));
  // 趋势窗口（01:00–02:00）的合格样本数不同于候选分析范围（480）；
  // 两者各自显示，配置摘要栏只跟随候选分析范围。
  assert(scopes.trend.startsWith('当前趋势窗口（展示保留完整历史）：筛选后 '),JSON.stringify(scopes));
  assert(!scopes.trend.includes('筛选后 480 点'),JSON.stringify(scopes));
  assert(scopes.preparation.startsWith('分析时间范围：筛选后 480 点'),JSON.stringify(scopes));
  assert(scopes.bar.includes('资格：筛选后 480 点'),JSON.stringify(scopes));
  assert.equal(scopes.overflow,0,JSON.stringify(scopes));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(artifacts,'screening-geometry.json'),JSON.stringify(reports,null,2));
  fs.writeFileSync(path.join(artifacts,'tag-workspace-geometry.json'),JSON.stringify(workspaceReports,null,2));
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
