"""Opt-in real ellipse interaction and region feature layout regression."""
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


def test_region_features_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ui._BASE_WEB, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(42)
    x = np.r_[rng.normal(-3, .2, 240), rng.normal(3, .2, 240)]
    tags = ["FLOW_" + "LONG_TAG_" * 12, *[f"TAG_{i}" for i in range(1, 14)]]
    frame = pd.DataFrame({tag: x * (i + 1) + rng.normal(size=480) for i, tag in enumerate(tags)})
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=480, freq="5min"))
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],reports=[];
 page.on('pageerror',e=>errors.push(e.message));
 const before=process.env.REGION_VISUAL_PHASE==='before',dir=process.env.REGION_ARTIFACT_DIR||process.env.REGION_TEST_DIR;
 fs.mkdirSync(dir,{recursive:true});
 try {
  await page.goto(process.env.REGION_TEST_URL);await page.evaluate(()=>document.fonts.ready);
  async function action(id,endpoint) {
   const pending=page.waitForResponse(r=>r.url().endsWith(endpoint));await page.locator('#'+id).click();
   const response=await pending,data=await response.json();assert(response.ok(),JSON.stringify(data));return data;
  }
  await page.locator('#fileInput').setInputFiles(process.env.REGION_TEST_CSV);
  await action('uploadButton','/api/upload');await page.locator('#timestampColumn').selectOption('time');
  await action('inspectButton','/api/inspect');
  await page.evaluate(()=>{showWorkflowStage('candidatePanel');showCandidateTool('candidatePreparationPanel');});
  await page.locator('#maxLag').fill('0');await page.locator('#filterMethod').selectOption('none');
  await page.evaluate(()=>showCandidateTool('stateExplorationPanel'));
  await page.locator('#explorationClusterCount').fill('2');
  const data=await action('stateExplorationButton','/api/state-exploration/run');
  await page.waitForFunction(()=>Boolean(document.querySelector('#explorationPcChart .js-plotly-plot')?._fullLayout));
  const invariant=await page.evaluate(()=>JSON.stringify([state.exploration.cluster_candidates,state.exploration.cluster_series_full,state.exploration.cluster_quality]));
  async function screenshot(width,stateName) {
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
   await page.locator('.exploration-result-grid').screenshot({path:path.join(dir,`${before?'before':'after'}-${width}-${stateName}.png`)});
  }
  async function draw(target='A',left=.06,right=.66) {
   await page.locator('#explorationRegionTarget').selectOption(target);
   if(!await page.evaluate(()=>state.preferredRegionDrawing)) await page.locator('#explorationRegionSelect').click();
   await page.locator('#explorationPcChart').scrollIntoViewIfNeeded();
   await page.waitForFunction(()=>Boolean(state.preferredRegionPlot)&&state.preferredRegionDrawing);
   const rect=await page.evaluate(()=>{
    const plot=document.querySelector('#explorationPcChart .js-plotly-plot'),r=plot.getBoundingClientRect(),s=plot._fullLayout._size;
    return {x:r.x+s.l,y:r.y+s.t,w:s.w,h:s.h};
   });
   const pending=page.waitForResponse(r=>r.url().endsWith(target==='B'?'/region-comparison':'/preferred-region'));
   await page.mouse.move(rect.x+rect.w*left,rect.y+rect.h*.06);await page.mouse.down();
   await page.mouse.move(rect.x+rect.w*right,rect.y+rect.h*.94,{steps:12});await page.mouse.up();
   const response=await pending,payload=await response.json();assert(response.ok(),JSON.stringify(payload));
   if(target==='B') {
    await page.waitForFunction(()=>Boolean(state.regionComparison?.variables));
    assert(payload.region_b.selected_sample_count>0); return payload;
   }
   await page.waitForFunction(seq=>state.preferredRegion?.selected_sample_count!==undefined&&state.preferredRegionUpdateSeq>=seq,payload.preferred_region_update_seq);
   assert(payload.selected_sample_count>0&&payload.selected_sample_count<data.full_point_count);
   return payload;
  }
  for(const width of [1440,900,390]) {
   await page.setViewportSize({width,height:1000});
   if(await page.locator('#explorationRegionClear').isEnabled()) await action('explorationRegionClear','/preferred-region');
   await screenshot(width,'empty');
   if(!before) {
    assert.equal(await page.locator('#explorationRegionFeaturesDetails').evaluate(n=>n.open),false);
    await page.locator('#explorationRegionFeaturesDetails > summary').click();
    assert((await page.locator('#explorationRegionFeatures').innerText()).includes('尚未定义优选运行区域'));
    await page.locator('#explorationRegionFeaturesDetails > summary').click();
   }
   const region=await draw();await screenshot(width,'selected-collapsed');
   const geometryBefore=await page.locator('.exploration-result-grid').evaluate(grid=>[...grid.querySelectorAll('.exploration-pc-card,.exploration-region-card')].map(n=>{const r=n.getBoundingClientRect();return [r.x,r.y+scrollY,r.width,r.height];}));
   if(!before) {
    assert.equal(region.variables.length,14);
    await page.locator('#explorationRegionFeaturesDetails > summary').click();
    assert.equal(await page.locator('#explorationRegionFeatures > .table-wrap tbody tr').count(),10);
    await screenshot(width,'selected-expanded');
    await page.locator('#explorationRegionFeatures details > summary').click();
    assert.equal(await page.locator('#explorationRegionFeatures tbody tr').count(),14);
    assert.equal(await page.locator('#explorationRegionFeatures tbody td.numeric').first().evaluate(n=>getComputedStyle(n).textAlign),'right');
    await screenshot(width,'all-tags');
    await page.locator('#stateExplorationPanel').screenshot({path:path.join(dir,`after-${width}-context.png`)});
    const featureLayout=await page.locator('.exploration-result-grid').evaluate(grid=>{
     const feature=grid.querySelector('#explorationRegionFeaturesDetails'),a=feature.getBoundingClientRect(),b=grid.getBoundingClientRect();
     return {widthDifference:Math.abs(a.width-b.width),leftDifference:Math.abs(a.left-b.left),
      tables:[...feature.querySelectorAll('.table-wrap')].map(n=>({height:n.getBoundingClientRect().height,overflow:getComputedStyle(n).overflowX}))};
    });
    assert(featureLayout.widthDifference<=1&&featureLayout.leftDifference<=1,JSON.stringify(featureLayout));
    assert(featureLayout.tables.every(n=>n.height<=280&&n.overflow==='auto'),JSON.stringify(featureLayout));
    const geometryAfter=await page.locator('.exploration-result-grid').evaluate(grid=>[...grid.querySelectorAll('.exploration-pc-card,.exploration-region-card')].map(n=>{const r=n.getBoundingClientRect();return [r.x,r.y+scrollY,r.width,r.height];}));
    assert.deepEqual(geometryAfter,geometryBefore);
    await action('explorationRegionClear','/preferred-region');
    assert((await page.locator('#explorationRegionFeatures').innerText()).includes('尚未定义优选运行区域'));
    await screenshot(width,'cleared');
    const redrawn=await draw();assert.deepEqual(redrawn.variables,region.variables);
    await screenshot(width,'redrawn');
    const aInvariant=await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows,state.preferredRegionUpdateSeq]));
    const plotIdentity=await page.evaluate(()=>{globalThis.abPlot=state.preferredRegionPlot;return Boolean(abPlot);});assert(plotIdentity);
    const pair=await draw('B',.35,.95);
    assert.equal(pair.region_a.selected_sample_count,redrawn.selected_sample_count);
    assert.equal(pair.a_only_sample_count+pair.intersection_sample_count,pair.region_a.selected_sample_count);
    assert.equal(pair.b_only_sample_count+pair.intersection_sample_count,pair.region_b.selected_sample_count);
    assert.equal(pair.variables.length,14);
    assert.equal(await page.evaluate(()=>state.preferredRegionPlot===abPlot),true);
    const sorted=pair.variables.filter(v=>v.standardized_difference!==null).map(v=>Math.abs(v.standardized_difference));
    assert(sorted.every((v,i)=>i===0||v<=sorted[i-1]));
    assert.equal(await page.locator('#explorationRegionFeatures > .table-wrap').last().locator('tbody tr').count(),10);
    await screenshot(width,'ab-top10');
    await page.locator('#explorationRegionFeatures details > summary').click();
    assert.equal(await page.locator('#explorationRegionFeatures > .table-wrap').last().locator('tbody tr').count(),10);
    assert.equal(await page.locator('#explorationRegionFeatures details tbody tr').count(),4);
    await screenshot(width,'ab-all-tags');
    await page.locator('#stateExplorationPanel').screenshot({path:path.join(dir,`after-${width}-ab-context.png`)});
    const projection=await page.evaluate(()=>{
     const plot=state.preferredRegionPlot,r=plot.getBoundingClientRect(),l=plot._fullLayout;
     return [...plot.querySelectorAll('.shapelayer path')].filter(n=>n.getBoundingClientRect().width>0).map((n,i)=>{
      const box=n.getBoundingClientRect(),s=plot.layout.shapes[i];
      return Math.max(Math.abs(box.left-(r.left+l.xaxis._offset+l.xaxis.c2p(s.x0))),Math.abs(box.width-Math.abs(l.xaxis.c2p(s.x1)-l.xaxis.c2p(s.x0))),Math.abs(box.height-Math.abs(l.yaxis.c2p(s.y1)-l.yaxis.c2p(s.y0))));
     });
    });
    assert.equal(projection.length,2);assert(projection.every(error=>error<2),JSON.stringify(projection));
    const overlap=await draw('B',.08,.54);assert.equal(await page.evaluate(()=>state.referenceRegionEllipses.length),2);
    assert(overlap.intersection_sample_count>0&&overlap.a_only_sample_count>0&&overlap.b_only_sample_count>0);
    await screenshot(width,'ab-overlap');
    await action('explorationRegionDelete','/region-comparison');
    await page.waitForFunction(()=>state.regionComparison?.region_b?.ellipses?.length===1);
    await screenshot(width,'ab-deleted');
    assert.equal(await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows,state.preferredRegionUpdateSeq])),aInvariant);
    await page.evaluate(async()=>updateExplorationReferenceRegion(state.preferredRegion.ellipses));
    assert.equal(await page.evaluate(()=>state.regionComparison.a_only_sample_count+state.regionComparison.b_only_sample_count),0);
    assert(await page.evaluate(()=>state.regionComparison.variables.every(v=>v.standardized_difference===null)));
    await screenshot(width,'ab-identical');
    await page.locator('#explorationRegionClear').click();
    assert.deepEqual(await page.evaluate(()=>state.referenceRegionEllipses),[]);
    assert.equal(await page.locator('#explorationRegionFeatures > .table-wrap tbody tr').count(),10);
    await screenshot(width,'ab-cleared');
    await page.evaluate(async()=>{
     const plot=state.preferredRegionPlot;globalThis.abRanges=[plot._fullLayout.xaxis.range.slice(),plot._fullLayout.yaxis.range.slice()];
     const x=abRanges[0],y=abRanges[1];await Plotly.relayout(plot,{'xaxis.range':[x[0]+.2,x[1]-.1],'yaxis.range':[y[0]+.04,y[1]+.04]});
    });
    await draw('B',.06,.66);
    await screenshot(width,'ab-zoom-pan');
    await page.evaluate(async()=>{await updateExplorationReferenceRegion([]);await Plotly.relayout(state.preferredRegionPlot,{'xaxis.range':abRanges[0],'yaxis.range':abRanges[1]});});
    await page.locator('#explorationRegionTarget').selectOption('A');
    await page.evaluate(async()=>updateExplorationPreferredRegion([{center_pc1:1e6,center_pc2:1e6,radius_pc1:1,radius_pc2:1}]));
    assert.equal(await page.evaluate(()=>state.preferredRegion.selected_sample_count),0);
    assert.equal(await page.locator('#explorationRegionFeatures').innerText(),'没有选中样本。');
    await screenshot(width,'zero-samples');
    await action('explorationRegionClear','/preferred-region');
    await draw();
    await page.locator('#explorationRegionFeaturesDetails > summary').click();
   }
   const report=await page.evaluate(()=>({width:innerWidth,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,
    gridColumns:getComputedStyle(document.querySelector('.exploration-result-grid')).gridTemplateColumns,
    variables:state.preferredRegion.variables,selected:state.preferredRegion.selected_sample_count}));
   assert.equal(report.overflow,0,JSON.stringify(report));reports.push(report);
   const baselinePath=path.join(dir,'before-geometry.json');
   if(!before&&fs.existsSync(baselinePath)) {
    const baseline=JSON.parse(fs.readFileSync(baselinePath,'utf8')).find(r=>r.width===width);
    if(baseline) {assert.equal(report.selected,baseline.selected);assert.equal(report.gridColumns,baseline.gridColumns);}
   }
   assert.equal(await page.evaluate(()=>JSON.stringify([state.exploration.cluster_candidates,state.exploration.cluster_series_full,state.exploration.cluster_quality])),invariant);
  }
  if(!before) {
   await page.locator('#explorationRegionFeaturesDetails > summary').click();
   const savedA=await page.evaluate(()=>state.preferredRegion.ellipses);
   await page.evaluate(async()=>updateExplorationReferenceRegion(state.preferredRegion.ellipses));
   await page.locator('#explorationRegionTarget').selectOption('A');
   await action('explorationRegionDelete','/preferred-region');
   await page.waitForFunction(()=>state.regionComparison?.region_a?.selected_sample_count===0);
   assert.deepEqual(await page.evaluate(()=>state.referenceRegionEllipses),savedA);
   await screenshot(390,'ab-a-deleted');
   await page.evaluate(async ellipses=>{await updateExplorationPreferredRegion(ellipses);await updateExplorationReferenceRegion([]);},savedA);
   const aKept=await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows]));
   await page.route('**/region-comparison',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({error:'比较失败测试'})}));
   await page.evaluate(async()=>updateExplorationReferenceRegion(state.preferredRegion.ellipses));
   assert((await page.locator('#explorationRegionFeatures').innerText()).includes('比较失败测试'));
   assert.equal(await page.evaluate(()=>state.referenceRegionEllipses.length),1);
   await screenshot(390,'ab-error');
   await page.unroute('**/region-comparison');
   let comparisonRelease,comparisonReceived;
   const comparisonGate=new Promise(resolve=>comparisonRelease=resolve),comparisonFetched=new Promise(resolve=>comparisonReceived=resolve);
   await page.route('**/region-comparison',async route=>{const response=await route.fetch();comparisonReceived();await comparisonGate;await route.fulfill({response});});
   await page.evaluate(()=>{globalThis.pendingComparison=updateExplorationReferenceRegion(state.preferredRegion.ellipses);});
   await comparisonFetched;assert((await page.locator('#explorationRegionFeatures').innerText()).includes('正在计算'));
   await page.evaluate(async()=>updateExplorationReferenceRegion([]));comparisonRelease();await page.evaluate(async()=>pendingComparison);
   assert.equal(await page.evaluate(()=>state.regionComparison),null);
   assert.equal(await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows])),aKept);
   await page.unroute('**/region-comparison');
   const single=await page.evaluate(()=>state.preferredRegion);
   await page.evaluate(async ellipse=>updateExplorationPreferredRegion([ellipse,ellipse]),single.ellipses[0]);
   assert.equal(await page.evaluate(()=>state.preferredRegion.selected_sample_count),single.selected_sample_count);
   assert.deepEqual(await page.evaluate(()=>state.preferredRegion.variables),single.variables);
   const retained=await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows]));
   await page.route('**/preferred-region',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({error:'选区失败测试'})}));
   await page.evaluate(async()=>updateExplorationPreferredRegion([]));
   assert.equal(await page.evaluate(()=>JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows])),retained);
   assert.equal(await page.locator('#explorationRegionFeatures tbody tr').count(),14);
   await page.unroute('**/preferred-region');
   let release,received,held=false;
   const gate=new Promise(resolve=>release=resolve),fetched=new Promise(resolve=>received=resolve);
   await page.route('**/preferred-region',async route=>{
    if(held){await route.continue();return;}held=true;
    const response=await route.fetch();received();await gate;await route.fulfill({response});
   });
   await page.evaluate(ellipse=>{globalThis.pendingRegionUpdate=updateExplorationPreferredRegion([ellipse]);},single.ellipses[0]);
   await fetched;
   assert((await page.locator('#explorationRegionFeatures').innerText()).includes('正在计算'));
   await page.evaluate(async()=>updateExplorationPreferredRegion([]));
   release();await page.evaluate(async()=>globalThis.pendingRegionUpdate);
   assert.deepEqual(await page.evaluate(()=>state.preferredRegion.ellipses),[]);
   assert.equal(await page.locator('#explorationRegionFeatures').innerText(),'尚未定义优选运行区域。');
   await page.unroute('**/preferred-region');
   await page.evaluate(async()=>updateExplorationReferenceRegion([{center_pc1:0,center_pc2:0,radius_pc1:10,radius_pc2:10}]));
   assert(await page.evaluate(()=>Boolean(state.regionComparison?.region_b)));
   const rerun=await action('stateExplorationButton','/api/state-exploration/run');
   assert.notEqual(rerun.exploration_run_id,data.exploration_run_id);
   await page.waitForFunction(id=>state.exploration?.exploration_run_id===id&&state.preferredRegion===null,rerun.exploration_run_id);
   assert.deepEqual(await page.evaluate(()=>state.referenceRegionEllipses),[]);
   assert.equal(await page.evaluate(()=>state.regionComparison),null);
   assert.equal(await page.locator('#explorationRegionFeatures').innerText(),'尚未定义优选运行区域。');
   await page.evaluate(async()=>updateExplorationReferenceRegion([{center_pc1:0,center_pc2:0,radius_pc1:10,radius_pc2:10}]));
   await page.evaluate(()=>showCandidateTool('candidatePreparationPanel'));
   await page.locator('#maxLag').fill('5');
   await page.locator('#maxLag').press('Tab');
   assert.equal(await page.evaluate(()=>state.exploration===null),true);
   assert.deepEqual(await page.evaluate(()=>state.referenceRegionEllipses),[]);
   assert.equal(await page.evaluate(()=>state.regionComparison),null);
   assert.equal(await page.locator('#explorationContent').evaluate(n=>n.hidden),true);
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(dir,`${before?'before':'after'}-geometry.json`),JSON.stringify(reports,null,2));
  console.log(JSON.stringify({status:'PASS',reports}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    env = {**os.environ, "REGION_TEST_URL": f"http://127.0.0.1:{server.server_port}",
           "REGION_TEST_CSV": str(csv), "REGION_TEST_DIR": str(tmp_path)}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["status"] == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
