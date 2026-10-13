"""Opt-in browser check using the same runtime as test_web_geometry.py."""

import json
import hashlib
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web_model_results as ui


def test_variable_diagnostics_fifty_tags_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(ui._BASE_WEB, "UPLOADS_DIR", tmp_path / "uploads")
    rng = np.random.default_rng(42)
    x = rng.normal(size=180)
    tags = ["FIC400001.SV_" + "LONG_TAG_" * 30, "FIC400002.SV", *[f"TAG_{i}" for i in range(2, 50)]]
    frame = pd.DataFrame({tag: (x * (i + 1) + rng.normal(scale=0.01, size=180) if i < 40 else rng.normal(size=180)) for i, tag in enumerate(tags)})
    frame[tags[48]] = rng.integers(0, 2, size=180)
    frame[tags[49]] = 1000 + rng.normal(scale=1e-4, size=180)
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=180, freq="5min"))
    csv = tmp_path / "history.csv"
    frame.to_csv(csv, index=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r"""
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT);
const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],reports=[];page.on('pageerror',error=>errors.push(error.message));
  try {
    const response=await page.goto(process.env.DIAGNOSTICS_TEST_URL);
    const crypto=require('node:crypto');
    assert.equal(crypto.createHash('sha256').update(await response.body()).digest('hex'),process.env.DIAGNOSTICS_HTML_HASH);
    assert.equal(await page.locator('meta[name="pca-model-builder-build"]').getAttribute('content'),process.env.DIAGNOSTICS_BUILD_ID);
    await page.evaluate(()=>document.fonts.ready);
    const dialog=page.locator('#groupProfileDialog'),body=page.locator('#groupProfileDialogBody');
    async function openProfile(source) {
      await page.locator(`${source} .group-profile-entry [data-profile-open]`).click();
      await dialog.waitFor({state:'visible'});
      assert.equal(await dialog.evaluate(node=>node.open),true);
      assert(await page.locator('#groupProfileDialogClose').evaluate(node=>node===document.activeElement));
      assert.equal(await dialog.getAttribute('aria-labelledby'),'groupProfileDialogTitle');
      assert((await page.locator('#groupProfileDialogTitle').innerText()).trim());
      assert.equal(await page.locator(`${source} [data-profile]`).count(),0);
    }
    async function action(id,endpoint) {
      const pending=page.waitForResponse(r=>r.url().endsWith(endpoint));
      await page.locator('#'+id).click();const response=await pending;
      const data=await response.json();assert(response.ok(),JSON.stringify(data));return data;
    }
    await page.locator('#fileInput').setInputFiles(process.env.DIAGNOSTICS_TEST_CSV);
    await action('uploadButton','/api/upload');
    await page.locator('#timestampColumn').selectOption('time');
    await action('inspectButton','/api/inspect');
    // 共享预处理参数位于「筛选准备」TAB：先在该面板改参数，再切到状态探索。
    await page.evaluate(()=>{showWorkflowStage('candidatePanel');showCandidateTool('candidatePreparationPanel');});
    await page.locator('#maxLag').fill('0');
    await page.evaluate(()=>showCandidateTool('stateExplorationPanel'));
    await page.locator('#explorationClusterCount').fill('2');
    const data=await action('stateExplorationButton','/api/state-exploration/run');
    await page.waitForFunction(()=>state.exploration?.variable_diagnostics?.summary.tag_count===50);
    const diagnostics=data.variable_diagnostics;
    assert.equal(await page.locator('[data-profile-performance]').count(),0);
    assert.equal(diagnostics.sample_count,data.full_point_count);
    assert.equal(diagnostics.cluster_features.length,50);
    assert(diagnostics.high_correlation_pairs.length>20);
    assert(diagnostics.tag_profiles.some(item=>item.flags.includes('近似无变化')));
    assert(diagnostics.tag_profiles.some(item=>item.flags.includes('低唯一值，疑似离散状态量')));
    assert.equal(await page.locator('.variable-diagnostics button').count(),1);
    assert.equal(await page.locator('.variable-diagnostics > details').evaluate(node=>node.open),false);
    const diagnosticSummary=await page.locator('#variableDiagnosticsTitle').innerText();
    for(const count of [diagnostics.summary.tag_count,diagnostics.summary.high_correlation_pair_count,diagnostics.summary.attention_tag_count]) assert(diagnosticSummary.includes(String(count)));
    await page.locator('.variable-diagnostics > details > summary').click();
    assert((await page.locator('#explorationVariableDiagnostics').innerText()).includes('高相关变量'));
    const cdp=await page.context().newCDPSession(page);
    for(const [width,dpr] of [[1920,1],[1440,1],[900,1],[390,1],[720,2]]) {
      await page.setViewportSize({width,height:1000});
      await cdp.send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:dpr,mobile:false});
      await page.locator('.variable-diagnostics').evaluate(section=>section.querySelector(':scope > details').open=true);
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const report=await page.locator('.variable-diagnostics').evaluate(section=>{
        const rect=section.getBoundingClientRect(),button=section.querySelector('button').getBoundingClientRect();
        const tables=[...section.querySelectorAll('.table-wrap')];
        return {width:innerWidth,dpr:devicePixelRatio,pageOverflowX:document.documentElement.scrollWidth-document.documentElement.clientWidth,
          sectionWidth:rect.width,sectionRight:rect.right,buttonRight:button.right,tableCount:tables.length,
          maxTableHeight:Math.max(...tables.map(e=>e.getBoundingClientRect().height)),
          localOverflow:tables.some(e=>e.scrollWidth>e.clientWidth),
          maxTagWidth:Math.max(...Array.from(section.querySelectorAll('tbody td:first-child'),e=>e.getBoundingClientRect().width)),
          tableOutside:tables.some(e=>e.getBoundingClientRect().right>rect.right+1)};
      });
      assert.equal(report.pageOverflowX,0,JSON.stringify(report));
      assert(report.sectionWidth>0 && report.sectionRight<=width+1 && report.buttonRight<=width+1,JSON.stringify(report));
      assert(report.tableCount>=5 && report.maxTableHeight<=280 && !report.tableOutside,JSON.stringify(report));
      if(width===390) {assert(report.localOverflow);assert(report.maxTagWidth<=224,JSON.stringify(report));}
      reports.push(report);
    }
    // Exercise profile selection on the real Plotly canvas without rebuilding it.
    await page.setViewportSize({width:1440,height:1000});
    await cdp.send('Emulation.clearDeviceMetricsOverride');
    await page.waitForFunction(()=>state.preferredRegionPlot?.data);
    await page.evaluate(async()=>{await updateExplorationPreferredRegion([{center_pc1:0,center_pc2:0,radius_pc1:1,radius_pc2:1}]);});
    assert.equal(await page.evaluate(()=>state.preferredRegionPlot.layout.shapes.length),1);
    const plotColors=await page.evaluate(()=>JSON.stringify(state.preferredRegionPlot.data.map(trace=>trace.marker?.color)));
    const snapshot=await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion]));
    await page.evaluate(()=>{globalThis.profilePlotBefore=state.preferredRegionPlot;return Plotly.relayout(state.preferredRegionPlot,{'xaxis.range':[-2,2],'yaxis.range':[-2,2]});});
    for(const width of [1440,900,390]) {
      await page.setViewportSize({width,height:1000});
      const entry=page.locator('#explorationClusterQuality .group-profile-entry [data-profile-open]');
      for(let repeat=0;repeat<2;repeat++) {
        await openProfile('#explorationClusterQuality');
        await page.keyboard.press('Escape');
        await dialog.waitFor({state:'hidden'});
        assert(await entry.evaluate(node=>node===document.activeElement));
        assert.equal(await body.locator('[data-profile]').count(),0);
      }
      await openProfile('#explorationClusterQuality');
      const geometry=await dialog.evaluate(node=>{
        const r=node.getBoundingClientRect(),body=document.getElementById('groupProfileDialogBody'),b=body.getBoundingClientRect();
        const close=document.getElementById('groupProfileDialogClose').getBoundingClientRect();
        return {width:innerWidth,left:r.left,right:r.right,top:r.top,bottom:r.bottom,height:r.height,
          bodyWidth:b.width,bodyHeight:b.height,bodyOverflow:body.scrollWidth-body.clientWidth,
          tableOutside:[...body.querySelectorAll('.table-wrap')].some(table=>{const t=table.getBoundingClientRect();return t.left<b.left-1||t.right>b.right+1;}),
          tableCount:body.querySelectorAll('.table-wrap').length,
          closeLeft:close.left,closeRight:close.right,closeTop:close.top,closeBottom:close.bottom,
          overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth};
      });
      assert(geometry.left>=0&&geometry.right<=width+1&&geometry.top>=0&&geometry.bottom<=1001&&geometry.height>0,JSON.stringify(geometry));
      assert(geometry.bodyWidth>0&&geometry.bodyHeight>0&&geometry.bodyOverflow<=1,JSON.stringify(geometry));
      assert(geometry.tableCount>0&&!geometry.tableOutside,JSON.stringify(geometry));
      assert(geometry.closeLeft>=geometry.left&&geometry.closeRight<=geometry.right+1&&geometry.closeTop>=geometry.top&&geometry.closeBottom<=geometry.bottom+1,JSON.stringify(geometry));
      assert.equal(geometry.overflow,0,JSON.stringify(geometry));reports.push({dialog:geometry});
      if(width===1440) assert.equal(await body.locator('[aria-pressed="true"]').count(),0);
      await body.locator('.group-profiles [data-profile]').first().click();
      await page.waitForFunction(()=>state.preferredRegionPlot.data.filter(t=>t.type==='scattergl').some(t=>t.marker.opacity===.12));
      assert(await page.evaluate(()=>state.preferredRegionPlot===globalThis.profilePlotBefore));
      assert.deepEqual(await page.evaluate(()=>state.preferredRegionPlot.layout.xaxis.range),[-2,2]);
      assert.deepEqual(await page.evaluate(()=>state.preferredRegionPlot.layout.yaxis.range),[-2,2]);
      assert.equal(await page.evaluate(()=>JSON.stringify(state.preferredRegionPlot.data.map(trace=>trace.marker?.color))),plotColors);
      assert.equal(await page.evaluate(()=>state.preferredRegionPlot.layout.shapes.length),1);
      const viewBefore=await page.evaluate(()=>JSON.stringify({x:state.preferredRegionPlot.layout.xaxis.range,y:state.preferredRegionPlot.layout.yaxis.range,dragmode:state.preferredRegionPlot.layout.dragmode,shapes:state.preferredRegionPlot.layout.shapes}));
      await body.locator('[data-profile-candidates]').click();
      await dialog.waitFor({state:'hidden'});
      await page.waitForFunction(()=>{const r=document.getElementById('explorationClusterCandidates').getBoundingClientRect();return r.height>0&&r.top>=0&&r.bottom<=innerHeight+1;});
      const selectedGroup=await page.evaluate(()=>state.groupProfileSelection.state_exploration);
      assert.equal(await page.locator('#explorationClusterCandidates tr[data-profile-focus="true"]').count(),data.cluster_candidates.filter(item=>item.cluster_id===selectedGroup).length);
      await openProfile('#explorationClusterQuality');
      await body.locator('[data-profile-pc]').click();
      await dialog.waitFor({state:'hidden'});
      await page.waitForFunction(()=>{const r=document.getElementById('explorationPcChart').getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight+1;});
      const pcRect=await page.locator('#explorationPcChart').boundingBox();
      assert(pcRect.y>=0&&pcRect.y+pcRect.height<=1001,JSON.stringify(pcRect));
      assert(await page.evaluate(()=>state.preferredRegionPlot===globalThis.profilePlotBefore));
      assert.equal(await page.evaluate(()=>JSON.stringify({x:state.preferredRegionPlot.layout.xaxis.range,y:state.preferredRegionPlot.layout.yaxis.range,dragmode:state.preferredRegionPlot.layout.dragmode,shapes:state.preferredRegionPlot.layout.shapes})),viewBefore);
      const opacity=await page.evaluate(()=>state.preferredRegionPlot.data.filter(trace=>trace.type==='scattergl').map(trace=>({name:trace.name,value:trace.marker.opacity,selected:trace.name===clusterUiLabel(state.groupProfileSelection.state_exploration)})));
      assert(opacity.every(trace=>trace.value===(trace.selected?.85:.12)),JSON.stringify(opacity));
      assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),snapshot);
      await openProfile('#explorationClusterQuality');
      await body.locator('[data-profile-timeline]').click();
      await dialog.waitFor({state:'hidden'});
      await page.waitForFunction(()=>{const r=document.getElementById('explorationTimeline').getBoundingClientRect();return r.height>0&&r.top>=0&&r.bottom<=innerHeight+1;});
      assert.equal(await page.locator('#explorationTimeline [data-profile-episode]').count(),data.cluster_quality.group_profiles[0].episode_count);
      await openProfile('#explorationClusterQuality');
      await body.locator('[data-profile-candidates]').click();
      await dialog.waitFor({state:'hidden'});
      assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),snapshot);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
      await openProfile('#explorationClusterQuality');
      await body.locator('.group-profiles [data-profile]').last().focus();
      await body.locator('.group-profiles [data-profile]').last().press('Enter');
      assert.equal(await body.locator('.group-profiles [data-profile]').last().getAttribute('aria-pressed'),'true');
      await page.waitForFunction(()=>state.preferredRegionPlot.data.filter(trace=>trace.type==='scattergl').every(trace=>trace.marker.opacity===((trace.legendgroup===`cluster:${state.groupProfileSelection.state_exploration}`||trace.name===clusterUiLabel(state.groupProfileSelection.state_exploration))?.85:.12)));
      assert(await page.evaluate(()=>state.preferredRegionPlot===globalThis.profilePlotBefore));
      assert.equal(await page.evaluate(()=>JSON.stringify({x:state.preferredRegionPlot.layout.xaxis.range,y:state.preferredRegionPlot.layout.yaxis.range,dragmode:state.preferredRegionPlot.layout.dragmode,shapes:state.preferredRegionPlot.layout.shapes})),viewBefore);
      assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),snapshot);
      assert.equal(await page.locator('[data-profile-performance]').count(),0);
      assert(/[↑↓]/.test(await body.locator('.group-profiles tbody').first().innerText()));
      if(process.env.PROFILE_ARTIFACT_DIR) {
        await page.screenshot({path:require('node:path').join(process.env.PROFILE_ARTIFACT_DIR,`group-profile-${width}.png`),fullPage:true});
        await dialog.screenshot({path:require('node:path').join(process.env.PROFILE_ARTIFACT_DIR,`group-profile-dialog-${width}.png`)});
      }
      await page.locator('#groupProfileDialogClose').click();
      await dialog.waitFor({state:'hidden'});
      assert(await entry.evaluate(node=>node===document.activeElement));
    }
    await page.setViewportSize({width:1440,height:1000});
    await openProfile('#explorationClusterQuality');
    await page.evaluate(()=>renderStateExploration(state.exploration));
    await dialog.waitFor({state:'hidden'});
    assert.equal(await body.locator('[data-profile]').count(),0);
    await openProfile('#explorationClusterQuality');
    await page.evaluate(()=>showCandidateTool('clusterPanel'));
    await dialog.waitFor({state:'hidden'});
    await page.evaluate(()=>showCandidateTool('stateExplorationPanel'));
    await openProfile('#explorationClusterQuality');
    await page.evaluate(()=>showWorkflowStage('configPanel'));
    await dialog.waitFor({state:'hidden'});
    await page.evaluate(()=>showWorkflowStage('candidatePanel'));
    await openProfile('#explorationClusterQuality');
    await page.evaluate(()=>{document.getElementById('explorationMinimumDuration').value='99999';document.getElementById('explorationMinimumDuration').dispatchEvent(new Event('change',{bubbles:true}));});
    await dialog.waitFor({state:'hidden'});
    assert(await page.locator('#explorationContent').evaluate(node=>node.hidden));
    await page.locator('#explorationMinimumDuration').fill('99999');
    await page.locator('#explorationClusterCount').fill('10');
    const ten=await action('stateExplorationButton','/api/state-exploration/run');
    assert.equal(ten.cluster_quality.group_profiles.length,10);assert.equal(ten.cluster_candidates.length,0);
    assert.equal(await page.evaluate(()=>state.groupProfileSelection.state_exploration),null);
    for(const width of [1440,900,390]) {
      await page.setViewportSize({width,height:1000});
      await openProfile('#explorationClusterQuality');
      assert.equal(await body.locator('.group-profiles [data-profile]').count(),10);
      await body.locator('.group-profiles [data-profile]').last().focus();
      const listScroll=await body.locator('.group-profiles > .table-wrap').evaluate(node=>{
        node.scrollTop=node.scrollHeight;node.scrollLeft=node.scrollWidth;
        return {top:node.scrollTop,left:node.scrollLeft};
      });
      await body.locator('.group-profiles [data-profile]').last().press('Enter');
      assert.equal(await body.locator('.group-profiles [data-profile]').last().getAttribute('aria-pressed'),'true');
      assert.deepEqual(await body.locator('.group-profiles > .table-wrap').evaluate(node=>({top:node.scrollTop,left:node.scrollLeft})),listScroll);
      assert((await body.locator('[data-profile-candidates]').textContent()).includes('（0）'));
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
      await page.locator('#groupProfileDialogClose').click();
    }
    const explorationSelection=await page.evaluate(()=>state.groupProfileSelection.state_exploration);
    await page.evaluate(()=>showCandidateTool('clusterPanel'));
    await page.locator('#clusterCount').fill('10');
    const assistance=await action('clusterButton','/api/cluster');
    assert.equal(assistance.cluster_quality.group_profiles.length,10);
    const assistanceSnapshot=await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion]));
    for(const width of [1440,900,390]) {
      await page.setViewportSize({width,height:1000});
      await openProfile('#assistanceClusterQuality');
      await body.locator('.group-profiles [data-profile]').last().click();
      assert.equal(await page.evaluate(()=>state.groupProfileSelection.state_exploration),explorationSelection);
      assert.equal(await page.evaluate(()=>state.groupProfileSelection.cluster_assistance),assistance.cluster_quality.group_profiles.at(-1).cluster_id);
      if(process.env.PROFILE_ARTIFACT_DIR) {
        await page.screenshot({path:require('node:path').join(process.env.PROFILE_ARTIFACT_DIR,`group-assistance-${width}.png`),fullPage:true});
        await dialog.screenshot({path:require('node:path').join(process.env.PROFILE_ARTIFACT_DIR,`group-assistance-dialog-${width}.png`)});
      }
      assert.equal(await body.locator('[data-profile-timeline]').count(),0);
      assert.equal(await body.locator('[data-profile-pc]').count(),0);
      await body.evaluate(node=>node.querySelectorAll('details').forEach(item=>item.open=true));
      const assistanceText=await body.innerText();
      for(const label of ['时间连续性','平均持续时间','最长连续时间','状态切换次数','主要区分变量']) assert(!assistanceText.includes(label));
      assert(await page.locator('#clusterChart circle[fill-opacity=".12"]').count()>0);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth),0);
      assert.equal(await page.evaluate(()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.preferredRegion])),assistanceSnapshot);
      await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});
      assert(await page.locator('#assistanceClusterQuality [data-profile-open]').evaluate(node=>node===document.activeElement));
    }
    const assistanceSelection=await page.evaluate(()=>state.groupProfileSelection.cluster_assistance);
    await page.evaluate(()=>showCandidateTool('stateExplorationPanel'));
    await openProfile('#explorationClusterQuality');
    assert.equal(await body.locator('.group-profiles [aria-pressed="true"]').getAttribute('data-profile'),explorationSelection);
    await body.locator('.group-profiles [data-profile]').last().click();
    assert.equal(await page.evaluate(()=>state.groupProfileSelection.cluster_assistance),assistanceSelection);
    await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});
    // Navigation itself preserves the evidence; editing uses existing invalidation.
    await page.locator('#diagnosticsTagConfig').click();
    assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
    assert(await page.locator('#tagSearch').evaluate(e=>e===document.activeElement));
    assert(await page.evaluate(()=>state.exploration!==null && state.selectedModelTags.size===50));
    await page.locator('#tagOptions input[type=checkbox]').first().uncheck();
    assert(await page.evaluate(()=>state.exploration===null && state.selectedModelTags.size===49));
    assert(await page.evaluate(()=>state.groupProfileSelection===null));
    assert(await page.locator('#explorationContent').evaluate(e=>e.hidden));
    assert((await page.locator('#explorationEmpty').textContent()).includes('请重新运行状态探索'));
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'PASS',reports,highPairs:diagnostics.high_correlation_pairs.length}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    env = {**os.environ, "DIAGNOSTICS_TEST_URL": os.environ.get("PROFILE_RUNNING_URL", f"http://127.0.0.1:{server.server_port}"), "DIAGNOSTICS_TEST_CSV": str(csv),
           "DIAGNOSTICS_HTML_HASH": hashlib.sha256(ui.INDEX_HTML.encode()).hexdigest(), "DIAGNOSTICS_BUILD_ID": ui.WEB_BUILD_ID}
    try:
        result = subprocess.run([shutil.which("node"), "-"], input=script, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report["status"] == "PASS"
        (tmp_path / "variable_diagnostics_geometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
