"""Geometry, original timestamps and missing-score semantics in Chromium."""
import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_model_results as ui


def test_stage_b_geometry_and_editing(tmp_path, monkeypatch):
    if not all((shutil.which("node"), os.environ.get("WEB_GEOMETRY_PLAYWRIGHT"), os.environ.get("WEB_GEOMETRY_BROWSER"))):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER")
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(80, 3)), columns=list("ABC"))
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=80, freq="5min"))
    uploaded = web.save_upload("geometry.csv", frame.to_csv(index=False).encode())
    trained = ui.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": list("ABC"),
                               "normal_start": frame.time.iloc[0].isoformat(), "normal_end": frame.time.iloc[-1].isoformat(),
                               "filter_method": "none", "sample_interval_minutes": 5, "max_lag_minutes": 0,
                               "lag_step_minutes": 5, "n_components": 2, "model_name": "geometry"})
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"trained": trained, "file_id": uploaded["file_id"]}), encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT),assert=require('node:assert/strict'),fs=require('node:fs');
const fixture=JSON.parse(fs.readFileSync(process.env.STAGE_B_FIXTURE,'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 try {
  const page=await browser.newPage(),errors=[],operations=[],reports=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.STAGE_B_URL);
  await page.evaluate(f=>{state.fileId=f.file_id;state.runId=f.trained.run_id;state.training=f.trained;renderTraining(f.trained);showWorkflowStage('modelPanel');},fixture);
  // Deliberately near-identical vectors retain coordinates and get separate labels.
  await page.evaluate(f=>renderTraining({...f.trained,loading_plot:{...f.trained.loading_plot,points:Array.from({length:12},(_,i)=>({tag:`TAG_${i}`,pc1:1,pc2:1+i*.0001,magnitude:2}))}}),fixture);
  for(const width of [1600,1440,1366,1280,1050,900,760,390]) {
   await page.setViewportSize({width,height:1000});await page.evaluate(()=>document.fonts.ready);
   const g=await page.evaluate(()=>{
    const score=el('scoreChart'),svg=score.querySelector('svg'),r=svg.getBoundingClientRect(),box=svg.viewBox.baseVal;
    const labels=[...el('loadingChart').querySelectorAll('text')].filter(e=>e.textContent.startsWith('TAG_'));
    const overlap=labels.some((e,i)=>labels.slice(i+1).some(o=>{const a=e.getBoundingClientRect(),b=o.getBoundingClientRect();return Math.min(a.right,b.right)>Math.max(a.left,b.left)+1&&Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+1;}));
    return {width:innerWidth,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,ratio:r.height/r.width,expected:box.height/box.width,scoreHeight:score.clientHeight,svgHeight:r.height,overlap,navHeight:el('workflowSteps').clientHeight};
   });
   assert.equal(g.overflow,0,JSON.stringify(g));assert(Math.abs(g.ratio-g.expected)<.01,JSON.stringify(g));assert(Math.abs(g.scoreHeight-g.svgHeight)<2,JSON.stringify(g));assert(!g.overlap,JSON.stringify(g));if(width<=760)assert(g.navHeight<190,JSON.stringify(g));reports.push(g);
  }
  await page.evaluate(()=>{
   showWorkflowStage('candidatePanel');showCandidateTool('trendPanel');
   state.trainingWindows=[{id:'precise',start:'2026-01-01T00:00:01.123456+08:00',end:'2026-01-01T01:00:02.654321+08:00',enabled:true,comment:'original',source:'manual'}];renderTrainingWindows();
  });
  await page.locator('#trainingWindows button').filter({hasText:/^编辑$/}).click();
  assert.equal(await page.locator('#trainingWindowEditor input[name=start]').inputValue(),'2026-01-01T00:00:01.123456+08:00');
  await page.locator('#trainingWindowEditor input[name=comment]').fill('discard');await page.locator('#trainingWindowEditor button').filter({hasText:/^取消$/}).click();
  assert.equal(await page.evaluate(()=>state.trainingWindows[0].comment),'original');
  await page.route('**/api/training-windows',async route=>{const p=route.request().postDataJSON();operations.push(p.operation);await route.fulfill({json:{training_windows:p.training_windows.map(w=>({...w,...p.operation.changes})),summary:[]}});});
  await page.locator('#trainingWindows button').filter({hasText:/^编辑$/}).click();await page.locator('#trainingWindowEditor input[name=comment]').fill('saved');await page.locator('#trainingWindowEditor button').filter({hasText:/^保存$/}).click();
  await page.waitForFunction(()=>state.trainingWindows[0].comment==='saved');
  assert.deepEqual(operations[0],{action:'update',id:'precise',changes:{start:'2026-01-01T00:00:01.123456+08:00',end:'2026-01-01T01:00:02.654321+08:00',comment:'saved'}});
  await page.evaluate(()=>{state.validationWindows=[{id:'v',type:'normal_validation',start:'2026-01-02T00:00:01.123456+08:00',end:'2026-01-02T01:00:02.654321+08:00',enabled:true}];state.training=null;renderValidationWindows();showWorkflowStage('validationPanel');});
  await page.locator('#validationWindowTable button').first().click();await page.locator('#validationComment').fill('note');await page.locator('#addValidationWindow').click();
  assert.equal(await page.evaluate(()=>state.validationWindows[0].start),'2026-01-02T00:00:01.123456+08:00');
  await page.route('**/api/frozen-replay',route=>route.fulfill({json:{notice:'测试',scores:[{timestamp:'2026-01-01T00:00',t2_limit_ratio:1,spe_limit_ratio:2},{timestamp:'2026-01-01T00:05',t2_limit_ratio:null,spe_limit_ratio:null},{timestamp:'2026-01-01T00:10',t2_limit_ratio:2,spe_limit_ratio:1}]}}));
  await page.evaluate(()=>{el('validatedModelDownload').hidden=false;el('frozenModelDownload').hidden=false;showWorkflowStage('releasePanel');el('frozenReplayStart').value='2026-01-01T00:00';el('frozenReplayEnd').value='2026-01-01T00:10';});
  await page.locator('#frozenReplayButton').click();await page.waitForFunction(()=>el('frozenReplayTrend').querySelector('svg'));
  assert.equal(await page.locator('#frozenReplayTrend polyline').count(),4);
  assert((await page.locator('#frozenReplayTrend').innerText()).includes('控制限 1.0'));
  assert.deepEqual(errors,[]);console.log(JSON.stringify({reports,operations}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=120,
                                env={**os.environ, "STAGE_B_URL": f"http://127.0.0.1:{server.server_port}", "STAGE_B_FIXTURE": str(fixture)})
        assert result.returncode == 0, result.stdout + result.stderr
        (tmp_path / "stage-b-geometry.json").write_text(result.stdout, encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
