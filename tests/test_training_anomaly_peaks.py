import hashlib
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from pca_model_builder import web_model_results as ui


def test_training_anomaly_events_and_navigation(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = Path("src/pca_model_builder/model_results.js").read_text(encoding="utf-8")
    helpers = source[source.index("  function trainingAnomalyPeaks("):source.index("  function renderTrainingConditionDiagnostic(")]
    script = r'''
const assert=require('node:assert/strict');
__HELPERS__
const point=(i,t2=2,spe=0.2,extra={})=>({timestamp:`2026-05-17T${String(10+Math.floor(i/60)).padStart(2,'0')}:${String(i%60).padStart(2,'0')}:00`,t2_limit_ratio:t2,spe_limit_ratio:spe,window_id:'w1',segment_id:1,...extra});
const points=[point(0),point(1,5),point(2,3),point(3,0.2),point(4,0.1,6),point(5,7,8,{break_before:true}),point(6,9,0.1,{segment_id:2}),point(7,10,0.1,{segment_id:2,window_id:'w2'}),point(12,11,0.1,{segment_id:2,window_id:'w2'})];
const peaks=trainingAnomalyPeaks(points,1);
assert.deepEqual(peaks.map(p=>p.severity),[11,10,9,8,6,5]);
assert.deepEqual(peaks.slice(-3).map(p=>p.peakType),['T² + SPE','SPE','T²']);
assert.equal(peaks.at(-1).timestamp,points[1].timestamp);
assert.equal(trainingAnomalyPeaks(Array.from({length:15},(_,i)=>point(i,i+1,0,{break_before:true})),1).length,10);
assert.deepEqual(trainingAnomalyPeaks([point(0,0.5,0.5),point(1,null,null)],1),[]);
assert.equal(trainingAnomalyPeaks([point(0,1,0)],1)[0].peakType,'T²');
const diagnostic={timeline:points};
const state={training:{model_quality:{training_condition_diagnostic:diagnostic}},inspection:{time_start:'2026-05-17T10:00:00',time_end:'2026-05-17T10:20:00'},selectedModelTags:new Set(['TAG1','TAG2']),trainingWindows:[{id:'w1'}],candidateWindows:[{id:'c1'}],exploration:{id:'e1'}};
const inputs={trendStart:{},trendEnd:{},trendPreset:{}};
const el=id=>inputs[id]; const messages=[],calls=[];
const setStatus=(message)=>messages.push(message); const displayTime=t=>t;
globalThis.showWorkflowStage=id=>calls.push(id);globalThis.showCandidateTool=id=>calls.push(id);
globalThis.showModelTagsTrend=async(tags,focus,current)=>{assert.deepEqual(tags,['TAG1','TAG2']);assert.equal(focus.start,'2026-05-17T10:00:00');assert.equal(focus.end,'2026-05-17T10:20:00');assert(current());};
(async()=>{const before=JSON.stringify(state);await focusTrainingAnomalyPeak(diagnostic,peaks.at(-1));assert.equal(JSON.stringify(state),before);assert.deepEqual(calls,['candidatePanel','trendPanel']);assert(messages.at(-1).includes(points[1].timestamp));state.trainingDiagnosticInvalidated=true;await focusTrainingAnomalyPeak(diagnostic,peaks[0]);assert(messages.at(-1).includes('已失效'));})().catch(e=>{console.error(e);process.exitCode=1});
'''.replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_training_anomaly_peaks_real_browser(tmp_path):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT),assert=require('node:assert/strict'),crypto=require('node:crypto');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 try {
 const page=await browser.newPage({timezoneId:'Asia/Shanghai'}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',route=>route.fulfill({json:{models:[],runs:[]}}));
 const response=await page.goto(process.env.PEAKS_PAGE_URL);assert.equal(crypto.createHash('sha256').update(await response.body()).digest('hex'),process.env.PEAKS_HTML_HASH);
 await page.evaluate(()=>{
  state.fileId='model-source';state.inspection={time_start:'2026-05-17T09:00:00',time_end:'2026-05-17T14:00:00'};
  state.selectedModelTags=new Set(['TAG1','TAG2','TAG3','TAG4','TAG5']);state.registry=Object.fromEntries([...state.selectedModelTags].map(tag=>[tag,{enabled:true}]));fillSelect(el('trendTags'),[...state.selectedModelTags]);el('sampleInterval').value='1';
  state.trainingWindows=[{id:'w1'}];state.candidateWindows=[{id:'c1'}];state.exploration={id:'e1'};
  const timeline=Array.from({length:36},(_,i)=>({timestamp:new Date(Date.UTC(2026,4,17,10,i)).toISOString().slice(0,19),window_id:'w1',segment_id:1,break_before:false,t2_limit_ratio:i%3===2?0.2:i+1,spe_limit_ratio:0.2,cluster_id:null}));
  const diagnostic={available:false,timeline};state.training={status_counts:{attention:0,abnormal:0},explained_variance:[],scores:[],t2_limits:{},q_limits:{},model_quality:{statistics:{t2:{limits:{},exceedance_rates:{}},spe:{limits:{},exceedance_rates:{}}},training_data:{sources:[]},engineering_messages:[],training_condition_diagnostic:diagnostic}};
  window.before=JSON.stringify([state.training,state.trainingWindows,state.candidateWindows,state.exploration]);renderTraining(state.training);
  showWorkflowStage('modelPanel');
 });
 for(const width of [1440,900]) {await page.setViewportSize({width,height:1000});assert.equal(await page.locator('#modelTrainingAnomalyPeaks tbody tr').count(),10);assert(await page.locator('#modelTrainingAnomalyTitle').isVisible());assert((await page.locator('#modelTrainingAnomalyPeaks tbody tr').first().innerText()).includes('35.000'));assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
 const requests=[];
 await page.route('**/api/trend',async route=>{const p=route.request().postDataJSON();requests.push(p);await route.fulfill({json:{tags:p.tags,series:p.tags.map(tag=>({name:tag,points:[{x:p.start,y:1},{x:p.end,y:2}]})),statistics:{},histograms:{},axis_limits:{},ranges:{},rows_count:2,raw_rows:2}});});
 await page.locator('#modelTrainingAnomalyPeaks button').first().click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('已定位训练'));
 assert.equal(requests.length,2);assert.deepEqual(requests.flatMap(p=>p.tags),['TAG1','TAG2','TAG3','TAG4','TAG5']);
 for(const p of requests){assert.equal(p.file_id,'model-source');assert.equal(p.start,'2026-05-17T10:04:00');assert.equal(p.end,'2026-05-17T11:04:00');assert.equal(p.display_mode,'raw');}
 assert.equal(await page.locator('#dpTrendStart').inputValue(),'2026-05-17T10:04');
 assert(await page.locator('#dpTrendChart .js-plotly-plot').count() || await page.locator('#dpTrendChart').evaluate(e=>!!e._fullData));
 assert(await page.evaluate(()=>window.before===JSON.stringify([state.training,state.trainingWindows,state.candidateWindows,state.exploration])));
 await page.evaluate(()=>{state.training.model_quality.training_condition_diagnostic.timeline=[];renderTraining(state.training);});
 assert.equal(await page.locator('#modelTrainingAnomalyPeaks').innerText(),'当前训练评分未发现 T²/SPE 95% 超限事件。');assert.deepEqual(errors,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    env = {**os.environ, "PEAKS_PAGE_URL": f"http://127.0.0.1:{server.server_port}", "PEAKS_HTML_HASH": hashlib.sha256(ui.INDEX_HTML.encode()).hexdigest()}
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 0, result.stderr
