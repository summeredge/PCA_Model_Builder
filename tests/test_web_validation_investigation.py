import json
import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_dataproject, web_model_results as ui


def test_validation_investigation_navigation_evidence_and_stale_context():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    range_helpers = web.INDEX_HTML[web.INDEX_HTML.index("function windowTimeRange("):web.INDEX_HTML.index("function trainingWindowsPayload(")]
    helpers = web.INDEX_HTML.split("function validationInvestigationWindowsKey(", 1)[1].split("function renderValidation(data)", 1)[0]
    script = r'''
const assert=require("node:assert/strict");
class Element {
  constructor(){this.children=[];this.dataset={};this.innerHTML="";this.textContent="";this.handlers={};}
  append(...nodes){nodes.forEach(node=>node.parentNode=this);this.children.push(...nodes)}
  replaceChildren(){this.children=[];this.innerHTML="";this.textContent=""}
  insertBefore(node,target){node.parentNode=this;this.children.splice(this.children.indexOf(target),0,node)}
  remove(){this.parentNode.children=this.parentNode.children.filter(node=>node!==this)}
  addEventListener(event,handler){this.handlers[event]=handler}
  scrollIntoView(){calls.push("scroll")}
  text(){return this.textContent+this.innerHTML+this.children.map(node=>node.text()).join(" ")}
}
const calls=[],messages=[],nodes={};
const parent=new Element();
function el(id){if(!nodes[id]){nodes[id]=new Element();parent.append(nodes[id]);}return nodes[id]}
const document={createElement:()=>new Element(),querySelectorAll:()=>parent.children.filter(node=>node.dataset.validationInvestigationContext)};
function escapeHtml(x){return String(x).replaceAll("<","&lt;")}
function displayTime(x){return x}
function displayUiValue(x){return x}
function clusterUiLabel(x){return x}
function candidateSourceLabel(window){return window.source||"未知来源"}
function percent(x){return x==null?"—":`${(x*100).toFixed(1)}%`}
function selectedTags(){return [...state.selectedModelTags]}
function setStatus(message){messages.push(message)}
globalThis.showWorkflowStage=id=>calls.push(id);
globalThis.showCandidateTool=id=>calls.push(id);
const n={id:"N03",type:"normal_validation",start:"2026-01-02T01:02:03",end:"2026-01-02T02:03:04",enabled:true};
const a={...n,id:"A02",type:"known_abnormal"};
const disabled={...n,id:"disabled",enabled:false};
const training={training_window_summary:[{id:"w1",start:"2026-01-01T00:00:00",end:"2026-01-01T23:59:00",enabled:true,status:"used",source:"manual"}],training_window_totals:{used_window_count:1,enabled_window_count:1,source_summary:{manual:{effective_sample_share:1}}},model_quality:{training_condition_diagnostic:{traceable_samples:80,unattributed_samples:20,groups:[{cluster_id:"cluster_001",share:.8,source_ref:"state-exploration-r-cluster_001"}]}}};
const data={run_id:"model-r",validation_windows:[n,a,disabled],validation_window_summaries:[{...n,status:"scored",t2_exceedance_95:.123,t2_exceedance_99:.01,spe_exceedance_95:.186,spe_exceedance_99:.02},{...a,status:"scored",continuous_events:[],t2_exceedance_95:0,spe_exceedance_95:0}],contributions:[{validation_window_id:"N03",tags:[{tag:"B",contribution_pct:90}]},{validation_window_id:"other",tags:[{tag:"C",contribution_pct:99}]}],validation_metrics:{known_abnormal:{windows:[{validation_window_id:"A02",first_detection_95:null,first_detection_99:null}]}}};
const state={runId:"model-r",fileId:"f",training,validation:data,validationWindows:[n,a,disabled],trainingWindows:[{id:"edited-window"}],candidateWindows:[{id:"c"}],selectedModelTags:new Set(["A","B","C"]),registry:{A:{role:"continuous_input"},B:{role:"continuous_input"},C:{role:"continuous_input"}}};
el("trendTags").options=["A","B","C","unused"].map(value=>({value,selected:false}));
function protectedState(){return JSON.stringify([state.trainingWindows,state.candidateWindows,state.validationWindows,[...state.selectedModelTags],state.training,state.validation]);}
function banner(){return parent.children.find(node=>node.dataset.validationInvestigationContext)}
__HELPERS__
(async()=>{
  renderValidationInvestigation(data);
  assert.equal(nodes.validationInvestigationWindows.children.length,2);
  assert.match(nodes.validationInvestigationWindows.text(),/12.3%/);
  assert.match(nodes.validationInvestigationWindows.text(),/18.6%/);
  assert.match(nodes.validationInvestigationWindows.text(),/未形成持续检出/);
  const context=state.validationInvestigationContext, before=protectedState();
  await nodes.validationInvestigationWindows.children[0].children.at(-1).children[0].handlers.click();
  assert(calls.includes("candidatePanel")&&calls.includes("trendPanel"));
  assert.equal(nodes.trendStart.value,n.start);
  assert.equal(nodes.trendEnd.value,n.end);
  assert.deepEqual(nodes.trendTags.options.filter(x=>x.selected).map(x=>x.value),["A","B","C"]);
  assert.deepEqual(state.validationInvestigationFocus.relatedTags,["B"]);
  assert.match(banner().text(),/model-r/);
  assert.match(banner().text(),/data-validation-related-tag/);
  assert.match(banner().text(),/不代表工艺根因/);
  banner().children[1].handlers.click();
  assert(calls.includes("validationPanel"));
  await focusValidationInvestigation(context,"N03","training");
  assert(calls.includes("modelPanel"));
  assert.match(banner().text(),/80.0% \/ 20.0%/);
  assert.match(banner().text(),/当前没有可靠的工况来源映射/);
  assert.match(banner().text(),/manual 100.0%/);
  await focusValidationInvestigation(context,"N03","tags");
  assert(calls.includes("configPanel"));
  for(const tag of ["A","B","C"]) assert(banner().text().includes(`<td>${tag}</td>`));
  await focusValidationInvestigation(context,"A02","tags");
  assert(!banner().text().includes("data-validation-related-tag"));
  await focusValidationInvestigation(context,"A02","training");
  assert.match(banner().text(),/该验证窗口与当前训练窗口无时间重叠/);
  assert(!banner().text().includes("该异常没有混入训练集"));
  assert.equal(protectedState(),before);
  // Snapshot evidence belongs to the trained run, not editable trainingWindows.
  training.training_window_summary.push({id:"overlap",start:a.end,end:a.end,enabled:true,status:"dropped",source:"performance",source_ref:"candidate-1"},{id:"disabled-overlap",start:a.start,end:a.end,enabled:false,status:"disabled"});
  await focusValidationInvestigation(context,"A02","training");
  assert.match(banner().text(),/2026-01-02T02:03:04 ~ 2026-01-02T02:03:04/);
  assert(!banner().text().includes("overlap"));
  assert.match(banner().text(),/dropped.*来源 performance/);
  assert(!banner().text().includes("disabled-overlap"));
  assert.match(banner().text(),/不证明重叠样本实际参与/);
  for(const summary of [{t2_exceedance_95:1,spe_exceedance_95:0},{t2_exceedance_95:0,spe_exceedance_95:1},{t2_exceedance_95:1,spe_exceedance_95:1}]) {
    Object.assign(data.validation_window_summaries[1],summary,{continuous_events:[{point_count:2}]});
    renderValidationInvestigation(data);
    assert.match(nodes.validationInvestigationWindows.text(),/这是统计解释/);
    assert(!nodes.validationInvestigationWindows.text().includes("未形成持续检出"));
  }
  Object.assign(data.validation_window_summaries[1],{continuous_events:[{point_count:1},{point_count:1}]});
  renderValidationInvestigation(data);
  assert.match(nodes.validationInvestigationWindows.text(),/未形成持续检出/);
  for(const key of ["runId","fileId","training","validation","validationWindows"]) {
    renderValidationInvestigation(data);
    await focusValidationInvestigation(state.validationInvestigationContext,"N03","tags");
    const old=state[key];state[key]=key==="validationWindows"?[{...n,end:a.start}]:key.endsWith("Id")?"changed":{};
    assert.equal(currentValidationInvestigationFocus(),null,key);
    assert.equal(banner(),undefined,key);
    state[key]=old;
  }
  renderValidationInvestigation(data);
  const oldContext=state.validationInvestigationContext;
  invalidateValidationInvestigation();
  await focusValidationInvestigation(oldContext,"N03","trend");
  assert.match(messages.at(-1),/已失效/);
  assert.equal(state.validationInvestigationFocus,null);
  assert.equal(nodes.validationInvestigationWindows.children.length,0);
  // Rendered report windows cannot be linked to newly edited validation windows.
  state.validationWindows=[{...n,end:a.start}];renderValidationInvestigation(data);
  assert.equal(state.validationInvestigationContext,null);
  console.log(JSON.stringify({ok:true}));
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__HELPERS__", range_helpers + "\nfunction validationInvestigationWindowsKey(" + helpers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]


def test_validation_investigation_trend_batches_ignore_stale_responses():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = web_dataproject._DATAPROJECT_TREND_SCRIPT
    helpers = source[source.index("  async function requestTrend("):source.index('  $("dpDrawTrend").addEventListener')]
    script = r'''
const assert=require("node:assert/strict"), nodes={},requests=[];
function $(id){return nodes[id]??={value:"1000",replaceChildren(){},textContent:""}}
const trendIds=["dpTrendVar1","dpTrendVar2","dpTrendVar3","dpTrendVar4"];
let focus, lastTrend=null, hasDraggedTrendSelection=true, rendered=null;
function commonPayload(){return {file_id:"current"}}
function populateSelectors(){}
function currentValidationInvestigationFocus(){return focus}
function renderTrendPage(data){rendered=data}
function api(url,options){return new Promise((resolve,reject)=>requests.push({url,payload:JSON.parse(options.body),resolve,reject}))}
const tags=Array.from({length:10},(_,i)=>`tag${i}`);
const initial={start:"2026-01-01T10:50:03",end:"2026-01-01T11:20:04"};
function resolveBatches(items){items.forEach(item=>item.resolve({series:item.payload.tags.map(name=>({name,points:[]})),statistics:Object.fromEntries(item.payload.tags.map(tag=>[tag,{current:{sample_count:3}}])),histograms:{},axis_limits:{},ranges:{}}))}
__HELPERS__
(async()=>{
 focus=initial;
 const pending=showValidationInvestigationTrend(tags,initial);
 assert.equal(requests.length,3);
 assert.deepEqual(requests.flatMap(item=>item.payload.tags),tags);
 assert(requests.every(item=>item.url==="/api/trend"&&item.payload.tags.length<=4&&item.payload.start===initial.start&&item.payload.end===initial.end));
 focus=null;resolveBatches(requests);await pending;
 assert.equal(rendered,null);assert.equal(lastTrend,null);
 focus={...initial};requests.length=0;
 const current=showValidationInvestigationTrend(tags,focus);
 resolveBatches(requests);await current;
 assert.deepEqual(rendered.series.map(item=>item.name),tags);
 assert.deepEqual(Object.keys(rendered.statistics),tags);
 assert.equal(nodes.dpTrendStart.value,initial.start);assert.equal(nodes.dpTrendEnd.value,initial.end);
 assert.equal(hasDraggedTrendSelection,false);
 assert.equal(lastTrend,rendered);
 requests.length=0;
 const failed=showValidationInvestigationTrend(tags,focus);
 requests.forEach(item=>item.reject(new Error("read failed")));
 await assert.rejects(failed,/read failed/);
 assert.equal(nodes.dpTrendChart.textContent,"read failed");
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_validation_request_cannot_restore_investigation_after_context_changes():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML
    handler = html.split('el("validateButton").addEventListener("click", async () => {', 1)[1].split('\nel("recordValidationDecision")', 1)[0].rsplit("});", 1)[0]
    invalidation = "function invalidateValidationInvestigation" + html.split("function invalidateValidationInvestigation", 1)[1].split("function currentValidationInvestigationContext", 1)[0]
    script = r'''
const assert=require("node:assert/strict"),nodes={},pending=[],messages=[];
const state={runId:"r",fileId:"f",training:{},validation:{old:true},validationWindows:[{id:"N03"}]};
function el(id){return nodes[id]??={value:"",replaceChildren(){}}}
const document={querySelectorAll:()=>[]};
function setBusy(){}
function setStatus(message){messages.push(message)}
let renders=0;
function renderValidation(){renders++}
function api(){return new Promise(resolve=>pending.push(resolve))}
__INVALIDATION__
async function validateClick(){__HANDLER__}
(async()=>{
 for(const change of ["new-validation","training","runId","fileId","validationWindows"]) {
  const before=state.validation, saved={...state}, task=validateClick();
  if(change==="new-validation") invalidateValidationInvestigation();
  else if(change==="validationWindows") {state.validationWindows=[{id:"changed"}];invalidateValidationInvestigation()}
  else state[change]=change.endsWith("Id")?"changed":{};
  pending.shift()({oldResponse:true});await task;
  assert.equal(state.validation,before,change);assert.equal(renders,0,change);
  assert.match(messages.at(-1),/丢弃过期/);
  Object.assign(state,saved);
 }
 const older=validateClick(),newer=validateClick(),response={latest:true};
 pending.pop()(response);await newer;
 pending.shift()({oldResponse:true});await older;
 assert.equal(state.validation,response);assert.equal(renders,1);
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__INVALIDATION__", invalidation).replace("__HANDLER__", handler)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_validation_investigation_in_real_browser(tmp_path, monkeypatch):
    if not (shutil.which("node") and os.environ.get("WEB_GEOMETRY_PLAYWRIGHT") and os.environ.get("WEB_GEOMETRY_BROWSER")):
        pytest.skip("Set WEB_GEOMETRY_PLAYWRIGHT and WEB_GEOMETRY_BROWSER for browser validation")
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(42)
    tags = [f"TAG_{i}" for i in range(10)]
    frame = pd.DataFrame(rng.normal(size=(240, 10)), columns=tags)
    frame.insert(0, "time", pd.date_range("2026-01-01", periods=240, freq="5min"))
    uploaded = web.save_upload("history.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    payload = {"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": tags,
               "normal_start": frame.time.iloc[0].isoformat(), "normal_end": frame.time.iloc[119].isoformat(),
               "sample_interval_minutes": 5, "filter_method": "none", "max_lag_minutes": 0,
               "lag_step_minutes": 5, "n_components": 2, "model_name": "investigation", "model_purpose": "normal_state"}
    trained = ui.train_payload(payload)
    windows = [{"id": "N03", "type": "normal_validation", "start": frame.time.iloc[130].isoformat(), "end": frame.time.iloc[149].isoformat(), "enabled": True, "comment": ""},
               {"id": "A02", "type": "known_abnormal", "start": frame.time.iloc[160].isoformat(), "end": frame.time.iloc[179].isoformat(), "enabled": True, "comment": ""}]
    validated = web.validate_payload({**payload, "run_id": trained["run_id"], "validation_windows": windows})
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"uploaded": uploaded, "trained": trained, "validated": validated, "tags": tags}), encoding="utf-8")
    artifacts = {path.name: path.read_bytes() for path in (tmp_path / "runs" / trained["run_id"]).iterdir() if path.is_file()}
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
const {chromium}=require(process.env.WEB_GEOMETRY_PLAYWRIGHT),assert=require("node:assert/strict"),fs=require('node:fs'),path=require('node:path');
const fixture=JSON.parse(require("node:fs").readFileSync(process.env.INVESTIGATION_FIXTURE,"utf8"));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.WEB_GEOMETRY_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}), errors=[],requests=[];
 page.on("pageerror",e=>errors.push(e.message));
 page.on("request",r=>{if(r.method()==="POST") requests.push(r)});
 try {
  await page.goto(process.env.INVESTIGATION_URL);
  await page.evaluate(f=>{
   state.fileId=f.uploaded.file_id;state.runId=f.trained.run_id;state.training=f.trained;state.validation=f.validated;
   state.inspection={numeric_columns:f.tags};state.registry=Object.fromEntries(f.tags.map(tag=>[tag,emptyTagConfig()]));
   state.selectedModelTags=new Set(f.tags);state.validationWindows=f.validated.validation_windows;
   state.trainingWindows=f.trained.training_window_summary;state.candidateWindows=[{id:"untouched"}];
   fillSelect(el("timestampColumn"),["time"]);fillSelect(el("trendTags"),f.tags);fillSelect(el("labelColumn"),[],"不使用");
   el("sampleInterval").value="5";el("filterMethod").value="none";el("maxLag").value="0";
   renderTraining(f.trained);renderValidationWindows();renderTagList();renderValidation(f.validated);showWorkflowStage("validationPanel");
  },fixture);
  const geometryReports=[];
  for(const width of [1440,900,390]) {
   await page.setViewportSize({width,height:1000});await page.evaluate(()=>document.fonts.ready);
   const report=await page.locator('#validationMetrics').evaluate(node=>{
    const bounds=node.getBoundingClientRect(),items=[...node.querySelectorAll('.metric')],rect=e=>e.getBoundingClientRect();
    return {width:innerWidth,overflow:document.documentElement.scrollWidth-document.documentElement.clientWidth,height:bounds.height,
     metrics:items.map(e=>[e.querySelector('span').textContent,e.querySelector('strong').textContent]),
     escaped:items.some(e=>rect(e).right>bounds.right+1||rect(e).left<bounds.left-1),
     clipped:items.flatMap(e=>[...e.children]).some(e=>e.scrollWidth>e.clientWidth+1),
     overlap:items.some((e,i)=>items.slice(i+1).some(other=>Math.min(rect(e).right,rect(other).right)>Math.max(rect(e).left,rect(other).left)+1&&Math.min(rect(e).bottom,rect(other).bottom)>Math.max(rect(e).top,rect(other).top)+1))};
   });
   assert.equal(report.overflow,0,JSON.stringify(report));assert(!report.escaped&&!report.clipped&&!report.overlap,JSON.stringify(report));
   assert.deepEqual(report.metrics,[['验证样本',String(fixture.validated.scored_rows)],...['normal','attention','abnormal'].map((key,index)=>[['正常','关注','异常'][index],String(fixture.validated.status_counts[key])]),['模型用途','正常状态模型'],['模型状态','候选'],['验证状态','验证回放完成，待工程师确认']]);
   geometryReports.push(report);
   if(process.env.SCREENING_ARTIFACT_DIR) await page.locator('#validationContent').screenshot({path:path.join(process.env.SCREENING_ARTIFACT_DIR,`validation-results-${width}.png`)});
  }
  if(process.env.SCREENING_ARTIFACT_DIR) fs.writeFileSync(path.join(process.env.SCREENING_ARTIFACT_DIR,'validation-summary-geometry.json'),JSON.stringify(geometryReports,null,2));
  await page.setViewportSize({width:1440,height:1000});
  const snapshot=()=>page.evaluate(()=>JSON.stringify([state.trainingWindows,state.candidateWindows,state.validationWindows,[...state.selectedModelTags],state.training,state.validation]));
  const before=await snapshot();
  await page.locator('[data-validation-investigation-window="N03"]').getByRole('button',{name:'查看该窗口趋势'}).click();
  await page.waitForFunction(()=>document.querySelector('#dpTrendChart')?.data?.length===10);
  assert.equal(Date.parse(await page.locator('#dpTrendStart').inputValue()),Date.parse(fixture.validated.validation_windows[0].start));
  assert.equal(Date.parse(await page.locator('#dpTrendEnd').inputValue()),Date.parse(fixture.validated.validation_windows[0].end));
  assert.equal(await page.locator('[data-validation-investigation-context="trend"] tbody tr').count(),10);
  await page.getByRole('button',{name:'返回验证结果'}).click();
  await page.locator('[data-validation-investigation-window="N03"]').getByRole('button',{name:'检查训练集覆盖'}).click();
  assert(await page.locator('#modelPanel').evaluate(e=>e.classList.contains('active')));
  assert((await page.locator('[data-validation-investigation-context="training"]').innerText()).includes('当前没有可靠的工况来源映射'));
  await page.getByRole('button',{name:'返回验证结果'}).click();
  await page.locator('[data-validation-investigation-window="A02"]').getByRole('button',{name:'查看当前建模 Tag'}).click();
  assert(await page.locator('#configPanel').evaluate(e=>e.classList.contains('active')));
  assert.equal(await page.locator('[data-validation-investigation-context="tags"] tbody tr').count(),10);
  await page.getByRole('button',{name:'返回验证结果'}).click();
  await page.locator('[data-validation-investigation-window="A02"]').getByRole('button',{name:'检查是否混入训练集'}).click();
  assert((await page.locator('[data-validation-investigation-context="training"]').innerText()).includes('该验证窗口与当前训练窗口无时间重叠'));
  assert.equal(await snapshot(),before);
  assert.equal(requests.length,3);assert(requests.every(r=>r.url().endsWith('/api/trend')&&r.postDataJSON().tags.length<=4));
  await page.evaluate(()=>{state.runId='new-run';showWorkflowStage('validationPanel')});
  assert.equal(await page.locator('[data-validation-investigation-context]').count(),0);
  assert.equal(await page.locator('[data-validation-investigation-window]').count(),0);
  assert.deepEqual(errors,[]);
 } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    try:
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=90,
                                env={**os.environ, "INVESTIGATION_URL": f"http://127.0.0.1:{server.server_port}", "INVESTIGATION_FIXTURE": str(fixture)})
        assert result.returncode == 0, result.stderr
        assert artifacts == {path.name: path.read_bytes() for path in (tmp_path / "runs" / trained["run_id"]).iterdir() if path.is_file()}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
