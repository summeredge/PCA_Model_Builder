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
function currentValidationSource(){return {file_id:state.fileId};}
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
let trendRequestRevision=0,trendSourceFileId=null;
function setBusy(){}
const state={fileId:"training",training:{id:"run"},trainingWindows:[{id:"train"}],selectedModelTags:["training-only"]};
let commonCalls=0;
function commonPayload(){commonCalls++;return {file_id:state.fileId,modeling_eligibility:{keep_conditions:[{column:"training-only",operator:">",value:0}]},state_filters:[{column:"training-only"}]}}
function populateSelectors(){}
function fillSelect(node,values){node.options=[...values]}
function currentValidationInvestigationFocus(){return focus}
function clearValidationInvestigationFocus(){focus=null}
function renderTrendPage(data){rendered=data}
function api(url,options){return new Promise((resolve,reject)=>requests.push({url,payload:JSON.parse(options.body),resolve,reject}))}
const tags=Array.from({length:10},(_,i)=>`tag${i}`);
const source={file_id:"independent-validation",timestamp_column:"validation_time",encoding:"gb18030",filename:"validation.csv",inspection:{numeric_candidate_columns:tags},label_column:"label"};
const initial={context:{source},start:"2026-01-01T10:50:03",end:"2026-01-01T11:20:04"};
function resolveBatches(items){items.forEach(item=>item.resolve({series:item.payload.tags.map(name=>({name,points:[]})),statistics:Object.fromEntries(item.payload.tags.map(tag=>[tag,{current:{sample_count:3}}])),histograms:{},axis_limits:{},ranges:{}}))}
__HELPERS__
(async()=>{
 const before=JSON.stringify(state);
 focus=initial;
 const pending=showValidationInvestigationTrend(tags,initial);
 assert.equal(requests.length,3);
 assert.deepEqual(requests.flatMap(item=>item.payload.tags),tags);
 assert(requests.every(item=>item.url==="/api/trend"&&item.payload.tags.length<=4&&item.payload.start===initial.start&&item.payload.end===initial.end));
 for(const {payload} of requests){
   assert.equal(payload.file_id,source.file_id);assert.equal(payload.timestamp_column,source.timestamp_column);assert.equal(payload.encoding,source.encoding);
   assert.equal(payload.display_mode,"raw");assert.equal(payload.purpose,"trend");
   for(const key of ["modeling_eligibility","state_filters","training_windows","label_column","inspection"]) assert.equal(key in payload,false,key);
 }
 assert.equal(commonCalls,0);assert.deepEqual(nodes.dpTrendVar1.options,tags);
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
 // An older failure must not replace a newer investigation, regardless of completion order.
 for(const oldFailsFirst of [true,false]){
   requests.length=0;focus={...initial};
   const old=showValidationInvestigationTrend(tags,focus);const oldRequests=requests.splice(0);
   focus={...initial};const newer=showValidationInvestigationTrend(tags,focus);
   const rejected=assert.doesNotReject(old);
   if(oldFailsFirst){oldRequests.forEach(item=>item.reject(new Error("old failure")));await rejected;}
   resolveBatches(requests);await newer;const latest=rendered,chart=nodes.dpTrendChart.textContent;
   if(!oldFailsFirst){oldRequests.forEach(item=>item.reject(new Error("old failure")));await rejected;}
   assert.equal(rendered,latest);assert.equal(lastTrend,latest);assert.equal(nodes.dpTrendChart.textContent,chart);
 }
 requests.length=0;focus={...initial};
 const oldSuccess=showValidationInvestigationTrend(tags,focus);const oldRequests=requests.splice(0);
 focus={...initial};const newSuccess=showValidationInvestigationTrend(tags,focus);
 resolveBatches(requests);await newSuccess;const latest=rendered;
 resolveBatches(oldRequests);await oldSuccess;assert.equal(rendered,latest);assert.equal(lastTrend,latest);
 requests.length=0;
 await showValidationInvestigationTrend(tags,initial);assert.equal(requests.length,0);
 focus={...initial,context:{}};
 assert.throws(()=>showValidationInvestigationTrend(tags,focus),/缺少数据源/);assert.equal(requests.length,0);
 assert.equal(JSON.stringify(state),before);assert.equal(commonCalls,0);
 const training=showModelTagsTrend(tags,{start:initial.start,end:initial.end},()=>true);
 assert.equal(commonCalls,3);assert(requests.every(item=>item.payload.file_id==="training"&&item.payload.modeling_eligibility));
 resolveBatches(requests);await training;
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_validation_investigation_trend_redraw_and_reset_keep_source():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = web_dataproject._DATAPROJECT_TREND_SCRIPT
    helpers = source[source.index("  async function requestTrend("):source.index('  $("dpTrendToAnalysis").addEventListener')]
    reset = source[source.index('  $("dpTrendReset").addEventListener'):source.index('  $("dpDrawScatter").addEventListener')]
    script = r'''
const assert=require("node:assert/strict"),nodes={},requests=[],messages=[];
function $(id){return nodes[id]??={value:"",textContent:"",innerHTML:"",handlers:{},replaceChildren(){this.innerHTML=""},addEventListener(event,fn){this.handlers[event]=fn},click(){return this.handlers.click()}}}
const trendIds=["dpTrendVar1","dpTrendVar2","dpTrendVar3","dpTrendVar4"];
const source={file_id:"validation",timestamp_column:"validation_time",encoding:"gb18030",inspection:{trend_default_start:"2026-02-01T00:00:00",trend_default_end:"2026-02-02T00:00:00"}};
let focus={context:{source}},lastTrend=null,rendered=null,hasDraggedTrendSelection=true,trendRequestRevision=0,commonCalls=0,trendSourceFileId=null;
const state={fileId:"training",training:{id:"run"},trainingWindows:[{id:"train"}],inspection:{trend_default_start:"2025-01-01",trend_default_end:"2025-01-02"}};
function commonPayload(){commonCalls++;return {file_id:state.fileId,modeling_eligibility:{keep_conditions:[{column:"training-only",operator:">",value:1}]}}}
function currentValidationInvestigationFocus(){return focus}
let investigationFocusClears=0;
function clearValidationInvestigationFocus(){investigationFocusClears++;focus=null}
function invalidateValidationInvestigation(){throw new Error("ordinary trends must preserve validation investigation context")}
function chosen(ids){return ids.map(id=>$(id).value).filter(Boolean)}
function syncToLegacy(){}
function setBusy(button,busy){button.disabled=busy}
function setStatus(message){messages.push(message)}
function renderTrendPage(data){rendered=data;$("dpTrendChart").textContent=data.marker}
function renderTrendChart(){throw new Error("must not redraw a cached training range during investigation")}
function localTime(value){return value}
function populateSelectors(){}
function fillSelect(){}
function api(url,options){return new Promise((resolve,reject)=>requests.push({payload:JSON.parse(options.body),resolve,reject}))}
function result(marker){return {marker,series:[],statistics:{},histograms:{},axis_limits:{},ranges:{},rows_count:1,raw_rows:1,max_points:1000}}
function snapshot(){return JSON.stringify([rendered,lastTrend,$("dpTrendChart").textContent,$("dpTrendStats").innerHTML,messages,$("dpDrawTrend").disabled])}
__HELPERS__
__RESET__
(async()=>{
 const before=JSON.stringify(state);
 $("dpTrendVar1").value="validation-tag";$("dpTrendStart").value="2026-02-01T01:00:00";$("dpTrendEnd").value="2026-02-01T02:00:00";$("dpTrendMaxPoints").value="1000";
 let pending=$("dpDrawTrend").click(),request=requests.shift();
 assert.equal(request.payload.file_id,source.file_id);assert.equal(request.payload.timestamp_column,source.timestamp_column);assert.equal(request.payload.encoding,source.encoding);
 assert.equal("modeling_eligibility" in request.payload,false);assert.deepEqual(request.payload.tags,["validation-tag"]);
 request.resolve(result("initial"));await pending;
 // Edited controls keep the same source; same-focus requests must not overwrite each other.
 for(const oldFails of [false,true]) for(const oldFirst of [false,true]){
   const old=$("dpDrawTrend").click(),oldRequest=requests.shift();
   $("dpTrendVar1").value="changed-tag";$("dpTrendStart").value="2026-02-01T03:00:00";
   const newer=$("dpDrawTrend").click(),newRequest=requests.shift();
   assert.equal(newRequest.payload.file_id,"validation");assert.equal(newRequest.payload.start,$("dpTrendStart").value);assert.deepEqual(newRequest.payload.tags,["changed-tag"]);
   const finishOld=()=>oldFails?oldRequest.reject(new Error("stale error")):oldRequest.resolve(result("stale"));
   if(oldFirst){const saved=snapshot();finishOld();await old;assert.equal(snapshot(),saved);assert.equal($("dpDrawTrend").disabled,true)}
   newRequest.resolve(result("latest"));await newer;const saved=snapshot();
   if(!oldFirst){finishOld();await old;assert.equal(snapshot(),saved)}
   assert.equal(rendered.marker,"latest");assert.equal($("dpDrawTrend").disabled,false);
 }
 // Changing or clearing the investigation invalidates pending redraws.
 for(const nextFocus of [{context:{source:{...source,file_id:"other"}}},null]) for(const fails of [false,true]){
   focus={context:{source}};const old=$("dpDrawTrend").click(),request=requests.shift();
   focus=nextFocus;const saved=JSON.stringify([rendered,lastTrend,messages,$("dpTrendChart").textContent]);
   if(fails)request.reject(new Error("stale focus"));else request.resolve(result("stale focus"));await old;
   assert.equal(JSON.stringify([rendered,lastTrend,messages,$("dpTrendChart").textContent]),saved);
 }
 // A manual redraw supersedes an outstanding initial investigation batch, including its error.
 for(const fails of [false,true]){
   focus={context:{source},start:"2026-02-01T00:00:00",end:"2026-02-02T00:00:00"};
   const initial=showValidationInvestigationTrend(["tag"],focus),batch=requests.shift();
   const redraw=$("dpDrawTrend").click(),request=requests.shift();request.resolve(result("redraw"));await redraw;
   const saved=snapshot();if(fails)batch.reject(new Error("stale batch"));else batch.resolve(result("stale batch"));await initial;assert.equal(snapshot(),saved);
 }
 // Opening another investigation also supersedes a pending manual redraw.
 const oldRedraw=$("dpDrawTrend").click(),oldRequest=requests.shift();
 focus={...focus};const initial=showValidationInvestigationTrend(["tag"],focus),batch=requests.shift();
 batch.resolve(result("new investigation"));await initial;const saved=snapshot();
 oldRequest.reject(new Error("old redraw"));await oldRedraw;assert.equal(snapshot(),saved);
 const failed=$("dpDrawTrend").click();requests.shift().reject(new Error("current failure"));await failed;
 assert.equal($("dpTrendChart").textContent,"current failure");assert.equal(messages.at(-1),"current failure");assert.equal($("dpDrawTrend").disabled,false);
 focus={context:{source}};$("dpTrendReset").click();request=requests.shift();
 assert.equal(request.payload.file_id,"validation");assert.equal(request.payload.start,source.inspection.trend_default_start);assert.equal(request.payload.end,source.inspection.trend_default_end);assert.equal(request.payload.max_points,100000);
 request.resolve(result("reset"));await new Promise(resolve=>setImmediate(resolve));
 focus={context:{source:{...source,inspection:null}}};const start=$("dpTrendStart").value;
 $("dpTrendReset").click();assert.equal(requests.length,0);assert.equal($("dpTrendStart").value,start);assert.match(messages.at(-1),/数据检查/);
 focus={context:{}};await $("dpDrawTrend").click();assert.equal(requests.length,0);assert.match(messages.at(-1),/缺少数据源/);
 assert.equal(commonCalls,0);assert.equal(JSON.stringify(state),before);
 focus={context:{source}};
 const report={run_id:"run",contributions:[{tag:"validation-tag"}]},context={source,report};
 state.validation=report;state.validationInvestigationContext=context;
 const diagnostic=showModelTagsTrend(["training-tag"],{start:"2025-01-01",end:"2025-01-02"},()=>true);
 request=requests.shift();assert.equal(focus,null);assert.equal(investigationFocusClears,1);assert.equal(request.payload.file_id,"training");
 request.resolve(result("diagnostic"));await diagnostic;
 assert.equal(state.validation,report);assert.equal(state.validationInvestigationContext,context);assert.equal(context.report,report);
 focus=null;const training=$("dpDrawTrend").click();request=requests.shift();assert.equal(request.payload.file_id,"training");assert.equal(commonCalls,2);request.resolve(result("training"));await training;
 assert.equal(state.validation,report);assert.equal(state.validationInvestigationContext,context);
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__HELPERS__", helpers).replace("__RESET__", reset)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_external_validation_trend_cannot_write_modeling_windows():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = web_dataproject._DATAPROJECT_TREND_SCRIPT
    handlers = source[source.index('  $("dpTrendToAnalysis").addEventListener'):source.index('  $("dpTrendReset").addEventListener')]
    script = r'''
const assert=require("node:assert/strict"),nodes={},messages=[],writes=[];
function $(id){return nodes[id]??={value:`old-${id}`,addEventListener(event,fn){this.click=fn}}}
let focus,lastTrend=null,trendSourceFileId="external";
const state={fileId:"training",training:{id:"run"},trainingWindows:[{id:"train"}],candidateWindows:[],excludedWindows:[]};
function currentValidationInvestigationFocus(){return focus}
function setStatus(message,level){messages.push({message,level})}
function invalidateModellingResults(){writes.push("invalidate")}
function addCandidateWindow(...args){state.candidateWindows.push(args)}
function addExcludedWindow(...args){state.excludedWindows.push(args)}
__HANDLERS__
const buttons=["dpTrendToAnalysis","dpTrendToReference","dpTrendToExclusion"];
$("dpTrendStart").value="validation-start";$("dpTrendEnd").value="validation-end";
for(const id of ["analysisStart","analysisEnd","explorationStart","explorationEnd"]) $(id);
focus={context:{source:{file_id:"external"}}};
const before=JSON.stringify(state),ranges=JSON.stringify(Object.fromEntries(Object.entries(nodes).map(([id,node])=>[id,node.value])));
buttons.forEach(id=>$(id).click());
assert.equal(JSON.stringify(state),before);assert.equal(JSON.stringify(Object.fromEntries(Object.entries(nodes).map(([id,node])=>[id,node.value]))),ranges);assert.deepEqual(writes,[]);
assert.equal(messages.length,3);assert(messages.every(item=>item.level==="warning"&&item.message.includes("独立验证文件")&&item.message.includes("不能写入")));
// Clearing investigation focus does not turn a displayed external chart into training data.
lastTrend={source_file_id:"external"};focus=null;buttons.forEach(id=>$(id).click());
assert.equal(JSON.stringify(state),before);assert.deepEqual(writes,[]);
assert.equal(JSON.stringify(Object.fromEntries(Object.entries(nodes).map(([id,node])=>[id,node.value]))),ranges);
assert.equal(messages.length,6);
lastTrend={source_file_id:"training"};
trendSourceFileId="training";
// Ordinary browsing and investigation of the training file retain their existing writes.
for(const nextFocus of [null,{context:{source:{file_id:"training"}}}]){
 focus=nextFocus;const candidates=state.candidateWindows.length,excluded=state.excludedWindows.length;
 buttons.forEach(id=>$(id).click());
 assert.equal($("analysisStart").value,"validation-start");assert.equal($("explorationEnd").value,"validation-end");
 assert.equal(state.candidateWindows.length,candidates+1);assert.equal(state.excludedWindows.length,excluded+1);
}
assert.deepEqual(state.training,{id:"run"});assert.deepEqual(state.trainingWindows,[{id:"train"}]);
'''.replace("__HANDLERS__", handlers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cached_training", [False, True], ids=["no-result", "training-result"])
def test_pending_validation_trend_range_keeps_source_after_focus_clear(cached_training):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = web_dataproject._DATAPROJECT_TREND_SCRIPT
    helpers = source[source.index("  async function requestTrend("):source.index('  $("dpTrendReset").addEventListener')]
    script = r'''
const assert=require("node:assert/strict"),nodes={},requests=[],messages=[];
function $(id){return nodes[id]??={value:"",textContent:"",handlers:{},replaceChildren(){},addEventListener(event,fn){this.handlers[event]=fn},click(){return this.handlers.click()}}}
const trendIds=["dpTrendVar1","dpTrendVar2","dpTrendVar3","dpTrendVar4"];
let focus=null,lastTrend=__CACHED__?{source_file_id:"training"}:null;
let trendSourceFileId=null,trendRequestRevision=0,hasDraggedTrendSelection=false,rendered=null;
const state={fileId:"training",training:{run_id:"fixed"},candidateWindows:[],excludedWindows:[],invalidations:0};
function commonPayload(){return {file_id:state.fileId,timestamp_column:"time"}}
function currentValidationInvestigationFocus(){return focus}
function clearValidationInvestigationFocus(){focus=null}
function populateSelectors(){} function fillSelect(){} function syncToLegacy(){}
function chosen(ids){return ids.map(id=>$(id).value).filter(Boolean)}
function setBusy(button,busy){button.disabled=busy}
function setStatus(message,level){messages.push({message,level})}
function renderTrendPage(data){rendered=data}
function invalidateModellingResults(){state.invalidations++}
function addCandidateWindow(...args){state.candidateWindows.push(args)}
function addExcludedWindow(...args){state.excludedWindows.push(args)}
function api(url,options){return new Promise((resolve,reject)=>requests.push({payload:JSON.parse(options.body),resolve,reject}))}
function result(marker){return {marker,series:[],statistics:{},histograms:{},axis_limits:{},ranges:{}}}
const buttons=["dpTrendToAnalysis","dpTrendToReference","dpTrendToExclusion"];
const rangeIds=["analysisStart","analysisEnd","explorationStart","explorationEnd"];
rangeIds.forEach(id=>$(id).value=`original-${id}`);
const original=JSON.stringify([state,rangeIds.map(id=>$(id).value)]);
function assertBlocked(){
 const count=messages.length;buttons.forEach(id=>$(id).click());
 assert.equal(JSON.stringify([state,rangeIds.map(id=>$(id).value)]),original);
 assert.equal(messages.length,count+3);
 assert(messages.slice(-3).every(item=>item.level==="warning"&&item.message.includes("独立验证文件")));
 assert.equal(trendSourceFileId,"validation");
}
function openValidation(){
 focus={context:{source:{file_id:"validation",timestamp_column:"vtime"}},start:"2026-02-01",end:"2026-02-02"};
 const task=showValidationInvestigationTrend(["A"],focus),request=requests.shift();
 assert.equal(request.payload.file_id,"validation");assert.equal(trendSourceFileId,"validation");
 return {task,request};
}
function openTraining(){
 const task=showModelTagsTrend(["A"],{start:"2026-01-01",end:"2026-01-02"},()=>true),request=requests.shift();
 assert.equal(focus,null);assert.equal(request.payload.file_id,"training");assert.equal(trendSourceFileId,"training");
 return {task,request};
}
__HELPERS__
(async()=>{
 // Waiting, failed, successful and stale external requests all retain the input range's identity.
 for(const fails of [false,true]) for(const clearBeforeFinish of [false,true]){
  lastTrend=__CACHED__?{source_file_id:"training"}:null;
  const {task,request}=openValidation();assertBlocked();
  const completion=fails&&!clearBeforeFinish?assert.rejects(task,/read failed/):task;
  if(clearBeforeFinish)clearValidationInvestigationFocus();
  assertBlocked();
  if(fails)request.reject(new Error("read failed"));else request.resolve(result("external"));
  await completion;clearValidationInvestigationFocus();assertBlocked();
 }
 // A redraw starts a new request but never loses external ownership when focus disappears.
 const initial=openValidation();
 $("dpTrendVar1").value="A";const redraw=$("dpDrawTrend").click(),redrawRequest=requests.shift();
 clearValidationInvestigationFocus();assertBlocked();redrawRequest.reject(new Error("redraw failed"));await redraw;assertBlocked();
 initial.request.resolve(result("old batch"));await initial.task;assertBlocked();
 // Both directions and completion orders: responses never assign source ownership.
 for(const validationIsNew of [false,true]) for(const oldFirst of [false,true]) for(const oldFails of [false,true]){
  const old=validationIsNew?openTraining():openValidation();
  const newer=validationIsNew?openValidation():openTraining();
  const expected=validationIsNew?"validation":"training";
  const finishOld=()=>oldFails?old.request.reject(new Error("old failure")):old.request.resolve(result("old"));
  if(oldFirst){finishOld();await old.task;assert.equal(trendSourceFileId,expected)}
  newer.request.resolve(result("new"));await newer.task;
  if(!oldFirst){finishOld();await old.task}
  assert.equal(trendSourceFileId,expected);assert.equal(rendered.marker,"new");
  if(validationIsNew){clearValidationInvestigationFocus();assertBlocked()}
 }
 // Explicitly return to training; all three operations work even before its request completes.
 const training=openTraining();buttons.forEach(id=>$(id).click());
 assert.equal($("analysisStart").value,"2026-01-01");assert.equal($("explorationEnd").value,"2026-01-02");
 assert.equal(state.candidateWindows.length,1);assert.equal(state.excludedWindows.length,1);assert.equal(state.invalidations,1);
 training.request.resolve(result("training"));await training.task;
 assert.equal(requests.length,0);
})().catch(error=>{console.error(error);process.exitCode=1});
'''.replace("__HELPERS__", helpers).replace("__CACHED__", str(cached_training).lower())
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
function currentValidationSource(){return {file_id:state.fileId,inspection:{}};}
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
 await newer;assert.equal(pending.length,1); // One writer per run until replay completes.
 invalidateValidationInvestigation();pending.shift()({oldResponse:true});await older;
 const latest=validateClick();pending.shift()(response);await latest;
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
   renderTraining(f.trained);renderValidationWindows();renderTagList();state.validation=f.validated;renderValidation(f.validated);showWorkflowStage("validationPanel");
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
  await page.getByRole('button',{name:'返回验证结果'}).click();
  await page.locator('#validationDecision').selectOption('failed');
  await page.locator('#validationWindowTable tr').first().getByRole('button',{name:'编辑',exact:true}).click();
  await page.locator('#validationEnd').fill('2026-01-01T12:20');
  await page.locator('#addValidationWindow').click();
  assert.equal(await page.locator('#validationDecision').inputValue(),'hold');
  await page.locator('#validationWindowTable tr').first().getByRole('button',{name:'编辑',exact:true}).click();
  assert.equal(await page.locator('#addValidationWindow').getAttribute('data-editing-id'),'N03');
  const inspected=page.waitForResponse(r=>r.url().endsWith('/api/inspect'));
  await page.evaluate(()=>{el('inspectButton').disabled=false;el('inspectButton').click();});
  assert((await inspected).ok());
  await page.waitForFunction(()=>!document.querySelector('#addValidationWindow').dataset.editingId);
  assert.equal(await page.locator('#addValidationWindow').innerText(),'加入验证时段');
  await page.evaluate(()=>showWorkflowStage('validationPanel'));
  await page.locator('#validationStart').fill('2026-01-01T10:50');await page.locator('#validationEnd').fill('2026-01-01T12:20');
  await page.locator('#addValidationWindow').click();
  assert.equal(await page.locator('#validationWindowTable tr').count(),1);
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
