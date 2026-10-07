import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pca_model_builder import web


def test_diagnostic_navigation_checks_run_and_preserves_discontinuous_ranges():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML
    helpers = html[html.index("function windowTimeRange("):html.index("function trainingWindowsPayload(")] + html[html.index("function trainingDiagnosticRun("):html.index("function renderExplorationTimeline(")]
    script = r'''
const state={candidateWindows:[{id:"c1"},{id:"c2"}],trainingWindows:[{id:"w1"}],exploration:{exploration_run_id:"run-a",cluster_summaries:[{cluster_id:"cluster_001"}]}};
const calls=[],messages=[];
function el(){return {querySelector(){return null},scrollIntoView(){calls.push("scroll")}}}
function renderStateExploration(){calls.push("render")}
function renderCandidateWindows(){}
function setStatus(message){messages.push(message)}
function clusterUiLabel(id){return id}
function displayTime(t){return t}
globalThis.showWorkflowStage=id=>calls.push(id);
globalThis.showCandidateTool=id=>calls.push(id);
__HELPERS__
const group={cluster_id:"cluster_001",source_ref:"state-exploration-run-a-cluster_001"};
const timeline=Array.from({length:8},(_,i)=>({timestamp:new Date(Date.UTC(2026,0,1,0,i)).toISOString(),cluster_id:"cluster_001",near_switch:i!==2,window_id:i<4?"w1":"w2",segment_id:i<6?1:2,break_before:i===7}));
const diagnostic={groups:[group],timeline,switch_diagnostic:{available:true}};
state.training={model_quality:{training_condition_diagnostic:diagnostic}};
const before=JSON.stringify([state.candidateWindows,state.trainingWindows,state.training]);
focusTrainingDiagnostic(diagnostic,group);
if(!calls.includes("candidatePanel")||!calls.includes("stateExplorationPanel")||state.trainingDiagnosticFocus.clusterId!=="cluster_001") throw Error("navigation");
focusTrainingDiagnostic(diagnostic);
if(JSON.stringify(state.trainingDiagnosticFocus.intervals.map(x=>[x.start,x.end]))!==JSON.stringify([[0,1],[3,3],[4,5],[6,6],[7,7]].map(pair=>pair.map(i=>timeline[i].timestamp)))) throw Error("boundaries");
if(before!==JSON.stringify([state.candidateWindows,state.trainingWindows,state.training])) throw Error("mutation");
state.exploration.exploration_run_id="other";
focusTrainingDiagnostic(diagnostic,group);
if(state.trainingDiagnosticFocus||!messages.at(-1).includes("不匹配")) throw Error("wrong run");
state.exploration.exploration_run_id="run-a";
focusTrainingDiagnostic(diagnostic,group);
state.training={};
if(currentTrainingDiagnosticFocus()) throw Error("stale training");
state.training={model_quality:{training_condition_diagnostic:diagnostic}};
state.trainingDiagnosticInvalidated=true;
focusTrainingDiagnostic(diagnostic,group);
if(state.trainingDiagnosticFocus||!messages.at(-1).includes("已失效")) throw Error("invalidated quality");
state.trainingDiagnosticInvalidated=false;
diagnostic.switch_diagnostic.available=false;
focusTrainingDiagnostic(diagnostic);
if(state.trainingDiagnosticFocus||!messages.at(-1).includes("没有可定位")) throw Error("unavailable switches");
focusTrainingDiagnostic(diagnostic,group); clearTrainingDiagnosticFocus();
if(state.trainingDiagnosticFocus) throw Error("not cleared");
state.exploration=null;
focusTrainingDiagnostic(diagnostic,group);
if(state.trainingDiagnosticFocus) throw Error("missing exploration");
console.log(JSON.stringify({ok:true}));
'''.replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"]


def test_diagnostic_actions_select_exact_parents_and_locate_current_training_windows():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML

    def function(name):
        start = html.index(f"function {name}(")
        return html[start:html.index("\nfunction ", start)]

    helpers = html[html.index("function windowTimeRange("):html.index("function trainingWindowsPayload(")] + html[html.index("function trainingDiagnosticCandidateMatches("):html.index("function renderExplorationTimeline(")]
    scope = html[html.index("let performanceRevision=0;"):html.index("function addPerformanceCandidate(")]
    script = r'''
const assert=require("node:assert/strict");
class Element {
  constructor(){this.children=[];this.style={};this.attrs={};this.value="";this.textContent="";this.selected=false;this.classList={toggle(){}};}
  append(...items){this.children.push(...items)}
  replaceChildren(){this.children=[]}
  setAttribute(k,v){this.attrs[k]=v}
  addEventListener(){}
  get options(){return this.children}
  get selectedOptions(){return this.children.filter(item=>item.selected)}
  scrollIntoView(){calls.push("scroll")}
  querySelector(){return null}
}
const nodes={},document={createElement:()=>new Element()};
function el(id){return nodes[id]??=new Element()}
const calls=[],messages=[];
function api(){throw Error("Navigation must not call an API")}
function setStatus(message){messages.push(message)}
function displayTime(value){return value}
function displayUiValue(value){return value}
function clusterUiLabel(id){return `工况组 ${Number(id.slice(8))}`}
function candidateSourceLabel(window){return clusterUiLabel(window.provenance?.cluster_id||"cluster_002")}
function renderStateExploration(){calls.push("exploration")}
function renderCandidateWindows(){refreshPerformanceScopeOptions()}
function renderPreprocessingPreviewWindow(){}
function windowSummary(){return {}}
globalThis.showWorkflowStage=id=>calls.push(id);
globalThis.showCandidateTool=id=>calls.push(id);
const group={cluster_id:"cluster_002",source_ref:"state-exploration-run-a-cluster_002"};
const diagnostic={available:true,groups:[group],timeline:[{window_id:"direct",cluster_id:"cluster_002"},{window_id:"other-run",cluster_id:"cluster_002"}]};
const direct={id:"a",source:"cluster",source_ref:"state-exploration-run-a-cluster_002-candidate-001",start:"2026-01-01T00:00",end:"2026-01-01T00:15"};
const refined={...direct,id:"refined",source:"performance",source_ref:"performance-refined",provenance:{origin_source:"cluster",origin_source_ref:direct.source_ref,exploration_run_id:"run-a",cluster_id:"cluster_002"}};
const otherRun={...direct,id:"b",source_ref:direct.source_ref.replace("run-a","run-b")};
const otherGroup={...direct,id:"other-group",source_ref:direct.source_ref.replace("cluster_002","cluster_003")};
const otherRefined={...refined,id:"other-refined",source_ref:"performance-other",provenance:{...refined.provenance,origin_source_ref:otherRun.source_ref,exploration_run_id:"run-b"}};
const invalidRefinements=[undefined,{}, {...refined.provenance,origin_source:"manual"}, {...refined.provenance,cluster_id:"cluster_003"}, {...refined.provenance,exploration_run_id:"run-b"}, {...refined.provenance,origin_source_ref:otherRun.source_ref}].map((provenance,i)=>({...refined,id:`invalid-${i}`,source_ref:`invalid-${i}`,provenance}));
const windows=[{...direct,id:"direct",enabled:true},{...refined,id:"refined-enabled",enabled:true},{...refined,id:"refined-disabled",enabled:false},{...otherRun,id:"other-run",enabled:true},{...otherGroup,id:"other-group",enabled:true},{...otherRefined,id:"other-refined",enabled:false}];
// Window objects use the existing schema; performance origin lives in the sidecar.
windows.forEach(window=>delete window.provenance);
const record=(window,candidate)=>({window_id:window.id,source:window.source,source_ref:window.source_ref,start:window.start,end:window.end,candidate});
const sidecar=[record(windows[1],refined),record(windows[2],refined),record(windows[5],otherRefined)];
const state={candidateWindows:[direct,otherRun,otherGroup,refined,otherRefined,...invalidRefinements],trainingWindows:windows,trainingCandidateProvenance:sidecar,training:{model_quality:{training_condition_diagnostic:diagnostic}},exploration:{exploration_run_id:"run-a",cluster_summaries:[group]}};
__SCOPE__
__HELPERS__
__RENDER__
assert.equal(trainingDiagnosticTrainingWindowMatches({...windows[0],source_ref:group.source_ref},{runId:"run-a",clusterId:group.cluster_id}),true);
assert.equal(trainingDiagnosticTrainingWindowMatches({...windows[0],source_ref:group.source_ref.replace("run-a","run-b")},{runId:"run-a",clusterId:group.cluster_id}),false);
const snapshot=()=>JSON.stringify([state.candidateWindows,state.trainingWindows,state.trainingCandidateProvenance,state.training]);
const original=snapshot();
for(const action of ["exploration","screen","training_windows"]) {
  calls.length=0;focusTrainingDiagnostic(diagnostic,group,action);
  assert.equal(snapshot(),original,action+" mutated model state");
  assert.equal(state.trainingDiagnosticFocus.runId,"run-a");
  assert.equal(state.trainingDiagnosticFocus.clusterId,"cluster_002");
  assert.ok(calls.includes("scroll"));
  if(action==="screen") {
    assert.ok(calls.includes("candidatePanel")&&calls.includes("performancePanel"));
    assert.deepEqual(performanceScopePayload().scope,{type:"candidate_windows"});
    assert.deepEqual(performanceScopePayload().parent_windows,[direct,refined]);
    assert.deepEqual(performanceScopeParentIds,["a","refined"]);
    assert.match(el("performanceScopeSummary").textContent,/继续筛选 工况组 2/);
    assert.ok(!el("performanceScopeSummary").textContent.includes("run-a"));
  } else if(action==="training_windows") {
    assert.ok(calls.includes("modelPanel"));
    assert.deepEqual(state.trainingDiagnosticFocus.trainingWindowIds,["direct","refined-enabled","refined-disabled"]);
    assert.match(messages.at(-1),/关联 3 个训练窗口：2 个启用，1 个已禁用/);
    const rows=el("trainingWindows").children[0].children[1].children;
    assert.deepEqual(rows.map(row=>!!row.attrs["data-diagnostic-focus"]),[true,true,true,false,false,false]);
    assert.deepEqual(rows.map(row=>row.children[0].children[0].checked),[true,true,false,true,true,false]);
  } else assert.ok(calls.includes("stateExplorationPanel"));
}
clearTrainingDiagnosticFocus();
assert.ok(el("trainingWindows").children[0].children[1].children.every(row=>!row.attrs["data-diagnostic-focus"]));
// Retained provenance works after the performance candidate was removed.
state.candidateWindows=[direct];
focusTrainingDiagnostic(diagnostic,group,"screen");
assert.deepEqual(performanceScopePayload().parent_windows,[direct]);
focusTrainingDiagnostic(diagnostic,group,"training_windows");
assert.deepEqual(state.trainingDiagnosticFocus.trainingWindowIds,["direct","refined-enabled","refined-disabled"]);
state.candidateWindows=[refined];
focusTrainingDiagnostic(diagnostic,group,"screen");
assert.deepEqual(performanceScopePayload().parent_windows,[refined]);
// No sidecar: a unique pool candidate still supplies reliable performance origin.
state.trainingCandidateProvenance=[];
focusTrainingDiagnostic(diagnostic,group,"training_windows");
assert.deepEqual(state.trainingDiagnosticFocus.trainingWindowIds,["direct","refined-enabled","refined-disabled"]);
state.candidateWindows=[refined,{...refined,id:"duplicate"}];
assert.equal(trainingDiagnosticTrainingWindowMatches(windows[1],state.trainingDiagnosticFocus),false);
state.candidateWindows=[refined];
for(const change of [{start:"changed"},{end:"changed"},{source_ref:"changed"},{source:"cluster"},{candidate:{...refined,source_ref:"changed"}},{candidate:{...refined,provenance:{...refined.provenance,exploration_run_id:"run-b"}}}]) {
  state.trainingCandidateProvenance=[{...sidecar[0],...change}];
  assert.equal(trainingDiagnosticTrainingWindowMatches(windows[1],{runId:"run-a",clusterId:"cluster_002"}),false);
}
state.trainingCandidateProvenance=[sidecar[0],sidecar[0]];
assert.equal(trainingDiagnosticTrainingWindowMatches(windows[1],{runId:"run-a",clusterId:"cluster_002"}),false);
state.trainingCandidateProvenance=sidecar;
// Reusing an ID for a different window must not inherit its highlight.
focusTrainingDiagnostic(diagnostic,group,"training_windows");
state.trainingWindows=windows.map(window=>window.id==="direct"?{...window,source_ref:otherRun.source_ref}:window);
renderTrainingWindows();assert.equal(currentTrainingDiagnosticFocus(),null);
assert.ok(el("trainingWindows").children[0].children[1].children.every(row=>!row.attrs["data-diagnostic-focus"]));
state.trainingWindows=windows;
state.candidateWindows=[];
focusTrainingDiagnostic(diagnostic,group,"screen");
assert.equal(state.trainingDiagnosticFocus,null);assert.match(messages.at(-1),/原候选已不存在/);
state.trainingWindows=[windows[3],windows[4],windows[5]];
focusTrainingDiagnostic(diagnostic,group,"training_windows");
assert.equal(state.trainingDiagnosticFocus,null);assert.match(messages.at(-1),/当前没有可定位/);
// Every action rejects stale, foreign and invalid diagnostic contexts.
state.candidateWindows=[direct,refined];state.trainingWindows=windows;
const training=state.training,exploration=state.exploration;
for(const action of ["exploration","screen","training_windows"]) {
  for(const invalidate of [()=>state.training={},()=>state.trainingDiagnosticInvalidated=true,()=>state.exploration={...exploration,exploration_run_id:"run-b"},()=>state.exploration=null,()=>training.model_quality.training_condition_diagnostic={...diagnostic}]) {
    focusTrainingDiagnostic(diagnostic,group,action);invalidate();calls.length=0;
    focusTrainingDiagnostic(diagnostic,group,action);
    assert.equal(state.trainingDiagnosticFocus,null);
    assert.ok(!calls.includes("modelPanel")&&!calls.includes("candidatePanel"));
    assert.match(messages.at(-1),/已失效|不匹配/);
    state.training=training;state.exploration=exploration;state.trainingDiagnosticInvalidated=false;training.model_quality.training_condition_diagnostic=diagnostic;
  }
  focusTrainingDiagnostic(diagnostic,{...group},action);
  assert.equal(state.trainingDiagnosticFocus,null);
}
focusTrainingDiagnostic(diagnostic,group,"training_windows");
training.model_quality.training_condition_diagnostic={...diagnostic};
assert.equal(currentTrainingDiagnosticFocus(),null);
'''.replace("__SCOPE__", scope).replace("__HELPERS__", helpers).replace("__RENDER__", function("renderTrainingWindows"))
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_diagnostic_group_buttons_share_context_and_hide_untraceable_sources():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    source = Path(web.__file__).with_name("model_results.js").read_text(encoding="utf-8")
    renderer = source[source.index("  function trainingAnomalyPeaks("):source.index("  async function focusTrainingAnomalyPeak(")] + source[source.index("  function renderTrainingConditionDiagnostic("):source.index("  function drawTrainingConditionTrend(")]
    html = web.INDEX_HTML
    parser = html[html.index("function trainingDiagnosticRun("):html.index("function trainingDiagnosticTrainingWindowMatches(")]
    script = r'''
const assert=require("node:assert/strict");
class Element {
  constructor(){this.children=[];this.classList={toggle(){}}}
  append(...nodes){this.children.push(...nodes)} replaceChildren(){this.children=[]}
}
const nodes={},document={getElementById:id=>nodes[id]??=new Element(),createElement:()=>new Element()};
function clusterUiLabel(id){return id}
function drawTrainingConditionTrend(){}
    function el(){return {value:"5"}}
const calls=[];globalThis.focusTrainingDiagnostic=(...args)=>calls.push(args);
__PARSER__
__RENDERER__
const group={cluster_id:"cluster_002",source_ref:"state-exploration-run-a-cluster_002"};
const diagnostic={available:true,groups:[group,{...group,source_ref:"legacy"},{...group,source_ref:"state-exploration-run-a-cluster_003"}]};
renderTrainingConditionDiagnostic(diagnostic);
const rows=nodes.modelTrainingConditionGroups.children[0].children[1].children;
const buttons=rows[0].children.at(-1).children[0].children;
assert.deepEqual(buttons.map(button=>button.textContent),["查看该工况组","继续筛选该工况组","查看关联训练窗口"]);
buttons.forEach(button=>button.onclick());
assert.deepEqual(calls.map(call=>call[2]),["exploration","screen","training_windows"]);
calls.forEach(([d,g])=>{assert.equal(d,diagnostic);assert.equal(g,group)});
rows.slice(1).forEach(row=>{assert.equal(row.children.at(-1).textContent,"来源不可追溯");assert.equal(row.children.at(-1).children.length,0)});
'''.replace("__PARSER__", parser).replace("__RENDERER__", renderer)
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


def test_diagnostic_focus_renders_timeline_cluster_and_multiple_candidates():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML
    helpers = html[html.index("function windowTimeRange("):html.index("function trainingWindowsPayload(")] + html[html.index("function trainingDiagnosticRun("):html.index("function renderExplorationTimeline(")]
    renderers = html[html.index("function renderExplorationTimeline("):html.index("function resetExplorationRegion(")]
    script = r'''
class Element {
  constructor(){this.children=[];this.style={};this.attrs={};this.innerHTML="";}
  append(...nodes){this.children.push(...nodes)}
  replaceChildren(){this.children=[]}
  setAttribute(k,v){this.attrs[k]=v}
}
const nodes={}, document={createElement:()=>new Element()};
function el(id){return nodes[id]??=new Element()}
function escapeHtml(x){return String(x)}
function displayTime(x){return String(x)}
function displayUiValue(x){return String(x)}
function explorationClusterColor(){return "#176b87"}
function clusterUiLabel(x){return x}
function explorationNumber(x){return x}
function explorationPercent(x){return x}
function stateExplorationCandidateSourceRef(r,c){return `state-exploration-${r}-${c}`}
const training={},state={training,exploration:{exploration_run_id:"r"},candidateWindows:[],trainingWindows:[]};
__HELPERS__
__RENDERERS__
const rows=[0,1,3,4].map((i,n)=>({timestamp:new Date(Date.UTC(2026,0,1,0,i)).toISOString(),cluster_id:"cluster_001",segment_id:n<2?1:2,break_before:n===2}));
state.trainingDiagnosticFocus={training,runId:"r",clusterId:"cluster_001",intervals:[]};
renderExplorationTimeline(rows,[]);
if((nodes.explorationTimeline.innerHTML.match(/data-diagnostic-focus/g)||[]).length!==2) throw Error("merged cluster ranges");
renderExplorationClusterTable([{cluster_id:"cluster_001"},{cluster_id:"cluster_002"}]);
if(nodes.explorationClusterTable.children[0].attrs["data-diagnostic-focus"]!=="true"||nodes.explorationClusterTable.children[1].attrs["data-diagnostic-focus"]) throw Error("cluster highlight");
const candidates=[1,2].map(n=>({candidate_id:`cluster_001-candidate-00${n}`,cluster_id:"cluster_001",start:rows[0].timestamp,end:rows[1].timestamp}));
renderExplorationCandidateTables(candidates,[],[]);
if((nodes.explorationClusterCandidates.innerHTML.match(/data-diagnostic-focus/g)||[]).length!==2) throw Error("candidate highlight");
state.trainingDiagnosticFocus={training,runId:"r",intervals:[{start:rows[0].timestamp,end:rows[1].timestamp},{start:rows[2].timestamp,end:rows[3].timestamp}]};
renderExplorationTimeline(rows,[]);
if((nodes.explorationTimeline.innerHTML.match(/data-diagnostic-focus/g)||[]).length!==2) throw Error("switch highlight");
state.exploration.exploration_run_id="new";
renderExplorationTimeline(rows,[]); renderExplorationClusterTable([{cluster_id:"cluster_001"}]); renderExplorationCandidateTables(candidates,[],[]);
if(nodes.explorationTimeline.innerHTML.includes("data-diagnostic-focus")||nodes.explorationClusterCandidates.innerHTML.includes("data-diagnostic-focus")||nodes.explorationClusterTable.children[0].attrs["data-diagnostic-focus"]) throw Error("stale highlight");
'''.replace("__HELPERS__", helpers).replace("__RENDERERS__", renderers)
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
