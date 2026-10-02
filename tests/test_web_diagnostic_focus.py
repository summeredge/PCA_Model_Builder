import json
import shutil
import subprocess

import pytest

from pca_model_builder import web


def test_diagnostic_navigation_checks_run_and_preserves_discontinuous_ranges():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML
    helpers = html[html.index("function trainingDiagnosticRun("):html.index("function renderExplorationTimeline(")]
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


def test_diagnostic_focus_renders_timeline_cluster_and_multiple_candidates():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required")
    html = web.INDEX_HTML
    helpers = html[html.index("function trainingDiagnosticRun("):html.index("function renderExplorationTimeline(")]
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
