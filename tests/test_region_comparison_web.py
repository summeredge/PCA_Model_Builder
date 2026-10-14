"""Read-only comparison API and latest-selection UI behavior."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web
from test_web import _run_node_javascript


def test_region_comparison_payload_is_read_only_and_uses_confirmed_a(monkeypatch):
    run_id = "e" * 32
    index = pd.date_range("2026-01-01", periods=8, freq="5min")
    points = pd.DataFrame({"pc1": np.arange(8.), "pc2": 0.,
                           "cluster_id": ["cluster_001"] * 8, "segment_id": 0}, index=index)
    ellipse = lambda x: {"center_pc1": x, "center_pc2": 0., "radius_pc1": 1.6, "radius_pc2": 1.}
    cached = {"cluster_series": points, "cluster_centers": {"cluster_001": [0., 0.]},
              "exploratory_model_summary": {"pc_columns": ["pc1", "pc2"]},
              "_color_series": pd.DataFrame({"TAG": np.arange(8.) * 10, "PERF": 99.}, index=index),
              "variable_diagnostics": {"tag_profiles": [{"tag": "TAG"}]},
              "preferred_region": {"ellipses": [ellipse(1.)]},
              "preferred_region_update_seq": 3,
              "preferred_region_candidates": [{"candidate_id": "keep"}],
              "candidate_decisions": [{"candidate_id": "keep", "decision": "accepted"}]}
    monkeypatch.setitem(web.STATE_EXPLORATION_RUNS, run_id, cached)
    before = copy.deepcopy(cached)
    payload = {"exploration_run_id": run_id, "a_ellipses": [ellipse(1.)], "b_ellipses": [ellipse(6.)]}
    result = web.state_exploration_region_comparison_payload(run_id, payload)
    assert result["a_matches"] is True
    assert result["region_a"]["selected_sample_count"] == result["region_b"]["selected_sample_count"] == 3
    assert result["intersection_sample_count"] == 0
    assert result["a_only_sample_count"] == result["b_only_sample_count"] == 3
    assert result["variables"][0]["a_mean"] == 10.
    assert result["variables"][0]["b_mean"] == 60.
    assert [v["tag"] for v in result["variables"]] == ["TAG"]
    json.dumps(result, allow_nan=False)
    assert not any(key.startswith("_") or key in {"cluster_series", "candidate_decisions", "preferred_region_candidates"} for key in result)
    for key, value in before.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(cached[key], value)
        else:
            assert cached[key] == value
    stale = web.state_exploration_region_comparison_payload(run_id, {**payload, "a_ellipses": []})
    assert stale == {"exploration_run_id": run_id, "a_matches": False}
    with pytest.raises(ValueError, match="不一致"):
        web.state_exploration_region_comparison_payload(run_id, {**payload, "exploration_run_id": "other"})
    with pytest.raises(ValueError, match="列表"):
        web.state_exploration_region_comparison_payload(run_id, {"exploration_run_id": run_id})
    with pytest.raises(ValueError):
        web.state_exploration_region_comparison_payload(run_id, {**payload, "b_ellipses": [{"radius_pc1": 0}]})
    with pytest.raises(web.StateExplorationNotFoundError):
        web.state_exploration_region_comparison_payload("f" * 32, {**payload, "exploration_run_id": "f" * 32})


def test_region_comparison_latest_response_failure_clear_and_a_updates():
    source = web.INDEX_HTML
    functions = source[source.index("async function updateExplorationPreferredRegion("):source.index('el("uploadButton").addEventListener')]
    reset = "function resetExplorationRegion()" + source.split("function resetExplorationRegion()", 1)[1].split("\n", 1)[0]
    result = _run_node_javascript(r"""
const assert=require('node:assert/strict');
const state={exploration:{exploration_run_id:'run',preferred_region_candidates:[{candidate_id:'keep'}],candidate_decisions:[{candidate_id:'keep'}]},preferredRegion:{ellipses:[{id:'a'}],selected_sample_count:4},preferredRegionRequest:0,preferredRegionUpdateSeq:0,referenceRegionEllipses:[],regionComparison:null,regionComparisonRequest:0,explorationPlotRevision:0,trainingWindows:[{id:'human'}]};
const calls=[];function api(url,options){return new Promise((resolve,reject)=>calls.push({url,body:JSON.parse(options.body),resolve,reject}));}
function renderExplorationRegionFeatures(){} function renderExplorationRegionControls(){} function refreshExplorationRegionPlot(){return true;} function renderExplorationRegionSummary(){} function renderExplorationCandidateTables(){} function setStatus(){} function clearTrainingDiagnosticFocus(){}
__FUNCTIONS__
__RESET__
(async()=>{
 const invariant=JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows,state.preferredRegionUpdateSeq]);
 const one=updateExplorationReferenceRegion([{id:1}]),two=updateExplorationReferenceRegion([{id:2}]);
 calls[1].resolve({variables:[],id:2});await two;calls[0].resolve({variables:[],id:1});await one;
 assert.equal(state.regionComparison.id,2);
 const old=updateExplorationReferenceRegion([{id:3}]),latest=updateExplorationReferenceRegion([{id:4}]);
 calls[3].resolve({variables:[],id:4});await latest;calls[2].reject(new Error('old'));await old;assert.equal(state.regionComparison.id,4);
 const fail=updateExplorationReferenceRegion([{id:5}]);calls[4].reject(new Error('offline'));await fail;
 assert(state.regionComparison.error.includes('offline'));assert.deepEqual(state.referenceRegionEllipses,[{id:5}]);
 const pending=updateExplorationReferenceRegion([{id:6}]);await updateExplorationReferenceRegion([]);calls[5].resolve({id:6});await pending;
 assert.equal(state.regionComparison,null);assert.deepEqual(state.referenceRegionEllipses,[]);
 assert.equal(JSON.stringify([state.preferredRegion,state.exploration.preferred_region_candidates,state.exploration.candidate_decisions,state.trainingWindows,state.preferredRegionUpdateSeq]),invariant);
 assert(calls.every(c=>c.url.endsWith('/region-comparison')));
 const waiting=updateExplorationReferenceRegion([{id:7}]);resetExplorationRegion();state.exploration={exploration_run_id:'new'};calls[6].resolve({id:7});await waiting;assert.equal(state.regionComparison,null);assert.deepEqual(state.referenceRegionEllipses,[]);
 state.preferredRegion={ellipses:[{id:'a'}],selected_sample_count:4};state.exploration={exploration_run_id:'new',preferred_region_candidates:[],candidate_decisions:[]};
 const a=updateExplorationPreferredRegion([{id:'newA'}]);await updateExplorationReferenceRegion([{id:8}]);assert.equal(calls.length,8);assert(state.regionComparison.pending);
 calls[7].resolve({preferred_region:{ellipses:[{id:'newA'}],selected_sample_count:3},preferred_region_update_seq:1});await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(calls[8].body.a_ellipses,[{id:'newA'}]);calls[8].resolve({variables:[],id:8});await a;assert.equal(state.regionComparison.id,8);
 const failedA=updateExplorationPreferredRegion([{id:'failed'}]);calls[9].reject(new Error('A failed'));await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(calls[10].body.a_ellipses,[{id:'newA'}]);calls[10].resolve({variables:[],id:9});await failedA;assert.equal(state.regionComparison.id,9);
 const stale=refreshExplorationRegionComparison();calls[11].resolve({a_matches:false});await stale;assert(state.regionComparison.error.includes('服务端不一致'));
 console.log(JSON.stringify({status:'PASS',requests:calls.length}));
})().catch(e=>{console.error(e);process.exitCode=1;});
""".replace("__FUNCTIONS__", functions).replace("__RESET__", reset))
    assert result == {"status": "PASS", "requests": 12}
