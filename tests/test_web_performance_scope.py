import shutil
import subprocess

import pytest

from pca_model_builder import web_model_results


def test_performance_scope_controls_live_in_candidate_stage_once():
    html = web_model_results.INDEX_HTML
    for identifier in ("performanceScope", "performanceParents", "performanceClusters", "performanceScopeSummary"):
        assert html.count(f'id="{identifier}"') == 1
    assert '<option value="all" selected>全部建模资格数据</option>' in html
    assert '<option value="candidate_windows">已有候选窗口</option>' in html
    assert '<option value="cluster">指定工况组</option>' in html
    assert 'id="performanceParents" multiple' in html
    assert 'id="performanceClusters" multiple' in html


def test_performance_scope_payload_rendering_provenance_and_stale_requests():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for WebUI regression tests")
    html = web_model_results.INDEX_HTML

    def function(name):
        start = html.index(f"function {name}(")
        return html[start:html.index("\nfunction ", start)]

    start = html.index("let performanceRevision=0;")
    helpers = html[start:html.index("\nfunction performanceConditionPayload(", start)]
    start = html.index('el("performanceButton").addEventListener("click", async')
    handler = html[start:html.index('\nel("performanceScope")', start)]
    script = r'''
      const assert=require("node:assert/strict");
      class Element {
        constructor(){this.children=[];this.listeners={};this.value="";this.selected=false;this.hidden=false;this.textContent="";this.innerHTML="";}
        append(...items){this.children.push(...items);}
        replaceChildren(){this.children=[];}
        get selectedOptions(){return this.children.filter(item=>item.selected);}
        addEventListener(event,listener){this.listeners[event]=listener;}
      }
      const nodes={}; function el(id){return nodes[id]||(nodes[id]=new Element());}
      const document={createElement(){return new Element();},querySelector(){return null;}};
      const state={candidateWindows:[],trainingWindows:[],performance:null,fileId:"file-1"};
      let eligibilityRevision=0,sequence=0,requests=[],onApi=()=>{},excluded=0;
      function candidateId(){return `ui-${++sequence}`;}
      function displayTime(value){return value;}
      function displayUiValue(value){return value;}
      function metric(label,value){return `${label}:${value};`;}
      function clusterUiLabel(value){return value.replace(/cluster_(\d+)/,(_,n)=>`工况组 ${Number(n)}`);}
      function renderCandidateWindows(){refreshPerformanceScopeOptions();}
      function setStatus(){} function setBusy(){}
      function numberValue(){return 5;}
      function performanceConditionPayload(){return [{column:"quality",minimum:1}];}
      function modelingEligibilityPayload(){return {keep_conditions:[],exclude_rule_groups:[]};}
      function excludePerformanceColumns(){excluded++;}
      const result={conditions:[],total_rows:2,matched_rows:2,match_share:1,scope:{type:"candidate_windows"},parent_count:1,matched_parent_count:1,unmatched_parent_count:0,derived_candidate_count:1,
        candidate_windows:[{start:"2026-01-01T00:00:00",end:"2026-01-01T00:05:00",count:2,source:"performance",source_ref:"performance-long-backend-hash",candidate_id:"backend-id",
          provenance:{parent_candidate_id:"parent-1",parent_source:"cluster",parent_source_ref:"state-exploration-run-cluster_002-candidate-001",origin_source:"cluster",origin_source_ref:"state-exploration-run-cluster_002-candidate-001",cluster_id:"cluster_002",exploration_run_id:"run",scope_type:"candidate_windows",ordinal_in_parent:1,conditions:[{column:"quality",minimum:1}],parent_provenance:{test:"preserved"}}}],
        representative_windows:[{start:"wrong-old-result",end:"wrong-old-result",count:99}]};
      async function api(url,options){requests.push(JSON.parse(options.body));onApi();return result;}
      __LABEL__
      __HELPERS__
      __RENDER__
      __HANDLER__
      (async()=>{
        assert.deepEqual(performanceScopePayload(),{scope:{type:"all"}});
        const first={id:"parent-1",start:"2026-01-01T00:00:00",end:"2026-01-01T00:05:00",source:"cluster",source_ref:"state-exploration-run-cluster_002-candidate-001",provenance:{keep:"original"}};
        const second={id:"parent-2",start:"2026-01-01T00:10:00",end:"2026-01-01T00:15:00",source:"manual",source_ref:null};
        state.candidateWindows=[first];el("performanceScope").value="candidate_windows";refreshPerformanceScopeOptions();
        el("performanceParents").children[0].selected=true;
        const initialRevision=performanceRevision;
        refreshPerformanceScopeOptions();
        assert.equal(performanceRevision,initialRevision); // Initial parent selection without results has no side effects.
        state.performance=result;
        state.candidateWindows.push(second);refreshPerformanceScopeOptions();
        assert.deepEqual(performanceScopeParentIds,["parent-1"]);
        assert.deepEqual(performanceScopePayload().parent_windows,[first]);
        assert.equal(state.performance,result); // An unselected candidate does not invalidate explicit parent scope.
        state.performance=null;
        el("performanceParents").children.forEach(option=>option.selected=false);
        state.candidateWindows=[first,second];refreshPerformanceScopeOptions();
        assert.equal(el("performanceParents").children.length,2);
        assert.equal(el("performanceClusters").children.length,1);
        el("performanceScope").value="candidate_windows";
        assert.throws(performanceScopePayload,/至少选择/);
        el("performanceParents").children.forEach(option=>option.selected=true);
        assert.deepEqual(performanceScopePayload(),{scope:{type:"candidate_windows"},parent_windows:[first,second]});
        refreshPerformanceScopeOptions();
        assert.equal(el("performanceParents").selectedOptions.length,2);
        assert.equal(el("performanceParentsLabel").hidden,false);
        assert.match(el("performanceScopeSummary").textContent,/候选 01/);
        el("performanceScope").value="cluster";
        assert.throws(performanceScopePayload,/没有可用于/);
        el("performanceClusters").children[0].selected=true;
        assert.deepEqual(performanceScopePayload(),{scope:{type:"cluster",cluster_ids:["cluster_002"]},parent_windows:[first,second]});
        refreshPerformanceScopeOptions();
        assert.deepEqual(performanceScopeParentIds,["parent-1"]);
        state.performance=result;
        const otherCluster={...first,id:"other-cluster",source_ref:"state-exploration-run-cluster_003-candidate-001"};
        state.candidateWindows.push(otherCluster);refreshPerformanceScopeOptions();
        assert.deepEqual(performanceScopeParentIds,["parent-1"]);
        assert.equal(state.performance,result); // An unselected cluster does not change the effective parents.
        state.candidateWindows=[first,second];refreshPerformanceScopeOptions();
        assert.equal(state.performance,result);
        await el("performanceButton").listeners.click();
        assert.deepEqual(requests[0].scope,{type:"cluster",cluster_ids:["cluster_002"]});
        assert.deepEqual(requests[0].parent_windows,[first,second]);
        assert.equal(state.performance,result);
        assert.equal(el("performanceTable").children.length,1);
        const button=el("performanceTable").children[0].children[3].children[0];
        button.listeners.click();button.listeners.click();
        assert.equal(state.candidateWindows.length,3);
        assert.equal(state.trainingWindows.length,0);
        assert.equal(state.candidateWindows[0],first);
        const child=state.candidateWindows[2];
        assert.equal(child.source_ref,result.candidate_windows[0].source_ref);
        assert.equal(child.provenance,result.candidate_windows[0].provenance);
        assert.notEqual(child.id,result.candidate_windows[0].candidate_id);
        assert.equal(performanceCandidateCluster(child),"cluster_002");
        assert.deepEqual(performanceScopeParentIds,["parent-1",child.id]);
        assert.equal(state.performance,null); // Adding a derived parent in the selected cluster invalidates old results.
        assert.equal(el("performanceContent").hidden,true);
        assert.equal(el("performanceEmpty").hidden,false);
        assert.match(candidateSourceLabel(child),/工况组 2/);
        assert.ok(!candidateSourceLabel(child).includes("backend-hash"));
        assert.ok(el("performanceMetrics").innerHTML.includes("父候选数:1"));
        state.performance=result;
        state.candidateWindows=[second,child];refreshPerformanceScopeOptions();
        assert.equal(el("performanceClusters").selectedOptions.length,1);
        assert.equal(state.performance,null); // Removing one parent invalidates results even if the cluster remains.
        state.candidateWindows=[second];refreshPerformanceScopeOptions();
        assert.equal(el("performanceClusters").selectedOptions.length,0);
        assert.equal(el("performanceParents").selectedOptions.length,1);
        assert.equal(state.performance,null);
        assert.throws(performanceScopePayload,/没有可用于/);
        el("performanceScope").value="all";
        onApi=()=>invalidatePerformance();excluded=0;
        await el("performanceButton").listeners.click();
        assert.equal(state.performance,null);assert.equal(excluded,0);
        onApi=()=>eligibilityRevision++;
        await el("performanceButton").listeners.click();
        assert.equal(state.performance,null);
        state.candidateWindows=[first];el("performanceScope").value="candidate_windows";refreshPerformanceScopeOptions();el("performanceParents").children[0].selected=true;
        onApi=()=>{state.candidateWindows=[];};
        await el("performanceButton").listeners.click();
        assert.equal(state.performance,null);
        const empty={...result,candidate_windows:[]};renderPerformance(empty);
        assert.match(el("performanceTable").children[0].children[0].textContent,/没有同时满足/);
        state.performance=null;addPerformanceCandidate(result.candidate_windows[0],result);
        assert.equal(state.candidateWindows.length,0);
        assert.equal(performanceCandidateCluster({source:"manual",cluster_id:"cluster_002"}),null);
      })().catch(error=>{console.error(error);process.exitCode=1;});
    '''
    script = script.replace("__LABEL__", function("candidateSourceLabel")).replace("__HELPERS__", helpers)
    script = script.replace("__RENDER__", function("renderPerformance")).replace("__HANDLER__", handler)
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
