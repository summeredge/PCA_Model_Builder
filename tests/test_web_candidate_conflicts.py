import copy
import re
import shutil
import subprocess

import pandas as pd
import pytest

from pca_model_builder import web, web_model_results


def window(identifier, start=0, end=10, enabled=True):
    return dict(id=identifier, start=f"2026-01-01T00:{start:02d}:00",
                end=f"2026-01-01T00:{end:02d}:00", source="cluster",
                source_ref="parent-source", enabled=enabled, comment="保留历史")


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=11, freq="5min"),
                            "A": range(11), "quality": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]})
    uploaded = web.save_upload("conflicts.csv", history.to_csv(index=False).encode())
    candidate = {key: value for key, value in window("B", 0, 50).items() if key != "enabled"}
    candidate.update(source="performance", source_ref="refined-source",
                     provenance={"parent_candidate_id": "A", "cluster_id": "cluster_002"})
    return dict(file_id=uploaded["file_id"], timestamp_column="time", training_windows=[],
                operation=dict(action="replace_with_candidate", candidate=candidate,
                               conflict_window_ids=[], excluded_windows=[]))


@pytest.mark.parametrize("existing", [[], [window("training-A")],
                         [window("training-A"), window("other", 20, 30)],
                         [window("disabled", enabled=False)]])
def test_replace_disables_all_actual_conflicts_and_preserves_history(payload, existing):
    payload["training_windows"] = existing
    payload["operation"]["conflict_window_ids"] = [w["id"] for w in existing if w["enabled"]]
    original = copy.deepcopy(payload)
    preview = web.training_windows_payload({**payload, "operation": {**payload["operation"], "action": "preview_candidate"}})
    assert [w["id"] for w in preview["conflicts"]] == payload["operation"]["conflict_window_ids"]
    result = web.training_windows_payload(payload)["training_windows"]
    assert result[:-1] == [{**w, "enabled": False} for w in existing]
    assert result[-1]["id"] == "training-B"
    assert result[-1]["enabled"] is True
    assert result[-1]["source"] == "performance"
    assert result[-1]["source_ref"] == "refined-source"
    assert payload == original
    if not any(w["enabled"] for w in existing):
        ordinary = web.training_windows_payload({**payload, "operation": {**payload["operation"], "action": "confirm_candidate"}})
        assert ordinary["training_windows"] == result


@pytest.mark.parametrize("ids", [["training-A"], [], ["training-A", "other", "extra"],
                              ["training-A", "other", "other"], None, [1]])
def test_changed_or_invalid_conflicts_reject_without_partial_mutation(payload, ids):
    payload["training_windows"] = [window("training-A"), window("other", 20, 30)]
    payload["operation"]["conflict_window_ids"] = ids
    original = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="训练窗口冲突已变化，请刷新后重新确认"):
        web.training_windows_payload(payload)
    assert payload == original


@pytest.mark.parametrize("cut", ["excluded", "eligibility"])
def test_conflicts_use_final_parts_not_candidate_envelope(payload, cut):
    payload["training_windows"] = [window("training-A", 0, 10), window("gap", 15, 25), window("other", 30, 40)]
    payload["operation"]["conflict_window_ids"] = ["other", "training-A"]
    if cut == "excluded":
        excluded = window("excluded", 15, 25)
        payload["operation"]["excluded_windows"] = [{k: v for k, v in excluded.items() if k not in {"enabled", "source_ref"}}]
    else:
        payload["modeling_eligibility"] = {"keep_conditions": [{"column": "quality", "minimum": 1}], "exclude_rule_groups": []}
    result = web.training_windows_payload(payload)["training_windows"]
    assert result[1] == payload["training_windows"][1]
    assert [w["enabled"] for w in result] == [False, True, False, True, True]
    assert [w["id"] for w in result[3:]] == ["training-B-part-001", "training-B-part-002"]
    preview = web.training_windows_payload({**payload, "operation": {**payload["operation"], "action": "preview_candidate"}})
    assert [w["id"] for w in preview["conflicts"]] == ["training-A", "other"]


@pytest.mark.parametrize("action", ["add", "update", "set_enabled", "confirm_candidate"])
def test_regular_operations_still_enforce_enabled_overlap(payload, action):
    payload["training_windows"] = [window("training-A"), window("disabled", 20, 30, False)]
    operations = {
        "add": dict(window=window("new", 10, 15)),
        "update": dict(id="training-A", changes={"end": "2026-01-01T00:30:00"}),
        "set_enabled": dict(id="disabled", enabled=True),
        "confirm_candidate": dict(candidate=payload["operation"]["candidate"]),
    }
    if action == "update":
        payload["training_windows"][1]["enabled"] = True
    if action == "set_enabled":
        payload["training_windows"][1].update(start="2026-01-01T00:10:00")
    payload["operation"] = dict(action=action, **operations[action])
    with pytest.raises(ValueError, match="启用的training_windows不能重叠"):
        web.training_windows_payload(payload)


def test_candidate_conflict_ui_and_replacement_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for WebUI regression tests")
    html = web_model_results.INDEX_HTML
    names = ["candidateTrainingWindows", "windowsOverlap", "candidateTrainingConflicts", "isParentTrainingWindow",
             "showCandidateConflicts", "renderCandidateWindows", "updateTrainingWindows", "confirmCandidateWindow"]
    functions = []
    for name in names:
        start = html.index(f"function {name}(")
        if html[start - 6:start] == "async ":
            start -= 6
        end = re.search(r"\n(?:async )?function ", html[start:]).start() + start
        functions.append(html[start:end])
    script = r'''
    const assert=require('node:assert/strict');
    class Element {
      constructor(){this.children=[];this.listeners={};this.textContent='';}
      append(...items){this.children.push(...items);} replaceChildren(){this.children=[];}
      addEventListener(event,fn){this.listeners[event]=fn;} scrollIntoView(){}
      text(){return this.textContent+' '+this.children.map(item=>item.text()).join(' ');}
      buttons(){return this.children.flatMap(item=>[...(item.type==='button'?[item]:[]),...item.buttons()]);}
    }
    const nodes={};function el(id){return nodes[id]||(nodes[id]=new Element());}
    const document={createElement:()=>new Element()};
    let eligibilityRevision=0,requests=[],invalidated=0,onApi=()=>{};
    const parent={id:'A',start:'2026-01-01T00:00:00',end:'2026-01-01T00:10:00'};
    const child={...parent,id:'B',provenance:{parent_candidate_id:'A'}};
    const first={...parent,id:'training-A',enabled:true,source_ref:'parent-ref'};
    const other={...parent,id:'unrelated',enabled:true};
    const state={candidateWindows:[parent,child],trainingWindows:[first],excludedWindows:[],trainingWindowSummary:[]};
    function commonPayload(){return {};} function trainingWindowsPayload(){return state.trainingWindows;}
    function displayTime(value){return value;}function displayUiValue(value){return value;}
    function candidateSourceLabel(value){return value.id;}
    function renderTrainingWindows(){} function updateQualityButtonAvailability(){}
    function invalidateQuality(){invalidated++;} function setStatus(){} function showCandidateTrend(){}
    async function api(url,options){const body=JSON.parse(options.body);requests.push(body);onApi();
      if(body.operation.action==='preview_candidate')return {conflicts:candidateTrainingConflicts(body.operation.candidate)};
      return {training_windows:[{...first,enabled:false},{...child,id:'training-B',enabled:true}],summary:[]};}
    __FUNCTIONS__
    (async()=>{
      assert(windowsOverlap(parent,{start:parent.end,end:parent.end}));
      assert(isParentTrainingWindow(child,first));
      assert(isParentTrainingWindow(child,{id:'training-A-part-002'}));
      assert(!isParentTrainingWindow(child,{id:'training-AX'}));
      assert(!isParentTrainingWindow(child,other));
      renderCandidateWindows();assert.match(el('candidateWindows').text(),/与 1 个/);
      assert.equal(el('candidateWindows').buttons().filter(b=>b.textContent==='确认作为训练窗口').length,1);
      await confirmCandidateWindow(child);assert.equal(requests.length,0);
      state.trainingWindows.push(other);renderCandidateWindows();assert.match(el('candidateWindows').text(),/与 2 个/);
      const details=new Element();await showCandidateConflicts(child,details);
      assert.match(details.text(),/父候选训练窗口/);assert.match(details.text(),/其它训练窗口/);
      assert.match(details.text(),/parent-ref/);assert.match(details.text(),/2026-01-01T00:10:00/);
      const before=JSON.stringify(state);details.buttons()[0].listeners.click();assert.equal(JSON.stringify(state),before);
      await showCandidateConflicts(child,details);
      state.trainingWindows.pop();await details.buttons()[1].listeners.click();assert.equal(requests.length,2);
      await showCandidateConflicts(child,details);await details.buttons()[1].listeners.click();
      assert.equal(requests.at(-1).operation.action,'replace_with_candidate');
      assert.deepEqual(requests.at(-1).operation.conflict_window_ids,['training-A']);
      assert.equal(invalidated,1);assert.match(el('candidateWindows').text(),/训练窗口已禁用/);
      assert.match(el('candidateWindows').text(),/已生成训练窗口/);
      assert.equal(candidateTrainingConflicts(parent).length,1);
      state.trainingWindows=[{...first,enabled:false}];assert.equal(candidateTrainingConflicts(child).length,0);
      onApi=()=>eligibilityRevision++;const prior=state.trainingWindows;
      assert.equal(await updateTrainingWindows({action:'replace_with_candidate'},true),false);
      assert.equal(state.trainingWindows,prior);assert.equal(invalidated,1);
      await showCandidateConflicts(child,details);assert.equal(requests.at(-1).operation.action,'preview_candidate');
    })().catch(error=>{console.error(error);process.exitCode=1;});
    '''.replace("__FUNCTIONS__", "\n".join(functions))
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
