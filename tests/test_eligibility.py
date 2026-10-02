import inspect
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web, web_model_results, web_dataproject
from pca_model_builder.eligibility import filter_modeling_eligibility, normalize_modeling_eligibility
from pca_model_builder.preprocessing import PreprocessingConfig, StateFilter, preprocess_window
from pca_model_builder.screening import screen_performance_states
from pca_model_builder.state_exploration import ExplorationConfig, run_state_exploration
from pca_model_builder.training import build_training_matrix
from pca_model_builder.model_io import load_model_package, load_deployment_package


def condition(column="gate", minimum=None, maximum=None):
    return {"column": column, "minimum": minimum, "maximum": maximum}


def history(count=120, frequency="5min"):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=count, freq=frequency),
                         "A": rng.normal(size=count), "B": rng.normal(size=count),
                         "C": rng.normal(size=count), "gate": np.arange(count, dtype=float)})
    frame["C"] = frame.A + frame.B + .1 * frame.C
    return frame


def config(**kwargs):
    return PreprocessingConfig(sample_interval_minutes=5, filter_method="none", max_lag_minutes=0, **kwargs)


def test_boolean_rules_inclusive_boundaries_priority_and_summary():
    indexed = history(10).set_index("time")
    indexed["other"] = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    rules = {"keep_conditions": [condition(minimum=1, maximum=8), condition("other", minimum=1)],
             "exclude_rule_groups": [[condition(minimum=2, maximum=4), condition("other", minimum=1)],
                                     [condition(minimum=7, maximum=9)]]}
    result = filter_modeling_eligibility(indexed, rules, config())
    assert result.frame.gate.tolist() == [1, 6]
    assert result.summary == {"original_samples": 10, "keep_pass_samples": 6,
                              "exclude_hit_samples": 5, "eligible_samples": 2,
                              "eligible_share": .2, "segment_count": 2}


@pytest.mark.parametrize("rules", [None, {}, {"keep_conditions": [], "exclude_rule_groups": []}])
def test_no_rules_preserve_every_sample_and_preprocessing(rules):
    indexed = history().set_index("time")
    eligible = filter_modeling_eligibility(indexed, rules, config())
    pd.testing.assert_frame_equal(eligible.frame, indexed)
    assert eligible.summary["eligible_share"] == 1
    expected = preprocess_window(indexed, ["A", "B"], config()).dynamic
    actual = preprocess_window(indexed, ["A", "B"], config(), modeling_eligibility=rules).dynamic
    pd.testing.assert_frame_equal(actual, expected)
    minute_data = history(30, "1min").set_index("time")
    expected_screen = screen_performance_states(minute_data, [condition(minimum=0)], 5)
    assert screen_performance_states(minute_data, [condition(minimum=0)], 5, modeling_eligibility=rules) == expected_screen


@pytest.mark.parametrize("rules,pattern", [
    ({"keep_conditions": [condition(minimum="bad")]}, "numeric"),
    ({"keep_conditions": [condition(minimum=float("inf"))]}, "finite"),
    ({"keep_conditions": [condition(minimum=float("nan"))]}, "finite"),
    ({"keep_conditions": [condition(minimum=3, maximum=2)]}, "reversed"),
    ({"keep_conditions": [condition()]}, "bound"),
    ({"keep_conditions": {}}, "列表"),
    ({"exclude_rule_groups": [[]]}, "至少一个条件"),
    ({"state_filters": []}, "modeling_eligibility"),
])
def test_invalid_payload(rules, pattern):
    with pytest.raises(ValueError, match=pattern):
        normalize_modeling_eligibility(rules)


def test_single_keep_empty_result_missing_and_non_numeric_column():
    indexed = history(10).set_index("time")
    rules = {"keep_conditions": [condition(minimum=3, maximum=5)]}
    assert filter_modeling_eligibility(indexed, rules, config()).frame.gate.tolist() == [3, 4, 5]
    with pytest.raises(ValueError, match="缺少列"):
        filter_modeling_eligibility(indexed.drop(columns="gate"), rules, config())
    indexed.loc[indexed.index[0], "gate"] = np.nan
    with pytest.raises(ValueError, match="有限数值"):
        filter_modeling_eligibility(indexed, rules, config())
    rules = {"keep_conditions": [condition(minimum=1000)]}
    with pytest.raises(ValueError, match="没有合格样本"):
        filter_modeling_eligibility(history(10).set_index("time"), rules, config())
    assert filter_modeling_eligibility(history(10).set_index("time"), rules, config(), allow_empty=True).summary["segment_count"] == 0


@pytest.mark.parametrize("method", ["none", "first_order", "trailing_mean"])
def test_filter_lag_training_and_candidates_never_cross_excluded_rows(method):
    frame = history()
    indexed = frame.set_index("time")
    rules = {"exclude_rule_groups": [[condition(minimum=40, maximum=59)]]}
    cfg = PreprocessingConfig(sample_interval_minutes=5, max_lag_minutes=5, lag_step_minutes=5,
                              smoothing_window_minutes=10, filter_method=method,
                              first_order_alpha=.5 if method == "first_order" else None,
                              gap_threshold_minutes=1000)
    eligible = filter_modeling_eligibility(indexed, rules, cfg)
    assert eligible.summary["segment_count"] == 2
    processed = preprocess_window(indexed, ["A", "B", "C"], cfg, modeling_eligibility=rules, include_intermediates=True)
    assert frame.time.iloc[60] not in processed.dynamic.index
    for start, end in [(0, 39), (60, 119)]:
        standalone = preprocess_window(indexed.iloc[start:end+1], ["A", "B", "C"], cfg).dynamic
        pd.testing.assert_frame_equal(processed.dynamic.loc[indexed.index[start]:indexed.index[end]], standalone)
    windows = [{"id": "manual", "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[-1].isoformat(), "enabled": True, "source": "manual", "source_ref": None, "comment": ""}]
    trained = build_training_matrix(frame, "time", ["A", "B", "C"], cfg, windows, modeling_eligibility=rules)
    pd.testing.assert_frame_equal(trained.dynamic, processed.dynamic)
    assert "gate" not in trained.dynamic.columns
    explored = run_state_exploration(indexed, ["A", "B", "C"], cfg,
                                     ExplorationConfig(cluster_count=2, minimum_candidate_duration_minutes=5), modeling_eligibility=rules)
    assert set(explored["cluster_series"].index).issubset(set(eligible.frame.index))
    assert explored["cluster_series"].segment_id.nunique() == 2
    for candidate in explored["cluster_candidates"]:
        assert not (pd.Timestamp(candidate["start"]) < frame.time.iloc[50] < pd.Timestamp(candidate["end"]))
    screened = screen_performance_states(indexed, [condition(minimum=0)], 5, modeling_eligibility=rules)
    assert screened["total_rows"] == 100
    assert sorted(window["count"] for window in screened["representative_windows"]) == [40, 60]


def test_resampling_drops_buckets_touching_eligibility_gap_and_preserves_source_cadence():
    indexed = history(60, "1min").set_index("time")
    rules = {"exclude_rule_groups": [[condition(minimum=22, maximum=22)]]}
    cfg = config(resampling_method="mean", gap_threshold_minutes=1000)
    processed = preprocess_window(indexed, ["A", "B"], cfg, modeling_eligibility=rules, include_intermediates=True)
    assert indexed.index[25] not in processed.dynamic.index
    assert processed.raw_segment_ids.nunique() == 2
    assert processed.summary.source_interval_minutes == 1


@pytest.fixture
def uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    frame = history()
    saved = web.save_upload("eligibility.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    payload = {"file_id": saved["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"],
               "sample_interval_minutes": 5, "filter_method": "none", "max_lag_minutes": 0,
               "normal_start": frame.time.iloc[0].isoformat(), "normal_end": frame.time.iloc[-1].isoformat(),
               "analysis_start": frame.time.iloc[0].isoformat(), "analysis_end": frame.time.iloc[-1].isoformat(),
               "exploration_start": frame.time.iloc[0].isoformat(), "exploration_end": frame.time.iloc[-1].isoformat(),
               "model_name": "eligible", "modeling_eligibility": {"exclude_rule_groups": [[condition(minimum=40, maximum=59)]]}}
    yield frame, payload, tmp_path
    web.clear_state_exploration_cache()


def test_web_summary_exploration_cluster_quality_training_and_manual_confirmation(uploaded):
    frame, payload, tmp_path = uploaded
    summary = web.modeling_eligibility_payload(payload)["summary"]
    assert summary["eligible_samples"] == 100
    assert summary["segment_count"] == 2
    qualified = web.modeling_eligibility_payload({**payload, "candidate_start": payload["normal_start"], "candidate_end": payload["normal_end"]})["eligible_windows"]
    assert qualified == [{"start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[39].isoformat()},
                         {"start": frame.time.iloc[60].isoformat(), "end": frame.time.iloc[-1].isoformat()}]
    explored = web.state_exploration_payload(payload)
    assert len(explored["cluster_series_full"]) == 100
    cluster = web.cluster_payload(payload)
    assert cluster["sample_count"] == 100
    screened = web.performance_screen_payload({**payload, "conditions": [condition(minimum=0)]})
    assert screened["total_rows"] == 100
    quality = web.quality_payload(payload)
    assert quality["training_window_totals"]["training_rows"] == 100
    trained = web.train_payload(payload)
    assert trained["training_rows"] == 100
    _, manifest = load_model_package(tmp_path / "runs" / trained["run_id"] / "model.pcamodel")
    assert manifest["config"]["state_filters"] == []
    assert "modeling_eligibility" not in manifest["config"]
    windows = web.training_windows_payload({**payload, "training_windows": [], "operation": {"action": "confirm_candidate", "candidate": {
        "id": "manual-candidate", "start": payload["normal_start"], "end": payload["normal_end"], "source": "manual", "source_ref": "manual-candidate", "comment": "test"}}})["training_windows"]
    assert len(windows) == 2
    assert windows[0]["end"] == frame.time.iloc[39].isoformat()
    assert windows[1]["start"] == frame.time.iloc[60].isoformat()
    assert all(window["source_ref"] == "manual-candidate" for window in windows)


def test_web_all_excluded_missing_column_and_trend_keeps_history(uploaded):
    frame, payload, _ = uploaded
    empty = {**payload, "modeling_eligibility": {"keep_conditions": [condition(minimum=1000)]}}
    assert web.modeling_eligibility_payload(empty)["summary"]["eligible_samples"] == 0
    for function in [web.state_exploration_payload, web.cluster_payload]:
        with pytest.raises(ValueError, match="没有合格样本"):
            function(empty)
    with pytest.raises(ValueError, match="没有合格训练样本"):
        web.train_payload(empty)
    assert web.quality_payload(empty)["can_train"] is False
    with pytest.raises(ValueError, match="missing|找不到|缺少"):
        web.modeling_eligibility_payload({**payload, "modeling_eligibility": {"keep_conditions": [condition("missing", minimum=0)]}})
    trend = web_dataproject.trend_payload({**empty, "purpose": "trend", "tags": ["A"], "start": payload["normal_start"], "end": payload["normal_end"]})
    assert trend["raw_rows"] == len(frame)
    assert trend["modeling_eligibility_summary"]["eligible_samples"] == 0


def test_state_filter_preprocessing_unchanged_and_runtime_has_no_eligibility():
    indexed = history().set_index("time")
    cfg = config(state_filters=(StateFilter("gate", minimum=10, maximum=90),))
    expected = preprocess_window(indexed, ["A", "B"], cfg)
    actual = preprocess_window(indexed, ["A", "B"], cfg, modeling_eligibility={})
    pd.testing.assert_frame_equal(expected.dynamic, actual.dynamic)
    assert cfg.to_dict()["state_filters"] == [{"column": "gate", "minimum": 10, "maximum": 90}]
    assert "modeling_eligibility" not in cfg.to_dict()
    for function in [web._validate_payload_locked, web._frozen_replay_payload_locked]:
        assert "modeling_eligibility" not in inspect.getsource(function)


def test_validation_frozen_replay_and_deployment_ignore_offline_eligibility(uploaded):
    frame, payload, tmp_path = uploaded
    rules = {"exclude_rule_groups": [[condition(minimum=20, maximum=29)]]}
    payload = {**payload, "normal_end": frame.time.iloc[59].isoformat(), "modeling_eligibility": rules,
               "state_filters": [condition(minimum=5, maximum=110)], "tag_configs": {"gate": {"role": "state_filter"}}}
    trained = web.train_payload(payload)
    assert trained["training_rows"] == 45
    candidate_model, _ = load_model_package(tmp_path / "runs" / trained["run_id"] / "model.pcamodel")
    for position, t2 in [(60, 0), (61, (candidate_model.t2_limits[.95] + candidate_model.t2_limits[.99]) / 2), (62, candidate_model.t2_limits[.99] * 2)]:
        frame.loc[position, payload["tags"]] = candidate_model.mean + candidate_model.scale * candidate_model.components[0] * np.sqrt(t2 * candidate_model.eigenvalues[0])
    validation_upload = web.save_upload("validation.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    windows = [{"id": name, "type": kind, "start": frame.time.iloc[start].isoformat(),
                "end": frame.time.iloc[end].isoformat(), "enabled": True, "comment": ""}
               for name, kind, start, end in [("normal", "normal_validation", 60, 79), ("abnormal", "known_abnormal", 80, 110)]]
    request = {"run_id": trained["run_id"], "file_id": validation_upload["file_id"], "timestamp_column": "time", "validation_windows": windows}
    expected = web.validate_payload(request)
    # Even a malformed offline eligibility rule must never be read by scoring.
    forbidden = {"keep_conditions": [condition("missing", minimum=float("inf"))]}
    actual = web.validate_payload({**request, "modeling_eligibility": forbidden})
    assert actual["scores"] == expected["scores"]
    assert actual["scored_rows"] == 51
    web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "test"})
    web.freeze_deployment_payload({"run_id": trained["run_id"], "model_id": "eligibility.test", "model_version": 1, "frozen_by": "test", "comment": "test"})
    replay_request = {"run_id": trained["run_id"], "file_id": validation_upload["file_id"], "timestamp_column": "time", "replay_start": frame.time.iloc[0].isoformat(), "replay_end": frame.time.iloc[-1].isoformat()}
    replay = web.frozen_replay_payload(replay_request)
    assert web.frozen_replay_payload({**replay_request, "modeling_eligibility": forbidden})["scores"] == replay["scores"]
    run_dir = tmp_path / "runs" / trained["run_id"]
    deployment_model, manifest = load_deployment_package(run_dir / "deployment_model.pcadeploy")
    assert manifest["preprocessing"]["state_filters"] == payload["state_filters"]
    assert "modeling_eligibility" not in manifest["preprocessing"]
    runtime_config = web.preprocessing_config_from_mapping(manifest["preprocessing"])
    processed = preprocess_window(frame.set_index("time"), payload["tags"], runtime_config)
    assert set(frame.time.iloc[20:30]).issubset(set(processed.dynamic.index))
    frozen = web.replay_frozen_model(run_dir / "frozen_model.pcamodel", frame.set_index("time"), frame.time.iloc[0], frame.time.iloc[-1])
    expected_scores = deployment_model.score_dynamic_features(processed.dynamic.to_numpy())
    np.testing.assert_allclose(frozen.scores.loc[processed.dynamic.index, "t2"], expected_scores.t2)
    np.testing.assert_allclose(frozen.scores.loc[processed.dynamic.index, "spe"], expected_scores.spe)


def test_ui_rules_above_discovery_unique_ids_and_stale_result_guards():
    html = web_model_results.INDEX_HTML
    assert html.index('id="modelingEligibility"') < html.index('id="stateExplorationPanel"')
    assert "当前探索仅使用建模资格通过的数据" in html
    for field in ["modelingEligibility", "eligibilityKeepConditions", "eligibilityExcludeGroups", "eligibilitySummary"]:
        assert html.count(f'id="{field}"') == 1
    assert "modeling_eligibility:modelingEligibilityPayload()" in html
    assert "if(revision!==eligibilityRevision)" in html
    assert "state.trainingWindows.map(window=>({...window,enabled:false}))" in html
    assert "exclude_rule_groups" in html and "keep_conditions" in html


def test_ui_editor_add_delete_payload_summary_and_invalidation():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for eligibility editor regression tests")
    html = web_model_results.INDEX_HTML
    script = html[html.index("let eligibilityRevision="):html.index("function stateFilterTags()")]
    renderer = html[html.index("function renderTrainingWindows()"):html.index("async function updateTrainingWindows(")]
    helpers = "\n".join(next(line for line in html.splitlines() if line.startswith(f"function {name}(")) for name in ["windowSummary", "updateQualityButtonAvailability"])
    harness = r'''
      class Element {
        constructor(tag="div") { this.tag=tag; this.children=[]; this.value=""; this.dataset={}; this.handlers={}; this.className=""; this.hidden=false; this.textContent=""; this.classList={contains:name=>this.className.split(" ").includes(name)}; }
        append(...nodes) { nodes.forEach(node=>{node.parentElement=this;this.children.push(node);}); }
        remove() { this.parentElement.children=this.parentElement.children.filter(node=>node!==this); }
        replaceChildren() { this.children=[]; }
        text() { return this.textContent+this.children.map(child=>child.text()).join(" "); }
        addEventListener(name,handler) { this.handlers[name]=handler; }
        querySelectorAll(selector) {
          const match=node=>selector===".eligibility-condition"?node.classList.contains("eligibility-condition"):selector==="input"?node.tag==="input":selector.includes("data-field")?selector.includes(`"${node.dataset.field}"`):false;
          return this.children.flatMap(child=>[...(match(child)?[child]:[]),...child.querySelectorAll(selector)]);
        }
        querySelector(selector) { return this.querySelectorAll(selector)[0]||null; }
      }
      const nodes=Object.fromEntries(["eligibilityKeepConditions","eligibilityExcludeGroups","eligibilitySummary","addEligibilityKeep","addEligibilityExcludeGroup","refreshEligibilitySummary","modelContent","modelEmpty","modelDownload","validateButton","trainingWindows","qualityButton"].map(id=>[id,new Element()]));
      const el=id=>nodes[id]||null, document={createElement:tag=>new Element(tag)};
      const state={inspection:{numeric_columns:["gate","other"]},selectedModelTags:new Set(["A"]),candidateWindows:[],trainingWindows:[{id:"manual",enabled:true,source_ref:"manual"}],trainingWindowSummary:[],quality:{},training:{},runId:"old",exploratoryRunId:"old",clustering:{},performance:{}};
      let invalidations=0;
      function invalidateModellingResults() { invalidations+=1;state.quality=null; }
      function renderCandidateWindows() {}
      function renderPreprocessingPreviewWindow() {}
      function candidateSourceLabel(window) { return window.source; }
      function displayTime(value) { return value; }
      function displayUiValue(value) { return value; }
      function setStatus() {}
      function fillSelect(select,columns) { select.value=columns[0]; }
      function formField(labelText,field,type) { const label=new Element("label"),input=new Element("input");input.dataset.field=field;label.append(input);return label; }
      function setTimeout() { return 0; }
      function clearTimeout() {}
      __RENDERER__
      __HELPERS__
      __SCRIPT__
      const keep=el("eligibilityKeepConditions"); addEligibilityCondition(keep); addEligibilityCondition(keep);
      keep.children.forEach((row,index)=>{ row.querySelector('[data-field="minimum"]').value=String(index+1);row.querySelector('[data-field="maximum"]').value="10"; });
      addEligibilityExcludeGroup();addEligibilityExcludeGroup();
      el("eligibilityExcludeGroups").children.forEach(group=>group.querySelector('[data-field="minimum"]').value="5");
      const payload=modelingEligibilityPayload();
      const first=keep.children[0];first.children[3].handlers.click();
      const group=el("eligibilityExcludeGroups").children[0];group.querySelector(".eligibility-condition").children[3].handlers.click();
      state.candidateWindows=[{id:"stale"}];state.trainingWindows=[{id:"manual",source:"manual",enabled:true,source_ref:"manual"}];state.training={};state.runId="stale";
      state.trainingWindowSummary=[{id:"manual",raw_samples:120,effective_samples:118,quality_status:"passed"}];
      renderTrainingWindows();updateQualityButtonAvailability();
      const before={text:el("trainingWindows").text(),disabled:el("qualityButton").disabled};
      eligibilityChanged();
      renderEligibilitySummary({original_samples:10,keep_pass_samples:8,exclude_hit_samples:3,eligible_samples:5,eligible_share:.5,segment_count:2});
      const invalid=keep.children[0];invalid.querySelector('[data-field="minimum"]').value="20";
      let error="";try { modelingEligibilityPayload(); } catch(caught) { error=caught.message; }
      console.log(JSON.stringify({before,windowSummary:state.trainingWindowSummary,windowText:el("trainingWindows").text(),qualityDisabled:el("qualityButton").disabled,payload,keep:keep.children.length,groups:el("eligibilityExcludeGroups").children.length,invalidations,summary:el("eligibilitySummary").textContent,error,candidates:state.candidateWindows,windows:state.trainingWindows,training:state.training,runId:state.runId,tags:[...state.selectedModelTags],modelHidden:el("modelContent").hidden}));
    '''.replace("__SCRIPT__", script).replace("__RENDERER__", renderer).replace("__HELPERS__", helpers)
    result = subprocess.run([node, "-"], input=harness, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["payload"] == {"keep_conditions": [condition(minimum=1, maximum=10), condition(minimum=2, maximum=10)], "exclude_rule_groups": [[condition(minimum=5)], [condition(minimum=5)]]}
    assert data["keep"] == data["groups"] == 1
    assert data["invalidations"] >= 7
    assert data["candidates"] == [] and data["training"] is None and data["runId"] is None
    assert data["windows"] == [{"id": "manual", "source": "manual", "enabled": False, "source_ref": "manual"}]
    assert "120 / 118" in data["before"]["text"] and "passed" in data["before"]["text"]
    assert data["before"]["disabled"] is False
    assert data["windowSummary"] == [] and data["qualityDisabled"] is True
    assert "120 / 118" not in data["windowText"] and "passed" not in data["windowText"]
    assert "待检查" in data["windowText"]
    assert data["modelHidden"] and data["tags"] == ["A"]
    assert "上下限反转" in data["error"]
    for text in ["原始样本 10", "保留条件通过 8", "排除规则命中 3", "最终合格 5", "50.0%", "连续段 2"]:
        assert text in data["summary"]


@pytest.mark.parametrize("action", ["confirm_candidate", "update", "remove", "set_enabled"])
@pytest.mark.parametrize("stale", [False, True])
def test_training_window_operations_discard_old_eligibility_responses(action, stale):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for training window regression tests")
    html = web_model_results.INDEX_HTML
    script = html[html.index("async function updateTrainingWindows("):html.index("async function confirmCandidateWindow(")]
    availability = next(line for line in html.splitlines() if line.startswith("function updateQualityButtonAvailability("))
    window = {"id": "manual", "source": "manual", "source_ref": "original", "enabled": True, "comment": "original"}
    expected_windows = [] if action == "remove" else [
        {**window, "comment": "edited"} if action == "update" else
        {**window, "enabled": False} if action == "set_enabled" else
        {**window, "id": "confirmed", "source_ref": "candidate"}
    ]
    expected_summary = [{"id": item["id"], "raw_samples": 100, "effective_samples": 99} for item in expected_windows]
    harness = r'''
      let eligibilityRevision=0, respond, request, renders=0, invalidations=0;
      const messages=[],button={disabled:false};
      const state={inspection:{},trainingWindows:[__WINDOW__],trainingWindowSummary:[{id:"manual",raw_samples:120,effective_samples:118}]};
      function el() { return button; }
      function commonPayload() { return {}; }
      function trainingWindowsPayload() { return state.trainingWindows; }
      function api(url,options) { request=JSON.parse(options.body); return new Promise(resolve=>{respond=resolve;}); }
      function renderTrainingWindows() { renders+=1; }
      function renderCandidateWindows() { renders+=1; }
      function invalidateQuality() { invalidations+=1; }
      function setStatus(message,type) { messages.push({message,type}); }
      __AVAILABILITY__
      __SCRIPT__
      (async()=>{
        const pending=updateTrainingWindows({action:__ACTION__,id:"manual"},true);
        if(__STALE__) { eligibilityRevision+=1;state.trainingWindows=state.trainingWindows.map(window=>({...window,enabled:false}));state.trainingWindowSummary=[];updateQualityButtonAvailability(); }
        respond({training_windows:__WINDOWS__,summary:__SUMMARY__});
        const returned=await pending;
        console.log(JSON.stringify({returned,windows:state.trainingWindows,summary:state.trainingWindowSummary,disabled:button.disabled,renders,invalidations,messages,request}));
      })();
    '''.replace("__WINDOW__", json.dumps(window)).replace("__WINDOWS__", json.dumps(expected_windows)).replace("__SUMMARY__", json.dumps(expected_summary)).replace("__STALE__", json.dumps(stale)).replace("__ACTION__", json.dumps(action)).replace("__AVAILABILITY__", availability).replace("__SCRIPT__", script)
    result = subprocess.run([node, "-"], input=harness, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["request"]["operation"]["action"] == action
    if stale:
        assert data["returned"] is False
        assert data["windows"] == [{**window, "enabled": False}] and data["summary"] == []
        assert data["disabled"] is True and data["renders"] == data["invalidations"] == 0
        assert data["messages"][-1]["type"] == "warning"
    else:
        assert data["returned"] is True
        assert data["windows"] == expected_windows and data["summary"] == expected_summary
        assert data["disabled"] == (not any(item["enabled"] for item in expected_windows))
        assert data["renders"] == 2 and data["invalidations"] == 1


def test_ui_manual_candidates_split_before_confirmation_and_discard_stale_requests():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for candidate eligibility regression tests")
    html = web_model_results.INDEX_HTML
    start = html.index("async function addCandidateWindow(")
    script = html[start:html.index("\nfunction ", start)]
    harness = r'''
      let eligibilityRevision=0, sequence=0, stale=false, request=null;
      const state={candidateWindows:[]};
      function modelingEligibilityPayload() { return {keep_conditions:[{column:"gate",minimum:1}],exclude_rule_groups:[]}; }
      function commonPayload() { return {modeling_eligibility:modelingEligibilityPayload()}; }
      function candidateId() { return `candidate-${++sequence}`; }
      function renderCandidateWindows() {}
      function setStatus() {}
      function el() { return {scrollIntoView(){}}; }
      async function api(url,options) { request=JSON.parse(options.body); if(stale) eligibilityRevision+=1; return {eligible_windows:[{start:"2026-01-01T00:00:00",end:"2026-01-01T00:10:00"},{start:"2026-01-01T00:20:00",end:"2026-01-01T00:30:00"}]}; }
      __SCRIPT__
      (async()=>{
        await addCandidateWindow("trend","2026-01-01T00:00:00","2026-01-01T00:30:00","trend-current","test");
        const candidates=state.candidateWindows.slice();state.candidateWindows=[];stale=true;
        await addCandidateWindow("manual","2026-01-01T00:00:00","2026-01-01T00:30:00",null,"");
        console.log(JSON.stringify({candidates,stale:state.candidateWindows,request}));
      })();
    '''.replace("__SCRIPT__", script)
    result = subprocess.run([node, "-"], input=harness, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert len(data["candidates"]) == 2
    assert data["candidates"][0]["end"] == "2026-01-01T00:10:00"
    assert data["candidates"][1]["start"] == "2026-01-01T00:20:00"
    assert all(candidate["source_ref"] == "trend-current" for candidate in data["candidates"])
    assert data["stale"] == []
    assert data["request"]["modeling_eligibility"]["keep_conditions"][0]["column"] == "gate"
