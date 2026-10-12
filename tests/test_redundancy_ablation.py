import copy
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.request

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web
from pca_model_builder import redundancy_ablation as ablation
from pca_model_builder.model_io import load_model_package
from pca_model_builder.validation import validate_model_windows


@pytest.fixture
def trained(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(61)
    x = rng.normal(size=700)
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=700, freq="5min"),
                          "A": x, "B": x + rng.normal(0, .03, 700), "C": -x + rng.normal(0, .03, 700),
                          "D": rng.normal(size=700), "E": rng.normal(size=700), "F": rng.normal(size=700)})
    for start, end in ((430, 469), (490, 529), (550, 589)):
        frame.loc[start:end, "D"] += 12
    uploaded = web.save_upload("history.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    payload = {"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": list("ABCDEF"),
               "model_name": "ablation-baseline", "model_purpose": "normal_state", "n_components": 2,
               "normal_start": frame.time.iloc[0].isoformat(), "normal_end": frame.time.iloc[299].isoformat(),
               "sample_interval_minutes": 5, "max_lag_minutes": 5, "lag_step_minutes": 5, "filter_method": "none"}
    result = web.train_payload(payload)
    request = {"run_id": result["run_id"], "file_id": uploaded["file_id"], "action": "evaluate",
               "group_id": "group-1", "drop_tags": ["A", "C"]}
    request["validation_windows"] = [
        {"id": name, "type": kind, "start": frame.time.iloc[start].isoformat(),
         "end": frame.time.iloc[end].isoformat(), "enabled": True, "comment": "人工已知窗口"}
        for name, kind, start, end in [("normal", "normal_validation", 310, 419),
                                       ("bad1", "known_abnormal", 430, 469),
                                       ("bad2", "known_abnormal", 490, 529),
                                       ("bad3", "known_abnormal", 550, 589)]]
    return frame, payload, request, tmp_path / "runs" / result["run_id"]


def test_groups_keep_signed_pairs_and_indirect_edges():
    # A-B and B-C are high correlations; A-C is below threshold.
    rng = np.random.default_rng(4)
    x, noise = rng.normal(size=(2, 2000))
    result = ablation.redundancy_candidates(pd.DataFrame({"A": x, "B": x + .24 * noise,
                                                          "C": -(x + .48 * noise), "constant": 1}))
    assert result["groups"][0]["tags"] == ["A", "B", "C"]
    pairs = result["groups"][0]["pairs"]
    assert len(pairs) == 3
    assert sum(pair["high_correlation"] for pair in pairs) == 2
    assert any(pair["pearson_r"] < -.95 for pair in pairs)
    assert result["diagnostics"]["sample_scope"] == "baseline_effective_training_timestamps"
    assert result["diagnostics"]["unavailable_pairs"][0]["unavailable_reason"]
    sparse = ablation.redundancy_candidates(pd.DataFrame({"A": [1, 2], "B": [-1, -2]}))
    assert sparse["groups"] == []
    assert "不足" in sparse["diagnostics"]["unavailable_pairs"][0]["unavailable_reason"]


def test_real_training_scoring_alignment_and_no_artifact_writes(trained, monkeypatch):
    frame, payload, request, run = trained
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    calls = []
    fit = ablation.fit_dpca

    def record_fit(dynamic, **kwargs):
        model = fit(dynamic, **kwargs)
        calls.append((dynamic.copy(), model))
        return model

    monkeypatch.setattr(ablation, "fit_dpca", record_fit)
    result = web.redundancy_ablation_payload(request)
    schemes = result["schemes"]
    assert len(schemes) == 3 and all(s["status"] == "evaluated" for s in schemes)
    assert len(calls) == 2
    for scheme, (dynamic, model) in zip(schemes[1:], calls):
        assert not any(name.startswith(scheme["dropped_tag"] + "__lag_") for name in dynamic)
        assert len(model.feature_names) == 10
        assert model.n_samples == scheme["training_rows"]
        np.testing.assert_allclose(model.mean, dynamic.mean().to_numpy())
        np.testing.assert_allclose(model.scale, dynamic.std(ddof=0).to_numpy())
        assert scheme["training_timestamps_match"]
        assert scheme["validation"]["normal_validation"]["common_rows"] == 110
        assert scheme["validation"]["known_abnormal"]["common_rows"] == 120
        assert scheme["validation"]["known_abnormal"]["metrics"]["valid_window_count"] == 3
        assert scheme["validation"]["normal_validation"]["coverage"] == 1
    assert calls[0][1].t2_limits is not calls[1][1].t2_limits
    assert calls[0][1].q_limits != calls[1][1].q_limits
    baseline, manifest = load_model_package(run / "model.pcamodel")
    direct = validate_model_windows(baseline, frame.set_index("time"), payload["tags"], web._preprocessing_config(manifest["config"]),
                                   [(pd.Timestamp(payload["normal_start"]), pd.Timestamp(payload["normal_end"]))], request["validation_windows"])
    for kind in ("normal_validation", "known_abnormal"):
        assert schemes[0]["validation"][kind]["metrics"] == direct["validation_metrics"][kind]
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    json.dumps(result, allow_nan=False)


def test_candidates_use_only_baseline_effective_training_rows(trained):
    frame, payload, request, run = trained
    candidates = web.redundancy_ablation_payload({**request, "action": "candidates", "model_sha256": None})
    assert candidates["diagnostics"]["sample_count"] == 299
    assert candidates["groups"][0]["tags"] == ["A", "B", "C"]
    assert candidates["diagnostics"]["high_correlation_pairs"][0]["valid_pair_count"] == 299


def test_real_model_results_http_route(trained):
    from pca_model_builder.web_model_results import ModelResultsHandler

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/redundancy-ablation",
            data=json.dumps({**trained[2], "action": "candidates"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert result["run_id"] == trained[2]["run_id"]
        assert result["groups"][0]["tags"] == ["A", "B", "C"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["none", "normal_only", "small"])
def test_missing_or_insufficient_abnormal_evidence_never_recommends_deletion(trained, mode):
    _, _, request, _ = trained
    if mode == "none":
        request["validation_windows"] = []
    elif mode == "normal_only":
        request["validation_windows"] = request["validation_windows"][:1]
    else:
        request["validation_windows"] = request["validation_windows"][:2]
    result = web.redundancy_ablation_payload(request)
    assert all(s["conclusion"] == "证据不足" for s in result["schemes"])
    assert any("异常" in reason for reason in result["schemes"][1]["reasons"])


def test_training_sample_changes_and_validation_intersection_are_reported(trained):
    frame, payload, request, run = trained
    # Re-train with a missing A row in training and another inside validation.
    frame.loc[[40, 340], "A"] = np.nan
    uploaded = web.save_upload("gaps.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    new = web.train_payload({**payload, "file_id": uploaded["file_id"]})
    result = web.redundancy_ablation_payload({**request, "file_id": uploaded["file_id"], "run_id": new["run_id"]})
    base, drop_a, drop_c = result["schemes"]
    assert not drop_a["training_timestamps_match"]
    assert drop_a["training_rows"] > base["training_rows"]
    assert drop_c["training_timestamps_match"]
    normal = [s["validation"]["normal_validation"] for s in result["schemes"]]
    assert normal[1]["original_rows"] == 110
    assert normal[0]["original_rows"] == normal[2]["original_rows"] < 110
    assert len({item["common_rows"] for item in normal}) == 1
    assert normal[1]["common_coverage"] < 1
    assert drop_a["conclusion"] == "证据不足"


def test_overlap_blocks_validation_conclusions(trained):
    _, payload, request, _ = trained
    request["validation_windows"][0]["start"] = payload["normal_start"]
    with pytest.raises(ValueError, match="overlap"):
        web.redundancy_ablation_payload(request)


def test_invalid_candidate_and_window_failure_are_explicit(trained, monkeypatch):
    _, _, request, _ = trained
    original = ablation.fit_dpca

    def fail_one(dynamic, **kwargs):
        if not any(name.startswith("A__lag_") for name in dynamic):
            raise ValueError("fixed components leave no effective residual space")
        return original(dynamic, **kwargs)

    monkeypatch.setattr(ablation, "fit_dpca", fail_one)
    result = web.redundancy_ablation_payload(request)
    assert result["schemes"][1]["status"] == "failed"
    assert "residual" in result["schemes"][1]["error"]
    assert result["schemes"][1]["validation"] == {}
    assert result["schemes"][2]["status"] == "evaluated"
    request["validation_windows"][0]["end"] = "2030-01-01T00:00:00"
    request["validation_windows"] = request["validation_windows"][:1]
    result = web.redundancy_ablation_payload(request)
    assert result["schemes"][0]["validation_errors"][0]["window_id"] == "normal"
    assert result["schemes"][2]["validation"]["normal_validation"]["metrics"]["overall"]["exceedance_rate_95"] is None


@pytest.mark.parametrize("override,match", [({"locked_tags": ["A"]}, "未锁定"),
    ({"drop_tags": []}, "请选择"), ({"drop_tags": ["D"]}, "所选组"),
    ({"drop_tags": ["A", "A"]}, "不重复"), ({"drop_tags": [1]}, "不重复"),
    ({"drop_tags": list("ABCDEFGHI")}, "1至8"),
    ({"model_sha256": "stale"}, "已变化"), ({"file_id": "other"}, "数据源"),
    ({"validation_windows": None}, "必须是列表"),
    ({"action": "other"}, "action"), ({"group_id": "other"}, "候选组")])
def test_request_boundaries(trained, override, match):
    with pytest.raises(ValueError, match=match):
        web.redundancy_ablation_payload({**trained[2], **override})


def test_changed_source_and_model_lifecycle_rejected(trained):
    _, _, request, run = trained
    source = next(web.UPLOADS_DIR.glob(request["file_id"] + ".csv"))
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="训练数据已变化"):
        web.redundancy_ablation_payload(request)


def test_real_fixed_component_count_is_not_silently_reduced(trained):
    _, payload, request, _ = trained
    result = web.train_payload({**payload, "n_components": 10})
    evaluation = web.redundancy_ablation_payload({**request, "run_id": result["run_id"]})
    assert evaluation["schemes"][0]["n_components"] == 10
    assert all(s["status"] == "failed" and "residual space" in s["error"] for s in evaluation["schemes"][1:])


def test_snapshot_preserves_source_encoding_for_evaluation(trained):
    frame, payload, request, _ = trained
    frame = frame.rename(columns={"A": "流量"})
    uploaded = web.save_upload("中文.csv", frame.to_csv(index=False).encode("gb18030"))
    result = web.train_payload({**payload, "file_id": uploaded["file_id"], "tags": ["流量", *list("BCDEF")], "encoding": "gb18030"})
    candidates = web.redundancy_ablation_payload({**request, "run_id": result["run_id"], "file_id": uploaded["file_id"],
                                                 "action": "candidates", "encoding": "utf-8-sig"})
    assert candidates["groups"][0]["tags"] == ["流量", "B", "C"]


def test_eligibility_is_preserved_even_when_its_tag_is_removed(trained):
    _, payload, request, _ = trained
    eligibility = {"keep_conditions": [{"column": "A", "operator": ">", "value": -.5}], "exclude_rule_groups": []}
    result = web.train_payload({**payload, "modeling_eligibility": eligibility})
    evaluation = web.redundancy_ablation_payload({**request, "run_id": result["run_id"], "drop_tags": ["A"],
                                                 "modeling_eligibility": {"keep_conditions": [], "exclude_rule_groups": []}})
    assert evaluation["schemes"][1]["status"] == "evaluated"
    assert evaluation["schemes"][1]["training_timestamps_match"]
    assert evaluation["schemes"][1]["training_rows"] == evaluation["schemes"][0]["training_rows"] < 299


def test_engineering_labels_are_counted_on_common_validation_samples(trained):
    frame, payload, request, _ = trained
    frame["label"] = np.where(frame.index >= 430, "已知异常", "正常")
    frame.loc[340, "A"] = np.nan
    uploaded = web.save_upload("labels.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    result = web.train_payload({**payload, "file_id": uploaded["file_id"]})
    evaluation = web.redundancy_ablation_payload({**request, "run_id": result["run_id"], "file_id": uploaded["file_id"], "label_column": "label"})
    for scheme in evaluation["schemes"]:
        for kind, label in (("normal_validation", "正常"), ("known_abnormal", "已知异常")):
            validation = scheme["validation"][kind]
            assert sum(validation["status_by_engineering_label"][label].values()) == validation["common_rows"]
            assert validation["common_rows"] == (108 if kind == "normal_validation" else 120)


def test_real_ablation_detects_loss_of_tag_specific_anomalies(trained):
    frame, payload, request, _ = trained
    for start, end in ((430, 469), (490, 529), (550, 589)):
        frame.loc[start:end, "D"] -= 12
        frame.loc[start:end, "A"] += 10
    uploaded = web.save_upload("tag-specific.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    result = web.train_payload({**payload, "file_id": uploaded["file_id"]})
    evaluation = web.redundancy_ablation_payload({**request, "run_id": result["run_id"], "file_id": uploaded["file_id"], "drop_tags": ["A"]})
    scheme = evaluation["schemes"][1]
    assert scheme["detection_delta"]["detection_coverage_rate_95"] < -.5
    assert scheme["conclusion"] == "建议保留"


def test_exploratory_model_is_not_an_ablation_baseline(trained):
    _, payload, request, _ = trained
    result = web.train_payload({**payload, "model_purpose": "exploratory"})
    with pytest.raises(ValueError, match="normal_state/candidate"):
        web.redundancy_ablation_payload({**request, "run_id": result["run_id"]})


def test_legacy_candidate_is_not_reinterpreted_with_new_preprocessing(trained, monkeypatch):
    load = web.load_model_package

    def legacy_candidate(path):
        model, manifest = load(path)
        return model, {**manifest, "schema_version": 4}

    monkeypatch.setattr(web, "load_model_package", legacy_candidate)
    with pytest.raises(ValueError, match="旧预处理语义.*重新正式训练"):
        web.redundancy_ablation_payload(trained[2])


def test_conclusions_detect_loss_and_non_degradation(trained):
    result = web.redundancy_ablation_payload(trained[2])
    base = result["schemes"][0]
    same = copy.deepcopy(base)
    same["dropped_tag"] = "A"
    assert ablation.ablation_conclusion(base, same)[0] == "可考虑剔除"
    same["validation"]["known_abnormal"]["metrics"]["windows"][0]["first_detection_delay_minutes_99"] = 5
    assert ablation.ablation_conclusion(base, same)[0] == "建议保留"
    same = copy.deepcopy(base)
    same["dropped_tag"] = "A"
    same["validation"]["normal_validation"]["metrics"]["overall"]["exceedance_rate_95"] += .01
    assert ablation.ablation_conclusion(base, same)[0] == "建议保留"


@pytest.mark.parametrize("statistic", ["t2", "spe"])
@pytest.mark.parametrize("confidence", [95, 99])
def test_independent_channel_window_loss_blocks_deletion_at_unchanged_overall_detection(trained, statistic, confidence):
    # 综合检出率、检出覆盖率与窗口延迟完全不变，仅单一统计量少检出一个异常窗口。
    result = web.redundancy_ablation_payload(trained[2])
    base = result["schemes"][0]
    same = copy.deepcopy(base)
    same["dropped_tag"] = "A"
    metrics = same["validation"]["known_abnormal"]["metrics"]
    key = f"{statistic}_detected_window_count_{confidence}"
    metrics[key] -= 1
    conclusion, reasons, delta = ablation.ablation_conclusion(base, same)
    baseline_metrics = base["validation"]["known_abnormal"]["metrics"]
    assert metrics["detection_rate_95"] == baseline_metrics["detection_rate_95"]
    assert metrics["detection_coverage_rate_95"] == baseline_metrics["detection_coverage_rate_95"]
    assert conclusion == "建议保留"
    assert any(f"{statistic.upper()} {confidence}%" in reason and "独立检出窗口数下降" in reason for reason in reasons)
    assert delta[key] == -1


def test_real_evaluation_reports_independent_window_counts(trained):
    result = web.redundancy_ablation_payload(trained[2])
    baseline = result["schemes"][0]["validation"]["known_abnormal"]["metrics"]
    for scheme in result["schemes"][1:]:
        for statistic in ("t2", "spe"):
            for confidence in (95, 99):
                key = f"{statistic}_detected_window_count_{confidence}"
                assert scheme["detection_delta"][key] == 0
                assert scheme["validation"]["known_abnormal"]["metrics"][key] == baseline[key]
