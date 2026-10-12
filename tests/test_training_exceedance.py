from dataclasses import replace
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import training_exceedance as diagnosis, web
from pca_model_builder.dpca import DPCAModel
from pca_model_builder.scoring_core import aggregate_tag_contributions, spe_feature_contributions, t2_feature_contributions


def _model():
    return DPCAModel(feature_names=("A__lag_0min", "A__lag_5min", "B__lag_0min"),
                     mean=np.zeros(3), scale=np.ones(3), components=np.array([[1., 0., 0.]]),
                     eigenvalues=np.ones(1), explained_variance_ratio=np.ones(1),
                     t2_limits={0.95: 4., 0.99: 9.}, q_limits={0.95: 4., 0.99: 9.}, n_samples=5)


def test_inclusive_valid_samples_independent_exceedance_and_mean_lag_contributions():
    model = _model()
    times = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10",
                            "2026-01-01 01:00", "2026-01-01 01:05"])
    dynamic = pd.DataFrame([[2, 0, 0], [0, 3, 0], [3, 1, 0], [0, 1, 4], [np.nan, 0, 0]],
                           index=times, columns=model.feature_names)
    scores = model.score(dynamic)
    result = diagnosis.interval_diagnosis(model, dynamic, scores, times[0], times[-1])
    assert result["valid_samples"] == 4
    assert result["statistics"]["t2"] == {"mean": 3.25, "max": 9., "max_limit_ratio": 2.25,
        "exceedance_95": {"count": 2, "rate": .5}, "exceedance_99": {"count": 1, "rate": .25}}
    assert result["statistics"]["spe"]["exceedance_95"] == {"count": 2, "rate": .5}
    assert result["contributions"]["t2"]["rows"][0]["mean_contribution"] == 6.5
    for statistic, calculate, positions in [("t2", t2_feature_contributions, [0, 2]),
                                           ("spe", spe_feature_contributions, [1, 3])]:
        mean = np.mean([calculate(dynamic.iloc[i].to_numpy(), feature_names=model.feature_names,
                    mean=model.mean, scale=model.scale, components=model.components,
                    eigenvalues=model.eigenvalues) for i in positions], axis=0)
        expected = aggregate_tag_contributions(model.feature_names, mean)
        actual = result["contributions"][statistic]
        assert actual["sample_count"] == 2
        for row, item in zip(actual["rows"], expected, strict=True):
            assert row["tag"] == item.tag
            assert row["contribution_pct"] == pytest.approx(item.contribution_pct)
            assert (row["lag_start_minutes"], row["lag_end_minutes"]) == (item.lag_start_minutes, item.lag_end_minutes)
    # A physical gap remains empty; no synthetic scoring or denominator rows.
    empty = diagnosis.interval_diagnosis(model, dynamic, scores, times[2] + pd.Timedelta(minutes=1), times[3] - pd.Timedelta(minutes=1))
    assert empty["valid_samples"] == 0
    assert empty["statistics"]["t2"]["mean"] is None
    assert empty["contributions"]["spe"] == {"sample_count": 0, "rows": []}
    single = diagnosis.interval_diagnosis(model, dynamic, scores, times[0], times[0])
    assert single["valid_samples"] == 1
    assert single["contributions"]["spe"]["rows"] == []
    with pytest.raises(ValueError):
        diagnosis.interval_diagnosis(model, dynamic, scores, times[1], times[0])


def test_interval_contributions_respect_model_feature_order():
    model = _model()
    times = pd.date_range("2026-01-01", periods=2, freq="5min")
    canonical = pd.DataFrame([[5., 0., 0.], [0., 0., 5.]],
                             index=times, columns=model.feature_names)
    scores = model.score(canonical)
    reordered = canonical.loc[:, list(reversed(model.feature_names))]

    expected = diagnosis.interval_diagnosis(model, canonical, scores, times[0], times[-1])
    actual = diagnosis.interval_diagnosis(model, reordered, scores, times[0], times[-1])

    assert actual["contributions"] == expected["contributions"]


def test_zero_spe_limit_does_not_return_infinity_or_fabricated_contribution():
    model = replace(_model(), q_limits={.95: 0., .99: 0.})
    dynamic = pd.DataFrame([[2., 0., 0.]], index=pd.date_range("2026-01-01", periods=1), columns=model.feature_names)
    # The scoring core marks non-finite limit ratios unscorable.
    result = diagnosis.interval_diagnosis(model, dynamic, model.score(dynamic), dynamic.index[0], dynamic.index[0])
    assert result["valid_samples"] == 0
    assert result["statistics"]["spe"]["max_limit_ratio"] is None


@pytest.mark.parametrize("anomaly_load_values,expected_range_overlap", [([100.], False), ([], None)])
def test_dual_intervals_filter_groups_align_load_and_average_spe_residuals(anomaly_load_values, expected_range_overlap):
    model = _model()
    times = pd.date_range("2026-01-01", periods=7, freq="5min")
    dynamic = pd.DataFrame(
        [[0., 3., 4.], [0., 1., 2.], [np.nan, 0., 0.],
         [0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [np.nan, 0., 0.]],
        index=times, columns=model.feature_names,
    )
    scores = model.score(dynamic)
    # Missing timestamps must stay missing; there is no nearest-value matching.
    load_index = times[[0, 3, 4, 5]] if anomaly_load_values else times[[3, 4, 5]]
    load = pd.Series([*anomaly_load_values, 10., 12., 11.], index=load_index)
    result = diagnosis.compare_diagnosis(
        model, dynamic, scores, load, "Load", times[0], times[2], times[3], times[6]
    )

    assert result["anomaly"]["interval_valid_samples"] == 2
    assert result["anomaly"]["sample_count"] == 2
    assert result["anomaly"]["spe_exceedance_count"] == 2
    assert result["anomaly"]["spe_exceedance_rate"] == 1
    assert result["anomaly"]["load"]["valid_samples"] == len(anomaly_load_values)
    assert result["anomaly"]["load"]["missing_rate"] == 1 - len(anomaly_load_values) / 2
    assert result["reference"]["interval_valid_samples"] == 3
    assert result["reference"]["sample_count"] == 3
    assert result["reference"]["spe_exceedance_count"] == 0
    assert result["reference"]["spe_exceedance_rate"] == 0
    assert result["reference"]["load"]["mean"] == pytest.approx(11)
    assert result["reference"]["load"]["median"] == pytest.approx(11)
    assert result["reference"]["load"]["p10"] == pytest.approx(10.2)
    assert result["reference"]["load"]["p90"] == pytest.approx(11.8)
    assert result["load_ranges_overlap"] is expected_range_overlap
    assert result["load_statistics_available"] is False
    assert result["load_p10_p90_overlap"] is None
    assert result["load_comparability_status"] == "insufficient_evidence"
    assert "负荷数据不足，不能认定同负荷；SPE 数值对照仅为描述性、未匹配负荷比较。" in result["warnings"]

    rows = {row["tag"]: row for row in result["contributions"]}
    # residual² means: anomaly=[0, 5, 10], reference=[0, 2/3, 1/3].
    assert rows["A"]["anomaly_mean"] == pytest.approx(5)
    assert rows["B"]["anomaly_mean"] == pytest.approx(10)
    assert rows["A"]["reference_mean"] == pytest.approx(2 / 3)
    assert rows["B"]["reference_mean"] == pytest.approx(1 / 3)
    assert rows["B"]["absolute_difference"] == pytest.approx(10 - 1 / 3)
    assert rows["A"]["anomaly_share_pct"] == pytest.approx(100 / 3)
    assert rows["B"]["reference_share_pct"] == pytest.approx(100 / 3)
    assert rows["B"]["anomaly_lag_start_minutes"] == 0
    assert rows["A"]["reference_lag_start_minutes"] == 5


def test_overlapping_extrema_with_disjoint_central_load_ranges_warns_to_reselect():
    model = _model()
    times = pd.date_range("2026-01-01", periods=20, freq="5min")
    dynamic = pd.DataFrame([[0., 5., 0.]] * 10 + [[0., 0., 0.]] * 10,
                           index=times, columns=model.feature_names)
    load = pd.Series([100.] + [0.] * 9 + [0.] + [50.] * 9, index=times)
    result = diagnosis.compare_diagnosis(
        model, dynamic, model.score(dynamic), load, "Load", times[0], times[9], times[10], times[19]
    )

    assert result["analysis_available"]
    assert result["load_ranges_overlap"] is True
    assert result["load_p10_p90_overlap"] is False
    assert result["load_comparability_status"] == "body_disjoint"
    assert "主体负荷分布不重叠/不匹配，应重新选择参考区间。" in result["warnings"]
    assert result["contributions"]


def test_dual_interval_zero_reference_energy_and_insufficient_samples():
    model = _model()
    times = pd.date_range("2026-01-01", periods=4, freq="5min")
    dynamic = pd.DataFrame([[0., 3., 4.], [0., 1., 2.], [1., 0., 0.], [2., 0., 0.]],
                           index=times, columns=model.feature_names)
    # The closed P10–P90 intervals touch exactly at 9 and therefore overlap.
    load = pd.Series([0., 10., 8., 18.], index=times)
    result = diagnosis.compare_diagnosis(
        model, dynamic, model.score(dynamic), load, "Load", times[0], times[1], times[2], times[3]
    )
    assert result["analysis_available"]
    assert result["load_p10_p90_overlap"] is True
    assert result["load_comparability_status"] == "manual_review"
    assert any("不自动判定为同负荷" in warning for warning in result["warnings"])
    assert all(row["reference_mean"] == 0 for row in result["contributions"])
    assert all(row["reference_share_pct"] is None for row in result["contributions"])
    assert all(row["reference_lag_start_minutes"] is None for row in result["contributions"])
    with pytest.raises(ValueError, match="不能重叠"):
        diagnosis.compare_diagnosis(
            model, dynamic, model.score(dynamic), load, "Load", times[0], times[1], times[1], times[3]
        )
    insufficient = diagnosis.compare_diagnosis(
        model, dynamic, model.score(dynamic), load, "Load", times[0], times[0], times[2], times[3]
    )
    assert not insufficient["analysis_available"]
    assert insufficient["contributions"] == []
    assert insufficient["load_comparability_status"] == "insufficient_evidence"
    assert insufficient["load_p10_p90_overlap"] is None
    assert any("SPE 样本不足，未生成贡献对比" in warning for warning in insufficient["warnings"])


@pytest.fixture
def trained_run(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "MAX_CHART_POINTS", 7)
    rng = np.random.default_rng(71)
    times = pd.date_range("2026-01-01", periods=200, freq="5min").delete(slice(90, 105))
    frame = pd.DataFrame({"time": times, **{tag: rng.normal(size=len(times)) for tag in "ABC"},
                          "Load": np.full(len(times), 7.0), "Note": ["steady"] * len(times)})
    frame.loc[125, "B"] = np.nan
    frame.loc[125, "Load"] = np.nan
    upload = web.save_upload("training.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    request = {"file_id": upload["file_id"], "timestamp_column": "time", "tags": list("ABC"),
               "normal_start": times[0].isoformat(), "normal_end": times[-1].isoformat(),
               "sample_interval_minutes": 5, "smoothing_window_minutes": 0, "filter_method": "none",
               "max_lag_minutes": 5, "lag_step_minutes": 5, "n_components": 2, "model_name": "interval-test",
               "tag_configs": {"B": {"unit": "bar"}}}
    trained = web.train_payload(request)
    return trained, frame, request


def test_run_overview_full_scores_interval_and_raw_trend(trained_run, monkeypatch):
    trained, frame, request = trained_run
    run = {"run_id": trained["run_id"]}
    run_dir = web.RUNS_DIR / trained["run_id"]
    artifacts_before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    def no_contribution(*args, **kwargs):
        raise AssertionError("overview must not compute contributions")
    def no_training_or_sampling(*args, **kwargs):
        raise AssertionError("diagnosis must neither fit a model nor use plot sampling")
    monkeypatch.setattr(web, "fit_dpca", no_training_or_sampling)
    monkeypatch.setattr(web, "_score_payload", no_training_or_sampling)
    with monkeypatch.context() as context:
        context.setattr(diagnosis, "t2_feature_contributions", no_contribution)
        overview = diagnosis.training_exceedance_payload({**run, "action": "overview"})
    assert len(overview["scores"]) == trained["training_rows"] > len(trained["scores"])
    assert overview["t2_limits"] == trained["t2_limits"]
    assert overview["q_limits"] == trained["q_limits"]
    assert "Load" in overview["load_tags"] and "Note" not in overview["load_tags"]
    assert sum(row["break_before"] for row in overview["scores"]) >= 2
    interval = diagnosis.training_exceedance_payload({**run, "action": "interval", "start": overview["start"], "end": overview["end"]})
    assert interval["valid_samples"] == trained["training_rows"]
    for statistic, limits in [("t2", overview["t2_limits"]), ("spe", overview["q_limits"])]:
        values = np.array([row[statistic] for row in overview["scores"]])
        summary = interval["statistics"][statistic]
        assert summary["mean"] == pytest.approx(values.mean())
        assert summary["max"] == pytest.approx(values.max())
        assert summary["exceedance_95"]["count"] == int((values >= limits["95"]).sum())
    start, end = frame.time.iloc[120].isoformat(), frame.time.iloc[130].isoformat()
    trend = diagnosis.training_exceedance_payload({**run, "action": "trend", "start": start, "end": end, "tag": "B"})
    assert trend["unit"] == "bar"
    assert trend["analysis_start"] == start and trend["analysis_end"] == end
    original = frame.set_index("time")["B"]
    for row in trend["rows"]:
        value = original.loc[pd.Timestamp(row["timestamp"])]
        assert row["value"] is None if pd.isna(value) else row["value"] == pytest.approx(value)
    assert any(row["value"] is None for row in trend["rows"])
    assert {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()} == artifacts_before
    with pytest.raises(ValueError, match="原始 Tag"):
        diagnosis.training_exceedance_payload({**run, "action": "trend", "start": start, "end": end, "tag": "unknown"})
    with pytest.raises(ValueError, match="时间无效"):
        diagnosis.training_exceedance_payload({**run, "action": "interval", "start": "NaT", "end": end})


def test_run_comparison_loads_non_model_tag_and_rejects_invalid_ranges(trained_run):
    trained, _, _ = trained_run
    run = {"run_id": trained["run_id"]}
    overview = diagnosis.training_exceedance_payload(run)
    high = [i for i, row in enumerate(overview["scores"])
            if row["spe"] is not None and row["spe"] >= overview["q_limits"]["95"]]
    low = [i for i, row in enumerate(overview["scores"])
           if row["spe"] is not None and row["spe"] < overview["q_limits"]["95"]]
    assert len(high) >= 2 and len(low) >= 2
    pairs = next(((high_pair, low_pair) for high_pair in ((high[0], high[1]), (high[-2], high[-1]))
                  for low_pair in ((low[0], low[1]), (low[-2], low[-1]))
                  if high_pair[1] < low_pair[0] or low_pair[1] < high_pair[0]), None)
    assert pairs is not None
    anomaly_pair, reference_pair = pairs
    rows = overview["scores"]
    payload = {**run, "action": "compare", "load_tag": "Load",
               "anomaly_start": rows[anomaly_pair[0]]["timestamp"],
               "anomaly_end": rows[anomaly_pair[1]]["timestamp"],
               "reference_start": rows[reference_pair[0]]["timestamp"],
               "reference_end": rows[reference_pair[1]]["timestamp"]}
    result = diagnosis.training_exceedance_payload(payload)
    model, _ = web.load_model_package(web.RUNS_DIR / trained["run_id"] / "model.pcamodel")
    assert "Load" not in model.feature_names
    assert result["load_tag"] == "Load"
    assert result["anomaly"]["sample_count"] >= 2
    assert result["reference"]["sample_count"] >= 2
    assert result["anomaly"]["load"]["valid_samples"] <= result["anomaly"]["sample_count"]
    assert result["reference"]["load"]["valid_samples"] <= result["reference"]["sample_count"]
    with pytest.raises(ValueError, match="数值型负荷 Tag"):
        diagnosis.training_exceedance_payload({**payload, "load_tag": "Note"})
    with pytest.raises(ValueError, match="不能重叠"):
        diagnosis.training_exceedance_payload({**payload,
            "reference_start": payload["anomaly_start"], "reference_end": payload["anomaly_end"]})
    with pytest.raises(ValueError, match="时区"):
        diagnosis.training_exceedance_payload({**payload, "anomaly_start": "2026-01-01T00:00:00+08:00"})


def test_changed_or_missing_source_is_explicit(trained_run):
    trained, frame, request = trained_run
    _, source, _ = web._upload_source(request)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="原始训练数据已变化"):
        diagnosis.training_exceedance_payload({"run_id": trained["run_id"]})
    source.unlink()
    with pytest.raises(web.WebStageError, match="上传文件不存在"):
        diagnosis.training_exceedance_payload({"run_id": trained["run_id"]})


def test_model_snapshot_mismatch_is_explicit(trained_run):
    trained, _, _ = trained_run
    path = web.RUNS_DIR / trained["run_id"] / "modeling_snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["candidate_model"]["sha256"] = "0" * 64
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="模型与建模快照不一致"):
        diagnosis.training_exceedance_payload({"run_id": trained["run_id"]})


def test_invalid_run_and_action():
    with pytest.raises(ValueError):
        diagnosis.training_exceedance_payload({"run_id": "../outside"})
    with pytest.raises(ValueError, match="action"):
        diagnosis.training_exceedance_payload({"run_id": "a" * 32, "action": "other"})


def test_top10_uses_all_tag_totals_for_percentages():
    names = tuple(f"T{i:02d}__lag_0min" for i in range(12))
    model = replace(_model(), feature_names=names, mean=np.zeros(12), scale=np.ones(12),
                    components=np.array([[1., *([0.] * 11)]]), n_samples=1)
    dynamic = pd.DataFrame([np.arange(12, dtype=float)],
                           index=pd.date_range("2026-01-01", periods=1), columns=names)
    result = diagnosis.interval_diagnosis(model, dynamic, model.score(dynamic), dynamic.index[0], dynamic.index[0])
    rows = result["contributions"]["spe"]["rows"]
    assert len(rows) == 10
    assert [row["tag"] for row in rows] == [f"T{i:02d}" for i in range(11, 1, -1)]
    assert sum(row["contribution_pct"] for row in rows) < 100
    assert rows[0]["mean_contribution"] == 121


def test_valid_interval_without_exceedances():
    model = _model()
    dynamic = pd.DataFrame([[1., 0., 1.], [0., 1., 0.]],
                           index=pd.date_range("2026-01-01", periods=2, freq="5min"), columns=model.feature_names)
    result = diagnosis.interval_diagnosis(model, dynamic, model.score(dynamic), dynamic.index[0], dynamic.index[-1])
    assert result["valid_samples"] == 2
    for statistic in ("t2", "spe"):
        assert result["statistics"][statistic]["exceedance_95"] == {"count": 0, "rate": 0.}
        assert result["contributions"][statistic] == {"sample_count": 0, "rows": []}


def test_http_query_routes_and_errors(trained_run):
    from pca_model_builder.web_model_results import ModelResultsHandler
    trained, frame, _ = trained_run
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def query(action, **fields):
        body = {"run_id": trained["run_id"], "action": action, **fields}
        request = Request(f"http://127.0.0.1:{server.server_port}/api/training-exceedance",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    try:
        overview = query("overview")
        assert len(overview["scores"]) == trained["training_rows"]
        interval = query("interval", start=overview["start"], end=overview["end"])
        assert interval["valid_samples"] == trained["training_rows"]
        trend = query("trend", start=frame.time.iloc[120].isoformat(), end=frame.time.iloc[130].isoformat(), tag="B")
        assert trend["tag"] == "B" and trend["unit"] == "bar"
        high = [i for i, row in enumerate(overview["scores"])
                if row["spe"] is not None and row["spe"] >= overview["q_limits"]["95"]]
        low = [i for i, row in enumerate(overview["scores"])
               if row["spe"] is not None and row["spe"] < overview["q_limits"]["95"]]
        assert len(high) >= 2 and len(low) >= 2
        pairs = next(((high_pair, low_pair)
                      for high_pair in ((high[0], high[1]), (high[-2], high[-1]))
                      for low_pair in ((low[0], low[1]), (low[-2], low[-1]))
                      if high_pair[1] < low_pair[0] or low_pair[1] < high_pair[0]), None)
        assert pairs is not None
        anomaly_pair, reference_pair = pairs
        rows = overview["scores"]
        dual = {"anomaly_start": rows[anomaly_pair[0]]["timestamp"],
                "anomaly_end": rows[anomaly_pair[1]]["timestamp"],
                "reference_start": rows[reference_pair[0]]["timestamp"],
                "reference_end": rows[reference_pair[1]]["timestamp"], "load_tag": "Load"}
        comparison = query("compare", **dual)
        assert comparison["load_tag"] == "Load" and comparison["contribution_basis"] == "mean_residual_squared"
        compared_trend = query("compare_trend", **dual, tag="B")
        assert compared_trend["tag"] == "B" and compared_trend["load_tag"] == "Load"
        assert {item["tag"] for item in compared_trend["anomaly"]["series"]} == {"B", "Load"}
        with pytest.raises(HTTPError) as error:
            query("interval", start="invalid", end=overview["end"])
        assert error.value.code == 400
        assert json.load(error.value)["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
