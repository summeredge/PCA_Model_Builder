from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.model_quality import model_quality_summary
from pca_model_builder.preprocessing import PreprocessingConfig


def _summary(spe=None, t2=None, ratios=(0.5, 0.3, 0.2), windows=None, purpose="normal_state"):
    spe = np.ones(80) if spe is None else np.asarray(spe, dtype=float)
    t2 = np.ones(len(spe)) if t2 is None else np.asarray(t2, dtype=float)
    scores = pd.DataFrame({"spe": spe, "t2": t2}, index=pd.date_range("2026-01-01", periods=len(spe), freq="5min"))
    model = SimpleNamespace(n_components=2, explained_variance_ratio=ratios, t2_limits={0.95: 4, 0.99: 6}, q_limits={0.95: 2, 0.99: 3})
    if windows is None:
        windows = [{"start": scores.index[0].isoformat(), "end": scores.index[-1].isoformat(), "source": "manual", "status": "used", "effective_samples": len(scores)}]
    return model_quality_summary(model, scores, windows, 5, purpose)


def test_quality_uses_all_scores_and_control_limit_boundaries():
    values = np.ones(3001)
    values[-2:] = [2, 3]
    result = _summary(values)
    spe = result["statistics"]["spe"]
    assert result["training_samples"] == 3001
    assert spe["mean"] == pytest.approx(values.mean())
    assert spe["exceedance_rates"] == {"95": 2 / 3001, "99": 1 / 3001}
    assert result["pc1_pc2_explained_variance"] == pytest.approx(0.8)
    assert result["training_data"]["effective_sample_hours"] == pytest.approx(3001 * 5 / 60)
    assert "当前主元数量可作为候选" in result["engineering_messages"][0]


@pytest.mark.parametrize("spe,t2,ratios,message", [
    (np.full(80, 4), None, (0.4, 0.25, 0.35), "模型残差频繁超限"),
    (None, np.full(80, 7), (0.7, 0.25, 0.05), "主元空间位置频繁偏离"),
    (None, None, (0.4, 0.25, 0.35), "仅凭解释率不能确定模型不足"),
])
def test_quality_explains_residual_and_score_space_separately(spe, t2, ratios, message):
    result = _summary(spe, t2, ratios)
    assert message in result["engineering_messages"][0]


def test_quality_trend_respects_independent_windows_and_missing_values():
    windows = [
        {"start": "2026-01-01T00:00:00", "end": "2026-01-01T03:15:00", "source": "manual", "status": "used", "effective_samples": 40},
        {"start": "2026-01-01T03:20:00", "end": "2026-01-01T06:35:00", "source": "manual", "status": "used", "effective_samples": 40},
    ]
    result = _summary(np.r_[np.full(40, 0.1), np.full(40, 1.5)], windows=windows)
    assert result["statistics"]["spe"]["trend"] == "stable"
    result = _summary(np.r_[np.full(40, 0.1), np.full(40, 1.5)])
    assert result["statistics"]["spe"]["trend"] == "changing"
    result = _summary(np.full(80, np.nan))
    assert result["statistics"]["spe"]["mean"] is None
    assert result["statistics"]["spe"]["exceedance_rates"]["95"] is None
    assert result["statistics"]["spe"]["trend"] == "insufficient"
    assert "有效评分不足" in result["engineering_messages"][0]


def test_quality_trend_does_not_bridge_time_gaps_or_invalid_scores():
    model = SimpleNamespace(n_components=2, explained_variance_ratio=[0.5, 0.3], t2_limits={0.95: 4, 0.99: 6}, q_limits={0.95: 2, 0.99: 3})
    index = pd.date_range("2026-01-01", periods=40, freq="5min").append(pd.date_range("2026-01-03", periods=40, freq="5min"))
    scores = pd.DataFrame({"spe": np.r_[np.full(40, 0.1), np.full(40, 1.5)], "t2": 1.0, "score_valid": True}, index=index)
    windows = [{"start": index[0].isoformat(), "end": index[-1].isoformat(), "source": "manual", "status": "used", "effective_samples": 80}]
    result = model_quality_summary(model, scores, windows, 5, "normal_state")
    assert result["statistics"]["spe"]["trend"] == "stable"
    assert result["statistics"]["spe"]["trend_segments_checked"] == 2
    scores.iloc[0, scores.columns.get_loc("score_valid")] = False
    result = model_quality_summary(model, scores, windows, 5, "normal_state")
    assert result["statistics"]["spe"]["valid_samples"] == 79


def test_quality_cluster_sources_aggregate_candidates_but_keep_runs_distinct():
    windows = [
        {"source": "cluster", "source_ref": f"state-exploration-{run}-cluster_001-candidate-{rank:03d}", "effective_samples": count, "status": "used", "start": "2026-01-01", "end": "2026-01-02"}
        for run, rank, count in [("a" * 32, 1, 20), ("a" * 32, 2, 20), ("b" * 32, 1, 10)]
    ]
    windows += [
        {"source": "cluster", "source_ref": None, "effective_samples": 10, "status": "used", "start": "2026-01-01", "end": "2026-01-02"},
        {"source": "manual", "effective_samples": 20, "status": "used", "start": "2026-01-01", "end": "2026-01-02"},
        {"source": "cluster", "source_ref": "cluster-3-time", "effective_samples": 100, "status": "dropped"},
    ]
    result = _summary(windows=windows)
    data = result["training_data"]
    assert data["traceable_cluster_count"] == 2
    assert data["unattributed_samples"] == 10
    assert [item["samples"] for item in data["sources"]] == [40, 10, 10, 20]
    assert sum(item["share"] for item in data["sources"]) == pytest.approx(1)
    assert data["sources"][0]["label"] == "工况组 1"
    assert "来源数量本身不能证明状态混杂" in result["engineering_messages"][-1]


def test_quality_exploratory_and_short_segments_are_not_certified():
    result = _summary(np.ones(5), purpose="exploratory")
    assert result["statistics"]["spe"]["trend"] == "insufficient"
    assert "探索草稿" in result["engineering_messages"][0]
    assert "不能替代独立验证" in result["notice"]


def test_quality_reports_both_abnormal_statistics_and_legacy_cluster_source():
    windows = [{"start": "2026-01-01", "end": "2026-01-02", "source": "cluster", "source_ref": "cluster-2-2026-01-01", "status": "used", "effective_samples": 80}]
    result = _summary(np.full(80, 4), np.full(80, 7), windows=windows)
    assert any("T² 也频繁超限" in message for message in result["engineering_messages"])
    assert result["training_data"]["sources"][0]["label"] == "工况组 2"


def _condition_summary(scores, windows, **kwargs):
    model = SimpleNamespace(n_components=2, explained_variance_ratio=[0.5, 0.3], t2_limits={0.95:4, 0.99:6}, q_limits={0.95:2, 0.99:3})
    return model_quality_summary(model, scores, windows, 5, "normal_state", **kwargs)["training_condition_diagnostic"]


def _condition_window(scores, start, end, cluster=1, source="cluster", window_id="window"):
    return {"id":window_id, "start":scores.index[start].isoformat(), "end":scores.index[end].isoformat(), "source":source,
            "source_ref":f"state-exploration-{'a'*32}-cluster_{cluster:03d}-candidate-001", "effective_samples":end-start+1, "status":"used"}


def test_training_conditions_use_full_scores_three_groups_and_provenance_only():
    times = pd.date_range("2026-01-01", periods=16, freq="5min")
    times = times[:3].append(times[3:] + pd.Timedelta(minutes=5))
    scores = pd.DataFrame({"t2":[4,6,1,6]+[1]*12, "spe":[0,2,3,0]+[1]*4+[3]*4+[1]*4}, index=times)
    windows = [_condition_window(scores, i*4, i*4+3, i+1, window_id=f"w{i}") for i in range(3)]
    windows += [_condition_window(scores, 12, 13, 9, source="manual", window_id="manual"),
                {**_condition_window(scores, 14, 15, 9, window_id="unknown"), "source_ref":"unparseable"}]
    result = _condition_summary(scores, windows)
    assert result["traceable_samples"] == 12
    assert result["unattributed_samples"] == 4
    assert [row["samples"] for row in result["groups"]] == [4,4,4]
    assert [row["share"] for row in result["groups"]] == [0.25]*3
    group = result["groups"][0]
    assert group["cluster_id"] == "cluster_001"
    assert group["t2_95_exceedance_rate"] == 3/4
    assert group["t2_99_exceedance_rate"] == 2/4
    assert group["spe_95_exceedance_rate"] == 2/4
    assert group["spe_99_exceedance_rate"] == 1/4
    assert group["t2_95_ratio_median"] == 1.25
    assert group["spe_95_ratio_median"] == 0.5
    assert group["longest_overall_95_minutes"] == 15
    assert result["timeline"][3]["break_before"] is True
    assert result["timeline"][4]["break_before"] is True
    assert result["timeline"][12]["cluster_id"] is None
    assert result["switch_diagnostic"]["available"] is False
    assert "工况组 3" in result["engineering_messages"][0]


@pytest.mark.parametrize("boundary", ["continuous", "gap", "physical_segment", "window"])
def test_condition_switch_requires_observed_labels_and_all_continuity_boundaries(boundary):
    times = pd.date_range("2026-01-01", periods=8, freq="5min")
    if boundary == "gap":
        times = times[:4].append(times[4:] + pd.Timedelta(minutes=5))
    scores = pd.DataFrame({"t2":[1,1,1,6,6,6,1,1], "spe":1.0}, index=times)
    observed = pd.DataFrame({"cluster_id":["cluster_001"]*4+["cluster_002"]*4,
                             "segment_id":[1]*4+([2]*4 if boundary == "physical_segment" else [1]*4)}, index=times)
    windows = [_condition_window(scores, 0, 7)]
    if boundary == "window":
        windows = [_condition_window(scores,0,3,window_id="w1"),_condition_window(scores,4,7,2,window_id="w2")]
    config = PreprocessingConfig(sample_interval_minutes=5, filter_method="none", resampling_method="none", max_lag_minutes=5, lag_step_minutes=5)
    result = _condition_summary(scores, windows, cluster_series={w["id"]:observed for w in windows}, preprocessing_config=config)
    comparison = result["switch_diagnostic"]
    assert comparison["available"] is True
    assert comparison["context_minutes"] == 5
    assert comparison["switch_count"] == (1 if boundary == "continuous" else 0)
    assert sum(row["samples"] for row in (comparison["stable"],comparison["near_switch"])) == 8
    if boundary == "continuous":
        assert comparison["near_switch"] == {"samples":3, "t2_95_exceedance_rate":1.0, "spe_95_exceedance_rate":0.0, "overall_95_exceedance_rate":1.0}
        assert comparison["stable"]["overall_95_exceedance_rate"] == 0
        assert "不表示切换造成异常" in result["engineering_messages"][-1]
    else:
        assert comparison["near_switch"]["samples"] == 0
        assert comparison["near_switch"]["overall_95_exceedance_rate"] is None
        assert result["timeline"][4]["break_before"] is True
        assert max(row["longest_overall_95_minutes"] for row in result["groups"]) == 10


def test_condition_diagnostic_missing_labels_invalid_scores_and_ambiguous_windows():
    scores = pd.DataFrame({"t2":6.0, "spe":3.0, "score_valid":[True,False,True,True]}, index=pd.date_range("2026-01-01",periods=4,freq="5min"))
    window = _condition_window(scores,0,3)
    assert _condition_summary(scores.iloc[:0],[window])["available"] is False
    result = _condition_summary(scores,[window])
    assert result["groups"][0]["samples"] == 3
    assert result["groups"][0]["longest_overall_95_minutes"] == 10
    assert result["timeline"][1]["t2_limit_ratio"] is None
    assert result["timeline"][2]["break_before"] is True
    missing = pd.DataFrame({"cluster_id":["cluster_001",None,"cluster_002",None], "segment_id":1},index=scores.index)
    assert _condition_summary(scores,[window],cluster_series={"window":missing})["switch_diagnostic"]["available"] is False
    result = _condition_summary(scores,[window,window])
    assert result["groups"] == []
    assert result["traceable_samples"] == 0
    assert "无法追溯" in result["message"]


def test_condition_groups_keep_runs_separate_and_independent_events_short():
    scores = pd.DataFrame({"t2":6.0,"spe":1.0},index=pd.date_range("2026-01-01",periods=6,freq="5min"))
    windows = [_condition_window(scores,0,1,window_id="w1"), _condition_window(scores,2,3,window_id="w2"), _condition_window(scores,4,5,window_id="w3")]
    windows[2]["source_ref"] = windows[2]["source_ref"].replace("a"*32,"b"*32)
    result = _condition_summary(scores,windows)
    assert [row["samples"] for row in result["groups"]] == [4,2]
    assert [row["longest_overall_95_minutes"] for row in result["groups"]] == [10,10]
    assert _condition_summary(scores,[{**windows[0],"source":"manual"}])["available"] is False


@pytest.mark.parametrize("method,expected", [("none",5),("trailing_mean",15),("first_order",None)])
def test_condition_context_reuses_filter_history_semantics(method,expected):
    scores = pd.DataFrame({"t2":1.0,"spe":1.0},index=pd.date_range("2026-01-01",periods=8,freq="5min"))
    labels = pd.DataFrame({"cluster_id":["cluster_001"]*4+["cluster_002"]*4,"segment_id":1},index=scores.index)
    config = PreprocessingConfig(sample_interval_minutes=5, filter_method=method, smoothing_window_minutes=15, first_order_alpha=0.3 if method=="first_order" else None, resampling_method="none", max_lag_minutes=5, lag_step_minutes=5)
    result = _condition_summary(scores,[_condition_window(scores,0,7)],cluster_series={"window":labels},preprocessing_config=config)
    assert result["switch_diagnostic"]["context_minutes"] == expected
    if method == "first_order":
        assert result["switch_diagnostic"]["context_policy"] == "full_segment_history"
        assert result["switch_diagnostic"]["near_switch"]["samples"] == 8
