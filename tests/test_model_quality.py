from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.model_quality import model_quality_summary


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
    assert data["sources"][0]["label"] == "Cluster_001"
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
    assert result["training_data"]["sources"][0]["label"] == "Cluster_002"
