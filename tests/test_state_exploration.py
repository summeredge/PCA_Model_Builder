import numpy as np
import pandas as pd
import pytest

from pca_model_builder.preprocessing import PreprocessingConfig, StateFilter
from pca_model_builder.state_exploration import (
    ExplorationConfig,
    PerformanceConfig,
    _cluster_candidates,
    _display_points,
    _performance_candidates,
    _summaries,
    run_state_exploration,
)


def test_state_exploration_is_draft_and_deterministic_with_display_limit():
    index = pd.date_range("2026-01-01", periods=80, freq="5min")
    values = np.r_[np.linspace(-3, -2, 40), np.linspace(2, 3, 40)]
    frame = pd.DataFrame({"A": values, "B": values**2, "C": np.sin(values)}, index=index)
    config = ExplorationConfig(cluster_count=2, minimum_candidate_duration_minutes=10, maximum_plot_points=12)
    first = run_state_exploration(frame, ["A", "B", "C"], PreprocessingConfig(5, 0, 0, 5, filter_method="none"), config)
    second = run_state_exploration(frame, ["A", "B", "C"], PreprocessingConfig(5, 0, 0, 5, filter_method="none"), config)

    assert first["exploratory_model_summary"]["model_purpose"] == "exploratory"
    assert first["exploratory_model_summary"]["model_status"] == "draft"
    assert first["exploratory_model_summary"]["n_components"] >= 2
    pd.testing.assert_frame_equal(first["cluster_series"], second["cluster_series"])
    assert len(first["cluster_series_display"]) <= 12
    assert first["cluster_series"].index.is_unique
    assert sum(item["sample_count"] for item in first["cluster_summaries"]) == len(first["cluster_series"])
    assert all(item["source"] == "cluster" and item["comment"] == "" for item in first["cluster_candidates"])


def test_cluster_metrics_use_the_complete_principal_component_space():
    index = pd.date_range("2026-01-01", periods=90, freq="5min")
    x = np.sin(np.linspace(0, 8, len(index)))
    y = np.cos(np.linspace(0, 8, len(index)))
    z = np.sin(np.linspace(0, 23, len(index)))
    frame = pd.DataFrame(
        {
            "A": x,
            "B": y,
            "C": z,
            "D": x + y + z + np.linspace(-0.001, 0.001, len(index)),
        },
        index=index,
    )
    result = run_state_exploration(
        frame,
        ["A", "B", "C", "D"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        ExplorationConfig(
            cluster_count=2,
            minimum_candidate_duration_minutes=10,
            maximum_plot_points=20,
        ),
    )

    assert "pc3" in result["cluster_series"].columns
    assert all(len(item["centroid_pc_scores"]) >= 3 for item in result["cluster_summaries"])
    assert all("centroid_distance" in item for item in result["cluster_candidates"])


def test_cluster_candidate_distance_and_summary_change_when_pc3_changes():
    index = pd.date_range("2026-01-01", periods=6, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(6),
            "pc2": np.zeros(6),
            "pc3": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cluster_id": ["cluster_001"] * 6,
            "segment_id": [0] * 6,
        },
        index=index,
    )
    config = ExplorationConfig(
        cluster_count=2,
        minimum_candidate_duration_minutes=10,
        candidate_count_per_cluster=1,
    )
    centers = {"cluster_001": np.zeros(3)}
    columns = ("pc1", "pc2", "pc3")
    candidates = _cluster_candidates(points, config, 5, centers, columns)
    dynamic = pd.DataFrame({"A__lag_000min": np.ones(6)}, index=index)
    summaries = _summaries(points, dynamic, ["A"], candidates, 5, centers, columns)

    changed = points.copy()
    changed["pc3"] += 10
    changed_candidate = _cluster_candidates(changed, config, 5, centers, columns)[0]
    changed_summary = _summaries(
        changed, dynamic, ["A"], [changed_candidate], 5, centers, columns
    )[0]

    assert candidates[0]["centroid_distance"] != changed_candidate["centroid_distance"]
    assert summaries[0]["median_distance_to_centroid"] != changed_summary[
        "median_distance_to_centroid"
    ]


def test_performance_candidates_support_all_directions_and_do_not_cross_segments():
    index = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-01", periods=6, freq="5min"),
            *pd.date_range("2026-01-01 01:00", periods=6, freq="5min"),
        ]
    )
    points = pd.DataFrame(
        {
            "pc1": np.zeros(12),
            "pc2": np.zeros(12),
            "pc3": np.zeros(12),
            "cluster_id": ["cluster_001"] * 6 + ["cluster_002"] * 6,
            "segment_id": [0] * 6 + [1] * 6,
        },
        index=index,
    )
    centers = {"cluster_001": np.zeros(3), "cluster_002": np.zeros(3)}
    columns = ("pc1", "pc2", "pc3")
    values = pd.Series([10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6], index=index)

    higher = _performance_candidates(
        points,
        values,
        PerformanceConfig("PERF", "higher_is_better", minimum_duration_minutes=20),
        5,
        centers,
        columns,
    )
    lower = _performance_candidates(
        points,
        values,
        PerformanceConfig("PERF", "lower_is_better", minimum_duration_minutes=20),
        5,
        centers,
        columns,
    )
    target = _performance_candidates(
        points,
        values,
        PerformanceConfig(
            "PERF", "target_range", target_min=2, target_max=5, minimum_duration_minutes=20
        ),
        5,
        centers,
        columns,
    )

    assert higher[0]["start"] == index[2].isoformat()
    assert lower[0]["start"] == index[6].isoformat()
    assert target[0]["start"] == index[7].isoformat()
    assert higher[0]["source"] == "performance"
    assert higher[0]["comment"] == ""
    assert higher[0]["associated_cluster_ids"] == ["cluster_001"]


@pytest.mark.parametrize(
    ("direction", "values", "kwargs"),
    [
        ("higher_is_better", [1] * 6 + [9] * 6 + [1] * 6, {}),
        ("lower_is_better", [9] * 6 + [1] * 6 + [9] * 6, {}),
        (
            "target_range",
            [0] * 6 + [3] * 6 + [0] * 6,
            {"target_min": 2, "target_max": 4},
        ),
    ],
)
def test_performance_candidates_rank_local_windows_within_one_continuous_segment(
    direction, values, kwargs
):
    index = pd.date_range("2026-01-01", periods=18, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(18),
            "pc2": np.zeros(18),
            "cluster_id": ["cluster_001"] * 18,
            "segment_id": [0] * 18,
        },
        index=index,
    )
    config = PerformanceConfig(
        "PERF",
        direction,
        minimum_duration_minutes=20,
        candidate_count=2,
        **kwargs,
    )
    centers = {"cluster_001": np.zeros(2)}

    first = _performance_candidates(
        points, pd.Series(values, index=index), config, 5, centers, ("pc1", "pc2")
    )
    second = _performance_candidates(
        points, pd.Series(values, index=index), config, 5, centers, ("pc1", "pc2")
    )

    assert first == second
    assert first[0]["start"] == index[6].isoformat()
    assert first[0]["end"] == index[9].isoformat()
    selected = [
        set(pd.date_range(item["start"], item["end"], freq="5min"))
        for item in first
    ]
    assert not selected[0].intersection(selected[1])


def test_performance_candidates_scale_to_long_continuous_history_without_copies(
    monkeypatch,
):
    count = 100_000
    window_rows = 12
    index = pd.date_range("2025-01-01", periods=count, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.sin(np.arange(count) / 25.0),
            "pc2": np.cos(np.arange(count) / 31.0),
            "pc3": np.sin(np.arange(count) / 43.0),
            "pc4": np.cos(np.arange(count) / 47.0),
            "cluster_id": np.where(
                np.arange(count) < count // 2, "cluster_001", "cluster_002"
            ),
            "segment_id": np.zeros(count, dtype=int),
        },
        index=index,
    )
    values = pd.Series(np.arange(count, dtype=float), index=index)
    centers = {"cluster_001": np.zeros(4), "cluster_002": np.zeros(4)}

    def fail_copy(*args, **kwargs):
        raise AssertionError("sliding performance windows must not copy DataFrames")

    monkeypatch.setattr(pd.DataFrame, "copy", fail_copy)
    first = _performance_candidates(
        points,
        values,
        PerformanceConfig(
            "PERF",
            "higher_is_better",
            minimum_duration_minutes=window_rows * 5,
            candidate_count=3,
        ),
        5,
        centers,
        ("pc1", "pc2", "pc3", "pc4"),
    )
    second = _performance_candidates(
        points,
        values,
        PerformanceConfig(
            "PERF",
            "higher_is_better",
            minimum_duration_minutes=window_rows * 5,
            candidate_count=3,
        ),
        5,
        centers,
        ("pc1", "pc2", "pc3", "pc4"),
    )

    assert first == second
    assert [item["start"] for item in first] == [
        index[count - window_rows * rank].isoformat() for rank in range(1, 4)
    ]
    assert all(
        item["associated_cluster_ids"] == ["cluster_002"]
        and np.isfinite(item["stability_score"])
        for item in first
    )
    selected = [
        set(pd.date_range(item["start"], item["end"], freq="5min"))
        for item in first
    ]
    assert all(not left.intersection(right) for left, right in zip(selected, selected[1:]))


def test_performance_config_rejects_incomplete_or_irrelevant_bounds():
    with pytest.raises(ValueError, match="requires target_min"):
        PerformanceConfig("PERF", "target_range")
    with pytest.raises(ValueError, match="only valid"):
        PerformanceConfig("PERF", "higher_is_better", target_min=0)
    with pytest.raises(ValueError, match="must not exceed"):
        PerformanceConfig("PERF", "target_range", target_min=2, target_max=1)


def test_display_points_preserves_boundaries_switches_extrema_and_is_deterministic():
    index = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-01", periods=6, freq="5min"),
            *pd.date_range("2026-01-01 01:00", periods=6, freq="5min"),
        ]
    )
    points = pd.DataFrame(
        {
            "pc1": [0, 1, 2, 100, 4, 5, 6, 7, -100, 9, 10, 11],
            "pc2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "cluster_id": ["cluster_001"] * 6 + ["cluster_002"] * 6,
            "segment_id": [0] * 6 + [1] * 6,
        },
        index=index,
    )
    first = _display_points(points, 10, 5)
    second = _display_points(points, 10, 5)

    assert len(first) <= 10
    assert first.index[0] == index[0]
    assert first.index[-1] == index[-1]
    assert {index[5], index[6]}.issubset(first.index)
    assert {index[3], index[8]}.issubset(first.index)
    pd.testing.assert_frame_equal(first, second)


def test_state_filter_loss_is_counted_once_and_cluster_switches_split_runs():
    index = pd.date_range("2026-01-01", periods=12, freq="5min")
    x = np.sin(np.linspace(0, 7, len(index)))
    y = np.cos(np.linspace(0, 7, len(index)))
    z = np.sin(np.linspace(0, 19, len(index)))
    frame = pd.DataFrame(
        {
            "A": x,
            "B": y,
            "C": z,
            "D": x + y + z + np.linspace(-0.001, 0.001, len(index)),
            "STATE": [0.0] * 6 + [1.0] * 6,
        },
        index=index,
    )
    result = run_state_exploration(
        frame,
        ["A", "B", "C", "D"],
        PreprocessingConfig(
            5,
            0,
            0,
            5,
            filter_method="none",
            state_filters=(StateFilter("STATE", minimum=0.5),),
        ),
        ExplorationConfig(
            cluster_count=2,
            minimum_candidate_duration_minutes=5,
            candidate_count_per_cluster=2,
        ),
    )

    losses = result["preprocessing_summary"]["loss_counts"]
    assert losses["state_filter_loss"] == 6
    assert result["preprocessing_summary"]["loss_count"] == sum(losses.values())
    assert result["preprocessing_summary"]["loss_denominator"] == 12

    points = result["cluster_series"].copy()
    points["cluster_id"] = [
        "cluster_001",
        "cluster_001",
        "cluster_002",
        "cluster_002",
        "cluster_001",
        "cluster_001",
    ]
    summaries = _summaries(
        points,
        result["cluster_series"].assign(**{"A__lag_000min": 1.0}),
        ["A"],
        [],
        5,
        result["cluster_centers"],
        tuple(result["exploratory_model_summary"]["pc_columns"]),
    )
    assert next(item for item in summaries if item["cluster_id"] == "cluster_001")[
        "segment_count"
    ] == 2
