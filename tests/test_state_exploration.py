import numpy as np
import pandas as pd
import pytest

from pca_model_builder.preprocessing import PreprocessingConfig, StateFilter
from pca_model_builder.preprocessing import preprocess_window
from pca_model_builder.state_exploration import (
    ExplorationConfig,
    PerformanceConfig,
    _cluster_candidates,
    _display_points,
    _performance_candidates,
    _preferred_region_candidates,
    _summaries,
    evaluate_preferred_region,
    evaluate_region_comparison,
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
    assert all(
        "performance_valid_count" not in item for item in first["cluster_summaries"]
    )
    assert all(item["source"] == "cluster" and item["comment"] == "" for item in first["cluster_candidates"])
    candidate_ids = {
        item["candidate_id"]
        for item in [*first["cluster_candidates"], *first["performance_candidates"]]
    }
    assert {item["candidate_id"] for item in first["candidate_decisions"]} == candidate_ids
    assert all(
        item["decision"] == "pending"
        and item["comment"] == ""
        and item["decided_at"] is None
        for item in first["candidate_decisions"]
    )


def test_variable_diagnostics_use_unfiltered_engineering_values_at_full_valid_dpca_samples():
    rng = np.random.default_rng(14)
    index = pd.date_range("2026-01-01", periods=240, freq="1min")
    frame = pd.DataFrame(rng.normal(size=(240, 4)), index=index, columns=["A", "B", "C", "D"])
    frame["A"] = 100 + frame["A"] * 10
    frame["B"] = frame["A"] * 2 + rng.normal(scale=0.01, size=240)
    frame["STATE"] = 1.0
    frame.loc[index[60:80], "STATE"] = 0.0
    frame.loc[index[120:125], "C"] = np.inf
    frame["PERF"] = rng.normal(size=240)
    frame = frame.drop(index[170:180])  # physical gap
    tags = ["A", "B", "C", "D"]
    config = PreprocessingConfig(
        sample_interval_minutes=5, resampling_method="mean",
        filter_method="trailing_mean", smoothing_window_minutes=10,
        max_lag_minutes=10, lag_step_minutes=5, gap_threshold_minutes=5,
        state_filters=(StateFilter("STATE", minimum=1),),
    )
    window = (index[0], index[-1])
    result = run_state_exploration(
        frame, tags, config, ExplorationConfig(cluster_count=2, maximum_plot_points=4),
        performance_config=PerformanceConfig("PERF", "higher_is_better"),
        resampling_window=window,
        color_columns=tags,
    )
    processed = preprocess_window(frame, tags, config, preserve_columns=["PERF"], include_intermediates=True, resampling_window=window)
    points = result["cluster_series"]
    expected = processed.resampled.reindex(points.index)[tags]
    diagnostics = result["variable_diagnostics"]
    assert diagnostics["sample_count"] == len(points) > len(result["cluster_series_display"])
    assert diagnostics["sample_scope"] == "full_valid_exploration_samples"
    assert diagnostics["value_scope"] == "resampled_before_filtering"
    assert list(points.index) == list(processed.dynamic.index)
    assert not expected.equals(processed.filtered.reindex(points.index)[tags])
    assert [item["tag"] for item in diagnostics["tag_profiles"]] == tags
    assert diagnostics["summary"]["tag_count"] == 4
    assert processed.dynamic.shape[1] == 12  # Lag stays internal
    pair = next(item for item in diagnostics["high_correlation_pairs"] if item["tag_a"] == "A" and item["tag_b"] == "B")
    assert pair["pearson_r"] == pytest.approx(expected.A.corr(expected.B))
    assert pair["valid_pair_count"] == len(points)
    for item in diagnostics["tag_profiles"]:
        values = expected[item["tag"]]
        assert item["profile"]["mean"] == pytest.approx(values.mean())
        assert item["profile"]["valid_count"] == len(points)
    assert diagnostics["cluster_features"] == result["cluster_quality"]["feature_contrasts"]
    region = evaluate_preferred_region(
        points, [{"center_pc1": 0, "center_pc2": 0, "radius_pc1": 1e6, "radius_pc2": 1e6}],
        result["exploratory_model_summary"]["pc_columns"], result["cluster_centers"],
        engineering_data=result["_color_series"].loc[:, tags],
    )
    assert region["selected_sample_count"] == len(expected)
    for item in region["variables"]:
        assert item["mean"] == pytest.approx(expected[item["tag"]].mean())
        assert item["overall_mean"] == pytest.approx(expected[item["tag"]].mean())
        assert item["deviation"] == pytest.approx(0)
    assert {item["tag"] for item in diagnostics["cluster_features"]} == set(tags)
    for item in diagnostics["cluster_features"]:
        means = expected[item["tag"]].groupby(points.cluster_id).mean()
        assert item["cluster_means"] == pytest.approx(means.to_dict())
        assert item["mean_difference"] == pytest.approx(means.max() - means.min())
        assert item["standardized_difference"] == pytest.approx((means.max() - means.min()) / expected[item["tag"]].std(ddof=0))


@pytest.mark.parametrize("selected_values,reason", [
    ([np.nan, np.inf, -np.inf], "本组无有效均值"),
    ([10., np.inf, np.nan], "有效样本不足"),
    ([10., 20., np.inf], None),
])
def test_region_variable_deviation_requires_two_finite_selected_values(selected_values, reason):
    index = pd.date_range("2026-01-01", periods=6, freq="5min")
    points = pd.DataFrame({"pc1": range(6), "pc2": 0., "cluster_id": "one"}, index=index)
    engineering = pd.DataFrame({"TAG": [*selected_values, 40., 50., 60.]}, index=index)
    region = evaluate_preferred_region(
        points, [{"center_pc1": 1., "center_pc2": 0., "radius_pc1": 1.1, "radius_pc2": 1.}],
        ("pc1", "pc2"), {"one": [0., 0.]}, engineering_data=engineering,
    )
    assert region["selected_sample_count"] == 3
    variable = region["variables"][0]
    numeric = engineering.TAG.replace([np.inf, -np.inf], np.nan)
    assert variable["overall_mean"] == pytest.approx(numeric.mean())
    assert variable["unavailable_reason"] == reason
    if numeric.iloc[:3].count():
        assert variable["mean"] == pytest.approx(numeric.iloc[:3].mean())
    else:
        assert variable["mean"] is None
    if reason is None:
        assert variable["deviation"] == pytest.approx((numeric.iloc[:3].mean() - numeric.mean()) / numeric.std(ddof=0))
    else:
        assert variable["deviation"] is None


def test_region_variables_use_full_deduplicated_union_and_population_std():
    import json

    index = pd.date_range("2026-01-01", periods=6, freq="5min")
    points = pd.DataFrame({"pc1": range(6), "pc2": 0., "pc3": [0., 0., 0., 0., 0., np.nan],
                           "cluster_id": "one"}, index=index)
    engineering = pd.DataFrame({"positive": [10., 20., 30., 40., 100., 9999.],
                                "negative": [-10., -20., -30., -40., -100., -9999.],
                                "constant": 7., "missing": [np.inf] * 6}, index=index)
    ellipses = [{"center_pc1": c, "center_pc2": 0., "radius_pc1": 1.1, "radius_pc2": 1.} for c in [1., 2.]]
    args = (points, ellipses, ("pc1", "pc2", "pc3"), {"one": [0., 0., 0.]})
    old = evaluate_preferred_region(*args)
    region = evaluate_preferred_region(*args, engineering_data=engineering.iloc[::-1])
    assert {key: value for key, value in region.items() if key != "variables"} == old
    assert region["selected_sample_count"] == 4
    by_tag = {item["tag"]: item for item in region["variables"]}
    for tag in ["positive", "negative"]:
        values = engineering[tag].iloc[:5]
        assert by_tag[tag]["mean"] == pytest.approx(values.iloc[:4].mean())
        assert by_tag[tag]["overall_mean"] == pytest.approx(values.mean())
        assert by_tag[tag]["deviation"] == pytest.approx((values.iloc[:4].mean() - values.mean()) / values.std(ddof=0))
    assert [item["tag"] for item in region["variables"]] == ["positive", "negative", "constant", "missing"]
    assert by_tag["constant"]["mean"] == 7
    assert by_tag["constant"]["deviation"] is None
    assert by_tag["missing"]["mean"] is None
    json.dumps(region, allow_nan=False)
    for ellipses in [[], [{"center_pc1": 99., "center_pc2": 0., "radius_pc1": 1., "radius_pc2": 1.}]]:
        empty = evaluate_preferred_region(points, ellipses, args[2], args[3], engineering_data=engineering)
        assert empty["selected_sample_count"] == 0
        assert all(item["mean"] is None and item["deviation"] is None for item in empty["variables"])


def test_preferred_region_uses_union_of_ellipses_and_full_pc_space():
    index = pd.date_range("2026-01-01", periods=5, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": [-1.0, 0.0, 1.0, 2.0, 3.0],
            "pc2": [0.0] * 5,
            "pc3": [0.0, 10.0, 0.0, 2.0, 3.0],
            "cluster_id": ["cluster_001", "cluster_001", "cluster_002", "cluster_002", "cluster_002"],
        },
        index=index,
    )
    result = evaluate_preferred_region(
        points,
        [
            {"center_pc1": 0.0, "center_pc2": 0.0, "radius_pc1": 1.1, "radius_pc2": 1.0},
            {"center_pc1": 2.0, "center_pc2": 0.0, "radius_pc1": 1.1, "radius_pc2": 1.0},
        ],
        ("pc1", "pc2", "pc3"),
        {"cluster_001": np.zeros(3), "cluster_002": np.zeros(3)},
        performance_values=pd.Series([1.0, 2.0, np.inf, 4.0, np.nan], index=index),
        performance_config={
            "performance_tag": "PERF",
            "direction": "target_range",
            "target_min": 1.5,
            "target_max": 4.5,
        },
    )

    assert result["selected_sample_count"] == 5
    assert result["full_valid_sample_count"] == 5
    assert result["selected_sample_ratio"] == 1.0
    assert [item["sample_count"] for item in result["cluster_counts"]] == [2, 3]
    assert [item["share"] for item in result["cluster_counts"]] == [0.4, 0.6]
    assert result["max_cluster_share"] == 0.6
    assert result["performance_valid_count"] == 3
    assert result["performance_target_count"] == 2
    assert result["performance_target_ratio"] == 2 / 3
    assert result["performance_median"] == 2.0

    changed = points.copy()
    changed["pc3"] = [0.0, 10.0, 0.0, 20.0, 3.0]
    changed_result = evaluate_preferred_region(
        changed,
        result["ellipses"],
        ("pc1", "pc2", "pc3"),
        {"cluster_001": np.zeros(3), "cluster_002": np.zeros(3)},
    )
    assert changed_result["stability_score"] != result["stability_score"]


def test_preferred_region_candidates_split_full_series_at_region_gaps_and_segments():
    index = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-01", periods=6, freq="5min"),
            *pd.date_range("2026-01-01 01:00", periods=6, freq="5min"),
        ]
    )
    points = pd.DataFrame(
        {
            "pc1": [0.0, 0.5, 0.0, 50.0, 5.0, 5.0, 5.0, 5.0, 50.0, 5.0, 5.0, 5.0],
            "pc2": np.zeros(12),
            "pc3": [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            "cluster_id": [
                "cluster_001",
                "cluster_001",
                "cluster_001",
                "cluster_002",
                "cluster_002",
                "cluster_002",
                "cluster_003",
                "cluster_003",
                "cluster_003",
                "cluster_003",
                "cluster_003",
                "cluster_003",
            ],
            "segment_id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        },
        index=index,
    )

    candidates = _preferred_region_candidates(
        points,
        [
            {"center_pc1": 0.0, "center_pc2": 0.0, "radius_pc1": 1.1, "radius_pc2": 1.0},
            {"center_pc1": 5.0, "center_pc2": 0.0, "radius_pc1": 0.5, "radius_pc2": 1.0},
        ],
        ("pc1", "pc2", "pc3"),
        {
            "cluster_001": np.zeros(3),
            "cluster_002": np.zeros(3),
            "cluster_003": np.zeros(3),
        },
        5,
        10,
    )

    assert [item["sample_count"] for item in candidates] == [3, 2, 2, 3]
    assert [(item["start"], item["end"]) for item in candidates] == [
        (index[9].isoformat(), index[11].isoformat()),
        (index[4].isoformat(), index[5].isoformat()),
        (index[6].isoformat(), index[7].isoformat()),
        (index[0].isoformat(), index[2].isoformat()),
    ]
    assert [item["associated_cluster_ids"] for item in candidates] == [
        ["cluster_003"],
        ["cluster_002"],
        ["cluster_003"],
        ["cluster_001"],
    ]
    assert len({item["candidate_id"] for item in candidates}) == 4
    assert candidates[0]["stability_score"] > candidates[-1]["stability_score"]
    assert all(item["source"] == "preferred_region" for item in candidates)


def test_preferred_region_candidates_rank_target_ratio_stability_and_duration():
    index = pd.date_range("2026-01-01", periods=12, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            "pc2": np.zeros(12),
            "pc3": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "cluster_id": ["cluster_001"] * 12,
            "segment_id": [0] * 12,
        },
        index=index,
    )
    candidates = _preferred_region_candidates(
        points,
        [{"center_pc1": 0.0, "center_pc2": 0.0, "radius_pc1": 1.0, "radius_pc2": 1.0}],
        ("pc1", "pc2", "pc3"),
        {"cluster_001": np.zeros(3)},
        5,
        10,
        performance_values=pd.Series(
            [1.0, np.nan, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 2.0],
            index=index,
        ),
        performance_config=PerformanceConfig(
            "PERF",
            "target_range",
            target_min=1.0,
            target_max=2.0,
            minimum_duration_minutes=10,
        ),
    )

    assert [item["start"] for item in candidates] == [
        index[8].isoformat(),
        index[4].isoformat(),
        index[0].isoformat(),
    ]
    invalid_performance = candidates[2]
    assert invalid_performance["performance_valid_count"] == 2
    assert invalid_performance["performance_target_count"] == 2
    assert invalid_performance["performance_target_ratio"] == 1.0
    assert invalid_performance["performance_median"] == 1.5
    assert candidates[1]["stability_score"] > invalid_performance["stability_score"]


def test_preferred_region_rejects_invalid_axes_and_empty_region():
    points = pd.DataFrame(
        {
            "pc1": [0.0, 1.0],
            "pc2": [0.0, 1.0],
            "cluster_id": ["cluster_001", "cluster_001"],
        }
    )
    centers = {"cluster_001": np.zeros(2)}

    empty = evaluate_preferred_region(points, [], ("pc1", "pc2"), centers)
    assert empty["selected_sample_count"] == 0
    assert empty["selected_sample_ratio"] == 0.0
    assert empty["stability_score"] is None

    with pytest.raises(ValueError, match="半轴"):
        evaluate_preferred_region(
            points,
            [{"center_pc1": 0, "center_pc2": 0, "radius_pc1": 0, "radius_pc2": 1}],
            ("pc1", "pc2"),
            centers,
        )
    with pytest.raises(ValueError, match="有限数字"):
        evaluate_preferred_region(
            points,
            [{"center_pc1": np.inf, "center_pc2": 0, "radius_pc1": 1, "radius_pc2": 1}],
            ("pc1", "pc2"),
            centers,
        )


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


def test_target_range_candidates_prioritize_stability_before_start_time():
    index = pd.date_range("2026-01-01", periods=8, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(8),
            "pc2": np.zeros(8),
            "pc3": [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            "cluster_id": ["cluster_001"] * 8,
            "segment_id": [0] * 4 + [1] * 4,
        },
        index=index,
    )
    candidates = _performance_candidates(
        points,
        pd.Series([5.0] * 8, index=index),
        PerformanceConfig(
            "PERF",
            "target_range",
            target_min=4,
            target_max=6,
            minimum_duration_minutes=20,
            candidate_count=2,
        ),
        5,
        {"cluster_001": np.zeros(3)},
        ("pc1", "pc2", "pc3"),
    )

    assert candidates[0]["start"] == index[4].isoformat()
    assert candidates[1]["start"] == index[0].isoformat()
    assert candidates[0]["stability_score"] > candidates[1]["stability_score"]
    assert candidates[0]["performance_summary"]["target_ratio"] == 1.0
    assert candidates[0]["performance_summary"]["mean_target_deviation"] == 0.0


def test_target_range_candidates_do_not_prefer_lower_mean_when_fully_in_range():
    index = pd.date_range("2026-01-01", periods=8, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(8),
            "pc2": np.zeros(8),
            "cluster_id": ["cluster_001"] * 8,
            "segment_id": [0] * 4 + [1] * 4,
        },
        index=index,
    )
    candidates = _performance_candidates(
        points,
        pd.Series([6.0] * 4 + [4.0] * 4, index=index),
        PerformanceConfig(
            "PERF",
            "target_range",
            target_min=2,
            target_max=8,
            minimum_duration_minutes=20,
            candidate_count=2,
        ),
        5,
        {"cluster_001": np.zeros(2)},
        ("pc1", "pc2"),
    )

    assert [item["start"] for item in candidates] == [
        index[0].isoformat(),
        index[4].isoformat(),
    ]


def test_display_points_preserves_representative_target_range_samples():
    index = pd.date_range("2026-01-01", periods=100, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(100),
            "pc2": np.zeros(100),
            "cluster_id": ["cluster_001"] * 100,
            "segment_id": [0] * 100,
            "performance_target_met": [
                position in {11, 73} for position in range(100)
            ],
        },
        index=index,
    )

    display = _display_points(points, 4, 5)

    assert len(display) == 4
    assert {index[11], index[73]}.issubset(display.index)


def test_display_points_reserves_target_sample_before_segment_break_budget():
    index = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-01", periods=50, freq="5min"),
            *pd.date_range("2026-01-01 06:00", periods=50, freq="5min"),
        ]
    )
    points = pd.DataFrame(
        {
            "pc1": np.zeros(100),
            "pc2": np.zeros(100),
            "cluster_id": ["cluster_001"] * 100,
            "segment_id": [0] * 50 + [1] * 50,
            "performance_target_met": [position == 30 for position in range(100)],
        },
        index=index,
    )

    display = _display_points(points, 4, 5)

    assert len(display) <= 4
    assert {index[0], index[-1], index[30]}.issubset(display.index)
    assert display["performance_target_met"].eq(True).any()


def test_display_points_target_sample_competes_with_cluster_switch_deterministically():
    index = pd.date_range("2026-01-01", periods=100, freq="5min")
    pc1 = np.zeros(100)
    pc1[20], pc1[80] = -100.0, 100.0
    points = pd.DataFrame(
        {
            "pc1": pc1,
            "pc2": np.zeros(100),
            "cluster_id": ["cluster_001"] * 50 + ["cluster_002"] * 50,
            "segment_id": [0] * 100,
            "performance_target_met": [position == 35 for position in range(100)],
        },
        index=index,
    )

    first = _display_points(points, 5, 5)
    second = _display_points(points, 5, 5)

    assert len(first) <= 5
    assert {index[0], index[-1], index[35], index[49], index[50]}.issubset(
        first.index
    )
    assert first["performance_target_met"].eq(True).any()
    pd.testing.assert_frame_equal(first, second)


def test_target_range_status_and_cluster_statistics_use_full_aligned_samples():
    index = pd.date_range("2026-01-01", periods=36, freq="5min")
    frame = pd.DataFrame(
        {
            "A": np.sin(np.linspace(0, 8, len(index))),
            "B": np.cos(np.linspace(0, 8, len(index))),
            "C": np.sin(np.linspace(0, 23, len(index))),
            "D": np.sin(np.linspace(0, 8, len(index)))
            + np.cos(np.linspace(0, 8, len(index)))
            + np.sin(np.linspace(0, 23, len(index)))
            + np.linspace(-0.001, 0.001, len(index)),
            "PERF": np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 6),
        },
        index=index,
    )
    frame.loc[index[4], "PERF"] = np.nan
    frame.loc[index[17], "PERF"] = np.inf
    result = run_state_exploration(
        frame,
        ["A", "B", "C", "D"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        ExplorationConfig(
            cluster_count=2,
            minimum_candidate_duration_minutes=10,
            maximum_plot_points=8,
        ),
        performance_config=PerformanceConfig(
            "PERF", "target_range", target_min=2, target_max=3
        ),
    )

    full = result["cluster_series"]
    finite = np.isfinite(frame["PERF"].to_numpy())
    expected_target = finite & (frame["PERF"].to_numpy() >= 2) & (
        frame["PERF"].to_numpy() <= 3
    )
    assert "performance_target_met" in full.columns
    from pca_model_builder.web import _exploration_scatter_series
    scatter = _exploration_scatter_series(full)
    assert [row["timestamp"] for row in scatter] == [timestamp.isoformat() for timestamp in full.index]
    assert len(scatter) > len(result["cluster_series_display"])
    assert [row["performance_target_met"] for row in scatter] == full.performance_target_met.tolist()
    for position, timestamp in enumerate(full.index):
        status = full.loc[timestamp, "performance_target_met"]
        if not finite[position]:
            assert status is None
        else:
            assert bool(status) is bool(expected_target[position])

    display = result["cluster_series_display"]
    for timestamp in display.index:
        assert display.loc[timestamp, "performance_target_met"] == full.loc[
            timestamp, "performance_target_met"
        ]

    summaries = result["cluster_summaries"]
    assert sum(item["performance_valid_count"] for item in summaries) == int(finite.sum())
    assert sum(item["performance_target_count"] for item in summaries) == int(
        expected_target.sum()
    )
    for item in summaries:
        valid_count = item["performance_valid_count"]
        expected_ratio = (
            item["performance_target_count"] / valid_count if valid_count else None
        )
        assert item["performance_target_ratio"] == expected_ratio
        assert item["performance_median"] is not None


def test_performance_tag_cannot_overlap_state_exploration_pca_inputs():
    index = pd.date_range("2026-01-01", periods=12, freq="5min")
    frame = pd.DataFrame(
        {
            "A": np.linspace(0.0, 1.0, len(index)),
            "B": np.linspace(1.0, 2.0, len(index)),
            "PERF": np.linspace(2.0, 3.0, len(index)),
        },
        index=index,
    )

    with pytest.raises(ValueError, match="性能 Tag.*PCA"):
        run_state_exploration(
            frame,
            ["A", "B", "PERF"],
            PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
            ExplorationConfig(cluster_count=2),
            performance_config=PerformanceConfig("PERF", "higher_is_better"),
        )


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


@pytest.mark.parametrize("state_filter,smoothing,lag", [(False, 0, 0), (True, 10, 10)])
def test_exploration_losses_close_with_invalid_rows_and_state_filter(state_filter, smoothing, lag):
    rng = np.random.default_rng(17)
    index = pd.date_range("2026-01-01", periods=100, freq="5min")
    frame = pd.DataFrame(rng.normal(size=(100, 3)), index=index, columns=["A", "B", "C"])
    frame["D"] = frame.A + frame.B + rng.normal(scale=0.01, size=100)
    frame.loc[index[20:23], "A"] = np.nan
    frame["STATE"] = 1.0
    frame.loc[index[50:54], "STATE"] = 0.0
    result = run_state_exploration(
        frame, ["A", "B", "C", "D"],
        PreprocessingConfig(5, smoothing, lag, 5,
                            filter_method="trailing_mean" if smoothing else "none",
                            state_filters=(StateFilter("STATE", minimum=1),) if state_filter else ()),
        ExplorationConfig(cluster_count=2),
    )
    summary = result["preprocessing_summary"]
    losses = summary["loss_counts"]
    assert losses["input_invalid_loss"] == 3
    assert losses["state_filter_loss"] == (4 if state_filter else 0)
    if smoothing:
        assert losses["filter_warmup_loss"] > 0
        assert losses["lag_warmup_loss"] > 0
    assert summary["loss_count"] == summary["resampled_row_count"] - result["full_point_count"]
    assert summary["loss_ratio"] == pytest.approx(1 - summary["effective_coverage_ratio"])


@pytest.fixture
def comparison_samples():
    index = pd.date_range("2026-01-01", periods=8, freq="5min")
    points = pd.DataFrame({
        "pc1": np.arange(8, dtype=float), "pc2": 0., "pc3": [0.] * 7 + [np.inf],
        "cluster_id": ["one"] * 4 + ["two"] * 4,
    }, index=index)
    engineering = pd.DataFrame({"value": [0., 2., 4., 6., 8., 10., 12., 10000.]}, index=index)
    return points, engineering, ("pc1", "pc2", "pc3"), {"one": [0., 0., 0.], "two": [5., 0., 0.]}


def comparison_ellipse(center, radius):
    return {"center_pc1": center, "center_pc2": 0., "radius_pc1": radius, "radius_pc2": 1.}


@pytest.mark.parametrize("direction,sign", [("higher_is_better", 1), ("lower_is_better", -1)])
@pytest.mark.parametrize("center,expected", [(1, -2), (3, 0), (5, 2)])
def test_preferred_region_monotonic_full_population(comparison_samples, direction, sign, center, expected):
    points, _, columns, centers = comparison_samples
    values = pd.Series(range(8), index=points.index).iloc[::-1]
    ellipse = comparison_ellipse(center, 1)
    result = evaluate_preferred_region(points, [ellipse, ellipse], columns, centers,
        performance_values=values, performance_config=PerformanceConfig("PERF", direction))
    assert result["performance_valid_count"] == 3
    assert result["performance_median"] == center
    assert result["performance_overall_median"] == 3
    assert result["performance_improvement"] == sign * expected
    assert "performance_target_ratio" not in result


@pytest.mark.parametrize("direction,sign", [("higher_is_better", 1), ("lower_is_better", -1)])
@pytest.mark.parametrize("a,b,expected", [(2, 4, -2), (4, 2, 2), (3, 3, 0)])
def test_region_comparison_monotonic_overlap(comparison_samples, direction, sign, a, b, expected):
    points, engineering, columns, centers = comparison_samples
    result = evaluate_region_comparison(points, [comparison_ellipse(a, 2)] * 2,
        [comparison_ellipse(b, 2)], columns, centers, engineering_data=engineering,
        performance_values=pd.Series(range(8), index=points.index),
        performance_config=PerformanceConfig("PERF", direction))
    assert result["region_a"]["performance_valid_count"] == 5
    assert result["region_b"]["performance_valid_count"] == 5
    assert result["region_a"]["performance_median"] == a
    assert result["region_b"]["performance_median"] == b
    assert result["performance_improvement"] == sign * expected


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
@pytest.mark.parametrize("values,ellipses,count,median,overall", [
    (["1", "bad", np.inf, -np.inf, 5, np.nan, 7, 9999], [comparison_ellipse(1, 1)], 1, 1, 5),
    ([np.nan] * 8, [comparison_ellipse(1, 1)], 0, None, None),
    (None, [comparison_ellipse(1, 1)], 0, None, None),
    (list(range(8)), [], 0, None, 3),
    ([np.inf] * 3 + [3, 4, 5, 6, 7], [comparison_ellipse(1, 1)], 0, None, 4.5),
])
def test_region_monotonic_missing_values(comparison_samples, direction, values, ellipses, count, median, overall):
    points, engineering, columns, centers = comparison_samples
    performance = pd.Series(values, index=points.index).iloc[::-1] if values is not None else None
    result = evaluate_region_comparison(points, ellipses, [comparison_ellipse(5, 1)],
        columns, centers, engineering_data=engineering, performance_values=performance,
        performance_config=PerformanceConfig("PERF", direction))
    a = result["region_a"]
    assert a["performance_valid_count"] == count
    assert a["performance_median"] == median
    assert a["performance_overall_median"] == overall
    if median is None:
        assert a["performance_improvement"] is None
    if median is None:
        assert result["performance_improvement"] is None
    else:
        assert result["performance_improvement"] == (-5 if direction == "higher_is_better" else 5)
    import json
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("a,b,counts", [
    ([comparison_ellipse(1, 1)], [comparison_ellipse(5, 1)], (0, 3, 3)),
    ([comparison_ellipse(2, 2)], [comparison_ellipse(4, 2)], (3, 2, 2)),
    ([comparison_ellipse(3, 3)], [comparison_ellipse(3, 3)], (7, 0, 0)),
    ([comparison_ellipse(1, 1), comparison_ellipse(2, 1)],
     [comparison_ellipse(5, 1), comparison_ellipse(6, 1)], (0, 4, 3)),
    ([], [comparison_ellipse(3, 3)], (0, 0, 7)),
])
def test_region_comparison_unions_overlap_and_full_summaries(comparison_samples, a, b, counts):
    import copy
    import json
    points, engineering, columns, centers = comparison_samples
    before_points, before_engineering = points.copy(deep=True), engineering.copy(deep=True)
    before_a, before_b, before_centers = copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(centers)
    performance = pd.Series(np.arange(8), index=points.index).iloc[::-1]
    config = PerformanceConfig("PERF", "target_range", target_min=2, target_max=5)
    result = evaluate_region_comparison(points, a, b, columns, centers,
        engineering_data=engineering.iloc[::-1], performance_values=performance, performance_config=config)
    assert (result["intersection_sample_count"], result["a_only_sample_count"], result["b_only_sample_count"]) == counts
    for key, ellipses in [("region_a", a), ("region_b", b)]:
        assert result[key] == evaluate_preferred_region(points, ellipses, columns, centers,
            performance_values=performance, performance_config=config)
        assert "variables" not in result[key]
        assert result[key]["full_valid_sample_count"] == 7
    variable = result["variables"][0]
    mask = lambda ellipses: np.array([any(abs(x - e["center_pc1"]) <= e["radius_pc1"] for e in ellipses) and x < 7 for x in range(8)])
    a_mask, b_mask = mask(a), mask(b)
    a_values = engineering.loc[a_mask & ~b_mask, "value"]
    b_values = engineering.loc[b_mask & ~a_mask, "value"]
    assert variable["a_valid_count"] == len(a_values)
    assert variable["b_valid_count"] == len(b_values)
    if len(a_values) and len(b_values):
        difference = a_values.mean() - b_values.mean()
        assert variable["mean_difference"] == pytest.approx(difference)
        assert variable["standardized_difference"] == pytest.approx(difference / engineering.value.iloc[:7].std(ddof=0))
    else:
        assert variable["standardized_difference"] is None
        assert variable["unavailable_reason"]
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(points, before_points)
    pd.testing.assert_frame_equal(engineering, before_engineering)
    assert (a, b, centers) == (before_a, before_b, before_centers)


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("values,reason", [
    ([np.nan, np.inf, -np.inf], "本组无有效均值"),
    ([2., np.nan, np.inf], "有效样本不足"),
    ([2., 4., np.inf], None),
])
def test_region_comparison_each_side_requires_two_finite_values(comparison_samples, side, values, reason):
    points, engineering, columns, centers = comparison_samples
    engineering.loc[points.index[:3] if side == "a" else points.index[4:7], "value"] = values
    result = evaluate_region_comparison(points, [comparison_ellipse(1, 1)], [comparison_ellipse(5, 1)],
        columns, centers, engineering_data=engineering)
    variable = result["variables"][0]
    finite = np.asarray(values)[np.isfinite(values)]
    assert variable[side + "_valid_count"] == len(finite)
    if len(finite):
        assert variable[side + "_mean"] == pytest.approx(finite.mean())
        assert variable["mean_difference"] == pytest.approx(variable["a_mean"] - variable["b_mean"])
    else:
        assert variable[side + "_mean"] is None
        assert variable["mean_difference"] is None
    assert variable["unavailable_reason"] == reason
    assert (variable["standardized_difference"] is None) == (reason is not None)


def test_region_comparison_constants_sorting_and_index_alignment(comparison_samples):
    import json
    points, engineering, columns, centers = comparison_samples
    engineering["negative"] = -engineering.value
    engineering["zero"] = [0., 1., 2., 5., 0., 1., 2., 10000.]
    engineering["constant"] = 7.
    engineering["missing"] = np.nan
    engineering["non_numeric"] = "bad"
    engineering.loc[pd.Timestamp("2027-01-01")] = [1e9] * 6
    result = evaluate_region_comparison(points, [comparison_ellipse(1, 1)], [comparison_ellipse(5, 1)],
        columns, centers, engineering_data=engineering.iloc[::-1])
    variables = result["variables"]
    assert [v["tag"] for v in variables] == ["value", "negative", "zero", "constant", "missing", "non_numeric"]
    assert variables[0]["standardized_difference"] == -variables[1]["standardized_difference"]
    assert variables[2]["standardized_difference"] == 0
    assert variables[3]["a_mean"] == variables[3]["b_mean"] == 7
    assert variables[3]["mean_difference"] == 0
    assert variables[3]["standardized_difference"] is None
    assert variables[3]["unavailable_reason"] == "精确常量"
    assert variables[4]["unavailable_reason"] == "本组无有效均值"
    json.dumps(result, allow_nan=False)
    missing_row = evaluate_region_comparison(points, [comparison_ellipse(1, 1)], [comparison_ellipse(5, 1)],
        columns, centers, engineering_data=engineering.drop(points.index[0]))["variables"][0]
    assert missing_row["a_valid_count"] == 2
    assert missing_row["a_mean"] == 3


@pytest.mark.parametrize("bad", [comparison_ellipse(1, 0), comparison_ellipse(np.inf, 1)])
@pytest.mark.parametrize("side", ["a", "b"])
def test_region_comparison_reuses_ellipse_validation(comparison_samples, bad, side):
    points, engineering, columns, centers = comparison_samples
    with pytest.raises(ValueError):
        evaluate_region_comparison(points, [bad] if side == "a" else [], [bad] if side == "b" else [],
            columns, centers, engineering_data=engineering)


def test_region_comparison_nonfinite_population_std_keeps_means(comparison_samples):
    points, engineering, columns, centers = comparison_samples
    engineering["value"] = [0., 1., 2., 1e308, 3., 4., 5., 0.]
    with np.errstate(over="ignore", invalid="ignore"):
        variable = evaluate_region_comparison(points, [comparison_ellipse(1, 1)], [comparison_ellipse(5, 1)],
            columns, centers, engineering_data=engineering)["variables"][0]
    assert variable["a_mean"] == 1.
    assert variable["b_mean"] == 4.
    assert variable["mean_difference"] == -3.
    assert variable["standardized_difference"] is None
    assert variable["unavailable_reason"] == "总体标准差不可计算"
