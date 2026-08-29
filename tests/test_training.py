import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    StateFilter,
    build_dynamic_matrix,
)
from pca_model_builder.training import build_training_matrix


def _windows(*ranges):
    return [
        {
            "id": f"window-{position:03d}",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": "manual",
            "source_ref": None,
            "enabled": enabled,
            "comment": "",
        }
        for position, (start, end, enabled) in enumerate(ranges, start=1)
    ]


def _frame(periods=80):
    time = pd.date_range("2026-01-01", periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "time": time,
            "A": np.arange(periods, dtype=float),
            "B": np.sin(np.arange(periods, dtype=float) / 3.0),
            "C": np.cos(np.arange(periods, dtype=float) / 5.0),
        }
    )


def _multistate_frame(second_value=20.0):
    rng = np.random.default_rng(73)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    return pd.DataFrame(
        {
            "time": time,
            "A": [10.0] * 60 + [second_value] * 60,
            "B": rng.normal(size=120),
            "C": rng.normal(size=120),
        }
    )


def _two_windows(frame):
    return _windows(
        (frame.time.iloc[0], frame.time.iloc[59], True),
        (frame.time.iloc[60], frame.time.iloc[119], True),
    )


def test_training_builds_windows_independently_without_lag_or_smoothing_leakage():
    frame = _frame()
    frame.loc[:19, ["A", "B", "C"]] = 10_000.0
    config = PreprocessingConfig(5, 10, 10, 5, filter_method="trailing_mean")
    windows = _windows(
        (frame.time.iloc[0], frame.time.iloc[19], True),
        (frame.time.iloc[40], frame.time.iloc[79], True),
    )

    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    second_window = result.dynamic.loc[result.dynamic.index >= frame.time.iloc[40]]

    assert not second_window.empty
    assert second_window.to_numpy().max() < 200.0
    assert [summary["effective_samples"] for summary in result.window_summaries] == [17, 37]


def test_training_summary_separates_resampling_reduction_from_warmup():
    time = pd.date_range("2026-01-01", periods=61, freq="1min")
    frame = pd.DataFrame(
        {
            "time": time,
            "A": np.arange(61, dtype=float),
            "B": np.sin(np.arange(61, dtype=float)),
            "C": np.cos(np.arange(61, dtype=float)),
        }
    )
    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(
            5, 0, 0, 5, resampling_method="mean", filter_method="none"
        ),
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
    )

    summary = result.window_summaries[0]
    assert summary["resampling_row_reduction"] == 48
    assert summary["partial_resampling_bin_loss"] == 1
    assert summary["partial_resampling_row_loss"] == 1
    assert summary["filter_warmup_loss"] == 0
    assert summary["lag_warmup_loss"] == 0
    assert summary["lag_context_invalid_loss"] == 0
    assert summary["smoothing_lag_loss"] == 0


def test_training_restarts_preprocessing_at_physical_time_gap_and_records_segments():
    frame = _frame(40)
    frame.loc[:19, ["A", "B", "C"]] = 10_000.0
    frame.loc[20:, "time"] += pd.Timedelta(minutes=20)
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows((frame.time.iloc[0], frame.time.iloc[-1], True))

    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    post_gap = result.dynamic.loc[result.dynamic.index >= frame.time.iloc[20]]
    summary = result.window_summaries[0]

    assert len(summary["segments"]) == 2
    assert all(segment["status"] == "used" for segment in summary["segments"])
    assert not post_gap.empty
    assert post_gap.to_numpy().max() < 200.0


def test_training_single_window_matches_existing_dynamic_matrix_and_disabled_windows_do_not_contribute():
    frame = _frame()
    frame.loc[40:, ["A", "B", "C"]] = 10_000.0
    config = PreprocessingConfig(5, 10, 10, 5)
    enabled = _windows((frame.time.iloc[0], frame.time.iloc[39], True))
    with_disabled = _windows(
        (frame.time.iloc[0], frame.time.iloc[39], True),
        (frame.time.iloc[40], frame.time.iloc[79], False),
    )

    expected = build_dynamic_matrix(
        frame.iloc[:40].set_index("time")[["A", "B", "C"]], ["A", "B", "C"], config
    )
    single = build_training_matrix(frame, "time", ["A", "B", "C"], config, enabled)
    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, with_disabled
    )

    pd.testing.assert_frame_equal(single.dynamic, expected)
    pd.testing.assert_frame_equal(result.dynamic, expected)
    assert result.window_summaries[1]["status"] == "disabled"
    assert result.window_summaries[0]["effective_sample_share"] == 1.0
    assert result.window_summaries[1]["effective_sample_share"] == 0.0
    assert result.training_window_totals == {
        "enabled_window_count": 1,
        "used_window_count": 1,
        "dropped_window_count": 0,
        "training_rows": len(expected),
        "used_segment_count": 1,
        "covered_day_count": 1,
        "max_window_id": "window-001",
        "max_window_effective_samples": len(expected),
        "max_window_effective_share": 1.0,
        "source_summary": {
            "manual": {
                "used_window_count": 1,
                "effective_samples": len(expected),
                "effective_sample_share": 1.0,
            }
        },
    }
    np.testing.assert_allclose(result.dynamic.mean().to_numpy(), expected.mean().to_numpy())


def test_training_reports_source_shares_segments_and_final_day_coverage():
    frame = _frame(24)
    frame.loc[12:, "time"] += pd.Timedelta(days=1)
    windows = [
        {
            "id": "manual-window",
            "start": frame.time.iloc[0].isoformat(),
            "end": frame.time.iloc[5].isoformat(),
            "source": "manual",
            "source_ref": None,
            "enabled": True,
            "comment": "",
        },
        {
            "id": "preferred-window",
            "start": frame.time.iloc[12].isoformat(),
            "end": frame.time.iloc[17].isoformat(),
            "source": "preferred_region",
            "source_ref": "region-001",
            "enabled": True,
            "comment": "",
        },
        {
            "id": "dropped-window",
            "start": "2026-01-03T00:00:00",
            "end": "2026-01-03T00:25:00",
            "source": "cluster",
            "source_ref": "cluster-001",
            "enabled": True,
            "comment": "",
        },
        {
            "id": "disabled-window",
            "start": frame.time.iloc[18].isoformat(),
            "end": frame.time.iloc[23].isoformat(),
            "source": "performance",
            "source_ref": "performance-001",
            "enabled": False,
            "comment": "",
        },
    ]

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        windows,
    )

    assert len(result.dynamic) == 12
    assert [item["status"] for item in result.window_summaries] == [
        "used",
        "used",
        "dropped",
        "disabled",
    ]
    assert [item["effective_sample_share"] for item in result.window_summaries] == [
        0.5,
        0.5,
        0.0,
        0.0,
    ]
    assert result.training_window_totals["used_segment_count"] == 2
    assert result.training_window_totals["covered_day_count"] == 2
    assert result.training_window_totals["max_window_id"] == "manual-window"
    assert result.training_window_totals["max_window_effective_samples"] == 6
    assert result.training_window_totals["max_window_effective_share"] == 0.5
    assert result.training_window_totals["source_summary"] == {
        "manual": {
            "used_window_count": 1,
            "effective_samples": 6,
            "effective_sample_share": 0.5,
        },
        "preferred_region": {
            "used_window_count": 1,
            "effective_samples": 6,
            "effective_sample_share": 0.5,
        },
        "cluster": {
            "used_window_count": 0,
            "effective_samples": 0,
            "effective_sample_share": 0.0,
        },
        "performance": {
            "used_window_count": 0,
            "effective_samples": 0,
            "effective_sample_share": 0.0,
        },
    }
    assert sum(
        item["effective_sample_share"]
        for item in result.training_window_totals["source_summary"].values()
    ) == 1.0


def test_training_composition_warnings_are_non_blocking_and_do_not_change_dynamic_rows():
    frame = _frame()
    windows = [
        {
            "id": "preferred-window",
            "start": frame.time.iloc[0].isoformat(),
            "end": frame.time.iloc[-1].isoformat(),
            "source": "preferred_region",
            "source_ref": "region-001",
            "enabled": True,
            "comment": "",
        },
        {
            "id": "dropped-window",
            "start": "2026-01-02T00:00:00",
            "end": "2026-01-02T00:25:00",
            "source": "manual",
            "source_ref": None,
            "enabled": True,
            "comment": "",
        },
    ]

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        windows,
    )

    assert result.training_window_totals["used_window_count"] == 1
    assert result.training_window_totals["covered_day_count"] == 1
    assert {warning["code"] for warning in result.global_quality_warnings} == {
        "single_used_training_window",
        "single_training_day",
        "preferred_region_only_training",
    }
    assert all(warning["severity"] == "warning" for warning in result.global_quality_warnings)
    assert len(result.dynamic) == result.training_window_totals["training_rows"]


def test_training_standardizes_from_all_enabled_dynamic_rows():
    frame = _frame(60)
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows(
        (frame.time.iloc[0], frame.time.iloc[29], True),
        (frame.time.iloc[30], frame.time.iloc[59], True),
    )
    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    expected = pd.concat(
        [
            build_dynamic_matrix(
                frame.iloc[:30].set_index("time")[["A", "B", "C"]],
                ["A", "B", "C"],
                config,
            ),
            build_dynamic_matrix(
                frame.iloc[30:].set_index("time")[["A", "B", "C"]],
                ["A", "B", "C"],
                config,
            ),
        ]
    )

    np.testing.assert_allclose(
        result.dynamic.mean().to_numpy(), expected.mean().to_numpy()
    )
    np.testing.assert_allclose(
        result.dynamic.std(ddof=0).to_numpy(),
        expected.std(ddof=0).to_numpy(),
    )


def test_training_records_short_dropped_segments_and_blocks_when_everything_is_invalid():
    frame = _frame(24)
    config = PreprocessingConfig(5, 10, 20, 5)
    valid_and_short = _windows(
        (frame.time.iloc[0], frame.time.iloc[19], True),
        (frame.time.iloc[20], frame.time.iloc[23], True),
    )
    only_short = _windows((frame.time.iloc[20], frame.time.iloc[23], True))

    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, valid_and_short
    )

    assert result.window_summaries[1]["status"] == "dropped"
    assert result.window_summaries[1]["dropped_reason"] == "insufficient_after_smoothing_and_lag"
    with pytest.raises(ValueError, match="所有启用窗口"):
        build_training_matrix(frame, "time", ["A", "B", "C"], config, only_short)


def test_training_checks_variability_after_merging_independent_windows():
    frame = _multistate_frame()
    config = PreprocessingConfig(5, 5, 0, 5)

    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, _two_windows(frame)
    )
    model = fit_dpca(result.dynamic, n_components=2)

    assert result.dynamic["A__lag_000min"].std(ddof=0) > 0
    assert len(result.dynamic) == 120
    assert [summary["effective_samples"] for summary in result.window_summaries] == [60, 60]
    np.testing.assert_allclose(model.mean, result.dynamic.mean().to_numpy())
    np.testing.assert_allclose(model.scale, result.dynamic.std(ddof=0).to_numpy())


def test_training_rejects_features_that_remain_constant_after_merging():
    frame = _multistate_frame(second_value=10.0)

    with pytest.raises(ValueError, match="常量动态特征.*A__lag_000min"):
        build_training_matrix(
            frame,
            "time",
            ["A", "B", "C"],
            PreprocessingConfig(5, 5, 0, 5),
            _two_windows(frame),
        )


def test_training_reports_global_near_constant_feature_without_window_blocking():
    frame = _multistate_frame()
    frame.loc[:59, "A"] = 10.0 + np.arange(60) * 1e-8
    frame.loc[60:, "A"] = 10.0 + np.arange(60) * 1e-8

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 5, 0, 5),
        _two_windows(frame),
    )

    assert any(
        warning["code"] == "near_constant_feature"
        and warning["feature"] == "A__lag_000min"
        for warning in result.global_quality_warnings
    )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda frame: frame.__setitem__("time", list(frame.time.iloc[:60]) + [frame.time.iloc[61]] + list(frame.time.iloc[61:])), "duplicate_timestamp"),
        (lambda frame: frame.__setitem__("time", list(frame.time.iloc[:60]) + [frame.time.iloc[60] + pd.Timedelta(minutes=2)] + list(frame.time.iloc[61:])), "irregular_sampling"),
    ],
)
def test_training_keeps_per_window_safety_checks(mutate, code):
    frame = _multistate_frame()
    mutate(frame)

    with pytest.raises(ValueError, match=code):
        build_training_matrix(
            frame,
            "time",
            ["A", "B", "C"],
            PreprocessingConfig(5, 5, 0, 5),
            _two_windows(frame),
            {"A": (-100.0, 100.0)},
        )


def test_training_excludes_engineering_range_rows_and_restarts_lag_history():
    frame = _multistate_frame()
    outlier_position = 61
    frame.loc[outlier_position, "A"] = 1000.0
    config = PreprocessingConfig(5, 0, 5, 5, filter_method="none")

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        config,
        _two_windows(frame),
        {"A": (-100.0, 100.0)},
        exclude_engineering_range=True,
    )

    summary = result.window_summaries[1]
    assert frame.loc[outlier_position, "A"] == 1000.0
    assert summary["engineering_range_loss"] == 1
    assert summary["engineering_range_loss_by_tag"] == {"A": 1}
    assert summary["input_invalid_loss"] == 0
    assert frame.time.iloc[outlier_position] not in result.dynamic.index
    assert frame.time.iloc[outlier_position + 1] not in result.dynamic.index
    assert result.dynamic.loc[frame.time.iloc[outlier_position + 2], "A__lag_005min"] == 20.0


def test_training_keeps_engineering_range_rows_without_normal_state_exclusion():
    frame = _multistate_frame()
    outlier_position = 61
    frame.loc[outlier_position, "A"] = 1000.0

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        _two_windows(frame),
        {"A": (-100.0, 100.0)},
    )

    assert result.window_summaries[1]["engineering_range_loss"] == 0
    assert result.dynamic.loc[frame.time.iloc[outlier_position], "A__lag_000min"] == 1000.0


def test_training_resamples_each_window_and_records_preprocessing_summary():
    periods = 61
    values = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=periods, freq="1min"),
            "A": values,
            "B": np.sin(values / 3),
            "C": np.cos(values / 5),
        }
    )
    config = PreprocessingConfig(
        5, 0, 0, 5, resampling_method="mean", filter_method="none"
    )

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        config,
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
    )

    summary = result.window_summaries[0]
    assert summary["raw_samples"] == 61
    assert summary["resampled_samples"] == 12
    assert summary["partial_resampling_bin_loss"] == 1
    assert summary["empty_bins"] == 0
    assert summary["filter_warmup_loss"] == 0
    assert summary["effective_samples"] == 12


@pytest.mark.parametrize("invalid", [None, "bad", float("inf"), float("-inf")])
def test_training_drops_invalid_model_rows_and_records_loss(invalid):
    frame = _frame(20)
    frame.loc[10, "A"] = invalid
    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
        validate_dynamic=False,
    )

    summary = result.window_summaries[0]
    assert summary["input_invalid_loss"] == 1
    assert frame.time.iloc[10] not in result.dynamic.index
    assert summary["resampled_samples"] == summary["empty_bins"] + summary["input_invalid_loss"] + summary["state_filter_input_rows"]


def test_training_state_filter_column_is_not_a_dynamic_feature_and_breaks_lag():
    frame = _frame(20)
    frame["LOAD"] = [1] * 8 + [0] * 4 + [1] * 8
    config = PreprocessingConfig(
        5,
        0,
        5,
        5,
        filter_method="first_order",
        first_order_alpha=0.5,
        state_filters=(StateFilter("LOAD", minimum=1),),
    )

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        config,
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
    )

    assert not any("LOAD" in column for column in result.dynamic.columns)
    assert frame.time.iloc[12] not in result.dynamic.index
    assert result.window_summaries[0]["state_filter_output_rows"] == 16
    assert result.window_summaries[0]["first_order_alpha"] == 0.5
    segments = result.window_summaries[0]["segments"]
    assert [segment["state_filter_input_rows"] for segment in segments] == [20]
    assert [segment["state_filter_output_rows"] for segment in segments] == [16]
    assert [segment["state_filter_loss"] for segment in segments] == [4]
    assert sum(segment["state_filter_input_rows"] for segment in segments) == result.window_summaries[0]["state_filter_input_rows"]
    assert sum(segment["state_filter_output_rows"] for segment in segments) == result.window_summaries[0]["state_filter_output_rows"]


def test_training_resampling_keeps_only_complete_window_bins():
    values = np.arange(11, dtype=float)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=11, freq="1min"),
            "A": values,
            "B": np.sin(values),
            "C": np.cos(values),
        }
    )
    config = PreprocessingConfig(
        5, 0, 0, 5, resampling_method="mean", filter_method="none"
    )

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        config,
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
        validate_dynamic=False,
    )

    assert result.dynamic.index.tolist() == [frame.time.iloc[5], frame.time.iloc[10]]
    assert result.reference.time.tolist() == [frame.time.iloc[5], frame.time.iloc[10]]
    assert result.window_summaries[0]["partial_resampling_bin_loss"] == 1


def test_training_drops_windows_split_inside_one_resampling_bin():
    values = np.arange(6, dtype=float)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="1min"),
            "A": values,
            "B": values + 10,
            "C": values + 20,
        }
    )
    config = PreprocessingConfig(
        5, 0, 0, 5, resampling_method="mean", filter_method="none"
    )
    windows = _windows(
        (frame.time.iloc[1], frame.time.iloc[2], True),
        (frame.time.iloc[3], frame.time.iloc[5], True),
    )

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        config,
        windows,
        validate_dynamic=False,
    )

    assert result.dynamic.empty
    assert result.reference.empty
    assert [item["dropped_reason"] for item in result.window_summaries] == [
        "no_complete_resampling_bins",
        "no_complete_resampling_bins",
    ]
    assert not result.dynamic.index.has_duplicates


def test_training_reference_uses_only_state_filtered_conditions():
    frame = _frame(12)
    frame["LOAD"] = [1] * 6 + [0] * 6
    frame["FIXED"] = [7.0] * 6 + list(np.arange(6, dtype=float))
    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(
            5,
            0,
            0,
            5,
            filter_method="none",
            state_filters=(StateFilter("LOAD", minimum=1),),
        ),
        _windows((frame.time.iloc[0], frame.time.iloc[-1], True)),
        validate_dynamic=False,
        reference_columns=["FIXED"],
    )

    assert result.reference.time.tolist() == frame.time.iloc[:6].tolist()
    assert result.reference["FIXED"].nunique() == 1
    assert not any("LOAD" in column for column in result.dynamic.columns)
