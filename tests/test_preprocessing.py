import numpy as np
import pandas as pd
import pytest

from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    StateFilter,
    apply_state_filters,
    build_dynamic_matrix,
    filter_segment,
    infer_segment_ids,
    preprocess_window,
    resample_segment,
)


def test_dynamic_lags_do_not_cross_physical_time_gaps():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
            "2026-01-01 00:40",
        ]
    )
    frame = pd.DataFrame(
        {"T1": [0.0, 1.0, 2.0, 10.0, 12.0, 14.0]}, index=index
    )
    segments = infer_segment_ids(index, sample_interval_minutes=5)
    config = PreprocessingConfig(
        sample_interval_minutes=5,
        smoothing_window_minutes=10,
        max_lag_minutes=5,
        lag_step_minutes=5,
    )

    dynamic = build_dynamic_matrix(frame, ["T1"], config, segments)

    assert dynamic.index.tolist() == [index[2], index[5]]
    assert dynamic.loc[index[5], "T1__lag_000min"] == 13.0
    assert dynamic.loc[index[5], "T1__lag_005min"] == 11.0


def test_smoothing_is_trailing_and_never_uses_future_samples():
    index = pd.date_range("2026-01-01", periods=4, freq="5min")
    frame = pd.DataFrame({"T1": [0.0, 10.0, 20.0, 30.0]}, index=index)
    config = PreprocessingConfig(
        sample_interval_minutes=5,
        smoothing_window_minutes=10,
        max_lag_minutes=0,
        lag_step_minutes=5,
    )

    dynamic = build_dynamic_matrix(
        frame,
        ["T1"],
        config,
        infer_segment_ids(index, sample_interval_minutes=5),
    )

    assert dynamic.index.tolist() == index[1:].tolist()
    assert dynamic.iloc[0, 0] == 5.0
    assert dynamic.iloc[-1, 0] == 25.0


def test_lag_configuration_must_match_sampling_grid():
    with pytest.raises(ValueError, match="integer multiple"):
        PreprocessingConfig(
            sample_interval_minutes=3,
            smoothing_window_minutes=9,
            max_lag_minutes=10,
            lag_step_minutes=5,
        )


def test_dynamic_matrix_rejects_non_finite_values():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({"T1": [1.0, np.inf, 2.0]}, index=index)
    config = PreprocessingConfig(5, 5, 0, 5)

    with pytest.raises(ValueError, match="non-finite"):
        build_dynamic_matrix(frame, ["T1"], config)


def test_segment_inference_rejects_interval_off_sampling_grid():
    index = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:17"]
    )

    with pytest.raises(ValueError, match="sampling grid"):
        infer_segment_ids(index, sample_interval_minutes=5)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", [2.0, 10.0]),
        ("median", [2.0, 10.0]),
        ("last", [3.0, 10.0]),
    ],
)
def test_resampling_uses_fixed_right_closed_epoch_buckets(method, expected):
    index = pd.to_datetime(["2026-01-01 00:01", "2026-01-01 00:04", "2026-01-01 00:06"])
    frame = pd.DataFrame({"A": [1.0, 3.0, 10.0]}, index=index)

    result, empty_bins = resample_segment(frame, method, 5)

    assert result.index.tolist() == pd.to_datetime(
        ["2026-01-01 00:05", "2026-01-01 00:10"]
    ).tolist()
    assert result["A"].tolist() == expected
    assert empty_bins == 0


def test_resampling_last_preserves_missing_last_row_and_empty_bins():
    index = pd.to_datetime(
        ["2026-01-01 00:01", "2026-01-01 00:04", "2026-01-01 00:11"]
    )
    frame = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=index)

    result, empty_bins = resample_segment(frame, "last", 5)

    assert pd.isna(result.loc[pd.Timestamp("2026-01-01 00:05"), "A"])
    assert pd.isna(result.loc[pd.Timestamp("2026-01-01 00:10"), "A"])
    assert result.loc[pd.Timestamp("2026-01-01 00:15"), "A"] == 3.0
    assert empty_bins == 1


def test_resampling_none_preserves_timestamps_and_upsampling_is_rejected():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)

    unchanged, empty_bins = resample_segment(frame, "none", 1)

    pd.testing.assert_frame_equal(unchanged, frame)
    assert empty_bins == 0
    with pytest.raises(ValueError, match="must not be shorter"):
        resample_segment(frame, "mean", 1)


def test_resampling_anchor_is_independent_of_batch_start():
    index = pd.to_datetime(
        ["2026-01-01 00:01", "2026-01-01 00:04", "2026-01-01 00:06"]
    )
    frame = pd.DataFrame({"A": [1.0, 3.0, 10.0]}, index=index)

    full, _ = resample_segment(frame, "mean", 5)
    first_bucket, _ = resample_segment(frame.iloc[:2], "mean", 5)

    assert full.loc[pd.Timestamp("2026-01-01 00:05"), "A"] == first_bucket.iloc[0, 0]


def test_resampling_never_merges_physical_segments_in_the_same_bucket():
    index = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:03", "2026-01-01 00:04"]
    )
    frame = pd.DataFrame({"A": [1.0, 2.0, 100.0, 200.0]}, index=index)
    config = PreprocessingConfig(
        5, 0, 0, 5, resampling_method="mean", filter_method="none"
    )

    with pytest.raises(ValueError, match="duplicate timestamps across segments"):
        preprocess_window(frame, ["A"], config)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", [0.0, 10.0, 100.0]),
        ("trailing_mean", [np.nan, 5.0, 55.0]),
        ("trailing_median", [np.nan, 5.0, 55.0]),
    ],
)
def test_filter_methods_are_causal_and_require_complete_windows(method, expected):
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({"A": [0.0, 10.0, 100.0]}, index=index)

    result = filter_segment(frame, method, 2)

    np.testing.assert_allclose(result["A"], expected, equal_nan=True)
    changed = frame.copy()
    changed.iloc[-1, 0] = 10000.0
    changed_result = filter_segment(changed, method, 2)
    pd.testing.assert_series_equal(result["A"].iloc[:-1], changed_result["A"].iloc[:-1])


def test_state_filters_support_bounds_and_and_semantics():
    index = pd.date_range("2026-01-01", periods=4, freq="5min")
    frame = pd.DataFrame(
        {"LOAD": [70.0, 80.0, 90.0, 100.0], "MODE": [1.0, 1.0, 0.0, 1.0]},
        index=index,
    )

    result = apply_state_filters(
        frame,
        [StateFilter("LOAD", minimum=80.0, maximum=100.0), StateFilter("MODE", minimum=1.0)],
    )

    assert result.index.tolist() == [index[1], index[3]]
    pd.testing.assert_frame_equal(apply_state_filters(frame, []), frame)
    with pytest.raises(ValueError, match="removed all"):
        apply_state_filters(frame, [StateFilter("LOAD", minimum=1000.0)])


def test_preprocess_window_reports_actual_counts_and_breaks_lag_at_state_filter_gap():
    index = pd.date_range("2026-01-01", periods=7, freq="5min")
    frame = pd.DataFrame(
        {"A": np.arange(7, dtype=float), "LOAD": [1, 1, 1, 0, 1, 1, 1]},
        index=index,
    )
    config = PreprocessingConfig(
        5,
        5,
        5,
        5,
        filter_method="none",
        state_filters=(StateFilter("LOAD", minimum=1.0),),
    )

    result = preprocess_window(frame, ["A"], config, include_intermediates=True)

    assert result.dynamic.index.tolist() == [index[1], index[2], index[5], index[6]]
    assert list(result.dynamic.columns) == ["A__lag_000min", "A__lag_005min"]
    assert result.summary.source_row_count == 7
    assert result.summary.resampled_row_count == 7
    assert result.summary.state_filter_input_rows == 7
    assert result.summary.state_filter_output_rows == 6
    assert result.summary.lag_warmup_loss == 2
    assert result.summary.final_dynamic_row_count == 4
    assert result.summary.dynamic_feature_count == 2


def test_preprocessing_summary_keeps_raw_gaps_separate_from_empty_bins():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:20",
            "2026-01-01 00:25",
        ]
    )
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=index)

    result = preprocess_window(
        frame,
        ["A"],
        PreprocessingConfig(5, 5, 0, 5),
        include_intermediates=True,
    )

    assert result.summary.raw_segment_count == 2
    assert result.summary.raw_gap_count == 1
    assert result.summary.empty_bin_count == 0
    assert result.summary.raw_gap_ranges == (
        {"start": index[1].isoformat(), "end": index[2].isoformat()},
    )


def test_preprocessing_config_validates_new_contract_fields():
    with pytest.raises(ValueError, match="unsupported resampling"):
        PreprocessingConfig(resampling_method="bad")
    with pytest.raises(ValueError, match="unsupported filter"):
        PreprocessingConfig(filter_method="bad")
    with pytest.raises(ValueError, match="gap threshold"):
        PreprocessingConfig(gap_threshold_minutes=4)
    PreprocessingConfig(smoothing_window_minutes=0, filter_method="none")


def test_filter_warmup_loss_excludes_empty_bins_when_filter_is_disabled():
    index = pd.to_datetime(
        ["2026-01-01 00:01", "2026-01-01 00:02", "2026-01-01 00:11"]
    )
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
    result = preprocess_window(
        frame,
        ["A"],
        PreprocessingConfig(
            5,
            0,
            0,
            5,
            resampling_method="mean",
            filter_method="none",
            gap_threshold_minutes=10,
        ),
        validate_quality=False,
    )

    assert result.summary.empty_bin_count == 1
    assert result.summary.filter_warmup_loss == 0


def test_filter_warmup_loss_is_structural_per_physical_segment():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
        ]
    )
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    result = preprocess_window(
        frame, ["A"], PreprocessingConfig(5, 10, 0, 5)
    )

    assert result.summary.filter_warmup_loss == 2
    assert result.filter_warmup_mask.tolist() == [True, False, False, True, False]


def test_original_missing_value_is_not_counted_as_filter_warmup():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame({"A": [np.nan, 2.0, 3.0]}, index=index)
    result = preprocess_window(
        frame,
        ["A"],
        PreprocessingConfig(5, 10, 0, 5, filter_method="trailing_mean"),
        validate_quality=False,
    )

    assert result.summary.filter_warmup_loss == 0


def test_lag_summary_separates_structural_warmup_from_missing_context():
    index = pd.date_range("2026-01-01", periods=5, freq="5min")
    normal = preprocess_window(
        pd.DataFrame({"A": range(5)}, index=index),
        ["A"],
        PreprocessingConfig(5, 0, 10, 5, filter_method="none"),
    )
    no_lag = preprocess_window(
        pd.DataFrame({"A": range(5)}, index=index),
        ["A"],
        PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
    )

    assert normal.summary.lag_warmup_loss == 2
    assert normal.summary.lag_context_invalid_loss == 0
    assert normal.summary.final_dynamic_row_count == 3
    assert no_lag.summary.lag_warmup_loss == 0


def test_lag_context_invalid_is_not_structural_warmup_after_empty_bin():
    index = pd.date_range("2026-01-01 00:01", periods=5, freq="1min").append(
        pd.date_range("2026-01-01 00:11", periods=5, freq="1min")
    )
    result = preprocess_window(
        pd.DataFrame({"A": np.arange(10, dtype=float)}, index=index),
        ["A"],
        PreprocessingConfig(
            5,
            0,
            5,
            5,
            resampling_method="mean",
            filter_method="none",
            gap_threshold_minutes=10,
        ),
        validate_quality=False,
    )

    assert result.summary.empty_bin_count == 1
    assert result.summary.lag_warmup_loss == 1
    assert result.summary.lag_context_invalid_loss == 1
    assert result.lag_context_invalid_mask.loc["2026-01-01 00:15"]


def test_resampling_row_reduction_is_not_reported_as_smoothing_or_lag_loss():
    index = pd.date_range("2026-01-01", periods=61, freq="1min")
    result = preprocess_window(
        pd.DataFrame({"A": np.arange(61, dtype=float)}, index=index),
        ["A"],
        PreprocessingConfig(
            5, 0, 0, 5, resampling_method="mean", filter_method="none"
        ),
    )

    assert result.summary.resampling_row_reduction == 48
    assert result.summary.partial_resampling_bin_loss == 0
    assert result.summary.filter_warmup_loss == 0
    assert result.summary.lag_warmup_loss == 0
    assert result.summary.lag_context_invalid_loss == 0
