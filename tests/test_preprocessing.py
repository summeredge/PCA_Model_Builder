import numpy as np
import pandas as pd
import pytest

from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)


def test_dynamic_lags_do_not_cross_physical_time_gaps():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
        ]
    )
    frame = pd.DataFrame({"T1": [0.0, 1.0, 2.0, 10.0, 11.0]}, index=index)
    segments = infer_segment_ids(index, sample_interval_minutes=5)
    config = PreprocessingConfig(
        sample_interval_minutes=5,
        smoothing_window_minutes=5,
        max_lag_minutes=5,
        lag_step_minutes=5,
    )

    dynamic = build_dynamic_matrix(frame, ["T1"], config, segments)

    assert dynamic.index.tolist() == [index[1], index[2], index[4]]
    assert dynamic.loc[index[4], "T1__lag_005min"] == 10.0


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

