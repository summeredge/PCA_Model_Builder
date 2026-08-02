import numpy as np
import pandas as pd
import pytest

from pca_model_builder.preprocessing import PreprocessingConfig, build_dynamic_matrix
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
            "B": np.arange(periods, dtype=float) * 2.0 + 1.0,
            "C": np.arange(periods, dtype=float) * -0.5,
        }
    )


def test_training_builds_windows_independently_without_lag_or_smoothing_leakage():
    frame = _frame()
    frame.loc[:19, ["A", "B", "C"]] = 10_000.0
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows(
        (frame.time.iloc[0], frame.time.iloc[19], True),
        (frame.time.iloc[40], frame.time.iloc[79], True),
    )

    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    second_window = result.dynamic.loc[result.dynamic.index >= frame.time.iloc[40]]

    assert not second_window.empty
    assert second_window.to_numpy().max() < 200.0
    assert [summary["effective_samples"] for summary in result.window_summaries] == [17, 37]


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
    np.testing.assert_allclose(result.dynamic.mean().to_numpy(), expected.mean().to_numpy())


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
