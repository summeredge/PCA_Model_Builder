import numpy as np
import pandas as pd
import pytest

import pca_model_builder.trend as trend_module
from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.trend import (
    downsample_trend,
    prepare_trend_frame,
    trend_axis_limits,
    trend_payload_data,
)


def test_trend_smoothing_is_trailing_and_does_not_cross_gap():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
        ]
    )
    frame = pd.DataFrame({"A": [0.0, 10.0, 20.0, 100.0, 120.0]}, index=index)
    raw, resampled, smoothed, segments, resampled_segments = prepare_trend_frame(
        frame, ["A"], PreprocessingConfig(5, 10, 0, 5)
    )

    assert raw.equals(frame)
    assert resampled.equals(frame)
    assert smoothed.loc[index[1], "A"] == 5.0
    assert smoothed.loc[index[2], "A"] == 15.0
    assert pd.isna(smoothed.loc[index[3], "A"])
    assert smoothed.loc[index[4], "A"] == 110.0
    assert segments.tolist() == [0, 0, 0, 1, 1]
    assert resampled_segments.tolist() == [0, 0, 0, 1, 1]


def test_trend_preview_reuses_resampling_and_filtering_core():
    index = pd.date_range("2026-01-01", periods=11, freq="1min")
    frame = pd.DataFrame({"A": np.arange(11, dtype=float)}, index=index)
    config = PreprocessingConfig(
        5,
        10,
        0,
        5,
        resampling_method="mean",
        filter_method="trailing_median",
    )

    raw, resampled, filtered, _, _ = prepare_trend_frame(frame, ["A"], config)

    assert raw.index.tolist() == index.tolist()
    assert raw["A"].tolist() == np.arange(11, dtype=float).tolist()
    assert resampled.index.tolist() == [index[0], index[5], index[10]]
    assert resampled["A"].tolist() == [0.0, 3.0, 8.0]
    assert pd.isna(filtered.iloc[0, 0])
    assert filtered.iloc[1, 0] == 1.5
    assert filtered.iloc[2, 0] == 5.5


def test_trend_payload_returns_statistics_histogram_ranges_and_preserves_input():
    index = pd.date_range("2026-01-01", periods=20, freq="5min")
    frame = pd.DataFrame({"A": np.arange(20, dtype=float)}, index=index)
    original = frame.copy(deep=True)
    result = trend_payload_data(
        frame,
        ["A"],
        PreprocessingConfig(5, 10, 0, 5),
        index[2],
        index[12],
        "both",
        {
            "A": {
                "engineering_min": 0,
                "engineering_max": 30,
                "normal_min": 2,
                "normal_max": 10,
                "alarm_min": -1,
                "alarm_max": 20,
            }
        },
        index[0],
        index[9],
    )

    pd.testing.assert_frame_equal(frame, original)
    assert result["statistics"]["A"]["full"]["sample_count"] == 20
    assert result["statistics"]["A"]["current"]["sample_count"] == 11
    assert sum(result["histogram"]["counts"]) == 11
    assert sum(result["histograms"]["reference"]["counts"]) == 10
    assert result["ranges"]["A"]["normal_max"] == 10
    assert {"A__raw", "A__smoothed"}.issubset(result["rows"][0])


def test_trend_downsampling_preserves_first_last_spike_and_gap_boundaries():
    first = pd.date_range("2026-01-01", periods=800, freq="5min")
    second = pd.date_range(first[-1] + pd.Timedelta(minutes=20), periods=800, freq="5min")
    index = first.append(second)
    values = np.zeros(len(index))
    values[713] = 100.0
    frame = pd.DataFrame({"A": values}, index=index)
    raw, _, smoothed, segments, _ = prepare_trend_frame(
        frame, ["A"], PreprocessingConfig(5, 5, 0, 5)
    )
    positions = downsample_trend(raw, smoothed, segments, limit=120)

    assert len(positions) <= 120
    assert positions[0] == 0
    assert positions[-1] == len(index) - 1
    assert 713 in positions
    assert 799 in positions and 800 in positions


def test_trend_rejects_more_than_eight_tags():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    frame = pd.DataFrame(
        {f"T{index}": [1.0, 2.0, 3.0] for index in range(9)},
        index=index,
    )
    with pytest.raises(ValueError, match="最多选择8"):
        prepare_trend_frame(frame, list(frame.columns), PreprocessingConfig(5, 5, 0, 5))


def test_trend_preserves_missing_point_without_modifying_source():
    index = pd.date_range("2026-01-01", periods=5, freq="5min")
    frame = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=index)
    original = frame.copy(deep=True)
    result = trend_payload_data(
        frame,
        ["A"],
        PreprocessingConfig(5, 5, 0, 5),
        index[0],
        index[-1],
        "raw",
        {"A": {}},
    )

    assert result["rows"][2]["A__raw"] is None
    pd.testing.assert_frame_equal(frame, original)


def test_trend_payload_separates_raw_resampled_and_filtered_stages():
    index = pd.date_range("2026-01-01", periods=11, freq="1min")
    frame = pd.DataFrame({"A": np.arange(11, dtype=float)}, index=index)
    result = trend_payload_data(
        frame,
        ["A"],
        PreprocessingConfig(
            5, 10, 0, 5, resampling_method="mean", filter_method="trailing_mean"
        ),
        index[0],
        index[-1],
        "both",
        {"A": {}},
    )

    assert [row["timestamp"] for row in result["stage_rows"]["raw"]] == [
        timestamp.isoformat() for timestamp in index
    ]
    assert [row["timestamp"] for row in result["stage_rows"]["resampled"]] == [
        index[0].isoformat(), index[5].isoformat(), index[10].isoformat()
    ]
    assert result["stage_rows"]["filtered"][0]["A"] is None
    assert result["series_stage"] == {
        "raw": "raw",
        "smoothed": "filtered",
        "resampling_applied": True,
    }


def test_stage_rows_are_display_limited_without_changing_statistics(monkeypatch):
    monkeypatch.setattr(trend_module, "MAX_TREND_POINTS", 12)
    first = pd.date_range("2026-01-01", periods=20, freq="5min")
    second = pd.date_range(first[-1] + pd.Timedelta(minutes=20), periods=20, freq="5min")
    index = first.append(second)
    values = np.zeros(len(index))
    values[10] = 100.0
    values[25] = -100.0
    values[30] = np.nan
    result = trend_payload_data(
        pd.DataFrame({"A": values}, index=index),
        ["A"],
        PreprocessingConfig(5, 5, 0, 5),
        index[0],
        index[-1],
        "both",
        {"A": {}},
    )

    for stage in ("raw", "resampled", "filtered"):
        rows = result["stage_rows"][stage]
        assert len(rows) <= 12
        assert result["stage_counts"][stage]["analysis_rows"] == 40
        assert result["stage_counts"][stage]["display_rows"] == len(rows)
        assert rows[0]["timestamp"] == index[0].isoformat()
        assert rows[-1]["timestamp"] == index[-1].isoformat()
    raw_rows = result["stage_rows"]["raw"]
    assert any(row["A"] == 100.0 for row in raw_rows)
    assert any(row["A"] == -100.0 for row in raw_rows)
    assert any(row["A"] is None for row in raw_rows)
    assert any(row["physical_gap_start"] for row in raw_rows)
    assert result["statistics"]["A"]["current"]["sample_count"] == 40


def test_physical_gap_marker_does_not_mark_series_or_display_start():
    first = pd.date_range("2026-01-01", periods=4, freq="5min")
    second = pd.date_range(first[-1] + pd.Timedelta(minutes=20), periods=4, freq="5min")
    index = first.append(second)
    result = trend_payload_data(
        pd.DataFrame({"A": np.arange(8, dtype=float)}, index=index),
        ["A"],
        PreprocessingConfig(5, 10, 0, 5),
        index[2],
        index[-1],
        "both",
        {"A": {}},
    )

    assert result["rows"][0]["physical_gap_start"] is False
    assert result["stage_rows"]["raw"][0]["physical_gap_start"] is False
    assert any(row["physical_gap_start"] for row in result["rows"])
    assert result["stage_rows"]["filtered"][0]["physical_gap_start"] is False


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([10000.0, 10020.0], "positive"),
        ([-20.0, -10.0], "negative"),
        ([-2.0, 3.0], "cross_zero"),
        ([100.0, 100.0], "constant"),
        ([None, np.nan, np.inf], "empty"),
    ],
)
def test_trend_axis_limits_scale_to_visible_values(values, expected):
    minimum, maximum = trend_axis_limits(values)

    if expected == "positive":
        assert 0 < minimum < 10000 < 10020 < maximum
    elif expected == "negative":
        assert minimum < -20 < -10 < maximum < 0
    elif expected == "cross_zero":
        assert minimum < -2 < 0 < 3 < maximum
    elif expected == "constant":
        assert 0 < minimum < 100 < maximum
    else:
        assert (minimum, maximum) == (0.0, 1.0)


def test_trend_axis_limits_include_configured_range_lines():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    result = trend_payload_data(
        pd.DataFrame({"A": [10000.0, 10010.0, 10020.0]}, index=index),
        ["A"],
        PreprocessingConfig(5, 5, 0, 5),
        index[0],
        index[-1],
        "raw",
        {"A": {"engineering_min": 9000.0, "engineering_max": 11000.0}},
    )

    limits = result["axis_limits"]["A"]
    assert limits["minimum"] < 9000.0
    assert limits["maximum"] > 11000.0
