import numpy as np
import pandas as pd
import pytest

from pca_model_builder.screening import screen_performance_states


def test_performance_conditions_use_inclusive_and_logic():
    frame = pd.DataFrame(
        {
            "yield": [90.0, 92.0, 94.0, 96.0, 98.0],
            "consumption": [12.0, 11.0, 10.0, 9.0, 8.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="5min"),
    )

    result = screen_performance_states(
        frame,
        [
            {"column": "yield", "minimum": 94},
            {"column": "consumption", "maximum": 10},
        ],
        sample_interval_minutes=5,
    )

    assert result["matched_rows"] == 3
    assert result["match_share"] == pytest.approx(0.6)
    assert [item["matched_rows"] for item in result["conditions"]] == [3, 3]
    assert result["representative_windows"] == [
        {
            "start": "2026-01-01T00:10:00",
            "end": "2026-01-01T00:20:00",
            "count": 3,
        }
    ]
    assert result["engineer_decision_required"] is True


def test_performance_windows_do_not_cross_physical_time_gaps():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
        ]
    )
    frame = pd.DataFrame({"quality": [1.0, 1.0, 1.0, 1.0]}, index=index)

    result = screen_performance_states(
        frame,
        [{"column": "quality", "minimum": 1.0, "maximum": 1.0}],
        sample_interval_minutes=5,
    )

    assert [window["count"] for window in result["representative_windows"]] == [2, 2]


def test_performance_screen_rejects_empty_data():
    frame = pd.DataFrame(
        {"yield": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([]),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        screen_performance_states(
            frame,
            [{"column": "yield", "minimum": 1}],
            sample_interval_minutes=5,
        )


@pytest.mark.parametrize(
    "conditions, message",
    [
        ([], "at least one"),
        ([{"column": "yield"}], "requires a bound"),
        ([{"column": "yield", "minimum": 2, "maximum": 1}], "reversed"),
        (
            [
                {"column": "yield", "minimum": 1},
                {"column": "yield", "maximum": 2},
            ],
            "cannot be repeated",
        ),
    ],
)
def test_performance_screen_rejects_invalid_conditions(conditions, message):
    frame = pd.DataFrame(
        {"yield": [1.0, 2.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="5min"),
    )

    with pytest.raises(ValueError, match=message):
        screen_performance_states(frame, conditions, sample_interval_minutes=5)
