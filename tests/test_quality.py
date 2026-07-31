import pandas as pd

from pca_model_builder.quality import inspect_data_quality


def test_quality_report_blocks_duplicate_timestamps_and_invalid_values():
    frame = pd.DataFrame(
        {
            "time": ["2026-01-01 00:00", "2026-01-01 00:00", "bad"],
            "T1": [1.0, 2.0, 3.0],
        }
    )

    report = inspect_data_quality(frame, "time", ["T1"])

    assert not report.can_train
    assert {issue.code for issue in report.issues} == {
        "invalid_timestamp",
        "duplicate_timestamp",
    }


def test_quality_report_detects_irregular_sampling_missing_and_range_violation():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:12"]
            ),
            "T1": [1.0, None, 20.0],
        }
    )

    report = inspect_data_quality(
        frame,
        "time",
        ["T1"],
        engineering_ranges={"T1": (0.0, 10.0)},
    )

    assert report.inferred_interval_minutes == 5.0
    assert {issue.code for issue in report.issues} == {
        "irregular_sampling",
        "missing_value",
        "engineering_range",
    }


def test_quality_report_allows_physical_gap_on_sampling_grid():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:05",
                    "2026-01-01 00:25",
                    "2026-01-01 00:30",
                ]
            ),
            "T1": [1.0, 2.0, 3.0, 4.0],
        }
    )

    report = inspect_data_quality(
        frame, "time", ["T1"], expected_interval_minutes=5
    )

    assert report.can_train
    assert [(issue.code, issue.severity, issue.count) for issue in report.issues] == [
        ("physical_time_gap", "warning", 1)
    ]


def test_quality_report_rejects_interval_shorter_than_sampling_period():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:07"]
            ),
            "T1": [1.0, 2.0, 3.0],
        }
    )

    report = inspect_data_quality(
        frame, "time", ["T1"], expected_interval_minutes=5
    )

    assert not report.can_train
    assert {issue.code for issue in report.issues} == {"irregular_sampling"}


def test_quality_report_blocks_unsorted_non_finite_and_interval_mismatch():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:10", "2026-01-01 00:00", "2026-01-01 00:05"]
            ),
            "T1": [1.0, float("inf"), 2.0],
        }
    )

    report = inspect_data_quality(
        frame,
        "time",
        ["T1"],
        expected_interval_minutes=1,
    )

    assert not report.can_train
    assert {issue.code for issue in report.issues} == {
        "unsorted_timestamp",
        "sampling_interval_mismatch",
        "non_finite_value",
    }
