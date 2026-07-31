import pandas as pd
import pytest

from pca_model_builder.quality import inspect_data_quality
from pca_model_builder.tag_profile import model_quality_payload, profile_tag


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
    assert next(
        issue for issue in report.issues if issue.code == "engineering_range"
    ).tag == "T1"


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


def test_quality_report_allows_only_ten_minute_physical_gaps():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:10",
                    "2026-01-01 00:20",
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
        ("physical_time_gap", "warning", 3)
    ]


def test_quality_report_allows_only_mixed_physical_gaps():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:10",
                    "2026-01-01 00:25",
                    "2026-01-01 00:45",
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
        ("physical_time_gap", "warning", 3)
    ]


@pytest.mark.parametrize(
    ("timestamps", "physical_gap_count"),
    [
        (
            ["2026-01-01 00:00", "2026-01-01 00:07", "2026-01-01 00:14"],
            0,
        ),
        (
            [
                "2026-01-01 00:00",
                "2026-01-01 00:10",
                "2026-01-01 00:22",
                "2026-01-01 00:32",
            ],
            2,
        ),
    ],
)
def test_quality_report_rejects_off_grid_intervals(
    timestamps, physical_gap_count
):
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps),
            "T1": list(range(len(timestamps))),
        }
    )

    report = inspect_data_quality(
        frame, "time", ["T1"], expected_interval_minutes=5
    )

    issues = {issue.code: issue for issue in report.issues}
    assert not report.can_train
    assert issues["irregular_sampling"].severity == "error"
    assert "sampling_interval_mismatch" not in issues
    if physical_gap_count:
        assert issues["physical_time_gap"].count == physical_gap_count
    else:
        assert "physical_time_gap" not in issues


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


def test_quality_report_blocks_unsorted_non_finite_and_reports_physical_gaps():
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
        "physical_time_gap",
        "non_finite_value",
    }


def test_constant_issue_contains_tag_and_reference_details():
    full = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="5min"),
            "A": [50.0, 50.0, 50.0, 50.0, 51.0, 52.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    reference = full.iloc[:4]
    result = model_quality_payload(
        full,
        reference,
        "time",
        ["A", "B"],
        {"A": {}, "B": {}},
        5,
    )
    tag = next(item for item in result["tags"] if item["tag"] == "A")
    issue = tag["issues"][0]

    assert result["summary"] == {"usable": 1, "review": 0, "blocking": 1}
    assert issue["code"] == "constant_tag"
    assert issue["tag"] == "A"
    assert issue["details"]["unique_count"] == 1
    assert issue["details"]["constant_value"] == 50.0
    assert issue["details"]["standard_deviation"] == 0.0
    assert issue["details"]["constant_in_full_data"] is False
    assert "A：" in issue["message"]
    assert "constant_tag(4)" not in issue["message"]


def test_near_constant_is_warning_and_time_issue_has_no_tag():
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:17"]
            ),
            "A": [10.0, 10.000001, 10.000002],
        }
    )
    report = inspect_data_quality(
        frame, "time", ["A"], expected_interval_minutes=5
    )
    issues = {issue.code: issue for issue in report.issues}

    assert issues["near_constant_tag"].severity == "warning"
    assert issues["near_constant_tag"].tag == "A"
    assert issues["irregular_sampling"].tag is None


def test_low_cardinality_continuous_tag_requires_role_review():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=300, freq="5min"),
            "MODE": [0.0, 1.0] * 150,
        }
    )
    result = model_quality_payload(
        frame,
        frame,
        "time",
        ["MODE"],
        {"MODE": {"role": "continuous_input"}},
        5,
    )

    tag = result["tags"][0]
    assert tag["status"] == "review"
    assert tag["issues"][0]["code"] == "suspected_discrete_state"


def test_multiple_constant_tags_are_reported_individually():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "A": [1.0] * 4,
            "B": [2.0] * 4,
        }
    )
    report = inspect_data_quality(
        frame, "time", ["A", "B"], expected_interval_minutes=5
    )

    constants = [issue for issue in report.issues if issue.code == "constant_tag"]
    assert [issue.tag for issue in constants] == ["A", "B"]


def test_tag_profile_separates_missing_non_numeric_and_non_finite_values():
    profile = profile_tag(pd.Series([1.0, None, "BAD", float("inf"), 2.0]))

    assert profile["sample_count"] == 5
    assert profile["valid_count"] == 2
    assert profile["missing_count"] == 1
    assert profile["missing_rate"] == pytest.approx(0.2)
    assert profile["non_numeric_count"] == 1
    assert profile["non_finite_count"] == 1
    assert (
        profile["valid_count"]
        + profile["missing_count"]
        + profile["non_numeric_count"]
        + profile["non_finite_count"]
        == profile["sample_count"]
    )

    report = inspect_data_quality(
        pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=5, freq="5min"),
                "T1": [1.0, None, "BAD", float("inf"), 2.0],
            }
        ),
        "time",
        ["T1"],
        expected_interval_minutes=5,
    )
    issues = {issue.code: issue for issue in report.issues}
    assert issues["missing_value"].count == 1
    assert issues["non_numeric_value"].count == 1
    assert issues["non_finite_value"].count == 1
