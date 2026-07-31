from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    count: int


@dataclass(frozen=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]
    inferred_interval_minutes: float | None

    @property
    def can_train(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def inspect_data_quality(
    frame: pd.DataFrame,
    timestamp_column: str,
    tag_columns: Sequence[str],
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
    expected_interval_minutes: float | None = None,
) -> QualityReport:
    """Inspect input without modifying it or silently repairing problems."""
    missing_columns = [
        column
        for column in [timestamp_column, *tag_columns]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"missing columns: {', '.join(missing_columns)}")

    issues: list[QualityIssue] = []
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    invalid_timestamp_count = int(timestamps.isna().sum())
    if invalid_timestamp_count:
        issues.append(
            QualityIssue(
                "invalid_timestamp",
                "error",
                "Timestamp values could not be parsed.",
                invalid_timestamp_count,
            )
        )

    valid_timestamps = timestamps.dropna()
    if not valid_timestamps.is_monotonic_increasing:
        issues.append(
            QualityIssue(
                "unsorted_timestamp",
                "error",
                "Timestamps are not in ascending order.",
                1,
            )
        )
    duplicate_count = int(valid_timestamps.duplicated(keep=False).sum())
    if duplicate_count:
        issues.append(
            QualityIssue(
                "duplicate_timestamp",
                "error",
                "Duplicate timestamps require an explicit aggregation decision.",
                duplicate_count,
            )
        )

    unique_sorted = pd.DatetimeIndex(valid_timestamps.unique()).sort_values()
    intervals = pd.Series(unique_sorted).diff().dropna().dt.total_seconds() / 60.0
    inferred_interval = None
    if not intervals.empty:
        inferred_interval = float(intervals.mode().iloc[0])
        sampling_interval = (
            float(expected_interval_minutes)
            if expected_interval_minutes is not None
            else inferred_interval
        )
        interval_values = intervals.to_numpy()
        normal = np.isclose(interval_values, sampling_interval)
        ratios = interval_values / sampling_interval
        physical_gaps = (
            (interval_values > sampling_interval)
            & np.isclose(ratios, np.round(ratios))
        )
        irregular_count = int((~normal & ~physical_gaps).sum())
        if irregular_count:
            issues.append(
                QualityIssue(
                    "irregular_sampling",
                    "error",
                    "Sampling intervals are below the configured period or off its grid.",
                    irregular_count,
                )
            )
        if physical_gaps.any():
            issues.append(
                QualityIssue(
                    "physical_time_gap",
                    "warning",
                    "Physical time gaps will start new preprocessing segments.",
                    int(physical_gaps.sum()),
                )
            )

    engineering_ranges = engineering_ranges or {}
    for tag in tag_columns:
        numeric = pd.to_numeric(frame[tag], errors="coerce")
        non_numeric_count = int((frame[tag].notna() & numeric.isna()).sum())
        if non_numeric_count:
            issues.append(
                QualityIssue(
                    "non_numeric_value",
                    "error",
                    f"{tag} contains non-numeric values.",
                    non_numeric_count,
                )
            )

        missing_count = int(numeric.isna().sum())
        if missing_count:
            issues.append(
                QualityIssue(
                    "missing_value",
                    "error",
                    f"{tag} contains missing values.",
                    missing_count,
                )
            )

        finite = numeric[np.isfinite(numeric)]
        non_finite_count = int((numeric.notna() & ~np.isfinite(numeric)).sum())
        if non_finite_count:
            issues.append(
                QualityIssue(
                    "non_finite_value",
                    "error",
                    f"{tag} contains infinite values.",
                    non_finite_count,
                )
            )
        if not finite.empty and finite.nunique() <= 1:
            issues.append(
                QualityIssue(
                    "constant_tag",
                    "error",
                    f"{tag} is constant and cannot be standardized.",
                    len(finite),
                )
            )

        if tag in engineering_ranges:
            lower, upper = engineering_ranges[tag]
            outside_count = int(((numeric < lower) | (numeric > upper)).sum())
            if outside_count:
                issues.append(
                    QualityIssue(
                        "engineering_range",
                        "error",
                        f"{tag} contains values outside its engineering range.",
                        outside_count,
                    )
                )

    return QualityReport(tuple(issues), inferred_interval)
