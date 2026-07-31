from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .quality import QualityIssue, inspect_data_quality


def profile_tag(
    series: pd.Series,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric = series.notna() & numeric.isna()
    finite_mask = numeric.notna() & np.isfinite(numeric)
    finite = numeric[finite_mask].astype(float)
    profile: dict[str, Any] = {
        "sample_count": int(len(series)),
        "valid_count": int(len(finite)),
        "missing_count": int(numeric.isna().sum()),
        "missing_rate": float(numeric.isna().mean()) if len(series) else 0.0,
        "non_numeric_count": int(non_numeric.sum()),
        "non_finite_count": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
        "unique_count": int(finite.nunique()),
        "minimum": _finite_stat(finite, "min"),
        "maximum": _finite_stat(finite, "max"),
        "mean": _finite_stat(finite, "mean"),
        "median": _finite_stat(finite, "median"),
        "standard_deviation": (
            float(finite.std(ddof=0)) if not finite.empty else None
        ),
        "p01": _quantile(finite, 0.01),
        "p05": _quantile(finite, 0.05),
        "p95": _quantile(finite, 0.95),
        "p99": _quantile(finite, 0.99),
    }
    for prefix in ("engineering", "normal", "alarm"):
        lower = config.get(f"{prefix}_min")
        upper = config.get(f"{prefix}_max")
        profile[f"{prefix}_range_outside_count"] = (
            int(((finite < float(lower)) | (finite > float(upper))).sum())
            if lower is not None and upper is not None
            else None
        )
    return profile


def model_quality_payload(
    full_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    timestamp_column: str,
    tags: Sequence[str],
    tag_configs: Mapping[str, Mapping[str, Any]],
    expected_interval_minutes: float,
) -> dict[str, Any]:
    engineering_ranges = {
        tag: (
            float(tag_configs[tag]["engineering_min"]),
            float(tag_configs[tag]["engineering_max"]),
        )
        for tag in tags
        if tag_configs.get(tag, {}).get("engineering_min") is not None
        and tag_configs.get(tag, {}).get("engineering_max") is not None
    }
    report = inspect_data_quality(
        reference_frame,
        timestamp_column,
        tags,
        engineering_ranges=engineering_ranges,
        expected_interval_minutes=expected_interval_minutes,
    )
    issues_by_tag: dict[str, list[QualityIssue]] = {tag: [] for tag in tags}
    time_issues: list[dict[str, Any]] = []
    for issue in report.issues:
        if issue.tag in issues_by_tag:
            issues_by_tag[issue.tag].append(issue)
        else:
            time_issues.append(asdict(issue))

    tag_results: list[dict[str, Any]] = []
    summary = {"usable": 0, "review": 0, "blocking": 0}
    for tag in tags:
        full_profile = profile_tag(full_frame[tag], tag_configs.get(tag))
        reference_profile = profile_tag(reference_frame[tag], tag_configs.get(tag))
        issues = issues_by_tag[tag]
        discrete_limit = min(
            10, max(2, int(reference_profile["valid_count"] * 0.01))
        )
        if (
            tag_configs.get(tag, {}).get("role", "continuous_input")
            == "continuous_input"
            and 1 < reference_profile["unique_count"] <= discrete_limit
        ):
            issues.append(
                QualityIssue(
                    "suspected_discrete_state",
                    "warning",
                    f"{tag}：唯一值较少，疑似离散状态量，请确认变量角色。",
                    reference_profile["valid_count"],
                    tag,
                    {
                        "unique_count": reference_profile["unique_count"],
                        "threshold": discrete_limit,
                    },
                )
            )
        for issue in issues:
            if issue.code == "constant_tag":
                issue.details["constant_in_full_data"] = (
                    full_profile["unique_count"] == 1
                )
        if any(issue.severity == "error" for issue in issues):
            status = "blocking"
        elif issues:
            status = "review"
        else:
            status = "usable"
        summary[status] += 1
        tag_results.append(
            {
                "tag": tag,
                "status": status,
                "role": tag_configs.get(tag, {}).get("role", "continuous_input"),
                "full": full_profile,
                "reference": reference_profile,
                "issues": [asdict(issue) for issue in issues],
                "suggested_action": (
                    "exclude_or_adjust_reference"
                    if any(issue.code == "constant_tag" for issue in issues)
                    else "review" if issues else "use"
                ),
            }
        )
    return {
        "summary": summary,
        "tags": tag_results,
        "time_issues": time_issues,
        "can_train": report.can_train,
    }


def constant_exclusion_record(tag_result: Mapping[str, Any]) -> dict[str, Any]:
    issue = next(
        (
            item
            for item in tag_result.get("issues", [])
            if item.get("code") == "constant_tag"
        ),
        None,
    )
    if issue is None:
        raise ValueError("Tag不是参考期精确常量")
    details = issue["details"]
    return {
        "tag": tag_result["tag"],
        "reason": "constant_in_reference_window",
        "sample_count": int(details["valid_count"]),
        "unique_count": 1,
        "constant_value": float(details["constant_value"]),
    }


def _finite_stat(series: pd.Series, method: str) -> float | None:
    if series.empty:
        return None
    return float(getattr(series, method)())


def _quantile(series: pd.Series, value: float) -> float | None:
    return None if series.empty else float(series.quantile(value))
