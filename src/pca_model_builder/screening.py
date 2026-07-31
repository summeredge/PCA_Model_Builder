from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def screen_performance_states(
    frame: pd.DataFrame,
    conditions: Sequence[Mapping[str, object]],
    sample_interval_minutes: int,
) -> dict[str, Any]:
    """Apply transparent AND range conditions to identify candidate periods."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("performance data index must be a DatetimeIndex")
    if frame.empty:
        raise ValueError("performance data must not be empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("performance timestamps must be sorted and unique")
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    if not conditions:
        raise ValueError("at least one performance condition is required")

    normalized = [_normalize_condition(condition) for condition in conditions]
    columns = [condition["column"] for condition in normalized]
    if len(columns) != len(set(columns)):
        raise ValueError("performance condition columns cannot be repeated")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing performance columns: {', '.join(missing)}")

    combined = pd.Series(True, index=frame.index)
    summaries = []
    for condition in normalized:
        column = condition["column"]
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"performance column {column} must contain finite numbers")
        matched = pd.Series(True, index=frame.index)
        if condition["minimum"] is not None:
            matched &= numeric >= condition["minimum"]
        if condition["maximum"] is not None:
            matched &= numeric <= condition["maximum"]
        combined &= matched
        summaries.append({**condition, "matched_rows": int(matched.sum())})

    return {
        "total_rows": len(frame),
        "matched_rows": int(combined.sum()),
        "match_share": float(combined.mean()),
        "conditions": summaries,
        "representative_windows": _matching_windows(
            frame.index, combined.to_numpy(), sample_interval_minutes
        ),
        "engineer_decision_required": True,
    }


def _normalize_condition(condition: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(condition, Mapping):
        raise ValueError("each performance condition must be an object")
    column = str(condition.get("column", "")).strip()
    if not column:
        raise ValueError("performance condition requires a column")
    minimum = _optional_float(condition.get("minimum"), column, "minimum")
    maximum = _optional_float(condition.get("maximum"), column, "maximum")
    if minimum is None and maximum is None:
        raise ValueError(f"performance condition for {column} requires a bound")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"performance condition for {column} has reversed bounds")
    return {"column": column, "minimum": minimum, "maximum": maximum}


def _optional_float(value: object, column: str, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"performance {label} for {column} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"performance {label} for {column} must be finite")
    return result


def _matching_windows(
    index: pd.DatetimeIndex,
    matched: np.ndarray,
    sample_interval_minutes: int,
) -> list[dict[str, Any]]:
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    windows: list[dict[str, Any]] = []
    start: int | None = None
    for position in range(len(index) + 1):
        continues = (
            position < len(index)
            and bool(matched[position])
            and (
                start is None
                or index[position] - index[position - 1] == expected
            )
        )
        if continues:
            if start is None:
                start = position
            continue
        if start is not None:
            windows.append(
                {
                    "start": index[start].isoformat(),
                    "end": index[position - 1].isoformat(),
                    "count": position - start,
                }
            )
            start = position if position < len(index) and matched[position] else None
    return sorted(windows, key=lambda item: int(item["count"]), reverse=True)[:10]
