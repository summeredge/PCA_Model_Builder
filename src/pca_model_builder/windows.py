from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


_WINDOW_FIELDS = {"id", "start", "end", "source", "source_ref", "enabled", "comment"}
_EXCLUDED_WINDOW_FIELDS = {"id", "start", "end", "source", "comment"}


def normalize_training_windows(
    value: object, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("training_windows必须是非空列表")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _WINDOW_FIELDS:
            raise ValueError("training_windows窗口字段无效")
        window_id = item["id"]
        if not isinstance(window_id, str) or not window_id.strip() or window_id in seen:
            raise ValueError("training_windows窗口ID无效或重复")
        seen.add(window_id)
        start, end = _window_bounds(item["start"], item["end"])
        source = item["source"]
        source_ref = item["source_ref"]
        enabled = item["enabled"]
        comment = item["comment"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("training_windows来源无效")
        if source_ref is not None and (not isinstance(source_ref, str) or not source_ref.strip()):
            raise ValueError("training_windows来源引用无效")
        if not isinstance(enabled, bool) or not isinstance(comment, str):
            raise ValueError("training_windows启用状态或备注无效")
        normalized.append(
            {
                "id": window_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "source": source,
                "source_ref": source_ref,
                "enabled": enabled,
                "comment": comment,
            }
        )
    ensure_non_overlapping_enabled_windows(normalized)
    return normalized


def legacy_training_windows_to_canonical(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("model package training_windows must be a non-empty list")
    windows = []
    for position, item in enumerate(value, start=1):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("model package training window must contain start and end")
        windows.append(
            {
                "id": f"legacy-window-{position:03d}",
                "start": item[0],
                "end": item[1],
                "source": "legacy",
                "source_ref": None,
                "enabled": True,
                "comment": "",
            }
        )
    return normalize_training_windows(windows)


def legacy_single_window_to_training_windows(start: object, end: object) -> list[dict[str, Any]]:
    return normalize_training_windows(
        [
            {
                "id": "legacy-window-001",
                "start": start,
                "end": end,
                "source": "legacy",
                "source_ref": None,
                "enabled": True,
                "comment": "",
            }
        ]
    )


def add_training_window(windows: object, window: Mapping[str, Any]) -> list[dict[str, Any]]:
    return normalize_training_windows(
        [*normalize_training_windows(windows, allow_empty=True), dict(window)],
        allow_empty=True,
    )


def update_training_window(windows: object, window_id: str, changes: Mapping[str, Any]) -> list[dict[str, Any]]:
    existing = normalize_training_windows(windows, allow_empty=True)
    if window_id not in {window["id"] for window in existing}:
        raise ValueError("training_windows窗口不存在")
    return normalize_training_windows(
        [{**window, **dict(changes)} if window["id"] == window_id else window for window in existing],
        allow_empty=True,
    )


def remove_training_window(windows: object, window_id: str) -> list[dict[str, Any]]:
    existing = normalize_training_windows(windows, allow_empty=True)
    result = [window for window in existing if window["id"] != window_id]
    if len(result) == len(existing):
        raise ValueError("training_windows窗口不存在")
    return normalize_training_windows(result, allow_empty=True)


def set_enabled_training_window(windows: object, window_id: str, enabled: bool) -> list[dict[str, Any]]:
    return update_training_window(windows, window_id, {"enabled": enabled})


def merge_excluded_windows(value: object) -> list[dict[str, Any]]:
    """Validate, sort, and merge overlapping or touching exclusion windows."""
    if not isinstance(value, list):
        raise ValueError("excluded_windows必须是列表")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _EXCLUDED_WINDOW_FIELDS:
            raise ValueError("excluded_windows窗口字段无效")
        window_id = item["id"]
        if not isinstance(window_id, str) or not window_id.strip() or window_id in seen:
            raise ValueError("excluded_windows窗口ID无效或重复")
        source = item["source"]
        comment = item["comment"]
        if not isinstance(source, str) or not source.strip() or not isinstance(comment, str):
            raise ValueError("excluded_windows来源或备注无效")
        seen.add(window_id)
        start, end = _window_bounds(item["start"], item["end"])
        normalized.append(
            {
                "id": window_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "source": source,
                "comment": comment,
            }
        )

    normalized.sort(key=lambda window: (window["start"], window["end"], window["id"]))
    merged: list[dict[str, Any]] = []
    for window in normalized:
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        _, previous_end = _window_bounds(previous["start"], previous["end"])
        start, end = _window_bounds(window["start"], window["end"])
        if start <= previous_end:
            if end > previous_end:
                previous["end"] = end.isoformat()
            continue
        merged.append(window)
    return merged


def subtract_excluded_windows(
    candidate: Mapping[str, Any], excluded_windows: object
) -> list[dict[str, str]]:
    """Return the candidate ranges that remain after exclusion-window cuts.

    Boundaries are retained exactly as selected; this function deliberately does
    not infer or apply a sampling interval.
    """
    start, end = _window_bounds(candidate.get("start"), candidate.get("end"))
    exclusions = merge_excluded_windows(excluded_windows)
    if start == end:
        if any(
            _window_bounds(window["start"], window["end"])[0] <= start
            <= _window_bounds(window["start"], window["end"])[1]
            for window in exclusions
        ):
            return []
        return [{"start": start.isoformat(), "end": end.isoformat()}]
    remaining: list[dict[str, str]] = []
    cursor = start
    for excluded in exclusions:
        excluded_start, excluded_end = _window_bounds(
            excluded["start"], excluded["end"]
        )
        if excluded_end < cursor or excluded_start > end:
            continue
        if excluded_start > cursor:
            remaining.append(
                {"start": cursor.isoformat(), "end": min(excluded_start, end).isoformat()}
            )
        cursor = max(cursor, excluded_end)
        if cursor >= end:
            break
    if cursor < end:
        remaining.append({"start": cursor.isoformat(), "end": end.isoformat()})
    return remaining


def ensure_non_overlapping_enabled_windows(windows: Sequence[Mapping[str, Any]]) -> None:
    enabled = [window for window in windows if window["enabled"]]
    for position, window in enumerate(enabled):
        start, end = _window_bounds(window["start"], window["end"])
        for other in enabled[position + 1 :]:
            other_start, other_end = _window_bounds(other["start"], other["end"])
            if start <= other_end and other_start <= end:
                raise ValueError("启用的training_windows不能重叠")


def summarize_training_windows(
    windows: object,
    timestamps: pd.Series | None = None,
    sample_interval_minutes: int | None = None,
) -> list[dict[str, Any]]:
    result = []
    for window in normalize_training_windows(windows, allow_empty=True):
        start, end = _window_bounds(window["start"], window["end"])
        summary = {**window, "duration_minutes": int((end - start).total_seconds() // 60)}
        if timestamps is not None:
            matched = timestamps.between(start, end, inclusive="both")
            summary["raw_samples"] = int(matched.sum())
            summary["effective_samples"] = summary["raw_samples"]
            if sample_interval_minutes:
                summary["expected_samples"] = int((end - start) / pd.Timedelta(minutes=sample_interval_minutes)) + 1
                summary["quality_status"] = (
                    "ready"
                    if summary["raw_samples"] == summary["expected_samples"]
                    else "sampling_gap"
                )
        result.append(summary)
    return result


def _window_bounds(start_value: object, end_value: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(start_value, str) or not start_value.strip() or not isinstance(end_value, str) or not end_value.strip():
        raise ValueError("training_windows起止时间无效")
    try:
        start, end = pd.Timestamp(start_value), pd.Timestamp(end_value)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("training_windows起止时间无效") from error
    return start, end
