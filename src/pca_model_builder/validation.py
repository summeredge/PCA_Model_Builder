from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .contribution import contribution_event_records, exceedance_contribution_tables
from .dpca import DPCAModel
from .preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)


TimeWindow = tuple[pd.Timestamp, pd.Timestamp]
_VALIDATION_WINDOW_FIELDS = {"id", "type", "start", "end", "enabled", "comment"}
_VALIDATION_TYPES = {"normal_validation", "known_abnormal"}
_ENGINEER_DECISIONS = {"passed", "insufficient", "failed"}


def normalize_validation_windows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("validation_windows必须是非空列表")
    windows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _VALIDATION_WINDOW_FIELDS:
            raise ValueError("validation_windows窗口字段无效")
        identifier = item["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("validation_windows窗口ID无效或重复")
        window_type = item["type"]
        if window_type not in _VALIDATION_TYPES:
            raise ValueError("validation_windows类型无效")
        if not isinstance(item["enabled"], bool) or not isinstance(item["comment"], str):
            raise ValueError("validation_windows启用状态或备注无效")
        start, end = _validation_bounds(item["start"], item["end"])
        identifiers.add(identifier)
        windows.append(
            {
                "id": identifier,
                "type": window_type,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "enabled": item["enabled"],
                "comment": item["comment"],
            }
        )
    enabled = [window for window in windows if window["enabled"]]
    ensure_disjoint_windows([], _window_bounds(enabled))
    return windows


def legacy_validation_window(start: object, end: object) -> list[dict[str, Any]]:
    return normalize_validation_windows(
        [
            {
                "id": "legacy-validation-001",
                "type": "normal_validation",
                "start": start,
                "end": end,
                "enabled": True,
                "comment": "",
            }
        ]
    )


def validation_windows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "validation_windows" in payload:
        return normalize_validation_windows(payload["validation_windows"])
    return legacy_validation_window(payload.get("validation_start"), payload.get("validation_end"))


def ensure_disjoint_windows(
    training_windows: Sequence[TimeWindow],
    validation_windows: Sequence[TimeWindow],
) -> None:
    for position, (validation_start, validation_end) in enumerate(validation_windows):
        if validation_start > validation_end:
            raise ValueError("validation window start must not follow its end")
        for other_start, other_end in validation_windows[position + 1 :]:
            if other_start > other_end:
                raise ValueError("validation window start must not follow its end")
            if max(validation_start, other_start) <= min(validation_end, other_end):
                raise ValueError("validation windows overlap")
    for train_start, train_end in training_windows:
        if train_start > train_end:
            raise ValueError("training window start must not follow its end")
        for validation_start, validation_end in validation_windows:
            if max(train_start, validation_start) <= min(train_end, validation_end):
                raise ValueError("training and validation windows overlap")


def validation_context_start(
    validation_start: pd.Timestamp,
    config: PreprocessingConfig,
) -> pd.Timestamp:
    warmup_minutes = (
        config.max_lag_minutes
        + config.smoothing_window_minutes
        - config.sample_interval_minutes
    )
    return validation_start - pd.Timedelta(minutes=warmup_minutes)


def build_validation_matrix(
    indexed_frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build validation features with pre-window history, then score from start."""
    context_start = validation_context_start(validation_start, config)
    context = indexed_frame.loc[context_start:validation_end]
    dynamic = build_dynamic_matrix(
        context,
        tag_columns,
        config,
        infer_segment_ids(context.index, config.sample_interval_minutes),
    )
    scoring = dynamic.loc[validation_start:validation_end]
    if scoring.empty or scoring.index[0] != validation_start:
        raise ValueError(
            "validation context is insufficient to score from the requested start"
        )
    if scoring.index[-1] != validation_end:
        raise ValueError("validation data do not cover the requested end")
    return scoring


def validate_model_windows(
    model: DPCAModel,
    indexed_frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    training_windows: Sequence[TimeWindow],
    validation_windows: object,
    tag_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score enabled typed validation windows without fitting or changing a model."""
    windows = normalize_validation_windows(validation_windows)
    enabled = [window for window in windows if window["enabled"]]
    if not enabled:
        raise ValueError("至少需要一个启用的validation_windows窗口")
    ensure_disjoint_windows(training_windows, _window_bounds(enabled))

    records: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    contribution_records: list[dict[str, Any]] = []
    for window in windows:
        if not window["enabled"]:
            records.append({**window, "status": "disabled", "scored_rows": 0})
            continue
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        dynamic = build_validation_matrix(indexed_frame, tag_columns, config, start, end)
        scores = model.score(dynamic)
        events = contribution_event_records(
            exceedance_contribution_tables(
                model,
                dynamic,
                scores,
                sample_interval_minutes=config.sample_interval_minutes,
            ),
            tag_configs,
        )
        continuous_events = _combined_exceedance_events(
            scores, model, config.sample_interval_minutes
        )
        expected_rows = int((end - start) / pd.Timedelta(minutes=config.sample_interval_minutes)) + 1
        record = {
            **window,
            "status": "scored",
            "scored_rows": len(scores),
            "expected_rows": expected_rows,
            "coverage": len(scores) / expected_rows,
            "status_counts": dict(Counter(scores["status"])),
            "t2_exceedance_95": float((scores["t2"] >= model.t2_limits[0.95]).mean()),
            "t2_exceedance_99": float((scores["t2"] >= model.t2_limits[0.99]).mean()),
            "spe_exceedance_95": float((scores["spe"] >= model.q_limits[0.95]).mean()),
            "spe_exceedance_99": float((scores["spe"] >= model.q_limits[0.99]).mean()),
            "event_count": len(continuous_events),
            "longest_event_minutes": _longest_event_minutes(
                continuous_events, config.sample_interval_minutes
            ),
            "maximum_t2": float(scores["t2"].max()),
            "maximum_spe": float(scores["spe"].max()),
            "continuous_events": continuous_events,
        }
        if window["type"] == "known_abnormal":
            record.update(
                _known_abnormal_summary(
                    scores, events, continuous_events, model, start
                )
            )
        records.append(record)
        scored = scores.copy()
        scored.insert(0, "validation_type", window["type"])
        scored.insert(0, "validation_window_id", window["id"])
        score_parts.append(scored)
        contribution_records.extend(
            [{**event, "validation_window_id": window["id"], "validation_type": window["type"]} for event in events]
        )

    combined = pd.concat(score_parts).sort_index()
    return {
        "validation_windows": windows,
        "window_summaries": records,
        "scores": combined,
        "contributions": contribution_records,
        "normal_validation_complete": any(
            item["enabled"] and item["type"] == "normal_validation" and item["status"] == "scored"
            for item in records
        ),
        "known_abnormal_complete": any(
            item["enabled"] and item["type"] == "known_abnormal" and item["status"] == "scored"
            for item in records
        ),
    }


def record_engineer_decision(
    manifest: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    decision: object,
    comment: object,
) -> dict[str, Any]:
    if (
        manifest.get("model_purpose") != "normal_state"
        or manifest.get("model_status") != "candidate"
    ):
        raise ValueError("只有normal_state/candidate模型可以记录验证结论")
    if decision not in _ENGINEER_DECISIONS:
        raise ValueError("工程师结论必须是passed、insufficient或failed")
    if not isinstance(comment, str):
        raise ValueError("工程师备注必须是文本")
    if decision == "passed" and not (
        validation_summary.get("normal_validation_complete")
        and validation_summary.get("known_abnormal_complete")
    ):
        raise ValueError("通过前必须完成正常验证和已知异常验证")
    return {
        "decision": decision,
        "comment": comment,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


def _window_bounds(windows: Sequence[Mapping[str, Any]]) -> list[TimeWindow]:
    return [(pd.Timestamp(window["start"]), pd.Timestamp(window["end"])) for window in windows]


def _validation_bounds(start_value: object, end_value: object) -> TimeWindow:
    try:
        start, end = pd.Timestamp(start_value), pd.Timestamp(end_value)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("validation_windows起止时间无效") from error
    return start, end


def _longest_event_minutes(
    events: Sequence[Mapping[str, Any]], sample_interval_minutes: int
) -> int:
    if not events:
        return 0
    return max(
        int(
            (pd.Timestamp(event["event_end"]) - pd.Timestamp(event["event_start"])).total_seconds()
            // 60
            + sample_interval_minutes
        )
        for event in events
    )


def _combined_exceedance_events(
    scores: pd.DataFrame, model: DPCAModel, sample_interval_minutes: int
) -> list[dict[str, Any]]:
    exceeded = (scores["t2"] >= model.t2_limits[0.95]) | (
        scores["spe"] >= model.q_limits[0.95]
    )
    timestamps = list(scores.index[exceeded])
    if not timestamps:
        return []
    expected_gap = pd.Timedelta(minutes=sample_interval_minutes)
    events: list[dict[str, Any]] = []
    start = previous = timestamps[0]
    points = 1
    for timestamp in timestamps[1:]:
        if timestamp - previous != expected_gap:
            events.append(
                {
                    "event_start": start.isoformat(),
                    "event_end": previous.isoformat(),
                    "point_count": points,
                }
            )
            start = timestamp
            points = 0
        previous = timestamp
        points += 1
    events.append(
        {
            "event_start": start.isoformat(),
            "event_end": previous.isoformat(),
            "point_count": points,
        }
    )
    return events


def _known_abnormal_summary(
    scores: pd.DataFrame,
    events: Sequence[Mapping[str, Any]],
    continuous_events: Sequence[Mapping[str, Any]],
    model: DPCAModel,
    start: pd.Timestamp,
) -> dict[str, Any]:
    exceeded = (scores["t2"] >= model.t2_limits[0.95]) | (scores["spe"] >= model.q_limits[0.95])
    if not exceeded.any():
        return {
            "first_exceedance": None,
            "first_exceedance_delay_minutes": None,
            "single_point_exceedance": False,
            "first_event_tags": [],
        }
    first_timestamp = pd.Timestamp(scores.index[exceeded.to_numpy().argmax()])
    first_events = [event for event in events if pd.Timestamp(event["event_start"]) == first_timestamp]
    return {
        "first_exceedance": first_timestamp.isoformat(),
        "first_exceedance_delay_minutes": int((first_timestamp - start).total_seconds() // 60),
        "single_point_exceedance": len(continuous_events) == 1
        and continuous_events[0]["point_count"] == 1,
        "first_event_tags": [tag for event in first_events for tag in event["tags"]],
    }
