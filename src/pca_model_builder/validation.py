from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from .contribution import contribution_event_records, exceedance_contribution_tables
from .dpca import DPCAModel
from .preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    PreprocessingQualityError,
    preprocess_window,
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
    filter_context = (
        0
        if config.filter_method == "none"
        else config.smoothing_window_minutes - config.sample_interval_minutes
    )
    resampling_context = (
        config.sample_interval_minutes if config.resampling_method != "none" else 0
    )
    warmup_minutes = config.max_lag_minutes + filter_context + resampling_context
    return validation_start - pd.Timedelta(minutes=warmup_minutes)


def build_validation_matrix(
    indexed_frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build validation features with pre-window history, then score from start."""
    scoring, _ = _preprocess_validation_window(
        indexed_frame,
        tag_columns,
        config,
        validation_start,
        validation_end,
        engineering_ranges,
    )
    return scoring


def _preprocess_validation_window(
    indexed_frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, PreprocessingResult]:
    context_start = validation_context_start(validation_start, config)
    context = indexed_frame.loc[context_start:validation_end]
    try:
        # Validation deliberately includes pre-start context so the requested
        # first score can use a complete bucket, filter history, and Lag history.
        processed = preprocess_window(
            context, tag_columns, config, engineering_ranges
        )
    except PreprocessingQualityError as error:
        details = "; ".join(
            f"{issue.code}({issue.count})" for issue in error.report.issues
            if issue.severity == "error"
        )
        raise ValueError(f"validation data quality review required: {details}") from error
    scoring = processed.dynamic.loc[validation_start:validation_end]
    if scoring.empty or scoring.index[0] != validation_start:
        raise ValueError(
            "validation context is insufficient to score from the requested start"
        )
    if scoring.index[-1] != validation_end:
        raise ValueError("validation data do not cover the requested end")
    return scoring, processed


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
    scored_windows: list[dict[str, Any]] = []
    configured_ranges = {
        tag: (
            float(tag_config["engineering_min"]),
            float(tag_config["engineering_max"]),
        )
        for tag, tag_config in (tag_configs or {}).items()
        if tag_config.get("engineering_min") is not None
        and tag_config.get("engineering_max") is not None
    }
    for window in windows:
        if not window["enabled"]:
            records.append({**window, "status": "disabled", "scored_rows": 0})
            continue
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        dynamic, preprocessing = _preprocess_validation_window(
            indexed_frame,
            tag_columns,
            config,
            start,
            end,
            configured_ranges,
        )
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
            "preprocessing_summary": {
                **preprocessing.summary.to_dict(),
                "scoring_row_count": len(dynamic),
                "first_scored_timestamp": dynamic.index[0].isoformat(),
            },
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
        scored_windows.append(
            {
                "id": window["id"],
                "type": window["type"],
                "start": start,
                "scores": scores,
                "continuous_events": continuous_events,
            }
        )
        contribution_records.extend(
            [{**event, "validation_window_id": window["id"], "validation_type": window["type"]} for event in events]
        )

    combined = pd.concat(score_parts).sort_index()
    return {
        "validation_windows": windows,
        "window_summaries": records,
        "scores": combined,
        "contributions": contribution_records,
        "validation_metrics": _validation_metrics(
            scored_windows, model, config.sample_interval_minutes
        ),
        "contribution_stability": _contribution_stability(
            contribution_records, model.feature_names
        ),
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
    if decision == "passed" and not _has_pr6_validation_evidence(validation_summary):
        raise ValueError("通过前必须重新执行独立验证以生成完整PR-6验证证据")
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


def _validation_metrics(
    scored_windows: Sequence[Mapping[str, Any]],
    model: DPCAModel,
    sample_interval_minutes: int,
) -> dict[str, Any]:
    normal_windows = [
        window for window in scored_windows if window["type"] == "normal_validation"
    ]
    abnormal_windows = [
        window for window in scored_windows if window["type"] == "known_abnormal"
    ]
    return {
        "normal_validation": _normal_validation_metrics(
            normal_windows, model, sample_interval_minutes
        ),
        "known_abnormal": _known_abnormal_metrics(abnormal_windows, model),
    }


def _normal_validation_metrics(
    windows: Sequence[Mapping[str, Any]],
    model: DPCAModel,
    sample_interval_minutes: int,
) -> dict[str, Any]:
    score_frames = [window["scores"] for window in windows]
    scoring_row_count = sum(len(scores) for scores in score_frames)

    def rate(statistic: str | None, confidence: float) -> float | None:
        if not scoring_row_count:
            return None
        if statistic == "t2":
            exceeded = sum(
                int((scores["t2"] >= model.t2_limits[confidence]).sum())
                for scores in score_frames
            )
        elif statistic == "spe":
            exceeded = sum(
                int((scores["spe"] >= model.q_limits[confidence]).sum())
                for scores in score_frames
            )
        else:
            exceeded = sum(
                int(
                    (
                        (scores["t2"] >= model.t2_limits[confidence])
                        | (scores["spe"] >= model.q_limits[confidence])
                    ).sum()
                )
                for scores in score_frames
            )
        return exceeded / scoring_row_count

    events = [
        event
        for window in windows
        for event in window["continuous_events"]
    ]
    return {
        "valid_window_count": len(windows),
        "scoring_row_count": scoring_row_count,
        "t2": {"exceedance_rate_95": rate("t2", 0.95), "exceedance_rate_99": rate("t2", 0.99)},
        "spe": {"exceedance_rate_95": rate("spe", 0.95), "exceedance_rate_99": rate("spe", 0.99)},
        "overall": {"exceedance_rate_95": rate(None, 0.95), "exceedance_rate_99": rate(None, 0.99)},
        "continuous_false_alarm_event_count_95": len(events),
        "longest_continuous_false_alarm_minutes": _longest_event_minutes(
            events, sample_interval_minutes
        ),
    }


def _known_abnormal_metrics(
    windows: Sequence[Mapping[str, Any]], model: DPCAModel
) -> dict[str, Any]:
    window_metrics: list[dict[str, Any]] = []
    detected_counts = {"overall": {}, "t2": {}, "spe": {}}
    for confidence in (0.95, 0.99):
        for statistic in detected_counts:
            detected_counts[statistic][confidence] = 0
    for window in windows:
        scores = window["scores"]
        item: dict[str, Any] = {"validation_window_id": window["id"]}
        for confidence in (0.95, 0.99):
            masks = {
                "t2": scores["t2"] >= model.t2_limits[confidence],
                "spe": scores["spe"] >= model.q_limits[confidence],
            }
            masks["overall"] = masks["t2"] | masks["spe"]
            for statistic, mask in masks.items():
                if mask.any():
                    detected_counts[statistic][confidence] += 1
            overall = masks["overall"]
            key = str(int(confidence * 100))
            if overall.any():
                timestamp = pd.Timestamp(scores.index[int(np.argmax(overall.to_numpy()))])
                item[f"first_detection_{key}"] = timestamp.isoformat()
                item[f"first_detection_delay_minutes_{key}"] = int(
                    (timestamp - window["start"]).total_seconds() // 60
                )
            else:
                item[f"first_detection_{key}"] = None
                item[f"first_detection_delay_minutes_{key}"] = None
        window_metrics.append(item)

    valid_window_count = len(windows)
    delays = [
        item["first_detection_delay_minutes_95"]
        for item in window_metrics
        if item["first_detection_delay_minutes_95"] is not None
    ]
    rate = lambda count: None if not valid_window_count else count / valid_window_count
    return {
        "valid_window_count": valid_window_count,
        "detected_window_count_95": detected_counts["overall"][0.95],
        "detection_rate_95": rate(detected_counts["overall"][0.95]),
        "detected_window_count_99": detected_counts["overall"][0.99],
        "detection_rate_99": rate(detected_counts["overall"][0.99]),
        "t2_detected_window_count_95": detected_counts["t2"][0.95],
        "t2_detected_window_count_99": detected_counts["t2"][0.99],
        "spe_detected_window_count_95": detected_counts["spe"][0.95],
        "spe_detected_window_count_99": detected_counts["spe"][0.99],
        "windows": window_metrics,
        "first_detection_delay_minutes_95_median": (
            float(np.median(delays)) if delays else None
        ),
        "first_detection_delay_minutes_95_max": max(delays) if delays else None,
    }


def _contribution_stability(
    events: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    tag_order = list(dict.fromkeys(name.rsplit("__lag_", 1)[0] for name in feature_names))
    return {
        validation_type: {
            statistic: _contribution_stability_group(
                [
                    event
                    for event in events
                    if event.get("validation_type") == validation_type
                    and event.get("statistic") == statistic
                ],
                tag_order,
            )
            for statistic in ("t2", "spe")
        }
        for validation_type in ("normal_validation", "known_abnormal")
    }


def _contribution_stability_group(
    events: Sequence[Mapping[str, Any]], tag_order: Sequence[str]
) -> dict[str, Any]:
    event_count = len(events)
    top_k = min(3, len(tag_order))
    if not event_count:
        return {
            "event_count": 0,
            "top_k": top_k,
            "top1_consistency_rate": None,
            "average_top_k_jaccard_similarity": None,
            "average_contribution_cosine_similarity": None,
            "tags": [],
        }

    vectors: list[np.ndarray] = []
    rankings: list[list[str]] = []
    for event in events:
        values = {str(item["tag"]): float(item["contribution_pct"]) for item in event["tags"]}
        vectors.append(np.array([values.get(tag, 0.0) for tag in tag_order], dtype=float))
        rankings.append(sorted(tag_order, key=lambda tag: (-values.get(tag, 0.0), tag)))
    top1 = [ranking[0] for ranking in rankings] if top_k else []
    pair_count = 0
    jaccard_total = 0.0
    cosine_total = 0.0
    for left in range(event_count):
        left_top_k = set(rankings[left][:top_k])
        for right in range(left + 1, event_count):
            right_top_k = set(rankings[right][:top_k])
            pair_count += 1
            if top_k:
                jaccard_total += len(left_top_k & right_top_k) / len(
                    left_top_k | right_top_k
                )
            cosine_total += _cosine_similarity(vectors[left], vectors[right])
    return {
        "event_count": event_count,
        "top_k": top_k,
        "top1_consistency_rate": max(Counter(top1).values()) / event_count if top1 else None,
        "average_top_k_jaccard_similarity": (
            jaccard_total / pair_count if pair_count and top_k else None
        ),
        "average_contribution_cosine_similarity": (
            cosine_total / pair_count if pair_count else None
        ),
        "tags": [
            {
                "tag": tag,
                "top1_count": top1.count(tag),
                "top_k_count": sum(tag in ranking[:top_k] for ranking in rankings),
                "top_k_recurrence_rate": sum(tag in ranking[:top_k] for ranking in rankings) / event_count,
                "average_contribution_pct": float(np.mean([vector[index] for vector in vectors])),
                "median_contribution_pct": float(np.median([vector[index] for vector in vectors])),
            }
            for index, tag in enumerate(tag_order)
        ],
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def _has_pr6_validation_evidence(validation_summary: Mapping[str, Any]) -> bool:
    metrics = validation_summary.get("validation_metrics")
    stability = validation_summary.get("contribution_stability")
    if not isinstance(metrics, Mapping) or not isinstance(stability, Mapping):
        return False
    return (
        _has_normal_validation_metrics(metrics.get("normal_validation"))
        and _has_known_abnormal_metrics(metrics.get("known_abnormal"))
        and all(
            _has_contribution_stability_group(
                stability.get(validation_type, {}).get(statistic)
                if isinstance(stability.get(validation_type), Mapping)
                else None
            )
            for validation_type in ("normal_validation", "known_abnormal")
            for statistic in ("t2", "spe")
        )
    )


def _has_normal_validation_metrics(value: object) -> bool:
    fields = (
        "valid_window_count",
        "scoring_row_count",
        "t2",
        "spe",
        "overall",
        "continuous_false_alarm_event_count_95",
        "longest_continuous_false_alarm_minutes",
    )
    if not _has_fields(value, fields):
        return False
    return (
        _is_positive_integer(value.get("valid_window_count"))
        and _is_positive_integer(value.get("scoring_row_count"))
        and all(
            isinstance(value.get(statistic), Mapping)
            and _has_fields(
                value[statistic], ("exceedance_rate_95", "exceedance_rate_99")
            )
            and all(
                _is_ratio(value[statistic].get(f"exceedance_rate_{confidence}"))
                for confidence in (95, 99)
            )
            for statistic in ("t2", "spe", "overall")
        )
        and _is_nonnegative_integer(value.get("continuous_false_alarm_event_count_95"))
        and _is_nonnegative_integer(value.get("longest_continuous_false_alarm_minutes"))
    )


def _has_known_abnormal_metrics(value: object) -> bool:
    count_fields = (
        "detected_window_count_95",
        "detected_window_count_99",
        "t2_detected_window_count_95",
        "t2_detected_window_count_99",
        "spe_detected_window_count_95",
        "spe_detected_window_count_99",
    )
    fields = (
        "valid_window_count",
        *count_fields,
        "detection_rate_95",
        "detection_rate_99",
        "windows",
        "first_detection_delay_minutes_95_median",
        "first_detection_delay_minutes_95_max",
    )
    if not _has_fields(value, fields) or not _is_positive_integer(
        value["valid_window_count"]
    ):
        return False
    valid_window_count = value["valid_window_count"]
    windows = value.get("windows")
    return (
        all(
            _is_nonnegative_integer(value.get(field))
            and value[field] <= valid_window_count
            for field in count_fields
        )
        and all(_is_ratio(value.get(field)) for field in ("detection_rate_95", "detection_rate_99"))
        and isinstance(windows, list)
        and len(windows) == valid_window_count
        and all(_has_detection_window(window) for window in windows)
        and all(
            _is_nonnegative_number_or_none(value.get(field))
            for field in (
                "first_detection_delay_minutes_95_median",
                "first_detection_delay_minutes_95_max",
            )
        )
    )


def _has_detection_window(value: object) -> bool:
    fields = (
        "validation_window_id",
        "first_detection_95",
        "first_detection_delay_minutes_95",
        "first_detection_99",
        "first_detection_delay_minutes_99",
    )
    if not _has_fields(value, fields) or not isinstance(
        value.get("validation_window_id"), str
    ) or not value["validation_window_id"]:
        return False
    return all(
        value.get(f"first_detection_{confidence}") is None
        or isinstance(value.get(f"first_detection_{confidence}"), str)
        for confidence in (95, 99)
    ) and all(
        _is_nonnegative_number_or_none(
            value.get(f"first_detection_delay_minutes_{confidence}")
        )
        for confidence in (95, 99)
    )


def _has_contribution_stability_group(value: object) -> bool:
    fields = (
        "event_count",
        "top_k",
        "top1_consistency_rate",
        "average_top_k_jaccard_similarity",
        "average_contribution_cosine_similarity",
        "tags",
    )
    if not _has_fields(value, fields):
        return False
    event_count = value.get("event_count")
    tags = value.get("tags")
    return (
        _is_nonnegative_integer(event_count)
        and _is_nonnegative_integer(value.get("top_k"))
        and all(
            _is_ratio(value.get(field))
            for field in (
                "top1_consistency_rate",
                "average_top_k_jaccard_similarity",
                "average_contribution_cosine_similarity",
            )
        )
        and isinstance(tags, list)
        and (event_count == 0 or bool(tags))
        and all(_has_tag_stability(value) for value in tags)
    )


def _has_tag_stability(value: object) -> bool:
    return (
        _has_fields(
            value,
            (
                "tag",
                "top1_count",
                "top_k_count",
                "top_k_recurrence_rate",
                "average_contribution_pct",
                "median_contribution_pct",
            ),
        )
        and isinstance(value.get("tag"), str)
        and bool(value["tag"])
        and _is_nonnegative_integer(value.get("top1_count"))
        and _is_nonnegative_integer(value.get("top_k_count"))
        and _is_ratio(value.get("top_k_recurrence_rate"))
        and _is_contribution_rate(value.get("average_contribution_pct"))
        and _is_contribution_rate(value.get("median_contribution_pct"))
    )


def _is_positive_integer(value: object) -> bool:
    return _is_nonnegative_integer(value) and value > 0


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _has_fields(value: object, fields: Sequence[str]) -> bool:
    return isinstance(value, Mapping) and all(field in value for field in fields)


def _is_ratio(value: object) -> bool:
    if value is None or not _is_finite_number(value, minimum=0.0):
        return value is None
    return float(value) <= 1.0 or math.isclose(
        float(value), 1.0, rel_tol=0.0, abs_tol=1e-12
    )


def _is_contribution_rate(value: object) -> bool:
    return _is_finite_number(value, minimum=0.0, maximum=100.0)


def _is_nonnegative_number_or_none(value: object) -> bool:
    return value is None or _is_finite_number(value, minimum=0.0)


def _is_finite_number(
    value: object, minimum: float, maximum: float | None = None
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric >= minimum and (
        maximum is None or numeric <= maximum
    )
