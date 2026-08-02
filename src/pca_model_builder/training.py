from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from .preprocessing import PreprocessingConfig, build_dynamic_matrix, infer_segment_ids
from .windows import normalize_training_windows


@dataclass(frozen=True)
class TrainingBuildResult:
    dynamic: pd.DataFrame
    window_summaries: list[dict[str, Any]]


def build_training_matrix(
    frame: pd.DataFrame,
    timestamp_column: str,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    training_windows: object,
) -> TrainingBuildResult:
    """Build each enabled window and physical segment independently."""
    windows = normalize_training_windows(training_windows)
    dynamic_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for window in windows:
        if not window["enabled"]:
            summaries.append({**window, "status": "disabled", "raw_samples": 0, "effective_samples": 0, "segments": []})
            continue
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        selected = frame.loc[frame[timestamp_column].between(start, end, inclusive="both")]
        indexed = (
            selected.loc[:, [timestamp_column, *tag_columns]]
            .sort_values(timestamp_column)
            .set_index(timestamp_column)
        )
        if indexed.empty:
            summaries.append(
                {**window, "status": "dropped", "raw_samples": 0, "effective_samples": 0, "dropped_reason": "no_raw_samples", "segments": []}
            )
            continue

        segment_ids = infer_segment_ids(indexed.index, config.sample_interval_minutes)
        segment_summaries: list[dict[str, Any]] = []
        window_parts: list[pd.DataFrame] = []
        for position, segment_id in enumerate(segment_ids.unique(), start=1):
            segment = indexed.loc[segment_ids.eq(segment_id)]
            dynamic = build_dynamic_matrix(segment, tag_columns, config)
            effective_samples = len(dynamic)
            status = "used" if effective_samples else "dropped"
            segment_summaries.append(
                {
                    "id": f"{window['id']}-segment-{position:03d}",
                    "start": segment.index[0].isoformat(),
                    "end": segment.index[-1].isoformat(),
                    "raw_samples": len(segment),
                    "effective_samples": effective_samples,
                    "smoothing_lag_loss": len(segment) - effective_samples,
                    "status": status,
                    "dropped_reason": None if effective_samples else "insufficient_after_smoothing_and_lag",
                }
            )
            if effective_samples:
                window_parts.append(dynamic)

        effective_samples = sum(len(part) for part in window_parts)
        summaries.append(
            {
                **window,
                "status": "used" if effective_samples else "dropped",
                "raw_samples": len(indexed),
                "effective_samples": effective_samples,
                "smoothing_lag_loss": len(indexed) - effective_samples,
                "dropped_reason": None if effective_samples else "insufficient_after_smoothing_and_lag",
                "segments": segment_summaries,
            }
        )
        dynamic_parts.extend(window_parts)

    if not dynamic_parts:
        raise ValueError("所有启用窗口在平滑和 Lag 扩展后均无有效训练样本")
    return TrainingBuildResult(
        dynamic=pd.concat(dynamic_parts).sort_index(),
        window_summaries=summaries,
    )
