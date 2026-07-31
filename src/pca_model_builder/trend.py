from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingConfig, infer_segment_ids
from .tag_profile import profile_tag


MAX_TREND_TAGS = 8
MAX_TREND_POINTS = 1200


def prepare_trend_frame(
    frame: pd.DataFrame,
    tags: Sequence[str],
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if len(tags) > MAX_TREND_TAGS:
        raise ValueError("趋势浏览一次最多选择8个Tag")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("趋势数据必须使用DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("趋势时间戳必须有序且唯一")
    missing = [tag for tag in tags if tag not in frame.columns]
    if missing:
        raise ValueError(f"找不到趋势Tag：{', '.join(missing)}")
    raw = frame.loc[:, tags].apply(pd.to_numeric, errors="coerce")
    segments = infer_segment_ids(frame.index, config.sample_interval_minutes)
    window_rows = config.smoothing_window_minutes // config.sample_interval_minutes
    smoothed = raw.groupby(segments, sort=False).transform(
        lambda group: group.rolling(
            window_rows, min_periods=window_rows
        ).mean()
    )
    return raw, smoothed, segments


def downsample_trend(
    raw: pd.DataFrame,
    smoothed: pd.DataFrame,
    segments: pd.Series,
    limit: int = MAX_TREND_POINTS,
) -> np.ndarray:
    if limit < 2:
        raise ValueError("趋势点数上限不得小于2")
    count = len(raw)
    if count <= limit:
        return np.arange(count, dtype=int)
    critical = {0, count - 1}
    segment_values = segments.to_numpy()
    gap_starts = np.flatnonzero(segment_values[1:] != segment_values[:-1]) + 1
    for position in gap_starts:
        critical.update((int(position - 1), int(position)))
    missing_positions = np.flatnonzero(raw.isna().any(axis=1).to_numpy())
    critical.update(int(position) for position in missing_positions)
    if len(critical) >= limit:
        middle = np.array(sorted(critical - {0, count - 1}), dtype=int)
        keep = _spread_positions(middle, limit - 2)
        return np.array(sorted({0, count - 1, *keep.tolist()}), dtype=int)

    remaining = limit - len(critical)
    series_count = max(1, raw.shape[1] * 2)
    bucket_count = max(1, remaining // (2 * series_count))
    candidates: set[int] = set()
    for positions in np.array_split(np.arange(count), bucket_count):
        if not len(positions):
            continue
        for values in (raw, smoothed):
            for tag in values.columns:
                bucket = values[tag].iloc[positions]
                finite = bucket[np.isfinite(bucket)]
                if finite.empty:
                    continue
                candidates.add(int(values.index.get_loc(finite.idxmin())))
                candidates.add(int(values.index.get_loc(finite.idxmax())))
    candidates.difference_update(critical)
    if len(candidates) > remaining:
        ordered = np.array(sorted(candidates), dtype=int)
        candidates = set(_spread_positions(ordered, remaining).tolist())
    selected = sorted(critical | candidates)
    return np.array(selected, dtype=int)


def trend_payload_data(
    indexed: pd.DataFrame,
    tags: Sequence[str],
    config: PreprocessingConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    tag_configs: Mapping[str, Mapping[str, Any]],
    reference_start: pd.Timestamp | None = None,
    reference_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if display_mode not in {"raw", "smoothed", "both"}:
        raise ValueError("趋势显示模式无效")
    raw, smoothed, segments = prepare_trend_frame(indexed, tags, config)
    mask = (raw.index >= start) & (raw.index <= end)
    if not mask.any():
        raise ValueError("趋势浏览窗口没有数据")
    current_raw = raw.loc[mask]
    current_smoothed = smoothed.loc[mask]
    current_segments = segments.loc[mask]
    positions = downsample_trend(
        current_raw, current_smoothed, current_segments
    )
    rows: list[dict[str, Any]] = []
    for position in positions:
        timestamp = current_raw.index[position]
        full_position = int(raw.index.get_loc(timestamp))
        item: dict[str, Any] = {"timestamp": timestamp.isoformat()}
        for tag in tags:
            if display_mode in {"raw", "both"}:
                item[f"{tag}__raw"] = _json_number(current_raw.iloc[position][tag])
            if display_mode in {"smoothed", "both"}:
                item[f"{tag}__smoothed"] = _json_number(
                    current_smoothed.iloc[position][tag]
                )
        item["gap_start"] = bool(
            full_position > 0
            and segments.iloc[full_position] != segments.iloc[full_position - 1]
        )
        rows.append(item)

    statistics: dict[str, Any] = {}
    reference_mask = None
    if reference_start is not None and reference_end is not None:
        reference_mask = (raw.index >= reference_start) & (raw.index <= reference_end)
    for tag in tags:
        statistics[tag] = {
            "full": profile_tag(raw[tag], tag_configs.get(tag)),
            "current": profile_tag(current_raw[tag], tag_configs.get(tag)),
            "reference": (
                profile_tag(raw.loc[reference_mask, tag], tag_configs.get(tag))
                if reference_mask is not None and reference_mask.any()
                else None
            ),
        }
    histogram = None
    histograms = {"current": None, "reference": None}
    if len(tags) == 1:
        histogram = _histogram(tags[0], current_raw[tags[0]])
        histograms["current"] = histogram
        if reference_mask is not None and reference_mask.any():
            histograms["reference"] = _histogram(
                tags[0], raw.loc[reference_mask, tags[0]]
            )
    return {
        "tags": list(tags),
        "display_mode": display_mode,
        "rows": rows,
        "statistics": statistics,
        "histogram": histogram,
        "histograms": histograms,
        "ranges": {
            tag: {
                key: tag_configs.get(tag, {}).get(key)
                for key in (
                    "engineering_min",
                    "engineering_max",
                    "normal_min",
                    "normal_max",
                    "alarm_min",
                    "alarm_max",
                )
            }
            for tag in tags
        },
    }


def _spread_positions(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(values):
        return np.array([], dtype=int)
    if len(values) <= count:
        return values
    return values[np.unique(np.linspace(0, len(values) - 1, count, dtype=int))]


def _json_number(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _histogram(tag: str, values: pd.Series) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    counts, edges = (
        np.histogram(finite, bins=min(20, max(1, int(np.sqrt(len(finite))))))
        if len(finite)
        else (np.array([], dtype=int), np.array([], dtype=float))
    )
    return {
        "tag": tag,
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
    }
