from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingConfig, build_dynamic_matrix, infer_segment_ids
from .quality import inspect_data_quality
from .windows import normalize_training_windows


@dataclass(frozen=True)
class TrainingBuildResult:
    dynamic: pd.DataFrame
    window_summaries: list[dict[str, Any]]
    reference: pd.DataFrame
    global_quality_warnings: list[dict[str, Any]]


def build_training_matrix(
    frame: pd.DataFrame,
    timestamp_column: str,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    training_windows: object,
    engineering_ranges: dict[str, tuple[float, float]] | None = None,
    validate_dynamic: bool = True,
) -> TrainingBuildResult:
    """Build each enabled window and physical segment independently."""
    windows = normalize_training_windows(training_windows)
    dynamic_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    reference_parts: list[pd.DataFrame] = []

    if not any(window["enabled"] for window in windows):
        raise ValueError("至少需要一个启用的training_windows窗口")

    for window in windows:
        if not window["enabled"]:
            summaries.append({**window, "status": "disabled", "raw_samples": 0, "effective_samples": 0, "segments": []})
            continue
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        selected = frame.loc[frame[timestamp_column].between(start, end, inclusive="both")]
        report = inspect_data_quality(
            selected,
            timestamp_column,
            tag_columns,
            engineering_ranges=engineering_ranges,
            expected_interval_minutes=config.sample_interval_minutes,
            include_variability=False,
        )
        if not report.can_train:
            raise ValueError(_window_quality_error(window["id"], report))
        reference_parts.append(selected)
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
        if validate_dynamic:
            raise ValueError("所有启用窗口在平滑和 Lag 扩展后均无有效训练样本")
        dynamic = pd.DataFrame()
    else:
        dynamic = pd.concat(dynamic_parts).sort_index()
    return TrainingBuildResult(
        dynamic=dynamic,
        window_summaries=summaries,
        reference=pd.concat(reference_parts).sort_values(timestamp_column),
        global_quality_warnings=_validate_dynamic_matrix(dynamic) if validate_dynamic else [],
    )


def _window_quality_error(window_id: str, report: Any) -> str:
    details = "；".join(
        f"{issue.code}({issue.count})"
        + (f"[{issue.tag}]" if issue.tag else "")
        + f"：{issue.message}"
        for issue in report.issues
        if issue.severity == "error"
    )
    return f"训练窗口 {window_id} 数据质量问题尚未处理：{details}"


def _validate_dynamic_matrix(dynamic: pd.DataFrame) -> list[dict[str, Any]]:
    values = dynamic.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("合并后的训练矩阵包含非有限数值")
    if len(dynamic) < 3:
        raise ValueError("合并后的训练矩阵至少需要三个有效动态样本")

    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    tolerance = np.finfo(float).eps
    constant_features = [
        name for name, value in zip(dynamic.columns, scale, strict=True) if value <= tolerance
    ]
    if constant_features:
        raise ValueError(
            "合并后的训练矩阵存在常量动态特征，无法标准化："
            + ", ".join(constant_features)
        )

    standardized = (values - mean) / scale
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    eigenvalues = singular_values**2 / (len(dynamic) - 1)
    effective_rank = int(np.sum(eigenvalues > tolerance))
    if effective_rank < 3:
        raise ValueError(
            "合并后的训练矩阵有效秩不足，无法同时保留PC1、PC2和SPE残差空间"
        )

    warnings: list[dict[str, Any]] = []
    for name, feature_mean, feature_scale in zip(
        dynamic.columns, mean, scale, strict=True
    ):
        threshold = max(abs(float(feature_mean)), 1.0) * 1e-6
        if feature_scale <= threshold:
            warnings.append(
                {
                    "code": "near_constant_feature",
                    "feature": name,
                    "standard_deviation": float(feature_scale),
                    "threshold": threshold,
                }
            )
    return warnings
