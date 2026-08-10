from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .preprocessing import (
    PreprocessingConfig,
    PreprocessingQualityError,
    preprocess_window,
    segment_raw_data,
)
from .windows import normalize_training_windows


@dataclass(frozen=True)
class TrainingBuildResult:
    dynamic: pd.DataFrame
    window_summaries: list[dict[str, Any]]
    training_window_totals: dict[str, int]
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
    reference_columns: Sequence[str] = (),
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
        selected_timestamps = pd.DatetimeIndex(selected[timestamp_column])
        if not selected_timestamps.is_monotonic_increasing:
            raise ValueError(
                f"训练窗口 {window['id']} 数据质量问题尚未处理：unsorted_timestamp(1)"
            )
        if selected_timestamps.has_duplicates:
            raise ValueError(
                f"训练窗口 {window['id']} 数据质量问题尚未处理：duplicate_timestamp(1)"
            )
        state_columns = [condition.column for condition in config.state_filters]
        indexed = (
            selected.loc[
                :,
                [
                    timestamp_column,
                    *dict.fromkeys(
                        [*tag_columns, *state_columns, *reference_columns]
                    ),
                ],
            ]
            .sort_values(timestamp_column)
            .set_index(timestamp_column)
        )
        if indexed.empty:
            summaries.append(
                {**window, "status": "dropped", "raw_samples": 0, "effective_samples": 0, "dropped_reason": "no_raw_samples", "segments": []}
            )
            continue

        try:
            segment_ids, source_interval, _ = segment_raw_data(
                indexed.index, config
            )
        except ValueError as error:
            code = (
                "duplicate_timestamp"
                if indexed.index.has_duplicates
                else "irregular_sampling"
            )
            raise ValueError(
                f"训练窗口 {window['id']} 数据质量问题尚未处理：{code}(1)"
            ) from error
        try:
            processed = preprocess_window(
                indexed,
                tag_columns,
                config,
                engineering_ranges,
                include_intermediates=True,
                include_variability=False,
                preserve_columns=reference_columns,
                # Training windows are isolated: never complete a boundary bucket
                # with samples outside the engineer-selected window.
                resampling_window=(start, end),
            )
        except PreprocessingQualityError as error:
            raise ValueError(
                _window_quality_error(window["id"], error.report)
            ) from error
        assert processed.resampled is not None
        assert processed.filtered is not None
        assert processed.state_filtered is not None
        dynamic = processed.dynamic
        resampled_segment_ids = processed.segment_ids
        segment_summaries: list[dict[str, Any]] = []
        for position, segment_id in enumerate(segment_ids.unique(), start=1):
            segment = indexed.loc[segment_ids.eq(segment_id)]
            resampled_mask = resampled_segment_ids.eq(segment_id)
            resampled_segment = processed.resampled.loc[resampled_mask]
            retained_index = processed.post_invalid_segment_ids.index[
                resampled_segment_ids.reindex(processed.post_invalid_segment_ids.index).eq(segment_id)
            ]
            filtered_segment = processed.filtered.loc[retained_index]
            state_segment = processed.state_filtered.loc[
                processed.state_filtered.index.intersection(retained_index)
            ]
            dynamic_segment = dynamic.loc[
                dynamic.index.intersection(resampled_segment.index)
            ]
            effective_samples = len(dynamic_segment)
            filter_loss = int(
                processed.filter_warmup_mask.reindex(resampled_segment.index)
                .fillna(False)
                .sum()
            )
            status = "used" if effective_samples else "dropped"
            segment_summaries.append(
                {
                    "id": f"{window['id']}-segment-{position:03d}",
                    "start": segment.index[0].isoformat(),
                    "end": segment.index[-1].isoformat(),
                    "raw_samples": len(segment),
                    "resampled_samples": len(resampled_segment),
                    "resampling_row_reduction": max(
                        0,
                        len(segment)
                        - processed.partial_resampling_row_loss_by_segment.get(
                            int(segment_id), 0
                        )
                        - int(
                            (~processed.empty_bin_mask.reindex(
                                resampled_segment.index
                            ).fillna(False)).sum()
                        ),
                    ) if config.resampling_method != "none" else 0,
                    "empty_bins": int(
                        processed.empty_bin_mask.reindex(resampled_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "partial_resampling_bin_loss": processed.partial_resampling_bin_loss_by_segment.get(
                        int(segment_id), 0
                    ),
                    "partial_resampling_row_loss": processed.partial_resampling_row_loss_by_segment.get(
                        int(segment_id), 0
                    ),
                    "filter_warmup_loss": filter_loss,
                    "filter_context_invalid_loss": int(
                        processed.filter_context_invalid_mask.reindex(state_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "state_filter_input_rows": len(filtered_segment),
                    "state_filter_output_rows": len(state_segment),
                    "state_filter_loss": len(filtered_segment) - len(state_segment),
                    "lag_warmup_loss": int(
                        processed.lag_warmup_mask.reindex(state_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "lag_context_invalid_loss": int(
                        processed.lag_context_invalid_mask.reindex(state_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "input_invalid_loss": int(
                        processed.input_invalid_mask.reindex(resampled_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "effective_samples": effective_samples,
                    "smoothing_lag_loss": filter_loss + int(
                        processed.lag_warmup_mask.reindex(state_segment.index)
                        .fillna(False)
                        .sum()
                    ),
                    "status": status,
                    "dropped_reason": (
                        None
                        if effective_samples
                        else "no_complete_resampling_bins"
                        if resampled_segment.empty
                        and config.resampling_method != "none"
                        else "insufficient_after_smoothing_and_lag"
                    ),
                }
            )

        effective_samples = len(dynamic)
        reference = processed.resampled.loc[
            processed.resampled.index.intersection(processed.state_filtered.index)
        ]
        if not reference.empty:
            reference_parts.append(reference.reset_index(names=timestamp_column))
        preprocessing_summary = processed.summary
        summaries.append(
            {
                **window,
                "status": "used" if effective_samples else "dropped",
                "raw_samples": len(indexed),
                "source_interval_minutes": source_interval,
                "target_interval_minutes": config.sample_interval_minutes,
                "resampling_method": config.resampling_method,
                "resampled_samples": preprocessing_summary.resampled_row_count,
                "resampling_row_reduction": preprocessing_summary.resampling_row_reduction,
                "empty_bins": preprocessing_summary.empty_bin_count,
                "partial_resampling_bin_loss": preprocessing_summary.partial_resampling_bin_loss,
                "partial_resampling_row_loss": preprocessing_summary.partial_resampling_row_loss,
                "raw_segment_count": preprocessing_summary.raw_segment_count,
                "raw_gap_count": preprocessing_summary.raw_gap_count,
                "raw_gap_ranges": list(preprocessing_summary.raw_gap_ranges),
                "filter_method": config.filter_method,
                "filter_warmup_loss": preprocessing_summary.filter_warmup_loss,
                "filter_context_invalid_loss": preprocessing_summary.filter_context_invalid_loss,
                "state_filter_input_rows": preprocessing_summary.state_filter_input_rows,
                "state_filter_output_rows": preprocessing_summary.state_filter_output_rows,
                "state_filter_loss": (
                    preprocessing_summary.state_filter_input_rows
                    - preprocessing_summary.state_filter_output_rows
                ),
                "lag_warmup_loss": preprocessing_summary.lag_warmup_loss,
                "lag_context_invalid_loss": preprocessing_summary.lag_context_invalid_loss,
                "input_invalid_loss": preprocessing_summary.input_invalid_loss,
                "effective_samples": effective_samples,
                "smoothing_lag_loss": (
                    preprocessing_summary.filter_warmup_loss
                    + preprocessing_summary.lag_warmup_loss
                ),
                "dropped_reason": (
                    None
                    if effective_samples
                    else "no_complete_resampling_bins"
                    if processed.resampled.empty
                    and config.resampling_method != "none"
                    else "insufficient_after_smoothing_and_lag"
                ),
                "segments": segment_summaries,
            }
        )
        if effective_samples:
            dynamic_parts.append(dynamic)

    if not dynamic_parts:
        if validate_dynamic:
            raise ValueError("所有启用窗口在平滑和 Lag 扩展后均无有效训练样本")
        dynamic = pd.DataFrame()
    else:
        dynamic = pd.concat(dynamic_parts).sort_index()
        if dynamic.index.has_duplicates:
            raise ValueError(
                "independent training windows produced duplicate dynamic timestamps"
            )
    reference = (
        pd.concat(reference_parts).sort_values(timestamp_column)
        if reference_parts
        else pd.DataFrame(columns=[timestamp_column, *dict.fromkeys([
            *tag_columns,
            *(condition.column for condition in config.state_filters),
            *reference_columns,
        ])])
    )
    if reference[timestamp_column].duplicated().any():
        raise ValueError(
            "independent training windows produced duplicate reference timestamps"
        )
    if sum(item.get("effective_samples", 0) for item in summaries) != len(dynamic):
        raise ValueError("training window summaries do not match merged dynamic rows")
    training_window_totals = {
        "enabled_window_count": sum(window["enabled"] for window in windows),
        "used_window_count": sum(item["status"] == "used" for item in summaries),
        "dropped_window_count": sum(
            item["status"] == "dropped" for item in summaries
        ),
        "training_rows": len(dynamic),
    }
    return TrainingBuildResult(
        dynamic=dynamic,
        window_summaries=summaries,
        training_window_totals=training_window_totals,
        reference=reference,
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
