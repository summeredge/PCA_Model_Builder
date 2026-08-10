"""Deterministic historical replay for an immutable frozen model package."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_io import load_model_package
from .preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    preprocess_window,
    preprocessing_config_from_mapping,
)
from .scoring_core import anomaly_tag_contributions, score_dynamic_feature_matrix


@dataclass(frozen=True)
class FrozenReplayResult:
    scores: pd.DataFrame
    summary: dict[str, Any]
    contributions: list[dict[str, Any]]


def replay_frozen_model(
    model_package: str | Path,
    historical_data: pd.DataFrame,
    replay_start: object,
    replay_end: object,
) -> FrozenReplayResult:
    """Replay one historical interval using only a schema-4 frozen package.

    ``historical_data`` must already have its timestamp as a unique, increasing
    ``DatetimeIndex``.  The package is read-only: neither its bytes nor its
    lifecycle metadata are changed by this operation.
    """
    source = Path(model_package)
    if source.suffix.lower() == ".pcadeploy":
        raise ValueError("deployment packages cannot be replayed")
    before = source.read_bytes()
    source_sha256 = hashlib.sha256(before).hexdigest()
    model, manifest = load_model_package(source)
    if (
        manifest.get("schema_version") != 4
        or manifest.get("model_purpose") != "normal_state"
        or manifest.get("model_status") != "frozen"
    ):
        raise ValueError("only schema 4 normal_state/frozen models can be replayed")

    start, end = _replay_bounds(replay_start, replay_end)
    _validate_history(historical_data)
    config_data = manifest["config"]
    tags = list(config_data["tags"])
    config = preprocessing_config_from_mapping(config_data)
    required = list(dict.fromkeys([*tags, *(item.column for item in config.state_filters)]))
    missing = [column for column in required if column not in historical_data.columns]
    if missing:
        raise ValueError(f"missing frozen model input Tags: {', '.join(missing)}")

    context_start = _context_start(start, config)
    context = historical_data.loc[context_start:end, required].copy()
    if context.empty:
        raise ValueError("replay interval contains no available historical data")
    processed = preprocess_window(
        context,
        tags,
        config,
        validate_quality=False,
        include_intermediates=True,
        allow_empty_state_filter=True,
        preprocessing_semantics="legacy",
    )
    if list(processed.dynamic.columns) != list(model.feature_names):
        raise ValueError("frozen model dynamic feature order does not match preprocessing")

    output_index = processed.state_filtered.index[
        processed.state_filtered.index.to_series().between(start, end, inclusive="both")
    ]
    scores = _empty_scores(output_index, model.n_components)
    reasons = _invalid_reasons(processed, tags, output_index, config)
    valid_index = output_index[processed.dynamic_valid_mask.reindex(output_index, fill_value=False)]
    if len(valid_index):
        dynamic = processed.dynamic.loc[valid_index, list(model.feature_names)]
        batch = score_dynamic_feature_matrix(
            dynamic.to_numpy(dtype=float),
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            components=model.components,
            eigenvalues=model.eigenvalues,
            t2_limits=model.t2_limits,
            q_limits=model.q_limits,
        )
        for number in range(model.n_components):
            scores.loc[valid_index, f"pc{number + 1}"] = batch.pc_scores[:, number]
        scores.loc[valid_index, "t2"] = batch.t2
        scores.loc[valid_index, "spe"] = batch.spe
        scores.loc[valid_index, "t2_limit_ratio"] = batch.t2_limit_ratio
        scores.loc[valid_index, "spe_limit_ratio"] = batch.spe_limit_ratio
        scores.loc[valid_index, "t2_status"] = batch.t2_status
        scores.loc[valid_index, "spe_status"] = batch.spe_status
        scores.loc[valid_index, "overall_status"] = batch.overall_status
        scores.loc[valid_index, "score_valid"] = batch.score_valid
        for position, timestamp in enumerate(valid_index):
            reasons.loc[timestamp] = batch.invalid_reason[position]
    scores["invalid_reason"] = reasons
    scores["status"] = scores["overall_status"]

    contributions = _contribution_records(model, manifest, processed, scores)
    replay_resampled = processed.resampled.loc[
        processed.resampled.index.to_series().between(start, end, inclusive="both")
    ]
    state_filter_excluded = len(replay_resampled) - len(scores)
    valid_scores = scores.loc[scores["score_valid"]]
    summary = {
        "model_id": manifest["model_id"],
        "model_version": manifest["model_version"],
        "source_frozen_sha256": source_sha256,
        "replay_start": start.isoformat(),
        "replay_end": end.isoformat(),
        "source_row_count": len(context),
        "output_row_count": len(scores),
        "score_valid_count": int(scores["score_valid"].sum()),
        "invalid_reason_counts": {
            key: int(value)
            for key, value in Counter(scores.loc[~scores["score_valid"], "invalid_reason"]).items()
        },
        "status_counts": {key: int(value) for key, value in Counter(scores["overall_status"]).items()},
        "state_filter_excluded_rows": int(state_filter_excluded),
        "t2_exceedance_95_count": int((valid_scores["t2"] >= model.t2_limits[0.95]).sum()),
        "t2_exceedance_99_count": int((valid_scores["t2"] >= model.t2_limits[0.99]).sum()),
        "spe_exceedance_95_count": int((valid_scores["spe"] >= model.q_limits[0.95]).sum()),
        "spe_exceedance_99_count": int((valid_scores["spe"] >= model.q_limits[0.99]).sum()),
        "maximum_t2": float(valid_scores["t2"].max()) if len(valid_scores) else None,
        "maximum_spe": float(valid_scores["spe"].max()) if len(valid_scores) else None,
        "preprocessing_summary": {
            **processed.summary.to_dict(),
            "replay_context_start": context_start.isoformat(),
        },
    }
    if source.read_bytes() != before:
        raise RuntimeError("frozen model package changed during replay")
    return FrozenReplayResult(scores=scores, summary=summary, contributions=contributions)


def _replay_bounds(start_value: object, end_value: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start, end = pd.Timestamp(start_value), pd.Timestamp(end_value)
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("replay start and end timestamps are invalid") from error
    return start, end


def _validate_history(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("historical replay data must use a DatetimeIndex")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("historical replay timestamps must be increasing and unique")


def _context_start(start: pd.Timestamp, config: PreprocessingConfig) -> pd.Timestamp:
    filter_context = (
        0
        if config.filter_method == "none"
        else config.smoothing_window_minutes - config.sample_interval_minutes
    )
    resampling_context = config.sample_interval_minutes if config.resampling_method != "none" else 0
    return start - pd.Timedelta(minutes=config.max_lag_minutes + filter_context + resampling_context)


def _empty_scores(index: pd.DatetimeIndex, component_count: int) -> pd.DataFrame:
    scores = pd.DataFrame(index=index)
    for number in range(component_count):
        scores[f"pc{number + 1}"] = np.nan
    for column in ("t2", "spe", "t2_limit_ratio", "spe_limit_ratio"):
        scores[column] = np.nan
    for column in ("t2_status", "spe_status", "overall_status"):
        scores[column] = "not_scored"
    scores["score_valid"] = False
    return scores


def _invalid_reasons(
    processed: PreprocessingResult,
    tags: list[str],
    output_index: pd.DatetimeIndex,
    config: PreprocessingConfig,
) -> pd.Series:
    reasons = pd.Series("insufficient_context", index=output_index, dtype=object)
    if not len(output_index):
        return reasons
    empty = processed.empty_bin_mask.reindex(output_index, fill_value=False)
    input_invalid = processed.input_invalid_mask.reindex(output_index, fill_value=False)
    warmup = processed.filter_warmup_mask.reindex(output_index, fill_value=False)
    filter_context = processed.filter_context_invalid_mask.reindex(output_index, fill_value=False)
    lag_warmup = processed.lag_warmup_mask.reindex(output_index, fill_value=False)
    lag_context = processed.lag_context_invalid_mask.reindex(output_index, fill_value=False)
    reasons.loc[empty] = "missing_input"
    if input_invalid.any():
        values = processed.resampled.loc[output_index[input_invalid], tags].to_numpy(dtype=float)
        non_finite = np.isinf(values).any(axis=1)
        positions = output_index[input_invalid]
        reasons.loc[positions[non_finite]] = "non_finite_input"
        reasons.loc[positions[~non_finite]] = "missing_input"
    raw_segments = processed.segment_ids.reindex(output_index)
    first_segment = processed.segment_ids.iloc[0] if len(processed.segment_ids) else None
    warmup_or_context = warmup | lag_warmup | filter_context | lag_context
    reasons.loc[filter_context | lag_context] = "insufficient_context"
    reasons.loc[warmup | lag_warmup] = "warming_up"
    final_segments = processed.final_segment_ids.reindex(output_index)
    first_final_segment = (
        processed.final_segment_ids.iloc[0] if len(processed.final_segment_ids) else None
    )
    state_context_reset = final_segments.ne(first_final_segment) & raw_segments.eq(first_segment)
    reasons.loc[state_context_reset & warmup_or_context] = "insufficient_context"
    gap_reset = raw_segments.ne(first_segment) & warmup_or_context
    reasons.loc[gap_reset] = "time_gap_reset"
    return reasons


def _contribution_records(
    model: Any,
    manifest: dict[str, Any],
    processed: PreprocessingResult,
    scores: pd.DataFrame,
) -> list[dict[str, Any]]:
    tag_configs = manifest["config"].get("tag_configs", {})
    records: list[dict[str, Any]] = []
    for timestamp, score in scores.loc[scores["score_valid"]].iterrows():
        dynamic = processed.dynamic.loc[timestamp, list(model.feature_names)].to_numpy(dtype=float)
        for statistic, limit in (("t2", model.t2_limits[0.95]), ("spe", model.q_limits[0.95])):
            tags = anomaly_tag_contributions(
                dynamic,
                _score_record(score),
                statistic,
                feature_names=model.feature_names,
                mean=model.mean,
                scale=model.scale,
                components=model.components,
                eigenvalues=model.eigenvalues,
                limit_95=limit,
            )
            if tags:
                records.append(
                    {
                        "timestamp": pd.Timestamp(timestamp).isoformat(),
                        "statistic": statistic,
                        "statistic_value": float(score[statistic]),
                        "limit_95": float(limit),
                        "tags": [
                            {
                                "tag": item.tag,
                                "description": str(tag_configs.get(item.tag, {}).get("description", "")),
                                "unit": str(tag_configs.get(item.tag, {}).get("unit", "")),
                                "contribution_pct": float(item.contribution_pct),
                                "lag_start_minutes": int(item.lag_start_minutes),
                                "lag_end_minutes": int(item.lag_end_minutes),
                            }
                            for item in tags
                        ],
                    }
                )
    return records


def _score_record(row: pd.Series) -> Any:
    from .scoring_core import SingleScoreResult

    return SingleScoreResult(
        pc_scores=np.array([row[column] for column in row.index if column.startswith("pc")], dtype=float),
        t2=float(row["t2"]),
        spe=float(row["spe"]),
        t2_limit_ratio=float(row["t2_limit_ratio"]),
        spe_limit_ratio=float(row["spe_limit_ratio"]),
        t2_status=str(row["t2_status"]),
        spe_status=str(row["spe_status"]),
        overall_status=str(row["overall_status"]),
        score_valid=True,
        invalid_reason=None,
    )
