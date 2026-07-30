from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)


TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def ensure_disjoint_windows(
    training_windows: Sequence[TimeWindow],
    validation_windows: Sequence[TimeWindow],
) -> None:
    for train_start, train_end in training_windows:
        if train_start > train_end:
            raise ValueError("training window start must not follow its end")
        for validation_start, validation_end in validation_windows:
            if validation_start > validation_end:
                raise ValueError("validation window start must not follow its end")
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
