from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessingConfig:
    sample_interval_minutes: int = 5
    smoothing_window_minutes: int = 10
    max_lag_minutes: int = 60
    lag_step_minutes: int = 5

    def __post_init__(self) -> None:
        values = {
            "sample interval": self.sample_interval_minutes,
            "smoothing window": self.smoothing_window_minutes,
            "lag step": self.lag_step_minutes,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_lag_minutes < 0:
            raise ValueError("maximum lag must not be negative")
        if self.smoothing_window_minutes % self.sample_interval_minutes:
            raise ValueError("smoothing window must be an integer multiple of sampling")
        if self.lag_step_minutes % self.sample_interval_minutes:
            raise ValueError("lag step must be an integer multiple of sampling")
        if self.max_lag_minutes % self.lag_step_minutes:
            raise ValueError("maximum lag must be an integer multiple of lag step")


def infer_segment_ids(
    index: pd.DatetimeIndex, sample_interval_minutes: int
) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("data index must be a DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("timestamps must be sorted and unique")
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    gaps = index.to_series().diff().gt(expected)
    return gaps.cumsum().astype(int).set_axis(index)


def build_dynamic_matrix(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    segment_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply causal smoothing and lag expansion within physical time segments."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("data index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("timestamps must be sorted and unique")
    missing = [tag for tag in tag_columns if tag not in frame.columns]
    if missing:
        raise ValueError(f"missing tag columns: {', '.join(missing)}")

    numeric = frame.loc[:, tag_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("model inputs contain non-finite values")

    if segment_ids is None:
        segment_ids = infer_segment_ids(
            frame.index, config.sample_interval_minutes
        )
    else:
        segment_ids = segment_ids.reindex(frame.index)
        if segment_ids.isna().any():
            raise ValueError("segment identifiers must cover every timestamp")

    window_rows = config.smoothing_window_minutes // config.sample_interval_minutes
    smoothed = numeric.groupby(segment_ids, sort=False).transform(
        lambda group: group.rolling(window_rows, min_periods=window_rows).mean()
    )

    dynamic_columns: dict[str, pd.Series] = {}
    for lag_minutes in range(
        0, config.max_lag_minutes + 1, config.lag_step_minutes
    ):
        lag_rows = lag_minutes // config.sample_interval_minutes
        lagged = smoothed.groupby(segment_ids, sort=False).shift(lag_rows)
        for tag in tag_columns:
            dynamic_columns[f"{tag}__lag_{lag_minutes:03d}min"] = lagged[tag]

    return pd.DataFrame(dynamic_columns, index=frame.index).dropna(how="any")

