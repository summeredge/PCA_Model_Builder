from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .quality import QualityReport, inspect_data_quality


_RESAMPLING_METHODS = {"none", "mean", "median", "last"}
_FILTER_METHODS = {"none", "trailing_mean", "trailing_median"}


@dataclass(frozen=True)
class StateFilter:
    column: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("state filter column must not be empty")
        if self.minimum is None and self.maximum is None:
            raise ValueError("state filter requires minimum or maximum")
        if any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
            )
            for value in (self.minimum, self.maximum)
        ):
            raise ValueError("state filter bounds must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("state filter minimum must not exceed maximum")


@dataclass(frozen=True)
class PreprocessingConfig:
    sample_interval_minutes: int = 5
    smoothing_window_minutes: int = 10
    max_lag_minutes: int = 60
    lag_step_minutes: int = 5
    resampling_method: str = "none"
    filter_method: str = "trailing_mean"
    gap_threshold_minutes: float | None = None
    state_filters: tuple[StateFilter, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("sample interval", self.sample_interval_minutes),
            ("smoothing window", self.smoothing_window_minutes),
            ("maximum lag", self.max_lag_minutes),
            ("lag step", self.lag_step_minutes),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.sample_interval_minutes <= 0:
            raise ValueError("sample interval must be positive")
        if self.smoothing_window_minutes < 0:
            raise ValueError("smoothing window must not be negative")
        if self.max_lag_minutes < 0:
            raise ValueError("maximum lag must not be negative")
        if self.lag_step_minutes <= 0:
            raise ValueError("lag step must be positive")
        if self.resampling_method not in _RESAMPLING_METHODS:
            raise ValueError(f"unsupported resampling method: {self.resampling_method}")
        if self.filter_method not in _FILTER_METHODS:
            raise ValueError(f"unsupported filter method: {self.filter_method}")
        if self.max_lag_minutes % self.lag_step_minutes:
            raise ValueError("maximum lag must be an integer multiple of lag step")
        if self.lag_step_minutes % self.sample_interval_minutes:
            raise ValueError("lag step must be an integer multiple of sampling")
        if self.filter_method != "none":
            if self.smoothing_window_minutes < self.sample_interval_minutes:
                raise ValueError("filter window must not be shorter than sampling")
            if self.smoothing_window_minutes % self.sample_interval_minutes:
                raise ValueError("smoothing window must be an integer multiple of sampling")
        if self.gap_threshold_minutes is not None and (
            self.gap_threshold_minutes <= 0
            or self.gap_threshold_minutes < self.sample_interval_minutes
        ):
            raise ValueError(
                "gap threshold must be positive and not shorter than sampling"
            )
        normalized_filters: list[StateFilter] = []
        for condition in self.state_filters:
            if isinstance(condition, StateFilter):
                normalized_filters.append(condition)
            elif isinstance(condition, Mapping):
                try:
                    normalized_filters.append(StateFilter(**condition))
                except TypeError as error:
                    raise ValueError("state filter fields are invalid") from error
            else:
                raise ValueError("state filters must be objects")
        object.__setattr__(self, "state_filters", tuple(normalized_filters))

    @property
    def resampling_origin(self) -> str:
        return "epoch"

    @property
    def resampling_closed(self) -> str:
        return "right"

    @property
    def resampling_label(self) -> str:
        return "right"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_interval_minutes": self.sample_interval_minutes,
            "resampling_method": self.resampling_method,
            "resampling_origin": self.resampling_origin,
            "resampling_closed": self.resampling_closed,
            "resampling_label": self.resampling_label,
            "filter_method": self.filter_method,
            "smoothing_window_minutes": self.smoothing_window_minutes,
            "gap_threshold_minutes": self.gap_threshold_minutes,
            "max_lag_minutes": self.max_lag_minutes,
            "lag_step_minutes": self.lag_step_minutes,
            "state_filters": [asdict(condition) for condition in self.state_filters],
        }


@dataclass(frozen=True)
class PreprocessingSummary:
    source_row_count: int
    source_interval_minutes: float | None
    target_interval_minutes: int
    resampling_method: str
    resampled_row_count: int
    empty_bin_count: int
    raw_segment_count: int
    raw_gap_count: int
    raw_gap_ranges: tuple[dict[str, str], ...]
    filter_method: str
    filter_window_minutes: int
    filter_warmup_loss: int
    state_filter_input_rows: int
    state_filter_output_rows: int
    lag_max_minutes: int
    lag_step_minutes: int
    lag_warmup_loss: int
    final_dynamic_row_count: int
    dynamic_feature_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["raw_gap_ranges"] = list(self.raw_gap_ranges)
        return value


@dataclass(frozen=True)
class PreprocessingResult:
    dynamic: pd.DataFrame
    segment_ids: pd.Series
    raw_segment_ids: pd.Series
    summary: PreprocessingSummary
    raw: pd.DataFrame | None = None
    resampled: pd.DataFrame | None = None
    filtered: pd.DataFrame | None = None
    state_filtered: pd.DataFrame | None = None


class PreprocessingQualityError(ValueError):
    def __init__(self, report: QualityReport) -> None:
        super().__init__("data quality review required")
        self.report = report


def preprocessing_config_from_mapping(value: Mapping[str, Any]) -> PreprocessingConfig:
    for field, expected in (
        ("resampling_origin", "epoch"),
        ("resampling_closed", "right"),
        ("resampling_label", "right"),
    ):
        if field in value and value[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    return PreprocessingConfig(
        sample_interval_minutes=_integer_config_value(
            value["sample_interval_minutes"], "sample interval"
        ),
        smoothing_window_minutes=_integer_config_value(
            value["smoothing_window_minutes"], "smoothing window"
        ),
        max_lag_minutes=_integer_config_value(
            value["max_lag_minutes"], "maximum lag"
        ),
        lag_step_minutes=_integer_config_value(value["lag_step_minutes"], "lag step"),
        resampling_method=str(value.get("resampling_method", "none")),
        filter_method=str(value.get("filter_method", "trailing_mean")),
        gap_threshold_minutes=(
            None
            if value.get("gap_threshold_minutes") is None
            else float(value["gap_threshold_minutes"])
        ),
        state_filters=tuple(value.get("state_filters", ())),
    )


def _integer_config_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def infer_segment_ids(
    index: pd.DatetimeIndex,
    sample_interval_minutes: int,
    gap_threshold_minutes: float | None = None,
) -> pd.Series:
    """Identify physical segments while preserving the existing grid contract."""
    _validate_index(index)
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    threshold = pd.Timedelta(
        minutes=gap_threshold_minutes or sample_interval_minutes
    )
    intervals = index.to_series().diff()
    valid_intervals = intervals.dropna()
    ratios = valid_intervals / expected
    on_grid = np.isclose(ratios, np.round(ratios))
    if ((valid_intervals < expected) | ~on_grid).any():
        raise ValueError("timestamps must follow the configured sampling grid")
    return intervals.gt(threshold).cumsum().astype(int).set_axis(index)


def segment_raw_data(
    index: pd.DatetimeIndex, config: PreprocessingConfig
) -> tuple[pd.Series, float | None, tuple[dict[str, str], ...]]:
    """Identify gaps on the raw time axis before any resampling."""
    _validate_index(index)
    intervals = index.to_series().diff()
    source_interval = _source_interval_minutes(index)
    if config.resampling_method == "none":
        segments = infer_segment_ids(
            index, config.sample_interval_minutes, config.gap_threshold_minutes
        )
    else:
        if (
            source_interval is not None
            and config.sample_interval_minutes < source_interval
        ):
            raise ValueError(
                "target sampling interval must not be shorter than source interval"
            )
        threshold_minutes = (
            config.gap_threshold_minutes
            if config.gap_threshold_minutes is not None
            else source_interval or config.sample_interval_minutes
        )
        expected = pd.Timedelta(minutes=source_interval or threshold_minutes)
        valid_intervals = intervals.dropna()
        ratios = valid_intervals / expected
        on_grid = np.isclose(ratios, np.round(ratios))
        if ((valid_intervals < expected) | ~on_grid).any():
            raise ValueError("timestamps must follow the source sampling grid")
        segments = intervals.gt(
            pd.Timedelta(minutes=threshold_minutes)
        ).cumsum().astype(int).set_axis(index)
    gaps = intervals[segments.diff().fillna(0).gt(0)]
    ranges = tuple(
        {
            "start": index[index.get_loc(timestamp) - 1].isoformat(),
            "end": timestamp.isoformat(),
        }
        for timestamp in gaps.index
    )
    return segments, source_interval, ranges


def resample_segment(
    frame: pd.DataFrame,
    method: str,
    sample_interval_minutes: int,
) -> tuple[pd.DataFrame, int]:
    """Resample one physical segment using fixed epoch-anchored right buckets."""
    _validate_index(frame.index)
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    if method not in _RESAMPLING_METHODS:
        raise ValueError(f"unsupported resampling method: {method}")
    if method == "none" or frame.empty:
        return frame.copy(), 0
    source_interval = _source_interval_minutes(frame.index)
    if source_interval is not None and sample_interval_minutes < source_interval:
        raise ValueError(
            "target sampling interval must not be shorter than source interval"
        )
    numeric = frame.apply(pd.to_numeric, errors="raise")
    rule = f"{sample_interval_minutes}min"
    kwargs = {"origin": "epoch", "closed": "right", "label": "right"}
    resampler = numeric.resample(rule, **kwargs)
    counts = resampler.size()
    if method == "mean":
        result = resampler.mean()
    elif method == "median":
        result = resampler.median()
    else:
        result = resampler.aggregate(
            lambda values: values.iloc[-1] if len(values) else np.nan
        )
    return result, int(counts.eq(0).sum())


def filter_window_rows(config: PreprocessingConfig) -> int:
    if config.filter_method == "none":
        return 1
    return config.smoothing_window_minutes // config.sample_interval_minutes


def filter_segment(
    frame: pd.DataFrame, method: str, window_rows: int
) -> pd.DataFrame:
    """Apply one causal trailing filter to a single physical segment."""
    if method not in _FILTER_METHODS:
        raise ValueError(f"unsupported filter method: {method}")
    if method == "none":
        return frame.copy()
    if window_rows < 1:
        raise ValueError("filter window rows must be positive")
    rolling = frame.rolling(window_rows, min_periods=window_rows)
    return rolling.mean() if method == "trailing_mean" else rolling.median()


def apply_state_filters(
    frame: pd.DataFrame, conditions: Sequence[StateFilter]
) -> pd.DataFrame:
    if not conditions:
        return frame.copy()
    keep = pd.Series(True, index=frame.index)
    for condition in conditions:
        if condition.column not in frame.columns:
            raise ValueError(f"missing state filter column: {condition.column}")
        values = pd.to_numeric(frame[condition.column], errors="raise")
        current = values.notna()
        if condition.minimum is not None:
            current &= values >= condition.minimum
        if condition.maximum is not None:
            current &= values <= condition.maximum
        keep &= current
    filtered = frame.loc[keep]
    if filtered.empty:
        raise ValueError("state filters removed all rows")
    return filtered


def preprocess_window(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
    *,
    validate_quality: bool = True,
    include_intermediates: bool = False,
    include_variability: bool = False,
    preserve_columns: Sequence[str] = (),
) -> PreprocessingResult:
    """Execute the single causal preprocessing contract for one independent window."""
    _validate_index(frame.index)
    required = [
        *tag_columns,
        *(item.column for item in config.state_filters),
        *preserve_columns,
    ]
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing tag columns: {', '.join(missing)}")
    raw = frame.loc[:, required].copy()
    raw_segments, source_interval, gap_ranges = segment_raw_data(raw.index, config)

    resampled_parts: list[pd.DataFrame] = []
    resampled_segment_parts: list[pd.Series] = []
    empty_bins = 0
    for segment_id in raw_segments.unique():
        segment = raw.loc[raw_segments.eq(segment_id)]
        resampled, segment_empty_bins = resample_segment(
            segment, config.resampling_method, config.sample_interval_minutes
        )
        resampled_parts.append(resampled)
        resampled_segment_parts.append(pd.Series(segment_id, index=resampled.index))
        empty_bins += segment_empty_bins
    resampled = pd.concat(resampled_parts).sort_index() if resampled_parts else raw
    resampled_segments = (
        pd.concat(resampled_segment_parts).reindex(resampled.index).astype(int)
        if resampled_segment_parts
        else pd.Series(dtype=int, index=resampled.index)
    )
    if resampled.index.has_duplicates:
        raise ValueError("resampling produced duplicate timestamps across segments")

    quality_frame = resampled.reset_index(names="__timestamp__")
    report = inspect_data_quality(
        quality_frame,
        "__timestamp__",
        tag_columns,
        engineering_ranges=engineering_ranges,
        expected_interval_minutes=config.sample_interval_minutes,
        include_variability=include_variability,
    )
    if validate_quality and not report.can_train:
        raise PreprocessingQualityError(report)

    numeric_resampled = resampled.apply(pd.to_numeric, errors="coerce")
    window_rows = filter_window_rows(config)
    filtered = numeric_resampled.groupby(resampled_segments, sort=False).transform(
        lambda group: filter_segment(group, config.filter_method, window_rows)
    )
    filter_warmup_loss = int(filtered.loc[:, tag_columns].isna().any(axis=1).sum())
    state_filter_input_rows = len(filtered)
    state_filtered = apply_state_filters(filtered, config.state_filters)
    state_filter_output_rows = len(state_filtered)
    original_ids = resampled_segments.reindex(state_filtered.index)
    breaks = original_ids.ne(original_ids.shift()) | state_filtered.index.to_series().diff().gt(
        pd.Timedelta(minutes=config.sample_interval_minutes)
    )
    final_segments = breaks.cumsum().astype(int).set_axis(state_filtered.index) - 1

    lag_input_rows = int(
        state_filtered.loc[:, tag_columns].notna().all(axis=1).sum()
    )
    dynamic = _build_lag_matrix(state_filtered, tag_columns, config, final_segments)
    lag_warmup_loss = lag_input_rows - len(dynamic)
    summary = PreprocessingSummary(
        source_row_count=len(raw),
        source_interval_minutes=source_interval,
        target_interval_minutes=config.sample_interval_minutes,
        resampling_method=config.resampling_method,
        resampled_row_count=len(resampled),
        empty_bin_count=empty_bins,
        raw_segment_count=int(raw_segments.nunique()) if len(raw_segments) else 0,
        raw_gap_count=len(gap_ranges),
        raw_gap_ranges=gap_ranges,
        filter_method=config.filter_method,
        filter_window_minutes=config.smoothing_window_minutes,
        filter_warmup_loss=filter_warmup_loss,
        state_filter_input_rows=state_filter_input_rows,
        state_filter_output_rows=state_filter_output_rows,
        lag_max_minutes=config.max_lag_minutes,
        lag_step_minutes=config.lag_step_minutes,
        lag_warmup_loss=lag_warmup_loss,
        final_dynamic_row_count=len(dynamic),
        dynamic_feature_count=dynamic.shape[1],
    )
    return PreprocessingResult(
        dynamic=dynamic,
        segment_ids=resampled_segments,
        raw_segment_ids=raw_segments,
        summary=summary,
        raw=raw if include_intermediates else None,
        resampled=numeric_resampled if include_intermediates else None,
        filtered=filtered if include_intermediates else None,
        state_filtered=state_filtered if include_intermediates else None,
    )


def build_dynamic_matrix(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    segment_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Compatibility entry for causal filtering and Lag expansion."""
    _validate_index(frame.index)
    missing = [tag for tag in tag_columns if tag not in frame.columns]
    if missing:
        raise ValueError(f"missing tag columns: {', '.join(missing)}")
    numeric = frame.loc[:, tag_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("model inputs contain non-finite values")
    if segment_ids is None:
        segment_ids = infer_segment_ids(
            frame.index,
            config.sample_interval_minutes,
            config.gap_threshold_minutes,
        )
    else:
        segment_ids = segment_ids.reindex(frame.index)
        if segment_ids.isna().any():
            raise ValueError("segment identifiers must cover every timestamp")
    window_rows = filter_window_rows(config)
    filtered = numeric.groupby(segment_ids, sort=False).transform(
        lambda group: filter_segment(group, config.filter_method, window_rows)
    )
    return _build_lag_matrix(filtered, tag_columns, config, segment_ids)


def _build_lag_matrix(
    filtered: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    segment_ids: pd.Series,
) -> pd.DataFrame:
    dynamic_columns: dict[str, pd.Series] = {}
    for lag_minutes in range(
        0, config.max_lag_minutes + 1, config.lag_step_minutes
    ):
        lag_rows = lag_minutes // config.sample_interval_minutes
        lagged = filtered.groupby(segment_ids, sort=False).shift(lag_rows)
        for tag in tag_columns:
            dynamic_columns[f"{tag}__lag_{lag_minutes:03d}min"] = lagged[tag]
    return pd.DataFrame(dynamic_columns, index=filtered.index).dropna(how="any")


def _source_interval_minutes(index: pd.DatetimeIndex) -> float | None:
    intervals = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    return float(intervals.mode().iloc[0]) if not intervals.empty else None


def _validate_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("data index must be a DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("timestamps must be sorted and unique")
