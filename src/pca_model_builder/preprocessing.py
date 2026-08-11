from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .quality import QualityReport, inspect_data_quality


_RESAMPLING_METHODS = {"none", "mean", "median", "last"}
_FILTER_METHODS = {"none", "first_order", "trailing_mean", "trailing_median"}


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
    filter_method: str = "none"
    first_order_alpha: float | None = None
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
        if self.filter_method == "first_order":
            if (
                not isinstance(self.first_order_alpha, (int, float))
                or isinstance(self.first_order_alpha, bool)
                or not 0 < float(self.first_order_alpha) <= 1
            ):
                raise ValueError("first_order_alpha must be in (0, 1]")
        elif self.first_order_alpha is not None and (
            not isinstance(self.first_order_alpha, (int, float))
            or isinstance(self.first_order_alpha, bool)
            or not np.isfinite(self.first_order_alpha)
        ):
            raise ValueError("first_order_alpha must be finite or null")
        if self.max_lag_minutes % self.lag_step_minutes:
            raise ValueError("maximum lag must be an integer multiple of lag step")
        if self.lag_step_minutes % self.sample_interval_minutes:
            raise ValueError("lag step must be an integer multiple of sampling")
        if self.filter_method in {"trailing_mean", "trailing_median"}:
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
            "first_order_alpha": (
                float(self.first_order_alpha)
                if self.filter_method == "first_order"
                else None
            ),
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
    resampling_row_reduction: int
    empty_bin_count: int
    partial_resampling_bin_loss: int
    partial_resampling_row_loss: int
    raw_segment_count: int
    raw_gap_count: int
    raw_gap_ranges: tuple[dict[str, str], ...]
    filter_method: str
    filter_window_minutes: int
    filter_warmup_loss: int
    filter_context_invalid_loss: int
    state_filter_input_rows: int
    state_filter_output_rows: int
    lag_max_minutes: int
    lag_step_minutes: int
    lag_warmup_loss: int
    lag_context_invalid_loss: int
    input_invalid_loss: int
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
    post_invalid_segment_ids: pd.Series
    summary: PreprocessingSummary
    empty_bin_mask: pd.Series
    filter_warmup_mask: pd.Series
    filter_context_invalid_mask: pd.Series
    final_segment_ids: pd.Series
    lag_warmup_mask: pd.Series
    lag_context_invalid_mask: pd.Series
    input_invalid_mask: pd.Series
    engineering_range_mask: pd.Series
    engineering_range_loss_by_tag: Mapping[str, int]
    dynamic_valid_mask: pd.Series
    partial_resampling_bin_loss_by_segment: Mapping[int, int]
    partial_resampling_row_loss_by_segment: Mapping[int, int]
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
        first_order_alpha=value.get("first_order_alpha"),
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
    result, counts = _resample_segment_data(
        frame, method, sample_interval_minutes
    )
    return result, int(counts.eq(0).sum())


def _resample_segment_data(
    frame: pd.DataFrame,
    method: str,
    sample_interval_minutes: int,
    *,
    numeric_errors: str = "raise",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return one resampled segment and the actual source count per bucket."""
    _validate_index(frame.index)
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    if method not in _RESAMPLING_METHODS:
        raise ValueError(f"unsupported resampling method: {method}")
    if method == "none" or frame.empty:
        return frame.copy(), pd.Series(1, index=frame.index, dtype=int)
    source_interval = _source_interval_minutes(frame.index)
    if source_interval is not None and sample_interval_minutes < source_interval:
        raise ValueError(
            "target sampling interval must not be shorter than source interval"
        )
    if numeric_errors not in {"raise", "coerce"}:
        raise ValueError("resampling numeric conversion mode is invalid")
    numeric = frame.apply(pd.to_numeric, errors=numeric_errors)
    rule = f"{sample_interval_minutes}min"
    resampler = numeric.resample(rule, **_resampling_kwargs())
    counts = resampler.size()
    if method == "mean":
        result = resampler.mean()
    elif method == "median":
        result = resampler.median()
    else:
        result = resampler.aggregate(
            lambda values: values.iloc[-1] if len(values) else np.nan
        )
    return result, counts


def _resampling_kwargs() -> dict[str, str]:
    return {"origin": "epoch", "closed": "right", "label": "right"}


def _engineering_range_bucket_mask(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    engineering_ranges: Mapping[str, tuple[float, float]] | None,
    method: str,
    sample_interval_minutes: int,
) -> pd.DataFrame:
    """Mark resampling buckets containing a finite engineering-range violation."""
    mask = pd.DataFrame(False, index=frame.index, columns=tag_columns)
    numeric = frame.loc[:, tag_columns].apply(pd.to_numeric, errors="coerce")
    for tag, (lower, upper) in (engineering_ranges or {}).items():
        if tag in mask:
            values = numeric[tag]
            mask[tag] = np.isfinite(values) & ((values < lower) | (values > upper))
    if method == "none" or frame.empty:
        return mask
    return (
        mask.resample(f"{sample_interval_minutes}min", **_resampling_kwargs())
        .max()
        .astype(bool)
    )


def filter_window_rows(config: PreprocessingConfig) -> int:
    if config.filter_method in {"none", "first_order"}:
        return 1
    return config.smoothing_window_minutes // config.sample_interval_minutes


def filter_segment(
    frame: pd.DataFrame, method: str, window_rows: int, first_order_alpha: float | None = None
) -> pd.DataFrame:
    """Apply one causal trailing filter to a single physical segment."""
    if method not in _FILTER_METHODS:
        raise ValueError(f"unsupported filter method: {method}")
    if method == "none":
        return frame.copy()
    if method == "first_order":
        if first_order_alpha is None:
            raise ValueError("first_order_alpha is required for first_order filtering")
        return frame.ewm(alpha=float(first_order_alpha), adjust=False).mean()
    if window_rows < 1:
        raise ValueError("filter window rows must be positive")
    rolling = frame.rolling(window_rows, min_periods=window_rows)
    return rolling.mean() if method == "trailing_mean" else rolling.median()


def filter_warmup_mask(
    index: pd.DatetimeIndex,
    segment_ids: pd.Series,
    method: str,
    window_rows: int,
) -> pd.Series:
    """Identify only rows unavailable because causal filter history is incomplete."""
    mask = pd.Series(False, index=index)
    if method in {"none", "first_order"} or window_rows <= 1:
        return mask
    aligned = segment_ids.reindex(index)
    for segment_id in aligned.drop_duplicates():
        positions = np.flatnonzero(aligned.eq(segment_id).to_numpy())
        mask.iloc[positions[: min(window_rows - 1, len(positions))]] = True
    return mask


def apply_state_filters(
    frame: pd.DataFrame, conditions: Sequence[StateFilter], *, allow_empty: bool = False
) -> pd.DataFrame:
    if not conditions or frame.empty:
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
    if filtered.empty and not allow_empty:
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
    resampling_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    allow_empty_state_filter: bool = False,
    exclude_engineering_range: bool = False,
    preprocessing_semantics: str = "schema5",
) -> PreprocessingResult:
    """Execute the single causal preprocessing contract for one independent window."""
    if preprocessing_semantics == "schema5":
        return _preprocess_window_schema5(
            frame,
            tag_columns,
            config,
            engineering_ranges,
            validate_quality=validate_quality,
            include_intermediates=include_intermediates,
            include_variability=include_variability,
            preserve_columns=preserve_columns,
            resampling_window=resampling_window,
            allow_empty_state_filter=allow_empty_state_filter,
            exclude_engineering_range=exclude_engineering_range,
        )
    if preprocessing_semantics != "legacy":
        raise ValueError("unsupported preprocessing semantics")
    return _preprocess_window_legacy(
        frame,
        tag_columns,
        config,
        engineering_ranges,
        validate_quality=validate_quality,
        include_intermediates=include_intermediates,
        include_variability=include_variability,
        preserve_columns=preserve_columns,
        resampling_window=resampling_window,
        allow_empty_state_filter=allow_empty_state_filter,
    )


def _preprocess_window_legacy(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
    *,
    validate_quality: bool = True,
    include_intermediates: bool = False,
    include_variability: bool = False,
    preserve_columns: Sequence[str] = (),
    resampling_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    allow_empty_state_filter: bool = False,
) -> PreprocessingResult:
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
    empty_bin_parts: list[pd.Series] = []
    empty_bins = 0
    partial_bin_loss_by_segment: dict[int, int] = {}
    partial_row_loss_by_segment: dict[int, int] = {}
    source_rows_in_complete_bins = 0
    non_empty_resampled_bins = 0
    for segment_id in raw_segments.unique():
        segment = raw.loc[raw_segments.eq(segment_id)]
        resampled, counts = _resample_segment_data(
            segment, config.resampling_method, config.sample_interval_minutes
        )
        partial_loss = 0
        if resampling_window is not None and config.resampling_method != "none":
            window_start, window_end = resampling_window
            interval = pd.Timedelta(minutes=config.sample_interval_minutes)
            complete = (resampled.index - interval >= window_start) & (
                resampled.index <= window_end
            )
            partial_loss = int((~complete).sum())
            partial_row_loss_by_segment[int(segment_id)] = int(
                counts.loc[~complete].sum()
            )
            resampled = resampled.loc[complete]
            counts = counts.loc[complete]
        else:
            partial_row_loss_by_segment[int(segment_id)] = 0
        partial_bin_loss_by_segment[int(segment_id)] = partial_loss
        if config.resampling_method != "none":
            source_rows_in_complete_bins += int(counts.sum())
            non_empty_resampled_bins += int(counts.gt(0).sum())
        resampled_parts.append(resampled)
        resampled_segment_parts.append(pd.Series(segment_id, index=resampled.index))
        empty_bin_parts.append(counts.eq(0))
        empty_bins += int(counts.eq(0).sum())
    resampled = pd.concat(resampled_parts).sort_index() if resampled_parts else raw
    resampled_segments = (
        pd.concat(resampled_segment_parts).reindex(resampled.index).astype(int)
        if resampled_segment_parts
        else pd.Series(dtype=int, index=resampled.index)
    )
    empty_bin_mask = (
        pd.concat(empty_bin_parts).reindex(resampled.index).astype(bool)
        if empty_bin_parts
        else pd.Series(dtype=bool, index=resampled.index)
    )
    if resampled.index.has_duplicates:
        raise ValueError("resampling produced duplicate timestamps across segments")

    if not resampled.empty:
        quality_frame = resampled.reset_index(names="__timestamp__")
        report = inspect_data_quality(
            quality_frame,
            "__timestamp__",
            tag_columns,
            engineering_ranges=engineering_ranges,
            expected_interval_minutes=config.sample_interval_minutes,
            include_variability=include_variability,
        )
        if validate_quality and (
            not report.can_train
            or any(issue.code == "engineering_range" for issue in report.issues)
        ):
            raise PreprocessingQualityError(report)

    numeric_resampled = resampled.apply(pd.to_numeric, errors="coerce")
    window_rows = filter_window_rows(config)
    filtered = (
        numeric_resampled.groupby(resampled_segments, sort=False).transform(
            lambda group: filter_segment(
                group, config.filter_method, window_rows, config.first_order_alpha
            )
        )
        if not numeric_resampled.empty
        else numeric_resampled.copy()
    )
    structural_filter_warmup = filter_warmup_mask(
        filtered.index, resampled_segments, config.filter_method, window_rows
    )
    input_is_valid = pd.Series(
        np.isfinite(numeric_resampled.loc[:, tag_columns].to_numpy(dtype=float)).all(
            axis=1
        ),
        index=numeric_resampled.index,
    )
    state_filter_input_rows = len(filtered)
    state_filtered = apply_state_filters(
        filtered, config.state_filters, allow_empty=allow_empty_state_filter
    )
    state_filter_output_rows = len(state_filtered)
    original_ids = resampled_segments.reindex(state_filtered.index)
    breaks = original_ids.ne(original_ids.shift()) | state_filtered.index.to_series().diff().gt(
        pd.Timedelta(minutes=config.sample_interval_minutes)
    )
    final_segments = breaks.cumsum().astype(int).set_axis(state_filtered.index) - 1

    final_empty_bins = empty_bin_mask.reindex(state_filtered.index, fill_value=False)
    final_input_valid = input_is_valid.reindex(state_filtered.index, fill_value=False)
    input_invalid_mask = ~final_empty_bins & ~final_input_valid
    final_filter_structural = structural_filter_warmup.reindex(
        state_filtered.index, fill_value=False
    )
    warmup_mask = final_filter_structural & final_input_valid & ~final_empty_bins
    filtered_is_valid = pd.Series(
        np.isfinite(filtered.loc[:, tag_columns].to_numpy(dtype=float)).all(axis=1),
        index=filtered.index,
    ).reindex(state_filtered.index, fill_value=False)
    filter_context_invalid_mask = (
        ~final_empty_bins
        & ~input_invalid_mask
        & ~warmup_mask
        & ~filtered_is_valid
    )
    eligible_for_lag = ~(
        final_empty_bins
        | input_invalid_mask
        | warmup_mask
        | filter_context_invalid_mask
    )
    lag_features = _lag_feature_frame(
        state_filtered, tag_columns, config, final_segments
    )
    lag_warmup_mask = _lag_warmup_mask(
        state_filtered.index, final_segments, config, ~eligible_for_lag
    )
    feature_is_valid = pd.Series(
        np.isfinite(lag_features.to_numpy(dtype=float)).all(axis=1),
        index=lag_features.index,
    )
    lag_context_invalid_mask = (
        ~feature_is_valid
        & eligible_for_lag
        & ~lag_warmup_mask
        & (config.max_lag_minutes > 0)
    )
    dynamic_valid_mask = eligible_for_lag & feature_is_valid
    if _mask_union_count(
        final_empty_bins,
        input_invalid_mask,
        warmup_mask,
        filter_context_invalid_mask,
        lag_warmup_mask,
        lag_context_invalid_mask,
        dynamic_valid_mask,
    ) != len(state_filtered):
        raise ValueError("preprocessing loss masks do not classify every row")
    dynamic = lag_features.loc[dynamic_valid_mask]
    lag_warmup_loss = int(lag_warmup_mask.sum())
    summary = PreprocessingSummary(
        source_row_count=len(raw),
        source_interval_minutes=source_interval,
        target_interval_minutes=config.sample_interval_minutes,
        resampling_method=config.resampling_method,
        resampled_row_count=len(resampled),
        resampling_row_reduction=(
            0
            if config.resampling_method == "none"
            else source_rows_in_complete_bins - non_empty_resampled_bins
        ),
        empty_bin_count=empty_bins,
        partial_resampling_bin_loss=sum(partial_bin_loss_by_segment.values()),
        partial_resampling_row_loss=sum(partial_row_loss_by_segment.values()),
        raw_segment_count=int(raw_segments.nunique()) if len(raw_segments) else 0,
        raw_gap_count=len(gap_ranges),
        raw_gap_ranges=gap_ranges,
        filter_method=config.filter_method,
        filter_window_minutes=config.smoothing_window_minutes,
        filter_warmup_loss=int(warmup_mask.sum()),
        filter_context_invalid_loss=int(filter_context_invalid_mask.sum()),
        state_filter_input_rows=state_filter_input_rows,
        state_filter_output_rows=state_filter_output_rows,
        lag_max_minutes=config.max_lag_minutes,
        lag_step_minutes=config.lag_step_minutes,
        lag_warmup_loss=lag_warmup_loss,
        lag_context_invalid_loss=int(lag_context_invalid_mask.sum()),
        input_invalid_loss=int(input_invalid_mask.sum()),
        final_dynamic_row_count=len(dynamic),
        dynamic_feature_count=dynamic.shape[1],
    )
    return PreprocessingResult(
        dynamic=dynamic,
        segment_ids=resampled_segments,
        raw_segment_ids=raw_segments,
        post_invalid_segment_ids=resampled_segments.reindex(state_filtered.index),
        summary=summary,
        empty_bin_mask=empty_bin_mask,
        filter_warmup_mask=warmup_mask,
        filter_context_invalid_mask=filter_context_invalid_mask,
        final_segment_ids=final_segments,
        lag_warmup_mask=lag_warmup_mask,
        lag_context_invalid_mask=lag_context_invalid_mask,
        input_invalid_mask=input_invalid_mask,
        engineering_range_mask=pd.Series(False, index=resampled.index),
        engineering_range_loss_by_tag={},
        dynamic_valid_mask=dynamic_valid_mask,
        partial_resampling_bin_loss_by_segment=partial_bin_loss_by_segment,
        partial_resampling_row_loss_by_segment=partial_row_loss_by_segment,
        raw=raw if include_intermediates else None,
        resampled=numeric_resampled if include_intermediates else None,
        filtered=filtered if include_intermediates else None,
        state_filtered=state_filtered if include_intermediates else None,
    )


def _preprocess_window_schema5(
    frame: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    engineering_ranges: Mapping[str, tuple[float, float]] | None = None,
    *,
    validate_quality: bool = True,
    include_intermediates: bool = False,
    include_variability: bool = False,
    preserve_columns: Sequence[str] = (),
    resampling_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    allow_empty_state_filter: bool = False,
    exclude_engineering_range: bool = False,
) -> PreprocessingResult:
    """Schema 5: discard invalid resampled inputs before segment-local filtering."""
    _validate_index(frame.index)
    state_columns = [item.column for item in config.state_filters]
    required = list(dict.fromkeys([*tag_columns, *state_columns, *preserve_columns]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing tag columns: {', '.join(missing)}")
    raw = frame.loc[:, required].copy()
    raw_segments, source_interval, gap_ranges = segment_raw_data(raw.index, config)

    resampled_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.Series] = []
    empty_parts: list[pd.Series] = []
    engineering_parts: list[pd.DataFrame] = []
    partial_bin_loss_by_segment: dict[int, int] = {}
    partial_row_loss_by_segment: dict[int, int] = {}
    source_rows_in_complete_bins = 0
    non_empty_resampled_bins = 0
    for segment_id in raw_segments.unique():
        segment = raw.loc[raw_segments.eq(segment_id)]
        resampled, counts = _resample_segment_data(
            segment,
            config.resampling_method,
            config.sample_interval_minutes,
            numeric_errors="coerce",
        )
        engineering = _engineering_range_bucket_mask(
            segment,
            tag_columns,
            engineering_ranges,
            config.resampling_method,
            config.sample_interval_minutes,
        )
        partial_loss = 0
        if resampling_window is not None and config.resampling_method != "none":
            start, end = resampling_window
            interval = pd.Timedelta(minutes=config.sample_interval_minutes)
            complete = (resampled.index - interval >= start) & (resampled.index <= end)
            partial_loss = int((~complete).sum())
            partial_row_loss_by_segment[int(segment_id)] = int(counts.loc[~complete].sum())
            resampled, counts = resampled.loc[complete], counts.loc[complete]
            engineering = engineering.loc[complete]
        else:
            partial_row_loss_by_segment[int(segment_id)] = 0
        partial_bin_loss_by_segment[int(segment_id)] = partial_loss
        if config.resampling_method != "none":
            source_rows_in_complete_bins += int(counts.sum())
            non_empty_resampled_bins += int(counts.gt(0).sum())
        resampled_parts.append(resampled)
        segment_parts.append(pd.Series(segment_id, index=resampled.index))
        empty_parts.append(counts.eq(0))
        engineering_parts.append(engineering)

    resampled = pd.concat(resampled_parts).sort_index() if resampled_parts else raw.iloc[0:0]
    resampled_segments = pd.concat(segment_parts).reindex(resampled.index).astype(int)
    empty_bin_mask = pd.concat(empty_parts).reindex(resampled.index).astype(bool)
    raw_engineering_mask = pd.concat(engineering_parts).reindex(
        resampled.index, fill_value=False
    )
    if resampled.index.has_duplicates:
        raise ValueError("resampling produced duplicate timestamps across segments")
    if not resampled.empty and validate_quality:
        report = inspect_data_quality(
            resampled.reset_index(names="__timestamp__"),
            "__timestamp__",
            tag_columns,
            engineering_ranges=engineering_ranges,
            expected_interval_minutes=config.sample_interval_minutes,
            include_variability=include_variability,
        )
        if any(
            issue.severity == "error"
            and issue.code not in {"missing_value", "non_numeric_value", "non_finite_value"}
            for issue in report.issues
        ):
            raise PreprocessingQualityError(report)

    numeric_resampled = resampled.apply(pd.to_numeric, errors="coerce")
    input_columns = [*tag_columns, *state_columns]
    input_valid = pd.Series(
        np.isfinite(numeric_resampled.loc[:, input_columns].to_numpy(dtype=float)).all(axis=1),
        index=resampled.index,
    )
    # Empty buckets are a distinct resampling loss, never an input-invalid loss.
    input_invalid_mask = ~empty_bin_mask & ~input_valid
    engineering_range_mask = pd.Series(False, index=resampled.index)
    engineering_range_loss_by_tag: dict[str, int] = {}
    if exclude_engineering_range:
        engineering_eligible = input_valid & ~empty_bin_mask
        for tag, (lower, upper) in (engineering_ranges or {}).items():
            if tag not in tag_columns:
                continue
            outside = engineering_eligible & raw_engineering_mask[tag]
            engineering_range_loss_by_tag[tag] = int(outside.sum())
            engineering_range_mask |= outside
    usable = input_valid & ~empty_bin_mask & ~engineering_range_mask
    usable_frame = numeric_resampled.loc[usable]
    if len(resampled) != (
        int(empty_bin_mask.sum())
        + int(input_invalid_mask.sum())
        + int(engineering_range_mask.sum())
        + len(usable_frame)
    ):
        raise ValueError("resampled preprocessing losses do not close")
    usable_raw_segments = resampled_segments.loc[usable]
    post_invalid_segments = _resegment_remaining(
        usable_frame.index,
        usable_raw_segments,
        config,
        forced_boundary_mask=engineering_range_mask,
    )

    window_rows = filter_window_rows(config)
    # State exclusions create a new causal boundary for every filter method.
    state_filter_input_rows = len(usable_frame)
    state_filtered = apply_state_filters(
        usable_frame, config.state_filters, allow_empty=allow_empty_state_filter
    )
    state_filter_output_rows = len(state_filtered)
    final_segments = _resegment_remaining(
        state_filtered.index,
        post_invalid_segments.reindex(state_filtered.index),
        config,
        source_index=usable_frame.index,
    )
    filtered = (
        state_filtered.groupby(final_segments, sort=False).transform(
            lambda group: filter_segment(
                group, config.filter_method, window_rows, config.first_order_alpha
            )
        )
        if not state_filtered.empty
        else state_filtered.copy()
    )
    structural_filter_warmup = filter_warmup_mask(
        filtered.index, final_segments, config.filter_method, window_rows
    )
    state_filtered = filtered
    final_filter_structural = structural_filter_warmup.reindex(
        state_filtered.index, fill_value=False
    )
    filtered_is_valid = pd.Series(
        np.isfinite(filtered.loc[:, tag_columns].to_numpy(dtype=float)).all(axis=1),
        index=filtered.index,
    ).reindex(state_filtered.index, fill_value=False)
    warmup_mask = final_filter_structural
    filter_context_invalid_mask = ~warmup_mask & ~filtered_is_valid
    eligible_for_lag = ~(warmup_mask | filter_context_invalid_mask)
    lag_features = _lag_feature_frame(state_filtered, tag_columns, config, final_segments)
    lag_warmup_mask = _lag_warmup_mask(
        state_filtered.index, final_segments, config, ~eligible_for_lag
    )
    feature_is_valid = pd.Series(
        np.isfinite(lag_features.to_numpy(dtype=float)).all(axis=1), index=lag_features.index
    )
    lag_context_invalid_mask = (
        ~feature_is_valid & eligible_for_lag & ~lag_warmup_mask & (config.max_lag_minutes > 0)
    )
    dynamic_valid_mask = eligible_for_lag & feature_is_valid
    if _mask_union_count(
        warmup_mask,
        filter_context_invalid_mask,
        lag_warmup_mask,
        lag_context_invalid_mask,
        dynamic_valid_mask,
    ) != len(state_filtered):
        raise ValueError("preprocessing loss masks do not classify every retained row")
    dynamic = lag_features.loc[dynamic_valid_mask]
    summary = PreprocessingSummary(
        source_row_count=len(raw),
        source_interval_minutes=source_interval,
        target_interval_minutes=config.sample_interval_minutes,
        resampling_method=config.resampling_method,
        resampled_row_count=len(resampled),
        resampling_row_reduction=(0 if config.resampling_method == "none" else source_rows_in_complete_bins - non_empty_resampled_bins),
        empty_bin_count=int(empty_bin_mask.sum()),
        partial_resampling_bin_loss=sum(partial_bin_loss_by_segment.values()),
        partial_resampling_row_loss=sum(partial_row_loss_by_segment.values()),
        raw_segment_count=int(raw_segments.nunique()) if len(raw_segments) else 0,
        raw_gap_count=len(gap_ranges),
        raw_gap_ranges=gap_ranges,
        filter_method=config.filter_method,
        filter_window_minutes=config.smoothing_window_minutes,
        filter_warmup_loss=int(warmup_mask.sum()),
        filter_context_invalid_loss=int(filter_context_invalid_mask.sum()),
        state_filter_input_rows=state_filter_input_rows,
        state_filter_output_rows=state_filter_output_rows,
        lag_max_minutes=config.max_lag_minutes,
        lag_step_minutes=config.lag_step_minutes,
        lag_warmup_loss=int(lag_warmup_mask.sum()),
        lag_context_invalid_loss=int(lag_context_invalid_mask.sum()),
        input_invalid_loss=int(input_invalid_mask.sum()),
        final_dynamic_row_count=len(dynamic),
        dynamic_feature_count=dynamic.shape[1],
    )
    return PreprocessingResult(
        dynamic=dynamic,
        segment_ids=resampled_segments,
        raw_segment_ids=raw_segments,
        post_invalid_segment_ids=post_invalid_segments,
        summary=summary,
        empty_bin_mask=empty_bin_mask,
        filter_warmup_mask=warmup_mask,
        filter_context_invalid_mask=filter_context_invalid_mask,
        final_segment_ids=final_segments,
        lag_warmup_mask=lag_warmup_mask,
        lag_context_invalid_mask=lag_context_invalid_mask,
        input_invalid_mask=input_invalid_mask,
        engineering_range_mask=engineering_range_mask,
        engineering_range_loss_by_tag=engineering_range_loss_by_tag,
        dynamic_valid_mask=dynamic_valid_mask,
        partial_resampling_bin_loss_by_segment=partial_bin_loss_by_segment,
        partial_resampling_row_loss_by_segment=partial_row_loss_by_segment,
        raw=raw if include_intermediates else None,
        resampled=numeric_resampled if include_intermediates else None,
        filtered=filtered if include_intermediates else None,
        state_filtered=state_filtered if include_intermediates else None,
    )


def _resegment_remaining(
    index: pd.DatetimeIndex,
    raw_segments: pd.Series,
    config: PreprocessingConfig,
    *,
    source_index: pd.DatetimeIndex | None = None,
    forced_boundary_mask: pd.Series | None = None,
) -> pd.Series:
    """Split retained rows at physical boundaries and deleted-row discontinuities."""
    if not len(index):
        return pd.Series(dtype=int, index=index)
    source = raw_segments.reindex(index)
    if source.isna().any():
        raise ValueError("raw segment identifiers must cover retained timestamps")
    threshold = pd.Timedelta(
        minutes=config.gap_threshold_minutes or config.sample_interval_minutes
    )
    breaks = source.ne(source.shift()) | index.to_series().diff().gt(threshold)
    if forced_boundary_mask is not None:
        forced = forced_boundary_mask.fillna(False).astype(bool)
        for timestamp in forced.index[forced.to_numpy(dtype=bool)]:
            next_position = index.searchsorted(timestamp, side="right")
            if next_position < len(index):
                breaks.iloc[next_position] = True
    if source_index is not None:
        positions = source_index.get_indexer(index)
        if (positions < 0).any():
            raise ValueError("source timestamps must cover retained timestamps")
        breaks |= pd.Series(
            np.r_[False, np.diff(positions) > 1], index=index
        )
    return (breaks.cumsum().astype(int) - 1).set_axis(index)


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
        lambda group: filter_segment(
            group, config.filter_method, window_rows, config.first_order_alpha
        )
    )
    return _build_lag_matrix(filtered, tag_columns, config, segment_ids)


def _build_lag_matrix(
    filtered: pd.DataFrame,
    tag_columns: Sequence[str],
    config: PreprocessingConfig,
    segment_ids: pd.Series,
) -> pd.DataFrame:
    return _lag_feature_frame(filtered, tag_columns, config, segment_ids).dropna(how="any")


def _lag_feature_frame(
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
    return pd.DataFrame(dynamic_columns, index=filtered.index)


def _lag_warmup_mask(
    index: pd.DatetimeIndex,
    segment_ids: pd.Series,
    config: PreprocessingConfig,
    excluded: pd.Series | None = None,
) -> pd.Series:
    mask = pd.Series(False, index=index)
    rows = config.max_lag_minutes // config.sample_interval_minutes
    if rows <= 0:
        return mask
    aligned = segment_ids.reindex(index)
    excluded_mask = (
        pd.Series(False, index=index)
        if excluded is None
        else excluded.reindex(index, fill_value=False)
    )
    for segment_id in aligned.drop_duplicates():
        positions = np.flatnonzero(aligned.eq(segment_id).to_numpy())
        mask.iloc[positions[: min(rows, len(positions))]] = True
    return mask & ~excluded_mask


def _mask_union_count(*masks: pd.Series) -> int:
    if not masks:
        return 0
    combined = pd.concat(masks, axis=1).astype(int)
    counts = combined.sum(axis=1)
    if not counts.eq(1).all():
        raise ValueError("preprocessing loss masks overlap")
    return len(counts)


def _source_interval_minutes(index: pd.DatetimeIndex) -> float | None:
    intervals = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    return float(intervals.mode().iloc[0]) if not intervals.empty else None


def _validate_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("data index must be a DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("timestamps must be sorted and unique")
