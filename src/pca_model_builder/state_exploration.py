from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from .clustering import cluster_model_scores
from .dpca import fit_dpca
from .preprocessing import PreprocessingConfig, PreprocessingResult, preprocess_window


DEFAULT_CLUSTER_COUNT = 4
DEFAULT_RANDOM_STATE = 0
DEFAULT_CANDIDATES_PER_CLUSTER = 3
DEFAULT_MINIMUM_CANDIDATE_DURATION_MINUTES = 30
DEFAULT_MAXIMUM_PLOT_POINTS = 1200
DEFAULT_CLUSTER_SAMPLE_COUNT_LOW_THRESHOLD = 3
DEFAULT_CANDIDATE_COUNT_WARNING_THRESHOLD = 1
DEFAULT_COVERAGE_WARNING_THRESHOLD = 0.5
DEFAULT_PREPROCESSING_LOSS_WARNING_THRESHOLD = 0.3
PERFORMANCE_DIRECTIONS = frozenset(
    {"higher_is_better", "lower_is_better", "target_range"}
)
DURATION_SEMANTICS = "coverage"


def _positive_integer(value: object, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _integer_value(value: object, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True)
class ExplorationConfig:
    cluster_count: int = DEFAULT_CLUSTER_COUNT
    random_state: int = DEFAULT_RANDOM_STATE
    minimum_candidate_duration_minutes: int = DEFAULT_MINIMUM_CANDIDATE_DURATION_MINUTES
    candidate_count_per_cluster: int = DEFAULT_CANDIDATES_PER_CLUSTER
    maximum_plot_points: int = DEFAULT_MAXIMUM_PLOT_POINTS

    def __post_init__(self) -> None:
        _positive_integer(self.cluster_count, "cluster_count")
        if not 2 <= self.cluster_count <= 10:
            raise ValueError("cluster_count must be between 2 and 10")
        _integer_value(self.random_state, "random_state")
        _positive_integer(
            self.minimum_candidate_duration_minutes,
            "minimum_candidate_duration_minutes",
        )
        _positive_integer(self.candidate_count_per_cluster, "candidate_count_per_cluster")
        if not isinstance(self.maximum_plot_points, int) or isinstance(
            self.maximum_plot_points, bool
        ):
            raise ValueError("maximum_plot_points must be an integer")
        if self.maximum_plot_points < 2:
            raise ValueError("maximum_plot_points must be at least 2")


@dataclass(frozen=True)
class PerformanceConfig:
    performance_tag: str
    direction: str
    target_min: float | None = None
    target_max: float | None = None
    minimum_duration_minutes: int = DEFAULT_MINIMUM_CANDIDATE_DURATION_MINUTES
    candidate_count: int = DEFAULT_CANDIDATES_PER_CLUSTER

    def __post_init__(self) -> None:
        if not isinstance(self.performance_tag, str) or not self.performance_tag.strip():
            raise ValueError("performance_tag must not be empty")
        if self.direction not in PERFORMANCE_DIRECTIONS:
            raise ValueError(
                "direction must be higher_is_better, lower_is_better, or target_range"
            )
        for name, value in (
            ("target_min", self.target_min),
            ("target_max", self.target_max),
        ):
            if value is not None and (
                not isinstance(value, (int, float, np.number))
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.direction == "target_range":
            if self.target_min is None or self.target_max is None:
                raise ValueError("target_range requires target_min and target_max")
            if self.target_min > self.target_max:
                raise ValueError("target_min must not exceed target_max")
        elif self.target_min is not None or self.target_max is not None:
            raise ValueError(
                "target_min and target_max are only valid for target_range"
            )
        _positive_integer(self.minimum_duration_minutes, "minimum_duration_minutes")
        _positive_integer(self.candidate_count, "candidate_count")


def run_state_exploration(
    indexed: pd.DataFrame,
    tag_columns: list[str],
    preprocessing_config: PreprocessingConfig,
    exploration_config: ExplorationConfig = ExplorationConfig(),
    performance_config: PerformanceConfig | Mapping[str, Any] | None = None,
    performance_series: pd.Series | None = None,
) -> dict[str, object]:
    """Explore historical operating states; never labels a state as normal."""
    normalized_performance = _normalize_performance_config(performance_config)
    preserve_columns = (
        [normalized_performance.performance_tag]
        if normalized_performance is not None
        and normalized_performance.performance_tag in indexed.columns
        and normalized_performance.performance_tag not in tag_columns
        else []
    )
    processed = preprocess_window(
        indexed,
        tag_columns,
        preprocessing_config,
        preserve_columns=preserve_columns,
        include_intermediates=True,
    )
    dynamic = processed.dynamic
    if len(dynamic) <= exploration_config.cluster_count:
        raise ValueError("有效动态样本数必须大于cluster_count")
    model = fit_dpca(dynamic)
    clustered = cluster_model_scores(
        model,
        dynamic,
        exploration_config.cluster_count,
        preprocessing_config.sample_interval_minutes,
        exploration_config.random_state,
    )
    points = clustered.points.rename(columns={"cluster": "cluster_id"}).copy()
    points["cluster_id"] = points["cluster_id"].map(
        lambda value: f"cluster_{int(value):03d}"
    )
    points["segment_id"] = processed.final_segment_ids.reindex(points.index).to_numpy()
    points = points.sort_index()
    centers = {
        f"cluster_{cluster:03d}": center.copy()
        for cluster, center in clustered.centers.items()
    }
    candidates = _cluster_candidates(
        points,
        exploration_config,
        preprocessing_config.sample_interval_minutes,
        centers,
        clustered.pc_columns,
    )
    summaries = _summaries(
        points,
        dynamic,
        tag_columns,
        candidates,
        preprocessing_config.sample_interval_minutes,
        centers,
        clustered.pc_columns,
    )
    performance_candidates: list[dict[str, object]] = []
    if normalized_performance is not None:
        values = _performance_values(
            indexed,
            processed,
            dynamic,
            normalized_performance,
            performance_series,
        )
        performance_candidates = _performance_candidates(
            points,
            values,
            normalized_performance,
            preprocessing_config.sample_interval_minutes,
            centers,
            clustered.pc_columns,
        )
    preprocessing_summary = _preprocessing_summary(processed)
    warnings = _warnings(
        points,
        summaries,
        candidates,
        performance_candidates,
        normalized_performance,
        preprocessing_summary,
    )
    display = _display_points(
        points, exploration_config.maximum_plot_points, preprocessing_config.sample_interval_minutes
    )
    full_point_count = len(points)
    returned_point_count = len(display)
    return {
        "exploration_run_id": uuid4().hex,
        "exploration_config": asdict(exploration_config),
        "performance_config": (
            asdict(normalized_performance)
            if normalized_performance is not None
            else None
        ),
        "preprocessing_summary": preprocessing_summary,
        "exploratory_model_summary": {
            "model_purpose": "exploratory",
            "model_status": "draft",
            "n_components": model.n_components,
            "pc_columns": list(clustered.pc_columns),
            "cluster_count": exploration_config.cluster_count,
        },
        "cluster_centers": {
            cluster_id: [float(value) for value in center]
            for cluster_id, center in centers.items()
        },
        "cluster_series": points,
        "cluster_series_display": display,
        "full_point_count": full_point_count,
        "returned_point_count": returned_point_count,
        "cluster_summaries": summaries,
        "cluster_candidates": candidates,
        "performance_candidates": performance_candidates,
        "warnings": warnings,
        "duration_semantics": DURATION_SEMANTICS,
    }


def _normalize_performance_config(
    value: PerformanceConfig | Mapping[str, Any] | None,
) -> PerformanceConfig | None:
    if value is None:
        return None
    if isinstance(value, PerformanceConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("performance_config must be an object")
    try:
        return PerformanceConfig(**dict(value))
    except TypeError as error:
        raise ValueError("performance_config fields are invalid") from error


def _performance_values(
    indexed: pd.DataFrame,
    processed: PreprocessingResult,
    dynamic: pd.DataFrame,
    config: PerformanceConfig,
    performance_series: pd.Series | None,
) -> pd.Series:
    if performance_series is not None:
        if not isinstance(performance_series.index, pd.DatetimeIndex):
            raise TypeError("performance series index must be a DatetimeIndex")
        if performance_series.index.has_duplicates:
            raise ValueError("performance series timestamps must be unique")
        source = performance_series.sort_index()
    elif config.performance_tag in indexed.columns:
        if processed.state_filtered is not None and config.performance_tag in processed.state_filtered:
            source = processed.state_filtered[config.performance_tag]
        else:
            source = indexed[config.performance_tag]
    else:
        raise ValueError(f"missing performance tag: {config.performance_tag}")
    values = pd.to_numeric(source.reindex(dynamic.index), errors="coerce")
    return values.astype(float).set_axis(dynamic.index)


def _cluster_candidates(
    points: pd.DataFrame,
    config: ExplorationConfig,
    interval: int,
    centers: Mapping[str, np.ndarray],
    pc_columns: Sequence[str],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    runs = _contiguous_runs(points, interval, split_cluster=True)
    for cluster_id in sorted(points["cluster_id"].unique()):
        cluster_runs = [
            run for run in runs if str(run["cluster_id"].iloc[0]) == str(cluster_id)
        ]
        eligible = [
            run
            for run in cluster_runs
            if _coverage_duration_minutes(run, interval)
            >= config.minimum_candidate_duration_minutes
        ]
        center = np.asarray(centers[str(cluster_id)], dtype=float)
        ranked = sorted(
            eligible,
            key=lambda run: (
                _centroid_distance(run, center, pc_columns),
                -_stability_score(run, {str(cluster_id): center}, pc_columns),
                -_coverage_duration_minutes(run, interval),
                run.index[0],
            ),
        )
        for rank, run in enumerate(
            ranked[: config.candidate_count_per_cluster], 1
        ):
            distance = _centroid_distance(run, center, pc_columns)
            candidates.append(
                {
                    "candidate_id": f"{cluster_id}-candidate-{rank:03d}",
                    "source": "cluster",
                    "cluster_id": str(cluster_id),
                    "start": run.index[0].isoformat(),
                    "end": run.index[-1].isoformat(),
                    "sample_count": len(run),
                    "duration_minutes": _coverage_duration_minutes(run, interval),
                    "centroid_distance": distance,
                    "stability_score": _stability_score(
                        run, {str(cluster_id): center}, pc_columns
                    ),
                    "completeness_ratio": 1.0,
                    "rank": rank,
                    "rank_within_cluster": rank,
                    "comment": "",
                }
            )
    return candidates


def _performance_candidates(
    points: pd.DataFrame,
    values: pd.Series,
    config: PerformanceConfig,
    interval: int,
    centers: Mapping[str, np.ndarray],
    pc_columns: Sequence[str],
) -> list[dict[str, object]]:
    aligned_values = values.reindex(points.index)
    finite = np.isfinite(aligned_values.to_numpy(dtype=float))
    usable = points.loc[finite].copy()
    usable_values = aligned_values.loc[usable.index]
    runs = _contiguous_runs(usable, interval, split_cluster=False)
    eligible = [
        run
        for run in runs
        if _coverage_duration_minutes(run, interval)
        >= config.minimum_duration_minutes
    ]

    def ranking_key(run: pd.DataFrame) -> tuple[object, ...]:
        summary = _performance_summary(usable_values.loc[run.index], config)
        # Rank by the configured direction first, then deterministic quality tie-breakers.
        if config.direction == "higher_is_better":
            direction_key = (-float(summary["mean"]), -float(summary["median"]))
        elif config.direction == "lower_is_better":
            direction_key = (float(summary["mean"]), float(summary["median"]))
        else:
            direction_key = (
                -float(summary["target_ratio"]),
                float(summary["mean_target_deviation"]),
                float(summary["mean"]),
            )
        return (*direction_key, -_coverage_duration_minutes(run, interval), run.index[0])

    ranked = sorted(eligible, key=ranking_key)
    result: list[dict[str, object]] = []
    for rank, run in enumerate(ranked[: config.candidate_count], 1):
        run_values = usable_values.loc[run.index]
        summary = _performance_summary(run_values, config)
        associated = sorted({str(value) for value in run["cluster_id"]})
        result.append(
            {
                "candidate_id": f"performance-candidate-{rank:03d}",
                "source": "performance",
                "start": run.index[0].isoformat(),
                "end": run.index[-1].isoformat(),
                "sample_count": len(run),
                "duration_minutes": _coverage_duration_minutes(run, interval),
                "performance_summary": summary,
                "associated_cluster_ids": associated,
                "stability_score": _performance_stability_score(
                    run, centers, pc_columns
                ),
                "rank": rank,
                "comment": "",
            }
        )
    return result


def _performance_summary(
    values: pd.Series, config: PerformanceConfig
) -> dict[str, float]:
    numeric = values.to_numpy(dtype=float)
    summary = {
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
    }
    if config.direction == "target_range":
        within = (numeric >= config.target_min) & (numeric <= config.target_max)
        deviation = np.where(
            numeric < config.target_min,
            config.target_min - numeric,
            np.where(numeric > config.target_max, numeric - config.target_max, 0.0),
        )
        summary["target_ratio"] = float(np.mean(within))
        summary["mean_target_deviation"] = float(np.mean(deviation))
    return summary


def _summaries(
    points: pd.DataFrame,
    dynamic: pd.DataFrame,
    tags: Sequence[str],
    candidates: Sequence[Mapping[str, object]],
    interval: int,
    centers: Mapping[str, np.ndarray],
    pc_columns: Sequence[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    runs = _contiguous_runs(points, interval, split_cluster=True)
    for cluster_id, group in points.groupby("cluster_id", sort=True):
        cluster_id = str(cluster_id)
        source = dynamic.loc[group.index]
        tag_stats: dict[str, dict[str, object]] = {}
        for tag in tags:
            values = source[f"{tag}__lag_000min"]
            tag_stats[tag] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "stage": "dynamic_lag_000",
            }
        center = np.asarray(centers[cluster_id], dtype=float)
        distances = np.linalg.norm(
            group.loc[:, list(pc_columns)].to_numpy(dtype=float) - center,
            axis=1,
        )
        cluster_runs = [
            run for run in runs if str(run["cluster_id"].iloc[0]) == cluster_id
        ]
        result.append(
            {
                "cluster_id": cluster_id,
                "sample_count": len(group),
                "coverage_ratio": len(group) / len(points) if len(points) else 0.0,
                "segment_count": len(cluster_runs),
                "total_duration_minutes": int(
                    sum(_coverage_duration_minutes(run, interval) for run in cluster_runs)
                ),
                "centroid_pc_scores": [float(value) for value in center],
                "median_distance_to_centroid": float(np.median(distances)),
                "pc_score_dispersion": float(np.std(distances, ddof=0)),
                "start_timestamp": group.index[0].isoformat(),
                "end_timestamp": group.index[-1].isoformat(),
                "tag_statistics": tag_stats,
                "candidate_count": sum(
                    item["cluster_id"] == cluster_id for item in candidates
                ),
            }
        )
    return result


def _preprocessing_summary(processed: PreprocessingResult) -> dict[str, object]:
    summary = processed.summary.to_dict()
    state_filter_loss = max(
        0,
        int(processed.summary.state_filter_input_rows)
        - int(processed.summary.state_filter_output_rows),
    )
    empty_bin_count = int(processed.summary.empty_bin_count)
    if processed.state_filtered is not None:
        state_filter_loss = max(
            0,
            len(processed.resampled) - len(processed.state_filtered)
            if processed.resampled is not None
            else state_filter_loss,
        )
        empty_bin_count = int(
            processed.empty_bin_mask.reindex(
                processed.state_filtered.index, fill_value=False
            ).sum()
        )
    loss_counts = {
        "empty_bin_count": empty_bin_count,
        "input_invalid_loss": int(processed.summary.input_invalid_loss),
        "filter_warmup_loss": int(processed.summary.filter_warmup_loss),
        "filter_context_invalid_loss": int(
            processed.summary.filter_context_invalid_loss
        ),
        "lag_warmup_loss": int(processed.summary.lag_warmup_loss),
        "lag_context_invalid_loss": int(processed.summary.lag_context_invalid_loss),
        "state_filter_loss": state_filter_loss,
    }
    denominator = int(processed.summary.resampled_row_count)
    total_loss = sum(loss_counts.values())
    coverage_denominator = int(processed.summary.resampled_row_count)
    coverage_ratio = (
        int(processed.summary.final_dynamic_row_count) / coverage_denominator
        if coverage_denominator
        else 0.0
    )
    summary.update(
        {
            "state_filter_loss": state_filter_loss,
            "loss_counts": loss_counts,
            "loss_count": total_loss,
            "loss_denominator": denominator,
            "loss_ratio": total_loss / denominator if denominator else 0.0,
            "effective_coverage_ratio": coverage_ratio,
        }
    )
    return summary


def _warnings(
    points: pd.DataFrame,
    summaries: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    performance_candidates: Sequence[Mapping[str, object]],
    performance_config: PerformanceConfig | None,
    preprocessing_summary: Mapping[str, object],
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    candidate_clusters = {str(item["cluster_id"]) for item in candidates}
    for summary in summaries:
        cluster_id = str(summary["cluster_id"])
        sample_count = int(summary["sample_count"])
        if sample_count < DEFAULT_CLUSTER_SAMPLE_COUNT_LOW_THRESHOLD:
            warnings.append(
                {
                    "code": "cluster_sample_count_low",
                    "cluster_id": cluster_id,
                    "value": sample_count,
                    "threshold": DEFAULT_CLUSTER_SAMPLE_COUNT_LOW_THRESHOLD,
                    "message": "该 Cluster 样本数较少，统计摘要应谨慎解释。",
                }
            )
        if cluster_id not in candidate_clusters:
            warnings.append(
                {
                    "code": "cluster_has_no_candidate",
                    "cluster_id": cluster_id,
                    "value": 0,
                    "threshold": DEFAULT_CANDIDATE_COUNT_WARNING_THRESHOLD,
                    "message": "该 Cluster 没有满足最小时长的连续候选窗口。",
                }
            )

    coverage = float(preprocessing_summary["effective_coverage_ratio"])
    if coverage < DEFAULT_COVERAGE_WARNING_THRESHOLD:
        warnings.append(
            {
                "code": "exploration_coverage_low",
                "value": coverage,
                "threshold": DEFAULT_COVERAGE_WARNING_THRESHOLD,
                "message": "有效探索样本覆盖率低于告警阈值。",
            }
        )
    loss_ratio = float(preprocessing_summary["loss_ratio"])
    if loss_ratio > DEFAULT_PREPROCESSING_LOSS_WARNING_THRESHOLD:
        warnings.append(
            {
                "code": "preprocessing_loss_ratio_high",
                "value": loss_ratio,
                "threshold": DEFAULT_PREPROCESSING_LOSS_WARNING_THRESHOLD,
                "message": "预处理互斥损失比例高于告警阈值。",
            }
        )
    if performance_config is not None and len(performance_candidates) < performance_config.candidate_count:
        warnings.append(
            {
                "code": "performance_candidate_insufficient",
                "value": len(performance_candidates),
                "threshold": performance_config.candidate_count,
                "message": "满足性能候选条件的连续窗口数量不足配置数量。",
            }
        )
    return warnings


def _contiguous_runs(
    points: pd.DataFrame, interval: int, *, split_cluster: bool
) -> list[pd.DataFrame]:
    if points.empty:
        return []
    expected = pd.Timedelta(minutes=interval)
    ordered = points.sort_index()
    breaks = ordered.index.to_series().diff().ne(expected)
    if "segment_id" in ordered.columns:
        breaks |= ordered["segment_id"].ne(ordered["segment_id"].shift())
    if split_cluster and "cluster_id" in ordered.columns:
        breaks |= ordered["cluster_id"].ne(ordered["cluster_id"].shift())
    keys = breaks.cumsum()
    return [run.copy() for _, run in ordered.groupby(keys, sort=False)]


def _coverage_duration_minutes(run: pd.DataFrame, interval: int) -> int:
    if run.empty:
        return 0
    elapsed = (run.index[-1] - run.index[0]).total_seconds() / 60.0
    return int(round(elapsed + interval))


def _centroid_distance(
    run: pd.DataFrame, center: np.ndarray, pc_columns: Sequence[str]
) -> float:
    mean_score = run.loc[:, list(pc_columns)].to_numpy(dtype=float).mean(axis=0)
    distance = float(np.linalg.norm(mean_score - center))
    return distance if np.isfinite(distance) else float("inf")


def _stability_score(
    run: pd.DataFrame,
    centers: Mapping[str, np.ndarray],
    pc_columns: Sequence[str],
) -> float:
    cluster_id = str(run["cluster_id"].iloc[0])
    center = np.asarray(centers[cluster_id], dtype=float)
    values = run.loc[:, list(pc_columns)].to_numpy(dtype=float)
    distances = np.linalg.norm(values - center, axis=1)
    median = float(np.median(distances))
    robust_dispersion = float(np.median(np.abs(distances - median)))
    score = 1.0 / (1.0 + 1.4826 * robust_dispersion)
    return float(score) if np.isfinite(score) else 0.0


def _performance_stability_score(
    run: pd.DataFrame,
    centers: Mapping[str, np.ndarray],
    pc_columns: Sequence[str],
) -> float:
    values = run.loc[:, list(pc_columns)].to_numpy(dtype=float)
    point_centers = np.vstack(
        [np.asarray(centers[str(cluster_id)], dtype=float) for cluster_id in run["cluster_id"]]
    )
    distances = np.linalg.norm(values - point_centers, axis=1)
    median = float(np.median(distances))
    robust_dispersion = float(np.median(np.abs(distances - median)))
    score = 1.0 / (1.0 + 1.4826 * robust_dispersion)
    return float(score) if np.isfinite(score) else 0.0


def _display_points(points: pd.DataFrame, limit: int, interval: int = 5) -> pd.DataFrame:
    if limit < 2:
        raise ValueError("maximum_plot_points must be at least 2")
    ordered = points.sort_index()
    if len(ordered) <= limit:
        return ordered.copy()
    selected: set[int] = set()

    def add_positions(positions: Sequence[int]) -> None:
        for position in sorted(set(int(value) for value in positions)):
            if len(selected) >= limit:
                return
            selected.add(position)

    add_positions((0, len(ordered) - 1))
    expected = pd.Timedelta(minutes=interval)
    for position in range(1, len(ordered)):
        physical_break = ordered.index[position] - ordered.index[position - 1] != expected
        segment_break = (
            "segment_id" in ordered.columns
            and ordered.segment_id.iloc[position] != ordered.segment_id.iloc[position - 1]
        )
        if physical_break or segment_break:
            add_positions((position - 1, position))
    cluster_column = (
        "cluster_id" if "cluster_id" in ordered.columns else "cluster"
    )
    for position in range(1, len(ordered)):
        if (
            cluster_column in ordered.columns
            and ordered[cluster_column].iloc[position]
            != ordered[cluster_column].iloc[position - 1]
        ):
            add_positions((position - 1, position))

    for column in ("pc1", "pc2"):
        if column not in ordered.columns:
            continue
        values = ordered[column].to_numpy(dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            add_positions((finite[np.argmin(values[finite])], finite[np.argmax(values[finite])]))

    remaining = np.array(
        [position for position in range(len(ordered)) if position not in selected],
        dtype=int,
    )
    if len(selected) < limit:
        selected.update(
            _spread_positions(remaining, limit - len(selected)).tolist()
        )
    return ordered.iloc[np.array(sorted(selected), dtype=int)].copy()


def _spread_positions(candidates: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(candidates):
        return np.array([], dtype=int)
    if count >= len(candidates):
        return candidates
    offsets = np.arange(count) * len(candidates) // count
    return candidates[offsets]
