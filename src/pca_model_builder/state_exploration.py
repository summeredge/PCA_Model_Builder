from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

import numpy as np
import pandas as pd

from .clustering import cluster_model_scores
from .dpca import fit_dpca
from .preprocessing import PreprocessingConfig, preprocess_window


DEFAULT_CLUSTER_COUNT = 4
DEFAULT_RANDOM_STATE = 0
DEFAULT_CANDIDATES_PER_CLUSTER = 3
DEFAULT_MINIMUM_CANDIDATE_DURATION_MINUTES = 30
DEFAULT_MAXIMUM_PLOT_POINTS = 1200


@dataclass(frozen=True)
class ExplorationConfig:
    cluster_count: int = DEFAULT_CLUSTER_COUNT
    random_state: int = DEFAULT_RANDOM_STATE
    minimum_candidate_duration_minutes: int = DEFAULT_MINIMUM_CANDIDATE_DURATION_MINUTES
    candidate_count_per_cluster: int = DEFAULT_CANDIDATES_PER_CLUSTER
    maximum_plot_points: int = DEFAULT_MAXIMUM_PLOT_POINTS

    def __post_init__(self) -> None:
        if not 2 <= self.cluster_count <= 10:
            raise ValueError("cluster_count must be between 2 and 10")
        if self.minimum_candidate_duration_minutes <= 0:
            raise ValueError("minimum_candidate_duration_minutes must be positive")
        if self.candidate_count_per_cluster <= 0:
            raise ValueError("candidate_count_per_cluster must be positive")
        if self.maximum_plot_points < 2:
            raise ValueError("maximum_plot_points must be at least 2")


def run_state_exploration(
    indexed: pd.DataFrame,
    tag_columns: list[str],
    preprocessing_config: PreprocessingConfig,
    exploration_config: ExplorationConfig = ExplorationConfig(),
) -> dict[str, object]:
    """Explore historical operating states; never labels a state as normal."""
    processed = preprocess_window(indexed, tag_columns, preprocessing_config)
    dynamic = processed.dynamic
    if len(dynamic) <= exploration_config.cluster_count:
        raise ValueError("有效动态样本数必须大于cluster_count")
    model = fit_dpca(dynamic)
    clustered = cluster_model_scores(
        model, dynamic, exploration_config.cluster_count,
        preprocessing_config.sample_interval_minutes, exploration_config.random_state,
    )
    points = clustered.points.rename(columns={"cluster": "cluster_id"}).copy()
    points["cluster_id"] = points["cluster_id"].map(lambda value: f"cluster_{int(value):03d}")
    points["segment_id"] = processed.final_segment_ids.reindex(points.index).to_numpy()
    candidates = _cluster_candidates(points, exploration_config, preprocessing_config.sample_interval_minutes)
    summaries = _summaries(points, dynamic, tag_columns, candidates, preprocessing_config.sample_interval_minutes)
    display = _display_points(points, exploration_config.maximum_plot_points)
    return {
        "exploration_run_id": uuid4().hex,
        "exploration_config": asdict(exploration_config),
        "preprocessing_summary": processed.summary.to_dict(),
        "exploratory_model_summary": {"model_purpose": "exploratory", "model_status": "draft", "n_components": model.n_components},
        "cluster_series": points,
        "cluster_series_display": display,
        "cluster_summaries": summaries,
        "cluster_candidates": candidates,
        "performance_candidates": [],
        "warnings": [],
    }


def _cluster_candidates(points: pd.DataFrame, config: ExplorationConfig, interval: int) -> list[dict[str, object]]:
    expected = pd.Timedelta(minutes=interval)
    candidates: list[dict[str, object]] = []
    for cluster_id, group in points.groupby("cluster_id", sort=True):
        center = group[["pc1", "pc2"]].mean()
        runs: list[pd.DataFrame] = []
        run_keys = (points.index.to_series().diff().ne(expected) | points.segment_id.ne(points.segment_id.shift()) | points.cluster_id.ne(points.cluster_id.shift())).cumsum()
        for _, run in points.loc[points.cluster_id.eq(cluster_id)].groupby(run_keys[points.cluster_id.eq(cluster_id)]):
            if len(run) * interval < config.minimum_candidate_duration_minutes:
                continue
            runs.append(run)
        ranked = sorted(runs, key=lambda run: (float(np.linalg.norm(run[["pc1", "pc2"]].mean().to_numpy()-center.to_numpy())), -len(run), run.index[0]))
        for rank, run in enumerate(ranked[:config.candidate_count_per_cluster], 1):
            distance = float(np.linalg.norm(run[["pc1", "pc2"]].mean().to_numpy()-center.to_numpy()))
            candidates.append({"candidate_id": f"{cluster_id}-candidate-{rank:03d}", "source": "cluster", "cluster_id": cluster_id, "start": run.index[0].isoformat(), "end": run.index[-1].isoformat(), "sample_count": len(run), "duration_minutes": len(run)*interval, "centroid_distance": distance, "stability_score": float(1/(1+run[["pc1", "pc2"]].std(ddof=0).to_numpy().mean())), "completeness_ratio": 1.0, "rank_within_cluster": rank, "comment": ""})
    return candidates


def _summaries(points: pd.DataFrame, dynamic: pd.DataFrame, tags: list[str], candidates: list[dict[str, object]], interval: int) -> list[dict[str, object]]:
    result=[]
    for cluster_id, group in points.groupby("cluster_id", sort=True):
        source = dynamic.loc[group.index]
        tag_stats={}
        for tag in tags:
            values=source[f"{tag}__lag_000min"]
            tag_stats[tag]={"mean":float(values.mean()),"std":float(values.std(ddof=0)),"median":float(values.median()),"minimum":float(values.min()),"maximum":float(values.max()),"stage":"dynamic_lag_000"}
        distances=np.linalg.norm(group[["pc1","pc2"]].to_numpy()-group[["pc1","pc2"]].mean().to_numpy(),axis=1)
        result.append({"cluster_id":cluster_id,"sample_count":len(group),"coverage_ratio":len(group)/len(points),"segment_count":int((group.index.to_series().diff().ne(pd.Timedelta(minutes=interval)) | group.segment_id.ne(group.segment_id.shift())).sum()),"total_duration_minutes":len(group)*interval,"centroid_pc_scores":[float(group.pc1.mean()),float(group.pc2.mean())],"median_distance_to_centroid":float(np.median(distances)),"pc_score_dispersion":float(group[["pc1","pc2"]].std(ddof=0).to_numpy().mean()),"start_timestamp":group.index[0].isoformat(),"end_timestamp":group.index[-1].isoformat(),"tag_statistics":tag_stats,"candidate_count":sum(item["cluster_id"]==cluster_id for item in candidates)})
    return result


def _display_points(points: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(points)<=limit: return points
    return points.iloc[np.unique(np.linspace(0,len(points)-1,limit,dtype=int))]
