from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .dpca import DPCAModel


@dataclass(frozen=True)
class OperatingStateClusters:
    points: pd.DataFrame
    summaries: tuple[dict[str, object], ...]
    n_components: int
    cumulative_explained_variance: float


def cluster_operating_states(
    dynamic: pd.DataFrame,
    n_clusters: int,
    variance_threshold: float = 0.95,
    sample_interval_minutes: int = 5,
) -> OperatingStateClusters:
    """Cluster standardized dynamic PCA scores for engineer-assisted review."""
    if not isinstance(dynamic.index, pd.DatetimeIndex):
        raise TypeError("dynamic matrix index must be a DatetimeIndex")
    if not 2 <= n_clusters <= 10:
        raise ValueError("cluster count must be between 2 and 10")
    if len(dynamic) <= n_clusters:
        raise ValueError("cluster analysis needs more samples than clusters")
    if not 0 < variance_threshold < 1:
        raise ValueError(
            "variance threshold must be in (0, 1) to preserve residual space for SPE"
        )
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")

    values = dynamic.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cluster inputs contain non-finite values")
    standardized = StandardScaler().fit_transform(values)
    if len(np.unique(standardized, axis=0)) < n_clusters:
        raise ValueError("cluster count exceeds the number of distinct states")

    pca = PCA(svd_solver="full").fit(standardized)
    selected = int(
        np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_),
            variance_threshold,
            side="left",
        )
        + 1
    )
    selected = min(max(2, selected), pca.components_.shape[0])
    scores = pca.transform(standardized)[:, :selected]
    return _cluster_scores(
        pd.DataFrame(scores, index=dynamic.index),
        n_clusters=n_clusters,
        sample_interval_minutes=sample_interval_minutes,
        cumulative_explained_variance=float(
            pca.explained_variance_ratio_[:selected].sum()
        ),
    )


def cluster_model_scores(
    model: DPCAModel,
    dynamic: pd.DataFrame,
    n_clusters: int,
    sample_interval_minutes: int = 5,
    random_state: int = 0,
) -> OperatingStateClusters:
    """Cluster scores from a saved exploratory DPCA model without refitting PCA."""
    if tuple(dynamic.columns) != model.feature_names:
        raise ValueError("dynamic features do not match exploratory model")
    scores = model.score(dynamic)
    pc_columns = [f"pc{index}" for index in range(1, model.n_components + 1)]
    return _cluster_scores(
        scores[pc_columns],
        n_clusters=n_clusters,
        sample_interval_minutes=sample_interval_minutes,
        cumulative_explained_variance=float(
            model.explained_variance_ratio[: model.n_components].sum()
        ),
        random_state=random_state,
    )


def _cluster_scores(
    scores: pd.DataFrame,
    n_clusters: int,
    sample_interval_minutes: int,
    cumulative_explained_variance: float,
    random_state: int = 0,
) -> OperatingStateClusters:
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("cluster scores index must be a DatetimeIndex")
    if not 2 <= n_clusters <= 10:
        raise ValueError("cluster count must be between 2 and 10")
    if len(scores) <= n_clusters:
        raise ValueError("cluster analysis needs more samples than clusters")
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cluster inputs contain non-finite values")
    if len(np.unique(values, axis=0)) < n_clusters:
        raise ValueError("cluster count exceeds the number of distinct states")

    fitted = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(values)
    order = np.lexsort(tuple(fitted.cluster_centers_[:, position] for position in range(fitted.cluster_centers_.shape[1] - 1, -1, -1)))
    remap = {int(label): position + 1 for position, label in enumerate(order)}
    labels = np.array([remap[int(label)] for label in fitted.labels_], dtype=int)
    centers = {
        remap[int(label)]: fitted.cluster_centers_[label]
        for label in range(n_clusters)
    }
    points = pd.DataFrame(
        {"pc1": values[:, 0], "pc2": values[:, 1], "cluster": labels},
        index=scores.index,
    )
    summaries = tuple(
        {
            "cluster": cluster,
            "count": int((labels == cluster).sum()),
            "share": float((labels == cluster).mean()),
            "pc1_center": float(centers[cluster][0]),
            "pc2_center": float(centers[cluster][1]),
            "representative_windows": _representative_windows(
                points.index, labels, cluster, sample_interval_minutes
            ),
        }
        for cluster in range(1, n_clusters + 1)
    )
    return OperatingStateClusters(
        points=points,
        summaries=summaries,
        n_components=values.shape[1],
        cumulative_explained_variance=cumulative_explained_variance,
    )


def _representative_windows(
    index: pd.DatetimeIndex,
    labels: np.ndarray,
    cluster: int,
    sample_interval_minutes: int,
) -> list[dict[str, object]]:
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    runs: list[dict[str, object]] = []
    run_start = 0
    for position in range(1, len(index) + 1):
        continues = (
            position < len(index)
            and labels[position] == labels[position - 1]
            and index[position] - index[position - 1] == expected
        )
        if continues:
            continue
        if labels[position - 1] == cluster:
            runs.append(
                {
                    "start": index[run_start].isoformat(),
                    "end": index[position - 1].isoformat(),
                    "count": position - run_start,
                }
            )
        run_start = position
    return sorted(runs, key=lambda item: int(item["count"]), reverse=True)[:3]
