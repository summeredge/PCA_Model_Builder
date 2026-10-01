from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def analyze_cluster_quality(
    scores: pd.DataFrame,
    labels,
    timestamps=None,
    raw_data: pd.DataFrame | None = None,
    feature_names=(),
    explained_variance_ratio=(),
    sample_interval_minutes: float = 5,
    cluster_centers=None,
) -> dict[str, object]:
    """Explain existing clusters; never fit or select a model or candidate.

    Feature contrast is the range of cluster means / global population std.
    Runs use sampling coverage (last - first + interval), split at time gaps.
    Direction uses center variance: >=80% on one axis, otherwise two axes.
    """
    result = {
        "cluster_count": 0, "silhouette_score": None,
        "silhouette_sample_count": 0, "silhouette_approximate": False,
        "explained_variance": {"pc1": None, "pc2": None},
        "centers": [], "center_orientation": "无明显方向",
        "temporal_metrics": None, "top_features": [],
        "engineering_hint": {}, "unavailable_reasons": {},
    }
    reasons = result["unavailable_reasons"]
    values = scores.to_numpy(dtype=float)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(values):
        reasons["analysis"] = "Cluster 标签与样本数量不一致"
        return result
    if not len(values):
        reasons["analysis"] = "没有聚类样本（空 Cluster）"
        return result
    if pd.isna(labels).any():
        reasons["analysis"] = "Cluster 标签存在缺失值"
        return result
    clusters = pd.unique(labels)
    result["cluster_count"] = len(clusters)
    if values.shape[1] < 2 or not np.isfinite(values).all():
        reasons["analysis"] = "需要至少两个主元且得分必须全部有效"
        return result
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    for position, key in enumerate(("pc1", "pc2")):
        if position < len(ratios) and np.isfinite(ratios[position]) and 0 <= ratios[position] <= 1:
            result["explained_variance"][key] = float(ratios[position])
    if any(value is None for value in result["explained_variance"].values()):
        reasons["explained_variance"] = "缺少有效的 PCA 主元解释率"
    centers = np.array([
        cluster_centers[cluster] if cluster_centers is not None else values[labels == cluster].mean(axis=0)
        for cluster in clusters
    ])
    result["centers"] = [
        {"cluster": str(cluster), "pc1": float(center[0]), "pc2": float(center[1])}
        for cluster, center in zip(clusters, centers, strict=True)
    ]
    variance = centers[:, :2].var(axis=0)
    total = float(variance.sum())
    if total > 1e-12:
        result["center_orientation"] = (
            "主要沿 PC1" if variance[0] / total >= 0.8 else
            "主要沿 PC2" if variance[1] / total >= 0.8 else "二维分布明显"
        )
    if not 2 <= len(clusters) <= 10:
        reasons["silhouette_score"] = "Silhouette 需要 2–10 个非空 Cluster"
    elif len(clusters) >= len(values):
        reasons["silhouette_score"] = "样本不足：样本数必须大于 Cluster 数量"
    else:
        selected = np.arange(len(values))
        if len(values) > 2000:
            # ponytail: bound pairwise memory; use chunked full silhouette if exact large-data metrics are needed.
            selected = np.unique(np.r_[
                np.random.default_rng(0).choice(len(values), 2000, replace=False),
                [np.flatnonzero(labels == cluster)[0] for cluster in clusters],
            ])
            result["silhouette_approximate"] = True
        result["silhouette_score"] = float(silhouette_score(values[selected], labels[selected]))
        result["silhouette_sample_count"] = len(selected)

    if timestamps is None:
        reasons["temporal_metrics"] = "没有时间列，无法统计时间连续性"
    else:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce"))
        interval = pd.Timedelta(minutes=sample_interval_minutes)
        if len(index) != len(labels) or index.hasnans or not index.is_monotonic_increasing or index.has_duplicates or interval <= pd.Timedelta(0):
            reasons["temporal_metrics"] = "时间戳需与样本对齐、有效、递增且唯一，采样间隔需为正"
        else:
            durations = []
            start = 0
            switches = 0
            for position in range(1, len(index) + 1):
                adjacent = position < len(index) and index[position] - index[position - 1] == interval
                changed = position < len(index) and labels[position] != labels[position - 1]
                if adjacent and changed:
                    switches += 1
                if adjacent and not changed:
                    continue
                durations.append((index[position - 1] - index[start] + interval).total_seconds() / 3600)
                start = position
            result["temporal_metrics"] = {
                "average_duration_hours": float(np.mean(durations)),
                "longest_duration_hours": float(max(durations)),
                "state_switch_count": switches, "run_count": len(durations),
                "duration_semantics": "coverage",
            }

    if raw_data is not None and raw_data.index.is_unique:
        aligned = raw_data.reindex(scores.index)
        for tag in dict.fromkeys(feature_names):
            if tag not in aligned or pd.api.types.is_datetime64_any_dtype(aligned[tag]):
                continue
            numeric = pd.to_numeric(aligned[tag], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
            std = numeric.std(ddof=0)
            means = [numeric.iloc[np.flatnonzero(labels == cluster)].mean() for cluster in clusters]
            if not np.isfinite(std) or std <= 0 or not np.isfinite(means).all() or len(clusters) < 2:
                continue
            difference = float(max(means) - min(means))
            result["top_features"].append({
                "tag": tag, "mean_difference": difference,
                "standardized_difference": difference / float(std),
                "cluster_means": {str(cluster): float(mean) for cluster, mean in zip(clusters, means, strict=True)},
            })
        result["top_features"] = sorted(result["top_features"], key=lambda item: -item["standardized_difference"])[:5]
    if not result["top_features"]:
        reasons["top_features"] = "无可比较的有效建模 Tag（需至少两个 Cluster、非恒定数值及各 Cluster 有效均值）"

    hints = []
    silhouette = result["silhouette_score"]
    if silhouette is not None and silhouette < 0.25:
        hints.append("当前 Cluster 分离度较弱，建议结合工艺状态确认。")
    pc1 = result["explained_variance"]["pc1"]
    if pc1 is not None and pc1 >= 0.6 and result["center_orientation"] == "主要沿 PC1":
        hints.append("当前状态划分主要沿 PC1 方向分离，可能反映连续运行变量变化，建议结合工艺变量确认。")
    evidence = "".join(hints) or "请结合中心分布、区分变量和时间连续性确认运行状态；统计分离不代表真实工艺状态。"
    result["engineering_hint"] = {
        "state_exploration": evidence + "是否作为正常状态候选，仍需结合优选区域与性能评价人工确认。",
        "cluster_assistance": evidence + "请确认数据是否存在离散运行状态，或只是连续负荷变化形成的切分。",
    }
    return result
