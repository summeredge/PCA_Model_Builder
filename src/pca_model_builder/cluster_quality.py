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
    segment_ids=None,
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
        "temporal_metrics": None, "top_features": [], "feature_contrasts": [],
        "engineering_hint": {}, "unavailable_reasons": {}, "group_profiles": [],
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
    result["sample_count"] = len(values)
    result["n_components"] = values.shape[1]
    if values.shape[1] < 2 or not np.isfinite(values).all():
        reasons["analysis"] = "需要至少两个主元且得分必须全部有效"
        return result
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    result["cumulative_explained_variance"] = float(ratios[:values.shape[1]].sum()) if len(ratios) >= values.shape[1] and np.isfinite(ratios[:values.shape[1]]).all() else None
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
                adjacent = position < len(index) and index[position] - index[position - 1] == interval and (segment_ids is None or segment_ids[position] == segment_ids[position - 1])
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
            numeric = (
                pd.to_numeric(aligned[tag], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
                if tag in aligned and not pd.api.types.is_datetime64_any_dtype(aligned[tag])
                else pd.Series(np.nan, index=scores.index)
            )
            std = numeric.std(ddof=0)
            means = [numeric.iloc[np.flatnonzero(labels == cluster)].mean() for cluster in clusters]
            difference = float(max(means) - min(means)) if np.isfinite(means).all() else None
            if difference is not None and not np.isfinite(difference):
                difference = None
            reason = (
                "需要至少两个 Cluster" if len(clusters) < 2 else
                "某些 Cluster 无有效均值或统计不可计算" if difference is None else
                "有效样本不足或标准差不可计算" if numeric.count() < 2 or not np.isfinite(std) else
                "精确常量，标准化差异不可计算" if std <= 0 else None
            )
            strength = difference / float(std) if reason is None else None
            if strength is not None and not np.isfinite(strength):
                strength, reason = None, "标准化差异不可计算"
            result["feature_contrasts"].append({
                "tag": tag, "mean_difference": difference,
                "standardized_difference": strength,
                "cluster_means": {str(cluster): float(mean) if np.isfinite(mean) else None for cluster, mean in zip(clusters, means, strict=True)},
                "unavailable_reason": reason,
            })
            overall = numeric.mean()
            finite_means = [mean for mean in means if np.isfinite(mean)]
            for cluster, mean in zip(clusters, means, strict=True):
                profile = next((item for item in result["group_profiles"] if item["cluster_id"] == str(cluster)), None)
                if profile is None:
                    count = int(np.count_nonzero(labels == cluster))
                    profile = {"cluster_id": str(cluster), "sample_count": count, "share": count / len(labels), "variables": []}
                    result["group_profiles"].append(profile)
                unavailable = ("本组无有效均值" if not np.isfinite(mean) else
                               "有效样本不足" if numeric.count() < 2 else
                               "总体标准差不可计算" if not np.isfinite(std) else
                               "精确常量" if std <= 0 else None)
                deviation = float((mean - overall) / std) if unavailable is None else None
                if deviation is not None and not np.isfinite(deviation):
                    deviation, unavailable = None, "相对总体偏离不可计算"
                rank = 1 + sum(bool(value > mean) for value in finite_means) if np.isfinite(mean) else None
                profile["variables"].append({"tag": tag, "mean": float(mean) if np.isfinite(mean) else None,
                    "overall_mean": float(overall) if np.isfinite(overall) else None,
                    "deviation": deviation, "rank": rank, "rank_count": len(finite_means),
                    "tied": sum(bool(value == mean) for value in finite_means) > 1,
                    "unavailable_reason": unavailable})
        result["feature_contrasts"] = sorted(
            result["feature_contrasts"],
            key=lambda item: (item["standardized_difference"] is None, -(item["standardized_difference"] or 0)),
        )
        result["top_features"] = [
            item for item in result["feature_contrasts"] if item["unavailable_reason"] is None
        ][:5]
    for profile in result["group_profiles"]:
        profile["top_variables"] = sorted(
            [item for item in profile["variables"] if item["deviation"] is not None and item["deviation"] != 0],
            key=lambda item: -abs(item["deviation"]),
        )[:5]
        profile["comparison_variables"] = [
            item for feature in result["top_features"] if feature["standardized_difference"] > 0
            for item in profile["variables"] if item["tag"] == feature["tag"]
        ][:5]
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
