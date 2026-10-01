from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def model_quality_summary(
    model: Any,
    scores: pd.DataFrame,
    windows: Sequence[Mapping[str, Any]],
    sample_interval_minutes: int,
    model_purpose: str,
) -> dict[str, Any]:
    """Training self-check from complete scores, before chart downsampling."""
    ratios = np.asarray(model.explained_variance_ratio, dtype=float)
    retained = float(ratios[: model.n_components].sum())
    used = [item for item in windows if item.get("status") == "used"]
    statistics = {
        key: _statistic_summary(scores, key, limits, used, sample_interval_minutes)
        for key, limits in (("t2", model.t2_limits), ("spe", model.q_limits))
    }
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for item in used:
        source = str(item["source"])
        reference = str(item.get("source_ref") or "")
        cluster = re.search(r"(?:^|-)(cluster_\d+)-candidate-\d+$", reference, re.IGNORECASE) if source == "cluster" else None
        legacy_cluster = re.match(r"^cluster-(\d+)-", reference) if source == "cluster" else None
        identity = (
            reference[: cluster.end(1)] if cluster else
            f"Cluster_{int(legacy_cluster[1]):03d}" if legacy_cluster else
            reference or "unavailable"
        )
        label = (
            cluster[1].replace("cluster_", "Cluster_") if cluster else
            f"Cluster_{int(legacy_cluster[1]):03d}" if legacy_cluster else
            "Cluster 来源未记录" if source == "cluster" else source
        )
        key = source, identity
        row = sources.setdefault(key, {"source": source, "source_ref": identity, "label": label, "samples": 0})
        row["samples"] += int(item["effective_samples"])
    source_rows = list(sources.values())
    for row in source_rows:
        row["share"] = row["samples"] / len(scores) if len(scores) else None
    cluster_count = sum(
        row["source"] == "cluster" and row["label"] != "Cluster 来源未记录"
        for row in source_rows
    )
    times = pd.DatetimeIndex(scores.index)
    valid_times = times[~times.isna()]
    messages = []
    spe, t2 = statistics["spe"], statistics["t2"]
    if not spe["valid_samples"] or not t2["valid_samples"]:
        messages.append("有效评分不足，暂不能判断模型能力。")
    elif spe["frequent_exceedance"]:
        messages.append(
            "模型残差频繁超限，当前数据可能未被充分描述。"
            "建议检查正常状态是否包含多个工况、主元数量是否不足，以及输入变量是否需要优化。"
        )
    elif t2["frequent_exceedance"]:
        messages.append(
            "主元空间位置频繁偏离训练控制边界；即使解释率较高，也应检查运行状态和训练状态覆盖。"
        )
    elif spe["trend"] == "stable" and retained >= 0.8:
        messages.append(
            "模型保留主要过程变化，剩余变化表现为较稳定残差，训练期超限率较低。"
            "当前主元数量可作为候选，仍需独立验证。"
        )
    elif retained < 0.8:
        messages.append(
            "累计解释率低于 80%；仅凭解释率不能确定模型不足，"
            "建议结合残差趋势、主元数量和训练状态组成检查。"
        )
    else:
        messages.append("未发现频繁超限，但残差趋势仍需检查；请结合独立验证判断模型可靠性。")
    if spe["trend"] == "changing":
        messages.append("连续训练段内 SPE 分块中位数变化较大，建议回看时间趋势与工况变化。")
    if spe["frequent_exceedance"] and t2["frequent_exceedance"]:
        messages.append("T² 也频繁超限，建议同时检查主元空间位置偏离与运行状态。")
    if cluster_count > 1:
        messages.append(
            f"训练样本来自 {cluster_count} 个可回溯 Cluster 来源；"
            "请确认这些工况是否均应属于正常状态，来源数量本身不能证明状态混杂。"
        )
    if model_purpose == "exploratory":
        messages.insert(0, "当前为探索草稿模型，质量摘要仅用于探索，不能据此确认正常状态模型。")
    return {
        "training_samples": len(scores),
        "retained_explained_variance": retained,
        "pc1_pc2_explained_variance": float(ratios[:2].sum()) if model.n_components >= 2 else None,
        "statistics": statistics,
        "training_data": {
            "sources": source_rows,
            "traceable_cluster_count": cluster_count or None,
            "unattributed_samples": sum(row["samples"] for row in source_rows if row["label"] == "Cluster 来源未记录"),
            "time_start": valid_times.min().isoformat() if len(valid_times) else None,
            "time_end": valid_times.max().isoformat() if len(valid_times) else None,
            "effective_sample_hours": len(valid_times) * sample_interval_minutes / 60,
        },
        "engineering_messages": messages,
        "notice": "训练期自检使用全部训练评分，控制限也由训练数据估计，低超限率不能替代独立验证或证明正常状态代表性。",
        "rules": "工程提示采用经验规则：累计解释率参考线 80%；95% 超限率 >10% 或 99% 超限率 >3% 视为频繁超限。趋势仅比较同一窗口连续段内至少 20 个有效点的四块中位数，极差超过 95% 控制限的 25% 提示变化；不跨窗口或时间缺口比较，不是统计平稳性检验。",
    }


def _statistic_summary(
    scores: pd.DataFrame,
    key: str,
    limits: Mapping[float, float],
    windows: Sequence[Mapping[str, Any]],
    interval: int,
) -> dict[str, Any]:
    values = scores[key].to_numpy(dtype=float)
    valid = np.isfinite(values)
    if "score_valid" in scores:
        valid &= scores["score_valid"].to_numpy(dtype=bool)
    finite = values[valid]
    rates = {
        str(int(level * 100)): float(np.mean(finite >= limits[level])) if len(finite) else None
        for level in (0.95, 0.99)
    }
    checked = 0
    insufficient = 0
    changing = False
    valid_scores = pd.Series(valid, index=scores.index)
    for window in windows:
        selected = scores.loc[window["start"]:window["end"]]
        # Invalid scores and gaps break continuity; independent windows stay separate.
        breaks = selected.index.to_series().diff().ne(pd.Timedelta(minutes=interval))
        good = valid_scores.reindex(selected.index)
        breaks |= ~good | ~good.shift(fill_value=False)
        for _, segment in selected.groupby(breaks.cumsum()):
            segment_values = segment.loc[good.loc[segment.index], key].to_numpy(dtype=float)
            if len(segment_values) < 20:
                insufficient += bool(len(segment_values))
                continue
            checked += 1
            medians = [float(np.median(block)) for block in np.array_split(segment_values, 4)]
            changing |= bool(np.ptp(medians) > 0.25 * limits[0.95])
    return {
        "valid_samples": len(finite),
        "invalid_samples": len(values) - len(finite),
        "mean": float(finite.mean()) if len(finite) else None,
        "limits": {str(int(level * 100)): float(limits[level]) for level in (0.95, 0.99)},
        "exceedance_rates": rates,
        "frequent_exceedance": bool(len(finite) and (rates["95"] > 0.10 or rates["99"] > 0.03)),
        "trend": "changing" if changing else "stable" if checked and not insufficient else "insufficient",
        "trend_segments_checked": checked,
        "trend_segments_insufficient": insufficient,
    }
