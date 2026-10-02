from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingConfig
from .validation import validation_context_start


def _cluster_source(reference: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(?:(state-exploration-.+)-)?(cluster_\d+)(?:-candidate-\d+)?", reference, re.IGNORECASE)
    if match:
        cluster_id = f"cluster_{int(match[2].split('_')[1]):03d}"
        return match[1] or "", cluster_id
    legacy = re.match(r"^cluster-(\d+)-", reference)
    return ("", f"cluster_{int(legacy[1]):03d}") if legacy else None


def model_quality_summary(
    model: Any,
    scores: pd.DataFrame,
    windows: Sequence[Mapping[str, Any]],
    sample_interval_minutes: int,
    model_purpose: str,
    *,
    cluster_series: Mapping[str, pd.DataFrame] | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
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
        cluster = _cluster_source(reference) if source == "cluster" else None
        identity = "-".join(filter(None, cluster)) if cluster else reference or "unavailable"
        label = f"工况组 {int(cluster[1].split('_')[1])}" if cluster else "工况组来源未记录" if source == "cluster" else source
        key = source, identity
        row = sources.setdefault(key, {"source": source, "source_ref": identity, "label": label, "samples": 0})
        row["samples"] += int(item["effective_samples"])
    source_rows = list(sources.values())
    for row in source_rows:
        row["share"] = row["samples"] / len(scores) if len(scores) else None
    cluster_count = sum(
        row["source"] == "cluster" and row["label"] != "工况组来源未记录"
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
            f"训练样本来自 {cluster_count} 个可回溯工况组；"
            "请确认这些工况是否均应属于当前正常状态，来源数量本身不能证明状态混杂。"
        )
    if model_purpose == "exploratory":
        messages.insert(0, "当前为探索草稿模型，质量摘要仅用于探索，不能据此确认正常状态模型。")
    return {
        "training_samples": len(scores),
        "retained_explained_variance": retained,
        "pc1_pc2_explained_variance": float(ratios[:2].sum()) if model.n_components >= 2 else None,
        "statistics": statistics,
        "training_condition_diagnostic": _training_condition_diagnostic(
            model, scores, used, sample_interval_minutes, cluster_series or {}, preprocessing_config,
        ),
        "training_data": {
            "sources": source_rows,
            "traceable_cluster_count": cluster_count or None,
            "unattributed_samples": sum(row["samples"] for row in source_rows if row["label"] == "工况组来源未记录"),
            "time_start": valid_times.min().isoformat() if len(valid_times) else None,
            "time_end": valid_times.max().isoformat() if len(valid_times) else None,
            "effective_sample_hours": len(valid_times) * sample_interval_minutes / 60,
        },
        "engineering_messages": messages,
        "notice": "训练期自检使用全部训练评分，控制限也由训练数据估计，低超限率不能替代独立验证或证明正常状态代表性。",
        "rules": "工程提示采用经验规则：累计解释率参考线 80%；95% 超限率 >10% 或 99% 超限率 >3% 视为频繁超限。趋势仅比较同一窗口连续段内至少 20 个有效点的四块中位数，极差超过 95% 控制限的 25% 提示变化；不跨窗口或时间缺口比较，不是统计平稳性检验。",
    }


def _training_condition_diagnostic(
    model: Any,
    scores: pd.DataFrame,
    windows: Sequence[Mapping[str, Any]],
    interval: int,
    cluster_series: Mapping[str, pd.DataFrame],
    config: PreprocessingConfig | None,
) -> dict[str, Any]:
    """Describe window provenance and observed labels without inferring clusters."""
    count = len(scores)
    times = pd.DatetimeIndex(scores.index)
    step = pd.Timedelta(minutes=interval)
    owner = np.full(count, -1, dtype=int)
    for number, window in enumerate(windows):
        mask = (times >= pd.Timestamp(window["start"])) & (times <= pd.Timestamp(window["end"]))
        # Ambiguous/overlapping provenance is never assigned twice.
        owner[mask] = np.where(owner[mask] == -1, number, -2)
    valid = np.isfinite(scores[["t2", "spe"]].to_numpy(dtype=float)).all(axis=1) & ~times.isna()
    if "score_valid" in scores:
        valid &= scores["score_valid"].fillna(False).to_numpy(dtype=bool)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    group_keys: list[tuple[str, str] | None] = [None] * count
    labels: list[str | None] = [None] * count
    segments: list[str | None] = [None] * count
    reliable = np.zeros(count, dtype=bool)
    for number, window in enumerate(windows):
        positions = np.flatnonzero(owner == number)
        reference = str(window.get("source_ref") or "")
        source = _cluster_source(reference) if window.get("source") == "cluster" else None
        if source:
            groups.setdefault(source, {"cluster_id": source[1], "source_ref": "-".join(filter(None, source)), "label": f"工况组 {int(source[1].split('_')[1])}"})
        observed = cluster_series.get(str(window.get("id", "")))
        if observed is not None and not observed.index.has_duplicates and {"cluster_id", "segment_id"} <= set(observed.columns):
            observed = list(observed.reindex(times[positions])[["cluster_id", "segment_id"]].itertuples(index=False, name=None))
        else:
            observed = None
        for offset, position in enumerate(positions):
            group_keys[position] = source
            labels[position] = source[1] if source else None
            for segment in window.get("segments", []):
                if pd.Timestamp(segment["start"]) <= times[position] <= pd.Timestamp(segment["end"]):
                    segments[position] = str(segment["id"])
                    break
            if observed is not None:
                label, segment_id = observed[offset]
                if pd.notna(label) and re.fullmatch(r"cluster_\d+", str(label), re.IGNORECASE) and pd.notna(segment_id):
                    labels[position] = f"cluster_{int(str(label).split('_')[1]):03d}"
                    segments[position] = f"{segments[position]}:observed-{segment_id}"
                    reliable[position] = True
    breaks = np.ones(count, dtype=bool)
    for position in range(1, count):
        breaks[position] = not (
            valid[position] and valid[position - 1] and owner[position] >= 0
            and owner[position] == owner[position - 1]
            and segments[position] == segments[position - 1]
            and times[position] - times[position - 1] == step
        )
    segment_numbers = breaks.cumsum()
    t2 = scores["t2"].to_numpy(dtype=float)
    spe = scores["spe"].to_numpy(dtype=float)
    ratios = {}
    for key, values, limits in (("t2", t2, model.t2_limits), ("spe", spe, model.q_limits)):
        field = f"{key}_limit_ratio"
        ratios[key] = scores[field].to_numpy(dtype=float) if field in scores else values / limits[0.95] if limits[0.95] > 0 else np.full(count, np.nan)

    def rates(mask: np.ndarray) -> dict[str, Any]:
        size = int(mask.sum())
        return {
            "samples": size,
            "t2_95_exceedance_rate": float(np.mean(t2[mask] >= model.t2_limits[0.95])) if size else None,
            "spe_95_exceedance_rate": float(np.mean(spe[mask] >= model.q_limits[0.95])) if size else None,
            "overall_95_exceedance_rate": float(np.mean((t2[mask] >= model.t2_limits[0.95]) | (spe[mask] >= model.q_limits[0.95]))) if size else None,
        }

    rows = []
    for key, group in groups.items():
        mask = np.array([item == key for item in group_keys], dtype=bool) & valid
        if not mask.any():
            continue
        longest = run = 0
        for position in range(count):
            exceeded = mask[position] and (t2[position] >= model.t2_limits[0.95] or spe[position] >= model.q_limits[0.95])
            run = (1 if breaks[position] else run + 1) if exceeded else 0
            longest = max(longest, run)
        medians = {}
        for statistic in ("t2", "spe"):
            values = ratios[statistic][mask & np.isfinite(ratios[statistic])]
            medians[f"{statistic}_95_ratio_median"] = float(np.median(values)) if len(values) else None
        rows.append({
            **group, **rates(mask), **medians,
            "share": float(mask.sum() / count) if count else None,
            "t2_99_exceedance_rate": float(np.mean(t2[mask] >= model.t2_limits[0.99])),
            "spe_99_exceedance_rate": float(np.mean(spe[mask] >= model.q_limits[0.99])),
            "longest_overall_95_minutes": longest * interval,
        })
    # Reuse the project's context calculation, including any resampling history.
    origin = pd.Timestamp("2000-01-01")
    full_history = config is not None and config.filter_method == "first_order"
    width = None if full_history else int((origin - validation_context_start(origin, config)).total_seconds() / 60) if config else 0
    available = bool(valid.any() and np.all(reliable[valid]) and np.all(owner[valid] >= 0))
    near = np.zeros(count, dtype=bool)
    near_delta = np.zeros(count + 1, dtype=int)
    switch_positions = []
    if available:
        for position in range(1, count):
            if not breaks[position] and labels[position] != labels[position - 1]:
                switch_positions.append(position)
                start = int(np.searchsorted(segment_numbers, segment_numbers[position], side="left"))
                end = int(np.searchsorted(segment_numbers, segment_numbers[position], side="right"))
                if not full_history:
                    start = max(start, int(times.searchsorted(times[position] - pd.Timedelta(minutes=width))))
                    end = min(end, int(times.searchsorted(times[position] + pd.Timedelta(minutes=width), side="right")))
                near_delta[start] += 1
                near_delta[end] -= 1
        near = near_delta[:-1].cumsum() > 0
    switches = {
        "available": available, "context_minutes": width,
        "context_policy": "full_segment_history" if full_history else "finite_context",
        "switch_count": len(switch_positions) if available else None,
        "stable": rates(valid & ~near) if available else None,
        "near_switch": rates(valid & near) if available else None,
        "message": "同一窗口、同一物理连续段且相邻时间间隔等于采样周期时才计切换；结果只做描述性比较，不表示因果。" if available else "当前训练样本缺少连续工况组标记，无法计算工况切换区诊断。",
    }
    messages = []
    if len(rows) > 1:
        highest = max(rows, key=lambda row: row["spe_95_exceedance_rate"])
        if highest["spe_95_exceedance_rate"] > min(row["spe_95_exceedance_rate"] for row in rows):
            messages.append(f"{highest['label']} 的 SPE 95%超限率在可追溯工况组中最高，建议检查该工况是否应纳入当前正常状态模型。该比较不设合格阈值。")
    if available and switches["stable"]["samples"] and switches["near_switch"]["samples"]:
        if any(switches["near_switch"][key] > switches["stable"][key] for key in ("t2_95_exceedance_rate", "spe_95_exceedance_rate")):
            messages.append("工况切换附近的 T²/SPE 超限比例高于稳定工况区，建议检查训练窗口是否包含较多状态过渡过程；该比较不表示切换造成异常。")
    timeline = []
    if rows:
        for position in range(count):
            timeline.append({
                "timestamp": times[position].isoformat() if pd.notna(times[position]) else None,
                "cluster_id": labels[position] if valid[position] else None,
                "window_id": str(windows[owner[position]].get("id", owner[position])) if owner[position] >= 0 else None,
                "segment_id": int(segment_numbers[position]), "break_before": bool(breaks[position]),
                "t2_limit_ratio": float(ratios["t2"][position]) if valid[position] and np.isfinite(ratios["t2"][position]) else None,
                "spe_limit_ratio": float(ratios["spe"][position]) if valid[position] and np.isfinite(ratios["spe"][position]) else None,
                "near_switch": bool(near[position]) if available else None,
            })
    traced = sum(row["samples"] for row in rows)
    return {
        "available": bool(rows), "groups": rows, "traceable_samples": traced,
        "unattributed_samples": count - traced, "timeline": timeline,
        "switch_diagnostic": switches, "engineering_messages": messages,
        "message": "工况组（Cluster）表示状态探索中具有相似运行特征的一组样本，不自动代表正常或异常。" if rows else "当前训练窗口无法追溯到状态探索工况组，无法执行按工况组诊断。",
        "duration_semantics": "coverage",
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
