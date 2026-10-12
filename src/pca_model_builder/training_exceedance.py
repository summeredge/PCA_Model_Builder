"""Read-only interval diagnosis of a training run, using its fixed data source."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import web
from .scoring_core import aggregate_tag_contributions, spe_feature_contributions, t2_feature_contributions
from .tag_config import engineering_ranges
from .training import build_training_matrix
from .trend import trend_payload_data


def _mean_feature_contribution(model, dynamic: pd.DataFrame, timestamps: pd.Index, calculate) -> np.ndarray:
    feature_names = list(model.feature_names)
    total = np.zeros(len(feature_names), dtype=float)
    for offset in range(0, len(timestamps), 1024):
        batch = dynamic.loc[timestamps[offset:offset + 1024]]
        if not batch.columns.equals(pd.Index(feature_names)):
            batch = batch.loc[:, feature_names]
        total += calculate(
            batch.to_numpy(dtype=float),
            feature_names=feature_names,
            mean=model.mean,
            scale=model.scale,
            components=model.components,
            eigenvalues=model.eigenvalues,
        ).sum(axis=0)
    return total / len(timestamps)


def interval_diagnosis(model, dynamic: pd.DataFrame, scores: pd.DataFrame,
                       start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """Use every valid scored row, inclusive of both timestamp boundaries."""
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("分析区间开始时间不得晚于结束时间，且时间必须有效")
    mask = (scores.index >= start) & (scores.index <= end)
    mask &= scores["score_valid"].to_numpy(dtype=bool)
    mask &= np.isfinite(scores[["t2", "spe"]].to_numpy()).all(axis=1)
    selected = scores.loc[mask]
    statistics, contributions = {}, {}
    for statistic, limits, calculate in (
        ("t2", model.t2_limits, t2_feature_contributions),
        ("spe", model.q_limits, spe_feature_contributions),
    ):
        values = selected[statistic]
        summary = {"mean": float(values.mean()) if len(values) else None,
                   "max": float(values.max()) if len(values) else None,
                   "max_limit_ratio": float(values.max() / limits[0.95])
                   if len(values) and limits[0.95] > 0 else None}
        for confidence in (95, 99):
            count = int((values >= limits[confidence / 100]).sum())
            summary[f"exceedance_{confidence}"] = {
                "count": count, "rate": count / len(values) if len(values) else None}
        statistics[statistic] = summary
        exceeded = selected.index[values >= limits[0.95]]
        rows = []
        if len(exceeded):
            # Average feature magnitudes first: percentages are relative to interval totals.
            mean_contribution = _mean_feature_contribution(model, dynamic, exceeded, calculate)
            total = float(mean_contribution.sum())
            if total > 0:
                for item in aggregate_tag_contributions(model.feature_names, mean_contribution)[:10]:
                    rows.append({**item.__dict__,
                                 "mean_contribution": total * item.contribution_pct / 100})
        contributions[statistic] = {"sample_count": len(exceeded), "rows": rows}
    return {"start": start.isoformat(), "end": end.isoformat(),
            "valid_samples": len(selected), "statistics": statistics,
            "contributions": contributions}


def _load_summary(values: pd.Series, sample_count: int) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return {
        "valid_samples": int(len(numeric)),
        "missing_rate": 1.0 - len(numeric) / sample_count if sample_count else None,
        "mean": float(numeric.mean()) if len(numeric) else None,
        "median": float(np.median(numeric)) if len(numeric) else None,
        "p10": float(np.quantile(numeric, 0.1)) if len(numeric) else None,
        "p90": float(np.quantile(numeric, 0.9)) if len(numeric) else None,
        "minimum": float(numeric.min()) if len(numeric) else None,
        "maximum": float(numeric.max()) if len(numeric) else None,
    }


def _mean_tag_contributions(model, feature_mean: np.ndarray) -> dict[str, dict[str, Any]]:
    total = float(feature_mean.sum())
    aggregated = aggregate_tag_contributions(model.feature_names, feature_mean)
    by_tag = {
        item.tag: {
            "mean": total * item.contribution_pct / 100.0,
            "share_pct": item.contribution_pct if total > 0 else None,
            "lag_start_minutes": item.lag_start_minutes if total > 0 and item.contribution_pct > 0 else None,
            "lag_end_minutes": item.lag_end_minutes if total > 0 and item.contribution_pct > 0 else None,
        }
        for item in aggregated
    }
    return by_tag


def compare_diagnosis(
    model,
    dynamic: pd.DataFrame,
    scores: pd.DataFrame,
    load_values: pd.Series,
    load_tag: str,
    anomaly_start: pd.Timestamp,
    anomaly_end: pd.Timestamp,
    reference_start: pd.Timestamp,
    reference_end: pd.Timestamp,
) -> dict[str, Any]:
    """Compare high-SPE rows with low-SPE rows without labeling reference residuals anomalous."""
    if pd.isna(anomaly_start) or pd.isna(anomaly_end) or anomaly_start > anomaly_end:
        raise ValueError("异常区间开始时间不得晚于结束时间，且时间必须有效")
    if pd.isna(reference_start) or pd.isna(reference_end) or reference_start > reference_end:
        raise ValueError("参考区间开始时间不得晚于结束时间，且时间必须有效")
    if anomaly_start <= reference_end and reference_start <= anomaly_end:
        raise ValueError("异常区间与参考区间不能重叠（边界时间也计入区间）")

    try:
        anomaly_interval = (scores.index >= anomaly_start) & (scores.index <= anomaly_end)
        reference_interval = (scores.index >= reference_start) & (scores.index <= reference_end)
    except (TypeError, ValueError) as error:
        raise ValueError("区间时区与训练数据不一致") from error
    score_valid = scores["score_valid"].to_numpy(dtype=bool)
    finite_spe = np.isfinite(scores["spe"].to_numpy(dtype=float))
    valid = score_valid & finite_spe
    limit = float(model.q_limits[0.95])
    spe_values = scores["spe"].to_numpy(dtype=float)
    anomaly_mask = anomaly_interval & valid & (spe_values >= limit)
    reference_mask = reference_interval & valid & (spe_values < limit)

    anomaly_index = scores.index[anomaly_mask]
    reference_index = scores.index[reference_mask]
    aligned_load = load_values.reindex(scores.index)

    def group_summary(interval_mask: np.ndarray, selected_index: pd.Index) -> dict[str, Any]:
        selected = scores.loc[selected_index, "spe"].to_numpy(dtype=float)
        selected_load = _load_summary(aligned_load.reindex(selected_index), len(selected))
        exceedances = selected >= limit
        return {
            "interval_valid_samples": int((interval_mask & valid).sum()),
            "sample_count": int(len(selected)),
            "spe_mean": float(selected.mean()) if len(selected) else None,
            "spe_exceedance_count": int(exceedances.sum()),
            "spe_exceedance_rate": float(exceedances.mean()) if len(selected) else None,
            "load": selected_load,
        }

    anomaly = group_summary(anomaly_interval, anomaly_index)
    reference = group_summary(reference_interval, reference_index)
    anomaly_load, reference_load = anomaly["load"], reference["load"]
    range_overlap = None
    if anomaly_load["valid_samples"] and reference_load["valid_samples"]:
        range_overlap = max(anomaly_load["minimum"], reference_load["minimum"]) <= min(
            anomaly_load["maximum"], reference_load["maximum"]
        )

    analysis_available = anomaly["sample_count"] >= 2 and reference["sample_count"] >= 2
    load_samples_available = anomaly_load["valid_samples"] >= 2 and reference_load["valid_samples"] >= 2
    load_p10_p90_overlap = None
    if load_samples_available:
        load_p10_p90_overlap = max(anomaly_load["p10"], reference_load["p10"]) <= min(
            anomaly_load["p90"], reference_load["p90"]
        )
    load_comparability_status = (
        "insufficient_evidence" if not load_samples_available
        else "body_disjoint" if load_p10_p90_overlap is False
        else "manual_review"
    )
    warnings = []
    if not analysis_available:
        warnings.append("至少需要异常组和参考组各2个有效纳入样本；当前未生成贡献对比。")
    if not load_samples_available:
        warnings.append("负荷数据不足，不能认定同负荷；" + (
            "SPE 数值对照仅为描述性、未匹配负荷比较。" if analysis_available
            else "SPE 样本不足，未生成贡献对比。"
        ))
    elif range_overlap is False:
        warnings.append("两组负荷范围不重叠，请重新选取参考区间。")
    elif range_overlap is True:
        warnings.append("负荷范围有重叠；请核对均值、中位数及分位数，不自动判定为同负荷。")
        if load_p10_p90_overlap is False:
            warnings.append("主体负荷分布不重叠/不匹配，应重新选择参考区间。")

    contribution_rows = []
    if analysis_available:
        anomaly_mean = _mean_feature_contribution(model, dynamic, anomaly_index, spe_feature_contributions)
        reference_mean = _mean_feature_contribution(model, dynamic, reference_index, spe_feature_contributions)
        anomaly_tags = _mean_tag_contributions(model, anomaly_mean)
        reference_tags = _mean_tag_contributions(model, reference_mean)
        for tag, anomaly_item in sorted(
            anomaly_tags.items(), key=lambda item: (-item[1]["mean"], item[0])
        )[:10]:
            reference_item = reference_tags[tag]
            contribution_rows.append({
                "tag": tag,
                "anomaly_mean": anomaly_item["mean"],
                "reference_mean": reference_item["mean"],
                "absolute_difference": abs(anomaly_item["mean"] - reference_item["mean"]),
                "anomaly_share_pct": anomaly_item["share_pct"],
                "reference_share_pct": reference_item["share_pct"],
                "anomaly_lag_start_minutes": anomaly_item["lag_start_minutes"],
                "anomaly_lag_end_minutes": anomaly_item["lag_end_minutes"],
                "reference_lag_start_minutes": reference_item["lag_start_minutes"],
                "reference_lag_end_minutes": reference_item["lag_end_minutes"],
            })

    return {
        "anomaly": anomaly,
        "reference": reference,
        "load_tag": load_tag,
        "load_ranges_overlap": range_overlap,
        "load_p10_p90_overlap": load_p10_p90_overlap,
        "load_comparability_status": load_comparability_status,
        "load_statistics_available": load_samples_available,
        "analysis_available": analysis_available,
        "contribution_basis": "mean_residual_squared",
        "contributions": contribution_rows,
        "warnings": warnings,
    }


def _requested_range(payload: dict[str, Any], start_key: str, end_key: str,
                     timestamps: pd.DatetimeIndex, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start = pd.Timestamp(web._required_text(payload, start_key))
        end = pd.Timestamp(web._required_text(payload, end_key))
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("invalid range")
        timestamps >= start
        timestamps <= end
        return start, end
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label}区间时间无效或时区与训练数据不一致") from error


def _raw_comparison_range(indexed: pd.DataFrame, tags: list[str], config,
                          tag_configs: dict[str, Any], start: pd.Timestamp,
                          end: pd.Timestamp) -> dict[str, Any]:
    context = max((end - start) * 0.1, pd.Timedelta(minutes=config.sample_interval_minutes * 5))
    view_start = max(indexed.index.min(), start - context)
    view_end = min(indexed.index.max(), end + context)
    series = []
    if view_start <= view_end:
        trend = trend_payload_data(indexed, tags, config, view_start, view_end,
                                   "raw", tag_configs)
        for tag in tags:
            series.append({
                "tag": tag,
                "unit": tag_configs.get(tag, {}).get("unit", ""),
                "rows": [{"timestamp": row["timestamp"], "value": row[f"{tag}__raw"],
                          "gap_start": row["raw_gap_start"]} for row in trend["rows"]],
            })
    else:
        series = [{"tag": tag, "unit": tag_configs.get(tag, {}).get("unit", ""), "rows": []}
                  for tag in tags]
        view_start, view_end = start, end
    return {"start": view_start.isoformat(), "end": view_end.isoformat(),
            "analysis_start": start.isoformat(), "analysis_end": end.isoformat(),
            "series": series}


def _run_source(run_id: str, load_tag: str | None = None):
    model_path = web.RUNS_DIR / run_id / "model.pcamodel"
    if not model_path.is_file():
        raise ValueError("训练运行数据已不可用，请重新训练")
    model, manifest = web.load_model_package(model_path)
    if manifest["schema_version"] != 5:
        raise ValueError("旧模型未保留可复现的训练诊断数据，请重新训练")
    record = web.modeling_snapshot_payload(run_id)
    if not record["available"]:
        raise ValueError("训练运行缺少建模快照，无法诊断")
    snapshot = record["snapshot"]
    if snapshot["candidate_model"]["sha256"] != web._file_sha256(model_path):
        raise ValueError("训练模型与建模快照不一致，请重新训练")
    source, stored = snapshot["data_source"], manifest["config"]
    config = web._preprocessing_config(stored)
    request = {"file_id": source["file_id"], "timestamp_column": source["timestamp_column"],
               "encoding": source.get("encoding", "utf-8-sig"),
               "modeling_eligibility": snapshot["modeling_eligibility"]}
    _, source_path, _ = web._upload_source(request)
    if not source.get("sha256") or web._file_sha256(source_path) != source["sha256"]:
        raise ValueError("原始训练数据已变化，无法诊断，请重新训练")
    loaded = web._load_modeling_upload(
        request, [*stored["tags"], *(condition.column for condition in config.state_filters),
                  *([load_tag] if load_tag else [])],
        "原始训练数据缺少 Tag：")
    return model, manifest, snapshot, loaded, config


def training_exceedance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = web._validated_id(web._required_text(payload, "run_id"), "run_id")
    action = payload.get("action", "overview")
    if action not in {"overview", "interval", "trend", "compare", "compare_trend"}:
        raise ValueError("超限诊断 action 无效")
    load_tag = None
    if action in {"compare", "compare_trend"}:
        load_tag = web._required_text(payload, "load_tag")
    with web._lifecycle_lock(run_id):
        model, manifest, snapshot, loaded, config = _run_source(
            run_id, load_tag
        )
        stored, source = manifest["config"], snapshot["data_source"]
        if load_tag and load_tag not in loaded.metadata.numeric_candidate_columns:
            raise ValueError("请选择当前训练源数据中的数值型负荷 Tag")
        timestamps = pd.DatetimeIndex(loaded.frame[source["timestamp_column"]])
        start = end = None
        anomaly_start = anomaly_end = reference_start = reference_end = None
        if action in {"interval", "trend"}:
            try:
                start = pd.Timestamp(web._required_text(payload, "start"))
                end = pd.Timestamp(web._required_text(payload, "end"))
                if pd.isna(start) or pd.isna(end) or start > end:
                    raise ValueError("invalid range")
                # Compare here to reject incompatible timezone inputs at the API boundary.
                timestamps >= start
                timestamps <= end
            except (ValueError, TypeError) as error:
                raise ValueError("分析区间时间无效或时区与训练数据不一致") from error
        elif action in {"compare", "compare_trend"}:
            anomaly_start, anomaly_end = _requested_range(
                payload, "anomaly_start", "anomaly_end", timestamps, "异常"
            )
            reference_start, reference_end = _requested_range(
                payload, "reference_start", "reference_end", timestamps, "参考"
            )
            if anomaly_start <= reference_end and reference_start <= anomaly_end:
                raise ValueError("异常区间与参考区间不能重叠（边界时间也计入区间）")
        if action == "trend":
            tag = web._required_text(payload, "tag")
            if tag not in stored["tags"]:
                raise ValueError("请选择当前模型的原始 Tag")
            indexed = loaded.frame.set_index(source["timestamp_column"])
            context = max((end - start) * 0.1, pd.Timedelta(minutes=config.sample_interval_minutes * 5))
            view_start, view_end = max(indexed.index.min(), start - context), min(indexed.index.max(), end + context)
            trend = trend_payload_data(indexed, [tag], config, view_start, view_end,
                                       "raw", stored["tag_configs"])
            rows = [{"timestamp": row["timestamp"], "value": row[f"{tag}__raw"],
                     "gap_start": row["raw_gap_start"]} for row in trend["rows"]]
            if not any(row["value"] is not None for row in rows):
                raise ValueError("该 Tag 在当前区间及上下文内没有有效原始数据")
            return {"run_id": run_id, "tag": tag, "unit": stored["tag_configs"][tag].get("unit", ""),
                    "start": view_start.isoformat(), "end": view_end.isoformat(), "rows": rows,
                    "analysis_start": start.isoformat(), "analysis_end": end.isoformat()}
        if action == "compare_trend":
            tag = web._required_text(payload, "tag")
            if tag not in stored["tags"]:
                raise ValueError("请选择当前模型的原始 Tag")
            tags = list(dict.fromkeys([tag, load_tag]))
            indexed = loaded.frame.set_index(source["timestamp_column"])
            return {
                "run_id": run_id,
                "tag": tag,
                "load_tag": load_tag,
                "anomaly": _raw_comparison_range(
                    indexed, tags, config, stored["tag_configs"], anomaly_start, anomaly_end
                ),
                "reference": _raw_comparison_range(
                    indexed, tags, config, stored["tag_configs"], reference_start, reference_end
                ),
            }
        training = build_training_matrix(
            loaded.frame, source["timestamp_column"], stored["tags"], config,
            manifest["training_windows"], engineering_ranges(stored["tag_configs"]),
            exclude_engineering_range=manifest["model_purpose"] == "normal_state",
            modeling_eligibility=snapshot["modeling_eligibility"])
        dynamic = training.dynamic
        if (tuple(dynamic.columns) != model.feature_names or len(dynamic) != model.n_samples
                or not np.allclose(dynamic.mean().to_numpy(), model.mean, rtol=1e-12, atol=1e-12)
                or not np.allclose(dynamic.std(ddof=0).to_numpy(), model.scale, rtol=1e-12, atol=1e-12)):
            raise ValueError("当前环境无法复现训练样本，请重新训练")
        scores = model.score(dynamic)
        if action == "interval":
            return {"run_id": run_id, **interval_diagnosis(model, dynamic, scores, start, end)}
        if action == "compare":
            load_reference = training.reference
            if load_tag not in load_reference.columns:
                load_training = build_training_matrix(
                    loaded.frame, source["timestamp_column"], [load_tag], config,
                    manifest["training_windows"], validate_dynamic=False,
                    modeling_eligibility=snapshot["modeling_eligibility"], validate_quality=False,
                )
                load_reference = load_training.reference
            load_values = (
                load_reference.set_index(source["timestamp_column"])[load_tag]
                if load_tag in load_reference.columns and not load_reference.empty
                else pd.Series(dtype=float)
            )
            return {
                "run_id": run_id,
                "anomaly_start": anomaly_start.isoformat(),
                "anomaly_end": anomaly_end.isoformat(),
                "reference_start": reference_start.isoformat(),
                "reference_end": reference_end.isoformat(),
                **compare_diagnosis(
                    model, dynamic, scores, load_values, load_tag,
                    anomaly_start, anomaly_end, reference_start, reference_end,
                ),
            }
        boundaries = set()
        for window in training.window_summaries:
            positions = dynamic.index[(dynamic.index >= pd.Timestamp(window["start"])) &
                                      (dynamic.index <= pd.Timestamp(window["end"]))]
            if len(positions):
                boundaries.add(positions[0])
            for segment in window.get("segments", []):
                positions = dynamic.index[(dynamic.index >= pd.Timestamp(segment["start"])) &
                                          (dynamic.index <= pd.Timestamp(segment["end"]))]
                if len(positions):
                    boundaries.add(positions[0])
        previous = None
        rows = []
        for timestamp, row in scores.iterrows():
            broken = previous is not None and (timestamp in boundaries or
                timestamp - previous != pd.Timedelta(minutes=config.sample_interval_minutes))
            rows.append({"timestamp": timestamp.isoformat(), "t2": float(row.t2) if row.score_valid else None,
                         "spe": float(row.spe) if row.score_valid else None, "break_before": broken})
            previous = timestamp
        windows = [window for window in manifest["training_windows"] if window["enabled"]]
        return {"run_id": run_id, "start": min(window["start"] for window in windows),
                "end": max(window["end"] for window in windows), "scores": rows,
                "t2_limits": web._limit_payload(model.t2_limits),
                "q_limits": web._limit_payload(model.q_limits),
                "load_tags": list(loaded.metadata.numeric_candidate_columns)}
