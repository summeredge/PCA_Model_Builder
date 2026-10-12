"""Temporary single-Tag experiments; no model-package or lifecycle writes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .dpca import DPCAModel, fit_dpca
from .preprocessing import PreprocessingConfig
from .tag_config import engineering_ranges
from .training import TrainingBuildResult, build_training_matrix
from .validation import (
    _combined_exceedance_events,
    _validation_metrics,
    ensure_disjoint_windows,
    normalize_validation_windows,
    validate_model_windows,
)
from .variable_diagnostics import analyze_variable_diagnostics


MAX_DROP_TAGS = 8
# Conservative evidence gates, not statistical significance or engineering approval.
EVIDENCE_POLICY = {"minimum_normal_rows": 100, "minimum_abnormal_rows": 30,
                   "minimum_abnormal_windows": 3, "allowed_degradation": 0}


def redundancy_candidates(reference: pd.DataFrame) -> dict[str, Any]:
    diagnostics = analyze_variable_diagnostics(reference)
    diagnostics["sample_scope"] = "baseline_effective_training_timestamps"
    tags = list(reference.columns)
    adjacency = {tag: set() for tag in tags}
    for pair in diagnostics["high_correlation_pairs"]:
        adjacency[pair["tag_a"]].add(pair["tag_b"])
        adjacency[pair["tag_b"]].add(pair["tag_a"])
    numeric = reference.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    correlations = numeric.corr(min_periods=diagnostics["minimum_pair_count"])
    unavailable = {(p["tag_a"], p["tag_b"]): p for p in diagnostics["unavailable_pairs"]}
    seen, groups = set(), []
    for tag in tags:
        if tag in seen or not adjacency[tag]:
            continue
        pending, members = [tag], set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(adjacency[current] - members)
        seen.update(members)
        ordered = [item for item in tags if item in members]
        pairs = []
        for first, second in combinations(ordered, 2):
            if (first, second) in unavailable:
                pairs.append({**unavailable[first, second], "high_correlation": False})
            else:
                r = float(np.clip(correlations.loc[first, second], -1, 1))
                pairs.append({"tag_a": first, "tag_b": second, "pearson_r": r,
                              "valid_pair_count": int(numeric[[first, second]].notna().all(axis=1).sum()),
                              "high_correlation": abs(r) >= diagnostics["correlation_threshold"]})
        groups.append({"id": f"group-{len(groups) + 1}", "tags": ordered, "pairs": pairs})
    return {"diagnostics": diagnostics, "groups": groups, "max_drop_tags": MAX_DROP_TAGS,
            "evidence_policy": EVIDENCE_POLICY}


def evaluate_ablation(
    frame: pd.DataFrame,
    timestamp_column: str,
    tags: Sequence[str],
    config: PreprocessingConfig,
    training_windows: list[dict[str, Any]],
    tag_configs: Mapping[str, Mapping[str, Any]],
    modeling_eligibility: object,
    baseline: DPCAModel,
    baseline_training: TrainingBuildResult,
    drop_tags: Sequence[str],
    variance_threshold: float,
    n_components: int | None,
    validation_windows: object,
    label_column: str = "",
) -> dict[str, Any]:
    """Fit each candidate independently and compare only shared valid timestamps."""
    windows = normalize_validation_windows(validation_windows) if validation_windows else []
    enabled = [window for window in windows if window["enabled"]]
    train_bounds = [(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]))
                    for w in training_windows if w["enabled"]]
    ensure_disjoint_windows(train_bounds, [(pd.Timestamp(w["start"]), pd.Timestamp(w["end"])) for w in enabled])
    indexed = frame.set_index(timestamp_column).sort_index()
    schemes, computations = [], []
    for position, dropped in enumerate([None, *drop_tags]):
        remaining = [tag for tag in tags if tag != dropped]
        scheme = {"name": "基准模型" if dropped is None else f"方案 {position}",
                  "dropped_tag": dropped, "tags": remaining, "tag_count": len(remaining),
                  "validation_errors": [], "validation": {}, "conclusion": "证据不足", "reasons": []}
        schemes.append(scheme)
        try:
            training = baseline_training if dropped is None else build_training_matrix(
                frame, timestamp_column, remaining, config, training_windows,
                engineering_ranges({tag: tag_configs[tag] for tag in remaining}),
                exclude_engineering_range=True, modeling_eligibility=modeling_eligibility)
            model = baseline if dropped is None else fit_dpca(
                training.dynamic, variance_threshold=variance_threshold, n_components=n_components)
        except ValueError as error:
            scheme.update(status="failed", error=str(error), reasons=[f"不可评估：{error}"])
            continue
        scheme.update(status="evaluated", dynamic_features=len(model.feature_names),
                      training_rows=len(training.dynamic), n_components=model.n_components,
                      cumulative_explained_variance=float(model.explained_variance_ratio[:model.n_components].sum()),
                      training_timestamps_match=training.dynamic.index.equals(baseline_training.dynamic.index),
                      training_common_rows=len(training.dynamic.index.intersection(baseline_training.dynamic.index)))
        scores = []
        for window in enabled:
            try:
                result = validate_model_windows(model, indexed, remaining, config, train_bounds,
                                                [window], tag_configs)
                scores.append(result["scores"].loc[result["scores"]["score_valid"]])
            except ValueError as error:
                scheme["validation_errors"].append({"window_id": window["id"], "type": window["type"], "error": str(error)})
        scored = pd.concat(scores).sort_index() if scores else pd.DataFrame()
        computations.append((scheme, model, scored))

    for kind in ("normal_validation", "known_abnormal"):
        typed = [(scheme, model, scores.loc[scores["validation_type"].eq(kind)] if not scores.empty else scores)
                 for scheme, model, scores in computations]
        common = None
        for _, _, scores in typed:
            common = scores.index if common is None else common.intersection(scores.index)
        selected_windows = [w for w in enabled if w["type"] == kind]
        expected = sum(int((pd.Timestamp(w["end"]) - pd.Timestamp(w["start"])) /
                           pd.Timedelta(minutes=config.sample_interval_minutes)) + 1 for w in selected_windows)
        for scheme, model, scores in typed:
            aligned = scores.loc[common] if not scores.empty else scores
            scored_windows = []
            for window in selected_windows:
                part = aligned.loc[aligned["validation_window_id"].eq(window["id"])] if not aligned.empty else aligned
                if not part.empty:
                    scored_windows.append({"id": window["id"], "type": kind, "start": pd.Timestamp(window["start"]),
                                           "scores": part, "continuous_events": _combined_exceedance_events(
                                               part, model, config.sample_interval_minutes)})
            metrics = _validation_metrics(scored_windows, model, config.sample_interval_minutes)[kind]
            scheme["validation"][kind] = {"original_rows": len(scores), "common_rows": len(aligned),
                                          "expected_rows": expected, "coverage": len(scores) / expected if expected else None,
                                          "common_coverage": len(aligned) / len(scores) if len(scores) else None,
                                          "metrics": metrics}
            if label_column and not aligned.empty:
                labels = indexed[label_column].reindex(aligned.index)
                scheme["validation"][kind]["status_by_engineering_label"] = {
                    str(label): {str(status): int(count) for status, count in part["status"].value_counts().items()}
                    for label, part in aligned.groupby(labels, dropna=False)
                }
    for scheme in schemes:
        if scheme["status"] == "evaluated":
            scheme["conclusion"], scheme["reasons"], scheme["detection_delta"] = ablation_conclusion(schemes[0], scheme)
    return {"schemes": schemes, "warnings": [] if enabled else ["未配置独立验证窗口，仅可比较模型结构；异常检出能力尚未验证。"],
            "evidence_policy": EVIDENCE_POLICY}


def ablation_conclusion(baseline: dict, scheme: dict) -> tuple[str, list[str], dict]:
    if scheme["dropped_tag"] is None:
        return "证据不足", ["基准模型，用于对照。"], {}
    reasons = []
    if not scheme["training_timestamps_match"]:
        reasons.append("训练样本时间戳不同，变化不能全部归因于变量剔除。")
    if baseline["validation_errors"] or scheme["validation_errors"]:
        reasons.append("存在验证窗口评分失败，验证证据不完整。")
    normal, abnormal = (scheme["validation"][kind] for kind in ("normal_validation", "known_abnormal"))
    base_normal, base_abnormal = (baseline["validation"][kind] for kind in ("normal_validation", "known_abnormal"))
    if normal["common_rows"] < EVIDENCE_POLICY["minimum_normal_rows"]:
        reasons.append("共同正常验证样本不足100个。")
    if abnormal["common_rows"] < EVIDENCE_POLICY["minimum_abnormal_rows"] or abnormal["metrics"]["valid_window_count"] < EVIDENCE_POLICY["minimum_abnormal_windows"]:
        reasons.append("已知异常验证证据不足（至少30个共同样本、3个窗口）；异常检出能力尚未充分验证。")
    if any(item["coverage"] != 1 or item["common_coverage"] != 1
           for item in (normal, abnormal, base_normal, base_abnormal)):
        reasons.append("验证覆盖不完整或共同交集损失样本。")
    delta = {}
    for key in ("detection_rate_95", "detection_rate_99", "detection_coverage_rate_95"):
        first, second = base_abnormal["metrics"].get(key), abnormal["metrics"].get(key)
        delta[key] = second - first if first is not None and second is not None else None
    # T²/SPE 独立通道窗口数：综合检出率不变时，单通道丢失窗口仍是监测能力损失。
    for statistic in ("t2", "spe"):
        for confidence in (95, 99):
            key = f"{statistic}_detected_window_count_{confidence}"
            first, second = base_abnormal["metrics"].get(key), abnormal["metrics"].get(key)
            delta[key] = second - first if first is not None and second is not None else None
    # Incomparable or small-sample evidence cannot support either deletion or retention.
    if reasons:
        return "证据不足", reasons, delta
    worse = []
    for statistic in ("t2", "spe", "overall"):
        for confidence in (95, 99):
            key = f"exceedance_rate_{confidence}"
            if normal["metrics"][statistic][key] > base_normal["metrics"][statistic][key]:
                worse.append(f"正常验证 {statistic.upper()} {confidence}% 超限率上升。")
    for key in ("continuous_false_alarm_event_count_95", "longest_continuous_false_alarm_minutes"):
        if normal["metrics"][key] > base_normal["metrics"][key]:
            worse.append("正常验证连续误报事件恶化。")
    if any(delta[key] is not None and delta[key] < 0
           for key in ("detection_rate_95", "detection_rate_99", "detection_coverage_rate_95")):
        worse.append("已知异常检出率或检出覆盖率下降。")
    for statistic in ("t2", "spe"):
        for confidence in (95, 99):
            key = f"{statistic}_detected_window_count_{confidence}"
            if delta[key] is not None and delta[key] < 0:
                worse.append(f"{statistic.upper()} {confidence}% 独立检出窗口数下降"
                             f"（{base_abnormal['metrics'][key]} → {abnormal['metrics'][key]}），监测能力损失。")
    baseline_windows = {item["validation_window_id"]: item for item in base_abnormal["metrics"]["windows"]}
    for window in abnormal["metrics"]["windows"]:
        previous = baseline_windows[window["validation_window_id"]]
        for confidence in (95, 99):
            key = f"first_detection_delay_minutes_{confidence}"
            if previous[key] is not None and (window[key] is None or window[key] > previous[key]):
                worse.append(f"异常窗口 {window['validation_window_id']} {confidence}% 检出丢失或延迟增加。")
        if window["detection_coverage_rate_95"] < previous["detection_coverage_rate_95"]:
            worse.append(f"异常窗口 {window['validation_window_id']} 检出覆盖率下降。")
    if worse:
        return "建议保留", list(dict.fromkeys(worse)), delta
    if base_abnormal["metrics"]["detection_rate_95"] != 1 or base_abnormal["metrics"]["detection_rate_99"] != 1:
        return "证据不足", ["基准模型未检出全部已知异常窗口，不能证明剔除无监测损失。"], delta
    return "可考虑剔除", ["在当前独立验证窗口上未观察到误报恶化或异常检出损失；仍需人工判断并重新正式训练。"], delta
