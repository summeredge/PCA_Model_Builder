"""Offline modeling eligibility; deliberately independent of state_filters."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingConfig, segment_raw_data, _resegment_remaining
from .screening import _normalize_condition


def normalize_modeling_eligibility(value: object = None) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping) or set(value) - {"keep_conditions", "exclude_rule_groups"}:
        raise ValueError("建模资格 modeling_eligibility 必须是保留条件和排除规则组对象")
    keep = value.get("keep_conditions", [])
    groups = value.get("exclude_rule_groups", [])
    if not isinstance(keep, list) or not isinstance(groups, list):
        raise ValueError("建模资格条件和规则组必须是列表")
    if any(not isinstance(group, list) or not group for group in groups):
        raise ValueError("建模资格每个排除规则组必须包含至少一个条件")
    try:
        return {
            "keep_conditions": [_normalize_condition(item) for item in keep],
            "exclude_rule_groups": [[_normalize_condition(item) for item in group] for group in groups],
        }
    except ValueError as error:
        raise ValueError(f"建模资格条件无效：{error}") from error


def eligibility_columns(value: object = None) -> list[str]:
    rules = normalize_modeling_eligibility(value)
    return list(dict.fromkeys(item["column"] for item in [
        *rules["keep_conditions"],
        *(item for group in rules["exclude_rule_groups"] for item in group),
    ]))


@dataclass(frozen=True)
class EligibilityResult:
    frame: pd.DataFrame
    mask: pd.Series
    segment_ids: pd.Series
    rules: dict[str, Any]
    summary: dict[str, Any]
    source_interval: float | None
    gap_ranges: tuple[dict[str, str], ...]


def filter_modeling_eligibility(
    indexed: pd.DataFrame, value: object, config: PreprocessingConfig,
    *, allow_empty: bool = False,
) -> EligibilityResult:
    rules = normalize_modeling_eligibility(value)
    columns = eligibility_columns(rules)
    missing = [column for column in columns if column not in indexed.columns]
    if missing:
        raise ValueError(f"建模资格缺少列：{', '.join(missing)}")
    numeric = indexed.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if columns and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("建模资格条件列必须包含有限数值")
    original_ids, source_interval, gap_ranges = segment_raw_data(indexed.index, config)

    def matches(conditions: list[dict[str, Any]]) -> pd.Series:
        matched = pd.Series(True, index=indexed.index)
        for condition in conditions:
            values = numeric[condition["column"]]
            if condition["minimum"] is not None:
                matched &= values.ge(condition["minimum"])
            if condition["maximum"] is not None:
                matched &= values.le(condition["maximum"])
        return matched

    keep = matches(rules["keep_conditions"])
    excluded = pd.Series(False, index=indexed.index)
    for group in rules["exclude_rule_groups"]:
        excluded |= matches(group)
    mask = keep & ~excluded
    retained = indexed.loc[mask].copy()
    if retained.empty and not allow_empty:
        raise ValueError("建模资格筛选后没有合格样本，请调整保留条件或排除规则")
    if mask.all():
        segments = original_ids
    else:
        segments = _resegment_remaining(retained.index, original_ids, config, source_index=indexed.index)
    return EligibilityResult(retained, mask, segments, rules, {
        "original_samples": len(indexed), "keep_pass_samples": int(keep.sum()),
        "exclude_hit_samples": int(excluded.sum()), "eligible_samples": len(retained),
        "eligible_share": len(retained) / len(indexed) if len(indexed) else 0.0,
        "segment_count": int(segments.nunique()),
    }, source_interval, gap_ranges)
