from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .tag_profile import profile_tag


HIGH_CORRELATION_THRESHOLD = 0.95
MINIMUM_PAIR_COUNT = 3


def analyze_variable_diagnostics(data: pd.DataFrame) -> dict[str, object]:
    """Describe aligned, unfiltered engineering values at valid DPCA timestamps.

    The caller supplies only original modelling Tags, never Lag columns or
    performance Tags. This function does not preprocess or change selection.
    """
    numeric = data.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    profiles = []
    for tag in numeric:
        profile = profile_tag(data[tag])
        flags = []
        if profile["valid_count"] < MINIMUM_PAIR_COUNT:
            flags.append("有效样本不足")
        if profile["valid_count"] < profile["sample_count"]:
            flags.append("存在无效数值")
        if profile["unique_count"] == 1:
            flags.append("精确常量")
        elif profile["valid_count"]:
            # Same variability conventions as quality.py and tag_profile.py.
            if np.isfinite([profile["standard_deviation"], profile["mean"]]).all() and 0 < profile["standard_deviation"] <= max(abs(profile["mean"]), 1.0) * 1e-6:
                flags.append("近似无变化")
            if 1 < profile["unique_count"] <= min(10, max(2, int(profile["valid_count"] * 0.01))):
                flags.append("低唯一值，疑似离散状态量")
        for key, value in profile.items():
            if isinstance(value, float) and not np.isfinite(value):
                profile[key] = None
                if "统计不可计算" not in flags:
                    flags.append("统计不可计算")
        profiles.append({"tag": tag, "profile": profile, "flags": flags})

    correlations = numeric.corr(method="pearson", min_periods=MINIMUM_PAIR_COUNT)
    finite = numeric.notna().astype(np.int64)
    pair_counts = finite.T.dot(finite)
    high_pairs, unavailable_pairs = [], []
    for first, second in combinations(numeric.columns, 2):
        count = int(pair_counts.loc[first, second])
        r = correlations.loc[first, second]
        pair = {"tag_a": first, "tag_b": second, "valid_pair_count": count}
        if count < MINIMUM_PAIR_COUNT:
            reason = f"有效配对样本不足（至少 {MINIMUM_PAIR_COUNT} 个）"
        elif not np.isfinite(r):
            paired = numeric[[first, second]].dropna()
            reason = "配对样本中存在常量变量" if (paired.nunique() <= 1).any() else "相关系数不可计算"
        else:
            reason = None
        if reason:
            unavailable_pairs.append({**pair, "pearson_r": None, "unavailable_reason": reason})
        elif abs(r) >= HIGH_CORRELATION_THRESHOLD:
            high_pairs.append({**pair, "pearson_r": float(np.clip(r, -1, 1))})
    high_pairs.sort(key=lambda pair: -abs(pair["pearson_r"]))
    attention_tags = {item["tag"] for item in profiles if item["flags"]}
    for pair in high_pairs:
        attention_tags.update((pair["tag_a"], pair["tag_b"]))
    return {
        "sample_scope": "full_valid_exploration_samples",
        "value_scope": "resampled_before_filtering",
        "sample_count": len(data),
        "correlation_threshold": HIGH_CORRELATION_THRESHOLD,
        "minimum_pair_count": MINIMUM_PAIR_COUNT,
        "summary": {"tag_count": len(data.columns), "high_correlation_pair_count": len(high_pairs), "attention_tag_count": len(attention_tags)},
        "high_correlation_pairs": high_pairs,
        "unavailable_pairs": unavailable_pairs,
        "tag_profiles": profiles,
    }
