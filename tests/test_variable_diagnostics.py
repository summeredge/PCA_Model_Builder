import json

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.variable_diagnostics import analyze_variable_diagnostics


def test_correlations_profiles_and_attention_are_evidence_only():
    x = np.arange(100, dtype=float)
    data = pd.DataFrame({
        "FIC400001.SV": x,
        "FIC400002.SV": x * 2 + 10,
        "negative": -x,
        "independent": np.random.default_rng(1).normal(size=100),
        "constant": 7.0,
        "near_constant": 1000 + x * 1e-7,
        "discrete": np.tile([0, 1], 50),
    })
    before = data.copy(deep=True)
    result = analyze_variable_diagnostics(data)
    profiles = {item["tag"]: item for item in result["tag_profiles"]}
    pair = next(item for item in result["high_correlation_pairs"] if item["tag_a"] == "FIC400001.SV" and item["tag_b"] == "FIC400002.SV")
    assert pair["pearson_r"] == pytest.approx(1)
    assert pair["valid_pair_count"] == 100
    assert any(item["pearson_r"] == pytest.approx(-1) for item in result["high_correlation_pairs"])
    assert not any("independent" in (item["tag_a"], item["tag_b"]) for item in result["high_correlation_pairs"])
    assert "精确常量" in profiles["constant"]["flags"]
    assert "近似无变化" in profiles["near_constant"]["flags"]
    assert "低唯一值，疑似离散状态量" in profiles["discrete"]["flags"]
    assert profiles["independent"]["flags"] == []
    stats = profiles["FIC400001.SV"]["profile"]
    assert stats["valid_count"] == stats["unique_count"] == 100
    assert stats["mean"] == stats["median"] == 49.5
    assert stats["standard_deviation"] == pytest.approx(x.std())
    assert stats["p05"] == pytest.approx(4.95)
    assert stats["p95"] == pytest.approx(94.05)
    assert result["summary"]["tag_count"] == 7
    assert result["summary"]["high_correlation_pair_count"] == len(result["high_correlation_pairs"])
    assert result["summary"]["attention_tag_count"] == 6
    assert all("常量" in item["unavailable_reason"] for item in result["unavailable_pairs"])
    pd.testing.assert_frame_equal(data, before)
    json.dumps(result, allow_nan=False)


def test_invalid_values_pairwise_constants_and_insufficient_pairs():
    data = pd.DataFrame({
        "A": [1, 2, 3, np.inf, "bad", None],
        "B": [2, 4, 6, 8, 10, 12],
        "pair_constant": [4, 4, 4, 5, 6, 7],
        "two_values": [1, None, 2, np.inf, np.nan, -np.inf],
        "empty": pd.Series([pd.NA] * 6, dtype="Float64"),
    })
    result = analyze_variable_diagnostics(data)
    high = result["high_correlation_pairs"]
    assert high == [{"tag_a": "A", "tag_b": "B", "valid_pair_count": 3, "pearson_r": 1.0}]
    unavailable = {(item["tag_a"], item["tag_b"]): item for item in result["unavailable_pairs"]}
    assert "常量" in unavailable["A", "pair_constant"]["unavailable_reason"]
    assert "样本不足" in unavailable["A", "two_values"]["unavailable_reason"]
    assert unavailable["A", "empty"]["valid_pair_count"] == 0
    profiles = {item["tag"]: item for item in result["tag_profiles"]}
    assert profiles["A"]["profile"]["valid_count"] == 3
    assert "存在无效数值" in profiles["A"]["flags"]
    assert profiles["empty"]["profile"]["mean"] is None
    assert profiles["empty"]["profile"]["unique_count"] == 0
    assert "有效样本不足" in profiles["empty"]["flags"]
    assert all(item["pearson_r"] is None for item in unavailable.values())
    json.dumps(result, allow_nan=False)


def test_fifty_tags_return_pairs_not_a_full_matrix_and_empty_input_is_defined():
    data = pd.DataFrame({f"TAG_{i}": np.arange(60) * (i + 1) for i in range(50)})
    result = analyze_variable_diagnostics(data)
    assert len(result["tag_profiles"]) == 50
    assert len(result["high_correlation_pairs"]) == 50 * 49 // 2
    assert "correlation_matrix" not in result
    empty = analyze_variable_diagnostics(pd.DataFrame(columns=["A", "B"]))
    assert empty["sample_count"] == 0
    assert empty["unavailable_pairs"][0]["valid_pair_count"] == 0
    assert all(item["profile"]["mean"] is None for item in empty["tag_profiles"])
    json.dumps(empty, allow_nan=False)


def test_non_finite_statistics_are_not_serialized_as_infinity():
    with np.errstate(over="ignore", invalid="ignore"):
        result = analyze_variable_diagnostics(pd.DataFrame({"A": [1e308, 1e308, -1e308], "B": [1, 2, 3]}))
    assert "统计不可计算" in result["tag_profiles"][0]["flags"]
    json.dumps(result, allow_nan=False)
