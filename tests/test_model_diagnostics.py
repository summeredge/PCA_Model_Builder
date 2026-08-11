from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pca_model_builder.dpca import DPCAModel, fit_dpca
from pca_model_builder.model_diagnostics import (
    compare_candidate_runs,
    model_structure_diagnostic,
)
from pca_model_builder.model_io import save_model_package


def _windows(
    start: str = "2026-01-01T00:00:00",
    *,
    window_id: str = "manual-001",
    source: str = "manual",
    source_ref: str | None = None,
    enabled: bool = True,
    comment: str = "",
) -> list[dict[str, object]]:
    return [
        {
            "id": window_id,
            "start": start,
            "end": "2026-01-02T12:00:00" if start.startswith("2026-01-02") else "2026-01-01T12:00:00",
            "source": source,
            "source_ref": source_ref,
            "enabled": enabled,
            "comment": comment,
        }
    ]


def _config(
    tags: list[str] | None = None,
    max_lag_minutes: int = 5,
    **changes: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "model_name": "diagnostic-model",
        "tags": tags or ["A", "B", "C"],
        "timestamp_column": "time",
        "sample_interval_minutes": 5,
        "resampling_method": "none",
        "filter_method": "trailing_mean",
        "smoothing_window_minutes": 5,
        "max_lag_minutes": max_lag_minutes,
        "lag_step_minutes": 5,
        "variance_threshold": 0.95,
        "state_filters": [],
    }
    value.update(changes)
    return value


def _model(
    tags: list[str],
    max_lag_minutes: int,
    lag_step_minutes: int = 5,
    n_components: int = 2,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    features = [
        f"{tag}__lag_{lag:03d}min"
        for lag in range(0, max_lag_minutes + 1, lag_step_minutes)
        for tag in tags
    ]
    data = rng.normal(size=(120, len(features)))
    data[:, 1] += data[:, 0] * 0.5
    import pandas as pd

    return fit_dpca(pd.DataFrame(data, columns=features), n_components=n_components)


def _save_candidate(
    runs: Path,
    run_id: str,
    *,
    tags: list[str] | None = None,
    max_lag_minutes: int = 5,
    windows: list[dict[str, object]] | None = None,
    model_purpose: str = "normal_state",
    model_status: str = "candidate",
    n_components: int = 2,
    **config_changes: object,
) -> Path:
    used_tags = tags or ["A", "B", "C"]
    lag_step_minutes = int(config_changes.get("lag_step_minutes", 5))
    path = runs / run_id / "model.pcamodel"
    save_model_package(
        path,
        _model(used_tags, max_lag_minutes, lag_step_minutes, n_components),
        _config(used_tags, max_lag_minutes, **config_changes),
        _windows() if windows is None else windows,
        model_purpose=model_purpose,
        model_status=model_status,
    )
    return path


def _energy_total(values: list[dict[str, object]]) -> float:
    return sum(float(value["energy"]) for value in values)


def test_structure_diagnostic_normalizes_tag_and_lag_loading_energy() -> None:
    model = _model(["A", "B", "C"], 5)
    manifest = {
        "model_purpose": "normal_state",
        "model_status": "candidate",
        "config": _config(),
        "training_windows": _windows(),
    }

    diagnostic = model_structure_diagnostic(model, manifest, "a" * 32)

    assert diagnostic["run_id"] == "a" * 32
    for component in ("pc1", "pc2", "retained_components"):
        assert _energy_total(diagnostic["tag_loading_energy"][component]) == pytest.approx(1)
        assert _energy_total(diagnostic["lag_loading_energy"][component]) == pytest.approx(1)
    assert diagnostic["lag_loading_energy"]["zero_lag_energy"] + diagnostic[
        "lag_loading_energy"
    ]["nonzero_lag_energy"] == pytest.approx(1)


def test_structure_diagnostic_handles_one_component_and_static_pca() -> None:
    model = DPCAModel(
        feature_names=("A__lag_000min", "B__lag_000min", "C__lag_000min"),
        mean=np.zeros(3),
        scale=np.ones(3),
        components=np.array([[1.0, 0.0, 0.0]]),
        eigenvalues=np.array([2.0, 1.0, 0.5]),
        explained_variance_ratio=np.array([0.5, 0.25, 0.125]),
        t2_limits={0.95: 2.0, 0.99: 3.0},
        q_limits={0.95: 1.0, 0.99: 2.0},
        n_samples=20,
    )
    manifest = {
        "model_purpose": "normal_state",
        "model_status": "candidate",
        "config": _config(max_lag_minutes=0),
        "training_windows": _windows(),
    }

    diagnostic = model_structure_diagnostic(model, manifest)

    assert diagnostic["tag_loading_energy"]["pc2"] is None
    assert diagnostic["lag_loading_energy"]["pc2"] is None
    assert diagnostic["lag_loading_energy"]["nonzero_lag_energy"] == 0
    assert diagnostic["components_for_cumulative_explained_variance"]["95"] is None


def test_candidate_comparison_reports_parameter_differences_and_source_limit(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    baseline = _save_candidate(runs, "a" * 32)
    _save_candidate(
        runs,
        "b" * 32,
        smoothing_window_minutes=10,
        max_lag_minutes=10,
        variance_threshold=0.9,
        n_components=3,
    )
    before = baseline.read_bytes(), baseline.stat().st_mtime_ns

    comparison = compare_candidate_runs(["a" * 32, "b" * 32], runs)

    assert comparison["comparability"] == {
        "comparable": True,
        "status": "structural_comparison_only",
        "reasons": ["训练源身份未固化，仅作结构比较"],
    }
    differences = comparison["parameter_differences"][0]["differences"]
    assert {
        "smoothing_window_minutes",
        "max_lag_minutes",
        "variance_threshold",
        "retained_component_count",
    } <= set(differences)
    assert baseline.read_bytes() == before[0]
    assert baseline.stat().st_mtime_ns == before[1]


def test_candidate_comparison_reports_first_order_alpha_and_gap_threshold(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _save_candidate(
        runs, "a" * 32, filter_method="first_order", first_order_alpha=0.2,
        gap_threshold_minutes=5,
    )
    _save_candidate(
        runs, "b" * 32, filter_method="first_order", first_order_alpha=0.6,
        gap_threshold_minutes=10,
    )

    comparison = compare_candidate_runs(["a" * 32, "b" * 32], runs)

    differences = comparison["parameter_differences"][0]["differences"]
    assert differences["first_order_alpha"] == {"baseline": 0.2, "value": 0.6}
    assert differences["gap_threshold_minutes"] == {"baseline": 5, "value": 10}
    preprocessing = comparison["diagnostics"][0]["preprocessing"]
    assert preprocessing["first_order_alpha"] == 0.2
    assert preprocessing["gap_threshold_minutes"] == 5


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"tags": ["B", "A", "C"]}, "原始Tag及顺序不一致"),
        ({"windows": _windows("2026-01-02T00:00:00")}, "标准化训练窗口范围或启用状态不一致"),
        ({"windows": _windows(enabled=False)}, "标准化训练窗口范围或启用状态不一致"),
        (
            {
                "sample_interval_minutes": 10,
                "smoothing_window_minutes": 10,
                "max_lag_minutes": 10,
                "lag_step_minutes": 10,
            },
            "采样间隔不一致",
        ),
        ({"resampling_method": "last"}, "重采样方法不一致"),
    ],
)
def test_candidate_comparison_rejects_required_comparability_mismatch(
    tmp_path: Path, changes: dict[str, object], reason: str
) -> None:
    runs = tmp_path / "runs"
    _save_candidate(runs, "a" * 32)
    _save_candidate(runs, "b" * 32, **changes)

    comparison = compare_candidate_runs(["a" * 32, "b" * 32], runs)

    assert comparison["comparability"]["comparable"] is False
    assert reason in comparison["comparability"]["reasons"]


def test_candidate_comparison_rejects_invalid_duplicate_missing_and_corrupt_packages(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _save_candidate(runs, "a" * 32)
    _save_candidate(runs, "c" * 32, model_purpose="exploratory", model_status="draft")
    (runs / ("b" * 32)).mkdir(parents=True)
    (runs / ("b" * 32) / "model.pcamodel").write_bytes(b"broken")

    with pytest.raises(ValueError, match="run_ids必须包含2至4"):
        compare_candidate_runs(["a" * 32], runs)
    with pytest.raises(ValueError, match="run_ids不能重复"):
        compare_candidate_runs(["a" * 32, "a" * 32], runs)
    with pytest.raises(ValueError, match="run_id无效"):
        compare_candidate_runs(["a" * 32, "bad"], runs)
    with pytest.raises(ValueError, match="运行记录不存在"):
        compare_candidate_runs(["a" * 32, "d" * 32], runs)
    with pytest.raises(ValueError, match="模型包损坏"):
        compare_candidate_runs(["a" * 32, "b" * 32], runs)
    with pytest.raises(ValueError, match="仅允许比较normal_state/candidate"):
        compare_candidate_runs(["a" * 32, "c" * 32], runs)


def test_candidate_comparison_ignores_unvalidated_source_identity(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _save_candidate(runs, "a" * 32, source_identity={"dataset": "history-a"})
    _save_candidate(runs, "b" * 32, source_identity={"dataset": "history-a"})

    comparison = compare_candidate_runs(["a" * 32, "b" * 32], runs)

    assert comparison["comparability"] == {
        "comparable": True,
        "status": "structural_comparison_only",
        "reasons": ["训练源身份未固化，仅作结构比较"],
    }

    _save_candidate(runs, "c" * 32, source_identity={"dataset": "history-b"})
    different_extension = compare_candidate_runs(["a" * 32, "c" * 32], runs)

    assert different_extension["comparability"] == comparison["comparability"]


def test_candidate_comparison_ignores_window_tracking_metadata(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _save_candidate(runs, "a" * 32)
    _save_candidate(
        runs,
        "b" * 32,
        windows=_windows(
            window_id="cluster-window-008",
            source="cluster",
            source_ref="cluster_003-candidate-002",
            comment="同一时间范围，不同追踪信息",
        ),
    )

    comparison = compare_candidate_runs(["a" * 32, "b" * 32], runs)

    assert comparison["comparability"] == {
        "comparable": True,
        "status": "structural_comparison_only",
        "reasons": ["训练源身份未固化，仅作结构比较"],
    }
    differences = comparison["parameter_differences"][0]["differences"]
    assert differences["training_windows"]["baseline"] != differences[
        "training_windows"
    ]["value"]
