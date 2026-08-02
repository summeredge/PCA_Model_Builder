from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from pca_model_builder import web, web_model_results
from pca_model_builder.cli import main
from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import load_model_package
from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)


def test_final_web_keeps_chinese_workflow_labels_and_separate_statistics() -> None:
    html = web_model_results.INDEX_HTML

    for text in (
        "历史数据",
        "建模 Tag",
        "参考状态与 DPCA 参数",
        "训练 DPCA 草稿模型",
        "运行状态聚类辅助",
        "验证结果",
        "训练期 T²",
        "训练期 SPE/Q",
        "验证期 T²",
        "验证期 SPE/Q",
        "下载模型包",
        "下载完整评分 CSV",
    ):
        assert text in html
    assert 'id="t2Chart"' in html
    assert 'id="speChart"' in html
    assert 'id="validationT2Chart"' in html
    assert 'id="validationSpeChart"' in html


def test_schema_v1_model_package_loads_and_preserves_scores(tmp_path: Path) -> None:
    frame = _dynamic_frame()
    model = fit_dpca(frame, n_components=2)
    path = tmp_path / "legacy-schema-v1.pcamodel"
    _write_schema_v1_package(path, model)

    loaded, manifest = load_model_package(path)

    assert manifest["schema_version"] == 1
    assert manifest["validation_status"] == "draft"
    pd.testing.assert_frame_equal(model.score(frame), loaded.score(frame))


def test_cli_and_web_single_window_training_produce_the_same_model(
    tmp_path: Path, monkeypatch
) -> None:
    history = _history_frame()
    csv_path = tmp_path / "history.csv"
    cli_path = tmp_path / "cli.pcamodel"
    history.to_csv(csv_path, index=False, encoding="utf-8-sig")

    assert main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            str(history.time.iloc[0]),
            "--normal-end",
            str(history.time.iloc[119]),
            "--sample-interval",
            "5",
            "--smoothing-window",
            "10",
            "--max-lag",
            "10",
            "--lag-step",
            "5",
            "--components",
            "2",
            "--model-name",
            "BASELINE_DPCA",
            "--output",
            str(cli_path),
        ]
    ) == 0

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    web_result = web.train_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "normal_start": history.time.iloc[0].isoformat(),
            "normal_end": history.time.iloc[119].isoformat(),
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
            "n_components": 2,
            "model_name": "BASELINE_DPCA",
        }
    )
    cli_model, _ = load_model_package(cli_path)
    web_model, _ = load_model_package(
        tmp_path / "runs" / web_result["run_id"] / "model.pcamodel"
    )

    assert cli_model.feature_names == web_model.feature_names
    np.testing.assert_allclose(cli_model.mean, web_model.mean)
    np.testing.assert_allclose(cli_model.scale, web_model.scale)
    np.testing.assert_allclose(cli_model.components, web_model.components)
    np.testing.assert_allclose(cli_model.eigenvalues, web_model.eigenvalues)
    assert cli_model.t2_limits == web_model.t2_limits
    assert cli_model.q_limits == web_model.q_limits


def _history_frame() -> pd.DataFrame:
    rng = np.random.default_rng(906)
    a = rng.normal(size=160)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=160, freq="5min"),
            "A": a,
            "B": 1.5 * a + rng.normal(scale=0.1, size=len(a)),
            "C": rng.normal(scale=0.2, size=len(a)),
        }
    )


def _dynamic_frame() -> pd.DataFrame:
    history = _history_frame().iloc[:120]
    indexed = history.set_index("time")[["A", "B", "C"]]
    config = PreprocessingConfig(5, 10, 0, 5)
    return build_dynamic_matrix(
        indexed,
        ["A", "B", "C"],
        config,
        infer_segment_ids(indexed.index, config.sample_interval_minutes),
    )


def _write_schema_v1_package(path: Path, model) -> None:
    arrays = BytesIO()
    np.savez_compressed(
        arrays,
        mean=model.mean,
        scale=model.scale,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
    )
    manifest = {
        "schema_version": 1,
        "validation_status": "draft",
        "feature_names": list(model.feature_names),
        "n_samples": model.n_samples,
        "n_components": model.n_components,
        "t2_limits": {str(key): value for key, value in model.t2_limits.items()},
        "q_limits": {str(key): value for key, value in model.q_limits.items()},
        "config": {
            "model_name": "BASELINE_SCHEMA_V1",
            "tags": ["A", "B", "C"],
            "timestamp_column": "time",
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
            "variance_threshold": 0.95,
        },
        "training_windows": [["2026-01-01", "2026-01-02"]],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays.getvalue())
