import json
import inspect

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.cli import build_parser
from pca_model_builder import web
from pca_model_builder.model_io import load_model_package
from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)


def _history_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01", periods=180, freq="5min")
    a = rng.normal(size=len(timestamps))
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "A": a,
            "B": 1.8 * a + rng.normal(scale=0.1, size=len(timestamps)),
            "C": rng.normal(scale=0.25, size=len(timestamps)),
            "engineering_label": ["normal"] * 130 + ["known_event"] * 50,
        }
    )
    frame.loc[145:160, "C"] += 8.0
    return frame


def test_web_uses_port_distinct_from_dataproject_and_exposes_workflow():
    args = build_parser().parse_args(["serve", "--no-open"])

    assert web.DEFAULT_PORT == 8775
    assert args.port == 8775
    assert "8765" not in web.INDEX_HTML
    assert '<link rel="icon" href="data:,">' in web.INDEX_HTML
    assert '[hidden] { display:none !important; }' in web.INDEX_HTML
    for element_id in (
        'id="fileInput"',
        'id="uploadButton"',
        'id="timestampColumn"',
        'id="tagOptions"',
        'id="tagConfigList"',
        'id="performanceButton"',
        'id="performanceConditions"',
        'id="performanceTable"',
        'id="clusterButton"',
        'id="clusterChart"',
        'id="clusterTable"',
        'id="trainButton"',
        'id="validateButton"',
        'id="t2Chart"',
        'id="speChart"',
        'id="scoreChart"',
        'id="contributionTable"',
        'id="scoresDownload"',
        'id="reportDownload"',
        'id="contributionsDownload"',
    ):
        assert element_id in web.INDEX_HTML
    assert "excludePerformanceColumns(data.conditions)" in web.INDEX_HTML
    assert 'id="varianceThreshold" type="number" min="0.01" max="0.99"' in web.INDEX_HTML


def test_web_service_trains_and_validates_uploaded_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    csv_bytes = history.to_csv(index=False).encode("utf-8-sig")

    uploaded = web.save_upload("history.csv", csv_bytes)
    inspected = web.inspect_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "encoding": "utf-8-sig",
        }
    )
    clustered = web.cluster_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "analysis_start": "2026-01-01T00:00:00",
            "analysis_end": "2026-01-01T14:55:00",
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
            "variance_threshold": 0.95,
            "n_clusters": 2,
        }
    )
    screened = web.performance_screen_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "analysis_start": "2026-01-01T00:00:00",
            "analysis_end": "2026-01-01T14:55:00",
            "sample_interval_minutes": 5,
            "conditions": [
                {"column": "A", "minimum": 0},
                {"column": "C", "maximum": 1},
            ],
        }
    )
    trained = web.train_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "tag_configs": {
                tag: {
                    "description": f"{tag}变量",
                    "unit": "unit",
                    "type": "continuous",
                    "engineering_min": -100,
                    "engineering_max": 200 if tag == "A" else 100,
                }
                for tag in ["A", "B", "C"]
            },
            "normal_start": "2026-01-01T00:00:00",
            "normal_end": "2026-01-01T09:55:00",
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
            "variance_threshold": 0.95,
            "model_name": "UNIT_DPCA_V1",
        }
    )
    validated = web.validate_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "validation_start": "2026-01-01T10:00:00",
            "validation_end": "2026-01-01T14:55:00",
            "label_column": "engineering_label",
        }
    )

    assert inspected["numeric_columns"] == ["A", "B", "C"]
    assert inspected["sample_interval_minutes"] == 5.0
    assert inspected["suggested_normal_end"] < inspected["suggested_validation_start"]
    assert clustered["engineer_decision_required"] is True
    assert clustered["sample_count"] == 177
    assert len(clustered["clusters"]) == 2
    assert {point["cluster"] for point in clustered["points"]} == {1, 2}
    assert screened["engineer_decision_required"] is True
    assert 0 < screened["matched_rows"] < screened["total_rows"]
    assert screened["representative_windows"]
    assert trained["validation_status"] == "draft"
    assert trained["n_components"] >= 2
    assert {"pc1", "pc2"}.issubset(trained["scores"][0])
    assert trained["training_rows"] > 0
    assert trained["model_download"].endswith(trained["run_id"])
    assert (tmp_path / "runs" / trained["run_id"] / "model.pcamodel").exists()
    loaded_model, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )
    assert manifest["config"]["tag_configs"]["A"]["description"] == "A变量"
    normal = history.iloc[:120].set_index("time")[["A", "B", "C"]]
    preprocessing = PreprocessingConfig(5, 10, 10, 5)
    dynamic = build_dynamic_matrix(
        normal,
        ["A", "B", "C"],
        preprocessing,
        infer_segment_ids(normal.index, 5),
    )
    assert loaded_model.mean[0] == pytest.approx(dynamic.iloc[:, 0].mean())
    assert loaded_model.mean[0] != pytest.approx(50.0)
    assert validated["engineer_decision_required"] is True
    assert "known_event" in validated["status_by_engineering_label"]
    assert validated["status_counts"].get("abnormal", 0) > 0
    assert validated["contributions"]
    assert all(
        item["statistic_value"] >= item["limit_95"]
        for item in validated["contributions"]
    )
    assert all(
        {"description", "unit"}.issubset(tag)
        for group in validated["contributions"]
        for tag in group["tags"]
    )
    assert {
        "pc1",
        "pc2",
        "t2_limit_ratio",
        "spe_limit_ratio",
        "t2_status",
        "spe_status",
    }.issubset(validated["scores"][0])
    run_dir = tmp_path / "runs" / trained["run_id"]
    saved_scores = pd.read_csv(run_dir / "validation_scores.csv", encoding="utf-8-sig")
    saved_report = json.loads(
        (run_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    saved_contributions = json.loads(
        (run_dir / "validation_contributions.json").read_text(encoding="utf-8")
    )
    assert len(saved_scores) == validated["scored_rows"]
    assert {"pc1", "pc2", "t2", "spe", "t2_status", "spe_status"}.issubset(
        saved_scores.columns
    )
    assert saved_report["engineer_decision_required"] is True
    assert "scores" not in saved_report
    assert saved_contributions == validated["contributions"]
    assert set(validated["validation_downloads"]) == {
        "scores",
        "report",
        "contributions",
    }


def test_validation_download_artifact_uses_fixed_whitelist():
    assert web._validation_artifact("scores") == (
        "validation_scores.csv",
        "text/csv; charset=utf-8",
    )
    with pytest.raises(ValueError, match="无效的验证工件类型"):
        web._validation_artifact("../model")


def test_windows_launcher_uses_web_port_8775():
    launcher = (web.PROJECT_ROOT / "start_app.bat").read_text(encoding="utf-8")

    assert "127.0.0.1:8775" in launcher
    assert "--port 8775" in launcher
    assert 'pushd "%~dp0src"' in launcher
    assert "8765" not in launcher


def test_upload_detects_gb18030_header(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    content = "时间,温度,压力\n2026-01-01 00:00,10,20\n".encode("gb18030")

    uploaded = web.save_upload("中文数据.csv", content)

    assert uploaded["encoding"] == "gb18030"
    assert uploaded["columns"] == ["时间", "温度", "压力"]


def test_web_training_blocks_values_outside_configured_engineering_range(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    uploaded = web.save_upload(
        "history.csv", _history_frame().to_csv(index=False).encode("utf-8-sig")
    )

    with pytest.raises(ValueError, match=r"engineering_range\(\d+\)"):
        web.train_payload(
            {
                "file_id": uploaded["file_id"],
                "timestamp_column": "time",
                "tags": ["A", "B"],
                "tag_configs": {
                    "A": {"engineering_min": -0.1, "engineering_max": 0.1},
                    "B": {},
                },
                "normal_start": "2026-01-01T00:00:00",
                "normal_end": "2026-01-01T09:55:00",
                "sample_interval_minutes": 5,
                "smoothing_window_minutes": 10,
                "max_lag_minutes": 10,
                "lag_step_minutes": 5,
                "model_name": "UNIT_DPCA_V1",
            }
        )


def test_web_training_and_clustering_allow_physical_time_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    history.loc[90:, "time"] += pd.Timedelta(minutes=15)
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    common = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "analysis_start": history.time.iloc[0].isoformat(),
        "analysis_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "variance_threshold": 0.95,
    }

    clustered = web.cluster_payload({**common, "n_clusters": 2})
    trained = web.train_payload(
        {
            **common,
            "normal_start": common["analysis_start"],
            "normal_end": common["analysis_end"],
            "model_name": "GAP_DPCA_V1",
        }
    )

    assert clustered["sample_count"] == 174
    assert trained["training_rows"] == 174
    assert trained["n_components"] >= 2


def test_web_training_rejects_variance_threshold_of_one(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )

    with pytest.raises(ValueError, match="residual space for SPE"):
        web.train_payload(
            {
                "file_id": uploaded["file_id"],
                "timestamp_column": "time",
                "tags": ["A", "B", "C"],
                "normal_start": history.time.iloc[0].isoformat(),
                "normal_end": history.time.iloc[-1].isoformat(),
                "sample_interval_minutes": 5,
                "smoothing_window_minutes": 10,
                "max_lag_minutes": 10,
                "lag_step_minutes": 5,
                "variance_threshold": 1.0,
                "model_name": "INVALID_DPCA_V1",
            }
        )


@pytest.mark.parametrize("statistic", ["t2", "spe"])
def test_score_payload_preserves_short_anomaly_missed_by_uniform_sampling(statistic):
    count = web.MAX_CHART_POINTS + 805
    scores = _chart_scores(count)
    old_positions = set(
        np.linspace(0, count - 1, web.MAX_CHART_POINTS, dtype=int).tolist()
    )
    anomaly = next(position for position in range(2, count - 2) if position not in old_positions)
    scores.iloc[anomaly, scores.columns.get_loc(statistic)] = 50.0
    scores.iloc[anomaly, scores.columns.get_loc(f"{statistic}_limit_ratio")] = 10.0
    scores.iloc[anomaly, scores.columns.get_loc(f"{statistic}_status")] = "abnormal"
    scores.iloc[anomaly, scores.columns.get_loc("status")] = "abnormal"

    payload = web._score_payload(scores)
    timestamps = {row["timestamp"] for row in payload}

    assert len(payload) <= web.MAX_CHART_POINTS
    assert scores.index[0].isoformat() in timestamps
    assert scores.index[-1].isoformat() in timestamps
    assert scores.index[anomaly].isoformat() in timestamps
    assert scores.index[anomaly - 1].isoformat() in timestamps
    assert scores.index[anomaly + 1].isoformat() in timestamps
    assert "np.linspace" not in inspect.getsource(web._score_payload)


def test_score_payload_buckets_when_critical_points_exceed_limit():
    count = web.MAX_CHART_POINTS * 2 + 5
    scores = _chart_scores(count)
    scores.loc[:, ["status", "t2_status", "spe_status"]] = "attention"
    scores.loc[:, ["t2_limit_ratio", "spe_limit_ratio"]] = 1.1
    abnormal_positions = np.arange(3, count, 7)
    scores.iloc[abnormal_positions, scores.columns.get_loc("status")] = "abnormal"
    scores.iloc[abnormal_positions, scores.columns.get_loc("t2_status")] = "abnormal"
    scores.iloc[731, scores.columns.get_loc("t2")] = 100.0
    scores.iloc[731, scores.columns.get_loc("t2_limit_ratio")] = 20.0
    scores.iloc[1873, scores.columns.get_loc("spe")] = 120.0
    scores.iloc[1873, scores.columns.get_loc("spe_limit_ratio")] = 24.0

    payload = web._score_payload(scores)
    timestamps = [row["timestamp"] for row in payload]

    assert len(payload) == web.MAX_CHART_POINTS
    assert timestamps == sorted(set(timestamps))
    assert scores.index[0].isoformat() == timestamps[0]
    assert scores.index[-1].isoformat() == timestamps[-1]
    assert scores.index[731].isoformat() in timestamps
    assert scores.index[1873].isoformat() in timestamps
    assert sum(row["status"] == "abnormal" for row in payload) > 100


def _chart_scores(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pc1": np.zeros(count),
            "pc2": np.zeros(count),
            "t2": np.full(count, 0.1),
            "spe": np.full(count, 0.1),
            "t2_limit_ratio": np.full(count, 0.1),
            "spe_limit_ratio": np.full(count, 0.1),
            "t2_status": np.full(count, "normal", dtype=object),
            "spe_status": np.full(count, "normal", dtype=object),
            "status": np.full(count, "normal", dtype=object),
        },
        index=pd.date_range("2026-01-01", periods=count, freq="5min"),
    )
