import json
import hashlib
import inspect
from io import BytesIO
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

import numpy as np
from openpyxl import load_workbook
import pandas as pd
import pytest

from pca_model_builder.cli import build_parser
from pca_model_builder import cli, web, web_model_results
import pca_model_builder.model_io as model_io
from pca_model_builder.model_io import load_model_package
from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)
from pca_model_builder.training import build_training_matrix


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


def _validation_windows() -> list[dict[str, object]]:
    return [
        {
            "id": "normal-001",
            "type": "normal_validation",
            "start": "2026-01-01T08:00:00",
            "end": "2026-01-01T09:55:00",
            "enabled": True,
            "comment": "normal",
        },
        {
            "id": "abnormal-001",
            "type": "known_abnormal",
            "start": "2026-01-01T10:50:00",
            "end": "2026-01-01T14:55:00",
            "enabled": True,
            "comment": "event",
        },
    ]


def _http_get(path: str) -> tuple[int, bytes]:
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}{path}"
        with urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()
    finally:
        thread.join(timeout=5)
        server.server_close()


def _create_passed_web_run(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    trained = web.train_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "normal_start": "2026-01-01T00:00:00",
            "normal_end": "2026-01-01T07:55:00",
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
            "model_name": "candidate",
        }
    )
    windows = _validation_windows()
    web.validate_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "validation_windows": windows,
        }
    )
    web.validation_decision_payload(
        {
            "run_id": trained["run_id"],
            "decision": "passed",
            "comment": "approved",
        }
    )
    return (
        trained["run_id"],
        uploaded["file_id"],
        tmp_path / "runs" / trained["run_id"],
        windows,
    )


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
        'id="tagSearch"',
        'id="showProblemTags"',
        'id="qualityButton"',
        'id="qualityPanel"',
        'id="currentTagQuality"',
        'id="tagDescription"',
        'id="tagRole"',
        'id="templateDownload"',
        'id="tagConfigFile"',
        'id="exportConfigButton"',
        'id="trendPanel"',
        'id="trendTags"',
        'id="trendButton"',
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
    assert 'id="tagConfigList"' not in web.INDEX_HTML
    for tab_name in ("Tag配置", "趋势浏览", "状态辅助", "模型训练", "验证结果"):
        assert tab_name in web.INDEX_HTML
    assert '<button id="trainButton" disabled>' in web.INDEX_HTML
    assert "function formField(" in web.INDEX_HTML
    assert "tagConfigField(" not in web.INDEX_HTML
    assert "excludePerformanceColumns(data.conditions)" in web.INDEX_HTML
    assert 'id="varianceThreshold" type="number" min="0.01" max="0.99"' in web.INDEX_HTML


def test_final_web_page_exposes_typed_validation_and_engineer_decision_controls():
    html = web_model_results.INDEX_HTML
    for element_id in (
        'id="validationType"',
        'id="validationWindowTable"',
        'id="recordValidationDecision"',
        'id="validatedModelDownload"',
    ):
        assert element_id in html
    for label in ("正常样本验证", "已知异常验证", "通过", "结论不足", "不通过"):
        assert label in html
    assert 'validatedModelDownload.removeAttribute("href")' in html
    assert html.count("hideValidatedModelDownload()") >= 5


def test_web_tag_selection_uses_persistent_state_not_rendered_dom():
    html = web.INDEX_HTML
    render_source = html.split("function renderTagList()", 1)[1].split(
        "function selectTag", 1
    )[0]

    assert "selectedModelTags:new Set()" in html
    assert "state.selectedModelTags.has(tag)" in render_source
    assert 'document.createElement("div")' in render_source
    assert 'document.createElement("label")' not in render_source
    assert "querySelectorAll('#tagOptions input:checked')" not in render_source
    assert "checked.size?" not in render_source
    assert "state.selectedModelTags.add(tag)" in render_source
    assert "state.selectedModelTags.delete(tag)" in render_source
    assert "state.selectedModelTags.clear()" in html
    assert (
        "state.selectedModelTags=new Set(data.numeric_columns.filter("
        in html
    )
    assert "state.selectedModelTags.delete(item.tag)" in html
    assert "columns.forEach(tag=>state.selectedModelTags.delete(tag))" in html
    assert (
        "if(config.role!==\"continuous_input\") "
        "state.selectedModelTags.delete(tag)"
    ) in html


def test_web_quality_tab_shows_selected_tag_and_trend_axis_uses_payload_limits():
    html = web.INDEX_HTML

    assert "function renderCurrentTagQuality()" in html
    assert "尚未执行或结果已失效" in html
    for field in (
        "sample_count",
        "valid_count",
        "missing_count",
        "missing_rate",
        "non_numeric_count",
        "non_finite_count",
        "unique_count",
        "minimum",
        "maximum",
        "mean",
        "median",
        "standard_deviation",
        "p01",
        "p05",
        "p95",
        "p99",
        "engineering_range_outside_count",
        "normal_range_outside_count",
        "alarm_range_outside_count",
    ):
        assert field in html
    assert "data.axis_limits[tag]" in html
    assert "Math.min(...values,0)" not in html
    assert "Math.max(...values,1)" not in html
    assert "data.warnings" in html


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
    assert trained["model_purpose"] == "normal_state"
    assert trained["model_status"] == "candidate"
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


def test_web_exploratory_model_clusters_saved_dpca_scores_and_cannot_validate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    training = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": "2026-01-01T00:00:00",
        "normal_end": "2026-01-01T09:55:00",
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "model_name": "SEMANTICS_DPCA",
    }
    exploratory = web.train_payload(
        {**training, "model_purpose": "exploratory"}
    )
    candidate = web.train_payload(training)

    assert exploratory["model_purpose"] == "exploratory"
    assert exploratory["model_status"] == "draft"
    assert candidate["model_purpose"] == "normal_state"
    assert candidate["model_status"] == "candidate"
    with pytest.raises(ValueError, match="探索模型不能执行独立验证"):
        web.validate_payload({"run_id": exploratory["run_id"]})
    with pytest.raises(ValueError, match="聚类必须引用探索模型"):
        web.cluster_payload({"exploratory_run_id": candidate["run_id"]})

    clustered = web.cluster_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "exploratory_run_id": exploratory["run_id"],
            "analysis_start": "2026-01-01T00:00:00",
            "analysis_end": "2026-01-01T14:55:00",
            "n_clusters": 2,
        }
    )
    model, _ = load_model_package(
        tmp_path / "runs" / exploratory["run_id"] / "model.pcamodel"
    )
    analysis = history.iloc[:180].set_index("time")[["A", "B", "C"]]
    config = PreprocessingConfig(5, 10, 10, 5)
    dynamic = build_dynamic_matrix(analysis, ["A", "B", "C"], config, infer_segment_ids(analysis.index, 5))
    expected_scores = model.score(dynamic)
    points = pd.DataFrame(clustered["points"])
    points.index = pd.to_datetime(points.pop("timestamp"))

    assert clustered["exploratory_run_id"] == exploratory["run_id"]
    np.testing.assert_allclose(points["pc1"], expected_scores["pc1"])
    np.testing.assert_allclose(points["pc2"], expected_scores["pc2"])


def test_training_windows_api_normalizes_operations_and_reports_summary():
    window = {
        "id": "window-001",
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-01T00:10:00",
        "source": "manual",
        "source_ref": None,
        "enabled": True,
        "comment": "",
    }
    result = web.training_windows_payload(
        {
            "training_windows": [window],
            "operation": {
                "action": "update",
                "id": "window-001",
                "changes": {"comment": "工程师确认"},
            },
        }
    )

    assert result["training_windows"][0]["comment"] == "工程师确认"
    assert result["summary"][0]["duration_minutes"] == 10


def test_web_quality_uses_all_enabled_candidate_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    windows = [
        {
            "id": "manual-window-001",
            "start": history.time.iloc[0].isoformat(),
            "end": history.time.iloc[79].isoformat(),
            "source": "manual",
            "source_ref": None,
            "enabled": True,
            "comment": "稳定工况一",
        },
        {
            "id": "trend-window-001",
            "start": history.time.iloc[100].isoformat(),
            "end": history.time.iloc[-1].isoformat(),
            "source": "trend",
            "source_ref": "trend-current",
            "enabled": True,
            "comment": "稳定工况二",
        },
    ]

    result = web.quality_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "training_windows": windows,
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
        }
    )

    assert [item["id"] for item in result["training_window_summary"]] == [
        "manual-window-001",
        "trend-window-001",
    ]
    assert all(item["status"] == "used" for item in result["training_window_summary"])
    assert all(item["effective_samples"] > 0 for item in result["training_window_summary"])


def test_training_windows_api_preserves_candidate_sources_and_disabled_state():
    windows = [
        {
            "id": "manual-window-001",
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-01T00:10:00",
            "source": "manual",
            "source_ref": None,
            "enabled": False,
            "comment": "手工候选",
        },
        *[
            {
                "id": f"{source}-window-001",
                "start": f"2026-01-01T0{index}:00:00",
                "end": f"2026-01-01T0{index}:10:00",
                "source": source,
                "source_ref": f"{source}-1",
                "enabled": False,
                "comment": f"{source}候选",
            }
            for index, source in enumerate(("cluster", "trend", "performance"), start=2)
        ],
    ]

    result = web.training_windows_payload({"training_windows": windows})

    assert [item["source"] for item in result["training_windows"]] == [
        "manual",
        "cluster",
        "trend",
        "performance",
    ]
    assert [item["enabled"] for item in result["training_windows"]] == [False] * 4
    assert result["training_windows"][1]["source_ref"] == "cluster-1"


def test_web_quality_and_training_reject_all_disabled_candidate_windows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    windows = [
        {
            "id": "suggested-window-001",
            "start": history.time.iloc[0].isoformat(),
            "end": history.time.iloc[79].isoformat(),
            "source": "suggested",
            "source_ref": "inspect-default",
            "enabled": False,
            "comment": "系统建议",
        },
        {
            "id": "manual-window-001",
            "start": history.time.iloc[100].isoformat(),
            "end": history.time.iloc[-1].isoformat(),
            "source": "manual",
            "source_ref": None,
            "enabled": False,
            "comment": "工程师尚未启用",
        },
    ]
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "training_windows": windows,
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "n_components": 2,
        "model_name": "DISABLED_WINDOWS",
    }

    for entry_point in (web.quality_payload, web.train_payload):
        with pytest.raises(ValueError, match="至少需要一个启用的training_windows窗口"):
            entry_point({**payload, "training_windows": windows})
        with pytest.raises(ValueError, match="非空列表"):
            entry_point({**payload, "training_windows": []})


def test_web_candidate_management_allows_empty_collection_and_last_removal():
    window = {
        "id": "manual-window-001",
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-01T00:10:00",
        "source": "manual",
        "source_ref": "manual-input",
        "enabled": False,
        "comment": "待工程师确认",
    }
    refreshed = web.training_windows_payload({"training_windows": [], "operation": None})
    added = web.training_windows_payload(
        {
            "training_windows": [],
            "operation": {"action": "add", "window": window},
        }
    )
    removed_disabled = web.training_windows_payload(
        {
            "training_windows": added["training_windows"],
            "operation": {"action": "remove", "id": window["id"]},
        }
    )
    removed_enabled = web.training_windows_payload(
        {
            "training_windows": [{**window, "enabled": True}],
            "operation": {"action": "remove", "id": window["id"]},
        }
    )

    assert refreshed == {"training_windows": [], "summary": []}
    assert added["training_windows"][0]["source_ref"] == "manual-input"
    assert added["training_windows"][0]["comment"] == "待工程师确认"
    assert added["training_windows"][0]["enabled"] is False
    assert removed_disabled == {"training_windows": [], "summary": []}
    assert removed_enabled == {"training_windows": [], "summary": []}


def test_explicitly_enabled_candidate_can_complete_quality_and_training(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    suggested = {
        "id": "suggested-window-001",
        "start": history.time.iloc[0].isoformat(),
        "end": history.time.iloc[79].isoformat(),
        "source": "suggested",
        "source_ref": "inspect-default",
        "enabled": False,
        "comment": "系统建议",
    }
    manual = {
        "id": "manual-window-001",
        "start": history.time.iloc[100].isoformat(),
        "end": history.time.iloc[-1].isoformat(),
        "source": "manual",
        "source_ref": None,
        "enabled": False,
        "comment": "工程师确认",
    }
    added = web.training_windows_payload(
        {
            "training_windows": [suggested],
            "operation": {"action": "add", "window": manual},
        }
    )
    assert added["training_windows"][1]["enabled"] is False
    enabled = web.training_windows_payload(
        {
            "training_windows": added["training_windows"],
            "operation": {
                "action": "set_enabled",
                "id": "manual-window-001",
                "enabled": True,
            },
        }
    )
    assert enabled["training_windows"][1]["enabled"] is True

    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "training_windows": enabled["training_windows"],
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "n_components": 2,
        "model_name": "EXPLICITLY_ENABLED",
    }
    quality = web.quality_payload(payload)
    trained = web.train_payload(payload)
    _, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )

    assert quality["can_train"]
    assert [item["status"] for item in trained["training_window_summary"]] == [
        "disabled",
        "used",
    ]
    assert manifest["training_windows"] == enabled["training_windows"]
    assert [item["status"] for item in manifest["config"]["training_summary"]] == [
        "disabled",
        "used",
    ]


def test_training_window_edits_and_removals_preserve_other_enablement():
    windows = [
        {
            "id": "enabled-window-001",
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-01T00:10:00",
            "source": "manual",
            "source_ref": None,
            "enabled": True,
            "comment": "已确认",
        },
        {
            "id": "disabled-window-001",
            "start": "2026-01-01T00:20:00",
            "end": "2026-01-01T00:30:00",
            "source": "trend",
            "source_ref": "trend-current",
            "enabled": False,
            "comment": "待确认",
        },
    ]
    edited = web.training_windows_payload(
        {
            "training_windows": windows,
            "operation": {
                "action": "update",
                "id": "disabled-window-001",
                "changes": {"comment": "已编辑"},
            },
        }
    )["training_windows"]
    without_disabled = web.training_windows_payload(
        {
            "training_windows": edited,
            "operation": {"action": "remove", "id": "disabled-window-001"},
        }
    )["training_windows"]
    without_enabled = web.training_windows_payload(
        {
            "training_windows": windows,
            "operation": {"action": "set_enabled", "id": "enabled-window-001", "enabled": False},
        }
    )["training_windows"]

    assert edited[1]["comment"] == "已编辑"
    assert edited[1]["enabled"] is False
    assert without_disabled == [windows[0]]
    assert not any(window["enabled"] for window in without_enabled)


def test_web_training_uses_shared_multiwindow_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    windows = [
        {"id": "window-001", "start": history.time.iloc[0].isoformat(), "end": history.time.iloc[159].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        {"id": "window-002", "start": history.time.iloc[160].isoformat(), "end": history.time.iloc[162].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
    ]
    original = web.build_training_matrix
    calls = []

    def recorded(*args, **kwargs):
        calls.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(web, "build_training_matrix", recorded)
    trained = web.train_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "training_windows": windows,
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
            "n_components": 2,
            "model_name": "MULTIWINDOW",
        }
    )

    assert [window["id"] for window in calls[0]] == ["window-001", "window-002"]
    assert trained["training_rows"] > 0
    assert len(trained["training_window_summary"]) == 2
    assert trained["training_window_summary"][1]["status"] == "dropped"
    assert trained["training_window_summary"][1]["dropped_reason"] == "insufficient_after_smoothing_and_lag"
    assert 'id="trainingWindowSummary"' in web.INDEX_HTML
    assert 'id="trainingQualityWarnings"' in web.INDEX_HTML
    assert "renderTrainingWindowSummary(data.training_window_summary||[])" in web.INDEX_HTML
    _, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )
    assert manifest["config"]["training_summary"] == trained["training_window_summary"]


def test_web_multistate_windows_allow_local_constants_and_use_global_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(95)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    frame = pd.DataFrame({"time": time, "A": [10.0] * 60 + [20.0] * 60, "B": rng.normal(size=120), "C": rng.normal(size=120), "FIXED": [7.0] * 60 + [8.0] * 60})
    uploaded = web.save_upload("history.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    windows = [
        {"id": "window-001", "start": time[0].isoformat(), "end": time[59].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        {"id": "window-002", "start": time[60].isoformat(), "end": time[-1].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
    ]
    payload = {"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "training_windows": windows, "sample_interval_minutes": 5, "smoothing_window_minutes": 5, "max_lag_minutes": 0, "lag_step_minutes": 5, "n_components": 2, "model_name": "multistate"}

    trained = web.train_payload(payload)
    model, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )
    expected = build_training_matrix(
        frame, "time", ["A", "B", "C"], PreprocessingConfig(5, 5, 0, 5), windows
    )

    assert len(trained["training_window_summary"]) == 2
    assert manifest["config"]["training_summary"] == trained["training_window_summary"]
    np.testing.assert_allclose(model.mean, expected.dynamic.mean().to_numpy())
    np.testing.assert_allclose(model.scale, expected.dynamic.std(ddof=0).to_numpy())
    with pytest.raises(ValueError, match="并非参考期精确常量"):
        web.train_payload({**payload, "excluded_tags": [{"tag": "FIXED", "reason": "constant_in_reference_window"}]})


def test_web_multistate_windows_reject_global_constant_feature(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(96)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    frame = pd.DataFrame({"time": time, "A": [10.0] * 120, "B": rng.normal(size=120), "C": rng.normal(size=120)})
    uploaded = web.save_upload("history.csv", frame.to_csv(index=False).encode("utf-8-sig"))

    with pytest.raises(ValueError, match="常量动态特征.*A__lag_000min"):
        web.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "training_windows": [
            {"id": "window-001", "start": time[0].isoformat(), "end": time[59].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
            {"id": "window-002", "start": time[60].isoformat(), "end": time[-1].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        ], "sample_interval_minutes": 5, "smoothing_window_minutes": 5, "max_lag_minutes": 0, "lag_step_minutes": 5, "n_components": 2, "model_name": "constant"})


def test_training_entrypoints_delegate_window_quality_to_shared_builder():
    assert "_require_clean_data(" not in inspect.getsource(cli._train)
    assert "_require_clean_data(" not in inspect.getsource(web.train_payload)
    assert "include_variability=False" in inspect.getsource(build_training_matrix)


@pytest.mark.parametrize("schema_version", [1, 2])
def test_web_validates_legacy_window_packages_without_reconversion(tmp_path, monkeypatch, schema_version):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "normal_start": "2026-01-01T00:00:00", "normal_end": "2026-01-01T09:55:00", "sample_interval_minutes": 5, "smoothing_window_minutes": 10, "max_lag_minutes": 10, "lag_step_minutes": 5, "model_name": "legacy"})
    path = tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    with zipfile.ZipFile(path) as package:
        manifest, arrays = json.loads(package.read("manifest.json")), package.read("arrays.npz")
    window = manifest["training_windows"][0]
    manifest["schema_version"], manifest["training_windows"] = schema_version, [[window["start"], window["end"]]]
    if schema_version == 1:
        manifest["validation_status"] = "draft"
        manifest.pop("model_purpose"); manifest.pop("model_status")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest)); package.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="overlap"):
        web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_start": "2026-01-01T00:00:00", "validation_end": "2026-01-01T00:10:00"})
    result = web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_start": "2026-01-01T10:00:00", "validation_end": "2026-01-01T14:55:00"})
    assert result["scored_rows"] and result["status_counts"] and result["validation_downloads"]


def test_web_typed_validation_decision_keeps_candidate_and_creates_copy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "normal_start": "2026-01-01T00:00:00", "normal_end": "2026-01-01T07:55:00", "sample_interval_minutes": 5, "smoothing_window_minutes": 10, "max_lag_minutes": 0, "lag_step_minutes": 5, "model_name": "candidate"})
    windows = [
        {"id": "normal-001", "type": "normal_validation", "start": "2026-01-01T08:00:00", "end": "2026-01-01T09:55:00", "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": "2026-01-01T10:50:00", "end": "2026-01-01T14:55:00", "enabled": True, "comment": "event"},
    ]
    result = web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    assert result["normal_validation_complete"] is True
    assert result["known_abnormal_complete"] is True
    assert {item["type"] for item in result["validation_window_summaries"]} == {"normal_validation", "known_abnormal"}
    report_path = tmp_path / "runs" / trained["run_id"] / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source_candidate_package"]["identifier"] == trained["run_id"]
    assert report["source_candidate_package"]["sha256"] == hashlib.sha256((tmp_path / "runs" / trained["run_id"] / "model.pcamodel").read_bytes()).hexdigest()

    run_dir = tmp_path / "runs" / trained["run_id"]
    candidate = run_dir / "model.pcamodel"
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "insufficient", "comment": "need more data"})["validated_model_download"] is None
    assert not (run_dir / "validated_model.pcamodel").exists()
    decision = web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved"})
    assert decision["model_status"] == "validated"
    assert (run_dir / "validated_model.pcamodel").exists()
    _, candidate_manifest = load_model_package(candidate)
    validated_model, validated_manifest = load_model_package(run_dir / "validated_model.pcamodel")
    assert candidate_manifest["model_status"] == "candidate"
    assert validated_manifest["model_status"] == "validated"
    assert validated_manifest["source_candidate_package"]["identifier"] == trained["run_id"]
    candidate_bytes = candidate.read_bytes()
    assert validated_model.feature_names == tuple(candidate_manifest["feature_names"])

    web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    assert not (run_dir / "validated_model.pcamodel").exists()
    assert "engineer_decision" not in json.loads(report_path.read_text(encoding="utf-8"))
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved again"})["model_status"] == "validated"
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "insufficient", "comment": "still insufficient"})["validated_model_download"] is None
    assert not (run_dir / "validated_model.pcamodel").exists()
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved final"})["model_status"] == "validated"
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "failed", "comment": "rejected"})["validated_model_download"] is None
    assert not (run_dir / "validated_model.pcamodel").exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["engineer_decision"]["decision"] == "failed"
    assert candidate.read_bytes() == candidate_bytes


def test_web_rejects_validation_report_after_candidate_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    common = {"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "normal_start": "2026-01-01T00:00:00", "normal_end": "2026-01-01T07:55:00", "sample_interval_minutes": 5, "smoothing_window_minutes": 10, "max_lag_minutes": 0, "lag_step_minutes": 5}
    first = web.train_payload({**common, "model_name": "first"})
    windows = [
        {"id": "normal-001", "type": "normal_validation", "start": "2026-01-01T08:00:00", "end": "2026-01-01T09:55:00", "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": "2026-01-01T10:50:00", "end": "2026-01-01T14:55:00", "enabled": True, "comment": "event"},
    ]
    web.validate_payload({"run_id": first["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    second = web.train_payload({**common, "model_name": "second"})
    first_path = tmp_path / "runs" / first["run_id"] / "model.pcamodel"
    second_path = tmp_path / "runs" / second["run_id"] / "model.pcamodel"
    first_path.write_bytes(second_path.read_bytes())

    with pytest.raises(ValueError, match="验证报告与当前候选模型包不匹配"):
        web.validation_decision_payload({"run_id": first["run_id"], "decision": "passed", "comment": "should reject"})
    assert not (tmp_path / "runs" / first["run_id"] / "validated_model.pcamodel").exists()


def test_web_review_transaction_keeps_report_and_candidate_on_commit_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "normal_start": "2026-01-01T00:00:00", "normal_end": "2026-01-01T07:55:00", "sample_interval_minutes": 5, "smoothing_window_minutes": 10, "max_lag_minutes": 0, "lag_step_minutes": 5, "model_name": "candidate"})
    windows = [
        {"id": "normal-001", "type": "normal_validation", "start": "2026-01-01T08:00:00", "end": "2026-01-01T09:55:00", "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": "2026-01-01T10:50:00", "end": "2026-01-01T14:55:00", "enabled": True, "comment": "event"},
    ]
    web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    run_dir = tmp_path / "runs" / trained["run_id"]
    candidate = run_dir / "model.pcamodel"
    report = run_dir / "validation_report.json"
    original_candidate = candidate.read_bytes()
    original_report = report.read_bytes()
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report_replace(source, destination):
        if Path(destination) == report and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated report commit failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report_replace)
    with pytest.raises(OSError, match="simulated report commit failure"):
        web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved"})
    assert candidate.read_bytes() == original_candidate
    assert report.read_bytes() == original_report
    assert not (run_dir / "validated_model.pcamodel").exists()


def test_web_revalidation_failure_preserves_previous_evidence_and_download(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, _ = _create_passed_web_run(tmp_path, monkeypatch)
    paths = [
        run_dir / "model.pcamodel",
        run_dir / "validation_report.json",
        run_dir / "validation_scores.csv",
        run_dir / "validation_contributions.json",
        run_dir / "validated_model.pcamodel",
    ]
    original = {path.name: path.read_bytes() for path in paths}

    with pytest.raises(ValueError, match="training and validation windows overlap"):
        web.validate_payload(
            {
                "run_id": run_id,
                "file_id": file_id,
                "timestamp_column": "time",
                "validation_windows": [
                    {
                        "id": "overlap-001",
                        "type": "normal_validation",
                        "start": "2026-01-01T00:00:00",
                        "end": "2026-01-01T00:10:00",
                        "enabled": True,
                        "comment": "invalid",
                    }
                ],
            }
        )

    assert {path.name: path.read_bytes() for path in paths} == original
    assert not list(run_dir.glob(".*"))
    status, body = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 200
    assert body == original["validated_model.pcamodel"]


def test_web_revalidation_score_write_failure_rolls_back_all_evidence(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, windows = _create_passed_web_run(tmp_path, monkeypatch)
    paths = [
        run_dir / "model.pcamodel",
        run_dir / "validation_report.json",
        run_dir / "validation_scores.csv",
        run_dir / "validation_contributions.json",
        run_dir / "validated_model.pcamodel",
    ]
    original = {path.name: path.read_bytes() for path in paths}
    original_to_csv = pd.DataFrame.to_csv

    def fail_scores(self, path_or_buf=None, *args, **kwargs):
        if Path(path_or_buf).name.startswith(".validation_scores.csv."):
            raise OSError("simulated score write failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_scores)
    with pytest.raises(OSError, match="simulated score write failure"):
        web.validate_payload(
            {
                "run_id": run_id,
                "file_id": file_id,
                "timestamp_column": "time",
                "validation_windows": windows,
            }
        )

    assert {path.name: path.read_bytes() for path in paths} == original
    assert not list(run_dir.glob(".*"))


def test_web_revalidation_report_commit_failure_restores_all_evidence(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, windows = _create_passed_web_run(tmp_path, monkeypatch)
    paths = [
        run_dir / "model.pcamodel",
        run_dir / "validation_report.json",
        run_dir / "validation_scores.csv",
        run_dir / "validation_contributions.json",
        run_dir / "validated_model.pcamodel",
    ]
    original = {path.name: path.read_bytes() for path in paths}
    report_path = run_dir / "validation_report.json"
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report_install(source, destination):
        if Path(destination) == report_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated validation report commit failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report_install)
    with pytest.raises(OSError, match="simulated validation report commit failure"):
        web.validate_payload(
            {
                "run_id": run_id,
                "file_id": file_id,
                "timestamp_column": "time",
                "validation_windows": windows,
            }
        )

    assert {path.name: path.read_bytes() for path in paths} == original
    assert not list(run_dir.glob(".*"))


def test_web_revalidation_contribution_write_failure_restores_all_evidence(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, windows = _create_passed_web_run(tmp_path, monkeypatch)
    paths = [
        run_dir / "model.pcamodel",
        run_dir / "validation_report.json",
        run_dir / "validation_scores.csv",
        run_dir / "validation_contributions.json",
        run_dir / "validated_model.pcamodel",
    ]
    original = {path.name: path.read_bytes() for path in paths}
    original_write_text = Path.write_text

    def fail_contributions(self, data, *args, **kwargs):
        if self.name.startswith(".validation_contributions.json."):
            raise OSError("simulated contribution write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_contributions)
    with pytest.raises(OSError, match="simulated contribution write failure"):
        web.validate_payload(
            {
                "run_id": run_id,
                "file_id": file_id,
                "timestamp_column": "time",
                "validation_windows": windows,
            }
        )

    assert {path.name: path.read_bytes() for path in paths} == original
    assert not list(run_dir.glob(".*"))


def test_web_revalidation_rejects_untrusted_previous_validated_artifact(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, windows = _create_passed_web_run(tmp_path, monkeypatch)
    report_path = run_dir / "validation_report.json"
    original = {
        path.name: path.read_bytes()
        for path in (
            run_dir / "model.pcamodel",
            report_path,
            run_dir / "validation_scores.csv",
            run_dir / "validation_contributions.json",
            run_dir / "validated_model.pcamodel",
        )
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["engineer_decision"]["comment"] = "tampered"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    tampered_report = report_path.read_bytes()

    with pytest.raises(ValueError, match="验证报告与当前已验证模型包不一致"):
        web.validate_payload(
            {
                "run_id": run_id,
                "file_id": file_id,
                "timestamp_column": "time",
                "validation_windows": windows,
            }
        )

    assert (run_dir / "validated_model.pcamodel").exists()
    assert report_path.read_bytes() == tampered_report
    assert (run_dir / "model.pcamodel").read_bytes() == original["model.pcamodel"]
    assert (run_dir / "validation_scores.csv").read_bytes() == original["validation_scores.csv"]
    assert (run_dir / "validation_contributions.json").read_bytes() == original[
        "validation_contributions.json"
    ]
    assert not list(run_dir.glob(".*"))


def test_validated_model_download_route_rechecks_all_evidence_states(
    tmp_path, monkeypatch
):
    run_id, file_id, run_dir, windows = _create_passed_web_run(tmp_path, monkeypatch)
    report_path = run_dir / "validation_report.json"
    candidate_path = run_dir / "model.pcamodel"
    validated_path = run_dir / "validated_model.pcamodel"
    passed_report = report_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    validated_bytes = validated_path.read_bytes()

    status, body = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 200
    assert body == validated_bytes

    failed_report = json.loads(passed_report.decode("utf-8"))
    failed_report["engineer_decision"]["decision"] = "failed"
    report_path.write_text(json.dumps(failed_report), encoding="utf-8")
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400

    insufficient_report = json.loads(passed_report.decode("utf-8"))
    insufficient_report["engineer_decision"]["decision"] = "insufficient"
    report_path.write_text(json.dumps(insufficient_report), encoding="utf-8")
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400

    report_path.write_bytes(passed_report)
    candidate_path.write_bytes(b"candidate was replaced")
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400
    candidate_path.write_bytes(candidate_bytes)

    with zipfile.ZipFile(validated_path) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    manifest["source_candidate_package"]["sha256"] = "0" * 64
    with zipfile.ZipFile(validated_path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400
    validated_path.write_bytes(validated_bytes)

    mismatched_report = json.loads(passed_report.decode("utf-8"))
    mismatched_report["status_counts"] = {"normal": 999}
    report_path.write_text(json.dumps(mismatched_report), encoding="utf-8")
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400

    no_decision_report = json.loads(passed_report.decode("utf-8"))
    no_decision_report.pop("engineer_decision")
    report_path.write_text(json.dumps(no_decision_report), encoding="utf-8")
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400

    report_path.write_bytes(passed_report)
    validated_path.unlink()
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400

    validated_path.write_bytes(validated_bytes)
    web.validate_payload(
        {
            "run_id": run_id,
            "file_id": file_id,
            "timestamp_column": "time",
            "validation_windows": windows,
        }
    )
    assert not validated_path.exists()
    validated_path.write_bytes(validated_bytes)
    status, _ = _http_get(f"/download/validated-model?run_id={run_id}")
    assert status == 400


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


def test_web_quality_blocks_constant_tag_and_training_records_confirmed_exclusion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(77)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=100, freq="5min"),
            "FIXED": np.full(100, 50.0),
            "A": rng.normal(size=100),
            "B": rng.normal(size=100),
            "C": rng.normal(size=100),
        }
    )
    uploaded = web.save_upload(
        "history.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )
    common = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "encoding": "utf-8-sig",
        "normal_start": frame.time.iloc[0].isoformat(),
        "normal_end": frame.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 5,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "tag_configs": {
            "FIXED": {"role": "exclude"},
            "A": {"role": "continuous_input"},
            "B": {"role": "continuous_input"},
            "C": {"role": "continuous_input"},
        },
    }
    blocking = web.quality_payload(
        {
            **common,
            "tag_configs": {
                **common["tag_configs"],
                "FIXED": {"role": "continuous_input"},
            },
            "tags": ["FIXED", "A", "B", "C"],
        }
    )
    issue = next(
        item
        for item in blocking["tags"]
        if item["tag"] == "FIXED"
    )["issues"][0]

    assert not blocking["can_train"]
    assert issue["tag"] == "FIXED"
    with pytest.raises(ValueError, match="常量动态特征.*FIXED__lag_000min"):
        web.train_payload(
            {
                **common,
                "tag_configs": {
                    **common["tag_configs"],
                    "FIXED": {"role": "continuous_input"},
                },
                "tags": ["FIXED", "A", "B", "C"],
                "model_name": "BLOCKED",
            }
        )

    trained = web.train_payload(
        {
            **common,
            "tags": ["A", "B", "C"],
            "excluded_tags": [
                {
                    "tag": "FIXED",
                    "reason": "constant_in_reference_window",
                }
            ],
            "n_components": 2,
            "model_name": "EXCLUDED_DPCA_V1",
        }
    )
    _, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )

    assert manifest["config"]["excluded_tags"][0] == {
        "tag": "FIXED",
        "reason": "constant_in_reference_window",
        "sample_count": 100,
        "unique_count": 1,
        "constant_value": 50.0,
    }
    assert all(
        not name.startswith("FIXED__") for name in manifest["feature_names"]
    )


def test_web_xlsx_preview_and_trend_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    common = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "encoding": "utf-8-sig",
    }
    template = web.tag_config_template_payload(common)
    preview = web.tag_config_import_payload(
        "config.xlsx",
        template,
        uploaded["file_id"],
        "time",
        "utf-8-sig",
    )
    trend = web.trend_payload(
        {
            **common,
            "tags": ["A"],
            "start": history.time.iloc[0].isoformat(),
            "end": history.time.iloc[30].isoformat(),
            "normal_start": history.time.iloc[0].isoformat(),
            "normal_end": history.time.iloc[20].isoformat(),
            "display_mode": "both",
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 10,
            "lag_step_minutes": 5,
            "tag_configs": {
                "A": {"normal_min": -1, "normal_max": 1},
            },
        }
    )

    assert preview["can_apply"]
    assert preview["configs"]["A"]["description"] == ""
    assert len(trend["rows"]) == 31
    assert trend["statistics"]["A"]["reference"]["sample_count"] == 21
    assert sum(trend["histogram"]["counts"]) == 31
    with pytest.raises(ValueError, match="无宏"):
        web.tag_config_import_payload(
            "config.xlsm",
            template,
            uploaded["file_id"],
            "time",
            "utf-8-sig",
        )


def test_web_xlsx_preview_allows_unknown_tag_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    template = web.tag_config_template_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "encoding": "utf-8-sig",
        }
    )
    workbook = load_workbook(BytesIO(template))
    workbook["Tags"].append(["OLD_TAG", "旧配置"])
    modified = BytesIO()
    workbook.save(modified)

    preview = web.tag_config_import_payload(
        "config.xlsx",
        modified.getvalue(),
        uploaded["file_id"],
        "time",
        "utf-8-sig",
    )

    assert preview["can_apply"]
    assert preview["errors"] == []
    assert preview["warnings"] == ["OLD_TAG：当前历史数据中不存在"]
    assert "OLD_TAG" not in preview["provided_configs"]


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
