import json
import inspect
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from openpyxl import load_workbook
import pandas as pd
import pytest

from pca_model_builder.cli import build_parser
from pca_model_builder import cli, web, web_model_results
from pca_model_builder.model_io import load_deployment_package, load_model_package
from pca_model_builder.preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)
from pca_model_builder.training import build_training_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TXT_FIXTURE = Path(__file__).parent / "fixtures" / "u400ph_desensitized.txt"


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


def _xlsx_bytes(frame: pd.DataFrame) -> bytes:
    content = BytesIO()
    frame.to_excel(content, index=False)
    return content.getvalue()


def _post_response(handler_class, path: str, payload: dict) -> tuple[dict, int]:
    captured = {}
    handler = object.__new__(handler_class)
    handler.path = path
    handler._json_body = lambda: payload
    handler._send_json = lambda value, status=200: captured.update(
        value=value, status=status
    )
    handler.do_POST()
    return captured["value"], captured["status"]


def _get_response(handler_class, path: str) -> tuple[dict, int]:
    captured = {}
    handler = object.__new__(handler_class)
    handler.path = path
    handler._send_json = lambda value, status=200: captured.update(
        value=value, status=status
    )
    handler.do_GET()
    return captured["value"], captured["status"]


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


def test_upload_reads_only_file_header_and_basic_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        web,
        "inspect_data_quality",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    uploaded = web.save_upload("history.csv", b"time,A,B\n2026-01-01,1,2\n")

    assert uploaded["columns"] == ["time", "A", "B"]
    assert uploaded["size_bytes"] > 0
    assert "rows" not in uploaded
    assert (web.UPLOADS_DIR / f'{uploaded["file_id"]}.csv').is_file()


@pytest.mark.parametrize("point_count", [29999, 30000, 30001])
def test_inspection_trend_default_uses_up_to_30000_real_timestamps(
    tmp_path, monkeypatch, point_count
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    timestamps = pd.date_range("2026-01-01", periods=point_count, freq="min")
    frame = pd.DataFrame(
        {"time": timestamps, "A": np.arange(point_count), "B": np.arange(point_count)}
    )
    uploaded = web.save_upload(
        "trend-default.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )

    inspected = web.inspect_payload(
        {"file_id": uploaded["file_id"], "timestamp_column": "time"}
    )

    assert inspected["trend_default_start"] == timestamps[0].isoformat()
    assert inspected["trend_default_end"] == timestamps[
        min(point_count, 30000) - 1
    ].isoformat()


def test_upload_csv_read_error_is_explicit_and_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")

    with pytest.raises(ValueError, match="CSV读取失败："):
        web.save_upload("invalid.csv", b"\xff\xfe\xff")

    assert not list(web.UPLOADS_DIR.glob("*.csv"))


def test_upload_accepts_xlsx_and_inspects_raw_data_without_preprocessing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    source = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 01:00"]
            ),
            "A": [1.0, 2.0, 3.0],
            "B": [4.0, 5.0, 6.0],
        }
    )
    observed = []
    original_inspect = web.inspect_data_quality

    def record_inspection(frame, timestamp_column, numeric_columns):
        observed.append(frame.copy(deep=True))
        return original_inspect(frame, timestamp_column, numeric_columns)

    monkeypatch.setattr(web, "inspect_data_quality", record_inspection)
    monkeypatch.setattr(
        web,
        "preprocess_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    uploaded = web.save_upload("history.xlsx", _xlsx_bytes(source))
    inspected, status = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": uploaded["file_id"], "timestamp_column": "time"},
    )

    assert status == 200
    assert uploaded["file_type"] == "xlsx"
    assert "encoding" not in uploaded
    assert (web.UPLOADS_DIR / f'{uploaded["file_id"]}.xlsx').is_file()
    assert inspected["rows"] == len(source)
    assert inspected["columns"] == ["time", "A", "B"]
    assert list(observed[0].columns) == ["time", "A", "B"]
    assert observed[0]["A"].tolist() == source["A"].tolist()


def test_upload_accepts_u400ph_txt_and_inspects_raw_data_without_preprocessing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    observed = []
    original_inspect = web.inspect_data_quality

    def record_inspection(frame, timestamp_column, numeric_columns):
        observed.append(frame.copy(deep=True))
        return original_inspect(frame, timestamp_column, numeric_columns)

    monkeypatch.setattr(web, "inspect_data_quality", record_inspection)
    monkeypatch.setattr(
        web,
        "preprocess_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    uploaded = web.save_upload("U400PH.txt", TXT_FIXTURE.read_bytes())
    inspected, status = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": uploaded["file_id"], "timestamp_column": "TIME"},
    )

    assert status == 200
    assert uploaded["file_type"] == "txt"
    assert "encoding" not in uploaded
    assert (web.UPLOADS_DIR / f'{uploaded["file_id"]}.txt').is_file()
    assert inspected["rows"] == 3
    assert inspected["columns"][:3] == ["TIME", "AI450006.PV", "AIC450005.PV"]
    assert len(inspected["columns"]) == 15
    assert observed[0]["TIME"].tolist() == list(
        pd.to_datetime(["2026-04-24 12:00", "2026-04-24 12:01", "2026-04-24 12:02"])
    )
    assert observed[0]["AI450006.PV"].tolist() == pytest.approx(
        [0.651876032, 0.651416361, 0.652001739]
    )


def test_inspect_payload_profiles_all_original_columns_and_range_hints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="5min"),
            "A": range(8),
            "B": range(10, 18),
            "MIX": [-1.0, 0.5, 2.0, None, " ", "BAD", float("inf"), -float("inf")],
            "CONST": [7.0] * 8,
            "EMPTY": [None, " ", None, " ", None, " ", None, " "],
            "TEXT": ["label"] * 8,
        }
    )
    uploaded = web.save_upload(
        "raw-quality.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )
    tag_configs = {
        "MIX": {
            "engineering_min": 0.0,
            "engineering_max": 1.0,
            "normal_min": 0.25,
            "normal_max": 0.75,
            "alarm_min": 0.0,
            "alarm_max": 1.0,
        }
    }

    inspected = web.inspect_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tag_configs": tag_configs,
        }
    )
    profiles = {profile["tag"]: profile for profile in inspected["column_profiles"]}

    assert "TEXT" not in inspected["numeric_columns"]
    assert "preview_tags" not in inspected
    assert set(profiles) == {"A", "B", "MIX", "CONST", "EMPTY", "TEXT"}
    assert profiles["MIX"] | {"suggestion": None} == {
        "tag": "MIX",
        "sample_count": 8,
        "total_count": 8,
        "valid_count": 3,
        "finite_valid_count": 3,
        "missing_count": 1,
        "empty_string_count": 1,
        "non_numeric_count": 1,
        "positive_infinite_count": 1,
        "negative_infinite_count": 1,
        "non_finite_count": 2,
        "invalid_count": 5,
        "finite_unique_count": 3,
        "minimum": -1.0,
        "maximum": 2.0,
        "engineering_range_outside_count": 2,
        "normal_range_outside_count": 2,
        "alarm_range_outside_count": 2,
        "suggestion": None,
        "missing_rate": pytest.approx(0.125),
        "invalid_rate": pytest.approx(0.625),
        "unique_count": 3,
    }
    assert profiles["EMPTY"]["suggestion"]["reason"] == "all_empty"
    assert profiles["TEXT"]["suggestion"]["reason"] == "no_finite_numeric_values"
    assert profiles["CONST"]["suggestion"]["reason"] == "exact_constant_finite_values"
    assert tag_configs["MIX"]["engineering_min"] == 0.0


def test_inspect_payload_with_one_numeric_candidate_still_profiles_raw_columns(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        web,
        "preprocess_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "A": [1.0, 2.0, 3.0],
            "EMPTY": [None, None, None],
            "TEXT": ["label", "label", "label"],
        }
    )
    uploaded = web.save_upload(
        "one-candidate.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )

    inspected = web.inspect_payload(
        {"file_id": uploaded["file_id"], "timestamp_column": "time"}
    )
    profiles = {profile["tag"]: profile for profile in inspected["column_profiles"]}

    assert inspected["numeric_columns"] == ["A"]
    assert set(profiles) == {"A", "EMPTY", "TEXT"}
    assert profiles["EMPTY"]["suggestion"]["reason"] == "all_empty"
    assert profiles["TEXT"]["suggestion"]["reason"] == "no_finite_numeric_values"
    assert inspected["modeling_tag_hint"] == {
        "code": "insufficient_continuous_tags",
        "candidate_count": 1,
        "minimum_count": 2,
        "message": "当前可建模连续数值 Tag 少于 2 个，不能进入后续建模。",
    }


def test_inspect_payload_with_no_numeric_candidates_still_returns_profiles(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        web,
        "preprocess_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "EMPTY": [None, None, None],
            "TEXT": ["label", "label", "label"],
        }
    )
    uploaded = web.save_upload(
        "no-candidate.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )

    inspected = web.inspect_payload(
        {"file_id": uploaded["file_id"], "timestamp_column": "time"}
    )
    profiles = {profile["tag"]: profile for profile in inspected["column_profiles"]}

    assert inspected["numeric_columns"] == []
    assert set(profiles) == {"EMPTY", "TEXT"}
    assert profiles["EMPTY"]["suggestion"]["reason"] == "all_empty"
    assert profiles["TEXT"]["suggestion"]["reason"] == "no_finite_numeric_values"
    assert inspected["modeling_tag_hint"]["candidate_count"] == 0


def test_high_noise_preview_tags_are_stable_and_handle_invalid_sequences():
    frame = pd.DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0],
            "B": [0.0, 10.0, 0.0, 10.0],
            "C": [0.0, 10.0, 0.0, 10.0],
            "D": [0.0, np.nan, 0.0, np.inf],
            "E": [1.0, np.nan, np.nan, np.nan],
            "F": [0.0, 2.0, 4.0, 6.0],
        }
    )

    assert web._difference_noise_score(frame["E"]) is None
    assert web._difference_noise_score(frame["D"]) is None
    assert web._stable_high_noise_tags(frame, list(frame.columns)) == [
        "B",
        "C",
        "A",
        "F",
        "D",
    ]


def test_preprocessing_preview_empty_tags_auto_selects_stable_top_five(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "A": [0.0, 1.0, 2.0, 3.0],
            "B": [0.0, 10.0, 0.0, 10.0],
            "C": [0.0, 10.0, 0.0, 10.0],
            "D": [0.0, np.nan, 0.0, np.inf],
            "E": [1.0, np.nan, np.nan, np.nan],
            "F": [0.0, 2.0, 4.0, 6.0],
        }
    )
    uploaded = web.save_upload(
        "preview-top-five.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )

    result = web.preprocessing_preview_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": [],
            "start": frame.time.iloc[0].isoformat(),
            "end": frame.time.iloc[-1].isoformat(),
            "sample_interval_minutes": 5,
            "filter_method": "none",
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
        }
    )

    assert result["preview_tags"] == ["B", "C", "A", "F", "D"]
    assert list(result["raw"][0]) == [
        "timestamp",
        "physical_gap_start",
        "gap_start",
        "B",
        "C",
        "A",
        "F",
        "D",
    ]


def test_preprocessing_preview_auto_select_returns_fewer_than_five_available_tags(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "A": [0.0, 1.0, 2.0, 3.0],
            "B": [0.0, 10.0, 0.0, 10.0],
            "C": [0.0, 2.0, 4.0, 6.0],
        }
    )
    uploaded = web.save_upload(
        "preview-three.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )

    result = web.preprocessing_preview_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": [],
            "start": frame.time.iloc[0].isoformat(),
            "end": frame.time.iloc[-1].isoformat(),
            "sample_interval_minutes": 5,
            "filter_method": "none",
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
        }
    )

    assert result["preview_tags"] == ["B", "A", "C"]


def test_upload_rejects_invalid_txt_and_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")

    with pytest.raises(ValueError, match="TXT读取失败：.*Tab 分隔"):
        web.save_upload("invalid.txt", b"TIME,A,B\n2026/4/24 12:00,1,2\n")

    assert not list(web.UPLOADS_DIR.glob("*.txt"))


@pytest.mark.parametrize("incorrect_timestamp", ["MISSING_TIME", "AI450006.PV"])
def test_txt_timestamp_request_error_keeps_upload_for_retry(
    tmp_path, monkeypatch, incorrect_timestamp
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    uploaded = web.save_upload("U400PH.txt", TXT_FIXTURE.read_bytes())

    rejected, status = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": uploaded["file_id"], "timestamp_column": incorrect_timestamp},
    )
    retried, retry_status = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": uploaded["file_id"], "timestamp_column": "TIME"},
    )

    assert status == 400
    assert rejected["stage"] == "parsing"
    assert (web.UPLOADS_DIR / f'{uploaded["file_id"]}.txt').is_file()
    assert retry_status == 200
    assert retried["columns"][0] == "TIME"


def test_upload_rejects_unsupported_files_and_cleans_invalid_xlsx(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")

    with pytest.raises(ValueError, match="CSV、XLSX 或 TXT"):
        web.save_upload("history.json", b"{}")
    with pytest.raises(ValueError, match="XLSX读取失败："):
        web.save_upload("invalid.xlsx", b"not an xlsx")

    assert not list(web.UPLOADS_DIR.glob("*.xlsx"))


def test_web_xlsx_headers_are_strings_and_ignore_csv_encoding(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    source = pd.DataFrame(
        [
            ["2026-01-01 00:00", 1.0, 2.0],
            ["2026-01-01 00:05", 3.0, 4.0],
            ["2026-01-01 00:10", 5.0, 6.0],
        ],
        columns=[1001, 2002, 3003],
    )
    uploaded = web.save_upload("numeric.xlsx", _xlsx_bytes(source))
    inspected, status = _post_response(
        web._Handler,
        "/api/inspect",
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "1001",
            "encoding": "gb18030",
        },
    )

    assert status == 200
    assert uploaded["columns"] == ["1001", "2002", "3003"]
    assert inspected["columns"] == ["1001", "2002", "3003"]
    assert inspected["numeric_columns"] == ["2002", "3003"]


def test_web_rejects_xlsx_as_csv_encoding_and_keeps_one_upload_control(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    uploaded = web.save_upload(
        "history.csv", b"time,A,B\n2026-01-01 00:00,1,2\n2026-01-01 00:05,3,4\n2026-01-01 00:10,5,6\n"
    )
    rejected, status = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": uploaded["file_id"], "timestamp_column": "time", "encoding": "xlsx"},
    )

    assert status == 400
    assert rejected == {
        "error": "CSV 编码仅支持 UTF-8-SIG 或 GB18030",
        "stage": "loading",
    }
    assert web.INDEX_HTML.count('id="fileInput"') == 1
    assert '<option value="xlsx">' not in web.INDEX_HTML
    assert 'accept=".csv,.xlsx,.txt,' in web.INDEX_HTML
    assert '<option value="ascii">' not in web.INDEX_HTML


def test_final_web_page_exposes_typed_validation_and_engineer_decision_controls():
    html = web_model_results.INDEX_HTML
    for element_id in (
        'id="validationType"',
        'id="validationWindowTable"',
        'id="recordValidationDecision"',
        'id="validationDecisionStatus"',
        'id="validatedModelDownload"',
        'id="validationMetricDetails"',
        'id="contributionStability"',
    ):
        assert element_id in html
    assert "不能替代工程师确认" in html
    for label in (
        "正常样本验证",
        "已知异常验证",
        "通过",
        "结论不足",
        "不通过",
        "平均贡献率",
        "中位贡献率",
    ):
        assert label in html
    for field in ("average_contribution_pct", "median_contribution_pct"):
        assert field in html
    assert "function contributionPercent(value)" in html
    assert "Number(value).toFixed(1)}%" in html
    assert "contributionPercent(tag.average_contribution_pct)" in html
    assert "contributionPercent(tag.median_contribution_pct)" in html
    assert "percent(tag.average_contribution_pct)" not in html
    assert "percent(tag.median_contribution_pct)" not in html
    formatter = html.split("function contributionPercent(value)", 1)[1].split("\n", 1)[0]
    assert "Number(value).toFixed(1)" in formatter
    assert "*100" not in formatter


def test_validation_decision_handler_uses_current_validation_lifecycle_state():
    html = web_model_results.INDEX_HTML
    source = html.split(
        'el("recordValidationDecision").addEventListener("click",async()=>{', 1
    )[1].split('el("freezeDeployment").addEventListener', 1)[0]

    assert 'if(!state.validation||!state.runId)' in source
    assert 'api("/api/validation-decision"' in source
    assert "run_id:state.runId" in source
    assert 'decision:el("validationDecision").value' in source
    assert 'comment:el("validationDecisionComment").value.trim()' in source
    assert 'setBusy(button,true,"保存中…")' in source
    assert 'decisionStatus=el("validationDecisionStatus")' in source
    assert 'decisionStatus.textContent="正在保存工程师结论。"' in source
    assert 'download.href=data.validated_model_download||"#"' in source
    assert 'download.hidden=!data.validated_model_download' in source
    assert (
        "state.validation={...state.validation,model_status:data.model_status,"
        "engineer_decision:data.engineer_decision}; renderValidation(state.validation);"
        in source
    )
    assert '工程师结论已保存并生成已验证模型；' in source
    assert '工程师结论已保存，候选模型保持不变。' in source
    assert 'decisionStatus.textContent=message;' in source
    assert 'decisionStatus.className="status success"' in source
    assert 'catch(error) { decisionStatus.textContent=error.message;' in source
    assert 'decisionStatus.className="status error"' in source
    assert 'setStatus(error.message,"error")' in source


def test_successful_validation_replay_resets_engineer_decision_status():
    html = web_model_results.INDEX_HTML
    source = html.split(
        'el("validateButton").addEventListener("click", async () => {', 1
    )[1].split('el("recordValidationDecision").addEventListener', 1)[0]

    assert 'state.validation=data; renderValidation(data);' in source
    assert 'decisionStatus=el("validationDecisionStatus")' in source
    assert 'decisionStatus.textContent="等待保存工程师结论。"' in source
    assert 'decisionStatus.className="status info"' in source
    assert source.index('renderValidation(data);') < source.index(
        'decisionStatus.textContent="等待保存工程师结论。"'
    ) < source.index('setStatus("独立窗口回放完成。')
    assert 'validationDecisionStatus' not in source.split('} catch (error)', 1)[1]


def test_final_web_page_exposes_state_exploration_workbench():
    html = web_model_results.INDEX_HTML
    for element_id in (
        'id="stateExplorationPanel"',
        'id="stateExplorationButton"',
        'id="explorationClusterCount"',
        'id="explorationRandomState"',
        'id="explorationPerformanceTag"',
        'id="explorationPerformanceDirection"',
        'id="explorationPcChart"',
        'id="explorationTimeline"',
        'id="explorationClusterTable"',
        'id="explorationClusterCandidates"',
        'id="explorationPerformanceCandidates"',
        'id="explorationPreferredRegionCandidates"',
        'id="saveExplorationCandidateDecisions"',
        'id="convertExplorationCandidates"',
    ):
        assert element_id in html
    for label in (
        "状态探索工作台",
        "运行状态探索",
        "Cluster PC1 / PC2 与中心",
        "Cluster 时间轴",
        "Cluster 候选表",
        "性能候选表",
        "性能有效样本",
        "性能达标样本",
        "性能达标率",
        "性能中位数",
        "优选区域候选表",
    ):
        assert label in html
    assert "自动正常 Cluster" not in html
    assert "自动正常窗口" not in html
    assert "接受仅表示允许加入候选窗口，不会自动参与训练" in html
    assert "exploration-candidate-select" in html
    assert "selectedExplorationCandidateRows" in html


def test_state_exploration_timeline_uses_shared_colors_and_time_boundaries():
    html = web.INDEX_HTML
    timeline = html.split("function renderExplorationTimeline(rows,candidates)", 1)[1].split(
        "function explorationTimelineDetails", 1
    )[0]

    assert "const EXPLORATION_CLUSTER_PALETTE" in html
    assert "function explorationClusterColor(clusterId)" in html
    assert "explorationClusterColor(row.cluster_id)" in html
    assert "renderExplorationTimeline(data.cluster_series||[],data.cluster_candidates||[])" in html
    assert '<svg viewBox="0 0 ${width} ${height}"' in timeline
    assert "next.break_before||next.segment_id!==row.segment_id" in timeline
    assert "物理连续段断点" in timeline
    assert "候选窗口" in timeline
    assert "candidate.candidate_id" in timeline
    assert "显示点之间的时间跨度可能来自抽样" in timeline
    assert "explorationTimelineDetails(ordered)" in timeline
    assert '<details><summary>查看显示抽样点明细</summary>' in html
    assert '.map((value,index)=>`' in timeline
    assert 'text-anchor="${index===0?"start":index===3?"end":"middle"}"' in timeline
    assert 'index===0?"start"' in timeline
    assert 'index===3?"end"' in timeline
    assert ':"middle"' in timeline
    assert 'data.performance_config?.direction==="target_range"' in html
    assert 'row.performance_target_met===true' in html
    assert 'stroke="#111827"' in html
    assert "◎ 性能达标" in html


def test_web_exposes_preferred_region_controls_and_full_sample_evaluation():
    html = web.INDEX_HTML
    for element_id in (
        'id="explorationRegionSelect"',
        'id="explorationRegionDelete"',
        'id="explorationRegionClear"',
        'id="explorationRegionSummary"',
    ):
        assert element_id in html
    for label in ("椭圆选择", "删除上一个", "清除区域", "优选运行区域质量统计"):
        assert label in html
    assert "preferred-region" in html
    assert "center_pc1" in html
    assert "radius_pc2" in html
    assert "getBoundingClientRect" in html
    assert "state.preferredRegionRequest" in html
    assert "preferredRegionUpdateSeq:0" in html
    assert "preferred_region_update_seq:updateSeq" in html
    assert "data.applied===false" in html
    assert "完整有效样本占比" in html
    assert "区域稳定性" in html


def test_preferred_region_api_uses_union_and_full_cached_series():
    web.clear_state_exploration_cache()
    run_id = "c" * 32
    index = pd.date_range("2026-01-01", periods=5, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": [-1.0, 0.0, 1.0, 2.0, 3.0],
            "pc2": [0.0] * 5,
            "pc3": [0.0, 10.0, 0.0, 2.0, 3.0],
            "cluster_id": ["cluster_001", "cluster_001", "cluster_002", "cluster_002", "cluster_002"],
            "segment_id": [0] * 5,
        },
        index=index,
    )
    web._store_state_exploration_run(
        run_id,
        {
            "exploratory_model_summary": {"pc_columns": ["pc1", "pc2", "pc3"]},
            "cluster_centers": {
                "cluster_001": [0.0, 0.0, 0.0],
                "cluster_002": [0.0, 0.0, 0.0],
            },
            "performance_config": {
                "performance_tag": "PERF",
                "direction": "target_range",
                "target_min": 1.5,
                "target_max": 4.5,
            },
            "_performance_values": pd.Series(
                [1.0, 2.0, np.inf, 4.0, np.nan], index=index
            ),
            "cluster_series": points,
            "cluster_series_display": points.iloc[[0, 4]],
            "cluster_candidates": [],
            "performance_candidates": [],
            "candidate_decisions": [],
        },
    )

    payload = {
        "exploration_run_id": run_id,
        "preferred_region_update_seq": 1,
        "ellipses": [
            {
                "center_pc1": 0.0,
                "center_pc2": 0.0,
                "radius_pc1": 1.1,
                "radius_pc2": 1.0,
            },
            {
                "center_pc1": 2.0,
                "center_pc2": 0.0,
                "radius_pc1": 1.1,
                "radius_pc2": 1.0,
            },
        ],
    }
    result, status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        payload,
    )

    assert status == 200
    assert len(result["ellipses"]) == 2
    assert result["selected_sample_count"] == 5
    assert result["full_valid_sample_count"] == 5
    assert result["cluster_counts"] == [
        {"cluster_id": "cluster_001", "sample_count": 2, "share": 0.4},
        {"cluster_id": "cluster_002", "sample_count": 3, "share": 0.6},
    ]
    assert result["performance_valid_count"] == 3
    assert result["performance_target_count"] == 2
    assert result["performance_target_ratio"] == 2 / 3
    assert result["performance_median"] == 2.0
    assert result["preferred_region"] == {
        key: result[key]
        for key in result["preferred_region"]
        if key != "ellipses"
    } | {"ellipses": result["ellipses"]}

    summary, summary_status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}"
    )
    assert summary_status == 200
    assert summary["preferred_region"]["selected_sample_count"] == 5
    assert "_performance_values" not in summary

    cleared, clear_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/region",
        {"ellipses": [], "preferred_region_update_seq": 2},
    )
    assert clear_status == 200
    assert cleared["selected_sample_count"] == 0
    assert cleared["ellipses"] == []

    invalid, invalid_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        {
            "preferred_region_update_seq": 3,
            "ellipses": [
                {
                    "center_pc1": 0,
                    "center_pc2": 0,
                    "radius_pc1": 0,
                    "radius_pc2": 1,
                }
            ]
        },
    )
    assert invalid_status == 400
    assert "半轴" in invalid["error"]
    assert "traceback" not in invalid["error"].lower()


def test_preferred_region_candidates_recompute_and_keep_conversion_lifecycle():
    web.clear_state_exploration_cache()
    run_id = "d" * 32
    index = pd.date_range("2026-01-01", periods=12, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(12),
            "pc2": [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "pc3": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "cluster_id": ["cluster_001"] * 3 + ["cluster_002"] * 9,
            "segment_id": [0] * 8 + [1] * 4,
        },
        index=index,
    )
    cluster_id = "cluster-aux-001"
    performance_id = "performance-aux-001"
    web._store_state_exploration_run(
        run_id,
        {
            "exploration_config": {"minimum_candidate_duration_minutes": 10},
            "preprocessing_summary": {"target_interval_minutes": 5},
            "exploratory_model_summary": {"pc_columns": ["pc1", "pc2", "pc3"]},
            "cluster_centers": {
                "cluster_001": [0.0, 0.0, 0.0],
                "cluster_002": [0.0, 0.0, 0.0],
            },
            "performance_config": {
                "performance_tag": "PERF",
                "direction": "target_range",
                "target_min": 1.0,
                "target_max": 2.0,
            },
            "_performance_values": pd.Series(
                [1.0, np.nan, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                index=index,
            ),
            "cluster_series": points,
            "cluster_series_display": points.iloc[[0, 11]],
            "cluster_candidates": [
                {
                    "candidate_id": cluster_id,
                    "source": "cluster",
                    "start": index[0].isoformat(),
                    "end": index[1].isoformat(),
                }
            ],
            "performance_candidates": [
                {
                    "candidate_id": performance_id,
                    "source": "performance",
                    "start": index[4].isoformat(),
                    "end": index[5].isoformat(),
                }
            ],
            "preferred_region_candidates": [],
            "candidate_decisions": [
                {
                    "candidate_id": cluster_id,
                    "decision": "accepted",
                    "comment": "保留",
                    "decided_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "candidate_id": performance_id,
                    "decision": "rejected",
                    "comment": "忽略",
                    "decided_at": "2026-01-01T00:00:00+00:00",
                },
            ],
        },
    )

    first, first_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        {
            "exploration_run_id": run_id,
            "preferred_region_update_seq": 1,
            "ellipses": [
                {
                    "center_pc1": 0.0,
                    "center_pc2": 0.0,
                    "radius_pc1": 1.0,
                    "radius_pc2": 1.0,
                },
                {
                    "center_pc1": 0.0,
                    "center_pc2": 10.0,
                    "radius_pc1": 1.0,
                    "radius_pc2": 1.0,
                },
            ],
        },
    )

    assert first_status == 200
    assert [item["sample_count"] for item in first["preferred_region_candidates"]] == [4, 3, 3]
    assert [item["start"] for item in first["preferred_region_candidates"]] == [
        index[8].isoformat(),
        index[0].isoformat(),
        index[4].isoformat(),
    ]
    assert first["preferred_region_candidates"][1]["performance_valid_count"] == 2
    assert first["preferred_region_candidates"][1]["performance_target_ratio"] == 1.0
    assert first["preferred_region_candidates"][1]["performance_median"] == 1.5
    decisions = {item["candidate_id"]: item for item in first["candidate_decisions"]}
    assert decisions[cluster_id]["decision"] == "accepted"
    assert decisions[performance_id]["decision"] == "rejected"
    old_region_candidate = next(
        item
        for item in first["preferred_region_candidates"]
        if item["start"] == index[0].isoformat()
    )

    accepted, accepted_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/decisions",
        {
            "candidate_id": old_region_candidate["candidate_id"],
            "decision": "accepted",
            "comment": "优选区域稳定",
        },
    )
    assert accepted_status == 200
    converted = web.state_exploration_training_windows_payload(
        run_id,
        {
            "candidate_ids": [old_region_candidate["candidate_id"]],
            "training_windows": [],
        },
    )
    converted_window = converted["training_windows"][0]
    assert converted_window["enabled"] is False
    assert converted_window["source"] == "preferred_region"

    second, second_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/region",
        {
            "preferred_region_update_seq": 2,
            "ellipses": [
                {
                    "center_pc1": 0.0,
                    "center_pc2": 10.0,
                    "radius_pc1": 1.0,
                    "radius_pc2": 1.0,
                }
            ],
        },
    )

    assert second_status == 200
    assert [item["start"] for item in second["preferred_region_candidates"]] == [
        index[8].isoformat()
    ]
    refreshed_decisions = {
        item["candidate_id"]: item for item in second["candidate_decisions"]
    }
    assert refreshed_decisions[cluster_id]["decision"] == "accepted"
    assert refreshed_decisions[performance_id]["decision"] == "rejected"
    assert old_region_candidate["candidate_id"] not in refreshed_decisions

    stale, stale_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/decisions",
        {
            "candidate_id": old_region_candidate["candidate_id"],
            "decision": "accepted",
            "comment": "不应继续有效",
        },
    )
    assert stale_status == 400
    assert "不属于当前状态探索运行" in stale["error"]

    new_region_candidate = second["preferred_region_candidates"][0]
    web.state_exploration_decisions_payload(
        run_id,
        {
            "candidate_id": new_region_candidate["candidate_id"],
            "decision": "accepted",
            "comment": "保留第二段",
        },
    )
    preserved = web.state_exploration_training_windows_payload(
        run_id,
        {
            "candidate_ids": [new_region_candidate["candidate_id"]],
            "training_windows": converted["training_windows"],
        },
    )
    assert preserved["training_windows"][0] == converted_window
    assert len(preserved["training_windows"]) == 2


def test_preferred_region_out_of_order_updates_keep_latest_cache_and_decisions():
    web.clear_state_exploration_cache()
    run_id = "e" * 32
    index = pd.date_range("2026-01-01", periods=12, freq="5min")
    points = pd.DataFrame(
        {
            "pc1": np.zeros(12),
            "pc2": [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 10.0],
            "pc3": np.zeros(12),
            "cluster_id": ["cluster_001"] * 8 + ["cluster_002"] * 4,
            "segment_id": [0] * 8 + [1] * 4,
        },
        index=index,
    )
    cluster_id = "cluster-aux-001"
    performance_id = "performance-aux-001"
    web._store_state_exploration_run(
        run_id,
        {
            "exploration_config": {"minimum_candidate_duration_minutes": 10},
            "preprocessing_summary": {"target_interval_minutes": 5},
            "exploratory_model_summary": {"pc_columns": ["pc1", "pc2", "pc3"]},
            "cluster_centers": {
                "cluster_001": [0.0, 0.0, 0.0],
                "cluster_002": [0.0, 0.0, 0.0],
            },
            "cluster_series": points,
            "cluster_series_display": points.iloc[[0, 11]],
            "cluster_candidates": [
                {
                    "candidate_id": cluster_id,
                    "source": "cluster",
                    "start": index[0].isoformat(),
                    "end": index[1].isoformat(),
                }
            ],
            "performance_candidates": [
                {
                    "candidate_id": performance_id,
                    "source": "performance",
                    "start": index[4].isoformat(),
                    "end": index[5].isoformat(),
                }
            ],
            "preferred_region_candidates": [],
            "candidate_decisions": [
                {
                    "candidate_id": cluster_id,
                    "decision": "accepted",
                    "comment": "保留 Cluster",
                    "decided_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "candidate_id": performance_id,
                    "decision": "rejected",
                    "comment": "忽略性能",
                    "decided_at": "2026-01-01T00:00:00+00:00",
                },
            ],
        },
    )
    old_ellipses = [
        {
            "center_pc1": 0.0,
            "center_pc2": 0.0,
            "radius_pc1": 1.0,
            "radius_pc2": 1.0,
        }
    ]
    new_ellipses = [
        {
            "center_pc1": 0.0,
            "center_pc2": 10.0,
            "radius_pc1": 1.0,
            "radius_pc2": 1.0,
        }
    ]

    newest, newest_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        {
            "exploration_run_id": run_id,
            "preferred_region_update_seq": 2,
            "ellipses": new_ellipses,
        },
    )
    assert newest_status == 200
    assert newest["applied"] is True
    assert newest["preferred_region_update_seq"] == 2
    newest_candidate_ids = {
        item["candidate_id"] for item in newest["preferred_region_candidates"]
    }
    assert [item["start"] for item in newest["preferred_region_candidates"]] == [
        index[8].isoformat()
    ]

    stale, stale_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        {
            "exploration_run_id": run_id,
            "preferred_region_update_seq": 1,
            "ellipses": old_ellipses,
        },
    )
    assert stale_status == 200
    assert stale["applied"] is False
    assert stale["preferred_region_update_seq"] == 2
    assert stale["ellipses"] == newest["ellipses"]
    assert stale["preferred_region_candidates"] == newest["preferred_region_candidates"]

    cached = web._state_exploration_run(run_id)
    assert cached["preferred_region_update_seq"] == 2
    assert cached["preferred_region"]["ellipses"] == newest["ellipses"]
    assert cached["preferred_region_candidates"] == newest["preferred_region_candidates"]
    assert {
        item["candidate_id"] for item in cached["preferred_region_candidates"]
    } == newest_candidate_ids
    assert all(
        item["start"] not in {index[0].isoformat(), index[4].isoformat()}
        for item in cached["preferred_region_candidates"]
    )
    assert cached["candidate_decisions"] == newest["candidate_decisions"]

    region_candidate = newest["preferred_region_candidates"][0]
    accepted, accepted_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/decisions",
        {
            "candidate_id": region_candidate["candidate_id"],
            "decision": "accepted",
            "comment": "保留最新优选区域",
        },
    )
    assert accepted_status == 200
    assert next(
        item
        for item in accepted["candidate_decisions"]
        if item["candidate_id"] == region_candidate["candidate_id"]
    )["decision"] == "accepted"

    existing_window = {
        "id": "existing-window",
        "start": index[0].isoformat(),
        "end": index[1].isoformat(),
        "source": "manual",
        "source_ref": None,
        "enabled": False,
        "comment": "已有窗口",
    }
    converted = web.state_exploration_training_windows_payload(
        run_id,
        {
            "candidate_ids": [region_candidate["candidate_id"]],
            "training_windows": [existing_window],
        },
    )
    assert converted["training_windows"][0] == existing_window
    assert converted["converted_candidate_ids"] == [region_candidate["candidate_id"]]
    assert converted["training_windows"][1]["source"] == "preferred_region"
    assert converted["training_windows"][1]["enabled"] is False


def test_exploration_series_uses_full_target_status_for_display_points():
    index = pd.date_range("2026-01-01", periods=3, freq="5min")
    full = pd.DataFrame(
        {
            "pc1": [0.0, 1.0, 2.0],
            "pc2": [2.0, 1.0, 0.0],
            "cluster_id": ["cluster_001"] * 3,
            "segment_id": [0] * 3,
            "performance_target_met": [True, False, None],
        },
        index=index,
    )
    display = full.iloc[[0, 2]].drop(columns="performance_target_met")

    rows = web._exploration_series(display, full, 5)

    assert [row["performance_target_met"] for row in rows] == [True, None]


def test_state_exploration_target_range_exposes_full_status_and_cluster_metrics(
    tmp_path, monkeypatch
):
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    web.clear_state_exploration_cache()
    history = _history_frame()
    history["PERF"] = np.r_[np.full(90, 5.0), np.full(90, 1.0)]
    uploaded = web.save_upload(
        "exploration-target.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "exploration_start": history.time.iloc[0].isoformat(),
        "exploration_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 0,
        "filter_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "exploration_config": {
            "cluster_count": 2,
            "minimum_candidate_duration_minutes": 10,
            "maximum_plot_points": 12,
        },
        "performance_config": {
            "performance_tag": "PERF",
            "direction": "target_range",
            "target_min": 4.0,
            "target_max": 6.0,
            "minimum_duration_minutes": 10,
            "candidate_count": 1,
        },
    }

    result = web.state_exploration_payload(payload)
    run_id = result["exploration_run_id"]
    cached = web._state_exploration_run(run_id)

    assert result["cluster_series"]
    assert result["preprocessing_summary"]["dynamic_feature_count"] == 3
    assert all("performance_target_met" in row for row in result["cluster_series"])
    assert {row["performance_target_met"] for row in result["cluster_series"]} == {
        True,
        False,
    }
    assert all(
        {
            "performance_valid_count",
            "performance_target_count",
            "performance_target_ratio",
            "performance_median",
        }.issubset(summary)
        for summary in result["cluster_summaries"]
    )
    assert sum(item["performance_valid_count"] for item in result["cluster_summaries"]) == 180
    assert sum(item["performance_target_count"] for item in result["cluster_summaries"]) == 90

    region, region_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/preferred-region",
        {
            "exploration_run_id": run_id,
            "preferred_region_update_seq": 1,
            "ellipses": [
                {
                    "center_pc1": 0.0,
                    "center_pc2": 0.0,
                    "radius_pc1": 1_000_000.0,
                    "radius_pc2": 1_000_000.0,
                }
            ],
        },
    )
    assert region_status == 200
    assert region["selected_sample_count"] == 180
    assert region["performance_valid_count"] == 180
    assert region["performance_target_count"] == 90
    assert region["performance_target_ratio"] == 0.5
    assert region["performance_median"] == 3.0

    bounded, status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}/series?max_points=5"
    )

    assert status == 200
    assert bounded["returned_point_count"] <= 5
    for row in bounded["cluster_series"]:
        assert row["performance_target_met"] == cached["cluster_series"].loc[
            pd.Timestamp(row["timestamp"]), "performance_target_met"
        ]


def test_final_web_compacts_basic_inspection_time_range_into_two_lines():
    html = web_model_results.INDEX_HTML

    assert ".metric.time-range strong" in html
    assert "font-size:14px" in html
    assert 'metric("时间范围",`${displayTime(data.time_start)}\\n${displayTime(data.time_end)}`,"time-range")' in html


def test_final_web_formats_displayed_timestamps_with_a_space():
    html = web_model_results.INDEX_HTML

    assert 'function displayTime(value,length=16)' in html
    assert '.replace("T"," ")' in html
    assert 'displayTime(window.start)' in html
    assert 'displayTime(rows[0].timestamp)' in html
    assert 'displayTime(firstTime)' in html
    assert 'el("analysisStart").value=localTime' in html
    assert 'row.timestamp.slice(0,19)' not in html


def test_final_web_entry_uses_cached_base_reading_paths() -> None:
    from pca_model_builder import web_dataproject

    assert "_BASE_WEB.train_payload(payload)" in inspect.getsource(
        web_model_results.train_payload
    )
    source = inspect.getsource(web_dataproject.trend_payload)
    assert "base_web._load_required_upload" in source
    assert "base_web._state_filter_columns" in source


def test_final_web_page_exposes_read_only_model_structure_comparison() -> None:
    html = web_model_results.INDEX_HTML
    source = (PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js").read_text(
        encoding="utf-8"
    )

    assert 'src="/assets/model-results.js"' in html
    assert "选择已训练候选模型" in source
    assert "选择 2—4 个已训练候选模型" not in source
    for text in (
        "模型结构与参数比较",
        "不能替代独立验证",
        "不会自动评分、推荐、验证或改变模型状态",
        'id="modelComparisonRuns"',
        "/api/model-comparison",
        "解释率累计曲线",
        "原始Tag平方载荷能量",
        "Lag平方载荷能量",
        "当前为探索草稿模型；仅正常状态候选模型显示候选模型结构诊断。",
    ):
        assert text in source
    assert "最佳模型" not in source


def test_model_structure_diagnostic_labels_retained_components_and_energy_tables() -> None:
    source = (PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js").read_text(
        encoding="utf-8"
    )
    html = web_model_results.INDEX_HTML

    assert "保留主元：${retained}；累计解释率：${(ratio * 100).toFixed(2)}%" in source
    assert "diagnostic.cumulative_explained_variance_ratio[retained - 1]" in source
    assert '"主元数量"' in source
    assert '"累计解释率（%）"' in source
    assert "x(point + 1)" in source
    assert "`保留 ${retained}`" in source
    assert 'container.className = `model-energy-table ${key === "tag" ? "tag-energy-table" : "lag-energy-table"}`;' in source
    assert 'container.className = "model-energy-grid"' in source
    assert "#modelStructureComparison .model-energy-table" in html
    assert "#modelStructureComparison .model-energy-grid" in html
    assert "grid-template-columns:minmax(0,1.35fr) minmax(220px,.65fr);" in html
    assert "width:min(100%,220px);" in html
    assert "table-layout:fixed;" in html
    assert "overflow-wrap:anywhere;" in html
    assert "#modelStructureComparison .model-energy-table th:nth-child(2)," in html
    assert "text-align:right;" in html
    assert "@media (max-width:760px)" in html


def test_model_comparison_layout_scopes_select_and_wraps_parameter_table_text() -> None:
    source = (PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js").read_text(
        encoding="utf-8"
    )
    html = web_model_results.INDEX_HTML

    assert "#modelStructureComparison #modelComparisonRuns" in html
    assert "height:auto;" in html
    assert "min-height:0;" in html
    assert 'table.className = "model-parameter-table";' in source
    assert "#modelStructureComparison .model-parameter-table" in html
    assert "max-width:100%;" in html
    assert "table-layout:fixed;" in html
    assert "#modelStructureComparison .model-parameter-table th:first-child," in html
    assert "width:12em;" in html
    assert "white-space:normal;" in html
    assert "overflow-wrap:anywhere;" in html
    assert "word-break:break-word;" in html


def test_final_web_model_comparison_routes_only_read_saved_candidates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    base = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[119].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 5,
        "lag_step_minutes": 5,
        "model_name": "comparison-candidate",
    }
    first = web_model_results.train_payload(base)
    second = web_model_results.train_payload({**base, "smoothing_window_minutes": 5})
    exploratory = web_model_results.train_payload(
        {**base, "model_purpose": "exploratory"}
    )
    first_path = tmp_path / "runs" / first["run_id"] / "model.pcamodel"
    before = first_path.read_bytes(), first_path.stat().st_mtime_ns

    diagnostic, diagnostic_status = _post_response(
        web_model_results.ModelResultsHandler,
        "/api/model-diagnostics",
        {"run_id": first["run_id"]},
    )
    candidates, candidates_status = _get_response(
        web_model_results.ModelResultsHandler, "/api/model-candidates"
    )
    compared, compared_status = _post_response(
        web_model_results.ModelResultsHandler,
        "/api/model-comparison",
        {"run_ids": [first["run_id"], second["run_id"]]},
    )

    assert diagnostic_status == 200
    assert diagnostic["run_id"] == first["run_id"]
    assert first["model_diagnostic"]["run_id"] == first["run_id"]
    assert "model_diagnostic" not in exploratory
    with pytest.raises(ValueError, match="仅允许查看normal_state/candidate模型诊断"):
        web_model_results.model_diagnostic_payload(
            {"run_id": exploratory["run_id"]}
        )
    assert candidates_status == 200
    assert {item["run_id"] for item in candidates["candidates"]} == {
        first["run_id"],
        second["run_id"],
    }
    assert compared_status == 200
    assert compared["comparability"]["comparable"] is True
    assert "smoothing_window_minutes" in compared["parameter_differences"][0]["differences"]
    assert first_path.read_bytes() == before[0]
    assert first_path.stat().st_mtime_ns == before[1]


def test_final_web_deletes_candidate_run_directory_and_refreshes_list(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[119].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 5,
        "lag_step_minutes": 5,
        "model_name": "delete-candidate",
    }
    first = web_model_results.train_payload(payload)
    second = web_model_results.train_payload({**payload, "smoothing_window_minutes": 5})
    first_dir = tmp_path / "runs" / first["run_id"]
    second_dir = tmp_path / "runs" / second["run_id"]

    candidates, list_status = _get_response(
        web_model_results.ModelResultsHandler, "/api/model-candidates"
    )
    deleted, delete_status = _post_response(
        web_model_results.ModelResultsHandler,
        "/api/model-candidates/delete",
        {"run_ids": [first["run_id"]]},
    )
    refreshed, refreshed_status = _get_response(
        web_model_results.ModelResultsHandler, "/api/model-candidates"
    )

    assert list_status == 200
    assert all(item["deletable"] for item in candidates["candidates"])
    assert delete_status == 200
    assert deleted == {"deleted_run_ids": [first["run_id"]]}
    assert not first_dir.exists()
    assert second_dir.is_dir()
    assert refreshed_status == 200
    assert {item["run_id"] for item in refreshed["candidates"]} == {second["run_id"]}


def test_final_web_candidate_deletion_validates_batch_before_deleting_any_run(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[119].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 5,
        "lag_step_minutes": 5,
        "model_name": "protected-candidate",
    }
    deletable = web_model_results.train_payload(payload)
    protected = web_model_results.train_payload(
        {**payload, "smoothing_window_minutes": 5}
    )
    protected_dir = tmp_path / "runs" / protected["run_id"]
    protected_artifacts = (
        "validated_model.pcamodel",
        "frozen_model.pcamodel",
        "deployment_model.pcadeploy",
    )

    (protected_dir / protected_artifacts[0]).write_bytes(b"protected")
    before, before_status = _get_response(
        web_model_results.ModelResultsHandler, "/api/model-candidates"
    )
    rejected, rejected_status = _post_response(
        web_model_results.ModelResultsHandler,
        "/api/model-candidates/delete",
        {"run_ids": [deletable["run_id"], protected["run_id"]]},
    )

    assert before_status == 200
    protected_item = next(
        item for item in before["candidates"] if item["run_id"] == protected["run_id"]
    )
    assert protected_item["deletable"] is False
    assert "validated_model.pcamodel" in protected_item["deletion_block_reason"]
    assert rejected_status == 400
    assert "正式下游工件" in rejected["error"]
    assert (tmp_path / "runs" / deletable["run_id"]).is_dir()
    assert protected_dir.is_dir()

    (protected_dir / protected_artifacts[0]).unlink()
    for artifact in protected_artifacts[1:]:
        (protected_dir / artifact).write_bytes(b"protected")
        rejected, rejected_status = _post_response(
            web_model_results.ModelResultsHandler,
            "/api/model-candidates/delete",
            {"run_ids": [protected["run_id"]]},
        )
        assert rejected_status == 400
        assert "正式下游工件" in rejected["error"]
        assert protected_dir.is_dir()
        (protected_dir / artifact).unlink()


def test_final_web_candidate_deletion_rejects_invalid_lifecycle_corrupt_package_and_run_id(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[119].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 5,
        "lag_step_minutes": 5,
        "model_name": "deletion-validation",
    }
    exploratory = web_model_results.train_payload(
        {**payload, "model_purpose": "exploratory"}
    )
    candidate = web_model_results.train_payload(payload)
    corrupt_id = "a" * 32
    corrupt_dir = tmp_path / "runs" / corrupt_id
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "model.pcamodel").write_bytes(b"not-a-model-package")

    for run_id, expected in (
        (exploratory["run_id"], "normal_state/candidate"),
        (corrupt_id, "模型包损坏"),
    ):
        rejected, rejected_status = _post_response(
            web_model_results.ModelResultsHandler,
            "/api/model-candidates/delete",
            {"run_ids": [run_id]},
        )
        assert rejected_status == 400
        assert expected in rejected["error"]
    rejected, rejected_status = _post_response(
        web_model_results.ModelResultsHandler,
        "/api/model-candidates/delete",
        {"run_ids": [candidate["run_id"], "../outside"]},
    )

    assert rejected_status == 400
    assert "无效的 run_id" in rejected["error"]
    assert (tmp_path / "runs" / candidate["run_id"]).is_dir()
    assert corrupt_dir.is_dir()


def test_final_web_candidate_deletion_frontend_guards_and_refreshes_comparison_state() -> None:
    source = (PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js").read_text(
        encoding="utf-8"
    )

    for text in (
        'id="deleteModelsButton"',
        "删除所选候选模型",
        "/api/model-candidates/delete",
        "当前正在使用的候选模型不能删除",
        "deletion_block_reason",
        "删除后不可恢复",
        "refreshCandidateOptions(state.runId)",
    ):
        assert text in source
    assert "模型比较需要选择 2—4 个候选模型。" in source
    assert source.index("当前正在使用的候选模型不能删除") < source.index(
        'fetch("/api/model-candidates/delete"'
    )
    assert source.index("所选候选模型不能删除") < source.index(
        'fetch("/api/model-candidates/delete"'
    )
    assert source.index('document.getElementById("modelComparisonResult").replaceChildren()') < source.index(
        "refreshCandidateOptions(state.runId)"
    )


def test_web_exposes_preprocessing_controls_and_preview_route():
    html = web_model_results.INDEX_HTML
    for element_id in (
        'id="resamplingMethod"',
        'id="filterMethod"',
        'id="firstOrderAlpha"',
        'id="gapThreshold"',
        'id="preprocessingPreviewButton"',
    ):
        assert element_id in html
    assert html.count('id="firstOrderAlpha"') == 1
    assert '<option value="none" selected>不滤波</option>' in html
    assert 'option value="trailing_median"' not in html
    assert "/api/preprocessing-preview" in html
    assert "查看抽样数据明细" not in html
    assert "preprocessing-preview-details" not in html
    for label in ("原始数据", "重采样后数据", "一阶低通滤波后数据", "移动平均后数据", "滤波后数据"):
        assert label in html
    assert '<option value="first_order">一阶低通滤波</option>' in html


def test_web_compacts_training_parameters_and_keeps_preview_below_resampling():
    html = web_model_results.INDEX_HTML
    training = html.split('class="training-parameter-grid"', 1)[1].split(
        '<details class="advanced-parameters">', 1
    )[0]
    advanced = html.split('<details class="advanced-parameters">', 1)[1].split(
        "</details>", 1
    )[0]

    assert 'class="training-parameter-grid"' in html
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in html
    assert "@media (max-width:1100px)" in html
    for field_id in ("lagStep", "components"):
        assert f'id="{field_id}"' in training
        assert f'id="{field_id}"' not in advanced
        assert html.count(f'id="{field_id}"') == 1
    assert training.index('id="maxLag"') < training.index('id="lagStep"') < training.index(
        'id="varianceThreshold"'
    ) < training.index('id="components"') < training.index('id="modelName"')
    assert 'class="model-name-field"' in training
    assert advanced.index('id="resamplingMethod"') < advanced.index(
        'id="gapThreshold"'
    ) < advanced.index('id="preprocessingPreviewButton"') < advanced.index(
        'id="preprocessingPreview"'
    )
    assert 'class="row advanced-preprocessing-row"' in advanced
    assert 'class="preprocessing-preview-area"' in advanced
    assert advanced.count('id="preprocessingPreviewButton"') == 1
    assert advanced.count('id="preprocessingPreview"') == 1
    assert html.count('id="resamplingMethod"') == 1
    assert html.count('id="gapThreshold"') == 1


def test_web_preprocessing_preview_validates_first_order_alpha_locally():
    html = web.INDEX_HTML
    preview_handler = html.split(
        'el("preprocessingPreviewButton").addEventListener("click",async()=>{', 1
    )[1].split("function renderPreprocessingPreview()", 1)[0]

    assert 'placeholder="例如 0.2"' in html
    assert 'function firstOrderAlphaError()' in html
    assert 'el("firstOrderAlpha").closest("label").hidden=!firstOrder' in html
    assert 'el("smoothingWindow").closest("label").hidden=!trailingMean' in html
    assert 'const alphaError=firstOrderAlphaError();' in preview_handler
    assert 'el("preprocessingPreview").className="status error"' in preview_handler
    assert 'el("preprocessingPreview").textContent=alphaError' in preview_handler
    assert "tags:[]" in preview_handler
    assert preview_handler.index('const alphaError=firstOrderAlphaError();') < preview_handler.index(
        'api("/api/preprocessing-preview"'
    )


def test_web_preprocessing_preview_uses_cached_single_tag_svg_comparison():
    html = web.INDEX_HTML
    tag_source = html.split("function preprocessingPreviewTags(data)", 1)[1].split(
        "function renderPreprocessingPreview()", 1
    )[0]
    preview_source = html.split("function renderPreprocessingPreview()", 1)[1].split(
        'el("trendZoom")', 1
    )[0]

    assert 'id="preprocessingPreviewTagSelect"' in preview_source
    assert "preprocessingPreview:null, preprocessingPreviewTag:null" in html
    assert "state.preprocessingPreview=null; state.preprocessingPreviewTag=null" in html
    assert "preprocessingPreviewSvg(data,tag)" in preview_source
    assert "function preprocessingPreviewTags(data)" in html
    assert "data.preview_tags" in tag_source
    assert "Object.keys(row)" not in tag_source
    assert ".slice(0,5)" not in tag_source
    assert "Date.parse(row.timestamp)" in preview_source
    assert "row.physical_gap_start||!valid" in preview_source
    assert "Number.isFinite(value)" in preview_source
    assert "查看抽样数据明细" not in preview_source
    assert "preprocessing-preview-details" not in html
    assert 'const tables=["raw","resampled","filtered"]' not in preview_source
    assert "if(!tags.includes(state.preprocessingPreviewTag)) state.preprocessingPreviewTag=tags[0]" in html
    assert "state.preprocessingPreviewTag=event.target.value; renderPreprocessingPreview();" in preview_source
    assert "/api/preprocessing-preview" not in preview_source
    assert "function preprocessingPreviewStages(data)" in html
    assert 'if(summary.resampling_method!=="none")' in preview_source
    assert 'if(summary.filter_method!=="none")' in preview_source
    assert "一阶低通滤波后数据" in preview_source
    assert "移动平均后数据" in preview_source
    assert "滤波后数据" in preview_source
    assert 'summary.resampling_method!=="none"?"滤波后数据"' in preview_source
    assert "const stages=preprocessingPreviewStages(data)" in preview_source
    assert "Lag" not in html.split("function preprocessingPreviewSvg", 1)[1].split('el("trendZoom")', 1)[0]
    assert "重采样未启用" not in preview_source
    assert "滤波未启用" not in preview_source
    assert "查看高噪声代表 Tag:" in preview_source
    assert "当前显示自动筛选的高噪声代表变量，用于评估预处理效果。" in preview_source
    preview_select_style = web_model_results.INDEX_HTML.split(
        ".preprocessing-preview-area #preprocessingPreviewTagSelect", 1
    )[1].split("}", 1)[0]
    assert "width:300px" in preview_select_style
    assert "min-width:0" in preview_select_style
    assert "max-width:100%" in preview_select_style
    assert "min-width:250px" not in preview_select_style


def test_web_preprocessing_preview_svg_rejects_nulls_and_uses_real_y_span():
    svg_source = web.INDEX_HTML.split("function preprocessingPreviewSvg", 1)[1].split(
        'el("trendZoom")', 1
    )[0]

    assert "if(value===null||value===undefined" in svg_source
    assert "const converted=Number(value)" in svg_source
    assert "return value===null?[]:[value]" in svg_source
    assert "value=numericValue(row[tag])" in svg_source
    assert "row.physical_gap_start||!valid" in svg_source
    assert "Math.max(1,maximum-minimum)" not in svg_source
    assert "hasUsableSpan?dataSpan" in svg_source
    assert "Number.EPSILON" in svg_source
    assert "yMinimum=hasUsableSpan?minimum:minimum-ySpan/2" in svg_source
    assert "y(minimum).toFixed(1)" in svg_source


def test_web_preprocessing_config_defaults_to_none():
    config = web._preprocessing_config({})

    assert config.filter_method == "none"
    assert config.first_order_alpha is None


def test_preprocessing_preview_uses_unified_core_and_preserves_empty_bins(
    tmp_path, monkeypatch
) -> None:
    from pca_model_builder import data_session
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    timestamps = pd.to_datetime(
        [*[f"2026-01-01 00:0{i}" for i in range(5)], "2026-01-01 00:11"]
    )
    frame = pd.DataFrame(
        {"time": timestamps, "A": np.arange(6, dtype=float), "B": np.arange(10, 16, dtype=float)}
    )
    uploaded = web.save_upload(
        "preview.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )
    web.inspect_payload(
        {"file_id": uploaded["file_id"], "timestamp_column": "time"}
    )
    original = data_session.pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(data_session.pd, "read_csv", recorded)

    result = web.preprocessing_preview_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B"],
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
            "sample_interval_minutes": 5,
            "resampling_method": "last",
            "filter_method": "none",
            "smoothing_window_minutes": 10,
            "gap_threshold_minutes": 10,
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
        }
    )

    assert result["summary"]["source_row_count"] == 6
    assert result["summary"]["resampled_row_count"] == 4
    assert result["summary"]["empty_bin_count"] == 1
    assert result["preview_tags"] == ["A", "B"]
    empty = next(row for row in result["resampled"] if row["timestamp"].endswith("00:10:00"))
    assert empty["A"] is None and empty["B"] is None
    assert result["data_usage"]["analysis_row_count"] == 6
    assert result["data_usage"]["cache_hit"]
    assert calls == []


def test_regular_cluster_loads_and_applies_state_filter_column(
    tmp_path, monkeypatch
) -> None:
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    history = _history_frame()
    history["LOAD"] = [1] * 120 + [0] * 60
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "analysis_start": history.time.iloc[0].isoformat(),
        "analysis_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 5,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "filter_method": "none",
        "state_filters": [{"column": "LOAD", "minimum": 1}],
        "tag_configs": {"LOAD": {"role": "state_filter"}},
        "n_clusters": 2,
    }

    result = web.cluster_payload(payload)

    assert result["sample_count"] == 120
    assert result["data_usage"]["loaded_column_count"] == 5
    with pytest.raises(ValueError, match="state_filter角色"):
        web.cluster_payload({**payload, "tag_configs": {}})
    with pytest.raises(ValueError, match="找不到 Tag：MISSING"):
        web.cluster_payload({**payload, "state_filters": [{"column": "MISSING", "minimum": 1}]})


def test_web_quality_reference_statistics_use_state_filtered_rows(
    tmp_path, monkeypatch
) -> None:
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    history = _history_frame()
    history["LOAD"] = [1] * 60 + [0] * 120
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    result = web.quality_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "normal_start": history.time.iloc[0].isoformat(),
            "normal_end": history.time.iloc[-1].isoformat(),
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 0,
            "filter_method": "none",
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
            "state_filters": [{"column": "LOAD", "minimum": 1}],
        }
    )

    assert result["data_usage"]["analysis_row_count"] == 60
    assert all(item["reference"]["sample_count"] == 60 for item in result["tags"])


def test_repeated_final_web_trend_reuses_only_requested_columns(
    tmp_path, monkeypatch
) -> None:
    from pca_model_builder import data_session, web_dataproject
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    common = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "encoding": "utf-8-sig",
        "tags": ["A", "B"],
        "purpose": "trend",
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "start": history.time.iloc[0].isoformat(),
        "end": history.time.iloc[-1].isoformat(),
        "max_points": 100,
    }
    web.inspect_payload(common)
    original = data_session.pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(data_session.pd, "read_csv", recorded)
    first = web_dataproject.trend_payload(common)
    second = web_dataproject.trend_payload({**common, "tags": ["B", "A"]})

    assert calls == []
    assert first["data_usage"]["loaded_column_count"] == 3
    assert first["data_usage"]["cache_hit"]
    assert second["data_usage"]["cache_hit"]
    assert second["tags"] == ["B", "A"]


def test_web_error_responses_report_processing_stage_and_keep_general_errors(
    tmp_path, monkeypatch
) -> None:
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    loading_error, _ = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": "0" * 32, "timestamp_column": "time"},
    )
    assert loading_error == {
        "error": "上传文件不存在，请重新上传",
        "stage": "loading",
    }
    invalid = web.save_upload("invalid.csv", b"time,A,B\ninvalid,1,2\n")
    parsing_error, _ = _post_response(
        web._Handler,
        "/api/inspect",
        {"file_id": invalid["file_id"], "timestamp_column": "time"},
    )
    assert parsing_error == {
        "error": "时间列包含无法解析的值",
        "stage": "parsing",
    }
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    training = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[119].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "model_name": "STAGE_TEST",
    }
    original_builder = web.build_training_matrix
    original_fit = web.fit_dpca

    def fail(message: str):
        def failing(*args, **kwargs):
            raise ValueError(message)

        return failing

    monkeypatch.setattr(
        web, "build_training_matrix", fail("数据质量问题尚未处理：missing_value")
    )
    quality_error, status = _post_response(web._Handler, "/api/quality", training)
    assert status == 400
    assert quality_error == {
        "error": "数据质量问题尚未处理：missing_value",
        "stage": "quality_check",
    }

    monkeypatch.setattr(web, "build_training_matrix", fail("preprocess failed"))
    preprocessing_error, _ = _post_response(web._Handler, "/api/train", training)
    assert preprocessing_error == {
        "error": "preprocess failed",
        "stage": "preprocessing",
    }

    monkeypatch.setattr(web, "build_training_matrix", original_builder)
    monkeypatch.setattr(web, "fit_dpca", fail("fit failed"))
    fitting_error, _ = _post_response(web._Handler, "/api/train", training)
    assert fitting_error == {"error": "fit failed", "stage": "fitting"}

    monkeypatch.setattr(web, "fit_dpca", original_fit)
    trained = web.train_payload(training)
    monkeypatch.setattr(web, "validate_model_windows", fail("score failed"))
    scoring_error, _ = _post_response(
        web._Handler,
        "/api/validate",
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "validation_start": history.time.iloc[120].isoformat(),
            "validation_end": history.time.iloc[-1].isoformat(),
        },
    )
    assert scoring_error == {"error": "score failed", "stage": "scoring"}

    general_error, _ = _post_response(web._Handler, "/api/quality", {})
    assert general_error == {
        "error": "缺少参数：timestamp_column",
        "stage": "failed",
    }


def test_final_web_handler_preserves_stage_error(monkeypatch) -> None:
    def failing(payload):
        raise web.WebStageError("fitting", ValueError("fit failed"))

    monkeypatch.setattr(web_model_results, "train_payload", failing)
    result, status = _post_response(
        web_model_results.ModelResultsHandler, "/api/train", {}
    )

    assert status == 400
    assert result == {"error": "fit failed", "stage": "fitting"}


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
    assert "function explorationPerformanceTag()" in html
    assert "tag!==performanceTag" in render_source
    assert "state.selectedModelTags.delete(performanceTag)" in html
    assert 'el("explorationPerformanceTag").addEventListener("change",syncExplorationPerformanceSelection)' in html


def test_uploaded_tag_names_are_not_ellipsis_clipped_before_inspection():
    html = web.INDEX_HTML
    upload_source = html.split("function renderUploadedColumns(columns)", 1)[1].split(
        "function renderBasicInspection(data)", 1
    )[0]

    assert ".tag-options label span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }" in html
    assert ".tag-options span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }" not in html
    assert 'row.className="tag-row"' in upload_source
    assert 'badge.textContent="待检查"' in upload_source


def test_tag_list_reserves_name_and_quality_columns():
    html = web.INDEX_HTML
    render_source = html.split("function renderTagList()", 1)[1].split(
        "function selectTag", 1
    )[0]
    upload_source = html.split("function renderUploadedColumns(columns)", 1)[1].split(
        "function renderBasicInspection(data)", 1
    )[0]

    assert ".tag-row:not(.pending) > .tag-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }" in html
    assert ".tag-row > .tag-state { min-width:max-content; white-space:nowrap; }" in html
    assert 'name.className="tag-name"' in render_source
    assert "name.title=tag" in render_source
    assert 'row.classList.add("pending")' in upload_source
    assert "name.title=column" in upload_source
    assert "#tagOptions .tag-row.pending" in web_model_results.INDEX_HTML
    assert "height:auto" in web_model_results.INDEX_HTML


def test_web_unifies_confirmed_exclusions_and_keeps_raw_suggestions_manual():
    html = web.INDEX_HTML
    inspect_source = html.split(
        'el("inspectButton").addEventListener("click", async () => {', 1
    )[1].split('el("tagSearch")', 1)[0]
    basic_source = html.split("function renderBasicInspection(data)", 1)[1].split(
        "function addPerformanceCondition", 1
    )[0]

    assert "function setTagExclusion(tag, record)" in html
    assert "function reconcileExcludedTags()" in html
    assert 'reason:"manual_exclude"' in html
    assert 'reason:"constant_in_reference_window"' in html
    assert "confirmSuggestedExclusion(profile)" in html
    assert 'previousRole==="exclude"&&config.role==="continuous_input"' in html
    assert 'confirm.textContent="确认排除"' in basic_source
    assert "state.inspection.numeric_columns.includes(profile.tag)" in basic_source
    assert "previousExcludedTags=state.excludedTags" in inspect_source
    assert "state.excludedTags=previousExcludedTags; reconcileExcludedTags();" in inspect_source
    assert "state.excludedTags=[]" not in inspect_source
    assert "constants.forEach(item=>excludeConstantTag(item,false));" in html


def test_train_payload_keeps_constant_exclusion_metadata_schema():
    html = web.INDEX_HTML
    train_source = html.split("async function trainModel(modelPurpose)", 1)[1].split(
        'el("trainExploratoryButton")', 1
    )[0]

    assert 'record.reason==="constant_in_reference_window"' in train_source
    assert 'state.registry[record.tag]?.role==="exclude"' in train_source
    assert "excluded_tags:excludedTags" in train_source


def test_web_quality_tab_shows_selected_tag_and_trend_axis_uses_payload_limits():
    html = web.INDEX_HTML

    assert "function renderCurrentTagQuality()" in html
    assert "尚未执行建模质量检查。" in html
    assert "已失效" not in html
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


def test_web_quality_tag_selector_uses_quality_results_and_rerenders_details():
    html = web.INDEX_HTML

    assert 'id="qualityTagSelect"' in html
    assert "state.quality.tags" in html
    assert "state.selectedTag = value;\n  renderCurrentTagQuality();" in html
    assert 'data.tags.filter(item=>item.status!=="usable")' in html
    assert 'metric("可直接使用",data.summary.usable)' in html
    assert "function renderCurrentTagQuality()" in html


def test_final_web_quality_tag_selector_survives_four_column_renderer_override():
    html = web_model_results.INDEX_HTML
    renderer = html.split(
        "window.renderCurrentTagQuality = function renderCurrentTagQualityFourColumns()",
        1,
    )[1].split("  };", 1)[0]

    assert 'id="qualityTagSelect"' in html
    assert "state.quality.tags" in renderer
    assert "select.disabled = !filteredTags.length;" in renderer
    assert "state.selectedTag = filteredTags[0].tag;" in renderer
    assert "select.value = state.selectedTag;" in renderer
    assert 'class="quality-profile-grid"' in renderer
    assert "没有可查看的建模 Tag。" in renderer
    assert "state.selectedTag = value;\n  renderCurrentTagQuality();" in html


def test_final_web_quality_tag_filter_and_compact_profile_tables():
    html = web_model_results.INDEX_HTML
    renderer = html.split('id="qualityProfileGridScript">', 1)[1].split(
        "</script>", 1
    )[0]

    assert 'filter.id = "qualityTagFilter";' in renderer
    assert "state.quality.tags" in renderer
    assert "String(item.tag).toLowerCase().includes" in renderer
    assert 'filter.addEventListener("input", () => window.renderCurrentTagQuality());' in renderer
    assert "filter.value = \"\";" in renderer
    assert "filteredTags.some((item) => item.tag === state.selectedTag)" in renderer
    assert 'option.textContent = "无匹配 Tag";' in renderer
    assert "function qualityProfileTable(title, profile)" in renderer
    assert "for (let index = 0; index < fields.length; index += 2)" in renderer
    assert '"<th></th><td></td>"' in renderer
    assert 'class="quality-profile-table"' in renderer
    assert ".quality-profile-table th:nth-child(odd)" in html
    assert ".quality-profile-table td:nth-child(even)" in html


def test_web_service_trains_and_validates_uploaded_csv(tmp_path, monkeypatch):
    from pca_model_builder import data_session

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    csv_bytes = history.to_csv(index=False).encode("utf-8-sig")

    uploaded = web.save_upload("history.csv", csv_bytes)
    original_read_csv = data_session.pd.read_csv
    csv_reads = []

    def recorded_read_csv(*args, **kwargs):
        csv_reads.append(kwargs.copy())
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(data_session.pd, "read_csv", recorded_read_csv)
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
            "filter_method": "trailing_mean",
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
            "filter_method": "trailing_mean",
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
    assert inspected["data_usage"] == {
        "source_row_count": 180,
        "analysis_row_count": 180,
        "display_point_count": 180,
        "loaded_column_count": 5,
        "cache_hit": False,
        "stage": "completed",
    }
    assert clustered["engineer_decision_required"] is True
    assert clustered["sample_count"] == 177
    assert len(clustered["clusters"]) == 2
    assert {point["cluster"] for point in clustered["points"]} == {1, 2}
    assert clustered["data_usage"]["source_row_count"] == 180
    assert clustered["data_usage"]["analysis_row_count"] == 180
    assert clustered["data_usage"]["display_point_count"] == len(
        clustered["points"]
    )
    assert clustered["data_usage"]["loaded_column_count"] == 4
    assert screened["engineer_decision_required"] is True
    assert 0 < screened["matched_rows"] < screened["total_rows"]
    assert screened["representative_windows"]
    assert screened["data_usage"]["loaded_column_count"] == 3
    assert trained["model_purpose"] == "normal_state"
    assert trained["model_status"] == "candidate"
    assert trained["n_components"] >= 2
    assert {"pc1", "pc2"}.issubset(trained["scores"][0])
    assert trained["training_rows"] > 0
    assert trained["data_usage"]["analysis_row_count"] == 120
    assert trained["data_usage"]["display_point_count"] == len(trained["scores"])
    assert trained["model_download"].endswith(trained["run_id"])
    assert (tmp_path / "runs" / trained["run_id"] / "model.pcamodel").exists()
    loaded_model, manifest = load_model_package(
        tmp_path / "runs" / trained["run_id"] / "model.pcamodel"
    )
    assert manifest["config"]["tag_configs"]["A"]["description"] == "A变量"
    normal = history.iloc[:120].set_index("time")[["A", "B", "C"]]
    preprocessing = PreprocessingConfig(5, 10, 10, 5, filter_method="trailing_mean")
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
    assert validated["data_usage"]["source_row_count"] == 180
    assert validated["data_usage"]["analysis_row_count"] == validated["scored_rows"]
    assert validated["data_usage"]["loaded_column_count"] == 5
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
    assert len(csv_reads) == 1
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
    history.loc[60, "A"] = 1000.0
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
        "filter_method": "trailing_mean",
        "max_lag_minutes": 10,
        "lag_step_minutes": 5,
        "model_name": "SEMANTICS_DPCA",
        "tag_configs": {"A": {"engineering_min": -10.0, "engineering_max": 10.0}},
    }
    exploratory = web.train_payload(
        {**training, "model_purpose": "exploratory"}
    )
    candidate = web.train_payload(training)

    assert exploratory["model_purpose"] == "exploratory"
    assert exploratory["model_status"] == "draft"
    assert candidate["model_purpose"] == "normal_state"
    assert candidate["model_status"] == "candidate"
    assert exploratory["training_window_summary"][0]["engineering_range_loss"] == 0
    assert candidate["training_window_summary"][0]["engineering_range_loss"] == 1
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
    config = PreprocessingConfig(5, 10, 10, 5, filter_method="trailing_mean")
    dynamic = build_dynamic_matrix(analysis, ["A", "B", "C"], config, infer_segment_ids(analysis.index, 5))
    expected_scores = model.score(dynamic)
    points = pd.DataFrame(clustered["points"])
    points.index = pd.to_datetime(points.pop("timestamp"))

    assert clustered["exploratory_run_id"] == exploratory["run_id"]
    np.testing.assert_allclose(points["pc1"], expected_scores["pc1"])
    np.testing.assert_allclose(points["pc2"], expected_scores["pc2"])


def test_state_exploration_api_reads_summary_and_bounded_series(tmp_path, monkeypatch):
    from pca_model_builder.data_session import DataSessionCache

    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "DATA_SESSIONS", DataSessionCache())
    web.clear_state_exploration_cache()
    history = _history_frame()
    history["PERF"] = np.r_[np.full(90, 5.0), np.full(90, 1.0)]
    uploaded = web.save_upload(
        "exploration.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "exploration_start": history.time.iloc[0].isoformat(),
        "exploration_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 0,
        "filter_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "exploration_config": {
            "cluster_count": 2,
            "random_state": 7,
            "minimum_candidate_duration_minutes": 10,
            "candidate_count_per_cluster": 1,
            "maximum_plot_points": 12,
        },
        "performance_config": {
            "performance_tag": "PERF",
            "direction": "higher_is_better",
            "minimum_duration_minutes": 10,
            "candidate_count": 1,
        },
    }

    result = web.state_exploration_payload(payload)
    run_id = result["exploration_run_id"]
    assert result["full_point_count"] >= len(result["cluster_series"])
    assert result["returned_point_count"] <= 12
    assert result["data_usage"]["loaded_column_count"] == 5
    assert result["performance_candidates"]
    assert result["preprocessing_summary"]["dynamic_feature_count"] == 3
    assert all(
        "performance_target_met" not in row for row in result["cluster_series"]
    )
    assert "cluster_series" not in _get_response(
        web._Handler, f"/api/state-exploration/{run_id}"
    )[0]

    summary, summary_status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}"
    )
    series, series_status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}/series?max_points=5"
    )
    invalid, invalid_status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}/series?max_points=1"
    )
    missing, missing_status = _get_response(
        web._Handler, f"/api/state-exploration/{'f' * 32}"
    )
    malformed, malformed_status = _get_response(
        web._Handler, "/api/state-exploration/not-a-run"
    )

    assert summary_status == 200
    assert summary["exploration_run_id"] == run_id
    assert summary["data_usage"]["loaded_column_count"] == 5
    assert series_status == 200
    assert series["full_point_count"] == result["full_point_count"]
    assert series["returned_point_count"] <= 5
    assert all("segment_id" in row and "break_before" in row for row in series["cluster_series"])
    assert invalid_status == 400 and "traceback" not in invalid["error"].lower()
    assert missing_status == 404
    assert malformed_status == 400

    invalid_config, invalid_config_status = _post_response(
        web._Handler,
        "/api/state-exploration/run",
        {
            **payload,
            "performance_config": {
                "performance_tag": "PERF",
                "direction": "target_range",
            },
        },
    )
    missing_tag, missing_tag_status = _post_response(
        web._Handler,
        "/api/state-exploration/run",
        {
            **payload,
            "performance_config": {
                "performance_tag": "MISSING_PERF",
                "direction": "higher_is_better",
            },
        },
    )
    assert invalid_config_status == 400
    assert "traceback" not in invalid_config["error"].lower()
    assert missing_tag_status == 400
    assert "MISSING_PERF" in missing_tag["error"]

    with pytest.raises(ValueError, match="性能 Tag.*PCA"):
        web.state_exploration_payload({**payload, "tags": ["A", "B", "PERF"]})


def test_state_exploration_uses_model_tag_engineering_ranges_only(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = _history_frame()
    history["PERF"] = np.linspace(0.0, 1.0, len(history))
    history["LOAD"] = 1.0
    uploaded = web.save_upload(
        "exploration.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "exploration_start": history.time.iloc[0].isoformat(),
        "exploration_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 0,
        "filter_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "state_filters": [{"column": "LOAD", "minimum": 1}],
        "performance_config": {
            "performance_tag": "PERF",
            "direction": "higher_is_better",
            "minimum_duration_minutes": 10,
            "candidate_count": 1,
        },
        "exploration_config": {"cluster_count": 2},
        "tag_configs": {
            "A": {"engineering_min": -0.1, "engineering_max": 0.1},
            "LOAD": {"role": "state_filter"},
        },
    }

    result = web.state_exploration_payload(payload)

    assert result["preprocessing_summary"]["dynamic_feature_count"] == 3


def test_state_exploration_drops_partial_resampling_boundaries_like_training(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    time = pd.date_range("2026-01-01", periods=65, freq="1min")
    values = np.linspace(0.0, 1.0, len(time))
    frame = pd.DataFrame(
        {"time": time, "A": values, "B": values**2, "C": np.sin(values)}
    )
    uploaded = web.save_upload("resampled.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    start, end = time[2], time[58]
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "exploration_start": start.isoformat(),
        "exploration_end": end.isoformat(),
        "sample_interval_minutes": 5,
        "resampling_method": "mean",
        "smoothing_window_minutes": 0,
        "filter_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "exploration_config": {"cluster_count": 2},
    }

    explored = web.state_exploration_payload(payload)
    cached = web._state_exploration_run(explored["exploration_run_id"])
    training = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(
            5, 0, 0, 5, resampling_method="mean", filter_method="none"
        ),
        [
            {
                "id": "window-001",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "source": "manual",
                "source_ref": None,
                "enabled": True,
                "comment": "",
            }
        ],
    )

    assert explored["preprocessing_summary"]["partial_resampling_bin_loss"] == 2
    assert explored["preprocessing_summary"]["final_dynamic_row_count"] == len(
        cached["cluster_series"]
    )
    assert cached["cluster_series"].index.tolist() == training.dynamic.index.tolist()


def test_state_exploration_cache_evicts_oldest_full_result(monkeypatch):
    web.clear_state_exploration_cache()
    monkeypatch.setattr(web, "MAX_STATE_EXPLORATION_RUNS", 1)
    first = {"cluster_series": pd.DataFrame(), "cluster_series_display": pd.DataFrame()}
    second = {"cluster_series": pd.DataFrame(), "cluster_series_display": pd.DataFrame()}

    web._store_state_exploration_run("a" * 32, first)
    web._store_state_exploration_run("b" * 32, second)

    assert list(web.STATE_EXPLORATION_RUNS) == ["b" * 32]
    assert first == {}


def test_state_exploration_candidate_decisions_and_window_conversion():
    web.clear_state_exploration_cache()
    run_id = "a" * 32
    cluster_id = "cluster_001-candidate-001"
    performance_id = "performance-candidate-001"
    web._store_state_exploration_run(
        run_id,
        {
            "cluster_candidates": [
                {
                    "candidate_id": cluster_id,
                    "source": "cluster",
                    "start": "2026-01-01T00:00:00",
                    "end": "2026-01-01T00:10:00",
                }
            ],
            "performance_candidates": [
                {
                    "candidate_id": performance_id,
                    "source": "performance",
                    "start": "2026-01-01T00:20:00",
                    "end": "2026-01-01T00:30:00",
                }
            ],
            "candidate_decisions": [
                {
                    "candidate_id": cluster_id,
                    "decision": "pending",
                    "comment": "",
                    "decided_at": None,
                },
                {
                    "candidate_id": performance_id,
                    "decision": "pending",
                    "comment": "",
                    "decided_at": None,
                },
            ],
            "cluster_series": pd.DataFrame(),
            "cluster_series_display": pd.DataFrame(),
        },
    )

    decisions, status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/decisions",
        {
            "decisions": [
                {
                    "candidate_id": cluster_id,
                    "decision": "accepted",
                    "comment": "Cluster稳定",
                },
                {
                    "candidate_id": performance_id,
                    "decision": "rejected",
                    "comment": "性能不足",
                },
            ]
        },
    )

    assert status == 200
    assert {item["decision"] for item in decisions["candidate_decisions"]} == {
        "accepted",
        "rejected",
    }
    assert all(item["decided_at"] for item in decisions["candidate_decisions"])
    summary, summary_status = _get_response(
        web._Handler, f"/api/state-exploration/{run_id}"
    )
    assert summary_status == 200
    assert summary["candidate_decisions"] == decisions["candidate_decisions"]

    rejected, rejected_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/training-windows",
        {"candidate_ids": [performance_id], "training_windows": []},
    )
    assert rejected_status == 400
    assert "只有已接受候选" in rejected["error"]

    converted, converted_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/training-windows",
        {"candidate_ids": [cluster_id], "training_windows": []},
    )
    window = converted["training_windows"][0]
    assert converted_status == 200
    assert converted["converted_candidate_ids"] == [cluster_id]
    assert window == {
        "id": f"state-exploration-{run_id}-{cluster_id}",
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-01T00:10:00",
        "source": "cluster",
        "source_ref": cluster_id,
        "enabled": False,
        "comment": "Cluster稳定",
    }
    repeated, repeated_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/training-windows",
        {"candidate_ids": [cluster_id], "training_windows": converted["training_windows"]},
    )
    assert repeated_status == 200
    assert repeated["training_windows"] == converted["training_windows"]
    assert repeated["converted_candidate_ids"] == []

    accepted_performance, accepted_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/decisions",
        {
            "candidate_id": performance_id,
            "decision": "accepted",
            "comment": "性能优秀",
        },
    )
    assert accepted_status == 200
    converted_performance, converted_performance_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{run_id}/training-windows",
        {
            "candidate_ids": [cluster_id, performance_id],
            "training_windows": converted["training_windows"],
        },
    )
    assert converted_performance_status == 200
    assert converted_performance["converted_candidate_ids"] == [performance_id]
    assert converted_performance["training_windows"][1]["source"] == "performance"
    assert converted_performance["training_windows"][1]["source_ref"] == performance_id
    assert converted_performance["training_windows"][1]["enabled"] is False
    assert converted_performance["training_windows"][1]["comment"] == "性能优秀"
    assert accepted_performance["candidate_decisions"][1]["decision"] == "accepted"

    for payload in (
        {
            "decisions": [
                {"candidate_id": cluster_id, "decision": "pending", "comment": ""},
                {"candidate_id": cluster_id, "decision": "accepted", "comment": ""},
            ]
        },
        {"candidate_id": "unknown", "decision": "accepted", "comment": ""},
        {"candidate_id": cluster_id, "decision": "invalid", "comment": ""},
        {"candidate_id": cluster_id, "decision": "accepted", "comment": None},
    ):
        error, error_status = _post_response(
            web._Handler, f"/api/state-exploration/{run_id}/decisions", payload
        )
        assert error_status == 400
        assert "traceback" not in error["error"].lower()
    missing, missing_status = _post_response(
        web._Handler,
        f"/api/state-exploration/{'f' * 32}/decisions",
        {"candidate_id": cluster_id, "decision": "accepted", "comment": ""},
    )
    assert missing_status == 404
    assert "traceback" not in missing["error"].lower()


def test_state_exploration_conversion_keeps_same_candidate_from_separate_runs():
    web.clear_state_exploration_cache()
    candidate_id = "cluster_001-candidate-001"
    first_run_id = "a" * 32
    second_run_id = "b" * 32
    for run_id, start, end, comment in (
        (first_run_id, "2026-01-01T00:00:00", "2026-01-01T00:10:00", "第一轮"),
        (second_run_id, "2026-01-02T00:00:00", "2026-01-02T00:20:00", "第二轮"),
    ):
        web._store_state_exploration_run(
            run_id,
            {
                "cluster_candidates": [
                    {
                        "candidate_id": candidate_id,
                        "source": "cluster",
                        "start": start,
                        "end": end,
                    }
                ],
                "performance_candidates": [],
                "candidate_decisions": [
                    {
                        "candidate_id": candidate_id,
                        "decision": "accepted",
                        "comment": comment,
                        "decided_at": None,
                    }
                ],
                "cluster_series": pd.DataFrame(),
                "cluster_series_display": pd.DataFrame(),
            },
        )

    first = web.state_exploration_training_windows_payload(
        first_run_id, {"candidate_ids": [candidate_id], "training_windows": []}
    )
    second = web.state_exploration_training_windows_payload(
        second_run_id,
        {"candidate_ids": [candidate_id], "training_windows": first["training_windows"]},
    )
    repeated = web.state_exploration_training_windows_payload(
        first_run_id,
        {"candidate_ids": [candidate_id], "training_windows": second["training_windows"]},
    )

    assert [window["id"] for window in second["training_windows"]] == [
        f"state-exploration-{first_run_id}-{candidate_id}",
        f"state-exploration-{second_run_id}-{candidate_id}",
    ]
    assert [window["source_ref"] for window in second["training_windows"]] == [candidate_id] * 2
    assert [(window["start"], window["end"], window["comment"]) for window in second["training_windows"]] == [
        ("2026-01-01T00:00:00", "2026-01-01T00:10:00", "第一轮"),
        ("2026-01-02T00:00:00", "2026-01-02T00:20:00", "第二轮"),
    ]
    assert all(window["enabled"] is False for window in second["training_windows"])
    assert first["converted_candidate_ids"] == [candidate_id]
    assert second["converted_candidate_ids"] == [candidate_id]
    assert repeated["training_windows"] == second["training_windows"]
    assert repeated["converted_candidate_ids"] == []


def test_state_exploration_conversion_adds_only_to_candidate_windows_in_web():
    assert "state.candidateWindows.some(window=>window.source_ref===candidateRef)" in web.INDEX_HTML
    assert "请在候选窗口列表确认作为训练窗口。" in web.INDEX_HTML
    conversion_source = web.INDEX_HTML.split(
        'el("convertExplorationCandidates").addEventListener', 1
    )[1].split('el("trainExploratoryButton")', 1)[0]
    assert "/training-windows" not in conversion_source


def test_cluster_representative_windows_use_distinct_physical_source_refs_in_web():
    cluster_source = web.INDEX_HTML.split("function renderClustering(data)", 1)[1].split(
        "function modelLifecycle", 1
    )[0]
    assert (
        'addCandidateWindow("cluster",window.start,window.end,'
        '`cluster-${item.cluster}-${window.start}-${window.end}`,"")'
    ) in cluster_source
    assert "state.candidateWindows.some(window=>window.source_ref===sourceRef)" in web.INDEX_HTML

    windows = [
        {"start": "2026-01-01T00:00:00", "end": "2026-01-01T00:10:00"},
        {"start": "2026-01-01T01:00:00", "end": "2026-01-01T01:10:00"},
        {"start": "2026-01-01T02:00:00", "end": "2026-01-01T02:10:00"},
    ]
    source_refs = [
        f"cluster-7-{window['start']}-{window['end']}" for window in windows
    ]
    candidates: list[dict[str, str]] = []

    def add_candidate(source_ref: str) -> bool:
        if any(window["source_ref"] == source_ref for window in candidates):
            return False
        candidates.append({"source_ref": source_ref})
        return True

    assert len(set(source_refs)) == 3
    assert [add_candidate(source_ref) for source_ref in source_refs] == [True] * 3
    assert not add_candidate(source_refs[0])
    assert source_refs[0] != f"cluster-8-{windows[0]['start']}-{windows[0]['end']}"


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


def test_candidate_confirmation_splits_training_windows_by_exclusions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    timestamps = pd.date_range(
        "2026-01-01T08:00:00", "2026-01-01T20:00:00", freq="5min"
    )
    history = pd.DataFrame(
        {
            "time": timestamps,
            "A": np.arange(len(timestamps), dtype=float),
            "B": np.arange(len(timestamps), dtype=float) + 1,
            "C": np.arange(len(timestamps), dtype=float) + 2,
        }
    )
    uploaded = web.save_upload(
        "candidate.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    candidate = {
        "id": "candidate-001",
        "start": "2026-01-01T08:00:00",
        "end": "2026-01-01T20:00:00",
        "source": "trend",
        "source_ref": "trend-current",
        "comment": "工程师确认",
    }
    excluded = [
        {
            "id": "exclude-1",
            "start": "2026-01-01T10:00:00",
            "end": "2026-01-01T11:00:00",
            "source": "trend",
            "comment": "波动",
        },
        {
            "id": "exclude-2",
            "start": "2026-01-01T16:00:00",
            "end": "2026-01-01T17:00:00",
            "source": "trend",
            "comment": "检修",
        },
    ]

    result = web.training_windows_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "training_windows": [],
            "operation": {
                "action": "confirm_candidate",
                "candidate": candidate,
                "excluded_windows": excluded,
            },
        }
    )

    assert [window["id"] for window in result["training_windows"]] == [
        "training-candidate-001-part-001",
        "training-candidate-001-part-002",
        "training-candidate-001-part-003",
    ]
    assert [(window["start"], window["end"]) for window in result["training_windows"]] == [
        ("2026-01-01T08:00:00", "2026-01-01T09:55:00"),
        ("2026-01-01T11:05:00", "2026-01-01T15:55:00"),
        ("2026-01-01T17:05:00", "2026-01-01T20:00:00"),
    ]
    assert all(window["enabled"] for window in result["training_windows"])
    excluded_points = set(
        pd.to_datetime(
            [
                "2026-01-01T10:00:00",
                "2026-01-01T11:00:00",
                "2026-01-01T16:00:00",
                "2026-01-01T17:00:00",
            ]
        )
    )
    selected = set()
    for window in result["training_windows"]:
        selected.update(
            timestamps[
                (timestamps >= pd.Timestamp(window["start"]))
                & (timestamps <= pd.Timestamp(window["end"]))
            ]
        )
    assert excluded_points.isdisjoint(selected)
    removed = web.training_windows_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "training_windows": result["training_windows"],
            "operation": {
                "action": "remove",
                "id": "training-candidate-001-part-001",
            },
        }
    )
    for window in list(removed["training_windows"]):
        removed = web.training_windows_payload(
            {
                "file_id": uploaded["file_id"],
                "timestamp_column": "time",
                "training_windows": removed["training_windows"],
                "operation": {"action": "remove", "id": window["id"]},
            }
        )
    reconfirmed = web.training_windows_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "training_windows": removed["training_windows"],
            "operation": {
                "action": "confirm_candidate",
                "candidate": candidate,
                "excluded_windows": excluded,
            },
        }
    )
    assert [window["id"] for window in reconfirmed["training_windows"]] == [
        "training-candidate-001-part-001",
        "training-candidate-001-part-002",
        "training-candidate-001-part-003",
    ]


def test_candidate_confirmation_keeps_legacy_id_without_intersecting_exclusions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    timestamps = pd.date_range(
        "2026-01-01T08:00:00", "2026-01-01T20:00:00", freq="5min"
    )
    history = pd.DataFrame(
        {"time": timestamps, "A": range(len(timestamps)), "B": range(len(timestamps))}
    )
    uploaded = web.save_upload(
        "candidate.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    candidate = {
        "id": "candidate-001",
        "start": "2026-01-01T08:00:00",
        "end": "2026-01-01T20:00:00",
        "source": "manual",
        "source_ref": None,
        "comment": "工程师确认",
    }
    outside = {
        "id": "exclude-outside",
        "start": "2026-01-01T21:00:00",
        "end": "2026-01-01T22:00:00",
        "source": "trend",
        "comment": "无关",
    }

    result = web.training_windows_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "training_windows": [],
            "operation": {
                "action": "confirm_candidate",
                "candidate": candidate,
                "excluded_windows": [outside],
            },
        }
    )

    assert [window["id"] for window in result["training_windows"]] == [
        "training-candidate-001"
    ]
    with pytest.raises(ValueError, match="完全覆盖"):
        web.training_windows_payload(
            {
                "file_id": uploaded["file_id"],
                "timestamp_column": "time",
                "training_windows": [],
                "operation": {
                    "action": "confirm_candidate",
                    "candidate": candidate,
                    "excluded_windows": [
                        {
                            **outside,
                            "start": candidate["start"],
                            "end": candidate["end"],
                        }
                    ],
                },
            }
    )


def test_web_quality_keeps_exploratory_available_when_normal_range_exclusion_is_empty(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    timestamps = pd.date_range("2026-01-01 00:01", periods=30, freq="1min")
    values = np.zeros((30, 3))
    values[[0, 5, 10, 15, 20, 25]] = np.array(
        [[20.0, 25.0, 30.0], [30.0, 20.0, 35.0], [25.0, 40.0, 20.0],
         [45.0, 25.0, 35.0], [30.0, 45.0, 25.0], [40.0, 30.0, 45.0]]
    )
    frame = pd.DataFrame(values, columns=["A", "B", "C"])
    frame.insert(0, "time", timestamps)
    uploaded = web.save_upload(
        "range-only-variation.csv", frame.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": "2026-01-01T00:00:00",
        "normal_end": "2026-01-01T00:30:00",
        "sample_interval_minutes": 5,
        "resampling_method": "mean",
        "smoothing_window_minutes": 0,
        "filter_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "n_components": 2,
        "tag_configs": {
            tag: {"engineering_min": -10.0, "engineering_max": 10.0}
            for tag in ("A", "B", "C")
        },
    }

    quality = web.quality_payload(payload)

    assert not quality["can_train"]
    assert not quality["training_readiness"]["normal_state"]["can_train"]
    assert quality["training_readiness"]["exploratory"]["can_train"]
    exploratory = web.train_payload(
        {**payload, "model_purpose": "exploratory", "model_name": "RANGE_DRAFT"}
    )
    assert exploratory["model_status"] == "draft"
    assert exploratory["training_window_summary"][0]["engineering_range_loss"] == 0
    assert 'el("trainButton").disabled=!readiness.normal_state.can_train' in web.INDEX_HTML
    assert 'el("trainExploratoryButton").disabled=!readiness.exploratory.can_train' in web.INDEX_HTML


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
            "filter_method": "trailing_mean",
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
    totals = result["training_window_totals"]
    assert totals["training_rows"] == sum(
        item["effective_samples"] for item in result["training_window_summary"]
    )
    assert totals["used_segment_count"] == 2
    assert totals["covered_day_count"] == 1
    assert totals["max_window_id"] in {"manual-window-001", "trend-window-001"}
    assert totals["max_window_effective_samples"] == max(
        item["effective_samples"] for item in result["training_window_summary"]
    )
    assert sum(
        item["effective_sample_share"] for item in result["training_window_summary"]
    ) == pytest.approx(1.0)
    assert sum(
        item["effective_sample_share"] for item in totals["source_summary"].values()
    ) == pytest.approx(1.0)
    assert result["training_readiness"]["normal_state"]["can_train"]
    assert result["training_readiness"]["exploratory"]["can_train"]


def test_web_quality_page_exposes_training_composition_and_non_blocking_warnings():
    html = web.INDEX_HTML
    for label in (
        "训练集组成审查",
        "有效训练样本",
        "used 窗口数",
        "used 连续段数",
        "覆盖日期数",
        "最大单窗口占比",
        "最大单窗口 ID",
        "有效样本占比",
        "这些指标用于检查训练集的代表性和时间覆盖度",
    ):
        assert label in html
    assert 'id="trainingCompositionReview"' in html
    quality_source = html.split("function renderQuality(data)", 1)[1].split(
        "function excludeConstantTag", 1
    )[0]
    assert "renderTrainingComposition(data.training_window_totals||{})" in quality_source
    assert "data.training_quality_warnings" in quality_source
    assert 'card.className="issue-card"' in quality_source


def test_web_quality_returns_normal_state_training_composition_after_range_exclusion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    history = _history_frame()
    history.loc[60, "A"] = 1000.0
    uploaded = web.save_upload(
        "composition.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    quality = web.quality_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "normal_start": history.time.iloc[0].isoformat(),
            "normal_end": history.time.iloc[119].isoformat(),
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 0,
            "filter_method": "none",
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
            "tag_configs": {
                "A": {"engineering_min": -10.0, "engineering_max": 10.0}
            },
        }
    )

    totals = quality["training_window_totals"]
    assert totals["training_rows"] == sum(
        item["effective_samples"] for item in quality["training_window_summary"]
    )
    assert quality["training_window_summary"][0]["engineering_range_loss"] == 1
    assert totals["source_summary"]["legacy"]["effective_samples"] == totals[
        "training_rows"
    ]
    assert totals["source_summary"]["legacy"]["effective_sample_share"] == 1.0


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
    assert trained["training_window_totals"] == {
        "enabled_window_count": 1,
        "used_window_count": 1,
        "dropped_window_count": 0,
        "training_rows": trained["training_rows"],
        "used_segment_count": 1,
        "covered_day_count": 1,
        "max_window_id": "manual-window-001",
        "max_window_effective_samples": trained["training_rows"],
        "max_window_effective_share": 1.0,
        "source_summary": {
            "suggested": {
                "used_window_count": 0,
                "effective_samples": 0,
                "effective_sample_share": 0.0,
            },
            "manual": {
                "used_window_count": 1,
                "effective_samples": trained["training_rows"],
                "effective_sample_share": 1.0,
            },
        },
    }
    assert manifest["config"]["preprocessing_summary"] == trained[
        "training_window_summary"
    ]
    assert manifest["config"]["training_window_totals"] == trained[
        "training_window_totals"
    ]


def test_web_training_creates_isolated_candidate_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    payload = {
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
        "model_name": "ISOLATED_CANDIDATE",
    }

    first = web.train_payload(payload)
    first_path = tmp_path / "runs" / first["run_id"] / "model.pcamodel"
    first_bytes = first_path.read_bytes()
    second = web.train_payload(payload)

    assert first["run_id"] != second["run_id"]
    assert first_path.read_bytes() == first_bytes
    assert (tmp_path / "runs" / second["run_id"] / "model.pcamodel").is_file()
    assert (first["model_purpose"], first["model_status"]) == (
        "normal_state",
        "candidate",
    )
    assert (second["model_purpose"], second["model_status"]) == (
        "normal_state",
        "candidate",
    )


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
            "filter_method": "trailing_mean",
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
    assert "启用 / 使用 / 丢弃窗口" in web.INDEX_HTML
    assert "部分桶原始行删除" in web.INDEX_HTML
    assert "滤波上下文无效" in web.INDEX_HTML
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
        web.train_payload({**payload, "tag_configs": {"FIXED": {"role": "exclude"}}, "excluded_tags": [{"tag": "FIXED", "reason": "constant_in_reference_window"}]})


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

    run_dir = tmp_path / "runs" / trained["run_id"]
    candidate = run_dir / "model.pcamodel"
    assert web.validation_decision_payload({"run_id": trained["run_id"], "decision": "insufficient", "comment": "need more data"})["validated_model_download"] is None
    assert not (run_dir / "validated_model.pcamodel").exists()
    report_path = run_dir / "validation_report.json"
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    old_report["validation_metrics"] = {
        "normal_validation": {},
        "known_abnormal": {},
    }
    old_report["contribution_stability"] = {
        validation_type: {statistic: {} for statistic in ("t2", "spe")}
        for validation_type in ("normal_validation", "known_abnormal")
    }
    report_path.write_text(json.dumps(old_report), encoding="utf-8")
    sentinel = run_dir / "validated_model.pcamodel"
    sentinel.write_bytes(b"do-not-overwrite")
    with pytest.raises(ValueError, match="重新执行独立验证"):
        web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "old report"})
    assert sentinel.read_bytes() == b"do-not-overwrite"
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    old_report["validation_metrics"] = result["validation_metrics"]
    old_report["contribution_stability"] = result["contribution_stability"]
    old_report["validation_metrics"]["normal_validation"]["t2"]["exceedance_rate_95"] = "invalid"
    report_path.write_text(json.dumps(old_report), encoding="utf-8")
    with pytest.raises(ValueError, match="重新执行独立验证"):
        web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "invalid field"})
    assert sentinel.read_bytes() == b"do-not-overwrite"
    result = web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    decision = web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved"})
    assert decision["model_status"] == "validated"
    assert (run_dir / "validated_model.pcamodel").exists()
    _, candidate_manifest = load_model_package(candidate)
    validated_model, validated_manifest = load_model_package(run_dir / "validated_model.pcamodel")
    assert candidate_manifest["model_status"] == "candidate"
    assert validated_manifest["model_status"] == "validated"
    assert validated_manifest["source_candidate_package"]["identifier"] == trained["run_id"]
    assert validated_model.feature_names == tuple(candidate_manifest["feature_names"])
    saved_report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert saved_report["validation_metrics"] == result["validation_metrics"]
    assert saved_report["contribution_stability"] == result["contribution_stability"]
    assert validated_manifest["validation_summary"]["validation_metrics"] == saved_report["validation_metrics"]
    assert validated_manifest["validation_summary"]["contribution_stability"] == saved_report["contribution_stability"]


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


def test_web_training_excludes_values_outside_configured_engineering_range(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    history.loc[60, "A"] = 1000.0
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))

    result = web.train_payload(
        {
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "tags": ["A", "B", "C"],
            "tag_configs": {
                "A": {"engineering_min": -10.0, "engineering_max": 10.0},
                "B": {},
                "C": {},
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

    assert result["training_window_summary"][0]["engineering_range_loss"] == 1


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
        "filter_method": "trailing_mean",
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
    assert trend["series_stage"]["raw"] == "raw"
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


def test_web_freezes_validated_model_and_returns_two_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({"file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B", "C"], "normal_start": "2026-01-01T00:00:00", "normal_end": "2026-01-01T07:55:00", "sample_interval_minutes": 5, "smoothing_window_minutes": 10, "max_lag_minutes": 0, "lag_step_minutes": 5, "model_name": "candidate"})
    _rewrite_model_schema(tmp_path / "runs" / trained["run_id"] / "model.pcamodel", 4)
    windows = [{"id":"normal", "type":"normal_validation", "start":"2026-01-01T08:00:00", "end":"2026-01-01T09:55:00", "enabled":True, "comment":""}, {"id":"abnormal", "type":"known_abnormal", "start":"2026-01-01T10:50:00", "end":"2026-01-01T14:55:00", "enabled":True, "comment":""}]
    web.validate_payload({"run_id": trained["run_id"], "file_id": uploaded["file_id"], "timestamp_column": "time", "validation_windows": windows})
    web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": "approved"})
    result = web.freeze_deployment_payload({"run_id": trained["run_id"], "model_id": "web.unit", "model_version": 1, "frozen_by": "engineer", "comment": "freeze"})
    assert result["model_status"] == "frozen"
    assert (tmp_path / "runs" / trained["run_id"] / "frozen_model.pcamodel").is_file()
    assert (tmp_path / "runs" / trained["run_id"] / "deployment_model.pcadeploy").is_file()
    _, deployment_manifest = load_deployment_package(
        tmp_path / "runs" / trained["run_id"] / "deployment_model.pcadeploy"
    )
    assert deployment_manifest["deployment_schema_version"] == 1
    frozen_path = tmp_path / "runs" / trained["run_id"] / "frozen_model.pcamodel"
    before = frozen_path.read_bytes()
    replay = web.frozen_replay_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "replay_start": history.time.iloc[130].isoformat(),
            "replay_end": history.time.iloc[-1].isoformat(),
        }
    )
    assert replay["summary"]["output_row_count"] > 0
    assert "contributions" not in replay
    assert replay["downloads"]["scores"].endswith("artifact=scores")
    assert frozen_path.read_bytes() == before
    run_dir = frozen_path.parent
    assert {path.name for path in run_dir.glob("frozen_replay_*")} == {
        "frozen_replay_scores.csv",
        "frozen_replay_summary.json",
        "frozen_replay_contributions.json",
    }
    replay_input = history.set_index("time")
    replay_start, replay_end = history.time.iloc[130], history.time.iloc[-1]
    stored = web.replay_frozen_model(frozen_path, replay_input, replay_start, replay_end)
    large_contributions = [{"timestamp": str(index), "tags": []} for index in range(1500)]
    monkeypatch.setattr(
        web,
        "replay_frozen_model",
        lambda *args: type(stored)(stored.scores, stored.summary, large_contributions),
    )
    compact = web.frozen_replay_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "replay_start": replay_start.isoformat(),
            "replay_end": replay_end.isoformat(),
        }
    )
    assert compact["contribution_count"] == len(large_contributions)
    assert "contributions" not in compact
    assert json.loads((run_dir / "frozen_replay_contributions.json").read_text(encoding="utf-8")) == large_contributions


def test_web_schema5_first_order_lifecycle_uses_fixed_alpha(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame()
    uploaded = web.save_upload(
        "history.csv", history.to_csv(index=False).encode("utf-8-sig")
    )
    alpha = 0.35
    payload = {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "normal_start": "2026-01-01T00:00:00",
        "normal_end": "2026-01-01T07:55:00",
        "sample_interval_minutes": 5,
        "filter_method": "first_order",
        "first_order_alpha": alpha,
        "smoothing_window_minutes": 0,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "n_components": 2,
        "model_name": "schema5_first_order",
    }
    preview_configs = []
    original_preview = web.preprocess_window

    def record_preview(*args, **kwargs):
        preview_configs.append(args[2])
        return original_preview(*args, **kwargs)

    monkeypatch.setattr(web, "preprocess_window", record_preview)
    preview = web.preprocessing_preview_payload(
        {
            **payload,
            "start": history.time.iloc[0].isoformat(),
            "end": history.time.iloc[-1].isoformat(),
        }
    )
    assert preview["filtered"]
    assert preview_configs[-1].filter_method == "first_order"
    assert preview_configs[-1].first_order_alpha == alpha

    training_configs = []
    original_build = web._build_training_matrix_with_stage

    def record_training(*args, **kwargs):
        training_configs.append(args[3])
        return original_build(*args, **kwargs)

    monkeypatch.setattr(web, "_build_training_matrix_with_stage", record_training)
    quality = web.quality_payload(payload)
    assert quality["can_train"]
    trained = web.train_payload(payload)
    assert training_configs[-1].filter_method == "first_order"
    assert training_configs[-1].first_order_alpha == alpha

    run_dir = tmp_path / "runs" / trained["run_id"]
    _, candidate_manifest = load_model_package(run_dir / "model.pcamodel")
    assert candidate_manifest["schema_version"] == 5
    assert candidate_manifest["config"]["filter_method"] == "first_order"
    assert candidate_manifest["config"]["first_order_alpha"] == alpha

    windows = [
        {"id": "normal", "type": "normal_validation", "start": "2026-01-01T08:00:00", "end": "2026-01-01T09:55:00", "enabled": True, "comment": ""},
        {"id": "abnormal", "type": "known_abnormal", "start": "2026-01-01T10:50:00", "end": "2026-01-01T14:55:00", "enabled": True, "comment": ""},
    ]
    validation = web.validate_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "validation_windows": windows,
        }
    )
    assert validation["model_status"] == "candidate"
    validated = web.validation_decision_payload(
        {"run_id": trained["run_id"], "decision": "passed", "comment": "approved"}
    )
    assert validated["model_status"] == "validated"

    frozen = web.freeze_deployment_payload(
        {"run_id": trained["run_id"], "model_id": "web.schema5", "model_version": 1, "frozen_by": "engineer", "comment": "freeze"}
    )
    assert frozen["model_status"] == "frozen"
    _, frozen_manifest = load_model_package(run_dir / "frozen_model.pcamodel")
    _, deployment_manifest = load_deployment_package(run_dir / "deployment_model.pcadeploy")
    assert frozen_manifest["config"]["first_order_alpha"] == alpha
    assert deployment_manifest["deployment_schema_version"] == 2
    assert deployment_manifest["preprocessing"]["filter_method"] == "first_order"
    assert deployment_manifest["preprocessing"]["first_order_alpha"] == alpha

    replay = web.frozen_replay_payload(
        {
            "run_id": trained["run_id"],
            "file_id": uploaded["file_id"],
            "timestamp_column": "time",
            "replay_start": history.time.iloc[130].isoformat(),
            "replay_end": history.time.iloc[-1].isoformat(),
        }
    )
    assert replay["summary"]["output_row_count"] > 0


def _replay_artifacts(run_dir, marker):
    scores = _chart_scores(3)
    result = web.FrozenReplayResult(
        scores=scores,
        summary={"marker": marker},
        contributions=[{"marker": marker}],
    )
    web._commit_frozen_replay_artifacts(run_dir, result, "time")
    return {
        path.name: path.read_bytes()
        for path in run_dir.glob("frozen_replay_*")
    }


@pytest.mark.parametrize(
    "failed_name",
    ("frozen_replay_summary.json", "frozen_replay_contributions.json"),
)
def test_frozen_replay_artifact_commit_restores_all_prior_files(tmp_path, monkeypatch, failed_name):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    before = _replay_artifacts(run_dir, "before")
    real_replace = web.os.replace

    def fail_new_artifact(source, destination):
        if Path(source).suffix == ".tmp" and Path(destination).name == failed_name:
            raise OSError("new artifact commit failed")
        return real_replace(source, destination)

    monkeypatch.setattr(web.os, "replace", fail_new_artifact)
    with pytest.raises(OSError, match="new artifact commit failed"):
        _replay_artifacts(run_dir, "after")
    assert {path.name: path.read_bytes() for path in run_dir.glob("frozen_replay_*")} == before
    assert not list(run_dir.glob("*.bak"))


def test_frozen_replay_artifact_commit_keeps_unrestored_backup(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _replay_artifacts(run_dir, "before")
    real_replace = web.os.replace

    def fail_commit_and_restore(source, destination):
        source_path, destination_path = Path(source), Path(destination)
        if source_path.suffix == ".tmp" and destination_path.name == "frozen_replay_contributions.json":
            raise OSError("new artifact commit failed")
        if source_path.suffix == ".bak" and destination_path.name == "frozen_replay_summary.json":
            raise OSError("backup restore failed")
        return real_replace(source, destination)

    monkeypatch.setattr(web.os, "replace", fail_commit_and_restore)
    with pytest.raises(RuntimeError, match="new artifact commit failed.*backup restore failed.*preserved backups") as error:
        _replay_artifacts(run_dir, "after")
    backups = list(run_dir.glob("*.bak"))
    assert len(backups) == 1
    assert str(backups[0]) in str(error.value)


def test_web_freeze_rolls_back_after_second_final_replace_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads"); monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    history = _history_frame(); uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({"file_id":uploaded["file_id"],"timestamp_column":"time","tags":["A","B","C"],"normal_start":"2026-01-01T00:00:00","normal_end":"2026-01-01T07:55:00","sample_interval_minutes":5,"smoothing_window_minutes":10,"max_lag_minutes":0,"lag_step_minutes":5,"model_name":"candidate"})
    _rewrite_model_schema(tmp_path / "runs" / trained["run_id"] / "model.pcamodel", 4)
    windows=[{"id":"normal","type":"normal_validation","start":"2026-01-01T08:00:00","end":"2026-01-01T09:55:00","enabled":True,"comment":""},{"id":"abnormal","type":"known_abnormal","start":"2026-01-01T10:50:00","end":"2026-01-01T14:55:00","enabled":True,"comment":""}]
    web.validate_payload({"run_id":trained["run_id"],"file_id":uploaded["file_id"],"timestamp_column":"time","validation_windows":windows}); web.validation_decision_payload({"run_id":trained["run_id"],"decision":"passed","comment":"approved"})
    real_replace = web.os.replace
    def fail_deployment(source, destination):
        if str(destination).endswith("deployment_model.pcadeploy"): raise OSError("replace failed")
        return real_replace(source, destination)
    monkeypatch.setattr(web.os, "replace", fail_deployment)
    payload={"run_id":trained["run_id"],"model_id":"web.unit","model_version":1,"frozen_by":"engineer","comment":"freeze"}
    with pytest.raises(OSError, match="replace failed"): web.freeze_deployment_payload(payload)
    run_dir=tmp_path / "runs" / trained["run_id"]
    assert not (run_dir / "frozen_model.pcamodel").exists() and not (run_dir / "deployment_model.pcadeploy").exists()
    assert not list(run_dir.glob(".*.pcamodel")) and not list(run_dir.glob(".*.pcadeploy"))
    monkeypatch.setattr(web.os, "replace", real_replace)
    assert web.freeze_deployment_payload(payload)["model_status"] == "frozen"


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


def _rewrite_model_schema(path: Path, schema_version: int) -> None:
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    manifest["schema_version"] = schema_version
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)
