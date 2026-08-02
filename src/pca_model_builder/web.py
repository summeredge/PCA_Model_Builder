from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from email.parser import BytesParser
from email.policy import default as email_policy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

import numpy as np
import pandas as pd

from .clustering import cluster_model_scores, cluster_operating_states
from .compat import (
    MODEL_PURPOSES,
    training_windows_from_payload,
)
from .contribution import contribution_event_records, exceedance_contribution_tables
from .dpca import fit_dpca
from .model_io import load_model_package, save_model_package
from .preprocessing import PreprocessingConfig, build_dynamic_matrix, infer_segment_ids
from .quality import QualityReport, inspect_data_quality
from .screening import screen_performance_states
from .tag_config import (
    engineering_ranges,
    normalize_tag_configs,
    normalize_tag_registry,
)
from .tag_config_io import (
    MAX_TAG_CONFIG_BYTES,
    build_tag_config_template,
    export_tag_config_workbook,
    parse_tag_config_workbook,
)
from .tag_profile import model_quality_payload, profile_tag
from .training import build_training_matrix
from .trend import trend_payload_data
from .validation import (
    build_validation_matrix,
    ensure_disjoint_windows,
    validation_context_start,
)
from .windows import (
    add_training_window,
    remove_training_window,
    set_enabled_training_window,
    summarize_training_windows,
    update_training_window,
)


DEFAULT_PORT = 8775
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DATA_DIR = PROJECT_ROOT / ".web_data"
UPLOADS_DIR = WEB_DATA_DIR / "uploads"
RUNS_DIR = WEB_DATA_DIR / "runs"
MAX_REQUEST_BODY_BYTES = 200 * 1024 * 1024
MAX_CHART_POINTS = 1200
MAX_XLSX_BODY_BYTES = MAX_TAG_CONFIG_BYTES
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_VALIDATION_ARTIFACTS = {
    "scores": ("validation_scores.csv", "text/csv; charset=utf-8"),
    "report": ("validation_report.json", "application/json; charset=utf-8"),
    "contributions": (
        "validation_contributions.json",
        "application/json; charset=utf-8",
    ),
}


def run_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"PCA Model Builder 本地服务已启动：{url}")
    print("关闭此窗口即可停止服务。")
    server.serve_forever()


def save_upload(filename: str, content: bytes) -> dict[str, Any]:
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("当前仅支持 CSV 文件")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_REQUEST_BODY_BYTES:
        raise ValueError("上传文件超过 200 MB 限制")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    path = UPLOADS_DIR / f"{file_id}.csv"
    path.write_bytes(content)
    encoding, columns = _read_header(path)
    if not columns:
        path.unlink(missing_ok=True)
        raise ValueError("CSV 不包含列")
    return {
        "file_id": file_id,
        "filename": Path(filename).name,
        "columns": columns,
        "encoding": encoding,
        "size_bytes": len(content),
    }


def _read_header(path: Path) -> tuple[str, list[str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return encoding, list(pd.read_csv(path, nrows=0, encoding=encoding).columns)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError("CSV 编码无法识别，请转换为 UTF-8 或 GB18030") from last_error


def inspect_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frame = _read_upload(payload)
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(frame, timestamp_column)
    numeric_columns = _numeric_candidates(parsed, timestamp_column)
    if len(numeric_columns) < 2:
        raise ValueError("至少需要两个可用的连续数值 Tag")
    report = inspect_data_quality(parsed, timestamp_column, numeric_columns)
    timestamps = parsed[timestamp_column].dropna().sort_values().drop_duplicates()
    if len(timestamps) < 3:
        raise ValueError("至少需要三个有效时间点")
    normal_end_index = max(0, min(len(timestamps) - 2, int(len(timestamps) * 0.65)))
    validation_start_index = normal_end_index + 1
    return {
        "rows": len(parsed),
        "columns": list(parsed.columns),
        "numeric_columns": numeric_columns,
        "time_start": timestamps.iloc[0].isoformat(),
        "time_end": timestamps.iloc[-1].isoformat(),
        "suggested_normal_end": timestamps.iloc[normal_end_index].isoformat(),
        "suggested_validation_start": timestamps.iloc[
            validation_start_index
        ].isoformat(),
        "sample_interval_minutes": report.inferred_interval_minutes,
        "can_train_without_review": report.can_train,
        "quality_issues": [asdict(issue) for issue in report.issues],
    }


def train_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frame = _read_upload(payload)
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(frame, timestamp_column)
    all_tags = _numeric_candidates(parsed, timestamp_column)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    tags = _required_tags(payload)
    _require_continuous_roles(tags, registry)
    model_purpose = _model_purpose(payload.get("model_purpose"))
    model_status = "draft" if model_purpose == "exploratory" else "candidate"
    config = _preprocessing_config(payload)
    tag_configs = normalize_tag_configs(
        tags, {tag: registry[tag] for tag in tags}
    )
    training_windows = training_windows_from_payload(payload)
    training_result = build_training_matrix(
        parsed,
        timestamp_column,
        tags,
        config,
        training_windows,
        engineering_ranges(tag_configs),
    )
    excluded_tags = _excluded_tag_records(
        payload.get("excluded_tags"), training_result.reference, tags, registry
    )
    dynamic = training_result.dynamic
    components_value = payload.get("n_components")
    n_components = None if components_value in {None, ""} else int(components_value)
    variance_threshold = float(payload.get("variance_threshold", 0.95))
    model = fit_dpca(
        dynamic,
        variance_threshold=variance_threshold,
        n_components=n_components,
    )

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_name = _required_text(payload, "model_name")
    stored_config = {
        "model_name": model_name,
        "tags": tags,
        "timestamp_column": timestamp_column,
        "sample_interval_minutes": config.sample_interval_minutes,
        "smoothing_window_minutes": config.smoothing_window_minutes,
        "max_lag_minutes": config.max_lag_minutes,
        "lag_step_minutes": config.lag_step_minutes,
        "variance_threshold": variance_threshold,
        "tag_configs": tag_configs,
        "source_tag_configs": registry,
        "excluded_tags": excluded_tags,
        "training_summary": training_result.window_summaries,
        "training_quality_warnings": training_result.global_quality_warnings,
    }
    save_model_package(
        run_dir / "model.pcamodel",
        model,
        config=stored_config,
        training_windows=training_windows,
        model_purpose=model_purpose,
        model_status=model_status,
    )
    scores = model.score(dynamic)
    return {
        "run_id": run_id,
        "model_name": model_name,
        "model_purpose": model_purpose,
        "model_status": model_status,
        "training_rows": len(dynamic),
        "training_window_summary": training_result.window_summaries,
        "training_quality_warnings": training_result.global_quality_warnings,
        "dynamic_features": dynamic.shape[1],
        "n_components": model.n_components,
        "cumulative_explained_variance": float(
            model.explained_variance_ratio[: model.n_components].sum()
        ),
        "explained_variance": [
            float(value) for value in model.explained_variance_ratio
        ],
        "t2_limits": _limit_payload(model.t2_limits),
        "q_limits": _limit_payload(model.q_limits),
        "status_counts": _status_counts(scores),
        "scores": _score_payload(scores),
        "model_download": f"/download/model?run_id={run_id}",
    }


def quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frame = _read_upload(payload)
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(frame, timestamp_column)
    all_tags = _numeric_candidates(parsed, timestamp_column)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    tags = _required_tags(payload)
    _require_continuous_roles(tags, registry)
    config = _preprocessing_config(payload)
    training_result = build_training_matrix(
        parsed,
        timestamp_column,
        tags,
        config,
        training_windows_from_payload(payload),
        engineering_ranges(normalize_tag_configs(tags, {tag: registry[tag] for tag in tags})),
        validate_dynamic=False,
    )
    result = model_quality_payload(
        parsed,
        training_result.reference,
        timestamp_column,
        tags,
        registry,
        config.sample_interval_minutes,
    )
    if result["can_train"] and training_result.dynamic.empty:
        result["time_issues"].append(
            {
                "code": "dynamic_matrix_empty",
                "severity": "error",
                "message": "平滑和Lag预热后没有有效动态样本。",
                "count": 0,
                "tag": None,
                "details": {},
            }
        )
        result["can_train"] = False
    elif result["can_train"] and np.linalg.matrix_rank(
        training_result.dynamic.to_numpy(dtype=float)
    ) < 3:
        result["time_issues"].append(
            {
                "code": "insufficient_effective_rank",
                "severity": "error",
                "message": "有效秩不足3，无法同时建立PC1/PC2和SPE残差空间。",
                "count": len(training_result.dynamic),
                "tag": None,
                "details": {},
            }
        )
        result["can_train"] = False
    result["training_window_summary"] = training_result.window_summaries
    result["training_quality_warnings"] = training_result.global_quality_warnings
    return result


def training_windows_payload(payload: dict[str, Any]) -> dict[str, Any]:
    windows = training_windows_from_payload(payload)
    operation = payload.get("operation")
    if operation is not None:
        if not isinstance(operation, dict):
            raise ValueError("training_windows操作必须是对象")
        action = operation.get("action")
        if action == "add":
            windows = add_training_window(windows, operation.get("window", {}))
        elif action == "update":
            windows = update_training_window(
                windows, str(operation.get("id", "")), operation.get("changes", {})
            )
        elif action == "remove":
            windows = remove_training_window(windows, str(operation.get("id", "")))
        elif action == "set_enabled":
            windows = set_enabled_training_window(
                windows, str(operation.get("id", "")), operation.get("enabled")
            )
        else:
            raise ValueError("training_windows操作无效")
    timestamps = None
    if payload.get("file_id"):
        timestamp_column = _required_text(payload, "timestamp_column")
        timestamps = _parse_timestamp_column(_read_upload(payload), timestamp_column)[
            timestamp_column
        ]
    return {
        "training_windows": windows,
        "summary": summarize_training_windows(
            windows, timestamps, int(payload.get("sample_interval_minutes", 5))
        ),
    }


def trend_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    all_tags = _numeric_candidates(parsed, timestamp_column)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ValueError("趋势Tag必须是非空列表")
    tags = [str(tag).strip() for tag in raw_tags]
    if len(tags) != len(set(tags)):
        raise ValueError("趋势Tag不能重复")
    if len(tags) > 8:
        raise ValueError("趋势浏览一次最多选择8个Tag")
    missing = [tag for tag in tags if tag not in all_tags]
    if missing:
        raise ValueError(f"找不到趋势Tag：{', '.join(missing)}")
    indexed = parsed.set_index(timestamp_column).sort_index()
    reference_start = _optional_timestamp(payload.get("normal_start"))
    reference_end = _optional_timestamp(payload.get("normal_end"))
    if "training_windows" in payload:
        window = _single_enabled_training_window(training_windows_from_payload(payload))
        reference_start = pd.Timestamp(window["start"])
        reference_end = pd.Timestamp(window["end"])
    return trend_payload_data(
        indexed,
        tags,
        _preprocessing_config(payload),
        pd.Timestamp(_required_text(payload, "start")),
        pd.Timestamp(_required_text(payload, "end")),
        str(payload.get("display_mode", "both")),
        registry,
        reference_start,
        reference_end,
    )


def tag_config_template_payload(payload: dict[str, Any]) -> bytes:
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    return build_tag_config_template(_numeric_candidates(parsed, timestamp_column))


def tag_config_export_payload(payload: dict[str, Any]) -> bytes:
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    tags = _numeric_candidates(parsed, timestamp_column)
    return export_tag_config_workbook(tags, payload.get("tag_configs"))


def tag_config_import_payload(
    filename: str,
    content: bytes,
    file_id: str,
    timestamp_column: str,
    encoding: str,
) -> dict[str, Any]:
    if Path(filename).suffix.lower() != ".xlsx":
        raise ValueError("工程配置仅接受无宏的.xlsx文件")
    parsed = _parse_timestamp_column(
        _read_upload(
            {
                "file_id": file_id,
                "timestamp_column": timestamp_column,
                "encoding": encoding,
            }
        ),
        timestamp_column,
    )
    return parse_tag_config_workbook(
        content, _numeric_candidates(parsed, timestamp_column)
    )


def cluster_payload(payload: dict[str, Any]) -> dict[str, Any]:
    exploratory_run_id = payload.get("exploratory_run_id")
    if exploratory_run_id not in {None, ""}:
        return _cluster_exploratory_payload(
            payload,
            _validated_id(str(exploratory_run_id), "exploratory_run_id"),
        )
    frame = _read_upload(payload)
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(frame, timestamp_column)
    all_tags = _numeric_candidates(parsed, timestamp_column)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    tags = _required_tags(payload)
    _require_continuous_roles(tags, registry)
    config = _preprocessing_config(payload)
    tag_configs = normalize_tag_configs(
        tags, {tag: registry[tag] for tag in tags}
    )
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    _require_clean_data(
        analysis,
        timestamp_column,
        tags,
        config.sample_interval_minutes,
        engineering_ranges(tag_configs),
    )
    indexed = _indexed_tags(analysis, timestamp_column, tags)
    dynamic = build_dynamic_matrix(
        indexed,
        tags,
        config,
        infer_segment_ids(indexed.index, config.sample_interval_minutes),
    )
    if dynamic.empty:
        raise ValueError("平滑和 Lag 扩展后没有足够的聚类样本")
    result = cluster_operating_states(
        dynamic,
        n_clusters=int(payload.get("n_clusters", 3)),
        variance_threshold=float(payload.get("variance_threshold", 0.95)),
        sample_interval_minutes=config.sample_interval_minutes,
    )
    return _cluster_result_payload(result)


def _cluster_exploratory_payload(
    payload: dict[str, Any], exploratory_run_id: str
) -> dict[str, Any]:
    model_path = RUNS_DIR / exploratory_run_id / "model.pcamodel"
    if not model_path.is_file():
        raise ValueError("探索模型运行记录不存在")
    model, manifest = load_model_package(model_path)
    if manifest["model_purpose"] != "exploratory":
        raise ValueError("聚类必须引用探索模型")
    config_data = manifest["config"]
    timestamp_column = _required_text(payload, "timestamp_column")
    if timestamp_column != config_data["timestamp_column"]:
        raise ValueError("探索模型时间戳列与聚类请求不一致")
    tags = list(config_data["tags"])
    tag_configs = normalize_tag_configs(tags, config_data.get("tag_configs"))
    config = PreprocessingConfig(
        sample_interval_minutes=int(config_data["sample_interval_minutes"]),
        smoothing_window_minutes=int(config_data["smoothing_window_minutes"]),
        max_lag_minutes=int(config_data["max_lag_minutes"]),
        lag_step_minutes=int(config_data["lag_step_minutes"]),
    )
    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    _require_clean_data(
        analysis,
        timestamp_column,
        tags,
        config.sample_interval_minutes,
        engineering_ranges(tag_configs),
    )
    indexed = _indexed_tags(analysis, timestamp_column, tags)
    dynamic = build_dynamic_matrix(
        indexed,
        tags,
        config,
        infer_segment_ids(indexed.index, config.sample_interval_minutes),
    )
    if dynamic.empty:
        raise ValueError("平滑和 Lag 扩展后没有足够的聚类样本")
    result = cluster_model_scores(
        model,
        dynamic,
        n_clusters=int(payload.get("n_clusters", 3)),
        sample_interval_minutes=config.sample_interval_minutes,
    )
    return {
        **_cluster_result_payload(result),
        "exploratory_run_id": exploratory_run_id,
    }


def _cluster_result_payload(result: Any) -> dict[str, Any]:
    points = result.points
    if len(points) > MAX_CHART_POINTS:
        positions = np.unique(
            np.linspace(0, len(points) - 1, MAX_CHART_POINTS, dtype=int)
        )
        points = points.iloc[positions]
    return {
        "sample_count": len(result.points),
        "n_components": result.n_components,
        "cumulative_explained_variance": result.cumulative_explained_variance,
        "clusters": list(result.summaries),
        "points": [
            {
                "timestamp": timestamp.isoformat(),
                "pc1": float(row.pc1),
                "pc2": float(row.pc2),
                "cluster": int(row.cluster),
            }
            for timestamp, row in points.iterrows()
        ],
        "engineer_decision_required": True,
    }


def performance_screen_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list):
        raise ValueError("性能条件必须是列表")
    indexed = analysis.set_index(timestamp_column)
    return screen_performance_states(
        indexed,
        raw_conditions,
        sample_interval_minutes=int(payload.get("sample_interval_minutes", 5)),
    )


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    model_path = RUNS_DIR / run_id / "model.pcamodel"
    if not model_path.is_file():
        raise ValueError("模型运行记录不存在")
    model, manifest = load_model_package(model_path)
    if manifest["model_purpose"] != "normal_state":
        raise ValueError("探索模型不能执行独立验证")
    config_data = manifest["config"]
    tags = list(config_data["tags"])
    tag_configs = normalize_tag_configs(tags, config_data.get("tag_configs"))
    timestamp_column = _required_text(payload, "timestamp_column")
    validation_window = (
        pd.Timestamp(_required_text(payload, "validation_start")),
        pd.Timestamp(_required_text(payload, "validation_end")),
    )
    training_windows = [
        (pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
        for window in manifest["training_windows"]
        if window["enabled"]
    ]
    ensure_disjoint_windows(training_windows, [validation_window])

    parsed = _parse_timestamp_column(_read_upload(payload), timestamp_column)
    config = PreprocessingConfig(
        sample_interval_minutes=int(config_data["sample_interval_minutes"]),
        smoothing_window_minutes=int(config_data["smoothing_window_minutes"]),
        max_lag_minutes=int(config_data["max_lag_minutes"]),
        lag_step_minutes=int(config_data["lag_step_minutes"]),
    )
    context_start = validation_context_start(validation_window[0], config)
    context = _select_window(
        parsed,
        timestamp_column,
        context_start.isoformat(),
        validation_window[1].isoformat(),
    )
    _require_clean_data(
        context,
        timestamp_column,
        tags,
        config.sample_interval_minutes,
        engineering_ranges(tag_configs),
    )
    indexed = _indexed_tags(context, timestamp_column, tags)
    dynamic = build_validation_matrix(
        indexed,
        tags,
        config,
        validation_window[0],
        validation_window[1],
    )
    scores = model.score(dynamic)
    focus_timestamp = _focus_timestamp(scores, model.t2_limits[0.99], model.q_limits[0.99])
    contributions = contribution_event_records(
        exceedance_contribution_tables(
            model,
            dynamic,
            scores,
            sample_interval_minutes=config.sample_interval_minutes,
        ),
        tag_configs,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "engineer_decision_required": True,
        "validation_window": [
            validation_window[0].isoformat(),
            validation_window[1].isoformat(),
        ],
        "scored_rows": len(scores),
        "status_counts": _status_counts(scores),
        "maximum_t2": float(scores["t2"].max()),
        "maximum_spe": float(scores["spe"].max()),
        "focus_timestamp": focus_timestamp.isoformat(),
        "t2_limits": _limit_payload(model.t2_limits),
        "q_limits": _limit_payload(model.q_limits),
        "scores": _score_payload(scores),
        "contributions": contributions,
    }
    label_column = str(payload.get("label_column", "")).strip()
    if label_column:
        validation = _select_window(
            parsed,
            timestamp_column,
            validation_window[0].isoformat(),
            validation_window[1].isoformat(),
        )
        if label_column not in validation.columns:
            raise ValueError(f"找不到工程标签列：{label_column}")
        labels = validation.set_index(timestamp_column)[label_column].reindex(scores.index)
        result["status_by_engineering_label"] = {
            str(label): {
                status: int(count)
                for status, count in Counter(scores.loc[labels == label, "status"]).items()
            }
            for label in labels.dropna().unique()
        }
    else:
        result["status_by_engineering_label"] = {}
    run_dir = RUNS_DIR / run_id
    scores.to_csv(
        run_dir / "validation_scores.csv",
        index_label=timestamp_column,
        encoding="utf-8-sig",
    )
    (run_dir / "validation_contributions.json").write_text(
        json.dumps(contributions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        key: value
        for key, value in result.items()
        if key not in {"scores", "contributions"}
    }
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["validation_downloads"] = {
        artifact: f"/download/validation?run_id={run_id}&artifact={artifact}"
        for artifact in _VALIDATION_ARTIFACTS
    }
    return result


def _read_upload(payload: dict[str, Any]) -> pd.DataFrame:
    file_id = _validated_id(_required_text(payload, "file_id"), "file_id")
    path = UPLOADS_DIR / f"{file_id}.csv"
    if not path.is_file():
        raise ValueError("上传文件不存在，请重新上传")
    return pd.read_csv(path, encoding=str(payload.get("encoding", "utf-8-sig")))


def _parse_timestamp_column(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise ValueError(f"找不到时间列：{timestamp_column}")
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if parsed.isna().any():
        raise ValueError("时间列包含无法解析的值")
    result = frame.copy()
    result[timestamp_column] = parsed
    return result


def _numeric_candidates(frame: pd.DataFrame, timestamp_column: str) -> list[str]:
    candidates = []
    for column in frame.columns:
        if column == timestamp_column:
            continue
        original_non_null = int(frame[column].notna().sum())
        if original_non_null == 0:
            continue
        numeric_count = int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
        if numeric_count / original_non_null >= 0.8:
            candidates.append(str(column))
    return candidates


def _preprocessing_config(payload: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        sample_interval_minutes=int(payload.get("sample_interval_minutes", 5)),
        smoothing_window_minutes=int(payload.get("smoothing_window_minutes", 10)),
        max_lag_minutes=int(payload.get("max_lag_minutes", 60)),
        lag_step_minutes=int(payload.get("lag_step_minutes", 5)),
    )


def _required_tags(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("tags")
    if not isinstance(raw, list):
        raise ValueError("Tag 必须是列表")
    tags = [str(tag).strip() for tag in raw if str(tag).strip()]
    if len(tags) < 2:
        raise ValueError("至少选择两个连续 Tag")
    if len(tags) != len(set(tags)):
        raise ValueError("Tag 不能重复")
    return tags


def _model_purpose(value: object) -> str:
    if value in {None, ""}:
        return "normal_state"
    if value not in MODEL_PURPOSES:
        raise ValueError("model_purpose必须是exploratory或normal_state")
    return str(value)


def _single_enabled_training_window(
    training_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    enabled = [window for window in training_windows if window["enabled"]]
    if len(enabled) != 1:
        raise ValueError("当前训练仅支持一个启用的training_windows窗口")
    return enabled[0]


def _require_continuous_roles(
    tags: Sequence[str], registry: dict[str, dict[str, Any]]
) -> None:
    missing = [tag for tag in tags if tag not in registry]
    if missing:
        raise ValueError(f"找不到 Tag：{', '.join(missing)}")
    invalid = [tag for tag in tags if registry[tag]["role"] != "continuous_input"]
    if invalid:
        raise ValueError(
            "只有continuous_input角色可以进入PCA：" + ", ".join(invalid)
        )


def _excluded_tag_records(
    value: object,
    reference: pd.DataFrame,
    selected_tags: Sequence[str],
    registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        raise ValueError("excluded_tags必须是列表")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("excluded_tags记录必须是对象")
        tag = str(item.get("tag", "")).strip()
        if not tag or tag in seen or tag not in registry:
            raise ValueError("excluded_tags包含无效、重复或未知Tag")
        if tag in selected_tags:
            raise ValueError(f"已排除Tag仍在建模选择中：{tag}")
        if item.get("reason") != "constant_in_reference_window":
            raise ValueError(f"{tag}的排除原因无效")
        profile = profile_tag(reference[tag], registry[tag])
        if profile["unique_count"] != 1 or profile["valid_count"] < 1:
            raise ValueError(f"{tag}并非参考期精确常量，不能记录为常量排除")
        finite = pd.to_numeric(reference[tag], errors="coerce")
        finite = finite[np.isfinite(finite)]
        records.append(
            {
                "tag": tag,
                "reason": "constant_in_reference_window",
                "sample_count": int(profile["valid_count"]),
                "unique_count": 1,
                "constant_value": float(finite.iloc[0]),
            }
        )
        seen.add(tag)
    return records


def _select_window(
    frame: pd.DataFrame,
    timestamp_column: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time > end_time:
        raise ValueError("时间窗口开始时间不能晚于结束时间")
    selected = frame.loc[
        frame[timestamp_column].between(start_time, end_time, inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError("所选时间窗口没有数据")
    return selected


def _require_clean_data(
    frame: pd.DataFrame,
    timestamp_column: str,
    tags: Sequence[str],
    expected_interval_minutes: float,
    configured_engineering_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    report = inspect_data_quality(
        frame,
        timestamp_column,
        tags,
        engineering_ranges=configured_engineering_ranges,
        expected_interval_minutes=expected_interval_minutes,
    )
    if not report.can_train:
        raise ValueError(_format_quality_errors(report))


def _format_quality_errors(report: QualityReport) -> str:
    details = "；".join(
        f"{issue.code}({issue.count})"
        + (f"[{issue.tag}]" if issue.tag else "")
        + f"：{issue.message}"
        for issue in report.issues
    )
    return f"数据质量问题尚未处理：{details}"


def _indexed_tags(
    frame: pd.DataFrame, timestamp_column: str, tags: Sequence[str]
) -> pd.DataFrame:
    missing = [tag for tag in tags if tag not in frame.columns]
    if missing:
        raise ValueError(f"找不到 Tag：{', '.join(missing)}")
    return (
        frame.loc[:, [timestamp_column, *tags]]
        .sort_values(timestamp_column)
        .set_index(timestamp_column)
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"缺少参数：{key}")
    return value


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or not str(value).strip():
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("时间值无法解析")
    return timestamp


def _validated_id(value: str, label: str) -> str:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"无效的 {label}")
    return value


def _validation_artifact(artifact: str) -> tuple[str, str]:
    try:
        return _VALIDATION_ARTIFACTS[artifact]
    except KeyError as error:
        raise ValueError("无效的验证工件类型") from error


def _limit_payload(limits: dict[float, float]) -> dict[str, float]:
    return {str(int(alpha * 100)): float(value) for alpha, value in limits.items()}


def _status_counts(scores: pd.DataFrame) -> dict[str, int]:
    counts = Counter(scores["status"])
    return {status: int(counts.get(status, 0)) for status in ("normal", "attention", "abnormal")}


def _score_payload(scores: pd.DataFrame) -> list[dict[str, Any]]:
    positions = _chart_positions(scores, MAX_CHART_POINTS)
    rows = scores.iloc[positions]
    pc_columns = [column for column in scores.columns if column.startswith("pc")]
    result = []
    for timestamp, row in rows.iterrows():
        record = {
            "timestamp": timestamp.isoformat(),
            "t2": float(row.t2),
            "spe": float(row.spe),
            "t2_limit_ratio": float(row.t2_limit_ratio),
            "spe_limit_ratio": float(row.spe_limit_ratio),
            "t2_status": str(row.t2_status),
            "spe_status": str(row.spe_status),
            "status": str(row.status),
        }
        record.update({column: float(row[column]) for column in pc_columns})
        result.append(record)
    return result


def _chart_positions(scores: pd.DataFrame, limit: int) -> np.ndarray:
    if limit < 2:
        raise ValueError("chart point limit must be at least 2")
    if len(scores) <= limit:
        return np.arange(len(scores), dtype=int)

    severity = scores["status"].map(
        {"normal": 0, "attention": 1, "abnormal": 2}
    ).to_numpy(dtype=int)
    critical = {0, len(scores) - 1}
    critical.update(np.flatnonzero(severity > 0).tolist())
    critical.add(int(np.argmax(scores["t2"].to_numpy())))
    critical.add(int(np.argmax(scores["spe"].to_numpy())))
    for column in ("status", "t2_status", "spe_status"):
        values = scores[column].to_numpy()
        switches = np.flatnonzero(values[1:] != values[:-1]) + 1
        critical.update(switches.tolist())
        critical.update((switches - 1).tolist())

    if len(critical) <= limit:
        selected = set(critical)
        candidates = np.array(
            [position for position in range(len(scores)) if position not in selected]
        )
        selected.update(
            _spread_positions(candidates, limit - len(selected)).tolist()
        )
        return np.array(sorted(selected), dtype=int)

    bucket_count = max(1, (limit - 2) // 3)
    if limit < 5:
        ratios = np.maximum(
            scores["t2_limit_ratio"].to_numpy(dtype=float),
            scores["spe_limit_ratio"].to_numpy(dtype=float),
        )
        ranked = sorted(
            range(len(scores)),
            key=lambda position: (severity[position], ratios[position]),
            reverse=True,
        )
        selected = {0, len(scores) - 1}
        selected.update(ranked[: limit - len(selected)])
        return np.array(sorted(selected), dtype=int)
    bucket_ids = _time_bucket_ids(scores.index, bucket_count)
    t2_ratio = scores["t2_limit_ratio"].to_numpy(dtype=float)
    spe_ratio = scores["spe_limit_ratio"].to_numpy(dtype=float)
    selected = {0, len(scores) - 1}
    for bucket in range(bucket_count):
        positions = np.flatnonzero(bucket_ids == bucket)
        if not len(positions):
            continue
        selected.add(
            int(
                max(
                    positions,
                    key=lambda position: (
                        severity[position],
                        max(t2_ratio[position], spe_ratio[position]),
                    ),
                )
            )
        )
        selected.add(int(positions[np.argmax(t2_ratio[positions])]))
        selected.add(int(positions[np.argmax(spe_ratio[positions])]))

    remaining = np.array(
        [position for position in range(len(scores)) if position not in selected]
    )
    selected.update(_spread_positions(remaining, limit - len(selected)).tolist())
    return np.array(sorted(selected), dtype=int)


def _time_bucket_ids(index: pd.Index, bucket_count: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex) and index[-1] > index[0]:
        elapsed = (index - index[0]).asi8.astype(float)
        return np.minimum(
            (elapsed / (elapsed[-1] + 1.0) * bucket_count).astype(int),
            bucket_count - 1,
        )
    return np.minimum(
        np.arange(len(index)) * bucket_count // len(index), bucket_count - 1
    )


def _spread_positions(candidates: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(candidates):
        return np.array([], dtype=int)
    if count >= len(candidates):
        return candidates
    offsets = np.arange(count) * len(candidates) // count
    return candidates[offsets]


def _focus_timestamp(
    scores: pd.DataFrame, t2_limit_99: float, q_limit_99: float
) -> pd.Timestamp:
    t2_ratio = scores["t2"] / max(t2_limit_99, np.finfo(float).eps)
    q_ratio = scores["spe"] / max(q_limit_99, np.finfo(float).eps)
    return pd.Timestamp(pd.concat([t2_ratio, q_ratio], axis=1).max(axis=1).idxmax())


class _Handler(BaseHTTPRequestHandler):
    server_version = "PCAModelBuilder/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok", "port": self.server.server_port})
            return
        if parsed.path == "/download/tag-config-template":
            try:
                query = parse_qs(parsed.query)
                body = tag_config_template_payload(
                    {
                        "file_id": query.get("file_id", [""])[0],
                        "timestamp_column": query.get("timestamp_column", [""])[0],
                        "encoding": query.get("encoding", ["utf-8-sig"])[0],
                    }
                )
                self._send_download_bytes(
                    body,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "PCA_Tag_Config_Template.xlsx",
                )
            except Exception as error:
                self._send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/download/model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_model(run_id)
            except Exception as error:
                self._send_json({"error": str(error)}, 400)
            return
        if parsed.path == "/download/validation":
            try:
                query = parse_qs(parsed.query)
                run_id = _validated_id(query.get("run_id", [""])[0], "run_id")
                artifact = query.get("artifact", [""])[0]
                filename, content_type = _validation_artifact(artifact)
                self._send_download(
                    RUNS_DIR / run_id / filename,
                    content_type,
                    filename,
                )
            except Exception as error:
                self._send_json({"error": str(error)}, 400)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                filename, content = self._multipart_file("file")
                self._send_json(save_upload(filename, content))
                return
            if parsed.path == "/api/tag-config/import":
                filename, content = self._multipart_file("file")
                query = parse_qs(parsed.query)
                self._send_json(
                    tag_config_import_payload(
                        filename,
                        content,
                        query.get("file_id", [""])[0],
                        query.get("timestamp_column", [""])[0],
                        query.get("encoding", ["utf-8-sig"])[0],
                    )
                )
                return
            payload = self._json_body()
            if parsed.path == "/api/inspect":
                self._send_json(inspect_payload(payload))
                return
            if parsed.path == "/api/quality":
                self._send_json(quality_payload(payload))
                return
            if parsed.path == "/api/training-windows":
                self._send_json(training_windows_payload(payload))
                return
            if parsed.path == "/api/trend":
                self._send_json(trend_payload(payload))
                return
            if parsed.path == "/api/tag-config/export":
                self._send_download_bytes(
                    tag_config_export_payload(payload),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "PCA_Tag_Config.xlsx",
                )
                return
            if parsed.path == "/api/cluster":
                self._send_json(cluster_payload(payload))
                return
            if parsed.path == "/api/performance-screen":
                self._send_json(performance_screen_payload(payload))
                return
            if parsed.path == "/api/train":
                self._send_json(train_payload(payload))
                return
            if parsed.path == "/api/validate":
                self._send_json(validate_payload(payload))
                return
            self._send_json({"error": "Not found"}, 404)
        except Exception as error:
            self._send_json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _content_length(self) -> int:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("无效的请求长度") from error
        if length <= 0 or length > MAX_REQUEST_BODY_BYTES:
            raise ValueError("请求为空或超过 200 MB 限制")
        return length

    def _json_body(self) -> dict[str, Any]:
        body = self.rfile.read(self._content_length())
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON 请求必须是对象")
        return value

    def _multipart_file(self, field_name: str) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("上传请求必须使用 multipart/form-data")
        body = self.rfile.read(self._content_length())
        message = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            + body
        )
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == field_name:
                filename = part.get_filename() or "upload.csv"
                return filename, part.get_payload(decode=True) or b""
        raise ValueError("上传请求缺少文件")

    def _send_json(self, value: Any, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_text(self, value: str, content_type: str, status: int = 200) -> None:
        self._send_bytes(value.encode("utf-8"), content_type, status)

    def _send_bytes(self, body: bytes, content_type: str, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_model(self, run_id: str) -> None:
        path = RUNS_DIR / run_id / "model.pcamodel"
        self._send_download(path, "application/zip", "model.pcamodel")

    def _send_download(
        self, path: Path, content_type: str, filename: str
    ) -> None:
        if not path.is_file():
            raise ValueError("下载文件不存在")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_download_bytes(
        self, body: bytes, content_type: str, filename: str
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local PCA Model Builder web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser=not args.no_open)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>PCA 状态模型构建工具</title>
  <style>
    :root { --bg:#f5f7fa; --panel:#fff; --line:#d7dee8; --line-soft:#e8edf3; --text:#17212b; --muted:#5f6c7b; --accent:#176b87; --accent-soft:#e7f3f7; --green:#146c43; --warn:#9a6700; --danger:#b42318; --normal:#16845b; --attention:#d19a20; --abnormal:#cf3f36; }
    * { box-sizing:border-box; }
    [hidden] { display:none !important; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; }
    header { padding:22px 28px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
    h1 { margin:0 0 6px; font-size:24px; }
    .subtitle { color:var(--muted); font-size:14px; }
    main { display:grid; grid-template-columns:minmax(330px,420px) 1fr; gap:18px; padding:18px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:16px; }
    .controls { display:grid; gap:11px; align-content:start; }
    .group { display:grid; gap:9px; padding:11px; background:#f9fbfc; border:1px solid var(--line-soft); border-radius:8px; }
    .group-title { font-size:13px; font-weight:700; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
    input, select { width:100%; min-height:34px; padding:7px 8px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); }
    input[type=file] { padding:5px; }
    button { border:0; border-radius:6px; padding:10px 14px; background:var(--accent); color:#fff; font-weight:650; cursor:pointer; }
    button.secondary { background:#e8edf3; color:var(--text); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; }
    .status { min-height:38px; padding:9px 10px; border-radius:7px; border:1px solid var(--line); color:var(--muted); white-space:pre-wrap; font-size:13px; }
    .status.info { background:#e8f4fa; color:#075985; border-color:#b9def0; }
    .status.success { background:#e4f5ed; color:#166534; border-color:#b9e4ce; }
    .status.warning { background:#fff4d6; color:#8a5a00; border-color:#f3d58c; }
    .status.error { background:#fee9e7; color:#991b1b; border-color:#f5c2bd; }
    .help { color:var(--muted); font-size:12px; line-height:1.45; }
    .tag-options { display:grid; gap:5px; max-height:260px; overflow:auto; padding:8px; background:#fff; border:1px solid var(--line); border-radius:6px; }
    .tag-options label { display:flex; align-items:center; gap:6px; color:var(--text); overflow:hidden; }
    .tag-options input { width:auto; min-height:auto; }
    .tag-options span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tag-row { display:grid !important; grid-template-columns:auto minmax(0,1fr) auto; cursor:pointer; padding:4px; border-radius:4px; }
    .tag-row.selected { background:var(--accent-soft); }
    .tag-state { font-size:11px; }
    .tag-state.usable { color:var(--normal); }
    .tag-state.review { color:var(--warn); }
    .tag-state.blocking { color:var(--danger); }
    .tag-toolbar { display:flex; gap:5px; flex-wrap:wrap; }
    .tag-toolbar button { padding:7px 9px; }
    .detail-fields { display:grid; gap:9px; max-width:760px; }
    .inner-tabs { display:flex; gap:6px; margin-bottom:10px; }
    .inner-tab { background:#e8edf3; color:var(--text); }
    .inner-tab.active { background:var(--accent); color:#fff; }
    .inner-panel { display:none; }
    .inner-panel.active { display:grid; gap:10px; }
    .issue-card { border:1px solid var(--line); border-left:4px solid var(--warn); border-radius:6px; padding:9px; display:grid; gap:5px; }
    .issue-card.blocking { border-left-color:var(--danger); }
    .trend-controls { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
    .trend-chart { min-height:260px; overflow:auto; border:1px solid var(--line); border-radius:7px; }
    .trend-chart svg { height:260px; display:block; }
    .compact-list { max-height:220px; overflow:auto; border:1px solid var(--line); border-radius:6px; padding:7px; display:grid; gap:4px; }
    textarea { width:100%; min-height:70px; padding:7px 8px; border:1px solid var(--line); border-radius:6px; font:inherit; }
    .condition-list { display:grid; gap:6px; }
    .condition-row { display:grid; grid-template-columns:1.4fr 1fr 1fr auto; gap:6px; align-items:end; }
    .condition-row button { padding:8px 10px; }
    .sub-title { font-size:12px; font-weight:700; padding-top:3px; border-top:1px solid var(--line-soft); }
    .results { display:grid; gap:14px; min-width:0; align-content:start; }
    .tabs { display:flex; gap:8px; border-bottom:1px solid var(--line); padding-bottom:8px; }
    .tab { background:#e8edf3; color:var(--text); }
    .tab.active { background:var(--accent); color:#fff; }
    .panel { display:none; gap:14px; }
    .panel.active { display:grid; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:9px; }
    .metric { padding:11px; background:#f8fafc; border:1px solid var(--line-soft); border-radius:8px; }
    .metric strong { display:block; font-size:21px; }
    .metric span { color:var(--muted); font-size:12px; }
    .chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .chart-card { display:grid; gap:7px; min-width:0; }
    .chart-card h3 { margin:0; font-size:14px; }
    .chart { height:260px; border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#fff; }
    .chart svg { width:100%; height:100%; display:block; }
    .empty { display:grid; place-items:center; min-height:120px; color:var(--muted); border:1px dashed var(--line); border-radius:7px; padding:18px; text-align:center; }
    .variance { display:flex; gap:5px; align-items:flex-end; height:130px; padding:10px; border:1px solid var(--line); border-radius:7px; overflow-x:auto; }
    .variance-bar { min-width:24px; flex:1; max-width:48px; background:var(--accent); border-radius:3px 3px 0 0; position:relative; }
    .variance-bar.selected { background:var(--green); }
    .variance-bar span { position:absolute; top:-18px; width:100%; text-align:center; font-size:10px; color:var(--muted); }
    .legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
    .swatch { width:18px; height:3px; display:inline-block; vertical-align:middle; margin-right:5px; }
    .table-wrap { overflow:auto; max-height:360px; border:1px solid var(--line); border-radius:7px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { padding:8px 9px; border-bottom:1px solid var(--line-soft); text-align:left; }
    th { position:sticky; top:0; background:#eef2f6; }
    td.numeric { text-align:right; font-variant-numeric:tabular-nums; }
    .download { color:#fff; background:var(--green); padding:8px 11px; border-radius:6px; text-decoration:none; font-size:13px; }
    .validation-box { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:8px; align-items:end; padding:10px; background:#f8fafc; border:1px solid var(--line-soft); border-radius:8px; }
    .notice { padding:9px 10px; border-left:4px solid var(--warn); background:#fff8e7; color:#765000; font-size:13px; }
    @media (max-width:1050px) { main { grid-template-columns:1fr; } }
    @media (max-width:760px) { .chart-grid,.validation-box,.trend-controls { grid-template-columns:1fr; } .row,.condition-row { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>PCA 状态模型构建工具</h1>
    <div class="subtitle">本地离线 DPCA 建模、T²/SPE 状态监控与贡献分析。统计贡献不等同于根因。</div>
  </header>
  <main>
    <section class="controls">
      <div class="group">
        <div class="group-title">1. 历史数据</div>
        <label>CSV 文件<input id="fileInput" type="file" accept=".csv,text/csv"></label>
        <div class="actions"><button id="uploadButton">上传并读取列</button><button id="resetButton" class="secondary">清空</button></div>
        <div class="row">
          <label>时间列<select id="timestampColumn"></select></label>
          <label>编码<select id="encoding"><option value="utf-8-sig">UTF-8</option><option value="gb18030">GB18030</option></select></label>
        </div>
        <button id="inspectButton" class="secondary" disabled>检查时间轴与数值列</button>
      </div>
      <div class="group">
        <div class="group-title">2. 建模 Tag</div>
        <input id="tagSearch" placeholder="搜索 Tag">
        <div class="tag-toolbar"><button id="selectAllTags" class="secondary">全选</button><button id="clearAllTags" class="secondary">取消全选</button><button id="showProblemTags" class="secondary">只看问题Tag</button></div>
        <div id="tagOptions" class="tag-options"><span class="help">检查数据后显示连续数值列。</span></div>
        <div class="help">仅勾选且角色为 continuous_input 的Tag进入PCA；点击Tag在右侧查看配置与质量。</div>
      </div>
      <div class="group">
        <div class="group-title">3. 参考状态与 DPCA 参数</div>
        <input id="normalStart" type="datetime-local" hidden><input id="normalEnd" type="datetime-local" hidden>
        <div class="row"><label>候选开始<input id="candidateStart" type="datetime-local"></label><label>候选结束<input id="candidateEnd" type="datetime-local"></label><label>备注<input id="candidateComment" type="text"></label><button id="addManualCandidate" class="secondary" type="button">加入手工候选</button></div>
        <h3>正常候选时段</h3><div id="trainingWindows" class="table-wrap"><div class="empty">检查数据后可管理候选时段。</div></div>
        <div class="help">候选加入后默认不启用；只有工程师明确启用的时段会参与质量检查和训练。</div>
        <div class="row"><label>采样间隔（分钟）<input id="sampleInterval" type="number" min="1" value="5"></label><label>尾随平滑（分钟）<input id="smoothingWindow" type="number" min="1" value="10"></label></div>
        <div class="row"><label>最大 Lag（分钟）<input id="maxLag" type="number" min="0" value="60"></label><label>Lag 步长（分钟）<input id="lagStep" type="number" min="1" value="5"></label></div>
        <div class="row"><label>累计解释率<input id="varianceThreshold" type="number" min="0.01" max="0.99" step="0.01" value="0.95"></label><label>主元数（可留空）<input id="components" type="number" min="2" placeholder="自动，至少2个"></label></div>
        <label>模型名称<input id="modelName" value="D330_DPCA_Model_V1"></label>
        <button id="qualityButton" class="secondary" disabled>执行统一数据质量检查</button>
        <div class="actions"><button id="trainExploratoryButton" class="secondary" disabled>建立探索模型</button><button id="trainButton" disabled>建立正常状态候选模型</button></div>
        <div class="notice">探索模型仅用于状态空间浏览和聚类辅助，不能作为正常状态模型。</div>
        <div class="notice">正常状态候选模型尚未验证，不能发布或用于部署。</div>
        <div class="notice">聚类结果必须由工程师判断，不能自动定义正常状态。</div>
        <div class="notice">探索模型和正常状态候选模型均不提供根因、因果或控制建议。</div>
      </div>
      <div id="status" class="status info" role="status" aria-live="polite">请先上传 CSV。</div>
      <div class="help">数据缺失、重复、乱序或采样间隔不一致时训练会停止，不会静默清洗。</div>
    </section>
    <section class="results">
      <div class="tabs" role="tablist">
        <button class="tab active" data-panel="configPanel">Tag配置</button>
        <button class="tab" data-panel="trendPanel">趋势浏览</button>
        <button class="tab" data-panel="statePanels">状态辅助</button>
        <button class="tab" data-panel="modelPanel">模型训练</button>
        <button class="tab" data-panel="validationPanel">验证结果</button>
      </div>
      <div id="configPanel" class="panel active">
        <div class="inner-tabs">
          <button class="inner-tab active" data-inner="engineeringPanel">工程配置</button>
          <button class="inner-tab" data-inner="qualityPanel">数据质量</button>
          <button class="inner-tab" data-inner="batchPanel">批量配置</button>
        </div>
        <div id="engineeringPanel" class="inner-panel active">
          <h3 id="selectedTagTitle">请选择左侧Tag</h3>
          <div class="detail-fields">
            <div class="row"><label>描述<input id="tagDescription"></label><label>单位<input id="tagUnit"></label></div>
            <div class="row"><label>变量角色<select id="tagRole"><option value="continuous_input">continuous_input</option><option value="state_filter">state_filter</option><option value="label_only">label_only</option><option value="exclude">exclude</option></select></label><label>备注<textarea id="tagComment"></textarea></label></div>
            <div class="row"><label>工程下限<input id="engineeringMin" type="number" step="any"></label><label>工程上限<input id="engineeringMax" type="number" step="any"></label></div>
            <div class="row"><label>正常下限<input id="normalMin" type="number" step="any"></label><label>正常上限<input id="normalMax" type="number" step="any"></label></div>
            <div class="row"><label>报警下限<input id="alarmMin" type="number" step="any"></label><label>报警上限<input id="alarmMax" type="number" step="any"></label></div>
            <button id="saveTagConfig" class="secondary">保存当前Tag配置</button>
          </div>
        </div>
        <div id="qualityPanel" class="inner-panel">
          <div id="qualitySummary" class="metrics"></div>
          <h3>当前Tag详情</h3>
          <div id="currentTagQuality" class="empty">尚未执行或结果已失效</div>
          <h3>全部问题Tag</h3>
          <div class="actions"><button id="excludeAllConstants" class="secondary" disabled>排除全部精确常量Tag</button></div>
          <div id="qualityIssues" class="empty">执行统一数据质量检查后，只显示需要确认或阻止训练的Tag。</div>
        </div>
        <div id="batchPanel" class="inner-panel">
          <div class="actions"><a id="templateDownload" class="download" href="#">下载XLSX模板</a><label class="secondary">导入XLSX配置<input id="tagConfigFile" type="file" accept=".xlsx"></label><button id="importConfigButton" class="secondary" disabled>预览导入</button><button id="applyConfigButton" disabled>确认应用非空字段</button><button id="exportConfigButton" class="secondary" disabled>导出当前配置</button></div>
          <div id="importSummary" class="status info">XLSX是可选工程元数据，导入不会跳过质量检查，也不会立即覆盖当前配置。</div>
        </div>
      </div>
      <div id="trendPanel" class="panel">
        <div class="trend-controls">
          <label>趋势Tag（最多8个）<select id="trendTags" multiple size="6"></select></label>
          <div><label>时间范围<select id="trendPreset"><option value="all">全部数据</option><option value="1">最近1天</option><option value="3">最近3天</option><option value="7">最近7天</option><option value="custom">自定义</option><option value="reference">参考状态期</option><option value="validation">验证期</option></select></label><div class="row"><label>开始<input id="trendStart" type="datetime-local"></label><label>结束<input id="trendEnd" type="datetime-local"></label></div></div>
          <div><label>显示<select id="trendMode"><option value="raw">原始值</option><option value="smoothed">平滑值</option><option value="both" selected>原始值和平滑值</option></select></label><label>缩放<input id="trendZoom" type="range" min="1" max="5" value="1"></label><button id="trendButton" class="secondary" disabled>浏览趋势</button></div>
        </div>
        <div class="actions"><button id="trendToAnalysis" class="secondary">将当前窗口设为分析期</button><button id="trendToReference" class="secondary">将当前窗口设为参考状态候选期</button><label>直方图范围<select id="histogramScope"><option value="current">当前窗口</option><option value="reference">参考期</option></select></label></div>
        <div id="trendChart" class="trend-chart"><div class="empty">选择Tag和时间范围后浏览原始值、尾随平滑、缺口及工程范围。</div></div>
        <div id="trendStats" class="table-wrap"></div>
        <div id="trendHistogram" class="chart"></div>
      </div>
      <div id="modelPanel" class="panel">
        <div id="modelEmpty" class="empty">完成训练后显示主元解释率、T²/SPE 和模型下载。</div>
        <div id="modelContent" hidden>
          <div id="modelMetrics" class="metrics"></div>
          <h3>训练窗口与连续段</h3><div id="trainingWindowSummary"></div>
          <div id="trainingQualityWarnings" class="hint"></div>
          <h3>主元解释率</h3><div id="varianceChart" class="variance"></div>
          <div class="chart-grid">
            <div class="chart-card"><h3>训练期 T²</h3><div id="t2Chart" class="chart"></div></div>
            <div class="chart-card"><h3>训练期 SPE/Q</h3><div id="speChart" class="chart"></div></div>
          </div>
          <div class="chart-card"><h3>主元得分 PC1 / PC2</h3><div id="scoreChart" class="chart"></div></div>
          <div class="legend"><span><i class="swatch" style="background:var(--accent)"></i>统计量</span><span><i class="swatch" style="background:var(--attention)"></i>95% 边界</span><span><i class="swatch" style="background:var(--abnormal)"></i>99% 边界</span></div>
          <div class="actions"><a id="modelDownload" class="download" href="#">下载模型包</a></div>
          <div class="notice">当前保存的是草稿模型。只有独立历史窗口回放并由工程师确认后，才能认为模型验证通过。</div>
        </div>
      </div>
      <div id="clusterPanel" class="panel">
        <div class="group">
          <div class="group-title">运行状态聚类辅助</div>
          <div class="row"><label>分析期开始<input id="analysisStart" type="datetime-local"></label><label>分析期结束<input id="analysisEnd" type="datetime-local"></label></div>
          <div class="row"><label>Cluster 数量<input id="clusterCount" type="number" min="2" max="10" value="3"></label><button id="clusterButton" class="secondary" disabled>生成运行状态聚类</button></div>
          <div class="help">聚类只辅助发现运行模式，不会自动认定正常状态。</div>
        </div>
        <div id="clusterEmpty" class="empty">检查数据后，可对选定历史窗口生成运行状态聚类。</div>
        <div id="clusterContent" hidden>
          <div id="clusterMetrics" class="metrics"></div>
          <div class="chart-card"><h3>聚类状态空间 PC1 / PC2</h3><div id="clusterChart" class="chart"></div></div>
          <h3>Cluster 概览与代表性连续时段</h3>
          <div class="table-wrap"><table><thead><tr><th>Cluster</th><th>样本</th><th>占比</th><th>中心 PC1 / PC2</th><th>人工选择正常候选时段</th></tr></thead><tbody id="clusterTable"></tbody></table></div>
          <div class="notice">Cluster 只表示数据中的相似运行状态，不代表正常或异常。只能将代表性连续时段加入候选，仍需工程师确认启用后再训练。</div>
        </div>
      </div>
      <div id="performancePanel" class="panel">
        <div class="group">
          <div class="group-title">性能条件筛选</div>
          <div id="performanceConditions" class="condition-list"><span class="help">检查数据后可添加筛选条件。</span></div>
          <div class="actions"><button id="addPerformanceCondition" class="secondary" disabled>添加条件</button><button id="performanceButton" class="secondary" disabled>筛选候选时段</button></div>
          <div class="help">全部条件按AND组合；性能列只用于筛选，不会自动进入PCA。</div>
        </div>
        <div id="performanceEmpty" class="empty">检查数据后，可使用透明的性能范围条件筛选候选时段。</div>
        <div id="performanceContent" hidden>
          <div id="performanceMetrics" class="metrics"></div>
          <h3>条件命中情况</h3>
          <div class="table-wrap"><table><thead><tr><th>性能列</th><th>条件</th><th>单条件命中</th></tr></thead><tbody id="performanceConditionTable"></tbody></table></div>
          <h3>代表性连续候选时段</h3>
          <div class="table-wrap"><table><thead><tr><th>开始</th><th>结束</th><th>样本数</th><th>人工选择</th></tr></thead><tbody id="performanceTable"></tbody></table></div>
          <div class="notice">性能条件仅用于标记优秀运行候选时段，不是预测模型，也不会自动认定正常状态。加入候选后仍须工程师明确启用。</div>
        </div>
      </div>
      <div id="validationPanel" class="panel">
        <div class="validation-box">
          <label>验证期开始<input id="validationStart" type="datetime-local"></label>
          <label>验证期结束<input id="validationEnd" type="datetime-local"></label>
          <label>工程标签列（可选）<select id="labelColumn"><option value="">不使用</option></select></label>
          <button id="validateButton" disabled>回放独立验证期</button>
        </div>
        <div id="validationEmpty" class="empty">训练草稿模型后可执行独立验证。</div>
        <div id="validationContent" hidden>
          <div id="validationMetrics" class="metrics"></div>
          <div class="chart-grid">
            <div class="chart-card"><h3>验证期 T²</h3><div id="validationT2Chart" class="chart"></div></div>
            <div class="chart-card"><h3>验证期 SPE/Q</h3><div id="validationSpeChart" class="chart"></div></div>
          </div>
          <h3>主要贡献 Tag</h3>
          <div class="help" id="contributionHint"></div>
          <div class="table-wrap"><table><thead><tr><th>事件 / 峰值</th><th>统计量</th><th>Tag</th><th>描述</th><th>单位</th><th>贡献</th><th>主要影响时间</th></tr></thead><tbody id="contributionTable"></tbody></table></div>
          <div class="actions"><a id="scoresDownload" class="download" href="#">下载完整评分 CSV</a><a id="reportDownload" class="download" href="#">下载验证摘要</a><a id="contributionsDownload" class="download" href="#">下载贡献记录</a></div>
          <div class="help">每次回放会更新当前模型最近一次验证的下载文件，不保存多次验证历史。</div>
          <div class="notice">贡献表示该时间点偏离在模型中的来源，不等同于工艺根因；最终通过或不通过由工程师确认。</div>
        </div>
      </div>
    </section>
  </main>
<script>
const state = { fileId:null, runId:null, exploratoryRunId:null, inspection:null, clustering:null, performance:null, training:null, trend:null, registry:{}, quality:null, selectedTag:null, selectedModelTags:new Set(), importPreview:null, excludedTags:[], showProblems:false, trainingWindows:[], trainingWindowSummary:[] };
const el = (id) => document.getElementById(id);

function setStatus(message, type="info") { const node=el("status"); node.textContent=message; node.className=`status ${type}`; }
function setBusy(button, busy, text) { if (!button.dataset.label) button.dataset.label=button.textContent; button.disabled=busy; button.textContent=busy?text:button.dataset.label; }
function localTime(value) { return value ? value.slice(0,16) : ""; }
function selectedTags() { return (state.inspection?.numeric_columns||[]).filter(tag=>state.selectedModelTags.has(tag)&&(state.registry[tag]?.role||"continuous_input")==="continuous_input"); }
function numberValue(id) { return Number(el(id).value); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch])); }
function formField(labelText,field,type="text") { const label=document.createElement("label"); label.textContent=labelText; const input=document.createElement("input"); input.type=type; input.dataset.field=field; if(type==="number") input.step="any"; label.append(input); return label; }
function emptyTagConfig() { return {description:"",unit:"",role:"continuous_input",engineering_min:null,engineering_max:null,normal_min:null,normal_max:null,alarm_min:null,alarm_max:null,comment:""}; }
function tagConfigPayload() { return state.registry; }
function qualityFor(tag) { return state.quality?.tags?.find(item=>item.tag===tag)||null; }
function renderTagList() {
  if(!state.inspection) return;
  const query=el("tagSearch").value.trim().toLowerCase(); const list=el("tagOptions"); list.replaceChildren();
  state.inspection.numeric_columns.forEach(tag=>{
    const quality=qualityFor(tag); const config=state.registry[tag]||emptyTagConfig(); const status=quality?.status||(config.role==="continuous_input"?"usable":"review");
    if(query&&!tag.toLowerCase().includes(query)) return;
    if(state.showProblems&&status==="usable") return;
    const row=document.createElement("div"); row.className=`tag-row ${state.selectedTag===tag?"selected":""}`; row.dataset.tag=tag;
    const input=document.createElement("input"); input.type="checkbox"; input.value=tag; input.checked=state.selectedModelTags.has(tag)&&config.role==="continuous_input";
    input.addEventListener("change",()=>{ if(input.checked) state.selectedModelTags.add(tag); else state.selectedModelTags.delete(tag); invalidateQuality("建模Tag已修改"); selectTag(tag); });
    const name=document.createElement("span"); name.textContent=tag;
    const badge=document.createElement("span"); badge.className=`tag-state ${status}`; badge.textContent=config.role!=="continuous_input"?config.role:(status==="blocking"?"阻止":status==="review"?"需确认":"正常");
    row.append(input,name,badge); row.addEventListener("click",event=>{ if(event.target!==input) selectTag(tag); }); list.append(row);
  });
}
function selectTag(tag) {
  state.selectedTag=tag; const config=state.registry[tag]||emptyTagConfig(); el("selectedTagTitle").textContent=tag;
  const fields={tagDescription:"description",tagUnit:"unit",tagRole:"role",tagComment:"comment",engineeringMin:"engineering_min",engineeringMax:"engineering_max",normalMin:"normal_min",normalMax:"normal_max",alarmMin:"alarm_min",alarmMax:"alarm_max"};
  Object.entries(fields).forEach(([id,key])=>{ el(id).value=config[key]??""; }); renderTagList(); renderCurrentTagQuality();
}
function optionalNumber(id) { const value=el(id).value.trim(); return value===""?null:Number(value); }
function saveCurrentTagConfig() {
  if(!state.selectedTag) throw new Error("请先选择Tag。");
  const tag=state.selectedTag; const config={description:el("tagDescription").value.trim(),unit:el("tagUnit").value.trim(),role:el("tagRole").value,comment:el("tagComment").value.trim(),engineering_min:optionalNumber("engineeringMin"),engineering_max:optionalNumber("engineeringMax"),normal_min:optionalNumber("normalMin"),normal_max:optionalNumber("normalMax"),alarm_min:optionalNumber("alarmMin"),alarm_max:optionalNumber("alarmMax")}; state.registry[tag]=config;
  if(config.role!=="continuous_input") state.selectedModelTags.delete(tag);
  invalidateQuality("Tag工程配置或角色已修改"); renderTagList();
}
function invalidateQuality(reason) { state.quality=null; el("trainButton").disabled=true; el("trainExploratoryButton").disabled=true; if(el("qualitySummary")) el("qualitySummary").innerHTML=metric("质量检查","已失效"); if(el("qualityIssues")) { el("qualityIssues").className="empty"; el("qualityIssues").textContent="尚未执行或结果已失效"; } el("excludeAllConstants").disabled=true; renderCurrentTagQuality(); if(reason) setStatus(`${reason}，请重新执行统一数据质量检查。`,"warning"); }
function commonPayload() { return {file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tag_configs:tagConfigPayload(),sample_interval_minutes:numberValue("sampleInterval"),smoothing_window_minutes:numberValue("smoothingWindow"),max_lag_minutes:numberValue("maxLag"),lag_step_minutes:numberValue("lagStep")}; }
function candidateId() { return globalThis.crypto?.randomUUID?.() || `window-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
function trainingWindowsPayload() { return state.trainingWindows; }
function windowSummary(id) { return state.trainingWindowSummary.find(item=>item.id===id)||{}; }
function showCandidateTrend(window) { el("trendStart").value=localTime(window.start); el("trendEnd").value=localTime(window.end); if(el("dpTrendStart")) el("dpTrendStart").value=localTime(window.start); if(el("dpTrendEnd")) el("dpTrendEnd").value=localTime(window.end); document.querySelector('[data-panel="trendPanel"]').click(); setStatus("已切换到候选时段趋势；训练候选未改变。","success"); }
function renderTrainingWindows() {
  const container=el("trainingWindows"); container.replaceChildren();
  if(!state.trainingWindows.length) { container.innerHTML='<div class="empty">尚无正常候选时段。</div>'; return; }
  const table=document.createElement("table"), head=document.createElement("thead"), body=document.createElement("tbody");
  const header=document.createElement("tr"); ["启用","来源","开始","结束","持续时间","原始 / 有效","质量","备注","操作"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  state.trainingWindows.forEach(window=>{ const summary=windowSummary(window.id); const row=document.createElement("tr");
    const enabled=document.createElement("input"); enabled.type="checkbox"; enabled.checked=window.enabled; enabled.addEventListener("change",()=>updateTrainingWindows({action:"set_enabled",id:window.id,enabled:enabled.checked},true)); const enabledCell=document.createElement("td"); enabledCell.append(enabled);
    const source=document.createElement("td"); source.textContent=window.source+(window.source_ref?` (${window.source_ref})`:"");
    const start=document.createElement("td"); start.textContent=localTime(window.start); const end=document.createElement("td"); end.textContent=localTime(window.end);
    const durationMinutes=summary.duration_minutes??Math.round((new Date(window.end)-new Date(window.start))/60000); const duration=document.createElement("td"); duration.textContent=`${durationMinutes} 分钟`;
    const counts=document.createElement("td"); counts.textContent=summary.raw_samples===undefined?"待检查":`${summary.raw_samples} / ${summary.effective_samples}`;
    const quality=document.createElement("td"); quality.textContent=summary.quality_status||summary.status||"待检查";
    const comment=document.createElement("td"); comment.textContent=window.comment||"—";
    const actions=document.createElement("td"); [
      ["查看趋势",()=>showCandidateTrend(window)],
      ["编辑",()=>editTrainingWindow(window)],
      ["删除",()=>updateTrainingWindows({action:"remove",id:window.id},window.enabled)],
    ].forEach(([label,handler])=>{ const button=document.createElement("button"); button.className="secondary"; button.type="button"; button.textContent=label; button.addEventListener("click",handler); actions.append(button); });
    row.append(enabledCell,source,start,end,duration,counts,quality,comment,actions); body.append(row);
  }); table.append(head,body); container.append(table);
}
async function updateTrainingWindows(operation, affectsTraining) {
  try { const data=await api("/api/training-windows",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...commonPayload(),training_windows:trainingWindowsPayload(),operation})}); state.trainingWindows=data.training_windows; state.trainingWindowSummary=data.summary; renderTrainingWindows(); if(affectsTraining) invalidateQuality("启用的正常候选时段已修改"); }
  catch(error) { renderTrainingWindows(); setStatus(error.message,"error"); }
}
function addCandidateWindow(source,start,end,sourceRef=null,comment="") { if(!start||!end) { setStatus("候选时段需要开始和结束时间。","warning"); return; } updateTrainingWindows({action:"add",window:{id:candidateId(),start,end,source,source_ref:sourceRef,enabled:false,comment}},false); }
function editTrainingWindow(window) { const start=prompt("候选开始时间",localTime(window.start)); if(start===null) return; const end=prompt("候选结束时间",localTime(window.end)); if(end===null) return; const comment=prompt("备注",window.comment||""); if(comment===null) return; updateTrainingWindows({action:"update",id:window.id,changes:{start,end,comment}},window.enabled); }
async function refreshTrainingWindows() { await updateTrainingWindows(null,false); }

async function api(path, options={}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败：${response.status}`);
  return data;
}

function fillSelect(node, values, blankLabel=null) {
  node.replaceChildren();
  if (blankLabel !== null) { const option=document.createElement("option"); option.value=""; option.textContent=blankLabel; node.append(option); }
  values.forEach(value=>{ const option=document.createElement("option"); option.value=value; option.textContent=value; node.append(option); });
}

function addPerformanceCondition() {
  const columns=state.inspection?.numeric_columns || []; if(!columns.length) return;
  const row=document.createElement("div"); row.className="condition-row";
  const columnLabel=document.createElement("label"); columnLabel.textContent="性能列"; const select=document.createElement("select"); select.className="performance-column"; columns.forEach(column=>{ const option=document.createElement("option"); option.value=column; option.textContent=column; select.append(option); }); columnLabel.append(select);
  const minimum=formField("下限（可空）","minimum","number"); const maximum=formField("上限（可空）","maximum","number");
  const remove=document.createElement("button"); remove.type="button"; remove.className="secondary"; remove.textContent="删除"; remove.addEventListener("click",()=>row.remove());
  row.append(columnLabel,minimum,maximum,remove); el("performanceConditions").append(row);
}
function renderPerformanceConditions(columns) { const list=el("performanceConditions"); list.replaceChildren(); if(columns.length) addPerformanceCondition(); }
function performanceConditionPayload() {
  const rows=[...document.querySelectorAll('#performanceConditions .condition-row')]; if(!rows.length) throw new Error("请至少添加一个性能条件。");
  return rows.map(row=>{ const minimum=row.querySelector('[data-field="minimum"]').value.trim(); const maximum=row.querySelector('[data-field="maximum"]').value.trim(); return {column:row.querySelector("select").value,minimum:minimum===""?null:Number(minimum),maximum:maximum===""?null:Number(maximum)}; });
}
function excludePerformanceColumns(conditions) { const columns=new Set(conditions.map(item=>item.column)); columns.forEach(tag=>state.selectedModelTags.delete(tag)); invalidateQuality("性能筛选列已从建模Tag取消"); renderTagList(); }

el("uploadButton").addEventListener("click", async () => {
  const file=el("fileInput").files[0]; if (!file) { setStatus("请选择 CSV 文件。","warning"); return; }
  const button=el("uploadButton"); setBusy(button,true,"上传中…");
  try {
    const form=new FormData(); form.append("file",file);
    const data=await api("/api/upload",{method:"POST",body:form});
    state.fileId=data.file_id; state.inspection=null; state.registry={}; state.quality=null; state.selectedTag=null; state.selectedModelTags.clear(); invalidateQuality(); fillSelect(el("timestampColumn"),data.columns); fillSelect(el("labelColumn"),data.columns,"不使用"); el("encoding").value=data.encoding;
    el("inspectButton").disabled=false; el("clusterButton").disabled=true; el("addPerformanceCondition").disabled=true; el("performanceButton").disabled=true; el("qualityButton").disabled=true; el("trendButton").disabled=true; el("trainButton").disabled=true; el("validateButton").disabled=true; el("importConfigButton").disabled=true; el("exportConfigButton").disabled=true;
    setStatus(`已上传 ${data.filename}，共 ${data.columns.length} 列。请选择时间列并检查数据。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("inspectButton").addEventListener("click", async () => {
  const button=el("inspectButton"); setBusy(button,true,"检查中…");
  try {
    const data=await api("/api/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value})});
    state.inspection=data; state.registry=Object.fromEntries(data.numeric_columns.map(tag=>[tag,emptyTagConfig()])); state.quality=null; state.selectedTag=null; state.excludedTags=[]; state.selectedModelTags=new Set(data.numeric_columns.filter(tag=>state.registry[tag].role==="continuous_input")); invalidateQuality(); renderPerformanceConditions(data.numeric_columns); renderTagList();
    fillSelect(el("trendTags"),data.numeric_columns); [...el("trendTags").options].slice(0,Math.min(3,data.numeric_columns.length)).forEach(option=>option.selected=true);
    el("analysisStart").value=localTime(data.time_start); el("analysisEnd").value=localTime(data.time_end); el("normalStart").value=localTime(data.time_start); el("normalEnd").value=localTime(data.suggested_normal_end); el("candidateStart").value=localTime(data.time_start); el("candidateEnd").value=localTime(data.suggested_normal_end); el("candidateComment").value=""; state.trainingWindows=[{id:"manual-window-001",start:el("normalStart").value,end:el("normalEnd").value,source:"manual",source_ref:null,enabled:true,comment:""}]; state.trainingWindowSummary=[]; await refreshTrainingWindows(); el("validationStart").value=localTime(data.suggested_validation_start); el("validationEnd").value=localTime(data.time_end);
    el("trendStart").value=localTime(data.time_start); el("trendEnd").value=localTime(data.time_end);
    if (data.sample_interval_minutes) el("sampleInterval").value=String(data.sample_interval_minutes);
    el("clusterButton").disabled=false; el("addPerformanceCondition").disabled=false; el("performanceButton").disabled=false; el("qualityButton").disabled=false; el("trendButton").disabled=false; el("importConfigButton").disabled=false; el("exportConfigButton").disabled=false;
    el("templateDownload").href=`/download/tag-config-template?file_id=${encodeURIComponent(state.fileId)}&timestamp_column=${encodeURIComponent(el("timestampColumn").value)}&encoding=${encodeURIComponent(el("encoding").value)}`;
    if(data.numeric_columns.length) selectTag(data.numeric_columns[0]);
    const issues=data.quality_issues.map(item=>`${item.code}(${item.count}) ${item.tag||""}`).join("、");
    setStatus(issues ? `初步检查完成：${data.rows} 行。发现 ${issues}；选择参考期后必须执行统一质量检查。` : `初步检查完成：${data.rows} 行，识别 ${data.numeric_columns.length} 个数值列。请选择参考期并执行统一质量检查。`, issues?"warning":"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("tagSearch").addEventListener("input",renderTagList);
el("selectAllTags").addEventListener("click",()=>{ state.selectedModelTags=new Set((state.inspection?.numeric_columns||[]).filter(tag=>(state.registry[tag]?.role||"continuous_input")==="continuous_input")); invalidateQuality("建模Tag已修改"); renderTagList(); });
el("clearAllTags").addEventListener("click",()=>{ state.selectedModelTags.clear(); invalidateQuality("建模Tag已修改"); renderTagList(); });
el("showProblemTags").addEventListener("click",()=>{ state.showProblems=!state.showProblems; el("showProblemTags").textContent=state.showProblems?"显示全部Tag":"只看问题Tag"; renderTagList(); });
el("saveTagConfig").addEventListener("click",()=>{ try { saveCurrentTagConfig(); } catch(error) { setStatus(error.message,"error"); } });
document.querySelectorAll(".inner-tab").forEach(button=>button.addEventListener("click",()=>{ document.querySelectorAll(".inner-tab").forEach(node=>node.classList.toggle("active",node===button)); document.querySelectorAll(".inner-panel").forEach(panel=>panel.classList.toggle("active",panel.id===button.dataset.inner)); }));
["sampleInterval","smoothingWindow","maxLag","lagStep"].forEach(id=>el(id).addEventListener("change",()=>invalidateQuality("预处理参数已修改")));
el("addManualCandidate").addEventListener("click",()=>addCandidateWindow("manual",el("candidateStart").value,el("candidateEnd").value,null,el("candidateComment").value.trim()));

el("tagConfigFile").addEventListener("change",()=>{ el("importConfigButton").disabled=!el("tagConfigFile").files[0]||!state.fileId; });
el("importConfigButton").addEventListener("click",async()=>{
  const file=el("tagConfigFile").files[0]; if(!file) return;
  const button=el("importConfigButton"); setBusy(button,true,"读取中…");
  try {
    const form=new FormData(); form.append("file",file);
    const query=`?file_id=${encodeURIComponent(state.fileId)}&timestamp_column=${encodeURIComponent(el("timestampColumn").value)}&encoding=${encodeURIComponent(el("encoding").value)}`;
    const data=await api(`/api/tag-config/import${query}`,{method:"POST",body:form}); state.importPreview=data; el("applyConfigButton").disabled=!data.can_apply;
    const messages=[...data.errors,...data.warnings]; el("importSummary").textContent=`匹配 ${data.matched_tags.length}；数据未配置 ${data.unconfigured_data_tags.length}；模板未知 ${data.unknown_template_tags.length}；重复 ${data.duplicate_tags.length}${messages.length?`。\n${messages.join("\n")}`:"。请确认后应用，当前配置尚未改变。"}`; el("importSummary").className=`status ${data.can_apply?(data.warnings.length?"warning":"success"):"error"}`;
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});
el("applyConfigButton").addEventListener("click",()=>{
  if(!state.importPreview?.can_apply) return;
  const overwrites=Object.entries(state.importPreview.provided_configs).some(([tag,config])=>Object.entries(config).some(([key,value])=>value!==null&&value!==""&&state.registry[tag]?.[key]!==null&&state.registry[tag]?.[key]!==""));
  if(overwrites&&!confirm("导入内容将覆盖页面已有的非空字段，是否继续？")) return;
  Object.entries(state.importPreview.provided_configs).forEach(([tag,config])=>{ Object.entries(config).forEach(([key,value])=>{ if(value!==null&&value!=="") state.registry[tag][key]=value; }); });
  for(const tag of [...state.selectedModelTags]) { if((state.registry[tag]?.role||"continuous_input")!=="continuous_input") state.selectedModelTags.delete(tag); }
  state.importPreview=null; el("applyConfigButton").disabled=true; if(state.selectedTag) selectTag(state.selectedTag); invalidateQuality("XLSX工程配置已应用"); renderTagList();
});
el("exportConfigButton").addEventListener("click",async()=>{
  try {
    const response=await fetch("/api/tag-config/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(commonPayload())});
    if(!response.ok) { const data=await response.json(); throw new Error(data.error||"导出失败"); }
    const link=document.createElement("a"); link.href=URL.createObjectURL(await response.blob()); link.download="PCA_Tag_Config.xlsx"; link.click(); URL.revokeObjectURL(link.href);
  } catch(error) { setStatus(error.message,"error"); }
});

el("qualityButton").addEventListener("click",async()=>{
  const tags=selectedTags(); if(tags.length<2) { setStatus("至少选择两个continuous_input Tag。","warning"); return; }
  const button=el("qualityButton"); setBusy(button,true,"检查中…");
  try {
    const payload={...commonPayload(),tags,training_windows:trainingWindowsPayload()};
    const data=await api("/api/quality",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); state.quality=data; state.trainingWindowSummary=data.training_window_summary||state.trainingWindowSummary; renderTrainingWindows(); renderQuality(data); renderTagList(); el("trainButton").disabled=!data.can_train; el("trainExploratoryButton").disabled=!data.can_train;
    document.querySelector('[data-panel="configPanel"]').click(); document.querySelector('[data-inner="qualityPanel"]').click();
    setStatus(data.can_train?"统一质量检查通过，可以训练草稿模型。":"仍有阻止训练的问题，请排除问题Tag或调整参考期后重新检查。",data.can_train?"success":"error");
  } catch(error) { setStatus(error.message,"error"); el("trainButton").disabled=true; el("trainExploratoryButton").disabled=true; }
  finally { setBusy(button,false,""); }
});

el("trendPreset").addEventListener("change",()=>{
  if(!state.inspection) return; const preset=el("trendPreset").value; const end=new Date(state.inspection.time_end);
  if(preset==="all") { el("trendStart").value=localTime(state.inspection.time_start); el("trendEnd").value=localTime(state.inspection.time_end); }
  else if(["1","3","7"].includes(preset)) { el("trendEnd").value=localTime(state.inspection.time_end); el("trendStart").value=localTime(new Date(end.getTime()-Number(preset)*86400000).toISOString()); }
  else if(preset==="reference") { const window=state.trainingWindows.find(item=>item.enabled)||state.trainingWindows[0]; if(window) { el("trendStart").value=localTime(window.start); el("trendEnd").value=localTime(window.end); } }
  else if(preset==="validation") { el("trendStart").value=el("validationStart").value; el("trendEnd").value=el("validationEnd").value; }
});
el("trendButton").addEventListener("click",async()=>{
  const tags=[...el("trendTags").selectedOptions].map(option=>option.value); if(!tags.length||tags.length>8) { setStatus("趋势浏览请选择1至8个Tag。","warning"); return; }
  const button=el("trendButton"); setBusy(button,true,"读取中…");
  try {
    const payload={...commonPayload(),tags,start:el("trendStart").value,end:el("trendEnd").value,display_mode:el("trendMode").value};
    const data=await api("/api/trend",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); state.trend=data; renderTrend(data); setStatus("趋势与统计已更新；数据未被修改或插值。","success");
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});
el("trendZoom").addEventListener("input",()=>{ el("trendChart").querySelectorAll("svg").forEach(svg=>svg.style.width=`${760*Number(el("trendZoom").value)}px`); });
el("trendToAnalysis").addEventListener("click",()=>{ el("analysisStart").value=el("trendStart").value; el("analysisEnd").value=el("trendEnd").value; setStatus("当前趋势窗口已设置为分析期。","success"); });
el("trendToReference").addEventListener("click",()=>addCandidateWindow("trend",el("trendStart").value,el("trendEnd").value,"trend-current",""));
el("histogramScope").addEventListener("change",()=>{ if(state.trend) renderHistogram(state.trend.histograms?.[el("histogramScope").value]); });

el("addPerformanceCondition").addEventListener("click",addPerformanceCondition);

el("performanceButton").addEventListener("click", async () => {
  const button=el("performanceButton"); setBusy(button,true,"筛选中…"); setStatus("正在按全部性能条件筛选连续候选时段。","info");
  try {
    const payload={file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,sample_interval_minutes:numberValue("sampleInterval"),conditions:performanceConditionPayload()};
    const data=await api("/api/performance-screen",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.performance=data; excludePerformanceColumns(data.conditions); renderPerformance(data); document.querySelector('[data-panel="statePanels"]').click();
    setStatus("性能条件筛选完成；相关性能列已取消建模勾选。请选择候选时段并由工程师确认工况。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("clusterButton").addEventListener("click", async () => {
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el("clusterButton"); setBusy(button,true,"聚类中…"); setStatus("正在构建动态状态空间并执行聚类。","info");
  try {
    const payload=state.exploratoryRunId?{file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,exploratory_run_id:state.exploratoryRunId,analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,n_clusters:numberValue("clusterCount")}:{file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tags,tag_configs:tagConfigPayload(tags),analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,sample_interval_minutes:numberValue("sampleInterval"),smoothing_window_minutes:numberValue("smoothingWindow"),max_lag_minutes:numberValue("maxLag"),lag_step_minutes:numberValue("lagStep"),variance_threshold:numberValue("varianceThreshold"),n_clusters:numberValue("clusterCount")};
    const data=await api("/api/cluster",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.clustering=data; renderClustering(data); document.querySelector('[data-panel="statePanels"]').click();
    setStatus("聚类完成。请由工程师判断 Cluster，并选择代表性连续时段作为正常候选。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

async function trainModel(modelPurpose) {
  if(!state.quality?.can_train) { setStatus("训练前必须重新执行并通过统一数据质量检查。","error"); return; }
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el(modelPurpose==="exploratory"?"trainExploratoryButton":"trainButton"); setBusy(button,true,"训练中…"); setStatus("正在构建动态矩阵并训练 DPCA，请勿关闭页面。","info");
  try {
    const components=el("components").value.trim();
    const payload={...commonPayload(),tags,excluded_tags:state.excludedTags,model_purpose:modelPurpose,training_windows:trainingWindowsPayload(),variance_threshold:numberValue("varianceThreshold"),n_components:components?Number(components):null,model_name:el("modelName").value};
    const data=await api("/api/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.runId=data.run_id; if(data.model_purpose==="exploratory") state.exploratoryRunId=data.run_id; state.training=data; renderTraining(data); el("validateButton").disabled=data.model_purpose==="exploratory"; document.querySelector('[data-panel="modelPanel"]').click();
    setStatus(`训练完成：${data.training_rows} 个动态样本，${data.dynamic_features} 个动态特征。当前为${data.model_purpose==="exploratory"?"探索草稿":"正常状态候选"}。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
}
el("trainExploratoryButton").addEventListener("click",()=>trainModel("exploratory"));
el("trainButton").addEventListener("click",()=>trainModel("normal_state"));

el("validateButton").addEventListener("click", async () => {
  const button=el("validateButton"); setBusy(button,true,"回放中…"); setStatus("正在使用训练参数回放独立验证窗口。","info");
  try {
    const payload={run_id:state.runId,file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,validation_start:el("validationStart").value,validation_end:el("validationEnd").value,label_column:el("labelColumn").value};
    const data=await api("/api/validate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    renderValidation(data); setStatus("独立窗口回放完成。请结合已知事件由工程师确认模型是否通过。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

function metric(label,value) { return `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`; }
function qualityProfileTable(title,profile) {
  const fields=[["sample_count","样本数"],["valid_count","有效数"],["missing_count","缺失数"],["missing_rate","缺失率"],["non_numeric_count","非数值数"],["non_finite_count","非有限值数"],["unique_count","唯一值"],["minimum","最小值"],["maximum","最大值"],["mean","均值"],["median","中位数"],["standard_deviation","标准差"],["p01","P1"],["p05","P5"],["p95","P95"],["p99","P99"],["engineering_range_outside_count","工程范围越界"],["normal_range_outside_count","正常范围外"],["alarm_range_outside_count","报警范围外"]];
  return `<h4>${title}</h4><div class="table-wrap"><table><tbody>${fields.map(([key,label])=>`<tr><th>${label}</th><td>${formatStat(key,profile[key])}</td></tr>`).join("")}</tbody></table></div>`;
}
function renderCurrentTagQuality() {
  const container=el("currentTagQuality"); if(!container) return;
  const item=state.selectedTag?qualityFor(state.selectedTag):null;
  if(!state.quality||!item) { container.className="empty"; container.textContent="尚未执行或结果已失效"; return; }
  const role=state.registry[item.tag]?.role||item.role; const issueHtml=item.issues.length?item.issues.map(issue=>`<li>${escapeHtml(issue.message)}</li>`).join(""):"<li>无质量问题</li>";
  container.className=""; container.innerHTML=`<div class="issue-card ${item.status}"><strong>${escapeHtml(item.tag)} · ${escapeHtml(role)} · ${escapeHtml(item.status)}</strong>${qualityProfileTable("全数据统计",item.full)}${qualityProfileTable("参考期统计",item.reference)}<h4>质量问题与建议</h4><ul>${issueHtml}</ul><span>建议操作：${escapeHtml(item.suggested_action)}</span></div>`;
}
function renderQuality(data) {
  el("qualitySummary").innerHTML=metric("可直接使用",data.summary.usable)+metric("需要确认",data.summary.review)+metric("阻止训练",data.summary.blocking)+metric("训练条件",data.can_train?"通过":"未通过");
  const container=el("qualityIssues"); container.className=""; container.replaceChildren();
  data.time_issues.forEach(issue=>{ const card=document.createElement("div"); card.className=`issue-card ${issue.severity==="error"?"blocking":""}`; card.innerHTML=`<strong>${escapeHtml(issue.code)}</strong><span>${escapeHtml(issue.message)}</span>`; container.append(card); });
  const problemTags=data.tags.filter(item=>item.status!=="usable"); problemTags.forEach(item=>{
    const card=document.createElement("div"); card.className=`issue-card ${item.status}`; const profile=item.reference;
    card.innerHTML=`<strong>${escapeHtml(item.tag)} · ${escapeHtml(item.status)}</strong><span>参考期样本 ${profile.sample_count}；有效 ${profile.valid_count}；唯一值 ${profile.unique_count}；标准差 ${profile.standard_deviation??"—"}</span>${item.issues.map(issue=>`<span>${escapeHtml(issue.message)}</span>`).join("")}`;
    const actions=document.createElement("div"); actions.className="actions";
    if(item.issues.some(issue=>issue.code==="constant_tag")) { const exclude=document.createElement("button"); exclude.className="secondary"; exclude.textContent="从本次模型排除"; exclude.addEventListener("click",()=>excludeConstantTag(item)); actions.append(exclude); }
    const trend=document.createElement("button"); trend.className="secondary"; trend.textContent="在趋势页查看"; trend.addEventListener("click",()=>{ [...el("trendTags").options].forEach(option=>option.selected=option.value===item.tag); el("trendStart").value=el("normalStart").value; el("trendEnd").value=el("normalEnd").value; document.querySelector('[data-panel="trendPanel"]').click(); }); actions.append(trend); card.append(actions); container.append(card);
  });
  el("excludeAllConstants").disabled=!data.tags.some(item=>item.issues.some(issue=>issue.code==="constant_tag"));
  if(!container.children.length) { container.className="empty"; container.textContent="所选Tag和时间轴没有需要处理的问题。"; }
  renderCurrentTagQuality();
}
function excludeConstantTag(item) {
  const issue=item.issues.find(value=>value.code==="constant_tag"); if(!issue) return;
  state.registry[item.tag].role="exclude"; state.excludedTags=state.excludedTags.filter(record=>record.tag!==item.tag); state.excludedTags.push({tag:item.tag,reason:"constant_in_reference_window",sample_count:issue.details.valid_count,unique_count:1,constant_value:issue.details.constant_value});
  state.selectedModelTags.delete(item.tag);
  if(state.selectedTag===item.tag) selectTag(item.tag); invalidateQuality(`${item.tag}已标记为排除`); renderTagList();
}
el("excludeAllConstants").addEventListener("click",()=>{
  const constants=state.quality?.tags.filter(item=>item.issues.some(issue=>issue.code==="constant_tag"))||[]; if(!constants.length) return;
  constants.forEach(item=>{ const issue=item.issues.find(value=>value.code==="constant_tag"); state.registry[item.tag].role="exclude"; state.excludedTags=state.excludedTags.filter(record=>record.tag!==item.tag); state.excludedTags.push({tag:item.tag,reason:"constant_in_reference_window",sample_count:issue.details.valid_count,unique_count:1,constant_value:issue.details.constant_value}); state.selectedModelTags.delete(item.tag); });
  invalidateQuality(`已标记排除 ${constants.length} 个精确常量Tag`); renderTagList();
});
function renderTrend(data) {
  const container=el("trendChart"); container.replaceChildren(); const zoom=Number(el("trendZoom").value);
  data.tags.forEach((tag,index)=>{ const card=document.createElement("div"); card.className="chart-card"; const title=document.createElement("h3"); title.textContent=`${tag}${state.registry[tag]?.unit?` (${state.registry[tag].unit})`:""}`; card.append(title); card.insertAdjacentHTML("beforeend",trendSvg(data,tag,index,zoom)); container.append(card); });
  const fields=[["sample_count","样本"],["valid_count","有效"],["missing_rate","缺失率"],["unique_count","唯一值"],["minimum","最小"],["maximum","最大"],["mean","均值"],["median","中位数"],["standard_deviation","标准差"],["p01","P1"],["p05","P5"],["p95","P95"],["p99","P99"]];
  let html="<table><thead><tr><th>Tag</th><th>范围</th>"+fields.map(item=>`<th>${item[1]}</th>`).join("")+"</tr></thead><tbody>";
  data.tags.forEach(tag=>{ ["full","current","reference"].forEach(scope=>{ const profile=data.statistics[tag][scope]; if(!profile) return; html+=`<tr><td>${escapeHtml(tag)}</td><td>${scope==="full"?"全数据":scope==="current"?"当前窗口":"参考期"}</td>${fields.map(([key])=>`<td>${formatStat(key,profile[key])}</td>`).join("")}</tr>`; }); }); el("trendStats").innerHTML=html+"</tbody></table>";
  renderHistogram(data.histograms?.[el("histogramScope").value]||data.histogram);
}
function formatStat(key,value) { if(value===null||value===undefined) return "—"; if(key==="missing_rate") return `${(Number(value)*100).toFixed(2)}%`; return typeof value==="number"?Number(value).toPrecision(5):escapeHtml(value); }
function trendSvg(data,tag,colorIndex,zoom) {
  const rows=data.rows,width=760*zoom,height=230,pad={l:48,r:16,t:16,b:28}; const fields=[]; if(data.display_mode!=="smoothed") fields.push([`${tag}__raw`,"#176b87","原始"]); if(data.display_mode!=="raw") fields.push([`${tag}__smoothed`,"#d97706","平滑"]);
  const ranges=data.ranges[tag]||{}; const limits=data.axis_limits[tag]; const minimum=limits.minimum,maximum=limits.maximum,span=maximum-minimum; const x=i=>pad.l+i/Math.max(1,rows.length-1)*(width-pad.l-pad.r),y=value=>height-pad.b-(Number(value)-minimum)/span*(height-pad.t-pad.b);
  const polylines=fields.map(([field,color,label])=>{ let segments=[],current=[]; rows.forEach((row,i)=>{ if(row[field]===null||row.gap_start) { if(current.length) segments.push(current); current=[]; } if(row[field]!==null) current.push(`${x(i).toFixed(1)},${y(row[field]).toFixed(1)}`); }); if(current.length) segments.push(current); const lines=segments.map(points=>`<polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="1.7"/>`).join(""); const dots=rows.map((row,i)=>{ if(row[field]===null) return `<circle cx="${x(i)}" cy="${height-pad.b}" r="3" fill="#cf3f36"><title>${escapeHtml(row.timestamp)} · 缺失值</title></circle>`; const outside=field.endsWith("__raw")&&ranges.engineering_min!==null&&ranges.engineering_min!==undefined&&(Number(row[field])<Number(ranges.engineering_min)||Number(row[field])>Number(ranges.engineering_max)); return `<circle cx="${x(i)}" cy="${y(row[field])}" r="${outside?3:2}" fill="${outside?"#cf3f36":color}"><title>${escapeHtml(row.timestamp)} · ${label}: ${row[field]}${outside?" · 工程量程越界":""}</title></circle>`; }).join(""); return lines+dots; }).join("");
  const rangeLine=(key,color,label)=>ranges[key]===null||ranges[key]===undefined?"":`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(ranges[key])}" y2="${y(ranges[key])}" stroke="${color}" stroke-dasharray="5 4"><title>${label}: ${ranges[key]}</title></line>`;
  const gaps=rows.map((row,i)=>row.gap_start?`<line x1="${x(i)}" x2="${x(i)}" y1="${pad.t}" y2="${height-pad.b}" stroke="#cf3f36" stroke-dasharray="3 3"><title>物理时间缺口</title></line>`:"").join("");
  return `<svg viewBox="0 0 ${width} ${height}" style="width:${width}px" aria-label="${escapeHtml(tag)}趋势">${gaps}${rangeLine("engineering_min","#64748b","工程下限")}${rangeLine("engineering_max","#64748b","工程上限")}${rangeLine("normal_min","#16845b","正常下限")}${rangeLine("normal_max","#16845b","正常上限")}${rangeLine("alarm_min","#cf3f36","报警下限")}${rangeLine("alarm_max","#cf3f36","报警上限")}${polylines}<text x="4" y="20" font-size="10">${maximum.toPrecision(4)}</text><text x="4" y="${height-pad.b}" font-size="10">${minimum.toPrecision(4)}</text></svg>`;
}
function renderHistogram(histogram) {
  const container=el("trendHistogram"); if(!histogram||!histogram.counts.length) { container.innerHTML='<div class="empty">单Tag且存在有效值时显示直方图。</div>'; return; }
  const width=760,height=250,pad=30,max=Math.max(...histogram.counts,1),barWidth=(width-pad*2)/histogram.counts.length; const bars=histogram.counts.map((count,index)=>`<rect x="${pad+index*barWidth}" y="${height-pad-count/max*(height-pad*2)}" width="${Math.max(1,barWidth-1)}" height="${count/max*(height-pad*2)}" fill="#176b87"><title>${histogram.edges[index].toPrecision(4)}～${histogram.edges[index+1].toPrecision(4)}: ${count}</title></rect>`).join(""); container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" aria-label="${escapeHtml(histogram.tag)}分布直方图">${bars}</svg>`;
}
function renderPerformance(data) {
  el("performanceEmpty").hidden=true; el("performanceContent").hidden=false;
  el("performanceMetrics").innerHTML=metric("分析样本",data.total_rows)+metric("全部条件命中",data.matched_rows)+metric("命中占比",`${(data.match_share*100).toFixed(1)}%`)+metric("组合方式","AND");
  const conditions=el("performanceConditionTable"); conditions.replaceChildren(); data.conditions.forEach(item=>{ const tr=document.createElement("tr"); const expression=`${item.minimum===null?"":`≥ ${item.minimum}`} ${item.maximum===null?"":`≤ ${item.maximum}`}`.trim(); [item.column,expression,item.matched_rows].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; tr.append(td); }); conditions.append(tr); });
  const windows=el("performanceTable"); windows.replaceChildren(); data.representative_windows.forEach((window,index)=>{ const tr=document.createElement("tr"); [window.start.slice(0,16),window.end.slice(0,16),window.count].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; tr.append(td); }); const action=document.createElement("td"); const button=document.createElement("button"); button.className="secondary"; button.textContent="加入正常候选"; button.addEventListener("click",()=>addCandidateWindow("performance",window.start,window.end,`performance-${index+1}`,"")); action.append(button); tr.append(action); windows.append(tr); });
  if(!data.representative_windows.length) { const tr=document.createElement("tr"); const td=document.createElement("td"); td.colSpan=4; td.textContent="没有同时满足全部条件的连续时段。"; tr.append(td); windows.append(tr); }
}

function renderClustering(data) {
  el("clusterEmpty").hidden=true; el("clusterContent").hidden=false;
  el("clusterMetrics").innerHTML=metric("聚类动态样本",data.sample_count)+metric("状态空间主元",data.n_components)+metric("累计解释率",`${(data.cumulative_explained_variance*100).toFixed(1)}%`)+metric("Cluster 数量",data.clusters.length);
  clusterScatter(el("clusterChart"),data.points);
  const body=el("clusterTable"); body.replaceChildren();
  data.clusters.forEach(item=>{
    const tr=document.createElement("tr");
    [`Cluster ${item.cluster}`,item.count,`${(item.share*100).toFixed(1)}%`,`${item.pc1_center.toFixed(2)} / ${item.pc2_center.toFixed(2)}`].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; tr.append(td); });
    const windows=document.createElement("td");
    item.representative_windows.forEach(window=>{ const button=document.createElement("button"); button.className="secondary"; button.style.margin="2px"; button.textContent=`加入候选：${window.start.slice(0,16)} ～ ${window.end.slice(11,16)} (${window.count}点)`; button.addEventListener("click",()=>addCandidateWindow("cluster",window.start,window.end,`cluster-${item.cluster}`,"")); windows.append(button); });
    tr.append(windows); body.append(tr);
  });
}

function renderTraining(data) {
  el("modelEmpty").hidden=true; el("modelContent").hidden=false;
  const purpose=data.model_purpose==="exploratory"?"探索模型":"正常状态模型"; const status=data.model_status==="draft"?"草稿":"候选";
  el("modelMetrics").innerHTML=metric("模型用途",purpose)+metric("模型状态",status)+metric("训练动态样本",data.training_rows)+metric("动态特征",data.dynamic_features)+metric("主元数",data.n_components)+metric("累计解释率",`${(data.cumulative_explained_variance*100).toFixed(1)}%`)+metric("关注 / 异常",`${data.status_counts.attention} / ${data.status_counts.abnormal}`);
  renderTrainingWindowSummary(data.training_window_summary||[]);
  const warnings=data.training_quality_warnings||[]; el("trainingQualityWarnings").textContent=warnings.length?`注意：${warnings.map(item=>`${item.feature} 全局变化极小`).join("；")}`:"";
  const variance=el("varianceChart"); variance.replaceChildren(); const max=Math.max(...data.explained_variance,0.01);
  data.explained_variance.slice(0,30).forEach((value,index)=>{ const bar=document.createElement("div"); bar.className=`variance-bar ${index<data.n_components?"selected":""}`; bar.style.height=`${Math.max(3,value/max*95)}px`; const label=document.createElement("span"); label.textContent=`${(value*100).toFixed(0)}%`; bar.title=`PC${index+1}: ${(value*100).toFixed(2)}%`; bar.append(label); variance.append(bar); });
  lineChart(el("t2Chart"),data.scores,"t2",data.t2_limits,"T²"); lineChart(el("speChart"),data.scores,"spe",data.q_limits,"SPE"); scoreScatter(el("scoreChart"),data.scores); el("modelDownload").href=data.model_download;
}

function renderTrainingWindowSummary(windows) {
  const container=el("trainingWindowSummary"); container.replaceChildren();
  if(!windows.length) { container.textContent="未提供训练窗口摘要。"; return; }
  const table=document.createElement("table"), head=document.createElement("thead"), body=document.createElement("tbody");
  const header=document.createElement("tr"); ["范围","状态","原始样本","有效动态样本","平滑/Lag 损失","原因"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  windows.forEach(window=>{
    const row=document.createElement("tr");
    [`窗口 ${window.id}: ${window.start.slice(0,16)} ～ ${window.end.slice(0,16)}`,window.status,window.raw_samples,window.effective_samples,window.smoothing_lag_loss??"—",window.dropped_reason??"—"].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; row.append(td); }); body.append(row);
    (window.segments||[]).forEach(segment=>{ const segmentRow=document.createElement("tr");
      [`连续段 ${segment.start.slice(0,16)} ～ ${segment.end.slice(0,16)}`,segment.status,segment.raw_samples,segment.effective_samples,segment.smoothing_lag_loss,segment.dropped_reason??"—"].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; segmentRow.append(td); }); body.append(segmentRow);
    });
  });
  table.append(head,body); container.append(table);
}

function renderValidation(data) {
  el("validationEmpty").hidden=true; el("validationContent").hidden=false;
  el("validationMetrics").innerHTML=metric("验证样本",data.scored_rows)+metric("正常",data.status_counts.normal)+metric("关注",data.status_counts.attention)+metric("异常",data.status_counts.abnormal)+metric("模型状态（草稿）","待工程确认");
  lineChart(el("validationT2Chart"),data.scores,"t2",data.t2_limits,"T²"); lineChart(el("validationSpeChart"),data.scores,"spe",data.q_limits,"SPE");
  el("contributionHint").textContent=data.contributions.length ? "按每个连续越过95%控制限的事件保存峰值贡献；事件不会跨物理时间缺口合并。" : "T² 和 SPE 均未达到 95% 控制限，不输出异常贡献。";
  const body=el("contributionTable"); body.innerHTML="";
  data.contributions.forEach(group=>group.tags.forEach(item=>{ const tr=document.createElement("tr"); const lag=item.lag_start_minutes===item.lag_end_minutes?`${item.lag_start_minutes} 分钟前`:`${item.lag_start_minutes}～${item.lag_end_minutes} 分钟前`; const event=`${group.event_start.slice(0,16)} ～ ${group.event_end.slice(11,16)}；峰值 ${group.peak_timestamp.slice(11,16)}`; tr.innerHTML=`<td>${escapeHtml(event)}</td><td>${escapeHtml(group.statistic.toUpperCase())}</td><td title="点击在趋势页查看">${escapeHtml(item.tag)}</td><td>${escapeHtml(item.description)}</td><td>${escapeHtml(item.unit)}</td><td class="numeric">${item.contribution_pct.toFixed(1)}%</td><td>${escapeHtml(lag)}</td>`; tr.children[2].style.cursor="pointer"; tr.children[2].addEventListener("click",()=>{ [...el("trendTags").options].forEach(option=>option.selected=option.value===item.tag); el("trendStart").value=localTime(group.event_start); el("trendEnd").value=localTime(group.event_end); document.querySelector('[data-panel="trendPanel"]').click(); }); body.append(tr); }));
  el("scoresDownload").href=data.validation_downloads.scores; el("reportDownload").href=data.validation_downloads.report; el("contributionsDownload").href=data.validation_downloads.contributions;
}

function lineChart(container, rows, field, limits, label) {
  if (!rows.length) { container.innerHTML='<div class="empty">无可展示数据</div>'; return; }
  const width=760,height=250,pad={l:48,r:16,t:15,b:30}; const values=rows.map(row=>Number(row[field])); const max=Math.max(...values,Number(limits["99"])*1.08,1e-9); const x=index=>pad.l+index/Math.max(1,rows.length-1)*(width-pad.l-pad.r); const y=value=>height-pad.b-value/max*(height-pad.t-pad.b); const points=values.map((value,index)=>`${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const limitLine=(value,color,text)=>`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(value)}" y2="${y(value)}" stroke="${color}" stroke-width="1.5" stroke-dasharray="6 4"/><text x="${width-pad.r-3}" y="${y(value)-4}" text-anchor="end" fill="${color}" font-size="10">${text}</text>`;
  container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}趋势"><line x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${height-pad.b}" stroke="#bcc6d1"/><line x1="${pad.l}" x2="${width-pad.r}" y1="${height-pad.b}" y2="${height-pad.b}" stroke="#bcc6d1"/>${limitLine(limits["95"],"#d19a20","95%")}${limitLine(limits["99"],"#cf3f36","99%")}<polyline points="${points}" fill="none" stroke="#176b87" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="4" y="${pad.t+8}" fill="#5f6c7b" font-size="10">${max.toFixed(2)}</text><text x="${pad.l}" y="${height-8}" fill="#5f6c7b" font-size="10">${escapeHtml(rows[0].timestamp.slice(0,16))}</text><text x="${width-pad.r}" y="${height-8}" text-anchor="end" fill="#5f6c7b" font-size="10">${escapeHtml(rows.at(-1).timestamp.slice(0,16))}</text></svg>`;
}

function scoreScatter(container, rows) {
  if (!rows.length || !("pc1" in rows[0]) || !("pc2" in rows[0])) { container.innerHTML='<div class="empty">当前模型不足两个保留主元。</div>'; return; }
  const width=760,height=250,pad=28; const xs=rows.map(row=>Number(row.pc1)),ys=rows.map(row=>Number(row.pc2)); const maxX=Math.max(...xs.map(Math.abs),1e-9),maxY=Math.max(...ys.map(Math.abs),1e-9); const x=value=>width/2+value/maxX*(width/2-pad); const y=value=>height/2-value/maxY*(height/2-pad); const colors={normal:"#16845b",attention:"#d19a20",abnormal:"#cf3f36"}; const circles=rows.map(row=>`<circle cx="${x(Number(row.pc1))}" cy="${y(Number(row.pc2))}" r="3" fill="${colors[row.status]}" fill-opacity=".72"/>`).join(""); container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PC1与PC2得分散点"><line x1="${pad}" x2="${width-pad}" y1="${height/2}" y2="${height/2}" stroke="#d7dee8"/><line x1="${width/2}" x2="${width/2}" y1="${pad}" y2="${height-pad}" stroke="#d7dee8"/>${circles}<text x="${width-pad}" y="${height/2-5}" text-anchor="end" fill="#5f6c7b" font-size="10">PC1</text><text x="${width/2+5}" y="${pad+10}" fill="#5f6c7b" font-size="10">PC2</text></svg>`;
}

function clusterScatter(container, rows) {
  if (!rows.length) { container.innerHTML='<div class="empty">无可展示数据</div>'; return; }
  const palette=["#176b87","#cf3f36","#16845b","#d19a20","#7c3aed","#db2777","#0891b2","#65a30d","#ea580c","#475569"];
  const width=760,height=250,pad=28; const xs=rows.map(row=>Number(row.pc1)),ys=rows.map(row=>Number(row.pc2)); const maxX=Math.max(...xs.map(Math.abs),1e-9),maxY=Math.max(...ys.map(Math.abs),1e-9); const x=value=>width/2+value/maxX*(width/2-pad); const y=value=>height/2-value/maxY*(height/2-pad);
  const circles=rows.map(row=>`<circle cx="${x(Number(row.pc1))}" cy="${y(Number(row.pc2))}" r="3" fill="${palette[(row.cluster-1)%palette.length]}" fill-opacity=".72"><title>Cluster ${row.cluster} · ${escapeHtml(row.timestamp.slice(0,16))}</title></circle>`).join("");
  const legend=[...new Set(rows.map(row=>row.cluster))].map(cluster=>`<text x="${pad+(cluster-1)*82}" y="15" fill="${palette[(cluster-1)%palette.length]}" font-size="10">● Cluster ${cluster}</text>`).join("");
  container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="运行状态聚类散点">${legend}<line x1="${pad}" x2="${width-pad}" y1="${height/2}" y2="${height/2}" stroke="#d7dee8"/><line x1="${width/2}" x2="${width/2}" y1="${pad}" y2="${height-pad}" stroke="#d7dee8"/>${circles}<text x="${width-pad}" y="${height/2-5}" text-anchor="end" fill="#5f6c7b" font-size="10">PC1</text><text x="${width/2+5}" y="${pad+10}" fill="#5f6c7b" font-size="10">PC2</text></svg>`;
}

document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>{ const target=button.dataset.panel; document.querySelectorAll(".tab").forEach(node=>node.classList.toggle("active",node===button)); document.querySelectorAll(".panel").forEach(panel=>panel.classList.toggle("active",target==="statePanels"?["clusterPanel","performancePanel"].includes(panel.id):panel.id===target)); }));
el("resetButton").addEventListener("click",()=>location.reload());
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
