from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
from email.parser import BytesParser
from email.policy import default as email_policy
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

import numpy as np
import pandas as pd

from .clustering import cluster_model_scores, cluster_operating_states
from .state_exploration import ExplorationConfig, run_state_exploration
from .compat import (
    MODEL_PURPOSES,
    training_windows_from_payload,
)
from .data_session import (
    DataLoadResult,
    DataSessionCache,
    DataSessionMetadata,
    DataSessionStageError,
    TXT_ENCODING,
    TXT_HEADER_ROW,
    TXT_SEPARATOR,
    normalize_column_names,
)
from .dpca import fit_dpca
from .model_io import (
    copy_validated_model_package,
    export_deployment_package,
    freeze_validated_model_package,
    load_deployment_package,
    load_model_package,
    save_model_package,
)
from .preprocessing import (
    PreprocessingConfig,
    PreprocessingQualityError,
    preprocess_window,
    preprocessing_config_from_mapping,
)
from .quality import QualityReport, inspect_data_quality, raw_column_profile
from .replay import FrozenReplayResult, replay_frozen_model
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
from .training import _validate_dynamic_matrix, build_training_matrix
from .trend import downsample_trend, trend_payload_data
from .validation import (
    build_validation_evidence,
    record_engineer_decision,
    validate_model_windows,
    verify_validation_evidence,
    validation_context_start,
    validation_windows_from_payload,
)
from .windows import (
    add_training_window,
    merge_excluded_windows,
    normalize_training_windows,
    remove_training_window,
    set_enabled_training_window,
    subtract_excluded_windows,
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
MAX_STATE_EXPLORATION_RUNS = 8
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CANDIDATE_DECISIONS = frozenset({"pending", "accepted", "rejected"})
_VALIDATION_ARTIFACTS = {
    "scores": ("validation_scores.csv", "text/csv; charset=utf-8"),
    "report": ("validation_report.json", "application/json; charset=utf-8"),
    "contributions": (
        "validation_contributions.json",
        "application/json; charset=utf-8",
    ),
}
_FROZEN_REPLAY_ARTIFACTS = {
    "scores": ("frozen_replay_scores.csv", "text/csv; charset=utf-8"),
    "summary": ("frozen_replay_summary.json", "application/json; charset=utf-8"),
    "contributions": (
        "frozen_replay_contributions.json",
        "application/json; charset=utf-8",
    ),
}
DATA_SESSIONS = DataSessionCache()
STATE_EXPLORATION_RUNS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_STATE_EXPLORATION_LOCK = threading.RLock()
_LIFECYCLE_LOCKS: dict[str, threading.Lock] = {}
_LIFECYCLE_LOCKS_LOCK = threading.Lock()
_REPLAY_LOCKS: dict[str, threading.Lock] = {}
_REPLAY_LOCKS_LOCK = threading.Lock()


class WebStageError(ValueError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage


class StateExplorationNotFoundError(ValueError):
    pass


@contextmanager
def _web_stage(stage: str):
    try:
        yield
    except WebStageError:
        raise
    except Exception as error:
        raise WebStageError(stage, error) from error


def error_payload(error: Exception) -> dict[str, str]:
    stage = error.stage if isinstance(error, WebStageError) else "failed"
    return {"error": str(error), "stage": stage}


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
    suffix = Path(filename).suffix
    normalized_suffix = suffix.lower()
    if normalized_suffix not in {".csv", ".xlsx", ".txt"}:
        raise ValueError("当前仅支持 CSV、XLSX 或 TXT 文件")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_REQUEST_BODY_BYTES:
        raise ValueError("上传文件超过 200 MB 限制")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    path = UPLOADS_DIR / f"{file_id}{suffix}"
    path.write_bytes(content)
    try:
        encoding, columns = _read_header(path)
    except Exception as error:
        path.unlink(missing_ok=True)
        DATA_SESSIONS.remove_dataset(file_id)
        file_type = _upload_file_type(normalized_suffix)
        raise ValueError(f"{file_type}读取失败：{error}") from error
    if not columns:
        path.unlink(missing_ok=True)
        DATA_SESSIONS.remove_dataset(file_id)
        raise ValueError(f"{_upload_file_type(normalized_suffix)} 不包含列")
    return {
        "file_id": file_id,
        "filename": Path(filename).name,
        "columns": columns,
        "file_type": normalized_suffix[1:],
        **({"encoding": encoding} if normalized_suffix == ".csv" else {}),
        "size_bytes": len(content),
    }


def _read_header(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() == ".xlsx":
        return "", normalize_column_names(
            pd.read_excel(path, nrows=0, sheet_name=0, engine="openpyxl").columns
        )
    if path.suffix.lower() == ".txt":
        frame = pd.read_csv(
            path,
            nrows=0,
            encoding=TXT_ENCODING,
            sep=TXT_SEPARATOR,
            header=TXT_HEADER_ROW,
        )
        if len(frame.columns) < 2:
            raise ValueError("TXT 格式必须使用 Tab 分隔且首行为表头")
        return TXT_ENCODING, normalize_column_names(frame.columns)
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return encoding, normalize_column_names(
                pd.read_csv(path, nrows=0, encoding=encoding).columns
            )
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError("CSV 编码无法识别，请转换为 UTF-8 或 GB18030") from last_error


def _upload_file_type(suffix: str) -> str:
    return {".csv": "CSV", ".xlsx": "XLSX", ".txt": "TXT"}[suffix]


def inspect_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    loaded = _load_upload(payload, None)
    parsed = loaded.frame
    with _web_stage("quality_check"):
        numeric_columns = list(loaded.metadata.numeric_candidate_columns)
        report = inspect_data_quality(parsed, timestamp_column, numeric_columns)
        raw_columns = [
            str(column) for column in parsed.columns if column != timestamp_column
        ]
        tag_configs = normalize_tag_registry(
            raw_columns, payload.get("tag_configs")
        )
        column_profiles = [
            {
                "tag": column,
                **raw_column_profile(parsed[column], tag_configs[column]),
            }
            for column in raw_columns
        ]
        timestamps = parsed[timestamp_column].dropna().sort_values().drop_duplicates()
        if len(timestamps) < 3:
            raise ValueError("至少需要三个有效时间点")
    normal_end_index = max(0, min(len(timestamps) - 2, int(len(timestamps) * 0.65)))
    validation_start_index = normal_end_index + 1
    result = {
        "rows": len(parsed),
        "columns": list(parsed.columns),
        "numeric_columns": numeric_columns,
        "time_start": timestamps.iloc[0].isoformat(),
        "time_end": timestamps.iloc[-1].isoformat(),
        "trend_default_start": timestamps.iloc[0].isoformat(),
        "trend_default_end": timestamps.iloc[
            min(len(timestamps) - 1, 30000 - 1)
        ].isoformat(),
        "suggested_normal_end": timestamps.iloc[normal_end_index].isoformat(),
        "suggested_validation_start": timestamps.iloc[
            validation_start_index
        ].isoformat(),
        "sample_interval_minutes": report.inferred_interval_minutes,
        "can_train_without_review": report.can_train,
        "quality_issues": [asdict(issue) for issue in report.issues],
        "column_profiles": column_profiles,
        "modeling_tag_hint": (
            {
                "code": "insufficient_continuous_tags",
                "candidate_count": len(numeric_columns),
                "minimum_count": 2,
                "message": "当前可建模连续数值 Tag 少于 2 个，不能进入后续建模。",
            }
            if len(numeric_columns) < 2
            else None
        ),
    }
    return _with_data_usage(result, loaded, len(parsed), len(parsed))


def train_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    tags = _required_tags(payload)
    excluded = payload.get("excluded_tags")
    excluded_tags = (
        [
            str(item.get("tag", "")).strip()
            for item in excluded
            if isinstance(item, dict)
        ]
        if isinstance(excluded, list)
        else []
    )
    loaded = _load_required_upload(
        payload,
        list(dict.fromkeys([*tags, *excluded_tags, *_state_filter_columns(payload)])),
        "找不到 Tag：",
    )
    parsed = loaded.frame
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    _require_continuous_roles(tags, registry)
    model_purpose = _model_purpose(payload.get("model_purpose"))
    model_status = "draft" if model_purpose == "exploratory" else "candidate"
    config = _preprocessing_config(payload)
    tag_configs = normalize_tag_configs(
        tags, {tag: registry[tag] for tag in tags}
    )
    training_windows = training_windows_from_payload(payload)
    training_result = _build_training_matrix_with_stage(
        parsed,
        timestamp_column,
        tags,
        config,
        training_windows,
        engineering_ranges(tag_configs),
        exclude_engineering_range=model_purpose == "normal_state",
        reference_columns=excluded_tags,
    )
    with _web_stage("quality_check"):
        excluded_tag_records = _excluded_tag_records(
            payload.get("excluded_tags"), training_result.reference, tags, registry
        )
    dynamic = training_result.dynamic
    components_value = payload.get("n_components")
    n_components = None if components_value in {None, ""} else int(components_value)
    variance_threshold = float(payload.get("variance_threshold", 0.95))
    with _web_stage("fitting"):
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
        **config.to_dict(),
        "variance_threshold": variance_threshold,
        "tag_configs": tag_configs,
        "source_tag_configs": registry,
        "excluded_tags": excluded_tag_records,
        "training_summary": training_result.window_summaries,
        "preprocessing_summary": training_result.window_summaries,
        "training_window_totals": training_result.training_window_totals,
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
    with _web_stage("scoring"):
        scores = model.score(dynamic)
    result = {
        "run_id": run_id,
        "model_name": model_name,
        "model_purpose": model_purpose,
        "model_status": model_status,
        "training_rows": len(dynamic),
        "training_window_summary": training_result.window_summaries,
        "training_window_totals": training_result.training_window_totals,
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
    return _with_data_usage(
        result,
        loaded,
        len(training_result.reference),
        len(result["scores"]),
    )


def quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    tags = _required_tags(payload)
    loaded = _load_required_upload(
        payload, [*tags, *_state_filter_columns(payload)], "找不到 Tag："
    )
    parsed = loaded.frame
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    _require_continuous_roles(tags, registry)
    config = _preprocessing_config(payload)
    normal_training = _build_training_matrix_with_stage(
        parsed,
        timestamp_column,
        tags,
        config,
        training_windows_from_payload(payload),
        engineering_ranges(
            normalize_tag_configs(tags, {tag: registry[tag] for tag in tags})
        ),
        exclude_engineering_range=True,
        validate_dynamic=False,
    )
    exploratory_training = _build_training_matrix_with_stage(
        parsed,
        timestamp_column,
        tags,
        config,
        training_windows_from_payload(payload),
        engineering_ranges(
            normalize_tag_configs(tags, {tag: registry[tag] for tag in tags})
        ),
        exclude_engineering_range=False,
        validate_dynamic=False,
    )
    with _web_stage("quality_check"):
        result = model_quality_payload(
            parsed,
            exploratory_training.reference,
            timestamp_column,
            tags,
            registry,
            config.sample_interval_minutes,
        )
        shared_can_train = result["can_train"]
        readiness = {
            purpose: _training_readiness(training.dynamic)
            for purpose, training in {
                "normal_state": normal_training,
                "exploratory": exploratory_training,
            }.items()
        }
        for item in readiness.values():
            item["can_train"] = shared_can_train and item["issue"] is None
        result["training_readiness"] = readiness
        result["can_train"] = readiness["normal_state"]["can_train"]
        if readiness["normal_state"]["issue"] is not None:
            result["time_issues"].append(readiness["normal_state"]["issue"])
    result["training_window_summary"] = normal_training.window_summaries
    result["training_quality_warnings"] = normal_training.global_quality_warnings
    return _with_data_usage(
        result, loaded, len(normal_training.reference), len(normal_training.reference)
    )


def training_windows_payload(payload: dict[str, Any]) -> dict[str, Any]:
    windows = (
        normalize_training_windows(payload["training_windows"], allow_empty=True)
        if "training_windows" in payload
        else training_windows_from_payload(payload)
    )
    timestamps = None
    loaded = None
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
        elif action == "confirm_candidate":
            candidate = operation.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("候选窗口无效")
            candidate_id = candidate.get("id")
            source = candidate.get("source")
            source_ref = candidate.get("source_ref")
            comment = candidate.get("comment")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError("候选窗口ID无效")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("候选窗口来源无效")
            if source_ref is not None and (
                not isinstance(source_ref, str) or not source_ref.strip()
            ):
                raise ValueError("候选窗口来源引用无效")
            if not isinstance(comment, str):
                raise ValueError("候选窗口备注无效")
            base_id = f"training-{candidate_id}"
            if any(
                window["id"] == base_id
                or window["id"].startswith(f"{base_id}-part-")
                for window in windows
            ):
                raise ValueError("该候选已生成训练窗口")
            excluded_windows = merge_excluded_windows(
                operation.get("excluded_windows", [])
            )
            timestamp_column = _required_text(payload, "timestamp_column")
            loaded = _load_upload(payload, [])
            timestamps = loaded.frame[timestamp_column]
            parts = subtract_excluded_windows(candidate, excluded_windows, timestamps)
            if not parts:
                raise ValueError("候选窗口已被排除窗口完全覆盖")
            candidate_start, candidate_end = candidate["start"], candidate["end"]
            is_cut = parts != [{"start": pd.Timestamp(candidate_start).isoformat(), "end": pd.Timestamp(candidate_end).isoformat()}]
            additions = [
                {
                    "id": (
                        f"{base_id}-part-{position:03d}"
                        if is_cut
                        else base_id
                    ),
                    "start": part["start"],
                    "end": part["end"],
                    "source": source,
                    "source_ref": source_ref or candidate_id,
                    "enabled": True,
                    "comment": comment,
                }
                for position, part in enumerate(parts, start=1)
            ]
            for addition in additions:
                windows = add_training_window(windows, addition)
        else:
            raise ValueError("training_windows操作无效")
    if payload.get("file_id") and loaded is None:
        timestamp_column = _required_text(payload, "timestamp_column")
        loaded = _load_upload(payload, [])
        timestamps = loaded.frame[timestamp_column]
    result = {
        "training_windows": windows,
        "summary": summarize_training_windows(
            windows, timestamps, int(payload.get("sample_interval_minutes", 5))
        ),
    }
    return (
        _with_data_usage(result, loaded, len(timestamps), len(timestamps))
        if loaded is not None and timestamps is not None
        else result
    )


def trend_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ValueError("趋势Tag必须是非空列表")
    tags = [str(tag).strip() for tag in raw_tags]
    if len(tags) != len(set(tags)):
        raise ValueError("趋势Tag不能重复")
    if len(tags) > 8:
        raise ValueError("趋势浏览一次最多选择8个Tag")
    state_columns = _state_filter_columns(payload)
    loaded = _load_required_upload(
        payload, [*tags, *state_columns], "找不到趋势Tag："
    )
    parsed = loaded.frame
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
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
    with _web_stage("preprocessing"):
        result = trend_payload_data(
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
    analysis_rows = int(result["statistics"][tags[0]]["current"]["sample_count"])
    return _with_data_usage(result, loaded, analysis_rows, len(result["rows"]))


def preprocessing_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ValueError("预处理预览Tag必须是非空列表")
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    if len(tags) != len(set(tags)):
        raise ValueError("预处理预览Tag不能重复")
    if len(tags) > 8:
        raise ValueError("预处理预览一次最多选择8个Tag")
    config = _preprocessing_config(payload)
    state_columns = [condition.column for condition in config.state_filters]
    loaded = _load_required_upload(
        payload, [*tags, *state_columns], "找不到预处理列："
    )
    selected = _select_window(
        loaded.frame,
        timestamp_column,
        _required_text(payload, "start"),
        _required_text(payload, "end"),
    )
    indexed = _indexed_tags(
        selected, timestamp_column, [*tags, *state_columns]
    )
    with _web_stage("preprocessing"):
        processed = preprocess_window(
            indexed,
            tags,
            config,
            validate_quality=False,
            include_intermediates=True,
        )
    assert processed.raw is not None
    assert processed.resampled is not None
    assert processed.filtered is not None
    result = {
        "summary": processed.summary.to_dict(),
        "raw": _preview_rows(
            processed.raw.loc[:, tags], processed.raw_segment_ids
        ),
        "resampled": _preview_rows(
            processed.resampled.loc[:, tags], processed.segment_ids
        ),
        "filtered": _preview_rows(
            processed.filtered.loc[:, tags], processed.segment_ids
        ),
    }
    display_rows = max(len(result["raw"]), len(result["resampled"]), len(result["filtered"]))
    return _with_data_usage(result, loaded, len(selected), display_rows)


def _preview_rows(
    frame: pd.DataFrame, segment_ids: pd.Series
) -> list[dict[str, Any]]:
    full_gap_starts = segment_ids.ne(segment_ids.shift())
    if len(full_gap_starts):
        full_gap_starts.iloc[0] = False
    gap_starts = full_gap_starts.reindex(frame.index, fill_value=False)
    if len(frame) > MAX_CHART_POINTS:
        positions = downsample_trend(
            frame,
            frame,
            segment_ids.reindex(frame.index),
            limit=MAX_CHART_POINTS,
        )
        frame = frame.iloc[positions]
        gap_starts = gap_starts.iloc[positions]
    return [
        {
            "timestamp": timestamp.isoformat(),
            "physical_gap_start": bool(gap_starts.loc[timestamp]),
            "gap_start": bool(gap_starts.loc[timestamp]),
            **{
                column: (
                    float(value) if pd.notna(value) and np.isfinite(value) else None
                )
                for column, value in row.items()
            },
        }
        for timestamp, row in frame.iterrows()
    ]


def tag_config_template_payload(payload: dict[str, Any]) -> bytes:
    timestamp_column = _required_text(payload, "timestamp_column")
    metadata, _ = _upload_metadata(payload, timestamp_column)
    return build_tag_config_template(list(metadata.numeric_candidate_columns))


def tag_config_export_payload(payload: dict[str, Any]) -> bytes:
    timestamp_column = _required_text(payload, "timestamp_column")
    metadata, _ = _upload_metadata(payload, timestamp_column)
    tags = list(metadata.numeric_candidate_columns)
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
    metadata, _ = _upload_metadata(
        {
            "file_id": file_id,
            "timestamp_column": timestamp_column,
            "encoding": encoding,
        },
        timestamp_column,
    )
    return parse_tag_config_workbook(
        content, list(metadata.numeric_candidate_columns)
    )


def _performance_config_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("performance_config")
    if value in (None, ""):
        if not str(payload.get("performance_tag", "")).strip():
            return None
        value = {
            "performance_tag": payload.get("performance_tag"),
            "direction": payload.get("performance_direction"),
            "target_min": payload.get("performance_target_min"),
            "target_max": payload.get("performance_target_max"),
            "minimum_duration_minutes": payload.get(
                "performance_minimum_duration_minutes", 30
            ),
            "candidate_count": payload.get("performance_candidate_count", 3),
        }
    if not isinstance(value, dict):
        raise ValueError("performance_config must be an object")
    return dict(value)


def state_exploration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_column = _required_text(payload, "timestamp_column")
    tags = _required_tags(payload)
    config = _preprocessing_config(payload)
    performance_config = _performance_config_payload(payload)
    performance_tag = (
        str(performance_config.get("performance_tag", "")).strip()
        if performance_config is not None
        else ""
    )
    state_columns = [condition.column for condition in config.state_filters]
    requested_columns = list(
        dict.fromkeys([*tags, *state_columns, *([performance_tag] if performance_tag else [])])
    )
    loaded = _load_required_upload(payload, requested_columns, "找不到 Tag：")
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    _require_continuous_roles(tags, registry)
    _require_state_filter_roles(state_columns, registry)
    tag_configs = normalize_tag_configs(
        tags, {tag: registry[tag] for tag in tags}
    )
    exploration_start = pd.Timestamp(_required_text(payload, "exploration_start"))
    exploration_end = pd.Timestamp(_required_text(payload, "exploration_end"))
    selected = _select_window(
        loaded.frame,
        timestamp_column,
        exploration_start.isoformat(),
        exploration_end.isoformat(),
    )
    exploration_config = ExplorationConfig(
        **dict(payload.get("exploration_config") or {})
    )
    with _web_stage("preprocessing"):
        try:
            exploration = run_state_exploration(
                _indexed_tags(
                    selected,
                    timestamp_column,
                    list(dict.fromkeys([*tags, *state_columns, *([performance_tag] if performance_tag else [])])),
                ),
                tags,
                config,
                exploration_config,
                performance_config=performance_config,
                engineering_ranges=engineering_ranges(tag_configs),
                resampling_window=(exploration_start, exploration_end),
            )
        except PreprocessingQualityError as error:
            raise WebStageError(
                "quality_check", ValueError(_format_quality_errors(error.report))
            ) from error
    run_id = str(exploration["exploration_run_id"])
    response = {key: value for key, value in exploration.items() if key not in {"cluster_series", "cluster_series_display"}}
    response["cluster_series"] = _exploration_series(
        exploration["cluster_series_display"],
        exploration["cluster_series"],
        config.sample_interval_minutes,
    )
    response = _with_data_usage(
        response, loaded, len(selected), len(response["cluster_series"])
    )
    exploration["data_usage"] = response["data_usage"]
    _store_state_exploration_run(run_id, exploration)
    return response


def _store_state_exploration_run(
    run_id: str, exploration: dict[str, Any]
) -> None:
    with _STATE_EXPLORATION_LOCK:
        STATE_EXPLORATION_RUNS[run_id] = exploration
        STATE_EXPLORATION_RUNS.move_to_end(run_id)
        while len(STATE_EXPLORATION_RUNS) > MAX_STATE_EXPLORATION_RUNS:
            _, evicted = STATE_EXPLORATION_RUNS.popitem(last=False)
            evicted.clear()


def clear_state_exploration_cache() -> None:
    with _STATE_EXPLORATION_LOCK:
        for exploration in STATE_EXPLORATION_RUNS.values():
            exploration.clear()
        STATE_EXPLORATION_RUNS.clear()


def _state_exploration_run(run_id: str) -> dict[str, Any]:
    with _STATE_EXPLORATION_LOCK:
        try:
            exploration = STATE_EXPLORATION_RUNS[run_id]
        except KeyError as error:
            raise StateExplorationNotFoundError("状态探索运行记录不存在") from error
        STATE_EXPLORATION_RUNS.move_to_end(run_id)
        # Keep the cached dict mutable for eviction while the current request
        # retains references to the DataFrames it is reading.
        return dict(exploration)


def _exploration_summary_payload(run_id: str) -> dict[str, Any]:
    exploration = _state_exploration_run(run_id)
    return {
        key: value
        for key, value in exploration.items()
        if key not in {"cluster_series", "cluster_series_display"}
    }


def _state_exploration_candidates(exploration: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in [
            *exploration["cluster_candidates"],
            *exploration["performance_candidates"],
        ]
    }


def _decision_requests(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("decisions")
    if value is None:
        value = [payload]
    if not isinstance(value, list) or not value:
        raise ValueError("decisions必须是非空列表")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("候选决策必须是对象")
        candidate_id = item.get("candidate_id")
        decision = item.get("decision")
        comment = item.get("comment")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id无效")
        if candidate_id in seen:
            raise ValueError("同一请求中的candidate_id不能重复")
        if decision not in _CANDIDATE_DECISIONS:
            raise ValueError("候选决策无效")
        if not isinstance(comment, str):
            raise ValueError("候选决策备注必须是字符串")
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "comment": comment,
            }
        )
        seen.add(candidate_id)
    return decisions


def state_exploration_decisions_payload(
    run_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    decisions = _decision_requests(payload)
    with _STATE_EXPLORATION_LOCK:
        try:
            exploration = STATE_EXPLORATION_RUNS[run_id]
        except KeyError as error:
            raise StateExplorationNotFoundError("状态探索运行记录不存在") from error
        candidates = _state_exploration_candidates(exploration)
        missing = [item["candidate_id"] for item in decisions if item["candidate_id"] not in candidates]
        if missing:
            raise ValueError("候选不属于当前状态探索运行：" + ", ".join(missing))
        records = {
            str(item["candidate_id"]): dict(item)
            for item in exploration["candidate_decisions"]
        }
        decided_at = pd.Timestamp.now(tz="UTC").isoformat()
        for item in decisions:
            records[item["candidate_id"]] = {
                **item,
                "decided_at": decided_at,
            }
        exploration["candidate_decisions"] = [
            records[candidate_id] for candidate_id in candidates
        ]
        STATE_EXPLORATION_RUNS.move_to_end(run_id)
        return {
            "exploration_run_id": run_id,
            "candidate_decisions": exploration["candidate_decisions"],
        }


def state_exploration_training_windows_payload(
    run_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    candidate_ids = payload.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError("candidate_ids必须是非空列表")
    if any(not isinstance(value, str) or not value.strip() for value in candidate_ids):
        raise ValueError("candidate_ids无效")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids不能重复")
    windows = normalize_training_windows(payload.get("training_windows"), allow_empty=True)
    with _STATE_EXPLORATION_LOCK:
        try:
            exploration = STATE_EXPLORATION_RUNS[run_id]
        except KeyError as error:
            raise StateExplorationNotFoundError("状态探索运行记录不存在") from error
        candidates = _state_exploration_candidates(exploration)
        decisions = {
            str(item["candidate_id"]): item
            for item in exploration["candidate_decisions"]
        }
        additions: list[dict[str, object]] = []
        existing_window_ids = {window["id"] for window in windows}
        for candidate_id in candidate_ids:
            try:
                candidate = candidates[candidate_id]
            except KeyError as error:
                raise ValueError("候选不属于当前状态探索运行：" + candidate_id) from error
            if decisions[candidate_id]["decision"] != "accepted":
                raise ValueError("只有已接受候选可以加入正常状态候选池：" + candidate_id)
            window_id = f"state-exploration-{run_id}-{candidate_id}"
            if window_id in existing_window_ids:
                continue
            additions.append(
                {
                    "id": window_id,
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "source": str(candidate["source"]),
                    "source_ref": candidate_id,
                    "enabled": False,
                    "comment": str(decisions[candidate_id]["comment"]),
                }
            )
        updated = normalize_training_windows([*windows, *additions], allow_empty=True)
        STATE_EXPLORATION_RUNS.move_to_end(run_id)
    return {
        "training_windows": updated,
        "summary": summarize_training_windows(updated),
        "converted_candidate_ids": [window["source_ref"] for window in additions],
    }


def _exploration_max_points(query: dict[str, list[str]], default: int) -> int:
    raw = query.get("max_points", [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("max_points must be a positive integer of at least 2") from error
    if value < 2:
        raise ValueError("max_points must be a positive integer of at least 2")
    return value


def _exploration_series_payload(
    run_id: str, query: dict[str, list[str]]
) -> dict[str, Any]:
    exploration = _state_exploration_run(run_id)
    config = dict(exploration.get("exploration_config") or {})
    limit = _exploration_max_points(
        query, int(config.get("maximum_plot_points", MAX_CHART_POINTS))
    )
    full = exploration["cluster_series"]
    display = exploration["cluster_series_display"]
    if limit < len(full):
        from .state_exploration import _display_points

        display = _display_points(
            full,
            limit,
            int(exploration["preprocessing_summary"]["target_interval_minutes"]),
        )
    return {
        "exploration_run_id": run_id,
        "full_point_count": int(len(full)),
        "returned_point_count": int(len(display)),
        "cluster_series": _exploration_series(
            display,
            full,
            int(exploration["preprocessing_summary"]["target_interval_minutes"]),
        ),
    }


def _exploration_series(
    series: pd.DataFrame,
    full_series: pd.DataFrame | None = None,
    interval: int = 5,
) -> list[dict[str, Any]]:
    full = series if full_series is None else full_series
    positions = full.index.get_indexer(series.index)
    expected = pd.Timedelta(minutes=interval)
    result = []
    for position, (timestamp, row) in zip(positions, series.iterrows(), strict=True):
        break_before = False
        if position > 0:
            break_before = bool(
                full.index[position] - full.index[position - 1] != expected
                or full.segment_id.iloc[position] != full.segment_id.iloc[position - 1]
            )
        result.append(
            {
                "timestamp": timestamp.isoformat(),
                "cluster_id": str(row.cluster_id),
                "pc1": float(row.pc1),
                "pc2": float(row.pc2),
                "segment_id": int(row.segment_id),
                "break_before": break_before,
            }
        )
    return result


def cluster_payload(payload: dict[str, Any]) -> dict[str, Any]:
    exploratory_run_id = payload.get("exploratory_run_id")
    if exploratory_run_id not in {None, ""}:
        return _cluster_exploratory_payload(
            payload,
            _validated_id(str(exploratory_run_id), "exploratory_run_id"),
        )
    timestamp_column = _required_text(payload, "timestamp_column")
    tags = _required_tags(payload)
    config = _preprocessing_config(payload)
    state_columns = [condition.column for condition in config.state_filters]
    loaded = _load_required_upload(
        payload, list(dict.fromkeys([*tags, *state_columns])), "找不到 Tag："
    )
    parsed = loaded.frame
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = normalize_tag_registry(all_tags, payload.get("tag_configs"))
    _require_continuous_roles(tags, registry)
    _require_state_filter_roles(state_columns, registry)
    tag_configs = normalize_tag_configs(
        tags, {tag: registry[tag] for tag in tags}
    )
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    with _web_stage("preprocessing"):
        indexed = _indexed_tags(
            analysis,
            timestamp_column,
            [*tags, *state_columns],
        )
        try:
            dynamic = preprocess_window(
                indexed, tags, config, engineering_ranges(tag_configs)
            ).dynamic
        except PreprocessingQualityError as error:
            raise WebStageError("quality_check", error) from error
        if dynamic.empty:
            raise ValueError("平滑和 Lag 扩展后没有足够的聚类样本")
    with _web_stage("fitting"):
        result = cluster_operating_states(
            dynamic,
            n_clusters=int(payload.get("n_clusters", 3)),
            variance_threshold=float(payload.get("variance_threshold", 0.95)),
            sample_interval_minutes=config.sample_interval_minutes,
        )
    response = _cluster_result_payload(result, config.sample_interval_minutes)
    return _with_data_usage(response, loaded, len(analysis), len(response["points"]))


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
    config = preprocessing_config_from_mapping(config_data)
    state_columns = [condition.column for condition in config.state_filters]
    loaded = _load_required_upload(payload, [*tags, *state_columns], "找不到 Tag：")
    parsed = loaded.frame
    registry = normalize_tag_registry(
        list(loaded.metadata.numeric_candidate_columns),
        config_data.get("source_tag_configs"),
    )
    _require_state_filter_roles(state_columns, registry)
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    with _web_stage("preprocessing"):
        indexed = _indexed_tags(analysis, timestamp_column, [*tags, *state_columns])
        try:
            dynamic = preprocess_window(
                indexed, tags, config, engineering_ranges(tag_configs)
            ).dynamic
        except PreprocessingQualityError as error:
            raise WebStageError("quality_check", error) from error
        if dynamic.empty:
            raise ValueError("平滑和 Lag 扩展后没有足够的聚类样本")
    with _web_stage("fitting"):
        result = cluster_model_scores(
            model,
            dynamic,
            n_clusters=int(payload.get("n_clusters", 3)),
            sample_interval_minutes=config.sample_interval_minutes,
        )
    response = {
        **_cluster_result_payload(result, config.sample_interval_minutes),
        "exploratory_run_id": exploratory_run_id,
    }
    return _with_data_usage(response, loaded, len(analysis), len(response["points"]))


def _cluster_result_payload(result: Any, interval: int = 5) -> dict[str, Any]:
    points = result.points
    if len(points) > MAX_CHART_POINTS:
        from .state_exploration import _display_points

        points = _display_points(points, MAX_CHART_POINTS, interval)
    return {
        "sample_count": len(result.points),
        "full_point_count": len(result.points),
        "returned_point_count": len(points),
        "n_components": result.n_components,
        "cumulative_explained_variance": result.cumulative_explained_variance,
        "clusters": list(result.summaries),
        "pc_columns": list(result.pc_columns),
        "cluster_centers": {
            str(cluster): [float(value) for value in center]
            for cluster, center in result.centers.items()
        },
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
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list):
        raise ValueError("性能条件必须是列表")
    columns = [
        str(item.get("column", "")).strip()
        for item in raw_conditions
        if isinstance(item, dict)
    ]
    loaded = _load_required_upload(
        payload,
        list(dict.fromkeys(columns)),
        "missing performance columns: ",
    )
    parsed = loaded.frame
    analysis = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "analysis_start"),
        _required_text(payload, "analysis_end"),
    )
    indexed = analysis.set_index(timestamp_column)
    with _web_stage("quality_check"):
        result = screen_performance_states(
            indexed,
            raw_conditions,
            sample_interval_minutes=int(payload.get("sample_interval_minutes", 5)),
        )
    return _with_data_usage(result, loaded, len(analysis), len(analysis))


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    with _lifecycle_lock(run_id):
        return _validate_payload_locked(payload)


def _validate_payload_locked(payload: dict[str, Any]) -> dict[str, Any]:
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
    validation_windows = validation_windows_from_payload(payload)
    training_windows = [
        (pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
        for window in manifest["training_windows"]
        if window["enabled"]
    ]
    label_column = str(payload.get("label_column", "")).strip()
    config = preprocessing_config_from_mapping(config_data)
    state_columns = [condition.column for condition in config.state_filters]
    requested_columns = list(dict.fromkeys([
        *tags, *state_columns, *([label_column] if label_column else [])
    ]))
    try:
        loaded = _load_upload(payload, requested_columns)
    except ValueError as error:
        missing = _missing_columns_from_error(error)
        if missing is None:
            raise
        if label_column and label_column in missing:
            raise ValueError(f"找不到工程标签列：{label_column}") from error
        raise ValueError(f"找不到 Tag：{', '.join(missing)}") from error
    parsed = loaded.frame
    for window in validation_windows:
        if not window["enabled"]:
            continue
        context = _select_window(
            parsed,
            timestamp_column,
            validation_context_start(pd.Timestamp(window["start"]), config).isoformat(),
            window["end"],
        )
        if config.resampling_method == "none" and manifest["schema_version"] <= 4:
            with _web_stage("quality_check"):
                _require_clean_data(
                    context,
                    timestamp_column,
                    tags,
                    config.sample_interval_minutes,
                    engineering_ranges(tag_configs),
                )
    with _web_stage("preprocessing"):
        indexed = _indexed_tags(parsed, timestamp_column, [*tags, *state_columns])
    with _web_stage("scoring"):
        validation_result = validate_model_windows(
            model,
            indexed,
            tags,
            config,
            training_windows,
            validation_windows,
            tag_configs,
            preprocessing_semantics=("legacy" if manifest["schema_version"] <= 4 else "schema5"),
        )
        scores = validation_result["scores"]
        focus_timestamp = _focus_timestamp(
            scores, model.t2_limits[0.99], model.q_limits[0.99]
        )
        contributions = validation_result["contributions"]

    result: dict[str, Any] = {
        "run_id": run_id,
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "engineer_decision_required": True,
        "validation_windows": validation_result["validation_windows"],
        "validation_window_summaries": validation_result["window_summaries"],
        "normal_validation_complete": validation_result["normal_validation_complete"],
        "known_abnormal_complete": validation_result["known_abnormal_complete"],
        "validation_metrics": validation_result["validation_metrics"],
        "contribution_stability": validation_result["contribution_stability"],
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
    if len(validation_result["validation_windows"]) == 1:
        window = validation_result["validation_windows"][0]
        result["validation_window"] = [window["start"], window["end"]]
    if label_column:
        if label_column not in parsed.columns:
            raise ValueError(f"找不到工程标签列：{label_column}")
        labels = parsed.set_index(timestamp_column)[label_column].reindex(scores.index)
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
    report = {
        key: value
        for key, value in result.items()
        if key not in {"scores", "contributions"}
    }
    _commit_web_validation_artifacts(run_dir, model_path, model, scores, contributions, report, timestamp_column, config.sample_interval_minutes)
    result["validation_downloads"] = {
        artifact: f"/download/validation?run_id={run_id}&artifact={artifact}"
        for artifact in _VALIDATION_ARTIFACTS
    }
    return _with_data_usage(result, loaded, len(scores), len(result["scores"]))


def validation_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    with _lifecycle_lock(run_id):
        return _validation_decision_payload_locked(payload)


def _validation_decision_payload_locked(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    run_dir = RUNS_DIR / run_id
    model_path = run_dir / "model.pcamodel"
    report_path = run_dir / "validation_report.json"
    if not model_path.is_file() or not report_path.is_file():
        raise ValueError("候选模型或验证报告不存在")
    model, manifest = load_model_package(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = preprocessing_config_from_mapping(manifest["config"])
    verify_validation_evidence(model_path, model, report, run_dir / "validation_scores.csv", run_dir / "validation_contributions.json", sample_interval_minutes=config.sample_interval_minutes)
    decision = record_engineer_decision(
        manifest,
        report,
        payload.get("decision"),
        payload.get("comment", ""),
    )
    report["engineer_decision"] = decision
    validated_path: Path | None = None
    if decision["decision"] == "passed":
        validated_path = run_dir / "validated_model.pcamodel"
    if validated_path is None:
        _write_web_json_atomic(report_path, report)
    else:
        temporary_validated = _web_temporary_path(validated_path)
        copy_validated_model_package(model_path, temporary_validated, validation_summary=report, engineer_decision=decision, source_identifier=run_id)
        _commit_web_paths(((_write_web_json_temp(report_path, report), report_path), (temporary_validated, validated_path)))
    return {
        "run_id": run_id,
        "engineer_decision": decision,
        "model_status": "validated" if validated_path else manifest["model_status"],
        "validated_model_download": (
            f"/download/validated-model?run_id={run_id}" if validated_path else None
        ),
    }


def clear_data_session_cache() -> None:
    DATA_SESSIONS.clear()


@contextmanager
def _lifecycle_lock(run_id: str):
    with _LIFECYCLE_LOCKS_LOCK:
        lock = _LIFECYCLE_LOCKS.setdefault(run_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise ValueError("当前运行正在执行生命周期操作，不能并发提交")
    try:
        yield
    finally:
        lock.release()


def remove_data_session(file_id: str) -> None:
    DATA_SESSIONS.remove_dataset(file_id)


def _upload_source(payload: dict[str, Any]) -> tuple[str, Path, str]:
    file_id = _validated_id(_required_text(payload, "file_id"), "file_id")
    paths = [
        path
        for path in UPLOADS_DIR.glob(f"{file_id}.*")
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".txt"}
    ]
    if len(paths) != 1:
        DATA_SESSIONS.remove_dataset(file_id)
        raise WebStageError("loading", ValueError("上传文件不存在，请重新上传"))
    path = paths[0]
    if path.suffix.lower() == ".xlsx":
        return file_id, path, ""
    if path.suffix.lower() == ".txt":
        return file_id, path, TXT_ENCODING
    encoding = str(payload.get("encoding", "utf-8-sig"))
    if encoding not in {"utf-8-sig", "gb18030"}:
        raise WebStageError("loading", ValueError("CSV 编码仅支持 UTF-8-SIG 或 GB18030"))
    return file_id, path, encoding


def _upload_metadata(
    payload: dict[str, Any], timestamp_column: str
) -> tuple[DataSessionMetadata, bool]:
    file_id, path, encoding = _upload_source(payload)
    try:
        return DATA_SESSIONS.get_metadata(
            file_id, path, encoding, timestamp_column
        )
    except DataSessionStageError as error:
        raise WebStageError(error.stage, error) from error


def _load_upload(
    payload: dict[str, Any], requested_columns: Sequence[str] | None
) -> DataLoadResult:
    timestamp_column = _required_text(payload, "timestamp_column")
    file_id, path, encoding = _upload_source(payload)
    try:
        return DATA_SESSIONS.load_columns(
            file_id,
            path,
            encoding,
            timestamp_column,
            requested_columns,
        )
    except DataSessionStageError as error:
        raise WebStageError(error.stage, error) from error


def _build_training_matrix_with_stage(*args: Any, **kwargs: Any) -> Any:
    try:
        return build_training_matrix(*args, **kwargs)
    except WebStageError:
        raise
    except Exception as error:
        stage = (
            "quality_check"
            if "数据质量问题尚未处理" in str(error)
            else "preprocessing"
        )
        raise WebStageError(stage, error) from error


def _training_readiness(dynamic: pd.DataFrame) -> dict[str, Any]:
    try:
        _validate_dynamic_matrix(dynamic)
    except ValueError as error:
        message = str(error)
        return {
            "can_train": False,
            "issue": {
                "code": (
                    "dynamic_matrix_empty"
                    if dynamic.empty
                    else "insufficient_effective_rank"
                    if "有效秩不足" in message
                    else "dynamic_matrix_invalid"
                ),
                "severity": "error",
                "message": message,
                "count": len(dynamic),
                "tag": None,
                "details": {},
            },
        }
    return {"can_train": True, "issue": None}


def _load_required_upload(
    payload: dict[str, Any],
    requested_columns: Sequence[str],
    missing_prefix: str,
) -> DataLoadResult:
    try:
        return _load_upload(payload, requested_columns)
    except ValueError as error:
        missing = _missing_columns_from_error(error)
        if missing is None:
            raise
        raise ValueError(missing_prefix + ", ".join(missing)) from error


def _missing_columns_from_error(error: ValueError) -> list[str] | None:
    marker = "找不到列："
    message = str(error)
    if not message.startswith(marker):
        return None
    return [column.strip() for column in message[len(marker) :].split(",")]


def _read_upload(payload: dict[str, Any]) -> pd.DataFrame:
    """Compatibility wrapper for callers that still require all CSV columns."""
    return _load_upload(payload, None).frame


def _with_data_usage(
    result: dict[str, Any],
    loaded: DataLoadResult,
    analysis_row_count: int,
    display_point_count: int,
) -> dict[str, Any]:
    result["data_usage"] = {
        "source_row_count": loaded.metadata.row_count,
        "analysis_row_count": int(analysis_row_count),
        "display_point_count": int(display_point_count),
        "loaded_column_count": loaded.loaded_column_count,
        "cache_hit": loaded.cache_hit,
        "stage": "completed",
    }
    return result


def _web_temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
        return Path(handle.name)


def _write_web_json_temp(destination: Path, value: Any) -> Path:
    temporary = _web_temporary_path(destination)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return temporary


def _write_web_json_atomic(destination: Path, value: Any) -> None:
    temporary = _write_web_json_temp(destination, value)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _commit_web_paths(entries: Sequence[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for _, destination in entries:
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
                os.replace(destination, backup)
                backups.append((backup, destination))
        for temporary, destination in entries:
            os.replace(temporary, destination)
            committed.append(destination)
    except Exception:
        for destination in committed:
            destination.unlink(missing_ok=True)
        for backup, destination in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup, _ in backups:
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in entries:
            temporary.unlink(missing_ok=True)


def _commit_web_validation_artifacts(run_dir: Path, candidate_path: Path, model: Any, scores: pd.DataFrame, contributions: list[dict[str, Any]], report: dict[str, Any], timestamp_column: str, sample_interval_minutes: int) -> None:
    scores_path = run_dir / "validation_scores.csv"
    contributions_path = run_dir / "validation_contributions.json"
    report_path = run_dir / "validation_report.json"
    scores_temp, contributions_temp = _web_temporary_path(scores_path), _web_temporary_path(contributions_path)
    try:
        scores.to_csv(scores_temp, index_label=timestamp_column, encoding="utf-8-sig")
        contributions_temp.write_text(json.dumps(contributions, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence = build_validation_evidence(candidate_path, model, scores_temp, contributions_temp, timestamp_column=timestamp_column, scores_row_count=len(scores))
        evidence["scores"]["filename"] = scores_path.name
        evidence["contributions"]["filename"] = contributions_path.name
        report["validation_evidence"] = evidence
        report["validation_evidence"] = verify_validation_evidence(
            candidate_path,
            model,
            report,
            scores_temp,
            contributions_temp,
            sample_interval_minutes=sample_interval_minutes,
            artifact_filenames=(scores_path.name, contributions_path.name),
            scores_frame=scores,
        )
        report_temp = _write_web_json_temp(report_path, report)
        _commit_web_paths(((scores_temp, scores_path), (contributions_temp, contributions_path), (report_temp, report_path)))
    finally:
        scores_temp.unlink(missing_ok=True)
        contributions_temp.unlink(missing_ok=True)


def freeze_deployment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    with _lifecycle_lock(run_id):
        return _freeze_deployment_payload_locked(payload, run_id)


def _freeze_deployment_payload_locked(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    validated_path = run_dir / "validated_model.pcamodel"
    frozen_path = run_dir / "frozen_model.pcamodel"
    deployment_path = run_dir / "deployment_model.pcadeploy"
    if not validated_path.is_file():
        raise ValueError("当前运行尚未生成已验证模型")
    if frozen_path.exists() or deployment_path.exists():
        raise ValueError("冻结或部署模型包已存在，拒绝覆盖")
    load_model_package(validated_path)

    temporary_frozen = run_dir / f".frozen-{uuid.uuid4().hex}.pcamodel"
    temporary_deployment = run_dir / f".deployment-{uuid.uuid4().hex}.pcadeploy"
    committed_frozen = False
    try:
        freeze_validated_model_package(
            validated_path,
            temporary_frozen,
            model_id=_required_text(payload, "model_id"),
            model_version=payload.get("model_version"),
            frozen_by=_required_text(payload, "frozen_by"),
            comment=str(payload.get("comment", "")),
        )
        load_model_package(temporary_frozen)
        export_deployment_package(
            temporary_frozen,
            temporary_deployment,
            source_filename=frozen_path.name,
        )
        load_deployment_package(temporary_deployment)
        if frozen_path.exists() or deployment_path.exists():
            raise ValueError("冻结或部署模型包已存在，拒绝覆盖")
        os.replace(temporary_frozen, frozen_path)
        committed_frozen = True
        os.replace(temporary_deployment, deployment_path)
    except Exception:
        if committed_frozen:
            frozen_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_frozen.unlink(missing_ok=True)
        temporary_deployment.unlink(missing_ok=True)
    return {
        "run_id": run_id,
        "model_status": "frozen",
        "frozen_model_download": f"/download/frozen-model?run_id={run_id}",
        "deployment_model_download": f"/download/deployment-model?run_id={run_id}",
    }


def frozen_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    with _REPLAY_LOCKS_LOCK:
        lock = _REPLAY_LOCKS.setdefault(run_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise ValueError("当前运行正在执行冻结模型回放，不能并发提交")
    try:
        return _frozen_replay_payload_locked(payload, run_id)
    finally:
        lock.release()


def _frozen_replay_payload_locked(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    frozen_path = run_dir / "frozen_model.pcamodel"
    if not frozen_path.is_file():
        raise ValueError("当前运行尚未生成冻结模型")
    _, manifest = load_model_package(frozen_path)
    if (
        manifest.get("schema_version") not in {4, 5}
        or manifest.get("model_purpose") != "normal_state"
        or manifest.get("model_status") != "frozen"
    ):
        raise ValueError("当前运行的冻结模型不可用于历史回放")
    timestamp_column = _required_text(payload, "timestamp_column")
    config = preprocessing_config_from_mapping(manifest["config"])
    tags = list(manifest["config"]["tags"])
    requested_columns = list(dict.fromkeys([*tags, *(item.column for item in config.state_filters)]))
    loaded = _load_required_upload(payload, requested_columns, "找不到 Tag：")
    parsed = loaded.frame
    indexed = _replay_indexed_frame(parsed, timestamp_column, requested_columns)
    result = replay_frozen_model(
        frozen_path,
        indexed,
        _required_text(payload, "replay_start"),
        _required_text(payload, "replay_end"),
    )
    _commit_frozen_replay_artifacts(run_dir, result, timestamp_column)
    response = {
        "run_id": run_id,
        "model_purpose": "normal_state",
        "model_status": "frozen",
        "notice": "历史回放用于检查冻结模型在历史数据上的表现，不属于独立验证，不改变模型状态。",
        "summary": result.summary,
        "scores": _score_payload(result.scores),
        "contribution_count": len(result.contributions),
        "downloads": {
            artifact: f"/download/frozen-replay?run_id={run_id}&artifact={artifact}"
            for artifact in _FROZEN_REPLAY_ARTIFACTS
        },
    }
    return _with_data_usage(response, loaded, len(result.scores), len(response["scores"]))


def _replay_indexed_frame(
    frame: pd.DataFrame, timestamp_column: str, columns: Sequence[str]
) -> pd.DataFrame:
    timestamps = frame[timestamp_column]
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("回放时间戳必须递增且唯一")
    return frame.loc[:, [timestamp_column, *columns]].set_index(timestamp_column)


def _commit_frozen_replay_artifacts(
    run_dir: Path, result: FrozenReplayResult, timestamp_column: str
) -> None:
    destinations = [run_dir / filename for filename, _ in _FROZEN_REPLAY_ARTIFACTS.values()]
    temporary: list[Path] = []
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    committed_successfully = False
    try:
        for destination in destinations:
            with tempfile.NamedTemporaryFile(
                dir=run_dir, prefix=f".{destination.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary.append(Path(handle.name))
        result.scores.to_csv(temporary[0], index_label=timestamp_column, encoding="utf-8-sig")
        temporary[1].write_text(json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary[2].write_text(json.dumps(result.contributions, ensure_ascii=False, indent=2), encoding="utf-8")
        for destination in destinations:
            if destination.exists():
                backup = run_dir / f".{destination.name}.{uuid.uuid4().hex}.bak"
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in zip(temporary, destinations, strict=True):
            os.replace(source, destination)
            committed.append(destination)
        committed_successfully = True
    except Exception as commit_error:
        for destination in committed:
            destination.unlink(missing_ok=True)
        recovery_failures: list[tuple[Path, Exception]] = []
        for destination, backup in backups.items():
            if backup.exists():
                try:
                    os.replace(backup, destination)
                except Exception as recovery_error:
                    recovery_failures.append((backup, recovery_error))
        if recovery_failures:
            details = "; ".join(
                f"{backup}: {error}" for backup, error in recovery_failures
            )
            preserved = ", ".join(str(backup) for backup, _ in recovery_failures)
            raise RuntimeError(
                "frozen replay artifact commit failed: "
                f"{commit_error}; recovery failed: {details}; preserved backups: {preserved}"
            ) from commit_error
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        if committed_successfully:
            for path in backups.values():
                path.unlink(missing_ok=True)


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
    return preprocessing_config_from_mapping(
        {
            "sample_interval_minutes": payload.get("sample_interval_minutes", 5),
            "smoothing_window_minutes": payload.get("smoothing_window_minutes", 10),
            "max_lag_minutes": payload.get("max_lag_minutes", 60),
            "lag_step_minutes": payload.get("lag_step_minutes", 5),
            "resampling_method": payload.get("resampling_method", "none"),
            "filter_method": payload.get("filter_method", "none"),
            "first_order_alpha": payload.get("first_order_alpha"),
            "gap_threshold_minutes": payload.get("gap_threshold_minutes"),
            "state_filters": payload.get("state_filters", []),
        }
    )


def _state_filter_columns(payload: dict[str, Any]) -> list[str]:
    return [condition.column for condition in _preprocessing_config(payload).state_filters]


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


def _require_state_filter_roles(
    columns: Sequence[str], registry: dict[str, dict[str, Any]]
) -> None:
    invalid = [
        column
        for column in columns
        if column not in registry or registry[column]["role"] != "state_filter"
    ]
    if invalid:
        raise ValueError(
            "状态过滤列必须配置为state_filter角色：" + ", ".join(invalid)
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
        if registry[tag]["role"] != "exclude":
            raise ValueError(f"{tag}尚未确认排除")
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


def _frozen_replay_artifact(artifact: str) -> tuple[str, str]:
    try:
        return _FROZEN_REPLAY_ARTIFACTS[artifact]
    except KeyError as error:
        raise ValueError("无效的冻结回放工件类型") from error


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
        {"not_scored": -1, "normal": 0, "attention": 1, "abnormal": 2}
    ).fillna(-1).to_numpy(dtype=int)
    critical = {0, len(scores) - 1}
    critical.update(np.flatnonzero(severity > 0).tolist())
    critical.add(int(np.nanargmax(np.nan_to_num(scores["t2"].to_numpy(), nan=-np.inf))))
    critical.add(int(np.nanargmax(np.nan_to_num(scores["spe"].to_numpy(), nan=-np.inf))))
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
        if parsed.path.startswith("/api/state-exploration/"):
            suffix = parsed.path[len("/api/state-exploration/") :]
            is_series = suffix.endswith("/series")
            raw_run_id = suffix[: -len("/series")].rstrip("/") if is_series else suffix
            try:
                run_id = _validated_id(raw_run_id, "run_id")
                result = (
                    _exploration_series_payload(
                        run_id, parse_qs(parsed.query, keep_blank_values=True)
                    )
                    if is_series
                    else _exploration_summary_payload(run_id)
                )
                self._send_json(result)
            except StateExplorationNotFoundError as error:
                self._send_json(error_payload(error), 404)
            except Exception as error:
                self._send_json(error_payload(error), 400)
            return
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
                self._send_json(error_payload(error), 400)
            return
        if parsed.path == "/download/model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_model(run_id)
            except Exception as error:
                self._send_json(error_payload(error), 400)
            return
        if parsed.path == "/download/validated-model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_download(
                    RUNS_DIR / run_id / "validated_model.pcamodel",
                    "application/octet-stream",
                    "validated_model.pcamodel",
                )
            except Exception as error:
                self._send_json(error_payload(error), 400)
            return
        if parsed.path == "/download/frozen-model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_download(
                    RUNS_DIR / run_id / "frozen_model.pcamodel",
                    "application/zip",
                    "frozen_model.pcamodel",
                )
            except Exception as error:
                self._send_json(error_payload(error), 400)
            return
        if parsed.path == "/download/deployment-model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_download(
                    RUNS_DIR / run_id / "deployment_model.pcadeploy",
                    "application/zip",
                    "deployment_model.pcadeploy",
                )
            except Exception as error:
                self._send_json(error_payload(error), 400)
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
                self._send_json(error_payload(error), 400)
            return
        if parsed.path == "/download/frozen-replay":
            try:
                query = parse_qs(parsed.query)
                run_id = _validated_id(query.get("run_id", [""])[0], "run_id")
                filename, content_type = _frozen_replay_artifact(
                    query.get("artifact", [""])[0]
                )
                self._send_download(RUNS_DIR / run_id / filename, content_type, filename)
            except Exception as error:
                self._send_json(error_payload(error), 400)
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
            state_action = re.fullmatch(
                r"/api/state-exploration/([^/]+)/(decisions|training-windows)",
                parsed.path,
            )
            if state_action is not None:
                try:
                    run_id = _validated_id(state_action.group(1), "run_id")
                    payload = self._json_body()
                    result = (
                        state_exploration_decisions_payload(run_id, payload)
                        if state_action.group(2) == "decisions"
                        else state_exploration_training_windows_payload(run_id, payload)
                    )
                    self._send_json(result)
                except StateExplorationNotFoundError as error:
                    self._send_json(error_payload(error), 404)
                except Exception as error:
                    self._send_json(error_payload(error), 400)
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
            if parsed.path == "/api/preprocessing-preview":
                self._send_json(preprocessing_preview_payload(payload))
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
            if parsed.path == "/api/state-exploration/run":
                self._send_json(state_exploration_payload(payload))
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
            if parsed.path == "/api/validation-decision":
                self._send_json(validation_decision_payload(payload))
                return
            if parsed.path == "/api/freeze-deployment":
                self._send_json(freeze_deployment_payload(payload))
                return
            if parsed.path == "/api/frozen-replay":
                self._send_json(frozen_replay_payload(payload))
                return
            self._send_json({"error": "Not found"}, 404)
        except Exception as error:
            self._send_json(error_payload(error), 400)

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
    .tag-options label span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tag-row { display:grid !important; grid-template-columns:auto minmax(0,1fr) auto; cursor:pointer; padding:4px; border-radius:4px; }
    .tag-row > .tag-name { min-width:0; }
    .tag-row:not(.pending) > .tag-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tag-row > .tag-state { min-width:max-content; white-space:nowrap; }
    .tag-row.pending { grid-template-columns:minmax(0,1fr) max-content; }
    .tag-row.pending > .tag-name { white-space:normal; overflow-wrap:anywhere; }
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
    .metric.time-range strong { font-size:14px; line-height:1.35; white-space:pre-line; }
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
    .preprocessing-preview-chart { border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#fff; }
    .preprocessing-preview-chart svg { display:block; width:100%; height:auto; }
    .preprocessing-preview-details summary { cursor:pointer; color:var(--muted); font-size:13px; }
    .preprocessing-preview-details[open] summary { margin-bottom:9px; }
    .table-wrap { overflow:auto; max-height:360px; border:1px solid var(--line); border-radius:7px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { padding:8px 9px; border-bottom:1px solid var(--line-soft); text-align:left; }
    th { position:sticky; top:0; background:#eef2f6; }
    td.numeric { text-align:right; font-variant-numeric:tabular-nums; }
    .download { color:#fff; background:var(--green); padding:8px 11px; border-radius:6px; text-decoration:none; font-size:13px; }
     .validation-box { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:8px; align-items:end; padding:10px; background:#f8fafc; border:1px solid var(--line-soft); border-radius:8px; }
     .exploration-controls { display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:8px; align-items:end; padding:10px; background:#f8fafc; border:1px solid var(--line-soft); border-radius:8px; }
     .exploration-timeline { border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#fff; }
     .exploration-timeline svg { display:block; width:100%; height:auto; min-height:190px; }
     .exploration-timeline details { border-top:1px solid var(--line-soft); }
     .exploration-timeline summary { cursor:pointer; padding:9px 10px; color:var(--muted); font-size:12px; }
     .exploration-timeline .timeline-note { margin:0; padding:9px 10px; color:var(--muted); font-size:12px; line-height:1.45; }
     .exploration-timeline .timeline-detail { max-height:220px; overflow:auto; }
     .notice { padding:9px 10px; border-left:4px solid var(--warn); background:#fff8e7; color:#765000; font-size:13px; }
     @media (max-width:1050px) { main { grid-template-columns:1fr; } }
     @media (max-width:760px) { .chart-grid,.validation-box,.exploration-controls,.trend-controls { grid-template-columns:1fr; } .row,.condition-row { grid-template-columns:1fr; } }
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
        <label>CSV / XLSX / TXT 文件<input id="fileInput" type="file" accept=".csv,.xlsx,.txt,text/csv,text/plain,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"></label>
        <div class="actions"><button id="uploadButton">上传并读取列</button><button id="resetButton" class="secondary">清空</button></div>
        <div class="row">
          <label>时间列<select id="timestampColumn"></select></label>
          <label>CSV 编码<select id="encoding"><option value="utf-8-sig">UTF-8-SIG</option><option value="gb18030">GB18030</option></select></label>
        </div>
        <button id="inspectButton" class="secondary" disabled>检查时间轴与数值列</button>
      </div>
      <div class="group">
        <div class="group-title">2. 建模 Tag</div>
        <input id="tagSearch" placeholder="搜索 Tag">
        <div class="tag-toolbar"><button id="selectAllTags" class="secondary">全选</button><button id="clearAllTags" class="secondary">取消全选</button><button id="showProblemTags" class="secondary">只看问题Tag</button></div>
        <div id="tagOptions" class="tag-options"><span class="help">检查数据后显示连续数值列。</span></div>
        <div class="help">仅勾选且角色为“连续输入”的 Tag 进入 PCA；点击 Tag 在右侧查看配置与质量。</div>
      </div>
      <div class="group">
        <div class="group-title">3. 参考状态与 DPCA 参数</div>
        <div class="row"><label>候选开始<input id="candidateStart" type="datetime-local"></label><label>候选结束<input id="candidateEnd" type="datetime-local"></label><label>备注<input id="candidateComment" type="text"></label><button id="addManualCandidate" class="secondary" type="button">加入候选窗口</button></div>
        <h3>候选窗口列表</h3><div id="candidateWindows" class="table-wrap"><div class="empty">检查数据后可管理候选窗口。</div></div>
        <div class="help">候选窗口不会修改训练窗口。先记录人工决策，再确认作为训练窗口。</div>
        <h3>排除窗口</h3><div id="excludedWindows" class="table-wrap"><div class="empty">尚无排除窗口。</div></div>
        <div class="help">排除窗口仅在确认候选时切分新的训练窗口，不会修改已生成的训练窗口。</div>
        <h3>训练窗口</h3><div id="trainingWindows" class="table-wrap"><div class="empty">尚无已确认训练窗口。</div></div>
        <div class="help">只有此处的 training_windows 会参与质量检查和训练。</div>
        <div class="row"><label>目标采样周期（分钟）<input id="sampleInterval" type="number" min="1" value="5"></label><label>重采样方法<select id="resamplingMethod"><option value="none">不重采样</option><option value="mean">均值</option><option value="median">中位数</option><option value="last">最后值</option></select></label></div>
        <div class="row"><label>滤波方法<select id="filterMethod"><option value="none" selected>不滤波</option><option value="first_order">一阶滤波</option><option value="trailing_mean">尾随均值</option></select></label><label>一阶滤波 alpha<input id="firstOrderAlpha" type="number" min="0" max="1" step="any" placeholder="仅一阶滤波需要" disabled></label><label>滤波窗口（分钟）<input id="smoothingWindow" type="number" min="0" value="10" disabled></label></div>
        <div class="row"><label>物理缺口阈值（分钟，可选）<input id="gapThreshold" type="number" min="1" placeholder="沿用默认规则"></label><div><button id="preprocessingPreviewButton" class="secondary" disabled>预览预处理</button><div id="preprocessingPreview" class="muted">尚未预览</div></div></div>
        <div class="row"><label>最大 Lag（分钟）<input id="maxLag" type="number" min="0" value="60"></label><label>Lag 步长（分钟）<input id="lagStep" type="number" min="1" value="5"></label></div>
        <div class="row"><label>累计解释率<input id="varianceThreshold" type="number" min="0.01" max="0.99" step="0.01" value="0.95"></label><label>主元数（可留空）<input id="components" type="number" min="2" placeholder="自动，至少2个"></label></div>
        <label>模型名称<input id="modelName" value="D330_DPCA_Model_V1"></label>
        <h3>建模质量检查</h3>
        <div id="modelQualityStatus" class="status info" role="status">未检查</div>
        <button id="qualityButton" class="secondary" disabled>执行建模质量检查</button>
        <div id="modelQualityResults">
          <div id="qualitySummary" class="metrics"></div>
          <h3>当前 Tag 建模质量详情</h3>
          <label>查看 Tag：<select id="qualityTagSelect" disabled></select></label>
          <div id="currentTagQuality" class="empty">尚未执行建模质量检查。</div>
          <h3>建模质量问题</h3>
          <div class="actions"><button id="excludeAllConstants" class="secondary" disabled>排除全部精确常量 Tag</button></div>
          <div id="qualityIssues" class="empty">执行建模质量检查后，显示需要确认或阻止训练的 Tag。</div>
        </div>
        <div class="actions"><button id="trainExploratoryButton" class="secondary" disabled>建立探索模型</button><button id="trainButton" disabled>建立正常状态候选模型</button></div>
        <div class="notice">探索模型仅用于状态空间浏览和聚类辅助，不能作为正常状态模型。</div>
        <div class="notice">正常状态候选模型尚未验证，不能发布或用于部署。</div>
        <div class="notice">聚类结果必须由工程师判断，不能自动定义正常状态。</div>
        <div class="notice">探索模型和正常状态候选模型均不提供根因、因果或控制建议。</div>
      </div>
      <div id="status" class="status info" role="status" aria-live="polite">请先上传 CSV。</div>
      <div class="help">时间戳重复、乱序或无法满足采样时间轴契约会阻断训练；建模 Tag 或启用状态过滤列中的缺失、非数字、NaN、Inf 在重采样后删除整行并重新分段；不插值、不补点、不自动修复异常值。</div>
    </section>
    <section class="results">
      <div class="tabs" role="tablist">
         <button class="tab active" data-panel="configPanel">Tag配置</button>
         <button class="tab" data-panel="stateExplorationPanel">状态探索</button>
         <button class="tab" data-panel="trendPanel">趋势浏览</button>
        <button class="tab" data-panel="statePanels">状态辅助</button>
        <button class="tab" data-panel="modelPanel">模型训练</button>
        <button class="tab" data-panel="validationPanel">验证结果</button>
      </div>
      <div id="configPanel" class="panel active">
        <div class="inner-tabs">
          <button class="inner-tab active" data-inner="engineeringPanel">工程配置</button>
          <button class="inner-tab" data-inner="qualityPanel">基础数据检查</button>
        </div>
        <div id="engineeringPanel" class="inner-panel active">
          <div class="batch-config">
            <div class="batch-config-title">批量配置</div>
            <div class="actions"><a id="templateDownload" class="download" href="#">下载XLSX模板</a><label class="secondary">导入XLSX配置<input id="tagConfigFile" type="file" accept=".xlsx"></label><button id="importConfigButton" class="secondary" disabled>预览导入</button><button id="applyConfigButton" disabled>确认应用非空字段</button><button id="exportConfigButton" class="secondary" disabled>导出当前配置</button></div>
            <div id="importSummary" class="status info">XLSX是可选工程元数据，导入不会跳过质量检查，也不会立即覆盖当前配置。</div>
          </div>
          <h3 id="selectedTagTitle">请选择左侧Tag</h3>
          <div class="detail-fields">
            <div class="row"><label>描述<input id="tagDescription"></label><label>单位<input id="tagUnit"></label></div>
            <div class="row"><label>变量角色<select id="tagRole"><option value="continuous_input">连续输入</option><option value="state_filter">状态过滤</option><option value="label_only">仅标签</option><option value="exclude">排除</option></select></label><label>备注<textarea id="tagComment"></textarea></label></div>
            <div class="row"><label>工程下限<input id="engineeringMin" type="number" step="any"></label><label>工程上限<input id="engineeringMax" type="number" step="any"></label></div>
            <div class="row"><label>正常下限<input id="normalMin" type="number" step="any"></label><label>正常上限<input id="normalMax" type="number" step="any"></label></div>
            <div class="row"><label>报警下限<input id="alarmMin" type="number" step="any"></label><label>报警上限<input id="alarmMax" type="number" step="any"></label></div>
            <button id="saveTagConfig" class="secondary">保存当前Tag配置</button>
          </div>
        </div>
        <div id="qualityPanel" class="inner-panel">
          <h3>上传后基础数据检查</h3>
          <div class="help">此处仅展示整体历史数据的时间轴与原始逐列检查结果。建模质量检查位于“模型训练”阶段，并依赖已启用的训练窗口。</div>
          <div id="basicInspectionSummary" class="metrics"></div>
          <div id="basicInspectionIssues" class="empty">上传并检查数据后显示基础检查结果。</div>
        </div>
      </div>
      <div id="stateExplorationPanel" class="panel">
        <div class="group">
          <div class="group-title">状态探索工作台</div>
          <div class="help">建模 Tag 复用左侧当前勾选的“连续输入”；预处理参数复用左侧表单。探索结果仅用于运行状态浏览和候选窗口比较。</div>
          <div class="exploration-controls">
            <label>探索开始时间<input id="explorationStart" type="datetime-local"></label>
            <label>探索结束时间<input id="explorationEnd" type="datetime-local"></label>
            <label>Cluster 数量<input id="explorationClusterCount" type="number" min="2" max="10" value="4"></label>
            <label>随机种子<input id="explorationRandomState" type="number" step="1" value="0"></label>
            <label>每个 Cluster 候选数量<input id="explorationCandidateCount" type="number" min="1" value="3"></label>
            <label>候选最小时长（分钟）<input id="explorationMinimumDuration" type="number" min="1" value="30"></label>
            <label>最大显示点数<input id="explorationMaximumPlotPoints" type="number" min="2" value="1200"></label>
            <button id="stateExplorationButton" type="button" disabled>运行状态探索</button>
          </div>
          <div class="sub-title">可选性能候选</div>
          <div class="exploration-controls">
            <label>性能 Tag<select id="explorationPerformanceTag"><option value="">不配置</option></select></label>
            <label>性能方向<select id="explorationPerformanceDirection"><option value="higher_is_better">越高越好</option><option value="lower_is_better">越低越好</option><option value="target_range">目标范围内</option></select></label>
            <label>目标下限<input id="explorationTargetMin" type="number" step="any"></label>
            <label>目标上限<input id="explorationTargetMax" type="number" step="any"></label>
            <label>性能候选最小时长（分钟）<input id="explorationPerformanceMinimumDuration" type="number" min="1" value="30"></label>
            <label>性能候选数量<input id="explorationPerformanceCandidateCount" type="number" min="1" value="3"></label>
          </div>
        </div>
        <div id="explorationEmpty" class="empty">运行状态探索后显示摘要、告警、PC1/PC2、Cluster 时间轴和候选窗口。</div>
        <div id="explorationContent" hidden>
          <h3>结果概览</h3>
          <div id="explorationOverview" class="metrics"></div>
          <div id="explorationWarnings" class="compact-list"><span class="help">暂无结构化告警。</span></div>
          <h3>预处理损失摘要</h3>
          <div id="explorationLossSummary" class="table-wrap"></div>
          <div class="chart-grid">
            <div class="chart-card"><h3>Cluster PC1 / PC2 与中心</h3><div id="explorationPcChart" class="chart"></div></div>
            <div class="chart-card"><h3>Cluster 时间轴</h3><div id="explorationTimeline" class="exploration-timeline"><div class="empty">暂无显示序列。</div></div></div>
          </div>
          <h3>Cluster 摘要表</h3>
          <div class="table-wrap"><table><thead><tr><th>Cluster ID</th><th>样本数</th><th>覆盖率</th><th>连续段数</th><th>覆盖时长</th><th>中心距离中位数</th><th>主元离散度</th><th>候选数量</th></tr></thead><tbody id="explorationClusterTable"></tbody></table></div>
          <h3>Cluster 候选表</h3>
          <div id="explorationClusterCandidates" class="table-wrap"></div>
          <h3>性能候选表</h3>
          <div id="explorationPerformanceCandidates" class="table-wrap"></div>
          <div class="actions"><button id="saveExplorationCandidateDecisions" class="secondary" type="button">保存所选候选决策</button><button id="convertExplorationCandidates" type="button">加入候选窗口</button></div>
          <div class="notice">接受仅表示允许加入候选窗口，不会自动参与训练；加入后仍需工程师人工确认，才能生成训练窗口。</div>
        </div>
      </div>
      <div id="trendPanel" class="panel">
        <div class="trend-controls">
          <label>趋势Tag（最多8个）<select id="trendTags" multiple size="6"></select></label>
          <div><label>时间范围<select id="trendPreset"><option value="all">全部数据</option><option value="1">最近1天</option><option value="3">最近3天</option><option value="7">最近7天</option><option value="custom">自定义</option><option value="reference">参考状态期</option><option value="validation">验证期</option></select></label><div class="row"><label>开始<input id="trendStart" type="datetime-local"></label><label>结束<input id="trendEnd" type="datetime-local"></label></div></div>
          <div><label>显示<select id="trendMode"><option value="raw">原始值</option><option value="smoothed">因果滤波值</option><option value="both" selected>原始值和因果滤波值</option></select></label><label>缩放<input id="trendZoom" type="range" min="1" max="5" value="1"></label><button id="trendButton" class="secondary" disabled>浏览趋势</button></div>
        </div>
        <div class="actions"><button id="trendToAnalysis" class="secondary">将当前窗口设为分析期</button><button id="trendToReference" class="secondary">加入候选窗口</button><label>直方图范围<select id="histogramScope"><option value="current">当前窗口</option><option value="reference">参考期</option></select></label></div>
        <div id="trendChart" class="trend-chart"><div class="empty">选择Tag和时间范围后浏览原始值、尾随平滑、缺口及工程范围。</div></div>
        <div id="trendStats" class="table-wrap"></div>
        <div id="trendHistogram" class="chart"></div>
      </div>
      <div id="modelPanel" class="panel">
        <div id="modelEmpty" class="empty">完成训练后显示主元解释率、T²/SPE 和模型下载。</div>
        <div id="modelContent" hidden>
          <div id="modelMetrics" class="metrics"></div>
          <h3>训练窗口与连续段</h3><div id="trainingWindowSummary" class="table-wrap"></div>
          <div id="trainingQualityWarnings" class="hint"></div>
          <h3>主元解释率</h3><div id="varianceChart" class="variance"></div>
          <div class="chart-grid">
            <div class="chart-card"><h3>训练期 T²</h3><div id="t2Chart" class="chart"></div></div>
            <div class="chart-card"><h3>训练期 SPE/Q</h3><div id="speChart" class="chart"></div></div>
          </div>
          <div class="chart-card"><h3>主元得分 PC1 / PC2</h3><div id="scoreChart" class="chart"></div></div>
          <div class="legend"><span><i class="swatch" style="background:var(--accent)"></i>统计量</span><span><i class="swatch" style="background:var(--attention)"></i>95% 边界</span><span><i class="swatch" style="background:var(--abnormal)"></i>99% 边界</span></div>
          <div class="actions"><a id="modelDownload" class="download" href="#">下载模型包</a></div>
          <div id="modelLifecycleNotice" class="notice"></div>
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
          <label>验证类型<select id="validationType"><option value="normal_validation">正常样本验证</option><option value="known_abnormal">已知异常验证</option></select></label>
          <label>验证备注<input id="validationComment" type="text"></label>
          <button id="addValidationWindow" class="secondary" type="button">加入验证时段</button>
          <label>工程标签列（可选）<select id="labelColumn"><option value="">不使用</option></select></label>
          <button id="validateButton" disabled>回放独立验证期</button>
        </div>
        <div class="table-wrap"><table><thead><tr><th>类型</th><th>开始</th><th>结束</th><th>备注</th><th>操作</th></tr></thead><tbody id="validationWindowTable"></tbody></table></div>
        <div id="validationEmpty" class="empty">只有正常状态候选模型可以执行独立验证。</div>
        <div id="validationContent" hidden>
          <h3>验证状态摘要</h3>
          <div id="validationMetrics" class="metrics"></div>
          <div class="validation-box"><label>工程师结论<select id="validationDecision"><option value="passed">通过</option><option value="insufficient">结论不足</option><option value="failed">不通过</option></select></label><label>审查备注<input id="validationDecisionComment" type="text"></label><button id="recordValidationDecision" type="button">保存人工结论</button><div id="validationDecisionStatus" class="status info" role="status" aria-live="polite">等待保存工程师结论。</div><a id="validatedModelDownload" class="download" href="#" hidden>下载已验证模型包</a></div>
          <h3>验证指标</h3>
          <div id="validationMetricDetails" class="table-wrap"></div>
          <div class="chart-grid">
            <div class="chart-card"><h3>验证期 T²</h3><div id="validationT2Chart" class="chart"></div></div>
            <div class="chart-card"><h3>验证期 SPE/Q</h3><div id="validationSpeChart" class="chart"></div></div>
          </div>
          <h3>主要贡献 Tag</h3>
          <div class="help" id="contributionHint"></div>
          <div class="table-wrap"><table><thead><tr><th>事件 / 峰值</th><th>统计量</th><th>Tag</th><th>描述</th><th>单位</th><th>贡献</th><th>主要影响时间</th></tr></thead><tbody id="contributionTable"></tbody></table></div>
          <h3>贡献稳定性</h3>
          <div id="contributionStability" class="table-wrap"></div>
          <div class="actions"><a id="scoresDownload" class="download" href="#">下载完整评分 CSV</a><a id="reportDownload" class="download" href="#">下载验证摘要</a><a id="contributionsDownload" class="download" href="#">下载贡献记录</a></div>
          <div class="validation-box"><label>模型标识<input id="frozenModelId" type="text"></label><label>模型版本<input id="frozenModelVersion" type="number" min="1" step="1" value="1"></label><label>冻结人<input id="frozenBy" type="text"></label><label>冻结备注<input id="freezeComment" type="text"></label><button id="freezeDeployment" type="button">冻结并导出部署包</button><a id="frozenModelDownload" class="download" href="#" hidden>下载冻结模型包</a><a id="deploymentModelDownload" class="download" href="#" hidden>下载部署模型包</a></div>
          <div class="help">每次回放会更新当前模型最近一次验证的下载文件；候选模型不会被验证结果原地修改。</div>
          <div class="notice">验证指标和贡献稳定性只提供工程证据，不能替代工程师确认。贡献表示该时间点偏离在模型中的来源，不等同于工艺根因；最终通过或不通过由工程师确认。frozen表示工程冻结，不表示已部署或已进入模型治理平台。</div>
        </div>
      </div>
    </section>
  </main>
<script>
const state = { fileId:null, runId:null, exploratoryRunId:null, inspection:null, clustering:null, exploration:null, performance:null, training:null, trend:null, preprocessingPreview:null, preprocessingPreviewTag:null, registry:{}, quality:null, qualityStatus:"unchecked", qualityError:"", selectedTag:null, selectedModelTags:new Set(), importPreview:null, excludedTags:[], excludedWindows:[], showProblems:false, candidateWindows:[], trainingWindows:[], trainingWindowSummary:[], validationWindows:[] };
const el = (id) => document.getElementById(id);

function setStatus(message, type="info") { const node=el("status"); node.textContent=message; node.className=`status ${type}`; }
function setBusy(button, busy, text) { if (!button.dataset.label) button.dataset.label=button.textContent; button.disabled=busy; button.textContent=busy?text:button.dataset.label; }
function localTime(value) { return value ? value.slice(0,16) : ""; }
function displayTime(value,length=16) { return value ? value.slice(0,length).replace("T"," ") : ""; }
function selectedTags() { return (state.inspection?.numeric_columns||[]).filter(tag=>state.selectedModelTags.has(tag)&&(state.registry[tag]?.role||"continuous_input")==="continuous_input"); }
function numberValue(id) { return Number(el(id).value); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch])); }
function displayUiValue(value) { const labels={continuous_input:"连续输入",state_filter:"状态过滤",label_only:"仅标签",exclude:"排除",manual:"手工选择",trend:"趋势选择",cluster:"聚类推荐",performance:"性能辅助",suggested:"系统建议",pending:"待决策",accepted:"已接受",rejected:"已拒绝",used:"已使用",dropped:"已丢弃",disabled:"已禁用",higher_is_better:"越高越好",lower_is_better:"越低越好",target_range:"目标范围内",no_raw_samples:"没有原始样本",no_complete_resampling_bins:"无完整重采样时间桶",insufficient_after_smoothing_and_lag:"滤波与 Lag 后样本不足",normal:"正常",attention:"关注",abnormal:"异常",usable:"可用",review:"需确认",blocking:"阻止"}; return labels[value]||value; }
function formField(labelText,field,type="text") { const label=document.createElement("label"); label.textContent=labelText; const input=document.createElement("input"); input.type=type; input.dataset.field=field; if(type==="number") input.step="any"; label.append(input); return label; }
function emptyTagConfig() { return {description:"",unit:"",role:"continuous_input",engineering_min:null,engineering_max:null,normal_min:null,normal_max:null,alarm_min:null,alarm_max:null,comment:""}; }
function tagConfigPayload() { return state.registry; }
function qualityFor(tag) { return state.quality?.tags?.find(item=>item.tag===tag)||null; }
function setTagExclusion(tag, record) { state.registry[tag]={...state.registry[tag],role:"exclude"}; state.selectedModelTags.delete(tag); state.excludedTags=[...state.excludedTags.filter(value=>value.tag!==tag),record]; }
function reconcileExcludedTags() { const candidates=new Set(state.inspection?.numeric_columns||[]); const existing=new Map(state.excludedTags.map(record=>[record.tag,record])); state.excludedTags=[...candidates].filter(tag=>state.registry[tag]?.role==="exclude").map(tag=>existing.get(tag)||{tag,reason:"manual_exclude"}); }
function confirmSuggestedExclusion(profile) { if(!state.inspection?.numeric_columns.includes(profile.tag)||!profile.suggestion) return; setTagExclusion(profile.tag,{tag:profile.tag,reason:profile.suggestion.reason}); if(state.selectedTag===profile.tag) selectTag(profile.tag); invalidateQuality(`${profile.tag}已确认排除`); renderBasicInspection(state.inspection); renderTagList(); }
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
    const name=document.createElement("span"); name.className="tag-name"; name.title=tag; name.textContent=tag;
    const badge=document.createElement("span"); badge.className=`tag-state ${status}`; badge.textContent=config.role!=="continuous_input"?displayUiValue(config.role):displayUiValue(status);
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
  const tag=state.selectedTag, previousRole=state.registry[tag]?.role; const config={description:el("tagDescription").value.trim(),unit:el("tagUnit").value.trim(),role:el("tagRole").value,comment:el("tagComment").value.trim(),engineering_min:optionalNumber("engineeringMin"),engineering_max:optionalNumber("engineeringMax"),normal_min:optionalNumber("normalMin"),normal_max:optionalNumber("normalMax"),alarm_min:optionalNumber("alarmMin"),alarm_max:optionalNumber("alarmMax")}; state.registry[tag]=config;
  if(config.role==="exclude"&&previousRole!=="exclude") setTagExclusion(tag,{tag,reason:"manual_exclude"}); else { if(previousRole==="exclude"&&config.role==="continuous_input") state.selectedModelTags.add(tag); else if(config.role!=="continuous_input") state.selectedModelTags.delete(tag); reconcileExcludedTags(); }
  invalidateQuality("Tag工程配置或角色已修改"); renderTagList();
}
function renderModelQualityStatus(status=state.qualityStatus,error=state.qualityError) {
  const node=el("modelQualityStatus"); if(!node) return;
  const labels={unchecked:"未检查",checking:"检查中",passed:"通过",issues:"有问题",changed:"配置已变更需重新检查",failed:"失败"};
  node.textContent=error?`${labels[status]}：${error}`:labels[status];
  node.className=`status ${status==="passed"?"success":status==="issues"||status==="changed"?"warning":status==="failed"?"error":"info"}`;
}
function invalidateQuality(reason) {
  const checked=Boolean(state.quality); state.quality=null; state.qualityError=""; state.qualityStatus=reason&&checked?"changed":"unchecked";
  el("trainButton").disabled=true; el("trainExploratoryButton").disabled=true;
  if(el("qualitySummary")) el("qualitySummary").innerHTML="";
  if(el("qualityIssues")) { el("qualityIssues").className="empty"; el("qualityIssues").textContent=state.qualityStatus==="changed"?"配置已变更，请重新执行建模质量检查。":"尚未执行建模质量检查。"; }
  el("excludeAllConstants").disabled=true; renderModelQualityStatus(); renderCurrentTagQuality();
  if(reason) setStatus(`${reason}，请重新执行建模质量检查。`,"warning");
}
function commonPayload() { const gap=el("gapThreshold").value, filterMethod=el("filterMethod").value, alpha=el("firstOrderAlpha").value; if(filterMethod==="first_order"&&(!Number.isFinite(Number(alpha))||!(Number(alpha)>0&&Number(alpha)<=1))) throw new Error("一阶滤波 alpha 必须大于 0 且不超过 1。"); return {file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tag_configs:tagConfigPayload(),sample_interval_minutes:numberValue("sampleInterval"),resampling_method:el("resamplingMethod").value,filter_method:filterMethod,first_order_alpha:filterMethod==="first_order"?Number(alpha):null,smoothing_window_minutes:filterMethod==="trailing_mean"?numberValue("smoothingWindow"):0,gap_threshold_minutes:gap===""?null:Number(gap),max_lag_minutes:numberValue("maxLag"),lag_step_minutes:numberValue("lagStep")}; }
function candidateId() { return globalThis.crypto?.randomUUID?.() || `window-${Date.now()}-${Math.random().toString(16).slice(2)}`; }
function trainingWindowsPayload() { return state.trainingWindows; }
function updateQualityButtonAvailability() { el("qualityButton").disabled=!state.inspection||!state.trainingWindows.some(window=>window.enabled); }
function windowSummary(id) { return state.trainingWindowSummary.find(item=>item.id===id)||{}; }
function candidateTrainingWindows(candidate) { const baseId=`training-${candidate.id}`; return state.trainingWindows.filter(window=>window.id===baseId||window.id.startsWith(`${baseId}-part-`)); }
function mergeExcludedWindows(windows) { const sorted=windows.map(window=>({...window})).sort((left,right)=>Date.parse(left.start)-Date.parse(right.start)||Date.parse(left.end)-Date.parse(right.end)||left.id.localeCompare(right.id)); return sorted.reduce((merged,window)=>{ const previous=merged.at(-1); if(previous&&Date.parse(window.start)<=Date.parse(previous.end)) { if(Date.parse(window.end)>Date.parse(previous.end)) previous.end=window.end; return merged; } merged.push(window); return merged; },[]); }
function renderExcludedWindows() {
  const container=el("excludedWindows"); if(!container) return; container.replaceChildren();
  if(!state.excludedWindows.length) { container.innerHTML='<div class="empty">尚无排除窗口。</div>'; return; }
  const table=document.createElement("table"), head=document.createElement("thead"), body=document.createElement("tbody"), header=document.createElement("tr"); ["开始","结束","来源","备注","操作"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  state.excludedWindows.forEach(window=>{ const row=document.createElement("tr"); [displayTime(window.start),displayTime(window.end),displayUiValue(window.source),window.comment||"—"].forEach(value=>{ const cell=document.createElement("td"); cell.textContent=value; row.append(cell); }); const actions=document.createElement("td"), button=document.createElement("button"); button.className="secondary"; button.type="button"; button.textContent="删除"; button.addEventListener("click",()=>removeExcludedWindow(window.id)); actions.append(button); row.append(actions); body.append(row); }); table.append(head,body); container.append(table);
}
function exclusionOverlapsTraining(window) { const start=Date.parse(window.start), end=Date.parse(window.end); return state.trainingWindows.some(training=>start<=Date.parse(training.end)&&Date.parse(training.start)<=end); }
function addExcludedWindow(source,start,end,sourceRef=null,comment="") { if(!start||!end||!Number.isFinite(Date.parse(start))||!Number.isFinite(Date.parse(end))||Date.parse(start)>Date.parse(end)) { setStatus("排除窗口需要有效的开始和结束时间。","warning"); return; } const window={id:`excluded-${candidateId()}`,start,end,source,comment}; state.excludedWindows=mergeExcludedWindows([...state.excludedWindows,window]); renderExcludedWindows(); globalThis.refreshTrendExcludedWindows?.(); globalThis.showWorkflowStage?.("candidatePanel"); const overlaps=exclusionOverlapsTraining(window); setStatus(overlaps?"排除窗口已加入；不会修改已有训练窗口。请删除关联训练窗口后重新确认候选。":"排除窗口已加入；确认候选时将据此切分训练窗口。",overlaps?"warning":"success"); }
function removeExcludedWindow(windowId) { state.excludedWindows=state.excludedWindows.filter(window=>window.id!==windowId); renderExcludedWindows(); globalThis.refreshTrendExcludedWindows?.(); setStatus("排除窗口已删除；已有训练窗口未被修改，如需重新切分请先删除关联训练窗口后重新确认。","warning"); }
function showCandidateTrend(window) { el("trendStart").value=localTime(window.start); el("trendEnd").value=localTime(window.end); if(el("dpTrendStart")) el("dpTrendStart").value=localTime(window.start); if(el("dpTrendEnd")) el("dpTrendEnd").value=localTime(window.end); document.querySelector('[data-panel="trendPanel"]').click(); setStatus("已切换到候选时段趋势；训练候选未改变。","success"); }
function renderCandidateWindows() {
  const container=el("candidateWindows"); container.replaceChildren();
  if(!state.candidateWindows.length) { container.innerHTML='<div class="empty">尚无候选窗口。</div>'; return; }
  const table=document.createElement("table"), head=document.createElement("thead"), body=document.createElement("tbody");
  const header=document.createElement("tr"); ["窗口","来源","时间范围","状态","操作"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  state.candidateWindows.forEach(window=>{ const row=document.createElement("tr");
    const name=document.createElement("td"); name.textContent=window.id;
    const source=document.createElement("td"); source.textContent=displayUiValue(window.source)+(window.source_ref?` (${window.source_ref})`:"");
    const range=document.createElement("td"); range.textContent=`${displayTime(window.start)} ～ ${displayTime(window.end)}`;
    const generated=candidateTrainingWindows(window).length>0; const status=document.createElement("td"); const select=document.createElement("select"); ["pending","accepted","rejected"].forEach(value=>{ const option=document.createElement("option"); option.value=value; option.textContent=displayUiValue(value); option.selected=window.status===value; select.append(option); }); select.disabled=generated; select.addEventListener("change",()=>{ window.status=select.value; renderCandidateWindows(); }); status.append(select); if(generated) { const note=document.createElement("div"); note.textContent="已生成训练窗口"; status.append(note); }
    const actions=document.createElement("td"); [["查看趋势",()=>showCandidateTrend(window)],["确认作为训练窗口",()=>confirmCandidateWindow(window)],["删除",()=>{ state.candidateWindows=state.candidateWindows.filter(item=>item.id!==window.id); renderCandidateWindows(); }]].forEach(([label,handler])=>{ const button=document.createElement("button"); button.className="secondary"; button.type="button"; button.textContent=label; button.disabled=label==="确认作为训练窗口"&&(window.status!=="accepted"||generated); button.addEventListener("click",handler); actions.append(button); });
    row.append(name,source,range,status,actions); body.append(row);
  }); table.append(head,body); container.append(table);
}
function renderTrainingWindows() {
  const container=el("trainingWindows"); container.replaceChildren();
  if(!state.trainingWindows.length) { container.innerHTML='<div class="empty">尚无已确认训练窗口。</div>'; return; }
  const table=document.createElement("table"), head=document.createElement("thead"), body=document.createElement("tbody");
  const header=document.createElement("tr"); ["参与训练","来源","开始","结束","持续时间","原始 / 有效","质量","备注","操作"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  state.trainingWindows.forEach(window=>{ const summary=windowSummary(window.id); const row=document.createElement("tr");
    const enabled=document.createElement("input"); enabled.type="checkbox"; enabled.checked=window.enabled; enabled.addEventListener("change",()=>updateTrainingWindows({action:"set_enabled",id:window.id,enabled:enabled.checked},true)); const enabledCell=document.createElement("td"); enabledCell.append(enabled);
    const source=document.createElement("td"); source.textContent=displayUiValue(window.source)+(window.source_ref?` (${window.source_ref})`:"");
    const start=document.createElement("td"); start.textContent=displayTime(window.start); const end=document.createElement("td"); end.textContent=displayTime(window.end);
    const durationMinutes=summary.duration_minutes??Math.round((new Date(window.end)-new Date(window.start))/60000); const duration=document.createElement("td"); duration.textContent=`${durationMinutes} 分钟`;
    const counts=document.createElement("td"); counts.textContent=summary.raw_samples===undefined?"待检查":`${summary.raw_samples} / ${summary.effective_samples}`;
    const quality=document.createElement("td"); quality.textContent=displayUiValue(summary.quality_status||summary.status||"待检查");
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
  try { const previous=JSON.stringify(state.trainingWindows); const data=await api("/api/training-windows",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...commonPayload(),training_windows:trainingWindowsPayload(),operation})}); state.trainingWindows=data.training_windows; state.trainingWindowSummary=data.summary; renderTrainingWindows(); renderCandidateWindows(); updateQualityButtonAvailability(); if(affectsTraining&&previous!==JSON.stringify(state.trainingWindows)) invalidateQuality("人工确认的训练窗口已修改"); return true; }
  catch(error) { renderTrainingWindows(); setStatus(error.message,"error"); return false; }
}
async function confirmCandidateWindow(candidate) { if(candidate.status!=="accepted") { setStatus("请先将候选标记为已接受。","warning"); return; } if(candidateTrainingWindows(candidate).length) { setStatus("该候选已生成训练窗口。","warning"); return; } const added=await updateTrainingWindows({action:"confirm_candidate",candidate,excluded_windows:state.excludedWindows},true); if(added) { const count=candidateTrainingWindows(candidate).length; renderCandidateWindows(); el("trainingWindows").scrollIntoView({behavior:"smooth",block:"start"}); setStatus(`已确认并生成 ${count} 个训练窗口；它们将分别参与质量检查和训练。`,"success"); } }
async function addCandidateWindow(source,start,end,sourceRef=null,comment="",status="pending") { if(!start||!end) { setStatus("候选窗口需要开始和结束时间。","warning"); return; } if(sourceRef&&state.candidateWindows.some(window=>window.source_ref===sourceRef)) { setStatus("该候选已在候选窗口列表中。","warning"); return; } state.candidateWindows.push({id:candidateId(),start,end,source,source_ref:sourceRef,comment,status}); renderCandidateWindows(); globalThis.showWorkflowStage?.("candidatePanel"); el("candidateWindows").scrollIntoView({behavior:"smooth",block:"start"}); setStatus("候选窗口已加入；不会修改训练窗口，请人工确认后再生成训练窗口。","success"); }
function editTrainingWindow(window) { const start=prompt("训练窗口开始时间",displayTime(window.start)); if(start===null) return; const end=prompt("训练窗口结束时间",displayTime(window.end)); if(end===null) return; const comment=prompt("备注",window.comment||""); if(comment===null) return; updateTrainingWindows({action:"update",id:window.id,changes:{start,end,comment}},window.enabled); }

async function api(path, options={}) {
  const response = await fetch(path, options);
  let data;
  try { data=await response.json(); }
  catch(error) { throw new Error(`服务返回无法读取的响应（${response.status}）`); }
  if (!response.ok) throw new Error(data.error || `请求失败：${response.status}`);
  return data;
}

function ensureInspectionPageReady() {
  const ids=["tagOptions","selectedTagTitle","tagDescription","tagUnit","tagRole","tagComment","engineeringMin","engineeringMax","normalMin","normalMax","alarmMin","alarmMax","candidateWindows","excludedWindows","trainingWindows","validationWindowTable","trendTags","explorationPerformanceTag","performanceConditions","basicInspectionSummary","basicInspectionIssues","modelQualityStatus","validatedModelDownload","frozenModelDownload","deploymentModelDownload","templateDownload","excludeAllConstants","clusterButton","stateExplorationButton","addPerformanceCondition","performanceButton","qualityButton","trendButton","preprocessingPreviewButton","trainButton","validateButton","importConfigButton","exportConfigButton"];
  const missing=ids.filter(id=>!el(id)); if(missing.length) throw new Error(`页面初始化不完整，缺少元素：${missing.join(", ")}`);
}

function fillSelect(node, values, blankLabel=null) {
  node.replaceChildren();
  if (blankLabel !== null) { const option=document.createElement("option"); option.value=""; option.textContent=blankLabel; node.append(option); }
  values.forEach(value=>{ const option=document.createElement("option"); option.value=value; option.textContent=value; node.append(option); });
}
function renderUploadedColumns(columns) {
  const list=el("tagOptions"); list.replaceChildren(); const visible=columns.slice(0,200);
  visible.forEach(column=>{ const row=document.createElement("div"); row.className="tag-row"; row.classList.add("pending"); const name=document.createElement("span"); name.className="tag-name"; name.title=column; name.textContent=column; const badge=document.createElement("span"); badge.className="tag-state"; badge.textContent="待检查"; row.append(name,badge); list.append(row); });
  if(columns.length>visible.length) { const more=document.createElement("span"); more.className="help"; more.textContent=`其余 ${columns.length-visible.length} 个列将在检查数据后显示。`; list.append(more); }
}
function renderBasicInspection(data) {
  const summary=el("basicInspectionSummary"), issues=el("basicInspectionIssues");
  if(!data) { summary.innerHTML=""; issues.className="empty"; issues.textContent="上传并检查数据后显示基础检查结果。"; return; }
  summary.innerHTML=metric("历史数据行数",data.rows)+metric("原始列",data.column_profiles.length)+metric("数值候选列",data.numeric_columns.length)+metric("时间范围",`${displayTime(data.time_start)}\n${displayTime(data.time_end)}`,"time-range")+metric("基础检查",data.can_train_without_review?"通过":"有问题");
  issues.className=""; issues.replaceChildren();
  if(!data.quality_issues.length) { const notice=document.createElement("div"); notice.className="empty"; notice.textContent="整体历史数据未发现基础时间轴或数值候选列问题。"; issues.append(notice); }
  else data.quality_issues.forEach(issue=>{ const card=document.createElement("div"); card.className=`issue-card ${issue.severity==="error"?"blocking":""}`; card.innerHTML=`<strong>${escapeHtml(issue.code)}</strong><span>${escapeHtml(issue.message)}</span>`; issues.append(card); });
  const profiles=data.column_profiles||[]; if(!profiles.length) return;
  const hasRanges=profiles.some(profile=>["engineering_range_outside_count","normal_range_outside_count","alarm_range_outside_count"].some(key=>profile[key]!==null&&profile[key]!==undefined));
  const table=document.createElement("table"); const head=document.createElement("thead"), headRow=document.createElement("tr");
  const headers=["Tag","有效数值","缺失 / 空白","非数字","+Inf / -Inf","最小 / 最大",...(hasRanges?["工程越界","正常越界","报警越界"]:[]),"状态 / 建议"];
  headers.forEach(value=>{ const cell=document.createElement("th"); cell.textContent=value; headRow.append(cell); }); head.append(headRow); table.append(head);
  const body=document.createElement("tbody"); profiles.forEach(profile=>{ const row=document.createElement("tr"); const suggestion=profile.suggestion; const rangeOutside=["engineering_range_outside_count","normal_range_outside_count","alarm_range_outside_count"].some(key=>Number(profile[key])>0); const values=[profile.tag,profile.valid_count,`${profile.missing_count} / ${profile.empty_string_count}`,profile.non_numeric_count,`${profile.positive_infinite_count} / ${profile.negative_infinite_count}`,`${formatStat("minimum",profile.minimum)} / ${formatStat("maximum",profile.maximum)}`,...(hasRanges?[profile.engineering_range_outside_count,profile.normal_range_outside_count,profile.alarm_range_outside_count]:[])]; values.forEach(value=>{ const cell=document.createElement("td"); cell.textContent=value===null||value===undefined?"—":value; row.append(cell); }); const suggestionCell=document.createElement("td"); suggestionCell.textContent=suggestion?suggestion.message:(profile.invalid_count?"提示":(rangeOutside?"范围提示":"正常")); if(suggestion&&state.inspection.numeric_columns.includes(profile.tag)) { const confirm=document.createElement("button"); confirm.type="button"; confirm.className="secondary"; confirm.textContent="确认排除"; confirm.disabled=state.registry[profile.tag]?.role==="exclude"; confirm.addEventListener("click",()=>confirmSuggestedExclusion(profile)); suggestionCell.append(document.createElement("br"),confirm); } row.append(suggestionCell); body.append(row); }); table.append(body); issues.append(table);
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
function stateExplorationPayload() {
  const tags=selectedTags(); if(tags.length<2) throw new Error("至少选择两个连续 Tag。");
  const payload={...commonPayload(),tags,exploration_start:el("explorationStart").value,exploration_end:el("explorationEnd").value,exploration_config:{cluster_count:numberValue("explorationClusterCount"),random_state:numberValue("explorationRandomState"),candidate_count_per_cluster:numberValue("explorationCandidateCount"),minimum_candidate_duration_minutes:numberValue("explorationMinimumDuration"),maximum_plot_points:numberValue("explorationMaximumPlotPoints")}};
  const performanceTag=el("explorationPerformanceTag").value.trim();
  if(performanceTag) payload.performance_config={performance_tag:performanceTag,direction:el("explorationPerformanceDirection").value,target_min:optionalNumber("explorationTargetMin"),target_max:optionalNumber("explorationTargetMax"),minimum_duration_minutes:numberValue("explorationPerformanceMinimumDuration"),candidate_count:numberValue("explorationPerformanceCandidateCount")};
  return payload;
}
function explorationClusterNumber(clusterId) { const match=String(clusterId).match(/(\d+)$/); return match?Number(match[1]):1; }
function explorationNumber(value,digits=2) { return value===null||value===undefined||!Number.isFinite(Number(value))?"—":Number(value).toFixed(digits); }
const EXPLORATION_CLUSTER_PALETTE=["#176b87","#cf3f36","#16845b","#d19a20","#7c3aed","#db2777","#0891b2","#65a30d","#ea580c","#475569"];
function explorationClusterColor(clusterId) { return EXPLORATION_CLUSTER_PALETTE[(explorationClusterNumber(clusterId)-1)%EXPLORATION_CLUSTER_PALETTE.length]; }
function renderStateExploration(data) {
  el("explorationEmpty").hidden=true; el("explorationContent").hidden=false;
  const summary=data.preprocessing_summary||{}; const coverage=Number(summary.effective_coverage_ratio||0);
  el("explorationOverview").innerHTML=metric("原始行数",summary.source_row_count)+metric("重采样行数",summary.resampled_row_count)+metric("最终动态样本数",summary.final_dynamic_row_count)+metric("有效覆盖率",`${(coverage*100).toFixed(1)}%`)+metric("Cluster 数量",(data.cluster_summaries||[]).length)+metric("显示点数",`${data.returned_point_count}/${data.full_point_count}`);
  const warnings=el("explorationWarnings"); warnings.replaceChildren(); (data.warnings||[]).forEach(item=>{ const row=document.createElement("div"); row.textContent=`${item.code}：${item.message}${item.cluster_id?`（${item.cluster_id}）`:``}`; warnings.append(row); }); if(!warnings.children.length) warnings.innerHTML='<span class="help">暂无结构化告警。</span>';
  renderExplorationLossSummary(summary.loss_counts||{}); renderExplorationPcChart(data); renderExplorationTimeline(data.cluster_series||[],data.cluster_candidates||[]); renderExplorationClusterTable(data.cluster_summaries||[]); renderExplorationCandidateTables(data.cluster_candidates||[],data.performance_candidates||[],data.candidate_decisions||[]);
}
function renderExplorationLossSummary(losses) {
  const fields=[["empty_bin_count","空桶"],["input_invalid_loss","输入无效"],["filter_warmup_loss","滤波预热"],["filter_context_invalid_loss","滤波上下文无效"],["lag_warmup_loss","Lag预热"],["lag_context_invalid_loss","Lag上下文无效"],["state_filter_loss","状态过滤损失"]];
  el("explorationLossSummary").innerHTML=`<table><thead><tr>${fields.map(([,label])=>`<th>${label}</th>`).join("")}</tr></thead><tbody><tr>${fields.map(([key])=>`<td class="numeric">${losses[key]??0}</td>`).join("")}</tr></tbody></table>`;
}
function renderExplorationPcChart(data) {
  const rows=data.cluster_series||[]; const container=el("explorationPcChart"); if(!rows.length){container.innerHTML='<div class="empty">无可展示序列。</div>';return;}
  const width=760,height=260,pad=34; const xs=rows.map(row=>Number(row.pc1)),ys=rows.map(row=>Number(row.pc2)); const maxX=Math.max(...xs.map(Math.abs),1e-9),maxY=Math.max(...ys.map(Math.abs),1e-9); const x=value=>width/2+value/maxX*(width/2-pad),y=value=>height/2-value/maxY*(height/2-pad);
  const points=rows.map(row=>`<circle cx="${x(Number(row.pc1)).toFixed(2)}" cy="${y(Number(row.pc2)).toFixed(2)}" r="3" fill="${explorationClusterColor(row.cluster_id)}" fill-opacity=".75"><title>${escapeHtml(displayTime(row.timestamp,19))} · ${escapeHtml(row.cluster_id)}</title></circle>`).join("");
  const centers=Object.entries(data.cluster_centers||{}).map(([cluster,center])=>{const cx=x(Number(center[0])),cy=y(Number(center[1])),color=explorationClusterColor(cluster);return `<g stroke="${color}" stroke-width="2"><line x1="${cx-7}" x2="${cx+7}" y1="${cy}" y2="${cy}"/><line x1="${cx}" x2="${cx}" y1="${cy-7}" y2="${cy+7}"/><title>${escapeHtml(cluster)} 中心</title></g>`;}).join("");
  const legend=[...new Set(rows.map(row=>row.cluster_id))].map(cluster=>{const number=explorationClusterNumber(cluster);return `<text x="${pad+(number-1)*88}" y="16" fill="${explorationClusterColor(cluster)}" font-size="10">● ${escapeHtml(cluster)}</text>`;}).join("");
  container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Cluster PC1 PC2 散点图">${legend}<line x1="${pad}" x2="${width-pad}" y1="${height/2}" y2="${height/2}" stroke="#d7dee8"/><line x1="${width/2}" x2="${width/2}" y1="${pad}" y2="${height-pad}" stroke="#d7dee8"/>${points}${centers}<text x="${width-pad}" y="${height/2-5}" text-anchor="end" fill="#5f6c7b" font-size="10">PC1</text><text x="${width/2+5}" y="${pad+10}" fill="#5f6c7b" font-size="10">PC2</text></svg>`;
}
function explorationTimelineTick(value) { const time=new Date(value); return `${String(time.getMonth()+1).padStart(2,"0")}/${String(time.getDate()).padStart(2,"0")} ${String(time.getHours()).padStart(2,"0")}:${String(time.getMinutes()).padStart(2,"0")}`; }
function renderExplorationTimeline(rows,candidates) {
  const container=el("explorationTimeline"); const ordered=rows.map(row=>({...row,time:new Date(row.timestamp)})).filter(row=>Number.isFinite(row.time.getTime())).sort((left,right)=>left.time-right.time);
  if(!ordered.length){container.innerHTML='<div class="empty">暂无显示序列。</div>';return;}
  const first=ordered[0].time.getTime(),last=ordered[ordered.length-1].time.getTime();
  if(first===last){container.innerHTML=`<div class="empty">仅有一个显示点，无法推断状态持续时间。</div><p class="timeline-note">时间轴基于状态探索显示序列；聚类计算仍使用全部有效样本。</p>${explorationTimelineDetails(ordered)}`;return;}
  const width=760,height=188,left=94,right=18,statusTop=34,statusHeight=36,candidateTop=92,candidateHeight=16,axisY=136;
  const x=value=>left+(new Date(value).getTime()-first)/(last-first)*(width-left-right);
  const blocks=[]; const breaks=[];
  ordered.slice(0,-1).forEach((row,index)=>{const next=ordered[index+1]; const segmentBreak=next.break_before||next.segment_id!==row.segment_id; if(segmentBreak){breaks.push(`<line x1="${x(next.timestamp).toFixed(2)}" x2="${x(next.timestamp).toFixed(2)}" y1="${statusTop-5}" y2="${statusTop+statusHeight+5}" stroke="#64748b" stroke-dasharray="3 2"><title>物理连续段断点</title></line>`);return;} const start=x(row.timestamp),end=x(next.timestamp); if(end>start) blocks.push(`<rect x="${start.toFixed(2)}" y="${statusTop}" width="${(end-start).toFixed(2)}" height="${statusHeight}" fill="${explorationClusterColor(row.cluster_id)}"><title>${escapeHtml(row.cluster_id)}&#10;开始时间：${escapeHtml(displayTime(row.timestamp,19))}&#10;结束时间：${escapeHtml(displayTime(next.timestamp,19))}</title></rect>`);});
  const windows=(candidates||[]).map(candidate=>{const start=Math.max(first,new Date(candidate.start).getTime()),end=Math.min(last,new Date(candidate.end).getTime()); if(!Number.isFinite(start)||!Number.isFinite(end)||end<start) return ""; const windowX=left+(start-first)/(last-first)*(width-left-right),windowWidth=Math.max(1,(end-start)/(last-first)*(width-left-right)); return `<rect x="${windowX.toFixed(2)}" y="${candidateTop}" width="${windowWidth.toFixed(2)}" height="${candidateHeight}" fill="${explorationClusterColor(candidate.cluster_id)}" fill-opacity=".35" stroke="${explorationClusterColor(candidate.cluster_id)}" stroke-width="1.5"><title>${escapeHtml(candidate.candidate_id)}&#10;${escapeHtml(candidate.cluster_id)}&#10;开始时间：${escapeHtml(displayTime(candidate.start,19))}&#10;结束时间：${escapeHtml(displayTime(candidate.end,19))}</title></rect>`;}).join("");
  const ticks=Array.from({length:4},(_,index)=>first+(last-first)*index/3).map((value,index)=>`<g><line x1="${(left+(value-first)/(last-first)*(width-left-right)).toFixed(2)}" x2="${(left+(value-first)/(last-first)*(width-left-right)).toFixed(2)}" y1="${axisY}" y2="${axisY+4}" stroke="#94a3b8"/><text x="${(left+(value-first)/(last-first)*(width-left-right)).toFixed(2)}" y="${axisY+17}" text-anchor="${index===0?"start":index===3?"end":"middle"}" fill="#5f6c7b" font-size="10">${escapeHtml(explorationTimelineTick(value))}</text></g>`).join("");
  container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Cluster 状态时间轴"><text x="8" y="${statusTop+14}" fill="#334155" font-size="11">Cluster 状态</text><text x="8" y="${candidateTop+12}" fill="#334155" font-size="11">候选窗口</text><line x1="${left}" x2="${width-right}" y1="${axisY}" y2="${axisY}" stroke="#94a3b8"/>${blocks.join("")}${breaks.join("")}${windows}${ticks}</svg><p class="timeline-note">时间轴基于状态探索显示序列；聚类计算仍使用全部有效样本。显示点之间的时间跨度可能来自抽样；空白/断点表示物理连续段中断，不代表 Cluster。</p>${explorationTimelineDetails(ordered)}`;
}
function explorationTimelineDetails(rows) {
  const body=rows.map(row=>`<tr><td>${escapeHtml(displayTime(row.timestamp,19))}${row.break_before?" · 断点":""}</td><td>${escapeHtml(row.cluster_id)}</td><td>${escapeHtml(String(row.segment_id))}</td></tr>`).join("");
  return `<details><summary>查看显示抽样点明细</summary><div class="timeline-detail"><table><thead><tr><th>时间</th><th>Cluster</th><th>数据段</th></tr></thead><tbody>${body}</tbody></table></div></details>`;
}
function renderExplorationClusterTable(summaries) {
  const body=el("explorationClusterTable"); body.replaceChildren(); summaries.forEach(item=>{const row=document.createElement("tr"); [item.cluster_id,item.sample_count,`${(Number(item.coverage_ratio)*100).toFixed(1)}%`,item.segment_count,`${item.total_duration_minutes} 分钟`,explorationNumber(item.median_distance_to_centroid,3),explorationNumber(item.pc_score_dispersion,3),item.candidate_count].forEach(value=>{const cell=document.createElement("td");cell.textContent=value;row.append(cell);});body.append(row);});
}
function renderExplorationCandidateTables(clusterCandidates,performanceCandidates,decisions) {
  const decisionById=Object.fromEntries(decisions.map(item=>[item.candidate_id,item]));
  const controls=item=>{const decision=decisionById[item.candidate_id]||{decision:"pending",comment:""};return `<td><input class="exploration-candidate-select" type="checkbox" data-candidate-id="${escapeHtml(item.candidate_id)}" aria-label="选择候选"></td><td><select class="exploration-candidate-decision" data-candidate-id="${escapeHtml(item.candidate_id)}"><option value="pending" ${decision.decision==="pending"?"selected":""}>待决策</option><option value="accepted" ${decision.decision==="accepted"?"selected":""}>已接受</option><option value="rejected" ${decision.decision==="rejected"?"selected":""}>已拒绝</option></select></td><td><input class="exploration-candidate-comment" data-candidate-id="${escapeHtml(item.candidate_id)}" value="${escapeHtml(decision.comment||"")}" aria-label="候选备注"></td>`;};
  const clusterHead="<table><thead><tr><th>选择</th><th>决策</th><th>备注</th><th>Cluster</th><th>开始</th><th>结束</th><th>覆盖时长</th><th>样本数</th><th>中心距离</th><th>稳定性</th><th>排名</th></tr></thead><tbody>";
  const clusterBody=clusterCandidates.map(item=>`<tr>${controls(item)}<td>${escapeHtml(item.cluster_id)}</td><td>${escapeHtml(displayTime(item.start,19))}</td><td>${escapeHtml(displayTime(item.end,19))}</td><td>${item.duration_minutes} 分钟</td><td>${item.sample_count}</td><td>${explorationNumber(item.centroid_distance,4)}</td><td>${explorationNumber(item.stability_score,4)}</td><td>${item.rank}</td></tr>`).join("");
  el("explorationClusterCandidates").innerHTML=clusterHead+(clusterBody||'<tr><td colspan="11">暂无满足条件的 Cluster 候选。</td></tr>')+"</tbody></table>";
  const performanceHead="<table><thead><tr><th>选择</th><th>决策</th><th>备注</th><th>开始</th><th>结束</th><th>覆盖时长</th><th>性能摘要</th><th>关联Cluster</th><th>稳定性</th><th>排名</th></tr></thead><tbody>";
  const performanceBody=performanceCandidates.map(item=>{const summary=item.performance_summary||{};const text=`均值 ${explorationNumber(summary.mean,3)}；中位数 ${explorationNumber(summary.median,3)}；最小 ${explorationNumber(summary.minimum,3)}；最大 ${explorationNumber(summary.maximum,3)}`;return `<tr>${controls(item)}<td>${escapeHtml(displayTime(item.start,19))}</td><td>${escapeHtml(displayTime(item.end,19))}</td><td>${item.duration_minutes} 分钟</td><td>${escapeHtml(text)}</td><td>${escapeHtml((item.associated_cluster_ids||[]).join(", "))}</td><td>${explorationNumber(item.stability_score,4)}</td><td>${item.rank}</td></tr>`;}).join("");
  el("explorationPerformanceCandidates").innerHTML=performanceHead+(performanceBody||'<tr><td colspan="10">暂无满足条件的性能候选。</td></tr>')+"</tbody></table>";
}

el("uploadButton").addEventListener("click", async () => {
  const file=el("fileInput").files[0]; if (!file) { setStatus("请选择 CSV、XLSX 或 TXT 文件。","warning"); return; }
  const button=el("uploadButton"); setBusy(button,true,"上传中…");
  try {
    setStatus("正在读取文件…","info"); await new Promise(resolve=>requestAnimationFrame(resolve));
    const form=new FormData(); form.append("file",file);
    const data=await api("/api/upload",{method:"POST",body:form});
    state.fileId=data.file_id; state.inspection=null; state.registry={}; state.quality=null; state.training=null; state.runId=null; state.exploratoryRunId=null; state.clustering=null; state.exploration=null; state.performance=null; state.trend=null; state.preprocessingPreview=null; state.preprocessingPreviewTag=null; state.excludedTags=[]; state.excludedWindows=[]; state.candidateWindows=[]; state.trainingWindows=[]; state.trainingWindowSummary=[]; state.selectedTag=null; state.selectedModelTags.clear(); el("preprocessingPreview").className="muted"; el("preprocessingPreview").textContent="尚未预览"; renderCandidateWindows(); renderExcludedWindows(); renderTrainingWindows(); invalidateQuality(); renderBasicInspection(null); renderUploadedColumns(data.columns); fillSelect(el("timestampColumn"),data.columns); fillSelect(el("labelColumn"),data.columns,"不使用"); fillSelect(el("explorationPerformanceTag"),[],"不配置"); if(data.encoding) el("encoding").value=data.encoding;
    el("inspectButton").disabled=false; el("clusterButton").disabled=true; el("stateExplorationButton").disabled=true; el("addPerformanceCondition").disabled=true; el("performanceButton").disabled=true; el("qualityButton").disabled=true; el("trendButton").disabled=true; el("preprocessingPreviewButton").disabled=true; el("trainButton").disabled=true; el("validateButton").disabled=true; el("importConfigButton").disabled=true; el("exportConfigButton").disabled=true;
    setStatus(`文件信息：${data.filename}（${Math.ceil(data.size_bytes/1024)} KB），已读取 ${data.columns.length} 个列名。请选择时间列，下一步：正在检查数据。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("inspectButton").addEventListener("click", async () => {
  const button=el("inspectButton"); setBusy(button,true,"检查中…");
  const controller=new AbortController(); let timedOut=false;
  const previousRegistry=state.registry, previousSelectedTags=new Set(state.selectedModelTags), previousExcludedTags=state.excludedTags, hadInspection=state.inspection!==null;
  const timeoutId=window.setTimeout(()=>{ timedOut=true; controller.abort(); },30000);
  const progressId=window.setTimeout(()=>setStatus("正在检查数据：读取时间列与候选 Tag，大文件可能需要数十秒。","info"),1500);
  try {
    setStatus("正在检查数据…","info"); await new Promise(resolve=>requestAnimationFrame(resolve));
    const data=await api("/api/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tag_configs:previousRegistry}),signal:controller.signal});
    ensureInspectionPageReady();
    state.inspection=data; state.registry=Object.fromEntries(data.numeric_columns.map(tag=>[tag,{...emptyTagConfig(),...(previousRegistry[tag]||{})}])); state.quality=null; state.selectedTag=null; state.excludedTags=previousExcludedTags; reconcileExcludedTags(); state.exploration=null; state.validation=null; el("validatedModelDownload").hidden=true; el("frozenModelDownload").hidden=true; el("deploymentModelDownload").hidden=true; if(hadInspection) state.selectedModelTags=new Set(data.numeric_columns.filter(tag=>previousSelectedTags.has(tag)&&state.registry[tag].role==="continuous_input")); else state.selectedModelTags=new Set(data.numeric_columns.filter(tag=>state.registry[tag].role==="continuous_input")); invalidateQuality(); renderBasicInspection(data); renderPerformanceConditions(data.numeric_columns); fillSelect(el("explorationPerformanceTag"),data.numeric_columns,"不配置"); renderTagList();
    fillSelect(el("trendTags"),data.numeric_columns); [...el("trendTags").options].slice(0,Math.min(3,data.numeric_columns.length)).forEach(option=>option.selected=true);
    el("analysisStart").value=localTime(data.time_start); el("analysisEnd").value=localTime(data.time_end); el("explorationStart").value=localTime(data.time_start); el("explorationEnd").value=localTime(data.time_end); el("candidateStart").value=localTime(data.time_start); el("candidateEnd").value=localTime(data.suggested_normal_end); el("candidateComment").value=""; state.excludedWindows=[]; state.candidateWindows=[{id:"suggested-window-001",start:el("candidateStart").value,end:el("candidateEnd").value,source:"suggested",source_ref:"inspect-default",status:"pending",comment:"系统建议的初始正常候选时段"}]; state.trainingWindows=[]; state.trainingWindowSummary=[]; renderCandidateWindows(); renderExcludedWindows(); renderTrainingWindows(); el("validationStart").value=localTime(data.suggested_validation_start); el("validationEnd").value=localTime(data.time_end); state.validationWindows=[]; renderValidationWindows();
    el("trendStart").value=localTime(data.trend_default_start); el("trendEnd").value=localTime(data.trend_default_end);
    if (data.sample_interval_minutes) el("sampleInterval").value=String(data.sample_interval_minutes);
    el("clusterButton").disabled=false; el("stateExplorationButton").disabled=false; el("addPerformanceCondition").disabled=false; el("performanceButton").disabled=false; el("qualityButton").disabled=true; el("trendButton").disabled=false; el("preprocessingPreviewButton").disabled=false; el("importConfigButton").disabled=false; el("exportConfigButton").disabled=false;
    el("templateDownload").href=`/download/tag-config-template?file_id=${encodeURIComponent(state.fileId)}&timestamp_column=${encodeURIComponent(el("timestampColumn").value)}&encoding=${encodeURIComponent(el("encoding").value)}`;
    if(data.numeric_columns.length) selectTag(data.numeric_columns[0]);
    const issues=data.quality_issues.map(item=>`${item.code}(${item.count}) ${item.tag||""}`).join("、");
    setStatus(issues ? `基础数据检查完成：${data.rows} 行。发现 ${issues}；确认训练窗口后必须执行建模质量检查。` : data.modeling_tag_hint ? `基础数据检查完成：${data.rows} 行。${data.modeling_tag_hint.message}` : `基础数据检查完成：${data.rows} 行，识别 ${data.numeric_columns.length} 个数值列。请确认训练窗口并执行建模质量检查。`, issues||data.modeling_tag_hint?"warning":"success");
  } catch (error) { console.error("数据检查失败:",error); setStatus(`数据检查失败:\n${timedOut?"数据检查超过 30 秒未完成。请确认文件格式或重新上传后重试。":error.message||String(error)}`,"error"); }
  finally { window.clearTimeout(timeoutId); window.clearTimeout(progressId); setBusy(button,false,""); }
});

el("tagSearch").addEventListener("input",renderTagList);
el("selectAllTags").addEventListener("click",()=>{ state.selectedModelTags=new Set((state.inspection?.numeric_columns||[]).filter(tag=>(state.registry[tag]?.role||"continuous_input")==="continuous_input")); invalidateQuality("建模Tag已修改"); renderTagList(); });
el("clearAllTags").addEventListener("click",()=>{ state.selectedModelTags.clear(); invalidateQuality("建模Tag已修改"); renderTagList(); });
el("showProblemTags").addEventListener("click",()=>{ state.showProblems=!state.showProblems; el("showProblemTags").textContent=state.showProblems?"显示全部Tag":"只看问题Tag"; renderTagList(); });
el("saveTagConfig").addEventListener("click",()=>{ try { saveCurrentTagConfig(); } catch(error) { setStatus(error.message,"error"); } });
document.querySelectorAll(".inner-tab").forEach(button=>button.addEventListener("click",()=>{ document.querySelectorAll(".inner-tab").forEach(node=>node.classList.toggle("active",node===button)); document.querySelectorAll(".inner-panel").forEach(panel=>panel.classList.toggle("active",panel.id===button.dataset.inner)); }));
["sampleInterval","resamplingMethod","filterMethod","firstOrderAlpha","smoothingWindow","gapThreshold","maxLag","lagStep"].forEach(id=>el(id).addEventListener("change",()=>invalidateQuality("预处理参数已修改")));
function syncFilterControls() { const filterMethod=el("filterMethod").value; el("firstOrderAlpha").disabled=filterMethod!=="first_order"; el("firstOrderAlpha").required=filterMethod==="first_order"; el("smoothingWindow").disabled=filterMethod!=="trailing_mean"; }
el("filterMethod").addEventListener("change",syncFilterControls);
syncFilterControls();
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
  for(const tag of [...state.selectedModelTags]) { if((state.registry[tag]?.role||"continuous_input")!=="continuous_input") state.selectedModelTags.delete(tag); } reconcileExcludedTags();
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
  const tags=selectedTags(); if(tags.length<2) { setStatus("至少选择两个“连续输入” Tag。","warning"); return; }
  const button=el("qualityButton"); state.quality=null; state.qualityStatus="checking"; state.qualityError=""; el("trainButton").disabled=true; el("trainExploratoryButton").disabled=true; el("qualitySummary").innerHTML=""; el("qualityIssues").className="empty"; el("qualityIssues").textContent="正在执行建模质量检查。"; el("excludeAllConstants").disabled=true; renderCurrentTagQuality(); renderModelQualityStatus(); setBusy(button,true,"检查中…");
  try {
    const payload={...commonPayload(),tags,training_windows:trainingWindowsPayload()};
    const data=await api("/api/quality",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); const readiness=data.training_readiness||{normal_state:{can_train:data.can_train},exploratory:{can_train:data.can_train}}; state.quality=data; if(!data.tags.some(item=>item.tag===state.selectedTag)) state.selectedTag=data.tags[0]?.tag||null; state.qualityStatus=readiness.normal_state.can_train&&readiness.exploratory.can_train?"passed":"issues"; state.trainingWindowSummary=data.training_window_summary||state.trainingWindowSummary; renderTrainingWindows(); renderQuality(data); renderTagList(); renderModelQualityStatus(); el("trainButton").disabled=!readiness.normal_state.can_train; el("trainExploratoryButton").disabled=!readiness.exploratory.can_train;
    globalThis.showWorkflowStage?.("modelPanel");
    setStatus(readiness.normal_state.can_train&&readiness.exploratory.can_train?"建模质量检查通过，可以训练两类模型。":readiness.exploratory.can_train?"探索模型可训练；正常状态候选受当前工程量程排除影响不可训练。":"建模质量检查发现问题，请排除问题 Tag 或调整训练窗口后重新检查。",readiness.exploratory.can_train?"success":"error");
  } catch(error) { state.qualityStatus="failed"; state.qualityError=error.message||String(error); renderModelQualityStatus(); setStatus(state.qualityError,"error"); el("trainButton").disabled=true; el("trainExploratoryButton").disabled=true; }
  finally { setBusy(button,false,""); }
});

el("qualityTagSelect").addEventListener("change",()=>{
  const value=el("qualityTagSelect").value;
  state.selectedTag = value;
  renderCurrentTagQuality();
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
    const data=await api("/api/trend",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); state.trend=data; renderTrend(data); const stage=data.series_stage?.resampling_applied?"原始数据与基于重采样的因果滤波结果":"原始数据与因果滤波结果"; setStatus(`趋势与统计已更新：${stage}；未插值或补点。`,"success");
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});
el("preprocessingPreviewButton").addEventListener("click",async()=>{
  const tags=[...el("trendTags").selectedOptions].map(option=>option.value); if(!tags.length||tags.length>8){setStatus("预处理预览请选择1至8个Tag。","warning");return;}
  const button=el("preprocessingPreviewButton"); setBusy(button,true,"预览中…");
  try { const data=await api("/api/preprocessing-preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...commonPayload(),tags,start:el("trendStart").value,end:el("trendEnd").value})}); state.preprocessingPreview={data,tags}; if(!tags.includes(state.preprocessingPreviewTag)) state.preprocessingPreviewTag=tags[0]; renderPreprocessingPreview(); setStatus("预处理预览已更新；显示抽样不会进入训练。","success"); }
  catch(error){setStatus(error.message,"error");} finally {setBusy(button,false);}
});
function renderPreprocessingPreview(){
  const preview=state.preprocessingPreview; if(!preview) return;
  const {data,tags}=preview;
  const tag=tags.includes(state.preprocessingPreviewTag)?state.preprocessingPreviewTag:tags[0]; state.preprocessingPreviewTag=tag;
  const s=data.summary; const resampledLabel=s.resampling_method==="none"?"未重采样输入":"重采样后";
  const summary=`源数据 ${s.source_row_count}；${resampledLabel} ${s.resampled_row_count}；正常聚合减少 ${s.resampling_row_reduction??"—"}；部分桶删除 ${s.partial_resampling_bin_loss??"—"}；部分桶原始行删除 ${s.partial_resampling_row_loss??"—"}；物理段 ${s.raw_segment_count}；原始缺口 ${s.raw_gap_count}；空桶 ${s.empty_bin_count}；滤波结构预热 ${s.filter_warmup_loss}；滤波上下文无效 ${s.filter_context_invalid_loss??"—"}；状态过滤损失 ${s.state_filter_input_rows-s.state_filter_output_rows}；Lag结构预热 ${s.lag_warmup_loss}；Lag上下文无效 ${s.lag_context_invalid_loss}；当前输入无效 ${s.input_invalid_loss??"—"}；最终动态样本 ${s.final_dynamic_row_count}；动态特征 ${s.dynamic_feature_count}`;
  const labels={raw:"原始数据",resampled:"重采样数据",filtered:"因果滤波数据"};
  const selector=`<label>查看 Tag：<select id="preprocessingPreviewTagSelect">${tags.map(value=>`<option value="${escapeHtml(value)}"${value===tag?" selected":""}>${escapeHtml(value)}</option>`).join("")}</select></label>`;
  const tables=["raw","resampled","filtered"].map(stage=>{const rows=data[stage]; const head=`<tr><th>时间</th>${tags.map(value=>`<th>${escapeHtml(value)}</th>`).join("")}</tr>`; const body=rows.map(row=>`<tr><td>${escapeHtml(displayTime(row.timestamp,19))}${row.physical_gap_start?"（物理缺口后）":""}</td>${tags.map(value=>`<td>${row[value]===null?"缺失":escapeHtml(String(row[value]))}</td>`).join("")}</tr>`).join(""); return `<h4>${labels[stage]}</h4><div class="table-wrap"><table>${head}<tbody>${body}</tbody></table></div>`;}).join("");
  el("preprocessingPreview").className=""; el("preprocessingPreview").innerHTML=`<p>${summary}</p>${selector}<div class="legend"><span><i class="swatch" style="background:#176b87"></i>原始数据</span><span><i class="swatch" style="background:#d97706"></i>${s.resampling_method==="none"?"重采样未启用（与原始数据相同）":"重采样数据"}</span><span><i class="swatch" style="background:#16845b"></i>${s.filter_method==="none"?"滤波未启用（与重采样数据相同）":"因果滤波数据"}</span></div><div class="preprocessing-preview-chart">${preprocessingPreviewSvg(data,tag)}</div><details class="preprocessing-preview-details"><summary>查看抽样数据明细</summary>${tables}</details>`;
  el("preprocessingPreviewTagSelect").addEventListener("change",event=>{ state.preprocessingPreviewTag=event.target.value; renderPreprocessingPreview(); });
}
function preprocessingPreviewSvg(data,tag) {
  const stages=[["raw","#176b87","原始数据"],["resampled","#d97706","重采样数据"],["filtered","#16845b","因果滤波数据"]];
  const rows=stages.flatMap(([stage])=>data[stage]); const datedRows=rows.map(row=>({...row,time:Date.parse(row.timestamp)})).filter(row=>Number.isFinite(row.time));
  const numericValue=value=>{ if(value===null||value===undefined||(typeof value==="string"&&!value.trim())) return null; const converted=Number(value); return Number.isFinite(converted)?converted:null; };
  const values=datedRows.flatMap(row=>{ const value=numericValue(row[tag]); return value===null?[]:[value]; }); if(!datedRows.length||!values.length) return '<div class="empty">当前 Tag 没有可绘制的有效抽样数据。</div>';
  const width=760,height=260,pad={l:52,r:18,t:18,b:38},start=Math.min(...datedRows.map(row=>row.time)),end=Math.max(...datedRows.map(row=>row.time)),startRow=datedRows.find(row=>row.time===start),endRow=datedRows.find(row=>row.time===end),minimum=Math.min(...values),maximum=Math.max(...values),xSpan=Math.max(1,end-start),dataSpan=maximum-minimum,zeroSpanThreshold=Number.EPSILON*Math.max(1,Math.abs(minimum),Math.abs(maximum)),hasUsableSpan=dataSpan>zeroSpanThreshold,ySpan=hasUsableSpan?dataSpan:Math.max(Math.abs(maximum)*0.02,1e-6),yMinimum=hasUsableSpan?minimum:minimum-ySpan/2,x=time=>pad.l+(time-start)/xSpan*(width-pad.l-pad.r),y=value=>height-pad.b-(value-yMinimum)/ySpan*(height-pad.t-pad.b);
  const paths=stages.map(([stage,color,label])=>{ const segments=[],current=[]; data[stage].forEach(row=>{ const time=Date.parse(row.timestamp),value=numericValue(row[tag]),valid=Number.isFinite(time)&&value!==null&&Number.isFinite(value); if(row.physical_gap_start||!valid) { if(current.length) segments.push(current.splice(0)); if(!valid) return; } current.push(`${x(time).toFixed(1)},${y(value).toFixed(1)}`); }); if(current.length) segments.push(current); return segments.map(points=>`<polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="1.8"><title>${label}</title></polyline>`).join(""); }).join("");
  const gaps=[...new Set(datedRows.filter(row=>row.physical_gap_start).map(row=>row.time))].map(time=>`<line x1="${x(time).toFixed(1)}" x2="${x(time).toFixed(1)}" y1="${pad.t}" y2="${height-pad.b}" stroke="#cf3f36" stroke-dasharray="3 3"><title>物理时间缺口</title></line>`).join("");
  const yLabels=hasUsableSpan?`<text x="4" y="${pad.t+4}" font-size="10">${maximum.toPrecision(5)}</text><text x="4" y="${height-pad.b}" font-size="10">${minimum.toPrecision(5)}</text>`:`<text x="4" y="${y(minimum).toFixed(1)}" font-size="10">${minimum.toPrecision(5)}</text>`;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(tag)}预处理趋势对比"><line x1="${pad.l}" x2="${width-pad.r}" y1="${height-pad.b}" y2="${height-pad.b}" stroke="#94a3b8"/><line x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${height-pad.b}" stroke="#94a3b8"/>${gaps}${paths}${yLabels}<text x="${pad.l}" y="${height-10}" font-size="10">${escapeHtml(displayTime(startRow.timestamp,19))}</text><text x="${width-pad.r}" y="${height-10}" text-anchor="end" font-size="10">${escapeHtml(displayTime(endRow.timestamp,19))}</text></svg>`;
}
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

el("stateExplorationButton").addEventListener("click", async () => {
  const button=el("stateExplorationButton"); setBusy(button,true,"探索中…"); setStatus("正在使用统一预处理构建完整状态空间并执行探索聚类。","info");
  try {
    const data=await api("/api/state-exploration/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(stateExplorationPayload())});
    state.exploration=data; renderStateExploration(data); document.querySelector('[data-panel="stateExplorationPanel"]').click(); setStatus(`状态探索完成：${data.full_point_count} 个完整样本，返回 ${data.returned_point_count} 个显示点。候选仅供工程师比较。`,"success");
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

function selectedExplorationCandidateRows() {
  return [...document.querySelectorAll(".exploration-candidate-select:checked")];
}

el("saveExplorationCandidateDecisions").addEventListener("click", async () => {
  const runId=state.exploration?.exploration_run_id, rows=selectedExplorationCandidateRows();
  if(!runId||!rows.length) { setStatus("请先勾选至少一个状态探索候选。","warning"); return; }
  const button=el("saveExplorationCandidateDecisions"); setBusy(button,true,"保存中…");
  try {
    const decisions=rows.map(input=>{const row=input.closest("tr");return {candidate_id:input.dataset.candidateId,decision:row.querySelector(".exploration-candidate-decision").value,comment:row.querySelector(".exploration-candidate-comment").value};});
    const data=await api(`/api/state-exploration/${encodeURIComponent(runId)}/decisions`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decisions})});
    state.exploration={...state.exploration,candidate_decisions:data.candidate_decisions}; renderStateExploration(state.exploration); setStatus("候选人工决策已保存；接受不会自动参与训练。","success");
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("convertExplorationCandidates").addEventListener("click", async () => {
  const runId=state.exploration?.exploration_run_id, rows=selectedExplorationCandidateRows();
  if(!runId||!rows.length) { setStatus("请勾选已接受的状态探索候选。","warning"); return; }
  const decisions=new Map((state.exploration.candidate_decisions||[]).map(item=>[item.candidate_id,item.decision]));
  const candidateIds=rows.map(input=>input.dataset.candidateId);
  if(candidateIds.some(candidateId=>decisions.get(candidateId)!=="accepted")) { setStatus("只有已接受候选可以加入正常状态候选池。","warning"); return; }
  const candidates=new Map([...(state.exploration.cluster_candidates||[]),...(state.exploration.performance_candidates||[])].map(item=>[item.candidate_id,item]));
  let added=0;
  candidateIds.forEach(candidateRef=>{ const candidate=candidates.get(candidateRef); if(candidate&&!state.candidateWindows.some(window=>window.source_ref===candidateRef)) { state.candidateWindows.push({id:candidateId(),start:candidate.start,end:candidate.end,source:candidate.source,source_ref:candidateRef,status:"accepted",comment:(state.exploration.candidate_decisions||[]).find(item=>item.candidate_id===candidateRef)?.comment||""}); added+=1; } });
  renderCandidateWindows(); globalThis.showWorkflowStage?.("candidatePanel");
  setStatus(added?"已加入候选窗口；请在候选窗口列表确认作为训练窗口。":"所选候选已在候选窗口列表中。",added?"success":"warning");
});

el("clusterButton").addEventListener("click", async () => {
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el("clusterButton"); setBusy(button,true,"聚类中…"); setStatus("正在构建动态状态空间并执行聚类。","info");
  try {
    const payload=state.exploratoryRunId?{file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,exploratory_run_id:state.exploratoryRunId,analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,n_clusters:numberValue("clusterCount")}:{...commonPayload(),tags,tag_configs:tagConfigPayload(tags),analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,variance_threshold:numberValue("varianceThreshold"),n_clusters:numberValue("clusterCount")};
    const data=await api("/api/cluster",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.clustering=data; renderClustering(data); document.querySelector('[data-panel="statePanels"]').click();
    setStatus("聚类完成。请由工程师判断 Cluster，并选择代表性连续时段作为正常候选。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

async function trainModel(modelPurpose) {
  const readiness=state.quality?.training_readiness?.[modelPurpose]||{can_train:state.quality?.can_train}; if(!readiness.can_train) { setStatus("训练前必须重新执行并通过对应模型用途的建模质量检查。","error"); return; }
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el(modelPurpose==="exploratory"?"trainExploratoryButton":"trainButton"); setBusy(button,true,"训练中…"); setStatus("正在构建动态矩阵并训练 DPCA，请勿关闭页面。","info");
  try {
    const components=el("components").value.trim();
    const excludedTags=state.excludedTags.filter(record=>state.registry[record.tag]?.role==="exclude"&&record.reason==="constant_in_reference_window"); const payload={...commonPayload(),tags,excluded_tags:excludedTags,model_purpose:modelPurpose,training_windows:trainingWindowsPayload(),variance_threshold:numberValue("varianceThreshold"),n_components:components?Number(components):null,model_name:el("modelName").value};
    const data=await api("/api/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.runId=data.run_id; if(data.model_purpose==="exploratory") state.exploratoryRunId=data.run_id; state.training=data; state.validation=null; el("validationContent").hidden=true; el("validationEmpty").hidden=false; el("validatedModelDownload").hidden=true; el("frozenModelDownload").hidden=true; el("deploymentModelDownload").hidden=true; renderTraining(data); el("validateButton").disabled=data.model_purpose==="exploratory"; document.querySelector('[data-panel="modelPanel"]').click();
    setStatus(`训练完成：${data.training_rows} 个动态样本，${data.dynamic_features} 个动态特征。当前为${data.model_purpose==="exploratory"?"探索草稿":"正常状态候选"}。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
}
el("trainExploratoryButton").addEventListener("click",()=>trainModel("exploratory"));
el("trainButton").addEventListener("click",()=>trainModel("normal_state"));

function renderValidationWindows() {
  const body=el("validationWindowTable"); body.replaceChildren();
  state.validationWindows.forEach(window=>{ const row=document.createElement("tr");
    const type=document.createElement("td"); type.textContent=window.type==="normal_validation"?"正常样本验证":"已知异常验证";
    const start=document.createElement("td"); start.textContent=displayTime(window.start); const end=document.createElement("td"); end.textContent=displayTime(window.end);
    const comment=document.createElement("td"); comment.textContent=window.comment||"—";
    const actions=document.createElement("td"); const remove=document.createElement("button"); remove.className="secondary"; remove.type="button"; remove.textContent="删除"; remove.addEventListener("click",()=>{ state.validationWindows=state.validationWindows.filter(item=>item.id!==window.id); renderValidationWindows(); }); actions.append(remove);
    row.append(type,start,end,comment,actions); body.append(row);
  });
}
el("addValidationWindow").addEventListener("click",()=>{
  const start=el("validationStart").value, end=el("validationEnd").value;
  if(!start||!end) { setStatus("验证时段需要开始和结束时间。","warning"); return; }
  state.validationWindows.push({id:`validation-${candidateId()}`,type:el("validationType").value,start,end,enabled:true,comment:el("validationComment").value.trim()});
  el("validationComment").value=""; renderValidationWindows();
});

el("validateButton").addEventListener("click", async () => {
  const button=el("validateButton"); setBusy(button,true,"回放中…"); setStatus("正在使用训练参数回放独立验证窗口。","info");
  try {
    if(!state.validationWindows.length) { state.validationWindows.push({id:"validation-default-001",type:"normal_validation",start:el("validationStart").value,end:el("validationEnd").value,enabled:true,comment:""}); renderValidationWindows(); }
    const payload={run_id:state.runId,file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,validation_windows:state.validationWindows,label_column:el("labelColumn").value};
    const data=await api("/api/validate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.validation=data; renderValidation(data); const decisionStatus=el("validationDecisionStatus"); decisionStatus.textContent="等待保存工程师结论。"; decisionStatus.className="status info"; setStatus("独立窗口回放完成。请结合已知事件由工程师确认模型是否通过。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("recordValidationDecision").addEventListener("click",async()=>{
  if(!state.validation||!state.runId) { setStatus("请先完成当前模型的独立验证，再保存工程师结论。","error"); return; }
  const button=el("recordValidationDecision"), decisionStatus=el("validationDecisionStatus"); setBusy(button,true,"保存中…"); decisionStatus.textContent="正在保存工程师结论。"; decisionStatus.className="status info";
  try {
    const data=await api("/api/validation-decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:state.runId,decision:el("validationDecision").value,comment:el("validationDecisionComment").value.trim()})});
    if(!data.engineer_decision) throw new Error("服务未返回工程师结论");
    const download=el("validatedModelDownload"); download.href=data.validated_model_download||"#"; download.hidden=!data.validated_model_download;
    state.validation={...state.validation,model_status:data.model_status,engineer_decision:data.engineer_decision}; renderValidation(state.validation);
    const message=data.engineer_decision.decision==="passed"?"工程师结论已保存并生成已验证模型；原候选模型未被原地修改。":"工程师结论已保存，候选模型保持不变。"; decisionStatus.textContent=message; decisionStatus.className="status success"; setStatus(message,"success");
  } catch(error) { decisionStatus.textContent=error.message; decisionStatus.className="status error"; setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("freezeDeployment").addEventListener("click",async()=>{
  const button=el("freezeDeployment"); setBusy(button,true,"冻结中…");
  try {
    const data=await api("/api/freeze-deployment",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:state.runId,model_id:el("frozenModelId").value.trim(),model_version:Number(el("frozenModelVersion").value),frozen_by:el("frozenBy").value.trim(),comment:el("freezeComment").value})});
    el("frozenModelDownload").href=data.frozen_model_download; el("frozenModelDownload").hidden=false; el("deploymentModelDownload").href=data.deployment_model_download; el("deploymentModelDownload").hidden=false;
    state.validation={...state.validation,model_purpose:"normal_state",model_status:data.model_status}; renderValidation(state.validation); setStatus("模型已工程冻结，并已导出部署模型包。","success");
  } catch(error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

function metric(label,value,className="") { return `<div class="metric${className?` ${className}`:""}"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`; }
function qualityProfileTable(title,profile) {
  const fields=[["sample_count","样本数"],["valid_count","有效数"],["missing_count","缺失数"],["missing_rate","缺失率"],["non_numeric_count","非数值数"],["non_finite_count","非有限值数"],["unique_count","唯一值"],["minimum","最小值"],["maximum","最大值"],["mean","均值"],["median","中位数"],["standard_deviation","标准差"],["p01","P1"],["p05","P5"],["p95","P95"],["p99","P99"],["engineering_range_outside_count","工程范围越界"],["normal_range_outside_count","正常范围外"],["alarm_range_outside_count","报警范围外"]];
  return `<h4>${title}</h4><div class="table-wrap"><table><tbody>${fields.map(([key,label])=>`<tr><th>${label}</th><td>${formatStat(key,profile[key])}</td></tr>`).join("")}</tbody></table></div>`;
}
function renderCurrentTagQuality() {
  const container=el("currentTagQuality"), select=el("qualityTagSelect"); if(!container) return;
  const tags=state.quality ? state.quality.tags : [];
  if(select) {
    select.replaceChildren(); select.disabled=!tags.length;
    if(tags.length) {
      if(!tags.some(item=>item.tag===state.selectedTag)) state.selectedTag=tags[0].tag;
      tags.forEach(item=>{ const option=document.createElement("option"); option.value=item.tag; option.textContent=item.tag; select.append(option); });
      select.value=state.selectedTag;
    }
  }
  const item=state.selectedTag?qualityFor(state.selectedTag):null;
  if(!state.quality||!item) { container.className="empty"; container.textContent=state.quality&&!tags.length?"没有可查看的建模 Tag。":state.qualityStatus==="changed"?"配置已变更，请重新执行建模质量检查。":state.qualityStatus==="checking"?"正在执行建模质量检查。":"尚未执行建模质量检查。"; return; }
  const role=state.registry[item.tag]?.role||item.role; const issueHtml=item.issues.length?item.issues.map(issue=>`<li>${escapeHtml(issue.message)}</li>`).join(""):"<li>无质量问题</li>";
  container.className=""; container.innerHTML=`<div class="issue-card ${item.status}"><strong>${escapeHtml(item.tag)} · ${escapeHtml(displayUiValue(role))} · ${escapeHtml(displayUiValue(item.status))}</strong>${qualityProfileTable("全数据统计",item.full)}${qualityProfileTable("参考期统计",item.reference)}<h4>质量问题与建议</h4><ul>${issueHtml}</ul><span>建议操作：${escapeHtml(item.suggested_action)}</span></div>`;
}
function renderQuality(data) {
  const readiness=data.training_readiness||{normal_state:{can_train:data.can_train},exploratory:{can_train:data.can_train}};
  el("qualitySummary").innerHTML=metric("可直接使用",data.summary.usable)+metric("需要确认",data.summary.review)+metric("阻止训练",data.summary.blocking)+metric("正常状态训练",readiness.normal_state.can_train?"通过":"未通过")+metric("探索训练",readiness.exploratory.can_train?"通过":"未通过");
  const container=el("qualityIssues"); container.className=""; container.replaceChildren();
  data.time_issues.forEach(issue=>{ const card=document.createElement("div"); card.className=`issue-card ${issue.severity==="error"?"blocking":""}`; card.innerHTML=`<strong>${escapeHtml(issue.code)}</strong><span>${escapeHtml(issue.message)}</span>`; container.append(card); });
  const problemTags=data.tags.filter(item=>item.status!=="usable"); problemTags.forEach(item=>{
    const card=document.createElement("div"); card.className=`issue-card ${item.status}`; const profile=item.reference;
    card.innerHTML=`<strong>${escapeHtml(item.tag)} · ${escapeHtml(displayUiValue(item.status))}</strong><span>参考期样本 ${profile.sample_count}；有效 ${profile.valid_count}；唯一值 ${profile.unique_count}；标准差 ${profile.standard_deviation??"—"}</span>${item.issues.map(issue=>`<span>${escapeHtml(issue.message)}</span>`).join("")}`;
    const actions=document.createElement("div"); actions.className="actions";
    if(item.issues.some(issue=>issue.code==="constant_tag")) { const exclude=document.createElement("button"); exclude.className="secondary"; exclude.textContent="从本次模型排除"; exclude.addEventListener("click",()=>excludeConstantTag(item)); actions.append(exclude); }
    const trend=document.createElement("button"); trend.className="secondary"; trend.textContent="在趋势页查看"; trend.addEventListener("click",()=>{ const window=state.trainingWindows.find(value=>value.enabled)||state.trainingWindows[0]; if(window) showCandidateTrend(window); [...el("trendTags").options].forEach(option=>option.selected=option.value===item.tag); document.querySelector('[data-panel="trendPanel"]').click(); }); actions.append(trend); card.append(actions); container.append(card);
  });
  el("excludeAllConstants").disabled=!data.tags.some(item=>item.issues.some(issue=>issue.code==="constant_tag"));
  if(!container.children.length) { container.className="empty"; container.textContent="所选Tag和时间轴没有需要处理的问题。"; }
  renderCurrentTagQuality();
}
function excludeConstantTag(item, refresh=true) {
  const issue=item.issues.find(value=>value.code==="constant_tag"); if(!issue) return;
  setTagExclusion(item.tag,{tag:item.tag,reason:"constant_in_reference_window",sample_count:issue.details.valid_count,unique_count:1,constant_value:issue.details.constant_value});
  state.selectedModelTags.delete(item.tag);
  if(refresh) { if(state.selectedTag===item.tag) selectTag(item.tag); invalidateQuality(`${item.tag}已标记为排除`); renderTagList(); }
}
el("excludeAllConstants").addEventListener("click",()=>{
  const constants=state.quality?.tags.filter(item=>item.issues.some(issue=>issue.code==="constant_tag"))||[]; if(!constants.length) return;
  constants.forEach(item=>excludeConstantTag(item,false));
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
  const rows=data.rows,width=760*zoom,height=230,pad={l:48,r:16,t:16,b:28}; const fields=[]; if(data.display_mode!=="smoothed") fields.push([`${tag}__raw`,"#176b87","原始"]); if(data.display_mode!=="raw") fields.push([`${tag}__smoothed`,"#d97706",data.series_stage?.resampling_applied?"重采样后因果滤波":"因果滤波"]);
  const ranges=data.ranges[tag]||{}; const limits=data.axis_limits[tag]; const minimum=limits.minimum,maximum=limits.maximum,span=maximum-minimum; const x=i=>pad.l+i/Math.max(1,rows.length-1)*(width-pad.l-pad.r),y=value=>height-pad.b-(Number(value)-minimum)/span*(height-pad.t-pad.b);
  const polylines=fields.map(([field,color,label])=>{ const gapField=field.endsWith("__raw")?"raw_physical_gap_start":"filtered_physical_gap_start"; let segments=[],current=[]; rows.forEach((row,i)=>{ if(row[field]===null||row[gapField]) { if(current.length) segments.push(current); current=[]; } if(row[field]!==null) current.push(`${x(i).toFixed(1)},${y(row[field]).toFixed(1)}`); }); if(current.length) segments.push(current); const lines=segments.map(points=>`<polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="1.7"/>`).join(""); const dots=rows.map((row,i)=>{ if(row[field]===null) return `<circle cx="${x(i)}" cy="${height-pad.b}" r="3" fill="#cf3f36"><title>${escapeHtml(displayTime(row.timestamp,19))} · 缺失值</title></circle>`; const outside=field.endsWith("__raw")&&ranges.engineering_min!==null&&ranges.engineering_min!==undefined&&(Number(row[field])<Number(ranges.engineering_min)||Number(row[field])>Number(ranges.engineering_max)); return `<circle cx="${x(i)}" cy="${y(row[field])}" r="${outside?3:2}" fill="${outside?"#cf3f36":color}"><title>${escapeHtml(displayTime(row.timestamp,19))} · ${label}: ${row[field]}${outside?" · 工程量程越界":""}</title></circle>`; }).join(""); return lines+dots; }).join("");
  const rangeLine=(key,color,label)=>ranges[key]===null||ranges[key]===undefined?"":`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(ranges[key])}" y2="${y(ranges[key])}" stroke="${color}" stroke-dasharray="5 4"><title>${label}: ${ranges[key]}</title></line>`;
  const gaps=rows.map((row,i)=>row.physical_gap_start?`<line x1="${x(i)}" x2="${x(i)}" y1="${pad.t}" y2="${height-pad.b}" stroke="#cf3f36" stroke-dasharray="3 3"><title>物理时间缺口</title></line>`:"").join("");
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
  const windows=el("performanceTable"); windows.replaceChildren(); data.representative_windows.forEach((window,index)=>{ const tr=document.createElement("tr"); [displayTime(window.start),displayTime(window.end),window.count].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; tr.append(td); }); const action=document.createElement("td"); const button=document.createElement("button"); button.className="secondary"; button.textContent="加入候选窗口"; button.addEventListener("click",()=>addCandidateWindow("performance",window.start,window.end,`performance-${index+1}`,"")); action.append(button); tr.append(action); windows.append(tr); });
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
    item.representative_windows.forEach(window=>{ const button=document.createElement("button"); button.className="secondary"; button.style.margin="2px"; button.textContent=`加入候选窗口：${displayTime(window.start)} ～ ${window.end.slice(11,16)} (${window.count}点)`; button.addEventListener("click",()=>addCandidateWindow("cluster",window.start,window.end,`cluster-${item.cluster}-${window.start}-${window.end}`,"")); windows.append(button); });
    tr.append(windows); body.append(tr);
  });
}

function modelLifecycle(data) {
  const key=`${data.model_purpose}/${data.model_status}`;
  if(key==="exploratory/draft") return {purpose:"探索模型",status:"草稿",notice:"探索草稿模型，仅用于状态探索，不能执行独立验证或作为正常状态模型。"};
  if(key==="normal_state/validated") return {purpose:"正常状态模型",status:"已验证",notice:"已验证模型，已完成独立验证和工程师确认；尚未执行工程冻结。"};
  if(key==="normal_state/frozen") return {purpose:"正常状态模型",status:"已冻结",notice:"已完成工程冻结；不表示已部署或已进入模型治理平台。"};
  return {purpose:"正常状态模型",status:"候选",notice:"正常状态候选模型，尚未完成独立验证和工程师确认。"};
}
function renderTraining(data) {
  el("modelEmpty").hidden=true; el("modelContent").hidden=false;
  const lifecycle=modelLifecycle(data);
  const totals=data.training_window_totals||{}; const windowCounts=`${totals.enabled_window_count??"—"} / ${totals.used_window_count??"—"} / ${totals.dropped_window_count??"—"}`;
  el("modelMetrics").innerHTML=metric("模型用途",lifecycle.purpose)+metric("模型状态",lifecycle.status)+metric("训练动态样本",data.training_rows)+metric("启用 / 使用 / 丢弃窗口",windowCounts)+metric("动态特征",data.dynamic_features)+metric("主元数",data.n_components)+metric("累计解释率",`${(data.cumulative_explained_variance*100).toFixed(1)}%`)+metric("关注 / 异常",`${data.status_counts.attention} / ${data.status_counts.abnormal}`);
  el("modelLifecycleNotice").textContent=lifecycle.notice;
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
  const header=document.createElement("tr"); ["范围","状态","重采样减少","部分桶","滤波预热","滤波上下文","状态过滤","Lag预热","Lag上下文","输入无效","有效动态样本","原因"].forEach(value=>{ const th=document.createElement("th"); th.textContent=value; header.append(th); }); head.append(header);
  windows.forEach(window=>{
    const row=document.createElement("tr");
    [`窗口 ${window.id}: ${displayTime(window.start)} ～ ${displayTime(window.end)}`,displayUiValue(window.status),window.resampling_row_reduction??"—",window.partial_resampling_bin_loss??"—",window.filter_warmup_loss??"—",window.filter_context_invalid_loss??"—",window.state_filter_loss??"—",window.lag_warmup_loss??"—",window.lag_context_invalid_loss??"—",window.input_invalid_loss??"—",window.effective_samples,displayUiValue(window.dropped_reason??"—")].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; row.append(td); }); body.append(row);
    (window.segments||[]).forEach(segment=>{ const segmentRow=document.createElement("tr");
      [`连续段 ${displayTime(segment.start)} ～ ${displayTime(segment.end)}`,displayUiValue(segment.status),segment.resampling_row_reduction??"—",segment.partial_resampling_bin_loss??"—",segment.filter_warmup_loss??"—",segment.filter_context_invalid_loss??"—",segment.state_filter_loss??"—",segment.lag_warmup_loss??"—",segment.lag_context_invalid_loss??"—",segment.input_invalid_loss??"—",segment.effective_samples,displayUiValue(segment.dropped_reason??"—")].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; segmentRow.append(td); }); body.append(segmentRow);
    });
  });
  table.append(head,body); container.append(table);
}

function renderValidation(data) {
  el("validationEmpty").hidden=true; el("validationContent").hidden=false;
  const lifecycle=modelLifecycle(data); const decisionLabels={passed:"通过",insufficient:"结论不足",failed:"不通过"}; const validationStatus=data.model_status==="frozen"?"已生成冻结和部署模型包":data.model_status==="validated"?"已生成已验证模型副本":data.engineer_decision?`工程师结论已保存：${decisionLabels[data.engineer_decision.decision]||data.engineer_decision.decision}`:"验证回放完成，待工程师确认";
  el("validationMetrics").innerHTML=metric("验证样本",data.scored_rows)+metric("正常",data.status_counts.normal)+metric("关注",data.status_counts.attention)+metric("异常",data.status_counts.abnormal)+metric("模型用途",lifecycle.purpose)+metric("模型状态",lifecycle.status)+metric("验证状态",validationStatus);
  renderValidationMetricDetails(data.validation_metrics||{}); renderContributionStability(data.contribution_stability||{});
  lineChart(el("validationT2Chart"),data.scores,"t2",data.t2_limits,"T²"); lineChart(el("validationSpeChart"),data.scores,"spe",data.q_limits,"SPE");
  el("contributionHint").textContent=data.contributions.length ? "按每个连续越过95%控制限的事件保存峰值贡献；事件不会跨物理时间缺口合并。" : "T² 和 SPE 均未达到 95% 控制限，不输出异常贡献。";
  const body=el("contributionTable"); body.innerHTML="";
  data.contributions.forEach(group=>group.tags.forEach(item=>{ const tr=document.createElement("tr"); const lag=item.lag_start_minutes===item.lag_end_minutes?`${item.lag_start_minutes} 分钟前`:`${item.lag_start_minutes}～${item.lag_end_minutes} 分钟前`; const event=`${displayTime(group.event_start)} ～ ${group.event_end.slice(11,16)}；峰值 ${group.peak_timestamp.slice(11,16)}`; tr.innerHTML=`<td>${escapeHtml(event)}</td><td>${escapeHtml(group.statistic.toUpperCase())}</td><td title="点击在趋势页查看">${escapeHtml(item.tag)}</td><td>${escapeHtml(item.description)}</td><td>${escapeHtml(item.unit)}</td><td class="numeric">${item.contribution_pct.toFixed(1)}%</td><td>${escapeHtml(lag)}</td>`; tr.children[2].style.cursor="pointer"; tr.children[2].addEventListener("click",()=>{ [...el("trendTags").options].forEach(option=>option.selected=option.value===item.tag); el("trendStart").value=localTime(group.event_start); el("trendEnd").value=localTime(group.event_end); document.querySelector('[data-panel="trendPanel"]').click(); }); body.append(tr); }));
  el("scoresDownload").href=data.validation_downloads.scores; el("reportDownload").href=data.validation_downloads.report; el("contributionsDownload").href=data.validation_downloads.contributions;
}

function percent(value) { return value===null||value===undefined?"—":`${(Number(value)*100).toFixed(1)}%`; }
function contributionPercent(value) { return value===null||value===undefined?"—":`${Number(value).toFixed(1)}%`; }
function renderValidationMetricDetails(metrics) {
  const normal=metrics.normal_validation||{}, abnormal=metrics.known_abnormal||{};
  const row=(label,values)=>`<tr><th>${label}</th>${values.map(value=>`<td class="numeric">${value}</td>`).join("")}</tr>`;
  el("validationMetricDetails").innerHTML=`<table><thead><tr><th>验证类型</th><th>有效窗口</th><th>评分行 / 检测率</th><th>T² 95% / 99%</th><th>SPE 95% / 99%</th><th>总体 95% / 99%</th><th>连续误报 / 首次95%延迟</th></tr></thead><tbody>${row("正常样本",[normal.valid_window_count??0,normal.scoring_row_count??0,`${percent(normal.t2?.exceedance_rate_95)} / ${percent(normal.t2?.exceedance_rate_99)}`,`${percent(normal.spe?.exceedance_rate_95)} / ${percent(normal.spe?.exceedance_rate_99)}`,`${percent(normal.overall?.exceedance_rate_95)} / ${percent(normal.overall?.exceedance_rate_99)}`,`${normal.continuous_false_alarm_event_count_95??0} / ${normal.longest_continuous_false_alarm_minutes??0} 分钟`])}${row("已知异常",[abnormal.valid_window_count??0,`${abnormal.detected_window_count_95??0} / ${percent(abnormal.detection_rate_95)}；99% ${abnormal.detected_window_count_99??0} / ${percent(abnormal.detection_rate_99)}`,`${abnormal.t2_detected_window_count_95??0} / ${abnormal.t2_detected_window_count_99??0}`,`${abnormal.spe_detected_window_count_95??0} / ${abnormal.spe_detected_window_count_99??0}`,"—",`${abnormal.first_detection_delay_minutes_95_median??"—"} / ${abnormal.first_detection_delay_minutes_95_max??"—"} 分钟`])}</tbody></table>`;
}
function renderContributionStability(stability) {
  const rows=[];
  [["normal_validation","正常样本"],["known_abnormal","已知异常"]].forEach(([type,label])=>["t2","spe"].forEach(statistic=>{const group=stability[type]?.[statistic]||{}; rows.push(`<tr><td>${label}</td><td>${statistic.toUpperCase()}</td><td class="numeric">${group.event_count??0}</td><td class="numeric">${group.top_k??0}</td><td class="numeric">${percent(group.top1_consistency_rate)}</td><td class="numeric">${percent(group.average_top_k_jaccard_similarity)}</td><td class="numeric">${percent(group.average_contribution_cosine_similarity)}</td><td>${(group.tags||[]).map(tag=>`${escapeHtml(tag.tag)}: Top1 ${tag.top1_count}, Top-K ${tag.top_k_count}, Top-K复现率 ${percent(tag.top_k_recurrence_rate)}，平均贡献率 ${contributionPercent(tag.average_contribution_pct)}，中位贡献率 ${contributionPercent(tag.median_contribution_pct)}`).join("<br>")||"—"}</td></tr>`);}));
  el("contributionStability").innerHTML=`<table><thead><tr><th>验证类型</th><th>统计量</th><th>事件数</th><th>Top-K</th><th>Top1一致率</th><th>Top-K Jaccard</th><th>贡献向量余弦</th><th>Tag复现统计</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function lineChart(container, rows, field, limits, label) {
  if (!rows.length) { container.innerHTML='<div class="empty">无可展示数据</div>'; return; }
  const width=760,height=250,pad={l:48,r:16,t:15,b:30}; const values=rows.map(row=>Number(row[field])); const max=Math.max(...values,Number(limits["99"])*1.08,1e-9); const x=index=>pad.l+index/Math.max(1,rows.length-1)*(width-pad.l-pad.r); const y=value=>height-pad.b-value/max*(height-pad.t-pad.b); const points=values.map((value,index)=>`${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const limitLine=(value,color,text)=>`<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(value)}" y2="${y(value)}" stroke="${color}" stroke-width="1.5" stroke-dasharray="6 4"/><text x="${width-pad.r-3}" y="${y(value)-4}" text-anchor="end" fill="${color}" font-size="10">${text}</text>`;
  container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}趋势"><line x1="${pad.l}" x2="${pad.l}" y1="${pad.t}" y2="${height-pad.b}" stroke="#bcc6d1"/><line x1="${pad.l}" x2="${width-pad.r}" y1="${height-pad.b}" y2="${height-pad.b}" stroke="#bcc6d1"/>${limitLine(limits["95"],"#d19a20","95%")}${limitLine(limits["99"],"#cf3f36","99%")}<polyline points="${points}" fill="none" stroke="#176b87" stroke-width="2" vector-effect="non-scaling-stroke"/><text x="4" y="${pad.t+8}" fill="#5f6c7b" font-size="10">${max.toFixed(2)}</text><text x="${pad.l}" y="${height-8}" fill="#5f6c7b" font-size="10">${escapeHtml(displayTime(rows[0].timestamp))}</text><text x="${width-pad.r}" y="${height-8}" text-anchor="end" fill="#5f6c7b" font-size="10">${escapeHtml(displayTime(rows.at(-1).timestamp))}</text></svg>`;
}

function scoreScatter(container, rows) {
  if (!rows.length || !("pc1" in rows[0]) || !("pc2" in rows[0])) { container.innerHTML='<div class="empty">当前模型不足两个保留主元。</div>'; return; }
  const width=760,height=250,pad=28; const xs=rows.map(row=>Number(row.pc1)),ys=rows.map(row=>Number(row.pc2)); const maxX=Math.max(...xs.map(Math.abs),1e-9),maxY=Math.max(...ys.map(Math.abs),1e-9); const x=value=>width/2+value/maxX*(width/2-pad); const y=value=>height/2-value/maxY*(height/2-pad); const colors={normal:"#16845b",attention:"#d19a20",abnormal:"#cf3f36"}; const circles=rows.map(row=>`<circle cx="${x(Number(row.pc1))}" cy="${y(Number(row.pc2))}" r="3" fill="${colors[row.status]}" fill-opacity=".72"/>`).join(""); container.innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PC1与PC2得分散点"><line x1="${pad}" x2="${width-pad}" y1="${height/2}" y2="${height/2}" stroke="#d7dee8"/><line x1="${width/2}" x2="${width/2}" y1="${pad}" y2="${height-pad}" stroke="#d7dee8"/>${circles}<text x="${width-pad}" y="${height/2-5}" text-anchor="end" fill="#5f6c7b" font-size="10">PC1</text><text x="${width/2+5}" y="${pad+10}" fill="#5f6c7b" font-size="10">PC2</text></svg>`;
}

function clusterScatter(container, rows) {
  if (!rows.length) { container.innerHTML='<div class="empty">无可展示数据</div>'; return; }
  const palette=["#176b87","#cf3f36","#16845b","#d19a20","#7c3aed","#db2777","#0891b2","#65a30d","#ea580c","#475569"];
  const width=760,height=250,pad=28; const xs=rows.map(row=>Number(row.pc1)),ys=rows.map(row=>Number(row.pc2)); const maxX=Math.max(...xs.map(Math.abs),1e-9),maxY=Math.max(...ys.map(Math.abs),1e-9); const x=value=>width/2+value/maxX*(width/2-pad); const y=value=>height/2-value/maxY*(height/2-pad);
  const circles=rows.map(row=>`<circle cx="${x(Number(row.pc1))}" cy="${y(Number(row.pc2))}" r="3" fill="${palette[(row.cluster-1)%palette.length]}" fill-opacity=".72"><title>Cluster ${row.cluster} · ${escapeHtml(displayTime(row.timestamp))}</title></circle>`).join("");
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
