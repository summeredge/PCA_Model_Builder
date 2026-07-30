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

from .clustering import cluster_operating_states
from .contribution import exceedance_contribution_tables
from .dpca import fit_dpca
from .model_io import load_model_package, save_model_package
from .preprocessing import PreprocessingConfig, build_dynamic_matrix, infer_segment_ids
from .quality import QualityReport, inspect_data_quality
from .tag_config import engineering_ranges, normalize_tag_configs
from .validation import (
    build_validation_matrix,
    ensure_disjoint_windows,
    validation_context_start,
)


DEFAULT_PORT = 8775
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DATA_DIR = PROJECT_ROOT / ".web_data"
UPLOADS_DIR = WEB_DATA_DIR / "uploads"
RUNS_DIR = WEB_DATA_DIR / "runs"
MAX_REQUEST_BODY_BYTES = 200 * 1024 * 1024
MAX_CHART_POINTS = 1200
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


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
    tags = _required_tags(payload)
    parsed = _parse_timestamp_column(frame, timestamp_column)
    config = _preprocessing_config(payload)
    tag_configs = normalize_tag_configs(tags, payload.get("tag_configs"))
    normal = _select_window(
        parsed,
        timestamp_column,
        _required_text(payload, "normal_start"),
        _required_text(payload, "normal_end"),
    )
    _require_clean_data(
        normal,
        timestamp_column,
        tags,
        config.sample_interval_minutes,
        engineering_ranges(tag_configs),
    )
    indexed = _indexed_tags(normal, timestamp_column, tags)
    dynamic = build_dynamic_matrix(
        indexed,
        tags,
        config,
        infer_segment_ids(indexed.index, config.sample_interval_minutes),
    )
    if dynamic.empty:
        raise ValueError("平滑和 Lag 扩展后没有足够的训练样本")
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
    training_window = [
        pd.Timestamp(payload["normal_start"]).isoformat(),
        pd.Timestamp(payload["normal_end"]).isoformat(),
    ]
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
    }
    save_model_package(
        run_dir / "model.pcamodel",
        model,
        config=stored_config,
        training_windows=[training_window],
    )
    scores = model.score(dynamic)
    return {
        "run_id": run_id,
        "model_name": model_name,
        "validation_status": "draft",
        "training_rows": len(dynamic),
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


def cluster_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frame = _read_upload(payload)
    timestamp_column = _required_text(payload, "timestamp_column")
    tags = _required_tags(payload)
    parsed = _parse_timestamp_column(frame, timestamp_column)
    config = _preprocessing_config(payload)
    tag_configs = normalize_tag_configs(tags, payload.get("tag_configs"))
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


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _validated_id(_required_text(payload, "run_id"), "run_id")
    model_path = RUNS_DIR / run_id / "model.pcamodel"
    if not model_path.is_file():
        raise ValueError("模型运行记录不存在")
    model, manifest = load_model_package(model_path)
    config_data = manifest["config"]
    tags = list(config_data["tags"])
    tag_configs = normalize_tag_configs(tags, config_data.get("tag_configs"))
    timestamp_column = _required_text(payload, "timestamp_column")
    validation_window = (
        pd.Timestamp(_required_text(payload, "validation_start")),
        pd.Timestamp(_required_text(payload, "validation_end")),
    )
    training_windows = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in manifest["training_windows"]
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
    contributions = []
    for statistic, timestamp, value, limit95, table in exceedance_contribution_tables(
        model, dynamic, scores
    ):
        contributions.append(
            {
                "statistic": statistic,
                "timestamp": timestamp.isoformat(),
                "statistic_value": value,
                "limit_95": limit95,
                "tags": _contribution_records(table.head(8), tag_configs),
            }
        )

    result: dict[str, Any] = {
        "run_id": run_id,
        "model_validation_status": manifest["validation_status"],
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
    details = "；".join(f"{issue.code}({issue.count})" for issue in report.issues)
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


def _validated_id(value: str, label: str) -> str:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"无效的 {label}")
    return value


def _limit_payload(limits: dict[float, float]) -> dict[str, float]:
    return {str(int(alpha * 100)): float(value) for alpha, value in limits.items()}


def _status_counts(scores: pd.DataFrame) -> dict[str, int]:
    counts = Counter(scores["status"])
    return {status: int(counts.get(status, 0)) for status in ("normal", "attention", "abnormal")}


def _score_payload(scores: pd.DataFrame) -> list[dict[str, Any]]:
    if len(scores) <= MAX_CHART_POINTS:
        positions = np.arange(len(scores))
    else:
        positions = np.linspace(0, len(scores) - 1, MAX_CHART_POINTS, dtype=int)
        positions = np.unique(positions)
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


def _focus_timestamp(
    scores: pd.DataFrame, t2_limit_99: float, q_limit_99: float
) -> pd.Timestamp:
    t2_ratio = scores["t2"] / max(t2_limit_99, np.finfo(float).eps)
    q_ratio = scores["spe"] / max(q_limit_99, np.finfo(float).eps)
    return pd.Timestamp(pd.concat([t2_ratio, q_ratio], axis=1).max(axis=1).idxmax())


def _contribution_records(
    table: pd.DataFrame, tag_configs: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    tag_configs = tag_configs or {}
    return [
        {
            "tag": str(row.tag),
            "description": str(tag_configs.get(str(row.tag), {}).get("description", "")),
            "unit": str(tag_configs.get(str(row.tag), {}).get("unit", "")),
            "contribution_pct": float(row.contribution_pct),
            "lag_start_minutes": int(row.lag_start_minutes),
            "lag_end_minutes": int(row.lag_end_minutes),
        }
        for row in table.itertuples(index=False)
    ]


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
        if parsed.path == "/download/model":
            try:
                run_id = _validated_id(
                    parse_qs(parsed.query).get("run_id", [""])[0], "run_id"
                )
                self._send_model(run_id)
            except Exception as error:
                self._send_json({"error": str(error)}, 400)
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/upload":
                filename, content = self._multipart_file("file")
                self._send_json(save_upload(filename, content))
                return
            payload = self._json_body()
            if self.path == "/api/inspect":
                self._send_json(inspect_payload(payload))
                return
            if self.path == "/api/cluster":
                self._send_json(cluster_payload(payload))
                return
            if self.path == "/api/train":
                self._send_json(train_payload(payload))
                return
            if self.path == "/api/validate":
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
        if not path.is_file():
            raise ValueError("模型文件不存在")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Disposition", 'attachment; filename="model.pcamodel"'
        )
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
    .tag-options { display:grid; grid-template-columns:1fr 1fr; gap:5px; max-height:170px; overflow:auto; padding:8px; background:#fff; border:1px solid var(--line); border-radius:6px; }
    .tag-options label { display:flex; align-items:center; gap:6px; color:var(--text); overflow:hidden; }
    .tag-options input { width:auto; min-height:auto; }
    .tag-options span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .tag-config-list { display:grid; gap:6px; max-height:310px; overflow:auto; }
    .tag-config-list details { background:#fff; border:1px solid var(--line); border-radius:6px; padding:7px 8px; }
    .tag-config-list summary { cursor:pointer; font-weight:650; font-size:12px; }
    .tag-config-fields { display:grid; gap:7px; padding-top:8px; }
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
    @media (max-width:760px) { .chart-grid,.validation-box { grid-template-columns:1fr; } .row { grid-template-columns:1fr; } }
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
        <div id="tagOptions" class="tag-options"><span class="help">检查数据后显示连续数值列。</span></div>
        <div class="help">V1 将所有选中连续 Tag 使用同一组 Lag；工程标签和离散状态不要加入模型。</div>
        <div class="help">工程参数可选。工程量程用于拦截无效数据；正常范围和报警范围仅用于工程解释。</div>
        <div id="tagConfigList" class="tag-config-list"><span class="help">检查数据后可配置描述、单位和工程范围。</span></div>
      </div>
      <div class="group">
        <div class="group-title">3. 聚类辅助识别（可选）</div>
        <div class="row"><label>分析期开始<input id="analysisStart" type="datetime-local"></label><label>分析期结束<input id="analysisEnd" type="datetime-local"></label></div>
        <div class="row"><label>Cluster 数量<input id="clusterCount" type="number" min="2" max="10" value="3"></label><span class="help">使用下方相同的平滑、Lag 和解释率参数。</span></div>
        <button id="clusterButton" class="secondary" disabled>生成运行状态聚类</button>
        <div class="help">聚类只辅助发现运行模式。算法不会自动认定正常状态。</div>
      </div>
      <div class="group">
        <div class="group-title">4. 正常状态与 DPCA 参数</div>
        <div class="row"><label>正常期开始<input id="normalStart" type="datetime-local"></label><label>正常期结束<input id="normalEnd" type="datetime-local"></label></div>
        <div class="row"><label>采样间隔（分钟）<input id="sampleInterval" type="number" min="1" value="5"></label><label>尾随平滑（分钟）<input id="smoothingWindow" type="number" min="1" value="10"></label></div>
        <div class="row"><label>最大 Lag（分钟）<input id="maxLag" type="number" min="0" value="60"></label><label>Lag 步长（分钟）<input id="lagStep" type="number" min="1" value="5"></label></div>
        <div class="row"><label>累计解释率<input id="varianceThreshold" type="number" min="0.5" max="1" step="0.01" value="0.95"></label><label>主元数（可留空）<input id="components" type="number" min="1" placeholder="自动"></label></div>
        <label>模型名称<input id="modelName" value="D330_DPCA_Model_V1"></label>
        <button id="trainButton" disabled>训练 DPCA 草稿模型</button>
      </div>
      <div id="status" class="status info" role="status" aria-live="polite">请先上传 CSV。</div>
      <div class="help">数据缺失、重复、乱序或采样间隔不一致时训练会停止，不会静默清洗。</div>
    </section>
    <section class="results">
      <div class="tabs" role="tablist">
        <button class="tab active" data-panel="modelPanel">建模结果</button>
        <button class="tab" data-panel="clusterPanel">聚类辅助</button>
        <button class="tab" data-panel="validationPanel">独立验证</button>
      </div>
      <div id="modelPanel" class="panel active">
        <div id="modelEmpty" class="empty">完成训练后显示主元解释率、T²/SPE 和模型下载。</div>
        <div id="modelContent" hidden>
          <div id="modelMetrics" class="metrics"></div>
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
        <div id="clusterEmpty" class="empty">检查数据后，可对选定历史窗口生成运行状态聚类。</div>
        <div id="clusterContent" hidden>
          <div id="clusterMetrics" class="metrics"></div>
          <div class="chart-card"><h3>聚类状态空间 PC1 / PC2</h3><div id="clusterChart" class="chart"></div></div>
          <h3>Cluster 概览与代表性连续时段</h3>
          <div class="table-wrap"><table><thead><tr><th>Cluster</th><th>样本</th><th>占比</th><th>中心 PC1 / PC2</th><th>人工选择正常候选时段</th></tr></thead><tbody id="clusterTable"></tbody></table></div>
          <div class="notice">Cluster 只表示数据中的相似运行状态，不代表正常或异常。点击时段只会填入正常期窗口，仍需工程师确认后再训练。</div>
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
          <div class="table-wrap"><table><thead><tr><th>统计量</th><th>Tag</th><th>描述</th><th>单位</th><th>贡献</th><th>主要影响时间</th></tr></thead><tbody id="contributionTable"></tbody></table></div>
          <div class="notice">贡献表示该时间点偏离在模型中的来源，不等同于工艺根因；最终通过或不通过由工程师确认。</div>
        </div>
      </div>
    </section>
  </main>
<script>
const state = { fileId:null, runId:null, inspection:null, clustering:null, training:null };
const el = (id) => document.getElementById(id);

function setStatus(message, type="info") { const node=el("status"); node.textContent=message; node.className=`status ${type}`; }
function setBusy(button, busy, text) { if (!button.dataset.label) button.dataset.label=button.textContent; button.disabled=busy; button.textContent=busy?text:button.dataset.label; }
function localTime(value) { return value ? value.slice(0,16) : ""; }
function selectedTags() { return [...document.querySelectorAll('#tagOptions input:checked')].map(node=>node.value); }
function numberValue(id) { return Number(el(id).value); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch])); }
function tagConfigField(labelText,field,type="text") { const label=document.createElement("label"); label.textContent=labelText; const input=document.createElement("input"); input.type=type; input.dataset.field=field; if(type==="number") input.step="any"; label.append(input); return label; }
function renderTagConfigs(tags) {
  const list=el("tagConfigList"); list.replaceChildren();
  tags.forEach(tag=>{ const details=document.createElement("details"); details.dataset.tag=tag; const summary=document.createElement("summary"); summary.textContent=`${tag} · 连续变量`; const fields=document.createElement("div"); fields.className="tag-config-fields";
    const identity=document.createElement("div"); identity.className="row"; identity.append(tagConfigField("描述","description"),tagConfigField("单位","unit"));
    const engineering=document.createElement("div"); engineering.className="row"; engineering.append(tagConfigField("工程下限","engineering_min","number"),tagConfigField("工程上限","engineering_max","number"));
    const normal=document.createElement("div"); normal.className="row"; normal.append(tagConfigField("正常下限","normal_min","number"),tagConfigField("正常上限","normal_max","number"));
    const alarm=document.createElement("div"); alarm.className="row"; alarm.append(tagConfigField("报警下限","alarm_min","number"),tagConfigField("报警上限","alarm_max","number"));
    fields.append(identity,engineering,normal,alarm); details.append(summary,fields); list.append(details); });
}
function tagConfigPayload(tags) {
  const result={}; document.querySelectorAll('#tagConfigList details').forEach(details=>{ const tag=details.dataset.tag; if(!tags.includes(tag)) return; const config={type:"continuous"}; details.querySelectorAll("input[data-field]").forEach(input=>{ const value=input.value.trim(); config[input.dataset.field]=input.type==="number"?(value===""?null:Number(value)):value; }); result[tag]=config; }); return result;
}

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

el("uploadButton").addEventListener("click", async () => {
  const file=el("fileInput").files[0]; if (!file) { setStatus("请选择 CSV 文件。","warning"); return; }
  const button=el("uploadButton"); setBusy(button,true,"上传中…");
  try {
    const form=new FormData(); form.append("file",file);
    const data=await api("/api/upload",{method:"POST",body:form});
    state.fileId=data.file_id; fillSelect(el("timestampColumn"),data.columns); fillSelect(el("labelColumn"),data.columns,"不使用"); el("encoding").value=data.encoding;
    el("inspectButton").disabled=false; el("clusterButton").disabled=true; el("trainButton").disabled=true; el("validateButton").disabled=true;
    setStatus(`已上传 ${data.filename}，共 ${data.columns.length} 列。请选择时间列并检查数据。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("inspectButton").addEventListener("click", async () => {
  const button=el("inspectButton"); setBusy(button,true,"检查中…");
  try {
    const data=await api("/api/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value})});
    state.inspection=data; const options=el("tagOptions"); options.replaceChildren(); renderTagConfigs(data.numeric_columns);
    data.numeric_columns.forEach(tag=>{ const label=document.createElement("label"); const input=document.createElement("input"); input.type="checkbox"; input.value=tag; input.checked=true; const span=document.createElement("span"); span.textContent=tag; label.append(input,span); options.append(label); });
    el("analysisStart").value=localTime(data.time_start); el("analysisEnd").value=localTime(data.time_end); el("normalStart").value=localTime(data.time_start); el("normalEnd").value=localTime(data.suggested_normal_end); el("validationStart").value=localTime(data.suggested_validation_start); el("validationEnd").value=localTime(data.time_end);
    if (data.sample_interval_minutes) el("sampleInterval").value=String(data.sample_interval_minutes);
    el("clusterButton").disabled=false; el("trainButton").disabled=false;
    const issues=data.quality_issues.map(item=>`${item.code}(${item.count})`).join("、");
    setStatus(issues ? `检查完成：${data.rows} 行。发现 ${issues}；选择的训练窗口仍会再次检查。` : `检查完成：${data.rows} 行，识别 ${data.numeric_columns.length} 个连续数值列。`, issues?"warning":"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("clusterButton").addEventListener("click", async () => {
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el("clusterButton"); setBusy(button,true,"聚类中…"); setStatus("正在构建动态状态空间并执行聚类。","info");
  try {
    const payload={file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tags,tag_configs:tagConfigPayload(tags),analysis_start:el("analysisStart").value,analysis_end:el("analysisEnd").value,sample_interval_minutes:numberValue("sampleInterval"),smoothing_window_minutes:numberValue("smoothingWindow"),max_lag_minutes:numberValue("maxLag"),lag_step_minutes:numberValue("lagStep"),variance_threshold:numberValue("varianceThreshold"),n_clusters:numberValue("clusterCount")};
    const data=await api("/api/cluster",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.clustering=data; renderClustering(data); document.querySelector('[data-panel="clusterPanel"]').click();
    setStatus("聚类完成。请由工程师判断 Cluster，并选择代表性连续时段作为正常候选。","success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

el("trainButton").addEventListener("click", async () => {
  const tags=selectedTags(); if (tags.length<2) { setStatus("至少选择两个连续 Tag。","warning"); return; }
  const button=el("trainButton"); setBusy(button,true,"训练中…"); setStatus("正在构建动态矩阵并训练 DPCA，请勿关闭页面。","info");
  try {
    const components=el("components").value.trim();
    const payload={file_id:state.fileId,timestamp_column:el("timestampColumn").value,encoding:el("encoding").value,tags,tag_configs:tagConfigPayload(tags),normal_start:el("normalStart").value,normal_end:el("normalEnd").value,sample_interval_minutes:numberValue("sampleInterval"),smoothing_window_minutes:numberValue("smoothingWindow"),max_lag_minutes:numberValue("maxLag"),lag_step_minutes:numberValue("lagStep"),variance_threshold:numberValue("varianceThreshold"),n_components:components?Number(components):null,model_name:el("modelName").value};
    const data=await api("/api/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    state.runId=data.run_id; state.training=data; renderTraining(data); el("validateButton").disabled=false;
    setStatus(`训练完成：${data.training_rows} 个动态样本，${data.dynamic_features} 个动态特征。模型仍为草稿。`,"success");
  } catch (error) { setStatus(error.message,"error"); }
  finally { setBusy(button,false,""); }
});

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
function renderClustering(data) {
  el("clusterEmpty").hidden=true; el("clusterContent").hidden=false;
  el("clusterMetrics").innerHTML=metric("聚类动态样本",data.sample_count)+metric("状态空间主元",data.n_components)+metric("累计解释率",`${(data.cumulative_explained_variance*100).toFixed(1)}%`)+metric("Cluster 数量",data.clusters.length);
  clusterScatter(el("clusterChart"),data.points);
  const body=el("clusterTable"); body.replaceChildren();
  data.clusters.forEach(item=>{
    const tr=document.createElement("tr");
    [`Cluster ${item.cluster}`,item.count,`${(item.share*100).toFixed(1)}%`,`${item.pc1_center.toFixed(2)} / ${item.pc2_center.toFixed(2)}`].forEach(value=>{ const td=document.createElement("td"); td.textContent=value; tr.append(td); });
    const windows=document.createElement("td");
    item.representative_windows.forEach(window=>{ const button=document.createElement("button"); button.className="secondary"; button.style.margin="2px"; button.textContent=`${window.start.slice(0,16)} ～ ${window.end.slice(11,16)} (${window.count}点)`; button.addEventListener("click",()=>{ el("normalStart").value=localTime(window.start); el("normalEnd").value=localTime(window.end); setStatus(`已将 Cluster ${item.cluster} 的代表时段填入正常期。请确认工况后再训练。`,"warning"); }); windows.append(button); });
    tr.append(windows); body.append(tr);
  });
}

function renderTraining(data) {
  el("modelEmpty").hidden=true; el("modelContent").hidden=false;
  el("modelMetrics").innerHTML=metric("训练动态样本",data.training_rows)+metric("动态特征",data.dynamic_features)+metric("主元数",data.n_components)+metric("累计解释率",`${(data.cumulative_explained_variance*100).toFixed(1)}%`)+metric("关注 / 异常",`${data.status_counts.attention} / ${data.status_counts.abnormal}`);
  const variance=el("varianceChart"); variance.replaceChildren(); const max=Math.max(...data.explained_variance,0.01);
  data.explained_variance.slice(0,30).forEach((value,index)=>{ const bar=document.createElement("div"); bar.className=`variance-bar ${index<data.n_components?"selected":""}`; bar.style.height=`${Math.max(3,value/max*95)}px`; const label=document.createElement("span"); label.textContent=`${(value*100).toFixed(0)}%`; bar.title=`PC${index+1}: ${(value*100).toFixed(2)}%`; bar.append(label); variance.append(bar); });
  lineChart(el("t2Chart"),data.scores,"t2",data.t2_limits,"T²"); lineChart(el("speChart"),data.scores,"spe",data.q_limits,"SPE"); scoreScatter(el("scoreChart"),data.scores); el("modelDownload").href=data.model_download;
}

function renderValidation(data) {
  el("validationEmpty").hidden=true; el("validationContent").hidden=false;
  el("validationMetrics").innerHTML=metric("验证样本",data.scored_rows)+metric("正常",data.status_counts.normal)+metric("关注",data.status_counts.attention)+metric("异常",data.status_counts.abnormal)+metric("模型状态（草稿）","待工程确认");
  lineChart(el("validationT2Chart"),data.scores,"t2",data.t2_limits,"T²"); lineChart(el("validationSpeChart"),data.scores,"spe",data.q_limits,"SPE");
  el("contributionHint").textContent=data.contributions.length ? "仅展示达到 95% 控制限的统计量；每类统计量使用其越界程度最高的时间点。" : "T² 和 SPE 均未达到 95% 控制限，不输出异常贡献。";
  const body=el("contributionTable"); body.innerHTML="";
  data.contributions.forEach(group=>group.tags.forEach(item=>{ const tr=document.createElement("tr"); const lag=item.lag_start_minutes===item.lag_end_minutes?`${item.lag_start_minutes} 分钟前`:`${item.lag_start_minutes}～${item.lag_end_minutes} 分钟前`; tr.innerHTML=`<td>${escapeHtml(group.statistic.toUpperCase())}</td><td>${escapeHtml(item.tag)}</td><td>${escapeHtml(item.description)}</td><td>${escapeHtml(item.unit)}</td><td class="numeric">${item.contribution_pct.toFixed(1)}%</td><td>${escapeHtml(lag)}</td>`; body.append(tr); }));
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

document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>{ document.querySelectorAll(".tab").forEach(node=>node.classList.toggle("active",node===button)); document.querySelectorAll(".panel").forEach(panel=>panel.classList.toggle("active",panel.id===button.dataset.panel)); }));
el("resetButton").addEventListener("click",()=>location.reload());
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
