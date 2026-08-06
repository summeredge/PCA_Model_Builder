from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import threading
from typing import Any, Sequence
from urllib.parse import urlparse
import webbrowser

from . import web_quality_layout as quality_app
from .loading_plot import loading_plot_payload
from .model_diagnostics import compare_candidate_runs, model_structure_diagnostic


_BASE_WEB = quality_app.app.base_web
_TREND_APP = quality_app.app
_ASSET_PATH = Path(__file__).with_name("model_results.js")
_SCATTER_SECTION = re.compile(
    r'\n    <section class="dp-scatter-section">.*?\n    </section>', re.DOTALL
)
_SCATTER_HANDLER = re.compile(
    r'\n  \$\("dpDrawScatter"\)\.addEventListener\("click", async \(\) => \{.*?\n  \}\);\n\n  function renderTrendPage',
    re.DOTALL,
)
_SCATTER_RENDERER = re.compile(
    r'\n  function renderScatterMatrix\(data, xTags, yTags\) \{.*?\n  \}\n\n  function finiteNumber',
    re.DOTALL,
)

_FORM_ALIGNMENT_STYLE = r"""
<style id="webFormAlignmentStyle">
  #batchPanel .actions {
    display:grid;
    grid-template-columns:max-content minmax(260px,1fr) max-content max-content max-content;
    gap:10px;
    align-items:end;
  }
  #batchPanel .actions > .download,
  #batchPanel .actions > button {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:42px;
    height:42px;
    white-space:nowrap;
  }
  #batchPanel .actions > label.secondary {
    display:grid;
    grid-template-rows:auto 42px;
    gap:4px;
    align-content:start;
    min-width:0;
    padding:0;
    background:transparent;
    color:var(--muted);
    font-size:12px;
  }
  #batchPanel #tagConfigFile {
    min-width:0;
    min-height:42px;
    height:42px;
    padding:5px 8px;
  }
  #batchPanel #importSummary { margin-top:10px; }

  #clusterPanel .group { gap:10px; }
  #clusterPanel .row {
    column-gap:12px;
    align-items:start;
  }
  #clusterPanel .row > label {
    min-width:0;
    align-content:start;
  }
  #clusterPanel .row input {
    min-height:42px;
    height:42px;
  }
  #clusterPanel #clusterButton {
    align-self:end;
    min-height:42px;
    height:42px;
  }

  #engineeringPanel .detail-fields {
    display:grid;
    gap:10px;
  }
  #engineeringPanel .detail-fields .row {
    column-gap:12px;
    align-items:start;
  }
  #engineeringPanel .detail-fields .row > label {
    min-width:0;
    align-self:start;
    align-content:start;
  }
  #engineeringPanel .detail-fields input,
  #engineeringPanel .detail-fields select {
    min-height:42px;
    height:42px;
  }
  #engineeringPanel #tagComment {
    display:block;
    min-height:70px;
    resize:vertical;
  }
  #engineeringPanel #tagRole { align-self:start; }
  #engineeringPanel #saveTagConfig {
    min-height:42px;
    margin-top:2px;
  }

  @media (max-width:900px) {
    #batchPanel .actions {
      grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    }
    #batchPanel .actions > label.secondary {
      grid-column:1 / -1;
    }
    #batchPanel .actions > .download,
    #batchPanel .actions > button {
      width:100%;
    }
  }
</style>
"""


_IBM_DESIGN_STYLE = r"""
<style id="ibmDesignStyle">
  :root {
    --bg:#ffffff;
    --panel:#ffffff;
    --line:#e0e0e0;
    --line-soft:#e0e0e0;
    --text:#161616;
    --muted:#525252;
    --accent:#0f62fe;
    --accent-soft:#edf5ff;
    --green:#24a148;
    --warn:#f1c21b;
    --danger:#da1e28;
    --normal:#24a148;
    --attention:#f1c21b;
    --abnormal:#da1e28;
  }

  *, *::before, *::after { border-radius:0 !important; box-shadow:none !important; }
  body {
    background:var(--bg);
    color:var(--text);
    font-family:"IBM Plex Sans","Microsoft YaHei","Segoe UI",Arial,sans-serif;
    font-size:16px;
    font-weight:400;
    letter-spacing:.16px;
    line-height:1.5;
  }
  header { padding:24px 32px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
  h1 { margin:0 0 8px; font-size:32px; font-weight:300; line-height:1.25; }
  h2 { font-size:24px; font-weight:400; line-height:1.33; }
  h3 { font-size:20px; font-weight:400; line-height:1.4; }
  h4 { font-size:16px; font-weight:600; line-height:1.29; }
  .subtitle, .help, label, .legend, .dp-legend { color:var(--muted); }
  .subtitle { font-size:14px; line-height:1.29; }
  .help, label, .legend, .dp-legend { font-size:14px; line-height:1.29; }
  main { max-width:1584px; margin:0 auto; gap:16px; padding:16px; }
  section { border:1px solid var(--line); padding:16px; background:var(--panel); }
  .controls { gap:16px; }
  .group, .metric, .validation-box, .exploration-controls, .dp-inline-help {
    background:#f4f4f4;
    border:1px solid var(--line);
  }
  .group { gap:12px; padding:16px; }
  .group-title, .sub-title { font-size:14px; font-weight:600; line-height:1.29; }
  .sub-title { border-top-color:var(--line); }
  input, select, textarea {
    background:#f4f4f4;
    border:0;
    border-bottom:1px solid #8c8c8c;
    color:var(--text);
    font:inherit;
    padding:11px 16px;
  }
  input:focus, select:focus, textarea:focus {
    outline:2px solid var(--accent);
    outline-offset:-2px;
    border-bottom:2px solid var(--accent);
  }
  button, .download {
    min-height:42px;
    border:1px solid var(--accent);
    background:var(--accent);
    color:#ffffff;
    font:400 14px/1.29 "IBM Plex Sans","Microsoft YaHei","Segoe UI",Arial,sans-serif;
    letter-spacing:.16px;
    padding:12px 16px;
    text-decoration:none;
  }
  button:hover, .download:hover { background:#0050e6; border-color:#0050e6; }
  button:active, .download:active { background:#002d9c; border-color:#002d9c; }
  button.secondary, .inner-tab, .tab {
    background:#161616;
    border-color:#161616;
    color:#ffffff;
  }
  button.secondary:hover, .inner-tab:hover, .tab:hover { background:#262626; border-color:#262626; }
  .tabs, .inner-tabs { gap:0; border-bottom:1px solid var(--line); padding-bottom:0; }
  .tab, .inner-tab { border:0; border-bottom:2px solid transparent; }
  .tab.active, .inner-tab.active {
    background:#ffffff;
    border-bottom-color:var(--accent);
    color:var(--text);
    font-weight:600;
  }
  .status, .notice, .issue-card, .tag-options, .compact-list, .table-wrap,
  .trend-chart, .chart, .dp-chart, .empty, .variance, .exploration-timeline,
  .dp-trend-stat-card, .dp-scatter-chart {
    border-color:var(--line);
    background:#ffffff;
  }
  .status { min-height:42px; padding:12px 16px; }
  .status.info { background:#edf5ff; color:#0043ce; border-color:#0f62fe; }
  .status.success { background:#e8f5e9; color:#0e6027; border-color:#24a148; }
  .status.warning, .notice { background:#fff8e1; color:#6f4e00; border-color:#f1c21b; }
  .status.error { background:#fff1f1; color:#a2191f; border-color:#da1e28; }
  .issue-card, .notice { border-left-width:4px; }
  .tag-row.selected { background:#edf5ff; }
  .metric { padding:16px; }
  .metric strong { font-size:32px; font-weight:300; line-height:1.25; }
  .chart-card { gap:12px; }
  .chart-card h3 { margin:0; }
  .chart, .dp-chart, .trend-chart { background:#ffffff; }
  .empty { border-style:dashed; color:var(--muted); }
  .variance-bar { background:var(--accent); }
  .variance-bar.selected { background:var(--green); }
  .table-wrap, .exploration-timeline { border:1px solid var(--line); }
  th, td { border-bottom-color:var(--line); padding:12px 16px; }
  th { background:#f4f4f4; color:var(--text); font-weight:600; }
  a:not(.download) { color:var(--accent); }
  button:focus-visible, .download:focus-visible, a:focus-visible {
    outline:2px solid var(--accent);
    outline-offset:2px;
  }
  @media (max-width:672px) {
    header { padding:16px; }
    h1 { font-size:32px; }
    main { padding:8px; }
    button, .download, input, select { min-height:48px; }
  }
</style>
"""


def apply_model_results_ui(html: str) -> str:
    """Remove XY scatter UI/code and load the final Web assets."""
    if 'src="/assets/model-results.js"' in html:
        return html
    result, section_count = _SCATTER_SECTION.subn("", html, count=1)
    result, handler_count = _SCATTER_HANDLER.subn(
        "\n\n  function renderTrendPage", result, count=1
    )
    result, renderer_count = _SCATTER_RENDERER.subn(
        "\n  function finiteNumber", result, count=1
    )
    result = re.sub(r'\n  const scatterIds = \[.*?\];', "", result, count=1)
    result = result.replace("[...trendIds, ...scatterIds].forEach", "trendIds.forEach", 1)
    result = result.replace('    $("dpDrawScatter").disabled = !values.length;\n', "", 1)
    if (section_count, handler_count, renderer_count) != (1, 1, 1):
        raise ValueError("无法完整移除趋势页XY散点矩阵")
    if any(marker in result for marker in ("XY 散点矩阵", "dpScatter", "renderScatterMatrix")):
        raise ValueError("趋势页仍残留XY散点矩阵代码")
    if "</head>" not in result or "</body>" not in result:
        raise ValueError("Web HTML缺少head或body结束标签")
    result = result.replace(
        "</head>", f"{_FORM_ALIGNMENT_STYLE}\n{_IBM_DESIGN_STYLE}\n</head>", 1
    )
    return result.replace(
        "</body>",
        '<script src="/assets/model-results.js" defer></script>\n</body>',
        1,
    )


def train_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _BASE_WEB.train_payload(payload)
    model_path = _BASE_WEB.RUNS_DIR / str(result["run_id"]) / "model.pcamodel"
    model, manifest = _BASE_WEB.load_model_package(model_path)
    result["loading_plot"] = loading_plot_payload(model, manifest)
    if (
        result["model_purpose"] == "normal_state"
        and result["model_status"] == "candidate"
    ):
        result["model_diagnostic"] = model_structure_diagnostic(
            model, manifest, str(result["run_id"])
        )
    return result


def model_diagnostic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = _BASE_WEB._validated_id(str(payload.get("run_id", "")), "run_id")
    model_path = _BASE_WEB.RUNS_DIR / run_id / "model.pcamodel"
    if not model_path.is_file():
        raise ValueError("候选模型运行记录不存在")
    try:
        model, manifest = _BASE_WEB.load_model_package(model_path)
    except ValueError as error:
        raise ValueError("候选模型包损坏") from error
    if (
        manifest["model_purpose"] != "normal_state"
        or manifest["model_status"] != "candidate"
    ):
        raise ValueError("仅允许查看normal_state/candidate模型诊断")
    return model_structure_diagnostic(model, manifest, run_id)


def candidate_models_payload() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if not _BASE_WEB.RUNS_DIR.is_dir():
        return {"candidates": candidates}
    for run_dir in sorted(_BASE_WEB.RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        model_path = run_dir / "model.pcamodel"
        if not model_path.is_file():
            continue
        try:
            model, manifest = _BASE_WEB.load_model_package(model_path)
        except ValueError:
            continue
        if (
            manifest["model_purpose"] == "normal_state"
            and manifest["model_status"] == "candidate"
        ):
            candidates.append(
                {
                    "run_id": run_dir.name,
                    "model_name": str(manifest["config"]["model_name"]),
                    "training_dynamic_samples": int(model.n_samples),
                }
            )
    return {"candidates": candidates}


INDEX_HTML = apply_model_results_ui(quality_app.INDEX_HTML)


class ModelResultsHandler(_BASE_WEB._Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/model-candidates":
            self._send_json(candidate_models_payload())
            return
        if path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/assets/model-results.js":
            self._send_text(
                _ASSET_PATH.read_text(encoding="utf-8"),
                "application/javascript; charset=utf-8",
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {
            "/api/train",
            "/api/trend",
            "/api/model-diagnostics",
            "/api/model-comparison",
        }:
            super().do_POST()
            return
        try:
            payload = self._json_body()
            result = (
                train_payload(payload)
                if path == "/api/train"
                else _TREND_APP.trend_payload(payload)
                if path == "/api/trend"
                else model_diagnostic_payload(payload)
                if path == "/api/model-diagnostics"
                else compare_candidate_runs(payload.get("run_ids"), _BASE_WEB.RUNS_DIR)
            )
            self._send_json(result)
        except Exception as error:
            self._send_json(_BASE_WEB.error_payload(error), 400)


def run_server(
    host: str = "127.0.0.1",
    port: int = _BASE_WEB.DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    _BASE_WEB.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _BASE_WEB.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), ModelResultsHandler)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"PCA Model Builder 本地服务已启动：{url}")
    print("关闭此窗口即可停止服务。")
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local PCA Model Builder web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_BASE_WEB.DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
