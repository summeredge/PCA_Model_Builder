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


def apply_model_results_ui(html: str) -> str:
    """Remove XY scatter UI/code and load the model-results asset."""
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
    if "</body>" not in result:
        raise ValueError("Web HTML缺少body结束标签")
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
    return result


INDEX_HTML = apply_model_results_ui(quality_app.INDEX_HTML)


class ModelResultsHandler(_BASE_WEB._Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
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
        if path not in {"/api/train", "/api/trend"}:
            super().do_POST()
            return
        try:
            payload = self._json_body()
            result = train_payload(payload) if path == "/api/train" else _TREND_APP.trend_payload(payload)
            self._send_json(result)
        except Exception as error:
            self._send_json({"error": str(error)}, 400)


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
