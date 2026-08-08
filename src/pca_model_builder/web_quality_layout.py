from __future__ import annotations

import argparse
from typing import Sequence

from . import web_dataproject as app


_QUALITY_GRID_CSS = r"""
<style id="qualityProfileGridStyle">
  .quality-profile-grid {
    display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:12px;
    align-items:start;
  }
  .quality-profile-section { min-width:0; }
  .quality-profile-section h4 { margin:8px 0; }
  .quality-profile-section .table-wrap {
    max-height:none;
    overflow:visible;
  }
  .quality-profile-section table { table-layout:fixed; }
  .quality-profile-section th,
  .quality-profile-section td { width:50%; }
  .quality-profile-section td {
    text-align:right;
    font-variant-numeric:tabular-nums;
  }
  @media (max-width:900px) {
    .quality-profile-grid { grid-template-columns:1fr; }
  }
</style>
"""


_QUALITY_GRID_SCRIPT = r"""
<script id="qualityProfileGridScript">
(() => {
  window.renderCurrentTagQuality = function renderCurrentTagQualityFourColumns() {
    const container = el("currentTagQuality");
    if (!container) return;
    const item = state.selectedTag ? qualityFor(state.selectedTag) : null;
    if (!state.quality || !item) {
      container.className = "empty";
      container.textContent = state.qualityStatus === "changed"
        ? "配置已变更，请重新执行建模质量检查。"
        : state.qualityStatus === "checking"
        ? "正在执行建模质量检查。"
        : "尚未执行建模质量检查。";
      return;
    }
    const role = state.registry[item.tag]?.role || item.role;
    const issueHtml = item.issues.length
      ? item.issues.map((issue) => `<li>${escapeHtml(issue.message)}</li>`).join("")
      : "<li>无质量问题</li>";
    container.className = "";
    container.innerHTML = `
      <div class="issue-card ${item.status}">
        <strong>${escapeHtml(item.tag)} · ${escapeHtml(role)} · ${escapeHtml(item.status)}</strong>
        <div class="quality-profile-grid">
          <div class="quality-profile-section">${qualityProfileTable("全数据统计", item.full)}</div>
          <div class="quality-profile-section">${qualityProfileTable("参考期统计", item.reference)}</div>
        </div>
        <h4>质量问题与建议</h4>
        <ul>${issueHtml}</ul>
        <span>建议操作：${escapeHtml(item.suggested_action)}</span>
      </div>`;
  };
})();
</script>
"""


def apply_quality_grid_ui(html: str) -> str:
    """Add the responsive four-column quality-profile layout once."""
    if 'id="qualityProfileGridStyle"' in html:
        return html
    if "</head>" not in html or "</body>" not in html:
        raise ValueError("Web HTML缺少head或body结束标签")
    return html.replace("</head>", _QUALITY_GRID_CSS + "\n</head>", 1).replace(
        "</body>", _QUALITY_GRID_SCRIPT + "\n</body>", 1
    )


INDEX_HTML = apply_quality_grid_ui(app.INDEX_HTML)


def run_server(
    host: str = "127.0.0.1",
    port: int = app.base_web.DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    app.INDEX_HTML = INDEX_HTML
    app.run_server(host, port, open_browser=open_browser)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local PCA Model Builder web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=app.base_web.DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
