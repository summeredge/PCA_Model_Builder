from __future__ import annotations

import argparse
from typing import Sequence

from . import web_dataproject as app


_QUALITY_GRID_CSS = r"""
<style id="qualityProfileGridStyle">
  .quality-profile-grid {
    display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:8px;
    align-items:start;
  }
  .quality-tag-controls {
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    align-items:end;
    margin:0 0 4px;
  }
  .quality-tag-controls label { min-width:180px; }
  .quality-profile-section { min-width:0; }
  .quality-profile-section h4 { margin:6px 0; }
  .quality-profile-section .table-wrap {
    max-height:none;
    overflow:visible;
  }
  .quality-profile-section .quality-profile-table {
    width:100%;
    table-layout:fixed;
  }
  .quality-profile-section .quality-profile-table th,
  .quality-profile-section .quality-profile-table td {
    width:auto;
    padding:5px 7px;
  }
  .quality-profile-section .quality-profile-table th:nth-child(odd) {
    width:24%;
    white-space:nowrap;
  }
  .quality-profile-section .quality-profile-table td:nth-child(even) {
    width:26%;
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
  const fields = [["sample_count","样本数"],["valid_count","有效数"],["missing_count","缺失数"],["missing_rate","缺失率"],["non_numeric_count","非数值数"],["non_finite_count","非有限值数"],["unique_count","唯一值"],["minimum","最小值"],["maximum","最大值"],["mean","均值"],["median","中位数"],["standard_deviation","标准差"],["p01","P1"],["p05","P5"],["p95","P95"],["p99","P99"],["engineering_range_outside_count","工程范围越界"],["normal_range_outside_count","正常范围外"],["alarm_range_outside_count","报警范围外"]];
  let previousQuality = null;

  function qualityProfileTable(title, profile) {
    const rows = [];
    for (let index = 0; index < fields.length; index += 2) {
      const first = fields[index];
      const second = fields[index + 1];
      rows.push(`<tr><th>${first[1]}</th><td>${formatStat(first[0], profile[first[0]])}</td>${second ? `<th>${second[1]}</th><td>${formatStat(second[0], profile[second[0]])}</td>` : "<th></th><td></td>"}</tr>`);
    }
    return `<h4>${title}</h4><div class="table-wrap"><table class="quality-profile-table"><tbody>${rows.join("")}</tbody></table></div>`;
  }

  function addTagFilter(select) {
    const tagLabel = select.closest("label");
    if (!tagLabel || el("qualityTagFilter")) return;
    const controls = document.createElement("div");
    controls.className = "quality-tag-controls";
    const filterLabel = document.createElement("label");
    filterLabel.textContent = "筛选 Tag：";
    const filter = document.createElement("input");
    filter.id = "qualityTagFilter";
    filter.type = "text";
    filter.placeholder = "输入位号筛选";
    filter.addEventListener("input", () => window.renderCurrentTagQuality());
    filterLabel.append(filter);
    tagLabel.before(controls);
    controls.append(filterLabel, tagLabel);
  }

  window.renderCurrentTagQuality = function renderCurrentTagQualityFourColumns() {
    const container = el("currentTagQuality");
    const select = el("qualityTagSelect");
    if (!container) return;
    if (select) addTagFilter(select);
    const filter = el("qualityTagFilter");
    const tags = state.quality ? state.quality.tags : [];
    if (filter && state.quality !== previousQuality) {
      filter.value = "";
      previousQuality = state.quality;
    }
    const filteredTags = tags.filter((item) =>
      String(item.tag).toLowerCase().includes((filter?.value || "").trim().toLowerCase())
    );
    if (select) {
      select.replaceChildren();
      select.disabled = !filteredTags.length;
      if (filteredTags.length) {
        if (!filteredTags.some((item) => item.tag === state.selectedTag)) {
          state.selectedTag = filteredTags[0].tag;
        }
        filteredTags.forEach((item) => {
          const option = document.createElement("option");
          option.value = item.tag;
          option.textContent = item.tag;
          select.append(option);
        });
        select.value = state.selectedTag;
      } else if (tags.length) {
        const option = document.createElement("option");
        option.textContent = "无匹配 Tag";
        option.disabled = true;
        option.selected = true;
        select.append(option);
      }
    }
    if (state.quality && tags.length && !filteredTags.length) {
      container.className = "empty";
      container.textContent = "未找到匹配的 Tag。";
      return;
    }
    const item = state.selectedTag ? qualityFor(state.selectedTag) : null;
    if (!state.quality || !item) {
      container.className = "empty";
      container.textContent = state.quality && !tags.length
        ? "没有可查看的建模 Tag。"
        : state.qualityStatus === "changed"
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
