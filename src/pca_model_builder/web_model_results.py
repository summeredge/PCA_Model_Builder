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
  #engineeringPanel .batch-config {
    width:100%;
    max-width:760px;
    display:grid;
    gap:10px;
  }
  #engineeringPanel .batch-config-title {
    font-size:16px;
    font-weight:600;
  }
  #engineeringPanel .batch-config .actions {
    display:grid;
    grid-template-columns:max-content minmax(260px,1fr) max-content max-content max-content;
    gap:10px;
    align-items:end;
  }
  #engineeringPanel .batch-config .actions > .download,
  #engineeringPanel .batch-config .actions > button {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:42px;
    height:42px;
    white-space:nowrap;
  }
  #engineeringPanel .batch-config .actions > label.secondary {
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
  #engineeringPanel .batch-config #tagConfigFile {
    min-width:0;
    min-height:42px;
    height:42px;
    padding:5px 8px;
  }
  #engineeringPanel .batch-config #importSummary { margin-top:10px; }

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
    #engineeringPanel .batch-config .actions {
      grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    }
    #engineeringPanel .batch-config .actions > label.secondary {
      grid-column:1 / -1;
    }
    #engineeringPanel .batch-config .actions > .download,
    #engineeringPanel .batch-config .actions > button {
      width:100%;
    }
  }
</style>
"""


_APPLE_DESIGN_STYLE = r"""
<style id="appleDesignStyle">
  :root {
    --bg:#f5f5f7;
    --panel:#ffffff;
    --line:#e0e0e0;
    --line-soft:#f0f0f0;
    --text:#1d1d1f;
    --muted:#7a7a7a;
    --accent:#0066cc;
    --accent-soft:#f5f5f7;
    --green:#24a148;
    --warn:#f1c21b;
    --danger:#da1e28;
    --normal:#24a148;
    --attention:#f1c21b;
    --abnormal:#da1e28;
  }

  *, *::before, *::after { box-shadow:none !important; }
  body {
    background:var(--bg);
    color:var(--text);
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
    font-size:17px;
    font-weight:400;
    letter-spacing:-.374px;
    line-height:1.47;
  }
  header { padding:12px 32px 10px; background:#000000; border-bottom:0; color:#ffffff; }
  h1 { margin:0 0 4px; font-size:21px; font-weight:600; line-height:1.19; letter-spacing:.231px; }
  h2 { font-size:34px; font-weight:600; line-height:1.1; letter-spacing:-.374px; }
  h3 { font-size:21px; font-weight:600; line-height:1.19; letter-spacing:.231px; }
  h4 { font-size:17px; font-weight:600; line-height:1.24; letter-spacing:-.374px; }
  .subtitle, .help, label, .legend, .dp-legend { color:var(--muted); }
  .subtitle { color:#cccccc; font-size:14px; line-height:1.43; }
  .help, label, .legend, .dp-legend { font-size:14px; line-height:1.43; }
  main {
    width:100%;
    max-width:none;
    margin:0;
    grid-template-columns:630px minmax(0,1fr);
    gap:24px;
    padding:24px 20px;
  }
  section { border:0; border-radius:0; padding:32px; background:var(--panel); }
  .controls, .controls .group { min-width:0; }
  .controls { gap:20px; }
  .results { gap:24px; }
  .row, .actions, .tag-toolbar, .detail-fields { gap:12px; }
  .inner-panel.active, .panel.active { gap:24px; }
  .group, .metric, .validation-box, .exploration-controls, .dp-inline-help {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:6px;
  }
  .group { gap:12px; padding:24px; }
  .group-title, .sub-title { font-size:14px; font-weight:600; line-height:1.29; }
  .sub-title { border-top-color:var(--line); }
  input, select, textarea {
    height:30px;
    min-height:30px;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:6px;
    color:var(--text);
    font:inherit;
    padding:4px 10px;
  }
  input:focus, select:focus, textarea:focus {
    outline:2px solid var(--accent);
    outline-offset:2px;
    border-color:var(--accent);
  }
  button, .download {
    box-sizing:border-box;
    height:30px;
    min-height:30px;
    max-width:100%;
    border:1px solid var(--accent);
    background:var(--accent);
    color:#ffffff;
    border-radius:4px;
    font:400 14px/20px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
    letter-spacing:-.374px;
    padding:4px 12px;
    text-decoration:none;
  }
  button.secondary {
    background:var(--bg);
    border-color:var(--line-soft);
    color:var(--accent);
    border-radius:4px;
  }
  #inspectButton, #qualityButton, #saveTagConfig { background:var(--accent); border-color:var(--accent); color:#ffffff; }
  .tag-toolbar button { min-height:36px; padding:8px 14px; font-size:14px; }
  #tagOptions {
    max-height:300px;
    padding:6px;
    gap:2px;
  }
  #tagOptions .tag-row {
    min-height:30px;
    height:30px;
    padding:3px 6px;
    grid-template-columns:22px minmax(0,1fr) max-content;
    align-items:center;
    font-size:14px;
    line-height:20px;
  }
  #tagOptions .tag-row input[type=checkbox] {
    width:16px;
    height:16px;
    min-height:16px;
    padding:0;
    margin:0;
  }
  #tagOptions .tag-state { font-size:12px; line-height:20px; }
  #engineeringPanel #tagRole,
  #engineeringPanel #tagComment {
    box-sizing:border-box;
    height:42px;
    min-height:42px;
  }
  button:focus-visible, .download:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .tabs, .inner-tabs {
    gap:0;
    border-bottom:1px solid var(--line);
    padding-bottom:0;
    background:var(--bg);
  }
  .tab, .inner-tab {
    border:0;
    border-bottom:2px solid transparent;
    border-radius:0;
    background:transparent;
    color:var(--text);
    font-size:14px;
    letter-spacing:-.224px;
    padding:8px 15px;
  }
  .tab:hover, .inner-tab:hover { background:var(--accent-soft); }
  .tab.active, .inner-tab.active {
    background:var(--panel);
    border-bottom-color:var(--accent);
    color:var(--text);
    font-weight:600;
  }
  .status, .notice, .issue-card, .tag-options, .compact-list, .table-wrap,
  .trend-chart, .chart, .dp-chart, .empty, .variance, .exploration-timeline,
  .dp-trend-stat-card, .dp-scatter-chart {
    border-color:var(--line);
    background:var(--panel);
  }
  .tag-options, .compact-list, .table-wrap, .trend-chart, .chart, .dp-chart,
  .empty, .variance, .exploration-timeline, .dp-trend-stat-card,
  .dp-scatter-chart { border-radius:6px; }
  .status { min-height:42px; padding:12px 16px; }
  .status.info { background:#edf5ff; color:#0043ce; border-color:#0f62fe; }
  .status.success { background:#e8f5e9; color:#0e6027; border-color:#24a148; }
  .status.warning, .notice { background:#fff8e1; color:#6f4e00; border-color:#f1c21b; }
  .status.error { background:#fff1f1; color:#a2191f; border-color:#da1e28; }
  #modelPanel #modelQualityStatus,
  #modelPanel #qualityButton { width:fit-content; justify-self:start; }
  #modelPanel #modelQualityStatus { max-width:100%; }
  #modelPanel #currentTagQuality { max-width:1200px; }
  .issue-card, .notice { border-left-width:4px; border-radius:6px; }
  .tag-row.selected { background:#edf5ff; }
  .metric { padding:24px; }
  .metrics { grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; }
  .metric { min-width:0; padding:14px 12px; }
  .metric strong { font-size:28px; font-weight:600; line-height:1.1; letter-spacing:0; white-space:nowrap; }
  .chart-card { gap:12px; }
  .chart-card h3 { margin:0; }
  .chart, .dp-chart, .trend-chart { background:var(--panel); }
  .empty { border-style:dashed; color:var(--muted); }
  .variance-bar { background:var(--accent); }
  .variance-bar.selected { background:var(--green); }
  .table-wrap, .exploration-timeline { border:1px solid var(--line); }
  th, td { border-bottom-color:var(--line); padding:12px 16px; }
  th { background:var(--line-soft); color:var(--text); font-weight:600; }
  a:not(.download) { color:var(--accent); }
  #engineeringPanel #tagComment { resize:vertical; }
  button:focus-visible, .download:focus-visible, a:focus-visible {
    outline:2px solid var(--accent);
    outline-offset:2px;
  }
  @media (max-width:640px) {
    header { padding:12px 16px 10px; }
    h1 { font-size:21px; }
    main { padding:12px; }
    section { padding:24px; }
    h2 { font-size:34px; }
    button, .download, input, select, textarea { height:30px; min-height:30px; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
</style>
"""


_WORKBENCH_UI_STYLE = r"""
<style id="workbenchUiStyle">
  main { grid-template-columns:280px minmax(0,1fr); align-items:start; }
  .workflow-sidebar { position:sticky; top:16px; display:grid; gap:14px; padding:20px; }
  .workflow-sidebar-title { margin:0; font-size:17px; font-weight:600; }
  .workflow-steps { display:grid; gap:8px; }
  .workflow-step {
    display:grid;
    grid-template-columns:30px minmax(0,1fr) auto;
    gap:8px;
    width:100%;
    height:auto;
    min-height:76px;
    padding:11px;
    border:1px solid var(--line);
    border-radius:6px;
    background:var(--panel);
    color:var(--text);
    text-align:left;
  }
  .workflow-step.active { border-color:var(--accent); background:#edf5ff; }
  .workflow-step.complete .workflow-step-number { background:var(--green); }
  .workflow-step-number {
    display:grid;
    place-items:center;
    width:28px;
    height:28px;
    border-radius:50%;
    background:var(--accent);
    color:#fff;
    font-weight:600;
  }
  .workflow-step-copy { display:grid; gap:3px; min-width:0; }
  .workflow-step-title { font-weight:600; }
  .workflow-step-summary, .workflow-step-next { color:var(--muted); font-size:12px; line-height:1.35; }
  .workflow-step-status { align-self:start; color:var(--muted); font-size:12px; white-space:nowrap; }
  .workflow-step.active .workflow-step-status { color:var(--accent); font-weight:600; }
  .workflow-step.complete .workflow-step-status { color:var(--green); }
  .data-preparation-grid { display:grid; grid-template-columns:minmax(280px,.75fr) minmax(320px,1.25fr); gap:18px; }
  .data-preparation-grid > .group { align-content:start; }
  .candidate-manager, .training-configuration { border-color:#bfd7ef; }
  .candidate-tool-tabs { display:flex; gap:8px; flex-wrap:wrap; border-bottom:1px solid var(--line); padding-bottom:10px; }
  .candidate-tool-tab { background:#f5f5f7; border-color:#f0f0f0; color:var(--accent); }
  .candidate-tool-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .candidate-tool-panel { display:none; gap:14px; }
  .candidate-tool-panel.active { display:grid; }
  .panel.active { padding:4px 0 24px; }
  .panel.active > h3 { margin:8px 0 0; }
  .advanced-parameters {
    border-top:1px solid var(--line);
    border-bottom:1px solid var(--line);
    padding:10px 0;
  }
  .advanced-parameters > summary {
    color:var(--text);
    cursor:pointer;
    font-weight:600;
  }
  .advanced-parameters[open] > summary { margin-bottom:12px; }
  .advanced-parameters > .row { margin-top:10px; }
  .operation-log {
    display:grid;
    gap:5px;
    border-left:4px solid var(--accent);
  }
  .operation-log::before {
    content:"运行日志";
    color:inherit;
    font-size:12px;
    font-weight:700;
  }
  button:disabled, input:disabled, select:disabled, textarea:disabled {
    background:#f3f4f6;
    border-color:var(--line);
    color:#6b7280;
    opacity:1;
  }
  button:disabled { cursor:not-allowed; }
  .status-label {
    display:inline-block;
    margin-right:6px;
    padding:1px 6px;
    border:1px solid currentColor;
    border-radius:3px;
    font-size:11px;
    font-weight:600;
    line-height:1.45;
    white-space:nowrap;
  }
  .status-label.normal, .status-label.usable,
  .status-label.accepted, .status-label.used { color:var(--normal); }
  .status-label.attention, .status-label.review,
  .status-label.pending { color:#8a5a00; }
  .status-label.abnormal, .status-label.blocking,
  .status-label.rejected, .status-label.dropped { color:var(--danger); }
  .table-wrap tbody tr:hover { background:#f7fbff; }
  .table-wrap th:first-child, .table-wrap td:first-child { position:sticky; left:0; z-index:1; }
  .table-wrap th:first-child { background:var(--line-soft); }
  .table-wrap td:first-child { background:inherit; }
  .table-wrap td.numeric {
    text-align:right;
    font-variant-numeric:tabular-nums;
    white-space:nowrap;
  }
  @media (max-width:760px) {
    main { grid-template-columns:minmax(0,1fr); padding:12px; gap:12px; }
    .workflow-sidebar { position:static; padding:14px; }
    .workflow-steps { grid-template-columns:repeat(5,minmax(190px,1fr)); overflow-x:auto; }
    section { padding:18px; }
    .data-preparation-grid { grid-template-columns:minmax(0,1fr); }
    #engineeringPanel .detail-fields .row,
    .validation-box, .exploration-controls, .trend-controls,
    .condition-row { grid-template-columns:minmax(0,1fr); }
    .panel .actions > * { width:100%; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .metric strong { font-size:22px; }
    .table-wrap { max-width:100%; }
  }
</style>
"""


_WORKBENCH_UI_SCRIPT = r"""
<script id="workbenchUiScript">
document.addEventListener("DOMContentLoaded", () => {
  const results = document.querySelector(".results");
  const dataPanel = document.getElementById("configPanel");
  const candidatePanel = document.getElementById("candidatePanel");
  const modelPanel = document.getElementById("modelPanel");
  const validationPanel = document.getElementById("validationPanel");
  const candidateTable = document.getElementById("candidateWindows");
  const releasePanel = document.getElementById("releasePanel");
  const workflowSteps = document.getElementById("workflowSteps");
  const toolPanels = ["trendPanel", "stateExplorationPanel", "clusterPanel", "performancePanel"].map(id => document.getElementById(id));
  const showCandidateTool = target => {
    toolPanels.forEach(panel => panel.classList.toggle("active", target === "statePanels" ? ["clusterPanel", "performancePanel"].includes(panel.id) : panel.id === target));
    document.querySelectorAll(".candidate-tool-tab").forEach(button => {
      const selected = button.dataset.panel === target;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
  };
  document.querySelectorAll(".candidate-tool-tab").forEach(button => button.addEventListener("click", () => {
    globalThis.showWorkflowStage("candidatePanel");
    showCandidateTool(button.dataset.panel);
  }));
  showCandidateTool("trendPanel");
  const validatedDownload = document.getElementById("validatedModelDownload");

  globalThis.showWorkflowStage = target => {
    [dataPanel, candidatePanel, modelPanel, validationPanel, releasePanel].forEach(panel => panel.classList.toggle("active", panel.id === target));
    workflowSteps.querySelectorAll(".workflow-step").forEach(button => {
      const selected = button.dataset.panel === target;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.querySelector(".workflow-step-status").textContent = button.classList.contains("complete") ? "已完成" : selected ? "当前" : "待开始";
    });
  };
  workflowSteps.querySelectorAll(".workflow-step").forEach(button => button.addEventListener("click", () => globalThis.showWorkflowStage(button.dataset.panel)));
  globalThis.showWorkflowStage("configPanel");

  const refreshWorkflow = () => {
    const candidateDecisions = [...candidateTable.querySelectorAll('tbody select')];
    const completed = [
      Boolean(document.getElementById("candidateStart").value),
      candidateDecisions.some(select => select.value === "accepted"),
      !document.getElementById("modelContent").hidden,
      !document.getElementById("validationContent").hidden,
      !document.getElementById("deploymentModelDownload").hidden,
    ];
    const canRelease = !validatedDownload.hidden;
    document.getElementById("releaseEmpty").hidden = canRelease;
    document.getElementById("releaseContent").hidden = !canRelease;
    workflowSteps.querySelectorAll(".workflow-step").forEach((button, index) => {
      button.classList.toggle("complete", completed[index]);
      button.querySelector(".workflow-step-status").textContent = completed[index] ? "已完成" : button.classList.contains("active") ? "当前" : "待开始";
    });
  };
  refreshWorkflow();
  new MutationObserver(() => requestAnimationFrame(refreshWorkflow)).observe(results, { childList:true, subtree:true, attributes:true, attributeFilter:["hidden"] });

  document.querySelectorAll(".inner-tab").forEach(button => {
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(button.classList.contains("active")));
    button.addEventListener("click", () => {
      const peers = button.closest(".tabs, .inner-tabs")?.querySelectorAll(".tab, .inner-tab") || [];
      peers.forEach(peer => peer.setAttribute("aria-selected", String(peer === button)));
    });
  });

  const labels = {
    normal:"正常", usable:"可用", attention:"关注", review:"需确认",
    abnormal:"异常", blocking:"阻止", pending:"待决策", accepted:"已接受",
    rejected:"已拒绝", used:"已使用", dropped:"已丢弃",
  };
  const numericPattern = /^[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?:\s*\/\s*[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)?(?:\s*(?:%|分钟|点|样本|条))?$/i;
  const enhanceTables = () => document.querySelectorAll(".table-wrap td").forEach(cell => {
    const value = cell.textContent.trim();
    const key = Object.keys(labels).find(statusKey => value === statusKey);
    if (key && !cell.dataset.uiEnhanced) {
      cell.dataset.uiEnhanced = "true";
      const badge = document.createElement("span");
      badge.className = `status-label ${key}`;
      badge.textContent = labels[key];
      cell.replaceChildren(badge);
      return;
    }
    if (numericPattern.test(value)) cell.classList.add("numeric");
  });
  enhanceTables();
  new MutationObserver(enhanceTables).observe(document.body, { childList:true, subtree:true });
});
</script>
"""


def _required_html_match(pattern: str, html: str, description: str) -> re.Match[str]:
    match = re.search(pattern, html, re.DOTALL)
    if match is None:
        raise ValueError(f"无法固定Web工作台结构：{description}")
    return match


def _candidate_manager_html() -> str:
    return """      <div class="group candidate-manager">
        <div class="group-title">候选窗口</div>
        <div class="help">手工选择、趋势选择、聚类推荐和性能辅助统一进入此列表。候选默认待确认，不会自动参与训练。</div>
        <div class="row"><label>候选开始<input id="candidateStart" type="datetime-local"></label><label>候选结束<input id="candidateEnd" type="datetime-local"></label><label>备注<input id="candidateComment" type="text"></label><button id="addManualCandidate" class="secondary" type="button">加入候选窗口</button></div>
        <h3>候选窗口列表</h3><div id="candidateWindows" class="table-wrap"><div class="empty">检查数据后可管理候选窗口。</div></div>
        <div class="help">候选窗口不会修改训练窗口。先记录人工决策，再确认作为训练窗口。</div>
        <h3>排除窗口</h3><div id="excludedWindows" class="table-wrap"><div class="empty">尚无排除窗口。</div></div>
        <div class="help">排除窗口仅在确认候选时切分新的训练窗口，不会修改已生成的训练窗口。</div>
      </div>"""


def _workflow_sidebar_html() -> str:
    steps = (
        ("configPanel", "数据准备", "上传数据并完成 Tag 配置"),
        ("candidatePanel", "正常状态候选", "确认候选后生成训练窗口"),
        ("modelPanel", "模型训练", "质量检查后训练 DPCA"),
        ("validationPanel", "模型验证", "执行独立验证并记录结论"),
        ("releasePanel", "模型发布", "冻结并导出部署包"),
    )
    buttons = "\n".join(
        "        <button type=\"button\" class=\"workflow-step{}\" data-panel=\"{}\" role=\"tab\" aria-selected=\"{}\"><span class=\"workflow-step-number\">{}</span><span class=\"workflow-step-copy\"><span class=\"workflow-step-title\">{}</span><span class=\"workflow-step-next\">下一步：{}</span></span><span class=\"workflow-step-status\">{}</span></button>".format(
            " active" if index == 0 else "",
            panel,
            str(index == 0).lower(),
            index + 1,
            title,
            next_step,
            "当前" if index == 0 else "待开始",
        )
        for index, (panel, title, next_step) in enumerate(steps)
    )
    return f"""    <section class="controls workflow-sidebar" aria-label="建模流程">
      <h2 class="workflow-sidebar-title">建模流程</h2>
      <div id="workflowSteps" class="workflow-steps" role="tablist">
{buttons}
      </div>
    </section>"""


def _stabilize_workbench_html(html: str) -> str:
    """Return the five-stage workbench as static server-rendered HTML."""
    main_match = _required_html_match(
        r"  <main>\n(?P<content>.*?)\n  </main>", html, "main"
    )
    sections = _required_html_match(
        r'    <section class="controls">\n(?P<controls>.*?)\n    </section>\n'
        r'    <section class="results">\n(?P<results>.*?)\n    </section>',
        main_match.group("content"),
        "原始左右区域",
    )
    controls = sections.group("controls")
    results = sections.group("results")
    tag_marker = '      <div class="group">\n        <div class="group-title">2. 建模 Tag</div>'
    parameters_marker = '      <div class="group">\n        <div class="group-title">3. 参考状态与 DPCA 参数</div>'
    status_marker = '      <div id="status" class="status info"'
    upload_group, remaining_controls = controls.split(tag_marker, 1)
    tag_group, remaining_controls = remaining_controls.split(parameters_marker, 1)
    parameter_group, status_area = remaining_controls.split(status_marker, 1)
    upload_group = upload_group.rstrip().replace(
        '<div class="group-title">1. 历史数据</div>',
        '<div class="group-title">历史数据</div>',
        1,
    )
    tag_group = (tag_marker + tag_group).replace(
        '<div class="group-title">2. 建模 Tag</div>',
        '<div class="group-title">建模 Tag</div>',
        1,
    ).rstrip()
    parameter_group = (parameters_marker + parameter_group).rstrip()
    parameter_group = re.sub(
        r'        <div class="group-title">3\. 参考状态与 DPCA 参数</div>\n'
        r'        <div class="row"><label>候选开始.*?'
        r'        <div class="help">候选窗口不会修改训练窗口。先记录人工决策，再确认作为训练窗口。</div>\n'
        r'        <h3>排除窗口</h3><div id="excludedWindows" class="table-wrap"><div class="empty">尚无排除窗口。</div></div>\n'
        r'        <div class="help">排除窗口仅在确认候选时切分新的训练窗口，不会修改已生成的训练窗口。</div>\n',
        '        <div class="group-title">模型训练配置（参考状态与 DPCA 参数）</div>\n',
        parameter_group,
        count=1,
        flags=re.DOTALL,
    )
    parameter_group = parameter_group.replace(
        '<div class="group">', '<div class="group training-configuration">', 1
    )
    parameter_rows = (
        '        <div class="row"><label>目标采样周期（分钟）<input id="sampleInterval" type="number" min="1" value="5"></label><label>重采样方法<select id="resamplingMethod"><option value="none">不重采样</option><option value="mean">均值</option><option value="median">中位数</option><option value="last">最后值</option></select></label></div>\n'
        '        <div class="row"><label>滤波方法<select id="filterMethod"><option value="trailing_mean">尾随均值</option><option value="trailing_median">尾随中位数</option><option value="none">不滤波</option></select></label><label>滤波窗口（分钟）<input id="smoothingWindow" type="number" min="0" value="10"></label></div>\n'
        '        <div class="row"><label>物理缺口阈值（分钟，可选）<input id="gapThreshold" type="number" min="1" placeholder="沿用默认规则"></label><div><button id="preprocessingPreviewButton" class="secondary" disabled>预览预处理</button><div id="preprocessingPreview" class="muted">尚未预览</div></div></div>\n'
        '        <div class="row"><label>最大 Lag（分钟）<input id="maxLag" type="number" min="0" value="60"></label><label>Lag 步长（分钟）<input id="lagStep" type="number" min="1" value="5"></label></div>\n'
        '        <div class="row"><label>累计解释率<input id="varianceThreshold" type="number" min="0.01" max="0.99" step="0.01" value="0.95"></label><label>主元数（可留空）<input id="components" type="number" min="2" placeholder="自动，至少2个"></label></div>\n'
        '        <label>模型名称<input id="modelName" value="D330_DPCA_Model_V1"></label>'
    )
    layered_parameter_rows = (
        '        <div class="row"><label>目标采样周期（分钟）<input id="sampleInterval" type="number" min="1" value="5"></label></div>\n'
        '        <div class="row"><label>滤波方法<select id="filterMethod"><option value="trailing_mean">尾随均值</option><option value="trailing_median">尾随中位数</option><option value="none">不滤波</option></select></label><label>滤波窗口（分钟）<input id="smoothingWindow" type="number" min="0" value="10"></label></div>\n'
        '        <div class="row"><label>最大 Lag（分钟）<input id="maxLag" type="number" min="0" value="60"></label></div>\n'
        '        <div class="row"><label>累计解释率<input id="varianceThreshold" type="number" min="0.01" max="0.99" step="0.01" value="0.95"></label></div>\n'
        '        <label>模型名称<input id="modelName" value="D330_DPCA_Model_V1"></label>\n'
        '        <details class="advanced-parameters">\n'
        '          <summary>高级预处理与 DPCA 参数</summary>\n'
        '          <div class="help">执行顺序保持为时间检查、缺口识别、重采样、数据检查、因果滤波、Lag 扩展和标准化。</div>\n'
        '          <div class="row"><label>重采样方法<select id="resamplingMethod"><option value="none">不重采样</option><option value="mean">均值</option><option value="median">中位数</option><option value="last">最后值</option></select></label></div>\n'
        '          <div class="row"><label>物理缺口阈值（分钟，可选）<input id="gapThreshold" type="number" min="1" placeholder="沿用默认规则"></label><div><button id="preprocessingPreviewButton" class="secondary" disabled>预览预处理</button><div id="preprocessingPreview" class="muted">尚未预览</div></div></div>\n'
        '          <div class="row"><label>Lag 步长（分钟）<input id="lagStep" type="number" min="1" value="5"></label></div>\n'
        '          <div class="row"><label>主元数（可留空）<input id="components" type="number" min="2" placeholder="自动，至少2个"></label></div>\n'
        '        </details>'
    )
    if parameter_rows not in parameter_group:
        raise ValueError("无法固定训练参数字段")
    parameter_group = parameter_group.replace(parameter_rows, layered_parameter_rows, 1)
    if 'id="candidateWindows"' in parameter_group:
        raise ValueError("无法固定候选窗口或训练参数区域")
    status_area = (status_marker + status_area).replace(
        'class="status info" role="status" aria-live="polite"',
        'class="status info operation-log" role="status" aria-live="polite" aria-label="运行日志"',
        1,
    ).rstrip()

    panel_markers = (
        "configPanel",
        "stateExplorationPanel",
        "trendPanel",
        "modelPanel",
        "clusterPanel",
        "performancePanel",
        "validationPanel",
    )
    panel_positions = [
        results.index(f'      <div id="{panel_id}"') for panel_id in panel_markers
    ]
    config_panel = results[panel_positions[0] : panel_positions[1]].rstrip()
    state_panel = results[panel_positions[1] : panel_positions[2]].rstrip()
    trend_panel = results[panel_positions[2] : panel_positions[3]].rstrip()
    model_panel = results[panel_positions[3] : panel_positions[4]].rstrip()
    cluster_panel = results[panel_positions[4] : panel_positions[5]].rstrip()
    performance_panel = results[panel_positions[5] : panel_positions[6]].rstrip()
    validation_panel = results[panel_positions[6] :].rstrip()

    config_panel = config_panel.replace(
        '>\n',
        f'>\n      <div class="data-preparation-grid">\n{upload_group}\n{tag_group}\n      </div>\n',
        1,
    )
    candidate_panels = []
    for panel in (trend_panel, state_panel, cluster_panel, performance_panel):
        candidate_panels.append(panel.replace('class="panel"', 'class="candidate-tool-panel"', 1))
    candidate_panels[0] = candidate_panels[0].replace(
        'class="candidate-tool-panel"', 'class="candidate-tool-panel active"', 1
    )

    validated_download = _required_html_match(
        r'<a id="validatedModelDownload" class="download" href="#" hidden>下载已验证模型包</a>',
        validation_panel,
        "已验证模型下载入口",
    ).group(0)
    freeze_box = _required_html_match(
        r'          <div class="validation-box"><label>模型标识.*?</div>',
        validation_panel,
        "冻结与部署入口",
    ).group(0)
    validation_panel = validation_panel.replace(validated_download, "", 1).replace(
        freeze_box, "", 1
    )
    release_panel = f"""      <div id="releasePanel" class="panel">
        <div id="releaseEmpty" class="empty">模型通过独立验证和工程师确认后，可在此冻结并导出部署包。</div>
        <div id="releaseContent" hidden>
          <h3>模型发布</h3>
          <div class="notice">冻结与部署导出沿用现有流程；frozen 表示工程冻结，不表示已经部署。</div>
          <div class="actions">{validated_download}</div>
{freeze_box}
        </div>
      </div>"""
    candidate_panel = "\n".join(
        (
            '      <div id="candidatePanel" class="panel">',
            _candidate_manager_html(),
            '        <div class="candidate-tool-tabs" role="tablist">',
            '          <button type="button" class="candidate-tool-tab active" data-panel="trendPanel" role="tab" aria-selected="true">趋势选择</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="stateExplorationPanel" role="tab" aria-selected="false">状态探索 / 聚类推荐</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="statePanels" role="tab" aria-selected="false">聚类与性能辅助</button>',
            '        </div>',
            *candidate_panels,
            '      </div>',
        )
    )
    model_panel_lines = model_panel.rsplit("\n", 1)
    if len(model_panel_lines) != 2 or model_panel_lines[1] != "      </div>":
        raise ValueError("无法固定模型训练结果区域")
    model_panel = f"{model_panel_lines[0]}\n{parameter_group}\n      </div>"
    static_main = "\n".join(
        (
            "  <main>",
            _workflow_sidebar_html(),
            '    <section class="results">',
            status_area,
            config_panel,
            candidate_panel,
            model_panel,
            validation_panel,
            release_panel,
            "    </section>",
            "  </main>",
        )
    )
    return html[: main_match.start()] + static_main + html[main_match.end() :]


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
    result = _stabilize_workbench_html(result)
    result = result.replace(
        "</head>",
        f"{_FORM_ALIGNMENT_STYLE}\n{_APPLE_DESIGN_STYLE}\n{_WORKBENCH_UI_STYLE}\n</head>",
        1,
    )
    return result.replace(
        "</body>",
        f'<script src="/assets/model-results.js" defer></script>\n{_WORKBENCH_UI_SCRIPT}\n</body>',
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
