from __future__ import annotations

import argparse
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import shutil
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
_CANDIDATE_PROTECTED_ARTIFACTS = (
    "validated_model.pcamodel",
    "frozen_model.pcamodel",
    "deployment_model.pcadeploy",
)
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
    max-width:100%;
    min-width:0;
    display:grid;
    grid-template-columns:minmax(0,1fr);
    gap:var(--space-2);
  }
  #engineeringPanel .batch-config-title {
    font-size:14px;
    font-weight:600;
    line-height:1.4;
  }
  #engineeringPanel .batch-config .actions {
    display:grid;
    /* 中间列是文件选择框，宽度必须可收缩到 0：1100px 视口下侧栏占 260px 后
       内容区只剩 713px，而四个 max-content 按钮 + 48px gap 固定占 509px，
       任何正的 minmax 下限都会让容器撑破并造成整页横向滚动。 */
    grid-template-columns:max-content minmax(0,1fr) max-content max-content max-content;
    gap:var(--space-2);
    align-items:end;
  }
  #engineeringPanel .batch-config .actions > .download,
  #engineeringPanel .batch-config .actions > button {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:var(--control-height);
    height:var(--control-height);
    white-space:nowrap;
  }
  #engineeringPanel .batch-config .actions > .download {
    min-width:var(--batch-action-width);
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
    min-height:var(--control-height);
    height:var(--control-height);
    padding:0 10px;
  }
  #engineeringPanel .batch-config #importSummary { margin-top:var(--space-2); }

  #clusterPanel .group { gap:var(--space-2); }
  #clusterPanel .row {
    column-gap:var(--space-2);
    align-items:start;
  }
  #clusterPanel .row > label {
    min-width:0;
    align-content:start;
  }
  #clusterPanel .row input {
    min-height:var(--control-height);
    height:var(--control-height);
  }
  #clusterPanel #clusterButton {
    align-self:end;
    min-height:var(--control-height);
    height:var(--control-height);
  }

  #engineeringPanel .detail-fields {
    display:grid;
    max-width:100%;
    min-width:0;
    gap:var(--space-2);
  }
  #engineeringPanel .detail-fields .row {
    column-gap:var(--space-2);
    align-items:start;
  }
  #engineeringPanel .detail-fields .row > label {
    min-width:0;
    align-self:start;
    align-content:start;
  }
  #engineeringPanel .detail-fields input,
  #engineeringPanel .detail-fields select {
    min-height:var(--control-height);
    height:var(--control-height);
  }
  #engineeringPanel #tagComment {
    display:block;
    height:84px;
    min-height:84px;
    resize:vertical;
  }
  #engineeringPanel #tagRole { align-self:start; }
  #engineeringPanel #saveTagConfig {
    min-height:var(--control-height);
    margin-top:var(--space-1);
    /* 与批量配置区的下载按钮等宽：两者共用 min-width，文本变长也不会让下方按钮超出上方宽度。 */
    min-width:var(--batch-action-width);
    justify-self:start;
  }

  /* 断点与工作台主布局（1050px）对齐：主布局塌成单列后，批量配置也必须同时收敛，
     否则 900~1050px 之间会同时出现“侧栏横条 + 批量配置两列硬切”的混合密度。
     窄屏改回 intrinsic-width 横向优先 + 自然换行：2 列网格会把三个固定宽按钮
     硬塞进两格，首行只剩一个 119px 下载按钮、右列空置，且行高 63/42 混排。 */
  @media (max-width:1050px) {
    #engineeringPanel .batch-config .actions {
      display:flex;
      flex-wrap:wrap;
      align-items:flex-end;
      gap:var(--space-2);
    }
    #engineeringPanel .batch-config .actions > * {
      flex:0 0 auto;
      width:auto;
    }
    #engineeringPanel .batch-config .actions > label.secondary { flex:1 1 240px; }
  }
  @media (max-width:760px) {
    /* 最窄一档让文件选择控件独占一行。 */
    #engineeringPanel .batch-config .actions > label.secondary { flex:1 1 100%; }
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
    --control-height:42px;
    --control-radius:6px;
    /* 批量配置区下载按钮的实测自然宽度，下方“保存当前”对齐到同宽。 */
    --batch-action-width:119px;
    --space-1:8px;
    --space-2:12px;
    --space-3:16px;
    --space-4:24px;
    --panel-padding:24px;
  }

  *, *::before, *::after { box-shadow:none !important; }
  body {
    background:var(--bg);
    color:var(--text);
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
    font-size:16px;
    font-weight:400;
    letter-spacing:0;
    line-height:1.5;
  }
  header { padding:16px clamp(20px,3vw,40px); background:#000000; border-bottom:0; color:#ffffff; }
  h1 { margin:0 0 4px; font-size:21px; font-weight:600; line-height:1.19; letter-spacing:.231px; }
  /* Application UI 里的 h2 与 h3 同级：34px 是营销页标题尺寸，会与工作台内
     25 个 21px 小节标题产生不必要的量级跳跃，破坏层级扫描。 */
  h2 { font-size:21px; font-weight:600; line-height:1.19; letter-spacing:.231px; }
  h3 { font-size:21px; font-weight:600; line-height:1.19; letter-spacing:.231px; }
  h4 { font-size:17px; font-weight:600; line-height:1.24; letter-spacing:0; }
  .subtitle, .help, label, .legend, .dp-legend { color:var(--muted); }
  .subtitle { color:#cccccc; font-size:14px; line-height:1.43; }
  .help, label, .legend, .dp-legend { font-size:14px; line-height:1.43; }
  h2, h3, h4 { margin:0; }
  main {
    width:100%;
    max-width:1800px;
    margin:0 auto;
    gap:var(--space-4);
    padding:var(--space-4) clamp(16px,2.5vw,32px);
    align-items:start;
  }
  section { min-width:0; border:0; border-radius:0; padding:var(--panel-padding); background:var(--panel); }
  main > section, .results > *, .panel, .inner-panel, .group, .chart-card { min-width:0; }
  .controls, .controls .group { min-width:0; }
  .controls { gap:var(--space-3); }
  .results { gap:24px; }
  .row { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:var(--space-2); align-items:end; }
  .actions, .tag-toolbar, .detail-fields { gap:var(--space-2); }
  .actions { display:flex; flex-wrap:wrap; align-items:center; justify-content:flex-start; }
  .inner-panel.active, .panel.active { gap:24px; }
  .group, .metric, .validation-box, .exploration-controls, .dp-inline-help {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:6px;
  }
  .group { gap:var(--space-2); padding:var(--panel-padding); }
  .group-title, .sub-title { margin:0; font-size:14px; font-weight:600; line-height:1.4; }
  .sub-title { border-top-color:var(--line); }
  input, select, textarea {
    width:100%;
    height:var(--control-height);
    min-height:var(--control-height);
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:var(--control-radius);
    color:var(--text);
    font:400 14px/20px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
    padding:0 12px;
  }
  textarea { height:auto; min-height:84px; padding:10px 12px; line-height:1.45; resize:vertical; }
  input[type=range] { height:24px; min-height:24px; padding:0; }
  /* Checkbox 默认继承 42px 控件高度会撑破表格行，统一收敛到可点击的 16px 方块。 */
  input[type=checkbox] {
    width:16px;
    height:16px;
    min-height:16px;
    max-height:16px;
    padding:0;
    margin:0;
    border:1px solid var(--line);
    border-radius:3px;
    accent-color:var(--accent);
    vertical-align:middle;
    cursor:pointer;
  }
  input[type=checkbox]:disabled { cursor:not-allowed; }
  select[multiple] { height:auto; min-height:132px; padding:8px 10px; }
  input[type=file] { padding:0 10px; }
  input:focus, select:focus, textarea:focus {
    outline:2px solid var(--accent);
    outline-offset:2px;
    border-color:var(--accent);
  }
  button, .download {
    box-sizing:border-box;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    vertical-align:middle;
    height:var(--control-height);
    min-height:var(--control-height);
    max-width:100%;
    border:1px solid var(--accent);
    background:var(--accent);
    color:#ffffff;
    border-radius:var(--control-radius);
    font:400 14px/20px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Arial,sans-serif;
    letter-spacing:0;
    padding:0 14px;
    text-decoration:none;
    cursor:pointer;
  }
  button:not(:disabled):hover, .download:hover { filter:brightness(.96); }
  button:not(:disabled):active, .download:active { filter:brightness(.9); }
  button.secondary {
    background:var(--bg);
    border-color:var(--line-soft);
    color:var(--accent);
    border-radius:var(--control-radius);
  }
  button.danger { background:var(--danger); border-color:var(--danger); color:#fff; }
  #inspectButton, #qualityButton, #saveTagConfig { background:var(--accent); border-color:var(--accent); color:#ffffff; }
  .tag-toolbar button { min-height:var(--control-height); padding:0 14px; font-size:14px; }
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
  #tagOptions .tag-row.pending {
    height:auto;
    min-height:30px;
    grid-template-columns:minmax(0,1fr) max-content;
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
    height:var(--control-height);
    min-height:var(--control-height);
  }
  #engineeringPanel #tagComment { height:84px; min-height:84px; }
  button:focus-visible, .download:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .tabs, .inner-tabs {
    min-width:0;
    flex-wrap:wrap;
    overflow-x:auto;
    gap:0;
    border-bottom:1px solid var(--line);
    padding-bottom:0;
    background:var(--bg);
  }
  .tab, .inner-tab {
    display:inline-flex;
    align-items:center;
    flex:0 0 auto;
    min-height:var(--control-height);
    height:var(--control-height);
    border:0;
    border-bottom:2px solid transparent;
    border-radius:0;
    background:transparent;
    color:var(--text);
    font-size:14px;
    letter-spacing:0;
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
  #modelPanel #qualityButton {
    width:fit-content;
    height:42px;
    min-height:42px;
    justify-self:start;
  }
  #modelPanel #modelQualityStatus {
    display:inline-flex;
    align-items:center;
    max-width:100%;
  }
  /* 质量状态与按钮并排成一行，仅作文字提示，不再占用卡片色块。 */
  #modelPanel #modelQualityStatus {
    width:auto;
    min-width:72px;
    height:auto;
    min-height:42px;
    background:#ffffff;
    border:0;
    box-shadow:none;
  }
  /* 按钮与状态块由父容器排布：宽度随内容自适应，空间不足才换行，不再用负 margin 硬挤到同一行。 */
  #modelPanel .quality-action-row {
    display:flex;
    align-items:center;
    flex-wrap:wrap;
    gap:var(--space-2);
    min-width:0;
  }
  #modelPanel .quality-action-row > * { min-width:0; max-width:100%; }
  /* flex-basis auto：短文案与按钮同排；长错误文案占满剩余空间后自动换到下一行，
     宽度不再由视口决定（fit-content 会把长文案拉到 1354px 满宽，可读性差）。 */
  #modelPanel .quality-action-row > #modelQualityStatus {
    flex:1 1 auto;
    width:auto;
  }
  #modelPanel #currentTagQuality { max-width:1200px; }
  .issue-card, .notice { border-left-width:4px; border-radius:6px; }
  .tag-row.selected { background:#edf5ff; }
  .metric { padding:24px; }
  .metrics { grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:8px; }
  .metric { min-width:0; padding:10px 11px; }
  .metric strong { font-size:22px; font-weight:600; line-height:1.15; letter-spacing:0; }
  .metric span { font-size:12px; line-height:1.35; }
  .chart-card { gap:12px; }
  .chart-card h3 { margin:0; }
  .chart, .dp-chart, .trend-chart { background:var(--panel); }
  .empty { border-style:dashed; color:var(--muted); }
  .variance-bar { background:var(--accent); }
  .variance-bar.selected { background:var(--green); }
  .table-wrap, .exploration-timeline { border:1px solid var(--line); }
  th, td { border-bottom-color:var(--line); padding:8px 12px; }
  th { background:var(--line-soft); color:var(--text); font-weight:600; }
  a:not(.download) { color:var(--accent); }
  #engineeringPanel #tagComment { resize:vertical; }
  button:focus-visible, .download:focus-visible, a:focus-visible {
    outline:2px solid var(--accent);
    outline-offset:2px;
  }
  @media (max-width:640px) {
    header { padding:14px 16px; }
    h1 { font-size:21px; }
    main { padding:12px; }
    section { padding:var(--panel-padding); }
    h2 { font-size:21px; }
    button, .download, input, select, textarea { height:var(--control-height); min-height:var(--control-height); }
    textarea, select[multiple] { height:auto; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
</style>
"""


_WORKBENCH_UI_STYLE = r"""
<style id="workbenchUiStyle">
  /* 侧栏宽度是「步骤正文列宽」的上限约束：step 三列 30px + 74px 正文 + 24px 状态
     在 218px 时正文只剩 74px，「正常状态候选」折 2 行、「下一步：仅已验证模型可冻结
     并导出部署包」折 4 行。收到 260px 后正文列 116px，标题基本单行。 */
  main { grid-template-columns:260px minmax(0,1fr); align-items:start; }
  /* sticky 侧栏必须自己限高：5 个步骤卡合计约 592px，视口高度低于该值时
     侧栏会被裁在视口外且随页面滚走，底部步骤（冻结与部署）无法访问。 */
  .workflow-sidebar {
    position:sticky;
    top:var(--space-3);
    display:grid;
    gap:var(--space-2);
    padding:var(--panel-padding);
    max-height:calc(100vh - var(--space-4));
    overflow-y:auto;
    overscroll-behavior-y:contain;
  }
  .workflow-sidebar-title { margin:0; font-size:17px; font-weight:600; }
  .workflow-steps { display:grid; gap:var(--space-1); }
  .workflow-step {
    display:grid;
    grid-template-columns:30px minmax(0,1fr) auto;
    gap:8px;
    width:100%;
    height:auto;
    min-height:76px;
    padding:12px;
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
  .data-preparation-grid { display:grid; grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr); gap:var(--space-3); align-items:stretch; }
  .data-preparation-grid > .group { align-content:start; }
  .candidate-manager, .shared-preprocessing, .training-configuration { border-color:#bfd7ef; }
  .candidate-manager .row { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .candidate-manager .row > * { min-width:0; }
  /* 候选时间/备注/按钮四项同排；时间控件需要更多宽度，备注与按钮更窄。 */
  .candidate-manager .candidate-window-row {
    grid-template-columns:minmax(0,1.5fr) minmax(0,1.5fr) minmax(0,1.4fr) max-content;
  }
  .candidate-manager .candidate-window-row > label { min-width:0; }
  .candidate-tool-tabs { display:flex; gap:var(--space-1); flex-wrap:wrap; align-items:center; border-bottom:1px solid var(--line); padding-bottom:var(--space-2); }
  .candidate-tool-tab { background:#f5f5f7; border-color:#f0f0f0; color:var(--accent); }
  .candidate-tool-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .candidate-tool-panel { display:none; gap:var(--space-3); }
  .candidate-tool-panel.active { display:grid; }
  .advanced-candidate-tools { border-top:1px solid var(--line); padding-top:var(--space-2); }
  .advanced-candidate-tools > summary { color:var(--muted); cursor:pointer; font-size:13px; font-weight:600; }
  .advanced-candidate-tools .candidate-tool-tabs { margin-top:var(--space-2); }
  .shared-preprocessing, .training-configuration { display:grid; gap:var(--space-2); }
  .panel.active { padding:var(--space-1) 0 var(--space-4); }
  .panel.active > h3 { margin:var(--space-1) 0 0; }
  .advanced-parameters {
    border-top:1px solid var(--line);
    border-bottom:1px solid var(--line);
    padding:var(--space-2) 0;
  }
  .advanced-parameters > summary {
    color:var(--text);
    cursor:pointer;
    font-weight:600;
  }
  .advanced-parameters[open] > summary { margin-bottom:var(--space-2); }
  .advanced-parameters > .row { margin-top:var(--space-2); }
  .training-parameter-grid {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:var(--space-2);
  }
  .training-parameter-grid > label,
  .filter-parameter-control,
  .model-name-field { min-width:0; }
  .filter-parameter-control > label { min-width:0; }
  .model-name-field { grid-column:span 2; }
  /* 共享预处理五项同排；列宽按内容收敛并左对齐，避免等分后每个输入框过宽。 */
  .shared-preprocessing .preprocessing-parameter-row {
    grid-template-columns:repeat(5,minmax(0,1fr));
  }
  /* 状态过滤块：最大 Lag、Lag 步长与“添加状态过滤条件”同排，按内容收敛列宽并贴底对齐。 */
  .shared-preprocessing .state-filter-parameter-row {
    grid-template-columns:repeat(3,max-content);
    justify-content:start;
  }
  .shared-preprocessing .state-filter-parameter-row > * { min-width:0; }
  .preprocessing-preview-controls {
    display:grid;
    grid-template-columns:minmax(0,1fr) max-content;
    gap:var(--space-2);
    align-items:end;
    margin-top:var(--space-2);
  }
  .preprocessing-preview-controls > label { min-width:0; }
  .preprocessing-preview-area,
  .preprocessing-preview-area #preprocessingPreview { width:100%; min-width:0; }
  .preprocessing-preview-area #preprocessingPreview { margin-top:var(--space-2); }
  .preprocessing-preview-area #preprocessingPreviewTagSelect {
    width:300px;
    min-width:0;
    max-width:100%;
  }
  .operation-log {
    display:grid;
    gap:var(--space-1);
    border-left:4px solid var(--accent);
  }
  /* 运行日志是全局唯一的操作反馈入口，但位置固定在结果区顶部：进入模型训练等
     后置阶段后，错误提示与触发它的控件相距近 800px，必须滚回顶部才看得到。
     保持 DOM 位置不变（仍由 aria-live 播报），改由脚本在阶段切换时把日志
     滚入视野：sticky 会遮挡滚动路径上的小节标题，不是合适的方案。 */
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
  /* 固定首列必须自带不透明底色：tr/table 背景是透明的，继承过来仍是透明，
     横向滚动时右邻列文字会直接透过 sticky 列叠印在一起。 */
  .table-wrap td:first-child {
    background:var(--panel);
    box-shadow:1px 0 0 var(--line);
  }
  .table-wrap tbody tr:hover td:first-child { background:#f7fbff; }
  .table-wrap td.numeric {
    text-align:right;
    font-variant-numeric:tabular-nums;
    white-space:nowrap;
  }
  #stateExplorationPanel .exploration-controls { border:0; }
  #explorationRegionSummary td:nth-child(2), #explorationRegionSummary td:nth-child(3) { text-align:center; }
  .validation-box, .exploration-controls { align-items:end; }
  .validation-box > button, .exploration-controls > button, .validation-box > .download, .exploration-controls > .download { align-self:end; }
  /* 验证/探索表单里的按钮是 Command，不是 Choice Option：默认按内容取宽，
     不该被 minmax(130px,1fr) 拉成整列宽，否则与相邻输入框同宽、失去按钮识别度。 */
  .validation-box > button, .validation-box > .download {
    justify-self:start;
    width:auto;
  }
  .exploration-controls > button, .exploration-controls > .download {
    justify-self:start;
    width:auto;
  }
  /* 标签换行时输入框仍需贴底对齐，否则同一行控件高低不齐。 */
  .exploration-controls > label { display:grid; align-content:start; }
  .candidate-tool-tabs .candidate-tool-tab { height:var(--control-height); }
  .panel > .actions, .inner-panel > .actions { margin-top:var(--space-1); }
  .issue-card, .notice, .status, .metric, .chart-card, .table-wrap, .empty, .validation-box, .exploration-controls { min-width:0; }
  .issue-card, .notice { padding:var(--space-2); }
  .metric { min-height:64px; align-content:start; }
  .chart-card { gap:var(--space-2); }
  .chart-grid { align-items:stretch; }
  .table-wrap { width:100%; }
  /* 优选运行区域统计按固定 540px 列宽排布，不随右侧列拉伸。 */
  .region-stats { width:540px; max-width:100%; }
  /* 表头与短状态值不逐字换行；窄表由 .table-wrap 横向滚动承接。 */
  .table-wrap th { white-space:nowrap; }
  /* 列宽按内容取值，避免日期被压成多行；宽度不足时由 .table-wrap 横向滚动。 */
  .table-wrap table { width:max-content; min-width:100%; }
  .table-wrap td:has(> .status-label),
  .table-wrap td:has(> input[type=checkbox]) { white-space:nowrap; text-align:center; }
  tr:has(> td > input[type=checkbox]) > td:nth-child(2) { white-space:nowrap; }
  /* 状态探索结果表统一居中：表头、单元格（含复选框、状态、数值列）都在单元格内水平居中；仅限本面板，不影响其他阶段表格。 */
  #stateExplorationPanel table th, #stateExplorationPanel table td { text-align:center; }
  /* 候选表内的勾选与备注控件按单元格密度收敛，不继承 42px 全高控件。 */
  .exploration-candidate-select { display:inline-block; }
  .exploration-candidate-comment {
    width:100%;
    min-width:120px;
    height:30px;
    min-height:30px;
    padding:0 8px;
    font-size:12px;
    text-align:center;
  }
  @media (max-width:1050px) {
    main { grid-template-columns:minmax(0,1fr); }
    .workflow-sidebar { position:static; }
    /* 单列布局下侧栏横条是导航，不该靠横向滚动藏步骤：5×190px 的硬下限在 900px
       视口装不下（容器 855px），第 5 步「冻结与部署」被推到滚动区外。
       改用 auto-fit 换行，步骤数与当前位置始终完整可见。 */
    .workflow-steps { grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }
  }
  @media (max-width:760px) {
    main { grid-template-columns:minmax(0,1fr); padding:var(--space-2); gap:var(--space-2); }
    .workflow-sidebar { position:static; padding:var(--space-3); }
    .workflow-steps { grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }
    section { padding:var(--panel-padding); }
    .data-preparation-grid { grid-template-columns:minmax(0,1fr); }
    .row { grid-template-columns:minmax(0,1fr); }
    .shared-preprocessing .preprocessing-parameter-row { grid-template-columns:minmax(0,1fr); }
    .shared-preprocessing .state-filter-parameter-row { grid-template-columns:minmax(0,1fr); }
    .candidate-manager .row { grid-template-columns:minmax(0,1fr); }
    .candidate-manager .candidate-window-row { grid-template-columns:minmax(0,1fr); }
    .candidate-manager .row > button { width:100%; }
    .candidate-tool-tabs { flex-wrap:nowrap; overflow-x:auto; }
    .candidate-tool-tabs > * { flex:0 0 auto; width:auto; }
    #engineeringPanel .detail-fields .row,
    .validation-box, .exploration-controls, .trend-controls,
    .condition-row { grid-template-columns:minmax(0,1fr); }
    .actions { align-items:stretch; }
    .panel .actions > *, .inner-panel .actions > * { width:100%; }
    .tabs > *, .inner-tabs > * { flex:1 1 0; width:auto; min-width:0; }
    .validation-box > *, .exploration-controls > * { width:100%; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .metric strong { font-size:20px; }
    .table-wrap { max-width:100%; }
  }
  @media (max-width:1100px) {
    .training-parameter-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
  @media (max-width:520px) {
    .training-parameter-grid { grid-template-columns:minmax(0,1fr); }
    .metrics { grid-template-columns:minmax(0,1fr); }
    .model-name-field { grid-column:auto; }
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
  const trainingTable = document.getElementById("trainingWindows");
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
    button.closest("details")?.setAttribute("open", "");
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
    // 运行日志固定在结果区顶部，进入后置阶段后它与操作点相距近 800px，
    // 错误提示会看不见。切阶段时把日志滚进视野；sticky 会遮挡滚动路径上的
    // 小节标题，不是合适的方案。只在日志完全不在视野内时才滚动。
    const log = document.getElementById("status");
    if (log) {
      const logRect = log.getBoundingClientRect();
      if (logRect.bottom < 0 || logRect.top > innerHeight) log.scrollIntoView({ block: "start" });
    }
  };
  workflowSteps.querySelectorAll(".workflow-step").forEach(button => button.addEventListener("click", () => globalThis.showWorkflowStage(button.dataset.panel)));
  globalThis.showWorkflowStage("configPanel");

  const refreshWorkflow = () => {
    const completed = [
      Boolean(document.getElementById("candidateStart").value),
      Boolean(trainingTable.querySelector("tbody tr")),
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

  // tablist 的键盘契约：左右方向键在同组 tab 间移动焦点（并切换选中）。
  // 此前三组 tablist 只能靠 Tab 逐个跳，键盘用户无法快速切换阶段。
  const wireTablistKeys = (tablist) => {
    if (!tablist) return;
    tablist.addEventListener("keydown", (event) => {
      const tabs = [...tablist.querySelectorAll('[role="tab"]')];
      if (!tabs.length) return;
      const index = tabs.indexOf(document.activeElement);
      if (index < 0) return;
      const step = event.key === "ArrowRight" || event.key === "ArrowDown"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowUp"
        ? -1
        : 0;
      if (!step) return;
      event.preventDefault();
      tabs[(index + step + tabs.length) % tabs.length].focus();
    });
  };
  document.querySelectorAll('[role="tablist"]').forEach(wireTablistKeys);

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


_MODEL_RESULTS_STYLE = r"""
<style id="modelResultsStyle">
  .model-projection-grid {
    display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:14px;
    align-items:stretch;
    margin-bottom:14px;
  }
  .model-projection-grid > .chart-card {
    min-width:0;
    margin:0;
    display:flex;
    flex-direction:column;
  }
  .model-projection-grid > .chart-card > .chart { flex:1 1 auto; min-height:420px; }
  .model-projection-grid #scoreChart svg,
  .model-projection-grid #loadingChart svg { width:100%; height:420px; display:block; }
  #componentLoadings .component-loading-list { display:grid; gap:12px; }
  #componentLoadings .component-loading-item { border:1px solid var(--line); border-radius:6px; padding:12px; }
  #componentLoadings .component-loading-item h4 { margin:0 0 8px; }
  #componentLoadings .component-loading-table { width:100%; table-layout:fixed; }
  #componentLoadings .component-loading-table th:first-child,
  #componentLoadings .component-loading-table td:first-child { text-align:left; overflow-wrap:anywhere; }
  #componentLoadings .component-loading-table th:nth-child(2),
  #componentLoadings .component-loading-table td:nth-child(2) { width:10em; text-align:right; white-space:nowrap; }
  #modelStructureComparison .model-variance-chart svg { display:block; width:100%; max-width:640px; height:auto; }
  #modelStructureComparison .model-variance-summary { display:flex; flex-wrap:wrap; gap:4px 14px; margin:0 0 10px; }
  #modelStructureComparison .model-energy-grid {
    display:grid;
    grid-template-columns:minmax(0,1.35fr) minmax(220px,.65fr);
    gap:12px 16px;
    align-items:start;
    max-width:900px;
  }
  #modelStructureComparison .model-energy-table { width:100%; max-width:100%; min-width:0; }
  #modelStructureComparison .model-energy-table table { width:100%; table-layout:fixed; }
  #modelStructureComparison .model-energy-table th:first-child,
  #modelStructureComparison .model-energy-table td:first-child { text-align:left; overflow-wrap:anywhere; }
  #modelStructureComparison .model-energy-table th:nth-child(2),
  #modelStructureComparison .model-energy-table td:nth-child(2) { width:7.5em; text-align:right; white-space:nowrap; }
  #modelStructureComparison .model-energy-table th,
  #modelStructureComparison .model-energy-table td { padding:5px 8px; }
  #modelStructureComparison .lag-energy-table { width:min(100%,220px); }
  #modelStructureComparison .lag-energy-table th:first-child,
  #modelStructureComparison .lag-energy-table td:first-child { text-align:right; white-space:nowrap; }
  #modelStructureComparison #modelComparisonRuns { height:auto; min-height:0; }
  #modelStructureComparison button.danger { background:var(--danger); border-color:var(--danger); color:#fff; }
  #modelStructureComparison .model-parameter-table { width:100%; max-width:100%; table-layout:fixed; }
  #modelStructureComparison .model-parameter-table th:first-child,
  #modelStructureComparison .model-parameter-table td:first-child { width:12em; }
  #modelStructureComparison .model-parameter-table th,
  #modelStructureComparison .model-parameter-table td { white-space:normal; overflow-wrap:anywhere; word-break:break-word; }
  #modelStructureComparison .model-parameter-table td.numeric,
  #modelStructureComparison .model-energy-table td.numeric { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  @media (max-width:1200px) {
    .model-projection-grid { grid-template-columns:1fr; }
    .model-projection-grid > .chart-card > .chart { min-height:360px; }
    .model-projection-grid #scoreChart svg,
    .model-projection-grid #loadingChart svg { height:360px; }
  }
  @media (max-width:760px) {
    #modelStructureComparison .model-energy-grid { grid-template-columns:1fr; }
    #modelStructureComparison .lag-energy-table { width:100%; }
  }
</style>
"""


def _required_html_match(pattern: str, html: str, description: str) -> re.Match[str]:
    match = re.search(pattern, html, re.DOTALL)
    if match is None:
        raise ValueError(f"无法固定Web工作台结构：{description}")
    return match


def _split_at_unique_anchor(html: str, anchor: str, description: str) -> tuple[str, str]:
    """Split at one stable HTML anchor and fail clearly if the base page changes."""
    count = html.count(anchor)
    if count != 1:
        raise ValueError(f"无法固定Web工作台结构：{description}锚点数量为{count}")
    return html.split(anchor, 1)


def _unique_anchor_index(html: str, anchor: str, description: str) -> int:
    count = html.count(anchor)
    if count != 1:
        raise ValueError(f"无法固定Web工作台结构：{description}锚点数量为{count}")
    return html.index(anchor)


def _label_for_unique_field(html: str, field_id: str, description: str) -> str:
    field_index = _unique_anchor_index(html, f'id="{field_id}"', description)
    start = html.rfind("<label", 0, field_index)
    end = html.find("</label>", field_index)
    if start == -1 or end == -1:
        raise ValueError(f"无法固定Web工作台结构：{description}标签边界")
    return html[start : end + len("</label>")]


def _element_with_unique_id(html: str, field_id: str, tag: str, description: str) -> str:
    field_index = _unique_anchor_index(html, f'id="{field_id}"', description)
    start = html.rfind(f"<{tag}", 0, field_index)
    end = html.find(f"</{tag}>", field_index)
    if start == -1 or end == -1:
        raise ValueError(f"无法固定Web工作台结构：{description}元素边界")
    return html[start : end + len(tag) + 3]


def _div_containing_unique_field(html: str, field_id: str, description: str) -> str:
    field_index = _unique_anchor_index(html, f'id="{field_id}"', description)
    start = html.rfind("<div", 0, field_index)
    if start == -1:
        raise ValueError(f"无法固定Web工作台结构：{description}容器开始")
    depth = 0
    for match in re.finditer(r"<div\b[^>]*>|</div>", html[start:]):
        depth += 1 if match.group().startswith("<div") else -1
        if depth == 0:
            end = start + match.end()
            if field_index >= end:
                break
            return html[start:end]
    raise ValueError(f"无法固定Web工作台结构：{description}容器结束")


def _row_containing_unique_field(html: str, field_id: str, description: str) -> tuple[int, int]:
    field_index = _unique_anchor_index(html, f'id="{field_id}"', description)
    divs = list(re.finditer(r"<div\b[^>]*>", html[: field_index + 1]))
    for match in reversed(divs):
        if not re.search(r'\bclass\s*=\s*["\'][^"\']*\brow\b', match.group()):
            continue
        depth = 0
        for token in re.finditer(r"<div\b[^>]*>|</div>", html[match.start() :]):
            depth += 1 if token.group().startswith("<div") else -1
            if depth == 0:
                end = match.start() + token.end()
                if field_index < end:
                    return match.start(), end
                break
    raise ValueError(f"无法固定Web工作台结构：{description}参数行")


def _candidate_manager_html(training_data_section: str) -> str:
    return """      <div class="group candidate-manager">
        <div class="group-title">正常状态候选窗口</div>
        <div class="help">手工选择、趋势选择和状态探索候选统一进入此列表。候选默认待确认，不会自动参与训练。</div>
        <div class="row candidate-window-row"><label>候选开始<input id="candidateStart" type="datetime-local"></label><label>候选结束<input id="candidateEnd" type="datetime-local"></label><label>备注<input id="candidateComment" type="text"></label><button id="addManualCandidate" class="secondary" type="button">加入候选窗口</button></div>
        <h3>候选窗口列表</h3><div id="candidateWindows" class="table-wrap"><div class="empty">检查数据后可管理候选窗口。</div></div>
        <div class="help">候选窗口不会修改训练窗口；确认作为训练窗口后才会生成训练窗口。</div>
        <h3>排除窗口</h3><div id="excludedWindows" class="table-wrap"><div class="empty">尚无排除窗口。</div></div>
        <div class="help">排除窗口仅在确认候选时切分新的训练窗口，不会修改已生成的训练窗口。</div>
%s      </div>""" % training_data_section


def _workflow_sidebar_html() -> str:
    steps = (
        ("configPanel", "数据与Tag", "上传数据并完成 Tag 配置"),
        ("candidatePanel", "正常状态候选", "确认候选后生成训练窗口"),
        ("modelPanel", "模型训练", "质量检查后训练 DPCA"),
        ("validationPanel", "模型验证", "独立验证窗口并记录工程师结论"),
        ("releasePanel", "冻结与部署", "仅已验证模型可冻结并导出部署包"),
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
    upload_group, remaining_controls = _split_at_unique_anchor(
        controls, tag_marker, "建模Tag"
    )
    tag_group, remaining_controls = _split_at_unique_anchor(
        remaining_controls, parameters_marker, "训练参数"
    )
    parameter_group, status_area = _split_at_unique_anchor(
        remaining_controls, status_marker, "运行日志"
    )
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
    candidate_anchor = '        <div class="row"><label>候选开始<input id="candidateStart"'
    training_windows_anchor = '        <h3>训练窗口</h3><div id="trainingWindows"'
    parameter_prefix, candidate_section = _split_at_unique_anchor(
        parameter_group, candidate_anchor, "候选窗口"
    )
    _, training_section = _split_at_unique_anchor(
        candidate_section, training_windows_anchor, "训练窗口"
    )
    training_section = training_windows_anchor + training_section
    sample_interval_anchor = _row_containing_unique_field(
        training_section, "sampleInterval", "共享预处理参数"
    )[0]
    model_configuration_anchor = _row_containing_unique_field(
        training_section, "varianceThreshold", "PCA 模型配置"
    )[0]
    training_data_section = training_section[:sample_interval_anchor]
    shared_preprocessing_section = training_section[
        sample_interval_anchor:model_configuration_anchor
    ]
    shared_preprocessing_section = "        " + shared_preprocessing_section.lstrip()
    training_data_section = training_data_section.rstrip() + "\n"
    training_parameter_tail = training_section[model_configuration_anchor:]
    shared_preprocessing_section = shared_preprocessing_section.replace(
        '<div class="sub-title">状态过滤条件</div>',
        '<div class="sub-title">状态过滤条件（只选择角色为“状态过滤”的 Tag）</div>',
        1,
    )
    preview_window_label = _label_for_unique_field(
        shared_preprocessing_section, "preprocessingPreviewWindow", "预览训练窗口"
    )
    preview_button = _element_with_unique_id(
        shared_preprocessing_section, "preprocessingPreviewButton", "button", "预处理预览按钮"
    )
    preview_area = _element_with_unique_id(
        shared_preprocessing_section, "preprocessingPreview", "div", "预处理预览区域"
    )
    preview_row, preview_row_end = _row_containing_unique_field(
        shared_preprocessing_section, "preprocessingPreviewButton", "预处理预览"
    )
    shared_preprocessing_section = (
        shared_preprocessing_section[:preview_row]
        + shared_preprocessing_section[preview_row_end:]
    ).rstrip()
    parameter_group = (
        '      <div class="group shared-preprocessing">\n'
        '        <div class="group-title">分析与建模共享参数</div>\n'
        '        <div class="help">这些参数同时用于趋势浏览、状态探索、独立聚类、建模质量检查和正式训练；正式训练沿用同一套取值，不再重复配置。</div>\n'
        + shared_preprocessing_section
        + '\n      </div>'
    )
    parameter_ids = ("sampleInterval", "resamplingMethod", "filterMethod", "firstOrderAlpha", "smoothingWindow", "gapThreshold", "maxLag", "lagStep")
    field_rows = {
        field_id: _row_containing_unique_field(parameter_group, field_id, field_id)
        for field_id in parameter_ids
    }
    if list(field_rows.values()) != sorted(field_rows.values()):
        raise ValueError("无法固定Web工作台结构：训练参数行")
    quality_start = training_parameter_tail.rfind(
        "        <h3>",
        0,
        _unique_anchor_index(training_parameter_tail, 'id="modelQualityStatus"', "建模质量检查"),
    )
    if quality_start == -1:
        raise ValueError("无法固定Web工作台结构：建模质量检查标题")
    training_action_start = _unique_anchor_index(
        training_parameter_tail,
        '        <div class="actions"><button id="trainExploratoryButton"',
        "训练操作",
    )
    quality_section = training_parameter_tail[quality_start:training_action_start].rstrip()
    training_actions_section = training_parameter_tail[training_action_start:].rstrip()
    model_configuration_rows = (
        '        <div class="training-parameter-grid">\n'
        '          <div class="model-name-field">\n'
        f'            {_label_for_unique_field(training_parameter_tail, "modelName", "模型名称")}\n'
        '          </div>\n'
        f'          {_label_for_unique_field(training_parameter_tail, "varianceThreshold", "累计解释率")}\n'
        f'          {_label_for_unique_field(training_parameter_tail, "components", "主元数")}\n'
        '        </div>\n'
    )
    training_data_summary = (
        '        <div id="modelTrainingDataSummary" class="notice">'
        '需重新执行建模质量检查后显示训练数据摘要。'
        '</div>\n'
    )
    preprocessing_preview_section = (
        '        <div class="preprocessing-preview-controls">'
        f'{preview_window_label}{preview_button}'
        '</div>\n'
        f'        <div class="preprocessing-preview-area">{preview_area}</div>\n'
    )
    model_configuration_group = (
        '      <div class="group training-configuration">\n'
        '        <div class="group-title">PCA 模型配置与训练</div>\n'
        + training_data_summary
        + preprocessing_preview_section
        + quality_section
        + model_configuration_rows
        + training_actions_section
    )
    exploratory_button = _element_with_unique_id(
        model_configuration_group, "trainExploratoryButton", "button", "探索模型入口"
    )
    model_configuration_group = model_configuration_group.replace(exploratory_button, "", 1)
    exploratory_tools = (
        '        <details class="advanced-parameters exploratory-model-tools">\n'
        '          <summary>高级操作：建立探索模型</summary>\n'
        '          <div class="help">探索模型仅用于兼容保留的状态空间/聚类辅助路径，不属于正常状态主流程。</div>\n'
        f'          <div class="actions">{exploratory_button}</div>\n'
        '          <div class="notice">探索模型仅用于状态空间浏览和聚类辅助，不能作为正常状态模型。</div>\n'
        '          <div class="notice">探索模型不能执行独立验证，也不能作为正常状态模型。</div>\n'
        '        </details>'
    )
    exploratory_notice = (
        '        <div class="notice">探索模型仅用于状态空间浏览和聚类辅助，不能作为正常状态模型。</div>'
    )
    if model_configuration_group.count(exploratory_notice) != 1:
        raise ValueError("无法固定Web工作台结构：探索模型入口")
    model_configuration_group = model_configuration_group.replace(exploratory_notice, exploratory_tools, 1)
    if 'id="candidateWindows"' in parameter_group + model_configuration_group:
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
    panel_positions = []
    for panel_id in panel_markers:
        anchor = f'      <div id="{panel_id}"'
        if results.count(anchor) != 1:
            raise ValueError(f"无法固定Web工作台结构：{panel_id}锚点数量为{results.count(anchor)}")
        panel_positions.append(results.index(anchor))
    config_panel = results[panel_positions[0] : panel_positions[1]].rstrip()
    state_panel = results[panel_positions[1] : panel_positions[2]].rstrip()
    trend_panel = results[panel_positions[2] : panel_positions[3]].rstrip()
    model_panel = results[panel_positions[3] : panel_positions[4]].rstrip()
    cluster_panel = results[panel_positions[4] : panel_positions[5]].rstrip()
    performance_panel = results[panel_positions[5] : panel_positions[6]].rstrip()
    validation_panel = results[panel_positions[6] :].rstrip()

    state_exploration_button = _element_with_unique_id(
        state_panel, "stateExplorationButton", "button", "运行状态探索按钮"
    )
    state_panel = state_panel.replace(state_exploration_button, "", 1)
    exploration_controls_end = (
        '          </div>\n'
        '        </div>\n'
        '        <div id="explorationEmpty"'
    )
    if state_panel.count(exploration_controls_end) != 1:
        raise ValueError("无法固定Web工作台结构：状态探索配置")
    state_panel = state_panel.replace(
        exploration_controls_end,
        f'          </div>\n'
        f'          <div class="actions">{state_exploration_button}</div>\n'
        '        </div>\n'
        '        <div id="explorationEmpty"',
        1,
    )
    state_panel = state_panel.replace(
        '<div class="group-title">状态探索工作台</div>',
        '<div class="group-title">状态探索配置</div>',
        1,
    )
    state_panel = state_panel.replace(
        '探索结果仅用于运行状态浏览和候选窗口比较。',
        '点击“运行状态探索”后查看完整探索证据；探索结果只提供候选，不自动判定正常状态。',
        1,
    )

    config_panel = config_panel.replace(
        '>\n',
        f'>\n      <div class="data-preparation-grid">\n{upload_group}\n{tag_group}\n      </div>\n',
        1,
    )
    candidate_panels = [
        panel.replace('class="panel"', 'class="candidate-tool-panel"', 1)
        for panel in (trend_panel, state_panel)
    ]
    candidate_panels[0] = candidate_panels[0].replace(
        'class="candidate-tool-panel"', 'class="candidate-tool-panel active"', 1
    )
    advanced_candidate_panels = [
        panel.replace('class="panel"', 'class="candidate-tool-panel"', 1)
        for panel in (cluster_panel, performance_panel)
    ]
    advanced_candidate_tools = "\n".join(
        (
            '        <details class="advanced-candidate-tools">',
            '          <summary>高级辅助：独立聚类与性能筛选</summary>',
            '          <div class="candidate-tool-tabs" role="tablist">',
            '            <button type="button" class="candidate-tool-tab" data-panel="statePanels" role="tab" aria-selected="false">打开高级辅助</button>',
            '          </div>',
            *advanced_candidate_panels,
            '        </details>',
        )
    )
    validated_download = _element_with_unique_id(
        validation_panel, "validatedModelDownload", "a", "已验证模型下载入口"
    )
    freeze_box = _div_containing_unique_field(
        validation_panel, "frozenModelId", "冻结与部署入口"
    )
    validation_panel = validation_panel.replace(validated_download, "", 1).replace(
        freeze_box, "", 1
    )
    release_panel = f"""      <div id="releasePanel" class="panel">
        <div id="releaseEmpty" class="empty">模型通过独立验证和工程师确认后，可在此冻结与部署。</div>
        <div id="releaseContent" hidden>
          <h3>冻结与部署</h3>
          <div class="notice">冻结与部署导出沿用现有流程；frozen 表示工程冻结，不表示已经部署。</div>
          <div class="actions">{validated_download}</div>
{freeze_box}
        </div>
      </div>"""
    candidate_panel = "\n".join(
        (
            '      <div id="candidatePanel" class="panel">',
            '        <div class="candidate-tool-tabs" role="tablist">',
            '          <button type="button" class="candidate-tool-tab active" data-panel="trendPanel" role="tab" aria-selected="true">趋势与分析范围</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="stateExplorationPanel" role="tab" aria-selected="false">状态探索</button>',
            '        </div>',
            parameter_group,
            *candidate_panels,
            advanced_candidate_tools,
            _candidate_manager_html(training_data_section),
            '      </div>',
        )
    )
    model_panel_lines = model_panel.rsplit("\n", 1)
    if len(model_panel_lines) != 2 or model_panel_lines[1] != "      </div>":
        raise ValueError("无法固定模型训练结果区域")
    model_panel_content = model_panel_lines[0].split("\n", 1)
    if len(model_panel_content) != 2:
        raise ValueError("无法固定模型训练页面结构")
    model_panel = f"{model_panel_content[0]}\n{model_configuration_group}\n{model_panel_content[1]}\n      </div>"
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
        f"{_FORM_ALIGNMENT_STYLE}\n{_APPLE_DESIGN_STYLE}\n{_WORKBENCH_UI_STYLE}\n{_MODEL_RESULTS_STYLE}\n</head>",
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
            protected_artifact = _candidate_deletion_block_reason(run_dir)
            candidates.append(
                {
                    "run_id": run_dir.name,
                    "model_name": str(manifest["config"]["model_name"]),
                    "training_dynamic_samples": int(model.n_samples),
                    "deletable": protected_artifact is None,
                    "deletion_block_reason": (
                        f"已存在{protected_artifact}"
                        if protected_artifact
                        else None
                    ),
                }
            )
    return {"candidates": candidates}


def delete_candidate_models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_ids = payload.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids:
        raise ValueError("run_ids必须包含至少一个候选运行")
    if any(not isinstance(run_id, str) for run_id in run_ids):
        raise ValueError("run_id无效")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_ids不能重复")

    validated_ids = [
        _BASE_WEB._validated_id(run_id, "run_id") for run_id in run_ids
    ]
    checked: list[tuple[str, Path]] = []
    with ExitStack() as locks:
        for run_id in sorted(validated_ids):
            locks.enter_context(_BASE_WEB._lifecycle_lock(run_id))
        runs_dir = _BASE_WEB.RUNS_DIR.resolve()
        for run_id in validated_ids:
            run_dir = _BASE_WEB.RUNS_DIR / run_id
            if run_dir.resolve().parent != runs_dir:
                raise ValueError(f"候选模型运行记录路径无效：{run_id}")
            model_path = run_dir / "model.pcamodel"
            if not model_path.is_file():
                raise ValueError(f"候选模型运行记录不存在：{run_id}")
            if model_path.resolve().parent != run_dir.resolve():
                raise ValueError(f"候选模型包路径无效：{run_id}")
            try:
                _, manifest = _BASE_WEB.load_model_package(model_path)
            except ValueError as error:
                raise ValueError(f"候选模型包损坏：{run_id}") from error
            if (
                manifest["model_purpose"] != "normal_state"
                or manifest["model_status"] != "candidate"
            ):
                raise ValueError(f"仅允许删除normal_state/candidate模型：{run_id}")
            if _candidate_deletion_block_reason(run_dir):
                raise ValueError(f"模型存在正式下游工件，不能删除：{run_id}")
            checked.append((run_id, run_dir))

        for _, run_dir in checked:
            shutil.rmtree(run_dir)
    return {"deleted_run_ids": validated_ids}


def _candidate_deletion_block_reason(run_dir: Path) -> str | None:
    return next(
        (
            artifact
            for artifact in _CANDIDATE_PROTECTED_ARTIFACTS
            if (run_dir / artifact).exists()
        ),
        None,
    )


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
            "/api/model-candidates/delete",
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
                else delete_candidate_models_payload(payload)
                if path == "/api/model-candidates/delete"
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
