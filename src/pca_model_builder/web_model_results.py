from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import stat
import threading
from typing import Any, Sequence
from urllib.parse import urlparse
import webbrowser

from . import web_quality_layout as quality_app
from .web import UploadBusyError, _upload_lock, _upload_in_use
from . import training_exceedance
from .loading_plot import loading_plot_payload
from .model_diagnostics import compare_candidate_runs, model_structure_diagnostic


_BASE_WEB = quality_app.app.base_web
_GROUP_PROFILE_DIALOG = r"""
<dialog id="groupProfileDialog" aria-labelledby="groupProfileDialogTitle">
  <header><h2 id="groupProfileDialogTitle">工况组画像</h2><button id="groupProfileDialogClose" type="button" class="secondary" autofocus>关闭</button></header>
  <div id="groupProfileDialogBody"></div>
</dialog>
"""
_GROUP_PROFILE_DIALOG_STYLE = r"""
<style id="groupProfileDialogStyle">
  .group-profile-entry { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:8px 0; min-width:0; }
  .group-profile-entry span { overflow-wrap:anywhere; }
  #clusterContent, #assistanceClusterQuality { min-width:0; }
  #groupProfileDialog { box-sizing:border-box; width:90vw; max-width:1600px; max-height:88vh; padding:16px; border:1px solid var(--line-control); border-radius:8px; overflow:auto; color:var(--text); }
  #groupProfileDialog::backdrop { background:rgb(15 23 42 / 45%); }
  #groupProfileDialog > header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; padding:0; background:none; color:inherit; }
  #groupProfileDialog h2 { margin:0; font-size:18px; }
  #groupProfileDialog .group-profiles { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.3fr); align-items:start; gap:16px; min-width:0; }
  #groupProfileDialog .group-profiles > *, #groupProfileDialog .group-profile-detail > * { min-width:0; max-width:100%; }
  #groupProfileDialog .table-wrap { overflow:auto; max-height:280px; }
  #groupProfileDialog .group-profiles > .table-wrap { max-height:60vh; }
  #groupProfileDialog .group-profiles > .table-wrap table { width:100%; table-layout:auto; }
  #groupProfileDialog .group-profiles > .table-wrap th { white-space:nowrap; }
  #groupProfileDialog .profile-variable { display:inline-block; max-width:100%; overflow-wrap:anywhere; }
  #groupProfileDialog td, #groupProfileDialog th, #groupProfileDialog p { overflow-wrap:anywhere; white-space:normal; }
  #groupProfileDialog td, #groupProfileDialog th { padding:6px; vertical-align:top; }
  #groupProfileDialog .group-profiles > .table-wrap :is(th,td):is(:nth-child(4),:nth-child(6)) { width:1%; white-space:nowrap; }
  #groupProfileDialog .group-profiles > .table-wrap td:nth-child(2), #groupProfileDialog .group-profiles > .table-wrap td:nth-child(5), #groupProfileDialog .group-profile-detail td:nth-child(2), #groupProfileDialog .group-profile-detail td:nth-child(3) { white-space:nowrap; font-variant-numeric:tabular-nums; }
  #groupProfileDialog [data-profile] { padding:4px 6px; white-space:nowrap; }
  #groupProfileDialog [data-profile][aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
  #groupProfileDialog .group-profile-detail > .table-wrap { max-height:none; }
  #groupProfileDialog .metrics { display:flex; flex-wrap:wrap; gap:8px 16px; }
  #groupProfileDialog .metric { border:0; padding:0; box-shadow:none; background:none; }
  #groupProfileDialog pre { white-space:pre-wrap; overflow-wrap:anywhere; }
  @media (max-width:1000px) { #groupProfileDialog .group-profiles { grid-template-columns:minmax(0,1fr); } }
  @media (max-width:480px) { #groupProfileDialog { width:96vw; padding:10px; } }
</style>
"""
_TREND_APP = quality_app.app
_ASSET_PATH = Path(__file__).with_name("model_results.js")
_CANDIDATE_PROTECTED_ARTIFACTS = (
    "validated_model.pcamodel",
    "frozen_model.pcamodel",
    "deployment_model.pcadeploy",
)
_HISTORY_UPLOAD = re.compile(r"^([0-9a-f]{32})\.(csv|xlsx|txt)$", re.IGNORECASE)
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

_FORM_WIDTH_STYLE = r"""
<style id="semanticFormWidthStyle">
  /* 五类语义宽度仅约束控件；容器继续负责已有 Grid / flex-wrap 排布。 */
  main {
    --field-number-width:140px;
    --field-select-width:220px;
    --field-datetime-width:240px;
    --field-tag-width:340px;
    --field-text-width:420px;
    --field-multiple-width:420px;
  }
  main :is(input, select, textarea) { min-width:0; max-width:100%; }
  main .panel > *, main .inner-panel > * { min-width:0; max-width:100%; }
  main input[type="number"] { width:min(100%,var(--field-number-width)); }
  main select { width:min(100%,var(--field-select-width)); }
  main input[type="datetime-local"] { width:min(100%,var(--field-datetime-width)); }
  /* Tag、变量、列和窗口选择有长名称，使用同一宽度语义。 */
  main select:is(#timestampColumn, #qualityTagSelect, #explorationPerformanceTag,
    #labelColumn, #validationTimestampColumn, #preprocessingPreviewTagSelect, #preprocessingPreviewWindow,
    #dpTrendVar1, #dpTrendVar2, #dpTrendVar3, #dpTrendVar4) {
    width:min(100%,var(--field-tag-width));
  }
  main select[multiple] { width:min(100%,var(--field-multiple-width)); }
  main input:not([type]), main input[type="text"] {
    width:min(100%,var(--field-text-width));
  }
  /* 明确的全宽例外：多行备注、搜索、文件上传和条件行内的单元格。 */
  main textarea, main #tagSearch, main input[type="file"] { width:100%; }
  main input[type="range"] { width:min(100%,var(--field-select-width)); }
  /* 数据源字段按各自语义宽度紧凑左对齐，不各占半个工作台。 */
  main .row:has(> label > #timestampColumn) {
    display:flex; flex-wrap:wrap; align-items:end;
  }
  main .row:has(> label > #timestampColumn) > label { flex:0 1 var(--field-select-width); min-width:0; max-width:100%; }
  main .row:has(> label > #timestampColumn) > label:has(> #timestampColumn) { flex-basis:var(--field-tag-width); }
  /* 数据源控件同排按内容角色取宽：文件按内容宽度，时间列用 Tag 宽度，编码选择保持短控件宽度。 */
  @media (min-width:761px) {
    main .group:has(#uploadButton) {
      display:flex;
      flex-wrap:wrap;
      align-items:end;
    }
    main .group:has(#uploadButton) > label { flex:0 0 auto; min-width:0; max-width:100%; }
    main .group:has(#uploadButton) > :is(.actions, .row) { display:contents; }
    main .group:has(#uploadButton) > :is(.actions, .row) > label > select { width:100%; }
  }
  @media (max-width:760px) {
    main .row:has(> label > #timestampColumn) { display:grid; grid-template-columns:minmax(0,1fr); }
  }
</style>
"""


_FORM_LAYOUT_STYLE = r"""
<style id="compactFormLayoutStyle">
  main :is(.row, .condition-row, .validation-box, .exploration-controls,
    .trend-controls, .dp-trend-controls, .dp-scatter-controls,
    .training-parameter-grid, .quality-tag-controls) {
    display:flex;
    flex-wrap:wrap;
    gap:var(--space-2);
    justify-content:start;
    align-items:end;
  }
  main :is(.row, .condition-row, .validation-box, .exploration-controls,
    .trend-controls, .dp-trend-controls, .dp-scatter-controls,
    .training-parameter-grid, .quality-tag-controls) > * {
    min-width:0;
    max-width:100%;
  }
  main :is(.row, .condition-row, .validation-box, .exploration-controls,
    .trend-controls, .dp-trend-controls, .dp-scatter-controls,
    .training-parameter-grid, .quality-tag-controls) > label { flex:0 0 auto; }
  main :is(.row, .condition-row, .validation-box, .exploration-controls,
    .trend-controls, .dp-trend-controls, .dp-scatter-controls,
    .training-parameter-grid, .quality-tag-controls) > :is(button, .download) {
    flex:0 0 auto;
    width:auto;
  }
  main .condition-row > label:first-child > select,
  main .dp-scatter-controls select { width:min(100%,var(--field-tag-width)); }
  main .dp-trend-bar > label { flex:0 0 auto; max-width:none; }
  main .preprocessing-preview-controls {
    display:flex;
    flex-wrap:wrap;
    gap:var(--space-2);
    justify-content:start;
    align-items:end;
    margin-top:var(--space-2);
  }
  main .preprocessing-preview-controls > label { flex:0 0 auto; }
  main .preprocessing-preview-controls > div { flex:1 1 300px; min-width:0; max-width:100%; }
  main #engineeringPanel .detail-fields {
    gap:var(--space-1);
    justify-content:start;
    align-items:end;
    max-width:none;
  }

  main #engineeringPanel .detail-fields > button,
  main .candidate-manager .row > button,
  main .dp-trend-bar > button,
  main :is(.actions, .tag-toolbar, .exploration-region-tools, .candidate-tool-tabs,
    .preprocessing-preview-controls) > button {
    flex:0 0 auto;
    width:auto;
  }
  @media (max-width:760px) {
    main :is(.row, .condition-row, .validation-box, .exploration-controls,
      .trend-controls, .dp-trend-controls, .dp-scatter-controls,
      .training-parameter-grid, .quality-tag-controls) {
      display:grid;
      grid-template-columns:minmax(0,1fr);
      justify-content:start;
    }
    main .exploration-controls.performance-controls { grid-template-columns:minmax(0,1fr); }
    main :is(.row, .condition-row, .validation-box, .exploration-controls,
      .trend-controls, .dp-trend-controls, .dp-scatter-controls,
      .training-parameter-grid, .quality-tag-controls) > label { width:100%; }
    main :is(.row, .condition-row, .validation-box, .exploration-controls,
      .trend-controls, .dp-trend-controls, .dp-scatter-controls,
      .training-parameter-grid, .quality-tag-controls,
      .preprocessing-preview-controls) > :is(button, .download) {
      justify-self:start;
      width:auto;
    }
    main .preprocessing-preview-controls { display:grid; grid-template-columns:minmax(0,1fr); }

    main #engineeringPanel .detail-fields > button { justify-self:start; }
  }
  /* Tag editor: text, unit, role, and paired limits keep their content roles. */
  main #engineeringPanel .detail-fields {
    display:grid;
    grid-template-columns:minmax(0,1fr);
  }
  main #engineeringPanel .detail-fields > .row:has(#tagDescription) > label:has(> #tagDescription) {
    flex:1 1 var(--field-text-width);
    max-width:var(--field-text-width);
  }
  main #engineeringPanel .detail-fields > .row:has(#tagUnit) > label:has(> #tagUnit) {
    flex:0 1 var(--field-number-width);
    max-width:var(--field-number-width);
  }
  main #engineeringPanel .detail-fields > .row:has(#tagRole) > label:has(> #tagRole) {
    flex:0 1 var(--field-select-width);
    max-width:var(--field-select-width);
  }
  main #engineeringPanel .detail-fields > .row:has(#engineeringMin) > label,
  main #engineeringPanel .detail-fields > .row:has(#normalMin) > label,
  main #engineeringPanel .detail-fields > .row:has(#alarmMin) > label {
    flex:0 1 var(--field-number-width);
    max-width:var(--field-number-width);
  }
</style>
"""


_SCREENING_LAYOUT_STYLE = r"""
<style id="screeningLayoutStyle">
  /* 显式单列轨道：摘要栏用 nowrap 保持单行，隐式 auto 轨道会按内容 min-content 定尺
     并把整页顶宽（与 .candidate-tool-panel.active 同一处理）。 */
  #candidatePanel { gap:var(--space-3); grid-template-columns:minmax(0,1fr); }
  #candidatePanel :is(#modelingEligibility, .shared-preprocessing),
  #stateExplorationPanel > .group { padding:var(--space-3); gap:var(--space-1); }
  #modelingEligibility p.help { margin:0; }
  #modelingEligibility .screening-rules {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));
    gap:var(--space-3); min-width:0;
  }
  #modelingEligibility .screening-rules > div { min-width:0; }
  /* 筛选准备区块内的 h3/h4 是小节标题：区块标题由 .group-title(17) 承担。 */
  #modelingEligibility :is(h3,h4) { font-size:var(--type-sub); }
  #modelingEligibility .eligibility-exclude-group {
    border:0; border-top:1px solid var(--line-soft); border-radius:0;
    padding:var(--space-2) 0 0; margin:0;
  }
  #modelingEligibility .condition-row { gap:var(--space-1); }
  #modelingEligibility .eligibility-condition {
    grid-template-columns:minmax(120px,1.25fr) minmax(100px,.9fr) minmax(74px,.6fr) minmax(95px,.8fr) max-content;
  }
  #modelingEligibility .condition-row > label { flex:1 1 120px; }
  #modelingEligibility .condition-row > label:first-child { flex-basis:200px; }
  #modelingEligibility .condition-row :is(input,select) { width:100%; }
  #modelingEligibility .eligibility-preview { grid-column:1 / -1; }
  @media (max-width:760px) {
    #modelingEligibility .eligibility-condition { grid-template-columns:minmax(0,1fr); }
  }
  #eligibilitySummary { padding:var(--space-1) 0; font-size:13px; color:var(--text); }
  #eligibilitySummary.warning { padding:var(--space-2); border-left:4px solid var(--warn); background:#fff8e7; color:#765000; }
  #modelingEligibility .candidate-analysis-range { width:100%; }
  #candidatePanel .shared-preprocessing { grid-template-columns:minmax(0,1fr); }
  #candidatePanel .shared-preprocessing > * { grid-column:1 / -1; }
  #candidatePanel .shared-preprocessing > .group-title + .help { grid-column:1; }
  #candidatePanel .shared-preprocessing .row:not(.state-filter-actions-row),
  #modelingEligibility .candidate-analysis-range > .row,
  #stateExplorationPanel .exploration-controls {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));
    gap:var(--space-1) var(--space-2); align-items:end; padding:0; background:transparent;
  }
  #modelingEligibility .candidate-analysis-range > .row {
    grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));
  }
  #candidatePanel .shared-preprocessing label,
  #modelingEligibility .candidate-analysis-range label,
  #stateExplorationPanel .exploration-controls > label { margin:0; gap:var(--space-1); }
  #candidatePanel .shared-preprocessing :is(input,select),
  #modelingEligibility .candidate-analysis-range input,
  #stateExplorationPanel .exploration-controls :is(input,select) { width:100%; }
  #candidatePanel .shared-preprocessing .lag-expansion-parameter-row {
    grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),240px));
  }
  #stateExplorationPanel { gap:var(--space-3); }
  #stateExplorationPanel > .screening-execute { margin:0; padding:0 0 var(--space-3); border-bottom:1px solid var(--line); }
  #stateExplorationPanel :is(.chart-card,.table-wrap) { border:0; border-radius:0; }
  #stateExplorationPanel .chart-card { padding:var(--space-2) 0; }
  /* 卡片标题统一到第三档 14px（见 .chart-card h3）；本条原有的 17px 覆盖已随标题分层修复移除。 */
  #stateExplorationPanel #explorationEvidence { margin:0; }
  #stateExplorationPanel #explorationContent { gap:var(--space-3); }
  #stateExplorationPanel #explorationClusterQuality { display:grid; gap:var(--space-2); min-width:0; }
  #stateExplorationPanel .screening-kpis { gap:var(--space-1); padding-top:0; }
  #stateExplorationPanel .screening-kpis > p { margin:0; line-height:1.45; }
  #stateExplorationPanel .screening-judgment { border-left:3px solid var(--accent); padding:var(--space-3); background:var(--accent-soft); }
  #stateExplorationPanel .screening-judgment p { margin:0; }
  #stateExplorationPanel #explorationQualityDetails { border-top:1px solid var(--line); }
  #stateExplorationPanel :is(#explorationClusterDetails,.exploration-candidate-details) > summary,
  #stateExplorationPanel .variable-diagnostics summary { cursor:pointer; }
  #stateExplorationPanel :is(#explorationClusterDetails,.exploration-candidate-details) .table-wrap { max-height:280px; }
  #stateExplorationPanel .exploration-candidate-details .table-wrap { max-width:100%; }
  #stateExplorationPanel .variable-diagnostics > details { min-width:0; }
  #stateExplorationPanel .screening-evidence-compact { grid-template-columns:minmax(0,1fr); gap:var(--space-1); }
  #stateExplorationPanel .screening-evidence-compact .help { margin:0; }
  #stateExplorationPanel .screening-center-heading { display:flex; flex-wrap:wrap; align-items:baseline; gap:var(--space-1) var(--space-3); }
  /* 该标题位于 .chart-card 内，属卡片标题（第三档 14px），与同面板其他卡片标题取同一字号；
     仅需清掉 h3 的默认外边距。 */
  #stateExplorationPanel .screening-center-heading h3 { margin:0; }
  #stateExplorationPanel .screening-center-table { width:max-content; max-width:100%; overflow-x:auto; }
  #stateExplorationPanel .screening-center-table table { width:max-content; min-width:0; }
  #stateExplorationPanel .screening-center-table :is(th,td) { padding:6px 8px; white-space:nowrap; }
  #stateExplorationPanel :is(th,td) { padding:6px 8px; }
  #stateExplorationPanel th { background:var(--line-soft); font-weight:600; }
  /* 对齐按语义：只把数值列右对齐，文本列回落默认左对齐。
     原来对 :not(:first-child) 一刀切右对齐，会把「主要变量」「时间范围」等文本列也右推。 */
  #stateExplorationPanel :is(th,td).numeric { text-align:right; font-variant-numeric:tabular-nums; }
  @media (min-width:761px) {
    #stateExplorationPanel .performance-controls > label:has(#explorationPerformanceTag) { grid-column:span 2; }
  }
</style>
"""


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
    display:flex;
    flex-wrap:wrap;
    align-items:flex-end;
    gap:var(--space-2);
  }
  #engineeringPanel .batch-config .actions > .download,
  #engineeringPanel .batch-config .actions > button {
    flex:0 0 auto;
    width:auto;
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
  /* 文件选择固定宽度，按钮按内容宽度换行。 */
  #engineeringPanel .batch-config .actions > label.secondary {
    flex:0 0 300px;
    max-width:100%;
    display:grid;
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

  #clusterPanel .group, #performancePanel .group { gap:var(--space-2); }
  #clusterPanel .row, #performancePanel .row {
    align-items:start;
  }
  #clusterPanel .row > label, #performancePanel .row > label {
    min-width:0;
    align-content:start;
  }
  #clusterPanel .row input, #performancePanel .row input {
    min-height:var(--control-height);
    height:var(--control-height);
  }
  #clusterPanel #clusterButton, #performancePanel #performanceButton {
    align-self:end;
    min-height:var(--control-height);
    height:var(--control-height);
  }

  #engineeringPanel .detail-fields {
    max-width:100%;
    min-width:0;
    gap:var(--space-2);
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
  #engineeringPanel #tagRole { align-self:start; }
  #engineeringPanel #saveTagConfig {
    min-height:var(--control-height);
    margin-top:var(--space-1);
    /* 与批量配置区的下载按钮等宽：两者共用 min-width，文本变长也不会让下方按钮超出上方宽度。 */
    min-width:var(--batch-action-width);
    justify-self:start;
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
    /* 交互控件的可选边界需要 3:1 非文本对比；--line-soft(1.14:1) 只适合装饰分隔线。 */
    --line-control:#8a8a8a;
    --text:#1d1d1f;
    /* 说明文字要在白底(5.3:1)与 --bg 浅灰底(4.9:1)上都满足 WCAG AA 4.5:1；
       #7a7a7a 实测 4.29:1 / 3.94:1 不达标。 */
    --muted:#6b6b6b;
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
    --panel-padding:16px;
    /* 三级字号：页面标题 21 / 区块标题 17 / 小节与卡片标题 14。
       层级按视觉角色分配，不跟 heading 标签绑定。 */
    --type-page:21px;
    --type-section:17px;
    --type-sub:14px;
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
  /* main 在 ≥1800px 时按 max-width 居中，页头必须用同一列宽表达式，否则宽屏下标题与内容错位。 */
  header { padding:16px max(clamp(16px,2.5vw,32px), calc((100% - 1800px) / 2 + clamp(16px,2.5vw,32px))); background:#000000; border-bottom:0; color:#ffffff; }
  h1 { margin:0 0 4px; font-size:var(--type-page); font-weight:600; line-height:1.19; letter-spacing:.231px; }
  /* 同一个 h3 既可能是区块标题（17），也可能是卡片小节标题（14），
     由角色类选择器决定层级，不再让 h3 出现第三种字号。 */
  h2, h3 { font-size:var(--type-section); font-weight:600; line-height:1.19; letter-spacing:.231px; }
  h4 { font-size:var(--type-sub); font-weight:600; line-height:1.24; letter-spacing:0; }
  /* 卡片标题与 h4 同属第三档：同一层级共用字号，避免同一区域内出现两种小节字号。 */
  .chart-card h3 { font-size:var(--type-sub); }
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
  /* .candidate-tool-panel 由 .panel 改名而来，应与 .panel/.inner-panel 同列补齐 min-width:0。
     注意：这一条单独并不能修好窄屏溢出（实测仅加它仍是 133px），真正的修复在
     .candidate-tool-panel.active 的 minmax(0,1fr) 轨道上；这里保留只是消除改名遗漏。 */
  main > section, .results > *, .panel, .inner-panel, .candidate-tool-panel, .group, .chart-card { min-width:0; }
  .controls, .controls .group { min-width:0; }
  .controls { gap:var(--space-3); }
  .results { gap:var(--space-3); container-type:inline-size; }
  :is(#explorationContent,#clusterContent,#performanceContent,#modelQualityResults,#modelContent,#validationContent,#releaseContent):not([hidden]) {
    display:grid; grid-template-columns:minmax(0,1fr); gap:var(--space-3); align-content:start; min-width:0;
  }
  .row { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:var(--space-2); align-items:end; }
  .actions, .tag-toolbar, .detail-fields { gap:var(--space-2); }
  .actions { display:flex; flex-wrap:wrap; align-items:center; justify-content:flex-start; }
  .inner-panel.active, .panel.active { gap:var(--space-3); }
  .group, .validation-box, .exploration-controls, .dp-inline-help {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:6px;
  }
  .group { gap:var(--space-2); padding:var(--panel-padding); }
  .group-title { margin:0; font-size:var(--type-section); font-weight:600; line-height:1.4; }
  .sub-title { margin:0; font-size:var(--type-sub); font-weight:600; line-height:1.4; }
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
  button.secondary, .download.secondary {
    background:var(--bg);
    border-color:var(--line-control);
    color:var(--accent);
    border-radius:var(--control-radius);
  }
  button.danger { background:var(--danger); border-color:var(--danger); color:#fff; }
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
  #engineeringPanel #tagRole {
    box-sizing:border-box;
    height:var(--control-height);
    min-height:var(--control-height);
  }
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
  .chart-card { gap:12px; }
  .chart-card h3 { margin:0; }
  .chart, .dp-chart, .trend-chart { background:var(--panel); }
  .empty { border-style:dashed; color:var(--muted); }
  .variance-item { color:var(--muted); }
  .variance-item.selected { color:var(--green); font-weight:600; }
  .table-wrap, .exploration-timeline { border:1px solid var(--line); }
  th, td { border-bottom-color:var(--line); padding:6px 8px; }
  th { background:var(--line-soft); color:var(--text); font-weight:600; }
  a:not(.download) { color:var(--accent); }
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
  .workflow-sidebar-title { margin:0; font-size:var(--type-section); font-weight:600; }
  .workflow-steps { display:grid; gap:var(--space-1); }
  .workflow-history-tab { width:100%; text-align:left; }
  .workflow-history-tab[aria-pressed="true"] { border-color:var(--accent); background:#edf5ff; }
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
  .candidate-workflow-step { min-width:0; align-self:start; }
  .history-cleanup-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .history-cleanup-heading h2 { margin:0; }
  .history-cleanup-name { overflow-wrap:anywhere; }
  .history-cleanup-reasons { color:var(--danger); white-space:normal; }
  .history-cleanup-confirm { display:flex; align-items:center; gap:8px; }
  #historyCleanupPanel[hidden] { display:none; }
  .workflow-steps:has(#candidateSectionNav:not([hidden])) > .workflow-step { align-self:start; }
  .candidate-section-nav { display:grid; gap:4px; padding:4px 0 8px 42px; }
  .candidate-section-nav[hidden] { display:none; }
  .candidate-section-nav a {
    min-width:0; padding:6px 8px; border-left:2px solid transparent;
    color:var(--muted); font-size:13px; line-height:1.4; text-decoration:none;
  }
  .candidate-section-nav a[aria-current="location"] { border-left-color:var(--accent); color:var(--accent); }
  .candidate-section-nav a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #candidatePanel :is(#modelingEligibility, #candidateAnalysis, #explorationEvidence,
    #trendEvidence, #clusterEvidence, #performanceEvidence, #candidateManagement) { scroll-margin-top:var(--space-4); }
  .tag-workspace { display:grid; grid-template-columns:minmax(350px,.85fr) minmax(0,1fr); gap:var(--space-3); align-items:start; }
  @container (max-width:820px) {
    .tag-workspace { grid-template-columns:minmax(0,1fr); }
  }
  .tag-workspace > *, .tag-detail { min-width:0; }
  #configPanel > .group { margin-bottom:var(--space-3); }
  #configPanel > #qualityPanel { display:grid; gap:var(--space-2); margin-top:var(--space-4); }
  .candidate-manager, .shared-preprocessing, .training-configuration { border-color:#bfd7ef; }
  .candidate-tool-tabs { display:flex; gap:var(--space-1); flex-wrap:wrap; align-items:center; border-bottom:1px solid var(--line); padding-bottom:var(--space-2); }
  .candidate-tool-tab, .training-tool-tab { background:#f5f5f7; border-color:#f0f0f0; color:var(--accent); }
  .candidate-tool-tab.active, .training-tool-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  /* 摘要栏固定单行：min-width:0 让 span 的 min-content 不再撑宽 #candidatePanel 的隐式
     auto 轨道，nowrap + ellipsis 让宽度不足时省略文本，按钮始终可见。 */
  .candidate-config-summary { display:flex; align-items:center; gap:8px; min-width:0; font-size:13px; line-height:1.45; }
  .candidate-config-summary[hidden] { display:none; }
  .candidate-config-summary span { flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .candidate-config-summary button { flex:none; }
  .candidate-tool-panel { display:none; gap:var(--space-3); }
  /* 窄屏溢出的修复点：display:grid 的隐式 auto 轨道按内容 min-content 定尺，宽表会把
     轨道连同整页顶宽（实测 390px 下 133px 溢出）。minmax(0,1fr) 给轨道 0 下限，让面板内的
     .table-wrap 拿到有界宽度，改由它自己的 overflow:auto 承接横向滚动。 */
  .candidate-tool-panel.active { display:grid; grid-template-columns:minmax(0,1fr); }
  .candidate-analysis-range { display:grid; max-width:100%; }
  .candidate-analysis-range label { min-width:0; }
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
  .training-parameter-grid > label,
  .filter-parameter-control,
  .model-name-field { min-width:0; }
  .filter-parameter-control > label { min-width:0; }
  .preprocessing-preview-area,
  .preprocessing-preview-area #preprocessingPreview { width:100%; min-width:0; }
  .preprocessing-preview-area #preprocessingPreview { margin-top:var(--space-2); }
  .preprocessing-preview-area #preprocessingPreviewTagSelect {
    width:min(100%,var(--field-tag-width));
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
  /* 停用态改为继承 token：硬编码的 #6b7280 on #f3f4f6 只有 4.39:1，低于 WCAG AA 的
     4.5:1（停用控件虽属 1.4.3 豁免，但没必要时贴着下限）。 */
  button:disabled, input:disabled, select:disabled, textarea:disabled {
    background:var(--bg);
    border-color:var(--line);
    color:var(--muted);
    opacity:1;
  }
  button:disabled { cursor:not-allowed; }
  .status-label {
    display:inline-block;
    margin-right:6px;
    padding:1px 6px;
    border:1px solid currentColor;
    border-radius:3px;
    font-size:12px;
    font-weight:600;
    line-height:1.45;
    white-space:nowrap;
  }
  .status-label.normal, .status-label.usable, .status-label.ready,
  .status-label.accepted, .status-label.used,
  .status-label[data-tone="normal"] { color:var(--normal); }
  .status-label.attention, .status-label.review,
  .status-label.pending, .status-label[data-tone="attention"] { color:#8a5a00; }
  .status-label.abnormal, .status-label.blocking,
  .status-label.rejected, .status-label.dropped,
  .status-label[data-tone="abnormal"] { color:var(--danger); }
  .status-label.disabled, .status-label[data-tone="neutral"] { color:var(--muted); }
  .table-wrap tbody tr:hover { background:#f7fbff; }
  .table-wrap th:first-child, .table-wrap td:first-child { position:sticky; left:0; z-index:1; }
  .table-wrap th:first-child { background:var(--line-soft); }
  /* 限高表格纵向滚动时表头跟随，否则滚动到底部后看不到列名。 */
  .table-wrap thead th { position:sticky; top:0; z-index:2; background:var(--line-soft); }
  .table-wrap thead th:first-child { z-index:3; }
  /* 固定首列必须自带不透明底色：tr/table 背景是透明的，继承过来仍是透明，
     横向滚动时右邻列文字会直接透过 sticky 列叠印在一起。 */
  /* 分隔线改用 border：全局 box-shadow:none !important 会吃掉原来的首列投影。 */
  .table-wrap td:first-child { background:var(--panel); border-right:1px solid var(--line); }
  .table-wrap tbody tr:hover td:first-child { background:#f7fbff; }
  /* 数值语义由 th 与 td 共用：数值列的标题与内容取同一对齐边。
     状态、文字、操作列不带 .numeric，继续左对齐或居中。 */
  .table-wrap td.numeric, .table-wrap th.numeric {
    text-align:right;
    font-variant-numeric:tabular-nums;
  }
  .table-wrap td.numeric { white-space:nowrap; }
  /* 状态/勾选列的单元格由 .table-wrap td:has(> .status-label) 与 td:has(> input[type=checkbox])
     居中；这里让同一列的表头跟随该对齐边，避免表头左对齐、内容居中的错位。 */
  .table-wrap th.control { text-align:center; }
  #stateExplorationPanel .exploration-controls { border:0; }
  /* 标签换行时输入框仍需贴底对齐，否则同一行控件高低不齐。 */
  .exploration-controls > label { display:grid; align-content:start; }
  .panel > .actions, .inner-panel > .actions { margin-top:var(--space-1); }
  .issue-card, .notice, .status, .metric, .chart-card, .table-wrap, .empty, .validation-box, .exploration-controls { min-width:0; }
  .issue-card, .notice { padding:var(--space-2); }
  .chart-card { gap:var(--space-2); }
  .chart-grid { align-items:stretch; }
  .table-wrap { width:100%; }
  /* 表头与短状态值不逐字换行；窄表由 .table-wrap 横向滚动承接。 */
  .table-wrap th { white-space:nowrap; }
  /* 列宽按内容取值，避免日期被压成多行；宽度不足时由 .table-wrap 横向滚动。 */
  .table-wrap table { width:max-content; min-width:100%; }
  /* 两列键值表（验证指标、建模快照等）按内容定宽：撑满容器会让值列留出数百像素空白。 */
  .table-wrap table:has(tbody tr > th:first-child:nth-last-child(2)) { min-width:0; }
  /* 优选运行区域统计：两张表按内容定宽左对齐，窄屏由 .table-wrap 在本区域内滚动。
     上面的 min-width:100% 与面板居中规则同样命中新表，因此紧凑覆盖放在其后。 */
  #stateExplorationPanel .region-stats table { min-width:0; font-size:12.5px; }
  #stateExplorationPanel .region-stats :is(th,td) { padding:4px 8px; }
  #stateExplorationPanel .region-stats .region-metric-table td:nth-child(odd),
  #stateExplorationPanel .region-stats .region-cluster-table :is(th,td):first-child { text-align:left; }
  #stateExplorationPanel .region-stats .region-metric-table td:nth-child(even),
  #stateExplorationPanel .region-stats .region-cluster-table :is(th,td):not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
  /* 两张紧凑表按内容定宽、通常不横向滚动，因此不做固定首列，也去掉首列分隔线。 */
  #stateExplorationPanel .region-stats :is(th,td):first-child { position:static; border-right:0; }
  #modelQualityResults > :is(#currentTagQuality, #qualityIssues).empty,
  .final-training-review > .empty,
  #basicInspectionIssues.empty, #explorationEmpty.empty,
  #clusterEmpty.empty, #performanceEmpty.empty, #modelEmpty.empty,
  #validationEmpty.empty, #releaseEmpty.empty {
    min-height:0;
    padding:var(--space-2);
  }
  .table-wrap td:has(> .status-label),
  .table-wrap td:has(> input[type=checkbox]) { white-space:nowrap; text-align:center; }
  tr:has(> td > input[type=checkbox]) > td:nth-child(2) { white-space:nowrap; }
  /* 状态探索结果表不再整表居中：整表居中会覆盖 .numeric，使数值列失去对齐语义。
     勾选 / 状态列沿用上方 .table-wrap 的居中规则；文本列回落默认左对齐。 */
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
  /* 表格内命令按钮按单元格密度收敛：42px 全高控件会把所在行撑到 55px，
     与同表 31px 的纯文本行交替出现。 */
  .table-wrap button { height:30px; min-height:30px; padding:0 10px; font-size:12px; }
  @media (max-width:1050px) {
    main { grid-template-columns:minmax(0,1fr); }
    .workflow-sidebar { position:static; }
    /* 单列布局下侧栏横条是导航，不该靠横向滚动藏步骤：5×190px 的硬下限在 900px
       视口装不下（容器 855px），第 5 步「冻结与部署」被推到滚动区外。
       改用 auto-fit 换行，步骤数与当前位置始终完整可见。 */
    .workflow-steps { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .workflow-step-next { display:none; }
  }
  @media (max-width:760px) {
    main { padding:var(--space-2); gap:var(--space-2); }
    .workflow-sidebar { position:static; padding:var(--space-3); }
    .workflow-steps { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .workflow-step { padding:6px; gap:6px; min-height:44px; grid-template-columns:24px minmax(0,1fr) auto; }
    .workflow-step-number { width:24px; height:24px; }
    .workflow-step-status { font-size:10px; }
    .candidate-analysis-range { grid-template-columns:minmax(0,1fr); }
    .candidate-tool-tabs { flex-wrap:nowrap; overflow-x:auto; }
    .candidate-tool-tabs > * { flex:0 0 auto; width:auto; }
    .actions { align-items:stretch; }
    /* 表单字段填满列宽；command button 保持 intrinsic 宽度并自然换行。 */
    .panel .actions > label, .inner-panel .actions > label { width:100%; }
    .tabs > *, .inner-tabs > * { flex:1 1 0; width:auto; min-width:0; }
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
  const trainingTable = document.getElementById("trainingWindows");
  const releasePanel = document.getElementById("releasePanel");
  const workflowSteps = document.getElementById("workflowSteps");
  const historyCleanupPanel = document.getElementById("historyCleanupPanel");
  const historyCleanupTab = document.getElementById("historyCleanupTab");
  const candidateNavigation = document.getElementById("candidateSectionNav");
  const sectionLinks = [...candidateNavigation.querySelectorAll("a")];
  const evidenceIds = {trendPanel:"trendEvidence", stateExplorationPanel:"explorationEvidence", clusterPanel:"clusterEvidence", performancePanel:"performanceEvidence"};
  let currentSection = 0, requestedSection = null, spyFrame = 0;
  const highlightCandidateSection = index => {
    currentSection = index;
    sectionLinks.forEach((link, position) => {
      if (position === index) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };
  const updateCandidateSpy = () => {
    spyFrame = 0;
    if (candidateNavigation.hidden) return;
    const positions = sectionLinks.map(link => link.hidden ? Infinity : document.getElementById(link.hash.slice(1)).getBoundingClientRect().top);
    const atBottom = scrollY + innerHeight >= document.documentElement.scrollHeight - 2;
    if (requestedSection !== null) {
      if (positions[requestedSection] > 64 && !atBottom) return;
      highlightCandidateSection(requestedSection);
      requestedSection = null;
    }
    let index = 0;
    positions.forEach((top, position) => { if (top <= 64) index = position; });
    if (atBottom) {
      // Short empty results cannot reach the top: retain the selected, visible heading.
      if (positions[currentSection] >= 0) return;
      index = positions.length - 1;
    } else if ((index > currentSection && positions[index] > 48) || (index < currentSection && positions[currentSection] < 80)) return;
    if (index !== currentSection) highlightCandidateSection(index);
  };
  const scheduleCandidateSpy = () => { if (!spyFrame) spyFrame = requestAnimationFrame(updateCandidateSpy); };
  sectionLinks.forEach((link, index) => link.addEventListener("click", event => {
    event.preventDefault();
    requestedSection = index;
    highlightCandidateSection(index);
    document.getElementById(link.hash.slice(1)).scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block:"start"});
    scheduleCandidateSpy();
  }));
  window.addEventListener("scroll", scheduleCandidateSpy, {passive:true});
  window.addEventListener("scrollend", () => { requestedSection = null; scheduleCandidateSpy(); });
  window.addEventListener("resize", scheduleCandidateSpy);
  new ResizeObserver(scheduleCandidateSpy).observe(candidatePanel);
  highlightCandidateSection(0);
  const toolPanels = ["candidatePreparationPanel", "trendPanel", "stateExplorationPanel", "clusterPanel", "performancePanel"].map(id => document.getElementById(id));
  const summary = document.getElementById("candidateConfigSummary");
  const refreshCandidateConfigSummary = () => {
    const value = id => {
      const control = document.getElementById(id);
      return control.selectedOptions?.[0]?.textContent || control.value || control.placeholder || "未设置";
    };
    // 只读候选分析范围的资格节点：趋势窗口摘要写入 #trendEligibilitySummary，不参与此栏。
    const eligibility = document.getElementById("eligibilitySummary");
    const filterMethod = document.getElementById("filterMethod").value;
    const summaryText = [
      eligibility.dataset.compact || eligibility.textContent,
      `分析范围：${value("analysisStart")} 至 ${value("analysisEnd")}`,
      `采样周期：${value("sampleInterval")} 分钟`,
      `滤波：${value("filterMethod")}`,
      ...(filterMethod === "first_order" ? [`α：${value("firstOrderAlpha")}`] : []),
      ...(filterMethod === "trailing_mean" ? [`窗口：${value("smoothingWindow")} 分钟`] : []),
      `最大 Lag：${value("maxLag")} 分钟`, `Lag 步长：${value("lagStep")} 分钟`,
      `重采样：${value("resamplingMethod")}`,
      `缺口阈值：${value("gapThreshold")}${document.getElementById("gapThreshold").value ? " 分钟" : ""}`
    ].join(" · ");
    // 单行 nowrap + ellipsis 会截断尾部参数，用 title 保留完整文本供悬停查看。
    const summarySpan = summary.querySelector("span");
    summarySpan.textContent = summaryText;
    summarySpan.title = summaryText;
  };
  summary.querySelector("button").addEventListener("click", () => showCandidateTool("candidatePreparationPanel"));
  candidatePanel.addEventListener("input", refreshCandidateConfigSummary);
  candidatePanel.addEventListener("change", refreshCandidateConfigSummary);
  new MutationObserver(refreshCandidateConfigSummary).observe(document.getElementById("eligibilitySummary"), {childList:true, characterData:true, subtree:true});
  const showCandidateTool = target => {
    globalThis.closeGroupProfileDialog?.();
    toolPanels.forEach(panel => panel.classList.toggle("active", panel.id === target));
    document.querySelectorAll(".candidate-tool-tab").forEach(button => {
      const selected = button.dataset.panel === target;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
    const preparing = target === "candidatePreparationPanel";
    document.getElementById("candidateConfigSummary").hidden = preparing;
    sectionLinks[1].hidden = preparing;
    if (!preparing) sectionLinks[1].setAttribute("href", "#" + evidenceIds[target]);
    requestedSection = null;
    highlightCandidateSection(0);
    refreshCandidateConfigSummary();
    scheduleCandidateSpy();
  };
  globalThis.showCandidateTool = showCandidateTool;
  document.querySelectorAll(".candidate-tool-tab").forEach(button => button.addEventListener("click", () => {
    globalThis.showWorkflowStage("candidatePanel");
    showCandidateTool(button.dataset.panel);
  }));
  showCandidateTool("trendPanel");
  const validatedDownload = document.getElementById("validatedModelDownload");

  globalThis.showWorkflowStage = target => {
    globalThis.closeGroupProfileDialog?.();
    globalThis.refreshValidationInvestigationContext?.();
    historyCleanupPanel.hidden = true;
    historyCleanupTab.setAttribute("aria-pressed", "false");
    [dataPanel, candidatePanel, modelPanel, validationPanel, releasePanel].forEach(panel => panel.classList.toggle("active", panel.id === target));
    candidateNavigation.hidden = target !== "candidatePanel";
    scheduleCandidateSpy();
    if (target === "modelPanel") {
      const fields = [["采样周期", "sampleInterval"], ["重采样", "resamplingMethod"], ["滤波", "filterMethod"], ["缺口阈值", "gapThreshold"], ["最大 Lag", "maxLag"], ["Lag 步长", "lagStep"]];
      document.getElementById("modelPreprocessingSummary").textContent = fields.map(([label, id]) => {
        const control = document.getElementById(id);
        return `${label}：${control.selectedOptions?.[0]?.textContent || control.value || "默认"}`;
      }).join(" · ");
    }
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
  globalThis.showHistoryCleanup = () => {
    globalThis.closeGroupProfileDialog?.();
    [dataPanel, candidatePanel, modelPanel, validationPanel, releasePanel].forEach(panel => panel.classList.remove("active"));
    candidateNavigation.hidden = true;
    historyCleanupPanel.hidden = false;
    historyCleanupTab.setAttribute("aria-pressed", "true");
    scheduleCandidateSpy();
  };
  historyCleanupTab.addEventListener("click", () => {
    globalThis.showHistoryCleanup();
    globalThis.refreshHistoryCleanup?.();
  });
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
    // The observer below watches hidden; unchanged writes would schedule another refresh.
    if (document.getElementById("releaseEmpty").hidden !== canRelease) document.getElementById("releaseEmpty").hidden = canRelease;
    if (document.getElementById("releaseContent").hidden !== !canRelease) document.getElementById("releaseContent").hidden = !canRelease;
    const canReplay = !document.getElementById("frozenModelDownload").hidden;
    const replay = document.getElementById("frozenReplay");
    if (replay.hidden !== !canReplay) replay.hidden = !canReplay;
    const replayButton = document.getElementById("frozenReplayButton");
    replayButton.disabled = !canReplay || replayButton.textContent === "回放中…";
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
    rejected:"已拒绝", used:"已使用", dropped:"已丢弃", ready:"可用", disabled:"已禁用",
  };
  const tones = {
    "正常":"normal", "可用":"normal", "已接受":"normal", "已使用":"normal",
    "关注":"attention", "需确认":"attention", "待决策":"attention",
    "异常":"abnormal", "阻止":"abnormal", "已拒绝":"abnormal", "已丢弃":"abnormal", "已禁用":"neutral",
  };
  const numericPattern = /^[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?:\s*\/\s*[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)?(?:\s*(?:%|分钟|点|样本|条))?$/i;
  // 数值列的表头与内容共用一个 .numeric 语义：整列数据（忽略空单元格）都是数值时，
  // 表头取同一对齐边；文字、状态、操作列不标记，维持左对齐或居中。
  const markNumericColumns = () => document.querySelectorAll(".table-wrap table").forEach(table => {
    const header = table.tHead?.rows?.[0];
    const rows = table.tBodies?.[0]?.rows;
    if (!header || !rows?.length) return;
    [...header.cells].forEach((th, index) => {
      const cells = [...rows].map(row => row.cells[index]).filter(Boolean);
      const isEmpty = cell => cell.textContent.trim() === "";
      // 至少一个数值单元格，且其余单元格同为数值或为空：避免把只有复选框的空列判成数值列。
      const numeric = cells.some(cell => cell.classList.contains("numeric"))
        && cells.every(cell => cell.classList.contains("numeric") || isEmpty(cell));
      th.classList.toggle("numeric", numeric);
      // 勾选/状态列的内容居中，表头取同一边，保证每列自洽。
      const control = cells.some(cell => cell.querySelector("input[type=checkbox], .status-label"));
      th.classList.toggle("control", control);
    });
  });
  const enhanceTables = () => {
    document.querySelectorAll(".table-wrap td").forEach(cell => {
      const value = cell.textContent.trim();
      const header=cell.closest("table")?.tHead?.rows?.[0]?.cells[cell.cellIndex]?.textContent.trim()||"";
      const statusColumn=/状态|决策|处理结果|质量/.test(header);
      const key = statusColumn ? Object.keys(labels).find(statusKey => value === statusKey) : undefined;
      const localized=statusColumn&&(Object.values(labels).includes(value)||value==="禁用");
      if ((key || localized) && !cell.querySelector('.status-label') && !cell.children.length) {
        cell.dataset.uiEnhanced = "true";
        const badge = document.createElement("span");
        badge.className = `status-label${key?` ${key}`:""}`;
        if(key) badge.dataset.status=key;
        badge.textContent = key?labels[key]:value==="禁用"?"已禁用":value;
        badge.dataset.tone = tones[badge.textContent];
        cell.replaceChildren(badge);
        return;
      }
      if (numericPattern.test(value)) cell.classList.add("numeric");
    });
    markNumericColumns();
  };
  enhanceTables();
  new MutationObserver(enhanceTables).observe(document.body, { childList:true, subtree:true });
});
</script>
"""


_MODEL_RESULTS_STYLE = r"""
<style id="modelResultsStyle">
  .model-quality-copy, #modelTrainingDataQuality > p { max-width:72ch; }
  #modelTrainingConditionDiagnostic { min-width:0; max-width:100%; }
  .training-condition-trend svg { display:block; width:100%; height:auto; }
  #modelTrainingConditionDiagnostic.condition-unavailable > :is(h4, .table-wrap, .training-condition-trend, #modelTrainingConditionHints):not(#modelTrainingAnomalyTitle):not(#modelTrainingAnomalyPeaks) { display:none; }
  .model-training-source-reference { overflow-wrap:anywhere; }
  .model-overview-grid {
    display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:14px;
    align-items:start;
  }
  .model-overview-grid > .chart-card { min-width:0; margin:0; }
  #ablationGroup { width:100%; min-width:0; }
  #explorationPcChart .textpoint text { paint-order:stroke; stroke:white; stroke-width:3px; stroke-linejoin:round; }
  .training-window-editor { display:flex; gap:8px; align-items:end; flex-wrap:wrap; }
  .training-window-editor label { display:grid; gap:4px; flex:1 1 240px; }
  .training-window-editor input { width:100%; }
  #validationWindowTable tr.editing { background:var(--accent-soft); outline:2px solid var(--accent); outline-offset:-2px; }
  #modelContent > section, .model-overview-grid > section { padding:0; }
  /* #modelContent 是单列 Grid，图外按钮会默认拉伸到整行；操作按钮按内容取宽、左对齐。 */
  #modelContent > button { justify-self:start; }
  /* 该审查区位于训练配置分组内部，分组已提供内边距；再叠加自身内边距会让标题相对同组其它小节缩进一格。 */
  .training-configuration > .final-training-review { padding:0; }
  #modelTrainingDataQuality > p { margin:4px 0 8px; }
  #modelQualitySummary { display:block; }
  #modelQualitySummary :is(th,td) { padding:6px 8px; }
  #modelQualitySummary td { text-align:right; font-variant-numeric:tabular-nums; }
  #modelMetrics { align-items:stretch; gap:8px; }
  #modelMetrics > .metric { flex:0 0 auto; flex-direction:column; flex-wrap:nowrap; align-items:flex-start; justify-content:center; gap:4px; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:var(--panel); }
  .model-analysis-column { min-width:0; display:grid; gap:14px; }
  .model-analysis-column > section { min-width:0; padding:0; }
  #modelEngineeringJudgment ul { padding-left:20px; margin:8px 0; }
  #modelEngineeringJudgment li { margin:6px 0; overflow-wrap:anywhere; }
  #varianceChart { display:grid; grid-template-columns:repeat(10,minmax(0,1fr)); align-items:end; gap:8px; padding:12px; }
  #varianceChart .variance-item { display:flex; flex-direction:column; justify-content:end; gap:4px; text-align:center; font-variant-numeric:tabular-nums; font-size:12px; }
  #varianceChart .variance-bar { display:block; height:var(--bar-height); background:var(--line); border-radius:3px 3px 0 0; }
  #varianceChart .selected .variance-bar { background:var(--green); }
  #varianceDetails td { text-align:right; font-variant-numeric:tabular-nums; }
  @media (max-width:760px) { #varianceChart { grid-template-columns:repeat(5,minmax(0,1fr)); } }
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
  .model-projection-grid > .chart-card > .chart { height:auto; min-height:0; flex:0 0 auto; }
  #frozenReplayTrend { height:auto; }
  #frozenReplayTrend svg { display:block; width:100%; height:auto; }
  .model-projection-grid #scoreChart svg,
  .model-projection-grid #loadingChart svg { width:100%; height:auto; display:block; }
  #componentLoadings .component-loading-list {
    display:grid;
    /* 自动列数在 1440px 排到 4 列：卡片 256px、变量名列 54px、名称折 6 行，故固定三档。 */
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:12px;
  }
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
    grid-template-columns:repeat(2,minmax(0,1fr));
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
  #modelStructureComparison .model-energy-table td { padding:6px 8px; }
  #modelStructureComparison .model-energy-table h4 { overflow-wrap:anywhere; }
  #modelStructureComparison .model-energy-table th button { background:none; border:0; border-radius:0; color:inherit; font:inherit; font-weight:inherit; padding:0; text-align:inherit; }
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
  @container (max-width:1100px) {
    .model-overview-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
  @container (max-width:700px) {
    .model-overview-grid { grid-template-columns:minmax(0,1fr); }
  }
  @media (max-width:1200px) {
    .model-projection-grid { grid-template-columns:1fr; }
    .model-projection-grid > .chart-card > .chart { min-height:0; }
    #componentLoadings .component-loading-list { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
  @media (max-width:760px) {
    #modelStructureComparison .model-energy-grid { grid-template-columns:1fr; }
    #modelStructureComparison .lag-energy-table { width:100%; }
    #componentLoadings .component-loading-list { grid-template-columns:minmax(0,1fr); }
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
    training_data_section = training_data_section.replace(
        '<h3>训练窗口</h3>', '<h3>已确认训练窗口</h3>', 1
    )
    return """      <div class="group candidate-manager">
        <div class="group-title" id="candidateManagement">正常状态候选管理</div>
        <div class="help">趋势、状态探索、聚类和条件筛选仅提供证据；候选必须由工程师确认后才能进入训练。</div>
        <div class="row candidate-window-row"><label>开始时间<input id="candidateStart" type="datetime-local"></label><label>结束时间<input id="candidateEnd" type="datetime-local"></label><label>备注<input id="candidateComment" type="text"></label><button id="addManualCandidate" class="secondary" type="button">加入候选窗口</button></div>
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
    buttons = [
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
    ]
    buttons[1] = '\n'.join((
        '        <div class="candidate-workflow-step">',
        buttons[1],
        '          <nav id="candidateSectionNav" class="candidate-section-nav" aria-label="正常状态候选页内导航" hidden>',
        '            <a href="#candidateAnalysis">状态分析</a>',
        '            <a href="#trendEvidence">结果证据</a>',
        '            <a href="#candidateManagement">候选管理</a>',
        '          </nav>',
        '        </div>',
    ))
    buttons = "\n".join(buttons)
    return f"""    <section class="controls workflow-sidebar" aria-label="建模流程">
      <h2 class="workflow-sidebar-title">建模流程</h2>
      <div id="workflowSteps" class="workflow-steps" role="tablist">
{buttons}
      </div>
      <button id="historyCleanupTab" type="button" class="secondary workflow-history-tab" aria-controls="historyCleanupPanel" aria-pressed="false">历史数据清理</button>
    </section>"""


def _history_cleanup_panel_html() -> str:
    return """      <section id="historyCleanupPanel" class="group" aria-labelledby="historyCleanupTitle" hidden>
        <div class="history-cleanup-heading"><h2 id="historyCleanupTitle">历史数据清理</h2><button id="historyCleanupRefresh" type="button" class="secondary">刷新</button></div>
        <div id="historyCleanupStorage" class="metrics"></div>
        <div id="historyCleanupStatus" class="status info" role="status" aria-live="polite"></div>
        <div class="table-wrap"><table>
          <thead><tr><th>选择</th><th>上传文件</th><th>时间</th><th>大小</th><th>关联运行</th><th>保护状态</th></tr></thead>
          <tbody id="historyCleanupRows"><tr><td colspan="6">正在读取历史记录…</td></tr></tbody>
        </table></div>
        <div class="actions"><button id="historyCleanupPreviewButton" type="button" class="secondary" disabled>预览所选文件</button></div>
        <section class="group" aria-labelledby="historyCleanupPreviewTitle">
          <h3 id="historyCleanupPreviewTitle">清理预览</h3>
          <div id="historyCleanupPreview" class="status info" role="status" aria-live="polite">选择文件后查看关联运行、保护原因和预计释放容量。</div>
          <label class="history-cleanup-confirm"><input id="historyCleanupConfirm" type="checkbox" disabled>我已核对预览并确认清理可删除对象</label>
          <button id="historyCleanupExecuteButton" type="button" disabled>执行清理</button>
        </section>
        <div>
          <h3>执行结果</h3>
          <div id="historyCleanupResult" class="status info" role="status" aria-live="polite">尚未执行清理。</div>
        </div>
      </section>"""


def _model_results_content_html() -> str:
    return """        <div id="modelContent" hidden>
          <h3>训练结果概览</h3>
          <button id="trainingExceedanceButton" type="button" class="secondary">超限诊断</button>
          <dialog id="trainingExceedanceDialog" aria-labelledby="exceedanceTitle">
            <style>
              #trainingExceedanceDialog { box-sizing:border-box; width:90vw; height:90vh; max-width:90vw; max-height:90vh; padding:16px; overflow:auto; border:1px solid var(--line-control); border-radius:8px; color:var(--text); }
              #trainingExceedanceDialog::backdrop { background:rgba(15,23,42,.45); }
              #trainingExceedanceDialog .exceedance-header { display:flex; justify-content:space-between; align-items:center; gap:12px; background:none; color:inherit; padding:0; }
              #trainingExceedanceDialog .exceedance-header h3 { margin:0; }
              #trainingExceedanceDialog .exceedance-controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:8px 0; }
              /* 选中态沿用同类模式按钮的既有约定（.candidate-tool-tab.active / .exploration-region-tools .active）：
                 主色填充 + 白字，比单改 1px 边框色更明确，也不只依赖颜色深浅。 */
              #trainingExceedanceDialog .exceedance-modes button[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
              #trainingExceedanceDialog .exceedance-compare-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
              #trainingExceedanceDialog .exceedance-compare-group { min-width:0; border:1px solid var(--line); border-radius:6px; padding:10px; }
              #trainingExceedanceDialog .exceedance-compare-group h5 { margin:0 0 8px; }
              #trainingExceedanceDialog .exceedance-compare-group input { width:100%; }
              #trainingExceedanceDialog .exceedance-pair-plot { height:300px; min-width:0; }
              #trainingExceedanceDialog label { display:flex; flex-direction:column; gap:4px; }
              #trainingExceedanceDialog input { width:min(100%,var(--field-datetime-width,240px)); }
              #trainingExceedanceDialog .exceedance-plot { height:250px; min-width:0; }
              #trainingExceedanceDialog .table-wrap { overflow-x:auto; }
              #trainingExceedanceDialog h4 { margin:12px 0 6px; }
              @media (max-width:640px) { #trainingExceedanceDialog .exceedance-compare-grid { grid-template-columns:minmax(0,1fr); } }
            </style>
            <header class="exceedance-header"><h3 id="exceedanceTitle">T² / SPE 超限诊断</h3><button id="exceedanceClose" type="button" class="secondary">关闭</button></header>
            <div id="exceedanceStatus" role="status" aria-live="polite"></div>
            <div class="exceedance-controls exceedance-modes" aria-label="趋势框选模式">
              <span>框选模式</span>
              <button id="exceedanceModeSingle" type="button" class="secondary" aria-pressed="true">单区间诊断</button>
              <button id="exceedanceModeAnomaly" type="button" class="secondary" aria-pressed="false">选择异常区间</button>
              <button id="exceedanceModeReference" type="button" class="secondary" aria-pressed="false">选择参考区间</button>
            </div>
            <div class="exceedance-controls">
              <label>开始时间<input id="exceedanceStart" type="text" placeholder="ISO 时间，可含微秒及时区"></label>
              <label>结束时间<input id="exceedanceEnd" type="text" placeholder="ISO 时间，可含微秒及时区"></label>
              <button id="exceedanceApply" type="button">分析区间</button>
              <button id="exceedanceReset" type="button" class="secondary">恢复全局视图</button>
            </div>
            <div class="help">横向框选分析区间；工具栏支持缩放和平移。贡献表示统计偏离来源，不表示工艺根因。</div>
            <div id="exceedanceT2" class="exceedance-plot"></div><div id="exceedanceSpe" class="exceedance-plot"></div>
            <h4>区间统计</h4><div id="exceedanceStatistics" class="table-wrap"></div>
            <div class="exceedance-controls"><h4>变量贡献 Top10</h4><label>统计量<select id="exceedanceMetric"><option value="t2">T²</option><option value="spe">SPE</option></select></label></div>
            <div id="exceedanceContributions" class="table-wrap"></div>
            <section id="exceedanceComparePanel">
              <h4>SPE 两区间比较</h4>
              <div class="exceedance-compare-grid">
                <div class="exceedance-compare-group">
                  <h5>高 SPE 异常区间</h5>
                  <div class="exceedance-controls"><label>开始时间<input id="exceedanceAnomalyStart" type="text" placeholder="ISO 时间"></label><label>结束时间<input id="exceedanceAnomalyEnd" type="text" placeholder="ISO 时间"></label><button id="exceedanceAnomalyClear" type="button" class="secondary">清除</button></div>
                </div>
                <div class="exceedance-compare-group">
                  <h5>低 SPE 参考区间</h5>
                  <div class="exceedance-controls"><label>开始时间<input id="exceedanceReferenceStart" type="text" placeholder="ISO 时间"></label><label>结束时间<input id="exceedanceReferenceEnd" type="text" placeholder="ISO 时间"></label><button id="exceedanceReferenceClear" type="button" class="secondary">清除</button></div>
                </div>
              </div>
              <div class="exceedance-controls"><label>负荷 Tag<select id="exceedanceLoadTag"><option value="">请选择数值型 Tag</option></select></label><button id="exceedanceCompareApply" type="button">比较两区间</button></div>
              <div id="exceedanceCompareStatus" role="status" aria-live="polite"></div>
              <div id="exceedanceCompareStatistics" class="table-wrap"></div>
              <h5>异常组 SPE 残差²均值与低 SPE 参考残差能量</h5>
              <div id="exceedanceCompareContributions" class="table-wrap"></div>
              <div class="help">参考列表示低 SPE 样本的残差能量分解。负荷范围重叠不等于同负荷，需工程师核对分布。</div>
            </section>
            <section id="exceedanceRawSection" hidden><h4 id="exceedanceRawTitle">原始趋势</h4><div id="exceedanceRawStatus" role="status"></div><div id="exceedanceRaw" class="exceedance-plot"></div></section>
            <section id="exceedanceCompareRawSection" hidden><h4 id="exceedanceCompareRawTitle">异常与参考原始趋势</h4><div id="exceedanceCompareRawStatus" role="status"></div><div id="exceedanceCompareRaw" class="exceedance-pair-plot"></div><div id="exceedanceCompareLoadTitle"></div><div id="exceedanceCompareLoadStatus" role="status"></div><div id="exceedanceCompareLoadRaw" class="exceedance-pair-plot"></div></section>
          </dialog>
          <section class="chart-card"><h3>模型摘要</h3><div id="modelMetrics" class="metrics"></div></section>
          <div class="model-overview-grid">
            <section class="chart-card model-quality-conclusion" aria-labelledby="modelQualityTitle">
              <h3 id="modelQualityTitle">模型质量判断</h3>
              <div id="modelLifecycleNotice" class="notice model-quality-copy"></div>
              <div id="modelEngineeringJudgment" class="model-quality-copy" aria-live="polite"></div>
              <div id="modelQualityNotice" class="help model-quality-copy"></div>
            </section>
            <div class="model-analysis-column">
            <section class="chart-card model-structure-summary">
              <h3>模型结构简述</h3>
              <div id="modelProjectionSummary" class="metric"></div>
            </section>
            <section class="chart-card">
              <h3>统计质量</h3>
              <div id="modelQualitySummary" class="chart-grid" aria-live="polite"></div>
            </section>
            </div>
          </div>
          <section class="chart-card"><h3>主元解释率</h3><p class="help">前 10 个主元；绿色为已保留主元，灰色为未保留主元。完整明细列出保留状态。</p><div id="varianceChart" class="variance"></div><details><summary>完整主元明细（含累计解释率）</summary><div id="varianceDetails" class="table-wrap" tabindex="0"></div></details></section>
          <section id="componentLoadings" class="chart-card">
            <h3>主元载荷分析</h3>
            <div class="help">仅显示前 6 个主元。载荷来自实际训练模型的 PCA 主元。同一原始变量的全部 Lag 先聚合为一个载荷强度 L_agg = √(Σ loading²)，聚合后不再有正负方向。按聚合强度从大到小显示前 10 个原始变量，平方载荷占比以该主元全部变量的平方载荷总和为分母，不对前 10 项重新归一化，不等同于 T²/SPE 异常贡献。</div>
            <div id="componentLoadingsContent"><div class="empty">完成 DPCA 训练后显示各主元组成。</div></div>
          </section>
          <section class="chart-card">
            <h3>训练数据组成</h3><div id="modelTrainingDataQuality"></div>
            <h4>训练窗口与连续段</h4><div id="trainingWindowSummary" class="table-wrap"></div>
            <div id="trainingQualityWarnings" class="hint"></div>
          </section>
          <section id="modelTrainingConditionDiagnostic" class="chart-card">
            <h3>训练工况诊断</h3>
            <div id="modelTrainingConditionMessage" class="help"></div>
            <div id="modelTrainingConditionGroups" class="table-wrap"></div>
            <h4>训练工况与统计量趋势</h4>
            <div id="modelTrainingConditionTrend" class="training-condition-trend"></div>
            <h4 id="modelTrainingAnomalyTitle">训练异常峰值</h4>
            <div class="help">Top10 连续超限事件峰值，仅用于定位统计异常时间，不代表工艺异常或根因。</div>
            <div id="modelTrainingAnomalyPeaks" class="table-wrap"></div>
            <h4>稳定工况区 / 工况切换附近</h4>
            <div id="modelTrainingConditionSwitch" class="table-wrap"></div>
            <div id="modelTrainingConditionHints" class="help"></div>
          </section>
          <h3>详细趋势和图表</h3>
          <div class="chart-grid">
            <div class="chart-card"><h3>训练期 T²</h3><div id="t2Chart" class="chart"></div></div>
            <div class="chart-card"><h3>训练期 SPE/Q</h3><div id="speChart" class="chart"></div></div>
          </div>
          <div class="legend"><span><i class="swatch" style="background:var(--accent)"></i>统计量</span><span><i class="swatch" style="background:var(--attention)"></i>95% 边界</span><span><i class="swatch" style="background:var(--abnormal)"></i>99% 边界</span></div>
          <div class="model-projection-grid">
            <div class="chart-card"><div class="chart-card-head"><h3>主元得分 PC1 / PC2</h3><div class="exploration-region-tools"><label>着色方式<select id="scoreColorMode"><option value="default">默认</option><option value="time">时间</option><option value="performance">性能状态</option></select></label></div></div><div id="scoreChart" class="chart"></div><p id="scoreChartNote" class="chart-note">默认按 T²/SPE 综合状态着色；切换着色方式只改变视觉编码，PC 坐标、样本数量和筛选范围不变。</p></div>
            <div class="chart-card"><h3>PC1 / PC2 原始Tag聚合载荷图</h3><div class="help">每条连线从原点连接到一个原始Tag的PC1/PC2聚合载荷；全部Lag按带符号L2能量聚合。连线方向和长度用于解释模型结构，不等同于异常贡献或工艺根因。</div><div id="loadingChart" class="chart empty">完成DPCA训练后显示载荷图。</div></div>
          </div>
          <h3>模型诊断</h3>
          <section id="modelStructureComparison" class="chart-card">
            <h3>模型结构与参数比较</h3>
            <div class="help">诊断用于辅助工程师选择模型结构，不能替代独立验证；不会自动评分、推荐、验证或改变模型状态。</div>
            <div id="singleModelDiagnostic" class="help">完成正常状态候选模型训练后显示结构诊断。</div>
            <div id="modelCandidateStatus" class="help" role="status" aria-live="polite">正在加载候选模型…</div>
            <label class="secondary">选择已训练候选模型<select id="modelComparisonRuns" multiple size="5" aria-label="候选模型比较"></select></label>
            <div class="actions"><button id="compareModelsButton" type="button">比较所选候选模型</button><button id="deleteModelsButton" class="danger" type="button">删除所选候选模型</button></div>
            <div id="modelComparisonResult" class="help">比较只读取已保存的正常状态候选模型包。</div>
          </section>
          <h3>模型文件</h3>
          <div class="actions"><a id="modelDownload" class="download" href="#">下载模型包</a><button id="modelingSnapshotButton" class="secondary" type="button">查看建模快照</button></div>
          <div id="modelingSnapshot" class="table-wrap" hidden></div>
        </div>"""


def _build_data_stage(config_panel: str, upload_group: str, tag_group: str) -> str:
    """Keep the data source, Tag workspace, and raw inspection in task order."""
    config_panel = config_panel.replace(
        '<div class="inner-tabs">\n',
        f'<h3>数据源与时间轴</h3>\n{upload_group}\n'
        '        <h3>Tag 配置工作区</h3>\n'
        '        <div class="inner-tabs">\n',
        1,
    ).replace(
        '<h3 id="selectedTagTitle">',
        f'<div class="tag-workspace">\n{tag_group}\n'
        '          <div class="tag-detail"><h3 id="selectedTagTitle">',
        1,
    ).replace(
        '        </div>\n        <div id="qualityPanel"',
        '          </div>\n        </div>\n        </div>\n        <div id="qualityPanel"',
        1,
    )
    config_panel = config_panel.replace(
        '        <div class="inner-tabs">\n'
        '          <button class="inner-tab active" data-inner="engineeringPanel">工程配置</button>\n'
        '          <button class="inner-tab" data-inner="qualityPanel">基础数据检查</button>\n'
        '        </div>\n',
        '',
        1,
    ).replace('id="qualityPanel" class="inner-panel"', 'id="qualityPanel" class="inner-panel active"', 1)
    return config_panel


def _build_candidate_stage(parameter_group: str, training_data_section: str, trend_panel: str, state_panel: str, cluster_panel: str, performance_panel: str) -> str:
    """Mount four peer discovery tools around one analysis range."""
    analysis_range = _div_containing_unique_field(cluster_panel, "analysisStart", "候选分析范围")
    cluster_panel = cluster_panel.replace(analysis_range, "", 1)
    conditions_start = _unique_anchor_index(cluster_panel, '<div class="sub-title">性能条件筛选</div>', "条件筛选")
    conditions_end = _unique_anchor_index(cluster_panel, '\n        </div>\n        <div id="clusterEmpty"', "聚类配置结束")
    conditions = cluster_panel[conditions_start:conditions_end]
    cluster_panel = cluster_panel[:conditions_start] + cluster_panel[conditions_end:]
    conditions = conditions.replace('性能条件筛选', '多变量工程条件筛选', 1).replace(
        '全部条件按AND组合；性能列只用于筛选，不会自动进入PCA。',
        '全部条件按 AND 组合。筛选列会取消建模勾选，不自动进入 PCA；工程师可随后调整 Tag 角色和勾选。',
        1,
    )
    conditions = conditions.replace(
        '<div id="performanceConditions"',
        '<div class="row"><label>筛选范围<select id="performanceScope">'
        '<option value="all" selected>全部建模资格数据</option>'
        '<option value="candidate_windows">已有候选窗口</option>'
        '<option value="cluster">指定工况组</option></select></label></div>'
        '<label id="performanceParentsLabel" hidden>父候选窗口（可多选）'
        '<select id="performanceParents" multiple size="5"></select></label>'
        '<label id="performanceClustersLabel" hidden>工况组（可多选）'
        '<select id="performanceClusters" multiple size="5"></select></label>'
        '<div id="performanceScopeSummary" class="help">筛选全部建模资格数据。</div>'
        '<div id="performanceConditions"',
        1,
    )
    performance_panel = performance_panel.replace(
        '<div id="performancePanel" class="panel">',
        '<div id="performancePanel" class="panel">\n'
        '        <div class="group">\n          ' + conditions.strip() + '\n        </div>',
        1,
    )
    cluster_panel = cluster_panel.replace('<h3>结果概览</h3>', '', 1)
    cluster_panel = cluster_panel.replace('<div id="clusterEmpty"', '<h3 id="clusterEvidence">结果证据</h3>\n        <div id="clusterEmpty"', 1)
    performance_panel = performance_panel.replace('<div id="performanceEmpty"', '<h3 id="performanceEvidence">结果证据</h3>\n        <div id="performanceEmpty"', 1)
    candidate_panels = [
        panel.replace('class="panel"', 'class="candidate-tool-panel"', 1)
        for panel in (trend_panel, state_panel)
    ]
    candidate_panels[0] = candidate_panels[0].replace(
        'class="candidate-tool-panel"', 'class="candidate-tool-panel active"', 1
    )
    other_candidate_panels = [
        panel.replace('class="panel"', 'class="candidate-tool-panel"', 1)
        for panel in (cluster_panel, performance_panel)
    ]
    candidate_panel = "\n".join(
        (
            '      <div id="candidatePanel" class="panel">',
            '        <div class="candidate-tool-tabs" id="candidateAnalysis" role="tablist">',
            '          <button type="button" class="candidate-tool-tab" data-panel="candidatePreparationPanel" role="tab" aria-selected="false">筛选准备</button>',
            '          <button type="button" class="candidate-tool-tab active" data-panel="trendPanel" role="tab" aria-selected="true">趋势选择</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="stateExplorationPanel" role="tab" aria-selected="false">状态探索</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="clusterPanel" role="tab" aria-selected="false">聚类辅助</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="performancePanel" role="tab" aria-selected="false">条件筛选</button>',
            '        </div>',
            '        <div id="candidatePreparationPanel" class="candidate-tool-panel">',
            """        <div id="modelingEligibility" class="group">
          <div class="group-title">筛选条件</div>
          <p class="help">只影响离线探索、候选与训练资格。比较条件支持 &gt;、≥、&lt;、≤、=、≠；也可切换为包含端点的闭区间。保留条件按 AND；排除组内按 AND、组间按 OR。最终样本须满足全部保留条件且未命中任何排除组。条件列不会自动成为 PCA 输入。</p>
          <div class="screening-rules">
            <div><div class="actions"><h3>保留条件</h3><button id="addEligibilityKeep" type="button" class="secondary">添加保留条件</button></div><div id="eligibilityKeepConditions"></div></div>
            <div><div class="actions"><h3>排除条件</h3><button id="addEligibilityExcludeGroup" type="button" class="secondary">添加排除规则组</button></div><div id="eligibilityExcludeGroups"></div></div>
          </div>
          <div id="eligibilitySummary" class="screening-summary" aria-live="polite">检查数据后显示资格摘要；无规则时全部样本具备资格。</div>
          <div class="sub-title">分析时间范围</div>""",
            '          <div class="candidate-analysis-range">' + analysis_range + '</div>',
            '          <div class="help">状态探索、聚类辅助和条件筛选共用此范围；趋势选择可将浏览窗口设为这里的分析范围。修改后请重新运行分析。</div></div>',
            parameter_group,
            '        </div>',
            '        <div id="candidateConfigSummary" class="candidate-config-summary"><span></span><button type="button" class="secondary">修改配置</button></div>',
            *candidate_panels,
            *other_candidate_panels,
            _candidate_manager_html(training_data_section),
            '      </div>',
        )
    )
    return candidate_panel


def _build_validation_and_release_stages(validation_panel: str) -> tuple[str, str]:
    """Place engineering decisions after evidence and freezing after validation."""
    validated_download = _element_with_unique_id(
        validation_panel, "validatedModelDownload", "a", "已验证模型下载入口"
    )
    freeze_box = _div_containing_unique_field(
        validation_panel, "frozenModelId", "冻结与部署入口"
    )
    frozen_download = _element_with_unique_id(
        freeze_box, "frozenModelDownload", "a", "冻结模型下载"
    )
    deployment_download = _element_with_unique_id(
        freeze_box, "deploymentModelDownload", "a", "部署模型下载"
    )
    freeze_box = freeze_box.replace(frozen_download, "", 1).replace(deployment_download, "", 1)
    full_freeze_box = _div_containing_unique_field(
        validation_panel, "frozenModelId", "冻结与部署入口"
    )
    validation_panel = validation_panel.replace(validated_download, "", 1).replace(full_freeze_box, "", 1)
    decision_box = _div_containing_unique_field(
        validation_panel, "recordValidationDecision", "工程师结论"
    )
    validation_panel = validation_panel.replace(decision_box, "", 1)
    validation_button = _element_with_unique_id(
        validation_panel, "validateButton", "button", "执行独立验证"
    )
    validation_panel = validation_panel.replace(validation_button, "", 1)
    validation_table_end = '</tbody></table></div>\n        <div id="validationEmpty"'
    if validation_panel.count(validation_table_end) != 1:
        raise ValueError("无法固定验证窗口列表")
    validation_panel = validation_panel.replace(
        validation_table_end,
        '</tbody></table></div>\n        <div class="actions">'
        + validation_button.replace("回放独立验证期", "执行独立验证")
        + '</div>\n        <div id="validationEmpty"',
        1,
    )
    validation_panel = validation_panel.replace(
        '<div id="validationPanel" class="panel">',
        '<div id="validationPanel" class="panel">\n        <h3>① 验证设置</h3>',
        1,
    ).replace(
        '<div id="validationContent" hidden>\n',
        '<div id="validationContent" hidden>\n          <h3>② 验证证据</h3>\n',
        1,
    )
    validation_downloads = _div_containing_unique_field(
        validation_panel, "scoresDownload", "验证证据下载"
    )
    validation_panel = validation_panel.replace(
        validation_downloads,
        validation_downloads + '\n          <h3>③ 工程师结论</h3>\n          ' + decision_box,
        1,
    )
    release_panel = f"""      <div id="releasePanel" class="panel">
        <div id="releaseEmpty" class="empty">模型通过独立验证和工程师确认后，可在此冻结与部署。</div>
        <div id="releaseContent" hidden>
          <h3>① 已验证模型</h3>
          <div class="notice">冻结与部署导出沿用现有流程；冻结状态表示工程冻结，不表示已经部署。</div>
          <div class="actions">{validated_download}</div>
          <h3>② 工程冻结</h3>
{freeze_box}
          <h3>③ 冻结结果</h3>
          <div class="help">冻结后可下载冻结模型包与部署模型包。</div>
          <div class="actions">{frozen_download}{deployment_download}</div>
          <section id="frozenReplay" hidden>
            <h3>④ 冻结模型历史回放</h3>
            <div class="notice">历史回放仅检查冻结模型在历史数据上的表现，不属于独立验证，也不改变模型状态。</div>
            <div class="validation-box"><label>回放开始<input id="frozenReplayStart" type="datetime-local"></label><label>回放结束<input id="frozenReplayEnd" type="datetime-local"></label><button id="frozenReplayButton" type="button" disabled>执行冻结模型回放</button></div>
            <div id="frozenReplaySummary" class="help">请先完成工程冻结，再选择历史区间执行回放。</div>
            <div class="chart-grid"><div class="chart-card"><h3>T² / SPE 限值比趋势</h3><div id="frozenReplayTrend" class="chart empty">尚无回放结果。</div></div><div class="chart-card"><h3>状态统计</h3><div id="frozenReplayStatus" class="help">尚无回放结果。</div></div></div>
            <div class="actions"><a id="frozenReplayScoresDownload" class="download secondary" href="#" hidden>下载完整评分 CSV</a><a id="frozenReplaySummaryDownload" class="download secondary" href="#" hidden>下载回放摘要</a><a id="frozenReplayContributionsDownload" class="download secondary" href="#" hidden>下载贡献记录</a></div>
          </section>
        </div>
      </div>"""
    return validation_panel, release_panel


def _build_model_stage(model_panel: str, model_configuration_group: str) -> str:
    existing_model_results = _required_html_match(
        r'        <div id="modelContent" hidden>.*?\n        </div>',
        model_panel,
        "模型结果区域",
    ).group()
    model_panel = model_panel.replace(existing_model_results, _model_results_content_html(), 1)
    model_panel_lines = model_panel.rsplit("\n", 1)
    if len(model_panel_lines) != 2 or model_panel_lines[1] != "      </div>":
        raise ValueError("无法固定模型训练结果区域")
    model_panel_content = model_panel_lines[0].split("\n", 1)
    if len(model_panel_content) != 2:
        raise ValueError("无法固定模型训练页面结构")
    model_panel = f"""{model_panel_content[0]}
        <div class="candidate-tool-tabs" role="tablist" aria-label="模型训练工具">
          <button id="trainingMainTab" type="button" class="training-tool-tab active" role="tab" aria-selected="true" aria-controls="trainingMainPanel">训练配置与结果</button>
          <button id="redundancyAblationTab" type="button" class="training-tool-tab" role="tab" aria-selected="false" aria-controls="redundancyAblationPanel">冗余消融评估</button>
        </div>
        <div id="trainingMainPanel" role="tabpanel" aria-labelledby="trainingMainTab">
{model_configuration_group}
{model_panel_content[1]}
        </div>
        <div id="redundancyAblationPanel" role="tabpanel" aria-labelledby="redundancyAblationTab" hidden>
          <h3>冗余消融评估</h3>
          <p class="help">相关性仅用于生成候选，变量去留由独立验证证据决定。每个方案只剔除一个原始 Tag 及其全部 Lag 特征；训练条件沿用基准模型。</p>
          <div id="ablationStatus" class="empty" role="status" aria-live="polite">请先完成正常状态候选模型训练，再加载冗余候选组。</div>
          <div id="ablationPolicy" class="help"></div>
          <div class="actions"><button id="ablationCandidatesButton" type="button" disabled>加载冗余候选组</button><button id="ablationEvaluateButton" type="button" class="secondary" disabled>执行逐 Tag 消融</button><button id="ablationApplyButton" type="button" class="secondary" disabled>应用为待训练 Tag 配置</button></div>
          <label>冗余候选组<select id="ablationGroup" disabled><option value="">尚未加载</option></select></label>
          <p class="help">候选组按高相关边连通，不代表所有成员两两高度相关。勾选待剔除 Tag，或锁定暂不参与剔除的关键 Tag。</p>
          <div id="ablationTags" class="table-wrap"></div>
          <div id="ablationEvidence" class="table-wrap"></div>
          <details><summary>数据质量及不可计算原因</summary><div id="ablationDiagnostics" class="table-wrap"></div></details>
          <p class="help">验证窗口沿用「模型验证」中的已配置窗口。未提供异常验证时，异常检出能力尚未验证；结构简化或解释率增加不足以支持删除。</p>
          <div id="ablationResults" class="table-wrap"></div>
          <div id="ablationWarnings" class="help"></div>
        </div>
      </div>"""
    return model_panel


def _assemble_workbench(status_area: str, config_panel: str, candidate_panel: str, model_panel: str, validation_panel: str, release_panel: str) -> str:
    return "\n".join(
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
            _history_cleanup_panel_html(),
            "    </section>",
            "  </main>",
        )
    )



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
        '        <div class="group-title">1. 历史数据</div>\n', '', 1
    )
    tag_group = (tag_marker + tag_group).replace(
        '<div class="group-title">2. 建模 Tag</div>',
        '<div class="group-title">建模 Tag</div>',
        1,
    ).rstrip()
    parameter_group = (parameters_marker + parameter_group).rstrip()
    candidate_anchor = '        <div class="row"><label>开始时间<input id="candidateStart"'
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
        '        <div class="group-title">分析参数</div>\n'
        '        <div class="help">这些参数同时用于趋势浏览、状态探索、独立聚类、建模质量检查和正式训练；正式训练沿用同一套取值，不再重复配置。</div>\n'
        '        <div class="sub-title">基础分析参数</div>\n'
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
    quality_section = training_parameter_tail[quality_start:training_action_start].rstrip().replace(
        '<h3>建模质量检查</h3>', '<h3>③ 建模质量检查</h3>', 1
    )
    training_actions_section = training_parameter_tail[training_action_start:].rstrip()
    model_configuration_rows = (
        '        <div class="training-parameter-grid">\n'
        '          <div class="model-name-field">\n'
        f'            {_label_for_unique_field(training_parameter_tail, "modelName", "模型名称")}\n'
        '          </div>\n'
        f'          {_label_for_unique_field(training_parameter_tail, "varianceThreshold", "累计解释率")}\n'
        f'          {_label_for_unique_field(training_parameter_tail, "components", "主元数")}\n'
        f'          {_label_for_unique_field(training_parameter_tail, "changeReason", "本轮建模说明")}\n'
        '        </div>\n'
    )
    training_data_summary = (
        '        <div id="modelTrainingDataSummary" class="notice">'
        '需重新执行建模质量检查后显示训练数据摘要。'
        '</div>\n'
        '        <div id="modelPreprocessingSummary" class="help">共享预处理配置沿用候选页取值。</div>\n'
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
        '        <h3>① 训练准备</h3>\n'
        + training_data_summary
        + preprocessing_preview_section
        + '        <h3>② PCA / DPCA 模型配置</h3>\n'
        + model_configuration_rows
        + quality_section
        + '        <h3>④ 正式训练</h3>\n'
        + training_actions_section
    )
    exploratory_button = _element_with_unique_id(
        model_configuration_group, "trainExploratoryButton", "button", "探索模型入口"
    )
    exploratory_tools = (
        '        <details class="advanced-parameters exploratory-model-tools">\n'
        '          <summary>高级操作：建立探索模型</summary>\n'
        '          <div class="help">探索模型仅用于兼容保留的状态空间/聚类辅助路径，不属于正常状态主流程。</div>\n'
        f'          <div class="actions">{exploratory_button}</div>\n'
        '        </details>\n'
    )
    group_close = "      </div>"
    if not model_configuration_group.endswith(group_close):
        raise ValueError("无法固定Web工作台结构：探索模型入口")
    model_configuration_group = (
        model_configuration_group[: -len(group_close)].replace(exploratory_button, "", 1).rstrip()
        + "\n"
        + exploratory_tools
        + group_close
    )
    if 'id="candidateWindows"' in parameter_group + model_configuration_group:
        raise ValueError("无法固定候选窗口或训练参数区域")
    status_area = (status_marker + status_area).replace(
        'class="status info" role="status" aria-live="polite"',
        'class="status info operation-log" role="status" aria-live="polite" aria-label="运行日志"',
        1,
    ).rstrip()

    panel_ids = (
        "configPanel",
        "stateExplorationPanel",
        "trendPanel",
        "modelPanel",
        "clusterPanel",
        "performancePanel",
        "validationPanel",
    )
    panels = {
        panel_id: _div_containing_unique_field(results, panel_id, panel_id).rstrip()
        for panel_id in panel_ids
    }
    config_panel = panels["configPanel"]
    state_panel = panels["stateExplorationPanel"]
    trend_panel = panels["trendPanel"]
    model_panel = panels["modelPanel"]
    cluster_panel = panels["clusterPanel"]
    performance_panel = panels["performancePanel"]
    validation_panel = panels["validationPanel"]

    state_exploration_button = _element_with_unique_id(
        state_panel, "stateExplorationButton", "button", "运行状态探索按钮"
    )
    state_panel = state_panel.replace(state_exploration_button, "", 1)
    state_exploration_button = state_exploration_button.replace("运行状态探索", "运行状态筛选")
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
        '        </div>\n'
        f'        <div class="actions screening-execute">{state_exploration_button}</div>\n'
        '        <h3 id="explorationEvidence">结果预览</h3>\n'
        '        <div id="explorationEmpty"',
        1,
    )
    for field_id in ("explorationStart", "explorationEnd"):
        state_panel = state_panel.replace(
            _label_for_unique_field(state_panel, field_id, "重复探索时间范围"), "", 1
        )
    state_panel = state_panel.replace(
        '<div class="group-title">状态探索工作台</div>',
        '<div class="group-title">状态识别参数</div>',
        1,
    )
    state_panel = state_panel.replace(
        '<div class="exploration-controls">',
        '<div class="sub-title">状态筛选配置</div>\n          <div class="exploration-controls">',
        1,
    ).replace(
        '<div class="exploration-controls performance-controls">',
        '<div class="sub-title">性能评价（可选）</div>\n'
        '          <div class="help">性能 Tag 仅用于状态探索后的后验评价，不参与状态探索 PCA。</div>\n'
        '          <div class="exploration-controls performance-controls">',
        1,
    )
    state_panel = state_panel.replace(
        '探索结果仅用于运行状态浏览和候选窗口比较。',
        '点击“运行状态筛选”后查看完整探索证据；仅使用资格通过的数据，结果只提供候选，不自动判定正常状态。',
        1,
    )
    state_panel = state_panel.replace('<h3>结果概览</h3>', '', 1)
    state_panel = state_panel.replace(
        '<div id="explorationClusterQuality"></div>',
        '<div id="explorationClusterQuality"></div>\n'
        '          <div id="explorationWarnings" class="compact-list" aria-live="polite"></div>\n'
        '          <div id="explorationQualityDetails"></div>',
        1,
    )
    state_panel = state_panel.replace(
        '<div id="explorationWarnings" class="compact-list"><span class="help">暂无结构化告警。</span></div>', '', 1
    )

    config_panel = _build_data_stage(config_panel, upload_group, tag_group)
    candidate_panel = _build_candidate_stage(parameter_group, training_data_section, trend_panel, state_panel, cluster_panel, performance_panel)
    validation_panel, release_panel = _build_validation_and_release_stages(validation_panel)
    model_panel = _build_model_stage(model_panel, model_configuration_group)
    static_main = _assemble_workbench(status_area, config_panel, candidate_panel, model_panel, validation_panel, release_panel)
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
    result = result.replace('<body>', '<body>' + _GROUP_PROFILE_DIALOG, 1)
    result = result.replace('</head>', _GROUP_PROFILE_DIALOG_STYLE + '</head>', 1)
    result = result.replace('<div id="dpTrendChart"', '<h3 id="trendEvidence">结果证据</h3>\n    <div id="dpTrendChart"', 1)
    # Only the final workbench presentation changes; payloads and calculations stay shared.
    result = result.replace(
        '  node.textContent=`${scope}：原始样本 ${summary.original_samples}；保留条件通过 ${summary.keep_pass_samples}；排除规则命中 ${summary.exclude_hit_samples}；最终合格 ${summary.eligible_samples}；合格占比 ${(summary.eligible_share*100).toFixed(1)}%；资格筛选后连续段 ${summary.segment_count}。${summary.eligible_samples?"":"没有合格样本，请调整资格规则。"}`;',
        '  node.classList.toggle("warning",!summary.eligible_samples);\n'
        '  node.textContent=`${scope}：筛选后 ${summary.eligible_samples.toLocaleString()} 点 · 保留条件通过 ${summary.keep_pass_samples.toLocaleString()} · 排除 ${summary.exclude_hit_samples.toLocaleString()} · 合格占比 ${(summary.eligible_share*100).toFixed(1)}% · 连续段 ${summary.segment_count}。${summary.eligible_samples?"":"没有合格样本，请调整资格规则。"}`;\n'
        '  node.dataset.compact=`资格：筛选后 ${summary.eligible_samples.toLocaleString()} 点 · 合格占比 ${(summary.eligible_share*100).toFixed(1)}% · 连续段 ${summary.segment_count}${summary.eligible_samples?"":"；没有合格样本"}`;',
        1,
    )
    summary_refresh = _required_html_match(
        r'async function refreshEligibilitySummary\(\).*?(?=function eligibilityChanged)',
        result, "资格摘要刷新",
    ).group()
    result = result.replace(summary_refresh, '''let eligibilitySummaryRevision=0;
function scheduleEligibilitySummary() {
  eligibilitySummaryRevision+=1;
  clearTimeout(eligibilitySummaryTimer);
  const node=el("eligibilitySummary");
  if(node&&state.fileId&&state.inspection) { node.classList.toggle("warning",false); if(node.dataset) node.dataset.compact=""; node.textContent="正在更新…"; }
  eligibilitySummaryTimer=setTimeout(refreshEligibilitySummary,250);
}
async function refreshEligibilitySummary() {
  if(!state.fileId||!state.inspection||!el("eligibilitySummary")) return;
  const revision=++eligibilitySummaryRevision, rulesRevision=eligibilityRevision;
  try {
    const payload={...commonPayload(),candidate_start:el("analysisStart").value,candidate_end:el("analysisEnd").value};
    const data=await api("/api/modeling-eligibility",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(revision===eligibilitySummaryRevision&&rulesRevision===eligibilityRevision) renderEligibilitySummary(data.summary,"分析时间范围");
  } catch(error) {
    if(revision===eligibilitySummaryRevision&&rulesRevision===eligibilityRevision) { const node=el("eligibilitySummary"); node.classList.toggle("warning",true); if(node.dataset) node.dataset.compact=""; node.textContent=error.message; }
  }
}
''', 1)
    result = result.replace('  clearTimeout(eligibilitySummaryTimer); eligibilitySummaryTimer=setTimeout(refreshEligibilitySummary,250);', '  scheduleEligibilitySummary();', 1)
    result = result.replace(
        '[value.querySelector("input"),minimum.querySelector("input"),maximum.querySelector("input")].forEach(input=>input.addEventListener("input",()=>renderEligibilityConditionPreview(row)));',
        '[value.querySelector("input"),minimum.querySelector("input"),maximum.querySelector("input")].forEach(input=>input.addEventListener("input",()=>renderEligibilityConditionPreview(row)));\n'
        '  [value,minimum,maximum].forEach(label=>label.querySelector("input").addEventListener("input",scheduleEligibilitySummary));',
        1,
    )
    result = result.replace(
        'el("refreshEligibilitySummary")?.addEventListener("click",refreshEligibilitySummary);',
        '["analysisStart","analysisEnd"].forEach(id=>["input","change"].forEach(event=>el(id).addEventListener(event,scheduleEligibilitySummary)));',
        1,
    )
    result = result.replace('renderPerformanceConditions(data.numeric_columns); refreshEligibilitySummary();', 'renderPerformanceConditions(data.numeric_columns); scheduleEligibilitySummary();', 1)
    result = result.replace('$("analysisEnd").value = $("dpTrendEnd").value;', '$("analysisEnd").value = $("dpTrendEnd").value;\n    globalThis.scheduleEligibilitySummary?.();', 1)
    renderer = _required_html_match(
        r'function renderClusterQuality\(.*?(?=function renderVariableDiagnostics)',
        result, "聚类质量展示",
    ).group()
    screening_renderer = renderer.replace(
        'function renderClusterQuality(container, quality, perspective) {',
        '''function screeningCenterView(centers, clusterCenters) {
  const coordinates=centers.map(item=>[item.pc1,item.pc2,...(clusterCenters[item.cluster]||[]).slice(2)]);
  const dimensions=coordinates.length?Math.min(...coordinates.map(row=>row.length)):2;
  const variance=Array.from({length:dimensions},(_,pc)=>{
    const values=coordinates.map(row=>row[pc]);
    if(!values.length||!values.every(Number.isFinite)) return 0;
    const mean=values.reduce((sum,value)=>sum+value,0)/values.length;
    return values.reduce((sum,value)=>sum+(value-mean)**2,0)/values.length;
  });
  const total=variance.reduce((sum,value)=>sum+value,0), columns=[0,1];
  const threshold=0.8-1e-12; // Absorb roundoff at the 80% boundary.
  let covered=variance[0]+variance[1];
  const ranked=variance.map((value,pc)=>pc).sort((a,b)=>variance[b]-variance[a]||a-b);
  for(const pc of ranked.filter(pc=>pc>=2&&pc<4)) {
    if(total<=0||covered/total>=threshold) break;
    if(variance[pc]>0) { columns.push(pc); covered+=variance[pc]; }
  }
  const primary=ranked[0];
  return {columns:columns.sort((a,b)=>a-b), coordinates, coverage:total>0?covered/total:null,
    orientation:total<=0?"无明显方向":variance[primary]/total>=threshold?`主要沿 PC${primary+1}`:dimensions>2?"多主元分布明显":"二维分布明显"};
}
function renderClusterQuality(container, quality, perspective, clusterCenters={}) {''',
        1,
    ).replace(
        '  if(!quality',
        '  if(perspective==="state_exploration") el("explorationQualityDetails").replaceChildren();\n  if(!quality',
        1,
    ).replace(
        '  const centers=',
        '  const centerView=perspective==="state_exploration"?screeningCenterView(quality.centers||[],clusterCenters):null;\n'
        '  let engineeringHint=quality.engineering_hint?.[perspective]||"请结合工艺状态人工确认。";\n'
        '  if(centerView&&centerView.orientation!=="主要沿 PC1") engineeringHint=engineeringHint.replace("当前状态划分主要沿 PC1 方向分离，可能反映连续运行变量变化，建议结合工艺变量确认。","");\n'
        '  const centerRows=centerView?(quality.centers||[]).map((item,index)=>`<tr><td>${escapeHtml(clusterUiLabel(item.cluster))}</td>${centerView.columns.map(pc=>`<td class="numeric">${number(centerView.coordinates[index][pc])}</td>`).join("")}</tr>`).join(""):"";\n'
        '  const centers=',
        1,
    ).replace(
        '  const centerCard=',
        '  const centerCard=centerView?`<div class="chart-card screening-evidence-compact"><div class="screening-center-heading"><h3>工况组中心与分离情况</h3><span class="help">${escapeHtml(centerView.orientation)}</span></div><div class="table-wrap screening-center-table" tabindex="0" role="region" aria-label="工况组中心坐标"><table><thead><tr><th>工况组</th>${centerView.columns.map(pc=>`<th>PC${pc+1}</th>`).join("")}</tr></thead><tbody>${centerRows}</tbody></table></div><p class="help">按工况中心方差选择主元，累计覆盖 ${centerView.coverage===null?"—":percent(centerView.coverage)}；单轴 ≥80% 为主要方向，最多 PC4。</p></div>`:',
        1,
    ).replace(
        'quality.engineering_hint?.[perspective]||"请结合工艺状态人工确认。"))}',
        'engineeringHint))}',
        1,
    ).replace(
        '<div class="chart-card"><h3>聚类质量摘要</h3><div class="metrics">',
        '<div class="chart-card screening-kpis">${perspective==="state_exploration"?"":"<h3>聚类质量摘要</h3>"}<div class="metrics">',
        1,
    ).replace(
        '<div class="chart-card"><h3>${title}</h3>',
        '<div class="chart-card screening-judgment"><h3>${title}</h3>'
        '${perspective==="state_exploration"?`<p>识别 ${quality.cluster_count} 个工况组 · 中心排列：${escapeHtml(centerView.orientation)}</p>`:""}',
        1,
    ).replace(
        '${details}`;',
        '${perspective==="state_exploration"?centerCard:details}`;\n'
        '  if(perspective==="state_exploration") el("explorationQualityDetails").replaceChildren();',
        1,
    )
    result = result.replace(renderer, screening_renderer, 1)
    # Keep the existing single-control assembly; move technical evidence below candidates.
    result = result.replace('<div id="explorationQualityDetails"></div>', '', 1)
    result = result.replace(
        '<details id="explorationClusterDetails">',
        '<details class="screening-technical-details"><summary>工况组技术详情</summary><div id="explorationQualityDetails"></div></details>'
        '<details id="explorationClusterDetails">', 1,
    )
    technical = _required_html_match(r'<details class="screening-technical-details">.*?</details>', result, "工况组技术详情").group()
    diagnostics = _required_html_match(r'<section class="chart-card variable-diagnostics".*?</section>', result, "变量诊断详情").group()
    result = result.replace(technical, '', 1).replace(diagnostics, '', 1)
    notice = '<div class="notice">选择候选后加入统一候选窗口列表；加入后仍需在候选窗口列表确认作为训练窗口。</div>'
    result = result.replace(notice, notice + technical + diagnostics, 1)
    overview = '<div id="explorationOverview" class="metrics"></div>'
    result = result.replace(overview, '', 1).replace('<div id="explorationClusterQuality"></div>', overview + '<div id="explorationClusterQuality"></div>', 1)
    result = result.replace('<details open><summary>工况组区分</summary>', '<details><summary>完整工况组变量对比</summary>', 1)
    result = result.replace(
        'renderClusterQuality(el("explorationClusterQuality"),data.cluster_quality,"state_exploration");',
        'renderClusterQuality(el("explorationClusterQuality"),data.cluster_quality,"state_exploration",data.cluster_centers);',
        1,
    )
    result = result.replace(
        "</head>",
        f"{_FORM_ALIGNMENT_STYLE}\n{_APPLE_DESIGN_STYLE}\n{_WORKBENCH_UI_STYLE}\n{_MODEL_RESULTS_STYLE}\n{_FORM_WIDTH_STYLE}\n{_FORM_LAYOUT_STYLE}\n{_SCREENING_LAYOUT_STYLE}\n</head>",
        1,
    )
    # Keep the base page owned by web.py; attach invalidation at its existing entries.
    for entry, hook in (
        ('function invalidateQuality(reason) {', 'globalThis.invalidateRedundancyAblation?.(reason, true);'),
        ('function renderValidationWindows() {', 'globalThis.refreshRedundancyValidation?.();'),
        ('function renderValidation(data) {', 'globalThis.invalidateRedundancyAblation?.("验证状态已更新");'),
        ('async function trainModel(modelPurpose) {', 'globalThis.invalidateRedundancyAblation?.("正在重新训练", true);'),
    ):
        if result.count(entry) != 1:
            raise ValueError(f"无法接入冗余消融失效入口：{entry}")
        result = result.replace(entry, entry + '\n  ' + hook, 1)
    result = result.replace('</head>', '<style>#trainingMainPanel:not([hidden]), #redundancyAblationPanel:not([hidden]) { display:grid; gap:var(--space-3); grid-template-columns:minmax(0,1fr); min-width:0; } #trainingMainPanel[hidden], #redundancyAblationPanel[hidden] { display:none; } #redundancyAblationPanel .table-wrap { overflow:auto; max-height:480px; } #redundancyAblationPanel td { overflow-wrap:anywhere; } #redundancyAblationPanel label { max-width:100%; } #redundancyAblationPanel select { max-width:100%; }</style></head>', 1)
    result = result.replace('</head>', '<style>#ablationResults td { white-space:normal; max-width:340px; vertical-align:top; } #ablationResults td:is(:nth-child(3),:nth-child(5),:nth-child(6),:nth-child(7)) { min-width:220px; } #ablationResults td:is(:first-child,:nth-child(8),:nth-child(9)) { white-space:nowrap; }</style></head>', 1)
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


def _is_reparse_point(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _path_safety_issue(path: Path) -> str | None:
    absolute = Path(os.path.abspath(path))
    for item in (absolute, *absolute.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            return f"无法检查路径 {item.name}：{error}"
        if _is_reparse_point(info):
            return f"路径包含符号链接或 Windows 重解析点：{item.name}"
    return None


def _measure_path(path: Path) -> tuple[int, list[str]]:
    total = 0
    issues: list[str] = []

    def visit(item: Path) -> None:
        nonlocal total
        try:
            info = item.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            issues.append(f"无法读取 {item.name}：{error}")
            return
        if _is_reparse_point(info):
            total += info.st_size
            issues.append(f"包含符号链接或 Windows 重解析点：{item.name}")
        elif stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            try:
                children = list(os.scandir(item))
            except OSError as error:
                issues.append(f"无法读取目录 {item.name}：{error}")
                return
            for child in children:
                visit(Path(child.path))
        else:
            issues.append(f"包含无法安全清理的特殊文件：{item.name}")

    visit(path)
    return total, issues


def _checked_run_for_deletion(
    run_id: str, *, allow_exploratory_draft: bool = False
) -> Path:
    runs_dir = Path(os.path.abspath(_BASE_WEB.RUNS_DIR))
    run_dir = Path(os.path.abspath(runs_dir / run_id))
    if (
        _path_safety_issue(runs_dir)
        or run_dir.parent != runs_dir
        or _path_safety_issue(run_dir)
    ):
        raise ValueError(f"候选模型运行记录路径无效：{run_id}")
    if not run_dir.is_dir():
        raise ValueError(f"候选模型运行记录不存在：{run_id}")
    model_path = run_dir / "model.pcamodel"
    if _path_safety_issue(model_path) or not model_path.is_file():
        raise ValueError(f"候选模型包路径无效：{run_id}")
    try:
        _, manifest = _BASE_WEB.load_model_package(model_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"候选模型包损坏：{run_id}") from error
    is_candidate = (
        manifest["model_purpose"] == "normal_state"
        and manifest["model_status"] == "candidate"
    )
    is_exploratory_draft = (
        allow_exploratory_draft
        and manifest["model_purpose"] == "exploratory"
        and manifest["model_status"] == "draft"
    )
    if not (is_candidate or is_exploratory_draft):
        raise ValueError(f"仅允许删除normal_state/candidate模型：{run_id}")
    protected_artifact = _candidate_deletion_block_reason(run_dir)
    if protected_artifact:
        raise ValueError(
            f"模型存在正式下游工件 {protected_artifact}，不能删除：{run_id}"
        )
    _, issues = _measure_path(run_dir)
    if issues:
        raise ValueError(f"运行目录无法安全清理：{run_id}（{issues[0]}）")
    return run_dir


@contextmanager
def _locked_runs_for_deletion(
    run_ids: Sequence[str], *, allow_exploratory_draft: bool = False
):
    with ExitStack() as locks:
        for run_id in sorted(run_ids):
            locks.enter_context(_BASE_WEB._lifecycle_lock(run_id))
        checked = [
            (run_id, _checked_run_for_deletion(
                run_id, allow_exploratory_draft=allow_exploratory_draft
            ))
            for run_id in run_ids
        ]
        yield checked


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
    with _locked_runs_for_deletion(validated_ids) as checked:
        for _, run_dir in checked:
            shutil.rmtree(run_dir)
    return {"deleted_run_ids": validated_ids}


def _history_inventory() -> dict[str, Any]:
    uploads_dir = Path(_BASE_WEB.UPLOADS_DIR)
    runs_dir = Path(_BASE_WEB.RUNS_DIR)
    upload_bytes, upload_measure_issues = _measure_path(uploads_dir)
    run_bytes, run_measure_issues = _measure_path(runs_dir)
    upload_scan_issues: list[str] = []
    association_issues: list[str] = []
    by_file: dict[str, list[Path]] = {}
    associations: dict[str, list[dict[str, Any]]] = {}

    uploads_issue = _path_safety_issue(uploads_dir)
    if uploads_issue:
        upload_scan_issues.append(uploads_issue)
    elif uploads_dir.is_dir():
        try:
            upload_paths = sorted(uploads_dir.iterdir())
        except OSError as error:
            upload_scan_issues.append(f"无法读取上传目录：{error}")
        else:
            for path in upload_paths:
                match = _HISTORY_UPLOAD.fullmatch(path.name)
                if not match:
                    continue
                file_id = match.group(1).lower()
                by_file.setdefault(file_id, []).append(path)

    run_count = 0
    runs_issue = _path_safety_issue(runs_dir)
    if runs_issue:
        association_issues.append(runs_issue)
    elif runs_dir.is_dir():
        try:
            run_paths = sorted(runs_dir.iterdir())
        except OSError as error:
            association_issues.append(f"无法读取运行目录：{error}")
        else:
            for run_dir in run_paths:
                try:
                    info = run_dir.lstat()
                except OSError as error:
                    association_issues.append(f"无法检查运行目录 {run_dir.name}：{error}")
                    continue
                if stat.S_ISDIR(info.st_mode):
                    run_count += 1
                if _is_reparse_point(info):
                    association_issues.append(
                        f"运行目录包含无法检查的重解析点：{run_dir.name}"
                    )
                    continue
                if not stat.S_ISDIR(info.st_mode):
                    continue
                run_id = run_dir.name
                if not _BASE_WEB._ID_PATTERN.fullmatch(run_id):
                    association_issues.append(f"运行目录标识无法识别：{run_id}")
                    continue
                snapshot_path = run_dir / "modeling_snapshot.json"
                if _path_safety_issue(snapshot_path) or not snapshot_path.is_file():
                    association_issues.append(f"运行 {run_id} 缺少可读取的建模快照")
                    continue
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    association_issues.append(f"运行 {run_id} 的建模快照无法读取")
                    continue
                if not isinstance(snapshot, dict):
                    association_issues.append(f"运行 {run_id} 的建模快照格式无效")
                    continue
                data_source = snapshot.get("data_source")
                file_id = data_source.get("file_id") if isinstance(data_source, dict) else None
                if (
                    snapshot.get("run_id") != run_id
                    or not isinstance(file_id, str)
                    or not _BASE_WEB._ID_PATTERN.fullmatch(file_id)
                ):
                    association_issues.append(f"运行 {run_id} 的源文件归属无法确认")
                    continue
                associations.setdefault(file_id, []).append(
                    {"run_id": run_id, "path": run_dir, "source": True}
                )
                report_path = run_dir / "validation_report.json"
                if _path_safety_issue(report_path) or not report_path.is_file():
                    continue
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    association_issues.append(
                        f"运行 {run_id} 的验证报告无法读取"
                    )
                    continue
                if not isinstance(report, dict):
                    association_issues.append(
                        f"运行 {run_id} 的验证报告格式无效"
                    )
                    continue
                validation_source = report.get("validation_source")
                if validation_source is None:
                    associations.setdefault(file_id, []).append(
                        {"run_id": run_id, "path": run_dir, "source": False}
                    )
                    continue
                if not isinstance(validation_source, dict):
                    association_issues.append(
                        f"运行 {run_id} 的验证来源无法确认"
                    )
                    continue
                validation_file_id = validation_source.get("file_id")
                if (
                    not isinstance(validation_file_id, str)
                    or not _BASE_WEB._ID_PATTERN.fullmatch(validation_file_id)
                ):
                    association_issues.append(
                        f"运行 {run_id} 的验证来源无法确认"
                    )
                    continue
                if validation_file_id == file_id:
                    continue
                associations.setdefault(validation_file_id, []).append(
                    {"run_id": run_id, "path": run_dir, "source": False}
                )

    files: dict[str, dict[str, Any]] = {}
    for file_id, paths in by_file.items():
        paths.sort(key=lambda item: item.name)
        reasons: list[str] = []
        sizes = 0
        uploaded_at = ""
        filename = file_id
        for path in paths:
            issue = _path_safety_issue(path)
            if issue:
                reasons.append(issue)
            try:
                info = path.lstat()
            except OSError as error:
                reasons.append(f"无法读取上传文件：{error}")
                continue
            if not stat.S_ISREG(info.st_mode) or _is_reparse_point(info):
                reasons.append("上传路径不是普通文件")
            else:
                sizes += info.st_size
                uploaded_at = datetime.fromtimestamp(
                    info.st_mtime, timezone.utc
                ).isoformat()
        if len(paths) != 1:
            reasons.append("同一 UUID 存在多个上传文件")
        metadata_path = uploads_dir / f"{file_id}.metadata.json"
        metadata_issue = _path_safety_issue(metadata_path)
        if metadata_issue:
            reasons.append(metadata_issue)
        else:
            try:
                metadata_info = metadata_path.lstat()
            except FileNotFoundError:
                metadata_info = None
            except OSError as error:
                reasons.append(f"无法读取上传元数据：{error}")
                metadata_info = None
            if metadata_info is not None:
                if _is_reparse_point(metadata_info) or not stat.S_ISREG(metadata_info.st_mode):
                    reasons.append("上传元数据不是普通文件")
                else:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if isinstance(metadata, dict):
                            raw_name = metadata.get("filename")
                            raw_time = metadata.get("uploaded_at")
                            if isinstance(raw_name, str) and Path(raw_name).name:
                                filename = Path(raw_name).name
                            if isinstance(raw_time, str) and raw_time.strip():
                                uploaded_at = raw_time
                    except (OSError, ValueError):
                        pass
        files[file_id] = {
            "file_id": file_id,
            "filename": filename,
            "uploaded_at": uploaded_at,
            "size_bytes": sizes,
            "paths": paths,
            "metadata_path": metadata_path,
            "issues": reasons,
            "runs": associations.get(file_id, []),
        }

    return {
        "files": files,
        "association_issues": association_issues,
        "upload_scan_issues": upload_scan_issues,
        "upload_count": sum(len(item["paths"]) for item in files.values()),
        "run_count": run_count,
        "upload_bytes": upload_bytes,
        "run_bytes": run_bytes,
        "storage_issues": upload_measure_issues + run_measure_issues,
    }


def _history_file_reasons(
    item: dict[str, Any], inventory: dict[str, Any]
) -> list[str]:
    reasons = list(item["issues"])
    if _upload_in_use(item["file_id"]):
        reasons.append("文件正在训练或验证操作中")
    if any(not run.get("source", True) for run in item["runs"]):
        run_ids = [
            run["run_id"] for run in item["runs"] if not run.get("source", True)
        ]
        reasons.append(
            "该文件被运行验证报告引用，受模型验证证据保护："
            + "、".join(run_ids)
        )
        for run in item["runs"]:
            if run.get("source", True):
                continue
            reasons.append(
                f"运行 {run['run_id']}：独立验证来源文件，保留该运行及其验证证据"
            )
    reasons.extend(
        f"无法确认运行关联：{issue}"
        for issue in inventory["association_issues"]
    )
    for run in item["runs"]:
        if not run.get("source", True):
            try:
                with _BASE_WEB._lifecycle_lock(run["run_id"]):
                    pass
            except ValueError as error:
                if "正在执行生命周期操作" in str(error):
                    reasons.append(f"运行 {run['run_id']}：{error}")
            continue
        try:
            with _BASE_WEB._lifecycle_lock(run["run_id"]):
                _checked_run_for_deletion(
                    run["run_id"], allow_exploratory_draft=True
                )
        except Exception as error:
            reasons.append(f"运行 {run['run_id']}：{error}")
    return list(dict.fromkeys(reasons))


def _history_public_file(
    item: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    reasons = _history_file_reasons(item, inventory)
    protected_by_validation = any(
        not run.get("source", True) for run in item["runs"]
    )
    runs = []
    for run in item["runs"]:
        size, _ = _measure_path(run["path"])
        run_reasons = [
            reason for reason in reasons if reason.startswith(f"运行 {run['run_id']}：")
        ]
        runs.append({
            "run_id": run["run_id"],
            "source": run.get("source", True),
            "size_bytes": size,
            "deletable": not protected_by_validation
            and run.get("source", True)
            and not run_reasons,
            "protection_reasons": run_reasons,
        })
    return {
        "file_id": item["file_id"],
        "filename": item["filename"],
        "uploaded_at": item["uploaded_at"],
        "size_bytes": item["size_bytes"],
        "run_count": len(runs),
        "runs": runs,
        "deletable": not reasons,
        "protection_reasons": reasons,
    }


def history_cleanup_payload() -> dict[str, Any]:
    inventory = _history_inventory()
    return {
        "storage": {
            "upload_count": inventory["upload_count"],
            "run_count": inventory["run_count"],
            "total_bytes": inventory["upload_bytes"] + inventory["run_bytes"],
            "complete": not inventory["storage_issues"],
            "issues": inventory["storage_issues"],
        },
        "files": [
            _history_public_file(item, inventory)
            for item in inventory["files"].values()
        ],
        "scan_issues": inventory["upload_scan_issues"],
    }


def _history_file_ids(payload: dict[str, Any]) -> list[str]:
    file_ids = payload.get("file_ids")
    if not isinstance(file_ids, list) or not file_ids:
        raise ValueError("file_ids必须包含至少一个历史上传文件")
    if any(not isinstance(file_id, str) for file_id in file_ids):
        raise ValueError("file_id无效")
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("file_ids不能重复")
    return [_BASE_WEB._validated_id(file_id, "file_id") for file_id in file_ids]


def history_cleanup_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inventory = _history_inventory()
    preview = []
    for file_id in _history_file_ids(payload):
        item = inventory["files"].get(file_id)
        if item is None:
            preview.append({
                "file_id": file_id,
                "filename": file_id,
                "deletable": False,
                "estimated_bytes": 0,
                "protection_reasons": ["上传文件不存在或无法识别"],
                "runs": [],
            })
            continue
        if any(not run.get("source", True) for run in item["runs"]):
            preview.append({
                **_history_public_file(item, inventory),
                "deletable": False,
                "estimated_bytes": 0,
            })
            continue
        public = _history_public_file(item, inventory)
        reasons = list(public["protection_reasons"])
        paths = [*item["paths"], item["metadata_path"]]
        paths.extend(run["path"] for run in item["runs"])
        sizes = []
        for path in paths:
            size, issues = _measure_path(path)
            sizes.append(size)
            reasons.extend(issues)
        reasons = list(dict.fromkeys(reasons))
        preview.append({
            **public,
            "deletable": not reasons,
            "estimated_bytes": sum(sizes) if not reasons else 0,
            "protection_reasons": reasons,
        })
    return {
        "files": preview,
        "estimated_bytes": sum(item["estimated_bytes"] for item in preview),
    }


def history_cleanup_execute_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("confirm") is not True:
        raise ValueError("必须在清理页面明确确认后才能执行")
    file_ids = _history_file_ids(payload)
    return _history_cleanup_execute_locked(file_ids)


def _history_busy_outcome(file_ids: Sequence[str], busy: str) -> dict[str, Any]:
    error = "文件正在训练或验证操作中，不能清理"
    return _history_cleanup_error_outcome(file_ids, error, only=busy)


def _history_cleanup_error_outcome(
    file_ids: Sequence[str], error: str, *, only: str | None = None
) -> dict[str, Any]:
    return {
        "files": [
            {
                "file_id": file_id,
                "filename": file_id,
                "deleted_file": False,
                "deleted_run_ids": [],
                "bytes_freed": 0,
                "errors": [error] if only in (None, file_id) else [],
            }
            for file_id in file_ids
        ],
        "deleted_file_ids": [],
        "deleted_run_ids": [],
        "bytes_freed": 0,
        "storage": history_cleanup_payload()["storage"],
    }


@contextmanager
def _history_file_locks(file_ids: Sequence[str]):
    """Take upload locks only; run locks are already held by the caller."""
    with ExitStack() as locks:
        for file_id in sorted(file_ids):
            locks.enter_context(
                _upload_lock(file_id, "历史清理文件", active=False)
            )
        yield


@contextmanager
def _history_run_lock(run_id: str):
    """Lifecycle lock for cleanup, with a readable conflict result.

    Validation's normal lock error is a 400 for a single run submit; cleanup is
    a batch preview flow, so it reports that run as skipped instead.
    """
    try:
        with _BASE_WEB._lifecycle_lock(run_id):
            yield
    except ValueError as error:
        if "正在执行生命周期操作" in str(error):
            raise ValueError(f"运行 {run_id} 正在执行生命周期操作，暂不能清理") from error
        raise


def _history_cleanup_execute_locked(file_ids: Sequence[str]) -> dict[str, Any]:
    try:
        with ExitStack() as locks:
            # Run locks always precede upload locks (validation uses the same
            # order). Scan under each run lock; once all are held, a fresh scan
            # decides which runs may be deleted.
            locked_runs: set[str] = set()
            for file_id in sorted(file_ids):
                for run in _history_inventory()["files"].get(file_id, {}).get("runs", []):
                    if run["run_id"] in locked_runs or not run.get("source", True):
                        continue
                    locks.enter_context(_history_run_lock(run["run_id"]))
                    locked_runs.add(run["run_id"])
            initial = _history_inventory()
            protected_runs = {
                run["run_id"]
                for file_id in file_ids
                for run in initial["files"].get(file_id, {}).get("runs", [])
                if not run.get("source", True)
            }
            current_run_ids = {
                run["run_id"]
                for file_id in file_ids
                for run in initial["files"].get(file_id, {}).get("runs", [])
                if run.get("source", True) and run["run_id"] not in protected_runs
            }
            if current_run_ids - locked_runs:
                raise ValueError("预览后运行关联已变化，请重新预览")
            busy = next(
                (file_id for file_id in file_ids if _upload_in_use(file_id)),
                None,
            )
            if busy is not None:
                return _history_busy_outcome(file_ids, busy)
            with _history_file_locks(file_ids):
                return _history_cleanup_storage_locked(file_ids, initial)
    except UploadBusyError:
        return _history_busy_outcome(file_ids, file_ids[0])
    except ValueError as error:
        return _history_cleanup_error_outcome(file_ids, str(error))


def _history_cleanup_storage_locked(
    file_ids: Sequence[str],
    initial: dict[str, Any],
) -> dict[str, Any]:
    outcomes = []
    deleted_file_ids: list[str] = []
    deleted_run_ids: list[str] = []

    for file_id in file_ids:
        item = initial["files"].get(file_id)
        if item is None:
            outcomes.append({
                "file_id": file_id,
                "filename": file_id,
                "deleted_file": False,
                "deleted_run_ids": [],
                "bytes_freed": 0,
                "errors": ["上传文件不存在或无法识别"],
            })
            continue
        run_ids = [
            run["run_id"] for run in item["runs"]
            if run.get("source", True)
        ]
        validation_run_ids = [
            run["run_id"] for run in item["runs"]
            if not run.get("source", True)
        ]
        removed_runs: list[str] = []
        errors: list[str] = []
        deleted_file = False
        bytes_freed = 0
        try:
            if validation_run_ids:
                raise ValueError(
                    "该文件被运行验证报告引用，受模型验证证据保护："
                    + "、".join(validation_run_ids)
                )
            current_inventory = _history_inventory()
            current = current_inventory["files"].get(file_id)
            if current is None:
                raise ValueError("上传文件已不存在或无法识别")
            if current_inventory["association_issues"]:
                raise ValueError(
                    f"无法确认运行关联：{current_inventory['association_issues'][0]}"
                )
            if current["issues"]:
                raise ValueError(current["issues"][0])
            if any(not run.get("source", True) for run in current["runs"]):
                current_validation_ids = [
                    run["run_id"] for run in current["runs"]
                    if not run.get("source", True)
                ]
                raise ValueError(
                    "该文件被运行验证报告引用，受模型验证证据保护："
                    + "、".join(current_validation_ids)
                )
            current_run_ids = sorted(
                run["run_id"] for run in current["runs"]
                if run.get("source", True)
            )
            if current_run_ids != sorted(run_ids):
                raise ValueError("预览后文件或运行关联已变化，请重新预览")
            checked = [
                (
                    run_id,
                    _checked_run_for_deletion(
                        run_id, allow_exploratory_draft=True
                    ),
                )
                for run_id in run_ids
            ]
            paths = [*current["paths"], current["metadata_path"]]
            paths.extend(run_dir for _, run_dir in checked)
            before_sizes = []
            for path in paths:
                if issue := _path_safety_issue(path):
                    raise ValueError(issue)
                size, issues = _measure_path(path)
                if issues:
                    raise ValueError(issues[0])
                before_sizes.append(size)
            before_bytes = sum(before_sizes)

            for run_id, run_dir in checked:
                try:
                    shutil.rmtree(run_dir)
                except OSError as error:
                    if run_dir.exists():
                        errors.append(f"运行 {run_id} 删除失败：{error}")
                    else:
                        removed_runs.append(run_id)
                else:
                    removed_runs.append(run_id)

            if not errors:
                source_path = current["paths"][0]
                try:
                    source_path.unlink(missing_ok=True)
                    deleted_file = not source_path.exists()
                    if deleted_file:
                        deleted_file_ids.append(file_id)
                        _BASE_WEB.remove_data_session(file_id)
                except OSError as error:
                    errors.append(f"上传文件删除失败：{error}")
                if deleted_file:
                    try:
                        current["metadata_path"].unlink(missing_ok=True)
                    except OSError as error:
                        errors.append(f"上传元数据删除失败：{error}")

            after_bytes = sum(_measure_path(path)[0] for path in paths)
            bytes_freed = max(0, before_bytes - after_bytes)
            deleted_run_ids.extend(removed_runs)
        except Exception as error:
            errors.append(str(error))

        outcomes.append({
            "file_id": file_id,
            "filename": item["filename"],
            "deleted_file": deleted_file,
            "deleted_run_ids": removed_runs,
            "bytes_freed": bytes_freed,
            "errors": errors,
        })

    refreshed = history_cleanup_payload()
    return {
        "files": outcomes,
        "deleted_file_ids": deleted_file_ids,
        "deleted_run_ids": deleted_run_ids,
        "bytes_freed": sum(item["bytes_freed"] for item in outcomes),
        "storage": refreshed["storage"],
    }


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


def _compute_web_build_id(html: str) -> str:
    digest = hashlib.sha256()
    for name, content in (
        ("index.html", html.encode("utf-8")),
        (_ASSET_PATH.name, _ASSET_PATH.read_bytes()),
        (
            _BASE_WEB.PLOTLY_JS_PATH.name,
            _BASE_WEB.PLOTLY_JS_PATH.read_bytes(),
        ),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:16]


WEB_BUILD_ID = _compute_web_build_id(INDEX_HTML)
for _asset_name in ("plotly.min.js", "model-results.js"):
    INDEX_HTML = INDEX_HTML.replace(
        f'src="/assets/{_asset_name}"',
        f'src="/assets/{_asset_name}?v={WEB_BUILD_ID}"',
        1,
    )
INDEX_HTML = INDEX_HTML.replace(
    "</head>",
    f'  <meta name="pca-model-builder-build" content="{WEB_BUILD_ID}">\n</head>',
    1,
)


class ModelResultsHandler(_BASE_WEB._Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/history-cleanup":
            try:
                self._send_json(history_cleanup_payload())
            except Exception as error:
                self._send_json(_BASE_WEB.error_payload(error), 400)
            return
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
            "/api/training-exceedance",
            "/api/trend",
            "/api/model-diagnostics",
            "/api/model-comparison",
            "/api/model-candidates/delete",
            "/api/history-cleanup/preview",
            "/api/history-cleanup/execute",
        }:
            super().do_POST()
            return
        try:
            payload = self._json_body()
            if path == "/api/history-cleanup/preview":
                result = history_cleanup_preview_payload(payload)
            elif path == "/api/history-cleanup/execute":
                result = history_cleanup_execute_payload(payload)
            else:
                result = (
                    training_exceedance.training_exceedance_payload(payload)
                    if path == "/api/training-exceedance"
                    else train_payload(payload)
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
    print(f"PCA Model Builder WebUI build: {WEB_BUILD_ID}")
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
