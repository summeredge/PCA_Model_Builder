from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
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
    #labelColumn, #preprocessingPreviewTagSelect, #preprocessingPreviewWindow,
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
  /* 数据源控件按内容角色自然换行：时间列用 Tag 宽度，编码选择保持短控件宽度。 */
  @media (min-width:761px) {
    main .group:has(#uploadButton) {
      display:flex;
      flex-wrap:wrap;
      align-items:end;
    }
    main .group:has(#uploadButton) > label { flex:1 1 100%; min-width:0; }
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
  #candidatePanel { gap:var(--space-3); }
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
  #modelingEligibility .condition-row > label { flex:1 1 120px; }
  #modelingEligibility .condition-row > label:first-child { flex-basis:200px; }
  #modelingEligibility .condition-row :is(input,select) { width:100%; }
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
  #stateExplorationPanel #explorationContent { gap:var(--space-2); }
  #stateExplorationPanel #explorationClusterQuality { display:grid; gap:var(--space-2); min-width:0; }
  #stateExplorationPanel .screening-kpis { gap:var(--space-1); padding-top:0; }
  #stateExplorationPanel .screening-kpis > .metrics { grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:var(--space-1); }
  #stateExplorationPanel .screening-kpis .metric { height:100%; }
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
  #stateExplorationPanel .screening-center-table :is(th,td) { padding:4px 10px; white-space:nowrap; }
  #stateExplorationPanel .screening-temporal-summary { display:flex; flex-wrap:wrap; gap:var(--space-1) var(--space-4); margin:0; }
  #stateExplorationPanel .screening-temporal-summary > div { display:flex; align-items:baseline; gap:var(--space-1); }
  #stateExplorationPanel .screening-temporal-summary dt { color:var(--muted); font-size:12px; }
  #stateExplorationPanel .screening-temporal-summary dd { margin:0; font-size:17px; font-variant-numeric:tabular-nums; }
  #stateExplorationPanel #explorationOverview .metric { border:0; padding:var(--space-1); min-height:0; }
  #stateExplorationPanel :is(th,td) { padding:6px 8px; }
  #stateExplorationPanel th { background:var(--line-soft); font-weight:600; }
  #stateExplorationPanel :is(th,td):first-child { text-align:left; }
  #stateExplorationPanel :is(th,td):not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
  @media (min-width:761px) {
    #stateExplorationPanel .performance-controls > label:has(#explorationPerformanceTag) { grid-column:span 2; }
  }
  @media (min-width:1200px) {
    #stateExplorationPanel .screening-kpis > .metrics { grid-template-columns:repeat(auto-fit,minmax(min(100%,128px),1fr)); }
  }
  @media (max-width:760px) {
    #stateExplorationPanel .screening-kpis > .metrics { grid-template-columns:repeat(auto-fit,minmax(min(100%,148px),1fr)); }
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
  /* 文件选择控件固定宽度；按钮按内容宽度靠左排布，不够时自然换行。 */
  #engineeringPanel .batch-config .actions > label.secondary {
    flex:0 0 200px;
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
    --panel-padding:24px;
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
  header { padding:16px clamp(20px,3vw,40px); background:#000000; border-bottom:0; color:#ffffff; }
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
  button.secondary {
    background:var(--bg);
    border-color:var(--line-control);
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
  .workflow-sidebar-title { margin:0; font-size:var(--type-section); font-weight:600; }
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
  .candidate-workflow-step { min-width:0; align-self:start; }
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
  .tag-workspace { display:grid; grid-template-columns:minmax(414px,.54fr) minmax(0,1.7fr); gap:var(--space-3); align-items:start; }
  @media (max-width:1480px) {
    .tag-workspace { grid-template-columns:minmax(0,1fr); }
  }
  .tag-workspace > *, .tag-detail { min-width:0; }
  #configPanel > .group { margin-bottom:var(--space-3); }
  #configPanel > #qualityPanel { display:grid; gap:var(--space-2); margin-top:var(--space-4); }
  .candidate-manager, .shared-preprocessing, .training-configuration { border-color:#bfd7ef; }
  .candidate-tool-tabs { display:flex; gap:var(--space-1); flex-wrap:wrap; align-items:center; border-bottom:1px solid var(--line); padding-bottom:var(--space-2); }
  .candidate-tool-tab { background:#f5f5f7; border-color:#f0f0f0; color:var(--accent); }
  .candidate-tool-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
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
  #explorationRegionSummary td:nth-child(2), #explorationRegionSummary td:nth-child(3) { text-align:center; }
  /* 标签换行时输入框仍需贴底对齐，否则同一行控件高低不齐。 */
  .exploration-controls > label { display:grid; align-content:start; }
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
    main { padding:var(--space-2); gap:var(--space-2); }
    .workflow-sidebar { position:static; padding:var(--space-3); }
    .candidate-analysis-range { grid-template-columns:minmax(0,1fr); }
    .candidate-tool-tabs { flex-wrap:nowrap; overflow-x:auto; }
    .candidate-tool-tabs > * { flex:0 0 auto; width:auto; }
    .actions { align-items:stretch; }
    /* 表单字段填满列宽；command button 保持 intrinsic 宽度并自然换行。 */
    .panel .actions > label, .inner-panel .actions > label { width:100%; }
    .tabs > *, .inner-tabs > * { flex:1 1 0; width:auto; min-width:0; }
    .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .metric strong { font-size:20px; }
    .table-wrap { max-width:100%; }
  }
  @media (max-width:520px) {
    .metrics { grid-template-columns:minmax(0,1fr); }
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
    const positions = sectionLinks.map(link => document.getElementById(link.hash.slice(1)).getBoundingClientRect().top);
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
  const toolPanels = ["trendPanel", "stateExplorationPanel", "clusterPanel", "performancePanel"].map(id => document.getElementById(id));
  const showCandidateTool = target => {
    toolPanels.forEach(panel => panel.classList.toggle("active", panel.id === target));
    document.querySelectorAll(".candidate-tool-tab").forEach(button => {
      const selected = button.dataset.panel === target;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
    sectionLinks[2].setAttribute("href", "#" + evidenceIds[target]);
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
    globalThis.refreshValidationInvestigationContext?.();
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
    rejected:"已拒绝", used:"已使用", dropped:"已丢弃",
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
    grid-template-columns:minmax(0,.9fr) minmax(0,1fr) minmax(0,1.3fr);
    gap:14px;
    align-items:start;
  }
  .model-overview-grid > .chart-card { min-width:0; margin:0; }
  #modelQualitySummary { grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr)); }
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
  #componentLoadings .component-loading-list {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr));
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
  @media (max-width:1599px) {
    .model-overview-grid { grid-template-columns:1fr; }
  }
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
        '            <a href="#modelingEligibility">筛选准备</a>',
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
    </section>"""


def _model_results_content_html() -> str:
    return """        <div id="modelContent" hidden>
          <h3>训练结果概览</h3>
          <div class="model-overview-grid">
            <section class="chart-card model-quality-conclusion" aria-labelledby="modelQualityTitle">
              <h3 id="modelQualityTitle">模型质量判断</h3>
              <div id="modelLifecycleNotice" class="notice model-quality-copy"></div>
              <div id="modelEngineeringJudgment" class="model-quality-copy" aria-live="polite"></div>
              <div id="modelQualityNotice" class="help model-quality-copy"></div>
            </section>
            <section class="chart-card model-structure-summary">
              <h3>模型结构</h3>
              <div id="modelMetrics" class="metrics"></div>
              <div id="modelProjectionSummary" class="metric"></div>
              <h4>主元解释率</h4><div id="varianceChart" class="variance"></div>
            </section>
            <section class="chart-card">
              <h3>统计质量</h3>
              <div id="modelQualitySummary" class="chart-grid" aria-live="polite"></div>
            </section>
          </div>
          <section id="componentLoadings" class="chart-card">
            <h3>主元贡献分析 / Loadings</h3>
            <div class="help">仅显示前 6 个主元。载荷来自实际训练模型的 PCA components。同一原始变量的全部 Lag 先聚合为一个 loading 强度 L_agg = √(Σ loading²)，聚合后不再有正负方向。按聚合强度从大到小显示 Top 10 原始变量，平方载荷占比以该主元全部变量的平方载荷总和为分母，不对 Top10 重新归一化，不等同于 T²/SPE 异常贡献。</div>
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
            """        <div id="modelingEligibility" class="group">
          <div class="group-title">筛选条件</div>
          <p class="help">只影响离线探索、候选与训练资格。上下限包含边界；保留条件全部满足（AND），排除组内全部满足（AND），任一排除组命中即剔除（OR），排除优先。条件列不会自动成为 PCA 输入。</p>
          <div class="screening-rules">
            <div><div class="actions"><h3>保留条件</h3><button id="addEligibilityKeep" type="button" class="secondary">添加保留条件</button></div><div id="eligibilityKeepConditions"></div></div>
            <div><div class="actions"><h3>排除条件</h3><button id="addEligibilityExcludeGroup" type="button" class="secondary">添加排除规则组</button></div><div id="eligibilityExcludeGroups"></div></div>
          </div>
          <div id="eligibilitySummary" class="screening-summary" aria-live="polite">检查数据后显示资格摘要；无规则时全部样本具备资格。</div>
          <div class="sub-title">分析时间范围</div>""",
            '          <div class="candidate-analysis-range">' + analysis_range + '</div>',
            '          <div class="help">状态探索、聚类辅助和条件筛选共用此范围；趋势选择可将浏览窗口设为这里的分析范围。修改后请重新运行分析。</div></div>',
            parameter_group,
            '        <div class="candidate-tool-tabs" id="candidateAnalysis" role="tablist">',
            '          <button type="button" class="candidate-tool-tab active" data-panel="trendPanel" role="tab" aria-selected="true">趋势选择</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="stateExplorationPanel" role="tab" aria-selected="false">状态探索</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="clusterPanel" role="tab" aria-selected="false">聚类辅助</button>',
            '          <button type="button" class="candidate-tool-tab" data-panel="performancePanel" role="tab" aria-selected="false">条件筛选</button>',
            '        </div>',
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
          <div class="notice">冻结与部署导出沿用现有流程；frozen 表示工程冻结，不表示已经部署。</div>
          <div class="actions">{validated_download}</div>
          <h3>② 工程冻结</h3>
{freeze_box}
          <h3>③ 冻结结果</h3>
          <div class="help">冻结后可下载 frozen 模型包与 deployment 模型包。</div>
          <div class="actions">{frozen_download}{deployment_download}</div>
          <section id="frozenReplay" class="chart-card" hidden>
            <h3>④ 冻结模型历史回放</h3>
            <div class="notice">历史回放仅检查冻结模型在历史数据上的表现，不属于独立验证，也不改变模型状态。</div>
            <div class="validation-box"><label>回放开始<input id="frozenReplayStart" type="datetime-local"></label><label>回放结束<input id="frozenReplayEnd" type="datetime-local"></label><button id="frozenReplayButton" type="button" disabled>执行冻结模型回放</button></div>
            <div id="frozenReplaySummary" class="help">请先完成工程冻结，再选择历史区间执行回放。</div>
            <div class="chart-grid"><div class="chart-card"><h3>T² / SPE 限值比趋势</h3><div id="frozenReplayTrend" class="chart empty">尚无回放结果。</div></div><div class="chart-card"><h3>状态统计</h3><div id="frozenReplayStatus" class="help">尚无回放结果。</div></div></div>
            <div class="actions"><a id="frozenReplayScoresDownload" class="download" href="#" hidden>下载完整评分 CSV</a><a id="frozenReplaySummaryDownload" class="download" href="#" hidden>下载回放摘要</a><a id="frozenReplayContributionsDownload" class="download" href="#" hidden>下载贡献记录</a></div>
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
    model_panel = f"{model_panel_content[0]}\n{model_configuration_group}\n{model_panel_content[1]}\n      </div>"
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
        '          <div class="help">性能 Tag 仅用于状态探索后的 post-hoc 评价，不参与状态探索 PCA。</div>\n'
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
    result = result.replace('<div id="dpTrendChart"', '<h3 id="trendEvidence">结果证据</h3>\n    <div id="dpTrendChart"', 1)
    # Only the final workbench presentation changes; payloads and calculations stay shared.
    result = result.replace(
        '  node.textContent=`${scope}：原始样本 ${summary.original_samples}；保留条件通过 ${summary.keep_pass_samples}；排除规则命中 ${summary.exclude_hit_samples}；最终合格 ${summary.eligible_samples}；合格占比 ${(summary.eligible_share*100).toFixed(1)}%；资格筛选后连续段 ${summary.segment_count}。${summary.eligible_samples?"":"没有合格样本，请调整资格规则。"}`;',
        '  node.classList.toggle("warning",!summary.eligible_samples);\n'
        '  node.textContent=`${scope}：筛选后 ${summary.eligible_samples.toLocaleString()} 点 · 保留条件通过 ${summary.keep_pass_samples.toLocaleString()} · 排除 ${summary.exclude_hit_samples.toLocaleString()} · 合格占比 ${(summary.eligible_share*100).toFixed(1)}% · 连续段 ${summary.segment_count}。${summary.eligible_samples?"":"没有合格样本，请调整资格规则。"}`;',
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
  if(node&&state.fileId&&state.inspection) { node.classList.toggle("warning",false); node.textContent="正在更新…"; }
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
    if(revision===eligibilitySummaryRevision&&rulesRevision===eligibilityRevision) { const node=el("eligibilitySummary"); node.classList.toggle("warning",true); node.textContent=error.message; }
  }
}
''', 1)
    result = result.replace('  clearTimeout(eligibilitySummaryTimer); eligibilitySummaryTimer=setTimeout(refreshEligibilitySummary,250);', '  scheduleEligibilitySummary();', 1)
    result = result.replace(
        '[select,minimum.querySelector("input"),maximum.querySelector("input")].forEach(input=>input.addEventListener("change",eligibilityChanged));',
        '[select,minimum.querySelector("input"),maximum.querySelector("input")].forEach(input=>input.addEventListener("change",eligibilityChanged));\n'
        '  [minimum,maximum].forEach(label=>label.querySelector("input").addEventListener("input",scheduleEligibilitySummary));',
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
        '  const centerRows=centerView?(quality.centers||[]).map((item,index)=>`<tr><td>${escapeHtml(clusterUiLabel(item.cluster))}</td>${centerView.columns.map(pc=>`<td>${number(centerView.coordinates[index][pc])}</td>`).join("")}</tr>`).join(""):"";\n'
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
    result = result.replace('</head>', '<style>.group-profiles { min-width:0; grid-template-columns:minmax(0,1fr); } #clusterContent, #assistanceClusterQuality { min-width:0; }.group-profiles .table-wrap { max-height:280px; overflow:auto; }.group-profiles td { white-space:normal; overflow-wrap:anywhere; }.group-profiles pre { white-space:pre-wrap; overflow-wrap:anywhere; }</style></head>', 1)
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
