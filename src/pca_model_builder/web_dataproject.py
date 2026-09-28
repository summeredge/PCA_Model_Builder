from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import trend as trend_module
from . import web as base_web
from .tag_profile import profile_tag


_BASE_TREND_PAYLOAD = base_web.trend_payload


_DATAPROJECT_TREND_CSS = r"""
<style id="dataprojectTrendStyle">
  .dp-trend-controls { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)) 150px auto; gap:10px; align-items:end; }
  .dp-trend-options { display:grid; grid-template-columns:repeat(3,minmax(160px,1fr)); gap:10px; align-items:end; }
  .dp-chart { min-height:280px; border:1px solid var(--line); border-radius:6px; background:var(--panel); overflow:hidden; }
  .dp-chart .plotly, .dp-chart .svg-container, .dp-chart .gl-container { width:100%!important; height:320px; }
  .dp-chart .modebar { top:2px; right:2px; }
  .dp-chart .js-plotly-plot .plotly .cursor-crosshair { cursor:crosshair; }
  .dp-legend { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
  .dp-swatch { width:18px; height:3px; border-radius:2px; display:inline-block; vertical-align:middle; margin-right:6px; }
  .dp-trend-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; align-items:start; }
  .dp-trend-stat-card { min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:10px; }
  .dp-trend-stat-card h3 { margin:0 0 8px; font-size:12px; overflow-wrap:anywhere; }
  .dp-trend-stat-card dl { display:grid; gap:4px; margin:0; }
  .dp-trend-stat-card dl div { display:grid; grid-template-columns:82px 1fr; gap:8px; font-size:11px; }
  .dp-trend-stat-card dt { color:var(--muted); }
  .dp-trend-stat-card dd { margin:0; color:var(--text); text-align:right; font-variant-numeric:tabular-nums; }
  .dp-histogram { width:100%; min-width:0; margin-top:10px; }
  .dp-histogram-title { margin-bottom:4px; color:var(--muted); font-size:11px; }
  .dp-histogram-bars { position:relative; display:flex; align-items:flex-end; gap:2px; width:100%; height:72px; overflow:hidden; border-bottom:1px solid var(--line); }
  .dp-histogram-bar { position:relative; z-index:1; flex:1 1 0; min-width:0; opacity:.58; border-radius:2px 2px 0 0; }
  .dp-histogram-curve { position:absolute; inset:0; z-index:2; width:100%; height:100%; pointer-events:none; }
  .dp-histogram-curve polyline { fill:none; stroke-width:2; vector-effect:non-scaling-stroke; }
  .dp-histogram-labels { display:flex; justify-content:space-between; gap:8px; margin-top:3px; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
  .dp-scatter-section { display:grid; gap:12px; margin-top:10px; padding:0; border:0; }
  .dp-scatter-controls { display:grid; grid-template-columns:repeat(3,minmax(150px,1fr)); gap:10px; align-items:end; }
  .dp-scatter-chart { min-height:280px; border:1px solid var(--line); border-radius:6px; background:var(--panel); overflow:auto; }
  .dp-scatter-chart canvas { display:block; }
  .dp-inline-help { color:var(--muted); font-size:12px; line-height:1.45; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:#f8fafc; }
  @media (max-width:1050px) { .dp-trend-controls { grid-template-columns:repeat(3,minmax(120px,1fr)); } .dp-trend-stats { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  @media (max-width:760px) { .dp-trend-controls,.dp-trend-options,.dp-scatter-controls,.dp-trend-stats { grid-template-columns:1fr; } }
</style>
"""


_DATAPROJECT_TREND_SCRIPT = r"""
<script id="dataprojectTrendScript">
(() => {
  const panel = document.getElementById("trendPanel");
  if (!panel || document.getElementById("dpTrendVar1")) return;

  const legacy = document.createElement("div");
  legacy.id = "legacyTrendPanel";
  legacy.hidden = true;
  while (panel.firstChild) legacy.appendChild(panel.firstChild);
  panel.appendChild(legacy);

  const markup = `
    <h2>趋势图</h2>
    <div class="dp-inline-help">最多选择 4 个位号，在同一张图中浏览真实原始趋势。可在绘图区横向拖拽选择时间窗口。物理时间缺口不会连线，页面不会插值、补点或修改原始数据。</div>
    <div class="dp-trend-controls">
      <label>数据 1<select id="dpTrendVar1"></select></label>
      <label>数据 2<select id="dpTrendVar2"></select></label>
      <label>数据 3<select id="dpTrendVar3"></select></label>
      <label>数据 4<select id="dpTrendVar4"></select></label>
      <label>Y 轴<select id="dpTrendAxisMode"><option value="shared">同一 Y 轴</option><option value="independent">独立 Y 轴</option></select></label>
      <button id="dpDrawTrend" type="button" disabled>显示趋势</button>
    </div>
    <div class="dp-trend-options">
      <label>开始时间<input id="dpTrendStart" type="datetime-local"></label>
      <label>结束时间<input id="dpTrendEnd" type="datetime-local"></label>
      <label>最大绘图点数<input id="dpTrendMaxPoints" type="number" min="100" max="100000" value="100000"></label>
    </div>
    <div class="actions">
      <button id="dpTrendToAnalysis" type="button" class="secondary">将当前窗口设为分析期</button>
      <button id="dpTrendToReference" type="button" class="secondary">加入候选窗口</button>
      <button id="dpTrendToExclusion" type="button" class="secondary">加入排除窗口</button>
      <button id="dpTrendReset" type="button" class="secondary">趋势复位</button>
    </div>
    <div id="dpTrendChart" class="dp-chart empty">选择 1 到 4 个数据后点击“显示趋势”。</div>
    <div id="dpTrendLegend" class="dp-legend"></div>
    <div id="dpTrendStats" class="dp-trend-stats"><div class="empty">选择数据并点击“显示趋势”后显示统计摘要。</div></div>
    <section class="dp-scatter-section">
      <h2>XY 散点矩阵</h2>
      <div class="dp-inline-help">最多选择 3 个 X 轴变量和 3 个 Y 轴变量，按组合显示最多 9 个散点子图。散点关系用于人工观察分群、异常点和变量关系，不代表因果关系。</div>
      <div class="dp-scatter-controls">
        <label>X 变量 1<select id="dpScatterX1"></select></label>
        <label>X 变量 2<select id="dpScatterX2"></select></label>
        <label>X 变量 3<select id="dpScatterX3"></select></label>
        <label>Y 变量 1<select id="dpScatterY1"></select></label>
        <label>Y 变量 2<select id="dpScatterY2"></select></label>
        <label>Y 变量 3<select id="dpScatterY3"></select></label>
        <button id="dpDrawScatter" type="button" disabled>显示散点矩阵</button>
      </div>
      <div id="dpScatterMeta" class="dp-inline-help">选择 X 和 Y 变量后点击“显示散点矩阵”。</div>
      <div id="dpScatterChart" class="dp-scatter-chart empty">选择至少一个 X 变量和一个 Y 变量。</div>
    </section>`;
  panel.insertAdjacentHTML("beforeend", markup);

  const $ = (id) => document.getElementById(id);
  const trendIds = ["dpTrendVar1", "dpTrendVar2", "dpTrendVar3", "dpTrendVar4"];
  const scatterIds = ["dpScatterX1", "dpScatterX2", "dpScatterX3", "dpScatterY1", "dpScatterY2", "dpScatterY3"];
  const colors = ["#176b87", "#c2410c", "#6d28d9", "#15803d"];
  let lastTrend = null;
  let hasDraggedTrendSelection = false;
  let resizeTimer = null;

  function availableTags() {
    return Array.from($("trendTags")?.options || []).map((option) => option.value).filter(Boolean);
  }

  function fillSelect(node, values, allowBlank = true) {
    if (!node) return;
    const current = node.value;
    node.replaceChildren();
    if (allowBlank) {
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "不选择";
      node.append(blank);
    }
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      node.append(option);
    });
    if (values.includes(current)) node.value = current;
  }

  function populateSelectors() {
    const values = availableTags();
    [...trendIds, ...scatterIds].forEach((id) => fillSelect($(id), values, true));
    if (values.length && !trendIds.some((id) => $(id).value)) {
      $("dpTrendVar1").value = values[0] || "";
      $("dpTrendVar2").value = values[1] || "";
      $("dpTrendVar3").value = values[2] || "";
    }
    $("dpDrawTrend").disabled = !values.length;
    $("dpDrawScatter").disabled = !values.length;
  }

  function syncFromLegacy() {
    populateSelectors();
    const selected = Array.from($("trendTags")?.selectedOptions || []).map((option) => option.value);
    trendIds.forEach((id, index) => { $(id).value = selected[index] || ""; });
    if ($("trendStart")?.value) $("dpTrendStart").value = $("trendStart").value;
    if ($("trendEnd")?.value) $("dpTrendEnd").value = $("trendEnd").value;
  }

  function syncToLegacy(tags) {
    Array.from($("trendTags")?.options || []).forEach((option) => { option.selected = tags.includes(option.value); });
    if ($("trendStart")) $("trendStart").value = $("dpTrendStart").value;
    if ($("trendEnd")) $("trendEnd").value = $("dpTrendEnd").value;
  }

  function chosen(ids) {
    return ids.map((id) => $(id).value).filter(Boolean);
  }

  async function requestTrend(tags, purpose) {
    const unique = Array.from(new Set(tags));
    const payload = {
      ...commonPayload(),
      tags: unique,
      start: $("dpTrendStart").value,
      end: $("dpTrendEnd").value,
      display_mode: "raw",
      max_points: Number($("dpTrendMaxPoints").value || 100000),
      purpose,
    };
    return api("/api/trend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
  }

  $("dpDrawTrend").addEventListener("click", async () => {
    const tags = chosen(trendIds);
    if (!tags.length) return setStatus("请至少选择一个趋势变量。", "warning");
    if (new Set(tags).size !== tags.length) return setStatus("趋势变量不能重复选择。", "warning");
    syncToLegacy(tags);
    const button = $("dpDrawTrend");
    setBusy(button, true, "读取中…");
    try {
      const data = await requestTrend(tags, "trend");
      lastTrend = data;
      renderTrendPage(data);
      setStatus(`趋势图已生成，原始 ${data.raw_rows} 点，显示 ${data.rows_count} 点，最大点数 ${data.max_points}。数据未被修改或插值。`, "success");
    } catch (error) {
      $("dpTrendChart").className = "dp-chart empty";
      $("dpTrendChart").textContent = error.message || String(error);
      $("dpTrendLegend").replaceChildren();
      $("dpTrendStats").innerHTML = '<div class="empty">没有可展示的统计摘要。</div>';
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(button, false, "");
    }
  });

  $("dpTrendToAnalysis").addEventListener("click", () => {
    $("analysisStart").value = $("dpTrendStart").value;
    $("analysisEnd").value = $("dpTrendEnd").value;
    setStatus("当前趋势窗口已设置为分析期。", "success");
  });

  $("dpTrendToReference").addEventListener("click", () => {
    addCandidateWindow("trend", $("dpTrendStart").value, $("dpTrendEnd").value, "trend-current", "");
  });

  $("dpTrendToExclusion").addEventListener("click", () => {
    addExcludedWindow("trend", $("dpTrendStart").value, $("dpTrendEnd").value, "trend-current", "");
  });

  $("dpTrendReset").addEventListener("click", () => {
    const defaults = state.inspection;
    if (!defaults?.trend_default_start || !defaults?.trend_default_end) {
      return setStatus("请先完成当前文件的数据检查。", "warning");
    }
    $("dpTrendStart").value = localTime(defaults.trend_default_start);
    $("dpTrendEnd").value = localTime(defaults.trend_default_end);
    $("dpTrendMaxPoints").value = "100000";
    hasDraggedTrendSelection = false;
    if (lastTrend) renderTrendChart(lastTrend);
    syncToLegacy(chosen(trendIds));
    $("dpDrawTrend").click();
  });

  $("dpDrawScatter").addEventListener("click", async () => {
    const xTags = chosen(["dpScatterX1", "dpScatterX2", "dpScatterX3"]);
    const yTags = chosen(["dpScatterY1", "dpScatterY2", "dpScatterY3"]);
    if (!xTags.length) return setStatus("请选择至少一个 X 轴变量。", "warning");
    if (!yTags.length) return setStatus("请选择至少一个 Y 轴变量。", "warning");
    const button = $("dpDrawScatter");
    setBusy(button, true, "读取中…");
    try {
      const data = await requestTrend([...xTags, ...yTags], "scatter");
      renderScatterMatrix(data, xTags, yTags);
      $("dpScatterMeta").textContent = `实际绘图 ${data.rows_count} 行；筛选前窗口 ${data.raw_rows} 行；${xTags.length} 个 X × ${yTags.length} 个 Y。`;
      setStatus("XY 散点矩阵生成完成。", "success");
    } catch (error) {
      $("dpScatterChart").className = "dp-scatter-chart empty";
      $("dpScatterChart").textContent = error.message || String(error);
      setStatus(error.message || String(error), "error");
    } finally {
      setBusy(button, false, "");
    }
  });

  function renderTrendPage(data) {
    renderTrendChart(data);
    $("dpTrendLegend").innerHTML = data.series.map((item, index) => `<span><i class="dp-swatch" style="background:${colors[index % colors.length]}"></i>${escapeHtml(item.name)}</span>`).join("");
    $("dpTrendStats").innerHTML = data.series.map((item, index) => renderStatCard(item.name, data.statistics[item.name]?.current, data.histograms[item.name], colors[index % colors.length])).join("");
  }

  function timestampMilliseconds(value) {
    const milliseconds = Date.parse(value);
    return Number.isFinite(milliseconds) ? milliseconds : null;
  }

  function datetimeLocalValue(milliseconds) {
    const local = new Date(milliseconds - new Date(milliseconds).getTimezoneOffset() * 60000);
    return local.toISOString().slice(0, 16);
  }

  function setTrendWindowFromSelection(start, end) {
    const earlier = Math.min(start, end);
    const later = Math.max(start, end);
    $("dpTrendStart").value = datetimeLocalValue(earlier);
    $("dpTrendEnd").value = datetimeLocalValue(later);
    if ($("trendStart")) $("trendStart").value = $("dpTrendStart").value;
    if ($("trendEnd")) $("trendEnd").value = $("dpTrendEnd").value;
  }

  function renderTrendChart(data) {
    const container = $("dpTrendChart");
    const series = data.series || [];
    if (!series.length) {
      container.className = "dp-chart empty";
      container.textContent = "没有可绘制的趋势数据。";
      return;
    }
    const mode = $("dpTrendAxisMode").value;
    const timestamps = series.flatMap((item) => item.points.map((point) => timestampMilliseconds(point.x)).filter((value) => value !== null));
    if (!timestamps.length) {
      container.className = "dp-chart empty";
      container.textContent = "趋势数据缺少可解析的时间戳。";
      return;
    }
    const timeStart = Math.min(...timestamps);
    const timeEnd = Math.max(...timestamps);
    if (!globalThis.Plotly) {
      container.className = "dp-chart empty";
      container.textContent = "Plotly 未加载，无法绘制趋势图。";
      return;
    }
    const traces = series.map((item, seriesIndex) => {
      const x = [];
      const y = [];
      item.points.forEach((point) => {
        const value = finiteNumber(point.y);
        const pointTime = timestampMilliseconds(point.x);
        // Plotly has no notion of physical time gaps, so every physical segment
        // break and every missing value becomes an explicit null break point.
        if (point.physical_gap_start && x.length) { x.push(null); y.push(null); }
        if (value === null || pointTime === null) { x.push(pointTime); y.push(null); return; }
        x.push(pointTime);
        y.push(value);
      });
      return {
        type: "scattergl",
        mode: "lines",
        x,
        y,
        name: item.name,
        connectgaps: false,
        yaxis: axisName(seriesIndex, mode),
        line: {color: colors[seriesIndex % colors.length], width: 1.6},
        hoverinfo: "x+y+name",
      };
    });
    const note = mode === "shared" ? "同一 Y 轴：所有曲线使用同一数值范围" : "独立 Y 轴：各曲线按自身范围缩放，仅比较趋势形态";
    const layout = {
      margin: {left: 68, right: mode === "independent" && series.length > 1 ? 60 : 24, top: 30, bottom: 40},
      height: 320,
      showlegend: false,
      hovermode: "x unified",
      dragmode: "select",
      selectdirection: "h",
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
      font: {size: 11, color: "#5f6b7a"},
      annotations: [{text: note, x: 0, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {size: 12, color: "#5f6b7a"}}],
      xaxis: {type: "date", range: [timeStart, timeEnd], gridcolor: "#edf1f5", zeroline: false, showspikes: false},
      yaxis: {gridcolor: "#edf1f5", zeroline: false, autorange: true},
      shapes: trendExclusionShapes(),
    };
    const initialSelection = hasDraggedTrendSelection ? currentTrendSelectionShape() : null;
    if (initialSelection) layout.shapes = layout.shapes.concat([initialSelection]);
    series.forEach((item, seriesIndex) => {
      if (mode !== "independent" || seriesIndex < 1) return;
      // Only the first independent axis is labelled (right side); the rest overlay
      // it to keep 4 curves readable while hover still reports real values.
      const labelled = seriesIndex === 1;
      const axis = {
        overlaying: labelled ? undefined : "y",
        side: labelled ? "right" : "left",
        anchor: labelled ? "free" : undefined,
        position: labelled ? 1 : undefined,
        showticklabels: labelled,
        showgrid: false,
        zeroline: false,
        title: labelled ? item.name : undefined,
      };
      const limits = data.axis_limits?.[item.name];
      if (limits) { axis.range = [limits.minimum, limits.maximum]; }
      layout[`yaxis${seriesIndex + 1}`] = axis;
    });
    if (mode === "independent" && series.length === 1) { layout.yaxis.title = series[0].name; }
    const config = {displayModeBar: true, modeBarButtonsToRemove: ["toImage", "sendDataToCloud"], displaylogo: false, responsive: true, scrollZoom: false};
    container.className = "dp-chart";
    container.textContent = "";
    globalThis.Plotly.react(container, traces, layout, config).then((plot) => {
      plot.removeAllListeners("plotly_selected");
      plot.removeAllListeners("plotly_deselect");
      plot.on("plotly_selected", (event) => {
        const range = event?.range?.x;
        if (!range || range.length < 2) return;
        setTrendWindowFromSelection(range[0], range[1]);
        hasDraggedTrendSelection = true;
        applySelectionShape(range[0], range[1]);
        setStatus("已按选区设置当前趋势窗口；候选与训练窗口未改变。", "success");
      });
      plot.on("plotly_deselect", () => applySelectionShape());
      plot.on("plotly_relayout", (event) => {
        // Pure zoom/pan stays visual only; dpTrendStart/dpTrendEnd are untouched here.
        if (event?.["xaxis.autorange"]) applySelectionShape();
      });
    });
  }

  function axisName(seriesIndex, mode) {
    if (mode === "shared" || seriesIndex === 0) return "y";
    return `y${seriesIndex + 1}`;
  }

  function trendExclusionShapes() {
    return (state.excludedWindows || []).map((window) => {
      const start = timestampMilliseconds(window.start);
      const end = timestampMilliseconds(window.end);
      if (start === null || end === null) return null;
      return {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: start,
        x1: end,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(220,38,38,0.16)",
        line: {width: 0},
        layer: "below",
      };
    }).filter(Boolean);
  }

  function applySelectionShape(start, end) {
    const plot = $("dpTrendChart");
    if (!plot || !plot.layout) return;
    const shapes = trendExclusionShapes();
    if (Number.isFinite(start) && Number.isFinite(end)) {
      shapes.push(selectionShape(start, end));
    }
    globalThis.Plotly.relayout(plot, {shapes});
  }

  function selectionShape(start, end) {
    return {
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: start,
      x1: end,
      y0: 0,
      y1: 1,
      fillcolor: "rgba(23,107,135,0.18)",
      line: {color: "#176b87", width: 1.5},
      layer: "above",
    };
  }

  function currentTrendSelectionShape() {
    const start = timestampMilliseconds($("dpTrendStart").value);
    const end = timestampMilliseconds($("dpTrendEnd").value);
    if (start === null || end === null || start === end) return null;
    return selectionShape(Math.min(start, end), Math.max(start, end));
  }

  function renderStatCard(tag, stats, histogram, color) {
    if (!stats) return `<div class="dp-trend-stat-card"><h3>${escapeHtml(tag)}</h3><div class="empty">无统计数据</div></div>`;
    const count = Number(stats.valid_count || 0);
    const sample = Number(stats.sample_count || 0);
    const rows = [
      ["均值", stats.mean],
      ["标准差", stats.standard_deviation],
      ["最大值", stats.maximum],
      ["最小值", stats.minimum],
      ["极差", Number(stats.maximum) - Number(stats.minimum)],
      ["中位数", stats.median],
      ["有效点数/占比", `${count} / ${sample ? (count/sample*100).toFixed(1) : "0.0"}%`],
    ];
    return `<div class="dp-trend-stat-card"><h3>${escapeHtml(tag)}</h3><dl>${rows.map(([label,value]) => { const numeric = finiteNumber(value); return `<div><dt>${label}</dt><dd>${typeof value === "string" ? escapeHtml(value) : numeric === null ? "—" : formatAxis(numeric)}</dd></div>`; }).join("")}</dl>${renderHistogram(histogram, stats, color, tag)}</div>`;
  }

  function renderHistogram(histogram, stats, color, tag) {
    if (!histogram || !histogram.counts?.length) return '<div class="dp-histogram"><div class="dp-histogram-title">数值分布</div><div class="empty">无有效数据</div></div>';
    const maxCount = Math.max(...histogram.counts, 1);
    const bars = histogram.counts.map((count, index) => {
      const height = count ? Math.max(3, count / maxCount * 100) : 0;
      return `<span class="dp-histogram-bar" style="height:${height}%;background:${color}" title="${formatAxis(histogram.edges[index])} ～ ${formatAxis(histogram.edges[index+1])}: ${count}"></span>`;
    }).join("");
    const curve = normalCurve(stats, histogram);
    return `<div class="dp-histogram"><div class="dp-histogram-title">数值分布</div><div class="dp-histogram-bars" role="img" aria-label="${escapeHtml(tag)} 数值分布">${bars}${curve ? `<svg class="dp-histogram-curve" viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points="${curve}" stroke="${color}"/></svg>` : ""}</div><div class="dp-histogram-labels"><span>${formatAxis(histogram.edges[0])}</span><span>${formatAxis(histogram.edges.at(-1))}</span></div></div>`;
  }

  function normalCurve(stats, histogram) {
    const mean = Number(stats.mean);
    const std = Number(stats.standard_deviation);
    const min = Number(histogram.edges[0]);
    const max = Number(histogram.edges.at(-1));
    if (![mean,std,min,max].every(Number.isFinite) || std <= 0 || min === max) return "";
    return Array.from({length:41}, (_,index) => {
      const ratio = index / 40;
      const value = min + (max-min) * ratio;
      const densityRatio = Math.exp(-0.5 * ((value-mean)/std) ** 2);
      return `${(ratio*100).toFixed(2)},${(100-densityRatio*100).toFixed(2)}`;
    }).join(" ");
  }

  function renderScatterMatrix(data, xTags, yTags) {
    const container = $("dpScatterChart");
    const rows = data.rows || [];
    if (!rows.length) {
      container.className = "dp-scatter-chart empty";
      container.textContent = "没有可绘制的散点数据。";
      return;
    }
    container.className = "dp-scatter-chart";
    container.replaceChildren();
    const canvas = document.createElement("canvas");
    container.append(canvas);
    const cellWidth = 280;
    const cellHeight = 220;
    const labelWidth = 110;
    const topHeight = 38;
    const cssWidth = labelWidth + xTags.length * cellWidth + 16;
    const cssHeight = topHeight + yTags.length * cellHeight + 20;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.font = "11px sans-serif";
    context.fillStyle = "#17212b";
    xTags.forEach((tag,index) => context.fillText(tag, labelWidth + index*cellWidth + 50, 24));
    yTags.forEach((tag,index) => context.fillText(tag, 8, topHeight + index*cellHeight + 22));
    yTags.forEach((yTag,rowIndex) => xTags.forEach((xTag,columnIndex) => {
      const left = labelWidth + columnIndex*cellWidth + 42;
      const top = topHeight + rowIndex*cellHeight + 22;
      const width = cellWidth - 60;
      const height = cellHeight - 46;
      const pairs = rows.map((row) => [finiteNumber(row[`${xTag}__raw`]), finiteNumber(row[`${yTag}__raw`])]).filter(([x,y]) => x !== null && y !== null);
      context.strokeStyle = "#d8dee8";
      context.strokeRect(left, top, width, height);
      if (!pairs.length) { context.fillStyle = "#5f6b7a"; context.fillText("无有效配对数据", left+12, top+24); return; }
      let xMin = Math.min(...pairs.map((pair) => pair[0])); let xMax = Math.max(...pairs.map((pair) => pair[0]));
      let yMin = Math.min(...pairs.map((pair) => pair[1])); let yMax = Math.max(...pairs.map((pair) => pair[1]));
      if (xMin === xMax) { xMin -= .5; xMax += .5; }
      if (yMin === yMax) { yMin -= .5; yMax += .5; }
      const xPad = (xMax-xMin)*.05; const yPad = (yMax-yMin)*.05; xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
      context.save(); context.globalAlpha = .35; context.fillStyle = "#176b87";
      pairs.forEach(([xValue,yValue]) => { const x = left + (xValue-xMin)/(xMax-xMin)*width; const y = top + height - (yValue-yMin)/(yMax-yMin)*height; context.beginPath(); context.arc(x,y,1.7,0,Math.PI*2); context.fill(); });
      context.restore(); context.fillStyle = "#44546a"; context.fillText(`n=${pairs.length}`, left+6, top+14);
    }));
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatAxis(value) {
    if (!Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs > 0 && (abs < .001 || abs >= 1000000)) return value.toExponential(2);
    if (abs >= 10000) return value.toLocaleString("zh-CN",{maximumFractionDigits:0});
    if (abs >= 100) return value.toLocaleString("zh-CN",{maximumFractionDigits:1});
    if (abs >= 1) return value.toLocaleString("zh-CN",{maximumFractionDigits:2});
    return value.toLocaleString("zh-CN",{maximumFractionDigits:4});
  }

  const legacySelect = $("trendTags");
  if (legacySelect) new MutationObserver(syncFromLegacy).observe(legacySelect, {childList:true, subtree:true});
  const trendTab = document.querySelector('[data-panel="trendPanel"]');
  if (trendTab) trendTab.addEventListener("click", () => requestAnimationFrame(syncFromLegacy));
  $("dpTrendAxisMode").addEventListener("change", () => { if (lastTrend) renderTrendChart(lastTrend); });
  window.addEventListener("resize", () => { if (!lastTrend) return; clearTimeout(resizeTimer); resizeTimer = setTimeout(() => renderTrendChart(lastTrend), 120); });
  globalThis.refreshTrendExcludedWindows = () => {
    if (!lastTrend || !$("dpTrendChart")?.layout) return;
    // Only exclusion overlays change here; keep the current Plotly zoom/selection view.
    const selection = hasDraggedTrendSelection ? currentTrendSelectionShape() : null;
    globalThis.Plotly.relayout($("dpTrendChart"), {shapes: trendExclusionShapes().concat(selection ? [selection] : [])});
  };
  syncFromLegacy();
})();
</script>
"""


def apply_dataproject_trend_ui(html: str) -> str:
    if "dataprojectTrendScript" in html:
        return html
    if "</head>" not in html or "</body>" not in html:
        raise ValueError("无法定位Web页面注入位置")
    return html.replace("</head>", f"{_DATAPROJECT_TREND_CSS}\n</head>", 1).replace(
        "</body>", f"{_DATAPROJECT_TREND_SCRIPT}\n</body>", 1
    )


def _histogram(tag: str, values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {"tag": tag, "counts": [], "edges": []}
    counts, edges = np.histogram(
        finite,
        bins=min(20, max(1, int(np.sqrt(len(finite))))),
    )
    return {
        "tag": tag,
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
    }


def _trend_payload_data(
    indexed: pd.DataFrame,
    tags: Sequence[str],
    config: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tag_configs: Mapping[str, Mapping[str, Any]],
    max_points: int,
    reference_start: pd.Timestamp | None,
    reference_end: pd.Timestamp | None,
) -> dict[str, Any]:
    raw, _, _, segments, _ = trend_module.prepare_trend_frame(indexed, tags, config)
    mask = (raw.index >= start) & (raw.index <= end)
    if not mask.any():
        raise ValueError("趋势浏览窗口没有数据")
    current = raw.loc[mask]
    current_segments = segments.loc[mask]
    positions = trend_module.downsample_trend(
        current,
        current,
        current_segments,
        limit=max_points,
    )
    rows: list[dict[str, Any]] = []
    for position in positions:
        timestamp = current.index[position]
        full_position = int(raw.index.get_loc(timestamp))
        physical_gap_start = bool(
            full_position > 0
            and segments.iloc[full_position] != segments.iloc[full_position - 1]
        )
        record: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "physical_gap_start": physical_gap_start,
            "gap_start": physical_gap_start,
        }
        for tag in tags:
            value = current.iloc[position][tag]
            record[f"{tag}__raw"] = (
                float(value) if pd.notna(value) and np.isfinite(float(value)) else None
            )
        rows.append(record)

    reference_mask = None
    if reference_start is not None and reference_end is not None:
        reference_mask = (raw.index >= reference_start) & (raw.index <= reference_end)

    statistics: dict[str, Any] = {}
    histograms: dict[str, Any] = {}
    ranges: dict[str, dict[str, Any]] = {}
    axis_limits: dict[str, dict[str, float]] = {}
    for tag in tags:
        config_data = tag_configs.get(tag, {})
        statistics[tag] = {
            "full": profile_tag(raw[tag], config_data),
            "current": profile_tag(current[tag], config_data),
            "reference": (
                profile_tag(raw.loc[reference_mask, tag], config_data)
                if reference_mask is not None and reference_mask.any()
                else None
            ),
        }
        histograms[tag] = _histogram(tag, current[tag])
        ranges[tag] = {
            key: config_data.get(key)
            for key in (
                "engineering_min",
                "engineering_max",
                "normal_min",
                "normal_max",
                "alarm_min",
                "alarm_max",
            )
        }
        minimum, maximum = trend_module.trend_axis_limits(
            [*current[tag].tolist(), *ranges[tag].values()]
        )
        axis_limits[tag] = {"minimum": minimum, "maximum": maximum}

    series = [
        {
            "name": tag,
            "points": [
                {
                    "x": row["timestamp"],
                    "y": row[f"{tag}__raw"],
                    "physical_gap_start": row["physical_gap_start"],
                    "gap_start": row["gap_start"],
                }
                for row in rows
            ],
        }
        for tag in tags
    ]
    return {
        "tags": list(tags),
        "display_mode": "raw",
        "series": series,
        "rows": rows,
        "rows_count": len(rows),
        "raw_rows": len(current),
        "max_points": max_points,
        "statistics": statistics,
        "histograms": histograms,
        "ranges": ranges,
        "axis_limits": axis_limits,
    }


def trend_payload(payload: dict[str, Any]) -> dict[str, Any]:
    purpose = str(payload.get("purpose", "")).strip().lower()
    if purpose not in {"trend", "scatter"}:
        return _BASE_TREND_PAYLOAD(payload)

    timestamp_column = base_web._required_text(payload, "timestamp_column")
    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise ValueError("趋势Tag必须是非空列表")
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    if len(tags) != len(set(tags)):
        raise ValueError("趋势Tag不能重复")
    maximum_tags = 6 if purpose == "scatter" else 4
    if len(tags) > maximum_tags:
        raise ValueError(
            "散点矩阵最多使用6个不同Tag"
            if purpose == "scatter"
            else "趋势图一次最多选择4个Tag"
        )
    state_columns = base_web._state_filter_columns(payload)
    loaded = base_web._load_required_upload(
        payload, [*tags, *state_columns], "找不到趋势Tag："
    )
    parsed = loaded.frame
    all_tags = list(loaded.metadata.numeric_candidate_columns)
    registry = base_web.normalize_tag_registry(all_tags, payload.get("tag_configs"))
    missing = [tag for tag in tags if tag not in all_tags]
    if missing:
        raise ValueError(f"找不到趋势Tag：{', '.join(missing)}")
    try:
        max_points = int(payload.get("max_points", 100000))
    except (TypeError, ValueError) as error:
        raise ValueError("最大绘图点数必须是整数") from error
    max_points = min(100000, max(100, max_points))
    indexed = parsed.set_index(timestamp_column).sort_index()
    reference_start = base_web._optional_timestamp(payload.get("normal_start"))
    reference_end = base_web._optional_timestamp(payload.get("normal_end"))
    if "training_windows" in payload:
        training_window = base_web._single_enabled_training_window(
            base_web.training_windows_from_payload(payload)
        )
        reference_start = pd.Timestamp(training_window["start"])
        reference_end = pd.Timestamp(training_window["end"])
    with base_web._web_stage("preprocessing"):
        result = _trend_payload_data(
            indexed,
            tags,
            base_web._preprocessing_config(payload),
            pd.Timestamp(base_web._required_text(payload, "start")),
            pd.Timestamp(base_web._required_text(payload, "end")),
            registry,
            max_points,
            reference_start,
            reference_end,
        )
    return base_web._with_data_usage(
        result,
        loaded,
        int(result["raw_rows"]),
        int(result["rows_count"]),
    )


INDEX_HTML = apply_dataproject_trend_ui(base_web.INDEX_HTML)


def run_server(
    host: str = "127.0.0.1",
    port: int = base_web.DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    base_web.INDEX_HTML = INDEX_HTML
    base_web.trend_payload = trend_payload
    base_web.run_server(host, port, open_browser=open_browser)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run PCA Model Builder with the DataProject-style trend page."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=base_web.DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
