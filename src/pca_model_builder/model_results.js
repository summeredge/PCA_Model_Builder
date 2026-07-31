(() => {
  "use strict";

  const modelContent = document.getElementById("modelContent");
  if (!modelContent || typeof window.renderTraining !== "function") return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const scoreCard = document.getElementById("scoreChart")?.closest(".chart-card");
  if (!scoreCard || !scoreCard.parentNode) return;

  const layoutStyle = document.createElement("style");
  layoutStyle.id = "modelProjectionGridStyle";
  layoutStyle.textContent = `
    .model-projection-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
      margin-bottom: 14px;
    }
    .model-projection-grid > .chart-card {
      min-width: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
    }
    .model-projection-grid > .chart-card > .chart {
      flex: 1 1 auto;
      min-height: 420px;
    }
    .model-projection-grid #scoreChart svg,
    .model-projection-grid #loadingChart svg {
      width: 100%;
      height: 420px;
      display: block;
    }
    @media (max-width: 1200px) {
      .model-projection-grid {
        grid-template-columns: 1fr;
      }
      .model-projection-grid > .chart-card > .chart {
        min-height: 360px;
      }
      .model-projection-grid #scoreChart svg,
      .model-projection-grid #loadingChart svg {
        height: 360px;
      }
    }
  `;
  document.head.append(layoutStyle);

  const section = document.createElement("div");
  section.className = "chart-card";
  const title = document.createElement("h3");
  title.textContent = "PC1 / PC2 原始Tag聚合载荷向量图";
  const note = document.createElement("div");
  note.className = "help";
  note.textContent = "每条向量从原点指向一个原始Tag的PC1/PC2聚合载荷；全部Lag按带符号L2能量聚合。向量方向和长度用于解释模型结构，不等同于异常贡献或工艺根因。";
  const chart = document.createElement("div");
  chart.id = "loadingChart";
  chart.className = "chart empty";
  chart.textContent = "完成DPCA训练后显示载荷向量图。";
  section.append(title, note, chart);

  const projectionGrid = document.createElement("div");
  projectionGrid.className = "model-projection-grid";
  scoreCard.parentNode.insertBefore(projectionGrid, scoreCard);
  projectionGrid.append(scoreCard, section);

  const originalRenderTraining = window.renderTraining;
  window.renderTraining = function renderTrainingWithLoadings(data) {
    originalRenderTraining(data);
    drawLoadingPlot(data.loading_plot);
  };

  function drawLoadingPlot(plot) {
    chart.replaceChildren();
    const points = Array.isArray(plot?.points) ? plot.points : [];
    if (!points.length) {
      chart.className = "chart empty";
      chart.textContent = "当前模型没有可展示的载荷。";
      return;
    }

    chart.className = "chart";
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 820 620");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "PC1与PC2原始Tag聚合载荷向量图");

    const plotLeft = 140;
    const plotTop = 40;
    const plotSize = 500;
    const plotRight = plotLeft + plotSize;
    const plotBottom = plotTop + plotSize;
    const maxAbs = Math.max(
      0.05,
      ...points.flatMap(point => [Math.abs(Number(point.pc1) || 0), Math.abs(Number(point.pc2) || 0)]),
    ) * 1.15;
    const x = value => plotLeft + (value + maxAbs) / (2 * maxAbs) * plotSize;
    const y = value => plotBottom - (value + maxAbs) / (2 * maxAbs) * plotSize;
    const originX = x(0);
    const originY = y(0);

    addArrowMarker(svg);
    addPlotFrame(svg, plotLeft, plotTop, plotSize);
    addGridAndTicks(svg, maxAbs, x, y, plotLeft, plotRight, plotTop, plotBottom);
    addLine(svg, originX, plotTop, originX, plotBottom, "#475569", 1.4);
    addLine(svg, plotLeft, originY, plotRight, originY, "#475569", 1.4);
    addAxisTitles(svg, plot, plotLeft, plotRight, plotTop, plotBottom);

    const labelled = new Set(
      [...points]
        .sort((left, right) => Number(right.magnitude || 0) - Number(left.magnitude || 0))
        .slice(0, 12)
        .map(point => point.tag),
    );

    points.forEach(point => {
      const endX = x(Number(point.pc1) || 0);
      const endY = y(Number(point.pc2) || 0);
      const vector = addLine(svg, originX, originY, endX, endY, "#9f3f3f", 1.6);
      vector.setAttribute("marker-end", "url(#loadingArrowHead)");
      vector.setAttribute("opacity", "0.82");

      const tooltip = document.createElementNS(SVG_NS, "title");
      tooltip.textContent = loadingTooltip(point);
      vector.append(tooltip);

      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", String(endX));
      circle.setAttribute("cy", String(endY));
      circle.setAttribute("r", "3.6");
      circle.setAttribute("fill", "#9f3f3f");
      const circleTooltip = document.createElementNS(SVG_NS, "title");
      circleTooltip.textContent = loadingTooltip(point);
      circle.append(circleTooltip);
      svg.append(circle);

      if (labelled.has(point.tag)) {
        const label = document.createElementNS(SVG_NS, "text");
        const rightSide = Number(point.pc1) >= 0;
        const above = Number(point.pc2) >= 0;
        label.setAttribute("x", String(endX + (rightSide ? 7 : -7)));
        label.setAttribute("y", String(endY + (above ? -7 : 14)));
        label.setAttribute("text-anchor", rightSide ? "start" : "end");
        label.setAttribute("font-size", "10");
        label.setAttribute("fill", "#1f2937");
        label.textContent = point.tag;
        svg.append(label);
      }
    });

    const origin = document.createElementNS(SVG_NS, "circle");
    origin.setAttribute("cx", String(originX));
    origin.setAttribute("cy", String(originY));
    origin.setAttribute("r", "3.2");
    origin.setAttribute("fill", "#111827");
    svg.append(origin);
    chart.append(svg);
  }

  function addArrowMarker(svg) {
    const defs = document.createElementNS(SVG_NS, "defs");
    const marker = document.createElementNS(SVG_NS, "marker");
    marker.setAttribute("id", "loadingArrowHead");
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "6");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("orient", "auto-start-reverse");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    path.setAttribute("fill", "#9f3f3f");
    marker.append(path);
    defs.append(marker);
    svg.append(defs);
  }

  function addPlotFrame(svg, left, top, size) {
    const frame = document.createElementNS(SVG_NS, "rect");
    frame.setAttribute("x", String(left));
    frame.setAttribute("y", String(top));
    frame.setAttribute("width", String(size));
    frame.setAttribute("height", String(size));
    frame.setAttribute("fill", "#ffffff");
    frame.setAttribute("stroke", "#94a3b8");
    frame.setAttribute("stroke-width", "1");
    svg.append(frame);
  }

  function addGridAndTicks(svg, maxAbs, x, y, left, right, top, bottom) {
    [-1, -0.5, 0, 0.5, 1].forEach(ratio => {
      const value = ratio * maxAbs;
      const px = x(value);
      const py = y(value);
      addLine(svg, px, top, px, bottom, ratio === 0 ? "#cbd5e1" : "#e5e7eb", 1);
      addLine(svg, left, py, right, py, ratio === 0 ? "#cbd5e1" : "#e5e7eb", 1);

      const xLabel = document.createElementNS(SVG_NS, "text");
      xLabel.setAttribute("x", String(px));
      xLabel.setAttribute("y", String(bottom + 20));
      xLabel.setAttribute("text-anchor", "middle");
      xLabel.setAttribute("font-size", "10");
      xLabel.setAttribute("fill", "#64748b");
      xLabel.textContent = formatLoading(value);
      svg.append(xLabel);

      const yLabel = document.createElementNS(SVG_NS, "text");
      yLabel.setAttribute("x", String(left - 10));
      yLabel.setAttribute("y", String(py + 3));
      yLabel.setAttribute("text-anchor", "end");
      yLabel.setAttribute("font-size", "10");
      yLabel.setAttribute("fill", "#64748b");
      yLabel.textContent = formatLoading(value);
      svg.append(yLabel);
    });
  }

  function addAxisTitles(svg, plot, left, right, top, bottom) {
    const xTitle = document.createElementNS(SVG_NS, "text");
    xTitle.setAttribute("x", String((left + right) / 2));
    xTitle.setAttribute("y", String(bottom + 58));
    xTitle.setAttribute("text-anchor", "middle");
    xTitle.setAttribute("font-size", "13");
    xTitle.setAttribute("fill", "#334155");
    xTitle.textContent = componentTitle("PC1载荷", plot?.x_explained_variance_ratio);
    svg.append(xTitle);

    const yTitle = document.createElementNS(SVG_NS, "text");
    yTitle.setAttribute("x", "52");
    yTitle.setAttribute("y", String((top + bottom) / 2));
    yTitle.setAttribute("text-anchor", "middle");
    yTitle.setAttribute("font-size", "13");
    yTitle.setAttribute("fill", "#334155");
    yTitle.setAttribute("transform", `rotate(-90 52 ${(top + bottom) / 2})`);
    yTitle.textContent = componentTitle("PC2载荷", plot?.y_explained_variance_ratio);
    svg.append(yTitle);
  }

  function componentTitle(name, ratio) {
    const value = Number(ratio);
    return Number.isFinite(value) ? `${name}（${(value * 100).toFixed(1)}%）` : name;
  }

  function loadingTooltip(point) {
    const prefix = [point.tag, point.description, point.unit].filter(Boolean).join(" · ");
    const pc1Lag = point.pc1_dominant_lag_minutes;
    const pc2Lag = point.pc2_dominant_lag_minutes;
    return `${prefix} · PC1 ${Number(point.pc1).toFixed(4)} · PC2 ${Number(point.pc2).toFixed(4)} · PC1主导Lag ${pc1Lag ?? "—"}min · PC2主导Lag ${pc2Lag ?? "—"}min`;
  }

  function formatLoading(value) {
    if (!Number.isFinite(value)) return "—";
    const absolute = Math.abs(value);
    if (absolute > 0 && absolute < 0.001) return value.toExponential(2);
    return value.toFixed(3);
  }

  function addLine(svg, x1, y1, x2, y2, stroke = "#64748b", width = 1) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", stroke);
    line.setAttribute("stroke-width", String(width));
    svg.append(line);
    return line;
  }
})();
