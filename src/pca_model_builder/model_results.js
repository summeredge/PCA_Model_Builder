(() => {
  "use strict";

  const modelContent = document.getElementById("modelContent");
  if (!modelContent || typeof window.renderTraining !== "function") return;

  const section = document.createElement("div");
  section.className = "chart-card";
  const title = document.createElement("h3");
  title.textContent = "PC1 / PC2 载荷图";
  const note = document.createElement("div");
  note.className = "help";
  note.textContent = "每个点代表一个原始Tag；全部Lag按带符号L2能量聚合。载荷不等同于异常贡献或工艺根因。";
  const chart = document.createElement("div");
  chart.id = "loadingChart";
  chart.className = "chart empty";
  chart.textContent = "完成DPCA训练后显示载荷图。";
  section.append(title, note, chart);
  document.getElementById("scoreChart")?.closest(".chart-card")?.insertAdjacentElement("afterend", section);

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
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 760 360");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "PC1与PC2载荷图");
    const maxAbs = Math.max(0.05, ...points.flatMap(point => [Math.abs(point.pc1), Math.abs(point.pc2)])) * 1.15;
    const x = value => 48 + (value + maxAbs) / (2 * maxAbs) * 664;
    const y = value => 312 - (value + maxAbs) / (2 * maxAbs) * 264;
    addLine(svg, x(0), 48, x(0), 312);
    addLine(svg, 48, y(0), 712, y(0));
    const labelled = new Set([...points].sort((a, b) => b.magnitude - a.magnitude).slice(0, 12).map(point => point.tag));
    points.forEach(point => {
      const circle = document.createElementNS(svg.namespaceURI, "circle");
      circle.setAttribute("cx", String(x(point.pc1)));
      circle.setAttribute("cy", String(y(point.pc2)));
      circle.setAttribute("r", "4");
      circle.setAttribute("fill", "#176b87");
      const tooltip = document.createElementNS(svg.namespaceURI, "title");
      tooltip.textContent = `${point.tag} · PC1 ${point.pc1.toFixed(4)} · PC2 ${point.pc2.toFixed(4)}`;
      circle.append(tooltip);
      svg.append(circle);
      if (labelled.has(point.tag)) {
        const label = document.createElementNS(svg.namespaceURI, "text");
        label.setAttribute("x", String(x(point.pc1) + 6));
        label.setAttribute("y", String(y(point.pc2) - 6));
        label.setAttribute("font-size", "10");
        label.textContent = point.tag;
        svg.append(label);
      }
    });
    chart.append(svg);
  }

  function addLine(svg, x1, y1, x2, y2) {
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", "#64748b");
    svg.append(line);
  }
})();
