(() => {
  "use strict";

  const modelContent = document.getElementById("modelContent");
  const releasePanel = document.getElementById("releasePanel");
  if (!modelContent || typeof window.renderTraining !== "function") return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const scoreCard = document.getElementById("scoreChart")?.closest(".chart-card");
  if (!scoreCard || !scoreCard.parentNode) return;

  const section = document.createElement("div");
  section.className = "chart-card";
  const title = document.createElement("h3");
  title.textContent = "PC1 / PC2 原始Tag聚合载荷图";
  const note = document.createElement("div");
  note.className = "help";
  note.textContent = "每条连线从原点连接到一个原始Tag的PC1/PC2聚合载荷；全部Lag按带符号L2能量聚合。连线方向和长度用于解释模型结构，不等同于异常贡献或工艺根因。";
  const chart = document.createElement("div");
  chart.id = "loadingChart";
  chart.className = "chart empty";
  chart.textContent = "完成DPCA训练后显示载荷图。";
  section.append(title, note, chart);

  const projectionGrid = document.createElement("div");
  projectionGrid.className = "model-projection-grid";
  scoreCard.parentNode.insertBefore(projectionGrid, scoreCard);
  projectionGrid.append(scoreCard, section);

  const diagnosticCard = document.createElement("section");
  diagnosticCard.className = "chart-card";
  diagnosticCard.id = "modelStructureComparison";
  diagnosticCard.innerHTML = `
    <h3>模型结构与参数比较</h3>
    <div class="help">诊断用于辅助工程师选择模型结构，不能替代独立验证；不会自动评分、推荐、验证或改变模型状态。</div>
    <div id="singleModelDiagnostic" class="help">完成正常状态候选模型训练后显示结构诊断。</div>
    <div id="modelCandidateStatus" class="help" role="status" aria-live="polite">正在加载候选模型…</div>
    <label class="secondary">选择已训练候选模型
      <select id="modelComparisonRuns" multiple size="5" aria-label="候选模型比较"></select>
    </label>
    <div class="actions"><button id="compareModelsButton" type="button">比较所选候选模型</button><button id="deleteModelsButton" class="danger" type="button">删除所选候选模型</button></div>
    <div id="modelComparisonResult" class="help">比较只读取已保存的正常状态候选模型包。</div>`;
  projectionGrid.insertAdjacentElement("afterend", diagnosticCard);

  const replayCard = document.createElement("section");
  replayCard.className = "chart-card";
  replayCard.id = "frozenReplay";
  replayCard.innerHTML = `
    <h3>冻结模型历史回放</h3>
    <div class="notice">历史回放用于检查冻结模型在历史数据上的表现，不属于独立验证，不改变模型状态。</div>
    <div class="validation-box"><label>回放开始<input id="frozenReplayStart" type="datetime-local"></label><label>回放结束<input id="frozenReplayEnd" type="datetime-local"></label><button id="frozenReplayButton" type="button">执行冻结模型回放</button></div>
    <div id="frozenReplaySummary" class="help">请先完成工程冻结，再选择历史区间执行回放。</div>
    <div class="chart-grid"><div class="chart-card"><h3>T² / SPE 限值比趋势</h3><div id="frozenReplayTrend" class="chart empty">尚无回放结果。</div></div><div class="chart-card"><h3>状态统计</h3><div id="frozenReplayStatus" class="help">尚无回放结果。</div></div></div>
    <div class="actions"><a id="frozenReplayScoresDownload" class="download" href="#" hidden>下载完整评分 CSV</a><a id="frozenReplaySummaryDownload" class="download" href="#" hidden>下载回放摘要</a><a id="frozenReplayContributionsDownload" class="download" href="#" hidden>下载贡献记录</a></div>`;
  releasePanel.append(replayCard);

  document.getElementById("frozenReplayButton").addEventListener("click", async () => {
    const button = document.getElementById("frozenReplayButton");
    const summary = document.getElementById("frozenReplaySummary");
    const start = document.getElementById("frozenReplayStart").value;
    const end = document.getElementById("frozenReplayEnd").value;
    if (!state.runId || !state.fileId || !start || !end) {
      setUiMessage(summary, "请先保留当前上传数据、完成工程冻结，并填写回放开始和结束时间。", "error");
      return;
    }
    setBusy(button, true, "回放中…");
    try {
      const response = await fetch("/api/frozen-replay", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({run_id: state.runId, file_id: state.fileId, timestamp_column: el("timestampColumn").value, encoding: el("encoding").value, replay_start: start, replay_end: end}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "冻结模型回放失败");
      renderFrozenReplay(data);
    } catch (error) {
      setUiMessage(summary, `冻结模型回放失败：${error.message}`, "error");
    } finally {
      setBusy(button, false);
    }
  });

  function renderFrozenReplay(data) {
    const summary = data.summary || {};
    const target = document.getElementById("frozenReplaySummary");
    setUiMessage(target, `${data.notice} 输出 ${summary.output_row_count ?? 0} 点；有效评分 ${summary.score_valid_count ?? 0} 点；状态过滤排除 ${summary.state_filter_excluded_rows ?? 0} 点；贡献记录 ${data.contribution_count ?? 0} 条。`, "success");
    const status = document.getElementById("frozenReplayStatus");
    status.textContent = Object.entries(summary.status_counts || {}).map(([key, value]) => `${displayValue(key)}：${value}`).join("；") || "无可展示评分点。";
    drawReplayTrend(data.scores || []);
    const downloads = data.downloads || {};
    [["scores", "frozenReplayScoresDownload"], ["summary", "frozenReplaySummaryDownload"], ["contributions", "frozenReplayContributionsDownload"]].forEach(([key, id]) => {
      const link = document.getElementById(id);
      link.href = downloads[key] || "#";
      link.hidden = !downloads[key];
    });
  }

  function drawReplayTrend(points) {
    const target = document.getElementById("frozenReplayTrend");
    target.replaceChildren();
    const finite = points.filter(point => Number.isFinite(point.t2_limit_ratio) || Number.isFinite(point.spe_limit_ratio));
    if (!finite.length) { target.className = "chart empty"; target.textContent = "所选区间没有可评分点。"; return; }
    target.className = "chart";
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 820 260");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "冻结模型回放 T2 和 SPE 状态趋势");
    const ratios = finite.flatMap(point => [point.t2_limit_ratio, point.spe_limit_ratio]).filter(Number.isFinite);
    const maximum = Math.max(1, ...ratios) * 1.1;
    const x = position => 45 + position / Math.max(points.length - 1, 1) * 750;
    const y = value => 225 - Math.min(value, maximum) / maximum * 190;
    addLine(svg, 45, y(1), 795, y(1), "#d19a20", 1);
    [["t2_limit_ratio", "#2563eb"], ["spe_limit_ratio", "#cf3f36"]].forEach(([field, color]) => {
      let segment = [];
      points.forEach((point, position) => {
        const value = Number(point[field]);
        if (Number.isFinite(value)) segment.push(`${x(position)},${y(value)}`);
        else if (segment.length) { replayLine(svg, segment, color); segment = []; }
      });
      if (segment.length) replayLine(svg, segment, color);
    });
    target.append(svg);
  }

  function replayLine(svg, points, color) {
    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("points", points.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2");
    svg.append(line);
  }

  const originalRenderTraining = window.renderTraining;
  window.renderTraining = function renderTrainingWithLoadings(data) {
    originalRenderTraining(data);
    drawLoadingPlot(data.loading_plot);
    renderSingleModelDiagnostic(data.model_diagnostic);
    refreshCandidateOptions(data.run_id);
  };

  document.getElementById("compareModelsButton").addEventListener("click", async () => {
    const select = document.getElementById("modelComparisonRuns");
    const runIds = [...select.selectedOptions].map(option => option.value);
    const target = document.getElementById("modelComparisonResult");
    const button = document.getElementById("compareModelsButton");
    if (runIds.length < 2 || runIds.length > 4) {
      setUiMessage(target, "模型比较需要选择 2—4 个候选模型。", "warning");
      return;
    }
    setBusy(button, true, "比较中…");
    setUiMessage(target, "正在比较所选候选模型…", "info");
    try {
      const response = await fetch("/api/model-comparison", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({run_ids: runIds}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "模型比较失败");
      renderComparison(data);
    } catch (error) {
      setUiMessage(target, `模型比较失败：${error.message}`, "error");
    } finally {
      setBusy(button, false);
    }
  });

  document.getElementById("deleteModelsButton").addEventListener("click", async () => {
    const select = document.getElementById("modelComparisonRuns");
    const selected = [...select.selectedOptions];
    const target = document.getElementById("modelCandidateStatus");
    const button = document.getElementById("deleteModelsButton");
    if (!selected.length) {
      setUiMessage(target, "请先选择要删除的候选模型。", "warning");
      return;
    }
    if (state.runId && selected.some(option => option.value === state.runId)) {
      setUiMessage(target, "当前正在使用的候选模型不能删除。", "error");
      return;
    }
    const protectedOption = selected.find(option => option.dataset.deletable !== "true");
    if (protectedOption) {
      setUiMessage(target, `所选候选模型不能删除：${protectedOption.dataset.blockReason || "已产生正式下游工件"}。`, "error");
      return;
    }
    const names = selected.map(option => `${option.dataset.modelName || "候选模型"}（${option.value.slice(0, 8)}）`);
    if (!window.confirm(`即将删除 ${selected.length} 个候选模型：${names.join("、")}。删除后不可恢复，是否继续？`)) return;
    setBusy(button, true, "删除中…");
    setUiMessage(target, `正在删除 ${selected.length} 个候选模型…`, "info");
    try {
      const response = await fetch("/api/model-candidates/delete", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({run_ids: selected.map(option => option.value)}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "候选模型删除失败");
      document.getElementById("modelComparisonResult").replaceChildren();
      setUiMessage(document.getElementById("modelComparisonResult"), "比较结果已清空。", "empty");
      await refreshCandidateOptions(state.runId);
      setUiMessage(target, `已删除 ${data.deleted_run_ids.length} 个候选模型。`, "success");
    } catch (error) {
      setUiMessage(target, `候选模型删除失败：${error.message}`, "error");
    } finally {
      setBusy(button, false);
    }
  });

  refreshCandidateOptions();

  async function refreshCandidateOptions(currentRunId) {
    const select = document.getElementById("modelComparisonRuns");
    const status = document.getElementById("modelCandidateStatus");
    setUiMessage(status, "正在加载候选模型…", "info");
    try {
      const response = await fetch("/api/model-candidates");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "无法读取候选模型");
      select.replaceChildren();
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      candidates.forEach(candidate => {
        const option = document.createElement("option");
        option.value = candidate.run_id;
        option.dataset.deletable = String(candidate.deletable !== false);
        option.dataset.blockReason = candidate.deletion_block_reason || "";
        option.dataset.modelName = candidate.model_name;
        const stateLabel = candidate.deletion_block_reason?.includes("validated") ? " · 已验证" : candidate.deletion_block_reason?.includes("frozen") ? " · 已冻结" : candidate.deletion_block_reason?.includes("deployment") ? " · 已部署" : "";
        option.textContent = `${candidate.model_name} · ${candidate.run_id.slice(0, 8)} · ${candidate.training_dynamic_samples} 样本${stateLabel}`;
        option.selected = candidate.run_id === currentRunId;
        select.append(option);
      });
      setUiMessage(status, candidates.length ? `已加载 ${candidates.length} 个候选模型。` : "暂无可比较候选模型。", candidates.length ? "success" : "empty");
    } catch (error) {
      setUiMessage(status, `候选模型加载失败：${error.message}`, "error");
    }
  }

  function renderSingleModelDiagnostic(diagnostic) {
    const target = document.getElementById("singleModelDiagnostic");
    if (!diagnostic) {
      target.className = "help";
      target.textContent = "当前为探索草稿模型；仅正常状态候选模型显示候选模型结构诊断。";
      return;
    }
    target.className = "";
    target.replaceChildren(
      diagnosticSummary(diagnostic),
      explainedVarianceChart([diagnostic]),
      energyTablePair(
        energyTable("原始Tag平方载荷能量（全部保留主元）", diagnostic.tag_loading_energy.retained_components, "tag"),
        lagEnergyTable(diagnostic),
      ),
    );
  }

  function renderComparison(data) {
    const target = document.getElementById("modelComparisonResult");
    target.className = "";
    const comparability = document.createElement("div");
    const reasons = data.comparability.reasons.length ? `：${data.comparability.reasons.join("；")}` : "";
    comparability.className = data.comparability.comparable ? "help" : "status error";
    if (!data.comparability.comparable) comparability.setAttribute("role", "alert");
    comparability.textContent = `可比性：${displayValue(data.comparability.status)}${reasons}`;
    target.replaceChildren(
      comparability,
      parameterTable(data.parameter_table),
      explainedVarianceChart(data.diagnostics),
      ...data.diagnostics.flatMap(diagnostic => [
        diagnosticSummary(diagnostic),
        energyTablePair(
          energyTable(`${diagnostic.model_name}：原始Tag平方载荷能量`, diagnostic.tag_loading_energy.retained_components, "tag"),
          lagEnergyTable(diagnostic),
        ),
      ]),
    );
  }

  function diagnosticSummary(diagnostic) {
    const block = document.createElement("div");
    block.className = "help";
    const limits = diagnostic.control_limits;
    block.textContent = `${diagnostic.model_name}（${diagnostic.run_id.slice(0, 8)}）：${diagnostic.training_dynamic_samples} 个训练动态样本，${diagnostic.raw_tag_count} 个原始Tag，${diagnostic.dynamic_feature_count} 个动态特征，保留 ${diagnostic.retained_component_count} 个主元；T² 95/99%=${limits.t2["95"].toFixed(3)}/${limits.t2["99"].toFixed(3)}，SPE 95/99%=${limits.spe["95"].toFixed(3)}/${limits.spe["99"].toFixed(3)}。`;
    return block;
  }

  function parameterTable(rows) {
    const table = document.createElement("table");
    table.className = "model-parameter-table";
    const runIds = Object.keys(rows[0]?.values || {});
    table.innerHTML = `<thead><tr><th>参数</th>${runIds.map(id => `<th>${id.slice(0, 8)}</th>`).join("")}</tr></thead>`;
    const body = document.createElement("tbody");
    rows.forEach(row => {
      const tr = document.createElement("tr");
      tr.append(cell(row.parameter));
      runIds.forEach(id => tr.append(cell(formatValue(row.values[id]))));
      body.append(tr);
    });
    table.append(body);
    return table;
  }

  function energyTable(title, rows, key) {
    const container = document.createElement("div");
    container.className = `model-energy-table ${key === "tag" ? "tag-energy-table" : "lag-energy-table"}`;
    const heading = document.createElement("h4");
    heading.textContent = title;
    const table = document.createElement("table");
    table.innerHTML = `<thead><tr><th>${key === "tag" ? "Tag" : "Lag（分钟）"}</th><th>能量占比</th></tr></thead>`;
    const body = document.createElement("tbody");
    rows.forEach(row => {
      const tr = document.createElement("tr");
      tr.append(cell(row[key]), cell(`${(row.energy * 100).toFixed(2)}%`));
      body.append(tr);
    });
    table.append(body);
    container.append(heading, table);
    return container;
  }

  function energyTablePair(tagTable, lagTable) {
    const container = document.createElement("div");
    container.className = "model-energy-grid";
    container.append(tagTable, lagTable);
    return container;
  }

  function lagEnergyTable(diagnostic) {
    const lag = diagnostic.lag_loading_energy;
    const container = energyTable(`${diagnostic.model_name}：Lag平方载荷能量（全部保留主元）`, lag.retained_components, "lag_minutes");
    const note = document.createElement("div");
    note.className = "help";
    note.textContent = `零Lag ${(lag.zero_lag_energy * 100).toFixed(2)}%，非零Lag ${(lag.nonzero_lag_energy * 100).toFixed(2)}%，主导Lag ${lag.dominant_lag_minutes} 分钟。`;
    container.append(note);
    return container;
  }

  function explainedVarianceChart(diagnostics) {
    const container = document.createElement("div");
    container.className = "model-variance-chart";
    const heading = document.createElement("h4");
    heading.textContent = "解释率累计曲线";
    const summary = document.createElement("div");
    summary.className = "model-variance-summary help";
    diagnostics.forEach(diagnostic => {
      const item = document.createElement("span");
      const retained = diagnostic.retained_component_count;
      const ratio = diagnostic.cumulative_explained_variance_ratio[retained - 1];
      const prefix = diagnostics.length > 1 ? `${diagnostic.model_name}：` : "";
      item.textContent = `${prefix}保留主元：${retained}；累计解释率：${(ratio * 100).toFixed(2)}%`;
      summary.append(item);
    });
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 640 245");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "主元累计解释率曲线，横轴为主元数量，纵轴为累计解释率百分比");
    const maxPoints = Math.max(...diagnostics.map(item => item.cumulative_explained_variance_ratio.length), 1);
    const plotLeft = 62;
    const plotRight = 610;
    const plotTop = 24;
    const plotBottom = 184;
    const x = component => plotLeft + (component - 1) / Math.max(maxPoints - 1, 1) * (plotRight - plotLeft);
    const y = ratio => plotBottom - ratio * (plotBottom - plotTop);
    addLine(svg, plotLeft, plotBottom, plotRight, plotBottom, "#64748b", 1);
    addLine(svg, plotLeft, plotTop, plotLeft, plotBottom, "#64748b", 1);
    [0, 0.25, 0.5, 0.75, 1].forEach(level => {
      const position = y(level);
      addLine(svg, plotLeft, position, plotRight, position, "#e2e8f0", 1);
      varianceLabel(svg, plotLeft - 8, position + 3, `${(level * 100).toFixed(0)}%`, "end");
    });
    [0.8, 0.9, 0.95].forEach(level => addLine(svg, plotLeft, y(level), plotRight, y(level), "#cbd5e1", 1));
    const componentStep = Math.max(1, Math.ceil(maxPoints / 12));
    for (let component = 1; component <= maxPoints; component += componentStep) {
      varianceLabel(svg, x(component), plotBottom + 17, String(component), "middle");
    }
    if ((maxPoints - 1) % componentStep !== 0) varianceLabel(svg, x(maxPoints), plotBottom + 17, String(maxPoints), "middle");
    varianceLabel(svg, (plotLeft + plotRight) / 2, 232, "主元数量", "middle", "13");
    const yTitle = varianceLabel(svg, 18, (plotTop + plotBottom) / 2, "累计解释率（%）", "middle", "13");
    yTitle.setAttribute("transform", `rotate(-90 18 ${(plotTop + plotBottom) / 2})`);
    diagnostics.forEach((diagnostic, index) => {
      const color = ["#9f3f3f", "#2563eb", "#059669", "#7c3aed"][index];
      const points = diagnostic.cumulative_explained_variance_ratio.map((value, point) => `${x(point + 1)},${y(value)}`).join(" ");
      const line = document.createElementNS(SVG_NS, "polyline");
      line.setAttribute("points", points);
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", "2");
      svg.append(line);
      const retained = diagnostic.retained_component_count;
      const ratio = diagnostic.cumulative_explained_variance_ratio[retained - 1];
      if (Number.isFinite(ratio)) {
        addLine(svg, x(retained), plotTop, x(retained), plotBottom, color, 1);
        const marker = document.createElementNS(SVG_NS, "circle");
        marker.setAttribute("cx", String(x(retained)));
        marker.setAttribute("cy", String(y(ratio)));
        marker.setAttribute("r", "4");
        marker.setAttribute("fill", color);
        svg.append(marker);
        varianceLabel(svg, x(retained), plotTop + 12 + index * 13, `保留 ${retained}`, "middle", "10", color);
      }
    });
    container.append(heading, summary, svg);
    return container;
  }

  function varianceLabel(svg, x, y, text, anchor = "start", size = "10", color = "#64748b") {
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(x));
    label.setAttribute("y", String(y));
    label.setAttribute("text-anchor", anchor);
    label.setAttribute("font-size", size);
    label.setAttribute("fill", color);
    label.textContent = text;
    svg.append(label);
    return label;
  }

  function cell(value) {
    const td = document.createElement("td");
    td.textContent = value;
    if (/^-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?:\s*\/\s*-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)?(?:\s*(?:%|分钟|点|样本|条))?$/i.test(String(value).trim())) {
      td.classList.add("numeric");
    }
    return td;
  }

  function displayValue(value) {
    return {continuous_input:"连续输入", state_filter:"状态过滤", label_only:"仅标签", exclude:"排除", higher_is_better:"越高越好", lower_is_better:"越低越好", target_range:"目标范围内", structural_comparison_only:"仅结构比较", not_comparable:"不可比较", pending:"待决策", accepted:"已接受", rejected:"已拒绝", normal:"正常", attention:"关注", abnormal:"异常", usable:"可用", review:"需确认", blocking:"阻止", used:"已使用", dropped:"已丢弃", trailing_mean:"尾随均值", trailing_median:"尾随中位数", mean:"均值", median:"中位数", last:"最后值", none:"不使用"}[value] || value;
  }

  function formatValue(value) {
    return typeof value === "object" ? JSON.stringify(value) : String(displayValue(value ?? "—"));
  }

  function setUiMessage(target, message, type = "info") {
    if (!target) return;
    target.className = type === "empty" ? "empty" : type === "error" ? "status error" : `status ${type}`;
    target.setAttribute("role", type === "error" ? "alert" : "status");
    target.textContent = message;
  }

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
    svg.setAttribute("aria-label", "PC1与PC2原始Tag聚合载荷图");

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
      vector.setAttribute("opacity", "0.72");

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
