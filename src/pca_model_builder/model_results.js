(() => {
  "use strict";

  const modelContent = document.getElementById("modelContent");
  if (!modelContent || typeof window.renderTraining !== "function") return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const chart = document.getElementById("loadingChart");
  if (!chart || !document.getElementById("componentLoadingsContent") || !document.getElementById("modelStructureComparison") || !document.getElementById("frozenReplay")) return;

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
    renderModelQuality(data.model_quality, data.training_window_totals);
    renderTrainingConditionDiagnostic(data.model_quality?.training_condition_diagnostic);
    drawLoadingPlot(data.loading_plot);
    renderComponentLoadings(data.loading_plot?.component_loadings);
    renderSingleModelDiagnostic(data.model_diagnostic);
    refreshCandidateOptions(data.run_id);
  };

  function renderModelQuality(quality, totals = {}) {
    const target = document.getElementById("modelQualitySummary");
    const projection = document.getElementById("modelProjectionSummary");
    const training = document.getElementById("modelTrainingDataQuality");
    const judgment = document.getElementById("modelEngineeringJudgment");
    const notice = document.getElementById("modelQualityNotice");
    if (!target || !projection || !training || !judgment || !notice) return;
    [target, projection, training, judgment, notice].forEach(item => item.replaceChildren());
    if (!quality) {
      target.textContent = "当前结果未提供统计质量数据，请重新训练后查看。";
      judgment.textContent = "当前结果未提供质量判断，请重新训练后查看。";
      return;
    }
    const paragraph = text => {
      const item = document.createElement("p");
      item.textContent = text;
      return item;
    };
    const metric = (value, label) => {
      const item = document.createElement("div");
      item.className = "metric";
      const strong = document.createElement("strong");
      strong.textContent = value;
      const caption = document.createElement("span");
      caption.textContent = label;
      item.append(strong, caption);
      return item;
    };
    const number = value => value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
    const percent = value => value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
    [["t2", "T²统计"], ["spe", "SPE/Q统计"]].forEach(([key, title]) => {
      const statistic = quality.statistics[key];
      const card = document.createElement("section");
      card.className = "chart-card";
      const heading = document.createElement("h4");
      heading.textContent = title;
      const values = document.createElement("div");
      values.className = "metrics";
      const trend = ({stable:"较稳定（经验检查）", changing:"存在变化", insufficient:"数据不足"})[statistic.trend] || "未知";
      values.append(
        metric(number(statistic.mean), "平均值"),
        metric(number(statistic.limits["95"]), "95% 控制限"),
        metric(percent(statistic.exceedance_rates["95"]), "95% 超限比例"),
        metric(trend, "趋势状态"),
      );
      card.append(heading, values, paragraph(`有效 / 无效评分：${statistic.valid_samples} / ${statistic.invalid_samples}；99% 控制限：${number(statistic.limits["99"])}；99% 超限比例：${percent(statistic.exceedance_rates["99"])}；已检查 ${statistic.trend_segments_checked ?? 0} 段。`));
      target.append(card);
    });
    notice.append(paragraph(quality.notice));
    const ratio = quality.pc1_pc2_explained_variance;
    const ratioValue = document.createElement("strong");
    ratioValue.textContent = percent(ratio);
    const ratioLabel = document.createElement("span");
    ratioLabel.textContent = "PC1+PC2 二维解释率";
    projection.append(ratioValue, ratioLabel, paragraph(ratio == null ? "当前模型不足两个主元。" : ratio >= 0.8 ? "二维图具有较好代表性。" : "二维图仅覆盖部分变化，建议结合更多主元和统计量判断。"));
    const data = quality.training_data;
    training.append(paragraph(`有效训练样本：${quality.training_samples}；有效样本覆盖时长：${number(data.effective_sample_hours)} h；覆盖日期数：${totals.covered_day_count ?? "—"}；已使用窗口：${totals.used_window_count ?? "—"}；已使用连续段：${totals.used_segment_count ?? "—"}；可回溯工况组来源数：${data.traceable_cluster_count ?? "未记录"}`),
      paragraph(`有效评分时间范围：${data.time_start ? displayTime(data.time_start) : "—"} ～ ${data.time_end ? displayTime(data.time_end) : "—"}。覆盖时长按有效样本数 × 采样周期计算，不包含窗口之间的空档。`));
    const sources = document.createElement("div");
    sources.className = "metrics";
    data.sources.forEach(row => {
      const item = document.createElement("div");
      item.className = "metric";
      const share = document.createElement("strong");
      share.textContent = percent(row.share);
      const label = document.createElement("span");
      label.textContent = ({manual:"手工", trend:"趋势", performance:"性能筛选", suggested:"建议窗口", cluster:"工况组"})[row.label] || clusterUiLabel(row.label);
      const reference = row.source_ref === "unavailable" ? "未记录" : row.source_ref;
      const details = document.createElement("span");
      details.className = "model-training-source-reference";
      details.textContent = `${row.samples} 有效样本`;
      details.title = reference;
      item.append(share, label, details);
      sources.append(item);
    });
    training.append(sources, paragraph(`工况组仅表示已确认训练窗口的来源，不能等同于实际状态数量；手工、趋势和性能窗口不能自动推断工况组。工况组来源缺失样本：${data.unattributed_samples}。`));
    quality.engineering_messages.forEach(message => judgment.append(paragraph(message)));
    const rules = document.createElement("details");
    const title = document.createElement("summary");
    title.textContent = "查看工程提示与趋势判断规则";
    rules.append(title, paragraph(quality.rules));
    judgment.append(rules);
  }

  function renderTrainingConditionDiagnostic(diagnostic) {
    const ids = ["modelTrainingConditionMessage", "modelTrainingConditionGroups", "modelTrainingConditionTrend", "modelTrainingConditionSwitch", "modelTrainingConditionHints"];
    const nodes = ids.map(id => document.getElementById(id));
    if (nodes.some(node => !node)) return;
    nodes.forEach(node => node.replaceChildren());
    const [message, groups, trend, switches, hints] = nodes;
    message.textContent = diagnostic?.message || "当前结果未提供训练工况诊断，请重新训练后查看。";
    const section = document.getElementById("modelTrainingConditionDiagnostic");
    if (section) section.classList.toggle("condition-unavailable", !diagnostic?.available);
    if (!diagnostic?.available) return;
    const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
    const number = value => Number.isFinite(value) ? value.toFixed(3) : "—";
    const table = (target, headers, rows) => {
      const element = document.createElement("table"), head = document.createElement("thead"), body = document.createElement("tbody");
      const appendRow = (parent, values, tag, reference) => {
        const row = document.createElement("tr");
        if (reference) row.title = reference;
        values.forEach(value => { const cell = document.createElement(tag); if (typeof value === "object") cell.append(value); else cell.textContent = value; row.append(cell); });
        parent.append(row);
      };
      appendRow(head, headers, "th");
      rows.forEach(row => appendRow(body, row.values, "td", row.reference));
      element.append(head, body); target.append(element);
    };
    const locate = (label, group, action="exploration") => { const button=document.createElement("button"); button.type="button"; button.className="secondary"; button.textContent=label; button.onclick=()=>globalThis.focusTrainingDiagnostic?.(diagnostic,group,action); return button; };
    const groupActions = group => {
      if(!globalThis.trainingDiagnosticRun?.(group)) return "来源不可追溯";
      const actions=document.createElement("div");
      actions.append(locate("查看该工况组",group),locate("继续筛选该工况组",group,"screen"),locate("查看关联训练窗口",group,"training_windows"));
      return actions;
    };
    table(groups, ["工况组", "评分样本数", "训练样本占比", "T² ≥95%", "T² ≥99%", "SPE ≥95%", "SPE ≥99%", "T²/95%限值比中位数", "SPE/95%限值比中位数", "Overall 95%最长连续时间", "定位"],
      diagnostic.groups.map(row => ({reference:row.source_ref, values:[clusterUiLabel(row.cluster_id), row.samples, percent(row.share), percent(row.t2_95_exceedance_rate), percent(row.t2_99_exceedance_rate), percent(row.spe_95_exceedance_rate), percent(row.spe_99_exceedance_rate), number(row.t2_95_ratio_median), number(row.spe_95_ratio_median), `${row.longest_overall_95_minutes} 分钟`, groupActions(row)]})));
    const comparison = diagnostic.switch_diagnostic;
    if (comparison?.available) {
      table(switches, ["区域", "评分样本数", "T² ≥95%", "SPE ≥95%", "Overall ≥95%"],
        [["稳定工况区", comparison.stable], ["工况切换附近", comparison.near_switch]].map(([label, row]) => ({values:[label, row.samples, percent(row.t2_95_exceedance_rate), percent(row.spe_95_exceedance_rate), percent(row.overall_95_exceedance_rate)]})));
      const rule = document.createElement("p");
      const context = comparison.context_policy === "full_segment_history" ? "一阶因果滤波依赖同段完整历史，含切换点的整个连续段计入切换附近，不假设固定历史长度。" : `附近范围为切换点前后各 ${comparison.context_minutes} 分钟（复用预处理上下文长度）。`;
      rule.textContent = `识别 ${comparison.switch_count} 个切换点；${context}${comparison.message}`;
      switches.append(rule);
      if ((diagnostic.timeline||[]).some(point=>point.near_switch===true&&Number.isFinite(Date.parse(point.timestamp)))) switches.append(locate("查看切换区",null));
    } else switches.textContent = comparison?.message || "当前训练样本缺少连续工况组标记，无法计算工况切换区诊断。";
    const coverage = document.createElement("p");
    coverage.textContent = `可追溯评分样本 ${diagnostic.traceable_samples}；未归属工况组 ${diagnostic.unattributed_samples}。分组统计按已确认窗口来源归属；状态带优先使用完整状态探索标记，缺少连续标记时使用窗口来源。Overall 95% 使用 T² ≥ T²95 或 SPE ≥ SPE95；连续时间按样本覆盖时长计算，遇窗口、物理段或时间缺口即中断。`;
    hints.append(coverage);
    (diagnostic.engineering_messages || []).forEach(text => { const paragraph = document.createElement("p"); paragraph.textContent = text; hints.append(paragraph); });
    drawTrainingConditionTrend(trend, diagnostic.timeline || []);
  }

  function drawTrainingConditionTrend(target, points) {
    const times = points.map(point => Date.parse(point.timestamp));
    const finiteTimes = times.filter(Number.isFinite);
    if (!finiteTimes.length) { target.textContent = "没有可绘制的训练评分时间。"; return; }
    const first = finiteTimes.reduce((value, time) => Math.min(value, time), Infinity), last = finiteTimes.reduce((value, time) => Math.max(value, time), -Infinity);
    const x = time => 125 + (time - first) / Math.max(last - first, 1) * 760;
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 920 400");
    svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "训练工况与统计量趋势，共享时间轴");
    const text = (label, px, py) => { const node = document.createElementNS(SVG_NS, "text"); node.setAttribute("x", px); node.setAttribute("y", py); node.setAttribute("font-size", "12"); node.setAttribute("fill", "#334155"); node.textContent = label; svg.append(node); };
    text("工况组", 8, 45); text("T² / T²95", 8, 140); text("SPE / Q95", 8, 275);
    [...new Set(points.map(point => point.cluster_id).filter(Boolean))].forEach((id, position) => text(clusterUiLabel(id), 125 + position * 72, 17));
    const tooltip = point => `${displayTime(point.timestamp,19)}\n${point.cluster_id ? clusterUiLabel(point.cluster_id) : "未归属工况组"}\nT²限值比：${point.t2_limit_ratio ?? "—"}\nSPE限值比：${point.spe_limit_ratio ?? "—"}`;
    const addTitle = (node, point) => { const title = document.createElementNS(SVG_NS, "title"); title.textContent = tooltip(point); node.append(title); };
    points.forEach((point, position) => {
      if (!Number.isFinite(times[position])) return;
      const next = points[position + 1];
      const continuous = next && !next.break_before && next.segment_id === point.segment_id && next.window_id === point.window_id;
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x(times[position])); rect.setAttribute("y", "28"); rect.setAttribute("height", "25");
      rect.setAttribute("width", continuous ? Math.max(1, x(times[position + 1]) - x(times[position])) : 2);
      rect.setAttribute("fill", point.cluster_id ? explorationClusterColor(point.cluster_id) : "#9aa7b4");
      addTitle(rect, point); svg.append(rect);
    });
    [["t2_limit_ratio", "#2563eb", 185], ["spe_limit_ratio", "#cf3f36", 325]].forEach(([field, color, bottom]) => {
      const maximum = points.reduce((value, point) => Number.isFinite(point[field]) ? Math.max(value, point[field]) : value, 1) * 1.1;
      const y = value => bottom - value / maximum * 105;
      addLine(svg, 125, y(1), 885, y(1), "#d19a20", 1);
      text("95%：ratio = 1", 125, bottom - 112);
      text("0", 108, bottom);
      text(maximum.toFixed(2), 92, bottom - 100);
      let segment = [];
      points.forEach((point, position) => {
        const previous = points[position - 1];
        if (point.break_before || (previous && (point.segment_id !== previous.segment_id || point.window_id !== previous.window_id)) || !Number.isFinite(point[field]) || !Number.isFinite(times[position])) {
          if (segment.length) replayLine(svg, segment, color);
          segment = [];
        }
        if (Number.isFinite(point[field]) && Number.isFinite(times[position])) {
          segment.push(`${x(times[position])},${y(point[field])}`);
          const dot = document.createElementNS(SVG_NS, "circle"); dot.setAttribute("cx", x(times[position])); dot.setAttribute("cy", y(point[field])); dot.setAttribute("r", "2"); dot.setAttribute("fill", color); addTitle(dot, point); svg.append(dot);
        }
      });
      if (segment.length) replayLine(svg, segment, color);
    });
    text(displayTime(new Date(first).toISOString(),19), 125, 375);
    const end = document.createElementNS(SVG_NS, "text"); end.setAttribute("x", "885"); end.setAttribute("y", "375"); end.setAttribute("text-anchor", "end"); end.setAttribute("font-size", "12"); end.textContent = displayTime(new Date(last).toISOString(),19); svg.append(end);
    text("时间", 480, 395);
    target.append(svg);
  }

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

  function renderComponentLoadings(components) {
    const target = document.getElementById("componentLoadingsContent");
    if (!target) return;
    target.replaceChildren();
    if (!Array.isArray(components) || !components.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "当前模型没有可展示的主元组成。";
      target.append(empty);
      return;
    }

    const list = document.createElement("div");
    list.className = "component-loading-list";
    components.slice(0, 6).forEach((component, index) => {
      const item = document.createElement("section");
      item.className = "component-loading-item";
      const title = document.createElement("h4");
      const name = component.component || `PC${index + 1}`;
      title.textContent = `${name} · explained variance ${formatExplainedVariance(component.explained_variance_ratio)}`;
      const topRows = sortedLoadings(component.top_loadings).slice(0, 10);
      const topTitle = document.createElement("div");
      topTitle.className = "help";
      topTitle.textContent = `Top ${topRows.length} 原始变量（同一Tag的全部Lag已聚合）`;
      item.append(title, topTitle, componentLoadingTable(topRows));
      list.append(item);
    });
    target.append(list);
  }

  function componentLoadingTable(rows) {
    const container = document.createElement("div");
    container.className = "table-wrap";
    const table = document.createElement("table");
    table.className = "component-loading-table";
    table.innerHTML = "<thead><tr><th>原始变量</th><th>聚合 loading 强度</th><th>平方载荷占比</th></tr></thead>";
    const body = document.createElement("tbody");
    (Array.isArray(rows) ? rows : []).forEach(row => {
      const tr = document.createElement("tr");
      tr.append(
        cell(row?.feature ?? "—"),
        cell(formatLoading(Number(row?.aggregated_loading))),
        cell(row?.loading_energy_share == null ? "—" : `${(row.loading_energy_share * 100).toFixed(2)}%`),
      );
      body.append(tr);
    });
    table.append(body);
    container.append(table);
    return container;
  }

  function sortedLoadings(rows) {
    return (Array.isArray(rows) ? [...rows] : []).sort(
      (left, right) => aggregatedLoading(right) - aggregatedLoading(left),
    );
  }

  function aggregatedLoading(row) {
    const value = Number(row?.aggregated_loading);
    return Number.isFinite(value) ? value : 0;
  }

  function formatExplainedVariance(value) {
    const ratio = Number(value);
    return Number.isFinite(ratio) ? `${(ratio * 100).toFixed(2)}%` : "—";
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
