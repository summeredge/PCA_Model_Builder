(() => {
  "use strict";

  const modelContent = document.getElementById("modelContent");
  if (!modelContent || typeof window.renderTraining !== "function") return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const chart = document.getElementById("loadingChart");
  if (!chart || !document.getElementById("componentLoadingsContent") || !document.getElementById("modelStructureComparison") || !document.getElementById("frozenReplay")) return;

  let ablationRevision = 0, ablationBusy = false, ablationInvalid = false;
  let ablationCandidates = null, ablationEvaluation = null, ablationIdentity = null;
  let ablationScheme = null;
  const ablationDrops = new Set(), ablationLocks = new Set();
  let ablationWindowsKey = JSON.stringify(state.validationWindows);
  const ablationConfigIds = ["components", "varianceThreshold", "sampleInterval", "resamplingMethod", "filterMethod", "firstOrderAlpha", "smoothingWindow", "gapThreshold", "maxLag", "lagStep", "timestampColumn", "encoding"];
  const ablationPercent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  function ablationKey() {
    return JSON.stringify([state.qualityRevision, [...state.selectedModelTags], state.registry, state.trainingWindows, state.validationWindows,
      ablationConfigIds.map(id => el(id)?.value), el("labelColumn").value, state.validation?.model_status]);
  }
  function ablationEligible() {
    return !ablationInvalid && state.runId && state.fileId && state.training?.run_id === state.runId &&
      state.training.model_purpose === "normal_state" && state.training.model_status === "candidate" &&
      (!state.validation?.model_status || state.validation.model_status === "candidate");
  }
  function ablationCurrent(identity) {
    return identity && identity.revision === ablationRevision && identity.runId === state.runId &&
      identity.fileId === state.fileId && identity.training === state.training && identity.key === ablationKey() && ablationEligible();
  }
  function ablationMessage(message, kind = "info") {
    setUiMessage(el("ablationStatus"), message, kind);
  }
  function updateAblationActions() {
    el("ablationCandidatesButton").disabled = ablationBusy || !ablationEligible();
    el("ablationGroup").disabled = ablationBusy || !ablationCandidates;
    el("ablationEvaluateButton").disabled = ablationBusy || !ablationCandidates || !ablationCurrent(ablationIdentity) || !el("ablationGroup").value || !ablationDrops.size;
    el("ablationApplyButton").disabled = ablationBusy || !ablationScheme || !ablationCurrent(ablationIdentity);
    el("ablationTags").querySelectorAll("input").forEach(input => { input.disabled = ablationBusy || !ablationCandidates || ablationLocks.has(input.dataset.ablationDrop); });
  }
  window.invalidateRedundancyAblation = function(reason, configurationChanged = false) {
    ablationRevision += 1;
    ablationBusy = false;
    if (configurationChanged) ablationInvalid = true;
    ablationCandidates = ablationEvaluation = ablationIdentity = ablationScheme = null;
    ablationDrops.clear(); ablationLocks.clear();
    ["ablationTags", "ablationEvidence", "ablationDiagnostics", "ablationResults", "ablationWarnings", "ablationPolicy"].forEach(id => el(id).replaceChildren());
    el("ablationGroup").replaceChildren(new Option("尚未加载", ""));
    ablationMessage(reason ? `当前评估结果已失效：${reason}。${configurationChanged ? "请重新检查质量并正式训练。" : "请重新加载候选组。"}` : "请先完成正常状态候选模型训练，再加载冗余候选组。");
    updateAblationActions();
  };
  window.refreshRedundancyValidation = function() {
    const key = JSON.stringify(state.validationWindows);
    if (key !== ablationWindowsKey) window.invalidateRedundancyAblation("验证窗口已修改");
    ablationWindowsKey = key;
  };
  ablationConfigIds.forEach(id => el(id)?.addEventListener("change", () => window.invalidateRedundancyAblation("训练配置已修改", true)));
  el("labelColumn").addEventListener("change", () => window.invalidateRedundancyAblation("验证工程标签已修改"));
  function showTrainingTool(panelId) {
    [["trainingMainTab", "trainingMainPanel"], ["redundancyAblationTab", "redundancyAblationPanel"]].forEach(([tabId, id]) => {
      const active = id === panelId;
      el(id).hidden = !active; el(tabId).classList.toggle("active", active); el(tabId).setAttribute("aria-selected", String(active));
    });
    updateAblationActions();
  }
  [["trainingMainTab", "trainingMainPanel"], ["redundancyAblationTab", "redundancyAblationPanel"]].forEach(([tabId, panelId]) => {
    el(tabId).addEventListener("click", () => showTrainingTool(panelId));
    el(tabId).addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? "trainingMainTab" : event.key === "End" ? "redundancyAblationTab" : tabId === "trainingMainTab" ? "redundancyAblationTab" : "trainingMainTab";
      el(next).click(); el(next).focus();
    });
  });
  function ablationTable(target, headers, rows) {
    target.replaceChildren();
    const table = document.createElement("table"), head = document.createElement("thead"), body = document.createElement("tbody"), titles = document.createElement("tr");
    headers.forEach(value => { const cell = document.createElement("th"); cell.textContent = value; titles.append(cell); }); head.append(titles);
    rows.forEach(values => {
      const row = document.createElement("tr");
      values.forEach(value => { const cell = document.createElement("td"); if (value instanceof Node) cell.append(value); else cell.textContent = value ?? "—"; row.append(cell); }); body.append(row);
    });
    table.append(head, body); target.append(table); target.tabIndex = 0;
  }
  function clearAblationSchemes() {
    ablationRevision += 1; ablationEvaluation = ablationScheme = null;
    el("ablationResults").replaceChildren(); el("ablationWarnings").replaceChildren();
    // Candidate evidence survives manual choices, but requests/results belong to that choice revision.
    if (ablationIdentity) ablationIdentity.revision = ablationRevision;
    updateAblationActions();
  }
  el("ablationGroup").addEventListener("change", () => {
    clearAblationSchemes(); ablationDrops.clear(); ablationLocks.clear();
    const group = ablationCandidates?.groups.find(item => String(item.id) === el("ablationGroup").value);
    if (!group) { el("ablationTags").replaceChildren(); el("ablationEvidence").replaceChildren(); updateAblationActions(); return; }
    ablationTable(el("ablationTags"), ["Tag", "待逐个剔除", "关键锁定"], group.tags.map(tag => {
      const drop = document.createElement("input"), lock = document.createElement("input"); drop.type = lock.type = "checkbox";
      drop.dataset.ablationDrop = tag;
      drop.setAttribute("aria-label", `待剔除 ${tag}`); lock.setAttribute("aria-label", `锁定 ${tag}`);
      drop.addEventListener("change", () => {
        if (drop.checked && ablationDrops.size >= ablationCandidates.max_drop_tags) { drop.checked = false; ablationMessage(`单次最多选择 ${ablationCandidates.max_drop_tags} 个待剔除 Tag。`, "warning"); return; }
        if (drop.checked) ablationDrops.add(tag); else ablationDrops.delete(tag);
        clearAblationSchemes(); ablationMessage(ablationDrops.size ? `已选择 ${ablationDrops.size} 个单 Tag 剔除方案。` : "未选择待评估变量。");
      });
      lock.addEventListener("change", () => { if (lock.checked) { ablationLocks.add(tag); ablationDrops.delete(tag); drop.checked = false; } else ablationLocks.delete(tag); clearAblationSchemes(); ablationMessage("关键锁定已更新；锁定 Tag 不参与剔除。"); });
      return [tag, drop, lock];
    }));
    ablationTable(el("ablationEvidence"), ["Tag A", "Tag B", "Pearson r", "有效配对样本数", "相关证据"], group.pairs.map(pair => [pair.tag_a, pair.tag_b, Number.isFinite(pair.pearson_r) ? pair.pearson_r.toFixed(4) : "不可计算", pair.valid_pair_count, pair.unavailable_reason || (pair.high_correlation ? "高相关" : "低于阈值")]));
    ablationMessage("未选择待评估变量。请勾选组内待剔除 Tag。"); updateAblationActions();
  });
  async function requestAblation(action) {
    if (!ablationEligible()) { ablationMessage("仅支持当前正常状态候选模型；配置变更后请重新训练。", "warning"); return; }
    if (action === "evaluate" && (!ablationCurrent(ablationIdentity) || !ablationDrops.size)) { window.invalidateRedundancyAblation("模型、配置或选择已变化"); return; }
    const identity = {revision: ++ablationRevision, runId: state.runId, fileId: state.fileId, training: state.training, key: ablationKey()};
    const sha = action === "evaluate" ? ablationCandidates.model_sha256 : null;
    ablationBusy = true; ablationScheme = null; el("ablationResults").replaceChildren(); el("ablationWarnings").replaceChildren(); updateAblationActions();
    ablationMessage(action === "evaluate" ? "正在执行逐 Tag 消融…" : "正在加载冗余候选组…");
    try {
      const data = await api("/api/redundancy-ablation", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({run_id: identity.runId, file_id: identity.fileId, action, ...(sha ? {model_sha256: sha} : {}), encoding: commonPayload().encoding, label_column: el("labelColumn").value, group_id: el("ablationGroup").value || null, drop_tags: [...ablationDrops], locked_tags: [...ablationLocks], validation_windows: state.validationWindows})});
      if (!ablationCurrent(identity)) return;
      if (data.run_id !== identity.runId || !data.model_sha256 || (sha && data.model_sha256 !== sha)) throw new Error("模型身份不匹配，请重新加载候选组。");
      ablationIdentity = identity;
      const policy = data.evidence_policy;
      el("ablationPolicy").textContent = policy ? `保守门槛：共同正常样本 ≥ ${policy.minimum_normal_rows}；共同异常样本 ≥ ${policy.minimum_abnormal_rows}、异常窗口 ≥ ${policy.minimum_abnormal_windows}；允许性能退化 ${ablationPercent(policy.allowed_degradation)}。覆盖须完整、训练时间戳须一致且验证无失败；结论仍需人工判断。` : "";
      if (action === "candidates") {
        ablationCandidates = data; ablationEvaluation = null; ablationDrops.clear(); ablationLocks.clear();
        el("ablationTags").replaceChildren(); el("ablationEvidence").replaceChildren();
        el("ablationGroup").replaceChildren(new Option("请选择候选组", ""), ...data.groups.map(group => new Option(`${group.id}：${group.tags.join("、")}`, String(group.id))));
        const diagnostics = data.diagnostics || {};
        ablationTable(el("ablationDiagnostics"), ["变量 / 变量对", "有效样本数", "原因"], [
          ...(diagnostics.tag_profiles || []).filter(item => item.flags?.length).map(item => [item.tag, item.profile?.valid_count, item.flags.join("；")]),
          ...(diagnostics.unavailable_pairs || []).map(pair => [`${pair.tag_a} / ${pair.tag_b}`, pair.valid_pair_count, pair.unavailable_reason]),
        ]);
        ablationMessage(data.groups.length ? `已加载 ${data.groups.length} 个候选组；|Pearson r| ≥ ${diagnostics.correlation_threshold ?? 0.95}，单次最多剔除 ${data.max_drop_tags} 个 Tag（分别训练）。` : "未发现高相关变量组。");
      } else {
        ablationEvaluation = data;
        renderAblationSchemes(data);
      }
    } catch (error) {
      if (ablationCurrent(identity)) { ablationCandidates = ablationEvaluation = ablationIdentity = null; ablationMessage(`冗余消融失败：${error.message}`, "error"); }
    } finally {
      if (identity.revision === ablationRevision) {
        ablationBusy = false;
        if (!ablationCurrent(identity)) window.invalidateRedundancyAblation("模型或配置已变化");
        else updateAblationActions();
      }
    }
  }
  el("ablationCandidatesButton").addEventListener("click", () => requestAblation("candidates"));
  el("ablationEvaluateButton").addEventListener("click", () => requestAblation("evaluate"));
  function ablationValidationText(value, abnormal) {
    if (!value) return abnormal ? "异常检出能力尚未验证" : "未提供正常验证";
    const metrics = value.metrics || {};
    const coverage = `有效 ${value.original_rows ?? "—"} / 共同 ${value.common_rows ?? "—"}；覆盖 ${ablationPercent(value.coverage)} / 共同覆盖 ${ablationPercent(value.common_coverage)}`;
    if (!(value.common_rows > 0)) return `无共同有效验证样本；${coverage}${abnormal ? "；异常检出能力尚未验证" : ""}`;
    const labels = Object.entries(value.status_by_engineering_label || {}).map(([label, counts]) => `${label}→正常 ${counts.normal ?? 0} / 关注 ${counts.attention ?? 0} / 异常 ${counts.abnormal ?? 0}`).join("；");
    const labelText = labels ? `；工程标签：${labels}` : "";
    if (abnormal) {
      const independent = [["T²", "t2"], ["SPE", "spe"]].map(([label, key]) => `${label}检出窗口 95% / 99% ${metrics[`${key}_detected_window_count_95`] ?? "—"} / ${metrics[`${key}_detected_window_count_99`] ?? "—"}（共 ${metrics.valid_window_count ?? "—"} 个异常窗口）`).join("；");
      return `${coverage}；检出覆盖率95% ${ablationPercent(metrics.detection_coverage_rate_95)}；异常窗口检出率95% / 99% ${ablationPercent(metrics.detection_rate_95)} / ${ablationPercent(metrics.detection_rate_99)}；${independent}；连续检出事件 ${metrics.continuous_detection_event_count_95 ?? "—"}${labelText}`;
    }
    return `${coverage}；${["t2", "spe", "overall"].map((key, i) => `${["T²", "SPE", "综合"][i]}95% / 99% ${ablationPercent(metrics[key]?.exceedance_rate_95)} / ${ablationPercent(metrics[key]?.exceedance_rate_99)}`).join("；")}；连续误报事件 ${metrics.continuous_false_alarm_event_count_95 ?? "—"}；最长误报 ${metrics.longest_continuous_false_alarm_minutes ?? "—"} min${labelText}`;
  }
  function renderAblationSchemes(data) {
    const deltaLabels = {detection_rate_95: ["异常窗口检出率95%", "rate"], detection_rate_99: ["异常窗口检出率99%", "rate"],
      detection_coverage_rate_95: ["检出覆盖率95%", "rate"], t2_detected_window_count_95: ["T²检出窗口95%", "count"],
      t2_detected_window_count_99: ["T²检出窗口99%", "count"], spe_detected_window_count_95: ["SPE检出窗口95%", "count"],
      spe_detected_window_count_99: ["SPE检出窗口99%", "count"]};
    ablationTable(el("ablationResults"), ["选择", "方案 / 剔除Tag", "结论 / 原因", "状态", "正常验证", "已知异常验证", "检出变化", "Tag数 / 动态维度 / 训练样本 / 共同训练样本", "主元 / 累计解释率", "训练时间戳一致"], data.schemes.map(scheme => {
      const radio = document.createElement("input"); radio.type = "radio"; radio.name = "ablationScheme"; radio.setAttribute("aria-label", `选择方案 ${scheme.name}`);
      radio.disabled = scheme.status !== "evaluated" || !scheme.dropped_tag;
      radio.addEventListener("change", () => { ablationScheme = scheme; updateAblationActions(); });
      const abnormal = scheme.validation?.known_abnormal;
      const reasons = [...(scheme.reasons || []), ...(scheme.validation_errors || []).map(error => typeof error === "string" ? error : `${error.window_id}（${error.type}）：${error.error}`)];
      if (!(abnormal?.common_rows > 0)) reasons.push("异常检出能力尚未验证");
      return [radio, `${scheme.name} / ${scheme.dropped_tag || "保留全部"}`, `${scheme.conclusion || "证据不足"}；${reasons.join("；")}`,
        scheme.status === "failed" ? `评估失败：${scheme.error || "未提供失败原因"}` : scheme.validation_errors?.length ? "验证不完整" : "已评估",
        scheme.status === "failed" ? "—" : ablationValidationText(scheme.validation?.normal_validation, false), scheme.status === "failed" ? "—" : ablationValidationText(abnormal, true),
        Object.entries(scheme.detection_delta || {}).map(([key, value]) => {
          const [label, kind] = deltaLabels[key] || [key, "rate"];
          return `${label}：${Number.isFinite(value) ? kind === "count" ? `${value} 个窗口` : `${(value * 100).toFixed(1)} 个百分点` : "—"}`;
        }).join("；") || "—",
        scheme.status === "failed" ? "—" : `${scheme.tag_count} / ${scheme.dynamic_features} / ${scheme.training_rows} / ${scheme.training_common_rows ?? "—"}`,
        scheme.status === "failed" ? "—" : `${scheme.n_components} / ${ablationPercent(scheme.cumulative_explained_variance)}`,
        scheme.training_timestamps_match === true ? "一致" : "不一致或未确认：训练样本时间戳不同，变化不能全部归因于变量剔除，存在可比性限制"];
    }));
    el("ablationWarnings").textContent = (data.warnings || []).map(item => typeof item === "string" ? item : item.message || JSON.stringify(item)).join("；");
    const partial = data.schemes.some(scheme => scheme.status === "failed" || scheme.validation_errors?.length);
    ablationMessage(partial ? "评估完成；部分方案评估失败或验证不完整，请查看具体原因。" : "评估完成；请选择方案后人工确认应用。", partial ? "warning" : "success");
  }
  el("ablationApplyButton").addEventListener("click", () => {
    if (!ablationCurrent(ablationIdentity) || !ablationEvaluation || ablationScheme?.status !== "evaluated") { window.invalidateRedundancyAblation("模型或配置已变化"); return; }
    const tag = ablationScheme.dropped_tag, tags = selectedTags();
    if (!tag || !tags.includes(tag) || ablationLocks.has(tag)) return;
    if (!window.confirm(`将从待训练配置移除 ${tag}，剩余 ${tags.length - 1} 个 Tag。原基准模型和已有验证结果保留。应用后须重新执行建模质量检查并正式训练。是否继续？`)) return;
    if (!ablationCurrent(ablationIdentity)) { window.invalidateRedundancyAblation("配置已变化"); return; }
    state.selectedModelTags.delete(tag);
    invalidateModellingResults(`消融方案已应用：移除 ${tag}`); renderTagList();
    showTrainingTool("trainingMainPanel"); setStatus(`已移除待训练 Tag ${tag}；请重新执行建模质量检查并正式训练。原模型及验证结果保留。`, "warning");
  });

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
    if (exceedanceDialog.open) exceedanceDialog.close();
    ablationInvalid = false;
    window.invalidateRedundancyAblation();
    ablationMessage(ablationEligible() ? "正常状态候选模型已就绪，可加载冗余候选组。" : "仅支持 normal_state / candidate 基准模型，请先训练正常状态候选模型。");
    renderModelQuality(data.model_quality, data.training_window_totals);
    renderTrainingConditionDiagnostic(data.model_quality?.training_condition_diagnostic);
    drawLoadingPlot(data.loading_plot);
    renderComponentLoadings(data.loading_plot?.component_loadings);
    renderSingleModelDiagnostic(data.model_diagnostic);
    refreshCandidateOptions(data.run_id);
  };

  const exceedanceDialog = el("trainingExceedanceDialog");
  let exceedanceTimezoneAware = false, exceedanceSelectionRevision = 0;
  let exceedanceSession = 0, intervalRequest = 0, trendRequest = 0, compareRequest = 0;
  let exceedanceIdentity = null, exceedanceOverview = null, exceedanceInterval = null, exceedanceComparison = null;
  let exceedanceSelectionMode = "single";
  let exceedanceQueue = Promise.resolve();
  const exceedancePlots = [el("exceedanceT2"), el("exceedanceSpe")];
  exceedancePlots.forEach(plot => plot.addEventListener("pointerdown",() => {
    plot._exceedanceSelection={mode:exceedanceSelectionMode,revision:exceedanceSelectionRevision};
  },true));
  const exceedanceNumber = value => Number.isFinite(value) ? value.toPrecision(6) : "—";
  const exceedanceTime = value => {
    if (typeof value !== "number") return String(value).replace(" ", "T");
    const iso = new Date(value).toISOString();
    return exceedanceTimezoneAware ? iso : iso.replace(/Z$/, "");
  };
  function exceedanceCurrent(session) {
    return !state.trainingDiagnosticInvalidated && exceedanceDialog.open && session === exceedanceSession && exceedanceIdentity &&
      state.runId === exceedanceIdentity.runId && state.training === exceedanceIdentity.training &&
      state.training?.run_id === state.runId;
  }
  function exceedanceMessage(text) { el("exceedanceStatus").textContent = text; }
  function exceedanceQuery(action, fields, session, current = () => true) {
    const runId = exceedanceIdentity.runId;
    // 同一训练运行的后端查询持有生命周期锁；串行并跳过已经被新选择替代的请求。
    exceedanceQueue = exceedanceQueue.catch(() => {}).then(async () => {
      if (!exceedanceCurrent(session) || !current()) return null;
      const data = await api("/api/training-exceedance", {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({run_id:runId, action, ...fields})});
      if (!exceedanceCurrent(session) || !current()) return null;
      if (data.run_id !== runId) throw new Error("训练身份不匹配，请重新打开诊断。");
      return data;
    });
    return exceedanceQueue;
  }
  function exceedanceTable(target, headers, rows) {
    const table = document.createElement("table"), head = table.createTHead().insertRow();
    headers.forEach(text => { const th = document.createElement("th"); th.textContent = text; head.append(th); });
    const body = table.createTBody();
    rows.forEach(values => { const row = body.insertRow(); values.forEach(value => {
      const td = row.insertCell(); if (value instanceof Node) td.append(value); else td.textContent = value;
    }); });
    target.replaceChildren(table);
  }
  function exceedanceShapes(start, end) {
    return [{type:"rect", xref:"x", yref:"paper", x0:start, x1:end, y0:0, y1:1,
      fillcolor:"rgba(234,179,8,.16)", line:{width:0}, layer:"below"}];
  }
  function validExceedanceRange(start,end) {
    start=start.trim();end=end.trim();
    return start&&end&&Number.isFinite(Date.parse(start))&&Number.isFinite(Date.parse(end))&&Date.parse(start)<=Date.parse(end)
      ? {start,end} : null;
  }
  function syncExceedanceShapes() {
    const shapes = [];
    const single=validExceedanceRange(el("exceedanceStart").value,el("exceedanceEnd").value);
    if (single) shapes.push({...exceedanceShapes(single.start,single.end)[0],name:"单区间"});
    for (const [kind,color] of [["anomaly","rgba(220,38,38,.18)"],["reference","rgba(37,99,235,.18)"]]) {
      const range = compareRangeFromInputs(kind);
      if (range) shapes.push({type:"rect",xref:"x",yref:"paper",x0:range.start,x1:range.end,y0:0,y1:1,
        fillcolor:color,line:{width:1,color:kind==="anomaly"?"#dc2626":"#2563eb"},layer:"below",name:kind});
    }
    exceedancePlots.forEach(plot => { if (plot._fullLayout) Plotly.relayout(plot,{shapes}); });
  }
  function exceedanceSeries(rows, field, gapField) {
    const x = [], y = [];
    rows.forEach(row => { if (row[gapField]) { x.push(row.timestamp); y.push(null); }
      x.push(row.timestamp); y.push(Number.isFinite(row[field]) ? row[field] : null); });
    return {x,y,type:"scatter",mode:"lines",connectgaps:false};
  }
  async function drawExceedanceOverview(data, session) {
    for (const [index, metric, limits, title] of [[0,"t2",data.t2_limits,"T²"],[1,"spe",data.q_limits,"SPE"]]) {
      const target = exceedancePlots[index];
      await Plotly.newPlot(target, [{...exceedanceSeries(data.scores,metric,"break_before"),name:title},
        ...[95,99].map(level => ({x:[data.start,data.end],y:[limits[level],limits[level]],type:"scatter",mode:"lines",
          name:`${level}% 控制限`,line:{dash:level === 95 ? "dash" : "dot"}}))],
        {title:{text:title},margin:{l:65,r:20,t:35,b:35},dragmode:"select",selectdirection:"h",
          xaxis:{type:"date",range:[data.start,data.end]},yaxis:{fixedrange:false},legend:{orientation:"h"}},
        {responsive:true,displaylogo:false,modeBarButtonsToAdd:["select2d","zoom2d","pan2d","resetScale2d"]});
      if (!exceedanceCurrent(session)) return;
      target.on("plotly_selected", event => {
        const selection=target._exceedanceSelection;target._exceedanceSelection=null;
        if (!selection || selection.revision!==exceedanceSelectionRevision || !exceedanceCurrent(session) || !event?.range?.x) return;
        const range = event.range.x.map(exceedanceTime);
        if (selection.mode === "anomaly" || selection.mode === "reference")
          setExceedanceCompareRange(selection.mode,...range);
        else selectExceedanceInterval(...range);
      });
      target.on("plotly_relayout", event => {
        if (!exceedanceCurrent(session)) return;
        const other = exceedancePlots[1-index];
        if (!other._fullLayout) return;
        const range = event["xaxis.range"] || (event["xaxis.range[0]"] !== undefined ? [event["xaxis.range[0]"],event["xaxis.range[1]"]] : null);
        if (range && JSON.stringify(other.layout.xaxis.range) !== JSON.stringify(range)) Plotly.relayout(other,{"xaxis.range":range});
        else if (event["xaxis.autorange"] === true && other.layout.xaxis.autorange !== true) Plotly.relayout(other,{"xaxis.autorange":true});
      });
    }
  }
  async function selectExceedanceInterval(start, end) {
    const session = exceedanceSession, request = ++intervalRequest;
    ++trendRequest; ++exceedanceSelectionRevision;
    exceedanceInterval = null;
    el("exceedanceRawSection").hidden = true;
    el("exceedanceStatistics").replaceChildren(); el("exceedanceContributions").replaceChildren();
    if (!exceedanceCurrent(session)) { exceedanceMessage("训练结果已变化，请关闭后重新打开诊断。"); return; }
    if (!start || !end || !Number.isFinite(Date.parse(start)) || !Number.isFinite(Date.parse(end)) || Date.parse(start) > Date.parse(end)) {
      syncExceedanceShapes();
      exceedanceMessage("请输入有效的开始、结束时间，且开始时间不晚于结束时间。"); return;
    }
    el("exceedanceStart").value = exceedanceTime(start); el("exceedanceEnd").value = exceedanceTime(end);
    syncExceedanceShapes();
    exceedanceMessage("正在计算区间统计与贡献…");
    try {
      const data = await exceedanceQuery("interval",{start,end},session,() => request === intervalRequest);
      if (!data || request !== intervalRequest) return;
      exceedanceInterval = data;
      const rows = [["分析区间",`${data.start} — ${data.end}`],["有效样本数",String(data.valid_samples)]];
      for (const [label, field] of [["平均值","mean"],["最大值","max"],["最大超限倍数（95%）","max_limit_ratio"]])
        rows.push([label,...["t2","spe"].map(metric => exceedanceNumber(data.statistics[metric][field]))]);
      for (const level of [95,99]) rows.push([`${level}% 超限样本数 / 比例`,...["t2","spe"].map(metric => {
        const value = data.statistics[metric][`exceedance_${level}`]; return `${value.count} / ${ablationPercent(value.rate)}`;
      })]);
      exceedanceTable(el("exceedanceStatistics"),["统计项目","T²","SPE"],rows);
      // 「区间」「样本数」对 T²/SPE 取值相同：跨两列显示，并标 .numeric 取右对齐，
      // 与下方数值列的右边线一致（此前 T² 表头左对齐、数据右对齐）。
      [...el("exceedanceStatistics").querySelectorAll("tbody tr")].slice(0,2).forEach(row => {
        row.cells[1].colSpan = 2; row.cells[1].classList.add("numeric");
      });
      renderExceedanceContributions();
      exceedanceMessage(data.valid_samples ? "区间分析完成。" : "所选区间没有有效评分样本。");
    } catch (error) { if (exceedanceCurrent(session) && request === intervalRequest) exceedanceMessage(error.message); }
  }
  function renderExceedanceContributions() {
    const target = el("exceedanceContributions");
    ++trendRequest; el("exceedanceRawSection").hidden = true;
    if (!exceedanceCurrent(exceedanceSession)) { target.replaceChildren(); exceedanceMessage("训练结果已变化，请关闭后重新打开诊断。"); return; }
    if (!exceedanceInterval) return;
    const contribution = exceedanceInterval.contributions[el("exceedanceMetric").value];
    if (!contribution.sample_count || !contribution.rows.length) { target.textContent = "当前统计量没有超限样本或可用贡献。"; return; }
    const showPct = contribution.rows.some(row => Number.isFinite(row.contribution_pct));
    exceedanceTable(target,["排名","原始 Tag","平均贡献",...(showPct ? ["贡献占比"] : []),"主要 Lag（min）"],
      contribution.rows.slice(0,10).map((row,index) => {
        const button = document.createElement("button"); button.type = "button"; button.className = "secondary"; button.textContent = row.tag;
        button.addEventListener("click", () => showExceedanceTag(row.tag));
        return [String(index+1),button,exceedanceNumber(row.mean_contribution),
          ...(showPct ? [Number.isFinite(row.contribution_pct) ? `${exceedanceNumber(row.contribution_pct)}%` : "—"] : []),
          `${row.lag_start_minutes ?? "—"} – ${row.lag_end_minutes ?? "—"}`];
      }));
  }
  async function showExceedanceTag(tag) {
    const session = exceedanceSession, request = ++trendRequest, interval = exceedanceInterval;
    if (!interval || !exceedanceCurrent(session)) return;
    el("exceedanceRawSection").hidden = false;
    el("exceedanceRawTitle").textContent = `${tag} · 原始趋势`;
    el("exceedanceRawStatus").textContent = "正在加载原始数据…";
    Plotly.purge(el("exceedanceRaw"));
    try {
      const data = await exceedanceQuery("trend",{start:interval.start,end:interval.end,tag},session,() => request === trendRequest);
      if (!data || request !== trendRequest) return;
      if (data.tag !== tag) throw new Error("原始 Tag 身份不匹配。");
      el("exceedanceRawTitle").textContent = `${data.tag}${data.unit ? `（${data.unit}）` : ""} · 原始趋势`;
      el("exceedanceRawStatus").textContent = !data.rows.some(row => Number.isFinite(row.value)) ? "没有可用原始数据，不进行插值。" :
        data.rows.some(row => !Number.isFinite(row.value)) ? "原始数据存在缺失，按缺失断线，不插值。" : "";
      await Plotly.newPlot(el("exceedanceRaw"),[{...exceedanceSeries(data.rows,"value","gap_start"),name:data.tag}],
        {margin:{l:65,r:20,t:20,b:35},xaxis:{type:"date",range:[data.start,data.end]},yaxis:{title:data.unit || "原始值"},
          shapes:exceedanceShapes(data.analysis_start,data.analysis_end)}, {responsive:true,displaylogo:false});
    } catch (error) { if (exceedanceCurrent(session) && request === trendRequest) el("exceedanceRawStatus").textContent = error.message; }
  }
  function compareRangeFromInputs(kind) {
    const prefix = kind === "anomaly" ? "exceedanceAnomaly" : "exceedanceReference";
    return validExceedanceRange(el(`${prefix}Start`).value,el(`${prefix}End`).value);
  }
  function clearExceedanceCompareOutput() {
    exceedanceComparison = null;
    el("exceedanceCompareStatistics").replaceChildren();
    el("exceedanceCompareContributions").replaceChildren();
    el("exceedanceCompareRawSection").hidden = true;
    if (globalThis.Plotly) { Plotly.purge(el("exceedanceCompareRaw")); Plotly.purge(el("exceedanceCompareLoadRaw")); }
  }
  function invalidateExceedanceCompare(message = "") {
    ++compareRequest; ++trendRequest; clearExceedanceCompareOutput();
    if (message) el("exceedanceCompareStatus").textContent = message;
  }
  function setExceedanceCompareRange(kind, start, end) {
    const prefix = kind === "anomaly" ? "exceedanceAnomaly" : "exceedanceReference";
    el(`${prefix}Start`).value = start; el(`${prefix}End`).value = end;
    invalidateExceedanceCompare(); syncExceedanceShapes();
    el("exceedanceCompareStatus").textContent = `${kind === "anomaly" ? "异常" : "参考"}区间已选，可修改时间或清除。`;
  }
  function editExceedanceCompareRange() {
    ++exceedanceSelectionRevision;
    invalidateExceedanceCompare("时间范围已修改，请重新比较。"); syncExceedanceShapes();
  }
  function setExceedanceSelectionMode(mode) {
    ++exceedanceSelectionRevision;exceedanceSelectionMode = mode;
    for (const [key,id] of [["single","exceedanceModeSingle"],["anomaly","exceedanceModeAnomaly"],["reference","exceedanceModeReference"]])
      el(id).setAttribute("aria-pressed",String(key===mode));
    exceedanceMessage(mode === "single" ? "横向框选单区间，或手动输入时间。" : `横向框选${mode === "anomaly" ? "异常" : "参考"}区间。`);
  }
  function exceedancePercent(value) { return Number.isFinite(value) ? `${(value*100).toFixed(1)}%` : "—"; }
  function exceedanceShare(value) { return Number.isFinite(value) ? `${value.toFixed(1)}%` : "—"; }
  function showExceedanceComparison(data) {
    const groups = [data.anomaly,data.reference], labels = ["异常组","低 SPE 参考组"];
    const loadStatus = {insufficient_evidence:"证据不足；不能认定同负荷",
      body_disjoint:"主体分布分离；请重新选择参考区间",
      manual_review:"需工程师核对；不自动认定同负荷"}[data.load_comparability_status] || "负荷可比性不可判断，请工程师核对";
    const loadCentralOverlap = data.load_p10_p90_overlap === true ? "相交（仍需工程师核对）" :
      data.load_p10_p90_overlap === false ? "不重叠" : "不可判断（负荷有效样本不足）";
    const rows = [
      ["时间区间",`${data.anomaly_start} — ${data.anomaly_end}`,`${data.reference_start} — ${data.reference_end}`],
      ["区间有效评分样本",...groups.map(group=>String(group.interval_valid_samples))],
      ["实际纳入样本",...groups.map(group=>String(group.sample_count))],
      ["SPE 均值",...groups.map(group=>exceedanceNumber(group.spe_mean))],
      ["SPE ≥ 95% 控制限",...groups.map(group=>`${group.spe_exceedance_count} / ${exceedancePercent(group.spe_exceedance_rate)}`)],
      ["负荷有效样本 / 缺失率",...groups.map(group=>`${group.load.valid_samples} / ${exceedancePercent(group.load.missing_rate)}`)],
      ["负荷均值 / 中位数",...groups.map(group=>`${exceedanceNumber(group.load.mean)} / ${exceedanceNumber(group.load.median)}`)],
      ["负荷 P10 / P90",...groups.map(group=>`${exceedanceNumber(group.load.p10)} / ${exceedanceNumber(group.load.p90)}`)],
      ["负荷范围",...groups.map(group=>`${exceedanceNumber(group.load.minimum)} – ${exceedanceNumber(group.load.maximum)}`)],
      ["负荷范围重叠",...groups.map(()=>data.load_ranges_overlap===null?"不可判断":data.load_ranges_overlap?"是":"否")],
      ["负荷可比性状态",loadStatus,loadStatus],
      ["P10–P90 主体区间重叠",loadCentralOverlap,loadCentralOverlap],
    ];
    exceedanceTable(el("exceedanceCompareStatistics"),["统计项",...labels],rows);
    [...el("exceedanceCompareStatistics").querySelectorAll("tbody tr")].slice(-2).forEach(row=>{
      row.cells[1].colSpan=2;row.cells[2].remove();
    });
    const warnings = data.warnings || [];
    el("exceedanceCompareStatus").textContent = warnings.join(" ") || (data.analysis_available ? "两区间统计完成。" : "样本不足，未生成贡献对比。");
    if (!data.contributions.length) {
      el("exceedanceCompareContributions").textContent = data.analysis_available ? "没有可显示的 SPE 残差²贡献。" : "异常组或参考组样本不足，未生成贡献对比。";
      return;
    }
    exceedanceTable(el("exceedanceCompareContributions"),["排名","原始 Tag","异常残差²均值","参考残差²均值","绝对差值","异常组占比","参考组占比","异常主要 Lag（min）","参考主要 Lag（min）"],
      data.contributions.map((row,index)=>{
        const button=document.createElement("button");button.type="button";button.className="secondary";button.textContent=row.tag;
        button.addEventListener("click",()=>showExceedanceComparisonTag(row.tag));
        const lag=(start,end)=>start==null?"—":start===end?String(start):`${start}–${end}`;
        return [String(index+1),button,exceedanceNumber(row.anomaly_mean),exceedanceNumber(row.reference_mean),
          exceedanceNumber(row.absolute_difference),exceedanceShare(row.anomaly_share_pct),
          exceedanceShare(row.reference_share_pct),lag(row.anomaly_lag_start_minutes,row.anomaly_lag_end_minutes),
          lag(row.reference_lag_start_minutes,row.reference_lag_end_minutes)];
      }));
  }
  async function compareExceedanceRanges() {
    const session=exceedanceSession, request=++compareRequest;
    ++trendRequest;clearExceedanceCompareOutput();
    if(!exceedanceCurrent(session)){el("exceedanceCompareStatus").textContent="训练结果已变化，请关闭后重新打开诊断。";return;}
    const anomaly=compareRangeFromInputs("anomaly"), reference=compareRangeFromInputs("reference"), loadTag=el("exceedanceLoadTag").value;
    if(!anomaly||!reference){el("exceedanceCompareStatus").textContent="请为异常区间和参考区间输入有效起止时间。";return;}
    if(Date.parse(anomaly.start)<=Date.parse(reference.end)&&Date.parse(reference.start)<=Date.parse(anomaly.end)){
      el("exceedanceCompareStatus").textContent="异常区间与参考区间不能重叠（边界时间也计入区间）。";return;
    }
    if(!loadTag){el("exceedanceCompareStatus").textContent="请选择训练源数据中的数值型负荷 Tag。";return;}
    syncExceedanceShapes();
    el("exceedanceCompareStatus").textContent="正在计算两区间 SPE 与负荷分布…";
    try{
      const data=await exceedanceQuery("compare",{anomaly_start:anomaly.start,anomaly_end:anomaly.end,
        reference_start:reference.start,reference_end:reference.end,load_tag:loadTag},session,()=>request===compareRequest);
      if(!data||request!==compareRequest)return;
      if(data.load_tag!==loadTag)throw new Error("负荷 Tag 身份不匹配，请重新选择。");
      exceedanceComparison=data;showExceedanceComparison(data);
    }catch(error){if(exceedanceCurrent(session)&&request===compareRequest)el("exceedanceCompareStatus").textContent=error.message;}
  }
  function pairedTrendAxis(rowsA,rowsB) {
    const values=[...rowsA,...rowsB].map(row=>row.value).filter(Number.isFinite);
    if(!values.length)return undefined;
    const low=Math.min(...values),high=Math.max(...values),pad=high>low?(high-low)*.05:Math.max(Math.abs(low)*.05,1e-6);
    return [low-pad,high+pad];
  }
  function drawPairedRawTrend(target,tag,unit,anomaly,reference) {
    const series=(range)=>range.series.find(item=>item.tag===tag)||{rows:[]};
    const a=series(anomaly),r=series(reference),axis=pairedTrendAxis(a.rows,r.rows);
    const traces=[{...exceedanceSeries(a.rows,"value","gap_start"),name:`异常 ${tag}`,xaxis:"x",yaxis:"y"},
      {...exceedanceSeries(r.rows,"value","gap_start"),name:`参考 ${tag}`,xaxis:"x2",yaxis:"y"}];
    const shapes=[{type:"rect",xref:"x",yref:"paper",x0:anomaly.analysis_start,x1:anomaly.analysis_end,y0:0,y1:1,
      fillcolor:"rgba(220,38,38,.12)",line:{width:1,color:"#dc2626"},layer:"below"},
      {type:"rect",xref:"x2",yref:"paper",x0:reference.analysis_start,x1:reference.analysis_end,y0:0,y1:1,
      fillcolor:"rgba(37,99,235,.12)",line:{width:1,color:"#2563eb"},layer:"below"}];
    return Plotly.newPlot(target,traces,{margin:{l:65,r:20,t:20,b:55},
      xaxis:{type:"date",domain:[0,.48],range:[anomaly.start,anomaly.end],title:{text:"异常区间时间"}},
      xaxis2:{type:"date",domain:[.52,1],anchor:"y",range:[reference.start,reference.end],title:{text:"参考区间时间"}},
      yaxis:{title:unit||"原始值",...(axis?{range:axis}:{})},shapes,legend:{orientation:"h"}},
      {responsive:true,displaylogo:false});
  }
  function rawPairStatus(anomaly,reference,tag) {
    const series=[anomaly.series.find(item=>item.tag===tag),reference.series.find(item=>item.tag===tag)].filter(Boolean);
    const rows=series.flatMap(item=>item.rows),finite=rows.some(row=>Number.isFinite(row.value));
    const missing=rows.some(row=>!Number.isFinite(row.value));
    return `${anomaly.analysis_start} — ${anomaly.analysis_end}（异常）；${reference.analysis_start} — ${reference.analysis_end}（参考）。`+
      (!finite?"没有有效原始值；未插值。":missing?"原始缺失按断线显示，未插值":"");
  }
  async function showExceedanceComparisonTag(tag) {
    const session=exceedanceSession,request=++trendRequest,comparison=exceedanceComparison;
    if(!comparison||!exceedanceCurrent(session))return;
    el("exceedanceCompareRawSection").hidden=false;
    el("exceedanceCompareRawTitle").textContent=`${tag} · 异常与参考原始趋势`;
    el("exceedanceCompareRawStatus").textContent="正在加载原始趋势…";
    el("exceedanceCompareLoadTitle").textContent="";el("exceedanceCompareLoadStatus").textContent="";
    Plotly.purge(el("exceedanceCompareRaw"));Plotly.purge(el("exceedanceCompareLoadRaw"));
    try{
      const data=await exceedanceQuery("compare_trend",{tag,load_tag:comparison.load_tag,
        anomaly_start:comparison.anomaly_start,anomaly_end:comparison.anomaly_end,
        reference_start:comparison.reference_start,reference_end:comparison.reference_end},session,
        ()=>request===trendRequest&&comparison===exceedanceComparison);
      if(!data||request!==trendRequest)return;
      if(data.tag!==tag||data.load_tag!==comparison.load_tag)throw new Error("趋势 Tag 身份不匹配。");
      const unit=data.anomaly.series.find(item=>item.tag===tag)?.unit||"";
      el("exceedanceCompareRawTitle").textContent=`${tag}${unit?`（${unit}）`:""} · 异常与参考原始趋势`;
      el("exceedanceCompareRawStatus").textContent=rawPairStatus(data.anomaly,data.reference,tag);
      await drawPairedRawTrend(el("exceedanceCompareRaw"),tag,unit,data.anomaly,data.reference);
      if(!exceedanceCurrent(session)||request!==trendRequest||comparison!==exceedanceComparison)return;
      if(data.load_tag!==tag){
        const loadUnit=data.anomaly.series.find(item=>item.tag===data.load_tag)?.unit||"";
        const loadRows=[data.anomaly.series.find(item=>item.tag===data.load_tag),data.reference.series.find(item=>item.tag===data.load_tag)].filter(Boolean).flatMap(item=>item.rows);
        el("exceedanceCompareLoadTitle").textContent=`${data.load_tag}${loadUnit?`（${loadUnit}）`:""} · 负荷原始趋势`;
        el("exceedanceCompareLoadStatus").textContent=loadRows.some(row=>Number.isFinite(row.value))?rawPairStatus(data.anomaly,data.reference,data.load_tag):"负荷 Tag 在所选区间内没有有效原始值；未插值。";
        await drawPairedRawTrend(el("exceedanceCompareLoadRaw"),data.load_tag,loadUnit,data.anomaly,data.reference);
      }
    }catch(error){if(exceedanceCurrent(session)&&request===trendRequest)el("exceedanceCompareRawStatus").textContent=error.message;}
  }
  el("trainingExceedanceButton").addEventListener("click", async () => {
    if (state.trainingDiagnosticInvalidated || !state.runId || state.training?.run_id !== state.runId) { alert("请先完成当前模型训练。"); return; }
    if (exceedanceIdentity?.runId !== state.runId) exceedanceQueue = Promise.resolve();
    exceedanceIdentity = {runId:state.runId,training:state.training};
    const session = ++exceedanceSession;
    exceedanceOverview = null; exceedanceInterval = null;
    exceedanceComparison = null;
    el("exceedanceStatistics").replaceChildren(); el("exceedanceContributions").replaceChildren();
    el("exceedanceRawSection").hidden = true; el("exceedanceStart").value = ""; el("exceedanceEnd").value = "";
    el("exceedanceApply").disabled = true; el("exceedanceReset").disabled = true;
    ["exceedanceAnomalyStart","exceedanceAnomalyEnd","exceedanceReferenceStart","exceedanceReferenceEnd"].forEach(id=>el(id).value="");
    el("exceedanceLoadTag").replaceChildren(new Option("请选择数值型 Tag",""));el("exceedanceLoadTag").disabled=true;
    el("exceedanceCompareApply").disabled=true;clearExceedanceCompareOutput();
    exceedancePlots.forEach(plot => { if (globalThis.Plotly) Plotly.purge(plot); plot.replaceChildren(); });
    exceedanceDialog.showModal(); setExceedanceSelectionMode("single"); exceedanceMessage("正在加载完整训练趋势…");
    try {
      if (!globalThis.Plotly) throw new Error("Plotly 未加载，无法显示诊断趋势。");
      const data = await exceedanceQuery("overview",{},session);
      if (!data) return;
      exceedanceOverview = data;
      exceedanceTimezoneAware = /(?:Z|[+-]\d{2}:\d{2})$/i.test(data.start);
      const loadSelector=el("exceedanceLoadTag");
      loadSelector.replaceChildren(new Option("请选择数值型 Tag",""));
      (data.load_tags||[]).forEach(tag=>loadSelector.add(new Option(tag,tag)));
      loadSelector.disabled=false;el("exceedanceCompareApply").disabled=false;
      await drawExceedanceOverview(data,session);
      if (!exceedanceCurrent(session)) return;
      syncExceedanceShapes();
      el("exceedanceStart").value = exceedanceTime(data.start); el("exceedanceEnd").value = exceedanceTime(data.end);
      el("exceedanceApply").disabled = false; el("exceedanceReset").disabled = false;
      exceedanceMessage("请选择分析区间，随后计算统计与贡献。");
    } catch (error) { if (exceedanceCurrent(session)) exceedanceMessage(error.message); }
  });
  el("exceedanceClose").addEventListener("click", () => exceedanceDialog.close());
  exceedanceDialog.addEventListener("close", () => { ++exceedanceSession; ++intervalRequest; ++trendRequest; ++compareRequest; ++exceedanceSelectionRevision; });
  el("exceedanceModeSingle").addEventListener("click",()=>setExceedanceSelectionMode("single"));
  el("exceedanceModeAnomaly").addEventListener("click",()=>setExceedanceSelectionMode("anomaly"));
  el("exceedanceModeReference").addEventListener("click",()=>setExceedanceSelectionMode("reference"));
  el("exceedanceApply").addEventListener("click", () => selectExceedanceInterval(el("exceedanceStart").value,el("exceedanceEnd").value));
  ["exceedanceStart","exceedanceEnd"].forEach(id => el(id).addEventListener("change", () => {
    if (exceedanceOverview) selectExceedanceInterval(el("exceedanceStart").value,el("exceedanceEnd").value);
  }));
  [["anomaly","exceedanceAnomalyStart","exceedanceAnomalyEnd"],["reference","exceedanceReferenceStart","exceedanceReferenceEnd"]].forEach(([kind,start,end])=>{
    [start,end].forEach(id=>el(id).addEventListener("input",editExceedanceCompareRange));
  });
  el("exceedanceAnomalyClear").addEventListener("click",()=>{
    ++exceedanceSelectionRevision;el("exceedanceAnomalyStart").value="";el("exceedanceAnomalyEnd").value="";
    invalidateExceedanceCompare("异常区间已清除。");syncExceedanceShapes();
  });
  el("exceedanceReferenceClear").addEventListener("click",()=>{
    ++exceedanceSelectionRevision;el("exceedanceReferenceStart").value="";el("exceedanceReferenceEnd").value="";
    invalidateExceedanceCompare("参考区间已清除。");syncExceedanceShapes();
  });
  el("exceedanceLoadTag").addEventListener("change",()=>invalidateExceedanceCompare("负荷 Tag 已修改，请重新比较。"));
  el("exceedanceCompareApply").addEventListener("click",compareExceedanceRanges);
  el("exceedanceMetric").addEventListener("change",renderExceedanceContributions);
  el("exceedanceReset").addEventListener("click", () => {
    if (!exceedanceOverview || !exceedanceCurrent(exceedanceSession)) return;
    exceedancePlots.forEach(plot => Plotly.relayout(plot,{"xaxis.range":[exceedanceOverview.start,exceedanceOverview.end],"yaxis.autorange":true}));
  });

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
    const number = value => value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
    const percent = value => value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
    const statistics = document.createElement("table");
    const head = document.createElement("thead"), heading = document.createElement("tr");
    // 表头与数值列取同一对齐边：数值列右对齐，标签列左对齐。
    [["指标", false], ["T²统计", true], ["SPE/Q统计", true]].forEach(([label, numeric]) => {
      const cell = document.createElement("th"); cell.textContent = label;
      if (numeric) cell.classList.add("numeric");
      heading.append(cell);
    });
    head.append(heading);
    const body = document.createElement("tbody");
    [
      ["平均值", statistic => number(statistic.mean)],
      ["95% 控制限", statistic => number(statistic.limits["95"])],
      ["99% 控制限", statistic => number(statistic.limits["99"])],
      ["95% 超限比例", statistic => percent(statistic.exceedance_rates["95"])],
      ["99% 超限比例", statistic => percent(statistic.exceedance_rates["99"])],
      ["有效 / 无效评分", statistic => `${statistic.valid_samples} / ${statistic.invalid_samples}`],
      ["趋势状态", statistic => ({stable:"较稳定（经验检查）", changing:"存在变化", insufficient:"数据不足"})[statistic.trend] || "未知"],
      ["已检查连续段", statistic => statistic.trend_segments_checked ?? 0],
    ].forEach(([label, value]) => {
      const row = document.createElement("tr"), title = document.createElement("th");
      title.textContent = label; row.append(title);
      ["t2", "spe"].forEach(key => {
        const cell = document.createElement("td"); cell.textContent = value(quality.statistics[key]); row.append(cell);
      });
      body.append(row);
    });
    statistics.append(head, body);
    const wrap = document.createElement("div"); wrap.className = "table-wrap"; wrap.tabIndex = 0;
    wrap.append(statistics); target.append(wrap);
    notice.append(paragraph(quality.notice));
    const ratio = quality.pc1_pc2_explained_variance;
    const ratioValue = document.createElement("strong");
    ratioValue.textContent = percent(ratio);
    const ratioLabel = document.createElement("span");
    ratioLabel.textContent = "PC1+PC2 二维解释率";
    projection.append(ratioLabel, ratioValue, paragraph(ratio == null ? "当前模型不足两个主元。" : ratio >= 0.8 ? "二维图具有较好代表性。" : "二维图仅覆盖部分变化，建议结合更多主元和统计量判断。"));
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
      item.append(label, share, details);
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

  function trainingAnomalyPeaks(points, intervalMinutes) {
    const events = [];
    let peak = null, previous = null;
    const finish = () => { if (peak) events.push(peak); peak = null; };
    for (const point of points) {
      const time = Date.parse(point.timestamp);
      if (point.break_before || (previous && (point.segment_id !== previous.segment_id || point.window_id !== previous.window_id || time <= Date.parse(previous.timestamp) || (intervalMinutes > 0 && time - Date.parse(previous.timestamp) !== intervalMinutes * 60000)))) finish();
      const t2 = Number.isFinite(point.t2_limit_ratio) ? point.t2_limit_ratio : -Infinity;
      const spe = Number.isFinite(point.spe_limit_ratio) ? point.spe_limit_ratio : -Infinity;
      const severity = Math.max(t2, spe);
      if (!Number.isFinite(time) || severity < 1) finish();
      else if (!peak || severity > peak.severity) peak = {...point, severity, peakType: t2 >= 1 && spe >= 1 ? "T² + SPE" : t2 >= 1 ? "T²" : "SPE"};
      previous = Number.isFinite(time) ? point : null;
    }
    finish();
    return events.sort((a, b) => b.severity - a.severity).slice(0, 10);
  }

  async function focusTrainingAnomalyPeak(diagnostic, peak) {
    if (state.training?.model_quality?.training_condition_diagnostic !== diagnostic || state.trainingDiagnosticInvalidated || !state.inspection) return setStatus("训练诊断已失效，请重新训练后定位。", "warning");
    const time = Date.parse(peak.timestamp);
    const start = Math.max(time - 30 * 60000, Date.parse(state.inspection.time_start));
    const end = Math.min(time + 30 * 60000, Date.parse(state.inspection.time_end));
    const tags = state.validationInvestigationModel?.training === state.training ? state.validationInvestigationModel.tags : [...state.selectedModelTags];
    if (!tags.length || !Number.isFinite(start) || !Number.isFinite(end) || start > end) return setStatus("当前数据没有可定位的模型 Tag 或时间范围。", "warning");
    const training = state.training;
    const inputTime = value => { const date = new Date(value); return new Date(value - date.getTimezoneOffset() * 60000).toISOString().slice(0,19); };
    const focus = {start: inputTime(start), end: inputTime(end)};
    el("trendStart").value = focus.start; el("trendEnd").value = focus.end; el("trendPreset").value = "custom";
    globalThis.showWorkflowStage("candidatePanel"); globalThis.showCandidateTool("trendPanel");
    try {
      await globalThis.showModelTagsTrend(tags, focus, () => state.training === training && !state.trainingDiagnosticInvalidated);
      if (state.training === training && !state.trainingDiagnosticInvalidated) setStatus(`已定位训练 ${peak.peakType} 峰值：${displayTime(peak.timestamp,19)}，查看前后各30分钟趋势（按数据边界裁剪）。`, "success");
    } catch (error) { setStatus(error.message || String(error), "error"); }
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
    const peaks = document.getElementById("modelTrainingAnomalyPeaks");
    if (peaks) {
      peaks.replaceChildren();
      const rows = trainingAnomalyPeaks(diagnostic?.timeline || [], Number(el("sampleInterval").value));
      if (!rows.length) peaks.textContent = "当前训练评分未发现 T²/SPE 95% 超限事件。";
      else {
        const table = document.createElement("table");
        table.innerHTML = "<thead><tr><th>时间</th><th>T²/95</th><th>SPE/95</th><th>工况组</th><th>训练窗口</th><th>峰值类型</th><th>操作</th></tr></thead>";
        const body = document.createElement("tbody");
        rows.forEach(peak => {
          const row = document.createElement("tr");
          [displayTime(peak.timestamp,19), ...[peak.t2_limit_ratio, peak.spe_limit_ratio].map(value => Number.isFinite(value) ? value.toFixed(3) : "—"), peak.cluster_id ? clusterUiLabel(peak.cluster_id) : "未归属", peak.window_id ?? "—", peak.peakType].forEach(value => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
          const cell = document.createElement("td"), button = document.createElement("button");
          button.type = "button"; button.className = "secondary"; button.textContent = "定位趋势";
          button.onclick = () => focusTrainingAnomalyPeak(diagnostic, peak); cell.append(button); row.append(cell); body.append(row);
        });
        table.append(body); peaks.append(table);
      }
    }
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
