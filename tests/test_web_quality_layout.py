from __future__ import annotations

from pathlib import Path

import pytest

from pca_model_builder import cli_entry, web_model_results, web_quality_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_quality_profiles_use_two_side_by_side_sections() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="qualityProfileGridStyle"' in html
    assert 'class="quality-profile-grid"' in html
    assert 'grid-template-columns:minmax(0,1fr) minmax(0,1fr)' in html
    assert 'qualityProfileTable("全数据统计", item.full)' in html
    assert 'qualityProfileTable("参考期统计", item.reference)' in html
    assert html.index('qualityProfileTable("全数据统计", item.full)') < html.index(
        'qualityProfileTable("参考期统计", item.reference)'
    )


def test_quality_grid_falls_back_to_one_column_on_narrow_screen() -> None:
    html = web_model_results.INDEX_HTML

    assert "@media (max-width:900px)" in html
    assert ".quality-profile-grid { grid-template-columns:1fr; }" in html


def test_quality_grid_transform_is_idempotent() -> None:
    once = web_quality_layout.apply_quality_grid_ui(
        "<html><head></head><body></body></html>"
    )
    twice = web_quality_layout.apply_quality_grid_ui(once)

    assert twice == once
    assert once.count('id="qualityProfileGridStyle"') == 1
    assert once.count('id="qualityProfileGridScript"') == 1


def test_quality_grid_transform_rejects_invalid_html() -> None:
    with pytest.raises(ValueError, match="head或body"):
        web_quality_layout.apply_quality_grid_ui("<html></html>")


def test_trend_page_no_longer_exposes_xy_scatter_matrix() -> None:
    html = web_model_results.INDEX_HTML

    assert '<h2>XY 散点矩阵</h2>' not in html
    assert 'class="dp-scatter-section"' not in html
    assert "renderScatterMatrix" not in html
    assert 'src="/assets/model-results.js"' in html


def test_trend_chart_drag_selection_uses_the_physical_time_domain() -> None:
    html = web_model_results.INDEX_HTML
    trend_source = html.split("function renderTrendChart(data)", 1)[1].split(
        "function renderStatCard", 1
    )[0]

    for marker in (
        'id="dpTrendSelectionHitbox"',
        'data-trend-selection',
        "const timeToX = (milliseconds)",
        "const xToTime = (position)",
        "const selectionThresholdPixels = 3;",
        "if (Math.abs(dragEnd-start) < selectionThresholdPixels) return restoreSelection();",
        "setTrendWindowFromSelection(xToTime(start), xToTime(dragEnd));",
        '$("dpTrendStart").value = datetimeLocalValue(earlier);',
        '$("dpTrendEnd").value = datetimeLocalValue(later);',
        '$("trendStart").value = $("dpTrendStart").value;',
        '$("trendEnd").value = $("dpTrendEnd").value;',
    ):
        assert marker in html

    assert "const earlier = Math.min(start, end);" in html
    assert "const later = Math.max(start, end);" in html
    assert "const pointTime = timestampMilliseconds(point.x);" in trend_source
    assert "timeToX(pointTime).toFixed(2)" in trend_source
    assert "timeToX(selection.start)" in trend_source
    assert "const exclusionMarkup = (state.excludedWindows || [])" in trend_source
    assert "data-trend-exclusion" in trend_source
    assert "${exclusionMarkup}${grid}" in trend_source
    assert "refreshTrendExcludedWindows = () => { if (lastTrend) renderTrendChart(lastTrend); }" in html
    assert "xToTime(dragStart)" in trend_source
    assert "item.points.forEach((point) =>" in trend_source
    assert "maxLength" not in trend_source
    assert "index / Math.max(1, maxLength - 1)" not in trend_source
    assert "point.physical_gap_start && current.length" in trend_source
    assert "(timeEnd-timeStart)" in html


def test_trend_reset_restores_uploaded_defaults_without_clearing_windows() -> None:
    html = web_model_results.INDEX_HTML
    reset_source = html.split('$("dpTrendReset").addEventListener("click", () => {', 1)[1].split(
        '$("dpDrawScatter").addEventListener', 1
    )[0]

    assert 'id="dpTrendMaxPoints" type="number" min="100" max="100000" value="30000"' in html
    assert html.index('id="dpTrendToExclusion"') < html.index('id="dpTrendReset"')
    assert "趋势复位" in html
    assert "defaults?.trend_default_start" in reset_source
    assert "defaults?.trend_default_end" in reset_source
    assert "localTime(defaults.trend_default_start)" in reset_source
    assert "localTime(defaults.trend_default_end)" in reset_source
    assert '$("dpTrendMaxPoints").value = "30000";' in reset_source
    assert "hasDraggedTrendSelection = false;" in reset_source
    assert "if (lastTrend) renderTrendChart(lastTrend);" in reset_source
    assert "syncToLegacy(chosen(trendIds));" in reset_source
    assert '$("dpDrawTrend").click();' in reset_source
    for preserved_state in (
        "state.excludedWindows=",
        "state.candidateWindows=",
        "state.trainingWindows=",
        '$("dpTrendAxisMode").value =',
        '$("dpTrendVar1").value =',
    ):
        assert preserved_state not in reset_source


def test_batch_cluster_and_tag_forms_use_consistent_alignment() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="webFormAlignmentStyle"' in html
    assert 'class="batch-config"' in html
    assert 'data-inner="batchPanel"' not in html
    assert "#engineeringPanel .batch-config .actions" in html
    assert (
        "grid-template-columns:max-content minmax(260px,1fr) "
        "max-content max-content max-content;"
    ) in html
    assert "#engineeringPanel .batch-config .actions > label.secondary" in html
    assert "grid-template-rows:auto 42px;" in html
    assert "#engineeringPanel .batch-config #tagConfigFile" in html
    assert "#clusterPanel #clusterButton" in html
    assert "align-self:end;" in html
    assert "#engineeringPanel .detail-fields .row > label" in html
    assert "align-content:start;" in html
    assert "#engineeringPanel #tagRole" in html
    assert "#engineeringPanel #tagComment" in html
    assert "@media (max-width:900px)" in html
    assert "grid-column:1 / -1;" in html
    assert html.rindex("#engineeringPanel .batch-config .actions") > html.index(
        ".actions { display:flex"
    )


def test_final_web_uses_compact_workbench_visual_tokens() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="appleDesignStyle"' in html
    assert "--accent:#0066cc;" in html
    assert "font-family:system-ui,-apple-system" in html
    assert "--bg:#f5f5f7;" in html
    for token in ("--panel:", "--line:", "--line-soft:", "--text:", "--muted:", "--danger:"):
        assert token in html
    assert "background:var(--panel);" in html
    assert "border-color:var(--accent);" in html
    assert "border-radius:9999px;" not in html
    assert "border-radius:18px;" not in html
    assert "backdrop-filter:" not in html
    assert "blur(" not in html
    assert "transform:scale(.95);" not in html
    assert ".tab, .inner-tab" in html
    assert "background:transparent;" in html
    assert "main { grid-template-columns:280px minmax(0,1fr); align-items:start; }" in html
    assert "grid-template-columns:630px minmax(0,1fr);" not in html
    assert ".controls, .controls .group { min-width:0; }" in html
    assert "max-width:100%;" in html
    assert ".results { gap:24px; }" in html
    assert ".empty, .variance, .exploration-timeline" in html
    assert "#engineeringPanel #tagRole," in html
    assert "height:42px;" in html
    assert "height:30px;" in html
    assert "grid-template-columns:repeat(6,minmax(0,1fr));" in html
    assert "font-size:28px;" in html
    assert "#tagOptions .tag-row" in html
    assert "height:30px;" in html
    assert "grid-template-columns:22px minmax(0,1fr) max-content;" in html

    for structure in (
        'class="controls workflow-sidebar"',
        'class="results"',
        'class="group training-configuration"',
        'id="status" class="status info operation-log"',
        'id="modelPanel"',
        'id="trainingWindowSummary"',
    ):
        assert structure in html
    for behavior in (
        "button:focus-visible",
        "button:disabled",
        ".status.error",
        ".status-label",
        "font-variant-numeric:tabular-nums;",
        ".table-wrap { overflow:auto;",
    ):
        assert behavior in html


def test_operation_log_is_shared_by_all_workflow_stages() -> None:
    html = web_model_results.INDEX_HTML

    assert html.count('id="status"') == 1
    assert (
        '<section class="results">\n'
        '      <div id="status" class="status info operation-log"'
    ) in html
    config_start = html.index('<div id="configPanel"')
    candidate_start = html.index('<div id="candidatePanel"')
    assert 'id="status"' not in html[config_start:candidate_start]


def test_final_web_preprocessing_notice_matches_schema5_invalid_row_policy() -> None:
    html = web_model_results.INDEX_HTML

    assert "数据缺失、重复、乱序或采样间隔不一致时训练会停止" not in html
    for text in (
        "时间戳重复、乱序或无法满足采样时间轴契约会阻断训练",
        "缺失、非数字、NaN、Inf 在重采样后删除整行并重新分段",
        "不插值、不补点、不自动修复异常值",
    ):
        assert text in html


def test_training_parameters_split_common_and_advanced_fields() -> None:
    html = web_model_results.INDEX_HTML
    config_start = html.index('<div class="group training-configuration">')
    advanced_start = html.index('<details class="advanced-parameters">', config_start)
    advanced_end = html.index('</details>', advanced_start)
    common_source = html[config_start:advanced_start]
    advanced_source = html[advanced_start:advanced_end]

    assert '<details class="advanced-parameters" open>' not in html
    assert html.count('<details class="advanced-parameters">') == 1
    for field_id in (
        "sampleInterval",
        "filterMethod",
        "firstOrderAlpha",
        "smoothingWindow",
        "maxLag",
        "varianceThreshold",
        "modelName",
    ):
        assert f'id="{field_id}"' in common_source
        assert f'id="{field_id}"' not in advanced_source
    for field_id in ("resamplingMethod", "gapThreshold", "lagStep", "components"):
        assert f'id="{field_id}"' in advanced_source
        assert f'id="{field_id}"' not in common_source
    for field_id in (
        "sampleInterval",
        "resamplingMethod",
        "filterMethod",
        "firstOrderAlpha",
        "smoothingWindow",
        "gapThreshold",
        "maxLag",
        "lagStep",
        "varianceThreshold",
        "components",
        "modelName",
    ):
        assert html.count(f'id="{field_id}"') == 1


def test_model_results_precede_training_configuration() -> None:
    html = web_model_results.INDEX_HTML
    model_start = html.index('<div id="modelPanel"')
    validation_start = html.index('<div id="validationPanel"', model_start)
    model_source = html[model_start:validation_start]

    assert model_source.index('id="modelContent"') < model_source.index(
        'class="group training-configuration"'
    )
    assert model_source.index('id="modelMetrics"') < model_source.index(
        'id="trainingWindowSummary"'
    )
    assert model_source.index('id="trainingWindowSummary"') < model_source.index(
        'id="varianceChart"'
    )
    assert model_source.index('id="varianceChart"') < model_source.index(
        'id="t2Chart"'
    )


def test_frozen_replay_is_mounted_in_the_release_stage() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")

    assert 'const releasePanel = document.getElementById("releasePanel");' in source
    assert "releasePanel.append(replayCard);" in source
    assert 'diagnosticCard.insertAdjacentElement("afterend", replayCard)' not in source


def test_loading_plot_uses_origin_lines_without_arrowheads() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")

    assert "原始Tag聚合载荷图" in source
    assert "每条连线从原点连接到" in source
    assert "addLine(svg, originX, originY, endX, endY" in source
    assert "PC1载荷" in source
    assert "PC2载荷" in source
    assert "x_explained_variance_ratio" in source
    assert "y_explained_variance_ratio" in source
    assert 'marker-end' not in source
    assert "loadingArrowHead" not in source
    assert "function addArrowMarker" not in source
    assert "每个点代表一个原始Tag" not in source


def test_model_score_and_loading_plots_use_side_by_side_grid() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")
    html = web_model_results.INDEX_HTML

    assert 'projectionGrid.className = "model-projection-grid"' in source
    assert "projectionGrid.append(scoreCard, section);" in source
    assert 'id="modelResultsStyle"' in html
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr);" in html
    assert ".model-projection-grid #scoreChart svg" in html
    assert ".model-projection-grid #loadingChart svg" in html
    assert 'document.createElement("style")' not in source
    assert "document.head.append(" not in source
    assert 'insertAdjacentElement("afterend", section)' not in source


def test_final_web_has_one_static_workbench_structure_and_asset_set() -> None:
    html = web_model_results.INDEX_HTML

    for element_id in (
        "workflowSteps",
        "status",
        "configPanel",
        "candidatePanel",
        "modelPanel",
        "validationPanel",
        "releasePanel",
        "trainingWindowSummary",
        "validatedModelDownload",
        "freezeDeployment",
    ):
        assert html.count(f'id="{element_id}"') == 1
    for style_id in (
        "webFormAlignmentStyle",
        "appleDesignStyle",
        "workbenchUiStyle",
        "modelResultsStyle",
    ):
        assert html.count(f'id="{style_id}"') == 1
    assert html.count('id="workbenchUiScript"') == 1
    assert html.count('src="/assets/model-results.js"') == 1


def test_workbench_assembly_uses_field_anchors_not_copied_parameter_markup() -> None:
    source = (PROJECT_ROOT / "src" / "pca_model_builder" / "web_model_results.py").read_text(
        encoding="utf-8"
    )

    assert "parameter_rows" not in source
    assert "模型标识.*?" not in source
    for helper in (
        "def _unique_anchor_index",
        "def _label_for_unique_field",
        "def _div_containing_unique_field",
    ):
        assert helper in source
    for field_id in (
        "sampleInterval",
        "resamplingMethod",
        "filterMethod",
        "firstOrderAlpha",
        "gapThreshold",
        "maxLag",
        "lagStep",
        "varianceThreshold",
        "components",
        "modelName",
        "frozenModelId",
    ):
        assert f'"{field_id}"' in source


def test_workbench_parameter_rows_allow_nonsemantic_div_attributes() -> None:
    base_html = web_model_results.quality_app.INDEX_HTML
    changed_html = base_html.replace(
        '<div class="row"><label>目标采样周期（分钟）',
        '<div data-test="stable" class="row compact"><label>目标采样周期（分钟）',
        1,
    )

    html = web_model_results._stabilize_workbench_html(changed_html)
    config_start = html.index('<div class="group training-configuration">')
    advanced_start = html.index('<details class="advanced-parameters">', config_start)
    advanced_end = html.index('</details>', advanced_start)
    common_source = html[config_start:advanced_start]
    advanced_source = html[advanced_start:advanced_end]

    for field_id in ("sampleInterval", "filterMethod", "firstOrderAlpha", "smoothingWindow", "maxLag", "varianceThreshold", "modelName"):
        assert f'id="{field_id}"' in common_source
    for field_id in ("resamplingMethod", "gapThreshold", "lagStep", "components"):
        assert f'id="{field_id}"' in advanced_source
    for field_id in (
        "sampleInterval", "resamplingMethod", "filterMethod", "firstOrderAlpha", "smoothingWindow", "gapThreshold",
        "maxLag", "lagStep", "varianceThreshold", "components", "modelName",
    ):
        assert html.count(f'id="{field_id}"') == 1


def test_supported_web_entrypoints_use_model_results_page() -> None:
    start_app = (PROJECT_ROOT / "start_app.bat").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "pca_model_builder.web_model_results" in start_app
    assert 'pca-model-builder = "pca_model_builder.cli_entry:main"' in pyproject
    assert (
        'pca-model-builder-web = "pca_model_builder.web_model_results:main"'
        in pyproject
    )


def test_final_web_entry_exposes_candidate_window_manager() -> None:
    html = web_model_results.INDEX_HTML

    for element_id in (
        'id="candidateStart"',
        'id="candidateEnd"',
        'id="candidateComment"',
        'id="addManualCandidate"',
        'id="candidateWindows"',
        'id="excludedWindows"',
        'id="trainingWindows"',
    ):
        assert element_id in html
    for label in (
        "窗口",
        "来源",
        "时间范围",
        "状态",
        "确认作为训练窗口",
        "参与训练",
        "查看趋势",
        "编辑",
        "删除",
    ):
        assert label in html
    assert "candidateWindows:[]" in html
    assert "excludedWindows:[]" in html
    assert "trainingWindows:[]" in html
    assert 'async function addCandidateWindow(source,start,end,sourceRef=null,comment="",status="pending")' in html
    assert 'status="pending"' in html
    assert 'addCandidateWindow("manual"' in html
    assert 'addCandidateWindow("cluster"' in html
    assert 'addCandidateWindow("trend"' in html
    assert 'addCandidateWindow("performance"' in html
    assert 'button.textContent="填入正常期"' not in html
    assert "normalStart" not in html
    assert "normalEnd" not in html


def test_final_web_keeps_candidate_decisions_manual_and_non_training() -> None:
    html = web_model_results.INDEX_HTML

    for element_id in (
        'id="saveExplorationCandidateDecisions"',
        'id="convertExplorationCandidates"',
    ):
        assert element_id in html
    assert "exploration-candidate-decision" in html
    assert "exploration-candidate-comment" in html
    assert "接受仅表示允许加入候选窗口，不会自动参与训练" in html
    assert "请在候选窗口列表确认作为训练窗口" in html


def test_model_quality_check_is_in_the_model_training_stage() -> None:
    html = web_model_results.INDEX_HTML
    model_start = html.index('<div id="modelPanel"')
    model_end = html.index('<div id="validationPanel"')
    model_source = html[model_start:model_end]

    assert "执行建模质量检查" in model_source
    assert 'id="modelQualityStatus"' in model_source
    assert 'id="modelQualityResults"' in model_source
    assert model_source.index('id="qualityButton"') < model_source.index('id="trainButton"')
    assert "上传后基础数据检查" in html
    assert "此处仅展示整体历史数据的时间轴与原始逐列检查结果" in html
    assert "column_profiles" in html
    assert "有效数值" in html
    assert "状态 / 建议" in html
    assert "已失效" not in html


def test_model_quality_controls_and_detail_table_do_not_stretch() -> None:
    html = web_model_results.INDEX_HTML

    assert "#modelPanel #modelQualityStatus," in html
    assert "#modelPanel #qualityButton { width:fit-content; justify-self:start; }" in html
    assert "#modelPanel #modelQualityStatus { max-width:100%; }" in html
    assert "#modelPanel #currentTagQuality { max-width:1200px; }" in html


def test_model_quality_status_tracks_check_and_configuration_changes() -> None:
    html = web_model_results.INDEX_HTML

    for label in (
        "未检查",
        "检查中",
        "通过",
        "有问题",
        "配置已变更需重新检查",
        "失败",
    ):
        assert label in html
    assert 'state.qualityStatus="checking"' in html
    assert 'el("trainButton").disabled=true' in html
    assert 'state.qualityStatus=data.can_train?"passed":"issues"' in html
    assert 'state.qualityStatus="failed"' in html
    assert 'state.qualityStatus=reason&&checked?"changed":"unchecked"' in html


def test_final_web_model_lifecycle_copy_matches_actual_model_semantics() -> None:
    html = web_model_results.INDEX_HTML

    for text in (
        "探索草稿模型，仅用于状态探索，不能执行独立验证或作为正常状态模型。",
        "正常状态候选模型，尚未完成独立验证和工程师确认。",
        "已验证模型，已完成独立验证和工程师确认；尚未执行工程冻结。",
        "只有正常状态候选模型可以执行独立验证。",
        "验证回放完成，待工程师确认",
        "已生成已验证模型副本",
        "原候选模型未被原地修改。",
    ):
        assert text in html
    for text in (
        "当前保存的是草稿模型",
        "训练草稿模型后可执行独立验证",
        "模型状态（草稿）",
        "已验证模型，可用于已完成工程师确认的正常状态监测。",
        "可用于监测",
        "可部署",
        "可上线",
    ):
        assert text not in html

    training_source = html.split("function renderTraining(data)", 1)[1].split(
        "function renderTrainingWindowSummary", 1
    )[0]
    validation_source = html.split("function renderValidation(data)", 1)[1].split(
        "function renderValidationWindows", 1
    )[0]
    decision_source = html.split('el("recordValidationDecision")', 1)[1].split(
        "function renderClustering", 1
    )[0]

    assert 'key==="exploratory/draft"' in html
    assert 'key==="normal_state/validated"' in html
    assert "const lifecycle=modelLifecycle(data);" in training_source
    assert 'el("modelLifecycleNotice").textContent=lifecycle.notice;' in training_source
    assert "const lifecycle=modelLifecycle(data);" in validation_source
    assert 'data.model_status==="validated"' in validation_source
    assert 'model_status:data.model_status' in decision_source
    assert "renderValidation(state.validation);" in decision_source


def test_validation_engineer_confirmation_precedes_validation_metrics() -> None:
    html = web_model_results.INDEX_HTML

    summary = html.index("<h3>验证状态摘要</h3>")
    confirmation = html.index('id="recordValidationDecision"')
    metrics = html.index("<h3>验证指标</h3>")

    assert summary < confirmation < metrics


def test_candidate_actions_do_not_replace_the_training_window() -> None:
    html = web_model_results.INDEX_HTML
    cluster_source = html.split("function renderClustering", 1)[1].split(
        "function renderTrainingWindowSummary", 1
    )[0]
    performance_source = html.split("function renderPerformance(data)", 1)[1].split(
        "function renderClustering", 1
    )[0]

    assert 'addCandidateWindow("cluster"' in cluster_source
    assert 'addCandidateWindow("performance"' in performance_source
    assert 'normalStart' not in cluster_source
    assert 'normalEnd' not in cluster_source
    assert 'normalStart' not in performance_source
    assert 'normalEnd' not in performance_source


def test_inspection_creates_only_a_pending_suggested_candidate() -> None:
    html = web_model_results.INDEX_HTML
    inspect_source = html.split('el("inspectButton").addEventListener("click", async () => {', 1)[1].split(
        'el("selectAllTags").addEventListener', 1
    )[0]

    assert 'state.candidateWindows=[{id:"suggested-window-001"' in inspect_source
    assert 'source:"suggested",source_ref:"inspect-default",status:"pending"' in inspect_source
    assert '系统建议的初始正常候选时段' in inspect_source
    assert 'state.trainingWindows=[]' in inspect_source
    assert 'el("qualityButton").disabled=true' in inspect_source


def test_upload_success_clears_candidate_and_previous_file_state() -> None:
    html = web_model_results.INDEX_HTML
    upload_source = html.split('el("uploadButton").addEventListener("click", async () => {', 1)[1].split(
        'el("inspectButton").addEventListener', 1
    )[0]

    for statement in (
        "state.candidateWindows=[]",
        "state.trainingWindows=[]",
        "state.trainingWindowSummary=[]",
        "renderCandidateWindows()",
        "renderTrainingWindows()",
        "state.quality=null",
        "state.training=null",
        "state.runId=null",
        "state.exploratoryRunId=null",
        "state.excludedTags=[]",
    ):
        assert statement in upload_source
    assert 'setStatus("正在读取文件…","info")' in upload_source
    assert "requestAnimationFrame" in upload_source
    assert '"/api/inspect"' not in upload_source


def test_data_inspection_has_visible_progress_and_timeout() -> None:
    html = web_model_results.INDEX_HTML
    inspect_source = html.split(
        'el("inspectButton").addEventListener("click", async () => {', 1
    )[1].split('el("tagSearch")', 1)[0]

    assert "new AbortController()" in inspect_source
    assert "controller.abort()" in inspect_source
    assert "超过 30 秒未完成" in inspect_source
    assert "读取时间列与候选 Tag" in inspect_source
    assert "signal:controller.signal" in inspect_source
    assert "ensureInspectionPageReady();" in inspect_source
    assert 'console.error("数据检查失败:",error)' in inspect_source
    assert "数据检查失败:" in inspect_source
    assert "setBusy(button,false" in inspect_source
    assert "function ensureInspectionPageReady()" in html
    assert "await response.json()" in html


def test_candidate_confirmation_is_separate_from_training_windows() -> None:
    html = web_model_results.INDEX_HTML
    view_source = html.split("function showCandidateTrend(window)", 1)[1].split(
        "function renderCandidateWindows", 1
    )[0]
    candidate_source = html.split("function renderCandidateWindows", 1)[1].split(
        "function renderTrainingWindows", 1
    )[0]
    mutation_source = html.split("function renderTrainingWindows", 1)[1].split(
        "async function api", 1
    )[0]

    assert "window.enabled" not in view_source
    assert "set_enabled" not in view_source
    assert "state.trainingWindows" not in view_source
    assert '["pending","accepted","rejected"]' in candidate_source
    assert "candidateTrainingWindows(window).length>0" in candidate_source
    assert 'label==="确认作为训练窗口"&&(window.status!=="accepted"||generated)' in candidate_source
    assert "async function confirmCandidateWindow(candidate)" in mutation_source
    assert 'action:"confirm_candidate",candidate,excluded_windows:state.excludedWindows' in mutation_source
    assert 'action:"set_enabled"' in mutation_source
    assert "function updateQualityButtonAvailability()" in html
    assert "!state.trainingWindows.some(window=>window.enabled)" in html
    assert "renderTrainingWindows(); renderCandidateWindows(); updateQualityButtonAvailability();" in mutation_source
    assert '["编辑",()=>editTrainingWindow(window)]' in mutation_source
    assert '["删除",()=>updateTrainingWindows({action:"remove",id:window.id},window.enabled)]' in mutation_source
    assert "if(affectsTraining&&previous!==JSON.stringify(state.trainingWindows)) invalidateQuality" in mutation_source


def test_last_training_window_removal_keeps_the_training_table_empty() -> None:
    html = web_model_results.INDEX_HTML
    update_source = html.split("async function updateTrainingWindows", 1)[1].split(
        "function addCandidateWindow", 1
    )[0]
    render_source = html.split("function renderTrainingWindows", 1)[1].split(
        "async function updateTrainingWindows", 1
    )[0]

    assert 'action:"remove"' in html
    assert "state.trainingWindows=data.training_windows" in update_source
    assert "state.trainingWindowSummary=data.summary" in update_source
    assert "renderTrainingWindows(); renderCandidateWindows(); updateQualityButtonAvailability();" in update_source
    assert "if(affectsTraining&&previous!==JSON.stringify(state.trainingWindows)) invalidateQuality" in update_source
    assert "suggested-window-001" not in update_source
    assert "尚无已确认训练窗口" in render_source


def test_cli_entry_restores_original_serve_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cli_entry.cli._serve
    monkeypatch.setattr(cli_entry.cli, "main", lambda argv=None: 0)

    assert cli_entry.main(["train"]) == 0
    assert cli_entry.cli._serve is original


def test_workbench_tables_map_all_runtime_statuses_and_align_numeric_cells() -> None:
    html = web_model_results.INDEX_HTML

    assert ".table-wrap td.numeric" in html
    assert "font-variant-numeric:tabular-nums;" in html
    for status, label in (
        ("pending", "待决策"),
        ("accepted", "已接受"),
        ("rejected", "已拒绝"),
        ("used", "已使用"),
        ("dropped", "已丢弃"),
    ):
        assert f'{status}:"{label}"' in html
    assert 'displayUiValue(summary.quality_status||summary.status||"待检查")' in html


def test_training_window_summary_uses_numeric_table_wrapper_and_runtime_labels() -> None:
    html = web_model_results.INDEX_HTML
    summary_source = html.split(
        "function renderTrainingWindowSummary(windows)", 1
    )[1].split("function renderValidation(data)", 1)[0]

    assert 'id="trainingWindowSummary" class="table-wrap"' in html
    for field in (
        "重采样减少",
        "部分桶",
        "滤波预热",
        "滤波上下文",
        "状态过滤",
        "Lag预热",
        "Lag上下文",
        "输入无效",
        "有效动态样本",
    ):
        assert field in summary_source
    assert "displayUiValue(window.status)" in summary_source
    assert "displayUiValue(segment.status)" in summary_source
    assert 'displayUiValue(window.dropped_reason??"—")' in summary_source
    assert 'displayUiValue(segment.dropped_reason??"—")' in summary_source
    for status, label in (
        ("disabled", "已禁用"),
        ("no_complete_resampling_bins", "无完整重采样时间桶"),
    ):
        assert f'{status}:"{label}"' in html


def test_model_results_use_visible_error_and_loading_states() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")

    assert 'setBusy(button, true, "比较中…")' in source
    assert 'setBusy(button, true, "回放中…")' in source
    assert "候选模型加载失败：${error.message}" in source
    assert "模型比较失败：${error.message}" in source
    assert "冻结模型回放失败：${error.message}" in source
    assert 'target.className = type === "empty" ? "empty" : type === "error" ? "status error" : `status ${type}`' in source
    assert 'comparability.className = data.comparability.comparable ? "help" : "status error"' in source
