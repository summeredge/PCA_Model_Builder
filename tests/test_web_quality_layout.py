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


def test_final_web_uses_the_apple_visual_tokens() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="appleDesignStyle"' in html
    assert "--accent:#0066cc;" in html
    assert "font-family:system-ui,-apple-system" in html
    assert "border-radius:9999px;" in html
    assert "--bg:#f5f5f7;" in html
    assert "transform:scale(.95);" in html
    assert ".tab, .inner-tab" in html
    assert "background:transparent;" in html
    assert "grid-template-columns:630px minmax(0,1fr);" in html
    assert ".controls, .controls .group { min-width:0; }" in html
    assert "max-width:100%;" in html
    assert "background:#fafafc;" in html
    assert ".results { gap:24px; }" in html
    assert ".empty, .variance, .exploration-timeline" in html
    assert "border-radius:18px;" in html
    assert "#engineeringPanel #tagRole," in html
    assert "height:42px;" in html
    assert "height:30px;" in html
    assert "grid-template-columns:repeat(6,minmax(0,1fr));" in html
    assert "font-size:28px;" in html
    assert "#tagOptions .tag-row" in html
    assert "height:30px;" in html
    assert "grid-template-columns:22px minmax(0,1fr) max-content;" in html


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

    assert 'layoutStyle.id = "modelProjectionGridStyle"' in source
    assert 'projectionGrid.className = "model-projection-grid"' in source
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);" in source
    assert "projectionGrid.append(scoreCard, section);" in source
    assert ".model-projection-grid #scoreChart svg" in source
    assert ".model-projection-grid #loadingChart svg" in source
    assert "@media (max-width: 1200px)" in source
    assert "grid-template-columns: 1fr;" in source
    assert 'insertAdjacentElement("afterend", section)' not in source


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
    assert "此处仅展示整体历史数据的时间轴与数值列检查结果" in html
    assert "已失效" not in html


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
    assert 'label==="确认作为训练窗口"&&(window.status!=="accepted"||Boolean(window.trainingWindowId))' in candidate_source
    assert "async function confirmCandidateWindow(candidate)" in mutation_source
    assert "enabled:true" in mutation_source
    assert 'action:"set_enabled"' in mutation_source
    assert "function updateQualityButtonAvailability()" in html
    assert "!state.trainingWindows.some(window=>window.enabled)" in html
    assert "renderTrainingWindows(); updateQualityButtonAvailability();" in mutation_source
    assert '["编辑",()=>editTrainingWindow(window)]' in mutation_source
    assert '["删除",()=>updateTrainingWindows({action:"remove",id:window.id},window.enabled)]' in mutation_source
    assert "if(affectsTraining) invalidateQuality" in mutation_source


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
    assert "renderTrainingWindows(); updateQualityButtonAvailability();" in update_source
    assert "if(affectsTraining) invalidateQuality" in update_source
    assert "suggested-window-001" not in update_source
    assert "尚无已确认训练窗口" in render_source


def test_cli_entry_restores_original_serve_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cli_entry.cli._serve
    monkeypatch.setattr(cli_entry.cli, "main", lambda argv=None: 0)

    assert cli_entry.main(["train"]) == 0
    assert cli_entry.cli._serve is original
