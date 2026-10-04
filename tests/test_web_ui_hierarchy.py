from html.parser import HTMLParser

from pca_model_builder import web_model_results


class _WorkbenchParser(HTMLParser):
    _VOID_ELEMENTS = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[tuple[str, dict[str, str]]] = []
        self.ancestors_by_id: dict[str, tuple[str, ...]] = {}
        self.ids: list[str] = []
        self.workflow_steps: list[dict[str, str]] = []
        self.candidate_tools: list[tuple[str, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag not in self._VOID_ELEMENTS:
            self._stack.append((tag, attributes))
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
            ancestors = self._stack[:-1] if tag not in self._VOID_ELEMENTS else self._stack
            self.ancestors_by_id[element_id] = tuple(
                item[1]["id"] for item in ancestors if item[1].get("id")
            )
        if "workflow-step" in attributes.get("class", "").split():
            self.workflow_steps.append(
                {"panel": attributes.get("data-panel", ""), "text": ""}
            )
        if "candidate-tool-tab" in attributes.get("class", "").split():
            self.candidate_tools.append((
                attributes.get("data-panel", ""),
                tuple(item[1]["id"] for item in self._stack if item[1].get("id")),
            ))

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1][0] == tag:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self.workflow_steps and any(
            "workflow-step" in item[1].get("class", "").split()
            for item in self._stack
        ):
            self.workflow_steps[-1]["text"] += data.strip()


def _workbench() -> _WorkbenchParser:
    parser = _WorkbenchParser()
    parser.feed(web_model_results.INDEX_HTML)
    return parser


def test_final_web_has_a_static_five_stage_workbench() -> None:
    parser = _workbench()

    assert [step["panel"] for step in parser.workflow_steps] == [
        "configPanel",
        "candidatePanel",
        "modelPanel",
        "validationPanel",
        "releasePanel",
    ]
    assert [
        title
        for title in ("数据与Tag", "正常状态候选", "模型训练", "模型验证", "冻结与部署")
        if any(title in step["text"] for step in parser.workflow_steps)
    ] == ["数据与Tag", "正常状态候选", "模型训练", "模型验证", "冻结与部署"]
    for panel_id in (
        "configPanel",
        "candidatePanel",
        "modelPanel",
        "validationPanel",
        "releasePanel",
    ):
        assert panel_id in parser.ancestors_by_id


def test_static_panels_own_their_existing_controls() -> None:
    parser = _workbench()

    expected_parent = {
        "fileInput": "configPanel",
        "tagOptions": "configPanel",
        "candidateWindows": "candidatePanel",
        "excludedWindows": "candidatePanel",
        "trainingWindows": "candidatePanel",
        "sampleInterval": "candidatePanel",
        "analysisStart": "candidatePanel",
        "analysisEnd": "candidatePanel",
        "performanceConditions": "performancePanel",
        "addPerformanceCondition": "performancePanel",
        "performanceButton": "performancePanel",
        "clusterCount": "clusterPanel",
        "clusterButton": "clusterPanel",
        "resamplingMethod": "candidatePanel",
        "filterMethod": "candidatePanel",
        "firstOrderAlpha": "candidatePanel",
        "smoothingWindow": "candidatePanel",
        "gapThreshold": "candidatePanel",
        "maxLag": "candidatePanel",
        "lagStep": "candidatePanel",
        "stateFilterConditions": "candidatePanel",
        "addStateFilterCondition": "candidatePanel",
        "preprocessingPreviewWindow": "modelPanel",
        "preprocessingPreviewButton": "modelPanel",
        "preprocessingPreview": "modelPanel",
        "qualityButton": "modelPanel",
        "modelQualityStatus": "modelPanel",
        "currentTagQuality": "modelPanel",
        "modelQualityResults": "modelPanel",
        "modelName": "modelPanel",
        "varianceThreshold": "modelPanel",
        "components": "modelPanel",
        "trainButton": "modelPanel",
        "loadingChart": "modelPanel",
        "componentLoadings": "modelPanel",
        "modelStructureComparison": "modelPanel",
        "validateButton": "validationPanel",
        "validationDecisionStatus": "validationPanel",
        "validatedModelDownload": "releasePanel",
        "freezeDeployment": "releasePanel",
        "deploymentModelDownload": "releasePanel",
        "frozenReplay": "releasePanel",
    }

    for element_id, panel_id in expected_parent.items():
        assert panel_id in parser.ancestors_by_id[element_id]


def test_candidate_tools_share_one_level_and_analysis_range() -> None:
    parser = _workbench()
    html = web_model_results.INDEX_HTML

    assert [panel for panel, _ in parser.candidate_tools] == [
        "trendPanel", "stateExplorationPanel", "clusterPanel", "performancePanel"
    ]
    assert all(parent == ("candidatePanel",) for _, parent in parser.candidate_tools)
    tabs_start = html.index('<div class="candidate-tool-tabs"')
    shared_start = html.index('<div class="group shared-preprocessing">')
    manager_start = html.index('<div class="group candidate-manager"', shared_start)
    tool_positions = [
        html.index(f'id="{panel}"', tabs_start)
        for panel in ("trendPanel", "stateExplorationPanel", "clusterPanel", "performancePanel")
    ]
    assert tabs_start < shared_start < tool_positions[0]
    assert tool_positions[0] < tool_positions[1] < tool_positions[2] < tool_positions[3]
    assert tool_positions[-1] < manager_start
    assert len(parser.ids) == len(set(parser.ids))
    assert 'id="explorationStart"' not in html
    assert 'id="explorationEnd"' not in html
    assert 'data-panel="statePanels"' not in html
    assert 'exploration_start:(el("explorationStart")||el("analysisStart")).value' in html
    assert 'exploration_end:(el("explorationEnd")||el("analysisEnd")).value' in html
    assert 'analysis_start:el("analysisStart").value' in html
    assert 'analysis_end:el("analysisEnd").value' in html
    assert '$("analysisStart").value = $("dpTrendStart").value' in html
    assert '$("analysisEnd").value = $("dpTrendEnd").value' in html


def test_training_validation_and_release_follow_decision_order() -> None:
    html = web_model_results.INDEX_HTML
    model = html[html.index('<div id="modelPanel"'):html.index('<div id="validationPanel"')]
    validation = html[html.index('<div id="validationPanel"'):html.index('<div id="releasePanel"')]
    release = html[html.index('<div id="releasePanel"'):]

    assert [model.index(label) for label in (
        "① 训练准备", "② PCA / DPCA 模型配置", "③ 建模质量检查",
        "最终训练集审查", "④ 正式训练", 'id="modelContent"', 'id="varianceChart"',
        'id="modelQualitySummary"', 'id="componentLoadings"',
        'id="modelTrainingDataQuality"', 'id="t2Chart"', 'id="speChart"',
        'id="scoreChart"', 'id="loadingChart"', 'id="modelStructureComparison"',
        'id="modelDownload"',
    )] == sorted(model.index(label) for label in (
        "① 训练准备", "② PCA / DPCA 模型配置", "③ 建模质量检查",
        "最终训练集审查", "④ 正式训练", 'id="modelContent"', 'id="varianceChart"',
        'id="modelQualitySummary"', 'id="componentLoadings"',
        'id="modelTrainingDataQuality"', 'id="t2Chart"', 'id="speChart"',
        'id="scoreChart"', 'id="loadingChart"', 'id="modelStructureComparison"',
        'id="modelDownload"',
    ))
    quality_section = model[
        model.index('<h3>③ 建模质量检查</h3>') : model.index('<h3>④ 正式训练</h3>')
    ]
    for heading in ("当前 Tag 建模质量详情", "建模质量问题"):
        assert f"<h4>{heading}</h4>" in quality_section
        assert f"<h3>{heading}</h3>" not in quality_section
    assert '<h4 id="finalTrainingReviewTitle">最终训练集审查</h4>' in quality_section
    assert '<h3 id="finalTrainingReviewTitle">最终训练集审查</h3>' not in quality_section
    assert validation.index('id="validateButton"') < validation.index('id="validationMetrics"') < validation.index('id="validationMetricDetails"') < validation.index('id="recordValidationDecision"')
    assert release.index('id="validatedModelDownload"') < release.index('id="freezeDeployment"') < release.index('id="frozenModelDownload"') < release.index('id="frozenReplay"')
    assert 'id="modelPreprocessingSummary"' in model
    assert model.index('id="modelTrainingDataQuality"') < model.index('id="modelTrainingConditionDiagnostic"') < model.index('id="t2Chart"')
    for element_id in ("modelTrainingConditionDiagnostic", "modelTrainingConditionMessage", "modelTrainingConditionGroups", "modelTrainingConditionTrend", "modelTrainingConditionSwitch", "modelTrainingConditionHints"):
        assert html.count(f'id="{element_id}"') == 1
    assert '<section id="frozenReplay" class="chart-card" hidden>' in release
    assert 'replayButton.disabled = !canReplay || replayButton.textContent === "回放中…"' in html


def test_model_results_only_fill_static_workbench_containers() -> None:
    source = web_model_results._ASSET_PATH.read_text(encoding="utf-8")
    html = web_model_results.INDEX_HTML

    for element_id in ("loadingChart", "componentLoadings", "modelStructureComparison", "frozenReplay"):
        assert html.count(f'id="{element_id}"') == 1
    assert "insertAdjacentElement(" not in source
    assert "releasePanel.append(" not in source
    assert 'document.getElementById("loadingChart")' in source


def test_each_shared_preprocessing_control_keeps_one_unique_id() -> None:
    parser = _workbench()

    for element_id in (
        "sampleInterval",
        "resamplingMethod",
        "filterMethod",
        "firstOrderAlpha",
        "smoothingWindow",
        "gapThreshold",
        "maxLag",
        "lagStep",
        "stateFilterConditions",
        "addStateFilterCondition",
        "preprocessingPreviewWindow",
        "preprocessingPreviewButton",
        "preprocessingPreview",
        "qualityButton",
        "modelQualityStatus",
        "modelQualityResults",
        "modelName",
        "varianceThreshold",
        "components",
    ):
        assert web_model_results.INDEX_HTML.count(f'id="{element_id}"') == 1, element_id


def test_workbench_script_only_updates_static_stage_state() -> None:
    source = web_model_results._WORKBENCH_UI_SCRIPT

    assert 'globalThis.showWorkflowStage = target =>' in source
    assert 'Boolean(trainingTable.querySelector("tbody tr"))' in source
    assert 'candidateDecisions.some(select => select.value === "accepted")' not in source
    assert '!document.getElementById("modelContent").hidden' in source
    assert '!document.getElementById("validationContent").hidden' in source
    assert '!document.getElementById("deploymentModelDownload").hidden' in source
    assert 'position:sticky' in web_model_results._WORKBENCH_UI_STYLE
    for forbidden in (
        'controls.innerHTML',
        'legacyTabs.remove()',
        'dataGrid.append(',
        'modelPanel.prepend(',
        'candidatePanel.append(',
        'results.insertBefore(',
        'results.append(releasePanel)',
        'releaseContent.append(',
        'document.createElement("div")',
        "textContent.includes",
    ):
        assert forbidden not in source


def test_trend_and_manual_selection_use_the_unified_candidate_action() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="addManualCandidate" class="secondary" type="button">加入候选窗口' in html
    assert 'id="dpTrendToReference" type="button" class="secondary">加入候选窗口' in html
    assert "将当前窗口设为参考状态候选期" not in html
    assert 'addCandidateWindow("trend"' in html
    assert 'id="dpTrendToExclusion" type="button" class="secondary">加入排除窗口' in html
    assert 'addExcludedWindow("trend"' in html
    assert 'id="dpTrendReset" type="button" class="secondary">趋势复位' in html
    assert 'globalThis.showWorkflowStage?.("candidatePanel")' in html


def test_final_web_keeps_algorithm_and_api_paths_out_of_ui_layer() -> None:
    source = web_model_results._WORKBENCH_UI_SCRIPT

    assert "fetch(" not in source
    assert "/api/" not in source
    assert "state." not in source


def test_web_translates_display_labels_without_changing_option_values() -> None:
    html = web_model_results.INDEX_HTML

    assert 'value="higher_is_better">越高越好' in html
    assert 'value="lower_is_better">越低越好' in html
    assert 'value="target_range">目标范围内' in html
    assert 'value="continuous_input">连续输入' in html
    assert 'pending:"待决策"' in html
    assert ">待决策</option>" not in html


def test_tag_instructions_name_regions_instead_of_left_or_right() -> None:
    html = web_model_results.INDEX_HTML

    assert "Tag 详情区查看配置与质量" in html
    assert "数据与 Tag”阶段当前勾选" in html
    assert "请选择左侧Tag" not in html
    assert "左侧当前勾选" not in html
    assert "右侧查看配置" not in html
