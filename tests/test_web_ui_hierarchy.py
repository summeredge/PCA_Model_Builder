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
        self.workflow_steps: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag not in self._VOID_ELEMENTS:
            self._stack.append((tag, attributes))
        element_id = attributes.get("id")
        if element_id:
            ancestors = self._stack[:-1] if tag not in self._VOID_ELEMENTS else self._stack
            self.ancestors_by_id[element_id] = tuple(
                item[1]["id"] for item in ancestors if item[1].get("id")
            )
        if "workflow-step" in attributes.get("class", "").split():
            self.workflow_steps.append(
                {"panel": attributes.get("data-panel", ""), "text": ""}
            )

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
        for title in ("数据准备", "正常状态候选", "模型训练", "模型验证", "模型发布")
        if any(title in step["text"] for step in parser.workflow_steps)
    ] == ["数据准备", "正常状态候选", "模型训练", "模型验证", "模型发布"]
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
        "trainingWindows": "modelPanel",
        "sampleInterval": "modelPanel",
        "maxLag": "modelPanel",
        "qualityButton": "modelPanel",
        "modelQualityStatus": "modelPanel",
        "modelQualityResults": "modelPanel",
        "trainButton": "modelPanel",
        "validateButton": "validationPanel",
        "validationDecisionStatus": "validationPanel",
        "validatedModelDownload": "releasePanel",
        "freezeDeployment": "releasePanel",
        "deploymentModelDownload": "releasePanel",
    }

    for element_id, panel_id in expected_parent.items():
        assert panel_id in parser.ancestors_by_id[element_id]


def test_workbench_script_only_updates_static_stage_state() -> None:
    source = web_model_results._WORKBENCH_UI_SCRIPT

    assert 'globalThis.showWorkflowStage = target =>' in source
    assert 'candidateDecisions.some(select => select.value === "accepted")' in source
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
    assert ">待决策</option>" in html
