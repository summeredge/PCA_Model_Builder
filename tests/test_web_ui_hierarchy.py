from pca_model_builder import web_model_results


def test_final_web_adds_ui_only_workflow_hierarchy_and_accessibility() -> None:
    html = web_model_results.INDEX_HTML

    for marker in (
        'id="workbenchUiStyle"',
        'id="workbenchUiScript"',
        '高级预处理与 DPCA 参数',
        '运行日志',
        'aria-selected',
        'status-label',
        'workflow-sidebar',
        'workflow-step-status',
        'candidate-tool-tabs',
        'button:disabled, input:disabled, select:disabled, textarea:disabled',
        '@media (max-width:760px)',
    ):
        assert marker in html

    for stage in (
        "数据准备",
        "正常状态候选",
        "模型训练",
        "模型验证",
        "模型发布",
    ):
        assert stage in html


def test_final_web_moves_controls_into_their_workflow_stages() -> None:
    source = web_model_results._WORKBENCH_UI_SCRIPT

    assert 'controls.className = "workflow-sidebar"' in source
    assert 'dataGrid.append(uploadGroup, tagGroup)' in source
    assert 'modelPanel.prepend(parameterGroup)' in source
    assert 'candidatePanel.append(candidateManager)' in source
    assert 'releaseContent.append(validatedDownload, freezeBox)' in source
    assert 'globalThis.showWorkflowStage = target =>' in source
    assert 'position:sticky' in web_model_results._WORKBENCH_UI_STYLE


def test_workflow_status_is_derived_from_existing_ui_state() -> None:
    source = web_model_results._WORKBENCH_UI_SCRIPT

    assert 'candidateDecisions.some(select => select.value === "accepted")' in source
    assert '!document.getElementById("modelContent").hidden' in source
    assert '!document.getElementById("validationContent").hidden' in source
    assert '!document.getElementById("deploymentModelDownload").hidden' in source
    assert '"已完成"' in source
    assert '"当前"' in source
    assert '"待开始"' in source


def test_trend_and_manual_selection_use_the_unified_candidate_action() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="addManualCandidate" class="secondary" type="button">加入候选窗口' in html
    assert 'id="dpTrendToReference" type="button" class="secondary">加入候选窗口' in html
    assert "将当前窗口设为参考状态候选期" not in html
    assert 'addCandidateWindow("trend"' in html
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
    assert '>待决策</option>' in html
