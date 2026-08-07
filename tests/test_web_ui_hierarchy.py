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
        'button:disabled, input:disabled, select:disabled, textarea:disabled',
        '@media (max-width:760px)',
    ):
        assert marker in html


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
