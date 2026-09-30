from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

from pca_model_builder import cli_entry, web_model_results, web_quality_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_web_javascript(source: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Web state-machine regression tests")
    result = subprocess.run(
        [node, "-e", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


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
        'selectdirection: "h"',
        'dragmode: "select"',
        'plot.on("plotly_selected"',
        "setTrendWindowFromSelection(range[0], range[1]);",
        '$("dpTrendStart").value = datetimeLocalValue(earlier);',
        '$("dpTrendEnd").value = datetimeLocalValue(later);',
        '$("trendStart").value = $("dpTrendStart").value;',
        '$("trendEnd").value = $("dpTrendEnd").value;',
    ):
        assert marker in html

    # Plotly date axes return date strings, so ranges are normalised to epoch
    # milliseconds before ordering.
    assert "const first = timestampMilliseconds(start);" in html
    assert "const second = timestampMilliseconds(end);" in html
    assert "const earlier = Math.min(first, second);" in html
    assert "const later = Math.max(first, second);" in html
    assert "const pointTime = timestampMilliseconds(point.x);" in trend_source
    assert "(state.excludedWindows || [])" in trend_source
    assert "trendExclusionShapes" in trend_source
    assert "globalThis.refreshTrendExcludedWindows = () =>" in html
    assert "globalThis.Plotly.relayout" in html
    assert "item.points.forEach((point) =>" in trend_source
    assert "maxLength" not in trend_source
    assert "index / Math.max(1, maxLength - 1)" not in trend_source


def test_trend_chart_uses_local_plotly_scattergl_without_svg_curves() -> None:
    html = web_model_results.INDEX_HTML
    trend_source = html.split("function renderTrendChart(data)", 1)[1].split(
        "function renderStatCard", 1
    )[0]

    assert 'src="/assets/plotly.min.js"' in html
    assert "cdn.plot.ly" not in html
    assert 'type: "scattergl"' in trend_source
    assert 'mode: "lines"' in trend_source
    assert "connectgaps: false" in trend_source
    assert "globalThis.Plotly.react(container, traces, layout, config)" in trend_source
    # The trend curve must no longer be drawn with hand-written SVG/polyline.
    assert "<polyline" not in trend_source
    assert "<svg" not in trend_source
    assert "timeToX" not in trend_source
    assert "xToTime" not in trend_source


def test_trend_physical_gap_becomes_an_explicit_plotly_break_point() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]

    assert "if (point.physical_gap_start && x.length) { x.push(null); y.push(null); }" in trend_source
    assert "if (value === null || pointTime === null) { x.push(pointTime); y.push(null); return; }" in trend_source


def test_trend_zoom_and_pan_do_not_change_the_current_window_inputs() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]

    relayout = trend_source.split('plot.on("plotly_relayout"', 1)[1].split("});", 1)[0]

    assert 'dpTrendStart").value' not in relayout
    assert 'dpTrendEnd").value' not in relayout
    assert "setTrendWindowFromSelection" not in relayout


def _trend_window_harness() -> str:
    """Extract the real time helpers so their behaviour can be executed."""
    html = web_model_results.INDEX_HTML
    start = html.index("function timestampMilliseconds")
    end = html.index("function renderTrendChart")
    return html[start:end]


def _timestamp_milliseconds_source() -> str:
    """The shipped timestamp parser, extracted verbatim."""
    html = web_model_results.INDEX_HTML
    return "function timestampMilliseconds" + html.split("function timestampMilliseconds", 1)[1].split(
        "function datetimeLocalValue", 1
    )[0]


def test_timestamp_parser_normalises_space_separated_plotly_dates() -> None:
    source = _timestamp_milliseconds_source()

    assert 'text.includes(" ") && !text.includes("T")' in source
    assert 'text.replace(" ", "T")' in source
    assert 'value.includes("T") || value.includes("-")' not in source

    # The space-between-date-and-time form must be converted explicitly rather
    # than relying on lenient Date.parse handling.
    _run_web_javascript(
        f"""
        // Observe what the parser actually hands to Date.parse: engines accept a
        // space separator leniently, so only the explicit "T" rewrite proves the
        // conversion happens here and is not left to runtime leniency.
        const realParse = Date.parse;
        const seen = [];
        Date.parse = value => {{ seen.push(value); return realParse(value); }};
        {source}

        const iso = realParse("2026-01-01T10:00:00");
        const cases = [
          ["2026-01-01 10:00:00.000", iso],
          ["2026-01-01T10:00:00", iso],
          ["2026-01-01 01:15:44.5946", realParse("2026-01-01T01:15:44.5946")],
          [1767225600000, 1767225600000],
        ];
        for (const [input, expected] of cases) {{
          const actual = timestampMilliseconds(input);
          if (actual !== expected) {{
            throw new Error(`${{JSON.stringify(input)}} -> ${{actual}}, expected ${{expected}}`);
          }}
        }}
        for (const bad of ["", "   ", "not-a-date", null, undefined, NaN, {{}}]) {{
          if (timestampMilliseconds(bad) !== null) {{
            throw new Error(`${{JSON.stringify(bad)}} should be null`);
          }}
        }}
        if (seen[0] !== "2026-01-01T10:00:00.000") {{
          throw new Error("space separator was not rewritten to T: " + seen[0]);
        }}
        if (seen[1] !== "2026-01-01T10:00:00") {{
          throw new Error("ISO string must be passed through unchanged: " + seen[1]);
        }}
        """
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        # Plotly date axes return date strings, not epoch milliseconds.
        ("2026-01-01 10:00:00.000", "2026-01-01 12:00:00.000", ("2026-01-01T10:00", "2026-01-01T12:00")),
        ("2026-01-01T10:00:00", "2026-01-01T12:00:00", ("2026-01-01T10:00", "2026-01-01T12:00")),
        # Reverse drag must normalise to chronological order.
        ("2026-01-01T12:00:00", "2026-01-01T10:00:00", ("2026-01-01T10:00", "2026-01-01T12:00")),
        ("2026-01-01 12:00:00.000", "2026-01-01 10:00:00.000", ("2026-01-01T10:00", "2026-01-01T12:00")),
    ],
)
def test_trend_selection_accepts_date_strings_and_reverse_drag(
    start: str, end: str, expected: tuple[str, str]
) -> None:
    _run_web_javascript(
        f"""
        const elements = new Map([
          ["dpTrendStart", {{value: ""}}],
          ["dpTrendEnd", {{value: ""}}],
          ["trendStart", {{value: ""}}],
          ["trendEnd", {{value: ""}}],
        ]);
        const $ = id => elements.get(id);
        {_trend_window_harness()}

        setTrendWindowFromSelection({json.dumps(start)}, {json.dumps(end)});
        const results = [
          [$("dpTrendStart").value, $("dpTrendEnd").value],
          [$("trendStart").value, $("trendEnd").value],
        ];
        if (JSON.stringify(results) !== JSON.stringify([{json.dumps(list(expected))}, {json.dumps(list(expected))}])) {{
          throw new Error("unexpected window: " + JSON.stringify(results));
        }}
        """
    )


def test_trend_selection_rejects_invalid_ranges_without_writing_inputs() -> None:
    _run_web_javascript(
        f"""
        const elements = new Map([
          ["dpTrendStart", {{value: "2026-01-01T00:00"}}],
          ["dpTrendEnd", {{value: "2026-01-01T01:00"}}],
          ["trendStart", {{value: "2026-01-01T00:00"}}],
          ["trendEnd", {{value: "2026-01-01T01:00"}}],
        ]);
        const $ = id => elements.get(id);
        {_trend_window_harness()}

        setTrendWindowFromSelection("not-a-date", null);
        if ($("dpTrendStart").value !== "2026-01-01T00:00") throw new Error("invalid start overwrote input");
        if ($("dpTrendEnd").value !== "2026-01-01T01:00") throw new Error("invalid end overwrote input");
        """
    )


def test_trend_selection_shape_is_drawn_for_date_string_ranges() -> None:
    html = web_model_results.INDEX_HTML
    body = html.split("function applySelectionShape(start, end)", 1)[1].split(
        "function selectionShape", 1
    )[0]
    # Reuse the shipped parser so a bug cannot hide in both places at once.
    parser = _timestamp_milliseconds_source()
    helpers = """
      const shapes = [];
      const relayouts = [];
      globalThis.Plotly = {relayout: (target, update) => relayouts.push(update)};
      function trendExclusionShapes() { return []; }
      function selectionShape(x0, x1) { return {x0, x1}; }
    """
    _run_web_javascript(
        f"""
        const plot = {{layout: {{}}}};
        const $ = id => id === "dpTrendChart" ? plot : null;
        {parser}
        {helpers}
        function applySelectionShape(start, end) {{{body}}}

        applySelectionShape("2026-01-01 10:00:00.000", "2026-01-01 12:00:00.000");
        const drawn = relayouts.at(-1).shapes;
        if (drawn.length !== 1) throw new Error("selection rectangle missing for date strings");
        if (drawn[0].x0 > drawn[0].x1) throw new Error("selection rectangle not normalised");
        if (Date.parse("2026-01-01T10:00:00") !== drawn[0].x0) throw new Error("wrong selection start");
        """
    )


def test_independent_y_axes_all_overlay_the_main_axis() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]

    axis_block = trend_source.split("series.forEach((item, seriesIndex) => {", 1)[1].split(
        "});", 1
    )[0]

    assert 'overlaying: "y"' in axis_block
    assert "overlaying: labelled ? undefined" not in axis_block
    assert 'showticklabels: labelled' in axis_block
    assert 'const labelled = seriesIndex === 1;' in axis_block
    # Independent axes keep their own real range instead of being normalized.
    assert "axis.range = [limits.minimum, limits.maximum]" in axis_block


def test_trend_layout_drops_the_in_plot_note_and_grows_the_curve_area() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]

    # The Y axis mode hint belongs to the controls, not the plot area.
    assert "annotations:" not in trend_source
    assert "同一 Y 轴：所有曲线使用同一数值范围" not in trend_source
    assert "独立 Y 轴：各曲线按自身范围缩放" not in trend_source

    assert "margin: {left: 55" in trend_source
    assert "top: 10" in trend_source
    assert "bottom: 60" in trend_source
    # Height follows the container instead of a hard-coded value.
    assert "height = Math.max(420, Math.floor(container.clientHeight || 520));" in trend_source
    assert "height: 320" not in trend_source


def test_trend_drops_rangeslider_and_rangeselector_without_touching_the_window() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]
    xaxis_block = trend_source.split("xaxis: {", 1)[1].split("yaxis: {", 1)[0]

    # The zoom-slider axis and the 1h/8h/24h buttons only steal plot area.
    assert "rangeslider" not in xaxis_block
    assert "rangeselector" not in xaxis_block
    for label in ("1h", "8h", "24h", "7d", "全部"):
        assert f'label: "{label}"' not in xaxis_block
    # The freed space becomes chart height.
    assert ".dp-chart { min-height:660px; height:660px;" in web_model_results.INDEX_HTML

    # Browsing controls are visual only: they must not write the business window.
    relayout = trend_source.split('plot.on("plotly_relayout"', 1)[1].split("});", 1)[0]
    assert 'dpTrendStart").value' not in relayout
    assert 'dpTrendEnd").value' not in relayout
    assert "setTrendWindowFromSelection" not in relayout
    # Only box selection writes the window.
    assert "setTrendWindowFromSelection(range[0], range[1]);" in trend_source


def test_trend_modebar_keeps_only_everyday_controls() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]
    config = trend_source.split("const config = {", 1)[1].split("};", 1)[0]

    # The whitelist is the single source of truth: zoom, pan, box select, reset.
    assert 'modeBarButtons: [["zoom2d", "pan2d", "select2d", "resetScale2d"]]' in config
    # A removal list on top of the whitelist is a no-op and must not drift from it.
    assert "modeBarButtonsToRemove" not in config
    for absent in ("toImage", "sendDataToCloud", "zoomIn2d", "zoomOut2d", "autoScale2d"):
        assert absent not in config
    assert "displaylogo: false" in config
    assert "scrollZoom: false" in config


def test_trend_axis_and_hover_use_the_short_industrial_time_format() -> None:
    trend_source = web_model_results.INDEX_HTML.split(
        "function renderTrendChart(data)", 1
    )[1].split("function renderStatCard", 1)[0]
    xaxis_block = trend_source.split("xaxis: {", 1)[1].split("yaxis: {", 1)[0]

    assert 'tickformat: "%m/%d %H:%M"' in xaxis_block
    assert 'hoverformat: "%m/%d %H:%M"' in xaxis_block
    assert 'hovermode: "x unified"' in trend_source
    # No year, seconds or ISO strings in the displayed format.
    assert "%Y" not in xaxis_block
    assert "%S" not in xaxis_block


def test_trend_legend_is_horizontal_and_replaces_the_duplicate_html_legend() -> None:
    html = web_model_results.INDEX_HTML
    trend_source = html.split("function renderTrendChart(data)", 1)[1].split(
        "function renderStatCard", 1
    )[0]

    assert 'legend: {orientation: "h", x: 0, y: -0.15' in trend_source
    assert "showlegend: true" in trend_source
    # The old hand-drawn legend would duplicate the Plotly one. (A dangling
    # `.dp-legend` CSS rule may remain in web_model_results.py styling; only the
    # markup and the renderer must be gone.)
    assert 'id="dpTrendLegend"' not in html
    assert "dpTrendLegend" not in html
    assert "dp-swatch" not in html


def test_plotly_bundle_is_vendored_and_served_without_network_dependency() -> None:
    from pca_model_builder import web

    assert web.PLOTLY_JS_PATH.is_file()
    bundle = web.PLOTLY_JS_PATH.read_text(encoding="utf-8", errors="ignore")
    assert bundle.startswith("/**")
    assert "plotly.js v" in bundle[:200]
    assert (web.PLOTLY_JS_PATH.with_name("plotly.min.js.LICENSE")).is_file()

    source = inspect.getsource(web._Handler.do_GET)

    assert 'parsed.path == "/assets/plotly.min.js"' in source
    assert 'PLOTLY_JS_PATH.read_bytes()' in source


def test_trend_reset_restores_uploaded_defaults_without_clearing_windows() -> None:
    html = web_model_results.INDEX_HTML
    reset_source = html.split('$("dpTrendReset").addEventListener("click", () => {', 1)[1].split(
        '$("dpDrawScatter").addEventListener', 1
    )[0]

    assert 'id="dpTrendMaxPoints" type="number" min="100" max="100000" value="100000"' in html
    assert html.index('id="dpTrendToExclusion"') < html.index('id="dpTrendReset"')
    assert "趋势复位" in html
    assert "defaults?.trend_default_start" in reset_source
    assert "defaults?.trend_default_end" in reset_source
    assert "localTime(defaults.trend_default_start)" in reset_source
    assert "localTime(defaults.trend_default_end)" in reset_source
    assert '$("dpTrendMaxPoints").value = "100000";' in reset_source
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
        "grid-template-columns:max-content minmax(0,1fr) "
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
    # 窄屏下文件选择控件独占一行：grid 两列形态用 grid-column，flex 形态用 flex-basis。
    assert "flex:1 1 100%;" in html
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
    # 侧栏宽度必须给步骤正文留够一行标题：218px 时正文列仅 74px，「正常状态候选」
    # 折 2 行、「下一步：仅已验证模型可冻结并导出部署包」折 4 行。260px 保留紧凑
    # 导航同时让正文列达到 116px，仍远小于已被禁用的 630px。
    assert "main { grid-template-columns:260px minmax(0,1fr); align-items:start; }" in html
    assert "grid-template-columns:630px minmax(0,1fr);" not in html
    assert ".controls, .controls .group { min-width:0; }" in html
    assert "max-width:100%;" in html
    assert ".results { gap:24px; }" in html
    assert ".empty, .variance, .exploration-timeline" in html
    assert "#engineeringPanel #tagRole," in html
    assert "height:42px;" in html
    assert "height:30px;" in html
    assert "grid-template-columns:repeat(auto-fit,minmax(132px,1fr));" in html
    assert "font-size:22px;" in html
    assert ".metric { min-height:64px; align-content:start; }" in html
    assert "th, td { border-bottom-color:var(--line); padding:8px 12px; }" in html
    assert "input[type=checkbox] {" in html
    assert "accent-color:var(--accent);" in html
    assert ".exploration-candidate-comment {" in html
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


def test_final_web_uses_shared_control_and_responsive_layout_tokens() -> None:
    html = web_model_results.INDEX_HTML

    for token in (
        "--control-height:42px;",
        "--control-radius:6px;",
        "--space-1:8px;",
        "--space-2:12px;",
        "--space-3:16px;",
        "--space-4:24px;",
        "--panel-padding:24px;",
    ):
        assert token in html
    for rule in (
        "height:var(--control-height);",
        "display:inline-flex;",
        "align-items:center;",
        "justify-content:center;",
        "min-width:0;",
        "max-width:100%;",
        "@media (max-width:1050px)",
        "main { grid-template-columns:minmax(0,1fr); }",
        "@media (max-width:760px)",
        ".candidate-manager .row { grid-template-columns:minmax(0,1fr); }",
        ".candidate-tool-tabs { flex-wrap:nowrap; overflow-x:auto; }",
    ):
        assert rule in html


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


def test_candidate_tools_precede_candidate_window_manager() -> None:
    html = web_model_results.INDEX_HTML
    candidate_start = html.index('<div id="candidatePanel"')
    model_start = html.index('<div id="modelPanel"', candidate_start)
    candidate_source = html[candidate_start:model_start]

    assert candidate_source.index('id="trendPanel"') < candidate_source.index(
        'id="stateExplorationPanel"'
    ) < candidate_source.index('id="candidateWindows"')
    assert candidate_source.count('id="candidateWindows"') == 1
    assert candidate_source.count('id="excludedWindows"') == 1


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
    shared_start = html.index('<div class="group shared-preprocessing">')
    shared_end = html.index('<div class="candidate-tool-tabs"', shared_start)
    shared_source = html[shared_start:shared_end]
    model_start = html.index('<div class="group training-configuration">')
    model_end = html.index('<details class="advanced-parameters exploratory-model-tools">', model_start)
    model_source = html[model_start:model_end]

    for field_id in (
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
    ):
        assert f'id="{field_id}"' in shared_source, field_id
        assert f'id="{field_id}"' not in model_source, field_id
    for field_id in ("varianceThreshold", "components", "modelName"):
        assert f'id="{field_id}"' in model_source, field_id
        assert f'id="{field_id}"' not in shared_source, field_id
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
    # 建模质量检查与预处理预览必须回到模型训练阶段，而不是留在候选阶段。
    for field_id in (
        "preprocessingPreviewWindow",
        "preprocessingPreviewButton",
        "preprocessingPreview",
        "qualityButton",
        "modelQualityStatus",
        "modelQualityResults",
    ):
        assert f'id="{field_id}"' in model_source, field_id
        assert f'id="{field_id}"' not in shared_source, field_id
    assert "高级预处理与 DPCA 参数" not in html
    assert "模型训练配置（参考状态与 DPCA 参数）" not in html
    assert "分析与建模共享参数" in shared_source
    assert "PCA 模型配置与训练" in model_source


def test_training_configuration_precedes_model_results() -> None:
    html = web_model_results.INDEX_HTML
    model_start = html.index('<div id="modelPanel"')
    validation_start = html.index('<div id="validationPanel"', model_start)
    model_source = html[model_start:validation_start]

    assert model_source.index('class="group training-configuration"') < model_source.index(
        'id="modelEmpty"'
    )
    assert model_source.index('id="trainButton"') < model_source.index(
        'id="modelContent"'
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

    html = web_model_results.INDEX_HTML
    release = html[html.index('<div id="releasePanel"'):]
    assert release.count('id="frozenReplay"') == 1
    assert release.index('id="freezeDeployment"') < release.index('id="frozenReplay"')
    assert 'releasePanel.append(replayCard)' not in source


def test_loading_plot_uses_origin_lines_without_arrowheads() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")

    assert "原始Tag聚合载荷图" in web_model_results.INDEX_HTML
    assert "每条连线从原点连接到" in web_model_results.INDEX_HTML
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

    model = html[html.index('<div id="modelPanel"'):html.index('<div id="validationPanel"')]
    projection = model[model.index('class="model-projection-grid"'):model.index('id="componentLoadings"')]
    assert projection.index('id="scoreChart"') < projection.index('id="loadingChart"')
    assert 'projectionGrid.append(scoreCard, section)' not in source
    assert 'id="modelResultsStyle"' in html
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr);" in html
    assert ".model-projection-grid #scoreChart svg" in html
    assert ".model-projection-grid #loadingChart svg" in html
    assert 'document.createElement("style")' not in source
    assert "document.head.append(" not in source
    assert 'insertAdjacentElement("afterend", section)' not in source


def test_component_loadings_show_first_six_in_responsive_grid() -> None:
    source = (
        PROJECT_ROOT / "src" / "pca_model_builder" / "model_results.js"
    ).read_text(encoding="utf-8")
    html = web_model_results.INDEX_HTML

    assert html.count('id="componentLoadings"') == 1
    assert html.count('id="componentLoadingsContent"') == 1
    assert "renderComponentLoadings(data.loading_plot?.component_loadings)" in source
    assert "component.top_loadings" in source
    assert "components.slice(0, 6).forEach((component, index) =>" in source
    assert "仅显示前 6 个主元。" in html
    assert "Top ${topRows.length} 原始变量（同一Tag的全部Lag已聚合）" in source
    assert "查看全部" not in source
    assert "component-loading-full" not in source
    assert "聚合 loading 强度" in source
    assert "cell(formatLoading(Number(row?.aggregated_loading)))" in source
    assert "T²/SPE 异常贡献" in html
    assert html.index('id="componentLoadings"') < html.index('id="modelStructureComparison"')
    assert "grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr));" in html
    assert "#componentLoadings .component-loading-table" in html


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
        '<div class="row preprocessing-parameter-row"><label>目标采样周期（分钟）',
        '<div data-test="stable" class="row compact preprocessing-parameter-row"><label>目标采样周期（分钟）',
        1,
    )

    html = web_model_results._stabilize_workbench_html(changed_html)
    shared_start = html.index('<div class="group shared-preprocessing">')
    shared_source = html[shared_start : html.index('<div class="candidate-tool-tabs"', shared_start)]
    model_start = html.index('<div class="group training-configuration">')
    model_source = html[
        model_start : html.index('<details class="advanced-parameters exploratory-model-tools">', model_start)
    ]

    for field_id in ("sampleInterval", "resamplingMethod", "filterMethod", "firstOrderAlpha", "smoothingWindow", "gapThreshold", "maxLag", "lagStep"):
        assert f'id="{field_id}"' in shared_source
        assert f'id="{field_id}"' not in model_source
    for field_id in ("varianceThreshold", "components", "modelName"):
        assert f'id="{field_id}"' in model_source
        assert f'id="{field_id}"' not in shared_source
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

    row = html.split('<div class="row candidate-window-row">', 1)[1].split(
        "</div>", 1
    )[0]
    for element_id in (
        'id="candidateStart"',
        'id="candidateEnd"',
        'id="candidateComment"',
        'id="addManualCandidate"',
    ):
        assert element_id in row
    assert row.count("<label>") == 3
    # 候选时间、备注与按钮四项同排，按钮按文案宽度收缩。
    assert ".candidate-manager .candidate-window-row {" in html
    assert "grid-template-columns:minmax(0,1.5fr) minmax(0,1.5fr) minmax(0,1.4fr) max-content;" in html
    # 窄屏仍回落为单列，避免时间控件被压扁。
    assert ".candidate-manager .candidate-window-row { grid-template-columns:minmax(0,1fr); }" in html

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
    assert 'async function addCandidateWindow(source,start,end,sourceRef=null,comment="")' in html
    assert 'status="pending"' not in html
    assert 'addCandidateWindow("manual"' in html
    assert 'addCandidateWindow("cluster"' in html
    assert 'addCandidateWindow("trend"' in html
    assert 'addCandidateWindow("performance"' in html
    assert 'button.textContent="填入正常期"' not in html
    assert "normalStart" not in html
    assert "normalEnd" not in html


def test_final_web_uses_read_only_candidate_status_and_non_training_conversion() -> None:
    html = web_model_results.INDEX_HTML

    assert 'id="convertExplorationCandidates"' in html
    assert 'id="saveExplorationCandidateDecisions"' not in html
    assert "exploration-candidate-decision" not in html
    assert "exploration-candidate-comment" in html
    assert "选择候选后加入统一候选窗口列表" in html
    assert "请在候选窗口列表确认作为训练窗口" in html
    assert "/decisions" not in html


def test_model_quality_check_is_in_the_model_training_stage() -> None:
    html = web_model_results.INDEX_HTML
    candidate_start = html.index('<div id="candidatePanel"')
    model_start = html.index('<div id="modelPanel"')
    model_end = html.index('<div id="validationPanel"')
    candidate_source = html[candidate_start:model_start]
    model_source = html[model_start:model_end]

    assert "执行建模质量检查" in model_source
    assert 'id="modelQualityStatus"' in model_source
    assert 'id="modelQualityResults"' in model_source
    assert model_source.index('id="qualityButton"') < model_source.index('id="modelQualityStatus"')
    assert model_source.index('id="modelQualityStatus"') < model_source.index('id="trainingCompositionReview"')
    assert 'id="modelTrainingDataSummary"' in model_source
    assert 'id="trainButton"' in model_source
    # 质量检查依赖 training_windows + 共享预处理 + Tag 配置，只能出现在模型训练阶段。
    assert 'id="qualityButton"' not in candidate_source
    assert 'id="modelQualityResults"' not in candidate_source
    assert "上传后基础数据检查" in html
    assert "此处仅展示整体历史数据的时间轴与原始逐列检查结果" in html
    assert "column_profiles" in html
    assert "有效数值" in html
    assert "状态 / 建议" in html
    # 不使用“stale 徽标”方案：配置变化后直接清空旧状态探索结果。
    assert "stale" not in html.lower()
    assert "已失效" not in html.split("<script>", 1)[0]


def test_model_training_stage_reads_summary_quality_preview_then_parameters() -> None:
    html = web_model_results.INDEX_HTML
    model_start = html.index('<div id="modelPanel"')
    model_end = html.index('<div id="validationPanel"', model_start)
    model_source = html[model_start:model_end]
    order = [
        'id="modelTrainingDataSummary"',
        'id="preprocessingPreviewWindow"',
        'id="preprocessingPreviewButton"',
        'id="preprocessingPreview"',
        'id="qualityButton"',
        'id="modelQualityStatus"',
        'id="modelQualityResults"',
        'id="modelName"',
        'id="varianceThreshold"',
        'id="components"',
        'id="trainButton"',
    ]
    positions = [model_source.index(item) for item in order]

    assert positions == sorted(positions)


def test_model_quality_controls_and_detail_table_do_not_stretch() -> None:
    html = web_model_results.INDEX_HTML
    candidate_start = html.index('<div id="candidatePanel"')
    model_start = html.index('<div id="modelPanel"', candidate_start)
    candidate_source = html[candidate_start:model_start]
    model_source = html[model_start : html.index('<div id="validationPanel"', model_start)]

    for element_id in ("qualityButton", "modelQualityStatus", "currentTagQuality"):
        assert f'id="{element_id}"' in model_source
        assert f'id="{element_id}"' not in candidate_source
    assert "#modelPanel #modelQualityStatus," in html
    assert "#modelPanel #qualityButton {" in html
    assert "#modelPanel #modelQualityStatus {" in html
    assert "height:42px;" in html
    assert "min-height:42px;" in html
    assert "#modelPanel #modelQualityStatus {" in html
    assert "max-width:100%;" in html.split("#modelPanel #modelQualityStatus {", 1)[1].split("}", 1)[0]
    # 状态与按钮并排成一行，纯文字提示，不占色块；宽度随文案自适应避免长错误被裁切。
    status_style = html.rsplit("#modelPanel #modelQualityStatus {", 1)[1].split("}", 1)[0]
    assert "background:#ffffff;" in status_style
    assert "border:0;" in status_style
    # 并排由父容器完成：负 margin(-55px/-150px) 是导致按钮被状态块水平重叠 146px
    # 的根因，改为 .quality-action-row 的 flex 排布后不得再出现任何负偏移。
    assert "margin-top:-55px;" not in status_style
    assert "margin-left:150px;" not in status_style
    assert "min-width:72px;" in status_style
    assert "width:auto;" in status_style
    assert "min-height:42px;" in status_style
    assert "#modelPanel .quality-action-row {" in html
    row_style = html.rsplit("#modelPanel .quality-action-row {", 1)[1].split("}", 1)[0]
    assert "display:flex;" in row_style
    assert "flex-wrap:wrap;" in row_style
    assert "align-items:center;" in row_style
    assert 'class="quality-action-row"' in model_source
    assert "#modelPanel #currentTagQuality { max-width:1200px; }" in html
    assert "#candidatePanel #modelQualityStatus" not in html
    assert "#candidatePanel #qualityButton" not in html
    assert "#candidatePanel #currentTagQuality" not in html


def test_model_training_summary_reuses_quality_totals_and_clears_on_invalidation() -> None:
    html = web_model_results.INDEX_HTML
    summary_source = html.split(
        "function renderModelTrainingDataSummary", 1
    )[1].split("function invalidateQuality", 1)[0]
    invalidate_source = html.split("function invalidateQuality", 1)[1].split(
        "function firstOrderAlphaError", 1
    )[0]
    quality_source = html.split("function renderQuality(data)", 1)[1].split(
        "function excludeConstantTag", 1
    )[0]

    assert "需重新执行建模质量检查后显示训练数据摘要。" in html
    for field in (
        "used_window_count",
        "enabled_window_count",
        "training_rows",
        "covered_day_count",
        "max_window_effective_share",
    ):
        assert f"totals.{field}" in summary_source
    assert "renderModelTrainingDataSummary(data.training_window_totals)" in quality_source
    assert "renderModelTrainingDataSummary(null" in invalidate_source
    assert "需重新执行建模质量检查" in invalidate_source
    assert "window_summaries" not in summary_source
    assert "trainingWindowSummary" not in summary_source


def test_model_quality_status_tracks_check_and_configuration_changes() -> None:
    html = web_model_results.INDEX_HTML
    invalidate_source = html.split("function invalidateQuality", 1)[1].split(
        "function firstOrderAlphaError", 1
    )[0]
    quality_request_source = html.split(
        'el("qualityButton").addEventListener("click",async()=>{', 1
    )[1].split('el("qualityTagSelect")', 1)[0]

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
    assert "qualityRevision:0" in html
    assert "state.qualityRevision+=1" in invalidate_source
    assert (
        'state.qualityStatus=reason&&(checked||checking)?"changed":"unchecked"'
        in invalidate_source
    )
    assert 'el("trainButton").disabled=true' in html
    assert 'state.qualityStatus=readiness.normal_state.can_train&&readiness.exploratory.can_train?"passed":"issues"' in html
    assert 'state.qualityStatus="failed"' in html
    assert "qualityRevision=state.qualityRevision" in quality_request_source
    assert quality_request_source.count("if(qualityRevision!==state.qualityRevision) return;") == 2
    assert quality_request_source.index(
        "if(qualityRevision!==state.qualityRevision) return;"
    ) < quality_request_source.index("state.quality=data;")
    assert quality_request_source.rindex(
        "if(qualityRevision!==state.qualityRevision) return;"
    ) < quality_request_source.index('state.qualityStatus="failed"')


def test_invalidate_quality_executes_actual_status_boundaries() -> None:
    html = web_model_results.INDEX_HTML
    function_source = "function invalidateQuality" + html.split("function invalidateQuality", 1)[1].split(
        "function firstOrderAlphaError", 1
    )[0]
    _run_web_javascript(
        f"""
        const functionSource = {json.dumps(function_source)};
        const elements = new Map([
          ["trainButton", {{disabled: false}}],
          ["trainExploratoryButton", {{disabled: false}}],
          ["qualitySummary", {{innerHTML: "old"}}],
          ["trainingCompositionReview", {{className: "", textContent: "old"}}],
          ["qualityIssues", {{className: "", textContent: "old"}}],
          ["modelTrainingDataSummary", {{className: "", textContent: "old"}}],
          ["excludeAllConstants", {{disabled: false}}],
          ["modelQualityStatus", {{className: "", textContent: ""}}],
          ["currentTagQuality", {{className: "", textContent: ""}}],
          ["qualityTagSelect", null],
        ]);
        const el = id => elements.get(id);
        const state = {{
          quality:null, qualityStatus:"unchecked", qualityRevision:0, qualityError:""
        }};
        function renderModelTrainingDataSummary() {{}}
        function renderModelQualityStatus() {{}}
        function renderCurrentTagQuality() {{}}
        function setStatus() {{}}
        {function_source}

        const snapshots = [];
        for (const scenario of [
          {{quality:null, status:"unchecked", reason:"Tag changed"}},
          {{quality:null, status:"passed", reason:"Tag changed"}},
          {{quality:{{tags:[]}}, status:"passed", reason:"Tag changed"}},
          {{quality:null, status:"failed", reason:"Tag changed"}},
          {{quality:null, status:"checking", reason:"Tag changed"}},
          {{quality:null, status:"unchecked", reason:undefined}},
        ]) {{
          state.quality=scenario.quality;
          state.qualityStatus=scenario.status;
          const before=state.qualityRevision;
          invalidateQuality(scenario.reason);
          snapshots.push([
            state.qualityStatus,
            state.qualityRevision-before,
            el("trainButton").disabled,
            el("trainExploratoryButton").disabled,
          ]);
        }}
        if (JSON.stringify(snapshots) !== JSON.stringify([
          ["unchecked",1,true,true],
          ["unchecked",1,true,true],
          ["changed",1,true,true],
          ["changed",1,true,true],
          ["changed",1,true,true],
          ["unchecked",1,true,true],
        ])) throw new Error(JSON.stringify(snapshots));
        """
    )


def test_quality_handler_executes_stale_and_current_result_boundaries() -> None:
    html = web_model_results.INDEX_HTML
    request_start = html.split(
        'el("qualityButton").addEventListener("click",async()=>{', 1
    )[1].split("\n  try {", 1)[0]
    try_block = "  try {" + html.split(
        'el("qualityButton").addEventListener("click",async()=>{', 1
    )[1].split("\n  try {", 1)[1].split("\n  finally", 1)[0]
    totals = {
        "enabled_window_count": 1,
        "used_window_count": 1,
        "training_rows": 8,
        "covered_day_count": 1,
        "max_window_effective_share": 1.0,
    }
    handler_source = "return (async()=>{" + request_start + try_block + "})();"
    harness = f"""
        const requestStart = {json.dumps(request_start)};
        const tryBlock = {json.dumps(try_block)};
        const handlerSource = {json.dumps(handler_source)};
        const elements = new Map([
          ["qualityButton", {{disabled:false, dataset:{{}}, textContent:"check"}}],
          ["trainButton", {{disabled:true}}],
          ["trainExploratoryButton", {{disabled:true}}],
          ["qualitySummary", {{innerHTML:""}}],
          ["qualityIssues", {{className:"", textContent:""}}],
          ["trainingCompositionReview", {{className:"", textContent:""}}],
          ["modelTrainingDataSummary", {{className:"", textContent:""}}],
          ["excludeAllConstants", {{disabled:true}}],
          ["modelQualityStatus", {{className:"", textContent:""}}],
          ["currentTagQuality", {{className:"", textContent:""}}],
          ["qualityTagSelect", null],
        ]);
        const el = id => elements.get(id);
        const state = {{
          fileId:"file", runId:null, exploratoryRunId:null, inspection:{{numeric_columns:["A","B"]}},
          registry:{{}}, quality:null, qualityStatus:"unchecked", qualityRevision:0, qualityError:"",
          selectedTag:null, selectedModelTags:new Set(["A","B"]), importPreview:null, excludedTags:[],
          excludedWindows:[], showProblems:false, candidateWindows:[], trainingWindows:[],
          trainingWindowSummary:[], validationWindows:[], trend:null,
        }};
        const display = value => value;
        const trainingWindowsPayload = () => state.trainingWindows;
        const firstOrderAlphaError = () => "";
        const commonPayload = () => ({{}});
        const selectedTags = () => ["A","B"];
        const setStatus = () => {{}};
        const setBusy = () => {{}};
        const renderModelQualityStatus = () => {{}};
        const renderCurrentTagQuality = () => {{}};
        const renderTrainingWindows = () => {{}};
        const renderQuality = data => {{
          el("modelTrainingDataSummary").totals=data.training_window_totals || null;
        }};
        const renderTrainingComposition = totals => {{ el("trainingCompositionReview").totals=totals; }};
        const trainingCompositionShare = value => `${{(Number(value)*100).toFixed(1)}}%`;
        const renderModelTrainingDataSummary = totals => {{
          el("modelTrainingDataSummary").totals=totals || state.quality?.training_window_totals || null;
          el("modelTrainingDataSummary").textContent="updated";
        }};
        const renderTagList = () => {{}};
        const AsyncFunction = Object.getPrototypeOf(async function(){{}}).constructor;
        let nextResult, nextError=null;
        const api = () => new Promise((resolve,reject) => {{
          nextResult=resolve; nextError=reject;
        }});
        const showWorkflowStage = () => {{}};
        const invoke = () => new AsyncFunction(
          "api", "globalThis", handlerSource
        ).call(el("qualityButton"), api, globalThis);

        const qualityData = {{
          training_readiness:{{normal_state:{{can_train:true}},exploratory:{{can_train:true}}}},
          can_train:true, summary:{{usable:2,review:0,blocking:0}}, tags:[], time_issues:[],
          training_quality_warnings:[], training_window_summary:[],
          training_window_totals:{json.dumps(totals, separators=(",", ":"))},
        }};
        const run = async () => {{
          const staleRequest=invoke();
          if (state.qualityStatus !== "checking") throw new Error("not checking");
          state.qualityRevision+=1;
          state.qualityStatus="changed";
          nextResult({{...qualityData, training_window_totals:{{training_rows:99, used_window_count:9, enabled_window_count:9, covered_day_count:9, max_window_effective_share:0.9}}}});
          await staleRequest;
          if (state.quality || state.qualityStatus!=="changed" || el("trainButton").disabled!==true)
            throw new Error("stale success was applied");
          if (el("modelTrainingDataSummary").totals!==null ||
              el("trainingCompositionReview").totals!==undefined ||
              state.trainingWindowSummary.length!==0)
            throw new Error("stale success updated quality projections");

          const errorRequest=invoke();
          const errorRevision=state.qualityRevision;
          state.qualityRevision+=1;
          state.qualityStatus="changed";
          nextError(new Error("stale error"));
          await errorRequest;
          if (state.qualityStatus!=="changed" || state.qualityError!=="")
            throw new Error("stale error was applied");
          if (el("modelTrainingDataSummary").totals!==null ||
              el("trainingCompositionReview").totals!==undefined ||
              state.trainingWindowSummary.length!==0)
            throw new Error("stale error updated quality projections");

          const currentRequest=invoke();
          if (state.qualityRevision!==errorRevision+1) throw new Error("bad revisions");
          nextResult(qualityData);
          await currentRequest;
          if (!state.quality || state.qualityStatus!=="passed") throw new Error("current success missing");
          if (el("trainButton").disabled || el("trainExploratoryButton").disabled)
            throw new Error("training gate did not recover");
          const summary=el("modelTrainingDataSummary");
          if (summary.totals.training_rows!==8 || summary.totals.used_window_count!==1 ||
              summary.totals.covered_day_count!==1 || summary.totals.max_window_effective_share!==1)
            throw new Error(`summary mismatch: ${{summary.textContent}}`);
        }};
        run().then(() => process.stdout.write("quality-state-ok"), error => {{
          process.stderr.write(String(error.stack||error)); process.exitCode=1;
        }});
        """
    _run_web_javascript(harness)


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


def test_validation_evidence_precedes_engineer_confirmation() -> None:
    html = web_model_results.INDEX_HTML

    summary = html.index("<h3>验证状态摘要</h3>")
    confirmation = html.index('id="recordValidationDecision"')
    metrics = html.index("<h3>验证指标</h3>")

    assert summary < metrics < confirmation


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
    assert 'source:"suggested",source_ref:"inspect-default",comment:' in inspect_source
    assert 'status:"pending"' not in inspect_source
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


def test_candidate_and_training_view_trend_reloads_the_window_without_mutation() -> None:
    html = web_model_results.INDEX_HTML
    body = html.split("function showCandidateTrend(window)", 1)[1].split(
        "function renderCandidateWindows", 1
    )[0]
    view_source = f"function showCandidateTrend(window){body}"

    # The trend request must actually run again for the selected window.
    assert "requestAnimationFrame" in view_source
    assert 'el("dpDrawTrend")' in view_source
    assert ".click()" in view_source
    # Pure navigation: no candidate/training/exclusion state may be touched.
    assert "state.candidateWindows" not in view_source
    assert "state.trainingWindows" not in view_source
    assert "state.excludedWindows" not in view_source
    assert "confirmCandidateWindow" not in view_source
    # The window itself is set before switching to the trend panel.
    assert 'el("dpTrendStart").value=localTime(window.start)' in view_source
    assert 'el("dpTrendEnd").value=localTime(window.end)' in view_source

    _run_web_javascript(
        f"""
        const functionSource = {json.dumps(view_source)};
        const elements = new Map([
          ["trendStart", {{value: ""}}],
          ["trendEnd", {{value: ""}}],
          ["dpTrendStart", {{value: ""}}],
          ["dpTrendEnd", {{value: ""}}],
          ["dpDrawTrend", {{disabled: false, clicks: 0, click() {{ this.clicks += 1; }}}}],
        ]);
        const el = id => elements.get(id);
        const state = {{candidateWindows: [], trainingWindows: [], excludedWindows: []}};
        let panelClicks = 0;
        const frames = [];
        globalThis.requestAnimationFrame = fn => frames.push(fn);
        globalThis.document = {{
          querySelector: () => ({{click: () => {{ panelClicks += 1; }}}}),
        }};
        function localTime(value) {{ return value.slice(0, 16); }}
        function setStatus() {{}}
        eval(functionSource);

        showCandidateTrend({{start: "2026-01-01T10:00:00", end: "2026-01-01T12:00:00"}});
        if (el("dpTrendStart").value !== "2026-01-01T10:00") throw new Error("start not set");
        if (el("dpTrendEnd").value !== "2026-01-01T12:00") throw new Error("end not set");
        if (panelClicks !== 1) throw new Error("trend panel not opened");
        if (el("dpDrawTrend").clicks !== 0) throw new Error("drew before panel settled");
        while (frames.length) frames.shift()();
        if (el("dpDrawTrend").clicks !== 1) throw new Error("trend was not reloaded");
        if (state.candidateWindows.length || state.trainingWindows.length || state.excludedWindows.length) {{
          throw new Error("window state was mutated");
        }}
        """
    )


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
    assert '["pending","accepted","rejected"]' not in candidate_source
    assert "candidateTrainingWindows(window).length>0" in candidate_source
    assert 'status.textContent=displayUiValue(generated?"accepted":"pending")' in candidate_source
    assert 'label==="确认作为训练窗口"&&generated' in candidate_source
    assert "window.status" not in candidate_source
    assert 'document.createElement("select")' not in candidate_source
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


def test_preprocessing_preview_uses_the_selected_enabled_training_window() -> None:
    html = web_model_results.INDEX_HTML
    render_source = "function renderPreprocessingPreviewWindow" + html.split(
        "function renderPreprocessingPreviewWindow", 1
    )[1].split("function selectedPreprocessingPreviewWindow", 1)[0]
    invalidate_preview_source = "function invalidatePreprocessingPreview" + html.split(
        "function invalidatePreprocessingPreview", 1
    )[1].split("function renderPreprocessingPreviewWindow", 1)[0]
    handler_source = html.split(
        'el("preprocessingPreviewButton").addEventListener("click",async()=>{', 1
    )[1].split("function preprocessingPreviewTags(data)", 1)[0]

    # 选项只来自 enabled === true 的训练窗口，默认第一个启用窗口。
    assert "state.trainingWindows.filter(window=>window.enabled)" in render_source
    assert "state.preprocessingPreviewWindowId=enabled[0]?.id||null" in render_source
    assert "没有启用的训练窗口" in render_source
    assert "select.disabled=!enabled.length; button.disabled=!enabled.length;" in render_source
    # 请求的 start/end 来自训练窗口，不再隐式读取 trendStart/trendEnd。
    assert "start:window.start,end:window.end" in handler_source
    assert 'el("trendStart")' not in handler_source
    assert 'el("trendEnd")' not in handler_source
    assert "没有启用的训练窗口，无法预览预处理。" in handler_source
    assert "state.trainingWindows.find(window=>window.enabled&&window.id===state.preprocessingPreviewWindowId)" in html
    # 训练窗口增删改启停后同步刷新选择器。
    assert "function renderTrainingWindows()" in html
    render_training = html.split("function renderTrainingWindows()", 1)[1].split(
        "async function updateTrainingWindows", 1
    )[0]
    assert "renderPreprocessingPreviewWindow();" in render_training
    # 重新渲染训练窗口表格本身不再无条件清空预览。
    assert "训练窗口已变化，请重新预览。" not in html
    assert "invalidatePreprocessingPreview(" in render_source
    # 预览请求记录窗口起止时间，用于判断窗口是否真的变化。
    assert "windowId:window.id,start:window.start,end:window.end" in html
    # UI 明确显示当前预览的是哪个训练窗口。
    preview_render = html.split("function renderPreprocessingPreview()", 1)[1].split(
        "function preprocessingPreviewStages", 1
    )[0]
    assert "预览训练窗口：" in preview_render

    _run_web_javascript(
        f"""
        const invalidatePreviewSource = {json.dumps(invalidate_preview_source)};
        const renderSource = {json.dumps(render_source)};
        const makeOption = () => ({{value:"", textContent:"", disabled:false, selected:false}});
        const makeSelect = () => ({{
          children:[], disabled:false, value:"",
          replaceChildren(){{ this.children.length = 0; }},
          append(node){{ this.children.push(node); if(node.selected) this.value=node.value; }},
        }});
        const elements = new Map([
          ["preprocessingPreviewWindow", makeSelect()],
          ["preprocessingPreviewButton", {{disabled:false}}],
          ["preprocessingPreview", {{className:"", textContent:""}}],
        ]);
        const el = id => elements.get(id);
        const state = {{trainingWindows:[], preprocessingPreview:null, preprocessingPreviewTag:null, preprocessingPreviewWindowId:null}};
        const displayTime = value => value;
        globalThis.document = {{createElement: tag => tag === "option" ? makeOption() : makeSelect()}};
        eval(invalidatePreviewSource);
        eval(renderSource);

        // 没有启用训练窗口：禁止预览并给出明确提示。
        renderPreprocessingPreviewWindow();
        if(!el("preprocessingPreviewWindow").disabled) throw new Error("select should be disabled");
        if(!el("preprocessingPreviewButton").disabled) throw new Error("button should be disabled");
        if(!el("preprocessingPreviewWindow").children[0].textContent.includes("没有启用的训练窗口"))
          throw new Error("missing empty hint");

        // 三个训练窗口、两个启用：默认选第一个启用窗口。
        state.trainingWindows=[
          {{id:"training-a", start:"2026-01-01T10:00", end:"2026-01-01T12:00", enabled:true}},
          {{id:"training-b", start:"2026-01-01T13:00", end:"2026-01-01T15:00", enabled:false}},
          {{id:"training-c", start:"2026-01-01T16:00", end:"2026-01-01T18:00", enabled:true}},
        ];
        state.preprocessingPreview={{data:{{}}, tags:["A"], windowId:"training-a", start:"2026-01-01T10:00", end:"2026-01-01T12:00"}};
        renderPreprocessingPreviewWindow();
        if(state.preprocessingPreviewWindowId!=="training-a") throw new Error("default window wrong");
        const options=el("preprocessingPreviewWindow").children;
        if(options.length!==2) throw new Error("disabled window must not be listed");
        if(options[0].value!=="training-a"||options[1].value!=="training-c") throw new Error("wrong options");
        if(!el("preprocessingPreviewWindow").children[0].textContent.includes("training-a"))
          throw new Error("option text must show the window id and range");
        if(state.preprocessingPreview===null) throw new Error("valid preview must be kept");
        if(el("preprocessingPreviewButton").disabled) throw new Error("button must be enabled");

        // 质量检查后仅重渲染训练窗口：窗口未变，预览保留。
        state.trainingWindowSummary=[{{id:"training-a", raw_samples:120, effective_samples:118}}];
        renderPreprocessingPreviewWindow();
        if(state.preprocessingPreview===null) throw new Error("quality rerender must keep preview");
        if(state.preprocessingPreviewTag!==null) throw new Error("preview tag untouched");

        // 编辑当前预览窗口的起止时间：预览失效。
        state.trainingWindows[0].end="2026-01-01T12:30";
        renderPreprocessingPreviewWindow();
        if(state.preprocessingPreview!==null||state.preprocessingPreviewTag!==null) throw new Error("edited window must drop preview");
        if(!el("preprocessingPreview").textContent.includes("起止时间")) throw new Error("missing edit hint");

        // 恢复时间范围后重新预览，再禁用当前窗口：选择器回退且预览失效。
        state.trainingWindows[0].end="2026-01-01T12:00";
        state.preprocessingPreview={{data:{{}}, tags:["A"], windowId:"training-a", start:"2026-01-01T10:00", end:"2026-01-01T12:00"}};
        state.preprocessingPreviewTag="A";
        state.trainingWindows[0].enabled=false;
        renderPreprocessingPreviewWindow();
        if(state.preprocessingPreviewWindowId!=="training-c") throw new Error("selector must fall back");
        if(state.preprocessingPreview!==null) throw new Error("disabled window must drop preview");

        // 删除当前预览窗口：预览失效。
        state.trainingWindows[0].enabled=true;
        state.preprocessingPreview={{data:{{}}, tags:["A"], windowId:"training-a", start:"2026-01-01T10:00", end:"2026-01-01T12:00"}};
        state.trainingWindows=state.trainingWindows.filter(window=>window.id!=="training-a");
        renderPreprocessingPreviewWindow();
        if(state.preprocessingPreview!==null) throw new Error("removed window must drop preview");
        if(state.preprocessingPreviewWindowId!=="training-c") throw new Error("selector not synced after removal");

        // 切换到第二个启用窗口后，解析出的 start/end 来自该窗口。
        state.preprocessingPreviewWindowId="training-c";
        """
    )

    resolve_source = "function selectedPreprocessingPreviewWindow" + html.split(
        "function selectedPreprocessingPreviewWindow", 1
    )[1].split("function windowSummary", 1)[0]
    _run_web_javascript(
        f"""
        const resolveSource = {json.dumps(resolve_source)};
        const state = {{trainingWindows:[
          {{id:"training-a", start:"2026-01-01T10:00", end:"2026-01-01T12:00", enabled:true}},
          {{id:"training-c", start:"2026-01-01T16:00", end:"2026-01-01T18:00", enabled:true}},
        ]}};
        eval(resolveSource);

        state.preprocessingPreviewWindowId="training-c";
        const chosen=selectedPreprocessingPreviewWindow();
        if(chosen.id!=="training-c") throw new Error("explicit selection ignored");
        if(chosen.start!=="2026-01-01T16:00"||chosen.end!=="2026-01-01T18:00") throw new Error("wrong range");
        state.preprocessingPreviewWindowId="training-removed";
        if(selectedPreprocessingPreviewWindow().id!=="training-a") throw new Error("fallback wrong");
        state.preprocessingPreviewWindowId="training-c";
        state.trainingWindows=[{{id:"training-a", enabled:true}},{{id:"training-c", enabled:false}}];
        if(selectedPreprocessingPreviewWindow().id!=="training-a") throw new Error("must fall back to an enabled window");
        state.trainingWindows=[{{id:"training-a", enabled:false}}];
        if(selectedPreprocessingPreviewWindow()!==null) throw new Error("no enabled window must resolve to null");
        """
    )


def test_preprocessing_preview_window_selector_ignores_trend_range() -> None:
    html = web_model_results.INDEX_HTML

    assert 'el("preprocessingPreviewWindow").addEventListener("change"' in html
    assert "state.preprocessingPreviewWindowId=el(\"preprocessingPreviewWindow\").value" in html
    # 趋势浏览时间范围与预处理预览互不影响。
    trend_handler = html.split('el("trendButton").addEventListener("click",async()=>{', 1)[1].split(
        'el("preprocessingPreviewButton")', 1
    )[0]
    assert "preprocessingPreviewWindowId" not in trend_handler


def test_invalidate_exploration_clears_results_and_unconfirmed_candidates() -> None:
    html = web_model_results.INDEX_HTML
    function_source = "function invalidateExploration" + html.split(
        "function invalidateExploration", 1
    )[1].split("async function updateExplorationPreferredRegion", 1)[0]
    helper_source = "function confirmedExplorationSourceRefs" + html.split(
        "function confirmedExplorationSourceRefs", 1
    )[1].split("function invalidateExploration", 1)[0]

    assert 'startsWith("state-exploration-")' in function_source
    assert "state.exploration=null" in function_source
    assert "resetExplorationRegion()" in function_source
    # 探索失效只负责 exploration / 候选；quality 由集中入口 invalidateModellingResults 统一调用。
    assert "invalidateQuality(" not in function_source
    assert "state.explorationRevision+=1" in function_source
    # 只删除未确认候选：已有训练窗口或已生成训练窗口的 exploration 候选保留。
    assert "confirmedExplorationSourceRefs()" in function_source
    assert "candidateTrainingWindows(window).length>0" in function_source
    # 已确认训练窗口不因探索失效而删除。
    assert "state.trainingWindows=state.trainingWindows.filter" not in function_source

    # 所有影响动态矩阵的配置变化都走同一个集中入口，且只调用一次 invalidateQuality。
    central = html.split("function invalidateModellingResults", 1)[1].split(
        "function renderModelQualityStatus", 1
    )[0]
    assert central.count("invalidateQuality(") == 1
    assert "invalidateExploration(reason);" in central
    assert "invalidatePreprocessingPreview();" in central
    for marker in (
        'invalidateModellingResults("建模Tag已修改")',
        'invalidateModellingResults("Tag工程配置或角色已修改")',
        'invalidateModellingResults("预处理参数已修改")',
        'invalidateModellingResults("状态过滤条件已修改")',
    ):
        assert marker in html, marker
    # 任一 handler 不再自行组合 exploration + quality 两套失效调用。
    assert 'invalidateExploration("建模Tag已修改")' not in html
    assert 'invalidateExploration("预处理参数已修改")' not in html
    assert 'invalidateExploration("状态过滤条件已修改")' not in html
    # 旧探索结果不能再加入候选窗口。
    convert = html.split('el("convertExplorationCandidates").addEventListener("click", () => {', 1)[1].split(
        "el(\"clusterButton\")", 1
    )[0]
    assert "if(!state.exploration)" in convert
    assert "请重新运行状态探索后再加入候选" in convert

    _run_web_javascript(
        f"""
        const helperSource = {json.dumps(helper_source)};
        const functionSource = {json.dumps(function_source)};
        const elements = new Map([
          ["explorationEmpty", {{hidden:false, textContent:""}}],
          ["explorationContent", {{hidden:false}}],
          ["explorationClusterCandidates", {{replaceChildren(){{this.n=0;}}}}],
          ["explorationPerformanceCandidates", {{replaceChildren(){{this.n=0;}}}}],
          ["explorationPreferredRegionCandidates", {{replaceChildren(){{this.n=0;}}}}],
        ]);
        const el = id => elements.get(id);
        const state = {{
          exploration:{{exploration_run_id:"run-1"}}, explorationRevision:0,
          preferredRegion:{{ellipses:[{{}}]}},
          preferredRegionDrawing:true, preferredRegionRequest:3, preferredRegionUpdateSeq:2,
          candidateWindows:[
            {{id:"manual-1", source_ref:null}},
            {{id:"explore-1", source_ref:"state-exploration-run-1-cluster-1"}},
            {{id:"explore-2", source_ref:"state-exploration-run-1-region-2"}},
            {{id:"explore-3", source_ref:"cluster-manual-1"}},
            {{id:"explore-4", source_ref:"state-exploration-run-1-cluster-7"}},
            {{id:"explore-5", source_ref:"state-exploration-run-1-cluster-9"}},
          ],
          trainingWindows:[
            {{id:"training-keep-1", enabled:true, source_ref:"state-exploration-run-1-cluster-9"}},
            {{id:"training-explore-4-part-001", enabled:true, source_ref:"state-exploration-run-1-cluster-7"}},
          ],
        }};
        function renderCandidateWindows() {{}}
        function renderExplorationRegionControls() {{}}
        function candidateTrainingWindows(candidate) {{
          const baseId=`training-${{candidate.id}}`;
          return state.trainingWindows.filter(window=>window.id===baseId||window.id.startsWith(`${{baseId}}-part-`));
        }}
        function resetExplorationRegion() {{
          state.preferredRegion=null; state.preferredRegionDrawing=false;
          state.preferredRegionRequest+=1; state.preferredRegionUpdateSeq=0;
          renderExplorationRegionControls();
        }}
        eval(helperSource);
        eval(functionSource);

        invalidateExploration("预处理参数已修改");
        if(state.explorationRevision!==1) throw new Error("exploration revision not bumped");
        if(state.exploration!==null) throw new Error("exploration not cleared");
        if(state.preferredRegion!==null||state.preferredRegionDrawing!==false) throw new Error("region not reset");
        if(state.preferredRegionRequest!==4) throw new Error("request counter not bumped");
        if(!el("explorationContent").hidden) throw new Error("content still visible");
        if(!el("explorationEmpty").textContent.includes("预处理参数已修改")) throw new Error("empty hint missing");
        const left=state.candidateWindows.map(item=>item.id);
        if(JSON.stringify(left)!==JSON.stringify(["manual-1","explore-3","explore-4","explore-5"])) throw new Error("wrong candidates left: "+JSON.stringify(left));
        if(state.trainingWindows.length!==2) throw new Error("training windows were removed");
        """
    )


def test_state_filter_change_invalidates_quality_without_any_exploration() -> None:
    html = web_model_results.INDEX_HTML
    central_source = "function invalidateModellingResults" + html.split(
        "function invalidateModellingResults", 1
    )[1].split("function renderModelQualityStatus", 1)[0]
    invalidate_quality_source = "function invalidateQuality" + html.split(
        "function invalidateQuality", 1
    )[1].split("function firstOrderAlphaError", 1)[0]
    invalidate_exploration_source = "function confirmedExplorationSourceRefs" + html.split(
        "function confirmedExplorationSourceRefs", 1
    )[1].split("async function updateExplorationPreferredRegion", 1)[0]

    # 无 exploration 时 invalidateExploration 必须提前返回，不得吞掉 quality 失效。
    assert 'if(!state.exploration&&!state.preferredRegion) return 0;' in invalidate_exploration_source
    assert "state.explorationRevision+=1" in invalidate_exploration_source
    # 集中入口无条件调用 invalidateQuality，且只调用一次。
    assert central_source.count("invalidateQuality(") == 1

    _run_web_javascript(
        f"""
        const centralSource = {json.dumps(central_source)};
        const invalidateExplorationSource = {json.dumps(invalidate_exploration_source)};
        const invalidateQualitySource = {json.dumps(invalidate_quality_source)};
        const elements = new Map([
          ["trainButton", {{disabled:false}}],
          ["trainExploratoryButton", {{disabled:false}}],
          ["qualitySummary", {{innerHTML:""}}],
          ["trainingCompositionReview", {{className:"", textContent:""}}],
          ["qualityIssues", {{className:"", textContent:""}}],
          ["modelTrainingDataSummary", {{className:"", textContent:""}}],
          ["excludeAllConstants", {{disabled:false}}],
          ["modelQualityStatus", {{className:"", textContent:""}}],
          ["currentTagQuality", {{className:"", textContent:""}}],
          ["qualityTagSelect", null],
          ["preprocessingPreview", {{className:"muted", textContent:""}}],
          ["explorationEmpty", {{hidden:false, textContent:""}}],
          ["explorationContent", {{hidden:false}}],
          ["explorationClusterCandidates", {{replaceChildren(){{}}}}],
          ["explorationPerformanceCandidates", {{replaceChildren(){{}}}}],
          ["explorationPreferredRegionCandidates", {{replaceChildren(){{}}}}],
        ]);
        const el = id => elements.get(id);
        const state = {{
          exploration:null, explorationRevision:0, preferredRegion:null, preferredRegionDrawing:false,
          preferredRegionRequest:0, preferredRegionUpdateSeq:0,
          quality:{{tags:[]}}, qualityStatus:"passed", qualityRevision:4, qualityError:"",
          candidateWindows:[], trainingWindows:[],
          preprocessingPreview:null, preprocessingPreviewTag:null, preprocessingPreviewWindowId:null,
        }};
        const statuses = [];
        function renderModelTrainingDataSummary() {{}}
        function renderModelQualityStatus() {{}}
        function renderCurrentTagQuality() {{}}
        function renderCandidateWindows() {{}}
        function renderExplorationRegionControls() {{}}
        function resetExplorationRegion() {{}}
        function invalidatePreprocessingPreview() {{}}
        function setStatus(message) {{ statuses.push(message); }}
        function candidateTrainingWindows() {{ return []; }}
        eval(invalidateQualitySource);
        eval(invalidateExplorationSource);
        eval(centralSource);

        el("trainButton").disabled=false;
        invalidateModellingResults("状态过滤条件已修改");
        if(state.qualityStatus!=="changed") throw new Error("quality must be invalidated: "+state.qualityStatus);
        if(state.quality!==null) throw new Error("stale quality kept");
        if(state.qualityRevision!==5) throw new Error("quality revision must advance exactly once");
        if(!el("trainButton").disabled) throw new Error("trainButton must not reuse old readiness");
        if(!el("trainExploratoryButton").disabled) throw new Error("trainExploratoryButton must be disabled");
        if(state.explorationRevision!==1) throw new Error("stale exploration request must be rejected");
        """
    )


def test_stale_state_exploration_response_cannot_restore_results() -> None:
    html = web_model_results.INDEX_HTML
    handler = html.split('el("stateExplorationButton").addEventListener("click", async () => {', 1)[
        1
    ].split("\n  finally", 1)[0]
    handler_source = "return (async()=>{" + handler + "})();"

    assert "explorationRevision:0" in html
    assert "const explorationRevision=state.explorationRevision;" in handler
    assert "if(explorationRevision!==state.explorationRevision)" in handler
    assert handler.index("const explorationRevision=state.explorationRevision;") < handler.index(
        'api("/api/state-exploration/run"'
    )
    assert handler.index("if(explorationRevision!==state.explorationRevision)") < handler.index(
        "state.exploration=data;"
    )
    assert "renderStateExploration(data)" in handler

    _run_web_javascript(
        f"""
        const handlerSource = {json.dumps(handler_source)};
        const rendered = [];
        const elements = new Map([
          ["stateExplorationButton", {{disabled:false, dataset:{{}}, textContent:"run"}}],
        ]);
        const el = id => elements.get(id);
        const state = {{exploration:null, explorationRevision:0}};
        const setBusy = () => {{}};
        const setStatus = () => {{}};
        const stateExplorationPayload = () => ({{file_id:"file"}});
        const resetExplorationRegion = () => {{}};
        const renderStateExploration = data => rendered.push(data.exploration_run_id);
        const document = {{querySelector: () => ({{click: () => {{}}}})}};
        const AsyncFunction = Object.getPrototypeOf(async function(){{}}).constructor;
        let resolveRun;
        const api = () => new Promise(resolve => {{ resolveRun=resolve; }});
        const invoke = () => new AsyncFunction("api","globalThis","document",handlerSource).call(
          el("stateExplorationButton"), api, globalThis, document
        );

        const run = async () => {{
          // 请求进行中：Lag / Tag / state_filter 变化导致 revision 递增。
          const staleLag=invoke();
          state.explorationRevision+=1;
          resolveRun({{exploration_run_id:"stale-lag", full_point_count:10, returned_point_count:10}});
          await staleLag;
          if(state.exploration!==null) throw new Error("stale Lag response restored exploration");
          if(rendered.length) throw new Error("stale Lag response rendered");

          const staleTag=invoke();
          state.explorationRevision+=1;
          resolveRun({{exploration_run_id:"stale-tag", full_point_count:10, returned_point_count:10}});
          await staleTag;
          if(state.exploration!==null) throw new Error("stale Tag response restored exploration");

          const staleFilter=invoke();
          state.explorationRevision+=1;
          resolveRun({{exploration_run_id:"stale-filter", full_point_count:10, returned_point_count:10}});
          await staleFilter;
          if(state.exploration!==null) throw new Error("stale state_filter response restored exploration");
          if(rendered.length) throw new Error("stale state_filter response rendered");

          // revision 未变：当前请求正常写回并渲染。
          const current=invoke();
          resolveRun({{exploration_run_id:"current", full_point_count:20, returned_point_count:20}});
          await current;
          if(state.exploration?.exploration_run_id!=="current") throw new Error("current response dropped");
          if(JSON.stringify(rendered)!==JSON.stringify(["current"])) throw new Error("current response not rendered");
        }};
        run().then(() => process.stdout.write("exploration-race-ok"), error => {{
          process.stderr.write(String(error.stack||error)); process.exitCode=1;
        }});
        """
    )


def test_exploration_invalidation_keeps_confirmed_candidates_and_training_windows() -> None:
    html = web_model_results.INDEX_HTML
    central_source = "function invalidateModellingResults" + html.split(
        "function invalidateModellingResults", 1
    )[1].split("function renderModelQualityStatus", 1)[0]
    invalidate_exploration_source = "function confirmedExplorationSourceRefs" + html.split(
        "function confirmedExplorationSourceRefs", 1
    )[1].split("async function updateExplorationPreferredRegion", 1)[0]
    invalidate_quality_source = "function invalidateQuality" + html.split(
        "function invalidateQuality", 1
    )[1].split("function firstOrderAlphaError", 1)[0]

    _run_web_javascript(
        f"""
        const centralSource = {json.dumps(central_source)};
        const invalidateExplorationSource = {json.dumps(invalidate_exploration_source)};
        const invalidateQualitySource = {json.dumps(invalidate_quality_source)};
        const candidateRenders = [];
        const elements = new Map([
          ["trainButton", {{disabled:false}}],
          ["trainExploratoryButton", {{disabled:false}}],
          ["qualitySummary", {{innerHTML:""}}],
          ["trainingCompositionReview", {{className:"", textContent:""}}],
          ["qualityIssues", {{className:"", textContent:""}}],
          ["modelTrainingDataSummary", {{className:"", textContent:""}}],
          ["excludeAllConstants", {{disabled:false}}],
          ["modelQualityStatus", {{className:"", textContent:""}}],
          ["currentTagQuality", {{className:"", textContent:""}}],
          ["qualityTagSelect", null],
          ["preprocessingPreview", {{className:"muted", textContent:"已预览"}}],
          ["explorationEmpty", {{hidden:false, textContent:""}}],
          ["explorationContent", {{hidden:false}}],
          ["explorationClusterCandidates", {{replaceChildren(){{}}}}],
          ["explorationPerformanceCandidates", {{replaceChildren(){{}}}}],
          ["explorationPreferredRegionCandidates", {{replaceChildren(){{}}}}],
        ]);
        const el = id => elements.get(id);
        const state = {{
          exploration:{{exploration_run_id:"run-1"}}, explorationRevision:0,
          preferredRegion:null, preferredRegionDrawing:false, preferredRegionRequest:0, preferredRegionUpdateSeq:0,
          quality:{{tags:[]}}, qualityStatus:"passed", qualityRevision:0, qualityError:"",
          candidateWindows:[
            {{id:"confirmed-1", source_ref:"state-exploration-run-1-cluster-1"}},
            {{id:"confirmed-2", source_ref:"state-exploration-run-1-region-2"}},
            {{id:"unconfirmed-1", source_ref:"state-exploration-run-1-cluster-3"}},
            {{id:"manual-1", source_ref:null}},
          ],
          trainingWindows:[
            {{id:"training-confirmed-1", enabled:true, source_ref:"state-exploration-run-1-cluster-1"}},
            {{id:"training-confirmed-2-part-001", enabled:true, source_ref:"state-exploration-run-1-region-2"}},
          ],
          preprocessingPreview:{{data:{{}}, tags:["A"], windowId:"training-confirmed-1", start:"s", end:"e"}},
          preprocessingPreviewTag:"A", preprocessingPreviewWindowId:"training-confirmed-1",
        }};
        function renderModelTrainingDataSummary() {{}}
        function renderModelQualityStatus() {{}}
        function renderCurrentTagQuality() {{}}
        function renderCandidateWindows() {{ candidateRenders.push(state.candidateWindows.length); }}
        function renderExplorationRegionControls() {{}}
        function resetExplorationRegion() {{}}
        function invalidatePreprocessingPreview() {{
          state.preprocessingPreview=null; state.preprocessingPreviewTag=null;
        }}
        function setStatus() {{}}
        function candidateTrainingWindows(candidate) {{
          const baseId=`training-${{candidate.id}}`;
          return state.trainingWindows.filter(window=>window.id===baseId||window.id.startsWith(`${{baseId}}-part-`));
        }}
        eval(invalidateQualitySource);
        eval(invalidateExplorationSource);
        eval(centralSource);

        invalidateModellingResults("预处理参数已修改");
        const left=state.candidateWindows.map(item=>item.id);
        if(JSON.stringify(left)!==JSON.stringify(["confirmed-1","confirmed-2","manual-1"]))
          throw new Error("wrong candidates left: "+JSON.stringify(left));
        if(state.trainingWindows.length!==2) throw new Error("training windows were removed");
        if(candidateRenders.length!==1) throw new Error("candidate table must re-render once");
        if(state.exploration!==null) throw new Error("exploration not cleared");
        if(state.preprocessingPreview!==null) throw new Error("shared preprocessing change must drop preview");
        if(state.qualityStatus!=="changed") throw new Error("quality not invalidated");
        """
    )


def test_preprocessing_preview_survives_quality_check_rerender() -> None:
    html = web_model_results.INDEX_HTML
    # 质量检查成功后只刷新训练窗口统计摘要，不触碰预处理预览状态。
    quality_source = html.split('el("qualityButton").addEventListener("click",async()=>{', 1)[
        1
    ].split('el("qualityTagSelect")', 1)[0]
    render_training_source = "function renderTrainingWindows" + html.split(
        "function renderTrainingWindows", 1
    )[1].split("async function updateTrainingWindows", 1)[0]

    assert "state.trainingWindowSummary=data.training_window_summary||state.trainingWindowSummary;" in quality_source
    assert "state.preprocessingPreview=null" not in quality_source
    assert "renderPreprocessingPreviewWindow();" in render_training_source


def test_state_filter_editor_only_offers_state_filter_tags_and_serializes_bounds() -> None:
    html = web_model_results.INDEX_HTML
    payload_source = "function stateFilterPayload" + html.split("function stateFilterPayload", 1)[1].split(
        "function commonPayload", 1
    )[0]
    common_source = html.split("function commonPayload", 1)[1].split("function candidateId", 1)[0]
    tags_source = "function stateFilterTags" + html.split("function stateFilterTags", 1)[1].split(
        "function addStateFilterCondition", 1
    )[0]

    # commonPayload 把当前条件序列化为 state_filters。
    assert "state_filters:stateFilterPayload()" in common_source
    # 只能选择 role 为 state_filter 的 Tag。
    assert 'state.registry[tag]?.role==="state_filter"' in tags_source
    # 同一 Tag 不重复配置；上下限至少一个；role 改离后阻止使用。
    assert "seen.has(column)" in payload_source
    assert "重复配置" in payload_source
    assert '至少需要下限或上限' in payload_source
    assert 'role!=="state_filter"' in payload_source
    assert "minimum:minimum===\"\"?null:Number(minimum)" in payload_source
    assert "maximum:maximum===\"\"?null:Number(maximum)" in payload_source
    # normal_min/normal_max 仍只用于工程解释，不能被当作状态过滤上下限。
    assert "normalMin" not in payload_source
    assert "normalMax" not in payload_source
    # 状态过滤条件变化同时失效探索与质量检查。
    assert 'remove.addEventListener("click",()=>{ row.remove(); invalidateModellingResults("状态过滤条件已修改"); })' in html
    assert 'invalidateModellingResults("状态过滤条件已修改")' in html


def test_state_filter_payload_builds_and_filters_in_the_browser() -> None:
    html = web_model_results.INDEX_HTML
    payload_source = "function stateFilterPayload" + html.split("function stateFilterPayload", 1)[1].split(
        "function commonPayload", 1
    )[0]

    _run_web_javascript(
        """
        const payloadSource = %s;
        const registry = {MODE:{role:"state_filter"}, LOAD:{role:"state_filter"}, TEMP:{role:"continuous_input"}};
        const state = {registry};
        const field = value => ({value});
        const row = (column, minimum, maximum) => ({
          querySelector: selector =>
            selector.includes("column") ? field(column)
            : selector.includes("minimum") ? field(minimum)
            : field(maximum),
        });
        const scenarios = [];
        const run = rows => {
          globalThis.document = {querySelectorAll: () => rows};
          eval(payloadSource);
          return stateFilterPayload();
        };
        scenarios.push(run([row("MODE","10","")]));
        scenarios.push(run([row("MODE","","20")]));
        scenarios.push(run([row("MODE","10","20"), row("LOAD","5","6")]));
        const errorOf = (rows, reg) => {
          state.registry = reg || registry;
          try { run(rows); return ""; } catch(error) { return error.message; }
        };
        const empty = errorOf([row("MODE","","")]);
        const duplicated = errorOf([row("MODE","1","2"), row("MODE","3","4")]);
        const roleChanged = errorOf([row("TEMP","1","2")]);
        const expected = [
          [{column:"MODE", minimum:10, maximum:null}],
          [{column:"MODE", minimum:null, maximum:20}],
          [{column:"MODE", minimum:10, maximum:20}, {column:"LOAD", minimum:5, maximum:6}],
        ];
        if(JSON.stringify(scenarios)!==JSON.stringify(expected)) throw new Error(JSON.stringify(scenarios));
        if(!empty.includes("至少需要下限或上限")) throw new Error("empty bounds not rejected: "+empty);
        if(!duplicated.includes("重复配置")) throw new Error("duplicate not rejected: "+duplicated);
        if(!roleChanged.includes("状态过滤")) throw new Error("role change not rejected: "+roleChanged);
        """
        % json.dumps(payload_source)
    )


def test_state_filter_tags_never_enter_continuous_model_tags() -> None:
    html = web_model_results.INDEX_HTML
    selected_source = html.split("function selectedTags()", 1)[1].split("function numberValue", 1)[0]

    assert '(state.registry[tag]?.role||"continuous_input")==="continuous_input"' in selected_source


def test_state_exploration_performance_controls_share_one_row() -> None:
    for html in (web_model_results.INDEX_HTML,):
        row = html.split('<div class="exploration-controls performance-controls">', 1)[1].split(
            "</div>", 1
        )[0]
        for element_id in (
            'id="explorationPerformanceTag"',
            'id="explorationPerformanceDirection"',
            'id="explorationTargetMin"',
            'id="explorationTargetMax"',
            'id="explorationPerformanceMinimumDuration"',
            'id="explorationPerformanceCandidateCount"',
        ):
            assert element_id in row
        assert row.count("<label>") == 6
        assert ".exploration-controls.performance-controls { grid-template-columns:repeat(6,minmax(0,1fr)); }" in html
        assert "@media (max-width:1050px) { .exploration-controls.performance-controls { grid-template-columns:repeat(3,minmax(0,1fr)); } }" in html
        # 标签换行时输入框仍贴底对齐，避免同一行控件高低不齐。
        assert ".exploration-controls > label { display:grid; align-content:start; }" in html


def test_workbench_tables_map_all_runtime_statuses_and_align_numeric_cells() -> None:
    html = web_model_results.INDEX_HTML

    assert ".table-wrap td.numeric" in html
    assert '#stateExplorationPanel .exploration-controls { border:0; }' in html
    assert "#explorationRegionSummary td:nth-child(2), #explorationRegionSummary td:nth-child(3) { text-align:center; }" in html
    # 状态探索结果表统一居中；.numeric 的右对齐在本面板内被覆盖。
    assert "#stateExplorationPanel table th, #stateExplorationPanel table td { text-align:center; }" in html
    assert html.index("#stateExplorationPanel table th") > html.index(".table-wrap td.numeric")
    comment_rule = html.split(".exploration-candidate-comment {", 1)[1].split("}", 1)[0]
    assert "text-align:center;" in comment_rule
    # 居中规则只覆盖状态探索面板内的表格，其他阶段表格不受影响。
    panel_start = html.index('<div id="stateExplorationPanel"')
    panel_end = html.index('<div id="clusterPanel"', panel_start)
    panel = html[panel_start:panel_end]
    for element_id in (
        'id="explorationClusterTable"',
        'id="explorationClusterCandidates"',
        'id="explorationPerformanceCandidates"',
        'id="explorationPreferredRegionCandidates"',
    ):
        assert element_id in panel
    for outside in ('id="candidateWindows"', 'id="trainingWindows"', 'id="clusterTable"'):
        assert html.index(outside) < panel_start or html.index(outside) > panel_end
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
