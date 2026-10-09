from __future__ import annotations

import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import (
    cli,
    web,
    web_dataproject,
    web_model_results,
    web_quality_layout,
)


def _loaded(frame: pd.DataFrame) -> SimpleNamespace:
    timestamp_column = str(frame.columns[0])
    return SimpleNamespace(
        frame=frame.copy(deep=True),
        metadata=SimpleNamespace(
            row_count=len(frame),
            numeric_candidate_columns=tuple(
                str(column) for column in frame.columns if column != timestamp_column
            ),
        ),
        loaded_column_count=len(frame.columns),
        cache_hit=False,
    )


def test_dataproject_trend_layout_is_injected_without_removing_legacy_controls() -> None:
    html = web_dataproject.INDEX_HTML

    assert 'id="dataprojectTrendStyle"' in html
    assert 'id="dataprojectTrendScript"' in html
    assert 'legacy.id = "legacyTrendPanel"' in html
    assert 'id="dpTrendVar1"' in html
    assert 'id="dpTrendVar4"' in html
    assert 'id="dpTrendAxisMode"' in html
    assert '<option value="independent" selected>独立 Y 轴</option>' in html
    assert 'id="dpTrendMaxPoints"' in html
    assert 'id="dpTrendStats"' in html
    assert 'id="dpScatterX1"' in html
    assert 'id="dpScatterY3"' in html
    assert 'id="dpScatterChart"' in html
    assert "物理时间缺口不会连线" in html
    assert "页面不会插值、补点或修改原始数据" in html
    assert "真实原始趋势" in html

    # Existing IDs remain in the hidden legacy container so the current
    # quality/contribution jump handlers do not break.
    assert 'id="trendTags"' in html
    assert 'id="trendStart"' in html
    assert 'id="trendEnd"' in html
    assert 'addCandidateWindow("trend", $("dpTrendStart").value, $("dpTrendEnd").value, "trend-current", "")' in html
    assert "normalStart" not in html
    assert "normalEnd" not in html


def test_trend_bar_inputs_cannot_overflow_their_labels() -> None:
    html = web_dataproject.INDEX_HTML

    # datetime-local 的固有宽度超过 label 的 flex 宽度时，min-width:auto 会让输入框
    # 盖住相邻输入框；必须显式 min-width:0 才能收缩。
    assert ".dp-trend-bar input { min-width:0; }" in html


def test_tag_statistics_keep_multivalue_groups_without_clipping_values() -> None:
    html = web_dataproject.INDEX_HTML
    assert '.dp-trend-stat-card dl { display:grid; gap:2px; margin:0; }' in html
    assert 'grid-template-columns:82px minmax(0,1fr);' in html
    assert '.dp-trend-stat-card dd {' in html
    card = html.split('.dp-trend-stat-card {', 1)[1].split('}', 1)[0]
    assert 'padding:8px;' in card and 'overflow:hidden' not in card
    assert '<dl>${rows.map(' in html


def test_trend_to_analysis_invalidates_only_when_exploration_range_changes() -> None:
    html = web_dataproject.INDEX_HTML
    handler = html.split(
        '$("dpTrendToAnalysis").addEventListener("click", () => {', 1
    )[1].split("\n  });", 1)[0]
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the DataProject range state test")

    result = subprocess.run(
        [
            node,
            "-e",
            f"""
            const handler = {json.dumps(handler)};
            function scenario(values, expectedInvalidations) {{
              const elements = {{
                analysisStart:{{value:values.analysisStart}}, analysisEnd:{{value:values.analysisEnd}},
                dpTrendStart:{{value:values.dpTrendStart}}, dpTrendEnd:{{value:values.dpTrendEnd}},
              }};
              if(values.explorationStart!==undefined) elements.explorationStart={{value:values.explorationStart}};
              if(values.explorationEnd!==undefined) elements.explorationEnd={{value:values.explorationEnd}};
              const $ = id => elements[id] || null;
              const invalidations = [];
              new Function("$","setStatus","invalidateModellingResults",handler)(
                $, () => {{}}, reason => invalidations.push(reason)
              );
              if(elements.analysisStart.value!==values.dpTrendStart||elements.analysisEnd.value!==values.dpTrendEnd)
                throw new Error("analysis range copy changed");
              if(elements.explorationStart&&elements.explorationStart.value!==values.dpTrendStart)
                throw new Error("independent exploration start was not copied");
              if(elements.explorationEnd&&elements.explorationEnd.value!==values.dpTrendEnd)
                throw new Error("independent exploration end was not copied");
              if(invalidations.length!==expectedInvalidations)
                throw new Error("expected "+expectedInvalidations+" invalidations, got "+invalidations.length);
              if(invalidations.some(reason=>reason!=="状态探索参数已修改"))
                throw new Error("unexpected invalidation reason");
            }}
            scenario({{
              analysisStart:"analysis-old", analysisEnd:"analysis-old-end",
              explorationStart:"same-start", explorationEnd:"same-end",
              dpTrendStart:"same-start", dpTrendEnd:"same-end",
            }},0);
            scenario({{
              analysisStart:"analysis-old", analysisEnd:"analysis-old-end",
              explorationStart:"old-start", explorationEnd:"old-end",
              dpTrendStart:"new-start", dpTrendEnd:"new-end",
            }},1);
            scenario({{
              analysisStart:"same-start", analysisEnd:"same-end",
              dpTrendStart:"same-start", dpTrendEnd:"same-end",
            }},0);
            scenario({{
              analysisStart:"old-start", analysisEnd:"old-end",
              dpTrendStart:"new-start", dpTrendEnd:"new-end",
            }},1);
            """,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_missing_values_are_not_converted_to_zero_by_frontend_contract() -> None:
    html = web_dataproject.INDEX_HTML

    assert "function finiteNumber(value)" in html
    assert 'value === null || value === undefined || value === ""' in html
    assert not re.search(r"(?<![A-Za-z])Number\(point\.y\)", html)
    assert not re.search(r"(?<![A-Za-z])Number\(row\[`\$\{xTag\}__raw`\]\)", html)
    assert not re.search(r"(?<![A-Za-z])Number\(row\[`\$\{yTag\}__raw`\]\)", html)


def test_trend_selection_fill_is_consistent_before_and_after_drag() -> None:
    html = web_dataproject.INDEX_HTML

    assert 'fillcolor: "rgba(23,107,135,0.18)"' in html
    assert 'function selectionShape(start, end)' in html
    assert 'xref: "x"' in html
    assert 'yref: "paper"' in html


def test_cli_serve_uses_final_model_results_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        web_model_results,
        "run_server",
        lambda host, port, open_browser: calls.append((host, port, open_browser)),
    )

    result = cli._serve(SimpleNamespace(host="127.0.0.1", port=8775, no_open=True))

    assert result == {"status": "stopped"}
    assert calls == [("127.0.0.1", 8775, False)]


def test_legacy_web_server_functions_use_final_model_results_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        web_model_results,
        "run_server",
        lambda host, port, open_browser: calls.append((host, port, open_browser)),
    )

    for run_server in (web.run_server, web_dataproject.run_server, web_quality_layout.run_server):
        run_server("127.0.0.1", 8775, open_browser=False)

    assert calls == [("127.0.0.1", 8775, False)] * 3


def test_dataproject_trend_payload_preserves_physical_gap_and_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "TIME": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:05",
                    "2026-01-01 00:20",
                ]
            ),
            "A": [10.0, 11.0, 12.0],
            "B": [20.0, 19.0, 18.0],
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )

    result = web_dataproject.trend_payload(
        {
            "purpose": "trend",
            "file_id": "ignored-by-test",
            "timestamp_column": "TIME",
            "encoding": "utf-8-sig",
            "tags": ["A", "B"],
            "tag_configs": {},
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 60,
            "lag_step_minutes": 5,
            "start": "2026-01-01 00:00",
            "end": "2026-01-01 00:20",
            "normal_start": "2026-01-01 00:00",
            "normal_end": "2026-01-01 00:05",
            "max_points": 100,
        }
    )

    assert result["raw_rows"] == 3
    assert result["rows_count"] == 3
    assert result["max_points"] == 100
    assert result["rows"][2]["gap_start"] is True
    assert [point["y"] for point in result["series"][0]["points"]] == [10.0, 11.0, 12.0]
    assert result["statistics"]["A"]["current"]["mean"] == pytest.approx(11.0)
    assert result["statistics"]["A"]["reference"]["sample_count"] == 2
    assert sum(result["histograms"]["A"]["counts"]) == 3
    assert result["data_usage"] == {
        "source_row_count": 3,
        "analysis_row_count": 3,
        "display_point_count": 3,
        "loaded_column_count": 3,
        "cache_hit": False,
        "stage": "completed",
    }


def test_dataproject_trend_payload_keeps_missing_as_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "TIME": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "A": [10.0, None, 12.0],
            "B": [20.0, 19.0, 18.0],
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )

    result = web_dataproject.trend_payload(
        {
            "purpose": "trend",
            "file_id": "ignored-by-test",
            "timestamp_column": "TIME",
            "tags": ["A", "B"],
            "tag_configs": {},
            "sample_interval_minutes": 5,
            "smoothing_window_minutes": 10,
            "max_lag_minutes": 60,
            "lag_step_minutes": 5,
            "start": "2026-01-01 00:00",
            "end": "2026-01-01 00:10",
            "max_points": 100,
        }
    )

    assert [point["y"] for point in result["series"][0]["points"]] == [10.0, None, 12.0]
    assert result["statistics"]["A"]["current"]["sample_count"] == 3
    assert result["statistics"]["A"]["current"]["valid_count"] == 2
    assert result["statistics"]["A"]["current"]["missing_count"] == 1
    assert sum(result["histograms"]["A"]["counts"]) == 2


def test_trend_and_scatter_tag_limits_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags = [f"T{index}" for index in range(1, 7)]
    frame = pd.DataFrame(
        {
            "TIME": pd.date_range("2026-01-01", periods=3, freq="5min"),
            **{tag: [1.0, 2.0, 3.0] for tag in tags},
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )
    common = {
        "file_id": "ignored-by-test",
        "timestamp_column": "TIME",
        "encoding": "utf-8-sig",
        "tag_configs": {},
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 60,
        "lag_step_minutes": 5,
        "start": "2026-01-01 00:00",
        "end": "2026-01-01 00:10",
        "max_points": 100,
    }

    with pytest.raises(ValueError, match="趋势图一次最多选择4个Tag"):
        web_dataproject.trend_payload({**common, "purpose": "trend", "tags": tags[:5]})

    result = web_dataproject.trend_payload(
        {**common, "purpose": "scatter", "tags": tags}
    )
    assert result["tags"] == tags


def test_old_trend_payload_path_is_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"legacy": True}
    monkeypatch.setattr(
        web_dataproject,
        "_BASE_TREND_PAYLOAD",
        lambda payload: sentinel,
    )

    assert web_dataproject.trend_payload({"tags": ["A"]}) is sentinel


def _trend_request(frame: pd.DataFrame, **overrides: object) -> dict:
    payload = {
        "purpose": "trend",
        "file_id": "ignored-by-test",
        "timestamp_column": "TIME",
        "tags": ["A"],
        "tag_configs": {},
        "sample_interval_minutes": 1,
        "smoothing_window_minutes": 0,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "start": frame.TIME.iloc[0].isoformat(),
        "end": frame.TIME.iloc[-1].isoformat(),
    }
    payload.update(overrides)
    return payload


def test_trend_default_max_points_is_100000_and_keeps_full_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 40000
    frame = pd.DataFrame(
        {
            "TIME": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "A": np.arange(row_count, dtype=float),
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )

    result = web_dataproject.trend_payload(_trend_request(frame))

    assert result["max_points"] == 100000
    # A window below the 100000 default must be shown in full, not downsampled.
    assert result["raw_rows"] == row_count
    assert result["rows_count"] == row_count
    assert len(result["series"][0]["points"]) == row_count


def test_trend_max_points_fallback_still_uses_downsample_for_oversized_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 2500
    frame = pd.DataFrame(
        {
            "TIME": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "A": np.arange(row_count, dtype=float),
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )

    result = web_dataproject.trend_payload(_trend_request(frame, max_points=200))

    assert result["max_points"] == 200
    assert result["raw_rows"] == row_count
    assert result["rows_count"] <= 200
    # Statistics stay based on the full current window, not the downsampled view.
    assert result["statistics"]["A"]["current"]["sample_count"] == row_count


def test_dataproject_trend_does_not_label_resampled_values_as_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "TIME": pd.date_range("2026-01-01", periods=11, freq="1min"),
            "A": range(11),
        }
    )
    monkeypatch.setattr(
        web_dataproject.base_web,
        "_load_required_upload",
        lambda payload, columns, prefix: _loaded(frame.loc[:, ["TIME", *columns]]),
    )

    result = web_dataproject.trend_payload(
        {
            "purpose": "trend",
            "file_id": "ignored-by-test",
            "timestamp_column": "TIME",
            "tags": ["A"],
            "sample_interval_minutes": 5,
            "resampling_method": "mean",
            "filter_method": "none",
            "smoothing_window_minutes": 0,
            "max_lag_minutes": 0,
            "lag_step_minutes": 5,
            "start": frame.TIME.iloc[0].isoformat(),
            "end": frame.TIME.iloc[-1].isoformat(),
            "max_points": 100,
        }
    )

    assert result["raw_rows"] == 11
    assert [point["y"] for point in result["series"][0]["points"]] == list(range(11))
