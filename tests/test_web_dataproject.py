from __future__ import annotations

import pandas as pd
import pytest

from pca_model_builder import web_dataproject


def test_dataproject_trend_layout_is_injected_without_removing_legacy_controls() -> None:
    html = web_dataproject.INDEX_HTML

    assert 'id="dataprojectTrendStyle"' in html
    assert 'id="dataprojectTrendScript"' in html
    assert 'id="legacyTrendPanel"' in html
    assert 'id="dpTrendVar1"' in html
    assert 'id="dpTrendVar4"' in html
    assert 'id="dpTrendAxisMode"' in html
    assert 'id="dpTrendMaxPoints"' in html
    assert 'id="dpTrendStats"' in html
    assert 'id="dpScatterX1"' in html
    assert 'id="dpScatterY3"' in html
    assert 'id="dpScatterChart"' in html
    assert "物理时间缺口不会连线" in html
    assert "页面不会插值、补点或修改原始数据" in html

    # Existing IDs remain in the hidden legacy container so the current
    # quality/contribution jump handlers do not break.
    assert 'id="trendTags"' in html
    assert 'id="trendStart"' in html
    assert 'id="trendEnd"' in html


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
    monkeypatch.setattr(web_dataproject.base_web, "_read_upload", lambda payload: frame)

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
    monkeypatch.setattr(web_dataproject.base_web, "_read_upload", lambda payload: frame)
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
