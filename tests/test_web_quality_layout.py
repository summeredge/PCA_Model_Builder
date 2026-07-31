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


def test_supported_web_entrypoints_use_model_results_page() -> None:
    start_app = (PROJECT_ROOT / "start_app.bat").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "pca_model_builder.web_model_results" in start_app
    assert 'pca-model-builder = "pca_model_builder.cli_entry:main"' in pyproject
    assert (
        'pca-model-builder-web = "pca_model_builder.web_model_results:main"'
        in pyproject
    )


def test_cli_entry_restores_original_serve_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cli_entry.cli._serve
    monkeypatch.setattr(cli_entry.cli, "main", lambda argv=None: 0)

    assert cli_entry.main(["train"]) == 0
    assert cli_entry.cli._serve is original
