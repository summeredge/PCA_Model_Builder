import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from pca_model_builder.cluster_quality import analyze_cluster_quality
from pca_model_builder.clustering import cluster_model_scores, cluster_operating_states
from pca_model_builder.dpca import fit_dpca
from pca_model_builder import state_exploration, web, web_model_results


def test_quality_scores_centers_time_and_standardized_tag_ranking():
    index = pd.date_range("2026-01-01", periods=6, freq="1h")
    scores = pd.DataFrame({"pc1": [-5, -4, 4, 5, -5, -4], "pc2": [0, 0, 0, 0, 0, 0]}, index=index)
    labels = [1, 1, 2, 2, 1, 1]
    raw = pd.DataFrame({"driver": [0, 0, 10, 10, 0, 0], "large_scale": [100, 200, 200, 300, 100, 200], "constant": 1, "time": index, "excluded": labels}, index=index)
    quality = analyze_cluster_quality(scores, labels, index, raw, ["large_scale", "driver", "constant", "time"], [0.68, 0.14], 60)
    assert quality["cluster_count"] == 2
    assert quality["silhouette_score"] == pytest.approx(silhouette_score(scores, labels))
    assert quality["explained_variance"] == {"pc1": 0.68, "pc2": 0.14}
    assert quality["centers"] == [{"cluster": "1", "pc1": -4.5, "pc2": 0.0}, {"cluster": "2", "pc1": 4.5, "pc2": 0.0}]
    assert quality["center_orientation"] == "主要沿 PC1"
    assert quality["temporal_metrics"] == {"average_duration_hours": 2, "longest_duration_hours": 2, "state_switch_count": 2, "run_count": 3, "duration_semantics": "coverage"}
    assert [item["tag"] for item in quality["top_features"]] == ["driver", "large_scale"]
    assert quality["top_features"][0]["standardized_difference"] == pytest.approx(10 / raw.driver.std(ddof=0))
    assert "连续运行变量变化" in quality["engineering_hint"]["state_exploration"]
    assert "离散运行状态" in quality["engineering_hint"]["cluster_assistance"]
    json.dumps(quality, allow_nan=False)


@pytest.mark.parametrize("centers,orientation", [
    ([[0, -5], [0, 5]], "主要沿 PC2"),
    ([[-5, -5], [5, 5]], "二维分布明显"),
    ([[0, 0], [0, 0]], "无明显方向"),
])
def test_center_orientation(centers, orientation):
    quality = analyze_cluster_quality(pd.DataFrame(np.repeat(centers, 2, axis=0)), [1, 1, 2, 2])
    assert quality["center_orientation"] == orientation


@pytest.mark.parametrize("labels,reason", [([1, 1, 1], "2–10"), ([1, 2, 3], "样本不足"), (list(range(11)), "2–10")])
def test_invalid_cluster_count_keeps_centers_but_explains_unavailable_silhouette(labels, reason):
    quality = analyze_cluster_quality(pd.DataFrame(np.zeros((len(labels), 2))), labels)
    assert quality["cluster_count"] == len(set(labels))
    assert quality["silhouette_score"] is None
    assert reason in quality["unavailable_reasons"]["silhouette_score"]
    assert "没有时间列" in quality["unavailable_reasons"]["temporal_metrics"]
    assert "有效建模 Tag" in quality["unavailable_reasons"]["top_features"]


@pytest.mark.parametrize("scores,labels", [(pd.DataFrame(columns=["pc1", "pc2"]), []), (pd.DataFrame([[0, 0]]), []), (pd.DataFrame([[0, 0]]), [None]), (pd.DataFrame([[np.nan, 0]]), [1])])
def test_empty_or_invalid_input_has_reason(scores, labels):
    quality = analyze_cluster_quality(scores, labels)
    assert quality["unavailable_reasons"]["analysis"]
    json.dumps(quality, allow_nan=False)


def test_time_gaps_are_not_stable_duration_or_observed_switches():
    index = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-02 00:00", "2026-01-02 01:00"])
    scores = pd.DataFrame([[0, 0], [0, 0], [5, 0], [5, 0]], index=index)
    quality = analyze_cluster_quality(scores, [1, 1, 2, 2], index, sample_interval_minutes=60)
    assert quality["temporal_metrics"]["average_duration_hours"] == 2
    assert quality["temporal_metrics"]["state_switch_count"] == 0
    for timestamps in ([None] * 4, index[::-1], [index[0]] * 4):
        assert analyze_cluster_quality(scores, [1, 1, 2, 2], timestamps)["temporal_metrics"] is None


def test_top_five_tags_ignore_invalid_and_non_model_columns_and_align_by_index():
    index = pd.RangeIndex(8)
    scores = pd.DataFrame({"pc1": [-1] * 4 + [1] * 4, "pc2": 0}, index=index)
    raw = pd.DataFrame({**{f"T{i}": np.arange(8) + np.array([0] * 4 + [i] * 4) for i in range(7)}, "bad": np.inf, "text": "bad", "ignored": [0] * 4 + [100] * 4}, index=index).iloc[::-1]
    quality = analyze_cluster_quality(scores, [1] * 4 + [2] * 4, raw_data=raw, feature_names=[*[f"T{i}" for i in range(7)], "bad", "text", "missing"])
    assert [item["tag"] for item in quality["top_features"]] == ["T6", "T5", "T4", "T3", "T2"]
    assert len(quality["feature_contrasts"]) == 10
    assert [item["tag"] for item in quality["feature_contrasts"][:7]] == [f"T{i}" for i in range(6, -1, -1)]
    assert all(item["unavailable_reason"] for item in quality["feature_contrasts"][7:])
    json.dumps(quality, allow_nan=False)


def test_nullable_tags_with_no_valid_values_have_reason():
    scores = pd.DataFrame([[0, 0], [0, 0], [1, 0], [1, 0]])
    raw = pd.DataFrame({"empty": pd.Series([pd.NA] * 4, dtype="Float64"), "text": pd.Series(["bad"] * 4, dtype="string")})
    quality = analyze_cluster_quality(scores, [1, 1, 2, 2], raw_data=raw, feature_names=raw.columns)
    assert quality["top_features"] == []
    assert "有效建模 Tag" in quality["unavailable_reasons"]["top_features"]


def test_large_silhouette_is_bounded_reproducible_and_keeps_rare_cluster():
    scores = pd.DataFrame(np.random.default_rng(1).normal(size=(2200, 3)))
    labels = np.r_[np.ones(1100), np.full(1099, 2), 3]
    first = analyze_cluster_quality(scores, labels)
    second = analyze_cluster_quality(scores, labels)
    assert first["silhouette_approximate"]
    assert first["silhouette_sample_count"] <= 2003
    assert first["silhouette_score"] == second["silhouette_score"]


def test_both_cluster_paths_preserve_pca_coordinates_and_ratios():
    dynamic = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 3)), index=pd.date_range("2026-01-01", periods=30, freq="5min"), columns=["A", "B", "C"])
    model = fit_dpca(dynamic, n_components=2)
    for result in (cluster_operating_states(dynamic, 2), cluster_model_scores(model, dynamic, 2)):
        response = web._cluster_result_payload(result, raw_data=dynamic, feature_names=dynamic.columns)
        quality = response["cluster_quality"]
        for center in quality["centers"]:
            np.testing.assert_allclose([center["pc1"], center["pc2"]], result.centers[int(center["cluster"])][:2])
        assert quality["explained_variance"]["pc1"] == result.explained_variance_ratio[0]
        assert quality["cluster_count"] == len(response["clusters"])
    assert state_exploration.analyze_cluster_quality is web.analyze_cluster_quality is analyze_cluster_quality


def test_quality_cards_are_inside_existing_result_panels_and_share_renderer():
    for html in (web.INDEX_HTML, web_model_results.INDEX_HTML):
        assert html.count('id="explorationClusterQuality"') == 1
        assert html.count('id="assistanceClusterQuality"') == 1
        assert html.index('id="explorationClusterQuality"') < html.index('id="explorationPcChart"')
        assert html.index('id="assistanceClusterQuality"') < html.index('id="clusterChart"')
        centers_argument = ",data.cluster_centers" if html is web_model_results.INDEX_HTML else ""
        assert f'renderClusterQuality(el("explorationClusterQuality"),data.cluster_quality,"state_exploration"{centers_argument})' in html
        assert 'renderClusterQuality(el("assistanceClusterQuality"),data.cluster_quality,"cluster_assistance")' in html
        assert "状态探索工程提示" in html and "状态结构解释" in html
        for element_id in ("explorationTimeline", "explorationClusterTable", "explorationClusterCandidates", "explorationPerformanceCandidates", "explorationPreferredRegionCandidates"):
            assert f'id="{element_id}"' in html
    state_renderer = web.INDEX_HTML.split("function renderStateExploration(data)", 1)[1].split("function renderExplorationLossSummary", 1)[0]
    for renderer in ("renderExplorationPcChart", "renderExplorationTimeline", "renderExplorationClusterTable", "renderExplorationCandidateTables"):
        assert f"{renderer}(" in state_renderer


def test_quality_renderer_shows_perspectives_missing_reasons_and_escapes_tags():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    renderer = "function renderClusterQuality" + web.INDEX_HTML.split("function renderClusterQuality", 1)[1].split("function renderStateExploration", 1)[0]
    renderer = "function clusterUiLabel" + web.INDEX_HTML.split("function clusterUiLabel", 1)[1].split("\n", 1)[0] + "\n" + renderer
    scores = pd.DataFrame({"pc1": [-2, -1, 1, 2], "pc2": 0})
    quality = analyze_cluster_quality(scores, [1, 1, 2, 2], raw_data=pd.DataFrame({"<driver>": [0, 0, 1, 1]}), feature_names=["<driver>"], explained_variance_ratio=[0.8, 0.2])
    source = """
    const escapeHtml=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
    const metric=(name,value)=>`<span>${escapeHtml(name)} ${escapeHtml(value)}</span>`;
    """ + renderer + f"const quality={json.dumps(quality, ensure_ascii=False)};" + """
    const container={innerHTML:''};
    renderClusterQuality(container,quality,'state_exploration'); const exploration=container.innerHTML;
    renderClusterQuality(container,quality,'cluster_assistance'); const assistance=container.innerHTML;
    const degraded=JSON.parse(JSON.stringify(quality));
    const clusterIds=Object.keys(degraded.top_features[0].cluster_means);
    degraded.top_features[0].cluster_means[clusterIds[0]]=NaN;
    delete degraded.top_features[0].cluster_means[clusterIds[1]];
    renderClusterQuality(container,degraded,'cluster_assistance'); const degradedMeans=container.innerHTML;
    const legacy=JSON.parse(JSON.stringify(quality));
    delete legacy.top_features[0].cluster_means;
    renderClusterQuality(container,legacy,'cluster_assistance'); const unavailableMeans=container.innerHTML;
    renderClusterQuality(container,null,'state_exploration'); const missing=container.innerHTML;
    renderClusterQuality(container,{unavailable_reasons:{analysis:'样本不足'}},'cluster_assistance');
    console.log(JSON.stringify({exploration,assistance,degradedMeans,unavailableMeans,missing,invalid:container.innerHTML}));
    """
    output = subprocess.run([node, "-e", source], capture_output=True, text=True, encoding="utf-8", check=True)
    rendered = json.loads(output.stdout)
    assert "状态探索工程提示" in rendered["exploration"]
    assert "状态结构解释" in rendered["assistance"]
    for perspective in ("exploration", "assistance"):
        assert "聚类质量摘要" in rendered[perspective]
        assert "没有时间列" in rendered[perspective]
    assert "主要区分变量" not in rendered["exploration"]  # merged into variable diagnostics
    assert "&lt;driver&gt;" in rendered["assistance"]
    assert "<driver>" not in rendered["assistance"]
    assert "<th>标准化差异</th><th>原始均值差</th><th>工况组 1</th><th>工况组 2</th>" in rendered["assistance"]
    assert "<tr><td>&lt;driver&gt;</td><td>2.000</td><td>1.000</td><td>0.000</td><td>1.000</td></tr>" in rendered["assistance"]
    assert "Top5" in rendered["assistance"]
    assert "<td>—</td><td>—</td>" in rendered["degradedMeans"]
    assert "<td>—</td><td>—</td>" in rendered["unavailableMeans"]
    assert "尚无聚类分析数据" in rendered["missing"]
    assert "样本不足" in rendered["invalid"]


def test_state_exploration_quality_uses_full_samples_and_existing_centers():
    from pca_model_builder.preprocessing import PreprocessingConfig
    index = pd.date_range("2026-01-01", periods=40, freq="5min")
    values = np.r_[np.linspace(-3, -2, 20), np.linspace(2, 3, 20)]
    frame = pd.DataFrame({"A": values, "B": values**2, "C": np.sin(values)}, index=index)
    result = state_exploration.run_state_exploration(
        frame, ["A", "B", "C"], PreprocessingConfig(5, 0, 0, 5, filter_method="none"),
        state_exploration.ExplorationConfig(cluster_count=2, maximum_plot_points=4),
    )
    quality = result["cluster_quality"]
    points = result["cluster_series"]
    assert quality["silhouette_sample_count"] == len(points) > len(result["cluster_series_display"])
    pc_columns = result["exploratory_model_summary"]["pc_columns"]
    assert quality["silhouette_score"] == pytest.approx(silhouette_score(points[pc_columns], points.cluster_id))
    for center in quality["centers"]:
        np.testing.assert_allclose([center["pc1"], center["pc2"]], result["cluster_centers"][center["cluster"]][:2])
    assert {item["tag"] for item in quality["top_features"]} <= {"A", "B", "C"}
    assert all(item["decision"] == "pending" for item in result["candidate_decisions"])


@pytest.mark.parametrize("variances,columns,orientation,coverage", [
    ([8, 2, 1, 1], [0, 1], "多主元分布明显", 10 / 12),
    ([4, 0, 1], [0, 1], "主要沿 PC1", .8),
    ([1, 0, 9, 0], [0, 1, 2], "主要沿 PC3", 1),
    ([2, 2, 5, 1], [0, 1, 2], "多主元分布明显", .9),
    ([1, 1, 4, 4], [0, 1, 2, 3], "多主元分布明显", 1),
    ([1, 1, 0, 9], [0, 1, 3], "主要沿 PC4", 1),
    ([1, 1, 2, 6], [0, 1, 3], "多主元分布明显", .8),
    ([1, 1, 2, 3, 93], [0, 1, 2, 3], "主要沿 PC5", .07),
    ([0, 0, 0, 0], [0, 1], "无明显方向", None),
    ([1e-16, 0, 9e-16], [0, 1, 2], "主要沿 PC3", 1),
])
def test_screening_centers_choose_columns_from_center_variance_and_keep_time_values(
    variances, columns, orientation, coverage,
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    coordinates = np.array([-np.sqrt(variances), np.sqrt(variances)])
    coordinates += np.arange(len(variances)) * 3 + 10
    index = pd.date_range("2026-01-01", periods=4, freq="1h")
    scores = pd.DataFrame(np.repeat(coordinates, 2, axis=0), index=index)
    quality = analyze_cluster_quality(scores, [1, 1, 2, 2], index,
                                      explained_variance_ratio=[.95, .03], sample_interval_minutes=60)
    centers = {"2": coordinates[1].tolist(), "1": coordinates[0].tolist()}
    html = web_model_results.INDEX_HTML
    renderer = "function screeningCenterView" + html.split("function screeningCenterView", 1)[1].split("function renderVariableDiagnostics", 1)[0]
    legacy = "function renderClusterQuality" + web.INDEX_HTML.split("function renderClusterQuality", 1)[1].split("function renderVariableDiagnostics", 1)[0]
    legacy = legacy.replace("function renderClusterQuality", "function renderLegacyClusterQuality", 1)
    labels = "function clusterUiLabel" + html.split("function clusterUiLabel", 1)[1].split("\n", 1)[0]
    source = r"""
const assert=require('node:assert/strict');
const escapeHtml=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const metric=(name,value)=>`<span>${escapeHtml(name)} ${escapeHtml(value)}</span>`;
const details={innerHTML:'',replaceChildren(){this.innerHTML='';}},el=()=>details;
""" + labels + renderer + legacy + f"const quality={json.dumps(quality, ensure_ascii=False)}, centers={json.dumps(centers)};" + r"""
const before=JSON.stringify([quality,centers]),container={innerHTML:''};
const view=screeningCenterView(quality.centers,centers);
renderClusterQuality(container,quality,'state_exploration',centers);
const result={view,details:details.innerHTML,summary:container.innerHTML};
renderClusterQuality(container,quality,'state_exploration');
result.fallback=details.innerHTML;
renderClusterQuality(container,quality,'cluster_assistance',centers);
const assistance=container.innerHTML.replace('chart-card screening-kpis','chart-card').replace('chart-card screening-judgment','chart-card');
renderLegacyClusterQuality(container,quality,'cluster_assistance');
assert.equal(assistance,container.innerHTML);
assert.equal(JSON.stringify([quality,centers]),before);
renderClusterQuality(container,null,'state_exploration');
assert.equal(details.innerHTML,'');
console.log(JSON.stringify(result));
"""
    output = subprocess.run([node, "-"], input=source, capture_output=True, text=True,
                            encoding="utf-8", check=True)
    rendered = json.loads(output.stdout)
    view = rendered["view"]
    assert view["columns"] == columns
    assert view["orientation"] == orientation
    if coverage is None:
        assert view["coverage"] is None
    else:
        assert view["coverage"] == pytest.approx(coverage)
    np.testing.assert_allclose(view["coordinates"], coordinates)
    assert re.findall(r"<th>(PC\d+)</th>", rendered["details"]) == [f"PC{pc + 1}" for pc in columns]
    assert re.findall(r"<th>(PC\d+)</th>", rendered["fallback"]) == ["PC1", "PC2"]
    assert orientation in rendered["details"] and orientation in rendered["summary"]
    if orientation != "主要沿 PC1":
        assert "当前状态划分主要沿 PC1" not in rendered["summary"]
    for pc in (0, 1):
        for center in coordinates:
            value = center[pc] if center[pc] != 0 else 0
            assert f"<td>{value:.3f}</td>" in rendered["details"]
    assert 'class="metrics"' not in rendered["details"]
    assert '<dt>平均持续时间</dt><dd><strong>2.000 h</strong>' in rendered["details"]
    assert '<dt>最长连续时间</dt><dd><strong>2.000 h</strong>' in rendered["details"]
    assert '<dt>状态切换次数</dt><dd><strong>1</strong>' in rendered["details"]
    assert "按采样覆盖时长统计；物理缺口分段，缺口两侧不计状态切换。" in rendered["details"]


def test_all_feature_contrasts_keep_constant_means_and_explain_missing_cluster_values():
    scores = pd.DataFrame({"pc1": [-1, -1, 1, 1], "pc2": 0})
    raw = pd.DataFrame({"constant": 5.0, "one_cluster": [1, 2, np.inf, np.nan], "driver": [10, 12, 20, 22]})
    quality = analyze_cluster_quality(scores, ["C001", "C001", "C002", "C002"], raw_data=raw, feature_names=raw.columns)
    contrasts = {item["tag"]: item for item in quality["feature_contrasts"]}
    assert [item["tag"] for item in quality["top_features"]] == ["driver"]
    assert contrasts["constant"]["cluster_means"] == {"C001": 5.0, "C002": 5.0}
    assert contrasts["constant"]["mean_difference"] == 0.0
    assert contrasts["constant"]["standardized_difference"] is None
    assert "常量" in contrasts["constant"]["unavailable_reason"]
    assert contrasts["one_cluster"]["cluster_means"] == {"C001": 1.5, "C002": None}
    assert contrasts["one_cluster"]["standardized_difference"] is None
    json.dumps(quality, allow_nan=False)
