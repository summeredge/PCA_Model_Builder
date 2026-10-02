import json
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
        assert 'renderClusterQuality(el("explorationClusterQuality"),data.cluster_quality,"state_exploration")' in html
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
    renderClusterQuality(container,degraded,'state_exploration'); const degradedMeans=container.innerHTML;
    const legacy=JSON.parse(JSON.stringify(quality));
    delete legacy.top_features[0].cluster_means;
    renderClusterQuality(container,legacy,'state_exploration'); const unavailableMeans=container.innerHTML;
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
        assert "&lt;driver&gt;" in rendered[perspective]
        assert "<driver>" not in rendered[perspective]
        assert "<th>标准化差异</th><th>原始均值差</th><th>Cluster 1</th><th>Cluster 2</th>" in rendered[perspective]
        assert "<tr><td>&lt;driver&gt;</td><td>2.000</td><td>1.000</td><td>0.000</td><td>1.000</td></tr>" in rendered[perspective]
        assert "Top5" in rendered[perspective]
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
