from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pca_model_builder.loading_plot import loading_plot_payload


def test_loading_plot_aggregates_all_lags_back_to_original_tags() -> None:
    model = SimpleNamespace(
        feature_names=(
            "TAG_A__lag_000min",
            "TAG_A__lag_005min",
            "TAG_B__lag_000min",
            "TAG_B__lag_005min",
        ),
        components=np.array(
            [
                [0.3, -0.4, 0.1, 0.2],
                [0.4, 0.3, -0.6, 0.0],
            ]
        ),
        explained_variance_ratio=np.array([0.42, 0.18, 0.10]),
    )
    manifest = {
        "config": {
            "tags": ["TAG_A", "TAG_B"],
            "source_tag_configs": {
                "TAG_A": {"description": "温度", "unit": "°C"},
                "TAG_B": {"description": "压力", "unit": "kPa"},
            },
        }
    }

    payload = loading_plot_payload(model, manifest)
    points = {point["tag"]: point for point in payload["points"]}
    component_loadings = payload["component_loadings"]

    assert payload["aggregation"] == "signed_l2_by_original_tag"
    assert payload["x_explained_variance_ratio"] == pytest.approx(0.42)
    assert payload["y_explained_variance_ratio"] == pytest.approx(0.18)
    assert [item["component"] for item in component_loadings] == ["PC1", "PC2"]
    assert component_loadings[0]["explained_variance_ratio"] == pytest.approx(0.42)
    assert [item["feature"] for item in component_loadings[0]["loadings"]] == list(
        model.feature_names
    )
    assert component_loadings[0]["loadings"][1]["loading"] == pytest.approx(-0.4)
    assert [item["feature"] for item in component_loadings[0]["top_loadings"]] == [
        "TAG_A",
        "TAG_B",
    ]
    assert component_loadings[0]["top_loadings"][0]["aggregated_loading"] == pytest.approx(
        0.5
    )
    assert component_loadings[0]["top_loadings"][0]["lag_feature_count"] == 2
    assert component_loadings[0]["top_loadings"][0]["loading_energy_share"] == pytest.approx(0.25 / 0.30)
    assert points["TAG_A"]["pc1"] == pytest.approx(-0.5)
    assert points["TAG_A"]["pc2"] == pytest.approx(0.5)
    assert points["TAG_A"]["pc1_dominant_lag_minutes"] == 5
    assert points["TAG_A"]["pc2_dominant_lag_minutes"] == 0
    assert points["TAG_A"]["description"] == "温度"
    assert points["TAG_A"]["lag_feature_count"] == 2
    assert points["TAG_B"]["pc1"] == pytest.approx(np.sqrt(0.05))
    assert points["TAG_B"]["pc2"] == pytest.approx(-0.6)


def test_loading_plot_requires_pc1_and_pc2() -> None:
    model = SimpleNamespace(
        feature_names=("TAG_A__lag_000min",),
        components=np.array([[1.0]]),
        explained_variance_ratio=np.array([1.0]),
    )

    payload = loading_plot_payload(model, {"config": {"tags": ["TAG_A"]}})

    assert payload["points"] == []
    assert [item["component"] for item in payload["component_loadings"]] == ["PC1"]
    assert payload["x_explained_variance_ratio"] == pytest.approx(1.0)
    assert payload["y_explained_variance_ratio"] is None


def test_loading_plot_tolerates_legacy_model_without_explained_ratio() -> None:
    model = SimpleNamespace(
        feature_names=("TAG_A__lag_000min", "TAG_A__lag_005min"),
        components=np.array([[0.5, 0.5], [0.5, -0.5]]),
    )

    payload = loading_plot_payload(model, {"config": {"tags": ["TAG_A"]}})

    assert payload["x_explained_variance_ratio"] is None
    assert payload["y_explained_variance_ratio"] is None
    assert payload["component_loadings"][0]["explained_variance_ratio"] is None
    assert len(payload["points"]) == 1


def test_component_loadings_group_all_lags_and_keep_top_ten_original_variables() -> None:
    feature_names = tuple(
        feature
        for index in range(12)
        for feature in (f"TAG_{index}__lag_000min", f"TAG_{index}__lag_005min")
    )
    model = SimpleNamespace(
        feature_names=feature_names,
        components=np.array(
            [
                [
                    0.05, -0.9, 0.2, 0.8, -0.4, 0.1, 0.7, -0.3, 0.6, 0.15, -0.02, 0.04,
                    0.01, 0.02, 0.03, -0.04, 0.05, -0.06, 0.07, 0.08, 0.09, -0.10, 0.11, -0.12,
                ],
                [
                    -0.6, 0.3, -0.2, 0.1, 0.7, -0.05, 0.4, -0.8, 0.15, 0.09, 0.02, -0.01,
                    -0.12, 0.11, -0.10, 0.09, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03, -0.02, 0.01,
                ],
            ]
        ),
        explained_variance_ratio=np.array([0.55, 0.25]),
    )

    payload = loading_plot_payload(model, {"config": {}})
    first = payload["component_loadings"][0]

    assert [item["feature"] for item in first["loadings"]] == list(feature_names)
    assert len(first["top_loadings"]) == 10
    assert [item["feature"] for item in first["top_loadings"]] == [
        "TAG_0",
        "TAG_1",
        "TAG_3",
        "TAG_4",
        "TAG_2",
        "TAG_11",
        "TAG_10",
        "TAG_9",
        "TAG_8",
        "TAG_7",
    ]
    assert first["top_loadings"][0]["aggregated_loading"] == pytest.approx(
        np.hypot(0.05, -0.9)
    )
    assert first["top_loadings"][0]["lag_feature_count"] == 2
    assert first["top_loadings"][0]["loading_energy_share"] == pytest.approx(
        (0.05**2 + 0.9**2) / np.square(model.components[0]).sum()
    )
    assert sum(item["loading_energy_share"] for item in first["top_loadings"]) < 1
    assert all("_lag_" not in item["feature"] for item in first["top_loadings"])
    assert len({item["feature"] for item in first["top_loadings"]}) == 10
    assert [item["aggregated_loading"] for item in first["top_loadings"]] == sorted(
        (item["aggregated_loading"] for item in first["top_loadings"]), reverse=True
    )
    assert {"TAG_5", "TAG_6"}.isdisjoint(
        item["feature"] for item in first["top_loadings"]
    )


def test_component_loadings_aggregate_lag_and_keep_plain_features_as_variables() -> None:
    feature_names = (
        "AI400014.PV__lag_000min",
        "AI400014.PV__lag_005min",
        "AI400014.PV__lag_010min",
        "STEAM.FLOW",
    )
    model = SimpleNamespace(
        feature_names=feature_names,
        components=np.array(
            [
                [0.6, 0.8, 0.0, -0.3],
                [0.1, -0.2, 0.05, 0.5],
            ]
        ),
        explained_variance_ratio=np.array([0.5, 0.25]),
    )

    payload = loading_plot_payload(model, {"config": {}})
    first = payload["component_loadings"][0]

    assert [item["feature"] for item in first["top_loadings"]] == [
        "AI400014.PV",
        "STEAM.FLOW",
    ]
    assert first["top_loadings"][0]["aggregated_loading"] == pytest.approx(
        np.sqrt(0.6**2 + 0.8**2 + 0.0**2)
    )
    assert first["top_loadings"][0]["lag_feature_count"] == 3
    assert first["top_loadings"][1]["aggregated_loading"] == pytest.approx(0.3)
    assert first["top_loadings"][1]["lag_feature_count"] == 1


def test_component_loadings_keep_tags_that_end_with_underscore_separate() -> None:
    feature_names = (
        "TAG_A__lag_000min",
        "TAG_A__lag_005min",
        "TAG_A___lag_000min",
    )
    model = SimpleNamespace(
        feature_names=feature_names,
        components=np.array([[0.3, -0.4, 0.12], [0.5, 0.5, 0.1]]),
        explained_variance_ratio=np.array([0.5, 0.25]),
    )

    payload = loading_plot_payload(model, {"config": {}})
    first = payload["component_loadings"][0]

    assert [item["feature"] for item in first["top_loadings"]] == ["TAG_A", "TAG_A_"]
    assert first["top_loadings"][0]["aggregated_loading"] == pytest.approx(0.5)
    assert first["top_loadings"][0]["lag_feature_count"] == 2
    assert first["top_loadings"][1]["aggregated_loading"] == pytest.approx(0.12)
    assert first["top_loadings"][1]["lag_feature_count"] == 1
