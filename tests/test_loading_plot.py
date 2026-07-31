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

    assert payload["aggregation"] == "signed_l2_by_original_tag"
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
    )

    payload = loading_plot_payload(model, {"config": {"tags": ["TAG_A"]}})

    assert payload["points"] == []
