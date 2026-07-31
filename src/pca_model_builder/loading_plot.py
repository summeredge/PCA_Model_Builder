from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np


_DYNAMIC_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")


def _signed_loading_energy(
    component: np.ndarray,
    indices: Sequence[int],
    lags: Sequence[int],
) -> tuple[float, int | None]:
    """Return signed L2 energy and dominant lag for one original Tag."""
    if not indices:
        return 0.0, None
    values = np.asarray(component, dtype=float)[list(indices)]
    dominant_position = int(np.argmax(np.abs(values)))
    magnitude = float(np.linalg.norm(values))
    if magnitude <= np.finfo(float).eps:
        return 0.0, int(lags[dominant_position])
    sign = -1.0 if values[dominant_position] < 0 else 1.0
    return sign * magnitude, int(lags[dominant_position])


def _explained_variance_ratio(model: Any, component_index: int) -> float | None:
    ratios = np.asarray(
        getattr(model, "explained_variance_ratio", ()), dtype=float
    ).reshape(-1)
    if component_index >= len(ratios):
        return None
    value = float(ratios[component_index])
    return value if np.isfinite(value) else None


def loading_plot_payload(model: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate DPCA lag-feature loadings back to original Tag coordinates."""
    components = np.asarray(model.components, dtype=float)
    x_ratio = _explained_variance_ratio(model, 0)
    y_ratio = _explained_variance_ratio(model, 1)
    if components.ndim != 2 or components.shape[0] < 2:
        return {
            "aggregation": "signed_l2_by_original_tag",
            "x_component": "PC1",
            "y_component": "PC2",
            "x_explained_variance_ratio": x_ratio,
            "y_explained_variance_ratio": y_ratio,
            "points": [],
        }

    groups: dict[str, list[tuple[int, int]]] = {}
    for index, feature_name in enumerate(model.feature_names):
        match = _DYNAMIC_FEATURE_PATTERN.fullmatch(str(feature_name))
        if match is None:
            continue
        groups.setdefault(match.group("tag"), []).append(
            (index, int(match.group("lag")))
        )

    config = manifest.get("config", {})
    configured_tags = config.get("tags", []) if isinstance(config, Mapping) else []
    ordered_tags = [str(tag) for tag in configured_tags if str(tag) in groups]
    ordered_tags.extend(tag for tag in groups if tag not in ordered_tags)

    source_configs: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        candidate = config.get("source_tag_configs") or config.get("tag_configs") or {}
        if isinstance(candidate, Mapping):
            source_configs = candidate

    points: list[dict[str, Any]] = []
    for tag in ordered_tags:
        entries = groups[tag]
        indices = [index for index, _ in entries]
        lags = [lag for _, lag in entries]
        pc1, pc1_lag = _signed_loading_energy(components[0], indices, lags)
        pc2, pc2_lag = _signed_loading_energy(components[1], indices, lags)
        metadata = source_configs.get(tag, {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        points.append(
            {
                "tag": tag,
                "description": str(metadata.get("description", "")),
                "unit": str(metadata.get("unit", "")),
                "pc1": pc1,
                "pc2": pc2,
                "magnitude": float(np.hypot(pc1, pc2)),
                "pc1_dominant_lag_minutes": pc1_lag,
                "pc2_dominant_lag_minutes": pc2_lag,
                "lag_feature_count": len(indices),
            }
        )

    points.sort(key=lambda point: point["magnitude"], reverse=True)
    return {
        "aggregation": "signed_l2_by_original_tag",
        "aggregation_description": (
            "同一原始Tag的各Lag载荷先计算L2能量，符号取绝对载荷最大的主导Lag。"
        ),
        "x_component": "PC1",
        "y_component": "PC2",
        "x_explained_variance_ratio": x_ratio,
        "y_explained_variance_ratio": y_ratio,
        "points": points,
    }
