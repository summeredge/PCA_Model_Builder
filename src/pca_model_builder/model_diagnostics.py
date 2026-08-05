from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .model_io import load_model_package


_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")


def model_structure_diagnostic(
    model: Any, manifest: Mapping[str, Any], run_id: str | None = None
) -> dict[str, Any]:
    """Return read-only, reproducible structural facts for one model package."""
    config = manifest["config"]
    components = np.asarray(model.components, dtype=float)
    ratios = np.asarray(model.explained_variance_ratio, dtype=float)
    eigenvalues = np.asarray(model.eigenvalues, dtype=float)
    retained = int(model.n_components)
    lags, tags = _dynamic_feature_coordinates(model.feature_names)
    cumulative = np.cumsum(ratios)
    windows = list(manifest["training_windows"])

    return {
        "run_id": run_id,
        "model_name": config["model_name"],
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "training_window_count": len(windows),
        "enabled_training_window_count": sum(window["enabled"] for window in windows),
        "training_dynamic_samples": int(model.n_samples),
        "raw_tag_count": len(config["tags"]),
        "dynamic_feature_count": len(model.feature_names),
        "preprocessing": {
            "timestamp_column": config["timestamp_column"],
            "sample_interval_minutes": config["sample_interval_minutes"],
            "resampling_method": config["resampling_method"],
            "filter_method": config["filter_method"],
            "smoothing_window_minutes": config["smoothing_window_minutes"],
            "max_lag_minutes": config["max_lag_minutes"],
            "lag_step_minutes": config["lag_step_minutes"],
        },
        "retained_component_count": retained,
        "eigenvalues": [float(value) for value in eigenvalues],
        "explained_variance_ratio": [float(value) for value in ratios],
        "cumulative_explained_variance_ratio": [float(value) for value in cumulative],
        "retained_eigenvalues": [float(value) for value in eigenvalues[:retained]],
        "retained_explained_variance_ratio": [
            float(value) for value in ratios[:retained]
        ],
        "retained_cumulative_explained_variance_ratio": [
            float(value) for value in cumulative[:retained]
        ],
        "components_for_cumulative_explained_variance": {
            str(int(level * 100)): _component_count_for_level(cumulative, level)
            for level in (0.80, 0.90, 0.95)
        },
        "dynamic_features_per_training_sample": float(
            len(model.feature_names) / model.n_samples
        ),
        "control_limits": {
            "t2": _control_limits(model.t2_limits),
            "spe": _control_limits(model.q_limits),
        },
        "tag_loading_energy": _energy_by_tag(components, tags, retained),
        "lag_loading_energy": _energy_by_lag(components, lags, retained),
    }


def compare_candidate_runs(
    run_ids: object, runs_dir: str | Path
) -> dict[str, Any]:
    """Compare 2--4 saved normal-state candidates without changing packages."""
    ids = _validated_run_ids(run_ids)
    loaded: list[tuple[str, Any, dict[str, Any]]] = []
    for run_id in ids:
        package_path = Path(runs_dir) / run_id / "model.pcamodel"
        if not package_path.is_file():
            raise ValueError(f"候选模型运行记录不存在：{run_id}")
        try:
            model, manifest = load_model_package(package_path)
        except ValueError as error:
            raise ValueError(f"候选模型包损坏：{run_id}") from error
        if (
            manifest["model_purpose"] != "normal_state"
            or manifest["model_status"] != "candidate"
        ):
            raise ValueError(f"仅允许比较normal_state/candidate模型：{run_id}")
        loaded.append((run_id, model, manifest))

    baseline_id, _, _ = loaded[0]
    parameters = [
        _comparison_parameters(run_id, manifest)
        for run_id, _, manifest in loaded
    ]
    differences = [
        {
            "run_id": run_id,
            "differences": {
                key: {"baseline": baseline_value, "value": value}
                for key, value in values.items()
                if key != "run_id" and value != parameters[0][key]
                for baseline_value in [parameters[0][key]]
            },
        }
        for run_id, values in zip(ids[1:], parameters[1:], strict=True)
    ]
    comparable, reasons = _comparability(parameters)
    return {
        "baseline_run_id": baseline_id,
        "diagnostics": [
            model_structure_diagnostic(model, manifest, run_id)
            for run_id, model, manifest in loaded
        ],
        "parameter_table": _parameter_table(parameters),
        "parameter_differences": differences,
        "comparability": {
            "comparable": comparable,
            "status": (
                "not_comparable"
                if not comparable
                else "strict_comparable"
                if _source_identity_state(parameters) == "same"
                else "structural_comparison_only"
            ),
            "reasons": reasons,
        },
    }


def _validated_run_ids(value: object) -> list[str]:
    if not isinstance(value, list) or not 2 <= len(value) <= 4:
        raise ValueError("run_ids必须包含2至4个候选运行")
    if any(
        not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id)
        for run_id in value
    ):
        raise ValueError("run_id无效")
    if len(value) != len(set(value)):
        raise ValueError("run_ids不能重复")
    return list(value)


def _dynamic_feature_coordinates(feature_names: Sequence[str]) -> tuple[list[int], list[str]]:
    lags: list[int] = []
    tags: list[str] = []
    for feature_name in feature_names:
        match = _FEATURE_PATTERN.fullmatch(str(feature_name))
        if match is None:
            raise ValueError("模型动态特征名称无效")
        tags.append(match.group("tag"))
        lags.append(int(match.group("lag")))
    return lags, tags


def _control_limits(limits: Mapping[float, float]) -> dict[str, float]:
    return {str(int(level * 100)): float(limits[level]) for level in (0.95, 0.99)}


def _component_count_for_level(cumulative: np.ndarray, level: float) -> int | None:
    matches = np.flatnonzero(cumulative >= level - 1e-12)
    return None if not len(matches) else int(matches[0] + 1)


def _energy_by_tag(
    components: np.ndarray, tags: Sequence[str], retained: int
) -> dict[str, list[dict[str, Any]] | None]:
    ordered_tags = list(dict.fromkeys(tags))
    return {
        "pc1": _group_energy(components[0], tags, ordered_tags),
        "pc2": _group_energy(components[1], tags, ordered_tags)
        if retained >= 2
        else None,
        "retained_components": _group_energy(
            components[:retained], tags, ordered_tags
        ),
    }


def _energy_by_lag(
    components: np.ndarray, lags: Sequence[int], retained: int
) -> dict[str, Any]:
    ordered_lags = sorted(set(lags))
    retained_energy = _group_energy(components[:retained], lags, ordered_lags)
    by_lag = {item["lag_minutes"]: item["energy"] for item in retained_energy}
    zero_lag = float(by_lag.get(0, 0.0))
    nonzero_lag = float(1.0 - zero_lag)
    dominant = max(retained_energy, key=lambda item: item["energy"])["lag_minutes"]
    return {
        "pc1": _group_energy(components[0], lags, ordered_lags),
        "pc2": _group_energy(components[1], lags, ordered_lags)
        if retained >= 2
        else None,
        "retained_components": retained_energy,
        "zero_lag_energy": zero_lag,
        "nonzero_lag_energy": nonzero_lag,
        "dominant_lag_minutes": dominant,
    }


def _group_energy(
    values: np.ndarray, coordinates: Sequence[Any], ordered: Sequence[Any]
) -> list[dict[str, Any]]:
    squared = np.square(np.asarray(values, dtype=float))
    if squared.ndim == 1:
        squared = squared[np.newaxis, :]
    total = float(squared.sum())
    return [
        {
            **(
                {"tag": coordinate}
                if isinstance(coordinate, str)
                else {"lag_minutes": coordinate}
            ),
            "energy": float(squared[:, index].sum() / total)
            if total > np.finfo(float).eps
            else 0.0,
        }
        for coordinate in ordered
        for index in [
            [position for position, value in enumerate(coordinates) if value == coordinate]
        ]
    ]


def _comparison_parameters(run_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest["config"]
    return {
        "run_id": run_id,
        "original_tags": list(config["tags"]),
        "training_windows": list(manifest["training_windows"]),
        "timestamp_column": config["timestamp_column"],
        "state_filters": list(config["state_filters"]),
        "sample_interval_minutes": config["sample_interval_minutes"],
        "resampling_method": config["resampling_method"],
        "filter_method": config["filter_method"],
        "smoothing_window_minutes": config["smoothing_window_minutes"],
        "max_lag_minutes": config["max_lag_minutes"],
        "lag_step_minutes": config["lag_step_minutes"],
        "variance_threshold": config["variance_threshold"],
        "retained_component_count": manifest["n_components"],
        "source_identity": config.get("source_identity"),
    }


def _parameter_table(parameters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "parameter": key,
            "values": {item["run_id"]: item[key] for item in parameters},
        }
        for key in parameters[0]
        if key != "run_id"
    ]


def _comparability(
    parameters: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    labels = {
        "original_tags": "原始Tag及顺序不一致",
        "training_windows": "标准化训练窗口范围或启用状态不一致",
        "timestamp_column": "时间列不一致",
        "state_filters": "状态过滤条件不一致",
        "sample_interval_minutes": "采样间隔不一致",
        "resampling_method": "重采样方法不一致",
    }
    baseline = parameters[0]
    reasons = [
        label
        for key, label in labels.items()
        if any(item[key] != baseline[key] for item in parameters[1:])
    ]
    if reasons:
        return False, reasons
    source_state = _source_identity_state(parameters)
    if source_state == "different":
        return False, ["训练源身份不一致，不能作为同数据A/B对照"]
    if source_state == "missing":
        return True, ["训练源身份未固化，仅作结构比较"]
    return True, []


def _source_identity_state(parameters: Sequence[Mapping[str, Any]]) -> str:
    identities = [item["source_identity"] for item in parameters]
    if any(identity in (None, "") for identity in identities):
        return "missing"
    return "same" if len({repr(identity) for identity in identities}) == 1 else "different"
