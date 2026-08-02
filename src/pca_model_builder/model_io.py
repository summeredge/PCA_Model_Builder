from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from .dpca import DPCAModel
from .preprocessing import PreprocessingConfig
from .tag_config import normalize_tag_configs, normalize_tag_registry
from .compat import (
    normalize_manifest_training_windows,
    normalize_model_semantics,
    normalize_training_windows_for_write,
    validate_new_model_semantics,
)
from .windows import normalize_training_windows


SCHEMA_VERSION = 3
_ARRAY_NAMES = {
    "mean",
    "scale",
    "components",
    "eigenvalues",
    "explained_variance_ratio",
}
_MANIFEST_FIELDS_V1 = {
    "schema_version",
    "validation_status",
    "feature_names",
    "n_samples",
    "n_components",
    "t2_limits",
    "q_limits",
    "config",
    "training_windows",
}
_MANIFEST_FIELDS_V2 = (_MANIFEST_FIELDS_V1 - {"validation_status"}) | {
    "model_purpose",
    "model_status",
}
_CONFIG_FIELDS = {
    "model_name",
    "tags",
    "timestamp_column",
    "sample_interval_minutes",
    "smoothing_window_minutes",
    "max_lag_minutes",
    "lag_step_minutes",
    "variance_threshold",
}
_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")


def save_model_package(
    path: str | Path,
    model: DPCAModel,
    config: dict[str, Any],
    training_windows: list[object],
    model_purpose: str = "normal_state",
    model_status: str = "candidate",
    validation_summary: dict[str, Any] | None = None,
    engineer_decision: dict[str, Any] | None = None,
    source_candidate_package: dict[str, str] | None = None,
) -> None:
    validate_new_model_semantics(model_purpose, model_status)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_purpose": model_purpose,
        "model_status": model_status,
        "feature_names": list(model.feature_names),
        "n_samples": model.n_samples,
        "n_components": model.n_components,
        "t2_limits": {str(key): value for key, value in model.t2_limits.items()},
        "q_limits": {str(key): value for key, value in model.q_limits.items()},
        "config": config,
        "training_windows": normalize_training_windows_for_write(training_windows),
    }
    if validation_summary is not None:
        manifest["validation_summary"] = validation_summary
    if engineer_decision is not None:
        manifest["engineer_decision"] = engineer_decision
    if source_candidate_package is not None:
        manifest["source_candidate_package"] = source_candidate_package
    arrays = BytesIO()
    np.savez_compressed(
        arrays,
        mean=model.mean,
        scale=model.scale,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as package:
            package.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
            package.writestr("arrays.npz", arrays.getvalue())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def copy_validated_model_package(
    source_path: str | Path,
    destination_path: str | Path,
    validation_summary: dict[str, Any],
    engineer_decision: dict[str, Any],
    source_identifier: str,
) -> None:
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve() == destination.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    model, manifest = load_model_package(source)
    if (
        manifest["model_purpose"] != "normal_state"
        or manifest["model_status"] != "candidate"
    ):
        raise ValueError("only normal_state/candidate models can become validated")
    save_model_package(
        destination,
        model,
        config=dict(manifest["config"]),
        training_windows=manifest["training_windows"],
        model_purpose="normal_state",
        model_status="validated",
        validation_summary=validation_summary,
        engineer_decision=engineer_decision,
        source_candidate_package={
            "identifier": source_identifier,
            "filename": source.name,
        },
    )


def load_model_package(path: str | Path) -> tuple[DPCAModel, dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            if names != {"manifest.json", "arrays.npz"}:
                raise ValueError("model package has unexpected or missing files")
            manifest = json.loads(package.read("manifest.json"))
            _validate_manifest_structure(manifest)
            with np.load(
                BytesIO(package.read("arrays.npz")), allow_pickle=False
            ) as arrays:
                if set(arrays.files) != _ARRAY_NAMES:
                    raise ValueError(
                        "model package arrays are unexpected or incomplete"
                    )
                model = DPCAModel(
                    feature_names=tuple(manifest["feature_names"]),
                    mean=arrays["mean"].copy(),
                    scale=arrays["scale"].copy(),
                    components=arrays["components"].copy(),
                    eigenvalues=arrays["eigenvalues"].copy(),
                    explained_variance_ratio=arrays[
                        "explained_variance_ratio"
                    ].copy(),
                    t2_limits={
                        float(key): float(value)
                        for key, value in manifest["t2_limits"].items()
                    },
                    q_limits={
                        float(key): float(value)
                        for key, value in manifest["q_limits"].items()
                    },
                    n_samples=int(manifest["n_samples"]),
                )
    except zipfile.BadZipFile as error:
        raise ValueError("model package is not a valid ZIP archive") from error
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("model package structure is invalid") from error
    manifest = {
        **manifest,
        **normalize_model_semantics(manifest),
        "training_windows": normalize_manifest_training_windows(manifest),
    }
    _validate_loaded_model(model, manifest)
    return model, manifest


def _validate_manifest_structure(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("model package manifest must be an object")
    schema_version = manifest.get("schema_version")
    fields = _MANIFEST_FIELDS_V1 if schema_version == 1 else _MANIFEST_FIELDS_V2
    missing = sorted(fields - set(manifest))
    if missing:
        raise ValueError(f"model package manifest is missing: {', '.join(missing)}")
    if schema_version not in {1, 2, SCHEMA_VERSION}:
        raise ValueError("unsupported model package schema version")
    normalize_model_semantics(manifest)
    if (
        not isinstance(manifest["n_samples"], int)
        or isinstance(manifest["n_samples"], bool)
        or manifest["n_samples"] < 3
        or not isinstance(manifest["n_components"], int)
        or isinstance(manifest["n_components"], bool)
        or manifest["n_components"] < 2
    ):
        raise ValueError("model package sample or component count is invalid")
    if not isinstance(manifest["t2_limits"], dict) or not isinstance(
        manifest["q_limits"], dict
    ):
        raise ValueError("model package control limits must be objects")


def _validate_loaded_model(model: DPCAModel, manifest: dict[str, Any]) -> None:
    feature_names = manifest.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise ValueError("model package feature names are invalid")
    config, preprocessing = _validate_config(manifest["config"])
    normalize_training_windows(manifest["training_windows"])
    _validate_dynamic_features(feature_names, config["tags"], preprocessing)

    feature_count = len(feature_names)
    component_count = model.n_components
    if manifest.get("n_components") != component_count:
        raise ValueError("model package component count is inconsistent")
    if (
        model.n_samples < 3
        or component_count < 1
        or component_count >= min(model.n_samples - 1, feature_count)
    ):
        raise ValueError("model package sample or component count is invalid")
    if model.mean.shape != (feature_count,) or model.scale.shape != (feature_count,):
        raise ValueError("model package standardization arrays have invalid shapes")
    if model.components.shape[1:] != (feature_count,):
        raise ValueError("model package component array has an invalid shape")
    if (
        model.eigenvalues.ndim != 1
        or model.explained_variance_ratio.shape != model.eigenvalues.shape
        or len(model.eigenvalues)
        != min(model.n_samples - 1, feature_count)
    ):
        raise ValueError("model package variance arrays have invalid shapes")

    numeric_arrays = (
        model.mean,
        model.scale,
        model.components,
        model.eigenvalues,
        model.explained_variance_ratio,
    )
    if not all(np.issubdtype(values.dtype, np.number) for values in numeric_arrays):
        raise ValueError("model package arrays must be numeric")
    if not all(np.isfinite(values).all() for values in numeric_arrays):
        raise ValueError("model package arrays contain non-finite values")
    if np.any(model.scale <= np.finfo(float).eps):
        raise ValueError("model package scale must be positive")
    if np.any(model.eigenvalues[:component_count] <= np.finfo(float).eps):
        raise ValueError("model package retained eigenvalues must be positive")
    if not np.any(model.eigenvalues[component_count:] > np.finfo(float).eps):
        raise ValueError("model package leaves no effective residual space")
    if np.any(model.explained_variance_ratio < 0):
        raise ValueError("model package explained variance must not be negative")
    if float(model.explained_variance_ratio.sum()) > 1.0 + 1e-6:
        raise ValueError("model package explained variance exceeds one")
    gram = model.components @ model.components.T
    if not np.allclose(gram, np.eye(component_count), rtol=1e-6, atol=1e-6):
        raise ValueError("model package component loadings are not orthonormal")

    if set(model.t2_limits) != {0.95, 0.99} or set(model.q_limits) != {0.95, 0.99}:
        raise ValueError("model package control limits are incomplete")
    limits = np.array([*model.t2_limits.values(), *model.q_limits.values()])
    if not np.isfinite(limits).all():
        raise ValueError("model package control limits must be finite")
    if not 0 < model.t2_limits[0.95] <= model.t2_limits[0.99]:
        raise ValueError("model package T2 limits are invalid")
    if not 0 <= model.q_limits[0.95] <= model.q_limits[0.99]:
        raise ValueError("model package SPE limits are invalid")


def _validate_config(config: object) -> tuple[dict[str, Any], PreprocessingConfig]:
    if not isinstance(config, dict):
        raise ValueError("model package config must be an object")
    missing = sorted(_CONFIG_FIELDS - set(config))
    if missing:
        raise ValueError(f"model package config is missing: {', '.join(missing)}")
    if not isinstance(config["model_name"], str) or not config["model_name"].strip():
        raise ValueError("model package model_name must be a non-empty string")
    if not isinstance(config["timestamp_column"], str) or not config[
        "timestamp_column"
    ].strip():
        raise ValueError("model package timestamp_column must be a non-empty string")
    tags = config["tags"]
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and tag.strip() for tag in tags)
        or len(tags) != len(set(tags))
    ):
        raise ValueError("model package tags must be non-empty unique strings")
    integer_fields = (
        "sample_interval_minutes",
        "smoothing_window_minutes",
        "max_lag_minutes",
        "lag_step_minutes",
    )
    if any(
        not isinstance(config[field], int) or isinstance(config[field], bool)
        for field in integer_fields
    ):
        raise ValueError("model package preprocessing values must be integers")
    variance_threshold = config["variance_threshold"]
    if (
        not isinstance(variance_threshold, (int, float))
        or isinstance(variance_threshold, bool)
        or not 0 < float(variance_threshold) < 1
    ):
        raise ValueError("model package variance threshold must be in (0, 1)")
    try:
        preprocessing = PreprocessingConfig(
            sample_interval_minutes=config["sample_interval_minutes"],
            smoothing_window_minutes=config["smoothing_window_minutes"],
            max_lag_minutes=config["max_lag_minutes"],
            lag_step_minutes=config["lag_step_minutes"],
        )
        if "tag_configs" in config and not isinstance(config["tag_configs"], dict):
            raise ValueError("tag_configs must be an object")
        if "tag_configs" in config:
            normalize_tag_configs(tags, config["tag_configs"])
        if "source_tag_configs" in config:
            source = config["source_tag_configs"]
            if not isinstance(source, dict) or not source:
                raise ValueError("source_tag_configs must be a non-empty object")
            registry = normalize_tag_registry(list(source), source)
            if any(
                tag not in registry or registry[tag]["role"] != "continuous_input"
                for tag in tags
            ):
                raise ValueError(
                    "trained Tags must be continuous_input in source_tag_configs"
                )
        if "excluded_tags" in config:
            excluded = _validate_excluded_tags(config["excluded_tags"])
            if excluded & set(tags):
                raise ValueError("excluded_tags must not contain trained Tags")
            if "source_tag_configs" in config and not excluded <= set(registry):
                raise ValueError("excluded_tags must exist in source_tag_configs")
    except ValueError as error:
        raise ValueError(f"model package config is invalid: {error}") from error
    return config, preprocessing


def _validate_excluded_tags(value: object) -> set[str]:
    if not isinstance(value, list):
        raise ValueError("excluded_tags must be a list")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("excluded_tags entries must be objects")
        required = {
            "tag",
            "reason",
            "sample_count",
            "unique_count",
            "constant_value",
        }
        if set(item) != required:
            raise ValueError("excluded_tags entry fields are invalid")
        tag = item["tag"]
        if not isinstance(tag, str) or not tag.strip() or tag in seen:
            raise ValueError("excluded_tags contain invalid or duplicate Tags")
        seen.add(tag)
        if item["reason"] != "constant_in_reference_window":
            raise ValueError("excluded_tags reason is invalid")
        if (
            not isinstance(item["sample_count"], int)
            or isinstance(item["sample_count"], bool)
            or item["sample_count"] < 1
            or item["unique_count"] != 1
            or not isinstance(item["constant_value"], (int, float))
            or isinstance(item["constant_value"], bool)
            or not np.isfinite(item["constant_value"])
        ):
            raise ValueError("excluded_tags constant metadata is invalid")
    return seen


def _validate_dynamic_features(
    feature_names: list[str],
    tags: list[str],
    config: PreprocessingConfig,
) -> None:
    if any(_FEATURE_PATTERN.fullmatch(name) is None for name in feature_names):
        raise ValueError("model package dynamic feature name is invalid")
    expected = [
        f"{tag}__lag_{lag_minutes:03d}min"
        for lag_minutes in range(
            0, config.max_lag_minutes + 1, config.lag_step_minutes
        )
        for tag in tags
    ]
    if feature_names != expected:
        raise ValueError(
            "model package dynamic features do not match configured Tags and Lags"
        )
