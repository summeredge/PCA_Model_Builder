from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np

from .dpca import DPCAModel


SCHEMA_VERSION = 1
_ARRAY_NAMES = {
    "mean",
    "scale",
    "components",
    "eigenvalues",
    "explained_variance_ratio",
}


def save_model_package(
    path: str | Path,
    model: DPCAModel,
    config: dict[str, Any],
    training_windows: list[list[str]],
    validation_status: str = "draft",
) -> None:
    if validation_status not in {"draft", "passed", "failed"}:
        raise ValueError("invalid validation status")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation_status,
        "feature_names": list(model.feature_names),
        "n_samples": model.n_samples,
        "n_components": model.n_components,
        "t2_limits": {str(key): value for key, value in model.t2_limits.items()},
        "q_limits": {str(key): value for key, value in model.q_limits.items()},
        "config": config,
        "training_windows": training_windows,
    }
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


def load_model_package(path: str | Path) -> tuple[DPCAModel, dict[str, Any]]:
    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        if names != {"manifest.json", "arrays.npz"}:
            raise ValueError("model package has unexpected or missing files")
        manifest = json.loads(package.read("manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("model package manifest must be an object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported model package schema version")
        with np.load(BytesIO(package.read("arrays.npz")), allow_pickle=False) as arrays:
            if set(arrays.files) != _ARRAY_NAMES:
                raise ValueError("model package arrays are unexpected or incomplete")
            try:
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
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError("model package structure is invalid") from error
    _validate_loaded_model(model, manifest)
    return model, manifest


def _validate_loaded_model(model: DPCAModel, manifest: dict[str, Any]) -> None:
    feature_names = manifest.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise ValueError("model package feature names are invalid")
    if manifest.get("validation_status") not in {"draft", "passed", "failed"}:
        raise ValueError("model package validation status is invalid")
    if not isinstance(manifest.get("config"), dict) or not isinstance(
        manifest.get("training_windows"), list
    ):
        raise ValueError("model package metadata is invalid")

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
        or len(model.eigenvalues) <= component_count
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

    if set(model.t2_limits) != {0.95, 0.99} or set(model.q_limits) != {0.95, 0.99}:
        raise ValueError("model package control limits are incomplete")
    limits = np.array([*model.t2_limits.values(), *model.q_limits.values()])
    if not np.isfinite(limits).all():
        raise ValueError("model package control limits must be finite")
    if not 0 < model.t2_limits[0.95] <= model.t2_limits[0.99]:
        raise ValueError("model package T2 limits are invalid")
    if not 0 <= model.q_limits[0.95] <= model.q_limits[0.99]:
        raise ValueError("model package SPE limits are invalid")
