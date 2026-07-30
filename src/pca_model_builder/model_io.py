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
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported model package schema version")
        with np.load(BytesIO(package.read("arrays.npz")), allow_pickle=False) as arrays:
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
    return model, manifest

