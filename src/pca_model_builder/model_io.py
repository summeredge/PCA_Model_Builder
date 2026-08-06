from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
from .preprocessing import PreprocessingConfig, preprocessing_config_from_mapping
from .tag_config import normalize_tag_configs, normalize_tag_registry
from .compat import (
    normalize_manifest_training_windows,
    normalize_model_semantics,
    normalize_training_windows_for_write,
    validate_new_model_semantics,
)
from .scoring_core import BatchScoreResult, score_dynamic_feature_matrix
from .validation import has_complete_validation_evidence
from .windows import normalize_training_windows


SCHEMA_VERSION = 4
DEPLOYMENT_SCHEMA_VERSION = 1
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
_FROZEN_MANIFEST_FIELDS = _MANIFEST_FIELDS_V2 | {
    "model_id",
    "model_version",
    "freeze_info",
    "source_validated_package",
    "status_rules",
    "contribution_rules",
}
_DEPLOYMENT_MANIFEST_FIELDS = {
    "deployment_schema_version",
    "model_id",
    "model_version",
    "created_at",
    "source_frozen_package",
    "arrays_sha256",
    "input_tags",
    "preprocessing",
    "dynamic_feature_names",
    "n_samples",
    "n_components",
    "t2_limits",
    "q_limits",
    "status_rules",
    "contribution_rules",
}
_DEPLOYMENT_PREPROCESSING_FIELDS = {
    "sample_interval_minutes",
    "resampling_method",
    "resampling_origin",
    "resampling_closed",
    "resampling_label",
    "filter_method",
    "smoothing_window_minutes",
    "gap_threshold_minutes",
    "max_lag_minutes",
    "lag_step_minutes",
    "state_filters",
}
_TRAINING_WINDOW_TOTAL_FIELDS = {
    "enabled_window_count",
    "used_window_count",
    "dropped_window_count",
    "training_rows",
}
_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

STATUS_RULES = {
    "t2_and_spe_independent": True,
    "normal": "below_95",
    "attention": "at_or_above_95_below_99",
    "abnormal": "at_or_above_99",
    "overall_status": "most_severe_of_t2_and_spe",
    "unscorable_status": "not_scored",
}
CONTRIBUTION_RULES = {
    "statistics": ["t2", "spe"],
    "only_95_exceedance": True,
    "aggregate_lags_to_input_tag": True,
    "meaning": "statistical_deviation_not_root_cause",
}


@dataclass(frozen=True)
class DeploymentModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    t2_limits: dict[float, float]
    q_limits: dict[float, float]

    def score_dynamic_features(self, dynamic_features: np.ndarray) -> BatchScoreResult:
        return score_dynamic_feature_matrix(
            dynamic_features,
            feature_names=self.feature_names,
            mean=self.mean,
            scale=self.scale,
            components=self.components,
            eigenvalues=self.eigenvalues,
            t2_limits=self.t2_limits,
            q_limits=self.q_limits,
        )


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
        "config": _normalize_preprocessing_config(config),
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
            "sha256": _sha256(source.read_bytes()),
        },
    )


def freeze_validated_model_package(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    model_id: str,
    model_version: int,
    frozen_by: str,
    comment: str = "",
) -> None:
    """Create a non-overwriting frozen package from a fully approved validation."""
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve() == destination.resolve():
        raise ValueError("frozen model output must differ from the validated package")
    if destination.exists():
        raise ValueError("frozen model output already exists")
    _validate_freeze_request(model_id, model_version, frozen_by, comment)
    try:
        with zipfile.ZipFile(source) as package:
            raw_manifest = json.loads(package.read("manifest.json"))
            if not isinstance(raw_manifest, dict):
                raise ValueError("manifest must be an object")
            source_schema_version = raw_manifest.get("schema_version")
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("validated model package cannot be read") from error
    if source_schema_version != SCHEMA_VERSION:
        raise ValueError("only schema 4 validated models can be frozen")
    model, manifest = load_model_package(source)
    if (
        manifest["model_purpose"] != "normal_state"
        or manifest["model_status"] != "validated"
    ):
        raise ValueError("only normal_state/validated models can be frozen")
    if manifest.get("engineer_decision", {}).get("decision") != "passed":
        raise ValueError("only engineer-passed models can be frozen")
    if not has_complete_validation_evidence(manifest.get("validation_summary", {})):
        raise ValueError("validated model has incomplete PR-6 validation evidence")
    evidence = manifest.get("validation_summary", {}).get("validation_evidence")
    source_candidate = manifest.get("source_candidate_package")
    if (
        not isinstance(evidence, dict)
        or evidence.get("verification_status") != "verified"
        or not isinstance(source_candidate, dict)
        or source_candidate.get("sha256") != evidence.get("candidate_model", {}).get("sha256")
    ):
        raise ValueError("validated model lacks bound candidate and artifact evidence")

    source_bytes = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frozen_manifest = {
        **manifest,
        "schema_version": SCHEMA_VERSION,
        "model_status": "frozen",
        "model_id": model_id,
        "model_version": model_version,
        "freeze_info": {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "frozen_by": frozen_by,
            "comment": comment,
        },
        "source_validated_package": {
            "filename": source.name,
            "sha256": _sha256(source_bytes),
        },
        "status_rules": STATUS_RULES,
        "contribution_rules": CONTRIBUTION_RULES,
    }
    _write_package(destination, frozen_manifest, _arrays_bytes(model), overwrite=False)


def export_deployment_package(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    source_filename: str | None = None,
) -> None:
    """Export the fixed scoring contract of a frozen model to a .pcadeploy ZIP."""
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve() == destination.resolve():
        raise ValueError("deployment output must differ from the frozen package")
    if destination.suffix != ".pcadeploy":
        raise ValueError("deployment package must use the .pcadeploy extension")
    if destination.exists():
        raise ValueError("deployment output already exists")
    model, manifest = load_model_package(source)
    if (
        manifest["model_purpose"] != "normal_state"
        or manifest["model_status"] != "frozen"
    ):
        raise ValueError("only normal_state/frozen models can be exported")
    arrays = _arrays_bytes(model)
    preprocessing = {
        field: manifest["config"][field] for field in _DEPLOYMENT_PREPROCESSING_FIELDS
    }
    deployment_manifest = {
        "deployment_schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "model_id": manifest["model_id"],
        "model_version": manifest["model_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_frozen_package": {
            "filename": source_filename or source.name,
            "sha256": _sha256(source.read_bytes()),
        },
        "arrays_sha256": _sha256(arrays),
        "input_tags": list(manifest["config"]["tags"]),
        "preprocessing": preprocessing,
        "dynamic_feature_names": list(model.feature_names),
        "n_samples": model.n_samples,
        "n_components": model.n_components,
        "t2_limits": {str(key): value for key, value in model.t2_limits.items()},
        "q_limits": {str(key): value for key, value in model.q_limits.items()},
        "status_rules": manifest["status_rules"],
        "contribution_rules": manifest["contribution_rules"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_deployment_package(destination, deployment_manifest, arrays, overwrite=False)


def load_deployment_package(path: str | Path) -> tuple[DeploymentModel, dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as package:
            if set(package.namelist()) != {"deployment_manifest.json", "arrays.npz"}:
                raise ValueError("deployment package has unexpected or missing files")
            manifest = json.loads(package.read("deployment_manifest.json"))
            arrays_bytes = package.read("arrays.npz")
            _validate_deployment_manifest(manifest, arrays_bytes)
            with np.load(BytesIO(arrays_bytes), allow_pickle=False) as arrays:
                if set(arrays.files) != _ARRAY_NAMES:
                    raise ValueError("deployment package arrays are unexpected or incomplete")
                model = DeploymentModel(
                    feature_names=tuple(manifest["dynamic_feature_names"]),
                    mean=arrays["mean"].copy(),
                    scale=arrays["scale"].copy(),
                    components=arrays["components"].copy(),
                    eigenvalues=arrays["eigenvalues"].copy(),
                    explained_variance_ratio=arrays["explained_variance_ratio"].copy(),
                    t2_limits={float(key): float(value) for key, value in manifest["t2_limits"].items()},
                    q_limits={float(key): float(value) for key, value in manifest["q_limits"].items()},
                )
    except zipfile.BadZipFile as error:
        raise ValueError("deployment package is not a valid ZIP archive") from error
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("deployment package structure is invalid") from error
    _validate_deployment_model(model, manifest)
    return model, manifest


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
        "config": _normalize_preprocessing_config(manifest["config"]),
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
    if schema_version not in {1, 2, 3, SCHEMA_VERSION}:
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
    if manifest.get("model_status") == "frozen":
        missing = sorted(_FROZEN_MANIFEST_FIELDS - set(manifest))
        if missing:
            raise ValueError(f"frozen model package manifest is missing: {', '.join(missing)}")
        _validate_frozen_manifest(manifest)


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

    totals = config.get("training_window_totals")
    if totals is not None and totals["training_rows"] != model.n_samples:
        raise ValueError("model package training_window_totals row count is inconsistent")

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


def _validate_freeze_request(
    model_id: object, model_version: object, frozen_by: object, comment: object
) -> None:
    if not isinstance(model_id, str) or _MODEL_ID_PATTERN.fullmatch(model_id) is None:
        raise ValueError("model_id must contain only letters, numbers, dots, underscores, and hyphens")
    if not isinstance(model_version, int) or isinstance(model_version, bool) or model_version <= 0:
        raise ValueError("model_version must be a positive integer")
    if not isinstance(frozen_by, str) or not frozen_by.strip():
        raise ValueError("frozen_by must be a non-empty string")
    if not isinstance(comment, str):
        raise ValueError("freeze comment must be a string")


def _validate_frozen_manifest(manifest: dict[str, Any]) -> None:
    _validate_freeze_request(
        manifest["model_id"],
        manifest["model_version"],
        manifest["freeze_info"].get("frozen_by") if isinstance(manifest["freeze_info"], dict) else None,
        manifest["freeze_info"].get("comment") if isinstance(manifest["freeze_info"], dict) else None,
    )
    freeze_info = manifest["freeze_info"]
    if not isinstance(freeze_info, dict) or set(freeze_info) != {"frozen_at", "frozen_by", "comment"}:
        raise ValueError("frozen model freeze_info is invalid")
    _validate_utc_timestamp(freeze_info["frozen_at"], "frozen_at")
    _validate_source_package(manifest["source_validated_package"], "source_validated_package")
    if manifest["status_rules"] != STATUS_RULES or manifest["contribution_rules"] != CONTRIBUTION_RULES:
        raise ValueError("frozen model scoring semantics are invalid")
    if manifest.get("engineer_decision", {}).get("decision") != "passed":
        raise ValueError("frozen model engineer decision is invalid")
    if not has_complete_validation_evidence(manifest.get("validation_summary", {})):
        raise ValueError("frozen model validation evidence is incomplete")


def _validate_deployment_manifest(manifest: object, arrays: bytes) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _DEPLOYMENT_MANIFEST_FIELDS:
        raise ValueError("deployment package manifest fields are invalid")
    if manifest["deployment_schema_version"] != DEPLOYMENT_SCHEMA_VERSION:
        raise ValueError("unsupported deployment package schema version")
    _validate_freeze_request(manifest["model_id"], manifest["model_version"], "deployment", "")
    _validate_utc_timestamp(manifest["created_at"], "created_at")
    _validate_source_package(manifest["source_frozen_package"], "source_frozen_package")
    if not _is_sha256(manifest["arrays_sha256"]) or manifest["arrays_sha256"] != _sha256(arrays):
        raise ValueError("deployment package arrays SHA-256 is invalid")
    tags = manifest["input_tags"]
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag for tag in tags) or len(tags) != len(set(tags)):
        raise ValueError("deployment package input Tags are invalid")
    preprocessing = manifest["preprocessing"]
    if not isinstance(preprocessing, dict) or set(preprocessing) != _DEPLOYMENT_PREPROCESSING_FIELDS:
        raise ValueError("deployment package preprocessing is invalid")
    try:
        config = preprocessing_config_from_mapping(preprocessing)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("deployment package preprocessing is invalid") from error
    _validate_dynamic_features(manifest["dynamic_feature_names"], tags, config)
    if not isinstance(manifest["n_samples"], int) or isinstance(manifest["n_samples"], bool) or manifest["n_samples"] < 3:
        raise ValueError("deployment package sample count is invalid")
    if not isinstance(manifest["n_components"], int) or isinstance(manifest["n_components"], bool) or manifest["n_components"] < 1:
        raise ValueError("deployment package component count is invalid")
    if manifest["status_rules"] != STATUS_RULES or manifest["contribution_rules"] != CONTRIBUTION_RULES:
        raise ValueError("deployment package scoring semantics are invalid")


def _validate_deployment_model(model: DeploymentModel, manifest: dict[str, Any]) -> None:
    full_model = DPCAModel(
        feature_names=model.feature_names,
        mean=model.mean,
        scale=model.scale,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
        t2_limits=model.t2_limits,
        q_limits=model.q_limits,
        n_samples=manifest["n_samples"],
    )
    _validate_loaded_model(
        full_model,
        {
            "feature_names": list(model.feature_names),
            "config": {"model_name": "deployment", "tags": manifest["input_tags"], "timestamp_column": "deployment", **manifest["preprocessing"], "variance_threshold": 0.95},
            "training_windows": [{"id": "deployment", "start": "2026-01-01", "end": "2026-01-02", "source": "deployment", "source_ref": None, "enabled": True, "comment": ""}],
            "n_components": manifest["n_components"],
        },
    )


def _validate_utc_timestamp(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UTC timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a UTC timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError(f"{label} must be a UTC timestamp")


def _validate_source_package(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"filename", "sha256"}:
        raise ValueError(f"{label} is invalid")
    if not isinstance(value["filename"], str) or not value["filename"] or not _is_sha256(value["sha256"]):
        raise ValueError(f"{label} is invalid")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _arrays_bytes(model: DPCAModel) -> bytes:
    arrays = BytesIO()
    np.savez_compressed(
        arrays,
        mean=model.mean,
        scale=model.scale,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
    )
    return arrays.getvalue()


def _write_package(
    destination: Path, manifest: dict[str, Any], arrays: bytes, *, overwrite: bool
) -> None:
    _write_zip(destination, {"manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"), "arrays.npz": arrays}, overwrite=overwrite)


def _write_deployment_package(
    destination: Path, manifest: dict[str, Any], arrays: bytes, *, overwrite: bool
) -> None:
    _write_zip(destination, {"deployment_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"), "arrays.npz": arrays}, overwrite=overwrite)


def _write_zip(destination: Path, members: dict[str, bytes], *, overwrite: bool) -> None:
    if not overwrite and destination.exists():
        raise ValueError("package output already exists")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as package:
            for name, value in members.items():
                package.writestr(name, value)
        if not overwrite and destination.exists():
            raise ValueError("package output already exists")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
        preprocessing = preprocessing_config_from_mapping(config)
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
            if any(
                condition.column not in registry
                or registry[condition.column]["role"] != "state_filter"
                for condition in preprocessing.state_filters
            ):
                raise ValueError(
                    "state filter columns must use state_filter role"
                )
        if {condition.column for condition in preprocessing.state_filters} & set(tags):
            raise ValueError("state filter columns must not be trained Tags")
        if "excluded_tags" in config:
            excluded = _validate_excluded_tags(config["excluded_tags"])
            if excluded & set(tags):
                raise ValueError("excluded_tags must not contain trained Tags")
            if "source_tag_configs" in config and not excluded <= set(registry):
                raise ValueError("excluded_tags must exist in source_tag_configs")
        if "training_window_totals" in config:
            _validate_training_window_totals(config["training_window_totals"])
    except ValueError as error:
        raise ValueError(f"model package config is invalid: {error}") from error
    return config, preprocessing


def _validate_training_window_totals(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _TRAINING_WINDOW_TOTAL_FIELDS:
        raise ValueError("training_window_totals fields are invalid")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in value.values()
    ):
        raise ValueError("training_window_totals values are invalid")
    if value["used_window_count"] + value["dropped_window_count"] != value[
        "enabled_window_count"
    ]:
        raise ValueError("training_window_totals window counts are inconsistent")


def _normalize_preprocessing_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("model package config must be an object")
    normalized = dict(config)
    normalized.setdefault("resampling_method", "none")
    normalized.setdefault("resampling_origin", "epoch")
    normalized.setdefault("resampling_closed", "right")
    normalized.setdefault("resampling_label", "right")
    normalized.setdefault("filter_method", "trailing_mean")
    normalized.setdefault("gap_threshold_minutes", None)
    normalized.setdefault("state_filters", [])
    return normalized


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
