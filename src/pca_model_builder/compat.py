from __future__ import annotations

from typing import Any

from .windows import (
    legacy_single_window_to_training_windows,
    legacy_training_windows_to_canonical,
    normalize_training_windows,
)


MODEL_PURPOSES = frozenset({"exploratory", "normal_state"})
_LOADABLE_MODEL_SEMANTICS = frozenset(
    {
        ("exploratory", "draft"),
        ("normal_state", "candidate"),
        ("normal_state", "validated"),
    }
)
_DIRECT_WRITABLE_MODEL_SEMANTICS = frozenset(
    {
        ("exploratory", "draft"),
        ("normal_state", "candidate"),
    }
)


def validate_loadable_model_semantics(
    model_purpose: object, model_status: object
) -> None:
    if (model_purpose, model_status) not in _LOADABLE_MODEL_SEMANTICS:
        raise ValueError("model purpose and status combination is invalid")


def validate_new_model_semantics(model_purpose: object, model_status: object) -> None:
    if (model_purpose, model_status) not in _DIRECT_WRITABLE_MODEL_SEMANTICS:
        raise ValueError("model purpose and status combination is invalid")


def normalize_model_semantics(manifest: dict[str, Any]) -> dict[str, str]:
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        legacy_status = manifest.get("validation_status")
        if legacy_status not in {"draft", "passed", "failed"}:
            raise ValueError("model package validation status is invalid")
        return {
            "model_purpose": "normal_state",
            "model_status": "draft",
            "legacy_validation_status": legacy_status,
        }
    validate_loadable_model_semantics(
        manifest.get("model_purpose"), manifest.get("model_status")
    )
    return {
        "model_purpose": str(manifest["model_purpose"]),
        "model_status": str(manifest["model_status"]),
    }


def normalize_manifest_training_windows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") in {1, 2}:
        return legacy_training_windows_to_canonical(manifest["training_windows"])
    return normalize_training_windows(manifest["training_windows"])


def normalize_training_windows_for_write(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return legacy_training_windows_to_canonical(value)
    return normalize_training_windows(value)


def training_windows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "training_windows" in payload:
        return normalize_training_windows(payload["training_windows"])
    return legacy_single_window_to_training_windows(
        payload.get("normal_start"), payload.get("normal_end")
    )
