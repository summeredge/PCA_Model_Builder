from __future__ import annotations

from typing import Any


MODEL_PURPOSES = frozenset({"exploratory", "normal_state"})
_WRITABLE_MODEL_SEMANTICS = {
    ("exploratory", "draft"),
    ("normal_state", "candidate"),
}


def validate_new_model_semantics(model_purpose: object, model_status: object) -> None:
    if (model_purpose, model_status) not in _WRITABLE_MODEL_SEMANTICS:
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
    validate_new_model_semantics(
        manifest.get("model_purpose"), manifest.get("model_status")
    )
    return {
        "model_purpose": str(manifest["model_purpose"]),
        "model_status": str(manifest["model_status"]),
    }
