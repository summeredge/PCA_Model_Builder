from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_RANGE_PAIRS = (
    ("engineering_min", "engineering_max", "engineering range"),
    ("normal_min", "normal_max", "normal operating range"),
    ("alarm_min", "alarm_max", "alarm range"),
)
TAG_ROLES = frozenset(
    {"continuous_input", "state_filter", "label_only", "exclude"}
)


def normalize_tag_registry(
    tags: Sequence[str], raw: object | None
) -> dict[str, dict[str, Any]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("tag config must be an object keyed by Tag name")
    unknown = sorted(set(str(key) for key in raw) - set(tags))
    if unknown:
        raise ValueError(f"tag config contains unknown Tags: {', '.join(unknown)}")

    result: dict[str, dict[str, Any]] = {}
    for tag in tags:
        value = raw.get(tag, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"tag config for {tag} must be an object")
        legacy_type = str(value.get("type", "")).strip()
        role = str(
            value.get(
                "role",
                "continuous_input" if legacy_type in {"", "continuous"} else legacy_type,
            )
        ).strip()
        if role not in TAG_ROLES:
            raise ValueError(f"tag config for {tag} has invalid role: {role}")
        normalized: dict[str, Any] = {
            "description": str(value.get("description", "")).strip(),
            "unit": str(value.get("unit", "")).strip(),
            "role": role,
            "comment": str(value.get("comment", "")).strip(),
        }
        for lower_key, upper_key, label in _RANGE_PAIRS:
            lower = _optional_float(value.get(lower_key), tag, lower_key)
            upper = _optional_float(value.get(upper_key), tag, upper_key)
            if (lower is None) != (upper is None):
                raise ValueError(f"{tag} {label} requires both lower and upper values")
            if lower is not None and lower >= upper:
                raise ValueError(f"{tag} {label} lower value must be less than upper")
            normalized[lower_key] = lower
            normalized[upper_key] = upper
        result[tag] = normalized
    return result


def normalize_tag_configs(
    tags: Sequence[str], raw: object | None
) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("tag config must be an object keyed by Tag name")
    unknown = sorted(set(str(key) for key in raw) - set(tags))
    if unknown:
        raise ValueError(f"tag config contains unselected Tags: {', '.join(unknown)}")
    for tag in tags:
        value = raw.get(tag, {})
        if (
            isinstance(value, Mapping)
            and "role" not in value
            and str(value.get("type", "continuous")).strip() != "continuous"
        ):
            raise ValueError(f"tag config for {tag} must have type continuous")
    registry = normalize_tag_registry(tags, raw)
    result: dict[str, dict[str, Any]] = {}
    for tag, config in registry.items():
        if config["role"] != "continuous_input":
            raise ValueError(f"tag config for {tag} must have role continuous_input")
        result[tag] = {**config, "type": "continuous"}
    return result


def engineering_ranges(
    tag_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[float, float]]:
    return {
        tag: (float(config["engineering_min"]), float(config["engineering_max"]))
        for tag, config in tag_configs.items()
        if config.get("engineering_min") is not None
        and config.get("engineering_max") is not None
    }


def _optional_float(value: object, tag: str, field: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{tag} {field} must be numeric") from error
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"{tag} {field} must be finite")
    return number
