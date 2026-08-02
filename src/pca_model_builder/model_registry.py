from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from .compat import validate_loadable_model_semantics
from .model_io import (
    _write_model_package,
    load_model_package,
    model_package_sha256,
)


DEFAULT_REGISTRY_DIR = Path(".web_data") / "models"
SOFTWARE_VERSION = "0.1.0"
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_VERSION_PATTERN = re.compile(r"^v(?P<number>\d{4})$")


def _sidecar_path(package_path: Path) -> Path:
    return package_path.with_name(f"{package_path.name}.sha256")


def _nonempty_scope(value: object) -> bool:
    return bool(
        (isinstance(value, str) and value.strip())
        or (isinstance(value, (list, dict)) and value)
    )


def _safe_model_id(value: str) -> str:
    if not isinstance(value, str) or _MODEL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("model_id只能包含字母、数字、点、下划线和短横线")
    return value


def _derived_model_id(source_path: Path) -> str:
    return f"model-{model_package_sha256(source_path)[:16]}"


def _version_number(version: str) -> int:
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError("模型版本号无效")
    return int(match.group("number"))


def _reserve_version_directory(registry_root: Path, model_id: str) -> tuple[str, Path]:
    model_root = registry_root / model_id
    model_root.mkdir(parents=True, exist_ok=True)
    existing = [
        _version_number(item.name)
        for item in model_root.iterdir()
        if item.is_dir() and _VERSION_PATTERN.fullmatch(item.name)
    ]
    next_number = max(existing, default=0) + 1
    while next_number <= 9999:
        version = f"v{next_number:04d}"
        destination = model_root / version
        try:
            destination.mkdir()
            return version, destination
        except FileExistsError:
            next_number += 1
    raise ValueError("模型版本号已耗尽")


def _latest_registry_parent(
    registry_root: str | Path, model_id: str
) -> tuple[str, str] | None:
    model_root = Path(registry_root) / model_id
    if not model_root.is_dir():
        return None
    candidates = sorted(
        (
            item
            for item in model_root.iterdir()
            if item.is_dir() and _VERSION_PATTERN.fullmatch(item.name)
        ),
        key=lambda item: _version_number(item.name),
    )
    for version_dir in reversed(candidates):
        package = version_dir / "model.pcamodel"
        try:
            _, manifest = load_model_package(package)
        except (OSError, ValueError):
            continue
        return str(manifest["model_id"]), str(manifest["version"])
    return None


def _write_external_hash(package_path: Path) -> str:
    digest = model_package_sha256(package_path)
    sidecar = _sidecar_path(package_path)
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary.write_text(f"{digest}  {package_path.name}\n", encoding="ascii")
    temporary.replace(sidecar)
    return digest


def _read_external_hash(package_path: Path) -> str | None:
    sidecar = _sidecar_path(package_path)
    if not sidecar.is_file():
        return None
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if not fields or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise ValueError("模型包外部SHA-256文件无效")
    if len(fields) > 1 and fields[-1] != package_path.name:
        raise ValueError("模型包外部SHA-256文件名不匹配")
    return fields[0]


def verify_model_package_integrity(
    package_path: str | Path,
    *,
    require_external: bool = False,
) -> dict[str, Any]:
    """Verify the safe package, its v4 array hash, and optional sidecar hash."""
    package = Path(package_path)
    if not package.is_file():
        raise ValueError("模型包不存在")
    _, manifest = load_model_package(package)
    actual_sha256 = model_package_sha256(package)
    external_sha256 = _read_external_hash(package)
    if require_external and external_sha256 is None:
        raise ValueError("模型包缺少外部SHA-256文件")
    if external_sha256 is not None and external_sha256 != actual_sha256:
        raise ValueError("模型包外部SHA-256校验失败")
    return {
        "valid": True,
        "path": str(package),
        "bytes": package.stat().st_size,
        "sha256": actual_sha256,
        "external_sha256": external_sha256,
        "schema_version": manifest["schema_version"],
    }


def _record(package_path: Path, manifest: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": manifest.get("model_id"),
        "version": manifest.get("version"),
        "parent_model_id": manifest.get("parent_model_id"),
        "parent_version": manifest.get("parent_version"),
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "created_at": manifest.get("created_at"),
        "software_version": manifest.get("software_version"),
        "engineer_comment": manifest.get("engineer_comment", ""),
        "applicability_scope": manifest.get("applicability_scope"),
        "file_hashes": manifest.get("file_hashes"),
        "path": str(package_path),
        "integrity": dict(integrity),
    }


def create_model_version(
    source_path: str | Path,
    registry_root: str | Path = DEFAULT_REGISTRY_DIR,
    *,
    model_id: str | None = None,
    parent_model_id: str | None = None,
    parent_version: str | None = None,
    model_purpose: str | None = None,
    model_status: str | None = None,
    engineer_comment: str | None = None,
    applicability_scope: object | None = None,
) -> dict[str, Any]:
    """Create one immutable schema-v4 copy without changing the source."""
    source = Path(source_path)
    model, manifest = load_model_package(source)
    source_sha256 = model_package_sha256(source)
    resolved_model_id = _safe_model_id(
        model_id or manifest.get("model_id") or _derived_model_id(source)
    )
    purpose = model_purpose or str(manifest["model_purpose"])
    status = model_status or str(manifest["model_status"])
    validate_loadable_model_semantics(purpose, status)
    if status == "published":
        raise ValueError("published版本只能通过显式发布操作创建")
    if parent_model_id is None and manifest.get("model_id"):
        parent_model_id = str(manifest["model_id"])
        parent_version = str(manifest.get("version"))
    if engineer_comment is None:
        engineer_comment = str(manifest.get("engineer_comment", ""))
    if applicability_scope is None:
        applicability_scope = manifest.get("applicability_scope", {})
    result = _write_registry_version(
        model,
        manifest,
        registry_root,
        model_id=resolved_model_id,
        parent_model_id=parent_model_id,
        parent_version=parent_version,
        model_purpose=purpose,
        model_status=status,
        engineer_comment=engineer_comment,
        applicability_scope=applicability_scope,
        source_sha256=source_sha256,
    )
    return result


def _write_registry_version(
    model: Any,
    manifest: Mapping[str, Any],
    registry_root: str | Path,
    *,
    model_id: str,
    parent_model_id: str | None,
    parent_version: str | None,
    model_purpose: str,
    model_status: str,
    engineer_comment: str,
    applicability_scope: object,
    source_sha256: str,
) -> dict[str, Any]:
    """Reserve one version directory and write its package exactly once."""
    registry = Path(registry_root)
    version, version_dir = _reserve_version_directory(registry, model_id)
    package = version_dir / "model.pcamodel"
    metadata = {
        "model_id": model_id,
        "version": version,
        "parent_model_id": parent_model_id,
        "parent_version": parent_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software_version": SOFTWARE_VERSION,
        "engineer_comment": engineer_comment,
        "applicability_scope": applicability_scope,
    }
    try:
        _write_model_package(
            package,
            model,
            config=dict(manifest["config"]),
            training_windows=manifest["training_windows"],
            model_purpose=model_purpose,
            model_status=model_status,
            validation_summary=manifest.get("validation_summary"),
            engineer_decision=manifest.get("engineer_decision"),
            source_candidate_package=manifest.get("source_candidate_package"),
            schema_version=4,
            lifecycle_metadata=metadata,
        )
        integrity_sha256 = _write_external_hash(package)
    except Exception:
        if package.exists():
            package.unlink()
        sidecar = _sidecar_path(package)
        if sidecar.exists():
            sidecar.unlink()
        version_dir.rmdir()
        raise
    _, stored_manifest = load_model_package(package)
    integrity = verify_model_package_integrity(package, require_external=True)
    result = _record(package, stored_manifest, integrity)
    result["source_sha256"] = source_sha256
    result["file_hashes"] = stored_manifest["file_hashes"]
    result["integrity"]["sha256"] = integrity_sha256
    return result


def list_model_versions(
    registry_root: str | Path = DEFAULT_REGISTRY_DIR,
) -> list[dict[str, Any]]:
    registry = Path(registry_root)
    if not registry.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for model_root in sorted(item for item in registry.iterdir() if item.is_dir()):
        for version_dir in sorted(
            (item for item in model_root.iterdir() if item.is_dir()),
            key=lambda item: (_version_number(item.name) if _VERSION_PATTERN.fullmatch(item.name) else 10000, item.name),
        ):
            package = version_dir / "model.pcamodel"
            if not package.is_file() or _VERSION_PATTERN.fullmatch(version_dir.name) is None:
                continue
            try:
                _, manifest = load_model_package(package)
                integrity = verify_model_package_integrity(package, require_external=True)
                records.append(_record(package, manifest, integrity))
            except (OSError, ValueError) as error:
                records.append(
                    {
                        "model_id": model_root.name,
                        "version": version_dir.name,
                        "path": str(package),
                        "integrity": {"valid": False, "error": str(error)},
                    }
                )
    return records


def _comparison_value(model: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest["config"]
    validation_summary = manifest.get("validation_summary", {})
    return {
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "software_version": manifest.get("software_version"),
        "feature_names": list(model.feature_names),
        "tags": list(config["tags"]),
        "training_windows": manifest["training_windows"],
        "preprocessing": {
            key: config[key]
            for key in (
                "sample_interval_minutes",
                "smoothing_window_minutes",
                "max_lag_minutes",
                "lag_step_minutes",
                "variance_threshold",
            )
        },
        "n_components": model.n_components,
        "explained_variance_ratio": model.explained_variance_ratio.tolist(),
        "t2_limits": {str(key): value for key, value in model.t2_limits.items()},
        "q_limits": {str(key): value for key, value in model.q_limits.items()},
        "validation_windows": validation_summary.get("validation_windows", []),
        "validation_summary": validation_summary,
        "engineer_decision": manifest.get("engineer_decision"),
        "applicability_scope": manifest.get("applicability_scope"),
    }


def compare_model_versions(
    left_path: str | Path,
    right_path: str | Path,
) -> dict[str, Any]:
    left_model, left_manifest = load_model_package(left_path)
    right_model, right_manifest = load_model_package(right_path)
    left = _comparison_value(left_model, left_manifest)
    right = _comparison_value(right_model, right_manifest)
    fields = {
        key: {"left": left[key], "right": right[key], "equal": left[key] == right[key]}
        for key in left
    }
    return {
        "left": {"path": str(left_path), "model_id": left_manifest.get("model_id"), "version": left_manifest.get("version")},
        "right": {"path": str(right_path), "model_id": right_manifest.get("model_id"), "version": right_manifest.get("version")},
        "equal": all(item["equal"] for item in fields.values()),
        "fields": fields,
    }


def validate_publish_preconditions(
    source_path: str | Path,
    *,
    engineer_confirmation: object,
    applicability_scope: object,
) -> dict[str, Any]:
    source = Path(source_path)
    model, manifest = load_model_package(source)
    del model
    if manifest.get("model_purpose") != "normal_state" or manifest.get("model_status") != "validated":
        raise ValueError("只有normal_state/validated模型可以发布")
    if not any(window.get("enabled") for window in manifest.get("training_windows", [])):
        raise ValueError("发布前必须存在启用的训练窗口")
    summary = manifest.get("validation_summary")
    validation_windows = (
        summary.get("validation_windows") if isinstance(summary, Mapping) else None
    )
    if not isinstance(validation_windows, list) or not any(
        isinstance(window, Mapping) and window.get("enabled")
        for window in validation_windows
    ):
        raise ValueError("发布前必须存在启用的验证窗口")
    decision = manifest.get("engineer_decision")
    if not isinstance(decision, Mapping) or decision.get("decision") != "passed":
        raise ValueError("发布前必须有工程师passed结论")
    if engineer_confirmation is not True:
        raise ValueError("发布必须提供明确的工程师确认")
    if not _nonempty_scope(applicability_scope):
        raise ValueError("发布必须提供适用范围")
    integrity = verify_model_package_integrity(
        source, require_external=manifest.get("schema_version") == 4
    )
    return {
        "manifest": manifest,
        "integrity": integrity,
        "applicability_scope": applicability_scope,
    }


def publish_model_version(
    source_path: str | Path,
    registry_root: str | Path = DEFAULT_REGISTRY_DIR,
    *,
    engineer_confirmation: object,
    applicability_scope: object,
    engineer_comment: str = "",
) -> dict[str, Any]:
    source = Path(source_path)
    preconditions = validate_publish_preconditions(
        source,
        engineer_confirmation=engineer_confirmation,
        applicability_scope=applicability_scope,
    )
    manifest = preconditions["manifest"]
    source_sha256 = preconditions["integrity"]["sha256"]
    model_id = manifest.get("model_id") or _derived_model_id(source)
    parent_model_id = manifest.get("model_id") or f"legacy-{source_sha256[:16]}"
    parent_version = manifest.get("version") or "legacy"
    model, source_manifest = load_model_package(source)
    resolved_model_id = _safe_model_id(str(model_id))
    latest_parent = _latest_registry_parent(registry_root, resolved_model_id)
    if latest_parent is not None:
        parent_model_id, parent_version = latest_parent
    result = _write_registry_version(
        model,
        source_manifest,
        registry_root,
        model_id=resolved_model_id,
        parent_model_id=str(parent_model_id),
        parent_version=str(parent_version),
        model_purpose="normal_state",
        model_status="published",
        engineer_comment=engineer_comment,
        applicability_scope=applicability_scope,
        source_sha256=source_sha256,
    )
    result["published_from_sha256"] = source_sha256
    return result
