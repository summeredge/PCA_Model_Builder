from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any

from .compat import validate_loadable_model_semantics
from .model_io import (
    _write_model_package,
    load_model_package,
    model_package_sha256,
)
from .validation import classify_validation_evidence, normalize_and_validate_validation_evidence


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


def _derived_model_id(source_path: Path, manifest: Mapping[str, Any]) -> tuple[str, str]:
    name = str(manifest.get("config", {}).get("model_name", "")).strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")[:80]
    if normalized and _MODEL_ID_PATTERN.fullmatch(normalized):
        return normalized, "model_name"
    return f"model-{model_package_sha256(source_path)[:16]}", "sha256_fallback"


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
) -> dict[str, Any] | None:
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
        try:
            return _resolve_registry_parent(registry_root, model_id, version_dir.name)
        except ValueError:
            continue
    return None


def _validate_parent_package(
    package: Path,
    manifest: Mapping[str, Any],
    model_id: str,
    version: str,
) -> None:
    if (
        package.name != "model.pcamodel"
        or manifest.get("schema_version") != 4
        or manifest.get("model_id") != model_id
        or manifest.get("version") != version
        or package.parent.name != version
        or package.parent.parent.name != model_id
        or manifest.get("model_purpose") != "normal_state"
        or manifest.get("model_status") not in {"validated", "published"}
    ):
        raise ValueError("父模型版本目录、manifest或状态无效")
    if classify_validation_evidence(manifest.get("validation_summary")) != "current":
        raise ValueError("旧验证证据仅支持只读查看，请重新执行完整验证")


def _resolve_registry_parent(
    registry_root: str | Path, model_id: str, version: str | None = None
) -> dict[str, Any] | None:
    if version is None:
        return _latest_registry_parent(registry_root, model_id)
    _version_number(version)
    package = Path(registry_root) / model_id / version / "model.pcamodel"
    try:
        validated = validate_registry_package(
            package,
            registry_root=registry_root,
            require_external=True,
            allowed_statuses={("normal_state", "validated"), ("normal_state", "published")},
        )
        model, manifest = validated["model"], validated["manifest"]
        integrity = validated["integrity"]
        _validate_parent_package(package, manifest, model_id, version)
    except (OSError, ValueError) as error:
        raise ValueError(f"指定父模型版本无效：{model_id}/{version}：{error}") from error
    return {
        "model_id": model_id,
        "version": version,
        "path": package,
        "model": model,
        "manifest": manifest,
        "integrity": integrity,
    }


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


def validate_registry_package(
    package_path: str | Path,
    *,
    registry_root: str | Path,
    require_external: bool = True,
    allowed_statuses: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate package integrity and its identity inside one registry."""
    package = Path(package_path).resolve()
    registry = Path(registry_root).resolve()
    if package.name != "model.pcamodel":
        raise ValueError("registry模型包文件名必须为model.pcamodel")
    try:
        relative = package.relative_to(registry)
    except ValueError as error:
        raise ValueError("模型包不在指定registry内") from error
    if len(relative.parts) != 3 or _VERSION_PATTERN.fullmatch(relative.parts[1]) is None:
        raise ValueError("模型版本目录结构无效")
    directory_model_id, directory_version, _ = relative.parts
    model, manifest = load_model_package(package)
    if manifest.get("schema_version") != 4:
        raise ValueError("registry模型包schema必须为4")
    integrity = verify_model_package_integrity(package, require_external=require_external)
    if (
        manifest.get("model_id") != directory_model_id
        or manifest.get("version") != directory_version
    ):
        raise ValueError("模型版本目录与manifest不一致")
    parent_model_id = manifest.get("parent_model_id")
    parent_version = manifest.get("parent_version")
    if (parent_model_id is None) != (parent_version is None):
        raise ValueError("模型父版本字段不完整")
    if parent_model_id is not None:
        if parent_model_id != directory_model_id:
            raise ValueError("父模型必须属于相同model_id")
        if not isinstance(parent_version, str) or _VERSION_PATTERN.fullmatch(parent_version) is None:
            raise ValueError("模型父版本号无效")
        if _version_number(parent_version) >= _version_number(directory_version):
            raise ValueError("父模型版本必须早于当前版本")
    validate_loadable_model_semantics(manifest["model_purpose"], manifest["model_status"])
    if allowed_statuses is not None and (
        manifest["model_purpose"], manifest["model_status"]
    ) not in allowed_statuses:
        raise ValueError("模型版本生命周期状态无效")
    return {
        "path": package,
        "model": model,
        "manifest": manifest,
        "integrity": integrity,
    }


def _record(package_path: Path, manifest: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": manifest.get("model_id"),
        "version": manifest.get("version"),
        "parent_model_id": manifest.get("parent_model_id"),
        "parent_version": manifest.get("parent_version"),
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "validation_evidence_status": manifest.get("validation_evidence_status"),
        "created_at": manifest.get("created_at"),
        "software_version": manifest.get("software_version"),
        "engineer_comment": manifest.get("engineer_comment", ""),
        "applicability_scope": manifest.get("applicability_scope"),
        "file_hashes": manifest.get("file_hashes"),
        "published_from": manifest.get("published_from"),
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
    as_existing_version: bool | None = None,
) -> dict[str, Any]:
    """Create one immutable schema-v4 copy without changing the source."""
    source = Path(source_path)
    model, manifest = load_model_package(source)
    if manifest.get("schema_version") == 4:
        verify_model_package_integrity(source, require_external=True)
    if (
        manifest.get("model_status") in {"validated", "published"}
        and classify_validation_evidence(manifest.get("validation_summary")) != "current"
    ):
        raise ValueError("旧验证证据仅支持只读查看，不能创建新的模型版本")
    source_sha256 = model_package_sha256(source)
    derived_id, id_source = _derived_model_id(source, manifest)
    resolved_model_id = _safe_model_id(model_id or manifest.get("model_id") or derived_id)
    purpose = model_purpose or str(manifest["model_purpose"])
    status = model_status or str(manifest["model_status"])
    validate_loadable_model_semantics(purpose, status)
    effective_manifest = dict(manifest)
    effective_manifest["model_purpose"] = purpose
    effective_manifest["model_status"] = status
    if status == "published":
        raise ValueError("published版本只能通过显式发布操作创建")
    if (parent_model_id is None) != (parent_version is None):
        raise ValueError("parent_model_id和parent_version必须同时提供")
    if parent_model_id is not None and parent_model_id != resolved_model_id:
        raise ValueError("父模型必须属于相同model_id")
    model_root = Path(registry_root) / resolved_model_id
    family_exists = model_root.is_dir()
    if not family_exists and parent_version is not None:
        raise ValueError("新模型族不得指定parent_version")
    if as_existing_version is False and family_exists:
        raise ValueError("model_id已存在，请明确选择作为现有模型的新版本")
    if as_existing_version is True and not family_exists:
        raise ValueError("所选现有model_id不存在")
    if family_exists and as_existing_version is not True and parent_version is None:
        raise ValueError("model_id已存在，请明确选择作为现有模型的新版本")
    selected_parent = None
    if family_exists:
        selected_parent = _resolve_registry_parent(
            registry_root, resolved_model_id, parent_version
        )
        if selected_parent is None:
            raise ValueError("模型族存在，但没有完整性通过的有效父版本")
    if selected_parent is not None:
        _validate_version_compatibility(
            model,
            effective_manifest,
            selected_parent["model"],
            selected_parent["manifest"],
        )
        parent_model_id = resolved_model_id
        parent_version = selected_parent["version"]
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
        published_from=None,
    )
    result["model_id_source"] = "explicit" if model_id else ("manifest" if manifest.get("model_id") else id_source)
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
    published_from: Mapping[str, Any] | None,
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
    if published_from is not None:
        metadata["published_from"] = dict(published_from)
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
        _, stored_manifest = load_model_package(package)
        integrity = verify_model_package_integrity(package, require_external=True)
        if package.name != "model.pcamodel" or version_dir.parent.name != stored_manifest.get("model_id") or version_dir.name != stored_manifest.get("version"):
            raise ValueError("模型版本目录与manifest不一致")
        result = _record(package, stored_manifest, integrity)
        result["source_sha256"] = source_sha256
        result["file_hashes"] = stored_manifest["file_hashes"]
        result["integrity"]["sha256"] = integrity_sha256
        return result
    except Exception as original_error:
        cleanup_error: Exception | None = None
        try:
            shutil.rmtree(version_dir)
            if version_dir.exists():
                raise OSError("删除后目录仍然存在")
            model_root = version_dir.parent
            if model_root.is_dir() and not any(model_root.iterdir()):
                try:
                    model_root.rmdir()
                except OSError:
                    pass
        except Exception as error:
            cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "模型版本创建失败，且无法清理无效目录：\n"
                f"原始错误：{original_error}\n清理错误：{cleanup_error}\n残留路径：{version_dir}"
            ) from original_error
        raise


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
                validated = validate_registry_package(
                    package, registry_root=registry, require_external=True
                )
                records.append(
                    _record(package, validated["manifest"], validated["integrity"])
                )
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
        "validation_evidence_status": manifest.get("validation_evidence_status"),
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
        "published_from": manifest.get("published_from"),
        "parent_model_id": manifest.get("parent_model_id"),
        "parent_version": manifest.get("parent_version"),
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
    if engineer_confirmation is not True:
        raise ValueError("发布必须提供明确的工程师确认")
    if not _nonempty_scope(applicability_scope):
        raise ValueError("发布必须提供适用范围")
    if not any(window.get("enabled") for window in manifest.get("training_windows", [])):
        raise ValueError("发布前必须存在启用的训练窗口")
    summary = manifest.get("validation_summary")
    evidence_kind = classify_validation_evidence(summary)
    if evidence_kind == "legacy":
        raise ValueError("旧验证证据仅支持只读查看，请重新执行完整验证后发布")
    if evidence_kind != "current":
        raise ValueError("验证证据无效，请重新执行完整验证后发布")
    normalize_and_validate_validation_evidence(summary)
    decision = manifest.get("engineer_decision")
    if not isinstance(decision, Mapping) or decision.get("decision") != "passed":
        raise ValueError("发布前必须有工程师passed结论")
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
    model_id: str | None = None,
    parent_version: str | None = None,
    as_existing_version: bool | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    if parent_version is not None and model_id is None:
        raise ValueError("--parent-version只能与--model-id同时使用")
    if parent_version is not None:
        _version_number(parent_version)
    preconditions = validate_publish_preconditions(
        source,
        engineer_confirmation=engineer_confirmation,
        applicability_scope=applicability_scope,
    )
    manifest = preconditions["manifest"]
    source_sha256 = preconditions["integrity"]["sha256"]
    derived_id, id_source = _derived_model_id(source, manifest)
    resolved_model_id = _safe_model_id(model_id or manifest.get("model_id") or derived_id)
    parent_model_id = None
    resolved_parent_version = None
    model, source_manifest = load_model_package(source)
    model_root = Path(registry_root) / resolved_model_id
    family_exists = model_root.is_dir()
    selected_parent = (
        _resolve_registry_parent(registry_root, resolved_model_id, parent_version)
        if parent_version is not None and family_exists
        else _resolve_registry_parent(registry_root, resolved_model_id)
    )
    if parent_version is not None and not family_exists:
        raise ValueError("新模型族不得指定parent_version")
    if family_exists and selected_parent is None:
        raise ValueError("模型族存在，但没有完整性通过的有效父版本")
    if as_existing_version is False and family_exists:
        raise ValueError("model_id已存在，请明确选择作为现有模型的新版本")
    if as_existing_version is True and not family_exists:
        raise ValueError("所选现有model_id不存在")
    if selected_parent is not None:
        latest_model, latest_manifest = selected_parent["model"], selected_parent["manifest"]
        _validate_version_compatibility(model, source_manifest, latest_model, latest_manifest)
        parent_model_id = resolved_model_id
        resolved_parent_version = selected_parent["version"]
    published_from = {
        "sha256": source_sha256,
        "filename": source.name,
        "model_id": manifest.get("model_id") if manifest.get("schema_version") == 4 else None,
        "version": manifest.get("version") if manifest.get("schema_version") == 4 else None,
        "schema_version": manifest["schema_version"],
    }
    result = _write_registry_version(
        model,
        source_manifest,
        registry_root,
        model_id=resolved_model_id,
        parent_model_id=str(parent_model_id) if parent_model_id is not None else None,
        parent_version=str(resolved_parent_version) if resolved_parent_version is not None else None,
        model_purpose="normal_state",
        model_status="published",
        engineer_comment=engineer_comment,
        applicability_scope=applicability_scope,
        source_sha256=source_sha256,
        published_from=published_from,
    )
    result["published_from_sha256"] = source_sha256
    result["model_id_source"] = "explicit" if model_id else ("manifest" if manifest.get("model_id") else id_source)
    return result


def _validate_version_compatibility(model: Any, manifest: Mapping[str, Any], existing_model: Any, existing_manifest: Mapping[str, Any]) -> None:
    config, existing = manifest["config"], existing_manifest["config"]
    differences = []
    if manifest["model_purpose"] != existing_manifest["model_purpose"]:
        differences.append("model_purpose")
    if list(model.feature_names) != list(existing_model.feature_names):
        differences.append("feature_names")
    for field in ("tags", "sample_interval_minutes", "smoothing_window_minutes", "max_lag_minutes", "lag_step_minutes"):
        if config.get(field) != existing.get(field):
            differences.append(field)
    if differences:
        raise ValueError(f"模型与现有model_id不兼容：{'、'.join(differences)}")
