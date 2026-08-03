from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from io import BytesIO
import hashlib
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
    validate_loadable_model_semantics,
    validate_new_model_semantics,
)
from .windows import normalize_training_windows
from .validation import (
    normalize_and_validate_validation_evidence,
    validation_artifact_metadata,
)


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
_MANIFEST_FIELDS_V4 = _MANIFEST_FIELDS_V2 | {
    "model_id",
    "version",
    "parent_model_id",
    "parent_version",
    "created_at",
    "software_version",
    "engineer_comment",
    "applicability_scope",
    "file_hashes",
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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    _write_model_package(
        path,
        model,
        config=config,
        training_windows=training_windows,
        model_purpose=model_purpose,
        model_status=model_status,
        validation_summary=validation_summary,
        engineer_decision=engineer_decision,
        source_candidate_package=source_candidate_package,
    )


def _write_model_package(
    path: str | Path,
    model: DPCAModel,
    config: dict[str, Any],
    training_windows: list[object],
    model_purpose: str,
    model_status: str,
    validation_summary: dict[str, Any] | None = None,
    engineer_decision: dict[str, Any] | None = None,
    source_candidate_package: dict[str, str] | None = None,
    schema_version: int = SCHEMA_VERSION,
    lifecycle_metadata: Mapping[str, Any] | None = None,
) -> None:
    validate_loadable_model_semantics(model_purpose, model_status)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": schema_version,
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
    if schema_version >= 4:
        if lifecycle_metadata is None:
            raise ValueError("schema v4 model package lifecycle metadata is required")
        manifest.update(dict(lifecycle_metadata))
    arrays = BytesIO()
    np.savez_compressed(
        arrays,
        mean=model.mean,
        scale=model.scale,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
    )
    arrays_bytes = arrays.getvalue()
    if schema_version >= 4:
        manifest["file_hashes"] = {
            "arrays.npz": {
                "sha256": hashlib.sha256(arrays_bytes).hexdigest(),
                "bytes": len(arrays_bytes),
            }
        }

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
            package.writestr("arrays.npz", arrays_bytes)
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
    scores_path: str | Path | None = None,
    contributions_path: str | Path | None = None,
) -> None:
    source = Path(source_path)
    destination = Path(destination_path)
    if source.resolve() == destination.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    if not isinstance(source_identifier, str) or not source_identifier.strip():
        raise ValueError("source candidate identifier must be a non-empty string")
    model, manifest = load_model_package(source)
    if (
        manifest["model_purpose"] != "normal_state"
        or manifest["model_status"] != "candidate"
    ):
        raise ValueError("only normal_state/candidate models can become validated")
    actual_sha256 = model_package_sha256(source)
    _validate_review_evidence(
        source,
        manifest,
        validation_summary,
        engineer_decision,
        expected_identifier=source_identifier,
        actual_sha256=actual_sha256,
        scores_path=scores_path,
        contributions_path=contributions_path,
    )
    source_binding = dict(validation_summary["source_candidate_package"])
    source_binding["sha256"] = actual_sha256
    summary = dict(validation_summary)
    summary["source_candidate_package"] = source_binding
    _write_model_package(
        destination,
        model,
        config=dict(manifest["config"]),
        training_windows=manifest["training_windows"],
        model_purpose="normal_state",
        model_status="validated",
        validation_summary=summary,
        engineer_decision=engineer_decision,
        source_candidate_package={
            "identifier": source_identifier,
            "filename": source.name,
            "sha256": actual_sha256,
        },
    )


def model_package_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_validation_report_binding(
    candidate_path: str | Path,
    manifest: dict[str, Any],
    report: Mapping[str, Any],
    expected_identifier: str | None = None,
) -> str:
    source = Path(candidate_path)
    if manifest.get("model_purpose") != "normal_state" or manifest.get(
        "model_status"
    ) != "candidate":
        raise ValueError("验证报告与当前候选模型包不匹配")
    binding = report.get("source_candidate_package")
    if not isinstance(binding, Mapping):
        raise ValueError("验证报告与当前候选模型包不匹配")
    identifier = binding.get("identifier")
    filename = binding.get("filename")
    reported_sha256 = binding.get("sha256")
    if (
        not isinstance(identifier, str)
        or not identifier.strip()
        or filename != source.name
        or not isinstance(reported_sha256, str)
        or _SHA256_PATTERN.fullmatch(reported_sha256) is None
    ):
        raise ValueError("验证报告与当前候选模型包不匹配")
    if expected_identifier is not None and identifier != expected_identifier:
        raise ValueError("验证报告与当前候选模型包不匹配")
    if report.get("model_purpose") != "normal_state" or report.get(
        "model_status"
    ) != "candidate":
        raise ValueError("验证报告与当前候选模型包不匹配")
    if reported_sha256 != model_package_sha256(source):
        raise ValueError("验证报告与当前候选模型包不匹配")
    return reported_sha256


def _validate_review_evidence(
    source: Path,
    manifest: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    engineer_decision: Mapping[str, Any],
    expected_identifier: str,
    actual_sha256: str,
    scores_path: str | Path | None = None,
    contributions_path: str | Path | None = None,
) -> None:
    normalize_and_validate_validation_evidence(
        validation_summary,
        candidate_path=source,
        expected_identifier=expected_identifier,
        scores_path=scores_path,
        contributions_path=contributions_path,
        require_artifact_files=True,
    )
    if not isinstance(engineer_decision, Mapping):
        raise ValueError("验证人工结论不完整")
    if engineer_decision.get("decision") != "passed":
        raise ValueError("只有passed结论可以生成validated模型")
    if not isinstance(engineer_decision.get("comment"), str):
        raise ValueError("工程师备注必须是文本")
    _validate_reviewed_at(engineer_decision.get("reviewed_at"))
    binding = validation_summary.get("source_candidate_package")
    if not isinstance(binding, Mapping):
        raise ValueError("验证报告与当前候选模型包不匹配")
    if binding.get("identifier") != expected_identifier or binding.get("filename") != source.name:
        raise ValueError("验证报告与当前候选模型包不匹配")
    if binding.get("sha256") != actual_sha256:
        raise ValueError("验证报告与当前候选模型包不匹配")


def _validate_reviewed_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("工程师审查时间无效")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("工程师审查时间无效") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("工程师审查时间必须带时区")


def validate_validated_model_artifact(
    candidate_path: str | Path,
    validated_path: str | Path,
    report: Mapping[str, Any],
    expected_identifier: str | None = None,
) -> dict[str, Any]:
    candidate = Path(candidate_path)
    validated = Path(validated_path)
    _, candidate_manifest = load_model_package(candidate)
    candidate_sha256 = validate_validation_report_binding(
        candidate, candidate_manifest, report, expected_identifier
    )
    engineer_decision = report.get("engineer_decision")
    if not isinstance(engineer_decision, Mapping) or engineer_decision.get(
        "decision"
    ) != "passed":
        raise ValueError("当前验证报告不是passed结论")
    if not validated.is_file():
        raise ValueError("已验证模型工件不存在")
    _, validated_manifest = load_model_package(validated)
    source_package = validated_manifest["source_candidate_package"]
    if (
        validated_manifest.get("model_purpose") != "normal_state"
        or validated_manifest.get("model_status") != "validated"
        or source_package.get("sha256") != candidate_sha256
        or source_package.get("filename") != candidate.name
        or (
            expected_identifier is not None
            and source_package.get("identifier") != expected_identifier
        )
        or validated_manifest.get("engineer_decision") != dict(engineer_decision)
        or validated_manifest.get("validation_summary") != dict(report)
    ):
        raise ValueError("验证报告与当前已验证模型包不一致")
    return validated_manifest


def commit_validation_artifacts(
    candidate_path: str | Path,
    validated_path: str | Path,
    report_path: str | Path,
    report: Mapping[str, Any],
    engineer_decision: Mapping[str, Any],
    source_identifier: str,
    previous_report: Mapping[str, Any] | None = None,
    scores_path: str | Path | None = None,
    contributions_path: str | Path | None = None,
) -> None:
    """Atomically commit the report and optional validated copy as one review."""
    candidate = Path(candidate_path)
    validated = Path(validated_path)
    report_file = Path(report_path)
    if candidate.resolve() == validated.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(source_identifier, str) or not source_identifier.strip():
        raise ValueError("source candidate identifier must be a non-empty string")
    decision = engineer_decision.get("decision")
    if decision not in {"passed", "insufficient", "failed"}:
        raise ValueError("工程师结论无效")
    if validated.exists():
        if previous_report is None:
            raise ValueError("已有validated工件来源无法验证，拒绝覆盖")
        validate_validated_model_artifact(
            candidate,
            validated,
            previous_report,
            expected_identifier=source_identifier,
        )

    temporary_validated: Path | None = None
    temporary_report: Path | None = None
    output_backup: Path | None = None
    report_backup: Path | None = None
    output_installed = False
    report_installed = False
    try:
        if decision == "passed":
            temporary_validated = _reserve_temporary_path(validated, ".pcamodel.tmp")
            copy_validated_model_package(
                candidate,
                temporary_validated,
                validation_summary=dict(report),
                engineer_decision=dict(engineer_decision),
                source_identifier=source_identifier,
                scores_path=scores_path,
                contributions_path=contributions_path,
            )
        temporary_report = _reserve_temporary_path(report_file, ".json.tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if validated.exists():
            output_backup = _reserve_temporary_path(validated, ".bak")
            os.replace(validated, output_backup)
        if report_file.exists():
            report_backup = _reserve_temporary_path(report_file, ".bak")
            os.replace(report_file, report_backup)
        if temporary_validated is not None:
            os.replace(temporary_validated, validated)
            output_installed = True
        os.replace(temporary_report, report_file)
        report_installed = True
    except Exception:
        if output_installed and validated.exists():
            validated.unlink()
        if report_installed and report_file.exists():
            report_file.unlink()
        if output_backup is not None and output_backup.exists():
            validated.unlink(missing_ok=True)
            os.rename(output_backup, validated)
            output_backup = None
        if report_backup is not None and report_backup.exists():
            report_file.unlink(missing_ok=True)
            os.rename(report_backup, report_file)
            report_backup = None
        raise
    finally:
        for temporary in (
            temporary_validated,
            temporary_report,
            output_backup,
            report_backup,
        ):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def commit_validation_run_artifacts(
    candidate_path: str | Path,
    report_path: str | Path,
    scores_path: str | Path,
    contributions_path: str | Path,
    validated_path: str | Path,
    report: Mapping[str, Any],
    scores: pd.DataFrame,
    contributions: Any,
    timestamp_column: str,
    previous_report: Mapping[str, Any] | None = None,
    source_identifier: str | None = None,
) -> None:
    """Commit one complete validation run without exposing partial artifacts."""
    candidate = Path(candidate_path)
    report_file = Path(report_path)
    scores_file = Path(scores_path)
    contributions_file = Path(contributions_path)
    validated = Path(validated_path)
    if not isinstance(report, Mapping) or "engineer_decision" in report:
        raise ValueError("新的验证报告不得包含人工结论")
    if validated.resolve() == candidate.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    if validated.exists():
        if previous_report is None or not isinstance(source_identifier, str) or not source_identifier.strip():
            raise ValueError("已有validated工件来源无法验证，拒绝覆盖")
        validate_validated_model_artifact(
            candidate,
            validated,
            previous_report,
            expected_identifier=source_identifier,
        )

    for target in (report_file, scores_file, contributions_file, validated):
        target.parent.mkdir(parents=True, exist_ok=True)

    temporary_scores: Path | None = None
    temporary_contributions: Path | None = None
    temporary_report: Path | None = None
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        temporary_scores = _reserve_temporary_path(scores_file, ".csv.tmp")
        temporary_scores.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(
            temporary_scores,
            index_label=timestamp_column,
            encoding="utf-8-sig",
        )
        temporary_contributions = _reserve_temporary_path(
            contributions_file, ".json.tmp"
        )
        temporary_contributions.write_text(
            json.dumps(contributions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not isinstance(report, dict):
            raise ValueError("验证报告必须是可更新对象")
        report["validation_artifacts"] = {
            "scores": validation_artifact_metadata(
                temporary_scores, filename=scores_file.name
            ),
            "contributions": validation_artifact_metadata(
                temporary_contributions, filename=contributions_file.name
            ),
        }
        normalize_and_validate_validation_evidence(
            report,
            candidate_path=candidate,
            scores_path=temporary_scores,
            contributions_path=temporary_contributions,
            require_artifact_files=True,
            expected_identifier=source_identifier,
            allow_temporary_artifact_names=True,
        )
        temporary_report = _reserve_temporary_path(report_file, ".json.tmp")
        temporary_report.write_text(
            json.dumps(dict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for target in (report_file, scores_file, contributions_file, validated):
            if target.exists():
                backup = _reserve_temporary_path(target, ".bak")
                os.replace(target, backup)
                backups[target] = backup

        for temporary, target in (
            (temporary_report, report_file),
            (temporary_scores, scores_file),
            (temporary_contributions, contributions_file),
        ):
            if temporary is not None:
                os.replace(temporary, target)
                installed.append(target)
        temporary_report = None
        temporary_scores = None
        temporary_contributions = None
    except Exception:
        for target in reversed(installed):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                target.unlink(missing_ok=True)
                os.rename(backup, target)
        raise
    finally:
        for temporary in (
            temporary_report,
            temporary_scores,
            temporary_contributions,
            *backups.values(),
        ):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _reserve_temporary_path(target: Path, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def load_model_package(path: str | Path) -> tuple[DPCAModel, dict[str, Any]]:
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            if names != {"manifest.json", "arrays.npz"}:
                raise ValueError("model package has unexpected or missing files")
            manifest = json.loads(package.read("manifest.json"))
            _validate_manifest_structure(manifest)
            arrays_bytes = package.read("arrays.npz")
            _validate_array_file_hash(manifest, arrays_bytes)
            with np.load(
                BytesIO(arrays_bytes), allow_pickle=False
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
    if schema_version == 1:
        fields = _MANIFEST_FIELDS_V1
    elif schema_version == 4:
        fields = _MANIFEST_FIELDS_V4
    else:
        fields = _MANIFEST_FIELDS_V2
    missing = sorted(fields - set(manifest))
    if missing:
        raise ValueError(f"model package manifest is missing: {', '.join(missing)}")
    if schema_version not in {1, 2, SCHEMA_VERSION, 4}:
        raise ValueError("unsupported model package schema version")
    normalize_model_semantics(manifest)
    if manifest.get("model_status") == "published" and schema_version != 4:
        raise ValueError("published model packages require schema version 4")
    if schema_version == 4:
        _validate_v4_metadata(manifest)
    if manifest.get("model_status") in {"validated", "published"}:
        _validate_validated_evidence(manifest)
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


def _validate_v4_metadata(manifest: dict[str, Any]) -> None:
    model_id_pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}"
    if not isinstance(manifest.get("model_id"), str) or re.fullmatch(model_id_pattern, manifest["model_id"]) is None:
        raise ValueError("model package model_id must be a non-empty string")
    version = manifest.get("version")
    if not isinstance(version, str) or re.fullmatch(r"v\d{4}", version) is None:
        raise ValueError("model package version is invalid")
    parent_model_id = manifest.get("parent_model_id")
    parent_version = manifest.get("parent_version")
    if (parent_model_id is None) != (parent_version is None):
        raise ValueError("model package parent reference is incomplete")
    if parent_model_id is not None:
        if not isinstance(parent_model_id, str) or re.fullmatch(model_id_pattern, parent_model_id) is None:
            raise ValueError("model package parent_model_id is invalid")
        if not isinstance(parent_version, str) or re.fullmatch(r"v\d{4}", parent_version) is None:
            raise ValueError("model package parent_version is invalid")
        if parent_model_id == manifest["model_id"] and parent_version == version:
            raise ValueError("model package parent reference cannot refer to itself")
    if not isinstance(manifest.get("software_version"), str) or not manifest[
        "software_version"
    ].strip():
        raise ValueError("model package software_version must be a non-empty string")
    _validate_reviewed_at(manifest.get("created_at"))
    if not isinstance(manifest.get("engineer_comment"), str):
        raise ValueError("model package engineer_comment must be text")
    scope = manifest.get("applicability_scope")
    if not isinstance(scope, (str, list, dict)):
        raise ValueError("model package applicability_scope is invalid")
    if manifest.get("model_status") == "published" and not (
        (isinstance(scope, str) and scope.strip())
        or (isinstance(scope, (list, dict)) and bool(scope))
    ):
        raise ValueError("published model package applicability_scope is required")
    if manifest.get("model_status") == "published":
        _validate_published_from(manifest.get("published_from"))
    if not isinstance(manifest.get("file_hashes"), dict):
        raise ValueError("model package file_hashes are required")


def _validate_published_from(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("published model package published_from is required")
    if set(value) != {"sha256", "filename", "model_id", "version", "schema_version"}:
        raise ValueError("published model package published_from is invalid")
    if not isinstance(value.get("sha256"), str) or _SHA256_PATTERN.fullmatch(value["sha256"]) is None:
        raise ValueError("published model package source SHA-256 is invalid")
    filename = value.get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("published model package source filename is invalid")
    schema_version = value.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version not in {1, 2, 3, 4}:
        raise ValueError("published model package source schema version is invalid")
    model_id, version = value.get("model_id"), value.get("version")
    if (model_id is None) != (version is None):
        raise ValueError("published model package source identity is incomplete")
    if model_id is not None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("published model package source model_id is invalid")
        if not isinstance(version, str) or re.fullmatch(r"v\d{4}", version) is None:
            raise ValueError("published model package source version is invalid")


def _validate_array_file_hash(manifest: Mapping[str, Any], arrays_bytes: bytes) -> None:
    if manifest.get("schema_version") != 4:
        return
    hashes = manifest.get("file_hashes")
    arrays_hash = hashes.get("arrays.npz") if isinstance(hashes, Mapping) else None
    if not isinstance(arrays_hash, Mapping):
        raise ValueError("model package arrays.npz hash is required")
    if (
        not isinstance(arrays_hash.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(arrays_hash["sha256"]) is None
        or not isinstance(arrays_hash.get("bytes"), int)
        or isinstance(arrays_hash.get("bytes"), bool)
        or arrays_hash["bytes"] != len(arrays_bytes)
        or arrays_hash["sha256"] != hashlib.sha256(arrays_bytes).hexdigest()
    ):
        raise ValueError("model package arrays.npz integrity check failed")


def _validate_validated_evidence(manifest: dict[str, Any]) -> None:
    validation_summary = manifest.get("validation_summary")
    engineer_decision = manifest.get("engineer_decision")
    source_package = manifest.get("source_candidate_package")
    if not isinstance(validation_summary, dict):
        raise ValueError("validated model package validation_summary is required")
    if not isinstance(engineer_decision, dict):
        raise ValueError("validated model package engineer_decision is required")
    if not isinstance(source_package, dict):
        raise ValueError("validated model package source_candidate_package is required")
    if engineer_decision.get("decision") != "passed":
        raise ValueError("validated model package decision must be passed")
    if not isinstance(engineer_decision.get("comment"), str):
        raise ValueError("validated model package decision comment is invalid")
    _validate_reviewed_at(engineer_decision.get("reviewed_at"))
    normalize_and_validate_validation_evidence(validation_summary)
    summary_binding = validation_summary.get("source_candidate_package")
    if not isinstance(summary_binding, dict):
        raise ValueError("validated model package validation binding is missing")
    for binding in (source_package, summary_binding):
        if (
            not isinstance(binding.get("identifier"), str)
            or not binding["identifier"].strip()
            or not isinstance(binding.get("filename"), str)
            or not binding["filename"].strip()
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_PATTERN.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("validated model package source binding is invalid")
    if summary_binding["sha256"] != source_package["sha256"]:
        raise ValueError("validated model package source binding is inconsistent")
    if (
        summary_binding["identifier"] != source_package["identifier"]
        or summary_binding["filename"] != source_package["filename"]
    ):
        raise ValueError("validated model package source binding is inconsistent")


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
