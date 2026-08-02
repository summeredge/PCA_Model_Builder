from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import (
    copy_validated_model_package,
    load_model_package,
    model_package_sha256,
    save_model_package,
)
from pca_model_builder.model_registry import (
    compare_model_versions,
    create_model_version,
    list_model_versions,
    publish_model_version,
    validate_publish_preconditions,
    verify_model_package_integrity,
)


def _config() -> dict[str, object]:
    return {
        "model_name": "REGISTRY_TEST",
        "tags": ["A", "B", "C"],
        "timestamp_column": "time",
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 5,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "variance_threshold": 0.95,
    }


def _candidate(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        np.random.default_rng(123).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    path = tmp_path / "candidate.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        _config(),
        [["2026-01-01", "2026-01-02"]],
    )
    return path


def _validated(tmp_path: Path) -> Path:
    candidate = _candidate(tmp_path)
    validated = tmp_path / "validated.pcamodel"
    candidate_sha = model_package_sha256(candidate)
    summary = {
        "model_purpose": "normal_state",
        "model_status": "candidate",
        "normal_validation_complete": True,
        "known_abnormal_complete": True,
        "validation_windows": [
            {
                "id": "normal-001",
                "type": "normal_validation",
                "start": "2026-01-03T00:00:00",
                "end": "2026-01-03T01:00:00",
                "enabled": True,
                "comment": "normal",
            }
        ],
        "source_candidate_package": {
            "identifier": "candidate-run",
            "filename": candidate.name,
            "sha256": candidate_sha,
        },
    }
    copy_validated_model_package(
        candidate,
        validated,
        summary,
        {
            "decision": "passed",
            "comment": "approved",
            "reviewed_at": "2026-01-04T00:00:00+00:00",
        },
        "candidate-run",
    )
    return validated


def test_create_model_version_writes_schema_v4_and_external_hash(tmp_path):
    source = _candidate(tmp_path)
    source_bytes = source.read_bytes()
    record = create_model_version(source, tmp_path / "models")

    package = Path(record["path"])
    _, manifest = load_model_package(package)
    assert manifest["schema_version"] == 4
    assert manifest["version"] == "v0001"
    assert manifest["model_status"] == "candidate"
    assert manifest["file_hashes"]["arrays.npz"]["bytes"] > 0
    assert package.with_name("model.pcamodel.sha256").is_file()
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {"manifest.json", "arrays.npz"}
    assert source.read_bytes() == source_bytes
    assert verify_model_package_integrity(package, require_external=True)["valid"] is True
    source_model, _ = load_model_package(source)
    copied_model, _ = load_model_package(package)
    score_frame = pd.DataFrame(
        np.random.default_rng(123).normal(size=(8, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    pd.testing.assert_frame_equal(
        source_model.score(score_frame), copied_model.score(score_frame)
    )


def test_publish_creates_published_child_without_overwriting_source(tmp_path):
    source = _validated(tmp_path)
    source_bytes = source.read_bytes()
    registry = tmp_path / "models"

    first = publish_model_version(
        source,
        registry,
        engineer_confirmation=True,
        applicability_scope={"unit": "D330", "mode": "normal"},
        engineer_comment="release review",
    )
    _, first_manifest = load_model_package(first["path"])
    assert first_manifest["model_status"] == "published"
    assert first_manifest["model_purpose"] == "normal_state"
    assert first_manifest["parent_version"] == "legacy"
    assert source.read_bytes() == source_bytes

    second = publish_model_version(
        source,
        registry,
        engineer_confirmation=True,
        applicability_scope="D330 normal",
    )
    assert Path(first["path"]).read_bytes() != b""
    assert second["version"] == "v0002"
    assert load_model_package(second["path"])[1]["parent_version"] == "v0001"
    assert len(list_model_versions(registry)) == 2
    assert load_model_package(source)[1]["model_status"] == "validated"


def test_registry_reserves_unique_versions_for_concurrent_copies(tmp_path):
    source = _candidate(tmp_path)
    registry = tmp_path / "models"
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(
            executor.map(
                lambda _: create_model_version(source, registry),
                range(2),
            )
        )
    assert {record["version"] for record in records} == {"v0001", "v0002"}
    assert len(list_model_versions(registry)) == 2


def test_published_source_is_not_a_direct_copy_target(tmp_path):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    published = publish_model_version(
        source,
        registry,
        engineer_confirmation=True,
        applicability_scope="D330",
    )
    with pytest.raises(ValueError, match="published版本"):
        create_model_version(
            published["path"],
            registry,
            model_status="published",
        )


def test_publish_preconditions_reject_missing_confirmation_scope_and_evidence(tmp_path):
    source = _validated(tmp_path)
    with pytest.raises(ValueError, match="明确的工程师确认"):
        validate_publish_preconditions(
            source, engineer_confirmation=False, applicability_scope="D330"
        )
    with pytest.raises(ValueError, match="适用范围"):
        validate_publish_preconditions(
            source, engineer_confirmation=True, applicability_scope=""
        )

    _, manifest = load_model_package(source)
    manifest["engineer_decision"]["decision"] = "failed"
    with zipfile.ZipFile(source) as archive:
        arrays = archive.read("arrays.npz")
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="decision must be passed"):
        validate_publish_preconditions(
            source, engineer_confirmation=True, applicability_scope="D330"
        )


def test_compare_versions_reports_required_fields(tmp_path):
    source = _candidate(tmp_path)
    registry = tmp_path / "models"
    first = create_model_version(source, registry)
    second = create_model_version(source, registry)
    comparison = compare_model_versions(first["path"], second["path"])
    assert comparison["equal"] is True
    assert {
        "feature_names",
        "tags",
        "training_windows",
        "preprocessing",
        "n_components",
        "explained_variance_ratio",
        "t2_limits",
        "q_limits",
        "validation_windows",
        "validation_summary",
        "engineer_decision",
        "applicability_scope",
    } <= set(comparison["fields"])


def test_integrity_detects_tampered_arrays_and_sidecar(tmp_path):
    source = _candidate(tmp_path)
    record = create_model_version(source, tmp_path / "models")
    package = Path(record["path"])
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = bytearray(archive.read("arrays.npz"))
    arrays[-1] ^= 1
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", bytes(arrays))
    with pytest.raises(ValueError, match="arrays.npz integrity"):
        verify_model_package_integrity(package, require_external=True)
