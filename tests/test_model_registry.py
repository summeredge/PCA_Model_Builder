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


def _candidate(tmp_path: Path, *, seed: int = 123, model_name: str = "REGISTRY_TEST") -> Path:
    frame = pd.DataFrame(
        np.random.default_rng(seed).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    path = tmp_path / "candidate.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        {**_config(), "model_name": model_name},
        [["2026-01-01", "2026-01-02"]],
    )
    return path


def _validated(tmp_path: Path, *, seed: int = 123, model_name: str = "REGISTRY_TEST") -> Path:
    candidate = _candidate(tmp_path, seed=seed, model_name=model_name)
    validated = tmp_path / f"validated-{seed}.pcamodel"
    candidate_sha = model_package_sha256(candidate)
    scores = tmp_path / f"scores-{seed}.csv"
    contributions = tmp_path / f"contributions-{seed}.json"
    scores.write_text("time,status\n2026-01-03,normal\n2026-01-04,normal\n", encoding="utf-8")
    contributions.write_text("[]", encoding="utf-8")
    from pca_model_builder.validation import validation_artifact_metadata
    windows = [
        {"id": "normal-001", "type": "normal_validation", "start": "2026-01-03T00:00:00", "end": "2026-01-03T01:00:00", "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": "2026-01-03T02:00:00", "end": "2026-01-03T03:00:00", "enabled": True, "comment": "abnormal"},
    ]
    summaries = [{**window, "status": "scored", "scored_rows": 1, "expected_rows": 1, "coverage": 1.0, "t2_exceedance_95": 0.0, "t2_exceedance_99": 0.0, "spe_exceedance_95": 0.0, "spe_exceedance_99": 0.0, "maximum_t2": 1.0, "maximum_spe": 1.0, "event_count": 0, "longest_event_minutes": 0} for window in windows]
    summary = {
        "validation_schema_version": 2, "model_purpose": "normal_state", "model_status": "candidate",
        "normal_validation_complete": True, "known_abnormal_complete": True,
        "validation_windows": windows, "validation_window_summaries": summaries,
        "scored_rows": 2, "status_counts": {"normal": 2}, "maximum_t2": 1.0, "maximum_spe": 1.0,
        "validation_artifacts": {"scores": validation_artifact_metadata(scores), "contributions": validation_artifact_metadata(contributions)},
        "source_candidate_package": {"identifier": "candidate-run", "filename": candidate.name, "sha256": candidate_sha},
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
        scores_path=scores,
        contributions_path=contributions,
    )
    return validated


def _candidate_with_config(
    tmp_path: Path,
    *,
    name: str,
    tags: list[str] | None = None,
    sample_interval_minutes: int = 5,
    smoothing_window_minutes: int = 5,
    max_lag_minutes: int = 0,
    lag_step_minutes: int = 5,
    model_purpose: str = "normal_state",
    model_status: str = "candidate",
) -> Path:
    selected_tags = tags or ["A", "B", "C"]
    lags = range(0, max_lag_minutes + 1, lag_step_minutes)
    columns = [f"{tag}__lag_{lag:03d}min" for lag in lags for tag in selected_tags]
    frame = pd.DataFrame(
        np.random.default_rng(321).normal(size=(100, len(columns))), columns=columns
    )
    path = tmp_path / f"{name}.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=min(2, len(columns) - 1)),
        {
            **_config(),
            "model_name": name,
            "tags": selected_tags,
            "sample_interval_minutes": sample_interval_minutes,
            "smoothing_window_minutes": smoothing_window_minutes,
            "max_lag_minutes": max_lag_minutes,
            "lag_step_minutes": lag_step_minutes,
        },
        [["2026-01-01", "2026-01-02"]],
        model_purpose=model_purpose,
        model_status=model_status,
    )
    return path


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
    assert first_manifest["parent_version"] is None
    assert first_manifest["published_from"] == {
        "sha256": model_package_sha256(source),
        "filename": source.name,
        "model_id": None,
        "version": None,
        "schema_version": 3,
    }
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


def test_registry_validated_version_is_published_as_its_child(tmp_path):
    registry = tmp_path / "models"
    validated = create_model_version(
        _validated(tmp_path), registry, model_id="D330_DPCA"
    )
    source = Path(validated["path"])
    source_bytes = source.read_bytes()
    source_sha256 = model_package_sha256(source)

    published = publish_model_version(
        source,
        registry,
        model_id="D330_DPCA",
        as_existing_version=True,
        engineer_confirmation=True,
        applicability_scope="D330",
    )

    _, manifest = load_model_package(published["path"])
    assert (published["version"], manifest["parent_version"]) == ("v0002", "v0001")
    assert manifest["parent_model_id"] == "D330_DPCA"
    assert manifest["published_from"] == {
        "sha256": source_sha256,
        "filename": "model.pcamodel",
        "model_id": "D330_DPCA",
        "version": "v0001",
        "schema_version": 4,
    }
    assert source.read_bytes() == source_bytes
    assert verify_model_package_integrity(published["path"], require_external=True)["valid"]
    assert [item["version"] for item in list_model_versions(registry)] == ["v0001", "v0002"]


def test_create_model_version_cross_registry_copy_has_no_parent(tmp_path):
    first = create_model_version(
        _validated(tmp_path), tmp_path / "registry-a", model_id="D330_DPCA"
    )
    copied = create_model_version(first["path"], tmp_path / "registry-b")
    _, manifest = load_model_package(copied["path"])
    assert copied["version"] == "v0001"
    assert manifest["parent_model_id"] is None
    assert manifest["parent_version"] is None


def test_create_model_version_validates_explicit_parent_before_reserving(tmp_path):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    with pytest.raises(ValueError, match="指定父模型版本无效"):
        create_model_version(
            source,
            registry,
            model_id="D330_DPCA",
            parent_model_id="D330_DPCA",
            parent_version="v9999",
        )
    assert not (registry / "D330_DPCA").exists()


def test_create_model_version_appends_to_real_compatible_parent(tmp_path):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    first = create_model_version(source, registry, model_id="D330_DPCA")
    first_bytes = Path(first["path"]).read_bytes()
    second = create_model_version(
        source,
        registry,
        model_id="D330_DPCA",
        parent_model_id="D330_DPCA",
        parent_version="v0001",
    )
    _, manifest = load_model_package(second["path"])
    assert manifest["parent_model_id"] == "D330_DPCA"
    assert manifest["parent_version"] == "v0001"
    assert Path(first["path"]).read_bytes() == first_bytes
    assert verify_model_package_integrity(second["path"], require_external=True)["valid"]


@pytest.mark.parametrize(
    ("field", "changes"),
    [
        ("tags", {"tags": ["B", "A", "C"]}),
        ("feature_names", {"tags": ["A", "B", "D"]}),
        ("sample_interval_minutes", {"sample_interval_minutes": 1}),
        ("smoothing_window_minutes", {"smoothing_window_minutes": 10}),
        ("max_lag_minutes", {"max_lag_minutes": 5}),
        ("lag_step_minutes", {"lag_step_minutes": 10}),
    ],
)
def test_create_model_version_rejects_each_incompatible_field_without_residue(
    tmp_path, field, changes
):
    registry = tmp_path / "models"
    source = _validated(tmp_path)
    first = create_model_version(source, registry, model_id="D330_DPCA")
    parent_bytes = Path(first["path"]).read_bytes()
    source_bytes = source.read_bytes()
    incompatible = _candidate_with_config(tmp_path, name=f"different-{field}", **changes)
    incompatible_bytes = incompatible.read_bytes()

    with pytest.raises(ValueError, match=field):
        create_model_version(
            incompatible,
            registry,
            model_id="D330_DPCA",
            parent_model_id="D330_DPCA",
            parent_version="v0001",
        )

    assert not (registry / "D330_DPCA" / "v0002").exists()
    assert Path(first["path"]).read_bytes() == parent_bytes
    assert source.read_bytes() == source_bytes
    assert incompatible.read_bytes() == incompatible_bytes
    assert [item["version"] for item in list_model_versions(registry)] == ["v0001"]


def test_create_model_version_uses_effective_purpose_for_parent_compatibility(tmp_path):
    registry = tmp_path / "models"
    parent = create_model_version(_validated(tmp_path), registry, model_id="D330_DPCA")
    parent_bytes = Path(parent["path"]).read_bytes()
    normal_source = _candidate(tmp_path)
    normal_bytes = normal_source.read_bytes()

    with pytest.raises(ValueError, match="model_purpose"):
        create_model_version(
            normal_source,
            registry,
            model_id="D330_DPCA",
            parent_model_id="D330_DPCA",
            parent_version="v0001",
            model_purpose="exploratory",
            model_status="draft",
        )
    assert not (registry / "D330_DPCA" / "v0002").exists()
    assert Path(parent["path"]).read_bytes() == parent_bytes
    assert normal_source.read_bytes() == normal_bytes

    exploratory = _candidate_with_config(
        tmp_path,
        name="exploratory-source",
        model_purpose="exploratory",
        model_status="draft",
    )
    exploratory_bytes = exploratory.read_bytes()
    child = create_model_version(
        exploratory,
        registry,
        model_id="D330_DPCA",
        parent_model_id="D330_DPCA",
        parent_version="v0001",
        model_purpose="normal_state",
        model_status="candidate",
    )
    _, child_manifest = load_model_package(child["path"])
    assert child_manifest["model_purpose"] == "normal_state"
    assert child_manifest["model_status"] == "candidate"
    assert child_manifest["parent_version"] == "v0001"
    assert exploratory.read_bytes() == exploratory_bytes


@pytest.mark.parametrize(
    ("parent_semantics", "allowed"),
    [
        ("validated", True),
        ("published", True),
        ("candidate", False),
        ("draft", False),
    ],
)
def test_create_model_version_enforces_parent_state_contract(
    tmp_path, parent_semantics, allowed
):
    registry = tmp_path / "models"
    if parent_semantics == "validated":
        create_model_version(_validated(tmp_path), registry, model_id="D330_DPCA")
    elif parent_semantics == "published":
        publish_model_version(
            _validated(tmp_path), registry, model_id="D330_DPCA",
            engineer_confirmation=True, applicability_scope="D330",
        )
    elif parent_semantics == "candidate":
        create_model_version(_candidate(tmp_path), registry, model_id="D330_DPCA")
    else:
        create_model_version(
            _candidate_with_config(
                tmp_path, name="draft-parent",
                model_purpose="exploratory", model_status="draft",
            ),
            registry,
            model_id="D330_DPCA",
        )

    operation = lambda: create_model_version(
        _candidate(tmp_path), registry, model_id="D330_DPCA",
        parent_model_id="D330_DPCA", parent_version="v0001",
    )
    if allowed:
        assert operation()["version"] == "v0002"
    else:
        with pytest.raises(ValueError, match="父模型版本.*无效|manifest或状态无效"):
            operation()
        assert not (registry / "D330_DPCA" / "v0002").exists()


def test_create_model_version_rejects_cross_family_parent(tmp_path):
    with pytest.raises(ValueError, match="相同model_id"):
        create_model_version(
            _validated(tmp_path),
            tmp_path / "models",
            model_id="MODEL_A",
            parent_model_id="MODEL_B",
            parent_version="v0001",
        )


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
        "published_from",
        "parent_model_id",
        "parent_version",
    } <= set(comparison["fields"])


def test_explicit_model_id_builds_stable_retraining_chain(tmp_path):
    source = _validated(tmp_path, model_name="D330_DPCA_Model_V1")
    retrained = _validated(tmp_path, seed=456, model_name="D330_DPCA_Model_V2")
    registry = tmp_path / "models"
    first = publish_model_version(source, registry, model_id="D330_DPCA", engineer_confirmation=True, applicability_scope="D330")
    second = publish_model_version(retrained, registry, model_id="D330_DPCA", parent_version="v0001", engineer_confirmation=True, applicability_scope="D330")
    assert (first["model_id"], first["version"]) == ("D330_DPCA", "v0001")
    assert (second["model_id"], second["version"]) == ("D330_DPCA", "v0002")
    assert load_model_package(second["path"])[1]["parent_version"] == "v0001"
    assert model_package_sha256(first["path"]) != model_package_sha256(second["path"])
    assert verify_model_package_integrity(first["path"], require_external=True)["valid"]
    assert verify_model_package_integrity(second["path"], require_external=True)["valid"]


def test_explicit_parent_must_exist_and_does_not_reserve_version(tmp_path):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    publish_model_version(source, registry, model_id="D330_DPCA", engineer_confirmation=True, applicability_scope="D330")
    with pytest.raises(ValueError, match="指定父模型版本无效"):
        publish_model_version(source, registry, model_id="D330_DPCA", parent_version="v9999", engineer_confirmation=True, applicability_scope="D330")
    assert not (registry / "D330_DPCA" / "v0002").exists()


def test_automatic_parent_skips_corrupt_latest_version(tmp_path):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    first = publish_model_version(source, registry, model_id="D330_DPCA", engineer_confirmation=True, applicability_scope="D330")
    second = publish_model_version(source, registry, model_id="D330_DPCA", engineer_confirmation=True, applicability_scope="D330")
    Path(f"{second['path']}.sha256").unlink()
    third = publish_model_version(source, registry, model_id="D330_DPCA", engineer_confirmation=True, applicability_scope="D330")
    assert load_model_package(third["path"])[1]["parent_version"] == first["version"]


@pytest.mark.parametrize("corrupt_latest", [False, True])
def test_automatic_parent_prefers_highest_valid_version_and_falls_back(
    tmp_path, corrupt_latest
):
    source = _validated(tmp_path)
    registry = tmp_path / "models"
    first = publish_model_version(
        source, registry, model_id="D330_DPCA",
        engineer_confirmation=True, applicability_scope="D330",
    )
    second = create_model_version(
        source,
        registry,
        model_id="D330_DPCA",
        parent_model_id="D330_DPCA",
        parent_version="v0001",
    )
    assert load_model_package(first["path"])[1]["model_status"] == "published"
    assert load_model_package(second["path"])[1]["model_status"] == "validated"
    if corrupt_latest:
        Path(f"{second['path']}.sha256").write_text("0" * 64, encoding="ascii")

    third = publish_model_version(
        source, registry, model_id="D330_DPCA",
        engineer_confirmation=True, applicability_scope="D330",
    )
    _, manifest = load_model_package(third["path"])
    assert third["version"] == "v0003"
    assert manifest["parent_model_id"] == "D330_DPCA"
    assert manifest["parent_version"] == ("v0001" if corrupt_latest else "v0002")


def test_cleanup_failure_reports_original_error_and_residual_path(tmp_path, monkeypatch):
    import pca_model_builder.model_registry as registry_module

    source = _candidate(tmp_path)
    monkeypatch.setattr(registry_module, "_write_external_hash", lambda path: (_ for _ in ()).throw(RuntimeError("write failed")))
    monkeypatch.setattr(registry_module.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.raises(RuntimeError, match=r"(?s)write failed.*cleanup failed.*ROLLBACK[/\\]v0001"):
        create_model_version(source, tmp_path / "models", model_id="ROLLBACK")


@pytest.mark.parametrize("failure", ["_write_external_hash", "load_model_package", "verify_model_package_integrity"])
def test_registry_write_failures_remove_reserved_version(tmp_path, monkeypatch, failure):
    import pca_model_builder.model_registry as registry_module

    source = _candidate(tmp_path)
    original = getattr(registry_module, failure)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(registry_module, failure, fail)
    with pytest.raises(RuntimeError, match="simulated failure"):
        create_model_version(source, tmp_path / "models", model_id="ROLLBACK")
    assert not (tmp_path / "models" / "ROLLBACK" / "v0001").exists()
    monkeypatch.setattr(registry_module, failure, original)


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
