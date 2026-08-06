from io import BytesIO
import json
import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import (
    copy_validated_model_package,
    export_deployment_package,
    freeze_validated_model_package,
    load_deployment_package,
    load_model_package,
    save_model_package,
)
from pca_model_builder.scoring_core import score_dynamic_feature_matrix


def _complete_validation_summary():
    stability = {
        validation_type: {
            statistic: {
                "event_count": 0,
                "top_k": 3,
                "top1_consistency_rate": None,
                "average_top_k_jaccard_similarity": None,
                "average_contribution_cosine_similarity": None,
                "tags": [],
            }
            for statistic in ("t2", "spe")
        }
        for validation_type in ("normal_validation", "known_abnormal")
    }
    return {
        "normal_validation_complete": True,
        "known_abnormal_complete": True,
        "validation_metrics": {
            "normal_validation": {
                "valid_window_count": 1, "scoring_row_count": 3,
                "t2": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "spe": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "overall": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "continuous_false_alarm_event_count_95": 0, "longest_continuous_false_alarm_minutes": 0,
            },
            "known_abnormal": {
                "valid_window_count": 1, "detected_window_count_95": 0, "detected_window_count_99": 0,
                "t2_detected_window_count_95": 0, "t2_detected_window_count_99": 0,
                "spe_detected_window_count_95": 0, "spe_detected_window_count_99": 0,
                "detection_rate_95": 0.0, "detection_rate_99": 0.0,
                "windows": [{"validation_window_id": "abnormal", "first_detection_95": None, "first_detection_delay_minutes_95": None, "first_detection_99": None, "first_detection_delay_minutes_99": None}],
                "first_detection_delay_minutes_95_median": None, "first_detection_delay_minutes_95_max": None,
            },
        },
        "contribution_stability": stability,
        "validation_evidence": {
            "verification_status": "verified",
            "candidate_model": {"sha256": "a" * 64},
        },
    }


def _validated_package(path, frame):
    save_model_package(
        path, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]],
        model_status="validated", validation_summary=_complete_validation_summary(),
        engineer_decision={"decision": "passed", "comment": "approved", "reviewed_at": "2026-01-03T00:00:00+00:00"},
        source_candidate_package={"identifier": "unit", "filename": "candidate.pcamodel", "sha256": "a" * 64},
    )


def test_model_package_round_trip_uses_json_and_npz(tmp_path):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        rng.normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    model = fit_dpca(frame, n_components=2)
    path = tmp_path / "unit.pcamodel"

    save_model_package(
        path,
        model,
        config=_valid_config(),
        training_windows=[["2026-01-01", "2026-01-02"]],
    )
    loaded, manifest = load_model_package(path)

    with zipfile.ZipFile(path) as package:
        assert set(package.namelist()) == {"manifest.json", "arrays.npz"}
        assert "validation_status" not in json.loads(package.read("manifest.json"))
    pd.testing.assert_frame_equal(model.score(frame), loaded.score(frame))
    assert manifest["schema_version"] == 4
    assert manifest["model_purpose"] == "normal_state"
    assert manifest["model_status"] == "candidate"
    assert "validation_status" not in manifest
    assert manifest["training_windows"] == [
        {
            "id": "legacy-window-001",
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-02T00:00:00",
            "source": "legacy",
            "source_ref": None,
            "enabled": True,
            "comment": "",
        }
    ]
    assert manifest["config"]["tags"] == ["A", "B", "C"]
    assert manifest["config"]["resampling_method"] == "none"
    assert manifest["config"]["filter_method"] == "trailing_mean"
    assert manifest["config"]["gap_threshold_minutes"] is None
    assert manifest["config"]["state_filters"] == []
    assert manifest["config"]["resampling_origin"] == "epoch"


@pytest.mark.parametrize("include_totals", [False, True])
def test_schema_v3_training_window_totals_are_optional_and_round_trip(
    tmp_path, include_totals
):
    frame = pd.DataFrame(
        np.random.default_rng(14).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    config = _valid_config()
    summaries = [{"id": "window-001", "status": "used", "effective_samples": 100}]
    config["training_summary"] = summaries
    config["preprocessing_summary"] = summaries
    if include_totals:
        config["training_window_totals"] = {
            "enabled_window_count": 1,
            "used_window_count": 1,
            "dropped_window_count": 0,
            "training_rows": len(frame),
        }
    path = tmp_path / f"totals-{include_totals}.pcamodel"

    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        config,
        [["2026-01-01", "2026-01-02"]],
    )
    model, manifest = load_model_package(path)

    assert manifest["schema_version"] == 4
    assert manifest["config"]["preprocessing_summary"] == summaries
    assert manifest["config"].get("training_window_totals") == (
        config.get("training_window_totals")
    )
    if include_totals:
        assert manifest["config"]["training_window_totals"]["training_rows"] == model.n_samples


def test_model_package_accepts_optional_source_registry_and_exclusion_metadata(
    tmp_path,
):
    frame = pd.DataFrame(
        np.random.default_rng(19).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    config = _valid_config()
    config["source_tag_configs"] = {
        "A": {"role": "continuous_input"},
        "B": {"role": "continuous_input"},
        "C": {"type": "continuous"},
        "MODE": {"role": "state_filter"},
        "FIXED": {"role": "exclude"},
    }
    config["excluded_tags"] = [
        {
            "tag": "FIXED",
            "reason": "constant_in_reference_window",
            "sample_count": 100,
            "unique_count": 1,
            "constant_value": 50.0,
        }
    ]
    config.update(
        {
            "resampling_method": "median",
            "filter_method": "trailing_median",
            "gap_threshold_minutes": 10.0,
            "state_filters": [
                {"column": "MODE", "minimum": 1.0, "maximum": None}
            ],
        }
    )
    path = tmp_path / "metadata.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        config,
        [["2026-01-01", "2026-01-02"]],
    )

    _, manifest = load_model_package(path)

    assert manifest["config"]["source_tag_configs"]["C"]["type"] == "continuous"
    assert manifest["config"]["excluded_tags"][0]["tag"] == "FIXED"
    assert manifest["config"]["resampling_method"] == "median"
    assert manifest["config"]["filter_method"] == "trailing_median"
    assert manifest["config"]["state_filters"][0]["column"] == "MODE"


def test_validated_copy_preserves_candidate_package_and_model_arrays(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(20).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    original = fit_dpca(frame, n_components=2)
    save_model_package(candidate, original, _valid_config(), [["2026-01-01", "2026-01-02"]])

    validation_summary = {"normal_validation_complete": True, "known_abnormal_complete": True, "validation_evidence": {"verification_status": "verified", "candidate_model": {"sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(), "feature_names": list(original.feature_names)}}}
    copy_validated_model_package(
        candidate,
        validated,
        validation_summary=validation_summary,
        engineer_decision={"decision": "passed", "comment": "reviewed", "reviewed_at": "2026-01-03T00:00:00+00:00"},
        source_identifier="run-001",
    )

    candidate_model, candidate_manifest = load_model_package(candidate)
    validated_model, validated_manifest = load_model_package(validated)
    assert candidate_manifest["model_status"] == "candidate"
    assert validated_manifest["model_purpose"] == "normal_state"
    assert validated_manifest["model_status"] == "validated"
    assert validated_manifest["validation_summary"]["known_abnormal_complete"] is True
    assert validated_manifest["engineer_decision"]["decision"] == "passed"
    assert validated_manifest["source_candidate_package"]["identifier"] == "run-001"
    assert validated_manifest["source_candidate_package"]["filename"] == "candidate.pcamodel"
    assert len(validated_manifest["source_candidate_package"]["sha256"]) == 64
    pd.testing.assert_frame_equal(candidate_model.score(frame), validated_model.score(frame))


def test_validated_copy_rejects_candidate_output_path(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(21).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])

    with pytest.raises(ValueError, match="must differ"):
        copy_validated_model_package(
            candidate,
            candidate,
            validation_summary={},
            engineer_decision={},
            source_identifier="run-001",
        )


def test_freeze_and_deployment_preserve_fixed_scoring_contract(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(44).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    validated, frozen, deployment = tmp_path / "validated.pcamodel", tmp_path / "frozen.pcamodel", tmp_path / "unit.pcadeploy"
    _validated_package(validated, frame)
    source_before = validated.read_bytes()
    freeze_validated_model_package(validated, frozen, model_id="unit.model-1", model_version=2, frozen_by="engineer", comment="frozen")
    frozen_model, frozen_manifest = load_model_package(frozen)
    assert validated.read_bytes() == source_before
    assert frozen_manifest["model_status"] == "frozen"
    assert frozen_manifest["source_validated_package"] == {"filename": "validated.pcamodel", "sha256": hashlib.sha256(source_before).hexdigest()}
    export_deployment_package(frozen, deployment)
    deployed, deployment_manifest = load_deployment_package(deployment)
    with zipfile.ZipFile(deployment) as package:
        assert set(package.namelist()) == {"deployment_manifest.json", "arrays.npz"}
    assert deployment_manifest["input_tags"] == frozen_manifest["config"]["tags"]
    assert deployment_manifest["dynamic_feature_names"] == frozen_manifest["feature_names"]
    expected = score_dynamic_feature_matrix(frame.to_numpy(), feature_names=frozen_model.feature_names, mean=frozen_model.mean, scale=frozen_model.scale, components=frozen_model.components, eigenvalues=frozen_model.eigenvalues, t2_limits=frozen_model.t2_limits, q_limits=frozen_model.q_limits)
    actual = deployed.score_dynamic_features(frame.to_numpy())
    np.testing.assert_allclose(actual.pc_scores, expected.pc_scores)
    np.testing.assert_allclose(actual.t2, expected.t2)
    np.testing.assert_allclose(actual.spe, expected.spe)
    assert actual.overall_status == expected.overall_status


@pytest.mark.parametrize("status", ["candidate", "draft"])
def test_freeze_rejects_nonvalidated_or_incomplete_packages(tmp_path, status):
    frame = pd.DataFrame(np.random.default_rng(45).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    source = tmp_path / f"{status}.pcamodel"
    save_model_package(source, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]], model_purpose="exploratory" if status == "draft" else "normal_state", model_status=status)
    with pytest.raises(ValueError, match="only normal_state/validated"):
        freeze_validated_model_package(source, tmp_path / "frozen.pcamodel", model_id="unit", model_version=1, frozen_by="engineer")


def test_freeze_rejects_incomplete_evidence_and_generic_save_cannot_forge_frozen(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(46).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    source = tmp_path / "validated.pcamodel"
    save_model_package(source, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]], model_status="validated", validation_summary={}, engineer_decision={"decision": "passed"})
    with pytest.raises(ValueError, match="incomplete"):
        freeze_validated_model_package(source, tmp_path / "frozen.pcamodel", model_id="unit", model_version=1, frozen_by="engineer")
    with pytest.raises(ValueError, match="purpose and status"):
        save_model_package(tmp_path / "forged.pcamodel", fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]], model_status="frozen")


@pytest.mark.parametrize("manifest", [[], None, "invalid", 4])
def test_freeze_rejects_nonobject_manifest_as_controlled_error(tmp_path, manifest):
    source = tmp_path / "invalid.pcamodel"
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", b"not-used")
    with pytest.raises(ValueError, match="cannot be read"):
        freeze_validated_model_package(source, tmp_path / "frozen.pcamodel", model_id="unit", model_version=1, frozen_by="engineer")


@pytest.mark.parametrize("schema_version", [2, 3])
def test_freeze_rejects_legacy_validated_package_before_normalization(tmp_path, schema_version):
    frame = pd.DataFrame(np.random.default_rng(51).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    source = tmp_path / "validated.pcamodel"
    _validated_package(source, frame)
    with zipfile.ZipFile(source) as package:
        manifest = json.loads(package.read("manifest.json")); arrays = package.read("arrays.npz")
    manifest["schema_version"] = schema_version
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest)); package.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="schema 4"):
        freeze_validated_model_package(source, tmp_path / "frozen.pcamodel", model_id="unit", model_version=1, frozen_by="engineer")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values.reshape(1, -1), "variance arrays have invalid shapes"),
        (lambda values: np.full_like(values, np.nan), "arrays contain non-finite"),
        (lambda values: -np.abs(values), "explained variance must not be negative"),
        (lambda values: np.full_like(values, 0.8), "explained variance exceeds one"),
    ],
)
def test_deployment_validates_actual_explained_variance_ratio(tmp_path, mutate, message):
    frame = pd.DataFrame(np.random.default_rng(52).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    validated, frozen, deployment = tmp_path / "validated.pcamodel", tmp_path / "frozen.pcamodel", tmp_path / "unit.pcadeploy"
    _validated_package(validated, frame)
    freeze_validated_model_package(validated, frozen, model_id="unit", model_version=1, frozen_by="engineer")
    export_deployment_package(frozen, deployment)
    with zipfile.ZipFile(deployment) as package:
        manifest = json.loads(package.read("deployment_manifest.json"))
        with np.load(BytesIO(package.read("arrays.npz")), allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
    arrays["explained_variance_ratio"] = mutate(arrays["explained_variance_ratio"])
    buffer = BytesIO(); np.savez_compressed(buffer, **arrays)
    manifest["arrays_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    with zipfile.ZipFile(deployment, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("deployment_manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", buffer.getvalue())
    with pytest.raises(ValueError, match=message):
        load_deployment_package(deployment)


@pytest.mark.parametrize(
    ("model_purpose", "model_status"),
    [
        ("exploratory", "candidate"),
        ("exploratory", "validated"),
        ("exploratory", "published"),
    ],
)
def test_model_package_rejects_invalid_exploratory_status(
    tmp_path, model_purpose, model_status
):
    frame = pd.DataFrame(
        np.random.default_rng(31).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )

    with pytest.raises(ValueError, match="purpose and status combination"):
        save_model_package(
            tmp_path / "invalid.pcamodel",
            fit_dpca(frame, n_components=2),
            _valid_config(),
            [["2026-01-01", "2026-01-02"]],
            model_purpose=model_purpose,
            model_status=model_status,
        )


@pytest.mark.parametrize("legacy_status", ["draft", "passed", "failed"])
def test_schema_v1_statuses_do_not_upgrade_model_semantics(tmp_path, legacy_status):
    frame = pd.DataFrame(
        np.random.default_rng(32).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    path = tmp_path / "legacy.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        _valid_config(),
        [["2026-01-01", "2026-01-02"]],
    )
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    manifest["schema_version"] = 1
    manifest["training_windows"] = [["2026-01-01", "2026-01-02"]]
    manifest["validation_status"] = legacy_status
    manifest.pop("model_purpose")
    manifest.pop("model_status")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)

    _, loaded = load_model_package(path)

    assert loaded["model_purpose"] == "normal_state"
    assert loaded["model_status"] == "draft"
    assert loaded["legacy_validation_status"] == legacy_status


def test_model_package_rejects_unexpected_files(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(1).normal(size=(100, 3)),
        columns=["A", "B", "C"],
    )
    path = tmp_path / "unexpected.pcamodel"
    save_model_package(path, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    with zipfile.ZipFile(path, "a") as package:
        package.writestr("unexpected.txt", "not allowed")

    with pytest.raises(ValueError, match="unexpected or missing files"):
        load_model_package(path)


@pytest.mark.parametrize(
    "array_name, corrupt, message",
    [
        ("scale", lambda values: np.zeros_like(values), "scale must be positive"),
        ("scale", lambda values: values.astype(str), "arrays must be numeric"),
        (
            "eigenvalues",
            lambda values: np.concatenate([values[:2], np.zeros_like(values[2:])]),
            "no effective residual space",
        ),
        (
            "components",
            lambda values: np.vstack([values[0], values[0]]),
            "not orthonormal",
        ),
        (
            "explained_variance_ratio",
            lambda values: np.array([0.8, 0.5, 0.1]),
            "explained variance exceeds one",
        ),
    ],
)
def test_model_package_rejects_invalid_numeric_arrays(
    tmp_path, array_name, corrupt, message
):
    frame = pd.DataFrame(
        np.random.default_rng(2).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    path = tmp_path / "invalid-scale.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        _valid_config(),
        [["2026-01-01", "2026-01-02"]],
    )

    with zipfile.ZipFile(path) as package:
        manifest = package.read("manifest.json")
        with np.load(BytesIO(package.read("arrays.npz"))) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
    arrays[array_name] = corrupt(arrays[array_name])
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", manifest)
        package.writestr("arrays.npz", buffer.getvalue())

    with pytest.raises(ValueError, match=message):
        load_model_package(path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda manifest: manifest.__setitem__("config", {}), "config is missing"),
        (
            lambda manifest: manifest["config"].pop("lag_step_minutes"),
            "config is missing: lag_step_minutes",
        ),
        (
            lambda manifest: manifest["config"].__setitem__(
                "tags", ["A", "A", "C"]
            ),
            "tags must be non-empty unique strings",
        ),
        (
            lambda manifest: manifest["config"].__setitem__("tag_configs", None),
            "tag_configs must be an object",
        ),
        (
            lambda manifest: manifest.__setitem__("training_windows", []),
            "training_windows必须是非空列表",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "training_windows", [{"id": "window-001"}]
            ),
            "training_windows窗口字段无效",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "training_windows", [{"id": "window-001", "start": "2026-01-02", "end": "2026-01-01", "source": "manual", "source_ref": None, "enabled": True, "comment": ""}]
            ),
            "training_windows起止时间无效",
        ),
        (
            lambda manifest: manifest["config"].__setitem__(
                "tags", ["X", "B", "C"]
            ),
            "dynamic features do not match",
        ),
        (
            lambda manifest: manifest["feature_names"].__setitem__(
                0, "A__lag_005min"
            ),
            "dynamic features do not match",
        ),
        (
            lambda manifest: manifest["config"].__setitem__(
                "excluded_tags", [{"tag": "A"}]
            ),
            "entry fields are invalid",
        ),
    ],
)
def test_model_package_rejects_invalid_metadata(tmp_path, mutate, message):
    frame = pd.DataFrame(
        np.random.default_rng(6).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    path = tmp_path / "invalid-metadata.pcamodel"
    save_model_package(
        path,
        fit_dpca(frame, n_components=2),
        _valid_config(),
        [["2026-01-01", "2026-01-02"]],
    )
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    mutate(manifest)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)

    with pytest.raises(ValueError, match=message):
        load_model_package(path)


def _valid_config() -> dict[str, object]:
    return {
        "model_name": "UNIT_DPCA_V1",
        "tags": ["A", "B", "C"],
        "timestamp_column": "time",
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 5,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "variance_threshold": 0.95,
    }
