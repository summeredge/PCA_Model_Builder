from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import (
    copy_validated_model_package,
    load_model_package,
    save_model_package,
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
    assert manifest["schema_version"] == 3
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

    assert manifest["schema_version"] == 3
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

    copy_validated_model_package(
        candidate,
        validated,
        validation_summary={"normal_validation_complete": True, "known_abnormal_complete": True},
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
    assert validated_manifest["source_candidate_package"] == {
        "identifier": "run-001",
        "filename": "candidate.pcamodel",
    }
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
