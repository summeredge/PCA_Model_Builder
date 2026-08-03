from io import BytesIO
import json
import zipfile
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pca_model_builder.model_io as model_io
from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import (
    commit_validation_artifacts,
    commit_validation_run_artifacts,
    copy_validated_model_package,
    load_model_package,
    save_model_package,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("candidate", "report"),
        ("candidate", "scores"),
        ("candidate", "contributions"),
        ("candidate", "validated"),
        ("report", "scores"),
        ("report", "contributions"),
        ("report", "validated"),
        ("scores", "contributions"),
        ("scores", "validated"),
        ("contributions", "validated"),
    ],
)
def test_validation_run_rejects_all_artifact_path_collisions_before_writing(
    tmp_path, left, right
):
    paths = {name: tmp_path / name for name in ("candidate", "report", "scores", "contributions", "validated")}
    for path in paths.values():
        path.write_bytes(path.name.encode())
    original = {name: path.read_bytes() for name, path in paths.items()}
    paths[right] = paths[left]

    with pytest.raises(ValueError, match="路径必须互不相同"):
        commit_validation_run_artifacts(
            paths["candidate"], paths["report"], paths["scores"],
            paths["contributions"], paths["validated"], {}, pd.DataFrame(), [], "time"
        )

    assert all(path.read_bytes() == original[name] for name, path in paths.items() if name != right)
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def _copy_validated_model_package(source, destination, validation_summary, engineer_decision, source_identifier, **kwargs):
    if validation_summary.get("validation_schema_version") != 2:
        windows = [
            {"id": "normal", "type": "normal_validation", "start": "2026-01-03T00:00:00", "end": "2026-01-03T01:00:00", "enabled": True, "comment": ""},
            {"id": "abnormal", "type": "known_abnormal", "start": "2026-01-03T02:00:00", "end": "2026-01-03T03:00:00", "enabled": True, "comment": ""},
        ]
        summaries = [{**w, "status": "scored", "scored_rows": 1, "expected_rows": 1, "coverage": 1.0, "t2_exceedance_95": 0.0, "t2_exceedance_99": 0.0, "spe_exceedance_95": 0.0, "spe_exceedance_99": 0.0, "maximum_t2": 1.0, "maximum_spe": 1.0, "event_count": 0, "longest_event_minutes": 0} for w in windows]
        validation_summary = {"validation_schema_version": 2, "validation_windows": windows, "validation_window_summaries": summaries, "scored_rows": 2, "status_counts": {"normal": 2}, "maximum_t2": 1.0, "maximum_spe": 1.0, **validation_summary}
    scores = Path(source).parent / "validation_scores.csv"
    contributions = Path(source).parent / "validation_contributions.json"
    scores.write_text("time,status\n0,normal\n", encoding="utf-8")
    contributions.write_text("[]", encoding="utf-8")
    validation_summary["validation_artifacts"] = {"scores": model_io.validation_artifact_metadata(scores), "contributions": model_io.validation_artifact_metadata(contributions)}
    return copy_validated_model_package(source, destination, validation_summary, engineer_decision, source_identifier, scores_path=scores, contributions_path=contributions, **kwargs)


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


def test_legacy_validated_package_loads_read_only_and_cannot_be_copied(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(119).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    current = tmp_path / "current.pcamodel"
    legacy = tmp_path / "legacy.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    _copy_validated_model_package(candidate, current, _bound_report(candidate, "run-legacy"), {"decision": "passed", "comment": "ok", "reviewed_at": "2026-01-03T00:00:00+00:00"}, "run-legacy")
    with zipfile.ZipFile(current) as package:
        manifest, arrays = json.loads(package.read("manifest.json")), package.read("arrays.npz")
    manifest["validation_summary"].pop("validation_schema_version")
    manifest["validation_summary"].pop("validation_artifacts")
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)

    _, loaded = load_model_package(legacy)

    assert loaded["model_status"] == "validated"
    assert loaded["validation_evidence_status"] == "legacy_read_only"
    with pytest.raises(ValueError, match="重新执行完整验证"):
        copy_validated_model_package(candidate, tmp_path / "new.pcamodel", loaded["validation_summary"], loaded["engineer_decision"], "run-legacy")


def test_validated_copy_preserves_candidate_package_and_model_arrays(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(20).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    original = fit_dpca(frame, n_components=2)
    save_model_package(candidate, original, _valid_config(), [["2026-01-01", "2026-01-02"]])
    candidate_bytes = candidate.read_bytes()
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report = {
        "model_purpose": "normal_state",
        "model_status": "candidate",
        "normal_validation_complete": True,
        "known_abnormal_complete": True,
        "source_candidate_package": {
            "identifier": "run-001",
            "filename": candidate.name,
            "sha256": candidate_sha256,
        },
    }

    _copy_validated_model_package(
        candidate,
        validated,
        validation_summary=report,
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
        "sha256": candidate_sha256,
    }
    assert validated_manifest["validation_summary"]["source_candidate_package"]["sha256"] == candidate_sha256
    assert candidate.read_bytes() == candidate_bytes
    pd.testing.assert_frame_equal(candidate_model.score(frame), validated_model.score(frame))


def test_validated_copy_rejects_candidate_output_path(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(21).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])

    with pytest.raises(ValueError, match="must differ"):
        _copy_validated_model_package(
            candidate,
            candidate,
            validation_summary={},
            engineer_decision={},
            source_identifier="run-001",
        )


def test_public_model_writer_cannot_create_validated_package(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(22).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    with pytest.raises(ValueError, match="purpose and status combination"):
        save_model_package(
            tmp_path / "direct-validated.pcamodel",
            fit_dpca(frame, n_components=2),
            _valid_config(),
            [["2026-01-01", "2026-01-02"]],
            model_purpose="normal_state",
            model_status="validated",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("source_candidate_package"),
        lambda report: report.__setitem__("normal_validation_complete", "true"),
        lambda report: report["source_candidate_package"].__setitem__("sha256", "0" * 64),
        lambda report: report.__setitem__("model_status", "draft"),
    ],
)
def test_validated_copy_requires_complete_bound_evidence(tmp_path, mutate):
    frame = pd.DataFrame(
        np.random.default_rng(23).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    report = _bound_report(candidate, "run-001")
    mutate(report)

    with pytest.raises(ValueError):
        _copy_validated_model_package(
            candidate,
            validated,
            report,
            {"decision": "passed", "comment": "ok", "reviewed_at": "2026-01-03T00:00:00+00:00"},
            "run-001",
        )
    assert not validated.exists()


@pytest.mark.parametrize("decision", ["failed", "insufficient", None])
def test_validated_copy_rejects_nonpassed_decisions(tmp_path, decision):
    frame = pd.DataFrame(
        np.random.default_rng(24).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    report = _bound_report(candidate, "run-001")

    with pytest.raises(ValueError):
        _copy_validated_model_package(
            candidate,
            validated,
            report,
            {"decision": decision, "comment": "not approved", "reviewed_at": "2026-01-03T00:00:00+00:00"},
            "run-001",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.pop("validation_summary"),
        lambda manifest: manifest.pop("engineer_decision"),
        lambda manifest: manifest.pop("source_candidate_package"),
        lambda manifest: manifest["engineer_decision"].__setitem__("decision", "failed"),
        lambda manifest: manifest["validation_summary"].__setitem__("known_abnormal_complete", False),
        lambda manifest: manifest["source_candidate_package"].__setitem__("sha256", "BAD"),
        lambda manifest: manifest["source_candidate_package"].__setitem__("sha256", "0" * 64),
        lambda manifest: manifest["engineer_decision"].__setitem__("reviewed_at", "2026-01-03T00:00:00"),
        lambda manifest: manifest["source_candidate_package"].__setitem__("identifier", ""),
    ],
)
def test_tampered_validated_manifest_is_rejected_on_load(tmp_path, mutate):
    frame = pd.DataFrame(
        np.random.default_rng(25).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    report = _bound_report(candidate, "run-001")
    _copy_validated_model_package(
        candidate,
        validated,
        report,
        {"decision": "passed", "comment": "ok", "reviewed_at": "2026-01-03T00:00:00+00:00"},
        "run-001",
    )
    with zipfile.ZipFile(validated) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    mutate(manifest)
    with zipfile.ZipFile(validated, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)

    with pytest.raises(ValueError):
        load_model_package(validated)


def test_validation_artifact_commit_rolls_back_when_report_replace_fails(tmp_path, monkeypatch):
    import pca_model_builder.model_io as model_io

    frame = pd.DataFrame(
        np.random.default_rng(26).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    validated = tmp_path / "validated.pcamodel"
    report_path = tmp_path / "validation_report.json"
    save_model_package(candidate, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    report = _bound_report(candidate, "run-001")
    report["engineer_decision_required"] = True
    decision = {"decision": "passed", "comment": "ok", "reviewed_at": "2026-01-03T00:00:00+00:00"}
    original_report = {"old": True}
    report_path.write_text(json.dumps(original_report), encoding="utf-8")
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report_replace(source, destination):
        if Path(destination) == report_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated report commit failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report_replace)
    with pytest.raises(OSError, match="simulated"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            report,
            decision,
            "run-001",
        )

    assert json.loads(report_path.read_text(encoding="utf-8")) == original_report
    assert not validated.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def _review_transaction_fixture(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(27).normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    candidate = tmp_path / "candidate.pcamodel"
    report_path = tmp_path / "validation_report.json"
    validated = tmp_path / "validated.pcamodel"
    save_model_package(
        candidate,
        fit_dpca(frame, n_components=2),
        _valid_config(),
        [["2026-01-01", "2026-01-02"]],
    )
    report = _bound_report(candidate, "run-001")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    decision = {
        "decision": "passed",
        "comment": "approved",
        "reviewed_at": "2026-01-03T00:00:00+00:00",
    }
    return candidate, report_path, validated, report, decision


def _assert_transaction_clean(tmp_path):
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_review_transaction_rolls_back_when_validated_copy_fails(tmp_path, monkeypatch):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    original_report = report_path.read_bytes()
    original_candidate = candidate.read_bytes()

    def fail_copy(*args, **kwargs):
        raise OSError("simulated validated write failure")

    monkeypatch.setattr(model_io, "copy_validated_model_package", fail_copy)
    with pytest.raises(OSError, match="simulated validated write failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            {**report, "engineer_decision": decision},
            decision,
            "run-001",
            previous_report=report,
        )
    assert report_path.read_bytes() == original_report
    assert candidate.read_bytes() == original_candidate
    assert not validated.exists()
    _assert_transaction_clean(tmp_path)


def test_review_transaction_rolls_back_when_temporary_report_write_fails(
    tmp_path, monkeypatch
):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    original_report = report_path.read_bytes()
    original_candidate = candidate.read_bytes()
    original_write_text = Path.write_text

    def fail_temporary_report(self, data, *args, **kwargs):
        if self.name.startswith(".validation_report.json."):
            raise OSError("simulated temporary report write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_temporary_report)
    with pytest.raises(OSError, match="simulated temporary report write failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            {**report, "engineer_decision": decision},
            decision,
            "run-001",
            previous_report=report,
        )
    assert report_path.read_bytes() == original_report
    assert candidate.read_bytes() == original_candidate
    assert not validated.exists()
    _assert_transaction_clean(tmp_path)


@pytest.mark.parametrize("failure_target", ["validated", "report"])
def test_review_transaction_rolls_back_install_failure(tmp_path, monkeypatch, failure_target):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    original_report = report_path.read_bytes()
    original_candidate = candidate.read_bytes()
    original_replace = model_io.os.replace
    failed = {"value": False}
    target = validated if failure_target == "validated" else report_path

    def fail_install(source, destination):
        if Path(destination) == target and not failed["value"]:
            failed["value"] = True
            raise OSError(f"simulated {failure_target} install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_install)
    with pytest.raises(OSError, match=f"simulated {failure_target} install failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            {**report, "engineer_decision": decision},
            decision,
            "run-001",
            previous_report=report,
        )
    assert report_path.read_bytes() == original_report
    assert candidate.read_bytes() == original_candidate
    assert not validated.exists()
    _assert_transaction_clean(tmp_path)


def test_review_transaction_with_old_validated_restores_both_on_report_failure(
    tmp_path, monkeypatch
):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    committed_report = {**report, "engineer_decision": decision}
    commit_validation_artifacts(
        candidate,
        validated,
        report_path,
        committed_report,
        decision,
        "run-001",
        previous_report=report,
    )
    old_report = report_path.read_bytes()
    old_validated = validated.read_bytes()
    original_candidate = candidate.read_bytes()
    new_report = {**report, "status_counts": {"normal": 2, "changed": 0}, "engineer_decision": decision}
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report(source, destination):
        if Path(destination) == report_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated report install failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report)
    with pytest.raises(OSError, match="simulated report install failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            new_report,
            decision,
            "run-001",
            previous_report=committed_report,
        )
    assert report_path.read_bytes() == old_report
    assert validated.read_bytes() == old_validated
    assert candidate.read_bytes() == original_candidate
    _assert_transaction_clean(tmp_path)


@pytest.mark.parametrize("backup_target", ["validated", "report"])
def test_review_transaction_restores_when_existing_artifact_backup_fails(
    tmp_path, monkeypatch, backup_target
):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    committed_report = {**report, "engineer_decision": decision}
    commit_validation_artifacts(
        candidate,
        validated,
        report_path,
        committed_report,
        decision,
        "run-001",
        previous_report=report,
    )
    old_report = report_path.read_bytes()
    old_validated = validated.read_bytes()
    original_replace = model_io.os.replace
    failed = {"value": False}
    prefix = ".validated.pcamodel." if backup_target == "validated" else ".validation_report.json."

    def fail_backup(source, destination):
        if (
            Path(destination).name.startswith(prefix)
            and Path(destination).name.endswith(".bak")
            and not failed["value"]
        ):
            failed["value"] = True
            raise OSError(f"simulated {backup_target} backup failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_backup)
    new_report = {**report, "engineer_decision": decision, "status_counts": {"normal": 2, "changed": 0}}
    with pytest.raises(OSError, match=f"simulated {backup_target} backup failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            new_report,
            decision,
            "run-001",
            previous_report=committed_report,
        )
    assert report_path.read_bytes() == old_report
    assert validated.read_bytes() == old_validated
    _assert_transaction_clean(tmp_path)


def test_nonpassed_review_restores_old_validated_when_report_replace_fails(
    tmp_path, monkeypatch
):
    candidate, report_path, validated, report, decision = _review_transaction_fixture(tmp_path)
    committed_report = {**report, "engineer_decision": decision}
    commit_validation_artifacts(
        candidate,
        validated,
        report_path,
        committed_report,
        decision,
        "run-001",
        previous_report=report,
    )
    old_report = report_path.read_bytes()
    old_validated = validated.read_bytes()
    failed_report = {
        **report,
        "engineer_decision": {
            "decision": "failed",
            "comment": "rejected",
            "reviewed_at": "2026-01-04T00:00:00+00:00",
        },
    }
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report(source, destination):
        if Path(destination) == report_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated nonpassed report failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report)
    with pytest.raises(OSError, match="simulated nonpassed report failure"):
        commit_validation_artifacts(
            candidate,
            validated,
            report_path,
            failed_report,
            failed_report["engineer_decision"],
            "run-001",
            previous_report=committed_report,
        )
    assert report_path.read_bytes() == old_report
    assert validated.read_bytes() == old_validated
    _assert_transaction_clean(tmp_path)


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


@pytest.mark.parametrize(
    ("parent_model_id", "parent_version"),
    [
        ("D330_DPCA", None),
        (None, "v0001"),
        ("D330_DPCA", "legacy"),
        ("D330_DPCA", "v1"),
        ("D330_DPCA", "v00001"),
        ("../D330", "v0001"),
    ],
)
def test_schema_v4_rejects_invalid_parent_reference(tmp_path, parent_model_id, parent_version):
    from pca_model_builder.model_registry import create_model_version

    frame = pd.DataFrame(np.random.default_rng(9).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    source = tmp_path / "source.pcamodel"
    save_model_package(source, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    package = Path(create_model_version(source, tmp_path / "models")["path"])
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = archive.read("arrays.npz")
    manifest["parent_model_id"] = parent_model_id
    manifest["parent_version"] = parent_version
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="parent"):
        load_model_package(package)


def test_schema_v4_rejects_self_referencing_parent(tmp_path):
    from pca_model_builder.model_registry import create_model_version

    frame = pd.DataFrame(np.random.default_rng(9).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    source = tmp_path / "source.pcamodel"
    save_model_package(source, fit_dpca(frame, n_components=2), _valid_config(), [["2026-01-01", "2026-01-02"]])
    package = Path(create_model_version(source, tmp_path / "models", model_id="D330_DPCA")["path"])
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = archive.read("arrays.npz")
    manifest["parent_model_id"] = manifest["model_id"]
    manifest["parent_version"] = manifest["version"]
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", arrays)
    with pytest.raises(ValueError, match="refer to itself"):
        load_model_package(package)


def _bound_report(path, identifier):
    windows = [
        {"id": "normal", "type": "normal_validation", "start": "2026-01-03T00:00:00", "end": "2026-01-03T01:00:00", "enabled": True, "comment": ""},
        {"id": "abnormal", "type": "known_abnormal", "start": "2026-01-03T02:00:00", "end": "2026-01-03T03:00:00", "enabled": True, "comment": ""},
    ]
    summaries = [{**w, "status": "scored", "scored_rows": 1, "expected_rows": 1, "coverage": 1.0, "t2_exceedance_95": 0.0, "t2_exceedance_99": 0.0, "spe_exceedance_95": 0.0, "spe_exceedance_99": 0.0, "maximum_t2": 1.0, "maximum_spe": 1.0, "event_count": 0, "longest_event_minutes": 0} for w in windows]
    scores = Path(path).parent / "validation_scores.csv"
    contributions = Path(path).parent / "validation_contributions.json"
    scores.write_text("time,status\n0,normal\n", encoding="utf-8")
    contributions.write_text("[]", encoding="utf-8")
    return {
        "validation_schema_version": 2, "model_purpose": "normal_state", "model_status": "candidate",
        "normal_validation_complete": True, "known_abnormal_complete": True,
        "validation_windows": windows, "validation_window_summaries": summaries, "scored_rows": 2,
        "status_counts": {"normal": 2}, "maximum_t2": 1.0, "maximum_spe": 1.0,
        "validation_artifacts": {"scores": model_io.validation_artifact_metadata(scores), "contributions": model_io.validation_artifact_metadata(contributions)},
        "source_candidate_package": {"identifier": identifier, "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
