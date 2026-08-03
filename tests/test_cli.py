import json
import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.cli import main
from pca_model_builder import cli
import pca_model_builder.model_io as model_io
from pca_model_builder.model_io import load_model_package
from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.training import build_training_matrix


def _create_cli_passed_run(tmp_path, prefix="candidate"):
    rng = np.random.default_rng(101)
    time = pd.date_range("2026-01-01", periods=180, freq="5min")
    frame = pd.DataFrame(
        {
            "time": time,
            "A": rng.normal(size=len(time)),
            "B": rng.normal(size=len(time)),
            "C": rng.normal(size=len(time)),
        }
    )
    csv_path = tmp_path / f"{prefix}.csv"
    candidate = tmp_path / f"{prefix}.pcamodel"
    report = tmp_path / f"{prefix}-report.json"
    validated = tmp_path / f"{prefix}-validated.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    assert main(
        [
            "train-normal",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            time[0].isoformat(),
            "--normal-end",
            time[79].isoformat(),
            "--max-lag",
            "0",
            "--components",
            "2",
            "--model-name",
            prefix,
            "--output",
            str(candidate),
        ]
    ) == 0
    windows = tmp_path / f"{prefix}-windows.json"
    windows.write_text(
        json.dumps(
            [
                {
                    "id": "normal-001",
                    "type": "normal_validation",
                    "start": time[90].isoformat(),
                    "end": time[119].isoformat(),
                    "enabled": True,
                    "comment": "normal",
                },
                {
                    "id": "abnormal-001",
                    "type": "known_abnormal",
                    "start": time[130].isoformat(),
                    "end": time[-1].isoformat(),
                    "enabled": True,
                    "comment": "event",
                },
            ]
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "validate",
            "--model",
            str(candidate),
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--validation-windows",
            str(windows),
            "--scores-output",
            str(tmp_path / f"{prefix}-scores.csv"),
            "--report-output",
            str(report),
            "--contributions-output",
            str(tmp_path / f"{prefix}-contributions.json"),
        ]
    ) == 0
    assert main(
        [
            "review-validation",
            "--model",
            str(candidate),
            "--validation-report",
            str(report),
            "--decision",
            "passed",
            "--comment",
            "approved",
            "--output",
            str(validated),
        ]
    ) == 0
    return csv_path, candidate, report, validated


def _rewrite_as_legacy_window_package(path, schema_version):
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    window = manifest["training_windows"][0]
    manifest["schema_version"] = schema_version
    manifest["training_windows"] = [[window["start"], window["end"]]]
    if schema_version == 1:
        manifest["validation_status"] = "draft"
        manifest.pop("model_purpose")
        manifest.pop("model_status")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)


def test_cli_trains_and_replays_independent_validation_window(tmp_path):
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01", periods=160, freq="5min")
    a = rng.normal(size=len(timestamps))
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "A": a,
            "B": 1.8 * a + rng.normal(scale=0.1, size=len(timestamps)),
            "engineering_label": ["normal"] * 120 + ["known_event"] * 40,
        }
    )
    frame.loc[130:140, "B"] += 8.0
    csv_path = tmp_path / "history.csv"
    model_path = tmp_path / "unit.pcamodel"
    scores_path = tmp_path / "scores.csv"
    report_path = tmp_path / "report.json"
    contributions_path = tmp_path / "contributions.json"
    tag_config_path = tmp_path / "tags.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tag_config_path.write_text(
        json.dumps(
            {
                "A": {
                    "description": "流量",
                    "unit": "t/h",
                    "engineering_min": -100,
                    "engineering_max": 100,
                },
                "B": {
                    "description": "压力",
                    "unit": "kPa",
                    "engineering_min": -100,
                    "engineering_max": 100,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    train_result = main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "--normal-start",
            str(timestamps[0]),
            "--normal-end",
            str(timestamps[99]),
            "--sample-interval",
            "5",
            "--smoothing-window",
            "10",
            "--max-lag",
            "10",
            "--lag-step",
            "5",
            "--model-name",
            "UNIT_DPCA_V1",
            "--tag-config",
            str(tag_config_path),
            "--output",
            str(model_path),
        ]
    )
    validation_result = main(
        [
            "validate",
            "--model",
            str(model_path),
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--validation-start",
            str(timestamps[100]),
            "--validation-end",
            str(timestamps[-1]),
            "--label-column",
            "engineering_label",
            "--scores-output",
            str(scores_path),
            "--report-output",
            str(report_path),
            "--contributions-output",
            str(contributions_path),
        ]
    )

    assert train_result == 0
    assert validation_result == 0
    assert model_path.exists()
    _, manifest = load_model_package(model_path)
    assert manifest["config"]["tag_configs"]["A"]["unit"] == "t/h"
    assert manifest["model_purpose"] == "normal_state"
    assert manifest["model_status"] == "candidate"
    assert scores_path.exists()
    scores = pd.read_csv(scores_path)
    assert {"pc1", "pc2"}.issubset(scores.columns)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["engineer_decision_required"] is True
    assert report["source_candidate_package"] == {
        "identifier": model_path.name,
        "filename": model_path.name,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    assert "known_event" in report["status_by_engineering_label"]
    contributions = json.loads(contributions_path.read_text(encoding="utf-8"))
    assert {item["statistic"] for item in contributions} == {"t2", "spe"}


def test_cli_models_list_compare_verify_and_publish(tmp_path, capsys):
    _, _, _, validated = _create_cli_passed_run(tmp_path, prefix="registry-cli")
    registry = tmp_path / "models"
    assert main(
        [
            "models",
            "publish",
            "--model",
            str(validated),
            "--registry",
            str(registry),
            "--confirm",
            "--applicability-scope",
            "D330 normal",
            "--engineer-comment",
            "CLI review",
        ]
    ) == 0
    published = next(registry.glob("*/v0001/model.pcamodel"))
    assert main(
        ["models", "verify", "--model-path", str(published), "--require-external"]
    ) == 0
    assert main(["models", "list", "--registry", str(registry)]) == 0
    assert main(["models", "compare", str(published), str(published)]) == 0
    assert main(
        [
            "models",
            "publish",
            "--model",
            str(validated),
            "--registry",
            str(registry),
            "--applicability-scope",
            "D330 normal",
        ]
    ) == 2
    assert "明确的工程师确认" in capsys.readouterr().err

    assert main(
        [
            "models", "publish", "--model", str(validated), "--registry", str(registry),
            "--confirm", "--applicability-scope", "D330", "--parent-version", "v0001",
        ]
    ) == 2
    assert "--parent-version只能与--model-id同时使用" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "model_purpose", "model_status"),
    [
        ("train-exploratory", "exploratory", "draft"),
        ("train-normal", "normal_state", "candidate"),
    ],
)
def test_cli_explicit_training_commands_preserve_model_semantics(
    tmp_path, command, model_purpose, model_status
):
    rng = np.random.default_rng(88)
    time = pd.date_range("2026-01-01", periods=100, freq="5min")
    a = rng.normal(size=len(time))
    frame = pd.DataFrame(
        {
            "time": time,
            "A": a,
            "B": 1.5 * a + rng.normal(scale=0.1, size=len(time)),
            "C": rng.normal(size=len(time)),
        }
    )
    csv_path = tmp_path / f"{command}.csv"
    model_path = tmp_path / f"{command}.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    assert main(
        [
            command,
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            time[0].isoformat(),
            "--normal-end",
            time[-1].isoformat(),
            "--max-lag",
            "0",
            "--model-name",
            command,
            "--output",
            str(model_path),
        ]
    ) == 0

    _, manifest = load_model_package(model_path)
    assert manifest["model_purpose"] == model_purpose
    assert manifest["model_status"] == model_status


def test_cli_rejects_exploratory_model_validation_before_creating_outputs(
    tmp_path, capsys
):
    rng = np.random.default_rng(89)
    timestamps = pd.date_range("2026-01-01", periods=100, freq="5min")
    a = rng.normal(size=len(timestamps))
    history = pd.DataFrame(
        {
            "time": timestamps,
            "A": a,
            "B": 1.5 * a + rng.normal(scale=0.1, size=len(a)),
            "C": rng.normal(size=len(a)),
        }
    )
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "exploratory.pcamodel"
    scores_path = tmp_path / "scores.csv"
    report_path = tmp_path / "report.json"
    contributions_path = tmp_path / "contributions.json"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    assert main(
        [
            "train-exploratory",
            "--csv",
            str(history_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            timestamps[0].isoformat(),
            "--normal-end",
            timestamps[-1].isoformat(),
            "--max-lag",
            "0",
            "--model-name",
            "EXPLORATORY_DPCA",
            "--output",
            str(model_path),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "validate",
            "--model",
            str(model_path),
            "--csv",
            str(tmp_path / "must-not-be-read.csv"),
            "--timestamp",
            "time",
            "--validation-start",
            timestamps[0].isoformat(),
            "--validation-end",
            timestamps[-1].isoformat(),
            "--scores-output",
            str(scores_path),
            "--report-output",
            str(report_path),
            "--contributions-output",
            str(contributions_path),
        ]
    ) == 2

    assert "探索模型不能执行独立验证" in capsys.readouterr().err
    assert not scores_path.exists()
    assert not report_path.exists()
    assert not contributions_path.exists()


def test_cli_training_windows_file_writes_canonical_window_objects(tmp_path):
    rng = np.random.default_rng(90)
    time = pd.date_range("2026-01-01", periods=100, freq="5min")
    a = rng.normal(size=len(time))
    frame = pd.DataFrame(
        {
            "time": time,
            "A": a,
            "B": a * 1.5 + rng.normal(scale=0.1, size=len(time)),
            "C": rng.normal(size=len(time)),
        }
    )
    csv_path = tmp_path / "history.csv"
    windows_path = tmp_path / "windows.json"
    model_path = tmp_path / "model.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    windows_path.write_text(
        json.dumps(
            [{"id": "window-001", "start": time[0].isoformat(), "end": time[-1].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": "稳定"}]
        ),
        encoding="utf-8",
    )

    assert main(
        ["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--training-windows", str(windows_path), "--max-lag", "0", "--model-name", "WINDOWS_DPCA", "--output", str(model_path)]
    ) == 0

    _, manifest = load_model_package(model_path)
    assert manifest["training_windows"][0]["id"] == "window-001"
    assert manifest["training_windows"][0]["comment"] == "稳定"


def test_cli_training_uses_shared_multiwindow_builder(tmp_path, monkeypatch):
    rng = np.random.default_rng(92)
    time = pd.date_range("2026-01-01", periods=100, freq="5min")
    frame = pd.DataFrame({"time": time, "A": rng.normal(size=100), "B": rng.normal(size=100), "C": rng.normal(size=100)})
    csv_path, windows_path, model_path = tmp_path / "history.csv", tmp_path / "windows.json", tmp_path / "model.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    windows_path.write_text(json.dumps([
        {"id": "window-001", "start": time[0].isoformat(), "end": time[89].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        {"id": "window-002", "start": time[90].isoformat(), "end": time[90].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
    ]), encoding="utf-8")
    original = cli.build_training_matrix
    calls = []

    def recorded(*args, **kwargs):
        calls.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "build_training_matrix", recorded)
    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--training-windows", str(windows_path), "--max-lag", "0", "--components", "2", "--model-name", "shared", "--output", str(model_path)]) == 0
    assert [window["id"] for window in calls[0]] == ["window-001", "window-002"]
    _, manifest = load_model_package(model_path)
    assert manifest["config"]["training_summary"][1]["status"] == "dropped"
    assert manifest["config"]["training_summary"][1]["dropped_reason"] == "insufficient_after_smoothing_and_lag"


def test_cli_multistate_windows_allow_local_constants_and_use_global_statistics(tmp_path):
    rng = np.random.default_rng(93)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    frame = pd.DataFrame({"time": time, "A": [10.0] * 60 + [20.0] * 60, "B": rng.normal(size=120), "C": rng.normal(size=120)})
    csv_path, windows_path, model_path = tmp_path / "history.csv", tmp_path / "windows.json", tmp_path / "model.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    windows = [
        {"id": "window-001", "start": time[0].isoformat(), "end": time[59].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        {"id": "window-002", "start": time[60].isoformat(), "end": time[-1].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
    ]
    windows_path.write_text(json.dumps(windows), encoding="utf-8")

    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--training-windows", str(windows_path), "--smoothing-window", "5", "--max-lag", "0", "--components", "2", "--model-name", "multistate", "--output", str(model_path)]) == 0

    model, manifest = load_model_package(model_path)
    expected = build_training_matrix(
        frame, "time", ["A", "B", "C"], PreprocessingConfig(5, 5, 0, 5), windows
    )
    assert len(manifest["config"]["training_summary"]) == 2
    np.testing.assert_allclose(model.mean, expected.dynamic.mean().to_numpy())
    np.testing.assert_allclose(model.scale, expected.dynamic.std(ddof=0).to_numpy())


def test_cli_multistate_windows_reject_global_constant_feature(tmp_path, capsys):
    rng = np.random.default_rng(94)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    frame = pd.DataFrame({"time": time, "A": [10.0] * 120, "B": rng.normal(size=120), "C": rng.normal(size=120)})
    csv_path, windows_path = tmp_path / "history.csv", tmp_path / "windows.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    windows_path.write_text(json.dumps([
        {"id": "window-001", "start": time[0].isoformat(), "end": time[59].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
        {"id": "window-002", "start": time[60].isoformat(), "end": time[-1].isoformat(), "source": "manual", "source_ref": None, "enabled": True, "comment": ""},
    ]), encoding="utf-8")

    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--training-windows", str(windows_path), "--smoothing-window", "5", "--max-lag", "0", "--components", "2", "--model-name", "constant", "--output", str(tmp_path / "constant.pcamodel")]) == 2
    assert "常量动态特征" in capsys.readouterr().err


@pytest.mark.parametrize("schema_version", [1, 2])
def test_cli_validates_legacy_window_packages_without_reconversion(tmp_path, schema_version):
    rng = np.random.default_rng(91)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    a = rng.normal(size=len(time))
    frame = pd.DataFrame({"time": time, "A": a, "B": 1.5 * a + rng.normal(scale=0.1, size=len(time)), "C": rng.normal(size=len(time))})
    csv_path, model_path = tmp_path / "history.csv", tmp_path / "legacy.pcamodel"
    scores, report, contributions = tmp_path / "scores.csv", tmp_path / "report.json", tmp_path / "contributions.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--normal-start", time[0].isoformat(), "--normal-end", time[59].isoformat(), "--max-lag", "0", "--model-name", "legacy", "--output", str(model_path)]) == 0
    _rewrite_as_legacy_window_package(model_path, schema_version)
    _, manifest = load_model_package(model_path)
    assert isinstance(manifest["training_windows"][0], dict)
    assert manifest["model_purpose"] == "normal_state"
    assert manifest["model_status"] == ("draft" if schema_version == 1 else "candidate")
    assert main(["validate", "--model", str(model_path), "--csv", str(csv_path), "--timestamp", "time", "--validation-start", time[0].isoformat(), "--validation-end", time[1].isoformat()]) == 2
    assert main(["validate", "--model", str(model_path), "--csv", str(csv_path), "--timestamp", "time", "--validation-start", time[60].isoformat(), "--validation-end", time[-1].isoformat(), "--scores-output", str(scores), "--report-output", str(report), "--contributions-output", str(contributions)]) == 0
    assert scores.exists() and report.exists() and contributions.exists()


def test_cli_typed_validation_review_creates_separate_validated_copy(tmp_path, capsys):
    rng = np.random.default_rng(97)
    time = pd.date_range("2026-01-01", periods=180, freq="5min")
    a = rng.normal(size=len(time))
    frame = pd.DataFrame({"time": time, "A": a, "B": 1.5 * a + rng.normal(scale=0.1, size=len(time)), "C": rng.normal(size=len(time))})
    csv_path = tmp_path / "history.csv"
    candidate = tmp_path / "candidate.pcamodel"
    windows_path = tmp_path / "validation-windows.json"
    report = tmp_path / "report.json"
    validated = tmp_path / "validated.pcamodel"
    failed_output = tmp_path / "failed.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    windows_path.write_text(json.dumps([
        {"id": "normal-001", "type": "normal_validation", "start": time[90].isoformat(), "end": time[119].isoformat(), "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": time[130].isoformat(), "end": time[-1].isoformat(), "enabled": True, "comment": "event"},
    ]), encoding="utf-8")

    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--normal-start", time[0].isoformat(), "--normal-end", time[79].isoformat(), "--max-lag", "0", "--model-name", "candidate", "--output", str(candidate)]) == 0
    assert main(["validate", "--model", str(candidate), "--csv", str(csv_path), "--timestamp", "time", "--validation-windows", str(windows_path), "--scores-output", str(tmp_path / "scores.csv"), "--report-output", str(report), "--contributions-output", str(tmp_path / "contributions.json")]) == 0
    validation_report = json.loads(report.read_text(encoding="utf-8"))
    assert validation_report["normal_validation_complete"] is True
    assert validation_report["known_abnormal_complete"] is True
    assert {item["type"] for item in validation_report["validation_window_summaries"]} == {"normal_validation", "known_abnormal"}

    candidate_b = tmp_path / "candidate-b.pcamodel"
    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--normal-start", time[0].isoformat(), "--normal-end", time[79].isoformat(), "--max-lag", "0", "--model-name", "candidate-b", "--output", str(candidate_b)]) == 0
    cross_output = tmp_path / "cross-candidate.pcamodel"
    assert main(["review-validation", "--model", str(candidate_b), "--validation-report", str(report), "--decision", "passed", "--output", str(cross_output)]) == 2
    assert not cross_output.exists()

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "insufficient", "--output", str(failed_output)]) == 0
    assert not failed_output.exists()
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "passed", "--comment", "approved", "--output", str(validated), "--source-id", "candidate-run"]) == 0
    candidate_model, candidate_manifest = load_model_package(candidate)
    validated_model, validated_manifest = load_model_package(validated)
    assert candidate_manifest["model_status"] == "candidate"
    assert validated_manifest["model_status"] == "validated"
    np.testing.assert_allclose(candidate_model.mean, validated_model.mean)
    np.testing.assert_allclose(candidate_model.scale, validated_model.scale)
    np.testing.assert_allclose(candidate_model.components, validated_model.components)

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "failed", "--output", str(validated)]) == 0
    assert not validated.exists()
    assert json.loads(report.read_text(encoding="utf-8"))["engineer_decision"]["decision"] == "failed"

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "passed", "--output", str(validated)]) == 0
    assert validated.exists()
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "insufficient", "--output", str(validated)]) == 0
    assert not validated.exists()

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), "--decision", "failed", "--output", str(candidate)]) == 2
    assert "must differ" in capsys.readouterr().err


def test_cli_review_transaction_keeps_report_and_candidate_on_commit_failure(
    tmp_path, monkeypatch
):
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    rng = np.random.default_rng(98)
    frame = pd.DataFrame({"time": time, "A": rng.normal(size=120), "B": rng.normal(size=120), "C": rng.normal(size=120)})
    csv_path = tmp_path / "history.csv"
    candidate = tmp_path / "candidate.pcamodel"
    report_path = tmp_path / "report.json"
    output = tmp_path / "validated.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    assert main(["train-normal", "--csv", str(csv_path), "--timestamp", "time", "--tags", "A", "B", "C", "--normal-start", time[0].isoformat(), "--normal-end", time[-1].isoformat(), "--max-lag", "0", "--components", "2", "--model-name", "candidate", "--output", str(candidate)]) == 0
    report = {
        "model_purpose": "normal_state",
        "model_status": "candidate",
        "normal_validation_complete": True,
        "known_abnormal_complete": True,
        "source_candidate_package": {"identifier": candidate.name, "filename": candidate.name, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    original_report = report_path.read_bytes()
    original_candidate = candidate.read_bytes()
    original_replace = model_io.os.replace
    failed = {"value": False}

    def fail_report_replace(source, destination):
        if Path(destination) == report_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated report commit failure")
        return original_replace(source, destination)

    monkeypatch.setattr(model_io.os, "replace", fail_report_replace)
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report_path), "--decision", "passed", "--output", str(output)]) == 2
    assert report_path.read_bytes() == original_report
    assert candidate.read_bytes() == original_candidate
    assert not output.exists()


def test_cli_failed_review_rejects_and_preserves_an_unrelated_output_file(tmp_path, capsys):
    _, candidate, report, output = _create_cli_passed_run(tmp_path, "ordinary")
    output.write_bytes(b"ordinary file")
    candidate_bytes = candidate.read_bytes()
    report_bytes = report.read_bytes()
    output_bytes = output.read_bytes()

    assert main(
        [
            "review-validation",
            "--model",
            str(candidate),
            "--validation-report",
            str(report),
            "--decision",
            "failed",
            "--output",
            str(output),
        ]
    ) == 2
    error = capsys.readouterr().err
    assert "已验证模型" in error or "ZIP" in error
    assert candidate.read_bytes() == candidate_bytes
    assert report.read_bytes() == report_bytes
    assert output.read_bytes() == output_bytes


def test_cli_failed_review_rejects_validated_output_from_another_candidate(tmp_path):
    _, candidate_a, _, validated_a = _create_cli_passed_run(tmp_path, "candidate-a")
    _, candidate_b, report_b, _ = _create_cli_passed_run(tmp_path, "candidate-b")
    candidate_a_bytes = candidate_a.read_bytes()
    candidate_b_bytes = candidate_b.read_bytes()
    report_b_bytes = report_b.read_bytes()
    validated_a_bytes = validated_a.read_bytes()

    assert main(
        [
            "review-validation",
            "--model",
            str(candidate_b),
            "--validation-report",
            str(report_b),
            "--decision",
            "failed",
            "--output",
            str(validated_a),
        ]
    ) == 2
    assert candidate_a.read_bytes() == candidate_a_bytes
    assert candidate_b.read_bytes() == candidate_b_bytes
    assert report_b.read_bytes() == report_b_bytes
    assert validated_a.read_bytes() == validated_a_bytes


def test_cli_failed_review_rejects_manual_old_validated_after_report_failed(tmp_path):
    _, candidate, report, output = _create_cli_passed_run(tmp_path, "manual-old")
    old_validated = output.read_bytes()
    candidate_bytes = candidate.read_bytes()

    assert main(
        [
            "review-validation",
            "--model",
            str(candidate),
            "--validation-report",
            str(report),
            "--decision",
            "failed",
            "--output",
            str(output),
        ]
    ) == 0
    failed_report = report.read_bytes()
    output.write_bytes(old_validated)

    assert main(
        [
            "review-validation",
            "--model",
            str(candidate),
            "--validation-report",
            str(report),
            "--decision",
            "insufficient",
            "--output",
            str(output),
        ]
    ) == 2
    assert candidate.read_bytes() == candidate_bytes
    assert report.read_bytes() == failed_report
    assert output.read_bytes() == old_validated


def test_cli_training_allows_physical_time_gap(tmp_path):
    rng = np.random.default_rng(3)
    timestamps = pd.date_range("2026-01-01", periods=120, freq="5min").to_series()
    timestamps.iloc[60:] += pd.Timedelta(minutes=15)
    frame = pd.DataFrame(
        {
            "time": timestamps.to_numpy(),
            "A": rng.normal(size=120),
            "B": rng.normal(size=120),
            "C": rng.normal(size=120),
        }
    )
    csv_path = tmp_path / "gap.csv"
    model_path = tmp_path / "gap.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            str(frame.time.iloc[0]),
            "--normal-end",
            str(frame.time.iloc[-1]),
            "--sample-interval",
            "5",
            "--smoothing-window",
            "10",
            "--max-lag",
            "10",
            "--lag-step",
            "5",
            "--model-name",
            "GAP_DPCA_V1",
            "--output",
            str(model_path),
        ]
    )

    assert result == 0
    assert model_path.exists()


def test_cli_training_allows_only_ten_minute_physical_gaps(tmp_path):
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=120, freq="10min"),
            "A": rng.normal(size=120),
            "B": rng.normal(size=120),
            "C": rng.normal(size=120),
        }
    )
    csv_path = tmp_path / "all_gaps.csv"
    model_path = tmp_path / "all_gaps.pcamodel"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            str(frame.time.iloc[0]),
            "--normal-end",
            str(frame.time.iloc[-1]),
            "--sample-interval",
            "5",
            "--smoothing-window",
            "5",
            "--max-lag",
            "0",
            "--lag-step",
            "5",
            "--components",
            "2",
            "--model-name",
            "ALL_GAPS_DPCA_V1",
            "--output",
            str(model_path),
        ]
    )

    assert result == 0
    assert model_path.exists()


def test_cli_training_reports_constant_tag_name(tmp_path, capsys):
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=20, freq="5min"),
            "FIXED": np.full(20, 50.0),
            "A": np.arange(20, dtype=float),
            "B": np.arange(20, dtype=float) ** 2,
        }
    )
    csv_path = tmp_path / "constant.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "FIXED",
            "A",
            "B",
            "--normal-start",
            str(frame.time.iloc[0]),
            "--normal-end",
            str(frame.time.iloc[-1]),
            "--model-name",
            "BLOCKED_DPCA",
            "--output",
            str(tmp_path / "blocked.pcamodel"),
        ]
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "合并后的训练矩阵存在常量动态特征" in error
    assert "FIXED__lag_000min" in error


def test_cli_training_rejects_variance_threshold_of_one(tmp_path):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=100, freq="5min"),
            "A": rng.normal(size=100),
            "B": rng.normal(size=100),
            "C": rng.normal(size=100),
        }
    )
    csv_path = tmp_path / "history.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = main(
        [
            "train",
            "--csv",
            str(csv_path),
            "--timestamp",
            "time",
            "--tags",
            "A",
            "B",
            "C",
            "--normal-start",
            str(frame.time.iloc[0]),
            "--normal-end",
            str(frame.time.iloc[-1]),
            "--variance-threshold",
            "1.0",
            "--model-name",
            "INVALID_DPCA_V1",
            "--output",
            str(tmp_path / "invalid.pcamodel"),
        ]
    )

    assert result == 2
