import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.cli import main
from pca_model_builder import cli
from pca_model_builder.model_io import load_model_package, save_model_package
from pca_model_builder.dpca import fit_dpca
from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.training import build_training_matrix


def test_cli_exposes_frozen_replay_without_model_or_preprocessing_overrides():
    parser = cli.build_parser()
    replay = next(action for action in parser._actions if getattr(action, "dest", None) == "command")
    frozen = replay.choices["replay-frozen"]
    options = {option for action in frozen._actions for option in action.option_strings}

    assert {
        "--model", "--csv", "--timestamp", "--replay-start", "--replay-end",
        "--scores-output", "--summary-output", "--contributions-output",
    } <= options
    assert not {"--sample-interval", "--max-lag", "--components", "--tag-config"} & options


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


def test_cli_freeze_and_export_reject_candidate_then_export_frozen(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(701).normal(size=(100, 3)), columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"])
    candidate, frozen, deployment = tmp_path / "candidate.pcamodel", tmp_path / "frozen.pcamodel", tmp_path / "unit.pcadeploy"
    save_model_package(candidate, fit_dpca(frame, n_components=2), {"model_name":"unit","tags":["A","B","C"],"timestamp_column":"time","sample_interval_minutes":5,"smoothing_window_minutes":5,"max_lag_minutes":0,"lag_step_minutes":5,"variance_threshold":0.95}, [["2026-01-01","2026-01-02"]])
    assert main(["freeze-model", "--model", str(candidate), "--model-id", "unit", "--model-version", "1", "--frozen-by", "engineer", "--output", str(frozen)]) == 2
    assert not frozen.exists()


@pytest.mark.parametrize("manifest", [[], None, "invalid", 1])
def test_cli_freeze_rejects_nonobject_manifest_without_traceback(tmp_path, capsys, manifest):
    source, frozen = tmp_path / "invalid.pcamodel", tmp_path / "frozen.pcamodel"
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
    assert main(["freeze-model", "--model", str(source), "--model-id", "unit", "--model-version", "1", "--frozen-by", "engineer", "--output", str(frozen)]) == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not frozen.exists()


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
            "--resampling-method",
            "none",
            "--filter-method",
            "trailing_mean",
            "--gap-threshold-minutes",
            "10",
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
    assert manifest["config"]["resampling_method"] == "none"
    assert manifest["config"]["filter_method"] == "trailing_mean"
    assert manifest["config"]["gap_threshold_minutes"] == 10.0
    assert manifest["model_purpose"] == "normal_state"
    assert manifest["model_status"] == "candidate"
    assert scores_path.exists()
    scores = pd.read_csv(scores_path)
    assert {"pc1", "pc2"}.issubset(scores.columns)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["engineer_decision_required"] is True
    assert "known_event" in report["status_by_engineering_label"]
    contributions = json.loads(contributions_path.read_text(encoding="utf-8"))
    assert {item["statistic"] for item in contributions} == {"t2", "spe"}


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
    model, manifest = load_model_package(model_path)
    assert manifest["config"]["training_summary"][1]["status"] == "dropped"
    assert manifest["config"]["training_summary"][1]["dropped_reason"] == "insufficient_after_smoothing_and_lag"
    assert manifest["config"]["preprocessing_summary"] == manifest["config"][
        "training_summary"
    ]
    assert manifest["config"]["training_window_totals"] == {
        "enabled_window_count": 2,
        "used_window_count": 1,
        "dropped_window_count": 1,
        "training_rows": model.n_samples,
    }


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
    result = main(["validate", "--model", str(model_path), "--csv", str(csv_path), "--timestamp", "time", "--validation-start", time[60].isoformat(), "--validation-end", time[-1].isoformat(), "--scores-output", str(scores), "--report-output", str(report), "--contributions-output", str(contributions)])
    assert result == (2 if schema_version == 1 else 0)
    assert scores.exists() is (schema_version == 2)
    assert report.exists() is (schema_version == 2)
    assert contributions.exists() is (schema_version == 2)


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
    assert {"normal_validation", "known_abnormal"} == set(validation_report["validation_metrics"])
    assert {"normal_validation", "known_abnormal"} == set(validation_report["contribution_stability"])
    assert {item["type"] for item in validation_report["validation_window_summaries"]} == {"normal_validation", "known_abnormal"}

    validation_report["validation_metrics"] = {
        "normal_validation": {},
        "known_abnormal": {},
    }
    validation_report["contribution_stability"] = {
        validation_type: {statistic: {} for statistic in ("t2", "spe")}
        for validation_type in ("normal_validation", "known_abnormal")
    }
    report.write_text(json.dumps(validation_report), encoding="utf-8")
    review_files = ["--scores", str(tmp_path / "scores.csv"), "--contributions", str(tmp_path / "contributions.json")]
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), *review_files, "--decision", "passed", "--comment", "old report", "--output", str(validated), "--source-id", "candidate-run"]) == 2
    assert not validated.exists()
    assert "重新执行独立验证" in capsys.readouterr().err
    assert main(["validate", "--model", str(candidate), "--csv", str(csv_path), "--timestamp", "time", "--validation-windows", str(windows_path), "--scores-output", str(tmp_path / "scores.csv"), "--report-output", str(report), "--contributions-output", str(tmp_path / "contributions.json")]) == 0
    validation_report = json.loads(report.read_text(encoding="utf-8"))
    validation_report["validation_metrics"]["normal_validation"]["t2"]["exceedance_rate_95"] = "invalid"
    report.write_text(json.dumps(validation_report), encoding="utf-8")
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), *review_files, "--decision", "passed", "--comment", "invalid field", "--output", str(validated), "--source-id", "candidate-run"]) == 2
    assert not validated.exists()
    assert main(["validate", "--model", str(candidate), "--csv", str(csv_path), "--timestamp", "time", "--validation-windows", str(windows_path), "--scores-output", str(tmp_path / "scores.csv"), "--report-output", str(report), "--contributions-output", str(tmp_path / "contributions.json")]) == 0

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), *review_files, "--decision", "insufficient", "--output", str(failed_output)]) == 0
    assert not failed_output.exists()
    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), *review_files, "--decision", "passed", "--comment", "approved", "--output", str(validated), "--source-id", "candidate-run"]) == 0
    candidate_model, candidate_manifest = load_model_package(candidate)
    validated_model, validated_manifest = load_model_package(validated)
    assert candidate_manifest["model_status"] == "candidate"
    assert validated_manifest["model_status"] == "validated"
    np.testing.assert_allclose(candidate_model.mean, validated_model.mean)
    np.testing.assert_allclose(candidate_model.scale, validated_model.scale)
    np.testing.assert_allclose(candidate_model.components, validated_model.components)

    assert main(["review-validation", "--model", str(candidate), "--validation-report", str(report), *review_files, "--decision", "failed", "--output", str(candidate)]) == 2
    assert "must differ" in capsys.readouterr().err


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
