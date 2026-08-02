import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.cli import main
from pca_model_builder.model_io import load_model_package


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
    assert "constant_tag(20) [FIXED]" in error
    assert "固定值50" in error


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
