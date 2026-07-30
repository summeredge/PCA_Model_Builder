import json

import numpy as np
import pandas as pd

from pca_model_builder.cli import main


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
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

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
    assert scores_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["engineer_decision_required"] is True
    assert "known_event" in report["status_by_engineering_label"]
    contributions = json.loads(contributions_path.read_text(encoding="utf-8"))
    assert {item["statistic"] for item in contributions} == {"t2", "spe"}

