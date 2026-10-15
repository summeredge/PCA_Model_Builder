import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from pca_model_builder import web


@pytest.fixture
def validation_case(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    rng = np.random.default_rng(42)
    history = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=180, freq="5min"),
        "A": rng.normal(size=180),
        "B": rng.normal(size=180),
        "LOAD": 2,
        "label": "training",
    })
    uploaded = web.save_upload("history.csv", history.to_csv(index=False).encode("utf-8-sig"))
    trained = web.train_payload({
        "file_id": uploaded["file_id"], "timestamp_column": "time", "tags": ["A", "B"],
        "normal_start": history.time.iloc[0].isoformat(),
        "normal_end": history.time.iloc[-1].isoformat(),
        "sample_interval_minutes": 5, "filter_method": "none",
        "smoothing_window_minutes": 0, "max_lag_minutes": 10, "lag_step_minutes": 5,
        "state_filters": [{"column": "LOAD", "minimum": 1}], "model_name": "candidate",
        "n_components": 2,
        "tag_configs": {"LOAD": {"role": "state_filter"}},
    })
    future = history.copy()
    future["time"] += pd.Timedelta(days=1)
    future["label"] = "validation"
    payload = {
        "run_id": trained["run_id"], "timestamp_column": "time",
        "validation_windows": [{
            "id": "normal", "type": "normal_validation", "enabled": True, "comment": "",
            "start": future.time.iloc[2].isoformat(), "end": future.time.iloc[-1].isoformat(),
        }],
    }
    return history, future, payload, uploaded


def _upload_payload(frame, payload):
    uploaded = web.save_upload("validation.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    return {**payload, "file_id": uploaded["file_id"]}


@pytest.mark.parametrize("file_type", ["csv", "xlsx", "txt"])
def test_new_file_after_full_history_training_records_actual_source(validation_case, file_type):
    _, future, payload, training_upload = validation_case
    if file_type == "csv":
        content = future.to_csv(index=False).encode("utf-8-sig")
        encoding = "utf-8-sig"
    elif file_type == "xlsx":
        buffer = io.BytesIO()
        future.to_excel(buffer, index=False)
        content = buffer.getvalue()
        encoding = ""
    else:
        exported = future.copy()
        exported["time"] = exported.time.dt.strftime("%Y/%m/%d %H:%M")
        content = exported.to_csv(index=False, sep="\t").encode("ascii")
        encoding = "ascii"
    upload = web.save_upload(f"validation.{file_type}", content)
    payload = {**payload, "file_id": upload["file_id"]}
    result = web.validate_payload(payload)
    path = web.UPLOADS_DIR / f'{payload["file_id"]}.{file_type}'
    expected = {
        "file_id": payload["file_id"], "filename": path.name,
        "timestamp_column": "time", "encoding": encoding,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert result["validation_source"] == expected
    assert expected["file_id"] != training_upload["file_id"]
    assert result["scored_rows"] == len(future) - 2
    assert result["model_status"] == "candidate"
    assert result["engineer_decision_required"] is True
    report = json.loads((web.RUNS_DIR / payload["run_id"] / "validation_report.json").read_text(encoding="utf-8"))
    assert report["validation_source"] == expected
    assert report["validation_evidence"]


@pytest.mark.parametrize("column", ["A", "LOAD", "label"])
def test_missing_validation_columns(validation_case, column):
    _, future, payload, _ = validation_case
    payload = _upload_payload(future.drop(columns=column), payload)
    if column == "label":
        payload["label_column"] = column
    with pytest.raises(ValueError, match=f"找不到.*{column}"):
        web.validate_payload(payload)
    assert not (web.RUNS_DIR / payload["run_id"] / "validation_report.json").exists()


@pytest.mark.parametrize("failure", ["column", "timestamp", "encoding"])
def test_validation_parse_errors(validation_case, failure):
    _, future, payload, _ = validation_case
    if failure == "timestamp":
        future["time"] = "invalid-time"
    payload = _upload_payload(future, payload)
    if failure == "column":
        payload["timestamp_column"] = "missing_time"
    if failure == "encoding":
        path = web.UPLOADS_DIR / f'{payload["file_id"]}.csv'
        path.write_bytes(b"time,A,B,LOAD\n\xff,1,2,2\n")
    with pytest.raises(ValueError):
        web.validate_payload(payload)
    assert not (web.RUNS_DIR / payload["run_id"] / "validation_report.json").exists()


def test_training_endpoint_overlap_is_rejected_for_separate_file(validation_case):
    history, future, payload, _ = validation_case
    future["time"] = pd.date_range(history.time.iloc[-1], periods=len(future), freq="5min")
    payload["validation_windows"][0].update(
        start=history.time.iloc[-1].isoformat(), end=future.time.iloc[-1].isoformat()
    )
    with pytest.raises(ValueError, match="training and validation windows overlap"):
        web.validate_payload(_upload_payload(future, payload))


def test_validation_windows_cannot_share_endpoint(validation_case):
    _, future, payload, _ = validation_case
    first = payload["validation_windows"][0]
    first["end"] = future.time.iloc[60].isoformat()
    payload["validation_windows"].append({
        **first, "id": "second", "start": first["end"], "end": future.time.iloc[-1].isoformat(),
    })
    with pytest.raises(ValueError, match="validation windows overlap"):
        web.validate_payload(_upload_payload(future, payload))


def test_validation_file_must_supply_its_own_context(validation_case):
    _, future, payload, _ = validation_case
    with pytest.raises(ValueError, match="context is insufficient"):
        web.validate_payload(_upload_payload(future.iloc[2:], payload))


def test_engineering_labels_come_from_validation_file(validation_case):
    _, future, payload, _ = validation_case
    payload = _upload_payload(future, {**payload, "label_column": "label"})
    result = web.validate_payload(payload)
    assert set(result["status_by_engineering_label"]) == {"validation"}
    assert sum(result["status_by_engineering_label"]["validation"].values()) == result["scored_rows"]
