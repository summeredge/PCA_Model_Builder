import hashlib
from http.server import ThreadingHTTPServer
import inspect
import json
import re
import threading
import urllib.request

import numpy as np
import pandas as pd
import pytest
import zipfile

from pca_model_builder import web, web_model_results
from pca_model_builder.model_io import load_model_package


def _history_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-01-01", periods=180, freq="5min")
    a = rng.normal(size=len(timestamps))
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "A": a,
            "B": 1.8 * a + rng.normal(scale=0.1, size=len(timestamps)),
            "C": rng.normal(scale=0.25, size=len(timestamps)),
        }
    )
    return frame


def _upload(tmp_path, monkeypatch, frame, name="history.csv"):
    monkeypatch.setattr(web, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(web, "RUNS_DIR", tmp_path / "runs")
    return web.save_upload(name, frame.to_csv(index=False).encode("utf-8-sig"))


def _base_payload(uploaded):
    return {
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "tags": ["A", "B", "C"],
        "sample_interval_minutes": 5,
        "filter_method": "none",
        "resampling_method": "none",
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
    }


def _provenance_setup(payload, frame):
    """One cluster window plus one performance refinement window with verified origins."""
    analysis = {
        "exploration_start": frame.time.iloc[0].isoformat(),
        "exploration_end": frame.time.iloc[119].isoformat(),
        "exploration_config": {"cluster_count": 2, "minimum_candidate_duration_minutes": 10},
    }
    exploration = web.state_exploration_payload({**payload, **analysis})
    run = exploration["exploration_run_id"]
    parents = list({item["cluster_id"]: item for item in exploration["cluster_candidates"]}.values())
    assert len(parents) == 2
    cluster_window = {
        "id": "training-cluster", "start": parents[0]["start"], "end": parents[0]["end"],
        "source": "cluster", "source_ref": f"state-exploration-{run}-{parents[0]['candidate_id']}",
        "enabled": True, "comment": "",
    }
    refined_parent = {
        "id": "performance-parent", "start": parents[1]["start"], "end": parents[1]["end"],
        "source": "cluster", "source_ref": f"state-exploration-{run}-{parents[1]['candidate_id']}",
        "comment": "",
    }
    performance = web.performance_screen_payload({
        **payload,
        "analysis_start": frame.time.iloc[0].isoformat(),
        "analysis_end": frame.time.iloc[119].isoformat(),
        "conditions": [{"column": "A", "minimum": -100}],
        "scope": {"type": "candidate_windows"},
        "parent_windows": [refined_parent],
    })
    refined = performance["candidate_windows"][0]
    refinement = {**refined, "id": "refined", "comment": ""}
    performance_window = {
        "id": "training-performance", "start": refined["start"], "end": refined["end"],
        "source": "performance", "source_ref": refined["source_ref"], "enabled": True, "comment": "",
    }
    return run, parents, cluster_window, refined_parent, refinement, performance_window


def _read_snapshot(run_id):
    path = web.RUNS_DIR / run_id / "modeling_snapshot.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar(window, candidate):
    return {
        "window_id": window["id"],
        **{key: window[key] for key in ("source", "source_ref", "start", "end")},
        "candidate": candidate,
    }


def test_train_writes_snapshot_with_verified_cluster_and_refinement_provenance(tmp_path, monkeypatch):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    payload = _base_payload(uploaded)
    run, parents, cluster_window, refined_parent, refinement, performance_window = _provenance_setup(payload, frame)
    request = {
        **payload,
        "training_windows": [cluster_window, performance_window],
        "training_candidates": [refinement],
        "model_name": "snapshot",
        "n_components": 2,
        "change_reason": "  排除每日清洗状态并细化高负荷工况  ",
        "source_filename": "history.csv",
    }

    trained = web.train_payload(request)
    run_dir = web.RUNS_DIR / trained["run_id"]
    snapshot = _read_snapshot(trained["run_id"])

    assert snapshot["snapshot_schema_version"] == 1
    assert snapshot["run_id"] == trained["run_id"]
    assert snapshot["model_name"] == "snapshot"
    assert snapshot["model_purpose"] == "normal_state"
    assert snapshot["model_status"] == "candidate"
    assert snapshot["candidate_model"] == {
        "filename": "model.pcamodel",
        "sha256": hashlib.sha256((run_dir / "model.pcamodel").read_bytes()).hexdigest(),
    }
    assert snapshot["change_reason"] == "排除每日清洗状态并细化高负荷工况"
    upload_path = web.UPLOADS_DIR / f"{uploaded['file_id']}.csv"
    assert snapshot["data_source"] == {
        "filename": "history.csv",
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "sha256": hashlib.sha256(upload_path.read_bytes()).hexdigest(),
        "size_bytes": upload_path.stat().st_size,
    }
    assert [item["tag"] for item in snapshot["modeling_tags"]] == ["A", "B", "C"]
    assert all(item["role"] == "continuous_input" and "description" in item for item in snapshot["modeling_tags"])
    assert snapshot["excluded_tags"] == []
    assert snapshot["modeling_eligibility"] == {"keep_conditions": [], "exclude_rule_groups": []}
    assert [{key: item[key] for key in cluster_window} for item in snapshot["training_windows"]] == [
        cluster_window, performance_window
    ]
    assert snapshot["training_window_summary"] == trained["training_window_summary"]
    assert snapshot["training_window_totals"] == trained["training_window_totals"]
    config = snapshot["model_config"]
    assert config["sample_interval_minutes"] == 5 and config["filter_method"] == "none"
    assert config["first_order_alpha"] is None and config["state_filters"] == []
    assert config["variance_threshold"] == 0.95 and config["requested_n_components"] == 2
    assert config["n_components"] == trained["n_components"]
    assert config["dynamic_features"] == trained["dynamic_features"]

    records = {item["window_id"]: item for item in snapshot["window_provenance"]}
    assert records["training-cluster"]["traceable"] is True
    assert records["training-cluster"]["origin_source"] == "cluster"
    assert records["training-cluster"]["exploration_run_id"] == run
    assert records["training-cluster"]["cluster_id"] == parents[0]["cluster_id"]
    assert records["training-cluster"]["origin_candidate_id"] == parents[0]["candidate_id"]
    refined_record = records["training-performance"]
    assert refined_record["traceable"] is True
    assert refined_record["source"] == "performance"
    assert refined_record["origin_source"] == "cluster"
    assert refined_record["exploration_run_id"] == run
    assert refined_record["cluster_id"] == parents[1]["cluster_id"]
    assert refined_record["origin_candidate_id"] == parents[1]["candidate_id"]
    assert refined_record["origin_source_ref"] == refined_parent["source_ref"]
    assert refined_record["parent_candidate_id"] == "refined"
    assert refined_record["parent_source_ref"] == refinement["source_ref"]
    assert refined_record["screening"] == {
        "scope_type": "candidate_windows",
        "conditions": [{"column": "A", "minimum": -100.0, "maximum": None}],
    }

    quality = web.quality_payload(request)
    assert trained["training_group_composition"] == quality["training_group_composition"]
    composition = trained["training_group_composition"]
    assert composition["untraceable_samples"] == 0
    assert {group["cluster_id"] for group in composition["groups"]} == {
        parents[0]["cluster_id"], parents[1]["cluster_id"]
    }
    assert sum(group["effective_samples"] for group in composition["groups"]) == trained["training_rows"]

    retained = web.train_payload({
        **request, "training_candidates": [], "training_candidate_provenance": [_sidecar(performance_window, refinement)],
    })
    retained_records = {item["window_id"]: item for item in _read_snapshot(retained["run_id"])["window_provenance"]}
    assert retained_records["training-performance"]["traceable"] is True
    assert retained_records["training-performance"]["cluster_id"] == parents[1]["cluster_id"]
    assert retained_records["training-performance"]["parent_candidate_id"] == "refined"
    assert retained["training_group_composition"] == composition


def test_snapshot_does_not_change_model_or_deployment_packages(tmp_path, monkeypatch):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    payload = _base_payload(uploaded)
    trained = web.train_payload({
        **payload, "model_name": "packages",
        "training_windows": [{
            "id": "normal", "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[79].isoformat(),
            "source": "manual", "source_ref": None, "enabled": True, "comment": "",
        }],
    })
    run_dir = web.RUNS_DIR / trained["run_id"]

    with zipfile.ZipFile(run_dir / "model.pcamodel") as package:
        assert sorted(package.namelist()) == ["arrays.npz", "manifest.json"]
        manifest = json.loads(package.read("manifest.json"))
    assert manifest["schema_version"] == 5
    assert "modeling_snapshot" not in json.dumps(manifest)
    for key in ("change_reason", "candidate_model", "window_provenance", "data_source", "modeling_tags", "modeling_eligibility"):
        assert key not in manifest and key not in manifest["config"]
    _read_snapshot(trained["run_id"])

    web.validate_payload({
        "run_id": trained["run_id"],
        "file_id": uploaded["file_id"],
        "timestamp_column": "time",
        "validation_windows": [{
            "id": "N01", "type": "normal_validation", "start": frame.time.iloc[120].isoformat(),
            "end": frame.time.iloc[139].isoformat(), "enabled": True, "comment": "",
        }, {
            "id": "A01", "type": "known_abnormal", "start": frame.time.iloc[145].isoformat(),
            "end": frame.time.iloc[160].isoformat(), "enabled": True, "comment": "",
        }],
    })
    web.validation_decision_payload({"run_id": trained["run_id"], "decision": "passed", "comment": ""})
    web.freeze_deployment_payload({
        "run_id": trained["run_id"], "model_id": "snapshot.package", "model_version": 1, "frozen_by": "engineer",
    })
    for name in ("validated_model.pcamodel", "frozen_model.pcamodel"):
        with zipfile.ZipFile(run_dir / name) as package:
            assert "modeling_snapshot" not in json.dumps(json.loads(package.read("manifest.json")))
    with zipfile.ZipFile(run_dir / "deployment_model.pcadeploy") as package:
        deployment = json.loads(package.read("deployment_manifest.json"))
        assert sorted(package.namelist()) == ["arrays.npz", "deployment_manifest.json"]
    assert deployment["deployment_schema_version"] == 2
    assert "modeling_snapshot" not in json.dumps(deployment)


def test_exploratory_training_writes_the_same_snapshot_format(tmp_path, monkeypatch):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    payload = _base_payload(uploaded)
    trained = web.train_payload({
        **payload,
        "model_purpose": "exploratory",
        "model_name": "draft",
        "training_windows": [{
            "id": "normal", "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[79].isoformat(),
            "source": "manual", "source_ref": None, "enabled": True, "comment": "",
        }],
    })

    snapshot = _read_snapshot(trained["run_id"])
    assert snapshot["snapshot_schema_version"] == 1
    assert snapshot["model_purpose"] == "exploratory" and snapshot["model_status"] == "draft"
    assert snapshot["change_reason"] == ""
    assert [item["window_id"] for item in snapshot["window_provenance"]] == ["normal"]
    assert snapshot["window_provenance"][0]["traceable"] is False
    assert trained["training_group_composition"]["untraceable_samples"] == trained["training_window_totals"]["training_rows"]


@pytest.mark.parametrize("forgery", [
    "window_id", "source", "source_ref", "start", "end", "duplicate",
    "origin_source", "exploration_run_id", "cluster_id", "origin_source_ref",
])
def test_snapshot_marks_forged_refinement_provenance_untraceable(tmp_path, monkeypatch, forgery):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    payload = _base_payload(uploaded)
    _, _, cluster_window, _, refinement, performance_window = _provenance_setup(payload, frame)
    sidecar = _sidecar(performance_window, refinement)
    if forgery in {"window_id", "source", "source_ref", "start", "end"}:
        sidecar[forgery] = "other"
        records = [sidecar]
    elif forgery == "duplicate":
        records = [sidecar, sidecar]
    else:
        provenance = {**refinement["provenance"]}
        if forgery == "origin_source":
            provenance["origin_source"] = "manual"
        elif forgery == "exploration_run_id":
            provenance["exploration_run_id"] = "b" * 32
        elif forgery == "cluster_id":
            provenance["cluster_id"] = "cluster_099"
        else:
            provenance["origin_source_ref"] = re.sub(r"candidate-\d+$", "candidate-999", provenance["origin_source_ref"])
        records = [{**sidecar, "candidate": {**refinement, "provenance": provenance}}]
    request = {
        **payload,
        "training_windows": [cluster_window, performance_window],
        "training_candidates": [],
        "training_candidate_provenance": records,
        "model_name": f"forged-{forgery}",
        "n_components": 2,
    }

    trained = web.train_payload(request)
    records_by_window = {item["window_id"]: item for item in _read_snapshot(trained["run_id"])["window_provenance"]}
    assert records_by_window["training-cluster"]["traceable"] is True
    assert records_by_window["training-performance"]["traceable"] is False
    composition = trained["training_group_composition"]
    assert composition == web.quality_payload(request)["training_group_composition"]
    assert {group["cluster_id"] for group in composition["groups"]} == {records_by_window["training-cluster"]["cluster_id"]}
    assert composition["untraceable_samples"] == trained["training_window_summary"][1]["effective_samples"]


def test_snapshot_missing_for_old_run_keeps_model_and_release_flow(tmp_path, monkeypatch):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    payload = _base_payload(uploaded)
    trained = web.train_payload({
        **payload, "model_name": "old-run",
        "training_windows": [{
            "id": "normal", "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[79].isoformat(),
            "source": "manual", "source_ref": None, "enabled": True, "comment": "",
        }],
    })
    run_id = trained["run_id"]
    run_dir = web.RUNS_DIR / run_id
    (run_dir / "modeling_snapshot.json").unlink()

    assert web.modeling_snapshot_payload(run_id) == {"run_id": run_id, "available": False}
    assert '"/api/modeling-snapshot"' in inspect.getsource(web._Handler.do_GET)
    _, manifest = load_model_package(run_dir / "model.pcamodel")
    assert manifest["model_status"] == "candidate"

    web.validate_payload({
        "run_id": run_id, "file_id": uploaded["file_id"], "timestamp_column": "time",
        "validation_windows": [{
            "id": "N01", "type": "normal_validation", "start": frame.time.iloc[120].isoformat(),
            "end": frame.time.iloc[139].isoformat(), "enabled": True, "comment": "",
        }, {
            "id": "A01", "type": "known_abnormal", "start": frame.time.iloc[145].isoformat(),
            "end": frame.time.iloc[160].isoformat(), "enabled": True, "comment": "",
        }],
    })
    web.validation_decision_payload({"run_id": run_id, "decision": "passed", "comment": ""})
    frozen = web.freeze_deployment_payload({
        "run_id": run_id, "model_id": "snapshot.old", "model_version": 1, "frozen_by": "engineer",
    })
    assert frozen["model_status"] == "frozen"
    assert (run_dir / "frozen_model.pcamodel").is_file() and (run_dir / "deployment_model.pcadeploy").is_file()
    assert not (run_dir / "modeling_snapshot.json").exists()


def test_snapshot_endpoint_serves_snapshot_and_reports_missing_file(tmp_path, monkeypatch):
    frame = _history_frame()
    uploaded = _upload(tmp_path, monkeypatch, frame)
    trained = web.train_payload({
        **_base_payload(uploaded), "model_name": "endpoint",
        "training_windows": [{
            "id": "normal", "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[79].isoformat(),
            "source": "manual", "source_ref": None, "enabled": True, "comment": "",
        }],
    })
    run_id = trained["run_id"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_model_results.ModelResultsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/modeling-snapshot?run_id={run_id}"
    try:
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["available"] is True and payload["snapshot"] == _read_snapshot(run_id)
        (web.RUNS_DIR / run_id / "modeling_snapshot.json").unlink()
        with urllib.request.urlopen(url) as response:
            assert json.loads(response.read().decode("utf-8")) == {"run_id": run_id, "available": False}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_web_ui_exposes_snapshot_entry_and_change_reason():
    html = web_model_results.INDEX_HTML

    assert 'id="changeReason"' in html and "本轮建模说明 / 修改理由" in html
    assert 'id="modelingSnapshotButton"' in html and "查看建模快照" in html
    assert 'id="modelingSnapshot"' in html
    assert 'api("/api/modeling-snapshot?run_id="+encodeURIComponent(state.runId))' in html
    assert "该模型运行没有建模快照。" in html
    assert 'state.fileName=data.filename;' in html
    assert ',change_reason:el("changeReason").value.trim(),source_filename:state.fileName||null};' in html
    training_source = html.split("function renderTraining(data)", 1)[1].split("function modelingSnapshotValue", 1)[0]
    assert 'el("modelingSnapshot").hidden=true; el("modelingSnapshot").replaceChildren();' in training_source
    assert "renderTrainingComposition(totals,data);" in training_source
