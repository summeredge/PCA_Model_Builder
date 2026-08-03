import pandas as pd
import pytest
from types import SimpleNamespace
import inspect

from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.validation import (
    classify_validation_evidence,
    normalize_and_validate_validation_evidence,
    _combined_exceedance_events,
    build_validation_matrix,
    ensure_disjoint_windows,
    normalize_validation_windows,
    record_engineer_decision,
    validate_model_windows,
)


def _evidence():
    windows = [
        {"id": "n", "type": "normal_validation", "start": "2026-01-01T00:00:00", "end": "2026-01-01T00:05:00", "enabled": True, "comment": ""},
        {"id": "a", "type": "known_abnormal", "start": "2026-01-01T01:00:00", "end": "2026-01-01T01:05:00", "enabled": True, "comment": ""},
    ]
    summaries = []
    for window in windows:
        summaries.append({**window, "status": "scored", "scored_rows": 2, "expected_rows": 2, "coverage": 1.0, "t2_exceedance_95": 0.0, "t2_exceedance_99": 0.0, "spe_exceedance_95": 0.0, "spe_exceedance_99": 0.0, "maximum_t2": 1.0, "maximum_spe": 1.0, "event_count": 0, "longest_event_minutes": 0.0})
    return {"validation_schema_version": 2, "model_purpose": "normal_state", "model_status": "candidate", "source_candidate_package": {"identifier": "run", "filename": "candidate.pcamodel", "sha256": "0" * 64}, "validation_windows": windows, "validation_window_summaries": summaries, "normal_validation_complete": True, "known_abnormal_complete": True, "scored_rows": 4, "status_counts": {"normal": 4}, "maximum_t2": 1.0, "maximum_spe": 1.0, "validation_artifacts": {"scores": {"filename": "validation_scores.csv", "sha256": "0" * 64, "bytes": 1}, "contributions": {"filename": "validation_contributions.json", "sha256": "1" * 64, "bytes": 1}}}


def test_validation_evidence_recomputes_completion_and_coverage():
    report = _evidence()
    normalize_and_validate_validation_evidence(report)
    report["validation_window_summaries"][0]["coverage"] = 0.5
    with pytest.raises(ValueError, match="coverage"):
        normalize_and_validate_validation_evidence(report)


@pytest.mark.parametrize("mutation", ["windows", "status", "rows", "type"])
def test_validation_evidence_rejects_forged_complete_flags(mutation):
    report = _evidence()
    if mutation == "windows":
        report["validation_windows"] = []
    elif mutation == "status":
        report["validation_window_summaries"][0]["status"] = "complete"
    elif mutation == "rows":
        report["validation_window_summaries"][0]["scored_rows"] = 0
    else:
        report["validation_windows"][1]["enabled"] = False
        report["validation_window_summaries"][1]["enabled"] = False
    with pytest.raises(ValueError):
        normalize_and_validate_validation_evidence(report)


def test_validation_evidence_rejects_legacy_schema():
    report = _evidence()
    report.pop("validation_schema_version")
    with pytest.raises(ValueError, match="重新执行完整验证"):
        normalize_and_validate_validation_evidence(report)


def test_validation_evidence_classification_and_top_level_consistency():
    assert classify_validation_evidence(_evidence()) == "current"
    assert classify_validation_evidence({}) == "legacy"
    assert classify_validation_evidence({"validation_schema_version": 1}) == "legacy"
    assert classify_validation_evidence({"validation_schema_version": 3}) == "invalid"
    report = _evidence()
    report["status_counts"] = {"normal": 3}
    with pytest.raises(ValueError, match="status_counts"):
        normalize_and_validate_validation_evidence(report)
    for field in ("maximum_t2", "maximum_spe"):
        report = _evidence()
        report[field] = 2.0
        with pytest.raises(ValueError, match=field):
            normalize_and_validate_validation_evidence(report)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "enabled"])
def test_validation_evidence_rejects_inconsistent_summaries(mutation):
    report = _evidence()
    if mutation == "duplicate":
        report["validation_window_summaries"].append(dict(report["validation_window_summaries"][0]))
    elif mutation == "unknown":
        report["validation_window_summaries"][0]["id"] = "unknown"
    else:
        report["validation_window_summaries"][0]["enabled"] = False
    with pytest.raises(ValueError):
        normalize_and_validate_validation_evidence(report)


def test_validation_windows_must_not_overlap_training_windows():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-20"))]

    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint_windows(training, validation)


def test_separate_validation_window_is_allowed():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"))]

    ensure_disjoint_windows(training, validation)


def test_validation_uses_pre_window_context_and_scores_from_requested_start():
    index = pd.date_range("2026-01-01", periods=30, freq="5min")
    frame = pd.DataFrame({"A": range(30), "B": range(100, 130)}, index=index)
    config = PreprocessingConfig(5, 10, 10, 5)

    dynamic = build_validation_matrix(frame, ["A", "B"], config, index[10], index[20])

    assert dynamic.index[0] == index[10]
    assert dynamic.index[-1] == index[20]
    assert len(dynamic) == 11


def test_validation_rejects_missing_pre_window_context():
    index = pd.date_range("2026-01-01", periods=10, freq="5min")
    frame = pd.DataFrame({"A": range(10), "B": range(10)}, index=index)
    config = PreprocessingConfig(5, 10, 10, 5)

    with pytest.raises(ValueError, match="insufficient"):
        build_validation_matrix(frame, ["A", "B"], config, index[0], index[-1])


def test_typed_validation_windows_reject_overlap_and_preserve_types():
    windows = [
        {"id": "normal-001", "type": "normal_validation", "start": "2026-02-01T00:00:00", "end": "2026-02-01T00:10:00", "enabled": True, "comment": "normal"},
        {"id": "abnormal-001", "type": "known_abnormal", "start": "2026-02-01T00:15:00", "end": "2026-02-01T00:25:00", "enabled": True, "comment": "event"},
    ]

    assert [window["type"] for window in normalize_validation_windows(windows)] == [
        "normal_validation",
        "known_abnormal",
    ]
    windows[1]["start"] = "2026-02-01T00:10:00"
    with pytest.raises(ValueError, match="overlap"):
        normalize_validation_windows(windows)


def test_engineer_pass_requires_both_validation_types_and_keeps_candidate_semantics():
    manifest = {"model_purpose": "normal_state", "model_status": "candidate"}
    with pytest.raises(ValueError, match="重新执行完整验证"):
        record_engineer_decision(
            manifest,
            {"normal_validation_complete": True, "known_abnormal_complete": False},
            "passed",
            "",
        )

    decision = record_engineer_decision(manifest, _evidence(), "passed", "approved")
    assert decision["decision"] == "passed"
    assert decision["comment"] == "approved"


def test_continuous_event_combines_t2_and_spe_exceedances_on_physical_time_axis():
    index = pd.date_range("2026-02-01", periods=5, freq="5min")
    scores = pd.DataFrame(
        {"t2": [0.0, 2.0, 0.0, 0.0, 0.0], "spe": [0.0, 0.0, 2.0, 0.0, 2.0]},
        index=index,
    )

    events = _combined_exceedance_events(
        scores,
        SimpleNamespace(t2_limits={0.95: 1.0}, q_limits={0.95: 1.0}),
        5,
    )

    assert [(event["event_start"], event["event_end"], event["point_count"]) for event in events] == [
        (index[1].isoformat(), index[2].isoformat(), 2),
        (index[4].isoformat(), index[4].isoformat(), 1),
    ]


def test_validation_service_does_not_fit_or_change_model_parameters():
    assert "fit_dpca(" not in inspect.getsource(validate_model_windows)
