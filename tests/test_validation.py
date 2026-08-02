import pandas as pd
import pytest
from types import SimpleNamespace
import inspect

from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.validation import (
    _combined_exceedance_events,
    build_validation_matrix,
    ensure_disjoint_windows,
    normalize_validation_windows,
    record_engineer_decision,
    validate_model_windows,
)


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
    with pytest.raises(ValueError, match="正常验证和已知异常验证"):
        record_engineer_decision(
            manifest,
            {"normal_validation_complete": True, "known_abnormal_complete": False},
            "passed",
            "",
        )

    decision = record_engineer_decision(
        manifest,
        {"normal_validation_complete": True, "known_abnormal_complete": True},
        "passed",
        "approved",
    )
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
