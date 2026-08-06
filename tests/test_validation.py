import pandas as pd
import pytest
from types import SimpleNamespace
import inspect

from pca_model_builder.preprocessing import PreprocessingConfig, StateFilter
from pca_model_builder.validation import (
    _combined_exceedance_events,
    _contribution_stability,
    _contribution_stability_group,
    _validation_metrics,
    build_validation_matrix,
    ensure_disjoint_windows,
    normalize_validation_windows,
    record_engineer_decision,
    validate_model_windows,
)


def _pr6_evidence():
    stability_group = {
        "event_count": 0,
        "top_k": 3,
        "top1_consistency_rate": None,
        "average_top_k_jaccard_similarity": None,
        "average_contribution_cosine_similarity": None,
        "tags": [],
    }
    return {
        "validation_metrics": {
            "normal_validation": {
                "valid_window_count": 1,
                "scoring_row_count": 1,
                "t2": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "spe": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "overall": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "continuous_false_alarm_event_count_95": 0,
                "longest_continuous_false_alarm_minutes": 0,
            },
            "known_abnormal": {
                "valid_window_count": 1,
                "detected_window_count_95": 0,
                "detection_rate_95": 0.0,
                "detected_window_count_99": 0,
                "detection_rate_99": 0.0,
                "t2_detected_window_count_95": 0,
                "t2_detected_window_count_99": 0,
                "spe_detected_window_count_95": 0,
                "spe_detected_window_count_99": 0,
                "windows": [{
                    "validation_window_id": "known-001",
                    "first_detection_95": None,
                    "first_detection_delay_minutes_95": None,
                    "first_detection_99": None,
                    "first_detection_delay_minutes_99": None,
                }],
                "first_detection_delay_minutes_95_median": None,
                "first_detection_delay_minutes_95_max": None,
            },
        },
        "contribution_stability": {
            validation_type: {statistic: dict(stability_group) for statistic in ("t2", "spe")}
            for validation_type in ("normal_validation", "known_abnormal")
        },
    }


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


def test_validation_uses_training_resampling_contract_and_excludes_context_rows():
    index = pd.date_range("2026-01-01", periods=21, freq="1min")
    frame = pd.DataFrame({"A": range(21), "B": range(100, 121)}, index=index)
    config = PreprocessingConfig(
        5, 0, 0, 5, resampling_method="mean", filter_method="none"
    )

    dynamic = build_validation_matrix(
        frame, ["A", "B"], config, index[5], index[15]
    )

    assert dynamic.index.tolist() == [index[5], index[10], index[15]]
    assert dynamic.index.min() >= index[5]


def test_validation_lag_does_not_cross_state_filter_break():
    index = pd.date_range("2026-01-01", periods=10, freq="5min")
    frame = pd.DataFrame(
        {"A": range(10), "B": range(100, 110), "LOAD": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]},
        index=index,
    )
    config = PreprocessingConfig(
        5,
        0,
        5,
        5,
        filter_method="none",
        state_filters=(StateFilter("LOAD", minimum=1),),
    )

    with pytest.raises(ValueError, match="insufficient"):
        build_validation_matrix(frame, ["A", "B"], config, index[5], index[8])


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

    with pytest.raises(ValueError, match="重新执行独立验证"):
        record_engineer_decision(
            manifest,
            {"normal_validation_complete": True, "known_abnormal_complete": True},
            "passed",
            "",
        )
    malformed = _pr6_evidence()
    malformed["validation_metrics"]["normal_validation"] = {}
    with pytest.raises(ValueError, match="重新执行独立验证"):
        record_engineer_decision(
            manifest,
            {
                "normal_validation_complete": True,
                "known_abnormal_complete": True,
                **malformed,
            },
            "passed",
            "",
        )
    malformed = _pr6_evidence()
    malformed["contribution_stability"]["known_abnormal"]["spe"] = {
        "event_count": 0,
        "top_k": 3,
        "top1_consistency_rate": float("nan"),
        "average_top_k_jaccard_similarity": None,
        "average_contribution_cosine_similarity": None,
        "tags": [],
    }
    with pytest.raises(ValueError, match="重新执行独立验证"):
        record_engineer_decision(
            manifest,
            {
                "normal_validation_complete": True,
                "known_abnormal_complete": True,
                **malformed,
            },
            "passed",
            "",
        )
    malformed = _pr6_evidence()
    malformed["contribution_stability"]["normal_validation"]["t2"] = {
        "event_count": 1,
        "top_k": 1,
        "top1_consistency_rate": 1.0,
        "average_top_k_jaccard_similarity": None,
        "average_contribution_cosine_similarity": None,
        "tags": [{
            "tag": "A",
            "top1_count": 1,
            "top_k_count": 1,
            "top_k_recurrence_rate": 1.0,
            "average_contribution_pct": float("inf"),
            "median_contribution_pct": 60.0,
        }],
    }
    with pytest.raises(ValueError, match="重新执行独立验证"):
        record_engineer_decision(
            manifest,
            {
                "normal_validation_complete": True,
                "known_abnormal_complete": True,
                **malformed,
            },
            "passed",
            "",
        )

    decision = record_engineer_decision(
        manifest,
        {
            "normal_validation_complete": True,
            "known_abnormal_complete": True,
            **_pr6_evidence(),
        },
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


def test_validation_metrics_separate_statistics_and_weight_normal_windows_by_rows():
    index = pd.date_range("2026-02-01", periods=6, freq="5min")
    limits = SimpleNamespace(t2_limits={0.95: 1.0, 0.99: 3.0}, q_limits={0.95: 1.0, 0.99: 3.0})
    normal_one = pd.DataFrame({"t2": [2.0, 0.0], "spe": [0.0, 0.0]}, index=index[:2])
    normal_two = pd.DataFrame({"t2": [0.0] * 4, "spe": [4.0, 0.0, 0.0, 0.0]}, index=index[2:])
    abnormal_one = pd.DataFrame({"t2": [0.0, 2.0], "spe": [0.0, 0.0]}, index=index[:2])
    abnormal_two = pd.DataFrame({"t2": [0.0, 0.0, 0.0], "spe": [0.0, 0.0, 4.0]}, index=index[3:])
    metrics = _validation_metrics(
        [
            {"id": "n1", "type": "normal_validation", "start": index[0], "scores": normal_one, "continuous_events": [{"event_start": index[0].isoformat(), "event_end": index[0].isoformat()}]},
            {"id": "n2", "type": "normal_validation", "start": index[2], "scores": normal_two, "continuous_events": [{"event_start": index[2].isoformat(), "event_end": index[2].isoformat()}]},
            {"id": "a1", "type": "known_abnormal", "start": index[0], "scores": abnormal_one, "continuous_events": []},
            {"id": "a2", "type": "known_abnormal", "start": index[3], "scores": abnormal_two, "continuous_events": []},
        ],
        limits,
        5,
    )

    normal = metrics["normal_validation"]
    assert normal["scoring_row_count"] == 6
    assert normal["t2"]["exceedance_rate_95"] == pytest.approx(1 / 6)
    assert normal["spe"]["exceedance_rate_95"] == pytest.approx(1 / 6)
    assert normal["overall"]["exceedance_rate_95"] == pytest.approx(2 / 6)
    assert normal["t2"]["exceedance_rate_99"] == 0
    assert normal["spe"]["exceedance_rate_99"] == pytest.approx(1 / 6)
    assert normal["continuous_false_alarm_event_count_95"] == 2

    abnormal = metrics["known_abnormal"]
    assert abnormal["detection_rate_95"] == 1
    assert abnormal["detection_rate_99"] == pytest.approx(1 / 2)
    assert abnormal["t2_detected_window_count_95"] == 1
    assert abnormal["spe_detected_window_count_95"] == 1
    assert abnormal["windows"][0]["first_detection_delay_minutes_95"] == 5
    assert abnormal["windows"][1]["first_detection_delay_minutes_99"] == 10
    assert abnormal["first_detection_delay_minutes_95_median"] == 7.5
    assert abnormal["first_detection_delay_minutes_95_max"] == 10


def test_contribution_stability_is_separate_deterministic_and_handles_boundaries():
    features = ["A__lag_000min", "B__lag_000min", "C__lag_000min", "D__lag_000min"]
    first = {"validation_type": "normal_validation", "statistic": "t2", "tags": [{"tag": "C", "contribution_pct": 10.0}, {"tag": "A", "contribution_pct": 60.0}, {"tag": "B", "contribution_pct": 30.0}]}
    same = {"validation_type": "normal_validation", "statistic": "t2", "tags": [{"tag": "B", "contribution_pct": 30.0}, {"tag": "A", "contribution_pct": 60.0}, {"tag": "C", "contribution_pct": 10.0}]}
    changed = {"validation_type": "normal_validation", "statistic": "t2", "tags": [{"tag": "D", "contribution_pct": 30.0}, {"tag": "B", "contribution_pct": 60.0}, {"tag": "C", "contribution_pct": 10.0}]}

    identical = _contribution_stability([first, same], features)["normal_validation"]["t2"]
    assert identical["top1_consistency_rate"] == 1
    assert identical["average_top_k_jaccard_similarity"] == 1
    assert identical["average_contribution_cosine_similarity"] == pytest.approx(1)
    assert identical == _contribution_stability([same, first], features)["normal_validation"]["t2"]

    unstable = _contribution_stability([first, changed], features)["normal_validation"]["t2"]
    assert unstable["top1_consistency_rate"] == pytest.approx(0.5)
    assert unstable["average_top_k_jaccard_similarity"] == pytest.approx(0.5)
    assert unstable["average_contribution_cosine_similarity"] < 1
    assert _contribution_stability([first], features)["known_abnormal"]["t2"]["event_count"] == 0
    single = _contribution_stability([first], features)["normal_validation"]["t2"]
    assert single["average_top_k_jaccard_similarity"] is None
    assert single["average_contribution_cosine_similarity"] is None


def test_contribution_stability_streams_pairs_with_exact_results():
    tags = ["A", "B", "C", "D"]
    events = [
        {"tags": [{"tag": "A", "contribution_pct": 60.0}, {"tag": "B", "contribution_pct": 30.0}, {"tag": "C", "contribution_pct": 10.0}]},
        {"tags": [{"tag": "B", "contribution_pct": 60.0}, {"tag": "A", "contribution_pct": 30.0}, {"tag": "D", "contribution_pct": 10.0}]},
        {"tags": [{"tag": "C", "contribution_pct": 60.0}, {"tag": "B", "contribution_pct": 30.0}, {"tag": "D", "contribution_pct": 10.0}]},
    ]
    direct_jaccards = []
    direct_cosines = []
    for left in range(len(events)):
        left_values = {item["tag"]: item["contribution_pct"] for item in events[left]["tags"]}
        left_top = set(sorted(tags, key=lambda tag: (-left_values.get(tag, 0.0), tag))[:3])
        left_vector = [left_values.get(tag, 0.0) for tag in tags]
        for right in range(left + 1, len(events)):
            right_values = {item["tag"]: item["contribution_pct"] for item in events[right]["tags"]}
            right_top = set(sorted(tags, key=lambda tag: (-right_values.get(tag, 0.0), tag))[:3])
            right_vector = [right_values.get(tag, 0.0) for tag in tags]
            direct_jaccards.append(len(left_top & right_top) / len(left_top | right_top))
            direct_cosines.append(
                sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
                / (sum(a * a for a in left_vector) * sum(b * b for b in right_vector)) ** 0.5
            )

    result = _contribution_stability_group(events, tags)
    assert result["average_top_k_jaccard_similarity"] == pytest.approx(
        sum(direct_jaccards) / len(direct_jaccards)
    )
    assert result["average_contribution_cosine_similarity"] == pytest.approx(
        sum(direct_cosines) / len(direct_cosines)
    )
    large = [events[index % len(events)] for index in range(128)]
    assert _contribution_stability_group(large, tags)["event_count"] == len(large)
    source = inspect.getsource(_contribution_stability_group)
    assert "pairs =" not in source
    assert "jaccards =" not in source
    assert "cosines =" not in source
