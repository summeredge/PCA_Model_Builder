import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import freeze_validated_model_package, save_model_package
from pca_model_builder import replay as replay_module
from pca_model_builder.replay import replay_frozen_model


def _validation_summary(candidate_path, feature_names):
    stability = {
        kind: {
            statistic: {
                "event_count": 0,
                "top_k": 3,
                "top1_consistency_rate": None,
                "average_top_k_jaccard_similarity": None,
                "average_contribution_cosine_similarity": None,
                "tags": [],
            }
            for statistic in ("t2", "spe")
        }
        for kind in ("normal_validation", "known_abnormal")
    }
    summary = {
        "normal_validation_complete": True,
        "known_abnormal_complete": True,
        "validation_metrics": {
            "normal_validation": {
                "valid_window_count": 1,
                "scoring_row_count": 3,
                "t2": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "spe": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "overall": {"exceedance_rate_95": 0.0, "exceedance_rate_99": 0.0},
                "continuous_false_alarm_event_count_95": 0,
                "longest_continuous_false_alarm_minutes": 0,
            },
            "known_abnormal": {
                "valid_window_count": 1,
                "detected_window_count_95": 0,
                "detected_window_count_99": 0,
                "t2_detected_window_count_95": 0,
                "t2_detected_window_count_99": 0,
                "spe_detected_window_count_95": 0,
                "spe_detected_window_count_99": 0,
                "detection_rate_95": 0.0,
                "detection_rate_99": 0.0,
                "windows": [{"validation_window_id": "abnormal", "first_detection_95": None, "first_detection_delay_minutes_95": None, "first_detection_99": None, "first_detection_delay_minutes_99": None}],
                "first_detection_delay_minutes_95_median": None,
                "first_detection_delay_minutes_95_max": None,
            },
        },
        "contribution_stability": stability,
    }
    evidence = {
        "verification_status": "verified", "verifier": "validation_artifact_verifier_v1",
        "candidate_model": {"filename": candidate_path.name, "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(), "feature_names": list(feature_names)},
        "scores": {"filename": "scores.csv", "sha256": "a" * 64, "bytes": 0, "row_count": 0, "timestamp_column": "time"},
        "contributions": {"filename": "contributions.json", "sha256": "b" * 64, "bytes": 0, "record_count": 0},
    }
    evidence["verification_digest"] = hashlib.sha256(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    summary["validation_evidence"] = evidence
    return summary


def _frozen_model(tmp_path, history, *, state_filters=(), filter_method="trailing_mean"):
    config = {
        "model_name": "unit",
        "tags": ["A", "B", "C"],
        "timestamp_column": "time",
        "sample_interval_minutes": 5,
        "smoothing_window_minutes": 10,
        "max_lag_minutes": 0,
        "lag_step_minutes": 5,
        "filter_method": filter_method,
        "state_filters": list(state_filters),
        "variance_threshold": 0.95,
        "tag_configs": {"A": {"description": "a", "unit": "x"}},
    }
    training = history.loc[:, ["A", "B", "C"]].iloc[:80].copy()
    training.columns = [f"{tag}__lag_000min" for tag in training.columns]
    model = fit_dpca(training, n_components=2)
    candidate = tmp_path / "candidate.pcamodel"
    save_model_package(candidate, model, config, [["2026-01-01", "2026-01-02"]])
    validated = tmp_path / "validated.pcamodel"
    frozen = tmp_path / "frozen.pcamodel"
    save_model_package(
        validated,
        model,
        config,
        [["2026-01-01", "2026-01-02"]],
        model_status="validated",
        validation_summary=_validation_summary(candidate, model.feature_names),
        engineer_decision={"decision": "passed", "comment": "ok", "reviewed_at": "2026-01-03T00:00:00+00:00"},
        source_candidate_package={"identifier": "unit", "filename": candidate.name, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()},
    )
    freeze_validated_model_package(validated, frozen, model_id="unit", model_version=1, frozen_by="engineer")
    with zipfile.ZipFile(frozen) as package:
        manifest = json.loads(package.read("manifest.json"))
        arrays = package.read("arrays.npz")
    manifest["schema_version"] = 4
    with zipfile.ZipFile(frozen, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", json.dumps(manifest))
        package.writestr("arrays.npz", arrays)
    return frozen


def test_frozen_replay_is_deterministic_and_preserves_unscorable_axis(tmp_path, monkeypatch):
    rng = np.random.default_rng(912)
    index = pd.date_range("2026-01-01", periods=120, freq="5min")
    values = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=["A", "B", "C"])
    frozen = _frozen_model(tmp_path, values)
    before = frozen.read_bytes()
    replay_input = values.copy()
    replay_input.loc[index[94], "A"] = np.nan
    replay_input.loc[index[95], "B"] = np.inf
    actual_preprocess = replay_module.preprocess_window
    semantics: list[str] = []

    def spy_preprocess(*args, **kwargs):
        semantics.append(kwargs["preprocessing_semantics"])
        return actual_preprocess(*args, **kwargs)

    monkeypatch.setattr(replay_module, "preprocess_window", spy_preprocess)

    first = replay_frozen_model(frozen, replay_input, index[90], index[-1])
    second = replay_frozen_model(frozen, replay_input, index[90], index[-1])

    pd.testing.assert_frame_equal(first.scores, second.scores)
    assert first.summary == second.summary
    assert frozen.read_bytes() == before
    assert first.scores.loc[index[94], "invalid_reason"] == "missing_input"
    assert first.scores.loc[index[95], "invalid_reason"] == "non_finite_input"
    assert not first.scores.loc[index[94], "score_valid"]
    assert semantics == ["legacy", "legacy"]
    assert set(first.scores["overall_status"]).issuperset({"not_scored", "normal"})
    assert first.summary["source_frozen_sha256"] == hashlib.sha256(before).hexdigest()


def test_frozen_replay_rejects_nonfrozen_and_disordered_history(tmp_path):
    index = pd.date_range("2026-01-01", periods=100, freq="5min")
    values = pd.DataFrame(np.random.default_rng(913).normal(size=(100, 3)), index=index, columns=["A", "B", "C"])
    frozen = _frozen_model(tmp_path, values)
    with pytest.raises(ValueError, match="increasing and unique"):
        replay_frozen_model(frozen, values.iloc[[1, 0, *range(2, len(values))]], index[10], index[-1])


def test_frozen_replay_resets_causal_filter_after_a_physical_gap(tmp_path):
    index = pd.date_range("2026-01-01", periods=120, freq="5min")
    values = pd.DataFrame(np.random.default_rng(914).normal(size=(120, 3)), index=index, columns=["A", "B", "C"])
    frozen = _frozen_model(tmp_path, values)

    replay = replay_frozen_model(frozen, values.drop(index[100]), index[90], index[-1])

    assert replay.scores.loc[index[101], "invalid_reason"] == "time_gap_reset"
    assert not replay.scores.loc[index[101], "score_valid"]


def test_frozen_replay_state_filter_can_keep_some_or_no_rows(tmp_path):
    index = pd.date_range("2026-01-01", periods=120, freq="5min")
    history = pd.DataFrame(
        np.random.default_rng(915).normal(size=(120, 3)), index=index, columns=["A", "B", "C"]
    )
    history["state"] = 1.0
    bounds = (index[90], index[110])
    frozen = _frozen_model(
        tmp_path,
        history,
        state_filters=({"column": "state", "minimum": 1.0},),
        filter_method="none",
    )

    all_matched = replay_frozen_model(frozen, history, *bounds)
    assert all_matched.summary["output_row_count"] == len(history.loc[bounds[0]:bounds[1]])

    partial_history = history.copy()
    partial_history.loc[index[96]:index[100], "state"] = 0.0
    partial = replay_frozen_model(frozen, partial_history, *bounds)
    assert partial.summary["state_filter_excluded_rows"] == 5
    assert len(partial.scores) == len(all_matched.scores) - 5

    empty_history = history.copy()
    empty_history["state"] = 0.0
    empty = replay_frozen_model(frozen, empty_history, *bounds)
    assert empty.scores.empty
    assert empty.contributions == []
    assert empty.summary["output_row_count"] == 0
    assert empty.summary["score_valid_count"] == 0
    assert empty.summary["state_filter_excluded_rows"] == len(history.loc[bounds[0]:bounds[1]])
    assert empty.summary["maximum_t2"] is None
    assert empty.summary["maximum_spe"] is None
