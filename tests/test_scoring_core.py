from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.contribution import (
    aggregate_tag_contributions as aggregate_dataframe_contributions,
    exceedance_contribution_tables,
)
from pca_model_builder.dpca import DPCAModel
from pca_model_builder.scoring_core import (
    aggregate_tag_contributions,
    anomaly_tag_contributions,
    score_dynamic_feature_matrix,
    score_dynamic_feature_vector,
    spe_feature_contributions,
    t2_feature_contributions,
    unscorable_score,
)


def _model() -> DPCAModel:
    return DPCAModel(
        feature_names=(
            "A__lag_000min",
            "A__lag_005min",
            "B__lag_000min",
            "B__lag_005min",
        ),
        mean=np.array([1.0, -1.0, 0.5, 2.0]),
        scale=np.array([2.0, 4.0, 0.5, 2.0]),
        components=np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        eigenvalues=np.array([2.0, 3.0, 0.5, 0.25]),
        explained_variance_ratio=np.array([0.5, 0.3, 0.15, 0.05]),
        t2_limits={0.95: 5.0, 0.99: 10.0},
        q_limits={0.95: 5.0, 0.99: 10.0},
        n_samples=100,
    )


def _parameters(model: DPCAModel) -> dict[str, object]:
    return {
        "feature_names": model.feature_names,
        "mean": model.mean,
        "scale": model.scale,
        "components": model.components,
        "eigenvalues": model.eigenvalues,
        "t2_limits": model.t2_limits,
        "q_limits": model.q_limits,
    }


def test_batch_and_single_scores_match_and_match_original_formula():
    model = _model()
    values = np.array([[1.0, -1.0, 0.5, 2.0], [5.0, 7.0, 1.5, 6.0]])

    batch = score_dynamic_feature_matrix(values, **_parameters(model))
    standardized = (values - model.mean) / model.scale
    expected_pc = standardized @ model.components.T
    expected_residual = standardized - expected_pc @ model.components
    expected_t2 = np.sum(expected_pc**2 / model.eigenvalues[: model.n_components], axis=1)
    expected_spe = np.sum(expected_residual**2, axis=1)

    np.testing.assert_allclose(batch.pc_scores, expected_pc)
    np.testing.assert_allclose(batch.t2, expected_t2)
    np.testing.assert_allclose(batch.spe, expected_spe)
    np.testing.assert_allclose(batch.t2_limit_ratio, expected_t2 / 5.0)
    np.testing.assert_allclose(batch.spe_limit_ratio, expected_spe / 5.0)
    compatibility_scores = model.score(pd.DataFrame(values, columns=model.feature_names))
    np.testing.assert_allclose(compatibility_scores[["pc1", "pc2"]], expected_pc)
    np.testing.assert_allclose(compatibility_scores["t2"], expected_t2)
    np.testing.assert_allclose(compatibility_scores["spe"], expected_spe)
    for position, row in enumerate(values):
        single = score_dynamic_feature_vector(row, **_parameters(model))
        np.testing.assert_allclose(single.pc_scores, batch.pc_scores[position])
        assert single.t2 == batch.t2[position]
        assert single.spe == batch.spe[position]
        assert single.t2_limit_ratio == batch.t2_limit_ratio[position]
        assert single.spe_limit_ratio == batch.spe_limit_ratio[position]
        assert single.t2_status == batch.t2_status[position]
        assert single.spe_status == batch.spe_status[position]
        assert single.overall_status == batch.overall_status[position]


def test_status_boundaries_and_overall_severity():
    model = _model()
    values = np.array(
        [
            [1.0 + 2.0 * np.sqrt(10.0), -1.0, 0.5, 2.0],
            [1.0 + 2.0 * np.sqrt(20.0), -1.0, 0.5, 2.0],
            [1.0 + 2.0 * np.sqrt(20.0), -1.0, 0.5, 2.0 + np.sqrt(5.0) * 2.0],
            [1.0 + 2.0 * np.sqrt(20.0), -1.0, 0.5, 2.0 + np.sqrt(10.0) * 2.0],
        ]
    )
    result = score_dynamic_feature_matrix(values, **_parameters(model))

    assert result.t2_status[:2] == ("attention", "abnormal")
    assert result.spe_status[0] == "normal"
    assert result.spe_status[2:] == ("attention", "abnormal")
    assert result.overall_status[2:] == ("abnormal", "abnormal")


def test_invalid_batch_row_does_not_block_valid_rows_or_mutate_inputs():
    model = _model()
    values = np.array([[1.0, -1.0, 0.5, 2.0], [np.nan, 0.0, 0.0, 0.0]])
    values_before = values.copy()
    parameters_before = {key: np.array(value, copy=True) for key, value in _parameters(model).items() if isinstance(value, np.ndarray)}

    result = score_dynamic_feature_matrix(values, **_parameters(model))

    assert result.score_valid == (True, False)
    assert result.invalid_reason == (None, "non_finite_input")
    assert result.overall_status[1] != "normal"
    assert np.isnan(result.t2[1]) and np.isnan(result.spe[1])
    np.testing.assert_array_equal(values, values_before)
    for key, before in parameters_before.items():
        np.testing.assert_array_equal(getattr(model, key), before)


@pytest.mark.parametrize(
    ("field", "limits"),
    (
        ("t2_limits", {0.95: 10.0, 0.99: 5.0}),
        ("q_limits", {0.95: 5.0, 0.99: -1.0}),
        ("t2_limits", {0.95: 0.0, 0.99: 10.0}),
        ("q_limits", {0.95: -1.0, 0.99: 10.0}),
        ("t2_limits", {0.95: np.nan, 0.99: 10.0}),
        ("q_limits", {0.95: 5.0, 0.99: np.inf}),
        ("t2_limits", {0.95: True, 0.99: 10.0}),
        ("q_limits", {0.95: 5.0, 0.99: 10.0, 0.9: 4.0}),
    ),
)
def test_score_rejects_invalid_control_limits(field, limits):
    model = _model()
    parameters = _parameters(model)
    parameters[field] = limits

    with pytest.raises(ValueError, match="control limits"):
        score_dynamic_feature_matrix(
            np.array([[1.0, -1.0, 0.5, 2.0]]), **parameters
        )


def test_non_finite_calculations_do_not_make_a_valid_score():
    model = _model()
    parameters = _parameters(model)
    parameters["scale"] = np.array([2.0, 4.0, np.finfo(float).tiny, 2.0])
    values = np.array(
        [
            [1.0, -1.0, 0.5, 2.0],
            [1.0, -1.0, np.finfo(float).max, 2.0],
        ]
    )

    result = score_dynamic_feature_matrix(values, **parameters)

    assert result.score_valid == (True, False)
    assert result.invalid_reason == (None, "non_finite_input")
    assert result.t2_status[1] == "not_scored"
    assert result.spe_status[1] == "not_scored"
    assert result.overall_status[1] == "not_scored"
    assert np.isnan(result.t2[1]) and np.isnan(result.spe_limit_ratio[1])


def test_zero_spe_limit_never_produces_an_undefined_valid_score():
    model = _model()
    parameters = _parameters(model)
    parameters["q_limits"] = {0.95: 0.0, 0.99: 0.0}

    result = score_dynamic_feature_matrix(
        np.array([[1.0, -1.0, 0.5, 2.0]]), **parameters
    )

    assert result.score_valid == (False,)
    assert result.invalid_reason == ("non_finite_input",)
    assert result.overall_status == ("not_scored",)


@pytest.mark.parametrize(
    "reason",
    (
        "warming_up",
        "insufficient_context",
        "missing_input",
        "non_finite_input",
        "sampling_mismatch",
        "time_gap_reset",
    ),
)
def test_unscorable_reasons_never_return_normal(reason):
    result = unscorable_score(reason, n_components=2)

    assert not result.score_valid
    assert result.invalid_reason == reason
    assert result.overall_status != "normal"
    assert result.t2 is None and result.spe is None


def test_score_inputs_reject_wrong_dimension_or_feature_count():
    model = _model()
    with pytest.raises(ValueError, match="two-dimensional"):
        score_dynamic_feature_matrix(np.zeros(4), **_parameters(model))
    with pytest.raises(ValueError, match="one-dimensional"):
        score_dynamic_feature_vector(np.zeros((1, 4)), **_parameters(model))
    with pytest.raises(ValueError, match="feature count"):
        score_dynamic_feature_matrix(np.zeros((1, 3)), **_parameters(model))
    with pytest.raises(ValueError, match="feature count"):
        score_dynamic_feature_vector(np.zeros(3), **_parameters(model))
    with pytest.raises(ValueError, match="not defined"):
        unscorable_score("other", n_components=2)


def test_pure_contributions_match_existing_formula_and_tag_aggregation():
    model = _model()
    values = np.array([3.0, 7.0, 1.5, 6.0])
    standardized = (values - model.mean) / model.scale
    principal_scores = standardized @ model.components.T
    inverse_covariance = (
        model.components.T
        @ np.diag(1.0 / model.eigenvalues[: model.n_components])
        @ model.components
    )
    expected_t2 = np.abs(standardized * (inverse_covariance @ standardized))
    expected_spe = (standardized - principal_scores @ model.components) ** 2

    t2 = t2_feature_contributions(values, **{key: value for key, value in _parameters(model).items() if key not in {"t2_limits", "q_limits"}})
    spe = spe_feature_contributions(values, **{key: value for key, value in _parameters(model).items() if key not in {"t2_limits", "q_limits"}})
    np.testing.assert_allclose(t2, expected_t2)
    np.testing.assert_allclose(spe, expected_spe)

    aggregate = aggregate_tag_contributions(model.feature_names, spe)
    assert sum(item.contribution_pct for item in aggregate) == pytest.approx(100.0)
    assert [(item.tag, item.contribution_pct) for item in aggregate] == sorted(
        ((item.tag, item.contribution_pct) for item in aggregate),
        key=lambda item: (-item[1], item[0]),
    )
    dataframe_result = aggregate_dataframe_contributions(
        model, pd.Series(values, index=model.feature_names), statistic="spe"
    )
    assert dataframe_result.to_dict("records") == [item.__dict__ for item in aggregate]
    tied = aggregate_tag_contributions(model.feature_names, np.array([1.0, 0.0, 1.0, 0.0]))
    assert [item.tag for item in tied] == ["A", "B"]
    assert sum(item.contribution_pct for item in tied) == pytest.approx(100.0)


def test_single_point_contributions_require_the_matching_statistic_threshold():
    model = _model()
    parameters = _parameters(model)
    contribution_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {"t2_limits", "q_limits"}
    }

    def contributions(values, score, statistic):
        return anomaly_tag_contributions(
            values,
            score,
            statistic,
            **contribution_parameters,
            limit_95=(
                model.t2_limits[0.95]
                if statistic == "t2"
                else model.q_limits[0.95]
            ),
        )

    t2_only = np.array([1.0 + 2.0 * np.sqrt(20.0), -1.0, 0.5, 2.0])
    t2_score = score_dynamic_feature_vector(t2_only, **parameters)
    assert contributions(t2_only, t2_score, "t2")
    assert contributions(t2_only, t2_score, "spe") == ()

    spe_only = np.array([1.0, -1.0, 0.5, 2.0 + 2.0 * np.sqrt(10.0)])
    spe_score = score_dynamic_feature_vector(spe_only, **parameters)
    assert contributions(spe_only, spe_score, "t2") == ()
    assert contributions(spe_only, spe_score, "spe")

    mean_point = np.array([1.0, -1.0, 0.5, 2.0])
    mean_score = score_dynamic_feature_vector(mean_point, **parameters)
    assert contributions(mean_point, mean_score, "t2") == ()
    assert contributions(mean_point, mean_score, "spe") == ()
    assert contributions(
        mean_point, replace(mean_score, t2=model.t2_limits[0.95]), "t2"
    ) == ()

    invalid_score = score_dynamic_feature_vector(
        np.array([np.nan, -1.0, 0.5, 2.0]), **parameters
    )
    assert contributions(mean_point, invalid_score, "t2") == ()

    tied = np.array(
        [1.0 + 2.0 * np.sqrt(20.0), -1.0, 0.5 + 0.5 * np.sqrt(30.0), 2.0]
    )
    tied_contributions = contributions(
        tied, score_dynamic_feature_vector(tied, **parameters), "t2"
    )
    assert [item.tag for item in tied_contributions] == ["A", "B"]
    assert sum(item.contribution_pct for item in tied_contributions) == pytest.approx(
        100.0
    )


def test_exceedance_contributions_require_the_corresponding_95_percent_limit():
    model = _model()
    dynamic = pd.DataFrame(
        [[1.0, -1.0, 0.5, 2.0]],
        index=[pd.Timestamp("2026-01-01")],
        columns=model.feature_names,
    )

    assert exceedance_contribution_tables(
        model, dynamic, model.score(dynamic), sample_interval_minutes=5
    ) == []


def test_dpca_score_preserves_status_alias_and_adds_contract_fields():
    model = _model()
    frame = pd.DataFrame(
        [[1.0, -1.0, 0.5, 2.0], [np.inf, 0.0, 0.0, 0.0]],
        columns=model.feature_names,
    )

    scores = model.score(frame)

    assert (scores["status"] == scores["overall_status"]).all()
    assert scores.loc[0, "score_valid"]
    assert not scores.loc[1, "score_valid"]
    assert scores.loc[1, "invalid_reason"] == "non_finite_input"
