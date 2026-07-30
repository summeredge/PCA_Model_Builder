import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca


def _training_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=300)
    x2 = 2.0 * x1 + rng.normal(scale=0.08, size=300)
    x3 = rng.normal(scale=0.15, size=300)
    return pd.DataFrame({"A__lag_000min": x1, "B__lag_000min": x2, "C__lag_000min": x3})


def test_fit_selects_components_and_scores_training_data():
    frame = _training_frame()

    model = fit_dpca(frame, variance_threshold=0.95)
    scores = model.score(frame)

    assert model.n_components == 2
    assert {
        "pc1",
        "pc2",
        "t2",
        "spe",
        "t2_limit_ratio",
        "spe_limit_ratio",
        "t2_status",
        "spe_status",
        "status",
    }.issubset(scores.columns)
    assert model.t2_limits[0.99] > model.t2_limits[0.95] > 0
    assert model.q_limits[0.99] > model.q_limits[0.95] >= 0


def test_mean_shift_increases_t2_and_unmodelled_shift_increases_spe():
    frame = _training_frame()
    model = fit_dpca(frame, n_components=1)
    baseline = model.score(frame.iloc[[0]])

    principal_shift = frame.iloc[[0]].copy()
    principal_shift[["A__lag_000min", "B__lag_000min"]] += [8.0, 16.0]
    residual_shift = frame.iloc[[0]].copy()
    residual_shift["C__lag_000min"] += 8.0

    assert model.score(principal_shift).iloc[0].t2 > baseline.iloc[0].t2
    assert model.score(residual_shift).iloc[0].spe > baseline.iloc[0].spe


def test_status_uses_more_severe_of_t2_and_spe():
    frame = _training_frame()
    model = fit_dpca(frame, n_components=1)
    abnormal = frame.iloc[[0]].copy()
    abnormal["C__lag_000min"] += 20.0

    scored = model.score(abnormal).iloc[0]

    assert scored.spe_status == "abnormal"
    assert scored.spe_limit_ratio >= 1
    assert scored.status == "abnormal"


def test_fit_rejects_model_without_effective_residual_space():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(100, 2)), columns=["A", "B"])

    with pytest.raises(ValueError, match="residual space"):
        fit_dpca(frame, n_components=2)
