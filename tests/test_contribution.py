import numpy as np
import pandas as pd
import pytest

from pca_model_builder.contribution import aggregate_tag_contributions
from pca_model_builder.dpca import DPCAModel


def test_contributions_aggregate_lags_to_original_tag():
    model = DPCAModel(
        feature_names=(
            "T1__lag_000min",
            "T1__lag_005min",
            "T2__lag_000min",
            "T2__lag_005min",
        ),
        mean=np.zeros(4),
        scale=np.ones(4),
        components=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        eigenvalues=np.ones(4),
        explained_variance_ratio=np.full(4, 0.25),
        t2_limits={0.95: 5.0, 0.99: 10.0},
        q_limits={0.95: 5.0, 0.99: 10.0},
        n_samples=100,
    )
    sample = pd.Series(
        [0.0, 20.0, 0.0, 1.0],
        index=model.feature_names,
    )

    contributions = aggregate_tag_contributions(model, sample, statistic="spe")

    assert contributions.iloc[0].tag == "T1"
    assert contributions.iloc[0].contribution_pct > 90
    assert contributions.iloc[0].lag_start_minutes == 5
    assert contributions.iloc[0].lag_end_minutes == 5
    assert contributions.contribution_pct.sum() == pytest.approx(100.0)
