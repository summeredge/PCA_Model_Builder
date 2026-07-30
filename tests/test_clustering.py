import numpy as np
import pandas as pd
import pytest

from pca_model_builder.clustering import cluster_operating_states


def _two_state_dynamic_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    first = rng.normal(loc=-3.0, scale=0.15, size=(60, 4))
    second = rng.normal(loc=3.0, scale=0.15, size=(60, 4))
    return pd.DataFrame(
        np.vstack([first, second]),
        index=pd.date_range("2026-01-01", periods=120, freq="5min"),
        columns=[f"T{i}__lag_000min" for i in range(4)],
    )


def test_cluster_operating_states_separates_states_and_reports_windows():
    result = cluster_operating_states(_two_state_dynamic_matrix(), n_clusters=2)

    first_label = result.points.iloc[:60].cluster.unique()
    second_label = result.points.iloc[60:].cluster.unique()
    assert len(first_label) == len(second_label) == 1
    assert first_label[0] != second_label[0]
    assert result.n_components >= 2
    assert sum(summary["count"] for summary in result.summaries) == 120
    assert sum(summary["share"] for summary in result.summaries) == pytest.approx(1.0)
    assert all(
        summary["representative_windows"][0]["count"] == 60
        for summary in result.summaries
    )


def test_cluster_operating_states_is_deterministic():
    dynamic = _two_state_dynamic_matrix()

    first = cluster_operating_states(dynamic, n_clusters=2)
    second = cluster_operating_states(dynamic, n_clusters=2)

    pd.testing.assert_frame_equal(first.points, second.points)


def test_representative_windows_do_not_cross_physical_time_gaps():
    dynamic = _two_state_dynamic_matrix()
    dynamic.index = dynamic.index.where(
        np.arange(len(dynamic)) < 30,
        dynamic.index + pd.Timedelta(minutes=30),
    )

    result = cluster_operating_states(dynamic, n_clusters=2)
    first_state = next(
        summary
        for summary in result.summaries
        if result.points.iloc[0].cluster == summary["cluster"]
    )

    assert [
        window["count"] for window in first_state["representative_windows"]
    ] == [30, 30]


def test_cluster_operating_states_rejects_too_many_clusters():
    dynamic = _two_state_dynamic_matrix().iloc[:2]

    with pytest.raises(ValueError, match="more samples"):
        cluster_operating_states(dynamic, n_clusters=2)
