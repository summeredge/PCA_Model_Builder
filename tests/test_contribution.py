import numpy as np
import pandas as pd
import pytest

from pca_model_builder.contribution import (
    aggregate_tag_contributions,
    contribution_event_records,
    exceedance_contribution_tables,
)
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


def test_only_exceeded_statistic_produces_anomaly_contributions():
    model = DPCAModel(
        feature_names=("T1__lag_000min", "T2__lag_000min", "T3__lag_000min"),
        mean=np.zeros(3),
        scale=np.ones(3),
        components=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        eigenvalues=np.ones(3),
        explained_variance_ratio=np.array([0.45, 0.45, 0.10]),
        t2_limits={0.95: 5.0, 0.99: 10.0},
        q_limits={0.95: 5.0, 0.99: 10.0},
        n_samples=100,
    )
    dynamic = pd.DataFrame(
        [[0.0, 0.0, 4.0]],
        index=[pd.Timestamp("2026-01-01")],
        columns=model.feature_names,
    )
    scores = model.score(dynamic)

    tables = exceedance_contribution_tables(
        model, dynamic, scores, sample_interval_minutes=5
    )

    assert [item.statistic for item in tables] == ["spe"]
    assert tables[0].statistic_value >= tables[0].limit_95


def test_separated_exceedances_create_two_events_and_contiguous_points_do_not():
    model = _event_model()
    index = pd.date_range("2026-01-01", periods=7, freq="5min")
    dynamic = _event_dynamic(index, model)
    scores = pd.DataFrame(
        {"t2": [0.0, 6.0, 7.0, 0.0, 8.0, 9.0, 0.0], "spe": 0.0},
        index=index,
    )

    events = exceedance_contribution_tables(
        model, dynamic, scores, sample_interval_minutes=5
    )

    assert [event.statistic for event in events] == ["t2", "t2"]
    assert [(event.event_start, event.event_end) for event in events] == [
        (index[1], index[2]),
        (index[4], index[5]),
    ]
    assert [event.peak_timestamp for event in events] == [index[2], index[5]]


def test_exceedance_events_do_not_cross_physical_time_gap():
    model = _event_model()
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:25",
            "2026-01-01 00:30",
        ]
    )
    dynamic = _event_dynamic(index, model)
    scores = pd.DataFrame({"t2": 0.0, "spe": [6.0, 7.0, 8.0, 9.0]}, index=index)

    events = exceedance_contribution_tables(
        model, dynamic, scores, sample_interval_minutes=5
    )

    assert [event.statistic for event in events] == ["spe", "spe"]
    assert [(event.event_start, event.event_end) for event in events] == [
        (index[0], index[1]),
        (index[2], index[3]),
    ]


def test_t2_and_spe_events_have_consistent_records_and_percentages():
    model = _event_model()
    index = pd.date_range("2026-01-01", periods=4, freq="5min")
    dynamic = _event_dynamic(index, model)
    scores = pd.DataFrame(
        {"t2": [0.0, 6.0, 0.0, 0.0], "spe": [0.0, 0.0, 7.0, 0.0]},
        index=index,
    )

    records = contribution_event_records(
        exceedance_contribution_tables(
            model, dynamic, scores, sample_interval_minutes=5
        )
    )

    assert [record["statistic"] for record in records] == ["t2", "spe"]
    assert all(
        {
            "statistic",
            "event_start",
            "event_end",
            "peak_timestamp",
            "statistic_value",
            "limit_95",
            "tags",
        }
        == set(record)
        for record in records
    )
    assert all(
        sum(tag["contribution_pct"] for tag in record["tags"])
        == pytest.approx(100.0)
        for record in records
    )


def _event_model() -> DPCAModel:
    return DPCAModel(
        feature_names=("T1__lag_000min", "T2__lag_000min", "T3__lag_000min"),
        mean=np.zeros(3),
        scale=np.ones(3),
        components=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        eigenvalues=np.ones(3),
        explained_variance_ratio=np.array([0.4, 0.35, 0.25]),
        t2_limits={0.95: 5.0, 0.99: 10.0},
        q_limits={0.95: 5.0, 0.99: 10.0},
        n_samples=100,
    )


def _event_dynamic(index: pd.DatetimeIndex, model: DPCAModel) -> pd.DataFrame:
    return pd.DataFrame(
        np.tile([2.0, 1.0, 3.0], (len(index), 1)),
        index=index,
        columns=model.feature_names,
    )
