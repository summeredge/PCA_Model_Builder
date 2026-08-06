from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .dpca import DPCAModel
from .scoring_core import (
    aggregate_tag_contributions as aggregate_core_tag_contributions,
    spe_feature_contributions,
    t2_feature_contributions,
)


@dataclass(frozen=True)
class ContributionEvent:
    statistic: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    peak_timestamp: pd.Timestamp
    statistic_value: float
    limit_95: float
    table: pd.DataFrame


def aggregate_tag_contributions(
    model: DPCAModel,
    sample: pd.Series,
    statistic: str,
) -> pd.DataFrame:
    """Aggregate dynamic feature contributions to operator-facing tag totals."""
    missing = [name for name in model.feature_names if name not in sample.index]
    if missing:
        raise ValueError(f"missing model features: {', '.join(missing)}")
    values = sample.loc[list(model.feature_names)].to_numpy(dtype=float)
    if statistic == "t2":
        magnitude = t2_feature_contributions(
            values,
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            components=model.components,
            eigenvalues=model.eigenvalues,
        )
    elif statistic == "spe":
        magnitude = spe_feature_contributions(
            values,
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            components=model.components,
            eigenvalues=model.eigenvalues,
        )
    else:
        raise ValueError("statistic must be 't2' or 'spe'")
    return pd.DataFrame(
        [item.__dict__ for item in aggregate_core_tag_contributions(model.feature_names, magnitude)]
    )


def exceedance_contribution_tables(
    model: DPCAModel,
    dynamic: pd.DataFrame,
    scores: pd.DataFrame,
    sample_interval_minutes: int,
) -> list[ContributionEvent]:
    """Return one peak contribution table per continuous 95% exceedance event."""
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    results: list[ContributionEvent] = []
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    definitions = (
        ("t2", "t2", model.t2_limits[0.95]),
        ("spe", "spe", model.q_limits[0.95]),
    )
    for statistic, column, limit in definitions:
        exceeded = scores[column].to_numpy(dtype=float) >= limit
        start: int | None = None
        for position in range(len(scores) + 1):
            continues = (
                position < len(scores)
                and bool(exceeded[position])
                and (
                    start is None
                    or scores.index[position] - scores.index[position - 1] == expected
                )
            )
            if continues:
                if start is None:
                    start = position
                continue
            if start is not None:
                event_positions = np.arange(start, position)
                values = scores.iloc[event_positions][column].to_numpy(dtype=float)
                peak_position = int(event_positions[np.argmax(values / limit)])
                peak_timestamp = pd.Timestamp(scores.index[peak_position])
                results.append(
                    ContributionEvent(
                        statistic=statistic,
                        event_start=pd.Timestamp(scores.index[start]),
                        event_end=pd.Timestamp(scores.index[position - 1]),
                        peak_timestamp=peak_timestamp,
                        statistic_value=float(scores.iloc[peak_position][column]),
                        limit_95=float(limit),
                        table=aggregate_tag_contributions(
                            model, dynamic.loc[peak_timestamp], statistic=statistic
                        ),
                    )
                )
                start = (
                    position
                    if position < len(scores) and bool(exceeded[position])
                    else None
                )
    return results


def contribution_event_records(
    events: list[ContributionEvent],
    tag_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tag_configs = tag_configs or {}
    records = []
    for event in events:
        tags = []
        for row in event.table.itertuples(index=False):
            config = tag_configs.get(str(row.tag), {})
            tags.append(
                {
                    "tag": str(row.tag),
                    "description": str(config.get("description", "")),
                    "unit": str(config.get("unit", "")),
                    "contribution_pct": float(row.contribution_pct),
                    "lag_start_minutes": int(row.lag_start_minutes),
                    "lag_end_minutes": int(row.lag_end_minutes),
                }
            )
        records.append(
            {
                "statistic": event.statistic,
                "event_start": event.event_start.isoformat(),
                "event_end": event.event_end.isoformat(),
                "peak_timestamp": event.peak_timestamp.isoformat(),
                "statistic_value": event.statistic_value,
                "limit_95": event.limit_95,
                "tags": tags,
            }
        )
    return records
