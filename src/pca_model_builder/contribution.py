from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .dpca import DPCAModel


_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")


def aggregate_tag_contributions(
    model: DPCAModel,
    sample: pd.Series,
    statistic: str,
) -> pd.DataFrame:
    """Aggregate dynamic feature contributions to operator-facing tag totals."""
    one_row = sample.to_frame().T
    standardized = model.standardize(one_row)[0]
    principal_scores = standardized @ model.components.T

    if statistic == "t2":
        inverse_covariance = (
            model.components.T
            @ np.diag(1.0 / model.eigenvalues[: model.n_components])
            @ model.components
        )
        magnitude = np.abs(standardized * (inverse_covariance @ standardized))
    elif statistic == "spe":
        residual = standardized - principal_scores @ model.components
        magnitude = residual**2
    else:
        raise ValueError("statistic must be 't2' or 'spe'")

    feature_rows: list[dict[str, float | int | str]] = []
    for name, value in zip(model.feature_names, magnitude, strict=True):
        match = _FEATURE_PATTERN.match(name)
        if match is None:
            raise ValueError(f"invalid dynamic feature name: {name}")
        feature_rows.append(
            {
                "tag": match.group("tag"),
                "lag": int(match.group("lag")),
                "magnitude": float(value),
            }
        )

    features = pd.DataFrame(feature_rows)
    total = float(features["magnitude"].sum())
    results: list[dict[str, float | int | str]] = []
    for tag, group in features.groupby("tag", sort=False):
        ordered = group.sort_values("lag").reset_index(drop=True)
        tag_total = float(ordered["magnitude"].sum())
        peak_position = int(ordered["magnitude"].to_numpy().argmax())
        threshold = float(ordered.loc[peak_position, "magnitude"]) * 0.5
        start = peak_position
        end = peak_position
        while start > 0 and ordered.loc[start - 1, "magnitude"] >= threshold:
            start -= 1
        while end + 1 < len(ordered) and ordered.loc[end + 1, "magnitude"] >= threshold:
            end += 1
        results.append(
            {
                "tag": tag,
                "contribution_pct": 0.0 if total == 0 else tag_total / total * 100.0,
                "lag_start_minutes": int(ordered.loc[start, "lag"]),
                "lag_end_minutes": int(ordered.loc[end, "lag"]),
            }
        )

    return pd.DataFrame(results).sort_values(
        "contribution_pct", ascending=False, ignore_index=True
    )


def exceedance_contribution_tables(
    model: DPCAModel,
    dynamic: pd.DataFrame,
    scores: pd.DataFrame,
) -> list[tuple[str, pd.Timestamp, float, float, pd.DataFrame]]:
    """Return one contribution table per statistic that actually exceeds 95%."""
    results = []
    definitions = (
        ("t2", "t2", model.t2_limits[0.95]),
        ("spe", "spe", model.q_limits[0.95]),
    )
    for statistic, column, limit in definitions:
        exceeded = scores[column] >= limit
        if not exceeded.any():
            continue
        timestamp = pd.Timestamp((scores.loc[exceeded, column] / limit).idxmax())
        table = aggregate_tag_contributions(
            model, dynamic.loc[timestamp], statistic=statistic
        )
        results.append(
            (statistic, timestamp, float(scores.loc[timestamp, column]), float(limit), table)
        )
    return results
