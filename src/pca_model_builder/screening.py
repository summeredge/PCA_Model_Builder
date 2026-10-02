from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import re
from typing import Any

import numpy as np
import pandas as pd


def screen_performance_states(
    frame: pd.DataFrame,
    conditions: Sequence[Mapping[str, object]],
    sample_interval_minutes: int,
    modeling_eligibility: object = None,
    scope: object = None,
    parent_windows: object = None,
) -> dict[str, Any]:
    """Apply transparent AND range conditions to identify candidate periods."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("performance data index must be a DatetimeIndex")
    if frame.empty:
        raise ValueError("performance data must not be empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("performance timestamps must be sorted and unique")
    if sample_interval_minutes <= 0:
        raise ValueError("sample interval must be positive")
    if not conditions:
        raise ValueError("at least one performance condition is required")

    original_frame = frame
    from .eligibility import normalize_modeling_eligibility
    modeling_eligibility = normalize_modeling_eligibility(modeling_eligibility)
    scope = _normalize_scope(scope)
    parents = _scope_parents(scope, parent_windows)
    segments = None
    if modeling_eligibility is not None or scope["type"] != "all":
        from .eligibility import filter_modeling_eligibility, eligibility_columns
        from .preprocessing import PreprocessingConfig
        if eligibility_columns(modeling_eligibility) or scope["type"] != "all":
            eligible = filter_modeling_eligibility(
                frame, modeling_eligibility,
                PreprocessingConfig(sample_interval_minutes=sample_interval_minutes, resampling_method="mean"),
                allow_empty=scope["type"] != "all",
            )
            frame, segments = eligible.frame, eligible.segment_ids
    if parents is not None:
        selected = pd.Series(False, index=frame.index)
        for parent in parents:
            selected |= frame.index.to_series().between(parent["start"], parent["end"])
        frame = frame.loc[selected]
    normalized = [_normalize_condition(condition) for condition in conditions]
    columns = [condition["column"] for condition in normalized]
    if len(columns) != len(set(columns)):
        raise ValueError("performance condition columns cannot be repeated")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing performance columns: {', '.join(missing)}")

    combined = pd.Series(True, index=frame.index)
    summaries = []
    for condition in normalized:
        column = condition["column"]
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"performance column {column} must contain finite numbers")
        matched = pd.Series(True, index=frame.index)
        if condition["minimum"] is not None:
            matched &= numeric >= condition["minimum"]
        if condition["maximum"] is not None:
            matched &= numeric <= condition["maximum"]
        combined &= matched
        summaries.append({**condition, "matched_rows": int(matched.sum())})

    parent_summaries = []
    candidate_windows = []
    if parents is None:
        windows = _matching_windows(frame.index, combined.to_numpy(), sample_interval_minutes,
                                    segment_ids=segments.to_numpy() if segments is not None else None)
        candidate_windows = [_derived_window(window, None, scope, normalized, modeling_eligibility, sample_interval_minutes, position)
                             for position, window in enumerate(windows, 1)]
        total_rows, matched_rows = len(frame), int(combined.sum())
        original_rows = len(original_frame)
    else:
        for parent in parents:
            mask = frame.index.to_series().between(parent["start"], parent["end"])
            index = frame.index[mask]
            windows = _matching_windows(index, combined.loc[index].to_numpy(), sample_interval_minutes,
                                        segment_ids=segments.reindex(index).to_numpy(), limit=None)
            candidate_windows.extend(_derived_window(window, parent, scope, normalized, modeling_eligibility, sample_interval_minutes, position)
                                     for position, window in enumerate(windows, 1))
            parent_summaries.append({"parent_candidate_id": parent["id"],
                                     "original_rows": int(original_frame.index.to_series().between(parent["start"], parent["end"]).sum()),
                                     "total_rows": len(index), "matched_rows": int(combined.loc[index].sum()),
                                     "derived_candidate_count": len(windows)})
        windows = candidate_windows
        original_rows = sum(item["original_rows"] for item in parent_summaries)
        total_rows = sum(item["total_rows"] for item in parent_summaries)
        matched_rows = sum(item["matched_rows"] for item in parent_summaries)
        for summary in summaries:
            values = frame[summary["column"]].astype(float)
            hits = pd.Series(True, index=frame.index)
            if summary["minimum"] is not None:
                hits &= values.ge(summary["minimum"])
            if summary["maximum"] is not None:
                hits &= values.le(summary["maximum"])
            summary["matched_rows"] = sum(int(hits.loc[parent["start"]:parent["end"]].sum()) for parent in parents)

    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "match_share": matched_rows / total_rows if total_rows else 0.0,
        "conditions": summaries,
        "representative_windows": windows,
        "candidate_windows": candidate_windows,
        "scope": scope,
        "original_rows": original_rows,
        "parent_count": len(parents or []),
        "matched_parent_count": sum(item["matched_rows"] > 0 for item in parent_summaries),
        "unmatched_parent_count": sum(item["matched_rows"] == 0 for item in parent_summaries),
        "parent_summaries": parent_summaries,
        "derived_candidate_count": len(candidate_windows),
        "engineer_decision_required": True,
    }


def _normalize_scope(value: object) -> dict[str, Any]:
    if value is None:
        return {"type": "all"}
    if not isinstance(value, Mapping) or set(value) - {"type", "cluster_ids"}:
        raise ValueError("performance scope must be an object with type and optional cluster_ids")
    kind = value.get("type", "all")
    if not isinstance(kind, str) or kind not in {"all", "candidate_windows", "cluster"}:
        raise ValueError("unknown performance scope type")
    if kind != "cluster":
        return {"type": kind}
    clusters = value.get("cluster_ids")
    if not isinstance(clusters, list) or not clusters or any(not isinstance(item, str) or not re.fullmatch(r"cluster_\d+", item) for item in clusters):
        raise ValueError("cluster scope requires cluster_ids")
    return {"type": kind, "cluster_ids": sorted(set(clusters))}


def candidate_cluster_source(parent: Mapping[str, Any]) -> dict[str, Any]:
    """Read recorded cluster provenance only; never infer clusters from time/data."""
    provenance = parent.get("provenance")
    if parent.get("source") == "performance" and isinstance(provenance, Mapping):
        reference = provenance.get("origin_source_ref") or provenance.get("parent_source_ref")
        source = provenance.get("origin_source") or provenance.get("parent_source")
    else:
        reference, source = parent.get("source_ref"), parent.get("source")
    if source != "cluster" or not isinstance(reference, str):
        return {}
    match = re.fullmatch(r"state-exploration-(.+)-(cluster_\d+)-candidate-(\d+)", reference)
    if match:
        return {"cluster_id": match[2], "exploration_run_id": match[1],
                "origin_candidate_id": f"{match[2]}-candidate-{match[3]}",
                "origin_source": source, "origin_source_ref": reference}
    legacy = re.match(r"^cluster-(\d+)-\d{4}-\d{2}-\d{2}T", reference)
    if legacy:
        return {"cluster_id": f"cluster_{int(legacy[1])}", "origin_source": source, "origin_source_ref": reference}
    return {}


def _scope_parents(scope: Mapping[str, Any], value: object) -> list[dict[str, Any]] | None:
    if scope["type"] == "all":
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("performance parent_windows must be a non-empty list")
    from .windows import _window_bounds
    parents, seen = [], set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("performance parent window must be an object")
        identifier = item.get("id") or item.get("candidate_id")
        source, reference = item.get("source"), item.get("source_ref")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("performance parent candidate_id must be unique")
        if not isinstance(source, str) or not source.strip() or (reference is not None and (not isinstance(reference, str) or not reference.strip())):
            raise ValueError("performance parent source/source_ref is invalid")
        seen.add(identifier)
        start, end = _window_bounds(item.get("start"), item.get("end"))
        parent = {**item, "id": identifier, "start": start, "end": end}
        if scope["type"] == "cluster" and candidate_cluster_source(parent).get("cluster_id") not in scope["cluster_ids"]:
            continue
        parents.append(parent)
    if not parents:
        raise ValueError("selected clusters have no traceable parent candidate windows")
    return parents


def _derived_window(window: Mapping[str, Any], parent: Mapping[str, Any] | None, scope: Mapping[str, Any],
                    conditions: list[dict[str, Any]], eligibility: object, interval: int, position: int) -> dict[str, Any]:
    provenance = {"scope_type": scope["type"], "conditions": conditions,
                  "ordinal_in_parent": position}
    if parent is not None:
        provenance.update({"parent_candidate_id": parent["id"], "parent_source": parent["source"],
                           "parent_source_ref": parent.get("source_ref"), **candidate_cluster_source(parent)})
        if isinstance(parent.get("provenance"), Mapping):
            provenance["parent_provenance"] = dict(parent["provenance"])
    identity = {"window": dict(window), "provenance": provenance, "scope": dict(scope),
                "modeling_eligibility": eligibility, "sample_interval_minutes": interval}
    reference = "performance-" + sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return {**window, "candidate_id": reference, "source": "performance", "source_ref": reference,
            "provenance": provenance}


def _normalize_condition(condition: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(condition, Mapping):
        raise ValueError("each performance condition must be an object")
    column = str(condition.get("column", "")).strip()
    if not column:
        raise ValueError("performance condition requires a column")
    minimum = _optional_float(condition.get("minimum"), column, "minimum")
    maximum = _optional_float(condition.get("maximum"), column, "maximum")
    if minimum is None and maximum is None:
        raise ValueError(f"performance condition for {column} requires a bound")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"performance condition for {column} has reversed bounds")
    return {"column": column, "minimum": minimum, "maximum": maximum}


def _optional_float(value: object, column: str, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"performance {label} for {column} must be numeric") from error
    if not np.isfinite(result):
        raise ValueError(f"performance {label} for {column} must be finite")
    return result


def _matching_windows(
    index: pd.DatetimeIndex,
    matched: np.ndarray,
    sample_interval_minutes: int,
    *, segment_ids: np.ndarray | None = None, limit: int | None = 10,
) -> list[dict[str, Any]]:
    expected = pd.Timedelta(minutes=sample_interval_minutes)
    windows: list[dict[str, Any]] = []
    start: int | None = None
    for position in range(len(index) + 1):
        continues = (
            position < len(index)
            and bool(matched[position])
            and (
                start is None
                or (index[position] - index[position - 1] == expected
                    and (segment_ids is None or segment_ids[position] == segment_ids[position - 1]))
            )
        )
        if continues:
            if start is None:
                start = position
            continue
        if start is not None:
            windows.append(
                {
                    "start": index[start].isoformat(),
                    "end": index[position - 1].isoformat(),
                    "count": position - start,
                }
            )
            start = position if position < len(index) and matched[position] else None
    return sorted(windows, key=lambda item: int(item["count"]), reverse=True)[:limit]
