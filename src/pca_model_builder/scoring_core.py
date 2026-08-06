from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np


_FEATURE_PATTERN = re.compile(r"^(?P<tag>.+)__lag_(?P<lag>\d+)min$")
_INVALID_REASONS = frozenset(
    {
        "warming_up",
        "insufficient_context",
        "missing_input",
        "non_finite_input",
        "sampling_mismatch",
        "time_gap_reset",
    }
)
_SEVERITY = {"normal": 0, "attention": 1, "abnormal": 2}
_NOT_SCORED = "not_scored"


@dataclass(frozen=True)
class BatchScoreResult:
    pc_scores: np.ndarray
    t2: np.ndarray
    spe: np.ndarray
    t2_limit_ratio: np.ndarray
    spe_limit_ratio: np.ndarray
    t2_status: tuple[str, ...]
    spe_status: tuple[str, ...]
    overall_status: tuple[str, ...]
    score_valid: tuple[bool, ...]
    invalid_reason: tuple[str | None, ...]


@dataclass(frozen=True)
class SingleScoreResult:
    pc_scores: np.ndarray
    t2: float | None
    spe: float | None
    t2_limit_ratio: float | None
    spe_limit_ratio: float | None
    t2_status: str
    spe_status: str
    overall_status: str
    score_valid: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class TagContribution:
    tag: str
    contribution_pct: float
    lag_start_minutes: int
    lag_end_minutes: int


def score_dynamic_feature_matrix(
    dynamic_features: np.ndarray,
    *,
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    t2_limits: Mapping[float, float],
    q_limits: Mapping[float, float],
) -> BatchScoreResult:
    """Score ordered dynamic feature rows without DataFrame or I/O dependencies."""
    values = np.asarray(dynamic_features, dtype=float)
    if values.ndim != 2:
        raise ValueError("batch dynamic features must be two-dimensional")
    parameter_values = _validate_model_parameters(
        feature_names, mean, scale, components, eigenvalues, t2_limits, q_limits
    )
    mean_values, scale_values, component_values, eigenvalue_values = parameter_values
    feature_count = len(feature_names)
    if values.shape[1] != feature_count:
        raise ValueError("dynamic feature count does not match model feature_names")

    row_count = values.shape[0]
    component_count = component_values.shape[0]
    pc_scores = np.full((row_count, component_count), np.nan, dtype=float)
    t2 = np.full(row_count, np.nan, dtype=float)
    spe = np.full(row_count, np.nan, dtype=float)
    t2_ratio = np.full(row_count, np.nan, dtype=float)
    spe_ratio = np.full(row_count, np.nan, dtype=float)
    t2_status = np.full(row_count, _NOT_SCORED, dtype=object)
    spe_status = np.full(row_count, _NOT_SCORED, dtype=object)
    overall_status = np.full(row_count, _NOT_SCORED, dtype=object)
    score_valid = np.zeros(row_count, dtype=bool)
    invalid_reason: list[str | None] = ["non_finite_input"] * row_count

    valid_rows = np.isfinite(values).all(axis=1)
    if valid_rows.any():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            standardized = (values[valid_rows] - mean_values) / scale_values
            valid_pc_scores = standardized @ component_values.T
            residual = standardized - valid_pc_scores @ component_values
            valid_t2 = np.sum(
                valid_pc_scores**2 / eigenvalue_values[:component_count], axis=1
            )
            valid_spe = np.sum(residual**2, axis=1)
            valid_t2_ratio = valid_t2 / float(t2_limits[0.95])
            valid_spe_ratio = valid_spe / float(q_limits[0.95])
        calculation_valid = (
            np.isfinite(standardized).all(axis=1)
            & np.isfinite(valid_pc_scores).all(axis=1)
            & np.isfinite(residual).all(axis=1)
            & np.isfinite(valid_t2)
            & np.isfinite(valid_spe)
            & np.isfinite(valid_t2_ratio)
            & np.isfinite(valid_spe_ratio)
        )
        valid_positions = np.flatnonzero(valid_rows)[calculation_valid]
        if len(valid_positions):
            valid_t2_status = _statistic_status(
                valid_t2[calculation_valid],
                float(t2_limits[0.95]),
                float(t2_limits[0.99]),
            )
            valid_spe_status = _statistic_status(
                valid_spe[calculation_valid],
                float(q_limits[0.95]),
                float(q_limits[0.99]),
            )
            valid_overall_status = tuple(
                left if _SEVERITY[left] >= _SEVERITY[right] else right
                for left, right in zip(valid_t2_status, valid_spe_status, strict=True)
            )
            pc_scores[valid_positions] = valid_pc_scores[calculation_valid]
            t2[valid_positions] = valid_t2[calculation_valid]
            spe[valid_positions] = valid_spe[calculation_valid]
            t2_ratio[valid_positions] = valid_t2_ratio[calculation_valid]
            spe_ratio[valid_positions] = valid_spe_ratio[calculation_valid]
            t2_status[valid_positions] = valid_t2_status
            spe_status[valid_positions] = valid_spe_status
            overall_status[valid_positions] = valid_overall_status
            score_valid[valid_positions] = True
        for position in valid_positions:
            invalid_reason[int(position)] = None

    return BatchScoreResult(
        pc_scores=_immutable_array(pc_scores),
        t2=_immutable_array(t2),
        spe=_immutable_array(spe),
        t2_limit_ratio=_immutable_array(t2_ratio),
        spe_limit_ratio=_immutable_array(spe_ratio),
        t2_status=tuple(str(value) for value in t2_status),
        spe_status=tuple(str(value) for value in spe_status),
        overall_status=tuple(str(value) for value in overall_status),
        score_valid=tuple(bool(value) for value in score_valid),
        invalid_reason=tuple(invalid_reason),
    )


def score_dynamic_feature_vector(
    dynamic_feature: np.ndarray,
    **model_parameters: object,
) -> SingleScoreResult:
    """Score one ordered dynamic feature vector."""
    values = np.asarray(dynamic_feature, dtype=float)
    if values.ndim != 1:
        raise ValueError("single dynamic feature must be one-dimensional")
    result = score_dynamic_feature_matrix(values[np.newaxis, :], **model_parameters)
    return _single_result_at(result, 0)


def unscorable_score(invalid_reason: str, n_components: int) -> SingleScoreResult:
    """Build the explicit null result used when preprocessing cannot score a row."""
    if invalid_reason not in _INVALID_REASONS:
        raise ValueError("invalid_reason is not defined by the scoring contract")
    if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    return SingleScoreResult(
        pc_scores=_immutable_array(np.full(n_components, np.nan, dtype=float)),
        t2=None,
        spe=None,
        t2_limit_ratio=None,
        spe_limit_ratio=None,
        t2_status=_NOT_SCORED,
        spe_status=_NOT_SCORED,
        overall_status=_NOT_SCORED,
        score_valid=False,
        invalid_reason=invalid_reason,
    )


def t2_feature_contributions(
    dynamic_feature: np.ndarray,
    *,
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return ordered T² contribution magnitudes for one dynamic feature vector."""
    standardized, component_values, eigenvalue_values = _standardize_for_contribution(
        dynamic_feature, feature_names, mean, scale, components, eigenvalues
    )
    inverse_covariance = (
        component_values.T
        @ np.diag(1.0 / eigenvalue_values[: component_values.shape[0]])
        @ component_values
    )
    return _immutable_array(np.abs(standardized * (inverse_covariance @ standardized)))


def spe_feature_contributions(
    dynamic_feature: np.ndarray,
    *,
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return ordered SPE contribution magnitudes for one dynamic feature vector."""
    standardized, component_values, _ = _standardize_for_contribution(
        dynamic_feature, feature_names, mean, scale, components, eigenvalues
    )
    principal_scores = standardized @ component_values.T
    residual = standardized - principal_scores @ component_values
    return _immutable_array(residual**2)


def anomaly_tag_contributions(
    dynamic_feature: np.ndarray,
    score: SingleScoreResult,
    statistic: str,
    *,
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    limit_95: float,
) -> tuple[TagContribution, ...]:
    """Return one statistic's Tag contributions only for a valid 95% exceedance."""
    if statistic not in {"t2", "spe"}:
        raise ValueError("statistic must be 't2' or 'spe'")
    if not _is_finite_number(limit_95):
        raise ValueError("95% control limit must be finite")
    if (statistic == "t2" and limit_95 <= 0) or (
        statistic == "spe" and limit_95 < 0
    ):
        raise ValueError("95% control limit is invalid")
    value = score.t2 if statistic == "t2" else score.spe
    if not score.score_valid or value is None or not _is_finite_number(value):
        return ()
    if float(value) < float(limit_95):
        return ()
    contributions = (
        t2_feature_contributions(
            dynamic_feature,
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            components=components,
            eigenvalues=eigenvalues,
        )
        if statistic == "t2"
        else spe_feature_contributions(
            dynamic_feature,
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            components=components,
            eigenvalues=eigenvalues,
        )
    )
    total = float(contributions.sum())
    if not np.isfinite(total) or total <= 0:
        return ()
    return aggregate_tag_contributions(feature_names, contributions)


def aggregate_tag_contributions(
    feature_names: Sequence[str], feature_contributions: np.ndarray
) -> tuple[TagContribution, ...]:
    """Aggregate ordered dynamic-feature magnitudes back to original Tags."""
    values = np.asarray(feature_contributions, dtype=float)
    if values.ndim != 1:
        raise ValueError("feature contributions must be one-dimensional")
    if len(values) != len(feature_names):
        raise ValueError("feature contribution count does not match model feature_names")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("feature contributions must be finite and non-negative")

    groups: dict[str, list[tuple[int, float]]] = {}
    for name, value in zip(feature_names, values, strict=True):
        match = _FEATURE_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid dynamic feature name: {name}")
        groups.setdefault(match.group("tag"), []).append(
            (int(match.group("lag")), float(value))
        )

    total = float(values.sum())
    results: list[TagContribution] = []
    for tag, entries in groups.items():
        entries.sort()
        magnitudes = np.array([value for _, value in entries], dtype=float)
        peak_position = int(magnitudes.argmax())
        threshold = float(magnitudes[peak_position]) * 0.5
        start = peak_position
        end = peak_position
        while start > 0 and magnitudes[start - 1] >= threshold:
            start -= 1
        while end + 1 < len(entries) and magnitudes[end + 1] >= threshold:
            end += 1
        tag_total = float(magnitudes.sum())
        results.append(
            TagContribution(
                tag=tag,
                contribution_pct=(0.0 if total == 0.0 else tag_total / total * 100.0),
                lag_start_minutes=entries[start][0],
                lag_end_minutes=entries[end][0],
            )
        )
    return tuple(sorted(results, key=lambda item: (-item.contribution_pct, item.tag)))


def _single_result_at(result: BatchScoreResult, position: int) -> SingleScoreResult:
    if result.score_valid[position]:
        return SingleScoreResult(
            pc_scores=_immutable_array(result.pc_scores[position]),
            t2=float(result.t2[position]),
            spe=float(result.spe[position]),
            t2_limit_ratio=float(result.t2_limit_ratio[position]),
            spe_limit_ratio=float(result.spe_limit_ratio[position]),
            t2_status=result.t2_status[position],
            spe_status=result.spe_status[position],
            overall_status=result.overall_status[position],
            score_valid=True,
            invalid_reason=None,
        )
    return unscorable_score(str(result.invalid_reason[position]), result.pc_scores.shape[1])


def _standardize_for_contribution(
    dynamic_feature: np.ndarray,
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(dynamic_feature, dtype=float)
    if values.ndim != 1:
        raise ValueError("single dynamic feature must be one-dimensional")
    mean_values, scale_values, component_values, eigenvalue_values = _validate_model_parameters(
        feature_names,
        mean,
        scale,
        components,
        eigenvalues,
    )
    if values.shape[0] != len(feature_names):
        raise ValueError("dynamic feature count does not match model feature_names")
    if not np.isfinite(values).all():
        raise ValueError("model inputs contain non-finite values")
    return (values - mean_values) / scale_values, component_values, eigenvalue_values


def _validate_model_parameters(
    feature_names: Sequence[str],
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    t2_limits: Mapping[float, float] | None = None,
    q_limits: Mapping[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_count = len(feature_names)
    mean_values = np.asarray(mean, dtype=float)
    scale_values = np.asarray(scale, dtype=float)
    component_values = np.asarray(components, dtype=float)
    eigenvalue_values = np.asarray(eigenvalues, dtype=float)
    if mean_values.shape != (feature_count,) or scale_values.shape != (feature_count,):
        raise ValueError("model standardization arrays do not match feature_names")
    if component_values.ndim != 2 or component_values.shape[1] != feature_count:
        raise ValueError("model components do not match feature_names")
    if component_values.shape[0] < 1 or eigenvalue_values.ndim != 1 or len(eigenvalue_values) < component_values.shape[0]:
        raise ValueError("model eigenvalues do not match components")
    if not (
        np.isfinite(mean_values).all()
        and np.isfinite(scale_values).all()
        and np.isfinite(component_values).all()
        and np.isfinite(eigenvalue_values).all()
    ):
        raise ValueError("model arrays must be finite")
    if np.any(scale_values <= 0) or np.any(eigenvalue_values[: component_values.shape[0]] <= 0):
        raise ValueError("model scale and retained eigenvalues must be positive")
    if (t2_limits is None) != (q_limits is None):
        raise ValueError("both model control limit mappings must be provided")
    if t2_limits is not None:
        _validate_control_limits(t2_limits, statistic="t2")
        _validate_control_limits(q_limits, statistic="spe")
    return mean_values, scale_values, component_values, eigenvalue_values


def _validate_control_limits(
    limits: Mapping[float, float], *, statistic: str
) -> None:
    if not isinstance(limits, Mapping) or set(limits) != {0.95, 0.99}:
        raise ValueError("model control limits must contain only 95% and 99% values")
    limit_95 = limits[0.95]
    limit_99 = limits[0.99]
    if not _is_finite_number(limit_95) or not _is_finite_number(limit_99):
        raise ValueError("model control limits must be finite numeric values")
    if statistic == "t2":
        valid = 0 < float(limit_95) <= float(limit_99)
    else:
        valid = 0 <= float(limit_95) <= float(limit_99)
    if not valid:
        raise ValueError(f"model {statistic.upper()} control limits are invalid")


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _statistic_status(values: np.ndarray, limit_95: float, limit_99: float) -> tuple[str, ...]:
    return tuple(
        "abnormal" if value >= limit_99 else "attention" if value >= limit_95 else "normal"
        for value in values
    )


def _immutable_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result
