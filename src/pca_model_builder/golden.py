"""Read-only acceptance checks for the committed golden DPCA vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .model_io import (
    CONTRIBUTION_RULES,
    STATUS_RULES,
    load_deployment_package,
    load_model_package,
)
from .preprocessing import preprocess_window, preprocessing_config_from_mapping
from .replay import replay_frozen_model
from .scoring_core import SingleScoreResult, anomaly_tag_contributions


_FILES = frozenset(
    {
        "fixture_manifest.json",
        "frozen_model.pcamodel",
        "deployment_model.pcadeploy",
        "raw_input.csv",
        "expected_dynamic_features.csv",
        "expected_scores.csv",
        "expected_contributions.json",
        "expected_summary.json",
    }
)
_HASHED_FILES = _FILES
_MANIFEST_FIELDS = {
    "golden_schema_version",
    "fixture_id",
    "model_id",
    "model_version",
    "replay_start",
    "replay_end",
    "timestamp_column",
    "input_tags",
    "dynamic_feature_names",
    "numeric_tolerance",
    "files",
    "manifest_sha256",
}
_RTOL = 1e-9
_ATOL = 1e-12


def verify_golden_vectors(bundle_path: str | Path) -> dict[str, Any]:
    """Verify a committed golden bundle without modifying any of its files.

    The manifest hashes every member.  Its own entry is a canonical SHA-256 that
    excludes only its self-checksum fields, avoiding a self-reference cycle.
    """
    bundle = Path(bundle_path)
    _validate_bundle_directory(bundle)
    manifest = _read_json(bundle / "fixture_manifest.json")
    _validate_manifest(manifest)
    _verify_hashes(bundle, manifest["files"])

    frozen_path = bundle / "frozen_model.pcamodel"
    deployment_path = bundle / "deployment_model.pcadeploy"
    frozen_before = frozen_path.read_bytes()
    deployment_before = deployment_path.read_bytes()
    try:
        frozen_model, frozen_manifest = load_model_package(frozen_path)
        deployment_model, deployment_manifest = load_deployment_package(deployment_path)
        _validate_models(manifest, frozen_model, frozen_manifest, deployment_model, deployment_manifest, frozen_before)

        raw = _read_raw_input(bundle / "raw_input.csv", manifest, deployment_manifest)
        replay = replay_frozen_model(
            frozen_path, raw, manifest["replay_start"], manifest["replay_end"]
        )
        dynamic, deployment_scores = _deployment_scores(
            raw, manifest, deployment_model, deployment_manifest
        )
        expected_dynamic = _read_dynamic(bundle / "expected_dynamic_features.csv", manifest)
        expected_scores = _read_scores(bundle / "expected_scores.csv", frozen_model.n_components)
        expected_contributions = _read_json(bundle / "expected_contributions.json")
        expected_summary = _read_json(bundle / "expected_summary.json")

        _compare_dynamic(expected_dynamic, dynamic)
        _compare_scores(expected_scores, replay.scores, "frozen replay")
        _compare_summary(expected_summary, replay.summary)
        _compare_json_value(expected_contributions, replay.contributions, "frozen replay contributions")
        _compare_deployment_scores(replay.scores, deployment_scores)
        _compare_scores(expected_scores.loc[deployment_scores.index], deployment_scores, "deployment scoring")
        _compare_contributions(
            _contribution_core(expected_contributions),
            _deployment_contributions(dynamic, deployment_scores, deployment_model),
            "deployment scoring",
        )
    finally:
        if frozen_path.read_bytes() != frozen_before:
            raise RuntimeError("frozen model package changed during golden verification")
        if deployment_path.read_bytes() != deployment_before:
            raise RuntimeError("deployment package changed during golden verification")

    return {
        "acceptance_status": "passed",
        "fixture_id": manifest["fixture_id"],
        "model_id": manifest["model_id"],
        "model_version": manifest["model_version"],
        "replay_row_count": int(len(replay.scores)),
        "score_valid_count": int(replay.scores["score_valid"].sum()),
        "dynamic_row_count": int(len(dynamic)),
        "max_absolute_error": {
            "dynamic_features": _max_abs(expected_dynamic.to_numpy(dtype=float), dynamic.to_numpy(dtype=float)),
            "pc_scores": _max_abs_scores(expected_scores, replay.scores, frozen_model.n_components),
            "t2": _max_abs_column(expected_scores, replay.scores, "t2"),
            "spe": _max_abs_column(expected_scores, replay.scores, "spe"),
            "t2_limit_ratio": _max_abs_column(expected_scores, replay.scores, "t2_limit_ratio"),
            "spe_limit_ratio": _max_abs_column(expected_scores, replay.scores, "spe_limit_ratio"),
            "contribution_pct": _max_contribution_error(expected_contributions, replay.contributions),
        },
    }


def _validate_bundle_directory(bundle: Path) -> None:
    if not bundle.is_dir():
        raise ValueError("golden vector bundle must be a directory")
    names = {path.name for path in bundle.iterdir()}
    if names != _FILES or any(not path.is_file() for path in bundle.iterdir()):
        raise ValueError("golden vector bundle files are unexpected or incomplete")


def _read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"golden JSON cannot be read: {path.name}") from error
    _validate_finite_json(value, path.name)
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _validate_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"golden JSON contains a non-finite value: {label}")
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item, label)


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("golden fixture manifest fields are invalid")
    if manifest["golden_schema_version"] != 1:
        raise ValueError("unsupported golden fixture schema version")
    if not isinstance(manifest["fixture_id"], str) or not manifest["fixture_id"]:
        raise ValueError("golden fixture_id is invalid")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"]:
        raise ValueError("golden model_id is invalid")
    if not isinstance(manifest["model_version"], int) or isinstance(manifest["model_version"], bool) or manifest["model_version"] < 1:
        raise ValueError("golden model_version is invalid")
    try:
        start, end = pd.Timestamp(manifest["replay_start"]), pd.Timestamp(manifest["replay_end"])
    except (TypeError, ValueError) as error:
        raise ValueError("golden replay timestamps are invalid") from error
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("golden replay timestamps are invalid")
    if not isinstance(manifest["timestamp_column"], str) or not manifest["timestamp_column"]:
        raise ValueError("golden timestamp column is invalid")
    for field in ("input_tags", "dynamic_feature_names"):
        values = manifest[field]
        if not isinstance(values, list) or not values or not all(isinstance(item, str) and item for item in values) or len(values) != len(set(values)):
            raise ValueError(f"golden {field} is invalid")
    tolerance = manifest["numeric_tolerance"]
    if tolerance != {"rtol": _RTOL, "atol": _ATOL}:
        raise ValueError("golden numeric tolerance is invalid")
    hashes = manifest["files"]
    if not isinstance(hashes, dict) or set(hashes) != _HASHED_FILES:
        raise ValueError("golden fixture file hashes are invalid")
    if not all(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value) for value in hashes.values()):
        raise ValueError("golden fixture SHA-256 is invalid")
    if manifest["manifest_sha256"] != _manifest_sha256(manifest):
        raise ValueError("golden fixture manifest SHA-256 mismatch")


def _verify_hashes(bundle: Path, hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        if name == "fixture_manifest.json":
            if expected != _manifest_sha256(_read_json(bundle / name)):
                raise ValueError("golden fixture manifest SHA-256 mismatch")
            continue
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"golden fixture SHA-256 mismatch: {name}")


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical["files"] = {
        key: value
        for key, value in canonical["files"].items()
        if key != "fixture_manifest.json"
    }
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_models(manifest: Mapping[str, Any], frozen_model: Any, frozen_manifest: Mapping[str, Any], deployment_model: Any, deployment_manifest: Mapping[str, Any], frozen_bytes: bytes) -> None:
    if frozen_manifest.get("model_id") != manifest["model_id"] or deployment_manifest.get("model_id") != manifest["model_id"]:
        raise ValueError("golden model_id does not match frozen and deployment packages")
    if frozen_manifest.get("model_version") != manifest["model_version"] or deployment_manifest.get("model_version") != manifest["model_version"]:
        raise ValueError("golden model_version does not match frozen and deployment packages")
    if deployment_manifest["source_frozen_package"]["sha256"] != hashlib.sha256(frozen_bytes).hexdigest():
        raise ValueError("deployment package source frozen SHA-256 does not match")
    expected_deployment_schema = 1 if frozen_manifest["schema_version"] <= 4 else 2
    if deployment_manifest["deployment_schema_version"] != expected_deployment_schema:
        raise ValueError("golden frozen and deployment schema versions do not match")
    if list(frozen_manifest["config"]["tags"]) != manifest["input_tags"] or deployment_manifest["input_tags"] != manifest["input_tags"]:
        raise ValueError("golden input Tag order does not match packages")
    if list(frozen_model.feature_names) != manifest["dynamic_feature_names"] or list(deployment_model.feature_names) != manifest["dynamic_feature_names"]:
        raise ValueError("golden dynamic feature order does not match packages")
    for name in ("mean", "scale", "components", "eigenvalues", "explained_variance_ratio"):
        if not np.array_equal(getattr(frozen_model, name), getattr(deployment_model, name)):
            raise ValueError(f"frozen and deployment model array differs: {name}")
    if frozen_model.t2_limits != deployment_model.t2_limits or frozen_model.q_limits != deployment_model.q_limits:
        raise ValueError("frozen and deployment control limits differ")
    if frozen_manifest["status_rules"] != STATUS_RULES or deployment_manifest["status_rules"] != STATUS_RULES:
        raise ValueError("golden package status rules differ")
    if frozen_manifest["contribution_rules"] != CONTRIBUTION_RULES or deployment_manifest["contribution_rules"] != CONTRIBUTION_RULES:
        raise ValueError("golden package contribution rules differ")
    frozen_preprocessing = {
        name: frozen_manifest["config"][name]
        for name in deployment_manifest["preprocessing"]
        if name in frozen_manifest["config"]
    }
    deployment_preprocessing = {
        name: value
        for name, value in deployment_manifest["preprocessing"].items()
        if name in frozen_manifest["config"]
    }
    if frozen_preprocessing != deployment_preprocessing:
        raise ValueError("frozen and deployment preprocessing differs")
    if expected_deployment_schema == 2 and (
        deployment_manifest["preprocessing"].get("invalid_row_policy")
        != "drop_nonfinite_model_tags_and_state_filters_after_resampling"
        or deployment_manifest["preprocessing"].get("continuous_segment_policy")
        != "resegment_after_invalid_row_and_state_filter;first_order_initializes_at_segment_start"
    ):
        raise ValueError("golden deployment schema 2 preprocessing semantics are invalid")


def _read_raw_input(path: Path, manifest: Mapping[str, Any], deployment_manifest: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    state_columns = [item["column"] for item in deployment_manifest["preprocessing"]["state_filters"]]
    expected_columns = [manifest["timestamp_column"], *manifest["input_tags"], *state_columns]
    if list(frame.columns) != expected_columns:
        raise ValueError("golden raw input columns or order are invalid")
    timestamps = pd.to_datetime(frame.pop(manifest["timestamp_column"]), errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("golden raw input timestamps are invalid")
    return frame.set_axis(pd.DatetimeIndex(timestamps), axis=0)


def _deployment_scores(raw: pd.DataFrame, manifest: Mapping[str, Any], deployment_model: Any, deployment_manifest: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = preprocessing_config_from_mapping(deployment_manifest["preprocessing"])
    start, end = pd.Timestamp(manifest["replay_start"]), pd.Timestamp(manifest["replay_end"])
    filter_context = 0 if config.filter_method in {"none", "first_order"} else config.smoothing_window_minutes - config.sample_interval_minutes
    resampling_context = config.sample_interval_minutes if config.resampling_method != "none" else 0
    semantics = "legacy" if deployment_manifest["deployment_schema_version"] == 1 else "schema5"
    context_start = (
        raw.index[0]
        if semantics == "schema5" and config.filter_method == "first_order"
        else start - pd.Timedelta(minutes=config.max_lag_minutes + filter_context + resampling_context)
    )
    required = [*manifest["input_tags"], *(item.column for item in config.state_filters)]
    context = raw.loc[context_start:end, list(dict.fromkeys(required))]
    processed = preprocess_window(
        context,
        manifest["input_tags"],
        config,
        validate_quality=False,
        include_intermediates=True,
        allow_empty_state_filter=True,
        preprocessing_semantics=semantics,
    )
    dynamic = processed.dynamic.loc[
        processed.dynamic.index.to_series().between(start, end, inclusive="both"),
        list(deployment_model.feature_names),
    ]
    batch = deployment_model.score_dynamic_features(dynamic.to_numpy(dtype=float))
    scores = _batch_scores(dynamic.index, batch)
    return dynamic, scores


def _batch_scores(index: pd.DatetimeIndex, batch: Any) -> pd.DataFrame:
    result = pd.DataFrame(index=index)
    for number in range(batch.pc_scores.shape[1]):
        result[f"pc{number + 1}"] = batch.pc_scores[:, number]
    result["t2"] = batch.t2
    result["spe"] = batch.spe
    result["t2_limit_ratio"] = batch.t2_limit_ratio
    result["spe_limit_ratio"] = batch.spe_limit_ratio
    result["t2_status"] = batch.t2_status
    result["spe_status"] = batch.spe_status
    result["overall_status"] = batch.overall_status
    result["score_valid"] = batch.score_valid
    result["invalid_reason"] = batch.invalid_reason
    result["status"] = result["overall_status"]
    return result


def _read_dynamic(path: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    expected_columns = [manifest["timestamp_column"], *manifest["dynamic_feature_names"]]
    if list(frame.columns) != expected_columns:
        raise ValueError("golden dynamic feature columns or order are invalid")
    return _numeric_indexed_csv(frame, manifest["timestamp_column"], "golden dynamic features")


def _read_scores(path: Path, component_count: int) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    numeric = [*(f"pc{number + 1}" for number in range(component_count)), "t2", "spe", "t2_limit_ratio", "spe_limit_ratio"]
    discrete = ["t2_status", "spe_status", "overall_status", "score_valid", "invalid_reason", "status"]
    expected_columns = ["timestamp", *numeric, *discrete]
    if list(frame.columns) != expected_columns:
        raise ValueError("golden score columns or order are invalid")
    scores = _numeric_indexed_csv(frame, "timestamp", "golden scores", numeric)
    score_valid = scores["score_valid"]
    if not score_valid.isin(["True", "False", True, False]).all():
        raise ValueError("golden score_valid values are invalid")
    scores["score_valid"] = score_valid.isin(["True", True])
    scores["invalid_reason"] = scores["invalid_reason"].replace("", None)
    return scores


def _numeric_indexed_csv(frame: pd.DataFrame, timestamp_column: str, label: str, numeric_columns: list[str] | None = None) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame.pop(timestamp_column), errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps are invalid")
    result = frame.set_axis(pd.DatetimeIndex(timestamps), axis=0)
    columns = numeric_columns or list(result.columns)
    for column in columns:
        values = pd.to_numeric(result[column].replace("", np.nan), errors="coerce")
        if result[column].replace("", np.nan).notna().sum() != values.notna().sum():
            raise ValueError(f"{label} numeric values are invalid")
        result[column] = values
    return result


def _compare_dynamic(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    _require_same_index_columns(expected, actual, "dynamic features")
    _assert_close(expected.to_numpy(dtype=float), actual.to_numpy(dtype=float), "dynamic features")


def _compare_scores(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> None:
    _require_same_index_columns(expected, actual, f"{label} scores")
    numeric = [column for column in expected.columns if column.startswith("pc") or column in {"t2", "spe", "t2_limit_ratio", "spe_limit_ratio"}]
    for column in numeric:
        _assert_close(expected[column].to_numpy(dtype=float), actual[column].to_numpy(dtype=float), f"{label} {column}", equal_nan=True)
    for column in ("t2_status", "spe_status", "overall_status", "score_valid", "invalid_reason", "status"):
        if expected[column].tolist() != actual[column].tolist():
            raise ValueError(f"{label} {column} differs")


def _compare_deployment_scores(replay: pd.DataFrame, deployment: pd.DataFrame) -> None:
    expected = replay.loc[replay["score_valid"]].copy()
    _compare_scores(expected, deployment, "frozen and deployment")


def _require_same_index_columns(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> None:
    if not expected.index.equals(actual.index):
        raise ValueError(f"{label} timestamps differ")
    if list(expected.columns) != list(actual.columns):
        raise ValueError(f"{label} columns or order differ")


def _assert_close(expected: np.ndarray, actual: np.ndarray, label: str, *, equal_nan: bool = False) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL, equal_nan=equal_nan)
    except AssertionError as error:
        raise ValueError(f"{label} exceeds fixed numeric tolerance") from error


def _compare_summary(expected: Any, actual: Any) -> None:
    _compare_json_value(expected, actual, "golden summary")


def _compare_json_value(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"{label} fields differ")
        for key in expected:
            _compare_json_value(expected[key], actual[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{label} length differs")
        for position, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_json_value(left, right, f"{label}[{position}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ValueError(f"{label} type differs")
        _assert_close(np.array([float(expected)]), np.array([float(actual)]), label)
    elif expected != actual:
        raise ValueError(f"{label} differs")


def _compare_contributions(expected: Any, actual: Any, label: str) -> None:
    _compare_json_value(_contribution_core(expected), _contribution_core(actual), f"{label} contributions")


def _contribution_core(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("golden contribution records are invalid")
    result: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"timestamp", "statistic", "statistic_value", "limit_95", "tags"}:
            raise ValueError("golden contribution record fields are invalid")
        tags = record.get("tags")
        if not isinstance(tags, list):
            raise ValueError("golden contribution Tags are invalid")
        core_tags: list[dict[str, Any]] = []
        for tag in tags:
            if not isinstance(tag, dict) or not {"tag", "contribution_pct", "lag_start_minutes", "lag_end_minutes"} <= set(tag):
                raise ValueError("golden contribution Tag fields are invalid")
            core_tags.append({
                "tag": tag["tag"],
                "contribution_pct": tag["contribution_pct"],
                "lag_start_minutes": tag["lag_start_minutes"],
                "lag_end_minutes": tag["lag_end_minutes"],
            })
        result.append({
            "timestamp": record.get("timestamp"),
            "statistic": record.get("statistic"),
            "statistic_value": record.get("statistic_value"),
            "limit_95": record.get("limit_95"),
            "tags": core_tags,
        })
    return result


def _deployment_contributions(dynamic: pd.DataFrame, scores: pd.DataFrame, model: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for timestamp, row in scores.iterrows():
        score = SingleScoreResult(
            pc_scores=row[[column for column in row.index if column.startswith("pc")]].to_numpy(dtype=float),
            t2=float(row["t2"]), spe=float(row["spe"]),
            t2_limit_ratio=float(row["t2_limit_ratio"]), spe_limit_ratio=float(row["spe_limit_ratio"]),
            t2_status=str(row["t2_status"]), spe_status=str(row["spe_status"]),
            overall_status=str(row["overall_status"]), score_valid=True, invalid_reason=None,
        )
        values = dynamic.loc[timestamp].to_numpy(dtype=float)
        for statistic, limit in (("t2", model.t2_limits[0.95]), ("spe", model.q_limits[0.95])):
            tags = anomaly_tag_contributions(values, score, statistic, feature_names=model.feature_names, mean=model.mean, scale=model.scale, components=model.components, eigenvalues=model.eigenvalues, limit_95=limit)
            if tags:
                records.append({
                    "timestamp": pd.Timestamp(timestamp).isoformat(), "statistic": statistic,
                    "statistic_value": float(row[statistic]), "limit_95": float(limit),
                    "tags": [{"tag": item.tag, "contribution_pct": float(item.contribution_pct), "lag_start_minutes": item.lag_start_minutes, "lag_end_minutes": item.lag_end_minutes} for item in tags],
                })
    return records


def _max_abs(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(expected - actual))) if expected.size else 0.0


def _max_abs_column(expected: pd.DataFrame, actual: pd.DataFrame, column: str) -> float:
    left, right = expected[column].to_numpy(dtype=float), actual[column].to_numpy(dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    return _max_abs(left[finite], right[finite]) if finite.any() else 0.0


def _max_abs_scores(expected: pd.DataFrame, actual: pd.DataFrame, component_count: int) -> float:
    return max((_max_abs_column(expected, actual, f"pc{number + 1}") for number in range(component_count)), default=0.0)


def _max_contribution_error(expected: Any, actual: Any) -> float:
    left = [tag["contribution_pct"] for record in _contribution_core(expected) for tag in record["tags"]]
    right = [tag["contribution_pct"] for record in _contribution_core(actual) for tag in record["tags"]]
    return _max_abs(np.asarray(left, dtype=float), np.asarray(right, dtype=float)) if left else 0.0
