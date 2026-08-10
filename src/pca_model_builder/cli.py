from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import pandas as pd

from .compat import (
    training_windows_from_payload,
)
from .dpca import fit_dpca
from .model_io import (
    copy_validated_model_package,
    export_deployment_package,
    freeze_validated_model_package,
    load_model_package,
    save_model_package,
)
from .golden import verify_golden_vectors
from .preprocessing import PreprocessingConfig, preprocessing_config_from_mapping
from .quality import QualityReport, inspect_data_quality
from .replay import replay_frozen_model
from .tag_config import engineering_ranges, normalize_tag_configs
from .training import build_training_matrix
from .validation import (
    build_validation_evidence,
    record_engineer_decision,
    validate_model_windows,
    verify_validation_evidence,
    validation_context_start,
    validation_windows_from_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pca-model-builder",
        description="Build and validate an offline dynamic PCA monitoring model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(
        subparsers,
        "train",
        "Train a normal-state candidate DPCA model (compatibility command)",
        "normal_state",
    )
    _add_train_parser(
        subparsers,
        "train-exploratory",
        "Train an exploratory draft DPCA model",
        "exploratory",
    )
    _add_train_parser(
        subparsers,
        "train-normal",
        "Train a normal-state candidate DPCA model",
        "normal_state",
    )

    validate = subparsers.add_parser(
        "validate", help="Replay an independent historical validation window"
    )
    _add_data_arguments(validate)
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--validation-start")
    validate.add_argument("--validation-end")
    validate.add_argument("--validation-windows", type=Path)
    validate.add_argument("--label-column")
    validate.add_argument(
        "--scores-output", type=Path, default=Path("validation_scores.csv")
    )
    validate.add_argument(
        "--report-output", type=Path, default=Path("validation_report.json")
    )
    validate.add_argument(
        "--contributions-output",
        type=Path,
        default=Path("validation_contributions.json"),
    )
    validate.set_defaults(handler=_validate)

    review = subparsers.add_parser(
        "review-validation", help="Record an engineer validation decision"
    )
    review.add_argument("--model", type=Path, required=True)
    review.add_argument("--validation-report", type=Path, required=True)
    review.add_argument("--scores", type=Path, required=True)
    review.add_argument("--contributions", type=Path, required=True)
    review.add_argument("--decision", choices=("passed", "insufficient", "failed"), required=True)
    review.add_argument("--comment", default="")
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--source-id")
    review.set_defaults(handler=_review_validation)

    freeze = subparsers.add_parser(
        "freeze-model", help="Freeze an approved normal-state validated model"
    )
    freeze.add_argument("--model", type=Path, required=True)
    freeze.add_argument("--model-id", required=True)
    freeze.add_argument("--model-version", type=int, required=True)
    freeze.add_argument("--frozen-by", required=True)
    freeze.add_argument("--comment", default="")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=_freeze_model)

    deployment = subparsers.add_parser(
        "export-deployment", help="Export a fixed deployment package from a frozen model"
    )
    deployment.add_argument("--model", type=Path, required=True)
    deployment.add_argument("--output", type=Path, required=True)
    deployment.set_defaults(handler=_export_deployment)

    replay = subparsers.add_parser(
        "replay-frozen", help="Replay historical data with an immutable frozen model"
    )
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--csv", type=Path, required=True)
    replay.add_argument("--timestamp", required=True)
    replay.add_argument("--replay-start", required=True)
    replay.add_argument("--replay-end", required=True)
    replay.add_argument("--scores-output", type=Path, required=True)
    replay.add_argument("--summary-output", type=Path, required=True)
    replay.add_argument("--contributions-output", type=Path, required=True)
    replay.set_defaults(handler=_replay_frozen)

    golden = subparsers.add_parser(
        "verify-golden", help="Verify the committed readonly golden acceptance vectors"
    )
    golden.add_argument("--bundle", type=Path, required=True)
    golden.set_defaults(handler=_verify_golden)

    serve = subparsers.add_parser("serve", help="Run the local web interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8775)
    serve.add_argument("--no-open", action="store_true")
    serve.set_defaults(handler=_serve)
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")


def _add_train_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    command: str,
    help_text: str,
    model_purpose: str,
) -> None:
    train = subparsers.add_parser(command, help=help_text)
    _add_data_arguments(train)
    train.add_argument("--tags", nargs="+", required=True)
    train.add_argument("--normal-start")
    train.add_argument("--normal-end")
    train.add_argument("--training-windows", type=Path)
    train.add_argument("--sample-interval", type=int, default=5)
    train.add_argument("--smoothing-window", type=int, default=10)
    train.add_argument("--max-lag", type=int, default=60)
    train.add_argument("--lag-step", type=int, default=5)
    train.add_argument(
        "--resampling-method", choices=("none", "mean", "median", "last"), default="none"
    )
    train.add_argument(
        "--filter-method",
        choices=("none", "trailing_mean", "trailing_median"),
        default="trailing_mean",
    )
    train.add_argument("--gap-threshold-minutes", type=float)
    train.add_argument("--variance-threshold", type=float, default=0.95)
    train.add_argument("--components", type=int)
    train.add_argument("--model-name", required=True)
    train.add_argument(
        "--tag-config",
        type=Path,
        help="Optional UTF-8 JSON object keyed by selected Tag name",
    )
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(handler=_train, model_purpose=model_purpose)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _train(args: argparse.Namespace) -> dict[str, Any]:
    raw = _read_csv(args.csv, args.timestamp, args.encoding)
    config = PreprocessingConfig(
        sample_interval_minutes=args.sample_interval,
        smoothing_window_minutes=args.smoothing_window,
        max_lag_minutes=args.max_lag,
        lag_step_minutes=args.lag_step,
        resampling_method=args.resampling_method,
        filter_method=args.filter_method,
        gap_threshold_minutes=args.gap_threshold_minutes,
    )
    tag_configs = _read_tag_configs(args.tag_config, args.tags)
    training_windows = _training_windows_from_args(args)
    training_result = build_training_matrix(
        raw,
        args.timestamp,
        args.tags,
        config,
        training_windows,
        engineering_ranges(tag_configs),
    )
    dynamic = training_result.dynamic
    model = fit_dpca(
        dynamic,
        variance_threshold=args.variance_threshold,
        n_components=args.components,
    )

    stored_config = {
        "model_name": args.model_name,
        "tags": list(args.tags),
        "timestamp_column": args.timestamp,
        **config.to_dict(),
        "variance_threshold": args.variance_threshold,
        "tag_configs": tag_configs,
        "training_summary": training_result.window_summaries,
        "preprocessing_summary": training_result.window_summaries,
        "training_window_totals": training_result.training_window_totals,
        "training_quality_warnings": training_result.global_quality_warnings,
    }
    save_model_package(
        args.output,
        model,
        config=stored_config,
        training_windows=training_windows,
        model_purpose=args.model_purpose,
        model_status="draft" if args.model_purpose == "exploratory" else "candidate",
    )
    return {
        "model": str(args.output),
        "model_purpose": args.model_purpose,
        "model_status": "draft"
        if args.model_purpose == "exploratory"
        else "candidate",
        "training_rows": len(dynamic),
        "training_window_summary": training_result.window_summaries,
        "training_quality_warnings": training_result.global_quality_warnings,
        "dynamic_features": dynamic.shape[1],
        "n_components": model.n_components,
        "cumulative_explained_variance": float(
            model.explained_variance_ratio[: model.n_components].sum()
        ),
    }


def _serve(args: argparse.Namespace) -> dict[str, Any]:
    from .web_dataproject import run_server

    run_server(args.host, args.port, open_browser=not args.no_open)
    return {"status": "stopped"}


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    path_entries = [
        ("输入CSV", args.csv),
        ("候选模型", args.model),
        ("评分输出", args.scores_output),
        ("贡献输出", args.contributions_output),
        ("报告输出", args.report_output),
    ]
    if args.validation_windows is not None:
        path_entries.append(("验证窗口", args.validation_windows))
    _require_distinct_paths(*path_entries)
    model, manifest = load_model_package(args.model)
    if manifest["model_purpose"] != "normal_state":
        raise ValueError("探索模型不能执行独立验证")
    if manifest["model_status"] not in {"candidate", "validated"}:
        raise ValueError("冻结模型不能执行独立验证")
    config_data = manifest["config"]
    tags = list(config_data["tags"])
    tag_configs = normalize_tag_configs(tags, config_data.get("tag_configs"))
    training_windows = [
        (pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
        for window in manifest["training_windows"]
        if window["enabled"]
    ]
    validation_windows = _validation_windows_from_args(args)

    raw = _read_csv(args.csv, args.timestamp, args.encoding)
    config = preprocessing_config_from_mapping(config_data)
    for window in validation_windows:
        if not window["enabled"]:
            continue
        context = _select_window(
            raw,
            args.timestamp,
            validation_context_start(pd.Timestamp(window["start"]), config).isoformat(),
            window["end"],
        )
        if config.resampling_method == "none" and manifest["schema_version"] <= 4:
            _require_clean_data(
                context,
                args.timestamp,
                tags,
                expected_interval_minutes=config.sample_interval_minutes,
                configured_engineering_ranges=engineering_ranges(tag_configs),
            )
    state_columns = [condition.column for condition in config.state_filters]
    indexed = _to_indexed_frame(raw, args.timestamp, [*tags, *state_columns])
    validation_result = validate_model_windows(
        model,
        indexed,
        tags,
        config,
        training_windows,
        validation_windows,
        tag_configs,
        preprocessing_semantics=("legacy" if manifest["schema_version"] <= 4 else "schema5"),
    )
    scores = validation_result["scores"]
    contribution_records = validation_result["contributions"]

    report: dict[str, Any] = {
        "model": str(args.model),
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "validation_windows": validation_result["validation_windows"],
        "validation_window_summaries": validation_result["window_summaries"],
        "normal_validation_complete": validation_result["normal_validation_complete"],
        "known_abnormal_complete": validation_result["known_abnormal_complete"],
        "validation_metrics": validation_result["validation_metrics"],
        "contribution_stability": validation_result["contribution_stability"],
        "scored_rows": len(scores),
        "status_counts": dict(Counter(scores["status"])),
        "maximum_t2": float(scores["t2"].max()),
        "maximum_spe": float(scores["spe"].max()),
        "engineer_decision_required": True,
    }
    if len(validation_result["validation_windows"]) == 1:
        window = validation_result["validation_windows"][0]
        report["validation_window"] = [window["start"], window["end"]]
    if args.label_column:
        if args.label_column not in raw.columns:
            raise ValueError(f"missing label column: {args.label_column}")
        labels = raw.set_index(args.timestamp)[args.label_column].reindex(scores.index)
        report["status_by_engineering_label"] = {
            str(label): dict(Counter(scores.loc[labels == label, "status"]))
            for label in labels.dropna().unique()
        }
    _commit_validation_artifacts(
        args.scores_output, args.contributions_output, args.report_output,
        scores, contribution_records, report, args.model, model, args.timestamp,
        config.sample_interval_minutes,
    )
    return report


def _review_validation(args: argparse.Namespace) -> dict[str, Any]:
    _require_distinct_paths(
        ("候选模型", args.model),
        ("验证报告", args.validation_report),
        ("评分工件", args.scores),
        ("贡献工件", args.contributions),
        ("已验证模型输出", args.output),
    )
    _, manifest = load_model_package(args.model)
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    config = preprocessing_config_from_mapping(manifest["config"])
    model, _ = load_model_package(args.model)
    verify_validation_evidence(
        args.model, model, report, args.scores, args.contributions,
        sample_interval_minutes=config.sample_interval_minutes,
    )
    decision = record_engineer_decision(manifest, report, args.decision, args.comment)
    report["engineer_decision"] = decision
    if args.decision != "passed":
        _write_json_atomic(args.validation_report, report)
        return {"engineer_decision": decision, "validated_model": None}
    temporary = _temporary_path(args.output)
    copy_validated_model_package(
        args.model, temporary,
        validation_summary=report,
        engineer_decision=decision,
        source_identifier=args.source_id or args.model.name,
    )
    _commit_paths((( _write_json_temp(args.validation_report, report), args.validation_report), (temporary, args.output)))
    return {"engineer_decision": decision, "validated_model": str(args.output)}


def _freeze_model(args: argparse.Namespace) -> dict[str, Any]:
    freeze_validated_model_package(
        args.model,
        args.output,
        model_id=args.model_id,
        model_version=args.model_version,
        frozen_by=args.frozen_by,
        comment=args.comment,
    )
    return {
        "frozen_model": str(args.output),
        "model_id": args.model_id,
        "model_version": args.model_version,
        "model_status": "frozen",
    }


def _export_deployment(args: argparse.Namespace) -> dict[str, Any]:
    export_deployment_package(args.model, args.output)
    return {"deployment_model": str(args.output), "model_status": "frozen"}


def _replay_frozen(args: argparse.Namespace) -> dict[str, Any]:
    _require_frozen_replay_model(args.model)
    raw = _read_csv(args.csv, args.timestamp, "utf-8-sig")
    indexed = _to_replay_indexed_frame(raw, args.timestamp)
    result = replay_frozen_model(
        args.model, indexed, args.replay_start, args.replay_end
    )
    _write_replay_outputs(
        args.scores_output,
        args.summary_output,
        args.contributions_output,
        result.scores,
        result.summary,
        result.contributions,
        args.timestamp,
    )
    return result.summary


def _verify_golden(args: argparse.Namespace) -> dict[str, Any]:
    return verify_golden_vectors(args.bundle)


def _require_frozen_replay_model(path: Path) -> None:
    if path.suffix.lower() == ".pcadeploy":
        raise ValueError("deployment packages cannot be replayed")
    _, manifest = load_model_package(path)
    if (
        manifest.get("schema_version") != 4
        or manifest.get("model_purpose") != "normal_state"
        or manifest.get("model_status") != "frozen"
    ):
        raise ValueError("only schema 4 normal_state/frozen models can be replayed")


def _to_replay_indexed_frame(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    timestamps = frame[timestamp_column]
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("CSV timestamps must be increasing and unique for frozen replay")
    return frame.set_index(timestamp_column)


def _write_replay_outputs(
    scores_path: Path,
    summary_path: Path,
    contributions_path: Path,
    scores: pd.DataFrame,
    summary: dict[str, Any],
    contributions: list[dict[str, Any]],
    timestamp_column: str,
) -> None:
    destinations = (scores_path, summary_path, contributions_path)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("frozen replay output paths must be distinct")
    if any(path.exists() for path in destinations):
        raise ValueError("frozen replay output already exists")
    temporary: list[Path] = []
    committed: list[Path] = []
    try:
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary.append(Path(handle.name))
        scores.to_csv(temporary[0], index_label=timestamp_column, encoding="utf-8-sig")
        temporary[1].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary[2].write_text(json.dumps(contributions, ensure_ascii=False, indent=2), encoding="utf-8")
        for source, destination in zip(temporary, destinations, strict=True):
            os.replace(source, destination)
            committed.append(destination)
    except Exception:
        for path in committed:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _read_csv(path: Path, timestamp_column: str, encoding: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding=encoding)
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if parsed.isna().any():
        raise ValueError("CSV contains invalid timestamps")
    frame = frame.copy()
    frame[timestamp_column] = parsed
    return frame


def _select_window(
    frame: pd.DataFrame, timestamp_column: str, start: str, end: str
) -> pd.DataFrame:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time > end_time:
        raise ValueError("time window start must not follow its end")
    selected = frame.loc[
        frame[timestamp_column].between(start_time, end_time, inclusive="both")
    ].copy()
    if selected.empty:
        raise ValueError("selected time window contains no data")
    return selected


def _require_clean_data(
    frame: pd.DataFrame,
    timestamp_column: str,
    tags: Sequence[str],
    expected_interval_minutes: float,
    configured_engineering_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    report = inspect_data_quality(
        frame,
        timestamp_column,
        tags,
        engineering_ranges=configured_engineering_ranges,
        expected_interval_minutes=expected_interval_minutes,
    )
    if not report.can_train:
        raise ValueError(_format_quality_errors(report))


def _format_quality_errors(report: QualityReport) -> str:
    return "data quality review required: " + "; ".join(
        f"{issue.code}({issue.count})"
        + (f" [{issue.tag}]" if issue.tag else "")
        + f": {issue.message}"
        for issue in report.issues
    )


def _to_indexed_frame(
    frame: pd.DataFrame, timestamp_column: str, tags: Sequence[str]
) -> pd.DataFrame:
    return (
        frame.loc[:, [timestamp_column, *tags]]
        .sort_values(timestamp_column)
        .set_index(timestamp_column)
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as handle:
        return Path(handle.name)


def _write_json_temp(destination: Path, value: Any) -> Path:
    temporary = _temporary_path(destination)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return temporary


def _write_json_atomic(destination: Path, value: Any) -> None:
    temporary = _write_json_temp(destination, value)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _commit_paths(entries: Sequence[tuple[Path, Path]]) -> None:
    if len({destination.resolve() for _, destination in entries}) != len(entries):
        raise ValueError("validation output paths must be distinct")
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for _, destination in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{next(tempfile._get_candidate_names())}.bak")
                os.replace(destination, backup)
                backups.append((backup, destination))
        for temporary, destination in entries:
            os.replace(temporary, destination)
            committed.append(destination)
    except Exception:
        for destination in committed:
            destination.unlink(missing_ok=True)
        for backup, destination in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup, _ in backups:
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in entries:
            temporary.unlink(missing_ok=True)


def _commit_validation_artifacts(
    scores_path: Path, contributions_path: Path, report_path: Path,
    scores: pd.DataFrame, contributions: list[dict[str, Any]], report: dict[str, Any],
    candidate_path: Path, model: Any, timestamp_column: str,
    sample_interval_minutes: int,
) -> None:
    if len({path.resolve() for path in (scores_path, contributions_path, report_path)}) != 3:
        raise ValueError("validation output paths must be distinct")
    scores_temp, contributions_temp = _temporary_path(scores_path), _temporary_path(contributions_path)
    try:
        scores.to_csv(scores_temp, index_label=timestamp_column)
        contributions_temp.write_text(json.dumps(contributions, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence = build_validation_evidence(candidate_path, model, scores_temp, contributions_temp, timestamp_column=timestamp_column, scores_row_count=len(scores))
        evidence["scores"]["filename"] = scores_path.name
        evidence["contributions"]["filename"] = contributions_path.name
        report["validation_evidence"] = evidence
        report["validation_evidence"] = verify_validation_evidence(
            candidate_path,
            model,
            report,
            scores_temp,
            contributions_temp,
            sample_interval_minutes=sample_interval_minutes,
            artifact_filenames=(scores_path.name, contributions_path.name),
            scores_frame=scores,
        )
        report_temp = _write_json_temp(report_path, report)
        _commit_paths(((scores_temp, scores_path), (contributions_temp, contributions_path), (report_temp, report_path)))
    finally:
        scores_temp.unlink(missing_ok=True)
        contributions_temp.unlink(missing_ok=True)


def _require_distinct_paths(*entries: tuple[str, Path]) -> None:
    resolved: dict[Path, str] = {}
    for label, path in entries:
        target = path.resolve()
        if target in resolved:
            if {label, resolved[target]} == {"已验证模型输出", "候选模型"}:
                raise ValueError("validated model output must differ from the candidate package")
            raise ValueError(f"路径冲突：{label}与{resolved[target]}不能使用同一文件")
        resolved[target] = label


def _read_tag_configs(
    path: Path | None, tags: Sequence[str]
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return normalize_tag_configs(tags, value)


def _training_windows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.training_windows is not None:
        if args.normal_start is not None or args.normal_end is not None:
            raise ValueError("--training-windows不能与--normal-start/--normal-end同时使用")
        value = json.loads(args.training_windows.read_text(encoding="utf-8-sig"))
        return training_windows_from_payload({"training_windows": value})
    return training_windows_from_payload(
        {"normal_start": args.normal_start, "normal_end": args.normal_end}
    )


def _validation_windows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.validation_windows is not None:
        if args.validation_start is not None or args.validation_end is not None:
            raise ValueError(
                "--validation-windows不能与--validation-start/--validation-end同时使用"
            )
        value = json.loads(args.validation_windows.read_text(encoding="utf-8-sig"))
        return validation_windows_from_payload({"validation_windows": value})
    if args.validation_start is None or args.validation_end is None:
        raise ValueError(
            "--validation-windows或--validation-start/--validation-end必须提供"
        )
    return validation_windows_from_payload(
        {
            "validation_start": args.validation_start,
            "validation_end": args.validation_end,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
