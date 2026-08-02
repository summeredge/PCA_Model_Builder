from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

from .compat import (
    training_windows_from_payload,
)
from .dpca import fit_dpca
from .model_io import copy_validated_model_package, load_model_package, save_model_package
from .preprocessing import PreprocessingConfig
from .quality import QualityReport, inspect_data_quality
from .tag_config import engineering_ranges, normalize_tag_configs
from .training import build_training_matrix
from .validation import (
    record_engineer_decision,
    validate_model_windows,
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
    review.add_argument("--decision", choices=("passed", "insufficient", "failed"), required=True)
    review.add_argument("--comment", default="")
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--source-id")
    review.set_defaults(handler=_review_validation)

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
        "sample_interval_minutes": config.sample_interval_minutes,
        "smoothing_window_minutes": config.smoothing_window_minutes,
        "max_lag_minutes": config.max_lag_minutes,
        "lag_step_minutes": config.lag_step_minutes,
        "variance_threshold": args.variance_threshold,
        "tag_configs": tag_configs,
        "training_summary": training_result.window_summaries,
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
    model, manifest = load_model_package(args.model)
    if manifest["model_purpose"] != "normal_state":
        raise ValueError("探索模型不能执行独立验证")
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
    config = PreprocessingConfig(
        sample_interval_minutes=int(config_data["sample_interval_minutes"]),
        smoothing_window_minutes=int(config_data["smoothing_window_minutes"]),
        max_lag_minutes=int(config_data["max_lag_minutes"]),
        lag_step_minutes=int(config_data["lag_step_minutes"]),
    )
    for window in validation_windows:
        if not window["enabled"]:
            continue
        context = _select_window(
            raw,
            args.timestamp,
            validation_context_start(pd.Timestamp(window["start"]), config).isoformat(),
            window["end"],
        )
        _require_clean_data(
            context,
            args.timestamp,
            tags,
            expected_interval_minutes=config.sample_interval_minutes,
            configured_engineering_ranges=engineering_ranges(tag_configs),
        )
    indexed = _to_indexed_frame(raw, args.timestamp, tags)
    validation_result = validate_model_windows(
        model,
        indexed,
        tags,
        config,
        training_windows,
        validation_windows,
        tag_configs,
    )
    scores = validation_result["scores"]
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.scores_output, index_label=args.timestamp)

    contribution_records = validation_result["contributions"]
    _write_json(args.contributions_output, contribution_records)

    report: dict[str, Any] = {
        "model": str(args.model),
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "validation_windows": validation_result["validation_windows"],
        "validation_window_summaries": validation_result["window_summaries"],
        "normal_validation_complete": validation_result["normal_validation_complete"],
        "known_abnormal_complete": validation_result["known_abnormal_complete"],
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
    _write_json(args.report_output, report)
    return report


def _review_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.model.resolve() == args.output.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    _, manifest = load_model_package(args.model)
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    decision = record_engineer_decision(manifest, report, args.decision, args.comment)
    report["engineer_decision"] = decision
    _write_json(args.validation_report, report)
    if args.decision != "passed":
        return {"engineer_decision": decision, "validated_model": None}
    copy_validated_model_package(
        args.model,
        args.output,
        validation_summary=report,
        engineer_decision=decision,
        source_identifier=args.source_id or args.model.name,
    )
    return {"engineer_decision": decision, "validated_model": str(args.output)}


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
