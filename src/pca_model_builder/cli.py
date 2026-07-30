from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

from .contribution import exceedance_contribution_tables
from .dpca import fit_dpca
from .model_io import load_model_package, save_model_package
from .preprocessing import (
    PreprocessingConfig,
    build_dynamic_matrix,
    infer_segment_ids,
)
from .quality import QualityReport, inspect_data_quality
from .validation import (
    build_validation_matrix,
    ensure_disjoint_windows,
    validation_context_start,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pca-model-builder",
        description="Build and validate an offline dynamic PCA monitoring model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a draft DPCA model")
    _add_data_arguments(train)
    train.add_argument("--tags", nargs="+", required=True)
    train.add_argument("--normal-start", required=True)
    train.add_argument("--normal-end", required=True)
    train.add_argument("--sample-interval", type=int, default=5)
    train.add_argument("--smoothing-window", type=int, default=10)
    train.add_argument("--max-lag", type=int, default=60)
    train.add_argument("--lag-step", type=int, default=5)
    train.add_argument("--variance-threshold", type=float, default=0.95)
    train.add_argument("--components", type=int)
    train.add_argument("--model-name", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(handler=_train)

    validate = subparsers.add_parser(
        "validate", help="Replay an independent historical validation window"
    )
    _add_data_arguments(validate)
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--validation-start", required=True)
    validate.add_argument("--validation-end", required=True)
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
    normal = _select_window(
        raw, args.timestamp, args.normal_start, args.normal_end
    )
    _require_clean_data(
        normal,
        args.timestamp,
        args.tags,
        expected_interval_minutes=config.sample_interval_minutes,
    )
    indexed = _to_indexed_frame(normal, args.timestamp, args.tags)
    segments = infer_segment_ids(indexed.index, config.sample_interval_minutes)
    dynamic = build_dynamic_matrix(indexed, args.tags, config, segments)
    model = fit_dpca(
        dynamic,
        variance_threshold=args.variance_threshold,
        n_components=args.components,
    )

    training_window = [
        pd.Timestamp(args.normal_start).isoformat(),
        pd.Timestamp(args.normal_end).isoformat(),
    ]
    stored_config = {
        "model_name": args.model_name,
        "tags": list(args.tags),
        "timestamp_column": args.timestamp,
        "sample_interval_minutes": config.sample_interval_minutes,
        "smoothing_window_minutes": config.smoothing_window_minutes,
        "max_lag_minutes": config.max_lag_minutes,
        "lag_step_minutes": config.lag_step_minutes,
        "variance_threshold": args.variance_threshold,
    }
    save_model_package(
        args.output,
        model,
        config=stored_config,
        training_windows=[training_window],
    )
    return {
        "model": str(args.output),
        "validation_status": "draft",
        "training_rows": len(dynamic),
        "dynamic_features": dynamic.shape[1],
        "n_components": model.n_components,
        "cumulative_explained_variance": float(
            model.explained_variance_ratio[: model.n_components].sum()
        ),
    }


def _serve(args: argparse.Namespace) -> dict[str, Any]:
    from .web import run_server

    run_server(args.host, args.port, open_browser=not args.no_open)
    return {"status": "stopped"}


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    model, manifest = load_model_package(args.model)
    config_data = manifest["config"]
    tags = list(config_data["tags"])
    training_windows = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in manifest["training_windows"]
    ]
    validation_window = (
        pd.Timestamp(args.validation_start),
        pd.Timestamp(args.validation_end),
    )
    ensure_disjoint_windows(training_windows, [validation_window])

    raw = _read_csv(args.csv, args.timestamp, args.encoding)
    config = PreprocessingConfig(
        sample_interval_minutes=int(config_data["sample_interval_minutes"]),
        smoothing_window_minutes=int(config_data["smoothing_window_minutes"]),
        max_lag_minutes=int(config_data["max_lag_minutes"]),
        lag_step_minutes=int(config_data["lag_step_minutes"]),
    )
    context_start = validation_context_start(validation_window[0], config)
    context = _select_window(
        raw, args.timestamp, context_start.isoformat(), args.validation_end
    )
    _require_clean_data(
        context,
        args.timestamp,
        tags,
        expected_interval_minutes=config.sample_interval_minutes,
    )
    indexed = _to_indexed_frame(context, args.timestamp, tags)
    dynamic = build_validation_matrix(
        indexed,
        tags,
        config,
        validation_window[0],
        validation_window[1],
    )
    scores = model.score(dynamic)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.scores_output, index_label=args.timestamp)

    contribution_records: list[dict[str, Any]] = []
    for statistic, timestamp, value, limit95, table in exceedance_contribution_tables(
        model, dynamic, scores
    ):
        contribution_records.append(
            {
                "timestamp": timestamp.isoformat(),
                "statistic": statistic,
                "statistic_value": value,
                "limit_95": limit95,
                "tags": table.head(5).to_dict(orient="records"),
            }
        )
    _write_json(args.contributions_output, contribution_records)

    report: dict[str, Any] = {
        "model": str(args.model),
        "model_validation_status": manifest["validation_status"],
        "validation_window": [
            validation_window[0].isoformat(),
            validation_window[1].isoformat(),
        ],
        "scored_rows": len(scores),
        "status_counts": dict(Counter(scores["status"])),
        "maximum_t2": float(scores["t2"].max()),
        "maximum_spe": float(scores["spe"].max()),
        "engineer_decision_required": True,
    }
    if args.label_column:
        validation = _select_window(
            raw, args.timestamp, args.validation_start, args.validation_end
        )
        if args.label_column not in validation.columns:
            raise ValueError(f"missing label column: {args.label_column}")
        labels = validation.set_index(args.timestamp)[args.label_column].reindex(scores.index)
        report["status_by_engineering_label"] = {
            str(label): dict(Counter(scores.loc[labels == label, "status"]))
            for label in labels.dropna().unique()
        }
    _write_json(args.report_output, report)
    return report


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
) -> None:
    report = inspect_data_quality(
        frame,
        timestamp_column,
        tags,
        expected_interval_minutes=expected_interval_minutes,
    )
    if not report.can_train:
        raise ValueError(_format_quality_errors(report))


def _format_quality_errors(report: QualityReport) -> str:
    return "data quality review required: " + "; ".join(
        f"{issue.code}({issue.count})" for issue in report.issues
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


if __name__ == "__main__":
    raise SystemExit(main())
