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
from .model_io import (
    commit_validation_artifacts,
    load_model_package,
    model_package_sha256,
    save_model_package,
    validate_validation_report_binding,
)
from .model_registry import (
    compare_model_versions,
    list_model_versions,
    publish_model_version,
    verify_model_package_integrity,
)
from .preprocessing import PreprocessingConfig
from .quality import QualityReport, inspect_data_quality
from .tag_config import engineering_ranges, normalize_tag_configs
from .training import build_training_matrix
from .validation import (
    normalize_and_validate_validation_evidence,
    record_engineer_decision,
    validation_artifact_metadata,
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
    review.add_argument("--scores", type=Path)
    review.add_argument("--contributions", type=Path)
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

    models = subparsers.add_parser("models", help="管理模型版本和发布包")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)

    models_list = model_subparsers.add_parser("list", help="列出模型版本")
    models_list.add_argument("--registry", type=Path, default=Path(".web_data/models"))
    models_list.set_defaults(handler=_models_list)

    models_compare = model_subparsers.add_parser("compare", help="比较两个模型版本")
    models_compare.add_argument("left", nargs="?", type=Path)
    models_compare.add_argument("right", nargs="?", type=Path)
    models_compare.add_argument("--left", dest="left_option", type=Path)
    models_compare.add_argument("--right", dest="right_option", type=Path)
    models_compare.add_argument("--left-model", dest="left_model", type=Path)
    models_compare.add_argument("--right-model", dest="right_model", type=Path)
    models_compare.set_defaults(handler=_models_compare)

    models_publish = model_subparsers.add_parser("publish", help="复制发布已验证模型")
    models_publish.add_argument("--model", type=Path, required=True)
    models_publish.add_argument("--registry", type=Path, default=Path(".web_data/models"))
    models_publish.add_argument("--confirm", "--confirm-publish", action="store_true")
    models_publish.add_argument("--applicability-scope", required=True)
    models_publish.add_argument("--engineer-comment", default="")
    models_publish.add_argument("--model-id")
    models_publish.add_argument("--parent-version")
    models_publish.set_defaults(handler=_models_publish)

    models_verify = model_subparsers.add_parser("verify", help="校验模型包完整性")
    models_verify.add_argument("model", nargs="?", type=Path)
    models_verify.add_argument("--model", dest="model_option", type=Path)
    models_verify.add_argument("--model-path", dest="model_path", type=Path)
    models_verify.add_argument("--require-external", action="store_true")
    models_verify.set_defaults(handler=_models_verify)
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


def _models_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    return list_model_versions(args.registry)


def _models_compare(args: argparse.Namespace) -> dict[str, Any]:
    left = args.left_model or args.left_option or args.left
    right = args.right_model or args.right_option or args.right
    if left is None or right is None:
        raise ValueError("compare需要两个模型包路径")
    return compare_model_versions(left, right)


def _parse_applicability_scope(value: str) -> object:
    text = value.strip()
    if not text:
        return ""
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("适用范围JSON无效") from error
    return text


def _models_publish(args: argparse.Namespace) -> dict[str, Any]:
    if args.parent_version is not None and args.model_id is None:
        raise ValueError("--parent-version只能与--model-id同时使用")
    return publish_model_version(
        args.model,
        args.registry,
        engineer_confirmation=args.confirm,
        applicability_scope=_parse_applicability_scope(args.applicability_scope),
        engineer_comment=args.engineer_comment,
        model_id=args.model_id,
        parent_version=args.parent_version,
    )


def _models_verify(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model_path or args.model_option or args.model
    if model is None:
        raise ValueError("verify需要模型包路径")
    return verify_model_package_integrity(
        model,
        require_external=args.require_external,
    )


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
        "validation_schema_version": 2,
        "model": str(args.model),
        "model_purpose": manifest["model_purpose"],
        "model_status": manifest["model_status"],
        "source_candidate_package": {
            "identifier": args.model.name,
            "filename": args.model.name,
            "sha256": model_package_sha256(args.model),
        },
        "validation_windows": validation_result["validation_windows"],
        "validation_window_summaries": validation_result["window_summaries"],
        "normal_validation_complete": validation_result["normal_validation_complete"],
        "known_abnormal_complete": validation_result["known_abnormal_complete"],
        "scored_rows": len(scores),
        "status_counts": dict(Counter(scores["status"])),
        "maximum_t2": float(scores["t2"].max()),
        "maximum_spe": float(scores["spe"].max()),
        "engineer_decision_required": True,
        "validation_artifacts": {
            "scores": validation_artifact_metadata(args.scores_output),
            "contributions": validation_artifact_metadata(args.contributions_output),
        },
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
    normalize_and_validate_validation_evidence(
        report,
        candidate_path=args.model,
        scores_path=args.scores_output,
        contributions_path=args.contributions_output,
        require_artifact_files=True,
        expected_identifier=args.model.name,
    )
    _write_json(args.report_output, report)
    return report


def _review_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.model.resolve() == args.output.resolve():
        raise ValueError("validated model output must differ from the candidate package")
    _, manifest = load_model_package(args.model)
    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    validate_validation_report_binding(args.model, manifest, report)
    binding_identifier = report["source_candidate_package"]["identifier"]
    source_identifier = args.source_id or binding_identifier
    if args.source_id:
        updated_binding = dict(report["source_candidate_package"])
        updated_binding["identifier"] = source_identifier
        report = dict(report)
        report["source_candidate_package"] = updated_binding
        validate_validation_report_binding(
            args.model, manifest, report, expected_identifier=source_identifier
        )
    if args.decision == "passed":
        normalize_and_validate_validation_evidence(
            report,
            candidate_path=args.model,
            scores_path=args.scores,
            contributions_path=args.contributions,
            require_artifact_files=True,
            expected_identifier=source_identifier,
        )
    decision = record_engineer_decision(manifest, report, args.decision, args.comment)
    updated_report = dict(report)
    updated_report["engineer_decision"] = decision
    commit_validation_artifacts(
        args.model,
        args.output,
        args.validation_report,
        report=updated_report,
        engineer_decision=decision,
        source_identifier=source_identifier,
        previous_report=report,
        scores_path=args.scores,
        contributions_path=args.contributions,
    )
    return {
        "engineer_decision": decision,
        "validated_model": str(args.output) if args.decision == "passed" else None,
    }


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
