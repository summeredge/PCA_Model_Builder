import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from pca_model_builder.cli import main
from pca_model_builder import golden
from pca_model_builder.golden import _manifest_sha256, verify_golden_vectors


_BUNDLE = Path(__file__).parents[1] / "golden_vectors" / "v1"
_SCHEMA5_BUNDLE = Path(__file__).parents[1] / "golden_vectors" / "v2"


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "v1"
    shutil.copytree(_BUNDLE, destination)
    return destination


def _refresh_member_sha(bundle: Path, name: str) -> None:
    manifest_path = bundle / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    checksum = _manifest_sha256(manifest)
    manifest["files"]["fixture_manifest.json"] = checksum
    manifest["manifest_sha256"] = checksum
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_static_golden_bundle_passes_repeatedly_and_remains_unchanged(monkeypatch):
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _BUNDLE.iterdir()}
    actual_preprocess = golden.preprocess_window
    semantics: list[str] = []

    def spy_preprocess(*args, **kwargs):
        semantics.append(kwargs["preprocessing_semantics"])
        return actual_preprocess(*args, **kwargs)

    monkeypatch.setattr(golden, "preprocess_window", spy_preprocess)

    first = verify_golden_vectors(_BUNDLE)
    second = verify_golden_vectors(_BUNDLE)

    assert first == second
    assert first["acceptance_status"] == "passed"
    assert first["replay_row_count"] == 14
    assert first["score_valid_count"] == 10
    assert semantics == ["legacy", "legacy"]
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _BUNDLE.iterdir()} == before


def test_static_vectors_cover_the_required_preprocessing_and_status_cases():
    manifest = json.loads((_BUNDLE / "fixture_manifest.json").read_text(encoding="utf-8"))
    dynamic = pd.read_csv(_BUNDLE / "expected_dynamic_features.csv")
    scores = pd.read_csv(_BUNDLE / "expected_scores.csv")
    contributions = json.loads((_BUNDLE / "expected_contributions.json").read_text(encoding="utf-8"))

    assert manifest["input_tags"] == ["A", "B"]
    assert manifest["dynamic_feature_names"] == [
        "A__lag_000min", "B__lag_000min", "A__lag_005min", "B__lag_005min"
    ]
    assert dynamic.loc[0, "A__lag_000min"] == pytest.approx(0.65)
    assert dynamic.loc[0, "A__lag_005min"] == pytest.approx(0.55)
    assert {"normal", "attention", "abnormal"} <= set(scores["overall_status"])
    assert "2026-01-01 00:45:00" not in set(scores["timestamp"])
    assert {"missing_input", "time_gap_reset"} <= set(scores["invalid_reason"].dropna())
    assert {record["statistic"] for record in contributions} == {"t2", "spe"}


def test_schema5_deployment_schema2_golden_covers_first_order_and_invalid_row_reset(monkeypatch):
    actual_preprocess = golden.preprocess_window
    semantics: list[str] = []

    def spy_preprocess(*args, **kwargs):
        semantics.append(kwargs["preprocessing_semantics"])
        return actual_preprocess(*args, **kwargs)

    monkeypatch.setattr(golden, "preprocess_window", spy_preprocess)
    result = verify_golden_vectors(_SCHEMA5_BUNDLE)
    manifest = json.loads((_SCHEMA5_BUNDLE / "fixture_manifest.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(_SCHEMA5_BUNDLE / "expected_scores.csv")

    assert result["acceptance_status"] == "passed"
    assert semantics == ["schema5"]
    assert manifest["fixture_id"] == "schema5-first-order-v1"
    assert "2026-02-01 08:35:00" not in set(scores["timestamp"])
    assert result["dynamic_row_count"] == result["score_valid_count"]


@pytest.mark.parametrize(
    "name",
    [
        "raw_input.csv",
        "frozen_model.pcamodel",
        "deployment_model.pcadeploy",
        "expected_dynamic_features.csv",
        "expected_scores.csv",
        "expected_contributions.json",
        "expected_summary.json",
    ],
)
def test_golden_rejects_member_tampering_without_hash_update(tmp_path, name):
    bundle = _copy_bundle(tmp_path)
    with (bundle / name).open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_golden_vectors(bundle)


def test_golden_rejects_manifest_tampering(tmp_path):
    bundle = _copy_bundle(tmp_path)
    path = bundle / "fixture_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["fixture_id"] = "tampered"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        verify_golden_vectors(bundle)


def test_golden_rejects_dynamic_feature_reordering_after_hash_refresh(tmp_path):
    bundle = _copy_bundle(tmp_path)
    path = bundle / "expected_dynamic_features.csv"
    frame = pd.read_csv(path)
    frame = frame[["time", "B__lag_000min", "A__lag_000min", "A__lag_005min", "B__lag_005min"]]
    frame.to_csv(path, index=False)
    _refresh_member_sha(bundle, path.name)

    with pytest.raises(ValueError, match="dynamic feature columns or order"):
        verify_golden_vectors(bundle)


def test_golden_rejects_status_change_after_hash_refresh(tmp_path):
    bundle = _copy_bundle(tmp_path)
    path = bundle / "expected_scores.csv"
    frame = pd.read_csv(path)
    valid = frame["score_valid"]
    frame.loc[valid.idxmax(), "overall_status"] = "abnormal"
    frame.to_csv(path, index=False)
    _refresh_member_sha(bundle, path.name)

    with pytest.raises(ValueError, match="frozen replay overall_status differs"):
        verify_golden_vectors(bundle)


def test_golden_rejects_numeric_change_after_hash_refresh(tmp_path):
    bundle = _copy_bundle(tmp_path)
    path = bundle / "expected_scores.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["score_valid"].idxmax(), "t2"] += 1e-6
    frame.to_csv(path, index=False)
    _refresh_member_sha(bundle, path.name)

    with pytest.raises(ValueError, match="frozen replay t2 exceeds fixed numeric tolerance"):
        verify_golden_vectors(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [("tag", "changed"), ("statistic", "changed"), ("lag_end_minutes", 99)],
)
def test_golden_rejects_contribution_changes_after_hash_refresh(tmp_path, field, value):
    bundle = _copy_bundle(tmp_path)
    path = bundle / "expected_contributions.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    target = records[0] if field == "statistic" else records[0]["tags"][0]
    target[field] = value
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    _refresh_member_sha(bundle, path.name)

    with pytest.raises(ValueError, match="frozen replay contributions"):
        verify_golden_vectors(bundle)


def test_verify_golden_cli_success_and_failure_exit_codes(tmp_path, capsys):
    assert main(["verify-golden", "--bundle", str(_BUNDLE)]) == 0
    assert json.loads(capsys.readouterr().out)["acceptance_status"] == "passed"

    bundle = _copy_bundle(tmp_path)
    with (bundle / "raw_input.csv").open("ab") as handle:
        handle.write(b"tampered")
    assert main(["verify-golden", "--bundle", str(bundle)]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
