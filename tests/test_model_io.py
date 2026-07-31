from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.model_io import load_model_package, save_model_package


def test_model_package_round_trip_uses_json_and_npz(tmp_path):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        rng.normal(size=(100, 3)),
        columns=["A__lag_000min", "B__lag_000min", "C__lag_000min"],
    )
    model = fit_dpca(frame, n_components=2)
    path = tmp_path / "unit.pcamodel"

    save_model_package(
        path,
        model,
        config={"tags": ["A", "B", "C"]},
        training_windows=[["2026-01-01", "2026-01-02"]],
    )
    loaded, manifest = load_model_package(path)

    with zipfile.ZipFile(path) as package:
        assert set(package.namelist()) == {"manifest.json", "arrays.npz"}
    pd.testing.assert_frame_equal(model.score(frame), loaded.score(frame))
    assert manifest["validation_status"] == "draft"
    assert manifest["config"]["tags"] == ["A", "B", "C"]


def test_model_package_rejects_unexpected_files(tmp_path):
    frame = pd.DataFrame(
        np.random.default_rng(1).normal(size=(100, 3)),
        columns=["A", "B", "C"],
    )
    path = tmp_path / "unexpected.pcamodel"
    save_model_package(path, fit_dpca(frame, n_components=2), {}, [])
    with zipfile.ZipFile(path, "a") as package:
        package.writestr("unexpected.txt", "not allowed")

    with pytest.raises(ValueError, match="unexpected or missing files"):
        load_model_package(path)


@pytest.mark.parametrize(
    "array_name, corrupt, message",
    [
        ("scale", lambda values: np.zeros_like(values), "scale must be positive"),
        ("scale", lambda values: values.astype(str), "arrays must be numeric"),
        (
            "eigenvalues",
            lambda values: np.concatenate([values[:2], np.zeros_like(values[2:])]),
            "no effective residual space",
        ),
    ],
)
def test_model_package_rejects_invalid_numeric_arrays(
    tmp_path, array_name, corrupt, message
):
    frame = pd.DataFrame(
        np.random.default_rng(2).normal(size=(100, 3)),
        columns=["A", "B", "C"],
    )
    path = tmp_path / "invalid-scale.pcamodel"
    save_model_package(path, fit_dpca(frame, n_components=2), {}, [])

    with zipfile.ZipFile(path) as package:
        manifest = package.read("manifest.json")
        with np.load(BytesIO(package.read("arrays.npz"))) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
    arrays[array_name] = corrupt(arrays[array_name])
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("manifest.json", manifest)
        package.writestr("arrays.npz", buffer.getvalue())

    with pytest.raises(ValueError, match=message):
        load_model_package(path)
