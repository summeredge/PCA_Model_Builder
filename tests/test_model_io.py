import zipfile

import numpy as np
import pandas as pd

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

