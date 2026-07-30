import pandas as pd
import pytest

from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.validation import (
    build_validation_matrix,
    ensure_disjoint_windows,
)


def test_validation_windows_must_not_overlap_training_windows():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-20"))]

    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint_windows(training, validation)


def test_separate_validation_window_is_allowed():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"))]

    ensure_disjoint_windows(training, validation)


def test_validation_uses_pre_window_context_and_scores_from_requested_start():
    index = pd.date_range("2026-01-01", periods=30, freq="5min")
    frame = pd.DataFrame({"A": range(30), "B": range(100, 130)}, index=index)
    config = PreprocessingConfig(5, 10, 10, 5)

    dynamic = build_validation_matrix(frame, ["A", "B"], config, index[10], index[20])

    assert dynamic.index[0] == index[10]
    assert dynamic.index[-1] == index[20]
    assert len(dynamic) == 11


def test_validation_rejects_missing_pre_window_context():
    index = pd.date_range("2026-01-01", periods=10, freq="5min")
    frame = pd.DataFrame({"A": range(10), "B": range(10)}, index=index)
    config = PreprocessingConfig(5, 10, 10, 5)

    with pytest.raises(ValueError, match="insufficient"):
        build_validation_matrix(frame, ["A", "B"], config, index[0], index[-1])
