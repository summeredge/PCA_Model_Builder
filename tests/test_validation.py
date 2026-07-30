import pandas as pd
import pytest

from pca_model_builder.validation import ensure_disjoint_windows


def test_validation_windows_must_not_overlap_training_windows():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-20"))]

    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint_windows(training, validation)


def test_separate_validation_window_is_allowed():
    training = [(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))]
    validation = [(pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-10"))]

    ensure_disjoint_windows(training, validation)
