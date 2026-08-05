import numpy as np
import pandas as pd

from pca_model_builder.preprocessing import PreprocessingConfig
from pca_model_builder.state_exploration import ExplorationConfig, run_state_exploration


def test_state_exploration_is_draft_and_deterministic_with_display_limit():
    index = pd.date_range("2026-01-01", periods=80, freq="5min")
    values = np.r_[np.linspace(-3, -2, 40), np.linspace(2, 3, 40)]
    frame = pd.DataFrame({"A": values, "B": values**2, "C": np.sin(values)}, index=index)
    config = ExplorationConfig(cluster_count=2, minimum_candidate_duration_minutes=10, maximum_plot_points=12)
    first = run_state_exploration(frame, ["A", "B", "C"], PreprocessingConfig(5, 0, 0, 5, filter_method="none"), config)
    second = run_state_exploration(frame, ["A", "B", "C"], PreprocessingConfig(5, 0, 0, 5, filter_method="none"), config)

    assert first["exploratory_model_summary"]["model_purpose"] == "exploratory"
    assert first["exploratory_model_summary"]["model_status"] == "draft"
    assert first["exploratory_model_summary"]["n_components"] >= 2
    pd.testing.assert_frame_equal(first["cluster_series"], second["cluster_series"])
    assert len(first["cluster_series_display"]) <= 12
    assert first["cluster_series"].index.is_unique
    assert sum(item["sample_count"] for item in first["cluster_summaries"]) == len(first["cluster_series"])
    assert all(item["source"] == "cluster" and item["comment"] == "" for item in first["cluster_candidates"])
