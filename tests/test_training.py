import numpy as np
import pandas as pd
import pytest

from pca_model_builder.dpca import fit_dpca
from pca_model_builder.preprocessing import PreprocessingConfig, build_dynamic_matrix
from pca_model_builder.training import build_training_matrix


def _windows(*ranges):
    return [
        {
            "id": f"window-{position:03d}",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": "manual",
            "source_ref": None,
            "enabled": enabled,
            "comment": "",
        }
        for position, (start, end, enabled) in enumerate(ranges, start=1)
    ]


def _frame(periods=80):
    time = pd.date_range("2026-01-01", periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "time": time,
            "A": np.arange(periods, dtype=float),
            "B": np.sin(np.arange(periods, dtype=float) / 3.0),
            "C": np.cos(np.arange(periods, dtype=float) / 5.0),
        }
    )


def _multistate_frame(second_value=20.0):
    rng = np.random.default_rng(73)
    time = pd.date_range("2026-01-01", periods=120, freq="5min")
    return pd.DataFrame(
        {
            "time": time,
            "A": [10.0] * 60 + [second_value] * 60,
            "B": rng.normal(size=120),
            "C": rng.normal(size=120),
        }
    )


def _two_windows(frame):
    return _windows(
        (frame.time.iloc[0], frame.time.iloc[59], True),
        (frame.time.iloc[60], frame.time.iloc[119], True),
    )


def test_training_builds_windows_independently_without_lag_or_smoothing_leakage():
    frame = _frame()
    frame.loc[:19, ["A", "B", "C"]] = 10_000.0
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows(
        (frame.time.iloc[0], frame.time.iloc[19], True),
        (frame.time.iloc[40], frame.time.iloc[79], True),
    )

    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    second_window = result.dynamic.loc[result.dynamic.index >= frame.time.iloc[40]]

    assert not second_window.empty
    assert second_window.to_numpy().max() < 200.0
    assert [summary["effective_samples"] for summary in result.window_summaries] == [17, 37]


def test_training_restarts_preprocessing_at_physical_time_gap_and_records_segments():
    frame = _frame(40)
    frame.loc[:19, ["A", "B", "C"]] = 10_000.0
    frame.loc[20:, "time"] += pd.Timedelta(minutes=20)
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows((frame.time.iloc[0], frame.time.iloc[-1], True))

    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    post_gap = result.dynamic.loc[result.dynamic.index >= frame.time.iloc[20]]
    summary = result.window_summaries[0]

    assert len(summary["segments"]) == 2
    assert all(segment["status"] == "used" for segment in summary["segments"])
    assert not post_gap.empty
    assert post_gap.to_numpy().max() < 200.0


def test_training_single_window_matches_existing_dynamic_matrix_and_disabled_windows_do_not_contribute():
    frame = _frame()
    frame.loc[40:, ["A", "B", "C"]] = 10_000.0
    config = PreprocessingConfig(5, 10, 10, 5)
    enabled = _windows((frame.time.iloc[0], frame.time.iloc[39], True))
    with_disabled = _windows(
        (frame.time.iloc[0], frame.time.iloc[39], True),
        (frame.time.iloc[40], frame.time.iloc[79], False),
    )

    expected = build_dynamic_matrix(
        frame.iloc[:40].set_index("time")[["A", "B", "C"]], ["A", "B", "C"], config
    )
    single = build_training_matrix(frame, "time", ["A", "B", "C"], config, enabled)
    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, with_disabled
    )

    pd.testing.assert_frame_equal(single.dynamic, expected)
    pd.testing.assert_frame_equal(result.dynamic, expected)
    assert result.window_summaries[1]["status"] == "disabled"
    np.testing.assert_allclose(result.dynamic.mean().to_numpy(), expected.mean().to_numpy())


def test_training_standardizes_from_all_enabled_dynamic_rows():
    frame = _frame(60)
    config = PreprocessingConfig(5, 10, 10, 5)
    windows = _windows(
        (frame.time.iloc[0], frame.time.iloc[29], True),
        (frame.time.iloc[30], frame.time.iloc[59], True),
    )
    result = build_training_matrix(frame, "time", ["A", "B", "C"], config, windows)
    expected = pd.concat(
        [
            build_dynamic_matrix(
                frame.iloc[:30].set_index("time")[["A", "B", "C"]],
                ["A", "B", "C"],
                config,
            ),
            build_dynamic_matrix(
                frame.iloc[30:].set_index("time")[["A", "B", "C"]],
                ["A", "B", "C"],
                config,
            ),
        ]
    )

    np.testing.assert_allclose(
        result.dynamic.mean().to_numpy(), expected.mean().to_numpy()
    )
    np.testing.assert_allclose(
        result.dynamic.std(ddof=0).to_numpy(),
        expected.std(ddof=0).to_numpy(),
    )


def test_training_records_short_dropped_segments_and_blocks_when_everything_is_invalid():
    frame = _frame(24)
    config = PreprocessingConfig(5, 10, 20, 5)
    valid_and_short = _windows(
        (frame.time.iloc[0], frame.time.iloc[19], True),
        (frame.time.iloc[20], frame.time.iloc[23], True),
    )
    only_short = _windows((frame.time.iloc[20], frame.time.iloc[23], True))

    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, valid_and_short
    )

    assert result.window_summaries[1]["status"] == "dropped"
    assert result.window_summaries[1]["dropped_reason"] == "insufficient_after_smoothing_and_lag"
    with pytest.raises(ValueError, match="所有启用窗口"):
        build_training_matrix(frame, "time", ["A", "B", "C"], config, only_short)


def test_training_checks_variability_after_merging_independent_windows():
    frame = _multistate_frame()
    config = PreprocessingConfig(5, 5, 0, 5)

    result = build_training_matrix(
        frame, "time", ["A", "B", "C"], config, _two_windows(frame)
    )
    model = fit_dpca(result.dynamic, n_components=2)

    assert result.dynamic["A__lag_000min"].std(ddof=0) > 0
    assert len(result.dynamic) == 120
    assert [summary["effective_samples"] for summary in result.window_summaries] == [60, 60]
    np.testing.assert_allclose(model.mean, result.dynamic.mean().to_numpy())
    np.testing.assert_allclose(model.scale, result.dynamic.std(ddof=0).to_numpy())


def test_training_rejects_features_that_remain_constant_after_merging():
    frame = _multistate_frame(second_value=10.0)

    with pytest.raises(ValueError, match="常量动态特征.*A__lag_000min"):
        build_training_matrix(
            frame,
            "time",
            ["A", "B", "C"],
            PreprocessingConfig(5, 5, 0, 5),
            _two_windows(frame),
        )


def test_training_reports_global_near_constant_feature_without_window_blocking():
    frame = _multistate_frame()
    frame.loc[:59, "A"] = 10.0 + np.arange(60) * 1e-8
    frame.loc[60:, "A"] = 10.0 + np.arange(60) * 1e-8

    result = build_training_matrix(
        frame,
        "time",
        ["A", "B", "C"],
        PreprocessingConfig(5, 5, 0, 5),
        _two_windows(frame),
    )

    assert any(
        warning["code"] == "near_constant_feature"
        and warning["feature"] == "A__lag_000min"
        for warning in result.global_quality_warnings
    )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda frame: frame.__setitem__("A", [1.0] * 60 + [None] + [2.0] * 59), "missing_value"),
        (lambda frame: frame.__setitem__("A", [1.0] * 60 + ["bad"] + [2.0] * 59), "non_numeric_value"),
        (lambda frame: frame.__setitem__("A", [1.0] * 60 + [float("inf")] + [2.0] * 59), "non_finite_value"),
        (lambda frame: frame.__setitem__("A", [1.0] * 60 + [1000.0] + [2.0] * 59), "engineering_range"),
        (lambda frame: frame.__setitem__("time", list(frame.time.iloc[:60]) + [frame.time.iloc[61]] + list(frame.time.iloc[61:])), "duplicate_timestamp"),
        (lambda frame: frame.__setitem__("time", list(frame.time.iloc[:60]) + [frame.time.iloc[60] + pd.Timedelta(minutes=2)] + list(frame.time.iloc[61:])), "irregular_sampling"),
    ],
)
def test_training_keeps_per_window_safety_checks(mutate, code):
    frame = _multistate_frame()
    mutate(frame)

    with pytest.raises(ValueError, match=code):
        build_training_matrix(
            frame,
            "time",
            ["A", "B", "C"],
            PreprocessingConfig(5, 5, 0, 5),
            _two_windows(frame),
            {"A": (-100.0, 100.0)},
        )
