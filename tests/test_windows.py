import pandas as pd
import pytest

from pca_model_builder.windows import (
    add_training_window,
    legacy_training_windows_to_canonical,
    normalize_training_windows,
    remove_training_window,
    set_enabled_training_window,
    summarize_training_windows,
    update_training_window,
)


def _window(window_id="window-001", start="2026-01-01T00:00:00", end="2026-01-01T00:10:00"):
    return {"id": window_id, "start": start, "end": end, "source": "manual", "source_ref": None, "enabled": True, "comment": "稳定"}


def test_training_window_operations_preserve_ids_comments_and_enablement():
    windows = add_training_window([_window()], _window("window-002", "2026-01-01T00:15:00", "2026-01-01T00:20:00"))
    windows = update_training_window(windows, "window-002", {"comment": "确认"})
    windows = set_enabled_training_window(windows, "window-002", False)

    assert windows[1]["comment"] == "确认"
    assert windows[1]["enabled"] is False
    assert remove_training_window(windows, "window-002") == [_window()]


def test_training_windows_reject_enabled_overlap_but_allow_adjacent_or_disabled():
    with pytest.raises(ValueError, match="不能重叠"):
        normalize_training_windows([_window(), _window("window-002", "2026-01-01T00:10:00", "2026-01-01T00:15:00")])

    adjacent = normalize_training_windows([_window(), _window("window-002", "2026-01-01T00:15:00", "2026-01-01T00:20:00")])
    disabled = normalize_training_windows([_window(), {**_window("window-002", "2026-01-01T00:05:00", "2026-01-01T00:15:00"), "enabled": False}])

    assert len(adjacent) == len(disabled) == 2


def test_legacy_windows_convert_and_summary_counts_raw_samples():
    windows = legacy_training_windows_to_canonical([["2026-01-01", "2026-01-01T00:10:00"]])
    summary = summarize_training_windows(
        windows,
        pd.Series(pd.date_range("2026-01-01", periods=4, freq="5min")),
        5,
    )

    assert windows[0]["id"] == "legacy-window-001"
    assert windows[0]["source"] == "legacy"
    assert summary[0]["raw_samples"] == summary[0]["effective_samples"] == 3
    assert summary[0]["expected_samples"] == 3
    assert summary[0]["quality_status"] == "ready"
