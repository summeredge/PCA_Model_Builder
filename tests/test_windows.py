import pandas as pd
import pytest

from pca_model_builder.windows import (
    add_training_window,
    legacy_training_windows_to_canonical,
    merge_excluded_windows,
    normalize_training_windows,
    remove_training_window,
    set_enabled_training_window,
    subtract_excluded_windows,
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


def test_candidate_window_operations_allow_an_empty_collection():
    normalized = normalize_training_windows([_window()])[0]

    assert remove_training_window([_window()], "window-001") == []
    assert add_training_window([], _window()) == [normalized]
    assert summarize_training_windows([]) == []
    assert normalize_training_windows([], allow_empty=True) == []
    with pytest.raises(ValueError, match="窗口不存在"):
        update_training_window([], "missing", {"comment": "无效"})
    with pytest.raises(ValueError, match="窗口不存在"):
        set_enabled_training_window([], "missing", True)
    with pytest.raises(ValueError, match="非空列表"):
        normalize_training_windows([])


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


def test_excluded_windows_merge_touching_ranges_and_split_a_candidate():
    excluded = merge_excluded_windows(
        [
            {
                "id": "exclude-2",
                "start": "2026-01-01T16:00:00",
                "end": "2026-01-01T17:00:00",
                "source": "trend",
                "comment": "检修",
            },
            {
                "id": "exclude-1",
                "start": "2026-01-01T10:00:00",
                "end": "2026-01-01T10:30:00",
                "source": "trend",
                "comment": "波动",
            },
            {
                "id": "exclude-1b",
                "start": "2026-01-01T10:30:00",
                "end": "2026-01-01T11:00:00",
                "source": "trend",
                "comment": "波动持续",
            },
        ]
    )

    assert [(item["start"], item["end"]) for item in excluded] == [
        ("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        ("2026-01-01T16:00:00", "2026-01-01T17:00:00"),
    ]
    assert subtract_excluded_windows(
        {"start": "2026-01-01T08:00:00", "end": "2026-01-01T20:00:00"},
        excluded,
    ) == [
        {"start": "2026-01-01T08:00:00", "end": "2026-01-01T10:00:00"},
        {"start": "2026-01-01T11:00:00", "end": "2026-01-01T16:00:00"},
        {"start": "2026-01-01T17:00:00", "end": "2026-01-01T20:00:00"},
    ]


def test_excluded_windows_outside_a_candidate_do_not_change_it():
    candidate = {"start": "2026-01-01T08:00:00", "end": "2026-01-01T20:00:00"}
    excluded = [
        {
            "id": "exclude-outside",
            "start": "2026-01-01T21:00:00",
            "end": "2026-01-01T22:00:00",
            "source": "trend",
            "comment": "无关",
        }
    ]

    assert subtract_excluded_windows(candidate, excluded) == [candidate]
