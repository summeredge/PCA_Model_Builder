from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from pca_model_builder.data_session import DataSessionCache


def _write_csv(path: Path, offset: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "alternate_time": pd.date_range("2026-02-01", periods=4, freq="10min"),
            "A": [1.0 + offset, 2.0, 3.0, 4.0],
            "B": [5.0, 6.0, 7.0, 8.0],
            "label": ["x", "x", "y", "y"],
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def test_metadata_cache_hits_and_keeps_timestamp_columns_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache()
    original = pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recorded)
    first, first_hit = cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    second, second_hit = cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    alternate, alternate_hit = cache.get_metadata(
        "dataset-1", path, "utf-8-sig", "alternate_time"
    )

    assert not first_hit
    assert second_hit
    assert not alternate_hit
    assert len(calls) == 2
    assert first.inferred_sample_interval == 5.0
    assert alternate.inferred_sample_interval == 10.0
    assert second.parsed_timestamp_series.equals(first.parsed_timestamp_series)


def test_column_cache_prunes_columns_normalizes_key_and_returns_safe_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.csv"
    expected = _write_csv(path)
    cache = DataSessionCache()
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    original = pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recorded)
    first = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A", "B"]
    )
    first.frame.loc[0, "A"] = 999.0
    second = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["B", "A"]
    )

    assert calls == [{"encoding": "utf-8-sig", "usecols": ["time", "A", "B"]}]
    assert first.loaded_column_count == 3
    assert second.column_cache_hit
    assert list(second.frame.columns) == ["time", "B", "A"]
    assert second.frame.loc[0, "A"] == expected.loc[0, "A"]
    direct = original(path, encoding="utf-8-sig", usecols=["time", "B", "A"])
    direct["time"] = pd.to_datetime(direct["time"])
    direct = direct.loc[:, ["time", "B", "A"]]
    pd.testing.assert_frame_equal(second.frame, direct)


def test_missing_column_is_explicit_and_file_change_invalidates_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache()
    first = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A"]
    )
    with pytest.raises(ValueError, match="找不到列：UNKNOWN"):
        cache.load_columns(
            "dataset-1", path, "utf-8-sig", "time", ["UNKNOWN"]
        )

    _write_csv(path, offset=100.0)
    path.write_bytes(path.read_bytes() + b"\n")
    changed = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A"]
    )

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    touched = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A"]
    )

    assert not first.cache_hit
    assert not changed.cache_hit
    assert changed.frame.loc[0, "A"] == 101.0
    assert not touched.cache_hit


def test_cache_limits_removal_and_clear_are_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_csv(first_path)
    _write_csv(second_path, offset=10.0)
    cache = DataSessionCache(max_datasets=1, max_column_entries=1)

    cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    cache.load_columns("first", first_path, "utf-8-sig", "time", ["B"])
    reloaded_a = cache.load_columns(
        "first", first_path, "utf-8-sig", "time", ["A"]
    )
    assert not reloaded_a.column_cache_hit

    cache.load_columns("second", second_path, "utf-8-sig", "time", ["A"])
    evicted = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not evicted.cache_hit

    cache.remove_dataset("first")
    removed = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not removed.cache_hit
    cache.clear()
    cleared = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not cleared.cache_hit
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.csv", "second.csv"]
