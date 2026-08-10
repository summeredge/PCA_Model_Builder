from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from pca_model_builder.data_session import DataSessionCache, DataSessionStageError


TXT_FIXTURE = Path(__file__).parent / "fixtures" / "u400ph_desensitized.txt"


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


def _write_xlsx(path: Path, offset: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "alternate_time": pd.date_range("2026-02-01", periods=4, freq="10min"),
            "A": [1.0 + offset, 2.0, 3.0, 4.0],
            "B": [5.0, 6.0, 7.0, 8.0],
            "label": ["x", "x", "y", "y"],
        }
    )
    frame.to_excel(path, index=False)
    return frame


def _write_txt_fixture(path: Path) -> None:
    path.write_bytes(TXT_FIXTURE.read_bytes())


def test_u400ph_txt_uses_fixed_sample_contract_and_shared_data_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "U400PH.txt"
    _write_txt_fixture(path)
    cache = DataSessionCache()

    metadata, metadata_hit = cache.get_metadata("txt", path, "ascii", "TIME")
    full = cache.load_columns("txt", path, "ascii", "TIME", None)
    subset = cache.load_columns("txt", path, "ascii", "TIME", ["AIC450005.PV"])

    assert not metadata_hit
    assert metadata.encoding == "ascii"
    assert metadata.column_names == (
        "TIME",
        "AI450006.PV",
        "AIC450005.PV",
        "UY400015CAL.P01",
        "FICQ400001.PV",
        "AI400014.PV",
        "FIC420091.PV",
        "FIC440403.PV",
        "FIC442403.PV",
        "FIC413001.PV",
        "FIC420207.PV",
        "FIC421002.PV",
        "AI421004.PV",
        "UFIA502005.PV",
        "FIC400002.SV",
    )
    assert metadata.numeric_candidate_columns == metadata.column_names[1:]
    assert metadata.time_start == pd.Timestamp("2026-04-24 12:00")
    assert metadata.time_end == pd.Timestamp("2026-04-24 12:02")
    assert metadata.inferred_sample_interval == 1.0
    assert full.frame.loc[0, "TIME"] == pd.Timestamp("2026-04-24 12:00")
    assert full.frame.loc[1, "AI450006.PV"] == pytest.approx(0.651416361)
    assert full.frame.loc[2, "AIC450005.PV"] == pytest.approx(88.10217285)
    pd.testing.assert_frame_equal(
        subset.frame,
        full.frame.loc[:, ["TIME", "AIC450005.PV"]],
    )
    assert any(
        key[1]
        == "txt:encoding=ascii:separator=tab:header=0:timestamp_format=%Y/%m/%d %H:%M"
        for key in cache._metadata
    )


def test_txt_contract_rejects_other_delimiters_and_timestamp_formats(
    tmp_path: Path,
) -> None:
    delimiter_path = tmp_path / "comma.txt"
    timestamp_path = tmp_path / "timestamp.txt"
    delimiter_path.write_text("TIME,A,B\n2026/4/24 12:00,1,2\n", encoding="ascii")
    timestamp_path.write_text(
        "TIME\tA\tB\n2026-04-24 12:00\t1\t2\n", encoding="ascii"
    )
    cache = DataSessionCache()

    with pytest.raises(DataSessionStageError, match="Tab 分隔") as delimiter:
        cache.get_metadata("delimiter", delimiter_path, "ascii", "TIME")
    with pytest.raises(DataSessionStageError, match="时间列包含无法解析") as timestamp:
        cache.get_metadata("timestamp", timestamp_path, "ascii", "TIME")

    assert delimiter.value.stage == "loading"
    assert timestamp.value.stage == "parsing"


def test_txt_file_change_invalidates_cached_columns(tmp_path: Path) -> None:
    path = tmp_path / "U400PH.txt"
    _write_txt_fixture(path)
    cache = DataSessionCache()
    first = cache.load_columns("txt", path, "ascii", "TIME", ["AI450006.PV"])

    changed = path.read_text(encoding="ascii").replace("0.651876032", "1.651876032")
    path.write_text(changed, encoding="ascii", newline="")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    reloaded = cache.load_columns("txt", path, "ascii", "TIME", ["AI450006.PV"])

    assert not first.cache_hit
    assert not reloaded.cache_hit
    assert reloaded.frame.loc[0, "AI450006.PV"] == pytest.approx(1.651876032)


def test_csv_encodings_and_xlsx_share_metadata_and_column_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    xlsx_path = tmp_path / "history.xlsx"
    expected = _write_csv(csv_path)
    with pd.ExcelWriter(xlsx_path) as writer:
        expected.to_excel(writer, sheet_name="history", index=False)
        pd.DataFrame({"not_history": [1, 2, 3, 4]}).to_excel(
            writer, sheet_name="other", index=False
        )
    gb_path = tmp_path / "gb18030.csv"
    expected.to_csv(gb_path, index=False, encoding="gb18030")
    cache = DataSessionCache()

    csv_metadata, _ = cache.get_metadata("csv", csv_path, "utf-8-sig", "time")
    xlsx_metadata, _ = cache.get_metadata("xlsx", xlsx_path, "xlsx", "time")
    gb = cache.load_columns("gb", gb_path, "gb18030", "time", ["B", "A"])
    csv = cache.load_columns("csv", csv_path, "utf-8-sig", "time", ["B", "A"])
    xlsx = cache.load_columns("xlsx", xlsx_path, "xlsx", "time", ["B", "A"])

    assert csv_metadata.column_names == xlsx_metadata.column_names
    assert csv_metadata.numeric_candidate_columns == xlsx_metadata.numeric_candidate_columns
    assert csv_metadata.time_start == xlsx_metadata.time_start
    assert csv_metadata.time_end == xlsx_metadata.time_end
    assert csv_metadata.inferred_sample_interval == xlsx_metadata.inferred_sample_interval
    assert list(csv.frame.columns) == ["time", "B", "A"]
    pd.testing.assert_frame_equal(csv.frame, xlsx.frame, check_dtype=False)
    pd.testing.assert_frame_equal(csv.frame, gb.frame, check_dtype=False)


def test_csv_and_xlsx_stringify_headers_and_exclude_boolean_candidates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "numeric-header.csv"
    xlsx_path = tmp_path / "numeric-header.xlsx"
    source = pd.DataFrame(
        [
            ["2026-01-01 00:00", 1.0, 10.0, True],
            ["2026-01-01 00:05", 2.0, 20.0, False],
            ["2026-01-01 00:10", 3.0, 30.0, True],
        ],
        columns=[1001, 2002, 3003, "enabled"],
    )
    csv_source = source.copy()
    csv_source.columns = [str(column) for column in source.columns]
    csv_source.to_csv(csv_path, index=False, encoding="utf-8-sig")
    source.to_excel(xlsx_path, index=False)
    cache = DataSessionCache()

    csv_metadata, _ = cache.get_metadata("csv", csv_path, "utf-8-sig", "1001")
    xlsx_metadata, _ = cache.get_metadata("xlsx", xlsx_path, "xlsx", "1001")
    csv = cache.load_columns("csv", csv_path, "utf-8-sig", "1001", ["3003", "2002"])
    xlsx = cache.load_columns("xlsx", xlsx_path, "xlsx", "1001", ["3003", "2002"])

    assert csv_metadata.column_names == ("1001", "2002", "3003", "enabled")
    assert csv_metadata.column_names == xlsx_metadata.column_names
    assert csv_metadata.numeric_candidate_columns == ("2002", "3003")
    assert csv_metadata.numeric_candidate_columns == xlsx_metadata.numeric_candidate_columns
    assert list(csv.frame.columns) == ["1001", "3003", "2002"]
    pd.testing.assert_frame_equal(csv.frame, xlsx.frame, check_dtype=False)


def test_stringified_duplicate_xlsx_headers_and_csv_xlsx_encoding_are_explicit(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate.xlsx"
    csv_path = tmp_path / "history.csv"
    pd.DataFrame([["2026-01-01", 1]], columns=[1001, "1001"]).to_excel(
        duplicate_path, index=False
    )
    _write_csv(csv_path)
    cache = DataSessionCache()

    with pytest.raises(DataSessionStageError, match="列名字符串化后重复：1001") as duplicate:
        cache.get_metadata("duplicate", duplicate_path, "xlsx", "1001")
    with pytest.raises(DataSessionStageError, match="CSV 编码仅支持 UTF-8-SIG 或 GB18030") as invalid:
        cache.get_metadata("csv", csv_path, "xlsx", "time")

    assert duplicate.value.stage == "loading"
    assert invalid.value.stage == "loading"


def test_xlsx_timestamp_errors_are_parsing_failures(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.xlsx"
    invalid_path = tmp_path / "invalid.xlsx"
    pd.DataFrame({"A": [1.0, 2.0]}).to_excel(missing_path, index=False)
    pd.DataFrame({"time": ["invalid", "2026-01-01"], "A": [1.0, 2.0]}).to_excel(
        invalid_path, index=False
    )
    cache = DataSessionCache()

    with pytest.raises(DataSessionStageError, match="找不到时间列：time") as missing:
        cache.get_metadata("missing", missing_path, "xlsx", "time")
    with pytest.raises(DataSessionStageError, match="时间列包含无法解析的值") as invalid:
        cache.get_metadata("invalid", invalid_path, "xlsx", "time")

    assert missing.value.stage == "parsing"
    assert invalid.value.stage == "parsing"


def test_xlsx_file_change_invalidates_cached_columns(tmp_path: Path) -> None:
    path = tmp_path / "history.xlsx"
    _write_xlsx(path)
    cache = DataSessionCache()
    first = cache.load_columns("dataset-1", path, "xlsx", "time", ["A"])
    _write_xlsx(path, offset=100.0)
    changed = cache.load_columns("dataset-1", path, "xlsx", "time", ["A"])

    assert not first.cache_hit
    assert not changed.cache_hit
    assert changed.frame.loc[0, "A"] == 101.0


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
    assert first is second
    assert not hasattr(first, "parsed_timestamp_series")
    assert cache.metadata_cache_bytes > 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == cache.metadata_cache_bytes
    with pytest.raises(FrozenInstanceError):
        first.row_count = 0  # type: ignore[misc]


def test_oversized_metadata_is_temporary_and_reparsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache(max_cache_bytes=1)
    original = pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recorded)
    first, first_hit = cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    second, second_hit = cache.get_metadata("dataset-1", path, "utf-8-sig", "time")

    assert first == second
    assert not first_hit
    assert not second_hit
    assert len(calls) == 2
    assert cache.metadata_cache_bytes == 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == 0
    assert not cache._metadata
    assert not cache._metadata_bytes
    assert not cache._dataset_lru


def test_loading_and_timestamp_parsing_failures_have_distinct_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache()
    original = pd.read_csv

    def fail_read(*args, **kwargs):
        raise OSError("read failed")

    monkeypatch.setattr(pd, "read_csv", fail_read)
    with pytest.raises(DataSessionStageError) as loading:
        cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    assert loading.value.stage == "loading"
    assert str(loading.value) == "read failed"

    monkeypatch.setattr(pd, "read_csv", original)
    path.write_text("time,A\ninvalid,1\n", encoding="utf-8-sig")
    with pytest.raises(DataSessionStageError) as parsing:
        cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    assert parsing.value.stage == "parsing"
    assert str(parsing.value) == "时间列包含无法解析的值"


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
    assert first.cache_hit
    assert not first.column_cache_hit
    assert first.column_cache_match == "miss"
    assert first.loaded_column_count == 3
    assert second.column_cache_hit
    assert second.column_cache_match == "exact"
    assert list(second.frame.columns) == ["time", "B", "A"]
    assert second.frame.loc[0, "A"] == expected.loc[0, "A"]
    direct = original(path, encoding="utf-8-sig", usecols=["time", "B", "A"])
    direct["time"] = pd.to_datetime(direct["time"])
    direct = direct.loc[:, ["time", "B", "A"]]
    pd.testing.assert_frame_equal(second.frame, direct)


def test_full_and_superset_caches_serve_subsets_without_csv_reads(
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
    full = cache.load_columns("dataset-1", path, "utf-8-sig", "time", None)
    subset = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["B", "A"]
    )
    subset.frame.loc[0, "A"] = 999.0
    repeated = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A", "B"]
    )

    assert len(calls) == 1
    assert not full.cache_hit
    assert not full.column_cache_hit
    assert subset.cache_hit and subset.column_cache_hit
    assert subset.column_cache_match == "superset"
    assert list(subset.frame.columns) == ["time", "B", "A"]
    assert repeated.frame.loc[0, "A"] == 1.0


def test_smallest_and_most_recent_superset_are_selected(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache()
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A", "B"])
    cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A", "B", "label"]
    )
    smallest_key = next(key for key in cache._columns if len(key[4]) == 3)

    result = cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])

    assert result.column_cache_match == "superset"
    assert next(reversed(cache._columns)) == smallest_key

    cache.clear()
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A", "B"])
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A", "label"])
    recent_key = next(reversed(cache._columns))
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])

    assert next(reversed(cache._columns)) == recent_key


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
    assert cache.cache_bytes == (
        cache.metadata_cache_bytes + cache.column_cache_bytes
    )
    assert cache.cache_bytes <= cache.max_cache_bytes

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    touched = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A"]
    )

    assert not first.cache_hit
    assert not changed.cache_hit
    assert changed.frame.loc[0, "A"] == 101.0
    assert not touched.cache_hit
    assert cache.cache_bytes == (
        cache.metadata_cache_bytes + cache.column_cache_bytes
    )
    assert cache.cache_bytes >= 0


def test_dataset_path_change_releases_old_cache(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_csv(first_path)
    _write_csv(second_path, offset=100.0)
    cache = DataSessionCache()
    cache.load_columns("dataset-1", first_path, "utf-8-sig", "time", ["A"])

    changed = cache.load_columns(
        "dataset-1", second_path, "utf-8-sig", "time", ["A"]
    )

    assert not changed.cache_hit
    assert changed.frame.loc[0, "A"] == 101.0
    assert {item.public.source_path for item in cache._metadata.values()} == {
        second_path
    }
    assert {key[0] for key in cache._metadata_bytes} == {"dataset-1"}
    assert {key[0] for key in cache._column_bytes} == {"dataset-1"}
    assert cache.cache_bytes == (
        cache.metadata_cache_bytes + cache.column_cache_bytes
    )


def test_cache_limits_removal_and_clear_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_csv(first_path)
    _write_csv(second_path, offset=10.0)
    cache = DataSessionCache(max_datasets=1, max_column_entries=1)
    original = pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recorded)

    cache.get_metadata("first", first_path, "utf-8-sig", "time")
    cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    cache.load_columns("first", first_path, "utf-8-sig", "time", ["B"])
    reloaded_a = cache.load_columns(
        "first", first_path, "utf-8-sig", "time", ["A"]
    )
    assert not reloaded_a.column_cache_hit

    cache.load_columns("second", second_path, "utf-8-sig", "time", ["A"])
    assert {key[0] for key in cache._metadata} == {"second"}
    assert {key[0] for key in cache._metadata_bytes} == {"second"}
    assert {key[0] for key in cache._columns} == {"second"}
    assert {key[0] for key in cache._column_bytes} == {"second"}
    assert list(cache._dataset_lru) == ["second"]
    reads_before_reload = len(calls)
    evicted = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not evicted.cache_hit
    assert len(calls) == reads_before_reload + 1

    cache.remove_dataset("first")
    assert cache.metadata_cache_bytes == 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == 0
    assert not cache._metadata_bytes
    assert not cache._column_bytes
    assert not cache._dataset_lru
    removed = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not removed.cache_hit
    cache.clear()
    assert cache.metadata_cache_bytes == 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == 0
    cleared = cache.load_columns("first", first_path, "utf-8-sig", "time", ["A"])
    assert not cleared.cache_hit
    assert sorted(path.name for path in tmp_path.iterdir()) == ["first.csv", "second.csv"]


def test_cache_byte_limit_evicts_lru_and_tracks_clear_and_remove(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    probe = DataSessionCache()
    probe.get_metadata("dataset-1", path, "utf-8-sig", "time")
    probe.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])
    metadata_bytes = probe.metadata_cache_bytes
    one_column_bytes = probe.column_cache_bytes
    one_entry_bytes = metadata_bytes + one_column_bytes

    cache = DataSessionCache(max_column_entries=10, max_cache_bytes=one_entry_bytes)
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])
    assert cache.cache_bytes == one_entry_bytes
    assert cache.metadata_cache_bytes == metadata_bytes
    assert cache.column_cache_bytes == one_column_bytes
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["B"])
    assert cache.cache_bytes == one_entry_bytes
    assert cache.cache_bytes == (
        cache.metadata_cache_bytes + cache.column_cache_bytes
    )
    assert len(cache._columns) == 1
    assert next(iter(cache._columns))[4] == ("B", "time")
    reloaded = cache.load_columns(
        "dataset-1", path, "utf-8-sig", "time", ["A"]
    )
    assert not reloaded.column_cache_hit
    assert cache.cache_bytes <= one_entry_bytes

    cache.remove_dataset("dataset-1")
    assert cache.metadata_cache_bytes == 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == 0
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])
    assert cache.cache_bytes > 0
    cache.clear()
    assert cache.metadata_cache_bytes == 0
    assert cache.column_cache_bytes == 0
    assert cache.cache_bytes == 0


def test_oversized_entry_is_returned_but_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.csv"
    _write_csv(path)
    cache = DataSessionCache(max_cache_bytes=1)
    cache.get_metadata("dataset-1", path, "utf-8-sig", "time")
    original = pd.read_csv
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recorded)
    first = cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])
    second = cache.load_columns("dataset-1", path, "utf-8-sig", "time", ["A"])

    assert first.frame.equals(second.frame)
    assert not first.column_cache_hit
    assert not second.column_cache_hit
    assert len(calls) == 2
    assert cache.cache_bytes == 0
