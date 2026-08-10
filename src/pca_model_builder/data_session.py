from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Sequence

import pandas as pd


DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024
_CSV_ENCODINGS = frozenset({"utf-8-sig", "gb18030"})
TXT_ENCODING = "ascii"
TXT_SEPARATOR = "\t"
TXT_HEADER_ROW = 0
TXT_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def normalize_column_names(columns: Sequence[object]) -> list[str]:
    names = [str(column) for column in columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"列名字符串化后重复：{', '.join(duplicates)}")
    return names


@dataclass(frozen=True)
class FileFingerprint:
    size: int
    modified_time_ns: int


@dataclass(frozen=True)
class DataSessionMetadata:
    dataset_id: str
    source_path: Path
    file_size: int
    file_modified_time: int
    encoding: str
    timestamp_column: str
    column_names: tuple[str, ...]
    numeric_candidate_columns: tuple[str, ...]
    row_count: int
    time_start: pd.Timestamp | None
    time_end: pd.Timestamp | None
    inferred_sample_interval: float | None


@dataclass(frozen=True)
class DataLoadResult:
    frame: pd.DataFrame
    metadata: DataSessionMetadata
    cache_hit: bool
    column_cache_hit: bool
    loaded_column_count: int
    column_cache_match: str = "miss"


@dataclass(frozen=True)
class _CachedDatasetMetadata:
    public: DataSessionMetadata
    parsed_timestamp_series: pd.Series
    read_key: str


class DataSessionStageError(ValueError):
    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(str(error))
        self.stage = stage


_ColumnCacheKey = tuple[
    str, str, str, FileFingerprint, tuple[str, ...]
]


class DataSessionCache:
    """Bounded, process-local table metadata and whole-column cache."""

    def __init__(
        self,
        max_datasets: int = 3,
        max_column_entries: int = 16,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    ) -> None:
        if max_datasets < 1 or max_column_entries < 1 or max_cache_bytes < 1:
            raise ValueError("cache limits must be positive")
        self.max_datasets = max_datasets
        self.max_column_entries = max_column_entries
        self.max_cache_bytes = max_cache_bytes
        self._metadata: dict[tuple[str, str, str], _CachedDatasetMetadata] = {}
        self._metadata_bytes: dict[tuple[str, str, str], int] = {}
        self._columns: OrderedDict[_ColumnCacheKey, pd.DataFrame] = OrderedDict()
        self._column_bytes: dict[_ColumnCacheKey, int] = {}
        self._metadata_cache_bytes = 0
        self._column_cache_bytes = 0
        self._dataset_lru: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def metadata_cache_bytes(self) -> int:
        with self._lock:
            return self._metadata_cache_bytes

    @property
    def column_cache_bytes(self) -> int:
        with self._lock:
            return self._column_cache_bytes

    @property
    def cache_bytes(self) -> int:
        with self._lock:
            return self._metadata_cache_bytes + self._column_cache_bytes

    def clear(self) -> None:
        with self._lock:
            self._metadata.clear()
            self._metadata_bytes.clear()
            self._columns.clear()
            self._column_bytes.clear()
            self._metadata_cache_bytes = 0
            self._column_cache_bytes = 0
            self._dataset_lru.clear()

    def remove_dataset(self, dataset_id: str) -> None:
        with self._lock:
            for key in [key for key in self._metadata if key[0] == dataset_id]:
                self._remove_metadata(key)
            for key in [key for key in self._columns if key[0] == dataset_id]:
                self._remove_column(key)
            self._dataset_lru.pop(dataset_id, None)

    def get_metadata(
        self,
        dataset_id: str,
        source_path: str | Path,
        encoding: str,
        timestamp_column: str,
    ) -> tuple[DataSessionMetadata, bool]:
        path = Path(source_path)
        with self._lock:
            fingerprint = self._fingerprint(path)
            read_key = self._read_key(path, encoding)
            self._invalidate_changed_dataset(dataset_id, path, fingerprint)
            key = (dataset_id, read_key, timestamp_column)
            cached_metadata = self._metadata.get(key)
            if cached_metadata is not None:
                self._touch_dataset(dataset_id)
                return cached_metadata.public, True
            frame = self._read_table(path, encoding)
            cached_metadata = self._parse_metadata(
                dataset_id, path, fingerprint, encoding, timestamp_column, frame, read_key
            )
            self._store_metadata(cached_metadata)
            return cached_metadata.public, False

    def load_columns(
        self,
        dataset_id: str,
        source_path: str | Path,
        encoding: str,
        timestamp_column: str,
        requested_columns: Sequence[str] | None,
    ) -> DataLoadResult:
        path = Path(source_path)
        with self._lock:
            fingerprint = self._fingerprint(path)
            read_key = self._read_key(path, encoding)
            self._invalidate_changed_dataset(dataset_id, path, fingerprint)
            metadata_key = (dataset_id, read_key, timestamp_column)
            cached_metadata = self._metadata.get(metadata_key)
            metadata_hit = cached_metadata is not None
            metadata_cached = metadata_hit
            initial_frame: pd.DataFrame | None = None
            if cached_metadata is None:
                initial_frame = self._read_table(path, encoding)
                cached_metadata = self._parse_metadata(
                    dataset_id,
                    path,
                    fingerprint,
                    encoding,
                    timestamp_column,
                    initial_frame,
                    read_key,
                )
                metadata_cached = self._store_metadata(cached_metadata)

            metadata = cached_metadata.public
            requested = self._requested_order(metadata, requested_columns)
            normalized = tuple(sorted(requested))
            cache_key = (
                dataset_id,
                read_key,
                timestamp_column,
                fingerprint,
                normalized,
            )
            cached_key, cache_match = self._find_column_cache(cache_key)
            if cached_key is not None:
                cached = self._columns[cached_key]
                self._columns.move_to_end(cached_key)
                self._touch_dataset(dataset_id)
                return DataLoadResult(
                    cached.loc[:, requested].copy(deep=True),
                    metadata,
                    True,
                    True,
                    len(requested),
                    cache_match,
                )

            if initial_frame is not None:
                initial_frame[timestamp_column] = (
                    cached_metadata.parsed_timestamp_series.to_numpy(copy=False)
                )
                full_columns = tuple(sorted(metadata.column_names))
                full_key = (
                    dataset_id,
                    read_key,
                    timestamp_column,
                    fingerprint,
                    full_columns,
                )
                if metadata_cached:
                    self._store_column(full_key, initial_frame)
                frame = initial_frame.loc[:, requested].copy(deep=True)
            else:
                frame = self._read_table(path, encoding, usecols=requested)
                frame[timestamp_column] = (
                    cached_metadata.parsed_timestamp_series.to_numpy(copy=False)
                )
                if metadata_cached:
                    self._store_column(cache_key, frame)
                frame = frame.loc[:, requested].copy(deep=True)
            if metadata_cached:
                self._touch_dataset(dataset_id)
            return DataLoadResult(
                frame,
                metadata,
                metadata_hit,
                False,
                len(requested),
                "miss",
            )

    @staticmethod
    def _read_table(
        path: Path, encoding: str, usecols: Sequence[str] | None = None
    ) -> pd.DataFrame:
        try:
            if path.suffix.lower() == ".csv":
                kwargs: dict[str, object] = {"encoding": encoding}
                if usecols is not None:
                    kwargs["usecols"] = list(usecols)
                frame = pd.read_csv(path, **kwargs)
            if path.suffix.lower() == ".xlsx":
                kwargs = {"sheet_name": 0, "engine": "openpyxl"}
                frame = pd.read_excel(path, **kwargs)
                if usecols is not None:
                    frame.columns = normalize_column_names(frame.columns)
                    return frame.loc[:, list(usecols)]
            elif path.suffix.lower() == ".txt":
                if encoding != TXT_ENCODING:
                    raise ValueError("TXT 编码固定为 ASCII")
                kwargs = {
                    "encoding": TXT_ENCODING,
                    "sep": TXT_SEPARATOR,
                    "header": TXT_HEADER_ROW,
                }
                if usecols is not None:
                    kwargs["usecols"] = list(usecols)
                frame = pd.read_csv(path, **kwargs)
                if len(frame.columns) < 2:
                    raise ValueError("TXT 格式必须使用 Tab 分隔且首行为表头")
            elif path.suffix.lower() != ".csv":
                raise ValueError("仅支持 CSV、XLSX 或 TXT 文件")
            frame.columns = normalize_column_names(frame.columns)
            return frame
        except Exception as error:
            raise DataSessionStageError("loading", error) from error

    @staticmethod
    def _read_key(path: Path, encoding: str) -> str:
        if path.suffix.lower() == ".csv":
            if encoding not in _CSV_ENCODINGS:
                raise DataSessionStageError(
                    "loading", ValueError("CSV 编码仅支持 UTF-8-SIG 或 GB18030")
                )
            return f"csv:{encoding}"
        if path.suffix.lower() == ".xlsx":
            return "xlsx:sheet=0:header=0"
        if path.suffix.lower() == ".txt":
            if encoding != TXT_ENCODING:
                raise DataSessionStageError("loading", ValueError("TXT 编码固定为 ASCII"))
            return (
                "txt:encoding=ascii:separator=tab:header=0:"
                "timestamp_format=%Y/%m/%d %H:%M"
            )
        raise DataSessionStageError("loading", ValueError("仅支持 CSV、XLSX 或 TXT 文件"))

    @staticmethod
    def _fingerprint(path: Path) -> FileFingerprint:
        try:
            stat = path.stat()
        except OSError as error:
            message = ValueError("上传文件不存在，请重新上传")
            raise DataSessionStageError("loading", message) from error
        return FileFingerprint(stat.st_size, stat.st_mtime_ns)

    def _invalidate_changed_dataset(
        self, dataset_id: str, path: Path, fingerprint: FileFingerprint
    ) -> None:
        resolved = path.resolve()
        changed = any(
            metadata.public.source_path.resolve() != resolved
            or metadata.public.file_size != fingerprint.size
            or metadata.public.file_modified_time != fingerprint.modified_time_ns
            for key, metadata in self._metadata.items()
            if key[0] == dataset_id
        )
        if changed:
            self.remove_dataset(dataset_id)

    def _store_metadata(self, metadata: _CachedDatasetMetadata) -> bool:
        public = metadata.public
        key = (public.dataset_id, metadata.read_key, public.timestamp_column)
        size = int(
            metadata.parsed_timestamp_series.memory_usage(index=True, deep=True)
        )
        if size > self.max_cache_bytes:
            return False
        if key in self._metadata:
            self._remove_metadata(key)
        self._evict_for_bytes(size, public.dataset_id)
        if self._total_cache_bytes() + size > self.max_cache_bytes:
            return False
        self._metadata[key] = metadata
        self._metadata_bytes[key] = size
        self._metadata_cache_bytes += size
        self._touch_dataset(public.dataset_id)
        return True

    def _find_column_cache(
        self, requested_key: _ColumnCacheKey
    ) -> tuple[_ColumnCacheKey | None, str]:
        exact = self._columns.get(requested_key)
        if exact is not None:
            return requested_key, "exact"
        prefix = requested_key[:4]
        requested = set(requested_key[4])
        candidates = [
            (len(key[4]), position, key)
            for position, key in enumerate(self._columns)
            if key[:4] == prefix and requested.issubset(key[4])
        ]
        if not candidates:
            return None, "miss"
        minimum_columns = min(item[0] for item in candidates)
        _, _, selected = max(
            (item for item in candidates if item[0] == minimum_columns),
            key=lambda item: item[1],
        )
        return selected, "superset"

    def _store_column(self, key: _ColumnCacheKey, frame: pd.DataFrame) -> None:
        size = int(frame.memory_usage(index=True, deep=True).sum())
        if size > self.max_cache_bytes:
            return
        if key in self._columns:
            self._remove_column(key)
        while len(self._columns) >= self.max_column_entries:
            self._remove_column(next(iter(self._columns)))
        self._evict_for_bytes(size, key[0])
        if self._total_cache_bytes() + size > self.max_cache_bytes:
            return
        self._columns[key] = frame
        self._column_bytes[key] = size
        self._column_cache_bytes += size

    def _evict_for_bytes(self, incoming_size: int, current_dataset: str) -> None:
        while (
            self._columns
            and self._total_cache_bytes() + incoming_size > self.max_cache_bytes
        ):
            self._remove_column(next(iter(self._columns)))
        for dataset_id in list(self._dataset_lru):
            if self._total_cache_bytes() + incoming_size <= self.max_cache_bytes:
                break
            if dataset_id != current_dataset:
                self.remove_dataset(dataset_id)

    def _total_cache_bytes(self) -> int:
        return self._metadata_cache_bytes + self._column_cache_bytes

    def _remove_metadata(self, key: tuple[str, str, str]) -> None:
        self._metadata.pop(key, None)
        self._metadata_cache_bytes -= self._metadata_bytes.pop(key, 0)

    def _remove_column(self, key: _ColumnCacheKey) -> None:
        self._columns.pop(key, None)
        self._column_cache_bytes -= self._column_bytes.pop(key, 0)

    def _touch_dataset(self, dataset_id: str) -> None:
        self._dataset_lru.pop(dataset_id, None)
        self._dataset_lru[dataset_id] = None
        while len(self._dataset_lru) > self.max_datasets:
            oldest, _ = self._dataset_lru.popitem(last=False)
            self.remove_dataset(oldest)

    @staticmethod
    def _parse_metadata(
        dataset_id: str,
        path: Path,
        fingerprint: FileFingerprint,
        encoding: str,
        timestamp_column: str,
        frame: pd.DataFrame,
        read_key: str,
    ) -> _CachedDatasetMetadata:
        try:
            if timestamp_column not in frame.columns:
                raise ValueError(f"找不到时间列：{timestamp_column}")
            timestamp_kwargs: dict[str, object] = {"errors": "coerce"}
            if path.suffix.lower() == ".txt":
                timestamp_kwargs["format"] = TXT_TIMESTAMP_FORMAT
            parsed = pd.to_datetime(frame[timestamp_column], **timestamp_kwargs)
            if parsed.isna().any():
                raise ValueError("时间列包含无法解析的值")
            candidates = []
            for column in frame.columns:
                if column == timestamp_column:
                    continue
                if pd.api.types.is_datetime64_any_dtype(frame[column]):
                    continue
                if pd.api.types.is_bool_dtype(frame[column]):
                    continue
                original_non_null = int(frame[column].notna().sum())
                if original_non_null == 0:
                    continue
                numeric_count = int(
                    pd.to_numeric(frame[column], errors="coerce").notna().sum()
                )
                if numeric_count / original_non_null >= 0.8:
                    candidates.append(str(column))
            unique_sorted = pd.DatetimeIndex(parsed.dropna().unique()).sort_values()
            intervals = (
                pd.Series(unique_sorted).diff().dropna().dt.total_seconds() / 60.0
            )
            inferred = float(intervals.mode().iloc[0]) if not intervals.empty else None
            public = DataSessionMetadata(
                dataset_id=dataset_id,
                source_path=path,
                file_size=fingerprint.size,
                file_modified_time=fingerprint.modified_time_ns,
                encoding=encoding,
                timestamp_column=timestamp_column,
                column_names=tuple(str(column) for column in frame.columns),
                numeric_candidate_columns=tuple(candidates),
                row_count=len(frame),
                time_start=unique_sorted[0] if len(unique_sorted) else None,
                time_end=unique_sorted[-1] if len(unique_sorted) else None,
                inferred_sample_interval=inferred,
            )
            return _CachedDatasetMetadata(
                public, parsed.reset_index(drop=True), read_key
            )
        except Exception as error:
            if isinstance(error, DataSessionStageError):
                raise
            raise DataSessionStageError("parsing", error) from error

    @staticmethod
    def _requested_order(
        metadata: DataSessionMetadata, requested_columns: Sequence[str] | None
    ) -> list[str]:
        if requested_columns is None:
            requested = list(metadata.column_names)
        else:
            requested = [metadata.timestamp_column]
            requested.extend(
                str(column)
                for column in requested_columns
                if str(column) != metadata.timestamp_column
            )
        if len(requested) != len(set(requested)):
            raise ValueError("请求列不能重复")
        missing = [column for column in requested if column not in metadata.column_names]
        if missing:
            raise ValueError(f"找不到列：{', '.join(missing)}")
        return requested
