from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Sequence

import pandas as pd


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
    parsed_timestamp_series: pd.Series
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


class DataSessionCache:
    """Bounded, process-local CSV metadata and whole-column cache."""

    def __init__(self, max_datasets: int = 3, max_column_entries: int = 16) -> None:
        if max_datasets < 1 or max_column_entries < 1:
            raise ValueError("cache limits must be positive")
        self.max_datasets = max_datasets
        self.max_column_entries = max_column_entries
        self._metadata: dict[tuple[str, str, str], DataSessionMetadata] = {}
        self._columns: OrderedDict[
            tuple[str, str, str, FileFingerprint, tuple[str, ...]], pd.DataFrame
        ] = OrderedDict()
        self._dataset_lru: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._metadata.clear()
            self._columns.clear()
            self._dataset_lru.clear()

    def remove_dataset(self, dataset_id: str) -> None:
        with self._lock:
            self._metadata = {
                key: value
                for key, value in self._metadata.items()
                if key[0] != dataset_id
            }
            for key in [key for key in self._columns if key[0] == dataset_id]:
                del self._columns[key]
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
            self._invalidate_changed_dataset(dataset_id, path, fingerprint)
            key = (dataset_id, encoding, timestamp_column)
            metadata = self._metadata.get(key)
            if metadata is not None:
                self._touch_dataset(dataset_id)
                return self._copy_metadata(metadata), True
            frame = pd.read_csv(path, encoding=encoding)
            metadata = self._build_metadata(
                dataset_id, path, fingerprint, encoding, timestamp_column, frame
            )
            self._store_metadata(metadata)
            return self._copy_metadata(metadata), False

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
            self._invalidate_changed_dataset(dataset_id, path, fingerprint)
            metadata_key = (dataset_id, encoding, timestamp_column)
            metadata = self._metadata.get(metadata_key)
            metadata_hit = metadata is not None
            initial_frame: pd.DataFrame | None = None
            if metadata is None:
                initial_frame = pd.read_csv(path, encoding=encoding)
                metadata = self._build_metadata(
                    dataset_id,
                    path,
                    fingerprint,
                    encoding,
                    timestamp_column,
                    initial_frame,
                )
                self._store_metadata(metadata)

            requested = self._requested_order(metadata, requested_columns)
            normalized = tuple(sorted(requested))
            cache_key = (
                dataset_id,
                encoding,
                timestamp_column,
                fingerprint,
                normalized,
            )
            cached = self._columns.get(cache_key)
            if cached is not None:
                self._columns.move_to_end(cache_key)
                self._touch_dataset(dataset_id)
                frame = cached.loc[:, requested].copy(deep=True)
                return DataLoadResult(
                    frame,
                    self._copy_metadata(metadata),
                    True,
                    True,
                    len(requested),
                )

            if initial_frame is None:
                frame = pd.read_csv(path, encoding=encoding, usecols=requested)
            else:
                frame = initial_frame.loc[:, requested].copy(deep=True)
            frame[timestamp_column] = metadata.parsed_timestamp_series.to_numpy(copy=True)
            stored = frame.loc[:, normalized].copy(deep=True)
            self._columns[cache_key] = stored
            self._columns.move_to_end(cache_key)
            while len(self._columns) > self.max_column_entries:
                self._columns.popitem(last=False)
            self._touch_dataset(dataset_id)
            return DataLoadResult(
                frame.loc[:, requested].copy(deep=True),
                self._copy_metadata(metadata),
                metadata_hit,
                False,
                len(requested),
            )

    @staticmethod
    def _fingerprint(path: Path) -> FileFingerprint:
        try:
            stat = path.stat()
        except FileNotFoundError as error:
            raise ValueError("上传文件不存在，请重新上传") from error
        return FileFingerprint(stat.st_size, stat.st_mtime_ns)

    def _invalidate_changed_dataset(
        self, dataset_id: str, path: Path, fingerprint: FileFingerprint
    ) -> None:
        resolved = path.resolve()
        changed = any(
            metadata.source_path.resolve() != resolved
            or metadata.file_size != fingerprint.size
            or metadata.file_modified_time != fingerprint.modified_time_ns
            for key, metadata in self._metadata.items()
            if key[0] == dataset_id
        )
        if changed:
            self.remove_dataset(dataset_id)

    def _store_metadata(self, metadata: DataSessionMetadata) -> None:
        key = (metadata.dataset_id, metadata.encoding, metadata.timestamp_column)
        self._metadata[key] = metadata
        self._touch_dataset(metadata.dataset_id)

    def _touch_dataset(self, dataset_id: str) -> None:
        self._dataset_lru.pop(dataset_id, None)
        self._dataset_lru[dataset_id] = None
        while len(self._dataset_lru) > self.max_datasets:
            oldest, _ = self._dataset_lru.popitem(last=False)
            self.remove_dataset(oldest)

    @staticmethod
    def _build_metadata(
        dataset_id: str,
        path: Path,
        fingerprint: FileFingerprint,
        encoding: str,
        timestamp_column: str,
        frame: pd.DataFrame,
    ) -> DataSessionMetadata:
        if timestamp_column not in frame.columns:
            raise ValueError(f"找不到时间列：{timestamp_column}")
        parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
        if parsed.isna().any():
            raise ValueError("时间列包含无法解析的值")
        candidates = []
        for column in frame.columns:
            if column == timestamp_column:
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
        intervals = pd.Series(unique_sorted).diff().dropna().dt.total_seconds() / 60.0
        inferred = float(intervals.mode().iloc[0]) if not intervals.empty else None
        return DataSessionMetadata(
            dataset_id=dataset_id,
            source_path=path,
            file_size=fingerprint.size,
            file_modified_time=fingerprint.modified_time_ns,
            encoding=encoding,
            timestamp_column=timestamp_column,
            parsed_timestamp_series=parsed.reset_index(drop=True),
            column_names=tuple(str(column) for column in frame.columns),
            numeric_candidate_columns=tuple(candidates),
            row_count=len(frame),
            time_start=unique_sorted[0] if len(unique_sorted) else None,
            time_end=unique_sorted[-1] if len(unique_sorted) else None,
            inferred_sample_interval=inferred,
        )

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

    @staticmethod
    def _copy_metadata(metadata: DataSessionMetadata) -> DataSessionMetadata:
        return DataSessionMetadata(
            **{
                **metadata.__dict__,
                "parsed_timestamp_series": metadata.parsed_timestamp_series.copy(
                    deep=True
                ),
            }
        )
