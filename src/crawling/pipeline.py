"""Two-phase, resumable arXiv crawling pipeline.

Phase 1 resolves every candidate ID and persists one discovery outcome per ID.
Phase 2 starts only after that barrier, then streams discovered metadata through
source processing while a bounded queue feeds the Semantic Scholar crawler.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import copy
import csv
import hashlib
import json
import logging
import os
import queue
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError

from src.crawling import metadata, processor
from src.crawling.logger import get_logger, setup_logging
from src.crawling.stats import DurationStats, ProgressMonitor

_NEW_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_OLD_ARXIV_ID = re.compile(
    r"^([a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
_RANGE_FIELDS = ("start_month", "start_id", "end_month", "end_id")
_RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}

logger = get_logger("pipeline")


def _ensure_logging() -> None:
    """Install the project console/file handlers when the pipeline is run directly."""
    app_logger = logging.getLogger("arxiv_project")
    if not app_logger.handlers:
        setup_logging()


@dataclass(frozen=True)
class InputRequest:
    """One CLI input source before normalization."""

    start_month: str | None = None
    start_id: int | None = None
    end_month: str | None = None
    end_id: int | None = None
    ranges_file: str | Path | None = None
    ids_file: str | Path | None = None
    limit: int | None = None


@dataclass(frozen=True)
class LoadedInput:
    """Canonical input definition used for fingerprinting and discovery."""

    kind: str
    input_path: str | None
    fingerprint: str
    candidate_count: int
    unique_count: int
    ranges_count: int
    _unique_ids: tuple[str, ...]
    found_limit: int | None = None
    range_spec: _Range | None = None

    def iter_unique_ids(self) -> Iterator[str]:
        """Yield normalized IDs without exposing a mutable collection."""
        return iter(self._unique_ids)

    def to_json(self) -> dict[str, object]:
        """Return the compact, reportable input.json payload."""
        return {
            "input_kind": self.kind,
            "input_path": self.input_path,
            "input_fingerprint": self.fingerprint,
            "candidate_count": self.candidate_count,
            "unique_count": self.unique_count,
            "ranges_count": self.ranges_count,
            "limit_mode": "found" if self.found_limit is not None else None,
            "found_limit": self.found_limit,
            "range": (
                {
                    "start_month": self.range_spec.start_month,
                    "start_id": self.range_spec.start_id,
                    "end_month": self.range_spec.end_month,
                    "end_id": self.range_spec.end_id,
                }
                if self.range_spec is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _Range:
    start_month: str
    start_id: int
    end_month: str
    end_id: int


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry and rate-limit circuit configuration."""

    max_retries: int = 5
    base_delay: float = 10.0
    max_delay: float = 300.0
    jitter_ratio: float = 0.25
    circuit_breaker_threshold: int = 3
    circuit_cooldown: float = 180.0
    pause_after_429: int = 6
    rate_limit_cooldown: float = 20.0
    rate_limit_window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must not be smaller than base_delay")
        if self.jitter_ratio < 0:
            raise ValueError("jitter_ratio must be non-negative")
        if self.circuit_breaker_threshold < 1 or self.pause_after_429 < 1:
            raise ValueError("circuit thresholds must be positive")
        if self.circuit_cooldown < 0:
            raise ValueError("circuit_cooldown must be non-negative")
        if self.rate_limit_cooldown < 0:
            raise ValueError("rate_limit_cooldown must be non-negative")
        if self.rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be positive")


@dataclass(frozen=True)
class DiscoveryFailure:
    """A per-ID failure returned by a discovery adapter."""

    arxiv_id: str
    retryable: bool
    error_type: str
    error: str


@dataclass(frozen=True)
class DiscoveryFetchResult:
    """Structured output for adapters that can isolate per-ID failures."""

    papers: tuple[Mapping[str, object], ...] = ()
    failures: tuple[DiscoveryFailure, ...] = ()


@dataclass(frozen=True)
class DiscoveryBatchOutcome:
    """Persistable discovery records plus retry telemetry for one batch."""

    records: tuple[dict[str, object], ...]
    api_time_seconds: float = 0.0
    retry_sleep_seconds: float = 0.0
    retry_count: int = 0
    http_429_count: int = 0
    http_503_count: int = 0
    consecutive_429: int = 0
    paused_rate_limit: bool = False
    cooldown_until_epoch: float | None = None


@dataclass(frozen=True)
class PrefetchedSource:
    """Completed source downloads waiting for Processor consumption.

    The queue stores paths and small status mappings only; archive bytes remain
    on disk in the download cache.  ``downloads`` maps a versioned ID to its
    download result, archive path and measured download duration.
    """

    metadata: Mapping[str, object]
    downloads: Mapping[str, tuple[Mapping[str, object], Path | None, float]]
    started_at: float | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime, batching and storage configuration for one run directory."""

    run_dir: str | Path
    corpus_dir: str | Path
    download_cache_dir: str | Path | None = None
    discovery_batch_size: int = 100
    discovery_fallback_batch_size: int = 25
    source_batch_size: int = 50
    download_workers: int = 10
    source_request_interval: float | None = None
    reference_workers: int | None = None
    reference_batch_size: int | None = None
    reference_queue_size: int = 2
    checkpoint_every: int = 25
    fsync_every: int = 10
    min_free_gb: float = 1.0
    max_runtime_hours: float | None = None
    grace_period_seconds: float = 600.0
    without_references: bool = False
    keep_downloads: bool = False
    debug_reference_metrics: bool = False
    retry_policy: RetryPolicy = RetryPolicy()
    benchmark_sample_interval: float = 1.0
    source_prefetch_queue_size: int = 2
    progress_monitor: bool = True
    progress_interval: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "discovery_batch_size": self.discovery_batch_size,
            "discovery_fallback_batch_size": self.discovery_fallback_batch_size,
            "source_batch_size": self.source_batch_size,
            "download_workers": self.download_workers,
            "source_prefetch_queue_size": self.source_prefetch_queue_size,
            "reference_queue_size": self.reference_queue_size,
            "checkpoint_every": self.checkpoint_every,
            "fsync_every": self.fsync_every,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.reference_workers is not None and self.reference_workers <= 0:
            raise ValueError("reference_workers must be greater than zero")
        if (
            self.source_request_interval is not None
            and self.source_request_interval < 0
        ):
            raise ValueError("source_request_interval must be non-negative")
        if self.reference_batch_size is not None and self.reference_batch_size <= 0:
            raise ValueError("reference_batch_size must be greater than zero")
        if self.min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative")
        if self.max_runtime_hours is not None and self.max_runtime_hours <= 0:
            raise ValueError("max_runtime_hours must be greater than zero")
        if self.grace_period_seconds < 0:
            raise ValueError("grace_period_seconds must be non-negative")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be greater than zero")

    @property
    def resolved_run_dir(self) -> Path:
        return Path(self.run_dir).expanduser().resolve()

    @property
    def resolved_corpus_dir(self) -> Path:
        return Path(self.corpus_dir).expanduser().resolve()

    @property
    def resolved_cache_dir(self) -> Path:
        if self.download_cache_dir is not None:
            return Path(self.download_cache_dir).expanduser().resolve()
        return self.resolved_corpus_dir.parent / "download-cache"

    @property
    def min_free_bytes(self) -> int:
        return int(self.min_free_gb * 1024**3)


class SourceProcessor(Protocol):
    def process(self, metadata: Mapping[str, object]) -> Mapping[str, object]: ...


class ReferenceBatchCrawler(Protocol):
    def process_batch(
        self, arxiv_ids: list[str], force_refresh: bool = False
    ) -> Mapping[str, object]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    fsync: bool = True,
) -> None:
    """Write JSON through a sibling temporary file and ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream valid object records; a truncated final line is ignored."""
    source = Path(path)
    if not source.exists():
        return
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid JSONL at %s:%d", source, line_number)
                continue
            if isinstance(value, dict):
                yield value


class JsonlJournal:
    """Thread-safe append-only journal with configurable fsync cadence."""

    def __init__(self, path: str | Path, fsync_every: int = 10) -> None:
        if fsync_every <= 0:
            raise ValueError("fsync_every must be greater than zero")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.fsync_every = fsync_every
        self._lock = threading.Lock()
        self._since_fsync = 0

    def append(
        self, records: Mapping[str, object] | Iterable[Mapping[str, object]]
    ) -> int:
        if isinstance(records, Mapping):
            values = [records]
        else:
            values = list(records)
        if not values:
            return 0
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            self._since_fsync += len(values)
            if self._since_fsync >= self.fsync_every:
                os.fsync(handle.fileno())
                self._since_fsync = 0
        return len(values)

    def sync(self) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
            self._since_fsync = 0


class StateStore:
    """Lock-protected atomic run_state.json checkpoint."""

    def __init__(self, path: str | Path, state: Mapping[str, object]) -> None:
        self.path = Path(path)
        self._state: dict[str, Any] = dict(state)
        self._lock = threading.RLock()

    @classmethod
    def load(cls, path: str | Path) -> StateStore:
        source = Path(path)
        return cls(source, _read_json(source))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def update(self, **changes: object) -> dict[str, Any]:
        with self._lock:
            self._state.update(changes)
            self._state["updated_at_utc"] = utc_now()
            atomic_write_json(self.path, self._state)
            return self.snapshot()

    def mutate(self, callback: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            callback(self._state)
            self._state["updated_at_utc"] = utc_now()
            atomic_write_json(self.path, self._state)
            return self.snapshot()


def _chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _latest_discovery_statuses(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    for record in iter_jsonl(path):
        arxiv_id = record.get("arxiv_id")
        status = record.get("status")
        if isinstance(arxiv_id, str) and isinstance(status, str):
            latest[arxiv_id] = status
    return latest


def _latest_discovery_version_counts(path: Path) -> dict[str, int]:
    """Return the latest discovered version count for each ``found`` ID."""
    latest: dict[str, int] = {}
    for record in iter_jsonl(path):
        arxiv_id = record.get("arxiv_id")
        if not isinstance(arxiv_id, str):
            continue
        if record.get("status") != "found":
            latest.pop(arxiv_id, None)
            continue
        metadata = record.get("metadata")
        versions = metadata.get("versions") if isinstance(metadata, Mapping) else None
        latest[arxiv_id] = len(versions) if isinstance(versions, list) else 0
    return latest


def _manifest_index(path: Path) -> dict[str, dict[str, str]]:
    """Reduce manifest upserts to compact per-stage status strings."""
    latest: dict[str, dict[str, str]] = {}
    for record in iter_jsonl(path):
        arxiv_id = record.get("arxiv_id")
        if not isinstance(arxiv_id, str):
            continue
        entry = latest.setdefault(arxiv_id, {})
        for stage in ("source", "processing", "metadata", "reference"):
            value = record.get(stage)
            if isinstance(value, Mapping) and isinstance(value.get("status"), str):
                entry[stage] = str(value["status"])
                if "retryable" in value:
                    entry[f"{stage}_retryable"] = str(
                        _stage_value_is_retryable(stage, value)
                    )
        if isinstance(record.get("overall_status"), str):
            entry["overall"] = str(record["overall_status"])
    return latest


def _stage_value_is_retryable(stage: str, value: Mapping[str, object]) -> bool:
    """Infer retryability without treating permanent partial source work as retryable."""
    if stage != "source" or value.get("status") != "partial_success":
        return bool(value.get("retryable", False))
    versions = value.get("versions")
    if not isinstance(versions, list):
        return bool(value.get("retryable", False))
    for version in versions:
        if not isinstance(version, Mapping):
            continue
        for part in (version.get("download"), version.get("processing")):
            if isinstance(part, Mapping) and bool(part.get("retryable")):
                return True
    return False


def failed_ids_by_stage(
    discovery_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, list[str]]:
    """Derive failed IDs without a redundant failed_ids.jsonl file."""
    return _failed_ids_from_indexes(
        _latest_discovery_statuses(Path(discovery_path)),
        _manifest_index(Path(manifest_path)),
    )


def _failed_ids_from_indexes(
    discovery_statuses: Mapping[str, str],
    manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, list[str]]:
    discovery_failed = sorted(
        arxiv_id
        for arxiv_id, status in discovery_statuses.items()
        if status in {"retryable_failed", "rate_limited", "permanent_failed"}
    )
    source_failed = sorted(
        arxiv_id
        for arxiv_id, statuses in manifest.items()
        if statuses.get("source")
        in {"retryable_failed", "permanent_failed", "partial_success", "failed"}
    )
    reference_failed = sorted(
        arxiv_id
        for arxiv_id, statuses in manifest.items()
        if statuses.get("reference")
        in {"retryable_failed", "permanent_failed", "failed"}
    )
    return {
        "discovery": discovery_failed,
        "source": source_failed,
        "reference": reference_failed,
    }


def normalize_arxiv_id(raw: object) -> str:
    """Normalize an arXiv ID or URL to its unversioned identifier."""
    value = str(raw).strip()
    value = re.sub(
        r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf|src)/",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    for pattern in (_NEW_ARXIV_ID, _OLD_ARXIV_ID):
        match = pattern.fullmatch(value)
        if match:
            return match.group(1)
    raise ValueError(f"invalid arXiv ID: {raw!r}")


def _normalize_month(raw: object) -> tuple[int, int, str]:
    value = str(raw).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
    else:
        match = re.fullmatch(r"(\d{2})(\d{2})", value)
        if not match:
            raise ValueError(f"invalid month {raw!r}; expected YYYY-MM or YYMM")
        year, month = 2000 + int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 01 and 12: {raw!r}")
    return year, month, f"{year:04d}-{month:02d}"


def _make_range(
    start_month: object,
    start_id: object,
    end_month: object,
    end_id: object,
) -> _Range:
    start_year, start_month_number, normalized_start = _normalize_month(start_month)
    end_year, end_month_number, normalized_end = _normalize_month(end_month)
    try:
        first = int(start_id)
        last = int(end_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("range IDs must be integers") from exc
    if not 1 <= first <= 99999 or not 1 <= last <= 99999:
        raise ValueError("range IDs must be between 1 and 99999")
    start_index = start_year * 12 + start_month_number - 1
    end_index = end_year * 12 + end_month_number - 1
    if start_index > end_index or (start_index == end_index and first > last):
        raise ValueError("range start must not be after range end")
    return _Range(normalized_start, first, normalized_end, last)


def _iter_range_ids(value: _Range) -> Iterator[str]:
    start_year, start_month, _ = _normalize_month(value.start_month)
    end_year, end_month, _ = _normalize_month(value.end_month)
    start_index = start_year * 12 + start_month - 1
    end_index = end_year * 12 + end_month - 1
    for month_index in range(start_index, end_index + 1):
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        first = value.start_id if month_index == start_index else 1
        last = value.end_id if month_index == end_index else 99999
        for number in range(first, last + 1):
            yield f"{year % 100:02d}{month:02d}.{number:05d}"


def _read_ranges(path: Path) -> list[_Range]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("ranges file must contain a header")
            normalized_names = {
                name.strip().lower().replace("-", "_"): name
                for name in reader.fieldnames
                if name is not None
            }
            missing = [name for name in _RANGE_FIELDS if name not in normalized_names]
            if missing:
                raise ValueError("ranges file header is missing: " + ", ".join(missing))
            ranges: list[_Range] = []
            for line_number, row in enumerate(reader, start=2):
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                try:
                    ranges.append(
                        _make_range(
                            row[normalized_names["start_month"]],
                            row[normalized_names["start_id"]],
                            row[normalized_names["end_month"]],
                            row[normalized_names["end_id"]],
                        )
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"invalid ranges file row {line_number}: {exc}"
                    ) from exc
            return ranges
    except OSError as exc:
        raise ValueError(f"cannot read ranges file: {path}") from exc


def _iter_ids_file(path: Path) -> Iterator[str]:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                try:
                    yield normalize_arxiv_id(value)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid ID at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read IDs file: {path}") from exc


def _limited(values: Iterable[str], limit: int | None) -> Iterator[str]:
    for index, value in enumerate(values):
        if limit is not None and index >= limit:
            break
        yield value


def _fingerprint(
    kind: str,
    unique_ids: Sequence[str],
    found_limit: int | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"pipeline-input-v1\0{kind}\0".encode())
    if found_limit is not None:
        digest.update(f"found-limit={found_limit}\0".encode())
    for arxiv_id in unique_ids:
        digest.update(arxiv_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_input(request: InputRequest) -> LoadedInput:
    """Validate, normalize and deduplicate one input definition.

    For a CLI range, ``limit`` is retained as a downstream discovery target:
    it limits the number of valid ``found`` papers rather than candidates.
    """
    if request.limit is not None and request.limit <= 0:
        raise ValueError("limit must be greater than zero")

    has_cli_range = any(
        value is not None
        for value in (
            request.start_month,
            request.start_id,
            request.end_month,
            request.end_id,
        )
    )
    if has_cli_range and not all(
        value is not None
        for value in (
            request.start_month,
            request.start_id,
            request.end_month,
            request.end_id,
        )
    ):
        raise ValueError("CLI range requires all four range arguments")

    selected = sum(
        (has_cli_range, request.ranges_file is not None, request.ids_file is not None)
    )
    if selected != 1:
        raise ValueError("select exactly one CLI range, ranges file, or IDs file")
    if request.limit is not None and (
        request.ranges_file is not None or request.ids_file is not None
    ):
        raise ValueError("limit is only supported with a single CLI range")

    input_path: str | None = None
    ranges_count = 0
    range_spec: _Range | None = None
    found_limit: int | None = None
    if has_cli_range:
        kind = "range"
        range_spec = _make_range(
            request.start_month,
            request.start_id,
            request.end_month,
            request.end_id,
        )
        ranges = [range_spec]
        ranges_count = 1
        raw_ids: Iterable[str] = (
            arxiv_id for value in ranges for arxiv_id in _iter_range_ids(value)
        )
        found_limit = request.limit
    elif request.ranges_file is not None:
        kind = "ranges_file"
        path = Path(request.ranges_file).expanduser().resolve()
        input_path = str(path)
        ranges = _read_ranges(path)
        ranges_count = len(ranges)
        raw_ids = (arxiv_id for value in ranges for arxiv_id in _iter_range_ids(value))
    else:
        kind = "ids_file"
        path = Path(request.ids_file).expanduser().resolve()  # type: ignore[arg-type]
        input_path = str(path)
        raw_ids = _iter_ids_file(path)

    seen: set[str] = set()
    unique_ids: list[str] = []
    candidate_count = 0
    for arxiv_id in raw_ids:
        candidate_count += 1
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            unique_ids.append(arxiv_id)

    return LoadedInput(
        kind=kind,
        input_path=input_path,
        fingerprint=_fingerprint(kind, unique_ids, found_limit),
        candidate_count=candidate_count,
        unique_count=len(unique_ids),
        ranges_count=ranges_count,
        _unique_ids=tuple(unique_ids),
        found_limit=found_limit,
        range_spec=range_spec,
    )


def _http_status(error: BaseException) -> int | None:
    for candidate in (
        getattr(error, "code", None),
        getattr(error, "status", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _retry_after(error: BaseException, now: Callable[[], float]) -> float | None:
    headers = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    value = str(value).strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, parsed.timestamp() - now())


def _classify_error(error: BaseException) -> tuple[bool, str, int | None]:
    status = _http_status(error)
    if status is not None:
        return (
            status in _RETRYABLE_HTTP_CODES,
            f"http_{status}",
            status,
        )
    if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower():
        return True, "timeout", None
    if isinstance(
        error,
        (ConnectionError, URLError, RemoteDisconnected),
    ) or any(
        word in type(error).__name__.lower() for word in ("connection", "network")
    ):
        return True, "network_error", None
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return False, "invalid_response", None
    if "invalidsourceformat" in type(error).__name__.lower():
        return False, "invalid_source_format", None
    return True, type(error).__name__.lower(), None


def _backoff_delay(
    retry_number: int,
    policy: RetryPolicy,
    random_value: Callable[[], float],
) -> float:
    base = min(policy.max_delay, policy.base_delay * (2 ** (retry_number - 1)))
    jitter = base * policy.jitter_ratio * max(0.0, random_value())
    return min(policy.max_delay, base + jitter)


def _failure_record(
    arxiv_id: str,
    *,
    status: str,
    retryable: bool,
    error_type: str,
    attempts: int,
    error: str,
) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "status": status,
        "retryable": retryable,
        "error_type": error_type,
        "attempts": attempts,
        "error": error,
    }


def discover_batch(
    arxiv_ids: Sequence[str],
    fetch: Callable[
        [list[str]],
        Sequence[Mapping[str, object]] | DiscoveryFetchResult,
    ],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], object] = time.sleep,
    monotonic: Callable[[], float] = time.perf_counter,
    wall_time: Callable[[], float] = time.time,
    random_value: Callable[[], float] = random.random,
) -> DiscoveryBatchOutcome:
    """Resolve one bounded ID batch with retry, Retry-After and a circuit.

    A successful response omission is ``not_found``. Exceptions are classified
    independently, so a request failure can never be recorded as not found.
    """
    requested = list(dict.fromkeys(normalize_arxiv_id(value) for value in arxiv_ids))
    if not requested:
        return DiscoveryBatchOutcome(records=())

    policy = policy or RetryPolicy()
    pending = list(requested)
    terminal: dict[str, dict[str, object]] = {}
    attempts_by_id = {arxiv_id: 0 for arxiv_id in requested}
    api_seconds = 0.0
    sleep_seconds = 0.0
    retries = 0
    http_429 = 0
    http_503 = 0
    consecutive_429 = 0
    cooldown_until: float | None = None
    paused = False
    last_error: BaseException | None = None
    last_error_type = "retryable_error"

    while pending:
        for arxiv_id in pending:
            attempts_by_id[arxiv_id] += 1
        started = monotonic()
        try:
            response = fetch(list(pending))
        except Exception as error:  # noqa: BLE001 - adapter errors are classified here.
            api_seconds += max(0.0, monotonic() - started)
            last_error = error
            retryable, error_type, status_code = _classify_error(error)
            last_error_type = error_type
            if status_code == 429:
                http_429 += 1
                consecutive_429 += 1
            else:
                consecutive_429 = 0
            if status_code == 503:
                http_503 += 1

            if not retryable:
                for arxiv_id in pending:
                    terminal[arxiv_id] = _failure_record(
                        arxiv_id,
                        status="permanent_failed",
                        retryable=False,
                        error_type=error_type,
                        attempts=attempts_by_id[arxiv_id],
                        error=str(error),
                    )
                pending = []
                break

            if status_code == 429 and consecutive_429 >= policy.pause_after_429:
                paused = True
                cooldown_until = wall_time() + policy.circuit_cooldown
                for arxiv_id in pending:
                    terminal[arxiv_id] = _failure_record(
                        arxiv_id,
                        status="rate_limited",
                        retryable=True,
                        error_type=error_type,
                        attempts=attempts_by_id[arxiv_id],
                        error=str(error),
                    )
                pending = []
                break

            retry_number = max(attempts_by_id[arxiv_id] for arxiv_id in pending)
            if retry_number > policy.max_retries:
                failed_status = (
                    "rate_limited" if status_code == 429 else "retryable_failed"
                )
                for arxiv_id in pending:
                    terminal[arxiv_id] = _failure_record(
                        arxiv_id,
                        status=failed_status,
                        retryable=True,
                        error_type=error_type,
                        attempts=attempts_by_id[arxiv_id],
                        error=str(error),
                    )
                pending = []
                break

            retries += 1
            delay = _backoff_delay(retry_number, policy, random_value)
            retry_after = _retry_after(error, wall_time)
            if retry_after is not None:
                delay = max(delay, retry_after)
            if (
                status_code == 429
                and consecutive_429 >= policy.circuit_breaker_threshold
            ):
                delay = max(delay, policy.circuit_cooldown)
                cooldown_until = wall_time() + delay
            logger.warning(
                "Discovery ID batch retry: pending=%d attempt=%d/%d status=%s error_type=%s delay=%.2fs",
                len(pending),
                retry_number,
                policy.max_retries,
                status_code,
                error_type,
                delay,
            )
            sleep(delay)
            sleep_seconds += delay
            continue
        else:
            api_seconds += max(0.0, monotonic() - started)
            consecutive_429 = 0

        if isinstance(response, DiscoveryFetchResult):
            papers = response.papers
            failures = {failure.arxiv_id: failure for failure in response.failures}
        else:
            papers = response
            failures = {}

        paper_by_id: dict[str, Mapping[str, object]] = {}
        for paper in papers:
            if not isinstance(paper, Mapping) or "arxiv_id" not in paper:
                raise ValueError("discovery returned a paper without arxiv_id")
            paper_id = normalize_arxiv_id(paper["arxiv_id"])
            if paper_id in pending:
                paper_by_id[paper_id] = paper

        next_pending: list[str] = []
        for arxiv_id in pending:
            paper = paper_by_id.get(arxiv_id)
            failure = failures.get(arxiv_id)
            if paper is not None:
                terminal[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "status": "found",
                    "retryable": False,
                    "attempts": attempts_by_id[arxiv_id],
                    "metadata": dict(paper),
                }
            elif failure is not None and failure.retryable:
                last_error_type = failure.error_type
                last_error = RuntimeError(failure.error)
                next_pending.append(arxiv_id)
            elif failure is not None:
                terminal[arxiv_id] = _failure_record(
                    arxiv_id,
                    status="permanent_failed",
                    retryable=False,
                    error_type=failure.error_type,
                    attempts=attempts_by_id[arxiv_id],
                    error=failure.error,
                )
            else:
                terminal[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "status": "not_found",
                    "retryable": False,
                    "attempts": attempts_by_id[arxiv_id],
                }
        pending = next_pending
        if not pending:
            break

        retry_number = max(attempts_by_id[arxiv_id] for arxiv_id in pending)
        if retry_number > policy.max_retries:
            for arxiv_id in pending:
                terminal[arxiv_id] = _failure_record(
                    arxiv_id,
                    status="retryable_failed",
                    retryable=True,
                    error_type=last_error_type,
                    attempts=attempts_by_id[arxiv_id],
                    error=str(last_error or "incomplete discovery response"),
                )
            pending = []
            break
        retries += 1
        delay = _backoff_delay(retry_number, policy, random_value)
        logger.warning(
            "Discovery ID batch retry: pending=%d attempt=%d/%d reason=%s delay=%.2fs",
            len(pending),
            retry_number,
            policy.max_retries,
            last_error_type,
            delay,
        )
        sleep(delay)
        sleep_seconds += delay

    records = tuple(terminal[arxiv_id] for arxiv_id in requested)
    return DiscoveryBatchOutcome(
        records=records,
        api_time_seconds=api_seconds,
        retry_sleep_seconds=sleep_seconds,
        retry_count=retries,
        http_429_count=http_429,
        http_503_count=http_503,
        consecutive_429=consecutive_429,
        paused_rate_limit=paused,
        cooldown_until_epoch=cooldown_until,
    )


class DiskSpaceLow(RuntimeError):
    """Raised before starting more work when the configured disk floor is hit."""


class RuntimeBudgetReached(RuntimeError):
    """Raised at a safe batch boundary before the Colab runtime deadline."""


class SourceRateLimitPaused(RuntimeError):
    """Downloader circuit opened after bounded automatic retries."""


class RateLimitPhasePaused(RuntimeError):
    """A downstream rate-limit circuit requested a safe phase checkpoint."""


class ArxivDiscoveryAdapter:
    """One-attempt adapter over the configured discovery client.

    Retry and persistence intentionally stay in :func:`discover_batch`, so the
    pipeline can retain HTTP status, attempt, sleep and circuit telemetry.
    """

    def fetch(self, arxiv_ids: list[str]) -> DiscoveryFetchResult:
        from src.crawling import discovery

        search = discovery.arxiv.Search(
            id_list=arxiv_ids,
            max_results=len(arxiv_ids),
        )
        papers = list(discovery.client.results(search))
        paper_by_id = {discovery.get_base_id(paper): paper for paper in papers}

        revision_ids: list[str] = []
        for arxiv_id, paper in paper_by_id.items():
            latest = discovery.get_latest_version(paper)
            revision_ids.extend(f"{arxiv_id}v{version}" for version in range(2, latest))

        revisions: dict[str, Any] = {}
        if revision_ids:
            revision_search = discovery.arxiv.Search(
                id_list=revision_ids,
                max_results=len(revision_ids),
            )
            revisions = {
                paper.get_short_id(): paper
                for paper in discovery.client.results(revision_search)
            }

        found: list[Mapping[str, object]] = []
        failures: list[DiscoveryFailure] = []
        for arxiv_id, paper in paper_by_id.items():
            try:
                latest = discovery.get_latest_version(paper)
                revised_dates: list[str] = []
                missing = False
                for version in range(2, latest):
                    revision = revisions.get(f"{arxiv_id}v{version}")
                    if revision is None:
                        missing = True
                        break
                    revised_dates.append(revision.updated.date().isoformat())
                if latest >= 2:
                    revised_dates.append(paper.updated.date().isoformat())
                if missing:
                    failures.append(
                        DiscoveryFailure(
                            arxiv_id,
                            True,
                            "missing_revision_metadata",
                            "arXiv omitted an expected revision",
                        )
                    )
                    continue
                found.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "submission_date": paper.published.date().isoformat(),
                        "revised_dates": revised_dates,
                        "venue": paper.journal_ref,
                        "versions": discovery.extract_versions(paper),
                    }
                )
            except (AttributeError, TypeError, ValueError) as exc:
                failures.append(
                    DiscoveryFailure(
                        arxiv_id,
                        False,
                        "invalid_schema",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        return DiscoveryFetchResult(tuple(found), tuple(failures))


class _LockedRateLimiter:
    """Serialize calls to downloader.RateLimiter, which is not thread-safe."""

    def __init__(self, limiter: object) -> None:
        self._limiter = limiter
        self._lock = threading.Lock()
        self._wait_seconds = 0.0

    def wait(self) -> float:
        started = time.monotonic()
        with self._lock:
            self._limiter.wait()  # type: ignore[attr-defined]
        elapsed = max(0.0, time.monotonic() - started)
        with self._lock:
            self._wait_seconds += elapsed
        return elapsed

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            self._limiter.cooldown(seconds)  # type: ignore[attr-defined]

    def record_success(self) -> None:
        with self._lock:
            self._limiter.record_success()  # type: ignore[attr-defined]

    @property
    def wait_seconds(self) -> float:
        with self._lock:
            return self._wait_seconds

    @property
    def current_interval(self) -> float:
        with self._lock:
            return float(getattr(self._limiter, "current_interval", 0.0))


class DefaultSourceProcessor:
    """Integrate downloader, processor and metadata without duplicate files."""

    def __init__(
        self,
        corpus_dir: str | Path,
        cache_dir: str | Path,
        config: PipelineConfig,
        *,
        metric_callback: Callable[[str, float | int], None] | None = None,
        sleep: Callable[[float], object] = time.sleep,
        random_value: Callable[[], float] = random.random,
        wall_time: Callable[[], float] = time.time,
        progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        from src.crawling import downloader
        from src.crawling.config import settings

        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir)
        self.pdf_fallback_dir = self.cache_dir / "pdf-fallback"
        self.config = config
        self.metric_callback = metric_callback or (lambda _name, _value: None)
        self.sleep = sleep
        self.random_value = random_value
        self.wall_time = wall_time
        self.progress_callback = progress_callback
        request_interval = (
            config.source_request_interval
            if config.source_request_interval is not None
            else settings.arxiv_api_delay
        )
        self.limiter = _LockedRateLimiter(
            downloader.RateLimiter(request_interval)
        )
        self._circuit_lock = threading.Lock()
        self._rate_limit_events: deque[float] = deque()

    @staticmethod
    def _log_event(event: str, **fields: object) -> None:
        """Write one compact, grep-friendly source lifecycle event."""
        details = " ".join(
            f"{key}={value}" for key, value in sorted(fields.items())
        )
        logger.info("event=%s%s", event, f" {details}" if details else "")

    def _record_rate_limit(self, policy: RetryPolicy) -> int:
        now = self.wall_time()
        cutoff = now - policy.rate_limit_window_seconds
        with self._circuit_lock:
            while self._rate_limit_events and self._rate_limit_events[0] < cutoff:
                self._rate_limit_events.popleft()
            self._rate_limit_events.append(now)
            return len(self._rate_limit_events)

    def _emit_progress(
        self,
        event: str,
        arxiv_id: str,
        *,
        versioned_id: str | None = None,
        current: int | None = None,
        total: int | None = None,
        status: str | None = None,
        message: str | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        payload: dict[str, object] = {
            "event": event,
            "arxiv_id": arxiv_id,
        }
        if versioned_id is not None:
            payload["versioned_id"] = versioned_id
        if current is not None:
            payload["current"] = current
        if total is not None:
            payload["total"] = total
        if status is not None:
            payload["status"] = status
        if message is not None:
            payload["message"] = message
        try:
            self.progress_callback(payload)
        except Exception:
            logger.warning("Source progress callback failed", exc_info=True)

    def process(self, raw_metadata: Mapping[str, object]) -> Mapping[str, object]:
        """Download and process one paper synchronously.

        This remains the compatibility path for callers that use the existing
        ``SourceProcessor`` API.  The pipeline can instead call
        :meth:`prefetch` followed by :meth:`process_prefetched` to overlap the
        next single-connection download with processing of the current paper.
        """
        return self._process(raw_metadata, None)

    def prefetch(self, raw_metadata: Mapping[str, object]) -> PrefetchedSource:
        """Download all missing versions for one paper without processing them.

        Only completed archive paths and compact status data are returned.  A
        retryable or permanent download failure is retained for the consumer to
        journal; a rate-limit pause is propagated so the pipeline can checkpoint
        and stop safely.
        """
        from src.crawling import downloader

        paper = dict(raw_metadata)
        source_started = time.perf_counter()
        self._log_event(
            "source_id_start",
            arxiv_id=paper.get("arxiv_id", ""),
            mode="prefetch",
        )
        try:
            versioned_ids = downloader.build_versioned_ids(paper)
        except (KeyError, TypeError, ValueError):
            # Let the consumer preserve the existing permanent_failed schema.
            return PrefetchedSource(
                metadata=paper,
                downloads={},
                started_at=source_started,
            )
        self._emit_progress(
            "paper_started",
            str(paper.get("arxiv_id", "")),
            total=len(versioned_ids),
            status="prefetching",
            message="starting paper download",
        )
        downloads: dict[str, tuple[Mapping[str, object], Path | None, float]] = {}
        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id"))
        for index, versioned_id in enumerate(versioned_ids, start=1):
            version = int(versioned_id.rsplit("v", 1)[1])
            corpus_version_dir = self._version_dir(arxiv_id, version)
            if self._complete_process_result(corpus_version_dir) is not None:
                self._emit_progress(
                    "version_skipped",
                    arxiv_id,
                    versioned_id=versioned_id,
                    current=index,
                    total=len(versioned_ids),
                    status="skipped",
                    message="processing already complete",
                )
                continue
            self._emit_progress(
                "version_started",
                arxiv_id,
                versioned_id=versioned_id,
                current=index,
                total=len(versioned_ids),
                status="downloading",
                message="starting version download",
            )
            started = time.perf_counter()
            result, archive_path = self._download(versioned_id)
            elapsed = time.perf_counter() - started
            self.metric_callback("source_download_time_seconds", elapsed)
            downloads[versioned_id] = (result, archive_path, elapsed)
            self._emit_progress(
                "version_downloaded" if archive_path is not None else "version_failed",
                arxiv_id,
                versioned_id=versioned_id,
                current=index,
                total=len(versioned_ids),
                status=str(result.get("status", "failed")),
                message="download completed" if archive_path is not None else "download failed",
            )
        return PrefetchedSource(
            metadata=paper,
            downloads=downloads,
            started_at=source_started,
        )

    def process_prefetched(self, prefetched: PrefetchedSource) -> Mapping[str, object]:
        """Process a paper whose source downloads are already complete."""
        return self._process(
            prefetched.metadata,
            prefetched.downloads,
            started_at=prefetched.started_at,
        )

    def _process(
        self,
        raw_metadata: Mapping[str, object],
        prefetched: Mapping[str, tuple[Mapping[str, object], Path | None, float]]
        | None,
        *,
        started_at: float | None = None,
    ) -> Mapping[str, object]:
        from src.crawling import downloader

        paper = dict(raw_metadata)
        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id"))
        source_started = started_at if started_at is not None else time.perf_counter()
        if started_at is None:
            self._log_event("source_id_start", arxiv_id=arxiv_id, mode="process")
        paper_root = metadata.paper_dir(self.corpus_dir, arxiv_id)
        metadata_started = time.perf_counter()
        try:
            metadata_path = metadata.save_metadata(paper, paper_root)
            metadata_result: dict[str, object] = {
                "status": "success",
                "path": metadata_path,
            }
        except Exception as exc:  # noqa: BLE001 - persist a stage-level failure.
            metadata_result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        metadata_seconds = time.perf_counter() - metadata_started

        version_results: list[dict[str, object]] = []
        try:
            versioned_ids = downloader.build_versioned_ids(paper)
        except (KeyError, TypeError, ValueError) as exc:
            source_seconds = max(0.0, time.perf_counter() - source_started)
            self._log_event(
                "source_id_end",
                arxiv_id=arxiv_id,
                status="permanent_failed",
                versions=0,
                completed_versions=0,
                failed_versions=0,
                source_seconds=source_seconds,
            )
            return {
                "source": {
                    "status": "permanent_failed",
                    "retryable": False,
                    "versions": [],
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "processing": {"status": "not_started"},
                "metadata": metadata_result,
                "timings": {
                    "metadata_seconds": metadata_seconds,
                    "source_seconds": source_seconds,
                    "total_seconds": source_seconds,
                },
            }

        self._emit_progress(
            "paper_started",
            arxiv_id,
            total=len(versioned_ids),
            status="processing",
            message="starting paper processing",
        )

        for index, versioned_id in enumerate(versioned_ids, start=1):
            version = int(versioned_id.rsplit("v", 1)[1])
            output_dir = metadata.version_dir(self.corpus_dir, arxiv_id, version)
            self._log_event(
                "source_version_start",
                arxiv_id=arxiv_id,
                versioned_id=versioned_id,
            )
            previous = self._complete_process_result(output_dir)
            if previous is not None:
                self._emit_progress(
                    "version_skipped",
                    arxiv_id,
                    versioned_id=versioned_id,
                    current=index,
                    total=len(versioned_ids),
                    status="skipped",
                    message="processing already complete",
                )
                version_results.append(
                    self._version_record(
                        versioned_id,
                        {"status": "skipped", "retryable": False},
                        previous,
                        0.0,
                        0.0,
                    )
                )
                self._log_event(
                    "source_version_end",
                    arxiv_id=arxiv_id,
                    versioned_id=versioned_id,
                    status="skipped",
                    download_seconds=0.0,
                    processing_seconds=0.0,
                    total_seconds=0.0,
                )
                continue

            self._emit_progress(
                "version_started",
                arxiv_id,
                versioned_id=versioned_id,
                current=index,
                total=len(versioned_ids),
                status="downloading",
                message="starting version download",
            )
            prefetched_download = prefetched.get(versioned_id) if prefetched else None
            if prefetched_download is None:
                download_started = time.perf_counter()
                download_result, archive_path = self._download(versioned_id)
                download_seconds = time.perf_counter() - download_started
                self.metric_callback("source_download_time_seconds", download_seconds)
            else:
                download_result, archive_path, download_seconds = prefetched_download
            if archive_path is None:
                self._emit_progress(
                    "version_failed",
                    arxiv_id,
                    versioned_id=versioned_id,
                    current=index,
                    total=len(versioned_ids),
                    status=str(download_result.get("status", "failed")),
                    message="download failed",
                )
                version_results.append(
                    self._version_record(
                        versioned_id,
                        download_result,
                        {"status": "not_started"},
                        download_seconds,
                        0.0,
                    )
                )
                self._log_event(
                    "source_version_end",
                    arxiv_id=arxiv_id,
                    versioned_id=versioned_id,
                    status=str(download_result.get("status", "failed")),
                    download_seconds=download_seconds,
                    processing_seconds=0.0,
                    total_seconds=download_seconds,
                )
                continue

            processing_started = time.perf_counter()
            processed = processor.process_version(
                archive_path,
                output_dir,
                delete_archive=(
                    not self.config.keep_downloads
                    and not bool(download_result.get("pdf_fallback"))
                ),
                skip_if_done=True,
            ).to_dict()
            processing_seconds = time.perf_counter() - processing_started
            self.metric_callback("source_processing_time_seconds", processing_seconds)
            self._emit_progress(
                "version_processed",
                arxiv_id,
                versioned_id=versioned_id,
                current=index,
                total=len(versioned_ids),
                status=str(processed.get("status", "unknown")),
                message="version processing completed",
            )
            version_results.append(
                self._version_record(
                    versioned_id,
                    download_result,
                    processed,
                    download_seconds,
                    processing_seconds,
                )
            )
            self._log_event(
                "source_version_end",
                arxiv_id=arxiv_id,
                versioned_id=versioned_id,
                status=str(processed.get("status", "unknown")),
                download_seconds=download_seconds,
                processing_seconds=processing_seconds,
                total_seconds=download_seconds + processing_seconds,
            )

        process_statuses = [
            str(result["processing"]["status"])  # type: ignore[index]
            for result in version_results
        ]
        download_statuses = [
            str(result["download"]["status"])  # type: ignore[index]
            for result in version_results
        ]
        retryable_version_failure = any(
            bool(result.get("download", {}).get("retryable"))
            or bool(result.get("processing", {}).get("retryable"))
            for result in version_results
        )
        usable = {"ok", "no_tex"}
        if version_results and all(status in usable for status in process_statuses):
            source_status = (
                "success_with_no_tex" if "no_tex" in process_statuses else "success"
            )
        elif any(status in usable for status in process_statuses):
            source_status = "partial_success"
        elif any(status == "retryable_failed" for status in download_statuses):
            source_status = "retryable_failed"
        else:
            source_status = "permanent_failed"
        if metadata_result["status"] != "success" and source_status in {
            "success",
            "success_with_no_tex",
        }:
            source_status = "partial_success"

        if process_statuses and all(status == "ok" for status in process_statuses):
            processing_status = "success"
        elif process_statuses and all(status in usable for status in process_statuses):
            processing_status = "success_with_no_tex"
        elif any(status in usable for status in process_statuses):
            processing_status = "partial_success"
        elif all(status == "not_started" for status in process_statuses):
            processing_status = "not_started"
        else:
            processing_status = "failed"

        source_seconds = max(0.0, time.perf_counter() - source_started)
        self._log_event(
            "source_id_end",
            arxiv_id=arxiv_id,
            status=source_status,
            versions=len(version_results),
            completed_versions=sum(
                status in usable for status in process_statuses
            ),
            failed_versions=sum(
                status not in usable for status in process_statuses
            ),
            source_seconds=source_seconds,
        )
        return {
            "source": {
                "status": source_status,
                "retryable": source_status == "retryable_failed"
                or (source_status == "partial_success" and retryable_version_failure),
                "versions": version_results,
            },
            "processing": {"status": processing_status},
            "metadata": metadata_result,
            "timings": {
                "metadata_seconds": metadata_seconds,
                "source_seconds": source_seconds,
                "total_seconds": source_seconds,
            },
        }

    def _version_dir(self, arxiv_id: str, version: int) -> Path:
        from src.crawling import metadata

        return metadata.version_dir(self.corpus_dir, arxiv_id, version)

    def _retry_delay(
        self,
        attempt: int,
        error: BaseException,
        status_code: int | None,
        rate_limit_count: int,
    ) -> float:
        policy = self.config.retry_policy
        delay = _backoff_delay(attempt, policy, self.random_value)
        retry_after = _retry_after(error, self.wall_time)
        if retry_after is not None:
            delay = max(delay, retry_after)
        if status_code == 429:
            if rate_limit_count >= policy.circuit_breaker_threshold:
                delay = max(delay, policy.circuit_cooldown)
            delay = max(delay, policy.rate_limit_cooldown)
            # Add jitter after the shared cooldown has been selected.  Jitter
            # inside the exponential backoff alone is ineffective once every
            # worker is clamped to the same Retry-After/circuit duration.
            delay += delay * policy.jitter_ratio * max(
                0.0, self.random_value()
            )
        return delay

    def _sleep_before_retry(
        self,
        delay: float,
        status_code: int | None,
    ) -> None:
        if delay > 0 and status_code == 429:
            self.limiter.cooldown(delay)
        if delay > 0:
            self.metric_callback("downloader_retry_sleep_seconds", delay)
            if status_code == 429:
                self.metric_callback(
                    "downloader_rate_limit_sleep_seconds",
                    delay,
                )
        self.metric_callback("downloader_retry_count", 1)
        self.sleep(delay)

    def _download(self, versioned_id: str) -> tuple[dict[str, object], Path | None]:
        from src.crawling import downloader

        if downloader.is_download_complete(
            versioned_id,
            self.cache_dir,
            fast=True,
        ):
            self.metric_callback("downloader_cache_hit_count", 1)
            path = self.cache_dir / f"{versioned_id}.source"
            return {"status": "skipped", "retryable": False, "attempts": 0}, path

        self.metric_callback("downloader_cache_miss_count", 1)
        policy = self.config.retry_policy
        for attempt in range(1, policy.max_retries + 2):
            try:
                path = downloader.download_version(
                    versioned_id,
                    self.cache_dir,
                    limiter=self.limiter,
                )
                if not downloader.is_download_complete(
                    versioned_id,
                    self.cache_dir,
                    fast=True,
                ):
                    raise downloader.IncompleteDownloadError(
                        "integrity marker did not validate"
                    )
                return {
                    "status": "success",
                    "retryable": False,
                    "attempts": attempt,
                    "path": str(path),
                }, path
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                retryable, error_type, status_code = _classify_error(exc)
                if status_code == 429:
                    self.metric_callback("downloader_http_429_count", 1)
                    rate_limit_count = self._record_rate_limit(policy)
                    if rate_limit_count == policy.circuit_breaker_threshold:
                        self.metric_callback(
                            "downloader_rate_limit_circuit_breaker_count",
                            1,
                        )
                else:
                    rate_limit_count = 0
                if status_code == 429 and rate_limit_count >= policy.pause_after_429:
                    raise SourceRateLimitPaused(str(exc)) from exc
                if status_code == 404:
                    return self._download_pdf_fallback(versioned_id, exc)
                if not retryable or attempt > policy.max_retries:
                    if status_code == 429:
                        delay = self._retry_delay(
                            attempt,
                            exc,
                            status_code,
                            rate_limit_count,
                        )
                        self.limiter.cooldown(delay)
                    return {
                        "status": (
                            "retryable_failed" if retryable else "permanent_failed"
                        ),
                        "retryable": retryable,
                        "error_type": error_type,
                        "attempts": attempt,
                        "error": str(exc),
                    }, None
                delay = self._retry_delay(
                    attempt,
                    exc,
                    status_code,
                    rate_limit_count,
                )
                self._sleep_before_retry(delay, status_code)
        raise AssertionError("unreachable downloader retry state")

    def _download_pdf_fallback(
        self,
        versioned_id: str,
        source_error: BaseException,
    ) -> tuple[dict[str, object], Path | None]:
        from src.crawling import downloader

        policy = self.config.retry_policy
        last_error: BaseException | None = None
        last_error_type = "pdf_fallback_failed"
        last_retryable = False
        last_status_code: int | None = None
        last_rate_limit_count = 0
        last_attempt = 0
        for attempt in range(1, policy.max_retries + 2):
            last_attempt = attempt
            try:
                path = downloader.download_pdf(
                    versioned_id,
                    self.pdf_fallback_dir,
                    limiter=self.limiter,
                )
                self.metric_callback("pdf_fallback_success_count", 1)
                return {
                    "status": "pdf_only",
                    "retryable": False,
                    "attempts": attempt,
                    "path": str(path),
                    "pdf_path": str(path),
                    "pdf_fallback": True,
                    "source_error_type": "http_404",
                    "source_error": str(source_error),
                }, path
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as pdf_error:
                last_error = pdf_error
                last_retryable, last_error_type, last_status_code = _classify_error(
                    pdf_error
                )
                if last_status_code == 429:
                    self.metric_callback("downloader_http_429_count", 1)
                    last_rate_limit_count = self._record_rate_limit(policy)
                    if last_rate_limit_count == policy.circuit_breaker_threshold:
                        self.metric_callback(
                            "downloader_rate_limit_circuit_breaker_count",
                            1,
                        )
                    if last_rate_limit_count >= policy.pause_after_429:
                        raise SourceRateLimitPaused(str(pdf_error)) from pdf_error
                if not last_retryable or attempt > policy.max_retries:
                    if last_status_code == 429:
                        delay = self._retry_delay(
                            attempt,
                            pdf_error,
                            last_status_code,
                            last_rate_limit_count,
                        )
                        self.limiter.cooldown(delay)
                    break
                delay = self._retry_delay(
                    attempt,
                    pdf_error,
                    last_status_code,
                    last_rate_limit_count,
                )
                self._sleep_before_retry(delay, last_status_code)

        assert last_error is not None
        return {
            "status": "retryable_failed" if last_retryable else "permanent_failed",
            "retryable": last_retryable,
            "attempts": last_attempt,
            "error_type": last_error_type,
            "error": str(source_error),
            "pdf_fallback_error": str(last_error),
        }, None

    @staticmethod
    def _version_record(
        versioned_id: str,
        download: Mapping[str, object],
        processing: Mapping[str, object],
        download_seconds: float,
        processing_seconds: float,
    ) -> dict[str, object]:
        process_status = str(processing.get("status", "not_started"))
        if process_status == "ok":
            status = "success"
        elif process_status == "no_tex":
            status = "success_with_no_tex"
        else:
            status = str(download.get("status", process_status))
        return {
            "versioned_id": versioned_id,
            "status": status,
            "download": dict(download),
            "processing": dict(processing),
            "timings": {
                "download_seconds": download_seconds,
                "processing_seconds": processing_seconds,
            },
        }

    @staticmethod
    def _complete_process_result(output_dir: Path) -> Mapping[str, object] | None:
        from src.crawling import processor

        previous = processor.read_status(output_dir)
        if previous is None or previous.status not in {
            processor.STATUS_OK,
            processor.STATUS_NO_TEX,
        }:
            return None
        total = 0
        for relative in previous.kept_files:
            path = output_dir / relative
            if not path.is_file():
                return None
            total += path.stat().st_size
        return previous.to_dict() if total == previous.kept_size else None


class Pipeline:
    """Coordinate the discovery barrier and bounded downstream phase."""

    _DISCOVERY_TERMINAL = frozenset({"found", "not_found", "permanent_failed"})
    _SOURCE_TERMINAL = frozenset({"success", "success_with_no_tex", "permanent_failed"})
    _REFERENCE_TERMINAL = frozenset(
        {"success", "not_found", "permanent_failed", "skipped"}
    )

    def __init__(
        self,
        config: PipelineConfig,
        loaded_input: LoadedInput,
        *,
        discovery_fetch: Callable[
            [list[str]], Sequence[Mapping[str, object]] | DiscoveryFetchResult
        ]
        | None = None,
        source_processor: SourceProcessor | None = None,
        reference_crawler: ReferenceBatchCrawler | None = None,
        sleep: Callable[[float], object] = time.sleep,
        monotonic: Callable[[], float] = time.perf_counter,
        wall_time: Callable[[], float] = time.time,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self.loaded_input = loaded_input
        self.discovery_fetch = discovery_fetch
        self.source_processor = source_processor
        self.reference_crawler = reference_crawler
        self.sleep = sleep
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.random_value = random_value

        self.run_dir = config.resolved_run_dir
        self.corpus_dir = config.resolved_corpus_dir
        self.cache_dir = config.resolved_cache_dir
        self.input_path = self.run_dir / "input.json"
        self.discovery_path = self.run_dir / "discovery.jsonl"
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.state_path = self.run_dir / "run_state.json"
        self.summary_path = self.run_dir / "summary.json"

        self.state: StateStore | None = None
        self.discovery_journal: JsonlJournal | None = None
        self.manifest_journal: JsonlJournal | None = None
        self.benchmark: Any = None
        self.initial_rss_bytes: int | None = None
        self._summary_lock = threading.RLock()
        self._metric_lock = threading.Lock()
        self._source_progress_lock = threading.Lock()
        self._discovery_progress_lock = threading.Lock()
        self._reference_progress_lock = threading.Lock()
        self._discovery_progress: dict[str, object] = {
            "found": 0,
            "batch": 0,
            "status": "pending",
        }
        self._source_progress: dict[str, object] = {
            "total_papers": 0,
            "total_versions": 0,
            "completed_papers": 0,
            "papers_success": 0,
            "papers_partial": 0,
            "papers_failed": 0,
            "versions_completed": 0,
            "versions_failed": 0,
            "worker_order": [],
            "worker_progress": {},
        }
        self._reference_progress: dict[str, object] = {
            "total_papers": 0,
            "completed_papers": 0,
            "references": 0,
            "batches_completed": 0,
            "current_batch": 0,
            "current_batch_total": 0,
            "queue_size": 0,
            "queue_maxsize": config.reference_queue_size,
        }
        self._source_id_duration_stats = DurationStats()
        self._version_download_duration_stats = DurationStats()
        self._version_processing_duration_stats = DurationStats()
        self.progress_monitor: ProgressMonitor | None = None
        self._pending_metrics: dict[str, float | int] = {}
        self._session_initial_rss_bytes: int | None = None
        self._session_initial_disk_used_bytes: int | None = None
        self._rss_samples_accounted = 0
        self._rss_sum_accounted_mb = 0.0
        self._session_started = 0.0
        self._deadline: float | None = None
        self.max_source_batch_seen = 0
        self.source_prefetch_queue_maxsize = config.source_prefetch_queue_size
        self.source_prefetch_queue_observed_max = 0
        self.reference_queue_maxsize = config.reference_queue_size
        self.reference_queue_observed_max = 0
        self._reference_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._reference_log_handler: logging.Handler | None = None
        self._reference_shared_handlers: list[logging.Handler] = []
        # Discovery state is persisted after every batch, but the expensive
        # summary/directory walk is intentionally less frequent.
        self._discovery_summary_batches = 0

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        """Run or resume in the same directory, always checkpointing on exit."""
        self._capture_session_baseline()
        self._prepare(resume)
        assert self.state is not None
        logger.info(
            "Starting pipeline run: run_dir=%s resume=%s candidates=%d unique=%d",
            self.run_dir,
            resume,
            self.loaded_input.candidate_count,
            self.loaded_input.unique_count,
        )
        self._session_started = self.monotonic()
        if self.config.max_runtime_hours is not None:
            self._deadline = (
                self._session_started + self.config.max_runtime_hours * 3600
            )
        self._start_benchmark()
        final_status = "failed"
        pending_exception: BaseException | None = None
        try:
            self.state.update(status="running", stop_reason=None)
            self._check_resources()
            if not bool(self.state.snapshot().get("discovery_complete")):
                logger.info("Phase 1 Discovery started")
                logger.info("event=discovery_start total_ids=%d", self.loaded_input.unique_count)
                discovery_started = self.monotonic()
                try:
                    discovery_complete = self._run_discovery()
                finally:
                    self._record_stage_timing("discovery", discovery_started)
                    logger.info(
                        "event=discovery_end elapsed_seconds=%.4f complete=%s",
                        max(0.0, self.monotonic() - discovery_started),
                        discovery_complete if "discovery_complete" in locals() else False,
                    )
                if not discovery_complete:
                    logger.warning(
                        "Discovery paused before barrier completion: status=%s",
                        self.state.snapshot().get("status"),
                    )
                    final_status = str(self.state.snapshot()["status"])
                    return self._finish(final_status)
                logger.info("Phase 1 Discovery completed; opening downstream barrier")

            self._check_resources()
            logger.info("Phase 2 Downstream started")
            self._run_downstream()
            statuses = _manifest_index(self.manifest_path)
            has_retryable = any(
                value.get("source") in {"retryable_failed", "partial_success", "failed"}
                or value.get("reference") in {"retryable_failed", "failed"}
                for value in statuses.values()
            )
            final_status = "completed_with_failures" if has_retryable else "completed"
            self.state.update(
                phase="downstream",
                status=final_status,
                stop_reason=None,
            )
            logger.info("Pipeline finished: status=%s", final_status)
            return self._finish(final_status)
        except KeyboardInterrupt as exc:
            pending_exception = exc
            final_status = "failed"
            self.state.update(status="failed", stop_reason="keyboard_interrupt")
        except DiskSpaceLow as exc:
            final_status = "failed"
            logger.error("Pipeline stopped: disk space is below the configured floor")
            self.state.update(status="failed", stop_reason="disk_low", error=str(exc))
        except RuntimeBudgetReached as exc:
            final_status = "paused_runtime"
            logger.warning("Pipeline paused at runtime budget boundary")
            self.state.update(
                status="paused_runtime", stop_reason="runtime_budget", error=str(exc)
            )
        except RateLimitPhasePaused as exc:
            final_status = "paused_rate_limit"
            logger.warning("Pipeline paused by a downstream rate-limit circuit")
            self.state.update(
                status="paused_rate_limit",
                stop_reason="downstream_rate_limit",
                error=str(exc),
            )
        except Exception as exc:
            pending_exception = exc
            final_status = "failed"
            logger.exception("Pipeline failed with an unexpected exception")
            self.state.update(
                status="failed",
                stop_reason=type(exc).__name__,
                error=str(exc),
            )
        result = self._finish(final_status)
        if pending_exception is not None:
            raise pending_exception
        return result

    def _prepare(self, resume: bool) -> None:
        _ensure_logging()
        if resume:
            if not self.state_path.is_file() or not self.input_path.is_file():
                raise ValueError("resume requires input.json and run_state.json")
            saved_input = _read_json(self.input_path)
            saved_fingerprint = saved_input.get("input_fingerprint")
            if saved_fingerprint != self.loaded_input.fingerprint:
                raise ValueError("input fingerprint does not match the resumed run")
            self.state = StateStore.load(self.state_path)
            if (
                self.state.snapshot().get("input_fingerprint")
                != self.loaded_input.fingerprint
            ):
                raise ValueError("input fingerprint does not match run_state.json")
            self._migrate_reference_timing_state()
        else:
            if self.run_dir.exists() and any(self.run_dir.iterdir()):
                raise ValueError(
                    f"run directory is not empty; use --resume: {self.run_dir}"
                )
            self.run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.input_path, self.loaded_input.to_json())
            now = utc_now()
            initial_state: dict[str, object] = {
                "run_id": f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
                "phase": "discovery",
                "status": "running",
                "discovery_complete": False,
                "last_discovery_batch": 0,
                "range_discovery_phase": None,
                "range_discovery_date_offset": 0,
                "range_discovery_date_start": None,
                "range_discovery_id_month": None,
                "range_discovery_id_batch": 0,
                "last_source_checkpoint": 0,
                "source_prefetch_pending_ids": [],
                "last_reference_checkpoint": 0,
                "input_fingerprint": self.loaded_input.fingerprint,
                "discovery_retry": {"pending_ids": []},
                "metrics": self._empty_metrics(),
                "timings": {
                    "discovery_time_seconds": 0.0,
                    "source_time_seconds": 0.0,
                    "reference_time_seconds": 0.0,
                    "reference_stage_wall_seconds": 0.0,
                    "reference_queue_wait_seconds": 0.0,
                    "wall_time_seconds": 0.0,
                },
                "created_at_utc": now,
                "updated_at_utc": now,
            }
            self.state = StateStore(self.state_path, initial_state)
            self.state.update()

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.discovery_journal = JsonlJournal(
            self.discovery_path, self.config.fsync_every
        )
        self.manifest_journal = JsonlJournal(
            self.manifest_path, self.config.fsync_every
        )

    def _migrate_reference_timing_state(self) -> None:
        """Keep legacy runs consistent after the reference timing split."""
        assert self.state is not None
        snapshot = self.state.snapshot()
        timings = snapshot.get("timings")
        if not isinstance(timings, Mapping):
            return
        if "reference_stage_wall_seconds" in timings:
            return
        legacy_reference = timings.get("reference_time_seconds", 0.0)
        if not isinstance(legacy_reference, (int, float)):
            legacy_reference = 0.0

        def migrate(state: dict[str, Any]) -> None:
            current = state.setdefault("timings", {})
            if not isinstance(current, dict):
                current = {}
                state["timings"] = current
            current["reference_stage_wall_seconds"] = float(legacy_reference)
            current["reference_queue_wait_seconds"] = 0.0
            current["reference_time_seconds"] = 0.0

        self.state.mutate(migrate)

    @staticmethod
    def _empty_metrics() -> dict[str, float | int]:
        return {
            "discovery_api_time_seconds": 0.0,
            "discovery_retry_sleep_seconds": 0.0,
            "discovery_http_429_count": 0,
            "discovery_http_503_count": 0,
            "discovery_retry_count": 0,
            "source_download_time_seconds": 0.0,
            "source_processing_time_seconds": 0.0,
            "downloader_http_429_count": 0,
            "downloader_retry_count": 0,
            "downloader_retry_sleep_seconds": 0.0,
            "downloader_rate_limit_sleep_seconds": 0.0,
            "downloader_rate_limit_circuit_breaker_count": 0,
            "downloader_cache_hit_count": 0,
            "downloader_cache_miss_count": 0,
            "pdf_fallback_success_count": 0,
            "reference_http_429_count": 0,
        }

    def _start_benchmark(self) -> None:
        assert self.state is not None

        def baseline(state: dict[str, Any]) -> None:
            resources = state.setdefault("resources", {})
            resources.setdefault("initial_rss_bytes", self._session_initial_rss_bytes)
            resources.setdefault(
                "disk_initial_used_bytes", self._session_initial_disk_used_bytes
            )

        snapshot = self.state.mutate(baseline)
        resource_state = snapshot.get("resources") or {}
        self.initial_rss_bytes = resource_state.get("initial_rss_bytes")
        from src.crawling.stats import RunBenchmark

        self.benchmark = RunBenchmark(
            monitor_paths=[self.run_dir],
            size_paths=[self.corpus_dir, self.run_dir, self.cache_dir],
            sample_interval=self.config.benchmark_sample_interval,
        )
        self.benchmark.start_sampling()
        if self.config.progress_monitor:
            self.progress_monitor = ProgressMonitor(
                self._progress_snapshot,
                interval=self.config.progress_interval,
            )
            self.progress_monitor.start()

    def _capture_session_baseline(self) -> None:
        try:
            import psutil

            self._session_initial_rss_bytes = psutil.Process().memory_info().rss
        except (ImportError, OSError, RuntimeError):
            self._session_initial_rss_bytes = None

        disk_path = self.run_dir
        while not disk_path.exists() and disk_path != disk_path.parent:
            disk_path = disk_path.parent
        try:
            usage = shutil.disk_usage(disk_path)
        except OSError:
            self._session_initial_disk_used_bytes = None
        else:
            self._session_initial_disk_used_bytes = usage.total - usage.free

    def _check_resources(self) -> None:
        free = shutil.disk_usage(self.corpus_dir).free
        if free < self.config.min_free_bytes:
            raise DiskSpaceLow(
                f"free disk {free} bytes is below minimum {self.config.min_free_bytes}"
            )
        if self._deadline is not None:
            remaining = self._deadline - self.monotonic()
            if remaining <= self.config.grace_period_seconds:
                raise RuntimeBudgetReached(
                    f"runtime grace boundary reached with {remaining:.1f}s remaining"
                )

    def _uses_cross_month_date_stream(self) -> bool:
        range_spec = self.loaded_input.range_spec
        return (
            range_spec is not None
            and range_spec.start_month != range_spec.end_month
        )

    @staticmethod
    def _select_found_limit_records(
        records: Sequence[Mapping[str, object]],
        found_count: int,
        found_limit: int | None,
    ) -> tuple[list[dict[str, object]], int]:
        """Keep at most the requested number of found papers in a batch."""
        selected: list[dict[str, object]] = []
        current = found_count
        for record in records:
            status = record.get("status")
            if status == "found":
                if found_limit is not None and current >= found_limit:
                    continue
                current += 1
            selected.append(dict(record))
        return selected, current

    def _fetch_date_stream_page(
        self,
        *,
        query: str,
        offset: int,
        target_yymm: str,
        min_id: int,
        max_found: int | None,
    ) -> tuple[dict[str, Any] | None, DiscoveryBatchOutcome]:
        """Fetch one date-stream page with the pipeline retry policy."""
        from src.crawling import discovery

        policy = self.config.retry_policy
        api_seconds = 0.0
        sleep_seconds = 0.0
        retries = 0
        http_429 = 0
        http_503 = 0
        consecutive_429 = 0
        cooldown_until: float | None = None
        last_error: BaseException | None = None
        for attempt in range(1, policy.max_retries + 2):
            started = self.monotonic()
            try:
                page = discovery.fetch_date_stream_batch_once(
                    query=query,
                    offset=offset,
                    batch_size=self.config.discovery_batch_size,
                    target_yymm=target_yymm,
                    min_id=min_id,
                    max_found=max_found,
                    min_subbatch_size=self.config.discovery_fallback_batch_size,
                )
                api_seconds += max(0.0, self.monotonic() - started)
                return page, DiscoveryBatchOutcome(
                    records=(),
                    api_time_seconds=api_seconds,
                    retry_sleep_seconds=sleep_seconds,
                    retry_count=retries,
                    http_429_count=http_429,
                    http_503_count=http_503,
                    consecutive_429=0,
                    cooldown_until_epoch=cooldown_until,
                )
            except Exception as exc:
                api_seconds += max(0.0, self.monotonic() - started)
                last_error = exc
                status_code = _http_status(exc)
                retryable, _error_type, _ = _classify_error(exc)
                if status_code == 429:
                    http_429 += 1
                    consecutive_429 += 1
                else:
                    consecutive_429 = 0
                if status_code == 503:
                    http_503 += 1
                if not retryable:
                    raise
                if status_code == 429 and consecutive_429 >= policy.pause_after_429:
                    cooldown_until = self.wall_time() + policy.circuit_cooldown
                    return None, DiscoveryBatchOutcome(
                        records=(),
                        api_time_seconds=api_seconds,
                        retry_sleep_seconds=sleep_seconds,
                        retry_count=retries,
                        http_429_count=http_429,
                        http_503_count=http_503,
                        consecutive_429=consecutive_429,
                        paused_rate_limit=True,
                        cooldown_until_epoch=cooldown_until,
                    )
                if attempt > policy.max_retries:
                    return None, DiscoveryBatchOutcome(
                        records=(),
                        api_time_seconds=api_seconds,
                        retry_sleep_seconds=sleep_seconds,
                        retry_count=retries,
                        http_429_count=http_429,
                        http_503_count=http_503,
                        consecutive_429=consecutive_429,
                        paused_rate_limit=status_code == 429,
                        cooldown_until_epoch=cooldown_until,
                    )
                retries += 1
                delay = _backoff_delay(attempt, policy, self.random_value)
                retry_after = _retry_after(exc, self.wall_time)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                if status_code == 429 and consecutive_429 >= policy.circuit_breaker_threshold:
                    delay = max(delay, policy.circuit_cooldown)
                    cooldown_until = self.wall_time() + delay
                logger.warning(
                    "Discovery date request retry: offset=%d attempt=%d/%d status=%s error_type=%s delay=%.2fs",
                    offset,
                    attempt,
                    policy.max_retries,
                    status_code,
                    _error_type,
                    delay,
                )
                self.sleep(delay)
                sleep_seconds += delay
        raise RuntimeError(f"date-stream request failed: {last_error}")

    def _run_cross_month_date_stream(self) -> bool:
        """Discover the first month by date, then later months by ID."""
        assert self.state is not None and self.discovery_journal is not None
        range_spec = self.loaded_input.range_spec
        assert range_spec is not None
        from src.crawling import discovery

        state = self.state.snapshot()
        latest = _latest_discovery_statuses(self.discovery_path)
        found_count = sum(status == "found" for status in latest.values())
        self._set_discovery_progress(found=found_count, status="running")
        found_limit = self.loaded_input.found_limit
        retry_state = state.get("discovery_retry")
        retry_ids = (
            [str(value) for value in retry_state.get("pending_ids", [])]
            if isinstance(retry_state, Mapping)
            and isinstance(retry_state.get("pending_ids"), list)
            else []
        )
        if retry_ids:
            retry_outcome = self._discover_with_fallback(retry_ids)
            retry_records, found_count = self._select_found_limit_records(
                retry_outcome.records, found_count, found_limit
            )
            self.discovery_journal.append(retry_records)
            self._set_discovery_progress(found=found_count, status="running")
            self._record_discovery_metrics(retry_outcome)
            unresolved = [
                str(record["arxiv_id"])
                for record in retry_records
                if record.get("status") not in self._DISCOVERY_TERMINAL
            ]
            if unresolved:
                self.state.update(
                    phase="discovery",
                    status="discovery_incomplete",
                    discovery_complete=False,
                    discovery_retry={"pending_ids": unresolved},
                )
                self._checkpoint_summary(
                    "discovery_incomplete", force=True, phase="discovery"
                )
                return False
            self.state.update(discovery_retry={"pending_ids": []})
            latest = _latest_discovery_statuses(self.discovery_path)
        phase = state.get("range_discovery_phase") or "month_1_date_stream"
        logger.info(
            "Discovery resume state: phase=%s date_offset=%s id_month=%s id_batch=%s found=%d limit=%s",
            phase,
            state.get("range_discovery_date_offset", 0),
            state.get("range_discovery_id_month"),
            state.get("range_discovery_id_batch", 0),
            found_count,
            found_limit,
        )
        if found_limit is not None and found_count >= found_limit:
            self.state.update(
                phase="downstream",
                status="running",
                discovery_complete=True,
                range_discovery_phase="completed",
            )
            return True

        start_year, start_month, _ = _normalize_month(range_spec.start_month)
        end_year, end_month, _ = _normalize_month(range_spec.end_month)
        if phase == "month_1_date_stream":
            date_start = state.get("range_discovery_date_start")
            if not isinstance(date_start, str) or not date_start:
                date_start = discovery.find_start_timestamp_month_1(
                    start_year, start_month, range_spec.start_id
                )
                self.state.update(range_discovery_date_start=date_start)
            last_day = calendar.monthrange(start_year, start_month)[1]
            query = f"submittedDate:[{date_start} TO {start_year}{start_month:02d}{last_day:02d}235959]"
            offset = int(state.get("range_discovery_date_offset", 0))
            page_number = offset // self.config.discovery_batch_size + 1
            logger.info(
                "Discovery date-stream started: query=%s offset=%d batch_size=%d",
                query,
                offset,
                self.config.discovery_batch_size,
            )
            while True:
                self._check_resources()
                remaining = (
                    max(0, found_limit - found_count)
                    if found_limit is not None
                    else None
                )
                page_started = self.monotonic()
                logger.debug(
                    "Discovery date page started: page=%d offset=%d remaining=%s",
                    page_number,
                    offset,
                    remaining,
                )
                page, outcome = self._fetch_date_stream_page(
                    query=query,
                    offset=offset,
                    target_yymm=f"{start_year % 100:02d}{start_month:02d}",
                    min_id=range_spec.start_id,
                    max_found=remaining,
                )
                page_elapsed = max(0.0, self.monotonic() - page_started)
                self._record_discovery_metrics(outcome)
                if page is None:
                    logger.warning(
                        "Discovery date page paused: page=%d offset=%d elapsed=%.2fs api=%.2fs retries=%d http_429=%d http_503=%d cooldown_until=%s",
                        page_number,
                        offset,
                        page_elapsed,
                        outcome.api_time_seconds,
                        outcome.retry_count,
                        outcome.http_429_count,
                        outcome.http_503_count,
                        outcome.cooldown_until_epoch,
                    )
                    status = "paused_rate_limit" if outcome.paused_rate_limit else "discovery_incomplete"
                    self.state.update(
                        phase="discovery",
                        status=status,
                        discovery_complete=False,
                        range_discovery_phase="month_1_date_stream",
                        range_discovery_date_offset=offset,
                        discovery_retry={
                            "pending_ids": [],
                            "cooldown_until_epoch": outcome.cooldown_until_epoch,
                        },
                    )
                    self._checkpoint_summary(status, force=True, phase="discovery")
                    return False

                found_papers = page.get("found_papers", [])
                failed_ids = [str(value) for value in page.get("failed_ids", [])]
                records: list[dict[str, object]] = []
                for paper in found_papers:
                    records.append(
                        {
                            "arxiv_id": str(paper["arxiv_id"]),
                            "status": "found",
                            "retryable": False,
                            "attempts": 1,
                            "metadata": dict(paper),
                        }
                    )
                for arxiv_id in failed_ids:
                    records.append(
                        {
                            "arxiv_id": arxiv_id,
                            "status": "retryable_failed",
                            "retryable": True,
                            "error_type": "missing_revision_metadata",
                            "attempts": 1,
                            "error": "missing revision metadata",
                        }
                    )
                self.discovery_journal.append(records)
                for record in records:
                    latest[str(record["arxiv_id"])] = str(record["status"])
                found_count += len(found_papers)
                self._set_discovery_progress(found=found_count, status="running")
                offset = int(page["offset"])
                status_counts: dict[str, int] = {}
                for record in records:
                    status = str(record.get("status"))
                    status_counts[status] = status_counts.get(status, 0) + 1
                logger.info(
                    "Discovery date page completed: page=%d next_offset=%d raw=%d records=%d statuses=%s found_total=%d elapsed=%.2fs api=%.2fs retries=%d http_429=%d http_503=%d",
                    page_number,
                    offset,
                    int(page["raw_count"]),
                    len(records),
                    status_counts,
                    found_count,
                    page_elapsed,
                    outcome.api_time_seconds,
                    outcome.retry_count,
                    outcome.http_429_count,
                    outcome.http_503_count,
                )
                self.state.update(
                    range_discovery_date_offset=offset,
                    range_discovery_phase="month_1_date_stream",
                )
                self._checkpoint_summary("running", phase="discovery")
                if failed_ids:
                    self.state.update(
                        phase="discovery",
                        status="discovery_incomplete",
                        discovery_complete=False,
                        discovery_retry={"pending_ids": failed_ids},
                    )
                    self._checkpoint_summary(
                        "discovery_incomplete", force=True, phase="discovery"
                    )
                    return False
                if found_limit is not None and found_count >= found_limit:
                    self.state.update(
                        phase="downstream",
                        status="running",
                        discovery_complete=True,
                        range_discovery_phase="completed",
                    )
                    self._checkpoint_summary("running", force=True, phase="discovery")
                    return True
                if int(page["raw_count"]) < self.config.discovery_batch_size:
                    logger.info(
                        "Discovery date-stream exhausted: page=%d raw=%d; switching to ID batches",
                        page_number,
                        int(page["raw_count"]),
                    )
                    phase = "remaining_id_batch"
                    next_month_index = start_year * 12 + start_month
                    next_year, next_month_zero = divmod(next_month_index, 12)
                    self.state.update(
                        range_discovery_phase=phase,
                        range_discovery_id_month=(
                            f"{next_year:04d}-{next_month_zero + 1:02d}"
                        ),
                        range_discovery_id_batch=0,
                    )
                    break
                page_number += 1

        if phase in {"month_2_id_batch", "remaining_id_batch"}:
            start_index = start_year * 12 + start_month - 1
            end_index = end_year * 12 + end_month - 1
            saved_month = self.state.snapshot().get("range_discovery_id_month")
            if isinstance(saved_month, str) and re.fullmatch(r"\d{4}-\d{2}", saved_month):
                saved_year, saved_month_number, _ = _normalize_month(saved_month)
                current_index = saved_year * 12 + saved_month_number - 1
            else:
                current_index = start_index + 1

            for month_index in range(current_index, end_index + 1):
                year, zero_based_month = divmod(month_index, 12)
                month = zero_based_month + 1
                last_id = range_spec.end_id if month_index == end_index else 99999
                completed_batch = int(
                    self.state.snapshot().get("range_discovery_id_batch", 0)
                ) if month_index == current_index else 0
                ids = (
                    f"{year % 100:02d}{month:02d}.{number:05d}"
                    for number in range(1, last_id + 1)
                )
                for batch_index, batch in enumerate(
                    _chunks(ids, self.config.discovery_batch_size), start=1
                ):
                    if batch_index <= completed_batch:
                        continue
                    self._check_resources()
                    batch_started = self.monotonic()
                    logger.debug(
                        "Discovery ID batch started: month=%04d-%02d batch=%d size=%d range=%s..%s",
                        year,
                        month,
                        batch_index,
                        len(batch),
                        batch[0],
                        batch[-1],
                    )
                    outcome = self._discover_with_fallback(batch)
                    selected, found_count = self._select_found_limit_records(
                        outcome.records, found_count, found_limit
                    )
                    self.discovery_journal.append(selected)
                    self._set_discovery_progress(found=found_count, status="running")
                    self.state.update(
                        range_discovery_id_month=f"{year:04d}-{month:02d}",
                        range_discovery_id_batch=batch_index,
                        range_discovery_phase="remaining_id_batch",
                    )
                    self._record_discovery_metrics(outcome)
                    batch_elapsed = max(0.0, self.monotonic() - batch_started)
                    status_counts: dict[str, int] = {}
                    for record in selected:
                        status = str(record.get("status"))
                        status_counts[status] = status_counts.get(status, 0) + 1
                    logger.info(
                        "Discovery ID batch completed: month=%04d-%02d batch=%d statuses=%s found_total=%d elapsed=%.2fs api=%.2fs retries=%d http_429=%d http_503=%d",
                        year,
                        month,
                        batch_index,
                        status_counts,
                        found_count,
                        batch_elapsed,
                        outcome.api_time_seconds,
                        outcome.retry_count,
                        outcome.http_429_count,
                        outcome.http_503_count,
                    )
                    self._checkpoint_summary("running", phase="discovery")
                    if found_limit is not None and found_count >= found_limit:
                        break
                if found_limit is not None and found_count >= found_limit:
                    break
                self.state.update(
                    range_discovery_id_month=(
                        f"{year + (1 if month == 12 else 0):04d}-"
                        f"{1 if month == 12 else month + 1:02d}"
                    ),
                    range_discovery_id_batch=0,
                    range_discovery_phase="remaining_id_batch",
                )
            self.state.update(
                phase="downstream",
                status="running",
                discovery_complete=True,
                range_discovery_phase="completed",
            )
            self._checkpoint_summary("running", force=True, phase="discovery")
            return True
        return False

    def _run_discovery(self) -> bool:
        assert self.state is not None and self.discovery_journal is not None
        if self.discovery_fetch is None:
            self.discovery_fetch = ArxivDiscoveryAdapter().fetch
        state_snapshot = self.state.snapshot()
        logger.info(
            "Discovery configuration: batch_size=%d fallback_batch_size=%d checkpoint_every=%d prior_phase=%s prior_batch=%s",
            self.config.discovery_batch_size,
            self.config.discovery_fallback_batch_size,
            self.config.checkpoint_every,
            state_snapshot.get("phase"),
            state_snapshot.get("last_discovery_batch"),
        )
        if self._uses_cross_month_date_stream():
            return self._run_cross_month_date_stream()

        state = self.state.snapshot()
        retry_state = state.get("discovery_retry")
        if isinstance(retry_state, Mapping):
            cooldown_until = retry_state.get("cooldown_until_epoch")
            if isinstance(cooldown_until, (int, float)):
                remaining = float(cooldown_until) - self.wall_time()
                if remaining > 0:
                    self.sleep(remaining)

        latest = _latest_discovery_statuses(self.discovery_path)
        found_limit = self.loaded_input.found_limit
        found_count = sum(status == "found" for status in latest.values())
        self._set_discovery_progress(found=found_count, status="running")
        if found_limit is not None and found_count >= found_limit:
            self.state.update(
                phase="downstream",
                status="running",
                discovery_complete=True,
                discovery_retry={"pending_ids": []},
            )
            return True
        pending_ids = (
            arxiv_id
            for arxiv_id in self.loaded_input.iter_unique_ids()
            if latest.get(arxiv_id) not in self._DISCOVERY_TERMINAL
        )
        next_batch = int(state.get("last_discovery_batch", 0)) + 1
        completed = True
        for batch in _chunks(pending_ids, self.config.discovery_batch_size):
            self._check_resources()
            batch_started = self.monotonic()
            logger.info(
                "Discovery batch %d started: %d candidate IDs range=%s..%s",
                next_batch,
                len(batch),
                batch[0] if batch else None,
                batch[-1] if batch else None,
            )
            outcome = self._discover_with_fallback(batch)
            records, found_count = self._select_found_limit_records(
                outcome.records, found_count, found_limit
            )
            self.discovery_journal.append(records)
            self._set_discovery_progress(found=found_count, batch=next_batch, status="running")
            for record in records:
                latest[str(record["arxiv_id"])] = str(record["status"])
            self._record_discovery_metrics(outcome)
            unresolved = [
                str(record["arxiv_id"])
                for record in records
                if record.get("status") not in self._DISCOVERY_TERMINAL
            ]
            if unresolved:
                logger.warning(
                    "Discovery batch %d incomplete: unresolved=%d paused_rate_limit=%s",
                    next_batch,
                    len(unresolved),
                    outcome.paused_rate_limit,
                )
                completed = False
                status = (
                    "paused_rate_limit"
                    if outcome.paused_rate_limit
                    else "discovery_incomplete"
                )
                self.state.update(
                    phase="discovery",
                    status=status,
                    discovery_complete=False,
                    discovery_retry={
                        "pending_ids": unresolved,
                        "last_error_type": next(
                            (
                                record.get("error_type")
                                for record in outcome.records
                                if record.get("arxiv_id") in unresolved
                            ),
                            None,
                        ),
                        "cooldown_until_epoch": outcome.cooldown_until_epoch,
                    },
                )
                self._checkpoint_summary(status, force=True, phase="discovery")
                break
            self.state.update(
                last_discovery_batch=next_batch,
                discovery_retry={"pending_ids": []},
            )
            self._set_discovery_progress(found=found_count, batch=next_batch, status="running")
            status_counts: dict[str, int] = {}
            for record in records:
                status = str(record.get("status"))
                status_counts[status] = status_counts.get(status, 0) + 1
            logger.info(
                "Discovery batch %d completed: statuses=%s elapsed=%.2fs api=%.2fs retries=%d http_429=%d http_503=%d",
                next_batch,
                status_counts,
                max(0.0, self.monotonic() - batch_started),
                outcome.api_time_seconds,
                outcome.retry_count,
                outcome.http_429_count,
                outcome.http_503_count,
            )
            next_batch += 1
            self._checkpoint_summary("running", phase="discovery")
            if found_limit is not None and found_count >= found_limit:
                break

        all_terminal = all(
            latest.get(arxiv_id) in self._DISCOVERY_TERMINAL
            for arxiv_id in self.loaded_input.iter_unique_ids()
        )
        reached_limit = found_limit is not None and found_count >= found_limit
        completed = completed and (all_terminal or reached_limit)
        if completed:
            self._set_discovery_progress(found=found_count, status="completed")
            self.state.update(
                phase="downstream",
                status="running",
                discovery_complete=True,
                discovery_retry={"pending_ids": []},
            )
        return completed

    def _record_stage_timing(self, stage: str, started: float) -> None:
        elapsed = max(0.0, self.monotonic() - started)
        self._add_timing(f"{stage}_time_seconds", elapsed)
        if self.benchmark is not None:
            self.benchmark.add_stage_wall_time(stage, elapsed)

    def _discover_with_fallback(self, batch: list[str]) -> DiscoveryBatchOutcome:
        """Retry the original batch, then split only non-rate-limit failures."""
        assert self.discovery_fetch is not None

        def request(values: list[str]) -> DiscoveryBatchOutcome:
            return discover_batch(
                values,
                self.discovery_fetch,
                policy=self.config.retry_policy,
                sleep=self.sleep,
                monotonic=self.monotonic,
                wall_time=self.wall_time,
                random_value=self.random_value,
            )

        initial = request(batch)
        retry_ids = [
            str(record["arxiv_id"])
            for record in initial.records
            if record.get("status") == "retryable_failed"
        ]
        if (
            initial.paused_rate_limit
            or len(retry_ids) <= self.config.discovery_fallback_batch_size
        ):
            return initial

        by_id = {str(record["arxiv_id"]): record for record in initial.records}
        totals = {
            "api": initial.api_time_seconds,
            "sleep": initial.retry_sleep_seconds,
            "retries": initial.retry_count,
            "http_429": initial.http_429_count,
            "http_503": initial.http_503_count,
        }
        paused = False
        cooldown = initial.cooldown_until_epoch
        consecutive = initial.consecutive_429
        for smaller in _chunks(retry_ids, self.config.discovery_fallback_batch_size):
            outcome = request(smaller)
            by_id.update(
                (str(record["arxiv_id"]), record) for record in outcome.records
            )
            totals["api"] += outcome.api_time_seconds
            totals["sleep"] += outcome.retry_sleep_seconds
            totals["retries"] += outcome.retry_count
            totals["http_429"] += outcome.http_429_count
            totals["http_503"] += outcome.http_503_count
            paused = paused or outcome.paused_rate_limit
            consecutive = outcome.consecutive_429
            cooldown = outcome.cooldown_until_epoch or cooldown
            if paused:
                break
        return DiscoveryBatchOutcome(
            records=tuple(by_id[arxiv_id] for arxiv_id in batch),
            api_time_seconds=float(totals["api"]),
            retry_sleep_seconds=float(totals["sleep"]),
            retry_count=int(totals["retries"]),
            http_429_count=int(totals["http_429"]),
            http_503_count=int(totals["http_503"]),
            consecutive_429=consecutive,
            paused_rate_limit=paused,
            cooldown_until_epoch=cooldown,
        )

    def _record_discovery_metrics(self, outcome: DiscoveryBatchOutcome) -> None:
        assert self.state is not None

        def update(state: dict[str, Any]) -> None:
            metrics = state.setdefault("metrics", self._empty_metrics())
            additions = {
                "discovery_api_time_seconds": outcome.api_time_seconds,
                "discovery_retry_sleep_seconds": outcome.retry_sleep_seconds,
                "discovery_http_429_count": outcome.http_429_count,
                "discovery_http_503_count": outcome.http_503_count,
                "discovery_retry_count": outcome.retry_count,
            }
            for key, value in additions.items():
                metrics[key] = metrics.get(key, 0) + value

        self.state.mutate(update)

    def _set_discovery_progress(
        self,
        *,
        found: int,
        batch: int | None = None,
        status: str | None = None,
    ) -> None:
        with self._discovery_progress_lock:
            self._discovery_progress["found"] = max(0, int(found))
            if batch is not None:
                self._discovery_progress["batch"] = max(0, int(batch))
            if status is not None:
                self._discovery_progress["status"] = status

    def _add_timing(self, name: str, elapsed: float) -> None:
        assert self.state is not None

        def update(state: dict[str, Any]) -> None:
            timings = state.setdefault("timings", {})
            timings[name] = float(timings.get(name, 0.0)) + elapsed

        self.state.mutate(update)

    def _iter_found_metadata(self) -> Iterator[dict[str, object]]:
        for record in iter_jsonl(self.discovery_path):
            if record.get("status") != "found":
                continue
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                yield dict(metadata)

    def _run_downstream(self) -> None:
        assert self.state is not None and self.manifest_journal is not None
        if self.source_processor is None:
            self.source_processor = DefaultSourceProcessor(
                self.corpus_dir,
                self.cache_dir,
                self.config,
                metric_callback=self._add_metric,
                sleep=self.sleep,
                random_value=self.random_value,
                wall_time=self.wall_time,
                progress_callback=self._on_source_progress,
            )
        phase_started = self.monotonic()
        source_timing_recorded = False
        manifest = _manifest_index(self.manifest_path)
        discovery_statuses = _latest_discovery_statuses(self.discovery_path)
        discovery_version_counts = _latest_discovery_version_counts(
            self.discovery_path
        )
        found_papers = sum(status == "found" for status in discovery_statuses.values())
        total_versions = sum(discovery_version_counts.values())
        with self._source_progress_lock:
            self._source_progress["total_papers"] = found_papers
            self._source_progress["total_versions"] = total_versions
        with self._reference_progress_lock:
            self._reference_progress["total_papers"] = found_papers
        manifest_lock = threading.RLock()
        reference_queue: queue.Queue[object] | None = None
        reference_thread: threading.Thread | None = None
        reference_sentinel = object()
        reference_buffer: list[str] = []
        reference_error: list[BaseException] = []
        reference_stop = threading.Event()
        reference_batch_size = 0
        if not self.config.without_references:
            if self.reference_crawler is None:
                self.reference_crawler = self._create_reference_crawler()
            reference_batch_size = self._effective_reference_batch_size()
            reference_queue = queue.Queue(maxsize=self.config.reference_queue_size)
            reference_thread = threading.Thread(
                target=self._reference_consumer,
                args=(
                    reference_queue,
                    reference_sentinel,
                    manifest,
                    manifest_lock,
                    reference_error,
                    reference_stop,
                ),
                name="pipeline-reference-consumer",
                daemon=True,
            )
            reference_thread.start()

        if self._source_prefetch_enabled():
            logger.info(
                "Source prefetch enabled: queue_size=%d downloader_workers=1",
                self.config.source_prefetch_queue_size,
            )

        processed = int(self.state.snapshot().get("last_source_checkpoint", 0))
        last_source_checkpoint = processed
        try:
            for source_batch in _chunks(
                self._iter_found_metadata(), self.config.source_batch_size
            ):
                self._check_resources()
                logger.info(
                    "Source batch started: %d discovered papers",
                    len(source_batch),
                )
                self.max_source_batch_seen = max(
                    self.max_source_batch_seen, len(source_batch)
                )
                with manifest_lock:
                    pending = [
                        metadata
                        for metadata in source_batch
                        if not self._stage_is_terminal(
                            manifest.get(str(metadata.get("arxiv_id")), {}),
                            "source",
                            self._SOURCE_TERMINAL,
                        )
                    ]
                    ready_for_reference = [
                        str(metadata["arxiv_id"])
                        for metadata in source_batch
                        if self._stage_is_terminal(
                            manifest.get(str(metadata["arxiv_id"]), {}),
                            "source",
                            self._SOURCE_TERMINAL,
                        )
                        and manifest.get(str(metadata["arxiv_id"]), {}).get("reference")
                        not in self._REFERENCE_TERMINAL
                    ]
                prefetch_enabled = self._source_prefetch_enabled()
                prefetch_pending = {str(metadata["arxiv_id"]) for metadata in pending}
                if prefetch_enabled:
                    self.state.update(
                        source_prefetch_pending_ids=sorted(prefetch_pending)
                    )
                if reference_queue is not None:
                    for arxiv_id in ready_for_reference:
                        self._buffer_reference(
                            reference_queue,
                            reference_buffer,
                            arxiv_id,
                            reference_batch_size,
                            reference_stop,
                            reference_error,
                        )
                pause_requested: str | None = None
                source_results = (
                    self._source_results_prefetched(pending)
                    if prefetch_enabled
                    else self._source_results(pending)
                )
                for record in source_results:
                    self._record_source_duration_stats(record)
                    if self.config.without_references:
                        record["reference"] = {
                            "status": "skipped",
                            "retryable": False,
                        }
                        record["overall_status"] = self._overall_status(record)
                    self.manifest_journal.append(record)
                    arxiv_id = str(record["arxiv_id"])
                    with manifest_lock:
                        entry = manifest.setdefault(arxiv_id, {})
                        for stage in (
                            "source",
                            "processing",
                            "metadata",
                            "reference",
                        ):
                            value = record.get(stage)
                            if isinstance(value, Mapping) and isinstance(
                                value.get("status"), str
                            ):
                                entry[stage] = str(value["status"])
                                if "retryable" in value:
                                    entry[f"{stage}_retryable"] = str(
                                        _stage_value_is_retryable(stage, value)
                                    )
                    processed += 1
                    prefetch_pending.discard(arxiv_id)
                    if reference_queue is not None:
                        with manifest_lock:
                            reference_status = manifest.get(arxiv_id, {}).get(
                                "reference"
                            )
                        if reference_status not in self._REFERENCE_TERMINAL:
                            self._buffer_reference(
                                reference_queue,
                                reference_buffer,
                                arxiv_id,
                                reference_batch_size,
                                reference_stop,
                                reference_error,
                            )
                    if (
                        processed - last_source_checkpoint
                        >= self.config.checkpoint_every
                    ):
                        self.state.update(last_source_checkpoint=processed)
                        if prefetch_enabled:
                            self.state.update(
                                source_prefetch_pending_ids=sorted(prefetch_pending)
                            )
                        self._checkpoint_summary("running")
                        last_source_checkpoint = processed
                    if record.get("_pause_reason"):
                        pause_requested = str(record["_pause_reason"])

                if self.config.without_references:
                    with manifest_lock:
                        skipped_ids = [
                            str(metadata["arxiv_id"])
                            for metadata in source_batch
                            if manifest.get(str(metadata["arxiv_id"]), {}).get(
                                "reference"
                            )
                            not in self._REFERENCE_TERMINAL
                        ]
                    for arxiv_id in skipped_ids:
                        update = {
                            "arxiv_id": arxiv_id,
                            "reference": {"status": "skipped", "retryable": False},
                            "overall_status": manifest.get(arxiv_id, {}).get(
                                "source", "failed"
                            ),
                            "updated_at_utc": utc_now(),
                        }
                        self.manifest_journal.append(update)
                        with manifest_lock:
                            manifest.setdefault(arxiv_id, {})["reference"] = "skipped"
                if last_source_checkpoint != processed:
                    self.state.update(last_source_checkpoint=processed)
                    self._checkpoint_summary("running")
                    last_source_checkpoint = processed
                if prefetch_enabled:
                    self.state.update(source_prefetch_pending_ids=[])
                if pause_requested is not None:
                    raise RateLimitPhasePaused(pause_requested)
                logger.info("Source batch completed: processed=%d", processed)

            self._record_stage_timing("source", phase_started)
            source_timing_recorded = True
            logger.info(
                "event=source_stage_end elapsed_seconds=%.4f processed_papers=%d",
                max(0.0, self.monotonic() - phase_started),
                processed,
            )

            if reference_queue is not None:
                if reference_buffer:
                    self._queue_reference_batch(
                        reference_queue,
                        list(reference_buffer),
                        reference_stop,
                        reference_error,
                    )
                    reference_buffer.clear()
                if not reference_stop.is_set():
                    reference_queue.put(reference_sentinel)
                reference_queue.join()
                if reference_thread is not None:
                    reference_thread.join(timeout=5.0)
                if reference_error:
                    raise reference_error[0]
        finally:
            if not source_timing_recorded:
                self._record_stage_timing("source", phase_started)
            if (
                reference_queue is not None
                and reference_thread is not None
                and reference_thread.is_alive()
            ):
                while reference_thread.is_alive() and not reference_stop.is_set():
                    try:
                        reference_queue.put(reference_sentinel, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                reference_thread.join(timeout=2.0)
            self._close_reference_crawler()

    def _source_results(
        self, metadata_batch: list[Mapping[str, object]]
    ) -> Iterator[dict[str, Any]]:
        """Yield each source result as soon as it completes."""
        if self.config.download_workers == 1:
            for metadata in metadata_batch:
                yield self._process_source(metadata)
            return
        if not metadata_batch:
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.config.download_workers, len(metadata_batch))
        ) as executor:
            futures = [
                executor.submit(self._process_source, metadata)
                for metadata in metadata_batch
            ]
            for future in concurrent.futures.as_completed(futures):
                yield future.result()

    def _source_prefetch_enabled(self) -> bool:
        """Return whether serialized download/processing prefetch is enabled."""
        return self.config.download_workers == 1 and callable(
            getattr(self.source_processor, "prefetch", None)
        ) and callable(getattr(self.source_processor, "process_prefetched", None))

    @staticmethod
    def _ensure_source_timing(
        fragment: Mapping[str, object],
        elapsed_seconds: float,
    ) -> dict[str, object]:
        """Add paper-level timing aliases for third-party source adapters."""
        result = dict(fragment)
        raw_timings = result.get("timings")
        timings = dict(raw_timings) if isinstance(raw_timings, Mapping) else {}
        source_seconds = timings.get("source_seconds", timings.get("total_seconds"))
        if not isinstance(source_seconds, (int, float)):
            source_seconds = elapsed_seconds
        timings.setdefault("source_seconds", max(0.0, float(source_seconds)))
        timings.setdefault("total_seconds", max(0.0, float(source_seconds)))
        result["timings"] = timings
        return result

    def _source_results_prefetched(
        self, metadata_batch: list[Mapping[str, object]]
    ) -> Iterator[dict[str, Any]]:
        """Overlap one serialized downloader with the Processor consumer.

        The producer only places :class:`PrefetchedSource` objects in a bounded
        queue.  Source bytes stay on disk, so queue capacity controls temporary
        disk usage rather than RAM.  A producer exception is re-raised after
        already queued papers are journaled, allowing the normal pipeline
        checkpoint/resume path to handle the unresolved paper.
        """
        if not metadata_batch:
            return
        assert self.source_processor is not None
        work_queue: queue.Queue[object] = queue.Queue(
            maxsize=self.config.source_prefetch_queue_size
        )
        sentinel = object()
        stop_event = threading.Event()
        producer_errors: list[BaseException] = []

        def enqueue(item: object) -> bool:
            while not stop_event.is_set():
                try:
                    work_queue.put(item, timeout=0.2)
                    self.source_prefetch_queue_observed_max = max(
                        self.source_prefetch_queue_observed_max,
                        work_queue.qsize(),
                    )
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for metadata in metadata_batch:
                    self._check_resources()
                    prefetch = self.source_processor.prefetch  # type: ignore[attr-defined]
                    prefetched = prefetch(metadata)
                    if not enqueue(prefetched):
                        return
            except BaseException as exc:  # noqa: BLE001 - re-raised by consumer.
                producer_errors.append(exc)
            finally:
                enqueue(sentinel)

        producer = threading.Thread(
            target=produce,
            name="pipeline-source-prefetch",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = work_queue.get()
                try:
                    if item is sentinel:
                        break
                    yield self._process_prefetched_source(item)
                finally:
                    work_queue.task_done()
        finally:
            stop_event.set()
            producer.join(timeout=2.0)
        if producer_errors:
            error = producer_errors[0]
            if isinstance(error, SourceRateLimitPaused):
                raise RateLimitPhasePaused(str(error)) from error
            raise error

    def _process_prefetched_source(self, prefetched: object) -> dict[str, Any]:
        assert self.source_processor is not None
        if not isinstance(prefetched, PrefetchedSource):
            raise TypeError("source prefetch adapter returned an invalid item")
        arxiv_id = normalize_arxiv_id(prefetched.metadata.get("arxiv_id"))
        started = self.monotonic()
        try:
            process_prefetched = self.source_processor.process_prefetched  # type: ignore[attr-defined]
            fragment = dict(process_prefetched(prefetched))
        except SourceRateLimitPaused as exc:
            fragment = {
                "source": {
                    "status": "retryable_failed",
                    "retryable": True,
                    "error_type": "http_429",
                    "error": str(exc),
                },
                "processing": {"status": "not_started"},
                "metadata": {"status": "not_started"},
                "_pause_reason": str(exc),
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one paper failure.
            fragment = {
                "source": {
                    "status": "retryable_failed",
                    "retryable": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "processing": {"status": "not_started"},
                "metadata": {"status": "failed"},
            }
        fragment = self._ensure_source_timing(
            fragment,
            max(0.0, self.monotonic() - started),
        )
        record: dict[str, Any] = {"arxiv_id": arxiv_id, **fragment}
        record.setdefault("source", {"status": "failed"})
        record.setdefault("processing", {"status": "not_started"})
        record.setdefault("metadata", {"status": "not_started"})
        record.setdefault("reference", {"status": "pending"})
        record["overall_status"] = self._overall_status(record)
        record["updated_at_utc"] = utc_now()
        source = record.get("source")
        source_status = (
            str(source.get("status", "failed"))
            if isinstance(source, Mapping)
            else "failed"
        )
        self._on_source_progress(
            {
                "event": "paper_completed",
                "arxiv_id": arxiv_id,
                "status": source_status,
                "source_seconds": record["timings"].get("source_seconds", 0.0),
                "message": "paper source completed",
            }
        )
        return record

    def _process_source(self, metadata: Mapping[str, object]) -> dict[str, Any]:
        assert self.source_processor is not None
        arxiv_id = normalize_arxiv_id(metadata.get("arxiv_id"))
        versions = metadata.get("versions")
        total_versions = len(versions) if isinstance(versions, list) else 0
        self._on_source_progress(
            {
                "event": "paper_started",
                "arxiv_id": arxiv_id,
                "total": total_versions,
                "status": "processing",
                "message": "starting paper source",
            }
        )
        started = self.monotonic()
        try:
            fragment = dict(self.source_processor.process(metadata))
        except SourceRateLimitPaused as exc:
            fragment = {
                "source": {
                    "status": "retryable_failed",
                    "retryable": True,
                    "error_type": "http_429",
                    "error": str(exc),
                },
                "processing": {"status": "not_started"},
                "metadata": {"status": "not_started"},
                "_pause_reason": str(exc),
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one paper failure.
            fragment = {
                "source": {
                    "status": "retryable_failed",
                    "retryable": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "processing": {"status": "not_started"},
                "metadata": {"status": "failed"},
            }
        fragment = self._ensure_source_timing(
            fragment,
            max(0.0, self.monotonic() - started),
        )
        record: dict[str, Any] = {"arxiv_id": arxiv_id, **fragment}
        record.setdefault("source", {"status": "failed"})
        record.setdefault("processing", {"status": "not_started"})
        record.setdefault("metadata", {"status": "not_started"})
        record.setdefault("reference", {"status": "pending"})
        record["overall_status"] = self._overall_status(record)
        record["updated_at_utc"] = utc_now()
        source = record.get("source")
        source_status = (
            str(source.get("status", "failed"))
            if isinstance(source, Mapping)
            else "failed"
        )
        self._on_source_progress(
            {
                "event": "paper_completed",
                "arxiv_id": arxiv_id,
                "status": source_status,
                "source_seconds": record["timings"].get("source_seconds", 0.0),
                "message": "paper source completed",
            }
        )
        return record

    @staticmethod
    def _overall_status(record: Mapping[str, object]) -> str:
        source = record.get("source")
        reference = record.get("reference")
        source_status = source.get("status") if isinstance(source, Mapping) else None
        reference_status = (
            reference.get("status") if isinstance(reference, Mapping) else None
        )
        if source_status not in {"success", "success_with_no_tex"}:
            return str(source_status or "failed")
        if reference_status in {None, "pending"}:
            return "pending_reference"
        if reference_status in {"success", "not_found", "skipped"}:
            return str(source_status)
        return str(reference_status)

    @staticmethod
    def _stage_is_terminal(
        statuses: Mapping[str, str],
        stage: str,
        terminal_statuses: frozenset[str],
    ) -> bool:
        status = statuses.get(stage)
        if status in terminal_statuses:
            return True
        return (
            status == "partial_success"
            and statuses.get(f"{stage}_retryable") == "False"
        )

    def _effective_reference_batch_size(self) -> int:
        if self.config.reference_batch_size is not None:
            return self.config.reference_batch_size
        crawler = self.reference_crawler
        workers = max(1, int(getattr(crawler, "max_workers", 1)))
        batch_size = max(1, int(getattr(crawler, "batch_size", 25)))
        return workers * batch_size

    def _buffer_reference(
        self,
        work_queue: queue.Queue[object],
        buffer: list[str],
        arxiv_id: str,
        batch_size: int,
        stop_event: threading.Event,
        errors: list[BaseException],
    ) -> None:
        buffer.append(arxiv_id)
        if len(buffer) < batch_size:
            return
        self._queue_reference_batch(
            work_queue,
            buffer[:batch_size],
            stop_event,
            errors,
        )
        del buffer[:batch_size]

    def _queue_reference_batch(
        self,
        work_queue: queue.Queue[object],
        arxiv_ids: list[str],
        stop_event: threading.Event,
        errors: list[BaseException],
    ) -> None:
        while not stop_event.is_set():
            try:
                work_queue.put(list(arxiv_ids), timeout=0.2)
                break
            except queue.Full:
                continue
        else:
            if errors:
                raise errors[0]
            raise RateLimitPhasePaused("reference consumer stopped")
        self.reference_queue_observed_max = max(
            self.reference_queue_observed_max, work_queue.qsize()
        )

    def _reference_consumer(
        self,
        work_queue: queue.Queue[object],
        sentinel: object,
        manifest: dict[str, dict[str, str]],
        manifest_lock: threading.RLock,
        errors: list[BaseException],
        stop_event: threading.Event,
    ) -> None:
        assert self.state is not None and self.manifest_journal is not None
        assert self.reference_crawler is not None
        stage_started: float | None = None
        crawler_seconds = 0.0
        queue_wait_seconds = 0.0
        completed = int(self.state.snapshot().get("last_reference_checkpoint", 0))
        consecutive_rate_limited_batches = 0
        try:
            while True:
                wait_started = self.monotonic()
                item = work_queue.get()
                wait_elapsed = max(0.0, self.monotonic() - wait_started)
                try:
                    if item is sentinel:
                        if stage_started is not None:
                            queue_wait_seconds += wait_elapsed
                        break
                    arxiv_ids = list(item)  # type: ignore[arg-type]
                    with self._reference_progress_lock:
                        self._reference_progress.update(
                            {
                                "current_batch": 0,
                                "current_batch_total": len(arxiv_ids),
                                "queue_size": work_queue.qsize(),
                            }
                        )
                    if stage_started is None:
                        stage_started = self.monotonic()
                    else:
                        queue_wait_seconds += wait_elapsed
                    crawler_started = self.monotonic()
                    summary: Mapping[str, object] | None = None
                    try:
                        logger.info(
                            "Reference batch started: %d papers",
                            len(arxiv_ids),
                        )
                        summary = self.reference_crawler.process_batch(
                            arxiv_ids, force_refresh=False
                        )
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:  # noqa: BLE001 - isolate one batch.
                        results = {
                            arxiv_id: {
                                "status": "retryable_failed",
                                "retryable": True,
                                "error": f"{type(exc).__name__}: {exc}",
                                "references_count": 0,
                            }
                            for arxiv_id in arxiv_ids
                        }
                    finally:
                        crawler_seconds += max(0.0, self.monotonic() - crawler_started)
                    if summary is not None:
                        results = self._reference_results(arxiv_ids, summary)

                    for arxiv_id in arxiv_ids:
                        reference = dict(results[arxiv_id])
                        with manifest_lock:
                            source_status = manifest.get(arxiv_id, {}).get(
                                "source", "failed"
                            )
                        update = {
                            "arxiv_id": arxiv_id,
                            "reference": reference,
                            "overall_status": self._overall_status(
                                {
                                    "source": {"status": source_status},
                                    "reference": reference,
                                }
                            ),
                            "updated_at_utc": utc_now(),
                        }
                        self.manifest_journal.append(update)
                        with manifest_lock:
                            entry = manifest.setdefault(arxiv_id, {})
                            entry["reference"] = str(reference["status"])
                            entry["overall"] = str(update["overall_status"])
                        completed += 1
                        with self._reference_progress_lock:
                            self._reference_progress["completed_papers"] = completed
                            self._reference_progress["references"] = (
                                int(self._reference_progress.get("references", 0))
                                + int(reference.get("references_count", 0) or 0)
                            )
                    self.state.update(last_reference_checkpoint=completed)
                    self._checkpoint_summary("running")
                    with self._reference_progress_lock:
                        self._reference_progress["batches_completed"] = (
                            int(self._reference_progress.get("batches_completed", 0)) + 1
                        )
                        self._reference_progress["current_batch"] = len(arxiv_ids)
                        self._reference_progress["current_batch_total"] = len(arxiv_ids)
                        self._reference_progress["queue_size"] = work_queue.qsize()
                    logger.info(
                        "Reference batch completed: processed=%d",
                        completed,
                    )
                    rate_limited = all(
                        result.get("status") == "retryable_failed"
                        and any(
                            token in str(result.get("error", "")).lower()
                            for token in ("429", "rate limit")
                        )
                        for result in results.values()
                    )
                    if rate_limited:
                        consecutive_rate_limited_batches += 1
                    else:
                        consecutive_rate_limited_batches = 0
                    if (
                        consecutive_rate_limited_batches
                        >= self.config.retry_policy.circuit_breaker_threshold
                    ):
                        raise RateLimitPhasePaused(
                            "reference crawler rate-limit circuit opened"
                        )
                finally:
                    work_queue.task_done()
        except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
            errors.append(exc)
            stop_event.set()
            while True:
                try:
                    abandoned = work_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    work_queue.task_done()
                    if abandoned is sentinel:
                        break
        finally:
            if stage_started is not None:
                stage_elapsed = max(0.0, self.monotonic() - stage_started)
                self._add_timing("reference_time_seconds", crawler_seconds)
                self._add_timing("reference_stage_wall_seconds", stage_elapsed)
                self._add_timing("reference_queue_wait_seconds", queue_wait_seconds)
                if self.benchmark is not None:
                    self.benchmark.add_stage_wall_time("reference", stage_elapsed)
                logger.info(
                    "event=reference_stage_end wall_seconds=%.4f crawler_work_seconds=%.4f queue_wait_seconds=%.4f completed_papers=%d",
                    stage_elapsed,
                    crawler_seconds,
                    queue_wait_seconds,
                    completed,
                )

    def _reference_results(
        self,
        arxiv_ids: list[str],
        summary: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        supplied = summary.get("results")
        results: dict[str, dict[str, object]] = {}
        if isinstance(supplied, Mapping):
            for arxiv_id in arxiv_ids:
                value = supplied.get(arxiv_id)
                if isinstance(value, Mapping):
                    results[arxiv_id] = self._normalize_reference_result(value)
        for arxiv_id in arxiv_ids:
            if arxiv_id in results:
                continue
            value = self._read_reference_file(arxiv_id)
            results[arxiv_id] = self._normalize_reference_result(value)
        return results

    @staticmethod
    def _normalize_reference_result(
        value: Mapping[str, object],
    ) -> dict[str, object]:
        raw_status = str(value.get("status", "retryable_failure"))
        status = {
            "retryable_failure": "retryable_failed",
            "permanent_failure": "permanent_failed",
        }.get(raw_status, raw_status)
        retryable = bool(value.get("retryable", status == "retryable_failed"))
        result: dict[str, object] = {
            "status": status,
            "retryable": retryable,
            "references_count": int(value.get("references_count", 0) or 0),
        }
        error = value.get("error") or value.get("error_message")
        if error:
            result["error"] = str(error)
        attempts = value.get("attempt") or value.get("attempts")
        if isinstance(attempts, int):
            result["attempts"] = attempts
        return result

    def _read_reference_file(self, arxiv_id: str) -> Mapping[str, object]:
        crawler = self.reference_crawler
        folder_method = getattr(crawler, "_get_folder_name", None)
        if callable(folder_method):
            folder_name = str(folder_method(arxiv_id))
        else:
            folder_name = arxiv_id.replace("/", "-").replace(".", "-")
        path = self.corpus_dir / folder_name / "references.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        return {
            "status": "retryable_failure",
            "retryable": True,
            "error_message": "reference result was not persisted",
            "references_count": 0,
        }

    def _create_reference_crawler(self) -> ReferenceBatchCrawler:
        from src.crawling.config import settings
        from src.crawling.reference_crawler import ReferenceCrawler

        if self.config.debug_reference_metrics:
            log_dir = self.run_dir.parent / "reference-debug"
        else:
            self._reference_temp_dir = tempfile.TemporaryDirectory(
                prefix="arxiv-reference-"
            )
            log_dir = Path(self._reference_temp_dir.name)
        crawler = ReferenceCrawler(
            corpus_dir=str(self.corpus_dir),
            log_dir=str(log_dir),
            max_retries=self.config.retry_policy.max_retries,
            batch_size=25,
            max_workers=self.config.reference_workers,
        )
        key_count = len(getattr(getattr(crawler, "key_pool", None), "api_keys", []))
        if key_count:
            requested = self.config.reference_workers or key_count
            crawler.max_workers = min(requested, key_count)
        reference_logger = logging.getLogger("ReferenceCrawler")
        app_logger = logging.getLogger("arxiv_project")
        common_log_path = (settings.log_dir / "arxiv.log").resolve()

        # Keep the crawler's dedicated log while also forwarding its records to
        # the common application log. Do not attach the common console handler:
        # ReferenceCrawler already owns one and would otherwise print twice.
        crawler_file_handler = getattr(crawler, "file_handler", None)
        if (
            isinstance(crawler_file_handler, logging.Handler)
            and crawler_file_handler in reference_logger.handlers
        ):
            reference_logger.removeHandler(crawler_file_handler)
            crawler_file_handler.close()

        common_file_handler = next(
            (
                handler
                for handler in app_logger.handlers
                if isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == common_log_path
            ),
            None,
        )
        if common_file_handler is not None:
            if common_file_handler not in reference_logger.handlers:
                reference_logger.addHandler(common_file_handler)
            if common_file_handler not in self._reference_shared_handlers:
                self._reference_shared_handlers.append(common_file_handler)
        try:
            settings.log_dir.mkdir(parents=True, exist_ok=True)
            reference_handler = logging.FileHandler(
                settings.log_dir / "reference_crawler.log",
                encoding="utf-8",
            )
            reference_handler.setLevel(logging.INFO)
            reference_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            logging.getLogger("ReferenceCrawler").addHandler(reference_handler)
            self._reference_log_handler = reference_handler
        except OSError as exc:
            logger.warning("Could not persist ReferenceCrawler log: %s", exc)
        self._instrument_reference_429(crawler)
        return crawler

    def _instrument_reference_429(self, crawler: object) -> None:
        key_pool = getattr(crawler, "key_pool", None)
        original = getattr(key_pool, "mark_cooldown", None)
        if not callable(original):
            return
        metric_lock = threading.Lock()

        def counted(key: str, retry_after: str | None = None) -> object:
            with metric_lock:
                self._add_metric("reference_http_429_count", 1)
            return original(key, retry_after)

        key_pool.mark_cooldown = counted

    def _close_reference_crawler(self) -> None:
        crawler = self.reference_crawler
        try:
            close = getattr(crawler, "close", None)
            if callable(close):
                close()
        finally:
            if self._reference_log_handler is not None:
                reference_logger = logging.getLogger("ReferenceCrawler")
                reference_logger.removeHandler(self._reference_log_handler)
                self._reference_log_handler.close()
                self._reference_log_handler = None
            if self._reference_shared_handlers:
                reference_logger = logging.getLogger("ReferenceCrawler")
                for handler in self._reference_shared_handlers:
                    reference_logger.removeHandler(handler)
                self._reference_shared_handlers.clear()
            if self._reference_temp_dir is not None:
                self._reference_temp_dir.cleanup()
                self._reference_temp_dir = None

    def _add_metric(self, name: str, value: float) -> None:
        with self._metric_lock:
            self._pending_metrics[name] = self._pending_metrics.get(name, 0) + value

    def _on_source_progress(self, event: Mapping[str, object]) -> None:
        """Update paper/version progress without blocking source work on disk I/O."""
        arxiv_id = str(event.get("arxiv_id", ""))
        worker = threading.current_thread().name
        event_name = str(event.get("event", "unknown"))
        with self._source_progress_lock:
            worker_order = self._source_progress["worker_order"]
            if isinstance(worker_order, list) and worker not in worker_order:
                worker_order.append(worker)
            worker_progress = self._source_progress["worker_progress"]
            if not isinstance(worker_progress, dict):
                worker_progress = {}
                self._source_progress["worker_progress"] = worker_progress
            current = worker_progress.setdefault(worker, {})
            if not isinstance(current, dict):
                current = {}
                worker_progress[worker] = current
            current.update(
                {
                    "paper": arxiv_id,
                    "status": event.get("status", "running"),
                    "message": event.get("message", event_name),
                }
            )
            for key in ("versioned_id", "current", "total"):
                if key in event:
                    current[key] = event[key]

            version_status = str(event.get("status", ""))
            version_succeeded = version_status in {
                "ok",
                "no_tex",
                "success",
                "success_with_no_tex",
                "skipped",
            }
            if event_name == "version_skipped" or (
                event_name == "version_processed" and version_succeeded
            ):
                completed_versions = self._source_progress["versions_completed"]
                self._source_progress["versions_completed"] = (
                    completed_versions + 1
                    if isinstance(completed_versions, int)
                    else 1
                )
            elif event_name in {"version_failed", "version_processed"}:
                failed_versions = self._source_progress["versions_failed"]
                self._source_progress["versions_failed"] = (
                    failed_versions + 1
                    if isinstance(failed_versions, int)
                    else 1
                )
            elif event_name == "paper_completed":
                completed_papers = self._source_progress["completed_papers"]
                self._source_progress["completed_papers"] = (
                    completed_papers + 1
                    if isinstance(completed_papers, int)
                    else 1
                )
                status = str(event.get("status", "failed"))
                if status in {"success", "success_with_no_tex"}:
                    key = "papers_success"
                elif status in {"partial_success", "retryable_failed"}:
                    key = "papers_partial"
                else:
                    key = "papers_failed"
                count = self._source_progress[key]
                self._source_progress[key] = (
                    count + 1 if isinstance(count, int) else 1
                )
                current.update({"status": "idle", "message": "paper completed"})
                logger.info(
                    "event=source_id_complete paper=%s status=%s worker=%s source_seconds=%.4f",
                    arxiv_id,
                    status,
                    worker,
                    float(event.get("source_seconds", 0.0) or 0.0),
                )

    def _record_source_duration_stats(self, record: Mapping[str, object]) -> None:
        timings = record.get("timings")
        if isinstance(timings, Mapping):
            source_seconds = timings.get("source_seconds", timings.get("total_seconds"))
            if isinstance(source_seconds, (int, float)):
                self._source_id_duration_stats.update(source_seconds)
        source = record.get("source")
        versions = source.get("versions") if isinstance(source, Mapping) else None
        if not isinstance(versions, list):
            return
        for version in versions:
            if not isinstance(version, Mapping):
                continue
            version_timings = version.get("timings")
            if not isinstance(version_timings, Mapping):
                continue
            download_seconds = version_timings.get("download_seconds")
            processing_seconds = version_timings.get("processing_seconds")
            if isinstance(download_seconds, (int, float)):
                self._version_download_duration_stats.update(download_seconds)
            if isinstance(processing_seconds, (int, float)):
                self._version_processing_duration_stats.update(processing_seconds)

    def _source_progress_snapshot(self, total_papers: int) -> dict[str, object]:
        with self._source_progress_lock:
            snapshot = dict(self._source_progress)
            snapshot["total_papers"] = total_papers
            worker_progress = snapshot.get("worker_progress")
            snapshot["worker_progress"] = {
                worker: dict(value)
                for worker, value in worker_progress.items()
                if isinstance(value, Mapping)
            } if isinstance(worker_progress, Mapping) else {}
            order = snapshot.get("worker_order")
            snapshot["worker_order"] = list(order) if isinstance(order, list) else []
            return snapshot

    def _progress_snapshot(self) -> dict[str, object]:
        """Build the low-cost aggregate state consumed by ProgressMonitor."""
        state = self.state.snapshot() if self.state is not None else {}
        timings = state.get("timings") if isinstance(state, Mapping) else {}
        metrics = state.get("metrics") if isinstance(state, Mapping) else {}
        timings = timings if isinstance(timings, Mapping) else {}
        metrics = metrics if isinstance(metrics, Mapping) else {}
        with self._discovery_progress_lock:
            discovery = dict(self._discovery_progress)
        discovery_status = str(discovery.get("status", state.get("status", "pending")))
        if bool(state.get("discovery_complete")):
            discovery_status = "completed"
        with self._source_progress_lock:
            source = copy.deepcopy(self._source_progress)
        with self._reference_progress_lock:
            reference = dict(self._reference_progress)
        source["http_429"] = metrics.get("downloader_http_429_count", 0)
        reference["http_429"] = metrics.get("reference_http_429_count", 0)
        elapsed = max(0.0, self.monotonic() - self._session_started)
        return {
            "elapsed_seconds": elapsed,
            "discovery": {
                "status": discovery_status,
                "found": discovery.get("found", 0),
                "total": self.loaded_input.unique_count,
                "batch": discovery.get("batch", state.get("last_discovery_batch", 0)),
                "http_429": metrics.get("discovery_http_429_count", 0),
                "elapsed_seconds": timings.get("discovery_time_seconds", 0.0),
            },
            "source": source,
            "reference": reference,
        }

    def _flush_metrics(self) -> None:
        assert self.state is not None
        with self._metric_lock:
            if not self._pending_metrics:
                return
            pending = dict(self._pending_metrics)

            def update(state: dict[str, Any]) -> None:
                metrics = state.setdefault("metrics", self._empty_metrics())
                for name, value in pending.items():
                    metrics[name] = metrics.get(name, 0) + value

            self.state.mutate(update)
            self._pending_metrics.clear()

    def _checkpoint_summary(
        self,
        status: str,
        *,
        force: bool = False,
        phase: str | None = None,
    ) -> None:
        """Persist a lightweight checkpoint and an occasional full summary.

        ``run_state.json`` remains current after every caller invocation via
        ``_flush_metrics``. Discovery summaries are intentionally throttled:
        reading all JSONL files and recursively sizing the corpus/cache is
        much more expensive than updating the atomic state file.
        """
        self._flush_metrics()
        if phase == "discovery" and not force:
            self._discovery_summary_batches += 1
            if self._discovery_summary_batches < self.config.checkpoint_every:
                return
            self._discovery_summary_batches = 0
        if self.benchmark is not None:
            self.benchmark.sample_directory_sizes()
        self._write_summary(status)
        if phase == "discovery":
            self._discovery_summary_batches = 0

    def _finish(self, status: str) -> dict[str, Any]:
        assert self.state is not None
        session_elapsed = max(0.0, self.monotonic() - self._session_started)
        self._add_timing("wall_time_seconds", session_elapsed)
        self._flush_metrics()
        if self.progress_monitor is not None:
            self.progress_monitor.stop()
            self.progress_monitor = None
        if self.discovery_journal is not None:
            self.discovery_journal.sync()
        if self.manifest_journal is not None:
            self.manifest_journal.sync()
        if self.benchmark is not None:
            self.benchmark.stop_sampling()
        summary = self._write_summary(status)
        logger.info(
            "Run summary persisted: status=%s path=%s", status, self.summary_path
        )
        return summary

    def _write_summary(self, status: str) -> dict[str, Any]:
        assert self.state is not None
        with self._summary_lock:
            state = self.state.snapshot()
            discovery_statuses = _latest_discovery_statuses(self.discovery_path)
            discovery_version_counts = _latest_discovery_version_counts(
                self.discovery_path
            )
            manifest = _manifest_index(self.manifest_path)
            discovery_counts: dict[str, int] = {}
            for value in discovery_statuses.values():
                discovery_counts[value] = discovery_counts.get(value, 0) + 1
            found_count = discovery_counts.get("found", 0)
            total_versions = sum(discovery_version_counts.values())
            average_versions_per_id = (
                total_versions / self.loaded_input.unique_count
                if self.loaded_input.unique_count
                else 0.0
            )
            average_versions_per_found_id = (
                total_versions / found_count if found_count else 0.0
            )
            source_success = sum(
                value.get("source") in {"success", "success_with_no_tex"}
                for value in manifest.values()
            )
            reference_success = sum(
                value.get("reference") == "success" for value in manifest.values()
            )
            timings = dict(state.get("timings") or {})
            metrics = dict(state.get("metrics") or {})
            source_progress = self._source_progress_snapshot(found_count)
            source_time_seconds = float(timings.get("source_time_seconds", 0.0))
            completed_source_papers = source_progress.get("completed_papers", 0)
            completed_source_papers = (
                completed_source_papers
                if isinstance(completed_source_papers, (int, float))
                else 0
            )
            completed_source_versions = source_progress.get("versions_completed", 0)
            completed_source_versions = (
                completed_source_versions
                if isinstance(completed_source_versions, (int, float))
                else 0
            )
            failed_source_versions = source_progress.get("versions_failed", 0)
            failed_source_versions = (
                failed_source_versions
                if isinstance(failed_source_versions, (int, float))
                else 0
            )
            cache_hits = metrics.get("downloader_cache_hit_count", 0)
            cache_misses = metrics.get("downloader_cache_miss_count", 0)
            cache_attempts = cache_hits + cache_misses
            source_limiter = getattr(self.source_processor, "limiter", None)
            limiter_wait_seconds = getattr(source_limiter, "wait_seconds", 0.0)
            current_request_interval = getattr(
                source_limiter,
                "current_interval",
                self.config.source_request_interval,
            )
            if current_request_interval is None:
                from src.crawling.config import settings

                current_request_interval = settings.arxiv_api_delay
            summary: dict[str, Any] = {
                "run_id": state.get("run_id"),
                "status": status,
                "phase": state.get("phase"),
                "input_fingerprint": self.loaded_input.fingerprint,
                "candidate_id_count": self.loaded_input.candidate_count,
                "unique_id_count": self.loaded_input.unique_count,
                "found_limit": self.loaded_input.found_limit,
                "total_papers": discovery_counts.get("found", 0),
                "found_count": found_count,
                "found_limit_reached": (
                    self.loaded_input.found_limit is not None
                    and found_count
                    >= self.loaded_input.found_limit
                ),
                "total_versions": total_versions,
                "average_versions_per_id": average_versions_per_id,
                "average_versions_per_found_id": average_versions_per_found_id,
                "not_found_count": discovery_counts.get("not_found", 0),
                "retryable_failed_count": (
                    discovery_counts.get("retryable_failed", 0)
                    + discovery_counts.get("rate_limited", 0)
                ),
                "permanent_failed_count": discovery_counts.get("permanent_failed", 0),
                "source_success_count": source_success,
                "reference_success_count": reference_success,
                "source_prefetch_queue_maxsize": self.source_prefetch_queue_maxsize,
                "source_prefetch_queue_observed_max": (
                    self.source_prefetch_queue_observed_max
                ),
                "discovery_time_seconds": float(
                    timings.get("discovery_time_seconds", 0.0)
                ),
                "discovery_wall_seconds": float(
                    timings.get("discovery_time_seconds", 0.0)
                ),
                "source_time_seconds": source_time_seconds,
                "source_stage_wall_seconds": source_time_seconds,
                "source_progress": source_progress,
                "source_worker_count": self.config.download_workers,
                "completed_papers": int(completed_source_papers),
                "completed_versions": int(completed_source_versions),
                "failed_versions": int(failed_source_versions),
                "source_request_interval_seconds": (
                    self.config.source_request_interval
                    if self.config.source_request_interval is not None
                    else current_request_interval
                ),
                "source_current_request_interval_seconds": current_request_interval,
                "downloader_limiter_wait_seconds": limiter_wait_seconds,
                "source_versions_per_hour": (
                    completed_source_versions / source_time_seconds * 3600
                    if source_time_seconds > 0
                    else 0.0
                ),
                "versions_per_hour": (
                    completed_source_versions / source_time_seconds * 3600
                    if source_time_seconds > 0
                    else 0.0
                ),
                "source_version_success_rate": (
                    completed_source_versions
                    / (completed_source_versions + failed_source_versions)
                    if completed_source_versions + failed_source_versions > 0
                    else 0.0
                ),
                "downloader_cache_hit_rate": (
                    cache_hits / cache_attempts
                    if cache_attempts > 0
                    else 0.0
                ),
                "source_papers_per_second": (
                    completed_source_papers / source_time_seconds
                    if source_time_seconds > 0
                    else 0.0
                ),
                "reference_time_seconds": float(
                    timings.get("reference_time_seconds", 0.0)
                ),
                "reference_api_time_seconds": float(
                    timings.get("reference_time_seconds", 0.0)
                ),
                "reference_crawler_work_seconds": float(
                    timings.get("reference_time_seconds", 0.0)
                ),
                "reference_stage_wall_seconds": float(
                    timings.get("reference_stage_wall_seconds", 0.0)
                ),
                "reference_queue_wait_seconds": float(
                    timings.get("reference_queue_wait_seconds", 0.0)
                ),
                "wall_time_seconds": float(timings.get("wall_time_seconds", 0.0)),
                **metrics,
                "failed_ids": _failed_ids_from_indexes(discovery_statuses, manifest),
                "initial_rss_bytes": self.initial_rss_bytes,
                "source_id_duration_stats": (
                    self._source_id_duration_stats.snapshot()
                ),
                "version_download_duration_stats": (
                    self._version_download_duration_stats.snapshot()
                ),
                "version_processing_duration_stats": (
                    self._version_processing_duration_stats.snapshot()
                ),
            }
            if self.benchmark is not None:
                resource = self.benchmark.snapshot(
                    summary,
                    required_paper_count=discovery_counts.get("found", 0),
                )
                summary["peak_rss_mb"] = resource.get("peak_rss_mb")
                summary["average_rss_mb"] = resource.get("average_sampled_rss_mb")
                session_peak = self._first_metric_value(
                    resource.get("disk_peak_used_bytes")
                )
                session_rss_peak = resource.get("peak_rss_mb")
                session_rss_average = resource.get("average_sampled_rss_mb")
                session_rss_samples = int(resource.get("memory_sample_count") or 0)
                if isinstance(session_rss_average, (int, float)):
                    session_rss_sum = float(session_rss_average) * session_rss_samples
                else:
                    session_rss_sum = 0.0
                new_rss_samples = max(
                    0, session_rss_samples - self._rss_samples_accounted
                )
                new_rss_sum = max(0.0, session_rss_sum - self._rss_sum_accounted_mb)

                def update_resources(state_value: dict[str, Any]) -> None:
                    saved = state_value.setdefault("resources", {})
                    if session_peak is not None:
                        saved["disk_peak_used_bytes"] = max(
                            int(saved.get("disk_peak_used_bytes", 0)),
                            session_peak,
                        )
                    if isinstance(session_rss_peak, (int, float)):
                        saved["peak_rss_mb"] = max(
                            float(saved.get("peak_rss_mb", 0.0)),
                            float(session_rss_peak),
                        )
                    if new_rss_samples:
                        saved["rss_sample_count"] = (
                            int(saved.get("rss_sample_count", 0)) + new_rss_samples
                        )
                        saved["rss_sample_sum_mb"] = (
                            float(saved.get("rss_sample_sum_mb", 0.0)) + new_rss_sum
                        )

                resource_state = self.state.mutate(update_resources).get(
                    "resources", {}
                )
                self._rss_samples_accounted = session_rss_samples
                self._rss_sum_accounted_mb = session_rss_sum
                disk_initial = resource_state.get("disk_initial_used_bytes")
                disk_peak = resource_state.get("disk_peak_used_bytes")
                rss_sample_count = int(resource_state.get("rss_sample_count", 0))
                rss_sample_sum = float(resource_state.get("rss_sample_sum_mb", 0.0))
                summary["initial_rss_bytes"] = resource_state.get("initial_rss_bytes")
                summary["peak_rss_mb"] = resource_state.get(
                    "peak_rss_mb", summary["peak_rss_mb"]
                )
                summary["average_rss_mb"] = (
                    round(rss_sample_sum / rss_sample_count, 3)
                    if rss_sample_count
                    else None
                )
                summary["disk_initial_used_bytes"] = disk_initial
                summary["disk_peak_used_bytes"] = disk_peak
                summary["disk_growth_bytes"] = (
                    max(0, int(disk_peak) - int(disk_initial))
                    if isinstance(disk_peak, (int, float))
                    and isinstance(disk_initial, (int, float))
                    else None
                )
            from src.crawling.stats import directory_size_bytes

            summary["final_corpus_size_bytes"] = directory_size_bytes(self.corpus_dir)
            summary["download_cache_size_bytes"] = directory_size_bytes(self.cache_dir)
            summary["final_output_size_bytes"] = directory_size_bytes(self.run_dir)
            for _ in range(3):
                atomic_write_json(self.summary_path, summary)
                measured = directory_size_bytes(self.run_dir)
                if measured == summary["final_output_size_bytes"]:
                    break
                summary["final_output_size_bytes"] = measured
            atomic_write_json(self.summary_path, summary)
            return summary

    @staticmethod
    def _first_metric_value(value: object) -> int | None:
        if isinstance(value, Mapping) and value:
            candidate = next(iter(value.values()))
            if isinstance(candidate, (int, float)):
                return int(candidate)
        return None


def _loaded_from_ids(arxiv_ids: Iterable[str], *, kind: str = "ids") -> LoadedInput:
    candidates = [normalize_arxiv_id(value) for value in arxiv_ids]
    unique = tuple(dict.fromkeys(candidates))
    return LoadedInput(
        kind=kind,
        input_path=None,
        fingerprint=_fingerprint(kind, unique),
        candidate_count=len(candidates),
        unique_count=len(unique),
        ranges_count=0,
        _unique_ids=unique,
    )


def run_pipeline(
    arxiv_ids: Iterable[str],
    reference_adapter: object | None = None,
    *,
    resume_dir: str | Path | None = None,
    keep_results: bool = False,
    checkpoint_every: int = 25,
    fsync_every: int = 10,
    reference_batch_size: int | None = None,
    min_free_bytes: int = 1024**3,
    source_batch_size: int = 50,
    download_workers: int = 10,
    request_interval: float | None = None,
    source_prefetch_queue_size: int = 2,
    reference_queue_size: int = 2,
    max_runtime_hours: float | None = None,
    grace_period_seconds: float = 600,
    input_spec: Mapping[str, object] | None = None,
    input_snapshot_path: str | Path | None = None,
    progress_monitor: bool = True,
    progress_interval: float = 1.0,
) -> dict[str, Any]:
    """Backward-compatible programmatic entry point.

    New callers should construct :class:`Pipeline` directly when injecting
    adapters. ``input_spec`` and ``input_snapshot_path`` remain accepted so old
    notebooks do not fail; canonical input.json is always written by Pipeline.
    """
    del input_spec, input_snapshot_path
    from src.crawling.config import settings

    run_dir = Path(resume_dir) if resume_dir is not None else settings.output_dir
    crawler = getattr(reference_adapter, "crawler", reference_adapter)
    if crawler is not None and not hasattr(crawler, "process_batch"):
        raise TypeError("reference_adapter must expose crawler.process_batch()")
    loaded = _loaded_from_ids(arxiv_ids)
    config = PipelineConfig(
        run_dir=run_dir,
        corpus_dir=settings.data_dir,
        source_batch_size=source_batch_size,
        download_workers=download_workers,
        source_request_interval=request_interval,
        source_prefetch_queue_size=source_prefetch_queue_size,
        reference_batch_size=reference_batch_size,
        reference_queue_size=reference_queue_size,
        checkpoint_every=checkpoint_every,
        fsync_every=fsync_every,
        min_free_gb=min_free_bytes / 1024**3,
        max_runtime_hours=max_runtime_hours,
        grace_period_seconds=grace_period_seconds,
        without_references=crawler is None,
        keep_downloads=keep_results,
        progress_monitor=progress_monitor,
        progress_interval=progress_interval,
    )
    resume = (run_dir / "run_state.json").is_file()
    return Pipeline(
        config,
        loaded,
        reference_crawler=crawler,  # type: ignore[arg-type]
    ).run(resume=resume)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI while retaining the earlier underscore aliases."""
    from src.crawling.config import settings

    parser = argparse.ArgumentParser(
        description="Two-phase resumable arXiv corpus pipeline"
    )
    parser.add_argument("--start-month", "--start_month")
    parser.add_argument("--start-id", "--start_id", type=int)
    parser.add_argument("--end-month", "--end_month")
    parser.add_argument("--end-id", "--end_id", type=int)
    parser.add_argument("--ranges-file", type=Path)
    parser.add_argument("--ids-file", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        "--max-candidates",
        type=int,
        help="maximum number of valid (found) papers for a CLI range",
    )
    selection.add_argument("--all", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument("--corpus-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--download-cache-dir", type=Path)
    parser.add_argument("--without-references", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-workers", type=int)
    parser.add_argument("--reference-batch-size", type=int)
    parser.add_argument("--source-batch-size", type=int, default=50)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="parallel source-download workers (default: 4)",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=None,
        help="minimum seconds between source requests; defaults to project settings",
    )
    parser.add_argument("--source-prefetch-queue-size", type=int, default=2)
    parser.add_argument("--reference-queue-size", type=int, default=2)
    parser.add_argument("--discovery-batch-size", type=int, default=100)
    parser.add_argument("--discovery-fallback-batch-size", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--fsync-every", type=int, default=10)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--max-runtime-hours", type=float)
    parser.add_argument("--grace-period-seconds", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=10.0)
    parser.add_argument("--max-delay", type=float, default=300.0)
    parser.add_argument("--rate-limit-cooldown", type=float, default=20.0)
    parser.add_argument("--rate-limit-window", type=float, default=60.0)
    parser.add_argument("--keep-results", action="store_true")
    parser.add_argument("--debug-reference-metrics", action="store_true")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="seconds between console progress snapshots (default: 1.0)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the live console progress monitor",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _request_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> InputRequest:
    has_range = any(
        value is not None
        for value in (args.start_month, args.start_id, args.end_month, args.end_id)
    )
    if (args.ranges_file is not None or args.ids_file is not None) and (
        args.limit is not None or args.all
    ):
        parser.error("--limit/--all may only be used with a single CLI range")
    if args.ranges_file is not None and args.ids_file is not None:
        parser.error("--ranges-file and --ids-file are mutually exclusive")
    if has_range and (args.ranges_file is not None or args.ids_file is not None):
        parser.error("a CLI range cannot be combined with an input file")
    return InputRequest(
        start_month=args.start_month,
        start_id=args.start_id,
        end_month=args.end_month,
        end_id=args.end_id,
        ranges_file=args.ranges_file,
        ids_file=args.ids_file,
        limit=args.limit,
    )


def _print_terminal_summary(summary: Mapping[str, object], run_dir: Path) -> None:
    """Print a compact CLI result while keeping the full JSON on disk."""
    failed = summary.get("failed_ids")
    failed_counts = {
        stage: len(values) if isinstance(values, list) else 0
        for stage, values in (failed.items() if isinstance(failed, Mapping) else [])
    }
    print("\nPipeline summary")
    print(f"status={summary.get('status')}")
    print(
        "papers="
        f"{summary.get('found_count', 0)} found / "
        f"{summary.get('not_found_count', 0)} not_found / "
        f"{summary.get('candidate_id_count', 0)} candidates"
    )
    print(
        "source_success="
        f"{summary.get('source_success_count', 0)} "
        f"reference_success={summary.get('reference_success_count', 0)}"
    )
    print(
        "versions="
        f"{summary.get('total_versions', 0)} total "
        f"avg_per_id:{float(summary.get('average_versions_per_id', 0.0)):.3f} "
        f"avg_per_found_id:{float(summary.get('average_versions_per_found_id', 0.0)):.3f}"
    )
    print(
        "failed_ids="
        f"discovery:{failed_counts.get('discovery', 0)} "
        f"source:{failed_counts.get('source', 0)} "
        f"reference:{failed_counts.get('reference', 0)}"
    )
    print(
        "time_seconds="
        f"discovery:{float(summary.get('discovery_time_seconds', 0.0)):.2f} "
        f"source:{float(summary.get('source_time_seconds', 0.0)):.2f} "
        f"reference_api:{float(summary.get('reference_api_time_seconds', summary.get('reference_time_seconds', 0.0))):.2f} "
        f"reference_stage:{float(summary.get('reference_stage_wall_seconds', 0.0)):.2f} "
        f"wall:{float(summary.get('wall_time_seconds', 0.0)):.2f}"
    )
    print(
        "source_throughput="
        f"{float(summary.get('source_papers_per_second', 0.0)):.3f} papers/s "
        f"{float(summary.get('source_versions_per_hour', 0.0)):.2f} versions/h "
        f"success_rate:{float(summary.get('source_version_success_rate', 0.0)):.3f} "
        f"workers:{summary.get('source_worker_count', 1)} "
        f"429:{summary.get('downloader_http_429_count', 0)} "
        f"retry_sleep:{float(summary.get('downloader_retry_sleep_seconds', 0.0)):.2f}s "
        f"limiter_wait:{float(summary.get('downloader_limiter_wait_seconds', 0.0)):.2f}s "
        f"cache_hit_rate:{float(summary.get('downloader_cache_hit_rate', 0.0)):.3f}"
    )
    print(
        "reference_queue_wait_seconds="
        f"{float(summary.get('reference_queue_wait_seconds', 0.0)):.2f}"
    )
    print(
        "resources="
        f"peak_rss_mb:{summary.get('peak_rss_mb')} "
        f"disk_growth_bytes:{summary.get('disk_growth_bytes')}"
    )
    print(f"summary={run_dir / 'summary.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by Colab via ``python -m src.crawling.pipeline``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        loaded = load_input(_request_from_args(parser, args))
        if args.dry_run:
            print(json.dumps(loaded.to_json(), ensure_ascii=False, indent=2))
            return 0
        config = PipelineConfig(
            run_dir=args.output_dir,
            corpus_dir=args.corpus_dir,
            download_cache_dir=args.download_cache_dir,
            discovery_batch_size=args.discovery_batch_size,
            discovery_fallback_batch_size=args.discovery_fallback_batch_size,
            source_batch_size=args.source_batch_size,
            download_workers=args.download_workers,
            source_request_interval=args.request_interval,
            source_prefetch_queue_size=args.source_prefetch_queue_size,
            reference_workers=args.reference_workers,
            reference_batch_size=args.reference_batch_size,
            reference_queue_size=args.reference_queue_size,
            checkpoint_every=args.checkpoint_every,
            fsync_every=args.fsync_every,
            min_free_gb=args.min_free_gb,
            max_runtime_hours=args.max_runtime_hours,
            grace_period_seconds=args.grace_period_seconds,
            without_references=args.without_references,
            keep_downloads=args.keep_results,
            debug_reference_metrics=args.debug_reference_metrics,
            progress_monitor=not args.no_progress,
            progress_interval=args.progress_interval,
            retry_policy=RetryPolicy(
                max_retries=args.max_retries,
                base_delay=args.base_delay,
                max_delay=args.max_delay,
                rate_limit_cooldown=args.rate_limit_cooldown,
                rate_limit_window_seconds=args.rate_limit_window,
            ),
        )
        summary = Pipeline(config, loaded).run(resume=args.resume)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    _print_terminal_summary(summary, args.output_dir.resolve())
    return 0 if str(summary.get("status", "")).startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
