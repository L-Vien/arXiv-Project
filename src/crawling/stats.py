"""Statistics and benchmarking utilities for the arXiv pipeline.

Responsibilities:
- Aggregate paper/version/reference results incrementally.
- Avoid storing all paper results in memory.
- Measure wall-clock time by pipeline stage.
- Measure average and peak process RSS.
- Measure filesystem disk growth.
- Measure final sizes of selected output directories.
- Produce report-ready success rates and average-size metrics.

This module does not perform network requests and does not write reports.
The caller, usually pipeline.py, decides when to persist the returned
statistics.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # Optional dependency.
    psutil = None

try:
    import resource
except ImportError:  # Not available on every platform.
    resource = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _number(value: Any) -> float:
    """Return a finite numeric value as float.

    Invalid values such as None, bool, strings, NaN and Infinity return 0.0.
    """
    if isinstance(value, bool):
        return 0.0

    if not isinstance(value, (int, float)):
        return 0.0

    converted = float(value)

    if not math.isfinite(converted):
        return 0.0

    return converted


def _non_negative_int(value: Any) -> int:
    """Convert a value to a non-negative integer safely."""
    numeric = _number(value)
    return max(0, int(numeric))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Return numerator / denominator or 0.0 when denominator is zero."""
    denominator_value = _number(denominator)

    if denominator_value <= 0:
        return 0.0

    return float(numerator) / denominator_value


def _percent(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage in the range 0..100."""
    return round(_ratio(numerator, denominator) * 100.0, 4)


def _normalize_paths(
    paths: Sequence[str | Path] | str | Path,
) -> tuple[Path, ...]:
    """Normalize one path or a sequence of paths."""
    if isinstance(paths, (str, Path)):
        return (Path(paths),)

    return tuple(Path(path) for path in paths)


# ---------------------------------------------------------------------------
# Streaming result aggregation
# ---------------------------------------------------------------------------

class SummaryAccumulator:
    """Incrementally aggregate pipeline records.

    The accumulator keeps only counters and scalar values. It never stores
    paper records, version records or reference lists.

    Therefore memory usage is approximately O(number_of_status_types), not
    O(number_of_processed_papers).
    """

    def __init__(self) -> None:
        # Status counters.
        self.paper_statuses: Counter[str] = Counter()
        self.version_statuses: Counter[str] = Counter()
        self.download_statuses: Counter[str] = Counter()
        self.processing_statuses: Counter[str] = Counter()
        self.reference_statuses: Counter[str] = Counter()
        self.errors_by_stage: Counter[str] = Counter()
        self.source_failed_ids: set[str] = set()
        self.reference_failed_ids: set[str] = set()
        self.no_tex_ids: set[str] = set()

        # Counts.
        self.total_papers = 0
        self.total_versions = 0
        self.total_references = 0

        # Byte totals.
        self.total_archive_bytes = 0
        self.total_raw_bytes = 0
        self.total_kept_bytes = 0

        # Cumulative timing totals.
        self.cumulative_paper_seconds = 0.0
        self.metadata_seconds = 0.0
        self.download_seconds = 0.0
        self.processing_seconds = 0.0
        self.references_seconds = 0.0

        # Derived counters.
        self.fully_completed_papers = 0
        self.fully_successful_papers = 0
        self.reference_cache_hits = 0
        self.metadata_successful_papers = 0
        self.metadata_failed_papers = 0

    def update(self, paper: Mapping[str, Any]) -> None:
        """Add exactly one paper result to the aggregate summary."""
        self.total_papers += 1

        # ------------------------------------------------------------------
        # Paper-level status and timing
        # ------------------------------------------------------------------
        paper_status = str(paper.get("status", "unknown"))
        self.paper_statuses[paper_status] += 1

        arxiv_id = paper.get("arxiv_id")
        if (
            isinstance(arxiv_id, str)
            and paper_status not in {
                "success",
                "success_with_no_tex",
            }
        ):
            self.source_failed_ids.add(arxiv_id)

        if (
            isinstance(arxiv_id, str)
            and paper_status == "success_with_no_tex"
        ):
            self.no_tex_ids.add(arxiv_id)

        metadata_status = str(
            paper.get("metadata_status", "not_started")
        )

        metadata_success = metadata_status == "success"

        if metadata_success:
            self.metadata_successful_papers += 1

        elif metadata_status == "failed":
            self.metadata_failed_papers += 1

        paper_timings = paper.get("timings") or {}

        self.cumulative_paper_seconds += _number(
            paper_timings.get("total_seconds")
        )
        self.metadata_seconds += _number(
            paper_timings.get("metadata_seconds")
        )
        self.references_seconds += _number(
            paper_timings.get("references_seconds")
        )

        # ------------------------------------------------------------------
        # Reference-level status and count
        # ------------------------------------------------------------------
        references = paper.get("references") or {}

        reference_status = str(
            references.get("status", "not_started")
        )

        if (
            isinstance(arxiv_id, str)
            and reference_status == "failed"
        ):
            self.reference_failed_ids.add(arxiv_id)

        reference_count = _non_negative_int(
            references.get("count")
        )

        self.reference_statuses[reference_status] += 1
        self.total_references += reference_count

        if references.get("cache_hit") is True:
            self.reference_cache_hits += 1

        # ------------------------------------------------------------------
        # Version-level status, size and timing
        # ------------------------------------------------------------------
        versions = paper.get("versions") or []
        version_statuses_for_paper: list[str] = []

        for raw_version in versions:
            if isinstance(raw_version, Mapping):
                version = raw_version
            else:
                version = {}

            self.total_versions += 1

            version_status = str(
                version.get("status", "unknown")
            )
            version_statuses_for_paper.append(version_status)
            self.version_statuses[version_status] += 1

            download = version.get("download") or {}
            processing = version.get("processing") or {}

            download_status = str(
                download.get("status", "not_started")
            )
            processing_status = str(
                processing.get("status", "not_started")
            )

            self.download_statuses[download_status] += 1
            self.processing_statuses[processing_status] += 1

            self.total_archive_bytes += _non_negative_int(
                processing.get("archive_size")
            )
            self.total_raw_bytes += _non_negative_int(
                processing.get("raw_size")
            )
            self.total_kept_bytes += _non_negative_int(
                processing.get("kept_size")
            )

            version_timings = version.get("timings") or {}

            self.download_seconds += _number(
                version_timings.get("download_seconds")
            )
            self.processing_seconds += _number(
                version_timings.get("processing_seconds")
            )

        # ------------------------------------------------------------------
        # Errors
        # ------------------------------------------------------------------
        for error in paper.get("errors") or []:
            if isinstance(error, Mapping):
                stage = str(error.get("stage", "unknown"))
            else:
                stage = "unknown"

            self.errors_by_stage[stage] += 1

        # ------------------------------------------------------------------
        # Full-completion counters
        # ------------------------------------------------------------------
        usable_versions = bool(version_statuses_for_paper) and all(
            status in {"success", "no_tex"}
            for status in version_statuses_for_paper
        )

        if (
            metadata_success
            and usable_versions
            and reference_status == "success"
        ):
            self.fully_completed_papers += 1

            if all(
                status == "success"
                for status in version_statuses_for_paper
            ):
                self.fully_successful_papers += 1

    def finalize(
        self,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a report-ready summary.

        ``context`` allows pipeline.py to add information that stats.py
        cannot know by itself, such as:

        - candidate_id_count
        - discovered_paper_count
        - not_found_count
        - discovery_failed_count
        - required_paper_count
        - manifest_records_written
        - end-to-end wall time
        """
        context_data = dict(context or {})

        source_successful_papers = (
            self.paper_statuses["success"]
            + self.paper_statuses["success_with_no_tex"]
        )

        source_fully_successful_papers = (
            self.paper_statuses["success"]
        )

        failed_papers = sum(
            count
            for status, count in self.paper_statuses.items()
            if status not in {
                "success",
                "success_with_no_tex",
                "partial_success",
                "no_versions",
            }
        )

        usable_versions = (
            self.version_statuses["success"]
            + self.version_statuses["no_tex"]
        )

        failed_processing_versions = sum(
            count
            for status, count in self.processing_statuses.items()
            if status not in {
                "ok",
                "no_tex",
                "not_started",
            }
        )

        reference_success_papers = (
            self.reference_statuses["success"]
        )
        reference_failed_papers = (
            self.reference_statuses["failed"]
        )
        reference_skipped_papers = (
            self.reference_statuses["skipped"]
        )

        reference_attempted_papers = (
            reference_success_papers
            + reference_failed_papers
        )

        # The default denominator is the number of processed papers.
        # Pipeline.py can override it by passing required_paper_count.
        required_paper_count = context_data.get(
            "required_paper_count"
        )

        success_denominator = (
            required_paper_count
            if required_paper_count is not None
            else self.total_papers
        )

        average_raw_bytes_per_version = _ratio(
            self.total_raw_bytes,
            self.total_versions,
        )

        average_kept_bytes_per_version = _ratio(
            self.total_kept_bytes,
            self.total_versions,
        )

        average_raw_bytes_per_paper = _ratio(
            self.total_raw_bytes,
            self.total_papers,
        )

        average_kept_bytes_per_paper = _ratio(
            self.total_kept_bytes,
            self.total_papers,
        )

        retention_ratio = _ratio(
            self.total_kept_bytes,
            self.total_raw_bytes,
        )

        reduction_ratio = (
            1.0 - retention_ratio
            if self.total_raw_bytes > 0
            else 0.0
        )

        summary: dict[str, Any] = {
            # --------------------------------------------------------------
            # Basic counts
            # --------------------------------------------------------------
            "total_papers": self.total_papers,
            "processed_paper_count": self.total_papers,
            "successful_papers": source_successful_papers,
            "source_successful_papers": source_successful_papers,
            "source_fully_successful_papers": (
                source_fully_successful_papers
            ),
            "partial_success_papers": (
                self.paper_statuses["partial_success"]
            ),
            "failed_papers": failed_papers,
            "no_versions_papers": (
                self.paper_statuses["no_versions"]
            ),

            "total_versions": self.total_versions,
            "successful_versions": (
                self.version_statuses["success"]
            ),
            "no_tex_versions": (
                self.version_statuses["no_tex"]
            ),
            "failed_versions": (
                self.total_versions - usable_versions
            ),

            "total_references": self.total_references,

            # --------------------------------------------------------------
            # Completion counters
            # --------------------------------------------------------------
            "fully_completed_papers": (
                self.fully_completed_papers
            ),
            "fully_successful_papers": (
                self.fully_successful_papers
            ),

            "metadata_successful_papers": (
                self.metadata_successful_papers
            ),
            "metadata_failed_papers": (
                self.metadata_failed_papers
            ),
            "versions_failed_papers": (
                self.paper_statuses["versions_failed"]
            ),

            # --------------------------------------------------------------
            # Download and processing counters
            # --------------------------------------------------------------
            "downloaded_versions": (
                self.download_statuses["downloaded"]
            ),
            "skipped_downloads": (
                self.download_statuses["skipped"]
            ),
            "failed_downloads": (
                self.download_statuses["failed"]
            ),

            "processed_versions": (
                self.processing_statuses["ok"]
            ),
            "failed_processing_versions": (
                failed_processing_versions
            ),

            # --------------------------------------------------------------
            # Reference counters
            # --------------------------------------------------------------
            "reference_success_papers": (
                reference_success_papers
            ),
            "reference_failed_papers": (
                reference_failed_papers
            ),
            "reference_skipped_papers": (
                reference_skipped_papers
            ),
            "reference_attempted_papers": (
                reference_attempted_papers
            ),
            "reference_cache_hits": (
                self.reference_cache_hits
            ),
            "source_failed_ids": sorted(
                self.source_failed_ids
            ),
            "reference_failed_ids": sorted(
                self.reference_failed_ids
            ),
            "no_tex_ids": sorted(self.no_tex_ids),

            # --------------------------------------------------------------
            # Success rates, stored as both ratio and percentage
            # --------------------------------------------------------------
            "paper_success_rate": _ratio(
                source_successful_papers,
                success_denominator,
            ),
            "paper_success_rate_percent": _percent(
                source_successful_papers,
                success_denominator,
            ),

            "source_full_success_rate": _ratio(
                source_fully_successful_papers,
                success_denominator,
            ),
            "source_full_success_rate_percent": _percent(
                source_fully_successful_papers,
                success_denominator,
            ),

            "fully_completed_rate": _ratio(
                self.fully_completed_papers,
                success_denominator,
            ),
            "fully_completed_rate_percent": _percent(
                self.fully_completed_papers,
                success_denominator,
            ),

            "version_success_rate": _ratio(
                self.version_statuses["success"],
                self.total_versions,
            ),
            "version_success_rate_percent": _percent(
                self.version_statuses["success"],
                self.total_versions,
            ),

            "version_usable_rate": _ratio(
                usable_versions,
                self.total_versions,
            ),
            "version_usable_rate_percent": _percent(
                usable_versions,
                self.total_versions,
            ),

            "metadata_success_rate": _ratio(
                self.metadata_successful_papers,
                self.total_papers,
            ),
            "metadata_success_rate_percent": _percent(
                self.metadata_successful_papers,
                self.total_papers,
            ),

            "download_completion_rate": _ratio(
                self.download_statuses["downloaded"]
                + self.download_statuses["skipped"],
                self.total_versions,
            ),
            "download_completion_rate_percent": _percent(
                self.download_statuses["downloaded"]
                + self.download_statuses["skipped"],
                self.total_versions,
            ),

            "processing_usable_rate": _ratio(
                self.processing_statuses["ok"]
                + self.processing_statuses["no_tex"],
                self.total_versions,
            ),
            "processing_usable_rate_percent": _percent(
                self.processing_statuses["ok"]
                + self.processing_statuses["no_tex"],
                self.total_versions,
            ),

            # Reference success rate among all processed papers.
            "reference_success_rate_all_papers": _ratio(
                reference_success_papers,
                self.total_papers,
            ),
            "reference_success_rate_all_papers_percent": _percent(
                reference_success_papers,
                self.total_papers,
            ),

            # Reference success rate only among papers where reference
            # crawling was attempted.
            "reference_success_rate_attempted": _ratio(
                reference_success_papers,
                reference_attempted_papers,
            ),
            "reference_success_rate_attempted_percent": _percent(
                reference_success_papers,
                reference_attempted_papers,
            ),

            # --------------------------------------------------------------
            # Source-size metrics
            # --------------------------------------------------------------
            "total_archive_bytes_processed": (
                self.total_archive_bytes
            ),
            "total_raw_bytes_processed": (
                self.total_raw_bytes
            ),
            "total_kept_bytes_processed": (
                self.total_kept_bytes
            ),

            "average_raw_bytes_per_version": (
                average_raw_bytes_per_version
            ),
            "average_kept_bytes_per_version": (
                average_kept_bytes_per_version
            ),
            "average_raw_bytes_per_paper": (
                average_raw_bytes_per_paper
            ),
            "average_kept_bytes_per_paper": (
                average_kept_bytes_per_paper
            ),

            "kept_to_raw_ratio": retention_ratio,
            "kept_to_raw_ratio_percent": round(
                retention_ratio * 100.0,
                4,
            ),
            "removed_data_ratio": reduction_ratio,
            "removed_data_ratio_percent": round(
                reduction_ratio * 100.0,
                4,
            ),
            "removed_bytes": max(
                0,
                self.total_raw_bytes
                - self.total_kept_bytes,
            ),

            # --------------------------------------------------------------
            # Reference-size metrics
            # --------------------------------------------------------------
            "average_references_per_processed_paper": _ratio(
                self.total_references,
                self.total_papers,
            ),
            "average_references_per_successful_reference_paper": _ratio(
                self.total_references,
                reference_success_papers,
            ),

            # --------------------------------------------------------------
            # Timing metrics
            # --------------------------------------------------------------
            "cumulative_paper_seconds": round(
                self.cumulative_paper_seconds,
                4,
            ),
            "total_processing_seconds": round(
                self.cumulative_paper_seconds,
                4,
            ),
            "cumulative_metadata_seconds": round(
                self.metadata_seconds,
                4,
            ),
            "cumulative_download_seconds": round(
                self.download_seconds,
                4,
            ),
            "cumulative_processing_seconds": round(
                self.processing_seconds,
                4,
            ),
            "cumulative_references_seconds": round(
                self.references_seconds,
                4,
            ),
            "average_paper_processing_seconds": round(
                _ratio(
                    self.cumulative_paper_seconds,
                    self.total_papers,
                ),
                4,
            ),
            # Backward-compatible field.
            "average_paper_seconds": round(
                _ratio(
                    self.cumulative_paper_seconds,
                    self.total_papers,
                ),
                4,
            ),
            "downloaded_versions_per_second": round(
                _ratio(
                    self.download_statuses["downloaded"],
                    self.download_seconds,
                ),
                4,
            ),
            "processed_versions_per_second": round(
                _ratio(
                    self.processing_statuses["ok"]
                    + self.processing_statuses["no_tex"],
                    self.processing_seconds,
                ),
                4,
            ),

            # --------------------------------------------------------------
            # Raw status dictionaries
            # --------------------------------------------------------------
            "paper_status_counts": dict(
                self.paper_statuses
            ),
            "version_status_counts": dict(
                self.version_statuses
            ),
            "download_status_counts": dict(
                self.download_statuses
            ),
            "processing_status_counts": dict(
                self.processing_statuses
            ),
            "reference_status_counts": dict(
                self.reference_statuses
            ),
            "errors_by_stage": dict(
                self.errors_by_stage
            ),
        }

        # Pipeline-level context is intentionally added at the end.
        # This allows pipeline.py to provide discovery and persistence
        # information without making this module depend on pipeline.py.
        summary.update(context_data)

        return summary


def summarize_results(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize results incrementally without converting to a list.

    This function is kept for backward compatibility. It accepts both lists
    and generators.
    """
    accumulator = SummaryAccumulator()

    for paper in results:
        accumulator.update(paper)

    return accumulator.finalize()


# ---------------------------------------------------------------------------
# Exact directory-size measurement
# ---------------------------------------------------------------------------

def directory_size_bytes(path: str | Path) -> int:
    """Return the recursive size of a directory in bytes.

    Symlinks are not followed. Missing or inaccessible paths contribute zero.
    This function is intentionally not called every second because walking
    a large corpus tree repeatedly would create substantial I/O overhead.
    """
    root = Path(path)

    if not root.exists():
        return 0

    try:
        if root.is_file():
            return root.stat().st_size
    except OSError:
        return 0

    total = 0
    pending: list[Path] = [root]

    while pending:
        current = pending.pop()

        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue

                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(
                                follow_symlinks=False
                            ).st_size

                    except OSError:
                        continue

        except OSError:
            continue

    return total


# ---------------------------------------------------------------------------
# Runtime, memory and disk benchmark
# ---------------------------------------------------------------------------

class RunBenchmark:
    """Measure wall time, stage time, memory and disk usage.

    Memory and filesystem sampling run in a background daemon thread only
    after start_sampling() is called.

    Parameters
    ----------
    monitor_paths:
        Paths whose containing filesystems should be monitored with
        shutil.disk_usage().

    size_paths:
        Directories whose exact recursive sizes should be measured. These
        directory walks are intentionally performed only at explicit sampling
        points, not every second.

    sample_interval:
        Background RSS/filesystem sampling interval in seconds.

    auto_start:
        If True, start background sampling inside the constructor.
        False is recommended because pipeline.py can start sampling after
        its output directories have been initialized.
    """

    def __init__(
        self,
        monitor_paths: Sequence[str | Path] | str | Path = (),
        size_paths: Sequence[str | Path] | str | Path = (),
        sample_interval: float = 1.0,
        auto_start: bool = False,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be greater than zero"
            )

        self._started_at = time.perf_counter()

        self.stage_seconds: dict[str, float] = defaultdict(float)
        self.stage_wall_seconds: dict[str, float] = defaultdict(float)

        self._monitor_paths = _normalize_paths(
            monitor_paths
        )
        self._size_paths = _normalize_paths(
            size_paths
        )
        self._sample_interval = float(sample_interval)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None

        # RSS metrics.
        self._current_rss_mb: float | None = None
        self._peak_rss_mb: float | None = None
        self._rss_sum_mb = 0.0
        self._rss_sample_count = 0

        # Disk/filesystem metrics.
        self._disk_initial_used: dict[str, int] = {}
        self._disk_last_used: dict[str, int] = {}
        self._disk_peak_used: dict[str, int] = {}
        self._disk_min_free: dict[str, int] = {}

        # Exact directory-size metrics.
        self._directory_initial_size: dict[str, int] = {}
        self._directory_last_size: dict[str, int] = {}
        self._directory_peak_size: dict[str, int] = {}

        self.sample_memory()
        self.sample_disk()
        self.sample_directory_sizes()

        if auto_start:
            self.start_sampling()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Measure and accumulate work time for one named operation."""
        started = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - started

            with self._lock:
                self.stage_seconds[stage] += elapsed

            self.sample_memory()
            self.sample_disk()

    def add_stage_wall_time(
        self,
        stage: str,
        elapsed_seconds: float,
    ) -> None:
        """Add elapsed wall time for a pipeline stage."""
        if elapsed_seconds < 0:
            raise ValueError(
                "elapsed_seconds must be non-negative"
            )

        with self._lock:
            self.stage_wall_seconds[stage] += float(
                elapsed_seconds
            )

    def restore_stage_times(
        self,
        benchmark: Mapping[str, Any],
    ) -> None:
        """Restore timing totals from a previous run checkpoint."""
        stage_work = benchmark.get("stage_work_seconds")
        legacy_stage_times = not isinstance(stage_work, Mapping)
        if not isinstance(stage_work, Mapping):
            stage_work = benchmark.get("stage_wall_seconds")

        stage_wall = benchmark.get("stage_wall_seconds")

        with self._lock:
            if isinstance(stage_work, Mapping):
                for stage, elapsed in stage_work.items():
                    value = _number(elapsed)
                    if value >= 0:
                        self.stage_seconds[str(stage)] += value

            if isinstance(stage_wall, Mapping) and not legacy_stage_times:
                for stage, elapsed in stage_wall.items():
                    value = _number(elapsed)
                    if value >= 0:
                        self.stage_wall_seconds[str(stage)] += value

    def start_sampling(self) -> None:
        """Start background RSS and filesystem sampling."""
        with self._lock:
            if (
                self._sampler_thread is not None
                and self._sampler_thread.is_alive()
            ):
                return

            self._stop_event.clear()

            self._sampler_thread = threading.Thread(
                target=self._sampling_loop,
                name="pipeline-benchmark-sampler",
                daemon=True,
            )
            self._sampler_thread.start()

    def stop_sampling(self) -> None:
        """Stop the sampler and take final lightweight samples."""
        self._stop_event.set()

        thread = self._sampler_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        self.sample_memory()
        self.sample_disk()
        self.sample_directory_sizes()

    def _sampling_loop(self) -> None:
        """Background RSS/filesystem sampling loop."""
        while not self._stop_event.wait(
            self._sample_interval
        ):
            self.sample_memory()
            self.sample_disk()

    def sample_memory(self) -> None:
        """Take one RSS sample and update O(1) memory metrics."""
        if psutil is None:
            return

        try:
            rss_mb = (
                psutil.Process().memory_info().rss
                / (1024 * 1024)
            )

            with self._lock:
                self._current_rss_mb = rss_mb
                self._rss_sum_mb += rss_mb
                self._rss_sample_count += 1

                if (
                    self._peak_rss_mb is None
                    or rss_mb > self._peak_rss_mb
                ):
                    self._peak_rss_mb = rss_mb

        except (
            OSError,
            RuntimeError,
            ValueError,
            AttributeError,
        ):
            return

    @staticmethod
    def _resource_peak_rss_mb() -> float | None:
        """Return OS-reported peak RSS when resource is available.

        On Linux, ru_maxrss is reported in KiB.
        On macOS, ru_maxrss is reported in bytes.
        """
        if resource is None:
            return None

        try:
            value = float(
                resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
            )

            if sys.platform == "darwin":
                value /= 1024 * 1024
            else:
                value /= 1024

            return round(value, 3)

        except (
            OSError,
            RuntimeError,
            ValueError,
            AttributeError,
        ):
            return None

    def sample_disk(self) -> None:
        """Sample used/free space for monitored filesystems.

        These values describe the entire filesystem containing each path,
        not the exact size of an individual directory.
        """
        for path in self._monitor_paths:
            key = str(path)

            try:
                usage = shutil.disk_usage(path)
            except OSError:
                continue

            used_bytes = usage.total - usage.free

            with self._lock:
                if key not in self._disk_initial_used:
                    self._disk_initial_used[key] = used_bytes

                self._disk_last_used[key] = used_bytes

                previous_peak = self._disk_peak_used.get(
                    key,
                    used_bytes,
                )
                self._disk_peak_used[key] = max(
                    previous_peak,
                    used_bytes,
                )

                previous_min_free = self._disk_min_free.get(
                    key,
                    usage.free,
                )
                self._disk_min_free[key] = min(
                    previous_min_free,
                    usage.free,
                )

    def sample_directory_sizes(self) -> None:
        """Measure exact recursive sizes of configured directories.

        This operation walks the directory trees and may be expensive.
        It is called at initialization, finalization and explicit checkpoints,
        but not on every background sampling tick.
        """
        for path in self._size_paths:
            key = str(path)
            size = directory_size_bytes(path)

            with self._lock:
                if key not in self._directory_initial_size:
                    self._directory_initial_size[key] = size

                self._directory_last_size[key] = size

                previous_peak = self._directory_peak_size.get(
                    key,
                    size,
                )
                self._directory_peak_size[key] = max(
                    previous_peak,
                    size,
                )

    def snapshot(
        self,
        summary: Mapping[str, Any],
        *,
        required_paper_count: int | None = None,
    ) -> dict[str, Any]:
        """Return current benchmark metrics.

        Call stop_sampling() before the final snapshot.

        required_paper_count should represent the number of real assigned
        papers discovered, not the number of numeric candidate IDs. If it is
        omitted, this method tries summary["required_paper_count"] and then
        summary["total_papers"].
        """
        self.sample_memory()
        self.sample_disk()
        self.sample_directory_sizes()

        elapsed = time.perf_counter() - self._started_at

        total_papers = _non_negative_int(
            summary.get("total_papers")
        )
        total_versions = _non_negative_int(
            summary.get("total_versions")
        )

        if required_paper_count is None:
            context_required_count = summary.get(
                "required_paper_count"
            )

            if context_required_count is not None:
                required_paper_count = _non_negative_int(
                    context_required_count
                )
            else:
                required_paper_count = total_papers

        with self._lock:
            stage_work_seconds = {
                stage: round(seconds, 4)
                for stage, seconds in sorted(
                    self.stage_seconds.items()
                )
            }
            stage_wall_seconds = {
                stage: round(seconds, 4)
                for stage, seconds in sorted(
                    self.stage_wall_seconds.items()
                )
            }

            peak_rss_mb = (
                round(self._peak_rss_mb, 3)
                if self._peak_rss_mb is not None
                else None
            )

            average_rss_mb = (
                round(
                    self._rss_sum_mb
                    / self._rss_sample_count,
                    3,
                )
                if self._rss_sample_count
                else None
            )

            disk_initial_used = dict(
                self._disk_initial_used
            )
            disk_last_used = dict(
                self._disk_last_used
            )
            disk_peak_used = dict(
                self._disk_peak_used
            )
            disk_min_free = dict(
                self._disk_min_free
            )

            disk_peak_growth = {
                key: max(
                    0,
                    disk_peak_used[key]
                    - disk_initial_used.get(key, 0),
                )
                for key in disk_peak_used
            }

            directory_initial_size = dict(
                self._directory_initial_size
            )
            directory_last_size = dict(
                self._directory_last_size
            )
            directory_peak_size = dict(
                self._directory_peak_size
            )

            directory_growth = {
                key: max(
                    0,
                    directory_peak_size[key]
                    - directory_initial_size.get(key, 0),
                )
                for key in directory_peak_size
            }

            memory_sample_count = self._rss_sample_count

        os_peak_rss_mb = self._resource_peak_rss_mb()

        return {
            # --------------------------------------------------------------
            # Runtime
            # --------------------------------------------------------------
            "wall_time_seconds": round(
                elapsed,
                4,
            ),
            "stage_wall_seconds": stage_wall_seconds,
            "stage_work_seconds": stage_work_seconds,
            "discovery_time_seconds": stage_wall_seconds.get(
                "discovery",
                0.0,
            ),
            "source_time_seconds": stage_wall_seconds.get(
                "source",
                0.0,
            ),
            "reference_time_seconds": stage_wall_seconds.get(
                "reference",
                0.0,
            ),

            "papers_per_second": round(
                total_papers / elapsed,
                6,
            ) if elapsed > 0 else 0.0,

            "versions_per_second": round(
                total_versions / elapsed,
                6,
            ) if elapsed > 0 else 0.0,

            "end_to_end_average_seconds_per_required_paper": round(
                elapsed / required_paper_count,
                4,
            ) if required_paper_count > 0 else 0.0,

            "required_paper_count_for_runtime": (
                required_paper_count
            ),

            # --------------------------------------------------------------
            # Memory
            # --------------------------------------------------------------
            "memory_sampling_available": (
                memory_sample_count > 0
            ),
            "peak_rss_mb": peak_rss_mb,
            "max_sampled_rss_mb": peak_rss_mb,
            "current_rss_mb": (
                round(self._current_rss_mb, 3)
                if self._current_rss_mb is not None
                else None
            ),
            "average_sampled_rss_mb": average_rss_mb,
            "avg_sampled_rss_mb": average_rss_mb,
            "memory_sample_count": memory_sample_count,

            # OS-level peak RSS since process start. This is an additional
            # measurement and may be more reliable than periodic sampling.
            "os_peak_rss_mb": os_peak_rss_mb,

            # --------------------------------------------------------------
            # Filesystem-level disk metrics
            # --------------------------------------------------------------
            "disk_sampling_available": bool(
                disk_last_used
            ),
            "disk_initial_used_bytes": (
                disk_initial_used
            ),
            "disk_last_used_bytes": (
                disk_last_used
            ),
            "disk_peak_used_bytes": (
                disk_peak_used
            ),
            "disk_peak_growth_bytes": (
                disk_peak_growth
            ),
            "disk_min_free_bytes": (
                disk_min_free
            ),

            # --------------------------------------------------------------
            # Exact directory metrics
            # --------------------------------------------------------------
            "directory_size_sampling_available": bool(
                directory_last_size
            ),
            "directory_initial_size_bytes": (
                directory_initial_size
            ),
            "directory_last_size_bytes": (
                directory_last_size
            ),
            "directory_peak_size_bytes": (
                directory_peak_size
            ),
            "directory_growth_bytes": (
                directory_growth
            ),

            # --------------------------------------------------------------
            # Measurement notes
            # --------------------------------------------------------------
            "notes": (
                "wall_time_seconds is end-to-end elapsed time from "
                "RunBenchmark initialization. stage_wall_seconds are "
                "cumulative stage measurements and are not necessarily "
                "additive when concurrency is used. peak_rss_mb is the "
                "maximum periodic psutil RSS sample. os_peak_rss_mb is "
                "the operating-system reported process peak when "
                "resource.getrusage is available. Filesystem disk "
                "metrics describe the filesystem containing monitor_paths. "
                "Exact directory metrics describe only size_paths. "
                "Use disk_peak_growth_bytes for estimated additional "
                "filesystem usage caused by the run."
            ),
        }
