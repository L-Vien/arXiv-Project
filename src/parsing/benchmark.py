"""Lightweight reproducible benchmark measurements for Milestone 2."""

from __future__ import annotations

import os
import platform
import threading
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - requirements include psutil.
    psutil = None


def directory_size_bytes(directory: Path) -> int:
    """Return the size of regular files below ``directory``."""

    if not directory.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    )


class BenchmarkRecorder:
    """Measure wall time, process-tree peak RSS, and generated output size.

    RSS is sampled in a lightweight background thread while the pipeline runs.
    The returned peak is therefore the maximum observed RSS at the configured
    sampling interval, with explicit samples taken at run/publication
    boundaries as well.
    """

    def __init__(self, *, sample_interval_seconds: float = 0.05) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self._sample_interval_seconds = sample_interval_seconds
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._start_rss: int | None = None
        self._peak_rss: int | None = None
        self._publication_started_at: dict[str, float] = {}
        self._publication_seconds: dict[str, float] = {}
        self._stage_seconds: dict[str, float] = defaultdict(float)
        self._sample_lock = threading.Lock()
        self._sampler_stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None

    @staticmethod
    def _rss() -> int | None:
        if psutil is None:
            return None
        try:
            process = psutil.Process(os.getpid())
            parent_rss = process.memory_info().rss
            try:
                children = process.children(recursive=True)
            except (OSError, psutil.Error):
                return int(parent_rss)
            total_rss = parent_rss
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (OSError, psutil.Error):
                    continue
            return int(total_rss)
        except (OSError, psutil.Error):
            return None

    def _sample(self) -> None:
        rss = self._rss()
        if rss is not None:
            with self._sample_lock:
                self._peak_rss = max(self._peak_rss or rss, rss)

    def _sampling_loop(self) -> None:
        while not self._sampler_stop.wait(self._sample_interval_seconds):
            self._sample()

    def _start_sampler(self) -> None:
        if psutil is None:
            return
        self._sampler_stop.clear()
        self._sampler_thread = threading.Thread(
            target=self._sampling_loop,
            name="m2-benchmark-memory",
            daemon=True,
        )
        self._sampler_thread.start()

    def _stop_sampler(self) -> None:
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            timeout = max(1.0, self._sample_interval_seconds * 4)
            self._sampler_thread.join(timeout=timeout)
            self._sampler_thread = None

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("benchmark has already been started")
        self._started_at = perf_counter()
        self._start_rss = self._rss()
        self._peak_rss = self._start_rss
        self._start_sampler()

    def start_publication(self, arxiv_id: str) -> None:
        if self._started_at is None:
            raise RuntimeError("benchmark must be started before a publication")
        self._publication_started_at[arxiv_id] = perf_counter()
        self._sample()

    def finish_publication(
        self,
        arxiv_id: str,
        elapsed_seconds: float | None = None,
    ) -> None:
        if elapsed_seconds is None:
            started_at = self._publication_started_at.get(arxiv_id)
            if started_at is None:
                raise RuntimeError(
                    f"publication {arxiv_id!r} was not started"
                )
            elapsed_seconds = perf_counter() - started_at
        self._publication_seconds[arxiv_id] = max(0.0, elapsed_seconds)
        self._sample()

    def record_stage(self, stage: str, elapsed_seconds: float) -> None:
        """Accumulate elapsed time for one named pipeline stage."""

        if not stage:
            raise ValueError("stage must be non-empty")
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        self._stage_seconds[stage] += elapsed_seconds

    def finish(
        self,
        *,
        output_dir: Path,
        publication_count: int,
        version_count: int,
        element_count: int,
    ) -> dict[str, Any]:
        if self._started_at is None:
            raise RuntimeError("benchmark has not been started")
        if self._finished_at is not None:
            raise RuntimeError("benchmark has already been finished")
        self._stop_sampler()
        self._finished_at = perf_counter()
        self._sample()
        elapsed = self._finished_at - self._started_at
        timed_publication_count = len(self._publication_seconds)
        average_publication_seconds = (
            sum(self._publication_seconds.values()) / timed_publication_count
            if timed_publication_count
            else None
        )
        return {
            "wall_time_seconds": elapsed,
            "publication_times_seconds": dict(
                sorted(self._publication_seconds.items())
            ),
            "average_publication_time_seconds": average_publication_seconds,
            "stage_timings_seconds": dict(sorted(self._stage_seconds.items())),
            "stage_timing_scope": (
                "aggregate elapsed time recorded by the parent and worker "
                "processes; parallel stage totals may exceed wall time"
            ),
            "publication_timing_scope": (
                "processing, validation, export, and per-publication statistics"
            ),
            "memory": {
                "start_rss_bytes": self._start_rss,
                "peak_rss_bytes": self._peak_rss,
                # Kept for consumers of the original TV1 output format.
                "peak_sampled_rss_bytes": self._peak_rss,
                "rss_scope": "parent_and_live_worker_processes_when_permitted",
                "sampling_interval_seconds": self._sample_interval_seconds,
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
            "output": {
                "directory": str(Path(output_dir)),
                "size_bytes": directory_size_bytes(Path(output_dir)),
                "publication_count": publication_count,
                "timed_publication_count": timed_publication_count,
                "version_count": version_count,
                "element_count": element_count,
            },
        }


__all__ = ["BenchmarkRecorder", "directory_size_bytes"]
