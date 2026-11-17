"""Periodic progress reporting for the Milestone 2 pipeline.

The renderer is intentionally shared with M1.  This module only adapts M2's
stage counters to the snapshot shape expected by M1's ``ProgressMonitor``.
The pipeline remains responsible for calling ``set_stage`` and ``complete``;
this module does not parse, normalize, or export any data.
"""

from __future__ import annotations

import logging
import sys
import threading
from time import monotonic
from typing import Any, TextIO

from src.crawling.logger import get_logger
from src.crawling.stats import ProgressMonitor as M1ProgressMonitor

STAGES = (
    "gather",
    "hierarchy",
    "references",
    "normalization",
    "deduplication",
    "validation",
    "export",
)

logger = get_logger("parsing.progress")


class M2ProgressMonitor:
    """Track M2 stages and render a snapshot every ``interval`` seconds.

    Counts are completion counts, so a failed or intentionally skipped item is
    still counted as processed and is exposed separately in the snapshot.
    ``versioned_id`` identifies the current input, for example
    ``2501.01234:v2``.
    """

    def __init__(
        self,
        *,
        interval: float = 1.0,
        enabled: bool = True,
        output: TextIO | None = None,
        target_logger: logging.Logger | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        self.interval = float(interval)
        self.enabled = enabled
        self._output = output
        self._logger = target_logger or logger
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._monitor: M1ProgressMonitor | None = None
        self._publication_total = 0
        self._version_total = 0
        self._current_publication = "-"
        self._current_version = "-"
        self._current_stage = "pending"
        self._input_description = "publication iterable"
        self._publication_completed = 0
        self._publication_success = 0
        self._publication_partial = 0
        self._publication_failed = 0
        self._reference_count = 0
        self._stages: dict[str, dict[str, Any]] = {
            stage: {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
                "current": "-",
                "input": "-",
                "message": "pending",
            }
            for stage in STAGES
        }

    def start(
        self,
        *,
        publication_count: int,
        version_count: int,
        input_description: str = "publication iterable",
    ) -> None:
        """Start the periodic renderer after the input totals are known."""

        if publication_count < 0 or version_count < 0:
            raise ValueError("progress totals must be non-negative")
        with self._lock:
            if self._started_at is not None:
                raise RuntimeError("progress monitor has already been started")
            self._started_at = monotonic()
            self._publication_total = publication_count
            self._version_total = version_count
            self._input_description = input_description
            for stage in STAGES:
                self._stages[stage]["total"] = (
                    version_count
                    if stage in {"gather", "hierarchy", "normalization"}
                    else publication_count
                )

        self._logger.info(
            "m2_progress_start input=%s publications=%d versions=%d "
            "interval_seconds=%.1f",
            input_description,
            publication_count,
            version_count,
            self.interval,
        )
        if not self.enabled:
            return
        # Snapshots are interactive terminal output, not event-log records.
        # The detailed stage events continue to go through ``target_logger``.
        output = self._output or sys.stdout
        self._monitor = M1ProgressMonitor(
            self.snapshot,
            interval=self.interval,
            output=output,
        )
        self._monitor.start()

    def set_publication(self, arxiv_id: str) -> None:
        """Set the publication currently being processed."""

        with self._lock:
            self._current_publication = arxiv_id
            self._current_version = "-"
            self._current_stage = "publication"

    def set_stage(
        self,
        stage: str,
        *,
        arxiv_id: str,
        version: int | None = None,
        message: str = "running",
    ) -> None:
        """Record the input and module currently active."""

        if stage not in self._stages:
            raise ValueError(f"unknown M2 progress stage: {stage}")
        versioned_id = arxiv_id if version is None else f"{arxiv_id}:v{version}"
        with self._lock:
            self._current_publication = arxiv_id
            self._current_version = versioned_id
            self._current_stage = stage
            self._stages[stage]["current"] = versioned_id
            self._stages[stage]["input"] = message
            self._stages[stage]["message"] = message

    def complete(
        self,
        stage: str,
        *,
        failed: bool = False,
        skipped: bool = False,
        message: str | None = None,
        reference_count: int | None = None,
    ) -> None:
        """Mark one input as processed by a stage."""

        if stage not in self._stages:
            raise ValueError(f"unknown M2 progress stage: {stage}")
        with self._lock:
            state = self._stages[stage]
            state["completed"] += 1
            if failed:
                state["failed"] += 1
            if skipped:
                state["skipped"] += 1
            if message is not None:
                state["message"] = message
            if reference_count is not None:
                self._reference_count = max(0, reference_count)

    def finish_publication(self, *, status: str) -> None:
        """Update aggregate publication counters after export/statistics."""

        if status not in {"success", "partial", "failed"}:
            raise ValueError("publication status must be success, partial, or failed")
        with self._lock:
            self._publication_completed += 1
            self._publication_success += status == "success"
            self._publication_partial += status == "partial"
            self._publication_failed += status == "failed"

    def snapshot(self) -> dict[str, object]:
        """Return the M1-compatible aggregate snapshot."""

        with self._lock:
            started_at = self._started_at
            stages = {
                name: dict(state) for name, state in self._stages.items()
            }
            current_stage = self._current_stage
            current_publication = self._current_publication
            current_version = self._current_version
            publication_total = self._publication_total
            version_total = self._version_total
            publication_completed = self._publication_completed
            publication_success = self._publication_success
            publication_partial = self._publication_partial
            publication_failed = self._publication_failed
            reference_count = self._reference_count

        elapsed = 0.0 if started_at is None else max(0.0, monotonic() - started_at)
        workers: dict[str, dict[str, object]] = {}
        for stage, state in stages.items():
            workers[stage] = {
                "versioned_id": state["current"],
                "current": state["completed"],
                "total": state["total"],
                "message": (
                    f"{state['input']} | {state['message']} "
                    f"failed={state['failed']} "
                    f"skipped={state['skipped']}"
                ),
            }

        return {
            "elapsed_seconds": elapsed,
            "discovery": {
                "status": current_stage,
                "found": publication_completed,
                "total": publication_total,
                "batch": 0,
                "input": self._input_description,
            },
            "source": {
                "completed_papers": publication_completed,
                "total_papers": publication_total,
                "versions_completed": stages["gather"]["completed"],
                "total_versions": version_total,
                "papers_success": publication_success,
                "papers_partial": publication_partial,
                "papers_failed": publication_failed,
                "worker_order": list(STAGES),
                "worker_progress": workers,
                "current_publication": current_publication,
                "current_version": current_version,
            },
            "reference": {
                "completed_papers": stages["references"]["completed"],
                "total_papers": publication_total,
                "references": reference_count,
                "queue_size": 0,
                "queue_maxsize": 0,
            },
        }

    def stop(self, *, render_final: bool = True) -> None:
        """Stop the renderer; safe to call when progress is disabled."""

        if self._monitor is not None:
            self._monitor.stop(render_final=render_final)
            self._monitor = None
        self._logger.info(
            "m2_progress_stop publications=%d/%d versions=%d/%d",
            self.snapshot()["source"]["completed_papers"],
            self._publication_total,
            self.snapshot()["source"]["versions_completed"],
            self._version_total,
        )


__all__ = ["STAGES", "M2ProgressMonitor"]
