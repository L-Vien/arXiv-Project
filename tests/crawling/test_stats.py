"""Tests for src.crawling.stats.

Run:
    python -m unittest tests.crawling.test_stats -v
"""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.crawling import stats


def make_version(
    versioned_id: str = "2401.12345v1",
    status: str = "success",
    download_status: str = "downloaded",
    processing_status: str = "ok",
    *,
    archive_size: int = 100,
    raw_size: int = 200,
    kept_size: int = 50,
    download_seconds: float = 1.0,
    processing_seconds: float = 2.0,
) -> dict:
    return {
        "versioned_id": versioned_id,
        "status": status,
        "download": {
            "status": download_status,
            "path": None,
            "error": None,
        },
        "processing": {
            "status": processing_status,
            "archive_size": archive_size,
            "raw_size": raw_size,
            "kept_size": kept_size,
            "kept_files": [],
            "error": None,
        },
        "timings": {
            "download_seconds": download_seconds,
            "processing_seconds": processing_seconds,
            "total_seconds": (
                download_seconds + processing_seconds
            ),
        },
        "errors": [],
    }


def make_paper(
    arxiv_id: str = "2401.12345",
    status: str = "success",
    *,
    metadata_status: str = "success",
    versions: list[dict] | None = None,
    reference_status: str = "success",
    reference_count: int = 3,
    reference_cache_hit: bool = False,
    total_seconds: float = 5.0,
    metadata_seconds: float = 0.5,
    references_seconds: float = 1.0,
    errors: list[dict] | None = None,
    overall_status: str | None = None,
) -> dict:
    if versions is None:
        versions = [make_version()]

    return {
        "arxiv_id": arxiv_id,
        "status": status,
        "overall_status": (
            overall_status
            if overall_status is not None
            else status
        ),
        "metadata_status": metadata_status,
        "metadata": None,
        "metadata_path": None,
        "versions": versions,
        "references": {
            "status": reference_status,
            "count": reference_count,
            "path": None,
            "error": None,
            "cache_hit": reference_cache_hit,
        },
        "timings": {
            "metadata_seconds": metadata_seconds,
            "versions_seconds": 3.5,
            "references_seconds": references_seconds,
            "total_seconds": total_seconds,
        },
        "errors": errors or [],
    }


class NumberHelperTests(unittest.TestCase):
    def test_number_accepts_finite_numbers(self):
        self.assertEqual(stats._number(5), 5.0)
        self.assertEqual(stats._number(1.5), 1.5)

    def test_number_rejects_invalid_values(self):
        invalid_values = [
            None,
            True,
            False,
            "123",
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                self.assertEqual(stats._number(value), 0.0)

    def test_ratio_and_percent_handle_zero_denominator(self):
        self.assertEqual(stats._ratio(5, 0), 0.0)
        self.assertEqual(stats._percent(5, 0), 0.0)

        self.assertEqual(stats._ratio(1, 4), 0.25)
        self.assertEqual(stats._percent(1, 4), 25.0)


class DirectorySizeTests(unittest.TestCase):
    def test_directory_size_counts_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "a.txt"
            second = root / "nested" / "b.txt"

            second.parent.mkdir()
            first.write_bytes(b"abc")
            second.write_bytes(b"12345")

            self.assertEqual(
                stats.directory_size_bytes(root),
                8,
            )

    def test_directory_size_returns_zero_for_missing_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing = Path(temporary_dir) / "missing"
            self.assertEqual(
                stats.directory_size_bytes(missing),
                0,
            )


class SummaryAccumulatorTests(unittest.TestCase):
    def test_empty_accumulator(self):
        accumulator = stats.SummaryAccumulator()
        summary = accumulator.finalize()

        self.assertEqual(summary["total_papers"], 0)
        self.assertEqual(summary["total_versions"], 0)
        self.assertEqual(summary["total_references"], 0)
        self.assertEqual(summary["paper_success_rate"], 0.0)
        self.assertEqual(summary["average_paper_seconds"], 0.0)
        self.assertEqual(
            summary["average_raw_bytes_per_version"],
            0.0,
        )

    def test_update_aggregates_papers_versions_references(self):
        paper_one = make_paper(
            arxiv_id="2401.12345",
            status="success",
            reference_count=3,
            reference_cache_hit=True,
            total_seconds=5.0,
            metadata_seconds=0.5,
            references_seconds=1.0,
            versions=[
                make_version(
                    archive_size=100,
                    raw_size=200,
                    kept_size=50,
                    download_seconds=1.0,
                    processing_seconds=2.0,
                )
            ],
        )

        paper_two = make_paper(
            arxiv_id="2401.12346",
            status="success_with_no_tex",
            reference_count=2,
            total_seconds=3.0,
            metadata_seconds=0.25,
            references_seconds=0.75,
            versions=[
                make_version(
                    versioned_id="2401.12346v1",
                    status="no_tex",
                    download_status="skipped",
                    processing_status="no_tex",
                    archive_size=40,
                    raw_size=60,
                    kept_size=0,
                    download_seconds=0.0,
                    processing_seconds=0.5,
                )
            ],
        )

        accumulator = stats.SummaryAccumulator()
        accumulator.update(paper_one)
        accumulator.update(paper_two)

        summary = accumulator.finalize(
            context={
                "required_paper_count": 2,
                "candidate_id_count": 10,
            }
        )

        self.assertEqual(summary["total_papers"], 2)
        self.assertEqual(summary["successful_papers"], 2)
        self.assertEqual(summary["total_versions"], 2)
        self.assertEqual(summary["successful_versions"], 1)
        self.assertEqual(summary["no_tex_versions"], 1)
        self.assertEqual(summary["no_tex_ids"], ["2401.12346"])
        self.assertEqual(summary["total_references"], 5)

        self.assertEqual(
            summary["fully_completed_papers"],
            2,
        )
        self.assertEqual(
            summary["fully_successful_papers"],
            1,
        )

        self.assertEqual(
            summary["reference_cache_hits"],
            1,
        )
        self.assertEqual(summary["downloaded_versions_per_second"], 1.0)
        self.assertEqual(summary["processed_versions_per_second"], 0.8)

        self.assertEqual(
            summary["total_raw_bytes_processed"],
            260,
        )
        self.assertEqual(
            summary["total_kept_bytes_processed"],
            50,
        )

        self.assertEqual(
            summary["average_raw_bytes_per_version"],
            130.0,
        )
        self.assertEqual(
            summary["average_kept_bytes_per_version"],
            25.0,
        )
        self.assertEqual(
            summary["average_raw_bytes_per_paper"],
            130.0,
        )
        self.assertEqual(
            summary["average_kept_bytes_per_paper"],
            25.0,
        )

        self.assertAlmostEqual(
            summary["kept_to_raw_ratio"],
            50 / 260,
        )

        self.assertAlmostEqual(
            summary["removed_data_ratio"],
            1 - (50 / 260),
        )

        self.assertEqual(
            summary["paper_success_rate"],
            1.0,
        )
        self.assertEqual(
            summary["paper_success_rate_percent"],
            100.0,
        )

        self.assertEqual(
            summary["version_success_rate"],
            0.5,
        )
        self.assertEqual(
            summary["reference_success_rate_attempted"],
            1.0,
        )

        self.assertEqual(
            summary["average_references_per_processed_paper"],
            2.5,
        )

        self.assertEqual(
            summary["cumulative_paper_seconds"],
            8.0,
        )
        self.assertEqual(
            summary["average_paper_seconds"],
            4.0,
        )
        self.assertEqual(
            summary["duration_stats"]["source_id"]["count"],
            2,
        )
        self.assertEqual(
            summary["duration_stats"]["source_id"]["total_seconds"],
            8.0,
        )
        self.assertEqual(
            summary["duration_stats"]["source_id"]["max_seconds"],
            5.0,
        )
        self.assertEqual(
            summary["duration_stats"]["version_download"]["total_seconds"],
            1.0,
        )
        self.assertEqual(
            summary["duration_stats"]["version_processing"]["average_seconds"],
            1.25,
        )

        self.assertEqual(
            summary["candidate_id_count"],
            10,
        )
        self.assertEqual(
            summary["required_paper_count"],
            2,
        )

    def test_failed_paper_and_error_stage_are_counted(self):
        paper = make_paper(
            arxiv_id="2401.12345",
            status="failed",
            metadata_status="failed",
            versions=[],
            reference_status="not_started",
            reference_count=0,
            errors=[
                {
                    "stage": "metadata",
                    "message": "permission denied",
                },
                {
                    "stage": "metadata",
                    "message": "metadata failed",
                },
            ],
        )

        accumulator = stats.SummaryAccumulator()
        accumulator.update(paper)

        summary = accumulator.finalize()

        self.assertEqual(summary["total_papers"], 1)
        self.assertEqual(summary["failed_papers"], 1)
        self.assertEqual(
            summary["metadata_failed_papers"],
            1,
        )
        self.assertEqual(
            summary["errors_by_stage"],
            {"metadata": 2},
        )

    def test_reference_success_rate_distinguishes_all_and_attempted(self):
        successful = make_paper(
            arxiv_id="2401.12345",
            reference_status="success",
            reference_count=4,
        )

        failed = make_paper(
            arxiv_id="2401.12346",
            status="failed",
            reference_status="failed",
            reference_count=0,
        )

        skipped = make_paper(
            arxiv_id="2401.12347",
            reference_status="skipped",
            reference_count=0,
        )

        summary = stats.summarize_results(
            [successful, failed, skipped]
        )

        self.assertAlmostEqual(
            summary["reference_success_rate_all_papers"],
            1 / 3,
        )
        self.assertEqual(
            summary["reference_success_rate_attempted"],
            0.5,
        )
        self.assertEqual(
            summary["reference_attempted_papers"],
            2,
        )

    def test_summary_separates_source_and_reference_failed_ids(self):
        source_failed = make_paper(
            arxiv_id="2401.12345",
            status="failed",
            reference_status="skipped",
            reference_count=0,
        )
        reference_failed = make_paper(
            arxiv_id="2401.12346",
            status="success",
            reference_status="failed",
            reference_count=0,
        )

        summary = stats.summarize_results(
            [source_failed, reference_failed]
        )

        self.assertEqual(
            summary["source_failed_ids"],
            ["2401.12345"],
        )
        self.assertEqual(
            summary["reference_failed_ids"],
            ["2401.12346"],
        )

    def test_cumulative_reference_seconds_uses_paper_timings(self):
        summary = stats.summarize_results(
            [
                make_paper(
                    "2401.12345",
                    references_seconds=1.25,
                ),
                make_paper(
                    "2401.12346",
                    references_seconds=0.75,
                ),
            ]
        )

        self.assertEqual(
            summary["cumulative_references_seconds"],
            2.0,
        )

    def test_summarize_results_accepts_generator(self):
        records = [
            make_paper("2401.12345"),
            make_paper("2401.12346"),
        ]

        generated_records = (
            record for record in records
        )

        summary = stats.summarize_results(
            generated_records
        )

        self.assertEqual(summary["total_papers"], 2)


class RunBenchmarkTests(unittest.TestCase):
    def test_stage_time_and_throughput(self):
        with patch.object(
            stats.RunBenchmark,
            "sample_memory",
        ), patch.object(
            stats.RunBenchmark,
            "sample_disk",
        ), patch.object(
            stats.RunBenchmark,
            "sample_directory_sizes",
        ), patch.object(
            stats.time,
            "perf_counter",
            side_effect=[
                10.0,  # constructor
                11.0,  # first measure start
                13.0,  # first measure end
                14.0,  # second measure start
                15.0,  # second measure end
                20.0,  # snapshot
            ],
        ):
            benchmark = stats.RunBenchmark()

            with benchmark.measure("discovery"):
                pass

            with benchmark.measure("discovery"):
                pass

            benchmark.add_stage_wall_time("source", 4.0)

            result = benchmark.snapshot(
                {
                    "total_papers": 2,
                    "total_versions": 4,
                },
                required_paper_count=2,
            )

        self.assertEqual(
            result["stage_wall_seconds"],
            {"source": 4.0},
        )
        self.assertEqual(
            result["stage_work_seconds"],
            {"discovery": 3.0},
        )
        self.assertEqual(
            result["source_time_seconds"],
            4.0,
        )
        self.assertEqual(
            result["wall_time_seconds"],
            10.0,
        )
        self.assertEqual(
            result["papers_per_second"],
            0.2,
        )
        self.assertEqual(
            result["versions_per_second"],
            0.4,
        )
        self.assertEqual(
            result["versions_per_hour"],
            1440.0,
        )
        self.assertEqual(
            result["discovery_wall_seconds"],
            0.0,
        )
        self.assertEqual(
            result["source_stage_wall_seconds"],
            4.0,
        )
        self.assertEqual(
            result[
                "end_to_end_average_seconds_per_required_paper"
            ],
            5.0,
        )

    def test_restore_stage_times_keeps_wall_and_work_separate(self):
        benchmark = stats.RunBenchmark()
        benchmark.restore_stage_times(
            {
                "stage_wall_seconds": {
                    "source": 4.0,
                },
                "stage_work_seconds": {
                    "source_processing": 7.0,
                },
            }
        )

        result = benchmark.snapshot(
            {
                "total_papers": 0,
                "total_versions": 0,
            }
        )

        self.assertEqual(
            result["stage_wall_seconds"],
            {"source": 4.0},
        )
        self.assertEqual(
            result["stage_work_seconds"],
            {"source_processing": 7.0},
        )

    def test_memory_sampling_tracks_peak_and_average(self):
        mib = 1024 * 1024
        rss_values = iter(
            [
                100 * mib,
                150 * mib,
                120 * mib,
            ]
        )

        fake_process = SimpleNamespace(
            memory_info=lambda: SimpleNamespace(
                rss=next(rss_values)
            )
        )

        fake_psutil = SimpleNamespace(
            Process=lambda: fake_process
        )

        with patch.object(
            stats,
            "psutil",
            fake_psutil,
        ), patch.object(
            stats,
            "resource",
            None,
        ):
            benchmark = stats.RunBenchmark()
            benchmark.sample_memory()

            result = benchmark.snapshot(
                {
                    "total_papers": 0,
                    "total_versions": 0,
                }
            )

        self.assertTrue(
            result["memory_sampling_available"]
        )
        self.assertEqual(
            result["memory_sample_count"],
            3,
        )
        self.assertEqual(
            result["peak_rss_mb"],
            150.0,
        )
        self.assertEqual(
            result["max_sampled_rss_mb"],
            150.0,
        )
        self.assertEqual(
            result["average_sampled_rss_mb"],
            round((100 + 150 + 120) / 3, 3),
        )

    def test_memory_sampling_without_psutil(self):
        with patch.object(stats, "psutil", None):
            benchmark = stats.RunBenchmark()
            result = benchmark.snapshot(
                {
                    "total_papers": 0,
                    "total_versions": 0,
                }
            )

        self.assertFalse(
            result["memory_sampling_available"]
        )
        self.assertIsNone(result["peak_rss_mb"])
        self.assertEqual(
            result["memory_sample_count"],
            0,
        )

    def test_disk_and_directory_metrics_are_returned(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "output"
            output_dir.mkdir()

            benchmark = stats.RunBenchmark(
                monitor_paths=[root],
                size_paths=[output_dir],
                sample_interval=0.01,
            )

            benchmark.start_sampling()

            first_file = output_dir / "first.bin"
            first_file.write_bytes(b"x" * 128)

            time.sleep(0.03)

            benchmark.stop_sampling()

            result = benchmark.snapshot(
                {
                    "total_papers": 0,
                    "total_versions": 0,
                }
            )

            key = str(output_dir)

            self.assertIn(
                key,
                result["directory_last_size_bytes"],
            )
            self.assertGreaterEqual(
                result["directory_last_size_bytes"][key],
                128,
            )
            self.assertIn(
                str(root),
                result["disk_peak_growth_bytes"],
            )
            self.assertIn(
                str(root),
                result["disk_min_free_bytes"],
            )

    def test_invalid_sample_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            stats.RunBenchmark(sample_interval=0)

        with self.assertRaises(ValueError):
            stats.RunBenchmark(sample_interval=-1)


class ProgressMonitorTests(unittest.TestCase):
    def test_render_includes_stage_progress_and_worker_state(self):
        output = io.StringIO()
        monitor = stats.ProgressMonitor(
            lambda: {
                "elapsed_seconds": 12.5,
                "discovery": {
                    "status": "completed",
                    "batch": 3,
                    "found": 4,
                    "total": 5,
                    "http_429": 1,
                },
                "source": {
                    "completed_papers": 2,
                    "total_papers": 4,
                    "versions_completed": 3,
                    "total_versions": 7,
                    "papers_success": 2,
                    "papers_partial": 0,
                    "papers_failed": 0,
                    "http_429": 1,
                    "worker_order": ["source-worker-0"],
                    "worker_progress": {
                        "source-worker-0": {
                            "paper": "2401.12345",
                            "versioned_id": "2401.12345v1",
                            "current": 1,
                            "total": 2,
                            "status": "downloading",
                            "message": "starting version download",
                        }
                    },
                },
                "reference": {
                    "completed_papers": 1,
                    "total_papers": 4,
                    "references": 9,
                    "queue_size": 1,
                    "queue_maxsize": 2,
                    "http_429": 0,
                },
            },
            interval=1.0,
            output=output,
        )

        monitor.render()
        rendered = output.getvalue()

        self.assertIn("Pipeline Snapshot", rendered)
        self.assertIn("Discovery: completed", rendered)
        self.assertIn("Source: 2/4 papers", rendered)
        self.assertIn("Reference: 1/4 papers", rendered)
        self.assertIn("2401.12345v1", rendered)


if __name__ == "__main__":
    unittest.main()
