from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from src.crawling import pipeline


class SourceDownloadDefaultsTests(unittest.TestCase):
    def test_cli_defaults_to_attached_parallel_download_strategy(self) -> None:
        args = pipeline.build_parser().parse_args(["--ids-file", "ids.txt"])

        self.assertEqual(args.download_workers, 4)
        self.assertIsNone(args.request_interval)


class InputTests(unittest.TestCase):
    def test_ranges_csv_skips_header_and_reads_multiple_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranges.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["start_month", "start_id", "end_month", "end_id"])
                writer.writerow(["2026-09", "1", "2026-09", "2"])
                writer.writerow(["2026-10", "3", "2026-10", "4"])

            loaded = pipeline.load_input(
                pipeline.InputRequest(ranges_file=path.resolve())
            )

        self.assertEqual(
            list(loaded.iter_unique_ids()),
            ["2609.00001", "2609.00002", "2610.00003", "2610.00004"],
        )
        self.assertEqual(loaded.kind, "ranges_file")
        self.assertEqual(loaded.input_path, str(path.resolve()))
        self.assertEqual(loaded.ranges_count, 2)
        self.assertEqual(loaded.candidate_count, 4)
        self.assertEqual(loaded.unique_count, 4)

    def test_overlapping_ranges_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranges.csv"
            path.write_text(
                "start_month,start_id,end_month,end_id\n"
                "2026-09,1,2026-09,3\n"
                "2026-09,2,2026-09,4\n",
                encoding="utf-8",
            )
            loaded = pipeline.load_input(pipeline.InputRequest(ranges_file=path))

        self.assertEqual(loaded.candidate_count, 6)
        self.assertEqual(loaded.unique_count, 4)
        self.assertEqual(
            list(loaded.iter_unique_ids()),
            ["2609.00001", "2609.00002", "2609.00003", "2609.00004"],
        )

    def test_ranges_file_accepts_an_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = (Path(tmp) / "ranges.csv").resolve()
            path.write_text(
                "start_month,start_id,end_month,end_id\n2026-09,1,2026-09,1\n",
                encoding="utf-8",
            )
            loaded = pipeline.load_input(pipeline.InputRequest(ranges_file=str(path)))

        self.assertEqual(loaded.input_path, str(path))
        self.assertEqual(list(loaded.iter_unique_ids()), ["2609.00001"])

    def test_limit_is_a_found_paper_limit_for_cli_ranges(self) -> None:
        loaded = pipeline.load_input(
            pipeline.InputRequest(
                start_month="2026-09",
                start_id=1,
                end_month="2026-09",
                end_id=10,
                limit=3,
            )
        )

        self.assertEqual(loaded.candidate_count, 10)
        self.assertEqual(loaded.unique_count, 10)
        self.assertEqual(loaded.found_limit, 3)
        self.assertEqual(
            list(loaded.iter_unique_ids()),
            [
                "2609.00001",
                "2609.00002",
                "2609.00003",
                "2609.00004",
                "2609.00005",
                "2609.00006",
                "2609.00007",
                "2609.00008",
                "2609.00009",
                "2609.00010",
            ],
        )

    def test_ids_file_ignores_blank_lines_comments_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text(
                "# assigned IDs\n\n arXiv:2609.22845v2 \n"
                "https://arxiv.org/abs/2609.22846\n",
                encoding="utf-8",
            )
            loaded = pipeline.load_input(pipeline.InputRequest(ids_file=path.resolve()))

        self.assertEqual(
            list(loaded.iter_unique_ids()),
            ["2609.22845", "2609.22846"],
        )
        self.assertEqual(len(loaded.fingerprint), 64)

    def test_invalid_ranges_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranges.csv"
            path.write_text(
                "start_month,start_id,end_month,end_id\n2026-13,1,2026-13,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "month"):
                pipeline.load_input(pipeline.InputRequest(ranges_file=path))

    def test_empty_ids_file_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("# no IDs\n\n", encoding="utf-8")
            loaded = pipeline.load_input(pipeline.InputRequest(ids_file=path))

        self.assertEqual(loaded.candidate_count, 0)
        self.assertEqual(loaded.unique_count, 0)
        self.assertEqual(list(loaded.iter_unique_ids()), [])

    def test_cli_found_limit_spans_adjacent_months(self) -> None:
        loaded = pipeline.load_input(
            pipeline.InputRequest(
                start_month="2026-07",
                start_id=99998,
                end_month="2026-08",
                end_id=3,
                limit=3,
            )
        )

        self.assertEqual(loaded.candidate_count, 5)
        self.assertEqual(loaded.unique_count, 5)
        self.assertEqual(loaded.found_limit, 3)
        self.assertEqual(loaded.range_spec.start_month, "2026-07")
        self.assertEqual(loaded.range_spec.end_month, "2026-08")


class DiscoveryBatchTests(unittest.TestCase):
    @staticmethod
    def _http_error(code: int, retry_after: str | None = None) -> HTTPError:
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        return HTTPError("https://export.arxiv.org", code, "error", headers, None)

    def test_found_and_not_found_are_distinguished(self) -> None:
        outcome = pipeline.discover_batch(
            ["2609.00001", "2609.00002"],
            lambda _ids: [
                {
                    "arxiv_id": "2609.00001",
                    "title": "Found",
                    "authors": ["A"],
                    "submission_date": "2026-09-01",
                    "revised_dates": [],
                    "venue": None,
                    "versions": ["v1"],
                }
            ],
            policy=pipeline.RetryPolicy(max_retries=0),
        )

        self.assertEqual(
            [record["status"] for record in outcome.records],
            ["found", "not_found"],
        )
        self.assertIn("metadata", outcome.records[0])
        self.assertNotIn("metadata", outcome.records[1])

    def test_http_429_is_retried_and_retry_after_is_respected(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def fetch(_ids: list[str]) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise self._http_error(429, "17")
            return [{"arxiv_id": "2609.00001", "versions": ["v1"]}]

        outcome = pipeline.discover_batch(
            ["2609.00001"],
            fetch,
            policy=pipeline.RetryPolicy(
                max_retries=2,
                base_delay=1,
                max_delay=100,
                jitter_ratio=0,
            ),
            sleep=sleeps.append,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [17.0])
        self.assertEqual(outcome.http_429_count, 1)
        self.assertEqual(outcome.retry_count, 1)
        self.assertEqual(outcome.records[0]["status"], "found")

    def test_found_record_contains_downstream_metadata(self) -> None:
        metadata = {"arxiv_id": "2609.00001", "versions": ["v1", "v2"]}
        outcome = pipeline.discover_batch(
            ["2609.00001"],
            lambda _ids: [metadata],
            policy=pipeline.RetryPolicy(max_retries=0),
        )
        self.assertEqual(outcome.records[0]["status"], "found")
        self.assertEqual(outcome.records[0]["metadata"], metadata)

    def test_successful_omission_is_not_found(self) -> None:
        outcome = pipeline.discover_batch(
            ["2609.00001"],
            lambda _ids: [],
            policy=pipeline.RetryPolicy(max_retries=0),
        )
        self.assertEqual(outcome.records[0]["status"], "not_found")
        self.assertFalse(outcome.records[0]["retryable"])

    def test_http_503_uses_exponential_backoff(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def fetch(_ids: list[str]) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise self._http_error(503)
            return [{"arxiv_id": "2609.00001", "versions": ["v1"]}]

        outcome = pipeline.discover_batch(
            ["2609.00001"],
            fetch,
            policy=pipeline.RetryPolicy(
                max_retries=2,
                base_delay=10,
                max_delay=300,
                jitter_ratio=0,
            ),
            sleep=sleeps.append,
        )

        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [10.0, 20.0])
        self.assertEqual(outcome.http_503_count, 2)
        self.assertEqual(outcome.retry_sleep_seconds, 30.0)

    def test_request_failure_is_retryable_not_not_found(self) -> None:
        outcome = pipeline.discover_batch(
            ["2609.00001"],
            lambda _ids: (_ for _ in ()).throw(TimeoutError("slow")),
            policy=pipeline.RetryPolicy(max_retries=0),
        )

        record = outcome.records[0]
        self.assertEqual(record["status"], "retryable_failed")
        self.assertTrue(record["retryable"])
        self.assertEqual(record["error_type"], "timeout")

    def test_permanent_http_error_is_not_retried(self) -> None:
        calls = 0

        def fetch(_ids: list[str]) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            raise self._http_error(403)

        outcome = pipeline.discover_batch(
            ["2609.00001"],
            fetch,
            policy=pipeline.RetryPolicy(max_retries=5),
        )

        self.assertEqual(calls, 1)
        self.assertEqual(outcome.records[0]["status"], "permanent_failed")
        self.assertFalse(outcome.records[0]["retryable"])

    def test_circuit_breaker_pauses_after_repeated_429(self) -> None:
        calls = 0

        def fetch(_ids: list[str]) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            raise self._http_error(429)

        outcome = pipeline.discover_batch(
            ["2609.00001"],
            fetch,
            policy=pipeline.RetryPolicy(
                max_retries=10,
                base_delay=0,
                jitter_ratio=0,
                circuit_breaker_threshold=2,
                circuit_cooldown=30,
                pause_after_429=3,
            ),
            sleep=lambda _seconds: None,
            wall_time=lambda: 1000.0,
        )

        self.assertEqual(calls, 3)
        self.assertTrue(outcome.paused_rate_limit)
        self.assertEqual(outcome.records[0]["status"], "rate_limited")
        self.assertEqual(outcome.consecutive_429, 3)
        self.assertEqual(outcome.cooldown_until_epoch, 1030.0)


class _SourceFake:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self.events = events

    def process(self, metadata: dict[str, object]) -> dict[str, object]:
        arxiv_id = str(metadata["arxiv_id"])
        self.calls.append(arxiv_id)
        if self.events is not None:
            self.events.append(f"source:{arxiv_id}")
        return {
            "source": {"status": "success"},
            "processing": {"status": "success"},
            "metadata": {"status": "success"},
        }


class _PrefetchSourceFake:
    def __init__(self) -> None:
        self.prefetch_calls: list[str] = []
        self.process_calls: list[str] = []
        self.processing_started = threading.Event()
        self.second_prefetch_finished = threading.Event()

    def prefetch(self, metadata: dict[str, object]) -> pipeline.PrefetchedSource:
        arxiv_id = str(metadata["arxiv_id"])
        self.prefetch_calls.append(arxiv_id)
        if arxiv_id.endswith("00002"):
            self.assert_processing_overlap = self.processing_started.wait(1.0)
            self.second_prefetch_finished.set()
        return pipeline.PrefetchedSource(metadata=metadata, downloads={})

    def process_prefetched(
        self, prefetched: pipeline.PrefetchedSource
    ) -> dict[str, object]:
        arxiv_id = str(prefetched.metadata["arxiv_id"])
        self.process_calls.append(arxiv_id)
        if arxiv_id.endswith("00001"):
            self.processing_started.set()
            self.assert_download_overlap = self.second_prefetch_finished.wait(1.0)
        return {
            "source": {"status": "success"},
            "processing": {"status": "success"},
            "metadata": {"status": "success"},
        }


class PersistenceAndBarrierTests(unittest.TestCase):
    def _loaded(self, root: Path, *ids: str) -> pipeline.LoadedInput:
        path = root / "ids.txt"
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        return pipeline.load_input(pipeline.InputRequest(ids_file=path))

    def _config(
        self,
        root: Path,
        *,
        retry_policy: pipeline.RetryPolicy | None = None,
    ) -> pipeline.PipelineConfig:
        return pipeline.PipelineConfig(
            run_dir=root / "run",
            corpus_dir=root / "corpus",
            discovery_batch_size=2,
            source_batch_size=1,
            checkpoint_every=1,
            fsync_every=1,
            min_free_gb=0,
            grace_period_seconds=0,
            without_references=True,
            retry_policy=retry_policy or pipeline.RetryPolicy(max_retries=0),
            benchmark_sample_interval=0.01,
        )

    @staticmethod
    def _paper(arxiv_id: str) -> dict[str, object]:
        return {
            "arxiv_id": arxiv_id,
            "title": "Paper",
            "authors": ["A"],
            "submission_date": "2026-09-01",
            "revised_dates": [],
            "venue": None,
            "versions": ["v1"],
        }

    def test_checkpoint_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"
            pipeline.atomic_write_json(path, {"status": "running"})

            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_discovery_summary_is_throttled_but_force_flushes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                checkpoint_every=3,
                min_free_gb=0,
                without_references=True,
            )
            runner = pipeline.Pipeline(
                config,
                self._loaded(root, "2609.00001"),
                discovery_fetch=lambda values: [self._paper(values[0])],
            )
            runner._prepare(resume=False)
            with patch.object(runner, "_write_summary") as write_summary:
                runner._checkpoint_summary("running", phase="discovery")
                runner._checkpoint_summary("running", phase="discovery")
                self.assertEqual(write_summary.call_count, 0)

                runner._checkpoint_summary("running", phase="discovery")
                self.assertEqual(write_summary.call_count, 1)

                runner._checkpoint_summary("paused_rate_limit", force=True, phase="discovery")
                self.assertEqual(write_summary.call_count, 2)

    def test_pipeline_initializes_project_logging_when_handlers_are_missing(
        self,
    ) -> None:
        app_logger = logging.getLogger("arxiv_project")
        previous_handlers = app_logger.handlers[:]
        for handler in previous_handlers:
            app_logger.removeHandler(handler)
        try:
            with patch("src.crawling.pipeline.setup_logging") as setup_logging:
                pipeline._ensure_logging()
                setup_logging.assert_called_once_with()
        finally:
            for handler in app_logger.handlers[:]:
                app_logger.removeHandler(handler)
            for handler in previous_handlers:
                app_logger.addHandler(handler)

    def test_reference_log_handler_is_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                without_references=False,
                min_free_gb=0,
            )
            runner = pipeline.Pipeline(config, loaded)
            with patch.dict(os.environ, {"S2_API_KEY_1": "test-key"}):
                pipeline._ensure_logging()
                crawler = runner._create_reference_crawler()
                try:
                    reference_logger = logging.getLogger("ReferenceCrawler")
                    handlers = reference_logger.handlers
                    common_handlers = logging.getLogger("arxiv_project").handlers
                    self.assertTrue(
                        any(
                            isinstance(handler, logging.FileHandler)
                            and Path(handler.baseFilename).name
                            == "reference_crawler.log"
                            for handler in handlers
                        )
                    )
                    self.assertTrue(
                        any(
                            isinstance(handler, logging.FileHandler)
                            and handler in handlers
                            for handler in common_handlers
                        )
                    )
                    marker = f"pipeline-reference-log-{id(runner)}"
                    reference_logger.info(marker)
                    for handler in handlers:
                        handler.flush()
                    common_file = next(
                        handler
                        for handler in common_handlers
                        if isinstance(handler, logging.FileHandler)
                    )
                    reference_file = next(
                        handler
                        for handler in handlers
                        if isinstance(handler, logging.FileHandler)
                        and Path(handler.baseFilename).name == "reference_crawler.log"
                    )
                    self.assertIn(
                        marker,
                        Path(common_file.baseFilename).read_text(encoding="utf-8"),
                    )
                    self.assertIn(
                        marker,
                        Path(reference_file.baseFilename).read_text(encoding="utf-8"),
                    )
                finally:
                    runner.reference_crawler = crawler
                    runner._close_reference_crawler()

    def test_terminal_summary_is_compact(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline._print_terminal_summary(
                {
                    "status": "completed",
                    "found_count": 1,
                    "not_found_count": 0,
                    "candidate_id_count": 1,
                    "source_success_count": 1,
                    "reference_success_count": 1,
                    "failed_ids": {"discovery": [], "source": [], "reference": []},
                    "discovery_time_seconds": 0.1,
                    "source_time_seconds": 0.2,
                    "reference_time_seconds": 0.3,
                    "wall_time_seconds": 0.4,
                    "peak_rss_mb": 50.0,
                    "disk_growth_bytes": 100,
                },
                Path("run"),
            )
        rendered = output.getvalue()
        self.assertIn("Pipeline summary", rendered)
        self.assertIn("summary=run/summary.json", rendered)
        self.assertNotIn('"status"', rendered)

    def test_discovery_barrier_precedes_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            events: list[str] = []

            def fetch(ids: list[str]) -> list[dict[str, object]]:
                events.append("discover:" + ",".join(ids))
                return [self._paper(arxiv_id) for arxiv_id in ids]

            source = _SourceFake(events)
            result = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=fetch,
                source_processor=source,
            ).run()

        first_source = next(
            i for i, value in enumerate(events) if value.startswith("source:")
        )
        last_discovery = max(
            i for i, value in enumerate(events) if value.startswith("discover:")
        )
        self.assertGreater(first_source, last_discovery)
        self.assertEqual(result["status"], "completed")


class _CrawlerFake:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.max_workers = 1
        self.batch_size = 1

    def process_batch(
        self, arxiv_ids: list[str], force_refresh: bool = False
    ) -> dict[str, object]:
        self.calls.append(list(arxiv_ids))
        return {
            "results": {
                arxiv_id: {
                    "status": "success",
                    "retryable": False,
                    "references_count": 1,
                }
                for arxiv_id in arxiv_ids
            }
        }


class DownstreamQueueTests(unittest.TestCase):
    _loaded = PersistenceAndBarrierTests._loaded
    _config = PersistenceAndBarrierTests._config
    _paper = staticmethod(PersistenceAndBarrierTests._paper)

    def _reference_config(
        self,
        root: Path,
        *,
        source_batch_size: int = 1,
        reference_batch_size: int = 1,
        queue_size: int = 2,
    ) -> pipeline.PipelineConfig:
        return pipeline.PipelineConfig(
            run_dir=root / "run",
            corpus_dir=root / "corpus",
            discovery_batch_size=10,
            source_batch_size=source_batch_size,
            reference_batch_size=reference_batch_size,
            reference_queue_size=queue_size,
            checkpoint_every=1,
            fsync_every=1,
            min_free_gb=0,
            grace_period_seconds=0,
            retry_policy=pipeline.RetryPolicy(max_retries=0),
            benchmark_sample_interval=0.01,
        )

    def test_source_batches_are_limited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"2609.{index:05d}" for index in range(1, 8)]
            loaded = self._loaded(root, *ids)
            runner = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=_SourceFake(),
            )
            runner.run()

        self.assertLessEqual(runner.max_source_batch_seen, 1)

    def test_cross_month_found_limit_moves_to_second_month(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = pipeline.load_input(
                pipeline.InputRequest(
                    start_month="2026-07",
                    start_id=99998,
                    end_month="2026-08",
                    end_id=3,
                    limit=3,
                )
            )

            def discovery_fetch(ids: list[str]) -> list[dict[str, object]]:
                if ids == ["2607.99998", "2607.99999"]:
                    return [
                        self._paper("2607.99998"),
                        self._paper("2607.99999"),
                    ]
                if ids[0] == "2608.00001":
                    return [self._paper("2608.00001")]
                return []

            config = self._config(root)
            result = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=discovery_fetch,
                source_processor=_SourceFake(),
            ).run()
            discovery_records = list(
                pipeline.iter_jsonl(root / "run" / "discovery.jsonl")
            )

        found_ids = [
            record["arxiv_id"]
            for record in discovery_records
            if record.get("status") == "found"
        ]
        self.assertEqual(
            found_ids,
            ["2607.99998", "2607.99999", "2608.00001"],
        )
        self.assertEqual(result["found_count"], 3)
        self.assertEqual(result["not_found_count"], 1)

    def test_cross_month_without_limit_uses_id_probe_for_month_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = pipeline.load_input(
                pipeline.InputRequest(
                    start_month="2026-07",
                    start_id=99998,
                    end_month="2026-08",
                    end_id=1,
                )
            )
            def discovery_fetch(ids: list[str]) -> list[dict[str, object]]:
                if ids == ["2607.99998", "2607.99999"]:
                    return [self._paper("2607.99998")]
                if ids == ["2608.00001"]:
                    return [self._paper("2608.00001")]
                return []

            with (
                patch(
                    "src.crawling.discovery.find_start_timestamp_month_1",
                    return_value="20260701000000",
                ),
                patch(
                    "src.crawling.discovery.fetch_date_stream_batch_once",
                    return_value={
                        "offset": 0,
                        "raw_count": 0,
                        "found_papers": [],
                        "failed_ids": [],
                    },
                ) as date_fetch,
            ):
                result = pipeline.Pipeline(
                    self._config(root),
                    loaded,
                    discovery_fetch=discovery_fetch,
                    source_processor=_SourceFake(),
                ).run()
            discovery_records = list(
                pipeline.iter_jsonl(root / "run" / "discovery.jsonl")
            )

        date_fetch.assert_not_called()
        statuses = {
            record["arxiv_id"]: record["status"]
            for record in discovery_records
        }
        self.assertEqual(
            statuses,
            {
                "2607.99998": "found",
                "2608.00001": "found",
            },
        )
        self.assertEqual(result["status"], "completed")

    def test_cross_month_probes_month_one_by_id_until_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = pipeline.load_input(
                pipeline.InputRequest(
                    start_month="2026-06",
                    start_id=100,
                    end_month="2026-07",
                    end_id=1,
                )
            )
            calls: list[list[str]] = []

            def discovery_fetch(ids: list[str]) -> list[dict[str, object]]:
                calls.append(list(ids))
                if ids[0] == "2606.00100":
                    return [self._paper(arxiv_id) for arxiv_id in ids]
                if ids == ["2607.00001"]:
                    return [self._paper("2607.00001")]
                return []

            config = pipeline.PipelineConfig(
                **{
                    **self._config(root).__dict__,
                    "discovery_batch_size": 10,
                }
            )

            with (
                patch(
                    "src.crawling.discovery.find_start_timestamp_month_1",
                    return_value="20260601000000",
                ),
                patch(
                    "src.crawling.discovery.fetch_date_stream_batch_once",
                    return_value={
                        "offset": 0,
                        "raw_count": 0,
                        "found_papers": [],
                        "failed_ids": [],
                    },
                ),
            ):
                result = pipeline.Pipeline(
                    config,
                    loaded,
                    discovery_fetch=discovery_fetch,
                    source_processor=_SourceFake(),
                ).run()

            records = list(pipeline.iter_jsonl(root / "run" / "discovery.jsonl"))

        self.assertEqual(
            [record["arxiv_id"] for record in records if record["status"] == "found"],
            [f"2606.{number:05d}" for number in range(100, 110)]
            + ["2607.00001"],
        )
        self.assertEqual(calls[0][0], "2606.00100")
        self.assertEqual(calls[0][-1], "2606.00109")
        self.assertEqual(calls[1][0], "2606.00110")
        self.assertEqual(calls[1][-1], "2606.00119")
        self.assertEqual(calls[2], ["2607.00001"])
        self.assertEqual(result["found_count"], 11)

    def test_cross_month_cli_initializes_default_discovery_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = pipeline.load_input(
                pipeline.InputRequest(
                    start_month="2026-07",
                    start_id=99998,
                    end_month="2026-08",
                    end_id=3,
                    limit=3,
                )
            )
            adapter = Mock()
            adapter.fetch.side_effect = lambda ids: [
                self._paper(value) for value in ids
            ]
            with (
                patch(
                    "src.crawling.discovery.find_start_timestamp_month_1",
                    return_value="20260701000000",
                ),
                patch(
                    "src.crawling.discovery.fetch_date_stream_batch_once",
                    return_value={
                        "offset": 0,
                        "raw_count": 0,
                        "found_papers": [],
                        "failed_ids": [],
                    },
                ),
                patch(
                    "src.crawling.pipeline.ArxivDiscoveryAdapter",
                    return_value=adapter,
                ),
            ):
                result = pipeline.Pipeline(
                    self._config(root),
                    loaded,
                    source_processor=_SourceFake(),
                ).run()

        self.assertEqual(result["found_count"], 3)
        self.assertGreaterEqual(adapter.fetch.call_count, 1)

    def test_cross_month_range_does_not_skip_intermediate_months(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = pipeline.load_input(
                pipeline.InputRequest(
                    start_month="2026-07",
                    start_id=99999,
                    end_month="2026-09",
                    end_id=2,
                    limit=3,
                )
            )

            def remaining_month_fetch(ids: list[str]) -> list[dict[str, object]]:
                if ids[0] == "2608.00001":
                    return [self._paper("2608.00001"), self._paper("2608.00002")]
                if ids[0] == "2609.00001":
                    return [self._paper("2609.00001")]
                return []

            date_page = {
                "offset": 0,
                "raw_count": 0,
                "found_papers": [],
                "failed_ids": [],
            }
            config = self._config(root)
            config = pipeline.PipelineConfig(
                **{
                    **config.__dict__,
                    "discovery_batch_size": 100000,
                }
            )
            with (
                patch(
                    "src.crawling.discovery.find_start_timestamp_month_1",
                    return_value="20260701000000",
                ),
                patch(
                    "src.crawling.discovery.fetch_date_stream_batch_once",
                    return_value=date_page,
                ),
            ):
                result = pipeline.Pipeline(
                    config,
                    loaded,
                    discovery_fetch=remaining_month_fetch,
                    source_processor=_SourceFake(),
                ).run()
            records = list(pipeline.iter_jsonl(root / "run" / "discovery.jsonl"))

        self.assertEqual(
            [record["arxiv_id"] for record in records if record["status"] == "found"],
            ["2608.00001", "2608.00002", "2609.00001"],
        )
        self.assertEqual(result["found_count"], 3)

    def test_source_prefetch_overlaps_processing_with_bounded_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            source = _PrefetchSourceFake()
            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                source_batch_size=2,
                source_prefetch_queue_size=1,
                checkpoint_every=1,
                fsync_every=1,
                min_free_gb=0,
                grace_period_seconds=0,
                without_references=True,
                retry_policy=pipeline.RetryPolicy(max_retries=0),
                benchmark_sample_interval=0.01,
            )
            runner = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=source,
            )
            result = runner.run()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(source.prefetch_calls, ["2609.00001", "2609.00002"])
        self.assertEqual(source.process_calls, ["2609.00001", "2609.00002"])
        self.assertTrue(source.assert_processing_overlap)
        self.assertTrue(source.assert_download_overlap)
        self.assertEqual(runner.source_prefetch_queue_maxsize, 1)
        self.assertLessEqual(runner.source_prefetch_queue_observed_max, 1)
        self.assertEqual(result["source_prefetch_queue_maxsize"], 1)
        self.assertLessEqual(result["source_prefetch_queue_observed_max"], 1)

    def test_source_prefetch_rate_limit_persists_pending_ids(self) -> None:
        class PausingSource(_PrefetchSourceFake):
            def prefetch(self, metadata):
                if str(metadata["arxiv_id"]).endswith("00002"):
                    self.second_prefetch_finished.set()
                    raise pipeline.SourceRateLimitPaused("HTTP 429")
                return super().prefetch(metadata)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            source = PausingSource()
            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                source_batch_size=2,
                source_prefetch_queue_size=1,
                checkpoint_every=1,
                fsync_every=1,
                min_free_gb=0,
                grace_period_seconds=0,
                without_references=True,
                retry_policy=pipeline.RetryPolicy(max_retries=0),
                benchmark_sample_interval=0.01,
            )
            result = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=source,
            ).run()
            state = json.loads((root / "run" / "run_state.json").read_text())

        self.assertEqual(result["status"], "paused_rate_limit")
        self.assertEqual(state["source_prefetch_pending_ids"], ["2609.00002"])

    def test_multiple_papers_run_in_parallel_and_report_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(
                root,
                "2609.00001",
                "2609.00002",
                "2609.00003",
            )
            lock = threading.Lock()
            active = 0
            peak_active = 0
            both_started = threading.Event()

            class ParallelSource(_SourceFake):
                def process(self, metadata):
                    nonlocal active, peak_active
                    with lock:
                        active += 1
                        peak_active = max(peak_active, active)
                        if active >= 2:
                            both_started.set()
                    both_started.wait(1.0)
                    time.sleep(0.01)
                    result = super().process(metadata)
                    with lock:
                        active -= 1
                    return result

            config = pipeline.PipelineConfig(
                **{
                    **self._config(root).__dict__,
                    "source_batch_size": 3,
                    "download_workers": 2,
                }
            )
            result = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=ParallelSource(),
            ).run()

        self.assertEqual(result["status"], "completed")
        self.assertGreaterEqual(peak_active, 2)
        self.assertEqual(result["source_worker_count"], 2)
        self.assertEqual(result["source_progress"]["total_papers"], 3)
        self.assertEqual(result["source_progress"]["completed_papers"], 3)
        self.assertEqual(result["source_progress"]["papers_success"], 3)

    def test_reference_queue_is_bounded_and_calls_process_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002", "2609.00003")
            crawler = _CrawlerFake()
            runner = pipeline.Pipeline(
                self._reference_config(root, queue_size=2),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=_SourceFake(),
                reference_crawler=crawler,
            )
            result = runner.run()

        self.assertEqual(runner.reference_queue_maxsize, 2)
        self.assertLessEqual(runner.reference_queue_observed_max, 2)
        self.assertEqual(
            [value for batch in crawler.calls for value in batch],
            ["2609.00001", "2609.00002", "2609.00003"],
        )
        self.assertEqual(result["reference_success_count"], 3)

    def test_source_and_reference_overlap_in_phase_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            second_source_done = threading.Event()
            reference_started = threading.Event()

            class Source(_SourceFake):
                def process(self, metadata: dict[str, object]) -> dict[str, object]:
                    result = super().process(metadata)
                    if metadata["arxiv_id"] == "2609.00002":
                        second_source_done.set()
                    return result

            class Crawler(_CrawlerFake):
                def process_batch(
                    self, arxiv_ids: list[str], force_refresh: bool = False
                ) -> dict[str, object]:
                    reference_started.set()
                    if arxiv_ids == ["2609.00001"]:
                        self.assert_overlap = second_source_done.wait(1.0)
                    return super().process_batch(arxiv_ids, force_refresh)

            crawler = Crawler()
            result = pipeline.Pipeline(
                self._reference_config(root),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=Source(),
                reference_crawler=crawler,
            ).run()

        self.assertTrue(reference_started.is_set())
        self.assertTrue(crawler.assert_overlap)
        self.assertEqual(result["status"], "completed")

    def test_reference_starts_during_a_single_source_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            reference_started = threading.Event()

            class Source(_SourceFake):
                def process(self, metadata: dict[str, object]) -> dict[str, object]:
                    if metadata["arxiv_id"] == "2609.00002":
                        self.overlapped = reference_started.wait(1.0)
                    return super().process(metadata)

            class Crawler(_CrawlerFake):
                def process_batch(
                    self, arxiv_ids: list[str], force_refresh: bool = False
                ) -> dict[str, object]:
                    reference_started.set()
                    return super().process_batch(arxiv_ids, force_refresh)

            source = Source()
            result = pipeline.Pipeline(
                self._reference_config(
                    root, source_batch_size=2, reference_batch_size=1
                ),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=source,
                reference_crawler=Crawler(),
            ).run()

        self.assertTrue(source.overlapped)
        self.assertEqual(result["status"], "completed")

    def test_resume_skips_successful_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            crawler = _CrawlerFake()
            pipeline.Pipeline(
                self._reference_config(root),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=_SourceFake(),
                reference_crawler=crawler,
            ).run()

            class FailCrawler(_CrawlerFake):
                def process_batch(self, arxiv_ids, force_refresh=False):
                    raise AssertionError("successful reference must be skipped")

            resumed = pipeline.Pipeline(
                self._reference_config(root),
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    AssertionError("discovery must be skipped")
                ),
                source_processor=_SourceFake(),
                reference_crawler=FailCrawler(),
            ).run(resume=True)

        self.assertEqual(crawler.calls, [["2609.00001"]])
        self.assertEqual(resumed["status"], "completed")

    def test_source_results_are_journaled_before_entire_batch_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            loaded = self._loaded(root, "2609.00001", "2609.00002", "2609.00003")

            class StreamingSource(_SourceFake):
                def process(self, metadata):
                    if metadata["arxiv_id"] == "2609.00002":
                        records = list(pipeline.iter_jsonl(run_dir / "manifest.jsonl"))
                        self.first_was_persisted = any(
                            value.get("arxiv_id") == "2609.00001" for value in records
                        )
                    return super().process(metadata)

            source = StreamingSource()
            pipeline.Pipeline(
                self._reference_config(
                    root, source_batch_size=3, reference_batch_size=3
                ),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=source,
                reference_crawler=_CrawlerFake(),
            ).run()

        self.assertTrue(source.first_was_persisted)

    def test_reference_rate_limit_circuit_pauses_safely(self) -> None:
        class RateLimitedCrawler(_CrawlerFake):
            def process_batch(self, arxiv_ids, force_refresh=False):
                self.calls.append(list(arxiv_ids))
                return {
                    "results": {
                        value: {
                            "status": "retryable_failure",
                            "retryable": True,
                            "error_message": "HTTP 429 rate limit",
                            "references_count": 0,
                        }
                        for value in arxiv_ids
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002", "2609.00003")
            crawler = RateLimitedCrawler()
            result = pipeline.Pipeline(
                self._reference_config(root),
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=_SourceFake(),
                reference_crawler=crawler,
            ).run()

        self.assertEqual(len(crawler.calls), 3)
        self.assertEqual(result["status"], "paused_rate_limit")


class RuntimeAndRecoveryTests(unittest.TestCase):
    _loaded = PersistenceAndBarrierTests._loaded
    _config = PersistenceAndBarrierTests._config
    _paper = staticmethod(PersistenceAndBarrierTests._paper)

    def test_failed_ids_are_derived_by_stage_and_latest_status_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovery = root / "discovery.jsonl"
            manifest = root / "manifest.jsonl"
            discovery.write_text(
                json.dumps({"arxiv_id": "2609.00001", "status": "retryable_failed"})
                + "\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "arxiv_id": "2609.00002",
                        "source": {"status": "retryable_failed"},
                        "reference": {"status": "retryable_failed"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "arxiv_id": "2609.00002",
                        "reference": {"status": "success"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            failed = pipeline.failed_ids_by_stage(discovery, manifest)

        self.assertEqual(failed["discovery"], ["2609.00001"])
        self.assertEqual(failed["source"], ["2609.00002"])
        self.assertEqual(failed["reference"], [])

    def test_permanent_partial_source_is_terminal_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "arxiv_id": "2609.00001",
                        "source": {
                            "status": "partial_success",
                            "retryable": True,
                            "versions": [
                                {
                                    "download": {
                                        "status": "success",
                                        "retryable": False,
                                    },
                                    "processing": {"status": "ok"},
                                },
                                {
                                    "download": {
                                        "status": "permanent_failed",
                                        "retryable": False,
                                    },
                                    "processing": {"status": "not_started"},
                                },
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            index = pipeline._manifest_index(manifest)

        self.assertEqual(index["2609.00001"]["source_retryable"], "False")
        self.assertTrue(
            pipeline.Pipeline._stage_is_terminal(
                index["2609.00001"],
                "source",
                pipeline.Pipeline._SOURCE_TERMINAL,
            )
        )

    def test_keyboard_interrupt_persists_state_before_propagating(self) -> None:
        class InterruptingSource:
            def process(self, metadata):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            with self.assertRaises(KeyboardInterrupt):
                pipeline.Pipeline(
                    self._config(root),
                    loaded,
                    discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                    source_processor=InterruptingSource(),
                ).run()

            state = json.loads((root / "run" / "run_state.json").read_text())
            summary = json.loads((root / "run" / "summary.json").read_text())

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stop_reason"], "keyboard_interrupt")
        self.assertEqual(summary["status"], "failed")

    def test_low_disk_stops_before_discovery_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            calls = 0

            def fetch(_ids):
                nonlocal calls
                calls += 1
                return []

            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                min_free_gb=10**9,
                without_references=True,
                benchmark_sample_interval=0.01,
            )
            result = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=fetch,
                source_processor=_SourceFake(),
            ).run()

        self.assertEqual(calls, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["phase"], "discovery")

    def test_summary_separates_stage_times_and_resource_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")

            def fetch(ids: list[str]) -> list[dict[str, object]]:
                records = [self._paper(value) for value in ids]
                records[0]["versions"] = ["v1", "v2", "v3"]
                return records

            summary = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=fetch,
                source_processor=_SourceFake(),
            ).run()

        for name in (
            "discovery_time_seconds",
            "source_time_seconds",
            "reference_time_seconds",
            "reference_api_time_seconds",
            "reference_stage_wall_seconds",
            "reference_queue_wait_seconds",
            "wall_time_seconds",
            "disk_initial_used_bytes",
            "disk_peak_used_bytes",
            "disk_growth_bytes",
            "final_corpus_size_bytes",
            "final_output_size_bytes",
            "download_cache_size_bytes",
            "initial_rss_bytes",
            "peak_rss_mb",
            "average_rss_mb",
            "pdf_fallback_success_count",
            "source_versions_per_hour",
            "source_version_success_rate",
            "downloader_limiter_wait_seconds",
            "downloader_cache_hit_rate",
            "discovery_wall_seconds",
            "source_stage_wall_seconds",
            "reference_crawler_work_seconds",
            "source_id_duration_stats",
            "version_download_duration_stats",
            "version_processing_duration_stats",
        ):
            self.assertIn(name, summary)
        self.assertEqual(summary["total_versions"], 4)
        self.assertEqual(summary["average_versions_per_id"], 2.0)
        self.assertEqual(summary["average_versions_per_found_id"], 2.0)
        self.assertGreaterEqual(summary["wall_time_seconds"], 0)

    def test_reference_timing_separates_crawler_work_and_queue_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001", "2609.00002")
            first_reference_done = threading.Event()

            class SlowSource(_SourceFake):
                def process(self, metadata: dict[str, object]) -> dict[str, object]:
                    if metadata["arxiv_id"] == "2609.00002":
                        first_reference_done.wait(1.0)
                        time.sleep(0.03)
                    return super().process(metadata)

            class TimedCrawler(_CrawlerFake):
                def process_batch(
                    self, arxiv_ids: list[str], force_refresh: bool = False
                ) -> dict[str, object]:
                    time.sleep(0.01)
                    result = super().process_batch(arxiv_ids, force_refresh)
                    if arxiv_ids == ["2609.00001"]:
                        first_reference_done.set()
                    return result

            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                source_batch_size=1,
                reference_batch_size=1,
                reference_queue_size=2,
                checkpoint_every=1,
                fsync_every=1,
                min_free_gb=0,
                grace_period_seconds=0,
                retry_policy=pipeline.RetryPolicy(max_retries=0),
                benchmark_sample_interval=0.01,
            )
            summary = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda values: [self._paper(value) for value in values],
                source_processor=SlowSource(),
                reference_crawler=TimedCrawler(),
            ).run()

        self.assertGreater(summary["reference_time_seconds"], 0.0)
        self.assertGreater(
            summary["reference_stage_wall_seconds"],
            summary["reference_time_seconds"],
        )
        self.assertGreater(summary["reference_queue_wait_seconds"], 0.01)

    def test_empty_input_completes_with_only_minimal_run_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root)
            result = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    AssertionError("empty input must not call discovery")
                ),
                source_processor=_SourceFake(),
            ).run()
            names = sorted(path.name for path in (root / "run").iterdir())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_papers"], 0)
        self.assertEqual(
            names,
            [
                "discovery.jsonl",
                "input.json",
                "manifest.jsonl",
                "run_state.json",
                "summary.json",
            ],
        )


class SourceAdapterTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "arxiv_id": "2609.00001",
            "title": "Paper",
            "authors": ["A"],
            "submission_date": "2026-09-01",
            "revised_dates": [],
            "venue": None,
            "versions": ["v1"],
        }

    def _config(self, root: Path) -> pipeline.PipelineConfig:
        return pipeline.PipelineConfig(
            run_dir=root / "run",
            corpus_dir=root / "corpus",
            download_cache_dir=root / "cache",
            min_free_gb=0,
            without_references=True,
            retry_policy=pipeline.RetryPolicy(max_retries=0),
        )

    def test_default_source_adapter_downloads_then_processes_tex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"\\documentclass{article}"
                info = tarfile.TarInfo("main.tex")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus", cache, self._config(root)
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True],
                ),
                patch("src.crawling.downloader.download_version", return_value=archive),
            ):
                result = adapter.process(self._metadata())

            extracted = root / "corpus" / "2609-00001" / "v1" / "main.tex"
            self.assertEqual(result["source"]["status"], "success")
            self.assertEqual(result["processing"]["status"], "success")
            self.assertTrue(extracted.exists())
            self.assertFalse(archive.exists())

    def test_default_source_adapter_prefetches_before_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"\\documentclass{article}"
                info = tarfile.TarInfo("main.tex")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus", cache, self._config(root)
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True],
                ),
                patch("src.crawling.downloader.download_version", return_value=archive),
            ):
                prefetched = adapter.prefetch(self._metadata())
                result = adapter.process_prefetched(prefetched)

            extracted = root / "corpus" / "2609-00001" / "v1" / "main.tex"
            self.assertEqual(prefetched.metadata["arxiv_id"], "2609.00001")
            self.assertIn("2609.00001v1", prefetched.downloads)
            self.assertEqual(result["source"]["status"], "success")
            self.assertTrue(extracted.exists())
            self.assertFalse(archive.exists())

    def test_default_source_adapter_emits_version_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"\\documentclass{article}"
                info = tarfile.TarInfo("main.tex")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            events: list[dict[str, object]] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                self._config(root),
                progress_callback=events.append,
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True],
                ),
                patch("src.crawling.downloader.download_version", return_value=archive),
            ):
                result = adapter.process(self._metadata())

        self.assertEqual(result["source"]["status"], "success")
        self.assertIn("source_seconds", result["timings"])
        self.assertIn("total_seconds", result["timings"])
        self.assertEqual(
            [event["event"] for event in events],
            ["paper_started", "version_started", "version_processed"],
        )
        self.assertEqual(events[-1]["versioned_id"], "2609.00001v1")

    def test_source_retry_uses_configured_rate_limit_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            error = HTTPError(
                url="https://export.arxiv.org/src/2609.00001v1",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            )
            sleeps: list[float] = []
            metrics: list[tuple[str, float | int]] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                pipeline.PipelineConfig(
                    **{
                        **self._config(root).__dict__,
                        "retry_policy": pipeline.RetryPolicy(
                            max_retries=1,
                            base_delay=1,
                            max_delay=10,
                            jitter_ratio=0,
                            rate_limit_cooldown=20,
                        ),
                    }
                ),
                metric_callback=lambda name, value: metrics.append((name, value)),
                sleep=sleeps.append,
            )
            with (
                patch("src.crawling.downloader.is_download_complete", side_effect=[False, True]),
                patch(
                    "src.crawling.downloader.download_version",
                    side_effect=[error, archive],
                ),
            ):
                result, path = adapter._download("2609.00001v1")

        self.assertEqual(result["status"], "success")
        self.assertEqual(path, archive)
        self.assertEqual(sleeps, [20])
        self.assertIn(("downloader_http_429_count", 1), metrics)
        self.assertIn(("downloader_retry_count", 1), metrics)
        self.assertIn(("downloader_rate_limit_sleep_seconds", 20), metrics)

    def test_source_rate_limit_window_survives_success_between_429s(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            first_archive = cache / "2609.00001v1.source"
            second_archive = cache / "2609.00002v1.source"
            rate_limit = HTTPError(
                "https://export.arxiv.org/src/2609.00001v1",
                429,
                "Too Many Requests",
                {},
                None,
            )
            sleeps: list[float] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                pipeline.PipelineConfig(
                    **{
                        **self._config(root).__dict__,
                        "retry_policy": pipeline.RetryPolicy(
                            max_retries=1,
                            base_delay=1,
                            max_delay=10,
                            jitter_ratio=0,
                            circuit_breaker_threshold=2,
                            circuit_cooldown=30,
                            rate_limit_cooldown=5,
                        ),
                    }
                ),
                sleep=sleeps.append,
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True, False, True],
                ),
                patch(
                    "src.crawling.downloader.download_version",
                    side_effect=[rate_limit, first_archive, rate_limit, second_archive],
                ),
            ):
                first_result, _ = adapter._download("2609.00001v1")
                second_result, _ = adapter._download("2609.00002v1")

        self.assertEqual(first_result["status"], "success")
        self.assertEqual(second_result["status"], "success")
        self.assertEqual(sleeps, [5, 30])

    def test_source_rate_limit_cooldown_adds_post_cooldown_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            error = HTTPError(
                "https://export.arxiv.org/src/2609.00001v1",
                429,
                "Too Many Requests",
                {},
                None,
            )
            sleeps: list[float] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                pipeline.PipelineConfig(
                    **{
                        **self._config(root).__dict__,
                        "retry_policy": pipeline.RetryPolicy(
                            max_retries=1,
                            base_delay=1,
                            max_delay=10,
                            jitter_ratio=0.5,
                            rate_limit_cooldown=10,
                        ),
                    }
                ),
                sleep=sleeps.append,
                random_value=lambda: 0.5,
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True],
                ),
                patch(
                    "src.crawling.downloader.download_version",
                    side_effect=[error, archive],
                ),
            ):
                result, _ = adapter._download("2609.00001v1")

        self.assertEqual(result["status"], "success")
        self.assertEqual(sleeps, [12.5])

    def test_default_source_adapter_keeps_no_tex_distinct_from_network_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            archive = cache / "2609.00001v1.source"
            archive.write_bytes(b"%PDF-1.7\n")
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus", cache, self._config(root)
            )
            with (
                patch(
                    "src.crawling.downloader.is_download_complete",
                    side_effect=[False, True],
                ),
                patch("src.crawling.downloader.download_version", return_value=archive),
            ):
                result = adapter.process(self._metadata())

        self.assertEqual(result["source"]["status"], "success_with_no_tex")
        self.assertEqual(result["processing"]["status"], "success_with_no_tex")

    def test_source_404_uses_one_pdf_fallback_and_keeps_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            pdf = cache / "pdf-fallback" / "2609.00001v1.pdf"
            pdf.parent.mkdir()
            pdf.write_bytes(b"%PDF-1.7\n")
            source_error = HTTPError(
                "https://export.arxiv.org/src/2609.00001v1",
                404,
                "Not Found",
                {},
                None,
            )
            metrics: list[tuple[str, float | int]] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                self._config(root),
                metric_callback=lambda name, value: metrics.append((name, value)),
            )
            with (
                patch("src.crawling.downloader.is_download_complete", return_value=False),
                patch("src.crawling.downloader.download_version", side_effect=source_error),
                patch("src.crawling.downloader.download_pdf", return_value=pdf) as fallback,
            ):
                result = adapter.process(self._metadata())
                pdf_exists = pdf.exists()

        fallback.assert_called_once_with(
            "2609.00001v1",
            cache / "pdf-fallback",
            limiter=adapter.limiter,
        )
        self.assertEqual(result["source"]["status"], "success_with_no_tex")
        self.assertEqual(result["processing"]["status"], "success_with_no_tex")
        version = result["source"]["versions"][0]
        self.assertEqual(version["download"]["status"], "pdf_only")
        self.assertEqual(version["processing"]["source_type"], "pdf")
        self.assertTrue(pdf_exists)
        self.assertIn(("pdf_fallback_success_count", 1), metrics)

    def test_pdf_fallback_429_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            pdf = cache / "pdf-fallback" / "2609.00001v1.pdf"
            pdf.parent.mkdir()
            source_error = HTTPError(
                "https://export.arxiv.org/src/2609.00001v1",
                404,
                "Not Found",
                {},
                None,
            )
            pdf_rate_limit = HTTPError(
                "https://export.arxiv.org/pdf/2609.00001v1",
                429,
                "Too Many Requests",
                {"Retry-After": "7"},
                None,
            )
            sleeps: list[float] = []
            adapter = pipeline.DefaultSourceProcessor(
                root / "corpus",
                cache,
                pipeline.PipelineConfig(
                    **{
                        **self._config(root).__dict__,
                        "retry_policy": pipeline.RetryPolicy(
                            max_retries=1,
                            base_delay=1,
                            max_delay=10,
                            jitter_ratio=0,
                            rate_limit_cooldown=5,
                        ),
                    }
                ),
                sleep=sleeps.append,
            )
            with (
                patch("src.crawling.downloader.is_download_complete", return_value=False),
                patch("src.crawling.downloader.download_version", side_effect=source_error),
                patch(
                    "src.crawling.downloader.download_pdf",
                    side_effect=[pdf_rate_limit, pdf],
                ) as fallback,
            ):
                result, path = adapter._download("2609.00001v1")

        self.assertEqual(result["status"], "pdf_only")
        self.assertEqual(path, pdf)
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(sleeps, [7.0])


class ResumeAndDiscoveryTests(unittest.TestCase):
    _loaded = PersistenceAndBarrierTests._loaded
    _config = PersistenceAndBarrierTests._config
    _paper = staticmethod(PersistenceAndBarrierTests._paper)

    def test_retryable_source_failure_is_retried_after_completed_run(self) -> None:
        class RetryableSource(_SourceFake):
            def process(self, metadata):
                self.calls.append(str(metadata["arxiv_id"]))
                return {
                    "source": {"status": "retryable_failed", "retryable": True},
                    "processing": {"status": "not_started"},
                    "metadata": {"status": "success"},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            failed_source = RetryableSource()
            first = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                source_processor=failed_source,
            ).run()

            successful_source = _SourceFake()
            second = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    AssertionError("discovery must not rerun")
                ),
                source_processor=successful_source,
            ).run(resume=True)

        self.assertEqual(first["status"], "completed_with_failures")
        self.assertEqual(failed_source.calls, ["2609.00001"])
        self.assertEqual(successful_source.calls, ["2609.00001"])
        self.assertEqual(second["status"], "completed")

    def test_retryable_reference_failure_is_retried_on_resume(self) -> None:
        class RetryCrawler(_CrawlerFake):
            def process_batch(self, arxiv_ids, force_refresh=False):
                self.calls.append(list(arxiv_ids))
                return {
                    "results": {
                        value: {
                            "status": "retryable_failure",
                            "retryable": True,
                            "error_message": "temporary",
                            "references_count": 0,
                        }
                        for value in arxiv_ids
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            config = DownstreamQueueTests()._reference_config(root)
            retry_crawler = RetryCrawler()
            first = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                source_processor=_SourceFake(),
                reference_crawler=retry_crawler,
            ).run()
            success_crawler = _CrawlerFake()
            second = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=lambda _ids: [],
                source_processor=_SourceFake(),
                reference_crawler=success_crawler,
            ).run(resume=True)

        self.assertEqual(first["status"], "completed_with_failures")
        self.assertEqual(success_crawler.calls, [["2609.00001"]])
        self.assertEqual(second["status"], "completed")

    def test_discovery_splits_only_after_original_batch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"2609.{index:05d}" for index in range(1, 7)]
            loaded = self._loaded(root, *ids)
            sizes: list[int] = []

            def fetch(values: list[str]) -> list[dict[str, object]]:
                sizes.append(len(values))
                if len(values) > 2:
                    raise DiscoveryBatchTests._http_error(503)
                return [self._paper(value) for value in values]

            config = pipeline.PipelineConfig(
                run_dir=root / "run",
                corpus_dir=root / "corpus",
                discovery_batch_size=6,
                discovery_fallback_batch_size=2,
                source_batch_size=2,
                min_free_gb=0,
                grace_period_seconds=0,
                without_references=True,
                retry_policy=pipeline.RetryPolicy(max_retries=0),
                benchmark_sample_interval=0.01,
            )
            result = pipeline.Pipeline(
                config,
                loaded,
                discovery_fetch=fetch,
                source_processor=_SourceFake(),
            ).run()

        self.assertEqual(sizes, [6, 2, 2, 2])
        self.assertEqual(result["found_count"], 6)
        self.assertEqual(result["status"], "completed")

    def test_retryable_discovery_failure_is_persisted_and_blocks_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            source = _SourceFake()
            result = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    TimeoutError("slow")
                ),
                source_processor=source,
            ).run()

            state = json.loads((root / "run" / "run_state.json").read_text())
            record = next(pipeline.iter_jsonl(root / "run" / "discovery.jsonl"))

        self.assertEqual(result["status"], "discovery_incomplete")
        self.assertFalse(state["discovery_complete"])
        self.assertEqual(state["phase"], "discovery")
        self.assertEqual(state["discovery_retry"]["pending_ids"], ["2609.00001"])
        self.assertEqual(record["status"], "retryable_failed")
        self.assertEqual(source.calls, [])

    def test_discovery_pause_can_resume_then_start_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            paused_config = self._config(
                root,
                retry_policy=pipeline.RetryPolicy(
                    max_retries=5,
                    base_delay=0,
                    max_delay=0,
                    jitter_ratio=0,
                    circuit_breaker_threshold=1,
                    circuit_cooldown=0,
                    pause_after_429=2,
                ),
            )
            first_source = _SourceFake()
            first = pipeline.Pipeline(
                paused_config,
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    DiscoveryBatchTests._http_error(429)
                ),
                source_processor=first_source,
                sleep=lambda _seconds: None,
            ).run()

            resumed_source = _SourceFake()
            second = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                source_processor=resumed_source,
            ).run(resume=True)

        self.assertEqual(first["status"], "paused_rate_limit")
        self.assertEqual(first_source.calls, [])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(resumed_source.calls, ["2609.00001"])

    def test_resume_rejects_input_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_input = self._loaded(root, "2609.00001")
            pipeline.Pipeline(
                self._config(root),
                first_input,
                discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                source_processor=_SourceFake(),
            ).run()

            second_path = root / "different.txt"
            second_path.write_text("2609.00002\n", encoding="utf-8")
            second_input = pipeline.load_input(
                pipeline.InputRequest(ids_file=second_path)
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                pipeline.Pipeline(
                    self._config(root),
                    second_input,
                    discovery_fetch=lambda _ids: [],
                    source_processor=_SourceFake(),
                ).run(resume=True)

    def test_resume_skips_completed_discovery_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._loaded(root, "2609.00001")
            source = _SourceFake()
            pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda ids: [self._paper(value) for value in ids],
                source_processor=source,
            ).run()

            resumed_source = _SourceFake()
            result = pipeline.Pipeline(
                self._config(root),
                loaded,
                discovery_fetch=lambda _ids: (_ for _ in ()).throw(
                    AssertionError("discovery must be skipped")
                ),
                source_processor=resumed_source,
            ).run(resume=True)

        self.assertEqual(source.calls, ["2609.00001"])
        self.assertEqual(resumed_source.calls, [])
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
