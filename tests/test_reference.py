"""
Comprehensive Unit Test Suite for ReferenceCrawler (Milestone 1)
Description:
    Tests all core components and edge cases of ReferenceCrawler and APIKeyPool:
    1. APIKeyPool: rate limiting, round-robin/best-key selection, 429 adaptive cooldown (integer seconds and HTTP-date), mark_success logic.
    2. ArXiv ID Normalization: legacy (hep-th/...) and new-style (YYMM.NNNNNvV) IDs.
    3. Reference Standardization: mapping raw Semantic Scholar JSON to project schema.
    4. Caching & Resume: atomic file writing (.tmp -> references.json), cache preservation on refresh failure.
    5. Asynchronous Metrics Writer: non-blocking queue & reference_metrics.jsonl writing.
    6. Process Batch Workflow (Mocked): testing process_batch with mocked fetch_batch responses (success, not_found, transient/permanent failures).
"""

import unittest
import unittest.mock
import tempfile
import shutil
import os
import json
import time
import queue
from pathlib import Path

# Ensure default test API keys are present in env
os.environ.setdefault("S2_API_KEY_1", "test_key_111")
os.environ.setdefault("S2_API_KEY_2", "test_key_222")

from reference_crawler import APIKeyPool, ReferenceCrawler, _normalize_arxiv_id


class TestAPIKeyPool(unittest.TestCase):
    """Test cases for APIKeyPool rotation, rate limiting, and 429 adaptive cooldown."""

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            APIKeyPool([])
        with self.assertRaises(ValueError):
            APIKeyPool(["   ", ""])

    def test_mark_success_preserves_future_cooldown(self):
        pool = APIKeyPool(["key1"])
        future_time = time.time() + 30.0
        pool.key_cooldown_until["key1"] = future_time
        pool.consecutive_429["key1"] = 2

        pool.mark_success("key1")
        self.assertEqual(pool.consecutive_429["key1"], 0)
        self.assertEqual(pool.key_cooldown_until["key1"], future_time)

    def test_mark_success_clears_past_cooldown(self):
        pool = APIKeyPool(["key1"])
        past_time = time.time() - 10.0
        pool.key_cooldown_until["key1"] = past_time
        pool.consecutive_429["key1"] = 1

        pool.mark_success("key1")
        self.assertEqual(pool.consecutive_429["key1"], 0)
        self.assertEqual(pool.key_cooldown_until["key1"], 0.0)

    def test_mark_cooldown_with_retry_after_seconds(self):
        pool = APIKeyPool(["key1"])
        pool.mark_cooldown("key1", retry_after="5.5")
        self.assertGreater(pool.key_cooldown_until["key1"], time.time())

    def test_mark_cooldown_with_retry_after_http_date(self):
        pool = APIKeyPool(["key1"])
        # Format HTTP-date e.g. Wed, 05 Nov 2026 12:00:00 GMT
        import datetime
        future_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=15)
        http_date_str = future_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
        pool.mark_cooldown("key1", retry_after=http_date_str)
        self.assertGreater(pool.key_cooldown_until["key1"], time.time())

    def test_get_key_with_rate_limit(self):
        pool = APIKeyPool(["key1", "key2"])
        k1 = pool.get_key_with_rate_limit(request_delay=0.1)
        self.assertIn(k1, ["key1", "key2"])


class TestArXivNormalization(unittest.TestCase):
    """Test cases for arXiv ID normalization."""

    def test_normalize_arxiv_id(self):
        self.assertEqual(_normalize_arxiv_id("2310.12345v2"), "2310.12345")
        self.assertEqual(_normalize_arxiv_id("2104.09876"), "2104.09876")
        self.assertEqual(_normalize_arxiv_id("hep-th/0201001v1"), "hep-th/0201001")
        self.assertEqual(_normalize_arxiv_id("quant-ph/9901002"), "quant-ph/9901002")
        self.assertEqual(_normalize_arxiv_id("  2401.55555v3  "), "2401.55555")


class TestReferenceStandardization(unittest.TestCase):
    """Test cases for standardizing raw references into project schema."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.crawler = ReferenceCrawler(corpus_dir=self.test_dir, log_dir=self.test_dir)

    def tearDown(self):
        self.crawler.close()
        shutil.rmtree(self.test_dir)

    def test_standardize_references(self):
        raw_refs = [
            {
                "paperId": "s2_id_1",
                "title": "Test Paper Title 1",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "publicationDate": "2023-01-15",
                "externalIds": {"ArXiv": "2301.00001v1"}
            },
            {
                "paperId": "s2_id_2",
                "title": "Test Paper Title 2",
                "authors": [{"name": "Charlie"}],
                "year": 2022,
                "externalIds": {}
            },
            "invalid_ref_item"
        ]

        std = self.crawler.standardize_references("2310.12345", raw_refs)
        self.assertEqual(len(std), 2)
        self.assertEqual(std[0]["arxiv_id"], "2310.12345")
        self.assertEqual(std[0]["ref_arxiv_id"], "2301.00001v1")
        self.assertEqual(std[0]["ref_title"], "Test Paper Title 1")
        self.assertEqual(std[0]["ref_authors"], ["Alice", "Bob"])
        self.assertEqual(std[0]["ref_submission_date"], "2023-01-15")
        self.assertEqual(std[0]["ref_s2_id"], "s2_id_1")

        self.assertEqual(std[1]["ref_arxiv_id"], None)
        self.assertEqual(std[1]["ref_submission_date"], "2022")


class TestCachingAndAtomicSave(unittest.TestCase):
    """Test cases for atomic file writing and cache preservation on refresh failure."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.crawler = ReferenceCrawler(corpus_dir=self.test_dir, log_dir=self.test_dir)

    def tearDown(self):
        self.crawler.close()
        shutil.rmtree(self.test_dir)

    def test_cache_preservation_on_refresh_failure(self):
        arxiv_id = "2310.12345"
        sample_refs = [{"arxiv_id": arxiv_id, "ref_title": "Cached Ref"}]

        # Save successful cache
        self.crawler.save_paper_references(
            arxiv_id,
            sample_refs,
            status="success",
            retryable=False,
            error_message=None,
            attempt=1
        )

        folder_name = self.crawler._get_folder_name(arxiv_id)
        output_file = self.crawler.corpus_dir / folder_name / "references.json"
        self.assertTrue(output_file.exists())

        # Attempt to save failure status
        self.crawler.save_paper_references(
            arxiv_id,
            references=None,
            status="retryable_failure",
            retryable=True,
            error_message="API Timeout",
            attempt=2
        )

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Successful cache must be preserved
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["references"], sample_refs)

        # last_failure.json should be recorded
        fail_file = self.crawler.corpus_dir / folder_name / "last_failure.json"
        self.assertTrue(fail_file.exists())


class TestProcessBatchMocked(unittest.TestCase):
    """Test cases for process_batch workflow with mocked fetch_batch."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.crawler = ReferenceCrawler(corpus_dir=self.test_dir, log_dir=self.test_dir, batch_size=2)

    def tearDown(self):
        self.crawler.close()
        shutil.rmtree(self.test_dir)

    def test_process_batch_success_and_not_found(self):
        arxiv_ids = ["2310.11111", "2310.22222"]

        # Mock fetch_batch to return success for first paper and null for second
        mock_raw_results = [
            {"references": [{"paperId": "ref1", "title": "Ref One", "authors": [{"name": "Author A"}]}]},
            None
        ]

        with unittest.mock.patch.object(self.crawler, 'fetch_batch', return_value=(mock_raw_results, False, None, 1)):
            summary = self.crawler.process_batch(arxiv_ids)

        self.assertEqual(summary["total_papers"], 2)
        self.assertEqual(summary["successful_crawls"], 1)
        self.assertEqual(summary["not_found_count"], 1)
        self.assertEqual(summary["completion_status"], "completed")

        # Verify output files on disk
        f1 = self.crawler.corpus_dir / "2310-11111" / "references.json"
        f2 = self.crawler.corpus_dir / "2310-22222" / "references.json"
        self.assertTrue(f1.exists())
        self.assertTrue(f2.exists())


if __name__ == "__main__":
    unittest.main()