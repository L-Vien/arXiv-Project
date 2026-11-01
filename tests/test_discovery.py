import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import discovery


def make_paper(arxiv_id):
    paper = Mock()
    paper.title = f"Paper {arxiv_id}"
    paper.authors = []
    paper.get_short_id.return_value = f"{arxiv_id}v1"
    paper.published.date.return_value.isoformat.return_value = "2023-01-02"
    paper.journal_ref = None
    return paper


class TestDiscovery(unittest.TestCase):

    def test_extract_versions_single_version(self):
        paper = Mock()
        paper.get_short_id.return_value = "2301.00001v1"

        self.assertEqual(discovery.extract_versions(paper), ["v1"])

    def test_extract_versions_multiple_versions(self):
        paper = Mock()
        paper.get_short_id.return_value = "2301.00001v3"

        self.assertEqual(
            discovery.extract_versions(paper),
            ["v1", "v2", "v3"]
        )

    def test_get_base_id_removes_version_suffix(self):
        paper = Mock()
        paper.get_short_id.return_value = "2301.00001v3"

        self.assertEqual(discovery.get_base_id(paper), "2301.00001")

    def test_generate_arxiv_ids_across_year_boundary(self):
        arxiv_ids = list(
            discovery.generate_arxiv_ids(
                start_month=12,
                start_year=2023,
                end_month=1,
                end_year=2024,
                start_id=99999,
                end_id=2
            )
        )

        self.assertEqual(
            arxiv_ids,
            ["2312.99999", "2401.00001", "2401.00002"]
        )

    def test_generate_arxiv_ids_rejects_invalid_month(self):
        with self.assertRaises(ValueError):
            list(
                discovery.generate_arxiv_ids(
                    start_month=12,
                    start_year=2023,
                    end_month=13,
                    end_year=2024,
                    start_id=1,
                    end_id=2
                )
            )

    def test_parse_input_requires_month_arguments(self):
        with (
            patch.object(sys, "argv", ["discovery.py"]),
            patch.object(discovery, "logger")
        ):
            with self.assertRaises(SystemExit):
                discovery.parse_input()

    def test_parse_input_rejects_invalid_month(self):
        argv = [
            "discovery.py",
            "--start_month", "2023-12",
            "--start_id", "1",
            "--end_month", "2024-13",
            "--end_id", "2"
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(discovery, "logger")
        ):
            with self.assertRaises(SystemExit):
                discovery.parse_input()

    def test_parse_input_accepts_december_to_january_range(self):
        argv = [
            "discovery.py",
            "--start_month", "2023-12",
            "--start_id", "99999",
            "--end_month", "2024-01",
            "--end_id", "2"
        ]
        with patch.object(sys, "argv", argv):
            args = discovery.parse_input()

        self.assertEqual(args["start_month"], 12)
        self.assertEqual(args["end_month"], 1)
        self.assertEqual(args["end_year"], 2024)

    def test_fetch_batch_uses_configured_retry_limit(self):
        with (
            patch.object(discovery, "client") as client,
            patch.object(discovery.time, "sleep"),
            patch.object(discovery, "logger")
        ):
            client.results.side_effect = RuntimeError("request failed")

            with self.assertRaises(RuntimeError):
                discovery.fetch_batch(
                    ["2301.00001"],
                    max_retries=1
                )

        self.assertEqual(client.results.call_count, 2)

    def test_retry_limit_drops_after_missing_streak_and_recovers(self):
        arxiv_ids = [
            f"2301.{paper_number:05d}"
            for paper_number in range(1, 7)
        ]
        retry_limits = []

        def fake_fetch_batch(batch, max_retries):
            retry_limits.append(max_retries)

            if len(retry_limits) <= 2:
                return []

            return [make_paper(arxiv_id) for arxiv_id in batch]

        def fake_batches(ids):
            ids = list(ids)
            if not ids:
                return iter([])
            return iter([ids[:3], ids[3:4], ids[4:5], ids[5:]])

        with (
            patch.object(discovery, "batch_ids", side_effect=fake_batches),
            patch.object(discovery, "fetch_batch", side_effect=fake_fetch_batch),
            patch.object(discovery, "logger")
        ):
            results, failed_ids, not_found_ids = discovery.discover_ids(arxiv_ids)

        self.assertEqual(retry_limits, [3, 3, 1, 3])
        self.assertEqual(len(results), 2)
        self.assertEqual(failed_ids, [])
        self.assertEqual(not_found_ids, arxiv_ids[:4])

    def test_skips_month_after_more_than_200_consecutive_missing_ids(self):
        month_ids = [
            f"2401.{paper_number:05d}"
            for paper_number in range(1, 206)
        ]
        next_month_ids = ["2402.00001", "2402.00002"]
        batches = [month_ids[:201], month_ids[201:] + next_month_ids]
        fetched_batches = []

        def fake_batches(ids):
            ids = list(ids)
            return iter(batches if ids else [])

        def fake_fetch_batch(batch, max_retries):
            fetched_batches.append(batch)
            if len(fetched_batches) == 1:
                return []
            return [make_paper(arxiv_id) for arxiv_id in batch]

        with (
            patch.object(discovery, "batch_ids", side_effect=fake_batches),
            patch.object(discovery, "fetch_batch", side_effect=fake_fetch_batch),
            patch.object(discovery, "logger")
        ):
            results, failed_ids, not_found_ids = discovery.discover_ids(
                month_ids + next_month_ids
            )

        self.assertEqual(fetched_batches, [month_ids[:201], next_month_ids])
        self.assertEqual(not_found_ids, month_ids[:201])
        self.assertEqual(
            [paper["arxiv_id"] for paper in results],
            next_month_ids
        )
        self.assertEqual(failed_ids, [])

    def test_does_not_skip_to_month_outside_input_range(self):
        month_ids = [
            f"2401.{paper_number:05d}"
            for paper_number in range(1, 206)
        ]
        batches = [month_ids[:201], month_ids[201:]]
        fetched_batches = []

        def fake_batches(ids):
            ids = list(ids)
            return iter(batches if ids else [])

        def fake_fetch_batch(batch, max_retries):
            fetched_batches.append(batch)
            return []

        with (
            patch.object(discovery, "batch_ids", side_effect=fake_batches),
            patch.object(discovery, "fetch_batch", side_effect=fake_fetch_batch),
            patch.object(discovery, "logger")
        ):
            results, failed_ids, not_found_ids = discovery.discover_ids(
                month_ids
            )

        self.assertEqual(fetched_batches, [month_ids[:201]])
        self.assertEqual(not_found_ids, month_ids[:201])
        self.assertEqual(results, [])
        self.assertEqual(failed_ids, [])


if __name__ == "__main__":
    unittest.main()
