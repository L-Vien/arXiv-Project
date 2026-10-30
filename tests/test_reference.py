"""
Unit tests for reference_crawler.py (Milestone 1)
Description:
    Tests core components of ReferenceCrawler including APIKeyPool rotation,
    data standardization, saving logic, etc. using unittest.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import os
import json

# Ensure API key is present for initialization
os.environ.setdefault("S2_API_KEY_1", "test_key_123")

from reference_crawler import APIKeyPool, ReferenceCrawler


class TestAPIKeyPool(unittest.TestCase):
    """Test cases for APIKeyPool round-robin rotation and thread safety."""

    def test_key_rotation(self):
        keys = ["key1", "key2", "key3"]
        pool = APIKeyPool(keys)

        self.assertEqual(pool.get_key(), "key1")
        self.assertEqual(pool.get_key(), "key2")
        self.assertEqual(pool.get_key(), "key3")
        self.assertEqual(pool.get_key(), "key1")  # Wraps around

    def test_invalid_keys(self):
        with self.assertRaises(ValueError):
            APIKeyPool([])
        with self.assertRaises(ValueError):
            APIKeyPool(["   ", ""])


class TestReferenceCrawlerStandardization(unittest.TestCase):
    """Test cases for data standardization and local file operations in ReferenceCrawler."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.environ["S2_API_KEY_1"] = "dummy_key"
        self.crawler = ReferenceCrawler(corpus_dir=self.test_dir, log_dir=self.test_dir)

    def tearDown(self):
        if hasattr(self, 'crawler') and self.crawler:
            self.crawler.close()
        shutil.rmtree(self.test_dir)

    def test_standardize_references(self):
        arxiv_id = "2310.12345"
        raw_refs = [
            {
                "paperId": "s2_id_001",
                "title": "Quantum Error Correction",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "year": 2023,
                "publicationDate": "2023-05-10",
                "externalIds": {"ArXiv": "2301.00001"}
            },
            {
                "paperId": "s2_id_002",
                "title": "Another Paper",
                "authors": [],
                "year": 2021,
                "publicationDate": None,
                "externalIds": {}
            }
        ]

        std_refs = self.crawler.standardize_references(arxiv_id, raw_refs)

        self.assertEqual(len(std_refs), 2)

        # Check first reference
        ref1 = std_refs[0]
        self.assertEqual(ref1["arxiv_id"], arxiv_id)
        self.assertEqual(ref1["ref_arxiv_id"], "2301.00001")
        self.assertEqual(ref1["ref_title"], "Quantum Error Correction")
        self.assertEqual(ref1["ref_authors"], ["Alice", "Bob"])
        self.assertEqual(ref1["ref_submission_date"], "2023-05-10")
        self.assertEqual(ref1["ref_s2_id"], "s2_id_001")

        # Check second reference (missing externalIds / authors / date fallback to year)
        ref2 = std_refs[1]
        self.assertEqual(ref2["arxiv_id"], arxiv_id)
        self.assertIsNone(ref2["ref_arxiv_id"])
        self.assertEqual(ref2["ref_title"], "Another Paper")
        self.assertEqual(ref2["ref_authors"], [])
        self.assertEqual(ref2["ref_submission_date"], "2021")
        self.assertEqual(ref2["ref_s2_id"], "s2_id_002")

    def test_save_paper_references(self):
        arxiv_id = "2310.12345"
        sample_data = [{"arxiv_id": arxiv_id, "ref_title": "Test Ref"}]

        saved_path = self.crawler.save_paper_references(arxiv_id, sample_data)
        self.assertTrue(saved_path.exists())

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, sample_data)


if __name__ == "__main__":
    unittest.main()
