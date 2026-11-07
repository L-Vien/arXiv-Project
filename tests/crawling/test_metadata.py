import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.crawling.metadata import (
    base_arxiv_id, normalize_paper_metadata, paper_dir_name,
    parse_arxiv_id, save_metadata, validate_metadata, version_dir,
)


class TestNormalize(unittest.TestCase):
    def test_valid_data(self):
        raw = {"arxiv_id": "2609.07834", "title": "Valid Paper",
               "authors": ["Author A", "Author B"], "submission_date": "2026-10-01",
               "revised_dates": ["2026-10-01"], "venue": "NeurIPS"}
        res = normalize_paper_metadata(raw)
        self.assertEqual(res["authors"], ["Author A", "Author B"])
        self.assertEqual(res["venue"], "NeurIPS")
        self.assertEqual(set(res), {"arxiv_id", "title", "authors", "submission_date", "revised_dates", "venue"})

    def test_invalid_data(self):
        res = normalize_paper_metadata({"arxiv_id": "  2609.9999  ", "title": "T",
                                        "authors": "Single Author", "garbage_field": "x"})
        self.assertEqual(res["arxiv_id"], "2609.9999")
        self.assertEqual(res["authors"], ["Single Author"])
        self.assertNotIn("garbage_field", res)
        self.assertEqual(res["revised_dates"], [])

    def test_none_values_do_not_become_string_none(self):
        res = normalize_paper_metadata({"arxiv_id": "2310.12345", "title": None,
                                        "submission_date": None, "authors": None, "venue": None})
        self.assertEqual(res["title"], "")
        self.assertEqual(res["submission_date"], "")
        self.assertEqual(res["authors"], [])
        self.assertIsNone(res["venue"])

    def test_versioned_id_becomes_base_id(self):
        self.assertEqual(normalize_paper_metadata({"arxiv_id": "2310.12345v2"})["arxiv_id"], "2310.12345")

    def test_datetime_to_iso_and_dedupe_dates(self):
        dt = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
        res = normalize_paper_metadata({"arxiv_id": "2310.12345", "submission_date": dt,
                                        "revised_dates": [dt, dt.isoformat(), "", None]})
        self.assertEqual(res["submission_date"], "2026-10-01T12:00:00+00:00")
        self.assertEqual(res["revised_dates"], ["2026-10-01T12:00:00+00:00"])

    def test_author_objects_and_blank_authors(self):
        class A:
            def __str__(self): return "Obj Author"
        res = normalize_paper_metadata({"arxiv_id": "2310.12345", "authors": [A(), "  ", "B"]})
        self.assertEqual(res["authors"], ["Obj Author", "B"])

    def test_title_whitespace_collapsed_and_blank_venue(self):
        res = normalize_paper_metadata({"arxiv_id": "2310.12345", "title": " A\n  B ", "venue": "  "})
        self.assertEqual(res["title"], "A B")
        self.assertIsNone(res["venue"])


class TestIds(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_arxiv_id("2310.12345v2"), ("2310.12345", 2))
        self.assertEqual(parse_arxiv_id("2310.12345"), ("2310.12345", None))
        self.assertEqual(parse_arxiv_id("arXiv:2310.1234"), ("2310.1234", None))
        self.assertEqual(parse_arxiv_id("https://arxiv.org/abs/2310.12345v3"), ("2310.12345", 3))
        self.assertEqual(parse_arxiv_id("hep-th/9901001v1"), ("hep-th/9901001", 1))

    def test_invalid(self):
        for bad in ["", "abc", "23.12345", "2310.123456789"]:
            with self.assertRaises(ValueError):
                parse_arxiv_id(bad)

    def test_dir_names(self):
        self.assertEqual(paper_dir_name("2310.12345"), "2310-12345")
        self.assertEqual(paper_dir_name("2310.12345v2"), "2310-12345")
        self.assertEqual(paper_dir_name("hep-th/9901001"), "hep-th_9901001")
        self.assertEqual(base_arxiv_id("2310.12345v7"), "2310.12345")

    def test_version_dir(self):
        self.assertEqual(version_dir("/c", "2310.12345", 2), Path("/c/2310-12345/v2"))


class TestSave(unittest.TestCase):
    def test_save_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = save_metadata({"arxiv_id": "2310.12345v1", "title": "T", "authors": ["A"]},
                                 Path(d) / "2310-12345")
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(data["arxiv_id"], "2310.12345")
            self.assertEqual(data["authors"], ["A"])
            self.assertFalse(list(Path(d).rglob("*.tmp")))


class TestValidate(unittest.TestCase):
    GOOD = {"arxiv_id": "2310.12345", "title": "T", "authors": ["A"],
            "submission_date": "2026-10-01", "revised_dates": [], "venue": None}

    def _write(self, d, obj):
        p = Path(d) / "metadata.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    def test_valid_after_save(self):
        with tempfile.TemporaryDirectory() as d:
            save_metadata({**self.GOOD}, d)
            self.assertTrue(validate_metadata(d))

    def test_missing_file_or_bad_json(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(validate_metadata(d))
            (Path(d) / "metadata.json").write_text("{dở dang", encoding="utf-8")
            self.assertFalse(validate_metadata(d))

    def test_missing_or_empty_fields(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ({k: v for k, v in self.GOOD.items() if k != "title"},
                        {**self.GOOD, "title": ""},
                        {**self.GOOD, "authors": []},
                        {**self.GOOD, "arxiv_id": "abc"}):
                self.assertFalse(validate_metadata(self._write(d, bad)))


if __name__ == "__main__":
    unittest.main()
