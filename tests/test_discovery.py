import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discovery import extract_versions, parse_entry


def test_extract_versions_single_version():
    paper = Mock()
    paper.get_short_id.return_value = "2301.00001v1"

    assert extract_versions(paper) == ["v1"]


def test_extract_versions_multiple_versions():
    paper = Mock()
    paper.get_short_id.return_value = "2301.00001v3"

    assert extract_versions(paper) == ["v1", "v2", "v3"]


def test_parse_entry_metadata():
    paper = Mock()
    paper.title = "Example Paper"
    paper.authors = [Mock()]
    paper.authors[0].name = "Author A"
    paper.get_short_id.return_value = "2301.00001v1"
    paper.published.date.return_value.isoformat.return_value = "2023-01-02"
    paper.journal_ref = None

    metadata = parse_entry(paper, "2301.00001")

    assert metadata["arxiv_id"] == "2301.00001"
    assert metadata["title"] == "Example Paper"
    assert metadata["authors"] == ["Author A"]
    assert metadata["submission_date"] == "2023-01-02"
    assert metadata["venue"] is None