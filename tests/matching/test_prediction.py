import pytest

from src.matching.contracts import BibtexRecord, LabelRecord
from src.matching.prediction import format_predictions
from src.matching.ranking import ScoredCandidate


def _label(citation_key, truth):
    return LabelRecord(
        arxiv_id="2609.07834",
        citation_key=citation_key,
        groundtruth_arxiv_id=truth,
        label_source="manual",
        status="match" if truth else "no_match",
    )


def test_format_predictions_fills_top_five_and_preserves_no_candidate_rows():
    entries = [
        BibtexRecord("2609.07834", "second", "article"),
        BibtexRecord("2609.07834", "first", "article"),
    ]
    candidates = [
        ScoredCandidate("2609.07834", "first", "1801.00001", 0.9),
        ScoredCandidate("2609.07834", "first", "1706.03762", 0.8),
        ScoredCandidate("2609.07834", "first", "1801.00002", 0.7),
        ScoredCandidate("2609.07834", "first", "1801.00003", 0.6),
        ScoredCandidate("2609.07834", "first", "1801.00004", 0.5),
        ScoredCandidate("2609.07834", "first", "1801.00005", 0.4),
    ]

    rows = format_predictions(
        entries,
        candidates,
        {"arXiv:2609.7834v2": "test"},
        labels=[_label("first", "1706.03762"), _label("second", None)],
    )

    assert [row.citation_key for row in rows] == ["first", "second"]
    assert [getattr(rows[0], f"cand_{index}") for index in range(1, 6)] == [
        "1801.00001",
        "1706.03762",
        "1801.00002",
        "1801.00003",
        "1801.00004",
    ]
    assert rows[0].groundtruth_arxiv_id == "1706.03762"
    assert rows[1].groundtruth_arxiv_id is None
    assert rows[1].cand_1 is None
    assert rows[0].partition.value == "test"


def test_format_predictions_rejects_candidate_for_unknown_bibtex_entry():
    with pytest.raises(ValueError, match="no matching BibTeX"):
        format_predictions(
            [BibtexRecord("2609.07834", "key", "article")],
            [ScoredCandidate("2609.07834", "other", "1706.03762", 1.0)],
            {"2609.07834": "test"},
        )


def test_format_predictions_rejects_uncertain_labels():
    uncertain = LabelRecord(
        arxiv_id="2609.07834",
        citation_key="key",
        groundtruth_arxiv_id=None,
        label_source="manual",
        status="uncertain",
    )
    with pytest.raises(ValueError, match="uncertain"):
        format_predictions(
            [BibtexRecord("2609.07834", "key", "article")],
            [],
            {"2609.07834": "test"},
            labels=[uncertain],
        )
