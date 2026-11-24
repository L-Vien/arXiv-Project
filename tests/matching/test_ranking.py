import pytest

from src.matching.contracts import BibtexRecord, FeatureRow, ReferenceRecord
from src.matching.ranking import (
    ScoredCandidate,
    TfidfTitleRanker,
    rank_candidates,
    rank_with_rule_baseline,
    rule_based_score,
)


def _feature(ref_id, values, citation_key="key"):
    return FeatureRow(
        arxiv_id="2609.07834",
        citation_key=citation_key,
        ref_arxiv_id=ref_id,
        partition="train",
        feature_schema_version="1.0",
        values=values,
    )


def test_rule_baseline_combines_features_and_orders_deterministic_ties():
    ranked = rank_with_rule_baseline(
        [
            _feature("1801.00002", {"title_jaccard": 0.5, "author_jaccard": 0.5}),
            _feature("1801.00001", {"title_jaccard": 0.5, "author_jaccard": 0.5}),
            _feature("1801.00003", {"doi_exact": True}),
        ]
    )

    assert [row.ref_arxiv_id for row in ranked[("2609.07834", "key")]] == [
        "1801.00001",
        "1801.00002",
        "1801.00003",
    ]
    assert rule_based_score({"year_abs_diff": 1}) == pytest.approx(0.05)


def test_ranking_deduplicates_and_breaks_ties_by_title_then_id():
    ranked = rank_candidates(
        [
            ScoredCandidate("2609.07834", "key", "1801.00002", 0.8, 0.7),
            ScoredCandidate("2609.07834", "key", "1801.00001", 0.8, 0.7),
            ScoredCandidate("2609.07834", "key", "1801.00003", 0.8, 0.9),
            ScoredCandidate("2609.07834", "key", "1801.00001", 0.7, 0.6),
        ]
    )

    assert [item.ref_arxiv_id for item in ranked] == [
        "1801.00003",
        "1801.00001",
        "1801.00002",
    ]


def test_tfidf_title_ranker_retrieves_matching_title_and_rejects_cross_publication():
    source_id = "2609.07834"
    bibtex = BibtexRecord(source_id, "key", "article", title="Neural networks for vision")
    training_bibtex = [BibtexRecord(source_id, "train", "article", title="Neural vision")]
    training_references = [
        ReferenceRecord(source_id, "1706.03762", ref_title="Neural networks for vision"),
        ReferenceRecord(source_id, "1801.00001", ref_title="Classical database systems"),
    ]
    ranker = TfidfTitleRanker().fit(training_bibtex, training_references)

    ranked = ranker.rank(bibtex, training_references)

    assert ranked[0].ref_arxiv_id == "1706.03762"
    with pytest.raises(ValueError, match="belong to the BibTeX publication"):
        ranker.rank(
            bibtex,
            [ReferenceRecord("2609.07835", "1801.00001", ref_title="Other")],
        )


def test_tfidf_ranker_requires_fit_and_training_titles():
    ranker = TfidfTitleRanker()
    bibtex = BibtexRecord("2609.07834", "key", "article")

    with pytest.raises(RuntimeError, match="fit"):
        ranker.rank(bibtex, [])
    with pytest.raises(ValueError, match="non-empty training title"):
        ranker.fit([], [])
