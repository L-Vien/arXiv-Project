from __future__ import annotations

import json

from src.crawling.reference_enrichment import (
    apply_candidate_metadata,
    enrich_reference_documents,
)


def test_apply_candidate_metadata_uses_arxiv_submission_without_overwriting_s2_publication_data():
    references = [
        {
            "ref_arxiv_id": "arXiv:1706.03762v7",
            "ref_submission_date": None,
            "ref_publication_date": "2018-06-01",
            "ref_publication_year": 2018,
            "ref_venue": "NeurIPS",
        },
        {"ref_arxiv_id": None, "ref_submission_date": None},
    ]
    candidates = {
        "1706.03762": {
            "submission_date": "2017-06-12",
            "updated_date": "2023-08-02",
            "doi": "10.48550/arXiv.1706.03762",
            "venue": "arXiv",
        }
    }

    enriched, updated = apply_candidate_metadata(references, candidates)

    assert updated == 1
    assert enriched[0]["ref_arxiv_id"] == "1706.03762"
    assert enriched[0]["ref_submission_date"] == "2017-06-12"
    assert enriched[0]["ref_arxiv_last_updated_date"] == "2023-08-02"
    assert enriched[0]["ref_doi"] == "10.48550/arxiv.1706.03762"
    assert enriched[0]["ref_publication_date"] == "2018-06-01"
    assert enriched[0]["ref_publication_year"] == 2018
    assert enriched[0]["ref_venue"] == "NeurIPS"
    assert enriched[1]["ref_submission_date"] is None


def test_enrichment_reuses_persistent_candidate_cache(tmp_path):
    paper = tmp_path / "2401.00001"
    paper.mkdir()
    (paper / "references.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "references": [{"ref_arxiv_id": "1706.03762", "ref_submission_date": None}],
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fetcher(ids: list[str]):
        calls.append(ids)
        return {arxiv_id: {"submission_date": "2017-06-12"} for arxiv_id in ids}

    first = enrich_reference_documents(tmp_path, fetcher=fetcher)
    second = enrich_reference_documents(tmp_path, fetcher=fetcher)

    assert first["fetched_candidates"] == 1
    assert second["fetched_candidates"] == 0
    assert second["cache_hits"] == 1
    assert calls == [["1706.03762"]]
