import json

import pytest

from src.matching.loading import (
    InputDataError,
    iter_bibtex_records,
    iter_bibtex_records_from_file,
    iter_matchable_reference_records,
    iter_matchable_reference_records_from_directory,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_bibtex_loader_streams_rows_and_applies_contract_normalization(tmp_path):
    _write_jsonl(
        tmp_path / "m2" / "range-01" / "bibtex.jsonl",
        [
            {
                "arxiv_id": "arXiv:2609.7834v2",
                "citation_key": "vaswani2017",
                "entry_type": "article",
                "title": " Attention Is All You Need ",
                "authors": [" Ashish Vaswani "],
                "year": 2017,
                "doi": "https://doi.org/10.48550/ARXIV.1706.03762",
                "journal": "",
                "url": "",
                "title_status": "present",
                "title_metadata_status": "metadata_available",
            }
        ],
    )

    records = list(iter_bibtex_records(tmp_path, "range-01"))

    assert records[0].arxiv_id == "2609.07834"
    assert records[0].authors == ("Ashish Vaswani",)
    assert records[0].doi == "10.48550/arxiv.1706.03762"
    assert records[0].journal is None
    assert records[0].title_status == "present"
    assert records[0].title_metadata_status == "metadata_available"


def test_bibtex_loader_accepts_an_explicit_file_path(tmp_path):
    path = tmp_path / "member-layout" / "normalized-bibtex.jsonl"
    _write_jsonl(
        path,
        [
            {
                "arxiv_id": "2609.07834",
                "citation_key": "key",
                "entry_type": "article",
            }
        ],
    )

    records = list(iter_bibtex_records_from_file(path))

    assert records[0].citation_key == "key"


def test_reference_loader_excludes_non_matchable_references(tmp_path):
    _write_json(
        tmp_path / "corpus" / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [
                {
                    "arxiv_id": "2609.07834",
                    "ref_arxiv_id": "1706.03762",
                    "ref_title": "Attention",
                    "ref_submission_date": "2017-06-12",
                    "ref_doi": "https://doi.org/10.48550/ARXIV.1706.03762",
                    "ref_publication_date": "2017-06-12",
                    "ref_publication_year": 2017,
                    "ref_venue": "arXiv",
                },
                {"arxiv_id": "2609.07834", "ref_arxiv_id": None, "ref_title": "Not matchable"},
            ],
        },
    )

    records = list(iter_matchable_reference_records(tmp_path, "range-01"))

    assert [(record.arxiv_id, record.ref_arxiv_id) for record in records] == [
        ("2609.07834", "1706.03762")
    ]
    assert records[0].ref_doi == "10.48550/arxiv.1706.03762"
    assert records[0].ref_publication_date == "2017-06-12"
    assert records[0].ref_publication_year == 2017
    assert records[0].ref_venue == "arXiv"


def test_reference_loader_rejects_a_reference_owned_by_another_citing_paper(tmp_path):
    _write_json(
        tmp_path / "corpus" / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [{"arxiv_id": "2609.07835", "ref_arxiv_id": "1706.03762"}],
        },
    )

    with pytest.raises(InputDataError, match="citing publication"):
        list(iter_matchable_reference_records(tmp_path, "range-01"))


def test_reference_loader_accepts_an_explicit_directory_path(tmp_path):
    references = tmp_path / "custom-m1-location"
    _write_json(
        references / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [{"ref_arxiv_id": "1706.03762"}],
        },
    )

    records = list(iter_matchable_reference_records_from_directory(references))

    assert records[0].ref_arxiv_id == "1706.03762"
