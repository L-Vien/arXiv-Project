import json

import pytest

from src.matching.loading import InputDataError, iter_bibtex_records, iter_matchable_reference_records


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
                "url": ""
            }
        ],
    )

    records = list(iter_bibtex_records(tmp_path, "range-01"))

    assert records[0].arxiv_id == "2609.07834"
    assert records[0].authors == ("Ashish Vaswani",)
    assert records[0].doi == "10.48550/arxiv.1706.03762"
    assert records[0].journal is None


def test_reference_loader_excludes_non_matchable_references(tmp_path):
    _write_json(
        tmp_path / "corpus" / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [
                {"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762", "ref_title": "Attention"},
                {"arxiv_id": "2609.07834", "ref_arxiv_id": None, "ref_title": "Not matchable"},
            ],
        },
    )

    records = list(iter_matchable_reference_records(tmp_path, "range-01"))

    assert [(record.arxiv_id, record.ref_arxiv_id) for record in records] == [
        ("2609.07834", "1706.03762")
    ]


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
