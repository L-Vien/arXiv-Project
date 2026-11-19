import json

from src.matching.inventory import build_publication_inventory


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_inventory_aggregates_bibtex_and_reference_quality_by_publication(tmp_path):
    _write_jsonl(
        tmp_path / "m2" / "range-01" / "bibtex.jsonl",
        [
            {"arxiv_id": "2609.07834", "citation_key": "a", "entry_type": "article", "title": "", "authors": [], "year": None},
            {"arxiv_id": "2609.07834", "citation_key": "b", "entry_type": "article", "title": "Title", "authors": ["Author"], "year": 2024},
        ],
    )
    _write_json(
        tmp_path / "corpus" / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [
                {"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762"},
                {"arxiv_id": "2609.07834", "ref_arxiv_id": None},
                {"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762"},
            ],
        },
    )

    rows = build_publication_inventory(tmp_path, "range-01", "24127023", {"2609.07834"})

    assert len(rows) == 1
    assert rows[0].bibtex_count == 2
    assert rows[0].reference_count == 3
    assert rows[0].arxiv_reference_count == 2
    assert rows[0].duplicate_reference_count == 1
    assert rows[0].missing_title_rate == 0.5
