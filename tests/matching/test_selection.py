import json

import pandas as pd
import pytest

from src.matching.selection import (
    N_MAIN,
    N_SPARE,
    build_inventory,
    load_references,
    select_manual,
)


def _inventory_rows(count):
    rows = []
    for number in range(1, count + 1):
        bibtex_count = 19 + number
        arxiv_reference_count = 8
        rows.append(
            {
                "arxiv_id": f"2609.{number:05d}",
                "bibtex_count": bibtex_count,
                "reference_count": arxiv_reference_count,
                "arxiv_reference_count": arxiv_reference_count,
                "duplicate_reference_count": 0,
                "arxiv_ref_ratio": arxiv_reference_count / bibtex_count,
                "missing_title_rate": 0.0,
                "missing_author_rate": 0.0,
                "missing_year_rate": 0.0,
            }
        )
    return rows


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_references_only_returns_requested_publications_and_checks_folder_id(tmp_path):
    _write_json(
        tmp_path / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [{"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762"}],
        },
    )
    _write_json(
        tmp_path / "range-01" / "2609-07835" / "references.json",
        {
            "arxiv_id": "2609.07835",
            "references": [{"arxiv_id": "2609.07835", "ref_arxiv_id": None}],
        },
    )

    result = load_references(tmp_path, {"2609.07834"})

    assert list(result) == ["2609.07834"]
    assert len(result["2609.07834"]) == 1


def test_load_references_rejects_document_id_mismatch(tmp_path):
    _write_json(
        tmp_path / "range-01" / "2609-07834" / "references.json",
        {"arxiv_id": "2609.07835", "references": []},
    )

    with pytest.raises(ValueError, match="Publication ID mismatch"):
        load_references(tmp_path, {"2609.07834"})


def test_build_inventory_counts_bibtex_and_reference_quality(tmp_path):
    bibtex = tmp_path / "bibtex.jsonl"
    _write_jsonl(
        bibtex,
        [
            {"arxiv_id": "2609.07834", "citation_key": "a"},
            {"arxiv_id": "2609-07834", "citation_key": "b"},
            {"arxiv_id": "2609.07835", "citation_key": "c"},
        ],
    )
    references = tmp_path / "references"
    _write_json(
        references / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [
                {"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762"},
                {"arxiv_id": "2609.07834", "ref_arxiv_id": None},
            ],
        },
    )
    _write_json(
        references / "2609-07835" / "references.json",
        {"arxiv_id": "2609.07835", "references": []},
    )

    inventory = build_inventory(
        bibtex,
        references,
        ["2609.07834", "2609.07835", "2609.07834"],
    )

    assert inventory["arxiv_id"].tolist() == ["2609.07834", "2609.07835"]
    row = inventory.set_index("arxiv_id").loc["2609.07834"]
    assert row["bibtex_count"] == 2
    assert row["reference_count"] == 2
    assert row["arxiv_reference_count"] == 1


def test_select_manual_filters_eligibility_and_assigns_partitions_deterministically():
    inventory = pd.DataFrame(_inventory_rows(8))
    ineligible = inventory.iloc[0].copy()
    ineligible["arxiv_id"] = "2609.99999"
    ineligible["bibtex_count"] = 19
    ineligible["arxiv_ref_ratio"] = 8 / 19
    inventory = pd.concat([pd.DataFrame([ineligible]), inventory], ignore_index=True)

    selected = select_manual(inventory, "24127023", "range-01")
    repeated = select_manual(inventory, "24127023", "range-01")

    assert len(selected) == N_MAIN + N_SPARE
    assert selected.equals(repeated)
    assert "2609.99999" not in set(selected["arxiv_id"])
    assert selected["partition"].value_counts().to_dict() == {
        "train": 3,
        "valid": 1,
        "test": 1,
        "spare": 1,
    }
    assert set(selected["member_id"]) == {"24127023"}
    assert set(selected["range_id"]) == {"range-01"}
    assert set(selected["label_source"]) == {"manual"}


def test_select_manual_requires_six_eligible_publications():
    inventory = pd.DataFrame(_inventory_rows(N_MAIN + N_SPARE - 1))

    with pytest.raises(ValueError, match="Không chọn được 6 publication"):
        select_manual(inventory, "24127023", "range-01")
