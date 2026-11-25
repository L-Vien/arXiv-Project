import json

import pandas as pd
import pytest

from src.matching.labeling import (
    LABEL_COLUMNS,
    build_parser,
    candidate_view,
    get_candidates,
    hint_arxiv_id,
    load_references,
    prepare_annotation_template,
    rank_candidates,
    save_table,
    validate_labels,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_references_reads_directory_recursively_and_groups_by_citing_paper(tmp_path):
    first_id = "2609.07834"
    second_id = "2609.07835"
    _write_json(
        tmp_path / "range-01" / "2609-07834" / "references.json",
        {
            "arxiv_id": first_id,
            "references": [
                {"arxiv_id": first_id, "ref_arxiv_id": "1706.03762"},
                {"arxiv_id": first_id, "ref_arxiv_id": None},
            ],
        },
    )
    _write_json(
        tmp_path / "range-01" / "2609-07835" / "references.json",
        {"arxiv_id": second_id, "references": [{"ref_arxiv_id": "1801.00001"}]},
    )

    references = load_references(tmp_path)

    assert len(references[first_id]) == 2
    assert len(references[second_id]) == 1


def test_get_candidates_skips_missing_ids_and_deduplicates_normalized_ids():
    candidates = get_candidates(
        [
            {"ref_arxiv_id": None, "ref_title": "No ID"},
            {"ref_arxiv_id": "1706.03762", "ref_title": "First"},
            {"ref_arxiv_id": "1706.03762", "ref_title": "Duplicate"},
            {"ref_arxiv_id": "hep-th/9901001", "ref_title": "Legacy"},
        ]
    )

    assert [candidate["ref_arxiv_id"] for candidate in candidates] == [
        "1706.03762",
        "hep-th/9901001",
    ]


def test_candidate_view_keeps_publication_and_submission_years_distinct():
    candidate = candidate_view(
        {
            "ref_arxiv_id": "1706.03762",
            "ref_publication_year": 2018,
            "ref_submission_date": "2017-06-12",
            "ref_venue": "NeurIPS",
            "ref_doi": "10.48550/arXiv.1706.03762",
            "ref_url": "https://arxiv.org/abs/1706.03762",
        }
    )

    assert candidate["ref_year"] == "2018"
    assert candidate["ref_submission_date"] == "2017-06-12"
    assert candidate["ref_venue"] == "NeurIPS"
    assert candidate["ref_doi"] == "10.48550/arXiv.1706.03762"
    assert candidate["ref_url"] == "https://arxiv.org/abs/1706.03762"


def test_rank_candidates_prefers_title_similarity_and_exposes_bibtex_hint():
    candidates = [
        {"ref_arxiv_id": "1801.00001", "ref_title": "Unrelated Paper"},
        {"ref_arxiv_id": "1706.03762", "ref_title": "Attention Is All You Need"},
    ]

    ranked = rank_candidates("Attention Is All You Need", candidates)

    assert [candidate["ref_arxiv_id"] for candidate in ranked] == [
        "1706.03762",
        "1801.00001",
    ]
    assert ranked[0]["score"] > ranked[1]["score"]
    assert hint_arxiv_id(
        {"url": "https://arxiv.org/abs/1706.03762v2"},
        ranked,
    ) == "1706.03762"


def test_prepare_template_uses_only_candidates_of_each_assigned_publication(tmp_path):
    assignment = tmp_path / "assignment.csv"
    pd.DataFrame(
        [
            {
                "arxiv_id": "2609.07834",
                "member_id": "24127023",
                "range_id": "range-01",
                "partition": "train",
            },
            {
                "arxiv_id": "2609.07835",
                "member_id": "24127023",
                "range_id": "range-01",
                "partition": "valid",
            },
        ]
    ).to_csv(assignment, index=False)
    bibtex = tmp_path / "bibtex.jsonl"
    _write_jsonl(
        bibtex,
        [
            {
                "arxiv_id": "2609.07834",
                "citation_key": "b",
                "entry_type": "article",
                "doi": "10.1000/example",
                "journal": "Example Journal",
                "url": "https://example.test/paper",
            },
            {"arxiv_id": "2609.07834", "citation_key": "a", "entry_type": "book"},
            {"arxiv_id": "2609.07835", "citation_key": "c", "entry_type": "article"},
            {"arxiv_id": "2609.07836", "citation_key": "ignored", "entry_type": "article"},
        ],
    )
    references = tmp_path / "references"
    _write_json(
        references / "2609-07834" / "references.json",
        {
            "arxiv_id": "2609.07834",
            "references": [
                {"arxiv_id": "2609.07834", "ref_arxiv_id": "1706.03762", "ref_title": "First"},
            ],
        },
    )
    _write_json(
        references / "2609-07835" / "references.json",
        {
            "arxiv_id": "2609.07835",
            "references": [
                {"arxiv_id": "2609.07835", "ref_arxiv_id": "1801.00001", "ref_title": "Second"},
            ],
        },
    )

    template = prepare_annotation_template(assignment, bibtex, references)

    assert template[["arxiv_id", "citation_key"]].values.tolist() == [
        ["2609.07834", "a"],
        ["2609.07834", "b"],
        ["2609.07835", "c"],
    ]
    assert template["candidate_count"].tolist() == [1, 1, 1]
    candidates = [json.loads(value)[0]["ref_arxiv_id"] for value in template["candidates_json"]]
    assert candidates == ["1706.03762", "1706.03762", "1801.00001"]
    bibtex_row = template.loc[template["citation_key"] == "b"].iloc[0]
    assert bibtex_row["bibtex_doi"] == "10.1000/example"
    assert bibtex_row["bibtex_journal"] == "Example Journal"
    assert bibtex_row["bibtex_url"] == "https://example.test/paper"


def test_prepare_template_rejects_duplicate_assignment_and_bibtex_keys(tmp_path):
    assignment = tmp_path / "assignment.csv"
    pd.DataFrame(
        [
            {"arxiv_id": "2609.07834"},
            {"arxiv_id": "2609.07834"},
        ]
    ).to_csv(assignment, index=False)
    bibtex = tmp_path / "bibtex.jsonl"
    bibtex.write_text("", encoding="utf-8")
    references = tmp_path / "references.json"
    _write_json(
        references,
        {"arxiv_id": "2609.07834", "references": []},
    )

    with pytest.raises(ValueError, match="Duplicate arxiv_id"):
        prepare_annotation_template(assignment, bibtex, references)


def test_validate_labels_checks_status_groundtruth_and_candidate_membership():
    row = {
        "arxiv_id": "2609.07834",
        "citation_key": "key",
        "groundtruth_arxiv_id": "1706.03762",
        "label_source": "manual",
        "status": "match",
        "annotator": "member",
        "reviewer": "",
        "confidence": "high",
        "notes": "",
        "candidates_json": json.dumps([{"ref_arxiv_id": "1706.03762"}]),
    }
    labels = pd.DataFrame([row])

    errors, warnings = validate_labels(labels)

    assert errors == []
    assert warnings == ["Publication 2609.07834 has only 1 labels (< 20)."]

    labels.loc[0, "groundtruth_arxiv_id"] = "1801.00001"
    errors, _ = validate_labels(labels)
    assert any("not in the candidate list" in error for error in errors)

    errors, _ = validate_labels(labels, allow_outside_candidates=True)
    assert errors == []


def test_validate_labels_rejects_invalid_final_and_duplicate_rows():
    row = {
        "arxiv_id": "2609.07834",
        "citation_key": "key",
        "groundtruth_arxiv_id": "",
        "label_source": "manual",
        "status": "uncertain",
        "annotator": "",
        "reviewer": "",
        "confidence": "high",
        "notes": "",
    }
    labels = pd.DataFrame([row, row], columns=LABEL_COLUMNS)

    errors, warnings = validate_labels(labels, allow_uncertain=False)

    assert any("Duplicate (arxiv_id, citation_key)" in error for error in errors)
    assert any("uncertain is not allowed" in error for error in errors)
    assert warnings == ["Publication 2609.07834 has only 2 labels (< 20)."]


def test_validate_labels_requires_independent_reviewer_for_final_valid_rows():
    row = {
        "arxiv_id": "2609.07834",
        "citation_key": "key",
        "groundtruth_arxiv_id": "",
        "label_source": "manual",
        "status": "no_match",
        "annotator": "member-a",
        "reviewer": "",
        "confidence": "high",
        "notes": "",
        "partition": "valid",
    }
    labels = pd.DataFrame([row])

    errors, _ = validate_labels(labels, final=True)
    assert any("reviewer required" in error for error in errors)

    labels.loc[0, "reviewer"] = "member-a"
    errors, _ = validate_labels(labels, final=True)
    assert any("reviewer must differ" in error for error in errors)

    labels.loc[0, "reviewer"] = "member-b"
    errors, _ = validate_labels(labels, final=True)
    assert errors == []


def test_build_parser_supports_prepare_review_command():
    args = build_parser().parse_args(
        [
            "prepare-review",
            "--input",
            "labels_template.csv",
            "--output-dir",
            "review",
        ]
    )

    assert args.command == "prepare-review"
    assert args.input == "labels_template.csv"
    assert args.output_dir == "review"
    assert args.no_prefill_no_candidates is False


def test_save_table_writes_csv_and_rejects_unsupported_extension(tmp_path):
    frame = pd.DataFrame([{"arxiv_id": "2609.07834"}])
    output = tmp_path / "nested" / "labels.csv"

    save_table(frame, output)

    assert pd.read_csv(output, dtype=str)["arxiv_id"].tolist() == ["2609.07834"]
    with pytest.raises(ValueError, match="Output must end"):
        save_table(frame, tmp_path / "labels.txt")
