import json

import pandas as pd
import pytest

from src.matching.review import (
    load_resume_table,
    prepare_review_bundle,
    prefill_no_candidate_labels,
    save_review_table,
)


def _row(arxiv_id, citation_key, candidates, status=""):
    return {
        "arxiv_id": arxiv_id,
        "citation_key": citation_key,
        "groundtruth_arxiv_id": "",
        "label_source": "manual",
        "status": status,
        "annotator": "member",
        "reviewer": "",
        "confidence": "",
        "notes": "",
        "candidates_json": json.dumps(candidates),
        "candidate_count": len(candidates),
    }


def test_prefill_only_blank_rows_without_candidates():
    labels = pd.DataFrame([
        _row("2609.00001", "no_candidates", []),
        _row("2609.00001", "has_candidate", [{"ref_arxiv_id": "1706.03762"}]),
        _row("2609.00002", "already_labeled", [], status="no_match"),
    ])

    result, changed = prefill_no_candidate_labels(labels)

    assert changed == 1
    assert result.loc[0, "status"] == "no_match"
    assert result.loc[1, "status"] == ""
    assert result.loc[2, "status"] == "no_match"


def test_prepare_review_bundle_writes_master_publication_files_and_summary(tmp_path):
    source = tmp_path / "labels_template.csv"
    pd.DataFrame([
        _row("2609.00002", "b", [{"ref_arxiv_id": "1706.00001"}]),
        _row("2609.00001", "a", []),
    ]).to_csv(source, index=False)

    result = prepare_review_bundle(source, tmp_path / "review")

    assert result["prefilled_no_candidate_entries"] == 1
    assert (tmp_path / "review" / "labels_review.csv").is_file()
    assert (tmp_path / "review" / "review_summary.csv").is_file()
    assert (tmp_path / "review" / "publications" / "2609.00001.csv").is_file()
    assert (tmp_path / "review" / "publications" / "2609.00002.csv").is_file()

    summary = pd.read_csv(tmp_path / "review" / "review_summary.csv")
    assert summary.loc[0, "total_entries"] == 2
    assert summary.loc[0, "prefilled_no_candidate_entries"] == 1


def test_load_resume_table_restores_saved_labels_without_replacing_template_metadata(
    tmp_path,
):
    template_path = tmp_path / "labels_review.csv"
    output_path = tmp_path / "labels_completed.csv"
    template = pd.DataFrame(
        [
            {
                **_row("2609.00001", "a", [{"ref_arxiv_id": "1706.00001"}]),
                "bibtex_title": "Current template title",
            },
            {
                **_row("2609.00001", "b", []),
                "bibtex_title": "Second title",
            },
        ]
    )
    save_review_table(template, template_path)

    saved = template.copy()
    saved.loc[0, "status"] = "match"
    saved.loc[0, "groundtruth_arxiv_id"] = "1706.00001"
    saved.loc[0, "confidence"] = "high"
    saved.loc[0, "notes"] = "Checked before restart"
    saved.loc[0, "bibtex_title"] = "Stale output title"
    save_review_table(saved, output_path)

    resumed, restored_count = load_resume_table(template_path, output_path)

    assert restored_count == 1
    assert resumed.loc[0, "status"] == "match"
    assert resumed.loc[0, "groundtruth_arxiv_id"] == "1706.00001"
    assert resumed.loc[0, "confidence"] == "high"
    assert resumed.loc[0, "notes"] == "Checked before restart"
    assert resumed.loc[0, "bibtex_title"] == "Current template title"
    assert resumed.loc[1, "status"] == ""

    resumed.loc[0, "status"] = "no_match"
    resumed.loc[0, "groundtruth_arxiv_id"] = ""
    resumed.loc[0, "notes"] = "Corrected after restart"
    save_review_table(resumed, output_path)

    edited, restored_count = load_resume_table(template_path, output_path)
    assert restored_count == 1
    assert edited.loc[0, "status"] == "no_match"
    assert edited.loc[0, "groundtruth_arxiv_id"] == ""
    assert edited.loc[0, "notes"] == "Corrected after restart"


def test_load_resume_table_rejects_saved_rows_from_another_template(tmp_path):
    template_path = tmp_path / "labels_review.csv"
    output_path = tmp_path / "labels_completed.csv"
    save_review_table(
        pd.DataFrame([_row("2609.00001", "a", [])]),
        template_path,
    )
    save_review_table(
        pd.DataFrame([_row("2609.99999", "other", [], status="no_match")]),
        output_path,
    )

    with pytest.raises(ValueError, match="not present in the input template"):
        load_resume_table(template_path, output_path)
