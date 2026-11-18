from src.matching.contracts import BibtexRecord, CandidatePair, ReferenceRecord
from src.matching.validation import (
    validate_bibtex_records,
    validate_candidate_pairs,
    validate_partition_records,
    validate_reference_records,
)


def test_validation_accepts_unique_in_range_matching_records():
    bibtex = [BibtexRecord("2609.07834", "vaswani2017", "article")]
    references = [ReferenceRecord("2609.07834", "1706.03762")]

    bibtex_report = validate_bibtex_records(bibtex, {"2609.07834"})
    reference_report = validate_reference_records(references, {"2609.07834"})
    candidate_report = validate_candidate_pairs(
        [CandidatePair("2609.07834", "vaswani2017", "1706.03762")],
        {("2609.07834", "vaswani2017")},
        {("2609.07834", "1706.03762")},
    )

    assert bibtex_report.error_count == 0
    assert reference_report.error_count == 0
    assert candidate_report.error_count == 0


def test_validation_reports_duplicate_and_out_of_range_keys():
    report = validate_bibtex_records(
        [
            BibtexRecord("2609.07834", "vaswani2017", "article"),
            BibtexRecord("2609.07834", "vaswani2017", "article"),
            BibtexRecord("2609.07835", "other", "article"),
        ],
        {"2609.07834"},
    )

    assert report.error_codes == {"DUPLICATE_BIBTEX_ENTRY", "PUBLICATION_OUT_OF_RANGE"}


def test_validation_rejects_a_candidate_from_a_different_citing_publication():
    report = validate_candidate_pairs(
        [CandidatePair("2609.07834", "vaswani2017", "1706.03762")],
        {("2609.07834", "vaswani2017")},
        {("2609.07835", "1706.03762")},
    )

    assert report.error_codes == {"CANDIDATE_REFERENCE_WRONG_PUBLICATION"}


def test_validation_reports_a_publication_in_multiple_partitions():
    report = validate_partition_records(
        [
            {"arxiv_id": "2609.07834", "partition": "train"},
            {"arxiv_id": "2609.07834", "partition": "test"},
        ]
    )

    assert report.error_codes == {"SPLIT_OVERLAP"}
