from src.parsing.contracts import (
    BibEntry,
    DeduplicatedPublication,
    Diagnostic,
    DiagnosticSeverity,
    ElementRow,
    NormalizedNode,
    NormalizedVersion,
    ProcessingStatus,
    ReferenceBundle,
    SourceBundle,
)
from src.parsing.statistics import StatisticsAccumulator
from src.parsing.validation import ValidationReport


def test_statistics_include_empty_versions_and_dedup_counts():
    stats = StatisticsAccumulator()
    stats.record_publication(
        sources=(
            SourceBundle("2501.01234", 1, ProcessingStatus.OK, "main.tex"),
            SourceBundle("2501.01234", 2, ProcessingStatus.EMPTY_NO_TEX, None),
        ),
        references=ReferenceBundle(
            "2501.01234",
            (BibEntry("key", "article"),),
            {1: {"key": "key"}},
        ),
        normalized_versions=(
            NormalizedVersion(
                "2501.01234",
                1,
                ProcessingStatus.OK,
                "root",
                (NormalizedNode("root", "document", "", None, 0),),
                3,
            ),
            NormalizedVersion(
                "2501.01234",
                2,
                ProcessingStatus.EMPTY_NO_TEX,
                None,
                (),
                0,
            ),
        ),
        publication=DeduplicatedPublication(
            "2501.01234",
            (),
            (ElementRow("element", "2501.01234", "document", ""),),
            (),
        ),
        validation=ValidationReport(),
        elapsed_seconds=0.25,
        expected_version_count=2,
        raw_reference_entry_count=2,
        diagnostics=(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "MISSING_INCLUDE",
                "Included file was not found",
            ),
        ),
    )

    result = stats.to_dict()
    assert result["versions"]["total"] == 2
    assert result["versions"]["without_tex"] == 1
    assert result["elements"]["raw_count"] == 1
    assert result["elements"]["deduplicated_count"] == 1
    assert result["references"]["unified_entries"] == 1
    assert result["references"]["entries_before_dedup"] == 2
    assert result["references"]["entries_after_dedup"] == 1
    assert result["diagnostics"]["warnings"] == 1
    assert result["per_publication"][0]["versions"]["coverage_rate"] == 1.0
    assert result["per_publication"][0]["versions"]["empty_no_tex"] == 1


def test_statistics_distinguish_partial_and_failed_publications():
    stats = StatisticsAccumulator()

    for arxiv_id, status, validation in (
        ("2501.00001", ProcessingStatus.PARTIAL_PARSE, ValidationReport()),
        (
            "2501.00002",
            ProcessingStatus.PARSE_ERROR,
            ValidationReport(
                diagnostics=(
                    Diagnostic(
                        DiagnosticSeverity.ERROR,
                        "PARSE_ERROR",
                        "Parser failed",
                    ),
                )
            ),
        ),
    ):
        stats.record_publication(
            sources=(SourceBundle(arxiv_id, 1, status, "main.tex"),),
            references=ReferenceBundle(arxiv_id, (), {1: {}}),
            normalized_versions=(),
            publication=DeduplicatedPublication(arxiv_id, (), (), ()),
            validation=validation,
            elapsed_seconds=0.1,
        )

    result = stats.to_dict()
    assert result["publications"]["partial"] == 1
    assert result["publications"]["failed"] == 1
    assert result["versions"]["parse_success_rate"] == 0.0
    assert result["validation"]["errors"] == 1


def test_statistics_mark_reference_diagnostics_as_partial():
    stats = StatisticsAccumulator()
    stats.record_publication(
        sources=(SourceBundle("2501.00003", 1, ProcessingStatus.OK, "main.tex"),),
        references=ReferenceBundle(
            "2501.00003",
            (),
            {1: {}},
            diagnostics=(
                Diagnostic(
                    DiagnosticSeverity.WARNING,
                    "BIBTEX_PARTIAL",
                    "Could not fully parse bibliography entry",
                    version=1,
                ),
            ),
        ),
        normalized_versions=(),
        publication=DeduplicatedPublication("2501.00003", (), (), ()),
        validation=ValidationReport(),
        elapsed_seconds=0.1,
    )

    result = stats.to_dict()
    assert result["publications"]["partial"] == 1
    assert result["versions"]["status_counts"]["bibtex_partial"] == 1


def test_statistics_mark_unresolved_citations_as_bibtex_partial():
    stats = StatisticsAccumulator()
    stats.record_publication(
        sources=(SourceBundle("2501.00004", 1, ProcessingStatus.OK, "main.tex"),),
        references=ReferenceBundle("2501.00004", (), {1: {}}),
        normalized_versions=(),
        publication=DeduplicatedPublication("2501.00004", (), (), ()),
        validation=ValidationReport(
            diagnostics=(
                Diagnostic(
                    DiagnosticSeverity.WARNING,
                    "UNRESOLVED_CITATION",
                    "Missing canonical BibTeX entry",
                    version=1,
                ),
            )
        ),
        elapsed_seconds=0.1,
    )

    result = stats.to_dict()
    assert result["publications"]["partial"] == 1
    assert result["versions"]["status_counts"]["bibtex_partial"] == 1


def test_statistics_merges_single_publication_worker_snapshot():
    worker = StatisticsAccumulator()
    worker.record_publication(
        sources=(SourceBundle("2501.00005", 1, ProcessingStatus.OK, "main.tex"),),
        references=ReferenceBundle("2501.00005", (), {1: {}}),
        normalized_versions=(),
        publication=DeduplicatedPublication("2501.00005", (), (), ()),
        validation=ValidationReport(),
        elapsed_seconds=0.1,
    )

    merged = StatisticsAccumulator()
    merged.merge_single_publication(worker.to_dict())

    assert merged.to_dict() == worker.to_dict()
