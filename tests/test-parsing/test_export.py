import json

from src.parsing.contracts import (
    BibEntry,
    DeduplicatedPublication,
    Diagnostic,
    DiagnosticSeverity,
    ElementRow,
    HierarchyRow,
    NormalizedVersion,
    ProcessingStatus,
    ReferenceBundle,
    SourceBundle,
    SourceFile,
)
from src.parsing.export import (
    PublicationBatchWriter,
    export_publication,
    finalize_run,
)
from src.parsing.validation import ValidationReport


def test_export_uses_flat_jsonl_schema_and_version_manifest(tmp_path):
    publication = DeduplicatedPublication(
        arxiv_id="2501.01234",
        versions=(
            NormalizedVersion(
                arxiv_id="2501.01234",
                version=1,
                status=ProcessingStatus.OK,
                root_local_id="local-root",
                nodes=(),
                raw_element_count=1,
            ),
        ),
        elements=(
            ElementRow("2501-01234#root-v1", "2501.01234", "document", ""),
        ),
        hierarchy=(
            HierarchyRow(
                "2501.01234",
                1,
                "2501-01234#sentence-1",
                "2501-01234#root-v1",
            ),
        ),
    )
    references = ReferenceBundle(
        arxiv_id="2501.01234",
        entries=(
            BibEntry(
                citation_key="vaswani2017",
                entry_type="article",
                title="Attention",
                authors=("A. Author", "B. Author"),
                year=2017,
            ),
        ),
        citation_maps={1: {"old": "vaswani2017"}},
    )
    sources = (
        SourceBundle(
            arxiv_id="2501.01234",
            version=1,
            status=ProcessingStatus.OK,
            root_file="main.tex",
            files=(SourceFile("main.tex", "", 0),),
        ),
        SourceBundle(
            arxiv_id="2501.01234",
            version=2,
            status=ProcessingStatus.EMPTY_NO_TEX,
            root_file=None,
            files=(),
        ),
    )

    publication_dir = export_publication(
        publication=publication,
        references=references,
        sources=sources,
        validation_report=ValidationReport(
            diagnostics=(
                Diagnostic(
                    DiagnosticSeverity.WARNING,
                    "UNRESOLVED_CITATION",
                    "Missing canonical BibTeX entry",
                    version=1,
                ),
            )
        ),
        output_dir=tmp_path,
    )
    finalize_run(tmp_path)

    element = json.loads((tmp_path / "elements.jsonl").read_text().strip())
    bibtex = json.loads((tmp_path / "bibtex.jsonl").read_text().strip())
    provenance = json.loads(
        (tmp_path / "bibtex_provenance.jsonl").read_text().strip()
    )
    manifests = [
        json.loads(line)
        for line in (tmp_path / "version_manifest.jsonl").read_text().splitlines()
    ]

    assert publication_dir.name == "2501-01234"
    assert set(element) == {"element_id", "arxiv_id", "kind", "content"}
    assert bibtex["authors"] == ["A. Author", "B. Author"]
    assert provenance == {
        "arxiv_id": "2501.01234",
        "citation_key": "vaswani2017",
        "raw_citation_key": "vaswani2017",
        "source_kind": "unknown",
        "parse_status": "parsed",
    }
    assert manifests[0]["status"] == "bibtex_partial"
    assert manifests[1]["status"] == "empty_no_tex"
    assert json.loads(
        (tmp_path / "validation_report.json").read_text(encoding="utf-8")
    )["status"] == "passed"


def test_export_manifest_marks_bibtex_partial_diagnostic(tmp_path):
    publication = DeduplicatedPublication(
        arxiv_id="2501.01234",
        versions=(
            NormalizedVersion(
                arxiv_id="2501.01234",
                version=1,
                status=ProcessingStatus.OK,
                root_local_id="local-root",
                nodes=(),
                raw_element_count=1,
            ),
        ),
        elements=(),
        hierarchy=(),
    )
    references = ReferenceBundle(
        arxiv_id="2501.01234",
        entries=(),
        citation_maps={1: {}},
    )
    sources = (
        SourceBundle(
            arxiv_id="2501.01234",
            version=1,
            status=ProcessingStatus.OK,
            root_file="main.tex",
            files=(SourceFile("main.tex", "", 0),),
        ),
    )

    export_publication(
        publication=publication,
        references=references,
        sources=sources,
        validation_report=ValidationReport(
            diagnostics=(
                Diagnostic(
                    DiagnosticSeverity.WARNING,
                    "BIBTEX_PARTIAL",
                    "A bibliography source was only partially parsed",
                    version=1,
                ),
            )
        ),
        output_dir=tmp_path,
    )
    finalize_run(tmp_path)

    manifest = json.loads((tmp_path / "version_manifest.jsonl").read_text())
    assert manifest["status"] == "bibtex_partial"


def test_batch_writer_reduces_shards_and_finalizes_contract(tmp_path):
    publication = DeduplicatedPublication(
        arxiv_id="2501.01234",
        versions=(),
        elements=(),
        hierarchy=(),
    )
    references = ReferenceBundle(
        arxiv_id="2501.01234",
        entries=(),
        citation_maps={},
    )
    validation = ValidationReport()

    with PublicationBatchWriter(tmp_path, 0) as writer:
        writer.write_publication(
            publication=publication,
            references=references,
            sources=(),
            validation_report=validation,
        )

    batch_dir = tmp_path / "batches" / "part-000000"
    assert sorted(path.name for path in batch_dir.iterdir()) == [
        "bibtex.jsonl",
        "bibtex_provenance.jsonl",
        "elements.jsonl",
        "hierarchy.jsonl",
        "validation_reports.jsonl",
        "version_manifest.jsonl",
    ]
    finalize_run(tmp_path, shard_dir=tmp_path)
    assert (tmp_path / "elements.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads(
        (tmp_path / "validation_report.json").read_text(encoding="utf-8")
    )["status"] == "passed"
