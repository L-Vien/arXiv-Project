import json
from pathlib import Path

import pytest

from src.parsing.contracts import (
    BibEntry,
    DeduplicatedPublication,
    Diagnostic,
    DiagnosticSeverity,
    ElementRow,
    NormalizedNode,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
    SourceFile,
    TreeNode,
)
from src.parsing.pipeline import (
    PipelineDependencies,
    PublicationInput,
    VersionInput,
    _merge_diagnostics,
    _publication_outcome,
    current_dependencies,
    process_publication,
    run_pipeline,
)
from src.parsing.benchmark import directory_size_bytes
from src.parsing.validation import ValidationReport


def _source(arxiv_id: str, version: int, *, empty: bool = False) -> SourceBundle:
    return SourceBundle(
        arxiv_id=arxiv_id,
        version=version,
        status=(
            ProcessingStatus.EMPTY_NO_TEX
            if empty
            else ProcessingStatus.OK
        ),
        root_file=None if empty else "main.tex",
        files=()
        if empty
        else (SourceFile("main.tex", "\\documentclass{article}", 0),),
    )


def _dependencies(calls: list[tuple[str, object]]) -> PipelineDependencies:
    def gather(source_dir: Path, arxiv_id: str, version: int) -> SourceBundle:
        calls.append(("gather", version))
        return _source(arxiv_id, version, empty=version == 2)

    def parse(source: SourceBundle) -> RawTree:
        calls.append(("parse", source.version))
        return RawTree(
            arxiv_id=source.arxiv_id,
            version=source.version,
            root=TreeNode(kind="document"),
            status=ProcessingStatus.OK,
        )

    def references(sources: tuple[SourceBundle, ...]) -> ReferenceBundle:
        calls.append(("references", tuple(source.version for source in sources)))
        return ReferenceBundle(
            arxiv_id="2501.01234",
            entries=(
                BibEntry(
                    citation_key="canonical",
                    entry_type="article",
                    title="A paper",
                ),
            ),
            citation_maps={1: {"old": "canonical"}, 2: {}},
        )

    def normalize(raw: RawTree, citation_map: dict[str, str]) -> NormalizedVersion:
        calls.append(("normalize", (raw.version, dict(citation_map))))
        return NormalizedVersion(
            arxiv_id=raw.arxiv_id,
            version=raw.version,
            status=ProcessingStatus.OK,
            root_local_id=f"local-{raw.version}",
            nodes=(
                NormalizedNode(
                    local_id=f"local-{raw.version}",
                    kind="document",
                    content="",
                    parent_local_id=None,
                    order=0,
                ),
            ),
            raw_element_count=1,
        )

    def deduplicate(versions: tuple[NormalizedVersion, ...], arxiv_id: str):
        calls.append(("deduplicate", tuple(version.version for version in versions)))
        return DeduplicatedPublication(
            arxiv_id=arxiv_id,
            versions=versions,
            elements=(
                ElementRow(
                    element_id="2501-01234#root-v1",
                    arxiv_id=arxiv_id,
                    kind="document",
                    content="",
                ),
            ),
            hierarchy=(),
        )

    return PipelineDependencies(
        gather_version=gather,
        parse_structure=parse,
        process_references=references,
        normalize_version=normalize,
        deduplicate_publication=deduplicate,
    )


def test_process_publication_delegates_and_preserves_empty_version():
    calls: list[tuple[str, object]] = []
    result = process_publication(
        PublicationInput(
            "2501.01234",
            (
                VersionInput(2, Path("v2")),
                VersionInput(1, Path("v1")),
            ),
        ),
        _dependencies(calls),
    )

    assert [name for name, _ in calls] == [
        "gather",
        "gather",
        "parse",
        "references",
        "normalize",
        "deduplicate",
    ]
    assert result.raw_trees[1].root is None
    assert result.raw_trees[1].status is ProcessingStatus.EMPTY_NO_TEX
    assert result.normalized_versions[1].nodes == ()
    assert calls[-2] == ("normalize", (1, {"old": "canonical"}))
    assert result.validation.passed


def test_publication_outcome_includes_source_and_reference_failures():
    failed_source = SourceBundle(
        "2501.00001",
        1,
        ProcessingStatus.ROOT_NOT_FOUND,
        None,
    )
    partial_source = SourceBundle(
        "2501.00001",
        1,
        ProcessingStatus.PARTIAL_PARSE,
        "main.tex",
    )
    warning_reference = ReferenceBundle(
        "2501.00001",
        (),
        {1: {}},
        diagnostics=(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "BIBTEX_PARTIAL",
                "Could not fully parse bibliography entry",
            ),
        ),
    )
    clean_reference = ReferenceBundle("2501.00001", (), {1: {}})

    assert _publication_outcome(
        (failed_source,), clean_reference, ValidationReport()
    ) == "failed"
    assert _publication_outcome(
        (partial_source,), clean_reference, ValidationReport()
    ) == "partial"
    assert _publication_outcome(
        (_source("2501.00001", 1),), warning_reference, ValidationReport()
    ) == "partial"
    assert _publication_outcome(
        (_source("2501.00001", 1),), clean_reference, ValidationReport()
    ) == "success"
    unresolved = ValidationReport(
        diagnostics=(
            Diagnostic(
                DiagnosticSeverity.WARNING,
                "UNRESOLVED_CITATION",
                "1 citation key has no canonical BibTeX entry: missing",
                arxiv_id="2501.00001",
                version=1,
            ),
        )
    )
    assert _publication_outcome(
        (_source("2501.00001", 1),), clean_reference, unresolved
    ) == "partial"


def test_process_publication_surfaces_module_diagnostics():
    calls: list[tuple[str, object]] = []
    dependencies = _dependencies(calls)
    original_gather = dependencies.gather_version

    def gather_with_warning(
        source_dir: Path,
        arxiv_id: str,
        version: int,
    ) -> SourceBundle:
        source = original_gather(source_dir, arxiv_id, version)
        if version == 1:
            return SourceBundle(
                arxiv_id=source.arxiv_id,
                version=source.version,
                status=source.status,
                root_file=source.root_file,
                files=source.files,
                diagnostics=(
                    Diagnostic(
                        DiagnosticSeverity.WARNING,
                        "MISSING_INCLUDE",
                        "Included file was not found",
                    ),
                ),
            )
        return source

    dependencies = PipelineDependencies(
        gather_version=gather_with_warning,
        parse_structure=dependencies.parse_structure,
        process_references=dependencies.process_references,
        normalize_version=dependencies.normalize_version,
        deduplicate_publication=dependencies.deduplicate_publication,
    )
    result = process_publication(
        PublicationInput(
            "2501.01234",
            (VersionInput(1, Path("v1")),),
        ),
        dependencies,
    )

    assert any(
        diagnostic.code == "MISSING_INCLUDE"
        for diagnostic in result.validation.warnings
    )


def test_merge_diagnostics_collapses_equivalent_missing_include_messages():
    source_diagnostic = Diagnostic(
        DiagnosticSeverity.WARNING,
        "MISSING_INCLUDE",
        "Included file was not found: epsf",
        file_path="main.tex",
        line=10,
    )
    tree_diagnostic = Diagnostic(
        DiagnosticSeverity.WARNING,
        "MISSING_INCLUDE",
        "Missing include: epsf.tex",
        file_path="main.tex",
        line=10,
    )
    source = SourceBundle(
        "2501.01234",
        1,
        ProcessingStatus.PARTIAL_PARSE,
        "main.tex",
        diagnostics=(source_diagnostic,),
    )
    tree = RawTree(
        "2501.01234",
        1,
        None,
        ProcessingStatus.PARTIAL_PARSE,
        diagnostics=(tree_diagnostic,),
    )

    report = _merge_diagnostics(
        ValidationReport(),
        (source,),
        (tree,),
        ReferenceBundle("2501.01234", (), {1: {}}),
        (),
    )

    assert len(report.warnings) == 1


def test_run_pipeline_writes_run_outputs(tmp_path):
    source_dir = tmp_path / "v1"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text(
        r"""\documentclass{article}
\begin{document}
\section{Run}
Pipeline output.
\end{document}
""",
        encoding="utf-8",
    )
    result = run_pipeline(
        [
            PublicationInput(
                "2501.01234",
                (VersionInput(1, source_dir),),
            )
        ],
        current_dependencies(),
        tmp_path / "run",
        collect_results=True,
    )

    output_dir = tmp_path / "run"
    assert result.publications[0].validation.passed
    assert (output_dir / "elements.jsonl").exists()
    assert (output_dir / "hierarchy.jsonl").exists()
    assert (output_dir / "bibtex.jsonl").exists()
    assert (output_dir / "version_manifest.jsonl").exists()
    assert (output_dir / "statistics.json").exists()
    assert (output_dir / "benchmark.json").exists()
    assert (output_dir / "validation_report.json").exists()
    assert not (output_dir / "publications").exists()
    benchmark = json.loads(
        (output_dir / "benchmark.json").read_text(encoding="utf-8")
    )
    assert benchmark["output"]["size_bytes"] == directory_size_bytes(output_dir)
    assert benchmark["execution"] == {"workers": 1, "batch_size": 8}
    assert set(benchmark["stage_timings_seconds"]) >= {
        "gather",
        "hierarchy",
        "references",
        "normalization",
        "deduplication",
        "validation",
        "export",
        "finalize",
        "run_statistics",
    }


def test_run_pipeline_counts_merged_diagnostics_in_statistics(tmp_path):
    source_dir = tmp_path / "v1"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text(
        r"""\documentclass{article}
\begin{document}
\input{missing}
Text.
\end{document}
""",
        encoding="utf-8",
    )

    result = run_pipeline(
        [
            PublicationInput(
                "2501.01234",
                (VersionInput(1, source_dir),),
            )
        ],
        current_dependencies(),
        tmp_path / "diagnostic-run",
        progress_enabled=False,
    )

    assert result.statistics["diagnostics"]["warnings"] >= 1
    assert result.statistics["validation"]["warnings"] >= 1
    assert result.statistics["per_publication"][0]["validation"]["warnings"] >= 1


def test_run_pipeline_propagates_later_partial_parse_status(tmp_path):
    source_dir = tmp_path / "v1"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text(
        r"""\documentclass{article}
\begin{document}
Text $$x+1
\end{document}
""",
        encoding="utf-8",
    )

    result = run_pipeline(
        [
            PublicationInput(
                "2501.01234",
                (VersionInput(1, source_dir),),
            )
        ],
        current_dependencies(),
        tmp_path / "partial-run",
        progress_enabled=False,
        collect_results=True,
    )

    assert result.publications[0].raw_trees[0].status is ProcessingStatus.PARTIAL_PARSE
    assert result.statistics["publications"]["partial"] == 1
    assert result.statistics["publications"]["successful"] == 0
    manifest = json.loads(
        (tmp_path / "partial-run" / "version_manifest.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert manifest["status"] == "partial_parse"


def test_pipeline_keeps_diagnostic_version_context(tmp_path):
    versions = []
    for version_number in (1, 2):
        source_dir = tmp_path / f"v{version_number}"
        source_dir.mkdir()
        (source_dir / "main.tex").write_text(
            r"""\documentclass{article}
\begin{document}
Text $$x+1
\end{document}
""",
            encoding="utf-8",
        )
        versions.append(VersionInput(version_number, source_dir))

    result = run_pipeline(
        [PublicationInput("2501.01234", tuple(versions))],
        current_dependencies(),
        tmp_path / "context-run",
        progress_enabled=False,
        collect_results=True,
    )

    diagnostics = [
        diagnostic
        for diagnostic in result.publications[0].validation.diagnostics
        if diagnostic.code == "UNCLOSED_MATH"
    ]
    assert {diagnostic.version for diagnostic in diagnostics} == {1, 2}


def test_run_pipeline_uses_real_tv2_tv3_tv4_tv5(tmp_path):
    version_dir = tmp_path / "v1"
    sections_dir = version_dir / "sections"
    sections_dir.mkdir(parents=True)
    (version_dir / "main.tex").write_text(
        r"""\documentclass{article}
\begin{document}
\section{Introduction}
\input{sections/body}
\begin{thebibliography}{9}
\bibitem{paper2020} A. Author. A useful paper. 2020.
\end{thebibliography}
\end{document}
""",
        encoding="utf-8",
    )
    (sections_dir / "body.tex").write_text(
        r"""This is the first sentence. This is the second sentence.

\begin{itemize}
\item One list item.
\end{itemize}
""",
        encoding="utf-8",
    )

    result = run_pipeline(
        [
            PublicationInput(
                "2501.01234",
                (VersionInput(1, version_dir),),
            )
        ],
        current_dependencies(),
        tmp_path / "real-run",
        collect_results=True,
    )

    publication = result.publications[0]
    assert publication.validation.passed
    assert publication.sources[0].root_file == "main.tex"
    assert publication.raw_trees[0].root is not None
    assert publication.normalized_versions[0].nodes
    assert publication.deduplicated.elements
    assert publication.references.entries
    assert result.statistics["publications"]["successful"] == 1
    assert (tmp_path / "real-run" / "elements.jsonl").exists()


def test_run_pipeline_streams_publication_results_by_default(tmp_path):
    source_dir = tmp_path / "v1"
    source_dir.mkdir()
    (source_dir / "main.tex").write_text(
        (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "Streaming output.\n"
            "\\end{document}\n"
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        [PublicationInput("2501.01234", (VersionInput(1, source_dir),))],
        current_dependencies(),
        tmp_path / "streaming-run",
        progress_enabled=False,
    )

    assert result.publications == ()
    assert result.publication_count == 1
    assert result.statistics["publications"]["total"] == 1


def test_parallel_pipeline_matches_serial_dataset_output(tmp_path):
    publications = []
    for number in (1, 2, 3):
        source_dir = tmp_path / f"v{number}"
        source_dir.mkdir()
        (source_dir / "main.tex").write_text(
            (
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                f"\\section{{Paper {number}}}\n"
                f"Text for paper {number}.\n"
                "\\end{document}\n"
            ),
            encoding="utf-8",
        )
        publications.append(
            PublicationInput(
                f"2501.0000{number}",
                (VersionInput(1, source_dir),),
            )
        )

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = run_pipeline(
        publications,
        current_dependencies(),
        serial_dir,
        progress_enabled=False,
        workers=1,
    )
    parallel = run_pipeline(
        publications,
        current_dependencies(),
        parallel_dir,
        progress_enabled=False,
        workers=2,
    )

    for filename in (
        "elements.jsonl",
        "hierarchy.jsonl",
        "bibtex.jsonl",
        "version_manifest.jsonl",
        "validation_report.json",
    ):
        assert (parallel_dir / filename).read_bytes() == (
            serial_dir / filename
        ).read_bytes()
    assert parallel.publications == ()
    assert parallel.publication_count == serial.publication_count == 3
    assert parallel.statistics["publications"] == serial.statistics["publications"]
    assert parallel.statistics["versions"] == serial.statistics["versions"]
    assert parallel.statistics["elements"] == serial.statistics["elements"]
    assert parallel.statistics["references"] == serial.statistics["references"]
    assert parallel.benchmark["stage_timings_seconds"]


def test_parallel_pipeline_rejects_retaining_large_results(tmp_path):
    with pytest.raises(ValueError, match="collect_results"):
        run_pipeline(
            (),
            current_dependencies(),
            tmp_path / "parallel",
            progress_enabled=False,
            workers=2,
            collect_results=True,
        )
