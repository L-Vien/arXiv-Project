"""Unit tests for TV5 reference extraction, deduplication, and citation rewriting."""

from __future__ import annotations

import pytest

from src.parsing.contracts import (
    BibEntry,
    DiagnosticSeverity,
    ProcessingStatus,
    SourceBundle,
    SourceFile,
)
from src.parsing.references import process_references
from src.parsing.citation_rewrite import rewrite_citations
from src.parsing.validation import validate_reference_bundle


def test_process_references_bib_and_bibitem():
    # Test bundle for version 1 with a .bib file and version 2 with a \bibitem
    source_v1 = SourceBundle(
        arxiv_id="2501.01234",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content="\\documentclass{article}\\begin{document}\\cite{vaswani2017}\\end{document}",
                order=0,
            ),
            SourceFile(
                relative_path="refs.bib",
                content="""
                @article{vaswani2017,
                  author = {Vaswani, Ashish and Shazeer, Noam},
                  title = {Attention Is All You Need % important paper},
                  year = {2017},
                  doi = {10.5555/3295222.3295349},
                  url = {https://arxiv.org/abs/1706.03762}
                }
                """,
                order=1,
            ),
        ),
    )

    source_v2 = SourceBundle(
        arxiv_id="2501.01234",
        version=2,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content="\\documentclass{article}\\begin{document}\\cite{vaswani_attn}\\end{document}",
                order=0,
            ),
            SourceFile(
                relative_path="main.tex",
                content="\\begin{thebibliography}{9}\n\\bibitem{vaswani_attn} A. Vaswani et al. Attention Is All You Need, 2017. doi:10.5555/3295222.3295349\n\\end{thebibliography}",
                order=1,
            ),
        ),
    )

    bundle = process_references([source_v1, source_v2])

    assert bundle.arxiv_id == "2501.01234"
    # Should deduplicate into 1 canonical entry based on DOI
    assert len(bundle.entries) == 1
    entry = bundle.entries[0]
    assert entry.citation_key == "vaswani2017"
    assert entry.year == 2017
    assert entry.doi == "10.5555/3295222.3295349"
    assert entry.authors == ("Vaswani, Ashish", "Shazeer, Noam")
    assert entry.title == "Attention Is All You Need"
    assert entry.url == "https://arxiv.org/abs/1706.03762"

    # Check citation maps
    assert bundle.citation_maps[1]["vaswani2017"] == "vaswani2017"
    assert bundle.citation_maps[2]["vaswani_attn"] == "vaswani2017"

    # Validate against validation.py
    diagnostics = validate_reference_bundle(bundle)
    assert not diagnostics


def test_citation_rewrite():
    content = "As shown in \\cite{vaswani_attn, devlin2018}, models achieve high accuracy."
    citation_map = {
        "vaswani_attn": "vaswani2017",
        "devlin2018": "devlin2018",
    }

    rewritten = rewrite_citations(content, citation_map)
    assert rewritten == "As shown in \\cite{vaswani2017,devlin2018}, models achieve high accuracy."


def test_advanced_bib_features():
    source = SourceBundle(
        arxiv_id="2501.99999",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="references.bib",
                content="""
                @misc{org2023,
                  author = {{National Research and Development Agency} and Smith, John},
                  title = "Quoted Title with, comma",
                  year = {2023},
                  howpublished = {\\url{https://example.org/report}},
                  note = {See also \\href{https://doi.org/10.1234/test}{DOI link}}
                }
                """,
                order=0,
            ),
        ),
    )
    bundle = process_references([source])
    assert len(bundle.entries) == 1
    e = bundle.entries[0]
    assert e.citation_key == "org2023"
    assert e.authors == ("National Research and Development Agency", "Smith, John")
    assert e.title == "Quoted Title with, comma"
    assert e.year == 2023
    assert e.url == "https://example.org/report"


def test_bbl_file_and_empty_version_citation_map():
    source_v1 = SourceBundle(
        arxiv_id="2501.88888",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bbl",
                content="\\begin{thebibliography}{1}\n\\bibitem{ref1} Author A. Paper Title. 2021.\n\\end{thebibliography}",
                order=0,
            ),
        ),
    )
    # Version 2 has no references/files
    source_v2 = SourceBundle(
        arxiv_id="2501.88888",
        version=2,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content="\\documentclass{article}\\begin{document}Hello\\end{document}",
                order=0,
            ),
        ),
    )

    bundle = process_references([source_v1, source_v2])
    assert 1 in bundle.citation_maps
    assert 2 in bundle.citation_maps
    assert bundle.citation_maps[2] == {}
