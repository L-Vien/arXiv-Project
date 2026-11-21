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


def test_malformed_bibitem_with_empty_key_is_skipped_with_diagnostic():
    source = SourceBundle(
        arxiv_id="2501.99998",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bbl",
                content=(
                    "\\begin{thebibliography}{1}\n"
                    "\\bibitem{   } Broken entry. 2020.\n"
                    "\\end{thebibliography}"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert bundle.entries == ()
    assert any(
        diagnostic.code == "BIBTEX_PARTIAL"
        for diagnostic in bundle.diagnostics
    )


def test_unstructured_bibitem_is_not_published_as_a_truncated_title():
    source = SourceBundle(
        arxiv_id="2501.99995",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    "\\begin{thebibliography}{1}\n"
                    "\\bibitem{long-citation} Doe, Jane and Roe, Richard. "
                    "A rendered bibliography string that is deliberately longer than one hundred characters "
                    "and cannot reliably be separated into an article title. 2024. doi:10.1000/long\n"
                    "\\end{thebibliography}"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert len(bundle.entries) == 1
    assert bundle.entries[0].title == ""
    assert bundle.entries[0].doi == "10.1000/long"
    assert any(d.code == "CITATION_TEXT_ONLY" for d in bundle.diagnostics)


def test_malformed_citation_key_is_replaced_but_preserved_in_citation_map():
    source = SourceBundle(
        arxiv_id="2501.99994",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content="@article{bad key, title={A Clean Title}, year={2024}}",
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert len(bundle.entries) == 1
    safe_key = bundle.entries[0].citation_key
    assert safe_key.startswith("ref_")
    assert bundle.citation_maps[1]["bad key"] == safe_key
    assert any(d.code == "INVALID_CITATION_KEY" for d in bundle.diagnostics)


def test_biber_bbl_entries_are_available_for_citation_mapping():
    source = SourceBundle(
        arxiv_id="2501.99997",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.bbl",
                content=(
                    "\\refsection{0}\n"
                    "\\entry{aleksanyan2024}{article}{}\n"
                    "  \\field{title}{A {Biber} Reference}\n"
                    "  \\field{year}{2024}\n"
                    "  \\field{doi}{10.1000/example}\n"
                    "\\endentry\n"
                    "\\endrefsection\n"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert [entry.citation_key for entry in bundle.entries] == ["aleksanyan2024"]
    assert bundle.entries[0].entry_type == "article"
    assert bundle.entries[0].title == "A Biber Reference"
    assert bundle.entries[0].year == 2024
    assert bundle.entries[0].doi == "10.1000/example"
    assert bundle.citation_maps[1] == {"aleksanyan2024": "aleksanyan2024"}


def test_harvard_bbl_entries_are_available_for_citation_mapping():
    source = SourceBundle(
        arxiv_id="2501.99996",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bbl",
                content=(
                    "\\harvarditem[Doe and Roe]{Doe, Jane and Roe, Richard}"
                    "{2024}{doe2024}\n"
                    "A Harvard style reference. doi:10.1000/harvard\n"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert [entry.citation_key for entry in bundle.entries] == ["doe2024"]
    assert bundle.entries[0].year == 2024
    assert bundle.entries[0].doi == "10.1000/harvard"
    assert bundle.citation_maps[1] == {"doe2024": "doe2024"}


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


def test_structured_bibtex_unwraps_formatting_and_latex_accents_for_matching_text():
    source = SourceBundle(
        arxiv_id="2501.99993",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@article{accented, "
                    "author={{Garc\\'i}a, Ana}, "
                    "title={{\\'E}tude on \\emph{T\\\"{o}rnqvist}}, "
                    "year={2024}}"
                ),
                order=0,
            ),
        ),
    )

    entry = process_references([source]).entries[0]

    assert entry.authors == ("García, Ana",)
    assert entry.title == "Étude on Törnqvist"


def test_dedup_keeps_complete_title_and_reports_same_key_identity_conflict():
    sources = [
        SourceBundle(
            arxiv_id="2501.99994",
            version=1,
            status=ProcessingStatus.OK,
            root_file="refs.bib",
            files=(
                SourceFile(
                    relative_path="refs.bib",
                    content=(
                        "@article{same, title={Complete Reference Title}, "
                        "author={Ada Lovelace}, year={2024}, doi={10.1000/one}}"
                    ),
                    order=0,
                ),
            ),
        ),
        SourceBundle(
            arxiv_id="2501.99994",
            version=2,
            status=ProcessingStatus.OK,
            root_file="refs.bib",
            files=(
                SourceFile(
                    relative_path="refs.bib",
                    content=(
                        "@article{same, title={Complete}, author={Ada Lovelace}, "
                        "year={2024}, doi={10.1000/two}}"
                    ),
                    order=0,
                ),
            ),
        ),
    ]

    bundle = process_references(sources)

    assert len(bundle.entries) == 2
    assert any(issue.code == "REFERENCE_IDENTITY_CONFLICT" for issue in bundle.diagnostics)


def test_malformed_bibtex_entry_emits_partial_diagnostic():
    source = SourceBundle(
        arxiv_id="2501.77777",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content="@article{broken, title={Unclosed title}",
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert len(bundle.entries) == 1
    assert bundle.entries[0].citation_key == "broken"
    diagnostics = [d for d in bundle.diagnostics if d.code == "BIBTEX_PARTIAL"]
    assert diagnostics
    assert diagnostics[0].arxiv_id == "2501.77777"
    assert diagnostics[0].version == 1


def test_valid_sparse_misc_entry_is_not_marked_partial():
    source = SourceBundle(
        arxiv_id="2501.77777",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@misc{supplemental,\n"
                    "  howpublished = {See Supplemental Material}\n"
                    "}\n"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert [entry.citation_key for entry in bundle.entries] == ["supplemental"]
    assert not any(d.code == "BIBTEX_PARTIAL" for d in bundle.diagnostics)


def test_comments_are_not_parsed_as_bibitems():
    source = SourceBundle(
        arxiv_id="2501.77778",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    "% \\bibitem{fake} Fake. 1900.\n"
                    "\\begin{verbatim}\n\\bibitem{also_fake}\n\\end{verbatim}\n"
                    "\\begin{thebibliography}{9}\n"
                    "\\bibitem{real} Real Author. Real Paper. 2020.\n"
                    "\\end{thebibliography}"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert [entry.citation_key for entry in bundle.entries] == ["real"]


def test_aps_bbl_bibitem_with_whitespace_and_optional_label_is_parsed():
    source = SourceBundle(
        arxiv_id="2501.77780",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.bbl",
                content=(
                    "\\begin{thebibliography}{2}\n"
                    "\\bibitem [{\\citenamefont {Qi}\\ and\\ "
                    "\\citenamefont {Zhang}(2011)}]{Qi2011}%\n"
                    "Qi and Zhang. Topological insulators. 2011.\n"
                    "\\bibitem\n"
                    "  [{\\citenamefont {Hasan}(2010)}]\n"
                    "  {Hasan2010}%\n"
                    "Hasan. Colloquium. 2010.\n"
                    "\\end{thebibliography}\n"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])

    assert [entry.citation_key for entry in bundle.entries] == [
        "Qi2011",
        "Hasan2010",
    ]
    assert not any(
        diagnostic.code == "BIBTEX_PARTIAL"
        for diagnostic in bundle.diagnostics
    )


def test_bib_entry_takes_precedence_over_generated_bbl_duplicate():
    source = SourceBundle(
        arxiv_id="2501.77779",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@article{same, title={Same Paper}, "
                    "author={Doe, Jane}, year={2020}, "
                    "doi={10.1234/same}}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path="main.bbl",
                content=(
                    "\\begin{thebibliography}{9}\n"
                    "\\bibitem{same} Doe, Jane. Same Paper. 2020.\n"
                    "\\end{thebibliography}"
                ),
                order=1,
            ),
        ),
    )

    bundle = process_references([source])

    assert len(bundle.entries) == 1
    assert bundle.entries[0].citation_key == "same"
    assert bundle.entries[0].doi == "10.1234/same"
    assert bundle.raw_entry_count == 1


def test_doi_normalization_decodes_escaped_urls_and_drops_invalid_values():
    source = SourceBundle(
        arxiv_id="2501.77777",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@article{encoded, title={Encoded DOI}, "
                    "doi={https://doi.org/10.1007\\%2Fs11590-024-02132-1}}\n"
                    "@misc{invalid, title={Not a DOI}, doi={2006.11894}}"
                ),
                order=0,
            ),
        ),
    )

    bundle = process_references([source])
    entries = {entry.citation_key: entry for entry in bundle.entries}

    assert entries["encoded"].doi == "10.1007/s11590-024-02132-1"
    assert entries["invalid"].doi == ""


def test_doi_and_arxiv_identifiers_are_normalized_for_deduplication():
    source_v1 = SourceBundle(
        arxiv_id="2501.66666",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@article{doi_old, author={Doe, Jane}, title={DOI Paper}, "
                    "year={2020}, doi={DOI:10.1234/ABC.1}}"
                ),
                order=0,
            ),
        ),
    )
    source_v2 = SourceBundle(
        arxiv_id="2501.66666",
        version=2,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@article{doi_new, author={Doe, Jane}, title={DOI Paper}, "
                    "year={2020}, doi={https://doi.org/10.1234/abc.1}}"
                ),
                order=0,
            ),
        ),
    )

    doi_bundle = process_references([source_v1, source_v2])
    assert len(doi_bundle.entries) == 1
    assert doi_bundle.entries[0].citation_key == "doi_old"
    assert doi_bundle.entries[0].doi == "10.1234/abc.1"
    assert doi_bundle.citation_maps[2]["doi_new"] == "doi_old"

    arxiv_v1 = SourceBundle(
        arxiv_id="2501.55555",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@misc{arxiv_old, title={ArXiv Paper}, year={2021}, "
                    "eprint={2101.12345}, archiveprefix={arXiv}}"
                ),
                order=0,
            ),
        ),
    )
    arxiv_v2 = SourceBundle(
        arxiv_id="2501.55555",
        version=2,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="refs.bib",
                content=(
                    "@misc{arxiv_new, title={ArXiv Paper}, year={2021}, "
                    "url={https://arxiv.org/abs/2101.12345v2}}"
                ),
                order=0,
            ),
        ),
    )

    arxiv_bundle = process_references([arxiv_v1, arxiv_v2])
    assert len(arxiv_bundle.entries) == 1
    assert arxiv_bundle.entries[0].citation_key == "arxiv_old"
    assert arxiv_bundle.citation_maps[2]["arxiv_new"] == "arxiv_old"


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
