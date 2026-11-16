from __future__ import annotations

import unittest

from src.parsing import DeduplicatedPublication as ExportedDeduplicatedPublication
from src.parsing import NormalizedVersion as ExportedNormalizedVersion
from src.parsing.contracts import (
    BibEntry,
    BibtexRow,
    DeduplicatedPublication,
    Diagnostic,
    DiagnosticSeverity,
    ElementRow,
    NodeKind,
    NormalizedNode,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
    SourceFile,
    TreeNode,
    VersionManifestRow,
)


class PublicExportTests(unittest.TestCase):
    def test_normalized_contracts_are_publicly_exported(self) -> None:
        self.assertIs(ExportedNormalizedVersion, NormalizedVersion)
        self.assertIs(ExportedDeduplicatedPublication, DeduplicatedPublication)


class SourceContractTests(unittest.TestCase):
    def test_source_bundle_keeps_compilation_order(self) -> None:
        bundle = SourceBundle(
            arxiv_id="2501.01234",
            version=1,
            status=ProcessingStatus.OK,
            root_file="main.tex",
            files=(
                SourceFile("main.tex", r"\\documentclass{article}", 0),
                SourceFile("sections/intro.tex", "Intro.", 1),
            ),
        )

        self.assertEqual(bundle.files[1].relative_path, "sections/intro.tex")
        self.assertEqual(bundle.files[1].order, 1)

    def test_empty_source_bundle_has_no_files(self) -> None:
        bundle = SourceBundle(
            arxiv_id="2501.01234",
            version=2,
            status="empty_no_tex",
            root_file=None,
        )

        self.assertEqual(bundle.status, ProcessingStatus.EMPTY_NO_TEX)
        self.assertEqual(bundle.files, ())

    def test_invalid_root_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SourceFile("/tmp/main.tex", "", 0)

class TreeContractTests(unittest.TestCase):
    def test_raw_tree_accepts_nested_nodes_and_coerces_kind(self) -> None:
        root = TreeNode(
            kind=NodeKind.DOCUMENT,
            children=(
                TreeNode(
                    kind="heading",
                    content="Introduction",
                    level=1,
                ),
                TreeNode(kind="paragraph", content="A sentence."),
            ),
        )
        tree = RawTree(
            arxiv_id="2501.01234",
            version=1,
            root=root,
            status="ok",
        )

        self.assertEqual(tree.root.kind, NodeKind.DOCUMENT.value)
        self.assertEqual(tree.root.children[0].kind, NodeKind.HEADING.value)

    def test_tree_attributes_are_read_only(self) -> None:
        node = TreeNode(
            kind=NodeKind.DOCUMENT,
            attributes={"role": "root"},
        )

        with self.assertRaises(TypeError):
            node.attributes["role"] = "changed"

    def test_empty_tree_has_no_root(self) -> None:
        tree = RawTree(
            arxiv_id="2501.01234",
            version=2,
            root=None,
            status=ProcessingStatus.EMPTY_NO_TEX,
        )

        self.assertIsNone(tree.root)

class OutputContractTests(unittest.TestCase):
    def test_final_element_rejects_raw_paragraph_kind(self) -> None:
        with self.assertRaises(ValueError):
            ElementRow(
                element_id="2501-01234#p1",
                arxiv_id="2501.01234",
                kind=NodeKind.PARAGRAPH,
                content="raw paragraph",
            )

    def test_bibtex_row_accepts_list_authors(self) -> None:
        row = BibtexRow(
            arxiv_id="2501.01234",
            citation_key="vaswani2017",
            entry_type="article",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani"],
            year=2017,
            doi="10.48550/arXiv.1706.03762",
            journal="",
            url="",
        )

        self.assertEqual(row.authors, ("Ashish Vaswani",))

    def test_normalized_version_requires_a_valid_root(self) -> None:
        root = NormalizedNode(
            local_id="root-v1",
            kind=NodeKind.DOCUMENT.value,
            content="Version 1 root",
            parent_local_id=None,
            order=0,
        )
        normalized = NormalizedVersion(
            arxiv_id="2501.01234",
            version=1,
            status=ProcessingStatus.OK,
            root_local_id="root-v1",
            nodes=(root,),
            raw_element_count=1,
        )

        self.assertEqual(normalized.nodes[0].local_id, "root-v1")

    def test_normalized_version_rejects_raw_paragraph(self) -> None:
        with self.assertRaises(ValueError):
            NormalizedNode(
                local_id="p1",
                kind=NodeKind.PARAGRAPH.value,
                content="raw paragraph",
                parent_local_id=None,
                order=0,
            )

class ReferenceContractTests(unittest.TestCase):
    def test_reference_bundle_keeps_version_specific_citation_maps(self) -> None:
        bundle = ReferenceBundle(
            arxiv_id="2501.01234",
            entries=(
                BibEntry(
                    citation_key="vaswani2017",
                    entry_type="article",
                    title="Attention Is All You Need",
                    authors=("Ashish Vaswani",),
                    year=2017,
                ),
            ),
            citation_maps={
                1: {"transformer": "vaswani2017"},
                2: {"vaswani2017": "vaswani2017"},
            },
        )

        self.assertEqual(bundle.citation_maps[1]["transformer"], "vaswani2017")

    def test_reference_maps_are_read_only(self) -> None:
        bundle = ReferenceBundle(
            arxiv_id="2501.01234",
            entries=(BibEntry(citation_key="canonical", entry_type="article"),),
            citation_maps={1: {"old": "canonical"}},
        )

        with self.assertRaises(TypeError):
            bundle.citation_maps[1]["old"] = "changed"


class DiagnosticAndManifestTests(unittest.TestCase):
    def test_diagnostic_accepts_string_severity(self) -> None:
        diagnostic = Diagnostic(
            severity="warning",
            code="MISSING_INCLUDE",
            message="Included file was not found",
            file_path="sections/method.tex",
            line=12,
        )

        self.assertEqual(diagnostic.severity, DiagnosticSeverity.WARNING)

    def test_manifest_preserves_empty_version_status(self) -> None:
        manifest = VersionManifestRow(
            arxiv_id="2501.01234",
            version=3,
            status=ProcessingStatus.EMPTY_NO_TEX,
            root_file=None,
        )

        self.assertEqual(manifest.status, ProcessingStatus.EMPTY_NO_TEX)

