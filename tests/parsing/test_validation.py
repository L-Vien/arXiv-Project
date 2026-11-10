from __future__ import annotations

import unittest

from src.parsing.contracts import (
    BibEntry,
    DeduplicatedPublication,
    ElementRow,
    HierarchyRow,
    NodeKind,
    NormalizedNode,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
    SourceFile,
    TreeNode,
)
from src.parsing.validation import (
    validate_all,
    validate_normalized_version,
    validate_publication,
    validate_raw_tree,
    validate_reference_bundle,
    validate_source_bundle,
)


def make_normalized_version(
    *,
    root_kind: str = NodeKind.DOCUMENT.value,
    parent_local_id: str | None = None,
    status: ProcessingStatus = ProcessingStatus.OK,
) -> NormalizedVersion:
    return NormalizedVersion(
        arxiv_id="2501.01234",
        version=1,
        status=status,
        root_local_id=None if status is ProcessingStatus.EMPTY_NO_TEX else "root-v1",
        nodes=(
            NormalizedNode(
                local_id="root-v1",
                kind=root_kind,
                content="Version 1 root",
                parent_local_id=parent_local_id,
                order=0,
            ),
        ) if status is not ProcessingStatus.EMPTY_NO_TEX else (),
        raw_element_count=0 if status is ProcessingStatus.EMPTY_NO_TEX else 1,
    )


class SourceValidationTests(unittest.TestCase):
    def test_empty_source_cannot_have_files(self) -> None:
        source = SourceBundle(
            arxiv_id="2501.01234",
            version=2,
            status=ProcessingStatus.EMPTY_NO_TEX,
            root_file="main.tex",
            files=(SourceFile("main.tex", "", 0),),
        )

        codes = {diagnostic.code for diagnostic in validate_source_bundle(source)}
        self.assertIn("INVALID_EMPTY_SOURCE", codes)

    def test_ok_source_requires_root_in_compilation_path(self) -> None:
        source = SourceBundle(
            arxiv_id="2501.01234",
            version=1,
            status=ProcessingStatus.OK,
            root_file="main.tex",
            files=(SourceFile("other.tex", "", 0),),
        )

        codes = {diagnostic.code for diagnostic in validate_source_bundle(source)}
        self.assertIn("ROOT_FILE_NOT_IN_PATH", codes)


class TreeValidationTests(unittest.TestCase):
    def test_successful_raw_tree_requires_root(self) -> None:
        tree = RawTree(
            arxiv_id="2501.01234",
            version=1,
            root=None,
            status=ProcessingStatus.OK,
        )

        codes = {diagnostic.code for diagnostic in validate_raw_tree(tree)}
        self.assertIn("MISSING_TREE_ROOT", codes)

    def test_empty_raw_tree_must_not_have_root(self) -> None:
        tree = RawTree(
            arxiv_id="2501.01234",
            version=2,
            root=TreeNode(kind=NodeKind.DOCUMENT),
            status=ProcessingStatus.EMPTY_NO_TEX,
        )

        codes = {diagnostic.code for diagnostic in validate_raw_tree(tree)}
        self.assertIn("UNEXPECTED_TREE_ROOT", codes)


class ReferenceValidationTests(unittest.TestCase):
    def test_duplicate_and_unknown_citation_keys_are_reported(self) -> None:
        references = ReferenceBundle(
            arxiv_id="2501.01234",
            entries=(
                BibEntry(citation_key="same", entry_type="article"),
                BibEntry(citation_key="same", entry_type="article"),
            ),
            citation_maps={1: {"old": "missing"}},
        )

        codes = {
            diagnostic.code
            for diagnostic in validate_reference_bundle(references)
        }
        self.assertEqual(
            codes,
            {"DUPLICATE_CITATION_KEY", "UNKNOWN_CITATION_TARGET"},
        )


class NormalizedValidationTests(unittest.TestCase):
    def test_unknown_parent_and_non_document_root_are_reported(self) -> None:
        version = make_normalized_version(
            root_kind=NodeKind.SENTENCE.value,
            parent_local_id="missing",
        )

        codes = {
            diagnostic.code
            for diagnostic in validate_normalized_version(version)
        }
        self.assertEqual(
            codes,
            {"UNKNOWN_PARENT_NODE", "ROOT_NOT_DOCUMENT"},
        )

    def test_empty_version_has_no_nodes_or_raw_elements(self) -> None:
        version = NormalizedVersion(
            arxiv_id="2501.01234",
            version=2,
            status=ProcessingStatus.EMPTY_NO_TEX,
            root_local_id="unexpected",
            nodes=(),
            raw_element_count=1,
        )

        codes = {
            diagnostic.code
            for diagnostic in validate_normalized_version(version)
        }
        self.assertEqual(
            codes,
            {"INVALID_EMPTY_VERSION", "INVALID_EMPTY_COUNT"},
        )


class PublicationValidationTests(unittest.TestCase):
    def test_unknown_hierarchy_ids_and_self_parent_are_reported(self) -> None:
        publication = DeduplicatedPublication(
            arxiv_id="2501.01234",
            versions=(make_normalized_version(),),
            elements=(
                ElementRow(
                    element_id="2501-01234#root-v1",
                    arxiv_id="2501.01234",
                    kind=NodeKind.DOCUMENT,
                    content="Version 1 root",
                ),
            ),
            hierarchy=(
                HierarchyRow(
                    arxiv_id="2501.01234",
                    version=1,
                    element_id="2501-01234#root-v1",
                    parent_id="2501-01234#missing",
                ),
                HierarchyRow(
                    arxiv_id="2501.01234",
                    version=1,
                    element_id="2501-01234#root-v1",
                    parent_id="2501-01234#root-v1",
                ),
            ),
        )

        codes = {diagnostic.code for diagnostic in validate_publication(publication)}
        self.assertIn("UNKNOWN_HIERARCHY_ELEMENT", codes)
        self.assertIn("SELF_PARENT_EDGE", codes)
        self.assertIn("MULTIPLE_PARENTS", codes)

    def test_cycle_is_reported_per_version(self) -> None:
        elements = (
            ElementRow("2501-01234#root", "2501.01234", "document", "root"),
            ElementRow("2501-01234#a", "2501.01234", "heading", "A"),
            ElementRow("2501-01234#b", "2501.01234", "heading", "B"),
        )
        publication = DeduplicatedPublication(
            arxiv_id="2501.01234",
            versions=(make_normalized_version(),),
            elements=elements,
            hierarchy=(
                HierarchyRow("2501.01234", 1, "2501-01234#a", "2501-01234#b"),
                HierarchyRow("2501.01234", 1, "2501-01234#b", "2501-01234#a"),
            ),
        )

        codes = {diagnostic.code for diagnostic in validate_publication(publication)}
        self.assertIn("HIERARCHY_CYCLE", codes)


class AggregateValidationTests(unittest.TestCase):
    def test_validate_all_returns_serializable_failed_report(self) -> None:
        report = validate_all(
            normalized_versions=(
                make_normalized_version(root_kind=NodeKind.SENTENCE.value),
            )
        )

        self.assertFalse(report.passed)
        self.assertEqual(report.to_dict()["status"], "failed")
        self.assertTrue(report.to_dict()["errors"])

