"""Cross-record validation for Milestone 2 parsing outputs.

The dataclasses in :mod:`src.parsing.contracts` validate individual fields.
This module validates relationships between records, versions, and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.parsing.contracts import (
    DeduplicatedPublication,
    Diagnostic,
    DiagnosticSeverity,
    NodeKind,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
)


@dataclass(frozen=True)
class ValidationReport:
    """Validation result that can be serialized as ``validation_report.json``."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.passed else "failed",
            "errors": [
                _diagnostic_to_dict(diagnostic)
                for diagnostic in self.errors
            ],
            "warnings": [
                _diagnostic_to_dict(diagnostic)
                for diagnostic in self.warnings
            ],
        }


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
) -> Diagnostic:
    return Diagnostic(severity=severity, code=code, message=message)


def _diagnostic_to_dict(diagnostic: Diagnostic) -> dict[str, object]:
    result: dict[str, object] = {
        "severity": diagnostic.severity.value,
        "code": diagnostic.code,
        "message": diagnostic.message,
    }
    if diagnostic.file_path is not None:
        result["file_path"] = diagnostic.file_path
    if diagnostic.line is not None:
        result["line"] = diagnostic.line
    return result


def validate_source_bundle(source: SourceBundle) -> tuple[Diagnostic, ...]:
    """Validate TV2's compilation-path result."""

    diagnostics: list[Diagnostic] = []
    file_paths = [source_file.relative_path for source_file in source.files]

    if len(file_paths) != len(set(file_paths)):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_SOURCE_FILE",
                "SourceBundle contains duplicate relative paths",
            )
        )

    if source.status is ProcessingStatus.EMPTY_NO_TEX:
        if source.root_file is not None or source.files:
            diagnostics.append(
                _diagnostic(
                    "INVALID_EMPTY_SOURCE",
                    "empty_no_tex source must have no root file or files",
                )
            )
    elif source.status is ProcessingStatus.OK:
        if source.root_file is None:
            diagnostics.append(
                _diagnostic(
                    "MISSING_ROOT_FILE",
                    "ok source must declare a root file",
                )
            )
        elif source.root_file not in file_paths:
            diagnostics.append(
                _diagnostic(
                    "ROOT_FILE_NOT_IN_PATH",
                    "root file is not present in the compilation path",
                )
            )

    return tuple(diagnostics)


def validate_raw_tree(tree: RawTree) -> tuple[Diagnostic, ...]:
    """Validate TV3's status/root relationship and raw node kinds."""

    diagnostics: list[Diagnostic] = []
    if tree.status in {
        ProcessingStatus.OK,
        ProcessingStatus.PARTIAL_PARSE,
    } and tree.root is None:
        diagnostics.append(
            _diagnostic(
                "MISSING_TREE_ROOT",
                f"{tree.status.value} raw tree must have a root",
            )
        )
    if tree.status in {
        ProcessingStatus.EMPTY_NO_TEX,
        ProcessingStatus.ROOT_NOT_FOUND,
    } and tree.root is not None:
        diagnostics.append(
            _diagnostic(
                "UNEXPECTED_TREE_ROOT",
                f"{tree.status.value} raw tree must not have a root",
            )
        )

    if tree.root is not None:
        allowed_kinds = {kind.value for kind in NodeKind}
        for node in _walk_tree(tree.root):
            if node.kind not in allowed_kinds:
                diagnostics.append(
                    _diagnostic(
                        "UNKNOWN_NODE_KIND",
                        f"unknown raw node kind: {node.kind!r}",
                    )
                )

    return tuple(diagnostics)


def _walk_tree(root: object):
    yield root
    for child in getattr(root, "children", ()):
        yield from _walk_tree(child)


def validate_reference_bundle(
    references: ReferenceBundle,
) -> tuple[Diagnostic, ...]:
    """Validate TV5's unified entries and citation maps."""

    diagnostics: list[Diagnostic] = []
    entry_keys = [entry.citation_key for entry in references.entries]
    if len(entry_keys) != len(set(entry_keys)):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_CITATION_KEY",
                "reference entries contain duplicate citation keys",
            )
        )

    known_keys = set(entry_keys)
    for version, mapping in references.citation_maps.items():
        for new_key in mapping.values():
            if new_key not in known_keys:
                diagnostics.append(
                    _diagnostic(
                        "UNKNOWN_CITATION_TARGET",
                        f"version {version} maps to unknown key {new_key!r}",
                    )
                )

    return tuple(diagnostics)


def validate_normalized_version(
    version: NormalizedVersion,
) -> tuple[Diagnostic, ...]:
    """Validate TV4's normalized nodes before publication deduplication."""

    diagnostics: list[Diagnostic] = []
    node_ids = [node.local_id for node in version.nodes]
    if len(node_ids) != len(set(node_ids)):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_LOCAL_NODE_ID",
                "normalized nodes contain duplicate local IDs",
            )
        )

    known_node_ids = set(node_ids)
    for node in version.nodes:
        if (
            node.parent_local_id is not None
            and node.parent_local_id not in known_node_ids
        ):
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_PARENT_NODE",
                    f"node {node.local_id!r} has an unknown parent",
                )
            )

    if version.status is ProcessingStatus.EMPTY_NO_TEX:
        if version.root_local_id is not None or version.nodes:
            diagnostics.append(
                _diagnostic(
                    "INVALID_EMPTY_VERSION",
                    "empty_no_tex version must have no root or nodes",
                )
            )
        if version.raw_element_count != 0:
            diagnostics.append(
                _diagnostic(
                    "INVALID_EMPTY_COUNT",
                    "empty_no_tex version must have zero raw elements",
                )
            )
    elif version.status in {
        ProcessingStatus.OK,
        ProcessingStatus.PARTIAL_PARSE,
    }:
        if version.root_local_id is None:
            diagnostics.append(
                _diagnostic(
                    "MISSING_NORMALIZED_ROOT",
                    f"{version.status.value} version must have a root",
                )
            )
        elif version.root_local_id not in known_node_ids:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_NORMALIZED_ROOT",
                    "root_local_id does not reference a normalized node",
                )
            )
        elif not any(
            node.local_id == version.root_local_id
            and node.kind == NodeKind.DOCUMENT.value
            for node in version.nodes
        ):
            diagnostics.append(
                _diagnostic(
                    "ROOT_NOT_DOCUMENT",
                    "root_local_id must reference a document node",
                )
            )
    elif (
        version.root_local_id is not None
        and version.root_local_id not in known_node_ids
    ):
        diagnostics.append(
            _diagnostic(
                "UNKNOWN_NORMALIZED_ROOT",
                "root_local_id does not reference a normalized node",
            )
        )

    return tuple(diagnostics)


def validate_publication(
    publication: DeduplicatedPublication,
) -> tuple[Diagnostic, ...]:
    """Validate TV4's final publication-level rows."""

    diagnostics: list[Diagnostic] = []
    version_numbers = [version.version for version in publication.versions]
    if len(version_numbers) != len(set(version_numbers)):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_VERSION",
                "publication contains duplicate version numbers",
            )
        )

    if any(
        version.arxiv_id != publication.arxiv_id
        for version in publication.versions
    ):
        diagnostics.append(
            _diagnostic(
                "VERSION_PUBLICATION_MISMATCH",
                "normalized version belongs to another publication",
            )
        )
    if any(row.arxiv_id != publication.arxiv_id for row in publication.elements):
        diagnostics.append(
            _diagnostic(
                "ELEMENT_PUBLICATION_MISMATCH",
                "element row belongs to another publication",
            )
        )
    if any(row.arxiv_id != publication.arxiv_id for row in publication.hierarchy):
        diagnostics.append(
            _diagnostic(
                "HIERARCHY_PUBLICATION_MISMATCH",
                "hierarchy row belongs to another publication",
            )
        )

    element_ids = [row.element_id for row in publication.elements]
    if len(element_ids) != len(set(element_ids)):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_ELEMENT_ID",
                "publication contains duplicate element IDs",
            )
        )

    known_ids = set(element_ids)
    known_versions = set(version_numbers)
    parents_by_version: dict[int, dict[str, str]] = {}
    for row in publication.hierarchy:
        if row.version not in known_versions:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_HIERARCHY_VERSION",
                    f"hierarchy references unknown version {row.version}",
                )
            )
        if row.element_id not in known_ids or row.parent_id not in known_ids:
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_HIERARCHY_ELEMENT",
                    "hierarchy references an unknown element ID",
                )
            )
        if row.element_id == row.parent_id:
            diagnostics.append(
                _diagnostic(
                    "SELF_PARENT_EDGE",
                    "hierarchy contains a self-parent edge",
                )
            )

        parent_map = parents_by_version.setdefault(row.version, {})
        if row.element_id in parent_map:
            diagnostics.append(
                _diagnostic(
                    "MULTIPLE_PARENTS",
                    f"element {row.element_id!r} has multiple parents",
                )
            )
        parent_map[row.element_id] = row.parent_id

    for parent_map in parents_by_version.values():
        if _contains_cycle(parent_map):
            diagnostics.append(
                _diagnostic(
                    "HIERARCHY_CYCLE",
                    "hierarchy contains a cycle",
                )
            )

    return tuple(diagnostics)


def _contains_cycle(parent_map: Mapping[str, str]) -> bool:
    for start in parent_map:
        visited: set[str] = set()
        current = start
        while current in parent_map:
            if current in visited:
                return True
            visited.add(current)
            current = parent_map[current]
    return False


def validate_all(
    *,
    source_bundles: Sequence[SourceBundle] = (),
    raw_trees: Sequence[RawTree] = (),
    reference_bundle: ReferenceBundle | None = None,
    normalized_versions: Sequence[NormalizedVersion] = (),
    publication: DeduplicatedPublication | None = None,
) -> ValidationReport:
    """Validate all available results from one pipeline run."""

    diagnostics: list[Diagnostic] = []
    for source in source_bundles:
        diagnostics.extend(validate_source_bundle(source))
    for tree in raw_trees:
        diagnostics.extend(validate_raw_tree(tree))
    if reference_bundle is not None:
        diagnostics.extend(validate_reference_bundle(reference_bundle))
    for version in normalized_versions:
        diagnostics.extend(validate_normalized_version(version))
    if publication is not None:
        diagnostics.extend(validate_publication(publication))
    return ValidationReport(tuple(diagnostics))


__all__ = [
    "ValidationReport",
    "validate_all",
    "validate_normalized_version",
    "validate_publication",
    "validate_raw_tree",
    "validate_reference_bundle",
    "validate_source_bundle",
]
