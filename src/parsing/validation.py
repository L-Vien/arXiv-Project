"""Cross-record validation for Milestone 2 parsing outputs.

The dataclasses in :mod:`src.parsing.contracts` validate individual fields.
This module validates relationships between records, versions, and tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
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


_STATUS_PRIORITY = {
    ProcessingStatus.EMPTY_NO_TEX: 0,
    ProcessingStatus.OK: 1,
    ProcessingStatus.MISSING_INCLUDE: 2,
    ProcessingStatus.PARTIAL_PARSE: 2,
    ProcessingStatus.BIBTEX_PARTIAL: 2,
    ProcessingStatus.ROOT_NOT_FOUND: 3,
    ProcessingStatus.PARSE_ERROR: 3,
    ProcessingStatus.ERROR: 3,
}

_CITATION_RE = re.compile(
    r"\\(?P<command>[A-Za-z]*[Cc]ite[A-Za-z]*)"
    r"\*?(?:\s*\[[^\]]*\]){0,2}\s*\{(?P<keys>[^{}]+)\}"
)
_UNSAFE_CITATION_KEY = re.compile(
    r"\s|[\x00-\x1f\x7f]|\\|^(?:https?://|doi:)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_TITLE_FIELD_LEAK_RE = re.compile(
    r"(?:abstract|author|authors|booktitle|doi|eprint|journal|journaltitle|"
    r"pages|publisher|url|volume|year)\s*=",
    re.IGNORECASE,
)
_TITLE_CITATION_BLOB_RE = re.compile(
    r"(?:\bet\s+al\.?\b.*\b(?:19|20)\d{2}\b.*https?://|"
    r"^(?:https?://|doi\s*:))",
    re.IGNORECASE,
)

_PARTIAL_DIAGNOSTIC_STATUSES = {
    "BIBTEX_PARTIAL": ProcessingStatus.BIBTEX_PARTIAL,
    "UNRESOLVED_CITATION": ProcessingStatus.BIBTEX_PARTIAL,
    "EMPTY_PARSED_DOCUMENT": ProcessingStatus.PARTIAL_PARSE,
    "ROOT_SELECTION_LOW_CONFIDENCE": ProcessingStatus.PARTIAL_PARSE,
}


def effective_status(*statuses: ProcessingStatus) -> ProcessingStatus:
    """Return the worst status observed across the M2 processing stages.

    TV2, TV3, and TV4 each report the status of their own stage.  The final
    version status must not hide a later parse/normalization failure behind an
    earlier ``ok`` gathering status.
    """

    if not statuses:
        raise ValueError("at least one processing status is required")
    return max(statuses, key=lambda status: _STATUS_PRIORITY[status])


def validation_status_for_version(
    validation: "ValidationReport",
    version: int,
) -> ProcessingStatus | None:
    """Return a partial status implied by version-scoped validation warnings."""

    statuses = [
        _PARTIAL_DIAGNOSTIC_STATUSES[diagnostic.code]
        for diagnostic in validation.diagnostics
        if diagnostic.version == version
        and diagnostic.code in _PARTIAL_DIAGNOSTIC_STATUSES
    ]
    return effective_status(*statuses) if statuses else None


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
    if diagnostic.arxiv_id is not None:
        result["arxiv_id"] = diagnostic.arxiv_id
    if diagnostic.version is not None:
        result["version"] = diagnostic.version
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
    elif source.status is ProcessingStatus.ROOT_NOT_FOUND:
        if source.root_file is not None or source.files:
            diagnostics.append(
                _diagnostic(
                    "INVALID_ROOT_NOT_FOUND_SOURCE",
                    "root_not_found source must have no root file or files",
                )
            )
    elif source.status in {
        ProcessingStatus.OK,
        ProcessingStatus.MISSING_INCLUDE,
        ProcessingStatus.PARTIAL_PARSE,
        ProcessingStatus.BIBTEX_PARTIAL,
    }:
        if source.root_file is None:
            diagnostics.append(
                _diagnostic(
                    "MISSING_ROOT_FILE",
                    f"{source.status.value} source must declare a root file",
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

    for entry in references.entries:
        if _UNSAFE_CITATION_KEY.search(entry.citation_key):
            diagnostics.append(
                _diagnostic(
                    "INVALID_CITATION_KEY",
                    f"citation key is not safe for a published table: {entry.citation_key!r}",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        if entry.year is not None and not 1600 <= entry.year <= date.today().year + 1:
            diagnostics.append(
                _diagnostic(
                    "INVALID_YEAR",
                    f"year is outside the accepted range: {entry.year}",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        if entry.doi and not _DOI_RE.fullmatch(entry.doi):
            diagnostics.append(
                _diagnostic(
                    "INVALID_DOI",
                    f"DOI does not have canonical syntax: {entry.doi!r}",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        if any(
            _LATEX_COMMAND_RE.search(value)
            for value in (entry.title, entry.journal, *entry.authors)
            if value
        ):
            diagnostics.append(
                _diagnostic(
                    "RESIDUAL_LATEX",
                    f"entry {entry.citation_key!r} still contains a LaTeX command",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        contaminated_values = (entry.title, entry.journal, *entry.authors)
        if any(
            value and _TITLE_FIELD_LEAK_RE.search(value)
            for value in contaminated_values
        ):
            diagnostics.append(
                _diagnostic(
                    "FIELD_BOUNDARY_LEAK",
                    f"entry {entry.citation_key!r} contains another field assignment inside a value",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
        if entry.title and _TITLE_CITATION_BLOB_RE.search(entry.title):
            diagnostics.append(
                _diagnostic(
                    "TITLE_IS_CITATION_BLOB",
                    f"entry {entry.citation_key!r} title looks like a rendered citation",
                    severity=DiagnosticSeverity.WARNING,
                )
            )

    return tuple(diagnostics)


def validate_citation_coverage(
    references: ReferenceBundle,
    versions: Sequence[NormalizedVersion],
) -> tuple[Diagnostic, ...]:
    """Report canonical citation keys missing from the unified bibliography."""

    known_keys = {entry.citation_key for entry in references.entries}
    diagnostics: list[Diagnostic] = []
    for version in versions:
        unresolved: set[str] = set()
        for node in version.nodes:
            for match in _CITATION_RE.finditer(node.content):
                unresolved.update(
                    key.strip()
                    for key in match.group("keys").split(",")
                    if (
                        key.strip()
                        and not (
                            match.group("command").casefold() == "nocite"
                            and key.strip() == "*"
                        )
                        and key.strip() not in known_keys
                    )
                )
        if unresolved:
            sample = ", ".join(sorted(unresolved)[:5])
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="UNRESOLVED_CITATION",
                    message=(
                        f"{len(unresolved)} citation key(s) have no canonical "
                        f"BibTeX entry: {sample}"
                    ),
                    arxiv_id=version.arxiv_id,
                    version=version.version,
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

    if version.status in {
        ProcessingStatus.EMPTY_NO_TEX,
        ProcessingStatus.ROOT_NOT_FOUND,
    }:
        if version.root_local_id is not None or version.nodes:
            diagnostics.append(
                _diagnostic(
                    "INVALID_EMPTY_VERSION",
                    f"{version.status.value} version must have no root or nodes",
                )
            )
        if version.raw_element_count != 0:
            diagnostics.append(
                _diagnostic(
                    "INVALID_EMPTY_COUNT",
                    f"{version.status.value} version must have zero raw elements",
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
        if (
            len(version.nodes) == 1
            and version.nodes[0].kind == NodeKind.DOCUMENT.value
            and not version.nodes[0].content.strip()
        ):
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="EMPTY_PARSED_DOCUMENT",
                    message=(
                        "Non-empty source produced only an empty document root"
                    ),
                    arxiv_id=version.arxiv_id,
                    version=version.version,
                )
            )
        if (
            version.root_local_id is not None
            and version.root_local_id not in known_node_ids
        ):
            diagnostics.append(
                _diagnostic(
                    "UNKNOWN_NORMALIZED_ROOT",
                    "root_local_id does not reference a normalized node",
                )
            )
        elif version.root_local_id is not None and not any(
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
    if reference_bundle is not None and normalized_versions:
        diagnostics.extend(
            validate_citation_coverage(
                reference_bundle,
                normalized_versions,
            )
        )
    if publication is not None:
        diagnostics.extend(validate_publication(publication))
    return ValidationReport(tuple(diagnostics))


__all__ = [
    "ValidationReport",
    "effective_status",
    "validate_all",
    "validate_citation_coverage",
    "validate_normalized_version",
    "validate_publication",
    "validate_raw_tree",
    "validate_reference_bundle",
    "validate_source_bundle",
    "validation_status_for_version",
]
