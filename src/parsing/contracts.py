"""Shared data contracts for Milestone 2 parsing.

This module defines the data exchanged by the parsing team.  It intentionally
contains no LaTeX parsing, normalization, deduplication, file I/O, or
benchmarking logic.  It performs only lightweight field-level checks; the
cross-record and hierarchy checks live in ``validation.py``.

The public contract is deliberately based on immutable dataclasses and enums:

``SourceBundle`` -> ``RawTree``
``Sequence[SourceBundle]`` -> ``ReferenceBundle``
``RawTree`` + citation map -> normalized/deduplicated rows
``ElementRow`` + ``HierarchyRow`` + ``BibtexRow`` -> exported tables

Lists that cross module boundaries are represented as tuples so that a
contract object cannot be changed accidentally after it has been handed to
another module.  Exporters may convert tuples to JSON/Parquet-compatible
lists at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

CONTRACT_VERSION = "1.0"
SCHEMA_VERSION = "1.0"

class ProcessingStatus(str, Enum):
    """Statuses shared by gathering, parsing, and reference processing."""

    OK = "ok"
    EMPTY_NO_TEX = "empty_no_tex"
    ROOT_NOT_FOUND = "root_not_found"
    MISSING_INCLUDE = "missing_include"
    PARTIAL_PARSE = "partial_parse"
    PARSE_ERROR = "parse_error"
    BIBTEX_PARTIAL = "bibtex_partial"
    ERROR = "error"


class DiagnosticSeverity(str, Enum):
    """Severity of a non-fatal or fatal processing diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class NodeKind(str, Enum):
    """Raw and final hierarchy node kinds.

    ``paragraph`` is a raw intermediate node.  It must not appear as a final
    element row; its prose is split into ``sentence`` rows by TV4.
    """

    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    SENTENCE = "sentence"
    BLOCK_FORMULA = "block_formula"
    FIGURE = "figure"


FINAL_ELEMENT_KINDS = frozenset(
    {
        NodeKind.DOCUMENT.value,
        NodeKind.HEADING.value,
        NodeKind.LIST.value,
        NodeKind.LIST_ITEM.value,
        NodeKind.SENTENCE.value,
        NodeKind.BLOCK_FORMULA.value,
        NodeKind.FIGURE.value,
    }
)

ELEMENT_COLUMNS = ("element_id", "arxiv_id", "kind", "content")
HIERARCHY_COLUMNS = ("arxiv_id", "version", "element_id", "parent_id")
BIBTEX_COLUMNS = (
    "arxiv_id",
    "citation_key",
    "entry_type",
    "title",
    "authors",
    "year",
    "doi",
    "journal",
    "url",
)


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _coerce_status(value: ProcessingStatus | str) -> ProcessingStatus:
    if isinstance(value, ProcessingStatus):
        return value
    try:
        return ProcessingStatus(value)
    except ValueError as exc:
        raise ValueError(f"unknown processing status: {value!r}") from exc


def _coerce_kind(value: NodeKind | str) -> str:
    if isinstance(value, NodeKind):
        return value.value
    _require_non_empty(value, "kind")
    return value


@dataclass(frozen=True)
class Diagnostic:
    """A structured warning or error that can be surfaced in reports."""

    severity: DiagnosticSeverity
    code: str
    message: str
    file_path: str | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, DiagnosticSeverity):
            try:
                object.__setattr__(
                    self,
                    "severity",
                    DiagnosticSeverity(self.severity),
                )
            except ValueError as exc:
                raise ValueError(
                    f"unknown diagnostic severity: {self.severity!r}"
                ) from exc
        _require_non_empty(self.code, "code")
        _require_non_empty(self.message, "message")
        if self.file_path is not None:
            _require_non_empty(self.file_path, "file_path")
        if self.line is not None:
            _require_positive(self.line, "line")


@dataclass(frozen=True)
class SourceFile:
    """One file on the resolved LaTeX compilation path."""

    relative_path: str
    content: str
    order: int

    def __post_init__(self) -> None:
        _require_non_empty(self.relative_path, "relative_path")
        if PurePosixPath(self.relative_path).is_absolute():
            raise ValueError("relative_path must be relative")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        _require_non_negative(self.order, "order")


@dataclass(frozen=True)
class SourceBundle:
    """Output of TV2 for one publication version."""

    arxiv_id: str
    version: int
    status: ProcessingStatus
    root_file: str | None
    files: tuple[SourceFile, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        _require_positive(self.version, "version")
        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)
        if self.root_file is not None:
            _require_non_empty(self.root_file, "root_file")
        if not isinstance(self.files, tuple):
            object.__setattr__(self, "files", tuple(self.files))
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

@dataclass(frozen=True)
class TreeNode:
    """A node in TV3's raw hierarchy tree."""

    kind: NodeKind | str
    content: str | None = None
    children: tuple["TreeNode", ...] = ()
    level: int | None = None
    source_file: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_kind(self.kind))
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None")
        if not isinstance(self.children, tuple):
            object.__setattr__(self, "children", tuple(self.children))
        if self.level is not None:
            _require_non_negative(self.level, "level")
        if self.source_file is not None:
            _require_non_empty(self.source_file, "source_file")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(dict(self.attributes)),
        )


@dataclass(frozen=True)
class RawTree:
    """Output of TV3 for one version."""

    arxiv_id: str
    version: int
    root: TreeNode | None
    status: ProcessingStatus
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        _require_positive(self.version, "version")
        status = _coerce_status(self.status)
        object.__setattr__(self, "status", status)
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
@dataclass(frozen=True)
class BibEntry:
    """A normalized bibliography entry before publication-level export."""

    citation_key: str
    entry_type: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str = ""
    journal: str = ""
    url: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.citation_key, "citation_key")
        _require_non_empty(self.entry_type, "entry_type")
        for field_name in ("title", "doi", "journal", "url"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        if not isinstance(self.authors, tuple):
            object.__setattr__(self, "authors", tuple(self.authors))
        if any(not isinstance(author, str) for author in self.authors):
            raise TypeError("authors must contain only strings")
        if self.year is not None:
            if isinstance(self.year, bool) or not isinstance(self.year, int):
                raise TypeError("year must be an integer or None")


@dataclass(frozen=True)
class ReferenceBundle:
    """Output of TV5 for all versions of one publication."""

    arxiv_id: str
    entries: tuple[BibEntry, ...]
    citation_maps: Mapping[int, Mapping[str, str]]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        if not isinstance(self.citation_maps, Mapping):
            raise TypeError("citation_maps must be a mapping")

        normalized_maps: dict[int, Mapping[str, str]] = {}
        for version, mapping in self.citation_maps.items():
            _require_positive(version, "citation map version")
            if not isinstance(mapping, Mapping):
                raise TypeError("each citation map must be a mapping")
            normalized_map = dict(mapping)
            for old_key, new_key in normalized_map.items():
                _require_non_empty(old_key, "citation map key")
                _require_non_empty(new_key, "citation map value")
            normalized_maps[version] = MappingProxyType(normalized_map)
        object.__setattr__(
            self,
            "citation_maps",
            MappingProxyType(normalized_maps),
        )


@dataclass(frozen=True)
class ElementRow:
    """One row of the final ``elements`` table."""

    element_id: str
    arxiv_id: str
    kind: NodeKind | str
    content: str

    def __post_init__(self) -> None:
        _require_non_empty(self.element_id, "element_id")
        _require_non_empty(self.arxiv_id, "arxiv_id")
        kind = _coerce_kind(self.kind)
        if kind not in FINAL_ELEMENT_KINDS:
            raise ValueError(
                f"kind {kind!r} is not valid for a final element row"
            )
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True)
class HierarchyRow:
    """One node-parent edge of the final ``hierarchy`` table."""

    arxiv_id: str
    version: int
    element_id: str
    parent_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        _require_positive(self.version, "version")
        _require_non_empty(self.element_id, "element_id")
        _require_non_empty(self.parent_id, "parent_id")
@dataclass(frozen=True)
class BibtexRow:
    """One row of the final ``bibtex`` table."""

    arxiv_id: str
    citation_key: str
    entry_type: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str
    journal: str
    url: str

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        entry = BibEntry(
            citation_key=self.citation_key,
            entry_type=self.entry_type,
            title=self.title,
            authors=self.authors,
            year=self.year,
            doi=self.doi,
            journal=self.journal,
            url=self.url,
        )
        object.__setattr__(self, "citation_key", entry.citation_key)
        object.__setattr__(self, "entry_type", entry.entry_type)
        object.__setattr__(self, "authors", entry.authors)


@dataclass(frozen=True)
class VersionManifestRow:
    """Internal coverage record for every publication version."""

    arxiv_id: str
    version: int
    status: ProcessingStatus
    root_file: str | None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        _require_positive(self.version, "version")
        object.__setattr__(self, "status", _coerce_status(self.status))
        if self.root_file is not None:
            _require_non_empty(self.root_file, "root_file")
        if self.error_message is not None:
            _require_non_empty(self.error_message, "error_message")


@dataclass(frozen=True)
class NormalizedNode:
    """One normalized node before publication-level deduplication."""

    local_id: str
    kind: NodeKind | str
    content: str
    parent_local_id: str | None
    order: int

    def __post_init__(self) -> None:
        _require_non_empty(self.local_id, "local_id")
        kind = _coerce_kind(self.kind)
        if kind not in FINAL_ELEMENT_KINDS:
            raise ValueError(
                f"kind {kind!r} is not valid for a normalized node"
            )
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.parent_local_id is not None:
            _require_non_empty(self.parent_local_id, "parent_local_id")
        _require_non_negative(self.order, "order")


@dataclass(frozen=True)
class NormalizedVersion:
    """Output of TV4 for one publication version."""

    arxiv_id: str
    version: int
    status: ProcessingStatus
    root_local_id: str | None
    nodes: tuple[NormalizedNode, ...]
    raw_element_count: int
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        _require_positive(self.version, "version")
        object.__setattr__(self, "status", _coerce_status(self.status))
        if not isinstance(self.nodes, tuple):
            object.__setattr__(self, "nodes", tuple(self.nodes))
        if not isinstance(self.diagnostics, tuple):
            object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _require_non_negative(self.raw_element_count, "raw_element_count")


@dataclass(frozen=True)
class DeduplicatedPublication:
    """Output of TV4 after deduplicating one publication's versions."""

    arxiv_id: str
    versions: tuple[NormalizedVersion, ...]
    elements: tuple[ElementRow, ...]
    hierarchy: tuple[HierarchyRow, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.arxiv_id, "arxiv_id")
        for field_name in ("versions", "elements", "hierarchy"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                object.__setattr__(self, field_name, tuple(value))


class SourceGatherer(Protocol):
    """Public interface implemented by TV2."""

    def gather(
        self,
        source_dir: Path,
        arxiv_id: str,
        version: int,
    ) -> SourceBundle:
        ...


class StructureParser(Protocol):
    """Public interface implemented by TV3."""

    def parse(self, source: SourceBundle) -> RawTree:
        ...


class ReferenceProcessor(Protocol):
    """Public interface implemented by TV5."""

    def process(self, sources: Sequence[SourceBundle]) -> ReferenceBundle:
        ...


class TreeNormalizer(Protocol):
    """Public interface implemented by TV4 before publication deduplication."""

    def normalize(
        self,
        tree: RawTree,
        citation_map: Mapping[str, str] | None = None,
    ) -> NormalizedVersion:
        ...


class PublicationDeduplicator(Protocol):
    """Public interface implemented by TV4's publication-level dedup step."""

    def deduplicate(
        self,
        versions: Sequence[NormalizedVersion],
        arxiv_id: str,
    ) -> DeduplicatedPublication:
        ...

__all__ = [
    "BIBTEX_COLUMNS",
    "CONTRACT_VERSION",
    "Diagnostic",
    "DiagnosticSeverity",
    "ELEMENT_COLUMNS",
    "ElementRow",
    "FINAL_ELEMENT_KINDS",
    "HIERARCHY_COLUMNS",
    "HierarchyRow",
    "BibEntry",
    "BibtexRow",
    "DeduplicatedPublication",
    "NodeKind",
    "NormalizedNode",
    "NormalizedVersion",
    "ProcessingStatus",
    "PublicationDeduplicator",
    "RawTree",
    "ReferenceBundle",
    "ReferenceProcessor",
    "SCHEMA_VERSION",
    "SourceBundle",
    "SourceFile",
    "SourceGatherer",
    "StructureParser",
    "TreeNode",
    "TreeNormalizer",
    "VersionManifestRow",
]
