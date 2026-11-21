"""Deterministic JSONL export for Milestone 2 outputs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from src.parsing.contracts import (
    BibEntry,
    BibtexRow,
    DeduplicatedPublication,
    ElementRow,
    HierarchyRow,
    ReferenceBundle,
    RawTree,
    SourceBundle,
    NormalizedVersion,
    VersionManifestRow,
)
from src.parsing.validation import (
    ValidationReport,
    effective_status,
    validation_status_for_version,
)


def _atomic_write_chunks(path: Path, chunks: Iterable[str]) -> None:
    """Atomically write text chunks without materializing the whole file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(chunk)
        Path(temporary_name).replace(path)
    except BaseException:
        os.unlink(temporary_name)
        raise


def _atomic_write(path: Path, content: str) -> None:
    _atomic_write_chunks(path, (content,))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON object atomically and deterministically."""

    content = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _atomic_write(Path(path), content)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_chunks(
        Path(path),
        (
            json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
            for row in rows
        ),
    )


def _write_jsonl_rows(
    handle: TextIO,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write JSONL rows to an already-open stream."""

    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
        handle.write("\n")


def _element_row(row: ElementRow) -> dict[str, Any]:
    return {
        "element_id": row.element_id,
        "arxiv_id": row.arxiv_id,
        "kind": row.kind,
        "content": row.content,
    }


def _hierarchy_row(row: HierarchyRow) -> dict[str, Any]:
    return {
        "arxiv_id": row.arxiv_id,
        "version": row.version,
        "element_id": row.element_id,
        "parent_id": row.parent_id,
    }


def _bibtex_row(arxiv_id: str, entry: BibEntry) -> dict[str, Any]:
    row = BibtexRow(
        arxiv_id=arxiv_id,
        citation_key=entry.citation_key,
        entry_type=entry.entry_type,
        title=entry.title,
        authors=entry.authors,
        year=entry.year,
        doi=entry.doi,
        journal=entry.journal,
        url=entry.url,
    )
    return {
        "arxiv_id": row.arxiv_id,
        "citation_key": row.citation_key,
        "entry_type": row.entry_type,
        "title": row.title,
        "authors": list(row.authors),
        "year": row.year,
        "doi": row.doi,
        "journal": row.journal,
        "url": row.url,
    }


def _bibtex_provenance_row(arxiv_id: str, entry: BibEntry) -> dict[str, Any]:
    """Internal audit metadata; it is intentionally not part of final Parquet."""

    return {
        "arxiv_id": arxiv_id,
        "citation_key": entry.citation_key,
        "raw_citation_key": entry.raw_citation_key,
        "source_kind": entry.source_kind,
        "parse_status": entry.parse_status,
    }


def _manifest_row(
    source: SourceBundle,
    raw_tree: RawTree | None = None,
    normalized_version: NormalizedVersion | None = None,
    validation_report: ValidationReport | None = None,
) -> VersionManifestRow:
    statuses = [source.status]
    if raw_tree is not None:
        statuses.append(raw_tree.status)
    if normalized_version is not None:
        statuses.append(normalized_version.status)
    if validation_report is not None:
        validation_status = validation_status_for_version(
            validation_report,
            source.version,
        )
        if validation_status is not None:
            statuses.append(validation_status)
    diagnostics = (
        *source.diagnostics,
        *(raw_tree.diagnostics if raw_tree is not None else ()),
        *(
            normalized_version.diagnostics
            if normalized_version is not None
            else ()
        ),
    )
    error_message = next(
        (
            diagnostic.message
            for diagnostic in diagnostics
            if diagnostic.severity.value == "error"
        ),
        None,
    )
    return VersionManifestRow(
        arxiv_id=source.arxiv_id,
        version=source.version,
        status=effective_status(*statuses),
        root_file=source.root_file,
        error_message=error_message,
    )


def _manifest_dict(row: VersionManifestRow) -> dict[str, Any]:
    return {
        "arxiv_id": row.arxiv_id,
        "version": row.version,
        "status": row.status.value,
        "root_file": row.root_file,
        "error_message": row.error_message,
    }


def _publication_dir(output_dir: Path, arxiv_id: str) -> Path:
    safe_id = arxiv_id.replace("/", "-").replace(".", "-")
    return output_dir / "publications" / safe_id


class PublicationBatchWriter:
    """Write many publications into a small deterministic batch shard.

    The historical exporter creates five files per publication.  The pipeline
    uses this writer so a worker creates five files per batch instead, while
    direct callers can continue using :func:`export_publication` unchanged.
    """

    def __init__(self, shard_dir: Path, batch_index: int) -> None:
        self.batch_dir = (
            Path(shard_dir) / "batches" / f"part-{batch_index:06d}"
        )
        self._handles: dict[str, TextIO] = {}

    def __enter__(self) -> "PublicationBatchWriter":
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "elements.jsonl",
            "hierarchy.jsonl",
            "bibtex.jsonl",
            "bibtex_provenance.jsonl",
            "version_manifest.jsonl",
            "validation_reports.jsonl",
        ):
            self._handles[filename] = (self.batch_dir / filename).open(
                "w",
                encoding="utf-8",
                newline="\n",
            )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def write_publication(
        self,
        *,
        publication: DeduplicatedPublication,
        references: ReferenceBundle,
        sources: tuple[SourceBundle, ...],
        validation_report: ValidationReport,
        raw_trees: tuple[RawTree, ...] = (),
        normalized_versions: tuple[NormalizedVersion, ...] = (),
    ) -> None:
        """Append one publication to this batch's five open streams."""

        if len(self._handles) != 6:
            raise RuntimeError(
                "PublicationBatchWriter must be used as a context manager"
            )
        raw_trees_by_version = {tree.version: tree for tree in raw_trees}
        normalized_by_version = {
            version.version: version for version in normalized_versions
        }
        _write_jsonl_rows(
            self._handles["elements.jsonl"],
            (_element_row(row) for row in publication.elements),
        )
        _write_jsonl_rows(
            self._handles["hierarchy.jsonl"],
            (_hierarchy_row(row) for row in publication.hierarchy),
        )
        _write_jsonl_rows(
            self._handles["bibtex.jsonl"],
            (
                _bibtex_row(publication.arxiv_id, entry)
                for entry in references.entries
            ),
        )
        _write_jsonl_rows(
            self._handles["bibtex_provenance.jsonl"],
            (
                _bibtex_provenance_row(publication.arxiv_id, entry)
                for entry in references.entries
            ),
        )
        _write_jsonl_rows(
            self._handles["version_manifest.jsonl"],
            (
                _manifest_dict(
                    _manifest_row(
                        source,
                        raw_trees_by_version.get(source.version),
                        normalized_by_version.get(source.version),
                        validation_report,
                    )
                )
                for source in sources
            ),
        )
        _write_jsonl_rows(
            self._handles["validation_reports.jsonl"],
            (
                {
                    "arxiv_id": publication.arxiv_id,
                    "report": validation_report.to_dict(),
                },
            ),
        )


def export_publication(
    *,
    publication: DeduplicatedPublication,
    references: ReferenceBundle,
    sources: tuple[SourceBundle, ...],
    validation_report: ValidationReport,
    output_dir: Path,
    shard_dir: Path | None = None,
    raw_trees: tuple[RawTree, ...] = (),
    normalized_versions: tuple[NormalizedVersion, ...] = (),
) -> Path:
    """Write one publication shard using the final contract rows.

    ``shard_dir`` is an internal staging location used by the full pipeline.
    When omitted, the historical ``output_dir/publications`` location is kept
    for direct callers and unit tests.
    """

    shard_base = Path(output_dir) if shard_dir is None else Path(shard_dir)
    publication_dir = _publication_dir(shard_base, publication.arxiv_id)
    raw_trees_by_version = {tree.version: tree for tree in raw_trees}
    normalized_by_version = {
        version.version: version for version in normalized_versions
    }
    _write_jsonl(
        publication_dir / "elements.jsonl",
        (_element_row(row) for row in publication.elements),
    )
    _write_jsonl(
        publication_dir / "hierarchy.jsonl",
        (_hierarchy_row(row) for row in publication.hierarchy),
    )
    _write_jsonl(
        publication_dir / "bibtex.jsonl",
        (
            _bibtex_row(publication.arxiv_id, entry)
            for entry in references.entries
        ),
    )
    _write_jsonl(
        publication_dir / "bibtex_provenance.jsonl",
        (
            _bibtex_provenance_row(publication.arxiv_id, entry)
            for entry in references.entries
        ),
    )
    _write_jsonl(
        publication_dir / "version_manifest.jsonl",
        (
            _manifest_dict(
                _manifest_row(
                    source,
                    raw_trees_by_version.get(source.version),
                    normalized_by_version.get(source.version),
                    validation_report,
                )
            )
            for source in sources
        ),
    )
    write_json(
        publication_dir / "validation_report.json",
        validation_report.to_dict(),
    )
    return publication_dir


def _merge_jsonl_shards(
    *,
    output_path: Path,
    shard_root: Path,
    filename: str,
) -> None:
    """Concatenate deterministic JSONL shards without loading their rows."""

    def lines() -> Iterable[str]:
        if not shard_root.exists():
            return
        for shard in sorted(path for path in shard_root.iterdir() if path.is_dir()):
            shard_file = shard / filename
            if not shard_file.is_file():
                continue
            with shard_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield line if line.endswith("\n") else f"{line}\n"

    _atomic_write_chunks(output_path, lines())


def _merge_validation_reports(shard_root: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if shard_root.exists():
        for shard in sorted(path for path in shard_root.iterdir() if path.is_dir()):
            batch_reports_path = shard / "validation_reports.jsonl"
            if batch_reports_path.is_file():
                with batch_reports_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        report = json.loads(line)["report"]
                        errors.extend(report.get("errors", []))
                        warnings.extend(report.get("warnings", []))
                continue
            report_path = shard / "validation_report.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            errors.extend(report.get("errors", []))
            warnings.extend(report.get("warnings", []))
    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
    }


def finalize_run(
    output_dir: Path,
    *,
    shard_dir: Path | None = None,
) -> None:
    """Merge publication shards into one canonical run-level output."""

    output_dir = Path(output_dir)
    shard_base = output_dir if shard_dir is None else Path(shard_dir)
    batch_root = shard_base / "batches"
    publication_root = shard_base / "publications"
    shard_root = batch_root if batch_root.exists() else publication_root
    for filename in (
        "elements.jsonl",
        "hierarchy.jsonl",
        "bibtex.jsonl",
        "bibtex_provenance.jsonl",
        "version_manifest.jsonl",
    ):
        _merge_jsonl_shards(
            output_path=output_dir / filename,
            shard_root=shard_root,
            filename=filename,
        )
    write_json(
        output_dir / "validation_report.json",
        _merge_validation_reports(shard_root),
    )


__all__ = [
    "PublicationBatchWriter",
    "export_publication",
    "finalize_run",
    "write_json",
]
