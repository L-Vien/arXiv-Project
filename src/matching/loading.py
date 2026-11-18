"""Streaming loaders for the frozen M1/M2 directory layout."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .contracts import BibtexRecord, ReferenceRecord, normalize_arxiv_id


class InputDataError(ValueError):
    """Raised when an M1/M2 input record cannot satisfy the M3 contract."""


def _range_directory(data_root: Path, source: str, range_id: str) -> Path:
    path = data_root / source / range_id
    if not path.is_dir():
        raise FileNotFoundError(f"missing {source} range directory: {path}")
    return path


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON-object rows without materializing a large JSONL file."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InputDataError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise InputDataError(f"expected a JSON object at {path}:{line_number}")
            yield row


def iter_bibtex_records(data_root: Path, range_id: str) -> Iterator[BibtexRecord]:
    """Yield normalized BibTeX records for one M2 range."""

    path = _range_directory(data_root, "m2", range_id) / "bibtex.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing M2 BibTeX output: {path}")
    for line_number, row in enumerate(iter_jsonl_rows(path), start=1):
        try:
            yield BibtexRecord(
                arxiv_id=row["arxiv_id"],
                citation_key=row["citation_key"],
                entry_type=row["entry_type"],
                title=row.get("title"),
                authors=row.get("authors"),
                year=row.get("year"),
                doi=row.get("doi"),
                journal=row.get("journal"),
                url=row.get("url"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InputDataError(f"invalid BibTeX row at {path}:{line_number}") from error


def iter_reference_documents(data_root: Path, range_id: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield one citing-paper ID and its raw M1 references at a time."""

    corpus_range = _range_directory(data_root, "corpus", range_id)
    for paper_directory in sorted(path for path in corpus_range.iterdir() if path.is_dir()):
        path = paper_directory / "references.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing M1 reference output: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if not isinstance(document, Mapping):
                raise TypeError("reference document must be a JSON object")
            arxiv_id = normalize_arxiv_id(document.get("arxiv_id"))
            references = document.get("references", [])
            if arxiv_id is None:
                raise ValueError("reference document has no arxiv_id")
            if not isinstance(references, list) or not all(isinstance(row, dict) for row in references):
                raise TypeError("references must be a JSON-object list")
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise InputDataError(f"invalid M1 reference output: {path}") from error
        yield arxiv_id, references


def iter_matchable_reference_records(data_root: Path, range_id: str) -> Iterator[ReferenceRecord]:
    """Yield only references that have an arXiv target and can form candidates."""

    for document_arxiv_id, references in iter_reference_documents(data_root, range_id):
        for reference in references:
            try:
                reference_arxiv_id = normalize_arxiv_id(reference.get("arxiv_id"))
                if reference_arxiv_id is not None and reference_arxiv_id != document_arxiv_id:
                    raise ValueError("reference arxiv_id differs from its citing publication")
                if normalize_arxiv_id(reference.get("ref_arxiv_id")) is None:
                    continue
                yield ReferenceRecord(
                    arxiv_id=reference_arxiv_id or document_arxiv_id,
                    ref_arxiv_id=reference["ref_arxiv_id"],
                    ref_title=reference.get("ref_title"),
                    ref_authors=reference.get("ref_authors"),
                    ref_submission_date=reference.get("ref_submission_date"),
                    ref_s2_id=reference.get("ref_s2_id"),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise InputDataError(
                    f"invalid matchable reference in citing publication {document_arxiv_id}"
                ) from error
