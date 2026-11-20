"""Enrich M1 reference documents with arXiv submission metadata.

Semantic Scholar's ``publicationDate`` is bibliographic publication metadata;
it is not the date an arXiv preprint was first submitted.  This module performs
the deliberately separate, cacheable arXiv lookup required to populate the
official ``ref_submission_date`` field for arXiv-keyed references.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import arxiv

from src.crawling.metadata import base_arxiv_id
from src.crawling.reference_crawler import _normalize_doi, _optional_text

CandidateMetadata = dict[str, str | None]
MetadataFetcher = Callable[[list[str]], Mapping[str, CandidateMetadata]]


def _canonical_arxiv_id(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return base_arxiv_id(text)
    except ValueError:
        return None


def apply_candidate_metadata(
    references: Iterable[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, Any]], int]:
    """Return copies of references enriched from canonical arXiv metadata.

    Existing Semantic Scholar publication fields remain intact.  arXiv values
    fill submission time and only backfill DOI/venue when those fields are
    missing, so sources are never silently conflated.
    """

    enriched: list[dict[str, Any]] = []
    updated = 0
    for reference in references:
        row = dict(reference)
        ref_arxiv_id = _canonical_arxiv_id(row.get("ref_arxiv_id"))
        row["ref_arxiv_id"] = ref_arxiv_id
        candidate = candidates.get(ref_arxiv_id) if ref_arxiv_id else None
        if candidate is None:
            enriched.append(row)
            continue

        submission_date = _optional_text(candidate.get("submission_date"))
        updated_date = _optional_text(candidate.get("updated_date"))
        candidate_doi = _normalize_doi(candidate.get("doi"))
        candidate_venue = _optional_text(candidate.get("venue"))
        if submission_date is not None:
            row["ref_submission_date"] = submission_date
            row["ref_submission_date_source"] = "arxiv"
        if updated_date is not None:
            row["ref_arxiv_last_updated_date"] = updated_date
        if _normalize_doi(row.get("ref_doi")) is None and candidate_doi:
            row["ref_doi"] = candidate_doi
            row["ref_doi_source"] = "arxiv"
        if _optional_text(row.get("ref_venue")) is None and candidate_venue:
            row["ref_venue"] = candidate_venue
            row["ref_venue_source"] = "arxiv"
        updated += 1
        enriched.append(row)
    return enriched, updated


def fetch_arxiv_metadata(
    arxiv_ids: list[str],
    *,
    client: arxiv.Client | None = None,
) -> dict[str, CandidateMetadata]:
    """Fetch candidate metadata for one bounded arXiv ID batch."""

    if not arxiv_ids:
        return {}
    api_client = client or arxiv.Client()
    search = arxiv.Search(id_list=arxiv_ids, max_results=len(arxiv_ids))
    records: dict[str, CandidateMetadata] = {}
    for paper in api_client.results(search):
        arxiv_id = _canonical_arxiv_id(paper.get_short_id())
        if arxiv_id is None:
            continue
        records[arxiv_id] = {
            "submission_date": paper.published.date().isoformat(),
            "updated_date": paper.updated.date().isoformat(),
            "doi": _optional_text(getattr(paper, "doi", None)),
            "venue": _optional_text(getattr(paper, "journal_ref", None)),
        }
    return records


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_candidate_cache(path: Path) -> dict[str, CandidateMetadata]:
    payload = _read_json(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping):
        return {}
    return {
        arxiv_id: dict(metadata)
        for arxiv_id, metadata in candidates.items()
        if isinstance(arxiv_id, str) and isinstance(metadata, Mapping)
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _reference_paths(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.rglob("references.json"))


def enrich_reference_documents(
    corpus_dir: str | Path,
    *,
    fetcher: MetadataFetcher = fetch_arxiv_metadata,
    batch_size: int = 100,
    cache_path: str | Path | None = None,
) -> dict[str, int]:
    """Populate submission metadata in all schema-v2 ``references.json`` files.

    The caller supplies a batching fetcher for testability.  Production uses
    arXiv's API client; its client-side pacing remains responsible for polite
    external access.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(corpus_dir)
    resolved_cache_path = Path(cache_path) if cache_path is not None else root / ".arxiv_candidate_metadata_cache.json"
    documents: list[tuple[Path, dict[str, Any]]] = []
    needed: set[str] = set()
    for path in _reference_paths(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            continue
        references = payload.get("references")
        if not isinstance(references, list):
            continue
        documents.append((path, payload))
        needed.update(
            arxiv_id
            for row in references
            if isinstance(row, Mapping)
            for arxiv_id in [_canonical_arxiv_id(row.get("ref_arxiv_id"))]
            if arxiv_id is not None
        )

    candidates = _load_candidate_cache(resolved_cache_path)
    sorted_ids = sorted(needed)
    uncached_ids = [arxiv_id for arxiv_id in sorted_ids if arxiv_id not in candidates]
    for start in range(0, len(uncached_ids), batch_size):
        candidates.update(fetcher(uncached_ids[start:start + batch_size]))
    if uncached_ids:
        _atomic_write_json(resolved_cache_path, {"schema_version": 1, "candidates": candidates})

    updated_documents = 0
    updated_references = 0
    for path, payload in documents:
        references = payload["references"]
        enriched, updated = apply_candidate_metadata(references, candidates)
        if updated:
            payload["references"] = enriched
            payload["references_count"] = len(enriched)
            payload["arxiv_enrichment"] = {
                "candidate_count": len(candidates),
                "updated_references": updated,
                "status": "completed",
            }
            _atomic_write_json(path, payload)
            updated_documents += 1
            updated_references += updated

    return {
        "documents": len(documents),
        "candidate_ids": len(sorted_ids),
        "resolved_candidates": sum(arxiv_id in candidates for arxiv_id in sorted_ids),
        "cache_hits": len(sorted_ids) - len(uncached_ids),
        "fetched_candidates": len(uncached_ids),
        "updated_documents": updated_documents,
        "updated_references": updated_references,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate arXiv submission dates for M1 reference candidates",
    )
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--cache-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        enrich_reference_documents(
            args.corpus_dir,
            batch_size=args.batch_size,
            cache_path=args.cache_path,
        ),
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
