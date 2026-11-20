from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping

from .contracts import BibtexRecord, CandidatePair, ReferenceRecord, normalize_arxiv_id


BibtexInput = BibtexRecord | Mapping[str, object]
ReferenceInput = ReferenceRecord | Mapping[str, object]


def _field(record: BibtexInput | ReferenceInput, name: str) -> object:
    """Read a field from a frozen contract record or a mapping fixture."""

    if isinstance(record, (BibtexRecord, ReferenceRecord)):
        return getattr(record, name)
    if isinstance(record, Mapping):
        return record.get(name)
    raise TypeError(
        "records must be BibtexRecord/ReferenceRecord instances or mappings"
    )


def _required_arxiv_id(value: object, field: str) -> str:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    normalized = normalize_arxiv_id(value)
    if normalized is None:
        raise ValueError(f"{field} must be a non-missing arXiv ID")
    return normalized


def _optional_reference_id(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("ref_arxiv_id must be a string or None")
    # The shared normalizer treats whitespace and null-like strings as missing.
    # Invalid but nonempty IDs must be surfaced, not silently discarded.
    return normalize_arxiv_id(value)


def _required_citation_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("citation_key must be a non-empty string")
    return value.strip()


def generate_candidate_pairs(
    bibtex_records: Iterable[BibtexInput],
    reference_records: Iterable[ReferenceInput],
) -> Iterator[CandidatePair]:
    """Yield every valid BibTeX-reference pair within each citing publication.

    Contract and behavior:

    * Output type: ``CandidatePair(arxiv_id, citation_key, ref_arxiv_id)``.
    * Canonicalize IDs using the shared ``normalize_arxiv_id`` function.
    * Reference targets must have a valid, non-missing ``ref_arxiv_id``.
    * Deduplicate references by ``(arxiv_id, ref_arxiv_id)``.
    * Reject duplicate BibTeX keys per publication (invalid frozen input).
    * Produce pairs in stable ``(arxiv_id, citation_key, ref_arxiv_id)`` order.
    * Never form cross-publication pairs or fabricate a no-match candidate.

    Entries with zero candidates yield no ``CandidatePair`` objects. They must
    still be retained by the calling pipeline in its original BibTeX table so
    the downstream labeling/prediction modules can handle no-match cases.

    The implementation materializes *only* the keys and unique reference IDs,
    not the full ``m × n`` matrix. Exhaustive generation itself requires
    ``O(sum(m_p * n_p))`` output work across publications.
    """

    bibtex_keys_by_paper: dict[str, set[str]] = defaultdict(set)
    for bibtex in bibtex_records:
        paper_id = _required_arxiv_id(_field(bibtex, "arxiv_id"), "arxiv_id")
        citation_key = _required_citation_key(_field(bibtex, "citation_key"))
        if citation_key in bibtex_keys_by_paper[paper_id]:
            raise ValueError(
                f"duplicate BibTeX entry: (arxiv_id={paper_id!r}, "
                f"citation_key={citation_key!r})"
            )
        bibtex_keys_by_paper[paper_id].add(citation_key)

    # Keep reference IDs only for publications that actually have BibTeX
    # entries. This avoids indexing irrelevant references in a batch/range.
    references_by_paper: dict[str, set[str]] = defaultdict(set)
    for reference in reference_records:
        paper_id = _required_arxiv_id(_field(reference, "arxiv_id"), "arxiv_id")
        ref_id = _optional_reference_id(_field(reference, "ref_arxiv_id"))
        if ref_id is not None and paper_id in bibtex_keys_by_paper:
            references_by_paper[paper_id].add(ref_id)

    # Sort each publication's references once; do not re-sort for every entry.
    for paper_id in sorted(bibtex_keys_by_paper):
        ref_ids = sorted(references_by_paper.get(paper_id, ()))
        for citation_key in sorted(bibtex_keys_by_paper[paper_id]):
            for ref_id in ref_ids:
                yield CandidatePair(
                    arxiv_id=paper_id,
                    citation_key=citation_key,
                    ref_arxiv_id=ref_id,
                )
