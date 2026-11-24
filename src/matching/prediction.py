"""Formatting ranked candidates into the stable M3 prediction contract."""

from __future__ import annotations

from typing import Iterable, Mapping

from .contracts import (
    BibtexRecord,
    LabelRecord,
    LabelStatus,
    Partition,
    PredictionRow,
    normalize_arxiv_id,
)
from .ranking import QueryKey, ScoredCandidate, rank_candidate_groups


def format_predictions(
    bibtex_records: Iterable[BibtexRecord],
    candidates: Iterable[ScoredCandidate],
    partitions: Mapping[str, Partition | str],
    *,
    labels: Iterable[LabelRecord] = (),
) -> list[PredictionRow]:
    """Build one contract-compliant prediction row per BibTeX record."""

    records_by_key: dict[QueryKey, BibtexRecord] = {}
    for record in bibtex_records:
        key = (record.arxiv_id, record.citation_key)
        if key in records_by_key:
            raise ValueError(f"duplicate BibTeX record: {key}")
        records_by_key[key] = record

    normalized_partitions: dict[str, Partition] = {}
    for raw_id, raw_partition in partitions.items():
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id is None:
            raise ValueError("partition map contains a missing arxiv_id")
        if arxiv_id in normalized_partitions:
            raise ValueError(f"duplicate partition assignment for publication: {arxiv_id}")
        normalized_partitions[arxiv_id] = Partition(raw_partition)

    groundtruth_by_key: dict[QueryKey, str | None] = {}
    for label in labels:
        key = (label.arxiv_id, label.citation_key)
        if key not in records_by_key:
            raise ValueError(f"label has no matching BibTeX record: {key}")
        if key in groundtruth_by_key:
            raise ValueError(f"duplicate label for citation: {key}")
        if label.status is LabelStatus.UNCERTAIN:
            raise ValueError(f"uncertain labels cannot be exported as predictions: {key}")
        groundtruth_by_key[key] = label.groundtruth_arxiv_id

    candidate_list = list(candidates)
    for candidate in candidate_list:
        if candidate.query_key not in records_by_key:
            raise ValueError(f"candidate has no matching BibTeX record: {candidate.query_key}")
    grouped = rank_candidate_groups(candidate_list, top_k=5)

    result: list[PredictionRow] = []
    for key in sorted(records_by_key):
        record = records_by_key[key]
        if record.arxiv_id not in normalized_partitions:
            raise ValueError(f"missing partition assignment for publication: {record.arxiv_id}")
        ranked = grouped.get(key, [])
        candidate_ids = [candidate.ref_arxiv_id for candidate in ranked]
        candidate_ids.extend([None] * (5 - len(candidate_ids)))
        result.append(
            PredictionRow(
                arxiv_id=record.arxiv_id,
                partition=normalized_partitions[record.arxiv_id],
                citation_key=record.citation_key,
                groundtruth_arxiv_id=groundtruth_by_key.get(key),
                cand_1=candidate_ids[0],
                cand_2=candidate_ids[1],
                cand_3=candidate_ids[2],
                cand_4=candidate_ids[3],
                cand_5=candidate_ids[4],
            )
        )
    return result
