"""Milestone 3 reference-matching pipeline contracts and utilities."""

from .contracts import (
    CONTRACT_VERSION,
    CandidatePair,
    BibtexRecord,
    FeatureRow,
    LabelRecord,
    LabelSource,
    LabelStatus,
    MemberRange,
    Partition,
    PartitionRecord,
    PredictionRow,
    ReferenceRecord,
    normalize_arxiv_id,
    normalize_doi,
)

__all__ = [
    "CONTRACT_VERSION",
    "BibtexRecord",
    "ReferenceRecord",
    "LabelRecord",
    "CandidatePair",
    "FeatureRow",
    "PredictionRow",
    "PartitionRecord",
    "MemberRange",
    "LabelSource",
    "LabelStatus",
    "Partition",
    "normalize_arxiv_id",
    "normalize_doi",
]
