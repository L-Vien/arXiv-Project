"""Stable records and normalization rules shared by Milestone 3 modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Iterable, Mapping


CONTRACT_VERSION = "1.0"
SCHEMA_VERSION = "1.0"

_MODERN_ID = re.compile(r"^(?P<month>\d{4})\.(?P<number>\d{4,5})$")
_LEGACY_ID = re.compile(r"^[A-Za-z][A-Za-z0-9.+-]*/\d{7}$")
_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_URL_PREFIX = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^arxiv\s*:\s*", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)
_DOI_VALUE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


class LabelSource(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class LabelStatus(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNCERTAIN = "uncertain"


class Partition(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("text fields must be strings or None")
    normalized = value.strip()
    return normalized or None


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return the canonical base arXiv ID, or ``None`` for a missing value.

    Matching-table keys never use filesystem slugs (``2609-07834``), version
    suffixes, prefixes, or blank/sentinel strings.
    """

    normalized = _optional_text(value)
    if normalized is None:
        return None
    if normalized.casefold() in {"none", "null"}:
        return None

    normalized = _ARXIV_URL_PREFIX.sub("", normalized)
    normalized = _ARXIV_PREFIX.sub("", normalized)
    normalized = normalized.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    normalized = _VERSION_SUFFIX.sub("", normalized).strip()

    modern = _MODERN_ID.fullmatch(normalized)
    if modern:
        return f"{modern.group('month')}.{int(modern.group('number')):05d}"
    if _LEGACY_ID.fullmatch(normalized):
        return normalized
    raise ValueError(f"invalid arXiv ID: {value!r}")


def normalize_doi(value: str | None) -> str | None:
    """Return a validated lowercase DOI without a resolver URL or prefix."""

    normalized = _optional_text(value)
    if normalized is None:
        return None
    normalized = _DOI_PREFIX.sub("", normalized).strip().lower()
    match = _DOI_VALUE.search(normalized)
    return match.group(0).rstrip(".,;:)]}").lower() if match else None


def _authors(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError("authors must be an iterable of strings, not one string")
    result = []
    for author in value:
        normalized = _optional_text(author)
        if normalized is not None:
            result.append(normalized)
    return tuple(result)


def _required(value: str, field_name: str) -> str:
    normalized = _optional_text(value)
    if normalized is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _required_arxiv_id(value: str, field_name: str) -> str:
    normalized = normalize_arxiv_id(value)
    if normalized is None:
        raise ValueError(f"{field_name} must be a non-missing arXiv ID")
    return normalized


@dataclass(frozen=True)
class BibtexRecord:
    arxiv_id: str
    citation_key: str
    entry_type: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    url: str | None = None
    title_status: str | None = None
    title_metadata_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "citation_key", _required(self.citation_key, "citation_key"))
        object.__setattr__(self, "entry_type", _required(self.entry_type, "entry_type"))
        object.__setattr__(self, "title", _optional_text(self.title))
        object.__setattr__(self, "authors", _authors(self.authors))
        if self.year is not None and (isinstance(self.year, bool) or not isinstance(self.year, int)):
            raise TypeError("year must be an integer or None")
        object.__setattr__(self, "doi", normalize_doi(self.doi))
        object.__setattr__(self, "journal", _optional_text(self.journal))
        object.__setattr__(self, "url", _optional_text(self.url))
        title_status = self.title_status or (
            "present" if self.title is not None else "missing_source"
        )
        title_metadata_status = self.title_metadata_status or (
            "metadata_available"
            if (
                self.authors
                or self.year is not None
                or self.doi
                or self.journal
                or self.url
            )
            else "metadata_empty"
        )
        if title_status not in {"present", "missing_source", "missing_parser"}:
            raise ValueError(f"unknown title_status: {title_status!r}")
        if title_metadata_status not in {"metadata_available", "metadata_empty"}:
            raise ValueError(
                f"unknown title_metadata_status: {title_metadata_status!r}"
            )
        object.__setattr__(self, "title_status", title_status)
        object.__setattr__(self, "title_metadata_status", title_metadata_status)


@dataclass(frozen=True)
class ReferenceRecord:
    arxiv_id: str
    ref_arxiv_id: str
    ref_title: str | None = None
    ref_authors: tuple[str, ...] = ()
    ref_submission_date: str | None = None
    ref_s2_id: str | None = None
    ref_doi: str | None = None
    ref_publication_date: str | None = None
    ref_publication_year: int | None = None
    ref_venue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "ref_arxiv_id", _required_arxiv_id(self.ref_arxiv_id, "ref_arxiv_id"))
        object.__setattr__(self, "ref_title", _optional_text(self.ref_title))
        object.__setattr__(self, "ref_authors", _authors(self.ref_authors))
        object.__setattr__(self, "ref_submission_date", _optional_text(self.ref_submission_date))
        object.__setattr__(self, "ref_s2_id", _optional_text(self.ref_s2_id))
        object.__setattr__(self, "ref_doi", normalize_doi(self.ref_doi))
        object.__setattr__(self, "ref_publication_date", _optional_text(self.ref_publication_date))
        if self.ref_publication_year is not None:
            if isinstance(self.ref_publication_year, bool) or not isinstance(self.ref_publication_year, int):
                raise TypeError("ref_publication_year must be an integer or None")
            if not 1600 <= self.ref_publication_year <= 2100:
                raise ValueError("ref_publication_year is outside the accepted range")
        object.__setattr__(self, "ref_venue", _optional_text(self.ref_venue))


@dataclass(frozen=True)
class LabelRecord:
    arxiv_id: str
    citation_key: str
    groundtruth_arxiv_id: str | None
    label_source: LabelSource
    status: LabelStatus
    annotator: str | None = None
    reviewer: str | None = None
    confidence: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "citation_key", _required(self.citation_key, "citation_key"))
        object.__setattr__(self, "groundtruth_arxiv_id", normalize_arxiv_id(self.groundtruth_arxiv_id))
        object.__setattr__(self, "label_source", LabelSource(self.label_source))
        object.__setattr__(self, "status", LabelStatus(self.status))
        object.__setattr__(self, "annotator", _optional_text(self.annotator))
        object.__setattr__(self, "reviewer", _optional_text(self.reviewer))
        object.__setattr__(self, "confidence", _optional_text(self.confidence))
        object.__setattr__(self, "notes", _optional_text(self.notes))
        if self.status is LabelStatus.MATCH and self.groundtruth_arxiv_id is None:
            raise ValueError("match labels require groundtruth_arxiv_id")
        if self.status is LabelStatus.NO_MATCH and self.groundtruth_arxiv_id is not None:
            raise ValueError("no_match labels require groundtruth_arxiv_id to be None")
        if self.label_source is LabelSource.AUTOMATIC and self.status is LabelStatus.UNCERTAIN:
            raise ValueError("automatic labels cannot use uncertain status")


@dataclass(frozen=True)
class CandidatePair:
    arxiv_id: str
    citation_key: str
    ref_arxiv_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "citation_key", _required(self.citation_key, "citation_key"))
        object.__setattr__(self, "ref_arxiv_id", _required_arxiv_id(self.ref_arxiv_id, "ref_arxiv_id"))


@dataclass(frozen=True)
class FeatureRow:
    """One candidate pair plus its versioned, label-free feature values."""

    arxiv_id: str
    citation_key: str
    ref_arxiv_id: str
    partition: Partition
    feature_schema_version: str
    values: Mapping[str, float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "citation_key", _required(self.citation_key, "citation_key"))
        object.__setattr__(self, "ref_arxiv_id", _required_arxiv_id(self.ref_arxiv_id, "ref_arxiv_id"))
        object.__setattr__(self, "partition", Partition(self.partition))
        object.__setattr__(self, "feature_schema_version", _required(self.feature_schema_version, "feature_schema_version"))
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        normalized_values: dict[str, float | int | bool] = {}
        for name, value in self.values.items():
            normalized_name = _required(name, "feature name")
            if isinstance(value, bool):
                normalized_values[normalized_name] = value
            elif isinstance(value, (int, float)):
                normalized_values[normalized_name] = value
            else:
                raise TypeError("feature values must be bool, int, or float")
        object.__setattr__(self, "values", MappingProxyType(normalized_values))


@dataclass(frozen=True)
class PredictionRow:
    arxiv_id: str
    partition: Partition
    citation_key: str
    groundtruth_arxiv_id: str | None = None
    cand_1: str | None = None
    cand_2: str | None = None
    cand_3: str | None = None
    cand_4: str | None = None
    cand_5: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "partition", Partition(self.partition))
        object.__setattr__(self, "citation_key", _required(self.citation_key, "citation_key"))
        object.__setattr__(self, "groundtruth_arxiv_id", normalize_arxiv_id(self.groundtruth_arxiv_id))
        candidates = tuple(normalize_arxiv_id(getattr(self, f"cand_{index}")) for index in range(1, 6))
        populated = tuple(candidate for candidate in candidates if candidate is not None)
        if len(set(populated)) != len(populated):
            raise ValueError("prediction candidates must not contain duplicates")
        if any(candidate is None and later is not None for candidate, later in zip(candidates, candidates[1:])):
            raise ValueError("prediction candidates must occupy a contiguous top-k prefix")
        for index, candidate in enumerate(candidates, start=1):
            object.__setattr__(self, f"cand_{index}", candidate)


@dataclass(frozen=True)
class PartitionRecord:
    arxiv_id: str
    member_id: str
    range_id: str
    partition: Partition
    label_source: LabelSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "arxiv_id", _required_arxiv_id(self.arxiv_id, "arxiv_id"))
        object.__setattr__(self, "member_id", _required(self.member_id, "member_id"))
        object.__setattr__(self, "range_id", _required(self.range_id, "range_id"))
        object.__setattr__(self, "partition", Partition(self.partition))
        object.__setattr__(self, "label_source", LabelSource(self.label_source))


@dataclass(frozen=True)
class MemberRange:
    member_id: str
    member_name: str
    range_id: str
    start_month: str
    start_id: int
    end_month: str
    end_id: int
    start_arxiv_id: str
    end_arxiv_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_id", _required(self.member_id, "member_id"))
        object.__setattr__(self, "member_name", _required(self.member_name, "member_name"))
        object.__setattr__(self, "range_id", _required(self.range_id, "range_id"))
        object.__setattr__(self, "start_month", _required(self.start_month, "start_month"))
        object.__setattr__(self, "end_month", _required(self.end_month, "end_month"))
        if self.start_id < 0 or self.end_id < 0:
            raise ValueError("range IDs must be non-negative")
        object.__setattr__(self, "start_arxiv_id", _required_arxiv_id(self.start_arxiv_id, "start_arxiv_id"))
        object.__setattr__(self, "end_arxiv_id", _required_arxiv_id(self.end_arxiv_id, "end_arxiv_id"))
