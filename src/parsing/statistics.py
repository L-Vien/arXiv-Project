"""Run statistics for the Milestone 2 parsing pipeline.

The accumulator keeps both run-level aggregates and a per-publication record.
This is intentional: the Milestone 2 report needs aggregate success rates, but
version coverage and empty-version handling are only meaningful when they can
also be inspected publication by publication.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from src.parsing.contracts import (
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    DeduplicatedPublication,
    Diagnostic,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
)
from src.parsing.validation import (
    ValidationReport,
    effective_status,
    validation_status_for_version,
)

_PARSED_STATUSES = frozenset(
    {
        ProcessingStatus.OK,
        ProcessingStatus.PARTIAL_PARSE,
        ProcessingStatus.MISSING_INCLUDE,
        ProcessingStatus.BIBTEX_PARTIAL,
    }
)
_FAILED_STATUSES = frozenset(
    {
        ProcessingStatus.ROOT_NOT_FOUND,
        ProcessingStatus.PARSE_ERROR,
        ProcessingStatus.ERROR,
    }
)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _status_counts(statuses: Sequence[ProcessingStatus]) -> dict[str, int]:
    counts = Counter(status.value for status in statuses)
    return {
        status.value: counts.get(status.value, 0)
        for status in ProcessingStatus
    }


class StatisticsAccumulator:
    """Collect deterministic metrics for one pipeline run.

    ``raw_reference_entry_count`` remains an override for compatibility with
    focused callers.  Normal pipeline runs use ``ReferenceBundle.raw_entry_count``
    when TV5 provides it, and otherwise expose the weaker citation-key basis.
    """

    def __init__(self) -> None:
        self._publication_records: list[dict[str, Any]] = []
        self._publication_status_counts: Counter[str] = Counter()
        self._version_counts: Counter[str] = Counter()
        self._expected_version_count = 0
        self._raw_element_count = 0
        self._deduplicated_element_count = 0
        self._unified_reference_count = 0
        self._source_citation_key_count = 0
        self._raw_reference_counts: list[int | None] = []
        self._citation_map_version_count = 0
        self._validation_error_count = 0
        self._validation_warning_count = 0
        self._diagnostic_error_count = 0
        self._diagnostic_warning_count = 0
        self._elapsed_seconds = 0.0

    def record_publication(
        self,
        *,
        sources: Sequence[SourceBundle],
        references: ReferenceBundle,
        normalized_versions: Sequence[NormalizedVersion],
        publication: DeduplicatedPublication,
        validation: ValidationReport,
        elapsed_seconds: float,
        raw_trees: Sequence[RawTree] = (),
        expected_version_count: int | None = None,
        raw_reference_entry_count: int | None = None,
        diagnostics: Sequence[Diagnostic] = (),
    ) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if expected_version_count is None:
            expected_version_count = len(sources)
        if expected_version_count < 0:
            raise ValueError("expected_version_count must be non-negative")
        if expected_version_count < len(sources):
            raise ValueError(
                "expected_version_count cannot be smaller than processed versions"
            )
        if raw_reference_entry_count is not None and raw_reference_entry_count < 0:
            raise ValueError("raw_reference_entry_count must be non-negative")

        raw_by_version = {tree.version: tree for tree in raw_trees}
        normalized_by_version = {
            version.version: version for version in normalized_versions
        }
        bibtex_partial_versions = {
            diagnostic.version
            for diagnostic in references.diagnostics
            if diagnostic.code == "BIBTEX_PARTIAL"
            and diagnostic.version is not None
        }
        statuses = tuple(
            effective_status(
                source.status,
                *(
                    stage.status
                    for stage in (
                        raw_by_version.get(source.version),
                        normalized_by_version.get(source.version),
                    )
                    if stage is not None
                ),
                *(
                    (ProcessingStatus.BIBTEX_PARTIAL,)
                    if source.version in bibtex_partial_versions
                    else ()
                ),
                *(
                    (validation_status,)
                    if (
                        validation_status := validation_status_for_version(
                            validation,
                            source.version,
                        )
                    ) is not None
                    else ()
                ),
            )
            for source in sources
        )
        status_counts = _status_counts(statuses)
        self._expected_version_count += expected_version_count
        self._version_counts.update(status.value for status in statuses)
        # Deduplication consumes NormalizedVersion.nodes.  Counting the raw
        # TV3 tree here would compare different stages and can make
        # after_dedup larger than before_dedup after sentence splitting.
        normalized_element_count = sum(
            len(version.nodes) for version in normalized_versions
        )
        self._raw_element_count += normalized_element_count
        self._deduplicated_element_count += len(publication.elements)
        unified_reference_count = len(references.entries)
        source_citation_key_count = sum(
            len(mapping) for mapping in references.citation_maps.values()
        )
        exact_raw_reference_count = (
            raw_reference_entry_count
            if raw_reference_entry_count is not None
            else references.raw_entry_count
        )
        reference_count_before_dedup = (
            exact_raw_reference_count
            if exact_raw_reference_count is not None
            else source_citation_key_count
        )
        reference_count_basis = (
            "raw_entries"
            if exact_raw_reference_count is not None
            else "source_citation_keys"
        )
        self._unified_reference_count += unified_reference_count
        self._source_citation_key_count += source_citation_key_count
        self._raw_reference_counts.append(exact_raw_reference_count)
        self._citation_map_version_count += len(references.citation_maps)
        self._validation_error_count += len(validation.errors)
        self._validation_warning_count += len(validation.warnings)
        diagnostics = tuple(diagnostics)
        self._diagnostic_error_count += sum(
            diagnostic.severity.value == "error" for diagnostic in diagnostics
        )
        self._diagnostic_warning_count += sum(
            diagnostic.severity.value == "warning" for diagnostic in diagnostics
        )
        self._elapsed_seconds += elapsed_seconds

        non_empty_count = len(sources) - status_counts[
            ProcessingStatus.EMPTY_NO_TEX.value
        ]
        parsed_count = sum(
            status_counts[status.value] for status in _PARSED_STATUSES
        )
        failed_count = sum(
            status_counts[status.value] for status in _FAILED_STATUSES
        )
        reference_warning = any(
            diagnostic.severity.value == "warning"
            for diagnostic in references.diagnostics
        )
        if not sources or failed_count or not validation.passed:
            publication_status = "failed"
        elif len(sources) < expected_version_count:
            publication_status = "partial"
        elif non_empty_count == 0:
            publication_status = "empty"
        elif any(
            status_counts[status.value] > 0
            for status in (
                ProcessingStatus.PARTIAL_PARSE,
                ProcessingStatus.MISSING_INCLUDE,
                ProcessingStatus.BIBTEX_PARTIAL,
            )
        ) or reference_warning:
            publication_status = "partial"
        else:
            publication_status = "ok"
        self._publication_status_counts[publication_status] += 1

        self._publication_records.append(
            {
                "arxiv_id": publication.arxiv_id,
                "status": publication_status,
                "versions": {
                    "expected": expected_version_count,
                    "processed": len(sources),
                    "coverage_rate": _ratio(
                        len(sources), expected_version_count
                    ),
                    "parsed": parsed_count,
                    "non_empty_tex": non_empty_count,
                    "parse_success_rate": _ratio(
                        status_counts[ProcessingStatus.OK.value],
                        non_empty_count,
                    ),
                    "empty_no_tex": status_counts[
                        ProcessingStatus.EMPTY_NO_TEX.value
                    ],
                    "failed": failed_count,
                    "status_counts": status_counts,
                },
                "elements": {
                    "before_dedup": normalized_element_count,
                    "after_dedup": len(publication.elements),
                    "deduplication_ratio": _ratio(
                        len(publication.elements),
                        sum(
                            len(version.nodes) for version in normalized_versions
                        ),
                    ),
                },
                "references": {
                    "entries_before_dedup": reference_count_before_dedup,
                    "before_dedup_basis": reference_count_basis,
                    "entries_after_dedup": unified_reference_count,
                    "source_citation_key_count": source_citation_key_count,
                    "deduplication_ratio": _ratio(
                        unified_reference_count,
                        reference_count_before_dedup,
                    ),
                },
                "validation": {
                    "errors": len(validation.errors),
                    "warnings": len(validation.warnings),
                },
                "timing": {
                    "seconds": elapsed_seconds,
                },
            }
        )

    def merge_single_publication(self, snapshot: Mapping[str, Any]) -> None:
        """Merge a one-publication snapshot produced by a worker process."""

        records = snapshot["per_publication"]
        if len(records) != 1:
            raise ValueError("snapshot must contain exactly one publication")
        record = records[0]
        versions = record["versions"]
        elements = record["elements"]
        references = record["references"]
        validation = record["validation"]
        diagnostics = snapshot["diagnostics"]

        self._publication_records.append(record)
        self._publication_status_counts[record["status"]] += 1
        self._expected_version_count += versions["expected"]
        self._version_counts.update(versions["status_counts"])
        self._raw_element_count += elements["before_dedup"]
        self._deduplicated_element_count += elements["after_dedup"]
        self._unified_reference_count += references["entries_after_dedup"]
        self._source_citation_key_count += references["source_citation_key_count"]
        self._raw_reference_counts.append(
            references["entries_before_dedup"]
            if references["before_dedup_basis"] == "raw_entries"
            else None
        )
        self._citation_map_version_count += snapshot["references"][
            "citation_map_versions"
        ]
        self._validation_error_count += validation["errors"]
        self._validation_warning_count += validation["warnings"]
        self._diagnostic_error_count += diagnostics["errors"]
        self._diagnostic_warning_count += diagnostics["warnings"]
        self._elapsed_seconds += record["timing"]["seconds"]

    def to_dict(self) -> dict[str, Any]:
        raw_count = self._raw_element_count
        dedup_count = self._deduplicated_element_count
        publication_count = len(self._publication_records)
        ratio = _ratio(dedup_count, raw_count)
        average_seconds = (
            self._elapsed_seconds / publication_count
            if publication_count
            else None
        )
        status_counts = {
            status.value: self._version_counts.get(status.value, 0)
            for status in ProcessingStatus
        }
        total_versions = sum(self._version_counts.values())
        raw_reference_count_known = bool(self._raw_reference_counts) and all(
            count is not None for count in self._raw_reference_counts
        )
        raw_reference_count = (
            sum(count or 0 for count in self._raw_reference_counts)
            if raw_reference_count_known
            else None
        )
        reference_count_before_dedup = (
            raw_reference_count
            if raw_reference_count is not None
            else self._source_citation_key_count
        )
        reference_count_basis = (
            "raw_entries"
            if raw_reference_count is not None
            else "source_citation_keys"
        )
        successful_publications = (
            self._publication_status_counts["ok"]
            + self._publication_status_counts["empty"]
        )
        parsed_versions = sum(
            status_counts[status.value] for status in _PARSED_STATUSES
        )
        non_empty_versions = total_versions - status_counts[
            ProcessingStatus.EMPTY_NO_TEX.value
        ]
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "publications": {
                "total": publication_count,
                "successful": successful_publications,
                "passed": successful_publications,
                "partial": self._publication_status_counts["partial"],
                "failed": self._publication_status_counts["failed"],
                "empty": self._publication_status_counts["empty"],
                "success_rate": _ratio(
                    successful_publications,
                    publication_count,
                ),
            },
            "versions": {
                "total": total_versions,
                "expected": self._expected_version_count,
                "processed": total_versions,
                "coverage_rate": _ratio(
                    total_versions,
                    self._expected_version_count,
                ),
                "parsed": parsed_versions,
                "non_empty_tex": non_empty_versions,
                "parse_success_rate": _ratio(
                    status_counts[ProcessingStatus.OK.value],
                    non_empty_versions,
                ),
                "status_counts": status_counts,
                "without_tex": status_counts[ProcessingStatus.EMPTY_NO_TEX.value],
            },
            "elements": {
                "raw_count": raw_count,
                "deduplicated_count": dedup_count,
                "before_dedup": raw_count,
                "after_dedup": dedup_count,
                "deduplicated_over_raw_ratio": ratio,
                "deduplication_ratio": ratio,
            },
            "references": {
                "unified_entries": self._unified_reference_count,
                "entries_before_dedup": reference_count_before_dedup,
                "before_dedup_basis": reference_count_basis,
                "entries_before_dedup_available": raw_reference_count_known,
                "entries_after_dedup": self._unified_reference_count,
                "source_citation_key_count": self._source_citation_key_count,
                "deduplication_ratio": _ratio(
                    self._unified_reference_count,
                    reference_count_before_dedup,
                ),
                "citation_map_versions": self._citation_map_version_count,
            },
            "validation": {
                "errors": self._validation_error_count,
                "warnings": self._validation_warning_count,
            },
            "diagnostics": {
                "errors": self._diagnostic_error_count,
                "warnings": self._diagnostic_warning_count,
            },
            "timing": {
                "publication_seconds_total": self._elapsed_seconds,
                "publication_seconds_average": average_seconds,
            },
            "per_publication": sorted(
                self._publication_records,
                key=lambda record: record["arxiv_id"],
            ),
        }


__all__ = ["StatisticsAccumulator"]
