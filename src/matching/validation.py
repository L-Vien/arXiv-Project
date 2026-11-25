"""Validation for M3 matching inputs, candidates, and publication splits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import (
    BibtexRecord,
    CandidatePair,
    Partition,
    ReferenceRecord,
    normalize_arxiv_id,
)
from .loading import (
    iter_bibtex_records,
    iter_bibtex_records_from_file,
    iter_matchable_reference_records,
    iter_matchable_reference_records_from_directory,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    arxiv_id: str | None = None


@dataclass
class ValidationReport:
    scope: str
    checked_records: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.issues)

    @property
    def error_codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def add(self, code: str, message: str, arxiv_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(code, message, arxiv_id))

    def extend(self, other: ValidationReport) -> None:
        self.checked_records += other.checked_records
        self.issues.extend(other.issues)

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "checked_records": self.checked_records,
            "errors": [issue.__dict__ for issue in self.issues],
            "error_count": self.error_count,
            "status": "passed" if self.error_count == 0 else "failed",
        }


def validate_bibtex_records(
    records: Iterable[BibtexRecord], allowed_paper_ids: set[str],
) -> ValidationReport:
    report = ValidationReport("bibtex")
    keys: set[tuple[str, str]] = set()
    for record in records:
        report.checked_records += 1
        key = (record.arxiv_id, record.citation_key)
        if key in keys:
            report.add("DUPLICATE_BIBTEX_ENTRY", f"duplicate BibTeX key: {key}", record.arxiv_id)
        else:
            keys.add(key)
        if record.arxiv_id not in allowed_paper_ids:
            report.add("PUBLICATION_OUT_OF_RANGE", "BibTeX publication is not assigned to this range", record.arxiv_id)
    return report


def validate_reference_records(
    records: Iterable[ReferenceRecord], allowed_paper_ids: set[str],
) -> ValidationReport:
    report = ValidationReport("references")
    keys: set[tuple[str, str]] = set()
    for record in records:
        report.checked_records += 1
        key = (record.arxiv_id, record.ref_arxiv_id)
        if key in keys:
            report.add("DUPLICATE_REFERENCE", f"duplicate reference key: {key}", record.arxiv_id)
        else:
            keys.add(key)
        if record.arxiv_id not in allowed_paper_ids:
            report.add("PUBLICATION_OUT_OF_RANGE", "reference publication is not assigned to this range", record.arxiv_id)
    return report


def validate_candidate_pairs(
    candidates: Iterable[CandidatePair],
    bibtex_keys: set[tuple[str, str]],
    reference_keys: set[tuple[str, str]],
) -> ValidationReport:
    report = ValidationReport("candidates")
    keys: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        report.checked_records += 1
        key = (candidate.arxiv_id, candidate.citation_key, candidate.ref_arxiv_id)
        if key in keys:
            report.add("DUPLICATE_CANDIDATE", f"duplicate candidate key: {key}", candidate.arxiv_id)
        else:
            keys.add(key)
        if (candidate.arxiv_id, candidate.citation_key) not in bibtex_keys:
            report.add("CANDIDATE_BIBTEX_NOT_FOUND", "candidate has no matching BibTeX entry", candidate.arxiv_id)
        if (candidate.arxiv_id, candidate.ref_arxiv_id) not in reference_keys:
            report.add(
                "CANDIDATE_REFERENCE_WRONG_PUBLICATION",
                "candidate reference does not belong to the same citing publication",
                candidate.arxiv_id,
            )
    return report


def validate_partition_records(records: Iterable[Mapping[str, object]]) -> ValidationReport:
    report = ValidationReport("partitions")
    assignments: dict[str, Partition] = {}
    for row in records:
        report.checked_records += 1
        try:
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            partition = Partition(row.get("partition"))
            if arxiv_id is None:
                raise ValueError("missing arxiv_id")
        except (TypeError, ValueError):
            report.add("INVALID_PARTITION_RECORD", "partition row has invalid arxiv_id or partition")
            continue
        previous = assignments.setdefault(arxiv_id, partition)
        if previous is not partition:
            report.add("SPLIT_OVERLAP", "publication appears in more than one partition", arxiv_id)
    return report


def validate_range_inputs(data_root: Path, range_id: str, allowed_paper_ids: set[str]) -> ValidationReport:
    """Validate all BibTeX and matchable-reference records for one range."""

    report = ValidationReport(f"range-inputs:{range_id}")
    report.extend(validate_bibtex_records(iter_bibtex_records(data_root, range_id), allowed_paper_ids))
    report.extend(
        validate_reference_records(iter_matchable_reference_records(data_root, range_id), allowed_paper_ids)
    )
    return report


def validate_input_paths(
    bibtex_path: Path,
    references_path: Path,
    allowed_paper_ids: set[str],
) -> ValidationReport:
    """Validate M2 BibTeX and M1 references supplied as explicit paths."""

    report = ValidationReport("input-paths")
    report.extend(
        validate_bibtex_records(
            iter_bibtex_records_from_file(bibtex_path),
            allowed_paper_ids,
        )
    )
    report.extend(
        validate_reference_records(
            iter_matchable_reference_records_from_directory(references_path),
            allowed_paper_ids,
        )
    )
    return report


def write_validation_report(report: ValidationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")


def _paper_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate M1/M2 inputs against a frozen paper-ID list."
    )
    parser.add_argument("--bibtex", type=Path, help="Path to the M2 bibtex.jsonl file.")
    parser.add_argument(
        "--references",
        type=Path,
        help="Path to the M1 range directory containing paper folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Legacy data root containing m2/<range> and corpus/<range>.",
    )
    parser.add_argument("--range-id", help="Range name used with --data-root.")
    parser.add_argument("--paper-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    uses_explicit_paths = arguments.bibtex is not None or arguments.references is not None
    uses_legacy_layout = arguments.data_root is not None or arguments.range_id is not None

    if uses_explicit_paths and uses_legacy_layout:
        parser.error(
            "use either --bibtex/--references or --data-root/--range-id, not both"
        )
    if uses_explicit_paths:
        if arguments.bibtex is None or arguments.references is None:
            parser.error("--bibtex and --references must be provided together")
        report = validate_input_paths(
            arguments.bibtex,
            arguments.references,
            _paper_ids(arguments.paper_ids),
        )
    elif arguments.data_root is not None and arguments.range_id is not None:
        report = validate_range_inputs(
            arguments.data_root,
            arguments.range_id,
            _paper_ids(arguments.paper_ids),
        )
    else:
        parser.error(
            "provide --bibtex/--references or --data-root/--range-id"
        )

    write_validation_report(report, arguments.output)
    print(f"{report.scope}: {report.error_count} error(s); report: {arguments.output}")


if __name__ == "__main__":
    main()
