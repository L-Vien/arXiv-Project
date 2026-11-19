"""Per-publication data-quality inventory for manual-label selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import normalize_arxiv_id
from .loading import iter_bibtex_records, iter_reference_documents


@dataclass(frozen=True)
class PublicationInventoryRow:
    member_id: str
    range_id: str
    arxiv_id: str
    bibtex_count: int
    reference_count: int
    arxiv_reference_count: int
    missing_title_rate: float
    missing_author_rate: float
    missing_year_rate: float
    duplicate_reference_count: int


def build_publication_inventory(
    data_root: Path,
    range_id: str,
    member_id: str,
    paper_ids: set[str],
) -> list[PublicationInventoryRow]:
    """Aggregate label-selection signals without loading a whole range at once."""

    bibtex_stats = {
        arxiv_id: {"count": 0, "missing_title": 0, "missing_author": 0, "missing_year": 0}
        for arxiv_id in paper_ids
    }
    for record in iter_bibtex_records(data_root, range_id):
        stats = bibtex_stats.get(record.arxiv_id)
        if stats is None:
            continue
        stats["count"] += 1
        stats["missing_title"] += record.title is None
        stats["missing_author"] += not record.authors
        stats["missing_year"] += record.year is None

    reference_stats = {
        arxiv_id: {"count": 0, "arxiv_count": 0, "duplicate_count": 0, "seen": set()}
        for arxiv_id in paper_ids
    }
    for arxiv_id, references in iter_reference_documents(data_root, range_id):
        stats = reference_stats.get(arxiv_id)
        if stats is None:
            continue
        for reference in references:
            stats["count"] += 1
            ref_arxiv_id = normalize_arxiv_id(reference.get("ref_arxiv_id"))
            if ref_arxiv_id is None:
                continue
            stats["arxiv_count"] += 1
            if ref_arxiv_id in stats["seen"]:
                stats["duplicate_count"] += 1
            else:
                stats["seen"].add(ref_arxiv_id)

    rows = []
    for arxiv_id in sorted(paper_ids):
        bibtex = bibtex_stats[arxiv_id]
        references = reference_stats[arxiv_id]
        count = bibtex["count"]
        rows.append(
            PublicationInventoryRow(
                member_id=member_id,
                range_id=range_id,
                arxiv_id=arxiv_id,
                bibtex_count=count,
                reference_count=references["count"],
                arxiv_reference_count=references["arxiv_count"],
                missing_title_rate=bibtex["missing_title"] / count if count else 0.0,
                missing_author_rate=bibtex["missing_author"] / count if count else 0.0,
                missing_year_rate=bibtex["missing_year"] / count if count else 0.0,
                duplicate_reference_count=references["duplicate_count"],
            )
        )
    return rows


def write_inventory_csv(rows: list[PublicationInventoryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PublicationInventoryRow.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _paper_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--range-id", required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--paper-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    rows = build_publication_inventory(
        arguments.data_root,
        arguments.range_id,
        arguments.member_id,
        _paper_ids(arguments.paper_ids),
    )
    write_inventory_csv(rows, arguments.output)
    print(f"wrote {len(rows)} publication rows to {arguments.output}")


if __name__ == "__main__":
    main()
