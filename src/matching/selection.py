
"""Data inventory and manual publication assignment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
MIN_BIBTEX = 20
N_MAIN = 5
N_SPARE = 1


def _normalize_arxiv_id(arxiv_id):
    """Normalize folder-style arXiv ID to standard format."""
    return str(arxiv_id).strip().replace("-", ".")


def _iter_jsonl(path):
    """Read one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_references(corpus_path, paper_ids):
    """
    Load references from:
    corpus_path / range_folder / paper_folder / references.json

    Supports:
    - One paper object containing a nested 'references' list.
    - A flat reference object.
    - A list of flat reference objects.
    """
    corpus_path = Path(corpus_path)
    paper_ids = set(paper_ids)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus folder does not exist: {corpus_path}"
        )

    if not corpus_path.is_dir():
        raise NotADirectoryError(
            f"Expected a corpus directory: {corpus_path}"
        )

    reference_files = sorted(
        corpus_path.rglob("references.json")
    )

    if not reference_files:
        raise FileNotFoundError(
            f"No references.json found inside: {corpus_path}"
        )

    refs_by_paper = defaultdict(list)

    for file_path in reference_files:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Get publication ID from its parent folder.
        folder_id = _normalize_arxiv_id(file_path.parent.name)

        if isinstance(data, dict):
            paper_id = _normalize_arxiv_id(data.get("arxiv_id", ""))

            if not paper_id:
                raise ValueError(
                    f"Missing arxiv_id in {file_path}"
                )

            if paper_id != folder_id:
                raise ValueError(
                    f"Publication ID mismatch: "
                    f"folder={folder_id}, "
                    f"file arxiv_id={paper_id}, "
                    f"file={file_path}"
                )

            if "references" in data:
                references = data["references"]
            elif "ref_arxiv_id" in data or "ref_title" in data:
                references = [data]
            else:
                raise ValueError(
                    f"Unsupported references format: {file_path}"
                )

            if not isinstance(references, list):
                raise ValueError(
                    f"'references' must be a list in {file_path}"
                )

            if paper_id in paper_ids:
                refs_by_paper[paper_id].extend(references)

        elif isinstance(data, list):
            for ref in data:
                if not isinstance(ref, dict):
                    raise ValueError(
                        f"Invalid reference record in {file_path}"
                    )

                paper_id = _normalize_arxiv_id(
                    ref.get("arxiv_id", "")
                )

                if not paper_id:
                    raise ValueError(
                        f"Missing arxiv_id in {file_path}"
                    )

                if paper_id != folder_id:
                    raise ValueError(
                        f"Publication ID mismatch: "
                        f"folder={folder_id}, "
                        f"file arxiv_id={paper_id}, "
                        f"file={file_path}"
                    )

                if paper_id in paper_ids:
                    refs_by_paper[paper_id].append(ref)

        else:
            raise ValueError(
                f"Invalid JSON structure in {file_path}"
            )

    return dict(refs_by_paper)


def build_inventory(bibtex_path, references_path, paper_ids):
    """Count BibTeX entries and references for each publication."""

    paper_ids = list(dict.fromkeys(
        _normalize_arxiv_id(a) for a in paper_ids
    ))
    ids = set(paper_ids)

    bib_counts = defaultdict(int)

    for record in _iter_jsonl(bibtex_path):
        arxiv_id = _normalize_arxiv_id(
            record.get("arxiv_id", "")
        )

        if arxiv_id in ids:
            bib_counts[arxiv_id] += 1

    refs = load_references(references_path, ids)

    rows = []

    for arxiv_id in paper_ids:
        reference_list = refs.get(arxiv_id, [])

        rows.append({
            "arxiv_id": arxiv_id,
            "bibtex_count": bib_counts[arxiv_id],
            "reference_count": len(reference_list),
            "arxiv_reference_count": sum(
                bool(ref.get("ref_arxiv_id"))
                for ref in reference_list
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("arxiv_id")
        .reset_index(drop=True)
    )


def select_manual(inv, member_id, range_id):
    """Select 5 main publications and 1 spare publication."""

    eligible = inv[
        inv["bibtex_count"] >= MIN_BIBTEX
    ]

    required = N_MAIN + N_SPARE

    if len(eligible) < required:
        raise ValueError(
            f"Only {len(eligible)} eligible publications "
            f"(bibtex_count >= {MIN_BIBTEX}), "
            f"but {required} are required."
        )

    selected = eligible.sample(
        n=required,
        random_state=SEED
    ).reset_index(drop=True)

    partitions = [
        "train",
        "train",
        "train",
        "valid",
        "test",
        "spare",
    ]

    return pd.DataFrame({
        "arxiv_id": selected["arxiv_id"],
        "member_id": member_id,
        "range_id": range_id,
        "partition": partitions,
        "label_source": "manual",
    })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bibtex", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--paper-ids", required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--range-id", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    with open(args.paper_ids, encoding="utf-8") as f:
        paper_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    inventory = build_inventory(
        args.bibtex,
        args.references,
        paper_ids,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(
        output_dir / "data_inventory.csv",
        index=False,
    )

    assignment = select_manual(
        inventory,
        args.member_id,
        args.range_id,
    )

    assignment.to_csv(
        output_dir / f"manual_publication_assignment_{args.member_id}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
