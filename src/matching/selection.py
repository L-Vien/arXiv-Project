
"""Data inventory and manual publication assignment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
N_MAIN = 5
N_SPARE = 1

# Tier 0 là chuẩn; các tier sau nới dần nếu range ít dữ liệu
TIERS = [
    dict(min_bib=20, max_bib=30, min_refs=8, max_refs=40,
         ratio=(0.3, 0.9), max_title=0.2, max_author=0.3, max_year=0.4, max_dup=2),
    dict(max_bib=35),
    dict(max_refs=60),
    dict(max_title=0.4, max_author=0.4, ratio=(0.2, 1.0), max_dup=5),
]
MIN_TOTAL_BIB = 100      # 5 pub chính thức >= 100 label
MAX_TRIES = 300


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
    paper_ids = list(dict.fromkeys(_normalize_arxiv_id(a) for a in paper_ids))
    ids = set(paper_ids)

    bib = defaultdict(lambda: dict(n=0, no_title=0, no_author=0, no_year=0))
    for r in _iter_jsonl(bibtex_path):
        a = _normalize_arxiv_id(r.get("arxiv_id", ""))
        if a not in ids:
            continue
        s = bib[a]
        s["n"] += 1
        s["no_title"] += not r.get("title")
        s["no_author"] += not r.get("authors")
        s["no_year"] += not r.get("year")

    refs = load_references(references_path, ids)

    rows = []
    for a in paper_ids:
        s = bib[a]
        n = s["n"]
        ref_list = refs.get(a, [])
        arxiv_ids = [
            str(r["ref_arxiv_id"]).strip().lower()
            for r in ref_list if r.get("ref_arxiv_id")
        ]
        rows.append({
            "arxiv_id": a,
            "bibtex_count": n,
            "reference_count": len(ref_list),
            "arxiv_reference_count": len(set(arxiv_ids)),      # unique, khớp candidate list
            "duplicate_reference_count": len(arxiv_ids) - len(set(arxiv_ids)),
            "arxiv_ref_ratio": len(set(arxiv_ids)) / n if n else 0.0,
            "missing_title_rate": s["no_title"] / n if n else 1.0,
            "missing_author_rate": s["no_author"] / n if n else 1.0,
            "missing_year_rate": s["no_year"] / n if n else 1.0,
        })

    return pd.DataFrame(rows).sort_values("arxiv_id").reset_index(drop=True)


def _filter(inv, p):
    return inv[
        inv.bibtex_count.between(p["min_bib"], p["max_bib"])
        & inv.arxiv_reference_count.between(p["min_refs"], p["max_refs"])
        & inv.arxiv_ref_ratio.between(*p["ratio"])
        & (inv.missing_title_rate <= p["max_title"])
        & (inv.missing_author_rate <= p["max_author"])
        & (inv.missing_year_rate <= p["max_year"])
        & (inv.duplicate_reference_count <= p["max_dup"])
    ]


def _assign_partitions(main, spare):
    """Pub nhỏ nhất đứng đầu (label trước, làm quen tool).
    Valid/test lấy cỡ trung bình để reviewer đỡ quá tải."""
    main = main.sort_values(["bibtex_count", "arxiv_reference_count"]).reset_index(drop=True)
    parts = ["train", "train", "valid", "test", "train"]
    main["partition"] = parts
    spare = spare.assign(partition="spare")
    return pd.concat([main, spare], ignore_index=True)


def select_manual(inv, member_id, range_id):
    params = {}
    need = N_MAIN + N_SPARE

    for tier_no, override in enumerate(TIERS):
        params = {**params, **override}
        pool = _filter(inv, params)
        if len(pool) < need:
            continue

        for k in range(MAX_TRIES):
            pick = pool.sample(n=need, random_state=SEED + k).reset_index(drop=True)
            main, spare = pick.iloc[:N_MAIN], pick.iloc[N_MAIN:]
            if main.bibtex_count.sum() < MIN_TOTAL_BIB:
                continue
            if not (main.arxiv_ref_ratio < 0.5).any():   # ít nhất 1 pub nhiều no_match
                continue

            if tier_no > 0:
                print(f"[WARN] member {member_id}: phải nới bộ lọc đến tier {tier_no}: {params}")
            out = _assign_partitions(main, spare)
            return pd.DataFrame({
                "arxiv_id": out.arxiv_id,
                "member_id": member_id,
                "range_id": range_id,
                "partition": out.partition,
                "label_source": "manual",
                "bibtex_count": out.bibtex_count,
                "arxiv_reference_count": out.arxiv_reference_count,
                "filter_tier": tier_no,
            })

    raise ValueError(f"Không chọn được {need} publication đạt chuẩn cho {member_id}; "
                     f"kiểm tra lại data_inventory.csv.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bibtex", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--paper-ids", required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--range-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-bib", type=int)
    parser.add_argument("--max-bib", type=int)
    parser.add_argument("--min-refs", type=int)
    parser.add_argument("--max-refs", type=int)

    args = parser.parse_args()
    
    overrides = {
        "min_bib": args.min_bib, "max_bib": args.max_bib,
        "min_refs": args.min_refs, "max_refs": args.max_refs,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        TIERS[0].update(overrides)
        print(f"[INFO] Override tier 0: {overrides}.")

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
