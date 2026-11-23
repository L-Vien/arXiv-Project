"""
labeling.py
Prepare and validate manual labels for the arXiv citation project.

Each row represents ONE BibTeX entry.
Primary key: (arxiv_id, citation_key)

Commands:
    prepare  : Create annotation template from assignment, BibTeX, references.
    validate : Validate completed annotation file and export final label schema.

ID convention (shared with selection.py):
    - publication ID: "-" -> "."  (2604-25535 -> 2604.25535)
    - reference ID  : strip URL prefix, ".pdf" and version suffix (v2)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rapidfuzz import fuzz

    def _similarity(a: str, b: str) -> int:
        return round(fuzz.token_set_ratio(a, b))

except ImportError:  # fallback so the tool still runs
    from difflib import SequenceMatcher

    def _similarity(a: str, b: str) -> int:
        return round(100 * SequenceMatcher(None, a, b).ratio())


LABEL_COLUMNS = [
    "arxiv_id",
    "citation_key",
    "groundtruth_arxiv_id",
    "label_source",
    "status",
    "annotator",
    "reviewer",
    "confidence",
    "notes",
]

VALID_STATUSES = {"match", "no_match", "uncertain"}
VALID_CONFIDENCE = {"high", "medium", "low"}

PUB_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
ARXIV_IN_TEXT_RE = re.compile(r"(\d{4}\.\d{4,5})")
MIN_LABELS_PER_PUB = 20
REVIEW_REQUIRED_PARTITIONS = {"valid", "test"}


# ============================================================
# 1. DATA LOADING + ID NORMALIZATION
# ============================================================

def iter_jsonl(path: str | Path):
    """Read JSONL file, one JSON object per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {path}, line {line_number}"
                )
            yield record


def clean_text(value: Any) -> str:
    """Convert metadata into readable text."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(i) for i in value if i is not None)
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_pub_id(value: Any) -> str:
    """Publication ID, same convention as selection.py."""
    return clean_text(value).replace("-", ".")


def normalize_arxiv_id(value: Any) -> str | None:
    """Reference arXiv ID: no URL prefix, no .pdf, no version suffix."""
    value = clean_text(value)
    if not value:
        return None
    value = re.sub(
        r"^https?://(www\.)?arxiv\.org/(abs|pdf)/", "", value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = value.rstrip("/").lower()
    value = re.sub(r"v\d+$", "", value)
    return value or None


def load_references(
    path: str | Path,
    only_ids: set[str] | None = None,
) -> dict[str, list[dict]]:
    """
    Load references into {publication_id: [reference dict, ...]}.

    Supports a single references.json or a directory searched recursively.
    only_ids: if given, skip files/records of other publications.
    NOTE: ideally import this from loading.py (Lan Vien) so that
    selection.py and labeling.py share one implementation.
    """
    path = Path(path)
    refs_by_paper: dict[str, list[dict]] = defaultdict(list)
    no_reference_key = 0

    def wanted(pub_id: str) -> bool:
        return only_ids is None or pub_id in only_ids

    def add_record(record):
        nonlocal no_reference_key
        if not isinstance(record, dict):
            return

        arxiv_id = normalize_pub_id(record.get("arxiv_id"))

        if arxiv_id and "references" in record:
            if wanted(arxiv_id):
                refs_by_paper[arxiv_id].extend(
                    r for r in (record.get("references") or [])
                    if isinstance(r, dict)
                )
        elif arxiv_id and (
            "ref_arxiv_id" in record or "ref_title" in record
        ):
            if wanted(arxiv_id):
                refs_by_paper[arxiv_id].append(record)
        elif arxiv_id:
            no_reference_key += 1
        else:
            # dictionary keyed by arxiv_id
            for paper_id, value in record.items():
                pid = normalize_pub_id(paper_id)
                if not wanted(pid):
                    continue
                if isinstance(value, dict):
                    refs = value.get("references") or []
                elif isinstance(value, list):
                    refs = value
                else:
                    continue
                refs_by_paper[pid].extend(
                    r for r in refs if isinstance(r, dict)
                )

    def read_json_file(file_path):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for record in data:
                add_record(record)
        else:
            add_record(data)

    if path.is_file():
        if path.name != "references.json":
            raise ValueError(f"Expected references.json, got: {path.name}")
        read_json_file(path)

    elif path.is_dir():
        files = sorted(path.rglob("references.json"))
        if not files:
            raise FileNotFoundError(f"No references.json found in: {path}")
        for file_path in files:
            folder_id = normalize_pub_id(file_path.parent.name)
            # skip unrelated publications early (corpus can be large)
            if only_ids is not None and folder_id not in only_ids:
                continue
            read_json_file(file_path)
    else:
        raise FileNotFoundError(f"References path not found: {path}")

    if no_reference_key:
        print(
            f"[WARN] {no_reference_key} record(s) had arxiv_id but no "
            "'references' key (treated as 0 references).",
            file=sys.stderr,
        )

    return dict(refs_by_paper)


# ============================================================
# 2. CANDIDATE GENERATION
# ============================================================

def candidate_view(reference: dict) -> dict:
    """Select useful metadata for displaying a candidate."""
    return {
        "ref_arxiv_id": normalize_arxiv_id(reference.get("ref_arxiv_id")) or "",
        "ref_title": clean_text(reference.get("ref_title")),
        "ref_authors": clean_text(reference.get("ref_authors")),
        "ref_doi": clean_text(reference.get("ref_doi") or reference.get("doi")),
        "ref_year": clean_text(
            reference.get("ref_year")
            or reference.get("year")
            or reference.get("ref_submission_date")
        ),
    }


def get_candidates(references: list[dict]) -> list[dict]:
    """Unique arXiv references from ONE publication (dedupe by base ID)."""
    candidates, seen = [], set()
    for reference in references:
        rid = normalize_arxiv_id(reference.get("ref_arxiv_id"))
        if not rid or rid in seen:
            continue
        seen.add(rid)
        candidates.append(candidate_view(reference))
    return candidates


def rank_candidates(bib_title: str, candidates: list[dict]) -> list[dict]:
    """Add a rough title-similarity score and sort (hint only)."""
    title = bib_title.lower()
    scored = []
    for c in candidates:
        score = _similarity(title, c["ref_title"].lower()) if title else 0
        scored.append({**c, "score": score})
    return sorted(scored, key=lambda c: (-c["score"], c["ref_arxiv_id"]))


def candidates_text(ranked: list[dict], top: int = 8) -> str:
    """Human-readable top candidates for one spreadsheet cell."""
    return "\n".join(
        f"{i}) {c['ref_arxiv_id']} | {c['score']} | "
        f"{c['ref_title'][:90]} | {c['ref_authors'][:50]} | {c['ref_year']}"
        for i, c in enumerate(ranked[:top], 1)
    )


def hint_arxiv_id(bib: dict, candidates: list[dict]) -> str:
    """arXiv ID found in BibTeX url/doi/eprint, only if it is a candidate."""
    ids = {c["ref_arxiv_id"] for c in candidates}
    for field in ("url", "doi", "eprint"):
        m = ARXIV_IN_TEXT_RE.search(clean_text(bib.get(field)))
        if m and m.group(1) in ids:
            return m.group(1)
    return ""


# ============================================================
# 3. TABLE I/O
# ============================================================

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).astype(str).replace("nan", "").fillna("")
    raise ValueError("Input must be .csv, .xlsx or .parquet")


def save_table(data: pd.DataFrame, output_path: str | Path):
    """Save as CSV, XLSX (IDs kept as text) or Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        try:
            data.astype(str).to_excel(output_path, index=False)
        except ImportError as exc:
            raise RuntimeError("pip install openpyxl") from exc
    elif suffix in {".parquet", ".pq"}:
        try:
            data.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise RuntimeError("pip install pyarrow") from exc
    else:
        raise ValueError("Output must end with .csv, .xlsx or .parquet")


# ============================================================
# 4. PREPARE ANNOTATION TEMPLATE
# ============================================================

def prepare_annotation_template(
    assignment_path: str | Path,
    bibtex_path: str | Path,
    references_path: str | Path,
) -> pd.DataFrame:
    """One annotation row per BibTeX entry."""

    assignments = pd.read_csv(assignment_path, dtype=str).fillna("")
    if "arxiv_id" not in assignments.columns:
        raise ValueError("Assignment file must contain arxiv_id.")

    assignments["arxiv_id"] = assignments["arxiv_id"].map(normalize_pub_id)
    assignments = assignments[assignments["arxiv_id"] != ""]

    if assignments["arxiv_id"].duplicated().any():
        raise ValueError("Duplicate arxiv_id in assignment file.")

    assignment_by_id = assignments.set_index("arxiv_id").to_dict("index")
    assigned_ids = {str(k) for k in assignment_by_id}

    references_by_paper = load_references(references_path, only_ids=assigned_ids)

    bibtex_by_key = {}
    for record in iter_jsonl(bibtex_path):
        publication_id = normalize_pub_id(record.get("arxiv_id"))
        if publication_id not in assigned_ids:
            continue

        citation_key = clean_text(record.get("citation_key"))
        if not citation_key:
            raise ValueError(f"Missing citation_key in publication {publication_id}")

        key = (publication_id, citation_key)
        if key in bibtex_by_key:
            raise ValueError(f"Duplicate BibTeX entry: {key}")
        bibtex_by_key[key] = record

    rows = []
    for (publication_id, citation_key), bib in sorted(bibtex_by_key.items()):
        # only references of THIS publication
        candidates = rank_candidates(
            clean_text(bib.get("title")),
            get_candidates(references_by_paper.get(publication_id, [])),
        )
        assignment = assignment_by_id[publication_id]

        rows.append({
            "arxiv_id": publication_id,
            "citation_key": citation_key,

            "bibtex_entry_type": clean_text(bib.get("entry_type")),
            "bibtex_title": clean_text(bib.get("title")),
            "bibtex_authors": clean_text(bib.get("authors")),
            "bibtex_year": clean_text(bib.get("year")),
            "bibtex_doi": clean_text(bib.get("doi")),
            "bibtex_url": clean_text(bib.get("url")),

            "partition": clean_text(assignment.get("partition")),
            "member_id": clean_text(assignment.get("member_id")),
            "range_id": clean_text(assignment.get("range_id")),

            "candidate_count": len(candidates),
            "hint_arxiv_id": hint_arxiv_id(bib, candidates),   # hint only
            "top_candidates": candidates_text(candidates),     # hint only
            "candidates_json": json.dumps(candidates, ensure_ascii=False),

            "groundtruth_arxiv_id": "",
            "label_source": "manual",
            "status": "",
            "annotator": clean_text(assignment.get("member_id")),
            "reviewer": "",
            "confidence": "",
            "notes": "",
        })

    if not rows:
        raise ValueError(
            "No BibTeX entries found for assigned publications. "
            "Check arxiv_id in assignment and BibTeX files."
        )

    result = pd.DataFrame(rows)
    if result.duplicated(["arxiv_id", "citation_key"]).any():
        raise ValueError("Duplicate (arxiv_id, citation_key) in output.")
    return result


# ============================================================
# 5. LABEL VALIDATION
# ============================================================

def normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Canonical form used both for validation and for the exported file."""
    out = labels.copy()
    out["arxiv_id"] = out["arxiv_id"].map(normalize_pub_id)
    out["citation_key"] = out["citation_key"].map(clean_text)
    out["status"] = out["status"].map(lambda v: clean_text(v).lower())
    out["confidence"] = out["confidence"].map(lambda v: clean_text(v).lower())
    out["label_source"] = out["label_source"].map(lambda v: clean_text(v).lower())
    out["groundtruth_arxiv_id"] = out["groundtruth_arxiv_id"].map(
        lambda v: normalize_arxiv_id(v) or ""
    )
    for col in ("annotator", "reviewer", "notes"):
        out[col] = out[col].map(clean_text)
    return out


def validate_labels(
    labels: pd.DataFrame,
    allow_uncertain: bool = True,
    allow_outside_candidates: bool = False,
    template: pd.DataFrame | None = None,
    final: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Returns (errors, warnings). Empty errors means validation passed.
    `labels` must already be normalized (see normalize_labels).
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = set(LABEL_COLUMNS) - set(labels.columns)
    if missing_columns:
        return [f"Missing columns: {sorted(missing_columns)}"], warnings

    # --- primary key ---
    if (labels[["arxiv_id", "citation_key"]] == "").any().any():
        errors.append("arxiv_id and citation_key cannot be empty.")

    bad_ids = labels.loc[~labels["arxiv_id"].str.match(PUB_ID_RE), "arxiv_id"]
    if len(bad_ids):
        errors.append(
            f"{len(bad_ids)} row(s) with invalid arxiv_id "
            f"(Excel may have altered it), e.g. {sorted(set(bad_ids))[:5]}"
        )

    if labels.duplicated(["arxiv_id", "citation_key"]).any():
        errors.append("Duplicate (arxiv_id, citation_key) keys.")

    # --- compare with template: no deleted / added rows ---
    if template is not None:
        tpl_keys = set(zip(
            template["arxiv_id"].map(normalize_pub_id),
            template["citation_key"].map(clean_text),
        ))
        lab_keys = set(zip(labels["arxiv_id"], labels["citation_key"]))
        missing = tpl_keys - lab_keys
        extra = lab_keys - tpl_keys
        if missing:
            errors.append(
                f"{len(missing)} BibTeX entr(ies) missing from labels "
                f"(deleted?), e.g. {sorted(missing)[:3]}"
            )
        if extra:
            errors.append(
                f"{len(extra)} row(s) not in template, e.g. {sorted(extra)[:3]}"
            )

    # --- row checks ---
    has_candidates = "candidates_json" in labels.columns
    has_partition = "partition" in labels.columns

    for row_number, (_, row) in enumerate(labels.iterrows(), start=2):
        status = row["status"]
        groundtruth = row["groundtruth_arxiv_id"]
        confidence = row["confidence"]

        if status not in VALID_STATUSES:
            errors.append(f"Row {row_number}: invalid status '{status}'.")
            continue

        if status == "match" and not groundtruth:
            errors.append(f"Row {row_number}: match requires groundtruth_arxiv_id.")

        if status in {"no_match", "uncertain"} and groundtruth:
            errors.append(f"Row {row_number}: {status} requires empty groundtruth.")

        if status == "uncertain" and not allow_uncertain:
            errors.append(f"Row {row_number}: uncertain is not allowed in final labels.")

        if row["label_source"] != "manual":
            errors.append(f"Row {row_number}: label_source must be manual.")

        if not row["annotator"]:
            errors.append(f"Row {row_number}: annotator is empty.")

        if confidence and confidence not in VALID_CONFIDENCE:
            errors.append(f"Row {row_number}: invalid confidence '{confidence}'.")

        # review requirement (final only)
        if final and not row["reviewer"]:
            partition = clean_text(row["partition"]).lower() if has_partition else ""
            if not has_partition or partition in REVIEW_REQUIRED_PARTITIONS:
                errors.append(
                    f"Row {row_number}: reviewer required in final labels "
                    f"(partition='{partition or 'unknown'}')."
                )

        if final and row["reviewer"] and row["reviewer"] == row["annotator"]:
            errors.append(f"Row {row_number}: reviewer must differ from annotator.")

        # groundtruth must be one of the candidates
        if (
            status == "match" and groundtruth
            and not allow_outside_candidates and has_candidates
        ):
            try:
                candidates = json.loads(clean_text(row["candidates_json"]) or "[]")
                ids = {
                    normalize_arxiv_id(c.get("ref_arxiv_id"))
                    for c in candidates if isinstance(c, dict)
                }
                if groundtruth not in ids:
                    errors.append(
                        f"Row {row_number}: groundtruth '{groundtruth}' is not in "
                        "the candidate list. If independently verified, use "
                        "--allow-outside-candidates."
                    )
            except (json.JSONDecodeError, TypeError):
                errors.append(f"Row {row_number}: invalid candidates_json.")

    # --- per-publication size ---
    counts = labels.groupby("arxiv_id").size()
    for pub, n in counts.items():
        if n < MIN_LABELS_PER_PUB:
            warnings.append(
                f"Publication {pub} has only {n} labels (< {MIN_LABELS_PER_PUB})."
            )

    # reviewer on train rows in final mode is recommended, not mandatory
    if final and has_partition:
        n_unrev = int(
            ((labels["reviewer"] == "")
             & (~labels["partition"].str.lower().isin(REVIEW_REQUIRED_PARTITIONS))).sum()
        )
        if n_unrev:
            warnings.append(f"{n_unrev} train row(s) not yet reviewed.")

    return errors, warnings


# ============================================================
# 6. COMMANDS
# ============================================================

def command_prepare(args):
    template = prepare_annotation_template(
        assignment_path=args.assignment,
        bibtex_path=args.bibtex,
        references_path=args.references,
    )
    save_table(template, args.output)

    print("\nAnnotation template created successfully!")
    print("------------------------------------------")
    print(f"Output: {args.output}")
    print(f"Publications: {template['arxiv_id'].nunique()}")
    print(f"BibTeX entries: {len(template)}")
    print(f"Entries without candidates: {(template['candidate_count'] == 0).sum()}")
    print("\nEntries per publication:")
    print(template.groupby("arxiv_id").size().to_string())
    print(
        "\nTip: open .csv with Google Sheets (disable 'Convert text to "
        "numbers') or use --output ....xlsx to avoid Excel changing IDs."
    )
    return 0


def command_validate(args):
    labels = normalize_labels(read_table(args.input))

    template = read_table(args.template) if args.template else None

    errors, warnings = validate_labels(
        labels,
        allow_uncertain=not args.final,
        allow_outside_candidates=args.allow_outside_candidates,
        template=template,
        final=args.final,
    )

    for w in warnings:
        print(f"[WARN] {w}")

    if errors:
        print(f"\nValidation failed: {len(errors)} issue(s)")
        for error in errors:
            print(f"- {error}")
        return 1

    output = labels[LABEL_COLUMNS].copy()
    # empty groundtruth -> real null in the exported file
    output["groundtruth_arxiv_id"] = output["groundtruth_arxiv_id"].replace("", None)
    for col in ("reviewer", "confidence", "notes"):
        output[col] = output[col].replace("", None)

    save_table(output, args.output)

    print("\nValidation passed!")
    print("------------------------------------------")
    print(f"Output: {args.output}")
    print(f"Total labels: {len(output)}")
    print("\nLabels per publication:")
    print(output.groupby("arxiv_id").size().to_string())
    print("\nStatus distribution:")
    print(output["status"].value_counts(dropna=False))
    return 0


# ============================================================
# 7. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Manual labeling tool for arXiv references."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Create annotation template.")
    prepare.add_argument("--assignment", required=True,
                         help="Manual publication assignment CSV.")
    prepare.add_argument("--bibtex", required=True, help="BibTeX JSONL file.")
    prepare.add_argument("--references", required=True,
                         help="references.json file or corpus directory.")
    prepare.add_argument("--output", required=True,
                         help="Template output (.csv, .xlsx or .parquet).")
    prepare.set_defaults(func=command_prepare)

    validate = sub.add_parser("validate", help="Validate completed labels.")
    validate.add_argument("--input", required=True, help="Completed annotation file.")
    validate.add_argument("--output", required=True, help="Validated labels output.")
    validate.add_argument("--template",
                          help="Original template, to detect deleted/added rows.")
    validate.add_argument("--final", action="store_true",
                          help="Final mode: no uncertain, reviewer required "
                               "for valid/test.")
    validate.add_argument("--allow-outside-candidates", action="store_true",
                          help="Allow independently verified ground truth "
                               "outside the candidate list.")
    validate.set_defaults(func=command_validate)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        exit_code = args.func(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()