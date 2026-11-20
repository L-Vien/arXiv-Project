
"""
labeling.py
Prepare and validate manual labels for the arXiv citation project.

Each row represents ONE BibTeX entry.
Primary key: (arxiv_id, citation_key)

Commands:
    prepare  : Create annotation template from assignment, BibTeX, references.
    validate : Validate completed annotation file and export final label schema.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from collections import defaultdict

import pandas as pd


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


# ============================================================
# 1. DATA LOADING
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
        return "; ".join(
            str(item) for item in value if item is not None
        )

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value).strip()


def normalize_arxiv_id(value: Any) -> str | None:
    """Normalize arXiv ID for comparison."""

    value = clean_text(value)

    if not value:
        return None

    value = re.sub(
        r"^https?://(www\.)?arxiv\.org/(abs|pdf)/",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)

    return value.rstrip("/").lower()


def load_references(path: str | Path) -> dict:
    path = Path(path)
    refs_by_paper = defaultdict(list)

    def add_record(record):
        if not isinstance(record, dict):
            return

        arxiv_id = record.get("arxiv_id")

        # Format: one paper object containing a references list
        if arxiv_id and "references" in record:
            refs_by_paper[str(arxiv_id)].extend(
                record.get("references") or []
            )

        # Format: one flat reference record
        elif arxiv_id and (
            "ref_arxiv_id" in record or "ref_title" in record
        ):
            refs_by_paper[str(arxiv_id)].append(record)

        # Format: dictionary keyed by arxiv_id
        elif not arxiv_id:
            for paper_id, value in record.items():
                if isinstance(value, dict):
                    references = value.get("references", [])
                elif isinstance(value, list):
                    references = value
                else:
                    continue

                refs_by_paper[str(paper_id)].extend(references)

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
            raise ValueError(
                f"Expected references.json, got: {path.name}"
            )

        read_json_file(path)

    elif path.is_dir():
        # Only find references.json recursively
        reference_files = sorted(
            path.rglob("references.json")
        )

        if not reference_files:
            raise FileNotFoundError(
                f"No references.json found in: {path}"
            )

        for file_path in reference_files:
            read_json_file(file_path)

    else:
        raise FileNotFoundError(
            f"References path not found: {path}"
        )

    return dict(refs_by_paper)


# ============================================================
# 2. CANDIDATE GENERATION
# ============================================================

def candidate_view(reference: dict) -> dict:
    """Select useful metadata for displaying a candidate."""

    return {
        "ref_arxiv_id": clean_text(
            reference.get("ref_arxiv_id")
        ),

        "ref_title": clean_text(
            reference.get("ref_title")
        ),

        "ref_authors": clean_text(
            reference.get("ref_authors")
        ),

        "ref_doi": clean_text(
            reference.get("ref_doi") or reference.get("doi")
        ),

        "ref_year": clean_text(
            reference.get("ref_year")
            or reference.get("year")
            or reference.get("ref_submission_date")
        ),
    }


def get_candidates(references: list[dict]) -> list[dict]:
    """
    Get unique arXiv references from ONE publication.

    Candidate references from other publications
    must never be included.
    """

    candidates = []
    seen_ids = set()

    for reference in references:

        ref_id = clean_text(
            reference.get("ref_arxiv_id")
        )

        normalized_id = normalize_arxiv_id(ref_id)

        # Ignore references without arXiv ID
        if not normalized_id:
            continue

        # Remove duplicate reference IDs
        if normalized_id in seen_ids:
            continue

        seen_ids.add(normalized_id)

        candidates.append(
            candidate_view(reference)
        )

    return candidates


# ============================================================
# 3. PREPARE ANNOTATION TEMPLATE
# ============================================================

def prepare_annotation_template(
    assignment_path: str | Path,
    bibtex_path: str | Path,
    references_path: str | Path,
) -> pd.DataFrame:
    """
    Create one annotation row per BibTeX entry.

    The annotator fills:
        groundtruth_arxiv_id
        status
        confidence
        notes
    """

    # Load assignments
    assignments = pd.read_csv(
        assignment_path,
        dtype=str,
    ).fillna("")

    if "arxiv_id" not in assignments.columns:
        raise ValueError(
            "Assignment file must contain arxiv_id."
        )

    assignments["arxiv_id"] = (
        assignments["arxiv_id"].str.strip()
    )

    assignments = assignments[
        assignments["arxiv_id"] != ""
    ]

    if assignments["arxiv_id"].duplicated().any():
        raise ValueError(
            "Duplicate arxiv_id in assignment file."
        )

    assignment_by_id = (
        assignments.set_index("arxiv_id").to_dict("index")
    )

    assigned_ids = set(assignment_by_id.keys())

    # Load references
    references_by_paper = load_references(
        references_path
    )

    # Load BibTeX entries
    bibtex_by_key = {}

    for record in iter_jsonl(bibtex_path):

        publication_id = clean_text(
            record.get("arxiv_id")
        )

        if publication_id not in assigned_ids:
            continue

        citation_key = clean_text(
            record.get("citation_key")
        )

        if not citation_key:
            raise ValueError(
                f"Missing citation_key in publication {publication_id}"
            )

        key = (publication_id, citation_key)

        if key in bibtex_by_key:
            raise ValueError(
                f"Duplicate BibTeX entry: {key}"
            )

        bibtex_by_key[key] = record

    rows = []

    # Create one row for each BibTeX entry
    for (publication_id, citation_key), bib in sorted(
        bibtex_by_key.items()
    ):

        # IMPORTANT:
        # Only use references belonging to this publication.
        references = references_by_paper.get(
            publication_id, []
        )

        candidates = get_candidates(references)

        assignment = assignment_by_id[publication_id]

        row = {
            # Primary key
            "arxiv_id": publication_id,
            "citation_key": citation_key,

            # BibTeX information
            "bibtex_entry_type": clean_text(
                bib.get("entry_type")
            ),
            "bibtex_title": clean_text(
                bib.get("title")
            ),
            "bibtex_authors": clean_text(
                bib.get("authors")
            ),
            "bibtex_year": clean_text(
                bib.get("year")
            ),
            "bibtex_doi": clean_text(
                bib.get("doi")
            ),
            "bibtex_url": clean_text(
                bib.get("url")
            ),

            # Assignment information
            "partition": clean_text(
                assignment.get("partition")
            ),
            "member_id": clean_text(
                assignment.get("member_id")
            ),
            "range_id": clean_text(
                assignment.get("range_id")
            ),

            # Candidate information
            "candidate_count": len(candidates),
            "candidates_json": json.dumps(
                candidates,
                ensure_ascii=False,
            ),

            # Label fields
            "groundtruth_arxiv_id": "",
            "label_source": "manual",
            "status": "",
            "annotator": clean_text(
                assignment.get("member_id")
            ),
            "reviewer": "",
            "confidence": "",
            "notes": "",
        }

        rows.append(row)

    if not rows:
        raise ValueError(
            "No BibTeX entries found for assigned publications. "
            "Please check arxiv_id in assignment and BibTeX files."
        )

    result = pd.DataFrame(rows)

    if result.duplicated(
        ["arxiv_id", "citation_key"]
    ).any():
        raise ValueError(
            "Duplicate (arxiv_id, citation_key) in output."
        )

    return result


# ============================================================
# 4. LABEL VALIDATION
# ============================================================

def validate_labels(
    labels: pd.DataFrame,
    allow_uncertain: bool = True,
    allow_outside_candidates: bool = False,
) -> list[str]:
    """
    Validate completed annotation rows.

    Returns a list of errors.
    Empty list means validation passed.
    """

    errors = []

    missing_columns = (
        set(LABEL_COLUMNS) - set(labels.columns)
    )

    if missing_columns:
        return [
            f"Missing columns: {sorted(missing_columns)}"
        ]

    # Check primary key
    if labels[
        ["arxiv_id", "citation_key"]
    ].isna().any().any():
        errors.append(
            "arxiv_id and citation_key cannot be null."
        )

    if labels[
        ["arxiv_id", "citation_key"]
    ].astype(str).apply(
        lambda col: col.str.strip().eq("")
    ).any().any():
        errors.append(
            "arxiv_id and citation_key cannot be empty."
        )

    if labels.duplicated(
        ["arxiv_id", "citation_key"]
    ).any():
        errors.append(
            "Duplicate (arxiv_id, citation_key) keys."
        )

    # Check individual rows
    for row_number, (_, row) in enumerate(labels.iterrows(), start=2):

        status = clean_text(
            row["status"]
        ).lower()

        groundtruth = clean_text(
            row["groundtruth_arxiv_id"]
        )

        confidence = clean_text(
            row["confidence"]
        ).lower()

        label_source = clean_text(
            row["label_source"]
        ).lower()

        if status not in VALID_STATUSES:
            errors.append(
                f"Row {row_number}: invalid status '{status}'."
            )
            continue

        # MATCH must have groundtruth
        if status == "match" and not groundtruth:
            errors.append(
                f"Row {row_number}: match requires groundtruth_arxiv_id."
            )

        # NO_MATCH and UNCERTAIN must not have groundtruth
        if status in {"no_match", "uncertain"} and groundtruth:
            errors.append(
                f"Row {row_number}: {status} requires empty groundtruth."
            )

        # Final labels cannot contain uncertain
        if status == "uncertain" and not allow_uncertain:
            errors.append(
                f"Row {row_number}: uncertain is not allowed in final labels."
            )

        if label_source != "manual":
            errors.append(
                f"Row {row_number}: label_source must be manual."
            )

        if confidence and confidence not in VALID_CONFIDENCE:
            errors.append(
                f"Row {row_number}: invalid confidence '{confidence}'."
            )

        # Check whether groundtruth belongs to candidate list
        if (
            status == "match"
            and groundtruth
            and not allow_outside_candidates
            and "candidates_json" in labels.columns
        ):

            try:
                candidates = json.loads(
                    clean_text(row["candidates_json"]) or "[]"
                )

                candidate_ids = {
                    normalize_arxiv_id(
                        candidate.get("ref_arxiv_id")
                    )
                    for candidate in candidates
                    if isinstance(candidate, dict)
                }

                if (
                    normalize_arxiv_id(groundtruth)
                    not in candidate_ids
                ):
                    errors.append(
                        f"Row {row_number}: groundtruth is not "
                        "in the candidate list. If independently verified, "
                        "use --allow-outside-candidates."
                    )

            except (json.JSONDecodeError, TypeError):
                errors.append(
                    f"Row {row_number}: invalid candidates_json."
                )

    return errors


# ============================================================
# 5. SAVE OUTPUT
# ============================================================

def save_table(
    data: pd.DataFrame,
    output_path: str | Path,
):
    """Save DataFrame as CSV or Parquet."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix.lower()

    if suffix == ".csv":

        data.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

    elif suffix in {".parquet", ".pq"}:

        try:
            data.to_parquet(
                output_path,
                index=False,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Install pyarrow to save Parquet files: "
                "pip install pyarrow"
            ) from exc

    else:
        raise ValueError(
            "Output must end with .csv or .parquet"
        )


# ============================================================
# 6. COMMAND: PREPARE
# ============================================================

def command_prepare(args):

    template = prepare_annotation_template(
        assignment_path=args.assignment,
        bibtex_path=args.bibtex,
        references_path=args.references,
    )

    save_table(
        template,
        args.output,
    )

    print("\nAnnotation template created successfully!")
    print("------------------------------------------")
    print(f"Output: {args.output}")
    print(
        f"Publications: {template['arxiv_id'].nunique()}"
    )
    print(
        f"BibTeX entries: {len(template)}"
    )
    print(
        "Entries without candidates: "
        f"{(template['candidate_count'] == 0).sum()}"
    )

    return 0


# ============================================================
# 7. COMMAND: VALIDATE
# ============================================================

def command_validate(args):

    input_path = Path(args.input)

    if input_path.suffix.lower() == ".csv":

        labels = pd.read_csv(
            input_path,
            dtype=str,
        ).fillna("")

    elif input_path.suffix.lower() in {".parquet", ".pq"}:

        labels = pd.read_parquet(
            input_path
        ).fillna("")

    else:
        raise ValueError(
            "Input must be CSV or Parquet."
        )

    errors = validate_labels(
        labels,
        allow_uncertain=not args.final,
        allow_outside_candidates=args.allow_outside_candidates,
    )

    if errors:

        print(
            f"\nValidation failed: {len(errors)} issue(s)"
        )

        for error in errors:
            print(f"- {error}")

        return 1

    # Export only the agreed label schema
    output = labels[LABEL_COLUMNS].copy()

    save_table(
        output,
        args.output,
    )

    print("\nValidation passed!")
    print("------------------------------------------")
    print(f"Output: {args.output}")
    print(f"Total labels: {len(output)}")
    print("\nStatus distribution:")
    print(
        output["status"].value_counts(dropna=False)
    )

    return 0


# ============================================================
# 8. CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description="Manual labeling tool for arXiv references."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Prepare command
    prepare = subparsers.add_parser(
        "prepare",
        help="Create annotation template.",
    )

    prepare.add_argument(
        "--assignment",
        required=True,
        help="Manual publication assignment CSV.",
    )

    prepare.add_argument(
        "--bibtex",
        required=True,
        help="BibTeX JSONL file.",
    )

    prepare.add_argument(
        "--references",
        required=True,
        help="References JSON or JSONL file.",
    )

    prepare.add_argument(
        "--output",
        required=True,
        help="Output annotation template CSV or Parquet.",
    )

    prepare.set_defaults(
        func=command_prepare
    )

    # Validate command
    validate = subparsers.add_parser(
        "validate",
        help="Validate completed labels.",
    )

    validate.add_argument(
        "--input",
        required=True,
        help="Completed annotation file.",
    )

    validate.add_argument(
        "--output",
        required=True,
        help="Validated labels output.",
    )

    validate.add_argument(
        "--final",
        action="store_true",
        help="Disallow uncertain labels in final output.",
    )

    validate.add_argument(
        "--allow-outside-candidates",
        action="store_true",
        help="Allow independently verified ground truth "
             "outside the candidate list.",
    )

    validate.set_defaults(
        func=command_validate
    )

    return parser


def main():

    parser = build_parser()
    args = parser.parse_args()

    try:
        exit_code = args.func(args)

    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as exc:

        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        exit_code = 1

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
