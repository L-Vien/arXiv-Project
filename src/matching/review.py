"""Prepare human-friendly review files for manual reference labeling."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from .labeling import LABEL_COLUMNS


KEY_COLUMNS = ["arxiv_id", "citation_key"]
EDITABLE_COLUMNS = [
    "groundtruth_arxiv_id",
    "label_source",
    "status",
    "annotator",
    "reviewer",
    "confidence",
    "notes",
]


def load_annotation_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet annotation table as strings."""

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path, dtype=str).fillna("")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path).fillna("")
    raise ValueError("Input must be CSV or Parquet.")


def save_review_table(data: pd.DataFrame, path: str | Path) -> None:
    """Atomically save a review table as CSV or Parquet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Output must end with .csv or .parquet.")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=suffix,
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    os.close(descriptor)
    try:
        if suffix == ".csv":
            data.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        else:
            data.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _indexed_table(data: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = set(KEY_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{source} is missing key columns: {sorted(missing)}")

    result = data.copy()
    for column in KEY_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
    if (result[KEY_COLUMNS] == "").any().any():
        raise ValueError(f"{source} contains an empty annotation key.")
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} contains duplicate annotation keys.")
    return result.set_index(KEY_COLUMNS, drop=False)


def load_resume_table(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[pd.DataFrame, int]:
    """Load the current template and overlay annotations saved previously."""

    template = _indexed_table(load_annotation_table(input_path), "Input template")
    saved_path = Path(output_path)
    if not saved_path.is_file():
        return template.reset_index(drop=True), 0

    saved = _indexed_table(load_annotation_table(saved_path), "Saved output")
    extra_keys = saved.index.difference(template.index)
    if len(extra_keys):
        examples = [tuple(key) for key in extra_keys[:3]]
        raise ValueError(
            f"Saved output contains {len(extra_keys)} row(s) not present in "
            f"the input template, e.g. {examples}."
        )

    result = template.copy()
    for column in EDITABLE_COLUMNS:
        if column in saved.columns and column in result.columns:
            result.loc[saved.index, column] = saved[column]

    restored_count = 0
    if "status" in saved.columns:
        restored_count = int(saved["status"].astype(str).str.strip().ne("").sum())
    return result.reset_index(drop=True), restored_count


def parse_candidates(value: object) -> list[dict[str, object]]:
    """Parse the candidates JSON stored in an annotation row."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [candidate for candidate in value if isinstance(candidate, dict)]
    try:
        parsed = json.loads(str(value).strip() or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid candidates_json: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("candidates_json must contain a JSON list.")
    return [candidate for candidate in parsed if isinstance(candidate, dict)]


def candidate_count(row: pd.Series) -> int:
    """Return a reliable candidate count even when the source is missing it."""

    value = str(row.get("candidate_count", "")).strip()
    if value:
        try:
            return max(0, int(float(value)))
        except ValueError:
            pass
    return len(parse_candidates(row.get("candidates_json", "[]")))


def prefill_no_candidate_labels(labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prefill only blank rows that have no arXiv candidate.

    The label source remains ``manual`` because the rows are still part of the
    manual review artifact and can be changed by the annotator.
    """

    required = {"arxiv_id", "citation_key", "status", "label_source"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = labels.copy()
    changed = 0
    for index, row in result.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status or candidate_count(row) != 0:
            continue

        result.at[index, "status"] = "no_match"
        result.at[index, "label_source"] = "manual"
        if "groundtruth_arxiv_id" in result.columns:
            result.at[index, "groundtruth_arxiv_id"] = ""
        if "notes" in result.columns and not str(row.get("notes", "")).strip():
            result.at[index, "notes"] = "Prefilled: publication has no arXiv candidate."
        changed += 1

    return result, changed


def _safe_publication_filename(arxiv_id: object) -> str:
    value = str(arxiv_id).strip() or "unknown-publication"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def prepare_review_bundle(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    prefill_no_candidates: bool = True,
) -> dict[str, object]:
    """Create a master review file and one review file per publication."""

    labels = load_annotation_table(input_path)
    missing = set(LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "candidates_json" not in labels.columns:
        raise ValueError("Input must contain candidates_json.")

    labels = labels.sort_values(["arxiv_id", "citation_key"], kind="stable").reset_index(drop=True)
    changed = 0
    if prefill_no_candidates:
        labels, changed = prefill_no_candidate_labels(labels)

    output_root = Path(output_dir)
    publication_dir = output_root / "publications"
    output_root.mkdir(parents=True, exist_ok=True)
    publication_dir.mkdir(parents=True, exist_ok=True)

    master_path = output_root / "labels_review.csv"
    save_review_table(labels, master_path)

    publication_files: list[str] = []
    for arxiv_id, publication in labels.groupby("arxiv_id", sort=True):
        publication_path = publication_dir / f"{_safe_publication_filename(arxiv_id)}.csv"
        save_review_table(publication.reset_index(drop=True), publication_path)
        publication_files.append(str(publication_path))

    counts = labels["status"].astype(str).str.strip().replace("", "pending").value_counts()
    candidate_counts = labels.apply(candidate_count, axis=1)
    summary = pd.DataFrame(
        [
            {
                "total_entries": len(labels),
                "publication_count": labels["arxiv_id"].nunique(),
                "candidate_entries": int((candidate_counts > 0).sum()),
                "no_candidate_entries": int((candidate_counts == 0).sum()),
                "prefilled_no_candidate_entries": changed,
                "pending_entries": int((labels["status"].astype(str).str.strip() == "").sum()),
                "match_entries": int(counts.get("match", 0)),
                "no_match_entries": int(counts.get("no_match", 0)),
                "uncertain_entries": int(counts.get("uncertain", 0)),
            }
        ]
    )
    summary_path = output_root / "review_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "master_path": str(master_path),
        "summary_path": str(summary_path),
        "publication_files": publication_files,
        "prefilled_no_candidate_entries": changed,
        "total_entries": len(labels),
    }
