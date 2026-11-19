"""Stream-validates a built six-table member dataset without modifying it."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypedDict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import SCHEMA_VERSION, normalize_arxiv_id
from .member_dataset import KEYS, SCHEMAS, TABLES, ids_in_manifest_range, load_member_ids


class ValidationReport(TypedDict):
    member_id: str | None
    schema_version: str
    validated_at: str
    passed: bool
    errors: list[dict[str, object]]
    warnings: list[dict[str, object]]
    checks: dict[str, dict[str, object]]
    row_counts: dict[str, dict[str, int]]
    range_summary: dict[str, int]

SENSITIVE_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+"),
)
LOCAL_PATH_PATTERN = re.compile(r"(?i)(?:[A-Z]:\\Users\\|/Users/[^/]+/|/home/[^/]+/|/mnt/data/)")
SENTINELS = {"NO_MATCH", "NONE", "NULL_REFERENCE"}


def _error(errors: list[dict[str, object]], code: str, table: str, description: str,
           affected_rows: int = 0) -> None:
    errors.append({
        "code": code,
        "table": table,
        "description": description,
        "affected_rows": affected_rows,
    })


def _iter_table_batches(path: Path, batch_size: int) -> Iterable[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch


def _contains_sensitive(table: pa.Table | pa.RecordBatch) -> tuple[bool, bool]:
    local = False
    token = False
    for column in table.columns:
        for value in column.to_pylist():
            if isinstance(value, list):
                value = "\n".join(str(item) for item in value)
            elif value is None:
                continue
            text = str(value)
            if LOCAL_PATH_PATTERN.search(text):
                local = True
            if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
                token = True
    return local, token


def validate_dataset(dataset_dir: Path, member_id: str | None = None,
                     range_manifest: Path | None = None, output: Path | None = None,
                     strict: bool = False, sample_rows: int = 1000) -> ValidationReport:
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    checks: dict[str, dict[str, object]] = {}
    row_counts: dict[str, dict[str, int]] = {}
    frames: dict[str, list[dict[str, object]]] = {table: [] for table in TABLES}

    if not dataset_dir.is_dir():
        _error(errors, "DATASET_DIR_MISSING", "*", "Dataset directory does not exist")
    else:
        allowed_ids: set[str] | None = None
        member_row: dict[str, str] | None = None
        if member_id is not None and range_manifest is not None:
            try:
                import csv
                with range_manifest.open("r", encoding="utf-8", newline="") as handle:
                    manifest_rows = list(csv.DictReader(handle))
                member_row = next((row for row in manifest_rows if str(row.get("member_id", "")) == str(member_id)), None)
                if member_row is None:
                    _error(errors, "MEMBER_NOT_IN_MANIFEST", "*", f"member_id '{member_id}' not found")
                else:
                    allowed_ids = load_member_ids(range_manifest, member_id)
                    if not member_row.get("paper_ids"):
                        allowed_ids = None
            except (OSError, ValueError) as exc:
                _error(errors, "RANGE_MANIFEST_ERROR", "*", str(exc))

        for table in TABLES:
            table_dir = dataset_dir / table
            shards = sorted(table_dir.glob("*.parquet")) if table_dir.is_dir() else []
            if not shards:
                _error(errors, "MISSING_TABLE", table, "No Parquet shard exists")
                row_counts[table] = {"rows": 0, "shards": 0}
                continue

            base_schema: pa.Schema | None = None
            count = 0
            duplicate_count = 0
            for shard in shards:
                try:
                    parquet = pq.ParquetFile(shard)
                    _ = parquet.metadata
                    schema = parquet.schema_arrow
                    if base_schema is None:
                        base_schema = schema
                    elif schema != base_schema:
                        _error(errors, "SCHEMA_SHARD_MISMATCH", table, f"Shard schema differs: {shard.name}")
                    if schema != SCHEMAS[table]:
                        _error(errors, "SCHEMA_MISMATCH", table, f"Schema does not match contract: {shard.name}")
                    for batch in _iter_table_batches(shard, sample_rows):
                        count += batch.num_rows
                        local, token = _contains_sensitive(batch)
                        if local:
                            _error(errors, "LOCAL_PATH_FOUND", table, f"Local path found in {shard.name}")
                        if token:
                            _error(errors, "TOKEN_FOUND", table, f"Token or secret found in {shard.name}")
                        if table in {"elements", "hierarchy"}:
                            pass
                        if len(frames[table]) < sample_rows:
                            frames[table].extend(batch.to_pylist())
                        if "arxiv_id" in batch.schema.names:
                            for value in batch.column(batch.schema.get_field_index("arxiv_id")).to_pylist():
                                normalized = normalize_arxiv_id(value)
                                if normalized is None:
                                    continue
                                if allowed_ids is not None and normalized not in allowed_ids:
                                    _error(errors, "PUBLICATION_OUT_OF_RANGE", table, f"arXiv ID outside range: {normalized}")
                                elif member_row is not None and allowed_ids is None and not ids_in_manifest_range(normalized, member_row):
                                    _error(errors, "PUBLICATION_OUT_OF_RANGE", table, f"arXiv ID outside range: {normalized}")
                except Exception as exc:
                    _error(errors, "PARQUET_READ_ERROR", table, f"Cannot read {shard.name}: {exc}")

            key_fields = KEYS[table]
            seen: set[tuple[str, ...]] = set()
            for row in frames[table]:
                if all(field in row for field in key_fields):
                    key = tuple(str(row[field]) for field in key_fields)
                    if key in seen:
                        duplicate_count += 1
                    seen.add(key)
            if duplicate_count:
                _error(errors, "DUPLICATE_KEY", table, f"{duplicate_count} duplicate key(s)", duplicate_count)
            row_counts[table] = {"rows": count, "shards": len(shards)}

        if allowed_ids is not None:
            range_paper_count = len(allowed_ids)
        elif member_row is not None:
            paper_count = len(frames["papers"])
            range_paper_count = paper_count
        else:
            range_paper_count = 0

        paper_ids = {row.get("arxiv_id") for row in frames["papers"] if row.get("arxiv_id")}
        for table in ("bibtex", "references", "elements", "hierarchy", "predictions"):
            if table not in row_counts:
                continue
            outsider = sum(
                1 for row in frames[table]
                if row.get("arxiv_id") not in paper_ids
            )
            if outsider:
                _error(errors, "ARXIV_ID_NOT_IN_PAPERS", table, f"{outsider} row(s) reference an absent paper")

        bibtex_keys = {(row.get("arxiv_id"), row.get("citation_key")) for row in frames["bibtex"]}
        prediction_keys = {(row.get("arxiv_id"), row.get("citation_key")) for row in frames["predictions"]}
        missing_predictions = bibtex_keys - prediction_keys
        if missing_predictions:
            _error(errors, "MISSING_PREDICTION", "predictions", f"{len(missing_predictions)} BibTeX entry/entries lack a prediction")

        references_by_publication: dict[str, set[str]] = {}
        for row in frames["references"]:
            arxiv_id = row.get("arxiv_id")
            ref_id = row.get("ref_arxiv_id")
            if isinstance(arxiv_id, str) and isinstance(ref_id, str):
                references_by_publication.setdefault(arxiv_id, set()).add(ref_id)
        for row in frames["predictions"]:
            candidates = [row.get(f"cand_{index}") for index in range(1, 6)]
            partition = row.get("partition")
            if partition not in {"train", "valid", "test"}:
                _error(errors, "INVALID_PARTITION", "predictions", f"Invalid partition: {partition}")
            populated = [candidate for candidate in candidates if candidate is not None]
            if any(str(candidate).upper() in SENTINELS for candidate in populated):
                _error(errors, "FAKE_CANDIDATE", "predictions", "Sentinel candidate value found")
            if len(set(populated)) != len(populated):
                _error(errors, "DUPLICATE_CANDIDATE", "predictions", "Candidate duplicated in one prediction")
            if any(candidate is None and later is not None for candidate, later in zip(candidates, candidates[1:])):
                _error(errors, "CANDIDATE_GAP", "predictions", "Candidates are not a contiguous prefix")
            arxiv_id = str(row.get("arxiv_id"))
            wrong = [candidate for candidate in populated if candidate not in references_by_publication.get(arxiv_id, set())]
            if wrong:
                _error(errors, "CANDIDATE_NOT_REFERENCE", "predictions", f"{len(wrong)} candidate(s) do not belong to publication references")

    card = dataset_dir / "README.md"
    card_text = card.read_text(encoding="utf-8", errors="replace") if card.is_file() else ""
    if not card.is_file():
        _error(errors, "DATASET_CARD_MISSING", "*", "README.md is missing")
    else:
        for table in TABLES:
            if table not in card_text:
                _error(errors, "DATASET_CARD_CONFIG_MISSING", "README.md", f"Card missing '{table}' config")
            if f"{table}/*.parquet" not in card_text:
                _error(errors, "DATASET_CARD_DATA_FILES_MISSING", "README.md", f"Card missing data_files for '{table}'")
        local, token = LOCAL_PATH_PATTERN.search(card_text) is not None, any(pattern.search(card_text) for pattern in SENSITIVE_PATTERNS)
        if local:
            _error(errors, "LOCAL_PATH_FOUND", "README.md", "Card contains a local path")
        if token:
            _error(errors, "TOKEN_FOUND", "README.md", "Card contains a token or secret")

    build_summary = dataset_dir / "build_summary.json"
    if build_summary.is_file():
        summary_text = build_summary.read_text(encoding="utf-8")
        if LOCAL_PATH_PATTERN.search(summary_text):
            _error(errors, "LOCAL_PATH_FOUND", "build_summary.json", "Summary contains a local path")
    else:
        warnings.append({"code": "BUILD_SUMMARY_MISSING", "description": "build_summary.json is missing"})

    extra_names = {path.name for path in dataset_dir.iterdir()} if dataset_dir.is_dir() else set()
    for forbidden in ("model.bin", ".env", "labels.csv", "manual_labels.csv"):
        if forbidden in extra_names:
            _error(errors, "FORBIDDEN_FILE", "*", f"Forbidden file found: {forbidden}")

    for table in TABLES:
        checks[table] = {"pass": not any(error["table"] in {table, "*"} for error in errors)}
    passed = not errors and (not strict or not warnings)
    report: ValidationReport = {
        "member_id": member_id,
        "schema_version": SCHEMA_VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "row_counts": row_counts,
        "range_summary": {
            "range_paper_count": range_paper_count if "range_paper_count" in locals() else 0,
            "dataset_paper_count": len(paper_ids) if "paper_ids" in locals() else 0,
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a member Parquet dataset")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--member-id")
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=1000)
    args = parser.parse_args()
    if bool(args.member_id) != bool(args.range_manifest):
        parser.error("--member-id and --range-manifest must be provided together")
    report = validate_dataset(
        args.dataset_dir,
        args.member_id,
        args.range_manifest,
        args.output or args.dataset_dir / "member_dataset_validation.json",
        args.strict,
        args.sample_rows,
    )
    print(
        f"Validation {'passed' if report['passed'] else 'failed'}: "
        f"{len(report['errors'])} error(s), {len(report['warnings'])} warning(s)"
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()