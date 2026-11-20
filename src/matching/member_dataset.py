
"""Build a range-scoped member dataset from final M1/M2/M3 Parquet tables."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .contracts import SCHEMA_VERSION, normalize_arxiv_id


TABLES = (
    "papers",
    "references",
    "elements",
    "hierarchy",
    "bibtex",
    "predictions",
)


SCHEMAS: dict[str, pa.Schema] = {
    "papers": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=True),
        pa.field("authors", pa.list_(pa.string()), nullable=True),
        pa.field("submission_date", pa.string(), nullable=True),
        pa.field("revised_dates", pa.list_(pa.string()), nullable=True),
        pa.field("venue", pa.string(), nullable=True),
    ]),
    "references": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("ref_arxiv_id", pa.string(), nullable=True),
        pa.field("ref_title", pa.string(), nullable=True),
        pa.field("ref_authors", pa.list_(pa.string()), nullable=True),
        pa.field("ref_submission_date", pa.string(), nullable=True),
        pa.field("ref_s2_id", pa.string(), nullable=True),
    ]),
    "elements": pa.schema([
        pa.field("element_id", pa.string(), nullable=False),
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]),
    "hierarchy": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("version", pa.int64(), nullable=False),
        pa.field("element_id", pa.string(), nullable=False),
        pa.field("parent_id", pa.string(), nullable=False),
    ]),
    "bibtex": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("entry_type", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=True),
        pa.field("authors", pa.list_(pa.string()), nullable=True),
        pa.field("year", pa.int64(), nullable=True),
        pa.field("doi", pa.string(), nullable=True),
        pa.field("journal", pa.string(), nullable=True),
        pa.field("url", pa.string(), nullable=True),
    ]),
    "predictions": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("groundtruth_arxiv_id", pa.string(), nullable=True),
        pa.field("cand_1", pa.string(), nullable=True),
        pa.field("cand_2", pa.string(), nullable=True),
        pa.field("cand_3", pa.string(), nullable=True),
        pa.field("cand_4", pa.string(), nullable=True),
        pa.field("cand_5", pa.string(), nullable=True),
    ]),
}


KEYS = {
    "papers": ("arxiv_id",),
    "references": ("arxiv_id", "ref_arxiv_id"),
    "elements": ("arxiv_id", "element_id"),
    "hierarchy": ("arxiv_id", "version", "element_id", "parent_id"),
    "bibtex": ("arxiv_id", "citation_key"),
    "predictions": ("arxiv_id", "citation_key"),
}


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_member_range(row: Mapping[str, str]) -> dict[str, object]:
    start = normalize_arxiv_id(row.get("start_arxiv_id"))
    end = normalize_arxiv_id(row.get("end_arxiv_id"))

    if start is None or end is None:
        raise ValueError(
            "Range manifest row has start_arxiv_id or end_arxiv_id missing"
        )

    return {
        "start": start,
        "end": end,
        "range_id": row.get("range_id"),
        "member_name": row.get("member_name"),
    }


def load_member_ids(
    range_manifest: Path,
    member_id: str,
    paper_id_list: Path | None = None,
) -> set[str]:
    """Load the member's official paper IDs, falling back to the manifest numeric range."""
    rows = _read_manifest_rows(range_manifest)

    member_rows = [
        row for row in rows
        if str(row.get("member_id", "")).strip() == str(member_id)
    ]

    if not member_rows:
        raise ValueError(
            f"member_id '{member_id}' was not found in range manifest: {range_manifest}"
        )

    if paper_id_list is not None:
        ids = {
            normalize_arxiv_id(line.strip())
            for line in paper_id_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        ids.discard(None)
        if not ids:
            raise ValueError(f"Paper ID list is empty: {paper_id_list}")
        return set(ids)  # type: ignore[return-value]

    duplicates: dict[str, str] = {}

    for row in rows:
        ids_for_row: set[str] = set()
        start_end = row.get("paper_ids")

        if start_end:
            for raw in str(start_end).split("|"):
                normalized = normalize_arxiv_id(raw.strip())
                if normalized:
                    ids_for_row.add(normalized)

        if ids_for_row:
            for paper_id in ids_for_row:
                if (
                    paper_id in duplicates
                    and duplicates[paper_id] != str(row["member_id"])
                ):
                    raise ValueError(
                        f"Paper ID {paper_id} is assigned to multiple members"
                    )

                duplicates[paper_id] = str(row["member_id"])

    return set(detect_range_ids(rows, member_id))


def detect_range_ids(
    rows: Sequence[Mapping[str, str]],
    member_id: str,
) -> list[str]:
    member_row = next(
        (
            row for row in rows
            if str(row.get("member_id", "")) == str(member_id)
        ),
        None,
    )

    if member_row is None:
        return []

    if member_row.get("paper_ids"):
        result = []

        for raw in str(member_row["paper_ids"]).split("|"):
            normalized = normalize_arxiv_id(raw.strip())
            if normalized:
                result.append(normalized)

        return sorted(set(result))

    start = normalize_arxiv_id(member_row.get("start_arxiv_id"))
    end = normalize_arxiv_id(member_row.get("end_arxiv_id"))

    if start is None or end is None:
        raise ValueError("Manifest range is missing start/end arXiv IDs")

    return []


def arxiv_range_key(arxiv_id: str) -> tuple[str, int]:
    normalized = normalize_arxiv_id(arxiv_id)

    if normalized is None:
        raise ValueError(f"Invalid arXiv ID: {arxiv_id!r}")

    if "." not in normalized:
        return ("legacy", 0)

    month, number = normalized.split(".", 1)
    return month, int(number)


def ids_in_manifest_range(
    arxiv_id: str,
    row: Mapping[str, str],
) -> bool:
    normalized = normalize_arxiv_id(arxiv_id)
    start = normalize_arxiv_id(row.get("start_arxiv_id"))
    end = normalize_arxiv_id(row.get("end_arxiv_id"))

    if normalized is None or start is None or end is None:
        return False

    try:
        return (
            arxiv_range_key(start)
            <= arxiv_range_key(normalized)
            <= arxiv_range_key(end)
        )
    except ValueError:
        return normalized == start or normalized == end


def normalize_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    try:
        return normalize_arxiv_id(value)
    except ValueError:
        return None


def is_in_ids(value: str, allowed: set[str]) -> bool:
    return value in allowed


def _find_source(input_root: Path, table: str) -> list[Path]:
    candidates = [
        input_root / f"{table}.parquet",
        input_root / table / "part",
        input_root / table,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return [candidate]

        if candidate.is_dir():
            files = sorted(candidate.rglob("*.parquet"))
            if files:
                return files

    jsonl = input_root / f"{table}.jsonl"

    if jsonl.is_file():
        return [jsonl]

    return []


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _iter_batches(
    paths: Sequence[Path],
    batch_size: int = 100_000,
) -> Iterator[pa.RecordBatch]:
    for path in paths:
        if path.suffix == ".jsonl":
            rows: list[dict[str, object]] = []

            for row in _iter_jsonl_rows(path):
                rows.append(row)

                if len(rows) >= batch_size:
                    yield pa.RecordBatch.from_pylist(rows)
                    rows = []

            if rows:
                yield pa.RecordBatch.from_pylist(rows)

        elif path.suffix == ".csv":
            import pandas as pd

            frame = pd.read_csv(path, chunksize=batch_size)

            for chunk in frame:
                yield pa.RecordBatch.from_pandas(
                    chunk,
                    preserve_index=False,
                )

        else:
            parquet = pq.ParquetFile(path)

            for batch in parquet.iter_batches(batch_size=batch_size):
                yield batch


def _canonicalize_batch(
    table: str,
    batch: pa.RecordBatch,
) -> pa.RecordBatch:
    columns: dict[str, pa.Array] = {}
    schema = SCHEMAS[table]

    for field in schema:
        if field.name not in batch.schema.names:
            columns[field.name] = pa.nulls(
                batch.num_rows,
                type=field.type,
            )
            continue

        array = batch.column(
            batch.schema.get_field_index(field.name)
        )

        if field.name == "arxiv_id":
            values = [
                normalize_arxiv_id(value.as_py())
                if value.is_valid else None
                for value in array
            ]
            columns[field.name] = pa.array(
                values,
                type=pa.string(),
            )

        elif pa.types.is_list(field.type):
            values = []

            for value in array:
                py = value.as_py()

                if py is None:
                    values.append(None)
                elif isinstance(py, list):
                    values.append([
                        str(item) if item is not None else None
                        for item in py
                    ])
                else:
                    values.append([str(py)])

            columns[field.name] = pa.array(
                values,
                type=field.type,
            )

        else:
            columns[field.name] = pa.array(
                array.to_pylist(),
                type=field.type,
            )

    return pa.RecordBatch.from_arrays(
        [columns[field.name] for field in schema],
        schema=schema,
    )


def _filter_batch(
    table: str,
    batch: pa.RecordBatch,
    allowed_ids: set[str] | None,
    member_row: Mapping[str, str],
) -> pa.RecordBatch:
    canonical = _canonicalize_batch(table, batch)

    arxiv = canonical.column(
        canonical.schema.get_field_index("arxiv_id")
    )

    keep: list[bool] = []

    for value in arxiv:
        normalized = normalize_id(value.as_py())

        if normalized is None:
            keep.append(False)
        elif allowed_ids is not None:
            keep.append(normalized in allowed_ids)
        else:
            keep.append(
                ids_in_manifest_range(normalized, member_row)
            )

    if not any(keep):
        empty_arrays = [
            pa.array([], type=field.type)
            for field in canonical.schema
        ]

        return pa.RecordBatch.from_arrays(
            empty_arrays,
            schema=canonical.schema,
        )

    indices = pa.array([
        index
        for index, included in enumerate(keep)
        if included
    ])

    return pc.take(canonical, indices)


def _sort_key(
    table: str,
    batch: pa.RecordBatch,
) -> list[tuple[str, ...]]:
    fields = KEYS[table]
    indexes = [
        batch.schema.get_field_index(field)
        for field in fields
    ]

    values: list[tuple[str, ...]] = []

    for row_index in range(batch.num_rows):
        values.append(
            tuple(
                str(batch.column(index)[row_index].as_py())
                for index in indexes
            )
        )

    return values


def _write_shards(
    table: str,
    batches: Iterable[pa.RecordBatch],
    output_dir: Path,
    shard_size_rows: int,
) -> tuple[int, list[str], int]:
    table_dir = output_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)

    for old in table_dir.glob("*.parquet"):
        old.unlink()

    writer: pq.ParquetWriter | None = None
    shard_index = 0
    rows_written = 0
    shard_names: list[str] = []
    pending: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal writer, shard_index, rows_written

        if not pending:
            return

        table_data = pa.Table.from_pylist(
            pending,
            schema=SCHEMAS[table],
        )

        if writer is not None:
            writer.close()

        name = f"part-{shard_index:05d}.parquet"
        shard_index += 1

        shard_names.append(f"{table}/{name}")

        new_writer = pq.ParquetWriter(
            table_dir / name,
            SCHEMAS[table],
            compression="snappy",
        )

        writer = new_writer
        new_writer.write_table(table_data)

        rows_written += table_data.num_rows
        pending.clear()

    for batch in batches:
        py_rows = batch.to_pylist()

        py_rows.sort(
            key=lambda row: tuple(
                str(row.get(field))
                for field in KEYS[table]
            )
        )

        for row in py_rows:
            pending.append(row)

            if len(pending) >= shard_size_rows:
                flush()

    flush()

    if writer is not None:
        writer.close()

    if shard_index == 0:
        name = "part-00000.parquet"

        pq.write_table(
            pa.Table.from_batches(
                [],
                schema=SCHEMAS[table],
            ),
            table_dir / name,
            compression="snappy",
        )

        shard_names.append(f"{table}/{name}")

    return rows_written, shard_names, len(shard_names)


def build_member_dataset(
    member_id: str,
    range_manifest: Path,
    input_root: Path,
    output_root: Path,
    shard_size_rows: int = 100_000,
    overwrite: bool = False,
    paper_id_list: Path | None = None,
) -> dict[str, object]:
    """Build and summarize one member's six-table dataset."""
    if shard_size_rows < 1:
        raise ValueError("--shard-size-rows must be at least 1")

    rows = _read_manifest_rows(range_manifest)

    member_row = next(
        (
            row for row in rows
            if str(row.get("member_id", "")).strip() == str(member_id)
        ),
        None,
    )

    if member_row is None:
        raise ValueError(
            f"member_id '{member_id}' was not found in range manifest: {range_manifest}"
        )

    official_ids: set[str] | None = None

    if paper_id_list is not None:
        official_ids = load_member_ids(
            range_manifest,
            member_id,
            paper_id_list,
        )
    elif member_row.get("paper_ids"):
        official_ids = set(
            detect_range_ids(rows, member_id)
        )

    output_dir = output_root / str(member_id)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}; use --overwrite"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []

    for table in TABLES:
        if not _find_source(input_root, table):
            missing.append(table)

    if missing:
        raise FileNotFoundError(
            "Missing required source table(s): " + ", ".join(missing)
        )

    row_counts: dict[str, int] = {}
    shard_counts: dict[str, int] = {}
    sharded_files: dict[str, list[str]] = {}

    for table in TABLES:
        source_paths = _find_source(input_root, table)
        filtered: list[pa.RecordBatch] = []

        for batch in _iter_batches(source_paths):
            filtered_batch = _filter_batch(
                table,
                batch,
                official_ids,
                member_row,
            )

            if filtered_batch.num_rows:
                filtered.append(filtered_batch)

        count, files, shards = _write_shards(
            table,
            filtered,
            output_dir,
            shard_size_rows,
        )

        row_counts[table] = count
        shard_counts[table] = shards
        sharded_files[table] = files

    summary: dict[str, object] = {
        "member_id": str(member_id),
        "member_name": member_row.get("member_name"),
        "range": {
            "range_id": member_row.get("range_id"),
            "start_arxiv_id": member_row.get("start_arxiv_id"),
            "end_arxiv_id": member_row.get("end_arxiv_id"),
            "paper_ids": (
                sorted(official_ids)
                if official_ids is not None
                else None
            ),
        },
        "schema_version": SCHEMA_VERSION,
        "row_counts": row_counts,
        "shard_counts": shard_counts,
        "shards": sharded_files,
        "compression": "snappy",
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a range-scoped Parquet member dataset"
    )

    parser.add_argument("--member-id", required=True)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-size-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paper-id-list", type=Path)

    args = parser.parse_args()

    summary = build_member_dataset(
        member_id=args.member_id,
        range_manifest=args.range_manifest,
        input_root=args.input_root,
        output_root=args.output_root,
        shard_size_rows=args.shard_size_rows,
        overwrite=args.overwrite,
        paper_id_list=args.paper_id_list,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
