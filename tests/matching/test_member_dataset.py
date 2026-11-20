import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.matching.member_dataset import TABLES, build_member_dataset


def _manifest(path: Path) -> None:
    path.write_text(
        "member_id,member_name,range_id,start_arxiv_id,end_arxiv_id\n"
        "24127268,Test Member,range-04,2604.25535,2605.02337\n",
        encoding="utf-8",
    )


def _write_table(root: Path, table: str, rows: list[dict]) -> None:
    if table == "papers":
        schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("title", pa.string())])
    elif table == "references":
        schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("ref_arxiv_id", pa.string())])
    elif table == "elements":
        schema = pa.schema([pa.field("element_id", pa.string()), pa.field("arxiv_id", pa.string()), pa.field("kind", pa.string()), pa.field("content", pa.string())])
    elif table == "hierarchy":
        schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("version", pa.int64()), pa.field("element_id", pa.string()), pa.field("parent_id", pa.string())])
    elif table == "bibtex":
        schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("citation_key", pa.string()), pa.field("entry_type", pa.string())])
    else:
        schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("partition", pa.string()), pa.field("citation_key", pa.string())])
    table_dir = root / table
    table_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), table_dir / "source.parquet")


@pytest.fixture
def dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "member_ranges.csv"
    _manifest(manifest)
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_table(input_root, "papers", [{"arxiv_id": "2604.25535", "title": "Inside"}, {"arxiv_id": "2605.02338", "title": "Outside"}])
    _write_table(input_root, "references", [{"arxiv_id": "2604.25535", "ref_arxiv_id": "1706.03762"}])
    _write_table(input_root, "elements", [{"element_id": "e1", "arxiv_id": "2604.25535", "kind": "sentence", "content": "text"}])
    _write_table(input_root, "hierarchy", [{"arxiv_id": "2604.25535", "version": 1, "element_id": "e1", "parent_id": "root"}])
    _write_table(input_root, "bibtex", [{"arxiv_id": "2604.25535", "citation_key": "key", "entry_type": "article"}])
    _write_table(input_root, "predictions", [{"arxiv_id": "2604.25535", "partition": "train", "citation_key": "key"}])
    return manifest, input_root


def test_build_filters_all_tables_and_is_deterministic(tmp_path: Path, dataset_fixture):
    manifest, input_root = dataset_fixture
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary1 = build_member_dataset("24127268", manifest, input_root, first, overwrite=True)
    summary2 = build_member_dataset("24127268", manifest, input_root, second, overwrite=True)
    assert summary1["row_counts"] == summary2["row_counts"]
    for table in TABLES:
        shards = sorted((first / "24127268" / table).glob("*.parquet"))
        assert shards
        for shard in shards:
            pq.ParquetFile(shard).read()


def test_references_keep_out_of_range_ref_id(tmp_path: Path, dataset_fixture):
    manifest, input_root = dataset_fixture
    output = tmp_path / "output"
    build_member_dataset("24127268", manifest, input_root, output, overwrite=True)
    rows = pq.read_table(output / "24127268" / "references" / "part-00000.parquet").to_pylist()
    assert rows == [{"arxiv_id": "2604.25535", "ref_arxiv_id": "1706.03762", "ref_title": None, "ref_authors": None, "ref_submission_date": None, "ref_s2_id": None}]


def test_empty_table_has_contract_schema(tmp_path: Path, dataset_fixture):
    manifest, input_root = dataset_fixture
    output = tmp_path / "output"
    build_member_dataset("24127268", manifest, input_root, output, overwrite=True)
    pq.write_table(pq.read_table(output / "24127268" / "elements" / "part-00000.parquet"), output / "unused.parquet")


def test_unknown_member_raises(tmp_path: Path, dataset_fixture):
    manifest, input_root = dataset_fixture
    with pytest.raises(ValueError, match="not found"):
        build_member_dataset("99999999", manifest, input_root, tmp_path / "out")
