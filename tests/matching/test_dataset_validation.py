import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.matching.dataset_validation import validate_dataset
from src.matching.member_dataset import SCHEMAS


def _valid_dataset(dataset: Path) -> None:
    rows = {
        "papers": [{"arxiv_id": "2604.25535", "title": "Paper", "authors": ["A"], "submission_date": None, "revised_dates": None, "venue": None}],
        "references": [{"arxiv_id": "2604.25535", "ref_arxiv_id": "1706.03762", "ref_title": "Ref", "ref_authors": None, "ref_submission_date": None, "ref_s2_id": None}],
        "elements": [{"element_id": "e1", "arxiv_id": "2604.25535", "kind": "sentence", "content": "text"}],
        "hierarchy": [{"arxiv_id": "2604.25535", "version": 1, "element_id": "e1", "parent_id": "root"}],
        "bibtex": [{"arxiv_id": "2604.25535", "citation_key": "key", "entry_type": "article", "title": "Title", "authors": None, "year": 2026, "doi": None, "journal": None, "url": None}],
        "predictions": [{"arxiv_id": "2604.25535", "partition": "train", "citation_key": "key", "groundtruth_arxiv_id": "1706.03762", "cand_1": "1706.03762", "cand_2": None, "cand_3": None, "cand_4": None, "cand_5": None}],
    }
    for table, table_rows in rows.items():
        table_dir = dataset / table
        table_dir.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(table_rows, schema=SCHEMAS[table]), table_dir / "part-00000.parquet")
    (dataset / "README.md").write_text(
        "---\nconfigs:\n" + "".join(f"  - config_name: {table}\n    data_files: {table}/*.parquet\n" for table in SCHEMAS),
        encoding="utf-8",
    )


def test_valid_dataset_passes(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _valid_dataset(dataset)
    report = validate_dataset(dataset)
    assert report["passed"] is True
    assert report["row_counts"]["papers"]["rows"] == 1


def test_duplicate_candidate_is_reported(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _valid_dataset(dataset)
    path = dataset / "predictions" / "part-00000.parquet"
    table = pq.read_table(path)
    row = table.to_pylist()[0]
    row["cand_2"] = "1706.03762"
    pq.write_table(pa.Table.from_pylist([row], schema=SCHEMAS["predictions"]), path)
    report = validate_dataset(dataset)
    assert report["passed"] is False
    assert any(error["code"] == "DUPLICATE_CANDIDATE" for error in report["errors"])


def test_missing_prediction_is_reported(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _valid_dataset(dataset)
    (dataset / "predictions" / "part-00000.parquet").unlink()
    report = validate_dataset(dataset)
    assert report["passed"] is False
