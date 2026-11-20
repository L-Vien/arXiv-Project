from pathlib import Path

import pytest

from src.matching.publishing import check_card, generate_card, upload_dataset, verify_public_repo


def test_card_has_all_configs_and_no_sensitive_text(tmp_path: Path):
    card = generate_card(tmp_path, "24217550", "2604.25535 to 2605.02337")
    text = card.read_text(encoding="utf-8")
    assert check_card(tmp_path) == []
    assert "data_files: predictions/*.parquet" in text
    assert "/Users/" not in text
    assert "hf_" not in text


def test_dry_run_lists_files_without_api(tmp_path: Path):
    generate_card(tmp_path, "24217550", "range")
    validation = tmp_path / "member_dataset_validation.json"
    validation.write_text('{"passed": true}', encoding="utf-8")
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "part-00000.parquet").write_bytes(b"x")
    url = upload_dataset(tmp_path, "alice/dataset", validation, dry_run=True)
    assert url.endswith("alice/dataset")


def test_validation_fail_blocks_upload(tmp_path: Path):
    generate_card(tmp_path, "24217550", "range")
    validation = tmp_path / "member_dataset_validation.json"
    validation.write_text('{"passed": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed"):
        upload_dataset(tmp_path, "alice/dataset", validation)


def test_verify_private_repo(tmp_path: Path):
    class FakeApi:
        def repo_info(self, repo_id: str, repo_type: str):
            class Info:
                private = True
            return Info()
    result = verify_public_repo("alice/dataset", api=FakeApi(), publish_report=tmp_path / "publish_report.json")
    assert result["public"] is False
    assert result["passed"] is False
