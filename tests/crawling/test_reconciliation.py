import json

from src.crawling.reconciliation import audit_version_coverage


def test_audit_version_coverage_reports_missing_and_terminal_versions(tmp_path):
    complete = tmp_path / "2401.00001"
    complete.mkdir()
    (complete / "metadata.json").write_text(
        json.dumps({"versions": ["v1", "v2"], "revised_dates": ["2024-01-02"]}),
        encoding="utf-8",
    )
    (complete / "v1").mkdir()
    (complete / "v1" / "source_status.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    (complete / "v2").mkdir()
    (complete / "v2" / "source_status.json").write_text(
        json.dumps({"status": "no_tex"}), encoding="utf-8"
    )

    incomplete = tmp_path / "2401.00002"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text(
        json.dumps({"versions": ["v1", "v2", "v3"]}), encoding="utf-8"
    )
    (incomplete / "v1").mkdir()
    (incomplete / "v1" / "source_status.json").write_text(
        json.dumps({"status": "extract_error"}), encoding="utf-8"
    )
    (incomplete / "v2").mkdir()

    report = audit_version_coverage(tmp_path)

    assert report["publications"] == 2
    assert report["expected_versions"] == 5
    assert report["terminal_versions"] == 2
    assert report["missing_version_records"] == 2
    assert report["retryable_or_failed_versions"] == 1
    assert report["records"]["2401.00002"]["missing_versions"] == ["v2", "v3"]
