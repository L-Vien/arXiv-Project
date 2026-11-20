"""Audit whether an M1 corpus contains an explicit record for every version.

This does not query arXiv.  It verifies the local corpus against each paper's
latest discovery metadata, which makes interrupted or silently missing version
downloads visible before M2 is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TERMINAL_SOURCE_STATUSES = {"ok", "no_tex", "missing"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _expected_versions(metadata: dict[str, Any]) -> list[str]:
    versions = metadata.get("versions")
    if isinstance(versions, list):
        cleaned = [str(version) for version in versions if str(version).startswith("v")]
        if cleaned:
            return cleaned
    revised_dates = metadata.get("revised_dates")
    count = len(revised_dates) + 1 if isinstance(revised_dates, list) else 1
    return [f"v{number}" for number in range(1, count + 1)]


def audit_version_coverage(corpus_dir: str | Path) -> dict[str, Any]:
    """Return a deterministic coverage report for all publications in ``corpus_dir``."""
    root = Path(corpus_dir)
    records: dict[str, dict[str, Any]] = {}
    totals = {
        "publications": 0,
        "expected_versions": 0,
        "terminal_versions": 0,
        "missing_version_records": 0,
        "retryable_or_failed_versions": 0,
        "invalid_metadata": 0,
    }

    for publication_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = _read_json(publication_dir / "metadata.json")
        if not metadata:
            totals["invalid_metadata"] += 1
            continue
        expected = _expected_versions(metadata)
        missing: list[str] = []
        terminal: list[str] = []
        failed: list[str] = []
        for version in expected:
            status_path = publication_dir / version / "source_status.json"
            status = _read_json(status_path).get("status")
            if not isinstance(status, str):
                missing.append(version)
            elif status in TERMINAL_SOURCE_STATUSES:
                terminal.append(version)
            else:
                failed.append(version)

        records[publication_dir.name] = {
            "expected_versions": expected,
            "terminal_versions": terminal,
            "missing_versions": missing,
            "retryable_or_failed_versions": failed,
        }
        totals["publications"] += 1
        totals["expected_versions"] += len(expected)
        totals["terminal_versions"] += len(terminal)
        totals["missing_version_records"] += len(missing)
        totals["retryable_or_failed_versions"] += len(failed)

    return {**totals, "records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit M1 version coverage")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_version_coverage(args.corpus_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["missing_version_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
