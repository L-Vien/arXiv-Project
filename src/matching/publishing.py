
"""Create a Hugging Face dataset card and publish a validated member dataset."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi


TABLES = (
    "papers",
    "references",
    "elements",
    "hierarchy",
    "bibtex",
    "predictions",
)

LOCAL_PATH_PATTERN = re.compile(
    r"(?i)(?:[A-Z]:\\Users\\|/Users/[^/]+/|/home/[^/]+/|/mnt/data/)"
)

TOKEN_PATTERN = re.compile(
    r"hf_[A-Za-z0-9]{20,}|api[_-]?key\s*[:=]\s*[^\s,;]+|"
    r"access[_-]?token\s*[:=]\s*[^\s,;]+|password\s*[:=]\s*[^\s,;]+|"
    r"secret\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)


def card_text(
    member_id: str,
    range_summary: str,
    schema_version: str = "1.0",
) -> str:
    return f"""---
configs:
  - config_name: papers
    data_files: papers/*.parquet
  - config_name: references
    data_files: references/*.parquet
  - config_name: elements
    data_files: elements/*.parquet
  - config_name: hierarchy
    data_files: hierarchy/*.parquet
  - config_name: bibtex
    data_files: bibtex/*.parquet
  - config_name: predictions
    data_files: predictions/*.parquet
---

# arXiv member dataset: {member_id}

This dataset contains only publications in the member range: {range_summary}.
The data comes from M1 and M2 sources. Schema version: `{schema_version}`.

## Tables

- `papers`: one row per citing publication (`arxiv_id`, metadata, authors).
- `references`: citing publication and reference metadata; `arxiv_id` is the citing paper.
- `elements`: parsed publication content elements.
- `hierarchy`: parent-child relationships for parsed elements.
- `bibtex`: BibTeX entries keyed by `(arxiv_id, citation_key)`.
- `predictions`: one prediction row per BibTeX entry.

## Predictions

`partition` is `train`, `valid`, or `test`. `cand_1` through `cand_5` are ranked
candidate arXiv IDs for the same publication. Null candidates mean no candidate
at that rank. Candidates are contiguous and never use sentinel strings.
`groundtruth_arxiv_id` is nullable.

## Limitations

This dataset is limited to one member range. Automatically generated predictions
and reference matches may contain errors.
"""


def generate_card(
    dataset_dir: Path,
    member_id: str,
    range_summary: str,
    schema_version: str = "1.0",
) -> Path:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    text = card_text(member_id, range_summary, schema_version)

    if LOCAL_PATH_PATTERN.search(text) or TOKEN_PATTERN.search(text):
        raise ValueError(
            "Generated dataset card contains a local path or secret"
        )

    path = dataset_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def check_card(dataset_dir: Path) -> list[str]:
    card = dataset_dir / "README.md"

    if not card.is_file():
        return ["README.md is missing"]

    text = card.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []

    for table in TABLES:
        if table not in text:
            errors.append(f"Dataset card missing {table}")

        if f"{table}/*.parquet" not in text:
            errors.append(
                f"Dataset card missing data_files for {table}"
            )

    if LOCAL_PATH_PATTERN.search(text):
        errors.append("Dataset card contains a local path")

    if TOKEN_PATTERN.search(text):
        errors.append("Dataset card contains a token or secret")

    return errors


def load_validation_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(report, dict):
        raise ValueError("Validation report must be a JSON object")

    return report


def validate_before_publish(
    validation_report: Path,
) -> dict[str, object]:
    report = load_validation_report(validation_report)

    if report.get("passed") is not True:
        raise ValueError(
            "Dataset validation has not passed; publishing was cancelled"
        )

    return report


def token_from_environment() -> str:
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise EnvironmentError("HF_TOKEN is not set")

    return token


def upload_dataset(
    dataset_dir: Path,
    repo_id: str,
    validation_report: Path,
    private: bool = True,
    dry_run: bool = False,
    commit_message: str = "Upload member dataset",
    token: str | None = None,
) -> str:
    validate_before_publish(validation_report)

    card_errors = check_card(dataset_dir)

    if card_errors:
        raise ValueError(
            "Invalid dataset card: " + "; ".join(card_errors)
        )

    files = sorted(
        path for path in dataset_dir.rglob("*") if path.is_file()
    )

    if dry_run:
        for path in files:
            print(
                f"{path.relative_to(dataset_dir)}\t"
                f"{path.stat().st_size} bytes"
            )

        return f"https://huggingface.co/datasets/{repo_id}"

    resolved_token = token or token_from_environment()

    api = HfApi(token=resolved_token)

    try:
        who = api.whoami(resolved_token)
        owner = repo_id.split("/", 1)[0]

        if who.get("name") != owner:
            raise PermissionError(
                f"Repository owner '{owner}' differs from "
                f"authenticated account '{who.get('name')}'"
            )

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(dataset_dir),
            commit_message=commit_message,
        )

    except Exception as exc:
        message = str(exc)

        if resolved_token in message:
            message = "[redacted]"

        raise RuntimeError(message) from exc

    return f"https://huggingface.co/datasets/{repo_id}"


def verify_public_repo(
    repo_id: str,
    api: HfApi | None = None,
    publish_report: Path | None = None,
) -> dict[str, object]:

    hub_api = api if api is not None else HfApi()

    try:
        info = hub_api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
        )

        is_public = getattr(info, "private", True) is False

    except Exception as exc:
        result: dict[str, object] = {
            "repo_id": repo_id,
            "public": False,
            "configs": {},
            "passed": False,
            "error": type(exc).__name__,
        }

        if publish_report is not None:
            publish_report.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            publish_report.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )

        return result

    # Check whether the dataset repository can be accessed.
    # This does not independently verify each config/table.
    try:
        hub_api.dataset_info(repo_id)
        config_results = {
            table: True for table in TABLES
        }

    except Exception:
        config_results = {
            table: False for table in TABLES
        }

    result = {
        "repo_id": repo_id,
        "public": is_public,
        "configs": config_results,
        "passed": is_public and all(config_results.values()),
    }

    if publish_report is not None:
        publish_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        publish_report.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a card or publish a member dataset"
    )

    parser.add_argument(
        "command",
        choices=("card", "upload", "publish", "verify"),
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--validation-report")
    parser.add_argument("--member-id")
    parser.add_argument("--range-summary", default="member range")
    parser.add_argument("--schema-version", default="1.0")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Upload member dataset",
    )

    args = parser.parse_args()

    try:
        if args.command == "card":
            member_id = args.member_id or "unknown"

            path = generate_card(
                args.dataset_dir,
                member_id,
                args.range_summary,
                args.schema_version,
            )

            print(path)
            return

        if args.command == "verify":
            if not args.repo_id:
                parser.error("--repo-id is required for verify")

            report = verify_public_repo(
                args.repo_id,
                publish_report=args.dataset_dir / "publish_report.json",
            )

            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["passed"] else 1)

        if not args.repo_id or not args.validation_report:
            parser.error(
                "--repo-id and --validation-report are required"
            )

        url = upload_dataset(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            validation_report=Path(args.validation_report),
            private=args.private,
            dry_run=args.dry_run,
            commit_message=args.commit_message,
        )

        print(url)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
