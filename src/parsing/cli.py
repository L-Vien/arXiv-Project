"""Command-line adapter for running Milestone 2 over an M1 corpus.

M1 discovers and downloads papers.  This CLI only selects already downloaded
publication directories and delegates parsing to :func:`run_pipeline`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.crawling.logger import get_logger
from src.crawling.metadata import base_arxiv_id
from src.parsing.pipeline import (
    PublicationInput,
    VersionInput,
    current_dependencies,
    run_pipeline,
)

_VERSION_PATTERN = re.compile(r"^v(\d+)$")
_NEW_ARXIV_ID_PATTERN = re.compile(r"^(\d{4})\.(\d{4,5})$")
_APP_LOGGER_NAME = "arxiv_project"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = get_logger("parsing.cli")


@dataclass(frozen=True)
class ArxivRange:
    """Inclusive numeric arXiv range used to filter local M1 data."""

    start: tuple[int, int]
    end: tuple[int, int]

    def contains(self, arxiv_id: str) -> bool:
        match = _NEW_ARXIV_ID_PATTERN.fullmatch(arxiv_id)
        if match is None:
            return False
        yymm = match.group(1)
        year, month = 2000 + int(yymm[:2]), int(yymm[2:])
        key = (year * 12 + month, int(match.group(2)))
        return self.start <= key <= self.end


def _parse_month(raw: str) -> tuple[int, int]:
    value = str(raw).strip()
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
    else:
        match = re.fullmatch(r"(\d{2})(\d{2})", value)
        if match is None:
            raise ValueError(
                f"invalid month {raw!r}; expected YYYY-MM or YYMM"
            )
        year, month = 2000 + int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 01 and 12: {raw!r}")
    return year, month


def _parse_range(
    start_month: str,
    start_id: int,
    end_month: str,
    end_id: int,
) -> ArxivRange:
    start_year, start_month_number = _parse_month(start_month)
    end_year, end_month_number = _parse_month(end_month)
    if not 1 <= start_id <= 99999 or not 1 <= end_id <= 99999:
        raise ValueError("range IDs must be between 1 and 99999")

    start = (start_year * 12 + start_month_number, start_id)
    end = (end_year * 12 + end_month_number, end_id)
    if start > end:
        raise ValueError("range start must not be after range end")
    return ArxivRange(start=start, end=end)


def _version_inputs(paper_dir: Path) -> tuple[VersionInput, ...]:
    versions: list[VersionInput] = []
    for version_dir in paper_dir.iterdir():
        if not version_dir.is_dir():
            continue
        match = _VERSION_PATTERN.fullmatch(version_dir.name)
        if match is None:
            continue
        versions.append(
            VersionInput(
                version=int(match.group(1)),
                source_dir=version_dir,
            )
        )
    return tuple(sorted(versions, key=lambda item: item.version))


def load_publications(
    corpus_dir: Path,
    *,
    arxiv_range: ArxivRange | None = None,
    limit: int | None = None,
) -> tuple[PublicationInput, ...]:
    """Build publication inputs from local M1 ``metadata.json`` directories."""

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus directory does not exist: {corpus_dir}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    publications: list[PublicationInput] = []
    for paper_dir in sorted(corpus_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        metadata_path = paper_dir / "metadata.json"
        if not metadata_path.is_file():
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            arxiv_id = base_arxiv_id(metadata["arxiv_id"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid metadata: {metadata_path}: {exc}") from exc

        if arxiv_range is not None and not arxiv_range.contains(arxiv_id):
            continue

        versions = _version_inputs(paper_dir)
        if not versions:
            # M1 keeps PDF-only or otherwise source-less publications as an
            # empty record.  Preserve them in M2 instead of aborting a full
            # corpus run; TV2 will return EMPTY_NO_TEX for this directory.
            versions = (
                VersionInput(
                    version=1,
                    source_dir=paper_dir,
                ),
            )
        publications.append(
            PublicationInput(arxiv_id=arxiv_id, versions=versions)
        )

    publications.sort(key=lambda item: item.arxiv_id)
    if limit is not None:
        publications = publications[:limit]
    return tuple(publications)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Milestone 2 parsing over an existing M1 corpus"
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="directory for M2 logs; defaults to <output-dir>/logs",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="process all local publications",
    )
    selection.add_argument(
        "--run-all",
        dest="all",
        action="store_true",
        help="alias for --all",
    )
    selection.add_argument("--start-month")

    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-month")
    parser.add_argument("--end-id", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="process at most this many selected publications",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected publications without parsing",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="seconds between terminal/log progress snapshots (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable periodic snapshots while retaining event logs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="publication worker processes; start with 2 or 3 for large runs",
    )
    return parser


def _configure_logging(log_dir: Path) -> None:
    """Configure console and rotating file logs for a standalone M2 run."""

    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    for handler in app_logger.handlers[:]:
        if getattr(handler, "_m2_log_handler", False):
            app_logger.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / "m2.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler._m2_log_handler = True
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in app_logger.handlers
    ):
        app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    app_logger.info("M2 logging initialized: log_dir=%s", log_dir)


def _select_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[tuple[PublicationInput, ...], str]:
    range_values = (
        args.start_month,
        args.start_id,
        args.end_month,
        args.end_id,
    )
    has_range = any(value is not None for value in range_values)
    if args.all and has_range:
        parser.error("--all cannot be combined with range arguments")
    if not args.all and not has_range:
        parser.error("choose --all or provide all four range arguments")
    if has_range and not all(value is not None for value in range_values):
        parser.error(
            "a range requires --start-month, --start-id, --end-month, and --end-id"
        )

    try:
        arxiv_range = (
            _parse_range(
                args.start_month,
                args.start_id,
                args.end_month,
                args.end_id,
            )
            if has_range
            else None
        )
        publications = load_publications(
            args.corpus_dir,
            arxiv_range=arxiv_range,
            limit=args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return publications, "range" if has_range else "all"


def _print_selection(
    publications: Sequence[PublicationInput],
    mode: str,
) -> None:
    version_count = sum(len(publication.versions) for publication in publications)
    print(f"selection={mode}")
    print(f"publications={len(publications)}")
    print(f"versions={version_count}")
    for publication in publications[:10]:
        versions = ",".join(str(item.version) for item in publication.versions)
        print(f"{publication.arxiv_id}: v{versions}")
    if len(publications) > 10:
        print(f"... and {len(publications) - 10} more publications")


def _section(value: object, name: str) -> Mapping[str, object]:
    section = value.get(name) if isinstance(value, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _decimal(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _ratio_text(value: object) -> str:
    ratio = _decimal(value)
    return "n/a" if ratio is None else f"{ratio:.3f} ({ratio:.1%})"


def _megabytes(value: object) -> str:
    number = _decimal(value)
    return "n/a" if number is None else f"{number / (1024 * 1024):.2f} MB"


def _print_terminal_summary(
    *,
    statistics: Mapping[str, object],
    benchmark: Mapping[str, object],
    output_dir: Path,
    log_dir: Path,
    selection_mode: str,
) -> None:
    """Print the report-facing M2 metrics after a completed run."""

    publications = _section(statistics, "publications")
    versions = _section(statistics, "versions")
    elements = _section(statistics, "elements")
    references = _section(statistics, "references")
    validation = _section(statistics, "validation")
    timing = _section(statistics, "timing")
    memory = _section(benchmark, "memory")
    benchmark_output = _section(benchmark, "output")

    failed = _integer(publications.get("failed"))
    partial = _integer(publications.get("partial"))
    warnings = _integer(validation.get("warnings"))
    if failed:
        status = "completed_with_failures"
    elif partial or warnings:
        status = "completed_with_warnings"
    else:
        status = "completed"

    print("\nM2 pipeline summary")
    print(f"status={status}")
    print(
        "publications="
        f"{_integer(publications.get('successful'))} passed / "
        f"{partial} partial / {failed} failed / "
        f"{_integer(publications.get('empty'))} empty / "
        f"{_integer(publications.get('total'))} total "
        f"success_rate={_ratio_text(publications.get('success_rate'))}"
    )
    print(
        "versions="
        f"{_integer(versions.get('processed'))}/"
        f"{_integer(versions.get('expected'))} processed "
        f"coverage={_ratio_text(versions.get('coverage_rate'))} "
        f"parse_success={_ratio_text(versions.get('parse_success_rate'))} "
        f"without_tex={_integer(versions.get('without_tex'))}"
    )
    print(
        "elements="
        f"before_dedup:{_integer(elements.get('before_dedup'))} "
        f"after_dedup:{_integer(elements.get('after_dedup'))} "
        f"ratio:{_ratio_text(elements.get('deduplication_ratio'))}"
    )
    print(
        "bibtex="
        f"before_dedup:{_integer(references.get('entries_before_dedup'))} "
        f"after_dedup:{_integer(references.get('entries_after_dedup'))} "
        f"basis:{references.get('before_dedup_basis', 'n/a')}"
    )
    print(
        "validation="
        f"errors:{_integer(validation.get('errors'))} "
        f"warnings:{warnings}"
    )
    print(
        "timing="
        f"wall:{_decimal(benchmark.get('wall_time_seconds')) or 0.0:.2f}s "
        f"publication_avg:{_decimal(timing.get('publication_seconds_average')) or 0.0:.3f}s "
        f"publication_total:{_decimal(timing.get('publication_seconds_total')) or 0.0:.3f}s"
    )
    print(
        "resources="
        f"peak_rss:{_megabytes(memory.get('peak_rss_bytes'))} "
        f"output_size:{_megabytes(benchmark_output.get('size_bytes'))}"
    )
    print(f"selection={selection_mode}")
    print(f"output_dir={output_dir}")
    print(f"statistics={output_dir / 'statistics.json'}")
    print(f"benchmark={output_dir / 'benchmark.json'}")
    print(f"validation_report={output_dir / 'validation_report.json'}")
    print(f"logs={log_dir / 'm2.log'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be greater than zero")
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")
    publications, mode = _select_from_args(parser, args)
    _print_selection(publications, mode)

    if args.dry_run:
        return 0

    log_dir = args.log_dir or args.output_dir / "logs"
    _configure_logging(log_dir)
    logger.info(
        "m2_selection mode=%s publications=%d versions=%d",
        mode,
        len(publications),
        sum(len(publication.versions) for publication in publications),
    )
    logger.info("m2_run_start output_dir=%s", args.output_dir)

    result = run_pipeline(
        publications=publications,
        dependencies=current_dependencies(),
        output_dir=args.output_dir,
        progress_enabled=not args.no_progress,
        progress_interval_seconds=args.progress_interval,
        workers=args.workers,
        input_description=(
            f"corpus={args.corpus_dir} selection={mode} "
            f"publications={len(publications)}"
        ),
    )
    failed = result.statistics["publications"]["failed"]
    logger.info(
        "m2_run_complete publications=%d failed=%d output_dir=%s",
        result.publication_count,
        failed,
        args.output_dir,
    )
    _print_terminal_summary(
        statistics=result.statistics,
        benchmark=result.benchmark,
        output_dir=args.output_dir,
        log_dir=log_dir,
        selection_mode=mode,
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArxivRange",
    "build_parser",
    "load_publications",
    "main",
]
