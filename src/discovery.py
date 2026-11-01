import arxiv
import json
import re
import argparse
import time
import random

from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Configuration
BATCH_SIZE = 200
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_CONSECUTIVE_NOT_FOUND_PER_MONTH = 200

client = arxiv.Client(
    page_size=BATCH_SIZE,
    delay_seconds=3,
    num_retries=0
)


def is_valid_month(month):
    return 1 <= month <= 12


def parse_year_month(value, argument_name, parser):
    try:
        year_value, month_value = value.split("-")
        year, month = int(year_value), int(month_value)
    except ValueError:
        parser.error(f"{argument_name} must be in YYYY-MM format")

    if year < 1:
        parser.error(f"{argument_name} year must be positive")

    if not is_valid_month(month):
        parser.error(f"{argument_name} month must be between 01 and 12")

    return year, month


# Parse command-line input
def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_month", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_month", type=str, required=True)
    parser.add_argument("--end_id", type=int, required=True)

    args = parser.parse_args()

    start_year, start_month = parse_year_month(
        args.start_month,
        "--start_month",
        parser
    )
    end_year, end_month = parse_year_month(
        args.end_month,
        "--end_month",
        parser
    )

    if not (
        1 <= args.start_id <= 99999
        and 1 <= args.end_id <= 99999
    ):
        parser.error("Paper ID must be between 1 and 99999")

    if (start_year, start_month, args.start_id) > (
        end_year, end_month, args.end_id
    ):
        parser.error("Start must not be later than end")

    return {
        "start_year": start_year,
        "start_month": start_month,
        "start_id": args.start_id,
        "end_year": end_year,
        "end_month": end_month,
        "end_id": args.end_id
    }


# Generate arXiv IDs
def generate_arxiv_ids(
    start_month,
    start_year,
    end_month,
    end_year,
    start_id,
    end_id
):
    if not is_valid_month(start_month) or not is_valid_month(end_month):
        raise ValueError("Month must be between 1 and 12")

    if start_year < 1 or end_year < 1:
        raise ValueError("Year must be positive")

    if (start_year, start_month) > (end_year, end_month):
        raise ValueError("Start month must not be later than end month")

    start = start_year * 12 + (start_month - 1)
    end = end_year * 12 + (end_month - 1)

    for month_index in range(start, end + 1):
        year = month_index // 12
        month = month_index % 12 + 1

        if start == end:
            first_id = start_id
            last_id = end_id

        elif month_index == start:
            first_id = start_id
            last_id = 99999

        elif month_index == end:
            first_id = 1
            last_id = end_id

        else:
            first_id = 1
            last_id = 99999

        for i in range(first_id, last_id + 1):
            yield f"{year % 100:02d}{month:02d}.{i:05d}"


# Batch utilities
def batch_ids(ids, batch_size=BATCH_SIZE):
    ids = iter(ids)

    while True:
        batch = []

        for _ in range(batch_size):
            try:
                batch.append(next(ids))
            except StopIteration:
                break

        if not batch:
            return

        yield batch


# Fetch a batch with retry
def fetch_batch(ids, max_retries=MAX_RETRIES):
    for attempt in range(max_retries + 1):
        try:
            search = arxiv.Search(
                id_list=ids,
                max_results=len(ids)
            )

            papers = list(client.results(search))

            return papers

        except Exception as e:
            if attempt == max_retries:
                logger.exception(
                    "Batch request failed after %d retries; batch_size=%d",
                    max_retries,
                    len(ids)
                )
                raise

            delay = RETRY_DELAY * (2 ** attempt)
            delay += random.uniform(0, 1)

            logger.warning(
                "Retry attempt %d/%d failed; batch_size=%d; error=%s",
                attempt + 1,
                max_retries,
                len(ids),
                e
            )

            logger.info(
                "Waiting %.2f seconds before retry",
                delay
            )

            time.sleep(delay)


# Extract version information
def get_base_id(paper):
    return re.sub(r"v\d+$", "", paper.get_short_id())

def get_latest_version(paper):
    match = re.search(r"v(\d+)$", paper.get_short_id())
    return int(match.group(1)) if match else 1


def extract_versions(paper):
    latest_version = get_latest_version(paper)

    return [
        f"v{i}"
        for i in range(1, latest_version + 1)
    ]


# Batch discovery
def discover_ids(arxiv_ids):
    start_time = time.perf_counter()

    results = []
    failed_ids = []
    not_found_ids = []
    paper_map = {}
    consecutive_not_found = 0
    current_month = None
    month_not_found_streak = 0
    skipped_months = set()

    logger.info("Starting discovery")

    for batch_number, batch in enumerate(
        batch_ids(arxiv_ids),
        start=1
    ):
        batch = [
            arxiv_id
            for arxiv_id in batch
            if arxiv_id[:4] not in skipped_months
        ]

        if not batch:
            continue

        batch_start = time.perf_counter()

        logger.info(
            "PROCESSING initial metadata | Batch %d | Papers %s - %s | Size: %d",
            batch_number,
            batch[0],
            batch[-1],
            len(batch)
        )

        try:
            max_retries = (
                1
                if consecutive_not_found > 3
                else MAX_RETRIES
            )
            papers = fetch_batch(
                batch,
                max_retries=max_retries
            ) or []

        except Exception:
            logger.exception(
                "Failed to fetch initial metadata: batch=%d, size=%d",
                batch_number,
                len(batch)
            )

            failed_ids.extend(batch)
            continue

        # Identify IDs not returned by the API.
        papers_by_id = {
            get_base_id(paper): paper
            for paper in papers
        }

        for arxiv_id in batch:
            month_key = arxiv_id[:4]
            if month_key != current_month:
                current_month = month_key
                month_not_found_streak = 0

            if month_key in skipped_months:
                continue

            paper = papers_by_id.get(arxiv_id)
            if paper is None:
                not_found_ids.append(arxiv_id)
                consecutive_not_found += 1
                month_not_found_streak += 1

                logger.warning(
                    "Paper not found: %s",
                    arxiv_id
                )

                if (
                    month_not_found_streak
                    > MAX_CONSECUTIVE_NOT_FOUND_PER_MONTH
                ):
                    skipped_months.add(month_key)
                    logger.warning(
                        "Skipping remaining papers for month %s after %d consecutive not-found IDs",
                        month_key,
                        month_not_found_streak
                    )
            else:
                paper_map[arxiv_id] = paper
                consecutive_not_found = 0
                month_not_found_streak = 0

        logger.debug(
            "Initial metadata batch completed: batch=%d, returned=%d, elapsed=%.2fs",
            batch_number,
            len(papers),
            time.perf_counter() - batch_start
        )

    # Collect historical revision IDs.
    revision_ids = []

    for arxiv_id, paper in paper_map.items():
        latest_version = get_latest_version(paper)

        for version in range(2, latest_version):
            revision_ids.append(
                f"{arxiv_id}v{version}"
            )

    logger.info(
        "Collected %d historical revision IDs",
        len(revision_ids)
    )

    revision_map = {}

    for batch_number, batch in enumerate(
        batch_ids(revision_ids),
        start=1
    ):
        batch_start = time.perf_counter()

        logger.info(
            "PROCESSING revision metadata | Batch %d | Versions %s - %s | Size: %d",
            batch_number,
            batch[0],
            batch[-1],
            len(batch)
        )

        try:
            papers = fetch_batch(batch) or []

        except Exception:
            logger.exception(
                "Revision batch failed: batch=%d, size=%d",
                batch_number,
                len(batch)
            )
            continue

        for paper in papers:
            revision_map[paper.get_short_id()] = paper

        logger.debug(
            "Revision batch completed: batch=%d, returned=%d, elapsed=%.2fs",
            batch_number,
            len(papers),
            time.perf_counter() - batch_start
        )

    # Build metadata records.
    for arxiv_id, paper in paper_map.items():
        latest_version = get_latest_version(paper)
        versions = extract_versions(paper)

        revised_dates = []
        missing_revision = False

        for version in range(2, latest_version):
            versioned_id = f"{arxiv_id}v{version}"
            version_paper = revision_map.get(versioned_id)

            if version_paper is None:
                logger.warning(
                    "Revision metadata missing: %s",
                    versioned_id
                )

                missing_revision = True
                break

            revised_dates.append(
                version_paper.updated.date().isoformat()
            )

        if latest_version >= 2:
            revised_dates.append(
                paper.updated.date().isoformat()
            )

        if missing_revision:
            failed_ids.append(arxiv_id)

            logger.error(
                "Cannot retrieve all revision metadata for: %s",
                arxiv_id
            )
            continue

        metadata = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [
                author.name
                for author in paper.authors
            ],
            "submission_date": (
                paper.published.date().isoformat()
            ),
            "revised_dates": revised_dates,
            "venue": paper.journal_ref,
            "versions": versions
        }

        results.append(metadata)

        logger.info(
            "PROCESSED paper | ID: %s | Versions: %d",
            arxiv_id,
            latest_version
        )

    # Preserve order and re move duplicate failed IDs.
    failed_ids = list(dict.fromkeys(failed_ids))

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Discovery summary | successful=%d | failed=%d | not_found=%d | elapsed=%.2fs",
        len(results),
        len(failed_ids),
        len(not_found_ids),
        elapsed
    )

    return results, failed_ids, not_found_ids


# Main execution
def main():
    setup_logging()

    args = parse_input()
    start_time = time.perf_counter()

    logger.info("Starting arXiv discovery pipeline")

    arxiv_ids = generate_arxiv_ids(
        start_month=args["start_month"],
        start_year=args["start_year"],
        end_month=args["end_month"],
        end_year=args["end_year"],
        start_id=args["start_id"],
        end_id=args["end_id"]
    )

    results, failed_ids, not_found_ids = discover_ids(
        arxiv_ids
    )

    with open(
        "discovery_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info(
        "Saved discovery results: discovery_results.json"
    )

    with open(
        "failed_ids.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            failed_ids,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Saved failed IDs: failed_ids.json")

    with open(
        "not_found_ids.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            not_found_ids,
            file,
            ensure_ascii=False,
            indent=4
        )

    logger.info("Saved not-found IDs: not_found_ids.json")

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Pipeline summary | successful=%d | failed=%d | not_found=%d",
        len(results),
        len(failed_ids),
        len(not_found_ids)
    )

    logger.info(
        "Total execution time: %.2f seconds (%.2f minutes)",
        elapsed_time,
        elapsed_time / 60
    )


if __name__ == "__main__":
    main()