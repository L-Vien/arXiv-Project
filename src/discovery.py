import arxiv
import json
import re
import argparse
import time
import random

from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Configuration
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5

client = arxiv.Client(
    page_size=BATCH_SIZE,
    delay_seconds=3,
    num_retries=0
)


# Parse command-line input
def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_month", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_month", type=str, required=True)
    parser.add_argument("--end_id", type=int, required=True)

    args = parser.parse_args()

    start_year, start_month = map(int, args.start_month.split("-"))
    end_year, end_month = map(int, args.end_month.split("-"))

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        parser.error("Month must be between 01 and 12")

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
def fetch_batch(ids):
    for attempt in range(MAX_RETRIES + 1):
        try:
            search = arxiv.Search(
                id_list=ids,
                max_results=len(ids)
            )

            papers = list(client.results(search))

            return papers

        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.exception(
                    "Batch request failed after %d retries; batch_size=%d",
                    MAX_RETRIES,
                    len(ids)
                )
                raise

            delay = RETRY_DELAY * (2 ** attempt)
            delay += random.uniform(0, 1)

            logger.warning(
                "Retry attempt %d/%d failed; batch_size=%d; error=%s",
                attempt + 1,
                MAX_RETRIES,
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

    logger.info("Starting discovery")

    for batch_number, batch in enumerate(
        batch_ids(arxiv_ids),
        start=1
    ):
        batch_start = time.perf_counter()

        logger.info(
            "PROCESSING initial metadata | Batch %d | Papers %s - %s | Size: %d",
            batch_number,
            batch[0],
            batch[-1],
            len(batch)
        )

        try:
            papers = fetch_batch(batch) or []

        except Exception:
            logger.exception(
                "Failed to fetch initial metadata: batch=%d, size=%d",
                batch_number,
                len(batch)
            )

            failed_ids.extend(batch)
            continue

        for paper in papers:
            base_id = get_base_id(paper)
            paper_map[base_id] = paper

        # Identify IDs not returned by the API.
        returned_ids = {
            get_base_id(paper)
            for paper in papers
        }

        for arxiv_id in batch:
            if arxiv_id not in returned_ids:
                not_found_ids.append(arxiv_id)

                logger.warning(
                    "Paper not found: %s",
                    arxiv_id
                )

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