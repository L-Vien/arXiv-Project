import arxiv
import json
import re
import argparse
import time

client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=0
)

BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5

# Parse from command line, 
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

    if not (1 <= args.start_id <= 99999 and 1 <= args.end_id <= 99999):
        parser.error("Paper ID must be between 1 and 99999")

    if (start_year, start_month, args.start_id) > (end_year, end_month, args.end_id):
        parser.error("Start must not be later than end")

    return {
        "start_year": start_year,
        "start_month": start_month,
        "start_id": args.start_id,
        "end_year": end_year,
        "end_month": end_month,
        "end_id": args.end_id
    }

# From the input of command line, extract each command into a list of arxiv ids
def generate_arxiv_ids(start_month, start_year, end_month, end_year, start_id, end_id):
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
            yield f"{year % 100:02d}{month:02d}.{i:05d}" # arxiv id format: YYMM.XXXXX

def extract_revised_dates(arxiv_id, latest_version):
    revised_dates = []

    for version in range(2, latest_version + 1):
        versioned_id = f"{arxiv_id}v{version}"

        search = arxiv.Search(
            id_list=[versioned_id]
        )

        paper = next(client.results(search), None)

        if paper is None:
            raise RuntimeError(
                f"Cannot retrieve version {versioned_id}"
            )

        revised_dates.append(
            paper.updated.date().isoformat()
        )

    return revised_dates

def discover_paper(arxiv_id):
    search = arxiv.Search(
        id_list=[arxiv_id]
    )

    paper = next(client.results(search), None)

    if paper is None:
        return None

    return paper

def discover_ids(arxiv_ids):
    results = []
    failed_ids = []

    for arxiv_id in arxiv_ids:
        try:
            metadata = fetch_arxiv_metadata(arxiv_id)

            if metadata is None:
                print(f"[NOT FOUND] {arxiv_id}")
                continue

            results.append(metadata)
            print(f"[SUCCESS] {arxiv_id}")

        except Exception as e:
            failed_ids.append(arxiv_id)
            print(f"[ERROR] {arxiv_id}: {e}")

    return results, failed_ids

def fetch_arxiv_metadata(arxiv_id):
    paper = discover_paper(arxiv_id)

    if paper is None:
        return None

    return parse_entry(paper, arxiv_id)


def parse_entry(paper, arxiv_id):
    versions = extract_versions(paper)

    latest_version = len(versions)

    revised_dates = extract_revised_dates(
        arxiv_id,
        latest_version
    )

    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "submission_date": paper.published.date().isoformat(),
        "revised_dates": revised_dates,
        "venue": paper.journal_ref
    }


def extract_versions(paper):
    paper_id = paper.get_short_id()
    match = re.search(r"v(\d+)$", paper_id)

    latest_version = int(match.group(1)) if match else 1

    versions = [
        f"v{i}"
        for i in range(1, latest_version + 1)
    ]

    return versions

def main():
    args = parse_input()
    start_time = time.perf_counter()

    arxiv_ids = generate_arxiv_ids(
        start_month=args["start_month"],
        start_year=args["start_year"],
        end_month=args["end_month"],
        end_year=args["end_year"],
        start_id=args["start_id"],
        end_id=args["end_id"]
    )

    results, failed_ids = discover_ids(arxiv_ids)

    with open("discovery_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    with open("failed_ids.json", "w", encoding="utf-8") as file:
        json.dump(failed_ids, file, ensure_ascii=False, indent=4)
        
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time    

    print(f"\nTotal papers discovered: {len(results)}")
    print(f"Total failed IDs: {len(failed_ids)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Total execution time: {elapsed_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()