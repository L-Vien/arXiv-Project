import os
import json
import re
import argparse
import time
import random
from typing import List, Dict, Any, Generator, Tuple, Optional

import arxiv
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Configuration Defaults
BATCH_SIZE = 100
MIN_SUBBATCH_SIZE = 25
MAX_RETRIES = 3
RETRY_DELAY = 5

client = arxiv.Client(
    page_size=BATCH_SIZE,
    delay_seconds=3,
    num_retries=0
)


def parse_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="arXiv Data Discovery Pipeline")

    parser.add_argument("--start_month", type=str, required=True, help="Format: YYYY-MM")
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_month", type=str, required=True, help="Format: YYYY-MM")
    parser.add_argument("--end_id", type=int, required=True)
    
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save checkpoint files")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Initial discovery batch size")
    parser.add_argument("--early_stop_not_found", type=int, default=0, 
                        help="Consecutive not_found limit to skip remaining range (0 to disable for strict coverage)")

    args = parser.parse_args()

    start_year, start_month = map(int, args.start_month.split("-"))
    end_year, end_month = map(int, args.end_month.split("-"))

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        parser.error("Month must be between 01 and 12")

    if not (1 <= args.start_id <= 99999 and 1 <= args.end_id <= 99999):
        parser.error("Paper ID must be between 1 and 99999")

    if (start_year, start_month, args.start_id) > (end_year, end_month, args.end_id):
        parser.error("Start must not be later than end")

    args.start_year = start_year
    args.start_month_num = start_month
    args.end_year = end_year
    args.end_month_num = end_month

    return args


def generate_arxiv_ids(
    start_month: int,
    start_year: int,
    end_month: int,
    end_year: int,
    start_id: int,
    end_id: int
) -> Generator[str, None, None]:
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


def batch_ids(ids: Generator[str, None, None], batch_size: int = BATCH_SIZE) -> Generator[List[str], None, None]:
    ids_iter = iter(ids)
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(ids_iter))
            except StopIteration:
                break
        if not batch:
            return
        yield batch


def fetch_batch_raw(ids: List[str]) -> List[arxiv.Result]:
    """Thực hiện gọi API với cơ chế exponential backoff retry cơ bản."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            search = arxiv.Search(id_list=ids, max_results=len(ids))
            return list(client.results(search))
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(
                    "Batch request failed after %d retries; batch_size=%d; error=%s",
                    MAX_RETRIES, len(ids), e
                )
                raise

            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.info("Retry attempt %d/%d for batch_size=%d in %.2fs", attempt + 1, MAX_RETRIES, delay)
            time.sleep(delay)

    raise RuntimeError(f"Batch request failed after {MAX_RETRIES} retries.")


def fetch_batch_with_subdivision(ids: List[str], min_batch_size: int = MIN_SUBBATCH_SIZE) -> Tuple[List[arxiv.Result], List[str]]:
    if not ids:
        return [], []

    try:
        papers = fetch_batch_raw(ids)
        return papers or [], []
    except Exception:
        if len(ids) <= min_batch_size:
            logger.error("Sub-batch failed completely at minimum size %d: %s - %s", len(ids), ids[0], ids[-1])
            return [], list(ids)

        num_splits = min(4, len(ids))
        sub_size = max(1, len(ids) // num_splits)

        logger.warning(
            "Batch size %d failed. Subdividing into %d smaller sub-batches of size ~%d",
            len(ids), num_splits, sub_size
        )

        all_papers: List[arxiv.Result] = []
        all_failed_ids: List[str] = []

        for i in range(0, len(ids), sub_size):
            sub_ids = ids[i:i + sub_size]
            sub_papers, sub_failed = fetch_batch_with_subdivision(sub_ids, min_batch_size=min_batch_size)
            all_papers.extend(sub_papers)
            all_failed_ids.extend(sub_failed)

        return all_papers, all_failed_ids


def get_base_id(paper: arxiv.Result) -> str:
    return re.sub(r"v\d+$", "", paper.get_short_id())


def get_latest_version(paper: arxiv.Result) -> int:
    match = re.search(r"v(\d+)$", paper.get_short_id())
    return int(match.group(1)) if match else 1


def extract_versions(paper: arxiv.Result) -> List[str]:
    latest_version = get_latest_version(paper)
    return [f"v{i}" for i in range(1, latest_version + 1)]


# Helper Persistence Operations
def append_jsonl(file_path: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    with open(file_path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def update_state_atomic(state_path: str, state_data: Dict[str, Any]) -> None:
    temp_path = f"{state_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, state_path)


def init_checkpoint_files(output_dir: str, current_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Tạo thư mục và khởi tạo các file checkpoint rỗng trên ổ đĩa."""
    os.makedirs(output_dir, exist_ok=True)
    files = {
        "papers": os.path.join(output_dir, "papers.jsonl"),
        "failed": os.path.join(output_dir, "discovery_failed.jsonl"),
        "not_found": os.path.join(output_dir, "discovery_not_found.jsonl"),
        "state": os.path.join(output_dir, "discovery_state.json")
    }

    for ftype, filepath in files.items():
        if ftype == "state":
            if not os.path.exists(filepath):
                initial_state = {
                    "input_params": current_params,
                    "last_completed_batch": 0,
                    "last_processed_id": None,
                    "total_found": 0,
                    "total_not_found": 0,
                    "total_failed": 0
                }
                update_state_atomic(filepath, initial_state)
        else:
            if not os.path.exists(filepath):
                with open(filepath, "a", encoding="utf-8"):
                    pass

    return files


def load_checkpoint_state(state_file: str, current_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    default_state = {
        "input_params": current_params,
        "last_completed_batch": 0,
        "last_processed_id": None,
        "total_found": 0,
        "total_not_found": 0,
        "total_failed": 0
    }

    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # ĐỐI SOÁT INPUT PARAMS: Nếu input khác với lần chạy trước, reset checkpoint!
            if current_params and "input_params" in state:
                if state["input_params"] != current_params:
                    logger.warning(
                        "Input parameters changed! Old: %s | New: %s. Resetting checkpoint state.",
                        state["input_params"], current_params
                    )
                    return default_state

            return state
        except Exception as e:
            logger.warning("Could not read state file: %s. Re-initializing.", e)
            
    return default_state


def discover_ids_stream(
    arxiv_ids: Generator[str, None, None],
    batch_size: int = BATCH_SIZE,
    min_subbatch_size: int = MIN_SUBBATCH_SIZE,
    early_stop_not_found: int = 0,
    output_dir: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None
) -> Generator[Dict[str, Any], None, None]:
    files = None
    state = None
    last_completed_batch = 0

    if output_dir:
        files = init_checkpoint_files(output_dir, current_params=input_params)
        state = load_checkpoint_state(files["state"], current_params=input_params)
        
        # Đảm bảo state lưu cấu hình input hiện tại
        state["input_params"] = input_params
        last_completed_batch = state.get("last_completed_batch", 0)
        last_processed_id = state.get("last_processed_id")

        if last_completed_batch > 0 or last_processed_id:
            logger.info(
                "Auto-resuming discovery from batch %d (Last processed ID: %s)",
                last_completed_batch + 1, last_processed_id
            )
        else:
            logger.info("Starting fresh discovery session (Batch 1)")

    consecutive_not_found = 0

    for batch_index, batch in enumerate(batch_ids(arxiv_ids, batch_size=batch_size), start=1):

        # Bỏ qua các batch đã hoàn thành trước đó nếu khớp checkpoint
        if output_dir and batch_index <= last_completed_batch:
            continue

        batch_start = time.perf_counter()
        logger.info(
            "PROCESSING initial metadata | Batch %d | Papers %s - %s | Size: %d",
            batch_index, batch[0], batch[-1], len(batch)
        )

        # 1. Fetch initial metadata
        papers, failed_initial_ids = fetch_batch_with_subdivision(batch, min_batch_size=min_subbatch_size)

        paper_map = {get_base_id(p): p for p in papers}
        returned_ids = set(paper_map.keys())

        # 2. Identify not_found
        batch_not_found = [
            arxiv_id for arxiv_id in batch
            if arxiv_id not in returned_ids and arxiv_id not in failed_initial_ids
        ]

        if early_stop_not_found > 0:
            if not returned_ids:
                consecutive_not_found += len(batch)
            else:
                consecutive_not_found = 0

        # 3. Batch revision discovery
        revision_ids = []
        for arxiv_id, paper in paper_map.items():
            latest_version = get_latest_version(paper)
            for version in range(2, latest_version):
                revision_ids.append(f"{arxiv_id}v{version}")

        revision_map = {}
        failed_revision_base_ids = set()

        if revision_ids:
            rev_papers, rev_failed_ids = fetch_batch_with_subdivision(
                revision_ids, min_batch_size=min_subbatch_size
            )
            for rev_paper in rev_papers:
                revision_map[rev_paper.get_short_id()] = rev_paper

            for rf_id in rev_failed_ids:
                base_id = re.sub(r"v\d+$", "", rf_id)
                failed_revision_base_ids.add(base_id)

        # 4. Build final paper records
        found_papers = []
        batch_failed_ids = list(failed_initial_ids)

        for arxiv_id, paper in paper_map.items():
            if arxiv_id in failed_revision_base_ids:
                logger.error("Missing revision metadata for paper: %s", arxiv_id)
                batch_failed_ids.append(arxiv_id)
                continue

            latest_version = get_latest_version(paper)
            versions = extract_versions(paper)
            revised_dates = []
            missing_revision = False

            for version in range(2, latest_version):
                versioned_id = f"{arxiv_id}v{version}"
                version_paper = revision_map.get(versioned_id)

                if version_paper is None:
                    missing_revision = True
                    break

                revised_dates.append(version_paper.updated.date().isoformat())

            if latest_version >= 2:
                revised_dates.append(paper.updated.date().isoformat())

            if missing_revision:
                batch_failed_ids.append(arxiv_id)
                continue

            metadata = {
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "submission_date": paper.published.date().isoformat(),
                "revised_dates": revised_dates,
                "venue": paper.journal_ref,
                "versions": versions
            }
            found_papers.append(metadata)

        batch_failed_ids = list(dict.fromkeys(batch_failed_ids))
        elapsed = time.perf_counter() - batch_start

        batch_result = {
            "batch_index": batch_index,
            "requested_ids": batch,
            "found_papers": found_papers,
            "found_ids": [p["arxiv_id"] for p in found_papers],
            "not_found_ids": batch_not_found,
            "failed_ids": batch_failed_ids
        }

        # Ghi đĩa & cập nhật State bao gồm cả input_params hiện tại
        if output_dir and files and state:
            append_jsonl(files["papers"], batch_result["found_papers"])
            append_jsonl(files["failed"], [{"arxiv_id": i} for i in batch_result["failed_ids"]])
            append_jsonl(files["not_found"], [{"arxiv_id": i} for i in batch_result["not_found_ids"]])

            state["last_completed_batch"] = batch_index
            state["last_processed_id"] = batch[-1]
            state["total_found"] += len(batch_result["found_papers"])
            state["total_not_found"] += len(batch_result["not_found_ids"])
            state["total_failed"] += len(batch_result["failed_ids"])
            update_state_atomic(files["state"], state)

        logger.debug(
            "Batch %d finished | found=%d | not_found=%d | failed=%d | elapsed=%.2fs",
            batch_index, len(found_papers), len(batch_not_found), len(batch_failed_ids), elapsed
        )

        yield batch_result

        if early_stop_not_found > 0 and consecutive_not_found >= early_stop_not_found:
            logger.warning("Early stop limit reached (%d consecutive not_found). Ending.", early_stop_not_found)
            break


def main():
    setup_logging()
    args = parse_input()
    start_time = time.perf_counter()

    current_input_params = {
        "start_month": args.start_month,
        "start_id": args.start_id,
        "end_month": args.end_month,
        "end_id": args.end_id,
        "batch_size": args.batch_size
    }

    logger.info("Starting arXiv discovery pipeline with output directory: %s", args.output_dir)

    all_ids_generator = generate_arxiv_ids(
        start_month=args.start_month_num,
        start_year=args.start_year,
        end_month=args.end_month_num,
        end_year=args.end_year,
        start_id=args.start_id,
        end_id=args.end_id
    )

    batch_stream = discover_ids_stream(
        arxiv_ids=all_ids_generator,
        batch_size=args.batch_size,
        min_subbatch_size=MIN_SUBBATCH_SIZE,
        early_stop_not_found=args.early_stop_not_found,
        output_dir=args.output_dir,
        input_params=current_input_params
    )

    for _ in batch_stream:
        pass

    elapsed_time = time.perf_counter() - start_time
    logger.info("Pipeline execution completed in %.2fs", elapsed_time)


if __name__ == "__main__":
    main()