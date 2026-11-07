import os
import json
import re
import argparse
import time
import random
import calendar
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import arxiv
from src.crawling.logger import get_logger, setup_logging

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


class DiscoveryCheckpointError(RuntimeError):
    """Raised when discovery checkpoint files cannot be resumed safely."""


def parse_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="arXiv Data Discovery Pipeline")

    parser.add_argument("--start_month", type=str, required=True, help="Format: YYYY-MM")
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_month", type=str, required=True, help="Format: YYYY-MM")
    parser.add_argument("--end_id", type=int, required=True)
    
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save checkpoint files")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Initial discovery batch size")
    parser.add_argument("--mode", type=str, choices=["all", "test"], default="all",
                        help="Execution mode: 'all' to append data & continue, 'test' to wipe prior output files on argument changes.")

    args = parser.parse_args()

    start_year, start_month = map(int, args.start_month.split("-"))
    end_year, end_month = map(int, args.end_month.split("-"))

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        parser.error("Month must be between 01 and 12")

    if not (1 <= args.start_id <= 99999 and 1 <= args.end_id <= 99999):
        parser.error("Paper ID must be between 1 and 99999")

    month_diff = (end_year * 12 + end_month) - (start_year * 12 + start_month)
    if month_diff < 0:
        parser.error("Start date must not be later than end date")
    if month_diff > 1:
        parser.error("Only same month or 2 consecutive months are supported.")

    args.start_year = start_year
    args.start_month_num = start_month
    args.end_year = end_year
    args.end_month_num = end_month
    args.month_diff = month_diff

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
    """Thực hiện gọi API theo ID với cơ chế exponential backoff retry."""
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
            logger.info("Retry attempt %d/%d for batch_size=%d in %.2fs", attempt + 1, MAX_RETRIES, len(ids), delay)
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


def _ids_fingerprint(ids: Iterable[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for arxiv_id in ids:
        digest.update(arxiv_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_jsonl_records(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        return records

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


# Helper persistence operations
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


def wipe_output_files(files: Dict[str, str]) -> None:
    """Xóa sạch nội dung các file .jsonl dữ liệu khi chạy mode 'test' mà thay đổi argument."""
    for key in ["papers", "failed", "not_found"]:
        if key in files and os.path.exists(files[key]):
            with open(files[key], "w", encoding="utf-8"):
                pass
    logger.info("Mode 'test': Successfully wiped previous .jsonl data files.")


def init_checkpoint_files(output_dir: str, current_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    files = {
        "papers": os.path.join(output_dir, "papers.jsonl"),
        "failed": os.path.join(output_dir, "discovery_failed.jsonl"),
        "not_found": os.path.join(output_dir, "discovery_not_found.jsonl"),
        "state": os.path.join(output_dir, "discovery_state.json")
    }

    data_files = (files["papers"], files["failed"], files["not_found"])
    if not os.path.exists(files["state"]) and any(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in data_files
    ):
        raise DiscoveryCheckpointError(
            "Discovery state is missing while discovery JSONL data exists. "
            "Use a new output directory or restore the state file."
        )

    for ftype, filepath in files.items():
        if ftype == "state":
            if not os.path.exists(filepath):
                initial_state = {
                    "input_params": current_params,
                    "current_phase": "MONTH_1_DATE_STREAM" if current_params and current_params.get("is_two_months") else "ID_BATCH",
                    "month_1_t_start_str": None,
                    "month_1_offset": 0,
                    "id_batch_last_completed": 0,
                    "id_batch_last_processed_id": None,
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


def load_checkpoint_state(state_file: str, current_params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
    """
    Trả về (state, params_changed).
    """
    default_state = {
        "input_params": current_params,
        "current_phase": "MONTH_1_DATE_STREAM" if current_params and current_params.get("is_two_months") else "ID_BATCH",
        "month_1_t_start_str": None,
        "month_1_offset": 0,
        "id_batch_last_completed": 0,
        "id_batch_last_processed_id": None,
        "total_found": 0,
        "total_not_found": 0,
        "total_failed": 0
    }

    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            if not isinstance(state, dict):
                raise ValueError("checkpoint must contain a JSON object")

            if current_params and "input_params" in state:
                # So sánh tham số bỏ qua trường 'mode'
                old_p = {k: v for k, v in state["input_params"].items() if k != "mode"}
                new_p = {k: v for k, v in current_params.items() if k != "mode"}

                if old_p != new_p:
                    logger.warning("Input parameters changed! Old: %s | New: %s", old_p, new_p)
                    return state, True

            return state, False
        except Exception as e:
            raise DiscoveryCheckpointError(
                f"Could not read discovery checkpoint: {state_file}"
            ) from e
            
    return default_state, False


# Date stream helpers cho Tháng 1 (Trường hợp 2)
def find_start_timestamp_month_1(year: int, month: int, start_id: int) -> str:
    """Dò bài báo hợp lệ từ start_id trở đi để lấy timestamp bắt đầu cho Month 1."""
    prefix = f"{year % 100:02d}{month:02d}"
    started = time.perf_counter()
    logger.info(
        "Discovery timestamp probe started: prefix=%s start_id=%05d max_probes=30",
        prefix,
        start_id,
    )
    for probe_offset in range(30):
        probe_id = f"{prefix}.{start_id + probe_offset:05d}"
        try:
            papers = fetch_batch_raw([probe_id])
            if papers:
                paper = papers[0]
                t_start_str = paper.published.strftime("%Y%m%d%H%M%S")
                logger.info(
                    "Found reference paper for start timestamp: %s (Published: %s, Solr TS: %s, probe=%d, elapsed=%.2fs)",
                    paper.get_short_id(),
                    paper.published,
                    t_start_str,
                    probe_offset + 1,
                    max(0.0, time.perf_counter() - started),
                )
                return t_start_str
        except Exception as e:
            logger.warning("Probe failed for %s: %s", probe_id, e)
            continue
            
    elapsed = max(0.0, time.perf_counter() - started)
    logger.error(
        "Discovery timestamp probe failed: prefix=%s start_id=%05d probes=30 elapsed=%.2fs",
        prefix,
        start_id,
        elapsed,
    )
    raise RuntimeError(
        f"Could not find any valid paper starting from {prefix}.{start_id:05d} within 30 probes to extract timestamp."
    )


def fetch_date_range_batch(query: str, offset: int, batch_size: int = BATCH_SIZE) -> List[arxiv.Result]:
    """Truy vấn metadata theo khoảng thời gian dựa trên submittedDate (Sử dụng client.results(search, offset=offset))."""
    search = arxiv.Search(
        query=query,
        max_results=offset + batch_size,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending
    )
    for attempt in range(MAX_RETRIES + 1):
        try:
            return list(client.results(search, offset=offset))
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning("Date query batch failed at offset %d after retries: %s", offset, e)
                raise
            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.info("Retry attempt %d/%d for date query offset=%d in %.2fs", attempt + 1, MAX_RETRIES, offset, delay)
            time.sleep(delay)

    raise RuntimeError(f"Date query failed at offset {offset}")


def fetch_date_stream_batch_once(
    query: str,
    offset: int,
    batch_size: int,
    target_yymm: str,
    min_id: int,
    max_found: int | None = None,
    min_subbatch_size: int = MIN_SUBBATCH_SIZE,
) -> Dict[str, Any]:
    """Fetch one date-stream page and convert it to pipeline metadata.

    The request itself is intentionally one-attempt; callers that own the
    pipeline checkpoint and retry policy can wrap this function without
    duplicating the legacy date-stream files.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if min_subbatch_size <= 0:
        raise ValueError("min_subbatch_size must be greater than zero")
    search = arxiv.Search(
        query=query,
        max_results=offset + batch_size,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    raw_papers = list(client.results(search, offset=offset))
    filtered: Dict[str, arxiv.Result] = {}
    for paper in raw_papers:
        base_id = get_base_id(paper)
        match = re.fullmatch(r"(\d{4})\.(\d{5})", base_id)
        if match and match.group(1) == target_yymm and int(match.group(2)) >= min_id:
            filtered[base_id] = paper

    selected = list(filtered.items())
    if max_found is not None:
        selected = selected[: max(0, max_found)]
    paper_map = dict(selected)
    found_papers, failed_ids = process_paper_batch_with_revisions(
        paper_map,
        failed_initial_ids=[],
        min_subbatch_size=min_subbatch_size,
    )
    return {
        "offset": offset + len(raw_papers),
        "raw_count": len(raw_papers),
        "found_papers": found_papers,
        "failed_ids": failed_ids,
    }


def process_paper_batch_with_revisions(
    paper_map: Dict[str, arxiv.Result], 
    failed_initial_ids: List[str],
    min_subbatch_size: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Xử lý lấy Metadata + Revisions cho danh sách bài báo tìm thấy."""
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

    return found_papers, list(dict.fromkeys(batch_failed_ids))


def discover_two_month_range_stream(
    args_params: Dict[str, Any],
    batch_size: int = BATCH_SIZE,
    min_subbatch_size: int = MIN_SUBBATCH_SIZE,
    output_dir: Optional[str] = None
) -> Generator[Dict[str, Any], None, None]:
    files = None
    state = None
    is_two_months = args_params.get("is_two_months", False)

    if output_dir:
        files = init_checkpoint_files(output_dir, current_params=args_params)
        raw_state, params_changed = load_checkpoint_state(files["state"], current_params=args_params)

        if params_changed:
            raise ValueError(
                "Discovery input does not match its checkpoint. "
                "Use a new output directory for a different input."
            )
        else:
            state = raw_state
            state["input_params"] = args_params

    # TRƯỜNG HỢP 2: Khác tháng liền kề -> Giai đoạn 1 (Date stream cho Tháng 1)
    if is_two_months and state and state.get("current_phase") == "MONTH_1_DATE_STREAM":
        start_year = args_params["start_year"]
        start_month = args_params["start_month_num"]
        start_id = args_params["start_id"]

        if not state.get("month_1_t_start_str"):
            logger.info("Probing start timestamp for Month 1 (%02d/%d from ID %d)...", start_month, start_year, start_id)
            t_start_str = find_start_timestamp_month_1(start_year, start_month, start_id)
            state["month_1_t_start_str"] = t_start_str
            if output_dir and files:
                update_state_atomic(files["state"], state)
        else:
            t_start_str = state["month_1_t_start_str"]

        last_day = calendar.monthrange(start_year, start_month)[1]
        t_end_str = f"{start_year}{start_month:02d}{last_day:02d}235959"
        query = f"submittedDate:[{t_start_str} TO {t_end_str}]"
        
        target_yymm = f"{start_year % 100:02d}{start_month:02d}"
        offset = state.get("month_1_offset", 0)

        logger.info("Starting Month 1 Date Query Stream | Query: %s | Resume Offset: %d", query, offset)

        while True:
            batch_start = time.perf_counter()
            logger.info("PROCESSING Month 1 Date Stream | Query Offset: %d", offset)
            
            raw_papers = fetch_date_range_batch(query, offset=offset, batch_size=batch_size)
            if not raw_papers:
                logger.info("Month 1 Date Stream completed. Transitioning to Month 2.")
                state["current_phase"] = "MONTH_2_ID_BATCH"
                if output_dir and files:
                    update_state_atomic(files["state"], state)
                break

            filtered_map = {}
            for p in raw_papers:
                base_id = get_base_id(p)
                match = re.match(r"^(\d{4})\.(\d{5})$", base_id)
                if match:
                    yymm, num_val = match.group(1), int(match.group(2))
                    if yymm == target_yymm and num_val >= start_id:
                        filtered_map[base_id] = p

            found_papers, batch_failed_ids = process_paper_batch_with_revisions(
                filtered_map, failed_initial_ids=[], min_subbatch_size=min_subbatch_size
            )

            offset += len(raw_papers)
            elapsed = time.perf_counter() - batch_start

            batch_result = {
                "phase": "MONTH_1_DATE_STREAM",
                "offset": offset,
                "found_papers": found_papers,
                "found_ids": [p["arxiv_id"] for p in found_papers],
                "not_found_ids": [],
                "failed_ids": batch_failed_ids
            }

            if output_dir and files and state:
                append_jsonl(files["papers"], batch_result["found_papers"])
                append_jsonl(files["failed"], [{"arxiv_id": i} for i in batch_result["failed_ids"]])
                state["month_1_offset"] = offset
                state["total_found"] += len(found_papers)
                state["total_failed"] += len(batch_failed_ids)
                update_state_atomic(files["state"], state)

            logger.debug("Month 1 stream batch completed | found=%d | elapsed=%.2fs", len(found_papers), elapsed)
            yield batch_result

            if len(raw_papers) < batch_size:
                logger.info("Reached end of Month 1 date results. Transitioning to Month 2.")
                state["current_phase"] = "MONTH_2_ID_BATCH"
                if output_dir and files:
                    update_state_atomic(files["state"], state)
                break

    # TRƯỜNG HỢP 1 (Cùng tháng) HOẶC TRƯỜNG HỢP 2 GIAI ĐOẠN 2 (Tháng 2 quét ID từ 00001 -> end_id)
    if is_two_months:
        logger.info("Executing Month 2 ID-based Batch Discovery (00001 -> %d)", args_params["end_id"])
        ids_gen = generate_arxiv_ids(
            start_month=args_params["end_month_num"],
            start_year=args_params["end_year"],
            end_month=args_params["end_month_num"],
            end_year=args_params["end_year"],
            start_id=1,
            end_id=args_params["end_id"]
        )
    else:
        logger.info("Executing Single Month ID-based Batch Discovery (%d -> %d)", args_params["start_id"], args_params["end_id"])
        ids_gen = generate_arxiv_ids(
            start_month=args_params["start_month_num"],
            start_year=args_params["start_year"],
            end_month=args_params["end_month_num"],
            end_year=args_params["end_year"],
            start_id=args_params["start_id"],
            end_id=args_params["end_id"]
        )

    last_completed_batch = state.get("id_batch_last_completed", 0) if state else 0

    for batch_index, batch in enumerate(batch_ids(ids_gen, batch_size=batch_size), start=1):
        if output_dir and batch_index <= last_completed_batch:
            continue

        batch_start = time.perf_counter()
        logger.info(
            "PROCESSING ID Batch %d | Range: %s - %s | Size: %d",
            batch_index, batch[0], batch[-1], len(batch)
        )

        papers, failed_initial_ids = fetch_batch_with_subdivision(batch, min_batch_size=min_subbatch_size)
        paper_map = {get_base_id(p): p for p in papers}
        returned_ids = set(paper_map.keys())

        batch_not_found = [
            arxiv_id for arxiv_id in batch
            if arxiv_id not in returned_ids and arxiv_id not in failed_initial_ids
        ]

        found_papers, batch_failed_ids = process_paper_batch_with_revisions(
            paper_map, failed_initial_ids=failed_initial_ids, min_subbatch_size=min_subbatch_size
        )

        elapsed = time.perf_counter() - batch_start

        batch_result = {
            "phase": "ID_BATCH",
            "batch_index": batch_index,
            "requested_ids": batch,
            "found_papers": found_papers,
            "found_ids": [p["arxiv_id"] for p in found_papers],
            "not_found_ids": batch_not_found,
            "failed_ids": batch_failed_ids
        }

        if output_dir and files and state:
            append_jsonl(files["papers"], batch_result["found_papers"])
            append_jsonl(files["failed"], [{"arxiv_id": i} for i in batch_result["failed_ids"]])
            append_jsonl(files["not_found"], [{"arxiv_id": i} for i in batch_result["not_found_ids"]])

            state["id_batch_last_completed"] = batch_index
            state["id_batch_last_processed_id"] = batch[-1]
            state["total_found"] += len(found_papers)
            state["total_not_found"] += len(batch_not_found)
            state["total_failed"] += len(batch_failed_ids)
            update_state_atomic(files["state"], state)

        logger.debug(
            "ID Batch %d finished | found=%d | not_found=%d | failed=%d | elapsed=%.2fs",
            batch_index, len(found_papers), len(batch_not_found), len(batch_failed_ids), elapsed
        )

        yield batch_result


def discover_ids_stream(
    arxiv_ids: Iterable[str],
    *,
    batch_size: int = BATCH_SIZE,
    min_subbatch_size: int = MIN_SUBBATCH_SIZE,
    output_dir: Optional[str] = None,
    input_params: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Discover metadata for an arbitrary ordered iterable of arXiv IDs.

    The checkpoint identifies the complete input definition. A changed input
    is rejected before any new JSONL record is appended, preventing unrelated
    runs from being mixed in one discovery directory.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if min_subbatch_size <= 0:
        raise ValueError("min_subbatch_size must be greater than zero")

    ids = list(dict.fromkeys(str(arxiv_id) for arxiv_id in arxiv_ids))
    params = dict(input_params or {})
    params.setdefault("candidate_count", len(ids))
    params.setdefault("candidate_ids_sha256", _ids_fingerprint(ids))
    params["batch_size"] = batch_size
    params["min_subbatch_size"] = min_subbatch_size

    files = None
    state = {
        "input_params": params,
        "id_batch_last_completed": 0,
        "id_batch_last_processed_id": None,
        "total_found": 0,
        "total_not_found": 0,
        "total_failed": 0,
    }

    if output_dir:
        files = init_checkpoint_files(output_dir, current_params=params)
        state, params_changed = load_checkpoint_state(
            files["state"],
            current_params=params,
        )
        if params_changed:
            raise ValueError(
                "Discovery input does not match its checkpoint. "
                "Use a new output directory for a different input."
            )

        state["input_params"] = params

        persisted_papers = _read_jsonl_records(files["papers"])
        persisted_failed = _read_jsonl_records(files["failed"])
        persisted_not_found = _read_jsonl_records(files["not_found"])

        paper_by_id = {
            str(record["arxiv_id"]): record
            for record in persisted_papers
            if record.get("arxiv_id")
        }
        failed_ids = [
            str(record["arxiv_id"])
            for record in persisted_failed
            if record.get("arxiv_id")
            and str(record["arxiv_id"]) not in paper_by_id
        ]
        not_found_ids = [
            str(record["arxiv_id"])
            for record in persisted_not_found
            if record.get("arxiv_id")
            and str(record["arxiv_id"]) not in paper_by_id
        ]

        if paper_by_id or failed_ids or not_found_ids:
            yield {
                "phase": "RESUME",
                "batch_index": 0,
                "requested_ids": [],
                "found_papers": list(paper_by_id.values()),
                "found_ids": list(paper_by_id),
                "failed_ids": list(dict.fromkeys(failed_ids)),
                "not_found_ids": list(dict.fromkeys(not_found_ids)),
            }

        retry_ids = set(failed_ids)
    else:
        retry_ids = set()

    completed_batch = int(state.get("id_batch_last_completed", 0))
    total_batches = (len(ids) + batch_size - 1) // batch_size
    if retry_ids and total_batches and completed_batch >= total_batches:
        completed_batch = total_batches - 1
    batches = batch_ids(ids, batch_size=batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        if batch_index <= completed_batch:
            continue

        requested_ids = list(batch)
        if retry_ids:
            requested_ids = list(dict.fromkeys(
                [arxiv_id for arxiv_id in requested_ids if arxiv_id in retry_ids]
                + [arxiv_id for arxiv_id in ids if arxiv_id in retry_ids]
            ))
            retry_ids.difference_update(requested_ids)

        if not requested_ids:
            continue

        papers, failed_initial_ids = fetch_batch_with_subdivision(
            requested_ids,
            min_batch_size=min_subbatch_size,
        )
        paper_map = {get_base_id(paper): paper for paper in papers}
        returned_ids = set(paper_map)
        batch_not_found = [
            arxiv_id
            for arxiv_id in requested_ids
            if arxiv_id not in returned_ids
            and arxiv_id not in failed_initial_ids
        ]
        found_papers, batch_failed_ids = process_paper_batch_with_revisions(
            paper_map,
            failed_initial_ids=failed_initial_ids,
            min_subbatch_size=min_subbatch_size,
        )

        batch_result = {
            "phase": "ID_BATCH",
            "batch_index": batch_index,
            "requested_ids": requested_ids,
            "found_papers": found_papers,
            "found_ids": [paper["arxiv_id"] for paper in found_papers],
            "not_found_ids": batch_not_found,
            "failed_ids": list(dict.fromkeys(batch_failed_ids)),
        }

        if files is not None:
            append_jsonl(files["papers"], found_papers)
            append_jsonl(
                files["failed"],
                [{"arxiv_id": arxiv_id} for arxiv_id in batch_result["failed_ids"]],
            )
            append_jsonl(
                files["not_found"],
                [{"arxiv_id": arxiv_id} for arxiv_id in batch_not_found],
            )
            state.update({
                "input_params": params,
                "id_batch_last_completed": batch_index,
                "id_batch_last_processed_id": batch[-1],
                "total_found": state.get("total_found", 0) + len(found_papers),
                "total_not_found": state.get("total_not_found", 0) + len(batch_not_found),
                "total_failed": state.get("total_failed", 0) + len(batch_result["failed_ids"]),
            })
            update_state_atomic(files["state"], state)

        yield batch_result


def main():
    setup_logging()
    args = parse_input()
    start_time = time.perf_counter()

    current_input_params = {
        "start_month": args.start_month,
        "start_id": args.start_id,
        "end_month": args.end_month,
        "end_id": args.end_id,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "start_year": args.start_year,
        "start_month_num": args.start_month_num,
        "end_year": args.end_year,
        "end_month_num": args.end_month_num,
        "is_two_months": (args.month_diff == 1)
    }

    logger.info("Starting arXiv discovery pipeline in [%s] mode with output directory: %s", args.mode, args.output_dir)

    batch_stream = discover_two_month_range_stream(
        args_params=current_input_params,
        batch_size=args.batch_size,
        min_subbatch_size=MIN_SUBBATCH_SIZE,
        output_dir=args.output_dir
    )

    for _ in batch_stream:
        pass

    elapsed_time = time.perf_counter() - start_time
    logger.info("Pipeline execution completed in %.2fs", elapsed_time)
    logger.info("Pipeline execution completed in %.2fm", elapsed_time/60)


if __name__ == "__main__":
    main()
