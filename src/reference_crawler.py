"""
Module: reference.py
Description:
    Crawl paper references from Semantic Scholar API for Milestone 1 of IntroDS Project.
    Features:
    - Multi-Key Rotation (APIKeyPool): Automatically rotates Semantic Scholar API keys (S2_API_KEY_1, S2_API_KEY_2, etc.) to handle 429 Rate Limits.
    - Multithreaded Batch Processing (ThreadPoolExecutor) with Bounded Submission: Concurrent paper crawling with controlled task queue size for high throughput and low RAM footprint.
    - Robust Caching / Resume mechanism with retryable failure distinction: Skips re-fetching if valid cache exists, retries only retryable failures.
    - Atomic File Writing: Writes references and temp files safely using .tmp and os.replace.
    - Streaming Metrics (reference_metrics.jsonl): Appends per-paper metrics line-by-line to avoid keeping unbounded state in RAM.
    - Data Standardization: Normalizes references to match the required project schema (arxiv_id, ref_arxiv_id, ref_title, ref_authors, ref_submission_date, ref_s2_id).
    - Unified Corpus Folder Naming: Replaces both '/' and '.' with '-' (e.g. 2604.25535 -> 2604-25535) to match Processor module.
    - Performance & Memory Tracking: Measures execution time, wall time per paper, and RAM usage via psutil.
    - Custom Logging: File and console logging with custom levels (SUCCESS, FAIL).

Requirements:
    - requests
    - psutil (optional, for memory tracking)
"""

import os
import json
import time
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Robust manual .env loading (no external dependency required)
try:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
except Exception:
    pass

# Define custom log levels: SUCCESS (25) and FAIL (45)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
def logger_success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)
logging.Logger.success = logger_success

FAIL_LEVEL = 45
logging.addLevelName(FAIL_LEVEL, "FAIL")
def logger_fail(self, message, *args, **kwargs):
    if self.isEnabledFor(FAIL_LEVEL):
        self._log(FAIL_LEVEL, message, args, **kwargs)
logging.Logger.fail = logger_fail

# Configure logging (Console)
logger = logging.getLogger("ReferenceCrawler")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console_handler)


class APIKeyPool:
    """
    Thread-safe pool for rotating multiple Semantic Scholar API keys in a round-robin fashion.
    Helps distribute request load and avoid 429 Rate Limit errors during large crawls.
    """
    def __init__(self, api_keys: List[str]):
        if not api_keys or all(not k.strip() for k in api_keys):
            raise ValueError("Danh sách API Key rỗng hoặc không hợp lệ!")
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.index = 0
        self.lock = threading.Lock()

    def get_key(self) -> str:
        """Retrieve the next API key thread-safely using round-robin indexing."""
        with self.lock:
            key = self.api_keys[self.index]
            self.index = (self.index + 1) % len(self.api_keys)
            return key


class ReferenceCrawler:
    """
    Main crawler class to fetch, standardize, cache, and save paper references from Semantic Scholar API.
    Supports multi-key rotation, bounded multithreaded batch processing, atomic saves, and streaming metrics.
    """
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        log_dir: str = "log",
        request_delay: float = 2.0,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the Reference Crawler with directories, rate limits, retry policies, and API key pool.
        """
        self.corpus_dir = Path(corpus_dir)

        # Ensure log_dir is placed at the same level as the 'data' directory (sibling to data)
        resolved_corpus = self.corpus_dir.resolve() if self.corpus_dir.is_absolute() else (Path.cwd() / self.corpus_dir).resolve()
        curr = resolved_corpus
        data_parent = None
        while curr != curr.parent:
            if curr.name == "data":
                data_parent = curr.parent
                break
            curr = curr.parent

        if log_dir == "log":
            if data_parent:
                self.log_dir = data_parent / "log"
            else:
                self.log_dir = self.corpus_dir.parent.parent / "log" if len(self.corpus_dir.parts) >= 2 else Path("log")
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.request_delay = request_delay
        self.max_retries = max_retries

        # Setup file logging (reference_crawler.log inside log directory)
        log_file = self.log_dir / "reference_crawler.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        # Avoid duplicate file handlers if initialized multiple times
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

        # Load API keys: supports explicit argument, S2_API_KEY_1, S2_API_KEY_2, etc., or legacy env vars
        raw_keys = []
        if api_key:
            raw_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            env_keys_dict = {}
            for k, v in os.environ.items():
                k_upper = k.upper()
                if (k_upper.startswith("S2_API_KEY_") or k_upper.startswith("SEMANTIC_SCHOLAR_API_KEY_")) and v.strip():
                    env_keys_dict[k_upper] = v.strip()

            if env_keys_dict:
                sorted_env_keys = [v for k, v in sorted(env_keys_dict.items(), key=lambda x: (len(x[0]), x[0]))]
                raw_keys.extend(sorted_env_keys)

            if not raw_keys:
                single_env = os.getenv("SEMANTIC_SCHOLAR_API_KEYS") or os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                if single_env:
                    raw_keys = [k.strip() for k in single_env.split(",") if k.strip()]

        if not raw_keys:
            raise ValueError(
                "Lỗi: Không tìm thấy Semantic Scholar API Key! "
                "Vui lòng cấu hình S2_API_KEY_1, S2_API_KEY_2... trong file .env"
            )

        # Initialize API Key Pool for rotation
        self.key_pool = APIKeyPool(raw_keys)

        # Configure max_workers (default to number of API keys if not specified)
        self.max_workers = max_workers if max_workers is not None else len(self.key_pool.api_keys)
        logger.info(f"Semantic Scholar API Key Pool initialized with {len(self.key_pool.api_keys)} key(s).")
        logger.info(f"Crawler configured to run with {self.max_workers} concurrent worker thread(s).")
        logger.info(f"Log directory set to: {self.log_dir.absolute()}")

        # Session setup for connection pooling and HTTP reuse
        self.session = requests.Session()
        self.base_headers = {
            "User-Agent": "IntroDS-ArXivCrawler/1.0 (Educational Project)"
        }

        # Thread lock for updating metrics safely across threads
        self.metrics_lock = threading.Lock()

        # Performance tracking metrics dictionary (lightweight counters)
        self.metrics = {
            "total_papers_processed": 0,
            "successful_crawls": 0,
            "failed_crawls": 0,
            "failed_arxiv_ids": [],
            "total_references_fetched": 0,
            "references_with_arxiv_id": 0,
            "start_time": None,
            "end_time": None,
            "peak_memory_mb": 0.0,
            "avg_memory_mb": 0.0,
            "memory_samples": []
        }

    def close(self):
        """Close file handlers to release file locks on Windows."""
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.close()
                    logger.removeHandler(h)
                except Exception:
                    pass

    def _get_process_memory_mb(self) -> float:
        """Helper method to query current process RSS memory usage in MB using psutil."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _log_metric_line(self, metric_record: Dict[str, Any]):
        """Append a single paper metric record to reference_metrics.jsonl (streaming/line-by-line)."""
        metrics_file = self.log_dir / "reference_metrics.jsonl"
        try:
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write metric line: {e}")

    def _get_folder_name(self, arxiv_id: str) -> str:
        """
        Normalize arXiv ID into folder name matching the Processor module (Học's processor):
        Replaces both '/' and '.' with '-' (e.g. 2604.25535 -> 2604-25535).
        """
        return arxiv_id.replace('/', '-').replace('.', '-')

    def fetch_references(self, arxiv_id: str) -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[str], int]:
        """
        Fetch reference list for a given arXiv ID from Semantic Scholar Graph API.
        Implements automatic key rotation, exponential backoff, and 429 rate limit handling.
        Returns: (references, retryable, error_message, attempts_used)
        """
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id and not arxiv_id.startswith('hep') else arxiv_id
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"

        params = {
            "fields": "references.title,references.authors,references.year,references.publicationDate,references.venue,references.externalIds"
        }

        max_429_retries = 4
        retries_429 = 0

        for attempt in range(1, self.max_retries + 1):
            current_api_key = self.key_pool.get_key()
            headers = {**self.base_headers, "X-API-KEY": current_api_key}

            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params, headers=headers, timeout=30)

                # Handle HTTP 429 Too Many Requests (Rate Limit Exceeded)
                if response.status_code == 429:
                    retries_429 += 1
                    if retries_429 > max_429_retries:
                        err = f"Exceeded max 429 retries ({max_429_retries}) for paper {arxiv_id}."
                        logger.fail(err)
                        return None, True, err, attempt

                    logger.warning(
                        f"[429 Rate Limit] Key {current_api_key[:6]}... hit limit for paper {arxiv_id}. "
                        f"Rotating to next key and retrying (Attempt {retries_429}/{max_429_retries})..."
                    )
                    time.sleep(1.5)
                    continue

                # Handle HTTP 404 Not Found (Permanent failure, not retryable)
                if response.status_code == 404:
                    logger.warning(f"Paper {arxiv_id} not found on Semantic Scholar (404).")
                    return [], False, "Paper not found (404)", attempt

                response.raise_for_status()
                data = response.json()
                return data.get("references", []), False, None, attempt

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching references for {arxiv_id} using key {current_api_key[:6]}... (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    err = f"Failed fetching references for {arxiv_id} after {self.max_retries} attempts: {e}"
                    logger.fail(err)
                    return None, True, err, attempt
                wait_time = min(2 * attempt, 6)
                time.sleep(wait_time)

        return None, True, "Max retries reached", self.max_retries

    def standardize_references(self, arxiv_id: str, raw_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize raw reference dictionaries from Semantic Scholar API into the standardized project schema:
        - arxiv_id: ID of the citing paper
        - ref_arxiv_id: arXiv ID of the reference (if available)
        - ref_title: Title of the referenced paper
        - ref_authors: List of author names
        - ref_submission_date: Publication date or year
        - ref_s2_id: Semantic Scholar paper ID
        """
        standardized_list = []
        for ref in raw_references:
            ref_s2_id = ref.get("paperId")
            ref_title = ref.get("title")

            raw_authors = ref.get("authors", [])
            ref_authors = [author.get("name", "") for author in raw_authors if author.get("name")]

            ref_date = ref.get("publicationDate") or str(ref.get("year", ""))

            external_ids = ref.get("externalIds", {}) or {}
            ref_arxiv_id = external_ids.get("ArXiv")

            if ref_arxiv_id:
                with self.metrics_lock:
                    self.metrics["references_with_arxiv_id"] += 1

            std_ref = {
                "arxiv_id": arxiv_id,
                "ref_arxiv_id": ref_arxiv_id,
                "ref_title": ref_title,
                "ref_authors": ref_authors,
                "ref_submission_date": ref_date,
                "ref_s2_id": ref_s2_id
            }
            standardized_list.append(std_ref)
        return standardized_list

    def save_paper_references(
        self,
        arxiv_id: str,
        references: Optional[List[Dict[str, Any]]],
        status: str = "success",
        retryable: bool = False,
        error_message: Optional[str] = None,
        attempt: int = 1
    ) -> Path:
        """
        Save standardized references with wrapper status and retryable flag to local disk
        under corpus/<paper_id_normalized>/references.json using atomic file replacement (.tmp -> final).
        """
        folder_name = self._get_folder_name(arxiv_id)
        paper_dir = self.corpus_dir / folder_name
        paper_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "arxiv_id": arxiv_id,
            "status": status,
            "retryable": retryable,
            "attempt": attempt,
            "error_message": error_message,
            "references_count": len(references) if references is not None else 0,
            "references": references
        }

        output_file = paper_dir / "references.json"
        temp_file = paper_dir / "references.json.tmp"

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, output_file)
        except Exception as e:
            logger.error(f"Failed atomic save for {arxiv_id}: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise

        return output_file

    def process_single_paper(self, arxiv_id: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Process a single paper: check local cache (respecting status and retryable flags),
        fetch from API if missing/retryable/forced, standardize, save atomically,
        update thread-safe metrics, and stream metric line to reference_metrics.jsonl.
        """
        folder_name = self._get_folder_name(arxiv_id)
        paper_dir = self.corpus_dir / folder_name
        output_file = paper_dir / "references.json"

        # Local Caching / Resume Check: inspect cache status & retryable flag
        if not force_refresh and output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    std_refs = data
                    status = "success"
                    retryable = False
                elif isinstance(data, dict):
                    status = data.get("status", "success")
                    retryable = data.get("retryable", False)
                    std_refs = data.get("references") or []
                else:
                    std_refs = []
                    status = "success"
                    retryable = False

                if status == "success":
                    logger.success(f"Loaded cached references for {arxiv_id} from {output_file} (skipping API call).")

                    with self.metrics_lock:
                        self.metrics["total_papers_processed"] += 1
                        self.metrics["successful_crawls"] += 1
                        self.metrics["total_references_fetched"] += len(std_refs)
                        for ref in std_refs:
                            if ref.get("ref_arxiv_id"):
                                self.metrics["references_with_arxiv_id"] += 1

                    mem_mb = self._get_process_memory_mb() if PSUTIL_AVAILABLE else 0.0
                    self._log_metric_line({
                        "arxiv_id": arxiv_id,
                        "status": "success",
                        "source": "cache",
                        "retryable": False,
                        "references_count": len(std_refs),
                        "memory_mb": mem_mb
                    })
                    return std_refs
                elif status == "failed" and not retryable and not force_refresh:
                    logger.warning(f"Cached status for {arxiv_id} is 'failed' and non-retryable (e.g. 404). Skipping re-fetch.")
                    with self.metrics_lock:
                        self.metrics["total_papers_processed"] += 1
                        self.metrics["failed_crawls"] += 1
                        if arxiv_id not in self.metrics["failed_arxiv_ids"]:
                            self.metrics["failed_arxiv_ids"].append(arxiv_id)

                    mem_mb = self._get_process_memory_mb() if PSUTIL_AVAILABLE else 0.0
                    self._log_metric_line({
                        "arxiv_id": arxiv_id,
                        "status": "failed",
                        "source": "cache_skipped",
                        "retryable": False,
                        "references_count": 0,
                        "memory_mb": mem_mb
                    })
                    return []
                else:
                    logger.warning(f"Cached status for {arxiv_id} is 'failed' (retryable). Re-fetching from API...")
            except Exception as e:
                logger.warning(f"Failed to load cache for {arxiv_id}: {e}. Re-fetching from API...")

        start_t = time.time()
        logger.info(f"Processing references for paper: {arxiv_id}")

        raw_refs, is_retryable, err_msg, attempts_used = self.fetch_references(arxiv_id)

        if raw_refs is not None:
            std_refs = self.standardize_references(arxiv_id, raw_refs)

            # Safeguard: preserve existing cached data if API transiently returns 0 references
            if not std_refs and output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    existing_refs = existing_data.get("references", []) if isinstance(existing_data, dict) else existing_data
                    if existing_refs:
                        logger.warning(f"API returned 0 references for {arxiv_id}. Preserving existing cached data ({len(existing_refs)} refs).")
                        std_refs = existing_refs
                except Exception:
                    pass

            self.save_paper_references(
                arxiv_id,
                std_refs,
                status="success",
                retryable=False,
                error_message=None,
                attempt=attempts_used
            )

            with self.metrics_lock:
                self.metrics["total_papers_processed"] += 1
                self.metrics["successful_crawls"] += 1
                self.metrics["total_references_fetched"] += len(std_refs)
                for ref in std_refs:
                    if ref.get("ref_arxiv_id"):
                        self.metrics["references_with_arxiv_id"] += 1
        else:
            error_msg = err_msg or f"Không thể crawl references cho {arxiv_id} sau các lần thử (FAIL)."
            logger.fail(error_msg)
            self.save_paper_references(
                arxiv_id,
                references=None,
                status="failed",
                retryable=is_retryable,
                error_message=error_msg,
                attempt=attempts_used
            )
            with self.metrics_lock:
                self.metrics["total_papers_processed"] += 1
                self.metrics["failed_crawls"] += 1
                if arxiv_id not in self.metrics["failed_arxiv_ids"]:
                    self.metrics["failed_arxiv_ids"].append(arxiv_id)
            std_refs = []

        mem_mb = self._get_process_memory_mb() if PSUTIL_AVAILABLE else 0.0
        if PSUTIL_AVAILABLE:
            with self.metrics_lock:
                self.metrics["memory_samples"].append(mem_mb)

        elapsed = time.time() - start_t
        if raw_refs is not None:
            logger.success(f"Crawl thành công {arxiv_id}: tìm thấy {len(std_refs)} references trong {elapsed:.2f}s")

        self._log_metric_line({
            "arxiv_id": arxiv_id,
            "status": "success" if raw_refs is not None else "failed",
            "retryable": is_retryable if raw_refs is None else False,
            "references_count": len(std_refs),
            "elapsed_seconds": elapsed,
            "attempts": attempts_used,
            "memory_mb": mem_mb
        })

        return std_refs

    def process_batch(self, arxiv_ids: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Process a batch of arXiv IDs concurrently using ThreadPoolExecutor with Bounded Submission
        (submitting at most max_workers * 2 tasks at a time to prevent RAM bloat on large corpora like 25,000 papers).
        Aggregates performance metrics, generates a unique run ID, and writes summary reports to disk.
        """
        import uuid
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        log_file = self.log_dir / "reference_crawler.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n[Run ID: {run_id}]\n")

        logger.info(f"Starting bounded multithreaded batch crawling for {len(arxiv_ids)} papers using {self.max_workers} worker(s)... [Run ID: {run_id}]")
        self.metrics["start_time"] = time.time()

        max_active = max(1, self.max_workers * 2)
        iter_ids = iter(arxiv_ids)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            # Prime initial batch of tasks up to max_active
            for _ in range(min(max_active, len(arxiv_ids))):
                try:
                    aid = next(iter_ids)
                    fut = executor.submit(self.process_single_paper, aid, force_refresh)
                    futures[fut] = aid
                except StopIteration:
                    break

            # As tasks complete, submit next tasks dynamically (bounded submission)
            while futures:
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    aid = futures.pop(fut)
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.fail(f"Paper {aid} generated an exception during multithreaded crawling: {exc}")

                    try:
                        next_aid = next(iter_ids)
                        new_fut = executor.submit(self.process_single_paper, next_aid, force_refresh)
                        futures[new_fut] = next_aid
                    except StopIteration:
                        pass

        self.metrics["end_time"] = time.time()

        total_time = self.metrics["end_time"] - self.metrics["start_time"]
        num_papers = max(1, self.metrics["total_papers_processed"])

        if PSUTIL_AVAILABLE and self.metrics["memory_samples"]:
            self.metrics["peak_memory_mb"] = max(self.metrics["memory_samples"])
            self.metrics["avg_memory_mb"] = sum(self.metrics["memory_samples"]) / len(self.metrics["memory_samples"])

        summary = {
            "run_id": run_id,
            "total_papers": num_papers,
            "successful_crawls": self.metrics["successful_crawls"],
            "failed_crawls": self.metrics["failed_crawls"],
            "failed_arxiv_ids": self.metrics["failed_arxiv_ids"],
            "total_references_collected": self.metrics["total_references_fetched"],
            "references_with_arxiv_id": self.metrics["references_with_arxiv_id"],
            "total_wall_time_seconds": total_time,
            "average_time_per_paper_seconds": total_time / num_papers,
            "peak_memory_mb": self.metrics["peak_memory_mb"],
            "average_memory_mb": self.metrics["avg_memory_mb"]
        }

        logger.success(f"Multithreaded batch crawling completed successfully! [Run ID: {run_id}]")

        # Save summary report to JSON file in log directory
        summary_path = self.log_dir / "crawler_performance_summary.json"
        existing_data = []
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        existing_data = content
                    else:
                        existing_data = [content]
            except Exception:
                existing_data = []

        if existing_data:
            existing_data.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        existing_data.append(summary)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        return summary
