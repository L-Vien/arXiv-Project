"""
Module: reference_crawler.py
Description:
    Official Production-Grade Multithreaded Batch Reference Crawler for Milestone 1 of IntroDS Project.
    Utilizes Semantic Scholar Paper Batch API (POST /graph/v1/paper/batch) with ThreadPoolExecutor concurrency.
    Features:
    - Multi-Key Rotation & Per-Key Rate Limiting (APIKeyPool): Automatically rotates API keys and ensures per-key rate limit compliance (1 req/sec).
    - Multithreaded Batch Processing (ThreadPoolExecutor): Concurrent batch requests across multiple worker threads.
    - Robust Caching / Resume mechanism: Inspects per-paper local cache and skips re-fetching if valid cache exists.
    - Atomic File Writing: Saves per-paper references safely using .tmp and os.replace.
    - Streaming Metrics (reference_metrics.jsonl): Appends per-paper metrics line-by-line.
    - Thread-Local Session: Ensures thread-safe HTTP connection pooling across workers.
    - Performance Tracking & Custom Logging (SUCCESS, FAIL, INFO, ERROR).
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

# Robust manual .env loading
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

# Custom log levels: SUCCESS (25) and FAIL (45)
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

logger = logging.getLogger("ReferenceCrawler")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console_handler)


class APIKeyPool:
    """
    Thread-safe pool for rotating multiple Semantic Scholar API keys in a round-robin fashion,
    with per-key rate limiting to ensure compliance with API rate limits (1 req/sec per key).
    """
    def __init__(self, api_keys: List[str]):
        if not api_keys or all(not k.strip() for k in api_keys):
            raise ValueError("Danh sách API Key rỗng hoặc không hợp lệ!")
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.index = 0
        self.lock = threading.Lock()
        self.key_last_request = {k: 0.0 for k in self.api_keys}

    def get_key_with_rate_limit(self, request_delay: float = 1.0) -> str:
        with self.lock:
            key = self.api_keys[self.index]
            self.index = (self.index + 1) % len(self.api_keys)

            now = time.time()
            elapsed = now - self.key_last_request[key]
            if elapsed < request_delay:
                sleep_time = request_delay - elapsed
                self.key_last_request[key] = now + sleep_time
            else:
                sleep_time = 0.0
                self.key_last_request[key] = now

        if sleep_time > 0:
            time.sleep(sleep_time)
        return key


class ReferenceCrawler:
    """
    Official Production-Grade Multithreaded Batch Reference Crawler utilizing Semantic Scholar Paper Batch endpoint (POST /graph/v1/paper/batch).
    Supports multi-key rotation, per-key rate limiting, concurrent batch execution via ThreadPoolExecutor,
    per-paper caching, atomic saves, and streaming metrics.
    """
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        log_dir: str = "log",
        request_delay: float = 1.2,
        max_retries: int = 5,
        batch_size: int = 25,
        api_key: Optional[str] = None,
        max_workers: Optional[int] = None
    ):
        self.corpus_dir = Path(corpus_dir)
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
        self.batch_size = batch_size

        # Setup file logging (reference_crawler.log inside log directory)
        log_file = self.log_dir / "reference_crawler.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

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
                sorted_env_keys = [v for _, v in sorted(env_keys_dict.items(), key=lambda x: (len(x[0]), x[0]))]
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

        self.key_pool = APIKeyPool(raw_keys)
        self.max_workers = max_workers if max_workers is not None else len(self.key_pool.api_keys)

        logger.info(f"Semantic Scholar API Key Pool initialized with {len(self.key_pool.api_keys)} key(s).")
        logger.info(f"Reference Crawler configured with batch_size={self.batch_size}, max_workers={self.max_workers}.")
        logger.info(f"Log directory set to: {self.log_dir.absolute()}")

        self.thread_local = threading.local()
        self.base_headers = {
            "User-Agent": "IntroDS-ArXivReferenceCrawler/1.0 (Educational Project)",
            "Content-Type": "application/json"
        }

        self.metrics_lock = threading.Lock()
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

    def _get_session(self) -> requests.Session:
        if not hasattr(self.thread_local, "session"):
            session = requests.Session()
            session.headers.update(self.base_headers)
            self.thread_local.session = session
        return self.thread_local.session

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
        Normalize arXiv ID into folder name matching the Processor module:
        Replaces both '/' and '.' with '-' (e.g. 2604.25535 -> 2604-25535).
        """
        return arxiv_id.replace('/', '-').replace('.', '-')

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

    def fetch_batch(self, arxiv_ids: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[str], int]:
        """
        Fetch a batch of papers using POST https://api.semanticscholar.org/graph/v1/paper/batch
        Implements per-key rate limiting, automatic key rotation, and 429 rate limit handling.
        Returns: (response_json_list, retryable, error_message, attempts_used)
        """
        clean_ids = [f"ARXIV:{aid.split('v')[0] if 'v' in aid and not aid.startswith('hep') else aid}" for aid in arxiv_ids]
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {
            "fields": "references.title,references.authors,references.year,references.publicationDate,references.venue,references.externalIds"
        }
        body = {"ids": clean_ids}

        max_429_retries = 3
        retries_429 = 0

        for attempt in range(1, self.max_retries + 1):
            current_api_key = self.key_pool.get_key_with_rate_limit(self.request_delay)
            headers = {**self.base_headers, "x-api-key": current_api_key}

            try:
                session = self._get_session()
                response = session.post(url, params=params, json=body, headers=headers, timeout=45)

                if response.status_code == 429:
                    retries_429 += 1
                    if retries_429 > max_429_retries:
                        err = f"Exceeded max 429 retries for batch of size {len(arxiv_ids)}."
                        logger.fail(err)
                        return None, True, err, attempt
                    logger.warning(f"[429 Rate Limit] Batch hit limit with key {current_api_key[:6]}... Rotating key and retrying ({retries_429}/{max_429_retries})...")
                    time.sleep(2.0)
                    continue

                response.raise_for_status()
                data = response.json()
                return data, False, None, attempt

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching batch using key {current_api_key[:6]}... (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    err = f"Failed batch request after {self.max_retries} attempts: {e}"
                    logger.fail(err)
                    return None, True, err, attempt
                time.sleep(min(2 * attempt, 6))

        return None, True, "Max retries reached", self.max_retries

    def process_batch(self, arxiv_ids: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Process a list of arXiv IDs using the batch endpoint concurrently via ThreadPoolExecutor.
        Checks cache first, fetches missing items in parallel batches, standardizes, saves atomically,
        updates thread-safe metrics, and streams metric lines to reference_metrics.jsonl.
        """
        import uuid
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        log_file = self.log_dir / "reference_crawler.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n[Run ID: {run_id}]\n")

        logger.info(f"Starting multithreaded batch reference crawling for {len(arxiv_ids)} papers using {self.max_workers} worker(s) (Batch Size: {self.batch_size})... [Run ID: {run_id}]")
        self.metrics["start_time"] = time.time()

        # Check cache and resume
        pending_ids = []
        for aid in arxiv_ids:
            folder_name = self._get_folder_name(aid)
            output_file = self.corpus_dir / folder_name / "references.json"
            if not force_refresh and output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    status = cache_data.get("status") if isinstance(cache_data, dict) else "success"
                    if status == "success":
                        logger.success(f"Loaded cached references for {aid} (skipping API call).")
                        with self.metrics_lock:
                            self.metrics["total_papers_processed"] += 1
                            self.metrics["successful_crawls"] += 1
                            refs = cache_data.get("references", []) if isinstance(cache_data, dict) else cache_data
                            self.metrics["total_references_fetched"] += len(refs)

                        mem_mb = self._get_process_memory_mb()
                        self._log_metric_line({
                            "arxiv_id": aid,
                            "status": "success",
                            "source": "cache",
                            "retryable": False,
                            "references_count": len(refs),
                            "memory_mb": mem_mb
                        })
                        continue
                except Exception:
                    pass
            pending_ids.append(aid)

        # Chunk pending IDs into batches
        batches = [pending_ids[i:i + self.batch_size] for i in range(0, len(pending_ids), self.batch_size)]

        def process_single_batch(batch_idx: int, batch: List[str]):
            start_t = time.time()
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} containing {len(batch)} paper(s)...")

            raw_results, is_retryable, err_msg, attempts = self.fetch_batch(batch)
            elapsed = time.time() - start_t
            mem_mb = self._get_process_memory_mb()

            if raw_results is not None:
                for i, aid in enumerate(batch):
                    paper_data = raw_results[i] if i < len(raw_results) else None
                    if paper_data and isinstance(paper_data, dict):
                        raw_refs = paper_data.get("references", [])
                        std_refs = self.standardize_references(aid, raw_refs)

                        # Safeguard: preserve existing cached data if API transiently returns 0 references
                        folder_name = self._get_folder_name(aid)
                        output_file = self.corpus_dir / folder_name / "references.json"
                        if not std_refs and output_file.exists():
                            try:
                                with open(output_file, "r", encoding="utf-8") as f:
                                    existing_data = json.load(f)
                                existing_refs = existing_data.get("references", []) if isinstance(existing_data, dict) else existing_data
                                if existing_refs:
                                    logger.warning(f"API returned 0 references for {aid}. Preserving existing cached data ({len(existing_refs)} refs).")
                                    std_refs = existing_refs
                            except Exception:
                                pass

                        self.save_paper_references(
                            aid,
                            std_refs,
                            status="success",
                            retryable=False,
                            error_message=None,
                            attempt=attempts
                        )

                        with self.metrics_lock:
                            self.metrics["total_papers_processed"] += 1
                            self.metrics["successful_crawls"] += 1
                            self.metrics["total_references_fetched"] += len(std_refs)
                            self.metrics["memory_samples"].append(mem_mb)

                        logger.success(f"Crawl thành công {aid}: tìm thấy {len(std_refs)} references trong {elapsed:.2f}s")
                        self._log_metric_line({
                            "arxiv_id": aid,
                            "status": "success",
                            "retryable": False,
                            "references_count": len(std_refs),
                            "elapsed_seconds": elapsed,
                            "attempts": attempts,
                            "memory_mb": mem_mb
                        })
                    else:
                        logger.warning(f"Paper {aid} not found or null in batch response.")
                        self.save_paper_references(
                            aid,
                            [],
                            status="success",
                            retryable=False,
                            error_message="Paper not found in batch response",
                            attempt=attempts
                        )
                        with self.metrics_lock:
                            self.metrics["total_papers_processed"] += 1
                            self.metrics["successful_crawls"] += 1
                            self.metrics["memory_samples"].append(mem_mb)

                        self._log_metric_line({
                            "arxiv_id": aid,
                            "status": "success",
                            "retryable": False,
                            "references_count": 0,
                            "elapsed_seconds": elapsed,
                            "attempts": attempts,
                            "memory_mb": mem_mb
                        })
            else:
                logger.fail(f"Batch {batch_idx + 1} failed: {err_msg}")
                for aid in batch:
                    self.save_paper_references(
                        aid,
                        references=None,
                        status="failed",
                        retryable=is_retryable,
                        error_message=err_msg,
                        attempt=attempts
                    )
                    with self.metrics_lock:
                        self.metrics["total_papers_processed"] += 1
                        self.metrics["failed_crawls"] += 1
                        if aid not in self.metrics["failed_arxiv_ids"]:
                            self.metrics["failed_arxiv_ids"].append(aid)
                        self.metrics["memory_samples"].append(mem_mb)

                    self._log_metric_line({
                        "arxiv_id": aid,
                        "status": "failed",
                        "retryable": is_retryable,
                        "references_count": 0,
                        "elapsed_seconds": elapsed,
                        "attempts": attempts,
                        "memory_mb": mem_mb
                    })

        # Run batch requests concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_batch, idx, batch) for idx, batch in enumerate(batches)]
            concurrent.futures.wait(futures)

        self.metrics["end_time"] = time.time()
        total_time = self.metrics["end_time"] - self.metrics["start_time"]
        num_papers = max(1, self.metrics["total_papers_processed"])

        if self.metrics["memory_samples"]:
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

        logger.success(f"Multithreaded batch reference crawling completed successfully! [Run ID: {run_id}]")

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