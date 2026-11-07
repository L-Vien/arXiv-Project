"""
Module: reference_crawler.py
Description:
    Official Production-Grade Multithreaded Batch Reference Crawler for Milestone 1 of IntroDS Project.
    Utilizes Semantic Scholar Paper Batch API (POST /graph/v1/paper/batch) with ThreadPoolExecutor concurrency.
    Features:
    - Multi-Key Rotation & Per-Key Rate Limiting & 429 Adaptive Cooldown (APIKeyPool).
    - Separate retry budgets for 429 rate limit attempts vs transient network/server errors.
    - Strict Batch Response Length Validation.
    - Cache preservation on refresh failure.
    - Interruption-resilient atomic summary writing with completion_status and error_reason.
    - Asynchronous Streaming Metrics via Queue & Background Writer Thread.
"""

import os
import json
import time
import random
import logging
import threading
import queue
import concurrent.futures
import re
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


def _normalize_arxiv_id(aid: str) -> str:
    """Robustly normalize arXiv ID by removing version suffix."""
    aid = aid.strip()
    if "/" in aid:
        cat, rest = aid.split("/", 1)
        cleaned = re.sub(r"v\d+$", "", rest)
        return f"{cat}/{cleaned}"
    return re.sub(r"v\d+$", "", aid)


class APIKeyPool:
    """Thread-safe pool for rotating Semantic Scholar API keys with per-key rate limiting and 429 adaptive cooldown."""
    def __init__(self, api_keys: List[str]):
        if not api_keys or all(not k.strip() for k in api_keys):
            raise ValueError("Danh sách API Key rỗng hoặc không hợp lệ!")
        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.lock = threading.Lock()
        self.key_next_request = {k: 0.0 for k in self.api_keys}
        self.key_cooldown_until = {k: 0.0 for k in self.api_keys}
        self.consecutive_429 = {k: 0 for k in self.api_keys}

    def mark_cooldown(self, key: str, retry_after: Optional[str] = None):
        with self.lock:
            self.consecutive_429[key] += 1
            cooldown = None
            if retry_after:
                try:
                    cooldown = float(retry_after)
                except ValueError:
                    try:
                        from email.utils import parsedate_to_datetime
                        import datetime
                        dt = parsedate_to_datetime(retry_after)
                        cooldown = (dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                        if cooldown < 0:
                            cooldown = 2.0
                    except Exception:
                        cooldown = None

            if cooldown is None:
                cooldown = min(2.0 * (2 ** (self.consecutive_429[key] - 1)), 300.0) + random.uniform(0.1, 1.0)

            self.key_cooldown_until[key] = max(self.key_cooldown_until.get(key, 0.0), time.time() + cooldown)

    def mark_success(self, key: str):
        with self.lock:
            self.consecutive_429[key] = 0
            if self.key_cooldown_until.get(key, 0.0) <= time.time():
                self.key_cooldown_until[key] = 0.0

    def get_key_with_rate_limit(self, request_delay: float = 1.2) -> str:
        while True:
            with self.lock:
                now = time.time()
                best_key = min(self.api_keys, key=lambda k: max(self.key_next_request[k], self.key_cooldown_until[k]))
                available_at = max(self.key_next_request[best_key], self.key_cooldown_until[best_key])

                if available_at <= now:
                    self.key_next_request[best_key] = max(now, self.key_next_request[best_key]) + request_delay
                    return best_key
                else:
                    sleep_time = available_at - now
            time.sleep(sleep_time)


class ReferenceCrawler:
    """Multithreaded Batch Reference Crawler utilizing Semantic Scholar Paper Batch endpoint."""
    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        log_dir: str = "log",
        request_delay: float = 1.2,
        max_retries: int = 5,
        batch_size: int = 100,
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
            self.log_dir = data_parent / "log" if data_parent else (self.corpus_dir.parent.parent / "log" if len(self.corpus_dir.parts) >= 2 else Path("log"))
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.batch_size = batch_size

        log_file = self.log_dir / "reference_crawler.log"
        self.file_handler = logging.FileHandler(log_file, encoding="utf-8")
        self.file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(self.file_handler)

        raw_keys = []
        if api_key:
            raw_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        else:
            env_keys_dict = {k.upper(): v.strip() for k, v in os.environ.items() if (k.upper().startswith("S2_API_KEY_") or k.upper().startswith("SEMANTIC_SCHOLAR_API_KEY_")) and v.strip()}
            if env_keys_dict:
                raw_keys.extend([v for _, v in sorted(env_keys_dict.items(), key=lambda x: (len(x[0]), x[0]))])
            if not raw_keys:
                single_env = os.getenv("SEMANTIC_SCHOLAR_API_KEYS") or os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                if single_env:
                    raw_keys = [k.strip() for k in single_env.split(",") if k.strip()]

        if not raw_keys:
            raise ValueError("Lỗi: Không tìm thấy Semantic Scholar API Key trong .env hoặc môi trường!")

        self.key_pool = APIKeyPool(raw_keys)
        self.max_workers = max_workers if max_workers is not None else len(self.key_pool.api_keys)

        logger.info(f"Initialized APIKeyPool with {len(self.key_pool.api_keys)} key(s), max_workers={self.max_workers}.")

        self.thread_local = threading.local()
        self.base_headers = {
            "User-Agent": "IntroDS-ArXivReferenceCrawler/1.0 (Educational Project)",
            "Content-Type": "application/json"
        }

        self.metrics_queue = queue.Queue()
        self._stop_metrics_writer = threading.Event()
        self._metrics_writer_thread = threading.Thread(target=self._metrics_writer_worker, daemon=True)
        self._metrics_writer_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_session(self) -> requests.Session:
        if not hasattr(self.thread_local, "session"):
            session = requests.Session()
            session.headers.update(self.base_headers)
            self.thread_local.session = session
        return self.thread_local.session

    def close(self):
        self._stop_metrics_writer.set()
        try:
            self.metrics_queue.join()
        except Exception:
            pass
        if hasattr(self, '_metrics_writer_thread') and self._metrics_writer_thread.is_alive():
            try:
                self._metrics_writer_thread.join(timeout=5.0)
            except Exception:
                pass
        if hasattr(self, 'file_handler') and self.file_handler:
            try:
                self.file_handler.close()
                if self.file_handler in logger.handlers:
                    logger.removeHandler(self.file_handler)
            except Exception:
                pass

    def _get_process_memory_mb(self) -> float:
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _metrics_writer_worker(self):
        metrics_file = self.log_dir / "reference_metrics.jsonl"
        while not self._stop_metrics_writer.is_set() or not self.metrics_queue.empty():
            try:
                try:
                    record = self.metrics_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                with open(metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.metrics_queue.task_done()
            except Exception as e:
                logger.error(f"Failed to write metric line: {e}")

    def _log_metric_line(self, record: Dict[str, Any]):
        self.metrics_queue.put(record)

    def _get_folder_name(self, arxiv_id: str) -> str:
        return arxiv_id.replace('/', '-').replace('.', '-')

    def standardize_references(self, arxiv_id: str, raw_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        std_list = []
        for ref in raw_references:
            if not isinstance(ref, dict):
                continue
            external_ids = ref.get("externalIds", {}) or {}
            std_list.append({
                "arxiv_id": arxiv_id,
                "ref_arxiv_id": external_ids.get("ArXiv") if isinstance(external_ids, dict) else None,
                "ref_title": ref.get("title"),
                "ref_authors": [a.get("name", "") for a in ref.get("authors", []) if isinstance(a, dict) and a.get("name")],
                "ref_submission_date": ref.get("publicationDate") or str(ref.get("year", "")),
                "ref_s2_id": ref.get("paperId")
            })
        return std_list

    def save_paper_references(
        self,
        arxiv_id: str,
        references: Optional[List[Dict[str, Any]]],
        status: str = "success",
        retryable: bool = False,
        error_message: Optional[str] = None,
        attempt: int = 1
    ) -> Path:
        folder_name = self._get_folder_name(arxiv_id)
        paper_dir = self.corpus_dir / folder_name
        paper_dir.mkdir(parents=True, exist_ok=True)
        output_file = paper_dir / "references.json"

        # Cache preservation on refresh failure
        if status in ("retryable_failure", "permanent_failure") and output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and existing.get("status") == "success" and existing.get("references"):
                    fail_record = paper_dir / "last_failure.json"
                    with open(fail_record, "w", encoding="utf-8") as ff:
                        json.dump({"status": status, "retryable": retryable, "error_message": error_message, "attempt": attempt, "timestamp": time.time()}, ff, ensure_ascii=False, indent=2)
                    return output_file
            except Exception:
                pass

        data = {
            "schema_version": 1,
            "crawler_version": "1.0.0",
            "arxiv_id": arxiv_id,
            "status": status,
            "retryable": retryable,
            "error_message": error_message,
            "attempt": attempt,
            "timestamp": time.time(),
            "references_count": len(references) if references is not None else 0,
            "references": references if references is not None else []
        }

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
        clean_ids = [f"ARXIV:{_normalize_arxiv_id(aid)}" for aid in arxiv_ids]
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {"fields": "references.title,references.authors,references.year,references.publicationDate,references.externalIds"}
        body = {"ids": clean_ids}

        transient_attempts = 0
        rate_limit_attempts = 0
        max_rate_limit_attempts = 2 * len(self.key_pool.api_keys)

        while transient_attempts < self.max_retries:
            key = self.key_pool.get_key_with_rate_limit(self.request_delay)
            headers = {**self.base_headers, "x-api-key": key}

            try:
                response = self._get_session().post(url, params=params, json=body, headers=headers, timeout=45)

                if response.status_code in (401, 403, 400, 404):
                    err = f"Permanent HTTP error {response.status_code}. Do not retry."
                    logger.fail(err)
                    return None, False, err, transient_attempts + 1

                if response.status_code == 429:
                    rate_limit_attempts += 1
                    retry_after = response.headers.get("Retry-After")
                    self.key_pool.mark_cooldown(key, retry_after)
                    logger.warning(f"[429 Rate Limit] Batch hit limit with key {key[:6]}... Rotating key and retrying ({rate_limit_attempts}/{max_rate_limit_attempts})...")
                    if rate_limit_attempts >= max_rate_limit_attempts:
                        err = f"Exceeded max rate limit attempts ({max_rate_limit_attempts})."
                        logger.fail(err)
                        return None, True, err, transient_attempts + 1
                    continue

                if 500 <= response.status_code <= 599:
                    transient_attempts += 1
                    err = f"Server error {response.status_code}"
                    if transient_attempts >= self.max_retries:
                        return None, True, err, transient_attempts
                    time.sleep(min(2 * transient_attempts, 6) + random.uniform(0.1, 0.5))
                    continue

                response.raise_for_status()
                self.key_pool.mark_success(key)

                try:
                    data = response.json()
                except (requests.exceptions.JSONDecodeError, json.JSONDecodeError) as jde:
                    transient_attempts += 1
                    if transient_attempts >= self.max_retries:
                        return None, True, f"JSONDecodeError: {jde}", transient_attempts
                    time.sleep(2.0 + random.uniform(0.1, 0.5))
                    continue

                if not isinstance(data, list) or len(data) != len(clean_ids):
                    transient_attempts += 1
                    err = f"Malformed or truncated batch response (expected {len(clean_ids)}, got {len(data) if isinstance(data, list) else type(data)})"
                    if transient_attempts >= self.max_retries:
                        return None, True, err, transient_attempts
                    time.sleep(2.0 + random.uniform(0.1, 0.5))
                    continue

                return [item if isinstance(item, dict) else None for item in data], False, None, transient_attempts + 1

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                transient_attempts += 1
                if transient_attempts >= self.max_retries:
                    err = f"Network/HTTP error after {self.max_retries} attempts: {e}"
                    logger.fail(err)
                    return None, True, err, transient_attempts
                time.sleep(min(2 * transient_attempts, 6) + random.uniform(0.1, 0.5))
            except Exception as e:
                logger.exception(f"Unexpected error in fetch_batch: {e}")
                return None, False, f"Unexpected error: {e}", transient_attempts + 1

        return None, True, "Max transient retries reached", self.max_retries

    def _save_summary(self, summary: Dict[str, Any]):
        try:
            summary_path = self.log_dir / "crawler_performance_summary.jsonl"
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save summary JSONL: {e}")

        try:
            summary_json_path = self.log_dir / "crawler_performance_summary.json"
            temp_json_path = summary_json_path.with_suffix(".json.tmp")
            existing = []
            if summary_json_path.exists():
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    existing = [item for item in (content if isinstance(content, list) else [content]) if item != "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"]
            existing.append(summary)
            with open(temp_json_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            os.replace(temp_json_path, summary_json_path)
        except Exception as e:
            logger.error(f"Failed to save summary JSON: {e}")

    def process_batch(self, arxiv_ids: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        import uuid
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        empty_summary = {
            "run_id": run_id, "completion_status": "completed", "error_reason": None,
            "total_papers": 0, "successful_crawls": 0, "not_found_count": 0, "failed_crawls": 0,
            "failed_arxiv_ids": [], "total_references_collected": 0, "references_with_arxiv_id": 0,
            "total_wall_time_seconds": 0.0, "average_time_per_paper_seconds": 0.0,
            "peak_memory_mb": 0.0, "average_memory_mb": 0.0
        }

        if not arxiv_ids:
            self._save_summary(empty_summary)
            return empty_summary

        unique_arxiv_ids = list(dict.fromkeys([_normalize_arxiv_id(aid) for aid in arxiv_ids]))
        start_time = time.time()
        memory_samples = []
        run_metrics_lock = threading.Lock()

        log_file = self.log_dir / "reference_crawler.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n[Run ID: {run_id}]\n")

        logger.info(f"Starting crawl for {len(unique_arxiv_ids)} unique papers with {self.max_workers} worker(s)... [Run ID: {run_id}]")

        completion_status = "completed"
        error_reason = None

        try:
            pending_ids = []
            for aid in unique_arxiv_ids:
                output_file = self.corpus_dir / self._get_folder_name(aid) / "references.json"
                if not force_refresh and output_file.exists():
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            cache = json.load(f)
                        if isinstance(cache, dict) and cache.get("schema_version") == 1:
                            status = cache.get("status")
                            if status == "success":
                                std_refs = self.standardize_references(aid, cache.get("references", []))
                                logger.success(f"Loaded cached references for {aid} (skipping API call).")
                                mem_mb = self._get_process_memory_mb()
                                with run_metrics_lock:
                                    memory_samples.append(mem_mb)
                                self._log_metric_line({"run_id": run_id, "arxiv_id": aid, "status": "success", "source": "cache", "retryable": False, "references_count": len(std_refs), "memory_mb": mem_mb})
                                continue
                            elif status == "not_found":
                                with run_metrics_lock:
                                    memory_samples.append(self._get_process_memory_mb())
                                continue
                    except Exception:
                        pass
                pending_ids.append(aid)

            batches = [pending_ids[i:i + self.batch_size] for i in range(0, len(pending_ids), self.batch_size)]

            def process_single_batch(batch_idx: int, batch: List[str]):
                start_t = time.time()
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} containing {len(batch)} paper(s)...")
                raw_results, is_retryable, err_msg, attempts = self.fetch_batch(batch)
                elapsed = time.time() - start_t
                mem_mb = self._get_process_memory_mb()

                with run_metrics_lock:
                    memory_samples.append(mem_mb)

                if raw_results is not None:
                    for i, aid in enumerate(batch):
                        paper_data = raw_results[i] if i < len(raw_results) else None
                        if paper_data and isinstance(paper_data, dict):
                            std_refs = self.standardize_references(aid, paper_data.get("references", []))
                            self.save_paper_references(aid, std_refs, status="success", retryable=False, attempt=attempts)
                            logger.success(f"Crawl thành công {aid}: tìm thấy {len(std_refs)} references trong {elapsed:.2f}s")
                            self._log_metric_line({"run_id": run_id, "arxiv_id": aid, "status": "success", "retryable": False, "references_count": len(std_refs), "elapsed_seconds": elapsed, "attempts": attempts, "memory_mb": mem_mb})
                        else:
                            logger.warning(f"Paper {aid} not found or null in batch response.")
                            self.save_paper_references(aid, [], status="not_found", retryable=False, error_message="Not found", attempt=attempts)
                            self._log_metric_line({"run_id": run_id, "arxiv_id": aid, "status": "not_found", "retryable": False, "references_count": 0, "elapsed_seconds": elapsed, "attempts": attempts, "memory_mb": mem_mb})
                else:
                    fail_status = "retryable_failure" if is_retryable else "permanent_failure"
                    for aid in batch:
                        self.save_paper_references(aid, None, status=fail_status, retryable=is_retryable, error_message=err_msg, attempt=attempts)
                        self._log_metric_line({"run_id": run_id, "arxiv_id": aid, "status": fail_status, "retryable": is_retryable, "references_count": 0, "elapsed_seconds": elapsed, "attempts": attempts, "memory_mb": mem_mb})

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_single_batch, idx, batch) for idx, batch in enumerate(batches)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Worker exception: {e}")

        except KeyboardInterrupt:
            completion_status = "interrupted_by_user"
            error_reason = "KeyboardInterrupt (Ctrl+C)"
            raise
        except Exception as exc:
            completion_status = "failed_with_exception"
            error_reason = str(exc)
            raise
        finally:
            end_time = time.time()
            total_time = end_time - start_time

            # Compute definitive metrics from corpus state on disk
            success_count, not_found_count, failed_count = 0, 0, 0
            failed_ids = []
            total_refs = 0
            refs_with_arxiv = 0

            for aid in unique_arxiv_ids:
                output_file = self.corpus_dir / self._get_folder_name(aid) / "references.json"
                if output_file.exists():
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        st = data.get("status")
                        if st == "success":
                            success_count += 1
                            refs = data.get("references", [])
                            total_refs += len(refs)
                            refs_with_arxiv += sum(1 for r in refs if isinstance(r, dict) and r.get("ref_arxiv_id"))
                        elif st == "not_found":
                            not_found_count += 1
                        else:
                            failed_count += 1
                            failed_ids.append(aid)
                    except Exception:
                        failed_count += 1
                        failed_ids.append(aid)
                else:
                    failed_count += 1
                    failed_ids.append(aid)

            num_papers = max(1, len(unique_arxiv_ids))
            summary = {
                "run_id": run_id,
                "completion_status": completion_status,
                "error_reason": error_reason,
                "total_papers": num_papers,
                "successful_crawls": success_count,
                "not_found_count": not_found_count,
                "failed_crawls": failed_count,
                "failed_arxiv_ids": failed_ids,
                "total_references_collected": total_refs,
                "references_with_arxiv_id": refs_with_arxiv,
                "total_wall_time_seconds": total_time,
                "average_time_per_paper_seconds": total_time / num_papers,
                "peak_memory_mb": max(memory_samples) if memory_samples else 0.0,
                "average_memory_mb": sum(memory_samples) / len(memory_samples) if memory_samples else 0.0
            }

            self._save_summary(summary)
            logger.info(f"Crawl run finished. Status: {completion_status}. Summary saved. [Run ID: {run_id}]")

        return summary