from pathlib import Path 
from urllib.request import Request, urlopen
import time 
import gzip 
import zlib
import re 
import json
import tarfile
import hashlib
import threading
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from datetime import datetime, timezone 
from email.utils import parsedate_to_datetime
from collections.abc import Callable
from src.crawling.config import settings
from src.crawling.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://export.arxiv.org/src"
PDF_BASE_URL = "https://export.arxiv.org/pdf"
VERSIONED_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}v[1-9]\d*")

class RateLimiter:
    def __init__(
        self,
        min_interval: float = 3.0,
        *,
        max_interval: float | None = None,
        success_window: int = 8,
        recovery_factor: float = 0.9,
    ):
        if min_interval < 0: 
            raise ValueError("min_interval must be non-negative")
        if max_interval is not None and max_interval < min_interval:
            raise ValueError("max_interval must not be smaller than min_interval")
        if success_window < 1:
            raise ValueError("success_window must be positive")
        if not 0 < recovery_factor <= 1:
            raise ValueError("recovery_factor must be in the range (0, 1]")

        self.min_interval = min_interval
        self.max_interval = max(
            min_interval,
            max_interval if max_interval is not None else max(60.0, min_interval * 16),
        )
        self.current_interval = min_interval
        self.success_window = success_window
        self.recovery_factor = recovery_factor
        self._success_streak = 0
        self.last_request_time = None
        self.cooldown_until = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            next_allowed = self.cooldown_until

            if self.last_request_time is not None:
                next_allowed = max(
                    next_allowed,
                    self.last_request_time + self.current_interval
                )

            remaining = next_allowed - now

            if remaining > 0:
                time.sleep(remaining)

            self.last_request_time = time.monotonic()
            return max(0.0, self.last_request_time - now)

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            self.cooldown_until = max(
                self.cooldown_until,
                time.monotonic() + max(0.0, seconds)
            )
            self.current_interval = min(
                self.max_interval,
                max(self.min_interval, self.current_interval * 2),
            )
            self._success_streak = 0

    def record_success(self) -> None:
        """Recover the pacing floor slowly after a stable success streak."""
        with self._lock:
            self._success_streak += 1
            if self._success_streak < self.success_window:
                return
            self.current_interval = max(
                self.min_interval,
                self.current_interval * self.recovery_factor,
            )
            self._success_streak = 0

DEFAULT_RATE_LIMITER = RateLimiter(min_interval=settings.arxiv_api_delay)

class IncompleteDownloadError(OSError): 
    """Raised when downloaded source data is empty or incomplete."""
    pass

class InvalidSourceFormatError(OSError):
    """Raised when the downloaded file is not a supported source format."""
    pass


def validate_versioned_id(versioned_id: str) -> str: 
    if (not isinstance(versioned_id, str) or not VERSIONED_ID_PATTERN.fullmatch(versioned_id)):
        raise ValueError(f"Invalid versioned arXiv ID: {versioned_id!r}")
    return versioned_id

def build_versioned_ids(metadata: dict) -> list[str]:
    versioned_ids = []
    # lấy arxiv_id từ metadata
    arxiv_id = metadata["arxiv_id"]

    # lấy danh sách versions từ metadata
    versions = metadata["versions"]

    if (not isinstance(arxiv_id, str) or not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id)):
        raise ValueError("Invalid base arXiv ID")

    if not isinstance(versions, list): 
        raise ValueError("versions must be a list")

    # ghép arxiv_id với từng version
    for version in versions: 
        version_id = validate_versioned_id(f"{arxiv_id}{version}")
        versioned_ids.append(version_id)

    return versioned_ids


def build_source_url(versioned_id: str) -> str: 
    versioned_id = validate_versioned_id(versioned_id)
    return f"{BASE_URL}/{versioned_id}"

def build_pdf_url(versioned_id: str) -> str:
    versioned_id = validate_versioned_id(versioned_id)
    return f"{PDF_BASE_URL}/{versioned_id}"

def calculate_sha256(path: str | Path) -> str:
    """Calculate SHA-256 using fixed-size chunks."""
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def write_integrity_marker(source_path: str | Path) -> None:
    """Write an atomic completion marker with size and SHA-256."""
    source_path = Path(source_path)

    marker = Path(f"{source_path}.ok")
    temp_marker = Path(f"{source_path}.ok.tmp")

    metadata = {
        "status": "downloaded",
        "size_bytes": source_path.stat().st_size,
        "mtime_ns": source_path.stat().st_mtime_ns,
        "sha256": calculate_sha256(source_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        temp_marker.write_text(
            json.dumps(metadata, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temp_marker.replace(marker)

    finally:
        temp_marker.unlink(missing_ok=True)

def _verify_file_integrity(
    path: Path,
    marker: Path,
    *,
    fast: bool = False,
) -> bool | None:
    """
    True: checksum matches.
    False: file or marker is invalid, or checksum mismatches.
    None: legacy marker without checksum.
    """
    if (
        not path.is_file()
        or not marker.is_file()
        or path.stat().st_size == 0
    ):
        return False

    try:
        value = marker.read_text(encoding="utf-8")

        # Backward compatibility
        if value.strip() == "downloaded":
            return None

        metadata = json.loads(value)

        if not isinstance(metadata, dict):
            return False

        if metadata.get("status") != "downloaded":
            return False

        expected_size = metadata.get("size_bytes")
        expected_sha256 = metadata.get("sha256")

        if type(expected_size) is not int or expected_size <= 0:
            return False

        if expected_size != path.stat().st_size:
            return False

        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            return False

        if fast:
            expected_mtime = metadata.get("mtime_ns")
            if (
                type(expected_mtime) is int
                and expected_mtime == path.stat().st_mtime_ns
            ):
                return True

        return calculate_sha256(path) == expected_sha256

    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def verify_integrity(
    versioned_id: str,
    output_dir: str | Path,
    *,
    fast: bool = False,
) -> bool | None:
    validate_versioned_id(versioned_id)
    output_dir = Path(output_dir)
    return _verify_file_integrity(
        output_dir / f"{versioned_id}.source",
        output_dir / f"{versioned_id}.source.ok",
        fast=fast,
    )


def verify_pdf_integrity(
    versioned_id: str,
    output_dir: str | Path,
    *,
    fast: bool = False,
) -> bool | None:
    validate_versioned_id(versioned_id)
    output_dir = Path(output_dir)
    return _verify_file_integrity(
        output_dir / f"{versioned_id}.pdf",
        output_dir / f"{versioned_id}.pdf.ok",
        fast=fast,
    )


def _download_file(
    versioned_id: str,
    url: str,
    output_path: Path,
    validator: Callable[[str | Path, str], None],
    limiter: RateLimiter,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{output_path}.part")
    Path(f"{output_path}.ok").unlink(missing_ok=True)
    temp_path.unlink(missing_ok=True)

    request = Request(url, headers={"User-Agent": settings.user_agent})
    try:
        limiter.wait()
        with urlopen(request, timeout=settings.request_timeout) as response:
            content_length = response.headers.get("Content-Length")
            expected_size = (
                int(content_length)
                if content_length is not None and content_length.isdecimal()
                else None
            )
            bytes_downloaded = 0
            with temp_path.open("wb") as file:
                while True:
                    try:
                        chunk = response.read(64 * 1024)
                    except (IncompleteRead, ConnectionError) as exc:
                        raise IncompleteDownloadError(
                            f"Network interrupted while downloading {versioned_id}"
                        ) from exc
                    if not chunk:
                        break
                    file.write(chunk)
                    bytes_downloaded += len(chunk)

        if bytes_downloaded == 0:
            raise IncompleteDownloadError("Empty download")
        if expected_size is not None and bytes_downloaded != expected_size:
            raise IncompleteDownloadError(
                f"Incomplete download: expected {expected_size} bytes, "
                f"got {bytes_downloaded} bytes"
            )

        validator(temp_path, versioned_id)
        temp_path.replace(output_path)
        write_integrity_marker(output_path)
        limiter.record_success()
        return output_path
    except HTTPError as exc:
        logger.warning("HTTP error | version=%s | code=%s", versioned_id, exc.code)
        raise
    except URLError as exc:
        logger.warning("URL error | version=%s | reason=%s", versioned_id, exc.reason)
        raise
    except TimeoutError as exc:
        logger.warning("Timeout | version=%s | error=%s", versioned_id, exc)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

def download_version(versioned_id: str, output_dir: str | Path, limiter: RateLimiter | None = None) -> Path:
    if limiter is None:
            limiter = DEFAULT_RATE_LIMITER
    output_dir = Path(output_dir)
    output_path = output_dir / f"{versioned_id}.source"
    if is_download_complete(versioned_id, output_dir, fast=True):
        return output_path
    return _download_file(
        versioned_id,
        build_source_url(versioned_id),
        output_path,
        validate_source,
        limiter,
    )


def _validate_pdf(path: str | Path, versioned_id: str) -> None:
    with Path(path).open("rb") as file:
        if not file.read(5).startswith(b"%PDF-"):
            raise InvalidSourceFormatError(
                f"Invalid PDF response for {versioned_id}"
            )


def download_pdf(
    versioned_id: str,
    output_dir: str | Path,
    limiter: RateLimiter | None = None,
) -> Path:
    if limiter is None:
        limiter = DEFAULT_RATE_LIMITER
    output_dir = Path(output_dir)
    output_path = output_dir / f"{versioned_id}.pdf"
    if verify_pdf_integrity(versioned_id, output_dir, fast=True) is True:
        return output_path
    return _download_file(
        versioned_id,
        build_pdf_url(versioned_id),
        output_path,
        _validate_pdf,
        limiter,
    )


def validate_source(path: str | Path, versioned_id: str) -> None: 
    path = Path(path)
    source_format = detect_source_format(path)

    if source_format == "gzip":
        validate_gzip(path)

        # Check whether decompressed content is HTML
        with gzip.open(path, "rb") as gz_file:
            header = gz_file.read(512)
            prefix = header.lstrip().lower()

            if (
                prefix.startswith(b"<html")
                or prefix.startswith(b"<!doctype html")
            ):
                raise InvalidSourceFormatError(
                    "Unexpected HTML inside GZIP source"
                )

        # Check whether the GZIP contains a TAR archive
        is_tar = (
            header[257:262] == b"ustar"
            or tarfile.is_tarfile(path)
        )

        if is_tar:
            validate_tar(path)

    elif source_format == "tar":
        validate_tar(path)

    elif source_format == "pdf":
        logger.warning(
            "Non-TeX source | version=%s | format=pdf",
            versioned_id,
        )

    elif source_format == "tex":
        logger.info(
            "Plain TeX source detected | version=%s",
            versioned_id,
        )

    elif source_format == "html":
        raise InvalidSourceFormatError("Unexpected HTML response")

    else:
        raise InvalidSourceFormatError(
            f"Unrecognized source format: {source_format}"
        )

# quyết định có nên gọi lại download_version() nếu thất bại hay không
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
def calculate_backoff(attempt: int) -> float: 
    return 3 * (2 ** (attempt - 1))

def is_retryable_http_error(status_code: int) -> bool: 
    return status_code in RETRYABLE_HTTP_CODES

def parse_retry_after(value: str | None) -> float | None: 
    if value is None: 
        return None

    value = value.strip()

    # case 1: retry-after is a number of seconds
    if value.isdecimal(): 
        return float(int(value))

    # case 2: retry-after is an HTTP-date
    # parse value into a datetime object
    try: 
        retry_time = parsedate_to_datetime(value)
    except (ValueError, TypeError):
        return None

    # check whether retry_time has timezone information
    if retry_time.tzinfo is None: 
        return None

    now = datetime.now(timezone.utc)
    remaining_seconds = (retry_time - now).total_seconds()

    # return a non-negative dalay
    return max(0.0, remaining_seconds)

def download_with_retry(versioned_id: str, output_dir: str | Path, max_attempts: int = settings.max_retries + 1) -> Path: 
    if max_attempts < 1: 
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try: 
            # call download_version() 
            # return immediately if successful
            return download_version(versioned_id, output_dir)
        
        except HTTPError as e:
            # Check whether the HTTP status is retryable
            if not is_retryable_http_error(e.code):
                raise

            # Special handling for HTTP 429
            if e.code == 429:
                retry_after_value = (
                    e.headers.get("Retry-After")
                    if e.headers is not None
                    else None
                )

                retry_after_seconds = parse_retry_after(retry_after_value)

                # Conservative fallback when Retry-After is absent
                fallback_delay = settings.arxiv_rate_limit_cooldown * (
                    2 ** (attempt - 1)
                )
                delay = max(fallback_delay, retry_after_seconds or 0.0)

                # Apply cooldown to subsequent requests
                DEFAULT_RATE_LIMITER.cooldown(delay)

                logger.warning(
                    "HTTP 429 cooldown | version=%s | attempt=%s/%s | delay=%.1fs",
                    versioned_id,
                    attempt,
                    max_attempts,
                    delay
                )

                if attempt == max_attempts:
                    raise

                # The next download_version() call invokes limiter.wait()
                continue

            # Handle other retryable HTTP errors (5xx)
            if attempt == max_attempts:
                raise

            delay = calculate_backoff(attempt)

            retry_after_value = (
                e.headers.get("Retry-After")
                if e.headers is not None
                else None
            )
            retry_after_seconds = parse_retry_after(retry_after_value)

            if retry_after_seconds is not None:
                delay = max(delay, retry_after_seconds)

            logger.warning(
                "Retry scheduled | version=%s | attempt=%s/%s | delay=%.1fs | HTTP=%s",
                versioned_id,
                attempt,
                max_attempts,
                delay,
                e.code
            )

            time.sleep(delay)

        except (URLError, TimeoutError) as e: 
            # check if this is the last attempt 
            if attempt == max_attempts: 
                raise

            # calculate exponential backoff delay 
            delay = calculate_backoff(attempt)
            logger.warning(
                "Retry scheduled | version=%s | attempt=%s/%s | delay=%.1fs | error=%s",
                versioned_id, 
                attempt, 
                max_attempts, 
                delay, 
                e
            )
            # wait before retrying
            time.sleep(delay)
        except IncompleteDownloadError as e: 
            if attempt == max_attempts: 
                raise
            delay = calculate_backoff(attempt)
            logger.warning(
                "Retry scheduled | version=%s | attempt=%s/%s | delay=%.1f | error=%s", 
                versioned_id, 
                attempt, 
                max_attempts, 
                delay, 
                e
            )
            # wait before retrying
            time.sleep(delay)

def is_download_complete(
    versioned_id: str,
    output_dir: str | Path,
    *,
    fast: bool = False,
) -> bool:
    return verify_integrity(versioned_id, output_dir, fast=fast) is True

def download_with_status(versioned_id: str, output_dir: str | Path) -> dict[str, str | None]:
    try: 
        if is_download_complete(versioned_id, output_dir):
            path = Path(output_dir) / f"{versioned_id}.source"

            # Clean up stale partial file even on a cache hit.
            part_path = Path(f"{path}.part")
            part_path.unlink(missing_ok=True)

            status = "skipped"
        else:
            path = download_with_retry(versioned_id, output_dir)
            status = "downloaded"

        logger.info(
            "Download result | version=%s | status=%s | path=%s",
            versioned_id,
            status,
            path
        )

        # return a dictionary containing: versioned_id, status, path, error
        return {
            "versioned_id": versioned_id, 
            "status": status, 
            "path": str(path), 
            "error": None
        }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as e: 
        logger.error(
            "Download failed | version=%s | error=%s", 
            versioned_id, 
            e
        )
        return {
            "versioned_id": versioned_id, 
            "status": "failed", 
            "path": None, 
            "error": str(e)
        }

def download_many_versions(
        versioned_ids: list[str], 
        output_dir: str | Path, 
        status_file: str | Path | None = None
    ) -> list[dict[str, str | None]]:
    results = []

    if status_file is not None: 
        status_file = Path(status_file)
        status_file.parent.mkdir(
            parents=True, 
            exist_ok=True
        )

    successful_results = {}

    for versioned_id in versioned_ids:

        if versioned_id in successful_results:
            result = successful_results[versioned_id].copy()
            result["status"] = "skipped"

        else:
            result = download_with_status(versioned_id, output_dir)

            if result["status"] in ("downloaded", "skipped"):
                successful_results[versioned_id] = result

        results.append(result)

        if status_file is not None:
            with status_file.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

    return results

def detect_source_format(path: str | Path) -> str: 
    path = Path(path)

    with path.open("rb") as file: 
        header = file.read(512)

        # detect gzip magic bytes
        if header.startswith(b"\x1f\x8b"):
            return "gzip"

        # dêtct html content
        prefix = header.lstrip().lower()
        if prefix.startswith(b"<html") or prefix.startswith(b"<!doctype html"):
            return "html"

        # detect pdf signature 
        if header.startswith(b"%PDF-"):
            return "pdf"

        # detect uncompressed TAR archive
        if tarfile.is_tarfile(path):
            return "tar"

        # dêtct plain TeX source
        if (
            b"\\documentclass" in header 
            or b"\\begin{document}" in header
            or b"\\input{" in header
        ):
            return "tex"

        return "unknown"
    

def validate_gzip(path: str | Path) -> None: 
    path = Path(path)

    try: 
        with gzip.open(path, "rb") as file: 
            while True: 
                chunk = file.read(64 * 1024)

                if not chunk: 
                    break

    except (gzip.BadGzipFile, EOFError, zlib.error) as e: 
        raise IncompleteDownloadError(f"Invalid GZIP source: {e}") from e

def validate_tar(path: str | Path) -> None: 
    path = Path(path)

    try: 
        with tarfile.open(path, mode="r:*") as archive: 
            for member in archive:
                if not member.isfile(): 
                    continue

                source_file = archive.extractfile(member)
                if source_file is None: 
                    raise IncompleteDownloadError(f"Cannot read TAR member: {member.name}")

                with source_file: 
                    while source_file.read(64 * 1024):
                        pass
    except (tarfile.TarError, EOFError) as e: 
        raise IncompleteDownloadError(f"Invalid TAR source: {e}") from e 

def remove_cached_source(
    versioned_id: str,
    output_dir: str | Path
) -> None:
    validate_versioned_id(versioned_id)

    output_dir = Path(output_dir)

    source_path = output_dir / f"{versioned_id}.source"
    marker_path = output_dir / f"{versioned_id}.source.ok"

    # Invalidate the completion marker first
    marker_path.unlink(missing_ok=True)
    source_path.unlink(missing_ok=True)
