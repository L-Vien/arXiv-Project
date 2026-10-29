from pathlib import Path 
from urllib.request import Request, urlopen
import time 
import gzip 
import zlib
import re 
import json
import tarfile
import hashlib
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from datetime import datetime, timezone 
from email.utils import parsedate_to_datetime
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://export.arxiv.org/src"
VERSIONED_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}v[1-9]\d*")

class RateLimiter: 
    def __init__(self, min_interval: float = 3.0):
        if min_interval < 0: 
            raise ValueError("min_interval must be non-negative")

        self.min_interval = min_interval
        self.last_request_time = None

    def wait(self) -> None: 
        now = time.monotonic()

        if self.last_request_time is not None: 
            # calculate elapsed time 
            elapsed = now - self.last_request_time

            # calculate remaining wait time 
            remaining = self.min_interval - elapsed

            # sleep if remaining > 0
            if remaining > 0: 
                time.sleep(remaining)

        # save the new request start time
        self.last_request_time = time.monotonic()

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
        "sha256": calculate_sha256(source_path),
    }

    try:
        temp_marker.write_text(
            json.dumps(metadata, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temp_marker.replace(marker)

    finally:
        temp_marker.unlink(missing_ok=True)

def verify_integrity(
    versioned_id: str,
    output_dir: str | Path,
) -> bool | None:
    """
    True: checksum matches.
    False: file or marker is invalid, or checksum mismatches.
    None: legacy marker without checksum.
    """
    validate_versioned_id(versioned_id)
    output_dir = Path(output_dir)

    source = output_dir / f"{versioned_id}.source"
    marker = output_dir / f"{versioned_id}.source.ok"

    if (
        not source.is_file()
        or not marker.is_file()
        or source.stat().st_size == 0
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

        if expected_size != source.stat().st_size:
            return False

        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            return False

        return calculate_sha256(source) == expected_sha256

    except (OSError, UnicodeError, ValueError, TypeError):
        return False

def download_version(versioned_id: str, output_dir: str | Path, limiter: RateLimiter | None = None) -> Path: 
    if limiter is None:
            limiter = DEFAULT_RATE_LIMITER
    
    url = build_source_url(versioned_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # file chính thức
    output_path = output_dir / f"{versioned_id}.source"
    # file tạm thời 
    temp_path = output_dir / f"{versioned_id}.source.part"

    success_marker = output_dir / f"{versioned_id}.source.ok"

    if is_download_complete(versioned_id, output_dir):
        return output_path

    # Invalidate stale completion marker before downloading
    success_marker.unlink(missing_ok=True)

    # create a request object with User-Agent
    request = Request(
        url,
        headers={
            "User-Agent": settings.user_agent
        }
    )

    try: 
        limiter.wait()
        # open the url using urlopen()
        with urlopen(request, timeout=settings.request_timeout) as response:
            content_length = response.headers.get("Content-Length")
            expected_size = (
                int(content_length)
                if content_length is not None and content_length.isdecimal()
                else None 
            )
            bytes_downloaded = 0 
            # read binary data and write to temp_path
            with temp_path.open("wb") as file: 
                while True: 
                    try: 
                        chunk = response.read(64 * 1024)
                    except (IncompleteRead, ConnectionError) as e:
                        raise IncompleteDownloadError(f"Network interrupted while downloading {versioned_id}") from e
    
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

        # source validation
        validate_source(temp_path, versioned_id)
        temp_path.replace(output_path)
        write_integrity_marker(output_path)
        return output_path
    except HTTPError as e: 
        logger.warning(
            "HTTP error | version=%s | code=%s",
            versioned_id,
            e.code
        )
        raise
    except URLError as e: 
        logger.warning(
            "URL error | version=%s | reason=%s",
            versioned_id,
            e.reason
        )
        raise
    except TimeoutError as e: 
        logger.warning(
            "Timeout | version=%s | error=%s",
            versioned_id,
            e
        )
        raise
    finally: 
        temp_path.unlink(missing_ok=True)


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
            # check whether e.code is retryable 
            if not is_retryable_http_error(e.code): 
                raise

            # stop if max_attempts is reached
            if attempt == max_attempts: 
                raise

            # calculate delay using exponential backoff
            delay = calculate_backoff(attempt)

            retry_after_value = (
                e.headers.get("Retry-After")
                if e.headers is not None
                else None
            )
            retry_after_seconds = parse_retry_after(retry_after_value)

            if retry_after_seconds is not None: 
                delay = max(delay, retry_after_seconds)

            logger.warning("Retry scheduled | version=%s | attempt=%s/%s | delay=%.1fs | HTTP=%s",
                           versioned_id, 
                           attempt, 
                           max_attempts, 
                           delay, 
                           e.code
            )
            # wait before the next attempt
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
) -> bool:
    validate_versioned_id(versioned_id)
    output_dir = Path(output_dir)

    output_path = output_dir / f"{versioned_id}.source"
    success_marker = output_dir / f"{versioned_id}.source.ok"

    if not output_path.is_file() or not success_marker.is_file():
        return False

    actual_size = output_path.stat().st_size

    if actual_size <= 0:
        return False

    try:
        value = success_marker.read_text(encoding="utf-8")

        # Legacy marker
        if value.strip() == "downloaded":
            return True

        metadata = json.loads(value)

        return (
            isinstance(metadata, dict)
            and metadata.get("status") == "downloaded"
            and type(metadata.get("size_bytes")) is int
            and metadata["size_bytes"] == actual_size
            and isinstance(metadata.get("sha256"), str)
            and re.fullmatch(
                r"[0-9a-f]{64}",
                metadata["sha256"],
            ) is not None
        )

    except (OSError, UnicodeError, ValueError, TypeError):
        return False

def download_with_status(versioned_id: str, output_dir: str | Path) -> dict[str, str | None]:
    try: 
        was_complete = is_download_complete(versioned_id, output_dir)
        path = download_with_retry(versioned_id, output_dir)
        # determine status 
        # "skipped" if was_complete is True otherwise "downloaded"
        status = "skipped" if was_complete else "downloaded"

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