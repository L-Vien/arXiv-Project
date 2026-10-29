import gzip
import io
import json
import tarfile
from datetime import datetime, timezone
from email.utils import format_datetime
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError
import pytest
from src.downloader import (
    IncompleteDownloadError,
    InvalidSourceFormatError,
    RateLimiter,
    build_versioned_ids,
    calculate_sha256,
    detect_source_format,
    download_many_versions,
    download_version,
    download_with_retry,
    download_with_status,
    is_download_complete,
    parse_retry_after,
    validate_gzip,
    validate_tar,
    verify_integrity,
    write_integrity_marker,
)

# 01. Metadata & input validation
def test_build_versioned_ids_single():
    metadata = {
        "arxiv_id": "2607.28694",
        "versions": ["v1", "v2"]
    }
    result = build_versioned_ids(metadata)

    assert result == ["2607.28694v1", "2607.28694v2"]

def test_build_versioned_ids_empty():
    metadata = {
        "arxiv_id": "2607.28694",
        "versions": []
    }
    result = build_versioned_ids(metadata)

    assert result == []


def test_build_versioned_ids_invalid_versions():
    metadata = {
        "arxiv_id": "2607.28694",
        "versions": ["v1", "../bad"]
    }

    with pytest.raises(ValueError):
        build_versioned_ids(metadata)


@pytest.mark.parametrize(
    "invalid_id",
    [
        "../../bad",
        "2607.28694",
        "2607.28694v0",
        "2607.28694v1/extra",
    ]
)
def test_invalid_versioned_id(tmp_path, invalid_id):
    result = download_with_status(
        invalid_id,
        tmp_path
    )

    assert result["status"] == "failed"
    assert result["path"] is None
    assert result["error"] is not None

    # Không được tạo file ngoài ý muốn
    assert list(tmp_path.iterdir()) == []

# 02. Source format detection
def test_detect_html(tmp_path):
    path = tmp_path / "sample.source"

    path.write_bytes(
        b"<!DOCTYPE html><html>Error</html>"
    )

    assert detect_source_format(path) == "html"


def test_detect_pdf(tmp_path):
    path = tmp_path / "sample.source"

    path.write_bytes(b"%PDF-1.7\n")

    assert detect_source_format(path) == "pdf"


def test_detect_tar(tmp_path):
    path = tmp_path / "sample.source"

    # Create a TAR archive containing one TeX file
    content = b"Hello arXiv"
    info = tarfile.TarInfo(name="main.tex")
    info.size = len(content)

    with tarfile.open(path, "w") as archive:
        archive.addfile(info, io.BytesIO(content))

    assert detect_source_format(path) == "tar"


def test_detect_unknown(tmp_path):
    path = tmp_path / "sample.source"

    path.write_bytes(b"Random binary content")

    assert detect_source_format(path) == "unknown"


def test_detect_plain_tex(tmp_path):
    path = tmp_path / "sample.source"

    path.write_bytes(
        b"\\documentclass{article}\n"
        b"\\begin{document}\n"
        b"Hello arXiv!\n"
        b"\\end{document}\n"
    )

    assert detect_source_format(path) == "tex"


def test_detect_plain_tex_with_input(tmp_path):
    path = tmp_path / "sample.source"

    path.write_bytes(
        b"\\input{main.tex}\n"
        b"\\bye\n"
    )

    assert detect_source_format(path) == "tex"

# 03. GZIP / TAR validation
def test_valid_gzip(tmp_path):
    # Arrange: Create a valid GZIP file
    path = tmp_path / "sample.source"

    with gzip.open(path, "wb") as file:
        file.write(b"Hello arXiv")

    # Act & Assert: Check source format
    assert detect_source_format(path) == "gzip"

    # Validate integrity: no exception expected
    validate_gzip(path)


def test_truncated_gzip(tmp_path):
    # Arrange: Create a valid GZIP file
    path = tmp_path / "broken.source"

    with gzip.open(path, "wb") as file:
        file.write(b"Hello arXiv")

    # Simulate an incomplete download
    data = path.read_bytes()
    path.write_bytes(data[:-8])

    # The GZIP magic bytes are still present
    assert detect_source_format(path) == "gzip"

    # Expect an exception during validation
    with pytest.raises(IncompleteDownloadError):
        validate_gzip(path)


def test_validate_truncated_tar(tmp_path):
    path = tmp_path / "broken.source"
    content = b"A" * 2048

    info = tarfile.TarInfo(name="main.tex")
    info.size = len(content)

    with tarfile.open(path, "w") as archive:
        archive.addfile(info, BytesIO(content))

    # Giữ header nhưng cắt mất một phần dữ liệu
    path.write_bytes(path.read_bytes()[:1200])

    assert detect_source_format(path) == "tar"

    with pytest.raises(IncompleteDownloadError):
        validate_tar(path)

# 04. Download & source safety
def test_download_version_success(tmp_path):
# Arrange: Prepare fake GZIP response
    versioned_id = "2607.28694v1"
    payload = gzip.compress(b"Hello arXiv")
    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }
# Act: Replace real urlopen with a fake response
    with patch(
        "src.downloader.urlopen",
        return_value=response
    ) as mock_urlopen:
        result = download_version(
            versioned_id,
            tmp_path,
            limiter=RateLimiter(0)
        )
# Assert: Verify downloaded files
    expected_path = tmp_path / f"{versioned_id}.source"
    marker = tmp_path / f"{versioned_id}.source.ok"
    assert result == expected_path
    assert expected_path.read_bytes() == payload
    assert marker.exists()
    mock_urlopen.assert_called_once()
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_data["status"] == "downloaded"
    assert marker_data["size_bytes"] == len(payload)
    assert marker_data["sha256"] == calculate_sha256(expected_path)
    assert not (tmp_path / f"{versioned_id}.source.ok.tmp").exists()


def test_download_pdf_source(tmp_path):
    versioned_id = "2607.28694v1"
    payload = b"%PDF-1.7\nMock PDF content\n%%EOF\n"

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch("src.downloader.urlopen", return_value=response):
        path = download_version(
            versioned_id,
            tmp_path,
            limiter=RateLimiter(0)
        )

    assert path.read_bytes() == payload
    assert detect_source_format(path) == "pdf"
    assert (tmp_path / f"{versioned_id}.source.ok").exists()


def test_download_plain_tex(tmp_path):
    versioned_id = "2607.28694v1"

    payload = (
        b"\\documentclass{article}\n"
        b"\\begin{document}\n"
        b"Hello arXiv!\n"
        b"\\end{document}\n"
    )

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch(
        "src.downloader.urlopen",
        return_value=response
    ) as mock_urlopen:

        path = download_version(
            versioned_id,
            tmp_path,
            limiter=RateLimiter(0)
        )

    marker = tmp_path / f"{versioned_id}.source.ok"
    temp_path = tmp_path / f"{versioned_id}.source.part"

    assert path.is_file()
    assert path.read_bytes() == payload
    assert marker.is_file()
    assert not temp_path.exists()

    mock_urlopen.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        b"<html>Error</html>",
        b"Random binary content",
    ],
)
def test_download_invalid_format(tmp_path, payload):
    versioned_id = "2607.28694v1"

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch("src.downloader.urlopen", return_value=response):
        with pytest.raises(InvalidSourceFormatError):
            download_version(
                versioned_id,
                tmp_path,
                limiter=RateLimiter(0)
            )

    assert not (tmp_path / f"{versioned_id}.source").exists()
    assert not (tmp_path / f"{versioned_id}.source.ok").exists()
    assert not (tmp_path / f"{versioned_id}.source.part").exists()


def test_download_truncated_gzip_tar(tmp_path):
    versioned_id = "2607.28694v1"
    tar_path = tmp_path / "sample.tar"

    content = b"A" * 2048
    info = tarfile.TarInfo(name="main.tex")
    info.size = len(content)

    with tarfile.open(tar_path, "w") as archive:
        archive.addfile(info, BytesIO(content))

    # TAR bị cắt ngắn rồi mới được nén GZIP
    broken_tar = tar_path.read_bytes()[:1200]
    payload = gzip.compress(broken_tar)

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch("src.downloader.urlopen", return_value=response):
        with pytest.raises(IncompleteDownloadError):
            download_version(
                versioned_id,
                tmp_path,
                limiter=RateLimiter(0)
            )

    assert not (tmp_path / f"{versioned_id}.source.ok").exists()
    assert not (tmp_path / f"{versioned_id}.source.part").exists()


def test_download_incomplete_content_length(tmp_path):
    versioned_id = "2607.28694v1"
    payload = gzip.compress(b"Hello arXiv")

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload) + 10)
    }

    with patch("src.downloader.urlopen", return_value=response):
        with pytest.raises(IncompleteDownloadError):
            download_version(
                versioned_id,
                tmp_path,
                limiter=RateLimiter(0)
            )

    assert not (tmp_path / f"{versioned_id}.source.ok").exists()
    assert not (tmp_path / f"{versioned_id}.source.part").exists()


def test_reject_gzip_html(tmp_path):
    versioned_id = "2607.28694v1"

    payload = gzip.compress(
        b"<html>Temporary error</html>"
    )

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch(
        "src.downloader.urlopen",
        return_value=response
    ):
        with pytest.raises(InvalidSourceFormatError):
            download_version(
                versioned_id,
                tmp_path,
                limiter=RateLimiter(0)
            )

    assert not (
        tmp_path / f"{versioned_id}.source.ok"
    ).exists()

# 05. RateLimiter & Retry policy
def test_rate_limiter_wait():
    limiter = RateLimiter(min_interval=3.0)

    with patch(
        "src.downloader.time.monotonic",
        side_effect=[100.0, 100.0, 101.0, 103.0]
    ):
        with patch("src.downloader.time.sleep") as mock_sleep:
            limiter.wait()
            limiter.wait()

    mock_sleep.assert_called_once_with(2.0)
    assert limiter.last_request_time == 103.0


def test_parse_retry_after_http_date():
    fake_now = datetime(
        2026, 10, 26, 10, 0, 0,
        tzinfo=timezone.utc
    )

    retry_time = datetime(
        2026, 10, 26, 10, 0, 30,
        tzinfo=timezone.utc
    )

    retry_after = format_datetime(
        retry_time,
        usegmt=True
    )

    with patch("src.downloader.datetime") as mock_datetime:
        mock_datetime.now.return_value = fake_now

        result = parse_retry_after(retry_after)

    assert result == 30.0


def test_retry_http_429(tmp_path):
    versioned_id = "2607.28694v1"
    expected_path = tmp_path / f"{versioned_id}.source"

    # Simulate HTTP 429 with Retry-After
    error = HTTPError(
        url=f"https://export.arxiv.org/src/{versioned_id}",
        code=429,
        msg="Too Many Requests",
        hdrs={"Retry-After": "8"},
        fp=None
    )

    # First attempt fails, second succeeds
    with patch(
        "src.downloader.download_version",
        side_effect=[error, expected_path]
    ) as mock_download:

        with patch(
            "src.downloader.time.sleep"
        ) as mock_sleep:

            result = download_with_retry(
                versioned_id,
                tmp_path,
                max_attempts=3
            )

    assert result == expected_path
    assert mock_download.call_count == 2
    mock_sleep.assert_called_once_with(8.0)


def test_retry_http_429_missing_headers(tmp_path):
    versioned_id = "2607.28694v1"

    expected_path = (
        tmp_path / f"{versioned_id}.source"
    )

    error = HTTPError(
        url=f"https://export.arxiv.org/src/{versioned_id}",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=None
    )

    with patch(
        "src.downloader.download_version",
        side_effect=[error, expected_path]
    ) as mock_download:

        with patch(
            "src.downloader.time.sleep"
        ) as mock_sleep:

            result = download_with_retry(
                versioned_id,
                tmp_path,
                max_attempts=2
            )

    assert result == expected_path
    assert mock_download.call_count == 2

    # Không có Retry-After nên dùng Exponential Backoff
    mock_sleep.assert_called_once_with(3)


@pytest.mark.parametrize("status_code", [403, 404])
def test_non_retryable_http_error(tmp_path, status_code):
    versioned_id = "2607.28694v1"

    error = HTTPError(
        url=f"https://export.arxiv.org/src/{versioned_id}",
        code=status_code,
        msg="HTTP Error",
        hdrs={},
        fp=None
    )

    with patch(
        "src.downloader.download_version",
        side_effect=error
    ) as mock_download:

        with patch("src.downloader.time.sleep") as mock_sleep:

            with pytest.raises(HTTPError) as exc_info:
                download_with_retry(
                    versioned_id,
                    tmp_path,
                    max_attempts=3
                )

    assert exc_info.value.code == status_code
    assert mock_download.call_count == 1
    mock_sleep.assert_not_called()


def test_http_503_max_attempts(tmp_path):
    versioned_id = "2607.28694v1"

    error = HTTPError(
        url=f"https://export.arxiv.org/src/{versioned_id}",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=None
    )

    with patch(
        "src.downloader.download_version",
        side_effect=error
    ) as mock_download:

        with patch("src.downloader.time.sleep") as mock_sleep:

            with pytest.raises(HTTPError):
                download_with_retry(
                    versioned_id,
                    tmp_path,
                    max_attempts=3
                )

    assert mock_download.call_count == 3

    assert mock_sleep.call_args_list == [
        call(3),
        call(6)
    ]


def test_retry_timeout_then_success(tmp_path):
    versioned_id = "2607.28694v1"
    expected_path = tmp_path / f"{versioned_id}.source"

    with patch(
        "src.downloader.download_version",
        side_effect=[TimeoutError("Request timed out"), expected_path]
    ) as mock_download:

        with patch("src.downloader.time.sleep") as mock_sleep:

            result = download_with_retry(
                versioned_id,
                tmp_path,
                max_attempts=3
            )

    assert result == expected_path
    assert mock_download.call_count == 2
    mock_sleep.assert_called_once_with(3)


def test_retry_corrupted_gzip(tmp_path):
    versioned_id = "2607.28694v1"

    valid_payload = gzip.compress(b"Hello arXiv")
    corrupted_payload = valid_payload[:-8]

    responses = []

    for payload in [corrupted_payload, valid_payload]:
        response = BytesIO(payload)
        response.headers = {
            "Content-Length": str(len(payload))
        }
        responses.append(response)

    with patch(
        "src.downloader.urlopen",
        side_effect=responses
    ) as mock_urlopen:

        with patch(
            "src.downloader.DEFAULT_RATE_LIMITER",
            RateLimiter(0)
        ):
            with patch("src.downloader.time.sleep") as mock_sleep:
                path = download_with_retry(
                    versioned_id,
                    tmp_path,
                    max_attempts=3
                )

    assert path.read_bytes() == valid_payload
    assert mock_urlopen.call_count == 2
    mock_sleep.assert_called_once_with(3)

    assert (tmp_path / f"{versioned_id}.source.ok").exists()
    assert not (tmp_path / f"{versioned_id}.source.part").exists()


def test_retry_incomplete_read(tmp_path):
    versioned_id = "2607.28694v1"
    payload = gzip.compress(b"Hello arXiv")

    class BrokenResponse(BytesIO):
        def read(self, size=-1):
            raise IncompleteRead(b"partial", 100)

    broken_response = BrokenResponse()
    broken_response.headers = {
        "Content-Length": "100"
    }

    valid_response = BytesIO(payload)
    valid_response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch(
        "src.downloader.urlopen",
        side_effect=[broken_response, valid_response]
    ) as mock_urlopen:

        with patch(
            "src.downloader.DEFAULT_RATE_LIMITER",
            RateLimiter(0)
        ):
            with patch("src.downloader.time.sleep"):
                path = download_with_retry(
                    versioned_id,
                    tmp_path,
                    max_attempts=2
                )

    assert mock_urlopen.call_count == 2
    assert path.read_bytes() == payload

    assert (
        tmp_path / f"{versioned_id}.source.ok"
    ).exists()

    assert not (
        tmp_path / f"{versioned_id}.source.part"
    ).exists()

# 06. Resume, statuses & batch recovery
def test_download_status_and_resume(tmp_path):
    versioned_id = "2607.28694v1"
    payload = gzip.compress(b"Hello arXiv")

    response = BytesIO(payload)
    response.headers = {
        "Content-Length": str(len(payload))
    }

    with patch("src.downloader.urlopen", return_value=response) as mock_urlopen:
        with patch("src.downloader.DEFAULT_RATE_LIMITER", RateLimiter(0)):

            first = download_with_status(versioned_id, tmp_path)
            second = download_with_status(versioned_id, tmp_path)

    assert first["status"] == "downloaded"
    assert second["status"] == "skipped"

    assert first["path"] == second["path"]
    assert first["error"] is None
    assert second["error"] is None

    mock_urlopen.assert_called_once()


def test_log_download_failed(tmp_path):
    versioned_id = "2607.28694v1"

    error = URLError("Network unavailable")

    with patch(
        "src.downloader.download_with_retry",
        side_effect=error
    ):
        with patch(
            "src.downloader.logger.error"
        ) as mock_log:

            result = download_with_status(
                versioned_id,
                tmp_path
            )

    assert result["status"] == "failed"
    assert result["path"] is None
    assert result["error"] is not None

    mock_log.assert_called_once()


def test_batch_continues_after_failure(tmp_path):
    versioned_ids = [
        "2607.28694v1",
        "2607.28695v1",
        "2607.28696v1",
    ]

    def fake_download(versioned_id, output_dir, max_attempts=3):
        if versioned_id == "2607.28695v1":
            raise URLError("Network error")

        return Path(output_dir) / f"{versioned_id}.source"

    with patch(
        "src.downloader.download_with_retry",
        side_effect=fake_download
    ):
        results = download_many_versions(versioned_ids, tmp_path)

    assert len(results) == 3

    assert [result["status"] for result in results] == [
        "downloaded",
        "failed",
        "downloaded",
    ]

    assert results[1]["path"] is None
    assert results[1]["error"] is not None


def test_batch_writes_status_jsonl(tmp_path):
    versioned_ids = [
        "2607.28694v1",
        "2607.28695v1",
    ]

    status_file = tmp_path / "status" / "download_status.jsonl"

    fake_results = [
        {
            "versioned_id": "2607.28694v1",
            "status": "downloaded",
            "path": str(tmp_path / "2607.28694v1.source"),
            "error": None,
        },
        {
            "versioned_id": "2607.28695v1",
            "status": "failed",
            "path": None,
            "error": "HTTP Error 404",
        },
    ]

    with patch(
        "src.downloader.download_with_status",
        side_effect=fake_results
    ):
        results = download_many_versions(
            versioned_ids,
            tmp_path,
            status_file=status_file
        )

    assert results == fake_results
    assert status_file.is_file()

    lines = status_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    records = [
        json.loads(line)
        for line in lines
    ]

    assert records == fake_results


def test_batch_preserves_status_after_interrupt(tmp_path):
    versioned_ids = [
        "2607.28694v1",
        "2607.28695v1",
    ]

    status_file = tmp_path / "download_status.jsonl"

    def fake_download(versioned_id, output_dir):
        if versioned_id == "2607.28695v1":
            raise KeyboardInterrupt()

        return {
            "versioned_id": versioned_id,
            "status": "downloaded",
            "path": str(
                tmp_path / f"{versioned_id}.source"
            ),
            "error": None
        }

    with patch(
        "src.downloader.download_with_status",
        side_effect=fake_download
    ):
        with pytest.raises(KeyboardInterrupt):
            download_many_versions(
                versioned_ids,
                tmp_path,
                status_file=status_file
            )

    assert status_file.is_file()

    lines = status_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["versioned_id"] == "2607.28694v1"
    assert record["status"] == "downloaded"


# 07. Cache integrity & compatibility
def test_integrity_detects_same_size_corruption(tmp_path):
    versioned_id = "2607.28694v1"
    source = tmp_path / f"{versioned_id}.source"
    source.write_bytes(b"ABCDEF")
    write_integrity_marker(source)

    # Original file is valid
    assert verify_integrity(versioned_id, tmp_path) is True
    assert is_download_complete(versioned_id, tmp_path) is True

    # Modify contents without changing the file size
    source.write_bytes(b"XXXXXX")

    # SHA-256 must detect the corruption
    assert verify_integrity(versioned_id, tmp_path) is False
    assert is_download_complete(versioned_id, tmp_path) is False


def test_integrity_legacy_marker(tmp_path):
    versioned_id = "2607.28694v1"
    source = tmp_path / f"{versioned_id}.source"
    marker = tmp_path / f"{versioned_id}.source.ok"
    source.write_bytes(b"Legacy source")
    marker.write_text("downloaded\n", encoding="utf-8")

    assert is_download_complete(versioned_id, tmp_path) is False
    # No stored checksum to verify
    assert verify_integrity(versioned_id, tmp_path) is None


def test_integrity_invalid_json_marker(tmp_path):
    versioned_id = "2607.28694v1"
    source = tmp_path / f"{versioned_id}.source"
    marker = tmp_path / f"{versioned_id}.source.ok"
    source.write_bytes(b"Source content")
    marker.write_text("{invalid json", encoding="utf-8")
    assert is_download_complete(versioned_id, tmp_path) is False
    assert verify_integrity(versioned_id, tmp_path) is False


def test_integrity_detects_size_mismatch(tmp_path):
    versioned_id = "2607.28694v1"
    source = tmp_path / f"{versioned_id}.source"
    source.write_bytes(b"Original content")
    write_integrity_marker(source)

    assert is_download_complete(versioned_id, tmp_path) is True
    # Truncate source after creating its marker
    source.write_bytes(b"Short")
    assert is_download_complete(versioned_id, tmp_path) is False
    assert verify_integrity(versioned_id, tmp_path) is False

# 08. Batch deduplication
def test_batch_dedup_success(tmp_path):
    versioned_id = "2607.28694v1"
    fake_result = {
        "versioned_id": versioned_id,
        "status": "downloaded",
        "path": str(tmp_path / f"{versioned_id}.source"),
        "error": None,
    }
    versioned_ids = [versioned_id, versioned_id, versioned_id]
    with patch(
        "src.downloader.download_with_status",
        return_value=fake_result,
    ) as mock_download:
        results = download_many_versions(
            versioned_ids,
            tmp_path,
        )
    assert mock_download.call_count == 1
    assert [result["status"] for result in results] == [
        "downloaded",
        "skipped",
        "skipped",
    ]
    assert len(results) == len(versioned_ids)


def test_batch_dedup_retries_failed_id(tmp_path):
    versioned_id = "2607.28694v1"
    failed_result = {
        "versioned_id": versioned_id,
        "status": "failed",
        "path": None,
        "error": "Network error",
    }
    success_result = {
        "versioned_id": versioned_id,
        "status": "downloaded",
        "path": str(tmp_path / f"{versioned_id}.source"),
        "error": None,
    }
    versioned_ids = [versioned_id, versioned_id, versioned_id]
    with patch(
        "src.downloader.download_with_status",
        side_effect=[failed_result, success_result],
    ) as mock_download:
        results = download_many_versions(
            versioned_ids,
            tmp_path,
        )
# Failed result must not enter the success cache
    assert mock_download.call_count == 2
    assert [result["status"] for result in results] == [
        "failed",
        "downloaded",
        "skipped",
    ]

def test_redownload_corrupted_source(tmp_path):
    versioned_id = "2607.28694v1"
    payload = gzip.compress(b"Valid arXiv source")

    def make_response():
        response = BytesIO(payload)
        response.headers = {
            "Content-Length": str(len(payload))
        }
        return response

    with patch(
        "src.downloader.urlopen",
        side_effect=[make_response(), make_response()]
    ) as mock_urlopen:

        with patch(
            "src.downloader.DEFAULT_RATE_LIMITER",
            RateLimiter(0)
        ):
            # First download
            first = download_with_status(
                versioned_id, tmp_path
            )

            source = tmp_path / f"{versioned_id}.source"

            assert first["status"] == "downloaded"
            assert verify_integrity(versioned_id, tmp_path) is True

            # Corrupt file without changing its size
            source.write_bytes(b"X" * len(payload))

            assert verify_integrity(versioned_id, tmp_path) is False

            # Second download must repair corrupted source
            second = download_with_status(
                versioned_id, tmp_path
            )

    assert second["status"] == "downloaded"
    assert mock_urlopen.call_count == 2
    assert source.read_bytes() == payload
    assert verify_integrity(versioned_id, tmp_path) is True