"""Xử lý gói nguồn arXiv đã tải về

Nhiệm vụ: nhận một file nguồn của MỘT phiên bản, giữ lại các file text cần cho
M2 (.tex, .bib, .bbl), bỏ hình/binary, và ghi lại trạng thái để
phiên bản không có TeX (PDF-only, docx, file hỏng) vẫn có bản ghi trong corpus.

arXiv /src/{id} có thể trả về:
  - tar.gz (phổ biến nhất)        -> giải nén, lọc theo đuôi file
  - .gz đơn (1 file .tex nén)     -> giải nén thành main.tex
  - PDF / docx / định dạng khác   -> không có TeX, chỉ ghi trạng thái
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tarfile
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from src.logger import get_logger

logger = get_logger(__name__)

# Các đuôi file text được giữ lại. Đề yêu cầu .tex và .bib; .bbl được giữ thêm vì
# một số bài chỉ nộp .bbl (không có .bib, không có \bibitem trong .tex), khi đó
# .bbl là nguồn tham khảo duy nhất cho M2. .sty/.cls/.bst... bị bỏ (file mẫu hội nghị).
KEEP_EXTS = {".tex", ".bib", ".bbl"}
STATUS_FILENAME = "source_status.json"

# Trạng thái trả về
STATUS_OK = "ok"                    # có ít nhất một file .tex
STATUS_NO_TEX = "no_tex"            # file hợp lệ nhưng không chứa TeX (PDF, docx, ...)
STATUS_MISSING = "missing"          # không tìm thấy file nguồn
STATUS_ERROR = "extract_error"      # file hỏng / không đọc được

# (tên member, kích thước chưa nén, hàm mở member để đọc)
_Member = tuple[str, int, Callable[[], object]]


@dataclass
class ProcessResult:
    status: str
    source_type: str = "unknown"
    archive_size: int = 0          # dung lượng file tải về (đã nén)
    raw_size: int = 0              # tổng dung lượng chưa nén của mọi file trong gói ("trước")
    kept_size: int = 0             # tổng dung lượng các file đã giữ lại ("sau")
    kept_files: list[str] = field(default_factory=list)
    error: str | None = None

    # Tên gọi khớp với yêu cầu báo cáo: dung lượng trước/sau khi bỏ hình.
    @property
    def before_size(self) -> int:
        return self.raw_size

    @property
    def after_size(self) -> int:
        return self.kept_size

    def to_dict(self) -> dict:
        return asdict(self)


def get_dir_size(path: str | Path = ".") -> int:
    """Tổng dung lượng (bytes) của mọi file trong thư mục, bỏ qua symlink."""
    total = 0
    if not os.path.exists(path):
        return total
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _is_tar(path: Path) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except (OSError, EOFError, tarfile.TarError, zlib.error):
        return False


def detect_source_type(path: str | Path) -> str:
    """Nhận diện định dạng bằng magic bytes: tar_gz, tar, gz_single, zip, pdf, unknown."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"\x1f\x8b"):
        return "tar_gz" if _is_tar(path) else "gz_single"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"           # gồm cả docx (docx là zip, sẽ không có .tex)
    if _is_tar(path):
        return "tar"
    return "unknown"


def _safe_target(out_root: Path, member_name: str) -> Path | None:
    """Đường dẫn đích an toàn; trả None nếu member cố thoát khỏi out_root (../, đường dẫn tuyệt đối)."""
    target = (out_root / member_name).resolve()
    try:
        target.relative_to(out_root)
    except ValueError:
        return None
    return target


def _iter_tar(tar: tarfile.TarFile) -> Iterator[_Member]:
    for m in tar:
        if m.isfile():  # bỏ thư mục, symlink, hardlink, device
            yield m.name, m.size, (lambda m=m: tar.extractfile(m))


def _iter_zip(zf: zipfile.ZipFile) -> Iterator[_Member]:
    for info in zf.infolist():
        if not info.is_dir():
            yield info.filename, info.file_size, (lambda info=info: zf.open(info))


def _extract_members(members: Iterator[_Member], out_dir: Path, result: ProcessResult) -> None:
    """Chỉ ghi ra đĩa các file có đuôi trong KEEP_EXTS; hình/binary không bao giờ được giải nén."""
    out_root = out_dir.resolve()
    for name, size, opener in members:
        result.raw_size += size
        if Path(name).suffix.lower() not in KEEP_EXTS:
            continue
        target = _safe_target(out_root, name)
        if target is None:
            logger.warning(f"Bỏ qua member có đường dẫn không an toàn: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with opener() as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        result.kept_files.append(target.relative_to(out_root).as_posix())
        result.kept_size += target.stat().st_size


def _extract_single_gz(path: Path, out_dir: Path, result: ProcessResult) -> None:
    """File .gz đơn: nội dung là một file TeX duy nhất (arXiv đặt tên là main.tex)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    result.raw_size = len(data)
    # PDF/PostScript/binary nén gz không phải TeX.
    if data.startswith((b"%PDF", b"%!PS")) or b"\x00" in data[:1024]:
        return
    (out_dir / "main.tex").write_bytes(data)
    result.kept_files.append("main.tex")
    result.kept_size = len(data)


def write_status(output_dir: str | Path, result: ProcessResult) -> Path:
    """Ghi source_status.json (qua file tạm để không để lại JSON dở dang)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STATUS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_status(output_dir: str | Path) -> ProcessResult | None:
    """Đọc source_status.json đã ghi trước đó; trả None nếu chưa có hoặc không đọc được."""
    path = Path(output_dir) / STATUS_FILENAME
    try:
        return ProcessResult(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _is_complete(output_dir: Path, status: ProcessResult) -> bool:
    """Mọi file đã ghi trong status còn tồn tại thật trên đĩa."""
    return all((output_dir / f).is_file() for f in status.kept_files)


def process_version(
    archive_path: str | Path,
    output_dir: str | Path,
    delete_archive: bool = True,
    skip_if_done: bool = True,
) -> ProcessResult:
    """Xử lý nguồn của một phiên bản vào output_dir và luôn ghi source_status.json.

    Giải nén vào thư mục tạm rồi đổi tên một lần, nên output_dir không bao giờ ở
    trạng thái dở dang. Không ném exception vì file nguồn xấu: lỗi ghi vào
    result.status/error. File nén chỉ bị xóa khi xử lý thành công (ok / no_tex).

    skip_if_done=True: nếu output_dir đã có source_status.json với status ok/no_tex
    VÀ mọi file đã ghi vẫn còn đủ thì trả về kết quả cũ, không xử lý lại.
    """
    archive_path, output_dir = Path(archive_path), Path(output_dir)

    if skip_if_done:
        previous = read_status(output_dir)
        if (previous is not None
                and previous.status in (STATUS_OK, STATUS_NO_TEX)
                and _is_complete(output_dir, previous)):
            logger.info(f"Bỏ qua {output_dir} (đã xử lý: {previous.status})")
            return previous

    if not archive_path.is_file():
        logger.error(f"Không tìm thấy file {archive_path}")
        result = ProcessResult(status=STATUS_MISSING, error="archive not found")
        write_status(output_dir, result)
        return result

    result = ProcessResult(status=STATUS_ERROR, archive_size=archive_path.stat().st_size)
    logger.info(f"Đang xử lý source: {archive_path} ({result.archive_size} bytes)")

    # Giải nén vào thư mục tạm (tên bắt đầu bằng "." để không lẫn với v1, v2...)
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)   # dọn rác của lần chạy bị ngắt
    tmp_dir.mkdir(parents=True)

    try:
        result.source_type = detect_source_type(archive_path)
        if result.source_type in ("tar_gz", "tar"):
            with tarfile.open(archive_path, "r:*") as tar:
                _extract_members(_iter_tar(tar), tmp_dir, result)
        elif result.source_type == "zip":
            with zipfile.ZipFile(archive_path) as zf:
                _extract_members(_iter_zip(zf), tmp_dir, result)
        elif result.source_type == "gz_single":
            _extract_single_gz(archive_path, tmp_dir, result)
        else:  # pdf, unknown: không có TeX để lấy
            result.raw_size = result.archive_size
        has_tex = any(f.lower().endswith(".tex") for f in result.kept_files)
        result.status = STATUS_OK if has_tex else STATUS_NO_TEX
    except (tarfile.TarError, zipfile.BadZipFile, gzip.BadGzipFile,
            EOFError, zlib.error, OSError) as e:
        result.status = STATUS_ERROR
        result.error = f"{type(e).__name__}: {e}"
        logger.error(f"Lỗi giải nén {archive_path}: {result.error}")

    if result.status in (STATUS_OK, STATUS_NO_TEX):
        write_status(tmp_dir, result)
        try:
            shutil.rmtree(output_dir, ignore_errors=True)   # bỏ dữ liệu cũ hoặc dở dang
            os.replace(tmp_dir, output_dir)                  # đổi tên một lần
        except OSError as e:
            result.status = STATUS_ERROR
            result.error = f"{type(e).__name__}: {e}"
            logger.error(f"Không đổi tên được {tmp_dir} -> {output_dir}: {result.error}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            write_status(output_dir, result)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        write_status(output_dir, result)

    if delete_archive and result.status in (STATUS_OK, STATUS_NO_TEX):
        try:
            archive_path.unlink()
        except OSError as e:
            logger.warning(f"Không xóa được {archive_path}: {e}")

    logger.info(
        f"{archive_path.name}: {result.status} ({result.source_type}), "
        f"giữ {len(result.kept_files)} file, {result.raw_size} -> {result.kept_size} bytes"
    )
    return result