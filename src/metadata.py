"""Chuẩn hóa và lưu metadata bài báo (bảng `papers`) + tiện ích ID/đường dẫn corpus (TV4)."""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

# ID mới: 2310.12345 ; ID cũ (trước 2007): hep-th/9901001 ; có thể kèm hậu tố v<n>.
_NEW_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v(\d+))?$")
_OLD_ID = re.compile(r"^([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v(\d+))?$")


# ---------- ID & đường dẫn corpus ----------

def parse_arxiv_id(raw: str) -> tuple[str, int | None]:
    """Tách ID gốc và số phiên bản. '2310.12345v2' -> ('2310.12345', 2).

    Chấp nhận tiền tố 'arXiv:' hoặc URL arxiv.org/abs|pdf/. Ném ValueError nếu không hợp lệ.
    """
    s = str(raw).strip()
    s = re.sub(r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf|src)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s)
    for pattern in (_NEW_ID, _OLD_ID):
        m = pattern.match(s)
        if m:
            return m.group(1), int(m.group(2)) if m.group(2) else None
    raise ValueError(f"arXiv ID không hợp lệ: {raw!r}")


def base_arxiv_id(raw: str) -> str:
    return parse_arxiv_id(raw)[0]


def paper_dir_name(arxiv_id: str) -> str:
    """Tên thư mục theo 'yymm-id': 2310.12345 -> 2310-12345; hep-th/9901001 -> hep-th_9901001."""
    return base_arxiv_id(arxiv_id).replace(".", "-").replace("/", "_")


def paper_dir(corpus_dir: str | Path, arxiv_id: str) -> Path:
    return Path(corpus_dir) / paper_dir_name(arxiv_id)


def version_dir(corpus_dir: str | Path, arxiv_id: str, version: int) -> Path:
    """<corpus>/<yymm-id>/v<version>/"""
    return paper_dir(corpus_dir, arxiv_id) / f"v{int(version)}"


# ---------- Chuẩn hóa metadata ----------

def _clean_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _to_iso(value: Any) -> str:
    """datetime/date -> chuỗi ISO 8601; chuỗi giữ nguyên (đã strip); None -> ''."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return _clean_str(value)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def normalize_paper_metadata(raw_metadata: dict) -> dict:
    """Đưa metadata thô về đúng schema bảng `papers`; loại field thừa, không tạo chuỗi 'None'."""
    arxiv_id = _clean_str(raw_metadata.get("arxiv_id"))
    try:
        arxiv_id = base_arxiv_id(arxiv_id)  # '2310.12345v2' -> '2310.12345'
    except ValueError:
        logger.warning(f"arxiv_id không theo định dạng chuẩn, giữ nguyên: {arxiv_id!r}")

    authors = [a for a in (_clean_str(x) for x in _as_list(raw_metadata.get("authors"))) if a]

    revised_dates, seen = [], set()
    for d in _as_list(raw_metadata.get("revised_dates")):
        d = _to_iso(d)
        if d and d not in seen:
            seen.add(d)
            revised_dates.append(d)

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(_clean_str(raw_metadata.get("title")).split()),  # gộp xuống dòng/khoảng trắng
        "authors": authors,
        "submission_date": _to_iso(raw_metadata.get("submission_date")),
        "revised_dates": revised_dates,
        "venue": _clean_str(raw_metadata.get("venue")) or None,
    }


def save_metadata(raw_metadata: dict, paper_root_dir: str | Path) -> str:
    """Ghi metadata.json vào thư mục bài báo (ghi qua file tạm để không để lại file dở dang)."""
    clean_meta = normalize_paper_metadata(raw_metadata)
    paper_root_dir = Path(paper_root_dir)
    paper_root_dir.mkdir(parents=True, exist_ok=True)

    file_path = paper_root_dir / "metadata.json"
    tmp_path = file_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(clean_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, file_path)

    logger.info(f"Đã xuất metadata.json cho bài báo: {clean_meta['arxiv_id']}")
    return str(file_path)

_REQUIRED_KEYS = {"arxiv_id", "title", "authors", "submission_date", "revised_dates", "venue"}


def validate_metadata(path: str | Path) -> bool:
    """True nếu metadata.json đọc được và đủ field đúng kiểu (dùng khi resume).

    path là file metadata.json hoặc thư mục bài báo chứa nó.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "metadata.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or not _REQUIRED_KEYS <= data.keys():
        return False
    try:
        parse_arxiv_id(data["arxiv_id"])
    except ValueError:
        return False
    authors, revised = data["authors"], data["revised_dates"]
    return (
        isinstance(data["title"], str) and data["title"].strip() != ""
        and isinstance(authors, list) and len(authors) > 0
        and all(isinstance(a, str) and a for a in authors)
        and isinstance(data["submission_date"], str) and data["submission_date"] != ""
        and isinstance(revised, list)
        and all(isinstance(d, str) and d for d in revised)
        and (data["venue"] is None or isinstance(data["venue"], str))
    )
