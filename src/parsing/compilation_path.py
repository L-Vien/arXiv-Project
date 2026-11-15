from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

try:
    from src.parsing.contracts import (
        Diagnostic,
        DiagnosticSeverity,
        ProcessingStatus,
        SourceBundle,
        SourceFile,
    )
except ImportError:  # pragma: no cover - fallback until src/parsing/contracts.py is created.
    class DiagnosticSeverity(str, Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    class ProcessingStatus(str, Enum):
        OK = "ok"
        EMPTY_NO_TEX = "empty_no_tex"
        ROOT_NOT_FOUND = "root_not_found"
        PARTIAL_PARSE = "partial_parse"
        PARSE_ERROR = "parse_error"
        BIBTEX_PARTIAL = "bibtex_partial"
        ERROR = "error"

    @dataclass(frozen=True)
    class Diagnostic:
        severity: DiagnosticSeverity
        code: str
        message: str
        file_path: str | None = None
        line: int | None = None

    @dataclass(frozen=True)
    class SourceFile:
        relative_path: str
        content: str
        order: int

    @dataclass(frozen=True)
    class SourceBundle:
        arxiv_id: str
        version: int
        status: ProcessingStatus
        root_file: str | None
        files: tuple[SourceFile, ...]
        diagnostics: tuple[Diagnostic, ...] = ()


DOCUMENTCLASS_PATTERN = re.compile(r"\\documentclass\b")

# Ví dụ cần nhận diện:
# \input{intro}
# \input{sections/method}
# \include{appendix}
INCLUDE_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")

__all__ = [
    "DOCUMENTCLASS_PATTERN",
    "INCLUDE_PATTERN",
    "Diagnostic",
    "DiagnosticSeverity",
    "ProcessingStatus",
    "SourceBundle",
    "SourceFile",
    "_remove_comments",
    "_find_root_file",
    "_resolve_include",
    "_collect_files",
    "gather_version",
]


def _to_relative_posix(path: Path, source_dir: Path) -> str:
    """Chuyển đường dẫn file sang chuỗi tương đối chuẩn POSIX so với source_dir."""
    try:
        return (
            path.resolve(strict=False)
            .relative_to(source_dir.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def _remove_comments(text: str) -> str:
    r"""Bỏ phần comment từ % đến hết dòng (không coi \% là comment), giữ nguyên số dòng."""
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        line_ending = ""
        if line.endswith("\r\n"):
            line_ending = "\r\n"
            body = line[:-2]
        elif line.endswith("\n") or line.endswith("\r"):
            line_ending = line[-1]
            body = line[:-1]
        else:
            body = line

        result: list[str] = []
        backslashes = 0
        for char in body:
            if char == "\\":
                backslashes += 1
                result.append(char)
                continue

            if char == "%" and backslashes % 2 == 0:
                break

            result.append(char)
            backslashes = 0

        lines.append("".join(result) + line_ending)
    return "".join(lines)


def _find_root_file(source_dir: Path) -> tuple[Path | None, list[Diagnostic]]:
    r"""Tìm file .tex chứa \documentclass và trả về (root_path, diagnostics).

    Quy tắc ưu tiên (Phương án A):
    1. File ở độ sâu thư mục nông nhất (gần source_dir nhất).
    2. Trong cùng độ sâu, ưu tiên main.tex.
    3. Sắp xếp theo thứ tự alphabet của đường dẫn POSIX.
    """
    diagnostics: list[Diagnostic] = []
    if not source_dir.exists() or not source_dir.is_dir():
        return None, diagnostics

    candidates = sorted(
        source_dir.rglob("*.tex"),
        key=lambda p: (
            len(p.relative_to(source_dir).parts),
            0 if p.name.lower() == "main.tex" else 1,
            p.as_posix(),
        ),
    )

    matched_roots: list[Path] = []
    for candidate in candidates:
        rel_path = _to_relative_posix(candidate, source_dir)
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="ROOT_CANDIDATE_READ_ERROR",
                    message=f"Could not read candidate TeX file: {exc}",
                    file_path=rel_path,
                )
            )
            continue

        if DOCUMENTCLASS_PATTERN.search(_remove_comments(content)):
            matched_roots.append(candidate.resolve())

    if not matched_roots:
        return None, diagnostics

    chosen = matched_roots[0]
    if len(matched_roots) > 1:
        chosen_rel = _to_relative_posix(chosen, source_dir)
        all_rel = ", ".join(_to_relative_posix(p, source_dir) for p in matched_roots)
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="MULTIPLE_ROOTS",
                message=f"Multiple root candidates found ({all_rel}); selected '{chosen_rel}'",
                file_path=chosen_rel,
            )
        )

    return chosen, diagnostics


def _is_escaping_source_dir(
    current_file: Path,
    include_target: str,
    source_dir: Path,
) -> bool:
    """Kiểm tra target có trỏ ra ngoài source_dir hoặc dùng đường dẫn tuyệt đối không."""
    target_path = Path(include_target)
    if target_path.is_absolute():
        return True

    resolved_root = source_dir.resolve(strict=False)
    candidate = (current_file.parent.resolve(strict=False) / target_path).resolve(
        strict=False
    )
    return not candidate.is_relative_to(resolved_root)


def _resolve_include(
    current_file: Path,
    include_target: str,
    source_dir: Path | None = None,
) -> Path | None:
    """Resolve đường dẫn include tương đối theo current_file (hoặc source_dir).

    Nếu target chưa có đuôi .tex, tự động thử bổ sung .tex.
    """
    target = include_target.strip()
    if not target or Path(target).is_absolute():
        return None

    bases = [current_file.parent.resolve(strict=False)]
    if source_dir is not None:
        resolved_source = source_dir.resolve(strict=False)
        if resolved_source not in bases:
            bases.append(resolved_source)

    resolved_root = source_dir.resolve(strict=False) if source_dir is not None else None

    for base in bases:
        candidate = (base / target).resolve(strict=False)
        attempts = [candidate]
        if candidate.suffix.lower() != ".tex":
            attempts.append(candidate.parent / f"{candidate.name}.tex")

        for attempt in attempts:
            if resolved_root is not None and not attempt.is_relative_to(resolved_root):
                continue
            if attempt.exists() and attempt.is_file():
                return attempt

    return None


def _collect_files(
    root_file: Path,
    source_dir: Path,
) -> tuple[list[SourceFile], list[Diagnostic], bool]:
    """Duyệt DFS theo thứ tự include, phân biệt lỗi đọc root với lỗi đọc file con.

    Trả về: (order, diagnostics, root_read_failed)
    """
    resolved_source_dir = source_dir.resolve(strict=False)
    root_resolved = root_file.resolve(strict=False)
    visited: set[Path] = set()
    recursion_stack: list[Path] = []
    order: list[SourceFile] = []
    diagnostics: list[Diagnostic] = []
    root_read_failed = False

    def visit(file_path: Path) -> None:
        nonlocal root_read_failed
        resolved = file_path.resolve(strict=False)
        rel_path = _to_relative_posix(resolved, resolved_source_dir)

        if resolved in visited:
            return
        if resolved in recursion_stack:
            cycle_start = recursion_stack.index(resolved)
            cycle_chain = [
                _to_relative_posix(p, resolved_source_dir)
                for p in recursion_stack[cycle_start:] + [resolved]
            ]
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="INCLUDE_CYCLE",
                    message=f"Detected include cycle: {' -> '.join(cycle_chain)}",
                    file_path=rel_path,
                )
            )
            return

        recursion_stack.append(resolved)
        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            if resolved == root_resolved:
                root_read_failed = True
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        code="ROOT_READ_ERROR",
                        message=f"Could not read root file: {exc}",
                        file_path=rel_path,
                    )
                )
            else:
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.ERROR,
                        code="FILE_READ_ERROR",
                        message=f"Could not read included file: {exc}",
                        file_path=rel_path,
                    )
                )
            recursion_stack.pop()
            return

        order.append(
            SourceFile(
                relative_path=rel_path,
                content=text,
                order=len(order),
            )
        )
        stripped = _remove_comments(text)
        for match in INCLUDE_PATTERN.finditer(stripped):
            include_target = match.group(1).strip()
            line_no = stripped[: match.start()].count("\n") + 1

            if not include_target:
                continue

            if _is_escaping_source_dir(resolved, include_target, resolved_source_dir):
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code="PATH_TRAVERSAL",
                        message=f"Include target escapes source directory: {include_target}",
                        file_path=rel_path,
                        line=line_no,
                    )
                )
                continue

            include_path = _resolve_include(
                resolved,
                include_target,
                source_dir=resolved_source_dir,
            )
            if include_path is None:
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code="MISSING_INCLUDE",
                        message=f"Included file was not found: {include_target}",
                        file_path=rel_path,
                        line=line_no,
                    )
                )
                continue

            visit(include_path)

        recursion_stack.pop()
        visited.add(resolved)

    visit(root_resolved)
    return order, diagnostics, root_read_failed


def gather_version(
    source_dir: Path,
    arxiv_id: str,
    version: int,
) -> SourceBundle:
    """Thu thập các file thuộc compilation path của một arXiv version."""
    source_dir = Path(source_dir)

    if not source_dir.exists() or not source_dir.is_dir():
        return SourceBundle(
            arxiv_id=arxiv_id,
            version=version,
            status=ProcessingStatus.ERROR,
            root_file=None,
            files=(),
            diagnostics=(
                Diagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="SOURCE_DIR_NOT_FOUND",
                    message=f"Source directory does not exist or is not readable: {source_dir}",
                ),
            ),
        )

    tex_files = sorted(source_dir.rglob("*.tex"))
    if not tex_files:
        return SourceBundle(
            arxiv_id=arxiv_id,
            version=version,
            status=ProcessingStatus.EMPTY_NO_TEX,
            root_file=None,
            files=(),
            diagnostics=(),
        )

    root_file, root_diagnostics = _find_root_file(source_dir)
    if root_file is None:
        has_candidate_read_error = any(
            d.code == "ROOT_CANDIDATE_READ_ERROR" for d in root_diagnostics
        )
        if has_candidate_read_error:
            return SourceBundle(
                arxiv_id=arxiv_id,
                version=version,
                status=ProcessingStatus.ERROR,
                root_file=None,
                files=(),
                diagnostics=tuple(root_diagnostics),
            )

        return SourceBundle(
            arxiv_id=arxiv_id,
            version=version,
            status=ProcessingStatus.ROOT_NOT_FOUND,
            root_file=None,
            files=(),
            diagnostics=(
                *root_diagnostics,
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="ROOT_NOT_FOUND",
                    message=f"No root TeX file with \\documentclass was found under {source_dir}",
                ),
            ),
        )

    rel_root = _to_relative_posix(root_file, source_dir)
    files, collect_diagnostics, root_read_failed = _collect_files(root_file, source_dir)
    all_diagnostics = (*root_diagnostics, *collect_diagnostics)

    if root_read_failed:
        return SourceBundle(
            arxiv_id=arxiv_id,
            version=version,
            status=ProcessingStatus.ERROR,
            root_file=rel_root,
            files=(),
            diagnostics=all_diagnostics,
        )

    status = ProcessingStatus.PARTIAL_PARSE if all_diagnostics else ProcessingStatus.OK
    return SourceBundle(
        arxiv_id=arxiv_id,
        version=version,
        status=status,
        root_file=rel_root,
        files=tuple(files),
        diagnostics=all_diagnostics,
    )