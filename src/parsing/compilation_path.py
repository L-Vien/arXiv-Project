from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import posixpath
import re

from src.parsing.contracts import (
    Diagnostic,
    DiagnosticSeverity,
    ProcessingStatus,
    SourceBundle,
    SourceFile,
)

DOCUMENTCLASS_PATTERN = re.compile(r"\\documentclass\b")
BEGIN_DOCUMENT_PATTERN = re.compile(r"\\begin\s*\{document\}")
PLAIN_TEX_PATTERN = re.compile(r"\\bye\b")

# Ví dụ cần nhận diện:
# \input{intro}
# \input{sections/method}
# \include{appendix}
# \subfile{sections/intro}
INCLUDE_PATTERN = re.compile(
    r"\\(?:input|include|subfile|subfileinclude)\s*\{([^{}]+)\}"
)
# Kiểu plain-TeX: \input intro   (không có dấu ngoặc nhọn)
BARE_INPUT_PATTERN = re.compile(r"\\input[ \t]+([A-Za-z0-9_./-]+)")

# Các environment mà LaTeX không thực thi nội dung -> không tính include/documentclass bên trong.
MASKED_ENV_PATTERN = re.compile(
    r"\\begin\s*\{(verbatim\*?|lstlisting|comment)\}.*?\\end\s*\{\1\}",
    re.DOTALL,
)

BIB_EXTS = frozenset({".bib", ".bbl"})

# \bibliography{a,b}  (BibTeX)  -- không khớp \bibliographystyle
BIBLIOGRAPHY_PATTERN = re.compile(r"\\bibliography\s*\{([^{}]*)\}")
# \addbibresource[opts]{a.bib}  (biblatex)
ADDBIB_PATTERN = re.compile(
    r"\\(?:addbibresource|addglobalbib|addsectionbib)\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}"
)
# \printbibliography / \bibliography -> LaTeX sẽ đọc file .bbl
PRINTBIB_PATTERN = re.compile(r"\\printbibliography\b")

# Include các file này không phải TeX source -> bỏ qua, không báo MISSING_INCLUDE.
NON_TEX_EXTS = {
    ".bbl", ".bib", ".sty", ".cls", ".def", ".cfg",
    ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".pdf_tex",
    ".tikz", ".pgf", ".txt", ".dat", ".csv",
}

# Chỉ những diagnostic này mới làm mất nội dung -> status = partial_parse.
DEGRADING_CODES = frozenset(
    {
        "MISSING_INCLUDE",
        "FILE_READ_ERROR",
        "PATH_TRAVERSAL",
        "BIB_FILE_READ_ERROR",
        "BIB_PATH_TRAVERSAL",
    }
)


def _is_degrading(diagnostic: Diagnostic) -> bool:
    """Diagnostic này có làm mất dữ liệu (=> status partial_parse) không?

    MISSING_BIB_FILE chỉ tính là mất dữ liệu khi mức WARNING, tức là không còn nguồn
    tham khảo nào khác (.bib/.bbl) cho version đó.
    """
    if diagnostic.code in DEGRADING_CODES:
        return True
    return (
        diagnostic.code == "MISSING_BIB_FILE"
        and diagnostic.severity is DiagnosticSeverity.WARNING
    )


__all__ = [
    "BEGIN_DOCUMENT_PATTERN",
    "BIB_EXTS",
    "DEGRADING_CODES",
    "DOCUMENTCLASS_PATTERN",
    "INCLUDE_PATTERN",
    "NON_TEX_EXTS",
    "Diagnostic",
    "DiagnosticSeverity",
    "ProcessingStatus",
    "SourceBundle",
    "SourceFile",
    "_classify_target",
    "_canonical_path",
    "_collect_bibliography",
    "_collect_files",
    "_find_root_file",
    "_iter_files",
    "_iter_tex_files",
    "_read_text",
    "_remove_comments",
    "_resolve_include",
    "_resolve_resource",
    "gather_version",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


def _mask_blocks(text: str) -> str:
    """Thay nội dung verbatim/lstlisting/comment bằng khoảng trắng, giữ nguyên số dòng."""
    return MASKED_ENV_PATTERN.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _strip_for_scan(text: str) -> str:
    """Văn bản dùng để dò \\documentclass / \\input: không comment, không verbatim."""
    return _mask_blocks(_remove_comments(text))


def _read_text(path: Path) -> tuple[str, str]:
    """Đọc file, trả về (text, encoding). Thử utf-8 -> cp1252 -> latin-1.

    Không dùng errors="replace" vì sẽ biến ký tự có dấu thành U+FFFD (mất dữ liệu).
    Raise OSError nếu không đọc được.
    """
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc), ("utf-8" if enc == "utf-8-sig" else enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def _iter_files(base_dir: Path, suffixes: frozenset[str] | set[str]) -> list[Path]:
    """Liệt kê file có đuôi thuộc `suffixes` (không phân biệt hoa/thường), bỏ rác macOS.

    Bỏ qua symlink trỏ ra ngoài base_dir (relative_path sẽ thành đường dẫn tuyệt đối,
    vi phạm SourceFile contract).
    """
    found: list[Path] = []
    resolved_base = base_dir.resolve(strict=False)
    for p in base_dir.rglob("*"):
        if p.suffix.lower() not in suffixes:
            continue
        if p.name.startswith("._"):
            continue
        try:
            parts = p.relative_to(base_dir).parts
        except ValueError:
            parts = p.parts
        if "__MACOSX" in parts:
            continue
        if not p.is_file():
            continue
        if not p.resolve(strict=False).is_relative_to(resolved_base):
            continue
        found.append(p)
    return found


def _iter_tex_files(base_dir: Path) -> list[Path]:
    """Liệt kê mọi file .tex."""
    return _iter_files(base_dir, {".tex"})


def _canonical_path(path: Path, root: Path) -> Path:
    """Trả về đường dẫn với chữ hoa/thường đúng như tên thật trên đĩa.

    Trên filesystem không phân biệt hoa/thường (macOS, Windows), `\\input{Intro}` mở được
    `intro.tex` nên `Path.resolve()` giữ nguyên tên "Intro.tex" -> relative_path sai.
    Hàm này đi từng thành phần dưới `root` và thay bằng tên thật.
    """
    root = root.resolve(strict=False)
    try:
        rel = path.resolve(strict=False).relative_to(root)
    except ValueError:
        return path

    current = root
    for part in rel.parts:
        try:
            names = [entry.name for entry in os.scandir(current)]
        except OSError:
            return path
        if part in names:
            actual = part
        else:
            matches = sorted(n for n in names if n.lower() == part.lower())
            if not matches:
                return path
            actual = matches[0]
        current = current / actual
    return current


def _classify_target(target: str) -> str:
    """Trả về 'dynamic' | 'non_tex' | 'tex'."""
    t = target.strip().strip('"')
    if not t or any(ch in t for ch in "#\\$"):
        return "dynamic"
    if PurePosixPath(t).suffix.lower() in NON_TEX_EXTS:
        return "non_tex"
    return "tex"


def _candidate_bases(
    current_file: Path,
    source_dir: Path | None,
    root_dir: Path | None,
) -> list[Path]:
    """Thứ tự thư mục dùng để resolve include.

    LaTeX resolve \\input theo thư mục làm việc (= thư mục của root file), nên root_dir
    đứng đầu; sau đó mới đến thư mục của file hiện tại và source_dir (cho dễ tính).
    """
    bases: list[Path] = []
    for b in (root_dir, current_file.parent, source_dir):
        if b is None:
            continue
        r = b.resolve(strict=False)
        if r not in bases:
            bases.append(r)
    return bases


def _is_escaping_source_dir(
    current_file: Path,
    include_target: str,
    source_dir: Path,
    root_dir: Path | None = None,
) -> bool:
    """True nếu target là đường dẫn tuyệt đối hoặc mọi cách resolve đều ra ngoài source_dir."""
    if Path(include_target).is_absolute():
        return True

    resolved_root = source_dir.resolve(strict=False)
    for base in _candidate_bases(current_file, source_dir, root_dir):
        if (base / include_target).resolve(strict=False).is_relative_to(resolved_root):
            return False
    return True


def _resolve_include(
    current_file: Path,
    include_target: str,
    source_dir: Path | None = None,
    root_dir: Path | None = None,
) -> Path | None:
    """Resolve đường dẫn include; thử thêm .tex, rồi fallback không phân biệt hoa/thường."""
    target = include_target.strip().strip('"')
    if not target or Path(target).is_absolute():
        return None

    resolved_root = source_dir.resolve(strict=False) if source_dir is not None else None
    bases = _candidate_bases(current_file, source_dir, root_dir)

    for base in bases:
        candidate = (base / target).resolve(strict=False)
        if candidate.suffix.lower() == ".tex":
            attempts = [candidate]
        else:
            # LaTeX thử "name.tex" trước, rồi mới đến "name".
            attempts = [candidate.parent / f"{candidate.name}.tex", candidate]

        for attempt in attempts:
            if resolved_root is not None and not attempt.is_relative_to(resolved_root):
                continue
            if attempt.is_file():
                if resolved_root is not None:
                    return _canonical_path(attempt, resolved_root)
                return attempt

    # Fallback: sai hoa/thường (\input{Intro} nhưng file là intro.tex).
    want = posixpath.normpath(target.replace("\\", "/")).lower()
    for base in bases:
        if not base.is_dir():
            continue
        for p in sorted(_iter_tex_files(base)):
            rel = p.relative_to(base).as_posix().lower()
            if rel in (want, f"{want}.tex"):
                resolved = p.resolve(strict=False)
                if resolved_root is None or resolved.is_relative_to(resolved_root):
                    return resolved

    return None


def _resolve_resource(
    current_file: Path,
    target: str,
    default_suffix: str,
    source_dir: Path,
    root_dir: Path | None = None,
) -> Path | None:
    """Resolve file tài nguyên (.bib/.bbl); thêm default_suffix nếu thiếu đuôi."""
    t = target.strip().strip('"')
    if not t or Path(t).is_absolute():
        return None

    resolved_root = source_dir.resolve(strict=False)
    bases = _candidate_bases(current_file, source_dir, root_dir)
    names = [t] if t.lower().endswith(tuple(BIB_EXTS)) else [t + default_suffix, t]

    for base in bases:
        for name in names:
            candidate = (base / name).resolve(strict=False)
            if candidate.is_relative_to(resolved_root) and candidate.is_file():
                return _canonical_path(candidate, resolved_root)

    wants = {posixpath.normpath(n.replace("\\", "/")).lower() for n in names}
    for base in bases:
        if not base.is_dir():
            continue
        for p in sorted(_iter_files(base, BIB_EXTS)):
            if p.relative_to(base).as_posix().lower() in wants:
                return p.resolve(strict=False)
    return None


# --------------------------------------------------------------------------- #
# Root detection
# --------------------------------------------------------------------------- #

def _find_root_file(source_dir: Path) -> tuple[Path | None, list[Diagnostic]]:
    r"""Tìm file .tex chứa \documentclass và trả về (root_path, diagnostics).

    Các bước (deterministic, không phụ thuộc thứ tự filesystem):
    1. Lấy mọi file có \documentclass (đã bỏ comment/verbatim).
    2. Ưu tiên file có \begin{document}.
    3. Loại các file bị file khác trong nhóm đó \input/\include (vd. hình standalone, subfile).
    4. Còn nhiều ứng viên -> chọn theo Phương án A:
       độ sâu nông nhất -> main.tex -> alphabet đường dẫn POSIX.
    """
    diagnostics: list[Diagnostic] = []
    if not source_dir.exists() or not source_dir.is_dir():
        return None, diagnostics

    candidates = sorted(
        _iter_tex_files(source_dir),
        key=lambda p: (
            len(p.relative_to(source_dir).parts),
            0 if p.name.lower() == "main.tex" else 1,
            p.relative_to(source_dir).as_posix(),
        ),
    )

    matched_roots: list[Path] = []
    scanned: dict[Path, str] = {}
    for candidate in candidates:
        rel_path = _to_relative_posix(candidate, source_dir)
        try:
            content, _ = _read_text(candidate)
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

        stripped = _strip_for_scan(content)
        if DOCUMENTCLASS_PATTERN.search(stripped):
            resolved = candidate.resolve()
            matched_roots.append(resolved)
            scanned[resolved] = stripped

    if not matched_roots:
        return None, diagnostics

    # Bộ lọc 1: có \begin{document}
    pool = [p for p in matched_roots if BEGIN_DOCUMENT_PATTERN.search(scanned[p])]
    if not pool:
        pool = matched_roots

    # Bộ lọc 2: bỏ file đã bị file khác trong pool include
    resolved_source = source_dir.resolve(strict=False)
    included: set[Path] = set()
    if len(pool) > 1:
        for p in pool:
            files, _, _ = _collect_files(p, source_dir)
            included.update(
                (resolved_source / f.relative_path).resolve(strict=False)
                for f in files[1:]
            )
    top = [p for p in pool if p not in included] or pool

    chosen = top[0]
    chosen_rel = _to_relative_posix(chosen, source_dir)
    if len(top) > 1:
        all_rel = ", ".join(_to_relative_posix(p, source_dir) for p in top)
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="MULTIPLE_ROOTS",
                message=f"Multiple root candidates found ({all_rel}); selected '{chosen_rel}'",
                file_path=chosen_rel,
            )
        )
    elif len(matched_roots) > 1:
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.INFO,
                code="ROOT_AMBIGUITY_RESOLVED",
                message=(
                    f"{len(matched_roots)} files contain \\documentclass; "
                    f"selected '{chosen_rel}'"
                ),
                file_path=chosen_rel,
            )
        )

    return chosen, diagnostics


# --------------------------------------------------------------------------- #
# Include traversal
# --------------------------------------------------------------------------- #

def _collect_files(
    root_file: Path,
    source_dir: Path,
) -> tuple[list[SourceFile], list[Diagnostic], bool]:
    """Duyệt DFS theo thứ tự include, phân biệt lỗi đọc root với lỗi đọc file con.

    Trả về: (files theo thứ tự, diagnostics, root_read_failed)
    """
    resolved_source_dir = source_dir.resolve(strict=False)
    root_resolved = root_file.resolve(strict=False)
    root_dir = root_resolved.parent
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
            text, encoding = _read_text(resolved)
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
            visited.add(resolved)  # tránh báo lỗi lặp lại nếu file bị include nhiều lần
            return

        if encoding != "utf-8":
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="ENCODING_FALLBACK",
                    message=f"File is not valid UTF-8; decoded as {encoding}",
                    file_path=rel_path,
                )
            )

        order.append(
            SourceFile(
                relative_path=rel_path,
                content=text,
                order=len(order),
            )
        )

        stripped = _strip_for_scan(text)
        matches = [(m.start(), m.group(1)) for m in INCLUDE_PATTERN.finditer(stripped)]
        matches += [(m.start(), m.group(1)) for m in BARE_INPUT_PATTERN.finditer(stripped)]
        matches.sort(key=lambda item: item[0])

        for start, raw_target in matches:
            include_target = raw_target.strip().strip('"')
            line_no = stripped[:start].count("\n") + 1

            if not include_target:
                continue

            kind = _classify_target(include_target)
            if (
                kind == "non_tex"
                and PurePosixPath(include_target).suffix.lower() in BIB_EXTS
            ):
                continue  # .bib/.bbl được xử lý bởi _collect_bibliography
            if kind != "tex":
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.INFO,
                        code=(
                            "INCLUDE_SKIPPED_DYNAMIC"
                            if kind == "dynamic"
                            else "NON_TEX_INCLUDE_SKIPPED"
                        ),
                        message=f"Skipped include target: {include_target}",
                        file_path=rel_path,
                        line=line_no,
                    )
                )
                continue

            if _is_escaping_source_dir(
                resolved, include_target, resolved_source_dir, root_dir
            ):
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
                root_dir=root_dir,
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


# --------------------------------------------------------------------------- #
# Bibliography collection (.bib / .bbl)
# --------------------------------------------------------------------------- #

def _collect_bibliography(
    root_file: Path,
    source_dir: Path,
    tex_files: list[SourceFile],
    start_order: int = 0,
) -> tuple[list[SourceFile], list[Diagnostic]]:
    r"""Thu thập .bib/.bbl mà compilation path thực sự tham chiếu.

    `start_order` là `order` của file đầu tiên (nối tiếp sau các file .tex để
    `SourceBundle.files` có order liên tục 0..n-1).

    Nguồn tham chiếu (đã bỏ comment/verbatim):
    - \bibliography{a,b}              -> a.bib, b.bib   (\jobname = tên file root)
    - \addbibresource[..]{a.bib}      -> a.bib          (biblatex)
    - \input{x.bbl} / \input{x.bib}  -> chính file đó
    - <root>.bbl nếu có \bibliography hoặc \printbibliography (file BibTeX sinh ra,
      LaTeX tự đọc nó)
    .bib/.bbl không được tham chiếu thì bị bỏ qua, giống file .tex dư thừa.
    """
    diagnostics: list[Diagnostic] = []
    bib_files: list[SourceFile] = []
    added: set[Path] = set()

    resolved_source = source_dir.resolve(strict=False)
    root_resolved = root_file.resolve(strict=False)
    root_dir = root_resolved.parent
    stem = root_resolved.stem

    # (target, default_suffix, tex SourceFile, line)
    requests: list[tuple[str, str, SourceFile, int]] = []
    uses_bibtex = False

    for sf in tex_files:
        stripped = _strip_for_scan(sf.content)
        found: list[tuple[int, str, str]] = []

        for m in BIBLIOGRAPHY_PATTERN.finditer(stripped):
            uses_bibtex = True
            for name in m.group(1).split(","):
                if name.strip():
                    found.append((m.start(), name.strip(), ".bib"))
        for m in ADDBIB_PATTERN.finditer(stripped):
            uses_bibtex = True
            found.append((m.start(), m.group(1).strip(), ".bib"))
        if PRINTBIB_PATTERN.search(stripped):
            uses_bibtex = True
        inputs = [(m.start(), m.group(1)) for m in INCLUDE_PATTERN.finditer(stripped)]
        inputs += [(m.start(), m.group(1)) for m in BARE_INPUT_PATTERN.finditer(stripped)]
        for start, raw in inputs:
            t = raw.strip().strip('"')
            suffix = PurePosixPath(t).suffix.lower()
            if suffix in BIB_EXTS:
                found.append((start, t, suffix))

        found.sort(key=lambda item: item[0])
        for start, target, suffix in found:
            requests.append((target, suffix, sf, stripped[:start].count("\n") + 1))

    missing: list[tuple[str, SourceFile, int]] = []

    def add(path: Path) -> None:
        if path in added:
            return
        rel = _to_relative_posix(path, resolved_source)
        try:
            text, encoding = _read_text(path)
        except OSError as exc:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIB_FILE_READ_ERROR",
                    message=f"Could not read bibliography file: {exc}",
                    file_path=rel,
                )
            )
            added.add(path)
            return
        added.add(path)
        if encoding != "utf-8":
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="ENCODING_FALLBACK",
                    message=f"File is not valid UTF-8; decoded as {encoding}",
                    file_path=rel,
                )
            )
        bib_files.append(
            SourceFile(relative_path=rel, content=text, order=start_order + len(bib_files))
        )

    for target, suffix, sf, line in requests:
        target = target.replace("\\jobname", stem)
        if "://" in target or any(ch in target for ch in "#\\$"):
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="BIB_RESOURCE_SKIPPED",
                    message=f"Skipped dynamic or remote bibliography target: {target}",
                    file_path=sf.relative_path,
                    line=line,
                )
            )
            continue

        current = resolved_source / sf.relative_path
        if _is_escaping_source_dir(current, target, resolved_source, root_dir):
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIB_PATH_TRAVERSAL",
                    message=f"Bibliography target escapes source directory: {target}",
                    file_path=sf.relative_path,
                    line=line,
                )
            )
            continue

        path = _resolve_resource(current, target, suffix, resolved_source, root_dir)
        if path is None:
            missing.append((target, sf, line))
        else:
            add(path)

    # File .bbl do BibTeX sinh ra: LaTeX chỉ đọc nó khi có \bibliography/\printbibliography.
    if uses_bibtex:
        bbl = _resolve_resource(
            root_resolved, f"{stem}.bbl", ".bbl", resolved_source, root_dir
        )
        if bbl is not None:
            add(bbl)

    for target, sf, line in missing:
        diagnostics.append(
            Diagnostic(
                # Còn dữ liệu tham khảo từ file khác -> chỉ là thông tin.
                severity=(
                    DiagnosticSeverity.INFO if bib_files else DiagnosticSeverity.WARNING
                ),
                code="MISSING_BIB_FILE",
                message=f"Bibliography file was not found: {target}",
                file_path=sf.relative_path,
                line=line,
            )
        )

    return bib_files, diagnostics


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _looks_like_plain_tex(tex_files: list[Path]) -> bool:
    for f in tex_files:
        try:
            text, _ = _read_text(f)
        except OSError:
            continue
        if PLAIN_TEX_PATTERN.search(_strip_for_scan(text)):
            return True
    return False


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

    tex_files = sorted(_iter_tex_files(source_dir), key=lambda p: p.as_posix())
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

        extra: list[Diagnostic] = []
        if _looks_like_plain_tex(tex_files):
            extra.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="PLAIN_TEX_SUSPECTED",
                    message=(
                        "No \\documentclass but \\bye found: source looks like Plain TeX, "
                        "which is not supported"
                    ),
                )
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
                *extra,
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

    # .bib/.bbl được tham chiếu nằm chung trong `files`, đứng sau các file .tex.
    bib_files, bib_diagnostics = _collect_bibliography(
        root_file, source_dir, files, start_order=len(files)
    )
    all_diagnostics = (*all_diagnostics, *bib_diagnostics)

    status = (
        ProcessingStatus.PARTIAL_PARSE
        if any(_is_degrading(d) for d in all_diagnostics)
        else ProcessingStatus.OK
    )
    return SourceBundle(
        arxiv_id=arxiv_id,
        version=version,
        status=status,
        root_file=rel_root,
        files=(*files, *bib_files),
        diagnostics=all_diagnostics,
    )