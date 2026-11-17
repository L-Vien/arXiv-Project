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
# Legacy source formats found in arXiv submissions. They are deliberately
# narrow fallbacks: a normal \documentclass root always wins.
OBFUSCATED_DOCUMENTCLASS_PATTERN = re.compile(
    r"\\csname\s*documentclass\s*\\endcsname(?![A-Za-z@])"
)
AMS_DOCUMENTSTYLE_PATTERN = re.compile(
    r"(?m)^[ \t]*\\documentstyle(?![A-Za-z@])"
)
AMS_DOCUMENT_PATTERN = re.compile(r"(?m)^[ \t]*\\document(?![A-Za-z@])")
AMS_ENDDOCUMENT_PATTERN = re.compile(r"(?m)^[ \t]*\\enddocument(?![A-Za-z@])")
PLAIN_TEX_PATTERN = re.compile(r"(?m)^[ \t]*\\bye(?![A-Za-z@])")

# Ví dụ cần nhận diện:
# \input{intro}
# \input{sections/method}
# \include{appendix}
INCLUDE_PATTERN = re.compile(
    r"\\(?:input|include)\s*\{([^{}]+)\}"
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
# A real citation is evidence that an otherwise undeclared .bib/.bbl file is
# likely part of the source bundle.  ``\\nocite{*}`` intentionally does not
# match: it requests all known entries but does not identify a missing source.
CITATION_COMMAND_PATTERN = re.compile(
    r"\\(?:[A-Za-z@]*cite[A-Za-z@]*)(?![A-Za-z@])"
    r"(?:\s*\[[^\]]*\])*\s*\{[^{}]*[A-Za-z0-9][^{}]*\}"
)

# Safe, zero-expansion subset of path macros used in real source archives.
# The value is deliberately limited to a static relative path.  Any macro
# with arguments, nested braces, or dynamic TeX remains unsupported.
SIMPLE_PATH_MACRO_PATTERN = re.compile(
    r"(?:"
    r"\\(?P<command_kind>providecommand|newcommand|renewcommand)"
    r"\s*\{\\(?P<command_name>[A-Za-z@]+)\}"
    r"|\\def\s*\\(?P<def_name>[A-Za-z@]+)"
    r")\s*\{(?P<value>[A-Za-z0-9_./-]+)\}"
)
PATH_MACRO_USE_PATTERN = re.compile(r"\\([A-Za-z@]+)[ \t]*")

# Include các file này không phải TeX source -> bỏ qua, không báo MISSING_INCLUDE.
NON_TEX_EXTS = {
    ".bbl", ".bib", ".sty", ".cls", ".def", ".cfg",
    ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".pdf_tex",
    ".tikz", ".pgf", ".tpx", ".txt", ".dat", ".csv",
    # LaTeX font/language definitions and generated auxiliary files.  These
    # are not paper content and are commonly unavailable in source archives.
    ".aux", ".bcf", ".clo", ".fd", ".ldf", ".lof", ".lot",
    ".nav", ".out", ".snm", ".toc", ".vrb",
}

# Bare ``\input`` names provided by common TeX installations rather than the
# publication archive.  If absent, report them but do not mark article content
# as incomplete.  A matching local file is still collected normally.
TEX_SUPPORT_INPUT_NAMES = frozenset({"epsf", "glyphtounicode", "xy"})
TEX_SUPPORT_INPUT_SUFFIXES = frozenset({".tikzstyles"})

# Chỉ những diagnostic này mới làm mất nội dung -> status = partial_parse.
DEGRADING_CODES = frozenset(
    {
        "MISSING_INCLUDE",
        "FILE_READ_ERROR",
        "PATH_TRAVERSAL",
        "BIB_FILE_READ_ERROR",
        "BIB_PATH_TRAVERSAL",
        "ROOT_SELECTION_LOW_CONFIDENCE",
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
    "is_known_tex_support_input",
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


_ReadResult = tuple[str, str] | OSError


def _read_cached(
    path: Path,
    cache: dict[Path, _ReadResult],
) -> tuple[str, str]:
    """Read a file at most once while gathering one source version."""

    resolved = path.resolve(strict=False)
    result = cache.get(resolved)
    if result is None:
        try:
            result = _read_text(resolved)
        except OSError as exc:
            result = exc
        cache[resolved] = result
    if isinstance(result, OSError):
        raise result
    return result


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


def _register_path_macros(text: str, macros: dict[str, str]) -> None:
    """Record static path macros declared before an include."""
    for definition in SIMPLE_PATH_MACRO_PATTERN.finditer(text):
        kind = definition.group("command_kind") or "def"
        name = definition.group("command_name") or definition.group("def_name")
        value = definition.group("value")
        if kind == "providecommand":
            macros.setdefault(name, value)
        else:
            macros[name] = value


def _expand_path_macros(target: str, macros: dict[str, str]) -> str:
    """Expand only known static path macros, with a bounded cycle guard."""
    expanded = target
    seen: set[str] = set()
    for _ in range(16):
        if expanded in seen:
            break
        seen.add(expanded)
        replacement = PATH_MACRO_USE_PATTERN.sub(
            lambda match: macros.get(match.group(1), match.group(0)),
            expanded,
        )
        if replacement == expanded:
            break
        expanded = replacement
    return expanded


def is_known_tex_support_input(target: str) -> bool:
    """Identify bare inputs supplied by common TeX installations.

    TV2 and TV3 share this classifier so that an absent support resource does
    not become a false missing-content diagnostic.
    """
    path = PurePosixPath(target.strip().strip('"'))
    return (
        path.stem.lower() in TEX_SUPPORT_INPUT_NAMES
        or path.name.lower() in TEX_SUPPORT_INPUT_NAMES
        or path.suffix.lower() in TEX_SUPPORT_INPUT_SUFFIXES
    )


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

_ROOT_PREFERRED_NAMES = re.compile(
    r"(?:^|[_ .-])(?:main|manuscript|paper|article|arxiv|final|draft)(?:$|[_ .-])",
    re.IGNORECASE,
)
_ROOT_AUXILIARY_NAMES = re.compile(
    r"(?:^|[_ .-])(?:supp|supplement|supplementary|si|sm|coverletter|"
    r"cover|response|rebuttal|template|checklist|preamble|figure)(?:$|[_ .-])",
    re.IGNORECASE,
)
_ROOT_AUXILIARY_DIRECTORIES = frozenset(
    {"archive", "backup", "old", "scratch"}
)


def _root_candidate_score(path: Path, source_dir: Path, text: str) -> int:
    """Rank plausible compilation roots using deterministic semantic signals."""

    resolved_source = source_dir.resolve(strict=False)
    relative = path.resolve(strict=False).relative_to(resolved_source).as_posix()
    relative_parts = PurePosixPath(relative).parts
    name = path.stem.lower()
    score = 0
    if path.name.lower() == "main.tex" and len(relative_parts) == 1:
        score += 500
    if _ROOT_PREFERRED_NAMES.search(name):
        score += 100
    if _ROOT_AUXILIARY_NAMES.search(name):
        score -= 300
    if any(
        part.lower() in _ROOT_AUXILIARY_DIRECTORIES
        for part in relative_parts[:-1]
    ):
        score -= 300
    if re.search(r"(?:^|[_ .-])old(?:$|[_ .-])", name):
        score -= 100

    revision = re.search(r"(?:^|[_ .-])(?:v|rev)(\d+)(?:$|[_ .-])", name)
    if revision:
        score += min(int(revision.group(1)), 20)

    section_count = len(
        re.findall(
            r"\\(?:part|chapter|section|subsection)(?![A-Za-z@])",
            text,
        )
    )
    score += min(section_count * 5, 100)
    if re.search(r"\\(?:begin\s*\{abstract\}|abstract\s*\{)", text):
        score += 20
    if re.search(r"\\title(?![A-Za-z@])", text):
        score += 10
    if re.search(r"\\(?:bibliography|printbibliography)(?![A-Za-z@])", text):
        score += 10
    if re.search(r"\\end\s*\{document\}", text):
        score += 10
    score += min(len(text) // 5000, 30)
    score -= max(len(relative_parts) - 1, 0) * 5
    return score


def _outer_root_wrappers(
    scanned: dict[Path, str],
    matched_roots: list[Path],
    source_dir: Path,
) -> list[Path]:
    """Find substantive outer files that include a documentclass fragment."""

    matched = set(matched_roots)
    wrappers: list[Path] = []
    resolved_source = source_dir.resolve(strict=False)
    for candidate, text in scanned.items():
        if candidate in matched:
            continue
        if not re.search(r"\\end\s*\{document\}", text):
            continue
        if not re.search(
            r"\\(?:title|abstract|part|chapter|section)(?![A-Za-z@])",
            text,
        ):
            continue

        requests = [
            match.group(1)
            for match in INCLUDE_PATTERN.finditer(text)
        ]
        requests.extend(
            match.group(1)
            for match in BARE_INPUT_PATTERN.finditer(text)
        )
        if any(
            _resolve_include(
                candidate,
                target,
                source_dir=resolved_source,
                root_dir=candidate.parent,
            )
            in matched
            for target in requests
        ):
            wrappers.append(candidate)
    return wrappers

def _find_root_file(
    source_dir: Path,
    read_cache: dict[Path, _ReadResult] | None = None,
) -> tuple[Path | None, list[Diagnostic]]:
    r"""Tìm file .tex chứa \documentclass và trả về (root_path, diagnostics).

    Các bước (deterministic, không phụ thuộc thứ tự filesystem):
    1. Lấy mọi file có \documentclass (đã bỏ comment/verbatim).
    2. Ưu tiên file có \begin{document}.
    3. Loại các file bị file khác trong nhóm đó \input/\include (vd. hình standalone).
    4. Còn nhiều ứng viên -> chọn theo Phương án A:
       độ sâu nông nhất -> main.tex -> alphabet đường dẫn POSIX.
    """
    diagnostics: list[Diagnostic] = []
    read_cache = {} if read_cache is None else read_cache
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
            content, _ = _read_cached(candidate, read_cache)
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
        resolved = candidate.resolve()
        scanned[resolved] = stripped
        if DOCUMENTCLASS_PATTERN.search(stripped):
            matched_roots.append(resolved)

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
            files, _, _ = _collect_files(p, source_dir, read_cache)
            included.update(
                (resolved_source / f.relative_path).resolve(strict=False)
                for f in files[1:]
            )
    top = [p for p in pool if p not in included] or pool
    wrappers = _outer_root_wrappers(
        scanned,
        matched_roots,
        source_dir,
    )
    root_options = list(dict.fromkeys((*top, *wrappers)))
    ranked = sorted(
        root_options,
        key=lambda path: (
            -_root_candidate_score(path, source_dir, scanned[path]),
            len(
                path.relative_to(source_dir.resolve(strict=False)).parts
            ),
            path.relative_to(source_dir.resolve(strict=False)).as_posix(),
        ),
    )
    chosen = ranked[0]
    chosen_rel = _to_relative_posix(chosen, source_dir)
    if len(root_options) > 1:
        all_rel = ", ".join(
            _to_relative_posix(path, source_dir)
            for path in root_options
        )
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="MULTIPLE_ROOTS",
                message=f"Multiple root candidates found ({all_rel}); selected '{chosen_rel}'",
                file_path=chosen_rel,
            )
        )
        best_score = _root_candidate_score(chosen, source_dir, scanned[chosen])
        second_score = _root_candidate_score(
            ranked[1],
            source_dir,
            scanned[ranked[1]],
        )
        if best_score == second_score:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="ROOT_SELECTION_LOW_CONFIDENCE",
                    message=(
                        "Root candidates have similar scores; selected "
                        f"'{chosen_rel}'"
                    ),
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
    if chosen in wrappers:
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.INFO,
                code="ROOT_WRAPPER_SELECTED",
                message=(
                    "Selected an outer source file that includes the "
                    "documentclass fragment"
                ),
                file_path=chosen_rel,
            )
        )

    return chosen, diagnostics


def _nonstandard_root_kind(text: str) -> str | None:
    """Return a narrowly recognized non-LaTeX root marker, if present."""
    if OBFUSCATED_DOCUMENTCLASS_PATTERN.search(text):
        return "obfuscated_documentclass"
    if (
        AMS_DOCUMENTSTYLE_PATTERN.search(text)
        and (
            AMS_DOCUMENT_PATTERN.search(text)
            or AMS_ENDDOCUMENT_PATTERN.search(text)
        )
    ):
        return "ams_tex"
    if PLAIN_TEX_PATTERN.search(text):
        return "plain_tex"
    return None


def _find_nonstandard_root_file(
    source_dir: Path,
    read_cache: dict[Path, _ReadResult] | None = None,
) -> tuple[Path | None, str | None]:
    """Find a conservative Plain/AMS-TeX fallback when no LaTeX root exists."""
    candidates: list[tuple[Path, str, str]] = []
    read_cache = {} if read_cache is None else read_cache
    for candidate in _iter_tex_files(source_dir):
        try:
            text, _ = _read_cached(candidate, read_cache)
        except OSError:
            continue
        stripped = _strip_for_scan(text)
        kind = _nonstandard_root_kind(stripped)
        if kind is not None:
            candidates.append((candidate.resolve(strict=False), kind, stripped))

    if not candidates:
        return None, None

    resolved_source = source_dir.resolve(strict=False)
    chosen, kind, _ = min(
        candidates,
        key=lambda item: (
            -_root_candidate_score(item[0], resolved_source, item[2]),
            len(item[0].relative_to(resolved_source).parts),
            item[0].relative_to(resolved_source).as_posix(),
        ),
    )
    return chosen, kind


# --------------------------------------------------------------------------- #
# Include traversal
# --------------------------------------------------------------------------- #

def _collect_files(
    root_file: Path,
    source_dir: Path,
    read_cache: dict[Path, _ReadResult] | None = None,
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
    path_macros: dict[str, str] = {}
    read_cache = {} if read_cache is None else read_cache

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
            text, encoding = _read_cached(resolved, read_cache)
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
        scan_cursor = 0

        for start, raw_target in matches:
            _register_path_macros(stripped[scan_cursor:start], path_macros)
            scan_cursor = start
            original_target = raw_target.strip().strip('"')
            include_target = _expand_path_macros(original_target, path_macros)
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
                support_input = is_known_tex_support_input(include_target)
                diagnostics.append(
                    Diagnostic(
                        severity=(
                            DiagnosticSeverity.INFO
                            if support_input
                            else DiagnosticSeverity.WARNING
                        ),
                        code=(
                            "MISSING_SUPPORT_INCLUDE"
                            if support_input
                            else "MISSING_INCLUDE"
                        ),
                        message=f"Included file was not found: {include_target}",
                        file_path=rel_path,
                        line=line_no,
                    )
                )
                continue

            visit(include_path)

        _register_path_macros(stripped[scan_cursor:], path_macros)

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

    # Some arXiv archives omit the declaration that normally connects their
    # source to an included bibliography file.  When no declared resource was
    # recovered but the document contains actual citations, retain local
    # bibliography files deterministically.  This is intentionally a fallback
    # rather than a general scan, so unused resources remain ignored in normal
    # compilable projects.
    has_citation = any(
        CITATION_COMMAND_PATTERN.search(_strip_for_scan(source_file.content))
        for source_file in tex_files
    )
    if not bib_files and has_citation:
        fallback_files = sorted(
            _iter_files(resolved_source, BIB_EXTS),
            key=lambda path: _to_relative_posix(path, resolved_source),
        )
        for path in fallback_files:
            add(path)
        if bib_files:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="UNDECLARED_BIB_RESOURCE_FALLBACK",
                    message=(
                        "Collected local bibliography resources because "
                        "citations were present but no declared resource was found"
                    ),
                    file_path=_to_relative_posix(root_resolved, resolved_source),
                )
            )

    return bib_files, diagnostics


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

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

    read_cache: dict[Path, _ReadResult] = {}
    root_file, root_diagnostics = _find_root_file(source_dir, read_cache)
    nonstandard_root_kind: str | None = None
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

        root_file, nonstandard_root_kind = _find_nonstandard_root_file(
            source_dir,
            read_cache,
        )
        if root_file is None:
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
                        message=(
                            "No root TeX file with \\documentclass or a supported "
                            f"Plain/AMS-TeX marker was found under {source_dir}"
                        ),
                    ),
                ),
            )
        root_diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="NONSTANDARD_TEX_ROOT",
                message=(
                    "No standard \\documentclass root was found; selected "
                    f"{nonstandard_root_kind} entrypoint"
                ),
                file_path=_to_relative_posix(root_file, source_dir),
            )
        )

    rel_root = _to_relative_posix(root_file, source_dir)
    files, collect_diagnostics, root_read_failed = _collect_files(
        root_file,
        source_dir,
        read_cache,
    )
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
        if nonstandard_root_kind is not None
        or any(_is_degrading(d) for d in all_diagnostics)
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
