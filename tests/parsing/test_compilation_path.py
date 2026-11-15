from pathlib import Path

import pytest

from src.parsing.compilation_path import (
    _collect_files,
    _find_root_file,
    _remove_comments,
    _resolve_include,
    gather_version,
)


def write_tex(base: Path, relative_path: str, content: str) -> Path:
    """Create a UTF-8 TeX file under base and return its path."""
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def file_paths(bundle) -> list[str]:
    """Return collected source paths in compilation order."""
    return [source_file.relative_path for source_file in bundle.files]


def diagnostic_codes(bundle) -> set[str]:
    return {diagnostic.code for diagnostic in bundle.diagnostics}


def test_remove_comments_removes_comment_but_preserves_newline():
    source = (
        r"\documentclass{article} % document class" "\n"
        r"\input{intro} % include intro" "\n"
        r"Visible text" "\n"
    )

    cleaned = _remove_comments(source)

    assert cleaned == (
        r"\documentclass{article} " "\n"
        r"\input{intro} " "\n"
        r"Visible text" "\n"
    )
    assert cleaned.count("\n") == source.count("\n")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"100\% complete % comment", r"100\% complete "),
        (r"one\\% comment", r"one\\"),
        (r"one\\\\% comment", r"one\\\\"),
        (r"\% not a comment", r"\% not a comment"),
    ],
)
def test_remove_comments_respects_escaped_percent(source, expected):
    assert _remove_comments(source) == expected


def test_remove_comments_preserves_crlf_line_endings():
    source = "first % comment\r\nsecond\r\n"
    assert _remove_comments(source) == "first \r\nsecond\r\n"


def test_find_root_file_ignores_documentclass_inside_comment(tmp_path):
    write_tex(tmp_path, "fake.tex", "% \\documentclass{fake}\n\\input{x}\n")
    real_root = write_tex(
        tmp_path,
        "main.tex",
        "\\documentclass{article}\n\\begin{document}\nHello\n",
    )

    root_file, diagnostics = _find_root_file(tmp_path)

    assert root_file == real_root.resolve()
    assert diagnostics == []


def test_find_root_file_returns_none_when_no_documentclass(tmp_path):
    write_tex(tmp_path, "content.tex", "\\section{Not a root}\n")

    root_file, diagnostics = _find_root_file(tmp_path)

    assert root_file is None
    assert diagnostics == []


def test_resolve_include_uses_current_file_directory(tmp_path):
    current = write_tex(tmp_path, "chapters/main.tex", "\\documentclass{article}")
    expected = write_tex(tmp_path, "chapters/parts/intro.tex", "Intro")