from __future__ import annotations

from pathlib import Path

import os

from src.parsing.compilation_path import ProcessingStatus, gather_version
from src.parsing.contracts import SourceBundle
from src.parsing.validation import validate_source_bundle

DOC = "\\documentclass{article}\n\\begin{document}\n%s\n\\end{document}\n"


def make(base: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
    return base


def codes(bundle) -> list[str]:
    return [d.code for d in bundle.diagnostics]


def paths(bundle) -> list[str]:
    return [f.relative_path for f in bundle.files]


# TC01
def test_single_file(tmp_path):
    make(tmp_path, {"main.tex": DOC % "Hello."})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert b.root_file == "main.tex"
    assert paths(b) == ["main.tex"]
    assert b.files[0].order == 0
    assert b.diagnostics == ()


# TC02
def test_nested_include_comment_and_unused_file(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{intro}\n% \\input{commented}\n\\include{sections/method}",
        "intro.tex": "Intro. \\input{sections/deep}",
        "sections/deep.tex": "Deep.",
        "sections/method.tex": "Method.",
        "commented.tex": "SHOULD NOT BE INCLUDED",
        "unused.tex": "Unused.",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert paths(b) == ["main.tex", "intro.tex", "sections/deep.tex", "sections/method.tex"]
    assert [f.order for f in b.files] == [0, 1, 2, 3]


# TC09
def test_empty_no_tex(tmp_path):
    make(tmp_path, {"figure.png": b"\x89PNG"})
    b = gather_version(tmp_path, "2501.01234", 2)
    assert b.status == ProcessingStatus.EMPTY_NO_TEX
    assert b.root_file is None and b.files == ()


def test_source_dir_missing(tmp_path):
    b = gather_version(tmp_path / "nope", "2501.01234", 1)
    assert b.status == ProcessingStatus.ERROR
    assert codes(b) == ["SOURCE_DIR_NOT_FOUND"]


# TC11
def test_missing_include_is_partial_parse(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "A\n\\input{sections/method}\n\\input{intro}",
        "intro.tex": "Intro.",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.PARTIAL_PARSE
    missing = [d for d in b.diagnostics if d.code == "MISSING_INCLUDE"]
    assert len(missing) == 1
    assert missing[0].file_path == "main.tex"
    assert missing[0].line == 4
    assert paths(b) == ["main.tex", "intro.tex"]


def test_root_not_found_plain_tex(tmp_path):
    make(tmp_path, {"paper.tex": "Some text\n\\bye\n"})
    b = gather_version(tmp_path, "2609.25098", 1)
    assert b.status == ProcessingStatus.ROOT_NOT_FOUND
    assert "ROOT_NOT_FOUND" in codes(b)
    assert "PLAIN_TEX_SUSPECTED" in codes(b)
    assert b.files == ()


def test_root_not_found_without_plain_tex(tmp_path):
    make(tmp_path, {"fragment.tex": "just text"})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.ROOT_NOT_FOUND
    assert "PLAIN_TEX_SUSPECTED" not in codes(b)


def test_documentclass_in_comment_is_not_root(tmp_path):
    make(tmp_path, {"a.tex": "% \\documentclass{article}\ntext"})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.ROOT_NOT_FOUND


def test_include_cycle(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{a}",
        "a.tex": "\\input{b}",
        "b.tex": "\\input{a}",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert "INCLUDE_CYCLE" in codes(b)
    assert paths(b) == ["main.tex", "a.tex", "b.tex"]
    assert b.status == ProcessingStatus.OK


def test_non_tex_and_dynamic_includes_do_not_cause_partial(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{refs.bbl}\n\\input{fig.pdf_tex}\n\\newcommand{\\x}[1]{\\input{#1}}",
        "refs.bbl": "\\begin{thebibliography}{1}\\end{thebibliography}",
        "fig.pdf_tex": "junk",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert paths(b) == ["main.tex"]
    assert "NON_TEX_INCLUDE_SKIPPED" in codes(b)
    assert "INCLUDE_SKIPPED_DYNAMIC" in codes(b)
    assert "MISSING_INCLUDE" not in codes(b)


def test_includegraphics_is_not_an_include(tmp_path):
    make(tmp_path, {"main.tex": DOC % "\\includegraphics{fig1}\n\\includeonly{a}"})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert codes(b) == []


def test_case_insensitive_include(tmp_path):
    make(tmp_path, {"main.tex": DOC % "\\input{Intro}", "intro.tex": "Intro."})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert paths(b) == ["main.tex", "intro.tex"]


def test_include_with_explicit_tex_extension_and_subfile(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{a.tex}\n\\subfile{b}",
        "a.tex": "A.",
        "b.tex": "B.",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert paths(b) == ["main.tex", "a.tex", "b.tex"]


def test_bare_input(tmp_path):
    make(tmp_path, {"main.tex": DOC % "\\input intro", "intro.tex": "Intro."})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert paths(b) == ["main.tex", "intro.tex"]


def test_include_inside_verbatim_and_comment_env_ignored(tmp_path):
    body = (
        "\\begin{verbatim}\n\\input{ghost1}\n\\end{verbatim}\n"
        "\\begin{comment}\n\\input{ghost2}\n\\end{comment}\n"
        "\\input{real}"
    )
    make(tmp_path, {"main.tex": DOC % body, "real.tex": "Real."})
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert paths(b) == ["main.tex", "real.tex"]


def test_include_relative_to_root_dir_not_including_file(tmp_path):
    # LaTeX resolve \input theo thư mục của root: sections/a.tex gọi \input{sections/b}
    make(tmp_path, {
        "main.tex": DOC % "\\input{sections/a}",
        "sections/a.tex": "\\input{sections/b}",
        "sections/b.tex": "B.",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert paths(b) == ["main.tex", "sections/a.tex", "sections/b.tex"]


def test_path_traversal(tmp_path):
    src = tmp_path / "src"
    make(tmp_path, {"outside.tex": "secret"})
    make(src, {"main.tex": DOC % "\\input{../outside}\n\\input{/etc/passwd}"})
    b = gather_version(src, "2501.01234", 1)
    assert b.status == ProcessingStatus.PARTIAL_PARSE
    assert codes(b).count("PATH_TRAVERSAL") == 2
    assert paths(b) == ["main.tex"]


# Multiple roots
def test_multiple_roots_resolved_when_one_is_included(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{figs/plot}",
        "figs/plot.tex": "\\documentclass{standalone}\n\\begin{document}X\\end{document}",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.root_file == "main.tex"
    assert b.status == ProcessingStatus.OK
    assert "MULTIPLE_ROOTS" not in codes(b)
    assert "ROOT_AMBIGUITY_RESOLVED" in codes(b)


def test_multiple_roots_prefers_file_with_begin_document(tmp_path):
    make(tmp_path, {
        "a_macros.tex": "\\documentclass{article}\n\\newcommand{\\x}{y}",
        "z_paper.tex": DOC % "Body",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.root_file == "z_paper.tex"


def test_multiple_roots_real_ambiguity_is_deterministic_and_not_partial(tmp_path):
    make(tmp_path, {
        "paper.tex": DOC % "P",
        "main.tex": DOC % "M",
        "response.tex": DOC % "R",
    })
    b1 = gather_version(tmp_path, "2501.01234", 1)
    b2 = gather_version(tmp_path, "2501.01234", 1)
    assert b1 == b2
    assert b1.root_file == "main.tex"
    assert b1.status == ProcessingStatus.OK
    assert "MULTIPLE_ROOTS" in codes(b1)


def test_subfiles_are_not_roots(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\subfile{chap1}",
        "chap1.tex": "\\documentclass[main.tex]{subfiles}\n\\begin{document}C1\\end{document}",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.root_file == "main.tex"
    assert paths(b) == ["main.tex", "chap1.tex"]


# Encoding / filesystem noise
def test_latin1_file_is_decoded_without_replacement_chars(tmp_path):
    make(tmp_path, {
        "main.tex": DOC % "\\input{a}",
        "a.tex": "Caf\xe9".encode("latin-1"),
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert "\ufffd" not in b.files[1].content
    assert "Café" in b.files[1].content
    assert "ENCODING_FALLBACK" in codes(b)
    assert b.status == ProcessingStatus.OK


def test_macosx_resource_forks_and_uppercase_extension(tmp_path):
    make(tmp_path, {
        "MAIN.TEX": DOC % "Hi",
        "__MACOSX/._main.tex": b"\x00\x05\x16\x07",
    })
    b = gather_version(tmp_path, "2501.01234", 1)
    assert b.status == ProcessingStatus.OK
    assert b.root_file == "MAIN.TEX"


def test_symlink_pointing_outside_source_dir_is_ignored(tmp_path):
    src = tmp_path / "src"
    make(tmp_path, {"outside.tex": DOC % "secret"})
    make(src, {"main.tex": DOC % "Hi"})
    os.symlink(tmp_path / "outside.tex", src / "link.tex")
    b = gather_version(src, "2501.01234", 1)  # không được raise
    assert b.root_file == "main.tex"
    assert paths(b) == ["main.tex"]


def test_all_bundles_satisfy_contract_and_validation(tmp_path):
    scenarios = {
        "single": {"main.tex": DOC % "Hi"},
        "nested": {"main.tex": DOC % "\\input{a}", "a.tex": "A"},
        "missing": {"main.tex": DOC % "\\input{nope}"},
        "empty": {"x.png": b"png"},
        "no_root": {"x.tex": "text"},
        "multi_root": {"main.tex": DOC % "M", "other.tex": DOC % "O"},
    }
    for name, files in scenarios.items():
        d = tmp_path / name
        make(d, files)
        b = gather_version(d, "2501.01234", 1)
        assert isinstance(b, SourceBundle)
        assert validate_source_bundle(b) == (), name