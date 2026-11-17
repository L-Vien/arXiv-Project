"""Structural LaTeX parser for IntroDS Milestone 2 (TV3).

Public API: parse_structure(source: SourceBundle) -> VersionTree.

This module constructs a *raw* tree.  Sentence splitting, LaTeX cleanup,
math normalization, citation rewriting, and element IDs belong to TV4.

Pure Python (standard library only); no filesystem or network access.
"""

from __future__ import annotations

import posixpath
import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .compilation_path import NON_TEX_EXTS, is_known_tex_support_input
from .contracts import (
    Diagnostic,
    DiagnosticSeverity,
    NodeKind,
    ProcessingStatus,
    RawTree,
    SourceBundle,
    TreeNode,
)

# A heading command's logical nesting depth (small numbers = high in tree).
_HEADING_LEVELS: dict[str, int] = {
    "part": 1,
    "chapter": 2,
    "section": 3,
    "subsection": 4,
    "subsubsection": 5,
    "paragraph": 6,
    "subparagraph": 7,
}

_LIST_ENVS = {
    "itemize",
    "enumerate",
    "description",
    "list",
    "itemize*",
    "enumerate*",
    "description*",
    "packeditemize",
    "boxlabel",
    "compactitem",
    "inparaenum",
    "compactenum",
    "packed_itemize",     # NEW
    "packed_enumerate", 
    "asparaenum",
    # Common list-like environments supplied by document classes/templates.
    # They are structural lists for M2 even when their implementation lives in
    # an external .sty/.cls file that is not part of SourceBundle.files.
    "IEEEitemize",
    "myitemize",
    "pts",
    "highlights",
    "keypoints",
    "affiliations",
    "romanenumerate",
    "algorithmic",
    "algor",
    "trivlist",
    "tablenotes",
    "TableNotes",
    "indented",
    "biog",
    # Common list wrappers used by journal templates and author sources.
    # They may be defined by an external class/package, so they cannot
    # always be discovered from the gathered source files.
    "Elist",
    "Ilist",
    "IIlist",
    "Steps",
    "pascal",
    "arabiclist",
    "longlist",
    "certalgorithm",
    "addendum",
    "itdesc",
    "ex",
    "icompact",
    "IEEEdescription",
    "tightitemize",
    "tightlist",
    "tightenumerate",
    "nomenclature",
}

# Theorem/proof environments implemented by amsthm and journal classes are
# commonly based on LaTeX's ``trivlist``. Some sources therefore use ``\item``
# directly inside them. Handle them as implicit lists only when a direct item
# appears before any nested explicit list.
_IMPLICIT_LIST_ENVS = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "remark",
    "example",
    "assumption",
    "axiom",
    "proof",
}

_MATH_ENVS = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "gather", "gather*", "multline", "multline*", "flalign", "flalign*",
    "eqnarray", "eqnarray*", "displaymath", "mathdisplay", "dmath", "dmath*",
}
_FIGURE_ENVS = {
    "figure", "figure*", "table", "table*", "tabular", "tabular*",
    "tabularx", "longtable", "longtable*", "sidewaystable", "sidewaysfigure",
    "wrapfigure", "wraptable",
}
_RAW_TEXT_ENVS = {
    "verbatim",
    "verbatim*",
    "lstlisting",
    "minted",
    "Verbatim",
    "alltt",
    "CodeInput",
    "CodeOutput",
    # Treat comments as opaque while discovering includes, just like TV2.
    # The structural parser still handles the environment via _SKIP_ENVS.
    "comment",
}
_SKIP_ENVS = {"comment", "thebibliography"}
_ACK_ENVS = {"acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment", "acks"}
_REFERENCE_TITLES = {
    "references", "reference", "bibliography", "works cited", "literature cited",
    "references and notes",
}
_SIMPLE_STRUCTURAL_MACRO_DEF_RE = re.compile(
    r"""
    \\(?:newcommand|renewcommand|providecommand)
    \s*\{\\(?P<name>[A-Za-z@]+)\}
    (?:\s*\[[0-9]+\])?
    \s*\{
        \s*\\(?P<kind>begin|end)\s*\{(?P<env>[^{}]+)\}
        (?:\[[^\]]*\])?
        \s*
    \}
    """,
    re.VERBOSE,
)
_INCLUDE_RE = re.compile(
    r"\\(?:input|include)(?![A-Za-z])"
    r"\s*(?:\{([^{}]*)\}|([A-Za-z0-9_./-]+))"
)
_SIMPLE_PATH_MACRO_RE = re.compile(
    r"(?:"
    r"\\(?P<command_kind>providecommand|newcommand|renewcommand)"
    r"\s*\{\\(?P<command_name>[A-Za-z@]+)\}"
    r"|\\(?P<def_kind>def)\s*\\(?P<def_name>[A-Za-z@]+)"
    r")\s*\{(?P<value>[^{}]*)\}"
)
_PATH_MACRO_VALUE_RE = re.compile(
    r"(?:\\[A-Za-z@]+|[A-Za-z0-9_./,+-]|[ \t])*"
)

_PATH_MACRO_USE_RE = re.compile(r"\\([A-Za-z@]+)[ \t]*")
_NEWLIST_RE = re.compile(
    r"\\newlist\s*"
    r"\{\s*([A-Za-z@][A-Za-z0-9@]*)\s*\}"
    r"\s*\{\s*(enumerate|itemize|description)\s*\}"
    r"\s*\{\s*(\d+)\s*\}"
)
_PATH_MACRO_USE_RE = re.compile(r"\\([A-Za-z@]+)[ \t]*")
_BEGIN_END_RE = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
_SPECIAL_RE = re.compile(
    r"\\(?:begin\s*\{[^{}]+\}|end\s*\{[^{}]+\}|"
    r"(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?![A-Za-z])|"
    r"(?:let|titleformat\*?|titlespacing\*?|RenewDocumentCommand)(?![A-Za-z@])|"
    r"comment(?![A-Za-z@])|"
    r"def(?=[ \t]*\\[A-Za-z@]+)|"
    r"item(?![A-Za-z])|appendix(?![A-Za-z])|"
    r"bibliography(?![A-Za-z])|printbibliography(?![A-Za-z])|"
    r"verb\*?(?![A-Za-z])|\[|\(|\]|\))|\$\$|\$"

)
_COMMAND_DEF_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)"
    r"\*?(?![A-Za-z@])"
    r"|\\def(?![A-Za-z@])"
)
_PARAGRAPH_BREAK_RE = re.compile(r"(?:\n[ \t]*\n+|\\par(?![A-Za-z])\s*)")
_VERBATIM_BEGIN_RE = re.compile(
    r"\\begin\s*\{("
    + "|".join(
        re.escape(env)
        for env in sorted(_RAW_TEXT_ENVS)
    )
    + r")\}"
)


@dataclass(slots=True)
class _MutableNode:
    kind: str
    content: str | None = None
    children: list[_MutableNode] = field(default_factory=list)
    level: int | None = None
    source_file: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _ListFrame:
    environment: str
    node: _MutableNode
    current_item: _MutableNode | None = None
    local_headings: list[_MutableNode] = field(default_factory=list)
    opening_offset: int | None = None


@dataclass(frozen=True, slots=True)
class _TextSpan:
    start: int
    end: int
    filename: str
    source_line: int | None = None

class _ExpandedSource:
    """Expanded LaTeX with coarse file provenance for debugging."""

    def __init__(self, text: str, spans: list[_TextSpan], fallback: str | None):
        self.text = text
        self.spans = spans
        self.starts = [span.start for span in spans]
        self.fallback = fallback

    def source_file_at(self, offset: int) -> str | None:
        if not self.spans:
            return self.fallback
        index = bisect_right(self.starts, offset) - 1
        if index >= 0 and offset < self.spans[index].end:
            return self.spans[index].filename
        return self.fallback

    def source_line_at(self, offset: int) -> int | None:
        index = bisect_right(self.starts, offset) - 1

        if index < 0 or not (
            0 <= offset < self.spans[index].end
        ):
            return None

        span = self.spans[index]

        if span.source_line is None:
            return None

        return span.source_line + self.text.count(
            "\n", span.start, offset
        )

def _simple_structural_body(body: str) -> tuple[str, str] | None:
    body = body.strip()
    match = _BEGIN_END_RE.match(body)

    if match is None:
        return None

    action, env = match.groups()
    cursor = match.end()

    if action == "begin":
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1

        optional = _balanced(body, cursor, "[", "]")
        if optional is not None:
            cursor = optional[1]

        # Standard LaTeX list has two mandatory arguments.
        if env == "list":
            for _ in range(2):
                while cursor < len(body) and body[cursor].isspace():
                    cursor += 1

                argument = _balanced(body, cursor, "{", "}")
                if argument is None:
                    return None

                cursor = argument[1]

        elif env in {"enumerate", "itemize", "description"}:
            # Support custom list macros such as:
            # \begin{enumerate}{}{\setlength{\itemsep}{0pt}}
            #
            # This is a narrow, best-effort heuristic.
            # Only accept an empty group followed by
            # a group containing layout commands.

            probe = cursor

            while probe < len(body) and body[probe].isspace():
                probe += 1

            first = _balanced(body, probe, "{", "}")

            if first is not None:
                first_content = body[probe + 1:first[1] - 1]

                if not first_content.strip():
                    probe = first[1]

                    while probe < len(body) and body[probe].isspace():
                        probe += 1

                    layout = _balanced(body, probe, "{", "}")

                    if layout is not None:
                        layout_content = body[probe + 1:layout[1] - 1]

                        layout_pattern = (
                            r"(?:\s*\\setlength\s*"
                            r"\{[^{}]+\}\s*\{[^{}]+\}\s*)+"
                        )

                        if re.fullmatch(layout_pattern, layout_content):
                            cursor = layout[1]

    tail = body[cursor:].strip()

    if tail and not (
        action == "end"
        and env in {"eqnarray", "eqnarray*"}
        and tail == r"\noindent"
    ):
        return None

    return action, env

def _collect_list_environments(tex: str) -> set[str]:
    environments = set(_LIST_ENVS)

    for match in _NEWLIST_RE.finditer(_strip_comments(tex)):
        name = match.group(1)
        environments.add(name)

    return environments


def _collect_parameterized_list_macros(
    text: str,
    known: set[str],
) -> dict[str, tuple[str, int, int]]:
    """Find safe parameterized macros that wrap one list environment.

    The list body must be exactly one macro argument between a literal begin
    and end pair.  This supports forms such as ``\\items{...}`` and
    ``\\itm{margin}{...}`` without attempting general macro expansion.
    """
    document_start = re.search(r"\\begin\s*\{document\}", text)
    preamble = text[:document_start.start()] if document_start else text
    result: dict[str, tuple[str, int, int]] = {}

    for match in _COMMAND_DEF_RE.finditer(preamble):
        if match.group().startswith(r"\def"):
            continue

        cursor = match.end()
        while cursor < len(preamble) and preamble[cursor].isspace():
            cursor += 1

        name_arg = _balanced(preamble, cursor, "{", "}")
        if name_arg is None:
            continue

        name_match = re.fullmatch(r"\\([A-Za-z@]+)", name_arg[0].strip())
        if name_match is None:
            continue

        cursor = name_arg[1]
        while cursor < len(preamble) and preamble[cursor].isspace():
            cursor += 1

        arity = _balanced(preamble, cursor, "[", "]")
        if arity is None or not arity[0].strip().isdigit():
            continue
        argument_count = int(arity[0].strip())
        if not 1 <= argument_count <= 9:
            continue

        body = _balanced(preamble, arity[1], "{", "}")
        if body is None:
            continue

        wrapper_body = body[0].strip()
        outer_group = _balanced(wrapper_body, 0, "{", "}")
        if outer_group is not None and outer_group[1] == len(wrapper_body):
            wrapper_body = outer_group[0].strip()

        wrapper = re.fullmatch(
            r"\s*\\begin\s*\{([^{}]+)\}(?:\s*\[[^\]]*\])?\s*"
            r"#([1-9])\s*\\end\s*\{\1\}\s*",
            wrapper_body,
            re.DOTALL,
        )
        if wrapper is None:
            continue

        environment, body_argument = wrapper.groups()
        body_index = int(body_argument)
        if environment in known and body_index <= argument_count:
            result[name_match.group(1)] = (
                environment,
                argument_count,
                body_index,
            )

    return result


def _collect_parameterized_list_openers(
    text: str,
    known: set[str],
) -> dict[str, tuple[str, int]]:
    """Find macros whose body only opens a list environment.

    Sources sometimes define an opener such as
    ``\\newcommand{\\customitemize}[1]{\\begin{enumerate}[label=(#1)]}``
    and close it later with an explicit ``\\end{enumerate}``.  This is
    different from a wrapper macro because the macro argument configures the
    environment rather than containing the list body.
    """
    document_start = re.search(r"\\begin\s*\{document\}", text)
    preamble = text[:document_start.start()] if document_start else text
    result: dict[str, tuple[str, int]] = {}

    for match in _COMMAND_DEF_RE.finditer(preamble):
        if match.group().startswith(r"\def"):
            continue

        cursor = match.end()
        while cursor < len(preamble) and preamble[cursor].isspace():
            cursor += 1

        name_arg = _balanced(preamble, cursor, "{", "}")
        if name_arg is None:
            continue

        name_match = re.fullmatch(r"\\([A-Za-z@]+)", name_arg[0].strip())
        if name_match is None:
            continue

        cursor = name_arg[1]
        while cursor < len(preamble) and preamble[cursor].isspace():
            cursor += 1

        arity_arg = _balanced(preamble, cursor, "[", "]")
        if arity_arg is None or not arity_arg[0].strip().isdigit():
            continue

        arity = int(arity_arg[0].strip())
        if arity < 1:
            continue

        cursor = arity_arg[1]
        while cursor < len(preamble) and preamble[cursor].isspace():
            cursor += 1

        body = _balanced(preamble, cursor, "{", "}")
        if body is None:
            continue

        event = _simple_structural_body(_strip_comments(body[0]))
        if event is None:
            continue

        action, environment = event
        if action == "begin" and environment in known:
            result[name_match.group(1)] = (environment, arity)

    return result

# def _collect_custom_list_environments(
#     text: str, known: set[str],
# ) -> dict[str, bool]:
#     """Recognize simple preamble list wrappers; bool marks an optional argument."""
#     document_start = re.search(r"\\begin\s*\{document\}", text)
#     if document_start is None:
#         return {}
#     preamble = text[:document_start.start()]
#     definitions = re.compile(r"\\(?:newenvironment|renewenvironment)\*?(?![A-Za-z@])")
#     if not definitions.search(preamble):
#         return {}
#     protected = _protected_include_spans(preamble)
#     result: dict[str, bool] = {}
#     consumed = 0

#     def argument(pos, opening="{", closing="}"):
#         while pos < len(preamble) and preamble[pos].isspace():
#             pos += 1
#         return _balanced(preamble, pos, opening, closing)

#     # Accept only simple layout commands around the underlying list.
#     layout = re.compile(
#         r"\s*(?:(?:\\(?:vspace\*?|hspace\*?)\s*\{[^{}]*\}"
#         r"|\\(?:setlength|addtolength)\s*\{[^{}]*\}\s*\{[^{}]*\})\s*)*"
#     )

#     for match in definitions.finditer(preamble):
#         start = match.start()
#         if start < consumed or any(a <= start < b for a, b in protected):
#             continue
#         if start and preamble[start - 1] == "\\":
#             continue
#         name = argument(match.end())
#         if name is None:
#             continue
#         cursor = name[1]
#         count = argument(cursor, "[", "]")
#         default = None
#         if count is not None:
#             cursor = count[1]
#             default = argument(cursor, "[", "]")
#             if default is not None:
#                 cursor = default[1]
#         begin = argument(cursor)
#         if begin is None:
#             continue
#         end = argument(begin[1])
#         if end is None:
#             continue
#         consumed = end[1]
#         env = name[0].strip()
#         # Only zero-argument or single optional-argument wrappers.
#         if not re.fullmatch(r"[A-Za-z@][A-Za-z0-9@_*-]*", env):
#             continue
#         result.pop(env, None)
#         if count is not None and not (
#             (count[0].strip() == "0" and default is None)
#             or (count[0].strip() == "1" and default is not None)
#         ):
#             continue
#         opening = list(_BEGIN_END_RE.finditer(begin[0]))
#         closing = list(_BEGIN_END_RE.finditer(end[0]))
#         if len(opening) != 1 or len(closing) != 1:
#             continue
#         a, b = opening[0], closing[0]
#         base = a.group(2)
#         if a.group(1) != "begin" or b.groups() != ("end", base) or base not in known:
#             continue
#         if not all(layout.fullmatch(part) for part in (
#             begin[0][:a.start()], end[0][:b.start()], end[0][b.end():],
#         )):
#             continue
#         body = begin[0]
#         pos = a.end()
#         while pos < len(body) and body[pos].isspace():
#             pos += 1
#         opt = _balanced(body, pos, "[", "]")
#         if opt is not None:
#             pos = opt[1]
#         if base == "list":
#             for _ in range(2):
#                 while pos < len(body) and body[pos].isspace():
#                     pos += 1
#                 arg = _balanced(body, pos, "{", "}")
#                 if arg is None:
#                     break
#                 pos = arg[1]
#             else:
#                 if layout.fullmatch(body[pos:]):
#                     result[env] = default is not None
#             continue
#         if layout.fullmatch(body[pos:]):
#             result[env] = default is not None
#     return result

def _collect_custom_list_environments(
    text: str, known: set[str],
) -> tuple[dict[str, bool], str]:
    """Recognize simple list wrappers and mask recognized definitions."""
    document_start = re.search(r"\\begin\s*\{document\}", text)
    if document_start is None:
        return {}, text
    preamble_end = document_start.start()
    scan_text = text
    chars = list(text)
    definitions = re.compile(r"\\(?:newenvironment|renewenvironment)\*?(?![A-Za-z@])")
    if not definitions.search(scan_text):
        return {}, text
    protected = _protected_include_spans(scan_text)
    result: dict[str, bool] = {}
    consumed = 0

    def argument(pos, opening="{", closing="}"):
        while pos < len(scan_text) and scan_text[pos].isspace():
            pos += 1
        return _balanced(scan_text, pos, opening, closing)

    # Accept only simple layout commands around the underlying list.
    layout = re.compile(
        r"\s*(?:(?:\\(?:vspace\*?|hspace\*?)\s*\{[^{}]*\}"
        r"|\\(?:setlength|addtolength)\s*\{[^{}]*\}\s*\{[^{}]*\})\s*)*"
    )

    def register(env, optional, start, end):
        result[env] = optional
        mask_definition(start, end)

    def mask_definition(start, end):
        for i in range(start, end):
            if chars[i] != "\n":
                chars[i] = " "

    for match in definitions.finditer(scan_text):
        start = match.start()
        if start < consumed or any(a <= start < b for a, b in protected):
            continue
        if start and scan_text[start - 1] == "\\":
            continue
        name = argument(match.end())
        if name is None:
            continue
        cursor = name[1]
        count = argument(cursor, "[", "]")
        default = None
        if count is not None:
            cursor = count[1]
            default = argument(cursor, "[", "]")
            if default is not None:
                cursor = default[1]
        begin = argument(cursor)
        if begin is None:
            continue
        end = argument(begin[1])
        if end is None:
            continue
        consumed = end[1]
        # Environment declarations are configuration, not document content.
        # Mask every balanced declaration, including non-list wrappers such as
        # proof environments whose implementation may contain \item.
        mask_definition(start, consumed)
        env = name[0].strip()
        # Only zero-argument or single optional-argument wrappers.
        if not re.fullmatch(r"[A-Za-z@][A-Za-z0-9@_*-]*", env):
            continue
        # For body definitions, avoid forward recognition and redefinitions.
        if start >= preamble_end:
            if env in known or env in result:
                continue
            earlier_use = re.search(
                r"\\begin\s*\{\s*" + re.escape(env) + r"\s*\}",
                text[preamble_end:start],
            )
            if earlier_use is not None:
                continue
        result.pop(env, None)
        if count is not None and not (
            (count[0].strip() == "0" and default is None)
            or (count[0].strip() == "1" and default is not None)
        ):
            continue
        opening = list(_BEGIN_END_RE.finditer(begin[0]))
        closing = list(_BEGIN_END_RE.finditer(end[0]))
        if len(opening) != 1 or len(closing) != 1:
            continue
        a, b = opening[0], closing[0]
        base = a.group(2)
        if a.group(1) != "begin" or b.groups() != ("end", base) or base not in known:
            continue
        if not all(layout.fullmatch(part) for part in (
            begin[0][:a.start()], end[0][:b.start()], end[0][b.end():],
        )):
            continue
        body = begin[0]
        pos = a.end()
        while pos < len(body) and body[pos].isspace():
            pos += 1
        opt = _balanced(body, pos, "[", "]")
        if opt is not None:
            pos = opt[1]
        if base == "list":
            for _ in range(2):
                while pos < len(body) and body[pos].isspace():
                    pos += 1
                arg = _balanced(body, pos, "{", "}")
                if arg is None:
                    break
                pos = arg[1]
            else:
                if layout.fullmatch(body[pos:]):
                    register(env, default is not None, start, consumed)
            continue
        if layout.fullmatch(body[pos:]):
            register(env, default is not None, start, consumed)
    return result, "".join(chars)


def _make_diagnostic(
    code: str,
    message: str,
    *,
    file_path: str | None = None,
    line: int | None = None,
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
) -> Diagnostic:
    return Diagnostic(
        severity=severity,
        code=code,
        message=message,
        file_path=file_path,
        line=line,
    )


def _status_name(status: object) -> str:
    return str(getattr(status, "value", status)).lower()


def _strip_comments(tex: str) -> str:
    """Remove real %-comments, preserving line breaks and escaped \\% tokens.

    Verbatim-like environments and \\verb are skipped without interpretation.
    This is lexical preprocessing, not TV4's semantic LaTeX cleanup.
    """
    output: list[str] = []
    n = len(tex)
    i = 0
    while i < n:
        match = _VERBATIM_BEGIN_RE.match(tex, i)
        if match:
            terminator = r"\end{" + match.group(1) + "}"
            end = tex.find(terminator, match.end())
            end = n if end < 0 else end + len(terminator)
            output.append(tex[i:end])
            i = end
            continue
        if tex.startswith(r"\verb", i) and (i + 5 == n or not tex[i + 5].isalpha()):
            j = i + 5
            if j < n and tex[j] == "*":
                j += 1
            if j < n and tex[j] not in " \n\t":
                end = tex.find(tex[j], j + 1)
                if end >= 0:
                    output.append(tex[i:end + 1])
                    i = end + 1
                    continue
        if tex[i] == "%":
            j = i - 1
            while j >= 0 and tex[j] == "\\":
                j -= 1
            escaped = (i - j - 1) % 2 == 1
            if not escaped:
                end = tex.find("\n", i)
                if end < 0:
                    break
                # Retain the newline so preceding and following tokens do not fuse.
                output.append("\n")
                i = end + 1
                continue
        output.append(tex[i])
        i += 1
    return "".join(output)

# Insert this helper after _balanced() in hierarchy.py.
# Requires the existing re import and _balanced helper.
def _heading_config_end(text: str, start: int, token: str) -> int | None:
    """Read known configuration commands without executing their arguments."""
    def skip_space(pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    def group(pos: int, opening: str = "{", closing: str = "}"):
        return _balanced(text, skip_space(pos), opening, closing)

    pos = skip_space(start)
    if token == r"\let":
        # A direct control-sequence assignment, optionally with '='.
        first = re.match(r"\\[A-Za-z@]+", text[pos:])
        if first is None:
            return None
        pos = skip_space(pos + first.end())
        if pos < len(text) and text[pos] == "=":
            pos = skip_space(pos + 1)
        second = re.match(r"\\[A-Za-z@]+", text[pos:])
        return pos + second.end() if second is not None else None

    command = token.rstrip("*")
    if command not in {r"\titleformat", r"\titlespacing", r"\RenewDocumentCommand"}:
        return None
    first = group(pos)
    if first is None or not re.fullmatch(r"\\[A-Za-z@]+", first[0].strip()):
        return None
    pos = first[1]
    # The short starred titleformat form takes only command and format.
    if command == r"\titleformat" and token.endswith("*"):
        arg = group(pos)
        return arg[1] if arg is not None else None
    if command == r"\titleformat":
        pos = skip_space(pos)
        if pos < len(text) and text[pos] == "[":
            arg = group(pos, "[", "]")
            if arg is None:
                return None
            pos = arg[1]
        required = 4
    elif command == r"\titlespacing":
        required = 3
    else:
        required = 2
    for _ in range(required):
        arg = group(pos)
        if arg is None:
            return None
        pos = arg[1]
    if command in {r"\titleformat", r"\titlespacing"}:
        probe = skip_space(pos)
        if probe < len(text) and text[probe] == "[":
            arg = group(probe, "[", "]")
            if arg is None:
                return None
            pos = arg[1]
    return pos



def _protected_include_spans(tex: str) -> list[tuple[int, int]]:
    """Places where a literal \\input must not be treated as an include."""
    spans: list[tuple[int, int]] = []
    until = 0
    for match in _VERBATIM_BEGIN_RE.finditer(tex):
        if match.start() < until:
            continue
        terminator = r"\end{" + match.group(1) + "}"
        end = tex.find(terminator, match.end())
        until = len(tex) if end < 0 else end + len(terminator)
        spans.append((match.start(), until))
    for match in re.finditer(r"\\verb\*?(?![A-Za-z])", tex):
        if any(a <= match.start() < b for a, b in spans):
            continue
        start = match.end()
        if start >= len(tex) or tex[start] in " \n\t":
            continue
        end = tex.find(tex[start], start + 1)
        if end >= 0:
            spans.append((match.start(), end + 1))
    return sorted(spans)


def _safe_relative_path(parent: str, name: str) -> str | None:
    """Resolve an include against the containing file without reading the disk."""
    name = name.strip().replace("\\\\", "/")
    if not name or name.startswith("/"):
        return None
    if not posixpath.splitext(name)[1]:
        name += ".tex"
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(parent), name))
    if candidate in {"..", "."} or candidate.startswith("../"):
        return None
    return candidate


def _resolve_include_path(
    root_file: str,
    current_file: str,
    name: str,
    contents: dict[str, str],
) -> str | None:
    """Resolve an include using available SourceBundle files."""

    name = name.strip()

    # A dotted basename does not necessarily have a real extension.
    # Example: "src/1.introducion" -> "src/1.introducion.tex"
    names = [name]

    if not name.endswith(".tex"):
        names.append(name + ".tex")

    candidates = []

    for candidate_name in names:
        for base_file in (root_file, current_file, ""):
            candidate = _safe_relative_path(base_file, candidate_name)

            if candidate is None:
                continue

            candidates.append(candidate)

            if candidate in contents:
                return candidate

    # Recover a unique case-insensitive match.
    for candidate in dict.fromkeys(candidates):
        matches = [
            path
            for path in contents
            if path.casefold() == candidate.casefold()
        ]

        if len(matches) == 1:
            return matches[0]

    return _safe_relative_path(current_file, name)


_IF_FILE_EXISTS_RE = re.compile(r"\\IfFileExists(?![A-Za-z])")


def _select_known_file_exists_branches(
    tex: str,
    root_file: str,
    current_file: str,
    contents: dict[str, str],
    origin_spans: list[tuple[int, int, int]] | None = None,
) -> str:
    """Select a known true branch and record source mapping."""
    pieces = []
    position = 0
    output_offset = 0

    def keep(start: int, stop: int) -> None:
        nonlocal output_offset

        if start >= stop:
            return

        pieces.append(tex[start:stop])

        if origin_spans is not None:
            origin_spans.append((
                output_offset,
                output_offset + stop - start,
                start,
            ))

        output_offset += stop - start

    protected = _protected_include_spans(tex)

    for match in _IF_FILE_EXISTS_RE.finditer(tex):
        if match.start() < position:
            continue

        if any(a <= match.start() < b for a, b in protected):
            continue

        cursor = match.end()
        arguments = []
        argument_starts = []

        for _ in range(3):
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1

            argument = _balanced(tex, cursor, "{", "}")
            if argument is None:
                break

            arguments.append(argument[0])
            argument_starts.append(cursor + 1)
            cursor = argument[1]

        if len(arguments) != 3:
            continue

        filename, true_branch, false_branch = arguments

        resolved = _resolve_include_path(
            root_file=root_file,
            current_file=current_file,
            name=filename,
            contents=contents,
        )

        # Only select a branch when file presence is confirmed.
        if resolved not in contents:
            continue

        keep(position, match.start())

        keep(
            argument_starts[1],
            argument_starts[1] + len(true_branch),
        )

        position = cursor

    keep(position, len(tex))
    return "".join(pieces)

def _mask_simple_false_blocks(tex: str) -> str:
    """Mask balanced, standalone literal false blocks in a root document."""

    if r"\iffalse" not in tex:
        return tex

    protected = _protected_include_spans(tex)

    document = next((
        m for m in re.finditer(
            r"\\begin\s*\{document\}", tex
        )
        if not any(
            a <= m.start() < b
            for a, b in protected
        )
        and (
            m.start() == 0
            or tex[m.start() - 1] != "\\"
        )
    ), None)

    if document is None:
        return tex

    tokens = re.compile(
        r"\\[A-Za-z@]+|\\[^\n]|[{}]"
    )

    def standalone(m: re.Match[str]) -> bool:
        left = tex.rfind("\n", 0, m.start()) + 1
        right = tex.find("\n", m.end())

        if right < 0:
            right = len(tex)

        return not (
            tex[left:m.start()].strip()
            or tex[m.end():right].strip()
        )

    def false_end(start: int) -> int | None:
        depth = 1

        for token in tokens.finditer(tex, start):
            value = token.group()

            if value in {r"\iffalse", r"\iftrue"}:
                if not standalone(token):
                    return None

                depth += 1

            elif value == r"\fi":
                if not standalone(token):
                    return None

                depth -= 1

                if depth == 0:
                    return token.end()

            elif (
                value in {r"\else", r"\or"}
                or value.startswith(r"\if")
            ):
                return None

        return None

    pieces = []
    cursor = position = document.end()

    pieces.append(tex[:cursor])

    depth = 0

    while (token := tokens.search(tex, position)) is not None:

        position = token.end()

        opaque_end = next((
            b for a, b in protected
            if a <= token.start() < b
        ), None)

        if opaque_end is not None:
            position = opaque_end
            continue

        value = token.group()

        if value == "{":
            depth += 1

        elif value == "}":
            depth -= 1

            if depth < 0:
                break

        elif value in {r"\begin", r"\end"}:

            env = _BEGIN_END_RE.match(
                tex, token.start()
            )

            if env is not None and env.groups() in {
                ("begin", "comment"),
                ("end", "document"),
            }:
                # Avoid interpreting opaque comments
                # or content after the document ends.
                break

        elif (
            value == r"\iffalse"
            and depth == 0
            and standalone(token)
        ):

            end = false_end(token.end())

            if end is None:
                # Leave ambiguous blocks unchanged.
                break

            pieces.append(
                tex[cursor:token.start()]
            )

            pieces.append("".join(
                ch if ch in "\r\n" else " "
                for ch in tex[token.start():end]
            ))

            cursor = position = end

    pieces.append(tex[cursor:])

    return "".join(pieces)


def _flat_literal_scope(text: str, start: int, stop: int, *,
                        check_envs: bool = True) -> bool:
    """Check literal grouping/conditional balance, not TeX execution."""
    protected = _protected_include_spans(text[:stop])
    tokens = re.compile(r"\\[A-Za-z@]+|\\[^\n]|[{}]")
    braces = conditions = groups = 0
    environments: list[str] = []
    pos = start
    while (m := tokens.search(text, pos, stop)) is not None:
        pos = m.end()
        opaque = next((b for a, b in protected if a <= m.start() < b), None)
        if opaque is not None:
            pos = opaque
            continue
        token = m.group()
        if token == "{":
            braces += 1
        elif token == "}":
            braces -= 1
        elif braces == 0:
            if token in {r"\begin", r"\end"}:
                env = _BEGIN_END_RE.match(text, m.start())
                if env is not None:
                    action, name = env.groups()
                    if name == "comment":
                        return False
                    if check_envs:
                        if action == "begin":
                            environments.append(name)
                        elif not environments or environments.pop() != name:
                            return False
                    pos = env.end()
            elif token.startswith(r"\if"):
                conditions += 1
            elif token == r"\fi":
                conditions -= 1
            elif token in {r"\begingroup", r"\bgroup"}:
                groups += 1
            elif token in {r"\endgroup", r"\egroup"}:
                groups -= 1
        if min(braces, conditions, groups) < 0:
            return False
    return not (braces or conditions or groups or environments)


def _known_root_flags(root_tex: str, contents: Mapping[str, str]) -> dict[str, int]:
    """Recognize standalone root-body flags with only literal ifdefined uses.

    Values are original root line numbers. No assumption that absent means false.
    """
    if not any(r"\ifdefined" in content for content in contents.values()):
        return {}
    document = re.search(r"\\begin\s*\{document\}", root_tex)
    if document is None:
        return {}
    # Restrict to explicit numeric flags used in included files only.
    defs = re.compile(r"^[ \t]*\\def[ \t]*\\([A-Za-z@]+)[ \t]*\{1\}[ \t]*\r?$", re.M)
    flags = {}
    protected = _protected_include_spans(root_tex)
    if any(a <= document.start() < b for a, b in protected):
        return {}
    for m in defs.finditer(root_tex, document.end()):
        if any(a <= m.start() < b for a, b in protected):
            continue
        name = m.group(1)
        use = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z@])")
        guard = re.compile(r"[ \t]*\\ifdefined[ \t]*\\" + re.escape(name) + r"[ \t]*\r?")
        if len(list(use.finditer(root_tex))) != 1:
            continue
        count = 0
        safe = True
        for content in contents.values():
            if content is root_tex:
                continue
            for occurrence in use.finditer(content):
                left = content.rfind("\n", 0, occurrence.start()) + 1
                right = content.find("\n", occurrence.end())
                if right < 0:
                    right = len(content)
                if not guard.fullmatch(content[left:right]):
                    safe = False
                    break
                count += 1
            if not safe:
                break
        if safe and count and _flat_literal_scope(root_tex, document.end(), m.start()):
            flags[name] = root_tex.count("\n", 0, m.start()) + 1
    return flags


def _mask_known_defined_branches(tex: str, flags: frozenset[str]) -> str:
    """Mask only balanced, non-nested standalone guards of proven flags."""
    if not flags or r"\ifdefined" not in tex:
        return tex
    guard = re.compile(r"^[ \t]*\\ifdefined[ \t]*\\([A-Za-z@]+)[ \t]*\r?$", re.M)
    markers = re.compile(r"\\(?:if[A-Za-z@]*|else|fi|or)(?![A-Za-z@])")
    for m in list(guard.finditer(tex)):
        if m.group(1) not in flags or not _flat_literal_scope(tex, 0, m.start(), check_envs=False):
            continue
        protected = _protected_include_spans(tex)
        if any(a <= m.start() < b for a, b in protected):
            continue
        branch = None
        end = None
        for marker in markers.finditer(tex, m.end()):
            left = tex.rfind("\n", 0, marker.start()) + 1
            right = tex.find("\n", marker.end())
            if right < 0:
                right = len(tex)
            if (tex[left:marker.start()].strip() or tex[marker.end():right].strip()
                    or any(a <= marker.start() < b for a, b in protected)):
                break
            if marker.group() == r"\else" and branch is None:
                branch = marker.start()
            elif marker.group() == r"\fi":
                end = marker
                break
            else:
                break
        if end is None:
            continue
        if any(a < end.end() and b > m.start() for a, b in protected):
            continue
        if re.search(r"\\(?:begin|end)\s*\{(?:comment|document)\}", tex[m.end():end.start()]):
            continue
        def blank(s: str) -> str:
            return "".join(c if c in "\r\n" else " " for c in s)
        cut = branch if branch is not None else end.start()
        if not _flat_literal_scope(tex, m.end(), cut, check_envs=False):
            continue
        if branch is not None and not _flat_literal_scope(
            tex, branch + len(r"\else"), end.start(), check_envs=False
        ):
            continue
        tex = (tex[:m.start()] + blank(tex[m.start():m.end()])
               + tex[m.end():cut] + blank(tex[cut:end.end()]) + tex[end.end():])
    return tex


def _recover_declared_float_end(text: str, raw: str, position: int, env: str) -> int | None:
    """Bounded fallback for literal float closing macros declared in the preamble."""
    if env not in {"figure", "figure*", "table", "table*"}:
        return None
    doc = re.search(r"\\begin\s*\{document\}", raw)
    if doc is None:
        return None
    definitions = re.compile(r"^[ \t]*\\newcommand[ \t]*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))[ \t]*(?:\[([1-9])\])?[ \t]*", re.M)
    recovered_ends = []
    protected = _protected_include_spans(raw[:doc.start()])
    for definition in definitions.finditer(raw, 0, doc.start()):
        if any(a <= definition.start() < b for a, b in protected):
            continue
        name = definition.group(1) or definition.group(2)
        body = _balanced(raw, definition.end(), "{", "}")
        if body is None or body[1] > doc.start():
            continue
        closings = [m.group(2) for m in _BEGIN_END_RE.finditer(body[0]) if m.group(1) == "end"]
        if not closings or closings[-1] != env or any(x not in {"center", "tabular", "tabular*"} for x in closings[:-1]):
            continue
        # Allow only literal inner closes, layout, and one balanced caption.
        rest = body[0]
        for inner in closings[:-1]:
            match = re.match(r"\s*\\end\s*\{" + re.escape(inner) + r"\}", rest)
            if match is None:
                break
            rest = rest[match.end():]
        else:
            rest = re.sub(r"^\s*\\vskip\s*[+-]?[0-9]+(?:\.[0-9]+)?\s*(?:ex|em|pt)\b", "", rest, count=1)
            caption = re.match(r"\s*\\caption\s*", rest)
            if caption is not None:
                arg = _balanced(rest, caption.end(), "{", "}")
                if arg is None or re.search(r"\\(?:begin|end|if[A-Za-z@]*|else|fi)\b", arg[0]):
                    continue
                rest = rest[arg[1]:]
            if not re.fullmatch(r"\s*\\end\s*\{" + re.escape(env) + r"\}\s*", rest):
                continue
            if not _flat_literal_scope(raw, 0, definition.start()):
                continue
            command = r"\\" + re.escape(name) + r"(?![A-Za-z@])"
            if re.search(command, raw[:definition.start()]):
                continue
            binding = r"\\(?:[a-zA-Z]*def|let|futurelet|[A-Za-z]*command|[A-Za-z]*Command)\*?\s*(?:\{\s*)?" + command
            if re.search(binding, raw[body[1]:]):
                continue
            call = re.compile(r"^[ \t]*" + command, re.M).search(text, position)
            if call is None:
                continue
            prefix = text[position:call.start()]
            if re.search(r"\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph|if[A-Za-z@]*|else|fi)(?![A-Za-z@])", prefix):
                continue
            if r"\begin{comment}" in prefix or _protected_include_spans(prefix):
                continue
            stack = [env]
            for event in _BEGIN_END_RE.finditer(prefix):
                action, current = event.groups()
                if action == "begin":
                    stack.append(current)
                elif not stack or stack.pop() != current:
                    break
            else:
                if list(reversed(stack)) != closings:
                    continue
                cursor = call.end()
                for _ in range(int(definition.group(3) or 0)):
                    while cursor < len(text) and text[cursor] in " \t":
                        cursor += 1
                    arg = _balanced(text, cursor, "{", "}")
                    if arg is None or re.search(r"\\(?:begin|end|if[A-Za-z@]*|else|fi)\b", arg[0]):
                        break
                    cursor = arg[1]
                else:
                    line_end = text.find("\n", cursor)
                    if line_end < 0:
                        line_end = len(text)
                    if not text[cursor:line_end].strip():
                        recovered_ends.append(cursor)
    return min(recovered_ends, default=None)


def _expand_bundle(source: SourceBundle, diagnostics: list[Diagnostic]) -> _ExpandedSource:
    """Expand includes at their actual insertion positions using provided files.

    TV2 should own compilation-path discovery.  TV3 only reconstructs a stream
    from SourceBundle.files; no filesystem lookup occurs here.
    """
    contents: dict[str, str] = {}
    for source_file in source.files:
        path = getattr(source_file, "relative_path", None) or getattr(source_file, "path", None)
        if not path:
            diagnostics.append(_make_diagnostic("INVALID_SOURCE_FILE", "SourceFile has no path"))
            continue
        path = posixpath.normpath(str(path).replace("\\\\", "/"))
        contents[path] = _strip_comments(source_file.content)

    root = posixpath.normpath(str(source.root_file)) if source.root_file else None
    if not root or root not in contents:
        diagnostics.append(_make_diagnostic(
            "ROOT_NOT_FOUND",
            "Root TeX file missing from SourceBundle",
            file_path=root,
        ))

        return _ExpandedSource("", [], root)

    root_flags = _known_root_flags(contents[root], contents)
    parts: list[str] = []
    spans: list[_TextSpan] = []
    offset = 0
    path_macros: dict[str, str] = {}

    def register_path_macros(fragment: str) -> None:
        for definition in _SIMPLE_PATH_MACRO_RE.finditer(fragment):
            kind = (
                definition.group("command_kind")
                or definition.group("def_kind")
            )
            name = (
                definition.group("command_name")
                or definition.group("def_name")
            )
            value = definition.group("value")

            if not _PATH_MACRO_VALUE_RE.fullmatch(value):
                continue

            if kind == "providecommand":
                path_macros.setdefault(name, value)
            else:
                path_macros[name] = value

    def expand_path_macros(raw_name: str) -> str:
        resolved_name = raw_name
        seen: set[str] = set()

        for _ in range(16):
            if resolved_name in seen:
                break

            seen.add(resolved_name)

            updated = _PATH_MACRO_USE_RE.sub(
                lambda m: path_macros.get(m.group(1), m.group(0)),
                resolved_name,
            )

            if updated == resolved_name:
                break

            resolved_name = updated

        return resolved_name

    def visit(
        filename: str,
        ancestors: tuple[str, ...],
        *,
        caller_file: str | None = None,
        caller_line: int | None = None,
        known_flags: frozenset[str] = frozenset(),
    ) -> None:
        nonlocal offset

        if len(ancestors) >= 64:
            diagnostics.append(_make_diagnostic(
                "INCLUDE_DEPTH_LIMIT",
                "Include nesting exceeds 64 levels",
                file_path=caller_file or filename,
                line=caller_line,
            ))
            return

        if filename in ancestors:
            diagnostics.append(_make_diagnostic(
                "INCLUDE_CYCLE",
                f"Circular include: {' -> '.join((*ancestors, filename))}",
                file_path=caller_file or filename,
                line=caller_line,
            ))
            return

        tex = contents.get(filename)

        if tex is None:
            diagnostics.append(_make_diagnostic(
                "MISSING_INCLUDE",
                f"Missing include: {filename}",
                file_path=caller_file or filename,
                line=caller_line,
            ))
            return

        if filename == root:
            tex = _mask_simple_false_blocks(tex)
        elif known_flags:
            tex = _mask_known_defined_branches(
                tex, known_flags
            )
        # Comment stripping preserves original newlines.
        original_newlines = [
            m.start()
            for m in re.finditer("\n", tex)
        ]

        origin_spans: list[tuple[int, int, int]] = []

        tex = _select_known_file_exists_branches(
            tex,
            root_file=root,
            current_file=filename,
            contents=contents,
            origin_spans=origin_spans,
        )

        origin_starts = [
            start for start, _, _ in origin_spans
        ]

        def original_line_at(pos: int) -> int | None:
            index = bisect_right(origin_starts, pos) - 1

            if index < 0:
                return None

            a, b, origin = origin_spans[index]

            if not a <= pos < b:
                return None

            return bisect_right(
                original_newlines,
                origin + pos - a - 1,
            ) + 1

        def append_slice(start: int, stop: int) -> None:
            nonlocal offset

            if start >= stop:
                return

            parts.append(tex[start:stop])

            index = max(
                0,
                bisect_right(origin_starts, start) - 1,
            )

            while index < len(origin_spans):
                a, b, _ = origin_spans[index]

                if a >= stop:
                    break

                left = max(start, a)
                right = min(stop, b)

                if left < right:
                    spans.append(_TextSpan(
                        offset + left - start,
                        offset + right - start,
                        filename,
                        original_line_at(left),
                    ))

                index += 1

            offset += stop - start
        # Stop expanding includes after the root document ends.
        # Stop expanding includes only after the document has begun.
        if filename == root:
            protected_root = _protected_include_spans(tex)

            def is_protected(pos: int) -> bool:
                return any(
                    start <= pos < stop
                    for start, stop in protected_root
                )

            begin_match = next(
                (
                    m for m in re.finditer(
                        r"\\begin\s*\{document\}", tex
                    )
                    if not is_protected(m.start())
                ),
                None,
            )

            if begin_match is not None:
                for end_match in re.finditer(
                    r"\\end\s*\{document\}", tex
                ):
                    if end_match.start() <= begin_match.start():
                        continue

                    if is_protected(end_match.start()):
                        continue

                    tex = tex[:end_match.end()]
                    break
        position = 0
        protected = _protected_include_spans(tex)
        # for match in _INCLUDE_RE.finditer(tex):
        #     if any(begin <= match.start() < end for begin, end in protected):
        #         continue
        #     append(tex[position:match.start()], filename)
        #     # Read simple path macro definitions encountered before this include.
        #     for definition in _SIMPLE_PATH_MACRO_RE.finditer(
        #         tex, position, match.start()
        #     ):
        #         kind = definition.group("kind")
        #         name = definition.group("name")
        #         value = definition.group("value")

        #         if kind == "providecommand":
        #             path_macros.setdefault(name, value)
        #         else:
        #             path_macros[name] = value
        #     # raw_name = match.group(1) or match.group(2)

        #     raw_name = match.group(1)
        for match in _INCLUDE_RE.finditer(tex):
            if any(begin <= match.start() < end for begin, end in protected):
                continue

            append_slice(position, match.start())

            # Register path macros defined before this include.
            register_path_macros(tex[position:match.start()])

            raw_name = match.group(1)

            if raw_name is None:
                raw_name = match.group(2)

            # An empty \input{} has no file to include.
            # Record a diagnostic instead of crashing the parser.
            if raw_name is None or not raw_name.strip():
                diagnostics.append(_make_diagnostic(
                    "INVALID_INCLUDE_PATH",
                    f"Empty include path: {match.group(0)!r}",
                    file_path=filename,
                    line=original_line_at(match.start()),
                ))

                # Preserve the original command and continue parsing.
                append_slice(match.start(), match.end())
                position = match.end()
                continue



            # Substitute known simple path macros.
            resolved_name = expand_path_macros(raw_name)

            # A parameterized include is a template, not a concrete file path.
            if re.search(r"#[1-9]", resolved_name):
                # Preserve the original command as raw LaTeX.
                append_slice(match.start(), match.end())
                position = match.end()
                continue

            # Skip compiled bibliography in hierarchy.
            if resolved_name.strip().lower().endswith((".bib", ".bbl")):
                position = match.end()
                continue

            # Keep TV3 consistent with TV2: known non-TeX resources are not
            # hierarchy source files.  They may be absent from the bundle
            # without making the textual parse partial.
            if PurePosixPath(resolved_name.strip()).suffix.lower() in NON_TEX_EXTS:
                position = match.end()
                continue

            child = _resolve_include_path(
                root_file=root,
                current_file=filename,
                name=resolved_name,
                contents=contents,
            )

            if child not in contents and is_known_tex_support_input(resolved_name):
                position = match.end()
                continue

            if child is None:
                diagnostics.append(_make_diagnostic(
                    "INVALID_INCLUDE_PATH",
                    f"Unsafe or empty include: {match.group(1)!r}",
                    file_path=filename,
                    line=original_line_at(match.start()),
                ))
            else:
                visit(
                    child,
                    (*ancestors, filename),
                    caller_file=filename,
                    caller_line=original_line_at(match.start()),
                    known_flags=(
                        frozenset(
                            name
                            for name, line in root_flags.items()
                            if (
                                original_line_at(match.start()) or 0
                            ) > line
                        )
                        if filename == root
                        else known_flags
                    ),
                )
            position = match.end()
        register_path_macros(tex[position:])
        append_slice(position, len(tex))

    visit(root, ())
    return _ExpandedSource("".join(parts), spans, root)


def _balanced(text: str, start: int, opening: str, closing: str) -> tuple[str, int] | None:
    """Read a {nested} or [nested] argument, respecting escaped delimiters."""
    if start >= len(text) or text[start] != opening:
        return None
    depth = 1
    i = start + 1
    while i < len(text):
        if (
            text[i] == "\\"
            and i + 1 < len(text)
            and text[i + 1] in {opening, closing, "\\"}
        ):
            i += 2
            continue
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None



def _matching_end(text: str, start: int, env: str) -> int | None:
    """Return index just after matching \\end{env}, allowing same-env nesting."""
    depth = 1
    for match in _BEGIN_END_RE.finditer(text, start):
        if match.group(2) != env:
            continue
        depth += 1 if match.group(1) == "begin" else -1
        if depth == 0:
            return match.end()
    return None

def _conditional_float_opening(
    text: str,
    start: int,
    stop: int,
    env: str,
) -> tuple[int, int] | None:
    """Recognize alternate standalone openings of the same float."""

    if env not in {
        "figure", "figure*",
        "table", "table*",
        "wrapfigure", "wraptable",
    }:
        return None

    line_start = text.rfind("\n", 0, start) + 1

    if line_start == 0 or text[line_start:start].strip():
        return None

    guard_start = text.rfind(
        "\n", 0, line_start - 1
    ) + 1

    guard = text[guard_start:line_start].strip()

    if not re.fullmatch(
        r"\\(?:ifdefined[ \t]*\\[A-Za-z@]+|if[A-Za-z@]+)",
        guard,
    ):
        return None

    def arguments_only(pos: int, end: int) -> bool:

        def skip_space(p: int) -> int:
            while p < end and text[p] in " \t\r":
                p += 1
            return p

        pos = skip_space(pos)

        if pos < end and text[pos] == "[":
            arg = _balanced(text, pos, "[", "]")

            if arg is None or arg[1] > end:
                return False

            pos = skip_space(arg[1])

        required = (
            2 if env in {"wrapfigure", "wraptable"}
            else 0
        )

        for _ in range(required):
            arg = _balanced(text, pos, "{", "}")

            if arg is None or arg[1] > end:
                return False

            pos = skip_space(arg[1])

        return pos == end

        # Allow a narrow set of standalone formatting commands
    # between an environment opening and its branch marker.
    layout_line = re.compile(
        r"\\centering"
        r"|\\vspace\*?[ \t]*\{[ \t]*[+-]?[ \t]*"
        r"(?:[0-9]+(?:\.[0-9]+)?[ \t]*)?"
        r"\\baselineskip[ \t]*\}"
    )

    def find_marker(
        after_line: int,
        marker: str,
    ) -> int | None:

        # Allow at most eight layout or blank lines.
        for _ in range(9):

            if after_line >= len(text):
                return None

            begin = after_line + 1
            end = text.find("\n", begin)

            if end < 0:
                end = len(text)

            value = text[begin:end].strip()

            if value == marker:
                return end

            # Reject meaningful content or unknown commands.
            if value and not layout_line.fullmatch(value):
                return None

            after_line = end

        return None

    first_end = text.find("\n", stop)

    if first_end < 0 or not arguments_only(stop, first_end):
        return None

    else_end = find_marker(first_end, r"\else")

    if else_end is None or else_end >= len(text):
        return None

    second_end = text.find("\n", else_end + 1)

    if second_end < 0:
        return None

    second_start = else_end + 1

    while (
        second_start < second_end
        and text[second_start] in " \t"
    ):
        second_start += 1

    second = _BEGIN_END_RE.match(text, second_start)

    if (
        second is None
        or second.groups() != ("begin", env)
        or not arguments_only(second.end(), second_end)
    ):
        return None

    fi_end = find_marker(second_end, r"\fi")

    if fi_end is None:
        return None

    return guard_start, fi_end

def _matching_end_with_macros(
    text: str,
    position: int,
    env: str,
    macros: Mapping[str, tuple[str, str]],
    call_pattern: re.Pattern[str] | None,
) -> int | None:

    depth = 1

    while True:
        normal = _BEGIN_END_RE.search(text, position)

        macro = (
            call_pattern.search(text, position)
            if call_pattern is not None else None
        )

        if normal is None and macro is None:
            return None

        if macro is not None and (
            normal is None or macro.start() < normal.start()
        ):
            action, current_env = macros[macro.group(1)]
            end = macro.end()
        else:
            action, current_env = normal.groups()
            end = normal.end()

        if current_env == env:
            depth += 1 if action == "begin" else -1

            if depth == 0:
                return end

        position = end


def _has_direct_item_before_explicit_list(
    text: str,
    start: int,
    stop: int,
    list_envs: set[str],
) -> bool:
    """Check whether an implicit theorem-like body uses a direct item command."""
    body = text[start:stop]
    item = re.search(r"\\item(?![A-Za-z@])", body)
    if item is None:
        return False
    for marker in _BEGIN_END_RE.finditer(body):
        if (
            marker.group(1) == "begin"
            and marker.group(2).strip() in list_envs
        ):
            return item.start() < marker.start()
    return True

def _is_reference_heading(text: str) -> bool:
    # Match a heading's human-readable title, not its formatting syntax.
    flattened = re.sub(r"\\[A-Za-z]+\*?", "", text)
    flattened = re.sub(r"[{}~]", " ", flattened)
    flattened = " ".join(flattened.lower().split()).strip(" :.")
    return flattened in _REFERENCE_TITLES


def _freeze(node: _MutableNode) -> TreeNode:
    return TreeNode(
        kind=NodeKind(node.kind),
        content=node.content,
        children=tuple(_freeze(child) for child in node.children),
        level=node.level,
        source_file=node.source_file,
        attributes=node.attributes,
    )

def _find_next_structural_macro_call(
    text: str,
    position: int,
    macros: Mapping[str, tuple[str, str]],
) -> re.Match[str] | None:
    if not macros:
        return None

    names = sorted(macros.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\\(" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z@])"
    )
    return pattern.search(text, position)

def _safe_early_math_alias(
    text: str,
    start: int,
    stop: int,
    document_start: int,
    name: str,
) -> bool:
    """Conservative guard for a new, standalone document-prefix alias."""

    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", stop)

    if line_end < 0:
        line_end = len(text)

    # Require a standalone macro definition.
    if (
        text[line_start:start].strip()
        or text[stop:line_end].strip()
    ):
        return False

    command = r"\\" + re.escape(name) + r"(?![A-Za-z@])"

    # Avoid applying aliases before their definitions.
    if re.search(command, text[:start]):
        return False

    # Reject macros with detectable later redefinitions.
    binding = (
        r"\\(?:def|gdef|edef|xdef|let|futurelet|"
        r"newcommand|renewcommand|providecommand|"
        r"DeclareRobustCommand|NewDocumentCommand|"
        r"RenewDocumentCommand|ProvideDocumentCommand|"
        r"DeclareDocumentCommand)"
        r"\*?(?![A-Za-z@])\s*(?:\{\s*)?"
        + command
    )

    if re.search(binding, text[stop:]):
        return False

    depth = 0

    tokens = re.compile(
        r"\\[{}\\]|[{}]|"
        r"\\(?:if[A-Za-z@]*|else|fi|begingroup|"
        r"endgroup|bgroup|egroup|begin|end)"
        r"(?![A-Za-z@])"
    )

    for token in tokens.finditer(
        text, document_start, start
    ):
        value = token.group()

        if value == "{":
            depth += 1

        elif value == "}":
            depth -= 1

            if depth < 0:
                return False

        elif depth == 0 and len(value) > 2:
            # Do not interpret explicit scopes or conditionals.
            return False

    return depth == 0


def _collect_simple_structural_macros(
    text: str,
) -> tuple[dict[str, tuple[str, str]], str]:

    macros: dict[str, tuple[str, str]] = {}
    chars = list(text)
    consumed = 0
    document_start = re.search(r"\\begin\s*\{document\}", text)
    preamble_end = document_start.start() if document_start else 0
    protected = _protected_include_spans(text[:preamble_end])
    body_start = document_start.end() if document_start else 0

    body_boundary = re.compile(
        r"\\(?:part|chapter|section|subsection|"
        r"subsubsection|paragraph|subparagraph)"
        r"\*?(?![A-Za-z@])"
        r"|\\(?:begin\s*\{abstract\}|end\s*\{document\})"
    ).search(text, body_start) if document_start else None

    body_limit = (
        body_boundary.start()
        if body_boundary
        else len(text)
    )

    for match in _COMMAND_DEF_RE.finditer(text):
        if match.start() < consumed:
            continue

        cursor = match.end()

        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        if match.group() == r"\def":

            in_preamble = match.start() < preamble_end

            in_early_body = bool(
                document_start
                and body_start <= match.start() < body_limit
            )

            if not in_preamble and not in_early_body:
                continue

            if any(
                a <= match.start() < b
                for a, b in protected
            ):
                continue

            if match.start() and text[match.start() - 1] == "\\":
                continue

            name_match = re.compile(
                r"\\([A-Za-z@]+)\s*"
            ).match(text, cursor)

            if name_match is None:
                continue

            body_arg = _balanced(
                text, name_match.end(), "{", "}"
            )

            if body_arg is None:
                continue

            limit = (
                preamble_end
                if in_preamble
                else body_limit
            )

            if body_arg[1] > limit:
                continue

            event = _simple_structural_body(body_arg[0])
            name = name_match.group(1)

            if in_preamble:
                # Preserve existing preamble behavior.
                if event is None or event[1] not in (
                    _LIST_ENVS | {"eqnarray", "eqnarray*"}
                ):
                    continue

            else:
                # Only accept simple early-body math aliases.
                if event is None or event[1] not in {
                    "equation",
                    "equation*",
                    "eqnarray",
                    "eqnarray*",
                }:
                    continue

                if name in macros or not _safe_early_math_alias(
                    text,
                    match.start(),
                    body_arg[1],
                    body_start,
                    name,
                ):
                    continue

            macros[name] = event
            consumed = body_arg[1]

            # Preserve source offsets and original line mapping.
            for i in range(match.start(), consumed):
                if chars[i] != "\n":
                    chars[i] = " "

            continue

        name_arg = _balanced(text, cursor, "{", "}")
        if name_arg is None:
            continue

        name_match = re.fullmatch(
            r"\\([A-Za-z@]+)", name_arg[0].strip()
        )
        if name_match is None:
            continue

        name = name_match.group(1)
        cursor = name_arg[1]

        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        arg_count = _balanced(text, cursor, "[", "]")
        arity = 0

        if arg_count is not None:
            value = arg_count[0].strip()
            arity = int(value) if value.isdigit() else -1
            cursor = arg_count[1]

        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        body_arg = _balanced(text, cursor, "{", "}")
        if body_arg is None:
            continue

        consumed = body_arg[1]

        # Mask macro definition, preserving offsets and newlines.
        for i in range(match.start(), consumed):
            if chars[i] != "\n":
                chars[i] = " "

        # Only zero-argument macros can be expanded safely here.
        event = (
            _simple_structural_body(body_arg[0])
            if arity == 0 else None
        )

        if event is not None:
            macros[name] = event

    return macros, "".join(chars)


def _structural_call_pattern(
    macros: Mapping[str, object],
) -> re.Pattern[str] | None:

    if not macros:
        return None

    names = sorted(macros, key=len, reverse=True)

    return re.compile(
        r"\\("
        + "|".join(re.escape(name) for name in names)
        + r")(?![A-Za-z@])"
    )

def _next_structural_token(
    text: str,
    position: int,
    macros: Mapping[str, tuple[str, str]],
    call_pattern: re.Pattern[str] | None,
    parameterized_macros: Mapping[str, tuple[str, int, int]] | None = None,
    parameterized_call_pattern: re.Pattern[str] | None = None,
) -> tuple[str, int, int] | None:

    normal = _SPECIAL_RE.search(text, position)

    macro = (
        call_pattern.search(text, position)
        if call_pattern is not None else None
    )

    parameterized = (
        parameterized_call_pattern.search(text, position)
        if parameterized_call_pattern is not None else None
    )

    if parameterized is not None and (
        normal is None or parameterized.start() < normal.start()
    ) and (
        macro is None or parameterized.start() < macro.start()
    ):
        environment = parameterized_macros[parameterized.group(1)][0]
        return (
            rf"\begin{{{environment}}}",
            parameterized.start(),
            parameterized.end(),
        )

    if macro is not None and (
        normal is None or macro.start() < normal.start()
    ):
        action, env = macros[macro.group(1)]

        return (
            rf"\{action}{{{env}}}",
            macro.start(),
            macro.end(),
        )

    if normal is None:
        return None

    return normal.group(0), normal.start(), normal.end()

def _find_unescaped_math_marker(
    text: str,
    marker: str,
    start: int,
    stop: int,
) -> int:
    """Find a delimiter not escaped by an odd number of backslashes."""
    cursor = start

    while True:
        candidate = text.find(marker, cursor, stop)

        if candidate < 0:
            return -1

        j = candidate - 1

        while j >= 0 and text[j] == "\\":
            j -= 1

        if (candidate - j - 1) % 2 == 0:
            return candidate

        cursor = candidate + 1


def _math_recovery_boundary(
    text: str,
    start: int,
    stop: int,
    list_envs: set[str],
    *,
    display_math: bool = False,
) -> int | None:
    """Find a likely outer structural boundary, not an inner math environment."""
    pattern = re.compile(
        r"^[ \t]*\\(?:(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
        r"\*?(?=\s*[\[{])|item(?![A-Za-z@])|(?:begin|end)\s*\{([^{}]+)\})",
        re.MULTILINE,
    )

    outer_envs = (
        list_envs
        | _MATH_ENVS
        | _FIGURE_ENVS
        | _RAW_TEXT_ENVS
        | {"document"}
    )
    # A matched tabular environment can appear
    # inside a display math block.
    paired_tabular_ends: set[int] = set()

    if display_math:
        open_tabular: list[int] = []

        tabular_pattern = re.compile(
            r"\\(begin|end)\s*\{tabular\}"
        )

        for marker in tabular_pattern.finditer(
            text, start, stop
        ):
            if marker.group(1) == "begin":
                open_tabular.append(marker.end())

            elif open_tabular:
                paired_tabular_ends.add(
                    open_tabular.pop()
                )
                paired_tabular_ends.add(marker.end())

    depth = 0
    cursor = start

    for match in pattern.finditer(text, start, stop):
        env = match.group(1)

        if (
            env == "tabular"
            and match.end() in paired_tabular_ends
        ):
            continue

        if env is not None and env not in outer_envs:
            continue

        while cursor < match.start():
            ch = text[cursor]

            if ch == "\\" and cursor + 1 < match.start():
                if text[cursor + 1] in "{}\\":
                    cursor += 2
                    continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)

            cursor += 1

        if depth == 0:
            return match.start()

    return None

def _parse_expanded(expanded: _ExpandedSource, diagnostics: list[Diagnostic]) -> TreeNode:
    """Build mutable hierarchy, then freeze to the team's TreeNode contract."""
    list_envs = _collect_list_environments(expanded.text)
    tex = expanded.text

    structural_macros, parse_text = (
        _collect_simple_structural_macros(expanded.text)
    )

    # A&A-style sources commonly use ``\bi``/``\ei`` as shorthand for an
    # itemize environment supplied by the document class.  The class file is
    # often external to the arXiv source, so there is no local definition for
    # _collect_simple_structural_macros to inspect.  Only enable the aliases
    # when the pair is actually present in the source, avoiding a global
    # reinterpretation of an unrelated one-off macro.
    for begin_name, end_name in (("bi", "ei"), ("bitem", "eitem")):
        if (
            re.search(rf"\\{begin_name}(?![A-Za-z@])", parse_text)
            and re.search(rf"\\{end_name}(?![A-Za-z@])", parse_text)
        ):
            structural_macros.setdefault(begin_name, ("begin", "itemize"))
            structural_macros.setdefault(end_name, ("end", "itemize"))

    parameterized_list_macros = _collect_parameterized_list_macros(
        expanded.text,
        list_envs,
    )
    parameterized_list_openers = _collect_parameterized_list_openers(
        expanded.text,
        list_envs,
    )
    for name, (environment, _) in parameterized_list_openers.items():
        structural_macros.setdefault(name, ("begin", environment))

    # Detect custom list environments from the preamble.
    custom_lists, parse_text = (
        _collect_custom_list_environments(
            parse_text, list_envs
        )
    )

    list_envs.update(custom_lists)

    macro_call_pattern = _structural_call_pattern(
        structural_macros
    )
    parameterized_call_pattern = _structural_call_pattern(
        parameterized_list_macros
    )
    root = _MutableNode("document", source_file=expanded.fallback)
    headings: list[_MutableNode] = []
    lists: list[_ListFrame] = []
    scoped_headings: list[tuple[str, _MutableNode]] = []
    pending: list[str] = []
    pending_start: int | None = None
    appendix = False
    suppressed_level: int | None = None
    parameterized_list_ends: list[tuple[int, int, str]] = []

    def parent() -> _MutableNode:
        if lists:
            frame = lists[-1]

            if frame.local_headings:
                return frame.local_headings[-1]

            return frame.current_item or frame.node

        return headings[-1] if headings else root

    def append_text(fragment: str, start: int) -> None:
        nonlocal pending_start
        if not fragment:
            return
        if not pending:
            pending_start = start
        pending.append(fragment)

    def flush() -> None:
        nonlocal pending_start
        if not pending:
            return
        raw = "".join(pending)
        source_file = expanded.source_file_at(pending_start or 0)
        for part in _PARAGRAPH_BREAK_RE.split(raw):
            # Keep intra-paragraph raw LaTeX. TV4 performs semantic cleanup.
            if part.strip():
                parent().children.append(_MutableNode(
                    "paragraph", content=part.strip(), source_file=source_file,
                ))
        pending.clear()
        pending_start = None

    def add_heading(title: str, level: int, position: int,
                    *, command: str, starred: bool = False,
                    extra_attrs: Mapping[str, str] | None = None) -> _MutableNode:
        flush()
        while headings and (headings[-1].level or 0) >= level:
            headings.pop()
        attrs = {"command": command, "unnumbered": str(starred).lower()}
        if appendix:
            attrs["appendix"] = "true"
        if extra_attrs:
            attrs.update(extra_attrs)
        node = _MutableNode(
            "heading", content=title, level=level,
            source_file=expanded.source_file_at(position), attributes=attrs,
        )
        (headings[-1] if headings else root).children.append(node)
        headings.append(node)
        return node

    # Skip the LaTeX preamble (\\documentclass, macro declarations, packages).
    # If a document delimiter is missing, parse the provided text as a fragment.
    document_start = re.search(r"\\begin\s*\{document\}", tex)
    position = document_start.end() if document_start else 0

    while position < len(parse_text):
        event: tuple[str, int, int] | None = None
        synthetic_list_end: tuple[str, int, int] | None = None

        if parameterized_list_ends:
            end_start, end_after, environment = parameterized_list_ends[-1]
            next_event = _next_structural_token(
                parse_text,
                position,
                structural_macros,
                macro_call_pattern,
                parameterized_list_macros,
                parameterized_call_pattern,
            )
            if position >= end_start or (
                next_event is None or next_event[1] >= end_start
            ):
                parameterized_list_ends.pop()
                synthetic_list_end = (
                    rf"\end{{{environment}}}",
                    end_start,
                    end_after,
                )
            else:
                event = next_event
        else:
            event = _next_structural_token(
                parse_text,
                position,
                structural_macros,
                macro_call_pattern,
                parameterized_list_macros,
                parameterized_call_pattern,
            )

        if synthetic_list_end is not None:
            event = synthetic_list_end

        if event is None:
            if suppressed_level is None:
                append_text(parse_text[position:], position)
            break

        token, token_start, token_end = event

        # Ignore escaped commands.
        if token.startswith("\\"):
            backslash_count = 0
            j = token_start - 1

            while j >= 0 and parse_text[j] == "\\":
                backslash_count += 1
                j -= 1

            if backslash_count % 2 == 1:
                if suppressed_level is None:
                    append_text(
                        parse_text[position:token_end],
                        position,
                    )

                position = token_end
                continue

        if suppressed_level is None:
            append_text(
                parse_text[position:token_start],
                position,
            )

        position = token_end

        if token in {"$", "$$"}:
            j = token_start - 1

            while j >= 0 and parse_text[j] == "\\":
                j -= 1

            if (token_start - j - 1) % 2 == 1:
                if suppressed_level is None:
                    append_text("$", token_start)

                position = token_start + 1
                continue

        if token == r"\def":
            signature = re.compile(
                r"[ \t]*\\([A-Za-z@]+)[ \t]*(?:#[1-9])+\s*"
            ).match(parse_text, position)

            if signature is not None:
                body = _balanced(
                    parse_text,
                    signature.end(),
                    "{",
                    "}",
                )

                if body is not None:
                    is_math_alias = bool(
                        re.search(
                            r"\\begin\s*\{(?:equation\*?|eqnarray\*?)\}",
                            body[0],
                        )
                    )
                    if suppressed_level is None and is_math_alias:
                        diagnostics.append(_make_diagnostic(
                            "UNSUPPORTED_STRUCTURAL_MACRO",
                            (
                                f"Parameterized math macro "
                                f"\\{signature.group(1)} "
                                "not expanded; definition skipped"
                            ),
                            file_path=expanded.source_file_at(
                                token_start
                            ),
                            line=expanded.source_line_at(
                                token_start
                            ),
                        ))

                    position = body[1]
                    continue

        if token == r"\comment":
            cursor = position
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            argument = _balanced(tex, cursor, "{", "}")
            if argument is not None:
                position = argument[1]
                continue

        if token == r"\def":
            # Macro definitions are configuration, not document content.
            # Generic definitions can contain literal ``\item`` tokens (for
            # example a shorthand macro or a class hook), which must not be
            # diagnosed as list items outside a list.  Structural aliases are
            # already collected and masked before this scan; this branch
            # handles the remaining definitions safely.
            cursor = position
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            if cursor < len(tex) and tex[cursor] == "\\":
                cursor += 1
                while cursor < len(tex) and (
                    tex[cursor].isalpha() or tex[cursor] == "@"
                ):
                    cursor += 1
                while cursor < len(tex) and tex[cursor] != "{":
                    cursor += 1
                body = _balanced(tex, cursor, "{", "}")
                if body is not None:
                    position = body[1]
                    continue
        
        if _BEGIN_END_RE.fullmatch(token):
            env_match = _BEGIN_END_RE.fullmatch(token)
            if not env_match:
                if suppressed_level is None:
                    append_text(
                        tex[token_start:token_end],
                        token_start,
                    )
                continue
            action, env = env_match.groups()
            raw_token = parse_text[token_start:token_end]
            parameterized_list_macro = parameterized_list_macros.get(
                raw_token[1:]
            )
            parameterized_opener = parameterized_list_openers.get(
                raw_token[1:]
            )
            if action == "begin":
                if env == "document":
                    flush()
                    continue
                if env in _SKIP_ENVS:
                    if suppressed_level is None:
                        flush()

                    end = _matching_end_with_macros(
                        parse_text,
                        position,
                        env,
                        structural_macros,
                        macro_call_pattern,
                    )

                    if end is None:
                        diagnostics.append(_make_diagnostic(
                            "UNCLOSED_ENV",
                            f"Unclosed {env}",
                            file_path=expanded.source_file_at(token_start),
                            line=expanded.source_line_at(token_start),
                        ))

                        position = len(tex)

                        # Recover from an unbalanced comment environment.
                        if env == "comment":
                            first_end = re.compile(
                                r"^[ \t]*\\end\s*\{comment\}[ \t]*(?=\r?$)",
                                re.MULTILINE,
                            ).search(parse_text, token_end)

                            if first_end is not None:
                                position = first_end.end()

                    else:
                        position = end

                    continue

                if env in _MATH_ENVS or env in _FIGURE_ENVS or env in _RAW_TEXT_ENVS:
                    end = _matching_end_with_macros(
                        parse_text,
                        position,
                        env,
                        structural_macros,
                        macro_call_pattern,
                    )

                    content_start = token_start

                    if end is None:
                        message = f"Unclosed {env}"

                        # Recover only simple alternate float openings.
                        if env in _FIGURE_ENVS:
                            alternative = _conditional_float_opening(
                                parse_text,
                                token_start,
                                token_end,
                                env,
                            )

                            if alternative is not None:
                                guard_start, after_fi = alternative

                                prefix = parse_text[
                                    guard_start:token_start
                                ]

                                recovered = _matching_end_with_macros(
                                    parse_text,
                                    after_fi,
                                    env,
                                    structural_macros,
                                    macro_call_pattern,
                                )

                                if recovered is not None and (
                                    suppressed_level is not None
                                    or (
                                        pending
                                        and pending[-1].endswith(prefix)
                                    )
                                ):
                                    # Keep the entire conditional block
                                    # in the raw figure content.
                                    if suppressed_level is None:
                                        pending[-1] = pending[-1][
                                            :-len(prefix)
                                        ]

                                    content_start = guard_start
                                    end = recovered

                                    message += (
                                        "; recovered alternate conditional "
                                        "openings (branch unresolved)"
                                    )

                        if end is None and env in _FIGURE_ENVS:
                            end = _recover_declared_float_end(
                                parse_text,
                                tex,
                                token_end,
                                env,
                            )

                            if end is not None:
                                message += (
                                    "; recovered declared closing macro"
                                )

                        diagnostics.append(_make_diagnostic(
                            "UNCLOSED_ENV",
                            message,
                            file_path=expanded.source_file_at(token_start),
                            line=expanded.source_line_at(token_start),
                        ))

                        if end is None:
                            end = len(tex)

                    if suppressed_level is None:
                        flush()

                        kind = (
                            "block_formula"
                            if env in _MATH_ENVS
                            else (
                                "figure"
                                if env in _FIGURE_ENVS
                                else "paragraph"
                            )
                        )

                        parent().children.append(_MutableNode(
                            kind,
                            content=tex[content_start:end],
                            source_file=expanded.source_file_at(content_start),
                            attributes={"environment": env},
                        ))

                    position = end
                    continue
                
                if env in list_envs:
                    if suppressed_level is None:
                        flush()
                        node = _MutableNode(
                            "list",
                            source_file=expanded.source_file_at(token_start),
                            attributes={"environment": env},
                        )
                        parent().children.append(node)
                        lists.append(_ListFrame(
                            env,
                            node,
                            opening_offset=token_start,
                        ))

                    # A real \begin{list} has two mandatory arguments.
                    # A synthetic token from a macro call does not.
                    is_macro_call = (
                        parse_text[token_start:token_end] != token
                    )

                    if parameterized_list_macro is not None:
                        _, argument_count, body_argument = parameterized_list_macro
                        cursor = token_end
                        arguments: list[tuple[int, tuple[str, int]]] = []
                        for _ in range(argument_count):
                            while cursor < len(tex) and tex[cursor].isspace():
                                cursor += 1
                            argument = _balanced(tex, cursor, "{", "}")
                            if argument is None:
                                break
                            arguments.append((cursor, argument))
                            cursor = argument[1]

                        if len(arguments) != argument_count:
                            diagnostics.append(_make_diagnostic(
                                "UNSUPPORTED_STRUCTURAL_MACRO",
                                (
                                    f"List macro \\{raw_token[1:]} "
                                    "has unbalanced arguments"
                                ),
                                file_path=expanded.source_file_at(token_start),
                                line=expanded.source_line_at(token_start),
                            ))
                        else:
                            body_start, body = arguments[body_argument - 1]
                            parameterized_list_ends.append(
                                (body[1] - 1, body[1], env)
                            )
                            position = body_start + 1
                            continue

                    if parameterized_opener is not None:
                        cursor = token_end
                        arguments_valid = True

                        for _ in range(parameterized_opener[1]):
                            while cursor < len(tex) and tex[cursor].isspace():
                                cursor += 1

                            argument = _balanced(tex, cursor, "{", "}")
                            if argument is None:
                                arguments_valid = False
                                break

                            cursor = argument[1]

                        if arguments_valid:
                            position = cursor
                        else:
                            diagnostics.append(_make_diagnostic(
                                "UNSUPPORTED_STRUCTURAL_MACRO",
                                (
                                    f"List opener \\{raw_token[1:]} "
                                    "has unbalanced arguments"
                                ),
                                file_path=expanded.source_file_at(token_start),
                                line=expanded.source_line_at(token_start),
                            ))

                        continue

                    # Inline enumeration may have an optional label format:
                    # \begin{inparaenum}[(1)]
                    # The label format is configuration, not paragraph content.
                    if (
                        env in {"inparaenum", "asparaenum"}
                        or custom_lists.get(env, False)
                    ) and not is_macro_call:
                        cursor = position

                        while cursor < len(tex) and tex[cursor].isspace():
                            cursor += 1

                        optional = _balanced(tex, cursor, "[", "]")

                        if optional is not None:
                            position = optional[1]



                    if env == "list" and not is_macro_call:
                        cursor = position
                        arguments_valid = True

                        for _ in range(2):
                            while cursor < len(tex) and tex[cursor].isspace():
                                cursor += 1

                            argument = _balanced(tex, cursor, "{", "}")

                            if argument is None:
                                arguments_valid = False
                                break

                            cursor = argument[1]

                        if arguments_valid:
                            position = cursor

                    continue

                if env in _IMPLICIT_LIST_ENVS:
                    matching_end = _matching_end_with_macros(
                        parse_text,
                        position,
                        env,
                        structural_macros,
                        macro_call_pattern,
                    )
                    if (
                        matching_end is not None
                        and _has_direct_item_before_explicit_list(
                            parse_text,
                            position,
                            matching_end,
                            list_envs,
                        )
                    ):
                        if suppressed_level is None:
                            flush()
                            node = _MutableNode(
                                "list",
                                source_file=expanded.source_file_at(token_start),
                                attributes={"environment": env},
                            )
                            parent().children.append(node)
                            lists.append(_ListFrame(
                                env,
                                node,
                                opening_offset=token_start,
                            ))
                        continue

                if env == "appendices":
                    appendix = True
                    continue
                if env == "abstract" or env.lower() in _ACK_ENVS:
                    if suppressed_level is None:
                        heading_title = "Abstract" if env == "abstract" else "Acknowledgements"
                        node = add_heading(heading_title, 3, token_start, command=f"environment:{env}", starred=True)
                        scoped_headings.append((env, node))
                    continue
                if suppressed_level is None:
                    append_text(token, token_start)
                continue

            # \\end{...}
            if env == "document":
                if suppressed_level is None:
                    flush()
                break
            if env in list_envs or (
                lists and lists[-1].environment == env
            ):
                if suppressed_level is None:
                    flush()
                    if lists and lists[-1].environment == env:
                        lists.pop()
                    else:
                        diagnostics.append(_make_diagnostic("LIST_MISMATCH", f"Unmatched end{{{env}}}", file_path=expanded.source_file_at(token_start), line=expanded.source_line_at(token_start),))
                continue
            if env == "appendices":
                appendix = False
                continue
            if scoped_headings and env == scoped_headings[-1][0]:
                if suppressed_level is None:
                    flush()
                    node = scoped_headings.pop()[1]
                    if node in headings:
                        while headings and headings[-1] is not node:
                            headings.pop()
                        headings.pop()
                continue
            if suppressed_level is None:
                append_text(
                    tex[token_start:token_end],
                    token_start,
                )
            continue
        if token in {
            r"\let",
            r"\titleformat",
            r"\titleformat*",
            r"\titlespacing",
            r"\titlespacing*",
            r"\RenewDocumentCommand",
        }:
            end = _heading_config_end(
                tex, position, token
            )

            if end is not None:
                if suppressed_level is None:
                    append_text(
                        tex[token_start:end],
                        token_start,
                    )

                position = end
                continue

        # Structural headings also terminate skipped References sections.
        head_match = re.fullmatch(
            r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)(\*)?",
            token,
        )
        if head_match:
            command, star = head_match.groups()
            level = _HEADING_LEVELS[command]
            cursor = position
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            optional = _balanced(tex, cursor, "[", "]")
            if optional:
                cursor = optional[1]
                while cursor < len(tex) and tex[cursor].isspace():
                    cursor += 1
            title_arg = _balanced(tex, cursor, "{", "}")
            if title_arg is None:
                if suppressed_level is None:
                    append_text(token, token_start)
                diagnostics.append(_make_diagnostic("INVALID_HEADING", f"Missing title for {command}", file_path=expanded.source_file_at(token_start), line=expanded.source_line_at(token_start),))
                continue
            title, position = title_arg
            if suppressed_level is not None:
                if level > suppressed_level:
                    continue
                suppressed_level = None
            if _is_reference_heading(title):
                flush()
                suppressed_level = level
                continue

            
            # Preserve paragraph/subparagraph headings inside list items.
            if (
                lists
                and command in {"paragraph", "subparagraph"}
                and lists[-1].current_item is not None
            ):
                flush()

                frame = lists[-1]

                # Close local headings at the same or deeper level.
                while (
                    frame.local_headings
                    and (frame.local_headings[-1].level or 0) >= level
                ):
                    frame.local_headings.pop()

                attrs = {
                    "command": command,
                    "unnumbered": str(bool(star)).lower(),
                }

                if appendix:
                    attrs["appendix"] = "true"

                local_heading = _MutableNode(
                    "heading",
                    content=title,
                    level=level,
                    source_file=expanded.source_file_at(token_start),
                    attributes=attrs,
                )

                # Attach the heading to the current item or
                # to its nearest active parent heading.
                local_parent = (
                    frame.local_headings[-1]
                    if frame.local_headings
                    else frame.current_item
                )

                local_parent.children.append(local_heading)
                frame.local_headings.append(local_heading)

                continue

            
            if lists:
                flush()
                diagnostics.append(
                    _make_diagnostic(
                        "HEADING_IN_LIST",
                        "Heading encountered before list ended",
                        file_path=expanded.source_file_at(token_start),
                        line=expanded.source_line_at(token_start),
                    )
                )
                lists.clear()

            add_heading(title, level, token_start, command=command, starred=bool(star))
            continue

        if suppressed_level is not None:
            continue

        if token.startswith(r"\item"):
            flush()
            if not lists:
                diagnostics.append(_make_diagnostic("ITEM_OUTSIDE_LIST", r"\item outside a list", file_path=expanded.source_file_at(token_start), line=expanded.source_line_at(token_start),))
                append_text(token, token_start)
                continue
            lists[-1].local_headings.clear()
            cursor = position
            while cursor < len(tex) and tex[cursor] in " \t":
                cursor += 1
            label = _balanced(tex, cursor, "[", "]")
            attrs: dict[str, str] = {}
            if label is not None:
                attrs["label"] = label[0]
                position = label[1]
            item = _MutableNode("list_item", source_file=expanded.source_file_at(token_start), attributes=attrs)
            lists[-1].node.children.append(item)
            lists[-1].current_item = item
            continue

        if token == r"\appendix":
            flush()
            appendix = True
            continue

        if token.startswith(r"\bibliography"):
            cursor = position
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            bib_arg = _balanced(tex, cursor, "{", "}")
            if bib_arg:
                position = bib_arg[1]
            continue
        if token.startswith(r"\printbibliography"):
            # May have optional [options], which is metadata for TV5, not prose.
            cursor = position
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1
            opt = _balanced(tex, cursor, "[", "]")
            if opt:
                position = opt[1]
            continue

        if token in {r"\[", "$$"}:
            closer = r"\]" if token == r"\[" else "$$"

            end = _find_unescaped_math_marker(
                parse_text, closer, position, len(tex)
            )

            boundary = (
                _math_recovery_boundary(
                    parse_text,
                    position,
                    end,
                    list_envs,
                    display_math=True,
                )
                if end >= 0
                else None
            )

            if boundary is not None:
                append_text(
                    tex[token_start:boundary],
                    token_start,
                )

                diagnostics.append(_make_diagnostic(
                    "UNCLOSED_MATH",
                    "Display math crossed a structural boundary; parser resynchronized",
                    file_path=expanded.source_file_at(token_start),
                    line=expanded.source_line_at(token_start),
                ))

                position = boundary

            elif end < 0:
                append_text(token, token_start)

                diagnostics.append(_make_diagnostic(
                    "UNCLOSED_MATH",
                    f"Unclosed {token}",
                    file_path=expanded.source_file_at(token_start),
                    line=expanded.source_line_at(token_start),
                ))

            else:
                flush()

                end += len(closer)

                parent().children.append(_MutableNode(
                    "block_formula",
                    content=tex[token_start:end],
                    source_file=expanded.source_file_at(token_start),
                    attributes={"environment": "display_math"},
                ))

                position = end

            continue

        
        if token in {r"\(", "$"}:
            closer = r"\)" if token == r"\(" else "$"
            wrong_closer = "$" if token == r"\(" else r"\)"

            correct_end = _find_unescaped_math_marker(
                parse_text, closer, position, len(tex)
            )

            line_end = tex.find("\n", position)

            if line_end < 0:
                line_end = len(tex)

            wrong_end = _find_unescaped_math_marker(
                parse_text,
                wrong_closer,
                position,
                line_end,
            )

            scan_limit = (
                correct_end
                if correct_end >= 0
                else len(tex)
            )

            mismatch = 0 <= wrong_end < scan_limit

            if mismatch:
                scan_limit = wrong_end

            boundary = _math_recovery_boundary(
                parse_text,
                position,
                scan_limit,
                list_envs,
            )

            if boundary is not None:
                append_text(
                    tex[token_start:boundary],
                    token_start,
                )

                diagnostics.append(_make_diagnostic(
                    "UNCLOSED_MATH",
                    "Inline math crossed a structural boundary; parser resynchronized",
                    file_path=expanded.source_file_at(token_start),
                    line=expanded.source_line_at(token_start),
                ))

                position = boundary

            elif mismatch:
                recovery_end = wrong_end + len(wrong_closer)

                append_text(
                    tex[token_start:recovery_end],
                    token_start,
                )

                diagnostics.append(_make_diagnostic(
                    "UNCLOSED_MATH",
                    f"Mismatched inline math: expected {closer!r} but found {wrong_closer!r}",
                    file_path=expanded.source_file_at(token_start),
                    line=expanded.source_line_at(token_start),
                ))

                position = recovery_end

            elif correct_end >= 0:
                end = correct_end + len(closer)

                append_text(
                    tex[token_start:end],
                    token_start,
                )

                position = end

            else:
                append_text(token, token_start)

                diagnostics.append(_make_diagnostic(
                    "UNCLOSED_MATH",
                    f"Unclosed inline math: expected {closer!r}",
                    file_path=expanded.source_file_at(token_start),
                    line=expanded.source_line_at(token_start),
                ))

            continue


        if token.startswith(r"\verb"):
            cursor = position
            if cursor < len(tex) and tex[cursor] not in " \t\n":
                closer = tex.find(tex[cursor], cursor + 1)
                if closer >= 0:
                    append_text(tex[token_start:closer + 1], token_start)
                    position = closer + 1
                    continue
            append_text(token, token_start)
            continue

        append_text(token, token_start)

    if suppressed_level is None:
        flush()

    if lists:
        opening = lists[0].opening_offset

        diagnostics.append(_make_diagnostic(
            "UNCLOSED_LIST",
            f"{len(lists)} list environment(s) not closed",
            file_path=(
                expanded.source_file_at(opening)
                if opening is not None else None
            ),
            line=(
                expanded.source_line_at(opening)
                if opening is not None else None
            ),
        ))

    return _freeze(root)


def parse_structure(source: SourceBundle) -> RawTree:
    """Convert one gathered arXiv version to a raw hierarchical tree.

    - document / heading / paragraph / list / list_item / block_formula / figure
    - original LaTeX is kept in paragraph and structural element contents
    - diagnostics preserve partial errors rather than silently dropping versions
    """
    diagnostics = list(source.diagnostics)
    source_status = _status_name(source.status)
    if source_status == "empty_no_tex" or not source.files:
        if source_status == "empty_no_tex":
            status = ProcessingStatus.EMPTY_NO_TEX
        elif source_status == "ok":
            status = ProcessingStatus.ROOT_NOT_FOUND
        else:
            # Preserve a gathering error/partial status when TV2 has no
            # readable files to parse.  Mapping every empty bundle to
            # root_not_found would hide SOURCE_DIR_NOT_FOUND and similar
            # diagnostics from downstream validation and statistics.
            status = source.status
        return RawTree(
            arxiv_id=source.arxiv_id, version=source.version,
            root=None,
            status=status, diagnostics=tuple(diagnostics),
        )

    expanded = _expand_bundle(source, diagnostics)
    if not expanded.text and any(d.code == "ROOT_NOT_FOUND" for d in diagnostics):
        return RawTree(
            arxiv_id=source.arxiv_id, version=source.version,
            root=None,
            status=ProcessingStatus.ROOT_NOT_FOUND, diagnostics=tuple(diagnostics),
        )

    root = _parse_expanded(expanded, diagnostics)
    # Input problems are carried forward, but unexpected code exceptions are
    # intentionally not swallowed: hiding programming errors breaks testing.
    is_partial = source_status not in {"ok"} or any(
        d.code in {
            # Include expansion errors
            "MISSING_INCLUDE",
            "INCLUDE_CYCLE",
            "INCLUDE_DEPTH_LIMIT",
            "INVALID_INCLUDE_PATH",

            # Math and environment errors
            "UNCLOSED_ENV",
            "UNCLOSED_MATH",
            "UNSUPPORTED_STRUCTURAL_MACRO",

            # List structure errors
            "ITEM_OUTSIDE_LIST",
            "LIST_MISMATCH",
            "UNCLOSED_LIST",
            "HEADING_IN_LIST",

            # Heading errors
            "INVALID_HEADING",
        }
        for d in diagnostics
    )
    return RawTree(
        arxiv_id=source.arxiv_id, version=source.version, root=root,
        status=ProcessingStatus.PARTIAL_PARSE if is_partial else ProcessingStatus.OK,
        diagnostics=tuple(diagnostics),
    )


