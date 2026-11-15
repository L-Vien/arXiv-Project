"""Structural LaTeX parser for IntroDS Milestone 2 (TV3).

Public API: parse_structure(source: SourceBundle) -> VersionTree.

This module constructs a *raw* tree.  Sentence splitting, LaTeX cleanup,
math normalization, citation rewriting, and element IDs belong to TV4.

Pure Python (standard library only); no filesystem or network access.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import posixpath
import re
from typing import Mapping

from .contracts import (
    Diagnostic,
    DiagnosticSeverity,
    NodeKind,
    ProcessingStatus,
    SourceBundle,
    TreeNode,
    RawTree,
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
_RAW_TEXT_ENVS = {"verbatim", "verbatim*", "lstlisting", "minted", "Verbatim", "alltt"}
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
    r"\\(?:input|include)(?![A-Za-z])\s*(?:\{([^{}]*)\}|([A-Za-z0-9_./-]+))"
)
_SIMPLE_PATH_MACRO_RE = re.compile(
    r"\\(?P<kind>providecommand|newcommand|renewcommand)"
    r"\s*\{\\(?P<name>[A-Za-z@]+)\}\s*"
    r"\{(?P<value>[A-Za-z0-9_./-]+)\}"
)
_NEWLIST_RE = re.compile(
    r"\\newlist\s*"
    r"\{\s*([A-Za-z@][A-Za-z0-9@]*)\s*\}"
    r"\s*\{\s*(enumerate|itemize|description)\s*\}"
    r"\s*\{\s*(\d+)\s*\}"
)
_PATH_MACRO_USE_RE = re.compile(r"\\([A-Za-z@]+)")
_BEGIN_END_RE = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
_SPECIAL_RE = re.compile(
    r"\\(?:begin\s*\{[^{}]+\}|end\s*\{[^{}]+\}|"
    r"(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?![A-Za-z])|"
    r"item(?![A-Za-z])|appendix(?![A-Za-z])|"
    r"bibliography(?![A-Za-z])|printbibliography(?![A-Za-z])|"
    r"verb\*?(?![A-Za-z])|\[|\(|\]|\))|\$\$|\$"
)
_COMMAND_DEF_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand)\*?(?![A-Za-z@])"
)
_PARAGRAPH_BREAK_RE = re.compile(r"(?:\n[ \t]*\n+|\\par(?![A-Za-z])\s*)")
_VERBATIM_BEGIN_RE = re.compile(r"\\begin\s*\{(verbatim\*?|lstlisting|minted|Verbatim|alltt)\}")


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


@dataclass(frozen=True, slots=True)
class _TextSpan:
    start: int
    end: int
    filename: str


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

    if body[cursor:].strip():
        return None

    return action, env

def _collect_list_environments(tex: str) -> set[str]:
    environments = set(_LIST_ENVS)

    for match in _NEWLIST_RE.finditer(_strip_comments(tex)):
        name = match.group(1)
        environments.add(name)

    return environments

def _make_diagnostic(code: str, message: str, *, file_path: str | None = None,
                     severity: DiagnosticSeverity = DiagnosticSeverity.WARNING) -> Diagnostic:
    return Diagnostic(severity=severity, code=code, message=message, file_path=file_path)


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

    # Prefer paths that are actually present in the bundle.
    for candidate_name in names:
        for base_file in (root_file, current_file, ""):
            candidate = _safe_relative_path(base_file, candidate_name)

            if candidate is not None and candidate in contents:
                return candidate

    # Preserve a useful path for MISSING_INCLUDE diagnostics.
    return _safe_relative_path(current_file, name)


_IF_FILE_EXISTS_RE = re.compile(r"\\IfFileExists(?![A-Za-z])")


def _select_known_file_exists_branches(
    tex: str,
    root_file: str,
    current_file: str,
    contents: dict[str, str],
) -> str:
    """Select the true branch when a checked file exists in SourceBundle."""
    pieces = []
    position = 0
    protected = _protected_include_spans(tex)

    for match in _IF_FILE_EXISTS_RE.finditer(tex):
        if match.start() < position:
            continue

        if any(a <= match.start() < b for a, b in protected):
            continue

        cursor = match.end()
        arguments = []

        for _ in range(3):
            while cursor < len(tex) and tex[cursor].isspace():
                cursor += 1

            argument = _balanced(tex, cursor, "{", "}")
            if argument is None:
                break

            arguments.append(argument[0])
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

        pieces.append(tex[position:match.start()])
        pieces.append(true_branch)
        position = cursor

    pieces.append(tex[position:])
    return "".join(pieces)


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
        diagnostics.append(_make_diagnostic("ROOT_NOT_FOUND", "Root TeX file missing from SourceBundle"))
        return _ExpandedSource("", [], root)

    parts: list[str] = []
    spans: list[_TextSpan] = []
    offset = 0
    path_macros: dict[str, str] = {}

    def append(fragment: str, filename: str) -> None:
        nonlocal offset
        if fragment:
            parts.append(fragment)
            spans.append(_TextSpan(offset, offset + len(fragment), filename))
            offset += len(fragment)

    def visit(filename: str, ancestors: tuple[str, ...]) -> None:
        if len(ancestors) >= 64:
            diagnostics.append(_make_diagnostic(
                "INCLUDE_DEPTH_LIMIT", "Include nesting exceeds 64 levels", file_path=filename,
            ))
            return
        if filename in ancestors:
            diagnostics.append(_make_diagnostic(
                "INCLUDE_CYCLE", f"Circular include: {' -> '.join((*ancestors, filename))}",
                file_path=filename,
            ))
            return
        tex = contents.get(filename)
        if tex is None:
            diagnostics.append(_make_diagnostic("MISSING_INCLUDE", f"Missing include: {filename}", file_path=filename))
            return
        tex = _select_known_file_exists_branches(
            tex,
            root_file=root,
            current_file=filename,
            contents=contents,
        )
        position = 0
        protected = _protected_include_spans(tex)
        for match in _INCLUDE_RE.finditer(tex):
            if any(begin <= match.start() < end for begin, end in protected):
                continue
            append(tex[position:match.start()], filename)
            # Read simple path macro definitions encountered before this include.
            for definition in _SIMPLE_PATH_MACRO_RE.finditer(
                tex, position, match.start()
            ):
                kind = definition.group("kind")
                name = definition.group("name")
                value = definition.group("value")

                if kind == "providecommand":
                    path_macros.setdefault(name, value)
                else:
                    path_macros[name] = value
            # raw_name = match.group(1) or match.group(2)

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
                ))

                # Preserve the original command and continue parsing.
                append(tex[match.start():match.end()], filename)
                position = match.end()
                continue



            # Substitute known simple path macros.
            resolved_name = _PATH_MACRO_USE_RE.sub(
                lambda m: path_macros.get(m.group(1), m.group(0)),
                raw_name,
            )

            # A parameterized include is a template, not a concrete file path.
            if re.search(r"#[1-9]", resolved_name):
                # Preserve the original command as raw LaTeX.
                append(tex[match.start():match.end()], filename)
                position = match.end()
                continue

            # Skip compiled bibliography in hierarchy.
            if resolved_name.strip().lower().endswith(".bbl"):
                position = match.end()
                continue

            child = _resolve_include_path(
                root_file=root,
                current_file=filename,
                name=resolved_name,
                contents=contents,
            )

            if child is None:
                diagnostics.append(_make_diagnostic(
                    "INVALID_INCLUDE_PATH", f"Unsafe or empty include: {match.group(1)!r}", file_path=filename,
                ))
            else:
                visit(child, (*ancestors, filename))
            position = match.end()
        append(tex[position:], filename)

    visit(root, ())
    return _ExpandedSource("".join(parts), spans, root)


def _balanced(text: str, start: int, opening: str, closing: str) -> tuple[str, int] | None:
    """Read a {nested} or [nested] argument, respecting escaped delimiters."""
    if start >= len(text) or text[start] != opening:
        return None
    depth = 1
    i = start + 1
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in {opening, closing}:
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



def _collect_simple_structural_macros(
    text: str,
) -> tuple[dict[str, tuple[str, str]], str]:

    macros: dict[str, tuple[str, str]] = {}
    chars = list(text)
    consumed = 0

    for match in _COMMAND_DEF_RE.finditer(text):
        if match.start() < consumed:
            continue

        cursor = match.end()

        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

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
    macros: Mapping[str, tuple[str, str]],
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
) -> tuple[str, int, int] | None:

    normal = _SPECIAL_RE.search(text, position)

    macro = (
        call_pattern.search(text, position)
        if call_pattern is not None else None
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

def _parse_expanded(expanded: _ExpandedSource, diagnostics: list[Diagnostic]) -> TreeNode:
    """Build mutable hierarchy, then freeze to the team's TreeNode contract."""
    list_envs = _collect_list_environments(expanded.text)
    tex = expanded.text
    structural_macros, parse_text = _collect_simple_structural_macros(expanded.text)
    macro_call_pattern = _structural_call_pattern(structural_macros)
    root = _MutableNode("document", source_file=expanded.fallback)
    headings: list[_MutableNode] = []
    lists: list[_ListFrame] = []
    scoped_headings: list[tuple[str, _MutableNode]] = []
    pending: list[str] = []
    pending_start: int | None = None
    appendix = False
    suppressed_level: int | None = None

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
        event = _next_structural_token(
            parse_text,
            position,
            structural_macros,
            macro_call_pattern,
        )

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


        if _BEGIN_END_RE.fullmatch(token):
            env_match = _BEGIN_END_RE.fullmatch(token)
            if not env_match:
                if suppressed_level is None:
                    append_text(token, token_start)
                continue
            action, env = env_match.groups()
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
                        diagnostics.append(_make_diagnostic("UNCLOSED_ENV", f"Unclosed {env}", file_path=expanded.source_file_at(token_start)))
                        position = len(tex)
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
                    if end is None:
                        diagnostics.append(_make_diagnostic("UNCLOSED_ENV", f"Unclosed {env}", file_path=expanded.source_file_at(token_start)))
                        end = len(tex)
                    if suppressed_level is None:
                        flush()
                        kind = "block_formula" if env in _MATH_ENVS else ("figure" if env in _FIGURE_ENVS else "paragraph")
                        parent().children.append(_MutableNode(
                            kind, content=tex[token_start:end],
                            source_file=expanded.source_file_at(token_start),
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
                        lists.append(_ListFrame(env, node))

                    # A real \begin{list} has two mandatory arguments.
                    # A synthetic token from a macro call does not.
                    is_macro_call = (
                        parse_text[token_start:token_end] != token
                    )

                    # Inline enumeration may have an optional label format:
                    # \begin{inparaenum}[(1)]
                    # The label format is configuration, not paragraph content.
                    if env == "inparaenum" and not is_macro_call:
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
            if env in list_envs:
                if suppressed_level is None:
                    flush()
                    if lists and lists[-1].environment == env:
                        lists.pop()
                    else:
                        diagnostics.append(_make_diagnostic("LIST_MISMATCH", f"Unmatched end{{{env}}}", file_path=expanded.source_file_at(token_start)))
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
                append_text(token, token_start)
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
                diagnostics.append(_make_diagnostic("INVALID_HEADING", f"Missing title for {command}", file_path=expanded.source_file_at(token_start)))
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
                diagnostics.append(_make_diagnostic("ITEM_OUTSIDE_LIST", r"\item outside a list", file_path=expanded.source_file_at(token_start)))
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
            end = tex.find(closer, position)
            if end < 0:
                append_text(token, token_start)
                diagnostics.append(_make_diagnostic("UNCLOSED_MATH", f"Unclosed {token}", file_path=expanded.source_file_at(token_start)))
            else:
                flush()
                end += len(closer)
                parent().children.append(_MutableNode(
                    "block_formula", content=tex[token_start:end],
                    source_file=expanded.source_file_at(token_start),
                    attributes={"environment": "display_math"},
                ))
                position = end
            continue

        
        if token in {r"\(", "$"}:
            # Ignore escaped opening dollar signs: \$.
            if token == "$":
                backslashes = 0
                j = token_start - 1

                while j >= 0 and tex[j] == "\\":
                    backslashes += 1
                    j -= 1

                if backslashes % 2:
                    append_text(token, token_start)
                    continue

            # Expected and mismatched closing delimiters.
            closer = r"\)" if token == r"\(" else "$"
            wrong_closer = "$" if token == r"\(" else r"\)"

            def find_valid_delimiter(
                marker: str,
                start: int,
                stop: int,
            ) -> int:
                """Find an unescaped delimiter in [start, stop)."""
                cursor = start

                while True:
                    candidate = tex.find(marker, cursor, stop)

                    if candidate < 0:
                        return -1

                    backslashes = 0
                    j = candidate - 1

                    while j >= 0 and tex[j] == "\\":
                        backslashes += 1
                        j -= 1

                    if backslashes % 2 == 0:
                        return candidate

                    cursor = candidate + len(marker)

            # Find the expected closing delimiter.
            correct_end = find_valid_delimiter(
                closer, position, len(tex)
            )

            # Conservative recovery: only inspect the
            # current line for a mismatched delimiter.
            line_end = tex.find("\n", position)

            if line_end < 0:
                line_end = len(tex)

            wrong_end = find_valid_delimiter(
                wrong_closer, position, line_end
            )

            # Recover only if the wrong delimiter appears
            # before the expected closing delimiter.
            if (
                wrong_end >= 0
                and (
                    correct_end < 0
                    or wrong_end < correct_end
                )
            ):
                recovery_end = wrong_end + len(wrong_closer)

                append_text(
                    tex[token_start:recovery_end],
                    token_start,
                )

                diagnostics.append(
                    _make_diagnostic(
                        "UNCLOSED_MATH",
                        (
                            "Mismatched inline math: "
                            f"expected {closer!r} "
                            f"but found {wrong_closer!r}"
                        ),
                        file_path=expanded.source_file_at(
                            token_start
                        ),
                    )
                )

                position = recovery_end
                continue

            # Normal case: correctly closed inline math.
            if correct_end >= 0:
                end = correct_end + len(closer)

                append_text(tex[token_start:end], token_start)
                position = end
                continue

            # Preserve existing behavior when no closing
            # delimiter can be found.
            append_text(token, token_start)
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
        diagnostics.append(_make_diagnostic("UNCLOSED_LIST", f"{len(lists)} list environment(s) not closed"))
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
        status = (ProcessingStatus.EMPTY_NO_TEX if source_status == "empty_no_tex" else ProcessingStatus.ROOT_NOT_FOUND)
        return RawTree(
            arxiv_id=source.arxiv_id, version=source.version,
            root=TreeNode(kind=NodeKind.DOCUMENT, source_file=source.root_file),
            status=status, diagnostics=tuple(diagnostics),
        )

    expanded = _expand_bundle(source, diagnostics)
    if not expanded.text and any(d.code == "ROOT_NOT_FOUND" for d in diagnostics):
        return RawTree(
            arxiv_id=source.arxiv_id, version=source.version,
            root=TreeNode(kind=NodeKind.DOCUMENT, source_file=source.root_file),
            status=ProcessingStatus.ROOT_NOT_FOUND, diagnostics=tuple(diagnostics),
        )

    root = _parse_expanded(expanded, diagnostics)
    # Input problems are carried forward, but unexpected code exceptions are
    # intentionally not swallowed: hiding programming errors breaks testing.
    is_partial = source_status not in {"ok"} or any(
        d.code in {"MISSING_INCLUDE", "INCLUDE_CYCLE", "INCLUDE_DEPTH_LIMIT", "INVALID_INCLUDE_PATH",
                   "UNCLOSED_ENV", "LIST_MISMATCH", "UNCLOSED_LIST", "UNCLOSED_MATH"}
        for d in diagnostics
    )
    return RawTree(
        arxiv_id=source.arxiv_id, version=source.version, root=root,
        status=ProcessingStatus.PARTIAL_PARSE if is_partial else ProcessingStatus.OK,
        diagnostics=tuple(diagnostics),
    )
