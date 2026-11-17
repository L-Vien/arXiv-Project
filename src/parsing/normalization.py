"""Tree normalization (TV4, Milestone 2).

normalize_tree(tree, citation_map) takes TV3's RawTree and returns a NormalizedVersion (nodes in document
order, local_id n0, n1, ...; order = pre-order index). clean_tree() is the intermediate step returning a
cleaned RawTree:
  - strips meaningless formatting commands and comments, collapses whitespace;
  - inline math -> $...$, block math -> \\begin{equation}...\\end{equation};
  - paragraphs are split into sentences and spliced into the parent (no paragraph in the final tree);
  - figure/table: content = cleaned caption;
  - citation keys are rewritten with citation_map (from TV5);
  - URLs, decimals, abbreviations, citations and math are protected during sentence splitting.

Pure functions: no file I/O and no global state.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace

from src.parsing.citation_rewrite import rewrite_citations
from src.parsing.contracts import NodeKind, NormalizedNode, NormalizedVersion, RawTree, TreeNode

# TreeNode.kind is a plain string in the contract (not an Enum), so compare against string values.
_DOCUMENT = NodeKind.DOCUMENT.value
_HEADING = NodeKind.HEADING.value
_PARAGRAPH = NodeKind.PARAGRAPH.value
_LIST = NodeKind.LIST.value
_LIST_ITEM = NodeKind.LIST_ITEM.value
_SENTENCE = NodeKind.SENTENCE.value
_BLOCK = NodeKind.BLOCK_FORMULA.value
_FIGURE = NodeKind.FIGURE.value

# ---------------------------------------------------------------------------
# Placeholders: keep spans that must stay untouched (math, URL, citation, footnote)
# ---------------------------------------------------------------------------
_PH_OPEN, _PH_CLOSE = "\ue000", "\ue001"
_FN_OPEN, _FN_CLOSE = "\ue002", "\ue003"  # footnotes use their own delimiters (see split_sentences)
_PH_RE = re.compile(f"[{_PH_OPEN}{_FN_OPEN}](\\d+)[{_PH_CLOSE}{_FN_CLOSE}]")


class _Protector:
    def __init__(self) -> None:
        self._items: list[str] = []

    def hold(self, text: str, footnote: bool = False) -> str:
        self._items.append(text)
        open_, close = (_FN_OPEN, _FN_CLOSE) if footnote else (_PH_OPEN, _PH_CLOSE)
        return f"{open_}{len(self._items) - 1}{close}"

    def restore(self, text: str) -> str:
        for _ in range(4):  # placeholders may be nested a few levels deep
            new = _PH_RE.sub(lambda m: self._items[int(m.group(1))], text)
            if new == text:
                break
            text = new
        return text


# ---------------------------------------------------------------------------
# Brace helpers
# ---------------------------------------------------------------------------
def _find_group(text: str, start: int) -> int:
    """text[start] == '{' -> index of the matching '}', or -1."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _command_re(names: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\\(?:%s)\*?(?![A-Za-z])(?:\s*\[[^\]]*\])*\s*(?=\{)" % "|".join(names))


def _unwrap(text: str, names: tuple[str, ...]) -> str:
    """\\textbf{x} -> x (keep the content, drop the command)."""
    pattern = _command_re(names)
    while True:
        m = pattern.search(text)
        if not m:
            return text
        open_i = m.end()
        close_i = _find_group(text, open_i)
        if close_i == -1:
            text = text[: m.start()] + text[open_i + 1:]
        else:
            text = text[: m.start()] + text[open_i + 1: close_i] + text[close_i + 1:]


def _drop_with_arg(text: str, names: tuple[str, ...]) -> str:
    """\\label{x} -> '' (drop both the command and its argument)."""
    pattern = _command_re(names)
    while True:
        m = pattern.search(text)
        if not m:
            return text
        open_i = m.end()
        close_i = _find_group(text, open_i)
        end = len(text) if close_i == -1 else close_i + 1
        text = text[: m.start()] + text[end:]


def _protect_with_arg(
    text: str, names: tuple[str, ...], prot: _Protector, n_args: int = 1, footnote: bool = False
) -> str:
    """Keep \\cmd{...}{...} verbatim behind a placeholder."""
    pattern = _command_re(names)
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            return text
        end, ok = m.end(), True
        for _ in range(n_args):
            while end < len(text) and text[end].isspace():
                end += 1
            if end >= len(text) or text[end] != "{":
                ok = False
                break
            close = _find_group(text, end)
            if close == -1:
                ok = False
                break
            end = close + 1
        if not ok:
            pos = m.end()
            continue
        placeholder = prot.hold(text[m.start(): end], footnote=footnote)
        text = text[: m.start()] + placeholder + text[end:]
        pos = m.start() + len(placeholder)


# ---------------------------------------------------------------------------
# LaTeX command tables
# ---------------------------------------------------------------------------
_UNWRAP_CMDS = (
    "textbf", "textit", "emph", "textsc", "textrm", "textsf", "texttt", "textsl", "textup",
    "textmd", "textnormal", "underline", "uline", "mbox", "hbox", "bf", "it", "em",
    "mathrm", "textcolor", "textgreek", "abstract",
)
_DROP_CMDS = (
    "label", "index", "vspace", "hspace", "thispagestyle", "pagestyle", "phantom", "hphantom", "vphantom",
    # front matter and file-level commands: never prose
    "title", "author", "affiliation", "affil", "altaffiliation", "email", "address", "thanks", "date",
    "orcid", "institute", "collaboration", "input", "include", "usepackage", "bibliography",
    "bibliographystyle", "nocite",
)
_BARE_CMDS = (
    "noindent", "centering", "raggedright", "raggedleft", "bfseries", "itshape", "scshape",
    "rmfamily", "sffamily", "ttfamily", "mdseries", "upshape", "slshape", "tiny", "scriptsize",
    "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge", "newpage",
    "clearpage", "medskip", "bigskip", "smallskip", "par", "linebreak", "newline", "quad", "qquad",
    "maketitle", "tableofcontents", "toprule", "midrule", "bottomrule",
)
# Environment markers left in text fragments (proof, lem, theorem, abstract, center, ...). Floats and display
# math are extracted before this runs, so only the markers are dropped, never their content.
_ENV_MARK_RE = re.compile(
    r"\\begin\{minipage\}(?:\[[^\]]*\])?\{[^{}]*\}"
    r"|\\begin\{[A-Za-z]+\*?\}(?:\s*\[[^\]]*\])?"
    r"|\\end\{[A-Za-z]+\*?\}"
)
_BARE_RE = re.compile(r"\\(?:%s)(?![A-Za-z])" % "|".join(_BARE_CMDS))
_SPACING_RE = re.compile(r"\\[,;:! ]")
_LINEBREAK_RE = re.compile(r"\\\\\*?(?:\s*\[[^\]]*\])?")
_DECL_RE = re.compile(r"\{\s*\\(?:bf|it|em|sl|sc|tt|rm|sf|bfseries|itshape)\s+([^{}]*)\}")
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")

_CITE_RE = re.compile(r"\\[A-Za-z]*[Cc]ite[A-Za-z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]*)\}")
_URL_PLAIN_RE = re.compile(r"https?://[^\s{}]*[^\s{}.,;:!?)\]]")

_ENVS = "equation|displaymath|align|gather|multline|eqnarray|flalign|alignat"
_DISPLAY_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|\\begin\{(?P<env>(?:%s)\*?)\}(?P<body>.+?)\\end\{(?P=env)\}" % _ENVS,
    re.DOTALL,
)
_INLINE_RE = re.compile(
    r"(?<!\\)\$(?!\$)(?P<d>(?:\\.|[^$\\])+?)(?<!\\)\$|\\\((?P<p>.+?)\\\)", re.DOTALL
)
_FLOAT_ENVS = (
    "figure|table|algorithm|wrapfigure|wraptable|sidewaysfigure|sidewaystable|deluxetable|longtable"
    "|tabular|tabularx|tabulary|supertabular"
)
_FLOAT_RE = re.compile(rf"\\begin\{{(?P<fenv>(?:{_FLOAT_ENVS})\*?)\}}.*?\\end\{{(?P=fenv)\}}", re.DOTALL)
_ITEM_SPLIT_RE = re.compile(r"\\item(?![A-Za-z])(?:\s*\[[^\]]*\])?")
_MATH_NOISE_RE = re.compile(
    r"\\(?:nonumber|notag|centering|toprule|midrule|bottomrule)(?![A-Za-z])"
)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
def _canon_inline(match: re.Match[str]) -> str:
    inner = match.group("d") if match.group("d") is not None else match.group("p")
    inner = _drop_with_arg(inner, ("label",))
    inner = _MATH_NOISE_RE.sub("", inner)
    return "$" + " ".join(inner.split()) + "$"


def _canon_block(body: str, env: str | None) -> str:
    body = _drop_with_arg(_COMMENT_RE.sub("", body), ("label",))
    body = _MATH_NOISE_RE.sub("", body)
    if env and env.startswith("alignat"):
        body = re.sub(r"^\s*\{\d+\}", "", body)
    body = " ".join(body.split())
    base = (env or "equation").rstrip("*")
    if base in ("align", "flalign", "alignat", "eqnarray"):
        body = rf"\begin{{aligned}}{body}\end{{aligned}}"
    elif base == "gather":
        body = rf"\begin{{gathered}}{body}\end{{gathered}}"
    elif base == "multline":
        body = rf"\begin{{multlined}}{body}\end{{multlined}}"
    return rf"\begin{{equation}}{body}\end{{equation}}"


def _canon_display(m: re.Match[str]) -> str:
    if m.group("dd") is not None:
        return _canon_block(m.group("dd"), None)
    if m.group("br") is not None:
        return _canon_block(m.group("br"), None)
    return _canon_block(m.group("body"), m.group("env"))


def normalize_block_formula(content: str) -> str:
    """Content of a block_formula node -> \\begin{equation}...\\end{equation}."""
    content = content.strip()
    m = _DISPLAY_RE.fullmatch(content)
    return _canon_display(m) if m else _canon_block(content, None)


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------
def rewrite_citation_keys(text: str, citation_map: Mapping[str, str]) -> str:
    """Delegate citation rewriting to TV5's shared implementation."""
    return rewrite_citations(text, citation_map)


# ---------------------------------------------------------------------------
# Text cleanup (fragments without display math)
# ---------------------------------------------------------------------------
def _prepare_text(text: str, cmap: Mapping[str, str], prot: _Protector) -> str:
    text = _INLINE_RE.sub(lambda m: prot.hold(_canon_inline(m)), text)
    text = _DECL_RE.sub(r"\1", text)
    text = _unwrap(text, _UNWRAP_CMDS)
    text = _drop_with_arg(text, _DROP_CMDS)
    text = _ENV_MARK_RE.sub(" ", text)
    text = _BARE_RE.sub(" ", text)
    text = _LINEBREAK_RE.sub(" ", text)
    text = _SPACING_RE.sub(" ", text)
    text = _CITE_RE.sub(lambda m: prot.hold(rewrite_citation_keys(m.group(0), cmap)), text)
    text = _protect_with_arg(text, ("footnote",), prot, footnote=True)
    text = text.replace("~", " ").replace("``", '"').replace("''", '"')
    for esc in ("%", "&", "#", "_"):
        text = text.replace("\\" + esc, esc)
    return " ".join(text.split())


_IFFALSE_RE = re.compile(r"\\iffalse(?![A-Za-z]).*?\\fi(?![A-Za-z])", re.DOTALL)
_COMMENT_ENV_RE = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.DOTALL)

# Macro definitions are code, not prose. Value = number of "units" (name or body group) to consume.
_DEF_UNITS = {
    "newcommand": 2, "renewcommand": 2, "providecommand": 2, "DeclareMathOperator": 2, "newtheorem": 2,
    "newenvironment": 3, "renewenvironment": 3, "setlength": 2, "setcounter": 2, "addtolength": 2,
    "def": 2, "let": 2,
}
_DEF_RE = re.compile(r"\\(%s)\*?(?![A-Za-z])" % "|".join(_DEF_UNITS))
_CS_RE = re.compile(r"\\(?:[A-Za-z]+|.)", re.DOTALL)


def _drop_definitions(text: str) -> str:
    """Remove \\newcommand / \\def / \\let / \\newtheorem / ... together with their arguments."""
    out: list[str] = []
    pos = 0
    while True:
        m = _DEF_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos: m.start()])
        i, done, units = m.end(), 0, _DEF_UNITS[m.group(1)]
        while done < units and i < len(text):
            j = i
            while j < len(text) and text[j] in " \t\n=":
                j += 1
            if j >= len(text):
                break
            c = text[j]
            if c == "{":
                close = _find_group(text, j)
                if close == -1:
                    break
                i, done = close + 1, done + 1
            elif c == "[" and done > 0:  # optional arguments: [n][default]
                close = text.find("]", j)
                if close == -1:
                    break
                i = close + 1
            elif c == "\\":
                cs = _CS_RE.match(text, j)
                i, done = cs.end(), done + 1
            elif c == "#":  # \def parameter text: #1#2
                i = j + 2
            else:
                break
        if m.group(1) == "newtheorem":  # optional counter: \newtheorem{thm}{Theorem}[section]
            k = i
            while k < len(text) and text[k] in " \t\n":
                k += 1
            if k < len(text) and text[k] == "[":
                close = text.find("]", k)
                i = close + 1 if close != -1 else i
        pos = i


# Shorthand macros many authors define for display math (\be ... \ee); the definitions live in the
# preamble, so common names are expanded heuristically (only when both ends appear in order).
_ENV_ALIASES = (
    ("be", "ee", "equation"), ("beq", "eeq", "equation"), ("bea", "eea", "eqnarray"),
    ("bq", "eq", "eqnarray"), ("ba", "ea", "eqnarray"),
)
_ALIAS_RES = [
    (re.compile(rf"\\{b}(?![A-Za-z])(.+?)\\{e}(?![A-Za-z])", re.DOTALL), env) for b, e, env in _ENV_ALIASES
]


def _expand_env_aliases(text: str) -> str:
    for pattern, env in _ALIAS_RES:
        text = pattern.sub(lambda m, env=env: rf"\begin{{{env}}}{m.group(1)}\end{{{env}}}", text)
    return text


def _pre_clean(raw: str, prot: _Protector) -> str:
    """CRLF -> LF, protect URLs, strip comments, dead code, macro definitions; expand \\be ... \\ee."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _protect_with_arg(raw, ("url",), prot)
    raw = _protect_with_arg(raw, ("href",), prot, n_args=2)
    raw = _URL_PLAIN_RE.sub(lambda m: prot.hold(m.group(0)), raw)
    raw = _COMMENT_RE.sub("", raw)
    raw = _COMMENT_ENV_RE.sub(" ", _IFFALSE_RE.sub(" ", raw))
    return _expand_env_aliases(_drop_definitions(raw))


def normalize_text(raw: str, citation_map: Mapping[str, str] | None = None) -> str:
    """Clean a single string (heading, caption) without sentence splitting."""
    prot = _Protector()
    text = _prepare_text(_pre_clean(raw, prot), citation_map or {}, prot)
    return prot.restore(text).strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
_NEVER_SPLIT = {
    "eg", "ie", "cf", "vs", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "ref", "refs",
    "vol", "pp", "p", "ch", "dr", "mr", "mrs", "ms", "prof", "approx", "resp", "viz", "ca",
    "inc", "ltd", "jr", "sr", "thm", "lem", "prop", "cor",
}
_SOFT_ABBREV = {"etc", "al"}  # split only if the next word is capitalized
_BOUNDARY_RE = re.compile(rf"([.!?]+)([\"')\]]*)((?:\s*{_FN_OPEN}\d+{_FN_CLOSE})*)\s+")
_TOKEN_BEFORE_RE = re.compile(r"(?:^|[\s(\[{])([A-Za-z][A-Za-z.]*)$")
_STARTERS = "\"'(["


def _is_boundary(before: str, punct: str, nxt: str) -> bool:
    if not (nxt.isupper() or nxt.isdigit() or nxt in _STARTERS or nxt == _PH_OPEN):
        return False
    if punct == ".":
        tok = _TOKEN_BEFORE_RE.search(before)
        if tok:
            raw = tok.group(1)
            word = raw.replace(".", "").lower()
            if word in _NEVER_SPLIT:
                return False
            if word in _SOFT_ABBREV:
                return nxt.isupper()
            if len(raw) == 1 and raw.isupper():  # author initial: "J. Smith"
                return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split text that already contains placeholders into sentences."""
    text = text.strip()
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        nxt = text[m.end(): m.end() + 1]
        if not nxt or not _is_boundary(text[start: m.start()], m.group(1), nxt):
            continue
        sentences.append(text[start: m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _paragraph_units(raw: str, cmap: Mapping[str, str]) -> list[tuple[str, str]]:
    """Raw paragraph content -> list of (sentence | block_formula | figure, content).

    Floats (figure, table, algorithm, bare tabular, ...) found inside the paragraph become figure units
    and display math becomes block_formula units; the text around them is split into sentences.
    """
    prot = _Protector()
    text = _pre_clean(raw, prot)
    units: list[tuple[str, str]] = []

    def add_text(segment: str, after_block: bool) -> None:
        if after_block:
            segment = segment.lstrip(" \t\n.,;:")
        # A stray \item (list that the parser left inside a paragraph) is a hard sentence boundary.
        for part in _ITEM_SPLIT_RE.split(segment):
            prepared = _prepare_text(part, cmap, prot)
            for s in split_sentences(prepared):
                s = prot.restore(s).strip()
                if s and any(ch.isalnum() for ch in s):
                    units.append((_SENTENCE, s))

    def add_with_math(segment: str) -> None:
        pos, after_block = 0, False
        for m in _DISPLAY_RE.finditer(segment):
            add_text(segment[pos: m.start()], after_block)
            units.append((_BLOCK, prot.restore(_canon_display(m))))
            pos, after_block = m.end(), True
        add_text(segment[pos:], after_block)

    pos = 0
    for m in _FLOAT_RE.finditer(text):
        add_with_math(text[pos: m.start()])
        units.append((_FIGURE, normalize_figure_content(prot.restore(m.group(0)), cmap)))
        pos = m.end()
    add_with_math(text[pos:])
    return units


# ---------------------------------------------------------------------------
# Figure / table
# ---------------------------------------------------------------------------
_CAPTION_RE = re.compile(r"\\caption\*?\s*(?:\[[^\]]*\])?\s*(?=\{)")
_FIGURE_NOISE_RE = re.compile(
    r"\\begin\{(tabular\*?|tabularx|longtable|array)\}.*?\\end\{\1\}"
    r"|\\includegraphics\*?(?:\[[^\]]*\])?\{[^}]*\}",
    re.DOTALL,
)


def normalize_figure_content(raw: str, cmap: Mapping[str, str]) -> str:
    """Figure/table content -> cleaned caption (multiple captions joined by one space)."""
    raw = _COMMENT_RE.sub("", raw)
    captions: list[str] = []
    for m in _CAPTION_RE.finditer(raw):
        close = _find_group(raw, m.end())
        captions.append(raw[m.end() + 1: close if close != -1 else len(raw)])
    if captions:
        return " ".join(c for c in (normalize_text(c, cmap) for c in captions) if c)
    return normalize_text(_FIGURE_NOISE_RE.sub(" ", raw), cmap)


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------
def _text_nodes(node: TreeNode, cmap: Mapping[str, str]) -> list[TreeNode]:
    return [
        TreeNode(kind=kind, content=content, source_file=node.source_file)
        for kind, content in _paragraph_units(node.content or "", cmap)
    ]


def _normalize_node(node: TreeNode, cmap: Mapping[str, str]) -> list[TreeNode]:
    kind = node.kind
    children: list[TreeNode] = []
    for child in node.children:
        children.extend(_normalize_node(child, cmap))

    if kind == _PARAGRAPH:  # splice into the parent node
        return _text_nodes(node, cmap) + children
    if kind in (_LIST_ITEM, _LIST):
        lead = _text_nodes(node, cmap) if node.content else []
        return [replace(node, content=None, children=tuple(lead + children))]
    if kind == _HEADING:
        return [replace(node, content=normalize_text(node.content or "", cmap), children=tuple(children))]
    if kind == _DOCUMENT:
        return [replace(node, content=None, children=tuple(children))]
    if kind == _BLOCK:
        return [replace(node, content=normalize_block_formula(node.content or ""), children=())]
    if kind == _FIGURE:
        return [replace(node, content=normalize_figure_content(node.content or "", cmap), children=())]
    if kind == _SENTENCE:  # already a sentence: clean again without splitting (idempotent)
        content = normalize_text(node.content or "", cmap)
        return [replace(node, content=content, children=())] if content else []
    return children  # unknown kind: keep its children


def clean_tree(tree: RawTree, citation_map: Mapping[str, str] | None = None) -> RawTree:
    """RawTree -> cleaned RawTree (no paragraph nodes left). A tree with root None is returned unchanged."""
    if tree.root is None:
        return tree
    roots = _normalize_node(tree.root, citation_map or {})
    if len(roots) == 1 and roots[0].kind == _DOCUMENT:
        root = roots[0]
    else:
        root = TreeNode(kind=_DOCUMENT, children=tuple(roots))
    return replace(tree, root=root)


def _count_nodes(root: TreeNode) -> int:
    total, stack = 0, [root]
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def _flatten(root: TreeNode) -> tuple[NormalizedNode, ...]:
    """Pre-order walk: order = pre-order index (root = 0), local_id = n<order>."""
    nodes: list[NormalizedNode] = []
    stack: list[tuple[TreeNode, str | None]] = [(root, None)]
    while stack:
        node, parent = stack.pop()
        order = len(nodes)
        local_id = f"n{order}"
        nodes.append(NormalizedNode(local_id, node.kind, node.content or "", parent, order))
        stack.extend((child, local_id) for child in reversed(node.children))
    return tuple(nodes)


def normalize_tree(
    tree: RawTree,
    citation_map: Mapping[str, str] | None = None,
) -> NormalizedVersion:
    """RawTree + citation_map -> NormalizedVersion.

    A version without a tree (root None, e.g. empty_no_tex) yields an empty NormalizedVersion that keeps
    its status and diagnostics, so TV1 can still count it in statistics.
    raw_element_count = number of nodes in the input RawTree (root included).
    """
    if tree.root is None:
        return NormalizedVersion(tree.arxiv_id, tree.version, tree.status, None, (), 0, tree.diagnostics)
    cleaned = clean_tree(tree, citation_map)
    nodes = _flatten(cleaned.root)
    return NormalizedVersion(
        arxiv_id=tree.arxiv_id,
        version=tree.version,
        status=tree.status,
        root_local_id=nodes[0].local_id,
        nodes=nodes,
        raw_element_count=_count_nodes(tree.root),
        diagnostics=tree.diagnostics,
    )


class Normalizer:
    """Implements the contracts.TreeNormalizer protocol."""

    def normalize(
        self, tree: RawTree, citation_map: Mapping[str, str] | None = None
    ) -> NormalizedVersion:
        return normalize_tree(tree, citation_map)
