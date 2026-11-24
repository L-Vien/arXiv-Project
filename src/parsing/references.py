"""Reference extraction and BibTeX deduplication for TV5.

This module processes a sequence of SourceBundle objects for all versions of a publication,
extracting references from \\bibitem entries and .bib/.bbl files, deduplicating them across versions,
merging fields, selecting canonical citation keys, and generating per-version citation maps.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import replace
from datetime import date
from typing import Mapping, Sequence, Tuple, List, Dict, Set, Optional, Any
from urllib.parse import unquote

from src.parsing.contracts import (
    BibEntry,
    Diagnostic,
    DiagnosticSeverity,
    ProcessingStatus,
    ReferenceBundle,
    SourceBundle,
)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace and punctuation."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(
    r"(?<![0-9])((?:\d{4}\.\d{4,5}|[a-z][a-z0-9-]+/\d{7})(?:v\d+)?)(?![0-9])",
    re.IGNORECASE,
)
_CITATION_COMMAND_PATTERN = re.compile(
    r"\\(?P<command>[A-Za-z]*cite[A-Za-z]*)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{(?P<keys>[^{}]+)\}",
    re.IGNORECASE,
)
_NOCITE_ALL_PATTERN = re.compile(
    r"\\nocite\s*\{\s*\*\s*\}",
    re.IGNORECASE,
)
_INVALID_CITATION_KEY = re.compile(
    r"\s|[\x00-\x1f\x7f]|\\|^(?:https?://|doi:)",
    re.IGNORECASE,
)

_REFERENCE_BLOCK_PATTERN = re.compile(
    r"\\begin\s*\{(?:verbatim\*?|lstlisting|comment)\}.*?"
    r"\\end\s*\{(?:verbatim\*?|lstlisting|comment)\}",
    re.IGNORECASE | re.DOTALL,
)

_BIB_ENTRY_START_PATTERN = re.compile(
    r"(?im)(?<!\\)@\s*(?:article|book|booklet|conference|dataset|inbook|"
    r"incollection|inproceedings|manual|mastersthesis|misc|online|phdthesis|"
    r"proceedings|software|techreport|unpublished)\s*[\{(]"
)
_TITLE_FIELD_LEAK_PATTERN = re.compile(
    r"(?:,\s*|\s{2,})(?:abstract|author|authors|booktitle|doi|eprint|isbn|issn|"
    r"journal|journaltitle|month|note|number|pages|publisher|school|series|url|"
    r"volume|year)\s*=",
    re.IGNORECASE,
)
_BIB_FIELD_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?:^|,\s*|\n\s*)(?P<name>abstract|address|arxivid|author|authors|"
    r"booktitle|chapter|date|doi|edition|eprint|howpublished|institution|isbn|"
    r"issn|journal|journaltitle|keywords|month|note|number|organization|pages|"
    r"publisher|school|series|title|type|url|volume|year)\s*=\s*"
)

_LATEX_COMMAND_REPLACEMENTS = {
    "apj": "The Astrophysical Journal",
    "apjl": "The Astrophysical Journal Letters",
    "apjs": "The Astrophysical Journal Supplement Series",
    "aaps": "Astronomy and Astrophysics Supplement Series",
    "aj": "The Astronomical Journal",
    "mnras": "Monthly Notices of the Royal Astronomical Society",
    "aap": "Astronomy & Astrophysics",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "times": "×",
    "pm": "±",
    "leq": "≤",
    "geq": "≥",
    "neq": "≠",
    "infty": "∞",
}


def _safe_citation_key(
    raw_key: str,
    *,
    file_path: str,
    sequence: int,
    diagnostics: List[Diagnostic],
) -> str:
    """Return a table-safe key while preserving the original source key.

    Citation commands must still map from their original key.  Invalid keys
    receive deterministic surrogates and are rewritten through citation maps
    rather than being silently trimmed.
    """

    if raw_key and not _INVALID_CITATION_KEY.search(raw_key):
        return raw_key
    diagnostics.append(
        Diagnostic(
            severity=DiagnosticSeverity.WARNING,
            code="INVALID_CITATION_KEY",
            message=f"Invalid citation key {raw_key!r} was replaced by a stable surrogate",
            file_path=file_path,
        )
    )
    digest = hashlib.sha256(
        f"{file_path}\x1f{sequence}\x1f{raw_key}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ref_{digest}"


def _entry_quality(entry: BibEntry) -> int:
    """Rank parser evidence before choosing a conflicting field value."""

    source_rank = {
        "bibtex": 40,
        "biber_bbl": 30,
        "bibitem": 20,
        "harvard_bbl": 10,
    }.get(entry.source_kind, 0)
    status_rank = {"parsed": 4, "partial": 2, "citation_text_only": 0}.get(
        entry.parse_status,
        0,
    )
    return source_rank + status_rank


def _prefer_title(current: BibEntry | None, candidate: BibEntry) -> BibEntry:
    """Choose title evidence without preferring a shorter string by default."""

    if current is None or not current.title:
        return candidate
    if not candidate.title:
        return current
    current_blob = bool(re.search(r"(doi:|https?://|\bet al\b)", current.title, re.IGNORECASE))
    candidate_blob = bool(re.search(r"(doi:|https?://|\bet al\b)", candidate.title, re.IGNORECASE))
    if current_blob != candidate_blob:
        return candidate if not candidate_blob else current
    if _entry_quality(candidate) != _entry_quality(current):
        return candidate if _entry_quality(candidate) > _entry_quality(current) else current
    return candidate if len(candidate.title) > len(current.title) else current


def _strip_reference_comments(content: str) -> str:
    """Remove comments and non-source environments before ``\\bibitem`` parsing."""

    content = _REFERENCE_BLOCK_PATTERN.sub(
        lambda match: re.sub(r"[^\n]", " ", match.group(0)),
        content,
    )
    lines: list[str] = []
    for line in content.splitlines(keepends=True):
        output: list[str] = []
        backslashes = 0
        for char in line:
            if char == "\\":
                backslashes += 1
                output.append(char)
                continue
            if char == "%" and backslashes % 2 == 0:
                break
            output.append(char)
            backslashes = 0
        lines.append("".join(output))
    return "".join(lines)


def _strip_bibtex_comments(content: str) -> str:
    """Strip BibTeX comments without removing percent signs inside values."""
    lines: list[str] = []
    brace_depth = 0
    in_quote = False
    escaped = False
    for line in content.splitlines(keepends=True):
        output: list[str] = []
        for char in line:
            if char == "\\" and not escaped:
                escaped = True
                output.append(char)
                continue
            if char == '"' and not escaped and brace_depth == 0:
                in_quote = not in_quote
            if char == "{" and not in_quote:
                brace_depth += 1
            elif char == "}" and not in_quote and brace_depth:
                brace_depth -= 1
            if char == "%" and not escaped and not in_quote and brace_depth == 0:
                break
            output.append(char)
            escaped = False
        lines.append("".join(output))
    return "".join(lines)


def _normalize_doi(value: str) -> str:
    """Return a stable bare DOI for output and cross-version matching."""
    cleaned = _clean_field_value(value)
    if not cleaned:
        return ""

    cleaned = unquote(cleaned.replace(r"\%", "%"))
    cleaned = re.sub(
        r"^(?:https?://)?(?:dx\.)?doi\.org/",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^doi\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    match = _DOI_PATTERN.search(cleaned)
    if match is None:
        return ""
    return match.group(0).rstrip(".,;:)]}").lower()


def _normalize_arxiv_id(value: str) -> str:
    """Return an arXiv identifier without a version suffix."""
    cleaned = _clean_field_value(value)
    match = _ARXIV_PATTERN.search(cleaned)
    if match is None:
        return ""
    return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE).lower()


def _cited_keys(source: SourceBundle) -> set[str] | None:
    """Return cited keys, ``None`` for ``nocite{*}``, or an empty set."""

    tex = "\n".join(
        file_obj.content
        for file_obj in source.files
        if file_obj.relative_path.lower().endswith(".tex")
    )
    tex = _strip_reference_comments(tex)
    if _NOCITE_ALL_PATTERN.search(tex):
        return None
    keys: set[str] = set()
    for match in _CITATION_COMMAND_PATTERN.finditer(tex):
        if match.group("command").casefold() == "nocite":
            keys.update(
                key.strip()
                for key in match.group("keys").split(",")
                if key.strip() != "*"
            )
        else:
            keys.update(key.strip() for key in match.group("keys").split(","))
    return keys


def _replace_href_labels(text: str) -> str:
    """Replace balanced LaTeX href commands with their visible labels."""

    output: list[str] = []
    cursor = 0
    pattern = re.compile(r"\\href(?:@noop)?(?![A-Za-z@])")
    while match := pattern.search(text, cursor):
        output.append(text[cursor:match.start()])
        url_group = _read_braced_group(text, match.end())
        label_group = (
            _read_braced_group(text, url_group[1])
            if url_group is not None
            else None
        )
        if url_group is None or label_group is None:
            output.append(match.group(0))
            cursor = match.end()
            continue
        output.append(label_group[0])
        cursor = label_group[1]
    output.append(text[cursor:])
    return "".join(output)


def _clean_latex_accents(text: str) -> str:
    """Convert common semantic LaTeX to stable display text."""
    if not text:
        return ""

    text = _replace_href_labels(text)
    text = re.sub(
        r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    wrapper = re.compile(
        r"\\(?:bibfnamefont|bibnamefont|bibtitlefont|emph|ensuremath|mbox|operatorname|path|"
        r"text|textit|textbf|texttt|textnormal|textrm|textsf|mathrm|mathit|"
        r"mathbf|url|JournalTitle)\s*\{([^{}]*)\}",
        re.IGNORECASE,
    )
    while wrapper.search(text):
        text = wrapper.sub(r"\1", text)

    accent_combining = {
        "'": "\u0301",
        "`": "\u0300",
        '"': "\u0308",
        "^": "\u0302",
        "~": "\u0303",
        "H": "\u030b",
        "=": "\u0304",
        ".": "\u0307",
        "u": "\u0306",
        "r": "\u030a",
    }

    def compose_accent(match: re.Match[str]) -> str:
        return unicodedata.normalize(
            "NFC",
            match.group(2) + accent_combining[match.group(1)],
        )

    text = re.sub(
        r"\\([`'\"^~H=.ur])\{\\([ij])\}",
        compose_accent,
        text,
    )
    text = re.sub(
        r"\\([`'\"^~H=.ur])\{([a-zA-Z])\}",
        compose_accent,
        text,
    )
    text = re.sub(
        r"\\([`'\"^~H=.])\\([ij])",
        compose_accent,
        text,
    )
    text = re.sub(
        r"\\([`'\"^~H=.])([a-zA-Z])",
        compose_accent,
        text,
    )

    accent_map = {
        "\\'a": "á", "\\`a": "à", "\\'e": "é", "\\`e": "è", "\\'i": "í", "\\`i": "ì",
        "\\'o": "ó", "\\`o": "ò", "\\'u": "ú", "\\`u": "ù", "\\'n": "ń", "\\'y": "ý",
        '\\"a': "ä", '\\"e': "ë", '\\"i': "ï", '\\"o': "ö", '\\"u': "ü", '\\"y': "ÿ",
        '\\^a': "â", '\\^e': "ê", '\\^i': "î", '\\^o': "ô", '\\^u': "û",
        '\\~a': "ã", '\\~n': "ñ", '\\~o': "õ",
    }
    for pat, rep in accent_map.items():
        text = text.replace(pat, rep)

    text = re.sub(r"\\v\{([a-zA-Z])\}", lambda m: {"c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž"}.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\v\s*([a-zA-Z])", lambda m: {"c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž"}.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\c\{([a-zA-Z])\}", lambda m: {"c": "ç", "C": "Ç"}.get(m.group(1), m.group(1)), text)
    while wrapper.search(text):
        text = wrapper.sub(r"\1", text)
    text = re.sub(
        r"\\(?:bf|it|em)(?![A-Za-z@])\s*",
        "",
        text,
    )
    text = text.replace(r"\LaTeX", "LaTeX").replace(r"\TeX", "TeX")
    text = re.sub(r"\\text([A-Z][A-Za-z0-9]*)", r"\1", text)
    for command, replacement in sorted(
        _LATEX_COMMAND_REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(
            rf"\\{re.escape(command)}(?![A-Za-z@])",
            lambda _match, value=replacement: value,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\\(?:quad|qquad|,|;|:|!)(?![A-Za-z@])", " ", text)
    text = re.sub(r"\\(?:&|%|#|_|\$)", lambda match: match.group(0)[1:], text)
    text = text.replace(r"\{", "").replace(r"\}", "")
    text = re.sub(r"\\{2,}", " ", text)
    text = re.sub(
        r"\\(?:bibfnamefont|bibnamefont|bibtitlefont|emph|ensuremath|mbox|operatorname|path|"
        r"text|textit|textbf|texttt|textnormal|textrm|textsf|mathrm|mathit|"
        r"mathbf|url|JournalTitle)\s*\{",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\\[a-zA-Z]+\{([a-zA-Z])\}", lambda m: m.group(1), text)
    text = re.sub(r"\\([A-Za-z@]+)(?![A-Za-z@])", r"\1", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace("~", " ")
    return unicodedata.normalize("NFC", text)


def _clean_field_value(val: str) -> str:
    """Clean field value: remove comments, unwrap braces/quotes, normalize whitespace, clean accents."""
    if not val:
        return ""
    val = val.strip()
    if (val.startswith("{") and val.endswith("}")) or (val.startswith('"') and val.endswith('"')):
        val = val[1:-1].strip()

    cleaned_lines = []
    for line in val.splitlines():
        p = -1
        i = 0
        in_quote = False
        escape = False
        while i < len(line):
            c = line[i]
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_quote = not in_quote
            elif c == '%' and not in_quote and (i == 0 or line[i-1] != '\\'):
                p = i
                break
            i += 1
        if p != -1:
            before = line[:p]
            after = line[p:]
            close_idx = after.find('}')
            if close_idx != -1:
                line = before + after[close_idx:]
            else:
                line = before
        cleaned_lines.append(line)
    val = "\n".join(cleaned_lines)
    if (val.startswith("{") and val.endswith("}")) or (val.startswith('"') and val.endswith('"')):
        val = val[1:-1].strip()
    val = _clean_latex_accents(val)
    val = re.sub(r"\s+", " ", val).strip()
    return val


def _clean_title_value(value: str) -> str:
    """Clean a title without retaining text leaked from malformed fields."""

    title = _clean_field_value(value)
    next_entry = _BIB_ENTRY_START_PATTERN.search(title)
    if next_entry is not None:
        title = title[:next_entry.start()]
    leaked_field = _TITLE_FIELD_LEAK_PATTERN.search(title)
    if leaked_field is not None:
        title = title[:leaked_field.start()]
    title = title.strip(" \t\r\n,;:'\"")
    title = re.sub(
        r"\s*:?[;,.]?\s*(?:see|available\s+(?:at|from))\s+https?://\S+.*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).rstrip(" \t\r\n,;:.")
    title = re.sub(
        r"\s*(?:https?://|doi\s*:\s*)\S+.*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).rstrip(" \t\r\n,;:.")
    title = re.sub(
        r"\s*(?:\.\s*)?arXiv\s+Preprint\s*\(?\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).rstrip(" \t\r\n,;:.")
    if re.fullmatch(r"(?:https?://|doi\s*:).+", title, flags=re.IGNORECASE):
        return ""
    return title


def _unwrap_outer_group(value: str) -> str:
    """Remove one outer BibTeX brace/quote pair without touching nested braces."""
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].strip()
    if not (value.startswith("{") and value.endswith("}")):
        return value

    depth = 0
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return value
    return value[1:-1].strip() if depth == 0 else value


def _remove_field_comments(value: str) -> str:
    """Remove BibTeX line comments while preserving escaped percent signs."""
    lines: list[str] = []
    for line in value.splitlines():
        escaped = False
        for index, char in enumerate(line):
            if char == "%" and not escaped:
                line = line[:index]
                break
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append(line)
    return " ".join(lines)


def _extract_arxiv_id(fields: Mapping[str, str]) -> str:
    """Extract an arXiv ID from eprint or an arXiv URL."""
    eprint = fields.get("eprint", "")
    if eprint:
        arxiv_id = _normalize_arxiv_id(eprint)
        if arxiv_id:
            return arxiv_id

    archiveprefix = _clean_field_value(fields.get("archiveprefix", ""))
    url = _clean_field_value(fields.get("url", ""))
    if "arxiv" in archiveprefix.lower() or "arxiv.org" in url.lower():
        return _normalize_arxiv_id(url)
    return ""


def _parse_authors(author_str: str) -> Tuple[str, ...]:
    """Parse author string separated by 'and', '\\and', ';', or commas into a tuple of author names."""
    if not author_str:
        return ()
    # Do not call _clean_field_value here: it removes all braces before the
    # splitter can use them to protect corporate names containing "and".
    cleaned = _remove_field_comments(author_str)
    cleaned = _unwrap_outer_group(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\\and\b", "and", cleaned, flags=re.IGNORECASE)

    parts = []
    current = []
    brace_depth = 0
    in_quote = False
    escape = False
    i = 0
    n = len(cleaned)
    while i < n:
        c = cleaned[i]
        if escape:
            current.append(c)
            escape = False
        elif c == '\\':
            escape = True
            current.append(c)
        elif c == '"' and not escape:
            in_quote = not in_quote
            current.append(c)
        elif c == '{' and not in_quote:
            brace_depth += 1
            current.append(c)
        elif c == '}' and not in_quote:
            if brace_depth > 0:
                brace_depth -= 1
            current.append(c)
        elif c == ';' and brace_depth == 0 and not in_quote:
            parts.append("".join(current))
            current = []
        elif brace_depth == 0 and not in_quote and cleaned[i:i+3].lower() == 'and':
            is_start = (i == 0 or not cleaned[i-1].isalnum())
            is_end = (i + 3 >= n or not cleaned[i+3].isalnum())
            if is_start and is_end:
                parts.append("".join(current))
                current = []
                i += 2
            else:
                current.append(c)
        else:
            current.append(c)
        i += 1
    if current:
        parts.append("".join(current))

    authors = []
    for p in parts:
        p = p.strip()
        if (p.startswith("{") and p.endswith("}")) or (p.startswith('"') and p.endswith('"')):
            p = p[1:-1].strip()
        p = _clean_latex_accents(p)
        p = re.sub(r"\s+", " ", p).strip()
        if p and p.lower() != "others":
            authors.append(p)
    return tuple(authors)


def _extract_url(field_dict: Dict[str, str]) -> str:
    """Extract URL from url field, howpublished, note, eprint, archiveprefix, \\url{}, \\href{}."""
    url = field_dict.get("url", "")
    if not url:
        for key in ("howpublished", "note", "eprint", "archiveprefix", "doi"):
            text = field_dict.get(key, "")
            if key == "doi" and text and not text.startswith("http"):
                normalized_doi = _normalize_doi(text)
                if normalized_doi:
                    url = f"https://doi.org/{normalized_doi}"
                    break
            href_match = re.search(r"\\href\{([^}]+)\}\{([^}]+)\}", text)
            if href_match:
                url = href_match.group(1)
                break
            url_match = re.search(r"\\url\{([^}]+)\}", text)
            if url_match:
                url = url_match.group(1)
                break
            http_match = re.search(r"(https?://[^\s{}]+)", text)
            if http_match:
                url = http_match.group(1)
                break
    url = _clean_field_value(url)
    if not url:
        arxiv_id = _extract_arxiv_id(field_dict)
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
    return url


def _parse_bib_fields(fields_text: str) -> Dict[str, str]:
    """Parse field_name = value pairs from fields text by splitting at comma level 0."""
    fields = {}
    parts = []
    current = []
    brace_depth = 0
    in_quote = False
    escape = False
    i = 0
    n = len(fields_text)
    while i < n:
        c = fields_text[i]
        if escape:
            current.append(c)
            escape = False
        elif c == '\\':
            escape = True
            current.append(c)
        elif c == '"' and brace_depth == 0 and not escape:
            in_quote = not in_quote
            current.append(c)
        elif c == '{' and not in_quote:
            brace_depth += 1
            current.append(c)
        elif c == '}' and not in_quote:
            if brace_depth > 0:
                brace_depth -= 1
            current.append(c)
        elif c == ',' and brace_depth == 0 and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        parts.append("".join(current))

    for part in parts:
        part = part.strip()
        if not part:
            continue
        eq_idx = part.find('=')
        if eq_idx != -1:
            f_name = part[:eq_idx].strip().lower()
            f_val = part[eq_idx+1:].strip()
            if (
                f_name == "title"
                and f_name in fields
                and not f_val.strip("{}\"' ")
            ):
                continue
            fields[f_name] = f_val
    return fields


def title_status_for_entry(entry: BibEntry) -> tuple[str, str]:
    """Classify a title and whether other metadata remains available.

    ``missing_parser`` is deliberately conservative: it is used only when the
    raw source contains an explicit, non-empty title field/wrapper that the
    normalized entry failed to publish.  All other empty titles are marked
    ``missing_source`` rather than guessing from prose.  The second value
    separates the two downstream cases (metadata available vs. no metadata),
    giving M3 an unambiguous four-way classification for missing titles.
    """

    metadata_status = (
        "metadata_available"
        if (
            entry.authors
            or entry.year is not None
            or entry.doi
            or entry.journal
            or entry.url
        )
        else "metadata_empty"
    )
    if entry.title.strip():
        return "present", metadata_status

    raw = entry.raw_entry or ""
    has_explicit_title = False
    if entry.source_kind == "bibtex" and raw:
        _, separator, fields_text = raw.partition(",")
        if separator:
            fields = _recover_bib_fields(
                fields_text,
                _parse_bib_fields(fields_text),
            )
            has_explicit_title = bool(_clean_title_value(fields.get("title", "")))
    else:
        for match in re.finditer(
            r"\\bibinfo\s*\{\s*title\s*\}",
            raw,
            flags=re.IGNORECASE,
        ):
            group = _read_braced_group(raw, match.end())
            if group is not None and _clean_title_value(group[0]):
                has_explicit_title = True
                break

    return (
        "missing_parser" if has_explicit_title else "missing_source",
        metadata_status,
    )


def _recover_bib_fields(fields_text: str, fields: Dict[str, str]) -> Dict[str, str]:
    """Recover explicit assignments after a malformed quote or brace.

    This fallback is intentionally narrow: it runs only when the normal
    top-level parser found at most one field or when the parsed title visibly
    contains another assignment. Values are bounded by the next recognized
    field name, so a damaged title cannot consume the rest of the entry.
    """

    cleaned_title = _clean_field_value(fields.get("title", ""))
    if len(fields) > 1 and not _TITLE_FIELD_LEAK_PATTERN.search(cleaned_title):
        return fields

    matches = list(_BIB_FIELD_ASSIGNMENT_PATTERN.finditer(fields_text))
    if not matches:
        return fields

    recovered = dict(fields)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(fields_text)
        value = fields_text[match.end():end].rstrip(" \t\r\n,")
        name = match.group("name").lower()
        if value and (name not in recovered or name == "title"):
            recovered[name] = value
    return recovered


def _parse_bib_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Parse BibTeX format content into BibEntry objects robustly."""
    entries: List[BibEntry] = []

    i = 0
    n = len(content)
    while i < n:
        if content[i] == '@':
            entry_start = i
            start_type = i + 1
            j = start_type
            while j < n and (content[j].isalnum() or content[j] in '_-'):
                j += 1
            entry_type = content[start_type:j].lower()
            if entry_type in ('string', 'comment', 'preamble'):
                i = j
                continue

            while j < n and content[j] in ' \t\n\r':
                j += 1
            if j < n and content[j] in '{(':
                open_char = content[j]
                close_char = '}' if open_char == '{' else ')'
                brace_depth = 1
                body_start = j + 1
                j += 1
                while j < n and brace_depth > 0:
                    if content[j] == open_char and (j == 0 or content[j-1] != '\\'):
                        brace_depth += 1
                    elif content[j] == close_char and (j == 0 or content[j-1] != '\\'):
                        brace_depth -= 1
                    j += 1

                entry_closed = brace_depth == 0
                if entry_closed:
                    body_end = j - 1
                    entry_end = j
                else:
                    next_entry = _BIB_ENTRY_START_PATTERN.search(content, body_start)
                    body_end = next_entry.start() if next_entry is not None else n
                    entry_end = body_end
                    j = body_end
                entry_body = content[body_start:body_end]
                raw_entry = content[entry_start:entry_end].strip()
                i = j

                comma_idx = entry_body.find(',')
                if comma_idx != -1:
                    raw_citation_key = entry_body[:comma_idx].strip()
                    raw_citation_key = re.sub(r"[\{\}\"']", "", raw_citation_key).strip()
                    fields_text = entry_body[comma_idx+1:]
                else:
                    raw_citation_key = entry_body.strip()
                    fields_text = ""

                # BibTeX permits comments after the citation key and between
                # fields (for example ``{key, % comment\n title = {...}}``).
                # Strip them before field-name detection; otherwise the
                # comment text becomes part of the first field name and the
                # title silently disappears from the structured output.
                fields_text = _strip_bibtex_comments(fields_text)

                citation_key = (
                    _safe_citation_key(
                        raw_citation_key,
                        file_path=file_path,
                        sequence=len(entries),
                        diagnostics=diagnostics,
                    )
                    if raw_citation_key
                    else ""
                )

                field_dict = _recover_bib_fields(
                    fields_text,
                    _parse_bib_fields(fields_text),
                )

                title = _clean_title_value(field_dict.get("title", ""))
                author_raw = field_dict.get("author", "") or field_dict.get("authors", "")
                authors = _parse_authors(author_raw)

                year = _year_from_text(
                    field_dict.get("year", ""),
                    file_path=file_path,
                    diagnostics=diagnostics,
                )

                doi = _normalize_doi(field_dict.get("doi", ""))
                journal = _clean_field_value(
                    field_dict.get("journal", "")
                    or field_dict.get("booktitle", "")
                    or field_dict.get("journaltitle", "")
                )
                url = _extract_url(field_dict)
                parse_status = "partial" if not entry_closed else "parsed"

                note = field_dict.get("note", "")
                note_has_structure = bool(
                    re.search(
                        r"\\(?:textit|emph|href|url)(?![A-Za-z@])|"
                        r"\{\s*\\(?:it|em)\s+|"
                        r"\bdoi\s*:\s*|\b10\.\d{4,9}/",
                        note,
                        re.IGNORECASE,
                    )
                )
                if (
                    note
                    and note_has_structure
                    and not any((title, authors, year, doi, journal, url))
                ):
                    # Keep formatting commands such as ``\textit{...}`` until
                    # the rendered-citation parser has identified the title.
                    # Cleaning first removes the emphasis boundary and makes a
                    # comma inside the title look like a field separator.
                    note_text = note
                    while True:
                        unwrapped = _unwrap_outer_group(note_text)
                        if unwrapped == note_text:
                            break
                        note_text = unwrapped
                    note_metadata = _parse_structured_bibitem(note_text)
                    if note_metadata is not None:
                        title = note_metadata.get("title", "") or title
                        authors = note_metadata.get("authors", ()) or authors
                        year = note_metadata.get("year") or year
                        doi = note_metadata.get("doi", "") or doi
                        journal = note_metadata.get("journal", "") or journal
                        url = note_metadata.get("url", "") or url
                        parse_status = (
                            "parsed"
                            if title
                            else "partial"
                        )

                # A syntactically valid entry can legitimately omit the
                # subset of fields represented by our flat output schema. For
                # example, an @misc entry may contain only howpublished or a
                # custom field. Treating that as malformed produced a large
                # number of false BIBTEX_PARTIAL warnings. Only report actual
                # structural/key failures here; missing optional output fields
                # remain represented by the contract's empty values.
                if not entry_closed or not citation_key:
                    diagnostics.append(
                        Diagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            code="BIBTEX_PARTIAL",
                            message=(
                                "BibTeX entry is not closed"
                                if not entry_closed
                                else "Could not fully parse bibliography entry"
                            ),
                            file_path=file_path,
                        )
                    )

                if citation_key:
                    entries.append(
                        BibEntry(
                            citation_key=citation_key,
                            entry_type=entry_type or "misc",
                            raw_citation_key=raw_citation_key,
                            raw_entry=raw_entry,
                            source_kind="bibtex",
                            parse_status=parse_status,
                            title=title,
                            authors=authors,
                            year=year,
                            doi=doi,
                            journal=journal,
                            url=url,
                        )
                    )
            else:
                diagnostics.append(
                    Diagnostic(
                        severity=DiagnosticSeverity.WARNING,
                        code="BIBTEX_PARTIAL",
                        message="BibTeX entry has no opening delimiter",
                        file_path=file_path,
                    )
                )
                i = j
        else:
            i += 1
    return entries


def _parse_rendered_authors(value: str) -> Tuple[str, ...]:
    """Parse the author block emitted by common BibTeX bibliography styles."""

    cleaned = _clean_field_value(value).strip()
    if not re.search(
        r"(?:^|[\s,])(?:[A-Z](?:-[A-Z])?\.){1,3}$",
        cleaned,
    ):
        cleaned = cleaned.rstrip(".")
    if not cleaned:
        return ()

    groups = re.split(
        r"\s+and\s+|,\s+and\s+|;\s*",
        cleaned,
        flags=re.IGNORECASE,
    )
    authors: list[str] = []
    for group in groups:
        group = re.sub(r"^\s*,?\s*and\s+", "", group, flags=re.IGNORECASE)
        comma_parts = [part.strip() for part in group.split(",") if part.strip()]
        surname_first = (
            len(comma_parts) >= 2
            and len(comma_parts) % 2 == 0
            and all(
                "." not in comma_parts[index]
                and not re.search(r"\d|https?://", comma_parts[index])
                and re.match(r"^[A-ZÀ-ÖØ-Þ]", comma_parts[index])
                and re.fullmatch(
                    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]+)*",
                    comma_parts[index],
                )
                and re.fullmatch(
                    r"(?:[A-ZÀ-ÖØ-Þ](?:[-.][A-ZÀ-ÖØ-Þ])?\.?|"
                    r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+)(?:\s+.*)?",
                    comma_parts[index + 1],
                )
                for index in range(0, len(comma_parts), 2)
            )
        )
        if surname_first:
            authors.extend(
                f"{comma_parts[index]}, {comma_parts[index + 1]}"
                for index in range(0, len(comma_parts), 2)
            )
        elif len(comma_parts) > 1 and all(
            re.match(
                r"^(?:[A-ZÀ-ÖØ-ÞŠŽČ](?:\.-?[A-ZÀ-ÖØ-ÞŠŽČ])?\.|"
                r"[A-ZÀ-ÖØ-ÞŠŽČ][a-zà-öø-ÿšžč]+\b)",
                part,
            )
            for part in comma_parts
        ):
            authors.extend(comma_parts)
        elif group.strip(" ,;"):
            authors.append(group.strip(" ,;"))
    return tuple(authors)


def _looks_like_author_segment(value: str) -> bool:
    """Recognize an author segment without mistaking an abbreviated venue for it."""

    cleaned = _clean_field_value(value).strip(" \t\r\n,;")
    cleaned = re.sub(r"^(?:and|&)\s+", "", cleaned, flags=re.IGNORECASE)
    if ":" in cleaned:
        return False
    if re.search(r"[$=<>×±]", cleaned):
        return False
    tokens = cleaned.split()
    if len(tokens) < 2:
        return False

    parsed_authors = _parse_rendered_authors(cleaned)
    if len(parsed_authors) >= 2 and _rendered_authors_are_plausible(parsed_authors):
        return True

    initial = re.compile(
        r"(?:[A-ZÀ-ÖØ-Þ](?:\.-?[A-Za-zÀ-ÖØ-öø-ÿ])?(?:[A-Za-zÀ-ÖØ-öø-ÿ])?\.){1,3}"
    )
    connector = re.compile(
        r"(?:de|del|der|di|dos?|du|la|le|van|von|bin|al)",
        re.IGNORECASE,
    )
    if not tokens[0].endswith(".") and all(
        initial.fullmatch(token) for token in tokens[1:]
    ):
        return True
    if initial.fullmatch(tokens[0]) and not tokens[-1].endswith("."):
        return True
    for split in range(1, len(tokens)):
        if all(initial.fullmatch(token) for token in tokens[split:]) and all(
            not token.endswith(".") for token in tokens[:split]
        ):
            return True
    if not tokens[-1].endswith(".") and all(
        initial.fullmatch(token) or connector.fullmatch(token)
        for token in tokens[:-1]
    ):
        return True
    return False


def _looks_like_venue_fragment(value: str) -> bool:
    """Identify a journal fragment using citation punctuation, not title guesses."""

    cleaned = _clean_field_value(value).strip(" \t\r\n,;")
    if not cleaned:
        return False
    if re.search(
        r"\b(?:journal|section|sect\.?|volume|vol\.?|proceedings|conference|"
        r"physical\s+review|nature|science|cell)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    abbreviations = [
        token
        for token in re.findall(r"\b[A-Z][A-Za-z]{1,5}\.", cleaned)
        if token.rstrip(".") not in {"I", "II", "III", "IV", "V", "VI"}
    ]
    return len(abbreviations) >= 1


def _rendered_authors_are_plausible(authors: Tuple[str, ...]) -> bool:
    """Reject prose fragments accidentally captured as rendered author names."""

    if not authors:
        return False
    for author in authors:
        if re.search(r"\d", author):
            return False
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", author)
        if len(words) < 2 or len(words) > 6 or author.count(",") > 1:
            return False
        if any(
            word.lower()
            in {
                "and",
                "for",
                "from",
                "in",
                "into",
                "of",
                "on",
                "the",
                "theory",
                "through",
                "with",
            }
            for word in words
        ):
            return False
        if not any(char.isupper() for char in author):
            return False
        if not any(
            len(word) > 1 and any(char.islower() for char in word)
            for word in words
        ):
            return False
    return True


def _extract_rendered_venue(value: str) -> str:
    """Return an explicitly formatted venue from a rendered bibliography block."""

    for pattern in (
        r"\{\s*\\(?:em|it)\s+([^{}]+)\}",
        r"\\(?:emph|textit)\s*\{([^{}]+)\}",
        r"\\JournalTitle\s*\{([^{}]+)\}",
    ):
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match is not None:
            return _clean_field_value(match.group(1)).strip(" \t\r\n,;")
    return ""


def _extract_href_parts(value: str) -> tuple[str, str] | None:
    """Read a LaTeX href with balanced URL and visible-label arguments."""

    match = re.search(r"\\href(?:@noop)?(?![A-Za-z@])", value)
    if match is None:
        return None
    url_group = _read_braced_group(value, match.end())
    label_group = (
        _read_braced_group(value, url_group[1])
        if url_group is not None
        else None
    )
    if url_group is None or label_group is None:
        return None
    return (
        _clean_field_value(url_group[0]),
        _clean_field_value(label_group[0]),
    )


def _extract_doi_wrapper(value: str) -> tuple[str, str] | None:
    r"""Extract DOI and visible citation text from ``\doi``/``\Doi`` wrappers."""

    for match in re.finditer(
        r"\\doi(?![A-Za-z@])", value, re.IGNORECASE
    ):
        doi_group = _read_braced_group(value, match.end())
        if doi_group is None:
            continue
        label_group = _read_braced_group(value, doi_group[1])
        doi = _normalize_doi(doi_group[0])
        if doi and label_group is not None:
            return doi, label_group[0]
    return None


def _parse_doi_wrapper_bibitem(item_text: str) -> dict[str, Any] | None:
    """Parse a DOI wrapper whose visible label contains journal metadata."""

    match = re.search(r"\\doi(?![A-Za-z@])", item_text, re.IGNORECASE)
    wrapper = _extract_doi_wrapper(item_text)
    if match is None or wrapper is None:
        return None

    doi, visible_label = wrapper
    volume_marker = re.search(
        r"\\textbf\s*\{\s*(?:[A-Z]\s*)?\d|"
        r"\{\s*\\bf\s*(?=(?:[A-Z]\s*)?\d)",
        visible_label,
        re.IGNORECASE,
    )
    if volume_marker is None:
        return None

    authors = _parse_rendered_authors(
        item_text[:match.start()].rstrip(" \t\r\n,;")
    )
    if not _rendered_authors_are_plausible(authors):
        return None

    journal = _clean_field_value(
        visible_label[:volume_marker.start()]
    ).strip(" \t\r\n,;")
    if not journal:
        return None
    return {
        "authors": authors,
        "title": "",
        "year": _last_plausible_year(visible_label),
        "doi": doi,
        "journal": journal,
        "url": f"https://doi.org/{doi}",
        "parse_status": "partial",
        "_title_absent": True,
    }


def _parse_href_venue_bibitem(item_text: str) -> dict[str, Any] | None:
    """Extract metadata when an href label is a venue, not an article title."""

    href_match = re.search(r"\\href(?:@noop)?(?![A-Za-z@])", item_text)
    if href_match is None:
        return None

    url_group = _read_braced_group(item_text, href_match.end())
    label_group = (
        _read_braced_group(item_text, url_group[1])
        if url_group is not None
        else None
    )
    if url_group is None or label_group is None:
        return None
    raw_label = label_group[0]
    href_parts = (
        _clean_field_value(url_group[0]),
        _clean_field_value(raw_label),
    )
    if not re.search(
        r"\\(?:em|it|emph|textit|textbf|bf)(?![A-Za-z@])|"
        r"\\JournalTitle(?![A-Za-z@])|"
        r"\bdoi\b|"
        r"\b(?:19|20)\d{2}\b|\b\d+\s*[,(:]",
        raw_label,
        re.IGNORECASE,
    ):
        return None

    journal = _extract_rendered_venue(raw_label)
    if not journal:
        volume_marker = re.search(
            r"\\textbf\s*\{\s*(?:[A-Z]\s*)?\d|"
            r"\{\s*\\bf\s*(?=(?:[A-Z]\s*)?\d)",
            raw_label,
            re.IGNORECASE,
        )
        if volume_marker is not None:
            journal = _clean_field_value(
                raw_label[:volume_marker.start()]
            ).strip(" \t\r\n,;")
        if not journal:
            volume_marker = re.search(r"\b\d+\s*[,(:]", raw_label)
            if volume_marker is not None:
                journal = _clean_field_value(
                    raw_label[:volume_marker.start()]
                ).strip(" \t\r\n,;")
    author_text = re.sub(
        r"\\newblock(?![A-Za-z@])",
        " ",
        item_text[:href_match.start()],
        flags=re.IGNORECASE,
    ).rstrip(" \t\r\n,;")
    author_year = _last_plausible_year(author_text)
    if author_year is not None:
        author_text = re.sub(
            rf"(?:[,;(]\s*)?\b{author_year}\b[.)]?\s*$",
            "",
            author_text,
        ).rstrip(" \t\r\n,;")
    authors = _parse_rendered_authors(author_text)
    if re.search(r"(?:https?://|urlprefix|\\url)", journal, re.IGNORECASE):
        journal = ""

    url, label = href_parts
    # Some bibliography styles put a real article title before a DOI href,
    # without an explicit delimiter, e.g. ``Authors 2022 Title J. Field
    # Robot. 39 ... \href{doi}{doi:...}``.  The generic href parser above
    # intentionally treats the label as venue metadata and would otherwise
    # discard that title.  When a publication year separates a plausible
    # author block from a title and an abbreviated venue precedes a volume,
    # recover the structured citation explicitly.
    if re.search(r"\bdoi\b", label, re.IGNORECASE):
        prefix = item_text[:href_match.start()].rstrip(" \t\r\n,;")
        year_match = re.search(r"\b((?:19|20)\d{2})\b", prefix)
        if year_match is not None:
            prefix_authors = _parse_rendered_authors(
                prefix[:year_match.start()].rstrip(" \t\r\n,;.")
            )
            publication = prefix[year_match.end():].strip(" \t\r\n,;.")
            volume_matches = list(
                re.finditer(r"\s+\d+(?:\s|\(|,)", publication)
            )
            volume_match = next(
                (
                    match
                    for match in volume_matches
                    if match.group().strip(" ,(") != year_match.group(1)
                ),
                volume_matches[0] if volume_matches else None,
            )
            if (
                publication
                and volume_match is not None
                and _rendered_authors_are_plausible(prefix_authors)
            ):
                before_volume = publication[:volume_match.start()].rstrip(
                    " \t\r\n,;"
                )
                venue_match = re.search(
                    r"(?P<venue>(?:[A-Z][A-Za-z]{0,12}\.?\s+){1,5}"
                    r"[A-Z][A-Za-z]{0,12}\.)$",
                    before_volume,
                )
                venue_start = venue_match.start() if venue_match is not None else None
                if venue_start is None:
                    proceedings = re.search(r"\bProc\.", before_volume)
                    if proceedings is not None:
                        venue_start = proceedings.start()
                if venue_start is None:
                    one_word_venue = re.search(
                        r"\b[A-Z][A-Za-z]{2,}\s*$",
                        before_volume,
                    )
                    if one_word_venue is not None:
                        venue_start = one_word_venue.start()
                if venue_start is not None:
                    recovered_title = _clean_title_value(
                        before_volume[:venue_start]
                    ).strip(" \t\r\n,;:." )
                    recovered_journal = _clean_field_value(
                        before_volume[venue_start:]
                    ).strip(" \t\r\n,;:.")
                    recovered_journal = re.sub(
                        r"\s+vol\.?\s*$",
                        "",
                        recovered_journal,
                        flags=re.IGNORECASE,
                    )
                    if (
                        len(recovered_title.split()) >= 3
                        and recovered_journal
                        and not _looks_like_venue_fragment(recovered_title)
                    ):
                        doi = _normalize_doi(url)
                        return {
                            "authors": prefix_authors,
                            "title": recovered_title,
                            "year": int(year_match.group(1)),
                            "doi": doi,
                            "journal": recovered_journal,
                            "url": f"https://doi.org/{doi}" if doi else url,
                            "parse_status": "parsed",
                        }
    if not _rendered_authors_are_plausible(authors):
        return None
    if not journal:
        return None
    return {
        "authors": authors,
        "title": "",
        "year": author_year or _last_plausible_year(label),
        "doi": _normalize_doi(url),
        "journal": journal,
        "url": (
            f"https://doi.org/{_normalize_doi(url)}"
            if _normalize_doi(url)
            else url
        ),
        "parse_status": "partial",
        "_title_absent": True,
    }


def _parse_bibinfo_bibitem(item_text: str) -> dict[str, Any] | None:
    """Extract fields emitted by REVTeX and related ``\bibinfo`` styles."""

    fields: dict[str, list[str]] = {}
    for match in re.finditer(r"\\bibinfo(?![A-Za-z@])", item_text):
        name_group = _read_braced_group(item_text, match.end())
        value_group = (
            _read_braced_group(item_text, name_group[1])
            if name_group is not None
            else None
        )
        if name_group is None or value_group is None:
            continue
        name = name_group[0].strip().lower()
        fields.setdefault(name, []).append(value_group[0])

    if not fields:
        return None

    parsed_authors = [
        author
        for author in (
            _clean_field_value(value).strip(" \t\r\n,;")
            for value in fields.get("author", [])
        )
        if author
    ]
    repaired_authors: list[str] = []
    index = 0
    while index < len(parsed_authors):
        author = parsed_authors[index]
        next_author = (
            parsed_authors[index + 1]
            if index + 1 < len(parsed_authors)
            else ""
        )
        if (
            re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", author)
            and re.fullmatch(r"[A-Z]", next_author)
        ):
            repaired_authors.append(f"{next_author}. {author}")
            index += 2
            continue
        if len(author) > 1:
            repaired_authors.append(author)
        index += 1
    authors = tuple(repaired_authors)
    title = _clean_title_value((fields.get("title") or [""])[0]).rstrip(
        " \t\r\n.,;:"
    )
    journal = _clean_field_value(
        (fields.get("journal") or fields.get("journaltitle") or [""])[0]
    ).strip(" \t\r\n,;")
    year = _year_from_text((fields.get("year") or fields.get("date") or [""])[0])
    publication_suffix = re.search(
        r"\s+\([^()]*,\s*((?:19|20)\d{2})\)$",
        title,
    )
    if publication_suffix is not None:
        year = year or int(publication_suffix.group(1))
        title = title[:publication_suffix.start()].rstrip(" \t\r\n.,;:")
    doi = _normalize_doi((fields.get("doi") or [""])[0])
    url = _extract_url({name: values[0] for name, values in fields.items()})
    return {
        "authors": authors,
        "title": title,
        "year": year,
        "doi": doi,
        "journal": journal,
        "url": url,
        "parse_status": "parsed" if title else "partial",
    }


def _last_plausible_year(value: str) -> Optional[int]:
    """Return the final plausible publication year in a bounded citation tail."""

    cleaned = _clean_field_value(value)
    years = [
        int(match.group(1))
        for match in re.finditer(r"\b(\d{4})\b", cleaned)
        if 1600 <= int(match.group(1)) <= date.today().year + 1
    ]
    if years:
        return years[-1]

    arxiv_match = re.search(
        r"\barxiv\s*:\s*(?:[a-z.-]+/)?(?P<year>\d{2})(?:\d{2})[.\d]",
        cleaned,
        re.IGNORECASE,
    )
    if arxiv_match is None:
        return None
    short_year = int(arxiv_match.group("year"))
    inferred_year = (1900 if short_year >= 91 else 2000) + short_year
    return inferred_year if inferred_year <= date.today().year + 1 else None


def _citation_year(value: str, boundary: int) -> Optional[int]:
    """Read publication year without treating a DOI or page as evidence."""

    prefix_years = re.findall(
        r"\b((?:19|20)\d{2})\b",
        _clean_field_value(value[:boundary]),
    )
    if prefix_years:
        return int(prefix_years[-1])

    tail = re.sub(
        r"(?:https?://|doi\s*:\s*|\\doi\s*\{)[^\s{}]+",
        " ",
        _clean_field_value(value[boundary:]),
        flags=re.IGNORECASE,
    )
    parenthesized = re.search(r"\(\s*((?:19|20)\d{2})\s*\)", tail)
    if parenthesized is not None:
        return int(parenthesized.group(1))
    years = re.findall(r"\b((?:19|20)\d{2})\b", tail)
    return int(years[0]) if years else None


def _parse_quoted_bibitem(item_text: str) -> dict[str, Any] | None:
    """Extract citations whose title is explicitly delimited by quotation marks."""

    title_match = re.search(r"``(?P<title>.*?),?''|“(?P<unicode_title>.*?)”", item_text)
    if title_match is None:
        return None

    raw_title = title_match.group("title") or title_match.group("unicode_title") or ""
    title = _clean_title_value(raw_title).rstrip(" \t\r\n.,;:")
    authors = _parse_rendered_authors(item_text[:title_match.start()].rstrip(" \t\r\n,;"))
    tail = item_text[title_match.end():].lstrip(" \t\r\n,;.")
    publication_tail = re.split(
        r"\[?\s*(?:accessed|retrieved)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    venue_boundary = re.search(
        r"\\(?:textbf|bf)(?![A-Za-z@])|\b(?:19|20)\d{2}\b|"
        r"\b\d+\s*[,(:]|\(\d{4}\)",
        tail,
    )
    venue_source = tail[:venue_boundary.start()] if venue_boundary is not None else ""
    journal = _clean_field_value(venue_source).strip(" \t\r\n,;.")
    if not title or not authors:
        return None
    return {
        "authors": authors,
        "title": title,
        "year": _last_plausible_year(publication_tail),
        "journal": journal,
        "parse_status": "parsed",
    }


def _formatted_bibliography_spans(value: str) -> list[tuple[int, int, str]]:
    """Return explicitly italicized/emphasized spans with source offsets."""

    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\\(?:emph|textit)(?![A-Za-z@])", value):
        group = _read_braced_group(value, match.end())
        if group is not None:
            spans.append((match.start(), group[1], _clean_field_value(group[0])))

    for match in re.finditer(
        r"\{\s*\\(?:em|it)(?![A-Za-z@])\s+",
        value,
    ):
        group = _read_braced_group(value, match.start())
        if group is None:
            continue
        content = re.sub(
            r"^\s*\\(?:em|it)(?![A-Za-z@])\s+",
            "",
            group[0],
            count=1,
        )
        content = re.split(
            r"\\bf(?![A-Za-z@])\s+(?=\d)",
            content,
            maxsplit=1,
        )[0]
        spans.append((match.start(), group[1], _clean_field_value(content)))

    return sorted(set(spans), key=lambda span: (span[0], span[1]))


def _parse_emphasized_bibitem(item_text: str) -> dict[str, Any] | None:
    """Extract citations that explicitly emphasize both title and venue."""

    spans = [
        span
        for span in _formatted_bibliography_spans(item_text)
        if _normalize_text(span[2]) not in {"et al", "and others"}
    ]
    newblock_count = len(re.findall(r"\\newblock(?![A-Za-z@])", item_text))
    if spans:
        first_span = spans[0]
        prefix = item_text[:first_span[0]].rstrip(" \t\r\n,;")
        direct_authors = _parse_rendered_authors(prefix)
        if (
            not _rendered_authors_are_plausible(direct_authors)
            and not re.search(r"(?:https?://|\\href)", prefix, re.IGNORECASE)
        ):
            prefix = re.sub(r"\bIn\s*$", "", prefix).rstrip(" \t\r\n,;")
            for boundary in re.finditer(
                r"\.\s+(?=[A-ZÀ-ÖØ-Þ{\\])",
                prefix,
            ):
                authors = _parse_rendered_authors(
                    prefix[:boundary.start() + 1]
                )
                title = _clean_title_value(prefix[boundary.end():]).rstrip(
                    " \t\r\n.,;:"
                )
                if title and _rendered_authors_are_plausible(authors):
                    return {
                        "authors": authors,
                        "title": title,
                        "year": _last_plausible_year(
                            item_text[first_span[1]:]
                        ),
                        "journal": _clean_field_value(first_span[2]).strip(
                            " \t\r\n,;"
                        ),
                        "parse_status": "parsed",
                    }
    single_emphasized_title = False
    if len(spans) == 1:
        span = spans[0]
        prefix = item_text[:span[0]].rstrip(" \t\r\n,;")
        candidate_authors = _parse_rendered_authors(
            prefix
        )
        formatted_tail = item_text[span[1]:]
        single_emphasized_title = (
            _rendered_authors_are_plausible(candidate_authors)
            and (
                _last_plausible_year(formatted_tail) is not None
                or re.search(
                    r"\\(?:textbf|bf)(?![A-Za-z@])|"
                    r"(?:https?://|\bdoi\s*:|\barxiv\s*:|\bto\s+appear\b)",
                    formatted_tail,
                    re.IGNORECASE,
                )
            )
        )
        if not single_emphasized_title:
            if re.search(r"(?:https?://|\\href)", prefix, re.IGNORECASE):
                return None
            sentence_boundaries = list(
                re.finditer(r"\.\s+(?=[A-ZÀ-ÖØ-Þ{\\])", prefix)
            )
            for boundary in sentence_boundaries:
                authors = _parse_rendered_authors(prefix[:boundary.start() + 1])
                title = _clean_title_value(prefix[boundary.end():]).rstrip(
                    " \t\r\n.,;:"
                )
                if title and _rendered_authors_are_plausible(authors):
                    return {
                        "authors": authors,
                        "title": title,
                        "year": _last_plausible_year(formatted_tail),
                        "journal": _clean_field_value(span[2]).strip(
                            " \t\r\n,;"
                        ),
                        "parse_status": "parsed",
                    }
    if (
        len(spans) < 2
        and not (newblock_count == 1 and spans)
        and not single_emphasized_title
    ):
        return None

    title_span = spans[0]
    title = _clean_title_value(title_span[2]).rstrip(" \t\r\n.,;:")
    authors = _parse_rendered_authors(
        item_text[:title_span[0]].rstrip(" \t\r\n,;")
    )
    if len(spans) > 1:
        journal = _clean_field_value(spans[1][2]).strip(" \t\r\n,;")
    else:
        plain_tail = item_text[title_span[1]:]
        venue_match = re.match(
            r"\s*,\s*([^,()]+),\s*(?:19|20)\d{2}\b",
            plain_tail,
        )
        if venue_match is not None:
            journal = _clean_field_value(venue_match.group(1)).strip(
                " \t\r\n,;"
            )
        else:
            venue_start = re.match(r"\s*[,;]\s*", plain_tail)
            venue_tail = plain_tail[
                venue_start.end() if venue_start is not None else 0:
            ]
            venue_end = re.search(
                r"\\(?:textbf|bf)(?![A-Za-z@])|"
                r"\s+\d+(?:\([^)]*\))?\s*(?=[:(,])",
                venue_tail,
            )
            journal = (
                _clean_field_value(venue_tail[:venue_end.start()]).strip(
                    " \t\r\n,;"
                )
                if venue_end is not None
                else ""
            )
    if not title or not _rendered_authors_are_plausible(authors):
        return None
    return {
        "authors": authors,
        "title": title,
        "year": _last_plausible_year(item_text[title_span[1]:]),
        "journal": journal,
        "parse_status": "parsed",
    }


def _parse_sentence_bibitem(item_text: str) -> dict[str, Any] | None:
    """Parse citations with explicit author, title, and standalone-year sentences."""

    match = re.fullmatch(
        r"(?P<body>.+)\.\s+(?P<year>(?:19|20)\d{2})\.\s*",
        item_text,
        flags=re.DOTALL,
    )
    if match is not None:
        body = match.group("body")
        boundaries = list(re.finditer(r"\.\s+", body))
        if not boundaries:
            return None
        boundary = boundaries[-1]
        authors = _parse_rendered_authors(body[:boundary.start() + 1])
        title = _clean_title_value(body[boundary.end():]).rstrip(" \t\r\n.,;:")
        if not authors or not title:
            return None
        return {
            "authors": authors,
            "title": title,
            "year": int(match.group("year")),
            "journal": "",
            "parse_status": "parsed",
        }

    if re.search(r"(?:https?://|\\href|\\url)", item_text, re.IGNORECASE):
        return None
    year = _last_plausible_year(item_text)

    boundaries = list(
        re.finditer(r"\.\s+(?=[A-ZÀ-ÖØ-Þ{\\])", item_text)
    )
    for author_boundary in boundaries:
        authors = _parse_rendered_authors(
            item_text[:author_boundary.start() + 1]
        )
        if not _rendered_authors_are_plausible(authors):
            continue
        remainder = item_text[author_boundary.end():]
        for title_boundary in re.finditer(
            r"\.\s+(?=[A-ZÀ-ÖØ-Þ{\\])",
            remainder,
        ):
            publication_tail = remainder[title_boundary.end():]
            volume_marker = re.search(
                r"\\(?:textbf|bf)(?![A-Za-z@])|"
                r"\s+\d+(?:\([^)]*\))?\s*(?::|,\s*(?:19|20)\d{2}\b)",
                publication_tail,
            )
            if volume_marker is None:
                continue
            title = _clean_title_value(
                remainder[:title_boundary.start() + 1]
            ).rstrip(" \t\r\n.,;:")
            journal = _clean_field_value(
                publication_tail[:volume_marker.start()]
            ).strip(" \t\r\n,;.")
            if not title or not journal or len(journal.split()) > 12:
                continue
            return {
                "authors": authors,
                "title": title,
                "year": year,
                "journal": journal,
                "parse_status": "parsed",
            }
    return None


def _parse_colon_bibitem(item_text: str) -> dict[str, Any] | None:
    """Parse rendered styles that separate authors and title with a colon."""

    for separator in re.finditer(r":\s+", item_text):
        authors = _parse_rendered_authors(item_text[:separator.start()])
        if not _rendered_authors_are_plausible(authors):
            continue

        remainder = item_text[separator.end():]
        title_boundary = re.search(
            r"\.\s+(?=[A-ZÀ-ÖØ-Þ{\\]|(?i:arxiv)\b)",
            remainder,
        )
        if title_boundary is None:
            continue
        title = _clean_title_value(
            remainder[:title_boundary.start() + 1]
        ).rstrip(" \t\r\n.,;:")
        publication_tail = remainder[title_boundary.end():]
        venue_end = re.search(
            r"\\(?:textbf|bf)(?![A-Za-z@])|\{\\bf\s+|"
            r"\s+\d+(?:\([^)]*\))?\s*(?=[:(,])|"
            r"\s*\(\s*(?:19|20)\d{2}\s*\)|\barxiv\s*:",
            publication_tail,
            re.IGNORECASE,
        )
        journal = (
            _clean_field_value(publication_tail[:venue_end.start()]).strip(
                " \t\r\n,;"
            )
            if venue_end is not None
            else ""
        )
        if not title:
            continue
        return {
            "authors": authors,
            "title": title,
            "year": _last_plausible_year(publication_tail),
            "journal": journal,
            "parse_status": "parsed",
        }
    return None


def _parse_bold_volume_bibitem(item_text: str) -> dict[str, Any] | None:
    """Parse comma styles whose numeric volume is explicitly bold."""

    volume_marker = re.search(
        r"\\textbf\s*\{\s*(?:[A-Z]\s*)?\d|"
        r"\{\s*\\bf\s*(?=(?:[A-Z]\s*)?\d)",
        item_text,
        re.IGNORECASE,
    )
    if volume_marker is None:
        return None

    prefix = item_text[:volume_marker.start()].rstrip(" \t\r\n{")
    segments = [
        segment.strip()
        for segment in prefix.split(",")
        if segment.strip()
    ]
    if len(segments) < 2:
        return None

    authors: list[str] = []
    title_start = 0
    for index, segment in enumerate(segments[:-1]):
        if not _looks_like_author_segment(segment):
            break
        parsed = _parse_rendered_authors(segment)
        if not _rendered_authors_are_plausible(parsed):
            break
        authors.extend(parsed)
        title_start = index + 1

    if not authors or title_start >= len(segments):
        return None
    before_volume = segments[title_start:]
    metadata_text = ", ".join(before_volume).strip(" \t\r\n,;")
    sentence_boundaries = []
    for candidate in re.finditer(
        r"\.\s+(?=[A-ZÀ-ÖØ-Þ])", metadata_text
    ):
        previous_word = re.search(
            r"([A-Za-zÀ-ÖØ-öø-ÿ]+)$", metadata_text[:candidate.start()]
        )
        if previous_word is not None:
            word = previous_word.group(1)
            suffix = metadata_text[candidate.end():]
            if (
                (len(word) >= 4 and word[0].islower())
                or (
                    len(word) >= 4
                    and _looks_like_venue_fragment(suffix.split(",", 1)[0])
                    and not re.match(
                        r"^[A-Z][A-Za-z]{1,5}\.",
                        suffix.strip(),
                    )
                )
            ):
                sentence_boundaries.append(candidate)
    if sentence_boundaries:
        boundary = sentence_boundaries[-1]
        sentence_title = _clean_title_value(
            metadata_text[:boundary.start()]
        ).strip(" \t\r\n,;:.")
        sentence_journal = _clean_field_value(
            metadata_text[boundary.end():]
        ).strip(" \t\r\n,;:.")
        if (
            sentence_title
            and sentence_journal
            and not re.match(
                r"^(?:[IVX]+|Part)\.",
                sentence_journal,
                re.IGNORECASE,
            )
        ):
            doi_wrapper = _extract_doi_wrapper(item_text)
            return {
                "authors": tuple(authors),
                "title": sentence_title,
                "year": _citation_year(item_text, volume_marker.start()),
                "doi": (doi_wrapper[0] if doi_wrapper else "")
                or _normalize_doi(item_text),
                "journal": sentence_journal,
                "parse_status": "parsed",
            }
    has_year_segment = (
        len(before_volume) > 1
        and re.fullmatch(r"(?:19|20)\d{2}", before_volume[-1]) is not None
    )
    metadata_segments = before_volume[:-1] if has_year_segment else before_volume
    if not metadata_segments:
        return None
    title_absent = len(metadata_segments) == 1 or all(
        _looks_like_venue_fragment(segment)
        for segment in metadata_segments
    )
    journal = _clean_field_value(
        ", ".join(metadata_segments) if title_absent else metadata_segments[-1]
    ).strip(" \t\r\n,;")
    if not journal:
        return None
    title = "" if title_absent else _clean_title_value(
        ", ".join(metadata_segments[:-1])
    ).strip(" \t\r\n,;:.")
    doi_wrapper = _extract_doi_wrapper(item_text)
    return {
        "authors": tuple(authors),
        "title": title,
        "year": _citation_year(item_text, volume_marker.start()),
        "doi": (doi_wrapper[0] if doi_wrapper else "")
        or _normalize_doi(item_text),
        "journal": journal,
        "parse_status": "parsed" if title else "partial",
        "_title_absent": title_absent,
    }


def _parse_comma_bibitem(item_text: str) -> dict[str, Any] | None:
    """Parse conventional rendered citations with comma-delimited fields."""

    if re.search(r"\\(?:bibinfo|newblock)(?![A-Za-z@])", item_text):
        return None

    author_end: int | None = None
    authors: Tuple[str, ...] = ()
    for comma in re.finditer(",", item_text):
        if re.search(r"\d", item_text[:comma.start()]):
            break
        segment_start = item_text.rfind(",", 0, comma.start()) + 1
        segment = item_text[segment_start:comma.start()].strip()
        candidate = _parse_rendered_authors(item_text[:comma.start()])
        if (
            _rendered_authors_are_plausible(candidate)
            and not (
                author_end is not None
                and not _looks_like_author_segment(segment)
            )
        ):
            author_end = comma.end()
            authors = candidate
        elif author_end is not None:
            break
    if author_end is None:
        return None

    tail = item_text[author_end:].strip()
    formatted_spans = [
        span
        for span in _formatted_bibliography_spans(tail)
        if _normalize_text(span[2]) not in {"et al", "and others"}
    ]
    journal = ""
    if formatted_spans:
        venue_span = formatted_spans[0]
        venue_start = venue_span[0]
        href_start = tail.rfind(r"\href", 0, venue_start)
        if href_start >= 0:
            venue_start = href_start
        title_source = tail[:venue_start].rstrip(" \t\r\n,;")
        journal = _clean_field_value(venue_span[2]).strip(" \t\r\n,;")
        if re.search(r"(?:https?://|urlprefix|\\url)", journal, re.IGNORECASE):
            journal = ""
    elif volume_marker := re.search(r"\\(?:textbf|bf)(?![A-Za-z@])", tail):
        before_volume = tail[:volume_marker.start()]
        title_end = before_volume.rfind(",")
        if title_end < 0:
            return None
        title_source = before_volume[:title_end]
        journal = _clean_field_value(before_volume[title_end + 1:]).strip(
            " \t\r\n,;"
        )
    else:
        plain_year = re.search(r"\b((?:19|20)\d{2})\b", tail)
        if plain_year is not None:
            plain_prefix = tail[:plain_year.start()].strip(" \t\r\n,;")
            sentence_boundaries = []
            for candidate in re.finditer(
                r"\.\s+(?=[A-ZÀ-ÖØ-Þ])", plain_prefix
            ):
                previous_word = re.search(
                    r"([A-Za-zÀ-ÖØ-öø-ÿ]+)$",
                    plain_prefix[:candidate.start()],
                )
                if previous_word is not None:
                    word = previous_word.group(1)
                    suffix = plain_prefix[candidate.end():]
                    if (
                        (len(word) >= 4 and word[0].islower())
                        or (
                            len(word) >= 4
                            and _looks_like_venue_fragment(
                                suffix.split(",", 1)[0]
                            )
                            and not re.match(
                                r"^[A-Z][A-Za-z]{1,5}\.",
                                suffix.strip(),
                            )
                        )
                    ):
                        sentence_boundaries.append(candidate)
            if sentence_boundaries:
                boundary = sentence_boundaries[-1]
                sentence_title = _clean_title_value(
                    plain_prefix[:boundary.start()]
                ).strip(" \t\r\n,;:.")
                sentence_journal = _clean_field_value(
                    plain_prefix[boundary.end():]
                ).strip(" \t\r\n,;:.()")
                # A one-word suffix such as ``Handbook`` is often part of a
                # book title (``... rare-earth elements. Handbook, ...``),
                # not a journal.  Require a multi-word venue here, except for
                # the common one-word journal names, so we do not truncate a
                # valid title while still handling ``Title. Nature, ...``.
                venue_is_unambiguous = (
                    len(sentence_journal.split()) >= 2
                    or sentence_journal in {"Science", "Nature", "Cell"}
                ) and not sentence_journal.lower().startswith("handbook")
                venue_is_unambiguous = venue_is_unambiguous and not re.match(
                    r"^(?:[IVX]+|Part)\.",
                    sentence_journal,
                    re.IGNORECASE,
                )
                if sentence_title and sentence_journal and venue_is_unambiguous:
                    return {
                        "authors": authors,
                        "title": sentence_title,
                        "year": int(plain_year.group(1)),
                        "journal": sentence_journal,
                        "parse_status": "parsed",
                    }
            plain_parts = [
                part.strip()
                for part in plain_prefix.split(",")
                if part.strip()
            ]
            numeric_part = next(
                (
                    index
                    for index, part in enumerate(plain_parts)
                    if re.search(r"\b\d+\b|\(\s*\d", part)
                ),
                None,
            )
            if (
                numeric_part is not None
                and numeric_part >= 2
                and not re.search(
                    r"\bvol\.?",
                    plain_parts[numeric_part],
                    re.IGNORECASE,
                )
            ):
                candidate_title = _clean_title_value(
                    ", ".join(plain_parts[:numeric_part])
                ).strip(" \t\r\n,;:.")
                journal = _clean_field_value(
                    ", ".join(plain_parts[numeric_part:])
                ).strip(" \t\r\n,;.")
                if len(candidate_title.split()) >= 3:
                    return {
                        "authors": authors,
                        "title": candidate_title,
                        "year": int(plain_year.group(1)),
                        "journal": journal,
                        "parse_status": "parsed",
                    }
            candidate_title = (
                _clean_title_value(plain_parts[0]).strip(" \t\r\n,;:.")
                if plain_parts
                else ""
            )
            if (
                (
                    len(candidate_title.split()) >= 3
                    or (
                        len(candidate_title.split()) == 2
                        and len(plain_parts) >= 2
                        and not _looks_like_author_segment(candidate_title)
                    )
                )
                and not re.search(r"\bet\s+al\b", plain_prefix, re.IGNORECASE)
                and not _looks_like_venue_fragment(candidate_title)
                and not _looks_like_author_segment(candidate_title)
            ):
                journal = _clean_field_value(
                    ", ".join(plain_parts[1:])
                ).strip(" \t\r\n,;.")
                return {
                    "authors": authors,
                    "title": candidate_title,
                    "year": int(plain_year.group(1)),
                    "journal": journal,
                    "parse_status": "parsed",
                }
            journal = _clean_field_value(
                tail[plain_year.end():].lstrip(" \t\r\n,;").split(",", 1)[0]
            ).strip(" \t\r\n,;.")
            if re.search(
                r"(?:https?://|urlprefix|\\url)", journal, re.IGNORECASE
            ):
                journal = ""
            return {
                "authors": authors,
                "title": "",
                "year": int(plain_year.group(1)),
                "journal": journal,
                "parse_status": "partial",
                "_title_absent": True,
            }
        title_end = tail.find(",")
        if title_end < 0:
            return None
        title_source = tail[:title_end]
        publication_tail = tail[title_end + 1:]
        venue_end = re.search(
            r"\b(?:19|20)\d{2}\b|https?://|\\(?:url|href)|\bdoi\s*:|\barxiv\s*:",
            publication_tail,
            re.IGNORECASE,
        )
        if venue_end is not None:
            journal = _clean_field_value(
                publication_tail[:venue_end.start()]
            ).strip(" \t\r\n,;.")

    title = _clean_title_value(title_source).strip(" `\t\r\n.,;:'\"")
    if not title or re.fullmatch(r"(?:19|20)\d{2}", title):
        return None
    publication_tail = re.split(
        r"\[?\s*(?:accessed|retrieved)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return {
        "authors": authors,
        "title": title,
        "year": _last_plausible_year(publication_tail),
        "journal": journal,
        "parse_status": "parsed",
    }


def _parse_newblock_bibitem(item_text: str) -> dict[str, Any] | None:
    """Extract stable fields from the block structure emitted by BibTeX styles."""

    blocks = [
        block.strip()
        for block in re.split(r"\\newblock(?![A-Za-z@])", item_text)
        if block.strip()
    ]
    if len(blocks) < 2:
        return None

    doi_block = _extract_href_parts(blocks[1])
    if doi_block is not None:
        doi_url, doi_label = doi_block
        doi = _normalize_doi(doi_url) or _normalize_doi(doi_label)
        doi_label_clean = _clean_field_value(doi_label)
        if doi and (
            re.search(r"(?:path)?doi\s*:", doi_label_clean, re.IGNORECASE)
            or re.search(r"https?://doi\.org/10\.", doi_label_clean, re.IGNORECASE)
        ):
            doi_url = f"https://doi.org/{doi}"
            candidates = [
                candidate
                for parser in (
                    _parse_bold_volume_bibitem,
                    _parse_comma_bibitem,
                    _parse_colon_bibitem,
                    _parse_sentence_bibitem,
                )
                if (candidate := parser(blocks[0])) is not None
            ]
            candidate = next(
                iter(
                    sorted(
                        candidates,
                        key=_structured_candidate_score,
                        reverse=True,
                    )
                ),
                None,
            )
            if candidate is not None and candidate.get("title"):
                return {
                    **candidate,
                    "doi": doi,
                    "url": doi_url,
                    "parse_status": "parsed",
                }
            author_prefix = blocks[0].split(",", 1)[0]
            authors = _parse_rendered_authors(author_prefix)
            return {
                "authors": authors,
                "title": "",
                "year": _last_plausible_year(blocks[0]),
                "doi": doi,
                "url": doi_url,
                "journal": "",
                "parse_status": "partial",
                "_title_absent": True,
            }

    author_year = _last_plausible_year(blocks[0])
    author_block = blocks[0]
    if author_year is not None:
        author_block = re.sub(
            rf"(?:[,;(]\s*)?\b{author_year}\b[.)]?\s*$",
            "",
            author_block,
        ).rstrip(" \t\r\n,;")
    authors = _parse_rendered_authors(author_block)
    title_index = 1
    explicit_year: Optional[int] = None
    cleaned_second_block = _clean_field_value(blocks[1]).strip(" \t\r\n.,;()")
    if re.fullmatch(r"(?:19|20)\d{2}", cleaned_second_block):
        explicit_year = int(cleaned_second_block)
        title_index += 1
    if title_index >= len(blocks):
        return None

    title_block = blocks[title_index]
    href_parts = _extract_href_parts(title_block)
    if href_parts is not None:
        href_label = _normalize_text(href_parts[1])
        if href_label in {"doi", "arxiv"} or re.fullmatch(
            r"(?:doi)?10\d+|(?:arxiv)?\d{4}\d{4,5}",
            href_label,
        ):
            return None
        title_source = href_parts[1]
    else:
        if (
            re.search(r"(?:https?://|\\url\s*\{)", title_block, re.IGNORECASE)
            or re.match(
                r"\s*In\b.*\b(?:editor|editors|proceedings|conference)\b",
                _clean_field_value(title_block),
                re.IGNORECASE,
            )
        ):
            return None
        title_source = title_block

    title_year = _last_plausible_year(title_source)
    title = _clean_title_value(title_source)
    if title_year is not None:
        title = re.sub(
            rf"(?:[,;(]\s*)?\b{title_year}\b[.)]?\s*$",
            "",
            title,
        )
    title = title.rstrip(" \t\r\n.,;:")
    publication_blocks = blocks[title_index + 1:]
    publication_block = " ".join(publication_blocks)
    if not authors or not title:
        return None

    journal = ""
    for block in publication_blocks:
        journal = _extract_rendered_venue(block)
        if journal:
            break

    return {
        "authors": authors,
        "title": title,
        "year": (
            explicit_year
            or _last_plausible_year(publication_block)
            or title_year
            or author_year
        ),
        "journal": journal,
        "url": href_parts[0] if href_parts else "",
        "parse_status": "parsed",
    }


def _structured_candidate_score(candidate: dict[str, Any]) -> int:
    """Rank candidates by explicit metadata and penalize contaminated fields."""

    title = str(candidate.get("title") or "").strip()
    journal = str(candidate.get("journal") or "").strip()
    authors = " ".join(candidate.get("authors") or ())
    if candidate.get("_title_absent"):
        return 100

    score = 0
    score += 4 if title else 0
    score += min(len(candidate.get("authors") or ()), 4)
    score += 2 if journal else 0
    score += 1 if candidate.get("year") is not None else 0
    score += 1 if candidate.get("doi") else 0
    score += 3 if len(title.split()) >= 2 else 0
    if title and journal and _normalize_text(title) == _normalize_text(journal):
        score -= 20
    if re.search(r"(?:https?://|doi\s*:|pathdoi:)", title, re.IGNORECASE):
        score -= 30
    if re.search(r"(?:https?://|doi\s*:|pathdoi:)", journal, re.IGNORECASE):
        score -= 30
    if len(journal.split()) > 12:
        score -= 12
    if title and journal and _normalize_text(title) in _normalize_text(journal):
        score -= 15
    if re.search(
        r"\b[A-Z][A-Za-z]{0,5}\.\s+[A-Z][A-Za-z]{0,8}$",
        title,
    ):
        score -= 6
    if re.search(r"(?:https?://|doi\s*:|href)", authors, re.IGNORECASE):
        score -= 30
    if title and _looks_like_venue_fragment(title) and journal:
        score -= 8
    return score


def _parse_structured_bibitem(item_text: str) -> dict[str, Any] | None:
    parsers = []
    if re.search(r"\\bibinfo(?![A-Za-z@])", item_text):
        parsers.append(_parse_bibinfo_bibitem)
    if re.search(r"``|“", item_text):
        parsers.append(_parse_quoted_bibitem)
    if re.search(r"\\newblock(?![A-Za-z@])", item_text):
        parsers.append(_parse_newblock_bibitem)
    if re.search(r"\\doi(?![A-Za-z@])", item_text, re.IGNORECASE):
        parsers.append(_parse_doi_wrapper_bibitem)
    if re.search(r"\\href(?:@noop)?(?![A-Za-z@])", item_text):
        parsers.append(_parse_href_venue_bibitem)
    if re.search(
        r"\\(?:emph|textit)(?![A-Za-z@])|"
        r"\{\s*\\(?:em|it)\s+",
        item_text,
    ):
        parsers.append(_parse_emphasized_bibitem)
    if re.search(
        r"\\textbf\s*\{\s*(?:[A-Z]\s*)?\d|"
        r"\{\s*\\bf\s*(?=(?:[A-Z]\s*)?\d)",
        item_text,
        re.IGNORECASE,
    ):
        parsers.append(_parse_bold_volume_bibitem)
    candidates = [
        candidate
        for parser in parsers
        if (candidate := parser(item_text)) is not None
    ]
    fallback_parsers = [
        _parse_comma_bibitem,
        _parse_sentence_bibitem,
    ]
    # Colon-separated author/title styles can coexist with a weaker comma
    # candidate (which may mistake the second author for the title).  They are
    # unambiguous when the citation itself has no href/DOI wrapper.
    if (
        ":" in item_text
        and not re.search(r"\\(?:href|doi)(?![A-Za-z@])", item_text, re.IGNORECASE)
    ):
        fallback_parsers.insert(0, _parse_colon_bibitem)
    candidates.extend(
        candidate
        for parser in fallback_parsers
        if (candidate := parser(item_text)) is not None
    )
    title_absent = next(
        (candidate for candidate in candidates if candidate.get("_title_absent")),
        None,
    )
    meaningful_title = any(
        candidate.get("title")
        and len(str(candidate.get("title")).split()) >= 3
        and not re.match(
            r"^[A-ZÀ-ÖØ-ÞŠŽČ]\.?[,;:]",
            str(candidate.get("title")).strip(),
        )
        and not re.search(r"\d|;", str(candidate.get("title")))
        and not _looks_like_venue_fragment(str(candidate.get("title")))
        for candidate in candidates
    )
    if title_absent is not None and not meaningful_title:
        return {
            key: value
            for key, value in title_absent.items()
            if not key.startswith("_")
        }
    if meaningful_title:
        candidates = [
            candidate
            for candidate in candidates
            if not candidate.get("_title_absent")
        ]
    if not candidates:
        return None

    primary = next(
        iter(sorted(candidates, key=_structured_candidate_score, reverse=True)),
        candidates[0],
    ).copy()
    for candidate in candidates:
        for field_name in ("authors", "title", "year", "doi", "journal", "url"):
            if not primary.get(field_name) and candidate.get(field_name):
                primary[field_name] = candidate[field_name]
    primary["parse_status"] = "parsed" if primary.get("title") else "partial"
    return {
        key: value
        for key, value in primary.items()
        if not key.startswith("_")
    }


def _parse_bibitem_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Parse \\bibitem entries from TeX or BBL files."""
    content = _strip_reference_comments(content)
    entries: List[BibEntry] = []
    text_only_count = 0
    item_count = len(re.findall(r"\\bibitem(?![A-Za-z@])", content))
    bibitem_pattern = re.compile(
        r"\\bibitem(?![A-Za-z@])\s*"
        r"(?:\[(?P<label>.*?)\]\s*)?"
        r"\{(?P<key>[^{}]+)\}"
        r"(?P<body>.*?)"
        r"(?=\\bibitem(?![A-Za-z@])|\\end\s*\{thebibliography\}|\Z)",
        re.DOTALL,
    )

    for match in bibitem_pattern.finditer(content):
        raw_citation_key = match.group("key").strip()
        citation_key = (
            _safe_citation_key(
                raw_citation_key,
                file_path=file_path,
                sequence=len(entries),
                diagnostics=diagnostics,
            )
            if raw_citation_key
            else ""
        )
        item_text = match.group("body").strip()

        structured = _parse_structured_bibitem(item_text)
        title = structured["title"] if structured is not None else ""
        authors = structured["authors"] if structured is not None else ()
        year = structured["year"] if structured is not None else None
        journal = structured["journal"] if structured is not None else ""

        cleaned_text = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", item_text)
        cleaned_text = _clean_latex_accents(cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        doi = (
            structured.get("doi", "") if structured is not None else ""
        ) or _normalize_doi(cleaned_text)

        url_match = re.search(r"(https?://[^\s]+)", cleaned_text)
        url = (
            structured.get("url", "") if structured is not None else ""
        ) or (url_match.group(1) if url_match else "")

        if not citation_key:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIBTEX_PARTIAL",
                    message="Could not fully parse bibliography entry",
                    file_path=file_path,
                )
            )

        if not citation_key:
            continue

        if structured is None or structured["parse_status"] != "parsed":
            text_only_count += 1
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="CITATION_TEXT_ONLY",
                    message="Unstructured \\bibitem body was retained without inventing a title",
                    file_path=file_path,
                )
            )

        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type="misc",
                raw_citation_key=raw_citation_key,
                raw_entry=item_text,
                source_kind="bibitem",
                parse_status=(
                    structured["parse_status"]
                    if structured is not None
                    else "citation_text_only"
                ),
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal=journal,
                url=url,
            )
        )

    if text_only_count:
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="BIBTEX_PARTIAL",
                message=(
                    f"{text_only_count} of {len(entries)} bibliography entries "
                    "could not be separated into complete structured metadata"
                ),
                file_path=file_path,
            )
        )

    if item_count != len(entries):
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="BIBTEX_PARTIAL",
                message=(
                    f"Parsed {len(entries)} of {item_count} bibliography items"
                ),
                file_path=file_path,
            )
        )

    return entries


def _read_braced_group(text: str, position: int) -> tuple[str, int] | None:
    """Read one balanced braced TeX argument without expanding macros."""
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        return None

    depth = 1
    cursor = position + 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1:cursor], cursor + 1
        cursor += 1
    return None


def _year_from_text(
    value: str,
    *,
    file_path: str = "",
    diagnostics: List[Diagnostic] | None = None,
) -> Optional[int]:
    """Return a plausible year only from an explicitly structured field."""

    match = re.search(r"\b(\d{4})\b", _clean_field_value(value))
    if match is None:
        return None
    year = int(match.group(1))
    if 1600 <= year <= date.today().year + 1:
        return year
    if diagnostics is not None:
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.WARNING,
                code="INVALID_YEAR",
                message=f"Discarded implausible structured year {year}",
                file_path=file_path,
            )
        )
    return None


def _parse_biber_bbl_content(
    content: str,
    file_path: str,
    diagnostics: List[Diagnostic],
) -> List[BibEntry]:
    r"""Extract the stable fields emitted by biblatex/Biber ``.bbl`` files.

    Biber does not use ``\bibitem``; it writes ``\entry`` plus ``\field``
    commands.  We deliberately read only those flat fields represented by the
    shared contract and do not attempt to execute Biber macros.
    """
    content = _strip_reference_comments(content)
    starts = list(re.finditer(r"\\entry\s*\{([^{}]+)\}\s*\{([^{}]+)\}", content))
    entries: List[BibEntry] = []

    for index, start in enumerate(starts):
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        block_end = content.find(r"\endrefsection", start.end(), next_start)
        if block_end < 0:
            block_end = next_start
        block = content[start.end():block_end]
        fields: Dict[str, str] = {}

        for field_match in re.finditer(r"\\field\s*\{([^{}]+)\}", block):
            value = _read_braced_group(block, field_match.end())
            if value is not None:
                fields[field_match.group(1).strip().lower()] = value[0]

        raw_citation_key = start.group(1).strip()
        if not raw_citation_key:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIBTEX_PARTIAL",
                    message="Biber entry has no citation key",
                    file_path=file_path,
                )
            )
            continue
        citation_key = _safe_citation_key(
            raw_citation_key,
            file_path=file_path,
            sequence=index,
            diagnostics=diagnostics,
        )

        title = _clean_title_value(fields.get("title", ""))
        year = _year_from_text(
            fields.get("year", "") or fields.get("date", ""),
            file_path=file_path,
            diagnostics=diagnostics,
        )
        doi = _normalize_doi(fields.get("doi", ""))
        journal = _clean_field_value(
            fields.get("journaltitle", "") or fields.get("journal", "")
        )
        url = _extract_url(fields)
        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type=start.group(2).strip().lower() or "misc",
                raw_citation_key=raw_citation_key,
                raw_entry=content[start.start():block_end].strip(),
                source_kind="biber_bbl",
                title=title,
                year=year,
                doi=doi,
                journal=journal,
                url=url,
            )
        )

    return entries


def _parse_harvard_bbl_content(
    content: str,
    file_path: str,
    diagnostics: List[Diagnostic],
) -> List[BibEntry]:
    r"""Extract ``\harvarditem`` records used by Harvard-style ``.bbl`` files."""
    content = _strip_reference_comments(content)
    starts = list(re.finditer(r"\\harvarditem\*?(?![A-Za-z@])", content))
    entries: List[BibEntry] = []

    for index, start in enumerate(starts):
        cursor = start.end()
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor < len(content) and content[cursor] == "[":
            depth = 1
            cursor += 1
            while cursor < len(content) and depth:
                if content[cursor] == "[":
                    depth += 1
                elif content[cursor] == "]":
                    depth -= 1
                cursor += 1

        author_group = _read_braced_group(content, cursor)
        year_group = _read_braced_group(content, author_group[1]) if author_group else None
        key_group = _read_braced_group(content, year_group[1]) if year_group else None
        if author_group is None or year_group is None or key_group is None:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIBTEX_PARTIAL",
                    message="Could not parse Harvard bibliography entry",
                    file_path=file_path,
                )
            )
            continue

        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(content)
        body = content[key_group[1]:next_start]
        cleaned_body = _clean_field_value(body)
        raw_citation_key = key_group[0].strip()
        if not raw_citation_key:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIBTEX_PARTIAL",
                    message="Harvard bibliography entry has no citation key",
                    file_path=file_path,
                )
            )
            continue
        citation_key = _safe_citation_key(
            raw_citation_key,
            file_path=file_path,
            sequence=index,
            diagnostics=diagnostics,
        )

        doi = _normalize_doi(cleaned_body)
        url_match = re.search(r"https?://[^\s{}]+", cleaned_body)
        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type="misc",
                raw_citation_key=raw_citation_key,
                raw_entry=body.strip(),
                source_kind="harvard_bbl",
                parse_status="citation_text_only",
                title="",
                authors=_parse_authors(author_group[0]),
                year=_year_from_text(
                    year_group[0],
                    file_path=file_path,
                    diagnostics=diagnostics,
                ),
                doi=doi,
                journal="",
                url=url_match.group(0) if url_match else "",
            )
        )
        diagnostics.append(
            Diagnostic(
                severity=DiagnosticSeverity.INFO,
                code="CITATION_TEXT_ONLY",
                message="Unstructured Harvard bibliography body was retained without inventing a title",
                file_path=file_path,
            )
        )

    return entries


def _parse_bbl_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Read the common BibTeX, Biber, and Harvard bibliography formats."""
    return [
        *_parse_bibitem_content(content, file_path, diagnostics),
        *_parse_biber_bbl_content(content, file_path, diagnostics),
        *_parse_harvard_bbl_content(content, file_path, diagnostics),
    ]


def _merge_same_key_entries(entries: Sequence[BibEntry]) -> list[BibEntry]:
    """Merge duplicate citation keys within one version.

    A source archive may contain both a structured ``.bib`` file and its
    generated ``.bbl`` file.  The structured record is normally encountered
    first; later records fill missing fields instead of creating a second
    reference row.
    """

    merged: dict[str, BibEntry] = {}
    for entry in entries:
        current = merged.get(entry.citation_key)
        if current is None:
            merged[entry.citation_key] = entry
            continue
        merged[entry.citation_key] = BibEntry(
            citation_key=current.citation_key,
            entry_type=(
                current.entry_type
                if current.entry_type != "misc"
                else entry.entry_type
            ),
            raw_citation_key=current.raw_citation_key,
            raw_entry=(
                current.raw_entry
                if _entry_quality(current) >= _entry_quality(entry)
                else entry.raw_entry
            ),
            source_kind=current.source_kind,
            parse_status=(
                "parsed"
                if current.parse_status == "parsed" or entry.parse_status == "parsed"
                else current.parse_status
            ),
            title=_prefer_title(current, entry).title,
            authors=(
                current.authors
                if len(current.authors) >= len(entry.authors)
                else entry.authors
            ),
            year=current.year if current.year is not None else entry.year,
            doi=_normalize_doi(current.doi or entry.doi),
            journal=current.journal or entry.journal,
            url=current.url or entry.url,
        )
    return list(merged.values())


def _reference_group_key(
    version: int,
    entry: BibEntry,
    sequence: int,
) -> str:
    """Build the deterministic identity used to deduplicate one reference."""
    doi = _normalize_doi(entry.doi)
    if doi:
        return f"doi:{doi}"

    arxiv_id = _normalize_arxiv_id(entry.url) if "arxiv.org" in entry.url.lower() else ""
    if arxiv_id:
        return f"arxiv:{arxiv_id}"

    if entry.title and entry.authors and entry.year:
        norm_title = _normalize_text(entry.title)
        norm_authors = "_".join(sorted(_normalize_text(a) for a in entry.authors))
        return f"t_a_y:{norm_title}|{norm_authors}|{entry.year}"

    normalized_raw_entry = _normalize_text(
        _clean_field_value(entry.raw_entry or "")
    )
    if normalized_raw_entry:
        digest = hashlib.sha256(normalized_raw_entry.encode("utf-8")).hexdigest()
        return f"raw:{digest}"

    return f"unique:{version}_{entry.citation_key}_{sequence}"


def process_references(sources: Sequence[SourceBundle]) -> ReferenceBundle:
    """Process all versions of a publication to extract, merge, and deduplicate references."""
    if not sources:
        return ReferenceBundle(
            arxiv_id="unknown",
            entries=(),
            citation_maps={},
            raw_entry_count=0,
        )

    arxiv_id = sources[0].arxiv_id
    diagnostics: List[Diagnostic] = []

    version_raw_entries: Dict[int, List[BibEntry]] = {}
    raw_entry_count = 0

    for source in sources:
        if source.status is ProcessingStatus.EMPTY_NO_TEX:
            continue
        cited_keys = _cited_keys(source)
        source_diagnostics: List[Diagnostic] = []
        bib_entries: List[BibEntry] = []
        supplemental_entries: List[BibEntry] = []
        for file_obj in source.files:
            path_lower = file_obj.relative_path.lower()
            if path_lower.endswith(".bib"):
                bib_entries.extend(
                    _parse_bib_content(
                        file_obj.content,
                        file_obj.relative_path,
                        source_diagnostics,
                    )
                )
            elif path_lower.endswith(".bbl"):
                supplemental_entries.extend(
                    _parse_bbl_content(
                        file_obj.content,
                        file_obj.relative_path,
                        source_diagnostics,
                    )
                )
            elif path_lower.endswith(".tex"):
                supplemental_entries.extend(
                    _parse_bibitem_content(
                        file_obj.content,
                        file_obj.relative_path,
                        source_diagnostics,
                    )
                )

        # Reference diagnostics originate from one concrete source version.
        # Preserve that provenance so TV1 can expose bibtex_partial at the
        # version level instead of only downgrading the whole publication.
        diagnostics.extend(
            replace(
                diagnostic,
                arxiv_id=diagnostic.arxiv_id or source.arxiv_id,
                version=diagnostic.version or source.version,
            )
            for diagnostic in source_diagnostics
        )

        bib_keys = {entry.citation_key for entry in bib_entries}
        v_entries = bib_entries + [
            entry
            for entry in supplemental_entries
            if entry.citation_key not in bib_keys
        ]
        if cited_keys:
            v_entries = [
                entry
                for entry in v_entries
                if (entry.raw_citation_key or entry.citation_key) in cited_keys
            ]
        raw_entry_count += len(v_entries)
        v_entries = _merge_same_key_entries(v_entries)
        version_raw_entries[source.version] = v_entries

    group_map: Dict[str, List[Tuple[int, BibEntry]]] = {}
    ungrouped_counter = 0

    for v_num, entries in version_raw_entries.items():
        for entry in entries:
            ungrouped_counter += 1
            g_key = _reference_group_key(v_num, entry, ungrouped_counter)
            group_map.setdefault(g_key, []).append((v_num, entry))

    entries_by_raw_key: Dict[str, List[BibEntry]] = {}
    for entries in version_raw_entries.values():
        for entry in entries:
            entries_by_raw_key.setdefault(
                entry.raw_citation_key or entry.citation_key,
                [],
            ).append(entry)
    for raw_key, entries in entries_by_raw_key.items():
        distinct_dois = {_normalize_doi(entry.doi) for entry in entries if _normalize_doi(entry.doi)}
        if len(distinct_dois) > 1:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="REFERENCE_IDENTITY_CONFLICT",
                    message=(
                        f"citation key {raw_key!r} has conflicting DOIs across versions: "
                        + ", ".join(sorted(distinct_dois))
                    ),
                )
            )

    merged_entries: List[BibEntry] = []
    citation_maps: Dict[int, Dict[str, str]] = {}
    used_canonical_keys: Set[str] = set()

    for g_key, items in group_map.items():
        items_sorted = sorted(items, key=lambda x: (x[0], x[1].citation_key))
        base_canonical_key = items_sorted[0][1].citation_key

        canonical_key = base_canonical_key
        counter = 1
        while canonical_key in used_canonical_keys:
            canonical_key = f"{base_canonical_key}_{counter}"
            counter += 1
        used_canonical_keys.add(canonical_key)

        entry_type = "misc"
        title_entry: BibEntry | None = None
        authors: Tuple[str, ...] = ()
        year: Optional[int] = None
        doi = ""
        journal = ""
        url = ""

        for _, entry in items_sorted:
            if entry.entry_type and entry.entry_type != "misc":
                entry_type = entry.entry_type
            if entry.title:
                title_entry = _prefer_title(title_entry, entry)
            if entry.authors:
                if not authors or len(entry.authors) > len(authors):
                    authors = entry.authors
            if entry.year is not None and year is None:
                year = entry.year
            normalized_doi = _normalize_doi(entry.doi)
            if normalized_doi and not doi:
                doi = normalized_doi
            if entry.journal and not journal:
                journal = entry.journal
            if entry.url and not url:
                url = entry.url

        merged_entry = BibEntry(
            citation_key=canonical_key,
            entry_type=entry_type,
            raw_citation_key=items_sorted[0][1].raw_citation_key,
            raw_entry=max(
                (entry for _, entry in items_sorted),
                key=_entry_quality,
            ).raw_entry,
            source_kind=next(
                (entry.source_kind for _, entry in items_sorted if entry.parse_status == "parsed"),
                items_sorted[0][1].source_kind,
            ),
            parse_status=(
                "parsed"
                if any(entry.parse_status == "parsed" for _, entry in items_sorted)
                else items_sorted[0][1].parse_status
            ),
            title=title_entry.title if title_entry is not None else "",
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            url=url,
        )
        merged_entries.append(merged_entry)

        for v_num, entry in items:
            citation_maps.setdefault(v_num, {})[
                entry.raw_citation_key or entry.citation_key
            ] = canonical_key

    for source in sources:
        if source.version not in citation_maps:
            citation_maps[source.version] = {}

    return ReferenceBundle(
        arxiv_id=arxiv_id,
        entries=tuple(merged_entries),
        citation_maps=citation_maps,
        diagnostics=tuple(diagnostics),
        raw_entry_count=raw_entry_count,
    )


__all__ = ["process_references"]
