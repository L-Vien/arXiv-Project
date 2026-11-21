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


def _clean_latex_accents(text: str) -> str:
    """Convert common semantic LaTeX to stable display text."""
    if not text:
        return ""

    text = re.sub(
        r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    wrapper = re.compile(
        r"\\(?:emph|ensuremath|mbox|operatorname|text|textit|textbf|texttt|"
        r"textnormal|textrm|textsf|mathrm|mathit|mathbf|url)\s*\{([^{}]*)\}",
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
    }

    def compose_accent(match: re.Match[str]) -> str:
        return unicodedata.normalize(
            "NFC",
            match.group(2) + accent_combining[match.group(1)],
        )

    text = re.sub(
        r"\\([`'\"^~])\{?\\([ij])\}?",
        compose_accent,
        text,
    )
    text = re.sub(
        r"\\([`'\"^~])\{?([a-zA-Z])\}?",
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
    text = re.sub(r"\\c\{([a-zA-Z])\}", lambda m: {"c": "ç", "C": "Ç"}.get(m.group(1), m.group(1)), text)
    while wrapper.search(text):
        text = wrapper.sub(r"\1", text)
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
        r"\\(?:emph|ensuremath|mbox|operatorname|text|textit|textbf|texttt|"
        r"textnormal|textrm|textsf|mathrm|mathit|mathbf|url)\s*\{",
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
            fields[f_name] = f_val
    return fields


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
                journal = _clean_field_value(field_dict.get("journal", "") or field_dict.get("booktitle", "") or field_dict.get("journaltitle", ""))
                url = _extract_url(field_dict)

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
                            parse_status="partial" if not entry_closed else "parsed",
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


def _parse_bibitem_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Parse \\bibitem entries from TeX or BBL files."""
    content = _strip_reference_comments(content)
    entries: List[BibEntry] = []
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

        # A rendered citation may mention several years, future horizons, or
        # access dates. Without a structured year field there is no reliable
        # way to identify the publication year, so leave it unknown for M3.
        year: Optional[int] = None
        before_year = item_text

        authors: Tuple[str, ...] = ()
        if before_year:
            cleaned_before = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", before_year)
            cleaned_before = _clean_latex_accents(cleaned_before)
            author_prefix = cleaned_before.split(".", maxsplit=1)[0].strip()
            authors = _parse_authors(author_prefix)

        cleaned_text = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", item_text)
        cleaned_text = _clean_latex_accents(cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        doi = _normalize_doi(cleaned_text)

        url_match = re.search(r"(https?://[^\s]+)", cleaned_text)
        url = url_match.group(1) if url_match else ""

        # A normal \bibitem body is a rendered citation, not a structured
        # title field. Publishing it as a title (and historically truncating it
        # at 100 characters) corrupts matching data. Keep the identity fields
        # that can be extracted deterministically and record a review signal.
        title = ""

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
                parse_status="citation_text_only",
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal="",
                url=url,
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
        if key_group is None:
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
