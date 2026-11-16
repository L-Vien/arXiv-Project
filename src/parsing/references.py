"""Reference extraction and BibTeX deduplication for TV5.

This module processes a sequence of SourceBundle objects for all versions of a publication,
extracting references from \\bibitem entries and .bib/.bbl files, deduplicating them across versions,
merging fields, selecting canonical citation keys, and generating per-version citation maps.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Tuple, List, Dict, Set, Optional, Any

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


def _clean_latex_accents(text: str) -> str:
    """Clean LaTeX accent commands and braces into clean unicode characters."""
    if not text:
        return ""
    # Unpack braces in commands like {\'n} -> \'n or {\'e} -> \'e
    text = re.sub(r"\\([\\`^~\"vvcstudb])\{([a-zA-Z])\}", r"\\\1\2", text)

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
    text = re.sub(r"\\[a-zA-Z]+\{([a-zA-Z])\}", lambda m: m.group(1), text)
    text = re.sub(r"[{}]", "", text)
    return text


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


def _parse_authors(author_str: str) -> Tuple[str, ...]:
    """Parse author string separated by 'and', '\\and', ';', or commas into a tuple of author names."""
    if not author_str:
        return ()
    cleaned = _clean_field_value(author_str)
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
                url = f"https://doi.org/{text}"
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
    return _clean_field_value(url)


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


def _parse_bib_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Parse BibTeX format content into BibEntry objects robustly."""
    entries: List[BibEntry] = []

    i = 0
    n = len(content)
    while i < n:
        if content[i] == '@':
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

                body_end = j - 1
                entry_body = content[body_start:body_end]
                i = j

                comma_idx = entry_body.find(',')
                if comma_idx != -1:
                    citation_key = entry_body[:comma_idx].strip()
                    citation_key = re.sub(r"[\{\}\"']", "", citation_key).strip()
                    fields_text = entry_body[comma_idx+1:]
                else:
                    citation_key = entry_body.strip()
                    fields_text = ""

                field_dict = _parse_bib_fields(fields_text)

                title = _clean_field_value(field_dict.get("title", ""))
                author_raw = field_dict.get("author", "") or field_dict.get("authors", "")
                authors = _parse_authors(author_raw)

                year_raw = field_dict.get("year", "")
                year: Optional[int] = None
                if year_raw:
                    clean_year_str = _clean_field_value(year_raw)
                    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", clean_year_str)
                    if year_match:
                        try:
                            year = int(year_match.group(1))
                        except ValueError:
                            pass

                doi = _clean_field_value(field_dict.get("doi", ""))
                journal = _clean_field_value(field_dict.get("journal", "") or field_dict.get("booktitle", "") or field_dict.get("journaltitle", ""))
                url = _extract_url(field_dict)

                if not citation_key or (not title and not authors and not doi):
                    diagnostics.append(
                        Diagnostic(
                            severity=DiagnosticSeverity.WARNING,
                            code="BIBTEX_PARTIAL",
                            message="Could not fully parse bibliography entry",
                            file_path=file_path,
                        )
                    )

                if citation_key:
                    entries.append(
                        BibEntry(
                            citation_key=citation_key,
                            entry_type=entry_type or "misc",
                            title=title,
                            authors=authors,
                            year=year,
                            doi=doi,
                            journal=journal,
                            url=url,
                        )
                    )
            else:
                i = j
        else:
            i += 1
    return entries


def _parse_bibitem_content(content: str, file_path: str, diagnostics: List[Diagnostic]) -> List[BibEntry]:
    """Parse \\bibitem entries from TeX or BBL files."""
    entries: List[BibEntry] = []
    bibitem_pattern = re.compile(
        r"\\bibitem(?:\[.*?\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
        re.DOTALL
    )

    for match in bibitem_pattern.finditer(content):
        citation_key = match.group(1).strip()
        item_text = match.group(2).strip()

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", item_text)
        year: Optional[int] = None
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                pass
            before_year = item_text[:year_match.start()]
        else:
            before_year = item_text

        authors: Tuple[str, ...] = ()
        if before_year:
            cleaned_before = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", before_year)
            cleaned_before = _clean_latex_accents(cleaned_before)
            parts = re.split(r"\.\s+|\,\s+|\bby\b|\band\b", cleaned_before, flags=re.IGNORECASE)
            if parts and len(parts) > 1:
                potential_authors = []
                for p in parts[:2]:
                    p = p.strip()
                    if p and len(p.split()) < 6 and any(char.isupper() for char in p):
                        potential_authors.append(p)
                if potential_authors:
                    authors = tuple(potential_authors)

        cleaned_text = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", item_text)
        cleaned_text = _clean_latex_accents(cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        doi_match = re.search(r"(?:doi[:\s]*)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", cleaned_text, re.IGNORECASE)
        doi = doi_match.group(1) if doi_match else ""

        url_match = re.search(r"(https?://[^\s]+)", cleaned_text)
        url = url_match.group(1) if url_match else ""

        title = cleaned_text[:100].strip()

        if not citation_key or not title:
            diagnostics.append(
                Diagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    code="BIBTEX_PARTIAL",
                    message="Could not fully parse bibliography entry",
                    file_path=file_path,
                )
            )

        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type="misc",
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal="",
                url=url,
            )
        )

    return entries


def process_references(sources: Sequence[SourceBundle]) -> ReferenceBundle:
    """Process all versions of a publication to extract, merge, and deduplicate references."""
    if not sources:
        return ReferenceBundle(arxiv_id="unknown", entries=(), citation_maps={})

    arxiv_id = sources[0].arxiv_id
    diagnostics: List[Diagnostic] = []

    version_raw_entries: Dict[int, List[BibEntry]] = {}

    for source in sources:
        if source.status is ProcessingStatus.EMPTY_NO_TEX:
            continue
        v_entries: List[BibEntry] = []
        for file_obj in source.files:
            path_lower = file_obj.relative_path.lower()
            if path_lower.endswith(".bib"):
                v_entries.extend(_parse_bib_content(file_obj.content, file_obj.relative_path, diagnostics))
            elif path_lower.endswith(".bbl"):
                v_entries.extend(_parse_bibitem_content(file_obj.content, file_obj.relative_path, diagnostics))
            elif path_lower.endswith(".tex"):
                v_entries.extend(_parse_bibitem_content(file_obj.content, file_obj.relative_path, diagnostics))
        version_raw_entries[source.version] = v_entries

    group_map: Dict[str, List[Tuple[int, BibEntry]]] = {}
    ungrouped_counter = 0

    for v_num, entries in version_raw_entries.items():
        for entry in entries:
            g_key = ""
            if entry.doi:
                g_key = f"doi:{entry.doi.lower().strip()}"
            elif entry.title and entry.authors and entry.year:
                norm_title = _normalize_text(entry.title)
                norm_authors = "_".join(sorted(_normalize_text(a) for a in entry.authors))
                g_key = f"t_a_y:{norm_title}|{norm_authors}|{entry.year}"
            else:
                ungrouped_counter += 1
                g_key = f"unique:{v_num}_{entry.citation_key}_{ungrouped_counter}"

            group_map.setdefault(g_key, []).append((v_num, entry))

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
        title = ""
        authors: Tuple[str, ...] = ()
        year: Optional[int] = None
        doi = ""
        journal = ""
        url = ""

        for _, entry in items_sorted:
            if entry.entry_type and entry.entry_type != "misc":
                entry_type = entry.entry_type
            if entry.title:
                if not title:
                    title = entry.title
                else:
                    is_current_blob = bool(re.search(r"(doi:|http|et al|\d{4})", title, re.IGNORECASE))
                    is_entry_blob = bool(re.search(r"(doi:|http|et al|\d{4})", entry.title, re.IGNORECASE))
                    if is_current_blob and not is_entry_blob:
                        title = entry.title
                    elif not is_current_blob and is_entry_blob:
                        pass
                    elif len(entry.title) < len(title):
                        title = entry.title
            if entry.authors:
                if not authors or len(entry.authors) > len(authors):
                    authors = entry.authors
            if entry.year is not None and year is None:
                year = entry.year
            if entry.doi and not doi:
                doi = entry.doi
            if entry.journal and not journal:
                journal = entry.journal
            if entry.url and not url:
                url = entry.url

        merged_entry = BibEntry(
            citation_key=canonical_key,
            entry_type=entry_type,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            url=url,
        )
        merged_entries.append(merged_entry)

        for v_num, entry in items:
            citation_maps.setdefault(v_num, {})[entry.citation_key] = canonical_key

    for source in sources:
        if source.version not in citation_maps:
            citation_maps[source.version] = {}

    for source in sources:
        if source.version not in citation_maps:
            citation_maps[source.version] = {}

    return ReferenceBundle(
        arxiv_id=arxiv_id,
        entries=tuple(merged_entries),
        citation_maps=citation_maps,
        diagnostics=tuple(diagnostics),
    )


__all__ = ["process_references"]
