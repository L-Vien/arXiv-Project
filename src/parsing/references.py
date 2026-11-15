"""Reference extraction and BibTeX deduplication for TV5.

This module processes a sequence of SourceBundle objects for all versions of a publication,
extracting references from \\bibitem entries and .bib/.bbl files, deduplicating them across versions,
merging fields, selecting canonical citation keys, and generating per-version citation maps.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Tuple, List, Dict, Set, Optional

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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_authors(author_str: str) -> Tuple[str, ...]:
    """Parse author string separated by 'and' or commas into a tuple of author names."""
    if not author_str:
        return ()
    # Clean up braces and quotes
    author_str = re.sub(r"[\{\}\"']", "", author_str)
    # Split by ' and ' or ';'
    parts = re.split(r"\s+and\s+|;", author_str, flags=re.IGNORECASE)
    authors = [p.strip() for p in parts if p.strip()]
    return tuple(authors)


def _parse_bib_content(content: str) -> List[BibEntry]:
    """Parse BibTeX format content into BibEntry objects."""
    entries: List[BibEntry] = []
    # Match @type{key, fields }
    # Simplified regex for bib entries
    bib_pattern = re.compile(
        r"@([a-zA-Z]+)\s*\{([^,\s]+)\s*,\s*(.*?)(?=\n\s*@|\Z)",
        re.DOTALL
    )

    for match in bib_pattern.finditer(content):
        entry_type = match.group(1).lower()
        citation_key = match.group(2).strip()
        fields_body = match.group(3)

        # Extract fields: field_name = {value} or field_name = "value" or field_name = value
        field_pattern = re.compile(
            r"([a-zA-Z_-]+)\s*=\s*(?:\{([^}]*)\}|" '"([^"]*)"' r"|([0-9]+))",
            re.DOTALL
        )

        field_dict: Dict[str, str] = {}
        for f_match in field_pattern.finditer(fields_body):
            f_name = f_match.group(1).lower()
            f_val = f_match.group(2) or f_match.group(3) or f_match.group(4) or ""
            field_dict[f_name] = f_val.strip()

        title = field_dict.get("title", "")
        # Clean title braces
        title = re.sub(r"[{}]", "", title)

        author_raw = field_dict.get("author", "")
        authors = _parse_authors(author_raw)

        year_raw = field_dict.get("year", "")
        year: Optional[int] = None
        if year_raw:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", year_raw)
            if year_match:
                try:
                    year = int(year_match.group(1))
                except ValueError:
                    pass

        doi = field_dict.get("doi", "")
        journal = field_dict.get("journal", "") or field_dict.get("booktitle", "")
        url = field_dict.get("url", "")

        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type=entry_type,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                journal=journal,
                url=url,
            )
        )

    return entries


def _parse_bibitem_content(content: str) -> List[BibEntry]:
    """Parse \\bibitem entries from TeX or BBL files."""
    entries: List[BibEntry] = []
    # Match \bibitem{key} or \bibitem[label]{key}
    bibitem_pattern = re.compile(
        r"\\bibitem(?:\[.*?\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
        re.DOTALL
    )

    for match in bibitem_pattern.finditer(content):
        citation_key = match.group(1).strip()
        item_text = match.group(2).strip()
        # Clean LaTeX commands and formatting in item_text
        cleaned_text = re.sub(r"\\[a-zA-Z]+(?:\[.*?\])?(?:\{.*?\})?", " ", item_text)
        cleaned_text = re.sub(r"[{}]", "", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Try to extract year
        year: Optional[int] = None
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned_text)
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                pass

        # Try to extract DOI
        doi_match = re.search(r"(?:doi[:\s]*)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", cleaned_text, re.IGNORECASE)
        doi = doi_match.group(1) if doi_match else ""

        # Title heuristic: first sentence or chunk before year/journal
        title = cleaned_text[:100].strip()

        entries.append(
            BibEntry(
                citation_key=citation_key,
                entry_type="misc",
                title=title,
                authors=(),
                year=year,
                doi=doi,
                journal="",
                url="",
            )
        )

    return entries


def process_references(sources: Sequence[SourceBundle]) -> ReferenceBundle:
    """Process all versions of a publication to extract, merge, and deduplicate references."""
    if not sources:
        return ReferenceBundle(arxiv_id="unknown", entries=(), citation_maps={})

    arxiv_id = sources[0].arxiv_id
    diagnostics: List[Diagnostic] = []

    # Collect entries per version and map original keys to entries
    version_raw_entries: Dict[int, List[BibEntry]] = {}

    for source in sources:
        if source.status is ProcessingStatus.EMPTY_NO_TEX:
            continue
        v_entries: List[BibEntry] = []
        for file_obj in source.files:
            path_lower = file_obj.relative_path.lower()
            if path_lower.endswith(".bib") or path_lower.endswith(".bbl"):
                v_entries.extend(_parse_bib_content(file_obj.content))
            # Also check bibitems in any .tex file
            if path_lower.endswith(".tex"):
                v_entries.extend(_parse_bibitem_content(file_obj.content))
        version_raw_entries[source.version] = v_entries

    # Deduplicate and merge entries across versions
    # Grouping key: DOI if available, else (normalized_title, year)
    group_map: Dict[str, List[Tuple[int, BibEntry]]] = {}

    for v_num, entries in version_raw_entries.items():
        for entry in entries:
            g_key = ""
            if entry.doi:
                g_key = f"doi:{entry.doi.lower()}"
            elif entry.title and entry.year:
                g_key = f"t_y:{_normalize_text(entry.title)[:50]}_{entry.year}"
            else:
                g_key = f"key:{entry.citation_key.lower()}"

            group_map.setdefault(g_key, []).append((v_num, entry))

    merged_entries: List[BibEntry] = []
    citation_maps: Dict[int, Dict[str, str]] = {}
    used_canonical_keys: Set[str] = set()

    for g_key, items in group_map.items():
        # Sort items by version number to prioritize earlier versions for canonical key
        items_sorted = sorted(items, key=lambda x: (x[0], x[1].citation_key))
        base_canonical_key = items_sorted[0][1].citation_key

        # Ensure uniqueness of canonical_key to prevent DUPLICATE_CITATION_KEY errors
        canonical_key = base_canonical_key
        counter = 1
        while canonical_key in used_canonical_keys:
            canonical_key = f"{base_canonical_key}_{counter}"
            counter += 1
        used_canonical_keys.add(canonical_key)

        # Merge fields across all items in group
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
            if entry.title and not title:
                title = entry.title
            if entry.authors and not authors:
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

        # Build citation map for each version
        for v_num, entry in items:
            citation_maps.setdefault(v_num, {})[entry.citation_key] = canonical_key

    # Ensure every version has a citation map (even if empty)
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
