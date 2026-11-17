"""Citation key rewriting utilities for TV5.

This module provides functions to rewrite \\cite{...} commands in TeX source files
using the canonical citation maps generated during reference deduplication.
"""

from __future__ import annotations

import re
from typing import Mapping


def rewrite_citations(content: str, citation_map: Mapping[str, str]) -> str:
    """Rewrite keys in common LaTeX citation commands.

    Optional prenotes/postnotes are kept intact, for example
    ``\\citep[see][p. 2]{old}``.
    """
    # Match natbib/biblatex-style commands while allowing up to two optional
    # bracket arguments before the citation key group.
    cite_pattern = re.compile(
        r"(\\[A-Za-z]*[Cc]ite[A-Za-z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{)([^{}]+)(\})"
    )

    def _replace_keys(match: re.Match) -> str:
        prefix = match.group(1)
        keys_body = match.group(2)
        suffix = match.group(3)

        # Split keys by comma, strip whitespace, and map them if present in citation_map
        new_keys: list[str] = []
        for key in (item.strip() for item in keys_body.split(",")):
            if not key:
                continue
            mapped = citation_map.get(key, key)
            if mapped not in new_keys:
                new_keys.append(mapped)
        return f"{prefix}{','.join(new_keys)}{suffix}"

    return cite_pattern.sub(_replace_keys, content)


__all__ = ["rewrite_citations"]
