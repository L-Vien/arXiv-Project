"""Citation key rewriting utilities for TV5.

This module provides functions to rewrite \\cite{...} commands in TeX source files
using the canonical citation maps generated during reference deduplication.
"""

from __future__ import annotations

import re
from typing import Mapping


def rewrite_citations(content: str, citation_map: Mapping[str, str]) -> str:
    """Rewrite citation keys inside \\cite{...}, \\citep{...}, \\citet{...} commands using citation_map."""
    if not citation_map:
        return content

    # Match citation commands like \cite{key1,key2}, \citep{key1}, etc.
    cite_pattern = re.compile(r"(\\(?:cite|citep|citet|citep\*|citet\*|citealt|citepauthor)\s*\{)([^}]+)(\})")

    def _replace_keys(match: re.Match) -> str:
        prefix = match.group(1)
        keys_body = match.group(2)
        suffix = match.group(3)

        # Split keys by comma, strip whitespace, and map them if present in citation_map
        keys = [k.strip() for k in keys_body.split(",")]
        new_keys = [citation_map.get(k, k) for k in keys]
        return f"{prefix}{','.join(new_keys)}{suffix}"

    return cite_pattern.sub(_replace_keys, content)


__all__ = ["rewrite_citations"]
