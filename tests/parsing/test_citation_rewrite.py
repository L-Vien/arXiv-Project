"""Unit tests specifically for citation key rewriting."""

from __future__ import annotations

import pytest

from src.parsing.citation_rewrite import rewrite_citations


def test_rewrite_citations_various_commands():
    content = (
        "Previous work \\cite{vaswani_attn} and \\citep{devlin2018, brown2020} "
        "along with \\citet*{he2016} explored this."
    )
    citation_map = {
        "vaswani_attn": "vaswani2017",
        "devlin2018": "devlin2018",
        "brown2020": "brown2020",
        "he2016": "he2015",
    }

    rewritten = rewrite_citations(content, citation_map)
    expected = (
        "Previous work \\cite{vaswani2017} and \\citep{devlin2018,brown2020} "
        "along with \\citet*{he2015} explored this."
    )
    assert rewritten == expected


def test_rewrite_citations_empty_map():
    content = "No changes \\cite{key1}."
    assert rewrite_citations(content, {}) == content
