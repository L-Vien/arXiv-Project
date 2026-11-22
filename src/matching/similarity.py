"""Similarity functions between a BibTeX entry and a crawled reference (TV4, Milestone 3).

Pure functions on already-cleaned inputs (see cleaning.py), standard library only.

Missing data: every function returns math.nan when either side has nothing to compare (empty title,
no authors, unknown year, missing identifier). NaN is "unknown", not "different": it propagates into
pandas / scikit-learn as a missing value, and any threshold test such as `score >= 0.9` is False for it.
The feature builder (Bao Nhi) decides the missing-value policy.

Why several title measures: crawled reference titles are sometimes polluted with journal, volume and
author text, e.g. "MNRAS, 487, 2005 Costanzi, M., Rozo, E., Simet, M., et al." Symmetric measures such
as Jaccard drop sharply on such titles, while coverage ("how many words of the BibTeX title appear in the
reference title") stays high when the true title is contained in the noise.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

from src.matching.cleaning import Author, CleanedRecord

MISSING = math.nan


def is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------------------------
# Titles (inputs: token tuples from cleaning.title_tokens or strings from cleaning.normalize_title)
# ---------------------------------------------------------------------------
def title_jaccard(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    """|A and B| / |A or B| over distinct title tokens."""
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return MISSING
    return len(a & b) / len(a | b)


def title_coverage(query_tokens: Iterable[str], target_tokens: Iterable[str]) -> float:
    """Fraction of the query's distinct tokens that occur in the target (asymmetric)."""
    q, t = set(query_tokens), set(target_tokens)
    if not q or not t:
        return MISSING
    return len(q & t) / len(q)


def _char_ngrams(text: str, n: int) -> set[str]:
    padded = f" {text} "
    return {padded[i: i + n] for i in range(max(1, len(padded) - n + 1))}


def title_char_dice(a: str, b: str, n: int = 3) -> float:
    """Dice coefficient of character n-grams of two normalized titles; tolerant to typos."""
    if not a or not b:
        return MISSING
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def title_sequence_ratio(a: str, b: str) -> float:
    """difflib ratio of two normalized titles (order-sensitive)."""
    if not a or not b:
        return MISSING
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def title_exact(a: str, b: str) -> float:
    """1.0 if the normalized titles are identical, else 0.0."""
    if not a or not b:
        return MISSING
    return float(a == b)


def title_contains(a: str, b: str, min_chars: int = 12) -> float:
    """1.0 if the shorter normalized title (at least min_chars long) is a substring of the longer one."""
    if not a or not b:
        return MISSING
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return float(len(short) >= min_chars and short in long_)


# ---------------------------------------------------------------------------
# Authors (inputs: tuples of cleaning.Author)
# ---------------------------------------------------------------------------
def _same_author(a: Author, b: Author, use_initials: bool) -> bool:
    if a.key != b.key:
        return False
    if use_initials and a.initials and b.initials:
        return a.initials[0] == b.initials[0]
    return True


def _matched_count(a: Sequence[Author], b: Sequence[Author], use_initials: bool) -> int:
    """Number of authors of `a` that can be paired one-to-one with an author of `b` (greedy first fit)."""
    pool: dict[str, list[str]] = defaultdict(list)
    for other in b:
        pool[other.key].append(other.initials)
    count = 0
    for author in a:
        candidates = pool.get(author.key)
        if not candidates:
            continue
        for i, initials in enumerate(candidates):
            if not use_initials or not author.initials or not initials or author.initials[0] == initials[0]:
                del candidates[i]
                count += 1
                break
    return count


def author_overlap(a: Sequence[Author], b: Sequence[Author], use_initials: bool = False) -> float:
    """Matched authors / size of the shorter list (1.0 = the shorter list is fully contained)."""
    if not a or not b:
        return MISSING
    return _matched_count(a, b, use_initials) / min(len(a), len(b))


def author_jaccard(a: Sequence[Author], b: Sequence[Author], use_initials: bool = False) -> float:
    """Matched authors / size of the union of both lists."""
    if not a or not b:
        return MISSING
    common = _matched_count(a, b, use_initials)
    return common / (len(a) + len(b) - common)


def first_author_match(a: Sequence[Author], b: Sequence[Author], use_initials: bool = False) -> float:
    """1.0 if both first authors have the same family name (and first initial when use_initials)."""
    if not a or not b:
        return MISSING
    return float(_same_author(a[0], b[0], use_initials))


# ---------------------------------------------------------------------------
# Year and identifiers
# ---------------------------------------------------------------------------
def year_diff(a: int | None, b: int | None) -> float:
    """Absolute difference in years (arXiv year and publication year often differ by 0 or 1)."""
    if a is None or b is None:
        return MISSING
    return float(abs(a - b))


def same_identifier(a: str | None, b: str | None) -> float:
    """1.0 / 0.0 when both identifiers (DOI or arXiv ID) are present, NaN otherwise."""
    if not a or not b:
        return MISSING
    return float(a == b)


# ---------------------------------------------------------------------------
# One call for all measures
# ---------------------------------------------------------------------------
COMPLETENESS_NAMES = ("has_title", "has_authors", "has_year", "has_doi", "has_arxiv_id")


def record_completeness(rec: CleanedRecord) -> dict[str, float]:
    """Which fields a cleaned record actually has (1.0 / 0.0), keyed by COMPLETENESS_NAMES.

    Use as indicator features next to the NaN-valued similarities, so a model can tell
    "no authors on the reference" apart from "authors present but different".
    """
    return {
        "has_title": float(bool(rec.title_tokens)),
        "has_authors": float(bool(rec.authors)),
        "has_year": float(rec.year is not None),
        "has_doi": float(rec.doi is not None),
        "has_arxiv_id": float(rec.arxiv_id is not None),
    }


FEATURE_NAMES = (
    "title_exact", "title_contains", "title_jaccard", "title_cov_bib_in_ref", "title_cov_ref_in_bib",
    "title_char_dice", "title_seq_ratio", "author_overlap", "author_jaccard", "first_author_match",
    "first_author_match_initial", "year_diff", "doi_match", "arxiv_id_match",
)


def compare_records(bib: CleanedRecord, ref: CleanedRecord, with_sequence_ratio: bool = True) -> dict[str, float]:
    """All measures between a BibTeX entry and a reference, keyed by FEATURE_NAMES (stable order).

    with_sequence_ratio=False skips the difflib ratio (about 60% of the cost per pair, and largely redundant
    with title_char_dice); `title_seq_ratio` is then NaN. Use it for the first pass over many candidates.
    """
    if bib.authors and ref.authors:
        common = _matched_count(bib.authors, ref.authors, False)
        overlap = common / min(len(bib.authors), len(ref.authors))
        jaccard = common / (len(bib.authors) + len(ref.authors) - common)
    else:
        overlap = jaccard = MISSING
    return {
        "title_exact": title_exact(bib.title, ref.title),
        "title_contains": title_contains(bib.title, ref.title),
        "title_jaccard": title_jaccard(bib.title_tokens, ref.title_tokens),
        "title_cov_bib_in_ref": title_coverage(bib.title_tokens, ref.title_tokens),
        "title_cov_ref_in_bib": title_coverage(ref.title_tokens, bib.title_tokens),
        "title_char_dice": title_char_dice(bib.title, ref.title),
        "title_seq_ratio": title_sequence_ratio(bib.title, ref.title) if with_sequence_ratio else MISSING,
        "author_overlap": overlap,
        "author_jaccard": jaccard,
        "first_author_match": first_author_match(bib.authors, ref.authors),
        "first_author_match_initial": first_author_match(bib.authors, ref.authors, use_initials=True),
        "year_diff": year_diff(bib.year, ref.year),
        "doi_match": same_identifier(bib.doi, ref.doi),
        "arxiv_id_match": same_identifier(bib.arxiv_id, ref.arxiv_id),
    }
