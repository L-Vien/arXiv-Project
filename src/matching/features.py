from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Set as AbstractSet
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import chain
from typing import Any
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import (
    BibtexRecord,
    CandidatePair,
    FeatureRow,
    Partition,
    ReferenceRecord,
    normalize_doi,
)


FEATURE_SCHEMA_VERSION = "1.0"
FEATURE_NAMES: tuple[str, ...] = (
    # Title (8)
    "title_exact",
    "title_jaccard",
    "title_edit_similarity",
    "title_token_set_ratio",
    "title_word_tfidf_cosine",
    "title_char_tfidf_cosine",
    "title_length_ratio",
    "title_token_containment",
    # Author (6)
    "first_author_match",
    "last_author_match",
    "author_overlap_count",
    "author_jaccard",
    "author_bib_coverage",
    "author_sequence_similarity",
    # Bibliographic (9)
    "doi_exact",
    "year_exact",
    "year_abs_diff",
    "venue_similarity",
    "bib_title_missing",
    "ref_title_missing",
    "bib_authors_missing",
    "ref_authors_missing",
    "year_missing",
    # Interaction (3)
    "high_title_author",
    "exact_title_year_conflict",
    "doi_match_title_missing",
)


def default_clean_text(value: str) -> str:
    """Provisional Unicode-aware cleaning, replace with TV3's agreed API."""

    value = unicodedata.normalize("NFKC", value).casefold()
    # Preserve letters/numbers from all scripts, treating punctuation as space.
    value = "".join(char if unicodedata.category(char)[0] in "LN" else " " for char in value)
    return " ".join(value.split())


def _set_overlap(a: AbstractSet[str], b: AbstractSet[str]) -> tuple[int, float, float]:
    if not a or not b:
        return 0, 0.0, 0.0
    intersection = len(a & b)
    return intersection, intersection / len(a | b), intersection / min(len(a), len(b))


@lru_cache(maxsize=16_384)
def _tokens(value: str) -> frozenset[str]:
    """Reuse tokenization for a title compared with many references."""

    return frozenset(value.split())


@lru_cache(maxsize=16_384)
def _author_set(authors: tuple[str, ...]) -> frozenset[str]:
    return frozenset(authors)


@lru_cache(maxsize=16_384)
def _author_sequence(authors: tuple[str, ...]) -> str:
    return "|".join(authors)


def _levenshtein_distance(a: str, b: str) -> int:
    """Exact Levenshtein distance using Myers' bit-vector algorithm.

    Python's arbitrary-size integers allow one bit per character in the
    shorter string. This avoids the nested Python loops in the dynamic-
    programming version while retaining the same distance for Unicode text.
    """

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    width = len(a)
    bitmask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks: dict[str, int] = {}
    for index, char in enumerate(a):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)

    positive, negative = bitmask, 0
    distance = width
    for char in b:
        matches = char_masks.get(char, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches | negative
        positive_shift = negative | ~(horizontal | positive)
        negative_shift = positive & horizontal
        if positive_shift & high_bit:
            distance += 1
        elif negative_shift & high_bit:
            distance -= 1
        positive_shift = ((positive_shift << 1) | 1) & bitmask
        negative_shift = (negative_shift << 1) & bitmask
        positive = (negative_shift | ~(vertical | positive_shift)) & bitmask
        negative = horizontal & positive_shift
    return distance


def _edit_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 1.0 - _levenshtein_distance(a, b) / max(len(a), len(b))


def _token_set_ratio(a: str, b: str) -> float:
    """Draft token-set ratio; deterministic stdlib approximation.

    The group registry proposes RapidFuzz token_set_ratio. This version uses
    SequenceMatcher and is NOT numerically identical to RapidFuzz for every
    input. Replace with the agreed similarity API before schema freeze.
    """

    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    common = sorted(tokens_a & tokens_b)
    left = " ".join(common + sorted(tokens_a - tokens_b))
    right = " ".join(common + sorted(tokens_b - tokens_a))
    if not common:
        return SequenceMatcher(None, left, right, autojunk=False).ratio()
    shared = " ".join(common)
    return max(
        SequenceMatcher(None, shared, left, autojunk=False).ratio(),
        SequenceMatcher(None, shared, right, autojunk=False).ratio(),
        SequenceMatcher(None, left, right, autojunk=False).ratio(),
    )


def _reference_quality(record: ReferenceRecord) -> tuple[int, int, int, str]:
    """Choose the most informative duplicate ref deterministically."""

    completeness = sum(
        bool(value)
        for value in (
            record.ref_title,
            record.ref_authors,
            record.ref_submission_date,
            record.ref_s2_id,
        )
    )
    return (
        completeness,
        len(record.ref_title or ""),
        len(record.ref_authors),
        repr((record.ref_title, record.ref_authors, record.ref_submission_date, record.ref_s2_id)),
    )


def _reference_year(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("year")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError("enriched reference year must be an int or None")
    return value


class FeatureBuilder:
    """Fit TF-IDF once on train, then generate 26 features per candidate."""

    def __init__(
        self,
        *,
        clean_text: Callable[[str], str] = default_clean_text,
        clean_author: Callable[[str], str] = default_clean_text,
    ) -> None:
        # Inject Quốc Học's frozen cleaning functions here when available.
        self.clean_text = clean_text
        self.clean_author = clean_author
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", lowercase=False
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=False
        )
        self.is_fitted = False

    def _title(self, title: str | None) -> str:
        if not title:
            return ""
        return " ".join(self.clean_text(title).split())

    def _authors(self, authors: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            cleaned
            for author in authors
            if (cleaned := " ".join(self.clean_author(author).split()))
        )

    def fit_on_train(
        self,
        bibtex_records: Iterable[BibtexRecord],
        reference_records: Iterable[ReferenceRecord],
        partition_by_paper: Mapping[str, Partition | str],
    ) -> "FeatureBuilder":
        """Fit both vectorizers using titles ONLY from train publications.

        The caller must pass all train records from all ranges together. Reject
        valid/test publications rather than silently introducing leakage.
        """

        if self.is_fitted:
            raise RuntimeError("TF-IDF already fitted; never refit on valid/test or by range")
        titles: set[str] = set()
        for record in chain(bibtex_records, reference_records):
            partition = partition_by_paper.get(record.arxiv_id)
            if partition is None or Partition(partition) is not Partition.TRAIN:
                raise ValueError(f"non-train or unassigned publication in TF-IDF fit: {record.arxiv_id}")
            title = record.title if isinstance(record, BibtexRecord) else record.ref_title
            normalized = self._title(title)
            if normalized:
                titles.add(normalized)
        if not titles:
            raise ValueError("cannot fit TF-IDF without any nonempty train title")
        documents = sorted(titles)
        self.word_vectorizer.fit(documents)
        self.char_vectorizer.fit(documents)
        self.is_fitted = True
        return self

    def build_rows(
        self,
        candidate_pairs: Iterable[CandidatePair],
        bibtex_records: Iterable[BibtexRecord],
        reference_records: Iterable[ReferenceRecord],
        partition_by_paper: Mapping[str, Partition | str],
        *,
        reference_metadata: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    ) -> Iterator[FeatureRow]:
        """Yield one versioned FeatureRow per candidate without storing pairs.

        ``reference_metadata`` is optional ENRICHMENT ONLY, keyed by
        (citing arxiv_id, ref_arxiv_id), with optional ``doi``, ``year``,
        ``venue`` fields. It must not contain ground truth or model labels.
        """

        if not self.is_fitted:
            raise RuntimeError("call fit_on_train() before building any feature rows")

        bibtex_by_key: dict[tuple[str, str], BibtexRecord] = {}
        for record in bibtex_records:
            key = (record.arxiv_id, record.citation_key)
            if key in bibtex_by_key:
                raise ValueError(f"duplicate BibTeX record: {key}")
            bibtex_by_key[key] = record

        reference_by_key: dict[tuple[str, str], ReferenceRecord] = {}
        for record in reference_records:
            key = (record.arxiv_id, record.ref_arxiv_id)
            old = reference_by_key.get(key)
            if old is None or _reference_quality(record) > _reference_quality(old):
                reference_by_key[key] = record

        # Clean metadata once per record, not once per candidate pair.
        bib_clean = {
            key: (self._title(record.title), self._authors(record.authors))
            for key, record in bibtex_by_key.items()
        }
        ref_clean = {
            key: (self._title(record.ref_title), self._authors(record.ref_authors))
            for key, record in reference_by_key.items()
        }

        # Transform each unique title once, then reuse its sparse TF-IDF row.
        normalized_titles = sorted(
            {
                title
                for title, _ in chain(bib_clean.values(), ref_clean.values())
                if title
            }
        )
        title_index = {title: position for position, title in enumerate(normalized_titles)}
        # sklearn rejects transform([]). If all titles are missing, cosine
        # immediately returns 0 and the sparse matrices are never accessed.
        word_vectors = self.word_vectorizer.transform(normalized_titles) if normalized_titles else None
        char_vectors = self.char_vectorizer.transform(normalized_titles) if normalized_titles else None

        def cosine(title_a: str, title_b: str, vectors: Any) -> float:
            if not title_a or not title_b:
                return 0.0
            a, b = title_index[title_a], title_index[title_b]
            # TfidfVectorizer returns L2-normalized rows, so dot == cosine.
            return min(1.0, max(0.0, float(vectors[a].multiply(vectors[b]).sum())))

        metadata_by_ref = reference_metadata or {}
        previous_key: tuple[str, str, str] | None = None
        for candidate in candidate_pairs:
            candidate_key = (candidate.arxiv_id, candidate.citation_key, candidate.ref_arxiv_id)
            # Candidate generator yields sorted pairs. A previous-key check
            # preserves O(1) validation memory rather than a huge seen set.
            if previous_key is not None and candidate_key <= previous_key:
                raise ValueError(f"candidate pairs must be sorted and unique: {candidate_key}")
            previous_key = candidate_key

            bib = bibtex_by_key.get(candidate_key[:2])
            ref = reference_by_key.get((candidate.arxiv_id, candidate.ref_arxiv_id))
            if bib is None or ref is None:
                raise ValueError(f"candidate has no matching same-publication source records: {candidate_key}")
            try:
                partition = Partition(partition_by_paper[candidate.arxiv_id])
            except KeyError as error:
                raise ValueError(f"missing partition for publication: {candidate.arxiv_id}") from error

            extra = metadata_by_ref.get((candidate.arxiv_id, candidate.ref_arxiv_id), {})
            forbidden = {"groundtruth_arxiv_id", "label", "label_source", "status", "annotator", "reviewer", "partition"}
            if forbidden.intersection(extra):
                raise ValueError("reference metadata must not contain label or ground-truth fields")
            values = self._pair_features(
                bib,
                extra,
                word_vectors,
                char_vectors,
                cosine,
                bib_clean[candidate_key[:2]],
                ref_clean[(candidate.arxiv_id, candidate.ref_arxiv_id)],
            )
            yield FeatureRow(
                arxiv_id=candidate.arxiv_id,
                citation_key=candidate.citation_key,
                ref_arxiv_id=candidate.ref_arxiv_id,
                partition=partition,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                values=values,
            )

    def _pair_features(
        self,
        bib: BibtexRecord,
        extra: Mapping[str, Any],
        word_vectors: Any,
        char_vectors: Any,
        cosine: Callable[[str, str, Any], float],
        bib_clean: tuple[str, tuple[str, ...]],
        ref_clean: tuple[str, tuple[str, ...]],
    ) -> dict[str, int | float]:
        title_b, authors_b = bib_clean
        title_r, authors_r = ref_clean
        tokens_b, tokens_r = _tokens(title_b), _tokens(title_r)
        common_titles, title_jaccard, title_containment = _set_overlap(tokens_b, tokens_r)
        del common_titles

        author_set_b, author_set_r = _author_set(authors_b), _author_set(authors_r)
        overlap, author_jaccard, _ = _set_overlap(author_set_b, author_set_r)
        bib_coverage = overlap / len(author_set_b) if authors_b and authors_r else 0.0
        author_sequence = _edit_similarity(_author_sequence(authors_b), _author_sequence(authors_r))

        ref_year = _reference_year(extra)
        year_missing = int(bib.year is None or ref_year is None)
        year_exact = int(not year_missing and bib.year == ref_year)
        year_abs_diff = abs(bib.year - ref_year) if not year_missing else 0
        doi_b, doi_r = bib.doi, normalize_doi(extra.get("doi"))
        doi_exact = int(bool(doi_b and doi_r and doi_b == doi_r))
        venue_b = self._title(bib.journal)
        venue_r = self._title(extra.get("venue"))

        title_exact = int(bool(title_b and title_r and title_b == title_r))
        title_edit = _edit_similarity(title_b, title_r)
        title_length = min(len(title_b), len(title_r)) / max(len(title_b), len(title_r)) if title_b and title_r else 0.0
        bib_title_missing, ref_title_missing = int(not title_b), int(not title_r)

        values: dict[str, int | float] = {
            "title_exact": title_exact,
            "title_jaccard": title_jaccard,
            "title_edit_similarity": title_edit,
            "title_token_set_ratio": _token_set_ratio(title_b, title_r),
            "title_word_tfidf_cosine": cosine(title_b, title_r, word_vectors),
            "title_char_tfidf_cosine": cosine(title_b, title_r, char_vectors),
            "title_length_ratio": title_length,
            "title_token_containment": title_containment,
            "first_author_match": int(bool(authors_b and authors_r and authors_b[0] == authors_r[0])),
            "last_author_match": int(bool(authors_b and authors_r and authors_b[-1] == authors_r[-1])),
            "author_overlap_count": overlap,
            "author_jaccard": author_jaccard,
            "author_bib_coverage": bib_coverage,
            "author_sequence_similarity": author_sequence,
            "doi_exact": doi_exact,
            "year_exact": year_exact,
            "year_abs_diff": year_abs_diff,
            "venue_similarity": _token_set_ratio(venue_b, venue_r),
            "bib_title_missing": bib_title_missing,
            "ref_title_missing": ref_title_missing,
            "bib_authors_missing": int(not authors_b),
            "ref_authors_missing": int(not authors_r),
            "year_missing": year_missing,
            "high_title_author": int(title_edit >= 0.90 and author_jaccard >= 0.50),
            "exact_title_year_conflict": int(title_exact == 1 and year_missing == 0 and year_exact == 0),
            "doi_match_title_missing": int(doi_exact == 1 and (bib_title_missing == 1 or ref_title_missing == 1)),
        }
        assert tuple(values) == FEATURE_NAMES, "feature names/order must match the registry"
        return values
