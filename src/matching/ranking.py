"""Deterministic candidate ranking and non-supervised baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .contracts import BibtexRecord, FeatureRow, ReferenceRecord, normalize_arxiv_id


QueryKey = tuple[str, str]
FeatureValues = Mapping[str, float | int | bool]


@dataclass(frozen=True)
class ScoredCandidate:
    arxiv_id: str
    citation_key: str
    ref_arxiv_id: str
    score: float
    title_similarity: float = 0.0

    def __post_init__(self) -> None:
        source_id = normalize_arxiv_id(self.arxiv_id)
        reference_id = normalize_arxiv_id(self.ref_arxiv_id)
        if source_id is None or reference_id is None:
            raise ValueError("candidate arXiv IDs must not be missing")
        if not isinstance(self.citation_key, str) or not self.citation_key.strip():
            raise ValueError("citation_key must be a non-empty string")
        for name in ("score", "title_similarity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "arxiv_id", source_id)
        object.__setattr__(self, "ref_arxiv_id", reference_id)
        object.__setattr__(self, "citation_key", self.citation_key.strip())

    @property
    def query_key(self) -> QueryKey:
        return (self.arxiv_id, self.citation_key)


@dataclass(frozen=True)
class RuleWeights:
    title: float = 0.40
    author: float = 0.25
    year: float = 0.10
    doi: float = 0.25

    def __post_init__(self) -> None:
        weights = (self.title, self.author, self.year, self.doi)
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float)) for weight in weights):
            raise TypeError("rule weights must be numbers")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("rule weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one rule weight must be positive")


def _feature_values(features: FeatureRow | FeatureValues) -> FeatureValues:
    return features.values if isinstance(features, FeatureRow) else features


def _signal(values: FeatureValues, names: Sequence[str]) -> float:
    for name in names:
        if name in values:
            value = values[name]
            if isinstance(value, bool):
                return float(value)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"feature {name!r} must have a finite numeric value")
            return float(value)
    return 0.0


def rule_based_score(
    features: FeatureRow | FeatureValues,
    *,
    weights: RuleWeights = RuleWeights(),
) -> float:
    """Combine the versioned title, author, year, and DOI feature families."""

    values = _feature_values(features)
    title = _signal(
        values,
        (
            "title_similarity",
            "title_jaccard",
            "title_word_tfidf_cosine",
            "title_token_set_ratio",
            "title_edit_similarity",
            "title_char_tfidf_cosine",
        ),
    )
    author = max(
        _signal(values, ("author_jaccard",)),
        _signal(values, ("author_bib_coverage",)),
        _signal(values, ("first_author_match",)),
    )
    year = _signal(values, ("year_exact",))
    if "year_exact" not in values and "year_abs_diff" in values:
        difference = _signal(values, ("year_abs_diff",))
        if difference < 0:
            raise ValueError("year_abs_diff must be non-negative")
        year = 1.0 / (1.0 + difference)
    doi = _signal(values, ("doi_exact",))
    return (
        weights.title * title
        + weights.author * author
        + weights.year * year
        + weights.doi * doi
    ) / sum((weights.title, weights.author, weights.year, weights.doi))


def _candidate_title_signal(features: FeatureRow) -> float:
    return _signal(
        features.values,
        (
            "title_similarity",
            "title_jaccard",
            "title_word_tfidf_cosine",
            "title_token_set_ratio",
            "title_edit_similarity",
            "title_char_tfidf_cosine",
        ),
    )


def _sort_and_deduplicate(candidates: Iterable[ScoredCandidate]) -> list[ScoredCandidate]:
    best_by_reference: dict[str, ScoredCandidate] = {}
    for candidate in candidates:
        previous = best_by_reference.get(candidate.ref_arxiv_id)
        if previous is None or _sort_key(candidate) < _sort_key(previous):
            best_by_reference[candidate.ref_arxiv_id] = candidate
    return sorted(best_by_reference.values(), key=_sort_key)


def _sort_key(candidate: ScoredCandidate) -> tuple[float, float, str]:
    return (-candidate.score, -candidate.title_similarity, candidate.ref_arxiv_id)


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
    ):
        raise ValueError("top_k must be a non-negative integer or None")


def rank_candidates(
    candidates: Iterable[ScoredCandidate],
    *,
    top_k: int | None = 5,
) -> list[ScoredCandidate]:
    """Sort and deduplicate one citation's candidates with deterministic ties."""

    _validate_top_k(top_k)
    items = list(candidates)
    query_keys = {candidate.query_key for candidate in items}
    if len(query_keys) > 1:
        raise ValueError("rank_candidates accepts candidates for exactly one citation")
    ranked = _sort_and_deduplicate(items)
    return ranked if top_k is None else ranked[:top_k]


def rank_candidate_groups(
    candidates: Iterable[ScoredCandidate],
    *,
    top_k: int | None = 5,
) -> dict[QueryKey, list[ScoredCandidate]]:
    """Group and rank candidates by ``(arxiv_id, citation_key)``."""

    _validate_top_k(top_k)
    grouped: dict[QueryKey, list[ScoredCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.query_key, []).append(candidate)
    return {
        key: rank_candidates(group, top_k=top_k)
        for key, group in sorted(grouped.items())
    }


def rank_with_rule_baseline(
    features: Iterable[FeatureRow],
    *,
    weights: RuleWeights = RuleWeights(),
    top_k: int | None = 5,
) -> dict[QueryKey, list[ScoredCandidate]]:
    """Score feature rows with the rule baseline and return grouped rankings."""

    scored = [
        ScoredCandidate(
            arxiv_id=row.arxiv_id,
            citation_key=row.citation_key,
            ref_arxiv_id=row.ref_arxiv_id,
            score=rule_based_score(row, weights=weights),
            title_similarity=_candidate_title_signal(row),
        )
        for row in features
    ]
    return rank_candidate_groups(scored, top_k=top_k)


class TfidfTitleRanker:
    """TF-IDF title retrieval fitted only on explicitly supplied training rows."""

    def __init__(self) -> None:
        self._vectorizer = None

    def fit(
        self,
        training_bibtex_records: Iterable[BibtexRecord],
        training_reference_records: Iterable[ReferenceRecord],
    ) -> "TfidfTitleRanker":
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [
            record.title
            for record in training_bibtex_records
            if record.title is not None and record.title.strip()
        ]
        texts.extend(
            record.ref_title
            for record in training_reference_records
            if record.ref_title is not None and record.ref_title.strip()
        )
        if not texts:
            raise ValueError("at least one non-empty training title is required")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
        )
        try:
            vectorizer.fit(texts)
        except ValueError as error:
            raise ValueError("training titles must contain at least one token") from error
        self._vectorizer = vectorizer
        return self

    def rank(
        self,
        bibtex_record: BibtexRecord,
        candidates: Iterable[ReferenceRecord],
        *,
        top_k: int | None = 5,
    ) -> list[ScoredCandidate]:
        from sklearn.metrics.pairwise import cosine_similarity

        _validate_top_k(top_k)
        if self._vectorizer is None:
            raise RuntimeError("fit the TF-IDF ranker on training titles before ranking")

        unique: dict[str, ReferenceRecord] = {}
        for candidate in candidates:
            if candidate.arxiv_id != bibtex_record.arxiv_id:
                raise ValueError("TF-IDF candidates must belong to the BibTeX publication")
            previous = unique.get(candidate.ref_arxiv_id)
            if previous is None or (candidate.ref_title or "").casefold() < (
                previous.ref_title or ""
            ).casefold():
                unique[candidate.ref_arxiv_id] = candidate

        ordered = [unique[key] for key in sorted(unique)]
        if not ordered:
            return []
        query = self._vectorizer.transform([bibtex_record.title or ""])
        candidate_titles = self._vectorizer.transform(
            [candidate.ref_title or "" for candidate in ordered]
        )
        similarities = cosine_similarity(query, candidate_titles).ravel()
        scored = [
            ScoredCandidate(
                arxiv_id=bibtex_record.arxiv_id,
                citation_key=bibtex_record.citation_key,
                ref_arxiv_id=candidate.ref_arxiv_id,
                score=float(similarity),
                title_similarity=float(similarity),
            )
            for candidate, similarity in zip(ordered, similarities)
        ]
        return rank_candidates(scored, top_k=top_k)
