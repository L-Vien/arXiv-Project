"""Ranking and no-match metrics for Milestone 3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import zip_longest
from typing import Iterable, Sequence

from .contracts import normalize_arxiv_id


@dataclass(frozen=True)
class NoMatchMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def reciprocal_rank_at_k(
    ranked_candidates: Sequence[str | None],
    groundtruth_arxiv_id: str | None,
    *,
    k: int = 5,
) -> float:
    """Return reciprocal rank of the first matching candidate in the top ``k``."""

    _validate_k(k)
    groundtruth = normalize_arxiv_id(groundtruth_arxiv_id)
    if groundtruth is None:
        return 0.0
    for rank, candidate in enumerate(ranked_candidates[:k], start=1):
        if normalize_arxiv_id(candidate) == groundtruth:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(
    ranked_candidate_lists: Iterable[Sequence[str | None]],
    groundtruth_arxiv_ids: Iterable[str | None],
    *,
    k: int = 5,
    include_no_match_as_zero: bool = False,
) -> float:
    """Compute MRR, excluding no-match entries unless explicitly requested.

    The default reports MRR over entries with a known matching reference.
    ``include_no_match_as_zero`` is available for the alternative denominator
    convention while the official no-match treatment is being confirmed.
    """

    _validate_k(k)
    sentinel = object()
    total = 0.0
    evaluated = 0
    for ranked, groundtruth in zip_longest(
        ranked_candidate_lists,
        groundtruth_arxiv_ids,
        fillvalue=sentinel,
    ):
        if ranked is sentinel or groundtruth is sentinel:
            raise ValueError("ranked candidates and ground truths must have equal length")
        if groundtruth is None and not include_no_match_as_zero:
            continue
        total += reciprocal_rank_at_k(ranked, groundtruth, k=k)  # type: ignore[arg-type]
        evaluated += 1
    if evaluated == 0:
        raise ValueError("MRR is undefined when there are no evaluated matching entries")
    return total / evaluated


def evaluate_no_match(
    true_no_match: Iterable[bool],
    best_candidate_scores: Iterable[float | None],
    *,
    threshold: float,
) -> NoMatchMetrics:
    """Evaluate no-match decisions; absent candidates always predict no-match.

    A present best score predicts no-match only when it is strictly below the
    threshold. Score equality therefore predicts a match.
    """

    if isinstance(threshold, bool) or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")

    sentinel = object()
    true_positive = false_positive = false_negative = true_negative = 0
    for actual, score in zip_longest(
        true_no_match,
        best_candidate_scores,
        fillvalue=sentinel,
    ):
        if actual is sentinel or score is sentinel:
            raise ValueError("labels and best candidate scores must have equal length")
        if not isinstance(actual, bool):
            raise TypeError("true_no_match values must be booleans")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("candidate scores must be numbers or None")
            if not math.isfinite(score):
                raise ValueError("candidate scores must be finite")
        predicted = score is None or score < threshold  # type: ignore[operator]
        if actual and predicted:
            true_positive += 1
        elif not actual and predicted:
            false_positive += 1
        elif actual:
            false_negative += 1
        else:
            true_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return NoMatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )
