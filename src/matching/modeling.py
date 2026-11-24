"""Supervised candidate matching models trained from M3 feature rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Protocol

from .contracts import FeatureRow, LabelRecord, LabelStatus, Partition
from .ranking import ScoredCandidate


ModelType = Literal["logistic_regression", "random_forest"]


class _ProbabilisticEstimator(Protocol):
    classes_: object

    def predict_proba(self, values: object) -> object:
        ...


@dataclass(frozen=True)
class SupervisedRanker:
    estimator: _ProbabilisticEstimator
    feature_names: tuple[str, ...]
    feature_schema_version: str
    model_type: ModelType

    def score(self, features: Iterable[FeatureRow]) -> list[ScoredCandidate]:
        """Return positive-class probabilities for each candidate feature row."""

        rows = list(features)
        if not rows:
            return []
        if any(row.feature_schema_version != self.feature_schema_version for row in rows):
            raise ValueError("feature schema version does not match the trained model")
        matrix = _feature_matrix(rows, self.feature_names)
        probabilities = self.estimator.predict_proba(matrix)
        classes = list(self.estimator.classes_)  # type: ignore[arg-type]
        if 1 not in classes:
            raise ValueError("trained estimator has no positive match class")
        positive_index = classes.index(1)
        return [
            ScoredCandidate(
                arxiv_id=row.arxiv_id,
                citation_key=row.citation_key,
                ref_arxiv_id=row.ref_arxiv_id,
                score=float(probabilities[index][positive_index]),
                title_similarity=_title_similarity(row),
            )
            for index, row in enumerate(rows)
        ]


def _title_similarity(row: FeatureRow) -> float:
    for name in (
        "title_similarity",
        "title_jaccard",
        "title_word_tfidf_cosine",
        "title_token_set_ratio",
        "title_edit_similarity",
        "title_char_tfidf_cosine",
    ):
        if name in row.values:
            value = row.values[name]
            if isinstance(value, bool):
                return float(value)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"feature {name!r} must have a finite numeric value")
            return float(value)
    return 0.0


def _feature_matrix(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        values = []
        for name in feature_names:
            raw_value = row.values.get(name, 0)
            if isinstance(raw_value, bool):
                value = float(raw_value)
            elif isinstance(raw_value, (int, float)):
                value = float(raw_value)
            else:
                raise TypeError(f"feature {name!r} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")
            values.append(value)
        matrix.append(values)
    return matrix


def train_supervised_ranker(
    features: Iterable[FeatureRow],
    labels: Iterable[LabelRecord],
    *,
    model_type: ModelType = "logistic_regression",
    random_state: int = 42,
) -> SupervisedRanker:
    """Fit on labeled training-publication candidate pairs only.

    A match label marks its ground-truth candidate positive and the other
    candidates from that citation negative. A no-match label marks all its
    candidates negative. Uncertain labels and unlabeled citations are ignored.
    """

    if model_type not in ("logistic_regression", "random_forest"):
        raise ValueError(f"unsupported model_type: {model_type!r}")

    training_rows = [row for row in features if row.partition is Partition.TRAIN]
    if not training_rows:
        raise ValueError("no training-partition feature rows were provided")

    label_by_query: dict[tuple[str, str], LabelRecord] = {}
    for label in labels:
        query = (label.arxiv_id, label.citation_key)
        if query in label_by_query:
            raise ValueError(f"duplicate label for citation: {query}")
        label_by_query[query] = label

    groups: dict[tuple[str, str], list[FeatureRow]] = {}
    keys: set[tuple[str, str, str]] = set()
    for row in training_rows:
        key = (row.arxiv_id, row.citation_key, row.ref_arxiv_id)
        if key in keys:
            raise ValueError(f"duplicate training feature row: {key}")
        keys.add(key)
        groups.setdefault((row.arxiv_id, row.citation_key), []).append(row)

    labeled_groups = {
        query: rows
        for query, rows in groups.items()
        if query in label_by_query and label_by_query[query].status is not LabelStatus.UNCERTAIN
    }
    if not labeled_groups:
        raise ValueError("no labeled training citations have candidate feature rows")

    versions = {row.feature_schema_version for rows in labeled_groups.values() for row in rows}
    if len(versions) != 1:
        raise ValueError("training rows must use exactly one feature schema version")
    feature_schema_version = next(iter(versions))

    training_examples: list[FeatureRow] = []
    targets: list[int] = []
    for query, rows in labeled_groups.items():
        label = label_by_query[query]
        candidate_ids = {row.ref_arxiv_id for row in rows}
        if label.status is LabelStatus.MATCH and label.groundtruth_arxiv_id not in candidate_ids:
            raise ValueError(f"ground-truth reference is missing from candidates for citation: {query}")
        for row in rows:
            training_examples.append(row)
            targets.append(
                int(
                    label.status is LabelStatus.MATCH
                    and row.ref_arxiv_id == label.groundtruth_arxiv_id
                )
            )

    if set(targets) != {0, 1}:
        raise ValueError("training data must contain both match and non-match candidates")
    feature_names = tuple(
        sorted({name for row in training_examples for name in row.values})
    )
    if not feature_names:
        raise ValueError("training feature rows contain no feature values")
    matrix = _feature_matrix(training_examples, feature_names)

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=200,
            n_jobs=1,
            random_state=random_state,
        )
    estimator.fit(matrix, targets)
    return SupervisedRanker(
        estimator=estimator,
        feature_names=feature_names,
        feature_schema_version=feature_schema_version,
        model_type=model_type,
    )
