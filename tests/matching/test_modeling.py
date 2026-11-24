import pytest

pytest.importorskip("sklearn")

from src.matching.contracts import FeatureRow, LabelRecord
from src.matching.modeling import train_supervised_ranker


def _feature(ref_id, title_score, partition="train", citation_key="match"):
    return FeatureRow(
        arxiv_id="2609.07834",
        citation_key=citation_key,
        ref_arxiv_id=ref_id,
        partition=partition,
        feature_schema_version="1.0",
        values={"title_jaccard": title_score, "doi_exact": title_score == 1.0},
    )


def _label(key, truth):
    return LabelRecord(
        arxiv_id="2609.07834",
        citation_key=key,
        groundtruth_arxiv_id=truth,
        label_source="manual",
        status="match" if truth else "no_match",
    )


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_supervised_ranker_trains_and_scores_positive_candidates(model_type):
    training = [
        _feature("1706.03762", 1.0),
        _feature("1801.00001", 0.0),
        _feature("1706.03763", 0.9, citation_key="no-match"),
        _feature("1801.00002", 0.2, citation_key="no-match"),
    ]
    labels = [
        _label("match", "1706.03762"),
        _label("no-match", None),
    ]
    model = train_supervised_ranker(training, labels, model_type=model_type)

    scores = model.score(training[:2])

    assert scores[0].score > scores[1].score
    assert model.feature_schema_version == "1.0"


def test_supervised_ranker_ignores_non_training_rows_and_rejects_missing_truth_candidate():
    features = [
        _feature("1706.03762", 1.0, partition="valid"),
        _feature("1801.00001", 0.0, partition="valid"),
    ]
    with pytest.raises(ValueError, match="no training-partition"):
        train_supervised_ranker(features, [])

    training = [
        _feature("1706.03762", 1.0),
        _feature("1801.00001", 0.0),
    ]
    with pytest.raises(ValueError, match="missing from candidates"):
        train_supervised_ranker(training, [_label("match", "1901.00001")])
