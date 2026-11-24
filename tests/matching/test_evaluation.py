import pytest

from src.matching.evaluation import (
    evaluate_no_match,
    mean_reciprocal_rank_at_k,
    reciprocal_rank_at_k,
)


@pytest.mark.parametrize(
    ("candidates", "groundtruth", "expected"),
    [
        (["1706.03762"], "1706.03762", 1.0),
        (["1801.00001", "1706.03762"], "1706.03762", 0.5),
        (
            ["1801.00001", "1801.00002", "1801.00003", "1801.00004", "1706.03762"],
            "1706.03762",
            0.2,
        ),
        (["1801.00001"], "1706.03762", 0.0),
        ([], "1706.03762", 0.0),
    ],
)
def test_reciprocal_rank_at_five(candidates, groundtruth, expected):
    assert reciprocal_rank_at_k(candidates, groundtruth) == expected


def test_reciprocal_rank_normalizes_arxiv_ids():
    assert reciprocal_rank_at_k(["arXiv:1706.03762v2"], "1706.03762") == 1.0


def test_mrr_excludes_no_match_by_default_and_can_include_it_as_zero():
    ranked = [["1706.03762"], []]
    truth = ["1706.03762", None]

    assert mean_reciprocal_rank_at_k(ranked, truth) == 1.0
    assert mean_reciprocal_rank_at_k(
        ranked,
        truth,
        include_no_match_as_zero=True,
    ) == 0.5


def test_mrr_rejects_mismatched_input_lengths_and_empty_evaluation_set():
    with pytest.raises(ValueError, match="equal length"):
        mean_reciprocal_rank_at_k([["1706.03762"]], [])
    with pytest.raises(ValueError, match="no evaluated"):
        mean_reciprocal_rank_at_k([[]], [None])


def test_no_match_metrics_include_threshold_boundary_and_missing_candidates():
    metrics = evaluate_no_match(
        [True, False, True, False],
        [0.2, 0.2, None, 0.8],
        threshold=0.2,
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1
    assert metrics.true_negative == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_no_match_metrics_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        evaluate_no_match([True], [], threshold=0.5)
