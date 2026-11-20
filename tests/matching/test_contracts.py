import pytest

from src.matching.contracts import LabelRecord, LabelSource, LabelStatus, PredictionRow, normalize_arxiv_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("arXiv:2609.7834v2", "2609.07834"),
        ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        (" null ", None),
    ],
)
def test_normalize_arxiv_id_uses_canonical_base_ids(raw, expected):
    assert normalize_arxiv_id(raw) == expected


def test_match_label_requires_a_groundtruth_id():
    with pytest.raises(ValueError, match="require groundtruth"):
        LabelRecord(
            arxiv_id="2609.07834",
            citation_key="vaswani2017",
            groundtruth_arxiv_id=None,
            label_source=LabelSource.MANUAL,
            status=LabelStatus.MATCH,
        )


def test_no_match_label_requires_a_null_groundtruth_id():
    with pytest.raises(ValueError, match="require groundtruth"):
        LabelRecord(
            arxiv_id="2609.07834",
            citation_key="vaswani2017",
            groundtruth_arxiv_id="1706.03762",
            label_source=LabelSource.MANUAL,
            status=LabelStatus.NO_MATCH,
        )


def test_prediction_rejects_duplicate_candidates():
    with pytest.raises(ValueError, match="duplicates"):
        PredictionRow(
            arxiv_id="2609.07834",
            partition="test",
            citation_key="vaswani2017",
            cand_1="1706.03762",
            cand_2="1706.03762",
        )


def test_prediction_rejects_a_gap_before_a_ranked_candidate():
    with pytest.raises(ValueError, match="contiguous"):
        PredictionRow(
            arxiv_id="2609.07834",
            partition="test",
            citation_key="vaswani2017",
            cand_2="1706.03762",
        )
