"""Candidate-generation tests; compatible with pytest or stdlib unittest."""

from __future__ import annotations

import unittest

from src.matching.candidates import generate_candidate_pairs
from src.matching.contracts import BibtexRecord, CandidatePair, ReferenceRecord
from src.matching.validation import validate_candidate_pairs


PAPER_A = "2607.28694"
PAPER_B = "2608.04006"
REF_A = "1706.03762"
REF_B = "1512.03385"


def bib(paper: str, key: str) -> BibtexRecord:
    return BibtexRecord(arxiv_id=paper, citation_key=key, entry_type="article")


def ref(paper: str, target: str) -> ReferenceRecord:
    return ReferenceRecord(arxiv_id=paper, ref_arxiv_id=target)


class CandidateGenerationTests(unittest.TestCase):
    def test_typed_loader_records_and_publication_scope(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [bib(PAPER_B, "ref_c"), bib(PAPER_A, "ref_b"), bib(PAPER_A, "ref_a")],
                [ref(PAPER_B, REF_B), ref(PAPER_A, REF_A)],
            )
        )
        self.assertEqual(
            pairs,
            [
                CandidatePair(PAPER_A, "ref_a", REF_A),
                CandidatePair(PAPER_A, "ref_b", REF_A),
                CandidatePair(PAPER_B, "ref_c", REF_B),
            ],
        )
        self.assertTrue(all(isinstance(pair, CandidatePair) for pair in pairs))

    def test_exhaustive_m_by_n_generation(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [bib(PAPER_A, "z"), bib(PAPER_A, "a")],
                [ref(PAPER_A, REF_B), ref(PAPER_A, REF_A)],
            )
        )
        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            [(pair.citation_key, pair.ref_arxiv_id) for pair in pairs],
            [("a", REF_B), ("a", REF_A), ("z", REF_B), ("z", REF_A)],
        )

    def test_reference_deduplication_after_id_normalization(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [{"arxiv_id": "arXiv:2607.28694v2", "citation_key": " key "}],
                [
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": "arXiv:1706.03762v2"},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": REF_A},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": "https://arxiv.org/abs/1706.03762"},
                ],
            )
        )
        self.assertEqual(pairs, [CandidatePair(PAPER_A, "key", REF_A)])

    def test_skip_missing_reference_ids_and_no_fake_no_match(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [bib(PAPER_A, "entry")],
                [
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": None},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": ""},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": "   "},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": "null"},
                    {"arxiv_id": PAPER_A, "ref_arxiv_id": "None"},
                ],
            )
        )
        self.assertEqual(pairs, [])

    def test_zero_candidates_and_empty_bibtex(self) -> None:
        self.assertEqual(list(generate_candidate_pairs([bib(PAPER_A, "a")], [])), [])
        self.assertEqual(list(generate_candidate_pairs([], [ref(PAPER_A, REF_A)])), [])

    def test_references_from_other_publications_are_not_candidates(self) -> None:
        self.assertEqual(
            list(generate_candidate_pairs([bib(PAPER_A, "x")], [ref(PAPER_B, REF_A)])),
            [],
        )

    def test_duplicate_bibtex_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate BibTeX entry"):
            list(generate_candidate_pairs([bib(PAPER_A, "x"), bib(PAPER_A, " x ")], []))

    def test_same_citation_key_in_different_publications_is_valid(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [bib(PAPER_A, "same"), bib(PAPER_B, "same")],
                [ref(PAPER_A, REF_A), ref(PAPER_B, REF_B)],
            )
        )
        self.assertEqual(len(pairs), 2)

    def test_invalid_publication_and_reference_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list(generate_candidate_pairs([{"arxiv_id": "", "citation_key": "x"}], []))
        with self.assertRaises(ValueError):
            list(generate_candidate_pairs([bib(PAPER_A, "x")], [{"arxiv_id": PAPER_A, "ref_arxiv_id": "invalid"}]))
        with self.assertRaises(TypeError):
            list(generate_candidate_pairs([bib(PAPER_A, "x")], [{"arxiv_id": PAPER_A, "ref_arxiv_id": 123}]))

    def test_stable_output_for_generators_in_different_orders(self) -> None:
        entries = [bib(PAPER_A, "b"), bib(PAPER_A, "a")]
        refs = [ref(PAPER_A, REF_A), ref(PAPER_A, REF_B)]
        first = list(generate_candidate_pairs(iter(entries), iter(refs)))
        second = list(generate_candidate_pairs(iter(reversed(entries)), iter(reversed(refs))))
        self.assertEqual(first, second)

    def test_output_passes_team_validation(self) -> None:
        pairs = list(
            generate_candidate_pairs(
                [bib(PAPER_A, "entry")],
                [ref(PAPER_A, REF_A)],
            )
        )
        report = validate_candidate_pairs(
            pairs,
            bibtex_keys={(PAPER_A, "entry")},
            reference_keys={(PAPER_A, REF_A)},
        )
        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.checked_records, 1)


if __name__ == "__main__":
    unittest.main()
