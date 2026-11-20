from __future__ import annotations

import math
import random
import unittest

from src.matching.candidates import generate_candidate_pairs
from src.matching.contracts import BibtexRecord, CandidatePair, FeatureRow, Partition, ReferenceRecord
from src.matching.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureBuilder,
    _levenshtein_distance,
    default_clean_text,
)


TRAIN = "2607.28694"
VALID = "2608.04006"
TEST = "2608.04007"
TARGET = "1706.03762"
SECOND_TARGET = "1512.03385"
PARTITIONS = {TRAIN: Partition.TRAIN, VALID: Partition.VALID, TEST: Partition.TEST}


def bib(paper: str, key: str, **kwargs: object) -> BibtexRecord:
    return BibtexRecord(arxiv_id=paper, citation_key=key, entry_type="article", **kwargs)


def ref(paper: str, target: str, **kwargs: object) -> ReferenceRecord:
    return ReferenceRecord(arxiv_id=paper, ref_arxiv_id=target, **kwargs)


def trained_builder() -> FeatureBuilder:
    builder = FeatureBuilder()
    builder.fit_on_train(
        [bib(TRAIN, "train", title="Attention Is All You Need")],
        [ref(TRAIN, TARGET, ref_title="Deep Learning")],
        PARTITIONS,
    )
    return builder


def one_row(
    builder: FeatureBuilder,
    b: BibtexRecord,
    r: ReferenceRecord,
    *,
    metadata: dict[tuple[str, str], dict[str, object]] | None = None,
) -> FeatureRow:
    return next(
        builder.build_rows(
            [CandidatePair(b.arxiv_id, b.citation_key, r.ref_arxiv_id)],
            [b],
            [r],
            PARTITIONS,
            reference_metadata=metadata,
        )
    )


class FeatureTests(unittest.TestCase):
    def test_fast_levenshtein_preserves_exact_unicode_distance(self) -> None:
        def dynamic_programming(left: str, right: str) -> int:
            previous = list(range(len(right) + 1))
            for i, char_left in enumerate(left, 1):
                current = [i]
                for j, char_right in enumerate(right, 1):
                    current.append(min(
                        current[j - 1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + (char_left != char_right),
                    ))
                previous = current
            return previous[-1]

        examples = [
            ("", ""),
            ("", "arxiv"),
            ("kitten", "sitting"),
            ("attention is all you need", "attention is all we need"),
            ("đại học quốc gia", "đại học khoa học"),
            ("模型學習", "模型机器学习"),
            ("🍎🙂", "🍊🙂"),
            ("abc" * 80, "adc" * 80),
        ]
        rng = random.Random(20261120)
        alphabet = "abc deé中🙂"
        examples.extend(
            (
                "".join(rng.choices(alphabet, k=rng.randrange(0, 75))),
                "".join(rng.choices(alphabet, k=rng.randrange(0, 75))),
            )
            for _ in range(100)
        )
        for left, right in examples:
            with self.subTest(left=left[:20], right=right[:20]):
                self.assertEqual(_levenshtein_distance(left, right), dynamic_programming(left, right))
                self.assertEqual(_levenshtein_distance(right, left), dynamic_programming(left, right))

    def test_end_to_end_from_candidate_generator(self) -> None:
        b = bib(TRAIN, "a", title="Deep Learning", authors=("Alice",))
        references = [
            ref(TRAIN, TARGET, ref_title="Deep Learning", ref_authors=("Alice",)),
            ref(TRAIN, SECOND_TARGET, ref_title="Unrelated Paper", ref_authors=("Bob",)),
        ]
        builder = FeatureBuilder().fit_on_train([b], references, PARTITIONS)
        pairs = generate_candidate_pairs([b], references)
        rows = list(builder.build_rows(pairs, [b], references, PARTITIONS))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.ref_arxiv_id for row in rows], [SECOND_TARGET, TARGET])
        self.assertEqual(rows[0].values["title_exact"], 0)
        self.assertEqual(rows[1].values["title_exact"], 1)

    def test_registry_has_exactly_26_unique_feature_names(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, "1.0")
        self.assertEqual(len(FEATURE_NAMES), 26)
        self.assertEqual(len(set(FEATURE_NAMES)), 26)

    def test_exact_normalized_title_and_author_features(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title="Attention, IS All You Need!", authors=("Vaswani, Ashish", "Shazeer, Noam")),
            ref(TRAIN, TARGET, ref_title="attention is all you need", ref_authors=("Vaswani, Ashish", "Shazeer, Noam")),
        )
        features = row.values
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual((row.arxiv_id, row.citation_key, row.ref_arxiv_id), (TRAIN, "a", TARGET))
        self.assertEqual(row.partition, Partition.TRAIN)
        self.assertEqual(row.feature_schema_version, FEATURE_SCHEMA_VERSION)
        for name in ("title_exact", "title_jaccard", "title_edit_similarity", "title_token_set_ratio", "title_length_ratio", "title_token_containment", "first_author_match", "last_author_match", "author_jaccard", "author_bib_coverage", "author_sequence_similarity", "high_title_author"):
            self.assertEqual(features[name], 1, name)
        self.assertEqual(features["author_overlap_count"], 2)
        self.assertAlmostEqual(features["title_word_tfidf_cosine"], 1.0, places=6)
        self.assertAlmostEqual(features["title_char_tfidf_cosine"], 1.0, places=6)

    def test_title_length_and_containment_are_not_the_same_metric(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title="Deep Learning"),
            ref(TRAIN, TARGET, ref_title="Deep Learning for Images"),
        )
        self.assertEqual(row.values["title_token_containment"], 1.0)
        self.assertLess(row.values["title_length_ratio"], 1.0)
        self.assertLess(row.values["title_jaccard"], 1.0)

    def test_author_bib_coverage_is_asymmetric(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title="Deep Learning", authors=("Alice", "Bob")),
            ref(TRAIN, TARGET, ref_title="Deep Learning", ref_authors=("Alice",)),
        )
        self.assertEqual(row.values["author_overlap_count"], 1)
        self.assertEqual(row.values["author_jaccard"], 0.5)
        self.assertEqual(row.values["author_bib_coverage"], 0.5)
        self.assertEqual(row.values["first_author_match"], 1)
        self.assertEqual(row.values["last_author_match"], 0)

    def test_doi_year_venue_use_explicit_reference_enrichment(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title="Deep Learning", year=2017, doi="10.1000/ABC", journal="NeurIPS"),
            ref(TRAIN, TARGET, ref_title="Deep Learning"),
            metadata={(TRAIN, TARGET): {"doi": "https://doi.org/10.1000/abc", "year": 2018, "venue": "NeurIPS"}},
        )
        self.assertEqual(row.values["doi_exact"], 1)
        self.assertEqual(row.values["year_exact"], 0)
        self.assertEqual(row.values["year_abs_diff"], 1)
        self.assertEqual(row.values["year_missing"], 0)
        self.assertEqual(row.values["venue_similarity"], 1)
        self.assertEqual(row.values["exact_title_year_conflict"], 1)

    def test_submission_date_is_not_assumed_to_be_publication_year(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title="A Study", year=2017, doi="10.1000/abc", journal="NeurIPS"),
            ref(TRAIN, TARGET, ref_title="A Study", ref_submission_date="2017-06-12"),
        )
        self.assertEqual(row.values["year_missing"], 1)
        self.assertEqual(row.values["year_exact"], 0)
        self.assertEqual(row.values["year_abs_diff"], 0)
        self.assertEqual(row.values["doi_exact"], 0)
        self.assertEqual(row.values["venue_similarity"], 0)
        self.assertEqual(row.values["exact_title_year_conflict"], 0)

    def test_doi_match_title_missing_interaction(self) -> None:
        row = one_row(
            trained_builder(),
            bib(TRAIN, "a", title=None, doi="doi:10.1000/abc"),
            ref(TRAIN, TARGET, ref_title="A Study"),
            metadata={(TRAIN, TARGET): {"doi": "10.1000/abc"}},
        )
        self.assertEqual(row.values["doi_exact"], 1)
        self.assertEqual(row.values["bib_title_missing"], 1)
        self.assertEqual(row.values["ref_title_missing"], 0)
        self.assertEqual(row.values["doi_match_title_missing"], 1)
        for feature in ("title_word_tfidf_cosine", "title_char_tfidf_cosine", "title_exact"):
            self.assertEqual(row.values[feature], 0)

    def test_both_titles_and_authors_missing(self) -> None:
        row = one_row(trained_builder(), bib(TRAIN, "a"), ref(TRAIN, TARGET))
        for name in ("bib_title_missing", "ref_title_missing", "bib_authors_missing", "ref_authors_missing", "year_missing"):
            self.assertEqual(row.values[name], 1)
        self.assertEqual(row.values["title_edit_similarity"], 0)
        self.assertEqual(row.values["author_jaccard"], 0)

    def test_train_only_vocabulary_and_valid_transform(self) -> None:
        builder = trained_builder()
        self.assertNotIn("validationonlyword", builder.word_vectorizer.vocabulary_)
        row = one_row(
            builder,
            bib(VALID, "a", title="validationonlyword"),
            ref(VALID, TARGET, ref_title="validationonlyword"),
        )
        self.assertEqual(row.partition, Partition.VALID)
        self.assertEqual(row.values["title_exact"], 1)
        self.assertEqual(row.values["title_word_tfidf_cosine"], 0)
        self.assertNotIn("validationonlyword", builder.word_vectorizer.vocabulary_)

    def test_fit_rejects_valid_and_test_titles(self) -> None:
        for paper in (VALID, TEST):
            with self.subTest(paper=paper):
                with self.assertRaisesRegex(ValueError, "non-train"):
                    FeatureBuilder().fit_on_train([bib(paper, "x", title="Leakage")], [], PARTITIONS)

    def test_refitting_or_building_before_fit_is_rejected(self) -> None:
        builder = trained_builder()
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            builder.fit_on_train([bib(TRAIN, "x", title="Leakage")], [], PARTITIONS)
        with self.assertRaisesRegex(RuntimeError, "fit_on_train"):
            list(FeatureBuilder().build_rows([], [], [], PARTITIONS))

    def test_no_nonempty_train_titles_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "without any nonempty train title"):
            FeatureBuilder().fit_on_train([bib(TRAIN, "x")], [ref(TRAIN, TARGET)], PARTITIONS)

    def test_duplicate_references_choose_richest_deterministically(self) -> None:
        builder = trained_builder()
        b = bib(TRAIN, "a", title="Attention Is All You Need")
        minimal = ref(TRAIN, TARGET)
        detailed = ref(TRAIN, TARGET, ref_title="Attention Is All You Need", ref_authors=("Vaswani",))
        pair = [CandidatePair(TRAIN, "a", TARGET)]
        first = list(builder.build_rows(pair, [b], [minimal, detailed], PARTITIONS))[0]
        second = list(builder.build_rows(pair, [b], [detailed, minimal], PARTITIONS))[0]
        self.assertEqual(dict(first.values), dict(second.values))
        self.assertEqual(first.values["ref_title_missing"], 0)

    def test_duplicate_bibtex_is_rejected(self) -> None:
        builder = trained_builder()
        b = bib(TRAIN, "a", title="Deep Learning")
        with self.assertRaisesRegex(ValueError, "duplicate BibTeX"):
            list(builder.build_rows([], [b, b], [], PARTITIONS))

    def test_missing_reference_or_wrong_publication_is_rejected(self) -> None:
        builder = trained_builder()
        with self.assertRaisesRegex(ValueError, "same-publication"):
            list(builder.build_rows(
                [CandidatePair(TRAIN, "a", TARGET)],
                [bib(TRAIN, "a")],
                [ref(VALID, TARGET)],
                PARTITIONS,
            ))

    def test_duplicate_or_unsorted_candidates_are_rejected(self) -> None:
        builder = trained_builder()
        a = CandidatePair(TRAIN, "a", TARGET)
        b = CandidatePair(TRAIN, "a", SECOND_TARGET)
        refs = [ref(TRAIN, TARGET), ref(TRAIN, SECOND_TARGET)]
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            list(builder.build_rows([a, a], [bib(TRAIN, "a")], refs, PARTITIONS))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            list(builder.build_rows([a, b], [bib(TRAIN, "a")], refs, PARTITIONS))

    def test_no_candidate_returns_no_feature_rows(self) -> None:
        builder = trained_builder()
        self.assertEqual(list(builder.build_rows([], [bib(TRAIN, "a")], [], PARTITIONS)), [])

    def test_no_labels_leak_into_numeric_features(self) -> None:
        row = one_row(trained_builder(), bib(TRAIN, "a", title="Deep Learning"), ref(TRAIN, TARGET, ref_title="Deep Learning"))
        self.assertEqual(set(row.values), set(FEATURE_NAMES))
        self.assertNotIn("partition", row.values)
        self.assertNotIn("groundtruth_arxiv_id", row.values)
        self.assertTrue(all(isinstance(value, (int, float, bool)) and math.isfinite(value) for value in row.values.values()))
        for feature in ("title_exact", "title_jaccard", "title_edit_similarity", "title_token_set_ratio", "title_word_tfidf_cosine", "title_char_tfidf_cosine", "title_length_ratio", "title_token_containment", "author_jaccard", "author_bib_coverage", "author_sequence_similarity", "venue_similarity"):
            self.assertGreaterEqual(row.values[feature], 0, feature)
            self.assertLessEqual(row.values[feature], 1, feature)

    def test_reject_label_fields_in_reference_enrichment(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain label"):
            one_row(
                trained_builder(),
                bib(TRAIN, "a", title="Deep Learning"),
                ref(TRAIN, TARGET, ref_title="Deep Learning"),
                metadata={(TRAIN, TARGET): {"groundtruth_arxiv_id": TARGET}},
            )

    def test_cleaner_can_be_replaced_with_group_cleaning_api(self) -> None:
        self.assertEqual(default_clean_text("  ATTENTION—Is ALL you need! "), "attention is all you need")
        customized = FeatureBuilder(clean_text=lambda value: "same title", clean_author=lambda value: "same author")
        customized.fit_on_train([bib(TRAIN, "x", title="Any")], [ref(TRAIN, TARGET, ref_title="Other")], PARTITIONS)
        row = one_row(
            customized,
            bib(TRAIN, "a", title="Very different", authors=("Alice",)),
            ref(TRAIN, TARGET, ref_title="Unrelated", ref_authors=("Bob",)),
        )
        self.assertEqual(row.values["title_exact"], 1)
        self.assertEqual(row.values["first_author_match"], 1)


if __name__ == "__main__":
    unittest.main()