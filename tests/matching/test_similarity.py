import math
import unittest

from src.matching.cleaning import Author, clean_record, parse_authors, title_tokens, normalize_title
from src.matching.similarity import (
    COMPLETENESS_NAMES, FEATURE_NAMES, MISSING, author_jaccard, author_overlap, compare_records, first_author_match, is_missing,
    same_identifier, title_char_dice, title_contains, title_coverage, title_exact, title_jaccard,
    title_sequence_ratio, year_diff, record_completeness,
)

A = parse_authors


class TestTitleMeasures(unittest.TestCase):
    def test_identical_and_disjoint(self):
        t = title_tokens("Attention is all you need")
        self.assertEqual(title_jaccard(t, t), 1.0)
        self.assertEqual(title_jaccard(t, title_tokens("Deep residual learning")), 0.0)

    def test_jaccard_partial(self):
        self.assertAlmostEqual(title_jaccard(("a1", "b1", "c1"), ("b1", "c1", "d1")), 0.5)

    def test_coverage_is_asymmetric_and_survives_noise(self):
        bib = title_tokens("GPy: A Gaussian process framework in python")
        ref = title_tokens("GPy: A Gaussian process framework in python. Sheffield ML group, 2012, version 1.10 manual")
        self.assertEqual(title_coverage(bib, ref), 1.0)                 # all bib words found in the noisy title
        self.assertLess(title_coverage(ref, bib), 1.0)
        self.assertLess(title_jaccard(bib, ref), 1.0)                   # symmetric measure is hurt by the noise

    def test_noisy_wrong_reference_scores_low(self):
        bib = title_tokens("Periodically driven quantum systems: effective Hamiltonians and engineered gauge fields")
        ref = title_tokens("MNRAS, 487, 2005 Costanzi, M., Rozo, E., Simet, M., et al.")
        self.assertEqual(title_coverage(bib, ref), 0.0)

    def test_char_dice_tolerates_typos(self):
        a = normalize_title("Attention is all you need")
        self.assertEqual(title_char_dice(a, normalize_title("Attention Is All You Need.")), 1.0)
        self.assertGreater(title_char_dice(a, normalize_title("Atention is all you need")), 0.8)
        self.assertLess(title_char_dice(a, normalize_title("Deep residual learning")), 0.2)

    def test_sequence_ratio_and_exact_and_contains(self):
        a = "attention is all you need"
        self.assertEqual(title_sequence_ratio(a, a), 1.0)
        self.assertEqual(title_exact(a, a), 1.0)
        self.assertEqual(title_exact(a, "attention is all you want"), 0.0)
        self.assertEqual(title_contains(a, "vaswani 2017 attention is all you need nips"), 1.0)
        self.assertEqual(title_contains("short", "short title here"), 0.0)   # below min_chars

    def test_missing_is_nan(self):
        for value in (title_jaccard((), ("x1",)), title_coverage(("x1",), ()), title_char_dice("", "abc"),
                      title_sequence_ratio("abc", ""), title_exact("", ""), title_contains("", "abc")):
            self.assertTrue(math.isnan(value))


class TestAuthorMeasures(unittest.TestCase):
    def test_overlap_and_jaccard(self):
        bib = A("Vaswani, A. and Shazeer, N. and Parmar, N. and Uszkoreit, J.")
        ref = A("Ashish Vaswani, Noam Shazeer, Niki Parmar")
        self.assertEqual(author_overlap(bib, ref), 1.0)               # the shorter list is contained
        self.assertAlmostEqual(author_jaccard(bib, ref), 3 / 4)
        self.assertEqual(first_author_match(bib, ref), 1.0)

    def test_disjoint_and_first_author_mismatch(self):
        self.assertEqual(author_overlap(A("Smith, J."), A("Jones, B.")), 0.0)
        self.assertEqual(first_author_match(A("Smith, J. and Lee, C."), A("Lee, C. and Smith, J.")), 0.0)

    def test_initials_change_the_decision_only_when_both_have_them(self):
        self.assertEqual(first_author_match(A("Smith, J."), A("Smith, B."), use_initials=True), 0.0)
        self.assertEqual(first_author_match(A("Smith, J."), A("Smith, B."), use_initials=False), 1.0)
        self.assertEqual(first_author_match(A("Smith"), A("Smith, B."), use_initials=True), 1.0)

    def test_accent_and_particle_robustness(self):
        self.assertEqual(author_overlap(A(r"P{\"a}prer, Paul"), A("Paul Päprer")), 1.0)
        self.assertEqual(author_overlap(A("van der Waals, J."), A("J. Waals")), 1.0)

    def test_repeated_names_are_paired_once(self):
        self.assertAlmostEqual(author_jaccard(A("Lee, A. and Lee, A."), A("Lee, A.")), 0.5)
        self.assertEqual(author_overlap(A("Lee, A. and Lee, B."), A("Lee, A. and Kim, C.")), 0.5)

    def test_empty_authors_are_missing(self):
        for value in (author_overlap((), A("Smith, J.")), author_jaccard(A("Smith, J."), ()),
                      first_author_match((), ())):
            self.assertTrue(math.isnan(value))


class TestYearAndIds(unittest.TestCase):
    def test_year_diff(self):
        self.assertEqual(year_diff(2017, 2018), 1.0)
        self.assertEqual(year_diff(2018, 2017), 1.0)
        self.assertTrue(math.isnan(year_diff(None, 2017)))

    def test_same_identifier(self):
        self.assertEqual(same_identifier("10.1/x", "10.1/x"), 1.0)
        self.assertEqual(same_identifier("1706.03762", "1706.03763"), 0.0)
        self.assertTrue(math.isnan(same_identifier(None, "1706.03762")))
        self.assertTrue(math.isnan(same_identifier("", "")))

    def test_is_missing(self):
        self.assertTrue(is_missing(MISSING))
        self.assertTrue(is_missing(None))
        self.assertFalse(is_missing(0.0))


class TestCompareRecords(unittest.TestCase):
    def setUp(self):
        self.bib = clean_record(
            title=r"Attention {I}s {A}ll {Y}ou {N}eed", authors=r"Vaswani, Ashish and Shazeer, Noam",
            year="2017", extra_text=["arXiv:1706.03762"])
        self.ref = clean_record(
            title="Attention is All you Need", authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            year="2017-06-12", arxiv_id="1706.03762v5")

    def test_true_pair(self):
        f = compare_records(self.bib, self.ref)
        self.assertEqual(tuple(f), FEATURE_NAMES)
        self.assertEqual((f["title_exact"], f["title_jaccard"], f["title_cov_bib_in_ref"]), (1.0, 1.0, 1.0))
        self.assertEqual((f["author_overlap"], f["first_author_match"], f["first_author_match_initial"]), (1.0, 1.0, 1.0))
        self.assertEqual((f["year_diff"], f["arxiv_id_match"]), (0.0, 1.0))
        self.assertTrue(math.isnan(f["doi_match"]))                   # neither side has a DOI

    def test_wrong_pair_scores_lower_on_every_signal(self):
        other = clean_record(title="Deep residual learning for image recognition",
                             authors=["Kaiming He", "Xiangyu Zhang"], year="2015", arxiv_id="1512.03385")
        good, bad = compare_records(self.bib, self.ref), compare_records(self.bib, other)
        for name in ("title_jaccard", "title_char_dice", "title_seq_ratio", "author_overlap"):
            self.assertGreater(good[name], bad[name], name)
        self.assertEqual(bad["arxiv_id_match"], 0.0)
        self.assertEqual(bad["year_diff"], 2.0)

    def test_reference_without_authors_keeps_title_features(self):
        noisy = clean_record(title="Attention is All you Need. NeurIPS 30, 2017", authors=[], year="2017")
        f = compare_records(self.bib, noisy)
        self.assertTrue(math.isnan(f["author_overlap"]))
        self.assertEqual(f["title_cov_bib_in_ref"], 1.0)
        self.assertLess(f["title_exact"] + f["title_jaccard"], 2.0)


class TestRealWorldPairs(unittest.TestCase):
    def test_bib_with_latex_and_mojibake_matches_plain_reference(self):
        bib = clean_record(
            title=r"Dark Energy Survey Year 3 results: Constraints on extensions to \ensuremath\Lambda CDM with weak lensing and galaxy clustering",
            authors=["Abbott, T.~M.~C.", "FertÃ©, A.", r"Garc\'\ia-Bellido, J.", "da Costa, L.~N.", "DES Collaboration"],
            year=2023)
        ref = clean_record(
            title="Dark Energy Survey Year 3 results: Constraints on extensions to ΛCDM with weak lensing and galaxy clustering",
            authors=["T. M. C. Abbott", "Agnès Ferté", "Juan García-Bellido", "Luiz N. da Costa"], year="2023-02-01")
        f = compare_records(bib, ref)
        self.assertEqual((f["title_exact"], f["title_jaccard"]), (1.0, 1.0))
        self.assertEqual((f["author_overlap"], f["first_author_match"]), (1.0, 1.0))
        self.assertEqual(f["year_diff"], 0.0)

    def test_long_author_lists_are_fast_and_correct(self):
        import time
        names = [f"Family{i}, A.~B." for i in range(150)]
        a, b = parse_authors(names), parse_authors(names[:100] + [f"Other{i}, C." for i in range(50)])
        start = time.perf_counter()
        for _ in range(200):
            overlap = author_overlap(a, b)
        self.assertAlmostEqual(overlap, 100 / 150)
        self.assertLess(time.perf_counter() - start, 1.0)


class TestFastPath(unittest.TestCase):
    def setUp(self):
        self.bib = clean_record(title="Attention is all you need", authors="Vaswani, A. and Shazeer, N. and Lee, C.", year=2017)
        self.ref = clean_record(title="Attention is all you need. NeurIPS", authors=["A. Vaswani", "N. Shazeer"], year=2017)

    def test_shared_author_computation_matches_the_single_functions(self):
        f = compare_records(self.bib, self.ref)
        self.assertEqual(f["author_overlap"], author_overlap(self.bib.authors, self.ref.authors))
        self.assertEqual(f["author_jaccard"], author_jaccard(self.bib.authors, self.ref.authors))

    def test_without_sequence_ratio_only_that_feature_is_nan(self):
        full, fast = compare_records(self.bib, self.ref), compare_records(self.bib, self.ref, with_sequence_ratio=False)
        self.assertTrue(math.isnan(fast["title_seq_ratio"]))
        self.assertFalse(math.isnan(full["title_seq_ratio"]))
        for name in FEATURE_NAMES:
            if name != "title_seq_ratio":
                self.assertTrue(full[name] == fast[name] or (math.isnan(full[name]) and math.isnan(fast[name])), name)
        self.assertEqual(tuple(fast), FEATURE_NAMES)


class TestCompleteness(unittest.TestCase):
    def test_flags(self):
        full = clean_record(title="Attention is all you need", authors="Vaswani, A.", year="2017",
                            doi="10.1000/abc", arxiv_id="1706.03762")
        self.assertEqual(record_completeness(full), dict.fromkeys(COMPLETENESS_NAMES, 1.0))
        empty = clean_record(title="", authors=[], year=None)
        self.assertEqual(record_completeness(empty), dict.fromkeys(COMPLETENESS_NAMES, 0.0))

    def test_title_of_only_stopwords_counts_as_missing(self):
        self.assertEqual(record_completeness(clean_record(title="The of and"))["has_title"], 0.0)

    def test_missing_pattern_matches_nan_pattern(self):
        bib = clean_record(title="Attention is all you need", authors="Vaswani, A.", year="2017")
        ref = clean_record(title="Attention is all you need", authors=[], year="2017")
        flags, feats = record_completeness(ref), compare_records(bib, ref)
        self.assertEqual(flags["has_authors"], 0.0)
        self.assertTrue(math.isnan(feats["author_overlap"]))
        self.assertFalse(math.isnan(feats["title_jaccard"]))


if __name__ == "__main__":
    unittest.main()
