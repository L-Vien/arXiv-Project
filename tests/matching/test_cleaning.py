import math
import unittest

from src.matching.cleaning import (
    Author, arxiv_id_to_key, clean_record, extract_arxiv_id, fix_mojibake, extract_doi, extract_year, find_arxiv_id,
    latex_to_text, normalize_arxiv_id, normalize_doi, normalize_title, parse_author_name, parse_authors,
    title_tokens,
)


def fam(authors):
    return [(a.family, a.initials) for a in authors]


class TestLatexToText(unittest.TestCase):
    def test_accents_from_real_bib_entries(self):
        self.assertEqual(latex_to_text(r'P{\"a}prer, Paul'), "Päprer, Paul")
        self.assertEqual(latex_to_text(r"Jean-Fran{\c{c}}ois"), "Jean-François")
        self.assertEqual(latex_to_text(r'Hungerl{\"a}nder'), "Hungerländer")

    def test_other_accent_forms(self):
        self.assertEqual(latex_to_text(r"J{\o}rgensen"), "Jørgensen")
        self.assertEqual(latex_to_text(r"Stra{\ss}e"), "Straße")
        self.assertEqual(latex_to_text(r"Fran\c cois"), "François")
        self.assertEqual(latex_to_text(r"{\'\i}"), "í")
        self.assertEqual(latex_to_text(r"\v{S}koda and \H{o}"), "Škoda and ő")

    def test_braces_escapes_math_and_commands(self):
        self.assertEqual(latex_to_text(r"Journal of Rail Transport Planning \& Management"),
                         "Journal of Rail Transport Planning & Management")
        self.assertEqual(latex_to_text(r"{{Floquet}} \textbf{x} $\mathbb{Z}_2$"), "Floquet x Z2")
        self.assertEqual(latex_to_text(r"IEEE Visualization Course \#1 Notes"), "IEEE Visualization Course #1 Notes")
        self.assertEqual(latex_to_text("pages 203--205"), "pages 203-205")
        self.assertEqual(latex_to_text(r"costs \$5"), "costs $5")

    def test_empty_inputs(self):
        self.assertEqual(latex_to_text(None), "")
        self.assertEqual(latex_to_text(""), "")


class TestTitles(unittest.TestCase):
    def test_normalize_title_from_bbl(self):
        raw = r"Periodically {{Driven Quantum Systems}}: {{Effective Hamiltonians}} and {{Engineered Gauge Fields}}"
        self.assertEqual(normalize_title(raw),
                         "periodically driven quantum systems effective hamiltonians and engineered gauge fields")

    def test_normalize_title_from_bib_with_protected_letters(self):
        raw = r"Cyclic scheduling: Assignment and routing {P}art {I} - A statement of the {F}risco railroad"
        self.assertEqual(normalize_title(raw),
                         "cyclic scheduling assignment and routing part i a statement of the frisco railroad")

    def test_accents_are_folded(self):
        self.assertEqual(normalize_title(r"Sch{\"o}dinger and Fran{\c{c}}ois"), "schodinger and francois")

    def test_tokens_drop_stopwords_and_single_letters(self):
        self.assertEqual(title_tokens("The rolling stock circulation planning problem: a tutorial with case study"),
                         ("rolling", "stock", "circulation", "planning", "problem", "tutorial", "case", "study"))
        self.assertEqual(title_tokens("A study of x and 2", drop_stopwords=False), ("study", "of", "and", "2"))

    def test_empty(self):
        self.assertEqual(normalize_title(None), "")
        self.assertEqual(title_tokens(""), ())


class TestArxivIds(unittest.TestCase):
    def test_normalize_whole_field(self):
        for raw, want in [
            ("arXiv:1706.03762v2", "1706.03762"), ("1706.03762", "1706.03762"),
            ("https://arxiv.org/abs/2008.05489", "2008.05489"), ("arxiv.org/pdf/2108.09314v3.pdf", "2108.09314"),
            ("hep-th/9901001v1", "hep-th/9901001"), ("arXiv:1403.0592 [cond-mat]", "1403.0592"),
            ("1706-03762", "1706.03762"), ("hep-th_9901001", "hep-th/9901001"), ("  2301.00001  ", "2301.00001"),
        ]:
            self.assertEqual(normalize_arxiv_id(raw), want, raw)

    def test_normalize_rejects_non_ids(self):
        for raw in [None, "", "10.1016/j.cie.2022.108625", "MNRAS, 487, 2005", "9999.12345", "1713.00001", "abc"]:
            self.assertIsNone(normalize_arxiv_id(raw), raw)

    def test_extract_from_real_bbl_fragments(self):
        self.assertEqual(extract_arxiv_id(r"\Eprint {https://arxiv.org/abs/2008.05489} {arXiv:2008.05489}"), "2008.05489")
        self.assertEqual(extract_arxiv_id(r"\href {http://arxiv.org/abs/2108.09314}"), "2108.09314")
        self.assertEqual(extract_arxiv_id("{arXiv:1403.0592 [cond-mat]}"), "1403.0592")
        self.assertEqual(extract_arxiv_id("see arXiv 2301.00001v2 for details"), "2301.00001")
        self.assertEqual(extract_arxiv_id("preprint hep-th/9901001 (1999)"), "hep-th/9901001")

    def test_extract_has_no_false_positives(self):
        for text in [
            r"eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/nav.3800040307}",   # real .bib eprint URL
            "10.1016/j.cie.2022.108625", "https://doi.org/10.1103/PhysRevX.4.031027", "MNRAS, 487, 2005",
            "Phys. Rev. B 2022.12345", "pages 1706.03762", "",
        ]:
            self.assertIsNone(extract_arxiv_id(text), text)

    def test_find_prefers_whole_field_then_text(self):
        self.assertEqual(find_arxiv_id(None, "2008.05489"), "2008.05489")
        self.assertEqual(find_arxiv_id("", "journal arXiv:1403.0592"), "1403.0592")
        self.assertIsNone(find_arxiv_id("x", None))

    def test_key_conversion(self):
        self.assertEqual(arxiv_id_to_key("1706.03762"), "1706-03762")
        self.assertEqual(arxiv_id_to_key("hep-th/9901001"), "hep-th_9901001")
        self.assertEqual(normalize_arxiv_id(arxiv_id_to_key("hep-th/9901001")), "hep-th/9901001")


class TestDoiAndYear(unittest.TestCase):
    def test_doi_forms_from_real_bib(self):
        self.assertEqual(extract_doi("https://doi.org/10.1111/itor.12062"), "10.1111/itor.12062")
        self.assertEqual(extract_doi("10.1016/J.JRTPM.2022.100366"), "10.1016/j.jrtpm.2022.100366")
        self.assertEqual(extract_doi("doi:10.1103/PhysRevX.4.031027."), "10.1103/physrevx.4.031027")
        self.assertEqual(extract_doi("https://link.aps.org/doi/10.1103/PhysRevX.4.031027"), "10.1103/physrevx.4.031027")
        self.assertEqual(normalize_doi("10.1002/nav.3800040307"), "10.1002/nav.3800040307")

    def test_no_doi(self):
        for raw in [None, "", "not a doi", "10.12/short"]:
            self.assertIsNone(extract_doi(raw), raw)

    def test_year_forms(self):
        self.assertEqual(extract_year("2020"), 2020)
        self.assertEqual(extract_year("2019-04-02T10:00:00"), 2019)
        self.assertEqual(extract_year("(2014)"), 2014)
        self.assertEqual(extract_year("Mon, 2 Apr 2007 19:18:42 GMT"), 2007)
        self.assertEqual(extract_year(1957), 1957)
        self.assertEqual(extract_year("MNRAS, 487, 2005 Costanzi"), 2005)

    def test_year_does_not_read_arxiv_ids_or_garbage(self):
        for value in [None, "", "n.d.", "1706.03762", 12, 99999]:
            self.assertIsNone(extract_year(value), value)


class TestAuthors(unittest.TestCase):
    def test_bibtex_last_first_with_accents_and_others(self):
        raw = r"P{\"a}prer, Paul and Joshi, Kanchan and Neufeld, Janis S and others"
        self.assertEqual(fam(parse_authors(raw)), [("paprer", "P"), ("joshi", "K"), ("neufeld", "JS")])

    def test_initial_styles(self):
        self.assertEqual(fam(parse_authors("Piu, Francesco and Speranza, M Grazia")), [("piu", "F"), ("speranza", "MG")])
        self.assertEqual(fam(parse_authors("Gleaves, V. B.")), [("gleaves", "VB")])
        self.assertEqual(fam(parse_authors(r"Cordeau, Jean-Fran{\c{c}}ois and Toth, Paolo")),
                         [("cordeau", "JF"), ("toth", "P")])
        self.assertEqual(fam(parse_authors("Smith, JM")), [("smith", "JM")])

    def test_first_last_forms(self):
        self.assertEqual(fam(parse_authors(r"N.~Goldman and M.~S. Rudner")), [("goldman", "N"), ("rudner", "MS")])
        self.assertEqual(fam(parse_authors("Ashish Vaswani")), [("vaswani", "A")])

    def test_particles_and_suffixes(self):
        self.assertEqual(fam(parse_authors("Maria van der Waals")), [("van der waals", "M")])
        self.assertEqual(parse_authors("van der Waals, J.")[0].key, "waals")
        self.assertEqual(fam(parse_authors("Smith Jr., John")), [("smith", "J")])
        self.assertEqual(fam(parse_authors("Martin Luther King Jr.")), [("king", "ML")])
        self.assertEqual(fam(parse_authors("D'Alessio, L.")), [("dalessio", "L")])

    def test_comma_lists(self):
        want = [("smith", "A"), ("jones", "B"), ("lee", "C")]
        self.assertEqual(fam(parse_authors("A. Smith, B. Jones, and C. Lee")), want)
        self.assertEqual(fam(parse_authors("A. Smith, B. Jones and C. Lee")), want)
        self.assertEqual(fam(parse_authors("Smith, A., Jones, B., and Lee, C.")), want)

    def test_list_input_and_corporate_names(self):
        self.assertEqual(fam(parse_authors(["Ashish Vaswani", "Noam Shazeer"])), [("vaswani", "A"), ("shazeer", "N")])
        self.assertEqual(fam(parse_authors("{Barnes and Noble} and Doe, J.")), [("barnes and noble", ""), ("doe", "J")])
        self.assertEqual(fam(parse_authors("ATLAS Collaboration")), [("atlas collaboration", "")])

    def test_et_al_and_empty(self):
        self.assertEqual(fam(parse_authors("Smith, J. and others")), [("smith", "J")])
        self.assertEqual(fam(parse_authors("Smith, J. et al.")), [("smith", "J")])
        self.assertEqual(parse_authors(None), ())
        self.assertEqual(parse_authors([]), ())
        self.assertEqual(parse_authors(""), ())
        self.assertIsNone(parse_author_name("others"))

    def test_key_is_last_family_word(self):
        self.assertEqual(Author("garcia marquez", "G").key, "marquez")
        self.assertEqual(Author("smith").key, "smith")


class TestCleanRecord(unittest.TestCase):
    def test_bibtex_entry(self):
        rec = clean_record(
            title=r"The locomotive assignment problem: a survey on optimization models",
            authors="Piu, Francesco and Speranza, M Grazia", year="2014",
            doi="https://doi.org/10.1111/itor.12062")
        self.assertEqual(rec.year, 2014)
        self.assertEqual(rec.doi, "10.1111/itor.12062")
        self.assertIsNone(rec.arxiv_id)
        self.assertEqual(rec.title, "the locomotive assignment problem a survey on optimization models")
        self.assertEqual(rec.title_tokens, ("locomotive", "assignment", "problem", "survey", "optimization", "models"))
        self.assertEqual(fam(rec.authors), [("piu", "F"), ("speranza", "MG")])

    def test_arxiv_id_found_in_url_field_of_bbl_entry(self):
        rec = clean_record(title="Some title", extra_text=["http://arxiv.org/abs/2108.09314", "Physical Review B"])
        self.assertEqual(rec.arxiv_id, "2108.09314")

    def test_eprint_url_is_not_an_arxiv_id(self):
        rec = clean_record(title="Cyclic scheduling",
                           extra_text=["https://onlinelibrary.wiley.com/doi/pdf/10.1002/nav.3800040307"])
        self.assertIsNone(rec.arxiv_id)
        self.assertEqual(rec.doi, "10.1002/nav.3800040307")

    def test_crawled_reference_with_noisy_title_and_no_authors(self):
        rec = clean_record(title="MNRAS, 487, 2005 Costanzi, M., Rozo, E., Simet, M., et al.", authors=[],
                           year="2019", arxiv_id="1706.03762")
        self.assertEqual(rec.authors, ())
        self.assertEqual(rec.year, 2019)
        self.assertEqual(rec.arxiv_id, "1706.03762")
        self.assertIn("costanzi", rec.title_tokens)

    def test_all_empty(self):
        rec = clean_record()
        self.assertEqual((rec.title, rec.title_tokens, rec.authors, rec.year, rec.doi, rec.arxiv_id),
                         ("", (), (), None, None, None))


class TestRealBibtexJsonl(unittest.TestCase):
    """Strings copied from the real bibtex.jsonl of range 02 (DES papers)."""

    def test_double_encoded_utf8_is_repaired(self):
        self.assertEqual(fix_mojibake("FertÃ©, A."), "Ferté, A.")
        self.assertEqual(fix_mojibake("Plazas MalagÃ³n"), "Plazas Malagón")
        self.assertEqual(fix_mojibake("SÃ¡nchez and Mena-FernÃ¡ndez"), "Sánchez and Mena-Fernández")
        self.assertEqual(fix_mojibake("Ferté, plain ASCII, Ã alone"), "Ferté, plain ASCII, Ã alone")
        self.assertEqual(fix_mojibake(""), "")

    def test_repaired_names_match_the_clean_spelling(self):
        for broken, clean in [("FertÃ©, A.", "Ferté, A."), ("Mena-FernÃ¡ndez, J.", "Mena-Fernández, J."),
                              ("Paz-ChinchÃ³n, F.", "Paz-Chinchón, F."), ("SÃ¡nchez, C.", "Sánchez, C.")]:
            self.assertEqual(parse_authors(broken), parse_authors(clean), broken)
        self.assertEqual(fam(parse_authors("FertÃ©, A.")), [("ferte", "A")])

    def test_mangled_dotless_i(self):
        self.assertEqual(fam(parse_authors(r"Garc\'\ia-Bellido, J.")), [("garcia bellido", "J")])
        self.assertEqual(fam(parse_authors(r"Garc\'{\i}a-Bellido, J.")), [("garcia bellido", "J")])

    def test_tilde_separated_initials_with_multiword_family(self):
        self.assertEqual(fam(parse_authors(r"da Costa, L.~N.")), [("da costa", "LN")])
        self.assertEqual(fam(parse_authors(r"Plazas Malag\'on, A.~A.")), [("plazas malagon", "AA")])
        self.assertEqual(fam(parse_authors(r"Carnero Rosell, A.~A. and Abbott, T.~M.~C.")),
                         [("carnero rosell", "AA"), ("abbott", "TMC")])
        self.assertEqual(fam(parse_authors(r"Leget, P. -F.")), [("leget", "PF")])
        self.assertEqual(fam(parse_authors("Dark Energy Survey Collaboration")), [("dark energy survey collaboration", "")])

    def test_greek_and_subscripts_in_titles(self):
        self.assertIn("λcdm", title_tokens(r"extensions to \ensuremath\Lambda CDM with weak lensing"))
        self.assertEqual(title_tokens(r"A Precise H$_0$ Estimate"), title_tokens("A Precise H0 Estimate"))
        self.assertEqual(normalize_title(r"constraints on \ensuremath\Lambda CDM"), normalize_title("Constraints on ΛCDM"))
        self.assertEqual(latex_to_text(r"$\alpha$-decay of $\beta_2$"), "α-decay of β2")


if __name__ == "__main__":
    unittest.main()
