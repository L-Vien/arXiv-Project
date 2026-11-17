import unittest

from src.parsing.contracts import (
    Diagnostic, DiagnosticSeverity, NodeKind, NormalizedVersion, ProcessingStatus, RawTree, TreeNode,
)
from src.parsing.normalization import (
    Normalizer, clean_tree, normalize_block_formula, normalize_text, normalize_tree, split_sentences,
)

K = NodeKind
AID = "2501.01234"


def node(kind, content=None, *children, **kw):
    return TreeNode(kind=kind, content=content, children=tuple(children), **kw)


def tree_of(*children, version=1):
    return RawTree(AID, version, node(K.DOCUMENT, None, *children), ProcessingStatus.OK)


def leaves(nv: NormalizedVersion):
    """(kind, content) of every node in document order, root excluded."""
    return [(n.kind, n.content) for n in nv.nodes[1:]]


def sentences_of(raw, cmap=None):
    nv = normalize_tree(tree_of(node(K.PARAGRAPH, raw)), cmap)
    return [c for k, c in leaves(nv) if k == "sentence"]


class TestCleanup(unittest.TestCase):
    def test_formatting_commands_removed(self):
        self.assertEqual(sentences_of(r"\textbf{Hello} \emph{big \textit{world}}."), ["Hello big world."])

    def test_declaration_style_and_noise(self):
        self.assertEqual(sentences_of(r"{\bf Bold} text\label{x}\vspace{2pt}\\ more."), ["Bold text more."])

    def test_comments_and_escaped_percent(self):
        self.assertEqual(sentences_of("Hello % hidden\nworld. 100\\% sure."), ["Hello world.", "100% sure."])

    def test_whitespace_and_crlf(self):
        self.assertEqual(sentences_of("A  first\r\n  sentence.\r\n\r\nSecond   one."),
                         ["A first sentence.", "Second one."])

    def test_heading_text(self):
        nv = normalize_tree(tree_of(node(K.HEADING, r"\textbf{Intro}  duction\label{s1}", level=1)))
        self.assertEqual(leaves(nv), [("heading", "Intro duction")])

    def test_abstract_command_keeps_its_text_without_the_command(self):
        self.assertEqual(
            sentences_of(r"\abstract{We study robust parsing.}"),
            ["We study robust parsing."],
        )

    def test_table_layout_commands_are_removed_from_text(self):
        self.assertEqual(
            sentences_of(r"First row. \midrule Second row."),
            ["First row.", "Second row."],
        )


class TestMath(unittest.TestCase):
    def test_inline_math_forms(self):
        self.assertEqual(sentences_of(r"Let \(x+y\) and $ a   b $ hold."), ["Let $x+y$ and $a b$ hold."])

    def test_inline_math_drops_labels(self):
        self.assertEqual(
            sentences_of(r"Let $x\label{eq:x}=1$ hold."),
            ["Let $x=1$ hold."],
        )

    def test_escaped_dollar_is_not_math(self):
        self.assertEqual(sentences_of(r"It costs \$5 and \$6."), [r"It costs \$5 and \$6."])

    def test_block_formula_forms(self):
        want = r"\begin{equation}E = mc^2\end{equation}"
        for raw in ("$$E = mc^2$$", r"\[ E = mc^2 \]", r"\begin{equation*}E = mc^2\end{equation*}",
                    r"\begin{equation} E = mc^2 \label{e1} \end{equation}", "E = mc^2"):
            self.assertEqual(normalize_block_formula(raw), want, raw)

    def test_block_formula_drops_layout_commands(self):
        self.assertEqual(
            normalize_block_formula(
                r"\begin{equation}\centering x = 1 \midrule \label{eq:x}\end{equation}"
            ),
            r"\begin{equation}x = 1\end{equation}",
        )

    def test_align_becomes_aligned_in_equation(self):
        self.assertEqual(
            normalize_block_formula(r"\begin{align} a &= b \\ c &= d \nonumber \end{align}"),
            r"\begin{equation}\begin{aligned}a &= b \\ c &= d\end{aligned}\end{equation}",
        )

    def test_block_node_in_tree(self):
        nv = normalize_tree(tree_of(node(K.BLOCK_FORMULA, "$$ x $$")))
        self.assertEqual(leaves(nv), [("block_formula", r"\begin{equation}x\end{equation}")])

    def test_display_math_inside_paragraph_is_split(self):
        raw = r"We have \begin{align} a &= b \\ c &= d \end{align} where $a$ is small. Done."
        nv = normalize_tree(tree_of(node(K.PARAGRAPH, raw)))
        self.assertEqual(leaves(nv), [
            ("sentence", "We have"),
            ("block_formula", r"\begin{equation}\begin{aligned}a &= b \\ c &= d\end{aligned}\end{equation}"),
            ("sentence", "where $a$ is small."),
            ("sentence", "Done."),
        ])


class TestSentenceSplitting(unittest.TestCase):
    def test_tc05_url_decimal_abbreviation_math(self):
        raw = (r"We reach 3.14 accuracy on Fig.~\ref{f1}, e.g. the model of Smith et al. is good. "
               r"See https://example.com/a.b for code. The value $x = 1.5$ holds. Next sentence?")
        self.assertEqual(sentences_of(raw), [
            r"We reach 3.14 accuracy on Fig. \ref{f1}, e.g. the model of Smith et al. is good.",
            "See https://example.com/a.b for code.",
            "The value $x = 1.5$ holds.",
            "Next sentence?",
        ])

    def test_url_command_with_period_inside(self):
        self.assertEqual(sentences_of(r"Code at \url{http://a.org/x. y} now. Next one."),
                         [r"Code at \url{http://a.org/x. y} now.", "Next one."])

    def test_initials_not_split(self):
        self.assertEqual(sentences_of("By J. Smith and A. Jones. Next."), ["By J. Smith and A. Jones.", "Next."])

    def test_et_al_soft_abbreviation(self):
        self.assertEqual(sentences_of("Smith et al. proposed it. Jones et al. We agree."),
                         ["Smith et al. proposed it.", "Jones et al.", "We agree."])

    def test_math_with_period_does_not_split(self):
        self.assertEqual(sentences_of("Take $x. y$ then stop. Go."), ["Take $x. y$ then stop.", "Go."])

    def test_split_sentences_empty(self):
        self.assertEqual(split_sentences("   "), [])


class TestRealWorldNoise(unittest.TestCase):
    def test_footnote_stays_with_previous_sentence(self):
        self.assertEqual(sentences_of(r"Our method works. \footnote{Note. Two.} It is fast."),
                         [r"Our method works. \footnote{Note. Two.}", "It is fast."])

    def test_footnote_inside_sentence_is_kept_inline(self):
        self.assertEqual(sentences_of(r"It works\footnote{Mostly.} on small data. Done."),
                         [r"It works\footnote{Mostly.} on small data.", "Done."])

    def test_environment_markers_and_title_commands_removed(self):
        self.assertEqual(sentences_of(r"\maketitle \begin{abstract} We study things. \end{abstract}"),
                         ["We study things."])
        self.assertEqual(sentences_of(r"\begin{minipage}[t]{0.5\textwidth} Hello there. \end{minipage}"),
                         ["Hello there."])


class TestFloatsAndMarkers(unittest.TestCase):
    def units_of(self, raw):
        return leaves(normalize_tree(tree_of(node(K.PARAGRAPH, raw))))

    def test_algorithm_inside_paragraph_becomes_figure(self):
        raw = (r"Intro text here. \begin{algorithm}[H]\caption{Solve$(a)$}\begin{algorithmic}\State x"
               r"\Return y\end{algorithmic}\end{algorithm} After text.")
        self.assertEqual(self.units_of(raw), [
            ("sentence", "Intro text here."), ("figure", "Solve$(a)$"), ("sentence", "After text."),
        ])

    def test_bare_tabular_becomes_empty_figure(self):
        raw = r"See below. \begin{tabular}{cc} a & b \\ c & d \end{tabular} Done."
        self.assertEqual(self.units_of(raw), [("sentence", "See below."), ("figure", ""), ("sentence", "Done.")])

    def test_theorem_like_markers_removed(self):
        self.assertEqual(sentences_of(r"\begin{proof} The choices give $x$. \end{proof}"), ["The choices give $x$."])
        self.assertEqual(sentences_of(r"\begin{theorem}[Main] Let $x$ hold. \end{theorem}"), ["Let $x$ hold."])

    def test_front_matter_and_include_commands_dropped(self):
        self.assertEqual(sentences_of(r"\title{A Paper}\author[1]{Jane \thanks{x}}\affiliation{Uni} Real text here."),
                         ["Real text here."])
        self.assertEqual(sentences_of(r"\input{sec1} Text goes here."), ["Text goes here."])

    def test_footnote_with_optional_argument(self):
        self.assertEqual(sentences_of(r"It works. \footnote[2]{Note.} Done."), [r"It works. \footnote[2]{Note.}", "Done."])


class TestRealWorldMacros(unittest.TestCase):
    def units_of(self, raw):
        return leaves(normalize_tree(tree_of(node(K.PARAGRAPH, raw))))

    def test_shorthand_equation_macros(self):
        self.assertEqual(self.units_of(r"We find, \be x = 1 \ee Next sentence here."), [
            ("sentence", "We find,"), ("block_formula", r"\begin{equation}x = 1\end{equation}"),
            ("sentence", "Next sentence here."),
        ])
        self.assertEqual(self.units_of(r"Then \bq a &=& b \eq ok."), [
            ("sentence", "Then"),
            ("block_formula", r"\begin{equation}\begin{aligned}a &=& b\end{aligned}\end{equation}"),
            ("sentence", "ok."),
        ])

    def test_beta_is_not_mistaken_for_be_macro(self):
        self.assertEqual(sentences_of(r"The \beta decay and $\beta$ hold."), [r"The \beta decay and $\beta$ hold."])

    def test_macro_definitions_dropped(self):
        raw = (r"\newcommand{\be}{\begin{equation}} \def\x#1{#1+1} \let\a\b "
               r"\renewcommand\foo[2][x]{body {nested}} \newtheorem{thm}{Theorem}[section] Real text here.")
        self.assertEqual(sentences_of(raw), ["Real text here."])

    def test_definition_before_alias_use(self):
        raw = r"\newcommand{\be}{\begin{equation}} Start. \be y \ee End."
        self.assertEqual(self.units_of(raw), [
            ("sentence", "Start."), ("block_formula", r"\begin{equation}y\end{equation}"), ("sentence", "End."),
        ])

    def test_iffalse_and_comment_environment_dropped(self):
        self.assertEqual(sentences_of(r"Keep this. \iffalse Hidden text. \fi Also keep. \begin{comment}Gone.\end{comment}"),
                         ["Keep this.", "Also keep."])

    def test_stray_items_become_separate_sentences(self):
        self.assertEqual(sentences_of(r"\begin{itemize} \item First thing \item[(b)] Second thing. \end{itemize}"),
                         ["First thing", "Second thing."])

    def test_textgreek_unwrapped(self):
        self.assertEqual(sentences_of(r"The \textgreek{a} letter."), ["The a letter."])


class TestCitations(unittest.TestCase):
    def test_nocite_wildcard_is_not_exported_as_prose(self):
        self.assertEqual(sentences_of(r"\nocite{*}"), [])

    def test_rewrite_and_dedupe_keys(self):
        got = sentences_of(r"As shown in \cite{a, b} and \citep[p. 5]{b}. Next.", {"a": "k1", "b": "k1"})
        self.assertEqual(got, [r"As shown in \cite{k1} and \citep[p. 5]{k1}.", "Next."])

    def test_unknown_key_kept(self):
        self.assertEqual(sentences_of(r"See \cite{zzz}."), [r"See \cite{zzz}."])

    def test_empty_map_still_canonical_spacing(self):
        self.assertEqual(sentences_of(r"See \cite{a, b}."), [r"See \cite{a,b}."])

    def test_extended_citation_commands_use_tv5_rewriter(self):
        self.assertEqual(
            sentences_of(r"See \parencite[see][p. 2]{a, b}.", {"a": "k", "b": "k"}),
            [r"See \parencite[see][p. 2]{k}."],
        )


class TestStructure(unittest.TestCase):
    def test_paragraph_is_flattened_under_heading(self):
        nv = normalize_tree(tree_of(node(K.HEADING, "Intro", node(K.PARAGRAPH, "First one. Second one."), level=1)))
        self.assertEqual(leaves(nv), [("heading", "Intro"), ("sentence", "First one."), ("sentence", "Second one.")])
        self.assertFalse(any(k == "paragraph" for k, _ in leaves(nv)))

    def test_list_items_with_text_and_nested_list(self):
        nv = normalize_tree(tree_of(node(K.LIST, None,
            node(K.LIST_ITEM, r"\textbf{One} item. Two.", node(K.LIST, None, node(K.LIST_ITEM, "Inner.")))
        )))
        self.assertEqual(leaves(nv), [
            ("list", ""), ("list_item", ""), ("sentence", "One item."), ("sentence", "Two."),
            ("list", ""), ("list_item", ""), ("sentence", "Inner."),
        ])

    def test_figure_caption_extracted(self):
        raw = r"\centering\includegraphics[width=3cm]{x.png}\caption{A \textbf{big} figure.}\label{f}"
        nv = normalize_tree(tree_of(node(K.FIGURE, raw)))
        self.assertEqual(leaves(nv), [("figure", "A big figure.")])

    def test_table_keeps_caption_only(self):
        raw = r"\begin{tabular}{cc} a & b \\ c & d \end{tabular}\caption{Results.}\subcaption{Left.}"
        nv = normalize_tree(tree_of(node(K.FIGURE, raw)))
        self.assertEqual(leaves(nv), [("figure", "Results.")])

    def test_figure_without_caption_is_empty(self):
        nv = normalize_tree(tree_of(node(K.FIGURE, r"\includegraphics{x.png}")))
        self.assertEqual(leaves(nv), [("figure", "")])

    def test_empty_paragraph_dropped(self):
        nv = normalize_tree(tree_of(node(K.PARAGRAPH, " % only a comment\n ")))
        self.assertEqual(leaves(nv), [])

    def test_unknown_kind_children_are_kept(self):
        nv = normalize_tree(tree_of(node("mystery", "x", node(K.SENTENCE, "Kept."))))
        self.assertEqual(leaves(nv), [("sentence", "Kept.")])


class TestNormalizedVersion(unittest.TestCase):
    def test_ids_order_parents_and_counts(self):
        raw = tree_of(node(K.HEADING, "Intro", node(K.PARAGRAPH, "One. Two."), level=1), node(K.PARAGRAPH, "Three."))
        nv = normalize_tree(raw)
        self.assertIsInstance(nv, NormalizedVersion)
        self.assertEqual([n.local_id for n in nv.nodes], ["n0", "n1", "n2", "n3", "n4"])
        self.assertEqual([n.order for n in nv.nodes], [0, 1, 2, 3, 4])
        self.assertEqual([n.parent_local_id for n in nv.nodes], [None, "n0", "n1", "n1", "n0"])
        self.assertEqual((nv.root_local_id, nv.nodes[0].kind, nv.nodes[0].content), ("n0", "document", ""))
        self.assertEqual(nv.raw_element_count, 4)  # raw tree: document + heading + 2 paragraphs
        self.assertEqual((nv.arxiv_id, nv.version, nv.status), (AID, 1, ProcessingStatus.OK))

    def test_root_none_gives_empty_version_and_keeps_status(self):
        diag = Diagnostic(DiagnosticSeverity.INFO, "NO_TEX", "no tex")
        t = RawTree(AID, 2, None, ProcessingStatus.EMPTY_NO_TEX, (diag,))
        nv = normalize_tree(t)
        self.assertEqual((nv.root_local_id, nv.nodes, nv.raw_element_count), (None, (), 0))
        self.assertEqual((nv.version, nv.status, nv.diagnostics), (2, ProcessingStatus.EMPTY_NO_TEX, (diag,)))

    def test_status_and_diagnostics_preserved(self):
        diag = Diagnostic(DiagnosticSeverity.WARNING, "MISSING_INCLUDE", "missing", file_path="a.tex")
        t = RawTree(AID, 3, node(K.DOCUMENT, None, node(K.PARAGRAPH, "Hi there.")),
                    ProcessingStatus.PARTIAL_PARSE, (diag,))
        nv = normalize_tree(t)
        self.assertEqual((nv.status, nv.diagnostics), (ProcessingStatus.PARTIAL_PARSE, (diag,)))

    def test_input_not_mutated(self):
        t = tree_of(node(K.PARAGRAPH, r"\textbf{A} b."))
        snapshot = tree_of(node(K.PARAGRAPH, r"\textbf{A} b."))
        normalize_tree(t)
        self.assertEqual(t, snapshot)

    def test_non_document_root_is_wrapped(self):
        t = RawTree(AID, 1, node(K.PARAGRAPH, "Only text."), ProcessingStatus.OK)
        nv = normalize_tree(t)
        self.assertEqual([(n.kind, n.content) for n in nv.nodes], [("document", ""), ("sentence", "Only text.")])

    def test_class_implements_protocol_method(self):
        t = tree_of(node(K.PARAGRAPH, "Hi there."))
        self.assertEqual(Normalizer().normalize(t), normalize_tree(t))

    def test_idempotent(self):
        t = tree_of(
            node(K.HEADING, r"\textbf{Intro}", node(
                K.PARAGRAPH, r"See \cite{a, b}. We have $$x$$ and $y$. Fig.~\ref{f} shows 3.5 gain."), level=1),
            node(K.FIGURE, r"\caption{Cap \emph{x}.}"),
            node(K.BLOCK_FORMULA, r"\begin{align} a &= b \end{align}"),
        )
        cmap = {"a": "k", "b": "k"}
        once = clean_tree(t, cmap)
        self.assertEqual(clean_tree(once, cmap), once)
        self.assertEqual(normalize_tree(once, cmap).nodes, normalize_tree(t, cmap).nodes)

    def test_normalize_text_helper(self):
        self.assertEqual(normalize_text(r"\emph{A}~b  $x$"), "A b $x$")


if __name__ == "__main__":
    unittest.main()
