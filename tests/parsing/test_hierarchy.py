"""TV3 tests. Run from repo root: python -m pytest tests/parsing -q"""

from src.parsing.contracts import (
    NodeKind, ProcessingStatus, SourceBundle, SourceFile,
)
from src.parsing.hierarchy import parse_structure
import pytest

def bundle(tex, *, extra=(), version=1, status=ProcessingStatus.OK):
    return SourceBundle(
        arxiv_id="2501.01234", version=version, status=status,
        root_file="main.tex",
        files=(SourceFile("main.tex", tex, 0), *extra),
    )


def find_all(root, kind):
    found = []
    def visit(node):
        if node.kind == kind:
            found.append(node)
        for child in node.children:
            visit(child)
    visit(root)
    return found


def test_headings_nested_and_preamble_excluded():
    tex = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
Before the first section.
\section{Introduction}
Intro body.
\subsection{Prior Work}
Prior body.
\subsubsection{Details}
Details body.
\section{Method}
Method body.
\end{document}
Should never be parsed.
"""
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.OK
    root = tree.root
    assert root.kind == NodeKind.DOCUMENT
    assert [child.kind for child in root.children] == [NodeKind.PARAGRAPH, NodeKind.HEADING, NodeKind.HEADING]
    intro, method = root.children[1:]
    assert intro.content == "Introduction"
    assert intro.level == 3
    assert intro.children[1].content == "Prior Work"
    assert intro.children[1].children[1].content == "Details"
    assert method.content == "Method"
    contents = str(root)
    assert r"\usepackage" not in contents
    assert "Should never be parsed." not in contents


def test_starred_acknowledgements_appendix_and_exclusion():
    tex = r"""\begin{document}
\section*{Introduction}
Intro.
\section*{References}
A bib reference that is NOT text.
\subsection{Nested bibliography}
Should be excluded too.
\section*{Acknowledgements}
Thanks to everyone.
\appendix
\section{Proofs}
Proof 1.
\end{document}"""
    tree = parse_structure(bundle(tex))
    headings = find_all(tree.root, NodeKind.HEADING)
    assert [h.content for h in headings] == ["Introduction", "Acknowledgements", "Proofs"]
    assert headings[0].attributes["unnumbered"] == "true"
    assert headings[2].attributes["appendix"] == "true"
    assert "Thanks to everyone." in str(tree.root)
    assert "Should be excluded" not in str(tree.root)


def test_lists_nested_items_order_and_description_label():
    tex = r"""\begin{document}
\section{Method}
Intro.
\begin{enumerate}
\item First item.
\begin{itemize}
\item Nested one.
\item Nested two.
\end{itemize}
\item[Label] Second item.
\end{enumerate}
Final paragraph.
\end{document}"""
    tree = parse_structure(bundle(tex))
    method = tree.root.children[0]
    assert [n.kind for n in method.children] == [NodeKind.PARAGRAPH, NodeKind.LIST, NodeKind.PARAGRAPH]
    outer = method.children[1]
    assert outer.attributes["environment"] == "enumerate"
    assert [n.kind for n in outer.children] == [NodeKind.LIST_ITEM, NodeKind.LIST_ITEM]
    nested = outer.children[0].children[1]
    assert nested.kind == NodeKind.LIST
    assert len(nested.children) == 2
    assert outer.children[1].attributes["label"] == "Label"
    assert "Second item." in outer.children[1].children[0].content


def test_math_and_figure_table_are_structural_leaves():
    tex = r"""\begin{document}
\section{Results}
Some text with $x^2$ and \(y^2\) inline.
\begin{equation}\label{eq:x} x=y \end{equation}
\[ z=w \]
$$ E=mc^2 $$
\begin{figure}[htbp]\centering\includegraphics{f.png}\caption{Example}\end{figure}
\begin{table}[ht]\begin{tabular}{cc} A & B \end{tabular}\end{table}
Conclusion.
\end{document}"""
    tree = parse_structure(bundle(tex))
    section = tree.root.children[0]
    assert [node.kind for node in section.children] == [
        NodeKind.PARAGRAPH,
        NodeKind.BLOCK_FORMULA, NodeKind.BLOCK_FORMULA, NodeKind.BLOCK_FORMULA,
        NodeKind.FIGURE, NodeKind.FIGURE, NodeKind.PARAGRAPH,
    ]
    assert "$x^2$" in section.children[0].content
    assert section.children[4].attributes["environment"] == "figure"
    assert section.children[5].attributes["environment"] == "table"


def test_included_files_follow_invocation_order_and_skip_comments():
    main = r"""\documentclass{article}
\begin{document}
\section{Start}
Start text.
\input{chapters/intro}
\section{End}
End text.
% \input{extra}
\end{document}"""
    files = (
        SourceFile("chapters/intro.tex", "Intro body.\\input{details}\\section{Middle}Mid.", 1),
        SourceFile("chapters/details.tex", "Detail before Middle.", 2),
        SourceFile("extra.tex", r"\section{Wrong}", 3),
    )
    tree = parse_structure(bundle(main, extra=files))
    assert tree.status == ProcessingStatus.OK
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Start", "Middle", "End"]
    assert "Detail before Middle." in str(tree.root)
    assert "Wrong" not in str(tree.root)
    assert find_all(tree.root, NodeKind.HEADING)[1].source_file == "chapters/intro.tex"


def test_references_thebibliography_excluded_but_later_sections_kept():
    tex = r"""\begin{document}
\section{Main}
Main text.
\begin{thebibliography}{9}
\bibitem{a} \section{Fake} bibliographic text.
\end{thebibliography}
\section{Conclusion}
Conclusion text.
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Main", "Conclusion"]
    assert "bibliographic text" not in str(tree.root)


def test_verbatim_does_not_trigger_false_heading_or_comment():
    tex = r"""\begin{document}
\section{Real}
\begin{verbatim}
\section{Fake}
% ignored? No, inside verbatim.
\end{verbatim}
\verb|\section{Also Fake}|
Escaped percent \% stays.
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Real"]
    assert "Fake" in str(tree.root)
    assert r"\%" in str(tree.root)


def test_repeated_include_kept_and_cycles_reported():
    main = r"\begin{document}\section{Main}\input{child}\input{child}\end{document}"
    child = SourceFile("child.tex", r"Child content.\input{main}", 1)
    tree = parse_structure(bundle(main, extra=(child,)))
    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert str(tree.root).count("Child content.") == 2
    assert any(d.code == "INCLUDE_CYCLE" for d in tree.diagnostics)


def test_missing_include_is_not_silently_ignored():
    tree = parse_structure(bundle(r"\begin{document}\section{A}\input{lost}After.\end{document}"))
    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "MISSING_INCLUDE" for d in tree.diagnostics)
    assert "After." in str(tree.root)


def test_empty_source_version():
    source = SourceBundle(
        "2501.01234", 2, ProcessingStatus.EMPTY_NO_TEX,
        None, (),
    )
    tree = parse_structure(source)
    assert tree.version == 2
    assert tree.status == ProcessingStatus.EMPTY_NO_TEX
    assert tree.root.kind == NodeKind.DOCUMENT
    assert tree.root.children == ()


def test_optional_section_title_and_nested_braces():
    tex = r"\begin{document}\section[Short]{A {Nested} Title}Text.\end{document}"
    tree = parse_structure(bundle(tex))
    assert tree.root.children[0].content == "A {Nested} Title"


def test_blank_lines_create_separate_raw_paragraphs():
    tex = "\\begin{document}\nFirst paragraph.\n\nSecond paragraph.\n\\end{document}"
    tree = parse_structure(bundle(tex))
    assert [p.content for p in tree.root.children] == ["First paragraph.", "Second paragraph."]


def test_math_cannot_create_fake_structure():
    tex = r"\begin{document}Inline $\section{Fake}$ here. \section{Real}Body.\end{document}"
    tree = parse_structure(bundle(tex))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Real"]


def test_unclosed_figure_diagnosed():
    tex = r"\begin{document}\section{A}\begin{figure}Missing end"
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "UNCLOSED_ENV" for d in tree.diagnostics)


def test_no_unnecessary_dedup_or_sentence_split():
    tex = r"\begin{document}\section{A}A sentence. Another sentence.\section{B}A sentence. Another sentence.\end{document}"
    tree = parse_structure(bundle(tex))
    paragraphs = find_all(tree.root, NodeKind.PARAGRAPH)
    assert len(paragraphs) == 2
    assert paragraphs[0].content == paragraphs[1].content
    assert not find_all(tree.root, NodeKind.SENTENCE)


def test_abstract_and_acknowledgements_environments_kept():
    tex = r"""\begin{document}
\begin{abstract}We propose a parser.\end{abstract}
\section{Introduction}Intro.
\begin{acknowledgements}Thank you.\end{acknowledgements}
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == [
        "Abstract", "Introduction", "Acknowledgements"
    ]
    assert "Thank you." in str(tree.root)


def test_cross_version_raw_parser_does_not_share_element_identity():
    source = bundle(r"\begin{document}\section{Intro}Content.\end{document}", version=3)
    tree = parse_structure(source)
    assert tree.version == 3
    assert tree.arxiv_id == "2501.01234"
    assert not hasattr(tree.root, "element_id")


def test_included_file_root_preserves_source_file_location():
    main = r"\begin{document}\input{sections/intro}\end{document}"
    intro = SourceFile("sections/intro.tex", r"\section{From Child}Content.", 1)
    tree = parse_structure(bundle(main, extra=(intro,)))
    heading = tree.root.children[0]
    assert heading.source_file == "sections/intro.tex"
    assert heading.children[0].source_file == "sections/intro.tex"


def test_section_star_and_heading_attributes():
    tex = r"\begin{document}\chapter{A}\section*{No number}\paragraph{Fine}Text.\end{document}"
    tree = parse_structure(bundle(tex))
    chapter = tree.root.children[0]
    section = chapter.children[0]
    paragraph_heading = section.children[0]
    assert (chapter.level, section.level, paragraph_heading.level) == (2, 3, 6)
    assert paragraph_heading.kind == NodeKind.HEADING
    assert section.attributes["unnumbered"] == "true"


def test_source_with_missing_root_returns_status():
    source = SourceBundle(
        "2501.01234", 1, ProcessingStatus.OK, "missing.tex",
        (SourceFile("other.tex", "Body", 0),),
    )
    tree = parse_structure(source)
    assert tree.status == ProcessingStatus.ROOT_NOT_FOUND
    assert any(d.code == "ROOT_NOT_FOUND" for d in tree.diagnostics)


def test_input_comment_and_math_preserve_text_order():
    tex = r"""\begin{document}
\section{Intro}
Before.
% \input{fake}
\[ x=1 \]
After.
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert [n.kind for n in tree.root.children[0].children] == [
        NodeKind.PARAGRAPH, NodeKind.BLOCK_FORMULA, NodeKind.PARAGRAPH,
    ]


def test_input_inside_verbatim_is_not_expanded():
    main = r"""\begin{document}
\begin{verbatim}
\input{fake}
\end{verbatim}
\input{real}
\end{document}"""
    extra = (
        SourceFile("fake.tex", r"\section{Should Not Exist}", 1),
        SourceFile("real.tex", r"\section{Real}", 2),
    )
    tree = parse_structure(bundle(main, extra=extra))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Real"]


def test_input_without_braces_supported():
    main = "\\begin{document}\\input chapter\\end{document}"
    tree = parse_structure(bundle(main, extra=(SourceFile("chapter.tex", r"\section{Included}", 1),)))
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["Included"]

@pytest.mark.parametrize(
    "include_name",
    ["section/child", "child"],
)
def test_nested_include_path_resolution(include_name):
    source = SourceBundle(
        arxiv_id="2501.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\input{section/parent}"
                    r"\end{document}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path="section/parent.tex",
                content=rf"\input{{{include_name}}}",
                order=1,
            ),
            SourceFile(
                relative_path="section/child.tex",
                content=r"\section{Included correctly}",
                order=2,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not result.diagnostics

    headings = [
        node
        for node in result.root.children
        if node.kind == NodeKind.HEADING
    ]

    assert len(headings) == 1
    assert headings[0].content == "Included correctly"


def test_escaped_display_math_delimiter():
    source = SourceBundle(
        arxiv_id="2501.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\section{Results}"
                    r"First row \\[2mm] Second row."
                    r"\[x^2 + y^2 = z^2\]"
                    r"\end{document}"
                ),
                order=0,
            ),
        ),
    )

    result = parse_structure(source)

    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    nodes = list(walk(result.root))

    formulas = [
        node for node in nodes
        if node.kind == NodeKind.BLOCK_FORMULA
    ]

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "UNCLOSED_MATH"
        for d in result.diagnostics
    )
    assert len(formulas) == 1
    assert "x^2 + y^2 = z^2" in formulas[0].content

def test_include_with_simple_path_macro():
    source = SourceBundle(
        arxiv_id="2501.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\input{figures/combined}"
                    r"\end{document}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path="figures/combined.tex",
                content=(
                    r"\providecommand{\FigureTwoRoot}{figures}"
                    r"\input{\FigureTwoRoot/geometry}"
                ),
                order=1,
            ),
            SourceFile(
                relative_path="figures/geometry.tex",
                content=r"\section{Geometry}",
                order=2,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "MISSING_INCLUDE"
        for d in result.diagnostics
    )

    assert any(
        child.kind == NodeKind.HEADING
        and child.content == "Geometry"
        for child in result.root.children
    )


def test_packeditemize_environment():
    source = SourceBundle(
        arxiv_id="2501.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\section{Contributions}"
                    r"\begin{packeditemize}"
                    r"\item First contribution"
                    r"\item Second contribution"
                    r"\end{packeditemize}"
                    r"\end{document}"
                ),
                order=0,
            ),
        ),
    )

    result = parse_structure(source)

    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    nodes = list(walk(result.root))

    lists = [
        node for node in nodes
        if node.kind == NodeKind.LIST
    ]

    items = [
        node for node in nodes
        if node.kind == NodeKind.LIST_ITEM
    ]

    assert result.status == ProcessingStatus.OK
    assert len(lists) == 1
    assert len(items) == 2

    assert not any(
        d.code == "ITEM_OUTSIDE_LIST"
        for d in result.diagnostics
    )

def test_compactitem_inside_frame():
    source = SourceBundle(
        arxiv_id="2501.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\begin{stylishframe}"
                    r"\begin{compactitem}"
                    r"\item First contribution"
                    r"\item Second contribution"
                    r"\end{compactitem}"
                    r"\end{stylishframe}"
                    r"\end{document}"
                ),
                order=0,
            ),
        ),
    )

    result = parse_structure(source)

    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    nodes = list(walk(result.root))

    assert sum(node.kind == NodeKind.LIST for node in nodes) == 1
    assert sum(node.kind == NodeKind.LIST_ITEM for node in nodes) == 2
    assert not any(
        d.code == "ITEM_OUTSIDE_LIST"
        for d in result.diagnostics
    )



def test_standard_latex_list_environment():
    main = r"""
    \begin{document}
    \section{Affiliations}
    \begin{list}{}{
        \setlength{\leftmargin}{1cm}
    }
    \item[1] First institution.
    \item[2] Second institution.
    \end{list}
    \end{document}
    """

    tree = parse_structure(bundle(main))

    def collect(node, kind):
        result = [node] if node.kind == kind else []
        for child in node.children:
            result.extend(collect(child, kind))
        return result

    lists = collect(tree.root, "list")
    items = collect(tree.root, "list_item")

    assert len(lists) == 1
    assert len(items) == 2
    assert [item.attributes.get("label") for item in items] == ["1", "2"]
    assert not any(
        diagnostic.code == "ITEM_OUTSIDE_LIST"
        for diagnostic in tree.diagnostics
    )


def test_inline_enumerate_environment():
    main = r"""
    \begin{document}
    \section{Introduction}
    \begin{enumerate*}
        \item First capability.
        \item Second capability.
    \end{enumerate*}
    \end{document}
    """

    tree = parse_structure(bundle(main))

    def collect(node, kind):
        found = [node] if node.kind == kind else []
        for child in node.children:
            found.extend(collect(child, kind))
        return found

    lists = collect(tree.root, "list")
    items = collect(tree.root, "list_item")

    assert len(lists) == 1
    assert len(items) == 2
    assert lists[0].attributes["environment"] == "enumerate*"
    assert not any(
        d.code == "ITEM_OUTSIDE_LIST"
        for d in tree.diagnostics
    )


def test_custom_list_macro_preserves_items():
    tex = r"""
\documentclass{article}
\newcommand{\squishlist}{\begin{list}{}{\setlength{\itemsep}{0pt}}}
\newcommand{\squishend}{\end{list}}

\begin{document}
\section{Evaluation}

\squishlist
\item vLLM
\item SGLang
\item Autellix
\item Continuum
\item ThunderAgent
\squishend

After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    list_node = lists[0]
    assert list_node.attributes["environment"] == "list"

    items = list_node.children
    assert len(items) == 5
    assert all(item.kind == NodeKind.LIST_ITEM for item in items)

    expected = [
        "vLLM",
        "SGLang",
        "Autellix",
        "Continuum",
        "ThunderAgent",
    ]

    for item, text in zip(items, expected):
        assert text in str(item)

    # Content after the macro must not be swallowed.
    assert "After the list." in str(tree.root)


def test_standard_list_consumes_two_arguments():
    tex = r"""
\begin{document}
\section{Methods}

\begin{list}{LABEL_MARKER}{\setlength{\leftmargin}{2em}}
\item First item.
\item Second item.
\end{list}

After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    list_node = lists[0]
    assert list_node.attributes["environment"] == "list"

    items = list_node.children
    assert len(items) == 2
    assert all(item.kind == NodeKind.LIST_ITEM for item in items)

    assert "First item." in str(items[0])
    assert "Second item." in str(items[1])
    assert "After the list." in str(tree.root)

    # The two mandatory arguments are list configuration,
    # not paragraph content.
    assert "LABEL_MARKER" not in str(tree.root)


@pytest.mark.parametrize("include_command", [
    r"\input{}",
    r"\input{ }",
])
def test_empty_include_reports_diagnostic_without_crashing(include_command):
    tex = rf"""
\begin{{document}}
\section{{Introduction}}
Before invalid include.

{include_command}

\subsection{{Results}}
After invalid include.
\end{{document}}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    invalid_includes = [
        diagnostic
        for diagnostic in tree.diagnostics
        if diagnostic.code == "INVALID_INCLUDE_PATH"
    ]

    assert len(invalid_includes) == 1
    assert invalid_includes[0].file_path == "main.tex"

    # The parser must recover and continue after invalid input.
    headings = find_all(tree.root, NodeKind.HEADING)
    assert [heading.content for heading in headings] == [
        "Introduction",
        "Results",
    ]

    assert "After invalid include." in str(tree.root)



def test_custom_enumerate_macro_with_layout_settings():
    tex = r"""
\newcommand{\squishenumerate}
{
    \begin{enumerate}{}
    {
        \setlength{\itemsep}{0pt}
        \setlength{\parsep}{3pt}
        \setlength{\leftmargin}{1.5em}
    }
}

\newcommand{\squishenumerateend}
{
    \end{enumerate}
}

\begin{document}
\section{Introduction}

\squishenumerate
\item First contribution.
\item Second contribution.
\item Third contribution.
\squishenumerateend

After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    list_node = lists[0]
    assert list_node.attributes["environment"] == "enumerate"

    items = list_node.children
    assert len(items) == 3
    assert all(item.kind == NodeKind.LIST_ITEM for item in items)

    for item, expected in zip(
        items,
        ["First contribution.", "Second contribution.", "Third contribution."],
    ):
        assert expected in str(item)

    assert "After the list." in str(tree.root)


def test_inline_enumerate_environment():
    tex = r"""
\begin{document}
\section{Method}

Two scenarios are considered:
\begin{inparaenum}[(1)]
    \item The first scenario.
    \item The second scenario.
\end{inparaenum}

After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    list_node = lists[0]
    assert list_node.attributes["environment"] == "inparaenum"

    items = list_node.children
    assert len(items) == 2
    assert all(item.kind == NodeKind.LIST_ITEM for item in items)

    assert "The first scenario." in str(items[0])
    assert "The second scenario." in str(items[1])

    assert "After the list." in str(tree.root)


def test_custom_newlist_nested_environments():
    tex = r"""
\documentclass{article}

\newlist{algorithmSteps}{enumerate}{6}
\setlist[algorithmSteps]{label={\arabic*.}, nosep}

\begin{document}

\section{Algorithm}

\begin{algorithmSteps}
    \item First outer step.

    \begin{algorithmSteps}
        \item First inner step.
        \item Second inner step.
    \end{algorithmSteps}

    \item Second outer step.
\end{algorithmSteps}

\section{Results}
After the algorithm.

\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 2

    outer, inner = lists

    assert outer.attributes["environment"] == "algorithmSteps"
    assert inner.attributes["environment"] == "algorithmSteps"

    # Outer list has two direct items.
    outer_items = [
        node for node in outer.children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(outer_items) == 2

    # Inner list is nested inside the first outer item.
    assert inner in find_all(outer_items[0], NodeKind.LIST)

    # Inner list contains two items.
    inner_items = [
        node for node in inner.children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(inner_items) == 2

    # Total items: 2 outer + 2 inner.
    assert len(find_all(tree.root, NodeKind.LIST_ITEM)) == 4

    # Closing the list must restore normal heading parsing.
    headings = find_all(tree.root, NodeKind.HEADING)
    assert [heading.content for heading in headings] == [
        "Algorithm",
        "Results",
    ]

    assert "After the algorithm." in str(tree.root)


def test_paragraph_inside_nested_list_preserves_list_structure():
    tex = r"""
\begin{document}
\section{Compiler}

\begin{itemize}
    \item Conjunctions.

    \begin{itemize}
        \item Exclusive.
        \item Inclusive.
    \end{itemize}

    \paragraph{Active Tracking Set Partitioning:}
    Tracking sets are defined here.

    \item Disjunctions.
\end{itemize}

\subsection{Next Section}
After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 2

    outer, inner = lists

    outer_items = [
        node for node in outer.children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(outer_items) == 2

    inner_items = [
        node for node in inner.children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(inner_items) == 2

    # The paragraph belongs to the first outer item.
    assert "Active Tracking Set Partitioning:" in str(outer_items[0])
    assert "Tracking sets are defined here." in str(outer_items[0])

    # The second outer item must still be recognized.
    assert "Disjunctions." in str(outer_items[1])

    # The local paragraph heading must be preserved as a node.
    local_headings = find_all(
        outer_items[0],
        NodeKind.HEADING,
    )

    assert len(local_headings) == 1

    assert local_headings[0].content == (
        "Active Tracking Set Partitioning:"
    )

    assert local_headings[0].attributes["command"] == "paragraph"


def test_mismatched_inline_math_preserves_list_structure():
    tex = r"""
\begin{document}
\section{Introduction}
\begin{itemize}
\item First $x = 1\) and $y = 2$
\item Second item
\item Third item
\end{itemize}

Outside the list, we have $z = 3$.

\subsection{Next Section}
Content after the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    codes = [d.code for d in tree.diagnostics]

    assert "UNCLOSED_MATH" in codes
    assert "UNCLOSED_LIST" not in codes
    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    items = [
        node for node in lists[0].children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(items) == 3

    headings = find_all(tree.root, NodeKind.HEADING)
    assert [h.content for h in headings] == [
        "Introduction",
        "Next Section",
    ]


def test_valid_mixed_inline_math_keeps_structure():
    tex = r"""
\begin{document}
\section{Introduction}
\begin{itemize}
\item First \(x = 1\) and $y = 2$
\item Second item
\item Third item
\end{itemize}

\subsection{Next Section}
Content after the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    items = [
        node for node in lists[0].children
        if node.kind == NodeKind.LIST_ITEM
    ]
    assert len(items) == 3

    headings = find_all(tree.root, NodeKind.HEADING)
    assert len(headings) == 2


def test_mixed_math_recovery_preserves_structure():
    tex = r"""
\begin{document}
\section{Introduction}

A malformed expression \(x = 1$ appears here.

\begin{table}
\begin{tabular}{cc}
Name & $\Delta \)prev. \\
\end{tabular}
\end{table}

\begin{itemize}
\item First $y = 2\) expression
\item Second
\item Third
\item Fourth
\end{itemize}

\subsection{Next Section}
Content after the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    codes = [d.code for d in tree.diagnostics]

    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert codes.count("UNCLOSED_MATH") == 2
    assert "UNCLOSED_LIST" not in codes

    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1

    items = find_all(tree.root, NodeKind.LIST_ITEM)
    assert len(items) == 4

    headings = find_all(tree.root, NodeKind.HEADING)
    assert [h.content for h in headings] == [
        "Introduction",
        "Next Section",
    ]

    figures = find_all(tree.root, NodeKind.FIGURE)
    assert len(figures) == 1


def test_valid_inline_math_preserves_structure():
    tex = r"""
\begin{document}
\section{Introduction}

Correct math \(x = 1\) and $y = 2$.

\begin{itemize}
\item First
\item Second
\end{itemize}

\subsection{Next Section}
Some content.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    assert len(find_all(tree.root, NodeKind.LIST)) == 1
    assert len(find_all(tree.root, NodeKind.LIST_ITEM)) == 2
    assert len(find_all(tree.root, NodeKind.HEADING)) == 2
