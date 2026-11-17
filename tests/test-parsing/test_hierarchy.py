"""TV3 tests. Run from repo root: python -m pytest tests/parsing -q"""

from src.parsing.contracts import (
    NodeKind, ProcessingStatus, SourceBundle, SourceFile,
)
from src.parsing.hierarchy import (
    parse_structure,
    _resolve_include_path,
)
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


def test_missing_non_tex_input_is_not_a_hierarchy_parse_error():
    tree = parse_structure(
        bundle(
            r"\begin{document}\input{figures/plot.tikz}Body.\end{document}"
        )
    )

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "MISSING_INCLUDE" for d in tree.diagnostics)
    assert "Body." in str(tree.root)


def test_empty_source_version():
    source = SourceBundle(
        "2501.01234", 2, ProcessingStatus.EMPTY_NO_TEX,
        None, (),
    )
    tree = parse_structure(source)
    assert tree.version == 2
    assert tree.status == ProcessingStatus.EMPTY_NO_TEX
    assert tree.root is None


def test_source_error_without_files_preserves_error_status():
    source = SourceBundle(
        arxiv_id="2501.01234",
        version=1,
        status=ProcessingStatus.ERROR,
        root_file=None,
        files=(),
    )

    tree = parse_structure(source)
    assert tree.status == ProcessingStatus.ERROR
    assert tree.root is None


def test_bibliography_input_is_not_added_to_hierarchy():
    source = bundle(
        r"\begin{document}\input{refs.bib}Body.\end{document}",
        extra=(SourceFile("refs.bib", "@article{a, title={Reference}}", 1),),
    )
    tree = parse_structure(source)
    assert "Reference" not in str(tree.root)
    assert "Body." in str(tree.root)


def test_unsupported_subfile_is_not_expanded():
    source = bundle(
        r"\begin{document}\subfile{sections/body}\end{document}",
        extra=(SourceFile("sections/body.tex", "Included body.", 1),),
    )
    tree = parse_structure(source)
    assert "Included body." not in str(tree.root)
    assert r"\subfile{sections/body}" in str(tree.root)


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


def test_missing_known_tex_support_input_does_not_make_hierarchy_partial():
    source = bundle(
        r"\begin{document}\input{glyphtounicode}Body.\end{document}"
    )
    tree = parse_structure(source)
    assert tree.status == ProcessingStatus.OK
    assert "MISSING_INCLUDE" not in [d.code for d in tree.diagnostics]

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


def test_starred_enumerate_environment():
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


@pytest.mark.parametrize(
    ("environment", "body"),
    [
        ("Elist", r"\item First.\item Second."),
        ("icompact", r"\item First.\item Second."),
        ("IEEEdescription", r"\item[Term] Description."),
        ("Steps", r"\item First.\item Second."),
        ("tightitemize", r"\item First.\item Second."),
    ],
)
def test_known_template_list_environments_preserve_items(environment, body):
    tex = rf"""
\begin{{document}}
\begin{{{environment}}}
{body}
\end{{{environment}}}
\end{{document}}
"""

    tree = parse_structure(bundle(tex))

    lists = find_all(tree.root, NodeKind.LIST)
    items = find_all(tree.root, NodeKind.LIST_ITEM)

    assert len(lists) == 1
    expected_items = 1 if environment == "IEEEdescription" else 2
    assert len(items) == expected_items
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)


def test_known_bi_ei_list_alias_preserves_items():
    tex = r"""
\begin{document}
\bi
\item First.
\item Second.
\ei
\end{document}
"""

    tree = parse_structure(bundle(tex))

    lists = find_all(tree.root, NodeKind.LIST)
    items = find_all(tree.root, NodeKind.LIST_ITEM)

    assert len(lists) == 1
    assert len(items) == 2
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)


def test_known_bitem_eitem_list_alias_preserves_items():
    tex = r"""
\begin{document}
\bitem
\item First.
\item Second.
\eitem
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert len(find_all(tree.root, NodeKind.LIST)) == 1
    assert len(find_all(tree.root, NodeKind.LIST_ITEM)) == 2
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)


def test_generic_macro_definition_does_not_emit_item_warning():
    tex = r"""
\begin{document}
\def\items#1{\\item{[#1]}}
Body text.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)
    assert "Body text." in str(tree.root)


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

def test_recovers_unique_case_mismatch():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=r"\begin{document}\input{Sections/Abstract}\end{document}",
                order=0,
            ),
            SourceFile(
                relative_path="Sections/abstract.tex",
                content=r"\section{Recovered}",
                order=1,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert any(
        node.kind == "heading" and node.content == "Recovered"
        for node in result.root.children
    )


def test_case_insensitive_include():
    contents = {
        "main.tex": "",
        "Sections/introduction.tex": "Hello",
    }

    result = _resolve_include_path(
        "main.tex",
        "main.tex",
        "Sections/Introduction",
        contents,
    )

    assert result == "Sections/introduction.tex"


def test_exact_match_has_priority():
    contents = {
        "Intro.tex": "Exact",
        "intro.tex": "Different",
    }

    result = _resolve_include_path(
        "main.tex", "main.tex", "Intro.tex", contents
    )

    assert result == "Intro.tex"


def test_ambiguous_case_insensitive_include():
    contents = {
        "INTRO.tex": "A",
        "intro.tex": "B",
    }

    result = _resolve_include_path(
        "main.tex", "main.tex", "Intro.tex", contents
    )

    # Không tự chọn một trong hai file.
    assert result == "Intro.tex"

def test_ignore_include_after_end_document():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\begin{document}"
                    r"\section{Main}"
                    r"\input{Appendices/appendix}"
                    r"\end{document}"
                    r"\input{appendix.tex}"
                    r"\end{document}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path="Appendices/appendix.tex",
                content=r"\section{Appendix}",
                order=1,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "MISSING_INCLUDE"
        for d in result.diagnostics
    )

    headings = [
        node.content
        for node in result.root.children
        if node.kind == "heading"
    ]

    assert headings == ["Main", "Appendix"]

def test_path_macro_defined_in_included_preamble():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\input{preamble-shared}"
                    r"\begin{document}"
                    r"\input{sections/results}"
                    r"\end{document}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path="preamble-shared.tex",
                content=(
                    r"\providecommand{\simtabledir}"
                    r"{tables/simulation}"
                ),
                order=1,
            ),
            SourceFile(
                relative_path="sections/results.tex",
                content=(
                    r"\input{\simtabledir/results.tex}"
                ),
                order=2,
            ),
            SourceFile(
                relative_path="tables/simulation/results.tex",
                content=r"\section{Recovered Results}",
                order=3,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "MISSING_INCLUDE"
        for d in result.diagnostics
    )

def test_nested_def_path_macros():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\def\tablesPath{Tables/}"
                    r"\def\emulatorsTablesPath"
                    r"{\tablesPath emulatorsAndSpanners/}"
                    r"\begin{document}"
                    r"\input{\emulatorsTablesPath known_emulators}"
                    r"\end{document}"
                ),
                order=0,
            ),
            SourceFile(
                relative_path=(
                    "Tables/emulatorsAndSpanners/"
                    "known_emulators.tex"
                ),
                content=r"\section{Known Emulators}",
                order=1,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "MISSING_INCLUDE"
        for d in result.diagnostics
    )

def test_end_document_inside_macro_definition_before_document():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    r"\def\End{\end{document}}"
                    r"\begin{document}"
                    r"\section{Introduction}"
                    r"\section{Methodology}"
                    r"\end{document}"
                    r"\input{missing_after_end}"
                ),
                order=0,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.OK
    assert not any(
        d.code == "MISSING_INCLUDE"
        for d in result.diagnostics
    )

    headings = []
    stack = [result.root]

    while stack:
        node = stack.pop()
        if node.kind == NodeKind.HEADING:
            headings.append(node)
        stack.extend(node.children)

    assert len(headings) == 2

def test_unclosed_inline_math_does_not_swallow_heading():
    source = SourceBundle(
        arxiv_id="9999.12345",
        version=1,
        status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(
            SourceFile(
                relative_path="main.tex",
                content=(
                    "\\begin{document}\n"
                    "\\section{First}\n"
                    "An incomplete expression $x + y\n"
                    "\\section{Second}\n"
                    "The next expression is $z + 1$.\n"
                    "\\end{document}"
                ),
                order=0,
            ),
        ),
    )

    result = parse_structure(source)

    assert result.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "UNCLOSED_MATH" for d in result.diagnostics)

    stack = [result.root]
    headings = []

    while stack:
        node = stack.pop()
        if node.kind == NodeKind.HEADING:
            headings.append(node.content)
        stack.extend(reversed(node.children))

    assert headings == ["First", "Second"]

@pytest.mark.parametrize("fragment, code", [
    (
        r"\item Orphan item.",
        "ITEM_OUTSIDE_LIST",
    ),
    (
        r"\section Missing braces.",
        "INVALID_HEADING",
    ),
    (
        r"\begin{itemize}\item First.",
        "HEADING_IN_LIST",
    ),
])
def test_structural_error_is_partial_and_preserves_heading(
    fragment, code
):
    tex = (
        r"\begin{document}\section{First}"
        + "\n"
        + fragment
        + "\n"
        + r"\section{Next}After.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert any(
        d.code == code and d.file_path == "main.tex"
        for d in tree.diagnostics
    )

    headings = find_all(tree.root, NodeKind.HEADING)

    assert [h.content for h in headings] == [
        "First",
        "Next",
    ]

    assert "After." in str(tree.root)

@pytest.mark.parametrize("fragment", [
    r"Text $x + y",
    r"Text \(x + y",
])
def test_unclosed_inline_math_at_end_is_reported(fragment):
    tex = (
        r"\begin{document}"
        + fragment
        + r"\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert any(
        d.code == "UNCLOSED_MATH"
        and d.file_path == "main.tex"
        for d in tree.diagnostics
    )

    paragraphs = find_all(
        tree.root, NodeKind.PARAGRAPH
    )

    assert any(
        fragment in (node.content or "")
        for node in paragraphs
    )

def test_valid_multiline_inline_math_stays_ok():
    tex = (
        "\\begin{document}\n"
        "Inline $x+\ny$.\n"
        "\\section{Next}\n"
        "After.\n"
        "\\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    headings = find_all(
        tree.root, NodeKind.HEADING
    )

    assert [h.content for h in headings] == ["Next"]

def test_compactenum_inside_definition():
    tex = r"""
\begin{document}
\section{Preliminaries}

\begin{definition}
We assume that the tree is finite.

\begin{compactenum}

\item[$(a)$] First definition with $x=1$.

\item[$(b)$] Second definition.

$$ C_s = C_k + s $$

\item[$(c)$] Third definition.

\end{compactenum}
\end{definition}

\section{Results}
After the definition.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    items = find_all(tree.root, NodeKind.LIST_ITEM)
    formulas = find_all(tree.root, NodeKind.BLOCK_FORMULA)
    headings = find_all(tree.root, NodeKind.HEADING)

    assert len(lists) == 1
    assert lists[0].attributes["environment"] == "compactenum"

    assert len(items) == 3

    assert [item.attributes["label"] for item in items] == [
        "$(a)$",
        "$(b)$",
        "$(c)$",
    ]

    assert len(formulas) == 1
    assert formulas[0] in find_all(
        items[1], NodeKind.BLOCK_FORMULA
    )

    assert [h.content for h in headings] == [
        "Preliminaries",
        "Results",
    ]

    assert "After the definition." in str(tree.root)


@pytest.mark.parametrize(
    "environment",
    [
        "IEEEitemize",
        "myitemize",
        "pts",
        "highlights",
        "keypoints",
        "affiliations",
        "romanenumerate",
    ],
)
def test_common_template_list_environments_preserve_items(environment):
    tex = (
        r"\begin{document}\section{Method}"
        + "\n"
        + rf"\begin{{{environment}}}\n"
        + r"\item First item.\n"
        + r"\item Second item.\n"
        + rf"\end{{{environment}}}\n"
        + r"After.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert lists[0].attributes["environment"] == environment
    assert len(find_all(lists[0], NodeKind.LIST_ITEM)) == 2


def test_parameterized_list_macro_preserves_items():
    tex = r"""
\documentclass{article}
\newcommand{\items}[1]{\begin{itemize} #1 \end{itemize}}
\begin{document}
\section{Method}
\items{
  \item First item.
  \item Second item.
}
After.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert len(find_all(lists[0], NodeKind.LIST_ITEM)) == 2


def test_two_argument_list_wrapper_uses_the_list_body_argument():
    tex = r"""
\documentclass{article}
\newcommand{\itm}[2]{{\begin{itemize}[leftmargin=#1pt]#2\end{itemize}}}
\begin{document}
\itm{25}{
  \item First item.
  \item Second item.
}
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert len(find_all(lists[0], NodeKind.LIST_ITEM)) == 2


@pytest.mark.parametrize("environment", ["algor", "nomenclature"])
def test_supported_list_like_environments_preserve_items(environment):
    tree = parse_structure(bundle(
        "\\begin{document}\n"
        f"\\begin{{{environment}}}\n"
        "\\item First item.\n"
        "\\item Second item.\n"
        f"\\end{{{environment}}}\n"
        "\\end{document}"
    ))

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)
    assert len(find_all(tree.root, NodeKind.LIST_ITEM)) == 2


def test_parameterized_list_opener_preserves_items_until_explicit_end():
    tex = r"""
\documentclass{article}
\newcommand{\customitemize}[1]{%
  \begin{enumerate}[label={(#1\arabic*)}, ref=#1\arabic*]
}
\begin{document}
\customitemize{P}
  \item First item.
  \item Second item.
\end{enumerate}
After.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not any(
        diagnostic.code in {"ITEM_OUTSIDE_LIST", "LIST_MISMATCH"}
        for diagnostic in tree.diagnostics
    )
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert len(find_all(lists[0], NodeKind.LIST_ITEM)) == 2


def test_comment_macro_hides_items_from_hierarchy():
    tex = r"""
\newcommand{\comment}[1]{}
\begin{document}
\comment{\item hidden draft content}
Visible text.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not any(d.code == "ITEM_OUTSIDE_LIST" for d in tree.diagnostics)
    assert "Visible text." in str(tree.root)
    assert "hidden draft content" not in str(tree.root)

@pytest.mark.parametrize("env", [
    "packed_itemize",
    "packed_enumerate",
])
def test_packed_list_aliases(env):
    tex = (
        r"\begin{document}\section{Method}"
        + "\n"
        + r"\begin{" + env + "}\n"
        + r"\item[\Contrib] First item." + "\n"
        + r"\item Second item." + "\n"
        + r"\end{" + env + "}\n"
        + r"\section{Results}After.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)

    assert len(lists) == 1
    assert lists[0].attributes["environment"] == env

    items = find_all(lists[0], NodeKind.LIST_ITEM)

    assert len(items) == 2
    assert items[0].attributes["label"] == r"\Contrib"

    assert "First item." in str(items[0])
    assert "Second item." in str(items[1])

    headings = find_all(tree.root, NodeKind.HEADING)

    assert [h.content for h in headings] == [
        "Method", "Results"
    ]

@pytest.mark.parametrize('name, definition, use, included', [
    ('MYITEMIZE', r'\newenvironment{MYITEMIZE}{\begin{list}{$\bullet$}{\setlength{\leftmargin}{15mm}\setlength{\itemsep}{1pt}}}{\end{list}}', '', False),
    ('attributes', r'\newenvironment{attributes}[1][]{\begin{itemize}[label=\textbullet, leftmargin=3.7em,#1]}{\end{itemize}}', '[itemsep=0pt]', True),
    ('packed_itemize_custom', r'\newenvironment{packed_itemize_custom}{\vspace{-0.1ex}\begin{list}{\labelitemi}{\leftmargin=1.7em}\setlength{\itemsep}{0pt}}{\end{list}\vspace{-0.1ex}}', '', True),
    ('ENUMERATE', r'\newenvironment{ENUMERATE}{\begin{enumerate}\setlength{\itemsep}{1pt}}{\end{enumerate}}', '', False),
])
def test_custom_list_wrapper(name, definition, use, included):
    preamble = r'\input{macros}' if included else definition
    extra = (SourceFile('macros.tex', definition, 1),) if included else ()
    tex = (preamble + '\n' + r'\begin{document}\section{Method}' + '\n'
           + r'\begin{' + name + '}' + use + '\n'
           + r'\item[Label] First.' + '\n' + r'\item Second.' + '\n'
           + r'\end{' + name + '}' + '\n' + r'\section{Results}After.\end{document}')
    tree = parse_structure(bundle(tex, extra=extra))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1 and lists[0].attributes['environment'] == name
    items = find_all(lists[0], NodeKind.LIST_ITEM)
    assert len(items) == 2 and items[0].attributes['label'] == 'Label'
    assert 'First.' in str(items[0]) and 'Second.' in str(items[1])
    assert '[itemsep=0pt]' not in str(tree.root)
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ['Method', 'Results']

@pytest.mark.parametrize('definition', [
    r'\newenvironment{custom}{\begin{quote}}{\end{quote}}',
    r'%\newenvironment{custom}{\begin{itemize}}{\end{itemize}}',
    r'\newenvironment{custom}{\begin{itemize}}{\end{enumerate}}',
])
def test_non_list_wrapper_not_promoted(definition):
    tex = definition + '\n' + r'\begin{document}\begin{custom}\item Orphan.\end{custom}\section{Next}After.\end{document}'
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == 'ITEM_OUTSIDE_LIST' for d in tree.diagnostics)
    assert not find_all(tree.root, NodeKind.LIST)

def test_asparaenum_with_optional_argument():
    tex = r"""
\begin{document}
\section{Method}

\begin{asparaenum}[\rm (a)]
    \item First statement.

    \begin{equation}
        x = y + 1
    \end{equation}

    \item Second statement.
\end{asparaenum}

\section{Results}
After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    items = find_all(tree.root, NodeKind.LIST_ITEM)
    headings = find_all(tree.root, NodeKind.HEADING)
    formulas = find_all(tree.root, NodeKind.BLOCK_FORMULA)

    assert len(lists) == 1
    assert lists[0].attributes["environment"] == "asparaenum"

    assert len(items) == 2
    assert len(formulas) == 1

    assert formulas[0] in find_all(
        items[0], NodeKind.BLOCK_FORMULA
    )

    assert [h.content for h in headings] == [
        "Method", "Results"
    ]

    assert r"\rm (a)" not in str(tree.root)

def test_asparaenum_with_optional_argument():
    tex = r"""
\begin{document}
\section{Method}

\begin{asparaenum}[\rm (a)]
    \item First statement.

    \begin{equation}
        x = y + 1
    \end{equation}

    \item Second statement.
\end{asparaenum}

\section{Results}
After the list.
\end{document}
"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    lists = find_all(tree.root, NodeKind.LIST)
    items = find_all(tree.root, NodeKind.LIST_ITEM)
    headings = find_all(tree.root, NodeKind.HEADING)
    formulas = find_all(tree.root, NodeKind.BLOCK_FORMULA)

    assert len(lists) == 1
    assert lists[0].attributes["environment"] == "asparaenum"

    assert len(items) == 2
    assert len(formulas) == 1

    assert formulas[0] in find_all(
        items[0], NodeKind.BLOCK_FORMULA
    )

    assert [h.content for h in headings] == [
        "Method", "Results"
    ]

    assert r"\rm (a)" not in str(tree.root)

# Append to tests/parsing/test_hierarchy.py (uses existing bundle/find_all/imports).

def test_plain_def_list_macros():
    tex = r'''
\def\begenu{\begin{enumerate}}
\def\endenu{\end{enumerate}}
\begin{document}
\section{Method}
\begenu
\item[First] Alpha.
\begin{equation}x=1\end{equation}
\item Beta.
\endenu
\section{Next}After.
\end{document}
'''
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert lists[0].attributes['environment'] == 'enumerate'
    items = find_all(lists[0], NodeKind.LIST_ITEM)
    assert len(items) == 2
    assert items[0].attributes['label'] == 'First'
    assert len(find_all(items[0], NodeKind.BLOCK_FORMULA)) == 1
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ['Method','Next']
    assert 'After.' in str(tree.root)


def test_parameterized_def_not_structural_alias():
    tree = parse_structure(bundle(r'\def\begenu#1{\begin{enumerate}}\begin{document}\begenu{X}\item Orphan.\end{document}'))
    assert not find_all(tree.root, NodeKind.LIST)
    assert any(d.code == 'ITEM_OUTSIDE_LIST' for d in tree.diagnostics)


def test_in_body_custom_list_definition():
    body = r'''
\newenvironment{resultlist}{%
 \begin{list}{$\bullet$}{%
 \setlength{\leftmargin}{1.25em}%
 \setlength{\itemsep}{0.2em}%
 }%
}{%
 \end{list}%
}
\begin{resultlist}
\item Alpha.
\item Beta.
\end{resultlist}
'''
    tree = parse_structure(bundle(r'\begin{document}\section{Results}\input{content/results}\section{Next}After.\end{document}', extra=(SourceFile('content/results.tex', body, 1),)))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    lists = find_all(tree.root, NodeKind.LIST)
    assert len(lists) == 1
    assert lists[0].attributes['environment'] == 'resultlist'
    assert len(find_all(lists[0], NodeKind.LIST_ITEM)) == 2
    assert 'Alpha.' in str(lists[0]) and 'Beta.' in str(lists[0])
    assert r'\newenvironment' not in str(tree.root)
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ['Results','Next']


# Append to tests/parsing/test_hierarchy.py.
@pytest.mark.parametrize('setup', [
    r'\let\savedsection\section',
    r'\let\section = \savedsection',
    r'\titleformat{\section}[display]{\bfseries}{\thesection}{1em}{\textbf{#1}}[]',
    r'\titlespacing*{\subsection}{0pt}{4pt}{1pt}',
    r'\RenewDocumentCommand{\section}{s o m}{\savedsection{#3}}',
])
def test_heading_config_is_not_heading(setup):
    tex = r'\begin{document}\section{First}Before.' + '\n' + setup + '\n' + r'\section{Next}After.\end{document}'
    result = parse_structure(bundle(tex))
    assert result.status == ProcessingStatus.OK
    assert not result.diagnostics
    assert [h.content for h in find_all(result.root, NodeKind.HEADING)] == ['First', 'Next']
    assert 'Before.' in str(result.root) and 'After.' in str(result.root)

@pytest.mark.parametrize('heading', [
    r'\section{\label{sec:intro}Introduction\protect\\}',
    r'\section[Short title]{Long title:\texorpdfstring{\\}{ }Second line}',
    r'\section{Escaped \{brace\} and line break \\}',
])
def test_heading_with_escaped_braces_and_linebreak(heading):
    result = parse_structure(bundle(r'\begin{document}' + heading + '\n' + r'Body.\section{Next}After.\end{document}'))
    assert result.status == ProcessingStatus.OK
    assert not result.diagnostics
    headings = find_all(result.root, NodeKind.HEADING)
    assert len(headings) == 2 and headings[-1].content == 'Next'
    assert 'Body.' in str(headings[0])
    assert 'After.' in str(headings[1])


def test_invalid_heading_still_partial_after_config():
    result = parse_structure(bundle(r'\begin{document}\titleformat{\section}{\bfseries}{}{0pt}{}\section Missing title.\section{Next}After.\end{document}'))
    assert result.status == ProcessingStatus.PARTIAL_PARSE
    assert sum(d.code == 'INVALID_HEADING' for d in result.diagnostics) == 1
    assert [h.content for h in find_all(result.root, NodeKind.HEADING)] == ['Next']
    assert 'Missing title.' in str(result.root) and 'After.' in str(result.root)

# Append to tests/parsing/test_hierarchy.py.
@pytest.mark.parametrize("expression", [
    r"\$$25$/hour",
    r"\$$1.68$ / \$$1.79$ per query",
    r"$x$ and \$5 and $y$",
    r"\\$x$",  # line-break command followed by math
])
def test_math_escaped_currency_and_adjacent_dollars(expression):
    tex = (r"\begin{document}\section{First}" + "\n" + expression
           + "\n" + r"\section{Next}Body $z$.\end{document}")
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["First", "Next"]
    assert not find_all(tree.root, NodeKind.BLOCK_FORMULA)
    assert any(expression in p.content for p in find_all(tree.root, NodeKind.PARAGRAPH))


def test_math_jss_code_input_is_literal():
    tex = r"""\begin{document}
\section{First}
\begin{CodeChunk}
\begin{CodeInput}
R> A = as.numeric(bmt$group > 1)
R> B = x %% 2
R> literal = "\input{not-a-file}"
R> literal = "\section{Fake}"
\end{CodeInput}
\end{CodeChunk}
\section{Next}
Body $z$.
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["First", "Next"]
    assert not find_all(tree.root, NodeKind.BLOCK_FORMULA)
    assert any("bmt$group" in p.content and "x %% 2" in p.content
               for p in find_all(tree.root, NodeKind.PARAGRAPH))


def test_math_escaped_dollar_pair_is_not_display_end():
    tex = r"""\begin{document}
\section{First}
$$x=\text{\$} + y$$
\section{Next}
Body $z$.
\end{document}"""
    tree = parse_structure(bundle(tex))
    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics
    assert len(find_all(tree.root, NodeKind.BLOCK_FORMULA)) == 1
    assert [h.content for h in find_all(tree.root, NodeKind.HEADING)] == ["First", "Next"]

def test_recovery_keeps_equation_and_list_before_next_heading():
    tex = r"""\begin{document}
\section{A}
Broken $x

\begin{equation}y=2\end{equation}
\begin{itemize}\item Retained $z$\end{itemize}
\section{B}Next.
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "UNCLOSED_MATH" for d in tree.diagnostics)

    assert [
        h.content for h in find_all(tree.root, NodeKind.HEADING)
    ] == ["A", "B"]

    assert [
        f.content for f in find_all(tree.root, NodeKind.BLOCK_FORMULA)
    ] == [r"\begin{equation}y=2\end{equation}"]

    assert len(find_all(tree.root, NodeKind.LIST_ITEM)) == 1

    assert any(
        "Broken $x" in p.content
        for p in find_all(tree.root, NodeKind.PARAGRAPH)
    )


def test_recovery_keeps_next_display_math_and_heading():
    tex = r"""\begin{document}
\section{A}
Broken $$x
\section{B}Next.
$$y$$
\section{C}Last.
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "UNCLOSED_MATH" for d in tree.diagnostics)

    assert [
        h.content for h in find_all(tree.root, NodeKind.HEADING)
    ] == ["A", "B", "C"]

    assert [
        f.content for f in find_all(tree.root, NodeKind.BLOCK_FORMULA)
    ] == ["$$y$$"]


def test_recovery_keeps_next_list_item_and_parent():
    tex = r"""\begin{document}
\section{A}
\begin{itemize}
\item Broken $x
\item Good $y$
\end{itemize}
\section{B}Next.
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE
    assert [d.code for d in tree.diagnostics] == ["UNCLOSED_MATH"]

    lists = find_all(tree.root, NodeKind.LIST)

    assert len(lists) == 1

    assert [n.kind for n in lists[0].children] == [
        NodeKind.LIST_ITEM,
        NodeKind.LIST_ITEM,
    ]

    assert "Good $y$" in str(lists[0].children[1])


def test_recovery_display_escaped_currency_not_closer():
    formula = r"$$x=\text{\$$5$} + y$$"

    tex = (
        r"\begin{document}\section{A}"
        + "\n"
        + formula
        + "\n"
        + r"\section{B}Next.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK

    assert [
        f.content for f in find_all(tree.root, NodeKind.BLOCK_FORMULA)
    ] == [formula]


def test_recovery_valid_aligned_math_and_multiline_text():
    formula = r"""\[
\begin{aligned}
x&=1\\
y&=2
\end{aligned}
\]"""

    tex = (
        r"\begin{document}\section{A}"
        + "\n"
        + formula
        + "\n"
        + r"\section{B}Next.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK

    assert [
        f.content for f in find_all(tree.root, NodeKind.BLOCK_FORMULA)
    ] == [formula]

    assert [
        h.content for h in find_all(tree.root, NodeKind.HEADING)
    ] == ["A", "B"]


def test_unclosed_display_preserves_inline_heading():
    tex = r"""\begin{document}
\section{First}
Broken $$x
Text \paragraph{Inline heading} Retained.
\section{Next}
More.
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert [d.code for d in tree.diagnostics] == [
        "UNCLOSED_MATH"
    ]

    assert [
        h.content
        for h in find_all(tree.root, NodeKind.HEADING)
    ] == [
        "First",
        "Inline heading",
        "Next",
    ]

@pytest.mark.parametrize(
    "opener,closer,wrap_array",
    [
        (r"\[", r"\]", False),
        ("$$", "$$", True),
    ],
)
def test_display_math_with_nested_tabular(
    opener, closer, wrap_array
):
    body = r"""\begin{tabular}{cc}
$x$ & $y$ \\[2em]
$1$ & $2$
\end{tabular}"""

    if wrap_array:
        body = (
            r"\begin{array}{c}"
            + "\n"
            + body
            + "\n"
            + r"\end{array}"
        )

    formula = (
        opener + "\n" + body + "\n" + closer
    )

    tex = (
        r"\begin{document}\section{A}"
        + "\n"
        + formula
        + "\n"
        + r"\section{B}Next.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    assert [
        f.content
        for f in find_all(
            tree.root, NodeKind.BLOCK_FORMULA
        )
    ] == [formula]

    assert [
        h.content
        for h in find_all(
            tree.root, NodeKind.HEADING
        )
    ] == ["A", "B"]

    assert not find_all(tree.root, NodeKind.FIGURE)


def test_broken_display_preserves_following_table():
    tex = r"""\begin{document}
\section{A}
Broken $$x
\begin{table}
\begin{tabular}{c}
$x$
\end{tabular}
\end{table}
\section{B}
$$y$$
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert any(
        d.code == "UNCLOSED_MATH"
        for d in tree.diagnostics
    )

    assert len(
        find_all(tree.root, NodeKind.FIGURE)
    ) == 1

    assert [
        f.content
        for f in find_all(
            tree.root, NodeKind.BLOCK_FORMULA
        )
    ] == ["$$y$$"]

    assert [
        h.content
        for h in find_all(
            tree.root, NodeKind.HEADING
        )
    ] == ["A", "B"]

@pytest.mark.parametrize(
    "opening,closing",
    [
        (r"\begin{eqnarray}", r"\eeqa"),
        (r"\beqa", r"\end{eqnarray}"),
        (r"\beqa", r"\eeqa"),
    ],
)
def test_plain_def_eqnarray_aliases(opening, closing):
    formula = opening + "\nx &=& 1\n" + closing

    tex = (
        r"\def \beqa{\begin{eqnarray}}"
        r"\def \eeqa{\end{eqnarray}}"
        r"\begin{document}\section{A}"
        + "\n" + formula + "\n"
        + r"\section{B}Retained.\end{document}"
    )

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.OK
    assert not tree.diagnostics

    assert [
        f.content
        for f in find_all(tree.root, NodeKind.BLOCK_FORMULA)
    ] == [formula]

    assert [
        n.content
        for n in find_all(tree.root, NodeKind.HEADING)
    ] == ["A", "B"]


def test_unbalanced_nested_comment_recovery():
    tex = r"""\begin{document}
\section{A}
\begin{comment}
\begin{comment}
\section{Hidden}
\end{comment}
\section{B}Retained.
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert [d.code for d in tree.diagnostics] == [
        "UNCLOSED_ENV"
    ]

    assert [
        n.content
        for n in find_all(tree.root, NodeKind.HEADING)
    ] == ["A", "B"]


def test_unclosed_comment_without_end_stays_hidden():
    tex = r"""\begin{document}
\section{A}
\begin{comment}
\section{Hidden}
\end{document}"""

    tree = parse_structure(bundle(tex))

    assert tree.status == ProcessingStatus.PARTIAL_PARSE

    assert [d.code for d in tree.diagnostics] == [
        "UNCLOSED_ENV"
    ]

    assert [
        n.content
        for n in find_all(tree.root, NodeKind.HEADING)
    ] == ["A"]

"""Regression tests for TV3 diagnostic original-source provenance."""

import pytest

from src.parsing.contracts import (
    SourceBundle,
    SourceFile,
    ProcessingStatus,
)

from src.parsing.hierarchy import (
    parse_structure,
    _expand_bundle,
)


# ============================================================
# Helpers
# ============================================================

def make_bundle(root_text, extra=(), root="main.tex"):
    return SourceBundle(
        arxiv_id="2501.01234",
        version=1,
        status=ProcessingStatus.OK,
        root_file=root,
        files=(
            SourceFile("main.tex", root_text, 0),
            *extra,
        ),
    )


def locations(result, code):
    return [
        (d.file_path, d.line)
        for d in result.diagnostics
        if d.code == code
    ]


# ============================================================
# 1. Original source line mapping
# ============================================================

def test_provenance_root_comments_and_escaped_percent():
    tex = (
        "\\begin{document}\n"
        "% \\item hidden\n"
        "Price 10\\%\n"
        "\\item Orphan\n"
        "\\section{Next}Retained.\n"
        "\\end{document}"
    )

    result = parse_structure(make_bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert locations(result, "ITEM_OUTSIDE_LIST") == [
        ("main.tex", 4)
    ]

    assert "Retained." in str(result.root)


# ============================================================
# 2. Nested and repeated includes
# ============================================================

def test_provenance_nested_repeated_includes_and_return_to_parent():
    tex = (
        "\\begin{document}\n"
        "\\input{sections/a}\n"
        "\\input{sections/a}\n"
        "\\item Root\n"
        "\\end{document}"
    )

    extra = (
        SourceFile(
            "sections/a.tex",
            "% ignored\n\\input{b}\n\\item A\n",
            1,
        ),
        SourceFile(
            "sections/b.tex",
            "\\item B\n",
            2,
        ),
    )

    result = parse_structure(make_bundle(tex, extra))

    assert locations(result, "ITEM_OUTSIDE_LIST") == [
        ("sections/b.tex", 1),
        ("sections/a.tex", 3),
        ("sections/b.tex", 1),
        ("sections/a.tex", 3),
        ("main.tex", 4),
    ]


def test_provenance_same_line_include():
    tex = (
        "\\begin{document}\n"
        "Text \\input{a} \\item Root\n"
        "\\end{document}"
    )

    extra = (
        SourceFile("a.tex", "\n\\item Child\n", 1),
    )

    result = parse_structure(make_bundle(tex, extra))

    assert locations(result, "ITEM_OUTSIDE_LIST") == [
        ("a.tex", 2),
        ("main.tex", 2),
    ]


# ============================================================
# 3. Conditional branch mapping
# ============================================================

def test_provenance_selected_conditional_branch_and_after():
    tex = (
        "\\begin{document}\n"
        "\\IfFileExists{a.tex}\n"
        "{\n"
        "\\item True\n"
        "\\input{missing}\n"
        "}\n"
        "{\n"
        "\\item False\n"
        "}\n"
        "\\item After\n"
        "\\end{document}"
    )

    extra = (
        SourceFile("a.tex", "Exists.", 1),
    )

    result = parse_structure(make_bundle(tex, extra))

    assert locations(result, "ITEM_OUTSIDE_LIST") == [
        ("main.tex", 4),
        ("main.tex", 10),
    ]

    assert locations(result, "MISSING_INCLUDE") == [
        ("main.tex", 5)
    ]

    assert "Missing include: missing.tex" in [
        d.message for d in result.diagnostics
    ]

    assert "False" not in str(result.root)


# ============================================================
# 4. Include diagnostics
# ============================================================

def test_provenance_missing_include_reports_caller_not_target():
    tex = (
        "\\begin{document}\n"
        "\\input{a}\n"
        "\\section{Next}Retained.\n"
        "\\end{document}"
    )

    extra = (
        SourceFile(
            "a.tex",
            "% first line\n\\input{absent}\n",
            1,
        ),
    )

    result = parse_structure(make_bundle(tex, extra))

    assert locations(result, "MISSING_INCLUDE") == [
        ("a.tex", 2)
    ]

    assert "absent.tex" in next(
        d.message
        for d in result.diagnostics
        if d.code == "MISSING_INCLUDE"
    )

    assert "Retained." in str(result.root)


@pytest.mark.parametrize(
    "command",
    [
        r"\input{}",
        r"\input{/outside}",
    ],
)
def test_provenance_invalid_include(command):
    tex = (
        "\\begin{document}\n"
        + command
        + "\n\\end{document}"
    )

    result = parse_structure(make_bundle(tex))

    assert locations(result, "INVALID_INCLUDE_PATH") == [
        ("main.tex", 2)
    ]


def test_provenance_cycle():
    tex = (
        "\\begin{document}\n"
        "\\input{a}\n"
        "\\end{document}"
    )

    extra = (
        SourceFile(
            "a.tex",
            "\n\\input{main}\n",
            1,
        ),
    )

    result = parse_structure(make_bundle(tex, extra))

    assert locations(result, "INCLUDE_CYCLE") == [
        ("a.tex", 2)
    ]

    assert result.status == ProcessingStatus.PARTIAL_PARSE


# ============================================================
# 5. Structural diagnostics
# ============================================================

@pytest.mark.parametrize(
    "body,code",
    [
        (
            r"\section",
            "INVALID_HEADING",
        ),
        (
            r"\end{enumerate}",
            "LIST_MISMATCH",
        ),
        (
            r"Broken $$x",
            "UNCLOSED_MATH",
        ),
        (
            r"\begin{equation}x",
            "UNCLOSED_ENV",
        ),
        (
            r"\begin{itemize}\item A",
            "UNCLOSED_LIST",
        ),
        (
            "\\begin{itemize}\n"
            "\\item A\n"
            "\\section{Next}",
            "HEADING_IN_LIST",
        ),
        (
            r"Broken $x",
            "UNCLOSED_MATH",
        ),
    ],
)
def test_provenance_structural_diagnostics(body, code):
    line = 4 if code == "HEADING_IN_LIST" else 2

    tex = (
        "\\begin{document}\n"
        + body
        + "\n\\end{document}"
    )

    result = parse_structure(make_bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert ("main.tex", line) in locations(result, code)


# ============================================================
# 6. Unknown provenance
# ============================================================

def test_provenance_root_unknown_no_fake_line():
    result = parse_structure(
        make_bundle("Text.", root=None)
    )

    assert result.status == ProcessingStatus.ROOT_NOT_FOUND

    assert locations(result, "ROOT_NOT_FOUND") == [
        (None, None)
    ]


def test_provenance_mapping_unknown_offset_no_fake_line():
    expanded = _expand_bundle(
        make_bundle("abc"),
        [],
    )

    assert expanded.source_line_at(0) == 1

    assert expanded.source_line_at(-1) is None

    assert expanded.source_line_at(
        len(expanded.text)
    ) is None

@pytest.mark.parametrize(
    "env",
    ["eqnarray", "eqnarray*"],
)
def test_eqnarray_closing_alias_noindent(env):
    formula = "\\beam\nx &=& 1\n\\eeam"

    tex = (
        rf"\newcommand{{\beam}}{{\begin{{{env}}}}}"
        "\n"
        rf"\newcommand{{\eeam}}{{\end{{{env}}}\noindent}}"
        "\n"
        r"\begin{document}"
        "\n"
        r"\section{A}"
        "\n"
        + formula
        + "\n"
        + r"\section{B}Retained.\end{document}"
    )

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.OK
    assert not result.diagnostics

    formulas = find_all(
        result.root,
        NodeKind.BLOCK_FORMULA,
    )

    assert [n.content for n in formulas] == [
        formula
    ]

    headings = find_all(
        result.root,
        NodeKind.HEADING,
    )

    assert [n.content for n in headings] == [
        "A", "B"
    ]

    paragraphs = find_all(
        result.root,
        NodeKind.PARAGRAPH,
    )

    assert [n.content for n in paragraphs] == [
        "Retained."
    ]

@pytest.mark.parametrize(
    "body",
    [
        r"\end{eqnarray}\section{Real heading}",
        r"\end{eqnarray}Visible text",
        r"\end{eqnarray}\noindent Extra text",
        r"\end{itemize}\noindent",
    ],
)
def test_does_not_expand_macro_with_other_suffix(body):
    from src.parsing.hierarchy import (
        _simple_structural_body,
    )

    assert _simple_structural_body(body) is None

@pytest.mark.parametrize(
    "env",
    ["equation", "eqnarray", "eqnarray*"],
)
def test_early_body_math_alias(env):
    definition = (
        rf"\def \opening{{\begin{{{env}}}}}"
        "\n"
        rf"\def \closing{{\end{{{env}}}}}"
    )

    formula = (
        rf"\begin{{{env}}}"
        "\nx=1\n"
        r"\closing"
    )

    second = "\\opening\ny=2\n\\closing"

    tex = (
        r"\begin{document}"
        "\n"
        + definition
        + "\n"
        + r"\begin{abstract}Summary.\end{abstract}"
        + "\n"
        + r"\section{A}"
        + "\n"
        + formula
        + "\n"
        + second
        + "\n"
        + r"\section{B}Retained.\end{document}"
    )

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.OK
    assert not result.diagnostics

    assert [
        n.content
        for n in find_all(
            result.root, NodeKind.BLOCK_FORMULA
        )
    ] == [formula, second]

    assert [
        n.content
        for n in find_all(
            result.root, NodeKind.HEADING
        )
    ] == ["Abstract", "A", "B"]

    assert "Retained." in [
        n.content
        for n in find_all(
            result.root, NodeKind.PARAGRAPH
        )
    ]

@pytest.mark.parametrize(
    "body",
    [
        # Used before definition.
        "\\alias\n\\def\\alias{\\end{eqnarray}}",

        # Later redefinition.
        "\\def\\alias{\\end{eqnarray}}\n"
        "\\renewcommand{\\alias}{Text}",

        # Local group.
        "{\n\\def\\alias{\\end{eqnarray}}\n}",

        # Conditional definition.
        "\\iffalse\n"
        "\\def\\alias{\\end{eqnarray}}\n"
        "\\fi",

        # Definition after the first heading.
        "\\section{A}\n"
        "\\def\\alias{\\end{eqnarray}}",

        # Parameterized macro.
        "\\def\\alias#1{\\end{eqnarray}}",

        # Unsupported early-body list alias.
        "\\def\\alias{\\end{itemize}}",
    ],
)
def test_early_body_alias_reject_unsafe(body):
    from src.parsing.hierarchy import (
        _collect_simple_structural_macros,
    )

    tex = (
        "\\begin{document}\n"
        + body
        + "\n\\end{document}"
    )

    macros, _ = _collect_simple_structural_macros(tex)

    assert "alias" not in macros

@pytest.mark.parametrize(
    "guard,env,a,b",
    [
        (
            r"\ifdefined\layout",
            "table",
            "[H]",
            "[t]",
        ),
        (
            r"\ificlr",
            "wrapfigure",
            r"[15]{r}{0.55\textwidth}",
            r"[17]{r}{0.58\textwidth}",
        ),
        (
            r"\ifdefined\layout",
            "table*",
            "[htbp]",
            "[p]",
        ),
    ],
)
def test_conditional_float_recovery(guard, env, a, b):
    raw = "\n".join([
        guard,
        rf"\begin{{{env}}}" + a,
        r"\else",
        rf"\begin{{{env}}}" + b,
        r"\fi",
        r"\caption{Caption}",
        rf"\end{{{env}}}",
    ])

    tex = "\n".join([
        r"\begin{document}",
        r"\section{A}",
        "Before.",
        raw,
        r"\section{B}",
        "Retained.",
        r"\end{document}",
    ])

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert [
        (d.code, d.file_path, d.line)
        for d in result.diagnostics
    ] == [
        ("UNCLOSED_ENV", "main.tex", 5)
    ]

    assert [
        n.content
        for n in find_all(result.root, NodeKind.FIGURE)
    ] == [raw]

    assert [
        n.content
        for n in find_all(result.root, NodeKind.HEADING)
    ] == ["A", "B"]

    assert [
        n.content
        for n in find_all(result.root, NodeKind.PARAGRAPH)
    ] == ["Before.", "Retained."]

@pytest.mark.parametrize(
    "head",
    [
        # Different environments.
        "\\ifdefined\\layout\n"
        "\\begin{table}[t]\n"
        "\\else\n"
        "\\begin{figure}[t]\n"
        "\\fi",

        # Extra visible content in a branch.
        "\\ifdefined\\layout\n"
        "\\begin{table}[t]\n"
        "Visible\n"
        "\\else\n"
        "\\begin{table}[t]\n"
        "\\fi",

        # Structural heading inside a branch.
        "\\ifdefined\\layout\n"
        "\\begin{table}[t]\n"
        "\\else\n"
        "\\begin{table}[t]\n"
        "\\section{B}\n"
        "\\fi",
    ],
)
def test_conditional_float_reject_complex(head):
    from src.parsing.hierarchy import (
        _conditional_float_opening,
    )

    start = head.index(r"\begin{table}")

    assert _conditional_float_opening(
        head,
        start,
        start + len(r"\begin{table}"),
        "table",
    ) is None


def test_normal_float_unchanged():
    raw = r"\begin{table}[t]Content.\end{table}"

    result = parse_structure(bundle(
        r"\begin{document}"
        + raw
        + r"\section{B}Retained.\end{document}"
    ))

    assert result.status == ProcessingStatus.OK

    assert [
        n.content
        for n in find_all(result.root, NodeKind.FIGURE)
    ] == [raw]


def test_conditional_float_without_closing_stays_partial():
    raw = "\n".join([
        r"\ifdefined\layout",
        r"\begin{table}[t]",
        r"\else",
        r"\begin{table}[H]",
        r"\fi",
        "Content.",
    ])

    result = parse_structure(bundle(
        "\\begin{document}\n"
        + raw
        + "\n\\end{document}"
    ))

    assert result.status == ProcessingStatus.PARTIAL_PARSE
    assert len(result.diagnostics) == 1
    assert "recovered" not in result.diagnostics[0].message

def test_conditional_wrapfigure_with_layout_commands():
    raw = "\n".join([
        r"\ificlr",
        r"\begin{wrapfigure}[15]{r}{0.42\textwidth}",
        r"    \centering",
        r"    \vspace{-\baselineskip}",
        r"\else",
        r"\begin{wrapfigure}[17]{r}{0.48\textwidth}",
        r"    \centering",
        r"\fi",
        r"\includegraphics{figure.pdf}",
        r"\caption{Ablation results}",
        r"\end{wrapfigure}",
    ])

    tex = "\n".join([
        r"\begin{document}",
        r"\section{A}",
        "Before.",
        raw,
        "After figure.",
        r"\section{B}",
        "Retained.",
        r"\end{document}",
    ])

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert [
        (d.code, d.file_path, d.line)
        for d in result.diagnostics
    ] == [
        ("UNCLOSED_ENV", "main.tex", 5)
    ]

    figures = find_all(
        result.root,
        NodeKind.FIGURE,
    )

    assert [n.content for n in figures] == [raw]

    headings = find_all(
        result.root,
        NodeKind.HEADING,
    )

    assert [n.content for n in headings] == [
        "A", "B"
    ]

    paragraphs = find_all(
        result.root,
        NodeKind.PARAGRAPH,
    )

    assert [n.content for n in paragraphs] == [
        "Before.",
        "After figure.",
        "Retained.",
    ]

@pytest.mark.parametrize(
    "extra",
    [
        "Visible paragraph.",
        r"\section{Real heading}",
        r"\caption{Real caption}",
        r"\centering Visible text",
        r"\vspace{\input{file}}",
        r"\iftrue",
        r"\begin{minipage}{1cm}",
    ],
)
def test_conditional_float_reject_nonlayout_content(extra):
    from src.parsing.hierarchy import (
        _conditional_float_opening,
    )

    opening = r"\begin{wrapfigure}"

    tex = "\n".join([
        r"\ificlr",
        opening + r"[15]{r}{0.42\textwidth}",
        r"\centering",
        extra,
        r"\else",
        opening + r"[17]{r}{0.48\textwidth}",
        r"\centering",
        r"\fi",
    ])

    start = tex.index(opening)

    assert _conditional_float_opening(
        tex,
        start,
        start + len(opening),
        "wrapfigure",
    ) is None

def test_literal_false_removes_hidden_content_and_keeps_provenance():
    from src.parsing.hierarchy import (
        _mask_simple_false_blocks,
    )

    tex = "\n".join([
        r"\begin{document}",
        r"\section{A}",
        "Before.",
        r"\iffalse",
        r"\section{Hidden}",
        r"\input{missing}",
        r"\begin{table}",
        r"\iffalse",
        r"\end{document}",
        r"\fi",
        r"\fi",
        r"\section{B}",
        "Retained.",
        r"\item Orphan",
        r"\end{document}",
    ])

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert [
        (d.code, d.file_path, d.line)
        for d in result.diagnostics
    ] == [
        ("ITEM_OUTSIDE_LIST", "main.tex", 14)
    ]

    assert [
        n.content
        for n in find_all(
            result.root, NodeKind.HEADING
        )
    ] == ["A", "B"]

    assert not find_all(
        result.root,
        NodeKind.FIGURE,
    )

    assert "Retained." in str(result.root)
    assert "Hidden" not in str(result.root)

    masked = _mask_simple_false_blocks(tex)

    assert len(masked) == len(tex)

    assert [
        i for i, c in enumerate(masked)
        if c == "\n"
    ] == [
        i for i, c in enumerate(tex)
        if c == "\n"
    ]

@pytest.mark.parametrize(
    "body",
    [
        # Conditional with an alternative branch.
        "\\iffalse\nHidden\n\\else\nVisible\n\\fi",

        # Unknown nested conditional.
        "\\iffalse\nHidden\n"
        "\\ifdefined\\layout\nText\n\\fi\n\\fi",

        # Unclosed conditional.
        "\\iffalse\nUnclosed",

        # Conditional inside a macro definition.
        "\\newcommand{\\example}{\n"
        "\\iffalse\nHidden\n\\fi\n}",

        # Literal verbatim content.
        "\\begin{verbatim}\n"
        "\\iffalse\nLiteral text\n\\fi\n"
        "\\end{verbatim}",

        # Opaque comment content.
        "\\begin{comment}\n"
        "\\iffalse\nLiteral text\n\\fi\n"
        "\\end{comment}",
    ],
)
def test_literal_false_rejects_ambiguous_or_opaque_blocks(body):
    from src.parsing.hierarchy import (
        _mask_simple_false_blocks,
    )

    tex = (
        "\\begin{document}\n"
        + body
        + "\n\\end{document}"
    )

    assert _mask_simple_false_blocks(tex) == tex

@pytest.mark.parametrize(
    "env",
    ["equation", "equation*", "eqnarray", "eqnarray*"],
)
def test_parameterized_definition_never_swallows_document(env):
    tex = "\n".join([
        r"\begin{document}",
        r"\def\beq#1{\begin{" + env + r"}\label{#1}}",
        r"\newcommand{\eeq}{\end{" + env + "}}",
        r"\begin{abstract}",
        "Abstract body.",
        r"\end{abstract}",
        r"\section{Introduction}",
        "Intro body.",
        r"\begin{equation}",
        "x=1",
        r"\end{equation}",
        r"\beq{custom} y=2 \eeq",
        r"\section{Method}",
        "Method body.",
        r"\end{document}",
    ])

    result = parse_structure(bundle(tex))

    assert result.status == ProcessingStatus.PARTIAL_PARSE

    assert [
        (d.code, d.file_path, d.line)
        for d in result.diagnostics
    ] == [
        ("UNSUPPORTED_STRUCTURAL_MACRO", "main.tex", 2)
    ]

    assert [
        n.content
        for n in find_all(
            result.root, NodeKind.HEADING
        )
    ] == [
        "Abstract",
        "Introduction",
        "Method",
    ]

    assert [
        n.content
        for n in find_all(
            result.root, NodeKind.BLOCK_FORMULA
        )
    ] == [
        "\\begin{equation}\nx=1\n\\end{equation}"
    ]

    assert r"\beq{custom} y=2 \eeq" in [
        n.content
        for n in find_all(
            result.root, NodeKind.PARAGRAPH
        )
    ]
def make_recovery_bundle(tex, extra=()):
    return SourceBundle(
        arxiv_id="2501.01234", version=1, status=ProcessingStatus.OK,
        root_file="main.tex",
        files=(SourceFile("main.tex", tex, 0), *extra),
    )


def nodes(root, kind):
    stack = [root]
    found = []
    while stack:
        node = stack.pop()
        if node.kind == kind:
            found.append(node.content)
        stack.extend(reversed(node.children))
    return found


def test_declared_parameterized_float_closer_recovers_following_text():
    tex = r"""\documentclass{article}
\newcommand\closefloat[2]{\end{tabular}\end{center}\vskip -4ex
    \caption{\label{fig:#1}#2}\end{figure}}
\begin{document}
\section{First}
\begin{figure}[ht]\begin{center}\begin{tabular}{c}
\includegraphics[width=8cm]{image.eps}
\closefloat{abc}{A caption}
Retained prose.
\section{Next}
\begin{equation}x=1\end{equation}
\end{document}"""
    result = parse_structure(make_bundle(tex))
    assert result.status == ProcessingStatus.PARTIAL_PARSE
    assert [(d.code, d.file_path, d.line) for d in result.diagnostics] == [
        ("UNCLOSED_ENV", "main.tex", 6)
    ]
    assert nodes(result.root, NodeKind.HEADING) == ["First", "Next"]
    assert nodes(result.root, NodeKind.FIGURE) == [
        '\\begin{figure}[ht]\\begin{center}\\begin{tabular}{c}\n'
        '\\includegraphics[width=8cm]{image.eps}\n'
        '\\closefloat{abc}{A caption}'
    ]
    assert "Retained prose." in nodes(result.root, NodeKind.PARAGRAPH)
    assert nodes(result.root, NodeKind.BLOCK_FORMULA) == [r"\begin{equation}x=1\end{equation}"]


@pytest.mark.parametrize("definition,call", [
    (r"\newcommand\closefloat[1]{\end{figure}Visible text}", r"\closefloat{abc}"),
    (r"\newcommand\closefloat[1]{\end{center}\end{figure}}", r"\closefloat{abc}"),
    (r"\newcommand\closefloat[2]{\caption{#1}\end{figure}}", r"\closefloat{abc}"),
    (r"\newcommand\closefloat[1]{\caption{#1}\end{figure}}", r"\closefloat{abc} trailing"),
])
def test_declared_closer_rejects_ambiguous_boundary(definition, call):
    tex = "\n".join([definition, r"\begin{document}", r"\begin{figure}", call, r"\section{Next}", r"\end{document}"])
    result = parse_structure(make_bundle(tex))
    assert result.status == ProcessingStatus.PARTIAL_PARSE
    assert any(d.code == "UNCLOSED_ENV" for d in result.diagnostics)
    assert not any("recovered declared closing macro" in d.message for d in result.diagnostics)


CHILD = r"""\ifdefined\MyFlag
\subsection{Active heading}
Active prose.
\begin{table*}[htbp]
\else
\begin{table*}[p]
\begin{minipage}{\columnwidth}
\subsection{Inactive heading}
Inactive prose.
\end{minipage}
\fi
\centering
\begin{tabular}{c}x\end{tabular}
\ifdefined\MyFlag
\end{table*}
\begin{table*}[htbp]
\else
\vspace{4pt}
\fi
\begin{tabular}{c}y\end{tabular}
\end{table*}
\section{Retained}
Retained prose.
\item orphan
"""


def test_known_flag_resolves_conditional_float_chain_and_provenance():
    root = r"""\begin{document}
\appendix
\def\MyFlag{1}
\input{appendix}
\end{document}"""
    result = parse_structure(make_bundle(root, (
        SourceFile("appendix.tex", r"\input{tables}", 1),
        SourceFile("tables.tex", CHILD, 2),
    )))
    assert nodes(result.root, NodeKind.HEADING) == ["Active heading", "Retained"]
    assert len(nodes(result.root, NodeKind.FIGURE)) == 2
    assert "Active prose." in nodes(result.root, NodeKind.PARAGRAPH)
    assert "Inactive" not in str(result.root)
    assert [(d.code, d.file_path, d.line) for d in result.diagnostics] == [
        ("ITEM_OUTSIDE_LIST", "tables.tex", 24)
    ]
    assert result.status == ProcessingStatus.PARTIAL_PARSE


@pytest.mark.parametrize("prefix,suffix", [
    ("", ""),
    ("", r"\def\MyFlag{1}"),
    ("\\iffalse\n\\def\\MyFlag{1}\n\\fi", ""),
    ("{\n\\def\\MyFlag{1}\n}", ""),
    ("\\begin{verbatim}\n\\def\\MyFlag{1}\n\\end{verbatim}", ""),
    (r"\def\MyFlag{1}", r"\let\MyFlag\undefined"),
])
def test_known_flag_does_not_guess(prefix, suffix):
    root = "\n".join([r"\begin{document}", prefix, r"\input{tables}", suffix, r"\end{document}"])
    result = parse_structure(make_bundle(root, (SourceFile("tables.tex", CHILD, 1),)))
    assert any(d.code == "UNCLOSED_ENV" for d in result.diagnostics)
    assert "Inactive heading" in str(result.root)
