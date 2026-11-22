"""Field cleaning for reference matching (TV4, Milestone 3).

Pure functions on strings, standard library only. They accept the raw forms seen in the data:
  - BibTeX fields with LaTeX markup:  P{\\"a}prer, Jean-Fran{\\c{c}}ois, {{Floquet}}, \\&
  - author strings:  "Last, First and Last, First", "First Last", "Last, F. M.", "... and others"
  - arXiv IDs as bare IDs, "arXiv:2008.05489 [cond-mat]" or URLs (arxiv.org/abs/...)
  - DOIs as bare DOIs, "doi:..." or resolver URLs
  - noisy titles from the crawler, e.g. "Nature Methods, 17, Wright, A. H., Hildebrandt, H., van den"

Main entry points:
  normalize_arxiv_id / extract_arxiv_id   -> canonical arXiv ID ("1706.03762") or None
  latex_to_text / normalize_title / title_tokens
  parse_authors / parse_author_name       -> tuple[Author, ...]
  extract_year, normalize_doi / extract_doi
  clean_record                            -> CleanedRecord (everything above in one object)
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# LaTeX -> plain text
# ---------------------------------------------------------------------------
_ACCENT_MARKS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "b": "\u0331", "d": "\u0323",
}
_SPECIAL_LETTERS = {
    "ss": "\u00df", "aa": "\u00e5", "AA": "\u00c5", "ae": "\u00e6", "AE": "\u00c6", "oe": "\u0153",
    "OE": "\u0152", "o": "\u00f8", "O": "\u00d8", "l": "\u0142", "L": "\u0141",
}
_MOJIBAKE_RE = re.compile(r"[\u00c2-\u00f4][\u0080-\u00bf]{1,3}")
_ACCENT_DOTLESS_RE = re.compile(r"\\([\"'`^~=.])\s*\{?\\([ij])\}?")   # \'\i  \'{\i}  and the mangled \'\ia
_MATH_SPAN_RE = re.compile(r"\$([^$]*)\$")


def _greek_table() -> dict[str, str]:
    table: dict[str, str] = {}
    lower = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
             "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
    upper = ["gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "upsilon", "phi", "psi", "omega"]
    uname = lambda n: "LAMDA" if n == "lambda" else n.upper()        # the Unicode name has no 'b'
    for n in lower:
        table[n] = unicodedata.lookup(f"GREEK SMALL LETTER {uname(n)}")
    for n in upper:
        table[n.capitalize()] = unicodedata.lookup(f"GREEK CAPITAL LETTER {uname(n)}")
    table.update({"varepsilon": table["epsilon"], "vartheta": "\u03d1", "varphi": table["phi"],
                  "varrho": table["rho"], "varsigma": "\u03c2"})
    return table


_GREEK = _greek_table()
_GREEK_RE = re.compile(r"\\(%s)(?![A-Za-z])\s*" % "|".join(sorted(_GREEK, key=len, reverse=True)))
_DOTLESS_RE = re.compile(r"\\([ij])(?![A-Za-z])\s*(?:\{\})?")
_SYMBOL_ACCENT_RE = re.compile(r"\\([\"'`^~=.])\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z]))")
_LETTER_ACCENT_RE = re.compile(r"\\([uvHckrbd])(?![A-Za-z])\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z]))")
_SPECIAL_RE = re.compile(r"\\(ss|aa|AA|ae|AE|oe|OE|o|O|l|L)(?![A-Za-z])(?:\{\})?\s*")
_DROP_WITH_ARG_RE = re.compile(r"\\(?:label|cite[a-z]*|ref|eqref|href)\s*\{[^{}]*\}")
_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?")
_ESCAPED_SYMBOLS = {"&": "&", "%": "%", "#": "#", "_": "_", "{": "{", "}": "}"}

# Characters that NFKD does not decompose; mapped by hand.
_FOLD = str.maketrans({
    "\u00df": "ss", "\u00f8": "o", "\u00d8": "O", "\u00e6": "ae", "\u00c6": "AE", "\u0153": "oe",
    "\u0152": "OE", "\u0142": "l", "\u0141": "L", "\u0111": "d", "\u0110": "D", "\u0131": "i",
    "\u00f0": "d", "\u00fe": "th", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
})


def _accent_repl(match: re.Match[str]) -> str:
    base = match.group(2) or match.group(3)
    return unicodedata.normalize("NFC", base + _ACCENT_MARKS[match.group(1)])


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1: 'FertÃ©' -> 'Ferté', 'SÃ¡nchez' -> 'Sánchez'."""
    def repl(m: re.Match[str]) -> str:
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)

    return _MOJIBAKE_RE.sub(repl, text)


def latex_to_text(text: str | None) -> str:
    """LaTeX markup -> readable Unicode text (mojibake and accents repaired, braces and commands stripped).

    Greek commands become Greek letters ('\\Lambda CDM' -> 'ΛCDM'); inside $...$ the sub/superscript marks
    are dropped ('H$_0$' -> 'H0') so that both match how the same title is usually written elsewhere.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", fix_mojibake(str(text)))
    text = _ACCENT_DOTLESS_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _ACCENT_MARKS[m.group(1)]), text)
    text = _DOTLESS_RE.sub(lambda m: m.group(1), text)
    text = _SYMBOL_ACCENT_RE.sub(_accent_repl, text)
    text = _LETTER_ACCENT_RE.sub(_accent_repl, text)
    text = _SPECIAL_RE.sub(lambda m: _SPECIAL_LETTERS[m.group(1)], text)
    text = _DROP_WITH_ARG_RE.sub(" ", text)
    text = _GREEK_RE.sub(lambda m: _GREEK[m.group(1)], text)
    text = text.replace("\\$", "\ue000")                     # escaped dollar must survive math stripping
    text = _MATH_SPAN_RE.sub(lambda m: re.sub(r"[_^]", "", m.group(1)), text)
    text = re.sub(r"\\([&%#_{}])", lambda m: _ESCAPED_SYMBOLS[m.group(1)], text)
    text = re.sub(r"\\[,;:! ]|\\\\", " ", text)
    text = _COMMAND_RE.sub("", text)                           # \mathbb{Z} -> {Z}, \textbf{x} -> {x}
    text = text.replace("$", "").replace("---", "-").replace("--", "-")
    text = text.replace("``", '"').replace("''", '"').replace("~", " ")
    text = text.replace("{", "").replace("}", "").replace("\ue000", "$")
    return " ".join(text.split())


def fold_ascii(text: str) -> str:
    """Strip diacritics: 'Päprer' -> 'Paprer', 'Fran\u00e7ois' -> 'Francois'."""
    text = text.translate(_FOLD)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "by", "from", "at", "as", "is",
    "are", "be", "via", "using", "its", "their", "we", "our", "this", "that", "these", "those", "into",
    "over", "under", "between", "through", "towards", "toward", "than", "vs", "versus", "not", "can",
})
_WORD_RE = re.compile(r"[^\W_]+")


def _words(text: str | None) -> list[str]:
    return _WORD_RE.findall(fold_ascii(latex_to_text(text)).lower())


def normalize_title(text: str | None) -> str:
    """Lowercase ASCII words separated by single spaces ('' for empty input)."""
    return " ".join(_words(text))


def title_tokens(text: str | None, drop_stopwords: bool = True) -> tuple[str, ...]:
    """Content words of a title. Single letters (not digits) and, optionally, stop-words are dropped."""
    tokens = [w for w in _words(text) if len(w) > 1 or w.isdigit()]
    if drop_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS]
    return tuple(tokens)


# ---------------------------------------------------------------------------
# arXiv IDs
# ---------------------------------------------------------------------------
_ARCHIVES = (
    "hep-th|hep-ph|hep-lat|hep-ex|astro-ph|cond-mat|gr-qc|math-ph|nlin|nucl-th|nucl-ex|quant-ph|physics|"
    "math|cs|q-bio|q-fin|stat|chao-dyn|solv-int|adap-org|alg-geom|dg-ga|funct-an|patt-sol|q-alg|cmp-lg|"
    "atom-ph|supr-con|mtrl-th|chem-ph|comp-gas|acc-phys|ao-sci|bayes-an|plasm-ph"
)
_NEW_ID = r"(?:0[7-9]|[1-9]\d)(?:0[1-9]|1[0-2])\.\d{4,5}"
_OLD_ID = rf"(?:{_ARCHIVES})(?:\.[A-Za-z]{{2}})?/\d{{7}}"
_VERSION = r"(?:v\d+)?"
_STRICT_ID_RE = re.compile(rf"(?P<id>{_NEW_ID}|{_OLD_ID}){_VERSION}", re.I)
_DIR_KEY_RE = re.compile(rf"(?P<new>(?:0[7-9]|[1-9]\d)(?:0[1-9]|1[0-2])-\d{{4,5}})|(?P<old>(?:{_ARCHIVES})_\d{{7}})", re.I)
_MARKED_ID_RE = re.compile(
    rf"arxiv(?:\.org/(?:abs|pdf|src)/|\s*[:/]?\s*(?:abs/)?)(?P<id>{_NEW_ID}|{_OLD_ID}){_VERSION}(?![\d])", re.I
)
_BARE_OLD_ID_RE = re.compile(rf"(?<![\w/.\-])(?P<id>{_OLD_ID}){_VERSION}(?!\d)", re.I)
_URL_PREFIX_RE = re.compile(r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf|src)/", re.I)


def normalize_arxiv_id(value: str | None) -> str | None:
    """Whole-field arXiv ID -> canonical form without version: 'arXiv:1706.03762v2' -> '1706.03762'.

    Also accepts the corpus directory key form ('1706-03762', 'hep-th_9901001'). Returns None when the
    whole value is not an arXiv ID (use extract_arxiv_id to search inside free text).
    """
    if not value:
        return None
    text = str(value).strip()
    text = _URL_PREFIX_RE.sub("", text)
    text = re.sub(r"^arxiv\s*:?\s*", "", text, flags=re.I)
    text = re.sub(r"\.pdf$", "", text, flags=re.I).strip()
    text = re.sub(r"\s*\[[^\]]*\]$", "", text)                    # trailing '[cond-mat]'
    m = _STRICT_ID_RE.fullmatch(text)
    if m:
        return m.group("id")
    m = _DIR_KEY_RE.fullmatch(text)
    if m:
        return m.group("new").replace("-", ".") if m.group("new") else m.group("old").replace("_", "/")
    return None


def extract_arxiv_id(text: str | None) -> str | None:
    """Find an arXiv ID inside free text or a URL.

    Only explicit forms are accepted: 'arXiv:2008.05489', 'arxiv.org/abs/2008.05489', or an old-style ID
    with a known archive ('hep-th/9901001'). Bare numbers such as '2005.12345' or DOI fragments never match.
    """
    if not text:
        return None
    text = str(text)
    m = _MARKED_ID_RE.search(text)
    if m:
        return m.group("id")
    m = _BARE_OLD_ID_RE.search(text)
    return m.group("id") if m else None


def find_arxiv_id(*values: str | None) -> str | None:
    """First arXiv ID found among several fields (whole-field form first, then inside text)."""
    for value in values:
        found = normalize_arxiv_id(value) or extract_arxiv_id(value)
        if found:
            return found
    return None


def arxiv_id_to_key(arxiv_id: str) -> str:
    """'1706.03762' -> '1706-03762' ; 'hep-th/9901001' -> 'hep-th_9901001' (same rule as M1's paper_dir_name)."""
    return arxiv_id.replace(".", "-").replace("/", "_")


# ---------------------------------------------------------------------------
# DOI and year
# ---------------------------------------------------------------------------
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>{}]+", re.I)
_YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d\d|20\d\d)(?!\.?\d)")


def extract_doi(text: str | None) -> str | None:
    """Lowercase DOI found in a bare DOI, a 'doi:' string or a resolver URL; None if absent."""
    if not text:
        return None
    m = _DOI_RE.search(str(text).replace("\\_", "_"))
    return m.group(0).rstrip(".,;:)]}'\"").lower() if m else None


normalize_doi = extract_doi


def extract_year(value: object) -> int | None:
    """First plausible year (1500-2099) in an int, '2019', '2019-04-02', '(2014)' or an RFC 2822 date."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value if 1500 <= value <= 2099 else None
    m = _YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------
_PARTICLES = frozenset({
    "van", "von", "der", "den", "de", "la", "le", "di", "da", "del", "della", "bin", "ibn", "al", "el",
    "ter", "ten", "dos", "du", "st", "ben", "mac",
})
_CORPORATE_WORDS = frozenset({
    "collaboration", "consortium", "team", "group", "project", "experiment", "committee", "network",
    "organization", "institute", "university", "survey",
})
_ET_AL = frozenset({"others", "et al", "et al.", "al.", "etal"})
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
_INITIALS_ONLY_RE = re.compile(r"^(?:[A-Z]\.?[\s\-]*)+$")


@dataclass(frozen=True)
class Author:
    """family: folded lowercase family name ('van der waals'); initials: uppercase given-name initials ('JF')."""

    family: str
    initials: str = ""

    @property
    def key(self) -> str:
        """Last word of the family name, the unit used for matching ('van der waals' -> 'waals')."""
        return self.family.rsplit(" ", 1)[-1]


def _top_level_positions(text: str) -> list[int]:
    depth, out = 0, []
    for ch in text:
        out.append(depth)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    return out


def _split_on_and(text: str) -> list[str]:
    """Split on ' and ' outside braces ('{Barnes and Noble}' stays whole)."""
    depth = _top_level_positions(text)
    parts, start = [], 0
    for m in re.finditer(r"\s+and\s+", text, flags=re.I):
        if depth[m.start()] == 0:
            parts.append(text[start: m.start()])
            start = m.end()
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _looks_full_name(piece: str) -> bool:
    """'A. Smith' / 'Maria Lopez' yes; 'Smith', 'J.', 'M. S.' no."""
    piece = latex_to_text(piece)
    return (" " in piece or "." in piece) and not _INITIALS_ONLY_RE.match(piece)


def _split_commas(part: str) -> list[str]:
    """Handle 'A. Smith, B. Jones' and 'Smith, A., Jones, B.'; leave 'Last, First' alone."""
    pieces = [p.strip() for p in re.split(r",\s*", part) if p.strip()]
    if len(pieces) >= 2 and all(_looks_full_name(p) for p in pieces):
        return pieces
    if len(pieces) >= 4 and len(pieces) % 2 == 0 and all(_INITIALS_ONLY_RE.match(latex_to_text(p)) for p in pieces[1::2]):
        return [f"{pieces[i]}, {pieces[i + 1]}" for i in range(0, len(pieces), 2)]
    return [part]


def _initials(given: str) -> str:
    letters: list[str] = []
    for token in re.findall(r"[^\W\d_]+", given):
        letters.extend(token if token.isupper() and len(token) <= 3 else token[0])
    return "".join(c.upper() for c in letters)


def _is_fully_braced(raw: str) -> bool:
    """'{Barnes and Noble}' -> True: one brace group spanning the whole name (BibTeX corporate author)."""
    raw = raw.strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return False
    depth = 0
    for i, ch in enumerate(raw):
        depth += (ch == "{") - (ch == "}")
        if depth == 0 and i < len(raw) - 1:
            return False
    return True


def parse_author_name(raw: str | None) -> Author | None:
    """One name ('Last, First', 'First Last', 'M. S. Rudner', 'van der Waals, J.') -> Author, or None."""
    corporate = bool(raw) and _is_fully_braced(str(raw))
    text = fold_ascii(latex_to_text(raw))
    text = " ".join(text.split()).strip(" ,;")
    if not text or text.lower().strip(". ") in _ET_AL:
        return None
    text = re.sub(r"\s+et\.?\s*al\.?$", "", text, flags=re.I)
    if corporate or any(w in _CORPORATE_WORDS for w in text.lower().split()):
        return Author(family=" ".join(_WORD_RE.findall(text.lower().replace("'", ""))))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        family, given = parts[0], parts[-1]
    else:
        tokens = text.split(" ")
        while len(tokens) > 1 and tokens[-1].lower().strip(".,") in _SUFFIXES:
            tokens.pop()
        if len(tokens) == 1:
            family, given = tokens[0], ""
        else:
            j = len(tokens) - 2
            while j >= 1 and tokens[j].lower() in _PARTICLES:
                j -= 1
            family, given = " ".join(tokens[j + 1:]), " ".join(tokens[: j + 1])
    words = _WORD_RE.findall(family.lower().replace("'", ""))
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    family = " ".join(words)
    return Author(family=family, initials=_initials(given)) if family else None


def parse_authors(raw: str | Sequence[str] | None) -> tuple[Author, ...]:
    """Author field -> tuple[Author, ...]. Accepts one string or a list of strings; 'others'/'et al.' dropped.

    Free text such as 'A. Smith, B. Jones and C. Lee' and BibTeX style 'Smith, A. and Jones, B.' both work;
    a single string with several commas and no 'and' is split only when the pattern is unambiguous.
    """
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, str) else [str(x) for x in raw if x is not None]
    names: list[str] = []
    for item in items:
        item = re.sub(r",\s+and\s+", ", ", item, flags=re.I)         # Oxford comma: 'A, B, and C'
        for part in _split_on_and(item):
            for sub in part.split(";"):
                names.extend(_split_commas(sub.strip()) if sub.strip() else [])
    return tuple(a for a in map(parse_author_name, names) if a is not None)


# ---------------------------------------------------------------------------
# One record = everything the matcher needs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanedRecord:
    title: str                          # normalize_title(...)
    title_tokens: tuple[str, ...]       # title_tokens(...)
    authors: tuple[Author, ...]
    year: int | None
    doi: str | None
    arxiv_id: str | None


def clean_record(
    *,
    title: str | None = None,
    authors: str | Sequence[str] | None = None,
    year: object = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    extra_text: Iterable[str | None] = (),
) -> CleanedRecord:
    """Clean one BibTeX entry or one crawled reference.

    extra_text: other fields that may carry an identifier (url, journal, note, eprint). The arXiv ID is
    searched in `arxiv_id`, then in extra_text, then in the title; the DOI in `doi`, then in extra_text.
    """
    extras = [e for e in extra_text if e]
    return CleanedRecord(
        title=normalize_title(title),
        title_tokens=title_tokens(title),
        authors=parse_authors(authors),
        year=extract_year(year),
        doi=extract_doi(doi) or next((d for d in map(extract_doi, extras) if d), None),
        arxiv_id=find_arxiv_id(arxiv_id, *extras, title),
    )
