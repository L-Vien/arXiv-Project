# M3 cleaning and similarity API 

Modules: `src/matching/cleaning.py`, `src/matching/similarity.py`. Pure functions, standard library only.
Tests: `tests/matching/test_cleaning.py`, `tests/matching/test_similarity.py`.

## cleaning.py

| Function | Input | Output |
|---|---|---|
| `normalize_arxiv_id(value)` | whole field: `arXiv:1706.03762v2`, URL, `hep-th/9901001`, `1706-03762` | canonical ID without version (`1706.03762`) or `None` |
| `extract_arxiv_id(text)` | free text or URL; explicit forms only (`arXiv:...`, `arxiv.org/abs/...`, old-style with known archive) | ID or `None`. Never matches DOI fragments, page ranges or bare numbers |
| `find_arxiv_id(*fields)` | several fields | first ID found |
| `arxiv_id_to_key(id)` | `1706.03762` | `1706-03762` (corpus directory key, same rule as M1) |
| `fix_mojibake(text)` | `FertÃ©`, `SÃ¡nchez` (UTF-8 decoded as Latin-1, present in the real `bibtex.jsonl`) | `Ferté`, `Sánchez`; text without mojibake is unchanged |
| `latex_to_text(text)` | LaTeX markup | Unicode text: mojibake and accents repaired, Greek commands become letters (`\Lambda CDM` gives `ΛCDM`), `H$_0$` gives `H0`, braces/commands/math delimiters stripped |
| `normalize_title(text)` | title | lowercase ASCII words, single spaces |
| `title_tokens(text, drop_stopwords=True)` | title | tuple of content words (single letters and stop-words dropped, digits kept) |
| `parse_authors(raw)` | one string or a list of strings | `tuple[Author, ...]`; `others` / `et al.` dropped |
| `parse_author_name(raw)` | one name | `Author` or `None` |
| `extract_year(value)` | `2019`, `2019-04-02`, `(2014)`, RFC 2822 date | int (1500-2099) or `None`; does not read arXiv IDs |
| `extract_doi(text)` / `normalize_doi` | bare DOI, `doi:...`, resolver URL | lowercase DOI or `None` |
| `clean_record(title, authors, year, doi, arxiv_id, extra_text)` | one BibTeX entry or one crawled reference | `CleanedRecord(title, title_tokens, authors, year, doi, arxiv_id)` |

`Author(family, initials)`: `family` is folded lowercase (`van der waals`), `initials` uppercase (`JF`),
`Author.key` is the last word of the family name and is the unit used for matching.

Author forms handled: `Last, First and Last, First`, `First Last`, `Last, F. M.`, `A. Smith, B. Jones, and C. Lee`,
particles (`van der Waals`), suffixes (`Jr.`), braced corporate names (`{Barnes and Noble}`), collaborations.
Known limit: a single string with several commas and no `and` that does not follow one of these patterns is read as
one `Last, First` name.

`clean_record(extra_text=...)`: pass url, journal, note and eprint here. The arXiv ID is looked for in `arxiv_id`,
then in `extra_text`, then in the title. An `eprint` that holds a URL which is not an arXiv ID is ignored.

## similarity.py

Every function returns `math.nan` when either side is empty (unknown, not "different"). `NaN >= threshold` is False.
The feature builder chooses the missing-value policy.

| Function | Meaning |
|---|---|
| `title_exact(a, b)` | normalized titles identical (0/1) |
| `title_contains(a, b, min_chars=12)` | shorter title is a substring of the longer one (0/1) |
| `title_jaccard(a_tokens, b_tokens)` | symmetric overlap of title tokens |
| `title_coverage(query_tokens, target_tokens)` | share of the query's tokens found in the target; robust to noisy crawled titles |
| `title_char_dice(a, b, n=3)` | character n-gram Dice; tolerant to typos |
| `title_sequence_ratio(a, b)` | difflib ratio; order-sensitive |
| `author_overlap(a, b, use_initials=False)` | matched authors / size of the shorter list |
| `author_jaccard(a, b, use_initials=False)` | matched authors / size of the union |
| `first_author_match(a, b, use_initials=False)` | same first author (0/1) |
| `year_diff(a, b)` | absolute year difference |
| `same_identifier(a, b)` | equal DOI or arXiv ID when both present (0/1) |
| `compare_records(bib, ref, with_sequence_ratio=True)` | dict with all measures, keys = `FEATURE_NAMES` (stable order). `with_sequence_ratio=False` skips the difflib ratio (NaN) and is about 4 times faster (57 vs 229 microseconds per pair on 100-author records) |
| `record_completeness(rec)` | `has_title`, `has_authors`, `has_year`, `has_doi`, `has_arxiv_id` (1.0/0.0), the indicator features that go next to the NaN-valued similarities |

Why coverage next to Jaccard: crawled titles are sometimes polluted, e.g.
`"MNRAS, 487, 2005 Costanzi, M., Rozo, E., Simet, M., et al."`. A noisy title that contains the true title keeps
coverage at 1.0 while Jaccard drops.

## Frozen behaviour (do not change names or meaning after the feature schema is frozen)

- IDs are canonical dotted form without version; use `arxiv_id_to_key` only for directory names.
- Title tokens drop stop-words and single letters; `normalize_title` keeps everything.
- Matching authors by family key plus optional first initial; accents and particles are normalized.

## Cost

A paper with 336 BibTeX entries and 20 arXiv-keyed references has 6,720 candidate pairs. Use `with_sequence_ratio=False`
for the first pass and let the candidate generator keep only the top-K references per entry by `title_cov_bib_in_ref`
before computing the full feature set.
