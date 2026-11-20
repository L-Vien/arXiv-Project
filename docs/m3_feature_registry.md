# M3 Feature Registry v1.0 — Bảo Nhi

**Status:** Initial schema for Stage 1; pending team agreement.  
**Implementation:** src/matching/features.py  
**Owner:** Bảo Nhi → handoff to Bá Đạt.

Each **FeatureRow** describes one candidate pair **(arxiv_id, citation_key, ref_arxiv_id)** from the same citing publication. It contains partition, feature_schema_version = 1.0, and a values map of **26 numeric features** in the order below. It does **not** contain ground truth or predictions.

## 1. Current features

### Title similarity (8)

| Feature | Meaning |
|---|---|
| title_exact | 1 if both cleaned titles are nonempty and identical. |
| title_jaccard | Jaccard similarity between sets of title tokens. |
| title_edit_similarity | 1 − Levenshtein distance / maximum title length. |
| title_token_set_ratio | Token-set similarity approximated with Python SequenceMatcher; range [0,1]. |
| title_word_tfidf_cosine | Cosine similarity of word TF-IDF title vectors. |
| title_char_tfidf_cosine | Cosine similarity of character TF-IDF title vectors. |
| title_length_ratio | Shorter cleaned title length / longer title length. |
| title_token_containment | Shared title token count / size of the smaller title-token set. |

### Author similarity (6)

| Feature | Meaning |
|---|---|
| first_author_match | 1 if the first cleaned authors are identical. |
| last_author_match | 1 if the last cleaned authors are identical. |
| author_overlap_count | Number of distinct cleaned author strings shared by both sides. |
| author_jaccard | Jaccard similarity between sets of cleaned authors. |
| author_bib_coverage | Fraction of distinct BibTeX authors present in the reference. |
| author_sequence_similarity | Normalized Levenshtein similarity of author strings joined in original order. |

### Bibliographic and missingness (9)

| Feature | Meaning |
|---|---|
| doi_exact | 1 if both normalized DOI values exist and match. |
| year_exact | 1 if both known publication years match. |
| year_abs_diff | Absolute year difference; 0 if a year is missing. |
| venue_similarity | Token-set similarity between cleaned BibTeX journal and reference venue. |
| bib_title_missing | 1 if cleaned BibTeX title is missing. |
| ref_title_missing | 1 if cleaned reference title is missing. |
| bib_authors_missing | 1 if cleaned BibTeX authors are missing. |
| ref_authors_missing | 1 if cleaned reference authors are missing. |
| year_missing | 1 if either publication year is missing. |

### Interaction features (3)

| Feature | Meaning |
|---|---|
| high_title_author | 1 if title_edit_similarity ≥ 0.90 and author_jaccard ≥ 0.50. |
| exact_title_year_conflict | 1 if titles match exactly but two known years differ. |
| doi_match_title_missing | 1 if DOI matches and at least one title is missing. |

## 2. Input, preprocessing, and missing values

- **Inputs:** BibtexRecord, ReferenceRecord, sorted unique CandidatePairs, and a partition manifest.
- **Cleaning:** current default applies Unicode NFKC, case folding, punctuation-to-space, and whitespace normalization. Each author is cleaned separately. The cleaning callbacks can be replaced with Quốc Học's agreed API.
- **Reference metadata:** DOI, publication year, and venue must be supplied separately via reference_metadata when available. They are not fields of the current base ReferenceRecord.
- **Missing values:** unavailable similarity values default to 0, with dedicated missing flags above. Do **not** treat ref_submission_date as publication year.
- **Token-set ratio:** currently uses SequenceMatcher; it is **not numerically identical to RapidFuzz** and must be agreed before freezing.

## 3. TF-IDF and output rules

- **Train-only fit:** fit once on the unique cleaned titles of *all training publications across member ranges*. Never fit on valid/test or refit separately per range.
- **Word TF-IDF:** unigrams + bigrams; **char_wb TF-IDF:** character n-grams 3–5. Both use L2-normalized vectors.
- **Output:** one FeatureRow per candidate with exactly these 26 columns in the listed order. Labels are joined downstream, not used to compute features.
- **Validation:** sorted/unique same-publication candidates, 26 numeric values, schema version 1.0, train-only vectorizers, and missing-value rules are covered by tests/matching/test_features.py.
- **No candidates:** keep the original BibTeX entry for downstream no-match prediction, even though it yields no feature row.

**Schema freeze:** formulas, cleaning, metadata enrichment, and column order must be confirmed with the team before official full train/valid/test feature export.
