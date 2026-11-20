# Milestone 3 matching specification

## 1. Scope and reproducibility

Milestone 3 matches each normalized BibTeX entry in a citing arXiv publication
to at most one arXiv reference from the **same** citing publication. The group
uses one frozen M1/M2 input version, one feature schema, one split policy, one
model-selection policy, and one output schema.

Generated data stays outside Git under a runtime data root. Set that root with
`M3_DATA_ROOT` or the pipeline's `--data-root` option. Local and Colab paths
therefore differ only in configuration, never in code or matching keys.

## 2. Range ownership

`input/member_ranges.csv` records the lecturer-assigned ranges. It is still
required even though the lecturer supplied the ranges: it is the machine-
readable source of truth used to validate ownership, build per-member datasets,
and make the exact assignment reproducible.

The frozen paper-ID lists in `input/member-paper-ids/`, rather than numeric
start/end comparisons, must be used when filtering a cross-month range.

## 3. Portable frozen-input manifest

`input/input_manifest.json` uses paths relative to the runtime data root.
Before the first full M3 run, copy it to:

```text
<data-root>/m3/frozen-inputs/input_manifest.json
```

Then replace `pending` fields with the M2 commit, freeze timestamp, row counts,
file sizes, SHA-256 checksums, and validation result. The confirmed M2 layout
is `m2/range-XX/{bibtex,elements,hierarchy}.jsonl`.

## 3.1 Loader interface

The loader is deliberately streaming because the complete M2 outputs contain
millions of rows. It accepts a portable `data_root` and a `range_id`:

```python
iter_bibtex_records(data_root, range_id) -> Iterator[BibtexRecord]
iter_matchable_reference_records(data_root, range_id) -> Iterator[ReferenceRecord]
iter_reference_documents(data_root, range_id) -> Iterator[(arxiv_id, raw_references)]
```

`iter_bibtex_records` reads `m2/<range_id>/bibtex.jsonl` one row at a time.
`iter_matchable_reference_records` reads M1
`corpus/<range_id>/<paper>/references.json` and yields only references with a
non-null `ref_arxiv_id`; raw references without an arXiv target remain in M1
for the final Hugging Face `references` table.

This is the handoff contract for Quốc Học and Bảo Nhi: they consume normalized
`BibtexRecord` and `ReferenceRecord`, never machine-specific paths or raw JSON.

## 4. Canonical values and missing-value policy

All `arxiv_id` and `ref_arxiv_id` values are strings in canonical base-ID form:

```text
arXiv:2609.7834v2 -> 2609.07834
1706.03762v7      -> 1706.03762
hep-th/9901001v2  -> hep-th/9901001
```

Version suffixes, `arXiv:` prefixes, URL prefixes, whitespace, filesystem slugs
such as `2609-07834`, and numeric casts are forbidden in table keys. Filesystem
slugs are only an M1 storage detail; `metadata.json.arxiv_id` is authoritative.

Missing values are represented as follows:

| Field type | Representation |
| --- | --- |
| Missing scalar metadata | Parquet null / Python `None` |
| Missing authors | Empty list |
| Missing prediction candidate | Null |
| Missing ground truth for `no_match` | Null |
| Missing IDs | Null; never `"None"`, `"null"`, or `""` |

Similarity features involving a missing source field are `0`, with a separate
missing indicator. `year_abs_diff` is `0` when either year is missing and must
be paired with `year_missing = 1`.

## 5. Shared records

The definitions in `src/matching/contracts.py` are the code source of truth.

| Record | Key | Required fields |
| --- | --- | --- |
| `BibtexRecord` | `(arxiv_id, citation_key)` | `entry_type`, title/authors/year/DOI metadata |
| `ReferenceRecord` | `(arxiv_id, ref_arxiv_id)` | cited-paper metadata |
| `LabelRecord` | `(arxiv_id, citation_key)` | ground truth, source, status, audit fields |
| `CandidatePair` | `(arxiv_id, citation_key, ref_arxiv_id)` | candidate reference ID |
| `PartitionRecord` | `arxiv_id` | member, range, partition, label source |
| `PredictionRow` | `(arxiv_id, citation_key)` | partition, nullable ground truth, `cand_1` … `cand_5` |

### Labels

`label_source` is `manual` or `automatic`.

```text
status = match    -> groundtruth_arxiv_id is non-null
status = no_match -> groundtruth_arxiv_id is null
status = uncertain -> temporary manual-review state; never train/evaluate on it
```

One BibTeX entry has at most one ground-truth reference. No-match must not be
represented as a fake candidate ID.

## 6. Candidate, split, and feature rules

- Candidates are generated only within a single citing publication.
- Only references with a non-null `ref_arxiv_id` produce candidates.
- Duplicate reference IDs are deduplicated before ranking.
- A publication with zero candidates retains its BibTeX entries and yields null
  prediction candidate columns.
- Ties break by score descending, title similarity descending, then
  `ref_arxiv_id` ascending.
- Split by publication, never by candidate pair or BibTeX row.
- Partition names are exactly `train`, `valid`, and `test`.
- Fit TF-IDF/vectorizers on training publications only; transform valid/test.
- Do not use ground truth, partition, label source, annotator, reviewer, or
  test-derived statistics as model features.

## 6.1 Input validation and manual-selection inventory

Before labeling, validate M3 input records per range:

- duplicate BibTeX key: `(arxiv_id, citation_key)`;
- duplicate matchable reference key: `(arxiv_id, ref_arxiv_id)`;
- citing publication outside its frozen ID list;
- candidate pair whose BibTeX/reference does not belong to the same citing
  publication;
- publication assigned to more than one split.

The publication inventory contains `bibtex_count`, `reference_count`,
`arxiv_reference_count`, title/author/year missing rates, and duplicate
reference count. Minh Thư should choose manual papers from it, favoring 20–35
BibTeX entries, several arXiv-reference candidates, no duplicate references,
and low missing rates.

## 7. Manual and automatic labels

Each member labels five assigned publications (3 train, 1 valid, 1 test) and
one backup publication. The labeling tool receives BibTeX records, references,
candidate pairs, and the assignment manifest; it emits one `LabelRecord` per
BibTeX entry.

Manual labels are reviewed before use. Train labels develop rules/models;
validation labels choose thresholds/models; manual test labels are held out.
Automatic labeling runs only on non-manual publications after a single shared
threshold is frozen. Automatic labels and manual labels are stored separately.

## 8. Outputs and ownership

```text
<data-root>/m3/
├── frozen-inputs/
├── manifests/
├── labels/manual/ and labels/automatic/
├── features/train/, features/valid/, features/test/
├── models/
├── predictions/raw/ and predictions/postprocessed/
├── member-datasets/<member_id>/
└── validation-reports/
```

Per-member Hugging Face datasets contain six configs: `papers`, `references`,
`elements`, `hierarchy`, `bibtex`, and `predictions`. Filter each config by the
citing `arxiv_id`; do **not** filter `references` by `ref_arxiv_id`.

## 9. Remaining group decisions before full implementation

These are intentionally not hardcoded because they require group/lecturer
confirmation:

1. The actual M2 export location and its final commit/hash/checksums.
2. The 25 manual publications and one backup per member.
3. The frozen `partition_manifest.csv` after inventory and manual selection.
4. Feature registry formulas and feature-schema version 1.0.
5. Automatic matcher thresholds, selected on manual validation only.
6. The lecturer's official treatment of no-match entries in MRR@5.
7. Exact pinned dependency versions after the shared environment smoke test.
