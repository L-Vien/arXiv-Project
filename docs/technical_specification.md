# Technical Specification — Milestone 1: Crawling

## 1. Scope

This milestone builds a working corpus of assigned arXiv papers. For each discovered paper, the system retrieves every available version, downloads its source, stores paper metadata, and collects reference metadata from Semantic Scholar.

The current specification covers Milestone 1 only. Later milestones will define their own technical specifications when their implementations are added.

```text
Assigned arXiv IDs
        ↓
Paper and version discovery
        ↓
Source download and processing
        ↓
Paper metadata and reference metadata
```

## 2. System boundaries

### Included

- Targeted discovery of assigned arXiv IDs.
- Retrieval of all available versions of each discovered paper.
- Download and integrity verification of source files.
- Extraction of usable text sources and paper metadata.
- Reference metadata retrieval through Semantic Scholar.
- Checkpointing, retry handling, logging, testing, and performance measurement.

### Excluded from the source repository

- Crawled source archives and corpus data.
- Generated run outputs, logs, caches, and notebook checkpoints.
- API keys and other secrets.

These artifacts are stored outside the source repository or in local environment configuration.

## 3. Repository organization

The repository currently organizes Milestone 1 code under `src/crawling/` and its tests under `tests/crawling/`.

```text
arXiv-Project/
├── src/
│   ├── __init__.py
│   └── crawling/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── discovery.py
│       ├── downloader.py
│       ├── processor.py
│       ├── metadata.py
│       ├── reference_crawler.py
│       ├── stats.py
│       ├── config.py
│       └── logger.py
├── tests/
│   └── crawling/
├── docs/
├── report/
├── requirements.txt
└── README.md
```

The M1 entry point is:

```bash
python -m src.crawling.pipeline
```

## 4. Pipeline architecture

The pipeline has two phases.

### 4.1. Discovery barrier

The first phase normalizes and deduplicates input IDs, queries the arXiv API, resolves paper metadata, identifies all available versions, and persists discovery results.

For ordinary ID batches, each requested ID receives a terminal result such as `found`, `not_found`, or a failure status. Cross-month range discovery may use a date stream and persists the papers and failures returned by that stream without materializing every absent candidate ID.

### 4.2. Downstream processing

After discovery completes, the pipeline downloads and processes each version, saves paper metadata, and crawls references. Independent papers or versions should continue when another item fails, provided the failure is recorded.

The pipeline is resumable. Progress is persisted in JSON/JSONL journals and run-state files so completed work can be reused after interruption.

## 5. Module responsibilities

| Module | Responsibility |
|---|---|
| `src/crawling/discovery.py` | Resolve input IDs, paper metadata, and available versions. |
| `src/crawling/downloader.py` | Download versioned source/PDF files, apply rate limits, retry transient failures, and verify integrity. |
| `src/crawling/processor.py` | Safely extract source archives and retain `.tex`, `.bib`, and `.bbl` files. |
| `src/crawling/metadata.py` | Normalize paper metadata and construct stable corpus paths. |
| `src/crawling/reference_crawler.py` | Retrieve and persist reference metadata from Semantic Scholar. |
| `src/crawling/pipeline.py` | Coordinate discovery, source processing, reference crawling, checkpoints, and resume. |
| `src/crawling/stats.py` | Collect runtime, memory, disk, throughput, and status metrics. |
| `src/crawling/config.py` | Resolve paths, API settings, retry defaults, and request pacing. |
| `src/crawling/logger.py` | Configure application logging to the console and log files. |

## 6. Data contracts

### 6.1. Paper metadata

One normalized record is stored for each base arXiv paper:

| Field | Type | Description |
|---|---|---|
| `arxiv_id` | string | Base arXiv ID without a version suffix. |
| `title` | string | Paper title. |
| `authors` | list of strings | Author names. |
| `submission_date` | ISO date string | Initial submission date. |
| `revised_dates` | list of ISO date strings | Dates of later versions. |
| `venue` | nullable string | Publication venue when available. |

### 6.2. Reference metadata

The working corpus retains all references returned by Semantic Scholar. The normalized reference table contains records for references with an arXiv ID:

| Field | Type | Description |
|---|---|---|
| `arxiv_id` | string | ID of the citing paper. |
| `ref_arxiv_id` | string | ID of the cited paper. |
| `ref_title` | string | Cited paper title. |
| `ref_authors` | list of strings | Cited paper authors. |
| `ref_submission_date` | string | Publication or submission date when available. |
| `ref_s2_id` | string | Semantic Scholar paper ID. |

References without an arXiv ID remain in the local working corpus even when they are not included in this table.

### 6.3. Version processing status

Each version keeps a status record containing the source type, archive size, raw size, retained size, retained files, and any processing error. The relevant statuses include:

- `ok`: at least one TeX file was retained;
- `no_tex`: the source was valid but contained no TeX file;
- `missing`: the expected archive was unavailable;
- `extract_error`: the archive could not be processed.

## 7. Working corpus layout

The corpus is stored outside the repository. The logical layout is:

```text
corpus/
└── 2310-12345/
    ├── metadata.json
    ├── references.json
    └── v1/
        ├── source_status.json
        ├── main.tex
        ├── references.bib
        └── references.bbl
```

A version may contain multiple `.tex`, `.bib`, and `.bbl` files. Their relative paths are preserved. A version without usable TeX still has a status record and is not silently discarded.

## 8. Inputs, outputs, and configuration

The CLI accepts exactly one input source:

- one ID range using `--start-month`, `--start-id`, `--end-month`, and `--end-id`;
- a CSV file containing `start_month`, `start_id`, `end_month`, and `end_id`;
- a text file containing one arXiv ID per line.

Typical run artifacts are stored in the configured output directory:

```text
outputs/<run>/
├── input.json
├── discovery.jsonl
├── manifest.jsonl
├── run_state.json
└── summary.json
```

Important environment variables are:

| Variable | Purpose | Default |
|---|---|---|
| `ARXIV_DATA_DIR` | Working corpus location | `../data/corpus` |
| `ARXIV_OUTPUT_DIR` | Default run output location | `outputs/run-001` |
| `ARXIV_LOG_DIR` | Log location | `logs` |
| `ARXIV_USER_AGENT` | HTTP client identification | `ArxivProject/1.0` |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar authentication | unset |

`S2_API_KEY` and `SEMANTIC_SCHOLAR_API_KEYS` are also supported for Semantic Scholar authentication.

## 9. Reliability and external services

- Use the arXiv API for targeted discovery.
- Download versioned sources through `export.arxiv.org/src/{versioned_id}`.
- Use a descriptive User-Agent and respect API pacing.
- Apply bounded retries with exponential backoff to transient failures.
- Respect `Retry-After` and pause after repeated rate-limit responses.
- Verify downloaded files using their integrity markers and SHA-256 checksums.
- Use the PDF endpoint as a fallback when source retrieval returns a missing-source response.
- Preserve failure records and continue independent work where possible.
- Write checkpoints atomically so interrupted runs can be resumed.
- Never commit API keys or other secrets.

## 10. Testing and reproducibility

Tests are grouped under `tests/crawling/` and cover configuration, discovery, downloading, processing, metadata, references, statistics, and pipeline integration.

Run the complete suite with:

```bash
pytest
```

An end-to-end run records its input fingerprint, configuration, status, failures, timings, and resource measurements. Reproduction requires the same input, configuration, code revision, and external API assumptions.

## 11. Performance measurement

The M1 implementation measures or reports:

- discovery, source, and reference processing time;
- end-to-end wall time;
- average and peak memory usage;
- disk growth and final output size;
- paper and version counts;
- source size before and after filtering;
- reference retrieval success and failure counts;
- retry and rate-limit metrics.

Detailed benchmark results and interpretation belong in the milestone report. This document only defines the measurements required by the implementation.

## 12. Integration rules

- Keep source code and working corpus separate.
- Use stable base IDs and version IDs throughout the pipeline.
- Do not silently discard papers, versions, references, or failure records.
- Keep paths configurable; do not depend on developer-specific absolute paths.
- Update tests and documentation together with interface changes.
- Keep API keys and local environment files out of version control.
