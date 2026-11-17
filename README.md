# arXiv Project

An end-to-end data science project for building a searchable arXiv corpus and matching references across paper versions.

The project is developed cumulatively across three milestones:

1. **Milestone 1 — Crawling:** discover assigned papers and all versions, download their source files, and collect paper and reference metadata.
2. **Milestone 2 — Parsing:** follow the LaTeX compilation path, build a document hierarchy, standardize content, deduplicate elements, and normalize BibTeX entries.
3. **Milestone 3 — Matching:** clean and label data, engineer features, match BibTeX entries to reference metadata, evaluate the model with MRR@5, and prepare the final dataset.

```text
Assigned IDs
    ↓
M1: raw corpus and metadata
    ↓
M2: parsed hierarchy and standardized BibTeX
    ↓
M3: reference matching and predictions
    ↓
Final Hugging Face dataset (Parquet)
```

The repository currently contains the Milestone 1 crawling implementation. Milestones 2 and 3 will be added to this same repository as the project progresses.

## Repository structure

```text
arXiv-Project/
├── src/
│   ├── __init__.py
│   ├── crawling/            # Milestone 1 implementation
│   │   ├── pipeline.py      # End-to-end pipeline orchestration
│   │   ├── discovery.py     # Paper and version discovery
│   │   ├── downloader.py    # Source/PDF download and integrity checks
│   │   ├── processor.py     # Safe source extraction and filtering
│   │   ├── metadata.py      # Metadata normalization and corpus paths
│   │   ├── reference_crawler.py
│   │   ├── stats.py
│   │   ├── config.py
│   │   └── logger.py
│   └── parsing/             # Milestone 2 implementation
│       ├── contracts.py
│       ├── compilation_path.py
│       ├── hierarchy.py
│       ├── normalization.py
│       ├── deduplication.py
│       ├── references.py
│       ├── citation_rewrite.py
│       ├── validation.py
│       ├── statistics.py
│       ├── benchmark.py
│       ├── export.py
│       ├── progress.py
│       ├── pipeline.py
│       └── cli.py
├── tests/                   # Unit and integration tests
│   ├── crawling/            # M1 tests
│   └── parsing/             # M2 tests
├── report/                  # Report sources and milestone material
├── docs/                    # Technical and milestone documentation
├── requirements.txt
└── README.md
```

Milestone 2 parsing code is under `src/parsing/`, with its tests under
`tests/parsing/`. Generated corpus data and run outputs remain outside the
source tree and are not committed to the repository.

Working data, downloaded archives, generated outputs, logs, notebook checkpoints, and final datasets must remain outside the source repository. Small input definitions may be stored in `input/`; assigned ranges can also remain local when they are not intended for publication.

## Environment setup

### Prerequisites

- Git
- Python 3.10 or newer (including `pip`)
- An internet connection for arXiv. A Semantic Scholar API key is needed only
  when collecting reference metadata.

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd arXiv-Project
```

Create and activate a virtual environment, then install the pinned project
dependencies.

**macOS / Linux (bash or zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate the
environment again. Verify the installation from the repository root:

```bash
python --version
python -m pytest
```

## Running Milestone 1

Run all commands below from the repository root with the virtual environment
activated. The CLI accepts exactly one input source: a single ID range, a CSV
file of ranges, or a text file containing one arXiv ID per line. Keep output
and corpus data outside the repository; the examples use the sibling directory
`../arxiv-data/`.

### Quick start (no API key required)

This command discovers one paper, downloads its source, and skips reference
crawling. It is the smallest useful end-to-end check of the installation.

```bash
python -m src.crawling.pipeline \
  --start-month 2401 --start-id 1 \
  --end-month 2401 --end-id 1 \
  --limit 1 \
  --without-references \
  --output-dir ../arxiv-data/runs/smoke-test \
  --corpus-dir ../arxiv-data/corpus
```

The month fields accept either `YYMM` (for example, `2401`) or `YYYY-MM`
(for example, `2024-01`). Use `--all` instead of `--limit` only when you
intend to process every paper in the specified range.

### Run from a single range

```bash
python -m src.crawling.pipeline \
  --start-month 2026-09 --start-id 22834 \
  --end-month 2026-09 --end-id 27833 \
  --limit 100 \
  --without-references \
  --output-dir ../arxiv-data/runs/run-001 \
  --corpus-dir ../arxiv-data/corpus
```

### Run from a CSV file of ranges

Create a CSV file with this exact header. Each non-empty row defines one
inclusive range; the header is required.

```csv
start_month,start_id,end_month,end_id
2024-02,1,2024-02,5
```

Then run it:

```bash
python -m src.crawling.pipeline \
  --ranges-file path/to/ranges.csv \
  --without-references \
  --output-dir ../arxiv-data/runs/run-001 \
  --corpus-dir ../arxiv-data/corpus
```

### Run from a text file of arXiv IDs

Put one ID on each line; blank lines and lines starting with `#` are ignored.

```text
# selected papers
2401.00001
```

```bash
python -m src.crawling.pipeline \
  --ids-file path/to/ids.txt \
  --without-references \
  --output-dir ../arxiv-data/runs/run-001 \
  --corpus-dir ../arxiv-data/corpus
```

### Enable reference crawling

Reference crawling requires a Semantic Scholar API key. Export it in the same
terminal before omitting `--without-references`:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:SEMANTIC_SCHOLAR_API_KEY = "your-api-key"
```

`S2_API_KEY` and `SEMANTIC_SCHOLAR_API_KEYS` are also supported. Never commit
API keys to the repository.

### Resume an interrupted run

Reuse the same input source, output directory, and corpus directory, adding
`--resume`:

```bash
python -m src.crawling.pipeline \
  --ranges-file path/to/ranges.csv \
  --without-references \
  --output-dir ../arxiv-data/runs/run-001 \
  --corpus-dir ../arxiv-data/corpus \
  --resume
```

For all available options, run `python -m src.crawling.pipeline --help`.

### Check the result

On completion, the terminal prints the path to
`../arxiv-data/runs/run-001/summary.json`. Inspect that file and confirm its
`status` starts with `completed`. The downloaded corpus is stored under the directory
given to `--corpus-dir`.

The pipeline runs in two phases: discovery first, followed by source processing and reference crawling. It supports checkpointing, retry/backoff, rate-limit handling, bounded queues, and runtime/resource metrics.

## Data products

Milestone 1 produces the working inputs for later milestones:

```text
corpus/<paper-id>/
├── metadata.json
├── references.json
└── v<version>/
    ├── source_status.json
    ├── *.tex
    ├── *.bib
    └── *.bbl
```

Milestone 2 adds the parsed and standardized tables:

- `elements`
- `hierarchy`
- `bibtex`

### Running Milestone 2

Run the parser over an existing M1 corpus. The CLI accepts either `--all` or
one inclusive arXiv ID range:

```bash
python -m src.parsing.cli \
  --corpus-dir ../arxiv-data/corpus \
  --output-dir ../arxiv-data/m2/run-001 \
  --all
```

The run writes `elements.jsonl`, `hierarchy.jsonl`, `bibtex.jsonl`,
`version_manifest.jsonl`, `statistics.json`, `benchmark.json`, and
`validation_report.json`. Use `--dry-run` to inspect the selected publications
without parsing them. Run the test suite with `python -m pytest`.

Milestone 3 adds `predictions`. Together with the M1 tables `papers` and `references`, these form the six final dataset tables published as sharded Parquet files.

## Configuration

| Variable | Default |
|---|---|
| `ARXIV_DATA_DIR` | `../data/corpus` |
| `ARXIV_OUTPUT_DIR` | `outputs/run-001` |
| `ARXIV_LOG_DIR` | `logs` |
| `ARXIV_USER_AGENT` | `ArxivProject/1.0` |
| `ARXIV_SOURCE_REQUEST_INTERVAL` | `0` seconds (wait only after HTTP 429) |

The CLI defaults to 4 concurrent source-download workers, matching the
download strategy used by the reference implementation. Override the
concurrency or opt into proactive pacing when needed:

```bash
python -m src.crawling.pipeline ... --download-workers 8 --request-interval 0
```

When arXiv responds with HTTP 429, the shared limiter pauses subsequent source
requests and the retry policy handles the backoff. A positive
`--request-interval` can be used for a more conservative run.

## Testing

```bash
pytest
```

Detailed M1 technical notes are available in [`docs/technical_specification.md`](docs/technical_specification.md). The `report/` folder contains report sources and milestone material for this project; the published Jupyter Book may be maintained separately according to the course submission format.

## References

- [Project statements](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/)
- [Milestone 1 — Crawling](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/crawling/)
- [Milestone 2 — Parsing](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/parsing/)
- [Milestone 3 — Matching](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/matching/)
