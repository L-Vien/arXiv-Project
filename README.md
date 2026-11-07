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
│   └── crawling/            # Milestone 1 implementation
│       ├── pipeline.py      # End-to-end pipeline orchestration
│       ├── discovery.py     # Paper and version discovery
│       ├── downloader.py    # Source/PDF download and integrity checks
│       ├── processor.py     # Safe source extraction and filtering
│       ├── metadata.py      # Metadata normalization and corpus paths
│       ├── reference_crawler.py
│       ├── stats.py
│       ├── config.py
│       └── logger.py
├── tests/                   # Unit and integration tests
│   └── crawling/            # M1 tests
├── report/                  # Report sources and milestone material
├── docs/                    # Technical and milestone documentation
├── requirements.txt
└── README.md
```

When Milestones 2 and 3 are implemented, add their code under `src/parsing/` and `src/matching/`, with corresponding test directories. Keep shared utilities in `src/common/` and keep milestone-specific tests close to their implementation.

Working data, downloaded archives, generated outputs, logs, notebook checkpoints, and final datasets must remain outside the source repository. Small input definitions may be stored in `input/`; assigned ranges can also remain local when they are not intended for publication.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Milestone 1 reference crawling requires a Semantic Scholar API key:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key"
```

`S2_API_KEY` and `SEMANTIC_SCHOLAR_API_KEYS` are also supported. Use `--without-references` when reference crawling is not required.

## Running Milestone 1

The current CLI accepts exactly one input source: a single ID range, a CSV file of ranges, or a text file containing one arXiv ID per line.

```bash
python -m src.crawling.pipeline \
  --ranges-file input/ranges.csv \
  --output-dir outputs/run-001 \
  --corpus-dir ../data/corpus
```

For a single range:

```bash
python -m src.crawling.pipeline \
  --start-month 2026-09 --start-id 22834 \
  --end-month 2026-09 --end-id 27833 \
  --limit 100 \
  --output-dir outputs/run-001 \
  --corpus-dir ../data/corpus
```

If a run is interrupted, resume it with the same output directory and input:

```bash
python -m src.crawling.pipeline \
  --ranges-file input/ranges.csv \
  --output-dir outputs/run-001 \
  --corpus-dir ../data/corpus \
  --resume
```

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

Milestone 3 adds `predictions`. Together with the M1 tables `papers` and `references`, these form the six final dataset tables published as sharded Parquet files.

## Configuration

| Variable | Default |
|---|---|
| `ARXIV_DATA_DIR` | `../data/corpus` |
| `ARXIV_OUTPUT_DIR` | `outputs/run-001` |
| `ARXIV_LOG_DIR` | `logs` |
| `ARXIV_USER_AGENT` | `ArxivProject/1.0` |

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
