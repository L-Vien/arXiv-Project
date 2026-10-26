
# Technical Specification — arXiv Project 

## 1. Project Overview

The objective of Milestone 1 is to crawl the assigned arXiv papers, including all available versions, source files, paper metadata, and reference metadata.

The pipeline consists of four main modules:

1. Discovery: resolves assigned arXiv IDs and identifies all versions.
2. Downloader: downloads the source archive of each version.
3. Processor and Metadata: extracts source files and organizes paper metadata.
4. Reference Crawler: collects metadata for references using Semantic Scholar.

The pipeline integrates these modules and records execution statistics.

## 2. Project Constraints

- Process only the arXiv ID range assigned to each group member.
- Crawl every available version of each assigned paper.
- Download source archives through the export.arxiv.org endpoint.
- Extract only `.tex` and `.bib` files into the working corpus.
- Preserve LaTeX commands and environments in `.tex` files.
- Keep a record even when a version has no usable TeX source.
- Keep all crawled reference metadata locally, including references
  without an arXiv ID.
- Store the working corpus outside the source-code repository.
- Do not commit crawled data, generated corpus files, or runtime outputs.
- Respect API rate limits and handle network errors.

## 3. Proposed Repository Structure

```text
arXiv-Project/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── pipeline.py
│   ├── stats.py
│   ├── discovery.py
│   ├── downloader.py
│   ├── processor.py
│   ├── metadata.py
│   └── reference_crawler.py
├── tests/
├── docs/
│   └── technical_specification.md
├── requirements.txt
├── README.md
└── .gitignore
```

The modules `discovery.py`, `downloader.py`, `processor.py`, `metadata.py`, and `reference_crawler.py` will be implemented by their assigned members.

## 4. Data Schema

### 4.1. Papers

The final `papers` table contains:

| Field | Type | Description |
|---|---|---|
| arxiv_id | string | Base arXiv ID |
| title | string | Paper title |
| authors | list of strings | Author names |
| submission_date | ISO string | Initial submission date |
| revised_dates | list of strings | Revision dates |
| venue | nullable string | Publication venue, if available |

### 4.2. References

The final `references` table contains one row per citing-cited paper pair when the cited paper has an arXiv ID.

| Field | Type | Description |
|---|---|---|
| arxiv_id | string | ID of the citing paper |
| ref_arxiv_id | string | ID of the cited paper |
| ref_title | string | Cited paper title |
| ref_authors | list of strings | Cited paper authors |
| ref_submission_date | string | Submission date, when available |
| ref_s2_id | string | Semantic Scholar paper ID |

All crawled references must be retained in the working corpus, even when they cannot be included in the final references table.

## 5. Working Corpus Organization

The corpus must be stored outside the source-code repository.

Suggested layout:

```text
corpus/
└── 2310-12345/
    ├── paper_metadata.json
    ├── references.json
    ├── v1/
    │   ├── metadata.json
    │   ├── source_status.json
    │   ├── paper.tex
    │   └── references.bib
    └── v2/
        ├── metadata.json
        ├── source_status.json
        ├── paper.tex
        └── references.bib
```

This is an illustrative layout. The actual filenames may differ. A version may contain multiple `.tex` and `.bib` files.

The corpus must preserve the version-level record even when the source archive does not contain usable TeX or BibTeX files.

## 6. Proposed Module Interfaces

The following interfaces are initial proposals and must be agreed upon by all group members before implementation.

### 6.1. Discovery

Input:
- A list of assigned base arXiv IDs.

Output:
- Paper metadata.
- The list of available version IDs for each paper.

### 6.2. Downloader

Input:
- A versioned arXiv ID.
- A destination directory.

Output:
- Download status.
- Archive location when successful.
- Error information when unsuccessful.

### 6.3. Processor and Metadata

Input:
- A downloaded source archive.
- A destination directory.

Output:
- Extracted `.tex` and `.bib` files.
- Source availability status.
- Organized paper and version metadata.

### 6.4. Reference Crawler

Input:
- A base arXiv ID.

Output:
- All available crawled reference metadata.
- arXiv ID and Semantic Scholar ID when available.
- Error or availability information when retrieval fails.

### 6.5. Pipeline

Responsibilities:
- Coordinate the four modules.
- Process assigned papers and all discovered versions.
- Continue processing other papers when an individual paper fails.
- Record status and errors.
- Save intermediate results so completed work is not unnecessarily lost.
- Collect execution statistics.

## 7. API Usage and Error Handling

- Respect arXiv API etiquette and wait at least three seconds
  between successive requests.
- Download source archives using the export.arxiv.org endpoint.
- Use a descriptive User-Agent for source downloads.
- Respect Semantic Scholar API limits.
- Handle HTTP errors, timeouts, rate limits, malformed responses,
  and unavailable source archives.
- Record failures in logs and retain enough information to retry them.
- Avoid silently discarding papers or references.

## 8. Performance Measurement

The integrated pipeline should measure:

- Total end-to-end execution time.
- Average processing time per assigned paper.
- Entry discovery time.
- Peak and average RAM usage.
- Maximum disk storage required.
- Final corpus size.
- Number and success rate of processed papers.
- Average paper size before and after excluding figure binaries.
- Average number of references per paper.
- Reference metadata retrieval success rate.

Final benchmarking must be performed on Google Colab in CPU-only mode.

## 9. Team Integration Rules

- All members use the same agreed module interfaces.
- Each member tests their own module before integration.
- Each member processes their assigned arXiv ID range.
- Shared code must not depend on another member's local absolute paths.
- The team must test the integrated pipeline end to end.
- Source code and working corpus must remain separate.
