# arXiv Project --- Problem Statements

## Logistics

The project is a **group work**. Each milestone is submitted as a group
via Moodle:

-   **Shared code repository**: One public code repository per group
    serves the whole project, with each milestone's code committed
    cumulatively so the commit history documents the group's progress.
    Milestones 1 and 2 submit its link via Moodle together with the
    report link. Milestone 3 submits it together with the poster and
    dataset.
-   **Jupyter Book for Milestones 1 and 2**: The milestone reports form
    one public **Jupyter Book**, hosted on its own separate repository's
    GitHub Pages. Milestone 1's report is the book's first section and
    Milestone 2 adds the next section. Report figures and tables must be
    generated during book compilation.
-   **Final data submission**: There is no intermediate data submission.
    The fully processed data is hosted as a **Hugging Face dataset** of
    sharded Parquet tables. The table set specified across the
    milestones includes papers and references in Milestone 1, elements,
    hierarchy and BibTeX in Milestone 2, and predictions in Milestone 3.
    Hosting follows the Lab Instructors' instructions. The dataset link
    is submitted via Moodle.
-   **Final grading**: A **poster session**. Each group submits one
    `.pptx` file containing a single-page poster in **portrait A2**
    format for review. Upon acceptance, the group prints the poster and
    presents it at the poster session.

## Milestones

-   [**Project Milestone 1 --- Data
    Crawling**](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/crawling):
    Build the raw corpus, discover the assigned arXiv papers in the
    range provided by the Lab Instructors, crawl every version of each
    publication with its full-text LaTeX sources, metadata, and
    reference metadata, and analyze the scraped corpus.
-   [**Project Milestone 2 --- Hierarchical Parsing and
    Standardization**](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/parsing):
    Structure the corpus, parse each version into a hierarchical tree,
    standardize the content, convert `\\bibitem` references into BibTeX,
    deduplicate across versions, and analyze the processed data.
-   [**Project Milestone 3 --- Reference Matching
    Pipeline**](https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/matching):
    Run the machine-learning pipeline, clean, label, engineer features,
    model, and evaluate a reference-matching system scored with Mean
    Reciprocal Rank at 5 (**MRR@5**).

Milestone-specific requirements and submission formats are detailed in
each milestone page.

## Regulations

Ensure your code is well-documented with clear comments. You may use
online documents for reference, but you must provide complete citations.
You are free to discuss your topics with classmates, but your work must
be implemented and interpreted according to your own understanding.

------------------------------------------------------------------------

# Project Milestone 1 --- Data Crawling

## Objectives

In this milestone, students use the data-scraping (crawling) knowledge
obtained from theoretical lectures to scrape an online repository. This
practice is expected to help convert that knowledge into practical
data-engineering skills. This milestone and the others will not cover
every skill needed for realistic projects, so students should continue
learning after completing the course.

The target repository is arXiv, the open-access repository of more than
two million papers. Unlike projects that focus exclusively on metadata,
this milestone collects the **full-text LaTeX sources** rather than
hard-to-process PDFs. The corpus built here is the working foundation
for Milestones 2 and 3, so the crawl must be complete and correctly
organized.

## Description

This milestone has three main areas: resolving the assigned paper
entries, downloading their full-text source files, and extracting
structured reference information.

### Entry Discovery

The first step in an arXiv crawling workflow is identifying which papers
to collect. In this project, the assigned range arrives as plain arXiv
IDs. The methods below are a toolbox rather than a separate task: the
API resolves assigned IDs into paper metadata and reveals each paper's
version list, while OAI-PMH and the Kaggle snapshot demonstrate
repository-scale discovery. Implementing a full repository-scale harvest
is **not** part of this milestone.

#### arXiv API for Targeted Discovery

The [arXiv API](https://export.arxiv.org/api/query) provides an HTTP
interface for retrieving paper information using keywords, authors, or
categories. It returns Atom XML that can be parsed programmatically. It
is useful for targeted queries and daily updates of new submissions.

**Key parameters**

-   `search_query`: Field-specific search, such as
    `ti:neural AND cat:cs.AI`
-   `id_list`: Comma-separated arXiv IDs for direct lookup
-   `start`, `max_results`: Pagination controls
-   `sortBy`, `sortOrder`: Sorting criteria, such as submission date

The `arxiv.py` library allows flexible interaction with the API:

1.  [arXiv API User
    Manual](https://info.arxiv.org/help/api/user-manual.html)
2.  [arXiv API Basics](https://info.arxiv.org/help/api/basics.html)
3.  [`arxiv.py` GitHub
    repository](https://github.com/lukasschwab/arxiv.py)

The API etiquette requests modest rates: wait at least 3 seconds between
successive calls and use one connection at a time. The client used below
enforces this delay automatically. The library also had breaking changes
from version 2.x through version 4.x in 2025--2026. Version 4.0.0 (May
2026) removed the old patterns `search.results()` and
`Result.download_pdf()` / `Result.download_source()`. Use the modern
`client.results(search)` style and consult the current README.

Example:

``` python
import arxiv

def find_papers(keyword, limit=5):
    search = arxiv.Search(
        query=f"all:{keyword}",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [paper.title for paper in arxiv.Client().results(search)]

titles = find_papers("quantum computing")
print(titles)
```

Results are structured objects containing metadata such as title,
authors, abstract, and arXiv ID. Two API facts matter for the assigned
range:

1.  An `id_list` lookup on the base ID returns the latest version's
    identifier, such as `2310.12345v4`. The suffix reveals the number of
    versions and drives the every-version requirement.
2.  An individual version is fetched separately through the versioned
    `/src/` endpoint described in the download section.

#### OAI-PMH for Metadata Harvesting

The Open Archives Initiative Protocol for Metadata Harvesting
(**OAI-PMH**) enables systematic retrieval of arXiv metadata. It is an
officially supported bulk-access protocol for academic repositories.
Since March 2025, the arXiv endpoint is `https://oaipmh.arxiv.org/oai`.
The old address `export.arxiv.org/oai2` was retired during arXiv's cloud
migration and may only redirect.

OAI-PMH uses verbs to request data:

-   `Identify`: Get information about the repository.
-   `ListRecords`: Retrieve multiple records, including metadata.
-   `ListIdentifiers`: Retrieve identifiers only.

Harvests can be scoped by date range using `from` and `until`, by
category using `set=cs`, and by metadata format; the `arXiv` prefix is
recommended. Since the March 2025 rewrite, resumption tokens no longer
encode collection size and expire daily. Design a full sweep to finish
within a day or resume gracefully from saved state. The same 3-second
pacing applies.

Example:

``` python
from sickle import Sickle

def harvest_arxiv_records(date_from, date_to):
    sickle = Sickle("https://oaipmh.arxiv.org/oai")
    records = sickle.ListRecords(
        metadataPrefix="arXiv",
        from_=date_from,
        until=date_to,
    )
    return [record.metadata for record in records]
```

This returns structured metadata dictionaries that can be stored as JSON
or in a database. Resumption tokens handle pagination automatically, but
large time-bounded sweeps should be designed to complete within the
token's validity period.

#### Kaggle Dataset for Pre-Aggregated Metadata

The [Cornell arXiv dataset on
Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
provides pre-collected metadata for well over two million papers. It can
be used without API calls and is suitable for local processing and
exploratory searches.

The dataset uses JSON Lines, with one paper's metadata per line:

``` python
import json
import pandas as pd

def load_arxiv_metadata(file_path, keyword):
    with open(file_path, "r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    df = pd.DataFrame(records)
    return df[df["title"].str.contains(keyword, case=False, na=False)]
```

After loading, extract fields such as titles, abstracts, authors, and
categories for the selected subset.

#### Summary of entry discovery methods

  Method           Interface type        Example tool
  ---------------- --------------------- --------------
  arXiv API        HTTP/XML endpoint     `arxiv.py`
  OAI-PMH          Metadata protocol     `sickle`
  Kaggle dataset   Static JSON dataset   `pandas`

### Full-Text Source Download

After identifying the papers, obtain their full source files. Retrieve
the original submission TeX sources, either as `.tar.gz` archives or in
other forms, without modifying their contents. Source formats vary;
identify which contain `.tex` or `.bib` files, as these are the only
file types retained in the working corpus. Other forms, such as PDFs or
Word documents, may not be convertible into TeX. For those versions,
keep an empty record in the corpus rather than dropping the publication.

arXiv provides source files for approximately 97% of submissions; the
rest may be PDF-only. A source archive is a gzipped tar containing
`.tex` files (often multiple files linked through `\\input` or
`\\include`), `.bib` bibliography files, style and class files,
sometimes a precompiled `.bbl`, and figure binaries.

Source archives are served by the dedicated mirror `export.arxiv.org`.
The preferred endpoint is:

``` text
https://export.arxiv.org/src/{arxiv_id}
```

It accepts versioned identifiers, such as `.../src/2310.12345v2`,
allowing each revision to be retrieved individually. Use this mirror
rather than the main `arxiv.org` site for automated downloads, because
aggressive crawling can trigger robot detection and HTTP 403 blocks. Use
a descriptive `User-Agent` header and maintain 3-second pacing.

#### arXiv API Download Using Python

The `arxiv.py` library wraps API queries, but version 4.x no longer
includes a download method. Fetch the source archive yourself, for
example with `urllib`.

``` python
import arxiv
import urllib.request
from pathlib import Path

def get_source(arxiv_id, save_dir="./sources"):
    paper = next(
        arxiv.Client().results(
            arxiv.Search(id_list=[arxiv_id])
        )
    )
    print(paper.title, paper.id)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    url = f"https://export.arxiv.org/src/{arxiv_id}"
    urllib.request.urlretrieve(url, save_path / f"{arxiv_id}.tar.gz")

get_source("2310.12345v1")
get_source("2310.12345v2")
```

#### Command-Line Download with `arxiv-downloader`

The open-source command-line utility `arxiv-downloader` can download
PDFs and sources. Install it with pip and provide a paper ID or URL.

``` bash
arxiv-downloader 2302.06544 -s -d ./downloads
```

The `-s` flag requests source files rather than PDFs, and `-d` specifies
the output directory.

#### Amazon S3 Bulk Repository Access

arXiv maintains complete source archives in an AWS S3 requester-pays
bucket at `s3://arxiv/src/`. This is intended for large-scale retrieval.
Monthly paper packages are `.tar` archives.

Before downloading, configure AWS credentials with `aws configure`, then
run:

``` bash
aws s3 cp s3://arxiv/src/arXiv_src_2401_001.tar ./bulk/ --request-payer requester
```

Archives follow the pattern `arXiv_src_YYMM_NNN.tar`. The complete
archive is several terabytes and growing, so consult
`arxiv_src_manifest.xml` for the exact chunk list. Alternatively,
`s3cmd get --recursive --requester-pays s3://arxiv/src/` retrieves the
available source archives. A free mirror is also available on Google
Cloud Storage at `gs://arxiv-dataset`.

#### Summary of source-download methods

  -----------------------------------------------------------------------
  Method                  Access interface        Example tool
  ----------------------- ----------------------- -----------------------
  arXiv API               Python client plus      `arxiv.py`, `urllib`
                          `/src/` endpoint        

  Command line            CLI utility             `arxiv-downloader`

  Amazon S3               Bulk cloud storage      `awscli` or `s3cmd`
  -----------------------------------------------------------------------

### Extracting References

Structured reference information helps describe citation relationships.
This section introduces the Semantic Scholar API as a way to retrieve
references and associated identifiers, including arXiv IDs, without
parsing LaTeX directly.

#### Semantic Scholar

[Semantic Scholar](https://www.semanticscholar.org/), developed by the
Allen Institute for AI, indexes over 214 million papers and provides a
free API for structured scholarly data. The endpoint is:

``` text
https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}
```

Use the arXiv ID as the paper identifier. Request fields such as
`references.title`, `references.authors`, `references.year`,
`references.publicationDate`, `references.venue`, and
`references.externalIds`. Request flat fields because nested forms such
as `authors.name` are rejected.

Each referenced paper includes a `paperId` (Semantic Scholar ID) and an
`externalIds` dictionary that may contain `ArXiv`, `CorpusId`, `DOI`, or
other identifiers. Check `externalIds.ArXiv` to determine whether a
crawled reference becomes a row in the `references` table.

``` python
import requests

def get_paper_references(arxiv_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {
        "fields": (
            "references.title,references.authors,references.year,"
            "references.publicationDate,references.venue,references.externalIds"
        )
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("references", [])
```

#### Extracting arXiv identifiers

Iterate through the `references` array and check each reference's
`externalIds` dictionary for the `ArXiv` key. Not every reference has an
arXiv ID: some cite journal articles, books, or other non-arXiv sources.
Coverage is higher in fields such as computer science, where arXiv is
widely used. References published before 2007 may use the old identifier
scheme, such as `hep-th/9901001`; retain these original identifiers.

#### Rate limits and usage

The API enforces limits of approximately one request per second and 100
requests per five-minute window for unauthenticated use. Exceeding these
limits can trigger HTTP 429 responses, so implement delays and error
handling. An API key can improve reliability: the free tier shares an
unauthenticated request pool, while a free API key guarantees a steady
rate of one request per second. Higher limits may be available upon
review.

#### Accuracy considerations and limitations

Semantic Scholar reports over 90% precision in reference extraction
through multi-source parsing, including PDFs and LaTeX where available.
However, recent uploads may take 1--7 days to be indexed, and version
differences may not be distinguished. Data may be missing for very new
papers, non-English content, or unconventional formats. Cross-check
against the source files when needed.

## Requirement

Students receive a range of arXiv paper IDs from the Lab Instructors and
must scrape those papers programmatically. The task is to resolve the
assigned IDs and locate **every version** of each paper, not to harvest
arXiv at repository scale.

For every assigned paper, **all available versions must be crawled**,
including the initial submission and every subsequent revision. Version
IDs append `v` and the version number to the base ID, for example
`2310.12345v1` and `2310.12345v2`. If a paper has only one version,
collect `v1`. Retrieve each version individually through the versioned
`/src/` endpoint.

The scraping process must retrieve full-text source files, metadata,
BibTeX references, and crawled reference information for each version.
Paper source archives may be large because of figures, so **remove
figure files by extraction rather than by editing**: extract only `.tex`
and `.bib` files from each archive. Figure binaries and other non-source
files must never enter the corpus. Leave LaTeX figure definitions,
including `figure` environments and `\\includegraphics` lines, unchanged
in the TeX files because Milestone 2 parses document structure rather
than image binaries.

Reference metadata must additionally be crawled for each publication
using the Semantic Scholar API. There is **no data submission** in this
milestone. The corpus remains the group's working input for Milestones 2
and 3 and later becomes part of the final Hugging Face submission. This
milestone specifies two tables; later milestones add their own tables.

### Suggested data organization

Groups may organize their working data as they choose. One suggested
convention is to keep one working folder per publication, for example
`2310-12345` for ID `2310.12345`, containing the `.tex` and `.bib` files
for every version and, for each version, paper metadata and crawled
reference metadata.

Store paper title, authors, submission and revision dates, venue when
applicable, and for every crawled reference its title, authors,
submission date, arXiv ID when available, and Semantic Scholar ID. Keep
**all** crawled references locally. At final submission, only references
with an arXiv ID become rows in the published `references` table, which
keys on arXiv identifiers and serves as the Milestone 3 matching pool.

### Tables specified in this milestone

  ---------------------------------------------------------------------
  Table                              Columns (types)
  ---------------------------------- ----------------------------------
  `papers`                           `arxiv_id` (string, non-null),
                                     `title` (string), `authors` (list
                                     of strings), `submission_date`
                                     (ISO string), `revised_dates`
                                     (list of strings), `venue`
                                     (string, nullable)

  `references`                       `arxiv_id` (string, non-null, the
                                     citing paper), `ref_arxiv_id`
                                     (string, non-null), `ref_title`
                                     (string), `ref_authors` (list of
                                     strings), `ref_submission_date`
                                     (string), `ref_s2_id` (string)
  ---------------------------------------------------------------------

There must be one `references` row per crawled reference and its citing
paper in the final dataset.

## Submission Format

### Source Code

The group maintains a single public GitHub repository for the entire
project. Each milestone's code is committed cumulatively, so the commit
history documents the group's progress. The repository contains source
files such as `.py` and `.ipynb`, but excludes data files, compiled
executables, and other non-source materials.

``` text
<group-repo>/
├── src/                   # Source code
│   ├── *.py
│   ├── *.ipynb
│   └── requirements.txt   # Reproducible environment
└── README.md              # Setup and execution instructions
```

Submit the public links to the shared code repository and the milestone
report via Moodle as a group. All submissions must use the official
language of the program. **No late submission is permitted.**

### Report Requirements (Jupyter Book)

The milestone report must be a **Jupyter Book** maintained in its own
public repository, separate from the shared code repository, and hosted
on that repository's GitHub Pages. The book grows throughout the
project: Milestone 1 is its first section and Milestone 2 adds another
section.

The report must present statistics and analysis of the scraped data.
Figures and tables must be **generated during book compilation**, not
pasted as static screenshots, to support quality and interactivity.

The report should explain the implementation process, including how
crawling works and which tools and libraries are used (for example, the
arXiv API, OAI-PMH, Semantic Scholar API, and `requests`). Include
relevant statistics such as:

-   Number of papers scraped successfully.
-   Overall crawling success rate.
-   Average paper size in bytes before and after figure removal.
-   Average number of references per paper.
-   Average success rate for scraping reference metadata.

#### Performance evaluation

Illustrate the scraper's performance through **running time** and
**memory footprint**.

Running time should include:

-   Total time to process one paper.
-   Average time to process each required paper.
-   Total time for entry discovery.

Memory footprint should include:

-   Maximum RAM used during scraping.
-   Average RAM consumption.
-   Maximum disk storage required.
-   Final output storage size.

Measure these in an auditable way. Examples include peak RAM via
`/usr/bin/time -v` or `psutil`, average RAM via periodic sampling, and
disk usage via `du`. These tools are examples, not requirements.

If the system consists of multiple programs, runtime is the end-to-end
wall time to run the whole system, while memory footprint is the total
memory consumed by the full system. Final benchmarking must use a
**CPU-only Google Colab instance**. There is no upper limit on runtime,
but maximum disk usage is constrained by the Colab instance.

### README

Keep `README.md` separate from the report. It must clearly explain
environment setup, including the Python version and packages installed
through pip or conda, and explain how to run the code with specific
configurations.

## Grading Criteria

The grade is based on five main criteria:

-   **Report (Jupyter Book):** Does the report present sufficient
    information, with figures and tables generated during book
    compilation?
-   **Source Code:** Is the code clean, readable, and runnable?
-   **Discovery & Crawling:** Are discovery and download methods
    appropriate, with complete version coverage for the assigned papers?
-   **Resulted Data:** How successfully has the scraper crawled the
    data, and how well does the corpus follow the suggested conventions?
-   **Scraper's Performance:** How efficient is the scraper compared
    with others?

------------------------------------------------------------------------

# Project Milestone 2 --- Hierarchical Parsing and Standardization

**Official page:**
https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/parsing/

## Objectives

Milestone 2 transforms the raw LaTeX sources collected in Milestone 1
into structured hierarchical data. The output must be cleaned,
normalized, and deduplicated. Milestone 3 uses this processed output for
reference matching.

## Description

### 1. Multi-file Gathering

A paper's LaTeX source may be spread across several `.tex` files through
commands such as `\input` and `\include`.

-   Identify the publication root: the `.tex` file containing
    `\documentclass`.
-   Follow `\input` and `\include` commands transitively to find files
    that contribute to the final document.
-   Treat commented-out commands as comments, not as includes.
-   Parse only files reachable from the root compilation path.
-   Ignore unused or redundant auxiliary `.tex` files in the source
    directory.

The goal is to reproduce the structure that LaTeX would compile into the
final paper.

### 2. Hierarchy Construction

Build a hierarchical tree for every version of every publication.

-   The `Document` is the root.
-   Chapters or sections usually form the second level.
-   Subsections, paragraphs, and other logical divisions form lower
    levels.
-   The parser must infer the appropriate hierarchy level from the
    document's actual structure.

#### Element granularity

Leaf nodes must be one of the following:

-   **Sentences:** prose split at sentence-ending periods.
-   **Block formulas:** mathematical blocks such as `equation`
    environments or `$$...$$`.
-   **Figures:** tables are also treated as a type of figure for this
    project.

Do not split sentences at periods that occur inside mathematics, LaTeX
commands, abbreviations such as `i.e.` and `et al.`, figure references
such as `Fig. 1`, URLs, or decimal numbers.

Other rules:

-   Do not parse the References section into the hierarchy.
-   Include acknowledgements and appendices, including unnumbered
    sections created with `\section*`.
-   Itemized lists branch into deeper levels: an `itemize` block is a
    higher-level component and each item is a child element.

### 3. Standardization and Deduplication

#### Cleanup and normalization

Clean and normalize the hierarchy while preserving semantic meaning.

-   Remove non-semantic formatting commands such as `\centering`,
    placement options such as `[htpb]`, and rules such as `\midrule`.
-   Normalize inline mathematics to a consistent form, for example
    `$...$`.
-   Normalize block mathematics to a consistent form, for example
    `\begin{equation}...\end{equation}`.

#### Reference extraction and BibTeX conversion

References defined using `\bibitem` inside `.tex` files must be
converted programmatically into standard BibTeX entries. Regular
expressions or other suitable parsing tools may be used.

#### Deduplication across versions

Deduplication is required at two levels.

**Reference-entry deduplication**

-   Deduplicate reference entries within a version and across versions
    of the same publication.
-   If different citation keys refer to the same underlying reference,
    choose one canonical key.
-   Rename `\cite{}` commands that use discarded keys to the chosen
    canonical key.
-   Merge the fields from duplicate entries instead of discarding
    information. For example, if one version provides title, author, and
    year while another adds a DOI, the unified entry should retain all
    four fields.

**Full-text content deduplication**

-   Compare hierarchical elements across versions after cleanup and
    normalization.
-   If the cleaned content matches exactly, store the content only once
    with a shared `element_id`.
-   Perform matching after cleanup so minor formatting or whitespace
    differences do not prevent duplicate detection.

### Requirement

Process every publication produced in Milestone 1 and every available
version of each publication.

For every version:

1.  Follow the LaTeX compilation path.
2.  Parse the document hierarchy.
3.  Standardize the content and mathematics.
4.  Extract and unify BibTeX references.
5.  Deduplicate reference entries and hierarchical content across
    versions.

If a version has no TeX source and was retained as an empty record in
Milestone 1, it must remain represented as empty in Milestone 2. Do not
silently skip it; coverage statistics must reflect the actual data.

There is no intermediate data submission. The processed corpus remains
the group's working input for Milestone 3.

### Suggested data organization

Keep the parsed structure of all versions of each publication together,
including:

-   Parsed elements.
-   Per-version parent relationships between elements.
-   Unified, deduplicated BibTeX entries.

Retain enough information to assemble the final tables without having to
re-parse the raw TeX sources.

### Tables specified in Milestone 2

  -----------------------------------------------------------------------
  Table                   Columns and types       Meaning
  ----------------------- ----------------------- -----------------------
  `elements`              `element_id` (string,   One row per distinct
                          non-null), `arxiv_id`   element across all
                          (string, non-null),     versions of a
                          `kind` (string,         publication. Identical
                          dictionary), `content`  cleaned content across
                          (string)                versions occupies one
                                                  row.

  `hierarchy`             `arxiv_id` (string,     One row per node-parent
                          non-null), `version`    edge in each version.
                          (int), `element_id`     
                          (string, non-null),     
                          `parent_id` (string,    
                          non-null)               

  `bibtex`                `arxiv_id` (string,     Unified BibTeX entries
                          non-null),              after deduplication and
                          `citation_key` (string, field merging.
                          non-null), `entry_type` 
                          (string, dictionary),   
                          `title` (string),       
                          `authors` (list of      
                          strings), `year` (int,  
                          nullable), `doi`        
                          (string), `journal`     
                          (string), `url`         
                          (string)                
  -----------------------------------------------------------------------

#### How `elements` and `hierarchy` represent versions

All versions of a publication share one `elements` store. When a cleaned
element appears in multiple versions, it is recorded once and uses the
same `element_id`.

The `hierarchy` table stores the node-parent edges separately for each
version. A shared element can therefore be connected to different
parents in different versions.

Example:

  -----------------------------------------------------------------------
  `element_id`            `kind`                  `content`
  ----------------------- ----------------------- -----------------------
  `2310-12345#root1`      `document`              Version 1 root

  `2310-12345#root2`      `document`              Version 2 root

  `2310-12345#h1`         `heading`               Introduction

  `2310-12345#h2`         `heading`               Background and model

  `2310-12345#s1`         `sentence`              We study the citation
                                                  matching problem
  -----------------------------------------------------------------------

Example hierarchy edges:

    `version` `element_id`      `parent_id`
  ----------- ----------------- --------------------
            1 `2310-12345#h1`   `2310-12345#root1`
            1 `2310-12345#s1`   `2310-12345#h1`
            2 `2310-12345#h2`   `2310-12345#root2`
            2 `2310-12345#s1`   `2310-12345#h2`

The sentence `s1` is shared across versions, but its parent differs
between versions.

## Submission Format

### Source code

Continue using the same public GitHub repository created for
Milestone 1. Commit Milestone 2 code cumulatively so the repository
history documents progress. Include source files such as `.py` and
`.ipynb`, `requirements.txt`, and a `README.md`; exclude datasets,
compiled executables, and other non-source materials.

``` text
<group-repo>/
├── src/
│   ├── *.py
│   ├── *.ipynb
│   └── requirements.txt
└── README.md
```

Submit the public code-repository link and report link through Moodle as
a group. Use the official language of the program. Late submissions are
not permitted.

### Jupyter Book report

Add the Milestone 2 report as a new section in the same public Jupyter
Book repository and GitHub Pages site used for Milestone 1. Do not
create a separate book.

Figures and tables must be generated during book compilation, not pasted
as static screenshots.

The report must cover:

-   **Hierarchical parsing:** how the parser detects document structure,
    builds the hierarchy, and handles multi-file inputs.
-   **Standardization:** data cleaning, mathematical normalization,
    reference extraction, and deduplication.
-   **Statistics:** parsing success rate, version coverage for every
    publication, and element/reference-entry counts before and after
    deduplication.
-   **Performance:** end-to-end wall time, average time per publication,
    maximum RAM usage, and final output storage size.

Benchmark on a CPU-only Google Colab instance. There is no runtime upper
limit, but disk usage is constrained by the Colab instance. Keep
`README.md` separate from the report and document the Python version,
dependencies, environment setup, and commands/configurations needed to
run the code.

## Grading Criteria

1.  **Hierarchical Parser:** Correct structure detection, multi-file
    handling, and identification of appropriate hierarchy levels.
2.  **Standardization and Deduplication:** Correct cleanup, math
    normalization, reference extraction, citation-key renaming, field
    unionization, and full-text deduplication across versions.
3.  **Parser Performance:** Runtime and memory footprint.
4.  **Code Quality and Reproducibility:** Clean, runnable code and clear
    setup/execution instructions.
5.  **Report (Jupyter Book):** Clarity, interactivity, and figures
    generated during book compilation.

------------------------------------------------------------------------

# Project Milestone 3 --- Reference Matching Pipeline

**Official page:**
https://hcmus-cs-ta.github.io/lab-book/intro-ds/statement/matching/

## Objectives

Milestone 3 applies machine learning to link the standardized BibTeX
entries produced in Milestone 2 with the corresponding external
reference metadata crawled in Milestone 1 through the Semantic Scholar
API.

This is an entity-resolution problem involving unstructured text
processing, standardization, feature engineering, and matching
algorithms.

## Description

The task is to match:

-   **BibTeX entries:** the unified `bibtex` table from Milestone 2.
-   **Crawled references:** the `references` table from Milestone 1.

For each BibTeX entry, identify the corresponding arXiv-keyed reference
from the same publication.

### 1. Data Cleaning

Apply any additional text preprocessing that improves matching. Possible
steps include:

-   Lowercasing text.
-   Removing stop words where appropriate.
-   Tokenizing author names and titles consistently.
-   Normalizing punctuation, whitespace, and common formatting
    variations.

Document the preprocessing choices and their rationale.

### 2. Data Labelling

Create a ground-truth dataset that links BibTeX entries to their correct
crawled references.

Each BibTeX entry must have either: - One correct reference association,
if a matching reference exists; or - No association, if none of the
crawled references is correct.

Matching pairs are restricted to the same publication. An entry must not
be linked to a reference from a different publication.

#### Manual labelling requirements

-   Manually label at least **5 publications**.
-   Each manually labelled publication must contain at least **20
    labelled BibTeX entries**.
-   The manual pool must contain at least **100 labelled entries** in
    total.

#### Automatic labelling requirements

-   Implement an automatic matching approach using classic methods such
    as regular expressions or string-similarity heuristics.
-   Automatically label every BibTeX entry in at least **10% of
    publications outside the manual pool**.
-   The automatic matcher must label every BibTeX entry in each
    publication to which it is applied.

The labels form the ground truth for training and evaluation. For each
labelled entry, complete its candidate-matrix row: one positive
association if a true reference exists, or an all-negative row if none
exists. The internal label format is up to the group, but it must be
clearly documented in the repository and presented in the poster.

### 3. Feature Engineering

First formulate matching as a supervised learning problem, such as
classification, regression, recommendation, or ranking. The chosen
formulation determines the input structure, for example entry-reference
pairs, triplets, or candidate lists.

Engineer features appropriate to that formulation. They may include:

-   String similarity scores.
-   Title and author matching scores.
-   Text embeddings.
-   Statistical features.
-   Features derived from the document hierarchies produced in Milestone
    2.

Justify each feature in the poster and repository documentation, and
support the justification with appropriate data analysis.

### 4. Data Modeling

The target is to match each standardized BibTeX entry with its
corresponding reference in the crawled metadata for the same
publication.

If a publication has `m` BibTeX entries and `n` crawled references with
arXiv IDs, its candidate space is an `m × n` binary matrix:

-   Row `i` represents BibTeX entry `i`.
-   Column `j` represents crawled reference `j`.
-   Cell `(i, j)` is positive if entry `i` corresponds to reference `j`,
    and negative otherwise.

For every labelled entry, the row must be complete:

-   Exactly one positive cell if a true reference exists in the
    candidate pool.
-   An all-negative row if no true reference exists.
-   Unlabelled entries do not contribute rows to the labelled candidate
    matrix.

#### Train/validation/test split

Split by publication, not by individual entry:

-   **Test set:** one manually labelled publication and one
    automatically matched publication.
-   **Validation set:** one manually labelled publication and one
    automatically matched publication.
-   **Training set:** all remaining publications from both the manual
    and automatic pools.

This publication-level split helps keep entries from the same paper from
leaking across partitions.

### 5. Model Evaluation

Evaluate the trained model on the test set containing the two selected
publications. For every labelled BibTeX entry, output a ranked list of
the top five candidate references. Post-processing is allowed.

The evaluation metric is **Mean Reciprocal Rank at 5 (MRR@5)**:

\[ `\mathrm{MRR@5}`{=tex} =
`\frac{1}{|Q|}`{=tex}`\sum`{=tex}\_{i=1}\^{\|Q\|}`\frac{1}{\mathrm{rank}_i}`{=tex}
\]

where:

-   `|Q|` is the number of labelled BibTeX entries in the test set.
-   `rank_i` is the rank of the correct reference among the top five
    candidates for entry `i`.
-   If the correct reference does not appear in the top five candidates,
    that entry contributes zero.
-   A correct result ranked first contributes `1`; ranked second
    contributes `1/2`, and so on.

## Requirement

Run the complete pipeline on the assigned publications:

1.  Cleaning.
2.  Labelling.
3.  Feature engineering.
4.  Model training and prediction.
5.  Evaluation.

Publish the matching results in the `predictions` table of the final
Hugging Face dataset. There is no intermediate dataset submission; the
final dataset is submitted in this milestone.

### `predictions` table

There is one row per BibTeX entry for each publication used in the
matching task.

  ------------------------------------------------------------------------
  Column                   Type                    Meaning
  ------------------------ ----------------------- -----------------------
  `arxiv_id`               string, non-null        Publication containing
                                                   the entry

  `partition`              string, dictionary      `train`, `valid`, or
                                                   `test`

  `citation_key`           string, non-null        BibTeX entry identifier

  `groundtruth_arxiv_id`   string, nullable        Correct arXiv ID of the
                                                   true reference, when
                                                   known

  `cand_1` to `cand_5`     string, nullable        Ranked top-five
                                                   candidate arXiv IDs,
                                                   where `cand_1` is best
                                                   and `cand_5` is last
  ------------------------------------------------------------------------

Only publications included in the training, validation, and test sets
have rows in `predictions`.

## Final Data Submission --- Hugging Face

The fully processed dataset must be hosted publicly on Hugging Face
according to the Lab Instructors' hosting instructions. Submit the
dataset link through Moodle.

### Storage format and schema

-   Use **sharded Parquet** files.
-   Publish six tables with one explicit, unified schema.
-   Keep columns flat and primitive.
-   One-level string lists, such as author lists and revised-date lists,
    are allowed.
-   Do not use nested objects or serialized JSON blobs.
-   Avoid custom loading scripts; the file patterns and dataset card
    should be sufficient for the Hub viewer and streaming backend.

### Six final dataset tables

  -----------------------------------------------------------------------
  Table                   Contents                Defined in
  ----------------------- ----------------------- -----------------------
  `papers`                Crawled paper metadata  Milestone 1

  `references`            Crawled reference       Milestone 1
                          metadata, one row per   
                          reference and citing    
                          paper; publish only     
                          references with arXiv   
                          IDs                     

  `elements`              Parsed elements across  Milestone 2
                          all versions,           
                          deduplicated into one   
                          store                   

  `hierarchy`             Per-version node-parent Milestone 2
                          edges between elements  

  `bibtex`                Unified BibTeX entries  Milestone 2
                          after deduplication and 
                          field merging           

  `predictions`           Reference-matching      Milestone 3
                          results                 
  -----------------------------------------------------------------------

### Suggested Hugging Face repository structure

``` text
<dataset-repo>/
├── README.md
└── data/
    ├── papers/
    ├── references/
    ├── elements/
    ├── hierarchy/
    ├── bibtex/
    └── predictions/
```

The root `README.md` acts as the dataset card and contains YAML metadata
mapping each dataset configuration to its Parquet files. Example:

``` yaml
---
configs:
  - config_name: papers
    data_files:
      - split: train
        path: data/papers/*.parquet
  - config_name: references
    data_files:
      - split: train
        path: data/references/*.parquet
  # Add one configuration per remaining table.
---
```

Use the actual schema and file layout required by the Lab Instructors.
The working data organization between milestones is up to each group;
the tables above define the final publishing format.

## Source Code Submission

Continue using the same public GitHub repository created for Milestones
1 and 2. Commit Milestone 3 code cumulatively so the repository history
documents the whole project.

The repository should contain source files, `requirements.txt`, and a
`README.md` with setup, execution, feature, and label documentation.
Exclude datasets, compiled executables, and other non-source materials.

``` text
<group-repo>/
├── src/
│   ├── *.py
│   ├── *.ipynb
│   └── requirements.txt
└── README.md
```

Submit the following through Moodle as a group:

-   Public Hugging Face dataset link.
-   Public shared code-repository link.
-   Poster file.

All submissions must use the official language of the program. Late
submission is not permitted.

## Poster and Poster Session

The final grading format is a poster session.

-   Submit one `.pptx` file containing a single-page **portrait A2**
    poster for review.
-   Present the group's strongest and most distinctive work.
-   Possible content includes engineered features and their
    justification, model design and MRR results, and notable statistics
    from the parsing and matching pipeline.
-   After the poster is accepted, print it and present it at the final
    poster session.

## Grading Criteria

1.  **ML Pipeline Implementation:** Correct implementation of cleaning,
    labelling, feature engineering, and modelling.
2.  **Evaluation and Analysis:** Correct use of MRR, well-justified
    features, and insightful analysis presented in the poster and
    repository documentation. Methods using external knowledge, such as
    LLMs or pretrained models, may be judged separately from other
    approaches.
3.  **Dataset Schema:** The published Parquet dataset follows the
    required table and column specifications and is ready for streaming.
4.  **Code Quality and Reproducibility:** Clean, runnable code with
    clear instructions.
5.  **Poster and Session:** Quality of the selected work, single-page
    portrait A2 poster, and final presentation.

------------------------------------------------------------------------

# Overall Project Checklist

## Milestone 1 --- Data Crawling

-   [ ] Resolve all assigned arXiv IDs.
-   [ ] Discover the complete version list for every publication.
-   [ ] Download every version separately from the versioned `/src/`
    endpoint.
-   [ ] Extract only `.tex` and `.bib` files; do not retain figure
    binaries.
-   [ ] Preserve figure-related LaTeX commands inside the TeX files.
-   [ ] Crawl paper metadata and Semantic Scholar reference metadata.
-   [ ] Keep all crawled references locally, including those without
    arXiv IDs.
-   [ ] Build the `papers` and `references` data structures.
-   [ ] Measure crawling success, source sizes, reference coverage,
    runtime, RAM, and disk usage.
-   [ ] Add the Milestone 1 section to the Jupyter Book and generate
    report figures/tables during compilation.

## Milestone 2 --- Parsing and Standardization

-   [ ] Process every publication and every version from Milestone 1.
-   [ ] Identify each root `.tex` file using `\documentclass`.
-   [ ] Follow active `\input` and `\include` commands transitively.
-   [ ] Build the document hierarchy with sentence, formula,
    figure/table, and list elements.
-   [ ] Exclude References, but include acknowledgements and appendices.
-   [ ] Clean LaTeX formatting and normalize mathematics.
-   [ ] Convert `\bibitem` references into BibTeX.
-   [ ] Deduplicate BibTeX entries across versions, merge fields, and
    rename citation keys.
-   [ ] Deduplicate identical cleaned hierarchy content across versions.
-   [ ] Preserve empty records for versions without TeX sources.
-   [ ] Build `elements`, `hierarchy`, and `bibtex`.
-   [ ] Measure parsing success, version coverage,
    pre/post-deduplication counts, runtime, RAM, and output size.
-   [ ] Add the Milestone 2 section to the same Jupyter Book.

## Milestone 3 --- Reference Matching and Final Submission

-   [ ] Clean and normalize BibTeX and reference text.
-   [ ] Manually label at least 5 publications, with at least 20 entries
    each and at least 100 entries total.
-   [ ] Automatically label all BibTeX entries in at least 10% of
    publications outside the manual pool.
-   [ ] Document the label format and automatic matching rules.
-   [ ] Define the supervised learning formulation and engineer
    justified features.
-   [ ] Split data by publication: one manual and one automatic
    publication each for validation and test.
-   [ ] Train and evaluate the matching model.
-   [ ] Produce top-five ranked candidates and calculate MRR@5.
-   [ ] Build the `predictions` table.
-   [ ] Publish all six tables as sharded Parquet files on Hugging Face.
-   [ ] Submit the dataset and code-repository links and the portrait A2
    poster through Moodle.
-   [ ] Print and present the poster after acceptance.

## Project-wide repository and report requirements

-   [ ] Maintain one cumulative public code repository for all three
    milestones.
-   [ ] Maintain a separate public Jupyter Book repository for
    Milestones 1 and 2.
-   [ ] Keep datasets and compiled artifacts out of the source-code
    repository.
-   [ ] Provide a reproducible environment and clear setup/run
    instructions in `README.md`.
-   [ ] Cite external references completely.
-   [ ] Ensure figures and tables in the Jupyter Book are generated
    during compilation.
-   [ ] Follow the Lab Instructors' instructions for dataset hosting and
    submission.
