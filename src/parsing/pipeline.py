"""Milestone 2 orchestration pipeline.

The pipeline coordinates the team modules.  It deliberately does not parse
LaTeX, normalize text, extract references, or deduplicate elements itself.
Those operations are injected as callables so every team can develop and test
independently, including with mocks before TV4 is available.
"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Protocol

from src.crawling.logger import get_logger
from src.parsing.benchmark import BenchmarkRecorder, directory_size_bytes
from src.parsing.contracts import (
    DeduplicatedPublication,
    Diagnostic,
    NormalizedVersion,
    ProcessingStatus,
    RawTree,
    ReferenceBundle,
    SourceBundle,
)
from src.parsing.export import (
    export_publication,
    finalize_run,
    write_json,
)
from src.parsing.progress import M2ProgressMonitor
from src.parsing.statistics import StatisticsAccumulator
from src.parsing.validation import (
    ValidationReport,
    effective_status,
    validate_all,
    validation_status_for_version,
)

GatherVersion = Callable[[Path, str, int], SourceBundle]
ParseStructure = Callable[[SourceBundle], RawTree]
ProcessReferences = Callable[[Sequence[SourceBundle]], ReferenceBundle]
NormalizeVersion = Callable[
    [RawTree, Mapping[str, str] | None],
    NormalizedVersion,
]
DeduplicatePublication = Callable[
    [Sequence[NormalizedVersion], str],
    DeduplicatedPublication,
]
logger = get_logger("parsing.pipeline")

_FAILED_SOURCE_STATUSES = frozenset(
    {
        ProcessingStatus.ROOT_NOT_FOUND,
        ProcessingStatus.PARSE_ERROR,
        ProcessingStatus.ERROR,
    }
)
_PARTIAL_SOURCE_STATUSES = frozenset(
    {
        ProcessingStatus.PARTIAL_PARSE,
        ProcessingStatus.MISSING_INCLUDE,
        ProcessingStatus.BIBTEX_PARTIAL,
    }
)


def _publication_outcome(
    sources: Sequence[SourceBundle],
    references: ReferenceBundle,
    validation: ValidationReport,
    *,
    raw_trees: Sequence[RawTree] = (),
    normalized_versions: Sequence[NormalizedVersion] = (),
) -> str:
    """Classify one publication consistently for logs and progress counters.

    Validation can pass with warnings, but a missing root or failed source is
    still a publication failure.  Reference diagnostics indicate a partial
    result unless they are already promoted to a validation error.
    """

    raw_by_version = {tree.version: tree for tree in raw_trees}
    normalized_by_version = {
        version.version: version for version in normalized_versions
    }
    statuses = tuple(
        effective_status(
            source.status,
            *(
                stage.status
                for stage in (
                    raw_by_version.get(source.version),
                    normalized_by_version.get(source.version),
                )
                if stage is not None
            ),
            *(
                (validation_status,)
                if (
                    validation_status := validation_status_for_version(
                        validation,
                        source.version,
                    )
                ) is not None
                else ()
            ),
        )
        for source in sources
    )
    if not validation.passed or any(status in _FAILED_SOURCE_STATUSES for status in statuses):
        return "failed"
    if any(status in _PARTIAL_SOURCE_STATUSES for status in statuses):
        return "partial"
    if any(diagnostic.severity.value == "warning" for diagnostic in references.diagnostics):
        return "partial"
    return "success"


@dataclass(frozen=True)
class PipelineDependencies:
    """Public TV1 wiring for TV2, TV3, TV4, and TV5."""

    gather_version: GatherVersion
    parse_structure: ParseStructure
    process_references: ProcessReferences
    normalize_version: NormalizeVersion
    deduplicate_publication: DeduplicatePublication


@dataclass(frozen=True)
class VersionInput:
    """Location of one publication version in the M1 corpus."""

    version: int
    source_dir: Path

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.source_dir, Path):
            object.__setattr__(self, "source_dir", Path(self.source_dir))


@dataclass(frozen=True)
class PublicationInput:
    """One publication and all versions that should be processed together."""

    arxiv_id: str
    versions: tuple[VersionInput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.arxiv_id, str) or not self.arxiv_id.strip():
            raise ValueError("arxiv_id must be a non-empty string")
        versions = tuple(self.versions)
        if len({item.version for item in versions}) != len(versions):
            raise ValueError("publication versions must be unique")
        object.__setattr__(
            self,
            "versions",
            tuple(sorted(versions, key=lambda item: item.version)),
        )


class PublicationProvider(Protocol):
    """Optional provider interface for corpus adapters."""

    def iter_publications(self) -> Iterator[PublicationInput]:
        ...


@dataclass(frozen=True)
class PublicationRunResult:
    """All intermediate and final values produced for one publication."""

    publication: PublicationInput
    sources: tuple[SourceBundle, ...]
    raw_trees: tuple[RawTree, ...]
    references: ReferenceBundle
    normalized_versions: tuple[NormalizedVersion, ...]
    deduplicated: DeduplicatedPublication
    validation: ValidationReport
    elapsed_seconds: float


@dataclass(frozen=True)
class PipelineRunResult:
    """Summary returned after the complete run has been exported.

    publications is populated only with collect_results=True. The default
    keeps memory bounded to the publication currently being processed.
    """

    publications: tuple[PublicationRunResult, ...]
    statistics: dict[str, object]
    benchmark: dict[str, object]
    publication_count: int = 0


@dataclass(frozen=True)
class PublicationWorkerSummary:
    """Compact worker result without parsed trees."""

    arxiv_id: str
    version_count: int
    element_count: int
    outcome: str
    elapsed_seconds: float
    statistics: dict[str, object]


def current_dependencies(
    *,
    normalize_version: NormalizeVersion | None = None,
    deduplicate_publication: DeduplicatePublication | None = None,
) -> PipelineDependencies:
    """Build TV1 wiring from the currently implemented team modules.

    With no overrides, this imports the real TV2, TV3, TV4, and TV5 functions.
    Optional overrides remain available for focused orchestration tests.
    """

    from src.parsing.compilation_path import gather_version
    from src.parsing.deduplication import deduplicate_publication as tv4_deduplicate
    from src.parsing.hierarchy import parse_structure
    from src.parsing.normalization import normalize_tree as tv4_normalize
    from src.parsing.references import process_references

    if (normalize_version is None) != (deduplicate_publication is None):
        raise ValueError(
            "normalize_version and deduplicate_publication must be provided together"
        )
    if normalize_version is None:
        normalize_version = tv4_normalize
        deduplicate_publication = tv4_deduplicate

    return PipelineDependencies(
        gather_version=gather_version,
        parse_structure=parse_structure,
        process_references=process_references,
        normalize_version=normalize_version,
        deduplicate_publication=deduplicate_publication,
    )


def _empty_raw_tree(source: SourceBundle) -> RawTree:
    """Create the canonical empty tree without changing TV3's implementation."""

    return RawTree(
        arxiv_id=source.arxiv_id,
        version=source.version,
        root=None,
        status=source.status,
        diagnostics=source.diagnostics,
    )


def _empty_normalized_version(tree: RawTree) -> NormalizedVersion:
    """Create the canonical empty normalized result for a missing TeX version."""

    return NormalizedVersion(
        arxiv_id=tree.arxiv_id,
        version=tree.version,
        status=tree.status,
        root_local_id=None,
        nodes=(),
        raw_element_count=0,
        diagnostics=tree.diagnostics,
    )


def _merge_diagnostics(
    validation: ValidationReport,
    sources: Sequence[SourceBundle],
    raw_trees: Sequence[RawTree],
    references: ReferenceBundle,
    normalized_versions: Sequence[NormalizedVersion],
) -> ValidationReport:
    """Expose module diagnostics in the TV1 validation report once."""

    diagnostics: list[Diagnostic] = []
    seen: set[tuple[object, str, str, str | None, int | None, str | None, int | None]] = set()

    def with_context(
        diagnostic: Diagnostic,
        *,
        arxiv_id: str | None = None,
        version: int | None = None,
    ) -> Diagnostic:
        return replace(
            diagnostic,
            arxiv_id=diagnostic.arxiv_id or arxiv_id,
            version=diagnostic.version or version,
        )

    candidates = list(validation.diagnostics)
    candidates.extend(
        with_context(diagnostic, arxiv_id=source.arxiv_id, version=source.version)
        for source in sources
        for diagnostic in source.diagnostics
    )
    candidates.extend(
        with_context(diagnostic, arxiv_id=tree.arxiv_id, version=tree.version)
        for tree in raw_trees
        for diagnostic in tree.diagnostics
    )
    candidates.extend(
        with_context(diagnostic, arxiv_id=references.arxiv_id)
        for diagnostic in references.diagnostics
    )
    candidates.extend(
        with_context(
            diagnostic,
            arxiv_id=version.arxiv_id,
            version=version.version,
        )
        for version in normalized_versions
        for diagnostic in version.diagnostics
    )
    for diagnostic in candidates:
        # TV2 records the missing include while TV3 may report the same
        # unresolved target again during expansion.  Their messages differ
        # slightly (``foo`` versus ``foo.tex``), so exact-message deduplication
        # leaves duplicate warnings in the run report.  Collapse only this
        # known equivalent case; all other diagnostics retain their full
        # provenance and message.
        message_key = diagnostic.message
        if diagnostic.code == "MISSING_INCLUDE":
            target = message_key.rsplit(": ", 1)[-1].strip()
            if target.lower().endswith(".tex"):
                target = target[:-4]
            message_key = target
        key = (
            diagnostic.severity,
            diagnostic.code,
            message_key,
            diagnostic.file_path,
            diagnostic.line,
            diagnostic.arxiv_id,
            diagnostic.version,
        )
        if key not in seen:
            seen.add(key)
            diagnostics.append(diagnostic)
    return ValidationReport(tuple(diagnostics))


def process_publication(
    publication: PublicationInput,
    dependencies: PipelineDependencies,
    progress: M2ProgressMonitor | None = None,
) -> PublicationRunResult:
    """Run TV2 -> TV3/TV5 -> TV4 for one publication.

    The only orchestration-specific behavior is preserving empty versions and
    selecting the citation map belonging to each version.  All domain logic is
    delegated to the injected team modules.
    """

    started = perf_counter()
    logger.info(
        "m2_publication_start arxiv_id=%s versions=%d",
        publication.arxiv_id,
        len(publication.versions),
    )
    gathered_sources: list[SourceBundle] = []
    for item in publication.versions:
        if progress is not None:
            progress.set_stage(
                "gather",
                arxiv_id=publication.arxiv_id,
                version=item.version,
                message=f"input={item.source_dir}",
            )
        source = dependencies.gather_version(
            item.source_dir,
            publication.arxiv_id,
            item.version,
        )
        gathered_sources.append(source)
        if progress is not None:
            progress.complete(
                "gather",
                failed=source.status
                in {
                    ProcessingStatus.ROOT_NOT_FOUND,
                    ProcessingStatus.PARSE_ERROR,
                    ProcessingStatus.ERROR,
                },
                message=(
                    f"status={source.status.value} files={len(source.files)} "
                    f"diagnostics={len(source.diagnostics)}"
                ),
            )
    sources = tuple(gathered_sources)

    raw_trees: list[RawTree] = []
    for source in sources:
        if progress is not None:
            progress.set_stage(
                "hierarchy",
                arxiv_id=source.arxiv_id,
                version=source.version,
                message=f"input={source.root_file or '-'}",
            )
        logger.info(
            "m2_version_gathered arxiv_id=%s version=%d status=%s "
            "files=%d diagnostics=%d",
            source.arxiv_id,
            source.version,
            source.status.value,
            len(source.files),
            len(source.diagnostics),
        )
        if (
            not source.files
            and source.status
            in {ProcessingStatus.EMPTY_NO_TEX, ProcessingStatus.ROOT_NOT_FOUND}
        ):
            raw_trees.append(_empty_raw_tree(source))
        else:
            raw_trees.append(dependencies.parse_structure(source))
        tree = raw_trees[-1]
        if progress is not None:
            progress.complete(
                "hierarchy",
                failed=tree.status
                in {
                    ProcessingStatus.ROOT_NOT_FOUND,
                    ProcessingStatus.PARSE_ERROR,
                    ProcessingStatus.ERROR,
                },
                skipped=tree.root is None,
                message=(
                    f"status={tree.status.value} root={tree.root is not None} "
                    f"diagnostics={len(tree.diagnostics)}"
                ),
            )
        logger.info(
            "m2_version_parsed arxiv_id=%s version=%d status=%s "
            "has_root=%s diagnostics=%d",
            tree.arxiv_id,
            tree.version,
            tree.status.value,
            tree.root is not None,
            len(tree.diagnostics),
        )

    if progress is not None:
        progress.set_stage(
            "references",
            arxiv_id=publication.arxiv_id,
            message=f"input=SourceBundle[{len(sources)}]",
        )
    references = dependencies.process_references(sources)
    if progress is not None:
        progress.complete(
            "references",
            failed=any(
                diagnostic.severity.value == "error"
                for diagnostic in references.diagnostics
            ),
            message=(
                f"entries={len(references.entries)} "
                f"diagnostics={len(references.diagnostics)}"
            ),
            reference_count=len(references.entries),
        )
    logger.info(
        "m2_references_complete arxiv_id=%s entries=%d "
        "citation_map_versions=%d diagnostics=%d",
        publication.arxiv_id,
        len(references.entries),
        len(references.citation_maps),
        len(references.diagnostics),
    )

    normalized_versions: list[NormalizedVersion] = []
    for tree in raw_trees:
        if progress is not None:
            progress.set_stage(
                "normalization",
                arxiv_id=tree.arxiv_id,
                version=tree.version,
                message="input=RawTree + citation_map",
            )
        if tree.root is None and tree.status in {
            ProcessingStatus.EMPTY_NO_TEX,
            ProcessingStatus.ROOT_NOT_FOUND,
        }:
            normalized_versions.append(_empty_normalized_version(tree))
            if progress is not None:
                progress.complete(
                    "normalization",
                    skipped=True,
                    message=f"status={tree.status.value} nodes=0",
                )
            continue
        citation_map = references.citation_maps.get(tree.version, {})
        normalized_versions.append(
            dependencies.normalize_version(tree, citation_map)
        )
        normalized = normalized_versions[-1]
        if progress is not None:
            progress.complete(
                "normalization",
                failed=normalized.status
                in {
                    ProcessingStatus.PARSE_ERROR,
                    ProcessingStatus.ERROR,
                },
                message=(
                    f"status={normalized.status.value} nodes={len(normalized.nodes)} "
                    f"diagnostics={len(normalized.diagnostics)}"
                ),
            )
        logger.info(
            "m2_version_normalized arxiv_id=%s version=%d status=%s "
            "nodes=%d diagnostics=%d",
            normalized.arxiv_id,
            normalized.version,
            normalized.status.value,
            len(normalized.nodes),
            len(normalized.diagnostics),
        )

    if progress is not None:
        progress.set_stage(
            "deduplication",
            arxiv_id=publication.arxiv_id,
            message=f"input=NormalizedVersion[{len(normalized_versions)}]",
        )
    deduplicated = dependencies.deduplicate_publication(
        normalized_versions,
        publication.arxiv_id,
    )
    if progress is not None:
        progress.complete(
            "deduplication",
            message=(
                f"elements={len(deduplicated.elements)} "
                f"hierarchy={len(deduplicated.hierarchy)}"
            ),
        )
    logger.info(
        "m2_dedup_complete arxiv_id=%s elements=%d hierarchy=%d",
        publication.arxiv_id,
        len(deduplicated.elements),
        len(deduplicated.hierarchy),
    )
    validation = validate_all(
        source_bundles=sources,
        raw_trees=raw_trees,
        reference_bundle=references,
        normalized_versions=normalized_versions,
        publication=deduplicated,
    )
    validation = _merge_diagnostics(
        validation,
        sources,
        raw_trees,
        references,
        normalized_versions,
    )
    outcome = _publication_outcome(
        sources,
        references,
        validation,
        raw_trees=raw_trees,
        normalized_versions=normalized_versions,
    )
    if progress is not None:
        progress.set_stage(
            "validation",
            arxiv_id=publication.arxiv_id,
            message="input=all module outputs",
        )
        progress.complete(
            "validation",
            failed=outcome == "failed",
            message=(
                f"outcome={outcome} passed={validation.passed} "
                f"errors={len(validation.errors)} warnings={len(validation.warnings)}"
            ),
        )
    elapsed_seconds = perf_counter() - started
    logger.info(
        "m2_publication_complete arxiv_id=%s status=%s errors=%d "
        "warnings=%d elapsed_seconds=%.3f",
        publication.arxiv_id,
        outcome,
        len(validation.errors),
        len(validation.warnings),
        elapsed_seconds,
    )

    return PublicationRunResult(
        publication=publication,
        sources=sources,
        raw_trees=tuple(raw_trees),
        references=references,
        normalized_versions=tuple(normalized_versions),
        deduplicated=deduplicated,
        validation=validation,
        elapsed_seconds=elapsed_seconds,
    )


def _silence_worker_logging() -> None:
    """Prevent workers from competing for the CLI rotating log file."""

    app_logger = logging.getLogger("arxiv_project")
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
        handler.close()
    app_logger.addHandler(logging.NullHandler())
    app_logger.setLevel(logging.WARNING)
    app_logger.propagate = False


def _process_and_export_publication(
    publication: PublicationInput,
    dependencies: PipelineDependencies,
    output_dir: Path,
    shard_dir: Path,
) -> PublicationWorkerSummary:
    """Run one publication in a worker and return aggregate metadata only."""

    _silence_worker_logging()
    result = process_publication(publication, dependencies)
    export_publication(
        publication=result.deduplicated,
        references=result.references,
        sources=result.sources,
        validation_report=result.validation,
        output_dir=output_dir,
        shard_dir=shard_dir,
        raw_trees=result.raw_trees,
        normalized_versions=result.normalized_versions,
    )

    worker_statistics = StatisticsAccumulator()
    worker_statistics.record_publication(
        sources=result.sources,
        references=result.references,
        normalized_versions=result.normalized_versions,
        raw_trees=result.raw_trees,
        publication=result.deduplicated,
        validation=result.validation,
        elapsed_seconds=result.elapsed_seconds,
        expected_version_count=len(result.publication.versions),
        diagnostics=result.validation.diagnostics,
    )
    outcome = _publication_outcome(
        result.sources,
        result.references,
        result.validation,
        raw_trees=result.raw_trees,
        normalized_versions=result.normalized_versions,
    )
    return PublicationWorkerSummary(
        arxiv_id=publication.arxiv_id,
        version_count=len(result.sources),
        element_count=len(result.deduplicated.elements),
        outcome=outcome,
        elapsed_seconds=result.elapsed_seconds,
        statistics=worker_statistics.to_dict(),
    )


def _complete_parallel_progress(
    progress: M2ProgressMonitor,
    summary: PublicationWorkerSummary,
) -> None:
    """Advance progress after a worker finishes one publication."""

    for stage in ("gather", "hierarchy", "normalization"):
        progress.set_stage(
            stage,
            arxiv_id=summary.arxiv_id,
            message="completed in worker",
        )
        for _ in range(summary.version_count):
            progress.complete(
                stage,
                failed=summary.outcome == "failed",
                message="completed in worker",
            )
    for stage in ("references", "deduplication", "validation", "export"):
        progress.set_stage(
            stage,
            arxiv_id=summary.arxiv_id,
            message="completed in worker",
        )
        progress.complete(
            stage,
            failed=summary.outcome == "failed",
            message="completed in worker",
        )


def _run_parallel_publications(
    publications: Sequence[PublicationInput],
    dependencies: PipelineDependencies,
    output_dir: Path,
    shard_dir: Path,
    workers: int,
    progress: M2ProgressMonitor,
    benchmark: BenchmarkRecorder,
    statistics: StatisticsAccumulator,
) -> tuple[int, int, int]:
    """Process a bounded number of independent publication tasks at once."""

    publication_iterator = iter(enumerate(publications, start=1))
    futures: dict[Future, tuple[int, PublicationInput]] = {}
    max_in_flight = workers * 2
    publication_count = 0
    version_count = 0
    element_count = 0

    def submit_next(executor: ProcessPoolExecutor) -> bool:
        try:
            index, publication = next(publication_iterator)
        except StopIteration:
            return False
        progress.set_publication(publication.arxiv_id)
        logger.info(
            "m2_progress publication=%d/%d arxiv_id=%s",
            index,
            len(publications),
            publication.arxiv_id,
        )
        future = executor.submit(
            _process_and_export_publication,
            publication,
            dependencies,
            output_dir,
            shard_dir,
        )
        futures[future] = (index, publication)
        return True

    with ProcessPoolExecutor(max_workers=workers) as executor:
        while len(futures) < max_in_flight and submit_next(executor):
            pass
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                _, publication = futures.pop(future)
                summary = future.result()
                statistics.merge_single_publication(summary.statistics)
                _complete_parallel_progress(progress, summary)
                benchmark.finish_publication(
                    publication.arxiv_id,
                    summary.elapsed_seconds,
                )
                progress.finish_publication(status=summary.outcome)
                publication_count += 1
                version_count += summary.version_count
                element_count += summary.element_count
                while len(futures) < max_in_flight and submit_next(executor):
                    pass

    return publication_count, version_count, element_count


def run_pipeline(
    publications: Iterable[PublicationInput],
    dependencies: PipelineDependencies,
    output_dir: Path,
    *,
    progress_enabled: bool = True,
    progress_interval_seconds: float = 1.0,
    input_description: str = "publication iterable",
    collect_results: bool = False,
    workers: int = 1,
) -> PipelineRunResult:
    """Process, validate, export, and benchmark a deterministic M2 run.

    Intermediate source text and trees are released after each publication by
    default. Callers that need complete intermediate results can opt in with
    collect_results=True.
    """

    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and collect_results:
        raise ValueError(
            "collect_results=True is not supported when workers is greater than 1"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = BenchmarkRecorder()
    statistics = StatisticsAccumulator()
    results: list[PublicationRunResult] = []
    processed_publication_count = 0
    processed_version_count = 0
    exported_element_count = 0

    ordered_publications = tuple(
        sorted(publications, key=lambda item: item.arxiv_id)
    )
    progress = M2ProgressMonitor(
        interval=progress_interval_seconds,
        enabled=progress_enabled,
    )
    progress.start(
        publication_count=len(ordered_publications),
        version_count=sum(
            len(publication.versions) for publication in ordered_publications
        ),
        input_description=input_description,
    )
    logger.info(
        "m2_pipeline_start publications=%d output_dir=%s",
        len(ordered_publications),
        output_dir,
    )
    benchmark.start()
    try:
        with TemporaryDirectory(prefix=".m2-shards-", dir=output_dir) as shard_tmp:
            shard_dir = Path(shard_tmp)
            sequential_publications = ordered_publications
            if workers > 1:
                (
                    processed_publication_count,
                    processed_version_count,
                    exported_element_count,
                ) = _run_parallel_publications(
                    ordered_publications,
                    dependencies,
                    output_dir,
                    shard_dir,
                    workers,
                    progress,
                    benchmark,
                    statistics,
                )
                sequential_publications = ()

            for index, publication in enumerate(sequential_publications, start=1):
                progress.set_publication(publication.arxiv_id)
                logger.info(
                    "m2_progress publication=%d/%d arxiv_id=%s",
                    index,
                    len(ordered_publications),
                    publication.arxiv_id,
                )
                benchmark.start_publication(publication.arxiv_id)
                result = process_publication(publication, dependencies, progress)
                progress.set_stage(
                    "export",
                    arxiv_id=publication.arxiv_id,
                    message=f"input={output_dir}",
                )
                export_publication(
                    publication=result.deduplicated,
                    references=result.references,
                    sources=result.sources,
                    validation_report=result.validation,
                    output_dir=output_dir,
                    shard_dir=shard_dir,
                    raw_trees=result.raw_trees,
                    normalized_versions=result.normalized_versions,
                )
                progress.complete(
                    "export",
                    message=(
                        f"elements={len(result.deduplicated.elements)} "
                        f"hierarchy={len(result.deduplicated.hierarchy)}"
                    ),
                )
                statistics.record_publication(
                    sources=result.sources,
                    references=result.references,
                    normalized_versions=result.normalized_versions,
                    raw_trees=result.raw_trees,
                    publication=result.deduplicated,
                    validation=result.validation,
                    elapsed_seconds=result.elapsed_seconds,
                    expected_version_count=len(result.publication.versions),
                    diagnostics=result.validation.diagnostics,
                )
                publication_status = _publication_outcome(
                    result.sources,
                    result.references,
                    result.validation,
                    raw_trees=result.raw_trees,
                    normalized_versions=result.normalized_versions,
                )
                benchmark.finish_publication(publication.arxiv_id)
                progress.finish_publication(status=publication_status)
                processed_publication_count += 1
                processed_version_count += len(result.sources)
                exported_element_count += len(result.deduplicated.elements)
                if collect_results:
                    results.append(result)
                else:
                    # Avoid retaining the final publication's source text and
                    # trees while the run-level shards are being merged.
                    del result

            finalize_run(output_dir, shard_dir=shard_dir)
        stats = statistics.to_dict()
        # Write statistics before measuring output size so the benchmark includes
        # the main run-level report in the generated storage footprint.
        write_json(output_dir / "statistics.json", stats)
        benchmark_result = benchmark.finish(
            output_dir=output_dir,
            publication_count=processed_publication_count,
            version_count=processed_version_count,
            element_count=exported_element_count,
        )
        write_json(output_dir / "benchmark.json", benchmark_result)
        # The benchmark file is itself part of the generated output.  Re-write
        # its size after measuring the first complete output tree so the
        # reported value describes the final run directory, not a pre-file
        # snapshot.  The short fixed-point loop handles a possible digit-width
        # change in the JSON representation without complicating the recorder.
        for _ in range(3):
            measured_size = directory_size_bytes(output_dir)
            if benchmark_result["output"]["size_bytes"] == measured_size:
                break
            benchmark_result["output"]["size_bytes"] = measured_size
            write_json(output_dir / "benchmark.json", benchmark_result)
        logger.info(
            "m2_pipeline_complete publications=%d output_dir=%s",
            processed_publication_count,
            output_dir,
        )
    finally:
        progress.stop()

    return PipelineRunResult(
        publications=tuple(results),
        statistics=stats,
        benchmark=benchmark_result,
        publication_count=processed_publication_count,
    )


__all__ = [
    "PipelineDependencies",
    "PipelineRunResult",
    "PublicationInput",
    "PublicationProvider",
    "PublicationRunResult",
    "VersionInput",
    "current_dependencies",
    "process_publication",
    "run_pipeline",
]
