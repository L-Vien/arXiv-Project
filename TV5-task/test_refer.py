"""Real-corpus test script for TV5 reference extraction with batch processing and json outputs."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so scripts can be executed directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from src.parsing.contracts import SourceBundle, SourceFile, ProcessingStatus
from src.parsing.references import process_references
from src.parsing.validation import validate_reference_bundle

# ==============================================================================
# Cấu hình test arXiv ID:
# - TARGET_ARXIV_IDS: Danh sách các ID cụ thể (nếu có, ưu tiên dùng list này).
# - START_ARXIV_ID / END_ARXIV_ID: Khoảng ID để test nhanh danh sách dài (ví dụ: từ "2609-22834" đến "2609-22850").
# - Nếu tất cả đều trống, script sẽ quét toàn bộ các folder trong data/corpus/.
# ==============================================================================
TARGET_ARXIV_IDS: list[str] = []
START_ARXIV_ID: str = "2609-22834"
END_ARXIV_ID: str = "2609-22873"


def run_batch_test(arxiv_ids: list[str] | None = None, corpus_root: str | None = None) -> None:
    corpus_path = Path(corpus_root) if corpus_root else project_root / "data" / "corpus"
    corpus_path = corpus_path.resolve()

    if not corpus_path.exists():
        print(f"[ERROR] Corpus directory not found at: {corpus_path}")
        return

    # Use parameter if provided, otherwise resolve from TARGET_ARXIV_IDS or START/END range, or scan all.
    if arxiv_ids is None:
        if TARGET_ARXIV_IDS:
            arxiv_ids = TARGET_ARXIV_IDS
        else:
            all_dirs = sorted([d.name for d in corpus_path.iterdir() if d.is_dir()])
            if START_ARXIV_ID or END_ARXIV_ID:
                arxiv_ids = [
                    d for d in all_dirs
                    if (not START_ARXIV_ID or d >= START_ARXIV_ID) and
                       (not END_ARXIV_ID or d <= END_ARXIV_ID)
                ]
            else:
                arxiv_ids = all_dirs

    if not arxiv_ids:
        print(f"[ERROR] No arXiv IDs specified or corpus directory is empty.")
        return

    detailed_results: list[dict] = []
    paper_summaries: list[dict] = []
    citation_map_records: list[dict] = []
    total_references = 0
    validation_passed_count = 0
    validation_failed_count = 0

    print(f"[INFO] Starting batch test for {len(arxiv_ids)} papers.")

    for arxiv_id in arxiv_ids:
        paper_dir = corpus_path / arxiv_id
        if not paper_dir.exists():
            paper_summaries.append({
                "arxiv_id": arxiv_id,
                "success": False,
                "source_status": "dir_not_found",
                "source_type": "unknown",
                "error_message": f"Paper directory not found: {paper_dir}",
                "entry_count": 0,
                "validation_passed": False,
                "diagnostic_count": 1,
            })
            continue

        source_bundles = []
        version_statuses = []

        for v_dir in sorted(
            paper_dir.glob("v*"),
            key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else -1,
        ):
            if not v_dir.is_dir():
                continue
            try:
                version_num = int(v_dir.name[1:])
            except ValueError:
                continue

            # Read source_status.json if present
            v_status_file = v_dir / "source_status.json"
            v_status_data = {}
            if v_status_file.exists():
                try:
                    v_status_data = json.loads(v_status_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            v_status_str = v_status_data.get("status", "unknown")
            v_source_type = v_status_data.get("source_type", "unknown")
            v_error = v_status_data.get("error")

            files = []
            root_file = None
            for file_path in sorted(v_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in {".tex", ".bib", ".bbl"}:
                    rel_path = file_path.relative_to(v_dir).as_posix()
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    if "\\documentclass" in content and not root_file:
                        root_file = rel_path
                    files.append(SourceFile(relative_path=rel_path, content=content, order=len(files)))

            status = ProcessingStatus.OK if files else ProcessingStatus.EMPTY_NO_TEX
            source_bundles.append(
                SourceBundle(
                    arxiv_id=arxiv_id,
                    version=version_num,
                    status=status,
                    root_file=root_file,
                    files=tuple(files),
                )
            )
            version_statuses.append({
                "version": version_num,
                "status": v_status_str,
                "source_type": v_source_type,
                "error": v_error,
                "has_tex_files": bool(files),
            })

        if not source_bundles:
            paper_summaries.append({
                "arxiv_id": arxiv_id,
                "success": False,
                "source_status": "no_version_or_files",
                "source_type": "unknown",
                "error_message": "No valid version or source files found in corpus directory",
                "entry_count": 0,
                "validation_passed": False,
                "diagnostic_count": 1,
            })
            continue

        latest_v = version_statuses[-1] if version_statuses else {}
        v_status = latest_v.get("status", "ok")
        v_type = latest_v.get("source_type", "unknown")
        v_err = latest_v.get("error")

        if not latest_v.get("has_tex_files", False):
            if v_status == "no_tex" or v_type == "pdf":
                err_msg = f"PDF-only source (source_type: {v_type}, status: {v_status}, no TeX files available)"
            elif v_status == "missing":
                err_msg = "Source missing on arXiv (404 not found)"
            elif v_status == "extract_error":
                err_msg = f"Source extraction error: {v_err or 'corrupted archive'}"
            else:
                err_msg = f"No .tex/.bib/.bbl files found (status: {v_status}, source_type: {v_type})"

            paper_summaries.append({
                "arxiv_id": arxiv_id,
                "success": False,
                "source_status": v_status,
                "source_type": v_type,
                "error_message": err_msg,
                "entry_count": 0,
                "validation_passed": True,
                "diagnostic_count": 0,
            })
            continue

        # Process references
        try:
            ref_bundle = process_references(source_bundles)
            diagnostics = [
                *ref_bundle.diagnostics,
                *validate_reference_bundle(ref_bundle),
            ]
            is_passed = not diagnostics
            entry_count = len(ref_bundle.entries)

            total_references += entry_count
            if is_passed:
                validation_passed_count += 1
            else:
                validation_failed_count += 1

            # Format entries into individual lines (one JSON object per reference)
            for idx, e in enumerate(ref_bundle.entries, 1):
                detailed_results.append({
                    "arxiv_id": ref_bundle.arxiv_id,
                    "entry_count": entry_count,
                    "validation_passed": is_passed,
                    "ref_index": idx,
                    "citation_key": e.citation_key,
                    "entry_type": e.entry_type,
                    "title": e.title,
                    "authors": list(e.authors),
                    "year": e.year,
                    "doi": e.doi,
                    "journal": e.journal,
                    "url": e.url,
                })

            success_flag = is_passed and entry_count > 0
            err_msg = None if is_passed else f"Validation failed with {len(diagnostics)} diagnostics"

            paper_summaries.append({
                "arxiv_id": ref_bundle.arxiv_id,
                "success": success_flag,
                "source_status": v_status,
                "source_type": v_type,
                "error_message": err_msg,
                "entry_count": entry_count,
                "validation_passed": is_passed,
                "diagnostic_count": len(diagnostics),
            })

            # Collect citation maps per version
            for v_num, cmap in ref_bundle.citation_maps.items():
                citation_map_records.append({
                    "arxiv_id": ref_bundle.arxiv_id,
                    "version": v_num,
                    "citation_map": dict(cmap),
                })
        except Exception as ex:
            paper_summaries.append({
                "arxiv_id": arxiv_id,
                "success": False,
                "source_status": v_status,
                "source_type": v_type,
                "error_message": f"Processing exception: {str(ex)}",
                "entry_count": 0,
                "validation_passed": False,
                "diagnostic_count": 1,
            })

    # Build summary statistics
    general_summary = {
        "total_arxiv_ids_processed": len(paper_summaries),
        "total_references": total_references,
        "validation_passed_count": validation_passed_count,
        "validation_failed_count": validation_failed_count,
        "papers": paper_summaries,
    }

    # Save files in logs directory (same level as src)
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jsonl_refer_file = log_dir / "test_refer.jsonl"
    summary_file = log_dir / "summary.json"
    jsonl_citation_map_file = log_dir / "citation_maps.jsonl"

    jsonl_content = "\n".join(json.dumps(item, ensure_ascii=False) for item in detailed_results)
    jsonl_refer_file.write_text(jsonl_content + ("\n" if jsonl_content else ""), encoding="utf-8")
    summary_file.write_text(json.dumps(general_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    citation_map_content = "\n".join(json.dumps(item, ensure_ascii=False) for item in citation_map_records)
    jsonl_citation_map_file.write_text(citation_map_content + ("\n" if citation_map_content else ""), encoding="utf-8")

    # Clean up obsolete files if present
    for old_name in ["test_refer.json", "test_refer_summary.json", "test_refer.log"]:
        old_file = project_root / old_name
        if old_file.exists():
            old_file.unlink()

    print(f"\n[DONE] Saved reference details (JSONL) to: {jsonl_refer_file}")
    print(f"[DONE] Saved citation maps (JSONL) to: {jsonl_citation_map_file}")
    print(f"[DONE] Saved summary statistics to: {summary_file}")


if __name__ == "__main__":
    ids_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    run_batch_test(ids_arg)
