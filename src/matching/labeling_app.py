"""Local Streamlit interface for manual arXiv reference labeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .review import (
        candidate_count,
        load_resume_table,
        parse_candidates,
        save_review_table,
    )
except ImportError:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.matching.review import (
        candidate_count,
        load_resume_table,
        parse_candidates,
        save_review_table,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _display(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _load_state(input_path: Path, output_path: Path) -> None:
    source = str(input_path.resolve())
    destination = str(output_path.resolve())
    if (
        "labels" not in st.session_state
        or st.session_state.get("loaded_input") != source
        or st.session_state.get("loaded_output") != destination
    ):
        labels, restored_count = load_resume_table(input_path, output_path)
        st.session_state.labels = labels
        st.session_state.loaded_input = source
        st.session_state.loaded_output = destination
        st.session_state.restored_count = restored_count
        st.session_state.current_row = 0


def _save_state(output_path: Path) -> None:
    save_review_table(st.session_state.labels, output_path)
    st.session_state.last_saved = str(output_path)


def _save_widget_value(
    row_index: int,
    column: str,
    widget_key: str,
    output_path: Path,
) -> None:
    st.session_state.labels.at[row_index, column] = st.session_state[widget_key]
    _save_state(output_path)


def _set_label(row_index: int, status: str, groundtruth: str = "") -> None:
    labels = st.session_state.labels
    labels.at[row_index, "status"] = status
    labels.at[row_index, "groundtruth_arxiv_id"] = groundtruth
    labels.at[row_index, "label_source"] = "manual"
    if "reviewer" in labels.columns and not str(labels.at[row_index, "reviewer"]).strip():
        labels.at[row_index, "reviewer"] = ""


def _next_pending_row(labels: pd.DataFrame, start: int, row_indexes: list[int]) -> int:
    for position in range(start + 1, len(row_indexes)):
        row_index = row_indexes[position]
        if not str(labels.at[row_index, "status"]).strip():
            return position
    return min(start + 1, len(row_indexes) - 1)


def _inject_sticky_bibtex_style() -> None:
    """Keep the BibTeX column visible while the candidate column scrolls."""

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(.bibtex-sticky-marker) {
            align-items: flex-start;
        }

        div[data-testid="stHorizontalBlock"]:has(.bibtex-sticky-marker)
        > div[data-testid="column"]:first-child {
            position: sticky;
            top: 1rem;
            align-self: flex-start;
            max-height: calc(100vh - 2rem);
            overflow-y: auto;
            padding: 1rem 0.85rem 0.75rem;
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.65rem;
            z-index: 2;
        }

        .bibtex-sticky-marker {
            display: none;
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"]:has(.bibtex-sticky-marker)
            > div[data-testid="column"]:first-child {
                position: static;
                max-height: none;
                overflow-y: visible;
                padding: 0;
                border: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    args = _arguments()
    st.set_page_config(page_title="arXiv Labeling", layout="wide")
    _inject_sticky_bibtex_style()
    st.title("arXiv reference labeling")
    st.caption("Chọn candidate đúng cho từng BibTeX entry, hoặc đánh dấu no-match.")

    with st.sidebar:
        st.header("Review setup")
        default_input = str(args.input) if args.input else ""
        input_text = st.text_input("Review CSV path", value=default_input)
        if not input_text:
            st.info("Chạy app với --input path/to/labels_review.csv")
            return
        input_path = Path(input_text).expanduser()
        if not input_path.is_file():
            st.error(f"Không tìm thấy file: {input_path}")
            return

        output_default = str(args.output or input_path.with_name("labels_completed.csv"))
        output_text = st.text_input("Output CSV path", value=output_default)
        output_path = Path(output_text).expanduser()

    try:
        _load_state(input_path, output_path)
    except (OSError, ValueError) as exc:
        st.error(f"Không thể khôi phục file gán nhãn: {exc}")
        return
    labels = st.session_state.labels
    restored_count = st.session_state.get("restored_count", 0)
    if restored_count:
        st.success(
            f"Đã khôi phục {restored_count}/{len(labels)} entry từ {output_path}."
        )

    publication_values = ["All"] + sorted(labels["arxiv_id"].astype(str).unique().tolist())
    with st.sidebar:
        publication = st.selectbox("Publication", publication_values)
        status_filter = st.selectbox("Status", ["All", "Pending", "match", "no_match", "uncertain"])
        candidates_only = st.checkbox("Chỉ hiện entry có candidate", value=False)
        filtered = labels
        if publication != "All":
            filtered = filtered[filtered["arxiv_id"].astype(str) == publication]
        if status_filter == "Pending":
            filtered = filtered[filtered["status"].astype(str).str.strip() == ""]
        elif status_filter != "All":
            filtered = filtered[filtered["status"].astype(str).str.strip() == status_filter]
        if candidates_only:
            filtered = filtered[filtered.apply(candidate_count, axis=1) > 0]

    row_indexes = filtered.index.tolist()
    if not row_indexes:
        st.success("Không có entry nào phù hợp bộ lọc.")
        return

    current_position = min(st.session_state.get("current_row", 0), len(row_indexes) - 1)
    with st.sidebar:
        current_position = st.number_input(
            "Entry number",
            min_value=1,
            max_value=len(row_indexes),
            value=current_position + 1,
            step=1,
        ) - 1
        st.session_state.current_row = int(current_position)

    row_index = row_indexes[int(current_position)]
    row = labels.loc[row_index]
    widget_scope = (
        f"{st.session_state.loaded_input}:{st.session_state.loaded_output}:{row_index}"
    )
    candidates = parse_candidates(row.get("candidates_json", "[]"))
    completed = labels["status"].astype(str).str.strip().ne("").sum()
    st.progress(completed / len(labels) if len(labels) else 0)
    st.write(f"Entry {int(current_position) + 1}/{len(row_indexes)} trong bộ lọc · {completed}/{len(labels)} đã gán nhãn")

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            '<span class="bibtex-sticky-marker" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        st.subheader("BibTeX entry")
        st.write(f"**Publication:** `{_display(row.get('arxiv_id'))}`")
        st.write(f"**Citation key:** `{_display(row.get('citation_key'))}`")
        st.write(f"**Type:** {_display(row.get('bibtex_entry_type'))}")
        st.write(f"**Title:** {_display(row.get('bibtex_title'))}")
        st.write(f"**Authors:** {_display(row.get('bibtex_authors'))}")
        st.write(f"**Year:** {_display(row.get('bibtex_year'))}")
        st.write(f"**Current status:** `{_display(row.get('status')) or 'pending'}`")
        confidence_options = ["", "high", "medium", "low"]
        current_confidence = _display(row.get("confidence"))
        confidence_key = f"confidence:{widget_scope}"
        notes_key = f"notes:{widget_scope}"
        confidence = st.selectbox(
            "Confidence",
            confidence_options,
            index=(
                confidence_options.index(current_confidence)
                if current_confidence in confidence_options
                else 0
            ),
            key=confidence_key,
            on_change=_save_widget_value,
            args=(row_index, "confidence", confidence_key, output_path),
        )
        notes = st.text_area(
            "Notes",
            value=_display(row.get("notes")),
            height=100,
            key=notes_key,
            on_change=_save_widget_value,
            args=(row_index, "notes", notes_key, output_path),
        )

    with right:
        st.subheader(f"Candidates ({len(candidates)})")
        if any(_display(candidate.get("score")) for candidate in candidates):
            st.caption(
                "Candidates được xếp theo độ giống tiêu đề giảm dần; điểm bằng "
                "nhau thì xếp theo arXiv ID. Điểm chỉ là gợi ý, không phải nhãn "
                "tự động."
            )
        elif candidates:
            st.caption(
                "Template cũ chưa có similarity score; thứ tự hiện tại được giữ "
                "nguyên từ file candidates_json."
            )
        if not candidates:
            st.warning("Publication này không có candidate arXiv.")
        if candidates:
            # Keep candidate browsing inside the right column. This is more
            # reliable than relying on position: sticky for Streamlit columns.
            with st.container(height=700, border=True):
                for number, candidate in enumerate(candidates, start=1):
                    ref_id = _display(candidate.get("ref_arxiv_id"))
                    title = _display(candidate.get("ref_title")) or "(không có title)"
                    authors = _display(candidate.get("ref_authors"))
                    year = _display(candidate.get("ref_year"))
                    score = _display(candidate.get("score"))
                    is_hint = ref_id == _display(row.get("hint_arxiv_id"))
                    with st.container(border=True):
                        hint_text = " · ⭐ BibTeX hint" if is_hint else ""
                        st.markdown(
                            f"**{number}. `{ref_id}`**{hint_text} — {title}"
                        )
                        score_text = f" · Title similarity: {score}/100" if score else ""
                        st.caption(
                            f"Authors: {authors or '—'} · Year: {year or '—'}"
                            f"{score_text}"
                        )
                        if st.button("Chọn candidate này", key=f"match_{row_index}_{number}", use_container_width=True):
                            _set_label(row_index, "match", ref_id)
                            labels.at[row_index, "confidence"] = confidence
                            labels.at[row_index, "notes"] = notes
                            _save_state(output_path)
                            st.session_state.current_row = _next_pending_row(labels, int(current_position), row_indexes)
                            st.rerun()

        no_match_col, uncertain_col = st.columns(2)
        with no_match_col:
            if st.button("No match", use_container_width=True):
                _set_label(row_index, "no_match")
                labels.at[row_index, "confidence"] = confidence
                labels.at[row_index, "notes"] = notes
                _save_state(output_path)
                st.session_state.current_row = _next_pending_row(labels, int(current_position), row_indexes)
                st.rerun()
        with uncertain_col:
            if st.button("Uncertain", use_container_width=True):
                _set_label(row_index, "uncertain")
                labels.at[row_index, "confidence"] = confidence
                labels.at[row_index, "notes"] = notes
                _save_state(output_path)
                st.session_state.current_row = _next_pending_row(labels, int(current_position), row_indexes)
                st.rerun()

    if st.button("Lưu hiện tại", type="secondary"):
        labels.at[row_index, "confidence"] = confidence
        labels.at[row_index, "notes"] = notes
        _save_state(output_path)
        st.success(f"Đã lưu: {output_path}")

    if st.session_state.get("last_saved"):
        st.caption(f"Lần lưu gần nhất: {st.session_state.last_saved}")


if __name__ == "__main__":
    main()
