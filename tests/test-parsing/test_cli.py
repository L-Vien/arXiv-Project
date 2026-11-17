import json

import pytest

from src.parsing.cli import _parse_range, build_parser, load_publications, main
from src.parsing.pipeline import VersionInput


def _write_publication(corpus_dir, folder, arxiv_id, versions=(1,)):
    paper_dir = corpus_dir / folder
    paper_dir.mkdir()
    (paper_dir / "metadata.json").write_text(
        json.dumps({"arxiv_id": arxiv_id}),
        encoding="utf-8",
    )
    for version in versions:
        version_dir = paper_dir / f"v{version}"
        version_dir.mkdir()
        (version_dir / "main.tex").write_text(
            "\\documentclass{article}\\begin{document}x\\end{document}",
            encoding="utf-8",
        )


def test_load_publications_filters_numeric_range(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_publication(corpus, "2501-00001", "2501.00001", (1, 2))
    _write_publication(corpus, "2501-00005", "2501.00005", (1,))
    _write_publication(corpus, "2502-00001", "2502.00001", (1,))

    args = build_parser().parse_args(
        [
            "--corpus-dir",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--start-month",
            "2501",
            "--start-id",
            "1",
            "--end-month",
            "2501",
            "--end-id",
            "5",
        ]
    )
    assert args.start_month == "2501"

    publications = load_publications(
        corpus,
        arxiv_range=_parse_range("2501", 1, "2501", 5),
    )
    assert [item.arxiv_id for item in publications] == [
        "2501.00001",
        "2501.00005",
    ]
    assert [item.version for item in publications[0].versions] == [1, 2]


def test_load_publications_all_includes_legacy_ids(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_publication(corpus, "hep-th_9901001", "hep-th/9901001")

    publications = load_publications(corpus)

    assert [item.arxiv_id for item in publications] == ["hep-th/9901001"]


def test_load_publications_keeps_publication_without_version_as_empty_v1(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    paper_dir = corpus / "2609-23861"
    paper_dir.mkdir()
    (paper_dir / "metadata.json").write_text(
        json.dumps({"arxiv_id": "2609.23861"}),
        encoding="utf-8",
    )

    publications = load_publications(corpus)

    assert len(publications) == 1
    assert publications[0].versions == (VersionInput(1, paper_dir),)


def test_cli_accepts_run_all_alias():
    args = build_parser().parse_args(
        [
            "--corpus-dir",
            "corpus",
            "--output-dir",
            "output",
            "--run-all",
        ]
    )
    assert args.all is True


def test_cli_accepts_worker_count():
    args = build_parser().parse_args(
        [
            "--corpus-dir",
            "corpus",
            "--output-dir",
            "output",
            "--all",
            "--workers",
            "3",
        ]
    )
    assert args.workers == 3


def test_cli_rejects_csv_and_explicit_id_modes():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ids-file", "ids.txt"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ranges-file", "ranges.csv"])


def test_cli_dry_run_does_not_create_output(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_publication(corpus, "2501-00001", "2501.00001")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--corpus-dir",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--all",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert "publications=1" in capsys.readouterr().out
    assert not output_dir.exists()


def test_cli_run_writes_progress_log_and_terminal_summary(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_publication(corpus, "2501-00001", "2501.00001")
    output_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"

    exit_code = main(
        [
            "--corpus-dir",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
            "--all",
        ]
    )

    assert exit_code == 0
    log_content = (log_dir / "m2.log").read_text(encoding="utf-8")
    assert "m2_pipeline_start" in log_content
    assert "m2_progress_snapshot" not in log_content
    assert "Pipeline Snapshot" not in log_content
    assert "m2_version_gathered" in log_content
    assert "m2_publication_complete" in log_content
    terminal = capsys.readouterr().out
    assert "M2 pipeline summary" in terminal
    assert "publications=1 passed" in terminal
    assert "versions=1/1 processed" in terminal
    assert "validation_report=" in terminal
