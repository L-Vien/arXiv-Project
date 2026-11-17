from io import StringIO

from src.parsing.progress import M2ProgressMonitor


def test_m2_progress_snapshot_contains_stage_input_and_counts():
    output = StringIO()
    monitor = M2ProgressMonitor(
        interval=0.01,
        output=output,
    )
    monitor.start(
        publication_count=2,
        version_count=3,
        input_description="corpus=/tmp/corpus selection=range",
    )
    monitor.set_publication("2501.00001")
    monitor.set_stage(
        "gather",
        arxiv_id="2501.00001",
        version=1,
        message="input=/tmp/corpus/paper/v1",
    )
    monitor.complete("gather", message="status=ok files=2 diagnostics=0")
    monitor.set_stage(
        "normalization",
        arxiv_id="2501.00001",
        version=1,
        message="input=RawTree + citation_map",
    )
    monitor.complete("normalization", message="status=ok nodes=4 diagnostics=0")
    monitor.finish_publication(status="success")
    monitor.stop()

    rendered = output.getvalue()
    assert "Pipeline Snapshot" in rendered
    assert "gather" in rendered
    assert "normalization" in rendered
    assert "2501.00001:v1" in rendered
    assert "1/3 versions" in rendered
    assert "input=/tmp/corpus/paper/v1" in rendered


def test_m2_progress_counts_failed_and_skipped_items():
    monitor = M2ProgressMonitor(interval=0.01, enabled=False)
    monitor.start(publication_count=1, version_count=2)
    monitor.set_stage(
        "hierarchy",
        arxiv_id="2501.00001",
        version=2,
        message="input=-",
    )
    monitor.complete("hierarchy", failed=True, skipped=True, message="empty")

    snapshot = monitor.snapshot()
    hierarchy = snapshot["source"]["worker_progress"]["hierarchy"]
    assert hierarchy["current"] == 1
    assert "failed=1" in hierarchy["message"]
    assert "skipped=1" in hierarchy["message"]
    monitor.stop(render_final=False)
