import json

import pytest

from src.parsing.benchmark import BenchmarkRecorder, directory_size_bytes


def test_benchmark_records_time_memory_and_output_size(tmp_path):
    (tmp_path / "output.txt").write_text("result", encoding="utf-8")
    benchmark = BenchmarkRecorder()
    benchmark.start()
    benchmark.start_publication("2501.01234")
    benchmark.finish_publication("2501.01234", 0.01)
    result = benchmark.finish(
        output_dir=tmp_path,
        publication_count=1,
        version_count=2,
        element_count=3,
    )

    assert result["wall_time_seconds"] >= 0
    assert result["publication_times_seconds"]["2501.01234"] == 0.01
    assert result["average_publication_time_seconds"] == 0.01
    assert result["stage_timings_seconds"] == {}
    assert "parallel stage totals" in result["stage_timing_scope"]
    assert "export" in result["publication_timing_scope"]
    assert "peak_rss_bytes" in result["memory"]
    assert result["memory"]["rss_scope"] == (
        "parent_and_live_worker_processes_when_permitted"
    )
    assert result["memory"]["sampling_interval_seconds"] == 0.05
    assert result["output"]["size_bytes"] == directory_size_bytes(tmp_path)
    json.dumps(result)


def test_benchmark_rejects_invalid_lifecycle(tmp_path):
    with pytest.raises(ValueError):
        BenchmarkRecorder(sample_interval_seconds=0)

    benchmark = BenchmarkRecorder()
    benchmark.start()
    with pytest.raises(RuntimeError):
        benchmark.start()
    benchmark.finish(
        output_dir=tmp_path,
        publication_count=0,
        version_count=0,
        element_count=0,
    )
    with pytest.raises(RuntimeError):
        benchmark.finish(
            output_dir=tmp_path,
            publication_count=0,
            version_count=0,
            element_count=0,
        )
