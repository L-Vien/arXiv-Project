import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from src.crawling import discovery


def make_paper(short_id, title=None, published_date="2023-01-02", updated_date="2023-02-03"):
    paper = Mock()
    paper.get_short_id.return_value = short_id
    paper.title = title or f"Paper {short_id}"
    paper.authors = [SimpleNamespace(name="Ada Lovelace")]
    paper.published.date.return_value.isoformat.return_value = published_date
    paper.published.strftime.return_value = "20230102123456"
    paper.updated.date.return_value.isoformat.return_value = updated_date
    paper.journal_ref = "Journal"
    return paper


def make_params(**overrides):
    params = {
        "start_month": "2023-01",
        "start_id": 1,
        "end_month": "2023-01",
        "end_id": 3,
        "batch_size": 3,
        "mode": "all",
        "start_year": 2023,
        "start_month_num": 1,
        "end_year": 2023,
        "end_month_num": 1,
        "is_two_months": False,
    }
    params.update(overrides)
    return params


class TestDiscovery(unittest.TestCase):

    def test_parse_input_returns_parsed_and_derived_values(self):
        argv = [
            "discovery.py",
            "--start_month", "2023-12",
            "--start_id", "99999",
            "--end_month", "2024-01",
            "--end_id", "2",
        ]
        with patch.object(sys, "argv", argv):
            args = discovery.parse_input()

        self.assertEqual(args.start_year, 2023)
        self.assertEqual(args.start_month_num, 12)
        self.assertEqual(args.end_year, 2024)
        self.assertEqual(args.end_month_num, 1)
        self.assertEqual(args.month_diff, 1)
        self.assertEqual(args.mode, "all")

    def test_parse_input_rejects_missing_required_arguments(self):
        with (
            patch.object(sys, "argv", ["discovery.py"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            discovery.parse_input()

    def test_parse_input_rejects_invalid_month_and_id(self):
        invalid_arguments = [
            ["--start_month", "2023-13", "--start_id", "1",
             "--end_month", "2024-01", "--end_id", "2"],
            ["--start_month", "2023-01", "--start_id", "0",
             "--end_month", "2023-01", "--end_id", "2"],
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with (
                    patch.object(sys, "argv", ["discovery.py", *arguments]),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    discovery.parse_input()

    def test_parse_input_rejects_reversed_or_nonconsecutive_months(self):
        for start_month, end_month in (
            ("2023-02", "2023-01"),
            ("2023-01", "2023-03"),
        ):
            argv = [
                "discovery.py",
                "--start_month", start_month,
                "--start_id", "1",
                "--end_month", end_month,
                "--end_id", "2",
            ]
            with self.subTest(start_month=start_month, end_month=end_month):
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    discovery.parse_input()

    def test_generate_arxiv_ids_for_one_month_and_year_boundary(self):
        same_month = list(discovery.generate_arxiv_ids(1, 2023, 1, 2023, 8, 10))
        across_year = list(discovery.generate_arxiv_ids(12, 2023, 1, 2024, 99999, 2))

        self.assertEqual(same_month, ["2301.00008", "2301.00009", "2301.00010"])
        self.assertEqual(
            across_year,
            ["2312.99999", "2401.00001", "2401.00002"],
        )

    def test_batch_ids_splits_iterable_and_handles_empty_input(self):
        batches = list(discovery.batch_ids((str(i) for i in range(5)), batch_size=2))

        self.assertEqual(batches, [["0", "1"], ["2", "3"], ["4"]])
        self.assertEqual(list(discovery.batch_ids((str(i) for i in range(0)))), [])

    def test_fetch_batch_raw_retries_then_returns_results(self):
        paper = make_paper("2301.00001v1")
        with (
            patch.object(discovery, "MAX_RETRIES", 2),
            patch.object(discovery.client, "results", side_effect=[RuntimeError("busy"), [paper]]) as results,
            patch.object(discovery.time, "sleep") as sleep,
            patch.object(discovery.random, "uniform", return_value=0),
        ):
            fetched = discovery.fetch_batch_raw(["2301.00001"])

        self.assertEqual(fetched, [paper])
        self.assertEqual(results.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(5)])

    def test_fetch_batch_raw_raises_after_retry_limit(self):
        with (
            patch.object(discovery, "MAX_RETRIES", 1),
            patch.object(discovery.client, "results", side_effect=RuntimeError("offline")) as results,
            patch.object(discovery.time, "sleep"),
            patch.object(discovery.random, "uniform", return_value=0),
            patch.object(discovery, "logger"),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                discovery.fetch_batch_raw(["2301.00001"])

        self.assertEqual(results.call_count, 2)

    def test_fetch_batch_with_subdivision_returns_empty_for_empty_input(self):
        with patch.object(discovery, "fetch_batch_raw") as fetch:
            self.assertEqual(discovery.fetch_batch_with_subdivision([]), ([], []))
        fetch.assert_not_called()

    def test_fetch_batch_with_subdivision_recovers_and_tracks_failed_ids(self):
        ids = ["2301.00001", "2301.00002", "2301.00003"]
        first_paper = make_paper(f"{ids[0]}v1")
        third_paper = make_paper(f"{ids[2]}v1")

        def fake_fetch(batch):
            if len(batch) > 1 or batch == [ids[1]]:
                raise RuntimeError("unavailable")
            return [first_paper] if batch == [ids[0]] else [third_paper]

        with (
            patch.object(discovery, "fetch_batch_raw", side_effect=fake_fetch),
            patch.object(discovery, "logger"),
        ):
            papers, failed_ids = discovery.fetch_batch_with_subdivision(ids, min_batch_size=1)

        self.assertEqual(papers, [first_paper, third_paper])
        self.assertEqual(failed_ids, [ids[1]])

    def test_version_helpers_handle_versioned_and_unversioned_ids(self):
        versioned = make_paper("2301.00001v3")
        unversioned = make_paper("2301.00002")

        self.assertEqual(discovery.get_base_id(versioned), "2301.00001")
        self.assertEqual(discovery.get_latest_version(versioned), 3)
        self.assertEqual(discovery.extract_versions(versioned), ["v1", "v2", "v3"])
        self.assertEqual(discovery.get_latest_version(unversioned), 1)
        self.assertEqual(discovery.extract_versions(unversioned), ["v1"])

    def test_append_jsonl_writes_utf8_records_and_skips_empty_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "papers.jsonl"
            discovery.append_jsonl(str(output_path), [])
            self.assertFalse(output_path.exists())

            discovery.append_jsonl(
                str(output_path),
                [{"title": "Résumé", "arxiv_id": "2301.00001"}],
            )
            discovery.append_jsonl(str(output_path), [{"arxiv_id": "2301.00002"}])

            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(records, [
            {"title": "Résumé", "arxiv_id": "2301.00001"},
            {"arxiv_id": "2301.00002"},
        ])

    def test_update_state_atomic_writes_json_and_replaces_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            discovery.update_state_atomic(str(state_path), {"total_found": 4})

            self.assertEqual(json.loads(state_path.read_text()), {"total_found": 4})
            self.assertFalse(Path(f"{state_path}.tmp").exists())

    def test_init_checkpoint_files_creates_files_and_initial_state(self):
        params = make_params()
        with tempfile.TemporaryDirectory() as temp_dir:
            files = discovery.init_checkpoint_files(temp_dir, current_params=params)

            self.assertEqual(set(files), {"papers", "failed", "not_found", "state"})
            self.assertTrue(all(Path(path).exists() for path in files.values()))
            state = json.loads(Path(files["state"]).read_text())

        self.assertEqual(state["input_params"], params)
        self.assertEqual(state["current_phase"], "ID_BATCH")
        self.assertEqual(state["id_batch_last_completed"], 0)

    def test_init_checkpoint_files_uses_month_stream_for_two_month_range(self):
        params = make_params(is_two_months=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            files = discovery.init_checkpoint_files(temp_dir, current_params=params)
            state = json.loads(Path(files["state"]).read_text())

        self.assertEqual(state["current_phase"], "MONTH_1_DATE_STREAM")

    def test_load_checkpoint_state_detects_parameter_change_but_ignores_mode(self):
        original_params = make_params()
        changed_mode = make_params(mode="test")
        changed_range = make_params(end_id=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            state_path.write_text(json.dumps({
                "input_params": original_params,
                "total_found": 7,
            }))

            same_state, same_changed = discovery.load_checkpoint_state(
                str(state_path), current_params=changed_mode
            )
            changed_state, params_changed = discovery.load_checkpoint_state(
                str(state_path), current_params=changed_range
            )

        self.assertEqual(same_state["total_found"], 7)
        self.assertFalse(same_changed)
        self.assertEqual(changed_state["total_found"], 7)
        self.assertTrue(params_changed)

    def test_load_checkpoint_state_rejects_corrupt_json(self):
        params = make_params()
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            state_path.write_text("{invalid")
            with self.assertRaises(discovery.DiscoveryCheckpointError):
                discovery.load_checkpoint_state(
                    str(state_path), current_params=params
                )

    def test_wipe_output_files_clears_jsonl_but_preserves_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files = {
                "papers": str(Path(temp_dir) / "papers.jsonl"),
                "failed": str(Path(temp_dir) / "failed.jsonl"),
                "not_found": str(Path(temp_dir) / "not_found.jsonl"),
                "state": str(Path(temp_dir) / "state.json"),
            }
            for path in files.values():
                Path(path).write_text("keep or clear")

            discovery.wipe_output_files(files)

            self.assertEqual(Path(files["papers"]).read_text(), "")
            self.assertEqual(Path(files["failed"]).read_text(), "")
            self.assertEqual(Path(files["not_found"]).read_text(), "")
            self.assertEqual(Path(files["state"]).read_text(), "keep or clear")

    def test_find_start_timestamp_month_1_probes_until_paper_found(self):
        paper = make_paper("2301.00002v1")
        with patch.object(discovery, "fetch_batch_raw", side_effect=[[], [paper]]) as fetch:
            timestamp = discovery.find_start_timestamp_month_1(2023, 1, 1)

        self.assertEqual(timestamp, "20230102123456")
        self.assertEqual(
            fetch.call_args_list,
            [call(["2301.00001"]), call(["2301.00002"])],
        )

    def test_find_start_timestamp_month_1_raises_after_all_probes(self):
        with (
            patch.object(discovery, "fetch_batch_raw", return_value=[]),
            patch.object(discovery, "logger"),
        ):
            with self.assertRaisesRegex(RuntimeError, "within 30 probes"):
                discovery.find_start_timestamp_month_1(2023, 1, 1)

    def test_fetch_date_range_batch_sends_query_and_offset(self):
        papers = [make_paper("2301.00001v1")]
        with patch.object(discovery.client, "results", return_value=papers) as results:
            fetched = discovery.fetch_date_range_batch("submittedDate:[a TO b]", 10, 5)

        self.assertEqual(fetched, papers)
        search = results.call_args.args[0]
        self.assertEqual(search.query, "submittedDate:[a TO b]")
        self.assertEqual(search.max_results, 15)
        self.assertEqual(results.call_args.kwargs, {"offset": 10})

    def test_process_paper_batch_with_revisions_builds_metadata(self):
        latest = make_paper("2301.00001v3", published_date="2023-01-01", updated_date="2023-03-01")
        revision = make_paper("2301.00001v2", updated_date="2023-02-01")
        with patch.object(
            discovery,
            "fetch_batch_with_subdivision",
            return_value=([revision], []),
        ) as fetch:
            papers, failed_ids = discovery.process_paper_batch_with_revisions(
                {"2301.00001": latest}, [], min_subbatch_size=1
            )

        fetch.assert_called_once_with(["2301.00001v2"], min_batch_size=1)
        self.assertEqual(failed_ids, [])
        self.assertEqual(papers, [{
            "arxiv_id": "2301.00001",
            "title": "Paper 2301.00001v3",
            "authors": ["Ada Lovelace"],
            "submission_date": "2023-01-01",
            "revised_dates": ["2023-02-01", "2023-03-01"],
            "venue": "Journal",
            "versions": ["v1", "v2", "v3"],
        }])

    def test_process_paper_batch_with_revisions_marks_missing_revision_failed(self):
        latest = make_paper("2301.00001v3")
        with patch.object(
            discovery,
            "fetch_batch_with_subdivision",
            return_value=([], ["2301.00001v2"]),
        ), patch.object(discovery, "logger"):
            papers, failed_ids = discovery.process_paper_batch_with_revisions(
                {"2301.00001": latest}, ["2301.00002", "2301.00002"], 1
            )

        self.assertEqual(papers, [])
        self.assertEqual(failed_ids, ["2301.00002", "2301.00001"])

    def test_discover_ids_stream_classifies_found_not_found_and_failed_ids(self):
        ids = ["2301.00001", "2301.00002", "2301.00003"]
        paper = make_paper("2301.00001v1")
        metadata = {"arxiv_id": ids[0], "title": paper.title}
        with (
            patch.object(
                discovery,
                "fetch_batch_with_subdivision",
                return_value=([paper], [ids[1]]),
            ),
            patch.object(
                discovery,
                "process_paper_batch_with_revisions",
                return_value=([metadata], [ids[1]]),
            ),
        ):
            batches = list(discovery.discover_ids_stream(ids, batch_size=3))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["phase"], "ID_BATCH")
        self.assertEqual(batches[0]["requested_ids"], ids)
        self.assertEqual(batches[0]["found_papers"], [metadata])
        self.assertEqual(batches[0]["found_ids"], [ids[0]])
        self.assertEqual(batches[0]["not_found_ids"], [ids[2]])
        self.assertEqual(batches[0]["failed_ids"], [ids[1]])

    def test_discover_ids_stream_persists_batch_results_and_checkpoint(self):
        ids = ["2301.00001", "2301.00002", "2301.00003"]
        params = {
            "candidate_count": len(ids),
            "candidate_ids_sha256": discovery._ids_fingerprint(ids),
            "batch_size": 3,
            "min_subbatch_size": discovery.MIN_SUBBATCH_SIZE,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(discovery, "fetch_batch_with_subdivision", return_value=([], [])),
                patch.object(discovery, "process_paper_batch_with_revisions", return_value=([], [])),
            ):
                batches = list(discovery.discover_ids_stream(
                    ids, batch_size=3, output_dir=temp_dir,
                    input_params=params,
                ))

            state_path = Path(temp_dir) / "discovery_state.json"
            state = json.loads(state_path.read_text())
            not_found = [
                json.loads(line)
                for line in (Path(temp_dir) / "discovery_not_found.jsonl").read_text().splitlines()
            ]

        self.assertEqual(len(batches), 1)
        self.assertEqual(state["id_batch_last_completed"], 1)
        self.assertEqual(state["id_batch_last_processed_id"], "2301.00003")
        self.assertEqual(state["total_not_found"], 3)
        self.assertEqual(not_found, [{"arxiv_id": "2301.00001"}, {"arxiv_id": "2301.00002"}, {"arxiv_id": "2301.00003"}])

    def test_discover_ids_stream_resumes_after_completed_batches(self):
        ids = ["2301.00001", "2301.00002", "2301.00003"]
        params = {
            "candidate_count": len(ids),
            "candidate_ids_sha256": discovery._ids_fingerprint(ids),
            "batch_size": 1,
            "min_subbatch_size": discovery.MIN_SUBBATCH_SIZE,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            files = discovery.init_checkpoint_files(temp_dir, current_params=params)
            state = json.loads(Path(files["state"]).read_text())
            state["id_batch_last_completed"] = 2
            discovery.update_state_atomic(files["state"], state)

            with patch.object(discovery, "fetch_batch_with_subdivision", return_value=([], [])) as fetch:
                batches = list(discovery.discover_ids_stream(
                    ids, batch_size=1, output_dir=temp_dir,
                    input_params=params,
                ))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["requested_ids"], ["2301.00003"])
        fetch.assert_called_once()

    def test_discover_ids_stream_rejects_missing_state_with_existing_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_path = Path(temp_dir) / "papers.jsonl"
            papers_path.write_text(
                json.dumps({"arxiv_id": "2301.00001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(discovery.DiscoveryCheckpointError):
                list(discovery.discover_ids_stream([], output_dir=temp_dir))

            self.assertFalse((Path(temp_dir) / "discovery_state.json").exists())

    def test_discover_ids_stream_rejects_corrupt_state_with_existing_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "papers.jsonl").write_text(
                json.dumps({"arxiv_id": "2301.00001"}) + "\n",
                encoding="utf-8",
            )
            (root / "discovery_state.json").write_text(
                "{invalid",
                encoding="utf-8",
            )

            with self.assertRaises(discovery.DiscoveryCheckpointError):
                list(discovery.discover_ids_stream([], output_dir=temp_dir))

    def test_discover_ids_stream_rejects_changed_input_without_appending(self):
        ids = ["2301.00001", "2301.00002"]
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                discovery,
                "fetch_batch_with_subdivision",
                return_value=([], []),
            ):
                list(
                    discovery.discover_ids_stream(
                        ids,
                        batch_size=2,
                        output_dir=temp_dir,
                    )
                )

            papers_path = Path(temp_dir) / "papers.jsonl"
            before = papers_path.read_text(encoding="utf-8")

            with self.assertRaises(ValueError):
                list(
                    discovery.discover_ids_stream(
                        [*ids, "2301.00003"],
                        batch_size=2,
                        output_dir=temp_dir,
                    )
                )

            self.assertEqual(
                papers_path.read_text(encoding="utf-8"),
                before,
            )

    def test_discover_two_month_stream_transitions_to_second_month(self):
        params = make_params(
            start_month="2023-01",
            start_id=5,
            end_month="2023-02",
            end_id=1,
            end_month_num=2,
            is_two_months=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(discovery, "find_start_timestamp_month_1", return_value="20230101000000"),
                patch.object(discovery, "fetch_date_range_batch", return_value=[]) as date_fetch,
                patch.object(discovery, "fetch_batch_with_subdivision", return_value=([], [])),
                patch.object(discovery, "process_paper_batch_with_revisions", return_value=([], [])),
            ):
                batches = list(discovery.discover_two_month_range_stream(
                    params, batch_size=10, output_dir=temp_dir
                ))

            state = json.loads((Path(temp_dir) / "discovery_state.json").read_text())

        date_fetch.assert_called_once_with(
            "submittedDate:[20230101000000 TO 20230131235959]",
            offset=0,
            batch_size=10,
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["requested_ids"], ["2302.00001"])
        self.assertEqual(state["current_phase"], "MONTH_2_ID_BATCH")

    def test_main_wires_arguments_to_discovery_stream(self):
        args = SimpleNamespace(
            start_month="2023-01",
            start_id=1,
            end_month="2023-01",
            end_id=3,
            batch_size=3,
            mode="test",
            output_dir="tmp-output",
            start_year=2023,
            start_month_num=1,
            end_year=2023,
            end_month_num=1,
            month_diff=0,
        )
        with (
            patch.object(discovery, "setup_logging") as setup_logging,
            patch.object(discovery, "parse_input", return_value=args),
            patch.object(discovery, "discover_two_month_range_stream", return_value=iter([{}])) as discover,
            patch.object(discovery.time, "perf_counter", side_effect=[1.0, 2.0]),
            patch.object(discovery, "logger"),
        ):
            discovery.main()

        setup_logging.assert_called_once_with()
        discover.assert_called_once()
        self.assertEqual(discover.call_args.kwargs["args_params"]["is_two_months"], False)
        self.assertEqual(discover.call_args.kwargs["output_dir"], "tmp-output")


if __name__ == "__main__":
    unittest.main()
