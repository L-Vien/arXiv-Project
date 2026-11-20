"""Small real-data integration fixture for exhaustive candidate generation."""

import json
from pathlib import Path
import unittest

from src.matching.candidates import generate_candidate_pairs
from src.matching.contracts import BibtexRecord, Partition, ReferenceRecord
from src.matching.features import FEATURE_NAMES, FeatureBuilder


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "matching" 


def load_jsonl(filename: str) -> list[dict]:
    with (FIXTURE_DIR / filename).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class RealCandidateFixtureTests(unittest.TestCase):
    def test_exhaustive_pairs_from_real_metadata(self) -> None:
        bibtex = load_jsonl("bibtex.jsonl")
        references = load_jsonl("references.jsonl")
        expected = load_jsonl("expected_candidates.jsonl")

        actual = [
            {
                "arxiv_id": pair.arxiv_id,
                "citation_key": pair.citation_key,
                "ref_arxiv_id": pair.ref_arxiv_id,
            }
            for pair in generate_candidate_pairs(bibtex, references)
        ]

        self.assertEqual(len(bibtex), 5)
        self.assertEqual(len(references), 5)
        self.assertEqual(len(actual), 25)
        self.assertEqual(actual, expected)
        self.assertTrue(all(row["arxiv_id"] == "2604.25560" for row in actual))
        self.assertEqual(len({(row["arxiv_id"], row["citation_key"], row["ref_arxiv_id"]) for row in actual}), 25)

    def test_real_fixture_candidates_flow_into_26_features(self) -> None:
        # This is a *temporary test role*, not an official partition assignment.
        bibtex = [BibtexRecord(**item) for item in load_jsonl("bibtex.jsonl")]
        references = [ReferenceRecord(**item) for item in load_jsonl("references.jsonl")]
        partitions = {"2604.25560": Partition.TRAIN}

        builder = FeatureBuilder().fit_on_train(bibtex, references, partitions)
        rows = list(builder.build_rows(
            generate_candidate_pairs(bibtex, references),
            bibtex,
            references,
            partitions,
        ))

        self.assertEqual(len(rows), 25)
        self.assertTrue(all(tuple(row.values) == FEATURE_NAMES for row in rows))


if __name__ == "__main__":
    unittest.main()
