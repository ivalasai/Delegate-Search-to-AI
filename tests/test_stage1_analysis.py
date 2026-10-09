import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_variance import classify_result_status
from generate_identity_exports import jaccard, pair_set


class Stage1AnalysisStatusTests(unittest.TestCase):
    def test_complete_legacy_result_is_complete(self):
        result = {"run_manifest": {}, "couplings": []}
        self.assertEqual(classify_result_status(result), "complete")

    def test_partial_result_is_not_complete(self):
        result = {"status": "in_progress", "run_manifest": {}, "couplings": []}
        self.assertEqual(classify_result_status(result, "running"), "in_progress")

    def test_absent_result_is_not_run(self):
        self.assertEqual(classify_result_status(None, None), "not_run")

    def test_identity_jaccard_is_set_based(self):
        self.assertEqual(jaccard({("P1", "S1")}, {("P1", "S1"), ("P2", "S2")}), 0.5)
        self.assertIsNone(jaccard(set(), set()))

    def test_pair_set_extracts_problem_solution_ids(self):
        payload = {"solution_decoupling": {"S2": "S2"}, "couplings": [{"problem_id": "P1", "solution_id": "S2"}]}
        self.assertEqual(pair_set(payload), {("P1", "S2")})


if __name__ == "__main__":
    unittest.main()
