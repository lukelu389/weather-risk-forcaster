"""Public verification stays useful without private data or research libraries."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_results", ROOT / "scripts/verify_results.py")
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


class PublicResultsTests(unittest.TestCase):
    def test_published_evidence_recomputes_complete_coverage_family(self):
        result = public.verify(ROOT)
        self.assertEqual(result["coverage_rows_recomputed"], 54)
        self.assertEqual(result["coverage_event_rows"], 27882)
        self.assertEqual(result["sealed_source_files"], 20)
        self.assertEqual(result["private_sealed_inputs_not_bundled"], 4)

    def test_missing_grid_rows_do_not_create_false_transition_pairs(self):
        result = public.coverage([0, 1, None, 1, 0, 0, 1], .95)
        self.assertEqual([result[k] for k in ("n00", "n01", "n10", "n11")], [1, 2, 1, 0])
        self.assertEqual(result["adjacent_pairs"], 4)
        self.assertEqual(result["exceedances"], 3)
        self.assertEqual(result["common_scored_rows"], 6)

    def test_exact_binomial_accounts_for_both_tails_and_probability_ties(self):
        self.assertAlmostEqual(public.binomial_pvalue(0, 10, .5), 2 / 1024)
        self.assertAlmostEqual(public.binomial_pvalue(10, 10, .5), 2 / 1024)
        self.assertAlmostEqual(public.binomial_pvalue(5, 10, .5), 1.)

    def test_unidentified_transition_row_remains_unavailable(self):
        result = public.coverage([0, 0, None, 0], .99)
        self.assertIsNone(result["independence_pvalue"])
        self.assertIsNone(result["conditional_pvalue"])
        self.assertEqual(result["exceedances"], 0)

    def test_holm_retains_unavailable_hypotheses_in_family_size(self):
        self.assertEqual(public.holm([.01, None, .02, .9]), [.04, None, .06, 1.])
        with self.assertRaises(ValueError):
            public.holm([float("nan")])

    def test_changed_artifact_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "scores.csv"
            path.write_text("original\n")
            manifest = {"artifacts": {"scores.csv": hashlib.sha256(path.read_bytes()).hexdigest()}}
            public.check_manifest(root, manifest)
            path.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                public.check_manifest(root, manifest)

    def test_manifest_cannot_escape_evidence_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes"):
                public.check_manifest(Path(temporary), {"artifacts": {"../private-input.csv": "0" * 64}})

    def test_duplicate_or_missing_event_grid_position_fails(self):
        header = "asset,date,grid_index,model,confidence,valid,exception\n"
        for index in (0, 2):
            content = (header + "wti_spot,2024-01-01,0,historical,0.95,1,0\n"
                       + f"wti_spot,2024-01-02,{index},historical,0.95,1,1\n")
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "events.csv.gz"
                path.write_bytes(gzip.compress(content.encode(), mtime=0))
                with self.assertRaisesRegex(ValueError, "grid row"):
                    public.load_events(path)


if __name__ == "__main__":
    unittest.main()
