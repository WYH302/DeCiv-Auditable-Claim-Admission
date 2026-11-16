"""Automated verification tests for the DeCiv benchmark, 5-fold CV, and ablation outputs."""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


class TestDeCivReproducibility(unittest.TestCase):
    def test_benchmark_composition(self):
        with (DATA / "evaluation_index_v2.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1800)
        parents = {r["parent_id"] for r in rows}
        self.assertEqual(len(parents), 450)

    def test_main_metrics_and_zero_risk_violations(self):
        with (RESULTS / "main_metrics.csv").open(encoding="utf-8") as f:
            metrics = {(r["method"], r["router"]): r for r in csv.DictReader(f)}
        self.assertEqual(len(metrics), 10)
        for method in ("majority", "reliability_weighted", "dawid_skene", "fixed", "adaptive"):
            self.assertEqual(int(metrics[(method, "risk")]["eligibility_violations"]), 0)
        self.assertAlmostEqual(float(metrics[("majority", "direct")]["far"]), 172 / 1630, places=6)
        self.assertAlmostEqual(float(metrics[("fixed", "risk")]["far"]), 1 / 1630, places=6)
        self.assertAlmostEqual(float(metrics[("adaptive", "risk")]["accuracy"]), 839 / 1800, places=6)
        self.assertAlmostEqual(float(metrics[("adaptive", "risk")]["non_defer_fraction"]), 0.25, places=6)
        self.assertAlmostEqual(float(metrics[("dawid_skene", "risk")]["accuracy"]), 1374 / 1800, places=6)
        self.assertAlmostEqual(float(metrics[("dawid_skene", "risk")]["far"]), 3 / 1630, places=6)

    def test_five_fold_cv_and_ablations(self):
        with (RESULTS / "five_fold_cv.csv").open(encoding="utf-8") as f:
            cv5 = {(r["method"], r["router"]): r for r in csv.DictReader(f)}
        self.assertEqual(len(cv5), 10)
        self.assertAlmostEqual(float(cv5[("adaptive", "risk")]["far_mean"]), 1 / 1630, delta=0.002)

        with (RESULTS / "ablation_metrics.csv").open(encoding="utf-8") as f:
            ab = {r["variant_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(ab), 11)
        self.assertEqual(int(ab["adaptive_full"]["eligibility_violations"]), 0)
        self.assertGreater(int(ab["adaptive_no_prov_gate"]["eligibility_violations"]), 40)
        self.assertGreater(int(ab["adaptive_no_repair_gate"]["eligibility_violations"]), 15)

        with (RESULTS / "sota_baselines.csv").open(encoding="utf-8") as f:
            sota = {r["baseline_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(len(sota), 5)
        self.assertEqual(int(sota["deciv_glad_risk"]["eligibility_violations"]), 0)
        self.assertAlmostEqual(float(sota["deciv_glad_risk"]["far"]), 1 / 1630, places=6)
        self.assertGreater(int(sota["selective_threshold"]["eligibility_violations"]), 150)

    def test_predictions_completeness(self):
        with (RESULTS / "predictions.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 18000)

    def test_paired_execution_trace_audit_hashes(self):
        trace = json.loads((RESULTS / "execution_trace.json").read_text(encoding="utf-8"))
        self.assertEqual(len(trace), 2)
        clean, drop = trace[0], trace[1]
        self.assertEqual(clean["packet_id"], "FEV_0003__v0")
        self.assertEqual(drop["packet_id"], "FEV_0003__v1")
        self.assertEqual(clean["votes"], drop["votes"])
        self.assertTrue(clean["writes_trusted_store"])
        self.assertFalse(drop["writes_trusted_store"])
        self.assertEqual(drop["action"], "defer")
        self.assertEqual(len(clean["audit_record_hash"]), 64)
        self.assertEqual(len(drop["audit_record_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
