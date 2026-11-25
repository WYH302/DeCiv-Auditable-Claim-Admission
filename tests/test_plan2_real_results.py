"""Automated verification tests for Plan 2 real 4-model GPU verifiers, RAG defense, and remediation."""
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class TestPlan2RealGPUExperiments(unittest.TestCase):
    def test_real_benchmark_and_verifier_caches(self):
        rows = [json.loads(l) for l in (ROOT / "data" / "deciv_real_1200.jsonl").open(encoding="utf-8") if l.strip()]
        self.assertEqual(len(rows), 1200)
        self.assertEqual(len({r["parent_id"] for r in rows}), 300)
        self.assertEqual(len({r["source_dataset"] for r in rows}), 4)
        for vid in ("deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b"):
            p = RESULTS / "real_verifiers" / f"{vid}_probs.json"
            self.assertTrue(p.exists(), f"Missing verifier output: {vid}")
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(d["n_unique_pairs"], 900)

    def test_real_correlation_and_admission_guarantees(self):
        with (RESULTS / "real_verifier_correlation.csv").open(encoding="utf-8") as f:
            corr = {(r["verifier_i"], r["verifier_j"]): float(r["error_pearson_rho"]) for r in csv.DictReader(f)}
        # LLM-LLM error correlation must exceed cross-family NLI-LLM correlation
        self.assertGreater(corr[("llama31_8b", "qwen35_9b")], 0.40)
        self.assertLess(corr[("bart_large_mnli", "llama31_8b")], 0.25)

        with (RESULTS / "real_admission_metrics.csv").open(encoding="utf-8") as f:
            adm = {r["method_id"]: r for r in csv.DictReader(f)}
        # PG-CRC strictly bounds FAR <= alpha and preserves EV = 0
        self.assertEqual(int(adm["deciv_hard_ds_pg_crc_010"]["eligibility_violations"]), 0)
        self.assertLessEqual(float(adm["deciv_hard_ds_pg_crc_010"]["far"]), 0.01)
        self.assertLess(float(adm["deciv_hard_ds_pg_crc_010"]["far"]), float(adm["deciv_hard_ds_raw"]["far"]) / 5.0)
        self.assertEqual(int(adm["deciv_sup_pg_crc_005"]["eligibility_violations"]), 0)
        self.assertLessEqual(float(adm["deciv_sup_pg_crc_005"]["far"]), 0.005)
        self.assertLess(float(adm["deciv_sup_pg_crc_005"]["mean_institutional_cost"]), 0.35)

    def test_real_remediation_and_rag_defense(self):
        rem = json.loads((RESULTS / "real_remediation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(rem["revise_queue_precision"], 1.0)
        self.assertGreater(rem["readmitted_evidence_entailment_rate"], 0.90)
        self.assertEqual(rem["merkle_proofs_verified"], f"{rem['n_routed_revise']}/{rem['n_routed_revise']}")

        with (RESULTS / "real_rag_defense_metrics.csv").open(encoding="utf-8") as f:
            rag = {r["regime_id"]: r for r in csv.DictReader(f)}
        # Unfiltered RAG suffers >10% poisoning ASR, while DeCiv PG-CRC reaches 0.0% ASR
        self.assertGreater(float(rag["no_filter"]["downstream_poisoning_asr"]), 0.10)
        self.assertEqual(float(rag["deciv_sup_pg_crc_005"]["downstream_poisoning_asr"]), 0.0)
        # Auto-revise loop boosts supported QA accuracy over base Soft-DS without increasing ASR
        self.assertGreater(
            float(rag["deciv_plus_revise"]["downstream_qa_accuracy_sup100"]),
            float(rag["deciv_soft_ds_sup_temp"]["downstream_qa_accuracy_sup100"]),
        )
        self.assertLessEqual(
            float(rag["deciv_plus_revise"]["downstream_poisoning_asr"]),
            float(rag["deciv_soft_ds_sup_temp"]["downstream_poisoning_asr"]),
        )

    def test_trans_diversity_ablation_and_pgcrc_alpha_sweep(self):
        with (RESULTS / "real_verifier_diversity_ablation.csv").open(encoding="utf-8") as f:
            div = {r["subset_id"]: r for r in csv.DictReader(f)}
        # LLM-only collapses on revise recall; full 4-model cross-family lifts accuracy > +15 pp
        self.assertEqual(float(div["llm_only"]["recall_revise"]), 0.0)
        self.assertGreater(float(div["full_4_cross"]["recall_revise"]), 0.55)
        self.assertGreater(
            float(div["full_4_cross"]["accuracy"]) - float(div["llm_only"]["accuracy"]),
            0.15,
        )

        with (RESULTS / "real_pgcrc_alpha_sweep.csv").open(encoding="utf-8") as f:
            sweep = list(csv.DictReader(f))
        self.assertEqual(len(sweep), 12)
        for row in sweep:
            # Strict Theorem 1 finite-sample compliance across all 12 alpha sweep points
            self.assertLessEqual(float(row["far_mean"]), float(row["alpha"]) + 1e-9)
        s05 = {r["base_estimator"]: r for r in sweep if abs(float(r["alpha"]) - 0.005) < 1e-9}
        self.assertEqual(float(s05["hard_ds_raw"]["recall_accept_mean"]), 0.0)
        self.assertGreater(float(s05["sup_soft_ds_temp"]["recall_accept_mean"]), 0.40)


if __name__ == "__main__":
    unittest.main()
