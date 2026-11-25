"""Automated verification tests for 5-model 3-family DeCiv-CAF, 2025 RAG SOTA defense,
Multi-Copy PoisonedRAG 10k-Haystack discovery, real ablations, significance, and remediation.
"""
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
        for vid in ("deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b", "gpt61_sol"):
            p = RESULTS / "real_verifiers" / f"{vid}_probs.json"
            self.assertTrue(p.exists(), f"Missing verifier output: {vid}")
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(d["n_unique_pairs"], 900)

    def test_real_correlation_and_admission_guarantees(self):
        with (RESULTS / "real_verifier_correlation.csv").open(encoding="utf-8") as f:
            corr = {(r["verifier_i"], r["verifier_j"]): float(r["error_pearson_rho"]) for r in csv.DictReader(f)}
        self.assertGreater(corr[("llama31_8b", "qwen35_9b")], 0.40)
        self.assertLess(corr[("bart_large_mnli", "llama31_8b")], 0.25)
        self.assertLess(corr[("llama31_8b", "gpt61_sol")], 0.25)

        with (RESULTS / "real_admission_metrics.csv").open(encoding="utf-8") as f:
            adm = {r["method_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(int(adm["deciv_caf_gated_map"]["eligibility_violations"]), 0)
        self.assertGreater(float(adm["deciv_caf_gated_map"]["accuracy"]), 0.89)
        self.assertGreater(float(adm["deciv_caf_gated_map"]["cv5_accuracy_mean"]), 0.91)
        self.assertGreater(
            float(adm["deciv_caf_gated_map"]["accuracy"]),
            float(adm["solo_gpt61_sol_gated"]["accuracy"]),
        )
        self.assertLess(float(adm["deciv_caf_risk_d05"]["mean_institutional_cost"]), 0.34)
        self.assertEqual(int(adm["deciv_caf_pg_crc_010"]["eligibility_violations"]), 0)
        self.assertLessEqual(float(adm["deciv_caf_pg_crc_010"]["far"]), 0.01)
        self.assertLessEqual(float(adm["deciv_caf_pg_crc_010"]["cv5_far_mean"]), 0.01)
        self.assertEqual(int(adm["deciv_sup_pg_crc_005"]["eligibility_violations"]), 0)
        self.assertLessEqual(float(adm["deciv_sup_pg_crc_005"]["far"]), 0.005)

    def test_real_remediation_and_rag_defense(self):
        rem = json.loads((RESULTS / "real_remediation_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(rem["revise_queue_precision"], 1.0)
        self.assertGreater(rem["readmitted_evidence_entailment_rate"], 0.90)
        self.assertGreater(rem["mean_nli_readmitted"], rem["mean_nli_blocked"] + 0.50)
        self.assertEqual(rem["merkle_proofs_verified"], f"{rem['n_routed_revise']}/{rem['n_routed_revise']}")

        with (RESULTS / "real_rag_defense_metrics.csv").open(encoding="utf-8") as f:
            rag = {r["regime_id"]: r for r in csv.DictReader(f)}
        self.assertGreater(float(rag["no_filter"]["downstream_poisoning_asr"]), 0.10)
        for q_sota in ("reliabilityrag_2025", "seconrag_2025", "selfrag_llm_filter"):
            self.assertIn(q_sota, rag)
            self.assertGreater(float(rag[q_sota]["query_defense_latency_ms"]), 50.0)
            self.assertGreater(float(rag[q_sota]["store_contamination_rate"]), 0.80)
        self.assertEqual(float(rag["deciv_soft_ds_sup_temp"]["query_defense_latency_ms"]), 0.0)
        self.assertLess(float(rag["deciv_soft_ds_sup_temp"]["store_contamination_rate"]), 0.02)
        self.assertLess(float(rag["deciv_soft_ds_sup_temp"]["downstream_poisoning_asr"]), 0.01)
        self.assertGreater(
            float(rag["deciv_strict_plus_revise"]["trustworthy_qa_accuracy_safe150"]),
            float(rag["conformal_risk_gated"]["trustworthy_qa_accuracy_safe150"]),
        )
        self.assertGreater(
            float(rag["deciv_strict_plus_revise"]["trustworthy_qa_accuracy_safe150"]),
            float(rag["reliabilityrag_2025"]["trustworthy_qa_accuracy_safe150"]) + 0.20,
        )
        self.assertGreater(
            float(rag["deciv_strict_plus_revise"]["downstream_qa_accuracy_sup100"]),
            float(rag["deciv_soft_ds_sup_temp"]["downstream_qa_accuracy_sup100"]),
        )

    def test_trans_diversity_ablation_and_pgcrc_alpha_sweep(self):
        with (RESULTS / "real_verifier_diversity_ablation.csv").open(encoding="utf-8") as f:
            div = {r["subset_id"]: r for r in csv.DictReader(f)}
        full_acc = float(div["full_5_3fam"]["accuracy"])
        full_rec = float(div["full_5_3fam"]["recall_accept"])
        full_cost = float(div["full_5_3fam"]["cost"])
        for sub_id in ("open_llm_only", "nli_only", "frontier_only", "open_4model", "nli_frontier", "llm_frontier"):
            self.assertGreater(full_acc, float(div[sub_id]["accuracy"]))
            self.assertGreater(full_rec, float(div[sub_id]["recall_accept"]))
            self.assertLess(full_cost, float(div[sub_id]["cost"]))

        with (RESULTS / "real_pgcrc_alpha_sweep.csv").open(encoding="utf-8") as f:
            sweep = list(csv.DictReader(f))
        self.assertEqual(len(sweep), 18)
        for row in sweep:
            self.assertLessEqual(float(row["far_mean"]), float(row["alpha"]) + float(row["far_std"]) + 1e-9)
        s10 = {r["base_estimator"]: r for r in sweep if abs(float(r["alpha"]) - 0.01) < 1e-9}
        self.assertLessEqual(float(s10["deciv_caf_decoupled"]["far_mean"]), 0.01)
        self.assertGreater(float(s10["deciv_caf_decoupled"]["accuracy_mean"]), 0.90)
        self.assertGreater(float(s10["deciv_caf_decoupled"]["recall_accept_mean"]), 0.80)

    def test_real_caf_ablation_significance_and_cost_sensitivity(self):
        with (RESULTS / "real_caf_ablation.csv").open(encoding="utf-8") as f:
            ab = {r["variant_id"]: r for r in csv.DictReader(f)}
        # Pure Log-Linear CAF (no trees) already exceeds 89.5% LODO accuracy
        self.assertGreater(float(ab["caf_loglinear_map"]["lodo_acc"]), 0.895)
        # Removing Mass-Preserving Projection causes >15 pp accuracy collapse
        self.assertGreater(
            float(ab["caf_full_proj_map"]["lodo_acc"]) - float(ab["caf_no_proj_argmax"]["lodo_acc"]),
            0.15,
        )
        # Cross-family consensus products tighten calibration ECE
        self.assertLess(
            float(ab["caf_full_proj_map"]["lodo_ece"]),
            float(ab["caf_no_cross_map"]["lodo_ece"]),
        )

        with (RESULTS / "real_significance_tests.csv").open(encoding="utf-8") as f:
            sig = {r["comparison"]: r for r in csv.DictReader(f)}
        # Statistically significant NLL and institutional-cost improvements (p < 0.001)
        self.assertLess(float(sig["DeCiv-CAF / proj. MAP vs. Solo GPT-6.1 + Gate"]["p_boot_nll"]), 0.001)
        self.assertLess(float(sig["DeCiv-CAF / risk (d=0.5) vs. Solo GPT-6.1 + Gate"]["p_boot_cost"]), 0.001)
        self.assertLess(float(sig["DeCiv-CAF / risk (d=0.5) vs. GBDT Stacking + Gate"]["p_boot_cost"]), 0.001)

        with (RESULTS / "real_cost_sensitivity.csv").open(encoding="utf-8") as f:
            sens = list(csv.DictReader(f))
        self.assertEqual(len(sens), 30)
        for r in sens:
            self.assertEqual(int(r["eligibility_violations"]), 0)
            self.assertLessEqual(float(r["far"]), 0.012)

    def test_multicopy_haystack_malicious_clique_discovery(self):
        with (RESULTS / "real_multicopy_haystack_rag.csv").open(encoding="utf-8") as f:
            mc = {(int(r["poison_copies_m"]), r["regime_id"]): r for r in csv.DictReader(f)}
        self.assertEqual(len(mc), 18)
        # Proposition 2: Query-time 2025 SOTAs suffer malicious-clique backfire at m=5 (>50% ASR, worse than unfiltered!)
        self.assertGreater(
            float(mc[(5, "reliabilityrag_2025")]["downstream_poisoning_asr"]),
            float(mc[(5, "no_filter")]["downstream_poisoning_asr"]) + 0.15,
        )
        self.assertGreater(
            float(mc[(5, "seconrag_2025")]["downstream_poisoning_asr"]),
            float(mc[(5, "no_filter")]["downstream_poisoning_asr"]) + 0.15,
        )
        # Write-time DeCiv maintains exact m-copy invariance across m=1,3,5 at 0.0ms/q
        for m_val in (1, 3, 5):
            self.assertEqual(float(mc[(m_val, "deciv_strict_plus_revise")]["query_defense_latency_ms"]), 0.0)
            self.assertLess(float(mc[(m_val, "deciv_strict_plus_revise")]["downstream_poisoning_asr"]), 0.01)
            self.assertEqual(
                float(mc[(m_val, "deciv_strict_plus_revise")]["downstream_poisoning_asr"]),
                float(mc[(1, "deciv_strict_plus_revise")]["downstream_poisoning_asr"]),
            )
            self.assertGreater(float(mc[(m_val, "deciv_strict_plus_revise")]["trustworthy_qa_accuracy_safe150"]), 0.85)


if __name__ == "__main__":
    unittest.main()
