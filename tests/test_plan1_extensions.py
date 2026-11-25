"""Unit tests for Plan 1 extensions: vectorized hard/soft DS, calibration, PG-CRC, and RFC 9162 Merkle log."""
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deciv_extensions import (
    MerkleTransparencyLog,
    VectorizedDawidSkene,
    apply_temperature,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    fit_temperature,
    route_pg_crc,
)

RESULTS = ROOT / "results"


class TestPlan1Extensions(unittest.TestCase):
    def test_soft_ds_reduces_to_hard_ds_on_onehot(self):
        rng = np.random.default_rng(42)
        votes = rng.integers(0, 4, size=(200, 4))
        onehot = np.eye(4, dtype=np.float64)[votes]
        ds_hard = VectorizedDawidSkene(smoothing=1e-2).fit_hard(votes)
        ds_soft = VectorizedDawidSkene(smoothing=1e-2).fit_soft(onehot)
        p_h = ds_hard.predict_proba_hard(votes, apply_temp=False)
        p_s = ds_soft.predict_proba_soft(onehot, apply_temp=False)
        self.assertLess(float(np.max(np.abs(p_h - p_s))), 1e-12)

    def test_temperature_scaling_reduces_nll(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 4, size=400)
        # Artificially overconfident predictions
        logits = rng.normal(0, 1.5, size=(400, 4))
        logits[np.arange(400), y] += 2.5
        logits *= 4.0
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        before = calibration_metrics(p, y)["nll"]
        t = fit_temperature(p, y)
        after = calibration_metrics(apply_temperature(p, t), y)["nll"]
        self.assertGreater(t, 1.5)
        self.assertLess(after, before)

    def test_pg_crc_strictly_enforces_hard_eligibility(self):
        ctx_bad = {
            "provenance_status": "incomplete",
            "evidence_auditability": "auditable",
            "human_review_required": "false",
            "repairability": "no_repair",
        }
        act, adm = route_pg_crc([1.0, 0.0, 0.0, 0.0], ctx_bad, tau_accept=0.0)
        self.assertEqual(act, "defer")
        self.assertTrue(adm)

    def test_merkle_inclusion_and_consistency_proofs_exhaustive(self):
        log = MerkleTransparencyLog()
        payloads = [f"rec-{i}".encode() for i in range(137)]
        roots = {}
        for i, p in enumerate(payloads):
            log.append(p)
            n = i + 1
            if n in (1, 2, 3, 4, 7, 8, 15, 16, 31, 64, 100, 137):
                roots[n] = log.root()
        root_final = log.root()
        for idx in (0, 1, 2, 7, 32, 63, 64, 99, 136):
            proof = log.inclusion_proof(idx)
            self.assertTrue(MerkleTransparencyLog.verify_inclusion(payloads[idx], idx, len(payloads), proof, root_final))
            self.assertFalse(MerkleTransparencyLog.verify_inclusion(b"forged", idx, len(payloads), proof, root_final))
        for m in (1, 2, 3, 4, 7, 8, 15, 16, 31, 64, 100):
            cproof = log.consistency_proof(m, len(payloads))
            self.assertTrue(MerkleTransparencyLog.verify_consistency(m, len(payloads), roots[m], root_final, cproof))
            self.assertFalse(MerkleTransparencyLog.verify_consistency(m, len(payloads), b"\x00" * 32, root_final, cproof))

    def test_plan1_result_artifacts_exist_and_pass_bounds(self):
        with (RESULTS / "calibration_and_estimators.csv").open(encoding="utf-8") as f:
            cal = {r["method_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(int(cal["ds_temp"]["eligibility_violations"]), 0)
        self.assertAlmostEqual(float(cal["ds_temp"]["far"]), 1 / 1630, places=6)
        self.assertLess(float(cal["ds_temp"]["mean_institutional_cost"]), float(cal["ds_raw"]["mean_institutional_cost"]))

        with (RESULTS / "correlated_verifier_stress.csv").open(encoding="utf-8") as f:
            stress = {(float(r["rho"]), r["method_id"]): r for r in csv.DictReader(f)}
        # Confirm raw DS FAR collapses at rho=0.6 while PG-CRC stays <= alpha=0.01
        self.assertGreater(float(stress[(0.6, "deciv_ds_raw")]["far_mean"]), 0.03)
        self.assertLessEqual(float(stress[(0.6, "deciv_pg_crc")]["far_mean"]), 0.01)
        self.assertLessEqual(float(stress[(0.6, "deciv_ds_temp")]["far_mean"]), 0.005)


if __name__ == "__main__":
    unittest.main()
