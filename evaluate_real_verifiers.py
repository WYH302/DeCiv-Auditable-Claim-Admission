"""Evaluate 4 real heterogeneous GPU verifiers on DeCiv-Real-1200 (zero-leakage CV).

Computes:
  1. Real empirical inter-model error correlation matrix & Cohen's kappa across
     (DeBERTa-v3-Large, BART-Large-MNLI, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3).
  2. Strict 4-fold Leave-One-Domain-Out + 5-fold stratified parent-cluster CV comparing:
       - Individual verifiers (solo 7B LLM / NLI baselines)
       - Majority / direct, Selective-Threshold (2023), Margin-Abstention (2024),
         Conformal-Risk (ICLR 2024), Post-Hoc Cost L2D (2022) — both un-gated & gated
       - DeCiv + Fixed / risk, DeCiv + Adaptive (0.85) / risk
       - DeCiv + Hard Dawid-Skene / risk
       - DeCiv + Soft Dawid-Skene / risk
       - DeCiv + Soft-DS + TempScale / risk
       - DeCiv + Soft-DS + PG-CRC (alpha=0.01 and alpha=0.005)
  3. Exports exact out-of-fold admission decisions per policy for downstream RAG QA
     and automated revise-queue remediation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deciv_extensions import (
    ACTIONS,
    ACTION_IDX,
    BASE_COST,
    STATUSES,
    STATUS_IDX,
    TIE_ORDER,
    VectorizedDawidSkene,
    apply_temperature,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    eligible_actions,
    fit_temperature,
    route_pg_crc,
    route_risk,
)
from run_comparison import realized_institutional_cost, summarize

DATA_PATH = ROOT / "data" / "deciv_real_1200.jsonl"
VERIFIER_DIR = ROOT / "results" / "real_verifiers"
RESULTS_DIR = ROOT / "results"
VERIFIERS = ("deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b")
VERIFIER_LABELS = {
    "deberta_v3_large": "DeBERTa-v3-Large",
    "bart_large_mnli": "BART-Large-MNLI",
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "qwen35_9b": "Qwen3.5-9B",
}


def text_key(claim: str, evidence: str) -> str:
    return hashlib.sha256((claim + "\n|||\n" + evidence).encode("utf-8")).hexdigest()[:24]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_real_study():
    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8") if l.strip()]
    n = len(rows)
    k = len(VERIFIERS)
    soft_probs = np.zeros((n, k, 4), dtype=np.float64)
    meta = {}
    for ki, vid in enumerate(VERIFIERS):
        d = json.loads((VERIFIER_DIR / f"{vid}_probs.json").read_text(encoding="utf-8"))
        meta[vid] = {key: d[key] for key in d if key != "probs"}
        pmap = d["probs"]
        for i, r in enumerate(rows):
            tk = text_key(r["claim_text"], r["evidence_text"])
            soft_probs[i, ki] = np.asarray(pmap[tk], dtype=np.float64)
    hard_votes = np.argmax(soft_probs, axis=2)  # (N, K)
    y_true = np.array([STATUS_IDX[r["semantic_status"]] for r in rows], dtype=np.int64)
    contexts = [
        {
            "provenance_status": r["provenance_status"],
            "evidence_auditability": r["evidence_auditability"],
            "human_review_required": r["human_review_required"],
            "repairability": r["repairability"],
        }
        for r in rows
    ]
    refs = [r["reference_action"] for r in rows]
    return rows, soft_probs, hard_votes, y_true, contexts, refs, meta


def compute_empirical_correlations(hard_votes: np.ndarray, y_true: np.ndarray, rows: list[dict]):
    """Compute 4x4 error correlation matrix & agreement statistics across real neural verifiers."""
    n, k = hard_votes.shape
    err = (hard_votes != y_true[:, None]).astype(np.float64)  # 1 = error, 0 = correct
    corr_rows = []
    pair_corrs = []
    for i in range(k):
        for j in range(k):
            # Pearson correlation of binary error indicators
            ei, ej = err[:, i], err[:, j]
            denom = np.std(ei) * np.std(ej)
            r_err = float(np.corrcoef(ei, ej)[0, 1]) if denom > 1e-12 else 0.0
            # Exact prediction agreement
            agree = float(np.mean(hard_votes[:, i] == hard_votes[:, j]))
            # Joint error overlap: P(both wrong | at least one wrong)
            both_wrong = float(np.mean((ei == 1) & (ej == 1)))
            either_wrong = float(np.mean((ei == 1) | (ej == 1)))
            jaccard_err = both_wrong / max(1e-12, either_wrong)
            # Identical wrong class given both wrong
            same_wrong_cls = float(
                np.mean((ei == 1) & (ej == 1) & (hard_votes[:, i] == hard_votes[:, j]))
                / max(1e-12, both_wrong)
            )
            corr_rows.append(
                {
                    "verifier_i": VERIFIERS[i],
                    "label_i": VERIFIER_LABELS[VERIFIERS[i]],
                    "verifier_j": VERIFIERS[j],
                    "label_j": VERIFIER_LABELS[VERIFIERS[j]],
                    "error_pearson_rho": round(r_err, 4),
                    "prediction_agreement": round(agree, 4),
                    "error_jaccard": round(jaccard_err, 4),
                    "same_wrong_class_given_both_wrong": round(same_wrong_cls, 4),
                }
            )
            if i < j:
                pair_corrs.append(r_err)

    mean_rho = float(np.mean(pair_corrs))
    # Also compute same-family (NLI-NLI, LLM-LLM) vs cross-family (NLI-LLM) correlation
    nli_nli = [r["error_pearson_rho"] for r in corr_rows if {r["verifier_i"], r["verifier_j"]} == {"deberta_v3_large", "bart_large_mnli"}][0]
    llm_llm = [r["error_pearson_rho"] for r in corr_rows if {r["verifier_i"], r["verifier_j"]} == {"llama31_8b", "qwen35_9b"}][0]
    cross_pairs = [
        r["error_pearson_rho"]
        for r in corr_rows
        if r["verifier_i"] in ("deberta_v3_large", "bart_large_mnli")
        and r["verifier_j"] in ("llama31_8b", "qwen35_9b")
    ]
    print(
        f"[Real Error Correlation] Overall mean rho={mean_rho:.3f} | "
        f"NLI-NLI rho={nli_nli:.3f} | LLM-LLM rho={llm_llm:.3f} | "
        f"Cross-family (NLI-LLM) mean rho={np.mean(cross_pairs):.3f}",
        flush=True,
    )
    write_csv(RESULTS_DIR / "real_verifier_correlation.csv", corr_rows)
    return {
        "mean_pairwise_rho": round(mean_rho, 4),
        "nli_nli_rho": round(nli_nli, 4),
        "llm_llm_rho": round(llm_llm, 4),
        "cross_family_mean_rho": round(float(np.mean(cross_pairs)), 4),
    }


def _conformal_q(probs_tr: np.ndarray, y_tr: np.ndarray, alpha: float = 0.10) -> float:
    scores = 1.0 - probs_tr[np.arange(len(y_tr)), y_tr]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _route_direct(p: np.ndarray, ctx: dict[str, str], gated: bool = False) -> tuple[str, bool]:
    mx = float(np.max(p))
    winners = [ACTIONS[j] for j in range(4) if abs(float(p[j]) - mx) <= 1e-12]
    act = winners[0] if len(winners) == 1 else "defer"
    allowed, _ = eligible_actions(ctx)
    if gated and act not in allowed:
        act = "defer"
    return act, act in allowed


def _route_sota(p: np.ndarray, ctx: dict[str, str], rname: str, q_hat: float = 0.2, gated: bool = False) -> tuple[str, bool]:
    p = np.asarray(p, dtype=np.float64)
    top_idx = int(np.argmax(p))
    sorted_p = np.sort(p)
    top_p = float(sorted_p[-1])
    margin = float(sorted_p[-1] - sorted_p[-2])

    if rname == "selective_threshold":
        act = ACTIONS[top_idx] if top_p >= 0.70 else "defer"
    elif rname == "margin_abstention":
        act = ACTIONS[top_idx] if margin >= 0.30 else "defer"
    elif rname == "conformal_risk":
        pred_set = [j for j in range(4) if (1.0 - float(p[j])) <= q_hat + 1e-12]
        act = ACTIONS[pred_set[0]] if len(pred_set) == 1 else "defer"
    elif rname == "posthoc_l2d":
        risks = BASE_COST @ p
        best = float(np.min(risks))
        act = next(a for a in TIE_ORDER if risks[ACTION_IDX[a]] <= best + 1e-12)
    else:
        raise ValueError(rname)

    allowed, _ = eligible_actions(ctx)
    if gated and act not in allowed:
        act = "defer"
    return act, act in allowed


def _fit_and_predict_split(
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    soft_probs: np.ndarray,
    hard_votes: np.ndarray,
    y_true: np.ndarray,
    contexts: list[dict],
    refs: list[str],
    parent_ids: list[str],
) -> dict[str, dict[int, tuple[str, bool, np.ndarray]]]:
    """Fit ALL competing methods strictly on tr_idx and predict on te_idx (zero leakage)."""
    s_tr, s_te = soft_probs[tr_idx], soft_probs[te_idx]
    v_tr, v_te = hard_votes[tr_idx], hard_votes[te_idx]
    y_tr = y_true[tr_idx]
    k_val = s_tr.shape[1]

    # 1. Reliability weights (Laplace-smoothed training accuracy)
    rel = np.array([(np.sum(v_tr[:, k] == y_tr) + 1.0) / (len(y_tr) + 2.0) for k in range(k_val)])

    # 2. Hard Dawid-Skene (unsupervised EM + TempScale)
    ds_hard = VectorizedDawidSkene(smoothing=1e-2).fit_hard(v_tr)
    p_tr_hds = ds_hard.predict_proba_hard(v_tr, apply_temp=False)
    p_te_hds = ds_hard.predict_proba_hard(v_te, apply_temp=False)
    t_hds = fit_temperature(p_tr_hds, y_tr)
    p_te_hds_t = apply_temperature(p_te_hds, t_hds)

    # 3. Soft Dawid-Skene (unsupervised EM on continuous neural probability simplexes!)
    ds_soft = VectorizedDawidSkene(smoothing=1e-2).fit_soft(s_tr)
    p_tr_sds = ds_soft.predict_proba_soft(s_tr, apply_temp=False)
    p_te_sds = ds_soft.predict_proba_soft(s_te, apply_temp=False)
    t_sds = fit_temperature(p_tr_sds, y_tr)
    p_tr_sds_t = apply_temperature(p_tr_sds, t_sds)
    p_te_sds_t = apply_temperature(p_te_sds, t_sds)

    # 4. Supervised Soft Dawid-Skene (closed-form confusion from training labels + TempScale)
    ds_soft_sup = VectorizedDawidSkene(smoothing=1.0).fit_soft(s_tr, y_true=y_tr)
    p_tr_ssup = ds_soft_sup.predict_proba_soft(s_tr, apply_temp=False)
    p_te_ssup = ds_soft_sup.predict_proba_soft(s_te, apply_temp=False)
    t_ssup = fit_temperature(p_tr_ssup, y_tr)
    p_tr_ssup_t = apply_temperature(p_tr_ssup, t_ssup)
    p_te_ssup_t = apply_temperature(p_te_ssup, t_ssup)

    # 5. Soft Linear Pool (mean of continuous neural posteriors + TempScale)
    p_tr_pool = np.mean(s_tr, axis=1)
    p_te_pool = np.mean(s_te, axis=1)
    t_pool = fit_temperature(p_tr_pool, y_tr)
    p_te_pool_t = apply_temperature(p_te_pool, t_pool)

    # 6. Standard Conformal quantile (ICLR 2024 baseline on Soft-DS+Temp)
    q_hat = _conformal_q(p_tr_sds_t, y_tr, alpha=0.10)

    # 7. PG-CRC thresholds calibrated strictly on training parent clusters!
    tr_ctxs = [contexts[i] for i in tr_idx]
    tr_refs = [refs[i] for i in tr_idx]
    tr_pids = [parent_ids[i] for i in tr_idx]
    # Calibrate PG-CRC on Hard-DS (raw), Soft-DS+Temp, and Supervised Soft-DS+Temp
    tau_hds_010, _ = calibrate_pg_crc_threshold(p_tr_hds, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
    tau_sds_010, _ = calibrate_pg_crc_threshold(p_tr_sds_t, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
    tau_sds_005, _ = calibrate_pg_crc_threshold(p_tr_sds_t, tr_ctxs, tr_refs, tr_pids, alpha=0.005)
    tau_ssup_010, _ = calibrate_pg_crc_threshold(p_tr_ssup_t, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
    tau_ssup_005, _ = calibrate_pg_crc_threshold(p_tr_ssup_t, tr_ctxs, tr_refs, tr_pids, alpha=0.005)

    out: dict[str, dict[int, tuple[str, bool, np.ndarray]]] = defaultdict(dict)
    for loc_i, gi in enumerate(te_idx):
        ctx = contexts[gi]
        v = v_te[loc_i]
        # Hard vote frequency (Majority / Fixed)
        p_fix = np.bincount(v, minlength=4).astype(np.float64) / k_val
        # Adaptive (lambda = 0.85)
        lam = 0.85
        w = np.array(
            [rel[k] * ((1.0 - lam) + lam * (np.sum(v == v[k]) - 1) / max(1, k_val - 1)) for k in range(k_val)],
            dtype=np.float64,
        )
        p_adp = np.array([np.sum(w[v == s]) for s in range(4)], dtype=np.float64) / np.sum(w)

        # Solo verifiers (direct un-gated AND DeCiv risk-routed)
        for ki, vid in enumerate(VERIFIERS):
            pk = s_te[loc_i, ki]
            out[f"solo_{vid}_direct"][gi] = (*_route_direct(pk, ctx, gated=False), pk)
            out[f"solo_{vid}_deciv"][gi] = (*route_risk(pk, ctx), pk)

        # Un-gated SOTA baselines (using calibrated Soft-DS probabilities for strongest possible SOTA!)
        p_sota = p_te_sds_t[loc_i]
        out["majority_direct"][gi] = (*_route_direct(p_fix, ctx, gated=False), p_fix)
        out["soft_pool_direct"][gi] = (*_route_direct(p_te_pool[loc_i], ctx, gated=False), p_te_pool[loc_i])
        out["selective_threshold"][gi] = (*_route_sota(p_sota, ctx, "selective_threshold", gated=False), p_sota)
        out["margin_abstention"][gi] = (*_route_sota(p_sota, ctx, "margin_abstention", gated=False), p_sota)
        out["conformal_risk"][gi] = (*_route_sota(p_sota, ctx, "conformal_risk", q_hat=q_hat, gated=False), p_sota)
        out["posthoc_l2d"][gi] = (*_route_sota(p_sota, ctx, "posthoc_l2d", gated=False), p_sota)

        # Provenance-gated SOTA controls
        out["selective_threshold_gated"][gi] = (*_route_sota(p_sota, ctx, "selective_threshold", gated=True), p_sota)
        out["conformal_risk_gated"][gi] = (*_route_sota(p_sota, ctx, "conformal_risk", q_hat=q_hat, gated=True), p_sota)

        # DeCiv family (hard admissibility + asymmetric expected risk + calibrated fusion)
        out["deciv_fixed"][gi] = (*route_risk(p_fix, ctx), p_fix)
        out["deciv_adaptive_085"][gi] = (*route_risk(p_adp, ctx), p_adp)
        out["deciv_soft_pool_temp"][gi] = (*route_risk(p_te_pool_t[loc_i], ctx), p_te_pool_t[loc_i])
        out["deciv_hard_ds_raw"][gi] = (*route_risk(p_te_hds[loc_i], ctx), p_te_hds[loc_i])
        out["deciv_hard_ds_temp"][gi] = (*route_risk(p_te_hds_t[loc_i], ctx), p_te_hds_t[loc_i])
        out["deciv_soft_ds_raw"][gi] = (*route_risk(p_te_sds[loc_i], ctx), p_te_sds[loc_i])
        out["deciv_soft_ds_temp"][gi] = (*route_risk(p_te_sds_t[loc_i], ctx), p_te_sds_t[loc_i])
        out["deciv_soft_ds_sup_temp"][gi] = (*route_risk(p_te_ssup_t[loc_i], ctx), p_te_ssup_t[loc_i])
        out["deciv_hard_ds_pg_crc_010"][gi] = (*route_pg_crc(p_te_hds[loc_i], ctx, tau_hds_010), p_te_hds[loc_i])
        out["deciv_pg_crc_010"][gi] = (*route_pg_crc(p_te_sds_t[loc_i], ctx, tau_sds_010), p_te_sds_t[loc_i])
        out["deciv_pg_crc_005"][gi] = (*route_pg_crc(p_te_sds_t[loc_i], ctx, tau_sds_005), p_te_sds_t[loc_i])
        out["deciv_sup_pg_crc_010"][gi] = (*route_pg_crc(p_te_ssup_t[loc_i], ctx, tau_ssup_010), p_te_ssup_t[loc_i])
        out["deciv_sup_pg_crc_005"][gi] = (*route_pg_crc(p_te_ssup_t[loc_i], ctx, tau_ssup_005), p_te_ssup_t[loc_i])

    return out


def _score_method(preds_map: dict[int, tuple[str, bool, np.ndarray]], rows: list[dict], y_true: np.ndarray) -> dict:
    eval_rows = []
    probs = np.zeros((len(rows), 4), dtype=np.float64)
    y_sub = np.zeros(len(rows), dtype=np.int64)
    for loc_i, r in enumerate(rows):
        gi = r["_idx"]
        act, adm, p = preds_map[gi]
        probs[loc_i] = p
        y_sub[loc_i] = y_true[gi]
        ref = r["reference_action"]
        cost = realized_institutional_cost(act, ref, r["semantic_status"], bool(adm))
        eval_rows.append(
            {
                "predicted_action": act,
                "reference_action": ref,
                "correct": int(act == ref),
                "false_accept": int(act == "accept" and ref != "accept"),
                "eligible": int(adm),
                "institutional_cost": cost,
            }
        )
    s = summarize(eval_rows)
    s.update(calibration_metrics(probs, y_sub))
    return s


def main():
    rows, soft_probs, hard_votes, y_true, contexts, refs, meta = load_real_study()
    for i, r in enumerate(rows):
        r["_idx"] = i
    parent_ids = [r["parent_id"] for r in rows]
    sources = sorted({r["source_dataset"] for r in rows})

    corr_summary = compute_empirical_correlations(hard_votes, y_true, rows)

    # 1. 4-fold Leave-One-Domain-Out CV
    preds_4f: dict[str, dict[int, tuple[str, bool, np.ndarray]]] = defaultdict(dict)
    for src in sources:
        tr_idx = np.array([i for i, r in enumerate(rows) if r["source_dataset"] != src], dtype=np.int64)
        te_idx = np.array([i for i, r in enumerate(rows) if r["source_dataset"] == src], dtype=np.int64)
        assert not ({parent_ids[i] for i in tr_idx} & {parent_ids[i] for i in te_idx})
        fold_out = _fit_and_predict_split(tr_idx, te_idx, soft_probs, hard_votes, y_true, contexts, refs, parent_ids)
        for m, d in fold_out.items():
            preds_4f[m].update(d)

    # 2. 5-fold stratified parent-cluster CV
    rng = np.random.default_rng(20261025)
    by_src = defaultdict(list)
    for r in rows:
        if r["parent_id"] not in by_src[r["source_dataset"]]:
            by_src[r["source_dataset"]].append(r["parent_id"])
    parent_fold = {}
    for src, plist in sorted(by_src.items()):
        plist = sorted(plist)
        perm = rng.permutation(len(plist))
        for pos, pidx in enumerate(perm):
            parent_fold[plist[pidx]] = pos % 5

    cv5_scores: dict[str, list[dict]] = defaultdict(list)
    preds_5f_full: dict[str, dict[int, tuple[str, bool, np.ndarray]]] = defaultdict(dict)
    for fid in range(5):
        tr_idx = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] != fid], dtype=np.int64)
        te_idx = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] == fid], dtype=np.int64)
        fold_out = _fit_and_predict_split(tr_idx, te_idx, soft_probs, hard_votes, y_true, contexts, refs, parent_ids)
        te_rows = [rows[i] for i in te_idx]
        for m, d in fold_out.items():
            cv5_scores[m].append(_score_method(d, te_rows, y_true))
            preds_5f_full[m].update(d)

    ordered_methods = [
        ("solo_deberta_v3_large_direct", "Solo DeBERTa-v3-Large / direct"),
        ("solo_bart_large_mnli_direct", "Solo BART-Large-MNLI / direct"),
        ("solo_llama31_8b_direct", "Solo Llama-3.1-8B-Instruct / direct"),
        ("solo_qwen35_9b_direct", "Solo Qwen3.5-9B / direct"),
        ("majority_direct", "Majority Vote / direct"),
        ("soft_pool_direct", "Soft Probability Pool / direct"),
        ("selective_threshold", "Selective-Threshold (2023)"),
        ("margin_abstention", "Margin-Abstention (2024)"),
        ("conformal_risk", "Conformal-Risk (ICLR 2024)"),
        ("posthoc_l2d", "Post-Hoc Cost L2D (2022)"),
        ("selective_threshold_gated", "Selective-Threshold (2023) + Gate"),
        ("conformal_risk_gated", "Conformal-Risk (2024) + Gate"),
        ("solo_qwen35_9b_deciv", "Solo Qwen3.5-9B + DeCiv / risk"),
        ("solo_llama31_8b_deciv", "Solo Llama-3.1-8B + DeCiv / risk"),
        ("deciv_fixed", "DeCiv + Fixed / risk"),
        ("deciv_adaptive_085", "DeCiv + Adaptive (0.85) / risk"),
        ("deciv_soft_pool_temp", "DeCiv + Soft-Pool + TempScale / risk"),
        ("deciv_hard_ds_raw", "DeCiv + Hard-DS (raw) / risk"),
        ("deciv_hard_ds_pg_crc_010", "DeCiv + Hard-DS + PG-CRC (alpha=0.01)"),
        ("deciv_hard_ds_temp", "DeCiv + Hard-DS + TempScale / risk"),
        ("deciv_soft_ds_raw", "DeCiv + Soft-DS (raw) / risk"),
        ("deciv_soft_ds_temp", "DeCiv + Soft-DS + TempScale / risk"),
        ("deciv_soft_ds_sup_temp", "DeCiv + Sup-Soft-DS + TempScale / risk"),
        ("deciv_pg_crc_010", "DeCiv + Soft-DS + PG-CRC (alpha=0.01)"),
        ("deciv_pg_crc_005", "DeCiv + Soft-DS + PG-CRC (alpha=0.005)"),
        ("deciv_sup_pg_crc_010", "DeCiv + Sup-Soft-DS + PG-CRC (alpha=0.01)"),
        ("deciv_sup_pg_crc_005", "DeCiv + Sup-Soft-DS + PG-CRC (alpha=0.005)"),
    ]

    main_out = []
    by_dom_out = []
    print("\n=== REAL 4-MODEL HETEROGENEOUS VERIFIER RESULTS (4-Fold LODO & 5-Fold Stratified CV) ===")
    print(f"{'Method':42s} | {'Acc':6s} | {'FAR':6s} | {'Prec':6s} | {'EV':3s} | {'Cost':6s} | {'ECE':6s} | {'5F-Acc':6s} | {'5F-FAR':6s}")
    print("-" * 108)
    for mid, label in ordered_methods:
        s4 = _score_method(preds_4f[mid], rows, y_true)
        f5 = cv5_scores[mid]
        row_rec = {
            "method_id": mid,
            "label": label,
            **s4,
            "cv5_accuracy_mean": float(np.mean([m["accuracy"] for m in f5])),
            "cv5_accuracy_std": float(np.std([m["accuracy"] for m in f5], ddof=1)),
            "cv5_far_mean": float(np.mean([m["far"] for m in f5])),
            "cv5_far_std": float(np.std([m["far"] for m in f5], ddof=1)),
            "cv5_cost_mean": float(np.mean([m["mean_institutional_cost"] for m in f5])),
            "cv5_cost_std": float(np.std([m["mean_institutional_cost"] for m in f5], ddof=1)),
        }
        main_out.append(row_rec)
        prec_str = f"{s4['accept_precision']*100:5.1f}%" if s4["accept_precision"] is not None else "  N/A "
        print(
            f"{label:42s} | {s4['accuracy']*100:5.2f}% | {s4['far']*100:5.2f}% | {prec_str} | "
            f"{int(s4['eligibility_violations']):3d} | {s4['mean_institutional_cost']:6.3f} | "
            f"{s4['ece']:6.4f} | {row_rec['cv5_accuracy_mean']*100:5.2f}% | {row_rec['cv5_far_mean']*100:5.2f}%"
        )
        for src in sources:
            sub_rows = [r for r in rows if r["source_dataset"] == src]
            sd = _score_method(preds_4f[mid], sub_rows, y_true)
            by_dom_out.append({"method_id": mid, "label": label, "source": src, **sd})

    write_csv(RESULTS_DIR / "real_admission_metrics.csv", main_out)
    write_csv(RESULTS_DIR / "real_by_domain.csv", by_dom_out)

    # Export exact out-of-fold 4-fold LODO decisions for downstream RAG & Automated Remediation!
    export_policies = [
        "majority_direct",
        "conformal_risk",
        "selective_threshold",
        "posthoc_l2d",
        "deciv_hard_ds_raw",
        "deciv_hard_ds_pg_crc_010",
        "deciv_soft_ds_raw",
        "deciv_soft_ds_temp",
        "deciv_soft_ds_sup_temp",
        "deciv_pg_crc_010",
        "deciv_sup_pg_crc_010",
        "deciv_sup_pg_crc_005",
    ]
    decisions_export = {
        "meta": {"verifiers": meta, "correlation_summary": corr_summary},
        "lodo_decisions": {
            pol: {rows[i]["packet_id"]: preds_4f[pol][i][0] for i in range(len(rows))}
            for pol in export_policies
        },
        "cv5_decisions": {
            pol: {rows[i]["packet_id"]: preds_5f_full[pol][i][0] for i in range(len(rows))}
            for pol in export_policies
        },
    }
    (RESULTS_DIR / "real_admitted_decisions.json").write_text(
        json.dumps(decisions_export, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSaved real_admission_metrics.csv, real_by_domain.csv, real_verifier_correlation.csv, real_admitted_decisions.json", flush=True)


if __name__ == "__main__":
    main()
