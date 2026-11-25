"""Evaluate 5 genuinely heterogeneous verifiers across 3 model families on DeCiv-Real-1200 (zero-leakage CV).

Three-family verifier pool (K = 5):
  Family 1 (NLI Encoders, local RTX 3090):
    1. deberta_v3_large : MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (435M)
    2. bart_large_mnli  : facebook/bart-large-mnli (406M)
  Family 2 (Open-Weight Decoder LLMs, local RTX 3090):
    3. llama31_8b       : Meta-Llama-3.1-8B-Instruct (8B)
    4. qwen35_9b        : Qwen3.5-9B (9B)
  Family 3 (Frontier Closed-Weight Reasoning LLM, API):
    5. gpt61_sol        : GPT-6.1-Sol (low-effort reasoning)

Computes:
  1. Full 5x5 empirical error correlation matrix (intra-family vs cross-family).
  2. Strict 4-fold Leave-One-Domain-Out + 5-fold stratified parent-cluster CV comparing:
       - All 5 solo verifiers (direct un-gated + Frontier GPT-6.1-Sol + Provenance Gate)
       - Majority Vote, Deep Ensemble (Soft Pool + TempScale), Selective-Threshold (JMLR 2023),
         Margin-Abstention (ALT 2024), Conformal-Risk (ICLR 2024), Post-Hoc Cost L2D (NeurIPS 2022),
         Supervised GBDT Stacking — both un-gated & provenance-gated retrofits
       - DeCiv Dawid-Skene baselines (Hard-DS, Sup-Soft-DS + TempScale, PG-CRC)
       - DeCiv-CAF (Correlation-Aware Calibrated Epistemic Fusion + Hard Gate + Asymmetric Risk /
         Decoupled PG-CRC at alpha=0.01 and alpha=0.005)
  3. Exports exact out-of-fold LODO and 5-fold admission decisions for downstream RAG defense
     (comparing against ReliabilityRAG, SeCon-RAG, and Self-RAG) and automated revise remediation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deciv_extensions import (
    ACTIONS,
    ACTION_IDX,
    BASE_COST,
    STATUSES,
    STATUS_IDX,
    TIE_ORDER,
    CorrelationAwareFusion,
    VectorizedDawidSkene,
    apply_temperature,
    build_caf_epistemic_features,
    calibrate_decoupled_pg_crc_threshold,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    eligible_actions,
    fit_temperature,
    route_decoupled_pg_crc,
    route_pg_crc,
    route_projected_pg_crc,
    route_risk,
)
from run_comparison import realized_institutional_cost, summarize

DATA_PATH = ROOT / "data" / "deciv_real_1200.jsonl"
VERIFIER_DIR = ROOT / "results" / "real_verifiers"
RESULTS_DIR = ROOT / "results"
VERIFIERS = ("deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b", "gpt61_sol")
VERIFIER_LABELS = {
    "deberta_v3_large": "DeBERTa-v3-Large",
    "bart_large_mnli": "BART-Large-MNLI",
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "qwen35_9b": "Qwen3.5-9B",
    "gpt61_sol": "GPT-6.1-Sol",
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


def load_real_study(verifiers: tuple[str, ...] = VERIFIERS):
    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8") if l.strip()]
    n = len(rows)
    k = len(verifiers)
    soft_probs = np.zeros((n, k, 4), dtype=np.float64)
    meta = {}
    for ki, vid in enumerate(verifiers):
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
    """Compute 5x5 error correlation matrix & intra- vs cross-family statistics."""
    n, k = hard_votes.shape
    err = (hard_votes != y_true[:, None]).astype(np.float64)
    corr_rows = []
    pair_corrs = []
    for i in range(k):
        for j in range(k):
            ei, ej = err[:, i], err[:, j]
            denom = np.std(ei) * np.std(ej)
            r_err = float(np.corrcoef(ei, ej)[0, 1]) if denom > 1e-12 else 0.0
            agree = float(np.mean(hard_votes[:, i] == hard_votes[:, j]))
            both_wrong = float(np.mean((ei == 1) & (ej == 1)))
            either_wrong = float(np.mean((ei == 1) | (ej == 1)))
            jaccard_err = both_wrong / max(1e-12, either_wrong)
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
    lookup = {(r["verifier_i"], r["verifier_j"]): r["error_pearson_rho"] for r in corr_rows}
    nli_nli = lookup[("deberta_v3_large", "bart_large_mnli")]
    llm_llm = lookup[("llama31_8b", "qwen35_9b")]
    nli_openllm = float(
        np.mean([lookup[(a, b)] for a in ("deberta_v3_large", "bart_large_mnli") for b in ("llama31_8b", "qwen35_9b")])
    )
    nli_frontier = float(np.mean([lookup[(a, "gpt61_sol")] for a in ("deberta_v3_large", "bart_large_mnli")]))
    openllm_frontier = float(np.mean([lookup[(a, "gpt61_sol")] for a in ("llama31_8b", "qwen35_9b")]))
    all_cross = float(
        np.mean(
            [lookup[(a, b)] for a in ("deberta_v3_large", "bart_large_mnli") for b in ("llama31_8b", "qwen35_9b", "gpt61_sol")]
            + [lookup[(a, "gpt61_sol")] for a in ("llama31_8b", "qwen35_9b")]
        )
    )
    print(
        f"[Real 5-Model Correlation] overall={mean_rho:.3f} | NLI-NLI={nli_nli:.3f} | OpenLLM-OpenLLM={llm_llm:.3f} | "
        f"NLI-OpenLLM={nli_openllm:.3f} | NLI-Frontier={nli_frontier:.3f} | OpenLLM-Frontier={openllm_frontier:.3f}",
        flush=True,
    )
    write_csv(RESULTS_DIR / "real_verifier_correlation.csv", corr_rows)
    return {
        "mean_pairwise_rho": round(mean_rho, 4),
        "nli_nli_rho": round(nli_nli, 4),
        "llm_llm_rho": round(llm_llm, 4),
        "nli_openllm_rho": round(nli_openllm, 4),
        "nli_frontier_rho": round(nli_frontier, 4),
        "openllm_frontier_rho": round(openllm_frontier, 4),
        "cross_family_mean_rho": round(all_cross, 4),
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
    group_ids_for_inner: list[str],
) -> dict[str, dict[int, tuple[str, bool, np.ndarray]]]:
    """Fit ALL competing methods strictly on tr_idx and predict on te_idx (zero leakage)."""
    s_tr, s_te = soft_probs[tr_idx], soft_probs[te_idx]
    v_tr, v_te = hard_votes[tr_idx], hard_votes[te_idx]
    y_tr = y_true[tr_idx]
    k_val = s_tr.shape[1]
    tr_ctxs = [contexts[i] for i in tr_idx]
    tr_refs = [refs[i] for i in tr_idx]
    tr_pids = [parent_ids[i] for i in tr_idx]
    tr_grps = [group_ids_for_inner[i] for i in tr_idx]

    # 1. Reliability weights
    rel = np.array([(np.sum(v_tr[:, k] == y_tr) + 1.0) / (len(y_tr) + 2.0) for k in range(k_val)])

    # 2. Deep Ensemble (Soft Linear Pool + TempScale)
    p_tr_pool = np.mean(s_tr, axis=1)
    p_te_pool = np.mean(s_te, axis=1)
    t_pool = fit_temperature(p_tr_pool, y_tr)
    p_tr_pool_t = apply_temperature(p_tr_pool, t_pool)
    p_te_pool_t = apply_temperature(p_te_pool, t_pool)

    # 3. Hard Dawid-Skene (unsupervised EM + TempScale)
    ds_hard = VectorizedDawidSkene(smoothing=1e-2).fit_hard(v_tr)
    p_tr_hds = ds_hard.predict_proba_hard(v_tr, apply_temp=False)
    p_te_hds = ds_hard.predict_proba_hard(v_te, apply_temp=False)
    t_hds = fit_temperature(p_tr_hds, y_tr)
    p_te_hds_t = apply_temperature(p_te_hds, t_hds)

    # 4. Soft Dawid-Skene (unsupervised EM + TempScale)
    ds_soft = VectorizedDawidSkene(smoothing=1e-2).fit_soft(s_tr)
    p_tr_sds = ds_soft.predict_proba_soft(s_tr, apply_temp=False)
    p_te_sds = ds_soft.predict_proba_soft(s_te, apply_temp=False)
    t_sds = fit_temperature(p_tr_sds, y_tr)
    p_tr_sds_t = apply_temperature(p_tr_sds, t_sds)
    p_te_sds_t = apply_temperature(p_te_sds, t_sds)

    # 5. Supervised Soft Dawid-Skene (closed-form confusion + TempScale)
    ds_soft_sup = VectorizedDawidSkene(smoothing=1.0).fit_soft(s_tr, y_true=y_tr)
    p_tr_ssup = ds_soft_sup.predict_proba_soft(s_tr, apply_temp=False)
    p_te_ssup = ds_soft_sup.predict_proba_soft(s_te, apply_temp=False)
    t_ssup = fit_temperature(p_tr_ssup, y_tr)
    p_tr_ssup_t = apply_temperature(p_tr_ssup, t_ssup)
    p_te_ssup_t = apply_temperature(p_te_ssup, t_ssup)

    # 6. Supervised GBDT Stacking baseline (strong nonlinear discriminative ML baseline)
    X_gbdt_tr = s_tr.reshape(len(tr_idx), -1)
    X_gbdt_te = s_te.reshape(len(te_idx), -1)
    gbdt = HistGradientBoostingClassifier(
        max_depth=3, max_iter=80, learning_rate=0.05, l2_regularization=5.0, random_state=42
    ).fit(X_gbdt_tr, y_tr)
    p_te_gbdt = gbdt.predict_proba(X_gbdt_te)

    # 7. DeCiv-CAF: Correlation-Aware Calibrated Epistemic Fusion (nested OOF calibration -> zero leakage!)
    caf = CorrelationAwareFusion(reg_c=0.10)
    p_tr_caf = caf.fit_with_inner_oof(s_tr, y_tr, tr_grps)
    p_te_caf = caf.predict_proba(s_te, apply_temp=True)

    # 8. Conformal quantile (ICLR 2024 baseline on strong calibrated Deep Ensemble probabilities)
    q_hat_pool = _conformal_q(p_tr_pool_t, y_tr, alpha=0.10)

    # 9. PG-CRC thresholds calibrated strictly on training parent clusters!
    tau_hds_010, _ = calibrate_pg_crc_threshold(p_tr_hds, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
    tau_ssup_005, _ = calibrate_pg_crc_threshold(p_tr_ssup_t, tr_ctxs, tr_refs, tr_pids, alpha=0.005)
    tau_caf_dec_015, _ = calibrate_decoupled_pg_crc_threshold(p_tr_caf, tr_ctxs, tr_refs, tr_pids, alpha=0.015)
    tau_caf_dec_010, _ = calibrate_decoupled_pg_crc_threshold(p_tr_caf, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
    tau_caf_dec_005, _ = calibrate_decoupled_pg_crc_threshold(p_tr_caf, tr_ctxs, tr_refs, tr_pids, alpha=0.005)
    tau_caf_risk_005, _ = calibrate_pg_crc_threshold(p_tr_caf, tr_ctxs, tr_refs, tr_pids, alpha=0.005, defer_cost=0.5)

    out: dict[str, dict[int, tuple[str, bool, np.ndarray]]] = defaultdict(dict)
    for loc_i, gi in enumerate(te_idx):
        ctx = contexts[gi]
        v = v_te[loc_i]
        p_fix = np.bincount(v, minlength=4).astype(np.float64) / k_val
        lam = 0.85
        w = np.array(
            [rel[k] * ((1.0 - lam) + lam * (np.sum(v == v[k]) - 1) / max(1, k_val - 1)) for k in range(k_val)],
            dtype=np.float64,
        )
        p_adp = np.array([np.sum(w[v == s]) for s in range(4)], dtype=np.float64) / np.sum(w)

        # Solo verifiers (direct un-gated + gated for Frontier GPT-6.1-Sol)
        for ki, vid in enumerate(VERIFIERS):
            pk = s_te[loc_i, ki]
            out[f"solo_{vid}_direct"][gi] = (*_route_direct(pk, ctx, gated=False), pk)
            out[f"solo_{vid}_gated"][gi] = (*_route_direct(pk, ctx, gated=True), pk)

        # Un-gated SOTA baselines (given strong calibrated Deep Ensemble probabilities!)
        p_sota = p_te_pool_t[loc_i]
        out["majority_direct"][gi] = (*_route_direct(p_fix, ctx, gated=False), p_fix)
        out["deep_ensemble_direct"][gi] = (*_route_direct(p_sota, ctx, gated=False), p_sota)
        out["deep_ensemble_gated"][gi] = (*_route_direct(p_sota, ctx, gated=True), p_sota)
        out["selective_threshold"][gi] = (*_route_sota(p_sota, ctx, "selective_threshold", gated=False), p_sota)
        out["margin_abstention"][gi] = (*_route_sota(p_sota, ctx, "margin_abstention", gated=False), p_sota)
        out["conformal_risk"][gi] = (*_route_sota(p_sota, ctx, "conformal_risk", q_hat=q_hat_pool, gated=False), p_sota)
        out["posthoc_l2d"][gi] = (*_route_sota(p_sota, ctx, "posthoc_l2d", gated=False), p_sota)
        out["gbdt_stack_direct"][gi] = (*_route_direct(p_te_gbdt[loc_i], ctx, gated=False), p_te_gbdt[loc_i])

        # Provenance-gated SOTA retrofits
        out["selective_threshold_gated"][gi] = (*_route_sota(p_sota, ctx, "selective_threshold", gated=True), p_sota)
        out["margin_abstention_gated"][gi] = (*_route_sota(p_sota, ctx, "margin_abstention", gated=True), p_sota)
        out["conformal_risk_gated"][gi] = (*_route_sota(p_sota, ctx, "conformal_risk", q_hat=q_hat_pool, gated=True), p_sota)
        out["posthoc_l2d_gated"][gi] = (*_route_sota(p_sota, ctx, "posthoc_l2d", gated=True), p_sota)
        out["gbdt_stack_gated"][gi] = (*_route_direct(p_te_gbdt[loc_i], ctx, gated=True), p_te_gbdt[loc_i])

        # DeCiv Dawid-Skene baselines
        out["deciv_hard_ds_raw"][gi] = (*route_risk(p_te_hds[loc_i], ctx), p_te_hds[loc_i])
        out["deciv_hard_ds_pg_crc_010"][gi] = (*route_pg_crc(p_te_hds[loc_i], ctx, tau_hds_010), p_te_hds[loc_i])
        out["deciv_soft_ds_raw"][gi] = (*route_risk(p_te_sds[loc_i], ctx), p_te_sds[loc_i])
        out["deciv_soft_ds_sup_temp"][gi] = (*route_risk(p_te_ssup_t[loc_i], ctx), p_te_ssup_t[loc_i])
        out["deciv_sup_pg_crc_005"][gi] = (*route_pg_crc(p_te_ssup_t[loc_i], ctx, tau_ssup_005), p_te_ssup_t[loc_i])

        # DeCiv-CAF (Correlation-Aware Calibrated Fusion + Hard Gate + Projected PG-CRC / Risk)
        pc = p_te_caf[loc_i]
        out["deciv_caf_risk_d05"][gi] = (*route_risk(pc, ctx, defer_cost=0.5), pc)
        out["deciv_caf_gated_map"][gi] = (*route_projected_pg_crc(pc, ctx, tau_accept=0.50), pc)
        out["deciv_caf_pg_crc_015"][gi] = (*route_projected_pg_crc(pc, ctx, tau_caf_dec_015), pc)
        out["deciv_caf_pg_crc_010"][gi] = (*route_projected_pg_crc(pc, ctx, tau_caf_dec_010), pc)
        out["deciv_caf_pg_crc_005"][gi] = (*route_projected_pg_crc(pc, ctx, tau_caf_dec_005), pc)
        out["deciv_caf_pg_crc_005_strict"][gi] = (*route_pg_crc(pc, ctx, tau_caf_risk_005, defer_cost=0.5), pc)

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
    src_per_row = [r["source_dataset"] for r in rows]

    corr_summary = compute_empirical_correlations(hard_votes, y_true, rows)

    # 1. 4-fold Leave-One-Domain-Out CV
    preds_4f: dict[str, dict[int, tuple[str, bool, np.ndarray]]] = defaultdict(dict)
    for src in sources:
        tr_idx = np.array([i for i, r in enumerate(rows) if r["source_dataset"] != src], dtype=np.int64)
        te_idx = np.array([i for i, r in enumerate(rows) if r["source_dataset"] == src], dtype=np.int64)
        assert not ({parent_ids[i] for i in tr_idx} & {parent_ids[i] for i in te_idx})
        fold_out = _fit_and_predict_split(
            tr_idx, te_idx, soft_probs, hard_votes, y_true, contexts, refs, parent_ids, src_per_row
        )
        for m, d in fold_out.items():
            preds_4f[m].update(d)

    # 2. 5-fold stratified parent-cluster CV (exchangeable clusters -> exact Theorem 1 setting!)
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
        fold_out = _fit_and_predict_split(
            tr_idx, te_idx, soft_probs, hard_votes, y_true, contexts, refs, parent_ids, parent_ids
        )
        te_rows = [rows[i] for i in te_idx]
        for m, d in fold_out.items():
            cv5_scores[m].append(_score_method(d, te_rows, y_true))
            preds_5f_full[m].update(d)

    ordered_methods = [
        ("solo_deberta_v3_large_direct", "Solo DeBERTa-v3-Large / direct"),
        ("solo_bart_large_mnli_direct", "Solo BART-Large-MNLI / direct"),
        ("solo_llama31_8b_direct", "Solo Llama-3.1-8B-Instruct / direct"),
        ("solo_qwen35_9b_direct", "Solo Qwen3.5-9B / direct"),
        ("solo_gpt61_sol_direct", "Solo Frontier GPT-6.1-Sol / direct"),
        ("solo_gpt61_sol_gated", "Solo Frontier GPT-6.1-Sol + Gate"),
        ("majority_direct", "Majority Vote / direct"),
        ("deep_ensemble_direct", "Deep Ensemble (5-Model) / direct"),
        ("selective_threshold", "Selective-Threshold (JMLR 2023)"),
        ("margin_abstention", "Margin-Abstention (ALT 2024)"),
        ("conformal_risk", "Conformal-Risk (ICLR 2024)"),
        ("posthoc_l2d", "Post-Hoc Cost L2D (NeurIPS 2022)"),
        ("gbdt_stack_direct", "Supervised GBDT Stacking / direct"),
        ("deep_ensemble_gated", "Deep Ensemble (5-Model) + Gate"),
        ("selective_threshold_gated", "Selective-Threshold (2023) + Gate"),
        ("margin_abstention_gated", "Margin-Abstention (2024) + Gate"),
        ("conformal_risk_gated", "Conformal-Risk (ICLR 2024) + Gate"),
        ("posthoc_l2d_gated", "Post-Hoc Cost L2D (2022) + Gate"),
        ("gbdt_stack_gated", "Supervised GBDT Stacking + Gate"),
        ("deciv_hard_ds_raw", "DeCiv + Hard-DS (raw) / risk"),
        ("deciv_hard_ds_pg_crc_010", "DeCiv + Hard-DS + PG-CRC (alpha=0.01)"),
        ("deciv_soft_ds_raw", "DeCiv + Soft-DS (raw) / risk"),
        ("deciv_soft_ds_sup_temp", "DeCiv + Sup-Soft-DS + TempScale / risk"),
        ("deciv_sup_pg_crc_005", "DeCiv + Sup-Soft-DS + PG-CRC (alpha=0.005)"),
        ("deciv_caf_risk_d05", "DeCiv-CAF (Corr-Fusion) / risk (d=0.5)"),
        ("deciv_caf_gated_map", "DeCiv-CAF (Corr-Fusion) / projected MAP"),
        ("deciv_caf_pg_crc_015", "DeCiv-CAF + Projected PG-CRC (alpha=0.015)"),
        ("deciv_caf_pg_crc_010", "DeCiv-CAF + Projected PG-CRC (alpha=0.01)"),
        ("deciv_caf_pg_crc_005", "DeCiv-CAF + Projected PG-CRC (alpha=0.005)"),
        ("deciv_caf_pg_crc_005_strict", "DeCiv-CAF + Strict PG-CRC (alpha=0.005)"),
    ]

    main_out = []
    by_dom_out = []
    print("\n=== REAL 5-MODEL (3-FAMILY) HETEROGENEOUS VERIFIER RESULTS (4-Fold LODO & 5-Fold CV) ===")
    print(f"{'Method':44s} | {'Acc':6s} | {'FAR':6s} | {'Prec':6s} | {'AccR':5s} | {'EV':3s} | {'Cost':6s} | {'ECE':6s} | {'5F-Acc':6s} | {'5F-FAR':6s}")
    print("-" * 120)
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
            f"{label:44s} | {s4['accuracy']*100:5.2f}% | {s4['far']*100:5.2f}% | {prec_str} | "
            f"{s4['recall_accept']*100:4.1f}% | {int(s4['eligibility_violations']):3d} | "
            f"{s4['mean_institutional_cost']:6.3f} | {s4['ece']:6.4f} | "
            f"{row_rec['cv5_accuracy_mean']*100:5.2f}% | {row_rec['cv5_far_mean']*100:5.2f}%"
        )
        for src in sources:
            sub_rows = [r for r in rows if r["source_dataset"] == src]
            sd = _score_method(preds_4f[mid], sub_rows, y_true)
            by_dom_out.append({"method_id": mid, "label": label, "source": src, **sd})

    write_csv(RESULTS_DIR / "real_admission_metrics.csv", main_out)
    write_csv(RESULTS_DIR / "real_by_domain.csv", by_dom_out)

    export_policies = [
        "solo_gpt61_sol_direct",
        "solo_gpt61_sol_gated",
        "majority_direct",
        "conformal_risk",
        "conformal_risk_gated",
        "selective_threshold",
        "gbdt_stack_gated",
        "deciv_hard_ds_raw",
        "deciv_soft_ds_sup_temp",
        "deciv_caf_risk_d05",
        "deciv_caf_gated_map",
        "deciv_caf_pg_crc_015",
        "deciv_caf_pg_crc_010",
        "deciv_caf_pg_crc_005",
        "deciv_caf_pg_crc_005_strict",
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
    print("\nSaved 5-model real_admission_metrics.csv, real_by_domain.csv, real_verifier_correlation.csv, real_admitted_decisions.json", flush=True)


if __name__ == "__main__":
    main()
