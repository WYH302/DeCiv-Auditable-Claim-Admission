"""Plan 1 surgical upgrades: calibration, PG-CRC, correlated-error stress, noisy gate, and Merkle log.

Executes deterministically on CPU using the exact DeCiv-1800 benchmark splits
with zero parent-cluster leakage. Saves all outputs to experiments/results/.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from deciv2.r12.semantic import SemanticDawidSkeneEstimator
from deciv_extensions import (
    ACTIONS,
    BASE_COST,
    STATUSES,
    STATUS_IDX,
    MerkleTransparencyLog,
    VectorizedDawidSkene,
    apply_temperature,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    eligible_actions,
    fit_temperature,
    route_pg_crc,
    route_risk,
)
from run_comparison import (
    SPEC,
    _glad_fit_and_predict,
    aggregate,
    digest,
    load_study,
    realized_institutional_cost,
    reference_action,
    route,
    summarize,
    write_csv,
)


def _extract_vote_matrix(rows, inputs):
    vids = sorted({v.validator_id for v in inputs[rows[0]["audit_join_id"]].semantic_votes})
    vid_idx = {v: i for i, v in enumerate(vids)}
    mat = np.zeros((len(rows), len(vids)), dtype=np.int64)
    for i, r in enumerate(rows):
        for v in inputs[r["audit_join_id"]].semantic_votes:
            mat[i, vid_idx[v.validator_id]] = STATUS_IDX[v.status]
    return mat, vids


def _evaluate_predictions(index, targets, contexts, post_dict, router_fn_factory):
    eval_rows = []
    probs_all = np.zeros((len(index), 4), dtype=np.float64)
    y_all = np.zeros(len(index), dtype=np.int64)
    for i, row in enumerate(index):
        key = row["audit_join_id"]
        p = np.asarray(post_dict[key], dtype=np.float64)
        probs_all[i] = p
        y_all[i] = STATUS_IDX[targets[key]]
        act, adm = router_fn_factory(row, p, contexts[key])
        ref = reference_action(targets[key], contexts[key])
        cost = realized_institutional_cost(act, ref, targets[key], bool(adm))
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
    cal = calibration_metrics(probs_all, y_all)
    s.update(cal)
    return s


def run_calibration_and_pgcrc(index, targets, contexts, inputs):
    """Benchmark Stage I calibration, DS-MAP, DS-Supervised-MLE, DS+TempScale, Adaptive(0.85), and PG-CRC."""
    sources = sorted({r["source_dataset"] for r in index})

    # Verify exact numerical parity (<1e-15) between scalar DS and VectorizedDawidSkene
    tr0 = [r for r in index if r["source_dataset"] != sources[0]]
    te0 = [r for r in index if r["source_dataset"] == sources[0]]
    ds_scalar = SemanticDawidSkeneEstimator().fit(tuple(inputs[r["audit_join_id"]] for r in tr0))
    scalar_probs = np.array(
        [[o.probability(s) for s in STATUSES] for o in ds_scalar.predict_proba(tuple(inputs[r["audit_join_id"]] for r in te0))]
    )
    v_tr0, _ = _extract_vote_matrix(tr0, inputs)
    v_te0, _ = _extract_vote_matrix(te0, inputs)
    ds_vec = VectorizedDawidSkene(smoothing=1e-3, max_iter=200, tol=1e-8).fit_hard(v_tr0)
    vec_probs = ds_vec.predict_proba_hard(v_te0, apply_temp=False)
    max_parity_diff = float(np.max(np.abs(scalar_probs - vec_probs)))
    assert max_parity_diff < 1e-12, f"Vectorized DS parity failed: {max_parity_diff}"
    print(f"[Parity Check] Scalar vs Vectorized DS max diff = {max_parity_diff:.3e}", flush=True)

    # 4-fold Leave-One-Source-Out predictions
    methods = [
        "fixed",
        "adaptive_050",
        "adaptive_085",
        "glad",
        "ds_raw",
        "ds_map_5",
        "ds_sup_mle",
        "ds_temp",
        "bayes_oracle",
    ]
    preds_4f = {m: {} for m in methods}
    pgcrc_taus_010 = {}
    pgcrc_taus_005 = {}
    temp_by_fold = {}

    for src in sources:
        tr = [r for r in index if r["source_dataset"] != src]
        te = [r for r in index if r["source_dataset"] == src]
        v_tr, _ = _extract_vote_matrix(tr, inputs)
        v_te, _ = _extract_vote_matrix(te, inputs)
        y_tr = np.array([STATUS_IDX[targets[r["audit_join_id"]]] for r in tr], dtype=np.int64)

        # Reliability for adaptive
        succ, tot = Counter(), Counter()
        for r in tr:
            for v in inputs[r["audit_join_id"]].semantic_votes:
                tot[v.validator_id] += 1
                succ[v.validator_id] += int(v.status == targets[r["audit_join_id"]])
        rel = {vid: (succ[vid] + 1) / (tot[vid] + 2) for vid in sorted(tot)}

        # Fit estimators strictly on training fold
        ds_raw = VectorizedDawidSkene(smoothing=1e-3).fit_hard(v_tr)
        p_tr_raw = ds_raw.predict_proba_hard(v_tr, apply_temp=False)
        p_te_raw = ds_raw.predict_proba_hard(v_te, apply_temp=False)

        t_opt = fit_temperature(p_tr_raw, y_tr)
        temp_by_fold[src] = t_opt
        p_tr_temp = apply_temperature(p_tr_raw, t_opt)
        p_te_temp = apply_temperature(p_te_raw, t_opt)

        ds_map5 = VectorizedDawidSkene(smoothing=5.0).fit_hard(v_tr)
        p_te_map5 = ds_map5.predict_proba_hard(v_te, apply_temp=False)

        ds_sup = VectorizedDawidSkene(smoothing=1.0).fit_hard(v_tr, y_true=y_tr)
        p_te_sup = ds_sup.predict_proba_hard(v_te, apply_temp=False)

        glad_te = _glad_fit_and_predict(
            tuple(inputs[r["audit_join_id"]] for r in tr),
            tuple(inputs[r["audit_join_id"]] for r in te),
        )

        # Bayes Oracle using true synthetic channel (0.7 / 0.1) + Laplace-smoothed fold training prior
        prior_tr = (np.bincount(y_tr, minlength=4).astype(np.float64) + 1.0) / (len(y_tr) + 4.0)
        oracle_conf = np.broadcast_to(
            np.where(np.eye(4, dtype=bool), 0.7, 0.1), (4, 4, 4)
        ).copy()
        p_te_oracle, _ = VectorizedDawidSkene._e_step_hard(v_te, prior_tr, oracle_conf)

        # Calibrate PG-CRC threshold on training parent clusters using raw DS posteriors
        tr_ctxs = [contexts[r["audit_join_id"]] for r in tr]
        tr_refs = [reference_action(targets[r["audit_join_id"]], contexts[r["audit_join_id"]]) for r in tr]
        tr_pids = [r["parent_id"] for r in tr]
        tau_010, info_010 = calibrate_pg_crc_threshold(p_tr_raw, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
        tau_005, info_005 = calibrate_pg_crc_threshold(p_tr_raw, tr_ctxs, tr_refs, tr_pids, alpha=0.005)
        pgcrc_taus_010[src] = tau_010
        pgcrc_taus_005[src] = tau_005

        for i, r in enumerate(te):
            k = r["audit_join_id"]
            votes = inputs[k].semantic_votes
            preds_4f["fixed"][k] = aggregate(votes, rel, "fixed")
            preds_4f["adaptive_050"][k] = aggregate(votes, rel, "adaptive", strength=0.50)
            preds_4f["adaptive_085"][k] = aggregate(votes, rel, "adaptive", strength=0.85)
            preds_4f["glad"][k] = glad_te[i]
            preds_4f["ds_raw"][k] = p_te_raw[i].tolist()
            preds_4f["ds_map_5"][k] = p_te_map5[i].tolist()
            preds_4f["ds_sup_mle"][k] = p_te_sup[i].tolist()
            preds_4f["ds_temp"][k] = p_te_temp[i].tolist()
            preds_4f["bayes_oracle"][k] = p_te_oracle[i].tolist()

    # 5-fold stratified parent-cluster CV (exact same seed/split as run_comparison.py)
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    by_src = defaultdict(list)
    for row in index:
        if row["parent_id"] not in by_src[row["source_dataset"]]:
            by_src[row["source_dataset"]].append(row["parent_id"])
    parent_fold = {}
    for src, plist in sorted(by_src.items()):
        plist = sorted(plist)
        perm = rng.permutation(len(plist))
        for idx_pos, p_idx in enumerate(perm):
            parent_fold[plist[p_idx]] = idx_pos % 5

    specs = [
        ("fixed", "DeCiv Fixed / risk", "fixed", "risk"),
        ("adaptive_050", "DeCiv Adaptive (lambda=0.50) / risk", "adaptive_050", "risk"),
        ("adaptive_085", "DeCiv Adaptive (lambda=0.85) / risk", "adaptive_085", "risk"),
        ("glad", "DeCiv + GLAD / risk", "glad", "risk"),
        ("ds_raw", "DeCiv + Dawid-Skene (eps=1e-3) / risk", "ds_raw", "risk"),
        ("ds_map_5", "DeCiv + DS-MAP (eps=5.0) / risk", "ds_map_5", "risk"),
        ("ds_sup_mle", "DeCiv + DS-Supervised-MLE / risk", "ds_sup_mle", "risk"),
        ("ds_temp", "DeCiv + DS + TempScale / risk", "ds_temp", "risk"),
        ("pg_crc_010", "DeCiv + DS + PG-CRC (alpha=0.01)", "ds_raw", "pg_crc_010"),
        ("pg_crc_005", "DeCiv + DS + PG-CRC (alpha=0.005)", "ds_raw", "pg_crc_005"),
        ("bayes_oracle", "Bayes-Oracle (0.7/0.1 + fold prior) / risk", "bayes_oracle", "risk"),
    ]

    cv5_metrics = defaultdict(list)
    for fold_id in range(5):
        tr = [r for r in index if parent_fold[r["parent_id"]] != fold_id]
        te = [r for r in index if parent_fold[r["parent_id"]] == fold_id]
        v_tr, _ = _extract_vote_matrix(tr, inputs)
        v_te, _ = _extract_vote_matrix(te, inputs)
        y_tr = np.array([STATUS_IDX[targets[r["audit_join_id"]]] for r in tr], dtype=np.int64)

        succ, tot = Counter(), Counter()
        for r in tr:
            for v in inputs[r["audit_join_id"]].semantic_votes:
                tot[v.validator_id] += 1
                succ[v.validator_id] += int(v.status == targets[r["audit_join_id"]])
        rel = {vid: (succ[vid] + 1) / (tot[vid] + 2) for vid in sorted(tot)}

        ds_raw = VectorizedDawidSkene(smoothing=1e-3).fit_hard(v_tr)
        p_tr_raw = ds_raw.predict_proba_hard(v_tr, apply_temp=False)
        p_te_raw = ds_raw.predict_proba_hard(v_te, apply_temp=False)
        t_opt = fit_temperature(p_tr_raw, y_tr)
        p_te_temp = apply_temperature(p_te_raw, t_opt)
        p_te_map5 = VectorizedDawidSkene(smoothing=5.0).fit_hard(v_tr).predict_proba_hard(v_te, apply_temp=False)
        p_te_sup = VectorizedDawidSkene(smoothing=1.0).fit_hard(v_tr, y_true=y_tr).predict_proba_hard(v_te, apply_temp=False)
        glad_te = _glad_fit_and_predict(
            tuple(inputs[r["audit_join_id"]] for r in tr),
            tuple(inputs[r["audit_join_id"]] for r in te),
        )
        prior_tr = (np.bincount(y_tr, minlength=4).astype(np.float64) + 1.0) / (len(y_tr) + 4.0)
        oracle_conf = np.broadcast_to(np.where(np.eye(4, dtype=bool), 0.7, 0.1), (4, 4, 4)).copy()
        p_te_oracle, _ = VectorizedDawidSkene._e_step_hard(v_te, prior_tr, oracle_conf)

        tr_ctxs = [contexts[r["audit_join_id"]] for r in tr]
        tr_refs = [reference_action(targets[r["audit_join_id"]], contexts[r["audit_join_id"]]) for r in tr]
        tr_pids = [r["parent_id"] for r in tr]
        tau_010_f, _ = calibrate_pg_crc_threshold(p_tr_raw, tr_ctxs, tr_refs, tr_pids, alpha=0.01)
        tau_005_f, _ = calibrate_pg_crc_threshold(p_tr_raw, tr_ctxs, tr_refs, tr_pids, alpha=0.005)

        fold_posts = {
            "fixed": {r["audit_join_id"]: aggregate(inputs[r["audit_join_id"]].semantic_votes, rel, "fixed") for r in te},
            "adaptive_050": {r["audit_join_id"]: aggregate(inputs[r["audit_join_id"]].semantic_votes, rel, "adaptive", strength=0.50) for r in te},
            "adaptive_085": {r["audit_join_id"]: aggregate(inputs[r["audit_join_id"]].semantic_votes, rel, "adaptive", strength=0.85) for r in te},
            "glad": {r["audit_join_id"]: glad_te[i] for i, r in enumerate(te)},
            "ds_raw": {r["audit_join_id"]: p_te_raw[i].tolist() for i, r in enumerate(te)},
            "ds_map_5": {r["audit_join_id"]: p_te_map5[i].tolist() for i, r in enumerate(te)},
            "ds_sup_mle": {r["audit_join_id"]: p_te_sup[i].tolist() for i, r in enumerate(te)},
            "ds_temp": {r["audit_join_id"]: p_te_temp[i].tolist() for i, r in enumerate(te)},
            "bayes_oracle": {r["audit_join_id"]: p_te_oracle[i].tolist() for i, r in enumerate(te)},
        }

        for sid, _, mkey, rmode in specs:
            if rmode == "risk":
                rf = lambda r, p, c: route_risk(p, c)
            elif rmode == "pg_crc_010":
                rf = lambda r, p, c, t=tau_010_f: route_pg_crc(p, c, t)
            else:
                rf = lambda r, p, c, t=tau_005_f: route_pg_crc(p, c, t)
            cv5_metrics[sid].append(_evaluate_predictions(te, targets, contexts, fold_posts[mkey], rf))

    out_rows = []
    for sid, label, mkey, rmode in specs:
        if rmode == "risk":
            rf = lambda r, p, c: route_risk(p, c)
        elif rmode == "pg_crc_010":
            rf = lambda r, p, c: route_pg_crc(p, c, pgcrc_taus_010[r["source_dataset"]])
        else:
            rf = lambda r, p, c: route_pg_crc(p, c, pgcrc_taus_005[r["source_dataset"]])
        s4 = _evaluate_predictions(index, targets, contexts, preds_4f[mkey], rf)
        f5 = cv5_metrics[sid]
        out_rows.append(
            {
                "method_id": sid,
                "label": label,
                **s4,
                "cv5_accuracy_mean": float(np.mean([m["accuracy"] for m in f5])),
                "cv5_accuracy_std": float(np.std([m["accuracy"] for m in f5], ddof=1)),
                "cv5_far_mean": float(np.mean([m["far"] for m in f5])),
                "cv5_far_std": float(np.std([m["far"] for m in f5], ddof=1)),
                "cv5_cost_mean": float(np.mean([m["mean_institutional_cost"] for m in f5])),
                "cv5_cost_std": float(np.std([m["mean_institutional_cost"] for m in f5], ddof=1)),
            }
        )

    write_csv("calibration_and_estimators.csv", out_rows)
    return {
        "max_parity_diff": max_parity_diff,
        "temp_by_fold": temp_by_fold,
        "pgcrc_taus_010": pgcrc_taus_010,
        "pgcrc_taus_005": pgcrc_taus_005,
    }


def _sample_correlated_votes(y_true: np.ndarray, acc: float, rho: float, rng: np.random.Generator, k_val: int = 4) -> np.ndarray:
    """Equicorrelated 4-class error channel preserving exact marginal accuracy `acc`.

    With probability `rho`, all K validators share a single latent decision (correct w.p. acc,
    else uniform wrong class); with probability `1 - rho`, each validator acts independently
    with marginal accuracy `acc` and uniform off-diagonal `(1 - acc) / 3`.
    This induces exact pairwise indicator correlation = rho while keeping every validator's
    marginal confusion matrix strictly identical (diag=acc, off=(1-acc)/3).
    """
    n = len(y_true)
    votes = np.empty((n, k_val), dtype=np.int64)
    shared_mask = rng.random(n) < rho
    # Shared error draw
    shared_ok = rng.random(n) < acc
    shared_off = rng.integers(1, 4, size=n)
    shared_vote = np.where(shared_ok, y_true, (y_true + shared_off) % 4)
    # Independent draws
    ind_ok = rng.random((n, k_val)) < acc
    ind_off = rng.integers(1, 4, size=(n, k_val))
    ind_votes = np.where(ind_ok, y_true[:, None], (y_true[:, None] + ind_off) % 4)
    votes[shared_mask] = shared_vote[shared_mask, None]
    votes[~shared_mask] = ind_votes[~shared_mask]
    return votes


def run_correlated_verifier_stress(index, targets, contexts, n_seeds: int = 10):
    """Stress test pairwise validator error correlation rho in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}."""
    sources = sorted({r["source_dataset"] for r in index})
    y_all = np.array([STATUS_IDX[targets[r["audit_join_id"]]] for r in index], dtype=np.int64)
    src_arr = np.array([r["source_dataset"] for r in index])
    ctx_list = [contexts[r["audit_join_id"]] for r in index]
    ref_list = [reference_action(targets[r["audit_join_id"]], contexts[r["audit_join_id"]]) for r in index]
    pid_list = [r["parent_id"] for r in index]
    non_acc_total = sum(r != "accept" for r in ref_list)

    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    methods = [
        ("majority_direct", "Majority / direct"),
        ("deciv_fixed", "DeCiv Fixed / risk"),
        ("deciv_adaptive_085", "DeCiv Adaptive (0.85) / risk"),
        ("deciv_ds_raw", "DeCiv + Dawid-Skene (raw) / risk"),
        ("deciv_ds_temp", "DeCiv + DS + TempScale / risk"),
        ("deciv_pg_crc", "DeCiv + DS + PG-CRC (alpha=0.01)"),
    ]

    rows_out = []
    for rho in rhos:
        per_method = {mid: defaultdict(list) for mid, _ in methods}
        for seed in range(n_seeds):
            rng = np.random.default_rng(20261025 + int(round(rho * 1000)) * 100 + seed)
            votes_all = _sample_correlated_votes(y_all, acc=0.70, rho=rho, rng=rng, k_val=4)

            # Evaluate 4-fold cross-domain
            preds = {mid: [None] * len(index) for mid, _ in methods}
            probs_ds_raw_all = np.zeros((len(index), 4))
            probs_ds_temp_all = np.zeros((len(index), 4))
            temps = []
            taus = []

            for src in sources:
                tr_mask = src_arr != src
                te_mask = src_arr == src
                tr_idx = np.where(tr_mask)[0]
                te_idx = np.where(te_mask)[0]
                v_tr, v_te = votes_all[tr_idx], votes_all[te_idx]
                y_tr = y_all[tr_idx]

                # Reliability
                rel = np.array([(np.sum(v_tr[:, k] == y_tr) + 1.0) / (len(y_tr) + 2.0) for k in range(4)])

                ds = VectorizedDawidSkene(smoothing=1e-3).fit_hard(v_tr)
                p_tr_raw = ds.predict_proba_hard(v_tr, apply_temp=False)
                p_te_raw = ds.predict_proba_hard(v_te, apply_temp=False)
                t_opt = fit_temperature(p_tr_raw, y_tr)
                temps.append(t_opt)
                p_te_temp = apply_temperature(p_te_raw, t_opt)

                tau_hat, _ = calibrate_pg_crc_threshold(
                    p_tr_raw,
                    [ctx_list[i] for i in tr_idx],
                    [ref_list[i] for i in tr_idx],
                    [pid_list[i] for i in tr_idx],
                    alpha=0.01,
                )
                taus.append(tau_hat)

                for loc_i, gi in enumerate(te_idx):
                    v = v_te[loc_i]
                    ctx = ctx_list[gi]
                    # Fixed / majority
                    p_fix = np.bincount(v, minlength=4).astype(np.float64) / 4.0
                    # Adaptive 0.85
                    lam = 0.85
                    w = np.array(
                        [rel[k] * ((1.0 - lam) + lam * (np.sum(v == v[k]) - 1) / 3.0) for k in range(4)],
                        dtype=np.float64,
                    )
                    p_adp = np.array([np.sum(w[v == s]) for s in range(4)], dtype=np.float64) / np.sum(w)

                    # Direct majority
                    mx = np.max(p_fix)
                    winners = [ACTIONS[j] for j in range(4) if abs(p_fix[j] - mx) <= 1e-12]
                    act_maj = winners[0] if len(winners) == 1 else "defer"
                    allowed, _ = eligible_actions(ctx)
                    preds["majority_direct"][gi] = (act_maj, act_maj in allowed)
                    preds["deciv_fixed"][gi] = route_risk(p_fix, ctx)
                    preds["deciv_adaptive_085"][gi] = route_risk(p_adp, ctx)
                    preds["deciv_ds_raw"][gi] = route_risk(p_te_raw[loc_i], ctx)
                    preds["deciv_ds_temp"][gi] = route_risk(p_te_temp[loc_i], ctx)
                    preds["deciv_pg_crc"][gi] = route_pg_crc(p_te_raw[loc_i], ctx, tau_hat)
                    probs_ds_raw_all[gi] = p_te_raw[loc_i]
                    probs_ds_temp_all[gi] = p_te_temp[loc_i]

            for mid, _ in methods:
                acc_c = 0
                fa_c = 0
                tot_acc = 0
                good_acc = 0
                cost_sum = 0.0
                for gi in range(len(index)):
                    act, adm = preds[mid][gi]
                    ref = ref_list[gi]
                    acc_c += int(act == ref)
                    fa_c += int(act == "accept" and ref != "accept")
                    tot_acc += int(act == "accept")
                    good_acc += int(act == "accept" and ref == "accept")
                    cost_sum += realized_institutional_cost(act, ref, STATUSES[y_all[gi]], bool(adm))
                per_method[mid]["accuracy"].append(acc_c / len(index))
                per_method[mid]["far"].append(fa_c / non_acc_total)
                per_method[mid]["accept_precision"].append(good_acc / tot_acc if tot_acc > 0 else 1.0)
                per_method[mid]["acceptance_rate"].append(tot_acc / len(index))
                per_method[mid]["mean_institutional_cost"].append(cost_sum / len(index))
                per_method[mid]["mean_temp"].append(float(np.mean(temps)) if mid == "deciv_ds_temp" else 1.0)
                per_method[mid]["mean_tau"].append(float(np.mean(taus)) if mid == "deciv_pg_crc" else 0.0)
                if mid == "deciv_ds_raw":
                    per_method[mid]["ece"].append(calibration_metrics(probs_ds_raw_all, y_all)["ece"])
                elif mid == "deciv_ds_temp":
                    per_method[mid]["ece"].append(calibration_metrics(probs_ds_temp_all, y_all)["ece"])
                else:
                    per_method[mid]["ece"].append(float("nan"))

        for mid, label in methods:
            d = per_method[mid]
            rows_out.append(
                {
                    "rho": rho,
                    "method_id": mid,
                    "label": label,
                    "accuracy_mean": float(np.mean(d["accuracy"])),
                    "accuracy_std": float(np.std(d["accuracy"], ddof=1)),
                    "far_mean": float(np.mean(d["far"])),
                    "far_std": float(np.std(d["far"], ddof=1)),
                    "accept_precision_mean": float(np.mean(d["accept_precision"])),
                    "acceptance_rate_mean": float(np.mean(d["acceptance_rate"])),
                    "cost_mean": float(np.mean(d["mean_institutional_cost"])),
                    "cost_std": float(np.std(d["mean_institutional_cost"], ddof=1)),
                    "ece_mean": float(np.nanmean(d["ece"])) if not np.all(np.isnan(d["ece"])) else "",
                    "mean_fitted_temp": float(np.mean(d["mean_temp"])),
                    "mean_pgcrc_tau": float(np.mean(d["mean_tau"])),
                }
            )
        print(
            f"[Correlated Stress rho={rho:.1f}] "
            f"DS-raw FAR={np.mean(per_method['deciv_ds_raw']['far'])*100:.2f}% "
            f"Cost={np.mean(per_method['deciv_ds_raw']['mean_institutional_cost']):.3f} | "
            f"DS+Temp FAR={np.mean(per_method['deciv_ds_temp']['far'])*100:.2f}% "
            f"(T={np.mean(per_method['deciv_ds_temp']['mean_temp']):.2f}) | "
            f"PG-CRC FAR={np.mean(per_method['deciv_pg_crc']['far'])*100:.2f}%",
            flush=True,
        )

    write_csv("correlated_verifier_stress.csv", rows_out)
    return rows_out


def run_noisy_gate_sensitivity(index, targets, contexts, inputs, n_seeds: int = 20):
    """Evaluate DeCiv resilience when metadata detectors (provenance, auditability, repairability) have noise eta."""
    sources = sorted({r["source_dataset"] for r in index})
    # Precompute clean 4-fold DS + TempScale posteriors
    post_temp = {}
    post_raw = {}
    for src in sources:
        tr = [r for r in index if r["source_dataset"] != src]
        te = [r for r in index if r["source_dataset"] == src]
        v_tr, _ = _extract_vote_matrix(tr, inputs)
        v_te, _ = _extract_vote_matrix(te, inputs)
        y_tr = np.array([STATUS_IDX[targets[r["audit_join_id"]]] for r in tr], dtype=np.int64)
        ds = VectorizedDawidSkene(smoothing=1e-3).fit_hard(v_tr)
        t_opt = fit_temperature(ds.predict_proba_hard(v_tr, apply_temp=False), y_tr)
        p_raw = ds.predict_proba_hard(v_te, apply_temp=False)
        p_tmp = apply_temperature(p_raw, t_opt)
        for i, r in enumerate(te):
            post_raw[r["audit_join_id"]] = p_raw[i]
            post_temp[r["audit_join_id"]] = p_tmp[i]

    noise_rates = [0.0, 0.01, 0.02, 0.05, 0.10]
    rows_out = []
    for eta in noise_rates:
        for mname, pmap in [("ds_temp", post_temp), ("ds_raw", post_raw)]:
            accs, fars, evrs, costs = [], [], [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng(777000 + int(round(eta * 1000)) * 100 + seed)
                eval_rows = []
                ev_count = 0
                for row in index:
                    k = row["audit_join_id"]
                    true_ctx = contexts[k]
                    noisy_ctx = dict(true_ctx)
                    if eta > 0.0:
                        if rng.random() < eta:
                            noisy_ctx["provenance_status"] = (
                                "incomplete" if true_ctx["provenance_status"] == "complete" else "complete"
                            )
                        if rng.random() < eta:
                            noisy_ctx["evidence_auditability"] = (
                                "unauditable" if true_ctx["evidence_auditability"] == "auditable" else "auditable"
                            )
                        if rng.random() < eta:
                            noisy_ctx["repairability"] = (
                                "no_repair" if true_ctx["repairability"] == "verified_repair" else "verified_repair"
                            )
                    # Route using NOISY runtime context, but score against TRUE ground-truth admissibility & reference!
                    act, _ = route_risk(pmap[k], noisy_ctx)
                    true_allowed, _ = eligible_actions(true_ctx)
                    true_adm = act in true_allowed
                    if not true_adm:
                        ev_count += 1
                    ref = reference_action(targets[k], true_ctx)
                    cost = realized_institutional_cost(act, ref, targets[k], true_adm)
                    eval_rows.append(
                        {
                            "predicted_action": act,
                            "reference_action": ref,
                            "correct": int(act == ref),
                            "false_accept": int(act == "accept" and ref != "accept"),
                            "eligible": int(true_adm),
                            "institutional_cost": cost,
                        }
                    )
                s = summarize(eval_rows)
                accs.append(s["accuracy"])
                fars.append(s["far"])
                evrs.append(ev_count / len(index))
                costs.append(s["mean_institutional_cost"])
            rows_out.append(
                {
                    "noise_rate": eta,
                    "method": mname,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs, ddof=1)) if n_seeds > 1 else 0.0,
                    "far_mean": float(np.mean(fars)),
                    "far_std": float(np.std(fars, ddof=1)) if n_seeds > 1 else 0.0,
                    "evr_mean": float(np.mean(evrs)),
                    "cost_mean": float(np.mean(costs)),
                    "cost_std": float(np.std(costs, ddof=1)) if n_seeds > 1 else 0.0,
                }
            )
        print(
            f"[Noisy Gate eta={eta*100:4.1f}%] DS+Temp: Acc={rows_out[-2]['accuracy_mean']*100:.2f}%, "
            f"FAR={rows_out[-2]['far_mean']*100:.2f}%, EVR={rows_out[-2]['evr_mean']*100:.2f}%, "
            f"Cost={rows_out[-2]['cost_mean']:.3f}",
            flush=True,
        )
    write_csv("noisy_gate_sensitivity.csv", rows_out)
    return rows_out


def run_merkle_and_vectorized_efficiency(index, inputs):
    """Benchmark vectorized DS speedup and RFC 9162 Merkle Transparency Log scaling."""
    sources = sorted({r["source_dataset"] for r in index})
    tr0 = [r for r in index if r["source_dataset"] != sources[0]]
    tr0_in = tuple(inputs[r["audit_join_id"]] for r in tr0)
    all_in = tuple(inputs[r["audit_join_id"]] for r in index)
    v_tr0, _ = _extract_vote_matrix(tr0, inputs)
    v_all, _ = _extract_vote_matrix(index, inputs)

    # 1. Scalar vs Vectorized Dawid-Skene
    t0 = time.perf_counter()
    ds_scalar = SemanticDawidSkeneEstimator().fit(tr0_in)
    t_scalar_fit_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(5):
        _ = ds_scalar.predict_proba(all_in)
    t_scalar_inf_us = (time.perf_counter() - t0) * 1e6 / (len(index) * 5)

    t0 = time.perf_counter()
    ds_vec = VectorizedDawidSkene(smoothing=1e-3).fit_hard(v_tr0)
    t_vec_fit_ms = (time.perf_counter() - t0) * 1e3

    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = ds_vec.predict_proba_hard(v_all, apply_temp=False)
    t_vec_inf_us = (time.perf_counter() - t0) * 1e6 / (len(index) * reps)

    # 2. Linear SHA-256 chain vs RFC 9162 Batched Merkle Transparency Log
    payloads_1800 = [f"receipt:{r['audit_join_id']}:{r['parent_id']}:accept".encode() for r in index]
    t0 = time.perf_counter()
    for _ in range(10):
        prev = b"\x00" * 32
        for p in payloads_1800:
            prev = hashlib.sha256(prev + p).digest()
    t_chain_us = (time.perf_counter() - t0) * 1e6 / (len(index) * 10)

    # Merkle append + root checkpoint every B=64 packets
    t0 = time.perf_counter()
    for _ in range(10):
        log = MerkleTransparencyLog()
        for i, p in enumerate(payloads_1800):
            log.append(p)
            if (i + 1) % 64 == 0 or (i + 1) == len(payloads_1800):
                _ = log.root()
    t_merkle_batch_us = (time.perf_counter() - t0) * 1e6 / (len(index) * 10)

    # Cryptographic correctness check & proof size scaling across N = 1k, 10k, 50k
    scaling_rows = []
    for n_rec in (1800, 10000, 50000):
        mlog = MerkleTransparencyLog()
        sample_payloads = [f"deciv-receipt-{i:06d}".encode() for i in range(n_rec)]
        t0 = time.perf_counter()
        mlog.extend(sample_payloads)
        root_full = mlog.root()
        build_ms = (time.perf_counter() - t0) * 1e3

        # Measure O(log N) inclusion & consistency proof generation + verification (averaged over 50 calls)
        half = n_rec // 2
        root_half = mlog.root(half)
        proof_inc = mlog.inclusion_proof(n_rec // 3)
        assert MerkleTransparencyLog.verify_inclusion(
            sample_payloads[n_rec // 3], n_rec // 3, n_rec, proof_inc, root_full
        )
        assert not MerkleTransparencyLog.verify_inclusion(
            b"tampered-receipt", n_rec // 3, n_rec, proof_inc, root_full
        )
        t0 = time.perf_counter()
        for _ in range(50):
            p_i = mlog.inclusion_proof(n_rec // 3)
            _ = MerkleTransparencyLog.verify_inclusion(
                sample_payloads[n_rec // 3], n_rec // 3, n_rec, p_i, root_full
            )
        t_inc_us = (time.perf_counter() - t0) * 1e6 / 50.0

        proof_cons = mlog.consistency_proof(half, n_rec)
        assert MerkleTransparencyLog.verify_consistency(half, n_rec, root_half, root_full, proof_cons)
        t0 = time.perf_counter()
        for _ in range(50):
            p_c = mlog.consistency_proof(half, n_rec)
            _ = MerkleTransparencyLog.verify_consistency(half, n_rec, root_half, root_full, p_c)
        t_cons_us = (time.perf_counter() - t0) * 1e6 / 50.0

        scaling_rows.append(
            {
                "log_records": n_rec,
                "tree_build_ms": round(build_ms, 2),
                "inclusion_proof_hashes": len(proof_inc),
                "inclusion_proof_bytes": len(proof_inc) * 32,
                "inclusion_verify_us": round(t_inc_us, 2),
                "consistency_proof_hashes": len(proof_cons),
                "consistency_verify_us": round(t_cons_us, 2),
                "linear_chain_verify_hashes": n_rec,
                "proof_compression_ratio": round(n_rec / max(1, len(proof_inc)), 1),
            }
        )

    eff_summary = {
        "scalar_ds_fit_ms": round(t_scalar_fit_ms, 2),
        "vec_ds_fit_ms": round(t_vec_fit_ms, 2),
        "fit_speedup_x": round(t_scalar_fit_ms / max(1e-9, t_vec_fit_ms), 1),
        "scalar_ds_inf_us": round(t_scalar_inf_us, 3),
        "vec_ds_inf_us": round(t_vec_inf_us, 3),
        "inf_speedup_x": round(t_scalar_inf_us / max(1e-9, t_vec_inf_us), 1),
        "linear_chain_us_per_pkt": round(t_chain_us, 3),
        "merkle_batch64_us_per_pkt": round(t_merkle_batch_us, 3),
        "scaling": scaling_rows,
    }
    write_csv("merkle_scaling.csv", scaling_rows)
    (ROOT / "results" / "plan1_efficiency_summary.json").write_text(
        json.dumps(eff_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[Efficiency] Vectorized DS: fit {t_scalar_fit_ms:.1f}ms -> {t_vec_fit_ms:.1f}ms ({eff_summary['fit_speedup_x']}x), "
        f"online {t_scalar_inf_us:.2f}us -> {t_vec_inf_us:.3f}us ({eff_summary['inf_speedup_x']}x) | "
        f"Merkle(B=64): {t_merkle_batch_us:.2f}us/pkt vs chain {t_chain_us:.2f}us/pkt",
        flush=True,
    )
    return eff_summary


def main():
    index, targets, contexts, inputs = load_study()
    cal_meta = run_calibration_and_pgcrc(index, targets, contexts, inputs)
    run_correlated_verifier_stress(index, targets, contexts, n_seeds=10)
    run_noisy_gate_sensitivity(index, targets, contexts, inputs, n_seeds=20)
    eff_meta = run_merkle_and_vectorized_efficiency(index, inputs)
    summary = {"calibration_meta": cal_meta, "efficiency_meta": eff_meta}
    (ROOT / "results" / "plan1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== PLAN 1 UPGRADES COMPLETE ===", flush=True)


if __name__ == "__main__":
    import hashlib  # used in linear chain benchmark
    main()
