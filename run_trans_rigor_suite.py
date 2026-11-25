"""Stage 1 IEEE Transactions Rigor Suite on DeCiv-Real-1200 (CPU, 100% reproducible):
  1. Real-data DeCiv-CAF component & architecture ablation (isolating pure Log-Linear CAF,
     Tree-Only CAF, cross-family consensus products, nested OOF temperature scaling, and
     mass-preserving admissibility projection) -> results/real_caf_ablation.csv
  2. Parent-cluster paired bootstrap (B=10,000) & exact McNemar significance tests vs.
     strongest gated baselines (Solo GPT-6.1-Sol + Gate, GBDT Stacking + Gate, Conformal + Gate)
     -> results/real_significance_tests.csv
  3. Real 5-model cost-matrix sensitivity (deferral cost d and false-accept penalty C_acc)
     -> results/real_cost_sensitivity.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deciv_extensions import (
    ACTION_IDX,
    BASE_COST,
    STATUS_IDX,
    TIE_ORDER,
    CorrelationAwareFusion,
    VectorizedDawidSkene,
    apply_temperature,
    build_caf_epistemic_features,
    calibrate_decoupled_pg_crc_threshold,
    calibration_metrics,
    eligible_actions,
    fit_temperature,
    route_projected_pg_crc,
    route_risk,
)
from evaluate_real_verifiers import (
    VERIFIERS,
    load_real_study,
)

RESULTS = ROOT / "results"


def make_5fold_parent_splits(rows: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20261025)
    by_dom: dict[str, list[str]] = {}
    for r in rows:
        lst = by_dom.setdefault(r["source_dataset"], [])
        if r["parent_id"] not in lst:
            lst.append(r["parent_id"])
    pid_to_fold: dict[str, int] = {}
    for dom in sorted(by_dom):
        plist = sorted(by_dom[dom])
        perm = rng.permutation(len(plist))
        for pos, pidx in enumerate(perm):
            pid_to_fold[plist[pidx]] = pos % 5
    folds = []
    for f in range(5):
        tr = np.array([i for i, r in enumerate(rows) if pid_to_fold[r["parent_id"]] != f], dtype=np.int64)
        te = np.array([i for i, r in enumerate(rows) if pid_to_fold[r["parent_id"]] == f], dtype=np.int64)
        folds.append((tr, te))
    return folds


def eval_actions(actions: list[str], rows: list[dict], probs: np.ndarray) -> dict:
    n = len(rows)
    refs = [r["reference_action"] for r in rows]
    acc = sum(a == r for a, r in zip(actions, refs)) / n
    non_acc = [i for i, r in enumerate(refs) if r != "accept"]
    fa = sum(actions[i] == "accept" for i in non_acc)
    far = fa / max(1, len(non_acc))
    pred_acc = [i for i, a in enumerate(actions) if a == "accept"]
    tp_acc = sum(refs[i] == "accept" for i in pred_acc)
    prec = tp_acc / len(pred_acc) if pred_acc else 1.0
    true_acc = [i for i, r in enumerate(refs) if r == "accept"]
    rec_acc = tp_acc / max(1, len(true_acc))
    non_defer = sum(a != "defer" for a in actions) / n
    ev = 0
    for a, r in zip(actions, rows):
        allowed, _ = eligible_actions(
            {
                "provenance_status": r["provenance_status"],
                "evidence_auditability": r["evidence_auditability"],
                "human_review_required": r["human_review_required"],
                "repairability": r["repairability"],
            }
        )
        if a not in allowed:
            ev += 1
    cost = float(np.mean([BASE_COST[ACTION_IDX[a], STATUS_IDX[r["semantic_status"]]] for a, r in zip(actions, rows)]))
    y_true = np.array([STATUS_IDX[r["semantic_status"]] for r in rows], dtype=np.int64)
    cal = calibration_metrics(probs, y_true)
    return {
        "accuracy": acc,
        "far": far,
        "accept_precision": prec,
        "recall_accept": rec_acc,
        "non_defer_fraction": non_defer,
        "eligibility_violations": ev,
        "mean_institutional_cost": cost,
        "ece": cal["ece"],
        "nll": cal["nll"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def build_features_ablation(soft_probs: np.ndarray, include_cross: bool = True) -> np.ndarray:
    if include_cross:
        return build_caf_epistemic_features(soft_probs)
    p = np.clip(np.asarray(soft_probs, dtype=np.float64), 1e-6, 1.0)
    n = p.shape[0]
    logp = np.log(p)
    clr = logp - np.mean(logp, axis=2, keepdims=True)
    return np.hstack([p.reshape(n, -1), clr.reshape(n, -1)])


class AblationCAF:
    """Configurable Stage-I CAF for exact component isolation."""

    def __init__(
        self,
        mode: str = "hybrid",  # 'loglinear', 'trees', 'hybrid'
        include_cross: bool = True,
        use_oof_temp: bool = True,
        reg_c: float = 0.10,
    ) -> None:
        self.mode = mode
        self.include_cross = include_cross
        self.use_oof_temp = use_oof_temp
        self.reg_c = reg_c
        self.temperature_ = 1.0
        self.fitted_ = None

    def _fit_one(self, X: np.ndarray, y: np.ndarray):
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=self.reg_c, max_iter=1000).fit(sc.transform(X), y) if self.mode in ("loglinear", "hybrid") else None
        gb = (
            HistGradientBoostingClassifier(
                max_depth=3, max_iter=80, learning_rate=0.05, l2_regularization=5.0, min_samples_leaf=15, random_state=42
            ).fit(X, y)
            if self.mode in ("trees", "hybrid")
            else None
        )
        et = (
            ExtraTreesClassifier(n_estimators=120, max_depth=8, min_samples_leaf=4, n_jobs=1, random_state=42).fit(X, y)
            if self.mode in ("trees", "hybrid")
            else None
        )
        return sc, lr, gb, et

    def _pred_one(self, X: np.ndarray, models) -> np.ndarray:
        sc, lr, gb, et = models
        if self.mode == "loglinear":
            return lr.predict_proba(sc.transform(X))
        if self.mode == "trees":
            return 0.60 * gb.predict_proba(X) + 0.40 * et.predict_proba(X)
        return 0.50 * lr.predict_proba(sc.transform(X)) + 0.30 * gb.predict_proba(X) + 0.20 * et.predict_proba(X)

    def fit_with_inner_oof(self, soft_probs: np.ndarray, y: np.ndarray, group_ids: list[str]) -> np.ndarray:
        X = build_features_ablation(soft_probs, self.include_cross)
        groups = np.asarray(group_ids)
        uniq = sorted(set(groups.tolist()))
        if len(uniq) <= 6:
            folds = [groups == g for g in uniq]
        else:
            g2f = {g: idx % 5 for idx, g in enumerate(uniq)}
            farr = np.array([g2f[g] for g in groups], dtype=np.int64)
            folds = [farr == f for f in range(5)]

        inner_raw = np.zeros((len(y), 4), dtype=np.float64)
        for vm in folds:
            tm = ~vm
            if not np.any(vm) or not np.any(tm):
                continue
            m = self._fit_one(X[tm], y[tm])
            inner_raw[vm] = self._pred_one(X[vm], m)

        self.temperature_ = fit_temperature(inner_raw, y) if self.use_oof_temp else 1.0
        inner_cal = apply_temperature(inner_raw, self.temperature_) if self.use_oof_temp else inner_raw
        self.fitted_ = self._fit_one(X, y)
        return inner_cal

    def predict_proba(self, soft_probs: np.ndarray) -> np.ndarray:
        X = build_features_ablation(soft_probs, self.include_cross)
        raw = self._pred_one(X, self.fitted_)
        return apply_temperature(raw, self.temperature_) if self.use_oof_temp else raw


def route_naive_argmax_no_proj(posterior: np.ndarray, context: dict[str, str], tau_accept: float = 0.50) -> str:
    """Naive argmax over admissible actions WITHOUT mass-preserving projection onto defer."""
    p = np.asarray(posterior, dtype=np.float64)
    allowed, _ = eligible_actions(context)
    if "accept" in allowed and float(p[0]) >= tau_accept and int(np.argmax(p)) == 0:
        return "accept"
    rem = allowed - {"accept"}
    best_p = max(float(p[ACTION_IDX[a]]) for a in rem)
    return next(a for a in TIE_ORDER if a in rem and float(p[ACTION_IDX[a]]) >= best_p - 1e-12)


def route_projected_map(posterior: np.ndarray, context: dict[str, str]) -> str:
    """Mass-preserving admissibility-projected MAP router (identical to Table 3 deciv_caf_gated_map)."""
    return route_projected_pg_crc(posterior, context, tau_accept=0.50)[0]


def route_custom_cost(posterior: np.ndarray, context: dict[str, str], defer_cost: float, c_acc: float) -> str:
    costs = BASE_COST.copy()
    costs[0, 1] = c_acc
    costs[0, 2] = 0.75 * c_acc
    costs[0, 3] = c_acc
    costs[3, :3] = defer_cost
    risks = costs @ np.asarray(posterior, dtype=np.float64)
    allowed, _ = eligible_actions(context)
    best = min(risks[ACTION_IDX[a]] for a in allowed)
    return next(a for a in TIE_ORDER if a in allowed and risks[ACTION_IDX[a]] <= best + 1e-12)


def run_ablation(rows, soft, y_true, contexts, refs, pids, srcs, folds5) -> tuple[list[dict], dict]:
    domains = sorted(set(srcs))
    n = len(rows)

    def eval_variant(vid: str, label: str, family: str, stage1_fn, router_mode: str, alpha: float = 0.01):
        # 4-Fold LODO
        oof_p = np.zeros((n, 4), dtype=np.float64)
        oof_a = [""] * n
        for dom in domains:
            tr = np.array([i for i in range(n) if srcs[i] != dom], dtype=np.int64)
            te = np.array([i for i in range(n) if srcs[i] == dom], dtype=np.int64)
            p_tr_oof, p_te = stage1_fn(tr, te, [srcs[i] for i in tr])
            oof_p[te] = p_te
            tau = 0.0
            if router_mode == "pgcrc":
                tau, _ = calibrate_decoupled_pg_crc_threshold(
                    p_tr_oof, [contexts[i] for i in tr], [refs[i] for i in tr], [pids[i] for i in tr], alpha=alpha
                )
            for j, idx in enumerate(te):
                if router_mode == "risk":
                    oof_a[idx] = route_risk(p_te[j], contexts[idx], defer_cost=0.5)[0]
                elif router_mode == "proj_map":
                    oof_a[idx] = route_projected_map(p_te[j], contexts[idx])
                elif router_mode == "naive_argmax":
                    oof_a[idx] = route_naive_argmax_no_proj(p_te[j], contexts[idx])
                elif router_mode == "pgcrc":
                    oof_a[idx] = route_projected_pg_crc(p_te[j], contexts[idx], tau_accept=tau)[0]

        m4 = eval_actions(oof_a, rows, oof_p)

        # 5-Fold Stratified Parent-Cluster CV
        f5_accs, f5_fars, f5_costs = [], [], []
        oof5_p = np.zeros((n, 4), dtype=np.float64)
        for tr, te in folds5:
            p_tr_oof, p_te = stage1_fn(tr, te, [pids[i] for i in tr])
            oof5_p[te] = p_te
            tau = 0.0
            if router_mode == "pgcrc":
                tau, _ = calibrate_decoupled_pg_crc_threshold(
                    p_tr_oof, [contexts[i] for i in tr], [refs[i] for i in tr], [pids[i] for i in tr], alpha=alpha
                )
            acts = []
            for j, idx in enumerate(te):
                if router_mode == "risk":
                    acts.append(route_risk(p_te[j], contexts[idx], defer_cost=0.5)[0])
                elif router_mode == "proj_map":
                    acts.append(route_projected_map(p_te[j], contexts[idx]))
                elif router_mode == "naive_argmax":
                    acts.append(route_naive_argmax_no_proj(p_te[j], contexts[idx]))
                elif router_mode == "pgcrc":
                    acts.append(route_projected_pg_crc(p_te[j], contexts[idx], tau_accept=tau)[0])
            mf = eval_actions(acts, [rows[i] for i in te], p_te)
            f5_accs.append(mf["accuracy"])
            f5_fars.append(mf["far"])
            f5_costs.append(mf["mean_institutional_cost"])

        cal5 = calibration_metrics(oof5_p, y_true)
        rec = {
            "variant_id": vid,
            "label": label,
            "stage1_family": family,
            "lodo_acc": round(m4["accuracy"], 4),
            "lodo_far": round(m4["far"], 4),
            "lodo_prec": round(m4["accept_precision"], 4),
            "lodo_rec_acc": round(m4["recall_accept"], 4),
            "lodo_cost": round(m4["mean_institutional_cost"], 4),
            "lodo_ece": round(m4["ece"], 4),
            "lodo_nll": round(m4["nll"], 4),
            "cv5_acc": round(float(np.mean(f5_accs)), 4),
            "cv5_far": round(float(np.mean(f5_fars)), 4),
            "cv5_cost": round(float(np.mean(f5_costs)), 4),
            "cv5_ece": round(float(cal5["ece"]), 4),
            "cv5_nll": round(float(cal5["nll"]), 4),
        }
        return rec, {"oof_p": oof_p, "oof_a": oof_a}

    def s1_sup_soft_ds(use_temp: bool):
        def fn(tr, te, grp):
            ds = VectorizedDawidSkene(smoothing=1.0).fit_soft(soft[tr], y_true[tr])
            if use_temp:
                ds.calibrate_temperature(ds.predict_proba_soft(soft[tr], apply_temp=False), y_true[tr])
            return ds.predict_proba_soft(soft[tr], apply_temp=use_temp), ds.predict_proba_soft(soft[te], apply_temp=use_temp)
        return fn

    def s1_caf(mode: str, include_cross: bool, use_oof_temp: bool):
        def fn(tr, te, grp):
            m = AblationCAF(mode=mode, include_cross=include_cross, use_oof_temp=use_oof_temp, reg_c=0.10)
            p_tr_oof = m.fit_with_inner_oof(soft[tr], y_true[tr], grp)
            p_te = m.predict_proba(soft[te])
            return p_tr_oof, p_te
        return fn

    specs = [
        ("sup_soft_ds_raw", "Sup-Soft-DS (raw, CI) / risk", "Conditional-Indep.", s1_sup_soft_ds(False), "risk"),
        ("sup_soft_ds_temp", "Sup-Soft-DS + TempScale / risk", "Conditional-Indep.", s1_sup_soft_ds(True), "risk"),
        ("caf_loglinear_map", "Pure Log-Linear CAF (no trees) / proj. MAP", "Log-Linear Only", s1_caf("loglinear", True, True), "proj_map"),
        ("caf_loglinear_pgcrc010", "Pure Log-Linear CAF + Projected PG-CRC (0.01)", "Log-Linear Only", s1_caf("loglinear", True, True), "pgcrc"),
        ("caf_trees_map", "Tree-Only CAF (GBDT+ET) / proj. MAP", "Tree-Only", s1_caf("trees", True, True), "proj_map"),
        ("caf_no_cross_map", "DeCiv-CAF w/o Cross-Family Products / proj. MAP", "Hybrid (no cross)", s1_caf("hybrid", False, True), "proj_map"),
        ("caf_no_temp_map", "DeCiv-CAF w/o Nested OOF TempScale / proj. MAP", "Hybrid (uncal.)", s1_caf("hybrid", True, False), "proj_map"),
        ("caf_no_proj_argmax", "DeCiv-CAF w/o Mass-Preserving Projection (naive argmax)", "Hybrid (no proj.)", s1_caf("hybrid", True, True), "naive_argmax"),
        ("caf_full_proj_map", "DeCiv-CAF (Full Hybrid) / projected MAP", "Full Hybrid", s1_caf("hybrid", True, True), "proj_map"),
        ("caf_full_pgcrc010", "DeCiv-CAF (Full Hybrid) + Projected PG-CRC (0.01)", "Full Hybrid", s1_caf("hybrid", True, True), "pgcrc"),
        ("caf_full_risk", "DeCiv-CAF (Full Hybrid) / risk (d=0.5)", "Full Hybrid", s1_caf("hybrid", True, True), "risk"),
    ]

    out_rows = []
    cache = {}
    print("\n=== REAL DeCiv-Real-1200 COMPONENT & ARCHITECTURE ABLATION ===")
    print(f"{'Variant':56s} | {'4F Acc':6s} | {'4F FAR':6s} | {'4F Cost':7s} | {'4F ECE':6s} | {'4F NLL':6s} | {'5F Acc':6s} | {'5F NLL':6s}")
    print("-" * 116)
    for vid, lbl, fam, fn, rmode in specs:
        rec, c = eval_variant(vid, lbl, fam, fn, rmode)
        out_rows.append(rec)
        cache[vid] = c
        print(
            f"{lbl:56s} | {rec['lodo_acc']*100:5.2f}% | {rec['lodo_far']*100:5.2f}% | "
            f"{rec['lodo_cost']:7.3f} | {rec['lodo_ece']:6.4f} | {rec['lodo_nll']:6.3f} | "
            f"{rec['cv5_acc']*100:5.2f}% | {rec['cv5_nll']:6.3f}"
        )
    return out_rows, cache


def compute_packet_metrics(actions: list[str], probs: np.ndarray, rows: list[dict]):
    n = len(rows)
    acc = np.array([float(actions[i] == rows[i]["reference_action"]) for i in range(n)], dtype=np.float64)
    is_non_acc = np.array([rows[i]["reference_action"] != "accept" for i in range(n)], dtype=bool)
    fa = np.array([float(actions[i] == "accept" and rows[i]["reference_action"] != "accept") for i in range(n)], dtype=np.float64)
    cost = np.array([BASE_COST[ACTION_IDX[actions[i]], STATUS_IDX[rows[i]["semantic_status"]]] for i in range(n)], dtype=np.float64)
    y_true = np.array([STATUS_IDX[rows[i]["semantic_status"]] for i in range(n)], dtype=np.int64)
    nll = -np.log(np.clip(probs[np.arange(n), y_true], 1e-12, 1.0))
    return acc, fa, is_non_acc, cost, nll


def run_significance_tests(rows, soft, y_true, contexts, refs, pids, srcs, cache) -> list[dict]:
    """Parent-cluster paired bootstrap (B=10,000) + exact McNemar test on 4-fold LODO OOF."""
    dec_json = json.loads((RESULTS / "real_admitted_decisions.json").read_text(encoding="utf-8"))["lodo_decisions"]
    n = len(rows)
    domains = sorted(set(srcs))

    # Reconstruct baseline LODO OOF probabilities identically to evaluate_real_verifiers.py
    p_gpt = soft[:, 4, :]
    p_mean_raw = np.mean(soft, axis=1)
    p_mean = np.zeros((n, 4), dtype=np.float64)
    p_gbdt = np.zeros((n, 4), dtype=np.float64)
    for dom in domains:
        tr = np.array([i for i in range(n) if srcs[i] != dom], dtype=np.int64)
        te = np.array([i for i in range(n) if srcs[i] == dom], dtype=np.int64)
        t_pool = fit_temperature(p_mean_raw[tr], y_true[tr])
        p_mean[te] = apply_temperature(p_mean_raw[te], t_pool)
        X_tr = soft[tr].reshape(len(tr), -1)
        X_te = soft[te].reshape(len(te), -1)
        gb = HistGradientBoostingClassifier(
            max_depth=3, max_iter=80, learning_rate=0.05, l2_regularization=5.0, random_state=42
        ).fit(X_tr, y_true[tr])
        p_gbdt[te] = gb.predict_proba(X_te)

    methods = {
        "deciv_caf_proj_map": ([dec_json["deciv_caf_gated_map"][r["packet_id"]] for r in rows], cache["caf_full_proj_map"]["oof_p"]),
        "deciv_caf_pgcrc010": ([dec_json["deciv_caf_pg_crc_010"][r["packet_id"]] for r in rows], cache["caf_full_pgcrc010"]["oof_p"]),
        "deciv_caf_risk_d05": ([dec_json["deciv_caf_risk_d05"][r["packet_id"]] for r in rows], cache["caf_full_risk"]["oof_p"]),
        "deciv_sup_soft_ds_temp": ([dec_json["deciv_soft_ds_sup_temp"][r["packet_id"]] for r in rows], cache["sup_soft_ds_temp"]["oof_p"]),
        "solo_gpt61_sol_gated": ([dec_json["solo_gpt61_sol_gated"][r["packet_id"]] for r in rows], p_gpt),
        "gbdt_stack_gated": ([dec_json["gbdt_stack_gated"][r["packet_id"]] for r in rows], p_gbdt),
        "conformal_risk_gated": ([dec_json["conformal_risk_gated"][r["packet_id"]] for r in rows], p_mean),
    }

    # Group indices by parent cluster for cluster-level paired bootstrap (300 clusters)
    cl_map: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        cl_map.setdefault(r["parent_id"], []).append(i)
    cl_idxs = [np.array(v, dtype=np.int64) for v in cl_map.values()]
    n_cl = len(cl_idxs)

    rng = np.random.default_rng(2026)
    B = 10000
    boot_weights = rng.multinomial(n_cl, np.full(n_cl, 1.0 / n_cl), size=B).astype(np.float64)  # (B, n_cl)

    pairs = [
        ("deciv_caf_proj_map", "solo_gpt61_sol_gated", "DeCiv-CAF / proj. MAP vs. Solo GPT-6.1 + Gate"),
        ("deciv_caf_proj_map", "gbdt_stack_gated", "DeCiv-CAF / proj. MAP vs. GBDT Stacking + Gate"),
        ("deciv_caf_proj_map", "conformal_risk_gated", "DeCiv-CAF / proj. MAP vs. Conformal-Risk + Gate"),
        ("deciv_caf_pgcrc010", "solo_gpt61_sol_gated", "DeCiv-CAF + PG-CRC (0.01) vs. Solo GPT-6.1 + Gate"),
        ("deciv_caf_pgcrc010", "gbdt_stack_gated", "DeCiv-CAF + PG-CRC (0.01) vs. GBDT Stacking + Gate"),
        ("deciv_caf_risk_d05", "solo_gpt61_sol_gated", "DeCiv-CAF / risk (d=0.5) vs. Solo GPT-6.1 + Gate"),
        ("deciv_caf_risk_d05", "gbdt_stack_gated", "DeCiv-CAF / risk (d=0.5) vs. GBDT Stacking + Gate"),
        ("deciv_sup_soft_ds_temp", "solo_gpt61_sol_gated", "DeCiv + Sup-Soft-DS + Temp vs. Solo GPT-6.1 + Gate"),
    ]

    sig_rows = []
    print("\n=== PARENT-CLUSTER PAIRED BOOTSTRAP (B=10,000) & EXACT McNEMAR SIGNIFICANCE TESTS ===")
    for m_a, m_b, desc in pairs:
        acc_a, fa_a, nona, cost_a, nll_a = compute_packet_metrics(methods[m_a][0], methods[m_a][1], rows)
        acc_b, fa_b, _, cost_b, nll_b = compute_packet_metrics(methods[m_b][0], methods[m_b][1], rows)

        # McNemar on packet-level accuracy
        n10 = int(np.sum((acc_a == 1.0) & (acc_b == 0.0)))
        n01 = int(np.sum((acc_a == 0.0) & (acc_b == 1.0)))
        p_mcnemar = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if (n10 + n01) > 0 else 1.0

        # Cluster-level sums for vectorized paired bootstrap
        cl_acc_diff = np.array([np.sum(acc_a[ix] - acc_b[ix]) for ix in cl_idxs])
        cl_cnt = np.array([len(ix) for ix in cl_idxs], dtype=np.float64)
        cl_fa_diff = np.array([np.sum(fa_a[ix] - fa_b[ix]) for ix in cl_idxs])
        cl_nona_cnt = np.array([np.sum(nona[ix]) for ix in cl_idxs], dtype=np.float64)
        cl_cost_diff = np.array([np.sum(cost_a[ix] - cost_b[ix]) for ix in cl_idxs])
        cl_nll_diff = np.array([np.sum(nll_a[ix] - nll_b[ix]) for ix in cl_idxs])

        tot_n = boot_weights @ cl_cnt
        tot_nona = boot_weights @ cl_nona_cnt
        b_acc = (boot_weights @ cl_acc_diff) / tot_n
        b_far = (boot_weights @ cl_fa_diff) / tot_nona
        b_cost = (boot_weights @ cl_cost_diff) / tot_n
        b_nll = (boot_weights @ cl_nll_diff) / tot_n

        def two_sided_p(arr: np.ndarray) -> float:
            p_pos = float(np.mean(arr >= 0.0))
            p_neg = float(np.mean(arr <= 0.0))
            return min(1.0, 2.0 * min(p_pos, p_neg))

        rec = {
            "comparison": desc,
            "delta_acc_pp": round(float(np.mean(acc_a - acc_b) * 100.0), 2),
            "p_mcnemar_acc": round(p_mcnemar, 6),
            "p_boot_acc": round(two_sided_p(b_acc), 6),
            "delta_far_pp": round(float((np.sum(fa_a) - np.sum(fa_b)) / np.sum(nona) * 100.0), 2),
            "p_boot_far": round(two_sided_p(b_far), 6),
            "delta_cost": round(float(np.mean(cost_a - cost_b)), 4),
            "cost_ci95_lo": round(float(np.percentile(b_cost, 2.5)), 4),
            "cost_ci95_hi": round(float(np.percentile(b_cost, 97.5)), 4),
            "p_boot_cost": round(two_sided_p(b_cost), 6),
            "delta_nll": round(float(np.mean(nll_a - nll_b)), 4),
            "p_boot_nll": round(two_sided_p(b_nll), 6),
        }
        sig_rows.append(rec)
        print(
            f"  {desc:52s} | dAcc={rec['delta_acc_pp']:+5.2f}pp (p={rec['p_boot_acc']:.4f}) | "
            f"dFAR={rec['delta_far_pp']:+5.2f}pp (p={rec['p_boot_far']:.4f}) | "
            f"dCost={rec['delta_cost']:+.3f} (p={rec['p_boot_cost']:.4f}) | "
            f"dNLL={rec['delta_nll']:+.3f} (p={rec['p_boot_nll']:.4f})"
        )
    return sig_rows


def run_real_cost_sensitivity(rows, contexts, cache) -> list[dict]:
    """Real 5-model sensitivity across deferral cost d and false-accept penalty C_acc."""
    out = []
    print("\n=== REAL 5-MODEL COST-MATRIX SENSITIVITY (DeCiv-Real-1200, 4F LODO) ===")
    for method_id, key in [
        ("deciv_caf_risk", "caf_full_risk"),
        ("deciv_sup_soft_ds_temp", "sup_soft_ds_temp"),
    ]:
        oof_p = cache[key]["oof_p"]
        for c_acc in (10.0, 18.0, 30.0):
            for d in (0.1, 0.25, 0.5, 1.0, 2.0):
                acts = [route_custom_cost(oof_p[i], contexts[i], defer_cost=d, c_acc=c_acc) for i in range(len(rows))]
                m = eval_actions(acts, rows, oof_p)
                out.append(
                    {
                        "method_id": method_id,
                        "c_acc": c_acc,
                        "defer_cost": d,
                        "accuracy": round(m["accuracy"], 4),
                        "far": round(m["far"], 4),
                        "accept_precision": round(m["accept_precision"], 4),
                        "recall_accept": round(m["recall_accept"], 4),
                        "non_defer_fraction": round(m["non_defer_fraction"], 4),
                        "mean_institutional_cost": round(m["mean_institutional_cost"], 4),
                        "eligibility_violations": m["eligibility_violations"],
                    }
                )
                if abs(c_acc - 18.0) < 1e-6:
                    print(
                        f"  {method_id:24s} C_acc={c_acc:4.0f} d={d:4.2f} | "
                        f"Acc={m['accuracy']*100:5.2f}% | FAR={m['far']*100:4.2f}% | "
                        f"AccRec={m['recall_accept']*100:5.1f}% | NonDef={m['non_defer_fraction']*100:5.1f}% | "
                        f"Cost={m['mean_institutional_cost']:.3f}"
                    )
    return out


def main():
    rows, soft, _, y_true, contexts, refs, _ = load_real_study()
    pids = [r["parent_id"] for r in rows]
    srcs = [r["source_dataset"] for r in rows]
    folds5 = make_5fold_parent_splits(rows)

    ab_rows, cache = run_ablation(rows, soft, y_true, contexts, refs, pids, srcs, folds5)
    sig_rows = run_significance_tests(rows, soft, y_true, contexts, refs, pids, srcs, cache)
    sens_rows = run_real_cost_sensitivity(rows, contexts, cache)

    write_csv(RESULTS / "real_caf_ablation.csv", ab_rows)
    write_csv(RESULTS / "real_significance_tests.csv", sig_rows)
    write_csv(RESULTS / "real_cost_sensitivity.csv", sens_rows)
    print("\nSaved real_caf_ablation.csv, real_significance_tests.csv, real_cost_sensitivity.csv")


if __name__ == "__main__":
    main()
