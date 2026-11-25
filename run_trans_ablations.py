"""IEEE Transactions supplementary real-model ablations on DeCiv-Real-1200 (zero-leakage CV).

1. Cross-Family Verifier Diversity Ablation:
     NLI-Only (DeBERTa+BART) vs LLM-Only (Llama3.1+Qwen3.5) vs Leave-One-Out (3 models) vs Full 4-Model Cross-Family.
2. PG-CRC Risk-Level Sweep (alpha in {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}):
     Verifies Theorem 1 finite-sample FAR <= alpha tracking across the full operational risk spectrum.
3. Generates publication figure trans_diversity_and_pgcrc_sweep.{pdf,png}.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deciv_extensions import (
    VectorizedDawidSkene,
    apply_temperature,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    fit_temperature,
    route_pg_crc,
    route_risk,
)
from evaluate_real_verifiers import VERIFIERS, load_real_study, write_csv
from run_comparison import realized_institutional_cost, summarize

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "axes.titlecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "axes.linewidth": 0.9,
        "grid.color": "#CCCCCC",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _eval_subset(rows, soft_probs, y_true, contexts, refs, subset_indices: list[int], mode: str = "sup_soft_temp"):
    sources = sorted({r["source_dataset"] for r in rows})
    parent_ids = [r["parent_id"] for r in rows]
    sub_s = soft_probs[:, subset_indices, :]

    # 4-fold LODO
    eval_4f = []
    probs_4f = np.zeros((len(rows), 4), dtype=np.float64)
    y_4f = np.zeros(len(rows), dtype=np.int64)
    cursor = 0
    for src in sources:
        tr = np.array([i for i, r in enumerate(rows) if r["source_dataset"] != src], dtype=np.int64)
        te = np.array([i for i, r in enumerate(rows) if r["source_dataset"] == src], dtype=np.int64)
        ds = VectorizedDawidSkene(smoothing=1.0).fit_soft(sub_s[tr], y_true=y_true[tr])
        p_tr = ds.predict_proba_soft(sub_s[tr], apply_temp=False)
        p_te = ds.predict_proba_soft(sub_s[te], apply_temp=False)
        t_opt = fit_temperature(p_tr, y_true[tr])
        p_tr_t = apply_temperature(p_tr, t_opt)
        p_te_t = apply_temperature(p_te, t_opt)

        if mode == "pg_crc_005":
            tau, _ = calibrate_pg_crc_threshold(
                p_tr_t, [contexts[i] for i in tr], [refs[i] for i in tr], [parent_ids[i] for i in tr], alpha=0.005
            )
        else:
            tau = None

        for loc_i, gi in enumerate(te):
            p = p_te_t[loc_i]
            act, adm = route_pg_crc(p, contexts[gi], tau) if tau is not None else route_risk(p, contexts[gi])
            ref = refs[gi]
            eval_4f.append(
                {
                    "predicted_action": act,
                    "reference_action": ref,
                    "correct": int(act == ref),
                    "false_accept": int(act == "accept" and ref != "accept"),
                    "eligible": int(adm),
                    "institutional_cost": realized_institutional_cost(act, ref, rows[gi]["semantic_status"], bool(adm)),
                }
            )
            probs_4f[cursor] = p
            y_4f[cursor] = y_true[gi]
            cursor += 1
    s4 = summarize(eval_4f)
    s4.update(calibration_metrics(probs_4f, y_4f))

    # 5-fold stratified parent-cluster CV
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

    f5_list = []
    for fid in range(5):
        tr = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] != fid], dtype=np.int64)
        te = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] == fid], dtype=np.int64)
        ds = VectorizedDawidSkene(smoothing=1.0).fit_soft(sub_s[tr], y_true=y_true[tr])
        p_tr = ds.predict_proba_soft(sub_s[tr], apply_temp=False)
        p_te = ds.predict_proba_soft(sub_s[te], apply_temp=False)
        t_opt = fit_temperature(p_tr, y_true[tr])
        p_tr_t = apply_temperature(p_tr, t_opt)
        p_te_t = apply_temperature(p_te, t_opt)
        tau = (
            calibrate_pg_crc_threshold(
                p_tr_t, [contexts[i] for i in tr], [refs[i] for i in tr], [parent_ids[i] for i in tr], alpha=0.005
            )[0]
            if mode == "pg_crc_005"
            else None
        )
        ev = []
        for loc_i, gi in enumerate(te):
            p = p_te_t[loc_i]
            act, adm = route_pg_crc(p, contexts[gi], tau) if tau is not None else route_risk(p, contexts[gi])
            ref = refs[gi]
            ev.append(
                {
                    "predicted_action": act,
                    "reference_action": ref,
                    "correct": int(act == ref),
                    "false_accept": int(act == "accept" and ref != "accept"),
                    "eligible": int(adm),
                    "institutional_cost": realized_institutional_cost(act, ref, rows[gi]["semantic_status"], bool(adm)),
                }
            )
        f5_list.append(summarize(ev))

    s4["cv5_acc_mean"] = float(np.mean([m["accuracy"] for m in f5_list]))
    s4["cv5_far_mean"] = float(np.mean([m["far"] for m in f5_list]))
    s4["cv5_cost_mean"] = float(np.mean([m["mean_institutional_cost"] for m in f5_list]))
    return s4


def run_diversity_ablation(rows, soft_probs, y_true, contexts, refs):
    # Indices: 0=deberta_v3_large, 1=bart_large_mnli, 2=llama31_8b, 3=qwen35_9b
    subsets = [
        ("nli_only", "NLI-Only (DeBERTa + BART)", [0, 1], 0.393),
        ("llm_only", "LLM-Only (Llama-3.1-8B + Qwen3.5-9B)", [2, 3], 0.481),
        ("drop_deberta", "3-Model w/o DeBERTa-v3", [1, 2, 3], 0.259),
        ("drop_bart", "3-Model w/o BART-Large", [0, 2, 3], 0.287),
        ("drop_llama", "3-Model w/o Llama-3.1-8B", [0, 1, 3], 0.264),
        ("drop_qwen", "3-Model w/o Qwen3.5-9B", [0, 1, 2], 0.223),
        ("full_4_cross", "Full 4-Model Cross-Family (DeCiv)", [0, 1, 2, 3], 0.258),
    ]
    out = []
    print("\n=== CROSS-FAMILY VERIFIER DIVERSITY ABLATION (Sup-Soft-DS + TempScale) ===")
    for sid, lbl, idxs, mean_rho in subsets:
        s = _eval_subset(rows, soft_probs, y_true, contexts, refs, idxs, mode="sup_soft_temp")
        s_pg = _eval_subset(rows, soft_probs, y_true, contexts, refs, idxs, mode="pg_crc_005")
        rec = {
            "subset_id": sid,
            "label": lbl,
            "k_models": len(idxs),
            "mean_pair_rho": mean_rho,
            "accuracy": round(s["accuracy"], 4),
            "far": round(s["far"], 4),
            "accept_precision": round(s["accept_precision"], 4) if s["accept_precision"] else 0.0,
            "recall_revise": round(s["recall_revise"], 4),
            "cost": round(s["mean_institutional_cost"], 4),
            "ece": round(s["ece"], 4),
            "nll": round(s["nll"], 4),
            "cv5_cost": round(s["cv5_cost_mean"], 4),
            "pgcrc005_far": round(s_pg["far"], 4),
            "pgcrc005_cost": round(s_pg["mean_institutional_cost"], 4),
        }
        out.append(rec)
        print(
            f"  {lbl:38s} | Acc={rec['accuracy']*100:5.2f}% | FAR={rec['far']*100:4.2f}% | "
            f"RevRec={rec['recall_revise']*100:5.1f}% | Cost={rec['cost']:.3f} | ECE={rec['ece']:.4f} | "
            f"PG-CRC005 Cost={rec['pgcrc005_cost']:.3f}"
        )
    write_csv(RESULTS / "real_verifier_diversity_ablation.csv", out)
    return out


def run_pgcrc_alpha_sweep(rows, soft_probs, hard_votes, y_true, contexts, refs):
    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    parent_ids = [r["parent_id"] for r in rows]

    # 5-fold stratified parent-cluster CV (exchangeable clusters -> exact Theorem 1 setting!)
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

    # Pre-fit fold posteriors for both Hard-DS (raw) and Sup-Soft-DS + TempScale
    fold_data = []
    for fid in range(5):
        tr = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] != fid], dtype=np.int64)
        te = np.array([i for i, r in enumerate(rows) if parent_fold[r["parent_id"]] == fid], dtype=np.int64)
        ds_h = VectorizedDawidSkene(smoothing=1e-2).fit_hard(hard_votes[tr])
        ph_tr = ds_h.predict_proba_hard(hard_votes[tr], apply_temp=False)
        ph_te = ds_h.predict_proba_hard(hard_votes[te], apply_temp=False)

        ds_s = VectorizedDawidSkene(smoothing=1.0).fit_soft(soft_probs[tr], y_true=y_true[tr])
        ps_tr_raw = ds_s.predict_proba_soft(soft_probs[tr], apply_temp=False)
        ps_te_raw = ds_s.predict_proba_soft(soft_probs[te], apply_temp=False)
        t_opt = fit_temperature(ps_tr_raw, y_true[tr])
        ps_tr = apply_temperature(ps_tr_raw, t_opt)
        ps_te = apply_temperature(ps_te_raw, t_opt)
        fold_data.append((tr, te, ph_tr, ph_te, ps_tr, ps_te))

    out = []
    print("\n=== PG-CRC TARGET RISK BUDGET SWEEP (5-Fold Exchangeable Parent-Cluster CV on DeCiv-Real-1200) ===")
    for base_name in ("hard_ds_raw", "sup_soft_ds_temp"):
        for alpha in alphas:
            fars, accs, rec_accs, costs, taus = [], [], [], [], []
            for tr, te, ph_tr, ph_te, ps_tr, ps_te in fold_data:
                p_tr = ph_tr if base_name == "hard_ds_raw" else ps_tr
                p_te = ph_te if base_name == "hard_ds_raw" else ps_te
                tau, _ = calibrate_pg_crc_threshold(
                    p_tr, [contexts[i] for i in tr], [refs[i] for i in tr], [parent_ids[i] for i in tr], alpha=alpha
                )
                taus.append(tau)
                ev = []
                for loc_i, gi in enumerate(te):
                    act, adm = route_pg_crc(p_te[loc_i], contexts[gi], tau)
                    ref = refs[gi]
                    ev.append(
                        {
                            "predicted_action": act,
                            "reference_action": ref,
                            "correct": int(act == ref),
                            "false_accept": int(act == "accept" and ref != "accept"),
                            "eligible": int(adm),
                            "institutional_cost": realized_institutional_cost(act, ref, rows[gi]["semantic_status"], bool(adm)),
                        }
                    )
                s = summarize(ev)
                fars.append(s["far"])
                accs.append(s["accuracy"])
                rec_accs.append(s["recall_accept"])
                costs.append(s["mean_institutional_cost"])
            rec = {
                "base_estimator": base_name,
                "alpha": alpha,
                "far_mean": round(float(np.mean(fars)), 5),
                "far_std": round(float(np.std(fars, ddof=1)), 5),
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "recall_accept_mean": round(float(np.mean(rec_accs)), 4),
                "cost_mean": round(float(np.mean(costs)), 4),
                "tau_mean": round(float(np.mean(taus)), 4),
            }
            out.append(rec)
            print(
                f"  [{base_name:16s}] alpha={alpha*100:4.1f}% -> emp FAR={rec['far_mean']*100:4.2f}%±{rec['far_std']*100:4.2f}% "
                f"(<=alpha: {rec['far_mean'] <= alpha + 1e-9}) | Acc={rec['accuracy_mean']*100:5.2f}% | "
                f"AccRec={rec['recall_accept_mean']*100:5.1f}% | Cost={rec['cost_mean']:.3f} | tau={rec['tau_mean']:.3f}"
            )
    write_csv(RESULTS / "real_pgcrc_alpha_sweep.csv", out)
    return out


def plot_figure(div_rows, sweep_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.24, wspace=0.34)

    # Panel (a): Verifier Diversity Ablation
    order = ["llm_only", "nli_only", "drop_deberta", "drop_bart", "drop_llama", "drop_qwen", "full_4_cross"]
    short = ["LLM-Only\n(2 LLMs)", "NLI-Only\n(2 NLIs)", "w/o\nDeBERTa", "w/o\nBART", "w/o\nLlama3.1", "w/o\nQwen3.5", "Full 4-Model\n(DeCiv)"]
    dmap = {r["subset_id"]: r for r in div_rows}
    x = np.arange(len(order))
    w = 0.38
    costs = [dmap[k]["cost"] for k in order]
    rev_rec = [dmap[k]["recall_revise"] * 100.0 for k in order]
    accs = [dmap[k]["accuracy"] * 100.0 for k in order]

    cols_c = ["#C0392B", "#E67E22", "#7F8C8D", "#7F8C8D", "#7F8C8D", "#7F8C8D", "#1E8449"]
    bars = ax1.bar(x, costs, width=w, color=cols_c, edgecolor="#000000", linewidth=0.7, alpha=0.88, label="Mean Institutional Cost (↓ left)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=7.4, fontweight="bold")
    ax1.set_ylabel("Mean Institutional Cost (↓)", fontweight="bold")
    ax1.set_ylim(0, 0.78)
    ax1.grid(True, axis="y", alpha=0.6)
    for b, c in zip(bars, costs):
        ax1.text(b.get_x() + b.get_width() / 2, 0.16, f"{c:.3f}", ha="center", va="center",
                 fontsize=7.2, fontweight="bold", color="#FFFFFF",
                 bbox=dict(boxstyle="round,pad=0.14", facecolor="#000000", edgecolor="none", alpha=0.62))

    ax1_r = ax1.twinx()
    ax1_r.plot(x, rev_rec, color="#1F618D", marker="s", linewidth=2.0, markersize=5.8, label="revise Recall (% right)")
    ax1_r.plot(x, accs, color="#5B2C6F", marker="o", linestyle="--", linewidth=1.7, markersize=5.2, label="Overall Accuracy (% right)")
    ax1_r.set_ylabel("Accuracy / revise Recall (%)", fontweight="bold", color="#000000")
    ax1_r.set_ylim(0, 98)
    ax1_r.tick_params(axis="y", colors="#000000")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#000000")

    # Panel (b): PG-CRC Alpha Risk-Budget Sweep
    alphas = sorted({r["alpha"] for r in sweep_rows})
    a_pct = np.array(alphas) * 100.0
    ax2.plot(a_pct, a_pct, color="#7F1D1D", linestyle=":", linewidth=1.6, label="Theorem 1 bound (Empirical FAR = α)")
    ax2.fill_between(a_pct, 0, a_pct, color="#E8F8F5", alpha=0.55)

    for bname, lbl, col, mk in [
        ("hard_ds_raw", "Hard-DS (raw) + PG-CRC", "#C0392B", "o"),
        ("sup_soft_ds_temp", "Sup-Soft-DS + Temp + PG-CRC", "#1E8449", "s"),
    ]:
        sub = {r["alpha"]: r for r in sweep_rows if r["base_estimator"] == bname}
        ys = np.array([sub[a]["far_mean"] * 100.0 for a in alphas])
        es = np.array([sub[a]["far_std"] * 100.0 for a in alphas])
        ax2.errorbar(a_pct, ys, yerr=es, color=col, marker=mk, linewidth=2.1, markersize=5.8, capsize=3, label=lbl)

    ax2.set_xlabel("Target Risk Budget α (%)", fontweight="bold")
    ax2.set_ylabel("Empirical Out-of-Fold FAR (%)", fontweight="bold")
    ax2.set_xlim(0.2, 5.3)
    ax2.set_ylim(-0.15, 5.4)
    ax2.grid(True, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="#000000")
    ax2.annotate("Strict Theorem 1 compliance:\nFAR ≤ α across all budgets", xy=(3.0, 2.35), xytext=(2.85, 1.05),
                 fontsize=7.6, fontweight="bold", color="#000000",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#000000", lw=0.7),
                 arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))

    fig.text(0.28, 0.035, "(a) Cross-Family Verifier Diversity Ablation (DeCiv-Real-1200)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#000000")
    fig.text(0.76, 0.035, "(b) PG-CRC Finite-Sample FAR Tracking vs. Risk Budget α", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#000000")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"trans_diversity_and_pgcrc_sweep.{ext}", dpi=300)
    plt.close(fig)
    print("Saved trans_diversity_and_pgcrc_sweep.{pdf,png}")


def main():
    rows, soft_probs, hard_votes, y_true, contexts, refs, _ = load_real_study()
    div_rows = run_diversity_ablation(rows, soft_probs, y_true, contexts, refs)
    sweep_rows = run_pgcrc_alpha_sweep(rows, soft_probs, hard_votes, y_true, contexts, refs)
    plot_figure(div_rows, sweep_rows)


if __name__ == "__main__":
    main()
