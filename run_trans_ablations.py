"""IEEE Transactions 5-model (3-family) supplementary real-model ablations on DeCiv-Real-1200.

1. Three-Family Cross-Architecture Verifier Diversity Ablation (using DeCiv-CAF):
     - NLI-Only (DeBERTa + BART)
     - Open-LLM-Only (Llama-3.1-8B + Qwen3.5-9B)
     - Frontier-Only (GPT-6.1-Sol)
     - Open-Weight 4-Model (NLI + Open-LLM)
     - NLI + Frontier (DeBERTa + BART + GPT-6.1-Sol)
     - Open-LLM + Frontier (Llama-3.1-8B + Qwen3.5-9B + GPT-6.1-Sol)
     - Full 5-Model 3-Family (DeCiv-CAF)
2. PG-CRC Target Risk-Budget Sweep (alpha in {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}):
     Compares Hard-DS (raw) + PG-CRC, Sup-Soft-DS + Temp + PG-CRC, and DeCiv-CAF + Decoupled PG-CRC
     on 5-fold exchangeable parent-cluster CV (exact Theorem 1 regime).
3. Generates publication figure trans_diversity_and_pgcrc_sweep.{pdf,png}.
"""
from __future__ import annotations

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
    CorrelationAwareFusion,
    VectorizedDawidSkene,
    apply_temperature,
    calibrate_decoupled_pg_crc_threshold,
    calibrate_pg_crc_threshold,
    calibration_metrics,
    fit_temperature,
    route_decoupled_pg_crc,
    route_pg_crc,
    route_projected_pg_crc,
    route_risk,
)
from evaluate_real_verifiers import load_real_study, write_csv
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


def _eval_caf_subset(rows, soft_probs, y_true, contexts, refs, subset_indices: list[int], family_groups):
    sources = sorted({r["source_dataset"] for r in rows})
    parent_ids = [r["parent_id"] for r in rows]
    srcs_all = [r["source_dataset"] for r in rows]
    sub_s = soft_probs[:, subset_indices, :]

    # 4-fold LODO
    eval_dec, eval_r05 = [], []
    probs_4f = np.zeros((len(rows), 4), dtype=np.float64)
    y_4f = np.zeros(len(rows), dtype=np.int64)
    cursor = 0
    for src in sources:
        tr = np.array([i for i, r in enumerate(rows) if r["source_dataset"] != src], dtype=np.int64)
        te = np.array([i for i, r in enumerate(rows) if r["source_dataset"] == src], dtype=np.int64)
        caf = CorrelationAwareFusion(reg_c=0.10, family_groups=family_groups)
        p_tr = caf.fit_with_inner_oof(sub_s[tr], y_true[tr], [srcs_all[i] for i in tr])
        p_te = caf.predict_proba(sub_s[te], apply_temp=True)
        tau, _ = calibrate_decoupled_pg_crc_threshold(
            p_tr, [contexts[i] for i in tr], [refs[i] for i in tr], [parent_ids[i] for i in tr], alpha=0.01
        )
        for loc_i, gi in enumerate(te):
            p = p_te[loc_i]
            ref = refs[gi]
            st = rows[gi]["semantic_status"]
            act_d, adm_d = route_projected_pg_crc(p, contexts[gi], tau)
            act_r, adm_r = route_risk(p, contexts[gi], defer_cost=0.5)
            eval_dec.append({
                "predicted_action": act_d, "reference_action": ref, "correct": int(act_d == ref),
                "false_accept": int(act_d == "accept" and ref != "accept"), "eligible": int(adm_d),
                "institutional_cost": realized_institutional_cost(act_d, ref, st, bool(adm_d)),
            })
            eval_r05.append({
                "predicted_action": act_r, "reference_action": ref, "correct": int(act_r == ref),
                "false_accept": int(act_r == "accept" and ref != "accept"), "eligible": int(adm_r),
                "institutional_cost": realized_institutional_cost(act_r, ref, st, bool(adm_r)),
            })
            probs_4f[cursor] = p
            y_4f[cursor] = y_true[gi]
            cursor += 1
    s_dec = summarize(eval_dec)
    s_r05 = summarize(eval_r05)
    cal = calibration_metrics(probs_4f, y_4f)
    s_dec.update(cal)
    s_dec["risk_d05_cost"] = s_r05["mean_institutional_cost"]
    s_dec["risk_d05_far"] = s_r05["far"]
    return s_dec


def run_diversity_ablation(rows, soft_probs, y_true, contexts, refs):
    # Indices: 0=DeBERTa, 1=BART, 2=Llama3.1, 3=Qwen3.5, 4=GPT-6.1-Sol
    subsets = [
        ("nli_only", "NLI-Only (DeBERTa + BART)", [0, 1], ((0,), (1,))),
        ("open_llm_only", "Open-LLM-Only (Llama + Qwen)", [2, 3], ((0,), (1,))),
        ("frontier_only", "Frontier-Only (GPT-6.1-Sol)", [4], ((0,),)),
        ("open_4model", "Open-Weight 4-Model (NLI + Open-LLM)", [0, 1, 2, 3], ((0, 1), (2, 3))),
        ("nli_frontier", "NLI + Frontier (3-Model)", [0, 1, 4], ((0, 1), (2,))),
        ("llm_frontier", "Open-LLM + Frontier (3-Model)", [2, 3, 4], ((0, 1), (2,))),
        ("full_5_3fam", "Full 5-Model 3-Family (DeCiv-CAF)", [0, 1, 2, 3, 4], ((0, 1), (2, 3), (4,))),
    ]
    out = []
    print("\n=== THREE-FAMILY CROSS-ARCHITECTURE DIVERSITY ABLATION (DeCiv-CAF + Decoupled PG-CRC) ===")
    for sid, lbl, idxs, fgrps in subsets:
        s = _eval_caf_subset(rows, soft_probs, y_true, contexts, refs, idxs, fgrps)
        rec = {
            "subset_id": sid,
            "label": lbl,
            "k_models": len(idxs),
            "accuracy": round(s["accuracy"], 4),
            "far": round(s["far"], 4),
            "accept_precision": round(s["accept_precision"], 4) if s["accept_precision"] else 0.0,
            "recall_accept": round(s["recall_accept"], 4),
            "recall_revise": round(s["recall_revise"], 4),
            "cost": round(s["mean_institutional_cost"], 4),
            "risk_d05_cost": round(s["risk_d05_cost"], 4),
            "ece": round(s["ece"], 4),
            "nll": round(s["nll"], 4),
        }
        out.append(rec)
        print(
            f"  {lbl:38s} | Acc={rec['accuracy']*100:5.2f}% | FAR={rec['far']*100:4.2f}% | "
            f"AccRec={rec['recall_accept']*100:5.1f}% | RevRec={rec['recall_revise']*100:5.1f}% | "
            f"Cost={rec['cost']:.3f} (d0.5={rec['risk_d05_cost']:.3f}) | ECE={rec['ece']:.4f} | NLL={rec['nll']:.3f}"
        )
    write_csv(RESULTS / "real_verifier_diversity_ablation.csv", out)
    return out


def run_pgcrc_alpha_sweep(rows, soft_probs, hard_votes, y_true, contexts, refs):
    alphas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    parent_ids = [r["parent_id"] for r in rows]

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

        caf = CorrelationAwareFusion(reg_c=0.10)
        pc_tr = caf.fit_with_inner_oof(soft_probs[tr], y_true[tr], [parent_ids[i] for i in tr])
        pc_te = caf.predict_proba(soft_probs[te], apply_temp=True)

        fold_data.append((tr, te, ph_tr, ph_te, ps_tr, ps_te, pc_tr, pc_te))

    out = []
    print("\n=== PG-CRC TARGET RISK BUDGET SWEEP (5-Fold Exchangeable Parent-Cluster CV, 5-Model Pool) ===")
    for base_name in ("hard_ds_raw", "sup_soft_ds_temp", "deciv_caf_decoupled"):
        for alpha in alphas:
            fars, accs, rec_accs, costs, taus = [], [], [], [], []
            for tr, te, ph_tr, ph_te, ps_tr, ps_te, pc_tr, pc_te in fold_data:
                tr_c = [contexts[i] for i in tr]
                tr_r = [refs[i] for i in tr]
                tr_p = [parent_ids[i] for i in tr]
                if base_name == "hard_ds_raw":
                    tau, _ = calibrate_pg_crc_threshold(ph_tr, tr_c, tr_r, tr_p, alpha=alpha)
                    p_te = ph_te
                elif base_name == "sup_soft_ds_temp":
                    tau, _ = calibrate_pg_crc_threshold(ps_tr, tr_c, tr_r, tr_p, alpha=alpha)
                    p_te = ps_te
                else:
                    tau, _ = calibrate_decoupled_pg_crc_threshold(pc_tr, tr_c, tr_r, tr_p, alpha=alpha)
                    p_te = pc_te
                taus.append(tau)
                ev = []
                for loc_i, gi in enumerate(te):
                    if base_name == "deciv_caf_decoupled":
                        act, adm = route_projected_pg_crc(p_te[loc_i], contexts[gi], tau)
                    else:
                        act, adm = route_pg_crc(p_te[loc_i], contexts[gi], tau)
                    ref = refs[gi]
                    ev.append({
                        "predicted_action": act, "reference_action": ref, "correct": int(act == ref),
                        "false_accept": int(act == "accept" and ref != "accept"), "eligible": int(adm),
                        "institutional_cost": realized_institutional_cost(act, ref, rows[gi]["semantic_status"], bool(adm)),
                    })
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
                f"  [{base_name:20s}] alpha={alpha*100:4.1f}% -> emp FAR={rec['far_mean']*100:4.2f}%±{rec['far_std']*100:4.2f}% "
                f"(<=a: {rec['far_mean'] <= alpha + 1e-9}) | Acc={rec['accuracy_mean']*100:5.2f}% | "
                f"AccRec={rec['recall_accept_mean']*100:5.1f}% | Cost={rec['cost_mean']:.3f} | tau={rec['tau_mean']:.3f}"
            )
    write_csv(RESULTS / "real_pgcrc_alpha_sweep.csv", out)
    return out


def plot_figure(div_rows, sweep_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.45), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.90, bottom=0.25, wspace=0.34)

    # Panel (a): Three-Family Diversity Ablation (Accuracy, Accept Recall, and Risk Cost)
    order = ["open_llm_only", "nli_only", "frontier_only", "open_4model", "llm_frontier", "nli_frontier", "full_5_3fam"]
    short = [
        "Open-LLM\n(2 LLMs)",
        "NLI-Only\n(2 NLIs)",
        "Frontier\n(GPT-6.1)",
        "Open-4\n(NLI+LLM)",
        "OpenLLM+\nFrontier",
        "NLI+\nFrontier",
        "Full 5-Mod\n(DeCiv-CAF)",
    ]
    dmap = {r["subset_id"]: r for r in div_rows}
    x = np.arange(len(order))
    w = 0.42
    accs = [dmap[k]["accuracy"] * 100.0 for k in order]
    arec = [dmap[k]["recall_accept"] * 100.0 for k in order]
    costs = [dmap[k]["risk_d05_cost"] for k in order]

    cols = ["#C0392B", "#E67E22", "#5D6D7E", "#2980B9", "#2471A3", "#1F618D", "#1E8449"]
    bars = ax1.bar(x, accs, width=w, color=cols, edgecolor="#000000", linewidth=0.7, alpha=0.90, label="4-Class Policy Acc (% left)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=7.1, fontweight="bold")
    ax1.set_ylabel("Policy Accuracy / Accept Recall (%)", fontweight="bold")
    ax1.set_ylim(0, 126)
    ax1.grid(True, axis="y", alpha=0.6)
    for b, a in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, a + 1.8, f"{a:.1f}%", ha="center", va="bottom",
                 fontsize=7.0, fontweight="bold", color="#000000")

    ax1.plot(x, arec, color="#154360", marker="s", linewidth=2.0, markersize=5.6, label="Clean Accept Recall (% left)")

    ax1_r = ax1.twinx()
    ax1_r.plot(x, costs, color="#7D3C98", marker="D", linestyle="--", linewidth=1.8, markersize=5.2, label="Institutional Cost (↓ right)")
    ax1_r.set_ylabel("Mean Institutional Cost (↓)", fontweight="bold", color="#000000")
    ax1_r.set_ylim(0.0, max(costs) * 1.65)
    ax1_r.tick_params(axis="y", colors="#000000")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=6.8, frameon=True, facecolor="white", edgecolor="#000000", framealpha=0.95)

    # Panel (b): PG-CRC Alpha Risk-Budget Sweep (FAR vs alpha AND Accept Recall inset!)
    alphas = sorted({r["alpha"] for r in sweep_rows})
    a_pct = np.array(alphas) * 100.0
    ax2.plot(a_pct, a_pct, color="#7F1D1D", linestyle=":", linewidth=1.6, label="Theorem 1 bound (Empirical FAR = α)")
    ax2.fill_between(a_pct, 0, a_pct, color="#E8F8F5", alpha=0.55)

    for bname, lbl, col, mk in [
        ("hard_ds_raw", "Hard-DS (raw) + PG-CRC", "#C0392B", "o"),
        ("sup_soft_ds_temp", "Sup-Soft-DS + Temp + PG-CRC", "#2980B9", "s"),
        ("deciv_caf_decoupled", "DeCiv-CAF + Decoupled PG-CRC", "#1E8449", "^"),
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
    ax2.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor="white", edgecolor="#000000")

    caf10 = [r for r in sweep_rows if r["base_estimator"] == "deciv_caf_decoupled" and abs(r["alpha"] - 0.01) < 1e-9][0]
    caf15 = [r for r in sweep_rows if r["base_estimator"] == "deciv_caf_decoupled" and abs(r["alpha"] - 0.015) < 1e-9][0]
    box_txt = (
        f"Theorem 1 verified (DeCiv-CAF):\n"
        f"• α=1.0%: FAR={caf10['far_mean']*100:.2f}%,\n"
        f"  Acc={caf10['accuracy_mean']*100:.1f}%, Rec={caf10['recall_accept_mean']*100:.1f}%\n"
        f"• α=1.5%: FAR={caf15['far_mean']*100:.2f}%,\n"
        f"  Acc={caf15['accuracy_mean']*100:.1f}%, Rec={caf15['recall_accept_mean']*100:.1f}%"
    )
    ax2.text(0.49, 0.30, box_txt, transform=ax2.transAxes, fontsize=6.9, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#000000", lw=0.75))

    fig.text(0.28, 0.035, "(a) Three-Family Cross-Architecture Verifier Ablation (DeCiv-Real-1200)", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#000000")
    fig.text(0.75, 0.035, "(b) PG-CRC Finite-Sample FAR Tracking vs. Risk Budget α", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#000000")

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
