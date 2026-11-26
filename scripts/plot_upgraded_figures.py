"""Generate unified publication-grade IEEE Transactions figures (300 DPI PNG + vector PDF).

Refined Visual & Typographic Discipline:
  - Restrained Muted Academic Palette ONLY (zero pure RGB #FF0000/#00FF00/#0000FF, zero rainbow/Jet, zero neon)
  - Delicate, crisp stroke weights (axes 0.65 pt, curves 1.15-1.35 pt, bar edges 0.50 pt, inset frames 0.75 pt)
  - Pure black (#000000) DejaVu Sans typography across ALL axes, ticks, legends, annotations, and sub-captions
  - Sleek full-width (\textwidth) aspect ratios engineered for flush alignment with full-width tables
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Unified Restrained Muted Academic Palette (Identical across all 6 data figures)
C_TEAL = "#3D8C7D"      # DeCiv Primary (Muted Sage Teal)
C_NAVY = "#2B5B84"      # DeCiv Secondary / Calibrated (Deep Academic Navy)
C_STEEL = "#7FA6C9"     # Gated Baseline / 2-Family (Soft Steel Blue)
C_CORAL = "#C05A46"     # Danger / Collapse / Inset Frame (Muted Terracotta)
C_OCHRE = "#C48F3B"     # Secondary Warning / Adaptive (Muted Warm Ochre)
C_PLUM = "#7A5C87"      # Fitted Temperature / Auxiliary (Muted Plum)
C_SLATE = "#98A2AD"     # Un-Gated / Single-Family Neutral (Soft Slate Gray)
C_BLACK = "#000000"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.8,
        "axes.edgecolor": C_BLACK,
        "axes.labelcolor": C_BLACK,
        "axes.titlecolor": C_BLACK,
        "xtick.color": C_BLACK,
        "ytick.color": C_BLACK,
        "text.color": C_BLACK,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "grid.color": "#D4DAE0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_legend(leg) -> None:
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor(C_BLACK)


def plot_real_verifiers_and_correlation() -> None:
    corr_rows = read_csv("real_verifier_correlation.csv")
    adm_rows = {r["method_id"]: r for r in read_csv("real_admission_metrics.csv")}

    fig = plt.figure(figsize=(13.2, 4.35), dpi=300)
    ax1 = fig.add_axes([0.095, 0.24, 0.275, 0.68])
    cax = fig.add_axes([0.382, 0.24, 0.013, 0.68])
    ax2 = fig.add_axes([0.640, 0.24, 0.345, 0.68])

    # --- Panel (a): 5x5 Error Correlation Matrix (Harmonized Academic PuBu Sequential Map) ---
    vids = ["deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b", "gpt61_sol"]
    vlabels = [
        "DeBERTa-v3\n(Enc NLI)",
        "BART-Large\n(Enc-Dec NLI)",
        "Llama-3.1-8B\n(Open LLM)",
        "Qwen3.5-9B\n(Open LLM)",
        "GPT-6.1-Sol\n(Frontier)",
    ]
    mat = np.zeros((5, 5), dtype=float)
    lookup = {(r["verifier_i"], r["verifier_j"]): float(r["error_pearson_rho"]) for r in corr_rows}
    for i, vi in enumerate(vids):
        for j, vj in enumerate(vids):
            mat[i, j] = lookup[(vi, vj)]

    im = ax1.imshow(mat, cmap="PuBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(vlabels, fontsize=7.1, fontweight="bold", rotation=20, ha="right", color=C_BLACK)
    ax1.set_yticklabels(vlabels, fontsize=7.1, fontweight="bold", color=C_BLACK)

    for i in range(5):
        for j in range(5):
            val = mat[i, j]
            txt = "1.00" if i == j else f"{val:.3f}"
            ax1.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold",
                color=C_BLACK if val < 0.62 else "#FFFFFF",
            )

    for b in (1.5, 3.5):
        ax1.axhline(b, color=C_BLACK, linewidth=0.95, linestyle="--")
        ax1.axvline(b, color=C_BLACK, linewidth=0.95, linestyle="--")
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(r"Error Pearson Correlation $\hat{\rho}$", fontsize=8.0, fontweight="bold", color=C_BLACK)
    cbar.ax.tick_params(labelsize=7.6, colors=C_BLACK, width=0.55)

    # --- Panel (b): 3-Tier Semantic Muted Bars ---
    show = [
        ("solo_llama31_8b_direct", "Solo Llama-3.1-8B / direct", C_SLATE),
        ("majority_direct", "Majority Vote / direct", C_SLATE),
        ("conformal_risk", "Conformal-Risk (ICLR '24)", C_SLATE),
        ("solo_gpt61_sol_direct", "Solo Frontier GPT-6.1 / direct", C_SLATE),
        ("conformal_risk_gated", "Conformal-Risk + Gate", C_STEEL),
        ("gbdt_stack_gated", "GBDT Stacking + Gate", C_STEEL),
        ("solo_gpt61_sol_gated", "Solo Frontier GPT-6.1 + Gate", C_STEEL),
        ("deciv_caf_pg_crc_010", r"DeCiv-CAF + PG-CRC ($\alpha$=0.01)", C_NAVY),
        ("deciv_caf_gated_map", "DeCiv-CAF / projected MAP", C_TEAL),
    ]
    y_pos = np.arange(len(show))
    accs = [float(adm_rows[m]["accuracy"]) * 100.0 for m, _, _ in show]
    fars = [float(adm_rows[m]["far"]) * 100.0 for m, _, _ in show]
    costs = [float(adm_rows[m]["mean_institutional_cost"]) for m, _, _ in show]
    cols = [c for _, _, c in show]
    labels = [l for _, l, _ in show]

    bars = ax2.barh(y_pos, accs, color=cols, edgecolor=C_BLACK, linewidth=0.5, height=0.64)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7.6, fontweight="bold", color=C_BLACK)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"4-Class Policy Accuracy ($\uparrow$ %) with FAR & Cost", fontsize=8.6, fontweight="bold", color=C_BLACK)
    ax2.set_xlim(0, 162)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, axis="x", alpha=0.65)
    ax2.set_axisbelow(True)

    ax2.axhline(3.5, color=C_BLACK, lw=0.55, ls=":", alpha=0.55)
    ax2.axhline(6.5, color=C_BLACK, lw=0.55, ls=":", alpha=0.55)

    for bar, acc, far, cost in zip(bars, accs, fars, costs):
        ax2.text(
            acc + 1.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{acc:.1f}% (FAR {far:.1f}%, C={cost:.2f})",
            va="center",
            ha="left",
            fontsize=6.9,
            fontweight="bold",
            color=C_BLACK,
        )

    fig.text(
        0.240,
        0.032,
        "(a) 5-Model 3-Family Error Correlation Matrix (DeCiv-Real-1200)",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.765,
        0.032,
        "(b) 5-Model Policy Accuracy, FAR & Cost (4-Fold LODO CV)",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"real_verifiers_and_correlation.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined real_verifiers_and_correlation.{pdf,png}")


def plot_diversity_and_pgcrc_sweep() -> None:
    div_rows = read_csv("real_verifier_diversity_ablation.csv")
    sweep_rows = read_csv("real_pgcrc_alpha_sweep.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.35), dpi=300)
    fig.subplots_adjust(left=0.060, right=0.942, top=0.84, bottom=0.22, wspace=0.29)

    order = ["open_llm_only", "nli_only", "frontier_only", "open_4model", "llm_frontier", "nli_frontier", "full_5_3fam"]
    short = [
        "Open-LLM\n(1 Fam)",
        "NLI-Only\n(1 Fam)",
        "Frontier\n(1 Fam)",
        "Open-4\n(2 Fam)",
        "OpenLLM+\nFrontier",
        "NLI+\nFrontier",
        "Full 5-Mod\n(DeCiv-CAF)",
    ]
    dmap = {r["subset_id"]: r for r in div_rows}
    x = np.arange(len(order))
    w = 0.44
    accs = [float(dmap[k]["accuracy"]) * 100.0 for k in order]
    arec = [float(dmap[k]["recall_accept"]) * 100.0 for k in order]
    costs = [float(dmap[k]["risk_d05_cost"]) for k in order]

    cols = [C_SLATE, C_SLATE, C_SLATE, C_STEEL, C_STEEL, C_STEEL, C_TEAL]
    bars = ax1.bar(x, accs, width=w, color=cols, edgecolor=C_BLACK, linewidth=0.5, alpha=0.94, label="4-Class Policy Acc (% left)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=7.0, fontweight="bold", color=C_BLACK)
    ax1.set_ylabel("Policy Accuracy / Accept Recall (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_ylim(0, 125)
    ax1.tick_params(labelsize=7.8)
    ax1.grid(True, axis="y", alpha=0.65)
    ax1.set_axisbelow(True)
    for b, a in zip(bars, accs):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            a + 1.8,
            f"{a:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold",
            color=C_BLACK,
        )

    ax1.plot(x, arec, color=C_NAVY, marker="s", linewidth=1.30, markersize=4.4, markeredgewidth=0.5, label="Clean Accept Recall (% left)")
    ax1_r = ax1.twinx()
    ax1_r.plot(x, costs, color=C_CORAL, marker="D", linestyle="--", linewidth=1.20, markersize=4.2, markeredgewidth=0.5, label=r"Institutional Cost ($\downarrow$ right)")
    ax1_r.set_ylabel(r"Mean Institutional Cost ($\downarrow$)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1_r.set_ylim(0.0, max(costs) * 1.65)
    ax1_r.tick_params(axis="y", labelsize=7.8, colors=C_BLACK)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    leg1 = ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, fontsize=6.6, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg1)

    alphas = sorted({float(r["alpha"]) for r in sweep_rows})
    a_pct = np.array(alphas) * 100.0
    ax2.plot(a_pct, a_pct, color=C_BLACK, linestyle=":", linewidth=0.95, label=r"Theorem 1 bound ($\mathrm{FAR} = \alpha$)")
    ax2.fill_between(a_pct, 0, a_pct, color="#E8F3F0", alpha=0.55)

    specs_b = [
        ("hard_ds_raw", "Hard-DS (raw) + PG-CRC", C_CORAL, "o", "--"),
        ("sup_soft_ds_temp", "Sup-Soft-DS + Temp + PG-CRC", C_NAVY, "s", "-"),
        ("deciv_caf_decoupled", "DeCiv-CAF + Decoupled PG-CRC", C_TEAL, "^", "-"),
    ]
    for bname, lbl, col, mk, ls in specs_b:
        sub = {float(r["alpha"]): r for r in sweep_rows if r["base_estimator"] == bname}
        ys = np.array([float(sub[a]["far_mean"]) * 100.0 for a in alphas])
        es = np.array([float(sub[a]["far_std"]) * 100.0 for a in alphas])
        ax2.errorbar(
            a_pct,
            ys,
            yerr=es,
            color=col,
            marker=mk,
            linestyle=ls,
            linewidth=1.30,
            elinewidth=0.85,
            capthick=0.7,
            markersize=4.5,
            capsize=2.2,
            label=lbl,
        )

    ax2.set_xlabel(r"Target Risk Budget $\alpha$ (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_ylabel("Empirical Out-of-Fold FAR (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_xlim(0.25, 5.25)
    ax2.set_ylim(-0.2, 6.25)
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, alpha=0.65)
    leg2 = ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        fontsize=6.6,
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.96,
    )
    style_legend(leg2)

    axins = inset_axes(ax2, width="42%", height="34%", loc="upper left", borderpad=1.45)
    axins.plot([0.4, 1.7], [0.4, 1.7], color=C_BLACK, linestyle=":", linewidth=0.85)
    axins.fill_between([0.4, 1.7], 0, [0.4, 1.7], color="#E8F3F0", alpha=0.55)
    for bname, _, col, mk, ls in specs_b:
        sub = {float(r["alpha"]): r for r in sweep_rows if r["base_estimator"] == bname}
        ys = np.array([float(sub[a]["far_mean"]) * 100.0 for a in alphas])
        es = np.array([float(sub[a]["far_std"]) * 100.0 for a in alphas])
        axins.errorbar(a_pct, ys, yerr=es, color=col, marker=mk, linestyle=ls, linewidth=1.05, elinewidth=0.75, capthick=0.65, markersize=3.8, capsize=1.8)

    caf10 = [r for r in sweep_rows if r["base_estimator"] == "deciv_caf_decoupled" and abs(float(r["alpha"]) - 0.01) < 1e-9][0]
    caf15 = [r for r in sweep_rows if r["base_estimator"] == "deciv_caf_decoupled" and abs(float(r["alpha"]) - 0.015) < 1e-9][0]
    axins.annotate(
        f"{float(caf10['far_mean'])*100:.2f}%",
        xy=(1.0, float(caf10["far_mean"]) * 100.0),
        xytext=(0.52, 1.25),
        fontsize=6.3,
        fontweight="bold",
        color=C_BLACK,
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.6),
    )
    axins.annotate(
        f"{float(caf15['far_mean'])*100:.2f}%",
        xy=(1.5, float(caf15["far_mean"]) * 100.0),
        xytext=(1.05, 1.50),
        fontsize=6.3,
        fontweight="bold",
        color=C_BLACK,
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.6),
    )
    axins.set_xlim(0.42, 1.65)
    axins.set_ylim(-0.05, 1.75)
    axins.set_xticks([0.5, 1.0, 1.5])
    axins.set_yticks([0.0, 0.5, 1.0, 1.5])
    axins.tick_params(labelsize=6.4, colors=C_BLACK, width=0.55)
    axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    axins.set_title(r"Zoom: $\alpha \in [0.5\%, 1.5\%]$", fontsize=6.4, fontweight="bold", color=C_BLACK, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor(C_CORAL)
        sp.set_linewidth(0.75)
    mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec=C_CORAL, lw=0.65, ls="--")

    fig.text(
        0.270,
        0.032,
        "(a) Three-Family Cross-Architecture Verifier Ablation (DeCiv-Real-1200)",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.745,
        0.032,
        r"(b) PG-CRC Finite-Sample FAR Tracking vs. Risk Budget $\alpha$",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"trans_diversity_and_pgcrc_sweep.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined trans_diversity_and_pgcrc_sweep.{pdf,png}")


def plot_rag_defense_and_remediation() -> None:
    rag_rows = {r["regime_id"]: r for r in read_csv("real_rag_defense_metrics.csv")}
    rem_recs = [json.loads(l) for l in (RESULTS / "real_remediation_samples.jsonl").open(encoding="utf-8")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.20), dpi=300, gridspec_kw={"width_ratios": [1.48, 1.0]})
    fig.subplots_adjust(left=0.056, right=0.978, top=0.91, bottom=0.25, wspace=0.22)

    regimes = [
        ("no_filter", "No Filter\n(Unfiltered)"),
        ("reliabilityrag_2025", "Reliability\nRAG ('25)"),
        ("seconrag_2025", "SeCon-\nRAG ('25)"),
        ("selfrag_llm_filter", "Self-RAG\nFilter"),
        ("conformal_risk_gated", "Conformal\n+ Gate"),
        ("solo_gpt61_sol_gated", "GPT-6.1\n+ Gate"),
        ("deciv_soft_ds_sup_temp", "DeCiv\nSup-DS"),
        ("deciv_strict_plus_revise", "DeCiv +\nAuto-Revise"),
    ]
    x = np.arange(len(regimes))
    w = 0.255
    safe150 = [float(rag_rows[r]["trustworthy_qa_accuracy_safe150"]) * 100.0 for r, _ in regimes]
    sup100 = [float(rag_rows[r]["downstream_qa_accuracy_sup100"]) * 100.0 for r, _ in regimes]
    asr = [float(rag_rows[r]["downstream_poisoning_asr"]) * 100.0 for r, _ in regimes]
    oracle_safe = float(rag_rows["oracle_clean"]["trustworthy_qa_accuracy_safe150"]) * 100.0

    b1 = ax1.bar(x - w, safe150, width=w, color=C_TEAL, edgecolor=C_BLACK, linewidth=0.45, label=r"Trustworthy QA Safe-150 ($\uparrow$ %)")
    b2 = ax1.bar(x, sup100, width=w, color=C_NAVY, edgecolor=C_BLACK, linewidth=0.45, label=r"Clean Verified QA Sup-100 ($\uparrow$ %)")
    b3 = ax1.bar(x + w, asr, width=w, color=C_CORAL, edgecolor=C_BLACK, linewidth=0.45, label=r"Downstream Poisoning ASR ($\downarrow$ %)")
    ax1.axhline(oracle_safe, color=C_BLACK, linestyle="--", linewidth=0.9, label=f"Oracle Clean Safe-150 ({oracle_safe:.1f}%)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if 85.5 <= h <= 91.5:
                # Place label cleanly inside top of bar so it never collides with the 92.0% dashed Oracle line
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h - 2.2,
                    f"{h:.0f}",
                    ha="center",
                    va="top",
                    fontsize=5.9,
                    fontweight="bold",
                    color="#FFFFFF",
                )
            else:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.3,
                    f"{h:.1f}" if h < 10 else f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.0,
                    fontweight="bold",
                    color=C_BLACK,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in regimes], fontsize=7.1, fontweight="bold", color=C_BLACK)
    ax1.set_ylabel("QA Accuracy / Poisoning ASR (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_ylim(0, 124)
    ax1.tick_params(labelsize=7.8)
    ax1.grid(True, axis="y", alpha=0.65)
    ax1.set_axisbelow(True)
    leg1 = ax1.legend(loc="upper center", ncol=2, fontsize=6.8, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg1)

    readm_nli = [r["nli_ev_supports_repaired"] for r in rem_recs if r["reverified_action"] == "accept"]
    defer_nli = [r["nli_ev_supports_repaired"] for r in rem_recs if r["reverified_action"] != "accept"]

    bins = np.linspace(0.0, 1.0, 11)
    ax2.hist(
        readm_nli,
        bins=bins,
        color=C_TEAL,
        alpha=0.88,
        edgecolor=C_BLACK,
        linewidth=0.5,
        label=f"Re-Admitted (n={len(readm_nli)}, mean={np.mean(readm_nli):.3f})",
    )
    ax2.hist(
        defer_nli,
        bins=bins,
        color=C_CORAL,
        alpha=0.86,
        edgecolor=C_BLACK,
        linewidth=0.5,
        label=f"Blocked / Deferred (n={len(defer_nli)}, mean={np.mean(defer_nli):.3f})",
    )
    ax2.axvline(0.65, color=C_BLACK, linestyle="--", linewidth=0.95, label="Entailment threshold (0.65)")
    ax2.set_xlabel(r"Primary-Evidence Entailment Score $P(\mathrm{Ev} \Rightarrow \mathrm{Repaired})$", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_ylabel("Number of Repaired Claims", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0, max(155, len(readm_nli) + 25))
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, axis="y", alpha=0.65)
    ax2.set_axisbelow(True)
    leg2 = ax2.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg2)

    fig.text(
        0.320,
        0.032,
        "(a) End-to-End Dense RAG Defense vs. 2025 SOTAs (BGE + FAISS + Llama-3.1-8B)",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.798,
        0.032,
        "(b) Automated revise 5-Model Re-Verification Entailment Distribution",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=C_BLACK,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rag_defense_and_remediation.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined rag_defense_and_remediation.{pdf,png}")


def plot_correlated_stress_and_noisy_gate() -> None:
    stress = read_csv("correlated_verifier_stress.csv")
    noisy = read_csv("noisy_gate_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.20), dpi=300)
    fig.subplots_adjust(left=0.060, right=0.940, top=0.91, bottom=0.22, wspace=0.29)

    rhos = sorted({float(r["rho"]) for r in stress})
    series = [
        ("deciv_ds_raw", "DeCiv + Hard-DS (raw)", C_CORAL, "o", "-"),
        ("deciv_adaptive_085", "DeCiv + Adaptive (0.85)", C_OCHRE, "^", "--"),
        ("deciv_pg_crc", r"DeCiv + DS + PG-CRC ($\alpha$=0.01)", C_NAVY, "s", "-"),
        ("deciv_ds_temp", "DeCiv + DS + TempScale", C_TEAL, "D", "-"),
    ]
    for mid, lbl, col, mk, ls in series:
        sub = {float(r["rho"]): r for r in stress if r["method_id"] == mid}
        ys = np.array([float(sub[rh]["far_mean"]) * 100.0 for rh in rhos])
        es = np.array([float(sub[rh]["far_std"]) * 100.0 for rh in rhos])
        ax1.plot(rhos, ys, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=1.30, markersize=4.4, markeredgewidth=0.5)
        ax1.fill_between(rhos, np.maximum(0.0, ys - es), ys + es, color=col, alpha=0.12)

    ax1.axhline(1.0, color=C_BLACK, linestyle=":", linewidth=0.95, label=r"PG-CRC target bound ($\alpha$ = 1.0%)")
    ax1.set_xlabel(r"Pairwise Verifier Error Correlation $\rho$", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_ylabel("False Acceptance Rate FAR (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_xlim(-0.02, 0.62)
    ax1.set_ylim(-0.18, 5.25)
    ax1.tick_params(labelsize=7.8)
    ax1.grid(True, alpha=0.65)

    ax1_r = ax1.twinx()
    sub_t = {float(r["rho"]): float(r["mean_fitted_temp"]) for r in stress if r["method_id"] == "deciv_ds_temp"}
    ts = [sub_t[rh] for rh in rhos]
    ax1_r.plot(rhos, ts, color=C_PLUM, linestyle="-.", marker="x", linewidth=1.15, markersize=4.5, label=r"Fitted Temp $T^*$ (right axis)")
    ax1_r.set_ylabel(r"Fitted Temperature $T^*$ (DS + TempScale)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1_r.set_ylim(0.5, 6.2)
    ax1_r.tick_params(axis="y", labelsize=7.8, colors=C_BLACK)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    leg1 = ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.8, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg1)
    ax1.annotate(
        r"64$\times$ FAR surge" + "\n" + r"(0.07% $\to$ 4.48%)",
        xy=(0.60, 4.48),
        xytext=(0.35, 4.30),
        fontsize=7.2,
        fontweight="bold",
        color=C_BLACK,
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.7),
    )

    axins = inset_axes(ax1, width="37%", height="24%", loc="center right", bbox_to_anchor=(0.0, -0.24, 0.98, 1.0), bbox_transform=ax1.transAxes)
    for mid, _, col, mk, ls in series[2:]:
        sub = {float(r["rho"]): r for r in stress if r["method_id"] == mid}
        ys = np.array([float(sub[rh]["far_mean"]) * 100.0 for rh in rhos])
        axins.plot(rhos, ys, color=col, marker=mk, linestyle=ls, linewidth=1.05, markersize=3.6, markeredgewidth=0.45)
    axins.set_xlim(0.18, 0.62)
    axins.set_ylim(-0.02, 0.14)
    axins.set_xticks([0.2, 0.4, 0.6])
    axins.set_yticks([0.00, 0.05, 0.10])
    axins.tick_params(labelsize=6.1, colors=C_BLACK, width=0.55)
    axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    axins.set_title("Zoom: FAR near 0%", fontsize=6.2, fontweight="bold", color=C_BLACK, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor(C_CORAL)
        sp.set_linewidth(0.75)
    mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec=C_CORAL, lw=0.65, ls="--")

    sub_n = [r for r in noisy if r["method"] == "ds_temp"]
    etas = [float(r["noise_rate"]) * 100.0 for r in sub_n]
    fars = [float(r["far_mean"]) * 100.0 for r in sub_n]
    evrs = [float(r["evr_mean"]) * 100.0 for r in sub_n]
    costs = [float(r["cost_mean"]) for r in sub_n]
    cost_stds = [float(r["cost_std"]) for r in sub_n]

    ax2.plot(etas, fars, color=C_NAVY, marker="o", linewidth=1.30, markersize=4.5, markeredgewidth=0.5, label="False Acceptance Rate FAR (%)")
    ax2.plot(etas, evrs, color=C_CORAL, marker="s", linestyle="--", linewidth=1.30, markersize=4.3, markeredgewidth=0.5, label="Eligibility Violation Rate EVR (%)")
    ax2.set_xlabel(r"Metadata Detector Noise Rate $\eta$ (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_ylabel("Safety Violation Rate (%)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_xlim(-0.4, 10.7)
    ax2.set_ylim(-0.02, 0.42)
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, alpha=0.65)

    ax2_r = ax2.twinx()
    ax2_r.errorbar(etas, costs, yerr=cost_stds, color=C_TEAL, marker="^", linestyle="-.", linewidth=1.20, elinewidth=0.85, capthick=0.7, markersize=4.4, capsize=2.2, label="Mean Institutional Cost (right axis)")
    ax2_r.set_ylabel(r"Mean Institutional Cost ($\downarrow$)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2_r.set_ylim(0.20, 0.30)
    ax2_r.tick_params(axis="y", labelsize=7.8, colors=C_BLACK)

    h1b, l1b = ax2.get_legend_handles_labels()
    h2b, l2b = ax2_r.get_legend_handles_labels()
    leg2 = ax2.legend(h1b + h2b, l1b + l2b, loc="upper left", fontsize=7.0, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg2)
    ax2.annotate(
        r"$\eta$=10%: EVR 0.28%",
        xy=(9.95, 0.28),
        xytext=(5.4, 0.305),
        fontsize=7.2,
        fontweight="bold",
        color=C_BLACK,
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.7),
    )
    ax2.annotate(
        "FAR 0.09%",
        xy=(9.95, 0.09),
        xytext=(6.6, 0.035),
        fontsize=7.2,
        fontweight="bold",
        color=C_BLACK,
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.7),
    )

    fig.text(
        0.265,
        0.032,
        "(a) Correlated-Verifier Collapse & PG-CRC / TempScale Mitigation",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.745,
        0.032,
        "(b) Noisy Metadata Detector Sensitivity (20-Seed Mean ± SD)",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"correlated_stress_and_noisy_gate.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined correlated_stress_and_noisy_gate.{pdf,png}")


def main():
    plot_real_verifiers_and_correlation()
    plot_diversity_and_pgcrc_sweep()
    plot_rag_defense_and_remediation()
    plot_correlated_stress_and_noisy_gate()
    print("ALL REFINED MUTED-ACADEMIC FIGURES GENERATED.")


if __name__ == "__main__":
    main()
