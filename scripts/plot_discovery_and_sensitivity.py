"""Generate publication-grade Figure 7:
  (a) Multi-Copy PoisonedRAG Malicious-Clique Collapse vs. DeCiv m-Copy Invariance
      inside a 10,000-Passage Real Wikipedia + PubMed Haystack (m in {1, 3, 5})
  (b) Real 5-Model DeCiv-Real-1200 Cost-Matrix Sensitivity (d in [0.1, 2.0], C_acc in {10, 18, 30})
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
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


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    mc = read_csv("real_multicopy_haystack_rag.csv")
    sens = read_csv("real_cost_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.45), dpi=300)
    fig.subplots_adjust(left=0.065, right=0.935, top=0.91, bottom=0.22, wspace=0.31)

    # --- Panel (a): Multi-Copy PoisonedRAG (m = 1, 3, 5) in 10,000-Doc Real Haystack ---
    ms = [1, 3, 5]
    series_a = [
        ("no_filter", "No Filter (Unfiltered, 0 ms)", "#7F8C8D", "x", ":"),
        ("reliabilityrag_2025", "ReliabilityRAG (NeurIPS '25, 272 ms/q)", "#C0392B", "o", "-"),
        ("seconrag_2025", "SeCon-RAG (NeurIPS '25, 271 ms/q)", "#D35400", "s", "--"),
        ("conformal_risk_gated", "Conformal-Risk + Gate (0 ms)", "#8E44AD", "^", "-."),
        ("deciv_caf_plus_revise", "DeCiv-CAF + PG-CRC + Revise (0 ms)", "#1F618D", "D", "-"),
        ("deciv_strict_plus_revise", "DeCiv Sup-DS + Auto-Revise (0 ms)", "#1E8449", "P", "-"),
    ]
    lookup_mc = {(int(r["poison_copies_m"]), r["regime_id"]): r for r in mc}
    for rid, lbl, col, mk, ls in series_a:
        ys = [float(lookup_mc[(m, rid)]["downstream_poisoning_asr"]) * 100.0 for m in ms]
        ax1.plot(ms, ys, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=2.15, markersize=6.0)
        if rid in ("reliabilityrag_2025", "seconrag_2025", "no_filter"):
            y_off = 1.6 if rid == "reliabilityrag_2025" else (-1.8 if rid == "seconrag_2025" else 0.0)
            ax1.annotate(
                f"{ys[-1]:.1f}%",
                xy=(5.0, ys[-1]),
                xytext=(5.12, ys[-1] + y_off),
                fontsize=7.6,
                fontweight="bold",
                color=col,
                va="center",
                ha="left",
            )

    ax1.annotate(
        "Malicious-clique backfire\n(+22 pp worse than unfiltered)",
        xy=(3.0, 55.3),
        xytext=(3.28, 43.0),
        fontsize=7.3,
        fontweight="bold",
        color="#922B21",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDF2F2", edgecolor="#A93226", lw=0.7),
        arrowprops=dict(arrowstyle="->", color="#922B21", lw=1.0),
    )
    ax1.annotate(
        "DeCiv m-copy invariant:\n0.7% (Sup-DS), 1.3% (CAF)",
        xy=(4.85, 1.0),
        xytext=(2.55, 11.5),
        fontsize=7.3,
        fontweight="bold",
        color="#145A32",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F8F5", edgecolor="#1E8449", lw=0.7),
        arrowprops=dict(arrowstyle="->", color="#145A32", lw=1.0),
    )

    ax1.set_xlabel("Adversarial Copy Multiplicity m (10,000-Passage Real Haystack)", fontweight="bold")
    ax1.set_ylabel("Downstream Poisoning ASR (↓ %)", fontweight="bold")
    ax1.set_xticks(ms)
    ax1.set_xticklabels(["m = 1", "m = 3", "m = 5"], fontweight="bold")
    ax1.set_xlim(0.75, 5.85)
    ax1.set_ylim(-2.5, 92.0)
    ax1.grid(True, alpha=0.6)
    ax1.legend(loc="upper left", ncol=1, fontsize=6.7, frameon=True, facecolor="white", edgecolor="#000000", framealpha=0.95)

    # --- Panel (b): Real 5-Model DeCiv-Real-1200 Cost-Matrix Sensitivity ---
    ds = [0.1, 0.25, 0.5, 1.0, 2.0]
    lookup_s = {(r["method_id"], float(r["c_acc"]), float(r["defer_cost"])): r for r in sens}
    c_specs = [
        ("deciv_caf_risk", 10.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=10$)", "#2980B9", "o", "--"),
        ("deciv_caf_risk", 18.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=18$, default)", "#1F618D", "s", "-"),
        ("deciv_caf_risk", 30.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=30$)", "#154360", "^", ":"),
        ("deciv_sup_soft_ds_temp", 18.0, r"Sup-Soft-DS + Temp ($C_{\mathrm{acc}}=18$)", "#1E8449", "D", "-"),
    ]
    for mid, cacc, lbl, col, mk, ls in c_specs:
        fars = [float(lookup_s[(mid, cacc, d)]["far"]) * 100.0 for d in ds]
        ax2.plot(ds, fars, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=2.0, markersize=5.5)

    ax2.set_xlabel("Deferral Escalation Cost d (Real 5-Model DeCiv-Real-1200)", fontweight="bold")
    ax2.set_ylabel("Out-of-Fold FAR (↓ %, Left Axis)", fontweight="bold")
    ax2.set_xlim(0.0, 2.12)
    ax2.set_ylim(-0.08, 1.65)
    ax2.grid(True, alpha=0.6)

    ax2_r = ax2.twinx()
    nd_caf = [float(lookup_s[("deciv_caf_risk", 18.0, d)]["non_defer_fraction"]) * 100.0 for d in ds]
    nd_ds = [float(lookup_s[("deciv_sup_soft_ds_temp", 18.0, d)]["non_defer_fraction"]) * 100.0 for d in ds]
    ax2_r.plot(ds, nd_caf, color="#CA6F1E", marker="x", linestyle="-.", linewidth=1.65, markersize=5.5, label="Non-Defer Throughput (CAF, right)")
    ax2_r.plot(ds, nd_ds, color="#7D3C98", marker="+", linestyle=":", linewidth=1.65, markersize=6.0, label="Non-Defer Throughput (Sup-DS, right)")
    ax2_r.set_ylabel("Non-Defer Resolution Throughput (↑ %, Right)", fontweight="bold", color="#000000")
    ax2_r.set_ylim(10, 82)
    ax2_r.tick_params(axis="y", colors="#000000")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.7, frameon=True, facecolor="white", edgecolor="#000000", framealpha=0.95)

    fig.text(0.27, 0.035, "(a) Multi-Copy PoisonedRAG Malicious-Clique Collapse vs. DeCiv Invariance", ha="center", va="center", fontsize=8.9, fontweight="bold")
    fig.text(0.75, 0.035, r"(b) Real 5-Model Cost-Matrix Sensitivity Across $d$ and $C_{\mathrm{acc}}$", ha="center", va="center", fontsize=8.9, fontweight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"multicopy_haystack_and_real_sensitivity.{ext}", dpi=300)
    plt.close(fig)
    print("Saved multicopy_haystack_and_real_sensitivity.{pdf,png}")


if __name__ == "__main__":
    main()
