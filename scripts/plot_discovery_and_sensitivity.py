"""Generate unified muted-academic Figure 5:
  (a) Multi-Copy PoisonedRAG Malicious-Clique Collapse vs. DeCiv m-Copy Invariance
      inside a 10,000-Passage Real Wikipedia + PubMed Haystack (with Zoom-In Inset)
  (b) Real 5-Model DeCiv-Real-1200 Cost-Matrix Sensitivity (d in [0.1, 2.0], C_acc in {10, 18, 30})
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Unified Restrained Muted Academic Palette (Identical across all 6 data figures)
C_TEAL = "#3D8C7D"
C_NAVY = "#2B5B84"
C_DEEPNAVY = "#1A3A54"
C_STEEL = "#7FA6C9"
C_CORAL = "#C05A46"
C_OCHRE = "#C48F3B"
C_SLATE = "#8C96A0"
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


def main() -> None:
    mc = read_csv("real_multicopy_haystack_rag.csv")
    sens = read_csv("real_cost_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.20), dpi=300)
    fig.subplots_adjust(left=0.058, right=0.940, top=0.91, bottom=0.22, wspace=0.28)

    ms = [1, 3, 5]
    series_a = [
        ("no_filter", "No Filter (Unfiltered, 0 ms)", C_SLATE, "x", ":"),
        ("reliabilityrag_2025", "ReliabilityRAG (NeurIPS '25, 272 ms/q)", C_CORAL, "o", "-"),
        ("seconrag_2025", "SeCon-RAG (NeurIPS '25, 271 ms/q)", C_OCHRE, "s", "--"),
        ("conformal_risk_gated", "Conformal-Risk + Gate (0 ms)", C_STEEL, "^", "-."),
        ("deciv_caf_plus_revise", "DeCiv-CAF + PG-CRC + Revise (0 ms)", C_NAVY, "D", "-"),
        ("deciv_strict_plus_revise", "DeCiv Sup-DS + Auto-Revise (0 ms)", C_TEAL, "P", "-"),
    ]
    lookup_mc = {(int(r["poison_copies_m"]), r["regime_id"]): r for r in mc}
    for rid, lbl, col, mk, ls in series_a:
        ys = [float(lookup_mc[(m, rid)]["downstream_poisoning_asr"]) * 100.0 for m in ms]
        ax1.plot(ms, ys, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=1.35, markersize=4.6, markeredgewidth=0.5)
        if rid in ("reliabilityrag_2025", "seconrag_2025", "no_filter"):
            y_off = 2.0 if rid == "reliabilityrag_2025" else (-2.0 if rid == "seconrag_2025" else 0.0)
            ax1.text(
                5.12,
                ys[-1] + y_off,
                f"{ys[-1]:.1f}%",
                fontsize=7.4,
                fontweight="bold",
                color=C_BLACK,
                va="center",
                ha="left",
            )

    ax1.set_xlabel(r"Adversarial Copy Multiplicity $m$ (10,000-Passage Real Haystack)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_ylabel(r"Downstream Poisoning ASR ($\downarrow$ %)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax1.set_xticks(ms)
    ax1.set_xticklabels([r"$m = 1$", r"$m = 3$", r"$m = 5$"], fontsize=8.0, fontweight="bold", color=C_BLACK)
    ax1.set_xlim(0.75, 5.85)
    ax1.set_ylim(-2.5, 93.0)
    ax1.tick_params(axis="y", labelsize=7.8)
    ax1.grid(True, alpha=0.65)
    leg1 = ax1.legend(loc="upper left", ncol=1, fontsize=6.6, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg1)

    axins = inset_axes(ax1, width="42%", height="22%", loc="lower right", bbox_to_anchor=(0.0, 0.075, 0.99, 1.0), bbox_transform=ax1.transAxes)
    for rid, _, col, mk, ls in series_a[3:]:
        ys = [float(lookup_mc[(m, rid)]["downstream_poisoning_asr"]) * 100.0 for m in ms]
        axins.plot(ms, ys, color=col, marker=mk, linestyle=ls, linewidth=1.05, markersize=3.6, markeredgewidth=0.45)
    axins.text(5.12, 0.70, "0.7%", fontsize=6.2, fontweight="bold", color=C_BLACK, va="center")
    axins.text(5.12, 1.33, "1.3%", fontsize=6.2, fontweight="bold", color=C_BLACK, va="center")
    axins.text(1.14, 2.04, "2.0%", fontsize=6.1, color=C_BLACK, va="center")
    axins.set_xlim(0.80, 5.85)
    axins.set_ylim(0.25, 2.45)
    axins.set_xticks([1, 3, 5])
    axins.set_yticks([0.5, 1.5, 2.5])
    axins.tick_params(labelsize=6.1, colors=C_BLACK, width=0.55)
    axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    axins.set_title("Zoom: Write-Time Floor (0–2.4%)", fontsize=6.2, fontweight="bold", color=C_BLACK, pad=2)
    for sp in axins.spines.values():
        sp.set_edgecolor(C_CORAL)
        sp.set_linewidth(0.75)
    mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec=C_CORAL, lw=0.65, ls="--")

    ds = [0.1, 0.25, 0.5, 1.0, 2.0]
    lookup_s = {(r["method_id"], float(r["c_acc"]), float(r["defer_cost"])): r for r in sens}
    c_specs = [
        ("deciv_caf_risk", 10.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=10$)", C_STEEL, "o", "--"),
        ("deciv_caf_risk", 18.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=18$, default)", C_NAVY, "s", "-"),
        ("deciv_caf_risk", 30.0, r"DeCiv-CAF ($C_{\mathrm{acc}}=30$)", C_DEEPNAVY, "^", ":"),
        ("deciv_sup_soft_ds_temp", 18.0, r"Sup-Soft-DS + Temp ($C_{\mathrm{acc}}=18$)", C_TEAL, "D", "-"),
    ]
    for mid, cacc, lbl, col, mk, ls in c_specs:
        fars = [float(lookup_s[(mid, cacc, d)]["far"]) * 100.0 for d in ds]
        ax2.plot(ds, fars, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=1.30, markersize=4.4, markeredgewidth=0.5)

    ax2.set_xlabel(r"Deferral Escalation Cost $d$ (Real 5-Model DeCiv-Real-1200)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_ylabel(r"Out-of-Fold FAR ($\downarrow$ %, Left Axis)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2.set_xlim(0.0, 2.12)
    ax2.set_ylim(-0.08, 1.65)
    ax2.tick_params(labelsize=7.8)
    ax2.grid(True, alpha=0.65)

    ax2_r = ax2.twinx()
    nd_caf = [float(lookup_s[("deciv_caf_risk", 18.0, d)]["non_defer_fraction"]) * 100.0 for d in ds]
    nd_ds = [float(lookup_s[("deciv_sup_soft_ds_temp", 18.0, d)]["non_defer_fraction"]) * 100.0 for d in ds]
    ax2_r.plot(ds, nd_caf, color=C_CORAL, marker="x", linestyle="-.", linewidth=1.15, markersize=4.6, label="Non-Defer Throughput (CAF, right)")
    ax2_r.plot(ds, nd_ds, color=C_OCHRE, marker="+", linestyle=":", linewidth=1.15, markersize=5.0, label="Non-Defer Throughput (Sup-DS, right)")
    ax2_r.set_ylabel(r"Non-Defer Resolution Throughput ($\uparrow$ %, Right)", fontsize=8.5, fontweight="bold", color=C_BLACK)
    ax2_r.set_ylim(10, 82)
    ax2_r.tick_params(axis="y", labelsize=7.8, colors=C_BLACK)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    leg2 = ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.6, frameon=True, facecolor="#FFFFFF", framealpha=0.96)
    style_legend(leg2)

    fig.text(
        0.270,
        0.032,
        "(a) Multi-Copy PoisonedRAG Collapse vs. DeCiv Invariance",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.748,
        0.032,
        r"(b) Real 5-Model Cost-Matrix Sensitivity Across $d$ and $C_{\mathrm{acc}}$",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=C_BLACK,
    )

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"multicopy_haystack_and_real_sensitivity.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined multicopy_haystack_and_real_sensitivity.{pdf,png}")


if __name__ == "__main__":
    main()
