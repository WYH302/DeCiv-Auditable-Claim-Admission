"""Generate Publication-Grade Teaser Figure 1 (Restrained Muted Academic Palette, Thin Strokes).

Left  (a): Mechanistic Intuition — Query-Time Malicious-Clique Flip vs. Write-Time Dual-Plane Isolation.
Right (b): Security–Utility–Latency Pareto Frontier with m=1 -> m=5 Trajectories (10k Real Haystack).

All numbers dynamically loaded from experiments/results/real_multicopy_haystack_rag.csv.
Outputs experiments/figures/figure1_teaser_pareto.{pdf,png}.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "figures"

# Unified Restrained Muted Academic Palette (Zero Pure RGB, Zero Neon, Pure Black Text)
C_TEAL = "#3D8C7D"      # DeCiv Primary (Muted Sage Teal)
C_NAVY = "#2B5B84"      # DeCiv Secondary (Deep Academic Navy)
C_STEEL = "#7FA6C9"     # Gated Baseline (Soft Steel Blue)
C_CORAL = "#C05A46"     # Danger / ReliabilityRAG / Inset Frame (Muted Terracotta)
C_OCHRE = "#C48F3B"     # SeCon-RAG (Muted Warm Ochre)
C_SLATE = "#8C96A0"     # Unfiltered / Neutral Slate
C_PLUM = "#7A5C87"      # Merkle / Audit Accent (Muted Plum)
C_BLACK = "#000000"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "text.color": C_BLACK,
        "axes.labelcolor": C_BLACK,
        "xtick.color": C_BLACK,
        "ytick.color": C_BLACK,
        "axes.edgecolor": C_BLACK,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_mc() -> dict[tuple[int, str], dict]:
    with (RES / "real_multicopy_haystack_rag.csv").open(encoding="utf-8") as f:
        return {(int(r["poison_copies_m"]), r["regime_id"]): r for r in csv.DictReader(f)}


def rbox(ax, x, y, w, h, fc, ec, lw=0.75, ls="-", r=0.022, z=2):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p1, p2, col=C_BLACK, lw=0.9, ls="-", ms=8.5, z=5):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=col,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=z,
        )
    )


def node_circle(ax, xy, ry, fc, ec=C_BLACK, lw=0.6, z=5):
    """Aspect-compensated true circle for Panel (a) (width/height ~ 1.80)."""
    ax.add_patch(Ellipse(xy, width=2.0 * ry / 1.80, height=2.0 * ry, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))


def draw_panel_a(ax, mc: dict) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    rel5_asr = float(mc[(5, "reliabilityrag_2025")]["downstream_poisoning_asr"]) * 100.0
    rel5_lat = float(mc[(5, "reliabilityrag_2025")]["query_defense_latency_ms"])
    dc5_asr = float(mc[(5, "deciv_strict_plus_revise")]["downstream_poisoning_asr"]) * 100.0
    dc5_safe = float(mc[(5, "deciv_strict_plus_revise")]["trustworthy_qa_accuracy_safe150"]) * 100.0

    # --- TOP BAND: Query-Time Consensus Failure ---
    rbox(ax, 0.01, 0.535, 0.98, 0.445, "#FCF6F4", C_CORAL, lw=0.85, ls="--", r=0.02, z=1)
    ax.text(
        0.03,
        0.945,
        "Conventional Query-Time Consensus (ReliabilityRAG / SeCon-RAG)",
        fontsize=8.2,
        fontweight="bold",
        color=C_BLACK,
        va="center",
        zorder=4,
    )

    rbox(ax, 0.03, 0.615, 0.21, 0.26, "#FFFFFF", C_SLATE, lw=0.7, r=0.018, z=2)
    ax.text(0.135, 0.835, "Ingestion Stream", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=4)
    node_circle(ax, (0.080, 0.755), 0.024, C_NAVY, lw=0.55, z=4)
    ax.text(0.112, 0.755, "1 Clean", fontsize=7.0, va="center", zorder=4)
    for k in range(5):
        node_circle(ax, (0.072 + k * 0.031, 0.680), 0.017, C_CORAL, lw=0.5, z=4)
    ax.text(0.135, 0.638, "+ 5 Poison Copies", fontsize=6.8, ha="center", va="center", zorder=4)

    rbox(ax, 0.295, 0.570, 0.415, 0.325, "#FFFFFF", C_CORAL, lw=0.7, r=0.018, z=2)
    ax.text(
        0.502,
        0.865,
        r"Top-$k$ Conflict Graph (Poisoned Store)",
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=4,
    )

    clean_xy = (0.370, 0.720)
    clique_pts = [
        (0.548, 0.798),
        (0.635, 0.790),
        (0.665, 0.708),
        (0.610, 0.635),
        (0.530, 0.660),
    ]
    for pt in clique_pts:
        ax.plot([clean_xy[0], pt[0]], [clean_xy[1], pt[1]], color=C_SLATE, lw=0.55, ls=":", zorder=3)
    for i in range(len(clique_pts)):
        for j in range(i + 1, len(clique_pts)):
            ax.plot(
                [clique_pts[i][0], clique_pts[j][0]],
                [clique_pts[i][1], clique_pts[j][1]],
                color=C_CORAL,
                lw=0.65,
                alpha=0.72,
                zorder=3,
            )
    node_circle(ax, clean_xy, 0.033, C_NAVY, lw=0.65, z=5)
    ax.text(clean_xy[0], clean_xy[1], r"$d_c$", fontsize=6.9, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)
    ax.plot([clean_xy[0] - 0.023, clean_xy[0] + 0.023], [clean_xy[1] - 0.040, clean_xy[1] + 0.040], color=C_BLACK, lw=1.25, zorder=7)
    ax.plot([clean_xy[0] - 0.023, clean_xy[0] + 0.023], [clean_xy[1] + 0.040, clean_xy[1] - 0.040], color=C_BLACK, lw=1.25, zorder=7)
    ax.text(0.370, 0.612, "Expelled!", fontsize=6.9, fontweight="bold", ha="center", va="center", zorder=5)

    for idx, pt in enumerate(clique_pts):
        node_circle(ax, pt, 0.026, C_CORAL, lw=0.55, z=5)
        ax.text(pt[0], pt[1], f"$p_{idx+1}$", fontsize=6.0, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)
    ax.text(0.595, 0.592, r"Clique wins ($\sum s_p > s_c$)", fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=5)

    rbox(ax, 0.765, 0.615, 0.205, 0.26, "#F8EAE6", C_CORAL, lw=0.8, r=0.018, z=2)
    ax.text(0.867, 0.835, "Backfire Outcome", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(
        0.867,
        0.720,
        f"ASR: {rel5_asr:.1f}%\n(+22 pp vs. raw)\n{rel5_lat:.1f} ms/query",
        fontsize=7.1,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.35,
        zorder=4,
    )

    arrow(ax, (0.24, 0.745), (0.295, 0.745), col=C_CORAL, lw=0.9)
    arrow(ax, (0.71, 0.745), (0.765, 0.745), col=C_CORAL, lw=0.9)

    # --- BOTTOM BAND: DeCiv Write-Time Dual-Plane Isolation ---
    rbox(ax, 0.01, 0.02, 0.98, 0.475, "#F3F8F6", C_TEAL, lw=0.85, r=0.02, z=1)
    ax.text(
        0.03,
        0.460,
        "DeCiv Write-Time Dual-Plane Firewall (Ours, Evidence-Anchored)",
        fontsize=8.2,
        fontweight="bold",
        color=C_BLACK,
        va="center",
        zorder=4,
    )

    rbox(ax, 0.03, 0.175, 0.21, 0.24, "#FFFFFF", C_SLATE, lw=0.7, r=0.018, z=2)
    ax.text(0.135, 0.375, "Same Stream", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=4)
    node_circle(ax, (0.080, 0.300), 0.024, C_NAVY, lw=0.55, z=4)
    ax.text(0.112, 0.300, "1 Clean", fontsize=7.0, va="center", zorder=4)
    for k in range(5):
        node_circle(ax, (0.072 + k * 0.031, 0.232), 0.017, C_CORAL, lw=0.5, z=4)
    ax.text(0.135, 0.194, "+ 5 Poison Copies", fontsize=6.8, ha="center", va="center", zorder=4)

    rbox(ax, 0.295, 0.185, 0.235, 0.23, "#EDF3F8", C_NAVY, lw=0.8, r=0.018, z=2)
    ax.text(0.412, 0.375, "DeCiv Write Gate", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(
        0.412,
        0.275,
        "Stage I: 3-Family CAF\nStage II: Hard Gate +\nProjected PG-CRC",
        fontsize=6.8,
        ha="center",
        va="center",
        linespacing=1.35,
        zorder=4,
    )

    rbox(ax, 0.60, 0.225, 0.37, 0.19, "#E8F3F0", C_TEAL, lw=0.8, r=0.018, z=2)
    ax.text(0.785, 0.382, r"Clean Retrieval Plane $\mathcal{R}$ (Top-$k$)", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=4)
    node_circle(ax, (0.650, 0.295), 0.029, C_NAVY, lw=0.65, z=5)
    ax.text(0.650, 0.295, r"$d_c$", fontsize=6.7, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)
    ax.text(
        0.822,
        0.298,
        f"ASR: {dc5_asr:.1f}% (m-Invariant)\nSafe-150: {dc5_safe:.1f}% | 0.0 ms/q",
        fontsize=7.1,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.35,
        zorder=4,
    )

    rbox(ax, 0.03, 0.042, 0.94, 0.095, "#F4F0F6", C_PLUM, lw=0.75, r=0.015, z=2)
    ax.text(
        0.50,
        0.090,
        r"Accountability Plane $\mathcal{M}$: All 5 Poison Copies Blocked & Logged (RFC 9162, $O(\log N)$ Proof)",
        fontsize=7.1,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=4,
    )

    arrow(ax, (0.24, 0.300), (0.295, 0.300), col=C_BLACK, lw=0.9)
    arrow(ax, (0.53, 0.320), (0.60, 0.320), col=C_TEAL, lw=0.95)
    ax.text(0.565, 0.355, "accept", fontsize=6.5, fontweight="bold", ha="center", zorder=5)
    arrow(ax, (0.412, 0.185), (0.412, 0.137), col=C_PLUM, lw=0.9, ls="--")
    ax.text(0.432, 0.160, "reject / defer (5 blocked)", fontsize=6.5, fontweight="bold", ha="left", va="center", zorder=5)


def draw_panel_b(ax, mc: dict) -> None:
    ax.axvspan(-3.0, 5.5, ymin=0.73, ymax=1.0, facecolor="#E8F3F0", alpha=0.6, zorder=0)
    ax.text(
        1.25,
        95.2,
        "Pareto Optimal",
        fontsize=6.9,
        fontweight="bold",
        color=C_BLACK,
        va="center",
        ha="center",
        zorder=4,
    )

    trajs = [
        ("reliabilityrag_2025", C_CORAL, "o", "ReliabilityRAG (NeurIPS '25, 272 ms/q)"),
        ("seconrag_2025", C_OCHRE, "^", "SeCon-RAG (NeurIPS '25, 271 ms/q)"),
        ("no_filter", C_SLATE, "X", "No Filter (Unfiltered, 0 ms/q)"),
    ]
    for rid, col, mk, lbl in trajs:
        x1 = float(mc[(1, rid)]["downstream_poisoning_asr"]) * 100.0
        y1 = float(mc[(1, rid)]["trustworthy_qa_accuracy_safe150"]) * 100.0
        x3 = float(mc[(3, rid)]["downstream_poisoning_asr"]) * 100.0
        y3 = float(mc[(3, rid)]["trustworthy_qa_accuracy_safe150"]) * 100.0
        x5 = float(mc[(5, rid)]["downstream_poisoning_asr"]) * 100.0
        y5 = float(mc[(5, rid)]["trustworthy_qa_accuracy_safe150"]) * 100.0

        ax.scatter([x1], [y1], s=30, facecolors="#FFFFFF", edgecolors=col, linewidths=0.95, marker=mk, zorder=4)
        ax.plot([x1, x3, x5], [y1, y3, y5], color=col, lw=1.05, ls="--", alpha=0.85, zorder=3)
        ax.annotate(
            "",
            xy=(x5, y5),
            xytext=(x3, y3),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.05, mutation_scale=9.5),
            zorder=4,
        )
        ax.scatter([x5], [y5], s=60, color=col, edgecolors=C_BLACK, linewidths=0.55, marker=mk, label=lbl, zorder=5)

    ax.text(8.5, 65.5, r"$m=1$", fontsize=7.5, fontweight="bold", color=C_BLACK, zorder=5)
    ax.annotate(
        r"Unfiltered ($m=5$)",
        xy=(40.0, 39.2),
        xytext=(34.0, 50.0),
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.65),
        zorder=6,
    )
    ax.annotate(
        "SeCon-RAG\n(271 ms/q, m=5)",
        xy=(58.6, 27.0),
        xytext=(46.5, 22.0),
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.65),
        zorder=6,
    )
    ax.annotate(
        "ReliabilityRAG\n(272 ms/q, m=5)",
        xy=(62.0, 25.5),
        xytext=(58.5, 42.0),
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_BLACK, lw=0.65),
        zorder=6,
    )

    wt = [
        ("conformal_risk_gated", C_STEEL, "s", 55, "Conformal-Risk + Gate (0 ms/q)"),
        ("deciv_caf_plus_revise", C_NAVY, "D", 62, "DeCiv-CAF + PG-CRC + Revise (0 ms/q)"),
        ("deciv_strict_plus_revise", C_TEAL, "*", 145, "DeCiv Sup-DS + Auto-Revise (0 ms/q, Ours)"),
    ]
    for rid, col, mk, sz, lbl in wt:
        x5 = float(mc[(5, rid)]["downstream_poisoning_asr"]) * 100.0
        y5 = float(mc[(5, rid)]["trustworthy_qa_accuracy_safe150"]) * 100.0
        ax.scatter([x5], [y5], s=sz, color=col, edgecolors=C_BLACK, linewidths=0.55, marker=mk, label=lbl, zorder=6)

    ax.set_xlim(-3.0, 70.0)
    ax.set_ylim(14.0, 98.5)
    ax.set_xlabel(r"Downstream Poisoning ASR ($\downarrow$ %, 10k Haystack, $m=1 \to 5$)", fontsize=8.8, fontweight="bold")
    ax.set_ylabel(r"Trustworthy QA Accuracy Safe-150 ($\uparrow$ %)", fontsize=8.8, fontweight="bold")
    ax.tick_params(labelsize=8.0)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.55, color="#D4DAE0")
    leg = ax.legend(
        loc="lower left",
        fontsize=6.7,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=C_BLACK,
        framealpha=0.96,
        borderpad=0.42,
        handletextpad=0.4,
    )
    leg.get_frame().set_linewidth(0.6)

    axins = inset_axes(ax, width="43%", height="37%", loc="upper right", borderpad=1.0)
    for rid, col, mk, sz, _ in wt:
        x5 = float(mc[(5, rid)]["downstream_poisoning_asr"]) * 100.0
        y5 = float(mc[(5, rid)]["trustworthy_qa_accuracy_safe150"]) * 100.0
        p5 = float(mc[(5, rid)]["poison_top3_retrieval_rate"]) * 100.0
        axins.scatter([x5], [y5], s=sz * 0.85, color=col, edgecolors=C_BLACK, linewidths=0.55, marker=mk, zorder=6)
        if rid == "deciv_strict_plus_revise":
            axins.text(x5 + 0.14, y5 - 0.55, f"DeCiv Sup-DS\n{x5:.1f}% ASR | {p5:.1f}% PoisR", fontsize=6.3, fontweight="bold", va="top")
        elif rid == "deciv_caf_plus_revise":
            axins.text(x5 + 0.12, y5 - 0.25, f"DeCiv-CAF\n{x5:.1f}% ASR | {p5:.1f}% PoisR", fontsize=6.2, va="center")
        else:
            axins.text(x5 + 0.12, y5 + 0.20, f"Conformal+Gate\n{x5:.1f}% ASR | {p5:.1f}% PoisR", fontsize=6.2, va="center")

    axins.set_xlim(0.2, 2.85)
    axins.set_ylim(81.8, 88.4)
    axins.set_xticks([0.5, 1.5, 2.5])
    axins.set_yticks([82, 85, 88])
    axins.tick_params(labelsize=6.6, colors=C_BLACK, width=0.55)
    axins.grid(True, linestyle=":", linewidth=0.4, alpha=0.55, color="#C8D0D8")
    for sp in axins.spines.values():
        sp.set_edgecolor(C_CORAL)
        sp.set_linewidth(0.75)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec=C_CORAL, lw=0.65, ls="--")


def main() -> None:
    mc = load_mc()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.8, 4.7), gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.17}
    )
    draw_panel_a(ax_a, mc)
    draw_panel_b(ax_b, mc)

    fig.text(
        0.265,
        0.015,
        "(a) Mechanistic Intuition: Query-Time Clique Flip vs. Write-Time Isolation",
        ha="center",
        fontsize=9.2,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.text(
        0.745,
        0.015,
        "(b) Security–Utility–Latency Pareto Frontier (10k Real Haystack)",
        ha="center",
        fontsize=9.2,
        fontweight="bold",
        color=C_BLACK,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.96, bottom=0.145)

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"figure1_teaser_pareto.{ext}", dpi=300)
    plt.close(fig)
    print("Saved refined figure1_teaser_pareto.{pdf,png}")


if __name__ == "__main__":
    main()
