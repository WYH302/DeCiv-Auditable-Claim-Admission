"""Generate publication-grade vector Figure 2 architecture diagram for DeCiv (IEEE Transactions).

100% notation-consistent with the compsoc IEEE Transactions manuscript:
  - 3-Family 5-Model Heterogeneous Verifier Pool (NLI Encoders / Open LLMs / Frontier GPT-6.1-Sol)
  - Stage I: Correlation-Aware Calibrated Epistemic Fusion (DeCiv-CAF) + Nested OOF TempScale T*
  - Stage II: Deterministic Hard Governance Gate A_elig(g_i) + Mass-Preserving Admissibility-Projected PG-CRC
  - Closed-Loop Auto-Revise (Grounded Editor -> 5-Model Re-Verification -> Re-Admit or Defer)
  - Dual-Plane Storage: Clean Retrieval Plane (0.0 ms/q) + RFC 9162 Merkle Accountability Plane
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": "#000000",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, x, y, w, h, fc, ec="#1F2937", lw=1.1, r=0.14, ls="-", z=2):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color="#1F2937", lw=1.4, ls="-", rad=0.0, z=5, ms=12):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
        mutation_scale=ms,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(a)
    return a


def main() -> None:
    fig = plt.figure(figsize=(14.0, 6.2), dpi=300)
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # =========================================================================
    # TOP CONTRAST STRIP: Conventional Un-Gated / Query-Time Vulnerability
    # =========================================================================
    box(ax, 0.15, 5.26, 13.70, 0.82, fc="#FDF2F2", ec="#A93226", lw=1.15, r=0.12, ls="--", z=1)
    ax.text(
        0.34, 5.88, "Conventional Pipelines (Un-Gated Write & Post-Hoc Query-Time Filtering)",
        fontsize=8.6, fontweight="bold", color="#922B21", va="center", ha="left", zorder=3,
    )
    box(ax, 0.34, 5.37, 2.75, 0.34, fc="#FFFFFF", ec="#C0392B", lw=0.85, r=0.08, z=2)
    ax.text(1.715, 5.54, r"Incoming Claim Stream $x_i$", fontsize=7.8, fontweight="bold", ha="center", va="center", zorder=3)

    box(ax, 3.62, 5.37, 3.65, 0.34, fc="#FFFFFF", ec="#C0392B", lw=0.85, r=0.08, z=2)
    ax.text(5.445, 5.54, "Single-Scalar Score / Direct Index Write", fontsize=7.6, fontweight="bold", ha="center", va="center", zorder=3)

    box(ax, 7.80, 5.37, 5.86, 0.34, fc="#FADBD8", ec="#922B21", lw=0.95, r=0.08, z=2)
    ax.text(
        10.73, 5.54,
        "Poisoned Store (83.3% Contam)  →  Malicious Cliques Flip Query-Time Defenses (79–1,021 ms/q)",
        fontsize=7.3, fontweight="bold", color="#78281F", ha="center", va="center", zorder=3,
    )
    arrow(ax, 3.13, 5.54, 3.58, 5.54, color="#922B21", lw=1.35, ms=12)
    arrow(ax, 7.31, 5.54, 7.76, 5.54, color="#922B21", lw=1.35, ms=12)

    # =========================================================================
    # 1. INPUT PACKET COLUMN (x = 0.15 .. 2.35, y = 1.25 .. 5.05)
    # =========================================================================
    box(ax, 0.15, 1.25, 2.20, 3.80, fc="#F4F6F8", ec="#2C3E50", lw=1.2, r=0.14, z=1)
    box(ax, 0.15, 4.67, 2.20, 0.38, fc="#2C3E50", ec="#2C3E50", lw=1.0, r=0.10, z=2)
    ax.text(1.25, 4.86, r"Candidate Packet $x_i$", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)

    box(ax, 0.30, 3.15, 1.90, 1.35, fc="#FFFFFF", ec="#1F618D", lw=1.0, r=0.10, z=2)
    ax.text(1.25, 4.29, "Epistemic Payload", fontsize=8.0, fontweight="bold", color="#1B4F72", ha="center", va="center", zorder=3)
    ax.text(1.25, 3.84, "• Claim text $c_i$\n• Primary evidence $e_i$\n• Citation locator $u_i$", fontsize=7.4, ha="center", va="center", linespacing=1.42, zorder=3)
    ax.text(1.25, 3.29, "(Zero metadata access)", fontsize=6.7, style="italic", color="#1F618D", ha="center", va="center", zorder=3)

    box(ax, 0.30, 1.43, 1.90, 1.52, fc="#FEF9E7", ec="#B7791F", lw=1.0, r=0.10, z=2)
    ax.text(1.25, 2.76, r"Governance Vector $\mathbf{g}_i$", fontsize=8.0, fontweight="bold", color="#7D5A00", ha="center", va="center", zorder=3)
    ax.text(
        1.25, 2.13,
        "• Provenance $g^{\\mathrm{prov}}$\n• Auditability $g^{\\mathrm{aud}}$\n• Human review $g^{\\mathrm{hr}}$\n• Repairability $g^{\\mathrm{rep}}$",
        fontsize=7.1, ha="center", va="center", linespacing=1.35, zorder=3,
    )
    ax.text(1.25, 1.56, "(Deterministic gate only)", fontsize=6.7, style="italic", color="#7D5A00", ha="center", va="center", zorder=3)

    # =========================================================================
    # 2. STAGE I: 3-FAMILY VERIFIER POOL & DECIV-CAF (x = 2.68 .. 6.32, y = 2.22 .. 5.05)
    # =========================================================================
    box(ax, 2.68, 2.22, 3.64, 2.83, fc="#EBF5FB", ec="#1F618D", lw=1.25, r=0.14, z=1)
    box(ax, 2.68, 4.67, 3.64, 0.38, fc="#1F618D", ec="#1F618D", lw=1.0, r=0.10, z=2)
    ax.text(4.50, 4.86, "Stage I: 3-Family Pool & DeCiv-CAF Fusion", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)

    fam_specs = [
        (4.11, "Family 1: NLI Encoders", "DeBERTa-v3-L, BART-L", "#D6EAF8", "#1B4F72"),
        (3.50, "Family 2: Open LLMs", "Llama-3.1-8B, Qwen3.5-9B", "#D1F2EB", "#117864"),
        (2.89, "Family 3: Frontier LLM", "GPT-6.1-Sol (API)", "#E8DAEF", "#5B2C6F"),
    ]
    for fy, ft, sub, ffc, fec in fam_specs:
        box(ax, 2.81, fy, 1.72, 0.48, fc=ffc, ec=fec, lw=0.9, r=0.08, z=2)
        ax.text(3.67, fy + 0.32, ft, fontsize=6.9, fontweight="bold", color="#000000", ha="center", va="center", zorder=3)
        ax.text(3.67, fy + 0.13, sub, fontsize=6.4, color="#1F2937", ha="center", va="center", zorder=3)

    ax.text(3.67, 2.52, "Cross-family $\\hat{\\rho}$ drops 2.3×–2.8×\n(0.481 → 0.169–0.206)", fontsize=6.5, fontweight="bold", color="#1B4F72", ha="center", va="center", zorder=3)

    box(ax, 4.78, 3.30, 1.40, 1.24, fc="#FFFFFF", ec="#1F618D", lw=1.0, r=0.09, z=2)
    ax.text(5.48, 4.35, "DeCiv-CAF", fontsize=8.0, fontweight="bold", color="#1B4F72", ha="center", va="center", zorder=3)
    ax.text(5.48, 3.85, "Simplex + Centered\nLog-Odds + Cross-\nFamily $\\mathbf{\\phi}(\\mathcal{Q}_i)$", fontsize=6.5, ha="center", va="center", linespacing=1.25, zorder=3)
    ax.text(5.48, 3.43, r"+ OOF TempScale $T^*$", fontsize=6.6, fontweight="bold", color="#5B2C6F", ha="center", va="center", zorder=3)

    box(ax, 4.78, 2.34, 1.40, 0.78, fc="#FFFFFF", ec="#1F618D", lw=0.9, r=0.08, z=2)
    ax.text(5.48, 3.00, r"Calibrated $\hat{\mathbf{p}}_i \in \Delta^3$", fontsize=6.9, fontweight="bold", color="#000000", ha="center", va="center", zorder=3)
    bar_cols = ["#1E8449", "#C0392B", "#D68910", "#2980B9"]
    bar_hs = [0.35, 0.08, 0.17, 0.12]
    bar_lbls = ["S", "C", "P", "I"]
    for idx_b, (bc, bh, bl) in enumerate(zip(bar_cols, bar_hs, bar_lbls)):
        bx = 4.94 + idx_b * 0.28
        ax.add_patch(Rectangle((bx, 2.49), 0.19, bh, facecolor=bc, edgecolor="#000000", linewidth=0.5, zorder=3))
        ax.text(bx + 0.095, 2.41, bl, fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=3)

    for fy in (4.35, 3.74, 3.13):
        arrow(ax, 4.56, fy, 4.76, 3.92, color="#1F618D", lw=1.05, ms=9)
    arrow(ax, 5.48, 3.28, 5.48, 3.14, color="#1F618D", lw=1.1, ms=9)

    # =========================================================================
    # 3. STAGE II: HARD GATE & PROJECTED PG-CRC (x = 6.66 .. 10.62, y = 1.25 .. 5.05)
    # =========================================================================
    box(ax, 6.66, 1.25, 3.96, 3.80, fc="#FEF9E7", ec="#B7791F", lw=1.25, r=0.14, z=1)
    box(ax, 6.66, 4.67, 3.96, 0.38, fc="#B7791F", ec="#B7791F", lw=1.0, r=0.10, z=2)
    ax.text(8.64, 4.86, "Stage II: Hard Gate & Projected PG-CRC", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)

    box(ax, 6.80, 1.43, 1.96, 1.32, fc="#FFF3CD", ec="#7D5A00", lw=1.1, r=0.10, z=2)
    ax.text(7.78, 2.55, "Hard Governance Gate", fontsize=7.8, fontweight="bold", color="#7D5A00", ha="center", va="center", zorder=3)
    ax.text(7.78, 2.01, "Admissible Set $\\mathcal{A}_{\\mathrm{elig}}(\\mathbf{g}_i)$\nDeterministic EV = 0\n(Zero Semantic Bypass)", fontsize=6.9, ha="center", va="center", linespacing=1.35, zorder=3)

    box(ax, 6.80, 3.04, 1.96, 1.46, fc="#FFFFFF", ec="#B7791F", lw=1.05, r=0.10, z=2)
    ax.text(7.78, 4.30, "Admissibility-Projected", fontsize=7.6, fontweight="bold", color="#000000", ha="center", va="center", zorder=3)
    ax.text(7.78, 4.10, "PG-CRC Router", fontsize=7.6, fontweight="bold", color="#1F618D", ha="center", va="center", zorder=3)
    ax.text(
        7.78, 3.51,
        "accept iff accept $\\in \\mathcal{A}_{\\mathrm{elig}}$\n  $\\wedge\\; \\hat{p}_{\\mathrm{sup}} \\geq \\hat{\\tau}$  (Thm. 1)\nElse project blocked mass\nonto defer & route in $\\mathcal{A}_{\\mathrm{elig}}$",
        fontsize=6.4, ha="center", va="center", linespacing=1.30, zorder=3,
    )
    arrow(ax, 7.78, 2.78, 7.78, 3.01, color="#7D5A00", lw=1.35, ms=11)

    act_specs = [
        (4.06, "accept", "Verified Write", "#D5F5E3", "#1E8449"),
        (3.22, "reject", "Expel Conflict", "#FADBD8", "#A93226"),
        (2.38, "revise", "Grounded Repair", "#FDEBD0", "#CA6F1E"),
        (1.54, "defer", "Audit Quarantine", "#D6EAF8", "#1F618D"),
    ]
    for ay, atitle, asub, afc, aec in act_specs:
        box(ax, 9.08, ay, 1.40, 0.54, fc=afc, ec=aec, lw=1.05, r=0.10, z=2)
        ax.text(9.78, ay + 0.36, atitle, fontsize=7.9, fontweight="bold", color="#000000", ha="center", va="center", zorder=3)
        ax.text(9.78, ay + 0.14, asub, fontsize=6.5, color="#1F2937", ha="center", va="center", zorder=3)
        arrow(ax, 8.79, 3.74, 9.05, ay + 0.27, color=aec, lw=1.2, ms=10)

    # =========================================================================
    # 4. DUAL-PLANE RETRIEVAL & CLOSED-LOOP AUTO-REVISE (x = 10.96 .. 13.85, y = 1.25 .. 5.05)
    # =========================================================================
    box(ax, 10.96, 3.30, 2.89, 1.75, fc="#E8F8F5", ec="#1E8449", lw=1.3, r=0.14, z=1)
    box(ax, 10.96, 4.67, 2.89, 0.38, fc="#1E8449", ec="#1E8449", lw=1.0, r=0.10, z=2)
    ax.text(12.405, 4.86, r"Retrieval Plane (Clean $\mathcal{R}$)", fontsize=8.4, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)
    ax.text(
        12.405, 3.98,
        "Authoritative Dense FAISS Store\n"
        "• 1.3%–4.0% Contam | 0.7%–1.3% ASR\n"
        "• 88.0% Trustworthy Safe-150 QA\n"
        "• 0.0 ms/q Query Defense Overhead\n"
        "• $m$-Copy Poisoning Invariant (Prop. 2)",
        fontsize=6.85, fontweight="bold", color="#145A32", ha="center", va="center", linespacing=1.42, zorder=3,
    )

    box(ax, 10.96, 1.25, 2.89, 1.80, fc="#FEF5E7", ec="#CA6F1E", lw=1.2, r=0.14, z=1)
    box(ax, 10.96, 2.69, 2.89, 0.36, fc="#CA6F1E", ec="#CA6F1E", lw=1.0, r=0.10, z=2)
    ax.text(12.405, 2.87, "Closed-Loop Auto-Revise", fontsize=8.3, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)
    ax.text(
        12.405, 1.96,
        "1. Evidence-Grounded Edit $c'_i = \\mathcal{E}(c_i, e_i)$\n"
        "   (Qwen3.5-9B, 100% Queue Prec.)\n"
        "2. 5-Model Stage I+II Re-Verification\n"
        "   • Grounded (93.1% NLI) → Re-Admit $\\mathcal{R}$\n"
        "   • Faulty Edit (0.275 NLI) → Block defer",
        fontsize=6.65, color="#1F2937", ha="center", va="center", linespacing=1.42, zorder=3,
    )

    # =========================================================================
    # 5. ACCOUNTABILITY PLANE (FULL-WIDTH BOTTOM BAR: x = 0.15 .. 13.85, y = 0.12 .. 0.90)
    # =========================================================================
    box(ax, 0.15, 0.12, 13.70, 0.78, fc="#F4ECF7", ec="#5B2C6F", lw=1.3, r=0.12, z=1)
    box(ax, 0.15, 0.56, 4.30, 0.34, fc="#5B2C6F", ec="#5B2C6F", lw=1.0, r=0.09, z=2)
    ax.text(2.30, 0.73, r"Accountability Plane: RFC 9162 Merkle Log $\mathcal{M}$", fontsize=7.9, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=3)
    ax.text(
        9.18, 0.73,
        "Append-Only SHA-256 (0x00 Leaf / 0x01 Node)  |  $O(1)$ Amortized Append  |  $O(\\log N)$ Inclusion & Consistency Proofs",
        fontsize=7.2, fontweight="bold", color="#4A235A", ha="center", va="center", zorder=3,
    )
    ax.text(
        7.00, 0.33,
        "Commits canonical receipt $m_t$ (packet hash, calibrated $\\hat{\\mathbf{p}}_i$, $\\mathcal{A}_{\\mathrm{elig}}$, action $\\hat{a}_i$, and revise→accept lineage pointer) for EVERY packet  •  52.3 µs/proof (3,125× compression at N=50k)",
        fontsize=6.85, color="#000000", ha="center", va="center", zorder=3,
    )

    # =========================================================================
    # INTER-STAGE ARROWS (Crisp 12pt heads, zero crossings!)
    # =========================================================================
    arrow(ax, 2.23, 3.82, 2.65, 3.82, color="#1F618D", lw=1.6, ms=13)
    arrow(ax, 2.23, 1.88, 6.77, 1.88, color="#B7791F", lw=1.65, ms=13)
    ax.text(4.50, 2.02, r"Isolated Governance Channel $\mathbf{g}_i$ (Bypassing Stage I)", fontsize=7.0, fontweight="bold", color="#7D5A00", ha="center", va="center", zorder=4)

    arrow(ax, 6.35, 3.74, 6.77, 3.74, color="#1F618D", lw=1.6, ms=13)
    arrow(ax, 10.51, 4.33, 10.93, 4.33, color="#1E8449", lw=1.65, ms=13)
    arrow(ax, 10.51, 2.65, 10.93, 2.65, color="#CA6F1E", lw=1.6, ms=13)

    # Re-verified grounded repair -> Retrieval Plane
    arrow(ax, 12.405, 3.08, 12.405, 3.28, color="#1E8449", lw=1.65, ms=12)
    ax.text(13.05, 3.17, "Re-Admit", fontsize=6.9, fontweight="bold", color="#1E8449", ha="center", va="center", zorder=5)

    # Downward audit receipts into RFC 9162 Accountability Plane
    arrow(ax, 9.78, 1.51, 9.78, 0.93, color="#5B2C6F", lw=1.5, ls="--", ms=12)
    ax.text(8.92, 1.09, "All 4 Actions Logged", fontsize=6.8, fontweight="bold", color="#5B2C6F", ha="center", va="center", zorder=4)
    arrow(ax, 12.405, 1.23, 12.405, 0.93, color="#5B2C6F", lw=1.5, ls="--", ms=12)
    ax.text(13.18, 1.08, "Repair Lineage", fontsize=6.8, fontweight="bold", color="#5B2C6F", ha="center", va="center", zorder=4)

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"figure1_workflow.{ext}", dpi=300)
    plt.close(fig)
    print(f"Saved crisp vector figure1_workflow.{{pdf,png}} -> {FIGS}")


if __name__ == "__main__":
    main()
