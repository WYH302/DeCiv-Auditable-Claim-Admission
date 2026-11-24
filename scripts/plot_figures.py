"""Generate publication-grade figures (Figures 2-5) from executed CSV results.

Enforces strict top-journal visual rules:
- Pure black (#000000) text globally for all axes, ticks, titles, legends, and annotations
- Bold centered panel sub-captions: (a) ..., (b) ..., (c) ..., (d) ...
- Zero overlapping labels or legends
- Visual Ground Truth vs. SOTA contamination & institutional-cost comparison in Figure 5(a)
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"


def rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def render_all(results_dir: Path, figs_dir: Path):
    def _rows(name):
        with (results_dir / name).open(encoding="utf-8-sig", newline="") as h:
            return list(csv.DictReader(h))

    figs_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.labelsize": 8.6,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#2b5c8f", "#d95f02", "#1b8a5a"]

    # Figure 2: Selectivity-throughput frontier (6.30 x 2.75 in, pure black text, zero overlap)
    data = _rows("main_metrics.csv")
    metrics = {(r["method"], r["router"]): r for r in data}
    sota = {r["baseline_id"]: r for r in _rows("sota_baselines.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.75), layout="constrained")
    ax_a = axes[0]
    pts_a = [
        (100 * float(metrics[("majority", "direct")]["non_defer_fraction"]), 100 * float(metrics[("majority", "direct")]["far"]), "Majority / direct", "#2b5c8f", "o", (-64, 2)),
        (100 * float(metrics[("dawid_skene", "direct")]["non_defer_fraction"]), 100 * float(metrics[("dawid_skene", "direct")]["far"]), "Dawid\u2013Skene / direct", "#1b8a5a", "^", (-46, 6)),
        (100 * float(sota["glad_direct"]["non_defer_fraction"]), 100 * float(sota["glad_direct"]["far"]), "GLAD EM / direct", "#7c3aed", "D", (-32, -15)),
        (100 * float(sota["conformal_risk"]["non_defer_fraction"]), 100 * float(sota["conformal_risk"]["far"]), "Conformal-Risk (2024)", "#0284c7", "X", (-68, -3)),
        (100 * float(sota["selective_threshold"]["non_defer_fraction"]), 100 * float(sota["selective_threshold"]["far"]), "Selective (2023)", "#d95f02", "s", (-64, 2)),
        (100 * float(sota["margin_rejector"]["non_defer_fraction"]), 100 * float(sota["margin_rejector"]["far"]), "Margin (2024)", "#e11d48", "v", (-60, -4)),
        (100 * float(sota["posthoc_l2d"]["non_defer_fraction"]), 100 * float(sota["posthoc_l2d"]["far"]), "Post-Hoc L2D (2022)", "#0f766e", "P", (-30, -14)),
    ]
    for x, y, lbl, col, mk, off in pts_a:
        ax_a.scatter(x, y, s=42, color=col, marker=mk, label=lbl, zorder=3)
        ax_a.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=off, textcoords="offset points", fontsize=6.0, color="#000000")
    ax_a.set(xlabel="Non-defer resolution fraction (%)", ylabel="False-acceptance rate (%)", xlim=(0, 118), ylim=(0, 23.6))
    ax_a.set_title("(a) Direct & SOTA baselines", fontweight="bold", loc="center", color="#000000")
    ax_a.set_yticks([0, 4, 8, 12, 16])
    ax_a.grid(alpha=0.25, color="#cbd5e1")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, fontsize=5.6, columnspacing=0.6, handletextpad=0.28, labelspacing=0.22, labelcolor="#000000")

    ax_b = axes[1]
    pts_b = [
        (100 * float(metrics[("fixed", "risk")]["non_defer_fraction"]), 100 * float(metrics[("fixed", "risk")]["far"]), "DeCiv Fixed", "#2b5c8f", "o", (-26, 9)),
        (100 * float(metrics[("adaptive", "risk")]["non_defer_fraction"]), 100 * float(metrics[("adaptive", "risk")]["far"]), "DeCiv Adaptive", "#d95f02", "s", (-18, -15)),
        (100 * float(sota["deciv_glad_risk"]["non_defer_fraction"]), 100 * float(sota["deciv_glad_risk"]["far"]), "DeCiv + GLAD", "#7c3aed", "D", (7, -3)),
        (100 * float(metrics[("dawid_skene", "risk")]["non_defer_fraction"]), 100 * float(metrics[("dawid_skene", "risk")]["far"]), "DeCiv + Dawid\u2013Skene", "#1b8a5a", "^", (7, -3)),
    ]
    ax_b.plot([p[0] for p in pts_b], [p[1] for p in pts_b], color="#475569", linestyle="--", linewidth=1.15, zorder=2)
    for x, y, lbl, col, mk, off in pts_b:
        ax_b.scatter(x, y, s=48, color=col, marker=mk, label=lbl, zorder=3)
        ax_b.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=off, textcoords="offset points", fontsize=6.4, color="#000000")
    ax_b.set(xlabel="Non-defer resolution fraction (%)", ylabel="False-acceptance rate (%)", xlim=(0, 112), ylim=(0, 0.285))
    ax_b.set_title("(b) DeCiv contextual risk routing", fontweight="bold", loc="center", color="#000000")
    ax_b.grid(alpha=0.25, color="#cbd5e1")
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.4, handletextpad=0.35, labelspacing=0.25, labelcolor="#000000")
    fig.savefig(figs_dir / "admission_tradeoff.pdf")
    fig.savefig(figs_dir / "admission_tradeoff.png", dpi=300)
    plt.close(fig)

    # Figure 3: Compact 2x2 four-action confusion matrices (6.30 x 3.32 in)
    data = _rows("action_confusion.csv")
    actions = ["accept", "reject", "revise", "defer"]
    ylabels = ["accept (170)", "reject (790)", "revise (450)", "defer (390)"]
    fig, axes = plt.subplots(2, 2, figsize=(6.30, 3.32), layout="constrained")
    cfgs = [
        ("majority", "direct", "(a) Majority / direct"),
        ("fixed", "risk", "(b) DeCiv Fixed / risk"),
        ("adaptive", "risk", "(c) DeCiv Adaptive / risk"),
        ("dawid_skene", "risk", "(d) DeCiv + Dawid\u2013Skene / risk"),
    ]
    for idx_ax, (ax, (mode, router, title)) in enumerate(zip(axes.flat, cfgs)):
        matrix = np.zeros((4, 4), dtype=int)
        for r in data:
            if r["method"] == mode and r["router"] == router:
                matrix[actions.index(r["reference_action"]), actions.index(r["predicted_action"])] = int(r["count"])
        percentages = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(percentages, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]}\n({percentages[i,j]:.1f}%)", ha="center", va="center", fontsize=6.9, color="white" if percentages[i, j] > 55 else "#000000")
        ax.set_xticks(range(4), actions if idx_ax >= 2 else [""] * 4)
        ax.set_yticks(range(4), ylabels if idx_ax % 2 == 0 else [""] * 4)
        ax.set_title(title, pad=4, fontweight="bold", loc="center", color="#000000")
        if idx_ax >= 2:
            ax.set_xlabel("Recommended disposition", color="#000000")
        if idx_ax % 2 == 0:
            ax.set_ylabel("Policy reference", color="#000000")
        ax.tick_params(length=0, colors="#000000")
    fig.savefig(figs_dir / "routing_confusions.pdf")
    fig.savefig(figs_dir / "routing_confusions.png", dpi=300)
    plt.close(fig)

    # Figure 4: Deferral cost sensitivity (6.30 x 2.65 in, pure black text)
    data = _rows("cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.65), layout="constrained")
    for ax in axes:
        ax.axvline(0.5, color="#475569", linestyle=":", linewidth=0.95, zorder=1)
    for j, mode in enumerate(["fixed", "adaptive", "dawid_skene"]):
        rs = sorted([r for r in data if r["method"] == mode], key=lambda r: float(r["defer_cost"]))
        for ax, k in zip(axes, ["non_defer_fraction", "far"]):
            ax.plot(
                [float(r["defer_cost"]) for r in rs],
                [100 * float(r[k]) for r in rs],
                label=["DeCiv Fixed", "DeCiv Adaptive", "DeCiv + Dawid\u2013Skene"][j],
                color=colors[j],
                marker=["o", "s", "^"][j],
                linestyle=["-", "--", "-."][j],
                linewidth=1.4,
                markersize=4.2,
                zorder=3,
            )
            ax.set(
                xlabel="Deferral cost parameter d",
                ylabel="Non-defer fraction (%)" if k == "non_defer_fraction" else "False-acceptance rate (%)",
                ylim=(0, 87 if k == "non_defer_fraction" else 0.9),
            )
            ax.set_title("(a) Resolution throughput vs. cost d" if k == "non_defer_fraction" else "(b) False-acceptance rate vs. cost d", fontweight="bold", loc="center", color="#000000")
            ax.set_xticks([0.1, 0.5, 1, 2])
            ax.grid(alpha=0.22, color="#cbd5e1")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.0, labelcolor="#000000")
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.0, labelcolor="#000000")
    fig.savefig(figs_dir / "routing_cost_sensitivity.pdf")
    fig.savefig(figs_dir / "routing_cost_sensitivity.png", dpi=300)
    plt.close(fig)

    # Figure 5: (a) Ground Truth vs. SOTA & DeCiv Accepted Claims Composition (Valid vs. Contamination + Mean Cost)
    #           (b) Component Ablation Impact on Accept Precision, Revise Recall & FAR (DeCiv Adaptive)
    ab = {r["variant_id"]: r for r in _rows("ablation_metrics.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.05), layout="constrained", gridspec_kw={"width_ratios": [1.14, 1.0]})

    ax0 = axes[0]
    def _counts(rec):
        tot_acc = int(round(float(rec["predicted_accept"])))
        tp = int(round(float(rec["recall_accept"]) * 170))
        fp = max(0, tot_acc - tp)
        cost = float(rec["mean_institutional_cost"])
        return tp, fp, cost

    gt_specs = [
        ("Ground Truth (Ref.)", 170, 0, 0.00),
        ("Dawid\u2013Skene / direct", *_counts(metrics[("dawid_skene", "direct")])),
        ("Conformal-Risk (2024)", *_counts(sota["conformal_risk"])),
        ("Majority / direct", *_counts(metrics[("majority", "direct")])),
        ("Selective (2023)", *_counts(sota["selective_threshold"])),
        ("Margin (2024)", *_counts(sota["margin_rejector"])),
        ("Post-Hoc L2D (2022)", *_counts(sota["posthoc_l2d"])),
        ("DeCiv Adaptive", *_counts(metrics[("adaptive", "risk")])),
        ("DeCiv + GLAD", *_counts(sota["deciv_glad_risk"])),
        ("DeCiv + DS", *_counts(metrics[("dawid_skene", "risk")])),
    ]
    y0 = np.arange(len(gt_specs))[::-1]
    tps = [s[1] for s in gt_specs]
    fps = [s[2] for s in gt_specs]
    costs = [s[3] for s in gt_specs]
    lbls0 = [s[0] for s in gt_specs]
    ax0.barh(y0, tps, height=0.62, color="#1b8a5a", edgecolor="white", linewidth=0.4, label="Valid (TP)", zorder=3)
    ax0.barh(y0, fps, left=tps, height=0.62, color="#dc2626", edgecolor="white", linewidth=0.4, label="Contamination (FA)", zorder=3)
    ax0.axhline(2.5, color="#475569", linestyle="--", linewidth=0.75, zorder=2)
    ax0.axhline(8.5, color="#475569", linestyle=":", linewidth=0.75, zorder=2)
    for yi, tp, fp, cst in zip(y0, tps, fps, costs):
        tag = f"Cost {cst:.2f}" if yi != y0[0] else "Ideal (0 FA)"
        ax0.text(tp + fp + 7, yi, f"{fp} FA | {tag}" if yi != y0[0] else tag, va="center", ha="left", fontsize=5.5, color="#000000", fontweight="bold" if yi <= 2 else "normal")
    ax0.set_yticks(y0, lbls0, fontsize=6.4, color="#000000")
    for tick_lbl, yi in zip(ax0.get_yticklabels(), y0):
        if yi <= 2 or yi == y0[0]:
            tick_lbl.set_fontweight("bold")
    ax0.set_xlim(0, 635)
    ax0.set_ylim(-0.6, len(gt_specs) + 0.7)
    ax0.set_xticks([0, 150, 300, 450])
    ax0.set_xlabel("Admitted claims (count)", fontsize=8.0, color="#000000")
    ax0.set_title("(a) Ground truth vs. SOTA contamination", fontsize=8.3, fontweight="bold", loc="center", pad=5, color="#000000")
    ax0.grid(axis="x", alpha=0.22, color="#cbd5e1")
    ax0.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, fontsize=5.6, columnspacing=0.6, handletextpad=0.3, labelcolor="#000000")

    ax1 = axes[1]
    ab_keys = [
        ("adaptive_full", "Full DeCiv Adaptive"),
        ("adaptive_no_agreement", "w/o Agreement Amp."),
        ("adaptive_no_prov_gate", "w/o Provenance Gate"),
        ("adaptive_symmetric_cost", "w/o Asymmetric Cost"),
        ("adaptive_no_defer", "w/o Defer Quarantine"),
    ]
    y_pos = np.arange(len(ab_keys))[::-1]
    prec_vals = [100 * float(ab[k]["accept_precision"]) for k, _ in ab_keys]
    rev_vals = [100 * float(ab[k]["recall_revise"]) for k, _ in ab_keys]
    far_vals = [100 * float(ab[k]["far"]) for k, _ in ab_keys]
    bh = 0.34
    ax1.barh(y_pos + bh / 2, prec_vals, height=bh, color="#1b8a5a", label="Accept Prec. (%)", zorder=3)
    ax1.barh(y_pos - bh / 2, rev_vals, height=bh, color="#d95f02", label="Revise Recall (%)", zorder=3)
    for y_i, p_val, r_val, f_val in zip(y_pos, prec_vals, rev_vals, far_vals):
        ax1.text(max(p_val, r_val) + 2.2, y_i, f"FAR {f_val:.2f}%", va="center", ha="left", fontsize=5.9, color="#000000", fontweight="bold" if y_i == y_pos[0] else "normal")
    ax1.set_yticks(y_pos, [lbl for _, lbl in ab_keys], fontsize=6.6, color="#000000")
    ax1.get_yticklabels()[0].set_fontweight("bold")
    ax1.set_xlim(0, 155)
    ax1.set_ylim(-0.6, 5.45)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("Metric score (%)", fontsize=8.0, color="#000000")
    ax1.set_title("(b) Component ablation", fontsize=8.3, fontweight="bold", loc="center", pad=5, color="#000000")
    ax1.grid(axis="x", alpha=0.22, color="#cbd5e1")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, fontsize=5.5, columnspacing=0.45, handletextpad=0.25, labelcolor="#000000")

    fig.savefig(figs_dir / "ablation_and_cv.pdf")
    fig.savefig(figs_dir / "ablation_and_cv.png", dpi=300)
    plt.close(fig)


def main():
    render_all(RESULTS, FIGS)
    print("Generated publication figures in figures/ directory.")


if __name__ == "__main__":
    main()
