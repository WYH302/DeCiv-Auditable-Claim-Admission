"""Generate publication-grade figures (Figures 2-5) from executed CSV results."""
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


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.labelsize": 8.6,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#2b5c8f", "#d95f02", "#1b8a5a"]

    # Figure 2: Selectivity-throughput frontier (6.30 x 2.75 in, zero overlapping annotations!)
    data = rows("main_metrics.csv")
    metrics = {(r["method"], r["router"]): r for r in data}
    sota = {r["baseline_id"]: r for r in rows("sota_baselines.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.75), layout="constrained")
    ax_a = axes[0]
    pts_a = [
        (100 * float(metrics[("majority", "direct")]["non_defer_fraction"]), 100 * float(metrics[("majority", "direct")]["far"]), "Majority / direct", "#2b5c8f", "o", (-62, 2)),
        (100 * float(metrics[("dawid_skene", "direct")]["non_defer_fraction"]), 100 * float(metrics[("dawid_skene", "direct")]["far"]), "Dawid\u2013Skene / direct", "#1b8a5a", "^", (-44, 6)),
        (100 * float(sota["glad_direct"]["non_defer_fraction"]), 100 * float(sota["glad_direct"]["far"]), "GLAD EM / direct", "#7c3aed", "D", (-30, -15)),
        (100 * float(sota["conformal_risk"]["non_defer_fraction"]), 100 * float(sota["conformal_risk"]["far"]), "Conformal-Risk (2024)", "#0284c7", "X", (-66, -3)),
        (100 * float(sota["selective_threshold"]["non_defer_fraction"]), 100 * float(sota["selective_threshold"]["far"]), "Selective (2023)", "#d95f02", "s", (-62, 2)),
        (100 * float(sota["margin_rejector"]["non_defer_fraction"]), 100 * float(sota["margin_rejector"]["far"]), "Margin (2024)", "#e11d48", "v", (-58, -4)),
        (100 * float(sota["posthoc_l2d"]["non_defer_fraction"]), 100 * float(sota["posthoc_l2d"]["far"]), "Post-Hoc L2D (2022)", "#0f766e", "P", (-28, -14)),
    ]
    for x, y, lbl, col, mk, off in pts_a:
        ax_a.scatter(x, y, s=42, color=col, marker=mk, label=lbl, zorder=3)
        ax_a.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=off, textcoords="offset points", fontsize=6.0, color="#1e293b")
    ax_a.set(xlabel="Non-defer resolution fraction (%)", ylabel="False-acceptance rate (%)", title="(a) Direct & SOTA baselines", xlim=(0, 118), ylim=(0, 23.6))
    ax_a.set_yticks([0, 4, 8, 12, 16])
    ax_a.grid(alpha=0.24, color="#cbd5e1")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, fontsize=5.6, columnspacing=0.6, handletextpad=0.28, labelspacing=0.22)

    ax_b = axes[1]
    pts_b = [
        (100 * float(metrics[("fixed", "risk")]["non_defer_fraction"]), 100 * float(metrics[("fixed", "risk")]["far"]), "DeCiv Fixed", "#2b5c8f", "o", (-28, 9)),
        (100 * float(metrics[("adaptive", "risk")]["non_defer_fraction"]), 100 * float(metrics[("adaptive", "risk")]["far"]), "DeCiv Adaptive", "#d95f02", "s", (-20, -15)),
        (100 * float(sota["deciv_glad_risk"]["non_defer_fraction"]), 100 * float(sota["deciv_glad_risk"]["far"]), "DeCiv + GLAD", "#7c3aed", "D", (7, -3)),
        (100 * float(metrics[("dawid_skene", "risk")]["non_defer_fraction"]), 100 * float(metrics[("dawid_skene", "risk")]["far"]), "DeCiv + Dawid\u2013Skene", "#1b8a5a", "^", (7, -3)),
    ]
    ax_b.plot([p[0] for p in pts_b], [p[1] for p in pts_b], color="#64748b", linestyle="--", linewidth=1.1, zorder=2)
    for x, y, lbl, col, mk, off in pts_b:
        ax_b.scatter(x, y, s=48, color=col, marker=mk, label=lbl, zorder=3)
        ax_b.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=off, textcoords="offset points", fontsize=6.4, color="#1e293b")
    ax_b.set(xlabel="Non-defer resolution fraction (%)", ylabel="False-acceptance rate (%)", title="(b) DeCiv contextual risk routing", xlim=(0, 112), ylim=(0, 0.285))
    ax_b.grid(alpha=0.24, color="#cbd5e1")
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.4, handletextpad=0.35, labelspacing=0.25)
    fig.savefig(FIGS / "admission_tradeoff.pdf")
    fig.savefig(FIGS / "admission_tradeoff.png", dpi=300)
    plt.close(fig)

    # Figure 3: Compact 2x2 four-action confusion matrices (6.30 x 3.32 in)
    data = rows("action_confusion.csv")
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
                ax.text(j, i, f"{matrix[i,j]}\n({percentages[i,j]:.1f}%)", ha="center", va="center", fontsize=6.9, color="white" if percentages[i, j] > 55 else "#0f172a")
        ax.set_xticks(range(4), actions if idx_ax >= 2 else [""] * 4)
        ax.set_yticks(range(4), ylabels if idx_ax % 2 == 0 else [""] * 4)
        ax.set_title(title, pad=4)
        if idx_ax >= 2:
            ax.set_xlabel("Recommended disposition")
        if idx_ax % 2 == 0:
            ax.set_ylabel("Policy reference")
        ax.tick_params(length=0)
    fig.savefig(FIGS / "routing_confusions.pdf")
    fig.savefig(FIGS / "routing_confusions.png", dpi=300)
    plt.close(fig)

    # Figure 4: Deferral cost sensitivity (6.30 x 2.65 in, sleek line weights)
    data = rows("cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.65), layout="constrained")
    for ax in axes:
        ax.axvline(0.5, color="#64748b", linestyle=":", linewidth=0.9, zorder=1)
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
                linewidth=1.35,
                markersize=4.2,
                zorder=3,
            )
            ax.set(
                xlabel="Deferral cost parameter d",
                ylabel="Non-defer fraction (%)" if k == "non_defer_fraction" else "False-acceptance rate (%)",
                title="(a) Resolution throughput vs. cost d" if k == "non_defer_fraction" else "(b) False-acceptance rate vs. cost d",
                ylim=(0, 87 if k == "non_defer_fraction" else 0.9),
            )
            ax.set_xticks([0.1, 0.5, 1, 2])
            ax.grid(alpha=0.22, color="#cbd5e1")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.0)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.0)
    fig.savefig(FIGS / "routing_cost_sensitivity.pdf")
    fig.savefig(FIGS / "routing_cost_sensitivity.png", dpi=300)
    plt.close(fig)

    # Figure 5: Multi-Metric SOTA Comparison & Component Ablation (6.30 x 2.75 in)
    ab = {r["variant_id"]: r for r in rows("ablation_metrics.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.75), layout="constrained")
    ax0 = axes[0]
    comp_specs = [
        ("Selective (2023)", "#94a3b8", [100 * (1 - float(sota["selective_threshold"]["far"])), 100 * float(sota["selective_threshold"]["accept_precision"]), 100 * float(sota["selective_threshold"]["accuracy"]), 100 * float(sota["selective_threshold"]["recall_revise"])]),
        ("Post-Hoc L2D", "#64748b", [100 * (1 - float(sota["posthoc_l2d"]["far"])), 100 * float(sota["posthoc_l2d"]["accept_precision"]), 100 * float(sota["posthoc_l2d"]["accuracy"]), 100 * float(sota["posthoc_l2d"]["recall_revise"])]),
        ("DeCiv Adaptive", "#d95f02", [100 * (1 - float(metrics[("adaptive", "risk")]["far"])), 100 * float(metrics[("adaptive", "risk")]["accept_precision"]), 100 * float(metrics[("adaptive", "risk")]["accuracy"]), 100 * float(metrics[("adaptive", "risk")]["recall_revise"])]),
        ("DeCiv + GLAD", "#7c3aed", [100 * (1 - float(sota["deciv_glad_risk"]["far"])), 100 * float(sota["deciv_glad_risk"]["accept_precision"]), 100 * float(sota["deciv_glad_risk"]["accuracy"]), 100 * float(sota["deciv_glad_risk"]["recall_revise"])]),
        ("DeCiv + DS", "#1b8a5a", [100 * (1 - float(metrics[("dawid_skene", "risk")]["far"])), 100 * float(metrics[("dawid_skene", "risk")]["accept_precision"]), 100 * float(metrics[("dawid_skene", "risk")]["accuracy"]), 100 * float(metrics[("dawid_skene", "risk")]["recall_revise"])]),
    ]
    cat_labels = ["Safety\n(100\u2212FAR)", "Accept\nPrec.", "4-Action\nAcc.", "Revise\nRecall"]
    x_pos = np.arange(len(cat_labels))
    bar_w = 0.155
    for idx_m, (m_lbl, m_col, vals) in enumerate(comp_specs):
        ax0.bar(x_pos + (idx_m - 2.0) * bar_w, vals, width=bar_w, color=m_col, edgecolor="white", linewidth=0.45, label=m_lbl, zorder=3)
    ax0.set_xticks(x_pos, cat_labels)
    ax0.set_ylim(0, 144)
    ax0.set_yticks([0, 20, 40, 60, 80, 100])
    ax0.set_ylabel("Metric score (%)")
    ax0.set_title("(a) SOTA baselines vs. DeCiv")
    ax0.grid(axis="y", alpha=0.22, color="#cbd5e1")
    ax0.legend(loc="upper left", bbox_to_anchor=(0.01, 1.01), ncol=2, frameon=False, fontsize=5.9, columnspacing=0.6, handletextpad=0.3, labelspacing=0.22)

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
        ax1.text(max(p_val, r_val) + 1.8, y_i, f"FAR {f_val:.2f}%", va="center", ha="left", fontsize=6.4, color="#1e293b")
    ax1.set_yticks(y_pos, [lbl for _, lbl in ab_keys])
    ax1.set_xlim(0, 140)
    ax1.set_ylim(-0.6, 5.45)
    ax1.set_xlabel("Metric score (%)")
    ax1.set_title("(b) Ablation impact (Adaptive)")
    ax1.grid(axis="x", alpha=0.22, color="#cbd5e1")
    ax1.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.2, columnspacing=0.65, handletextpad=0.35)

    fig.savefig(FIGS / "ablation_and_cv.pdf")
    fig.savefig(FIGS / "ablation_and_cv.png", dpi=300)
    plt.close(fig)
    print("Generated publication figures in figures/ directory.")


if __name__ == "__main__":
    main()
