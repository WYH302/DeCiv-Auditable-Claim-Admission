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
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#2b5c8f", "#d95f02", "#1b8a5a"]

    # Figure 2: Admission tradeoff
    data = rows("main_metrics.csv")
    metrics = {(r["method"], r["router"]): r for r in data}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax, router in zip(axes, ["direct", "risk"]):
        labels = (
            [("majority", "Majority / fixed"), ("reliability_weighted", "Reliability / adaptive"), ("dawid_skene", "Dawid\u2013Skene")]
            if router == "direct"
            else [("fixed", "Fixed / majority / reliability"), ("adaptive", "DeCiv Adaptive"), ("dawid_skene", "DeCiv + Dawid\u2013Skene")]
        )
        offsets = [(-36, -15), (-82, -3), (-64, 6)] if router == "direct" else [(-36, 8), (6, -13), (-34, -15)]
        for j, (mode, label) in enumerate(labels):
            r = metrics[(mode, router)]
            x = 100 * float(r["non_defer_fraction"])
            y = 100 * float(r["far"])
            ax.scatter(x, y, s=56, color=colors[j], marker=["o", "s", "^"][j], label=label, zorder=3)
            ax.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=offsets[j], textcoords="offset points", fontsize=7.2, color="#1f2937")
        ax.set(
            xlabel="Non-defer resolution fraction (%)",
            ylabel="False-acceptance rate (%)",
            title="(a) Direct semantic mapping" if router == "direct" else "(b) Contextual risk routing",
            xlim=(0, 108),
            ylim=(0, 16.8 if router == "direct" else 0.23),
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="lower left" if router == "direct" else "upper left", frameon=False, fontsize=7.2)
    fig.savefig(FIGS / "admission_tradeoff.pdf")
    fig.savefig(FIGS / "admission_tradeoff.png", dpi=240)
    plt.close(fig)

    # Figure 3: Routing confusions
    data = rows("action_confusion.csv")
    actions = ["accept", "reject", "revise", "defer"]
    ylabels = ["accept\n(n=170)", "reject\n(n=790)", "revise\n(n=450)", "defer\n(n=390)"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
    configs = [
        ("majority", "direct", "(a) Majority / direct"),
        ("fixed", "risk", "(b) DeCiv Fixed / risk"),
        ("adaptive", "risk", "(c) DeCiv Adaptive / risk"),
        ("dawid_skene", "risk", "(d) DeCiv + Dawid\u2013Skene / risk"),
    ]
    for ax, (mode, router, title) in zip(axes.flat, configs):
        matrix = np.zeros((4, 4), dtype=int)
        for r in data:
            if r["method"] == mode and r["router"] == router:
                matrix[actions.index(r["reference_action"]), actions.index(r["predicted_action"])] = int(r["count"])
        percentages = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(percentages, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{matrix[i,j]}\n({percentages[i,j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if percentages[i, j] > 55 else "#111827",
                )
        ax.set_xticks(range(4), actions)
        ax.set_yticks(range(4), ylabels)
        ax.set(xlabel="Recommended disposition", ylabel="Policy reference", title=title)
        ax.tick_params(length=0)
    fig.savefig(FIGS / "routing_confusions.pdf")
    fig.savefig(FIGS / "routing_confusions.png", dpi=240)
    plt.close(fig)

    # Figure 4: Cost sensitivity
    data = rows("cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax in axes:
        ax.axvline(0.5, color="#6b7280", linestyle=":", linewidth=1.1, zorder=1)
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
                linewidth=1.7,
                markersize=4.8,
                zorder=3,
            )
            ax.set(
                xlabel="Deferral cost parameter d",
                ylabel="Non-defer fraction (%)" if k == "non_defer_fraction" else "False-acceptance rate (%)",
                title="(a) Resolution throughput vs. deferral cost" if k == "non_defer_fraction" else "(b) False-acceptance rate vs. deferral cost",
                ylim=(0, 80 if k == "non_defer_fraction" else 0.9),
            )
            ax.set_xticks([0.1, 0.5, 1, 2])
            ax.grid(alpha=0.22)
    axes[0].text(0.54, 6, "d = 0.5 (default)", fontsize=7.2, color="#4b5563")
    axes[1].text(0.54, 0.46, "d = 0.5 (default)", fontsize=7.2, color="#4b5563")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.3)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.3)
    fig.savefig(FIGS / "routing_cost_sensitivity.pdf")
    fig.savefig(FIGS / "routing_cost_sensitivity.png", dpi=240)
    plt.close(fig)

    # Figure 5: 5-Fold Stratified CV & Component Ablation
    cv5 = {(r["method"], r["router"]): r for r in rows("five_fold_cv.csv")}
    ab = {r["variant_id"]: r for r in rows("ablation_metrics.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    ax0 = axes[0]
    m_specs = [
        (("majority", "direct"), "Majority Direct", "#94a3b8"),
        (("fixed", "risk"), "DeCiv Fixed", "#3b82f6"),
        (("adaptive", "risk"), "DeCiv Adaptive", "#d95f02"),
        (("dawid_skene", "risk"), "DeCiv + Dawid\u2013Skene", "#1b8a5a"),
    ]
    cat_labels = ["Safety\n(100\u2212FAR)", "Accept\nPrec.", "4-Action\nAcc.", "Revise\nRecall"]
    x_pos = np.arange(len(cat_labels))
    bar_w = 0.19
    for idx_m, (m_key, m_lbl, m_col) in enumerate(m_specs):
        r5 = cv5[m_key]
        means = [
            100 * (1.0 - float(r5["far_mean"])),
            100 * float(r5["accept_precision_mean"]),
            100 * float(r5["accuracy_mean"]),
            100 * float(r5["recall_revise_mean"]),
        ]
        stds = [
            100 * float(r5["far_std"]),
            100 * float(r5["accept_precision_std"]),
            100 * float(r5["accuracy_std"]),
            100 * float(r5["recall_revise_std"]),
        ]
        ax0.bar(
            x_pos + (idx_m - 1.5) * bar_w,
            means,
            width=bar_w,
            yerr=stds,
            capsize=2.2,
            color=m_col,
            edgecolor="white",
            linewidth=0.5,
            label=m_lbl,
            error_kw={"elinewidth": 0.8, "ecolor": "#334155"},
            zorder=3,
        )
    ax0.set_xticks(x_pos, cat_labels)
    ax0.set_ylim(0, 128)
    ax0.set_ylabel("Score across 5 folds (%)")
    ax0.set_title("(a) 5-Fold stratified CV (mean \u00b1 SD)")
    ax0.grid(axis="y", alpha=0.22)
    ax0.legend(loc="upper center", ncol=2, frameon=False, fontsize=6.6)

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
        ax1.text(max(p_val, r_val) + 1.8, y_i, f"FAR {f_val:.2f}%", va="center", ha="left", fontsize=6.7, color="#1f2937")
    ax1.set_yticks(y_pos, [lbl for _, lbl in ab_keys])
    ax1.set_xlim(0, 134)
    ax1.set_ylim(-0.6, 5.45)
    ax1.set_xlabel("Metric score (%) [annotated with FAR]")
    ax1.set_title("(b) Component ablation impact")
    ax1.grid(axis="x", alpha=0.22)
    ax1.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.5)

    fig.savefig(FIGS / "ablation_and_cv.pdf")
    fig.savefig(FIGS / "ablation_and_cv.png", dpi=240)
    plt.close(fig)
    print("Generated publication figures in figures/ directory.")


if __name__ == "__main__":
    main()
