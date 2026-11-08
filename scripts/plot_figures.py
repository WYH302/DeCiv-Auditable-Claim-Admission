"""Generate publication-grade figures from executed CSV results."""
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
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#386cb0", "#d95f02", "#238b45"]

    # Figure 2: Admission tradeoff
    data = rows("main_metrics.csv")
    metrics = {(r["method"], r["router"]): r for r in data}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for ax, router in zip(axes, ["direct", "risk"]):
        labels = (
            [("majority", "Majority / fixed"), ("reliability_weighted", "Reliability / adaptive"), ("dawid_skene", "Dawid–Skene")]
            if router == "direct"
            else [("fixed", "Fixed / majority / reliability"), ("adaptive", "Adaptive"), ("dawid_skene", "Dawid–Skene")]
        )
        offsets = [(-36, -15), (-82, -3), (-64, 6)] if router == "direct" else [(-36, 8), (6, -13), (-34, -15)]
        for j, (mode, label) in enumerate(labels):
            r = metrics[(mode, router)]
            x = 100 * float(r["non_defer_fraction"])
            y = 100 * float(r["far"])
            ax.scatter(x, y, s=52, color=colors[j], marker=["o", "s", "^"][j], label=label, zorder=3)
            ax.annotate(f"({x:.1f}%, {y:.2f}%)", xy=(x, y), xytext=offsets[j], textcoords="offset points", fontsize=7.2, color="#222222")
        ax.set(
            xlabel="Non-defer fraction (%)",
            ylabel="False-acceptance rate (%)",
            title="(a) Direct semantic mapping" if router == "direct" else "(b) Contextual risk routing",
            xlim=(0, 108),
            ylim=(0, 16.8 if router == "direct" else 0.23),
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="lower left" if router == "direct" else "upper left", frameon=False, fontsize=7.2)
    fig.savefig(FIGS / "admission_tradeoff.pdf")
    fig.savefig(FIGS / "admission_tradeoff.png", dpi=200)
    plt.close(fig)

    # Figure 3: Routing confusions
    data = rows("action_confusion.csv")
    actions = ["accept", "reject", "revise", "defer"]
    ylabels = ["accept\n(n=170)", "reject\n(n=790)", "revise\n(n=450)", "defer\n(n=390)"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.5), layout="constrained")
    configs = [
        ("majority", "direct", "(a) Majority / direct"),
        ("fixed", "risk", "(b) Fixed / risk"),
        ("adaptive", "risk", "(c) Adaptive / risk"),
        ("dawid_skene", "risk", "(d) Dawid–Skene / risk"),
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
                    color="white" if percentages[i, j] > 55 else "black",
                )
        ax.set_xticks(range(4), actions)
        ax.set_yticks(range(4), ylabels)
        ax.set(xlabel="Recommended disposition", ylabel="Policy reference", title=title)
        ax.tick_params(length=0)
    fig.savefig(FIGS / "routing_confusions.pdf")
    fig.savefig(FIGS / "routing_confusions.png", dpi=200)
    plt.close(fig)

    # Figure 4: Cost sensitivity
    data = rows("cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for ax in axes:
        ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1.0, zorder=1)
    for j, mode in enumerate(["fixed", "adaptive", "dawid_skene"]):
        rs = sorted([r for r in data if r["method"] == mode], key=lambda r: float(r["defer_cost"]))
        for ax, k in zip(axes, ["non_defer_fraction", "far"]):
            ax.plot(
                [float(r["defer_cost"]) for r in rs],
                [100 * float(r[k]) for r in rs],
                label=["Fixed", "Adaptive", "Dawid–Skene"][j],
                color=colors[j],
                marker=["o", "s", "^"][j],
                linestyle=["-", "--", "-."][j],
                linewidth=1.6,
                markersize=4.5,
                zorder=3,
            )
            ax.set(
                xlabel="Deferral cost d",
                ylabel="Non-defer fraction (%)" if k == "non_defer_fraction" else "False-acceptance rate (%)",
                title="(a) Resolution throughput vs. cost" if k == "non_defer_fraction" else "(b) False-acceptance rate vs. cost",
                ylim=(0, 80 if k == "non_defer_fraction" else 0.9),
            )
            ax.set_xticks([0.1, 0.5, 1, 2])
            ax.grid(alpha=0.22)
    axes[0].text(0.53, 6, "d = 0.5 (default)", fontsize=7.2, color="#555555")
    axes[1].text(0.53, 0.46, "d = 0.5 (default)", fontsize=7.2, color="#555555")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.savefig(FIGS / "routing_cost_sensitivity.pdf")
    fig.savefig(FIGS / "routing_cost_sensitivity.png", dpi=200)
    plt.close(fig)
    print("Generated publication figures in figures/ directory.")


if __name__ == "__main__":
    main()
