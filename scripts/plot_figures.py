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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    for ax, router in zip(axes, ["direct", "risk"]):
        labels = (
            [("majority", "Majority"), ("reliability_weighted", "Reliability / adaptive"), ("dawid_skene", "Dawid-Skene")]
            if router == "direct"
            else [("fixed", "Fixed / majority / reliability"), ("adaptive", "Adaptive"), ("dawid_skene", "Dawid-Skene")]
        )
        for j, (mode, label) in enumerate(labels):
            r = metrics[(mode, router)]
            x = 100 * float(r["non_defer_fraction"])
            y = 100 * float(r["far"])
            ax.scatter(x, y, s=45, color=colors[j], marker=["o", "s", "^"][j], label=label, zorder=3)
        ax.set(
            xlabel="Non-defer fraction (%)",
            ylabel="False acceptance (%)",
            title="Direct mapping" if router == "direct" else "Risk routing",
            xlim=(0, 105),
            ylim=(0, 16 if router == "direct" else 0.22),
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left" if router == "direct" else "upper left", frameon=False, fontsize=6.8)
    fig.savefig(FIGS / "admission_tradeoff.pdf")
    fig.savefig(FIGS / "admission_tradeoff.png", dpi=180)
    plt.close(fig)

    # Figure 3: Routing confusions
    data = rows("action_confusion.csv")
    actions = ["accept", "reject", "revise", "defer"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.3), layout="constrained")
    configs = [
        ("majority", "direct", "Majority / direct"),
        ("fixed", "risk", "Fixed / risk"),
        ("adaptive", "risk", "Adaptive / risk"),
        ("dawid_skene", "risk", "Dawid-Skene / risk"),
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
                    f"{matrix[i,j]}\n{percentages[i,j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if percentages[i, j] > 55 else "black",
                )
        ax.set_xticks(range(4), actions)
        ax.set_yticks(range(4), actions)
        ax.set(xlabel="Recommended action", ylabel="Policy reference", title=title)
        ax.tick_params(length=0)
    fig.savefig(FIGS / "routing_confusions.pdf")
    fig.savefig(FIGS / "routing_confusions.png", dpi=180)
    plt.close(fig)

    # Figure 4: Cost sensitivity
    data = rows("cost_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    for j, mode in enumerate(["fixed", "adaptive", "dawid_skene"]):
        rs = sorted([r for r in data if r["method"] == mode], key=lambda r: float(r["defer_cost"]))
        for ax, k in zip(axes, ["non_defer_fraction", "far"]):
            ax.plot(
                [float(r["defer_cost"]) for r in rs],
                [100 * float(r[k]) for r in rs],
                label=["Fixed", "Adaptive", "Dawid-Skene"][j],
                color=colors[j],
                marker=["o", "s", "^"][j],
                linestyle=["-", "--", "-."][j],
                linewidth=1.5,
                markersize=4,
            )
            ax.set(
                xlabel="Deferral cost d",
                ylabel="Non-defer fraction (%)" if k == "non_defer_fraction" else "False acceptance (%)",
                ylim=(0, 80 if k == "non_defer_fraction" else 0.9),
            )
            ax.set_xticks([0.1, 0.5, 1, 2])
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(FIGS / "routing_cost_sensitivity.pdf")
    fig.savefig(FIGS / "routing_cost_sensitivity.png", dpi=180)
    plt.close(fig)
    print("Generated figures in", FIGS)


if __name__ == "__main__":
    main()
