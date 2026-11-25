"""Generate publication-grade Plan 1 & Plan 2 figures (300 DPI PNG + vector PDF).

Strict typographic rules:
  - Pure black (#000000) for all axes, ticks, titles, legends, and annotations
  - Bold centered sub-captions (a) ..., (b) ... beneath each panel
  - Clean colorblind-safe Okabe-Ito / Tol academic palette
  - Zero overlapping labels or clipped margins
"""
from __future__ import annotations

import csv
import json
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
        "font.size": 9.5,
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


def plot_correlated_stress_and_noisy_gate() -> None:
    stress = read_csv("correlated_verifier_stress.csv")
    noisy = read_csv("noisy_gate_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.91, bottom=0.22, wspace=0.32)

    # --- Panel (a): Correlated-Verifier Collapse Stress Test ---
    rhos = sorted({float(r["rho"]) for r in stress})
    series = [
        ("deciv_ds_raw", "DeCiv + Hard-DS (raw)", "#C0392B", "o", "-"),
        ("deciv_adaptive_085", "DeCiv + Adaptive (0.85)", "#E67E22", "^", "--"),
        ("deciv_pg_crc", "DeCiv + DS + PG-CRC (α=0.01)", "#1F618D", "s", "-"),
        ("deciv_ds_temp", "DeCiv + DS + TempScale", "#1E8449", "D", "-"),
    ]
    for mid, lbl, col, mk, ls in series:
        sub = {float(r["rho"]): r for r in stress if r["method_id"] == mid}
        ys = np.array([float(sub[rh]["far_mean"]) * 100.0 for rh in rhos])
        es = np.array([float(sub[rh]["far_std"]) * 100.0 for rh in rhos])
        ax1.plot(rhos, ys, label=lbl, color=col, marker=mk, linestyle=ls, linewidth=2.0, markersize=5.5)
        ax1.fill_between(rhos, np.maximum(0.0, ys - es), ys + es, color=col, alpha=0.14)

    ax1.axhline(1.0, color="#7F1D1D", linestyle=":", linewidth=1.4, label="PG-CRC target bound (α = 1.0%)")
    ax1.set_xlabel("Pairwise Verifier Error Correlation ρ", fontweight="bold")
    ax1.set_ylabel("False Acceptance Rate — FAR (%)", fontweight="bold")
    ax1.set_xlim(-0.02, 0.62)
    ax1.set_ylim(-0.15, 5.2)
    ax1.grid(True, alpha=0.6)

    # Secondary y-axis: automatically fitted temperature T* for DS + TempScale
    ax1_r = ax1.twinx()
    sub_t = {float(r["rho"]): float(r["mean_fitted_temp"]) for r in stress if r["method_id"] == "deciv_ds_temp"}
    ts = [sub_t[rh] for rh in rhos]
    ax1_r.plot(rhos, ts, color="#5B2C6F", linestyle="-.", marker="x", linewidth=1.5, markersize=5.5, label="Fitted Temp T* (right axis)")
    ax1_r.set_ylabel("Fitted Temperature T* (DS + TempScale)", fontweight="bold", color="#000000")
    ax1_r.set_ylim(0.5, 6.2)
    ax1_r.tick_params(axis="y", colors="#000000")

    # Combine legends cleanly
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="#000000", framealpha=0.95)
    ax1.annotate("64× FAR surge\n(0.07% → 4.48%)", xy=(0.60, 4.48), xytext=(0.39, 4.35),
                 fontsize=7.8, fontweight="bold", color="#000000",
                 arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))

    # --- Panel (b): Noisy Metadata Gate Sensitivity ---
    sub_n = [r for r in noisy if r["method"] == "ds_temp"]
    etas = [float(r["noise_rate"]) * 100.0 for r in sub_n]
    fars = [float(r["far_mean"]) * 100.0 for r in sub_n]
    evrs = [float(r["evr_mean"]) * 100.0 for r in sub_n]
    costs = [float(r["cost_mean"]) for r in sub_n]
    cost_stds = [float(r["cost_std"]) for r in sub_n]

    ax2.plot(etas, fars, color="#1F618D", marker="o", linewidth=2.1, markersize=6, label="False Acceptance Rate — FAR (%)")
    ax2.plot(etas, evrs, color="#D35400", marker="s", linestyle="--", linewidth=2.1, markersize=5.5, label="Eligibility Violation Rate — EVR (%)")
    ax2.set_xlabel("Metadata Detector Noise Rate η (%)", fontweight="bold")
    ax2.set_ylabel("Safety Violation Rate (%)", fontweight="bold")
    ax2.set_xlim(-0.4, 10.6)
    ax2.set_ylim(-0.02, 0.42)
    ax2.grid(True, alpha=0.6)

    ax2_r = ax2.twinx()
    ax2_r.errorbar(etas, costs, yerr=cost_stds, color="#1E8449", marker="^", linestyle="-.", linewidth=1.8, capsize=3, label="Mean Institutional Cost (right axis)")
    ax2_r.set_ylabel("Mean Institutional Cost (↓)", fontweight="bold", color="#000000")
    ax2_r.set_ylim(0.20, 0.30)
    ax2_r.tick_params(axis="y", colors="#000000")

    h1b, l1b = ax2.get_legend_handles_labels()
    h2b, l2b = ax2_r.get_legend_handles_labels()
    ax2.legend(h1b + h2b, l1b + l2b, loc="upper left", fontsize=7.8, frameon=True, facecolor="white", edgecolor="#000000", framealpha=0.95)
    ax2.annotate("At η = 10% noise:\nFAR = 0.09%, EVR = 0.28%", xy=(9.9, 0.28), xytext=(3.0, 0.255),
                 fontsize=7.8, fontweight="bold", color="#000000",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#000000", alpha=0.92, lw=0.6),
                 arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))

    fig.text(0.265, 0.035, "(a) Correlated-Verifier Collapse & PG-CRC / TempScale Mitigation", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#000000")
    fig.text(0.745, 0.035, "(b) Noisy Metadata Detector Sensitivity (20-Seed Mean ± SD)", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#000000")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"correlated_stress_and_noisy_gate.{ext}", dpi=300)
    plt.close(fig)
    print("Saved correlated_stress_and_noisy_gate.{pdf,png}")


def plot_real_verifiers_and_correlation() -> None:
    corr_rows = read_csv("real_verifier_correlation.csv")
    adm_rows = {r["method_id"]: r for r in read_csv("real_admission_metrics.csv")}

    fig = plt.figure(figsize=(12.2, 4.65), dpi=300)
    # Explicit non-overlapping axes: Panel (a) + colorbar on left, generous margin, Panel (b) on right
    ax1 = fig.add_axes([0.11, 0.22, 0.28, 0.70])
    cax = fig.add_axes([0.402, 0.22, 0.015, 0.70])
    ax2 = fig.add_axes([0.655, 0.22, 0.325, 0.70])

    # --- Panel (a): 4x4 Empirical Error Correlation Heatmap ---
    vids = ["deberta_v3_large", "bart_large_mnli", "llama31_8b", "qwen35_9b"]
    vlabels = ["DeBERTa-v3\n(Encoder NLI)", "BART-Large\n(Enc-Dec NLI)", "Llama-3.1-8B\n(Decoder LLM)", "Qwen3.5-9B\n(Decoder LLM)"]
    mat = np.zeros((4, 4), dtype=float)
    lookup = {(r["verifier_i"], r["verifier_j"]): float(r["error_pearson_rho"]) for r in corr_rows}
    for i, vi in enumerate(vids):
        for j, vj in enumerate(vids):
            mat[i, j] = lookup[(vi, vj)]

    im = ax1.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(vlabels, fontsize=7.6, fontweight="bold", rotation=18, ha="right")
    ax1.set_yticklabels(vlabels, fontsize=7.6, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            txt = "1.000" if i == j else f"{val:.3f}"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=9.0, fontweight="bold",
                     color="#000000" if val < 0.75 else "#FFFFFF")

    ax1.axhline(1.5, color="#000000", linewidth=1.6, linestyle="--")
    ax1.axvline(1.5, color="#000000", linewidth=1.6, linestyle="--")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Error Pearson Correlation ρ̂", fontsize=8.2, fontweight="bold", color="#000000")
    cbar.ax.tick_params(labelsize=8, colors="#000000")

    # --- Panel (b): Real 4-Model Out-of-Fold Cost & FAR Comparison ---
    show = [
        ("solo_llama31_8b_direct", "Solo Llama-3.1-8B / direct", "#95A5A6"),
        ("majority_direct", "Majority Vote / direct", "#7F8C8D"),
        ("conformal_risk", "Conformal-Risk (ICLR '24)", "#D35400"),
        ("posthoc_l2d", "Post-Hoc Cost L2D ('22)", "#CA6F1E"),
        ("deciv_hard_ds_raw", "DeCiv + Hard-DS (raw)", "#C0392B"),
        ("conformal_risk_gated", "Conformal-Risk + Gate", "#E67E22"),
        ("deciv_hard_ds_pg_crc_010", "DeCiv + Hard-DS + PG-CRC", "#2980B9"),
        ("deciv_soft_ds_sup_temp", "DeCiv + Sup-Soft-DS + Temp", "#1F618D"),
        ("deciv_sup_pg_crc_005", "DeCiv + Sup-Soft-DS + PG-CRC", "#1E8449"),
    ]
    y_pos = np.arange(len(show))
    costs = [float(adm_rows[m]["mean_institutional_cost"]) for m, _, _ in show]
    fars = [float(adm_rows[m]["far"]) * 100.0 for m, _, _ in show]
    cols = [c for _, _, c in show]
    labels = [l for _, l, _ in show]

    bars = ax2.barh(y_pos, costs, color=cols, edgecolor="#000000", linewidth=0.7, height=0.66)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8.1, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Realized Institutional Cost (↓ Lower is Better)", fontweight="bold")
    ax2.set_xlim(0, 8.65)
    ax2.grid(True, axis="x", alpha=0.6)

    for bar, cost, far in zip(bars, costs, fars):
        ax2.text(cost + 0.12, bar.get_y() + bar.get_height() / 2.0,
                 f"{cost:.3f} (FAR {far:.1f}%)", va="center", ha="left",
                 fontsize=7.6, fontweight="bold", color="#000000")

    fig.text(0.255, 0.032, "(a) Empirical Inter-Model Error Correlation (DeCiv-Real-1200)", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#000000")
    fig.text(0.75, 0.032, "(b) Real 4-Model Admission Risk & FAR (4-Fold LODO CV)", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#000000")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"real_verifiers_and_correlation.{ext}", dpi=300)
    plt.close(fig)
    print("Saved real_verifiers_and_correlation.{pdf,png}")


def plot_rag_defense_and_remediation() -> None:
    rag_rows = {r["regime_id"]: r for r in read_csv("real_rag_defense_metrics.csv")}
    rem_recs = [json.loads(l) for l in (RESULTS / "real_remediation_samples.jsonl").open(encoding="utf-8")]
    merkle = read_csv("merkle_scaling.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.55), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.96, top=0.90, bottom=0.25, wspace=0.28)

    # --- Panel (a): End-to-End Dense RAG Contamination Defense ---
    regimes = [
        ("no_filter", "No Filter\n(Unfiltered)"),
        ("majority_direct", "Majority\nDirect"),
        ("conformal_risk", "Conformal\n(ICLR '24)"),
        ("deciv_hard_ds_raw", "DeCiv\nHard-DS\n(raw)"),
        ("deciv_soft_ds_sup_temp", "DeCiv\nSup-Soft\n+ Temp"),
        ("deciv_plus_revise", "DeCiv\n+ Revise\nLoop"),
        ("deciv_sup_pg_crc_005", "DeCiv +\nPG-CRC\n(α=0.005)"),
    ]
    x = np.arange(len(regimes))
    w = 0.26
    contam = [float(rag_rows[r]["store_contamination_rate"]) * 100.0 for r, _ in regimes]
    pois_r = [float(rag_rows[r]["poison_top3_retrieval_rate"]) * 100.0 for r, _ in regimes]
    asr = [float(rag_rows[r]["downstream_poisoning_asr"]) * 100.0 for r, _ in regimes]

    b1 = ax1.bar(x - w, contam, width=w, color="#C0392B", edgecolor="#000000", linewidth=0.6, label="Store Contamination (%)")
    b2 = ax1.bar(x, pois_r, width=w, color="#E67E22", edgecolor="#000000", linewidth=0.6, label="Poisoned Top-3 Retrieval (%)")
    b3 = ax1.bar(x + w, asr, width=w, color="#1F618D", edgecolor="#000000", linewidth=0.6, label="Downstream Poisoning ASR (↓ %)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.3, f"{h:.1f}" if h < 10 else f"{h:.0f}",
                     ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#000000")

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in regimes], fontsize=7.3, fontweight="bold")
    ax1.set_ylabel("Contamination / Poisoning Rate (%)", fontweight="bold")
    ax1.set_ylim(0, 104)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white", edgecolor="#000000")

    ax1.annotate("0.0% ASR\n(Oracle parity)", xy=(6 + w, 5.5), xytext=(4.85, 36),
                 fontsize=7.8, fontweight="bold", color="#000000",
                 arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))

    # --- Panel (b): Automated revise-Queue Re-Verification Quality + Merkle Scaling Inset ---
    readm_nli = [r["nli_ev_supports_repaired"] for r in rem_recs if r["reverified_action"] == "accept"]
    defer_nli = [r["nli_ev_supports_repaired"] for r in rem_recs if r["reverified_action"] != "accept"]

    bins = np.linspace(0.0, 1.0, 11)
    ax2.hist(readm_nli, bins=bins, color="#1E8449", alpha=0.82, edgecolor="#000000", linewidth=0.7,
             label=f"Re-Admitted (n={len(readm_nli)}, mean={np.mean(readm_nli):.3f})")
    ax2.hist(defer_nli, bins=bins, color="#C0392B", alpha=0.80, edgecolor="#000000", linewidth=0.7,
             label=f"Blocked / Deferred (n={len(defer_nli)}, mean={np.mean(defer_nli):.3f})")
    ax2.axvline(0.65, color="#000000", linestyle="--", linewidth=1.2, label="Entailment threshold (0.65)")
    ax2.set_xlabel("Primary-Evidence Entailment Score P(Ev ⇒ Repaired)", fontweight="bold")
    ax2.set_ylabel("Number of Repaired Claims", fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0, 106)
    ax2.grid(True, axis="y", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="#000000")

    # Compact clean summary box inside Panel (b) showing Merkle O(log N) verification
    m50 = [r for r in merkle if int(r["log_records"]) == 50000][0]
    info_txt = (
        f"Closed-Loop Remediation:\n"
        f"• 95.1% re-admitted grounded\n"
        f"• Sup-100 RAG Acc: 73% → 76%\n"
        f"• Merkle proofs: 124/124 OK\n"
        f"• At N=50k: {m50['inclusion_proof_hashes']} hashes "
        f"({m50['inclusion_verify_us']} µs)"
    )
    ax2.text(0.04, 0.57, info_txt, transform=ax2.transAxes, fontsize=7.4, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9F9", edgecolor="#000000", linewidth=0.8))

    fig.text(0.30, 0.035, "(a) End-to-End Dense RAG Defense (BGE-large + FAISS + Llama-3.1-8B)", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#000000")
    fig.text(0.795, 0.035, "(b) Automated revise Re-Verification & Merkle Lineage", ha="center", va="center", fontsize=8.9, fontweight="bold", color="#000000")

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"rag_defense_and_remediation.{ext}", dpi=300)
    plt.close(fig)
    print("Saved rag_defense_and_remediation.{pdf,png}")


def main():
    plot_correlated_stress_and_noisy_gate()
    plot_real_verifiers_and_correlation()
    plot_rag_defense_and_remediation()
    print("ALL UPGRADED PUBLICATION FIGURES GENERATED.")


if __name__ == "__main__":
    main()
