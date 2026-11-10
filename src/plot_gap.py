"""Figure: per-network advantage of PA over normalized L3 vs. connectance and nestedness.

Run from the repo root after src.run_heuristics and src.descriptors:
    python -m src.plot_gap
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

RESULTS = Path("results")
FIG = Path("figures") / "gap_vs_structure.png"

POINT = "#2a78d6"
INK, MUTED = "#1f1f1d", "#6b6a63"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PANELS = [("connectance", "Connectance"), ("NODF", "Nestedness (NODF)")]


def main():
    res = pd.read_csv(RESULTS / "heuristics_raw.csv")
    per_net = res.groupby(["network", "method"]).roc_auc.mean().unstack()
    gap = (per_net["PA"] - per_net["L3_norm"]).rename("gap")
    df = pd.read_csv(RESULTS / "network_descriptors.csv", index_col="network").join(gap, how="inner")

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
    for ax, (col, label) in zip(axes, PANELS):
        ax.axhline(0, color=AXIS, lw=1, ls="--", zorder=1)
        ax.scatter(df[col], df["gap"], s=22, color=POINT, alpha=0.75,
                   edgecolor="white", linewidth=0.6, zorder=2)
        rho, p = spearmanr(df[col], df["gap"])
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.set_title(f"Spearman ρ = {rho:.2f}, {p_txt}", loc="left",
                     color=MUTED, fontsize=8)
        ax.set_xlabel(label)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    axes[0].set_ylabel("ROC-AUC gap\n(PA − normalized L3)")
    axes[0].text(0.02, 0.02, "below line: L3 better", transform=axes[0].transAxes,
                 color=MUTED, fontsize=7, va="bottom")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300)
    print(f"Saved {FIG} ({len(df)} networks)")


if __name__ == "__main__":
    main()
