"""EDA on network descriptors: correlations (Spearman) and histograms.

Run from the repo root after src.descriptors has produced the CSV:
    python -m src.eda
"""
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns

DESC_CSV = Path("results/network_descriptors.csv")
FIG_DIR = Path("figures")
COLS = ["size", "connectance", "NODF", "modularity"]


def plot_histograms(df, path):
    fig, axes = plt.subplots(1, len(COLS), figsize=(4 * len(COLS), 3.2))
    for ax, col in zip(axes, COLS):
        ax.hist(df[col], bins=min(15, max(5, len(df) // 4)), edgecolor="black")
        ax.axvline(df[col].median(), color="red", ls="--", lw=1, label="median")
        ax.set_xlabel(col)
        ax.set_ylabel("# networks")
    axes[0].legend()
    fig.suptitle(f"Descriptor distributions (n = {len(df)} networks)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(corr, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1,
                cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Spearman correlation of descriptors")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pairplot(df, path):
    plot_df = df[COLS].copy()
    g = sns.pairplot(plot_df, corner=True, diag_kind="hist",
                     plot_kws={"s": 20, "alpha": 0.7})
    g.savefig(path, dpi=200)
    plt.close(g.figure)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DESC_CSV, index_col="network")
    print(f"{len(df)} networks")

    # Summary stats (use these for the Data section table)
    summary = df[COLS].agg(["min", "median", "max"]).T.round(3)
    summary.to_csv("results/descriptor_summary.csv")
    print(summary)

    # Correlations
    corr = df[COLS].corr(method="spearman")
    corr.round(3).to_csv("results/descriptor_correlations.csv")
    print("\nSpearman correlations:\n", corr.round(2))

    # Flag strongly collinear pairs: these can't be separated in Section 7
    print("\nPairs with |rho| > 0.7 (confounded in later analysis):")
    flagged = False
    for i, a in enumerate(COLS):
        for b in COLS[i + 1:]:
            if abs(corr.loc[a, b]) > 0.7:
                print(f"  {a} vs {b}: rho = {corr.loc[a, b]:.2f}")
                flagged = True
    if not flagged:
        print("  none")

    plot_histograms(df, FIG_DIR / "descriptor_histograms.png")
    plot_correlation_heatmap(corr, FIG_DIR / "descriptor_correlations.png")
    plot_pairplot(df, FIG_DIR / "descriptor_pairplot.png")
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()