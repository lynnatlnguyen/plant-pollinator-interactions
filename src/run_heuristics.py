"""Preliminary results: heuristic baselines on every kept network.

Run from the repo root (after the CSVs are in data/raw/):
    python -m src.run_heuristics
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import load_all
from src.heuristics import HEURISTICS, score_pairs
from src.split import check_no_leakage, split_edges

SEEDS = [0, 1, 2, 3, 4]
OUT = Path("results")


def evaluate(S, pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[score_pairs(S, pos), score_pairs(S, neg)]
    return roc_auc_score(y, s), average_precision_score(y, s)


def main():
    kept, _ = load_all()
    rows, split_info = [], []
    for nid, B in kept.items():
        for seed in SEEDS:
            s = split_edges(B, seed=seed)
            check_no_leakage(s, B)
            split_info.append({"network": nid, "seed": seed, **s.info})
            A = s.train_matrix()  # the only graph the heuristics ever see
            for name, fn in HEURISTICS.items():
                auc, ap = evaluate(fn(A), s.test_pos, s.test_neg)
                rows.append({"network": nid, "seed": seed, "method": name,
                             "roc_auc": auc, "ap": ap})

    OUT.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "heuristics_raw.csv", index=False)
    pd.DataFrame(split_info).to_csv(OUT / "split_info.csv", index=False)

    # average over seeds within each network, then mean ± std across networks
    per_net = res.groupby(["method", "network"])[["roc_auc", "ap"]].mean()
    summary = per_net.groupby("method").agg(["mean", "std"]).round(3)
    summary.to_csv(OUT / "heuristics_summary.csv")
    print(f"{len(kept)} networks x {len(SEEDS)} seeds")
    print(summary)


if __name__ == "__main__":
    main()
