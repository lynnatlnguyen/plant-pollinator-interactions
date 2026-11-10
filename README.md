# Plant–Pollinator Link Prediction

ELEC 573 project. We ask whether Node2Vec embeddings
recover masked plant–pollinator interactions better than simple bipartite heuristics, and how
the answer depends on network structure (size, connectance, nestedness, modularity).

All results are interpreted as **recovery of masked recorded links**, not discovery of
true unobserved interactions.

## Status

| Stage | Status |
|---|---|
| Data loading, filtering, descriptors, EDA | Done |
| Leakage-safe split + negative sampling | Done |
| Path-based heuristics (L3 family) + preferential attachment | Done |
| Degree-matched negatives | Planned |
| Node2Vec + classifiers, tuning on validation | Planned |
| Structural analysis of all methods | In progress (heuristics only) |

## Pipeline

```
Web of Life matrices
      │  binarize, drop empty rows/cols, filter small networks
      ▼
Network descriptors ── size, connectance, NODF, modularity
      │
      ▼
Per-network edge split ── positives → train / val / test (70/10/20)
      │                    every species keeps >= 1 train edge
      ▼
Negative sampling ─────── disjoint non-edge sets for train / val / test
      │                    drawn from non-edges of the FULL network
      │                    negatives are labels only, never added to any graph
      ▼
Training graph ────────── train positive edges ONLY
      │
      ├──► Heuristics: L3, normalized L3, RA3, AA3 (length-3 paths),
      │                preferential attachment
      │
      └──► Node2Vec (train graph) → edge features (Hadamard, concat, L1, L2)   [planned]
                 → LogReg / RandomForest (train pos + train neg)
                 → plus raw embedding dot-product score
      ▼
Tuning ────────────────── Node2Vec + classifier params on VAL only
      ▼
Final test ────────────── score TEST once: ROC-AUC, Average Precision
      │                    5 seeds × all networks → mean ± std
      ▼
Analysis ──────────────── performance vs. connectance / nestedness / size / modularity
```

Why length-3 paths: in a bipartite graph a plant and a pollinator share no common neighbors,
so common-neighbor-style scores are always zero. The shortest connection between a plant and a
pollinator is a path of length 3 (plant → pollinator → plant → pollinator).

## Data

Plant–pollinator networks from the [Web of Life](https://www.web-of-life.es/) database,
stored as `data/raw/M_PL_*.csv` (174 networks; plants are rows, pollinators are columns).
Interaction weights are binarized (any positive value → 1). Networks are kept only if they have
at least 10 plants, 10 pollinators and 50 links; 91 networks pass the filter
(see `results/filter_log.csv`).

## Layout

```
data/raw/           Web of Life CSVs
data/processed/     binarized, filtered matrices (written by src.descriptors)
src/
  data.py           load, binarize, filter networks
  descriptors.py    size, connectance, NODF, modularity → results/network_descriptors.csv
  eda.py            descriptor summary, correlations, histograms
  split.py          leakage-safe edge split, negative sampling, leakage checks
  heuristics.py     L3, normalized L3, RA3, AA3, preferential attachment
  run_heuristics.py evaluates all heuristics on all networks × 5 seeds
  plot_gap.py       PA vs. normalized L3 gap against connectance / nestedness
notebooks/          exploration
results/            CSV outputs
figures/            PNG figures
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

From the repo root, in this order:

```bash
python -m src.descriptors      # filter networks, compute descriptors
python -m src.eda              # descriptor summary + figures
python -m src.run_heuristics   # heuristic baselines → results/heuristics_*.csv
python -m src.plot_gap         # figures/gap_vs_structure.png
```

## Preliminary results

Test ROC-AUC and average precision across 91 networks (each averaged over 5 seeds):

| Method | ROC-AUC | AP |
|---|---|---|
| L3 | 0.711 ± 0.086 | 0.709 ± 0.084 |
| Normalized L3 | 0.726 ± 0.089 | 0.743 ± 0.084 |
| RA3 | 0.726 ± 0.090 | 0.742 ± 0.085 |
| AA3 | 0.725 ± 0.089 | 0.742 ± 0.085 |
| Preferential attachment | **0.806 ± 0.058** | **0.801 ± 0.057** |

- **Preferential attachment wins on 83 of 91 networks.** Under uniform negative sampling,
  held-out interactions mostly involve generalists, so degree alone is a strong signal.
- **Path-based scores close the gap in denser and more nested networks** (Spearman
  ρ = −0.39 with connectance, ρ = −0.34 with NODF; both p ≤ 0.001).
- **Next:** degree-matched negatives to separate structure from popularity, then Node2Vec,
  which has to beat preferential attachment to show it learns more than degree.
