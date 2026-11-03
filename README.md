# Plant–Pollinator Link Prediction

ELEC 573 project. We ask whether Node2Vec embeddings recover masked plant–pollinator
interactions better than simple bipartite heuristics, and how the answer depends on network
structure (size, connectance, nestedness, modularity).

All results are interpreted as **recovery of masked recorded links**, not discovery of
true unobserved interactions.

## Pipeline

```
Web of Life matrices
      │  download, binarize, filter tiny networks
      ▼
Network descriptors ── size, connectance, NODF, modularity, degree skew
      │
      ▼
Per-network edge split ── positives → train / val / test (70/10/20)
      │                    no node loses all train edges (or logged)
      ▼
Negative sampling ─────── disjoint non-edge sets for train / val / test
      │                    val/test negatives = non-edges of the FULL network
      │                    negatives are labels only, never added to any graph
      ▼
Training graph ────────── train positive edges ONLY
      │
      ├──► Heuristics: L3 path count, degree-normalized L3, preferential attachment,
      │                bipartite RA / AA over length-3 paths
      │
      └──► Node2Vec (train graph) → edge features (Hadamard, concat, L1, L2)
                 → LogReg / RandomForest (train pos + train neg)
                 → plus raw embedding dot-product score
      ▼
Tuning ────────────────── p, q, dim, walk length, classifier params on VAL only
      ▼
Final test ────────────── score TEST once: ROC-AUC, Average Precision
      │                    repeat over seeds × networks → mean ± std
      ▼
Analysis ──────────────── performance vs. nestedness / connectance / size
```

Why length-3 paths: in a bipartite graph a plant and a pollinator share no common neighbors,
so common-neighbor-style scores are always zero. The shortest non-trivial plant→pollinator
connection is length 3 (plant–pollinator–plant–pollinator).

## Layout

```
data/raw/           downloaded Web of Life files (gitignored)
data/processed/     binarized/filtered matrices, descriptors (gitignored)
src/                project code
notebooks/          exploration
results/, figures/  outputs (gitignored by default)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
