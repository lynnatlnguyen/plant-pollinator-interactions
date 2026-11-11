from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

RAW_DIR = Path("data/raw")

# Filtering thresholds: justify these in your write-up
MIN_PER_SIDE = 10    # min plants AND min pollinators
MIN_LINKS = 50       # enough edges that a 20% test split is meaningful


def load_binary_matrix(path):
    """Read a Web of Life CSV (no header, no labels) and binarize: any positive weight -> 1."""
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (df > 0).astype(int)


def clean_and_filter(B):
    """Drop empty rows/cols. Return (matrix, None) if kept, else (None, reason)."""
    B = B.loc[B.sum(axis=1) > 0, B.sum(axis=0) > 0]
    n_p, n_a = B.shape
    if min(n_p, n_a) < MIN_PER_SIDE:
        return None, f"too small ({n_p}x{n_a})"
    if B.values.sum() < MIN_LINKS:
        return None, f"too few links ({B.values.sum()})"
    return B, None


def to_bipartite_graph(B):
    """NetworkX bipartite graph. Prefixes avoid name collisions between sets."""
    G = nx.Graph()
    plants = [f"P_{n}" for n in B.index]
    pollinators = [f"A_{n}" for n in B.columns]
    G.add_nodes_from(plants, bipartite=0)
    G.add_nodes_from(pollinators, bipartite=1)
    rows, cols = np.nonzero(B.values)
    G.add_edges_from((plants[i], pollinators[j]) for i, j in zip(rows, cols))
    return G


def load_all(raw_dir=RAW_DIR):
    """Return (kept: {network_id: matrix}, log: DataFrame of every file and outcome)."""
    kept, log = {}, []
    for path in sorted(Path(raw_dir).glob("M_PL_*.csv")):
        nid = path.stem
        try:
            B, reason = clean_and_filter(load_binary_matrix(path))
        except Exception as e:
            B, reason = None, f"read error: {e}"
        if B is None:
            log.append({"network": nid, "kept": False, "reason": reason})
        else:
            kept[nid] = B
            log.append({"network": nid, "kept": True, "reason": ""})
    return kept, pd.DataFrame(log)