import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
from src.data import load_all, to_bipartite_graph
from pathlib import Path


def nodf(B):
    """NODF nestedness (Almeida-Neto et al. 2008), scaled 0-100."""
    B = np.asarray(B)

    def paired_overlap(M):
        deg = M.sum(axis=1)
        shared = M @ M.T
        vals = []
        for i in range(len(deg)):
            for j in range(i + 1, len(deg)):
                if deg[i] == deg[j]:
                    vals.append(0.0)           # equal fill contributes 0
                else:
                    lo = i if deg[i] < deg[j] else j
                    vals.append(shared[i, j] / deg[lo])
        return vals

    all_vals = paired_overlap(B) + paired_overlap(B.T)
    return 100 * float(np.mean(all_vals))


def modularity(G, seed=0):
    """Louvain modularity on the bipartite graph (approximation, see note)."""
    comms = community.louvain_communities(G, seed=seed)
    return community.modularity(G, comms)


def describe(B):
    n_p, n_a = B.shape
    L = int(B.values.sum())
    G = to_bipartite_graph(B)
    return {
        "plants": n_p,
        "pollinators": n_a,
        "size": n_p + n_a,
        "links": L,
        "connectance": L / (n_p * n_a),
        "NODF": nodf(B.values),
        "modularity": modularity(G),
    }


if __name__ == "__main__":
    kept, log = load_all()
    log.to_csv("results/filter_log.csv", index=False)
    table = pd.DataFrame({k: describe(B) for k, B in kept.items()}).T
    table.index.name = "network"
    table.round(3).to_csv("results/network_descriptors.csv")
    print(f"Kept {len(kept)} of {len(log)} networks")
    print(log[~log.kept].reason.value_counts())
    print(table.describe().round(3))

    # Save the binarized, filtered matrices
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    for nid, B in kept.items():
        B.to_csv(out / f"{nid}_binary.csv")