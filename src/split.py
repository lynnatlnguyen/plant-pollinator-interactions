"""Leakage-safe edge split and negative sampling for one bipartite network.

Input B: 0/1 matrix (plants x pollinators), e.g. a DataFrame from data.load_all().
Every edge array holds integer positions (plant_idx, pollinator_idx) into B;
use B.index / B.columns to get species names back.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Split:
    shape: tuple
    train_pos: np.ndarray  # (k, 2) array of (plant_idx, pollinator_idx)
    val_pos: np.ndarray
    test_pos: np.ndarray
    train_neg: np.ndarray
    val_neg: np.ndarray
    test_neg: np.ndarray
    info: dict = field(default_factory=dict)

    def train_matrix(self):
        """Biadjacency from train positives ONLY. The only graph any method may see."""
        A = np.zeros(self.shape, dtype=float)
        A[self.train_pos[:, 0], self.train_pos[:, 1]] = 1
        return A


def split_edges(B, fracs=(0.7, 0.1, 0.2), neg_ratio=1, seed=0):
    B = np.asarray(B)
    rng = np.random.default_rng(seed)
    n_p, n_a = B.shape

    pos = np.argwhere(B > 0)
    pos = pos[rng.permutation(len(pos))]
    E = len(pos)

    # 1) Coverage: walk the shuffled edges and force an edge into train if either
    #    endpoint has no train edge yet, so every node keeps >= 1 train edge.
    forced = np.zeros(E, dtype=bool)
    plant_seen = np.zeros(n_p, dtype=bool)
    poll_seen = np.zeros(n_a, dtype=bool)
    for k, (i, j) in enumerate(pos):
        if not (plant_seen[i] and poll_seen[j]):
            forced[k] = True
            plant_seen[i] = poll_seen[j] = True

    # 2) Fill train up to its quota, then split the rest into val/test.
    n_train_target = int(round(fracs[0] * E))
    rest = pos[~forced]
    n_extra = max(0, n_train_target - int(forced.sum()))
    train_pos = np.vstack([pos[forced], rest[:n_extra]])
    remaining = rest[n_extra:]
    n_val = int(round(len(remaining) * fracs[1] / (fracs[1] + fracs[2])))
    val_pos, test_pos = remaining[:n_val], remaining[n_val:]

    # 3) Negatives: non-edges of the FULL network, disjoint across splits.
    non = np.argwhere(B == 0)
    non = non[rng.permutation(len(non))]
    sizes = np.array([len(train_pos), len(val_pos), len(test_pos)]) * neg_ratio
    if sizes.sum() > len(non):
        warnings.warn(f"only {len(non)} non-edges; scaling negatives down")
        sizes = (sizes * len(non) / sizes.sum()).astype(int)
    c = np.cumsum(sizes)
    train_neg, val_neg, test_neg = non[:c[0]], non[c[0]:c[1]], non[c[1]:c[2]]

    info = {
        "n_edges": E,
        "n_forced": int(forced.sum()),
        "frac_train": len(train_pos) / E,
        "frac_val": len(val_pos) / E,
        "frac_test": len(test_pos) / E,
        "quota_exceeded": bool(forced.sum() > n_train_target),
        # degree-1 nodes: their only edge is always in train, so never evaluated
        "deg1_plants": int((B.sum(axis=1) == 1).sum()),
        "deg1_pollinators": int((B.sum(axis=0) == 1).sum()),
    }
    return Split(B.shape, train_pos, val_pos, test_pos,
                 train_neg, val_neg, test_neg, info)


def check_no_leakage(s, B):
    """Raise AssertionError if any leakage rule is broken."""
    B = np.asarray(B)
    as_set = lambda a: set(map(tuple, a))
    tr, va, te = as_set(s.train_pos), as_set(s.val_pos), as_set(s.test_pos)
    ntr, nva, nte = as_set(s.train_neg), as_set(s.val_neg), as_set(s.test_neg)
    full = as_set(np.argwhere(B > 0))

    assert len(tr) + len(va) + len(te) == len(full) and tr | va | te == full, \
        "positive splits are not a partition of the edges"
    assert not ((ntr | nva | nte) & full), "a negative is a real edge"
    assert not (ntr & nva or ntr & nte or nva & nte), "negative splits overlap"
    A = s.train_matrix()
    assert as_set(np.argwhere(A > 0)) == tr, "train graph contains non-train edges"
    assert (A.sum(axis=1) > 0).all() and (A.sum(axis=0) > 0).all(), \
        "a node has no train edge"
