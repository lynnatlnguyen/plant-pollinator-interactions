"""Bipartite link-prediction heuristics. Input A is ALWAYS split.train_matrix().

Plants are rows, pollinators are columns. Common-neighbour scores are always zero
between a plant and a pollinator, so we score length-3 paths
    plant u -> pollinator a -> plant b -> pollinator v.
For any pair (u, v) that is not a train edge, every such walk is a simple path and
both intermediate nodes have degree >= 2.
"""
import numpy as np
from scipy import sparse


def degrees(A):
    return A.sum(axis=1), A.sum(axis=0)  # plant degrees, pollinator degrees


def _safe_pow(k, p):
    out = np.zeros(len(k), dtype=float)
    ok = k > 0
    out[ok] = k[ok] ** p
    return out


def _weighted_l3(A, w_poll, w_plant):
    """Sum over paths u-a-b-v of w_poll[a] * w_plant[b], i.e. A Wa A^T Wp A."""
    A = sparse.csr_matrix(A)
    S = A @ sparse.diags(w_poll) @ A.T @ sparse.diags(w_plant) @ A
    return S.toarray()


def l3(A):
    """Raw number of length-3 paths."""
    kp, ka = degrees(A)
    return _weighted_l3(A, np.ones_like(ka), np.ones_like(kp))


def l3_normalized(A):
    """Kovacs et al. (2019): each path weighted by 1 / sqrt(k_a * k_b)."""
    kp, ka = degrees(A)
    return _weighted_l3(A, _safe_pow(ka, -0.5), _safe_pow(kp, -0.5))


def resource_allocation_l3(A):
    """Each path weighted by 1 / (k_a * k_b)."""
    kp, ka = degrees(A)
    return _weighted_l3(A, _safe_pow(ka, -1.0), _safe_pow(kp, -1.0))


def adamic_adar_l3(A):
    """Each path weighted by 1 / (log k_a * log k_b). Degree-1 nodes never sit in the
    middle of a path between an unobserved pair (see module docstring), so weight 0."""
    kp, ka = degrees(A)
    inv_log = lambda k: np.where(k > 1, 1.0 / np.log(np.maximum(k, 2)), 0.0)
    return _weighted_l3(A, inv_log(ka), inv_log(kp))


def preferential_attachment(A):
    """Plant degree x pollinator degree."""
    kp, ka = degrees(A)
    return np.outer(kp, ka)


HEURISTICS = {
    "L3": l3,
    "L3_norm": l3_normalized,
    "RA3": resource_allocation_l3,
    "AA3": adamic_adar_l3,
    "PA": preferential_attachment,
}


def score_pairs(S, pairs):
    """Look up scores for an array of (plant_idx, pollinator_idx) pairs."""
    return S[pairs[:, 0], pairs[:, 1]]
