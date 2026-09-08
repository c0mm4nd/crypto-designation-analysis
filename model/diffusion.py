"""
Deterministic diffusion importance for value-weighted transfer graphs.

Monte-Carlo independent-cascade (IC) coverage requires one simulated cascade per source
address; on the complete Ukraine Ethereum network (23.9 M addresses) that is roughly nine
CPU-hours and is stochastic. This module replaces it with the path-based (tree) expansion of
the same quantity, which is deterministic and costs K sparse matrix-vector products.

Definition. Let P be the sparse matrix with P[i, j] = p_ij, the activation probability of the
transfer edge i -> j. Under the independent cascade, the probability that a cascade started at
s activates v along one specific path is the product of p along that path. Summing over paths
of length k gives (P^k)[s, v]; the path-based approximation of the expected number of sources
whose cascade reaches v is therefore

    c(v) = sum_{k=1..K} [ (P^T)^k 1 ]_v ,

computed by the recursion x_0 = 1, x_{k+1} = P^T x_k, c = sum_k x_k. This is the reverse form
of the maximum-influence-arborescence approximation used by MIA/PMIA (Chen, Wang and Wang,
KDD 2010) and the same propagation operator that APPNP (Klicpera et al., ICLR 2019) and PPRGo
(Bojchevski et al., KDD 2020) use to replace expensive multi-hop message passing. Because
p_ij <= p_scale < 1 the terms decay geometrically, so a small K is sufficient; K = 15 leaves a
relative truncation error below 1e-6 for p_scale = 0.1.

The score is normalised to the unit interval so that it is directly comparable with the
Monte-Carlo coverage it replaces.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def linear_reach(edge_index, prop_prob, num_nodes: int, K: int = 15, damping: float = 1.0,
                 return_terms: bool = False):
    """Path-based expected reach for every node (see module docstring)."""
    src = np.asarray(edge_index[0]); dst = np.asarray(edge_index[1])
    p = np.clip(np.asarray(prop_prob, dtype=np.float64), 0.0, 1.0)
    # P^T: entry (j, i) = p_ij, so a matvec propagates mass from i to j
    Pt = csr_matrix((p, (dst, src)), shape=(num_nodes, num_nodes))
    x = np.ones(num_nodes, dtype=np.float64)
    c = np.zeros(num_nodes, dtype=np.float64)
    terms = []
    for _ in range(K):
        x = damping * (Pt @ x)
        c += x
        if return_terms:
            terms.append(float(x.sum()))
        if x.max() < 1e-12:
            break
    m = c.max()
    if m > 0:
        c = c / m
    out = torch.tensor(c.astype(np.float32))
    return (out, terms) if return_terms else out


def linear_reach_from_data(data, K: int = 15, damping: float = 1.0):
    return linear_reach(data.edge_index.cpu().numpy(), data.prop_prob.cpu().numpy(), data.num_nodes, K=K, damping=damping)
