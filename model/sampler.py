"""
Minimal CSR-based neighbour sampler for mini-batch training of the Attri-GAT encoder
on graphs too large for full-batch training (tens of millions of edges), without
pyg-lib / torch-sparse.

For a batch of target nodes it samples up to `fanout[l]` incoming neighbours per node
at each of L layers (messages flow along edge direction src -> dst, so a target's
representation depends on its in-neighbours), returns the induced edge list on the
sampled node set with target nodes placed first, and the corresponding edge
attributes. Full-neighbourhood inference is done with the same routine using a large
fanout, in chunks of target nodes.
"""

from __future__ import annotations

import numpy as np
import torch


class CSRGraph:
    def __init__(self, edge_index: np.ndarray, edge_attr: np.ndarray, num_nodes: int):
        src, dst = edge_index
        order = np.argsort(dst, kind="stable")
        self.src = src[order].astype(np.int64)
        self.eid = order.astype(np.int64)  # original edge id for attributes
        self.attr = edge_attr
        counts = np.bincount(dst, minlength=num_nodes)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n = num_nodes

    def sample(self, targets: np.ndarray, fanout: list[int], rng: np.random.Generator):
        """Return (n_id, edge_index_local, edge_attr, n_targets)."""
        n_id = list(targets)
        pos = {int(t): i for i, t in enumerate(targets)}
        frontier = targets
        src_l, dst_l, eid_l = [], [], []
        for f in fanout:
            new_frontier = []
            for u in frontier:
                s, e = self.indptr[u], self.indptr[u + 1]
                if e == s:
                    continue
                if e - s > f:
                    sel = rng.choice(e - s, size=f, replace=False) + s
                else:
                    sel = np.arange(s, e)
                for k in sel:
                    v = int(self.src[k])
                    if v not in pos:
                        pos[v] = len(n_id); n_id.append(v); new_frontier.append(v)
                    src_l.append(pos[v]); dst_l.append(pos[int(u)]); eid_l.append(self.eid[k])
            frontier = np.array(new_frontier, dtype=np.int64)
        if src_l:
            ei = torch.tensor(np.stack([src_l, dst_l]), dtype=torch.long)
            ea = torch.tensor(self.attr[np.array(eid_l)], dtype=torch.float)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long); ea = torch.zeros((0, self.attr.shape[1]), dtype=torch.float)
        return np.array(n_id, dtype=np.int64), ei, ea, len(targets)


def sample_batches(graph: CSRGraph, x: torch.Tensor, batch_size: int, fanout: list[int], rng: np.random.Generator, shuffle: bool = True, targets: np.ndarray | None = None):
    """Yield (batch_x, edge_index, edge_attr, n_id, n_targets) over all nodes (or `targets`)."""
    nodes = np.arange(graph.n) if targets is None else targets
    if shuffle:
        nodes = rng.permutation(nodes)
    for i in range(0, len(nodes), batch_size):
        t = nodes[i:i + batch_size]
        n_id, ei, ea, nt = graph.sample(t, fanout, rng)
        yield x[n_id], ei, ea, n_id, nt
