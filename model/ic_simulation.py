"""
Coverage IC — full-graph implementation.

Key design choices:
1. ALL N nodes are used as sources (R=1 each), giving complete coverage.
2. Pre-allocated boolean array (reset in-place after each run) avoids
   O(N) allocation per cascade — O(cascade_size) instead.
3. Numpy vectorised edge trials within each BFS frontier step.

score(v) = (# source cascades that infected v) / N
         = proxy for "how central v is to receiving cascades from the full graph"

Seeds (recipients with many in-edges) are reached by many cascades → high scores.
Isolated leaf senders are rarely reached → low scores.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data

_ADJ = None  # shared (fork) CSR adjacency for worker processes


def _run_chunk(args):
    """One chunk of source cascades. Accumulates only the nodes actually reached, so the
    per-worker memory is proportional to the cascade sizes rather than to the graph size
    (which matters at tens of millions of nodes)."""
    sources, N, seed = args
    rng = np.random.default_rng(seed)
    adj = _ADJ
    infected = np.zeros(N, dtype=bool)
    visited_parts = []
    for src in sources:
        infected[src] = True
        frontier = np.array([src], dtype=np.int32)
        newly = [src]
        while frontier.size > 0:
            parts = []
            for u in frontier:
                s, e = adj.indptr[u], adj.indptr[u + 1]
                if s == e:
                    continue
                nbrs = adj.indices[s:e]
                probs = adj.data[s:e]
                mask = ~infected[nbrs]
                cands = nbrs[mask]
                if cands.size == 0:
                    continue
                won = cands[rng.random(cands.size) < probs[mask]]
                if won.size > 0:
                    infected[won] = True
                    newly.extend(won.tolist())
                    parts.append(won)
            frontier = np.concatenate(parts) if parts else np.empty(0, np.int32)
        visited_parts.append(np.asarray(newly, dtype=np.int64))
        for v in newly:
            infected[v] = False
    if not visited_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    allv = np.concatenate(visited_parts)
    nz, cnt = np.unique(allv, return_counts=True)
    return nz, cnt.astype(np.float32)


def simulate_ic_parallel(data: Data, seed: int = 42, n_workers: int | None = None, chunks_per_worker: int = 8, max_sources: int | None = None) -> torch.Tensor:
    """Coverage IC from every node, parallelised over source chunks with fork-shared adjacency."""
    global _ADJ
    import multiprocessing as mp
    N = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    pp = np.clip(data.prop_prob.cpu().numpy(), 1e-6, 1.0)
    _ADJ = csr_matrix((pp, (ei[0], ei[1])), shape=(N, N))
    n_workers = n_workers or max(1, min(64, os.cpu_count() or 1))
    n_workers = max(1, min(n_workers, 32))  # the pool is memory-bound, not CPU-bound, at this scale
    sources = np.arange(N, dtype=np.int32)
    if max_sources is not None and max_sources < N:
        sources = np.random.default_rng(seed).choice(N, size=max_sources, replace=False).astype(np.int32)
    chunks = np.array_split(sources, max(n_workers * chunks_per_worker, 1))
    jobs = [(c, N, seed + i) for i, c in enumerate(chunks)]
    print(f"  Coverage-IC (parallel): {N:,} sources, {n_workers} workers, {len(jobs)} chunks", flush=True)
    ctx = mp.get_context("fork")
    hit_count = np.zeros(N, dtype=np.float64)
    with ctx.Pool(n_workers) as pool:
        for nz, vals in pool.imap_unordered(_run_chunk, jobs):
            hit_count[nz] += vals
    scores = (hit_count / len(sources)).astype(np.float32)
    if scores.max() > 1e-12:
        scores /= scores.max()
    pct = np.percentile(scores, [50, 90, 99])
    print(f"  IC scores: nonzero={int((scores > 0).sum()):,}/{N:,} p50={pct[0]:.4f} p90={pct[1]:.4f} p99={pct[2]:.4f}", flush=True)
    return torch.tensor(scores, dtype=torch.float)


def simulate_ic(
    data: Data,
    R: int = 1,              # runs per source; default 1 for full-N coverage
    max_nodes: int | None = None,   # None = use ALL nodes as sources
    seed: int = 42,
    direction: str = "coverage",
) -> torch.Tensor:
    """
    Full-coverage IC: run from every node (or max_nodes sampled nodes),
    R times each. score(v) = fraction of cascades that infected v.

    Parameters
    ----------
    data       : PyG Data with edge_index and prop_prob
    R          : Monte Carlo repetitions per source node (default 1)
    max_nodes  : None → use ALL nodes; int → random subsample of sources
    seed       : random seed
    direction  : kept for API compat; always runs forward IC

    Returns
    -------
    importance : FloatTensor shape (N,) in [0, 1]
    """
    N  = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    pp = np.clip(data.prop_prob.cpu().numpy(), 1e-6, 1.0)

    src_e, dst_e = ei[0], ei[1]
    adj = csr_matrix((pp, (src_e, dst_e)), shape=(N, N))

    rng = np.random.default_rng(seed)

    if max_nodes is None or max_nodes >= N:
        sources = np.arange(N, dtype=np.int32)
    else:
        sources = rng.choice(N, size=max_nodes, replace=False).astype(np.int32)

    n_src = len(sources)
    print(f"  Coverage-IC: {n_src:,} sources × R={R} (full-graph mode) ...", flush=True)

    # ── Pre-allocated infected array (reset per-run with O(cascade) cost) ──
    infected = np.zeros(N, dtype=bool)
    hit_count = np.zeros(N, dtype=np.float64)

    for i, src in enumerate(sources):
        for _ in range(R):
            # --- IC run ---
            infected[src] = True
            frontier = np.array([src], dtype=np.int32)
            newly    = [src]

            while frontier.size > 0:
                new_parts = []
                for u in frontier:
                    s, e = adj.indptr[u], adj.indptr[u + 1]
                    if s == e:
                        continue
                    nbrs  = adj.indices[s:e]
                    probs = adj.data[s:e]
                    mask  = ~infected[nbrs]
                    cands = nbrs[mask]
                    if cands.size == 0:
                        continue
                    won = cands[rng.random(cands.size) < probs[mask]]
                    if won.size > 0:
                        infected[won] = True
                        newly.extend(won.tolist())
                        new_parts.append(won)
                frontier = np.concatenate(new_parts) if new_parts else np.empty(0, np.int32)

            # Record hits
            hit_count[newly] += 1.0

            # Reset infected array (O(cascade_size), not O(N))
            for v in newly:
                infected[v] = False

        if (i + 1) % 10000 == 0 or (i + 1) == n_src:
            pct = (i + 1) / n_src * 100
            covered = int((hit_count > 0).sum())
            print(f"  Coverage-IC: {i+1:,}/{n_src:,} ({pct:.0f}%)  "
                  f"nodes covered: {covered:,}/{N:,}", flush=True)

    scores = (hit_count / (n_src * R)).astype(np.float32)

    s_max = scores.max()
    if s_max > 1e-12:
        scores /= s_max

    pct_arr = np.percentile(scores, [50, 90, 99])
    nonzero  = int((scores > 0).sum())
    print(f"  IC scores: nonzero={nonzero:,}/{N:,}  "
          f"p50={pct_arr[0]:.4f}  p90={pct_arr[1]:.4f}  "
          f"p99={pct_arr[2]:.4f}  max={scores.max():.4f}", flush=True)

    return torch.tensor(scores, dtype=torch.float)
