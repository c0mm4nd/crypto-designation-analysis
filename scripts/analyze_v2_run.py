#!/usr/bin/env python3
"""Convert a WCFRM v2 run into the wcfrm_<tag>_analysis.json format used by the figures.

Per role: size, mean in/out degree (transfer counts), mean IC coverage, anchor count and
density, and removal metrics (largest-connected-component share over retained addresses,
number of components) computed on unique directed address pairs. Node order is taken from
the run's _nodes.txt and checked against the rebuilt graph.

Usage:
  python scripts/analyze_v2_run.py --csv <edges.csv> --run v2_runs/v2_<tag>_seed42 \
      --seeds <anchors file> --chain tron --dataset <name> --out wcfrm_<name>_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train_v2 import load_seed_addresses  # noqa: E402


def conn_metrics(n, src, dst, remove_mask):
    keep = ~remove_mask
    em = keep[src] & keep[dst]
    n_keep = int(keep.sum())
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[keep] = np.arange(n_keep)
    s, d = new_id[src[em]], new_id[dst[em]]
    adj = coo_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(n_keep, n_keep))
    n_cc, labels = connected_components(adj, directed=False)
    largest = int(np.bincount(labels).max())
    return {"connectivity": round(largest / n_keep, 6), "n_components": int(n_cc), "removed": int(n - n_keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--agg-edges", default=None)
    ap.add_argument("--agg-nodes", default=None)
    ap.add_argument("--run", required=True, help="run prefix, e.g. v2_runs/v2_israel_tron_seed42")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--chain", default="tron")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--value-cap", type=float, default=1e8)
    args = ap.parse_args()

    nodes = [l.rstrip("\n") for l in open(args.run + "_nodes.txt")]
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    if args.agg_edges:
        from model.features import load_agg_graph_arrays
        agg_nodes, n2, src_a, dst_a, in_cnt_a, out_cnt_a = load_agg_graph_arrays(args.agg_edges, args.agg_nodes)
        assert agg_nodes == nodes, "node order mismatch between run and aggregate tables"
        src, dst = src_a, dst_a
        in_deg, out_deg = in_cnt_a, out_cnt_a
        raw_src = src  # transfer counts come from the node table
    else:
        df = pd.read_csv(args.csv, usecols=["from", "to", "value", "block_timestamp"])
        df = df[(df["value"] > 0) & (df["value"] <= args.value_cap)]
        if args.chain == "eth":
            df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
        assert set(df["from"]).union(df["to"]) <= set(idx), "graph nodes not in run node list"
        raw_src = df["from"].map(idx).to_numpy(np.int64)
        raw_dst = df["to"].map(idx).to_numpy(np.int64)
        pair = np.unique(raw_src * n + raw_dst)
        src, dst = pair // n, pair % n
        in_deg = np.bincount(raw_dst, minlength=n)
        out_deg = np.bincount(raw_src, minlength=n)
    clusters = np.load(args.run + "_clusters.npy").astype(int)
    scores = np.load(args.run + "_scores_ic.npy")
    scores_mlp = np.load(args.run + "_scores_mlp.npy") if os.path.exists(args.run + "_scores_mlp.npy") else None
    assert len(clusters) == n == len(scores)
    anchors = np.zeros(n, dtype=bool)
    if args.seeds:
        for a in load_seed_addresses(args.seeds, args.chain):
            if a in idx:
                anchors[idx[a]] = True
    K = int(clusters.max()) + 1
    role_stats, dism = {}, {"baseline": conn_metrics(n, src, dst, np.zeros(n, dtype=bool))}
    for k in range(K):
        m = clusters == k
        size = int(m.sum())
        sc = int(anchors[m].sum())
        role_stats[str(k)] = {"size": size, "avg_in": round(float(in_deg[m].mean()), 3) if size else 0.0, "avg_out": round(float(out_deg[m].mean()), 3) if size else 0.0,
                              "avg_ic": round(float(scores[m].mean()), 4) if size else 0.0, "seed_count": sc, "seed_pct": round(100 * sc / size, 4) if size else 0.0,
                              "mean_score": round(float(scores_mlp[m].mean()), 4) if (scores_mlp is not None and size) else None}
        dism[f"role_{k}"] = conn_metrics(n, src, dst, m)
    seed_cluster = max(range(K), key=lambda k: (role_stats[str(k)]["seed_count"], role_stats[str(k)]["seed_pct"]))
    out = {"dataset": args.dataset, "n_nodes": n, "n_edges": int(len(src)), "n_transfers": int(in_deg.sum()), "n_roles": K, "n_seeds_in_graph": int(anchors.sum()),
           "seed_cluster": seed_cluster, "role_stats": role_stats, "dismantling": dism, "model": "ROTOR", "run": args.run}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[{args.dataset}] n={n:,} edges={len(src):,} K={K} anchors={anchors.sum()} seed_cluster=R{seed_cluster}")
    for k in range(K):
        r, d = role_stats[str(k)], dism[f"role_{k}"]
        print(f"  R{k}: size={r['size']:>7,} anchors={r['seed_count']:>3} dens={r['seed_pct']:.3f}% in={r['avg_in']:.2f} out={r['avg_out']:.2f} conn_after={d['connectivity']:.3f}")


if __name__ == "__main__":
    main()
