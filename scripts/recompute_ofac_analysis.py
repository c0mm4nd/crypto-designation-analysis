#!/usr/bin/env python3
"""Recompute per-role statistics and role-removal metrics for the OFAC TRON networks.

The original compute_ofac_analysis.py indexed nodes from the unfiltered edge CSV,
whereas the cluster/score arrays were produced by model.features.build_graph after
dropping zero-value transfers. The two orderings differ, so cluster labels were
attached to the wrong addresses. This script rebuilds the node index exactly as
build_graph does (value > 0 filter, first-appearance order over from/to) and
asserts that the array lengths match before computing anything.

Connectivity is defined as |largest connected component| / |retained nodes|,
the same definition used for the primary benchmark and the Ukraine networks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = {
    "ofac_iran_tron": ("ofac_iran_tron_usdt_edges_2hop.csv", "wcfrm_ofac_iran_tron_results_clusters.npy", "wcfrm_ofac_iran_tron_results_scores.npy", "ofac_iran.csv"),
    "ofac_russia_ukraine_tron": ("ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "wcfrm_ofac_russia_ukraine_tron_results_clusters.npy", "wcfrm_ofac_russia_ukraine_tron_results_scores.npy", "ofac_russia_ukraine.csv"),
    "ofac_terrorist_financing_tron": ("ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "wcfrm_ofac_terrorist_financing_tron_results_clusters.npy", "wcfrm_ofac_terrorist_financing_tron_results_scores.npy", "ofac_terrorist_financing.csv"),
}


def build_index(csv_name: str):
    df = pd.read_csv(ROOT / csv_name, usecols=["from", "to", "value"])
    df = df[df["value"] > 0]
    nodes = pd.unique(pd.concat([df["from"], df["to"]]))
    idx = {n: i for i, n in enumerate(nodes)}
    src = df["from"].map(idx).to_numpy(np.int64)
    dst = df["to"].map(idx).to_numpy(np.int64)
    n = len(nodes)
    pair = np.unique(src * n + dst)
    # degrees count individual transfers (multigraph), as in the primary-benchmark analysis;
    # connectivity uses unique directed pairs
    return nodes, n, pair // n, pair % n, src, dst


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
    for tag, (csv_name, clusters_name, scores_name, seeds_name) in REGISTRY.items():
        nodes, n, src, dst, raw_src, raw_dst = build_index(csv_name)
        clusters = np.load(ROOT / clusters_name)
        scores = np.load(ROOT / scores_name)
        assert len(clusters) == n == len(scores), f"{tag}: clusters={len(clusters)} scores={len(scores)} nodes={n}"
        sdf = pd.read_csv(ROOT / seeds_name)
        seeds = set(sdf[sdf["address"].str.startswith("T", na=False)]["address"])
        is_seed = np.array([a in seeds for a in nodes])
        in_deg = np.bincount(raw_dst, minlength=n)
        out_deg = np.bincount(raw_src, minlength=n)
        K = int(clusters.max()) + 1
        role_stats, dismantling = {}, {"baseline": conn_metrics(n, src, dst, np.zeros(n, dtype=bool))}
        for k in range(K):
            m = clusters == k
            size = int(m.sum())
            seed_count = int(is_seed[m].sum())
            role_stats[str(k)] = {
                "size": size,
                "avg_in": round(float(in_deg[m].mean()), 3) if size else 0.0,
                "avg_out": round(float(out_deg[m].mean()), 3) if size else 0.0,
                "avg_ic": round(float(scores[m].mean()), 4) if size else 0.0,
                "seed_count": seed_count,
                "seed_pct": round(100 * seed_count / size, 4) if size else 0.0,
                "mean_score": round(float(scores[m].mean()), 4) if size else 0.0,
            }
            dismantling[f"role_{k}"] = conn_metrics(n, src, dst, m)
        seed_cluster = max(range(K), key=lambda k: (role_stats[str(k)]["seed_count"], role_stats[str(k)]["seed_pct"]))
        out = {
            "dataset": tag,
            "n_nodes": n,
            "n_edges": int(len(src)),
            "n_transfers": int(len(raw_src)),
            "n_roles": K,
            "n_seeds_in_graph": int(is_seed.sum()),
            "seed_cluster": seed_cluster,
            "role_stats": role_stats,
            "dismantling": dismantling,
            "note": "Recomputed with node index aligned to model.features.build_graph (value>0 filter); edges are unique directed pairs.",
        }
        with open(ROOT / f"wcfrm_{tag}_analysis.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{tag}] n={n:,} edges={len(src):,} seeds_in_graph={is_seed.sum()} K={K} seed_cluster=R{seed_cluster}")
        for k in range(K):
            r, d = role_stats[str(k)], dismantling[f"role_{k}"]
            print(f"   R{k}: size={r['size']:>6,} seeds={r['seed_count']:>2} dens={r['seed_pct']:.3f}% in={r['avg_in']:.2f} out={r['avg_out']:.2f} conn_after={d['connectivity']:.3f} cc={d['n_components']}")


if __name__ == "__main__":
    main()
