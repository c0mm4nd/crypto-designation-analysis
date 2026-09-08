#!/usr/bin/env python3
"""Budget-matched controls for role-targeted dismantling.

For every learned role in a network, the role-removal experiment deletes |R| addresses.
This script deletes the same number of addresses chosen (i) uniformly at random
(three seeds) and (ii) by highest total degree, and reports the same connectivity
metrics, so that role removal can be compared against budget-matched baselines.

Outputs dismantling_controls.json with, per dataset and per role budget:
  role connectivity / components (from the existing analysis JSON),
  random-removal mean connectivity / components (+ min/max over seeds),
  top-degree-removal connectivity / components.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv", "wcfrm_israel_tron_analysis.json"),
    ("ukraine_tron", "ukraine_tron_usdt_edges_2hop.csv", "wcfrm_ukraine_tron_analysis.json"),
    ("ukraine_eth", "AGG:ukraine_eth_full_edges_agg.csv|ukraine_eth_full_nodes_agg.csv", "wcfrm_ukraine_eth_analysis.json"),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv", "wcfrm_ofac_iran_tron_analysis.json"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "wcfrm_ofac_russia_ukraine_tron_analysis.json"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "wcfrm_ofac_terrorist_financing_tron_analysis.json"),
]
SEEDS = [0, 1, 2]


def load_graph(csv_name: str):
    if csv_name.startswith("AGG:"):
        import sys
        sys.path.insert(0, str(ROOT))
        from model.features import load_agg_graph_arrays
        e_csv, n_csv = csv_name[4:].split("|")
        _, n, src, dst, _, _ = load_agg_graph_arrays(str(ROOT / e_csv), str(ROOT / n_csv))
        deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        return n, src, dst, deg
    df = pd.read_csv(ROOT / csv_name, usecols=["from", "to", "value"])
    df = df[(df["value"] > 0) & (df["value"] <= 1e8)]  # same filter as model training (overflow artefacts removed)
    if csv_name.startswith("ukraine_eth"):
        df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
    nodes = pd.unique(pd.concat([df["from"], df["to"]]))
    idx = {n: i for i, n in enumerate(nodes)}
    src = df["from"].map(idx).to_numpy(np.int64)
    dst = df["to"].map(idx).to_numpy(np.int64)
    n = len(nodes)
    # aggregate to unique directed pairs
    pair = np.unique(src * n + dst)
    src, dst = pair // n, pair % n
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
    return n, src, dst, deg


def metrics_after_removal(n, src, dst, remove_mask):
    keep = ~remove_mask
    em = keep[src] & keep[dst]
    n_keep = int(keep.sum())
    if n_keep == 0:
        return {"connectivity": 0.0, "n_components": 0}
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[keep] = np.arange(n_keep)
    s, d = new_id[src[em]], new_id[dst[em]]
    adj = coo_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(n_keep, n_keep))
    n_cc, labels = connected_components(adj, directed=False)
    largest = int(np.bincount(labels).max())
    return {"connectivity": round(largest / n_keep, 6), "n_components": int(n_cc)}


def main():
    out = {}
    for tag, csv_name, analysis_name in DATASETS:
        probe = csv_name[4:].split("|")[0] if csv_name.startswith("AGG:") else csv_name
        if not (ROOT / probe).exists():
            print(f"[{tag}] edge file missing, skipped")
            continue
        analysis = json.load(open(ROOT / analysis_name))
        n, src, dst, deg = load_graph(csv_name)
        print(f"[{tag}] n={n:,} edges={len(src):,} (analysis JSON n={analysis['n_nodes']:,})", flush=True)
        base = metrics_after_removal(n, src, dst, np.zeros(n, dtype=bool))
        order_by_degree = np.argsort(-deg, kind="stable")
        rows = {}
        for key, stat in sorted(analysis["role_stats"].items(), key=lambda kv: int(kv[0])):
            budget = int(stat["size"])
            role = analysis["dismantling"][f"role_{key}"]
            rnd = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                mask = np.zeros(n, dtype=bool)
                mask[rng.choice(n, size=min(budget, n), replace=False)] = True
                rnd.append(metrics_after_removal(n, src, dst, mask))
            mask = np.zeros(n, dtype=bool)
            mask[order_by_degree[:budget]] = True
            topdeg = metrics_after_removal(n, src, dst, mask)
            rows[f"role_{key}"] = {
                "budget": budget,
                "budget_share_pct": round(100 * budget / n, 4),
                "role_connectivity": role["connectivity"],
                "role_n_components": role["n_components"],
                "random_connectivity_mean": round(float(np.mean([r["connectivity"] for r in rnd])), 6),
                "random_connectivity_min": min(r["connectivity"] for r in rnd),
                "random_connectivity_max": max(r["connectivity"] for r in rnd),
                "random_n_components_mean": round(float(np.mean([r["n_components"] for r in rnd])), 1),
                "topdegree_connectivity": topdeg["connectivity"],
                "topdegree_n_components": topdeg["n_components"],
            }
            print(f"  R{key}: budget={budget:,}  role={role['connectivity']:.3f}  random={rows[f'role_{key}']['random_connectivity_mean']:.3f}  topdeg={topdeg['connectivity']:.3f}", flush=True)
        out[tag] = {"n_nodes": n, "n_edges": int(len(src)), "baseline": base, "roles": rows}
    with open(ROOT / "dismantling_controls.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved dismantling_controls.json")


if __name__ == "__main__":
    main()
