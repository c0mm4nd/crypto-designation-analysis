#!/usr/bin/env python3
"""Partition-free and partition-robustness tests of the 'periphery versus backbone' claim.

For every network:
  A. Remove all anchor addresses; remove the same number of highest-degree undesignated
     addresses; remove the same number at random (three seeds). Report connectivity loss.
     Also report where anchors sit in the degree ranking.
  B. Role partition from k-means on the 12 raw structural features (same K as WCFRM):
     anchor-densest role versus most damaging role, with budget-matched random control,
     to test whether the role-level divergence depends on the partition.
Outputs backbone_tests.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph  # noqa: E402
from model.train_v2 import load_seed_addresses  # noqa: E402
from scripts.analyze_v2_run import conn_metrics  # noqa: E402

NETWORKS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv", "IsraelAddrs.xlsx", "tron", 9),
    ("ukraine_tron", "ukraine_tron_usdt_edges_2hop.csv", "seeds_ukraine_tron.txt", "tron", 9),
    ("ukraine_eth", "AGG:ukraine_eth_full_edges_agg.csv|ukraine_eth_full_nodes_agg.csv", "seeds_ukraine_eth.txt", "eth", 9),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv", "ofac_iran.csv", "tron", 9),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "ofac_russia_ukraine.csv", "tron", 9),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "ofac_terrorist_financing.csv", "tron", 9),
]


def loss(c):
    return round(100 * (1 - c["connectivity"]), 4)


def main():
    out = {}
    for tag, csv, seeds, chain, K in NETWORKS:
        if csv.startswith("AGG:"):
            e_csv, n_csv = csv[4:].split("|")
            if not os.path.exists(e_csv):
                print(f"[{tag}] aggregate tables missing, skipped"); continue
            from model.features import build_graph_from_agg
            data, n2i, i2n, _ = build_graph_from_agg(e_csv, n_csv, ts_unit="s", seed_set=load_seed_addresses(seeds, chain),
                                                     lower=(chain == "eth"), return_addresses=False)
        else:
            df = pd.read_csv(csv, usecols=["from", "to", "value", "block_timestamp"])
            df = df[(df["value"] > 0) & (df["value"] <= 1e8)]
            if chain == "eth":
                df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
            data, n2i, i2n, _ = build_graph(df)
        n = data.num_nodes
        ei = data.edge_index.numpy()
        pair = np.unique(ei[0].astype(np.int64) * n + ei[1])
        src, dst = pair // n, pair % n
        deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        if n2i is None:
            anchors = data.y.numpy().astype(bool)   # set by build_graph_from_agg
        else:
            anchors = np.zeros(n, dtype=bool)
            for a in load_seed_addresses(seeds, chain):
                if a in n2i:
                    anchors[n2i[a]] = True
        k = int(anchors.sum())
        order = np.argsort(-deg, kind="stable")
        rank = np.empty(n, dtype=int); rank[order] = np.arange(n)
        res = {"n_nodes": int(n), "n_edges": int(len(src)), "n_anchors": k}
        if k > 0:
            top_und = np.zeros(n, dtype=bool); top_und[[i for i in order if not anchors[i]][:k]] = True
            rnd = []
            for s in range(3):
                rng = np.random.default_rng(s); m = np.zeros(n, dtype=bool); m[rng.choice(n, k, replace=False)] = True
                rnd.append(loss(conn_metrics(n, src, dst, m)))
            res["A_partition_free"] = {
                "remove_anchors_loss_pct": loss(conn_metrics(n, src, dst, anchors)),
                "remove_top_degree_undesignated_loss_pct": loss(conn_metrics(n, src, dst, top_und)),
                "remove_random_loss_pct_mean": float(np.mean(rnd)), "remove_random_loss_pct_range": [min(rnd), max(rnd)],
                "anchor_median_degree": float(np.median(deg[anchors])), "anchor_median_degree_rank": int(np.median(rank[anchors])),
                "anchor_share_in_top_1pct_degree": float((rank[anchors] < 0.01 * n).mean()),
                "anchor_share_in_top_k_degree": float((rank[anchors] < k).mean()),
            }
            for m_ in (1000, 5000):
                if m_ < n:
                    mask = np.zeros(n, dtype=bool); mask[[i for i in order if not anchors[i]][:m_]] = True
                    res["A_partition_free"][f"remove_top_{m_}_degree_undesignated_loss_pct"] = loss(conn_metrics(n, src, dst, mask))
        X = StandardScaler().fit_transform(data.x.numpy())
        part = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
        rows = []
        rng = np.random.default_rng(0)
        for r in range(K):
            m = part == r; size = int(m.sum())
            rm = np.zeros(n, dtype=bool); rm[rng.choice(n, size, replace=False)] = True
            rows.append({"role": r, "size": size, "share_pct": round(100 * size / n, 3), "anchors": int(anchors[m].sum()), "density_pct": round(100 * anchors[m].sum() / size, 4),
                         "mean_in": float(np.bincount(dst, minlength=n)[m].mean()), "mean_out": float(np.bincount(src, minlength=n)[m].mean()),
                         "loss_pct": loss(conn_metrics(n, src, dst, m)), "random_loss_pct": loss(conn_metrics(n, src, dst, rm))})
        dens = max(rows, key=lambda r: r["density_pct"]); worst = max(rows, key=lambda r: r["loss_pct"])
        res["B_raw_feature_partition"] = {"K": K, "roles": rows, "anchor_densest_role": dens["role"], "anchor_densest_loss_pct": dens["loss_pct"], "anchor_densest_density_pct": dens["density_pct"],
                                         "most_damaging_role": worst["role"], "most_damaging_loss_pct": worst["loss_pct"], "same_role": dens["role"] == worst["role"]}
        out[tag] = res
        print(f"[{tag}] n={n:,} anchors={k}: remove anchors {res.get('A_partition_free', {}).get('remove_anchors_loss_pct')}% | top-degree same count {res.get('A_partition_free', {}).get('remove_top_degree_undesignated_loss_pct')}% | raw-feature partition: densest R{dens['role']} ({dens['density_pct']:.3f}%, loss {dens['loss_pct']:.1f}%) vs worst R{worst['role']} (loss {worst['loss_pct']:.1f}%)", flush=True)
    json.dump(out, open("backbone_tests.json", "w"), indent=1)


if __name__ == "__main__":
    main()
