#!/usr/bin/env python3
"""How much of the removal result is the crawl boundary, and what survives it.

Every network in this study except the complete ones is a two-hop neighbourhood grown
outward from a set of anchor addresses. Two things follow that a removal experiment on
such a graph cannot ignore.

First, the boundary decides the answer. In the two-hop graph the hop-2 addresses are
overwhelmingly degree-one leaves attached to hop-1 counterparties, so removing hop-1 hubs
disconnects almost everything and removing the anchors does not. In the induced subgraph
on hop 0 and 1, every address is present because it transacted with an anchor, so removing
the anchors disconnects almost everything and removing the highest-degree undesignated
addresses does not. Both statements are properties of the sampling, and they have opposite
signs. We compute both and report both.

Second, the natural control is wrong. Removing the highest-degree undesignated addresses
compares the anchors with the largest nodes of a graph that was grown from the anchors.
The comparison that holds the sampling fixed is a degree-matched one: undesignated
addresses drawn to have the same degree distribution as the designated set. That is the
comparison this script adds, with three independent draws.

Usage:
  python scripts/compute_boundary_sensitivity.py [--out boundary_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train_v2 import load_seed_addresses  # noqa: E402

NETWORKS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv", "IsraelAddrs.xlsx", "tron"),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv", "ofac_iran.csv", "tron"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "ofac_russia_ukraine.csv", "tron"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "ofac_terrorist_financing.csv", "tron"),
]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402


def build(csv: str, seeds: str, chain: str, value_cap: float = 1e8):
    df = pd.read_csv(find(csv), usecols=["from", "to", "value"])
    df = df[(df["value"] > 0) & (df["value"] <= value_cap)]
    if chain == "eth":
        df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
    nodes = sorted(set(df["from"]) | set(df["to"]))
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    s = df["from"].map(idx).to_numpy(np.int64); d = df["to"].map(idx).to_numpy(np.int64)
    pair = np.unique(s * n + d)
    anchors = np.zeros(n, bool)
    for a in load_seed_addresses(str(find(seeds)), chain):
        if a in idx:
            anchors[idx[a]] = True
    return n, (pair // n).astype(np.int64), (pair % n).astype(np.int64), anchors


def lcc_share(n, s, d, removed):
    keep = ~removed
    m = keep[s] & keep[d]
    A = csr_matrix((np.ones(int(m.sum()), np.int8), (s[m], d[m])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="weak")
    return np.bincount(lab[keep]).max() / keep.sum()


def degree_matched(deg, anchors, n_draws=3, window=60):
    """Undesignated addresses drawn to match the designated degree distribution."""
    und = np.where(~anchors)[0]
    order = np.argsort(deg[und])
    sorted_deg = deg[und][order]; sorted_und = und[order]
    target = deg[anchors]
    out = []
    for seed in range(n_draws):
        rng = np.random.default_rng(seed)
        chosen: set[int] = set()
        for g in target:
            lo = np.searchsorted(sorted_deg, g, "left"); hi = np.searchsorted(sorted_deg, g, "right")
            cand = [c for c in sorted_und[max(0, lo - window):hi + window] if c not in chosen]
            chosen.add(int(cand[rng.integers(len(cand))]) if cand else int(sorted_und[0]))
        out.append(np.array(sorted(chosen)))
    return out


def experiments(n, s, d, anchors, label):
    deg = np.bincount(s, minlength=n) + np.bincount(d, minlength=n)
    base = lcc_share(n, s, d, np.zeros(n, bool))
    na = int(anchors.sum())
    loss = lambda mask: float(100 * (1 - lcc_share(n, s, d, mask) / base))
    res = {"n_addresses": int(n), "n_pairs": int(len(s)), "n_designated": na,
           "baseline_lcc_share": float(base),
           "designated_median_degree": float(np.median(deg[anchors])),
           "network_median_degree": float(np.median(deg))}
    m = anchors.copy(); res["remove_designated_pct"] = loss(m)
    order = np.argsort(-deg, kind="stable")
    top = order[~anchors[order]][:na]
    m = np.zeros(n, bool); m[top] = True
    res["remove_top_degree_undesignated_pct"] = loss(m)
    dm = []
    for pick in degree_matched(deg, anchors):
        m = np.zeros(n, bool); m[pick] = True
        dm.append(loss(m))
    res["remove_degree_matched_pct"] = dm
    res["remove_degree_matched_pct_mean"] = float(np.mean(dm))
    rnd = []
    und = np.where(~anchors)[0]
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        m = np.zeros(n, bool); m[rng.choice(und, na, replace=False)] = True
        rnd.append(loss(m))
    res["remove_random_pct_mean"] = float(np.mean(rnd))
    print(f"[{label}] n={n:,} anchors={na}: designated {res['remove_designated_pct']:.2f}% | "
          f"degree-matched {res['remove_degree_matched_pct_mean']:.2f}% | "
          f"top-degree {res['remove_top_degree_undesignated_pct']:.2f}% | "
          f"random {res['remove_random_pct_mean']:.2f}%", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="boundary_sensitivity.json")
    args = ap.parse_args()
    out = {}
    for tag, csv, seeds, chain in NETWORKS:
        n, s, d, anchors = build(csv, seeds, chain)
        rec = {"two_hop": experiments(n, s, d, anchors, f"{tag} two-hop")}
        # induced subgraph on the anchors and their direct counterparties
        nb = np.unique(np.concatenate([d[anchors[s]], s[anchors[d]]]))
        sub = np.zeros(n, bool); sub[nb] = True; sub[anchors] = True
        ki = np.where(sub)[0]; remap = -np.ones(n, np.int64); remap[ki] = np.arange(len(ki))
        m = sub[s] & sub[d]
        rec["hop1"] = experiments(len(ki), remap[s[m]], remap[d[m]], anchors[ki], f"{tag} hop<=1")
        out[tag] = rec
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
