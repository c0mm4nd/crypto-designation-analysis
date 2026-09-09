#!/usr/bin/env python3
"""Removal tests weighted by value rather than by address count.

The component-share measure counts addresses, and on a payment network most addresses are
small, so it answers "how many accounts lose their only route" rather than "how much value
loses its route". This script repeats the removal tests with three measures that weight by
what moved:

  share of transferred value on edges whose endpoints both survive and remain in the
    largest component;
  share of transfers, the same thing unweighted by size;
  share of addresses, the measure used elsewhere, for comparison.

Run on the crawled sanctions networks, where per-edge values are available. The value-weighted
measure on the complete network is computed separately, from the value-carrying export that
scripts/export_full_tron_network.sh produces, by scripts/full_network_value_removal.py and
scripts/full_network_stranded.py. The crawled networks are where the metric objection bites
hardest in any case, since that is where degree-one addresses dominate.

Usage:
  python scripts/flow_weighted_removal.py [--out flow_weighted_removal.json]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
sys.path.insert(0, ROOT)
from model.train_v2 import load_seed_addresses  # noqa: E402

NETWORKS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv", "IsraelAddrs.xlsx"),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv", "ofac_iran.csv"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "ofac_russia_ukraine.csv"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "ofac_terrorist_financing.csv"),
]
VALUE_CAP = 1e8


def build(csv: str, seeds: str):
    df = pd.read_csv(find(csv), usecols=["from", "to", "value"])
    df = df[(df["value"] > 0) & (df["value"] <= VALUE_CAP)]
    nodes = sorted(set(df["from"]) | set(df["to"]))
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    g = df.groupby([df["from"].map(idx), df["to"].map(idx)])["value"].agg(["sum", "size"])
    pairs = np.array(list(g.index))
    anchors = np.zeros(n, bool)
    for a in load_seed_addresses(str(find(seeds)), "tron"):
        if a in idx:
            anchors[idx[a]] = True
    return n, pairs[:, 0], pairs[:, 1], g["sum"].to_numpy(), g["size"].to_numpy(), anchors


def measures(n, s, d, val, cnt, removed):
    keep = ~removed
    m = keep[s] & keep[d]
    A = csr_matrix((np.ones(int(m.sum()), np.int8), (s[m], d[m])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="weak")
    big = np.bincount(lab[keep]).argmax()
    inbig = (lab == big) & keep
    live = m & inbig[s] & inbig[d]
    return {"addresses": inbig.sum() / keep.sum(),
            "value": val[live].sum() / val.sum(),
            "transfers": cnt[live].sum() / cnt.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flow_weighted_removal.json")
    args = ap.parse_args()
    out = {}
    for tag, csv, seeds in NETWORKS:
        n, s, d, val, cnt, anchors = build(csv, seeds)
        na = int(anchors.sum())
        base = measures(n, s, d, val, cnt, np.zeros(n, bool))
        deg = np.bincount(s, minlength=n) + np.bincount(d, minlength=n)
        order = np.argsort(-deg, kind="stable")
        rec = {"n_addresses": int(n), "n_designated": na, "baseline": base, "removals": {}}
        sets = {"designated": anchors.copy()}
        m = np.zeros(n, bool); m[order[~anchors[order]][:na]] = True
        sets["top_degree_undesignated"] = m
        rng = np.random.default_rng(0)
        und = np.where(~anchors)[0]
        m = np.zeros(n, bool); m[rng.choice(und, na, replace=False)] = True
        sets["random_undesignated"] = m
        for name, mask in sets.items():
            after = measures(n, s, d, val, cnt, mask)
            rec["removals"][name] = {k: float(100 * (1 - after[k] / base[k])) for k in base}
        out[tag] = rec
        r = rec["removals"]
        print(f"[{tag:30}] n={n:>8,} designated {r['designated']['addresses']:6.2f}% addr "
              f"{r['designated']['value']:6.2f}% value | top-degree "
              f"{r['top_degree_undesignated']['addresses']:6.2f}% addr "
              f"{r['top_degree_undesignated']['value']:6.2f}% value", flush=True)
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
