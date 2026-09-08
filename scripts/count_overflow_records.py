#!/usr/bin/env python3
"""Count the integer-overflow artefacts removed from each crawled network.

The block-explorer crawl occasionally returns a transfer whose value field has overflowed,
giving amounts far larger than the entire supply of the token. These records are dropped from
every volume analysis at a threshold of 1e8 USDT. The Methods quote the count, so it has to be
reproducible; this script produces it per network rather than as a single unattributed number.

The full-node exports are unaffected: the contract event logs carry the value as it was
emitted, and the designated-address history contains no record above the threshold.

Usage:
  python scripts/count_overflow_records.py [--out overflow_records.json]
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = 1e8

# The six networks of the study. Ukraine Ethereum is assembled from a full-node export
# aggregated to directed pairs rather than from a crawl, so it has no crawl edge list.
NETWORKS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv"),
    ("ukraine_tron", "ukraine_tron_usdt_edges_2hop.csv"),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv"),
]
FULL_NODE = [
    ("designated_addresses", os.path.join("ch_data", "designated_usdt_transfers_complete_490.csv"), "value"),
    ("ukraine_eth_anchor", os.path.join("ch_data", "ukraine_eth_anchor_usdt.csv"), None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="overflow_records.json")
    args = ap.parse_args()

    out = {"value_cap_usdt": CAP, "crawled_networks": {}, "full_node_exports": {}}
    total = 0
    for name, fname in NETWORKS:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} not present")
            continue
        v = pd.read_csv(path, usecols=["value"])["value"]
        n = int((v > CAP).sum())
        total += n
        out["crawled_networks"][name] = {"records": int(len(v)), "over_cap": n,
                                         "zero_value": int((v == 0).sum())}
        print(f"[{name:32}] {len(v):>10,} records, {n:>6,} over cap")
    out["total_over_cap_crawled"] = total
    print(f"total over cap across the crawled networks: {total:,}")

    for name, fname, col in FULL_NODE:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            continue
        cols = pd.read_csv(path, nrows=0).columns
        c = col if col in cols else next((x for x in cols if x.lower() == "value"), None)
        if c is None:
            continue
        v = pd.read_csv(path, usecols=[c])[c]
        n = int((v > CAP).sum())
        out["full_node_exports"][name] = {"records": int(len(v)), "over_cap": n}
        print(f"[{name:32}] {len(v):>10,} records, {n:>6,} over cap (full-node export)")

    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
