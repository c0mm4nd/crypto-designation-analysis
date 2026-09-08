#!/usr/bin/env python3
"""Removal tests on the complete TRON USDT network, weighted by value as well as by count.

The component-share measure counts addresses. On a payment network most addresses are
small, so it answers how many accounts lose their only route rather than how much value
does. This script repeats the complete-network removal tests carrying the aggregated USDT
value of every directed pair, and reports three measures side by side:

    addresses   share of retained addresses in the largest weakly connected component
    value       share of transferred USDT on pairs whose endpoints both survive and stay
                in that component
    pairs       the same, unweighted

Reads the value-carrying export produced by scripts/export_full_tron_network.sh with the
value column (tron_full_val/bucket_*.bin, 24-byte records: two UInt64 hashes and a Float64).

Usage:
  python scripts/full_network_value_removal.py [--out full_tron_value_removal.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BUCKETS = sorted(glob.glob("/home/c0mm4nd/wcfrm/repo/tron_full_val/bucket_*.bin"))
DT = np.dtype([("f", "<u8"), ("t", "<u8"), ("v", "<f8")])
CACHE = "/home/c0mm4nd/wcfrm/repo/tron_full2/graph_cache"
DESIGNATED = "/tmp/designated_hash.tsv"
OUT_DIR = "/home/c0mm4nd/wcfrm/repo"


def load():
    """Map the value-carrying pairs onto the address indices already built for the count graph."""
    nodes = np.load(CACHE + "_nodes.npy")
    n = len(nodes)
    total = sum(os.path.getsize(f) // DT.itemsize for f in BUCKETS)
    si = np.empty(total, np.int32); di = np.empty(total, np.int32); val = np.empty(total, np.float64)
    at = 0
    for i, f in enumerate(BUCKETS):
        a = np.fromfile(f, dtype=DT)
        k = len(a)
        si[at:at + k] = np.searchsorted(nodes, a["f"])
        di[at:at + k] = np.searchsorted(nodes, a["t"])
        val[at:at + k] = a["v"]
        at += k
        del a
        if (i + 1) % 8 == 0:
            print(f"  read {i+1}/{len(BUCKETS)} buckets", flush=True)
    arr = np.array(sorted(int(l.split()[1]) for l in open(DESIGNATED)), dtype=np.uint64)
    pos = np.searchsorted(nodes, arr); pos = pos[pos < n]
    ok = nodes[pos] == arr[: len(pos)]
    anchors = np.zeros(n, bool); anchors[pos[ok]] = True
    return n, si, di, val, anchors


def measures(n, si, di, val, removed):
    keep = ~removed
    m = keep[si] & keep[di]
    A = csr_matrix((np.ones(int(m.sum()), np.int8), (si[m], di[m])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="weak")
    big = np.bincount(lab[keep]).argmax()
    inbig = (lab == big) & keep
    live = m & inbig[si] & inbig[di]
    return {"addresses": float(inbig.sum() / keep.sum()),
            "value": float(val[live].sum() / val.sum()),
            "pairs": float(live.sum() / len(val))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="full_tron_value_removal.json")
    args = ap.parse_args()
    n, si, di, val, anchors = load()
    na = int(anchors.sum())
    print(f"n={n:,} pairs={len(si):,} designated={na} total value={val.sum()/1e12:.3f} trillion USDT", flush=True)
    base = measures(n, si, di, val, np.zeros(n, bool))
    print(f"baseline: addresses {100*base['addresses']:.4f}%  value {100*base['value']:.4f}%", flush=True)
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    order = np.argsort(-deg, kind="stable")
    out = {"n_addresses": int(n), "n_pairs": int(len(si)), "n_designated": na,
           "total_value_usdt": float(val.sum()), "baseline": base, "removals": {}}

    sets = {"designated": anchors.copy()}
    for b, label in ((na, f"top_degree_{na}"), (1000, "top_degree_1000"), (10000, "top_degree_10000")):
        m = np.zeros(n, bool); m[order[~anchors[order]][:b]] = True
        sets[label] = m
    und = np.where(~anchors)[0]; ud = deg[und]
    o = np.argsort(ud); sd = ud[o]; ou = und[o]
    rr = np.random.default_rng(0); chosen = set()
    for g in deg[anchors]:
        lo = int(np.searchsorted(sd, g, "left")); hi = int(np.searchsorted(sd, g, "right"))
        if hi <= lo:
            lo, hi = max(0, lo - 1), min(len(ou), lo + 1)
        for _ in range(40):
            c = int(ou[rr.integers(lo, hi)])
            if c not in chosen:
                chosen.add(c); break
    m = np.zeros(n, bool); m[list(chosen)] = True
    sets["degree_matched_undesignated"] = m

    for name, mask in sets.items():
        a = measures(n, si, di, val, mask)
        out["removals"][name] = {k: float(100 * (1 - a[k] / base[k])) for k in base}
        r = out["removals"][name]
        print(f"[{name:28}] addresses {r['addresses']:8.4f}%  value {r['value']:8.4f}%  pairs {r['pairs']:8.4f}%",
              flush=True)
    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
