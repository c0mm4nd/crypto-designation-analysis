#!/usr/bin/env python3
"""Split the value lost by a removal into what the removed set itself moved and what is stranded.

The value-weighted removal measure counts every USDT on a pair that no longer has both
endpoints inside the largest component. Two very different things contribute to it. Value
on a pair incident to the removed set disappears because one of its endpoints is gone: that
is the throughput of the removed addresses themselves, and says only how big they were.
Value on a pair between two survivors that nevertheless fall out of the component is
stranded by the removal: that is the part that says something about the removed set holding
the rest of the network together.

This script reports both, for every removal set, so that the measure can be read for what it
is. It also reports a reachability-style statistic that excludes incident value entirely:
the share of surviving value whose two endpoints end up in different weakly connected
components.

Usage:
  python scripts/full_network_stranded.py [--out full_tron_stranded.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Paths are configurable so the scripts run outside the machine they were written on.
DATA = os.environ.get("ROTOR_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUCKETS = sorted(glob.glob(os.path.join(DATA, "tron_full_val", "bucket_*.bin")))
DT = np.dtype([("f", "<u8"), ("t", "<u8"), ("v", "<f8")])
CACHE = os.path.join(DATA, "tron_full2/graph_cache")
DESIGNATED = os.environ.get("ROTOR_DESIGNATED_HASHES", os.path.join(DATA, "designated_hash.tsv"))
OUT_DIR = os.environ.get("ROTOR_OUT", DATA)


def load():
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


def decompose(n, si, di, val, removed, total):
    """Value incident to the removed set, and value stranded between surviving addresses."""
    keep = ~removed
    incident = ~(keep[si] & keep[di])
    surv = ~incident
    A = csr_matrix((np.ones(int(surv.sum()), np.int8), (si[surv], di[surv])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="weak")
    big = np.bincount(lab[keep]).argmax()
    inbig = (lab == big) & keep
    stranded = surv & ~(inbig[si] & inbig[di])
    return {"incident_pct": float(100 * val[incident].sum() / total),
            "stranded_pct": float(100 * val[stranded].sum() / total),
            "incident_usdt": float(val[incident].sum()),
            "stranded_usdt": float(val[stranded].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="full_tron_stranded.json")
    args = ap.parse_args()
    n, si, di, val, anchors = load()
    total = float(val.sum())
    na = int(anchors.sum())
    print(f"n={n:,} pairs={len(si):,} designated={na} total={total/1e12:.3f} trillion USDT", flush=True)
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    order = np.argsort(-deg, kind="stable")
    sets = {"designated": anchors.copy()}
    for b, label in ((na, f"top_degree_{na}"), (1000, "top_degree_1000"), (10000, "top_degree_10000")):
        m = np.zeros(n, bool); m[order[~anchors[order]][:b]] = True
        sets[label] = m
    out = {"n_addresses": int(n), "n_pairs": int(len(si)), "n_designated": na,
           "total_value_usdt": total, "decomposition": {}}
    for name, mask in sets.items():
        d = decompose(n, si, di, val, mask, total)
        out["decomposition"][name] = d
        print(f"[{name:20}] incident {d['incident_pct']:8.4f}%  stranded {d['stranded_pct']:10.6f}%  "
              f"({d['stranded_usdt']/1e6:.1f} M USDT stranded)", flush=True)
    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
