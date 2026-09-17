#!/usr/bin/env python3
"""Does the structural comparison survive dropping the pairs that carry no value?

The complete TRON USDT network is built from transfer records with no value filter, so an
address that only ever received a dust or address-poisoning transfer is a node in it and a
zero-value pair is an edge. Throughput is unaffected by the choice, because a pair carrying no
value contributes nothing to it, but the component-based measures need not be: a zero-value
edge can be the link that attaches a value-carrying component to the rest, so removing such
edges could move either the address-count or the stranded-value result in either direction.

This script settles that by running the same removal test twice, once on all pairs and once on
the subgraph of pairs that carry a positive amount, and reporting the two side by side. Within
each graph the removal sets are chosen on that graph's own degrees, so each column is the test
as it would have been run had the graph been built that way from the start.

It also computes the throughput and stranded value of the random-removal control, which the
earlier runs reported only as an address count.

Usage:
  ROTOR_DATA=... ROTOR_OUT=... ROTOR_DESIGNATED_HASHES=... \
      python scripts/zero_value_sensitivity.py [--out zero_value_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATA = os.environ.get("ROTOR_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(DATA, "tron_full2/graph_cache")
DESIGNATED = os.environ.get("ROTOR_DESIGNATED_HASHES", os.path.join(DATA, "designated_hash.tsv"))
OUT_DIR = os.environ.get("ROTOR_OUT", DATA)


def load():
    nodes = np.load(CACHE + "_nodes.npy")
    n = len(nodes)
    si = np.load(CACHE + "_si.npy")
    di = np.load(CACHE + "_di.npy")
    val = np.load(CACHE + "_val.npy")
    arr = np.array(sorted(int(l.split()[1]) for l in open(DESIGNATED)), dtype=np.uint64)
    pos = np.searchsorted(nodes, arr)
    pos = pos[pos < n]
    ok = nodes[pos] == arr[: len(pos)]
    anchors = np.zeros(n, bool)
    anchors[pos[ok]] = True
    del nodes, arr
    return n, si, di, val, anchors


def analyse(n, si, di, val, anchors, label, out):
    """Every measure of the removal test on one graph."""
    total = float(val.sum(dtype=np.float64))
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    present = deg > 0
    n_present = int(present.sum())
    na = int((anchors & present).sum())
    print(f"[{label}] pairs={len(si):,} addresses_with_an_edge={n_present:,} "
          f"designated_present={na} value={total/1e12:.3f} trillion", flush=True)

    def components(removed):
        """Labels of the weak components of the graph with `removed` taken out."""
        keep = ~removed
        surv = keep[si] & keep[di]
        A = csr_matrix((np.ones(int(surv.sum()), np.int8), (si[surv], di[surv])), shape=(n, n))
        _, lab = connected_components(A, directed=True, connection="weak")
        del A
        return keep, surv, lab

    def measure(removed):
        keep, surv, lab = components(removed)
        # The share is taken over addresses that have an edge in this graph, so that addresses
        # which exist only through a zero-value pair do not count as lost when those pairs go.
        pool = keep & present
        big = np.bincount(lab[pool]).argmax()
        inbig = (lab == big) & pool
        share = inbig.sum() / pool.sum()
        incident = ~surv
        stranded = surv & ~(inbig[si] & inbig[di])
        # `where=` keeps these sums from materialising a filtered copy of a 745-million-element array.
        vi = float(val.sum(where=incident, dtype=np.float64))
        vs = float(val.sum(where=stranded, dtype=np.float64))
        r = {"lcc_share": float(share),
             "incident_pct": 100 * vi / total, "incident_usdt": vi,
             "stranded_pct": 100 * vs / total, "stranded_usdt": vs}
        del lab, keep, surv, inbig, incident, stranded
        return r

    base = measure(np.zeros(n, bool))
    print(f"[{label}] baseline largest component {100*base['lcc_share']:.4f}% of addresses", flush=True)

    order = np.argsort(-deg, kind="stable")
    und = np.where(~anchors & present)[0]
    ud = deg[und]
    o = np.argsort(ud)
    sd, ou = ud[o], und[o]

    sets = {}
    sets["designated"] = (anchors & present).copy()
    m = np.zeros(n, bool)
    m[order[(~anchors & present)[order]][:na]] = True
    sets["top_degree_undesignated"] = m   # as many as there are designated addresses present
    rr = np.random.default_rng(0)
    chosen = set()
    for g in deg[anchors & present]:
        lo = int(np.searchsorted(sd, g, "left")); hi = int(np.searchsorted(sd, g, "right"))
        if hi <= lo:
            lo, hi = max(0, lo - 1), min(len(ou), lo + 1)
        for _ in range(40):
            c = int(ou[rr.integers(lo, hi)])
            if c not in chosen:
                chosen.add(c); break
    m = np.zeros(n, bool); m[list(chosen)] = True
    sets["degree_matched_undesignated"] = m
    rr = np.random.default_rng(100)
    m = np.zeros(n, bool); m[rr.choice(und, na, replace=False)] = True
    sets["random_undesignated"] = m

    res = {"n_pairs": int(len(si)), "n_addresses_with_an_edge": n_present,
           "n_designated_present": na, "total_value_usdt": total,
           "baseline_lcc_share": base["lcc_share"], "removals": {}}
    for name, mask in sets.items():
        r = measure(mask)
        r["connectivity_loss_pct"] = float(100 * (1 - r["lcc_share"] / base["lcc_share"]))
        res["removals"][name] = r
        print(f"[{label}] {name:32} addresses_lost {r['connectivity_loss_pct']:9.4f}%  "
              f"throughput {r['incident_pct']:8.4f}%  stranded {r['stranded_usdt']:,.0f} USDT", flush=True)
    out[label] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="zero_value_sensitivity.json")
    args = ap.parse_args()
    n, si, di, val, anchors = load()
    out = {"description": ("The address-level removal test on the complete TRON USDT network run twice: "
                           "on all pairs, and on the subgraph of pairs carrying a positive amount. "
                           "Removal sets are chosen on each graph's own degrees. The share of addresses "
                           "lost from the largest component is taken over addresses that have an edge in "
                           "the graph in question.")}
    analyse(n, si, di, val, anchors, "all_pairs", out)
    keep = val > 0
    print(f"dropping {int((~keep).sum()):,} zero-value pairs of {len(keep):,}", flush=True)
    si2, di2, val2 = si[keep], di[keep], val[keep]
    del si, di, val, keep
    analyse(n, si2, di2, val2, anchors, "value_carrying_pairs", out)
    a, b = out["all_pairs"]["removals"], out["value_carrying_pairs"]["removals"]
    out["comparison"] = {k: {"connectivity_loss_pct": [a[k]["connectivity_loss_pct"], b[k]["connectivity_loss_pct"]],
                             "stranded_usdt": [a[k]["stranded_usdt"], b[k]["stranded_usdt"]]}
                         for k in a if k in b}
    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
