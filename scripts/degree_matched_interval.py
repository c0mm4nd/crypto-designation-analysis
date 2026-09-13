#!/usr/bin/env python3
"""An interval for the degree-matched control on the complete network.

The control draws undesignated addresses with the same degree distribution as the
designated set. Three draws are not enough to say that two removals cost the same, so this
computes many. A full component recomputation on 745 million pairs takes minutes, so the
draws use the quantity that the removals actually differ in: the addresses whose entire
neighbourhood falls inside the removed set, which leave the component by construction and
are, on this graph, 95 to 99.6 per cent of everything that leaves it. The exact component
loss is computed for the first few draws so that the two can be compared, and the value
measures are computed exactly for every draw, since they need no component structure.

Reports, for the designated set and for the draws: the share of addresses isolated by the
removal, the throughput of the removed set, and the value stranded between survivors.

The exact component recomputations and the incidence index are the two memory peaks, and
they need not coexist: `--phase draws` builds the index and runs every draw without the
exact block, `--phase exact` computes the exact block for the designated set and the first
draws from edge masks alone and merges it into the draws output. The default runs both in
one process. The draws are seeded, so the two phases see identical removed sets.

Usage:
  python scripts/degree_matched_interval.py [--draws 200] [--exact 3] [--phase all|draws|exact]
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



def _cached_value_graph():
    """Load the value graph from the cache when the streaming rebuild has written it.

    scripts/rerun_complete_network_local.py stores the deduplicated network as index arrays
    plus a value array, so the 24-byte bucket files need not exist; this returns the same
    (n, si, di, val, anchors) tuple load() builds from them, or None if the cache is absent.
    """
    vpath = CACHE + "_val.npy"
    if not os.path.exists(vpath):
        return None
    nodes = np.load(CACHE + "_nodes.npy"); n = len(nodes)
    si = np.load(CACHE + "_si.npy"); di = np.load(CACHE + "_di.npy")
    val = np.load(vpath).astype(np.float64)
    arr = np.array(sorted(int(l.split()[1]) for l in open(DESIGNATED)), dtype=np.uint64)
    pos = np.searchsorted(nodes, arr); pos = pos[pos < n]
    ok = nodes[pos] == arr[: len(pos)]
    anchors = np.zeros(n, bool); anchors[pos[ok]] = True
    return n, si, di, val, anchors

def load():
    cached = _cached_value_graph()
    if cached is not None:
        return cached
    nodes = np.load(CACHE + "_nodes.npy")
    n = len(nodes)
    total = sum(os.path.getsize(f) // DT.itemsize for f in BUCKETS)
    si = np.empty(total, np.int32); di = np.empty(total, np.int32); val = np.empty(total, np.float64)
    at = 0
    for f in BUCKETS:
        a = np.fromfile(f, dtype=DT); k = len(a)
        si[at:at + k] = np.searchsorted(nodes, a["f"])
        di[at:at + k] = np.searchsorted(nodes, a["t"])
        val[at:at + k] = a["v"]; at += k
        del a
    arr = np.array(sorted(int(l.split()[1]) for l in open(DESIGNATED)), dtype=np.uint64)
    pos = np.searchsorted(nodes, arr); pos = pos[pos < n]
    ok = nodes[pos] == arr[: len(pos)]
    anchors = np.zeros(n, bool); anchors[pos[ok]] = True
    return n, si, di, val, anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--exact", type=int, default=3)
    ap.add_argument("--out", default="degree_matched_interval.json")
    ap.add_argument("--phase", choices=["all", "draws", "exact"], default="all")
    args = ap.parse_args()

    n, si, di, val, anchors = load()
    total = float(val.sum()); na = int(anchors.sum())
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    print(f"n={n:,} pairs={len(si):,} designated={na}", flush=True)

    # An index from address to incident edges, built once. With it a draw touches only the
    # edges incident to the four hundred removed addresses instead of all 745 million, which
    # is what makes a two-hundred-draw interval affordable. The edge identifier is carried as
    # the CSR data, offset by one so that edge zero is not confused with a structural zero;
    # the pairs are unique by construction, so nothing is summed.
    A_out = A_in = None
    if args.phase != "exact":
        print("building the incidence index...", flush=True)
        eid = (np.arange(len(si), dtype=np.int32) + 1)
        A_out = csr_matrix((eid, (si, di)), shape=(n, n))
        A_in = csr_matrix((eid, (di, si)), shape=(n, n))
        del eid
        print(f"  index built over {A_out.nnz + A_in.nnz:,} endpoints", flush=True)

    def incident(rem):
        """Edge ids and surviving neighbours of a removed set."""
        eids, nbrs = [], []
        for A in (A_out, A_in):
            for v in rem:
                a, b = A.indptr[v], A.indptr[v + 1]
                if b > a:
                    eids.append(A.data[a:b]); nbrs.append(A.indices[a:b])
        e = np.unique(np.concatenate(eids) - 1) if eids else np.zeros(0, np.int64)
        nb = np.concatenate(nbrs) if nbrs else np.zeros(0, np.int32)
        return e, nb

    def stats(mask, exact=False):
        keep = ~mask
        if A_out is not None:
            rem = np.where(mask)[0]
            einc, nb = incident(rem)
            cand = nb[keep[nb]]
            if len(cand):
                u, c = np.unique(cand, return_counts=True)
                iso_n = int(((c == deg[u]) & (deg[u] > 0)).sum())
            else:
                iso_n = 0
            out = {"isolated_share_pct": float(100 * iso_n / keep.sum()),
                   "throughput_pct": float(100 * val[einc].sum() / total)}
        else:
            # Without the index, the same two quantities from full edge masks: an address is
            # isolated by the removal when none of its edges survives it.
            surv = keep[si] & keep[di]
            da = np.bincount(si[surv], minlength=n) + np.bincount(di[surv], minlength=n)
            iso_n = int((keep & (da == 0) & (deg > 0)).sum())
            del da
            out = {"isolated_share_pct": float(100 * iso_n / keep.sum()),
                   "throughput_pct": float(100 * val[~surv].sum() / total)}
        if exact:
            surv = keep[si] & keep[di]
            A = csr_matrix((np.ones(int(surv.sum()), np.int8), (si[surv], di[surv])), shape=(n, n))
            _, lab = connected_components(A, directed=True, connection="weak")
            big = np.bincount(lab[keep]).argmax()
            inbig = (lab == big) & keep
            out["connectivity_loss_pct"] = float(100 * (1 - (inbig.sum() / keep.sum()) / 0.9999869268965178))
            stranded = surv & ~(inbig[si] & inbig[di])
            out["stranded_usdt"] = float(val[stranded].sum())
        return out

    do_exact = args.phase != "draws"
    n_draws = args.draws if args.phase != "exact" else min(args.exact, args.draws)
    res = {"n_designated": na, "designated": stats(anchors, exact=do_exact), "draws": []}
    print(f"designated: {res['designated']}", flush=True)

    und = np.where(~anchors)[0]; ud = deg[und]
    o = np.argsort(ud); sd = ud[o]; ou = und[o]
    target = deg[anchors]
    for seed in range(n_draws):
        rr = np.random.default_rng(seed); chosen = set()
        for g in target:
            lo = int(np.searchsorted(sd, g, "left")); hi = int(np.searchsorted(sd, g, "right"))
            if hi <= lo:
                lo, hi = max(0, lo - 1), min(len(ou), lo + 1)
            for _ in range(40):
                c = int(ou[rr.integers(lo, hi)])
                if c not in chosen:
                    chosen.add(c); break
        m = np.zeros(n, bool); m[list(chosen)] = True
        res["draws"].append(stats(m, exact=(do_exact and seed < args.exact)))
        if (seed + 1) % 20 == 0:
            iso = np.array([d["isolated_share_pct"] for d in res["draws"]])
            print(f"  draw {seed+1}/{args.draws}: isolated mean {iso.mean():.6f}%", flush=True)

    for key in ("isolated_share_pct", "throughput_pct"):
        v = np.array([d[key] for d in res["draws"]])
        res[key + "_summary"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                 "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                 "designated": res["designated"][key],
                                 "designated_percentile": float(100 * (v < res["designated"][key]).mean())}
        s = res[key + "_summary"]
        print(f"{key}: designated {s['designated']:.6f}, draws {s['mean']:.6f} "
              f"[{s['ci'][0]:.6f}, {s['ci'][1]:.6f}], designated at the {s['designated_percentile']:.0f}th percentile",
              flush=True)
    out_file = os.path.join(OUT_DIR, args.out)
    if args.phase == "exact" and os.path.exists(out_file):
        # Merge the exact block into the draws output, which holds the full draw list.
        full = json.load(open(out_file))
        assert full["n_designated"] == na and len(full["draws"]) >= len(res["draws"])
        for k in ("connectivity_loss_pct", "stranded_usdt"):
            full["designated"][k] = res["designated"][k]
            for i, d in enumerate(res["draws"]):
                full["draws"][i][k] = d[k]
        for i, d in enumerate(res["draws"]):
            for k in ("isolated_share_pct", "throughput_pct"):
                assert abs(full["draws"][i][k] - d[k]) < 1e-9, (i, k, full["draws"][i][k], d[k])
        res = full
        print("merged the exact block into the draws output", flush=True)
    ex = [d for d in res["draws"] if "stranded_usdt" in d]
    if ex:
        res["stranded_usdt_exact_draws"] = [d["stranded_usdt"] for d in ex]
        res["connectivity_loss_exact_draws"] = [d["connectivity_loss_pct"] for d in ex]
        print(f"exact draws: stranded {[f'{d/1e6:.1f}M' for d in res['stranded_usdt_exact_draws']]}, "
              f"connectivity {[f'{c:.4f}%' for c in res['connectivity_loss_exact_draws']]}", flush=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
