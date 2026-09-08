#!/usr/bin/env python3
"""What the connectivity loss on the complete network is actually made of.

The removal test reports the share of retained addresses that leave the largest weakly
connected component. On a transfer network where most addresses appear once, that share is
dominated by pendant vertices: an address whose only counterparty is removed leaves the
component whether or not anything was routed through it. A hub that mints one-time deposit
addresses and a hub that routes value between operators therefore score the same way, and
the metric on its own does not distinguish them.

This script says how much of each loss is that effect and what is left when it is removed:

  * the share of the addresses dropped from the component that had degree one, and the
    share whose entire neighbourhood was removed;
  * the same removal tests on the 2-core, obtained by iteratively deleting every address of
    degree one, which is the subgraph in which no address depends on a single counterparty;
  * the degree-matched control with many more draws than a full recomputation allows, by
    counting the addresses whose neighbourhood lies entirely inside the removed set. That
    count is exact for the pendant part of the loss and is what the draws differ in.

Usage:
  python scripts/full_network_kcore.py [--draws 200] [--out full_tron_kcore.json]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

CACHE = "/home/c0mm4nd/wcfrm/repo/tron_full2/graph_cache"
DESIGNATED = "/tmp/designated_hash.tsv"
OUT_DIR = "/home/c0mm4nd/wcfrm/repo"


def load():
    nodes = np.load(CACHE + "_nodes.npy")
    si = np.load(CACHE + "_si.npy")
    di = np.load(CACHE + "_di.npy")
    arr = np.array(sorted(int(l.split()[1]) for l in open(DESIGNATED)), dtype=np.uint64)
    pos = np.searchsorted(nodes, arr)
    pos = pos[pos < len(nodes)]
    ok = nodes[pos] == arr[: len(pos)]
    anchors = np.zeros(len(nodes), bool)
    anchors[pos[ok]] = True
    return len(nodes), si, di, anchors


def two_core(n, si, di, tol: int = 2000):
    """Iteratively strip degree-one addresses.

    Each pass recomputes degrees over all 745 million pairs, and the tail of the peeling
    removes a few dozen addresses per pass, so we stop once a pass removes fewer than `tol`
    of them. The residual is reported; it is a rounding error on 187 million addresses.
    """
    alive = np.ones(n, bool)
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    while True:
        drop = alive & (deg <= 1)
        k = int(drop.sum())
        if k == 0:
            break
        alive &= ~drop
        m = alive[si] & alive[di]
        deg = np.bincount(si[m], minlength=n) + np.bincount(di[m], minlength=n)
        deg[~alive] = 0
        print(f"  2-core pass: removed {k:,}, {alive.sum():,} remain", flush=True)
        if k < tol:
            print(f"  stopping: residual below {tol}", flush=True)
            break
    return alive


def lcc(n, si, di, removed):
    keep = ~removed
    m = keep[si] & keep[di]
    A = csr_matrix((np.ones(int(m.sum()), np.int8), (si[m], di[m])), shape=(n, n))
    _, lab = connected_components(A, directed=True, connection="weak")
    return np.bincount(lab[keep]).max() / keep.sum(), lab, keep


def isolated_by(n, si, di, removed):
    """Addresses not removed whose every neighbour was removed."""
    keep = ~removed
    m = keep[si] & keep[di]
    deg_after = np.bincount(si[m], minlength=n) + np.bincount(di[m], minlength=n)
    deg_before = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    return keep & (deg_after == 0) & (deg_before > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default="full_tron_kcore.json")
    args = ap.parse_args()

    n, si, di, anchors = load()
    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    na = int(anchors.sum())
    print(f"n={n:,} pairs={len(si):,} designated={na}", flush=True)
    print(f"degree-one share of all addresses: {100*(deg == 1).mean():.1f}%", flush=True)

    out = {"n_addresses": int(n), "n_pairs": int(len(si)), "n_designated": na,
           "degree_one_share": float((deg == 1).mean())}

    base, lab0, keep0 = lcc(n, si, di, np.zeros(n, bool))
    big0 = np.bincount(lab0).argmax()
    inlcc0 = lab0 == big0
    print(f"baseline LCC {100*base:.4f}%", flush=True)

    order = np.argsort(-deg, kind="stable")
    sets = {"designated": anchors.copy()}
    m = np.zeros(n, bool); m[order[~anchors[order]][:na]] = True
    sets[f"top_degree_{na}"] = m
    for b in (1000, 10000):
        m = np.zeros(n, bool); m[order[~anchors[order]][:b]] = True
        sets[f"top_degree_{b}"] = m

    detail = {}
    for name, mask in sets.items():
        share, lab, keep = lcc(n, si, di, mask)
        dropped = keep & inlcc0 & ~(lab == np.bincount(lab[keep]).argmax())
        iso = isolated_by(n, si, di, mask)
        detail[name] = {
            "connectivity_loss_pct": float(100 * (1 - share / base)),
            "n_dropped_from_lcc": int(dropped.sum()),
            "share_of_dropped_with_degree_one": float((deg[dropped] == 1).mean()) if dropped.any() else 0.0,
            "share_of_dropped_isolated_by_removal": float(iso[dropped].mean()) if dropped.any() else 0.0,
        }
        d = detail[name]
        print(f"[{name:18}] loss {d['connectivity_loss_pct']:8.4f}%  dropped {d['n_dropped_from_lcc']:>12,}  "
              f"degree-one {100*d['share_of_dropped_with_degree_one']:5.1f}%  "
              f"isolated {100*d['share_of_dropped_isolated_by_removal']:5.1f}%", flush=True)
    out["full_graph"] = detail
    # Write what we have before the expensive part, so a run interrupted in the 2-core
    # still leaves the decomposition on disk.
    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote the full-graph decomposition to {args.out}", flush=True)

    if args.draws == 0:
        print("skipping the 2-core (--draws 0)", flush=True)
        return

    print("building the 2-core...", flush=True)
    alive = two_core(n, si, di)
    ki = np.where(alive)[0]
    remap = -np.ones(n, np.int64); remap[ki] = np.arange(len(ki))
    mm = alive[si] & alive[di]
    s2, d2 = remap[si[mm]].astype(np.int32), remap[di[mm]].astype(np.int32)
    n2 = len(ki); a2 = anchors[ki]
    deg2 = np.bincount(s2, minlength=n2) + np.bincount(d2, minlength=n2)
    print(f"2-core: {n2:,} addresses, {len(s2):,} pairs, {int(a2.sum())} designated", flush=True)
    base2, _, _ = lcc(n2, s2, d2, np.zeros(n2, bool))
    core = {"n_addresses": int(n2), "n_pairs": int(len(s2)), "n_designated": int(a2.sum()),
            "baseline_lcc_share": float(base2)}
    na2 = int(a2.sum())
    m = a2.copy()
    core["remove_designated_pct"] = float(100 * (1 - lcc(n2, s2, d2, m)[0] / base2))
    o2 = np.argsort(-deg2, kind="stable")
    m = np.zeros(n2, bool); m[o2[~a2[o2]][:na2]] = True
    core["remove_top_degree_pct"] = float(100 * (1 - lcc(n2, s2, d2, m)[0] / base2))
    out["two_core"] = core
    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"  2-core removals written: designated {core['remove_designated_pct']:.4f}%  "
          f"top-degree {core['remove_top_degree_pct']:.4f}%", flush=True)
    # A full component recomputation on 187 million addresses costs minutes, so the many
    # draws needed for an interval use a proxy: the number of retained addresses whose whole
    # neighbourhood is inside the removed set. Those addresses leave the component by
    # construction, and on this graph they are 95-99% of everything that leaves it, so the
    # proxy tracks the exact loss closely. We compute both on the first few draws to say how
    # closely, and the proxy alone on the rest.
    def isolated_frac(mask):
        keep = ~mask
        m = keep[s2] & keep[d2]
        da = np.bincount(s2[m], minlength=n2) + np.bincount(d2[m], minlength=n2)
        return float((keep & (da == 0) & (deg2 > 0)).sum() / keep.sum())

    und = np.where(~a2)[0]; ud = deg2[und]; o = np.argsort(ud); sd = ud[o]; ou = und[o]
    def draw(seed):
        rr = np.random.default_rng(seed); chosen = set()
        for g in deg2[a2]:
            lo = int(np.searchsorted(sd, g, "left")); hi = int(np.searchsorted(sd, g, "right"))
            if hi <= lo:
                lo, hi = max(0, lo - 1), min(len(ou), lo + 1)
            for _ in range(40):
                c = int(ou[rr.integers(lo, hi)])
                if c not in chosen:
                    chosen.add(c); break
        m = np.zeros(n2, bool); m[list(chosen)] = True
        return m

    exact, proxy = [], []
    n_exact = 5
    for seed in range(args.draws):
        m = draw(seed)
        proxy.append(100 * isolated_frac(m))
        if seed < n_exact:
            exact.append(float(100 * (1 - lcc(n2, s2, d2, m)[0] / base2)))
        if (seed + 1) % 25 == 0:
            print(f"  draw {seed+1}/{args.draws}: proxy mean {np.mean(proxy):.5f}%", flush=True)
    core["proxy_vs_exact_first_draws"] = {"exact": exact, "proxy": proxy[:n_exact]}
    core["designated_proxy_pct"] = float(100 * isolated_frac(a2))
    core["degree_matched_proxy_pct"] = proxy
    dm = np.array(exact) if exact else np.array(proxy)
    core["remove_degree_matched_pct_mean"] = float(dm.mean())
    core["remove_degree_matched_pct_sd"] = float(dm.std(ddof=1))
    core["remove_degree_matched_pct_ci"] = [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))]
    core["n_draws"] = int(len(dm))
    core["designated_percentile_within_draws"] = float(100 * (dm < core["remove_designated_pct"]).mean())
    print(f"2-core: designated {core['remove_designated_pct']:.4f}%  "
          f"degree-matched {dm.mean():.4f}% [{core['remove_degree_matched_pct_ci'][0]:.4f}, "
          f"{core['remove_degree_matched_pct_ci'][1]:.4f}] over {len(dm)} draws  "
          f"top-degree {core['remove_top_degree_pct']:.4f}%", flush=True)
    out["two_core"] = core

    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
