#!/usr/bin/env python3
"""Address-level backbone test on the complete TRON USDT transfer network.

The networks used elsewhere in this study are two-hop neighbourhoods crawled outward from
the designated addresses. Any statement about structural position measured in such a graph
is conditioned on where the crawl stopped, and we show in the manuscript that the sign of
the result reverses between the one-hop and two-hop boundaries. This script removes the
boundary: it builds the entire USDT transfer graph on TRON from the archival node, with no
seeding and no sampling, and repeats the removal test there.
"""
import glob, json, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DESIGNATED = "/tmp/designated_hash.tsv"

BUCKETS = sorted(glob.glob("/home/c0mm4nd/wcfrm/repo/tron_full2/bucket_*.bin"))
DT = np.dtype([("f", "<u8"), ("t", "<u8"), ("c", "<u4")])


def node_universe():
    """Sorted array of every address hash in the network.

    The buckets are partitioned on cityHash64 of the sender, so the senders of one bucket
    cannot appear as senders of another: their per-bucket uniques concatenate to a globally
    unique set with no cross-bucket merge. Only the recipients need a global deduplication,
    which is one sort of the recipient column.
    """
    senders = []
    total = sum(os.path.getsize(f) // DT.itemsize for f in BUCKETS)
    recips = np.empty(total, np.uint64)
    at = 0
    for i, f in enumerate(BUCKETS):
        a = np.fromfile(f, dtype=DT)
        senders.append(np.unique(a["f"]))
        recips[at:at + len(a)] = a["t"]; at += len(a)
        del a
        if (i + 1) % 8 == 0:
            print(f"  read {i+1}/{len(BUCKETS)} buckets", flush=True)
    s_all = np.sort(np.concatenate(senders)); del senders
    print(f"  senders: {len(s_all):,}", flush=True)
    r_all = np.unique(recips); del recips
    print(f"  recipients: {len(r_all):,}", flush=True)
    u = np.union1d(s_all, r_all)
    del s_all, r_all
    return u


def edge_arrays(nodes):
    """Endpoint indices into `nodes`, as int32, read one bucket at a time."""
    total = sum(os.path.getsize(f) // DT.itemsize for f in BUCKETS)
    si = np.empty(total, np.int32); di = np.empty(total, np.int32)
    at = 0
    for f in BUCKETS:
        a = np.fromfile(f, dtype=DT)
        k = len(a)
        si[at:at + k] = np.searchsorted(nodes, a["f"])
        di[at:at + k] = np.searchsorted(nodes, a["t"])
        at += k
        del a
    return si, di


CACHE = "/home/c0mm4nd/wcfrm/repo/tron_full2/graph_cache"


def main():
    if os.path.exists(CACHE + "_nodes.npy"):
        nodes = np.load(CACHE + "_nodes.npy", mmap_mode=None)
        si = np.load(CACHE + "_si.npy"); di = np.load(CACHE + "_di.npy")
        n = len(nodes)
        print(f"addresses: {n:,} (from cache)", flush=True)
    else:
        nodes = node_universe()
        n = len(nodes)
        print(f"addresses: {n:,}", flush=True)
        si, di = edge_arrays(nodes)
        np.save(CACHE + "_nodes.npy", nodes); np.save(CACHE + "_si.npy", si); np.save(CACHE + "_di.npy", di)
    print(f"edges (unique directed pairs): {len(si):,}", flush=True)

    hashes = {}
    for line in open(DESIGNATED):
        h, v = line.split()
        hashes[int(v)] = h
    pos = np.searchsorted(nodes, np.array(sorted(hashes), dtype=np.uint64))
    pos = pos[(pos < n)]
    keep = nodes[pos] == np.array(sorted(hashes), dtype=np.uint64)[: len(pos)]
    anchor_idx = pos[keep]
    anchors = np.zeros(n, bool); anchors[anchor_idx] = True
    print(f"designated addresses present: {anchors.sum()} of {len(hashes)}", flush=True)

    deg = np.bincount(si, minlength=n) + np.bincount(di, minlength=n)
    print(f"degree: designated median {np.median(deg[anchors]):.0f}, network median {np.median(deg):.0f}", flush=True)
    rank = np.empty(n, np.int64); rank[np.argsort(-deg, kind="stable")] = np.arange(n)
    print(f"designated median degree rank {np.median(rank[anchors]):,.0f} of {n:,}; "
          f"share in top 1% by degree {100*(rank[anchors] < n//100).mean():.1f}%", flush=True)

    def lcc(removed):
        """Share of retained addresses in the largest weakly connected component."""
        m = ~(removed[si] | removed[di])
        a = si[m]; b = di[m]
        A = csr_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
        del a, b, m
        _, lab = connected_components(A, directed=True, connection="weak")
        del A
        keep = ~removed
        return np.bincount(lab[keep]).max() / keep.sum()

    print("computing baseline component structure...", flush=True)
    base = lcc(np.zeros(n, bool))
    print(f"baseline largest weakly connected component: {100*base:.3f}% of addresses", flush=True)
    out = {"n_addresses": int(n), "n_pairs": int(len(si)), "n_designated": int(anchors.sum()),
           "baseline_lcc_share": float(base)}

    def loss(mask, label):
        v = 100 * (1 - lcc(mask) / base)
        print(f"  {label:44} {v:8.4f}%", flush=True)
        return float(v)

    m = anchors.copy()
    out["remove_designated_pct"] = loss(m, "remove all designated")
    na = int(anchors.sum())
    order = np.argsort(-deg, kind="stable")
    top = order[~anchors[order]][:na]
    m = np.zeros(n, bool); m[top] = True
    out["remove_top_degree_undesignated_pct"] = loss(m, f"remove {na} top-degree undesignated")
    und = np.where(~anchors)[0]; ud = deg[und]; o = np.argsort(ud); sd = ud[o]; ou = und[o]
    ad = deg[anchors]; dm = []
    for seed in range(3):
        rr = np.random.default_rng(seed); chosen = set()
        for g in ad:
            # draw uniformly from the addresses of exactly this degree, by index, so that
            # a degree shared by a hundred million addresses costs no more than a rare one
            lo = int(np.searchsorted(sd, g, "left")); hi = int(np.searchsorted(sd, g, "right"))
            if hi <= lo:
                lo, hi = max(0, lo - 1), min(len(ou), lo + 1)
            pick = None
            for _ in range(40):
                c = int(ou[rr.integers(lo, hi)])
                if c not in chosen:
                    pick = c; break
            if pick is None:
                w = 1
                while pick is None and w < len(ou):
                    for c in ou[max(0, lo - w):min(len(ou), hi + w)]:
                        if int(c) not in chosen:
                            pick = int(c); break
                    w *= 8
            chosen.add(pick)
        m = np.zeros(n, bool); m[list(chosen)] = True
        dm.append(loss(m, f"remove {na} degree-matched undesignated (seed {seed})"))
    out["remove_degree_matched_pct_mean"] = float(np.mean(dm)); out["remove_degree_matched_pct"] = dm
    rnd = []
    for seed in range(3):
        rr = np.random.default_rng(100 + seed)
        m = np.zeros(n, bool); m[rr.choice(und, na, replace=False)] = True
        rnd.append(loss(m, f"remove {na} random undesignated (seed {seed})"))
    out["remove_random_pct_mean"] = float(np.mean(rnd))
    out["designated_median_degree"] = float(np.median(deg[anchors]))
    out["network_median_degree"] = float(np.median(deg))
    out["designated_share_top1pct_degree"] = float((rank[anchors] < n // 100).mean())
    json.dump(out, open("/home/c0mm4nd/wcfrm/repo/full_tron_backbone.json", "w"), indent=1)
    print("saved full_tron_backbone.json", flush=True)

if __name__ == "__main__":
    main()
