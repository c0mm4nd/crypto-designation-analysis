#!/usr/bin/env python3
"""Extended screening evaluation on the primary NBCTF (Israel) TRON USDT benchmark.

Compares the IC-coverage importance score used by WCFRM with simple centralities
(in-degree, out-degree, total degree, PageRank) under three candidate sets:
  all      : all 600k addresses (the evaluation used in the manuscript)
  hop1     : sanctioned seeds versus their direct counterparties only
  hop2     : sanctioned seeds versus hop-2 addresses only
and reports ROC-AUC with 200-resample bootstrap CIs, paired bootstrap differences
versus the IC score, recall@K, flagged-set sizes at each score threshold, and the
degree distribution by crawl hop. Output: screening_extended.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = [500, 1000, 2000, 5000, 10000]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_BOOT = 200


def israel_seeds(path: Path) -> list[str]:
    ws = openpyxl.load_workbook(path).active
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) < 5 or not row[3]:
            continue
        addr, cur = str(row[3]).strip(), str(row[4] or "").upper()
        if cur in ("USDT", "TRX") and addr.startswith("T"):
            out.append(addr)
    return list(dict.fromkeys(out))


def pagerank(n, src, dst, alpha=0.85, iters=100):
    out_deg = np.bincount(src, minlength=n).astype(float)
    w = np.where(out_deg[src] > 0, 1.0 / out_deg[src], 0.0)
    M = csr_matrix((w, (dst, src)), shape=(n, n))
    r = np.full(n, 1.0 / n)
    dangling = out_deg == 0
    for _ in range(iters):
        r_new = alpha * (M @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < 1e-10:
            r = r_new
            break
        r = r_new
    return r


def boot_auc(scores, labels, rng):
    n = len(labels)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0 or labels[idx].sum() == n:
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot_diff(a, b, labels, rng):
    n = len(labels)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        diffs.append(roc_auc_score(labels[idx], a[idx]) - roc_auc_score(labels[idx], b[idx]))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def recall_at_k(scores, labels, k):
    top = np.argpartition(scores, -k)[-k:]
    return float(labels[top].sum() / labels.sum())


def main():
    df = pd.read_csv(ROOT / "israel_tron_usdt_edges_2hop.csv", usecols=["from", "to", "value"])
    df = df[(df["value"] > 0) & (df["value"] <= 1e8)]
    nodes = pd.unique(pd.concat([df["from"], df["to"]]))
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    src = df["from"].map(idx).to_numpy(np.int64)
    dst = df["to"].map(idx).to_numpy(np.int64)
    pair = np.unique(src * n + dst)
    src, dst = pair // n, pair % n
    run = ROOT / "v6_runs" / "v2_israel_tron_seed42"
    if not (run.parent / (run.name + "_scores_ic.npy")).exists():
        run = ROOT / "v2_runs" / "v2_israel_tron_seed42"
    if (run.parent / (run.name + "_scores_ic.npy")).exists():
        run_nodes = [l.rstrip("\n") for l in open(str(run) + "_nodes.txt")]
        assert run_nodes == list(nodes), "node order mismatch with v2 run"
        scores = np.load(str(run) + "_scores_ic.npy").astype(float)
        scores_mlp = np.load(str(run) + "_scores_mlp.npy").astype(float)
    else:
        scores = np.load(ROOT / "wcfrm_results_scores.npy").astype(float)
        scores_mlp = None
    assert len(scores) == n

    seeds = israel_seeds(ROOT / "IsraelAddrs.xlsx")
    labels = np.array([a in set(seeds) for a in nodes], dtype=int)
    print(f"nodes={n:,} edges={len(src):,} seeds_listed={len(seeds)} seeds_in_graph={labels.sum()}")

    in_deg = np.bincount(dst, minlength=n).astype(float)
    out_deg = np.bincount(src, minlength=n).astype(float)
    tot_deg = in_deg + out_deg
    pr = pagerank(n, src, dst)

    # crawl hop of each node: seeds = 0, neighbours of seeds = 1, rest = 2
    adj = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n)).tocsr()
    und = adj + adj.T
    seed_idx = np.where(labels == 1)[0]
    hop = np.full(n, 2, dtype=int)
    hop[seed_idx] = 0
    nb = np.unique(und[seed_idx].indices)
    hop[nb[hop[nb] == 2]] = 1

    rng = np.random.default_rng(42)
    methods = {"diffusion reach (ROTOR)": scores, "in-degree": in_deg, "out-degree": out_deg, "total degree": tot_deg, "PageRank": pr}
    if scores_mlp is not None:
        methods["learned importance (ROTOR)"] = scores_mlp
    out = {"n_nodes": n, "n_edges": int(len(src)), "n_seeds_listed": len(seeds), "n_seeds_in_graph": int(labels.sum()),
           "hop_counts": {int(h): int((hop == h).sum()) for h in (0, 1, 2)},
           "degree_by_hop": {int(h): {"mean_total_degree": float(tot_deg[hop == h].mean()), "median_total_degree": float(np.median(tot_deg[hop == h]))} for h in (0, 1, 2)},
           "candidate_sets": {}}
    sets = {"all": np.ones(n, dtype=bool), "hop1": (hop <= 1), "hop2": (hop != 1)}
    for name, mask in sets.items():
        res = {"n_candidates": int(mask.sum()), "n_positives": int(labels[mask].sum()), "methods": {}}
        y = labels[mask]
        for m, s in methods.items():
            auc = roc_auc_score(y, s[mask])
            lo, hi = boot_auc(s[mask], y, rng)
            entry = {"auc": round(auc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
            if m != "diffusion reach (ROTOR)":
                d, dlo, dhi = paired_boot_diff(scores[mask], s[mask], y, rng)
                entry["auc_diff_ic_minus_method"] = {"mean": round(d, 4), "ci_low": round(dlo, 4), "ci_high": round(dhi, 4)}
            if name == "all":
                entry["recall_at_k"] = {str(k): round(recall_at_k(s, labels, k), 4) for k in K_VALUES}
            res["methods"][m] = entry
            print(f"[{name}] {m:22s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]")
        out["candidate_sets"][name] = res

    out["ic_threshold_flagged"] = {str(t): {"flagged": int((scores >= t).sum()), "seeds_recovered": int(labels[scores >= t].sum())} for t in THRESHOLDS}
    with open(ROOT / "screening_extended.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved screening_extended.json")


if __name__ == "__main__":
    main()
