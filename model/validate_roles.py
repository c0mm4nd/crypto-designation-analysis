#!/usr/bin/env python3
"""
Role validation for WCFRM v2 runs.

For a set of runs (same network, different training seeds) this script reports
  * stability of the role partition across training seeds (pairwise ARI),
  * stability across clustering initialisations (from the run reports),
  * silhouette over K on the learned embedding (from the run reports),
  * external validity against third-party entity labels (label purity / normalised
    mutual information between roles and label categories on labelled addresses),
  * anchor concentration (entropy of anchor distribution over roles; max density),
and compares the WCFRM partition with two role-discovery baselines computed on the
same graph: k-means on the 12 raw structural features (ReFeX-style) and RolX-lite
(recursive neighbourhood aggregation of the raw features followed by NMF).

Usage:
  python model/validate_roles.py --csv <edges.csv> --runs v2_runs/v2_<tag>_seed*_report.json
        [--chain tron|eth] [--labels labels_cache/labels.db] [--out validation_<tag>.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph  # noqa: E402


def label_map(db_path: str, chain: str) -> dict[str, str]:
    from labels.classify import classify
    con = sqlite3.connect(db_path)
    recs = defaultdict(dict)
    for addr, source, raw in con.execute("select address, source, raw_json from labels where chain=?", (chain,)):
        try:
            recs[addr][source] = json.loads(raw) if raw else None
        except Exception:
            recs[addr][source] = None
    out = {}
    for addr, rec in recs.items():
        try:
            out[addr] = classify(rec.get("web3resear"), rec.get("slowmist"))[0]
        except Exception:
            out[addr] = "UNKNOWN"
    return {a: c for a, c in out.items() if c != "UNKNOWN"}


def refex_features(X: np.ndarray, edge_index: np.ndarray, n: int, levels: int = 2) -> np.ndarray:
    """ReFeX-style recursive features: mean and sum of neighbour features over in- and out-neighbourhoods."""
    src, dst = edge_index
    A_out = csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A_in = A_out.T.tocsr()
    feats = [X]
    cur = X
    for _ in range(levels):
        new = []
        for A in (A_out, A_in):
            deg = np.asarray(A.sum(1)).ravel()
            s = A @ cur
            new.append(s / np.maximum(deg, 1)[:, None])
            new.append(np.log1p(np.abs(s)) * np.sign(s))
        cur = np.hstack(new)
        feats.append(cur)
    F = np.hstack(feats)
    return StandardScaler().fit_transform(F)


def rolx_lite(F: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    Fp = F - F.min(0, keepdims=True)
    W = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=300).fit_transform(Fp)
    return W.argmax(1)


def anchor_concentration(part: np.ndarray, anchors: np.ndarray) -> dict:
    k = int(part.max()) + 1
    counts = np.array([anchors[part == r].sum() for r in range(k)], dtype=float)
    sizes = np.array([(part == r).sum() for r in range(k)], dtype=float)
    dens = np.divide(counts, sizes, out=np.zeros(k), where=sizes > 0)
    p = counts / max(counts.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(k))
    return {"max_density_pct": float(100 * dens.max()), "entropy_normalised": ent, "share_in_densest_role": float(p.max()),
            "densest_role_share_of_addresses": float(sizes[dens.argmax()] / sizes.sum())}


def label_validity(part: np.ndarray, lab_idx: np.ndarray, lab_cat: np.ndarray) -> dict:
    if len(lab_idx) == 0:
        return {}
    nmi = float(normalized_mutual_info_score(lab_cat, part[lab_idx]))
    # purity: for each role, share of the majority label category among labelled members
    df = pd.DataFrame({"role": part[lab_idx], "cat": lab_cat})
    purity = float(df.groupby("role")["cat"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))
    cex_share = df[df["cat"] == "CEX"].groupby("role").size()
    k = int(part.max()) + 1
    cex_by_role = [int(cex_share.get(r, 0)) for r in range(k)]
    cex_density = [100 * cex_by_role[r] / max((part == r).sum(), 1) for r in range(k)]
    return {"n_labelled": int(len(lab_idx)), "nmi": nmi, "purity": purity, "cex_by_role": cex_by_role, "cex_density_pct_by_role": [float(x) for x in cex_density]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs", required=True, help="glob of *_report.json for one network")
    ap.add_argument("--chain", default="tron")
    ap.add_argument("--labels", default="labels_cache/labels.db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--value-cap", type=float, default=1e8)
    args = ap.parse_args()

    reports = [json.load(open(p)) for p in sorted(glob.glob(args.runs))]
    assert reports, "no run reports"
    prefixes = [p.replace("_report.json", "") for p in sorted(glob.glob(args.runs))]
    parts = [np.load(pre + "_clusters.npy").astype(int) for pre in prefixes]
    nodes = [l.rstrip("\n") for l in open(prefixes[0] + "_nodes.txt")]
    k = reports[0]["n_clusters"]

    df = pd.read_csv(args.csv, usecols=["from", "to", "value", "block_timestamp"])
    df = df[(df["value"] > 0) & (df["value"] <= args.value_cap)]
    if args.chain == "eth":
        df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
    data, node_to_idx, idx_to_node, _ = build_graph(df)
    assert idx_to_node == nodes, "node order mismatch between run and rebuilt graph"
    X = data.x.numpy()
    ei = data.edge_index.numpy()
    n = data.num_nodes

    # anchors from the report's cluster counts are not per-node; rebuild anchor vector from labels file if present in report args
    anchors = np.zeros(n, dtype=int)
    seeds_path = reports[0]["args"].get("seeds")
    if seeds_path and os.path.exists(seeds_path):
        from model.train_v2 import load_seed_addresses
        for a in load_seed_addresses(seeds_path, args.chain):
            if a in node_to_idx:
                anchors[node_to_idx[a]] = 1

    lm = label_map(args.labels, args.chain) if os.path.exists(args.labels) else {}
    lab_idx = np.array([node_to_idx[a] for a in lm if a in node_to_idx], dtype=int)
    lab_cat = np.array([lm[idx_to_node[i]] for i in lab_idx])

    out = {"tag": reports[0]["tag"], "n_runs": len(reports), "k": k, "n_nodes": int(n), "n_anchors": int(anchors.sum()),
           "silhouette_over_k": [r["silhouette_over_k"] for r in reports],
           "init_stability_ari": [r["clustering_stability"]["ari_mean"] for r in reports],
           "auc": [r.get("auc") for r in reports]}
    if len(parts) > 1:
        aris = [adjusted_rand_score(parts[i], parts[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))]
        out["seed_stability_ari"] = {"mean": float(np.mean(aris)), "min": float(np.min(aris)), "pairs": len(aris)}

    methods = {"ROTOR (seed %d)" % reports[0]["seed"]: parts[0]}
    Xs = StandardScaler().fit_transform(X)
    methods["k-means on raw features"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    xraw_path = prefixes[0] + "_Xraw.npy"
    if os.path.exists(xraw_path):
        Xb = np.load(xraw_path)
        methods["k-means on behavioural features"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xb)
    F = refex_features(X, ei, n)
    methods["RolX-lite (ReFeX + NMF)"] = rolx_lite(F, k)
    methods["k-means on ReFeX features"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(F)
    out["methods"] = {}
    for name, part in methods.items():
        out["methods"][name] = {"sizes": np.bincount(part, minlength=k).tolist(),
                                "anchor_concentration": anchor_concentration(part, anchors) if anchors.sum() else {},
                                "label_validity": label_validity(part, lab_idx, lab_cat),
                                "ari_vs_wcfrm": float(adjusted_rand_score(parts[0], part))}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({kk: vv for kk, vv in out.items() if kk not in ("silhouette_over_k",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
