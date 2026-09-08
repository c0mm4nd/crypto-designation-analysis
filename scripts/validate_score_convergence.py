#!/usr/bin/env python3
"""What the reported importance score is, and what its truncation costs.

The score sums weighted walks of length up to K into each address. Summing over walks
double-counts the ones that share nodes, and on these graphs the operator's spectral radius
exceeds one, so the untruncated sum diverges: the reported score is the partial sum at
K = 15, not an approximation to a convergent series. This script measures that directly and
bounds what it costs, by comparing the ranking with a Katz centrality on the same matrix
using an attenuation inside the radius of convergence.

Reports, for each network: the largest row sum of the transposed probability matrix, the
spectral radius by power iteration, the growth of the term totals, and the Spearman
correlation between the truncated score and Katz at a valid attenuation.

Usage:
  python scripts/validate_score_convergence.py [--out score_convergence.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NETWORKS = [
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv"),
    ("ukraine_tron", "ukraine_tron_usdt_edges_2hop.csv"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv"),
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv"),
]
P_SCALE, W_REF, VALUE_CAP, K = 0.1, 1e8, 1e8, 15


def build(csv: str):
    df = pd.read_csv(os.path.join(ROOT, csv), usecols=["from", "to", "value"])
    df = df[(df["value"] > 0) & (df["value"] <= VALUE_CAP)]
    nodes = sorted(set(df["from"]) | set(df["to"]))
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    w = df.groupby([df["from"].map(idx), df["to"].map(idx)])["value"].sum()
    pairs = np.array(list(w.index))
    p = P_SCALE * np.log1p(w.to_numpy()) / np.log1p(W_REF)
    return n, csr_matrix((p, (pairs[:, 1], pairs[:, 0])), shape=(n, n))


def spectral_radius(Pt, n, iters: int = 200) -> float:
    x = np.random.default_rng(0).random(n)
    for _ in range(iters):
        y = Pt @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            return 0.0
        x = y / nrm
    return float(np.linalg.norm(Pt @ x))


def walk_sum(Pt, n, k, alpha=1.0):
    c = np.zeros(n); v = np.ones(n)
    totals = []
    for _ in range(k):
        v = alpha * (Pt @ v)
        c += v
        totals.append(float(v.sum()))
    return c, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="score_convergence.json")
    args = ap.parse_args()
    out = {"p_scale": P_SCALE, "w_ref": W_REF, "K": K, "networks": {}}
    for tag, csv in NETWORKS:
        n, Pt = build(csv)
        rho = spectral_radius(Pt, n)
        c15, totals = walk_sum(Pt, n, K)
        rec = {"n_addresses": int(n), "max_row_sum": float(Pt.sum(1).max()),
               "spectral_radius": rho, "term_totals": totals,
               "term_growth_last_step": float(totals[-1] / totals[-2]) if totals[-2] > 0 else float("nan"),
               "diverges": bool(rho > 1)}
        if rho > 0:
            for frac in (0.5, 0.85):
                ck, _ = walk_sum(Pt, n, 60, alpha=frac / rho)
                rec[f"spearman_vs_katz_alpha_{frac}_over_rho"] = float(spearmanr(c15, ck).statistic)
        out["networks"][tag] = rec
        print(f"[{tag:30}] n={n:>9,} max row sum {rec['max_row_sum']:7.1f}  rho={rho:6.3f}  "
              f"growth {rec['term_growth_last_step']:5.2f}x/step  "
              f"Spearman vs Katz(0.85/rho) {rec.get('spearman_vs_katz_alpha_0.85_over_rho', float('nan')):.4f}",
              flush=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
