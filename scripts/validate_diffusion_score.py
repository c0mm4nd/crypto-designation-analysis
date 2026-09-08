#!/usr/bin/env python3
"""Validate the deterministic diffusion score against Monte-Carlo IC coverage.

Reports, per network: the agreement (Spearman) between two independent single-pass Monte-Carlo
runs, between a single pass and an R-pass average, and between the deterministic score and the
R-pass average, together with the runtime of each. Output: diffusion_validation.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph
from model.diffusion import linear_reach_from_data
from model.ic_simulation import simulate_ic_parallel
from model.train_v2 import load_seed_addresses

NETS = [
    ("ofac_iran_tron", "ofac_iran_tron_usdt_edges_2hop.csv", "ofac_iran.csv"),
    ("ofac_russia_ukraine_tron", "ofac_russia_ukraine_tron_usdt_edges_2hop.csv", "ofac_russia_ukraine.csv"),
    ("ofac_terrorist_financing_tron", "ofac_terrorist_financing_tron_usdt_edges_2hop.csv", "ofac_terrorist_financing.csv"),
    ("ukraine_tron", "ukraine_tron_usdt_edges_2hop.csv", "seeds_ukraine_tron.txt"),
    ("israel_tron", "israel_tron_usdt_edges_2hop.csv", "IsraelAddrs.xlsx"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="diffusion_validation.json")
    args = ap.parse_args()
    out = {}
    for tag, csv, seeds in NETS:
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv, usecols=["from", "to", "value", "block_timestamp"])
        df = df[(df["value"] > 0) & (df["value"] <= 1e8)]
        data, n2i, i2n, _ = build_graph(df)
        del df
        y = np.zeros(data.num_nodes, dtype=int)
        for a in load_seed_addresses(seeds, "tron"):
            if a in n2i:
                y[n2i[a]] = 1
        t = time.time(); lin = linear_reach_from_data(data).numpy(); t_lin = time.time() - t
        t = time.time(); mc1 = simulate_ic_parallel(data, seed=42, n_workers=args.workers).numpy(); t_mc = time.time() - t
        mc1b = simulate_ic_parallel(data, seed=7, n_workers=args.workers).numpy()
        acc = np.zeros(data.num_nodes)
        t = time.time()
        for r in range(args.reps):
            acc += simulate_ic_parallel(data, seed=1000 + r, n_workers=args.workers).numpy()
        mcR = acc / args.reps; t_mcR = time.time() - t
        rec = {"n_nodes": int(data.num_nodes), "n_edges": int(data.edge_index.size(1)), "n_anchors": int(y.sum()), "reps": args.reps,
               "seconds_linear": t_lin, "seconds_mc_1pass": t_mc, "seconds_mc_Rpass": t_mcR,
               "spearman_mc1_vs_mc1_other_seed": float(spearmanr(mc1, mc1b).statistic),
               "spearman_mc1_vs_mcR": float(spearmanr(mc1, mcR).statistic),
               "spearman_linear_vs_mcR": float(spearmanr(lin, mcR).statistic),
               "spearman_linear_vs_mc1": float(spearmanr(lin, mc1).statistic)}
        if y.sum() >= 5:
            rec["auc_linear"] = float(roc_auc_score(y, lin)); rec["auc_mc1"] = float(roc_auc_score(y, mc1)); rec["auc_mcR"] = float(roc_auc_score(y, mcR))
        out[tag] = rec
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
