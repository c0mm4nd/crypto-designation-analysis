#!/usr/bin/env python3
"""Independent-cascade coverage for a large network, computed from the light graph builder.

Building the scenario feature matrix before the simulation leaves too little memory for the
worker pool on the complete Ukraine Ethereum network, so the IC target is computed here from
the aggregate edge/node tables alone (node order identical to the feature builder) and cached
as a .npy file that model/train_large.py loads with --ic-cache.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph_from_agg
from model.ic_simulation import simulate_ic_parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-edges", required=True)
    ap.add_argument("--agg-nodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ts-unit", default="s")
    ap.add_argument("--lower", type=int, default=1)
    ap.add_argument("--sources", type=int, default=0, help="0 = every address as a source")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-scale", type=float, default=0.1)
    ap.add_argument("--w-ref", type=float, default=1e8)
    args = ap.parse_args()
    t0 = time.time()
    data, _, idx_to_node, _ = build_graph_from_agg(args.agg_edges, args.agg_nodes, ts_unit=args.ts_unit,
                                                   lower=bool(args.lower), p_scale=args.p_scale, w_ref=args.w_ref,
                                                   return_addresses=False)
    print(f"graph: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges ({time.time()-t0:.0f}s)", flush=True)
    data.x = None  # the simulation only needs edge_index and prop_prob
    ic = simulate_ic_parallel(data, seed=args.seed, n_workers=args.workers,
                              max_sources=(args.sources if args.sources > 0 else None))
    np.save(args.out, ic.numpy().astype(np.float32))
    print(f"saved {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
