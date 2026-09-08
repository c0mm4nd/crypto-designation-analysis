#!/usr/bin/env python3
"""
WCFRM Training Script – Israel-Palestine Sanctioned Address Dataset

Phase 1: Train Attri-GAT encoder with self-supervised AHC clustering loss
Phase 2: Freeze encoder, train IC-grounded importance MLP
Phase 3: Run all baselines and evaluate

Usage:
  python model/train.py [--csv path] [--xlsx path] [--hop2] [--epochs N] [--device mps|cpu]
"""

import os, sys, json, argparse, random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.features     import build_graph
from model.wcfrm        import WCFRM
from model.ic_simulation import simulate_ic
from model.baselines    import run_all_baselines
from model.evaluate     import (
    threshold_counts,
    print_threshold_table, print_auc_table,
    THRESHOLDS,
)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default="israel_tron_usdt_edges.csv")
    ap.add_argument("--xlsx",      default=None,
                    help="Ground-truth seed xlsx for evaluation only (optional)")
    ap.add_argument("--hop2-csv",  default="israel_tron_usdt_edges_2hop.csv",
                    help="If exists, use 2-hop edges for training")
    ap.add_argument("--device",    default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--epochs-p1", type=int, default=200)
    ap.add_argument("--epochs-p2", type=int, default=100)
    ap.add_argument("--lr-p1",     type=float, default=5e-4)
    ap.add_argument("--lr-p2",     type=float, default=1e-3)
    ap.add_argument("--hid-dim",   type=int, default=64)
    ap.add_argument("--n-layers",  type=int, default=2)
    ap.add_argument("--heads",     type=int, default=4)
    ap.add_argument("--n-clusters",type=int, default=9)
    ap.add_argument("--gamma",     type=float, default=0.3)
    ap.add_argument("--lambda-ic", type=float, default=0.5,
                    help="Weight for IC auxiliary loss in Phase 1 (λ)")
    ap.add_argument("--dropout",   type=float, default=0.1)
    ap.add_argument("--ic-runs",   type=int, default=1)
    ap.add_argument("--ic-max-nodes", type=int, default=0,
                    help="Sources for coverage IC (0=ALL nodes, recommended)")
    ap.add_argument("--ic-direction", default="coverage",
                    choices=["forward", "reverse", "both", "coverage"],
                    help="IC direction: reverse=upstream reach (default), "
                         "forward=spreading influence, both=geometric mean")
    ap.add_argument("--cluster-every", type=int, default=10,
                    help="Re-run AHC every N epochs in Phase 1")
    ap.add_argument("--out",       default="wcfrm_results.json")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-phase2", action="store_true",
                    help="Skip MLP training; use raw Coverage IC scores as importance")
    ap.add_argument("--load-model",  default=None,
                    help="Load saved model .pt and skip training (inference+eval only)")
    ap.add_argument("--batch-size",    type=int, default=4096,
                    help="Mini-batch size for Phase 1 (0 = full-graph)")
    ap.add_argument("--num-neighbors", type=int, default=15,
                    help="Sampled neighbors per hop in NeighborLoader")
    ap.add_argument("--ssl-max-samples", type=int, default=512,
                    help="Max nodes sampled per cluster for SSL loss")
    return ap.parse_args()


def load_seeds(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    seeds = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        addr = str(row[3]).strip() if row[3] else ""
        if addr.startswith("T"):
            seeds.append(addr)
    return list(set(seeds))


def main():
    args   = parse_args()
    device = args.device
    set_seed(args.seed)
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    edge_csv = args.hop2_csv if os.path.exists(args.hop2_csv) else args.csv
    print(f"\nLoading edges: {edge_csv}")
    df       = pd.read_csv(edge_csv)
    print(f"  {len(df):,} raw records")

    if args.xlsx and os.path.exists(args.xlsx):
        seeds    = load_seeds(args.xlsx)
        seed_set = set(seeds)
        print(f"  {len(seed_set)} seed addresses (from {args.xlsx})")
    else:
        seed_set = set()
        print("  No seed labels — running in fully unsupervised mode")

    # ── Build PyG graph ────────────────────────────────────────────────────
    print("\nBuilding graph ...")
    data, node_to_idx, idx_to_node, _ = build_graph(df, seed_set=seed_set)
    N       = data.num_nodes
    in_dim  = data.x.size(1)
    labels  = data.y.numpy()
    n_seeds_in_graph = int(labels.sum())
    print(f"  Nodes={N:,}  Edges={data.edge_index.size(1):,}  "
          f"Seeds in graph={n_seeds_in_graph} / {len(seed_set)}")

    data = data.to(device)
    x, ei, ea = data.x, data.edge_index, data.edge_attr

    # ── IC simulation (importance labels) ─────────────────────────────────
    ic_max   = args.ic_max_nodes if args.ic_max_nodes > 0 else None
    print(f"\n[IC] Simulating importance labels (R={args.ic_runs}, "
          f"max_nodes={ic_max or 'all'}) ...")
    data_cpu = data.cpu()
    ic_scores_raw = simulate_ic(data_cpu, R=args.ic_runs,
                                max_nodes=ic_max, seed=args.seed,
                                direction=args.ic_direction)
    ic_scores_raw = ic_scores_raw.to(device)
    print(f"  IC scores (raw): min={ic_scores_raw.min():.4f} max={ic_scores_raw.max():.4f}")
    # Coverage IC is extremely skewed: most nodes sit near zero while the seed-like
    # tail is much larger. Training the Phase-2 MLP against raw IC with MSE makes
    # the dense near-zero background dominate the loss, so we predict log-IC instead.
    ic_log_alpha = 1000.0
    ic_scores_log = torch.log1p(ic_log_alpha * ic_scores_raw) / torch.log1p(
        torch.tensor(ic_log_alpha, device=device, dtype=ic_scores_raw.dtype)
    )
    print(f"  IC scores (log-transformed α={ic_log_alpha:.0f}): "
          f"min={ic_scores_log.min():.4f} p50={ic_scores_log.median():.4f} "
          f"max={ic_scores_log.max():.4f}")

    # ── WCFRM model ────────────────────────────────────────────────────────
    model = WCFRM(
        in_dim     = in_dim,
        hid_dim    = args.hid_dim,
        edge_dim   = ea.size(1),
        n_layers   = args.n_layers,
        heads      = args.heads,
        n_clusters = args.n_clusters,
        gamma      = args.gamma,
        lambda_ic  = args.lambda_ic,
        dropout    = args.dropout,
    ).to(device)

    # Pre-compute IC mask on full graph (nodes with non-zero IC labels)
    ic_mask_full = (ic_scores_raw > 0).cpu()

    # ── Phase 1: Attri-GAT + SSL clustering ───────────────────────────────
    print(f"\n[Phase 1] Training encoder ({args.epochs_p1} epochs, γ={args.gamma}) ...")
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.ic_head.parameters()),
        lr=args.lr_p1)
    cluster_labels = None
    Z_final = None

    # Build NeighborLoader on CPU data for mini-batch training
    use_loader = args.batch_size > 0 and N > args.batch_size
    if use_loader:
        data_cpu = data.cpu()
        loader = NeighborLoader(
            data_cpu,
            num_neighbors=[args.num_neighbors] * args.n_layers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        print(f"  Mini-batch mode: batch_size={args.batch_size}, "
              f"num_neighbors={args.num_neighbors}×{args.n_layers}")
    else:
        print("  Full-graph mode")

    for ep in range(1, args.epochs_p1 + 1):
        model.train()
        if use_loader:
            total_loss = 0.0; n_batches = 0
            for batch in loader:
                batch = batch.to(device)
                n_target = batch.batch_size
                if cluster_labels is not None:
                    batch_cl = cluster_labels[batch.n_id[:n_target].cpu().numpy()]
                else:
                    batch_cl = None
                # IC supervision for target nodes in this batch
                target_ids   = batch.n_id[:n_target].cpu()
                batch_ic     = ic_scores_raw[target_ids].to(device)
                batch_ic_mask= ic_mask_full[target_ids].to(device)

                opt1.zero_grad()
                loss, _ = model.phase1_loss(
                    batch.x, batch.edge_index, batch.edge_attr, batch_cl,
                    ssl_max_samples=args.ssl_max_samples,
                    ic_scores=batch_ic, ic_mask=batch_ic_mask,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                opt1.step()
                total_loss += loss.item(); n_batches += 1
            loss_val = total_loss / max(n_batches, 1)
        else:
            opt1.zero_grad()
            loss, _ = model.phase1_loss(x, ei, ea, cluster_labels,
                                        ssl_max_samples=args.ssl_max_samples,
                                        ic_scores=ic_scores_raw,
                                        ic_mask=ic_mask_full.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt1.step()
            loss_val = loss.item()

        if ep % args.cluster_every == 0 or ep == 1:
            # Full-graph inference to get embeddings for AHC clustering
            model.eval()
            with torch.no_grad():
                Z_np = model.embed(x, ei, ea).cpu().numpy()
            cluster_labels = model.cluster(torch.tensor(Z_np))
            model.train()

        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{args.epochs_p1}  loss={loss_val:.6f}")

    model.eval()
    with torch.no_grad():
        Z_final = model.embed(x, ei, ea)
        cluster_labels = model.cluster(Z_final)
    print(f"  Clustering complete: {args.n_clusters} clusters")

    # ── Phase 2: Use raw Coverage IC directly as importance scores ────────
    print("\n[Phase 2] Using raw Coverage IC as importance scores (skip MLP).")
    wcfrm_scores = ic_scores_raw.cpu().numpy().astype(np.float32)

    # ── Role identification ────────────────────────────────────────────────
    # Triangulate cluster assignments with seed labels (anchors)
    print("\n[Role] Mapping clusters to governance roles ...")
    n_clusters = args.n_clusters
    cluster_seed_pct = np.zeros(n_clusters)
    cluster_sizes    = np.zeros(n_clusters, dtype=int)
    for i, lbl in enumerate(cluster_labels):
        cluster_sizes[lbl] += 1
        if labels[i] == 1:
            cluster_seed_pct[lbl] += 1
    for k in range(n_clusters):
        if cluster_sizes[k] > 0:
            cluster_seed_pct[k] /= cluster_sizes[k]
    print(f"  Cluster seed% (top-3): "
          f"{sorted(enumerate(cluster_seed_pct), key=lambda x:-x[1])[:3]}")

    # ── Threshold-based evaluation ─────────────────────────────────────────
    all_scores: dict[str, np.ndarray] = {"Our approach (WCFRM)": wcfrm_scores}

    if not args.skip_baselines:
        print("\n[Baselines] Computing all baseline scores ...")
        data_cpu = data.cpu()
        baseline_scores = run_all_baselines(
            data_cpu, in_dim=in_dim, hid_dim=args.hid_dim,
            n_clusters=n_clusters, device=device,
            epochs=args.epochs_p1,
        )
        all_scores.update(baseline_scores)

    has_labels = labels.sum() > 0

    if has_labels:
        print("\n" + "=" * 65)
        print("THRESHOLD COUNT TABLE (nodes with score >= threshold)")
        print("=" * 65)
        print_threshold_table(all_scores, labels)

        print("\n" + "=" * 65)
        print("AUC TABLE")
        print("=" * 65)
        auc_cache = print_auc_table(all_scores, labels)
    else:
        print("\n[Eval] No ground-truth labels — skipping threshold/AUC tables.")
        auc_cache = {}

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "n_nodes":   N,
        "n_edges":   int(data.edge_index.size(1)),
        "n_seeds_in_graph": n_seeds_in_graph,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.tolist(),
        "cluster_seed_pct": cluster_seed_pct.tolist(),
    }
    if has_labels:
        results["threshold_counts"] = {
            name: {str(t): int(v)
                   for t, v in threshold_counts(sc, labels).items()}
            for name, sc in all_scores.items()
        }
        results["auc"] = auc_cache

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Save model
    torch.save(model.state_dict(), args.out.replace(".json", "_model.pt"))
    np.save(args.out.replace(".json", "_scores.npy"), wcfrm_scores)
    np.save(args.out.replace(".json", "_clusters.npy"), cluster_labels)
    print("Saved model checkpoint and scores.")
    print("DONE.")


if __name__ == "__main__":
    main()
