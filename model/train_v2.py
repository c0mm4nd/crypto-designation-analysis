#!/usr/bin/env python3
"""
WCFRM v2 training: role learning with size-independent clustering, a trained
importance MLP, and built-in role validation outputs.

Usage:
  python model/train_v2.py --csv <edges.csv> --tag <name> [--seeds seeds.txt|xlsx]
        [--n-clusters 9] [--epochs-p1 200] [--epochs-p2 100] [--seed 42] [--device cpu]
        [--ts-unit ms|s] [--lower]

Outputs (prefix v2_<tag>_seed<seed>):
  _Z.npy            learned embeddings (N, hid)
  _clusters.npy     role assignment with K = --n-clusters
  _scores_ic.npy    simulated IC coverage (importance target)
  _scores_mlp.npy   Phase-2 MLP prediction (learned importance)
  _nodes.txt        node order (address per line)
  _report.json      losses, silhouette over K, clustering stability, AUCs if labels given
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph  # noqa: E402
from model.diffusion import linear_reach_from_data  # noqa: E402
from model.ic_simulation import simulate_ic_parallel  # noqa: E402
from model.wcfrm_v2 import WCFRMv2, clustering_stability, silhouette_over_k  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_seed_addresses(path: str, chain: str) -> set:
    if path.endswith(".xlsx"):
        import openpyxl
        ws = openpyxl.load_workbook(path).active
        out = set()
        for r in ws.iter_rows(min_row=2, values_only=True):
            if len(r) >= 5 and r[3]:
                a, cur = str(r[3]).strip(), str(r[4] or "").upper()
                if cur in ("USDT", "TRX") and a.startswith("T"):
                    out.add(a)
        return out
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = "address"
        if chain == "eth":
            return set(df[df[col].str.startswith("0x", na=False)][col].str.lower())
        return set(df[df[col].str.startswith("T", na=False)][col])
    return set(l.strip().lower() if chain == "eth" else l.strip() for l in open(path) if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default=None, help="xlsx/csv/txt of anchor addresses (evaluation only)")
    ap.add_argument("--chain", default="tron", choices=["tron", "eth"])
    ap.add_argument("--n-clusters", type=int, default=9)
    ap.add_argument("--epochs-p1", type=int, default=200)
    ap.add_argument("--epochs-p2", type=int, default=100)
    ap.add_argument("--lr-p1", type=float, default=5e-4)
    ap.add_argument("--lr-p2", type=float, default=1e-3)
    ap.add_argument("--hid-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--lambda-ic", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cluster-every", type=int, default=10)
    ap.add_argument("--ssl-max-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--k-range", default="3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--out-dir", default="v2_runs")
    ap.add_argument("--value-cap", type=float, default=1e8)
    ap.add_argument("--importance", default="diffusion", choices=["diffusion", "mc"], help="deterministic path-based diffusion (default) or Monte-Carlo IC coverage")
    ap.add_argument("--diffusion-k", type=int, default=15, help="number of propagation steps for the deterministic score")
    ap.add_argument("--p-scale", type=float, default=0.1, help="IC activation probability of a w_ref USDT transfer")
    ap.add_argument("--recon", default="neighbour", choices=["neighbour", "feature"], help="phase-1 reconstruction objective")
    ap.add_argument("--cluster-head", default="kmeans", choices=["kmeans", "sinkhorn"], help="kmeans = post-hoc spherical k-means (v2/v3); sinkhorn = prototype head with optimal-transport targets, assignment read off the head")
    ap.add_argument("--own-dim", type=int, default=0, help="width of the address's own behavioural block (view A); 0 = infer (30 for v3 features)")
    ap.add_argument("--proto-temp", type=float, default=0.1)
    ap.add_argument("--proto-eps", type=float, default=0.05)
    ap.add_argument("--proto-residual", action="store_true", help="carry the address's own behavioural features into the prototype head alongside the embedding, so learning adds to the behavioural signature rather than replacing it")
    ap.add_argument("--proto-warmup", type=int, default=20, help="epochs for which the prototype head is trained against a seed-independent partition of the behavioural features before the swapped-prediction objective takes over")
    ap.add_argument("--proto-balance", type=float, default=1.0, help="strength of the prototype marginal constraint in [0,1]; 1 = equipartition (SwAV), 0 = unconstrained. Sizes of the learned roles are free to differ below 1.")
    ap.add_argument("--role-context", type=float, default=0.0, help="weight of the neighbourhood role-composition target (0 = off)")
    ap.add_argument("--encoder", default="attri", choices=["attri", "role"], help="attri = v1/v2 attention encoder; role = direction-aware encoder with a self term")
    ap.add_argument("--edge-sample", type=float, default=1.0, help="fraction of edges used per training step (DropEdge-style; 1.0 = full graph)")
    ap.add_argument("--features", default="v1", choices=["v1", "v3"], help="node feature set: v1 (12 structural) or v3 (behavioural + neighbourhood aggregates)")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="save model state every N phase-1 epochs so an interrupted run can be resumed")
    ap.add_argument("--resume", action="store_true", help="continue phase-1 training from the checkpoint if one exists")
    ap.add_argument("--ts-unit", default="ms", choices=["ms", "s"])
    ap.add_argument("--w-ref", type=float, default=1e8)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"v2_{args.tag}_seed{args.seed}")
    t0 = time.time()

    df = pd.read_csv(args.csv, usecols=["from", "to", "value", "block_timestamp"])
    df = df[(df["value"] > 0) & (df["value"] <= args.value_cap)]
    if args.chain == "eth":
        df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
    seed_set = load_seed_addresses(args.seeds, args.chain) if args.seeds else set()
    if args.features == "v3":
        from model.features_v3 import build_graph_v3
        data, node_to_idx, idx_to_node, _ = build_graph_v3(df, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref, ts_unit=args.ts_unit)
    else:
        data, node_to_idx, idx_to_node, _ = build_graph(df, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref)
    N = data.num_nodes
    labels = data.y.numpy()
    print(f"[{args.tag}] nodes={N:,} edges={data.edge_index.size(1):,} anchors_in_graph={int(labels.sum())} ({time.time()-t0:.0f}s)", flush=True)

    if args.importance == "diffusion":
        ic_raw = linear_reach_from_data(data, K=args.diffusion_k)
        print(f"[{args.tag}] deterministic diffusion score, K={args.diffusion_k}", flush=True)
    else:
        ic_raw = simulate_ic_parallel(data, seed=args.seed, n_workers=max(4, args.threads))
    alpha = 1000.0
    ic_log = torch.log1p(alpha * ic_raw) / torch.log1p(torch.tensor(alpha))
    ic_mask = ic_raw > 0
    print(f"[{args.tag}] IC done ({time.time()-t0:.0f}s)", flush=True)

    device = args.device
    data = data.to(device)
    x, ei, ea = data.x, data.edge_index, data.edge_attr
    ic_raw_d, ic_log_d, ic_mask_d = ic_raw.to(device), ic_log.to(device), ic_mask.to(device)

    recon_target = None
    if args.recon == "feature" and args.features == "v3":
        recon_target = x.clone()
    elif args.recon == "feature":
        from scipy.sparse import csr_matrix
        ei_np = ei.cpu().numpy(); Xn = x.cpu().numpy()
        A_out = csr_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(N, N)); A_in = A_out.T.tocsr()
        m_out = (A_out @ Xn) / np.maximum(np.asarray(A_out.sum(1)).ravel(), 1)[:, None]
        m_in = (A_in @ Xn) / np.maximum(np.asarray(A_in.sum(1)).ravel(), 1)[:, None]
        recon_target = torch.tensor(np.hstack([Xn, m_in, m_out]).astype(np.float32), device=device)
    model = WCFRMv2(in_dim=x.size(1), hid_dim=args.hid_dim, edge_dim=ea.size(1), n_layers=args.n_layers, heads=args.heads,
                    n_clusters=args.n_clusters, gamma=args.gamma, lambda_ic=args.lambda_ic, dropout=args.dropout,
                    recon_mode=args.recon, target_dim=(x.size(1) if args.features == "v3" else 3 * x.size(1)), encoder=args.encoder).to(device)
    if args.cluster_head == "sinkhorn":
        own_dim = args.own_dim or (30 if args.features == "v3" else x.size(1))
        model.enable_prototypes(own_dim, temperature=args.proto_temp, epsilon=args.proto_eps, balance=args.proto_balance, residual=args.proto_residual)
        model = model.to(device)
        print(f"[{args.tag}] prototype head: K={args.n_clusters}, view A = own {own_dim} behavioural dims, view B = graph embedding, balance={args.proto_balance}", flush=True)
        if args.proto_warmup > 0:
            from sklearn.cluster import KMeans as _KM
            warm = _KM(n_clusters=args.n_clusters, n_init=10, random_state=0).fit_predict(x[:, :own_dim].cpu().numpy())
            model.set_prototype_warmup(warm)
            np.save(prefix + "_warm.npy", warm.astype(np.int16))
            print(f"[{args.tag}] prototype warm-up partition from behavioural features, sizes={np.bincount(warm, minlength=args.n_clusters).tolist()} ({time.time()-t0:.0f}s)", flush=True)
    if args.role_context > 0:
        model.enable_role_context(args.n_clusters, weight=args.role_context)
        from scipy.sparse import csr_matrix as _csr
        _ei = ei.cpu().numpy()
        _A_out = _csr((np.ones(_ei.shape[1]), (_ei[0], _ei[1])), shape=(N, N)); _A_in = _A_out.T.tocsr()
        _dout = np.maximum(np.asarray(_A_out.sum(1)).ravel(), 1)[:, None]; _din = np.maximum(np.asarray(_A_in.sum(1)).ravel(), 1)[:, None]
        def role_hist(labels_np):
            H = np.zeros((N, args.n_clusters), dtype=np.float32); H[np.arange(N), labels_np] = 1.0
            return torch.tensor(np.hstack([(_A_in @ H) / _din, (_A_out @ H) / _dout]).astype(np.float32), device=device)
    role_target = None
    p1_params = list(model.encoder.parameters()) + list(model.ic_head.parameters()) + (list(model.decoder.parameters()) if model.decoder is not None else []) + (list(model.role_decoder.parameters()) if model.role_decoder is not None else [])
    if model.proto_head is not None:
        p1_params += list(model.proto_head.parameters()) + list(model.behaviour_mlp.parameters())

    def assign(Z):
        return model.prototype_assign(Z, x) if model.proto_head is not None else model.cluster(Z, seed=args.seed)
    opt1 = torch.optim.Adam(p1_params, lr=args.lr_p1)
    cluster_labels = None
    losses = []
    start_ep = 1
    ckpt_path = prefix + "_ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt1.load_state_dict(ck["opt1"])
        losses = ck["losses"]; start_ep = ck["epoch"] + 1
        cluster_labels = ck["cluster_labels"]
        if args.role_context > 0 and cluster_labels is not None:
            role_target = role_hist(cluster_labels)
        print(f"[{args.tag}] resumed from epoch {ck['epoch']} ({time.time()-t0:.0f}s)", flush=True)
    E = ei.size(1)
    gen = torch.Generator().manual_seed(args.seed)
    for ep in range(start_ep, args.epochs_p1 + 1):
        model.train()
        opt1.zero_grad()
        if args.edge_sample < 1.0:
            keep = torch.rand(E, generator=gen) < args.edge_sample
            ei_s, ea_s = ei[:, keep.to(ei.device)], ea[keep.to(ea.device)]
        else:
            ei_s, ea_s = ei, ea
        model.proto_warm = model.proto_head is not None and ep <= args.proto_warmup
        loss, _ = model.phase1_loss(x, ei_s, ea_s, cluster_labels, ssl_max_samples=args.ssl_max_samples, ic_scores=ic_raw_d, ic_mask=ic_mask_d, recon_target=recon_target, role_target=role_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        opt1.step()
        losses.append(float(loss.item()))
        if ep % args.cluster_every == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                Z = model.embed(x, ei, ea)
            cluster_labels = assign(Z)
            if args.role_context > 0:
                role_target = role_hist(cluster_labels)
        if ep % 20 == 0 or ep == 1:
            print(f"[{args.tag}] ep {ep:3d}/{args.epochs_p1} loss={loss.item():.5f} ({time.time()-t0:.0f}s)", flush=True)
        if args.checkpoint_every and ep % args.checkpoint_every == 0:
            torch.save({"model": model.state_dict(), "opt1": opt1.state_dict(), "losses": losses,
                        "epoch": ep, "cluster_labels": cluster_labels}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    model.eval()
    with torch.no_grad():
        Z = model.embed(x, ei, ea)
    Z_np = Z.cpu().numpy()
    clusters = assign(Z)
    print(f"[{args.tag}] phase 1 done ({time.time()-t0:.0f}s)", flush=True)

    # Phase 2: importance MLP on log-IC targets (encoder frozen)
    opt2 = torch.optim.Adam(model.mlp.parameters(), lr=args.lr_p2)
    Zd = Z.detach()
    for ep in range(1, args.epochs_p2 + 1):
        model.mlp.train()
        opt2.zero_grad()
        loss2 = model.phase2_loss(Zd, ic_log_d, ic_mask_d)
        loss2.backward()
        opt2.step()
    model.mlp.eval()
    with torch.no_grad():
        s_mlp = model.mlp(Zd).cpu().numpy().astype(np.float32)
    print(f"[{args.tag}] phase 2 done, final MSE={loss2.item():.5f} ({time.time()-t0:.0f}s)", flush=True)

    # Role validation on the learned embedding
    ks = [int(k) for k in args.k_range.split(",")]
    sil = silhouette_over_k(Z_np, ks, seed=args.seed)
    stab = clustering_stability(Z_np, args.n_clusters)
    stab.pop("partitions")
    stab["applies_to"] = "k-means on the embedding" if model.proto_head is None else "k-means on the embedding (reported for comparability; the reported partition comes from the prototype head and has no initialisation variance)"
    print(f"[{args.tag}] silhouette over K: {sil}; stability ARI mean={stab['ari_mean']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    report = {"tag": args.tag, "seed": args.seed, "n_nodes": int(N), "n_edges": int(data.edge_index.size(1)), "n_anchors_in_graph": int(labels.sum()),
              "n_clusters": args.n_clusters, "cluster_sizes": np.bincount(clusters, minlength=args.n_clusters).tolist(),
              "cluster_anchor_counts": [int(labels[clusters == k].sum()) for k in range(args.n_clusters)],
              "phase1_loss": losses, "phase2_final_mse": float(loss2.item()), "silhouette_over_k": sil, "clustering_stability": stab,
              "cluster_head": args.cluster_head, "args": vars(args), "runtime_s": time.time() - t0}
    if labels.sum() >= 5:
        from sklearn.metrics import roc_auc_score
        report["auc"] = {"ic_coverage": float(roc_auc_score(labels, ic_raw.numpy())), "mlp": float(roc_auc_score(labels, s_mlp))}
        print(f"[{args.tag}] AUC ic={report['auc']['ic_coverage']:.3f} mlp={report['auc']['mlp']:.3f}", flush=True)

    np.save(prefix + "_Z.npy", Z_np.astype(np.float32))
    if args.features == "v3":
        np.save(prefix + "_Xraw.npy", data.x_raw.cpu().numpy().astype(np.float32))
    np.save(prefix + "_clusters.npy", clusters.astype(np.int16))
    np.save(prefix + "_scores_ic.npy", ic_raw.numpy().astype(np.float32))
    np.save(prefix + "_scores_mlp.npy", s_mlp)
    with open(prefix + "_nodes.txt", "w") as f:
        f.write("\n".join(idx_to_node))
    with open(prefix + "_report.json", "w") as f:
        json.dump(report, f, indent=1)
    torch.save(model.state_dict(), prefix + "_model.pt")
    print(f"[{args.tag}] saved {prefix}_* ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
