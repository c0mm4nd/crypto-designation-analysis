#!/usr/bin/env python3
"""
WCFRM v2 training for graphs too large for full-batch training (e.g. the complete
Ukraine Ethereum two-hop network: ~25 M addresses, ~40 M unique edges).

Differences from train_v2.py
  * mini-batch training with CSR neighbour sampling (model/sampler.py); no pyg-lib needed
  * IC coverage estimated from a random subset of source addresses (--ic-sources)
  * embeddings for all nodes obtained by sampled inference in chunks after each epoch
  * clustering by k-means fitted on a random sample of embeddings and predicted for all
  * silhouette / stability computed on samples

Outputs the same files as train_v2.py (prefix v2_<tag>_seed<seed>) so that the
downstream analysis scripts apply unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import build_graph  # noqa: E402
from model.diffusion import linear_reach_from_data  # noqa: E402
from model.ic_simulation import simulate_ic_parallel  # noqa: E402
from model.sampler import CSRGraph, sample_batches  # noqa: E402
from model.train_v2 import load_seed_addresses  # noqa: E402
from model.wcfrm_v2 import WCFRMv2, ssl_clustering_loss  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def normalise(Z):
    """L2-normalise rows; returns a new array (use normalise_inplace for large Z)."""
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def normalise_inplace(Z, chunk=2_000_000):
    for i in range(0, len(Z), chunk):
        block = Z[i:i + chunk]
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)
    return Z


def fit_predict_kmeans(Z, k, seed, sample=1_000_000, n_init=10, chunk=2_000_000):
    """k-means on a normalised sample, then chunked prediction. Z is normalised in place
    (the embedding is only used through cosine geometry afterwards)."""
    rng = np.random.default_rng(seed)
    normalise_inplace(Z, chunk)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z[idx])
    del idx
    out = np.empty(len(Z), dtype=np.int16)
    for i in range(0, len(Z), chunk):
        out[i:i + chunk] = km.predict(Z[i:i + chunk])
    return out


def sampled_inference(model, graph, x, fanout, batch_size, rng, hid, threads_msg=""):
    Z = np.zeros((graph.n, hid), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for bx, ei, ea, n_id, nt in sample_batches(graph, x, batch_size, fanout, rng, shuffle=False):
            z = model.embed(bx, ei, ea)[:nt]
            Z[n_id[:nt]] = z.numpy()
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="raw transfer rows (from,to,value,block_timestamp)")
    ap.add_argument("--agg-edges", default=None, help="pre-aggregated edge table (from,to,w,cnt,mean_ts)")
    ap.add_argument("--agg-nodes", default=None, help="pre-aggregated node table (address,in_w,out_w,in_cnt,out_cnt,first_ts,last_ts)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--chain", default="eth", choices=["tron", "eth"])
    ap.add_argument("--ts-unit", default="s", choices=["ms", "s"])
    ap.add_argument("--features", default="v1", choices=["v1", "v3"])
    ap.add_argument("--n-clusters", type=int, default=9)
    ap.add_argument("--epochs-p1", type=int, default=3)
    ap.add_argument("--epochs-p2", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--fanout", default="10,10")
    ap.add_argument("--infer-fanout", default="25,25")
    ap.add_argument("--lr-p1", type=float, default=5e-4)
    ap.add_argument("--lr-p2", type=float, default=1e-3)
    ap.add_argument("--hid-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--lambda-ic", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recon", default="neighbour", choices=["neighbour", "feature"])
    ap.add_argument("--encoder", default="attri", choices=["attri", "role"])
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--role-context", type=float, default=0.0, help="weight of the neighbourhood role-composition target (0 = off)")
    ap.add_argument("--cluster-head", default="kmeans", choices=["kmeans", "sinkhorn"], help="kmeans = post-hoc k-means on the embedding; sinkhorn = prototype head with optimal-transport targets")
    ap.add_argument("--own-dim", type=int, default=0, help="width of the address's own behavioural block (view A); 0 = infer")
    ap.add_argument("--proto-temp", type=float, default=0.1)
    ap.add_argument("--proto-eps", type=float, default=0.05)
    ap.add_argument("--proto-residual", action="store_true", help="carry the address's own behavioural features into the prototype head alongside the embedding")
    ap.add_argument("--proto-warmup", type=int, default=1, help="epochs for which the prototype head is trained against a seed-independent partition of the behavioural features")
    ap.add_argument("--proto-balance", type=float, default=0.25, help="strength of the prototype marginal constraint in [0,1]; 1 = equipartition (SwAV), 0 = unconstrained")
    ap.add_argument("--ic-sources", type=int, default=1_000_000)
    ap.add_argument("--ic-cache", default=None, help="path to a precomputed IC coverage .npy (skips the simulation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="v2_runs")
    ap.add_argument("--value-cap", type=float, default=1e8)
    ap.add_argument("--importance", default="diffusion", choices=["diffusion", "mc"], help="deterministic path-based diffusion (default) or Monte-Carlo IC coverage")
    ap.add_argument("--diffusion-k", type=int, default=15, help="number of propagation steps for the deterministic score")
    ap.add_argument("--p-scale", type=float, default=0.1)
    ap.add_argument("--w-ref", type=float, default=1e8)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full pass over all nodes")
    ap.add_argument("--resume", action="store_true", help="load the checkpointed embedding and skip phase-1 training")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"v2_{args.tag}_seed{args.seed}")
    t0 = time.time()
    fanout = [int(f) for f in args.fanout.split(",")]
    infer_fanout = [int(f) for f in args.infer_fanout.split(",")]

    seed_set = load_seed_addresses(args.seeds, args.chain) if args.seeds else set()
    if args.agg_edges and args.features == "v3":
        from model.features_v3 import build_graph_v3_large
        data, node_to_idx, idx_to_node, _ = build_graph_v3_large(args.csv, args.agg_nodes, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref,
                                                                 ts_unit=args.ts_unit, lower=(args.chain == "eth"), value_cap=args.value_cap)
    elif args.agg_edges:
        from model.features import build_graph_from_agg
        data, node_to_idx, idx_to_node, _ = build_graph_from_agg(args.agg_edges, args.agg_nodes, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref,
                                                                 ts_unit=args.ts_unit, lower=(args.chain == "eth"))
    else:
        df = pd.read_csv(args.csv, usecols=["from", "to", "value", "block_timestamp"])
        df = df[(df["value"] > 0) & (df["value"] <= args.value_cap)]
        if args.chain == "eth":
            df["from"] = df["from"].str.lower(); df["to"] = df["to"].str.lower()
        print(f"[{args.tag}] {len(df):,} transfers loaded ({time.time()-t0:.0f}s)", flush=True)
        if args.features == "v3":
            from model.features_v3 import build_graph_v3
            data, node_to_idx, idx_to_node, _ = build_graph_v3(df, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref, ts_unit=args.ts_unit)
        else:
            data, node_to_idx, idx_to_node, _ = build_graph(df, seed_set=seed_set, p_scale=args.p_scale, w_ref=args.w_ref)
        del df
    N = data.num_nodes
    labels = data.y.numpy()
    print(f"[{args.tag}] nodes={N:,} edges={data.edge_index.size(1):,} anchors={int(labels.sum())} ({time.time()-t0:.0f}s)", flush=True)

    # IC coverage from sampled sources
    if args.importance == "diffusion" and not (args.ic_cache and os.path.exists(args.ic_cache)):
        ic_raw = linear_reach_from_data(data, K=args.diffusion_k)
        print(f"[{args.tag}] deterministic diffusion score, K={args.diffusion_k}", flush=True)
    elif args.ic_cache and os.path.exists(args.ic_cache):
        ic_raw = torch.from_numpy(np.load(args.ic_cache))
        assert len(ic_raw) == N, f"cached IC has {len(ic_raw)} entries, graph has {N}"
        print(f"[{args.tag}] loaded cached IC coverage from {args.ic_cache}", flush=True)
    else:
        ic_raw = simulate_ic_parallel(data, seed=args.seed, n_workers=max(4, args.threads), max_sources=args.ic_sources)
    alpha = 1000.0
    ic_log = torch.log1p(alpha * ic_raw) / torch.log1p(torch.tensor(alpha))
    ic_mask = ic_raw > 0
    np.save(prefix + "_scores_ic.npy", ic_raw.numpy().astype(np.float32))
    with open(prefix + "_nodes.txt", "w") as f:
        f.write("\n".join(idx_to_node))
    print(f"[{args.tag}] IC done and checkpointed ({time.time()-t0:.0f}s)", flush=True)

    x = data.x
    ei_np = data.edge_index.numpy(); ea_np = data.edge_attr.numpy()
    graph = CSRGraph(ei_np, ea_np, N)
    n_edges_total = int(data.edge_index.size(1))
    src_e = ei_np[0].astype(np.int64); dst_e = ei_np[1].astype(np.int64)
    recon_target = None
    if args.recon == "feature":
        if args.features == "v3":
            recon_target = x.clone()
        else:
            from scipy.sparse import csr_matrix
            Xn = x.numpy()
            A_out = csr_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(N, N)); A_in = A_out.T.tocsr()
            m_out = (A_out @ Xn) / np.maximum(np.asarray(A_out.sum(1)).ravel(), 1)[:, None]
            m_in = (A_in @ Xn) / np.maximum(np.asarray(A_in.sum(1)).ravel(), 1)[:, None]
            recon_target = torch.tensor(np.hstack([Xn, m_in, m_out]).astype(np.float32))
    if args.recon != "feature" and args.role_context <= 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long); data.edge_attr = torch.zeros((0, ea_np.shape[1]))
        del ei_np
        gc.collect()
    model = WCFRMv2(in_dim=x.size(1), hid_dim=args.hid_dim, edge_dim=ea_np.shape[1], n_layers=args.n_layers, heads=args.heads, n_clusters=args.n_clusters,
                    gamma=args.gamma, lambda_ic=args.lambda_ic, dropout=args.dropout, recon_mode=args.recon,
                    target_dim=(x.size(1) if args.features == "v3" else 3 * x.size(1)), encoder=args.encoder)
    if args.cluster_head == "sinkhorn":
        own_dim = args.own_dim or (30 if args.features == "v3" else x.size(1))
        model.enable_prototypes(own_dim, temperature=args.proto_temp, epsilon=args.proto_eps, balance=args.proto_balance, residual=args.proto_residual)
        print(f"[{args.tag}] prototype head: K={args.n_clusters}, view A = own {own_dim} behavioural dims, balance={args.proto_balance}, residual={args.proto_residual}", flush=True)
        if args.proto_warmup > 0:
            warm = fit_predict_kmeans(x[:, :own_dim].numpy(), args.n_clusters, 0, n_init=10).astype(np.int64)
            model.set_prototype_warmup(warm)
            print(f"[{args.tag}] prototype warm-up partition from behavioural features, sizes={np.bincount(warm, minlength=args.n_clusters).tolist()} ({time.time()-t0:.0f}s)", flush=True)
    if args.role_context > 0:
        model.enable_role_context(args.n_clusters, weight=args.role_context)

        def role_hist(labels_np):
            """In- and out-neighbour role composition per address, from the edge arrays."""
            K = args.n_clusters
            lab = labels_np.astype(np.int64)
            hin = np.bincount(dst_e * K + lab[src_e], minlength=N * K).reshape(N, K).astype(np.float32)
            hout = np.bincount(src_e * K + lab[dst_e], minlength=N * K).reshape(N, K).astype(np.float32)
            hin /= np.maximum(hin.sum(1, keepdims=True), 1); hout /= np.maximum(hout.sum(1, keepdims=True), 1)
            return torch.from_numpy(np.hstack([hin, hout]))
    role_target = None
    params = list(model.encoder.parameters()) + list(model.ic_head.parameters()) + (list(model.decoder.parameters()) if model.decoder is not None else []) + (list(model.role_decoder.parameters()) if model.role_decoder is not None else [])
    if model.proto_head is not None:
        params += list(model.proto_head.parameters()) + list(model.behaviour_mlp.parameters())
    opt1 = torch.optim.Adam(params, lr=args.lr_p1)

    def assign_all(Z_arr, n_init=10, chunk=2_000_000):
        """Role assignment for every address: from the prototype head, or post-hoc k-means."""
        if model.proto_head is None:
            return fit_predict_kmeans(Z_arr, args.n_clusters, args.seed, n_init=n_init).astype(np.int64)
        out = np.empty(len(Z_arr), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(Z_arr), chunk):
                v = torch.from_numpy(np.ascontiguousarray(Z_arr[i:i + chunk]))
                if getattr(model, "proto_residual", False):
                    v = torch.cat([v, x[i:i + chunk, :model.own_dim]], dim=-1)
                out[i:i + chunk] = model.proto_head.scores(v).argmax(-1).numpy()
        return out
    rng = np.random.default_rng(args.seed)
    cluster_labels = None
    losses = []
    resumed = False
    if args.resume and os.path.exists(prefix + "_Z.npy"):
        Z_all = np.load(prefix + "_Z.npy")
        if os.path.exists(prefix + "_model.pt"):
            model.load_state_dict(torch.load(prefix + "_model.pt"))
        cluster_labels = assign_all(Z_all)
        np.save(prefix + "_clusters.npy", cluster_labels.astype(np.int16))
        resumed = True
        print(f"[{args.tag}] resumed from checkpointed embedding ({time.time()-t0:.0f}s)", flush=True)
    for ep in range(1, 0 if resumed else args.epochs_p1 + 1):
        model.train()
        step = 0; run_loss = 0.0
        for bx, ei, ea, n_id, nt in sample_batches(graph, x, args.batch_size, fanout, rng, shuffle=True):
            tgt = n_id[:nt]
            opt1.zero_grad()
            Z = model.embed(bx, ei, ea)
            if args.recon == "feature":
                l_rec = torch.nn.functional.mse_loss(model.decoder(Z[:nt]), recon_target[tgt])
            else:
                from model.attri_gat import AttrGATEncoder
                l_rec = AttrGATEncoder.reconstruction_loss(Z, ei, bx.size(0))
            if model.proto_head is not None:
                model.proto_head.normalise_prototypes()
                l_ssl = (model.prototype_warmup_loss(Z[:nt], bx[:nt], index=tgt)
                         if ep <= args.proto_warmup and model._warm_labels is not None
                         else model.prototype_loss(Z[:nt], bx[:nt]))
            else:
                l_ssl = ssl_clustering_loss(Z[:nt], cluster_labels[tgt], n_pairs=model.ssl_pairs) if cluster_labels is not None else torch.tensor(0.0)
            m = ic_mask[tgt]
            l_ic = torch.nn.functional.mse_loss(model.ic_head(Z[:nt][m]).squeeze(-1), ic_raw[tgt][m]) if m.sum() > 0 else torch.tensor(0.0)
            loss = (1 - args.gamma) * l_rec + args.gamma * l_ssl + args.lambda_ic * l_ic
            if role_target is not None and model.role_decoder is not None:
                logits = model.role_decoder(Z[:nt]); K = args.n_clusters
                rt = role_target[tgt]
                l_role = -(rt[:, :K] * torch.nn.functional.log_softmax(logits[:, :K], dim=1)).sum(1).mean() \
                         - (rt[:, K:] * torch.nn.functional.log_softmax(logits[:, K:], dim=1)).sum(1).mean()
                loss = loss + model.role_weight * l_role
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt1.step()
            run_loss += float(loss.item()); step += 1
            if step % 500 == 0:
                print(f"[{args.tag}] ep {ep} step {step} loss={run_loss/step:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
        losses.append(run_loss / max(step, 1))
        if "Z_all" in dir():
            del Z_all
        gc.collect()
        Z_all = sampled_inference(model, graph, x, infer_fanout, 4096, np.random.default_rng(args.seed + ep), args.hid_dim)
        last_epoch = ep == args.epochs_p1
        cluster_labels = assign_all(Z_all, n_init=(10 if last_epoch else 3))
        if args.role_context > 0 and not last_epoch:
            role_target = role_hist(cluster_labels)
        np.save(prefix + "_Z.npy", Z_all)
        np.save(prefix + "_clusters.npy", cluster_labels.astype(np.int16))
        torch.save(model.state_dict(), prefix + "_model.pt")
        gc.collect()
        print(f"[{args.tag}] epoch {ep} done, mean loss={losses[-1]:.4f}, clusters refreshed and checkpointed ({time.time()-t0:.0f}s)", flush=True)

    Z_np = Z_all
    clusters = cluster_labels.astype(np.int16)
    # Phase 2 on all nodes (batched)
    Zt = torch.from_numpy(Z_np)  # shares memory with Z_np
    opt2 = torch.optim.Adam(model.mlp.parameters(), lr=args.lr_p2)
    idx_lab = torch.where(ic_mask)[0]
    for ep in range(1, args.epochs_p2 + 1):
        model.mlp.train()
        sel = idx_lab[torch.randperm(len(idx_lab))[:200_000]]
        opt2.zero_grad()
        loss2 = torch.nn.functional.mse_loss(model.mlp(Zt[sel]), ic_log[sel])
        loss2.backward(); opt2.step()
    model.mlp.eval()
    s_mlp = np.zeros(N, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, 1_000_000):
            s_mlp[i:i + 1_000_000] = model.mlp(Zt[i:i + 1_000_000]).numpy()
    # validation on samples
    rs = np.random.default_rng(args.seed)
    sidx = rs.choice(N, size=min(200_000, N), replace=False)
    Zs = normalise(np.array(Z_np[sidx]))
    sil = {}
    for k in range(3, 13):
        lab = KMeans(n_clusters=k, n_init=5, random_state=args.seed).fit_predict(Zs)
        sub = rs.choice(len(Zs), size=min(20_000, len(Zs)), replace=False)
        sil[k] = float(silhouette_score(Zs[sub], lab[sub]))
    parts = [KMeans(n_clusters=args.n_clusters, n_init=10, random_state=s).fit_predict(Zs) for s in range(3)]
    aris = [adjusted_rand_score(parts[i], parts[j]) for i in range(3) for j in range(i + 1, 3)]
    report = {"tag": args.tag, "seed": args.seed, "n_nodes": int(N), "n_edges": n_edges_total, "n_anchors_in_graph": int(labels.sum()), "n_clusters": args.n_clusters,
              "cluster_sizes": np.bincount(clusters, minlength=args.n_clusters).tolist(), "cluster_anchor_counts": [int(labels[clusters == k].sum()) for k in range(args.n_clusters)],
              "phase1_loss": losses, "phase2_final_mse": float(loss2.item()), "silhouette_over_k": sil,
              "clustering_stability": {"k": args.n_clusters, "seeds": [0, 1, 2], "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "note": "on a 200k-address sample"},
              "cluster_head": args.cluster_head, "args": vars(args), "runtime_s": time.time() - t0, "training": "mini-batch neighbour sampling; IC from sampled sources"}
    if labels.sum() >= 5:
        from sklearn.metrics import roc_auc_score
        report["auc"] = {"ic_coverage": float(roc_auc_score(labels, ic_raw.numpy())), "mlp": float(roc_auc_score(labels, s_mlp))}
    np.save(prefix + "_Z.npy", Z_np.astype(np.float32))
    np.save(prefix + "_clusters.npy", clusters)
    np.save(prefix + "_scores_ic.npy", ic_raw.numpy().astype(np.float32))
    np.save(prefix + "_scores_mlp.npy", s_mlp)
    if args.features == "v3":
        np.save(prefix + "_Xraw.npy", data.x_raw.numpy().astype(np.float32))
    with open(prefix + "_nodes.txt", "w") as f:
        f.write("\n".join(idx_to_node))
    with open(prefix + "_report.json", "w") as f:
        json.dump(report, f, indent=1)
    torch.save(model.state_dict(), prefix + "_model.pt")
    print(f"[{args.tag}] saved {prefix}_* ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
