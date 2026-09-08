"""
Baseline methods for comparison with WCFRM.

Centrality-based (practice-oriented):
  - Degree Centrality
  - Betweenness Centrality
  - PageRank

Unsupervised GNN + KMeans (research baselines):
  - GCN, GAT, GIN, GraphSAGE, SuperGAT, GATv2, GNN-FiLM

All return node importance scores in [0, 1] (higher = higher priority).
For GNN methods: score = embedding norm (distance from origin),
reflecting how distinctive each node's representation is.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from sklearn.cluster   import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, SAGEConv,
    SuperGATConv, GATv2Conv,
)
from torch_geometric.data import Data


# ── Centrality baselines ───────────────────────────────────────────────────

def degree_centrality(data: Data) -> np.ndarray:
    N  = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    dc = np.bincount(ei[1], minlength=N).astype(float)
    mx = dc.max() + 1e-9
    return (dc / mx).astype(np.float32)


def betweenness_centrality(data: Data, k: int = 200) -> np.ndarray:
    N  = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    G  = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(ei[0], ei[1]))
    bc  = nx.betweenness_centrality(G, k=min(k, N), normalized=True)
    arr = np.array([bc[i] for i in range(N)], dtype=np.float32)
    mx  = arr.max() + 1e-9
    return arr / mx


def pagerank_centrality(data: Data, alpha: float = 0.85) -> np.ndarray:
    N  = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    G  = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(ei[0], ei[1]))
    pr  = nx.pagerank(G, alpha=alpha)
    arr = np.array([pr[i] for i in range(N)], dtype=np.float32)
    mx  = arr.max() + 1e-9
    return arr / mx


# ── Generic GNN encoder wrapper ───────────────────────────────────────────

class _GNNEncoder(nn.Module):
    def __init__(self, gnn_type: str, in_dim: int, hid_dim: int,
                 n_layers: int = 2, heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            _in = in_dim if l == 0 else hid_dim
            if gnn_type == "GCN":
                self.layers.append(GCNConv(_in, hid_dim))
            elif gnn_type == "GAT":
                h = heads if l < n_layers - 1 else 1
                out = hid_dim // (h if l < n_layers - 1 else 1)
                self.layers.append(GATConv(_in, out, heads=h, concat=(l < n_layers-1)))
            elif gnn_type == "GIN":
                mlp = nn.Sequential(nn.Linear(_in, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, hid_dim))
                self.layers.append(GINConv(mlp))
            elif gnn_type == "SAGE":
                self.layers.append(SAGEConv(_in, hid_dim))
            elif gnn_type == "SuperGAT":
                self.layers.append(SuperGATConv(_in, hid_dim // heads, heads=heads,
                                                concat=(l < n_layers-1)))
            elif gnn_type == "GATv2":
                h = heads if l < n_layers - 1 else 1
                out = hid_dim // (h if l < n_layers - 1 else 1)
                self.layers.append(GATv2Conv(_in, out, heads=h, concat=(l < n_layers-1)))
            elif gnn_type == "FiLM":
                # GNN-FiLM: approximated via SAGEConv + feature-wise linear modulation
                self.layers.append(_FiLMConv(_in, hid_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.gnn_type = gnn_type
        self.hid_dim  = hid_dim

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class _FiLMConv(nn.Module):
    """GNN-FiLM: graph sage-style aggregation with feature-wise linear modulation."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sage = SAGEConv(in_ch, out_ch)
        self.film = nn.Linear(in_ch, 2 * out_ch)  # predicts gamma and beta

    def forward(self, x, edge_index):
        h   = self.sage(x, edge_index)
        gb  = self.film(x)                       # (N, 2*out_ch)
        N   = h.size(0); out_ch = h.size(1)
        # Align to actual out_ch in case sage has different output
        gb_cut = gb[:, :2 * out_ch]
        gamma  = gb_cut[:, :out_ch]
        beta   = gb_cut[:, out_ch:]
        return gamma * h + beta


def _train_gnn_encoder(model: _GNNEncoder, data: Data,
                        epochs: int = 200, lr: float = 1e-3,
                        device: str = "cpu") -> torch.Tensor:
    """Train GNN with reconstruction (neighborhood-smoothing) loss."""
    model = model.to(device)
    x     = data.x.to(device)
    ei    = data.edge_index.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        Z   = model(x, ei)
        # Reconstruction loss: same as Attri-GAT baseline
        src, dst = ei[0], ei[1]
        nb_sum   = torch.zeros_like(Z)
        cnt      = torch.zeros(Z.size(0), 1, device=device)
        nb_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, Z.size(1)), Z[src])
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=device))
        nb_mean = nb_sum / cnt.clamp(min=1.0)
        loss    = ((Z - nb_mean) ** 2).sum(1).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        Z = model(x, ei)
    return Z.cpu()


def gnn_kmeans_score(
    gnn_type: str,
    data: Data,
    in_dim: int,
    hid_dim: int = 64,
    n_clusters: int = 9,
    n_layers: int = 2,
    heads: int = 4,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """
    Train GNN, run KMeans on embeddings, return importance scores.

    Score = norm of embedding (||z_i||), normalized to [0, 1].
    Nodes with unusual (extreme) embeddings → higher importance.
    """
    torch.manual_seed(seed)
    enc = _GNNEncoder(gnn_type, in_dim, hid_dim, n_layers, heads)
    Z   = _train_gnn_encoder(enc, data, epochs=epochs, lr=lr, device=device)
    Z_np = Z.numpy()

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(Z_np)

    # Score = distance from own cluster centroid (inverted: farther = more anomalous)
    # Actually we use embedding norm as a proxy for "distinctiveness"
    norms = np.linalg.norm(Z_np, axis=1).astype(np.float32)
    mx    = norms.max() + 1e-9
    return norms / mx


# ── Convenience: run all baselines ────────────────────────────────────────

CENTRALITY_BASELINES = {
    "Degree Centrality": degree_centrality,
    "Betweenness":       betweenness_centrality,
    "PageRank":          pagerank_centrality,
}

GNN_BASELINES = ["GCN", "GAT", "GIN", "SAGE", "SuperGAT", "GATv2", "FiLM"]
GNN_LABELS    = {
    "GCN":      "GCN+KMeans",
    "GAT":      "GAT+KMeans",
    "GIN":      "GIN+KMeans",
    "SAGE":     "GraphSAGE+KMeans",
    "SuperGAT": "SuperGAT+KMeans",
    "GATv2":    "GATv2+KMeans",
    "FiLM":     "GNN-FiLM+KMeans",
}


def run_all_baselines(
    data: Data,
    in_dim: int,
    hid_dim: int = 64,
    n_clusters: int = 9,
    device: str = "cpu",
    epochs: int = 200,
    bc_k: int = 200,
) -> dict[str, np.ndarray]:
    """
    Returns {method_name: importance_score_array (N,)} for all baselines.
    """
    scores: dict[str, np.ndarray] = {}

    print("[baselines] Computing centrality scores ...")
    scores["Degree Centrality"] = degree_centrality(data)
    print("  Degree done")
    scores["Betweenness"] = betweenness_centrality(data, k=bc_k)
    print("  Betweenness done")
    scores["PageRank"] = pagerank_centrality(data)
    print("  PageRank done")

    print("[baselines] Training GNN encoders ...")
    for gnn_type in GNN_BASELINES:
        label = GNN_LABELS[gnn_type]
        print(f"  Training {label} ...")
        scores[label] = gnn_kmeans_score(
            gnn_type, data, in_dim=in_dim, hid_dim=hid_dim,
            n_clusters=n_clusters, epochs=epochs, device=device,
        )
        print(f"  {label} done")

    return scores
