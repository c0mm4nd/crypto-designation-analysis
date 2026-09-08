"""
WCFRM: Wartime Cryptocurrency Financing Role Mapping

Two-phase training:
  Phase 1 – Attri-GAT encoder + self-supervised AHC clustering loss
            + IC-supervised auxiliary head (joint training)
  Phase 2 – Fine-tune IC-grounded importance MLP

Phase-1 joint loss (Section 5.3):
  L_Total = (1-γ)·L_AGAT + γ·L_SSL + λ·L_IC_aux

  L_AGAT  = Σ_i || z_i - mean_{j∈N(i)} z_j ||^2   (reconstruction)
  L_SSL   = intra-cluster dist - inter-cluster dist  (self-supervised clustering)
  L_IC_aux = MSE(ICHead(Z[S]), ŝ[S])               (IC supervision on sampled nodes S)

By jointly optimising L_IC_aux in Phase 1, the GAT encoder learns to embed
IC-coverage information into Z, allowing Phase 2 to generalize IC predictions
to all nodes via graph message-passing.

Phase-2 MLP:
  S_log = MLP(Z)   (scalar per node in log-IC space)
  L_IC = MSE(S_log, ŝ_log)   (fine-tune on log-transformed IC labels)
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from .attri_gat import AttrGATEncoder


# ── SSL clustering loss ────────────────────────────────────────────────────

def ssl_clustering_loss(
    Z: torch.Tensor,
    labels: np.ndarray,
    max_samples: int = 512,
) -> torch.Tensor:
    """
    Compute L_SSL on a randomly chosen pair of clusters.
    Samples up to `max_samples` nodes per cluster to keep memory bounded.
    Returns scalar tensor.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return torch.tensor(0.0, device=Z.device, requires_grad=True)

    k, l = random.sample(list(unique), 2)
    idx_k_np = np.where(labels == k)[0]
    idx_l_np = np.where(labels == l)[0]

    if len(idx_k_np) > max_samples:
        idx_k_np = np.random.choice(idx_k_np, max_samples, replace=False)
    if len(idx_l_np) > max_samples:
        idx_l_np = np.random.choice(idx_l_np, max_samples, replace=False)

    idx_k = torch.tensor(idx_k_np, dtype=torch.long, device=Z.device)
    idx_l = torch.tensor(idx_l_np, dtype=torch.long, device=Z.device)

    # L2-normalize to unit sphere so pairwise distances are bounded in [0, 4]
    import torch.nn.functional as _F
    Zk = _F.normalize(Z[idx_k], dim=-1)
    Zl = _F.normalize(Z[idx_l], dim=-1)

    # Intra-cluster distance (mean pairwise squared)
    if len(idx_k) > 1:
        diff_k  = Zk.unsqueeze(0) - Zk.unsqueeze(1)     # (S, S, D)
        intra   = (diff_k ** 2).sum(-1).mean()
    else:
        intra   = torch.tensor(0.0, device=Z.device)

    # Inter-cluster distance (mean pairwise squared)
    diff_kl = Zk.unsqueeze(1) - Zl.unsqueeze(0)         # (S_k, S_l, D)
    inter   = (diff_kl ** 2).sum(-1).mean()

    return intra - inter


# ── Importance MLP ─────────────────────────────────────────────────────────

class ImportanceMLP(nn.Module):
    """3-layer MLP: Z → scalar importance score in (0, 1)."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),      nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)   # (N,)


# ── Main WCFRM model ───────────────────────────────────────────────────────

class WCFRM(nn.Module):
    """
    Full WCFRM model.

    Parameters
    ----------
    in_dim       : node feature dimension
    hid_dim      : GAT hidden / embedding dimension
    edge_dim     : edge feature dimension
    n_layers     : number of GAT layers
    heads        : GAT attention heads
    n_clusters   : number of AHC clusters (K)
    gamma        : weight for SSL loss  (γ in paper; note paper uses λ but
                   the algorithm uses γ — we follow the algorithm)
    dropout      : attention dropout
    mlp_hidden   : hidden size of importance MLP
    """

    def __init__(
        self,
        in_dim:     int = 12,
        hid_dim:    int = 64,
        edge_dim:   int = 2,
        n_layers:   int = 2,
        heads:      int = 4,
        n_clusters: int = 9,
        gamma:      float = 0.3,
        lambda_ic:  float = 0.5,
        dropout:    float = 0.1,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.encoder    = AttrGATEncoder(in_dim, hid_dim, edge_dim, n_layers, heads, dropout)
        self.mlp        = ImportanceMLP(hid_dim, mlp_hidden)
        # Lightweight auxiliary IC head used jointly in Phase 1
        self.ic_head    = nn.Sequential(
            nn.Linear(hid_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1), nn.Sigmoid(),
        )
        self.n_clusters = n_clusters
        self.gamma      = gamma
        self.lambda_ic  = lambda_ic
        self._cluster_labels: np.ndarray | None = None

    # ── Embed ──────────────────────────────────────────────────────────────
    def embed(self, x, edge_index, edge_attr) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_attr)

    # ── AHC / KMeans clustering ────────────────────────────────────────────
    _LARGE_N_THRESHOLD = 50_000   # use MiniBatchKMeans above this

    @torch.no_grad()
    def cluster(self, Z: torch.Tensor) -> np.ndarray:
        Z_np = Z.cpu().numpy()
        if len(Z_np) > self._LARGE_N_THRESHOLD:
            # AgglomerativeClustering is O(N²) — use MiniBatchKMeans for large graphs
            km = MiniBatchKMeans(
                n_clusters=self.n_clusters, random_state=0,
                batch_size=min(10240, len(Z_np)), n_init=3,
            )
            labels = km.fit_predict(Z_np)
        else:
            ahc = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage="ward"
            )
            labels = ahc.fit_predict(Z_np)
        self._cluster_labels = labels
        return labels

    # ── Phase-1 loss (joint: reconstruction + SSL + IC auxiliary) ────────
    def phase1_loss(
        self,
        x, edge_index, edge_attr,
        cluster_labels: np.ndarray | None = None,
        ssl_max_samples: int = 512,
        ic_scores: torch.Tensor | None = None,   # (n_target,) for batch nodes
        ic_mask:   torch.Tensor | None = None,   # (n_target,) bool — nodes w/ IC label
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Z = self.embed(x, edge_index, edge_attr)

        l_agat = AttrGATEncoder.reconstruction_loss(Z, edge_index, x.size(0))

        if cluster_labels is not None:
            l_ssl = ssl_clustering_loss(Z, cluster_labels, max_samples=ssl_max_samples)
        else:
            l_ssl = torch.tensor(0.0, device=Z.device)

        # IC auxiliary supervision on sampled nodes (Phase-1 joint term)
        l_ic = torch.tensor(0.0, device=Z.device)
        if ic_scores is not None and ic_mask is not None and ic_mask.sum() > 0:
            n_target = ic_scores.size(0)
            Z_target = Z[:n_target]                      # only target nodes in batch
            Z_sel    = Z_target[ic_mask]
            ic_sel   = ic_scores[ic_mask]
            pred     = self.ic_head(Z_sel).squeeze(-1)
            l_ic     = F.mse_loss(pred, ic_sel)

        loss = (1 - self.gamma) * l_agat + self.gamma * l_ssl + self.lambda_ic * l_ic
        return loss, Z

    # ── Phase-2 loss ───────────────────────────────────────────────────────
    def phase2_loss(
        self,
        Z: torch.Tensor,
        ic_scores: torch.Tensor,
        ic_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Train the Phase-2 MLP on the targets passed in from train.py.
        These targets are log-transformed IC scores so the near-zero background
        mass does not swamp the higher-IC tail under MSE.
        """
        if ic_mask is not None and ic_mask.sum() > 0:
            S    = self.mlp(Z[ic_mask])
            loss = F.mse_loss(S, ic_scores[ic_mask])
        else:
            S    = self.mlp(Z)
            loss = F.mse_loss(S, ic_scores)
        return loss

    # ── Full forward (inference) ───────────────────────────────────────────
    def forward(self, x, edge_index, edge_attr):
        Z = self.embed(x, edge_index, edge_attr)
        S = self.mlp(Z)
        return Z, S
