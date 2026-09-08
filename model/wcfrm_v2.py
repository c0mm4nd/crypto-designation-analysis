"""
WCFRM v2: role learning with a single, size-independent clustering procedure.

Changes relative to wcfrm.py (v1):
  * Clustering is spherical k-means (sklearn KMeans, n_init=10) on L2-normalised
    embeddings for every graph size; v1 switched between Ward agglomerative
    clustering (<50k nodes) and MiniBatchKMeans (n_init=3) above that threshold,
    so role partitions were not comparable across networks.
  * The self-supervised clustering loss averages over several random cluster
    pairs per step instead of one, which lowers gradient variance.
  * Phase 2 (importance MLP on log-IC targets) is actually trained and its
    predictions are saved next to the simulated IC coverage, so both the
    simulated and the learned score are available for evaluation.
  * Utilities for role validation: silhouette over K on the embedding,
    adjusted Rand index across clustering seeds, and embedding export.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .attri_gat import AttrGATEncoder
from .role_gat import RoleGATEncoder
from .prototypes import BehaviourMLP, PrototypeHead


def _pair_loss(Zk: torch.Tensor, Zl: torch.Tensor) -> torch.Tensor:
    intra = ((Zk.unsqueeze(0) - Zk.unsqueeze(1)) ** 2).sum(-1).mean() if len(Zk) > 1 else torch.tensor(0.0, device=Zk.device)
    inter = ((Zk.unsqueeze(1) - Zl.unsqueeze(0)) ** 2).sum(-1).mean()
    return intra - inter


def ssl_clustering_loss(Z: torch.Tensor, labels: np.ndarray, max_samples: int = 512, n_pairs: int = 8) -> torch.Tensor:
    """Mean over `n_pairs` random cluster pairs of (intra - inter) squared distance on the unit sphere."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return torch.tensor(0.0, device=Z.device, requires_grad=True)
    Zn = F.normalize(Z, dim=-1)
    losses = []
    for _ in range(n_pairs):
        k, l = random.sample(list(unique), 2)
        ik = np.where(labels == k)[0]
        il = np.where(labels == l)[0]
        if len(ik) > max_samples:
            ik = np.random.choice(ik, max_samples, replace=False)
        if len(il) > max_samples:
            il = np.random.choice(il, max_samples, replace=False)
        losses.append(_pair_loss(Zn[torch.as_tensor(ik, device=Z.device)], Zn[torch.as_tensor(il, device=Z.device)]))
    return torch.stack(losses).mean()


class ImportanceMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).squeeze(-1)


def spherical_kmeans(Z: np.ndarray, k: int, seed: int = 0, n_init: int = 10) -> np.ndarray:
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return km.fit_predict(Zn)


def silhouette_over_k(Z: np.ndarray, ks, seed: int = 0, sample: int = 20000) -> dict[int, float]:
    """Silhouette on L2-normalised embeddings for each K, evaluated on a fixed random sample."""
    rng = np.random.default_rng(seed)
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    idx = rng.choice(len(Zn), size=min(sample, len(Zn)), replace=False)
    out = {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Zn)
        out[int(k)] = float(silhouette_score(Zn[idx], lab[idx]))
    return out


def clustering_stability(Z: np.ndarray, k: int, seeds=(0, 1, 2, 3, 4)) -> dict:
    """ARI between k-means runs with different initialisations on the same embedding."""
    parts = [spherical_kmeans(Z, k, seed=s) for s in seeds]
    aris = [adjusted_rand_score(parts[i], parts[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))]
    return {"k": int(k), "seeds": list(seeds), "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "partitions": parts}


class WCFRMv2(nn.Module):
    """recon_mode: 'neighbour' reproduces v1 (embedding pulled towards the neighbourhood mean,
    a proximity objective); 'feature' reconstructs the address's own structural features and
    the mean features of its in- and out-neighbourhoods from the embedding (a structural-
    equivalence objective, in the spirit of ReFeX/RolX), so that addresses with similar
    transaction signatures share roles even when they are not connected."""

    def __init__(self, in_dim=12, hid_dim=64, edge_dim=2, n_layers=2, heads=4, n_clusters=9, gamma=0.3, lambda_ic=0.5, dropout=0.1, mlp_hidden=128, ssl_pairs=8, recon_mode="neighbour", target_dim=None, encoder="attri"):
        super().__init__()
        self.encoder_kind = encoder
        self.encoder = (RoleGATEncoder if encoder == "role" else AttrGATEncoder)(in_dim, hid_dim, edge_dim, n_layers, heads, dropout)
        self.recon_mode = recon_mode
        self.decoder = nn.Sequential(nn.Linear(hid_dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, target_dim or 3 * in_dim)) if recon_mode == "feature" else None
        self.role_decoder = None
        self.role_weight = 0.0
        self.proto_head = None
        self.behaviour_mlp = None
        self.proto_weight = 0.0
        self.mlp = ImportanceMLP(hid_dim, mlp_hidden)
        self.ic_head = nn.Sequential(nn.Linear(hid_dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1), nn.Sigmoid())
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.lambda_ic = lambda_ic
        self.ssl_pairs = ssl_pairs
        self._cluster_labels = None

    def enable_prototypes(self, own_dim: int, weight: float = 1.0, temperature: float = 0.1, epsilon: float = 0.05, hidden: int = 128, balance: float = 1.0, residual: bool = False):
        """Replace post-hoc k-means by a prototype head with optimal-transport targets.

        `own_dim` is the width of the address's own behavioural block (the leading columns
        of the feature matrix), which forms view A. View B is the graph embedding.
        """
        hid = self.mlp.net[0].in_features
        self.behaviour_mlp = BehaviourMLP(own_dim, hid, hidden=hidden)
        self.proto_residual = residual
        self.proto_head = PrototypeHead(hid + own_dim if residual else hid, self.n_clusters, temperature=temperature, epsilon=epsilon, balance=balance)
        self.proto_weight = weight
        self.own_dim = own_dim
        self._warm_labels = None

    def set_prototype_warmup(self, labels: np.ndarray):
        """Fix a seed-independent starting partition for the prototype head.

        The prototypes are otherwise initialised at random, so role identity varies with
        the training seed even when the embedding does not. Anchoring the head to a
        partition determined by the data alone (k-means on the standardised behavioural
        features, fixed initialisation) removes that source of variance; the swapped-
        prediction objective then refines the partition for the remaining epochs.
        """
        self._warm_labels = torch.as_tensor(labels, dtype=torch.long)

    def prototype_warmup_loss(self, Z, x, index=None):
        tgt = self._warm_labels if index is None else self._warm_labels[index]
        tgt = tgt.to(Z.device)
        z_a, z_b = self._proto_views(Z, x)
        s_b = self.proto_head.scores(z_b) / self.proto_head.temperature
        s_a = self.proto_head.scores(z_a) / self.proto_head.temperature
        return 0.5 * F.cross_entropy(s_a, tgt) + 0.5 * F.cross_entropy(s_b, tgt)

    def _proto_views(self, Z, x):
        """The two views the prototype head compares.

        With `residual`, the address's own standardised behavioural features are carried
        into the head alongside each view. A partition read from the head can then always
        fall back on the behavioural signature, and what the encoder contributes is added
        to that signature rather than substituted for it; without the skip the learned
        partition can be worse than one obtained from the features alone.
        """
        own = x[:, : self.own_dim]
        z_a, z_b = self.behaviour_mlp(own), Z
        if getattr(self, "proto_residual", False):
            z_a, z_b = torch.cat([z_a, own], dim=-1), torch.cat([z_b, own], dim=-1)
        return z_a, z_b

    def prototype_loss(self, Z, x):
        z_a, z_b = self._proto_views(Z, x)
        return self.proto_head.swapped_loss(z_a, z_b)

    @torch.no_grad()
    def prototype_assign(self, Z, x=None) -> np.ndarray:
        if getattr(self, "proto_residual", False):
            assert x is not None, "residual prototype head needs the feature matrix to assign"
            return self.proto_head.assign(self._proto_views(Z, x)[1]).astype(int)
        return self.proto_head.assign(Z).astype(int)

    def enable_role_context(self, n_clusters: int, weight: float = 1.0, mlp_hidden: int = 128):
        hid = self.mlp.net[0].in_features
        self.role_decoder = nn.Sequential(nn.Linear(hid, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 2 * n_clusters))
        self.role_weight = weight

    def embed(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    @torch.no_grad()
    def cluster(self, Z: torch.Tensor, seed: int = 0) -> np.ndarray:
        labels = spherical_kmeans(Z.cpu().numpy(), self.n_clusters, seed=seed)
        self._cluster_labels = labels
        return labels

    def phase1_loss(self, x, edge_index, edge_attr, cluster_labels=None, ssl_max_samples=512, ic_scores=None, ic_mask=None, recon_target=None, role_target=None):
        Z = self.embed(x, edge_index, edge_attr)
        if self.recon_mode == "feature":
            l_agat = F.mse_loss(self.decoder(Z), recon_target)
            if role_target is not None and self.role_decoder is not None:
                # neighbourhood role composition (in- and out-neighbour role histograms): regular-equivalence signal
                logits = self.role_decoder(Z)
                K = role_target.size(1) // 2
                l_role = -(role_target[:, :K] * F.log_softmax(logits[:, :K], dim=1)).sum(1).mean() - (role_target[:, K:] * F.log_softmax(logits[:, K:], dim=1)).sum(1).mean()
                l_agat = l_agat + self.role_weight * l_role
        else:
            l_agat = AttrGATEncoder.reconstruction_loss(Z, edge_index, x.size(0))
        if self.proto_head is not None:
            self.proto_head.normalise_prototypes()
            l_ssl = self.prototype_warmup_loss(Z, x) if getattr(self, "proto_warm", False) and self._warm_labels is not None else self.prototype_loss(Z, x)
        else:
            l_ssl = ssl_clustering_loss(Z, cluster_labels, max_samples=ssl_max_samples, n_pairs=self.ssl_pairs) if cluster_labels is not None else torch.tensor(0.0, device=Z.device)
        l_ic = torch.tensor(0.0, device=Z.device)
        if ic_scores is not None and ic_mask is not None and ic_mask.sum() > 0:
            Zt = Z[: ic_scores.size(0)]
            l_ic = F.mse_loss(self.ic_head(Zt[ic_mask]).squeeze(-1), ic_scores[ic_mask])
        return (1 - self.gamma) * l_agat + self.gamma * l_ssl + self.lambda_ic * l_ic, Z

    def phase2_loss(self, Z, targets, mask=None):
        if mask is not None and mask.sum() > 0:
            return F.mse_loss(self.mlp(Z[mask]), targets[mask])
        return F.mse_loss(self.mlp(Z), targets)

    def forward(self, x, edge_index, edge_attr):
        Z = self.embed(x, edge_index, edge_attr)
        return Z, self.mlp(Z)
