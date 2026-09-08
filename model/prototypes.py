"""
Prototype-based role assignment with optimal-transport targets (WCFRM v4).

Motivation. Versions up to v3 followed the DeepCluster recipe: train an encoder, run
k-means on the embedding, use the resulting partition as a pseudo-label for a
self-supervised term, and repeat. That pipeline has two known weaknesses, both of which
show up in our validation: the pseudo-labels drift between refreshes, and the final
partition inherits the variance of k-means initialisation, so partitions from
independently trained models agree only moderately.

This module replaces it with the assignment scheme of SwAV (Caron et al., NeurIPS 2020)
and SeLa (Asano et al., ICLR 2020): a set of K learnable prototypes, soft assignments
obtained by a Sinkhorn-Knopp projection onto the transport polytope (which prevents the
degenerate solution without an ad-hoc collapse penalty), and a swapped-prediction loss.
The role of an address is then read directly off the prototype head, so no post-hoc
clustering step is involved at all.

The two views are chosen to encode the role concept rather than generic augmentation
invariance:

    view A  the address's own behavioural signature (an MLP on its own features),
    view B  its position in the graph (the direction-aware encoder over the
            neighbourhood-augmented features).

Requiring the two views to select the same prototype states that two addresses share a
role when they behave alike *and* sit in comparable neighbourhoods, which is the
definition of regular equivalence that role discovery targets, rather than the proximity
that a reconstruction loss encourages.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3, balance: float = 1.0) -> torch.Tensor:
    """Relaxed Sinkhorn-Knopp projection of prototype scores onto the transport polytope.

    `scores` is (N, K) of prototype logits; the return is (N, K) soft assignments.

    The balanced projection (`balance=1`) forces every prototype to receive N/K of the
    mass. That is the right constraint for the image collections SwAV was designed for,
    where classes are roughly equinumerous, but it is the wrong prior for a transfer
    network: functional roles in a heavy-tailed payment graph differ in size by orders of
    magnitude, and imposing equipartition would manufacture the uniformity rather than
    measure it.

    We therefore use the scaling iteration of unbalanced optimal transport (Chizat et al.,
    Math. Comp. 2018), which replaces the hard marginal constraint on the prototypes by a
    Kullback-Leibler penalty of strength rho. Its effect on the iteration is to raise the
    prototype scaling factor to the power `balance` = rho / (rho + epsilon), so that
    `balance=1` recovers the balanced projection and `balance=0` removes the prototype
    constraint entirely, leaving nothing to prevent collapse. Intermediate values keep the
    constraint as an anti-collapse device while letting role sizes be determined by the
    data. The constraint on the address side stays exact: every address distributes one
    unit of mass over the prototypes.
    """
    Q = torch.exp(scores / epsilon).t()  # (K, N)
    Q = Q / torch.clamp(Q.sum(), min=1e-12)
    K, N = Q.shape
    for _ in range(n_iters):
        if balance > 0:
            scale = 1.0 / torch.clamp(K * Q.sum(dim=1, keepdim=True), min=1e-12)
            Q = Q * scale.pow(balance)
        Q = Q / torch.clamp(Q.sum(dim=0, keepdim=True), min=1e-12) / N
    return (Q * N).t()


class PrototypeHead(nn.Module):
    """Shared L2-normalised prototypes with a swapped-prediction objective."""

    def __init__(self, dim: int, n_prototypes: int, temperature: float = 0.1, epsilon: float = 0.05, sinkhorn_iters: int = 3, balance: float = 1.0):
        super().__init__()
        self.prototypes = nn.Linear(dim, n_prototypes, bias=False)
        self.temperature = temperature
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.balance = balance

    def normalise_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.weight.data
            self.prototypes.weight.data = F.normalize(w, dim=1)

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        return self.prototypes(F.normalize(z, dim=-1))

    def swapped_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        s_a, s_b = self.scores(z_a), self.scores(z_b)
        with torch.no_grad():
            q_a = sinkhorn(s_a.detach(), self.epsilon, self.sinkhorn_iters, self.balance)
            q_b = sinkhorn(s_b.detach(), self.epsilon, self.sinkhorn_iters, self.balance)
        log_p_a = F.log_softmax(s_a / self.temperature, dim=-1)
        log_p_b = F.log_softmax(s_b / self.temperature, dim=-1)
        return -(0.5 * (q_b * log_p_a).sum(-1).mean() + 0.5 * (q_a * log_p_b).sum(-1).mean())

    @torch.no_grad()
    def assign(self, z: torch.Tensor, chunk: int = 2_000_000) -> np.ndarray:
        out = np.empty(z.size(0), dtype=np.int16)
        for i in range(0, z.size(0), chunk):
            out[i:i + chunk] = self.scores(z[i:i + chunk]).argmax(-1).cpu().numpy()
        return out


class BehaviourMLP(nn.Module):
    """View A: the address's own behavioural signature, without any graph context."""

    def __init__(self, in_dim: int, hid_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hid_dim))

    def forward(self, x):
        return self.net(x)
