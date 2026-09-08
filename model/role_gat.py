"""
Direction-aware role encoder (WCFRM v3 architecture).

The v1/v2 encoder (model/attri_gat.py) computes z_i = sigma(sum_j alpha_ij W z_j):
the output of a layer contains only aggregated neighbour information, the node's own
signature is not carried through, and incoming and outgoing transfers are pooled into
one neighbourhood. Both properties work against role discovery on directed, value-
weighted transfer graphs, where a node's own behaviour (how much it receives, how
quickly it forwards) and the asymmetry between its in- and out-side are exactly what
distinguishes a collection wallet from a relay or an exchange hot wallet.

RoleGATConv keeps three parts per layer and concatenates them:

    h_i = W_self x_i  ||  sum_{j -> i} alpha^in_ij  (W_in x_j || W_e e_ji)
                      ||  sum_{i -> j} alpha^out_ij (W_out x_j || W_e e_ij)

with separate attention parameters for the in- and out-side, followed by a linear
projection and a non-linearity. The self term makes the encoder a refinement of the
node's own features rather than a smoother of its neighbourhood, and the split gives
the model the direction asymmetry that defines the functional families.

Losses
  * reconstruction of the node's own features and of the mean features of its in- and
    out-neighbourhoods (structural equivalence, ReFeX-style),
  * optional reconstruction of the role composition of the in- and out-neighbourhoods
    (regular equivalence), handled by WCFRMv2.enable_role_context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class _DirectedAttention(MessagePassing):
    """One-sided attention aggregation over a fixed edge direction."""

    def __init__(self, in_ch: int, out_ch: int, edge_dim: int, heads: int, dropout: float, negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.out_ch, self.dropout, self.negative_slope = heads, out_ch, dropout, negative_slope
        self.W_h = nn.Linear(in_ch, heads * out_ch, bias=False)
        self.W_e = nn.Linear(edge_dim, heads * out_ch, bias=False)
        self.a = nn.Parameter(torch.empty(1, heads, 3 * out_ch))
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr, num_nodes):
        Wx = self.W_h(x).view(-1, self.heads, self.out_ch)
        We = self.W_e(edge_attr).view(-1, self.heads, self.out_ch)
        out = self.propagate(edge_index, Wx=Wx, We=We, size=(num_nodes, num_nodes))
        return out.reshape(-1, self.heads * self.out_ch)

    def message(self, Wx_i, Wx_j, We, index):
        e = (torch.cat([Wx_i, Wx_j, We], dim=-1) * self.a).sum(dim=-1)
        alpha = softmax(F.leaky_relu(e, self.negative_slope), index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return Wx_j * alpha.unsqueeze(-1)


class RoleGATConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, edge_dim: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        per = max(out_ch // heads, 1)
        self.att_in = _DirectedAttention(in_ch, per, edge_dim, heads, dropout)
        self.att_out = _DirectedAttention(in_ch, per, edge_dim, heads, dropout)
        self.self_lin = nn.Linear(in_ch, out_ch)
        self.proj = nn.Linear(out_ch + 2 * heads * per, out_ch)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x, edge_index, edge_attr):
        n = x.size(0)
        rev = torch.stack([edge_index[1], edge_index[0]])
        h_in = self.att_in(x, edge_index, edge_attr, n)      # messages along j -> i
        h_out = self.att_out(x, rev, edge_attr, n)           # messages along i -> j (reversed)
        h = torch.cat([self.self_lin(x), h_in, h_out], dim=-1)
        return self.norm(self.proj(h))


class RoleGATEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, edge_dim: int = 2, n_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            self.layers.append(RoleGATConv(in_dim if l == 0 else hid_dim, hid_dim, edge_dim, heads, dropout))
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, x, edge_index, edge_attr):
        h = x
        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index, edge_attr)
            h = F.elu(h_new) if i < self.n_layers - 1 else h_new
        return h

    @staticmethod
    def reconstruction_loss(z, edge_index, num_nodes):
        """Kept for API compatibility; the role encoder is trained with feature reconstruction."""
        from .attri_gat import AttrGATEncoder
        return AttrGATEncoder.reconstruction_loss(z, edge_index, num_nodes)
