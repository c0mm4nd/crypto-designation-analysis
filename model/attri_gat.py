"""
Attri-GAT Encoder (Section 5.2 of WCFRM paper).

Attention formula (per paper):
  α_ij = softmax_j( LeakyReLU( a^T [ W_h z_i || W_h z_j || W_e [A_ij, T_ij] ] ) )
  z_i^(l) = σ( Σ_{j∈N(i)} α_ij * W_h * z_j^(l-1) )

Multi-head: outputs are concatenated (all layers except last, which averages heads).

Reconstruction loss:
  L_AGAT = Σ_i || z_i^(L) - mean_{j∈N(i)} z_j^(L) ||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops


class AttrGATConv(MessagePassing):
    """Single Attri-GAT layer with edge-attribute-augmented attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: int = 2, heads: int = 4,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 concat: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.heads    = heads
        self.concat   = concat
        self.out_ch   = out_channels
        self.dropout  = dropout
        self.negative_slope = negative_slope

        # W_h: node feature transform
        self.W_h = nn.Linear(in_channels, heads * out_channels, bias=False)
        # W_e: edge attribute transform
        self.W_e = nn.Linear(edge_dim, heads * out_channels, bias=False)
        # Attention vector a  (size = 3 * heads * out_channels for concat of z_i, z_j, e_ij)
        self.a   = nn.Parameter(torch.empty(1, heads, 3 * out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat
                                             else out_channels))

        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_channels); edge_attr: (E, edge_dim)
        Wx = self.W_h(x).view(-1, self.heads, self.out_ch)   # (N, H, C)
        We = self.W_e(edge_attr).view(-1, self.heads, self.out_ch)  # (E, H, C)
        return self.propagate(edge_index, Wx=Wx, We=We, size=(x.size(0), x.size(0)))

    def message(self, Wx_i, Wx_j, We, index):
        # Wx_i, Wx_j: (E, H, C); We: (E, H, C)
        cat = torch.cat([Wx_i, Wx_j, We], dim=-1)      # (E, H, 3C)
        e   = (cat * self.a).sum(dim=-1)                # (E, H)
        e   = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, index)                        # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (Wx_j * alpha.unsqueeze(-1))              # (E, H, C)

    def update(self, aggr_out):
        # aggr_out: (N, H, C)
        if self.concat:
            out = aggr_out.view(-1, self.heads * self.out_ch)
        else:
            out = aggr_out.mean(dim=1)
        return out + self.bias


class AttrGATEncoder(nn.Module):
    """
    L-layer Attri-GAT encoder.

    Parameters
    ----------
    in_dim   : input node feature dimension
    hid_dim  : hidden / output embedding dimension
    edge_dim : edge feature dimension (default 2: log_weight, norm_ts)
    n_layers : number of GAT layers
    heads    : number of attention heads (last layer averages heads)
    dropout  : dropout on attention coefficients
    """

    def __init__(self, in_dim: int, hid_dim: int, edge_dim: int = 2,
                 n_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            _in   = in_dim if l == 0 else heads * hid_dim
            _out  = hid_dim
            _cat  = (l < n_layers - 1)  # concat all but last layer
            self.layers.append(
                AttrGATConv(_in, _out, edge_dim=edge_dim,
                             heads=heads, dropout=dropout, concat=_cat)
            )
        self.hid_dim   = hid_dim
        self.n_layers  = n_layers

    def forward(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < self.n_layers - 1:
                x = F.elu(x)
        return x   # (N, hid_dim)

    @staticmethod
    def reconstruction_loss(z, edge_index, num_nodes: int) -> torch.Tensor:
        """
        L_AGAT = Σ_i || z_i - mean_{j∈N(i)} z_j ||^2
        Uses destination edges (edge_index[1] = target i, edge_index[0] = source j).
        """
        src, dst = edge_index[0], edge_index[1]
        # mean of source embeddings for each target
        neighbor_mean = torch.zeros_like(z)
        cnt           = torch.zeros(num_nodes, 1, device=z.device)
        neighbor_mean.scatter_add_(0, dst.unsqueeze(1).expand(-1, z.size(1)), z[src])
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=z.device))
        cnt     = cnt.clamp(min=1.0)
        nb_mean = neighbor_mean / cnt
        loss    = ((z - nb_mean) ** 2).sum(dim=1).mean()
        return loss
