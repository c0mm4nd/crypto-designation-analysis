"""
Feature extraction: build node feature matrix and PyG Data object from edge DataFrame.

Node features (12-dim after normalization):
  log(in_degree+1), log(out_degree+1),
  log(in_weight+1), log(out_weight+1),
  log(in_count+1),  log(out_count+1),
  log(mean_in_amount+1), log(mean_out_amount+1),
  first_ts_norm, last_ts_norm, active_duration_norm,
  log(tx_frequency+1)

Edge features (2-dim, per aggregated directed edge):
  log_weight_norm  (log amount / log max_amount)
  mean_ts_norm     ((mean_timestamp - ts_min) / (ts_max - ts_min))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def build_graph(
    df: pd.DataFrame,
    seed_set: set | None = None,
    p_scale: float = 0.1,
    w_ref: float = 1e8,
) -> tuple[Data, dict, list, StandardScaler]:
    """
    Parameters
    ----------
    df       : DataFrame with columns [from, to, value, block_timestamp]
    seed_set : set of seed address strings for ground-truth labels

    Returns
    -------
    data         : PyG Data object
    node_to_idx  : {address: int}
    idx_to_node  : [address]
    scaler       : fitted StandardScaler (for later inverse-transform)
    """
    df = df[df["value"] > 0].copy()
    df["value"]           = df["value"].astype(float)
    df["block_timestamp"] = df["block_timestamp"].astype(float)

    # ── Node index ────────────────────────────────────────────────────────
    all_nodes   = pd.unique(pd.concat([df["from"], df["to"]]))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = list(all_nodes)
    N           = len(all_nodes)

    src_idx = df["from"].map(node_to_idx).values.astype(int)
    dst_idx = df["to"].map(node_to_idx).values.astype(int)
    vals    = df["value"].values
    ts      = df["block_timestamp"].values

    # ── Node features (vectorised) ────────────────────────────────────────
    in_deg  = np.bincount(dst_idx, minlength=N).astype(float)
    out_deg = np.bincount(src_idx, minlength=N).astype(float)
    in_w    = np.bincount(dst_idx, weights=vals, minlength=N)
    out_w   = np.bincount(src_idx, weights=vals, minlength=N)
    in_cnt  = in_deg.copy()   # same as in_deg for raw transactions
    out_cnt = out_deg.copy()

    mean_in_a  = np.divide(in_w,  in_cnt,  where=in_cnt>0,  out=np.zeros(N))
    mean_out_a = np.divide(out_w, out_cnt, where=out_cnt>0, out=np.zeros(N))

    # Temporal: first / last timestamp per node
    ts_min = ts.min(); ts_max = ts.max() + 1.0
    ts_norm = (ts - ts_min) / (ts_max - ts_min)

    first_ts = np.ones(N)
    last_ts  = np.zeros(N)
    np.minimum.at(first_ts, dst_idx, ts_norm)
    np.minimum.at(first_ts, src_idx, ts_norm)
    np.maximum.at(last_ts,  dst_idx, ts_norm)
    np.maximum.at(last_ts,  src_idx, ts_norm)

    actv_dur = np.maximum(last_ts - first_ts, 0.0)
    tx_total = in_cnt + out_cnt
    tx_freq  = np.divide(tx_total, np.maximum(actv_dur, 1e-6))

    def sl(x):
        return np.log1p(x)

    X = np.stack([
        sl(in_deg),    sl(out_deg),
        sl(in_w),      sl(out_w),
        sl(in_cnt),    sl(out_cnt),
        sl(mean_in_a), sl(mean_out_a),
        first_ts, last_ts, actv_dur,
        sl(tx_freq),
    ], axis=1).astype(np.float32)   # (N, 12)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)

    # ── Aggregate multi-edges → single directed edge with features ────────
    # Key: (src_idx, dst_idx); aggregate weight and mean timestamp
    edge_key = src_idx * N + dst_idx   # unique int key per directed pair
    edge_df  = pd.DataFrame({
        "key":   edge_key,
        "src":   src_idx,
        "dst":   dst_idx,
        "val":   vals,
        "ts_n":  ts_norm,
    })
    agg = edge_df.groupby("key").agg(
        src    = ("src", "first"),
        dst    = ("dst", "first"),
        w      = ("val", "sum"),
        cnt    = ("val", "count"),
        mean_t = ("ts_n", "mean"),
    ).reset_index(drop=True)

    w_max  = agg["w"].max() + 1e-6
    log_w  = (np.log1p(agg["w"].values) / np.log1p(w_max)).astype(np.float32)
    mean_t = agg["mean_t"].values.astype(np.float32)

    edge_index = torch.tensor(
        np.stack([agg["src"].values, agg["dst"].values], axis=0), dtype=torch.long
    )
    edge_attr  = torch.tensor(
        np.stack([log_w, mean_t], axis=1), dtype=torch.float
    )
    x = torch.tensor(X, dtype=torch.float)

    # ── Labels ────────────────────────────────────────────────────────────
    y = torch.zeros(N, dtype=torch.long)
    if seed_set:
        for addr, idx in node_to_idx.items():
            if addr in seed_set:
                y[idx] = 1

    # Propagation probabilities for the IC simulation. v1 normalised by the data
    # maximum, which on the primary network was an integer-overflow artefact
    # (1.2e71); v2 uses an explicit reference scale so that a transfer of w_ref
    # USDT activates with probability p_scale and a 1 USDT transfer with
    # p_scale * log(2)/log(1+w_ref).
    data           = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    prop = p_scale * np.log1p(agg["w"].values) / np.log1p(w_ref)
    data.prop_prob = torch.tensor(np.clip(prop, 1e-6, 1.0), dtype=torch.float)
    data.num_nodes = N

    return data, node_to_idx, idx_to_node, scaler


def _read_cols_chunked(path, addr_cols, num_cols, width, chunksize=4_000_000, lower=True):
    """Read a large CSV into fixed-width byte arrays (addresses) and float32 arrays (numbers).

    Pandas keeps address columns as Python string objects (~90 bytes each), which does not
    fit for the complete Ukraine Ethereum network (47 M address occurrences). Converting each
    chunk to an S<width> byte array immediately keeps the whole table under a few GB.
    """
    addr_parts = {c: [] for c in addr_cols}
    num_parts = {c: [] for c in num_cols}
    for chunk in pd.read_csv(path, usecols=addr_cols + num_cols, chunksize=chunksize,
                             dtype={c: str for c in addr_cols}):
        for c in addr_cols:
            col = chunk[c]
            if lower:
                col = col.str.lower()
            addr_parts[c].append(col.to_numpy(dtype=f"S{width}"))
        for c in num_cols:
            num_parts[c].append(chunk[c].to_numpy(dtype=np.float64))
        del chunk
    return ({c: np.concatenate(v) for c, v in addr_parts.items()},
            {c: np.concatenate(v) for c, v in num_parts.items()})


def build_graph_from_agg(edges_csv: str, nodes_csv: str, seed_set: set | None = None, p_scale: float = 0.1,
                         w_ref: float = 1e8, ts_unit: str = "s", addr_width: int = 42, return_addresses: bool = True,
                         lower: bool = True):
    """Build the same v1 graph as build_graph from pre-aggregated tables.

    edges_csv columns: from, to, w (sum of value), cnt, mean_ts
    nodes_csv columns: address, in_w, out_w, in_cnt, out_cnt, first_ts, last_ts
    Node order is the order of the node table. Used for networks whose raw transfer rows do
    not fit in memory (complete Ukraine Ethereum two-hop network, 85 M transfers).
    """
    na, nn = _read_cols_chunked(nodes_csv, ["address"], ["in_w", "out_w", "in_cnt", "out_cnt", "first_ts", "last_ts"], addr_width, lower=lower)
    addr = na["address"]; del na
    N = len(addr)
    order = np.argsort(addr, kind="stable")
    sorted_addr = addr[order]
    ea_, en_ = _read_cols_chunked(edges_csv, ["from", "to"], ["w", "cnt", "mean_ts"], addr_width, lower=lower)

    def to_idx(col):
        pos = np.searchsorted(sorted_addr, col)
        bad = (pos >= N) | (sorted_addr[np.minimum(pos, N - 1)] != col)
        assert not bad.any(), f"{int(bad.sum())} edge endpoints missing from the node table"
        return order[pos].astype(np.int64)

    src = to_idx(ea_["from"]); dst = to_idx(ea_["to"])
    del ea_
    w = en_["w"]; mean_ts = en_["mean_ts"]; del en_
    in_cnt, out_cnt = nn["in_cnt"], nn["out_cnt"]
    in_w, out_w = nn["in_w"], nn["out_w"]
    scale = 1000.0 if ts_unit == "ms" else 1.0
    first_ts = nn["first_ts"] / scale; last_ts = nn["last_ts"] / scale
    del nn
    ts_min = float(np.nanmin(first_ts)); ts_max = float(np.nanmax(last_ts)) + 1.0
    first_n = np.nan_to_num((first_ts - ts_min) / (ts_max - ts_min), nan=1.0)
    last_n = np.nan_to_num((last_ts - ts_min) / (ts_max - ts_min), nan=0.0)
    del first_ts, last_ts
    actv = np.maximum(last_n - first_n, 0.0)
    tx_total = in_cnt + out_cnt
    X = np.stack([
        np.log1p(in_cnt), np.log1p(out_cnt), np.log1p(in_w), np.log1p(out_w), np.log1p(in_cnt), np.log1p(out_cnt),
        np.log1p(np.divide(in_w, in_cnt, out=np.zeros(N), where=in_cnt > 0)),
        np.log1p(np.divide(out_w, out_cnt, out=np.zeros(N), where=out_cnt > 0)),
        first_n, last_n, actv, np.log1p(np.divide(tx_total, np.maximum(actv, 1e-6))),
    ], axis=1).astype(np.float32)
    del in_w, out_w, tx_total, first_n, last_n, actv
    X -= X.mean(0); X /= np.maximum(X.std(0), 1e-9)
    mean_t = np.clip((mean_ts / scale - ts_min) / (ts_max - ts_min), 0, 1).astype(np.float32)
    del mean_ts
    log_w = (np.log1p(w) / np.log1p(w.max() + 1e-6)).astype(np.float32)
    edge_index = torch.from_numpy(np.stack([src, dst]))
    edge_attr = torch.from_numpy(np.stack([log_w, mean_t], 1))
    y = torch.zeros(N, dtype=torch.long)
    if seed_set:
        for a in seed_set:
            b = (a.lower() if lower else a).encode()
            pos = int(np.searchsorted(sorted_addr, b))
            if pos < N and sorted_addr[pos] == b:
                y[int(order[pos])] = 1
    data = Data(x=torch.from_numpy(X), edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.prop_prob = torch.from_numpy(np.clip(p_scale * np.log1p(w) / np.log1p(w_ref), 1e-6, 1.0).astype(np.float32))
    data.num_nodes = N
    data.in_cnt = in_cnt; data.out_cnt = out_cnt
    del order, sorted_addr, w
    idx_to_node = [a.decode() for a in addr] if return_addresses else None
    node_to_idx = None
    return data, node_to_idx, idx_to_node, None


def load_agg_graph_arrays(edges_csv: str, nodes_csv: str, addr_width: int = 42, lower: bool = True):
    """Node list, unique directed edge arrays and transfer-count degrees from aggregate tables."""
    data, _, idx_to_node, _ = build_graph_from_agg(edges_csv, nodes_csv, ts_unit="s", addr_width=addr_width, lower=lower)
    ei = data.edge_index.numpy()
    return idx_to_node, data.num_nodes, ei[0], ei[1], data.in_cnt, data.out_cnt
