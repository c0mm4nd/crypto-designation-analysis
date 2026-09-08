"""
Scenario-aware node features (v3) for wartime cryptocurrency transfer networks.

The v1 feature set (12 dimensions) describes only degree, volume and lifetime.
v3 adds behavioural signatures that distinguish collection wallets, pass-through
relays, exchange hot wallets and retail donors in stablecoin transfer data:

  volume / count      : log in/out transfer counts and values, mean and median amounts,
                        maximum amount, share of small (<100 USDT) transfers, share of
                        round amounts (multiples of 100 USDT), Gini of incoming amounts
  counterparties      : unique in/out counterparties, repeat-interaction ratios,
                        reciprocity (share of counterparties on both sides)
  balance / flow      : net-flow ratio (in-out)/(in+out), forwarded share of inflow,
                        median time from receipt to next outgoing transfer (pass-through
                        latency), share of outflow within 24 h of an inflow
  temporal            : lifetime, active days, mean and coefficient of variation of
                        inter-event times (burstiness), maximum share of transfers in a
                        single day, hour-of-day entropy, weekend share
  neighbourhood       : mean and log-sum aggregates of the above over in- and out-
                        neighbourhoods (ReFeX-style, one level), so that the structural
                        context enters the encoder input as well as the message passing

All features are computed from the raw transfer rows with vectorised pandas/numpy
operations and standardised. The function returns a PyG Data object with the same
edge construction as model.features.build_graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

FEATURE_NAMES: list[str] = []


def _gini(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    """Gini coefficient of `values` within each group (0 for groups with <2 values)."""
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    cnt = np.bincount(g, minlength=n)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    rank = np.arange(len(v)) - np.repeat(starts, cnt) + 1
    s = np.bincount(g, weights=v, minlength=n)
    sr = np.bincount(g, weights=v * rank, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        gini = (2 * sr) / (cnt * s) - (cnt + 1) / cnt
    gini[(cnt < 2) | (s <= 0)] = 0.0
    return np.nan_to_num(gini)


def _entropy_by_group(bins: np.ndarray, groups: np.ndarray, n: int, nbins: int, chunk: int = 4_000_000) -> np.ndarray:
    """Normalised entropy of `bins` within each group, accumulated in int32 chunks so that
    the count matrix stays small for graphs with tens of millions of nodes."""
    key = groups.astype(np.int64) * nbins + bins
    counts = np.bincount(key, minlength=n * nbins).astype(np.int32)
    del key
    h = np.zeros(n, dtype=np.float64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        c = counts[i * nbins:j * nbins].reshape(j - i, nbins).astype(np.float64)
        tot = c.sum(1, keepdims=True)
        pr = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h[i:j] = -(pr * np.log(np.where(pr > 0, pr, 1))).sum(1)
        del c, tot, pr
    return h / np.log(nbins)


def build_graph_v3(df: pd.DataFrame, seed_set: set | None = None, p_scale: float = 0.1, w_ref: float = 1e8, ts_unit: str = "ms"):
    df = df[df["value"] > 0].copy()
    df["value"] = df["value"].astype(float)
    ts = df["block_timestamp"].astype(float).to_numpy()
    if ts_unit == "ms":
        ts = ts / 1000.0
    all_nodes = pd.unique(pd.concat([df["from"], df["to"]]))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = list(all_nodes)
    N = len(all_nodes)
    src = df["from"].map(node_to_idx).to_numpy(np.int64)
    dst = df["to"].map(node_to_idx).to_numpy(np.int64)
    val = df["value"].to_numpy(float)
    del df
    y_idx = [node_to_idx[a] for a in (seed_set or set()) if a in node_to_idx]
    data = build_graph_v3_from_arrays(src, dst, val, ts, N, y_idx=y_idx, p_scale=p_scale, w_ref=w_ref)
    return data, node_to_idx, idx_to_node, None


def build_graph_v3_from_arrays(src, dst, val, ts, N, y_idx=None, p_scale: float = 0.1, w_ref: float = 1e8):
    """Core v3 feature computation from integer node ids and transfer arrays (seconds).

    Written to keep the peak memory near the size of the transfer arrays themselves: ids are
    int32, timestamps are int32 seconds relative to the first transfer, per-node aggregates
    are accumulated with bincount over the in- and out-side separately, and every large
    temporary is released as soon as it has been consumed. This is what makes the complete
    Ukraine Ethereum network (85 M transfers, 23.9 M addresses) tractable.
    """
    global FEATURE_NAMES
    src = np.asarray(src, dtype=np.int32); dst = np.asarray(dst, dtype=np.int32)
    val = np.asarray(val, dtype=np.float64)
    ts0 = float(np.min(ts)); ts1 = float(np.max(ts)) + 1.0
    tsec = (np.asarray(ts, dtype=np.float64) - ts0).astype(np.int64)  # seconds since first transfer
    del ts
    M = len(src)
    feats: dict[str, np.ndarray] = {}

    def bc(idx, w=None):
        return np.bincount(idx, weights=w, minlength=N)

    in_cnt = bc(dst).astype(np.float64); out_cnt = bc(src).astype(np.float64)
    in_val = bc(dst, val); out_val = bc(src, val)
    feats["log_in_cnt"] = np.log1p(in_cnt); feats["log_out_cnt"] = np.log1p(out_cnt)
    feats["log_in_val"] = np.log1p(in_val); feats["log_out_val"] = np.log1p(out_val)
    feats["log_mean_in"] = np.log1p(np.divide(in_val, in_cnt, out=np.zeros(N), where=in_cnt > 0))
    feats["log_mean_out"] = np.log1p(np.divide(out_val, out_cnt, out=np.zeros(N), where=out_cnt > 0))
    tot_cnt = in_cnt + out_cnt

    # amount shape: max, share of small and round amounts (both sides, from the original values)
    mx = np.zeros(N); np.maximum.at(mx, dst, val); np.maximum.at(mx, src, val)
    feats["log_max_amt"] = np.log1p(mx); del mx
    small = (val < 100).astype(np.float64)
    feats["share_small"] = np.divide(bc(dst, small) + bc(src, small), tot_cnt, out=np.zeros(N), where=tot_cnt > 0)
    del small
    rnd = ((np.abs(val / 100 - np.round(val / 100)) < 1e-9) & (val >= 100)).astype(np.float64)
    feats["share_round"] = np.divide(bc(dst, rnd) + bc(src, rnd), tot_cnt, out=np.zeros(N), where=tot_cnt > 0)
    del rnd
    feats["gini_in"] = _gini(val, dst.astype(np.int64), N)

    # median amount over both sides (float32 is enough for a median)
    node_all = np.concatenate([dst, src])
    val32 = np.concatenate([val, val]).astype(np.float32)
    order = np.lexsort((val32, node_all))
    v_sorted = val32[order]; del val32, order
    cnt_all = np.bincount(node_all, minlength=N)
    starts = np.concatenate([[0], np.cumsum(cnt_all)[:-1]])
    med_idx = np.minimum(starts + cnt_all // 2, len(v_sorted) - 1)
    med = np.zeros(N); has = cnt_all > 0
    med[has] = v_sorted[med_idx[has]]
    feats["log_median_amt"] = np.log1p(med)
    del v_sorted, med, med_idx, starts

    # counterparties
    pair_in = np.unique(dst.astype(np.int64) * N + src)
    uin = np.bincount((pair_in // N).astype(np.int64), minlength=N).astype(np.float64)
    pair_out = np.unique(src.astype(np.int64) * N + dst)
    uout = np.bincount((pair_out // N).astype(np.int64), minlength=N).astype(np.float64)
    feats["log_uniq_in"] = np.log1p(uin); feats["log_uniq_out"] = np.log1p(uout)
    feats["repeat_in"] = np.divide(in_cnt, uin, out=np.zeros(N), where=uin > 0) - 1
    feats["repeat_out"] = np.divide(out_cnt, uout, out=np.zeros(N), where=uout > 0) - 1
    both = np.intersect1d(pair_in, pair_out, assume_unique=True)
    del pair_in, pair_out
    recip = np.bincount((both // N).astype(np.int64), minlength=N).astype(np.float64); del both
    feats["reciprocity"] = np.divide(recip, np.maximum(uin + uout - recip, 1)); del recip, uin, uout

    feats["net_flow_ratio"] = np.divide(in_val - out_val, in_val + out_val, out=np.zeros(N), where=(in_val + out_val) > 0)
    feats["forward_share"] = np.clip(np.divide(out_val, in_val, out=np.zeros(N), where=in_val > 0), 0, 5)
    del in_val, out_val

    # event-ordered features: pass-through latency, fast forwarding, burstiness
    ev_node = node_all  # reuse: [dst..., src...]
    ev_ts = np.concatenate([tsec, tsec])
    ev_dir = np.concatenate([np.zeros(M, dtype=np.int8), np.ones(M, dtype=np.int8)])
    o = np.lexsort((ev_dir, ev_ts, ev_node))
    ev_node = ev_node[o]; ev_ts = ev_ts[o]; ev_dir = ev_dir[o]; del o
    is_in = ev_dir == 0
    idx = np.where(is_in, np.arange(len(ev_node), dtype=np.int64), 0)
    np.maximum.accumulate(idx, out=idx)
    node_start = np.concatenate([[0], np.cumsum(np.bincount(ev_node, minlength=N))[:-1]])[ev_node]
    valid = (idx >= node_start) & is_in[idx] & (~is_in)
    del node_start
    lag_days = np.where(valid, (ev_ts - ev_ts[idx]) / 86400.0, np.nan)
    del idx, valid
    has_lag = ~np.isnan(lag_days)
    sum_lag = np.bincount(ev_node[has_lag], weights=np.log1p(lag_days[has_lag]), minlength=N)
    n_lag = np.bincount(ev_node[has_lag], minlength=N).astype(np.float64)
    feats["log_pass_latency"] = np.divide(sum_lag, n_lag, out=np.full(N, np.log1p(365.0)), where=n_lag > 0)
    del sum_lag, n_lag
    fast = (has_lag & (lag_days <= 1)).astype(np.float64)
    feats["share_out_within_1d"] = np.divide(np.bincount(ev_node, weights=fast, minlength=N), np.maximum(out_cnt, 1))
    del fast, has_lag, lag_days, is_in, ev_dir
    dt = np.diff(ev_ts).astype(np.float64) / 86400.0
    same = ev_node[1:] == ev_node[:-1]
    nodes_dt = ev_node[1:][same]; dts = dt[same]; del dt, same
    n_dt = np.bincount(nodes_dt, minlength=N).astype(np.float64)
    mean_dt = np.divide(np.bincount(nodes_dt, weights=dts, minlength=N), n_dt, out=np.zeros(N), where=n_dt > 0)
    m2 = np.divide(np.bincount(nodes_dt, weights=dts ** 2, minlength=N), n_dt, out=np.zeros(N), where=n_dt > 0)
    del nodes_dt, dts, n_dt
    std_dt = np.sqrt(np.clip(m2 - mean_dt ** 2, 0, None)); del m2
    feats["log_mean_interevent"] = np.log1p(mean_dt)
    feats["burstiness"] = np.divide(std_dt - mean_dt, std_dt + mean_dt, out=np.zeros(N), where=(std_dt + mean_dt) > 0)
    del std_dt, mean_dt

    # calendar features
    first = np.full(N, np.iinfo(np.int64).max, dtype=np.int64); last = np.full(N, -1, dtype=np.int64)
    np.minimum.at(first, ev_node, ev_ts); np.maximum.at(last, ev_node, ev_ts)
    seen = last >= 0
    life = np.zeros(N); life[seen] = (last[seen] - first[seen]) / 86400.0
    feats["log_lifetime_days"] = np.log1p(life)
    ts0_int = int(np.floor(ts0))
    abs_day = (ev_ts + ts0_int) // 86400          # calendar day (UTC), not days since the first transfer
    day = abs_day - abs_day.min()
    dk = np.unique(ev_node.astype(np.int64) * 100000 + day)
    active_days = np.bincount((dk // 100000).astype(np.int64), minlength=N).astype(np.float64); del dk
    feats["log_active_days"] = np.log1p(active_days)
    feats["tx_per_active_day"] = np.log1p(np.divide(tot_cnt, active_days, out=np.zeros(N), where=active_days > 0))
    del active_days
    uk, ck = np.unique(ev_node.astype(np.int64) * 100000 + day, return_counts=True)
    maxday = np.zeros(N); np.maximum.at(maxday, (uk // 100000).astype(np.int64), ck.astype(np.float64)); del uk, ck
    feats["max_day_share"] = np.divide(maxday, tot_cnt, out=np.zeros(N), where=tot_cnt > 0); del maxday
    hour = (((ev_ts + ts0_int) % 86400) // 3600).astype(np.int64)   # hour of day (UTC)
    feats["hour_entropy"] = _entropy_by_group(hour, ev_node.astype(np.int64), N, 24); del hour
    weekend = (((abs_day + 4) % 7) >= 5).astype(np.float64); del day, abs_day
    feats["weekend_share"] = np.divide(np.bincount(ev_node, weights=weekend, minlength=N), tot_cnt, out=np.zeros(N), where=tot_cnt > 0)
    del weekend, ev_node, ev_ts, node_all
    feats["first_ts_norm"] = np.where(seen, first / (ts1 - ts0), 0.0)
    feats["last_ts_norm"] = np.where(seen, last / (ts1 - ts0), 0.0)
    del first, last, seen, in_cnt, out_cnt, tot_cnt

    names = list(feats.keys())
    X = np.empty((N, len(names)), dtype=np.float32)
    for i, k in enumerate(names):
        X[:, i] = np.nan_to_num(feats[k]).astype(np.float32)
        feats[k] = None
    feats.clear()
    X -= X.mean(0); X /= np.maximum(X.std(0), 1e-9)

    # edges: unique directed pairs with aggregated weight and mean time
    key = src.astype(np.int64) * N + dst
    del src, dst
    o = np.argsort(key, kind="stable")
    key_s = key[o]; del key
    ts_s = tsec[o].astype(np.float64); val_s = val[o]; del o, tsec, val
    uniq, start = np.unique(key_s, return_index=True); del key_s
    w = np.add.reduceat(val_s, start); del val_s
    cnt_e = np.diff(np.append(start, len(ts_s)))
    mean_t = (np.add.reduceat(ts_s, start) / cnt_e / (ts1 - ts0)).astype(np.float32); del ts_s, start, cnt_e
    e_src = (uniq // N).astype(np.int64); e_dst = (uniq % N).astype(np.int64); del uniq
    log_w = (np.log1p(w) / np.log1p(w.max() + 1e-6)).astype(np.float32)

    # one-level ReFeX aggregates over in- and out-neighbourhoods, written into a preallocated matrix
    D = X.shape[1]
    X_all = np.empty((N, 3 * D), dtype=np.float32)
    X_all[:, :D] = X
    A_out = csr_matrix((np.ones(len(e_src), dtype=np.float32), (e_src, e_dst)), shape=(N, N))
    for j, A in enumerate((A_out, A_out.T.tocsr())):
        d = np.maximum(np.asarray(A.sum(1)).ravel(), 1).astype(np.float32)[:, None]
        X_all[:, (j + 1) * D:(j + 2) * D] = (A @ X) / d
    del A_out
    FEATURE_NAMES = names + [f"out_nb_mean_{k}" for k in names] + [f"in_nb_mean_{k}" for k in names]

    edge_index = torch.from_numpy(np.stack([e_src, e_dst]))
    edge_attr = torch.from_numpy(np.stack([log_w, np.clip(mean_t, 0, 1)], 1))
    y = torch.zeros(N, dtype=torch.long)
    for i in (y_idx or []):
        y[i] = 1
    data = Data(x=torch.from_numpy(X_all), edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.prop_prob = torch.from_numpy(np.clip(p_scale * np.log1p(w) / np.log1p(w_ref), 1e-6, 1.0).astype(np.float32))
    data.num_nodes = N
    data.x_raw = torch.from_numpy(X)
    return data


def build_graph_v3_large(raw_csv: str, nodes_agg_csv: str, seed_set: set | None = None, p_scale: float = 0.1, w_ref: float = 1e8,
                         ts_unit: str = "s", addr_width: int = 42, lower: bool = True, chunksize: int = 5_000_000,
                         value_cap: float = 1e8):
    """v3 features for graphs whose transfer rows do not fit as pandas strings.

    Node order is taken from the aggregate node table; the raw transfer file is streamed in
    chunks and mapped to integer ids, so the peak memory is the integer/float arrays rather
    than tens of millions of Python strings. Feature definitions are identical to build_graph_v3.
    """
    from .features import _read_cols_chunked
    na, _ = _read_cols_chunked(nodes_agg_csv, ["address"], [], addr_width, lower=lower)
    addr = na["address"]; del na
    N = len(addr)
    order = np.argsort(addr, kind="stable").astype(np.int64)
    sorted_addr = addr[order]

    def to_idx(col):
        pos = np.searchsorted(sorted_addr, col)
        bad = (pos >= N) | (sorted_addr[np.minimum(pos, N - 1)] != col)
        assert not bad.any(), f"{int(bad.sum())} transfer endpoints missing from the node table"
        return order[pos]

    src_p, dst_p, val_p, ts_p = [], [], [], []
    scale = 1000.0 if ts_unit == "ms" else 1.0
    for chunk in pd.read_csv(raw_csv, usecols=["from", "to", "value", "block_timestamp"], chunksize=chunksize,
                             dtype={"from": str, "to": str}):
        v = chunk["value"].to_numpy(np.float64)
        keep = (v > 0) & (v <= value_cap)
        f = chunk["from"]; t = chunk["to"]
        if lower:
            f = f.str.lower(); t = t.str.lower()
        src_p.append(to_idx(f.to_numpy(dtype=f"S{addr_width}")[keep]).astype(np.int32))
        dst_p.append(to_idx(t.to_numpy(dtype=f"S{addr_width}")[keep]).astype(np.int32))
        val_p.append(v[keep])  # keep float64: the round-amount test needs exact values
        ts_p.append(chunk["block_timestamp"].to_numpy(np.float64)[keep] / scale)
        del chunk, f, t, v, keep
    src = np.concatenate(src_p).astype(np.int64); del src_p
    dst = np.concatenate(dst_p).astype(np.int64); del dst_p
    val = np.concatenate(val_p).astype(np.float64); del val_p
    ts = np.concatenate(ts_p); del ts_p
    y_idx = []
    for a in (seed_set or set()):
        b = (a.lower() if lower else a).encode()
        pos = int(np.searchsorted(sorted_addr, b))
        if pos < N and sorted_addr[pos] == b:
            y_idx.append(int(order[pos]))
    idx_to_node = [a.decode() for a in addr]
    del order, sorted_addr, addr
    data = build_graph_v3_from_arrays(src, dst, val, ts, N, y_idx=y_idx, p_scale=p_scale, w_ref=w_ref)
    return data, None, idx_to_node, None
