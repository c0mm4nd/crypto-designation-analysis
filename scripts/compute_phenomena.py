#!/usr/bin/env python3
"""Phenomenon-level analyses used by the main text (event study, persistence, donations).

Outputs phenomena.json. All USDT values above VALUE_CAP are dropped as uint-overflow
artefacts of the transfer export.

Analyses
  A. NBCTF seizure orders: flows through designated addresses by order; timing of designation
     relative to the last observed activity; weekly event study around the order signing date
     (aggregate and per-address normalised) with a placebo built from non-designated hop-1
     addresses assigned pseudo-event dates drawn from the real order dates.
  B. Persistence of the counterparty layer after designation.
  C. Concentration of designated-address volume over counterparties.
  D. Aid for Ukraine donations on TRON and Ethereum: daily series, surge, concentration.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import find, out_path, exists as _has  # noqa: E402
sys.path.insert(0, str(ROOT))
from labels.classify import classify  # noqa: E402

VALUE_CAP = 1e8
DATA_END = pd.Timestamp("2025-01-01")
W = 26  # weeks either side of the event
MIN_POST_DAYS = 90
UKR_TRON = "TEFccmfQ38cZS1DTZVhsxKVDckA8Y6VfCy"
UKR_ETH = "0x165cd37b4c644c2921454429e7f9358d18a45e14"
INVASION = pd.Timestamp("2022-02-24")


B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def tron_hex_to_b58(hex40: str) -> str:
    import hashlib
    payload = bytes.fromhex("41" + hex40)
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(payload + chk, "big")
    out = ""
    while n > 0:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return out


def load_complete_designated_transfers(truncate: bool = True) -> pd.DataFrame | None:
    """Complete USDT TRC-20 transfer histories of the designated addresses (ClickHouse export).

    The event study needs a fixed observation window, so by default the history is cut at
    DATA_END. The enforcement analysis must not be: many addresses were frozen after that
    date, and computing a balance at freeze from a history that stops before the freeze
    would understate it. Pass truncate=False for those statistics.
    """
    # The 490-address export is the one to use: an earlier export listed only the 389
    # addresses the two-hop crawl had reached, so the other 101 appeared in it only through
    # transfers with one of those, and their balances and last-transfer dates were partial.
    path = find("ch_data/designated_usdt_transfers_complete_490.csv", required=False)
    if not path.exists():
        path = find("ch_data/designated_usdt_transfers_complete.csv", required=False)
    if not path.exists():
        return None
    raw = pd.read_csv(path, usecols=["blockTimestamp", "transactionHash", "logIndex", "from_hex", "to_hex", "value"])
    raw = raw.drop_duplicates(subset=["transactionHash", "logIndex"])
    raw = raw[(raw["value"] > 0) & (raw["value"] <= VALUE_CAP)]
    cache: dict[str, str] = {}
    def conv(h):
        if h not in cache:
            cache[h] = tron_hex_to_b58(h)
        return cache[h]
    df = pd.DataFrame({"from": raw["from_hex"].map(conv), "to": raw["to_hex"].map(conv), "value": raw["value"].astype(float), "t": pd.to_datetime(raw["blockTimestamp"], unit="ms")})
    return (df[df["t"] < DATA_END] if truncate else df).reset_index(drop=True)


def tron_b58_to_hex(addr: str) -> str:
    n = 0
    for c in addr:
        n = n * 58 + B58.index(c)
    return n.to_bytes(25, "big").hex()[2:42]


def tether_enforcement(df: pd.DataFrame, seeds: pd.DataFrame, act: pd.DataFrame, signed: dict,
                       txn: pd.DataFrame | None = None) -> dict:
    """Match designated addresses against Tether TRC-20 blacklist events (from ClickHouse export)."""
    path = find("ch_data/tron_usdt_blacklist_added.csv", required=False)
    if not path.exists():
        return {}
    bl = pd.read_csv(path)
    bl["t"] = pd.to_datetime(bl["blockTimestamp"], unit="ms")
    first_bl = bl.groupby(bl["addr_hex"].str.lower())["t"].min()
    des = pd.read_csv(find("ch_data/tron_usdt_blackfunds_destroyed.csv"))
    des["amount"] = des["data"].apply(lambda x: int(str(x)[-64:], 16) / 1e6 if isinstance(x, str) and len(str(x)) >= 64 else 0.0)
    destroyed = des.groupby(des["addr_hex"].str.lower())["amount"].sum()
    rows = []
    for a, d in signed.items():
        h = tron_b58_to_hex(a)
        fr = first_bl.get(h, pd.NaT)
        f = act[act["addr"] == a]
        inflow = f[f["to"] == a]["value"]
        outflow = f[f["from"] == a]["value"]
        row = {"address": a, "signed": d, "frozen_at": fr, "lifetime_in": float(inflow.sum()), "lifetime_out": float(outflow.sum()), "n_transfers": int(len(f)),
               "destroyed_usdt": float(destroyed.get(h, 0.0))}
        if pd.notna(fr):
            row["days_signed_to_frozen"] = (fr - d).days
            pre = f[f["t"] < fr]
            row["balance_at_freeze"] = float(pre[pre["to"] == a]["value"].sum() - pre[pre["from"] == a]["value"].sum())
            last30 = pre[pre["t"] >= fr - pd.Timedelta(days=30)]
            row["out_last30d"] = float(last30[last30["from"] == a]["value"].sum())
            row["in_last30d"] = float(last30[last30["to"] == a]["value"].sum())
            row["transfers_after_freeze"] = int((f["t"] > fr).sum())
            # the interval to the freeze must use the last transfer *before* it; a blacklisted
            # address can still receive, and counting those made the interval negative
            row["last_transfer_to_freeze_days"] = (fr - pre["t"].max()).days if len(pre) else None
        rows.append(row)
    e = pd.DataFrame(rows)
    fz = e[e["frozen_at"].notna()].copy()
    by_order = []
    for d, g in e.groupby("signed"):
        gf = g[g["frozen_at"].notna()]
        by_order.append({"signed": str(d.date()), "n": int(len(g)), "n_frozen": int(len(gf)), "median_days_signed_to_frozen": float(gf["days_signed_to_frozen"].median()) if len(gf) else None,
                         "n_frozen_before_signing": int((gf["days_signed_to_frozen"] < 0).sum()), "n_frozen_within_30d": int(((gf["days_signed_to_frozen"] >= 0) & (gf["days_signed_to_frozen"] <= 30)).sum()),
                         "median_days_last_transfer_to_freeze": float(gf["last_transfer_to_freeze_days"].median()) if len(gf) else None, "balance_at_freeze_usdt": float(gf["balance_at_freeze"].sum()) if len(gf) else 0.0,
                         "lifetime_volume_usdt": float((g["lifetime_in"] + g["lifetime_out"]).sum())})
    out = {"n_designated": int(len(e)), "n_frozen": int(len(fz)), "share_frozen": float(len(fz) / len(e)),
           "n_frozen_before_signing": int((fz["days_signed_to_frozen"] < 0).sum()), "n_frozen_within_30d": int(((fz["days_signed_to_frozen"] >= 0) & (fz["days_signed_to_frozen"] <= 30)).sum()),
           "n_frozen_after_30d": int((fz["days_signed_to_frozen"] > 30).sum()),
           "median_days_signed_to_frozen_all": float(fz["days_signed_to_frozen"].median()), "median_days_signed_to_frozen_after": float(fz.loc[fz["days_signed_to_frozen"] >= 0, "days_signed_to_frozen"].median()),
           "balance_at_freeze_total_usdt": float(fz["balance_at_freeze"].clip(lower=0).sum()), "lifetime_inflow_frozen_usdt": float(fz["lifetime_in"].sum()),
           "share_of_frozen_addresses_with_positive_balance": float((fz["balance_at_freeze"] > 1).mean()),
           "share_of_frozen_addresses_with_any_balance": float((fz["balance_at_freeze"] >= 1e-6).mean()),
           "median_balance_at_freeze_usdt": float(fz["balance_at_freeze"].clip(lower=0).median()),
           "out_last30d_total_usdt": float(fz["out_last30d"].sum()), "in_last30d_total_usdt": float(fz["in_last30d"].sum()),
           "share_frozen_addresses_with_transfer_after_freeze": float((fz["transfers_after_freeze"] > 0).mean()),
           "median_days_last_transfer_to_freeze": float(fz["last_transfer_to_freeze_days"].median()),
           "destroyed_usdt_total": float(e["destroyed_usdt"].sum()), "n_addresses_with_destroyed_funds": int((e["destroyed_usdt"] > 0).sum()),
           "days_signed_to_frozen": fz["days_signed_to_frozen"].astype(float).tolist(), "by_order": by_order,
           "days_last_transfer_to_freeze": [float(x) for x in fz["last_transfer_to_freeze_days"].dropna()],
           "per_address_lifetime_inflow_usdt": [float(x) for x in fz["lifetime_in"]],
           "per_address_balance_at_freeze_usdt": [float(x) for x in fz["balance_at_freeze"].clip(lower=0.0)],
           "n_never_frozen_2021_2022_orders": int(e[(e["signed"] < pd.Timestamp("2023-01-01")) & e["frozen_at"].isna()].shape[0]),
           "n_2021_2022_orders": int(e[e["signed"] < pd.Timestamp("2023-01-01")].shape[0]),
           "blacklist_total_events": int(len(bl)), "blacklist_unique_addresses": int(first_bl.shape[0]), "blacklist_first": str(bl["t"].min().date()), "blacklist_last": str(bl["t"].max().date())}
    # What happens after a freeze. The blacklist blocks the address from sending, but not
    # others from sending to it, so the two directions have to be reported separately.
    ev_fr = dict(zip(fz["address"], fz["frozen_at"]))
    tx = act.drop_duplicates(subset=["from", "to", "value", "t"]) if txn is None else txn
    post_in = tx[tx["to"].isin(ev_fr)].copy(); post_in["ev"] = post_in["to"].map(ev_fr)
    post_in = post_in[post_in["t"] > post_in["ev"]]
    post_out = tx[tx["from"].isin(ev_fr)].copy(); post_out["ev"] = post_out["from"].map(ev_fr)
    post_out = post_out[post_out["t"] > post_out["ev"]]
    out["after_freeze"] = {
        "n_frozen": int(len(ev_fr)),
        "n_addresses_receiving_after_freeze": int(post_in["to"].nunique()),
        "n_transfers_received_after_freeze": int(len(post_in)),
        "usdt_received_after_freeze": float(post_in["value"].sum()),
        "n_addresses_sending_after_freeze": int(post_out["from"].nunique()),
        "usdt_sent_after_freeze": float(post_out["value"].sum()),
    }

    fin = df[df["to"].isin(ev_fr)]
    fout = df[df["from"].isin(ev_fr)]
    vi, ci = weekly_series(fin, "to", ev_fr)
    vo, co = weekly_series(fout, "from", ev_fr)
    out["event_study_freeze"] = {"n_addresses": len(ev_fr), "weeks": list(range(-W, W + 1)), "inflow_usdt": vi.tolist(), "outflow_usdt": vo.tolist(), "transfers": (ci + co).tolist()}
    e.to_csv(out_path("ch_data/designated_tether_enforcement.csv"), index=False)
    return out


def load_edges(name: str, ts_unit: str = "ms", lower: bool = False) -> pd.DataFrame:
    df = pd.read_csv(find(name), usecols=["from", "to", "value", "block_timestamp"])
    df = df[(df["value"] > 0) & (df["value"] <= VALUE_CAP)].copy()
    df["t"] = pd.to_datetime(df["block_timestamp"], unit=ts_unit)
    if lower:
        df["from"] = df["from"].str.lower()
        df["to"] = df["to"].str.lower()
    return df.drop(columns="block_timestamp")


def israel_seed_dates() -> pd.DataFrame:
    ws = openpyxl.load_workbook(find("IsraelAddrs.xlsx")).active
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if len(r) < 5 or not r[3]:
            continue
        addr, cur = str(r[3]).strip(), str(r[4] or "").upper()
        if cur in ("USDT", "TRX") and addr.startswith("T"):
            order = str(r[0]).replace("​", "").replace("\xa0", " ").strip()
            order = order.split("ASO")[1].split(")")[0].replace("-", "").strip() if "ASO" in order else order
            rows.append({"address": addr, "order": "ASO " + order, "signed": pd.Timestamp(r[1]), "published": pd.Timestamp(r[2]) if r[2] else pd.NaT})
    return pd.DataFrame(rows).drop_duplicates("address")


def label_map(chain: str) -> dict[str, str]:
    if not _has("labels_cache/labels.db"):
        # Third-party entity labels are used under the providers' terms and are not
        # redistributed, so they are absent from the deposited archive. The label-derived
        # statistics are lower bounds and are reported as such; everything else is unaffected.
        print("labels_cache/labels.db not present: label-derived statistics will be empty")
        return {}
    con = sqlite3.connect(find("labels_cache/labels.db"))
    recs: dict[str, dict] = defaultdict(dict)
    for addr, source, raw in con.execute("select address, source, raw_json from labels where chain=?", (chain,)):
        try:
            recs[addr][source] = json.loads(raw) if raw else None
        except Exception:
            recs[addr][source] = None
    out = {}
    for addr, rec in recs.items():
        try:
            out[addr] = classify(rec.get("web3resear"), rec.get("slowmist"))[0]
        except Exception:
            out[addr] = "UNKNOWN"
    return out


def weekly_series(frame: pd.DataFrame, addr_col: str, event: dict) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate weekly volume and transfer counts relative to each address's event date."""
    f = frame[frame[addr_col].isin(event.keys())]
    rel = ((f["t"] - f[addr_col].map(event)).dt.days // 7).astype(int)
    m = (rel >= -W) & (rel <= W)
    vol = np.zeros(2 * W + 1)
    cnt = np.zeros(2 * W + 1)
    np.add.at(vol, (rel[m] + W).to_numpy(), f.loc[m, "value"].to_numpy())
    np.add.at(cnt, (rel[m] + W).to_numpy(), 1)
    return vol, cnt


def per_address_ratio(frame_in: pd.DataFrame, frame_out: pd.DataFrame, event: dict) -> dict:
    """Per-address post/pre activity: share of addresses active in each relative week, and
    the fraction of each address's window volume that occurs after the event."""
    rows = []
    for addr, d in event.items():
        f = pd.concat([frame_in[frame_in["to"] == addr], frame_out[frame_out["from"] == addr]])
        rel = ((f["t"] - d).dt.days // 7)
        m = (rel >= -W) & (rel <= W)
        f, rel = f[m], rel[m]
        if len(f) == 0:
            continue
        pre = f.loc[rel < 0, "value"].sum()
        post = f.loc[rel > 0, "value"].sum()
        active = np.zeros(2 * W + 1, dtype=bool)
        active[(rel + W).to_numpy()] = True
        rows.append({"addr": addr, "pre": pre, "post": post, "active": active, "n_pre": int((rel < 0).sum()), "n_post": int((rel > 0).sum())})
    if not rows:
        return {}
    active = np.mean([r["active"] for r in rows], axis=0)
    post_share = [r["post"] / (r["pre"] + r["post"]) for r in rows if (r["pre"] + r["post"]) > 0]
    return {"n": len(rows), "share_active_by_week": active.tolist(), "median_post_share_of_window_volume": float(np.median(post_share)),
            "share_with_any_post_activity": float(np.mean([r["n_post"] > 0 for r in rows])), "share_with_any_pre_activity": float(np.mean([r["n_pre"] > 0 for r in rows]))}


def main():
    out = {"value_cap_usdt": VALUE_CAP, "data_end": str(DATA_END.date()), "window_weeks": W}
    rng = np.random.default_rng(42)

    # ============================================================ A. NBCTF network
    df = load_edges("israel_tron_usdt_edges_2hop.csv")
    seeds = israel_seed_dates()
    nodes = set(df["from"]).union(df["to"])
    # The timing, event-study and enforcement analyses run on the designated addresses' own
    # complete on-chain histories, so there is no reason to condition them on the address
    # having been reached by the two-hop crawl. Three addresses are in the first set and not
    # the second. Structural analyses, which need the graph, are restricted separately.
    seeds_in_graph = seeds[seeds["address"].isin(nodes)].reset_index(drop=True)
    seed_set = set(seeds["address"])
    seed_set_in_graph = set(seeds_in_graph["address"])
    signed = dict(zip(seeds["address"], seeds["signed"]))
    complete = load_complete_designated_transfers()
    if complete is not None:
        crawl_seed_rows = int(df["to"].isin(seed_set).sum() + df["from"].isin(seed_set).sum())
        sin = complete[complete["to"].isin(seed_set)]
        sout = complete[complete["from"].isin(seed_set)]
        raw_seed_rows = int(pd.read_csv(find("israel_tron_usdt_edges_2hop.csv"), usecols=["from", "to"])
                            .isin(seed_set).sum().sum())
        out["designated_history_source"] = {"source": "complete on-chain history (ClickHouse full node export)",
                                            "transfers_complete": int(len(sin) + len(sout)),
                                            "transfers_in_crawl": crawl_seed_rows,
                                            "transfers_in_crawl_unfiltered": raw_seed_rows,
                                            "crawl_capture_filtered": crawl_seed_rows / (len(sin) + len(sout)),
                                            "crawl_capture_unfiltered": raw_seed_rows / (len(sin) + len(sout))}
        print(f"designated-address transfers: complete {len(sin)+len(sout):,} vs crawl {crawl_seed_rows:,}")
    else:
        sin = df[df["to"].isin(seed_set)]
        sout = df[df["from"].isin(seed_set)]
    act = pd.concat([sin.assign(addr=sin["to"]), sout.assign(addr=sout["from"])])
    fl = act.groupby("addr")["t"].agg(["min", "max", "count"]).join(act.groupby("addr")["value"].sum().rename("volume"))
    fl = seeds.set_index("address").join(fl)
    fl["days_last_to_signed"] = (fl["signed"] - fl["max"]).dt.days
    fl["days_signed_to_published"] = (fl["published"] - fl["signed"]).dt.days

    by_order = []
    for order, g in fl.groupby("order"):
        by_order.append({"order": order, "signed": str(g["signed"].iloc[0].date()), "published": str(g["published"].iloc[0].date()) if pd.notna(g["published"].iloc[0]) else None,
                         "n_addresses": int(len(g)), "volume_usdt": float(g["volume"].fillna(0).sum()), "transfers": int(g["count"].fillna(0).sum()),
                         "median_days_last_activity_to_signed": float(g["days_last_to_signed"].median()) if g["days_last_to_signed"].notna().any() else None,
                         "first_activity": str(g["min"].min().date()) if g["min"].notna().any() else None})
    by_order.sort(key=lambda r: r["signed"])
    out["nbctf_orders"] = by_order
    # The largest order is quoted in the text on both the in-window and the complete basis,
    # and separately by direction, so record all four rather than leaving the reader to
    # reconstruct which of them a single "volume" figure means.
    largest = seeds["order"].value_counts().idxmax()
    sep = seeds[seeds["order"] == largest]
    sep_set = set(sep["address"])
    src = load_complete_designated_transfers(truncate=False)
    if src is None:
        src = act
    win = src[src["t"] < DATA_END]
    def io(frame, addrs):
        return (float(frame.loc[frame["to"].isin(addrs), "value"].sum()),
                float(frame.loc[frame["from"].isin(addrs), "value"].sum()))
    win_in, win_out = io(win, sep_set)
    all_in, all_out = io(src, sep_set)
    out["nbctf_largest_order"] = {
        "order": str(largest), "signed": str(sep["signed"].iloc[0].date()), "n_named": int(len(sep)),
        "n_with_transfer_in_window": int(len(set(win.loc[win["to"].isin(sep_set), "to"]).union(
            win.loc[win["from"].isin(sep_set), "from"]))),
        "inflow_in_window_usdt": win_in, "outflow_in_window_usdt": win_out,
        "inflow_complete_usdt": all_in, "outflow_complete_usdt": all_out}
    out["nbctf_totals"] = {"n_seeds_in_graph": int(len(seed_set_in_graph)), "n_seeds_named": int(len(seed_set)), "transfers_in_graph": int(len(df)), "volume_in_graph_usdt": float(df["value"].sum()),
                           "seed_inflow_usdt": float(sin["value"].sum()), "seed_outflow_usdt": float(sout["value"].sum()),
                           "seed_in_transfers": int(len(sin)), "seed_out_transfers": int(len(sout)),
                           "first_seed_activity": str(fl["min"].min().date()), "n_hop1": int(len(set(sin["from"]).union(sout["to"]) - seed_set))}

    inwin = fl[(fl["signed"] <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)) & fl["max"].notna()]
    postwin = fl[fl["signed"] > DATA_END]
    # The complement of the in-window set has to be counted on the same basis as the set
    # itself, that is among addresses with an observed transfer. Counting every named address
    # instead mixes in addresses that never appear in the data and does not add up to 400.
    obs = fl[fl["max"].notna()]
    outside = obs[obs["signed"] > DATA_END - pd.Timedelta(days=MIN_POST_DAYS)]
    out["nbctf_timing"] = {
        "n_designated_in_window": int(len(inwin)), "n_designated_after_window": int(len(postwin)),
        "n_observed": int(len(obs)),
        "n_observed_outside_window": int(len(outside)),
        "n_observed_signed_after_window": int((outside["signed"] > DATA_END).sum()),
        "n_observed_signed_within_90d_of_window_end": int((outside["signed"] <= DATA_END).sum()),
        "median_days_last_activity_to_signed": float(inwin["days_last_to_signed"].median()),
        "iqr_days_last_activity_to_signed": inwin["days_last_to_signed"].quantile([0.25, 0.75]).tolist(),
        "share_dormant_30d_at_signing": float((inwin["days_last_to_signed"] > 30).mean()),
        "share_dormant_90d_at_signing": float((inwin["days_last_to_signed"] > 90).mean()),
        "share_active_after_signing": float((inwin["max"] > inwin["signed"]).mean()),
        "median_days_signed_to_published": float(fl["days_signed_to_published"].median()),
        "days_last_activity_to_signed": inwin["days_last_to_signed"].astype(float).tolist(),
        "days_last_activity_to_signed_by_order": {
            str(o): [float(x) for x in g["days_last_to_signed"].dropna()]
            for o, g in inwin.groupby("order")},
        "signed_by_order": {str(o): str(g["signed"].iloc[0].date()) for o, g in inwin.groupby("order")},
        "volume_after_signing_share": float(act[act["addr"].isin(inwin.index) & (act["t"] > act["addr"].map(signed))]["value"].sum() / act[act["addr"].isin(inwin.index)]["value"].sum()),
    }

    # event study, aggregate
    ev_in = dict(zip(inwin.index, inwin["signed"]))
    vol_in, cnt_in = weekly_series(sin, "to", ev_in)
    vol_out, cnt_out = weekly_series(sout, "from", ev_in)
    vol, cnt = vol_in + vol_out, cnt_in + cnt_out
    out["nbctf_event_study"] = {
        "n_addresses": len(ev_in), "weeks": list(range(-W, W + 1)), "volume_usdt": vol.tolist(), "transfers": cnt.tolist(),
        "inflow_usdt": vol_in.tolist(), "outflow_usdt": vol_out.tolist(),
        "pre_mean_weekly_volume": float(vol[:W].mean()), "post_mean_weekly_volume": float(vol[W + 1:].mean()),
        "post_weeks_7_26_mean_volume": float(vol[W + 7:].mean()), "weeks_0_4_mean_volume": float(vol[W:W + 5].mean()),
        "pre_mean_weekly_transfers": float(cnt[:W].mean()), "post_mean_weekly_transfers": float(cnt[W + 1:].mean()),
        "per_address": per_address_ratio(sin, sout, ev_in),
    }

    # How concentrated the aggregate series is. The event study pools eight orders, but the
    # designated addresses differ in size by four orders of magnitude, so the aggregate
    # weekly volume can be one operator's series. Report the share each order contributes to
    # the pre-event total, and the participation trend, which is a different quantity from
    # aggregate volume and behaves differently.
    ev_act = act[act["addr"].isin(ev_in)].copy()
    ev_act["order"] = ev_act["addr"].map(dict(zip(seeds["address"], seeds["order"])))
    ev_act["week"] = (ev_act["t"] - ev_act["addr"].map(signed)).dt.days // 7
    pre_w = ev_act[(ev_act["week"] >= -W) & (ev_act["week"] < 0)]
    by_ord = pre_w.groupby("order")["value"].sum().sort_values(ascending=False)
    fl_first = act.groupby("addr")["t"].min()
    covered = [a_ for a_ in ev_in
               if fl_first[a_] <= signed[a_] - pd.Timedelta(weeks=W) and DATA_END >= signed[a_] + pd.Timedelta(weeks=W)]
    cov_act = ev_act[ev_act["addr"].isin(covered)]
    act_share = (cov_act[(cov_act["week"] >= -W) & (cov_act["week"] <= W)]
                 .groupby("week")["addr"].nunique() / max(len(covered), 1) * 100)
    pre_share = act_share.reindex(range(-W, 0)).fillna(0.0)
    slope = float(np.polyfit(pre_share.index, pre_share.to_numpy(), 1)[0]) if len(pre_share) > 1 else 0.0
    out["nbctf_event_concentration"] = {
        "pre_event_volume_by_order_usdt": {str(k): float(v) for k, v in by_ord.items()},
        "pre_event_volume_share_by_order": {str(k): float(v / by_ord.sum()) for k, v in by_ord.items()},
        "largest_order": str(by_ord.index[0]),
        "largest_order_pre_event_share": float(by_ord.iloc[0] / by_ord.sum()),
        "largest_order_window_share": float(ev_act.groupby("order")["value"].sum().max()
                                            / ev_act["value"].sum()),
        "n_addresses_covering_full_window": int(len(covered)),
        "active_share_pct_by_week": {str(k): float(v) for k, v in act_share.items()},
        "pre_event_active_share_slope_pp_per_week": slope,
        "active_share_week_minus26": float(pre_share.iloc[0]), "active_share_week_minus1": float(pre_share.iloc[-1]),
    }
    print(f"event concentration: {by_ord.index[0]} is {100*by_ord.iloc[0]/by_ord.sum():.1f}% of pre-event volume; "
          f"active share {pre_share.iloc[0]:.1f}% -> {pre_share.iloc[-1]:.1f}% before the order "
          f"({slope:+.2f} pp/week, n={len(covered)})")

    # placebo: non-designated hop-1 addresses, pseudo-event dates drawn from the real signing dates
    hop1 = list(set(sin["from"]).union(sout["to"]) - seed_set)
    h1_frames = df[df["from"].isin(hop1) | df["to"].isin(hop1)]
    h1_act = pd.concat([h1_frames.assign(addr=h1_frames["from"]), h1_frames.assign(addr=h1_frames["to"])])
    h1_act = h1_act[h1_act["addr"].isin(hop1)]
    dates = inwin["signed"].to_numpy()
    cand = h1_act.groupby("addr")["t"].agg(["min", "max"])
    placebo = {}
    for addr, r in cand.sample(frac=1.0, random_state=42).iterrows():
        d = pd.Timestamp(rng.choice(dates))
        # require activity within the 26 weeks before the pseudo-event (as designated addresses have)
        if r["min"] <= d and r["max"] >= d - pd.Timedelta(weeks=W):
            placebo[addr] = d
        if len(placebo) >= 3000:
            break
    p_in = h1_act[h1_act["addr"].isin(placebo) & (h1_act["to"] == h1_act["addr"])]
    p_out = h1_act[h1_act["addr"].isin(placebo) & (h1_act["from"] == h1_act["addr"])]
    pv_in, pc_in = weekly_series(p_in, "addr", placebo)
    pv_out, pc_out = weekly_series(p_out, "addr", placebo)
    pv, pc = pv_in + pv_out, pc_in + pc_out
    out["placebo_event_study"] = {"n_addresses": len(placebo), "volume_usdt": pv.tolist(), "transfers": pc.tolist(),
                                  "pre_mean_weekly_volume": float(pv[:W].mean()), "post_mean_weekly_volume": float(pv[W + 1:].mean()),
                                  "post_weeks_7_26_mean_volume": float(pv[W + 7:].mean()),
                                  "pre_mean_weekly_transfers": float(pc[:W].mean()), "post_mean_weekly_transfers": float(pc[W + 1:].mean()),
                                  "per_address": per_address_ratio(p_in.rename(columns={"addr": "_a"}).assign(to=lambda x: x["_a"]).drop(columns="_a"),
                                                                   p_out.rename(columns={"addr": "_a"}).assign(**{"from": lambda x: x["_a"]}).drop(columns="_a"), placebo)}

    # The placebo is one random draw of 3,000 addresses and pseudo-event dates. Reporting a
    # single draw as if it were the placebo overstates its precision, so repeat the whole
    # construction under a set of seeds and report the spread.
    seed_ratios, seed_tx = [], []
    for sd in range(1, 13):
        r2 = np.random.default_rng(sd)
        pl = {}
        for addr, r in cand.sample(frac=1.0, random_state=sd).iterrows():
            dd = pd.Timestamp(r2.choice(dates))
            if r["min"] <= dd and r["max"] >= dd - pd.Timedelta(weeks=W):
                pl[addr] = dd
            if len(pl) >= 3000:
                break
        q_in = h1_act[h1_act["addr"].isin(pl) & (h1_act["to"] == h1_act["addr"])]
        q_out = h1_act[h1_act["addr"].isin(pl) & (h1_act["from"] == h1_act["addr"])]
        qv_in, qc_in = weekly_series(q_in, "addr", pl)
        qv_out, qc_out = weekly_series(q_out, "addr", pl)
        qv, qc = qv_in + qv_out, qc_in + qc_out
        if qv[:W].mean() > 0:
            seed_ratios.append(float(qv[W + 7:].mean() / qv[:W].mean()))
            seed_tx.append(float(qc[W + 7:].mean() / qc[:W].mean()))
    out["placebo_event_study"]["across_seeds"] = {
        "n_seeds": len(seed_ratios),
        "volume_ratio_min": min(seed_ratios), "volume_ratio_max": max(seed_ratios),
        "volume_ratio_median": float(np.median(seed_ratios)),
        "transfer_ratio_min": min(seed_tx), "transfer_ratio_max": max(seed_tx),
        "transfer_ratio_median": float(np.median(seed_tx))}
    print(f"placebo across {len(seed_ratios)} seeds: volume ratio "
          f"{min(seed_ratios):.2f}-{max(seed_ratios):.2f} (median {np.median(seed_ratios):.2f}), "
          f"transfers {min(seed_tx):.2f}-{max(seed_tx):.2f}")

    # ============================================================ B. counterparty persistence
    cps = pd.concat([sin[sin["to"].isin(ev_in)].rename(columns={"from": "cp", "to": "seed"}), sout[sout["from"].isin(ev_in)].rename(columns={"to": "cp", "from": "seed"})])
    cps = cps[~cps["cp"].isin(seed_set)]
    cps = cps.assign(d=cps["seed"].map(signed))
    # The question is whether a counterparty that dealt with a designated address *before* its
    # order kept transacting afterwards. Counting every counterparty, including those whose
    # only contact came after the order, answers a different question and answers it
    # circularly: such an address is active after the order by construction.
    n_cp_all = int(cps["cp"].nunique())
    cps_pre = cps[cps["t"] < cps["d"]]
    cp_first = cps_pre.groupby("cp")["d"].min()
    # A counterparty's own later activity is only observable if the crawl reached it; the
    # designated addresses have complete histories but their counterparties do not.
    cp_last = h1_act[h1_act["addr"].isin(cp_first.index)].groupby("addr")["t"].max()
    cp_df = pd.DataFrame({"d": cp_first, "last": cp_last}).dropna()
    n_cp_pre = int(len(cp_first))
    cp_df = cp_df[cp_df["d"] <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)]
    # volume-weighted persistence: share of counterparty volume (with anyone) that occurs after the order
    cp_vol = h1_act[h1_act["addr"].isin(cp_df.index)].assign(d=lambda x: x["addr"].map(cp_df["d"]))
    out["counterparty_persistence"] = {"n_counterparties": int(len(cp_df)),
                                       "n_counterparties_all": n_cp_all,
                                       "n_counterparties_before_order": n_cp_pre,
                                       "n_counterparties_only_after_order": n_cp_all - n_cp_pre,
                                       "n_dropped_not_in_crawl": n_cp_pre - int(len(cp_df)),
                                       "share_active_after_order": float((cp_df["last"] > cp_df["d"]).mean()),
                                       "share_active_90d_after_order": float((cp_df["last"] > cp_df["d"] + pd.Timedelta(days=90)).mean()),
                                       "volume_share_after_order": float(cp_vol.loc[cp_vol["t"] > cp_vol["d"], "value"].sum() / cp_vol["value"].sum()),
                                       "designated_share_active_after_order": out["nbctf_timing"]["share_active_after_signing"]}

    # ============================================================ C. concentration and labels
    lm = label_map("tron")
    cpv = pd.concat([sin.rename(columns={"from": "cp"})[["cp", "value"]], sout.rename(columns={"to": "cp"})[["cp", "value"]]])
    cpv = cpv[~cpv["cp"].isin(seed_set)].groupby("cp")["value"].sum().sort_values(ascending=False)
    cum = (cpv.cumsum() / cpv.sum()).to_numpy()
    seed_vol = act.groupby("addr")["value"].sum().sort_values(ascending=False)
    scum = (seed_vol.cumsum() / seed_vol.sum()).to_numpy()
    out["concentration"] = {"n_counterparties": int(len(cpv)), "top10_share": float(cum[9]), "top100_share": float(cum[99]), "top1pct_share": float(cum[len(cpv) // 100 - 1]),
                            "counterparty_lorenz": [float(cum[int(i)]) for i in np.linspace(0, len(cpv) - 1, 200)],
                            "counterparty_lorenz_log_rank": [int(i) + 1 for i in np.unique(np.geomspace(1, len(cpv), 400).astype(int)) - 1],
                            "counterparty_lorenz_log": [float(cum[int(i)]) for i in np.unique(np.geomspace(1, len(cpv), 400).astype(int)) - 1],
                            "n_counterparties_ranked": int(len(cpv)),
                            "n_designated_with_volume": int(len(seed_vol)), "designated_top10_share": float(scum[9]), "designated_top1pct_share": float(scum[max(len(seed_vol) // 100 - 1, 0)]),
                            "labelled_addresses": int(sum(1 for a in nodes if lm.get(a, "UNKNOWN") != "UNKNOWN")),
                            "seed_inflow_share_from_labelled_cex": float(sin.loc[sin["from"].map(lambda a: lm.get(a, "UNKNOWN")) == "CEX", "value"].sum() / sin["value"].sum()),
                            "seed_outflow_share_to_labelled_cex": float(sout.loc[sout["to"].map(lambda a: lm.get(a, "UNKNOWN")) == "CEX", "value"].sum() / sout["value"].sum()),
                            "seeds_with_labelled_cex_counterparty": int(len(set(sin.loc[sin["from"].map(lambda a: lm.get(a, "UNKNOWN")) == "CEX", "to"]) | set(sout.loc[sout["to"].map(lambda a: lm.get(a, "UNKNOWN")) == "CEX", "from"])))}

    full = load_complete_designated_transfers(truncate=False)
    if full is not None:
        fin_, fout_ = full[full["to"].isin(seed_set)], full[full["from"].isin(seed_set)]
        act_enf = pd.concat([fin_.assign(addr=fin_["to"]), fout_.assign(addr=fout_["from"])])
        print(f"enforcement statistics use the untruncated history to {full['t'].max().date()} "
              f"({len(act_enf):,} address-transfers, against {len(act):,} inside the event window)")
    else:
        act_enf = act
    out["tether_enforcement"] = tether_enforcement(df, seeds, act_enf, signed, txn=full)
    out["tether_enforcement"]["history_end"] = str((full if full is not None else complete)["t"].max().date())

    # monthly volume through designated addresses and through the whole 2-hop network
    monthly = act.set_index("t").resample("MS")["value"].sum()
    monthly_all = df.set_index("t").resample("MS")["value"].sum()
    monthly_active = act.set_index("t").groupby(pd.Grouper(freq="MS"))["addr"].nunique()
    out["nbctf_monthly"] = {"month": [str(k.date()) for k in monthly.index], "designated_volume_usdt": monthly.tolist(), "network_volume_usdt": monthly_all.reindex(monthly.index).fillna(0).tolist(),
                            "designated_active_addresses": monthly_active.reindex(monthly.index).fillna(0).astype(int).tolist()}

    # ============================================================ D. Ukraine donations
    ukr = {}
    for tag, fname, unit, anchor, lower in [("tron", "ukraine_tron_usdt_edges_2hop.csv", "ms", UKR_TRON, False), ("eth", "ukraine_eth_usdt_edges_2hop.csv", "s", UKR_ETH, True)]:
        ch = find("ch_data/ukraine_eth_anchor_usdt.csv", required=False)
        if tag == "eth" and ch.exists():
            raw = pd.read_csv(ch)
            du = pd.DataFrame({"from": raw["from_addr"].str.lower(), "to": raw["to_addr"].str.lower(), "value": raw["raw_value"].astype(float) / 1e6, "t": pd.to_datetime(raw["blockTimestamp"], unit="s")})
            du = du[(du["value"] > 0) & (du["value"] <= VALUE_CAP) & (du["t"] < DATA_END)]
        else:
            du = load_edges(fname, unit, lower)
        don = du[du["to"] == anchor]
        outflow = du[du["from"] == anchor]
        if len(don) == 0:
            ukr[tag] = {"n_donations": 0}
            continue
        daily = don.set_index("t").resample("D")["value"].agg(["sum", "count"])
        donors_daily = don.set_index("t").groupby(pd.Grouper(freq="D"))["from"].nunique()
        dv = don.groupby("from")["value"].sum().sort_values(ascending=False)
        windows = {}
        for lo, hi, lab in [(0, 7, "week1"), (0, 30, "days0_30"), (0, 90, "days0_90"), (90, 365, "days90_365"), (365, 2000, "after_1y")]:
            m = (don["t"] >= INVASION + pd.Timedelta(days=lo)) & (don["t"] < INVASION + pd.Timedelta(days=hi))
            windows[lab] = {"volume_usdt": float(don.loc[m, "value"].sum()), "donations": int(m.sum()), "donors": int(don.loc[m, "from"].nunique())}
        lmc = label_map(tag) if tag == "eth" else lm
        ukr[tag] = {"anchor": anchor, "n_donations": int(len(don)), "volume_usdt": float(don["value"].sum()), "n_donors": int(don["from"].nunique()),
                    "outflow_usdt": float(outflow["value"].sum()), "n_outflows": int(len(outflow)), "first_donation": str(don["t"].min()), "last_donation": str(don["t"].max()),
                    "median_donation_usdt": float(don["value"].median()), "mean_donation_usdt": float(don["value"].mean()),
                    "top1pct_donor_volume_share": float(dv.iloc[:max(1, len(dv) // 100)].sum() / dv.sum()), "top10_donor_volume_share": float(dv.iloc[:10].sum() / dv.sum()),
                    "share_donations_below_100": float((don["value"] < 100).mean()), "cex_volume_share": float(don.loc[don["from"].map(lambda a: lmc.get(a, "UNKNOWN")) == "CEX", "value"].sum() / don["value"].sum()),
                    "windows": windows, "peak_day": str(daily["sum"].idxmax().date()), "peak_day_volume": float(daily["sum"].max()),
                    "donor_lorenz": [float(x) for x in (dv.cumsum() / dv.sum()).to_numpy()[np.linspace(0, len(dv) - 1, 200).astype(int)]],
                    "donor_lorenz_log_rank": [int(i) + 1 for i in np.unique(np.geomspace(1, len(dv), 400).astype(int)) - 1],
                    "donor_lorenz_log": [float(x) for x in (dv.cumsum() / dv.sum()).to_numpy()[np.unique(np.geomspace(1, len(dv), 400).astype(int)) - 1]],
                    "n_donors_ranked": int(len(dv)),
                    "donation_size_hist": {"bin_edges_usdt": [float(x) for x in np.logspace(-1, 7, 33)], "counts": [int(c) for c in np.histogram(don["value"], bins=np.logspace(-1, 7, 33))[0]]},
                    "daily": {"date": [str(k.date()) for k in daily.index], "volume_usdt": daily["sum"].tolist(), "donations": daily["count"].astype(int).tolist(), "donors": donors_daily.reindex(daily.index).fillna(0).astype(int).tolist()}}
    out["ukraine"] = ukr

    with open(out_path("phenomena.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    t, e, p, c, k, u = out["nbctf_timing"], out["nbctf_event_study"], out["placebo_event_study"], out["counterparty_persistence"], out["concentration"], out["ukraine"]
    print(f"designated in window: {t['n_designated_in_window']}, after window: {t['n_designated_after_window']}; median days last activity->signed {t['median_days_last_activity_to_signed']}; dormant>30d {t['share_dormant_30d_at_signing']:.2f}; active after {t['share_active_after_signing']:.2f}; signed->published median {t['median_days_signed_to_published']} d; volume after signing share {t['volume_after_signing_share']:.3f}")
    print(f"event study: pre {e['pre_mean_weekly_volume']/1e6:.1f}M/wk, weeks0-4 {e['weeks_0_4_mean_volume']/1e6:.1f}M, post7-26 {e['post_weeks_7_26_mean_volume']/1e6:.1f}M; transfers {e['pre_mean_weekly_transfers']:.0f}->{e['post_mean_weekly_transfers']:.0f}; per-address active after {e['per_address']['share_with_any_post_activity']:.2f}, median post share {e['per_address']['median_post_share_of_window_volume']:.2f}")
    print(f"placebo (n={p['n_addresses']}): pre {p['pre_mean_weekly_volume']/1e6:.1f}M/wk, post7-26 {p['post_weeks_7_26_mean_volume']/1e6:.1f}M; transfers {p['pre_mean_weekly_transfers']:.0f}->{p['post_mean_weekly_transfers']:.0f}; per-address active after {p['per_address']['share_with_any_post_activity']:.2f}, median post share {p['per_address']['median_post_share_of_window_volume']:.2f}")
    print(f"counterparties: n={c['n_counterparties']}, active after {c['share_active_after_order']:.2f}, 90d after {c['share_active_90d_after_order']:.2f}, volume share after {c['volume_share_after_order']:.2f}")
    print(f"concentration: top10 {k['top10_share']:.3f}, top100 {k['top100_share']:.3f}, top1% {k['top1pct_share']:.3f}; designated top10 {k['designated_top10_share']:.3f}, top1% {k['designated_top1pct_share']:.3f}; CEX in/out {k['seed_inflow_share_from_labelled_cex']:.3f}/{k['seed_outflow_share_to_labelled_cex']:.3f}")
    for tag, r in u.items():
        if r.get("n_donations"):
            print(f"ukraine {tag}: {r['n_donations']:,} donations {r['volume_usdt']/1e6:.2f}M from {r['n_donors']:,} donors; week1 {r['windows']['week1']['volume_usdt']/1e6:.2f}M; median {r['median_donation_usdt']:.0f}; top1% {r['top1pct_donor_volume_share']:.2f}; <100 USDT {r['share_donations_below_100']:.2f}; cex {r['cex_volume_share']:.2f}; peak {r['peak_day']}")
    print("saved phenomena.json")


if __name__ == "__main__":
    main()
