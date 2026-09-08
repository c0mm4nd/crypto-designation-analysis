#!/usr/bin/env python3
"""Exploratory phenomenon-level analyses for the NC reframing.

1. NBCTF designation event study: USDT flows through designated addresses in the
   weeks before and after the seizure order that names them.
2. Exchange mediation: share of designated-address inflow/outflow with labelled CEX counterparties.
3. Aid for Ukraine (TRON) donation timeline and concentration.
Outputs phenomena.json and prints a summary.
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
sys.path.insert(0, str(ROOT))
from labels.classify import classify  # noqa: E402

VALUE_CAP = 1e8  # USDT; larger values are uint overflow artefacts
DATA_END = pd.Timestamp("2025-01-01")


def load_edges(name: str, ts_unit: str = "ms") -> pd.DataFrame:
    df = pd.read_csv(ROOT / name, usecols=["from", "to", "value", "block_timestamp"])
    df = df[(df["value"] > 0) & (df["value"] <= VALUE_CAP)].copy()
    df["t"] = pd.to_datetime(df["block_timestamp"], unit=ts_unit)
    return df.drop(columns="block_timestamp")


def israel_seed_dates() -> pd.DataFrame:
    ws = openpyxl.load_workbook(ROOT / "IsraelAddrs.xlsx").active
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if len(r) < 5 or not r[3]:
            continue
        addr, cur = str(r[3]).strip(), str(r[4] or "").upper()
        if cur in ("USDT", "TRX") and addr.startswith("T"):
            rows.append({"address": addr, "order": str(r[0]).strip().replace("​", "").replace("\xa0", " "), "order_date": pd.Timestamp(r[1])})
    df = pd.DataFrame(rows).drop_duplicates("address")
    return df


def label_map(chain: str) -> dict[str, str]:
    con = sqlite3.connect(ROOT / "labels_cache" / "labels.db")
    recs: dict[str, dict] = defaultdict(dict)
    for addr, source, raw in con.execute("select address, source, raw_json from labels where chain=?", (chain,)):
        try:
            recs[addr][source] = json.loads(raw) if raw else None
        except Exception:
            recs[addr][source] = None
    out = {}
    for addr, rec in recs.items():
        try:
            c, _ = classify(rec.get("web3resear"), rec.get("slowmist"))
        except Exception:
            c = "UNKNOWN"
        out[addr] = c
    return out


def main():
    out = {}
    # ------------------------------------------------------------------ NBCTF network
    df = load_edges("israel_tron_usdt_edges_2hop.csv")
    seeds = israel_seed_dates()
    nodes = set(df["from"]).union(df["to"])
    seeds = seeds[seeds["address"].isin(nodes)].reset_index(drop=True)
    seed_set = set(seeds["address"])
    print(f"NBCTF: {len(df):,} transfers after value filter; total volume {df['value'].sum()/1e9:.2f} bn USDT; seeds in graph {len(seed_set)}")
    print("orders:", seeds.groupby("order")["order_date"].agg(["first", "count"]).to_string())

    # flows involving seeds
    sin = df[df["to"].isin(seed_set)].copy()
    sout = df[df["from"].isin(seed_set)].copy()
    print(f"seed inflow {sin['value'].sum()/1e6:.1f} M USDT over {len(sin):,} transfers; outflow {sout['value'].sum()/1e6:.1f} M over {len(sout):,}")
    out["nbctf_totals"] = {"transfers": int(len(df)), "volume_usdt": float(df["value"].sum()), "seed_inflow_usdt": float(sin["value"].sum()), "seed_outflow_usdt": float(sout["value"].sum()), "seed_in_transfers": int(len(sin)), "seed_out_transfers": int(len(sout))}

    # first/last activity of seeds relative to designation
    act = pd.concat([sin.assign(addr=sin["to"]), sout.assign(addr=sout["from"])])
    first_last = act.groupby("addr")["t"].agg(["min", "max", "count"])
    fl = seeds.set_index("address").join(first_last)
    fl["days_last_to_order"] = (fl["order_date"] - fl["max"]).dt.days
    fl["days_first_to_order"] = (fl["order_date"] - fl["min"]).dt.days
    in_window = fl[fl["order_date"] <= DATA_END - pd.Timedelta(days=90)]
    print(f"seeds designated >=90 days before data end: {len(in_window)}")
    print("days from last observed transfer to order (median, IQR):", in_window["days_last_to_order"].median(), in_window["days_last_to_order"].quantile([0.25, 0.75]).tolist())
    print("share of seeds with any transfer after order:", float((in_window["max"] > in_window["order_date"]).mean()))
    print("share dormant >30 days before order:", float((in_window["days_last_to_order"] > 30).mean()))
    post = fl[fl["order_date"] > DATA_END]
    print(f"seeds designated after data end (all activity pre-designation): {len(post)}")

    # weekly event study (relative weeks) for in-window seeds: volume and transfers, in and out
    ev = defaultdict(lambda: defaultdict(float))
    seed_date = dict(zip(seeds["address"], seeds["order_date"]))
    for frame, key, col in [(sin, "in", "to"), (sout, "out", "from")]:
        f = frame[frame[col].isin(set(in_window.index))].copy()
        f["rel_week"] = ((f["t"] - f[col].map(seed_date)).dt.days // 7)
        g = f[(f["rel_week"] >= -26) & (f["rel_week"] <= 26)].groupby("rel_week")
        vol = g["value"].sum()
        cnt = g.size()
        for w in range(-26, 27):
            ev[f"{key}_volume"][w] = float(vol.get(w, 0.0))
            ev[f"{key}_transfers"][w] = int(cnt.get(w, 0))
    pre_v = sum(ev["in_volume"][w] + ev["out_volume"][w] for w in range(-26, 0)) / 26
    post_v = sum(ev["in_volume"][w] + ev["out_volume"][w] for w in range(1, 27)) / 26
    pre_c = sum(ev["in_transfers"][w] + ev["out_transfers"][w] for w in range(-26, 0)) / 26
    post_c = sum(ev["in_transfers"][w] + ev["out_transfers"][w] for w in range(1, 27)) / 26
    print(f"event study (n={len(in_window)} seeds): weekly volume pre {pre_v/1e3:.1f}k -> post {post_v/1e3:.1f}k USDT; transfers pre {pre_c:.1f} -> post {post_c:.1f}")
    print("weekly volume by rel week (k USDT):", {w: round((ev['in_volume'][w] + ev['out_volume'][w]) / 1e3, 1) for w in range(-8, 9)})
    out["nbctf_event_study"] = {"n_seeds": int(len(in_window)), "weeks": {k: {str(w): v for w, v in d.items()} for k, d in ev.items()},
                                "pre_weekly_volume": pre_v, "post_weekly_volume": post_v, "pre_weekly_transfers": pre_c, "post_weekly_transfers": post_c,
                                "share_active_after_order": float((in_window["max"] > in_window["order_date"]).mean()),
                                "median_days_last_activity_to_order": float(in_window["days_last_to_order"].median())}

    # counterparties of in-window seeds: does their activity continue after the order?
    cps = pd.concat([sin[sin["to"].isin(set(in_window.index))].rename(columns={"from": "cp", "to": "seed"}),
                     sout[sout["from"].isin(set(in_window.index))].rename(columns={"to": "cp", "from": "seed"})])
    cps = cps[~cps["cp"].isin(seed_set)]
    cp_first_order = cps.assign(od=cps["seed"].map(seed_date)).groupby("cp")["od"].min()
    cp_all = df[df["from"].isin(cp_first_order.index) | df["to"].isin(cp_first_order.index)]
    cp_act = pd.concat([cp_all.assign(cp=cp_all["from"]), cp_all.assign(cp=cp_all["to"])])
    cp_act = cp_act[cp_act["cp"].isin(cp_first_order.index)]
    cp_last = cp_act.groupby("cp")["t"].max()
    cp_df = pd.DataFrame({"od": cp_first_order, "last": cp_last})
    cp_df = cp_df[cp_df["od"] <= DATA_END - pd.Timedelta(days=90)]
    print(f"counterparties of in-window seeds: {len(cp_df):,}; share active after the order: {float((cp_df['last'] > cp_df['od']).mean()):.3f}")
    out["nbctf_counterparties"] = {"n": int(len(cp_df)), "share_active_after_order": float((cp_df["last"] > cp_df["od"]).mean())}

    # exchange mediation
    lm = label_map("tron")
    cat = lambda a: lm.get(a, "UNKNOWN")
    sin["cat"] = sin["from"].map(cat)
    sout["cat"] = sout["to"].map(cat)
    def share(frame):
        tot = frame["value"].sum()
        return {c: float(frame.loc[frame["cat"] == c, "value"].sum() / tot) for c in ["CEX", "DEX", "PROTOCOL", "CONTRACT", "BRIDGE", "INSTITUTION", "SUSPICIOUS", "OTHER", "UNKNOWN"]}
    print("seed inflow by counterparty category (volume share):", {k: round(v, 3) for k, v in share(sin).items() if v > 0})
    print("seed outflow by counterparty category (volume share):", {k: round(v, 3) for k, v in share(sout).items() if v > 0})
    n_labelled = sum(1 for a in nodes if lm.get(a, "UNKNOWN") != "UNKNOWN")
    print(f"labelled addresses in graph: {n_labelled} of {len(nodes):,}")
    # transfer-count share via CEX, and share of seeds with any CEX counterparty
    cex_in = sin[sin["cat"] == "CEX"]; cex_out = sout[sout["cat"] == "CEX"]
    seeds_with_cex = set(cex_in["to"]).union(cex_out["from"])
    print(f"seeds with at least one labelled-CEX counterparty: {len(seeds_with_cex)} / {len(seed_set)}; CEX share of seed transfers in {len(cex_in)/len(sin):.3f}, out {len(cex_out)/len(sout):.3f}")
    out["nbctf_exchange"] = {"inflow_share": share(sin), "outflow_share": share(sout), "seeds_with_cex_counterparty": len(seeds_with_cex), "n_labelled_addresses": n_labelled}

    # counterparty concentration: how many hop-1 addresses carry the seed volume
    cp_vol = pd.concat([sin.rename(columns={"from": "cp"})[["cp", "value"]], sout.rename(columns={"to": "cp"})[["cp", "value"]]])
    cp_vol = cp_vol[~cp_vol["cp"].isin(seed_set)].groupby("cp")["value"].sum().sort_values(ascending=False)
    cum = cp_vol.cumsum() / cp_vol.sum()
    print(f"hop-1 counterparties: {len(cp_vol):,}; top 10 carry {cum.iloc[9]:.3f} of seed-linked volume, top 100 carry {cum.iloc[99]:.3f}; top-10 categories: {[lm.get(a,'UNKNOWN') for a in cp_vol.index[:10]]}")
    out["nbctf_counterparty_concentration"] = {"n_hop1": int(len(cp_vol)), "top10_share": float(cum.iloc[9]), "top100_share": float(cum.iloc[99]), "top10_categories": [lm.get(a, "UNKNOWN") for a in cp_vol.index[:10]]}

    # ------------------------------------------------------------------ Ukraine TRON
    du = load_edges("ukraine_tron_usdt_edges_2hop.csv")
    anchor = "TEFccmfQ38cZS1DTZVhsxKVDckA8Y6VfCy"
    don = du[du["to"] == anchor].copy()
    outflow = du[du["from"] == anchor].copy()
    lmu = lm
    don["cat"] = don["from"].map(lambda a: lmu.get(a, "UNKNOWN"))
    daily = don.set_index("t").resample("D")["value"].agg(["sum", "count"])
    print(f"Ukraine TRON anchor: {len(don):,} donations, {don['value'].sum()/1e6:.2f} M USDT, from {don['from'].nunique():,} donors; outflow {outflow['value'].sum()/1e6:.2f} M in {len(outflow)} transfers")
    print("first donation:", don["t"].min(), " peak day:", daily["sum"].idxmax(), daily["sum"].max())
    d0 = pd.Timestamp("2022-02-24")
    for lo, hi, lab in [(0, 7, "week 1"), (0, 30, "first 30 days"), (0, 90, "first 90 days"), (90, 365, "days 90-365"), (365, 1100, "after 1 year")]:
        m = (don["t"] >= d0 + pd.Timedelta(days=lo)) & (don["t"] < d0 + pd.Timedelta(days=hi))
        print(f"  {lab}: {don.loc[m,'value'].sum()/1e6:.2f} M USDT, {m.sum():,} donations, {don.loc[m,'from'].nunique():,} donors")
    dv = don.groupby("from")["value"].sum().sort_values(ascending=False)
    print(f"  donor concentration: top 1% of donors give {dv.iloc[:max(1,len(dv)//100)].sum()/dv.sum():.3f}; median donation {don['value'].median():.1f} USDT; share of volume from labelled CEX {don.loc[don['cat']=='CEX','value'].sum()/don['value'].sum():.3f}")
    out["ukraine_tron"] = {"n_donations": int(len(don)), "volume_usdt": float(don["value"].sum()), "n_donors": int(don["from"].nunique()), "median_donation": float(don["value"].median()),
                           "top1pct_donor_share": float(dv.iloc[:max(1, len(dv) // 100)].sum() / dv.sum()), "cex_volume_share": float(don.loc[don["cat"] == "CEX", "value"].sum() / don["value"].sum()),
                           "daily": {str(k.date()): [float(v["sum"]), int(v["count"])] for k, v in daily.iterrows() if v["count"] > 0}}
    with open(ROOT / "phenomena.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("saved phenomena.json")


if __name__ == "__main__":
    main()
