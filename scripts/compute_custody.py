#!/usr/bin/env python3
"""What kind of accounts the designated addresses are: self-custodied wallets or exchange deposit addresses.

Several findings of the paper admit a mundane reading if the designated addresses were exchange-hosted
deposit sub-accounts rather than wallets their operators controlled: a near-zero balance at freezing is then
a property of sweep-to-hot-wallet custody, not evidence that funds were moved away, and a collapse of flows on
publication is an account closure by the custodian. This script measures, from the complete USDT histories of
the 490 designated addresses, the indicators that separate the two:

  * the address's own third-party entity label ("Binance. User" marks an exchange deposit address);
  * where its outflows went: the share of outflow value sent to its single largest recipient, and that
    recipient's label;
  * whether inflows were swept: the share of inflow value that left the address within 24 hours, and the median
    time value dwelt in the address;
  * the balance path: the peak balance the address ever held, the balance at signing, at publication and at
    freezing, and the value that left between signing and freezing.

Labels are fetched for the largest counterparties through the project's label cache (labels/fetch.py) when the
cache is present; per-address labels are not redistributed, only the aggregates are.

Usage:
  python scripts/compute_custody.py [--out custody.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT, find, out_path  # noqa: E402
from compute_phenomena import (israel_seed_dates, load_complete_designated_transfers,  # noqa: E402
                               tron_hex_to_b58)

SWEEP_HOURS = 24
DUST = 1.0  # USDT


def freeze_dates() -> dict[str, pd.Timestamp]:
    bl = pd.read_csv(find("ch_data/tron_usdt_blacklist_added.csv"))
    bl = bl[bl["event"] == "AddedBlackList"]
    bl["addr"] = bl["addr_hex"].map(tron_hex_to_b58)
    bl["t"] = pd.to_datetime(bl["blockTimestamp"], unit="ms")
    return bl.groupby("addr")["t"].min().to_dict()


def raw_tags() -> dict[str, list[str]]:
    """Every name tag the label cache holds per address, read raw so that the '. User' suffix that marks an
    exchange deposit address survives (label_map() reduces tags to an entity name)."""
    import sqlite3
    tags: dict[str, list[str]] = defaultdict(list)
    try:
        con = sqlite3.connect(find("labels_cache/labels.db"))
    except FileNotFoundError:
        return tags
    for addr, raw in con.execute("select address, raw_json from labels where chain='tron' and raw_json is not null"):
        try:
            recs = json.loads(raw)
        except Exception:
            continue
        for e in recs:
            nt = e.get("name_tag") or ""
            lab = e.get("label") or ""
            if nt:
                tags[addr].append(nt)
            elif lab.startswith("cex:"):
                tags[addr].append("Exchange: " + lab[4:])
    return tags


def entity(lm: dict, addr: str) -> str | None:
    t = lm.get(addr)
    if not t:
        return None
    # prefer the tag that says what the address is, e.g. "Exchange: Binance. User"
    full = [x for x in t if x.startswith("Exchange:") or x.endswith(". User")]
    return (full or t)[0]


def is_exchange_user(tag: str | None) -> bool:
    return bool(tag) and tag.endswith(". User")


def is_exchange(tag: str | None) -> bool:
    if not tag:
        return False
    t = tag.lower()
    return any(k in t for k in ("exchange", "binance", "okx", "huobi", "htx", "mexc", "kucoin", "bybit", "gate",
                                "bitget", "kraken", "coinbase", "zedcex", "nobitex", "bitkeep", "whitebit",
                                "poloniex", "bitfinex", "crypto.com", "bingx", "lbank", "bitmart", "coinex"))


def balance_path(f: pd.DataFrame, addr: str, marks: dict[str, pd.Timestamp | None]) -> dict:
    """Balance of `addr` through its history and at the marked dates, plus sweep statistics."""
    # a self-transfer neither adds to nor removes from the balance, so it is dropped here
    g = f[f["to"] != f["from"]].sort_values("t")
    sign = np.where(g["to"] == addr, 1.0, -1.0)
    delta = sign * g["value"].to_numpy()
    bal = np.cumsum(delta)
    t = g["t"].to_numpy()
    out = {"peak_balance": float(bal.max()) if len(bal) else 0.0,
           "final_balance": float(bal[-1]) if len(bal) else 0.0}
    for k, d in marks.items():
        if d is None or pd.isna(d):
            out[f"balance_at_{k}"] = None
        else:
            i = np.searchsorted(t, np.datetime64(d), side="right")
            out[f"balance_at_{k}"] = float(bal[i - 1]) if i > 0 else 0.0
    # Sweep: for each inflow, the time until the balance first falls back below max(DUST, 1% of the
    # post-inflow balance); an inflow counts as swept if that happens within SWEEP_HOURS.
    ins = np.where(sign > 0)[0]
    swept_value = 0.0
    dwell = []
    for i in ins:
        target = max(DUST, 0.01 * bal[i])
        later = np.where(bal[i + 1:] < target)[0]
        if len(later):
            hours = (t[i + 1 + later[0]] - t[i]) / np.timedelta64(1, "h")
            dwell.append(hours)
            if hours <= SWEEP_HOURS:
                swept_value += delta[i]
    inflow = float(delta[ins].sum()) if len(ins) else 0.0
    out["inflow"] = inflow
    out["sweep_share_24h"] = float(swept_value / inflow) if inflow > 0 else None
    out["median_dwell_hours"] = float(np.median(dwell)) if dwell else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="custody.json")
    ap.add_argument("--fetch-labels", action="store_true", help="fetch labels for the largest counterparties first")
    args = ap.parse_args()

    seeds = israel_seed_dates()
    signed = dict(zip(seeds["address"], seeds["signed"]))
    published = dict(zip(seeds["address"], seeds["published"]))
    order_of = dict(zip(seeds["address"], seeds["order"]))
    frozen = freeze_dates()
    df = load_complete_designated_transfers(truncate=False)
    if df is None:
        raise SystemExit("complete designated histories not found")
    seed_set = set(seeds["address"])
    hist_end = df["t"].max()
    print(f"{len(df):,} transfers to {hist_end.date()} for {len(seed_set)} designated addresses", flush=True)

    # largest counterparties of each designated address, by value, for the label fetch
    ins = df[df["to"].isin(seed_set)]
    outs = df[df["from"].isin(seed_set)]
    top_send = ins.groupby(["to", "from"])["value"].sum().reset_index().sort_values("value", ascending=False).groupby("to").head(2)
    top_recv = outs.groupby(["from", "to"])["value"].sum().reset_index().sort_values("value", ascending=False).groupby("from").head(2)
    want = set(top_send["from"]) | set(top_recv["to"]) | seed_set
    if args.fetch_labels:
        try:
            sys.path.insert(0, str(ROOT))
            from labels.fetch import fetch_web3resear  # noqa: E402
            fetch_web3resear("tron", sorted(want))
        except Exception as e:  # the cache is optional; aggregates are lower bounds without it
            print(f"label fetch skipped: {e}", flush=True)
    lm = raw_tags()
    print(f"labels: {sum(1 for a in want if a in lm)} of {len(want)} addresses of interest carry an entity label", flush=True)

    rows = []
    for addr in sorted(seed_set):
        f = df[((df["to"] == addr) | (df["from"] == addr)) & (df["to"] != df["from"])]
        if f.empty:
            continue
        o = f[f["from"] == addr]
        i = f[f["to"] == addr]
        by_recv = o.groupby("to")["value"].sum().sort_values(ascending=False)
        by_send = i.groupby("from")["value"].sum().sort_values(ascending=False)
        top_r = by_recv.index[0] if len(by_recv) else None
        top_s = by_send.index[0] if len(by_send) else None
        fz = frozen.get(addr)
        r = {"address": addr, "order": order_of[addr], "signed": str(signed[addr].date()),
             "published": str(published[addr].date()) if not pd.isna(published[addr]) else None,
             "frozen": str(fz.date()) if fz is not None else None,
             "n_transfers": int(len(f)), "n_senders": int(by_send.size), "n_recipients": int(by_recv.size),
             "outflow": float(o["value"].sum()),
             "top_recipient_share": float(by_recv.iloc[0] / by_recv.sum()) if len(by_recv) else None,
             "own_label": entity(lm, addr),
             "top_recipient_label": entity(lm, top_r) if top_r else None,
             "top_sender_label": entity(lm, top_s) if top_s else None}
        r.update(balance_path(f, addr, {"signing": signed[addr], "publication": published[addr], "freeze": fz}))
        if fz is not None:
            r["left_between_signing_and_freeze"] = float(o[(o["t"] >= signed[addr]) & (o["t"] < fz)]["value"].sum())
        # tiers of evidence that the address is an exchange deposit address
        r["deposit_by_own_label"] = is_exchange_user(r["own_label"])
        r["deposit_by_recipient"] = bool(r["top_recipient_share"] is not None and r["top_recipient_share"] >= 0.9
                                          and is_exchange(r["top_recipient_label"]))
        r["deposit_by_behaviour"] = bool(r["sweep_share_24h"] is not None and r["sweep_share_24h"] >= 0.9
                                          and r["median_dwell_hours"] is not None and r["median_dwell_hours"] <= SWEEP_HOURS
                                          and r["n_recipients"] <= 3)
        r["deposit_any"] = r["deposit_by_own_label"] or r["deposit_by_recipient"] or r["deposit_by_behaviour"]
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"{len(t)} designated addresses with a USDT transfer", flush=True)

    def agg(g: pd.DataFrame) -> dict:
        fzg = g[g["frozen"].notna()]
        return {"n": int(len(g)),
                "deposit_by_own_label": int(g["deposit_by_own_label"].sum()),
                "deposit_by_recipient": int(g["deposit_by_recipient"].sum()),
                "deposit_by_behaviour": int(g["deposit_by_behaviour"].sum()),
                "deposit_any": int(g["deposit_any"].sum()),
                "share_deposit_any": float(g["deposit_any"].mean()),
                "median_top_recipient_share": float(g["top_recipient_share"].median()),
                "share_top_recipient_ge_0_9": float((g["top_recipient_share"] >= 0.9).mean()),
                "median_sweep_share_24h": float(g["sweep_share_24h"].median()),
                "share_sweep_share_ge_0_9": float((g["sweep_share_24h"] >= 0.9).mean()),
                "median_dwell_hours": float(g["median_dwell_hours"].median()),
                "median_recipients": float(g["n_recipients"].median()),
                "median_peak_balance": float(g["peak_balance"].median()),
                "median_balance_at_signing": float(g["balance_at_signing"].median()),
                "share_balance_at_signing_gt_1": float((g["balance_at_signing"] > DUST).mean()),
                "balance_at_signing_total": float(g["balance_at_signing"].sum()),
                "peak_balance_total": float(g["peak_balance"].sum()),
                "inflow_total": float(g["inflow"].sum()),
                "n_frozen": int(len(fzg)),
                "balance_at_freeze_total": float(fzg["balance_at_freeze"].sum()) if len(fzg) else None,
                "left_between_signing_and_freeze_total": float(fzg["left_between_signing_and_freeze"].sum()) if len(fzg) else None,
                "share_frozen_with_balance_at_signing_gt_1": float((fzg["balance_at_signing"] > DUST).mean()) if len(fzg) else None,
                "top_recipient_labels": dict(Counter(x.split(".")[0] for x in g["top_recipient_label"].dropna()).most_common(8)),
                "own_labels": dict(Counter(x.split(".")[0] for x in g["own_label"].dropna()).most_common(8))}

    out = {"description": ("Custody indicators for the designated addresses from their complete USDT histories: own entity "
                           "label, destination of outflows, sweep behaviour and balance path. Per-address labels are not "
                           "redistributed; the per-address table carries only derived quantities."),
           "history_end": str(hist_end.date()), "sweep_window_hours": SWEEP_HOURS, "dust_usdt": DUST,
           "n_addresses_with_transfer": int(len(t)),
           "label_coverage": {"designated_with_label": int(t["own_label"].notna().sum()),
                              "top_recipients_with_label": int(t["top_recipient_label"].notna().sum()),
                              "top_senders_with_label": int(t["top_sender_label"].notna().sum())},
           "overall": agg(t),
           "by_order": {k: agg(g) for k, g in t.groupby("order")},
           "per_address": t.drop(columns=["own_label", "top_recipient_label", "top_sender_label"]).to_dict(orient="records")}
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1, default=str)
    o = out["overall"]
    print(f"deposit-like: own label {o['deposit_by_own_label']}, by recipient {o['deposit_by_recipient']}, "
          f"by behaviour {o['deposit_by_behaviour']}, any {o['deposit_any']} of {o['n']} "
          f"({100*o['share_deposit_any']:.0f}%)", flush=True)
    print(f"median top-recipient share {o['median_top_recipient_share']:.2f}; median sweep share {o['median_sweep_share_24h']:.2f}; "
          f"median dwell {o['median_dwell_hours']:.1f} h; median peak balance {o['median_peak_balance']:,.0f}; "
          f"median balance at signing {o['median_balance_at_signing']:,.0f} USDT", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
