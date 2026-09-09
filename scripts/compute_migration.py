#!/usr/bin/env python3
"""Did the flows stop, or move to addresses that were not named?

The event study shows flows through designated addresses stopping. That is consistent with the
activity ending and with it moving, and the two have opposite policy readings, so the paper
should not leave it open. A successor operator inherits a customer list, so it should deal with
an order's former counterparties more comprehensively than an address of its size otherwise
would.

Four things this test has to get right, each of which it is easy to get wrong:

  the denominator   Most of an order's counterparties are known only through the designated
                    addresses' own complete histories; the crawl never reached them, so their
                    later activity is unobservable. The test runs on the counterparties the
                    crawl reached *and* that transact after the order, and reports that number
                    rather than the full count.

  dust              Overlap counted without a value floor is dominated by address-poisoning
                    senders, which touch thousands of addresses for a fraction of a cent. Pairs
                    are counted only above a floor (default 1 USDT exchanged).

  the reference     Comparing a newcomer against the largest incumbent compares it against the
                    busiest exchange on the chain, which no successor need beat. The reference
                    here is the designated addresses' own overlap with their counterparties,
                    which is what an operator of this business looks like, reported as a
                    distribution rather than a maximum.

  what a hit means  An address that inherits a customer list may be a successor or may be
                    infrastructure that everyone deals with. Candidates are therefore reported
                    with their rank in the complete TRON USDT network: an address in the top
                    handful of a 213-million-address network is infrastructure, whatever its
                    overlap.

A mode this cannot detect: an operator that shifts volume to a wallet already in use is an
incumbent by construction and will not show up as a newcomer. Stated in the manuscript.

Usage:
  python scripts/compute_migration.py [--out migration.json] [--min-usdt 1.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import find, out_path  # noqa: E402
from compute_phenomena import (DATA_END, MIN_POST_DAYS, israel_seed_dates, label_map,  # noqa: E402
                               load_complete_designated_transfers, load_edges)


def load_crawl() -> pd.DataFrame:
    return load_edges("israel_tron_usdt_edges_2hop.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="migration.json")
    ap.add_argument("--min-designated", type=int, default=5)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-usdt", type=float, default=1.0)
    args = ap.parse_args()

    seeds = israel_seed_dates()
    designated = set(seeds["address"])
    signed = dict(zip(seeds["address"], seeds["signed"]))
    order_of = dict(zip(seeds["address"], seeds["order"]))
    published = dict(zip(seeds["address"], seeds["published"]))

    comp = load_complete_designated_transfers(truncate=True)
    sin = comp[comp["to"].isin(designated)]
    sout = comp[comp["from"].isin(designated)]
    act = pd.concat([sin.assign(addr=sin["to"]), sout.assign(addr=sout["from"])], ignore_index=True)
    last = act.groupby("addr")["t"].max()
    in_window = {a for a in designated
                 if a in signed and signed[a] <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)
                 and a in last.index}

    crawl = load_crawl()
    labels = label_map("tron")
    # rank in the complete network, so a candidate that is simply infrastructure is visible
    hub_rank: dict[str, int] = {}
    hub_tx: dict[str, int] = {}
    try:
        hubs = json.load(open(find("complete_network_hubs.json")))
        for i, (a, n) in enumerate(zip(hubs["addresses_ranked"], hubs["transfers"])):
            hub_rank[a] = i + 1
            hub_tx[a] = n
    except FileNotFoundError:
        print("complete_network_hubs.json absent: candidates reported without a network rank")
    out: dict = {"orders": {}, "note": (
        "Computed on the crawled two-hop graph, whose per-address histories are truncated "
        "newest-first; 'first seen' is therefore not used as a criterion.")}

    orders = defaultdict(list)
    for a in in_window:
        orders[order_of[a]].append(a)

    for order, addrs in sorted(orders.items(), key=lambda kv: -len(kv[1])):
        if len(addrs) < args.min_designated:
            continue
        d = signed[addrs[0]]
        pub = published[addrs[0]]
        cut = pub if pd.notna(pub) and pub > d else d

        # counterparties with contact before the order
        pre = comp[((comp["to"].isin(addrs)) | (comp["from"].isin(addrs))) & (comp["t"] < d)]
        cps = set(pre["from"]) | set(pre["to"])
        cps -= designated
        if not cps:
            continue

        # --- where their business went after the order became public
        post = crawl[(crawl["t"] > cut) & (crawl["from"].isin(cps))]
        dest = post.groupby("to")["value"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        dest = dest[~dest.index.isin(designated)]
        total = float(dest["sum"].sum())
        top = []
        for a, r in dest.head(args.top).iterrows():
            top.append({"address": a, "usdt": float(r["sum"]), "transfers": int(r["count"]),
                        "share_of_post_volume": float(r["sum"] / total) if total else 0.0,
                        "entity": labels.get(a), "network_rank": hub_rank.get(a)})
        lab_share = float(dest[dest.index.map(lambda a: a in labels)]["sum"].sum() / total) if total else 0.0

        # --- who inherits the customer list, counted only over pairs above the value floor
        post_all = crawl[crawl["t"] > cut]
        touch = pd.concat([
            post_all[post_all["from"].isin(cps)].rename(columns={"to": "other", "from": "cp"})[["other", "cp", "value"]],
            post_all[post_all["to"].isin(cps)].rename(columns={"from": "other", "to": "cp"})[["other", "cp", "value"]],
        ], ignore_index=True)
        touch = touch[~touch["other"].isin(designated)]
        pairv = touch.groupby(["other", "cp"])["value"].sum()
        pairv = pairv[pairv >= args.min_usdt]
        ov = (pairv.reset_index().groupby("other")
              .agg(overlap=("cp", "nunique"), usdt=("value", "sum"))
              .sort_values("overlap", ascending=False))
        # the observable denominator: counterparties the crawl reached that transact afterwards
        observable = set(pairv.reset_index()["cp"])

        # reference 1: how comprehensively the designated addresses themselves dealt with these
        # counterparties before the order, which is what this business looks like
        dpair = pre.assign(other=np.where(pre["to"].isin(addrs), pre["to"], pre["from"]),
                           cp=np.where(pre["to"].isin(addrs), pre["from"], pre["to"]))
        dpv = dpair.groupby(["other", "cp"])["value"].sum()
        dpv = dpv[dpv >= args.min_usdt]
        d_ov = dpv.reset_index().groupby("other")["cp"].nunique()

        # reference 2: incumbents, as a distribution rather than a maximum
        pre_all = crawl[crawl["t"] < d]
        touch_pre = pd.concat([
            pre_all[pre_all["from"].isin(cps)].rename(columns={"to": "other", "from": "cp"})[["other", "cp", "value"]],
            pre_all[pre_all["to"].isin(cps)].rename(columns={"from": "other", "to": "cp"})[["other", "cp", "value"]],
        ], ignore_index=True)
        touch_pre = touch_pre[~touch_pre["other"].isin(designated)]
        ppv = touch_pre.groupby(["other", "cp"])["value"].sum()
        ppv = ppv[ppv >= args.min_usdt]
        ov_pre = ppv.reset_index().groupby("other")["cp"].nunique()
        new_only = ov[~ov.index.isin(set(ov_pre.index))]

        def q(series, ps=(50, 75, 90, 100)):
            return {f"p{x}": float(np.percentile(series, x)) for x in ps} if len(series) else {}

        cands = []
        for a, r in new_only.head(10).iterrows():
            cands.append({"address": a, "overlap": int(r["overlap"]), "usdt": float(r["usdt"]),
                          "entity": labels.get(a), "network_rank": hub_rank.get(a),
                          "network_transfers": hub_tx.get(a)})
        out["orders"][str(order)] = {
            "signed": str(d.date()), "public_from": str(cut.date()),
            "n_designated_in_window": len(addrs),
            "n_counterparties_before_order": len(cps),
            "n_counterparties_observable_after": len(observable),
            "post_volume_from_counterparties_usdt": total,
            "share_of_post_volume_to_labelled_entities": lab_share,
            "largest_single_recipient_share": float(dest["sum"].iloc[0] / total) if total and len(dest) else 0.0,
            "min_usdt_per_pair": args.min_usdt,
            "designated_own_overlap": q(d_ov.to_numpy()),
            "incumbent_overlap": q(ov_pre.to_numpy()),
            "newcomer_overlap": q(new_only["overlap"].to_numpy()),
            "top_new_by_overlap": cands,
            "top_destinations": top,
        }
        e = out["orders"][str(order)]
        top_new = cands[0] if cands else None
        print(f"[{order}] {len(addrs)} designated; {len(cps):,} counterparties, {len(observable):,} observable after; "
              f"{total/1e6:,.0f} M USDT moved on, largest recipient {100*e['largest_single_recipient_share']:.1f}%; "
              f"overlap median designated {e['designated_own_overlap'].get('p50', 0):.0f} / "
              f"best newcomer {top_new['overlap'] if top_new else 0} "
              f"(network rank {top_new['network_rank'] if top_new else '-'})", flush=True)

    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
