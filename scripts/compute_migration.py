#!/usr/bin/env python3
"""Where the counterparties' business went after the order.

The event study shows flows through designated addresses stopping. That is consistent with
two very different things: the activity ended, or it moved to addresses that were not named.
The sanctions literature would raise the second first, and the two have opposite policy
readings, so the paper should not leave it open.

Two measurements, both from the crawled graph, because they need the counterparties' own
histories and only the designated addresses have complete ones:

  destinations  Take the undesignated addresses that transacted with a designated address
                *before* its order. Follow their transfers *after* the order and rank the
                addresses that received them. If the receiving end is dispersed across
                exchanges, the business left for the regulated perimeter; if a few unlabelled
                addresses absorb a large share, those are successor candidates.

  overlap       For each undesignated address, count how many of one order's pre-order
                counterparties it deals with after that order. A successor operator inherits
                a customer list, so it should show an overlap that ordinary addresses of the
                same size do not. Reported against the distribution for addresses that were
                already active before the order, which is the null this comparison needs.

The crawl truncates per-address histories newest-first, so "first seen" is biased late and is
not used as a criterion. Both measurements above are computed on observed transfers only.

Usage:
  python scripts/compute_migration.py [--out migration.json] [--min-designated 5]
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
                        "entity": labels.get(a), "was_a_counterparty": a in cps})
        lab_share = float(dest[dest.index.map(lambda a: a in labels)]["sum"].sum() / total) if total else 0.0

        # --- overlap: which addresses inherit this order's customer list
        post_all = crawl[crawl["t"] > cut]
        touch = pd.concat([
            post_all[post_all["from"].isin(cps)].rename(columns={"to": "other", "from": "cp"})[["other", "cp", "value"]],
            post_all[post_all["to"].isin(cps)].rename(columns={"from": "other", "to": "cp"})[["other", "cp", "value"]],
        ], ignore_index=True)
        touch = touch[~touch["other"].isin(designated)]
        ov = touch.groupby("other").agg(overlap=("cp", "nunique"), usdt=("value", "sum"))
        ov = ov.sort_values("overlap", ascending=False)
        # the null: addresses that were already dealing with these counterparties before the
        # order, which is what an ordinary hub of the same kind looks like
        pre_all = crawl[crawl["t"] < d]
        touch_pre = pd.concat([
            pre_all[pre_all["from"].isin(cps)].rename(columns={"to": "other", "from": "cp"})[["other", "cp"]],
            pre_all[pre_all["to"].isin(cps)].rename(columns={"from": "other", "to": "cp"})[["other", "cp"]],
        ], ignore_index=True)
        touch_pre = touch_pre[~touch_pre["other"].isin(designated)]
        ov_pre = touch_pre.groupby("other")["cp"].nunique()
        existing = set(ov_pre.index)
        new_only = ov[~ov.index.isin(existing)]

        out["orders"][str(order)] = {
            "signed": str(d.date()), "public_from": str(cut.date()),
            "n_designated_in_window": len(addrs),
            "n_counterparties_before_order": len(cps),
            "post_volume_from_counterparties_usdt": total,
            "share_of_post_volume_to_labelled_entities": lab_share,
            "top_destinations": top,
            "max_overlap_existing": int(ov_pre.max()) if len(ov_pre) else 0,
            "max_overlap_new": int(new_only["overlap"].max()) if len(new_only) else 0,
            "top_new_by_overlap": [
                {"address": a, "overlap": int(r["overlap"]), "usdt": float(r["usdt"]),
                 "entity": labels.get(a)}
                for a, r in new_only.head(10).iterrows()],
            "n_new_with_overlap_above_max_existing": int((new_only["overlap"] > (ov_pre.max() if len(ov_pre) else 0)).sum()),
        }
        e = out["orders"][str(order)]
        print(f"[{order}] {len(addrs)} designated, {len(cps):,} pre-order counterparties; "
              f"{total/1e6:.1f} M USDT moved after publication, {100*lab_share:.1f}% to labelled entities; "
              f"max overlap: existing {e['max_overlap_existing']}, new {e['max_overlap_new']}", flush=True)

    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
