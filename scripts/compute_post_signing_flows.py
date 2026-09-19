#!/usr/bin/env python3
"""Where the USDT that left designated addresses after signing went.

For every designated address that Tether later froze, the outflows between the signing of its order and
its freeze are grouped by recipient: other designated addresses (internal to the programme), addresses that
carry an exchange or payment-processor label, and unlabelled addresses; the largest recipients are ranked
with their labels where available. Reported for all frozen addresses and for the May 2023 order, whose
addresses moved almost all of that value. Labels are used under their providers' terms and only aggregates
and label categories are written out.

Usage:
  python scripts/compute_post_signing_flows.py [--out post_signing_flows.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import out_path  # noqa: E402
from compute_custody import freeze_dates, raw_tags, entity, is_exchange  # noqa: E402
from compute_phenomena import israel_seed_dates, load_complete_designated_transfers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="post_signing_flows.json")
    args = ap.parse_args()
    seeds = israel_seed_dates(); signed = dict(zip(seeds["address"], seeds["signed"])); order_of = dict(zip(seeds["address"], seeds["order"]))
    frozen = freeze_dates(); tags = raw_tags()
    df = load_complete_designated_transfers(truncate=False); df = df[df["from"] != df["to"]]
    seed_set = set(seeds["address"])
    rows = []
    for addr in seed_set:
        fz = frozen.get(addr)
        if fz is None:
            continue
        o = df[(df["from"] == addr) & (df["t"] >= signed[addr]) & (df["t"] < fz)]
        if o.empty:
            continue
        for to, v in o.groupby("to")["value"].sum().items():
            rows.append({"order": order_of[addr], "from": addr, "to": to, "value": float(v)})
    t = pd.DataFrame(rows)
    def cat(to):
        if to in seed_set:
            return "designated (same programme)"
        e = entity(tags, to)
        if e and is_exchange(e):
            return "labelled exchange or payment processor"
        if e:
            return "other labelled"
        return "unlabelled"
    t["category"] = t["to"].map(cat)
    def summary(g):
        by = g.groupby("category")["value"].sum().sort_values(ascending=False)
        total = float(g["value"].sum())
        top = g.groupby("to")["value"].sum().sort_values(ascending=False)
        top10 = [{"rank": i + 1, "share_of_outflow": float(v / total), "category": cat(a),
                  "label": (entity(tags, a).split(".")[0] if entity(tags, a) else None),
                  "designated_order": order_of.get(a)} for i, (a, v) in enumerate(top.head(10).items())]
        return {"total_usdt": total, "n_sending_addresses": int(g["from"].nunique()), "n_recipients": int(g["to"].nunique()),
                "share_by_category": {k: float(v / total) for k, v in by.items()},
                "top10_recipients": top10, "share_top10": float(top.head(10).sum() / total), "share_top1": float(top.iloc[0] / total)}
    # what the largest recipients are, from the archive node when it is reachable: their activity over the whole
    # USDT history, when they first appeared, and whether they belong to the 400 most active addresses
    node = {}
    url, auth = os.environ.get("CH_URL"), os.environ.get("CH_AUTH")
    top_all = t.groupby("to")["value"].sum().sort_values(ascending=False).head(20)
    if url and auth:
        import subprocess
        from compute_phenomena import tron_b58_to_hex
        hexes = {tron_b58_to_hex(a): a for a in top_all.index}
        lst = ",".join(f"'{h}'" for h in hexes)
        sql = f"""SELECT a, count() n, min(ts) first_ms, uniqExact(cp) n_counterparties FROM (
              SELECT substring(topic1,25,40) a, substring(topic2,25,40) cp, blockTimestamp ts FROM tron.events
               WHERE address='a614f803b6fd780986a42c78ec9c7f77e6ded13c' AND topic0='ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef' AND substring(topic1,25,40) IN ({lst})
              UNION ALL
              SELECT substring(topic2,25,40) a, substring(topic1,25,40) cp, blockTimestamp ts FROM tron.events
               WHERE address='a614f803b6fd780986a42c78ec9c7f77e6ded13c' AND topic0='ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef' AND substring(topic2,25,40) IN ({lst})
              ) GROUP BY a FORMAT TSV"""
        r = subprocess.run(["curl", "-s", "--max-time", "7200", "-u", auth, url + "/?max_threads=32&max_memory_usage=30000000000", "--data-binary", sql], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            h, n, first_ms, ncp = line.split("\t")
            node[hexes[h]] = {"transfers": int(n), "first_seen": str(pd.to_datetime(int(first_ms), unit="ms").date()), "counterparties": int(ncp)}
    hubs = set(json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "complete_network_hubs.json"))).get("addresses_ranked", [])) if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "complete_network_hubs.json")) else set()
    top20 = []
    for i, (a, v) in enumerate(top_all.items()):
        rec = {"rank": i + 1, "share_of_outflow": float(v / t["value"].sum()), "category": cat(a), "label": (entity(tags, a).split(".")[0] if entity(tags, a) else None),
               "in_top400_by_activity": a in hubs}
        rec.update(node.get(a, {}))
        top20.append(rec)
    out = {"description": ("USDT sent by frozen designated addresses between the signing of their order and their freeze, by "
                           "category of recipient; labels are third-party entity tags and are sparse, so the labelled shares are lower bounds."),
           "all_frozen": summary(t), "top20_recipients_all_frozen": top20}
    for k, g in t.groupby("order"):
        if g["value"].sum() > 1e6:
            out[k] = summary(g)
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    for r in out["top20_recipients_all_frozen"][:10]:
        print("  recipient", r)
    a = out["all_frozen"]; print(f"all frozen: {a['total_usdt']/1e6:.1f} M USDT from {a['n_sending_addresses']} addresses to {a['n_recipients']} recipients; shares {a['share_by_category']}; top10 {a['share_top10']:.2f}")
    for k in out:
        if k.startswith("ASO"):
            a = out[k]; print(f"{k}: {a['total_usdt']/1e6:.1f} M; shares {a['share_by_category']}; top1 {a['share_top1']:.2f} top10 {a['share_top10']:.2f}; top10 labels {[x['label'] or x['category'][:10] for x in a['top10_recipients']]}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
