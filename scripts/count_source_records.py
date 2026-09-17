#!/usr/bin/env python3
"""Counts quoted in the text that come straight from the source records rather than from an analysis.

Records, so that every number in the manuscript is traceable to a deposited output: the share of addresses in
the crawled NBCTF network with a single counterparty; how many designated addresses were frozen at least 90
days before the end of the event-study data; the number of fund-destruction events in the USDT contract logs;
and, per OFAC programme, how many listed TRON addresses the crawl reached.

Usage:
  python scripts/count_source_records.py [--out source_counts.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
from compute_phenomena import DATA_END, israel_seed_dates, load_edges, tron_hex_to_b58  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="source_counts.json")
    args = ap.parse_args()
    out = {"data_end": str(DATA_END.date())}
    df = load_edges("israel_tron_usdt_edges_2hop.csv")
    und = pd.concat([df[["from", "to"]], df[["to", "from"]].rename(columns={"to": "from", "from": "to"})]).drop_duplicates()
    cps = und.groupby("from")["to"].nunique()
    out["nbctf_crawl"] = {"n_addresses": int(len(cps)), "share_with_single_counterparty": float((cps == 1).mean())}
    seeds = israel_seed_dates()
    bl = pd.read_csv(find("ch_data/tron_usdt_blacklist_added.csv"))
    bl = bl[bl["event"] == "AddedBlackList"]
    fz = pd.Series(pd.to_datetime(bl["blockTimestamp"], unit="ms").values, index=bl["addr_hex"].map(tron_hex_to_b58)).groupby(level=0).min()
    s = seeds.set_index("address"); s["frozen"] = fz.reindex(s.index)
    nodes = set(df["from"]) | set(df["to"])
    early = s["frozen"] <= DATA_END - pd.Timedelta(days=90)
    out["frozen_at_least_90d_before_data_end"] = {"of_all_designated": int(early.sum()), "of_crawl_reached": int((early & s.index.isin(nodes)).sum()),
                                                 "n_designated": int(len(s)), "n_crawl_reached": int(s.index.isin(nodes).sum())}
    d = pd.read_csv(find("ch_data/tron_usdt_blackfunds_destroyed.csv"))
    out["destroyed_black_funds_events"] = {"events": int(len(d)), "distinct_addresses": int(d["addr_hex"].nunique())}
    ofac = {}
    for prog in ("ofac_iran", "ofac_russia_ukraine", "ofac_terrorist_financing"):
        x = pd.read_excel(find(f"{prog}_tron_seeds.xlsx"))
        col = [c for c in x.columns if "addr" in str(c).lower()] or [x.columns[0]]
        listed = {str(a).strip() for a in x[col[0]].dropna() if str(a).startswith("T")}
        e = pd.read_csv(find(f"{prog}_tron_usdt_edges_2hop.csv"), usecols=["from", "to"])
        ofac[prog] = {"listed": len(listed), "present_in_crawl": len(listed & (set(e["from"]) | set(e["to"])))}
    out["ofac_tron_addresses"] = ofac
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
