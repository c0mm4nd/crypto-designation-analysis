#!/usr/bin/env python3
"""Timing and on-chain enforcement for the OFAC-listed TRON addresses, the same measures as for the NBCTF orders.

For every TRON USDT address listed under the three OFAC programmes used in the paper, the listing date is the
date the address itself was added to the SDN List (ofac_listing_dates.json, from the OFAC Sanctions List Service
change history; entity listing date where the address predates the history), and the address's complete USDT
history is read from the archive node. Reported per address and pooled: the
lag from the last transfer before listing to the listing date, the share dormant for more than 30 and 90 days
at listing, activity after listing, the post-to-pre volume ratio in weeks 7-26 after listing against the 26
weeks before, whether and when Tether blacklisted the address, the balance at the freeze and the peak balance.

Needs CH_URL and CH_AUTH.

Usage:
  python scripts/compute_ofac_timing.py [--out ofac_timing.json]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
from compute_phenomena import W, tron_b58_to_hex, tron_hex_to_b58  # noqa: E402
from compute_custody import freeze_dates  # noqa: E402

USDT = "a614f803b6fd780986a42c78ec9c7f77e6ded13c"
TRANSFER = "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
VALUE_CAP = 1e8
PROGRAMMES = ("ofac_iran", "ofac_russia_ukraine", "ofac_terrorist_financing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ofac_timing.json")
    args = ap.parse_args()
    url, auth = os.environ.get("CH_URL"), os.environ.get("CH_AUTH")
    if not url or not auth:
        raise SystemExit("set CH_URL and CH_AUTH")
    dates = json.load(open(find("ofac_listing_dates.json")))["addresses"]
    rows = []
    for prog in PROGRAMMES:
        for r in csv.DictReader(open(find(f"{prog}.csv"))):
            if r["address"].startswith("T") and r["currency"] == "USDT":
                d = dates[r["address"]]
                rows.append({"programme": prog, "address": r["address"], "profile_id": r["profile_id"], "entity": r["entity_name"],
                             "listed": d["date_listed"], "entity_listed": d["entity_date_published"], "date_source": d["date_source"]})
    seeds = pd.DataFrame(rows).drop_duplicates("address")
    seeds["listed"] = pd.to_datetime(seeds["listed"]); seeds["entity_listed"] = pd.to_datetime(seeds["entity_listed"])
    print(f"{len(seeds)} OFAC TRON USDT addresses under {seeds['profile_id'].nunique()} entities; listing dates {seeds['listed'].min().date()} to {seeds['listed'].max().date()}; "
          f"{(seeds['date_source'] == 'delta file').sum()} dated by the list's change history", flush=True)
    hexes = {tron_b58_to_hex(a): a for a in seeds["address"]}
    lst = ",".join(f"'{h}'" for h in hexes)
    sql = f"""SELECT a, dir, blockTimestamp AS ts, reinterpretAsUInt64(reverse(unhex(substring(data, 49, 16)))) / 1e6 AS v, transactionHash AS h, logIndex AS li,
                     substring(topic1, 25, 40) AS f, substring(topic2, 25, 40) AS t
              FROM (SELECT *, substring(topic1, 25, 40) AS a, 'out' AS dir FROM tron.events WHERE address = '{USDT}' AND topic0 = '{TRANSFER}' AND topic2 IS NOT NULL AND substring(topic1, 25, 40) IN ({lst})
                    UNION ALL
                    SELECT *, substring(topic2, 25, 40) AS a, 'in' AS dir FROM tron.events WHERE address = '{USDT}' AND topic0 = '{TRANSFER}' AND topic2 IS NOT NULL AND substring(topic2, 25, 40) IN ({lst}))
              FORMAT TSVWithNames"""
    r = subprocess.run(["curl", "-sS", "--max-time", "7200", "-u", auth, url + "/?max_threads=32&max_memory_usage=30000000000", "--data-binary", sql], capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.startswith("Code:"):
        raise SystemExit(r.stdout[:400] + r.stderr[:400])
    ev = pd.read_csv(io.StringIO(r.stdout), sep="\t")
    ev = ev.drop_duplicates(subset=["h", "li", "dir"])
    ev = ev[(ev["v"] > 0) & (ev["v"] <= VALUE_CAP) & (ev["f"] != ev["t"])]
    ev["addr"] = ev["a"].map(hexes); ev["t"] = pd.to_datetime(ev["ts"], unit="ms")
    hist_end = ev["t"].max()
    print(f"{len(ev):,} transfers to {hist_end.date()}", flush=True)
    frozen = freeze_dates()
    per = []
    for _, s in seeds.iterrows():
        g = ev[ev["addr"] == s["address"]].sort_values("t")
        d = s["listed"]
        rec = {"programme": s["programme"], "entity": s["entity"], "profile_id": s["profile_id"], "listed": str(d.date()),
               "entity_listed": str(s["entity_listed"].date()), "days_entity_listing_to_address_listing": float((d - s["entity_listed"]).days),
               "date_source": s["date_source"], "n_transfers": int(len(g)), "volume": float(g["v"].sum())}
        if len(g):
            before = g[g["t"] < d]; after = g[g["t"] >= d]
            rec.update({"first": str(g["t"].min().date()), "last": str(g["t"].max().date()),
                        "days_last_before_listing_to_listing": float((d - before["t"].max()).days) if len(before) else None,
                        "active_before": bool(len(before)), "active_after": bool(len(after)),
                        "volume_before": float(before["v"].sum()), "volume_after": float(after["v"].sum())})
            rel = ((g["t"] - d).dt.days // 7).astype(int)
            pre = g[(rel >= -W) & (rel < 0)]["v"].sum() / W
            post = g[(rel >= 7) & (rel <= W)]["v"].sum() / (W - 6)
            rec["ratio_post_7_26_to_pre"] = float(post / pre) if pre > 0 else None
            rec["pre_mean_weekly_volume"] = float(pre)
            sign = np.where(g["dir"] == "in", 1.0, -1.0); bal = np.cumsum(sign * g["v"].to_numpy())
            rec["peak_balance"] = float(bal.max())
            fz = frozen.get(s["address"])
            rec["frozen"] = str(fz.date()) if fz is not None else None
            if fz is not None:
                rec["days_listing_to_freeze"] = float((fz - d).days)
                i = np.searchsorted(g["t"].to_numpy(), np.datetime64(fz), side="right")
                rec["balance_at_freeze"] = float(bal[i - 1]) if i > 0 else 0.0
                rec["sent_after_freeze"] = bool((g[(g["t"] > fz) & (g["dir"] == "out")]).shape[0])
        else:
            rec["frozen"] = str(frozen[s["address"]].date()) if s["address"] in frozen else None
        per.append(rec)
    t = pd.DataFrame(per)
    act = t[t["n_transfers"] > 0]
    lag = act["days_last_before_listing_to_listing"].dropna()
    fz = act[act["frozen"].notna()]
    def pooled(g, lagg):
        return {"n_listed": int(len(g)), "n_with_transfer": int((g["n_transfers"] > 0).sum()),
                "n_active_before_listing": int(lagg.notna().sum()),
                "median_days_last_to_listing": float(lagg.median()) if len(lagg) else None,
                "share_dormant_30d": float((lagg > 30).mean()) if len(lagg) else None,
                "share_dormant_90d": float((lagg > 90).mean()) if len(lagg) else None,
                "share_active_after_listing": float(g[g["n_transfers"] > 0]["active_after"].mean()) if (g["n_transfers"] > 0).any() else None,
                "volume_share_after_listing": float(g["volume_after"].sum() / g["volume"].sum()) if g["volume"].sum() > 0 else None,
                "pooled_ratio_post_7_26_to_pre": float(sum(x["ratio_post_7_26_to_pre"] * x["pre_mean_weekly_volume"] for _, x in g.iterrows() if x.get("ratio_post_7_26_to_pre") is not None and not pd.isna(x.get("ratio_post_7_26_to_pre"))) / max(1e-9, sum(x["pre_mean_weekly_volume"] for _, x in g.iterrows() if x.get("ratio_post_7_26_to_pre") is not None and not pd.isna(x.get("ratio_post_7_26_to_pre"))))),
                "n_frozen": int(g["frozen"].notna().sum()),
                "median_days_listing_to_freeze": float(g["days_listing_to_freeze"].median()) if g["days_listing_to_freeze"].notna().any() else None,
                "n_frozen_before_listing": int((g["days_listing_to_freeze"] < 0).sum()),
                "balance_at_freeze_total": float(g["balance_at_freeze"].sum()) if "balance_at_freeze" in g else 0.0,
                "peak_balance_total": float(g["peak_balance"].sum()), "volume_total": float(g["volume"].sum()),
                "n_sent_after_freeze": int(g["sent_after_freeze"].fillna(False).sum()) if "sent_after_freeze" in g else 0}
    out = {"description": ("Timing and Tether enforcement for the OFAC-listed TRON USDT addresses, measured as for the NBCTF orders "
                           "from complete histories in the archive node; an address is dated by the SDN List publication that added it."),
           "n_dated_by_change_history": int((seeds["date_source"] == "delta file").sum()),
           "history_end": str(hist_end.date()), "blacklist_end": str(max(frozen.values()).date()),
           "pooled": pooled(t, lag), "by_programme": {p: pooled(t[t["programme"] == p], t[t["programme"] == p]["days_last_before_listing_to_listing"].dropna()) for p in PROGRAMMES},
           "by_entity": {}, "per_address": t.drop(columns=["entity"]).assign(entity=t["entity"]).to_dict(orient="records")}
    for e, g in t.groupby("entity"):
        out["by_entity"][e] = pooled(g, g["days_last_before_listing_to_listing"].dropna())
        out["by_entity"][e]["listed"] = str(g["listed"].iloc[0]); out["by_entity"][e]["entity_listed"] = str(g["entity_listed"].iloc[0])
        out["by_entity"][e]["programmes"] = sorted(g["programme"].unique().tolist())
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1, default=str)
    p = out["pooled"]; print(json.dumps(p, indent=1))
    for e, v in out["by_entity"].items():
        print(f"{e[:40]:40} listed {v['listed']} n={v['n_listed']} tx={v['n_with_transfer']} lag={v['median_days_last_to_listing']} after={v['share_active_after_listing']} frozen={v['n_frozen']} lagfreeze={v['median_days_listing_to_freeze']} bal={v['balance_at_freeze_total']:.0f} vol={v['volume_total']/1e6:.2f}M")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
