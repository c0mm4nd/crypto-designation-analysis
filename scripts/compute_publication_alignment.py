#!/usr/bin/env python3
"""Where, relative to signing and publication, the flows through each order's addresses fell.

The event study aligns on the signing date. Orders were published later, by 38 days for the order that
supplies almost all of the post-signing volume, and the freeze and any custodial action can only follow
publication. This script gives, for every order, the weekly USDT volume through its addresses from 26 weeks
before signing to 26 weeks after, the week in which publication falls, the volume in the weeks between
signing and publication relative to the pre-signing mean, and the same for the weeks after publication.

Usage:
  python scripts/compute_publication_alignment.py [--out publication_alignment.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import out_path  # noqa: E402
from compute_phenomena import DATA_END, W, israel_seed_dates, load_complete_designated_transfers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="publication_alignment.json")
    args = ap.parse_args()
    seeds = israel_seed_dates()
    df = load_complete_designated_transfers(truncate=True)
    seed_set = set(seeds["address"])
    act = pd.concat([df[df["to"].isin(seed_set)].assign(addr=lambda x: x["to"]),
                     df[df["from"].isin(seed_set)].assign(addr=lambda x: x["from"])])
    out = {"data_end": str(DATA_END.date()), "window_weeks": W, "orders": {}}
    for order, g in seeds.groupby("order"):
        d = g["signed"].iloc[0]
        p = g["published"].iloc[0]
        if d + pd.Timedelta(weeks=W) > DATA_END:
            continue
        a = act[act["addr"].isin(set(g["address"]))]
        rel = ((a["t"] - d).dt.days // 7).astype(int)
        m = (rel >= -W) & (rel <= W)
        vol = np.zeros(2 * W + 1)
        np.add.at(vol, (rel[m] + W).to_numpy(), a.loc[m, "value"].to_numpy())
        pre = vol[:W].mean()
        pub_week = int((p - d).days // 7) if not pd.isna(p) else None
        rec = {"signed": str(d.date()), "published": str(p.date()) if not pd.isna(p) else None,
               "days_signing_to_publication": int((p - d).days) if not pd.isna(p) else None,
               "publication_week": pub_week, "n_addresses": int(len(g)),
               "pre_mean_weekly_volume": float(pre),
               "weekly_volume": vol.tolist()}
        if pre > 0:
            post = vol[W:]  # week 0 onwards
            rec["ratio_by_week_after_signing"] = (post / pre).tolist()
            if pub_week is not None and pub_week > 0:
                rec["ratio_weeks_between_signing_and_publication"] = float(post[0:pub_week].mean() / pre)
                rec["ratio_weeks_after_publication_to_26"] = float(post[pub_week + 1:].mean() / pre) if pub_week + 1 < len(post) else None
            rec["ratio_weeks_7_26"] = float(post[7:].mean() / pre)
            # the first week after signing in which volume is below 10% of the pre mean and stays there
            below = post / pre < 0.1
            first = None
            for k in range(len(below)):
                if below[k:].all():
                    first = k
                    break
            rec["first_week_below_10pct_and_staying"] = first
        out["orders"][order] = rec
        print(f"{order}: signed {d.date()}, published {rec['published']} (week {pub_week}); "
              f"pre {pre/1e6:.2f} M/week; weeks 0-{pub_week}: {rec.get('ratio_weeks_between_signing_and_publication', float('nan')):.3f}; "
              f"after publication: {rec.get('ratio_weeks_after_publication_to_26', float('nan')):.4f}; "
              f"stays below 10% from week {rec.get('first_week_below_10pct_and_staying')}", flush=True)
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
