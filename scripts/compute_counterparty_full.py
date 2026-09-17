#!/usr/bin/env python3
"""Counterparty persistence on complete histories, on the same footing as the designated addresses.

The counterparty layer's persistence was measured on histories the block-explorer crawl returned, which are
truncated newest-first by page limits, so early transfers are the ones missing and the share of a
counterparty's volume that falls after the order is biased upwards; the designated addresses' 9% comes from
complete histories. This script recomputes the three persistence measures for every counterparty that
transacted with a designated address before that address's order, from the full USDT transfer history of
each counterparty in the archive node, cut at the same DATA_END as the rest of the event study, and reports
them alongside the designated addresses' own values computed identically.

Needs the ClickHouse node: CH_URL and CH_AUTH in the environment.

Usage:
  CH_URL=http://host:8123 CH_AUTH=user:password python scripts/compute_counterparty_full.py [--out counterparty_full.json]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import out_path  # noqa: E402
from compute_phenomena import DATA_END, MIN_POST_DAYS, israel_seed_dates, load_complete_designated_transfers, tron_b58_to_hex  # noqa: E402

USDT = "a614f803b6fd780986a42c78ec9c7f77e6ded13c"
TRANSFER = "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
VALUE_CAP = 1e8


def query(url, auth, sql, data=None):
    # an INSERT sends its rows as the body and the statement in the URL, so the body is not parsed as SQL
    import urllib.parse
    target = url if data is None else url + "&query=" + urllib.parse.quote(sql)
    cmd = ["curl", "-sS", "--max-time", "14400", "-u", auth, target, "--data-binary", sql if data is None else "@-"]
    r = subprocess.run(cmd, input=data, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.startswith("Code:"):
        raise RuntimeError((r.stdout + r.stderr)[:500])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="counterparty_full.json")
    args = ap.parse_args()
    url, auth = os.environ.get("CH_URL"), os.environ.get("CH_AUTH")
    if not url or not auth:
        raise SystemExit("set CH_URL and CH_AUTH")

    seeds = israel_seed_dates()
    seed_set = set(seeds["address"])
    signed = dict(zip(seeds["address"], seeds["signed"]))
    df = load_complete_designated_transfers(truncate=True)
    sin = df[df["to"].isin(seed_set)].rename(columns={"from": "cp", "to": "seed"})
    sout = df[df["from"].isin(seed_set)].rename(columns={"to": "cp", "from": "seed"})
    cps = pd.concat([sin, sout])
    cps = cps[~cps["cp"].isin(seed_set)].assign(d=lambda x: x["seed"].map(signed))
    cps_pre = cps[cps["t"] < cps["d"]]
    cp_first = cps_pre.groupby("cp")["d"].min()
    cp_first = cp_first[cp_first <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)]
    print(f"{len(cp_first):,} counterparties with pre-order contact and {MIN_POST_DAYS} days of follow-up", flush=True)

    # the designated addresses themselves, on the same footing
    des = seeds[seeds["signed"] <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)]
    targets = pd.concat([pd.DataFrame({"addr": cp_first.index, "d": cp_first.values, "kind": "counterparty"}),
                         pd.DataFrame({"addr": des["address"], "d": des["signed"], "kind": "designated"})])
    targets["hex"] = targets["addr"].map(tron_b58_to_hex)
    # milliseconds since the epoch regardless of the datetime resolution pandas chose for the column
    targets["d_ms"] = ((pd.to_datetime(targets["d"]) - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64")

    q = lambda s, data=None: query(url + "/?max_threads=32&max_memory_usage=40000000000&max_bytes_before_external_group_by=15000000000&max_execution_time=0", auth, s, data)
    q("CREATE DATABASE IF NOT EXISTS scratch")
    q("DROP TABLE IF EXISTS scratch.cp_targets")
    q("CREATE TABLE scratch.cp_targets (hex String, d_ms UInt64, kind String) ENGINE = MergeTree ORDER BY hex")
    q("INSERT INTO scratch.cp_targets FORMAT TSV", data="\n".join(f"{h}\t{d}\t{k}" for h, d, k in zip(targets["hex"], targets["d_ms"], targets["kind"])) + "\n")
    n = q("SELECT count() FROM scratch.cp_targets").strip()
    print(f"{n} targets loaded", flush=True)
    end_ms = int(DATA_END.value // 10**6)
    sql = f"""
    SELECT t.hex AS hex, t.kind AS kind,
           countIf(e.ts < t.d_ms) AS n_before, countIf(e.ts >= t.d_ms) AS n_after,
           sumIf(e.v, e.ts < t.d_ms) AS vol_before, sumIf(e.v, e.ts >= t.d_ms) AS vol_after,
           min(e.ts) AS first_ms, max(e.ts) AS last_ms
    FROM (
        SELECT a, ts, v FROM (
            SELECT substring(topic1, 25, 40) AS a, blockTimestamp AS ts,
                   reinterpretAsUInt64(reverse(unhex(substring(data, 49, 16)))) / 1e6 AS v
            FROM tron.events
            WHERE address = '{USDT}' AND topic0 = '{TRANSFER}' AND topic2 IS NOT NULL AND blockTimestamp < {end_ms}
              AND substring(topic1, 25, 40) IN (SELECT hex FROM scratch.cp_targets)
            UNION ALL
            SELECT substring(topic2, 25, 40) AS a, blockTimestamp AS ts,
                   reinterpretAsUInt64(reverse(unhex(substring(data, 49, 16)))) / 1e6 AS v
            FROM tron.events
            WHERE address = '{USDT}' AND topic0 = '{TRANSFER}' AND topic2 IS NOT NULL AND blockTimestamp < {end_ms}
              AND substring(topic2, 25, 40) IN (SELECT hex FROM scratch.cp_targets)
        ) WHERE v > 0 AND v <= {VALUE_CAP}
    ) AS e
    INNER JOIN scratch.cp_targets AS t ON e.a = t.hex
    GROUP BY hex, kind
    FORMAT TSVWithNames"""
    print("querying complete histories (this scans the whole USDT event table)...", flush=True)
    res = pd.read_csv(io.StringIO(q(sql)), sep="\t")
    res = res.merge(targets[["hex", "addr", "d"]], on="hex", how="left")
    res["last"] = pd.to_datetime(res["last_ms"], unit="ms")
    print(f"{len(res):,} addresses returned", flush=True)

    def measures(g: pd.DataFrame) -> dict:
        return {"n": int(len(g)),
                "share_active_after_order": float((g["n_after"] > 0).mean()),
                "share_active_90d_after_order": float((g["last"] > g["d"] + pd.Timedelta(days=90)).mean()),
                "volume_share_after_order": float(g["vol_after"].sum() / (g["vol_before"].sum() + g["vol_after"].sum())),
                "median_per_address_post_share": float((g["vol_after"] / (g["vol_before"] + g["vol_after"])).median()),
                "volume_before": float(g["vol_before"].sum()), "volume_after": float(g["vol_after"].sum())}

    cp = res[res["kind"] == "counterparty"]
    de = res[res["kind"] == "designated"]
    out = {"description": ("Counterparty persistence recomputed from the complete USDT history of every counterparty "
                           "that transacted with a designated address before that address's order, cut at data_end, "
                           "with the designated addresses measured identically."),
           "data_end": str(DATA_END.date()), "min_post_days": MIN_POST_DAYS,
           "counterparties": measures(cp), "designated": measures(de)}
    # the volume-weighted measure is dominated by exchanges; also give it without the 1% largest counterparties
    tot = cp["vol_before"] + cp["vol_after"]
    small = cp[tot <= tot.quantile(0.99)]
    out["counterparties_excluding_top_1pct_by_volume"] = measures(small)
    # check against the CSV-based designated numbers
    des178 = set(des["address"])
    des_csv = df[df["to"].isin(des178) | df["from"].isin(des178)]
    out["check_designated_transfer_count_csv_vs_node"] = {"csv": int(len(des_csv)), "node": int(de["n_before"].sum() + de["n_after"].sum())}
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    c, d = out["counterparties"], out["designated"]
    print(f"counterparties (n={c['n']:,}): active after {100*c['share_active_after_order']:.0f}%, 90 d {100*c['share_active_90d_after_order']:.0f}%, "
          f"volume share after {100*c['volume_share_after_order']:.0f}% (median per address {100*c['median_per_address_post_share']:.0f}%)")
    print(f"designated    (n={d['n']:,}): active after {100*d['share_active_after_order']:.0f}%, 90 d {100*d['share_active_90d_after_order']:.0f}%, "
          f"volume share after {100*d['volume_share_after_order']:.0f}%")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
