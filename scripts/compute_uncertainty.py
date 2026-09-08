#!/usr/bin/env python3
"""Bootstrap intervals for the descriptive quantities the paper reports.

Seizure orders are not randomly assigned, so nothing here is a test of a hypothesis about
an intervention. The quantities are still statistics of a finite sample of addresses and
orders, and a reader is entitled to know how much they would move under a different draw
of that sample. We therefore resample the unit that generated the variation and report a
percentile interval.

The unit differs by quantity. Statistics over designated addresses resample addresses;
statistics whose variation is driven by which orders happened to fall in the window
resample orders and carry their addresses with them, since addresses named in the same
order share a date, an issuer and usually an operator.

Usage:
  python scripts/compute_uncertainty.py [--out uncertainty.json] [--resamples 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_phenomena import (  # noqa: E402
    DATA_END, MIN_POST_DAYS, israel_seed_dates, load_complete_designated_transfers,
)


def ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def boot_addresses(stat, frame: pd.DataFrame, n: int, rng) -> np.ndarray:
    idx = np.arange(len(frame))
    out = []
    for _ in range(n):
        out.append(stat(frame.iloc[rng.choice(idx, len(idx), replace=True)]))
    return np.array([v for v in out if np.isfinite(v)])


def boot_orders(stat, frame: pd.DataFrame, order_col: str, n: int, rng) -> np.ndarray:
    groups = {o: g for o, g in frame.groupby(order_col)}
    keys = list(groups)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys), replace=True)
        out.append(stat(pd.concat([groups[k] for k in pick])))
    return np.array([v for v in out if np.isfinite(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="uncertainty.json")
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    N = args.resamples

    seeds = israel_seed_dates().set_index("address")
    comp = load_complete_designated_transfers(truncate=False)
    act = pd.concat([comp.assign(addr=comp["from"]), comp.assign(addr=comp["to"])])
    act = act[act["addr"].isin(set(seeds.index))]
    last = act.groupby("addr")["t"].max()

    fl = seeds.join(last.rename("last")).dropna(subset=["last"])
    fl = fl[fl["signed"] <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)]
    fl["days_last_to_signed"] = (fl["signed"] - fl["last"]).dt.days
    out: dict = {"n_resamples": N, "unit": "addresses unless stated"}

    q = {}
    q["median_days_last_activity_to_signed"] = {
        "point": float(fl["days_last_to_signed"].median()),
        "ci_addresses": ci(boot_addresses(lambda f: f["days_last_to_signed"].median(), fl, N, rng)),
        "ci_orders": ci(boot_orders(lambda f: f["days_last_to_signed"].median(), fl, "order", N, rng)),
        "n": int(len(fl))}
    q["share_dormant_30d_at_signing"] = {
        "point": float((fl["days_last_to_signed"] > 30).mean()),
        "ci_addresses": ci(boot_addresses(lambda f: (f["days_last_to_signed"] > 30).mean(), fl, N, rng)),
        "ci_orders": ci(boot_orders(lambda f: (f["days_last_to_signed"] > 30).mean(), fl, "order", N, rng)),
        "n": int(len(fl))}

    enf = pd.read_csv(os.path.join(ROOT, "ch_data", "designated_tether_enforcement.csv"))
    enf["frozen"] = enf["frozen_at"].notna()
    enf = enf.join(seeds["order"], on="address")
    q["share_frozen"] = {
        "point": float(enf["frozen"].mean()),
        "ci_addresses": ci(boot_addresses(lambda f: f["frozen"].mean(), enf, N, rng)),
        "ci_orders": ci(boot_orders(lambda f: f["frozen"].mean(), enf, "order", N, rng)),
        "n": int(len(enf))}
    fz = enf[enf["frozen"]].copy()
    fz["bal"] = fz["balance_at_freeze"].clip(lower=0)
    q["frozen_share_of_inflow"] = {
        "point": float(fz["bal"].sum() / fz["lifetime_in"].sum()),
        "ci_addresses": ci(boot_addresses(lambda f: f["bal"].sum() / max(f["lifetime_in"].sum(), 1e-9), fz, N, rng)),
        "ci_orders": ci(boot_orders(lambda f: f["bal"].sum() / max(f["lifetime_in"].sum(), 1e-9), fz, "order", N, rng)),
        "n": int(len(fz))}
    q["median_days_last_transfer_to_freeze"] = {
        "point": float(fz["last_transfer_to_freeze_days"].median()),
        "ci_addresses": ci(boot_addresses(lambda f: f["last_transfer_to_freeze_days"].median(), fz, N, rng)),
        "ci_orders": ci(boot_orders(lambda f: f["last_transfer_to_freeze_days"].median(), fz, "order", N, rng)),
        "n": int(len(fz))}

    ph = json.load(open(os.path.join(ROOT, "phenomena.json")))
    cp = ph["counterparty_persistence"]
    p_hat, n_cp = cp["share_active_after_order"], cp["n_counterparties"]
    se = float(np.sqrt(p_hat * (1 - p_hat) / n_cp))
    q["counterparty_share_active_after"] = {
        "point": p_hat, "ci_addresses": (p_hat - 1.96 * se, p_hat + 1.96 * se), "n": int(n_cp),
        "note": "normal approximation over independent counterparties"}

    out["quantities"] = q
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in q.items():
        a = v["ci_addresses"]; o = v.get("ci_orders")
        line = f"{k:42} {v['point']:9.4f}  addresses [{a[0]:.4f}, {a[1]:.4f}]"
        if o:
            line += f"  orders [{o[0]:.4f}, {o[1]:.4f}]"
        print(line + f"  n={v['n']}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
