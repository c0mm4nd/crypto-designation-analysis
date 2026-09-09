#!/usr/bin/env python3
"""Robustness of the designation event study.

The main analysis aligns each designated address on the date its seizure order was signed
and compares the twenty-six weeks either side. Three things about that design need to be
checked rather than asserted, and this script checks them.

1. The event date. An order is signed, published a few days later, and the addresses are
   usually frozen by the issuer of the stablecoin at some point around both. Which of the
   three an address's counterparties could actually observe differs, and for the order that
   dominates the aggregate the gap between signature and publication is over a month. We
   recompute the series on all three definitions.

2. The pre-trend. A collapse that begins before the event is not caused by it. We report
   the aggregate series relative to its first week rather than to its own pre-period mean,
   and the ordinary-least-squares slope of log weekly volume over the pre-period.

3. What carries the result. Seven orders fall inside the observation window and one of them
   accounts for three quarters of the designated volume, so the effective sample is orders,
   not addresses. We report the headline ratio leaving out one order at a time and a
   confidence interval from resampling orders with replacement.

We also report the per-address statistics for the designated and placebo groups in the same
form, because an aggregate ratio can be dominated by a handful of large addresses on either
side, and we repeat the designated series using the crawled histories instead of the
complete ones, since the placebo can only be built from crawled histories and those are
truncated newest-first by the collector's page limit.

Usage:
  python scripts/compute_event_robustness.py [--out event_robustness.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402
sys.path.insert(0, ROOT)

from scripts.compute_phenomena import (  # noqa: E402
    DATA_END, MIN_POST_DAYS, VALUE_CAP, W, israel_seed_dates,
    load_complete_designated_transfers,
)


def weekly(act: pd.DataFrame, event: dict[str, pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
    """Weekly USDT volume and transfer count relative to each address's own event date."""
    a = act[act["addr"].isin(event)]
    if a.empty:
        return np.zeros(2 * W + 1), np.zeros(2 * W + 1)
    off = ((a["t"] - a["addr"].map(event)).dt.days // 7).to_numpy()
    keep = (off >= -W) & (off <= W)
    idx = (off[keep] + W).astype(int)
    vol = np.bincount(idx, weights=a["value"].to_numpy()[keep], minlength=2 * W + 1)
    cnt = np.bincount(idx, minlength=2 * W + 1)
    return vol, cnt


def summarise(vol: np.ndarray, cnt: np.ndarray) -> dict:
    pre = vol[:W]
    post = vol[W + 7:]
    slope = float(np.polyfit(np.arange(W), np.log10(np.maximum(pre, 1.0)), 1)[0]) if (pre > 0).any() else float("nan")
    return {
        "volume_usdt": vol.tolist(), "transfers": cnt.tolist(),
        "pre_mean_weekly_volume": float(pre.mean()),
        "post_weeks_7_26_mean_volume": float(post.mean()),
        "ratio_post_to_pre": float(post.mean() / pre.mean()) if pre.mean() > 0 else float("nan"),
        "pre_trend_log10_per_week": slope,
        "pre_first_week_volume": float(vol[0]), "pre_last_week_volume": float(vol[W - 1]),
        "pre_last_over_first": float(vol[W - 1] / vol[0]) if vol[0] > 0 else float("nan"),
    }


def per_address(act: pd.DataFrame, event: dict[str, pd.Timestamp]) -> dict:
    """Share of addresses transacting after the event, and each address's post share of volume."""
    a = act[act["addr"].isin(event)].copy()
    a["ev"] = a["addr"].map(event)
    a["post"] = a["t"] > a["ev"]
    g = a.groupby("addr")
    any_post = g["post"].any()
    share_post = g.apply(lambda x: x.loc[x["post"], "value"].sum() / max(x["value"].sum(), 1e-9), include_groups=False)
    return {"n": int(len(any_post)), "share_any_post_activity": float(any_post.mean()),
            "median_post_share_of_volume": float(share_post.median()),
            "mean_post_share_of_volume": float(share_post.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="event_robustness.json")
    args = ap.parse_args()

    seeds = israel_seed_dates().set_index("address")
    comp = load_complete_designated_transfers()
    assert comp is not None, "complete designated histories are required"
    act = pd.concat([comp.assign(addr=comp["from"]), comp.assign(addr=comp["to"])])
    act = act[act["addr"].isin(set(seeds.index))]

    # Tether freeze dates for the designated addresses
    bl = pd.read_csv(find("ch_data/designated_blacklist_match.csv"))
    frozen = dict(zip(bl["address"], pd.to_datetime(bl["blacklisted_at"], errors="coerce")))

    out: dict = {"window_weeks": W, "data_end": str(DATA_END.date()), "value_cap_usdt": VALUE_CAP}

    # ---- 1. three definitions of the event date
    defs = {}
    observed = set(act["addr"].unique())
    for name, col in (("signed", "signed"), ("published", "published")):
        ev = {a: pd.Timestamp(r[col]) for a, r in seeds.iterrows()
              if a in observed and pd.notna(r[col])
              and pd.Timestamp(r[col]) <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)}
        v, c = weekly(act, ev)
        defs[name] = {"n_addresses": len(ev), **summarise(v, c), "per_address": per_address(act, ev)}
    ev_f = {a: d for a, d in frozen.items()
            if a in seeds.index and pd.notna(d) and d <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)}
    if ev_f:
        v, c = weekly(act, ev_f)
        defs["frozen"] = {"n_addresses": len(ev_f), **summarise(v, c), "per_address": per_address(act, ev_f)}
    out["event_date_definitions"] = defs

    # ---- 2. leave-one-order-out and cluster-by-order bootstrap, on the signing date
    ev = {a: pd.Timestamp(r["signed"]) for a, r in seeds.iterrows()
          if a in observed and pd.notna(r["signed"])
          and pd.Timestamp(r["signed"]) <= DATA_END - pd.Timedelta(days=MIN_POST_DAYS)}
    addr_order = {a: seeds.loc[a, "order"] for a in ev}
    order_list = sorted(set(addr_order.values()))
    loo = {}
    for o in order_list:
        sub = {a: d for a, d in ev.items() if addr_order[a] != o}
        if not sub:
            continue
        v, c = weekly(act, sub)
        loo[o] = {"n_addresses_dropped": sum(1 for a in ev if addr_order[a] == o),
                  "ratio_post_to_pre": summarise(v, c)["ratio_post_to_pre"]}
    out["leave_one_order_out"] = loo

    # ---- 2b. one order at a time, and the two enforcement regimes separately.
    # The pooled ratio is dominated by the order that moved the most, and that order predates
    # the point at which the issuer began freezing designated addresses at all. Splitting on
    # that date says whether the collapse looks different once enforcement was in place. The
    # post-October-2023 group is small and its addresses have short post-windows, so this is
    # descriptive.
    REGIME = pd.Timestamp("2023-10-01")
    per_order = {}
    for o in order_list:
        sub = {a: d for a, d in ev.items() if addr_order[a] == o}
        if not sub:
            continue
        v, c = weekly(act, sub)
        sm = summarise(v, c)
        v4 = float(v[W - 4:W].mean())
        per_order[o] = {"n_addresses": len(sub), "signed": str(min(sub.values()).date()),
                        "pre_mean_weekly_volume": sm["pre_mean_weekly_volume"],
                        "last_4_pre_weeks_mean_volume": v4,
                        "share_of_pre_volume_in_last_4_weeks": float(v4 / sm["pre_mean_weekly_volume"])
                        if sm["pre_mean_weekly_volume"] else 0.0,
                        "ratio_post_to_pre": sm["ratio_post_to_pre"]}
    out["per_order"] = per_order
    regimes = {}
    for name, keep in (("before_october_2023", lambda d: d < REGIME),
                       ("from_october_2023", lambda d: d >= REGIME)):
        sub = {a: d for a, d in ev.items() if keep(d)}
        if not sub:
            continue
        v, c = weekly(act, sub)
        regimes[name] = {"n_addresses": len(sub),
                         "n_orders": len({addr_order[a] for a in sub}), **summarise(v, c)}
    out["enforcement_regimes"] = regimes
    for k, r in regimes.items():
        print(f"  regime {k}: {r['n_orders']} orders, {r['n_addresses']} addresses, "
              f"ratio {r['ratio_post_to_pre']:.4f}")

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        pick = rng.choice(order_list, size=len(order_list), replace=True)
        sub = {}
        for i, o in enumerate(pick):
            for a, d in ev.items():
                if addr_order[a] == o:
                    sub[f"{a}#{i}"] = d
        a2 = act[act["addr"].isin({a.split('#')[0] for a in sub})].copy()
        rows = []
        for i, o in enumerate(pick):
            members = [a for a in ev if addr_order[a] == o]
            r = a2[a2["addr"].isin(members)].copy()
            r["addr"] = r["addr"] + f"#{i}"
            rows.append(r)
        a3 = pd.concat(rows) if rows else a2
        v, c = weekly(a3, sub)
        boot.append(summarise(v, c)["ratio_post_to_pre"])
    boot = np.array([b for b in boot if np.isfinite(b)])
    out["cluster_bootstrap_ratio_post_to_pre"] = {
        "point": defs["signed"]["ratio_post_to_pre"], "n_orders": len(order_list),
        "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
        "n_resamples": int(len(boot))}

    # ---- 3. the placebo, reported in exactly the same form
    # The placebo can only be built from the crawled graph, whose per-address histories are
    # truncated newest-first by the collector's page limit. We therefore also recompute the
    # designated series from the crawled graph, so that the two arms are cut the same way.
    from scripts.compute_phenomena import load_edges
    df = load_edges("israel_tron_usdt_edges_2hop.csv")
    df = df[(df["value"] > 0) & (df["value"] <= VALUE_CAP) & (df["t"] < DATA_END)]
    seed_set = set(seeds.index)
    touch = df[df["from"].isin(seed_set) | df["to"].isin(seed_set)]
    hop1 = sorted((set(touch["from"]) | set(touch["to"])) - seed_set)
    h = df[df["from"].isin(hop1) | df["to"].isin(hop1)]
    h_act = pd.concat([h.assign(addr=h["from"]), h.assign(addr=h["to"])])
    h_act = h_act[h_act["addr"].isin(set(hop1))]
    span = h_act.groupby("addr")["t"].agg(["min", "max"])
    rng2 = np.random.default_rng(42)
    dates = np.array(sorted(ev.values()))
    placebo: dict[str, pd.Timestamp] = {}
    for addr, r in span.sample(frac=1.0, random_state=42).iterrows():
        d0 = pd.Timestamp(rng2.choice(dates))
        if r["min"] <= d0 and r["max"] >= d0 - pd.Timedelta(weeks=W):
            placebo[addr] = d0
        if len(placebo) >= 3000:
            break
    pv, pc = weekly(h_act, placebo)
    crawled_act = pd.concat([df.assign(addr=df["from"]), df.assign(addr=df["to"])])
    crawled_act = crawled_act[crawled_act["addr"].isin(seed_set)]
    dv, dc = weekly(crawled_act, ev)
    out["like_for_like"] = {
        "designated_crawled": {"n_addresses": len(ev), **{k: v for k, v in summarise(dv, dc).items() if k not in ("volume_usdt", "transfers")},
                               "per_address": per_address(crawled_act, ev)},
        "placebo_crawled": {"n_addresses": len(placebo), **{k: v for k, v in summarise(pv, pc).items() if k not in ("volume_usdt", "transfers")},
                            "per_address": per_address(h_act, placebo)},
    }
    # A placebo matched only on having been active before the event does not control for
    # size, and designated addresses move far more than a typical counterparty, so some of
    # the contrast could be regression to the mean after selecting on past volume. We
    # therefore match each designated address to an undesignated counterparty with the
    # nearest pre-event volume at the same event date, without replacement, and repeat with
    # five different orderings.
    pre_vol = {}
    for addr, ev_d in list(ev.items()):
        r = crawled_act[(crawled_act["addr"] == addr)]
        w = r[(r["t"] >= ev_d - pd.Timedelta(weeks=W)) & (r["t"] < ev_d)]["value"].sum()
        if w > 0:
            pre_vol[addr] = float(w)
    cand = {}
    for addr, d0 in placebo.items():
        r = h_act[h_act["addr"] == addr]
        w = r[(r["t"] >= d0 - pd.Timedelta(weeks=W)) & (r["t"] < d0)]["value"].sum()
        if w > 0:
            cand[addr] = float(w)
    cand_addr = np.array(list(cand)); cand_v = np.array([cand[a] for a in cand_addr])
    order_v = np.argsort(cand_v); cand_addr, cand_v = cand_addr[order_v], cand_v[order_v]
    matched_runs = []
    for seed in range(5):
        rr = np.random.default_rng(seed)
        used = set(); sel = {}; matched_to = []
        for addr in rr.permutation(list(pre_vol)):
            j = int(np.searchsorted(cand_v, pre_vol[addr]))
            for off in range(len(cand_addr)):
                for k in (j + off, j - off - 1):
                    if 0 <= k < len(cand_addr) and cand_addr[k] not in used:
                        used.add(cand_addr[k]); sel[cand_addr[k]] = placebo[cand_addr[k]]
                        matched_to.append(addr)
                        break
                else:
                    continue
                break
        mv, mc = weekly(h_act, sel)
        # match quality: how far apart the matched pre-event volumes are, in decades
        q = np.abs(np.log10(np.array([cand[a] for a in sel.keys()]) /
                            np.array([pre_vol[a] for a in matched_to])))
        matched_runs.append({"n": len(sel), "match_log10_abs_median": float(np.median(q)),
                             "match_log10_abs_p90": float(np.percentile(q, 90)),
                             **{k: v for k, v in summarise(mv, mc).items()
                                if k not in ("volume_usdt", "transfers")},
                             "per_address": per_address(h_act, sel)})
    # the designated comparators must be computed on the addresses that were actually
    # matched, not on the full in-window set
    matched_designated = {a: ev[a] for a in matched_to}
    dv2, dc2 = weekly(crawled_act, matched_designated)
    out["volume_matched_placebo"] = {
        "n_designated_with_pre_volume": len(pre_vol),
        "designated_matched_subset": {"n": len(matched_designated),
                                      **{k: v for k, v in summarise(dv2, dc2).items()
                                         if k not in ("volume_usdt", "transfers")},
                                      "per_address": per_address(crawled_act, matched_designated)},
        "ratio_post_to_pre": [r["ratio_post_to_pre"] for r in matched_runs],
        "share_any_post_activity": [r["per_address"]["share_any_post_activity"] for r in matched_runs],
        "runs": matched_runs}
    vm = out["volume_matched_placebo"]
    print(f"[volume-matched placebo] ratio {min(vm['ratio_post_to_pre']):.3f}-{max(vm['ratio_post_to_pre']):.3f} "
          f"any-post {min(vm['share_any_post_activity']):.2f}-{max(vm['share_any_post_activity']):.2f} "
          f"(designated 0.0285 / 0.39)", flush=True)

    lf = out["like_for_like"]
    print(f"[like-for-like, crawled histories] designated post/pre={lf['designated_crawled']['ratio_post_to_pre']:.4f} "
          f"any-post={lf['designated_crawled']['per_address']['share_any_post_activity']:.2f} | "
          f"placebo post/pre={lf['placebo_crawled']['ratio_post_to_pre']:.4f} "
          f"any-post={lf['placebo_crawled']['per_address']['share_any_post_activity']:.2f}", flush=True)

    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in defs.items():
        print(f"[{k:9}] n={v['n_addresses']:3d} post/pre={v['ratio_post_to_pre']:.4f} "
              f"pre-trend={v['pre_trend_log10_per_week']:+.3f} dex/week  "
              f"pre wk-1/wk-26={v['pre_last_over_first']:.2f}  "
              f"any post activity={v['per_address']['share_any_post_activity']:.2f}", flush=True)
    b = out["cluster_bootstrap_ratio_post_to_pre"]
    print(f"cluster-by-order bootstrap: {b['point']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}] over {b['n_orders']} orders")
    print(f"leave-one-out range: {min(x['ratio_post_to_pre'] for x in loo.values()):.4f} "
          f"to {max(x['ratio_post_to_pre'] for x in loo.values()):.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
