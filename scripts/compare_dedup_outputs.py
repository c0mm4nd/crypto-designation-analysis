#!/usr/bin/env python3
"""Compare the deduplicated complete-network outputs with the deposited ones.

The deposited full_tron_*.json were computed on an export that counted duplicate event rows
(about 6% of rows). Pairs are unaffected, so the connectivity figures should agree closely;
counts and value sums should fall by roughly the duplicate share. This prints both sets side
by side so the manuscript can be updated from the numbers rather than from expectation.

Usage: python scripts/compare_dedup_outputs.py OLD_DIR NEW_DIR
"""
import json, os, sys

old_dir, new_dir = sys.argv[1], sys.argv[2]
def load(d, n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else None

def show(label, a, b, fmt="{:,.4f}"):
    if a is None or b is None: print(f"  {label:58} {a} -> {b}"); return
    ch = (b / a - 1) * 100 if isinstance(a, (int, float)) and a else float('nan')
    print(f"  {label:58} {fmt.format(a):>22} -> {fmt.format(b):>22}  ({ch:+.2f}%)")

for name in ("full_tron_backbone.json", "full_tron_value_removal.json", "full_tron_stranded.json",
             "full_tron_kcore.json", "zero_value_pairs.json", "degree_matched_interval.json"):
    o, n = load(old_dir, name), load(new_dir, name)
    print(f"\n== {name}: old={'yes' if o else 'no'} new={'yes' if n else 'no'}")
    if not (o and n): continue
    if name == "full_tron_backbone.json":
        for k in ("n_addresses", "n_pairs", "n_designated", "baseline_lcc_share", "remove_designated_pct",
                  "remove_top_degree_undesignated_pct", "remove_degree_matched_pct_mean", "remove_random_pct_mean",
                  "designated_median_degree", "network_median_degree", "designated_share_top1pct_degree",
                  "remove_top_degree_undesignated_1000_pct", "remove_top_degree_undesignated_10000_pct"):
            show(k, o.get(k), n.get(k))
    elif name == "full_tron_value_removal.json":
        show("total_value_usdt", o["total_value_usdt"], n["total_value_usdt"], "{:,.0f}")
        for s in o["removals"]:
            for m in ("addresses", "value", "pairs"):
                show(f"{s}.{m}", o["removals"][s][m], n["removals"].get(s, {}).get(m))
    elif name == "full_tron_stranded.json":
        show("total_value_usdt", o["total_value_usdt"], n["total_value_usdt"], "{:,.0f}")
        for s in o["decomposition"]:
            for m in ("incident_pct", "stranded_pct", "incident_usdt", "stranded_usdt"):
                show(f"{s}.{m}", o["decomposition"][s][m], n["decomposition"].get(s, {}).get(m), "{:,.6f}")
    elif name == "zero_value_pairs.json":
        for k in o: show(k, o[k], n.get(k), "{:,.0f}" if isinstance(o[k], int) else "{:,.4f}")
    elif name == "degree_matched_interval.json":
        for k in ("isolated_share_pct_summary", "throughput_pct_summary"):
            for m in ("mean", "designated", "designated_percentile"):
                show(f"{k}.{m}", o[k][m], n[k][m], "{:,.5f}")
            show(f"{k}.ci_low", o[k]["ci"][0], n[k]["ci"][0], "{:,.5f}"); show(f"{k}.ci_high", o[k]["ci"][1], n[k]["ci"][1], "{:,.5f}")
        show("stranded_usdt_exact_draws", str(o.get("stranded_usdt_exact_draws")), str(n.get("stranded_usdt_exact_draws")), "{}")
    elif name == "full_tron_kcore.json":
        for k in ("n_addresses", "n_pairs", "degree_one_share"): show(k, o.get(k), n.get(k))
