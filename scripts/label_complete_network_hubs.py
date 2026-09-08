#!/usr/bin/env python3
"""Who the highest-activity undesignated addresses of the complete TRON USDT network are.

The removal test shows that four hundred undesignated addresses strand 3.4 billion USDT
between addresses that survive their removal, where the four hundred designated addresses
strand 590. That result is easy to over-read: it invites the reading that enforcement is
missing the "real" infrastructure, when the addresses in question may simply be exchanges,
which no counter-terrorism designation would ever name. This script settles the question by
naming them, so the manuscript can state what they are instead of implying it.

It ranks undesignated addresses by USDT transfer count over the whole network to the cut,
which is the cheap proxy the node can compute without materialising 213 million group keys,
and matches the top of that ranking against the third-party entity label cache. The labels
themselves are used under their providers' terms and are not redistributed: the deposited
artefact carries the ranking and the aggregate composition only.

The query is bucketed by cityHash64 of the address modulo 32, as in
scripts/export_full_tron_network.sh, because a single GROUP BY over every address exhausts
memory on the node.

Usage:
  CH_URL=http://host:8123 CH_AUTH=user:password \
      python scripts/label_complete_network_hubs.py [--top 400] [--out complete_network_hubs.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
USDT = "a614f803b6fd780986a42c78ec9c7f77e6ded13c"
TRANSFER = "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
CUT_MS = 1735689600000  # 2025-01-01T00:00:00Z


def tron_hex_to_b58(hex40: str) -> str:
    payload = bytes.fromhex("41" + hex40)
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n = int.from_bytes(payload + chk, "big")
    out = ""
    while n > 0:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return out


def query(url: str, auth: str, sql: str) -> str:
    cmd = ["curl", "-s", "--max-time", "3600", "-u", auth, url, "--data-binary", sql]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    return r.stdout


def fetch(url: str, auth: str, per_bucket: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for b in range(32):
        sql = f"""SELECT hex(a) AS addr, count() AS n FROM (
  SELECT assumeNotNull(substring(topic1,25,40)) AS a FROM tron.events
   WHERE address='{USDT}' AND topic0='{TRANSFER}' AND topic2 IS NOT NULL AND blockTimestamp < {CUT_MS}
     AND cityHash64(assumeNotNull(substring(topic1,25,40))) % 32 = {b}
  UNION ALL
  SELECT assumeNotNull(substring(topic2,25,40)) AS a FROM tron.events
   WHERE address='{USDT}' AND topic0='{TRANSFER}' AND topic2 IS NOT NULL AND blockTimestamp < {CUT_MS}
     AND cityHash64(assumeNotNull(substring(topic2,25,40))) % 32 = {b}
) GROUP BY a ORDER BY n DESC LIMIT {per_bucket}"""
        out = query(url + "/?max_memory_usage=18000000000&max_bytes_before_external_group_by=5000000000",
                    auth, sql)
        for line in out.splitlines():
            if not line.strip():
                continue
            h, n = line.split()
            # hex() is applied to a value that is already a hex string, so decode once
            rows.append((bytes.fromhex(h).decode().lower() if len(h) == 80 else h.lower(), int(n)))
        print(f"  bucket {b}: {len(rows)} rows so far", flush=True)
    return rows


def entity_of(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        recs = json.loads(raw)
    except Exception:
        return None
    for e in recs:
        lab = e.get("label") or ""
        if lab.startswith("cex:"):
            return lab.split(":", 1)[1]
    for e in recs:
        nt = e.get("name_tag") or ""
        if nt:
            return nt.split(".")[0].replace("Exchange: ", "").strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--per-bucket", type=int, default=60)
    ap.add_argument("--out", default="complete_network_hubs.json")
    ap.add_argument("--rows", help="TSV of hex address and transfer count, to skip the query")
    args = ap.parse_args()

    if args.rows:
        rows = []
        for line in open(args.rows):
            h, n = line.split()
            rows.append((bytes.fromhex(h).decode().lower() if len(h) == 80 else h.lower(), int(n)))
    else:
        url = os.environ.get("CH_URL")
        auth = os.environ.get("CH_AUTH")
        if not url or not auth:
            raise SystemExit("set CH_URL and CH_AUTH, or pass --rows")
        rows = fetch(url, auth, args.per_bucket)
    rows.sort(key=lambda r: -r[1])

    sys.path.insert(0, str(find("scripts").parent))
    from scripts.compute_phenomena import israel_seed_dates  # noqa: E402
    designated = set(israel_seed_dates()["address"])

    top, seen = [], set()
    for h, n in rows:
        a = tron_hex_to_b58(h)
        if a in designated or a in seen:
            continue
        seen.add(a)
        top.append((a, n))
        if len(top) == args.top:
            break

    labels: dict[str, str] = {}
    try:
        con = sqlite3.connect(find("labels_cache/labels.db"))
        labels = {a: r for a, r in con.execute("select address, raw_json from labels where chain='tron'")}
    except FileNotFoundError:
        print("label cache absent: reporting the ranking without entity composition")

    named = [(a, entity_of(labels.get(a)), n) for a, n in top]
    comp = Counter(e for _, e, _ in named if e)
    out = {
        "description": ("The highest-activity undesignated addresses of the complete TRON USDT network "
                        "to 1 January 2025, ranked by USDT transfer count, with the number carrying a "
                        "third-party entity label and the composition of those labels. Per-address "
                        "labels are not redistributed; the addresses are public blockchain data."),
        "ranking": "number of USDT transfers before the cut, sender and receiver side combined",
        "n_addresses": len(named),
        "n_with_third_party_label": int(sum(comp.values())),
        "label_composition": dict(comp.most_common()),
        "addresses_ranked": [a for a, _, _ in named],
        "transfers": [n for _, _, n in named],
    }
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{out['n_with_third_party_label']} of {out['n_addresses']} carry a label: {out['label_composition']}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
