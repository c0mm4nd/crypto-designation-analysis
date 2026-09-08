#!/usr/bin/env python3
"""Map the designated TRON addresses to the identifiers used in the complete-network export.

The complete network carries addresses as cityHash64 of the 20-byte address, so the
designated addresses have to be converted the same way to be located in it. Base58Check
decoding gives the 21-byte payload whose first byte is the TRON prefix 0x41; the export
strips that prefix, so the hashed string is the remaining 20 bytes in lower-case hex.

cityHash64 is ClickHouse's own implementation, so the hashes are computed by ClickHouse
rather than reimplemented here.

Usage:
  python scripts/designated_hashes.py [--out /tmp/designated_hash.tsv]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.train_v2 import load_seed_addresses  # noqa: E402

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58_to_hex20(addr: str) -> str | None:
    """Base58Check TRON address to the 20-byte body, lower-case hex, without the 0x41 prefix."""
    n = 0
    for c in addr:
        n = n * 58 + B58.index(c)
    raw = n.to_bytes(25, "big")
    return raw[1:21].hex() if raw[0] == 0x41 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=os.path.join(ROOT, "IsraelAddrs.xlsx"))
    ap.add_argument("--out", default="/tmp/designated_hash.tsv")
    ap.add_argument("--container", default="clickhouse-analyticaldb-1")
    args = ap.parse_args()

    hexes = sorted({h for a in load_seed_addresses(args.seeds, "tron") if (h := b58_to_hex20(a))})
    print(f"{len(hexes)} designated addresses converted to hex")

    lst = ",".join(f"'{h}'" for h in hexes)
    query = f"SELECT a, cityHash64(a) FROM (SELECT arrayJoin([{lst}]) AS a)"
    cid = subprocess.run(["docker", "ps", "-qf", f"name={args.container}"],
                         capture_output=True, text=True).stdout.strip().split("\n")[0]
    res = subprocess.run(
        ["docker", "exec", "-i", cid, "clickhouse-client",
         "--user", os.environ.get("CH_USER", "w3r"),
         "--password", os.environ["CH_PASSWORD"],
         "--format", "TabSeparated", "--query", query],
        capture_output=True, text=True, check=True)
    with open(args.out, "w") as f:
        f.write(res.stdout)
    print(f"wrote {len(res.stdout.strip().splitlines())} hashes to {args.out}")


if __name__ == "__main__":
    main()
