#!/usr/bin/env python3
"""Rebuild the complete TRON USDT network, deduplicated, streaming from ClickHouse over HTTP.

The deposited export ran clickhouse-client inside Docker on the node and wrote 32 bucket files
that were then loaded by the analysis scripts. This runner does the same work from any machine
that can reach the node's HTTP port, and it does it without ever holding the raw export on
disk: each bucket is streamed to a temporary file, mapped onto sorted node indices, appended to
preallocated array files, and deleted. The result is the `graph_cache` the analysis scripts
already know how to load (`_nodes.npy`, `_si.npy`, `_di.npy`) plus `_val.npy`, which the value
scripts use instead of re-reading bucket files when it is present.

Every query deduplicates on (transactionHash, logIndex) first, because the node's event table
holds duplicate rows for re-ingested blocks; see export_full_tron_network.sh.

Buckets are by cityHash64(sender) % 64 and are very uneven, because a handful of exchange
senders carry a large share of all pairs. Each bucket is retried until its size is a whole
number of 28-byte records. Progress is recorded per bucket, so the run resumes where it stopped.

Usage:
  CH_URL=http://host:8123 CH_AUTH=user:pass python scripts/rerun_complete_network_local.py \
      --out tron_full2 [--buckets 64] [--concurrency 2] [--expected-pairs 760000000]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

USDT = "a614f803b6fd780986a42c78ec9c7f77e6ded13c"
TRANSFER = "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
CUT_MS = 1735689600000
DT = np.dtype([("f", "<u8"), ("t", "<u8"), ("c", "<u4"), ("v", "<f8")])
SETTINGS = "max_memory_usage=14000000000&max_bytes_before_external_group_by=4000000000&max_threads=8"


def ch(url: str, auth: str, sql: str, out: str) -> bool:
    r = subprocess.run(["curl", "-s", "--max-time", "7200", "-u", auth, f"{url}/?{SETTINGS}",
                        "--data-binary", sql, "-o", out, "-w", "%{http_code}"], capture_output=True, text=True)
    return r.stdout.strip() == "200"


def node_query() -> str:
    base = (f"FROM tron.events WHERE address='{USDT}' AND topic0='{TRANSFER}' AND topic2 IS NOT NULL "
            f"AND blockTimestamp < {CUT_MS}")
    return (f"SELECT h FROM (SELECT cityHash64(assumeNotNull(substring(topic1,25,40))) AS h {base} "
            f"UNION ALL SELECT cityHash64(assumeNotNull(substring(topic2,25,40))) AS h {base}) "
            f"GROUP BY h ORDER BY h FORMAT RowBinary")


def bucket_query(b: int, nb: int) -> str:
    return (f"SELECT cityHash64(f) AS fi, cityHash64(t) AS ti, toUInt32(count()) AS cnt, "
            f"sum(reinterpretAsUInt64(reverse(unhex(substring(d,49,16))))/1000000.) AS val "
            f"FROM (SELECT any(assumeNotNull(substring(topic1,25,40))) AS f, any(assumeNotNull(substring(topic2,25,40))) AS t, "
            f"any(assumeNotNull(data)) AS d FROM tron.events WHERE address='{USDT}' AND topic0='{TRANSFER}' "
            f"AND topic2 IS NOT NULL AND blockTimestamp < {CUT_MS} "
            f"AND cityHash64(assumeNotNull(substring(topic1,25,40))) % {nb} = {b} "
            f"GROUP BY transactionHash, logIndex) GROUP BY fi, ti FORMAT RowBinary")


def fetch_bucket(url, auth, b, nb, tmpdir):
    out = os.path.join(tmpdir, f"bucket_{b}.bin")
    for attempt in range(1, 6):
        t0 = time.time()
        ok = ch(url, auth, bucket_query(b, nb), out)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        if ok and sz > 0 and sz % DT.itemsize == 0:
            print(f"  bucket {b:2d}: {sz // DT.itemsize:>11,} pairs in {time.time() - t0:5.0f} s", flush=True)
            return out
        print(f"  bucket {b:2d}: attempt {attempt} failed (ok={ok}, size={sz}); retrying", flush=True)
        time.sleep(60 * attempt)
    raise RuntimeError(f"bucket {b} failed")


def fix_header_and_truncate(path: str, n: int, dtype):
    """Shrink a preallocated .npy to its filled length by rewriting the header in place."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
        reader(f)
        hlen = f.tell()
    import io
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": (n,)})
    hdr = buf.getvalue()
    assert len(hdr) == hlen, f"header length changed ({len(hdr)} vs {hlen}); refusing to rewrite {path}"
    with open(path, "r+b") as f:
        f.write(hdr)
        f.truncate(hlen + n * np.dtype(dtype).itemsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tron_full2")
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--expected-pairs", type=int, default=760_000_000)
    args = ap.parse_args()
    url, auth = os.environ["CH_URL"], os.environ["CH_AUTH"]
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "graph_cache")
    tmpdir = os.path.join(args.out, "tmp"); os.makedirs(tmpdir, exist_ok=True)

    # 1. sorted node list
    nodes_path = cache + "_nodes.npy"
    if not os.path.exists(nodes_path):
        raw = os.path.join(tmpdir, "nodes.bin")
        print("fetching sorted node list ...", flush=True)
        t0 = time.time()
        for attempt in range(1, 6):
            if ch(url, auth, node_query(), raw) and os.path.getsize(raw) % 8 == 0 and os.path.getsize(raw) > 0:
                break
            print(f"  node list attempt {attempt} failed; retrying", flush=True); time.sleep(120 * attempt)
        nodes = np.fromfile(raw, dtype="<u8")
        assert np.all(np.diff(nodes) > 0), "node list is not strictly sorted"
        np.save(nodes_path, nodes); os.remove(raw)
        print(f"  {len(nodes):,} addresses in {time.time() - t0:.0f} s", flush=True)
    nodes = np.load(nodes_path)
    n = len(nodes)

    # 2. preallocated array files, filled bucket by bucket
    N = args.expected_pairs
    paths = {k: cache + f"_{k}.npy" for k in ("si", "di", "val")}
    prog = os.path.join(args.out, "progress.txt")
    done = set(); at = 0
    if os.path.exists(prog):
        for line in open(prog):
            b, cnt = line.split(); done.add(int(b)); at += int(cnt)
    if not all(os.path.exists(p) for p in paths.values()):
        for k, dt in (("si", np.int32), ("di", np.int32), ("val", np.float32)):
            np.lib.format.open_memmap(paths[k], mode="w+", dtype=dt, shape=(N,)).flush()
        done, at = set(), 0; open(prog, "w").close()
    si = np.load(paths["si"], mmap_mode="r+"); di = np.load(paths["di"], mmap_mode="r+"); val = np.load(paths["val"], mmap_mode="r+")
    print(f"resuming with {len(done)} buckets done, {at:,} pairs written", flush=True)

    todo = [b for b in range(args.buckets) if b not in done]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {}
        pending = list(todo)
        while pending or futures:
            while pending and len(futures) < args.concurrency:
                b = pending.pop(0); futures[ex.submit(fetch_bucket, url, auth, b, args.buckets, tmpdir)] = b
            # consume in submission order so the arrays stay bucket-ordered
            fut = next(iter(futures)); b = futures.pop(fut); path = fut.result()
            a = np.fromfile(path, dtype=DT); k = len(a)
            assert at + k <= N, "expected-pairs too small; rerun with a larger value"
            fi = np.searchsorted(nodes, a["f"]); ti = np.searchsorted(nodes, a["t"])
            assert fi.max() < n and ti.max() < n and np.all(nodes[fi] == a["f"]) and np.all(nodes[ti] == a["t"]), f"unknown address in bucket {b}"
            si[at:at + k] = fi; di[at:at + k] = ti; val[at:at + k] = a["v"]
            at += k; del a, fi, ti
            os.remove(path)
            with open(prog, "a") as f:
                f.write(f"{b} {k}\n")
            print(f"  bucket {b:2d} stored; {at:,} pairs so far", flush=True)
    si.flush(); di.flush(); val.flush(); del si, di, val
    print("truncating arrays to", at, "rows", flush=True)
    for k, dt in (("si", np.int32), ("di", np.int32), ("val", np.float32)):
        fix_header_and_truncate(paths[k], at, dt)
    print(f"done: {n:,} addresses, {at:,} unique directed pairs, {np.load(paths['val'], mmap_mode='r').astype(np.float64).sum()/1e12:.3f} trillion USDT", flush=True)
    open(os.path.join(args.out, "COMPLETE"), "w").write(f"{n} {at}\n")


if __name__ == "__main__":
    main()
