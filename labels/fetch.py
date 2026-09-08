"""
Third-party address label fetcher with SQLite cache.

Sources
-------
1. label.web3resear.ch  – aggregates OKLink + Rabby + Allium.  Batch GET, no auth.
2. SlowMist MistTrack   – optional, single-address GET, requires SLOWMIST_API_KEY.

Cache
-----
SQLite file `labels_cache/labels.db`.  Schema:
    (chain, address, source) PRIMARY KEY
    raw_json TEXT               full provider payload
    fetched_at INTEGER           unix epoch
"""

from __future__ import annotations
import os, sys, json, time, sqlite3, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter

# Load .env (prefer python-dotenv when available; else parse manually).
_DOTENV_PATH = Path(__file__).parent.parent / ".env"
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_DOTENV_PATH)
except Exception:
    if _DOTENV_PATH.exists():
        for line in _DOTENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


# Bypass any system HTTP proxy (e.g., local Clash at 127.0.0.1:7890); use a
# shared Session for HTTP/1.1 keep-alive — without pooling, per-request TLS
# handshakes dominate latency on this endpoint.
_SESSION_LOCAL = threading.local()


def _session() -> requests.Session:
    s = getattr(_SESSION_LOCAL, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (wcfrm-label-fetcher)"})
        s.proxies = {"http": "", "https": ""}
        s.trust_env = False
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION_LOCAL.s = s
    return s

W3R_BASE      = "https://label.web3resear.ch"
MIST_BASE     = "https://openapi.misttrack.io/v1"
MIST_KEY      = os.environ.get("SLOWMIST_API_KEY", "")
BATCH_SIZE    = 20
WORKERS       = 8
RETRY_SLEEPS  = ()           # fail fast; pass-2 handles retries
REQUEST_TIMEOUT = 60

CACHE_DIR     = Path(__file__).parent.parent / "labels_cache"
DB_PATH       = CACHE_DIR / "labels.db"


def _db() -> sqlite3.Connection:
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            chain      TEXT NOT NULL,
            address    TEXT NOT NULL,
            source     TEXT NOT NULL,
            raw_json   TEXT,
            fetched_at INTEGER NOT NULL,
            PRIMARY KEY (chain, address, source)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_addr ON labels(chain, address)")
    conn.commit()
    return conn


def _addr_norm(chain: str, addr: str) -> str:
    return addr.lower() if chain == "eth" else addr


def existing_addresses(chain: str, source: str = "web3resear") -> set[str]:
    with _db() as c:
        rows = c.execute(
            "SELECT address FROM labels WHERE chain=? AND source=?", (chain, source)
        ).fetchall()
    return {r[0] for r in rows}


# ─────────────────────────────── web3resear.ch ───────────────────────────────

def _w3r_batch(chain: str, addrs: list[str]) -> dict[str, list] | None:
    """Return {address: [labels]} or None on permanent failure."""
    url = f"{W3R_BASE}/label/query"
    params = {"chain": chain, "addresses": ",".join(addrs)}
    for sleep in RETRY_SLEEPS + (None,):
        try:
            r = _session().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                d = r.json()
                out: dict[str, list] = {}
                for item in d.get("data", []) or []:
                    a = item.get("address")
                    if a is not None:
                        key = a.lower() if chain == "eth" else a
                        out[key] = item.get("labels", []) or []
                return out
            if r.status_code in (429, 500, 502, 503, 504):
                if sleep is None:
                    return None
                time.sleep(sleep)
                continue
            return None
        except requests.exceptions.RequestException:
            if sleep is None:
                return None
            time.sleep(sleep)
    return None


def _pass(chain: str, todo: list[str], batch_size: int, workers: int,
          label: str, progress_every: int) -> set[str]:
    """Run one fetch pass; return set of addresses the server responded about.
    Addresses NOT in the response set are left uncached for re-try in a later
    pass with smaller batches.
    """
    if not todo:
        return set()
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    responded: set[str] = set()
    t0 = time.time()
    conn = _db()
    done = 0

    def work(batch):
        return batch, _w3r_batch(chain, batch)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, b) for b in batches]
        for fut in as_completed(futs):
            batch, result = fut.result()
            now = int(time.time())
            if result is None:
                # request failed; skip so we retry in next pass
                pass
            else:
                rows = []
                for a in batch:
                    if a in result:
                        rows.append((chain, a, "web3resear",
                                     json.dumps(result[a]), now))
                        responded.add(a)
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)", rows)
                    conn.commit()
            done += len(batch)
            if done % progress_every < batch_size or done >= len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [{chain}/{label}] {done:,}/{len(todo):,}  "
                      f"responded={len(responded):,}  "
                      f"{rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    conn.close()
    return responded


def fetch_web3resear(
    chain: str,
    addresses: Iterable[str],
    batch_size: int = BATCH_SIZE,
    workers: int = WORKERS,
    progress_every: int = 500,
):
    """Fetch labels for `addresses` on `chain` ('eth' | 'tron'), cache to SQLite.
    Skips addresses already cached with source='web3resear'.

    Two passes:
      1. Batches of `batch_size` (default 40).
      2. Batches of ceil(batch_size/4) for addresses the server dropped
         in pass 1 (server silently omits unresolvable addresses).
    Addresses still missing after pass 2 are recorded with empty labels to
    avoid perpetual re-fetch on subsequent runs.
    """
    addrs = list({_addr_norm(chain, a) for a in addresses})
    cached = existing_addresses(chain, "web3resear")
    todo = [a for a in addrs if a not in cached]
    print(f"[web3resear:{chain}] total={len(addrs):,}  cached={len(cached):,}  "
          f"todo={len(todo):,}")
    if not todo:
        return

    # ── pass 1 ────────────────────────────────────────────────────────────
    responded = _pass(chain, todo, batch_size, workers, "p1", progress_every)
    missing = [a for a in todo if a not in responded]
    if missing:
        print(f"  [{chain}] pass-1 missing {len(missing):,}; retrying at smaller batch")
        # ── pass 2 ────────────────────────────────────────────────────────
        responded2 = _pass(chain, missing, max(batch_size // 4, 5), workers,
                           "p2", progress_every)
        still_missing = [a for a in missing if a not in responded2]
        if still_missing:
            # mark as empty to prevent infinite retry
            print(f"  [{chain}] pass-2 still missing {len(still_missing):,}; "
                  f"caching as empty")
            conn = _db()
            now = int(time.time())
            rows = [(chain, a, "web3resear", json.dumps([]), now)
                    for a in still_missing]
            conn.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)", rows)
            conn.commit()
            conn.close()


# ─────────────────────────────── SlowMist MistTrack ──────────────────────────

_MIST_COIN = {"eth": "ETH", "tron": "TRX"}


def _mist_one(chain: str, addr: str) -> dict | None:
    if not MIST_KEY:
        return None
    coin = _MIST_COIN[chain]
    url = f"{MIST_BASE}/address_labels"
    params = {"coin": coin, "address": addr, "api_key": MIST_KEY}
    for sleep in RETRY_SLEEPS + (None,):
        try:
            r = _session().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                if sleep is None:
                    return None
                time.sleep(sleep)
                continue
            return None
        except requests.exceptions.RequestException:
            if sleep is None:
                return None
            time.sleep(sleep)
    return None


def fetch_slowmist(
    chain: str,
    addresses: Iterable[str],
    workers: int = 4,
    progress_every: int = 200,
):
    """Single-address queries to MistTrack.  Only use for targeted lookups
    (e.g., addresses classified 'unknown' by web3resear)."""
    if not MIST_KEY:
        print("SLOWMIST_API_KEY not set; skipping SlowMist.")
        return
    addrs = list({_addr_norm(chain, a) for a in addresses})
    cached = existing_addresses(chain, "slowmist")
    todo = [a for a in addrs if a not in cached]
    print(f"[slowmist:{chain}] total={len(addrs):,}  cached={len(cached):,}  todo={len(todo):,}")
    if not todo:
        return
    conn = _db()
    done = 0
    t0 = time.time()

    def work(a):
        return a, _mist_one(chain, a)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, a) for a in todo]
        for fut in as_completed(futs):
            a, d = fut.result()
            now = int(time.time())
            conn.execute(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                (chain, a, "slowmist", json.dumps(d or {}), now),
            )
            done += 1
            if done % progress_every == 0 or done == len(todo):
                conn.commit()
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [{chain}] slowmist {done:,}/{len(todo):,}  "
                      f"{rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    conn.commit()
    conn.close()


# ─────────────────────────────── retrieval ───────────────────────────────────

def load_cached(chain: str, sources: tuple[str, ...] = ("web3resear", "slowmist")) -> dict[str, dict]:
    """Return {address: {source: parsed_json}} for given chain."""
    out: dict[str, dict] = {}
    with _db() as c:
        rows = c.execute(
            f"SELECT address, source, raw_json FROM labels "
            f"WHERE chain=? AND source IN ({','.join('?'*len(sources))})",
            (chain, *sources),
        ).fetchall()
    for addr, src, raw in rows:
        try:
            parsed = json.loads(raw) if raw else None
        except Exception:
            parsed = None
        out.setdefault(addr, {})[src] = parsed
    return out


if __name__ == "__main__":
    # Smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, choices=["eth", "tron"])
    ap.add_argument("--addr",  required=True)
    args = ap.parse_args()
    fetch_web3resear(args.chain, [args.addr])
    print(load_cached(args.chain).get(_addr_norm(args.chain, args.addr)))
