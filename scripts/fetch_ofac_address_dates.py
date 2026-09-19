#!/usr/bin/env python3
"""Date on which each OFAC-listed digital-currency address was added to the SDN List.

An entity's SDN publication date is not the date its addresses were listed: addresses are often added to an
existing entry years later. The OFAC Sanctions List Service publishes a delta file for every list update from
September 2023, in which each feature carries an action attribute. This script enumerates every publication in
the change history, downloads each delta, and records the first publication that adds a digital-currency address
feature, together with the entity's own publication date. Addresses not added in any delta since the history
begins, or already present in the baseline snapshot, are dated by the entity's listing and flagged.

Usage:
  python scripts/fetch_ofac_address_dates.py [--out ofac_listing_dates.json] [--cache DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import find, out_path  # noqa: E402

SLS = "https://sanctionslistservice.ofac.treas.gov"
PROGRAMMES = ("ofac_iran", "ofac_russia_ukraine", "ofac_terrorist_financing")


def get(url, out=None, tries=10):
    for k in range(tries):
        r = subprocess.run(["curl", "-sS", "-L", "-m", "180", "-w", "%{http_code}", "-o", out or "-", url], capture_output=True, text=True)
        body, code = (r.stdout[:-3], r.stdout[-3:]) if out is None else ("", r.stdout[-3:])
        if code == "200":
            return body
        time.sleep((20 if code == "429" else 3) * (k + 1))
    raise RuntimeError(f"{url}: {code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ofac_listing_dates.json")
    ap.add_argument("--cache", default=os.environ.get("OFAC_DELTA_CACHE", "ofac_delta"))
    ap.add_argument("--entities", default=os.environ.get("OFAC_ENTITY_CACHE", "ofac"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    pubs = []
    y, m = 2023, 9
    now = time.gmtime()
    while (y, m) <= (now.tm_year, now.tm_mon):
        pubs += json.loads(get(f"{SLS}/changes/history/{y}/{m:02d}"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    pubs = sorted({p["publicationID"]: p for p in pubs}.values(), key=lambda p: p["publicationID"])
    print(f"{len(pubs)} publications from {pubs[0]['datePublished'][:10]} to {pubs[-1]['datePublished'][:10]}", flush=True)

    def fetch(p):
        path = os.path.join(args.cache, f"{p['publicationID']}.xml")
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            get(f"{SLS}/changes/{p['publicationID']}", out=path)
        return path
    with ThreadPoolExecutor(2) as ex:
        paths = list(ex.map(fetch, pubs))
    # a feature added to an existing entity carries its own action; a newly added entity's features carry none
    feat = re.compile(r'<feature id="(\d+)"(?: action="(\w+)")?>\s*<type featureTypeId="\d+">Digital Currency Address - (\w+)</type>.*?<value>([^<]+)</value>', re.S)
    ent = re.compile(r'<entity id="(\d+)"(?: action="(\w+)")?')
    added = {}
    baseline = pubs[0]["publicationID"]  # the first publication is a full snapshot of the list, not a day's changes
    for p, path in zip(pubs, paths):
        s = open(path).read()
        d = re.search(r"<datePublished>(\d{4}-\d{2}-\d{2})", s)
        d = d.group(1) if d else p["datePublished"][:10]
        # walk entities in order so each feature knows its entity
        pos = [(m.start(), m.group(1), m.group(2)) for m in ent.finditer(s)]
        for f in feat.finditer(s):
            eid, eact = max((e for e in pos if e[0] < f.start()), default=(0, None, None))[1:]
            if (f.group(2) == "add" or (f.group(2) is None and eact == "add")) and f.group(4) not in added:
                added[f.group(4)] = {"date_added": d, "publication_id": p["publicationID"], "entity_id": eid, "currency": f.group(3)}
    print(f"{len(added)} digital-currency addresses added in the change history", flush=True)

    entities = {}
    for f in os.listdir(args.entities):
        if f.endswith(".xml"):
            s = open(os.path.join(args.entities, f)).read()
            m = re.search(r'<sanctionsList[^>]*datePublished="(\d{4}-\d{2}-\d{2})"[^>]*>([^<]*)</sanctionsList>', s)
            name = re.search(r"<formattedFullName>([^<]*)</formattedFullName>", s)
            entities[f[:-4]] = {"date_published": m.group(1) if m else None, "list": m.group(2) if m else None, "name": name.group(1) if name else None}
    rows = {}
    for prog in PROGRAMMES:
        for r in csv.DictReader(open(find(f"{prog}.csv"))):
            a = r["address"]
            e = entities.get(r["profile_id"], {})
            rec = rows.setdefault(a, {"profile_id": r["profile_id"], "entity": r["entity_name"], "currency": r["currency"], "programmes": [],
                                      "entity_date_published": e.get("date_published")})
            rec["programmes"].append(prog)
            if a in added and added[a]["publication_id"] != baseline:
                rec.update({"date_listed": added[a]["date_added"], "date_source": "delta file", "publication_id": added[a]["publication_id"]})
            elif a in added:
                rec.update({"date_listed": e.get("date_published"), "publication_id": baseline,
                            "date_source": f"entity listing (address already on the list in the baseline snapshot of {added[a]['date_added']})"})
            else:
                rec.update({"date_listed": e.get("date_published"), "date_source": "entity listing (address not found in the change history)"})
    n_delta = sum(r["date_source"] == "delta file" for r in rows.values())
    print(f"{len(rows)} addresses in the three programme files: {n_delta} dated by a delta file, {len(rows) - n_delta} by the entity listing")
    for a, r in sorted(rows.items(), key=lambda kv: (kv[1]["date_listed"] or "")):
        if r["currency"] == "USDT" and a.startswith("T"):
            print(f"  {a} {r['entity'][:30]:30} entity {r['entity_date_published']} address {r['date_listed']} ({r['date_source'][:10]})")
    first_delta = next(open(p_).read() for p_ in paths[1:2])
    first_date = re.search(r"<datePublished>(\d{4}-\d{2}-\d{2})", first_delta).group(1)
    out = {"source": f"OFAC Sanctions List Service ({SLS}): entity records, the baseline snapshot and the delta file of every publication from {first_date} to {pubs[-1]['datePublished'][:10]}, accessed {time.strftime('%d %B %Y')}",
           "publications": len(pubs), "entities": entities, "addresses": rows}
    with open(out_path(args.out), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
