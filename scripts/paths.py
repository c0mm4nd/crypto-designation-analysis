"""Resolve inputs and outputs in either the working layout or the deposited archive.

The scripts were written against the working tree, where every analysis output sits in the
repository root and every record-level input in `ch_data/`. The Zenodo archive groups them
instead into `analysis/`, `data/` and `tables/`, which is easier to read but means a reader
who extracts the archive and runs a script gets FileNotFoundError, or worse, an empty result.

This module resolves a name against both layouts, so the same script runs either way, and
writes outputs next to the inputs it found. Import it rather than building paths by hand:

    from paths import ROOT, find, out_path
    d = json.load(open(find("phenomena.json")))
    raw = find("ch_data/designated_usdt_transfers_complete_490.csv")
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ROTOR_ROOT", Path(__file__).resolve().parents[1]))

# Where each kind of file lives in the archive, relative to ROOT.
_ARCHIVE_DIRS = ("analysis", "data", "tables", ".")


def find(name: str, required: bool = True) -> Path:
    """Return the path to `name`, searching the working layout then the archive layout.

    `name` may carry a directory (`ch_data/x.csv`); the basename is also tried under the
    archive directories, since the archive flattens `ch_data/` into `data/`.
    """
    rel = Path(name)
    candidates = [ROOT / rel]
    if rel.parent != Path("."):
        candidates.append(ROOT / rel.name)
    for d in _ARCHIVE_DIRS:
        candidates.append(ROOT / d / rel.name)
    for c in candidates:
        if c.exists():
            return c
    for c in candidates:
        gz = c.with_suffix(c.suffix + ".gz")
        if gz.exists():
            return gz
    if required:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"{name} not found. Tried:\n  {tried}")
    return ROOT / rel


def exists(name: str) -> bool:
    try:
        find(name)
        return True
    except FileNotFoundError:
        return False


def out_path(name: str) -> Path:
    """Where to write `name`: beside the existing copy if there is one, else in ROOT."""
    try:
        return find(name)
    except FileNotFoundError:
        d = ROOT / "analysis"
        return (d / name) if d.is_dir() else (ROOT / name)
