#!/usr/bin/env python3
"""Count the directed pairs of the complete network that carry no value.

The complete-network export applies no value filter, so an address that received only a dust
or address-poisoning transfer is a node and a zero-value pair is an edge. This records how
many, so that the share quoted in Methods has an artefact behind it.

Usage:
  ROTOR_DATA=/path/to/exports python scripts/count_zero_value_pairs.py
"""
import glob
import json
import os

import numpy as np

DATA = os.environ.get("ROTOR_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = np.dtype([("f", "<u8"), ("t", "<u8"), ("v", "<f8")])


def main():
    z = n = 0
    for f in sorted(glob.glob(os.path.join(DATA, "tron_full_val", "bucket_*.bin"))):
        a = np.fromfile(f, dtype=DT)
        z += int((a["v"] == 0).sum())
        n += len(a)
        del a
    out = {"n_pairs": n, "n_zero_value_pairs": z, "zero_value_share": z / n if n else 0.0}
    with open(os.path.join(os.environ.get("ROTOR_OUT", DATA), "zero_value_pairs.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)


if __name__ == "__main__":
    main()
