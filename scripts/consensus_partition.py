#!/usr/bin/env python3
"""Consensus role partition across independently trained models.

A partition read off a prototype head is a deterministic function of the trained model,
but the trained model still depends on its random seed, so the reported partition depends
on which seed is reported. Rather than pick one, we align the partitions from all training
seeds by matching their roles (Hungarian assignment on the contingency table, which is the
standard way to resolve the label permutation between two clusterings) and take the
majority label per address. The consensus partition is a single object that no seed owns,
and the share of seeds agreeing on an address's role is a per-address confidence that can
be reported alongside it.

Usage:
  python scripts/consensus_partition.py --runs v6_runs/v2_israel_tron_seed{42,43,44}_clusters.npy \
      --out v6_runs/v2_israel_tron_consensus.npy
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment


def align(reference: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Relabel `other` so its roles correspond to those of `reference`."""
    M = np.zeros((k, k), dtype=np.int64)
    np.add.at(M, (reference, other), 1)
    row, col = linear_sum_assignment(-M)
    mapping = np.arange(k)
    mapping[col] = row
    return mapping[other]


def consensus(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    k = int(max(p.max() for p in parts)) + 1
    aligned = [parts[0]] + [align(parts[0], p, k) for p in parts[1:]]
    votes = np.zeros((len(parts[0]), k), dtype=np.int16)
    for p in aligned:
        votes[np.arange(len(p)), p] += 1
    lab = votes.argmax(1)
    conf = votes.max(1) / len(parts)
    return lab.astype(np.int16), conf.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = [np.load(f).astype(int) for f in a.runs]
    lab, conf = consensus(parts)
    np.save(a.out, lab)
    np.save(a.out.replace(".npy", "_confidence.npy"), conf)
    k = int(lab.max()) + 1
    print(f"{len(parts)} partitions, K={k}, sizes={np.bincount(lab, minlength=k).tolist()}")
    print(f"unanimous on {100*(conf == 1).mean():.1f}% of addresses, "
          f"majority-only on {100*((conf < 1) & (conf > 1/len(parts))).mean():.1f}%")


if __name__ == "__main__":
    main()
