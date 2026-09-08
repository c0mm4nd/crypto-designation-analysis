"""
Evaluation utilities for WCFRM.

- threshold_counts : for each threshold t, count labelled (seed) nodes with score >= t
- recall_at_k      : recall at top-K nodes
- compute_auc      : ROC-AUC and 95% CI via bootstrap
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def threshold_counts(
    scores:  np.ndarray,
    labels:  np.ndarray,
    thresholds: list[float] = THRESHOLDS,
) -> dict[float, int]:
    """
    Count labelled-positive nodes whose score >= threshold.

    Parameters
    ----------
    scores    : (N,) float array, normalized to [0, 1]
    labels    : (N,) binary int array (1 = seed/sanctioned, 0 = non-seed)
    thresholds: list of threshold values

    Returns
    -------
    dict {threshold: count_of_labelled_nodes_above_threshold}
    """
    result = {}
    for t in thresholds:
        flagged  = scores >= t
        result[t] = int((flagged & labels.astype(bool)).sum())
    return result


def compute_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute ROC-AUC with 95% CI via bootstrap.

    Returns
    -------
    (auc, ci_low, ci_high)
    """
    auc = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    N   = len(labels)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        try:
            boot_aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except Exception:
            pass
    ci_low  = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))
    return auc, ci_low, ci_high


def print_threshold_table(
    method_scores: dict[str, np.ndarray],
    labels:        np.ndarray,
    thresholds:    list[float] = THRESHOLDS,
):
    """Print a summary table similar to Table 1 in the paper."""
    header = "Method".ljust(30) + "  ".join(f"{t:.1f}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for name, scores in method_scores.items():
        counts = threshold_counts(scores, labels, thresholds)
        row    = name.ljust(30) + "  ".join(
            f"{counts[t]:3d}" for t in thresholds
        )
        print(row)


def print_auc_table(
    method_scores: dict[str, np.ndarray],
    labels:        np.ndarray,
    n_bootstrap:   int = 200,
) -> dict:
    """Print AUC table with 95% CI. Returns dict of results for reuse."""
    print(f"{'Method':<30} {'AUC':>6}  {'95% CI'}")
    print("-" * 55)
    cached = {}
    for name, scores in method_scores.items():
        try:
            auc, lo, hi = compute_auc(scores, labels, n_bootstrap=n_bootstrap)
            print(f"{name:<30} {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
            cached[name] = {"auc": auc, "ci_low": lo, "ci_high": hi}
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")
    return cached
