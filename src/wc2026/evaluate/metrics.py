"""
W/D/L probabilistic metrics.

All functions take:
  probs:    (n, 3) array, columns ordered [P(home win), P(draw), P(away win)]
  outcomes: (n,) int array with values in {0, 1, 2} matching the column index

Conventions
-----------
  outcome = 0  →  home win
  outcome = 1  →  draw
  outcome = 2  →  away win
"""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def _as_arrays(probs: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"probs must have shape (n, 3); got {probs.shape}")
    if outcomes.shape != (probs.shape[0],):
        raise ValueError(f"outcomes shape {outcomes.shape} != ({probs.shape[0]},)")
    return probs, outcomes


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the true outcomes. Lower is better."""
    probs, outcomes = _as_arrays(probs, outcomes)
    chosen = probs[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(chosen, EPS, 1.0)).mean())


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error vs one-hot truth. Lower is better.

    Range [0, 2]; a uniform 1/3 forecast yields ~0.667.
    """
    probs, outcomes = _as_arrays(probs, outcomes)
    truth = np.zeros_like(probs)
    truth[np.arange(len(outcomes)), outcomes] = 1.0
    return float(((probs - truth) ** 2).sum(axis=1).mean())


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score for ordinal outcomes (home / draw / away). Lower is better.

    RPS is the football-standard metric for 1X2 markets — it rewards forecasts
    whose CDF is close to the realized CDF, so a confident "wrong direction"
    miss costs more than a confident "right direction" miss.
    """
    probs, outcomes = _as_arrays(probs, outcomes)
    truth = np.zeros_like(probs)
    truth[np.arange(len(outcomes)), outcomes] = 1.0
    cdf_p = np.cumsum(probs, axis=1)
    cdf_t = np.cumsum(truth, axis=1)
    # Average over the K-1 = 2 transition points
    return float(((cdf_p[:, :-1] - cdf_t[:, :-1]) ** 2).sum(axis=1).mean())


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Fraction of matches whose argmax-prob outcome matches the truth."""
    probs, outcomes = _as_arrays(probs, outcomes)
    return float((probs.argmax(axis=1) == outcomes).mean())


def calibration_buckets(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Bucket every (probability, was-correct) pair and report mean predicted vs observed.

    Flattens across all 3 classes: each match contributes 3 (p, hit) pairs.
    Returns one dict per non-empty bin with keys:
      bin_low, bin_high, mean_pred, mean_obs, n
    """
    probs, outcomes = _as_arrays(probs, outcomes)
    truth = np.zeros_like(probs)
    truth[np.arange(len(outcomes)), outcomes] = 1.0

    flat_p = probs.ravel()
    flat_t = truth.ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (flat_p >= lo) & (flat_p <= hi)
        else:
            mask = (flat_p >= lo) & (flat_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "mean_pred": float(flat_p[mask].mean()),
                "mean_obs": float(flat_t[mask].mean()),
                "n": n,
            }
        )
    return buckets
