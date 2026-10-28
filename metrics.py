"""Basic metrics on binary pore images (all restricted to an evaluation mask)."""
from __future__ import annotations

import numpy as np


def porosity(pore: np.ndarray, eval_mask: np.ndarray) -> float:
    """Area fraction of pore pixels (by the Delesse principle, an unbiased
    estimator of the volume fraction for a stationary, isotropic medium)."""
    n = eval_mask.sum()
    return float(pore[eval_mask].sum() / n) if n else np.nan


def dice(a: np.ndarray, b: np.ndarray, eval_mask: np.ndarray) -> float:
    a, b = a[eval_mask], b[eval_mask]
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0


def jaccard(a: np.ndarray, b: np.ndarray, eval_mask: np.ndarray) -> float:
    a, b = a[eval_mask], b[eval_mask]
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def pixel_accuracy(a: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray) -> float:
    return float((a[eval_mask] == truth[eval_mask]).mean())
