"""Binary post-processing of the PORE phase.

All operations act on the pore phase (True = pore). This matters: the legacy
code applied opening/closing to the *solid* phase and inverted the skeleton
(skeleton pixels were written as solid), so its 'skeletonized porosity' was
1 - skeleton fraction (~0.95-0.99).

Levels
------
none        identity
median      3x3 median (= majority vote for binary images)
open_close  opening followed by closing of the pore phase (removes pores and
            throats narrower than the structuring element, then fills solid
            gaps narrower than it)
skeleton    medial axis (1-px wide). NOT area preserving - reported as a
            skeleton (medial-axis) density, never as porosity.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

LEVELS = ("none", "median", "open_close", "skeleton")
AREA_PRESERVING = ("none", "median", "open_close")
LABELS = {"none": "None", "median": "Median 3x3", "open_close": "Open-close 3x3", "skeleton": "Skeleton"}


def structuring_element(shape: str, size: int) -> np.ndarray:
    if shape == "square":
        return np.ones((size, size), bool)
    if shape == "disk":
        r = size // 2
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return (xx ** 2 + yy ** 2) <= r ** 2
    raise ValueError(f"unknown structuring element {shape}")


def apply(pore: np.ndarray, level: str, cfg: dict, mask: np.ndarray | None = None) -> np.ndarray:
    pp = cfg["postprocessing"]
    if level == "none":
        out = pore.copy()
    elif level == "median":
        out = ndi.median_filter(pore.astype(np.uint8), size=int(pp["median_size"])).astype(bool)
    elif level == "open_close":
        se = structuring_element(pp["morph_se"], int(pp["morph_size"]))
        out = ndi.binary_opening(pore, structure=se)
        out = ndi.binary_closing(out, structure=se)
    elif level == "skeleton":
        out = skeletonize(pore)
    else:
        raise ValueError(f"unknown post-processing level '{level}'")
    if mask is not None:
        out &= mask
    return out
