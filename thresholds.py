"""The eight thresholding algorithms, applied to ROI-masked, normalised images.

Conventions
-----------
* Input images are float in [0, 1] (see :func:`normalize`), so every threshold
  is expressed in *normalised grey level* and is comparable across samples.
* Global thresholds are computed from the histogram of the ROI pixels only
  (air around the core or the holder wall never enters the histogram).
* Local thresholds (Sauvola, Niblack) use *masked* local statistics
  (normalised convolution), so pixels outside the ROI never bias the local
  mean/standard deviation. Inside the ROI and away from its border the maps are
  identical to ``skimage.filters.threshold_sauvola/niblack``.
* Pore phase = pixels with ``I <= T`` (low attenuation).

Taxonomy follows Sezgin & Sankur (2004, J. Electron. Imaging 13:146-168).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import filters

METHOD_INFO: dict[str, dict] = {
    "Otsu":     dict(family="global", category="Clustering (between-class variance)",
                     reference="Otsu (1979) IEEE Trans. SMC 9:62-66"),
    "Li":       dict(family="global", category="Entropy (minimum cross-entropy)",
                     reference="Li & Tam (1998) Pattern Recogn. Lett. 19:771-776"),
    "Yen":      dict(family="global", category="Entropy (maximum entropic correlation)",
                     reference="Yen et al. (1995) IEEE Trans. Image Process. 4:370-378"),
    "Mean":     dict(family="global", category="Attribute (mean grey level)",
                     reference="Glasbey (1993) CVGIP: Graph. Models Image Process. 55:532-537"),
    "Triangle": dict(family="global", category="Histogram shape",
                     reference="Zack et al. (1977) J. Histochem. Cytochem. 25:741-753"),
    "Isodata":  dict(family="global", category="Clustering (iterative intermeans)",
                     reference="Ridler & Calvard (1978) IEEE Trans. SMC 8:630-632"),
    "Sauvola":  dict(family="local", category="Local adaptive (mean, s.d.)",
                     reference="Sauvola & Pietikainen (2000) Pattern Recogn. 33:225-236"),
    "Niblack":  dict(family="local", category="Local adaptive (mean, s.d.)",
                     reference="Niblack (1986) An Introduction to Digital Image Processing"),
}
GLOBAL_METHODS = tuple(m for m, i in METHOD_INFO.items() if i["family"] == "global")
LOCAL_METHODS = tuple(m for m, i in METHOD_INFO.items() if i["family"] == "local")


@dataclass
class ThresholdResult:
    method: str
    family: str
    value: float | np.ndarray      # scalar (global) or bbox-shaped map (local)
    summary: float                 # scalar threshold, or mean of the map inside the ROI
    params: dict = field(default_factory=dict)

    @property
    def is_map(self) -> bool:
        return isinstance(self.value, np.ndarray)


# ----------------------------------------------------------------------------
# normalisation
# ----------------------------------------------------------------------------
def normalize(img: np.ndarray, mask: np.ndarray, p_low: float, p_high: float):
    """Robust min-max scaling to [0, 1] using percentiles of the ROI pixels."""
    lo, hi = np.percentile(img[mask], [p_low, p_high])
    if hi <= lo:
        raise ValueError("degenerate intensity range inside ROI")
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out, float(lo), float(hi)


def to_raw(t_norm: float, lo: float, hi: float) -> float:
    return lo + t_norm * (hi - lo)


# ----------------------------------------------------------------------------
# masked local statistics
# ----------------------------------------------------------------------------
def masked_local_mean_std(img: np.ndarray, mask: np.ndarray, window: int):
    """Local mean and (population) s.d. over a ``window x window`` box,
    using only pixels inside ``mask`` (normalised convolution)."""
    if window % 2 == 0:
        raise ValueError("window must be odd")
    m = mask.astype(np.float64)
    x = np.where(mask, img, 0.0).astype(np.float64)
    cnt = ndi.uniform_filter(m, window, mode="constant", cval=0.0)
    s1 = ndi.uniform_filter(x, window, mode="constant", cval=0.0)
    s2 = ndi.uniform_filter(x * x, window, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / cnt, 0.0)
        var = np.where(cnt > 0, s2 / cnt - mean ** 2, 0.0)
    return mean, np.sqrt(np.clip(var, 0.0, None))


def sauvola_map(img, mask, window=25, k=0.2, r=0.5):
    """T = m * (1 + k * (s / r - 1))   (Sauvola & Pietikainen 2000)."""
    m, s = masked_local_mean_std(img, mask, window)
    return m * (1.0 + k * (s / r - 1.0))


def niblack_map(img, mask, window=25, k=0.2):
    """T = m - k * s   (scikit-image sign convention for dark foreground)."""
    m, s = masked_local_mean_std(img, mask, window)
    return m - k * s


# ----------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------
def default_params(method: str, cfg: dict) -> dict:
    t = cfg["thresholds"]
    if method == "Sauvola":
        return dict(window=int(t["sauvola"]["window"]), k=float(t["sauvola"]["k"]), r=float(t["sauvola"]["r"]))
    if method == "Niblack":
        return dict(window=int(t["niblack"]["window"]), k=float(t["niblack"]["k"]))
    if method in ("Otsu", "Yen", "Triangle", "Isodata"):
        return dict(nbins=int(t["nbins"]))
    return {}


def compute_threshold(method: str, img: np.ndarray, mask: np.ndarray, params: dict) -> ThresholdResult:
    if method not in METHOD_INFO:
        raise KeyError(f"unknown method {method}")
    fam = METHOD_INFO[method]["family"]
    if fam == "global":
        v = img[mask].astype(np.float64)
        nb = params.get("nbins", 256)
        if method == "Otsu":
            t = filters.threshold_otsu(v, nbins=nb)
        elif method == "Li":
            t = filters.threshold_li(v)
        elif method == "Yen":
            t = filters.threshold_yen(v, nbins=nb)
        elif method == "Mean":
            t = filters.threshold_mean(v)
        elif method == "Triangle":
            t = filters.threshold_triangle(v, nbins=nb)
        elif method == "Isodata":
            t = filters.threshold_isodata(v, nbins=nb)
            t = float(np.atleast_1d(t)[0])
        return ThresholdResult(method, fam, float(t), float(t), dict(params))
    if method == "Sauvola":
        tm = sauvola_map(img, mask, params["window"], params["k"], params["r"])
    else:
        tm = niblack_map(img, mask, params["window"], params["k"])
    return ThresholdResult(method, fam, tm.astype(np.float32), float(tm[mask].mean()), dict(params))


def segment(img: np.ndarray, thr: ThresholdResult | float | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pore phase (bool): ``img <= T`` inside the mask."""
    t = thr.value if isinstance(thr, ThresholdResult) else thr
    return (img <= t) & mask
