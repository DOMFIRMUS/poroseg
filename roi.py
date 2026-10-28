"""Sample geometry and regions of interest (ROIs).

Two acquisition geometries occur in the data set:

* ``free``      - a free-standing cylindrical core surrounded by air. The core
                  boundary is found by Otsu thresholding of a smoothed image,
                  filling holes, and a RANSAC circle fit to the boundary.
* ``container`` - grains packed in a cylindrical holder. The holder wall is a
                  thick, homogeneous annulus; the packing ends where a ray cast
                  from the centre towards the wall *enters* the wall intensity
                  band. Edge points from 720 rays are fitted with a RANSAC circle.

The usable disc is the fitted circle shrunk by a safety margin
(``roi.free_margin_frac`` or ``roi.container_margin_px``). Manual overrides
(``roi_cx, roi_cy, roi_r`` in samples.csv) take precedence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, transform


# ----------------------------------------------------------------------------
# data structures
# ----------------------------------------------------------------------------
@dataclass
class Geometry:
    cx: float
    cy: float
    r_boundary: float          # fitted sample (free) or inner-wall (container) radius, px
    r_usable: float            # radius of the usable disc, px
    holder: str
    source: str                # 'auto' | 'manual'
    fit_rms_px: float = np.nan
    inlier_frac: float = np.nan
    info: dict = field(default_factory=dict)


@dataclass
class ROI:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray           # bool, bbox-shaped: pixels that belong to the ROI
    eval_mask: np.ndarray      # bool, bbox-shaped: ROI minus a thin rim (metrics are computed here)

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    @property
    def n_eval(self) -> int:
        return int(self.eval_mask.sum())


# ----------------------------------------------------------------------------
# circle fitting helpers
# ----------------------------------------------------------------------------
def _ransac_circle(y: np.ndarray, x: np.ndarray, thr: float, seed: int = 0):
    model, inliers = measure.ransac(np.c_[x, y], measure.CircleModel, min_samples=3,
                                    residual_threshold=thr, max_trials=3000, rng=seed)
    xc, yc = model.center
    r = model.radius
    res = np.hypot(x - xc, y - yc) - r
    rms = float(np.sqrt(np.mean(res[inliers] ** 2)))
    return float(xc), float(yc), float(r), rms, float(inliers.mean())


def _rescale01(a: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    p1, p99 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / max(p99 - p1, 1e-12), 0, 1)


# ----------------------------------------------------------------------------
# detectors
# ----------------------------------------------------------------------------
def detect_free(img: np.ndarray, seed: int = 0) -> Geometry:
    s = ndi.gaussian_filter(img, 3)
    t = filters.threshold_otsu(s[::2, ::2])
    lab = measure.label(s > t)
    if lab.max() == 0:
        raise RuntimeError("free-core detection: no foreground found")
    largest = max(measure.regionprops(lab), key=lambda r: r.area).label
    region = ndi.binary_fill_holes(lab == largest)
    contour = max(measure.find_contours(region.astype(float), 0.5), key=len)
    xc, yc, r, rms, inl = _ransac_circle(contour[:, 0], contour[:, 1], thr=3.0, seed=seed)
    return Geometry(cx=xc, cy=yc, r_boundary=r, r_usable=np.nan, holder="free", source="auto",
                    fit_rms_px=rms, inlier_frac=inl, info=dict(otsu_smoothed=float(t)))


def detect_container(img: np.ndarray, seed: int = 0, n_rays: int = 720) -> Geometry:
    # 1) outer ring (strongest large circle) -> centre and search radius
    ds = 4
    small = _rescale01(transform.downscale_local_mean(img, (ds, ds)))
    edges = feature.canny(small, sigma=2)
    m = min(small.shape)
    radii = np.arange(int(0.30 * m), int(0.49 * m))
    hs = transform.hough_circle(edges, radii)
    _, cxs, cys, rrs = transform.hough_circle_peaks(hs, radii, total_num_peaks=1)
    x0, y0, r_out = cxs[0] * ds + ds / 2, cys[0] * ds + ds / 2, float(rrs[0] * ds)

    # 2) polar unwrap of the smoothed image around (x0, y0)
    sm = ndi.gaussian_filter(img, 2)
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    rg = np.arange(0, int(r_out))
    Y = y0 + np.outer(np.sin(th), rg)
    X = x0 + np.outer(np.cos(th), rg)
    P = ndi.map_coordinates(sm, [Y, X], order=1, mode="nearest")

    # 3) wall = the contiguous run of radii (next to the outer ring) where the
    #    angular inter-quartile range is small (homogeneous annulus)
    q75, q25 = np.percentile(P, [75, 25], axis=0)
    iqr = q75 - q25
    pack_iqr = np.median(iqr[int(0.2 * r_out):int(0.6 * r_out)])
    low = iqr < 0.35 * pack_iqr
    r = int(r_out) - 12
    while r > 0 and not low[r]:
        r -= 1
    r_hi = r
    while r > 0 and low[r]:
        r -= 1
    r_lo = r + 1
    if r_hi - r_lo < 10:
        raise RuntimeError("container detection: holder wall not found")
    wall = P[:, r_lo + 3:r_hi - 3].ravel()
    wl = float(np.median(wall))
    wsd = float(1.4826 * np.median(np.abs(wall - wl)))

    # 4) per ray, walk inwards from inside the wall until 3 consecutive samples
    #    leave the wall band (pore = darker, grain = brighter)
    pts = []
    for k in range(n_rays):
        out = np.abs(P[k] - wl) > 4 * wsd
        rr = r_lo + 5
        while rr > 3 and not (out[rr] and out[rr - 1] and out[rr - 2]):
            rr -= 1
        pts.append((Y[k, rr], X[k, rr]))
    pts = np.asarray(pts)
    xc, yc, rin, rms, inl = _ransac_circle(pts[:, 0], pts[:, 1], thr=2.0, seed=seed)
    return Geometry(cx=xc, cy=yc, r_boundary=rin, r_usable=np.nan, holder="container", source="auto",
                    fit_rms_px=rms, inlier_frac=inl,
                    info=dict(outer_ring_r=r_out, wall_run=(int(r_lo), int(r_hi)), wall_level=wl,
                              wall_mad=wsd, edge_points=pts))


def detect_geometry(img: np.ndarray, holder: str, cfg: dict, override: dict | None = None,
                    seed: int = 0) -> Geometry:
    if override and all(np.isfinite(override.get(k, np.nan)) for k in ("roi_cx", "roi_cy", "roi_r")):
        g = Geometry(cx=float(override["roi_cx"]), cy=float(override["roi_cy"]),
                     r_boundary=float(override["roi_r"]), r_usable=float(override["roi_r"]),
                     holder=holder, source="manual")
        return g
    g = detect_container(img, seed) if holder == "container" else detect_free(img, seed)
    if holder == "container":
        g.r_usable = g.r_boundary - float(cfg["roi"]["container_margin_px"])
    else:
        g.r_usable = g.r_boundary * (1.0 - float(cfg["roi"]["free_margin_frac"]))
    # never extend beyond the image
    h, w = img.shape
    g.r_usable = float(min(g.r_usable, g.cx, g.cy, w - 1 - g.cx, h - 1 - g.cy))
    return g


# ----------------------------------------------------------------------------
# ROI construction
# ----------------------------------------------------------------------------
def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    return ndi.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=px, border_value=0)


def disk_roi(shape: tuple[int, int], g: Geometry, eval_border: int) -> ROI:
    h, w = shape
    r = g.r_usable
    y0, y1 = max(int(np.floor(g.cy - r)) - 1, 0), min(int(np.ceil(g.cy + r)) + 2, h)
    x0, x1 = max(int(np.floor(g.cx - r)) - 1, 0), min(int(np.ceil(g.cx + r)) + 2, w)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (yy - g.cy) ** 2 + (xx - g.cx) ** 2 <= r ** 2
    return ROI("disk", y0, y1, x0, x1, mask, _erode(mask, eval_border))


def center_square_roi(shape: tuple[int, int], size: int, eval_border: int) -> ROI:
    """Square ROI centred on the IMAGE centre (reproduces the original design)."""
    h, w = shape
    if h < size or w < size:
        raise ValueError(f"image {shape} smaller than the {size}px ROI")
    y0, x0 = (h - size) // 2, (w - size) // 2
    mask = np.ones((size, size), bool)
    return ROI(f"center{size}", y0, y0 + size, x0, x0 + size, mask, _erode(mask, eval_border))


def build_rois(shape: tuple[int, int], g: Geometry, cfg: dict) -> dict[str, ROI]:
    eb = int(cfg["roi"]["eval_border_px"])
    out: dict[str, ROI] = {}
    for name in cfg["roi"]["designs"]:
        if name == "disk":
            out[name] = disk_roi(shape, g, eb)
        elif name.startswith("center"):
            out[name] = center_square_roi(shape, int(cfg["roi"]["center_size_px"]), eb)
        else:
            raise ValueError(f"unknown ROI design '{name}'")
    return out


def radial_distance(roi: ROI, g: Geometry) -> np.ndarray:
    """Distance (px) of every bbox pixel from the sample centre."""
    yy, xx = np.mgrid[roi.y0:roi.y1, roi.x0:roi.x1]
    return np.hypot(yy - g.cy, xx - g.cx)
