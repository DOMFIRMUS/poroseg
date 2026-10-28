"""Morphological, topological and grey-scale (partial-volume) descriptors of
2-D binary pore images, plus the small-sample statistics used by the
'morphology' stage.

Every function is *pure* (no I/O, no global state) and works on bbox-shaped
arrays: ``pore`` (bool, True = pore) and ``window`` (bool, the evaluation mask
in which the quantity is estimated, normally ``ROI.eval_mask``). All
quantities are returned in *pixel units*; the stage converts them to physical
units with the pixel size Δ:

==========================  =====================  ==========================
quantity                    pixel unit             physical unit
==========================  =====================  ==========================
porosity                    -                      -
boundary length density     px⁻¹                   mm⁻¹      (÷ Δ[mm])
number / Euler density      px⁻²                   mm⁻²      (÷ Δ[mm]²)
diameters                   px                     µm        (× Δ[µm])
skeleton length density     px⁻¹                   mm mm⁻²   (÷ Δ[mm])
==========================  =====================  ==========================

Stereological background
------------------------
For a stationary random set observed in a window W, the *area fraction* A_A,
the *boundary length density* L_A and the *Euler number density* χ_A are the
three additive (Minkowski) functionals of 2-D image analysis (Hadwiger's
theorem; Serra 1982; Ohser & Mücklich 2000). Each is estimated here by
*additive local counts restricted to configurations that lie entirely inside
the window*, which makes the estimators free of edge effects (minus-sampling)
and unbiased for stationary structures:

* A_A - fraction of window pixels that are pore (Delesse 1847).
* L_A - Crofton/Cauchy intercept formula ``L_A = (π/2)·P̄_L``, where
  ``P_L(θ)`` is the number of pore/solid transitions per unit length of test
  line in direction θ, averaged over θ ∈ {0°, 45°, 90°, 135°}; only pixel
  pairs with *both* pixels in W are used, so the ROI border never creates
  boundary (Ohser & Mücklich 2000, ch. 4; Lehmann & Legland 2012).
  For an isotropic 3-D structure ``S_V = (4/π)·L_A`` (Underwood 1970).
* χ_A - Gray's (1971) bit-quad counts of 2×2 configurations, with the pore
  phase 8-connected and the solid 4-connected (complementary connectivities,
  required for a consistent digital topology):
  ``χ = [n(Q1) − n(Q3) − 2·n(QD)] / 4`` (pore 8-conn.) and
  ``χ = [n(Q1) − n(Q3) + 2·n(QD)] / 4`` (pore 4-conn.), where Q1/Q3 are quads
  with exactly one/three pore pixels and QD the two diagonal quads. Summed over
  all quads of a zero-padded image this equals the Euler number
  (#components − #holes) exactly; restricted to quads inside W and divided by
  their number it is an unbiased estimator of χ_A (Ohser, Nagel & Schladitz
  2003, Image Anal. Stereol. 22:11-19).

Pore objects that touch the window boundary are truncated, so object counts
and object-size distributions use the Miles-Lantuéjoul (Miles 1974;
Lantuéjoul 1980) minus-sampling correction: an object X that lies entirely
in the interior window W' is counted with weight ``1/A(W' ⊖ X̌)``, the area
of the set of translations that keep X inside W'. For a disc window of radius
R and a convex object of mean width b̄ (Steiner/Cauchy, curvature radii ≤ R)
``A(W' ⊖ X̌) = π R² − R·U(X) + A(X)`` with perimeter ``U = π b̄``; we use
``π (R − b̄/2)²``, exact for discs and segments to first order and accurate to
O(b̄²/R²) otherwise. For a rectangular window of a × b pixels the exact count
is ``(a − w_x)(b − w_y)`` with w the projected extents of X.

Granulometry: the opening of the pore phase P by discs D_r of increasing
radius r (Matheron 1975; Serra 1982) removes every pore pixel that is not
covered by a disc of radius r fitting inside P. ``G(r) = A(γ_r P)/A(P)`` is a
decreasing distribution function; ``F = 1 − G`` is the *pore-area-weighted*
cumulative size distribution (identical to the distribution of the 'local
thickness' of Hildebrand & Rüegsegger 1997). It needs neither object
segmentation nor connectivity and is therefore meaningful also for connected
pore networks (granular packs), where object-based sizes are not.
The opening is computed exactly with Euclidean distance transforms:
erosion by D_r = {x : d(x) > r} (d = distance to the nearest solid pixel)
and dilation by D_r = {y : dist(y, eroded) ≤ r}.

Grey-scale (partial-volume) porosity
------------------------------------
Under a two-phase linear-attenuation model each pixel value is
``I = φ_loc μ_p + (1 − φ_loc) μ_s + ε`` with φ_loc ∈ [0, 1] the pore fraction
of the voxel, μ_s / μ_p the grey levels of pure solid / pure pore (air) and ε
zero-mean noise. Averaging over the ROI and solving for the mean pore
fraction gives ``φ_PV = (μ_s − Ī)/(μ_s − μ_p)``. Because the estimator is
linear in I, zero-mean noise cancels in Ī and φ_PV is unbiased; clipping
per-pixel fractions to [0, 1] would *not* be (it removes the negative half of
the noise in the solid and biases φ_PV upwards; Ketcham & Carlson 2001;
Taud et al. 2005, J. Petrol. Sci. Eng. 47:209-220; Lin et al. 2016, Phys.
Rev. E 94:043113). Assumptions: linear (FBP) reconstruction, no cupping /
beam hardening, two phases with constant attenuation. Dense inclusions
brighter than μ_s raise Ī and bias φ_PV low; homogeneous sub-resolution
porosity of the matrix itself is absorbed into μ_s and is invisible to φ_PV.

References
----------
Gray SB (1971) IEEE Trans. Comput. C-20:551-561.
Hildebrand T, Rüegsegger P (1997) J. Microsc. 185:67-75.
Lantuéjoul C (1980) J. Microsc. 119:247-251.
Lehmann G, Legland D (2012) Insight Journal (Crofton perimeter in ITK).
Matheron G (1975) Random Sets and Integral Geometry. Wiley.
Miles RE (1974) in: Stochastic Geometry (Harding & Kendall eds), Wiley.
Ohser J, Mücklich F (2000) Statistical Analysis of Microstructures in
    Materials Science. Wiley.
Serra J (1982) Image Analysis and Mathematical Morphology. Academic Press.
Underwood EE (1970) Quantitative Stereology. Addison-Wesley.
Zhang TY, Suen CY (1984) Commun. ACM 27:236-239 (skeletonisation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

S8 = np.ones((3, 3), bool)                     # 8-connectivity
S4 = ndi.generate_binary_structure(2, 1)       # 4-connectivity
SQRT2 = math.sqrt(2.0)


# =============================================================================
# helpers
# =============================================================================
def _pair_views(a: np.ndarray, dy: int, dx: int):
    """Views (u, v) of equal shape with ``v[i, j] = a[i + dy, j + dx]`` and
    ``u[i, j] = a[i, j]`` (dy >= 0)."""
    h, w = a.shape
    ya, yb = slice(0, h - dy), slice(dy, h)
    if dx >= 0:
        xa, xb = slice(0, w - dx), slice(dx, w)
    else:
        xa, xb = slice(-dx, w), slice(0, w + dx)
    return a[ya, xa], a[yb, xb]


# (dy, dx, step length in px) for the four Crofton directions 0°, 45°, 90°, 135°
CROFTON_DIRS = {0: (0, 1, 1.0), 45: (1, -1, SQRT2), 90: (1, 0, 1.0), 135: (1, 1, SQRT2)}


def disc_offsets_count(r: float) -> int:
    """Number of integer offsets v with |v| <= r (pixels of the discrete disc D_r)."""
    k = int(math.floor(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return int(((yy * yy + xx * xx) <= r * r + 1e-9).sum())


def disc_equivalent_diameter(r: float) -> float:
    """Area-equivalent diameter (px) of the discrete disc D_r: 2·sqrt(n/π).

    D_0 (a single pixel) has 2/sqrt(π) = 1.128 px, the same value as the
    equivalent diameter of a one-pixel pore object, so granulometric and
    object-based sizes share one convention."""
    return 2.0 * math.sqrt(disc_offsets_count(r) / math.pi)


# default radii (px): every distinct small disc, then ~15 % geometric steps
DEFAULT_RADII = tuple([1.0, SQRT2, 2.0, math.sqrt(5), math.sqrt(8), 3.0, math.sqrt(13), 4.0,
                       math.sqrt(20), 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 19.0, 22.0,
                       26.0, 30.0, 35.0, 41.0, 48.0, 56.0, 65.0, 76.0, 89.0, 104.0, 121.0, 141.0])


# =============================================================================
# boundary length (Crofton) and Euler number (bit quads)
# =============================================================================
def crofton_boundary_density(pore: np.ndarray, window: np.ndarray) -> dict:
    """Boundary length per unit area L_A (px⁻¹) by the Crofton intercept formula.

    ``L_A = (π/2) · mean_θ P_L(θ)``, ``P_L(θ) = n_transitions(θ) / (n_pairs(θ)·step(θ))``.

    Only pixel pairs with both pixels inside ``window`` are counted, so the
    window boundary contributes nothing (no artificial perimeter where pores
    are cut by the ROI). Unbiased for isotropic structures; for anisotropic
    ones it is the 4-direction discretisation of the Cauchy-Crofton integral.

    Returns dict(L_A, P_L0, P_L45, P_L90, P_L135, n_pairs).
    """
    pore = np.asarray(pore, bool)
    window = np.asarray(window, bool)
    out, pl = {}, []
    for th, (dy, dx, step) in CROFTON_DIRS.items():
        pa, pb = _pair_views(pore, dy, dx)
        wa, wb = _pair_views(window, dy, dx)
        both = wa & wb
        n = int(both.sum())
        t = int(((pa != pb) & both).sum())
        p = t / (n * step) if n else np.nan
        out[f"P_L{th}"] = p
        pl.append(p)
        out.setdefault("n_pairs", n)
    out["L_A"] = (math.pi / 2.0) * float(np.mean(pl))
    return out


def euler_quads(pore: np.ndarray, window: np.ndarray | None = None, connectivity: int = 8) -> tuple[float, int]:
    """Euler number from Gray's bit-quad counts.

    Parameters
    ----------
    pore : bool array (pore = True).
    window : bool array or None. If given, only 2×2 quads whose four pixels
        are inside ``window`` are counted (unbiased local estimator of χ_A);
        if None, all quads of the zero-padded image are counted, which gives
        the exact Euler number (#components − #holes) of the whole image.
    connectivity : 8 (pore 8-connected, solid 4-connected; default) or 4.

    Returns
    -------
    (chi, n_quads) with chi the summed local contributions (may be a
    multiple of 1/4 for windowed estimates) and n_quads the number of quads
    used (= window area in px² for the density ``chi / n_quads``).
    """
    x = np.asarray(pore, bool)
    if window is None:
        x = np.pad(x, 1)
    x = x.astype(np.uint8)
    a, b, c, d = x[:-1, :-1], x[:-1, 1:], x[1:, :-1], x[1:, 1:]
    s = a + b + c + d
    q1 = s == 1
    q3 = s == 3
    qd = (s == 2) & (a == d)
    if window is None:
        inside = np.ones(q1.shape, bool)
    else:
        w = np.asarray(window, bool)
        inside = w[:-1, :-1] & w[:-1, 1:] & w[1:, :-1] & w[1:, 1:]
    n1, n3, nd = int((q1 & inside).sum()), int((q3 & inside).sum()), int((qd & inside).sum())
    if connectivity == 8:
        chi = (n1 - n3 - 2 * nd) / 4.0
    elif connectivity == 4:
        chi = (n1 - n3 + 2 * nd) / 4.0
    else:
        raise ValueError("connectivity must be 4 or 8")
    return chi, int(inside.sum())


# =============================================================================
# pore objects: labelling, Miles-Lantuéjoul correction, sizes
# =============================================================================
@dataclass
class ObjectTable:
    area: np.ndarray          # px, per object
    mean_width: np.ndarray    # px (pixel-centre convex hull, Cauchy formula over n_dir directions)
    wx: np.ndarray            # px, horizontal extent (max − min of column index)
    wy: np.ndarray            # px, vertical extent
    border: np.ndarray        # bool, touches the window boundary (truncated object)
    weight: np.ndarray        # Miles-Lantuéjoul weight 1/A(W' ⊖ X̌) in px⁻² (0 for border objects)
    interior_area: int        # A(W') in px² (window minus its 1-px inner rim)

    @property
    def n(self) -> int:
        return int(self.area.size)


def object_table(pore: np.ndarray, window: np.ndarray, window_kind: str = "disc",
                 n_dir: int = 8) -> ObjectTable:
    """Label pore objects (8-connected) inside ``window`` and compute their
    area, mean width, extents, border flag and Miles-Lantuéjoul weight.

    ``window_kind`` = 'disc' (window approximated by the disc of equal area)
    or 'rect' (exact rectangular-window formula, uses the bbox of W').
    """
    pore = np.asarray(pore, bool)
    window = np.asarray(window, bool)
    p = pore & window
    lab, n = ndi.label(p, structure=S8)
    interior = ndi.binary_erosion(window, S8, border_value=0)
    a_int = int(interior.sum())
    if n == 0:
        z = np.zeros(0)
        return ObjectTable(z, z, z, z, np.zeros(0, bool), z, a_int)
    rim = window & ~interior
    border = np.zeros(n + 1, bool)
    border[np.unique(lab[rim & p])] = True
    border = border[1:]
    w = lab.shape[1]
    idx = np.flatnonzero(lab)
    labs = lab.ravel()[idx]
    order = np.argsort(labs, kind="stable")
    labs, idx = labs[order], idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(labs)) + 1]
    area = np.diff(np.r_[starts, labs.size]).astype(np.float64)
    y, x = np.divmod(idx, w)
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    widths = np.empty((n_dir, n))
    for k in range(n_dir):
        th = math.pi * k / n_dir
        proj = x * math.cos(th) + y * math.sin(th)
        widths[k] = np.maximum.reduceat(proj, starts) - np.minimum.reduceat(proj, starts)
    bbar = widths.mean(axis=0)
    wx = np.maximum.reduceat(x, starts) - np.minimum.reduceat(x, starts)
    wy = np.maximum.reduceat(y, starts) - np.minimum.reduceat(y, starts)
    if window_kind == "disc":
        R = math.sqrt(a_int / math.pi)
        A = math.pi * np.clip(R - bbar / 2.0, 0.0, None) ** 2
    elif window_kind == "rect":
        iy, ix = np.nonzero(interior)
        a_ = ix.max() - ix.min() + 1
        b_ = iy.max() - iy.min() + 1
        A = np.clip(a_ - wx, 0.0, None) * np.clip(b_ - wy, 0.0, None)
    else:
        raise ValueError("window_kind must be 'disc' or 'rect'")
    weight = np.where(border, 0.0, 1.0 / np.maximum(A, 1.0))
    return ObjectTable(area, bbar, wx, wy, border, weight, a_int)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Quantiles of x with non-negative weights w (midpoint interpolation of the
    weighted empirical CDF; reduces to the Hazen definition for equal weights)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    if x.size == 0:
        return np.full(np.shape(q), np.nan)
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, x)


def object_size_stats(t: ObjectTable) -> dict:
    """Number density and equivalent-diameter statistics from an ObjectTable.

    * ``N_A`` (px⁻²) = Σ_interior w_i (Miles-Lantuéjoul), unbiased number density.
    * ``deq_mean``, ``deq_median``: number-weighted (weights w_i) mean / median of
      the equivalent diameter d_eq = 2 sqrt(a/π) (px).
    * ``deq_d10/d50/d90``: *area-weighted* quantiles (weights a_i w_i): d50 is the
      diameter below which half of the pore area of interior objects lies.
    * ``frac_area_interior``: fraction of the pore area in objects that do not
      touch the window (object-based sizes describe only this part).
    """
    out = dict(n_objects=t.n, n_border=int(t.border.sum()), N_A=np.nan, deq_mean=np.nan,
               deq_median=np.nan, deq_d10=np.nan, deq_d50=np.nan, deq_d90=np.nan,
               frac_area_interior=np.nan, mean_area_interior=np.nan)
    if t.n == 0:
        out["N_A"] = 0.0
        return out
    inner = ~t.border
    out["N_A"] = float(t.weight.sum())
    out["frac_area_interior"] = float(t.area[inner].sum() / t.area.sum())
    if inner.any():
        d = 2.0 * np.sqrt(t.area[inner] / math.pi)
        w = t.weight[inner]
        out["deq_mean"] = float(np.sum(w * d) / np.sum(w))
        out["deq_median"] = float(weighted_quantile(d, w, 0.5))
        out["deq_d10"], out["deq_d50"], out["deq_d90"] = map(
            float, weighted_quantile(d, w * t.area[inner], [0.1, 0.5, 0.9]))
        out["mean_area_interior"] = float(np.sum(w * t.area[inner]) / np.sum(w))
    return out


def sub_se_fraction(pore: np.ndarray, window: np.ndarray, se: np.ndarray) -> dict:
    """Fraction of the pore area (inside ``window``) lying in 8-connected pore
    objects that contain no translate of the structuring element ``se``.

    Such objects are deleted *entirely* by a morphological opening with
    ``se`` (an object survives an opening iff its erosion is non-empty), so
    this is the part of the porosity that a 3×3 open-close removes as
    'speckle'. Also returns the fraction of pore area removed by the opening
    (deleted objects + thin protrusions/throats of surviving objects).
    """
    pore = np.asarray(pore, bool)
    window = np.asarray(window, bool)
    a0 = int((pore & window).sum())
    if a0 == 0:
        return dict(frac_area_sub_se=np.nan, frac_area_removed_by_opening=np.nan)
    lab, n = ndi.label(pore, structure=S8)
    ero = ndi.binary_erosion(pore, structure=se, border_value=0)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[ero])] = True
    keep[0] = False
    sub = pore & ~keep[lab]
    opened = ndi.binary_dilation(ero, structure=se[::-1, ::-1]) & pore
    return dict(frac_area_sub_se=float((sub & window).sum() / a0),
                frac_area_removed_by_opening=float(1.0 - (opened & window).sum() / a0))


# =============================================================================
# granulometry (opening-based, pore-area-weighted size distribution)
# =============================================================================
def granulometry(pore: np.ndarray, window: np.ndarray, radii=DEFAULT_RADII) -> np.ndarray:
    """Pore-area fraction retained by openings with discrete discs D_r.

    ``G[k] = |γ_{r_k}(P) ∩ W| / |P ∩ W|`` (decreasing, G = 0 once no disc of
    radius r_k fits in the pore space). The structuring discs are
    D_r = {v ∈ Z² : |v| ≤ r}. Pixels outside the ROI are solid (the ROI
    boundary truncates pores: sizes of pores cut by the ROI are
    underestimated within ~r of the boundary).

    Exact Euclidean implementation: erosion = {d > r}, d = EDT of P; dilation
    of the eroded set by D_r = {dist ≤ r}, computed on the bounding box of the
    eroded set padded by r + 1 (fast once the eroded set is small).
    """
    pore = np.asarray(pore, bool)
    window = np.asarray(window, bool)
    a0 = int((pore & window).sum())
    G = np.zeros(len(radii))
    if a0 == 0:
        return np.full(len(radii), np.nan)
    d = ndi.distance_transform_edt(pore)
    h, w = pore.shape
    for k, r in enumerate(radii):
        E = d > r + 1e-9
        if not E.any():
            break                                   # all larger radii: G = 0
        ys = np.flatnonzero(E.any(axis=1))
        xs = np.flatnonzero(E.any(axis=0))
        pad = int(math.ceil(r)) + 1
        y0, y1 = max(ys[0] - pad, 0), min(ys[-1] + pad + 1, h)
        x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + pad + 1, w)
        sub = E[y0:y1, x0:x1]
        dist = ndi.distance_transform_edt(~sub)
        opened = dist <= r + 1e-9
        G[k] = (opened & window[y0:y1, x0:x1]).sum() / a0
    return G


def psd_quantiles(radii, G, q=(0.1, 0.5, 0.9)) -> dict:
    """Quantiles d_q (px, area-equivalent disc diameter) of the pore-area-weighted
    granulometric size distribution F(D) = 1 − G(r(D)).

    F(D_0 = 1.128 px) = 0 (every pore pixel is covered by a single pixel);
    d_q is interpolated linearly in log D between the grid points bracketing
    F = q. Returns NaN when the pore phase is empty, and +inf when the largest
    radius in the grid still retains more than 1 − q of the pore area.
    """
    G = np.asarray(G, float)
    if not np.all(np.isfinite(G)):
        return {f"d{int(round(100 * qq))}": np.nan for qq in q}
    D = np.r_[2.0 / math.sqrt(math.pi), [disc_equivalent_diameter(r) for r in radii]]
    F = np.r_[0.0, 1.0 - G]
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
    out = {}
    for qq in q:
        key = f"d{int(round(100 * qq))}"
        k = np.flatnonzero(F >= qq)
        if k.size == 0:
            out[key] = np.inf
            continue
        k = int(k[0])
        if k == 0:
            out[key] = float(D[0])
            continue
        f0, f1 = F[k - 1], F[k]
        t = (qq - f0) / (f1 - f0) if f1 > f0 else 1.0
        out[key] = float(math.exp(math.log(D[k - 1]) + t * (math.log(D[k]) - math.log(D[k - 1]))))
    return out


# =============================================================================
# skeleton descriptors
# =============================================================================
def _neighbour_stack(sk: np.ndarray) -> np.ndarray:
    """8 neighbours in circular order (N, NE, E, SE, S, SW, W, NW) as uint8."""
    p = np.pad(sk.astype(np.uint8), 1)
    h, w = sk.shape
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in offs])


def skeleton_metrics(sk: np.ndarray, window: np.ndarray) -> dict:
    """Length and node counts of a 1-px wide (8-connected) skeleton.

    * length (px): orthogonal links count 1, diagonal links √2; a diagonal
      link is ignored when one of the two pixels completing its 2×2 block is
      also skeleton (the path then runs through the orthogonal links; avoids
      double counting of 'corner triangles'). Only links with both pixels in
      ``window`` are counted (minus-sampling). Digital-length bias of this
      link estimator is ≤ ~8 % depending on orientation (Kulpa 1977).
    * end points: crossing number CN = 1 (CN = number of 0→1 transitions in
      the circular 8-neighbourhood; Rutovitz 1966); isolated points: no
      neighbour (skeletons of one-pixel pores); branch points: CN ≥ 3, with
      8-adjacent branch pixels merged into one junction, which is counted
      when its first pixel (raster order) lies in ``window``.

    Returns dict(length, n_end, n_branch, n_isolated, n_px) in px units.
    """
    sk = np.asarray(sk, bool)
    window = np.asarray(window, bool)
    length = 0.0
    for dy, dx, step in ((0, 1, 1.0), (1, 0, 1.0)):
        a, b = _pair_views(sk, dy, dx)
        wa, wb = _pair_views(window, dy, dx)
        length += step * int((a & b & wa & wb).sum())
    h, w = sk.shape
    # diagonal (i,j)-(i+1,j+1): completing pixels (i,j+1) and (i+1,j)
    a, b = sk[:-1, :-1], sk[1:, 1:]
    c1, c2 = sk[:-1, 1:], sk[1:, :-1]
    wa, wb = window[:-1, :-1], window[1:, 1:]
    length += SQRT2 * int((a & b & ~c1 & ~c2 & wa & wb).sum())
    # anti-diagonal (i,j+1)-(i+1,j): completing (i,j) and (i+1,j+1)
    a, b = sk[:-1, 1:], sk[1:, :-1]
    c1, c2 = sk[:-1, :-1], sk[1:, 1:]
    wa, wb = window[:-1, 1:], window[1:, :-1]
    length += SQRT2 * int((a & b & ~c1 & ~c2 & wa & wb).sum())

    nb = _neighbour_stack(sk)
    n8 = nb.sum(axis=0)
    cn = (np.abs(np.diff(np.concatenate([nb, nb[:1]]).astype(np.int8), axis=0)).sum(axis=0)) // 2
    inside = sk & window
    n_end = int((inside & (cn == 1)).sum())
    n_iso = int((inside & (n8 == 0)).sum())
    br = sk & (cn >= 3)
    lab, nbr = ndi.label(br, structure=S8)
    n_branch = 0
    if nbr:
        idx = np.flatnonzero(lab)
        first = ndi.minimum(idx, labels=lab.ravel()[idx], index=np.arange(1, nbr + 1)).astype(np.int64)
        n_branch = int(window.ravel()[first].sum())
    return dict(length=float(length), n_end=n_end, n_branch=n_branch, n_isolated=n_iso,
                n_px=int(inside.sum()))


# =============================================================================
# grey-scale (partial-volume) porosity
# =============================================================================
def silverman_bandwidth(v: np.ndarray) -> float:
    """Silverman's rule of thumb h = 0.9 min(σ, IQR/1.349) n^(-1/5)."""
    v = np.asarray(v, float)
    n = v.size
    sd = float(np.std(v, ddof=1))
    q75, q25 = np.percentile(v, [75, 25])
    s = min(sd, (q75 - q25) / 1.349) if q75 > q25 else sd
    return 0.9 * s * n ** (-0.2)


@dataclass
class GreyGrid:
    lo: float          # left edge of the first bin (raw grey level)
    width: float       # bin width (integer grey levels for integer data)
    nbins: int

    @property
    def centers(self) -> np.ndarray:
        return self.lo + self.width * (np.arange(self.nbins) + 0.5)

    def index(self, v: np.ndarray) -> np.ndarray:
        return np.clip(((np.asarray(v, float) - self.lo) // self.width).astype(np.int64), 0, self.nbins - 1)


def make_grid(v: np.ndarray, target_bins: int = 2048, integer: bool = True, pad_frac: float = 0.02) -> GreyGrid:
    """Histogram grid spanning the data; integer-aligned bins for integer data
    (avoids comb artefacts from uneven numbers of integer levels per bin)."""
    lo, hi = float(np.min(v)), float(np.max(v))
    span = max(hi - lo, 1.0)
    lo -= pad_frac * span
    hi += pad_frac * span
    width = (hi - lo) / target_bins
    if integer:
        width = max(1.0, float(math.ceil(width)))
        lo = math.floor(lo) - 0.5
    nb = int(math.ceil((hi - lo) / width)) + 1
    return GreyGrid(lo, width, nb)


def smooth_hist(counts: np.ndarray, grid: GreyGrid, bandwidth: float) -> np.ndarray:
    """Binned Gaussian KDE (density per grey level) along the last axis."""
    c = np.asarray(counts, np.float64)
    sig = max(bandwidth / grid.width, 0.5)
    s = ndi.gaussian_filter1d(c, sig, axis=-1, mode="constant", truncate=5.0)
    tot = c.sum(axis=-1, keepdims=True)
    return s / np.where(tot > 0, tot, 1.0) / grid.width


def refine_peak(dens: np.ndarray, i: int) -> float:
    """Sub-bin peak position (in bin units) by a parabola through 3 points."""
    if i <= 0 or i >= dens.size - 1:
        return float(i)
    y0, y1, y2 = dens[i - 1], dens[i], dens[i + 1]
    den = y0 - 2 * y1 + y2
    return float(i + (0.5 * (y0 - y2) / den if den < 0 else 0.0))


def mode_in_range(dens: np.ndarray, grid: GreyGrid, lo: float = -np.inf, hi: float = np.inf) -> float:
    """Grey level of the highest density within [lo, hi) (sub-bin refined).
    Works on 1-D densities or row-wise on a 2-D stack (returns an array)."""
    x = grid.centers
    sel = (x >= lo) & (x < hi)
    if not sel.any():
        return np.nan
    j = np.flatnonzero(sel)
    if dens.ndim == 1:
        i = int(j[np.argmax(dens[j])])
        return grid.lo + grid.width * (refine_peak(dens, i) + 0.5)
    sub = dens[:, j]
    ii = j[np.argmax(sub, axis=1)]
    out = np.empty(dens.shape[0])
    for r, i in enumerate(ii):
        out[r] = grid.lo + grid.width * (refine_peak(dens[r], int(i)) + 0.5)
    return out


def peak_top_centre(dens: np.ndarray, grid: GreyGrid, mode: float, frac: float = 0.5) -> float:
    """Centre of a Gaussian fitted (log-parabola, least squares) to the part of
    the peak around ``mode`` whose density is ≥ ``frac`` × the peak density.
    Robust to the asymmetric tails that partial-volume mixing adds to a peak."""
    x = grid.centers
    i = int(np.clip(round((mode - grid.lo) / grid.width - 0.5), 0, dens.size - 1))
    top = dens[i] * frac
    l = i
    while l > 0 and dens[l - 1] >= top:
        l -= 1
    r = i
    while r < dens.size - 1 and dens[r + 1] >= top:
        r += 1
    if r - l < 2:
        return float(mode)
    xs, ys = x[l:r + 1], np.log(dens[l:r + 1])
    c2, c1, _ = np.polyfit(xs - mode, ys, 2)
    if c2 >= 0:
        return float(mode)
    return float(mode - c1 / (2 * c2))


def phi_pv(mean_I, mu_s, mu_p):
    """Grey-scale porosity of the two-phase linear model: (μ_s − Ī)/(μ_s − μ_p)."""
    return (np.asarray(mu_s, float) - np.asarray(mean_I, float)) / (np.asarray(mu_s, float) - np.asarray(mu_p, float))


def tile_ids(window: np.ndarray, tile: int) -> tuple[np.ndarray, int]:
    """Tile index (row-major, tile × tile px) of every window pixel; -1 outside."""
    h, w = window.shape
    ny, nx = int(math.ceil(h / tile)), int(math.ceil(w / tile))
    yy, xx = np.mgrid[:h, :w]
    t = (yy // tile) * nx + (xx // tile)
    t = np.where(window, t, -1)
    return t, ny * nx


# =============================================================================
# small-sample statistics (unit = image)
# =============================================================================
def signed_rank_exact(d) -> dict:
    """Exact two-sided Wilcoxon signed-rank test by complete enumeration of the
    2^n sign flips (permutation distribution conditional on the observed
    midranks; zeros are ranked and keep a zero sign - Pratt 1959 - and ties
    get midranks, so the p-value stays exact with ties and zeros).

    Statistic T = Σ sign(d_i)·rank(|d_i|); p = P(|T*| ≥ |T_obs|).
    Matches ``scipy.stats.wilcoxon(method='exact')`` when there are no ties
    or zeros. NaNs are dropped (n reported). Practical for n ≤ 20.
    """
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return dict(n=0, T=np.nan, p=np.nan, n_pos=0, n_neg=0)
    r = rankdata(np.abs(d))
    s = np.sign(d)
    T = float(np.sum(s * r))
    a = np.abs(s) * r
    if n > 20:
        raise ValueError("exact enumeration limited to n <= 20")
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    Ts = signs @ a
    p = float(np.mean(np.abs(Ts) >= abs(T) - 1e-9))
    return dict(n=n, T=T, p=min(p, 1.0), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))


def holm(p) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (NaNs ignored and kept)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    o = ok[np.argsort(p[ok], kind="stable")]
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out[o] = np.minimum(adj, 1.0)
    return out


def boot_median_ci(x, rng: np.random.Generator, n_boot: int, level: float = 0.95) -> tuple[float, float, float]:
    """Median and percentile-bootstrap CI (resampling the units, e.g. images).
    NaNs are dropped. With n < 3 the CI is returned as NaN."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if x.size < 3:
        return med, np.nan, np.nan
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    bm = np.median(x[idx], axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(bm, [a, 1 - a])
    return med, float(lo), float(hi)
