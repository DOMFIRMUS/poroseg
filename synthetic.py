"""Synthetic phantoms with known ground truth and a micro-CT degradation model.

The real images of this study have no pixel-level ground truth: the
laboratory porosity is a bulk value that includes porosity below the image
resolution, and nobody knows which pixel is pore. This module creates images
for which the truth *is* known, degrades them with the physical effects that
make segmentation hard (partial volume, blur, noise, beam-hardening cupping),
and normalises them exactly as the real pipeline does, so that the same eight
thresholding algorithms and the same post-processing can be scored against
the truth (stage ``synthetic``; also used by the ``ml_baseline`` stage).

Public API (stable contract)
----------------------------
``make_phantom(kind, size, rng, supersample=4) -> dict``
    Binary two-phase microstructure drawn on a grid ``supersample`` times
    finer than the image grid and area-averaged to the image grid. Keys:

    ``truth_frac``  float32 (size, size) pore area fraction of each image
                    pixel, in [0, 1] (partial-volume pixels are fractional);
    ``truth``       bool (size, size), pixel-grid truth ``truth_frac >= 0.5``;
    ``phi_total``   float, total pore area fraction (fine-grid porosity, =
                    mean of ``truth_frac``);
    ``phi_pixel``   float, porosity of the pixel-grid truth (mean of ``truth``);
    ``phi_resolved`` float, area fraction of the *resolved* pore component
                    only (= ``phi_total`` except for ``'multiscale'``);
    ``resolved_frac`` float32 (size, size) area fraction of the resolved component;
    ``kind``, ``n_objects`` (dict), ``params`` (dict of the drawn parameters).

``degrade(truth_frac, blur_sigma, noise_sd, cupping, rng, mu_p=MU_P, mu_s=MU_S) -> float32``
    Forward model of a reconstructed attenuation image (see below).

``generate_case(kind, blur_sigma, noise_sd, cupping, rep, seed, size=512) -> dict``
    One complete, deterministic benchmark image: phantom (seeded by
    ``(seed, kind, rep)`` only, hence *shared* by every degradation level -
    common random numbers), degradation (seeded by every argument) and the
    pipeline normalisation. Keys:

    ``grey``       float32 (size, size) normalised image in [0, 1], 0 outside ``mask``
                   (identical to ``Context.norm(...).img`` for a real image);
    ``mask``       bool (size, size), the inscribed disc (analogue of the 'disk' ROI);
    ``eval_mask``  bool, ``mask`` minus a ``eval_border``-px rim (metrics are computed here);
    ``truth``, ``truth_frac``, ``resolved_frac`` as in :func:`make_phantom`;
    ``phi_total``, ``phi_pixel``, ``phi_resolved`` - computed INSIDE ``eval_mask``;
    ``raw``        float32 degraded image before normalisation;
    ``lo``, ``hi`` raw grey levels mapped to 0 and 1 by the normalisation;
    ``kind``, ``rep``, ``seed``, ``blur_sigma``, ``noise_sd``, ``cupping``.

    The streams are derived exactly like :meth:`poroseg.context.Context.rng`
    (SHA-256 of ``"seed|tag|..."``), so ``generate_case(..., seed=ctx.seed)``
    is reproducible from the project's master seed.

Helpers: :func:`derive_rng`, :func:`phantom` (cached ``make_phantom``),
:func:`disc_masks`, :func:`segment_all` (8 methods x area-preserving post
levels with the pipeline's own code), :func:`segmentation_metrics`,
:func:`euler2d`.

Phantoms
--------
``'grain_pack'`` (mimics the granular packs CF4-CF10)
    A 2-D section through a random packing of spheroidal grains. The section
    of a sphere of radius ``R`` cut at a height ``z`` uniformly distributed in
    ``[0, R)`` is a disc of radius ``a = R sqrt(1 - (z/R)^2)`` (Underwood
    1970); spheroids give ellipses with aspect ratio ``b/a`` in [0.75, 1].
    Sphere radii are U(25, 42) px and ``z/R`` is U(0, 0.9), so most grain
    sections are 40-84 px across with a tail of small caps. Grains are added
    until their area reaches ``1 - phi_target`` (``phi_target`` ~ U(0.32,
    0.42)) and are then packed without overlap by a force-biased relaxation
    (Moscinski et al. 1989) on a periodic square: every overlapping pair of
    circumscribed discs (inflated by half a minimum gap of 1.5 px) is pushed
    apart along the line of centres, the lighter grain moving more, until the
    largest overlap is < 0.05 px. The pore space is the complement of the
    grains, so pore throats between nearly touching grains are only a few
    pixels wide - exactly where blur and partial volume act. Realised
    2-D porosity is about 0.32-0.45.

``'boolean_pores'`` (mimics the resolved porosity of VN04)
    Boolean model (Serra 1982; Chiu et al. 2013) of overlapping pore discs
    in a solid matrix. Germs form a homogeneous Poisson process of intensity
    ``lambda``; radii are log-normal (median 4 px, log-s.d. 0.7) truncated to
    [0.75, 30] px. For a Boolean model the pore fraction is
    ``phi = 1 - exp(-lambda * E[pi r^2])``, so ``lambda`` is set from a
    target ``phi`` ~ U(0.11, 0.19) with the closed-form truncated log-normal
    moment ``E[r^2 | a < r < b] = exp(2 mu + 2 s^2) [Phi(beta - 2s) -
    Phi(alpha - 2s)] / [Phi(beta) - Phi(alpha)]``, ``alpha = (ln a - mu)/s``,
    ``beta = (ln b - mu)/s``.

``'multiscale'`` (resolved + sub-resolution porosity)
    Resolved Boolean pores (log-normal radii, median 6 px, log-s.d. 0.5,
    truncated to [2.5, 30] px; target fraction U(0.08, 0.14)) plus a dense
    Boolean population of micropores with radii U(0.5, 1.5) px (area
    fraction U(0.08, 0.14)). Everything is drawn on the 4x finer grid, so
    after area averaging most micropores only *lower the grey level* of matrix
    pixels (genuine partial-volume pixels) and very few reach the 0.5
    pixel-truth level. ``phi_total`` (fine grid) therefore exceeds
    ``phi_pixel`` and ``phi_resolved``: the gap is the unresolved porosity
    that no binary segmentation can recover - the synthetic analogue of the
    difference between image and laboratory porosity of the granular packs.

All phantoms are periodic (objects crossing an edge re-enter on the opposite
side), which makes the textures stationary and lets the blur use wrap-around
boundaries.

Degradation (forward) model
---------------------------
1. Linear attenuation with partial volume (each pixel is a mixture):
   ``I0 = mu_p + (mu_s - mu_p) (1 - f)`` with ``f`` = pore area fraction of the pixel
   (Ketcham & Carlson 2001). The detector integration over the pixel is
   already contained in the area averaging of the fine grid.
2. Point-spread function: isotropic Gaussian of s.d. ``blur_sigma`` px
   (``scipy.ndimage.gaussian_filter``, periodic boundary).
3. Cupping (beam-hardening-like low-frequency gain, Brooks & Di Chiro 1976),
   the same form as the perturbation stage applies to the real images:
   ``I2 = I1 [1 - c (1 - (r/R)^2)]`` with ``r`` the distance to the image
   centre and ``R`` the radius of the evaluation disc (gain clipped at 1 for
   ``r > R``): the centre is darker by a fraction ``c``.
4. Additive white Gaussian noise of s.d. ``noise_sd |mu_s - mu_p|`` (noise is
   expressed relative to the phase contrast, i.e. CNR = 1 / noise_sd).
   Noise is added after the gain because reconstruction noise is not scaled
   by the beam-hardening error.

``MU_P / MU_S = 0.60`` is the pore/solid grey-level ratio measured on the four
granular packs (two-component Gaussian mixture of the raw grey values in the
disc ROI gives 0.57-0.62; the ``synthetic`` stage recomputes and reports it).
The default noise levels of ``config.yaml`` (0.02-0.10 of the contrast) bracket
the pixel noise of the granular-pack images (about 0.05-0.13 of the contrast,
estimated from neighbouring-pixel differences).

Pixel truth and metrics
-----------------------
The pixel-grid truth is ``truth_frac >= 0.5`` (majority rule: a pixel is
pore if more than half of its area is pore); this is the best any binary
segmentation on the image grid can do in the absence of blur and noise.
Porosity error is measured against ``phi_pixel`` (and against ``phi_total``,
which differs materially only for ``'multiscale'``). Dice
``2|A n B| / (|A| + |B|)`` (Dice 1945) and Jaccard ``|A n B| / |A u B|``
(Jaccard 1912) are computed against the pixel truth inside ``eval_mask``.
The Euler number of the pore phase (Gray 1971; 8-connected pores,
4-connected solid; ``skimage.measure.euler_number(connectivity=2)``) is
number of pore clusters minus number of solid islands enclosed by pore; its
relative error is ``(chi_seg - chi_true) / max(|chi_true|, 1)``.

References
----------
Brooks RA, Di Chiro G (1976) Beam hardening in X-ray reconstructive tomography.
    Phys. Med. Biol. 21:390-398.
Chiu SN, Stoyan D, Kendall WS, Mecke J (2013) Stochastic Geometry and its
    Applications, 3rd ed. Wiley.
Dice LR (1945) Measures of the amount of ecologic association between species.
    Ecology 26:297-302.
Gray SB (1971) Local properties of binary images in two dimensions. IEEE Trans.
    Comput. C-20:551-561.
Jaccard P (1912) The distribution of the flora in the alpine zone. New Phytol.
    11:37-50.
Ketcham RA, Carlson WD (2001) Acquisition, optimization and interpretation of
    X-ray computed tomographic imagery: applications to the geosciences.
    Comput. Geosci. 27:381-400.
Moscinski J, Bargiel M, Rycerz ZA, Jacobs PWM (1989) The force-biased algorithm
    for the irregular close packing of equal hard spheres. Mol. Simul. 3:201-212.
Serra J (1982) Image Analysis and Mathematical Morphology. Academic Press.
Underwood EE (1970) Quantitative Stereology. Addison-Wesley.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr
from skimage.measure import euler_number

from . import postprocess, thresholds
from .metrics import dice as _dice
from .metrics import jaccard as _jaccard
from .roi import Geometry, disk_roi

KINDS = ("grain_pack", "boolean_pores", "multiscale")
KIND_LABELS = {"grain_pack": "Grain pack", "boolean_pores": "Boolean pores", "multiscale": "Multiscale"}

MU_P = 0.60     # grey level of pure pore (arbitrary units)
MU_S = 1.00     # grey level of pure solid; MU_P / MU_S = pore/solid ratio of the CF images

# Generator parameters (documented in the module docstring). Changing them
# changes the benchmark: they are recorded in each phantom's 'params'.
PARAMS: dict[str, dict[str, Any]] = {
    "grain_pack": dict(phi_target=(0.32, 0.42), sphere_radius=(25.0, 42.0), cut_height_max=0.9,
                       aspect=(0.75, 1.0), min_gap=1.5, tol=0.05, max_iter=6000),
    "boolean_pores": dict(phi_target=(0.11, 0.19), r_median=4.0, r_logsd=0.7, r_range=(0.75, 30.0)),
    "multiscale": dict(phi_resolved_target=(0.08, 0.14), r_median=6.0, r_logsd=0.5, r_range=(2.5, 30.0),
                       phi_micro_target=(0.10, 0.16), micro_r_range_fine=(0.5, 1.5)),
}


# ----------------------------------------------------------------------------
# random streams
# ----------------------------------------------------------------------------
def _tag(x: Any) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def derive_rng(seed: int, *tags: Any) -> np.random.Generator:
    """Independent reproducible stream, identical to ``Context.rng`` when
    ``seed`` is the project seed: SHA-256 of ``"seed|tag1|tag2..."``.
    Floats are formatted with ``%g`` so that 0.5 and 0.50 give the same stream."""
    h = hashlib.sha256(("|".join(_tag(t) for t in (seed,) + tags)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# ----------------------------------------------------------------------------
# drawing on the fine grid (periodic)
# ----------------------------------------------------------------------------
def _paint_ellipses(canvas: np.ndarray, s: int, L: float, cx, cy, a, b, theta) -> None:
    """OR ellipses into a periodic fine-grid canvas.

    Coordinates (``cx, cy, a, b``) are in image pixels on the continuous square
    [0, L); fine pixel (i, j) has centre ((j + 0.5)/s, (i + 0.5)/s).
    ``theta`` is the orientation of the semi-axis ``a``.
    """
    nf = canvas.shape[0]
    for x0, y0, ai, bi, th in zip(np.atleast_1d(cx), np.atleast_1d(cy), np.atleast_1d(a),
                                  np.atleast_1d(b), np.atleast_1d(theta)):
        c, sn = np.cos(th), np.sin(th)
        hx = np.hypot(ai * c, bi * sn)          # half extents of the rotated bounding box
        hy = np.hypot(ai * sn, bi * c)
        sxs = [0.0] + ([L] if x0 - hx < 0 else []) + ([-L] if x0 + hx > L else [])
        sys_ = [0.0] + ([L] if y0 - hy < 0 else []) + ([-L] if y0 + hy > L else [])
        for sx in sxs:
            xc = x0 + sx
            j0 = max(int(np.floor((xc - hx) * s - 0.5)), 0)
            j1 = min(int(np.ceil((xc + hx) * s + 0.5)), nf)
            if j0 >= j1:
                continue
            X = (np.arange(j0, j1) + 0.5) / s - xc
            for sy in sys_:
                yc = y0 + sy
                i0 = max(int(np.floor((yc - hy) * s - 0.5)), 0)
                i1 = min(int(np.ceil((yc + hy) * s + 0.5)), nf)
                if i0 >= i1:
                    continue
                Y = (np.arange(i0, i1) + 0.5) / s - yc
                u = (X[None, :] * c + Y[:, None] * sn) / ai
                v = (-X[None, :] * sn + Y[:, None] * c) / bi
                canvas[i0:i1, j0:j1] |= (u * u + v * v) <= 1.0


def _paint_small_discs(canvas: np.ndarray, s: int, x, y, r, chunk: int = 50000) -> None:
    """Vectorised, periodic rasterisation of many tiny discs (radius <= 2 fine px).

    A fine pixel is set when its centre lies within ``r`` of the disc centre;
    for a uniformly random centre the expected number of covered fine pixels
    equals the disc area, so the fine-grid porosity is unbiased."""
    nf = canvas.shape[0]
    off = np.arange(-2, 3)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    for k in range(0, len(x), chunk):
        fx = np.asarray(x[k:k + chunk]) * s - 0.5          # fine-index coordinates of the centre
        fy = np.asarray(y[k:k + chunk]) * s - 0.5
        rf = np.asarray(r[k:k + chunk]) * s
        jx = np.round(fx).astype(np.int64)[:, None] + ox[None, :]
        iy = np.round(fy).astype(np.int64)[:, None] + oy[None, :]
        inside = (jx - fx[:, None]) ** 2 + (iy - fy[:, None]) ** 2 <= (rf ** 2)[:, None]
        canvas[np.mod(iy[inside], nf), np.mod(jx[inside], nf)] = True


def _block_mean(fine: np.ndarray, s: int) -> np.ndarray:
    """Area average of an (n s, n s) fine grid to the (n, n) image grid."""
    n = fine.shape[0] // s
    return fine.reshape(n, s, n, s).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _truncated_lognormal(rng, n, median, logsd, lo, hi) -> np.ndarray:
    """Rejection sampler of a log-normal (``median, logsd``) truncated to [lo, hi]."""
    out = np.empty(0)
    mu = np.log(median)
    while out.size < n:
        r = np.exp(mu + logsd * rng.standard_normal(max(2 * (n - out.size), 16)))
        out = np.concatenate([out, r[(r >= lo) & (r <= hi)]])
    return out[:n]


def truncated_lognormal_m2(median: float, logsd: float, lo: float, hi: float) -> float:
    """E[r^2] of a log-normal truncated to [lo, hi] (closed form, see module docstring)."""
    mu, s = np.log(median), logsd
    al, be = (np.log(lo) - mu) / s, (np.log(hi) - mu) / s
    return float(np.exp(2 * mu + 2 * s * s) * (ndtr(be - 2 * s) - ndtr(al - 2 * s)) / (ndtr(be) - ndtr(al)))


def _boolean_discs(rng, L, phi_target, m2, sampler):
    """Germs and radii of a periodic Boolean model of discs with target cover ``phi_target``.

    ``lambda = -ln(1 - phi) / (pi E[r^2])``; number of germs ~ Poisson(lambda L^2)."""
    lam = -np.log1p(-phi_target) / (np.pi * m2)
    n = int(rng.poisson(lam * L * L))
    x, y = rng.uniform(0, L, n), rng.uniform(0, L, n)
    r = sampler(n)
    return x, y, r


# ----------------------------------------------------------------------------
# grain pack: force-biased relaxation of circumscribed discs
# ----------------------------------------------------------------------------
def relax_discs(pos: np.ndarray, radii: np.ndarray, L: float, tol: float = 0.05,
                max_iter: int = 6000, step: float = 1.0) -> tuple[np.ndarray, float, int]:
    """Remove the overlaps of discs in a periodic square by pairwise repulsion.

    At each iteration every overlapping pair (i, j) with overlap
    ``o_ij = r_i + r_j - d_ij > 0`` (minimum-image distance ``d_ij``) is
    separated along the line of centres; disc i moves by
    ``o_ij * w_j / (w_i + w_j)`` with ``w = r^2`` (area-weighted, so small
    grains yield). This is a simple force-biased scheme (Moscinski et al. 1989).

    Returns the relaxed positions, the final maximum overlap (px) and the
    number of iterations.
    """
    P = pos.astype(np.float64).copy()
    n = len(radii)
    w = radii ** 2
    wsum = w[:, None] + w[None, :]
    share = w[None, :] / wsum                    # fraction of the overlap that i takes
    rr = radii[:, None] + radii[None, :]
    eye = np.eye(n, dtype=bool)
    max_ov = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        d = P[:, None, :] - P[None, :, :]
        d -= L * np.round(d / L)
        dist = np.sqrt((d ** 2).sum(-1))
        dist[eye] = np.inf
        ov = rr - dist
        ov[ov < 0] = 0.0
        max_ov = float(ov.max()) if n > 1 else 0.0
        if max_ov < tol:
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(ov > 0, ov * share / np.maximum(dist, 1e-9), 0.0)
        disp = (f[:, :, None] * d).sum(axis=1)
        P = np.mod(P + step * disp, L)
    return P, max_ov, it


def _grain_pack(L: int, s: int, rng: np.random.Generator) -> dict:
    p = PARAMS["grain_pack"]
    phi_t = rng.uniform(*p["phi_target"])
    target_solid = (1.0 - phi_t) * L * L
    a_l, b_l, th_l = [], [], []
    area = 0.0
    while area < target_solid:
        R3 = rng.uniform(*p["sphere_radius"])
        z = rng.uniform(0.0, p["cut_height_max"])
        a = R3 * np.sqrt(1.0 - z * z)
        b = a * rng.uniform(*p["aspect"])
        a_l.append(a); b_l.append(b); th_l.append(rng.uniform(0.0, np.pi))
        area += np.pi * a * b
    a, b, th = np.array(a_l), np.array(b_l), np.array(th_l)
    pos = rng.uniform(0, L, (len(a), 2))
    env = a + 0.5 * p["min_gap"]
    shrink = 1.0
    P, max_ov, n_it = relax_discs(pos, env, L, p["tol"], p["max_iter"])
    while max_ov >= 0.5:                         # jammed: shrink everything by 1 % and retry
        shrink *= 0.99
        P, max_ov, k = relax_discs(P, env * shrink, L, p["tol"], p["max_iter"] // 2)
        n_it += k
    a, b = a * shrink, b * shrink
    solid = np.zeros((L * s, L * s), bool)
    _paint_ellipses(solid, s, L, P[:, 0], P[:, 1], a, b, th)
    pore = ~solid
    frac = _block_mean(pore, s)
    return dict(truth_frac=frac, resolved_frac=frac,
                n_objects=dict(grains=int(len(a))),
                params=dict(phi_target=float(phi_t), n_grains=int(len(a)), relax_iter=int(n_it),
                            relax_max_overlap_px=float(max_ov), shrink=float(shrink),
                            mean_section_diam_px=float(2 * np.mean(np.sqrt(a * b)))))


def _boolean_pores(L: int, s: int, rng: np.random.Generator) -> dict:
    p = PARAMS["boolean_pores"]
    phi_t = rng.uniform(*p["phi_target"])
    m2 = truncated_lognormal_m2(p["r_median"], p["r_logsd"], *p["r_range"])
    x, y, r = _boolean_discs(rng, L, phi_t, m2,
                             lambda n: _truncated_lognormal(rng, n, p["r_median"], p["r_logsd"], *p["r_range"]))
    pore = np.zeros((L * s, L * s), bool)
    _paint_ellipses(pore, s, L, x, y, r, r, np.zeros_like(r))
    frac = _block_mean(pore, s)
    return dict(truth_frac=frac, resolved_frac=frac, n_objects=dict(pores=int(len(r))),
                params=dict(phi_target=float(phi_t), n_pores=int(len(r)),
                            expected_phi=float(1 - np.exp(-len(r) / (L * L) * np.pi * m2))))


def _multiscale(L: int, s: int, rng: np.random.Generator) -> dict:
    p = PARAMS["multiscale"]
    phi_r = rng.uniform(*p["phi_resolved_target"])
    phi_m = rng.uniform(*p["phi_micro_target"])
    m2 = truncated_lognormal_m2(p["r_median"], p["r_logsd"], *p["r_range"])
    x, y, r = _boolean_discs(rng, L, phi_r, m2,
                             lambda n: _truncated_lognormal(rng, n, p["r_median"], p["r_logsd"], *p["r_range"]))
    lo, hi = (v / s for v in p["micro_r_range_fine"])          # fine-grid px -> image px
    m2_micro = (hi ** 3 - lo ** 3) / (3.0 * (hi - lo))          # E[r^2], r ~ U(lo, hi)
    xm, ym, rm = _boolean_discs(rng, L, phi_m, m2_micro, lambda n: rng.uniform(lo, hi, n))
    macro = np.zeros((L * s, L * s), bool)
    _paint_ellipses(macro, s, L, x, y, r, r, np.zeros_like(r))
    pore = macro.copy()
    _paint_small_discs(pore, s, xm, ym, rm)
    frac = _block_mean(pore, s)
    rfrac = _block_mean(macro, s)
    return dict(truth_frac=frac, resolved_frac=rfrac,
                n_objects=dict(pores=int(len(r)), micropores=int(len(rm))),
                params=dict(phi_resolved_target=float(phi_r), phi_micro_target=float(phi_m),
                            n_pores=int(len(r)), n_micropores=int(len(rm))))


_GENERATORS = {"grain_pack": _grain_pack, "boolean_pores": _boolean_pores, "multiscale": _multiscale}


def make_phantom(kind: str, size: int, rng: np.random.Generator, supersample: int = 4) -> dict:
    """Draw one phantom (see module docstring for the three kinds and the keys).

    Parameters
    ----------
    kind : {'grain_pack', 'boolean_pores', 'multiscale'}
    size : int
        Side of the square image grid (px).
    rng : numpy.random.Generator
        Source of randomness (use :func:`derive_rng` / ``Context.rng``).
    supersample : int
        Fine-grid factor; the binary microstructure is drawn on a
        ``size * supersample`` grid and area-averaged, which creates genuine
        partial-volume pixels at every interface.
    """
    if kind not in _GENERATORS:
        raise KeyError(f"unknown phantom kind '{kind}' (expected one of {KINDS})")
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    d = _GENERATORS[kind](int(size), int(supersample), rng)
    frac = d["truth_frac"]
    truth = frac >= 0.5
    d.update(kind=kind, truth=truth, phi_total=float(frac.mean(dtype=np.float64)),
             phi_pixel=float(truth.mean()), phi_resolved=float(d["resolved_frac"].mean(dtype=np.float64)),
             supersample=int(supersample))
    return d


@lru_cache(maxsize=24)
def _phantom_cached(kind: str, size: int, rep: int, seed: int, supersample: int) -> dict:
    d = make_phantom(kind, size, derive_rng(seed, "synthetic", "phantom", kind, rep), supersample)
    for k in ("truth_frac", "truth", "resolved_frac"):
        d[k].setflags(write=False)                # cached arrays are shared: make them read-only
    return d


def phantom(kind: str, size: int, rep: int, seed: int, supersample: int = 4) -> dict:
    """Cached phantom for replicate ``rep`` (stream ``derive_rng(seed, 'synthetic', 'phantom', kind, rep)``).
    Arrays are read-only (shared between calls)."""
    return _phantom_cached(str(kind), int(size), int(rep), int(seed), int(supersample))


# ----------------------------------------------------------------------------
# degradation and pipeline normalisation
# ----------------------------------------------------------------------------
def cupping_gain(shape: tuple[int, int], cupping: float, radius: float | None = None,
                 center: tuple[float, float] | None = None) -> np.ndarray:
    """Multiplicative radial gain ``1 - c (1 - min(r/R, 1)^2)`` (centre darker by ``c``)."""
    h, w = shape
    cy, cx = center if center is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    R = float(radius) if radius is not None else min(h, w) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rho2 = np.minimum(((yy - cy) ** 2 + (xx - cx) ** 2) / (R * R), 1.0)
    return (1.0 - cupping * (1.0 - rho2)).astype(np.float32)


def degrade(truth_frac: np.ndarray, blur_sigma: float, noise_sd: float, cupping: float,
            rng: np.random.Generator, mu_p: float = MU_P, mu_s: float = MU_S, *,
            radius: float | None = None, center: tuple[float, float] | None = None) -> np.ndarray:
    """Forward model: partial volume -> Gaussian PSF -> cupping -> additive noise.

    Parameters
    ----------
    truth_frac : (H, W) array in [0, 1]
        Pore area fraction per pixel.
    blur_sigma : float
        s.d. of the Gaussian PSF (px); 0 = no blur. Periodic boundaries.
    noise_sd : float
        s.d. of the additive white Gaussian noise **relative to the phase
        contrast** ``|mu_s - mu_p|`` (CNR = 1/noise_sd).
    cupping : float
        Relative darkening ``c`` of the centre, gain ``1 - c (1 - (r/R)^2)``.
    rng : numpy.random.Generator
        Stream for the noise.
    mu_p, mu_s : float
        Grey levels of pure pore and pure solid.
    radius, center : optional
        ``R`` and centre of the cupping profile (default: inscribed circle).

    Returns
    -------
    float32 array, the simulated reconstructed grey image.
    """
    f = np.asarray(truth_frac, dtype=np.float32)
    img = (mu_p + (mu_s - mu_p) * (1.0 - f)).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        img = ndi.gaussian_filter(img, float(blur_sigma), mode="wrap")
    if cupping and cupping != 0:
        img = img * cupping_gain(img.shape, float(cupping), radius, center)
    if noise_sd and noise_sd > 0:
        img = img + np.float32(noise_sd * abs(mu_s - mu_p)) * rng.standard_normal(img.shape, dtype=np.float32)
    return img.astype(np.float32, copy=False)


@lru_cache(maxsize=8)
def _disc_masks_cached(size: int, eval_border: int):
    r = size / 2.0 - 1.0
    c = (size - 1) / 2.0
    g = Geometry(cx=c, cy=c, r_boundary=r, r_usable=r, holder="synthetic", source="synthetic")
    R = disk_roi((size, size), g, int(eval_border))
    mask = np.zeros((size, size), bool)
    ev = np.zeros((size, size), bool)
    mask[R.slices] = R.mask
    ev[R.slices] = R.eval_mask
    mask.setflags(write=False)
    ev.setflags(write=False)
    return mask, ev, r, (c, c)


def disc_masks(size: int, eval_border: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Inscribed-disc ROI (radius ``size/2 - 1``) and its evaluation mask
    (rim of ``eval_border`` px removed), built with :func:`poroseg.roi.disk_roi`
    exactly as for the real images. Read-only arrays."""
    m, e, _, _ = _disc_masks_cached(int(size), int(eval_border))
    return m, e


def generate_case(kind: str, blur_sigma: float, noise_sd: float, cupping: float, rep: int, seed: int,
                  size: int = 512, *, supersample: int = 4, mu_p: float = MU_P, mu_s: float = MU_S,
                  p_low: float = 0.1, p_high: float = 99.9, eval_border: int = 2) -> dict:
    """One benchmark image (deterministic for given arguments); see module docstring.

    ``p_low``, ``p_high`` and ``eval_border`` default to the values of
    ``config.yaml`` (``normalization`` and ``roi.eval_border_px``); the stage
    passes the configured values explicitly.
    """
    ph = phantom(kind, size, rep, seed, supersample)
    mask, ev, R, ctr = _disc_masks_cached(int(size), int(eval_border))
    rng = derive_rng(seed, "synthetic", "degrade", kind, int(rep), float(blur_sigma), float(noise_sd),
                     float(cupping))
    raw = degrade(ph["truth_frac"], blur_sigma, noise_sd, cupping, rng, mu_p, mu_s, radius=R, center=ctr)
    grey, lo, hi = thresholds.normalize(raw, mask, p_low, p_high)
    grey[~mask] = 0.0
    tf = ph["truth_frac"]
    return dict(grey=grey, raw=raw, mask=mask, eval_mask=ev, truth=ph["truth"], truth_frac=tf,
                resolved_frac=ph["resolved_frac"],
                phi_total=float(tf[ev].mean(dtype=np.float64)), phi_pixel=float(ph["truth"][ev].mean()),
                phi_resolved=float(ph["resolved_frac"][ev].mean(dtype=np.float64)),
                lo=lo, hi=hi, kind=kind, rep=int(rep), seed=int(seed), blur_sigma=float(blur_sigma),
                noise_sd=float(noise_sd), cupping=float(cupping))


# ----------------------------------------------------------------------------
# segmentation with the pipeline's own code, and metrics
# ----------------------------------------------------------------------------
def segment_all(case: dict, cfg: dict, methods=None, posts=postprocess.AREA_PRESERVING) -> dict:
    """Apply the pipeline's thresholding (default parameters from ``cfg``) and
    post-processing to a case. Returns ``{(method, post): bool pore image}``
    plus ``{('threshold', method): ThresholdResult}``."""
    methods = list(methods or cfg["thresholds"]["methods"])
    out: dict = {}
    for m in methods:
        T = thresholds.compute_threshold(m, case["grey"], case["mask"], thresholds.default_params(m, cfg))
        p0 = thresholds.segment(case["grey"], T, case["mask"])
        out[("threshold", m)] = T
        for p in posts:
            out[(m, p)] = p0 if p == "none" else postprocess.apply(p0, p, cfg, case["mask"])
    return out


def euler2d(pore: np.ndarray, eval_mask: np.ndarray | None = None, connectivity: int = 2) -> int:
    """Euler number of the pore phase (clusters - enclosed solid islands) inside ``eval_mask``.
    ``connectivity=2``: 8-connected pores / 4-connected solid (scikit-image)."""
    a = pore if eval_mask is None else (pore & eval_mask)
    return int(euler_number(np.ascontiguousarray(a), connectivity=connectivity))


def segmentation_metrics(pore: np.ndarray, truth: np.ndarray, eval_mask: np.ndarray, phi_true: float,
                         phi_total: float | None = None, chi_true: int | None = None,
                         connectivity: int = 2) -> dict:
    """Accuracy of one binary pore image against the pixel truth.

    Returns ``porosity`` (fraction, inside ``eval_mask``), ``error_pp`` =
    100 (porosity - phi_true) and ``abs_error_pp``; ``error_total_pp`` /
    ``abs_error_total_pp`` against ``phi_total`` (if given); ``dice``,
    ``jaccard`` (vs ``truth``); ``euler``, ``euler_true`` and
    ``euler_rel_error = (chi - chi_true) / max(|chi_true|, 1)``.
    """
    n = int(eval_mask.sum())
    phi = float(np.count_nonzero(pore & eval_mask) / n)
    out = dict(porosity=phi, error_pp=100.0 * (phi - phi_true), abs_error_pp=100.0 * abs(phi - phi_true),
               dice=_dice(pore, truth, eval_mask), jaccard=_jaccard(pore, truth, eval_mask))
    if phi_total is not None:
        out["error_total_pp"] = 100.0 * (phi - phi_total)
        out["abs_error_total_pp"] = abs(out["error_total_pp"])
    chi = euler2d(pore, eval_mask, connectivity)
    if chi_true is None:
        chi_true = euler2d(truth, eval_mask, connectivity)
    out.update(euler=chi, euler_true=int(chi_true), euler_rel_error=(chi - chi_true) / max(abs(chi_true), 1))
    return out
