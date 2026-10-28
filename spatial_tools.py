"""Spatial statistics of binary (pore / solid) images inside an arbitrary mask.

Pure, image-agnostic functions used by the ``spatial`` stage. Everything here
is deterministic except the resampling helpers, which take an explicit
``numpy.random.Generator``.

Notation
--------
``I(x)``  pore indicator (1 = pore, 0 = solid) of pixel ``x``;
``W(x)``  support (evaluation mask) indicator; only pixels with ``W = 1`` exist;
``phi``   porosity = area fraction of pore inside the support
          (Delesse principle: unbiased estimator of the volume fraction of a
          stationary medium; Weibel 1979);
``h``     a lag vector (pixels); ``r = |h|``.

Two-point statistics (masked)
-----------------------------
For a field observed through a window ``W`` the usual circular FFT estimators
are biased by the window edge. The *mask-normalised* (pair-counting) estimators

    S2(h)  = sum_x W(x) W(x+h) I(x) I(x+h)            / N(h)
    C(h)   = sum_x W(x) W(x+h) (I(x)-phi)(I(x+h)-phi)  / N(h)
    N(h)   = sum_x W(x) W(x+h)        (number of pixel pairs at lag h)

average only over pairs of pixels that are both inside the support
(Torquato 2002, Random Heterogeneous Materials, ch. 2; Jiao, Stillinger &
Torquato 2007, Phys. Rev. E 76:031110 for the FFT implementation; the masked
normalisation is the standard "normalised cross-correlation" of
Padfield 2012, IEEE Trans. Image Process. 21:2706-2718). Every sum is an
auto-correlation and is evaluated with zero-padded real FFTs; padding to
``n + max_lag`` per axis is enough to avoid wrap-around for ``|h_i| <= max_lag``.

``rho(h) = C(h) / (phi (1 - phi))`` is the normalised covariance
(``rho(0) = 1``). For a stationary medium ``S2(h) - phi^2 = C(h)``; with a mask
the two differ slightly because the pair subsets have their own means, so we
use the centred form for every derived quantity.

Correlation length and integral range
-------------------------------------
* ``l_c``  (correlation length): the first lag at which the radially averaged
  ``rho(r)`` drops below ``1/e`` (linear interpolation between integer-radius
  bins). This is the e-folding length of an exponential covariance
  (Debye, Anderson & Brumberger 1957, J. Appl. Phys. 28:679).
* ``lambda`` (integral scale) = int_0^{r0} rho(r) dr, with ``r0`` below.
* ``A`` (2-D integral range; Matheron 1971; Lantuejoul 1991, Math. Geol.
  23:149-160; Lantuejoul 2002, Geostatistical Simulation, ch. 3)

      A = int_{R^2} rho(h) dh   (px^2),

  so that for a window of area ``S >> A``

      Var[phi_S] ~= phi (1 - phi) A / S.                         (1)

  ``A`` is the area that behaves as one independent "pixel" (Kanit et al. 2003,
  Int. J. Solids Struct. 40:3647-3679). For an iid Bernoulli field ``A = 1 px^2``.
  **Truncation rule**: the empirical covariance is noisy at large lags and,
  because ``phi`` is estimated from the same window, it is biased negative
  there. We integrate ``rho(h)`` over the lattice disc ``|h| <= r0`` where
  ``r0`` is the first zero crossing of the radially averaged ``rho(r)``. If
  ``rho`` never crosses zero within ``max_lag`` (long-range trend, e.g. cupping
  or a wall effect) we integrate up to ``max_lag`` and flag the estimate as
  *not converged* (it is then a lower bound of the trend-inflated value). For
  media with a "hole effect" (negative lobe of ``rho``, e.g. grain packs) the
  positive-lobe truncation over-estimates ``A`` (conservative SE).

Representative elementary area (REV) by tiling
----------------------------------------------
Non-overlapping ``L x L`` tiles lying entirely inside the support give tile
porosities ``phi_L``; their variance decays as ``L^-2`` (SD as ``L^-1``) once
``L >> l_c`` (eq. 1 with ``S = L^2``; Kanit et al. 2003). The side needed for a
single window to estimate ``phi`` within ``+/- e`` with probability
``1 - alpha`` is

      L_REV = z_{1-alpha/2} sqrt(phi (1 - phi) A) / e            (2)

and the same criterion for a region of area ``S`` holds iff ``S >= L_REV^2``.

Uncertainty of a window porosity
--------------------------------
* block bootstrap (Kunsch 1989, Ann. Stat. 17:1217-1241; Lahiri 2003,
  Resampling Methods for Dependent Data; Hall 1985 for spatial tiles): tiles of
  side ``b >= 3 l_c`` are nearly independent; resampling them with replacement
  gives the sampling distribution of the mean tile porosity;
* integral range: ``SE_IR = sqrt(phi (1 - phi) A / S)`` (eq. 1);
* naive pixel-iid bootstrap: resampling the ``N`` pixels with replacement makes
  the pore count ``Binomial(N, phi)`` *exactly*, so the bootstrap distribution
  of the porosity is ``Binomial(N, phi)/N`` and ``SE_iid = sqrt(phi(1-phi)/N)``
  (Efron & Tibshirani 1993). It ignores spatial correlation, i.e. it treats
  ``N / A`` effective samples as ``N`` (pseudo-replication; Hurlbert 1984,
  Ecol. Monogr. 54:187-211); its CI is too narrow by ``~ sqrt(A)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.stats import norm

INV_E = float(np.exp(-1.0))


# ============================================================================
# two-point statistics with mask normalisation
# ============================================================================
def _lag_indices(n: int, max_lag: int) -> np.ndarray:
    """Indices into a length-``n`` circular FFT output for lags -max_lag..max_lag."""
    return np.arange(-max_lag, max_lag + 1) % n


class MaskedCorrelator:
    """Pair-counting auto-correlation of fields observed through a fixed mask.

    The FFT of the mask (and the pair counts ``N(h)``) is computed once and
    re-used for every indicator image on the same support.

    Parameters
    ----------
    mask : (ny, nx) bool
        Support ``W``.
    max_lag : int
        Largest lag (px) along each axis; output arrays have shape
        ``(2 max_lag + 1, 2 max_lag + 1)`` with lag (0, 0) at the centre.
    workers : int
        Threads used by ``scipy.fft``.
    min_pairs_frac : float
        Lags with fewer than ``min_pairs_frac * N(0)`` pairs are set to NaN.
    """

    def __init__(self, mask: np.ndarray, max_lag: int, workers: int = 2, min_pairs_frac: float = 0.05):
        mask = np.asarray(mask, bool)
        ny, nx = mask.shape
        max_lag = int(min(max_lag, ny - 1, nx - 1))
        self.mask = mask
        self.max_lag = max_lag
        self.workers = workers
        self.shape = (sfft.next_fast_len(ny + max_lag, real=True), sfft.next_fast_len(nx + max_lag, real=True))
        self._iy = _lag_indices(self.shape[0], max_lag)
        self._ix = _lag_indices(self.shape[1], max_lag)
        w = mask.astype(np.float64)
        npairs = np.rint(self._autocorr(w))
        self.n0 = float(npairs[max_lag, max_lag])
        self.npairs = npairs
        self.valid = npairs >= max(1.0, min_pairs_frac * self.n0)
        k = np.arange(-max_lag, max_lag + 1)
        self.lag_r = np.hypot(k[:, None], k[None, :])

    def _autocorr(self, f: np.ndarray) -> np.ndarray:
        F = sfft.rfft2(f, s=self.shape, workers=self.workers)
        F *= np.conj(F)
        ac = sfft.irfft2(F, s=self.shape, workers=self.workers)
        return ac[np.ix_(self._iy, self._ix)]

    def two_point(self, indicator: np.ndarray, phi: float | None = None) -> dict:
        """S2(h), centred covariance C(h) and normalised covariance rho(h).

        Parameters
        ----------
        indicator : bool array, same shape as the mask (values outside the mask ignored)
        phi : float, optional
            Porosity used for centring; default = mean of the indicator on the mask.

        Returns
        -------
        dict with ``phi``, ``s2``, ``cov``, ``rho`` (2-D lag arrays, NaN where
        the pair count is too small), ``npairs`` and ``lag_r``.
        """
        ind = np.asarray(indicator, bool) & self.mask
        if phi is None:
            phi = float(ind.sum() / self.mask.sum())
        f = ind.astype(np.float64)
        num_s2 = self._autocorr(f)
        g = np.where(self.mask, f - phi, 0.0)
        num_c = self._autocorr(g)
        with np.errstate(invalid="ignore", divide="ignore"):
            s2 = np.where(self.valid, num_s2 / self.npairs, np.nan)
            cov = np.where(self.valid, num_c / self.npairs, np.nan)
        var = phi * (1.0 - phi)
        rho = cov / var if var > 0 else np.full_like(cov, np.nan)
        return dict(phi=phi, s2=s2, cov=cov, rho=rho, num_c=num_c, npairs=self.npairs, lag_r=self.lag_r)


def two_point_bruteforce(indicator: np.ndarray, mask: np.ndarray, max_lag: int, phi: float | None = None):
    """Direct (O(N max_lag^2)) pair-counting estimator - reference for tests."""
    ind = np.asarray(indicator, bool) & mask
    if phi is None:
        phi = ind.sum() / mask.sum()
    ny, nx = mask.shape
    L = 2 * max_lag + 1
    s2 = np.full((L, L), np.nan)
    cov = np.full((L, L), np.nan)
    npairs = np.zeros((L, L))
    f = ind.astype(float)
    g = np.where(mask, f - phi, 0.0)
    w = mask.astype(float)
    for i, dy in enumerate(range(-max_lag, max_lag + 1)):
        for j, dx in enumerate(range(-max_lag, max_lag + 1)):
            ys = slice(max(0, -dy), min(ny, ny - dy))
            xs = slice(max(0, -dx), min(nx, nx - dx))
            yt = slice(max(0, dy), min(ny, ny + dy))
            xt = slice(max(0, dx), min(nx, nx + dx))
            n = (w[ys, xs] * w[yt, xt]).sum()
            npairs[i, j] = n
            if n > 0:
                s2[i, j] = (f[ys, xs] * f[yt, xt]).sum() / n
                cov[i, j] = (g[ys, xs] * g[yt, xt]).sum() / n
    return dict(phi=phi, s2=s2, cov=cov, npairs=npairs)


def radial_average(field: np.ndarray, weights: np.ndarray, lag_r: np.ndarray, max_r: int | None = None):
    """Pair-weighted radial average over integer-radius bins ``round(|h|) = k``.

    With ``field = C(h)`` and ``weights = N(h)`` this is the pooled estimator
    ``sum_{h in bin} num(h) / sum_{h in bin} N(h)``.

    Returns
    -------
    r : (K,) int bin radii 0..max_r;  value : (K,) float;  w : (K,) total weight.
    """
    k = np.rint(lag_r).astype(int)
    ok = np.isfinite(field) & (weights > 0)
    if max_r is None:
        max_r = int((lag_r.shape[0] - 1) // 2)
    sel = ok & (k <= max_r)
    num = np.bincount(k[sel], weights=(field * weights)[sel], minlength=max_r + 1)[: max_r + 1]
    den = np.bincount(k[sel], weights=weights[sel], minlength=max_r + 1)[: max_r + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(den > 0, num / den, np.nan)
    return np.arange(max_r + 1), val, den


def first_crossing(r: np.ndarray, y: np.ndarray, level: float) -> float:
    """First ``r`` (linear interpolation) where ``y`` falls to or below ``level``.

    Returns NaN if ``y`` never reaches ``level`` (finite values only).
    """
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    good = np.isfinite(y)
    r, y = r[good], y[good]
    below = np.nonzero(y <= level)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(r[0])
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(r[i])
    return float(r[i - 1] + (level - y0) * (r[i] - r[i - 1]) / (y1 - y0))


def correlation_length(r: np.ndarray, rho: np.ndarray, level: float = INV_E) -> float:
    """Correlation length ``l_c``: first lag with ``rho(r) <= 1/e`` (px)."""
    return first_crossing(r, rho, level)


def integral_scale(r: np.ndarray, rho: np.ndarray, r_trunc: float) -> float:
    """1-D integral scale ``lambda = int_0^{r_trunc} rho(r) dr`` (trapezoid, px)."""
    r = np.asarray(r, float)
    rho = np.asarray(rho, float)
    sel = (r <= r_trunc) & np.isfinite(rho)
    if sel.sum() < 2:
        return float("nan")
    return float(np.trapezoid(rho[sel], r[sel]))


@dataclass
class CovarianceSummary:
    phi: float
    lc_px: float          # correlation length (1/e)
    r0_px: float          # truncation radius (first zero crossing, or max_lag)
    converged: bool       # True if rho crossed zero within max_lag
    scale_px: float       # 1-D integral scale
    A_px2: float          # 2-D integral range (truncated as documented)
    r: np.ndarray
    rho_r: np.ndarray
    s2_r: np.ndarray
    npairs_r: np.ndarray


def summarize_covariance(tp: dict, max_r: int | None = None) -> CovarianceSummary:
    """Correlation length, integral scale and integral range from ``two_point`` output.

    Truncation rule for ``A`` (see module docstring): lattice sum of ``rho(h)``
    over ``|h| <= r0``, ``r0`` = first zero crossing of the radial ``rho(r)``;
    if there is none, ``r0 = max_r`` and ``converged = False``.
    """
    lag_r = tp["lag_r"]
    if max_r is None:
        max_r = int((lag_r.shape[0] - 1) // 2)
    r, rho_r, w = radial_average(tp["rho"], tp["npairs"], lag_r, max_r)
    _, s2_r, _ = radial_average(tp["s2"], tp["npairs"], lag_r, max_r)
    lc = correlation_length(r, rho_r)
    r0 = first_crossing(r, rho_r, 0.0)
    converged = bool(np.isfinite(r0))
    if not converged:
        last = np.nonzero(np.isfinite(rho_r))[0]
        r0 = float(r[last[-1]]) if len(last) else float("nan")
    sel = (lag_r <= r0) & np.isfinite(tp["rho"])
    A = float(np.sum(tp["rho"][sel])) if np.isfinite(r0) else float("nan")
    lam = integral_scale(r, rho_r, r0)
    return CovarianceSummary(phi=float(tp["phi"]), lc_px=lc, r0_px=float(r0), converged=converged,
                             scale_px=lam, A_px2=A, r=r, rho_r=rho_r, s2_r=s2_r, npairs_r=w)


def running_integral_range(tp: dict, max_r: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Running (truncated) integral range ``A(R) = sum_{|h| <= R} rho(h)`` (px^2).

    Documents the truncation rule: for an iid field ``A(R) = 1`` for every
    ``R``; for a stationary correlated field ``A(R)`` rises and levels off at
    the integral range once ``R`` exceeds the correlation range; for a medium
    with a hole effect it overshoots and comes back down (negative lobe); for
    a field with a large-scale trend it keeps growing (no convergence).

    Returns
    -------
    R : (K,) float, truncation radii 0..max_r (px);  A : (K,) float.
    """
    lag_r = tp["lag_r"]
    if max_r is None:
        max_r = int((lag_r.shape[0] - 1) // 2)
    rho = tp["rho"]
    ok = np.isfinite(rho)
    # exact radii |h| (not rounded): sort the finite lags by radius and accumulate
    rr = lag_r[ok]
    vv = rho[ok]
    o = np.argsort(rr, kind="stable")
    rr, cs = rr[o], np.cumsum(vv[o])
    R = np.arange(max_r + 1, dtype=float)
    idx = np.searchsorted(rr, R, side="right") - 1
    A = np.where(idx >= 0, cs[np.clip(idx, 0, None)], np.nan)
    return R, A


# ============================================================================
# tiling
# ============================================================================
def summed_area_table(a: np.ndarray) -> np.ndarray:
    """Zero-padded summed-area table: ``S[i, j] = sum(a[:i, :j])`` (int64)."""
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    np.cumsum(np.cumsum(a, axis=0, dtype=np.int64), axis=1, out=S[1:, 1:])
    return S


def _box_sums(S: np.ndarray, y0: np.ndarray, x0: np.ndarray, L: int) -> np.ndarray:
    return S[y0 + L, x0 + L] - S[y0, x0 + L] - S[y0 + L, x0] + S[y0, x0]


@dataclass
class Tiling:
    L: int
    y0: np.ndarray        # tile top-left corners (bbox coordinates)
    x0: np.ndarray
    gi: np.ndarray        # integer grid indices (row, col) of each tile
    gj: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.y0))


def tile_grid(mask: np.ndarray, L: int, mask_sat: np.ndarray | None = None) -> Tiling:
    """All non-overlapping ``L x L`` tiles that lie entirely inside ``mask``.

    The regular grid is centred on the bounding box of the mask (symmetric
    leftover margins), which maximises the number of complete tiles in a disc
    for a centred grid and makes the result independent of the bbox padding.
    """
    ys, xs = np.nonzero(np.any(mask, axis=1))[0], np.nonzero(np.any(mask, axis=0))[0]
    if len(ys) == 0 or L <= 0:
        e = np.zeros(0, int)
        return Tiling(L, e, e, e, e)
    by0, by1, bx0, bx1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    ny, nx = (by1 - by0) // L, (bx1 - bx0) // L
    if ny == 0 or nx == 0:
        e = np.zeros(0, int)
        return Tiling(L, e, e, e, e)
    oy = by0 + ((by1 - by0) - ny * L) // 2
    ox = bx0 + ((bx1 - bx0) - nx * L) // 2
    gi, gj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    y0, x0 = oy + gi * L, ox + gj * L
    S = summed_area_table(mask) if mask_sat is None else mask_sat
    full = _box_sums(S, y0, x0, L) == L * L
    return Tiling(L, y0[full], x0[full], gi[full], gj[full])


def tile_fractions(pore: np.ndarray, tiling: Tiling, pore_sat: np.ndarray | None = None) -> np.ndarray:
    """Pore fraction of every tile of ``tiling`` (tiles are fully inside the mask)."""
    if tiling.n == 0:
        return np.zeros(0)
    S = summed_area_table(pore) if pore_sat is None else pore_sat
    return _box_sums(S, tiling.y0, tiling.x0, tiling.L) / float(tiling.L ** 2)


def morans_i(values: np.ndarray, gi: np.ndarray, gj: np.ndarray) -> float:
    """Moran's I of tile values with rook (4-neighbour) binary weights.

    ``I = (n / S0) * sum_{i~j} z_i z_j / sum_i z_i^2`` with ``z = v - mean(v)``
    and ``S0`` the number of ordered neighbour pairs (Moran 1950, Biometrika
    37:17-23). Used only as a descriptive diagnostic of residual correlation
    between adjacent bootstrap blocks (E[I] = -1/(n-1) without correlation).
    """
    v = np.asarray(values, float)
    n = len(v)
    if n < 3:
        return float("nan")
    z = v - v.mean()
    den = np.sum(z * z)
    if den == 0:
        return float("nan")
    pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(gi, gj))}
    num, S0 = 0.0, 0
    for k, (a, b) in enumerate(zip(gi, gj)):
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            m = pos.get((int(a) + da, int(b) + db))
            if m is not None:
                num += z[k] * z[m]
                S0 += 1
    if S0 == 0:
        return float("nan")
    return float(n / S0 * num / den)


# ============================================================================
# variance scaling, REV
# ============================================================================
def z_value(ci_level: float = 0.95) -> float:
    return float(norm.ppf(0.5 + ci_level / 2.0))


def se_integral_range(phi: float, A_px2: float, area_px: float) -> float:
    """``SE = sqrt(phi (1 - phi) A / S)`` (eq. 1); fraction units."""
    return float(np.sqrt(phi * (1.0 - phi) * A_px2 / area_px))


def sd_window_theory(phi: float, A_px2: float, L_px: np.ndarray) -> np.ndarray:
    """Theoretical SD of the porosity of an ``L x L`` window (eq. 1)."""
    return np.sqrt(phi * (1.0 - phi) * A_px2) / np.asarray(L_px, float)


def l_rev_theory(phi: float, A_px2: float, e: float, z: float = 1.959964) -> float:
    """Window side (px) whose (1-alpha) half-width equals ``e`` (eq. 2)."""
    return float(z * np.sqrt(phi * (1.0 - phi) * A_px2) / e)


def fit_variance_scaling(L: np.ndarray, sd: np.ndarray, n: np.ndarray, phi: float,
                         L_min: float = 0.0, min_tiles: int = 10) -> dict:
    """Kanit et al. (2003) fit of tile variance vs tile area.

    Uses tile sizes with ``L >= L_min`` and at least ``min_tiles`` tiles.
    Returns the integral range at the theoretical exponent
    (``A_tile = median(Var * L^2) / (phi (1-phi))``, robust to single noisy
    sizes) and the free log-log slope ``beta`` of SD vs L (theory: -1),
    estimated by least squares weighted by ``n - 1`` (the d.o.f. of each SD).
    """
    L = np.asarray(L, float)
    sd = np.asarray(sd, float)
    n = np.asarray(n, float)
    sel = (L >= L_min) & (n >= min_tiles) & np.isfinite(sd) & (sd > 0)
    out = dict(n_sizes=int(sel.sum()), A_tile_px2=float("nan"), slope=float("nan"))
    if sel.sum() >= 1 and 0 < phi < 1:
        out["A_tile_px2"] = float(np.median(sd[sel] ** 2 * L[sel] ** 2) / (phi * (1 - phi)))
    if sel.sum() >= 2:
        x, y, w = np.log(L[sel]), np.log(sd[sel]), n[sel] - 1
        X = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        out["slope"] = float(beta[1])
    return out


def l_rev_empirical(L: np.ndarray, sd: np.ndarray, n: np.ndarray, e: float, z: float = 1.959964,
                    min_tiles: int = 10) -> tuple[float, str]:
    """Smallest window side (px) whose single-window half-width ``z SD(L) <= e``.

    Interpolates log SD vs log L between measured tile sizes (only sizes with
    ``>= min_tiles`` tiles are trusted). If the criterion is not met at the
    largest trusted size, extrapolates from that size with the theoretical
    slope -1 (``L_REV = L_last z SD_last / e``) and returns status
    ``'extrapolated'``; if it is met already at the smallest size, returns that
    size with status ``'<='``.
    """
    L = np.asarray(L, float)
    sd = np.asarray(sd, float)
    n = np.asarray(n, float)
    sel = (n >= min_tiles) & np.isfinite(sd) & (sd > 0)
    if sel.sum() == 0:
        return float("nan"), "undefined"
    L, hw = L[sel], z * sd[sel]
    o = np.argsort(L)
    L, hw = L[o], hw[o]
    if hw[0] <= e:
        return float(L[0]), "<="
    idx = np.nonzero(hw <= e)[0]
    if len(idx):
        i = idx[0]
        x0, x1, y0, y1 = np.log(L[i - 1]), np.log(L[i]), np.log(hw[i - 1]), np.log(hw[i])
        x = x0 + (np.log(e) - y0) * (x1 - x0) / (y1 - y0)
        return float(np.exp(x)), "interpolated"
    return float(L[-1] * hw[-1] / e), "extrapolated"


def choose_block_size(sizes: list[int], n_tiles: dict[int, int], lc_px: float, b_min: int,
                      min_blocks: int = 20) -> tuple[int, str]:
    """Block (tile) side for the block bootstrap.

    Rule: the smallest configured tile size ``b >= max(b_min, 3 l_c)`` that
    still yields ``>= min_blocks`` complete tiles. If no size satisfies both
    (``l_c`` unknown/too long, or ROI too small), the largest size with
    ``>= min_blocks`` tiles is used and the returned status says why.
    """
    sizes = sorted(int(s) for s in sizes)
    need = max(float(b_min), 3.0 * lc_px) if np.isfinite(lc_px) else np.inf
    ok = [s for s in sizes if s >= need and n_tiles.get(s, 0) >= min_blocks]
    if ok:
        return ok[0], "ok"
    enough = [s for s in sizes if n_tiles.get(s, 0) >= min_blocks]
    if not enough:
        s = max(sizes, key=lambda q: (n_tiles.get(q, 0), -q))
        return s, "too_few_blocks"
    return enough[-1], ("b<3lc" if np.isfinite(lc_px) else "lc_undefined")


# ============================================================================
# resampling
# ============================================================================
def block_bootstrap_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                         ci_level: float = 0.95, chunk: int = 2000) -> dict:
    """Bootstrap of the mean of (approximately independent) block values.

    Resamples the ``n`` blocks with replacement ``n_boot`` times; returns the
    observed mean, the bootstrap SE (SD of the replicates, ddof=1) and the
    percentile interval (Efron & Tibshirani 1993, ch. 13).
    """
    v = np.asarray(values, float)
    n = len(v)
    if n < 2:
        return dict(mean=float(v.mean()) if n else float("nan"), se=float("nan"), lo=float("nan"),
                    hi=float("nan"), n=n)
    reps = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        reps[s:s + k] = v[rng.integers(0, n, size=(k, n))].mean(axis=1)
    a = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return dict(mean=float(v.mean()), se=float(reps.std(ddof=1)), lo=float(lo), hi=float(hi), n=n)


def iid_bootstrap_binomial(phi: float, n_pixels: int, n_boot: int, rng: np.random.Generator,
                           ci_level: float = 0.95) -> dict:
    """Naive pixel-iid bootstrap of a porosity, computed exactly.

    Resampling ``N`` binary pixels with replacement, the number of pore pixels
    in a replicate is ``Binomial(N, phi)``; drawing it directly is the same
    bootstrap at O(n_boot) cost. ``se_analytic = sqrt(phi(1-phi)/N)``.
    """
    reps = rng.binomial(int(n_pixels), float(phi), size=int(n_boot)) / float(n_pixels)
    a = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return dict(se_analytic=float(np.sqrt(phi * (1 - phi) / n_pixels)), se_boot=float(reps.std(ddof=1)),
                lo=float(lo), hi=float(hi))


def iid_bootstrap_explicit(pixels: np.ndarray, n_boot: int, rng: np.random.Generator,
                           ci_level: float = 0.95) -> dict:
    """Literal pixel-iid bootstrap (resample pixel values) - verification only."""
    v = np.asarray(pixels).astype(np.uint8).ravel()
    n = len(v)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = v[rng.integers(0, n, size=n)].mean()
    a = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(reps, [a, 1.0 - a])
    return dict(se_boot=float(reps.std(ddof=1)), lo=float(lo), hi=float(hi))


# ============================================================================
# radial profiles
# ============================================================================
def radial_profile(pore: np.ndarray, mask: np.ndarray, dist_px: np.ndarray, bin_px: float,
                   d_max: float | None = None) -> dict:
    """Porosity in rings of width ``bin_px`` of a distance map.

    Parameters
    ----------
    pore, mask : bool arrays (same shape)
    dist_px : float array, distance (px) of every pixel to the reference
        (holder wall / core edge, or centre)
    bin_px : ring width (px)

    Returns
    -------
    dict with ring edges ``lo``, ``hi``, centre ``mid``, pixel-mean distance
    ``d_mean`` (differs from ``mid`` for partially covered rings), ``porosity``
    and pixel count ``n`` (rings with n = 0 are omitted). The pixel-weighted
    mean of the ring porosities equals the porosity of the whole mask.
    """
    d = dist_px[mask]
    p = pore[mask]
    k = np.floor(d / bin_px).astype(int)
    k = np.clip(k, 0, None)
    if d_max is not None:
        keep = d < d_max
        k, p, d = k[keep], p[keep], d[keep]
    nb = int(k.max()) + 1 if len(k) else 0
    n = np.bincount(k, minlength=nb)
    s = np.bincount(k, weights=p.astype(float), minlength=nb)
    sd = np.bincount(k, weights=d.astype(float), minlength=nb)
    ok = n > 0
    idx = np.nonzero(ok)[0]
    return dict(lo=idx * bin_px, hi=(idx + 1) * bin_px, mid=(idx + 0.5) * bin_px,
                d_mean=sd[ok] / n[ok], porosity=s[ok] / n[ok], n=n[ok], k=idx)
