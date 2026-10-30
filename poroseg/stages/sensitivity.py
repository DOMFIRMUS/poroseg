"""Stage 'sensitivity': how strongly does image porosity depend on the threshold?

This stage replaces the 'Thermal Sensitivity Matrix (TSM)' and the 'Pore
Stability Index (PSI = 1 - |dPorosity/dIntensity|)' of the legacy manuscript,
which were never implemented and were not dimensionless (dPorosity/dIntensity
has units of 1/grey level, so its value changed with the bit depth or the
scaling of the image). Everything below is expressed in *normalised grey
level*: every ROI is rescaled to [0, 1] by its robust (0.1-99.9 percentile)
dynamic range before thresholding (``thresholds.normalize``), so an offset
``delta = 0.05`` means "5 % of the robust dynamic range" in every image and
all indices are dimensionless and comparable across samples. Analysis set:
primary ROI (largest usable disc), no post-processing, all eight methods,
metrics evaluated on ``ROI.eval_mask`` (ROI minus a 2 px rim).

1. Porosity-threshold curve
---------------------------
For a global threshold ``t`` the image porosity is the empirical cumulative
distribution function (ECDF) of the N grey levels I_i inside the evaluation
mask,

    phi(t) = (1/N) * sum_i 1{I_i <= t} = F_N(t),                         (1)

because the pore phase is ``I <= T`` (low attenuation). Each global method is a
point (T_m, phi(T_m)) on this single curve; the methods differ only in where
they put T. Local adaptive methods (Sauvola, Niblack) compare each pixel with
its own threshold T(x): their porosity is *not* a point of curve (1) and is
shown at the mean of the threshold map for orientation only.

2. Threshold Sensitivity Matrix (TSM)
-------------------------------------
For an additive offset ``delta`` applied to the threshold (to the whole map for
local methods),

    phi_m(delta) = (1/N) * sum_i 1{I_i - T_m(x_i) <= delta} = F_{R,m}(delta), (2)

i.e. the ECDF of the residuals R_i = I_i - T_m(x_i) evaluated at ``delta``.
For global methods T_m(x) is constant and (2) reduces to phi(T_m + delta) of
(1). The TSM stores, for every sample x method x delta (config
``sensitivity.tsm_deltas``), the absolute change 100 [phi_m(delta) -
phi_m(0)] (percentage points, pp) and the relative change [phi_m(delta) -
phi_m(0)] / phi_m(0).

3. Pore Stability Index (PSI), elasticity and absolute sensitivity
------------------------------------------------------------------
With delta = ``sensitivity.psi_delta`` (0.05),

    PSI_delta = 1 - min(1, |phi_m(+delta) - phi_m(-delta)| / (2 phi_m(0)))  (3)

which is dimensionless, bounded in [0, 1], and equals 1 when the porosity is
insensitive to a +/- delta shift of the threshold. Equivalently, PSI = 1 -
min(1, P(|R| <= delta) / (2 phi)): one minus half the fraction of ROI pixels
lying within +/- delta of the threshold, relative to the porosity.

Why PSI is high in a histogram valley and low on a flank. Let f be the
probability density of the grey levels inside the ROI and F its distribution
function. The expectation of (1) is F(t) = integral_{-inf}^{t} f(u) du, so by
the fundamental theorem of calculus

    d phi / d t = F'(t) = f(t),                                          (4)

i.e. the slope of the porosity-threshold curve *is* the grey-level histogram
density at the threshold. The central difference used here,

    D_h(T) = [phi(T + h) - phi(T - h)] / (2h)
           = (1 / 2hN) * #{i : T - h < I_i <= T + h},                      (5)

is exactly a box-kernel (bandwidth h) density estimate of f at T. Inserting
(5) with h = delta into (3) gives

    PSI_delta = 1 - min(1, delta * D_delta(T) / phi(T))
             ~= 1 - min(1, (delta / T) * |E|),                           (6)

with E the elasticity defined below. A threshold placed in the valley of a
bimodal histogram (small f(T)) therefore has a PSI close to 1, whereas a
threshold on a flank or at the mode of a unimodal histogram (large f(T))
has a low PSI. For local methods (2) gives d phi / d delta = f_R(0), the density
of the residual image I - T(x) at zero, and the same interpretation holds
with respect to the residuals.

Elasticity (dimensionless, percentage change of porosity per percentage
change of threshold):

    E = (d phi / d t) * (T / phi) ~= D_h(T) * T / phi(T),                 (7)

estimated with a narrower central difference h = ``sensitivity.deriv_h``
(default 0.01, i.e. ~2.6 bins of the 256-bin histogram); T is the mean of the
map for local methods.

Absolute sensitivity in percentage points per 0.01 of normalised grey level:

    S = 100 * D_h(T) * 0.01 = D_h(T)   [pp per 0.01]                     (8)

(numerically equal to the density f(T) in units of 1/normalised grey level).

Important: a high PSI does NOT imply an accurate porosity. A method that
returns a fixed quantile of the noise (Niblack, see item 5) can be insensitive
to the threshold *and* systematically wrong. PSI must be read together with
the agreement with the laboratory reference.

4. Local-parameter sweep
------------------------
Sauvola (T = m [1 + k (s / r - 1)], r = 0.5) and Niblack (T = m - k s) are run
for every window w in ``sensitivity.sweep_windows`` and every k in
``sweep_sauvola_k`` / ``sweep_niblack_k``. Porosity is reported for every
sample and, for the 7 samples with a laboratory value, the error against the
laboratory reference (``ctx.lab_ref.ref``).

5. Homogeneous-region false porosity
------------------------------------
Every method is applied to synthetic *homogeneous* fields (true porosity 0):
Gaussian noise of mean 0.7 and SD in {0.01, 0.03, 0.06} (normalised units, as
for a matrix region of a normalised image), a field with a smooth linear
intensity gradient (0.6 -> 0.8 across the field, SD 0.03), a homogeneous dark
'pore' region (mean 0.2, SD 0.03; true porosity 1) and the matrix field
re-normalised to [0, 1] as the pipeline does when the *whole* ROI is
homogeneous. Analytic expectations for a Gaussian field N(mu, sigma^2) when the
local mean and s.d. are estimated without error (window >> 1 px):

* Niblack, T = m - k s:  P(I <= mu - k sigma) = Phi(-k), independent of mu and
  sigma (Phi(-0.8) = 0.212). Niblack therefore labels ~21 % of ANY region
  wider than its window as pore - matrix or pore alike.
* Sauvola, T = m [1 + k (s / r - 1)]:  P(I <= T) = Phi(k mu (1/r - 1/sigma)).
* Global methods split any unimodal histogram somewhere; the false porosity of
  a homogeneous field is the quantile at which the method cuts the noise.

Statistics
----------
The experimental unit is the image (n = 9; n = 7 with a laboratory value).
Per-method summaries are medians over samples with 95 % percentile-bootstrap
CIs (resampling samples; Efron & Tibshirani 1993) and, for reference, the
distribution-free order-statistic CI of the median (Conover 1999, sec. 3.2).
Differences in PSI among the eight methods are tested with the Friedman test
(subject = sample; Friedman 1937), with Kendall's coefficient of concordance
W = chi2_F / (n (k - 1)) as effect size (Kendall & Babington Smith 1939), an
asymptotic chi-square p-value and a within-sample permutation p-value (more
reliable for n = 9). Pairwise exact Wilcoxon signed-rank tests with
Hodges-Lehmann shifts are Holm-adjusted (Holm 1979); with n = 9 the smallest
attainable exact two-sided p is 2/2^9 = 0.0039 > 0.05/28, so no pairwise
comparison among 8 methods can survive Holm adjustment - power is
insufficient by design and this is reported, not hidden. Method failures
(porosity < 5 % or > 90 %: Yen/Triangle on TQ06, VN01, VN02 - the cut is
placed at the natural gap of the data, next lowest value 7.6 %) are kept in
the primary analysis; a sensitivity analysis excludes them.

References
----------
Conover W.J. (1999) Practical Nonparametric Statistics, 3rd ed. Wiley.
Efron B., Tibshirani R.J. (1993) An Introduction to the Bootstrap. Chapman & Hall.
Friedman M. (1937) J. Am. Stat. Assoc. 32:675-701.
Hodges J.L., Lehmann E.L. (1963) Ann. Math. Stat. 34:598-611.
Holm S. (1979) Scand. J. Stat. 6:65-70.
Kendall M.G., Babington Smith B. (1939) Ann. Math. Stat. 10:275-287.
Niblack W. (1986) An Introduction to Digital Image Processing. Prentice-Hall.
Sauvola J., Pietikainen M. (2000) Pattern Recogn. 33:225-236.
Sezgin M., Sankur B. (2004) J. Electron. Imaging 13:146-168.

Outputs
-------
data/threshold_curves.csv           phi(t) on a 501-point grid, per sample
data/threshold_operating_points.csv threshold and porosity of every method; lab-matching threshold
data/tsm.csv                        sample x method x delta: phi, absolute (pp) and relative change
data/psi.csv                        contract: sample_id, label, group, method, family, phi, t, psi,
                                    elasticity, sens_pp_per_001 (+ diagnostics)
data/psi_method_summary.csv         per-method medians, bootstrap and order-statistic CIs
data/psi_pairwise.csv               pairwise Wilcoxon / Hodges-Lehmann / Holm
data/local_sweep.csv                Sauvola/Niblack window x k sweep, porosity and error vs lab
data/homogeneous_false_porosity.csv every method on synthetic homogeneous fields
figures: threshold_porosity_curves (supp), threshold_sensitivity_matrix (main),
         psi_by_method (main), local_parameter_sweep (supp)
tables:  psi_summary (main), homogeneous_false_porosity (supp)
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats

from .. import style
from .. import thresholds as pthr
from ..metrics import porosity as _porosity

FAIL_LO, FAIL_HI = 0.05, 0.90     # gross method failures (porosity < 5 % or > 90 %)
DERIV_H_DEFAULT = 0.01            # half-width of the central difference for E and S
CURVE_GRID = np.linspace(0.0, 1.0, 501)


# =============================================================================
# pure functions: porosity as a function of a threshold offset
# =============================================================================
def sorted_residuals(img: np.ndarray, thr, eval_mask: np.ndarray) -> np.ndarray:
    """Sorted residuals R_i = I_i - T(x_i) of the pixels inside ``eval_mask``.

    ``thr`` is a :class:`poroseg.thresholds.ThresholdResult`, a scalar or a
    threshold map with the shape of ``img``. A scalar threshold is first cast
    to the image dtype, exactly as NumPy does in ``img <= T`` (``segment``),
    so ``porosity_at_offsets(res, 0)`` reproduces the porosity of the binary
    image bit for bit. Residuals are returned in float64.
    """
    t = thr.value if isinstance(thr, pthr.ThresholdResult) else thr
    v = img[eval_mask].astype(np.float64)
    if np.ndim(t) == 0:
        t64 = float(np.asarray(t, dtype=img.dtype)) if np.issubdtype(img.dtype, np.floating) else float(t)
        r = v - t64
    else:
        t_arr = np.asarray(t)
        r = v - t_arr[eval_mask].astype(np.float64)
    r.sort()
    return r


def porosity_at_offsets(res_sorted: np.ndarray, deltas) -> np.ndarray:
    """phi(delta) = fraction of residuals <= delta (eq. 2), vectorised over deltas.

    For a global threshold this is phi(T + delta) of the porosity-threshold
    curve (eq. 1); for a local threshold map it is the porosity obtained when
    ``delta`` is added to the whole map.
    """
    d = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
    n = res_sorted.size
    if n == 0:
        return np.full(d.shape, np.nan)
    return np.searchsorted(res_sorted, d, side="right") / n


def psi_index(phi0, phi_plus, phi_minus) -> np.ndarray:
    """Pore Stability Index, eq. (3): 1 - min(1, |phi(+d) - phi(-d)| / (2 phi(0))).

    Dimensionless, in [0, 1], 1 = insensitive. If phi(0) = 0 the relative
    change is undefined; PSI is then 0 when the band contains pixels
    (any shift creates porosity from nothing) and 1 otherwise.
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    band = np.abs(np.asarray(phi_plus, dtype=np.float64) - np.asarray(phi_minus, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(phi0 > 0, band / (2.0 * phi0), np.where(band > 0, np.inf, 0.0))
    return 1.0 - np.minimum(1.0, ratio)


def central_slope(phi_plus_h, phi_minus_h, h: float) -> np.ndarray:
    """Central difference D_h = [phi(T+h) - phi(T-h)] / (2h) (eq. 5), in 1/(normalised grey level).

    For a global threshold this is a box-kernel estimate of the grey-level
    density f(T) (eq. 4)."""
    return (np.asarray(phi_plus_h, float) - np.asarray(phi_minus_h, float)) / (2.0 * h)


def elasticity(slope, t, phi0) -> np.ndarray:
    """E = (d phi / d t) (T / phi) (eq. 7); NaN when phi = 0."""
    phi0 = np.asarray(phi0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(phi0 > 0, np.asarray(slope, float) * np.asarray(t, float) / phi0, np.nan)


def slope_pp_per_001(slope) -> np.ndarray:
    """S = 100 * (d phi / d t) * 0.01: percentage points per 0.01 of normalised grey level (eq. 8)."""
    return np.asarray(slope, float) * 100.0 * 0.01


def histogram_density(values: np.ndarray, t: float, nbins: int = 256) -> float:
    """Density (1/normalised grey level) of the ``nbins``-bin histogram on [0, 1]
    in the bin that contains ``t`` - the quantity eq. (4) says d phi/dt equals."""
    counts, edges = np.histogram(values, bins=nbins, range=(0.0, 1.0))
    j = int(np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbins - 1))
    return float(counts[j] / (values.size * (edges[1] - edges[0])))


def lab_matching_threshold(values_sorted: np.ndarray, phi_target: float) -> float:
    """Smallest global threshold t with phi(t) >= phi_target (generalised inverse of eq. 1)."""
    n = values_sorted.size
    if n == 0 or not np.isfinite(phi_target):
        return np.nan
    j = int(np.clip(math.ceil(phi_target * n) - 1, 0, n - 1))
    return float(values_sorted[j])


# =============================================================================
# pure functions: statistics over samples
# =============================================================================
def bootstrap_median_ci(x, n_boot: int, rng: np.random.Generator, level: float = 0.95):
    """Median and percentile-bootstrap CI of the median (resampling the units in ``x``).

    NaNs are ignored. Returns (median, lo, hi). With n < 10 the bootstrap
    distribution of a median is discrete (it can only take order statistics
    and their midpoints), so the interval is coarse."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        return float(x[0]), np.nan, np.nan
    idx = rng.integers(0, x.size, size=(int(n_boot), x.size))
    meds = np.median(x[idx], axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(meds, [a, 1.0 - a])
    return float(np.median(x)), float(lo), float(hi)


def order_statistic_median_ci(x, level: float = 0.95):
    """Distribution-free CI of the median from order statistics (Conover 1999).

    Returns (lo, hi, coverage) with (lo, hi) = (x_(j), x_(n-j+1)) for the
    largest j such that the exact binomial coverage 1 - 2 P(B <= j - 1) >=
    ``level``, B ~ Bin(n, 1/2). For n = 9: (x_(2), x_(8)), coverage 0.961."""
    x = np.sort(np.asarray(x, dtype=np.float64)[np.isfinite(np.asarray(x, dtype=np.float64))])
    n = x.size
    best = None
    for j in range(1, n // 2 + 1):
        cov = 1.0 - 2.0 * stats.binom.cdf(j - 1, n, 0.5)
        if cov >= level:
            best = (j, cov)
    if best is None:
        return np.nan, np.nan, np.nan
    j, cov = best
    return float(x[j - 1]), float(x[n - j]), float(cov)


def friedman_kendall(X) -> dict:
    """Friedman test (rows = subjects, columns = treatments) with tie correction
    and Kendall's W = chi2_F / (n (k - 1)).

    chi2_F = [12 / (n k (k+1)) * sum_j R_j^2 - 3 n (k+1)] / C,
    C = 1 - sum(t^3 - t) / (n k (k^2 - 1)), R_j = rank sum of treatment j,
    ranks taken within each row (average ranks for ties). Identical to
    ``scipy.stats.friedmanchisquare`` and ``pingouin.friedman``."""
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    R = stats.rankdata(X, axis=1)
    Rj = R.sum(axis=0)
    ties = 0.0
    for row in X:
        _, c = np.unique(row, return_counts=True)
        ties += float(np.sum(c ** 3 - c))
    C = 1.0 - ties / (n * k * (k * k - 1))
    if C <= 0:
        return dict(chi2=np.nan, df=k - 1, p=np.nan, W=np.nan, n=n, k=k, mean_ranks=Rj / n)
    chi2 = (12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3.0 * n * (k + 1)) / C
    return dict(chi2=float(chi2), df=k - 1, p=float(stats.chi2.sf(chi2, k - 1)),
                W=float(chi2 / (n * (k - 1))), n=n, k=k, mean_ranks=Rj / n)


def friedman_permutation_p(X, n_perm: int, rng: np.random.Generator) -> float:
    """Permutation p-value of the Friedman statistic: method labels are permuted
    independently within each subject (exchangeability under H0). The tie
    correction is invariant under these permutations, so sum_j R_j^2 is used
    as the test statistic. p = (1 + #{T* >= T_obs}) / (1 + n_perm)."""
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    R = stats.rankdata(X, axis=1)
    obs = np.sum(R.sum(axis=0) ** 2)
    hits = 0
    done = 0
    chunk = 5000
    while done < n_perm:
        b = min(chunk, n_perm - done)
        idx = np.argsort(rng.random((b, n, k)), axis=2)
        Rp = np.take_along_axis(np.broadcast_to(R, (b, n, k)), idx, axis=2)
        stat = np.sum(Rp.sum(axis=1) ** 2, axis=1)
        hits += int(np.sum(stat >= obs - 1e-9))
        done += b
    return (1.0 + hits) / (1.0 + n_perm)


def hodges_lehmann(d) -> float:
    """One-sample Hodges-Lehmann estimator: median of the Walsh averages (d_i + d_j)/2, i <= j."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    i, j = np.triu_indices(d.size)
    return float(np.median((d[i] + d[j]) / 2.0))


def pairwise_wilcoxon_holm(wide: pd.DataFrame) -> pd.DataFrame:
    """Exact (or, with ties/zeros, scipy 'auto') Wilcoxon signed-rank tests for all
    column pairs of ``wide`` (rows = subjects), Hodges-Lehmann shift and Holm
    adjustment over the whole family."""
    from statsmodels.stats.multitest import multipletests
    cols = list(wide.columns)
    rows = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            d = (wide[cols[a]] - wide[cols[b]]).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if d.size < 2 or np.all(d == 0):
                p = 1.0
            else:
                try:
                    p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
                except ValueError:
                    p = 1.0
            rows.append(dict(method_a=cols[a], method_b=cols[b], n=int(d.size),
                             hl_shift=hodges_lehmann(d), median_diff=float(np.median(d)) if d.size else np.nan,
                             p=p))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out["p"].to_numpy(), method="holm")[1]
    return out


def min_exact_wilcoxon_p(n: int) -> float:
    """Smallest attainable two-sided exact p of the signed-rank test with n pairs: 2 / 2^n."""
    return 2.0 / 2.0 ** n


# =============================================================================
# pure functions: homogeneous fields (true porosity 0 or 1)
# =============================================================================
def homogeneous_field(size: int, mean: float, sd: float, rng: np.random.Generator,
                      gradient: tuple[float, float] | None = None) -> np.ndarray:
    """Gaussian noise field (float32, clipped to [0, 1]). With ``gradient=(a, b)``
    the mean increases linearly from a to b across the columns (smooth shading,
    e.g. residual beam hardening) instead of being constant."""
    if gradient is None:
        base = np.full((size, size), mean, dtype=np.float64)
    else:
        base = np.broadcast_to(np.linspace(gradient[0], gradient[1], size)[None, :], (size, size))
    img = base + rng.normal(0.0, sd, (size, size))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def field_porosity(img: np.ndarray, method: str, cfg: dict, renormalise: bool = False,
                   rim: int = 2, params: dict | None = None) -> tuple[float, float]:
    """Porosity returned by ``method`` on a whole field (mask = field), evaluated
    without a ``rim``-px border, exactly as the pipeline does for a ROI.

    ``renormalise=True`` first applies the pipeline's robust percentile
    normalisation (the case of a ROI that is homogeneous as a whole).
    Returns (porosity, threshold summary)."""
    mask = np.ones(img.shape, bool)
    if renormalise:
        n = cfg["normalization"]
        img, _, _ = pthr.normalize(img, mask, n["p_low"], n["p_high"])
    p = pthr.default_params(method, cfg)
    if params:
        p.update(params)
    T = pthr.compute_threshold(method, img, mask, p)
    pore = pthr.segment(img, T, mask)
    ev = mask.copy()
    if rim > 0:
        ev[:rim, :] = ev[-rim:, :] = ev[:, :rim] = ev[:, -rim:] = False
    return _porosity(pore, ev), float(T.summary)


def niblack_false_porosity_analytic(k: float) -> float:
    """Niblack on N(mu, sigma^2): P(I <= mu - k sigma) = Phi(-k)."""
    return float(stats.norm.cdf(-k))


def sauvola_false_porosity_analytic(mu: float, sigma: float, k: float, r: float) -> float:
    """Sauvola on N(mu, sigma^2): T = mu [1 + k (sigma/r - 1)] -> Phi(k mu (1/r - 1/sigma))."""
    return float(stats.norm.cdf(k * mu * (1.0 / r - 1.0 / sigma)))


def renormalised_moments(p_low: float, p_high: float) -> tuple[float, float]:
    """Mean and s.d. of a Gaussian field after percentile normalisation to [0, 1]."""
    zl, zh = stats.norm.ppf(p_low / 100.0), stats.norm.ppf(p_high / 100.0)
    return float(-zl / (zh - zl)), float(1.0 / (zh - zl))


# =============================================================================
# helpers bound to the Context
# =============================================================================
def _release(ctx, sid: str) -> None:
    """Drop the large per-sample arrays held by the Context caches (memory budget).

    The Context keeps every threshold map it computes (``_thr`` is unbounded);
    with ~40 parameter combinations x 9 samples that would exceed 2 GB, so the
    stage evicts what it no longer needs. Purely a runtime cache operation."""
    try:
        ctx._raw.pop(sid, None)
        for cache in (ctx._norm, ctx._thr, ctx._pore):
            for key in [k for k in cache if k[0] == sid]:
                cache.pop(key, None)
        ctx._pore_order = [k for k in ctx._pore_order if k[0] != sid]
    except AttributeError:
        pass


def _evict_params(ctx, sid: str, method: str) -> None:
    """Evict threshold maps / pore images computed with non-default parameters."""
    try:
        dflt = json.dumps(pthr.default_params(method, ctx.cfg), sort_keys=True)
        for key in [k for k in ctx._thr if k[0] == sid and k[2] == method and k[3] != dflt]:
            ctx._thr.pop(key, None)
        drop = [k for k in ctx._pore if k[0] == sid and k[2] == method and k[4] != "{}"]
        for key in drop:
            ctx._pore.pop(key, None)
        ctx._pore_order = [k for k in ctx._pore_order if k not in set(drop)]
    except AttributeError:
        pass


def _fmt_ci(med, lo, hi, d=2) -> str:
    if not np.isfinite(med):
        return "–"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{med:.{d}f}"
    return f"{med:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _is_failure(phi: float) -> bool:
    return bool(phi < FAIL_LO or phi > FAIL_HI)


# =============================================================================
# stage
# =============================================================================
def run(ctx) -> None:
    sc = ctx.cfg["sensitivity"]
    deltas = np.asarray([float(d) for d in sc["tsm_deltas"]], dtype=np.float64)
    psi_d = float(sc["psi_delta"])
    h = float(sc.get("deriv_h", DERIV_H_DEFAULT))
    roi = ctx.primary_roi
    seg = ctx.seg
    segp = seg[(seg["roi"] == roi) & (seg["post"] == "none")].set_index(["sample_id", "method"])["porosity"]
    ref = ctx.lab_ref.set_index("sample_id")["ref"]
    ref = ref[ref.index.isin(ctx.sample_ids)]

    curve_rows, op_rows, tsm_rows, psi_rows = [], [], [], []
    max_diff = 0.0
    offsets = np.concatenate([[0.0], deltas, [psi_d, -psi_d, h, -h]])
    nd = len(deltas)
    for sid in ctx.sample_ids:
        R = ctx.roi(sid, roi)
        N = ctx.norm(sid, roi)
        ev = R.eval_mask
        vals = np.sort(N.img[ev].astype(np.float64))
        phi_grid = porosity_at_offsets(vals, CURVE_GRID)
        lab, grp = ctx.label(sid), ctx.group(sid)
        for t, p in zip(CURVE_GRID, phi_grid):
            curve_rows.append(dict(sample_id=sid, label=lab, group=grp, t=float(t), phi=float(p)))
        t_lab = lab_matching_threshold(vals, float(ref.get(sid, np.nan)))
        for m in ctx.methods:
            T = ctx.threshold(sid, m, roi)
            res = sorted_residuals(N.img, T, ev)
            ph = porosity_at_offsets(res, offsets)
            phi0 = float(ph[0])
            seg_phi = float(segp.get((sid, m), np.nan))
            if np.isfinite(seg_phi):
                max_diff = max(max_diff, abs(phi0 - seg_phi))
            t_sum = float(T.summary)
            fam = T.family
            for d, p in zip(deltas, ph[1:1 + nd]):
                tsm_rows.append(dict(sample_id=sid, label=lab, group=grp, method=m, family=fam,
                                     delta=float(d), t=t_sum, phi0=phi0, phi_delta=float(p),
                                     abs_change_pp=100.0 * (float(p) - phi0),
                                     rel_change=(float(p) - phi0) / phi0 if phi0 > 0 else np.nan))
            p_plus, p_minus, p_ph, p_mh = ph[1 + nd:]
            slope = float(central_slope(p_ph, p_mh, h))
            psi = float(psi_index(phi0, p_plus, p_minus))
            psi_rows.append(dict(
                sample_id=sid, label=lab, group=grp, method=m, family=fam, phi=phi0, t=t_sum, psi=psi,
                elasticity=float(elasticity(slope, t_sum, phi0)), sens_pp_per_001=float(slope_pp_per_001(slope)),
                phi_plus=float(p_plus), phi_minus=float(p_minus), band_mass=float(abs(p_plus - p_minus)),
                psi_delta=psi_d, deriv_h=h, slope_per_unit=slope,
                hist_density=histogram_density(vals, t_sum) if fam == "global" else np.nan,
                failure=_is_failure(phi0), lab_ref=float(ref.get(sid, np.nan))))
            op_rows.append(dict(sample_id=sid, label=lab, group=grp, method=m, family=fam, t=t_sum, phi=phi0,
                                phi_curve_at_t=float(porosity_at_offsets(vals, t_sum)[0]),
                                on_curve=fam == "global", t_lab=t_lab, lab_ref=float(ref.get(sid, np.nan))))
        ctx.log("%s: thresholds and offsets done", lab)
        _release(ctx, sid)
    if max_diff > 1e-9:
        raise RuntimeError(f"porosity at zero offset differs from segmentation_factorial by {max_diff:.3g}")
    ctx.log("consistency: max |phi(T) - factorial porosity| = %.2e", max_diff)

    curves = pd.DataFrame(curve_rows)
    ops = pd.DataFrame(op_rows)
    tsm = pd.DataFrame(tsm_rows)
    psi = pd.DataFrame(psi_rows)
    ctx.save_data(curves, "threshold_curves")
    ctx.save_data(ops, "threshold_operating_points")
    ctx.save_data(tsm, "tsm")
    contract = ["sample_id", "label", "group", "method", "family", "phi", "t", "psi", "elasticity",
                "sens_pp_per_001"]
    psi = psi[contract + [c for c in psi.columns if c not in contract]]
    ctx.save_data(psi, "psi")

    summ, fr, fr_nf, pw = _psi_statistics(ctx, psi)
    sweep = _local_sweep(ctx, ref)
    homog = _homogeneous(ctx)

    _fig_curves(ctx, curves, ops, ref)
    _fig_tsm(ctx, tsm, deltas)
    _fig_psi(ctx, psi, summ, fr)
    _fig_sweep(ctx, sweep)
    _tables(ctx, psi, summ, fr, fr_nf, homog)
    _keys(ctx, psi, summ, fr, fr_nf, pw, tsm, ops, sweep, homog, max_diff)


# -----------------------------------------------------------------------------
# PSI statistics
# -----------------------------------------------------------------------------
def _psi_statistics(ctx, psi: pd.DataFrame):
    level = float(ctx.cfg["statistics"]["ci_level"])
    rows = []
    for m in ctx.methods:
        g = psi[psi["method"] == m]
        med, lo, hi = bootstrap_median_ci(g["psi"], ctx.n_boot, ctx.rng("sensitivity", "psi_median", m), level)
        olo, ohi, ocov = order_statistic_median_ci(g["psi"], level)
        gnf = g[~g["failure"]]
        mnf, lnf, hnf = bootstrap_median_ci(gnf["psi"], ctx.n_boot, ctx.rng("sensitivity", "psi_median_nf", m),
                                            level)
        rows.append(dict(method=m, family=pthr.METHOD_INFO[m]["family"], n=len(g),
                         psi_median=med, psi_ci_lo=lo, psi_ci_hi=hi,
                         psi_os_lo=olo, psi_os_hi=ohi, psi_os_coverage=ocov,
                         psi_min=float(g["psi"].min()), psi_max=float(g["psi"].max()),
                         abs_elasticity_median=float(np.nanmedian(np.abs(g["elasticity"]))),
                         sens_median=float(np.median(g["sens_pp_per_001"])),
                         n_failures=int(g["failure"].sum()),
                         psi_median_nofail=mnf, psi_nofail_ci_lo=lnf, psi_nofail_ci_hi=hnf, n_nofail=len(gnf)))
    summ = pd.DataFrame(rows)
    wide = psi.pivot(index="sample_id", columns="method", values="psi")[ctx.methods]
    fr = dict(chi2=np.nan, df=np.nan, p=np.nan, W=np.nan, n=len(wide), k=len(ctx.methods), p_perm=np.nan,
              mean_ranks=np.full(len(ctx.methods), np.nan))
    if len(wide) >= 2:
        fr = friedman_kendall(wide.to_numpy())
        fr["p_perm"] = friedman_permutation_p(wide.to_numpy(), ctx.n_boot, ctx.rng("sensitivity", "friedman_perm"))
    summ["mean_rank"] = np.asarray(fr["mean_ranks"], float)
    # sensitivity analysis: only samples where no method failed (complete block design)
    fail_s = set(psi.loc[psi["failure"], "sample_id"])
    wnf = wide.loc[[s for s in wide.index if s not in fail_s]]
    fr_nf = dict(chi2=np.nan, df=np.nan, p=np.nan, W=np.nan, n=len(wnf), k=len(ctx.methods), p_perm=np.nan)
    if len(wnf) >= 2:
        fr_nf = friedman_kendall(wnf.to_numpy())
        fr_nf["p_perm"] = friedman_permutation_p(wnf.to_numpy(), ctx.n_boot,
                                                 ctx.rng("sensitivity", "friedman_perm_nofail"))
    fr_nf["excluded_samples"] = sorted(ctx.label(s) for s in fail_s)
    pw = pairwise_wilcoxon_holm(wide) if len(wide) >= 2 else pd.DataFrame()
    ctx.save_data(summ, "psi_method_summary")
    if len(pw):
        ctx.save_data(pw, "psi_pairwise")
    ctx.log("Friedman PSI: chi2=%.2f df=%s p=%.3g p_perm=%.3g W=%.2f (n=%d)", fr["chi2"], fr["df"], fr["p"],
            fr["p_perm"], fr["W"], fr["n"])
    return summ, fr, fr_nf, pw


# -----------------------------------------------------------------------------
# local-parameter sweep
# -----------------------------------------------------------------------------
def _local_sweep(ctx, ref: pd.Series) -> pd.DataFrame:
    sc = ctx.cfg["sensitivity"]
    tcfg = ctx.cfg["thresholds"]
    main = {"Sauvola": (int(tcfg["sauvola"]["window"]), float(tcfg["sauvola"]["k"])),
            "Niblack": (int(tcfg["niblack"]["window"]), float(tcfg["niblack"]["k"]))}
    grids = {"Sauvola": [float(k) for k in sc["sweep_sauvola_k"]],
             "Niblack": [float(k) for k in sc["sweep_niblack_k"]]}
    rows = []
    roi = ctx.primary_roi
    for sid in ctx.sample_ids:
        R = ctx.roi(sid, roi)
        rv = float(ref.get(sid, np.nan))
        for m, ks in grids.items():
            for w in sc["sweep_windows"]:
                for k in ks:
                    params = {"window": int(w), "k": float(k)}
                    pore = ctx.pore(sid, m, "none", roi, params=params)
                    phi = _porosity(pore, R.eval_mask)
                    tmean = float(ctx.threshold(sid, m, roi, params=params).summary)
                    _evict_params(ctx, sid, m)
                    err = 100.0 * (phi - rv) if np.isfinite(rv) else np.nan
                    rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), method=m,
                                     window=int(w), k=float(k), porosity=phi, t_mean=tmean,
                                     is_main=(int(w), float(k)) == main[m], lab_ref=rv, error_pp=err,
                                     abs_error_pp=abs(err) if np.isfinite(err) else np.nan))
        ctx.log("%s: local sweep done", ctx.label(sid))
        _release(ctx, sid)
    sw = pd.DataFrame(rows)
    # sanity check: the main configuration reproduces the factorial porosity
    seg = ctx.seg
    segp = seg[(seg["roi"] == roi) & (seg["post"] == "none")].set_index(["sample_id", "method"])["porosity"]
    mm = sw[sw["is_main"]]
    d = np.abs(mm["porosity"].to_numpy() - np.array([segp.get((s, m), np.nan) for s, m in
                                                     zip(mm["sample_id"], mm["method"])]))
    if len(d) and np.nanmax(d) > 1e-9:
        raise RuntimeError(f"local sweep main configuration differs from factorial by {np.nanmax(d):.3g}")
    ctx.save_data(sw, "local_sweep")
    return sw


# -----------------------------------------------------------------------------
# homogeneous fields
# -----------------------------------------------------------------------------
HOMOG_FIELDS = [
    # name, label, mean, sd, gradient, true porosity
    ("matrix_sd001", "Matrix, σ = 0.01", 0.7, 0.01, None, 0.0),
    ("matrix_sd003", "Matrix, σ = 0.03", 0.7, 0.03, None, 0.0),
    ("matrix_sd006", "Matrix, σ = 0.06", 0.7, 0.06, None, 0.0),
    ("matrix_gradient", "Matrix + gradient", 0.7, 0.03, (0.6, 0.8), 0.0),
    ("pore_sd003", "Pore region, σ = 0.03", 0.2, 0.03, None, 1.0),
]


def _homogeneous(ctx) -> pd.DataFrame:
    size = int(ctx.cfg.get("synthetic", {}).get("size_px", 512))
    n_rep = 1 if ctx.quick else 3
    tc = ctx.cfg["thresholds"]
    nc = ctx.cfg["normalization"]
    mu_n, sd_n = renormalised_moments(float(nc["p_low"]), float(nc["p_high"]))
    rows = []
    for name, flabel, mu, sd, grad, truth in HOMOG_FIELDS:
        for rep in range(n_rep):
            img = homogeneous_field(size, mu, sd, ctx.rng("sensitivity", "homog", name, rep), grad)
            conds = [("as_is", False)]
            if name == "matrix_sd003":
                conds.append(("roi_renormalised", True))
            for cond, ren in conds:
                for m in ctx.methods:
                    phi, t = field_porosity(img, m, ctx.cfg, renormalise=ren)
                    if m == "Niblack":
                        an = niblack_false_porosity_analytic(float(tc["niblack"]["k"]))
                    elif m == "Sauvola" and grad is None:
                        mu_e, sd_e = (mu_n, sd_n) if ren else (mu, sd)
                        an = sauvola_false_porosity_analytic(mu_e, sd_e, float(tc["sauvola"]["k"]),
                                                             float(tc["sauvola"]["r"]))
                    else:
                        an = np.nan
                    rows.append(dict(field=name, field_label=flabel, condition=cond, mean=mu, sd=sd,
                                     gradient=str(grad) if grad else "", true_porosity=truth, rep=rep,
                                     method=m, family=pthr.METHOD_INFO[m]["family"], porosity=phi,
                                     threshold=t, analytic=an, size_px=size))
    hg = pd.DataFrame(rows)
    ctx.save_data(hg, "homogeneous_false_porosity")
    return hg


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------
def _grid(n: int, ncol: int = 3):
    return int(math.ceil(n / ncol)), min(ncol, n)


_GLOBAL_SIZES = {"Otsu": 7.0, "Li": 6.2, "Yen": 5.6, "Mean": 5.0, "Triangle": 4.4, "Isodata": 3.6}


def _fig_curves(ctx, curves: pd.DataFrame, ops: pd.DataFrame, ref: pd.Series) -> None:
    sids = ctx.sample_ids
    nrow, ncol = _grid(len(sids))
    fig, axs = plt.subplots(nrow, ncol, figsize=style.figsize("double", 44 * nrow + 14), sharex=True,
                            sharey=True, layout="constrained")
    axs = np.atleast_1d(axs).ravel()
    for i, (ax, sid) in enumerate(zip(axs, sids)):
        c = curves[curves["sample_id"] == sid]
        ax.plot(c["t"], c["phi"], color=style.INK, lw=1.0, zorder=2)
        if sid in ref.index and np.isfinite(ref[sid]):
            ax.axhline(ref[sid], color=style.MUTED, lw=0.8, ls=(0, (4, 2)), zorder=1)
            ax.text(0.02, ref[sid], f"Lab {100 * ref[sid]:.1f} %", color=style.INK2, va="bottom", ha="left",
                    fontsize=6.5)
        o = ops[ops["sample_id"] == sid].set_index("method")
        for m in [x for x in style.METHOD_ORDER if x in o.index and pthr.METHOD_INFO[x]["family"] == "global"]:
            ax.plot(o.at[m, "t"], o.at[m, "phi"], ls="none", marker=style.METHOD_MARKERS[m],
                    ms=_GLOBAL_SIZES.get(m, 4.5), mfc=style.METHOD_COLORS[m], mec="white", mew=0.4, zorder=3)
        for m in [x for x in style.METHOD_ORDER if x in o.index and pthr.METHOD_INFO[x]["family"] == "local"]:
            ax.plot(o.at[m, "t"], o.at[m, "phi"], ls="none", marker=style.METHOD_MARKERS[m], ms=4.6,
                    mfc="white", mec=style.METHOD_COLORS[m], mew=1.0, zorder=4)
        ax.set_title(f"{ctx.label(sid)}", pad=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style.pct_axis(ax)
        style.hgrid(ax, "both")
        style.panel_label(ax, "abcdefghijklmnop"[i])
        if i % ncol == 0:
            ax.set_ylabel("Porosity φ(t)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("Normalised grey level t")
    for ax in axs[len(sids):]:
        ax.axis("off")
    hs = [Line2D([], [], color=style.INK, lw=1.0, label="φ(t) = ECDF of ROI grey levels")]
    for m in style.METHOD_ORDER:
        if pthr.METHOD_INFO[m]["family"] == "global":
            hs.append(Line2D([], [], ls="none", marker="o", ms=4.5, mfc=style.METHOD_COLORS[m], mec="white",
                             mew=0.4, label=m))
        else:
            hs.append(Line2D([], [], ls="none", marker="s", ms=4.6, mfc="white", mec=style.METHOD_COLORS[m],
                             mew=1.0, label=f"{m} (mean T, off-curve)"))
    hs.append(Line2D([], [], color=style.MUTED, lw=0.8, ls=(0, (4, 2)), label="Laboratory reference"))
    fig.legend(handles=hs, loc="outside lower center", ncol=6)
    n_lab = int(sum(1 for s in sids if s in ref.index))
    ctx.save_fig(fig, "threshold_porosity_curves",
                 "Porosity-threshold curves. For each image (n = 9; primary ROI = largest usable disc, metrics on "
                 "the ROI minus a 2 px rim, no post-processing) the black curve is φ(t), the fraction of ROI pixels "
                 "with normalised grey level ≤ t (the empirical cumulative distribution of the ROI grey levels; grey "
                 "levels normalised to [0, 1] by the 0.1-99.9 percentiles of the ROI). Filled circles: operating "
                 "points (T, φ(T)) of the six global methods, which by construction lie on the curve (marker size "
                 "decreases from Otsu to Isodata so that coincident thresholds, e.g. Otsu and Isodata, remain "
                 "visible as concentric rings). Open squares: local adaptive methods (Sauvola, Niblack) plotted at "
                 "the mean of their threshold map and at their actual porosity; the curve does not apply to them "
                 "because each pixel has its own threshold. Dashed line: laboratory reference porosity "
                 f"(n = {n_lab} images with a laboratory value). The slope of the curve at T equals the grey-level "
                 "histogram density at T, i.e. the sensitivity of porosity to the threshold.", kind="supp")


def _fig_tsm(ctx, tsm: pd.DataFrame, deltas: np.ndarray) -> None:
    sids = ctx.sample_ids
    nrow, ncol = _grid(len(sids))
    methods = ctx.methods
    vmax = float(np.nanmax(np.abs(tsm["abs_change_pp"]))) if len(tsm) else 1.0
    lim = max(5.0, 10.0 * math.ceil(vmax / 10.0))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, axs = plt.subplots(nrow, ncol, figsize=style.figsize("double", 40 * nrow + 16), sharex=True,
                            sharey=True, layout="constrained")
    axs = np.atleast_1d(axs).ravel()
    im = None
    for i, (ax, sid) in enumerate(zip(axs, sids)):
        d = tsm[tsm["sample_id"] == sid].pivot(index="method", columns="delta", values="abs_change_pp")
        d = d.reindex(index=methods, columns=deltas)
        im = ax.imshow(d.to_numpy(), cmap=style.CMAP_DIV, norm=norm, aspect="auto", interpolation="nearest")
        n_glob = sum(pthr.METHOD_INFO[m]["family"] == "global" for m in methods)
        ax.axhline(n_glob - 0.5, color="white", lw=1.2)
        j0 = int(np.argmin(np.abs(deltas)))
        ax.axvline(j0 - 0.5, color="white", lw=0.5)
        ax.axvline(j0 + 0.5, color="white", lw=0.5)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels([f"{x:+.2f}".replace("+0.00", "0").replace("-", "−") for x in deltas], rotation=90)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ctx.label(sid), pad=2)
        style.panel_label(ax, "abcdefghijklmnop"[i], x=-0.03)
        if i // ncol == nrow - 1 or i + ncol >= len(sids):
            ax.set_xlabel("Threshold offset δ (normalised grey level)")
    for ax in axs[len(sids):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs[:len(sids)].tolist(), shrink=0.6, aspect=30, pad=0.01, extend="neither")
    cb.set_label("Change in porosity φ(T+δ) − φ(T) (pp)")
    cb.outline.set_linewidth(0.4)
    ctx.save_fig(fig, "threshold_sensitivity_matrix",
                 "Threshold Sensitivity Matrix (TSM). Each panel is one image (n = 9; primary ROI = largest usable "
                 "disc, metrics on the ROI minus a 2 px rim, no post-processing); rows are the eight thresholding "
                 "methods (global above the white line, local adaptive below), columns are additive offsets δ of the "
                 "threshold in normalised grey-level units (fractions of the robust 0.1-99.9 percentile dynamic "
                 "range of the ROI; for Sauvola and Niblack δ is added to the whole threshold map). Colour: change "
                 "in porosity φ(T+δ) − φ(T) in percentage points (diverging scale centred at 0, identical for all "
                 f"panels, ±{lim:.0f} pp; blue = less pore, red = more pore). The δ = 0 column (between thin white "
                 "lines) is zero by definition. Relative changes are given in the data file tsm.csv.", kind="main")


def _fig_psi(ctx, psi: pd.DataFrame, summ: pd.DataFrame, fr: dict) -> None:
    methods = ctx.methods
    sids = ctx.sample_ids
    offs = np.linspace(-0.27, 0.27, len(sids)) if len(sids) > 1 else np.zeros(1)
    off = dict(zip(sids, offs))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=style.figsize("double", 68),
                                 gridspec_kw=dict(width_ratios=[1.55, 1.0]), layout="constrained")
    for j, m in enumerate(methods):
        g = psi[psi["method"] == m]
        for _, r in g.iterrows():
            filled = r["group"] == "granular_pack"
            ax.plot(j + off[r["sample_id"]], r["psi"], ls="none", marker=style.METHOD_MARKERS[m], ms=3.6,
                    mfc=style.METHOD_COLORS[m] if filled else "white", mec=style.METHOD_COLORS[m], mew=0.8,
                    zorder=3)
        s = summ[summ["method"] == m].iloc[0]
        if np.isfinite(s["psi_ci_lo"]):
            ax.plot([j + 0.36, j + 0.36], [s["psi_ci_lo"], s["psi_ci_hi"]], color=style.INK, lw=1.0, zorder=4)
        ax.plot([j + 0.30, j + 0.42], [s["psi_median"]] * 2, color=style.INK, lw=1.6, zorder=4,
                solid_capstyle="butt")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_xlim(-0.5, len(methods) - 0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(f"PSI (δ = ±{psi['psi_delta'].iloc[0]:.2f})")
    style.hgrid(ax)
    style.panel_label(ax, "a")
    # (b) PSI vs absolute error against the laboratory reference
    lab = psi[np.isfinite(psi["lab_ref"])].copy()
    lab["abs_err_pp"] = 100.0 * np.abs(lab["phi"] - lab["lab_ref"])
    for m in methods:
        g = lab[lab["method"] == m]
        for _, r in g.iterrows():
            filled = r["group"] == "granular_pack"
            bx.plot(r["psi"], r["abs_err_pp"], ls="none", marker=style.METHOD_MARKERS[m], ms=3.6,
                    mfc=style.METHOD_COLORS[m] if filled else "white", mec=style.METHOD_COLORS[m], mew=0.8,
                    zorder=3)
    bx.set_xlim(-0.03, 1.03)
    bx.set_ylim(bottom=0)
    bx.set_xlabel("PSI")
    bx.set_ylabel("|Image − laboratory porosity| (pp)")
    style.hgrid(bx, "both")
    style.panel_label(bx, "b")
    hs = style.method_handles(methods, ms=3.6)
    hs += [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=style.MUTED, mec=style.MUTED, label="Granular pack"),
           Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=style.MUTED, mew=0.8,
                  label="Consolidated"),
           Line2D([], [], color=style.INK, lw=1.6, label="Median"),
           Line2D([], [], color=style.INK, lw=1.0, label="95 % bootstrap CI")]
    fig.legend(handles=hs, loc="outside lower center", ncol=7)
    n_lab = int(lab["sample_id"].nunique())
    ctx.save_fig(fig, "psi_by_method",
                 "Pore Stability Index (PSI) of the eight thresholding methods. PSI = 1 − min(1, |φ(T+δ) − "
                 "φ(T−δ)| / (2φ(T))) with δ = 0.05 of the normalised grey-level range; PSI is dimensionless, 1 = "
                 "porosity insensitive to a ±δ threshold shift, 0 = the shift changes porosity by ≥ 100 % of its "
                 "value. Primary ROI (largest usable disc), no post-processing. (a) One point per image (n = 9 per "
                 "method; filled = granular packs, n = 4; open = consolidated cores, n = 5; horizontal offsets are "
                 "fixed per image); thick black bar = median over images, thin vertical line = 95 % percentile "
                 f"bootstrap CI of the median ({ctx.n_boot} resamples of images). Friedman test across methods "
                 f"(subject = image): χ²({fr['df']}) = {fr['chi2']:.1f}, permutation p = {fr['p_perm']:.3g}, "
                 f"Kendall's W = {fr['W']:.2f}. (b) PSI versus the absolute difference between image and "
                 f"laboratory porosity for the {n_lab} images with a laboratory value (one point per image and "
                 "method; points are not independent). A high PSI does not imply accuracy: PSI measures only how "
                 "much porosity moves when the threshold moves; a method can be insensitive and systematically "
                 "wrong (e.g. Niblack returns a fixed quantile, ≈Φ(−k), of the local noise, and global methods "
                 "placed in a histogram valley of a granular pack are stable but miss the unresolved porosity).",
                 kind="main")


def _fig_sweep(ctx, sw: pd.DataFrame) -> None:
    methods = ["Sauvola", "Niblack"]
    has_err = bool(np.isfinite(sw["abs_error_pp"]).any())
    ncol = 2 if has_err else 1
    fig, axs = plt.subplots(2, ncol, figsize=style.figsize("onehalf", 92), squeeze=False, layout="constrained")
    agg = (sw.groupby(["method", "window", "k"])
             .agg(phi_med=("porosity", "median"), err_med=("abs_error_pp", lambda x: np.nanmedian(x)
                                                           if np.isfinite(x).any() else np.nan),
                  is_main=("is_main", "first"))
             .reset_index())
    pmax = 100.0 * float(agg["phi_med"].max())
    norm_p = Normalize(0, 10.0 * math.ceil(pmax / 10.0))
    emax = float(np.nanmax(agg["err_med"])) if has_err else 1.0
    norm_e = Normalize(0, 10.0 * math.ceil(emax / 10.0))
    ims = {}
    for i, m in enumerate(methods):
        a = agg[agg["method"] == m]
        wins = sorted(a["window"].unique())
        ks = sorted(a["k"].unique())
        for j, (col, nrm, lab) in enumerate([("phi_med", norm_p, "Median porosity (%)"),
                                             ("err_med", norm_e, "Median |error| vs lab (pp)")][:ncol]):
            ax = axs[i, j]
            M = a.pivot(index="k", columns="window", values=col).reindex(index=ks, columns=wins).to_numpy()
            if col == "phi_med":
                M = 100.0 * M
            ims[col] = ax.imshow(M, cmap=style.CMAP_SEQ, norm=nrm, aspect="auto", origin="lower")
            for yi in range(M.shape[0]):
                for xi in range(M.shape[1]):
                    v = M[yi, xi]
                    if np.isfinite(v):
                        ax.text(xi, yi, f"{v:.1f}", ha="center", va="center", fontsize=6.3,
                                color="white" if nrm(v) > 0.55 else style.INK)
            mrow = a[a["is_main"]]
            if len(mrow):
                xi = wins.index(int(mrow["window"].iloc[0]))
                yi = ks.index(float(mrow["k"].iloc[0]))
                ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, ec=style.PALETTE[7], lw=1.4,
                                           zorder=5))
            ax.set_xticks(range(len(wins)))
            ax.set_xticklabels([str(w) for w in wins])
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels([f"{k:g}" for k in ks])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel("Window w (px)")
            ax.set_ylabel(f"{m}  k")
            ax.set_title(lab, pad=2)
            style.panel_label(ax, "abcd"[i * ncol + j], x=-0.08)
    for j, (col, lab) in enumerate([("phi_med", "Median porosity (%)"),
                                    ("err_med", "Median |error| vs lab (pp)")][:ncol]):
        cb = fig.colorbar(ims[col], ax=axs[:, j].tolist(), location="bottom", shrink=0.8, aspect=30, pad=0.02)
        cb.set_label(lab)
        cb.outline.set_linewidth(0.4)
    n_all = sw["sample_id"].nunique()
    n_lab = sw.loc[np.isfinite(sw["abs_error_pp"]), "sample_id"].nunique()
    ctx.save_fig(fig, "local_parameter_sweep",
                 "Sensitivity of the local adaptive methods to their parameters. Rows: Sauvola (T = m[1 + k(s/r − "
                 "1)], r = 0.5) and Niblack (T = m − k s), with local mean m and s.d. s in a w × w window; "
                 f"columns: median porosity over the n = {n_all} images and median absolute difference from the "
                 f"laboratory reference over the n = {n_lab} images with a laboratory value (percentage points). "
                 "Primary ROI (largest usable disc), no post-processing. Red outline: configuration used in the "
                 "main analysis (w = 25 px; k = 0.2 for Sauvola, 0.8 for Niblack). Window sizes in pixels (25 µm "
                 "or 40 µm per pixel, see the dataset table). Per-image values are in local_sweep.csv.",
                 kind="supp")


# -----------------------------------------------------------------------------
# tables
# -----------------------------------------------------------------------------
def _tables(ctx, psi, summ, fr, fr_nf, homog) -> None:
    rows = []
    for _, s in summ.iterrows():
        rows.append({"Method": s["method"], "Family": s["family"].capitalize(),
                     "PSI median [95 % CI]": _fmt_ci(s["psi_median"], s["psi_ci_lo"], s["psi_ci_hi"]),
                     "PSI range": f"{s['psi_min']:.2f}–{s['psi_max']:.2f}",
                     "Mean rank": float(s["mean_rank"]),
                     "Median |E|": float(s["abs_elasticity_median"]),
                     "Median S (pp per 0.01)": float(s["sens_median"]),
                     "Failures": int(s["n_failures"]),
                     "PSI median, no failures [95 % CI]": _fmt_ci(s["psi_median_nofail"], s["psi_nofail_ci_lo"],
                                                                  s["psi_nofail_ci_hi"]) + f" (n = {s['n_nofail']})"})
    tab = pd.DataFrame(rows)
    excl = ", ".join(fr_nf.get("excluded_samples", [])) or "none"
    ctx.save_table(
        tab, "psi_summary",
        "Threshold sensitivity of the eight methods (primary ROI = largest usable disc, no post-processing, "
        f"n = {fr['n']} images per method). PSI = 1 − min(1, |φ(T+δ) − φ(T−δ)| / (2φ(T))), δ = 0.05 of the "
        "normalised grey-level range (dimensionless, 1 = insensitive); median over images with 95 % percentile "
        f"bootstrap CI ({ctx.n_boot} resamples of images). Mean rank: mean within-image rank of PSI (1 = least "
        f"stable). |E|: absolute elasticity (dφ/dt)(T/φ); S: absolute sensitivity dφ/dt in percentage points per "
        f"0.01 of normalised grey level; both from central differences with h = {psi['deriv_h'].iloc[0]:.2f}. "
        f"Failures: images with φ < {100 * FAIL_LO:.0f} % or > {100 * FAIL_HI:.0f} %; last column repeats the "
        "median without them. High PSI does not imply accuracy.",
        notes=(f"Friedman test across methods (subject = image, n = {fr['n']}, k = {fr['k']}): χ²({fr['df']}) = "
               f"{fr['chi2']:.2f}, asymptotic p = {fr['p']:.3g}, permutation p = {fr['p_perm']:.3g} "
               f"({ctx.n_boot} within-image permutations), Kendall's W = {fr['W']:.2f}. Without images containing "
               f"a failure ({excl}; n = {fr_nf['n']}): χ² = {fr_nf['chi2']:.2f}, permutation p = "
               f"{fr_nf['p_perm']:.3g}, W = {fr_nf['W']:.2f}. Pairwise Wilcoxon tests cannot reach significance "
               f"after Holm adjustment over 28 pairs with n = {fr['n']} (smallest attainable exact p = "
               f"{min_exact_wilcoxon_p(int(fr['n'])):.4f}); see psi_pairwise.csv."))

    # homogeneous-region false porosity
    hg = homog.groupby(["field", "condition", "method"]).agg(phi=("porosity", "mean"), sd=("porosity", "std"),
                                                             analytic=("analytic", "first")).reset_index()
    cols = [("matrix_sd001", "as_is", "Matrix σ=0.01"), ("matrix_sd003", "as_is", "Matrix σ=0.03"),
            ("matrix_sd006", "as_is", "Matrix σ=0.06"), ("matrix_gradient", "as_is", "Matrix + gradient"),
            ("matrix_sd003", "roi_renormalised", "Matrix σ=0.03, ROI-renormalised"),
            ("pore_sd003", "as_is", "Pore region σ=0.03 (true 100 %)")]
    trows = []
    for m in ctx.methods:
        d = {"Method": m, "Family": pthr.METHOD_INFO[m]["family"].capitalize()}
        for f, c, lab in cols:
            v = hg[(hg["field"] == f) & (hg["condition"] == c) & (hg["method"] == m)]
            d[lab + " (%)"] = 100.0 * float(v["phi"].iloc[0]) if len(v) else np.nan
        v = hg[(hg["field"] == "matrix_sd003") & (hg["condition"] == "as_is") & (hg["method"] == m)]
        d["Analytic, matrix σ=0.03 (%)"] = 100.0 * float(v["analytic"].iloc[0]) if len(v) else np.nan
        trows.append(d)
    tc = ctx.cfg["thresholds"]
    n_rep = int(homog["rep"].nunique())
    size = int(homog["size_px"].iloc[0])
    ctx.save_table(
        pd.DataFrame(trows), "homogeneous_false_porosity",
        "Porosity returned by each method on synthetic homogeneous fields (true porosity 0 % for the matrix "
        f"fields, 100 % for the pore region). Fields: {size} × {size} px Gaussian noise in normalised grey-level "
        "units, mean 0.7 (matrix) or 0.2 (pore region), SD σ as indicated; 'gradient': mean increasing linearly "
        "from 0.6 to 0.8 across the field, σ = 0.03; 'ROI-renormalised': the σ = 0.03 matrix field rescaled to "
        "[0, 1] by its 0.1-99.9 percentiles, as the pipeline does when an entire ROI is homogeneous. Values are "
        f"means over {n_rep} independent noise realisations (evaluation without a 2 px rim). Analytic values for "
        f"a Gaussian field: Niblack Φ(−k) = {100 * niblack_false_porosity_analytic(float(tc['niblack']['k'])):.1f} % "
        f"(k = {tc['niblack']['k']}), independent of mean and σ; Sauvola Φ(kμ(1/r − 1/σ)) "
        f"(k = {tc['sauvola']['k']}, r = {tc['sauvola']['r']}).",
        kind="supp", float_format="{:.1f}",
        notes="Global methods have no analytic value (they cut a unimodal histogram at a method-specific quantile). "
              "SDs over realisations are in homogeneous_false_porosity.csv.")


# -----------------------------------------------------------------------------
# key numbers
# -----------------------------------------------------------------------------
def _keys(ctx, psi, summ, fr, fr_nf, pw, tsm, ops, sweep, homog, max_diff) -> None:
    K = ctx.add_key
    n = int(psi["sample_id"].nunique())
    dlt = float(psi["psi_delta"].iloc[0])
    K("sensitivity_consistency_max_abs_diff", float(max_diff),
      "max |φ(T) recomputed from the residual ECDF − porosity in segmentation_factorial| over all samples and "
      "methods (primary ROI, post none); must be 0", "fraction")
    for _, s in summ.iterrows():
        m = s["method"]
        K(f"sensitivity_psi_median_{m.lower()}", [s["psi_median"], s["psi_ci_lo"], s["psi_ci_hi"]],
          f"PSI (δ = ±{dlt}) of {m}: median over n = {int(s['n'])} images [95 % percentile bootstrap CI over "
          f"images, {ctx.n_boot} resamples]; primary ROI, post none; dimensionless, 1 = insensitive")
        K(f"sensitivity_psi_range_{m.lower()}", [s["psi_min"], s["psi_max"]],
          f"min and max PSI of {m} over the n = {int(s['n'])} images (primary ROI, post none)")
        K(f"sensitivity_abs_elasticity_median_{m.lower()}", s["abs_elasticity_median"],
          f"median over images of |E| = |dφ/dt · T/φ| for {m} (central difference h = "
          f"{psi['deriv_h'].iloc[0]}); primary ROI, post none; dimensionless")
        K(f"sensitivity_sens_median_{m.lower()}", s["sens_median"],
          f"median over images of S = dφ/dt for {m}, in percentage points of porosity per 0.01 of normalised "
          "grey level; primary ROI, post none", "pp per 0.01")
    K("sensitivity_friedman_psi", dict(chi2=fr["chi2"], df=fr["df"], p_asymptotic=fr["p"], p_permutation=fr["p_perm"],
                                       kendall_W=fr["W"], n=fr["n"], k=fr["k"]),
      "Friedman test of PSI across the 8 methods (subject = image), tie-corrected χ², asymptotic and "
      f"within-image permutation p ({ctx.n_boot} permutations), Kendall's W effect size; primary ROI, post none")
    K("sensitivity_friedman_psi_nofail", dict(chi2=fr_nf["chi2"], p_permutation=fr_nf["p_perm"], kendall_W=fr_nf["W"],
                                              n=fr_nf["n"], excluded=", ".join(fr_nf.get("excluded_samples", []))),
      f"Friedman test of PSI restricted to images where no method failed (φ < {100 * FAIL_LO:.0f} % or > "
      f"{100 * FAIL_HI:.0f} %)")
    if len(pw):
        K("sensitivity_pairwise_min_p_holm", float(pw["p_holm"].min()),
          f"smallest Holm-adjusted p among the 28 pairwise Wilcoxon tests of PSI (n = {n} images)")
        K("sensitivity_pairwise_min_attainable_p", min_exact_wilcoxon_p(n),
          f"smallest attainable exact two-sided Wilcoxon p with n = {n} pairs (= 2/2^n); compare with 0.05/28 "
          "for Holm over 28 pairs: pairwise comparisons are underpowered by design")
    ranks = summ.set_index("method")["mean_rank"]
    K("sensitivity_psi_mean_rank", {m: float(v) for m, v in ranks.items()},
      "mean within-image rank of PSI per method (1 = least stable, 8 = most stable), n images")
    # group contrast: mean PSI of the global methods per image
    glob = psi[psi["family"] == "global"].groupby(["sample_id", "group"])["psi"].mean().reset_index()
    gp = glob[glob["group"] == "granular_pack"]["psi"].to_numpy()
    gc = glob[glob["group"] == "consolidated"]["psi"].to_numpy()
    if len(gp) and len(gc):
        mw = stats.mannwhitneyu(gp, gc, alternative="two-sided", method="exact")
        diffs = (gp[:, None] - gc[None, :]).ravel()
        K("sensitivity_psi_global_by_group", dict(granular_median=float(np.median(gp)), n_granular=len(gp),
                                                  consolidated_median=float(np.median(gc)), n_consolidated=len(gc),
                                                  hl_shift=float(np.median(diffs)), p_exact_mannwhitney=float(mw.pvalue)),
          "per-image mean PSI over the 6 global methods: median by group, Hodges-Lehmann shift (granular − "
          "consolidated) and exact two-sided Mann-Whitney p (exploratory, single planned contrast: bimodal vs "
          "near-unimodal histograms)")
    # derivative = histogram density check (global methods)
    g = psi[(psi["family"] == "global") & np.isfinite(psi["hist_density"])]
    if len(g) > 2:
        rho = stats.spearmanr(g["slope_per_unit"], g["hist_density"]).statistic
        K("sensitivity_slope_vs_hist_density_spearman", float(rho),
          "Spearman correlation between the central-difference slope dφ/dt at T and the 256-bin histogram "
          f"density at T over all global (sample, method) pairs (n = {len(g)}); checks eq. (4)")
    # TSM: typical change for small offsets
    for d in (0.02, 0.04, 0.08):
        for fam in ("global", "local"):
            sel = tsm[(tsm["family"] == fam) & np.isclose(np.abs(tsm["delta"]), d)]
            if len(sel):
                K(f"sensitivity_tsm_median_abs_change_{fam}_d{int(round(100 * d)):02d}",
                  float(np.median(np.abs(sel["abs_change_pp"]))),
                  f"median over images, {fam} methods and δ = ±{d} of |φ(T+δ) − φ(T)| (percentage points); "
                  f"n = {len(sel)} (image, method, sign) cells; primary ROI, post none", "pp")
    K("sensitivity_tsm_max_abs_change_pp", float(np.max(np.abs(tsm["abs_change_pp"]))),
      f"largest |φ(T+δ) − φ(T)| over all images, methods and δ in [{tsm['delta'].min()}, {tsm['delta'].max()}]",
      "pp")
    # lab-matching threshold vs Otsu
    o = ops[(ops["method"] == "Otsu") & np.isfinite(ops["t_lab"])]
    if len(o):
        shift = (o["t_lab"] - o["t"]).to_numpy()
        K("sensitivity_lab_matching_threshold_shift_vs_otsu",
          {r["label"]: float(r["t_lab"] - r["t"]) for _, r in o.iterrows()},
          "global threshold offset (normalised grey level) that would make φ(t) equal the laboratory porosity, "
          "relative to the Otsu threshold, per image with a laboratory value (positive = threshold must be raised)")
        gp_ = o[o["group"] == "granular_pack"]
        if len(gp_):
            K("sensitivity_lab_matching_shift_granular_median", float(np.median(gp_["t_lab"] - gp_["t"])),
              f"median over the n = {len(gp_)} granular packs of the offset (normalised grey level) from the Otsu "
              "threshold to the threshold reproducing the laboratory porosity", "normalised grey level")
        _ = shift
    # local sweep
    if len(sweep):
        rng_ = sweep.groupby(["method", "sample_id"])["porosity"].agg(lambda x: 100 * (x.max() - x.min()))
        for m in ("Sauvola", "Niblack"):
            if m in rng_.index.get_level_values(0):
                r = rng_.loc[m]
                K(f"sensitivity_sweep_porosity_range_{m.lower()}",
                  dict(median=float(r.median()), min=float(r.min()), max=float(r.max()), n=int(len(r))),
                  f"range (max − min, pp) of {m} porosity across the window × k grid, per image: median, min and "
                  "max over images (primary ROI, post none)", "pp")
                s = sweep[(sweep["method"] == m)]
                e = s.groupby(["window", "k"])["abs_error_pp"].median()
                if np.isfinite(e).any():
                    best = e.idxmin()
                    main_e = float(s[s["is_main"]]["abs_error_pp"].median())
                    K(f"sensitivity_sweep_best_{m.lower()}",
                      dict(window=int(best[0]), k=float(best[1]), median_abs_error_pp=float(e.min()),
                           main_config_median_abs_error_pp=main_e),
                      f"{m}: configuration of the sweep with the smallest median |error| vs lab over the "
                      f"n = {int(s.loc[np.isfinite(s['abs_error_pp']), 'sample_id'].nunique())} images with a "
                      "laboratory value, compared with the main-analysis configuration (in-sample choice, optimistic)",
                      "pp")
    # homogeneous fields
    hg = homog.groupby(["field", "condition", "method"])["porosity"].mean()
    tc = ctx.cfg["thresholds"]
    K("sensitivity_niblack_analytic_false_porosity", niblack_false_porosity_analytic(float(tc["niblack"]["k"])),
      f"analytic Niblack false porosity Φ(−k) of any homogeneous Gaussian region, k = {tc['niblack']['k']}",
      "fraction")
    for (f, c, m), v in hg.items():
        if m in ("Niblack", "Sauvola", "Otsu", "Li", "Mean", "Yen", "Triangle", "Isodata"):
            K(f"sensitivity_homog_{f}_{c}_{m.lower()}", float(v),
              f"porosity returned by {m} on the synthetic homogeneous field '{f}' ({c}); true porosity "
              f"{'1' if f.startswith('pore') else '0'}; mean over {int(homog['rep'].nunique())} realisations",
              "fraction")
