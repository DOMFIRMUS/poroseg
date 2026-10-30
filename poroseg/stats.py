"""Statistical engine of the poroseg pipeline.

This module contains every inferential procedure used by the analysis stages
(chiefly ``inference`` and ``ml_baseline``).  All functions are *pure*: they
take arrays / tidy ``DataFrame`` objects and return plain ``dict`` objects or
tidy ``DataFrame`` objects, never write files and never touch global random
state (every resampling routine takes an explicit :class:`numpy.random.Generator`;
inside the pipeline always pass ``ctx.rng("purpose", ...)``).

Design principles
-----------------
* **The experimental unit is the image (sample).**  Pixels and tiles are
  spatially correlated and are never treated as replicates here.  Repeated
  measures designs (the same image segmented by several methods /
  post-processing levels) are analysed with the image as *subject*; resampling
  procedures resample whole images (cluster bootstrap, Field & Welsh 2007).
* **Effect sizes with confidence intervals first, p-values second.**  With
  n = 9 images (7 with a laboratory reference) exact, permutation and
  rank-based procedures are preferred and every result reports the n it is
  based on.
* **Explicit handling of degenerate data.**  Ties, zero differences, constant
  columns and incomplete subjects are reported (``n_dropped``, ``p_method``,
  ``note`` fields) and never silently discarded.

Sections
--------
1. Utilities: :func:`p_adjust`, :func:`fmt_p`, :func:`fmt_ci`, :func:`logit`,
   :func:`inv_logit`, :func:`arcsine_sqrt`, :func:`to_wide`.
2. Resampling: :func:`bootstrap_ci`, :func:`cluster_bootstrap`,
   :func:`permutation_test_paired`.
3. Repeated measures (subject = image): :func:`kendall_w`, :func:`friedman`,
   :func:`wilcoxon_signed_rank`, :func:`hodges_lehmann`,
   :func:`pairwise_paired`, :func:`sphericity`, :func:`rm_anova_oneway`,
   :func:`rm_anova_twoway`.
4. Factorial variance decomposition: :func:`variance_decomposition`.
5. Agreement / method comparison (x = reference, y = test):
   :func:`bland_altman`, :func:`lin_ccc`, :func:`deming`,
   :func:`passing_bablok`, :func:`error_metrics`, :func:`prob_best`,
   :func:`icc`.

Validation
----------
``tests/test_stats.py`` cross-checks the procedures against independent
implementations (pingouin 0.6, scipy 1.18, closed-form solutions, brute-force
test inversion and small coverage simulations).

References
----------
Bakeman R (2005) Recommended effect size statistics for repeated measures
    designs. Behav Res Methods 37:379-384.
Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate. J R Stat
    Soc B 57:289-300.
Bland JM, Altman DG (1986) Statistical methods for assessing agreement between
    two methods of clinical measurement. Lancet 327:307-310.
Bland JM, Altman DG (1999) Measuring agreement in method comparison studies.
    Stat Methods Med Res 8:135-160.
Carkeet A (2015) Exact parametric confidence intervals for Bland-Altman limits
    of agreement. Optom Vis Sci 92:e71-e80.
Cornbleet PJ, Gochman N (1979) Incorrect least-squares regression coefficients
    in method-comparison analysis. Clin Chem 25:432-438.
Cox DR (1970) The Analysis of Binary Data. Methuen, London.
Davison AC, Hinkley DV (1997) Bootstrap Methods and their Application.
    Cambridge University Press.
Efron B (1987) Better bootstrap confidence intervals. J Am Stat Assoc
    82:171-185.
Field CA, Welsh AH (2007) Bootstrapping clustered data. J R Stat Soc B
    69:369-390.
Friedman M (1937) The use of ranks to avoid the assumption of normality
    implicit in the analysis of variance. J Am Stat Assoc 32:675-701.
Greenhouse SW, Geisser S (1959) On methods in the analysis of profile data.
    Psychometrika 24:95-112.
Hodges JL, Lehmann EL (1963) Estimates of location based on rank tests. Ann
    Math Stat 34:598-611.
Hollander M, Wolfe DA, Chicken E (2014) Nonparametric Statistical Methods,
    3rd ed. Wiley.
Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J
    Stat 6:65-70.
Huynh H, Feldt LS (1976) Estimation of the Box correction for degrees of
    freedom from sample data in randomized block and split-plot designs.
    J Educ Stat 1:69-82.
Iman RL, Davenport JM (1980) Approximations of the critical region of the
    Friedman statistic. Commun Stat Theory Methods 9:571-595.
Kendall MG, Babington Smith B (1939) The problem of m rankings. Ann Math Stat
    10:275-287.
Kerby DS (2014) The simple difference formula: an approach to teaching
    nonparametric correlation. Compr Psychol 3:11.IT.3.1.
Krouwer JS (2008) Why Bland-Altman plots should use X, not (Y+X)/2 when X is a
    reference method. Stat Med 27:778-780.
Lin LI-K (1989) A concordance correlation coefficient to evaluate
    reproducibility. Biometrics 45:255-268.
Lin LI-K (2000) A note on the concordance correlation coefficient. Biometrics
    56:324-325.
Linnet K (1990) Estimation of the linear relationship between the measurements
    of two methods with proportional errors. Stat Med 9:1463-1473.
Linnet K (1993) Evaluation of regression procedures for methods comparison
    studies. Clin Chem 39:424-432.
Mauchly JW (1940) Significance test for sphericity of a normal n-variate
    distribution. Ann Math Stat 11:204-209.
McGraw KO, Wong SP (1996) Forming inferences about some intraclass
    correlation coefficients. Psychol Methods 1:30-46.
Olejnik S, Algina J (2003) Generalized eta and omega squared statistics:
    measures of effect size for some common research designs. Psychol Methods
    8:434-447.
Passing H, Bablok W (1983) A new biometrical procedure for testing the
    equality of measurements from two different analytical methods.
    J Clin Chem Clin Biochem 21:709-720.
Phipson B, Smyth GK (2010) Permutation p-values should never be zero.
    Stat Appl Genet Mol Biol 9:39.
Searle SR, Casella G, McCulloch CE (1992) Variance Components. Wiley.
Shrout PE, Fleiss JL (1979) Intraclass correlations: uses in assessing rater
    reliability. Psychol Bull 86:420-428.
Warton DI, Hui FKC (2011) The arcsine is asinine: the analysis of proportions
    in ecology. Ecology 92:3-10.
"""
from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import linalg as _sla
from scipy import special as _sps
from scipy import stats as sps

__all__ = [
    # utilities
    "p_adjust", "fmt_p", "fmt_ci", "clip_unit", "logit", "inv_logit", "arcsine_sqrt",
    "to_wide",
    # resampling
    "bootstrap_ci", "cluster_bootstrap", "permutation_test_paired",
    # repeated measures
    "kendall_w", "friedman", "wilcoxon_signed_rank", "hodges_lehmann", "pairwise_paired",
    "sphericity", "rm_anova_oneway", "rm_anova_twoway",
    # variance decomposition
    "variance_decomposition",
    # agreement
    "bland_altman", "lin_ccc", "deming", "passing_bablok", "error_metrics", "prob_best", "icc",
]

_EXACT_LIMIT = 1_000_000   # enumerate all 2**n sign patterns when 2**n <= this


# =============================================================================
# 0. small private helpers
# =============================================================================
def _check_rng(rng) -> np.random.Generator:
    """Return a Generator; refuse implicit global randomness.

    The pipeline must be reproducible, so resampling functions require an
    explicit generator (``ctx.rng(...)``) or an integer seed.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError("pass an explicit numpy Generator (ctx.rng(...)) or an int seed; "
                    "global random state is not allowed")


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _pair_xy(x, y) -> tuple[np.ndarray, np.ndarray, int]:
    """Paired finite values of x and y and the number of incomplete pairs dropped."""
    x = _as_1d(x)
    y = _as_1d(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length ({x.size} vs {y.size})")
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], int((~ok).sum())


def _zq(level: float) -> float:
    """Two-sided standard-normal quantile z_{1-alpha/2} for confidence ``level``."""
    return float(sps.norm.ppf(0.5 + level / 2.0))


def _tq(level: float, df: float) -> float:
    """Two-sided Student-t quantile t_{1-alpha/2, df}."""
    return float(sps.t.ppf(0.5 + level / 2.0, df)) if df > 0 else np.nan


def _pct_ci(boot: np.ndarray, level: float, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval of a bootstrap distribution (NaN replicates ignored)."""
    a = (1.0 - level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanquantile(boot, a, axis=axis)
        hi = np.nanquantile(boot, 1.0 - a, axis=axis)
    return lo, hi


def _nanstd(a: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=axis, ddof=1)


def _levels(s: pd.Series) -> list:
    """Levels of a factor in a stable order (categories order, else first appearance)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        present = set(s.dropna().unique())
        return [c for c in s.cat.categories if c in present]
    return list(pd.unique(s.dropna()))


def _signrank_pmf(n: int) -> np.ndarray:
    """Exact null distribution of the Wilcoxon signed-rank sum T+ (no ties).

    Under H0 each rank 1..n carries a + sign with probability 1/2
    independently, so the generating function of T+ is
    prod_{i=1..n} (1 + z^i) / 2.  Computed by dynamic programming in
    probability space (float64, relative error ~1e-15).  Returns
    ``pmf[t] = P(T+ = t)`` for t = 0 .. n(n+1)/2.
    """
    m = n * (n + 1) // 2
    p = np.zeros(m + 1)
    p[0] = 1.0
    for i in range(1, n + 1):
        q = p.copy()
        q[i:] += p[: m + 1 - i]
        p = 0.5 * q
    return p


def _sign_matrix(n: int, start: int, stop: int) -> np.ndarray:
    """Rows ``start..stop-1`` of the 2**n x n matrix of all +/-1 sign patterns."""
    idx = np.arange(start, stop, dtype=np.int64)[:, None]
    bits = (idx >> np.arange(n, dtype=np.int64)[None, :]) & 1
    return 1.0 - 2.0 * bits


# =============================================================================
# 1. utilities
# =============================================================================
def p_adjust(p, method: str = "holm") -> np.ndarray:
    """Adjust a family of p-values for multiple testing.

    Parameters
    ----------
    p : array_like
        Unadjusted p-values.  Non-finite entries (NaN) are *not* part of the
        family: they are returned as NaN and do not increase ``m``.
    method : {'holm', 'fdr_bh', 'bonferroni', 'none'}
        * ``'holm'`` - Holm (1979) step-down family-wise error control:
          for sorted p_(1) <= ... <= p_(m),
          p~_(i) = max_{j<=i} min(1, (m - j + 1) p_(j)).
        * ``'fdr_bh'`` - Benjamini & Hochberg (1995) false discovery rate:
          p~_(i) = min_{j>=i} min(1, m p_(j) / j).
        * ``'bonferroni'`` - min(1, m p).

    Returns
    -------
    numpy.ndarray
        Adjusted p-values with the shape of ``p``.
    """
    arr = np.asarray(p, dtype=float)
    flat = arr.ravel()
    ok = np.isfinite(flat)
    pv = flat[ok]
    m = pv.size
    out = np.full(flat.shape, np.nan)
    if m:
        method = (method or "none").lower()
        if method == "none":
            adj = pv.copy()
        elif method == "bonferroni":
            adj = np.minimum(pv * m, 1.0)
        elif method == "holm":
            order = np.argsort(pv, kind="mergesort")
            s = np.maximum.accumulate((m - np.arange(m)) * pv[order])
            adj = np.empty(m)
            adj[order] = np.minimum(s, 1.0)
        elif method in ("fdr_bh", "bh", "fdr"):
            order = np.argsort(pv, kind="mergesort")[::-1]
            rank = m - np.arange(m)                     # ascending rank of each sorted value
            s = np.minimum.accumulate(pv[order] * m / rank)
            adj = np.empty(m)
            adj[order] = np.minimum(s, 1.0)
        else:
            raise ValueError(f"unknown p_adjust method {method!r}")
        out[ok] = adj
    return out.reshape(arr.shape)


def fmt_p(p, digits: int = 3, prefix: bool = True) -> str:
    """Format a p-value in APA style (no leading zero; ``p < .001``).

    >>> fmt_p(0.0432)
    'p = .043'
    >>> fmt_p(2e-7)
    'p < .001'
    """
    if p is None or not np.isfinite(p):
        s = "n/a"
        return f"p = {s}" if prefix else s
    thr = 10.0 ** (-digits)
    if p < thr:
        s = "< " + f"{thr:.{digits}f}".lstrip("0")
    elif p > 1.0 - thr / 2.0:
        s = "> " + f"{1.0 - thr:.{digits}f}".lstrip("0")
    else:
        s = "= " + f"{p:.{digits}f}".lstrip("0")
    return f"p {s}" if prefix else s


def fmt_ci(est, lo, hi, digits: int = 2, unit: str = "", level: float = 0.95,
           style: str = "apa", scale: float = 1.0) -> str:
    """Format an estimate with its confidence interval.

    Parameters
    ----------
    est, lo, hi : float
        Point estimate and interval bounds (multiplied by ``scale`` before
        printing, e.g. ``scale=100`` to print a porosity fraction in %).
    digits : int
        Decimals.
    unit : str
        Unit appended to the estimate (e.g. ``'pp'``, ``'%'``).
    style : {'apa', 'bracket', 'text'}
        ``'apa'``: ``'12.34 pp, 95% CI [10.10, 14.50]'``;
        ``'bracket'``: ``'12.34 [10.10, 14.50] pp'``;
        ``'text'``: ``'12.34 pp (95% CI 10.10 to 14.50)'``.
    """
    def f(v):
        return "n/a" if v is None or not np.isfinite(v) else f"{v * scale:.{digits}f}"
    u = f" {unit}" if unit else ""
    lv = f"{100 * level:g}% CI"
    if style == "bracket":
        return f"{f(est)} [{f(lo)}, {f(hi)}]{u}"
    if style == "text":
        return f"{f(est)}{u} ({lv} {f(lo)} to {f(hi)})"
    if style == "apa":
        return f"{f(est)}{u}, {lv} [{f(lo)}, {f(hi)}]"
    raise ValueError(f"unknown style {style!r}")


def clip_unit(p, eps: float | None = None, n: int | None = None) -> np.ndarray:
    """Clip proportions into [eps, 1 - eps] before a logit / variance-stabilising transform.

    Porosities of exactly 0 or 1 (complete method failure) would map to
    -inf / +inf.  Following the *empirical logit* idea (Cox 1970) the natural
    clipping constant for a proportion measured on ``n`` pixels is half a
    pixel, ``eps = 0.5 / n``; if ``n`` is not given ``eps`` defaults to 1e-4
    (0.01 %, far below any porosity that is physically resolved here).
    The clipping is recorded by the caller; it only affects failed
    segmentations (porosity < eps or > 1 - eps).
    """
    if eps is None:
        eps = 0.5 / n if n else 1e-4
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p, eps: float | None = None, n: int | None = None) -> np.ndarray:
    """Log-odds ``log(p / (1 - p))`` after :func:`clip_unit` clipping.

    The logit is the recommended scale for analysing proportions (Warton &
    Hui 2011): it makes multiplicative-odds effects additive and removes the
    boundary at 0 and 1.
    """
    q = clip_unit(p, eps, n)
    return np.log(q) - np.log1p(-q)


def inv_logit(z) -> np.ndarray:
    """Inverse logit ``1 / (1 + exp(-z))`` (numerically stable, scipy.special.expit)."""
    return _sps.expit(np.asarray(z, dtype=float))


def arcsine_sqrt(p) -> np.ndarray:
    """Angular transform ``asin(sqrt(p))`` (radians) of proportions clipped to [0, 1].

    Variance-stabilising for binomial proportions (var ~ 1/(4n)); kept for
    comparison with older literature although the logit is usually
    preferable (Warton & Hui 2011).  Values outside [0, 1] are clipped.
    """
    return np.arcsin(np.sqrt(np.clip(np.asarray(p, dtype=float), 0.0, 1.0)))


def to_wide(df: pd.DataFrame, dv: str, within: str | Sequence[str], subject: str,
            dropna: bool = True) -> pd.DataFrame:
    """Pivot a tidy repeated-measures table to subjects x conditions.

    Parameters
    ----------
    df : DataFrame
        Tidy table with one row per (subject, condition) cell.
    dv : str
        Dependent variable column.
    within : str or list of str
        Within-subject factor(s).  With several factors the columns form a
        full ``MultiIndex.from_product`` in level order (categorical order if
        the column is categorical, otherwise order of first appearance).
    subject : str
        Subject (experimental unit, here the image) column.
    dropna : bool
        Drop subjects with at least one missing cell (complete-case analysis).
        The dropped subject labels are stored in ``wide.attrs['dropped']``.

    Raises
    ------
    ValueError
        If a (subject, condition) cell occurs more than once - the caller must
        aggregate explicitly; silent averaging is not allowed.
    """
    within = [within] if isinstance(within, str) else list(within)
    keys = [subject] + within
    d = df[keys + [dv]]
    dup = d.duplicated(keys)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated {keys} cells: aggregate first")
    subj = _levels(d[subject])
    levs = [_levels(d[w]) for w in within]
    if len(within) == 1:
        wide = d.pivot(index=subject, columns=within[0], values=dv).reindex(columns=levs[0])
    else:
        wide = d.pivot(index=subject, columns=within, values=dv)
        wide = wide.reindex(columns=pd.MultiIndex.from_product(levs, names=within))
    wide = wide.reindex(index=subj).astype(float)
    dropped: list = []
    if dropna:
        bad = wide.isna().any(axis=1).to_numpy()
        dropped = list(wide.index[bad])
        wide = wide.loc[~bad]
    wide.attrs["dropped"] = dropped
    return wide


# =============================================================================
# 2. resampling
# =============================================================================
_STATS = {"mean": np.mean, "median": np.median}


def bootstrap_ci(data, statistic: Callable | str = "mean", n_boot: int = 10000, rng=None,
                 method: str = "bca", level: float = 0.95, paired: bool = False,
                 vectorized: bool | None = None, return_distribution: bool = False) -> dict:
    """Non-parametric bootstrap confidence interval (wrapper of ``scipy.stats.bootstrap``).

    Parameters
    ----------
    data : array_like or tuple of array_like
        A single 1-D sample, or a *tuple* of samples for multi-sample
        statistics (e.g. a difference of means).  Only a ``tuple`` is
        interpreted as several samples; lists / arrays are one sample.
    statistic : callable or {'mean', 'median'}
        ``statistic(*samples)`` returning a scalar.  If it accepts an
        ``axis`` keyword it is called vectorised (much faster).
    n_boot : int
        Bootstrap replicates B.
    rng : numpy.random.Generator or int
        Random generator (required; use ``ctx.rng(...)``).
    method : {'bca', 'percentile', 'basic'}
        * ``'percentile'`` - quantiles theta*_(alpha/2), theta*_(1-alpha/2) of
          the bootstrap distribution.
        * ``'bca'`` - bias-corrected and accelerated interval (Efron 1987):
          the percentile levels are shifted to
          Phi(z0 + (z0 + z_a) / (1 - a (z0 + z_a))) with bias correction
          z0 = Phi^-1(#{theta* < theta^}/B) and acceleration ``a`` from the
          jackknife skewness.  Second-order accurate; preferred default.
        * ``'basic'`` - 2 theta^ - quantiles.
    level : float
        Confidence level.
    paired : bool
        Resample the samples *jointly* (same indices) - required for
        paired data such as (reference, test) pairs.
    return_distribution : bool
        Also return the bootstrap replicates under ``'boot'``.

    Returns
    -------
    dict
        ``estimate`` (statistic on the original data), ``lo``, ``hi``,
        ``se`` (SD of the replicates, ddof = 1), ``bias`` (mean replicate -
        estimate), ``method`` (method actually used; BCa falls back to the
        percentile interval when the jackknife is degenerate), ``level``,
        ``n_boot``, ``n`` (size of the first sample) and ``note``.

    Notes
    -----
    If all values are identical the bootstrap distribution is degenerate;
    the interval is then reported as [estimate, estimate] with a note.
    The bootstrap assumes independent observations: pass *image-level*
    values, never pixels.
    """
    rng = _check_rng(rng)
    samples = tuple(_as_1d(s) for s in data) if isinstance(data, tuple) else (_as_1d(data),)
    samples = tuple(s[np.isfinite(s)] for s in samples) if not paired else samples
    if paired:
        ok = np.all([np.isfinite(s) for s in samples], axis=0)
        samples = tuple(s[ok] for s in samples)
    if isinstance(statistic, str):
        statistic = _STATS[statistic]
    est = float(statistic(*samples))
    n = samples[0].size
    out = dict(estimate=est, lo=np.nan, hi=np.nan, se=np.nan, bias=np.nan, method=method,
               level=level, n_boot=int(n_boot), n=int(n), note="")
    if n < 2:
        out["note"] = "fewer than 2 observations"
        return out
    meth = {"bca": "BCa", "percentile": "percentile", "basic": "basic"}[method.lower()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sps.bootstrap(samples, statistic, n_resamples=int(n_boot), paired=paired,
                            vectorized=vectorized, confidence_level=level, method=meth, rng=rng)
    boot = np.asarray(res.bootstrap_distribution, dtype=float)
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        if np.nanstd(boot) == 0:
            lo = hi = est
            out["note"] = "degenerate bootstrap distribution (all replicates equal)"
        else:
            lo, hi = (float(v) for v in _pct_ci(boot, level))
            out["method"] = "percentile"
            out["note"] = f"{meth} undefined (degenerate jackknife); percentile interval used"
    out.update(lo=lo, hi=hi, se=float(np.std(boot, ddof=1)), bias=float(np.mean(boot) - est))
    if return_distribution:
        out["boot"] = boot
    return out


def cluster_bootstrap(df: pd.DataFrame, cluster_col: str, func: Callable[[pd.DataFrame], Any],
                      n_boot: int = 10000, rng=None, level: float = 0.95, relabel: bool = True,
                      return_distribution: bool = False):
    """Cluster (subject-level) bootstrap: resample whole images with replacement.

    Each replicate draws ``G`` clusters (``G`` = number of distinct values of
    ``cluster_col``) with replacement and concatenates *all* their rows, so
    that the within-image dependence structure (all methods, post-processing
    levels, ROIs and pixels of an image) is preserved and only the
    between-image variability is resampled (Davison & Hinkley 1997, sect.
    3.8; Field & Welsh 2007).  This is the correct bootstrap when the image is
    the experimental unit.

    Parameters
    ----------
    df : DataFrame
        Tidy data.
    cluster_col : str
        Column identifying the cluster (image / sample).
    func : callable
        ``func(df) -> float`` or ``-> dict / Series`` of floats.  Exceptions
        or non-finite results in a replicate are recorded as NaN.
    n_boot : int
        Number of replicates.
    rng : numpy.random.Generator or int
        Random generator (required).
    level : float
        Level of the percentile interval.
    relabel : bool
        If True a cluster drawn twice gets distinct labels
        (``'<label>#<draw>'``) so that functions which pivot by subject treat
        the copies as two subjects (as they must).  Set False only if
        ``func`` needs the original labels and does not pivot.
    return_distribution : bool
        Also return the replicate matrix.

    Returns
    -------
    dict or DataFrame
        Scalar ``func``: dict with ``estimate`` (func on the original data),
        ``se`` (SD of the replicates, ddof = 1), ``lo``, ``hi`` (percentile
        interval), ``n_valid`` (finite replicates), ``n_boot``,
        ``n_clusters``, ``level``.  Vector ``func``: DataFrame indexed by the
        returned keys with the same columns.

    Notes
    -----
    With few clusters (here 7-9 images) the percentile interval is only a
    rough guide: it is too narrow on average (the bootstrap variance carries
    a factor (G-1)/G) and statistics bounded at 0 (variance shares) are
    biased in resamples with duplicated clusters.  Always report G with it.
    """
    rng = _check_rng(rng)
    labels = _levels(df[cluster_col])
    groups = df.groupby(cluster_col, sort=False, observed=True).indices
    idx = [np.asarray(groups[c]) for c in labels]
    sizes = np.array([len(i) for i in idx])
    G = len(labels)
    est = func(df)
    scalar = np.ndim(est) == 0 and not isinstance(est, (dict, pd.Series))
    keys = None if scalar else list(pd.Series(est).index)
    est_vec = np.array([float(est)]) if scalar else pd.Series(est).astype(float).to_numpy()
    boots = np.full((int(n_boot), est_vec.size), np.nan)
    str_labels = np.array([str(c) for c in labels], dtype=object)
    for b in range(int(n_boot)):
        pick = rng.integers(0, G, G)
        rows = np.concatenate([idx[i] for i in pick])
        d = df.iloc[rows].copy()
        if relabel:
            new = np.repeat(np.array([f"{str_labels[i]}#{k}" for k, i in enumerate(pick)], dtype=object),
                            sizes[pick])
            d[cluster_col] = new
        try:
            v = func(d)
            v = np.array([float(v)]) if scalar else np.array(pd.Series(v).reindex(keys), dtype=float)
        except Exception:  # noqa: BLE001 - a failed replicate is recorded as NaN
            v = np.full(est_vec.size, np.nan)
        v[~np.isfinite(v)] = np.nan
        boots[b] = v
    lo, hi = _pct_ci(boots, level)
    se = _nanstd(boots)
    n_valid = np.isfinite(boots).sum(axis=0)
    if scalar:
        out = dict(estimate=float(est_vec[0]), se=float(se[0]), lo=float(lo[0]), hi=float(hi[0]),
                   n_valid=int(n_valid[0]), n_boot=int(n_boot), n_clusters=G, level=level,
                   method="cluster bootstrap, percentile")
        if return_distribution:
            out["boot"] = boots[:, 0]
        return out
    out = pd.DataFrame(dict(estimate=est_vec, se=se, lo=lo, hi=hi, n_valid=n_valid), index=keys)
    out.index.name = "term"
    out.attrs.update(n_boot=int(n_boot), n_clusters=G, level=level)
    if return_distribution:
        out.attrs["boot"] = boots
    return out


def permutation_test_paired(x, y=None, statistic: str | Callable = "mean", n_perm: int = 10000,
                            rng=None, alternative: str = "two-sided",
                            exact_limit: int = _EXACT_LIMIT) -> dict:
    """Sign-flip permutation test for paired data (H0: differences symmetric about 0).

    Under H0 (exchangeability of the two conditions within each subject) every
    difference d_i = x_i - y_i is equally likely to have either sign, so the
    reference distribution of a statistic T(d) is obtained from T(s * d) over
    sign vectors s in {-1, +1}^n.  When 2^n <= ``exact_limit`` (n <= 19 with
    the default 1e6) all sign patterns are enumerated and the p-value is
    *exact*; otherwise ``n_perm`` random sign vectors are drawn and
    p = (1 + #{T* at least as extreme}) / (1 + n_perm) (Phipson & Smyth 2010),
    which is never zero and is a valid (slightly conservative) p-value.

    Parameters
    ----------
    x, y : array_like
        Paired observations (``y=None``: ``x`` already holds the differences).
    statistic : {'mean', 'median', 'sum'} or callable
        Odd statistic of the differences.  A callable must accept a 2-D array
        (sign patterns x n) and an ``axis`` keyword.
    alternative : {'two-sided', 'greater', 'less'}
        Two-sided uses |T*| >= |T_obs| (valid because the sign-flip
        distribution of an odd statistic is symmetric).
    rng : Generator or int
        Needed only in the Monte-Carlo case.

    Returns
    -------
    dict
        ``statistic``, ``p``, ``n`` (pairs used; zero differences are kept -
        they contribute nothing to the statistic), ``n_missing``,
        ``exact`` (bool), ``n_perm`` (patterns evaluated), ``alternative``.
    """
    d = _as_1d(x) if y is None else _as_1d(x) - _as_1d(y)
    n_missing = int((~np.isfinite(d)).sum())
    d = d[np.isfinite(d)]
    n = d.size
    funcs = {"mean": np.mean, "median": np.median, "sum": np.sum}
    f = funcs[statistic] if isinstance(statistic, str) else statistic
    t_obs = float(f(d[None, :], axis=1)[0])
    tol = 1e-12 * max(1.0, abs(t_obs))

    def _count(tstar):
        if alternative == "two-sided":
            return int(np.sum(np.abs(tstar) >= abs(t_obs) - tol))
        if alternative == "greater":
            return int(np.sum(tstar >= t_obs - tol))
        if alternative == "less":
            return int(np.sum(tstar <= t_obs + tol))
        raise ValueError(f"unknown alternative {alternative!r}")

    if n == 0:
        return dict(statistic=np.nan, p=np.nan, n=0, n_missing=n_missing, exact=True, n_perm=0,
                    alternative=alternative)
    total = 2 ** n
    if total <= exact_limit:
        count = 0
        chunk = 1 << 16
        for start in range(0, total, chunk):
            s = _sign_matrix(n, start, min(total, start + chunk))
            count += _count(f(s * d[None, :], axis=1))
        p = count / total
        exact, n_eval = True, total
    else:
        rng = _check_rng(rng)
        count = 0
        done = 0
        while done < n_perm:
            b = min(20000, n_perm - done)
            s = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
            count += _count(f(s * d[None, :], axis=1))
            done += b
        p = (1 + count) / (1 + n_perm)
        exact, n_eval = False, int(n_perm)
    return dict(statistic=t_obs, p=float(min(1.0, p)), n=int(n), n_missing=n_missing, exact=exact,
                n_perm=int(n_eval), alternative=alternative)


# =============================================================================
# 3. repeated measures (subject = image)
# =============================================================================
def _rank_rows(wide: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid-ranks within each row and the tie term sum(t^3 - t) per row."""
    ranks = np.apply_along_axis(sps.rankdata, 1, wide)
    ties = np.zeros(wide.shape[0])
    for i, row in enumerate(wide):
        _, c = np.unique(row, return_counts=True)
        ties[i] = float(np.sum(c ** 3 - c))
    return ranks, ties


def kendall_w(wide) -> dict:
    """Kendall's coefficient of concordance W with tie correction.

    ``wide`` is subjects (m rows) x conditions (k columns).  Each subject ranks
    the k conditions; W = 12 S / (m^2 (k^3 - k) - m sum T), with
    S = sum_j (R_j - m(k+1)/2)^2 and T = sum over tie groups of (t^3 - t)
    (Kendall & Babington Smith 1939).  W = Q / (m (k - 1)) where Q is the
    tie-corrected Friedman statistic.  W in [0, 1]; 1 = identical rankings.
    """
    a = np.asarray(wide, dtype=float)
    m, k = a.shape
    ranks, ties = _rank_rows(a)
    R = ranks.sum(axis=0)
    S = float(np.sum((R - m * (k + 1) / 2.0) ** 2))
    denom = m ** 2 * (k ** 3 - k) - m * ties.sum()
    W = 12.0 * S / denom if denom > 0 else np.nan
    return dict(W=W, S=S, m=m, k=k, tie_term=float(ties.sum()))


def friedman(df: pd.DataFrame, dv: str, within: str, subject: str, n_boot: int = 0, rng=None,
             level: float = 0.95) -> dict:
    """Friedman test across the levels of ``within`` (subject = image).

    Returns the tie-corrected chi-square statistic Q (df = k - 1) with its
    asymptotic p-value, the Iman & Davenport (1980) F version (more accurate
    for small m), Kendall's W and, if ``n_boot > 0``, a percentile CI for W by
    cluster bootstrap over subjects.
    """
    wide = to_wide(df, dv, within, subject)
    a = wide.to_numpy()
    m, k = a.shape
    W = kendall_w(a)["W"]
    Q = W * m * (k - 1) if np.isfinite(W) else np.nan
    p = float(sps.chi2.sf(Q, k - 1)) if np.isfinite(Q) else np.nan
    F_id = (m - 1) * W / (1 - W) if np.isfinite(W) and W < 1 else np.inf
    df1, df2 = k - 1, (k - 1) * (m - 1)
    p_id = float(sps.f.sf(F_id, df1, df2)) if np.isfinite(F_id) else 0.0
    out = dict(Q=float(Q), df=k - 1, p=p, F_iman_davenport=float(F_id), df1=df1, df2=df2, p_iman_davenport=p_id,
               W=float(W), n=m, k=k, n_dropped=len(wide.attrs.get("dropped", [])), W_lo=np.nan, W_hi=np.nan)
    if n_boot and n_boot > 0:
        sub = df[df[subject].isin(wide.index)]

        def fW(d):
            return kendall_w(to_wide(d, dv, within, subject).to_numpy())["W"]
        cb = cluster_bootstrap(sub, subject, fW, n_boot=n_boot, rng=rng, level=level)
        out.update(W_lo=cb["lo"], W_hi=cb["hi"], W_se=cb["se"])
    return out


def wilcoxon_signed_rank(d, alternative: str = "two-sided") -> dict:
    """Exact Wilcoxon signed-rank test of median(d) = 0.

    Zero differences are dropped (Wilcoxon's convention, reported as
    ``n_zero``); tied |d| receive mid-ranks and the p-value is the exact
    conditional permutation p-value obtained by enumerating all 2^n sign
    patterns of the observed (mid-)ranks, so it remains exact with ties.
    ``p_min`` is the smallest p-value attainable with this n.
    """
    d = _as_1d(d)
    d = d[np.isfinite(d)]
    n_zero = int(np.sum(d == 0))
    d = d[d != 0]
    n = d.size
    if n == 0:
        return dict(T_plus=np.nan, p=1.0, n=0, n_zero=n_zero, exact=True, p_min=1.0)
    r = sps.rankdata(np.abs(d))
    T = float(r[d > 0].sum())
    E = r.sum() / 2.0
    if 2 ** n <= _EXACT_LIMIT:
        parts = []
        for start in range(0, 2 ** n, 1 << 16):
            s = _sign_matrix(n, start, min(2 ** n, start + (1 << 16)))
            parts.append(((s > 0) * r[None, :]).sum(axis=1))
        tot = np.concatenate(parts)
        tol = 1e-9
        if alternative == "two-sided":
            p = np.mean(np.abs(tot - E) >= abs(T - E) - tol)
        elif alternative == "greater":
            p = np.mean(tot >= T - tol)
        else:
            p = np.mean(tot <= T + tol)
        exact = True
        p_min = 2.0 / 2 ** n if alternative == "two-sided" else 1.0 / 2 ** n
    else:
        res = sps.wilcoxon(d, alternative=alternative, method="approx")
        p, exact, p_min = float(res.pvalue), False, np.nan
    return dict(T_plus=T, p=float(min(1.0, p)), n=n, n_zero=n_zero, exact=exact, p_min=float(p_min))


def hodges_lehmann(d, level: float = 0.95) -> dict:
    """One-sample Hodges-Lehmann estimator with the exact distribution-free CI.

    The estimate is the median of the n(n+1)/2 Walsh averages (d_i + d_j)/2,
    i <= j.  The CI is [W_(C), W_(M+1-C)] (order statistics of the Walsh
    averages, M = n(n+1)/2) where C is the number of values t with
    P(T+ <= t) <= alpha/2 under the exact signed-rank null distribution
    (Hollander, Wolfe & Chicken 2014, sect. 3.3).  Because the distribution is
    discrete the achieved confidence ``conf_achieved`` is >= ``level``; for
    very small n no interval at the requested level exists and the full range
    is returned with a note.
    """
    d = _as_1d(d)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return dict(estimate=np.nan, lo=np.nan, hi=np.nan, n=0, conf_achieved=np.nan, note="no data")
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    M = walsh.size
    est = float(np.median(walsh))
    cdf = np.cumsum(_signrank_pmf(n))
    alpha = 1.0 - level
    C = int(np.sum(cdf <= alpha / 2.0 + 1e-12))
    if C < 1:
        return dict(estimate=est, lo=float(walsh[0]), hi=float(walsh[-1]), n=n,
                    conf_achieved=float(1 - 2 * cdf[0]), note=f"n = {n} too small for a {level:.0%} interval")
    lo, hi = float(walsh[C - 1]), float(walsh[M - C])
    return dict(estimate=est, lo=lo, hi=hi, n=n, conf_achieved=float(1.0 - 2.0 * cdf[C - 1]), note="")


def pairwise_paired(df: pd.DataFrame, dv: str, within: str, subject: str, p_adjust_method: str = "holm",
                    level: float = 0.95, pairs: Sequence[tuple] | None = None) -> pd.DataFrame:
    """All pairwise paired comparisons between the levels of ``within``.

    For each pair (a, b) and difference d = b - a over subjects: mean and
    median difference, Hodges-Lehmann estimate with exact CI, exact Wilcoxon
    signed-rank p, paired t (CI and p), matched-pairs rank-biserial
    correlation r = (T+ - T-)/(T+ + T-) (Kerby 2014) and the p-value adjusted
    over the family of pairs (Holm by default).
    """
    wide = to_wide(df, dv, within, subject)
    levs = list(wide.columns)
    pairs = pairs or list(itertools.combinations(levs, 2))
    rows = []
    for a, b in pairs:
        d = (wide[b] - wide[a]).to_numpy()
        n = d.size
        hl = hodges_lehmann(d, level)
        w = wilcoxon_signed_rank(d)
        nz = d[d != 0]
        r_ab = sps.rankdata(np.abs(nz))
        tp, tm = r_ab[nz > 0].sum(), r_ab[nz < 0].sum()
        rbc = (tp - tm) / (tp + tm) if (tp + tm) > 0 else 0.0
        sd = np.std(d, ddof=1) if n > 1 else np.nan
        se = sd / math.sqrt(n) if n > 1 else np.nan
        tval = np.mean(d) / se if se and se > 0 else np.nan
        tq = _tq(level, n - 1)
        rows.append(dict(a=a, b=b, n=n, mean_diff=float(np.mean(d)), mean_lo=float(np.mean(d) - tq * se),
                         mean_hi=float(np.mean(d) + tq * se), median_diff=float(np.median(d)),
                         hl=hl["estimate"], hl_lo=hl["lo"], hl_hi=hl["hi"], hl_conf=hl["conf_achieved"],
                         T_plus=w["T_plus"], p_wilcoxon=w["p"], p_min=w["p_min"],
                         t=float(tval), p_t=float(2 * sps.t.sf(abs(tval), n - 1)) if np.isfinite(tval) else np.nan,
                         rank_biserial=float(rbc)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj"] = p_adjust(out["p_wilcoxon"].to_numpy(), p_adjust_method)
        out["p_adjust_method"] = p_adjust_method
    return out


def _orthonormal_contrasts(k: int) -> np.ndarray:
    """k x (k-1) orthonormal (normalised Helmert) contrasts."""
    C = np.zeros((k, k - 1))
    for j in range(1, k):
        C[:j, j - 1] = 1.0
        C[j, j - 1] = -float(j)
        C[:, j - 1] /= np.linalg.norm(C[:, j - 1])
    return C


def sphericity(Y: np.ndarray, C: np.ndarray) -> dict:
    """Mauchly's test and Greenhouse-Geisser / Huynh-Feldt epsilons.

    ``Y`` is subjects x cells, ``C`` the cells x p matrix of orthonormal
    contrasts of the effect.  With S = cov(Y C): eps_GG = tr(S)^2 / (p tr(S^2))
    (Greenhouse & Geisser 1959); eps_HF = (n p eps_GG - 2) / (p (n - 1 - p eps_GG))
    capped at 1 (Huynh & Feldt 1976; one-group design); Mauchly (1940)
    W = det(S) / (tr(S)/p)^p with the Bartlett chi-square approximation.
    """
    Z = Y @ C
    n, p = Z.shape
    S = np.atleast_2d(np.cov(Z, rowvar=False, ddof=1))
    tr = float(np.trace(S))
    eps_gg = float(tr ** 2 / (p * np.trace(S @ S))) if tr > 0 else np.nan
    den = p * (n - 1 - p * eps_gg) if np.isfinite(eps_gg) else np.nan
    eps_hf = float(min(1.0, (n * p * eps_gg - 2) / den)) if den and den > 0 else np.nan
    if p == 1:
        Wm, chi, dfm, pm = 1.0, 0.0, 0, 1.0
    elif n > p:
        Wm = float(np.linalg.det(S) / (tr / p) ** p) if tr > 0 else np.nan
        f = (2 * p ** 2 + p + 2) / (6 * p)
        chi = -(n - 1 - f) * math.log(Wm) if Wm and Wm > 0 else np.nan
        dfm = p * (p + 1) / 2 - 1
        pm = float(sps.chi2.sf(chi, dfm)) if np.isfinite(chi) else np.nan
    else:
        Wm, chi, dfm, pm = np.nan, np.nan, p * (p + 1) / 2 - 1, np.nan
    return dict(eps_gg=eps_gg, eps_hf=eps_hf, eps_lb=1.0 / p, mauchly_W=Wm, mauchly_chi2=chi,
                mauchly_df=dfm, mauchly_p=pm)


def _anova_terms(arr: np.ndarray) -> dict[tuple, float]:
    """Sums of squares of every term of a balanced full factorial (one obs/cell).

    ``arr`` has one axis per factor.  The effect of a term S (set of axes) is
    tau_S = sum_{T subset S} (-1)^{|S|-|T|} (mean over the axes not in T), and
    SS_S = sum over all cells of tau_S^2; the SS of all non-empty terms add
    up exactly to the total SS around the grand mean.
    """
    axes = tuple(range(arr.ndim))
    means = {}
    for r in range(arr.ndim + 1):
        for T in itertools.combinations(axes, r):
            other = tuple(a for a in axes if a not in T)
            means[T] = np.mean(arr, axis=other, keepdims=True) if other else arr
    ss = {}
    for r in range(1, arr.ndim + 1):
        for S in itertools.combinations(axes, r):
            tau = np.zeros(arr.shape)
            for q in range(len(S) + 1):
                for T in itertools.combinations(S, q):
                    tau = tau + (-1) ** (len(S) - len(T)) * np.broadcast_to(means[T], arr.shape)
            ss[S] = float(np.sum(tau ** 2))
    return ss


def _cube(wide: pd.DataFrame, factors: list[str]) -> np.ndarray:
    """Subjects x factor1 x factor2 ... array from a to_wide() table."""
    if len(factors) == 1:
        return wide.to_numpy()
    shape = [wide.shape[0]] + [len(dict.fromkeys(wide.columns.get_level_values(f))) for f in factors]
    return wide.to_numpy().reshape(shape)


def _rm_core(wide: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """All-within ANOVA (1 or more within factors) from a subjects x cells table."""
    arr = _cube(wide, factors)
    n = arr.shape[0]
    ks = list(arr.shape[1:])
    ss = _anova_terms(arr)
    Cs = [_orthonormal_contrasts(k) for k in ks]
    Y = wide.to_numpy()
    effect_terms = [S for S in ss if 0 not in S]
    error_ss = {S: ss[(0,) + S] for S in effect_terms}
    ss_subj = ss[(0,)]
    sum_err = sum(error_ss.values())
    rows = []
    for S in sorted(effect_terms, key=len):
        name = ":".join(factors[a - 1] for a in S)
        df1 = int(np.prod([ks[a - 1] - 1 for a in S]))
        df2 = df1 * (n - 1)
        ss_e, ss_err = ss[S], error_ss[S]
        F = (ss_e / df1) / (ss_err / df2) if ss_err > 0 else np.inf
        p = float(sps.f.sf(F, df1, df2))
        mats = [Cs[i] if (i + 1) in S else np.ones((ks[i], 1)) / math.sqrt(ks[i]) for i in range(len(ks))]
        C = mats[0]
        for M_ in mats[1:]:
            C = np.kron(C, M_)
        sph = sphericity(Y, C)
        eps = sph["eps_gg"] if df1 > 1 else 1.0
        p_gg = float(sps.f.sf(F, df1 * eps, df2 * eps)) if np.isfinite(eps) else np.nan
        rows.append(dict(term=name, SS=ss_e, df1=df1, df2=df2, MS=ss_e / df1, SS_error=ss_err, F=F, p=p,
                         eps_gg=eps, p_gg=p_gg, eps_hf=sph["eps_hf"] if df1 > 1 else 1.0,
                         mauchly_W=sph["mauchly_W"], mauchly_p=sph["mauchly_p"],
                         eta2_partial=ss_e / (ss_e + ss_err),
                         eta2_generalized=ss_e / (ss_e + ss_subj + sum_err)))
    out = pd.DataFrame(rows)
    out.attrs.update(n=n, SS_subject=ss_subj, SS_total=sum(ss.values()))
    return out


def rm_anova_oneway(df: pd.DataFrame, dv: str, within: str, subject: str) -> dict:
    """One-way repeated-measures ANOVA with sphericity diagnostics.

    Generalised eta-squared follows Olejnik & Algina (2003) / Bakeman (2005)
    for an all-within design: SS_effect / (SS_effect + SS_subjects + SS_errors).
    """
    wide = to_wide(df, dv, within, subject)
    t = _rm_core(wide, [within]).iloc[0].to_dict()
    t.update(n=wide.shape[0], n_dropped=len(wide.attrs.get("dropped", [])))
    return t


def rm_anova_twoway(df: pd.DataFrame, dv: str, subject: str, a: str, b: str) -> pd.DataFrame:
    """Two-way (a x b) all-within repeated-measures ANOVA, GG correction per term."""
    wide = to_wide(df, dv, [a, b], subject)
    out = _rm_core(wide, [a, b])
    out.attrs["n_dropped"] = len(wide.attrs.get("dropped", []))
    return out


# =============================================================================
# 4. factorial variance decomposition
# =============================================================================
def variance_decomposition(df: pd.DataFrame, dv: str, factors: Sequence[str], subject: str | None = None,
                           n_boot: int = 0, rng=None, level: float = 0.95) -> pd.DataFrame:
    """Decompose the total sum of squares of a fully crossed design (one obs/cell).

    ``factors`` lists every factor (e.g. ['sample_id', 'method', 'post']); the
    subject (image) factor is ``subject`` or the first one.  For each term the
    table gives SS, df, the share of the total SS (eta^2 = SS/SS_total, %) and
    a 'source' label: 'material' for the subject main effect and 'analyst' for
    every other term (main effects of the analyst's choices and all their
    interactions, including those with the image - the choice matters
    differently for different materials).  The highest-order term is the
    residual of the saturated model.  With ``n_boot > 0`` percentile CIs of
    each share come from a cluster bootstrap over subjects.  Rows
    'material (total)' and 'analyst (total)' give the two aggregated shares.
    """
    factors = list(factors)
    subject = subject or factors[0]
    factors = [subject] + [f for f in factors if f != subject]
    within = factors[1:]

    def _shares(d: pd.DataFrame) -> pd.Series:
        ss = _anova_terms(_cube(to_wide(d, dv, within, subject), within))
        tot = sum(ss.values())
        s = pd.Series({":".join(factors[a] for a in S): 100.0 * v / tot for S, v in ss.items()})
        s["material (total)"] = s[subject]
        s["analyst (total)"] = 100.0 - s[subject]
        return s

    wide = to_wide(df, dv, within, subject)
    arr = _cube(wide, within)
    ss = _anova_terms(arr)
    tot = sum(ss.values())
    rows = []
    for S, v in ss.items():
        dfree = int(np.prod([arr.shape[a] - 1 for a in S]))
        rows.append(dict(term=":".join(factors[a] for a in S), order=len(S), SS=v, df=dfree,
                         MS=v / dfree if dfree else np.nan, pct=100.0 * v / tot,
                         source="material" if S == (0,) else "analyst", residual=len(S) == len(factors)))
    out = pd.DataFrame(rows).sort_values(["order"], kind="stable").reset_index(drop=True)
    tot_rows = pd.DataFrame([dict(term="material (total)", order=99, SS=ss[(0,)], df=np.nan, MS=np.nan,
                                  pct=100.0 * ss[(0,)] / tot, source="material", residual=False),
                             dict(term="analyst (total)", order=99, SS=tot - ss[(0,)], df=np.nan, MS=np.nan,
                                  pct=100.0 * (tot - ss[(0,)]) / tot, source="analyst", residual=False)])
    out = pd.concat([out, tot_rows], ignore_index=True)
    out["lo"], out["hi"] = np.nan, np.nan
    if n_boot and n_boot > 0:
        sub = df[df[subject].isin(wide.index)]
        cb = cluster_bootstrap(sub, subject, _shares, n_boot=n_boot, rng=rng, level=level)
        out["lo"] = out["term"].map(cb["lo"])
        out["hi"] = out["term"].map(cb["hi"])
    out.attrs.update(SS_total=tot, n_subjects=wide.shape[0], factors=factors,
                     n_dropped=len(wide.attrs.get("dropped", [])))
    return out


# =============================================================================
# 5. agreement / method comparison (x = reference, y = test)
# =============================================================================
def bland_altman(x, y, level: float = 0.95, loa_z: float = 1.959963984540054) -> dict:
    """Bland-Altman analysis of d = y - x.

    Bias with its t-based CI; limits of agreement bias +/- z s (s with
    ddof = 1); CIs of the limits by (i) the approximate formula of Bland &
    Altman (1999), SE = sqrt(3 s^2 / n), and (ii) the exact interval of
    Carkeet (2015): the upper LoA estimates the normal quantile mu + z sigma,
    whose exact CI is mean + s * t'_{n-1}(q; z sqrt(n)) / sqrt(n) with t' the
    non-central t.  Proportional bias: OLS of d on the reference x
    (Krouwer 2008), slope with CI and p.
    """
    x, y, n_drop = _pair_xy(x, y)
    n = x.size
    d = y - x
    bias = float(np.mean(d))
    s = float(np.std(d, ddof=1)) if n > 1 else np.nan
    tq = _tq(level, n - 1)
    se_b = s / math.sqrt(n)
    lo_loa, hi_loa = bias - loa_z * s, bias + loa_z * s
    se_loa = math.sqrt(3 * s ** 2 / n)
    a2 = (1 - level) / 2
    nc = loa_z * math.sqrt(n)
    k_lo = sps.nct.ppf(a2, n - 1, nc) / math.sqrt(n)
    k_hi = sps.nct.ppf(1 - a2, n - 1, nc) / math.sqrt(n)
    out = dict(n=n, n_dropped=n_drop, bias=bias, bias_lo=bias - tq * se_b, bias_hi=bias + tq * se_b, sd=s,
               loa_lo=lo_loa, loa_hi=hi_loa,
               loa_lo_ci_approx=(lo_loa - tq * se_loa, lo_loa + tq * se_loa),
               loa_hi_ci_approx=(hi_loa - tq * se_loa, hi_loa + tq * se_loa),
               loa_hi_ci_exact=(bias + s * k_lo, bias + s * k_hi),
               loa_lo_ci_exact=(bias - s * k_hi, bias - s * k_lo))
    if n > 2 and np.std(x) > 0:
        lr = sps.linregress(x, d)
        tq2 = _tq(level, n - 2)
        out.update(prop_slope=float(lr.slope), prop_slope_lo=float(lr.slope - tq2 * lr.stderr),
                   prop_slope_hi=float(lr.slope + tq2 * lr.stderr), prop_p=float(lr.pvalue))
    return out


def lin_ccc(x, y, level: float = 0.95, n_boot: int = 0, rng=None) -> dict:
    """Lin's concordance correlation coefficient with Fisher-z CI.

    rho_c = 2 s_xy / (s_x^2 + s_y^2 + (m_x - m_y)^2) with 1/n moments
    (Lin 1989); rho_c = r * C_b where r (Pearson) measures precision and
    C_b accuracy.  The CI uses the asymptotic variance of z = atanh(rho_c)
    (Lin 1989, corrected in Lin 2000); optionally a percentile bootstrap CI.
    """
    x, y, n_drop = _pair_xy(x, y)
    n = x.size
    mx, my = x.mean(), y.mean()
    sx2, sy2 = np.mean((x - mx) ** 2), np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    rc = float(2 * sxy / (sx2 + sy2 + (mx - my) ** 2))
    r = float(sxy / math.sqrt(sx2 * sy2)) if sx2 > 0 and sy2 > 0 else np.nan
    out = dict(ccc=rc, r=r, Cb=rc / r if r else np.nan, n=n, n_dropped=n_drop, lo=np.nan, hi=np.nan)
    if n > 3 and np.isfinite(r) and abs(rc) < 1 and r != 0:
        u = (mx - my) / (sx2 * sy2) ** 0.25
        var_z = ((1 - r ** 2) * rc ** 2 / ((1 - rc ** 2) * r ** 2)
                 + 2 * rc ** 3 * (1 - rc) * u ** 2 / (r * (1 - rc ** 2) ** 2)
                 - rc ** 4 * u ** 4 / (2 * r ** 2 * (1 - rc ** 2) ** 2)) / (n - 2)
        if var_z > 0:
            z, q = math.atanh(rc), _zq(level)
            out.update(lo=math.tanh(z - q * math.sqrt(var_z)), hi=math.tanh(z + q * math.sqrt(var_z)))
    if n_boot and n_boot > 0:
        rng = _check_rng(rng)
        bs = np.full(int(n_boot), np.nan)
        for b, ii in enumerate(rng.integers(0, n, size=(int(n_boot), n))):
            xb, yb = x[ii], y[ii]
            den = np.var(xb) + np.var(yb) + (xb.mean() - yb.mean()) ** 2
            if den > 0:
                bs[b] = 2 * np.mean((xb - xb.mean()) * (yb - yb.mean())) / den
        lo, hi = _pct_ci(bs, level)
        out.update(boot_lo=float(lo), boot_hi=float(hi))
    return out


def deming(x, y, lam: float = 1.0, level: float = 0.95) -> dict:
    """Deming regression of y on x; ``lam`` = Var(error of y) / Var(error of x).

    slope = [s_yy - lam s_xx + sqrt((s_yy - lam s_xx)^2 + 4 lam s_xy^2)] / (2 s_xy),
    intercept = mean(y) - slope mean(x); SEs and t-CIs by jackknife
    (Linnet 1990, 1993).  lam = 1 is orthogonal regression.
    """
    x, y, n_drop = _pair_xy(x, y)
    n = x.size

    def fit(xx, yy):
        sxx, syy = np.var(xx, ddof=1), np.var(yy, ddof=1)
        sxy = np.cov(xx, yy, ddof=1)[0, 1]
        b = (syy - lam * sxx + math.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
        return b, yy.mean() - b * xx.mean()
    b, a = fit(x, y)
    jk = np.array([fit(np.delete(x, i), np.delete(y, i)) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))
    tq = _tq(level, n - 2)
    return dict(slope=float(b), slope_lo=float(b - tq * se[0]), slope_hi=float(b + tq * se[0]),
                intercept=float(a), intercept_lo=float(a - tq * se[1]), intercept_hi=float(a + tq * se[1]),
                lam=lam, n=n, n_dropped=n_drop)


def passing_bablok(x, y, level: float = 0.95) -> dict:
    """Passing-Bablok regression (Passing & Bablok 1983) with rank-based CIs.

    Slopes S_ij of all pairs (pairs with dx = dy = 0 or S = -1 excluded); K =
    number of slopes < -1; the slope is the median of the slopes shifted by K;
    CI from order statistics M1 = round((N - C)/2), M2 = N - M1 + 1 with
    C = z sqrt(n (n - 1)(2n + 5)/18).  Intercept = median(y - b x) with CI
    from the slope limits.  Reliable CIs need roughly n >= 30.
    """
    x, y, n_drop = _pair_xy(x, y)
    n = x.size
    i, j = np.triu_indices(n, 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    keep = ~((dx == 0) & (dy == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(dx[keep] == 0, np.sign(dy[keep]) * np.inf, dy[keep] / dx[keep])
    S = np.sort(S[S != -1])
    N = S.size
    K = int(np.sum(S < -1))
    if N == 0:
        return dict(slope=np.nan, intercept=np.nan, n=n, n_dropped=n_drop)
    def _at(i):
        return float(S[min(max(i, 0), N - 1)])
    b = _at((N + 1) // 2 + K - 1) if N % 2 else 0.5 * (_at(N // 2 + K - 1) + _at(N // 2 + K))
    C = _zq(level) * math.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    M1 = int(round((N - C) / 2.0))
    M2 = N - M1 + 1
    lo = float(S[min(N - 1, M1 + K - 1)]) if M1 >= 1 else np.nan
    hi = float(S[min(N - 1, M2 + K - 1)]) if M2 <= N else np.nan
    a = float(np.median(y - b * x))
    a_lo = float(np.median(y - hi * x)) if np.isfinite(hi) else np.nan
    a_hi = float(np.median(y - lo * x)) if np.isfinite(lo) else np.nan
    return dict(slope=b, slope_lo=lo, slope_hi=hi, intercept=a, intercept_lo=a_lo, intercept_hi=a_hi,
                n=n, n_dropped=n_drop, n_slopes=N, K=K,
                note="n < 30: Passing-Bablok CIs are unreliable" if n < 30 else "")


def error_metrics(x, y, n_boot: int = 10000, rng=None, level: float = 0.95) -> dict:
    """Bias, MAE, RMSE, MAPE (% of x) and max |error| of y vs reference x, each
    with a percentile bootstrap CI that resamples samples (pairs)."""
    x, y, n_drop = _pair_xy(x, y)
    n = x.size
    e = y - x

    def m(ee, xx):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([ee.mean(), np.abs(ee).mean(), math.sqrt(np.mean(ee ** 2)),
                             100 * np.mean(np.abs(ee) / np.abs(xx)), np.abs(ee).max()])
    names = ["bias", "mae", "rmse", "mape", "max_abs"]
    out = {k: float(v) for k, v in zip(names, m(e, x))}
    out.update(n=n, n_dropped=n_drop)
    if n_boot and n > 1:
        rng = _check_rng(rng)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        B = np.array([m(e[ii], x[ii]) for ii in idx])
        lo, hi = _pct_ci(B, level)
        for k, l_, h_ in zip(names, lo, hi):
            out[f"{k}_lo"], out[f"{k}_hi"] = float(l_), float(h_)
    return out


def prob_best(errors: pd.DataFrame, rng=None, n_boot: int = 10000, metric: str = "mae") -> pd.Series:
    """Bootstrap probability that each method (column) has the smallest MAE
    (or RMSE) when the samples (rows) are resampled with replacement.  Ties
    split the credit equally."""
    rng = _check_rng(rng)
    E = errors.to_numpy(dtype=float)
    n, k = E.shape
    wins = np.zeros(k)
    for _ in range(int(n_boot)):
        sub = E[rng.integers(0, n, n)]
        s = np.abs(sub).mean(axis=0) if metric == "mae" else np.sqrt((sub ** 2).mean(axis=0))
        best = np.flatnonzero(np.isclose(s, s.min()))
        wins[best] += 1.0 / best.size
    return pd.Series(wins / n_boot, index=errors.columns, name=f"prob_best_{metric}")


def icc(df: pd.DataFrame, targets: str, raters: str, ratings: str) -> pd.DataFrame:
    """Intraclass correlations (Shrout & Fleiss 1979; McGraw & Wong 1996).

    Wrapper of ``pingouin.intraclass_corr`` that labels each coefficient:
    ICC2/ICC2k = two-way random, ABSOLUTE agreement; ICC3/ICC3k = two-way
    mixed, CONSISTENCY (systematic offsets between raters are ignored).
    """
    import pingouin as pg
    t = pg.intraclass_corr(data=df, targets=targets, raters=raters, ratings=ratings).copy()
    lab = {"ICC1": "one-way random, absolute", "ICC1k": "one-way random, absolute",
           "ICC2": "two-way random, absolute agreement", "ICC2k": "two-way random, absolute agreement",
           "ICC3": "two-way mixed, consistency", "ICC3k": "two-way mixed, consistency"}
    t["agreement"] = t["Type"].map(lab)
    return t
