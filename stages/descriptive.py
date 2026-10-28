"""Stage 'descriptive': the descriptive and qualitative core of the paper.

What this stage answers
-----------------------
1. *What do the data look like?*  Overview of the nine slices with both ROI
   designs (``samples_overview``) and a schematic of the whole analysis
   (``study_design``).
2. *Why do the algorithms disagree?*  Grey-level histograms of the primary ROI
   with every threshold marked, and two complementary bimodality statistics
   (Sarle's bimodality coefficient, Ashman's D of a two-component Gaussian
   mixture). The association between bimodality and inter-method dispersion is
   tested across images (the image is the experimental unit, n = 9).
3. *How different are the binary images?*  A qualitative gallery on identical
   6 x 6 mm windows, per-sample supplementary grids (method x post-processing),
   and pixel-wise Dice / Jaccard overlap between every pair of methods,
   summarised by an average-linkage dendrogram.
4. *How large are the numbers?*  Porosity tables (method x sample, and the full
   factorial) and the effect of the ROI design (largest usable disc vs. the
   original central 500 x 500 px square).

Statistics used here (formulae in the docstrings of the pure functions)
-----------------------------------------------------------------------
* Sarle's finite-sample bimodality coefficient
  BC = (g1^2 + 1) / (g2 + 3 (n-1)^2 / ((n-2)(n-3))), g1 and g2 the adjusted
  (Fisher-Pearson) sample skewness and excess kurtosis; BC > 5/9 (the value of
  a uniform distribution) suggests bi- or multimodality (SAS Institute 1990;
  Pfister et al. 2013, Front. Psychol. 4:700).
* Ashman's D = sqrt(2) |mu1 - mu2| / sqrt(s1^2 + s2^2) for a two-component
  Gaussian mixture; D > 2 is required for a clean separation of the two
  modes (Ashman, Bird & Zepf 1994, Astron. J. 108:2348).
* Dice = 2|A n B| / (|A| + |B|) (Dice 1945, Ecology 26:297) and Jaccard
  = |A n B| / |A u B| (Jaccard 1912, New Phytol. 11:37); J = D / (2 - D).
* Spearman's rank correlation with an *exact* permutation p-value (all n!
  pairings enumerated for n <= 10), a percentile bootstrap CI over images
  (Efron & Tibshirani 1993) and the Bonett & Wright (2000, Psychometrika
  65:23) Fisher-z interval as a cross-check; Holm (1979, Scand. J. Stat.
  6:65) adjustment within each family of tests.

Pixels are never treated as replicates: every inferential statement here is
made across images (n = 9; n = 4 granular packs, n = 5 consolidated cores).

Outputs
-------
figures  samples_overview, study_design, histograms_thresholds,
         segmentation_gallery, method_similarity (main);
         segmentation_grid_<label>, roi_design_effect,
         bimodality_vs_dispersion (supplementary)
tables   porosity_by_method (main); histogram_bimodality, porosity_factorial,
         roi_design_effect (supplementary)
data     bimodality, histograms, threshold_marks, bimodality_association,
         dice_pairs, dice_mean, dice_family_summary, zoom_windows,
         roi_design_effect, roi_design_summary, intermethod_dispersion
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import ndimage as ndi
from scipy import stats as sps

from .. import postprocess, style
from ..thresholds import METHOD_INFO

# ----------------------------------------------------------------------------
# settings of this stage (documented in the captions / methods)
# ----------------------------------------------------------------------------
NBINS = 256                       # histogram bins on [0, 1] (= the global methods' resolution)
FAIL_LO, FAIL_HI = 0.01, 0.90     # 'degenerate' segmentation: phi < 1 % or phi > 90 % (strict)
COLLAPSE_LO = 0.05                # lenient definition adds 'collapsed' segmentations: phi < 5 %
BC_CRIT = 5.0 / 9.0               # Sarle's BC of a uniform distribution
D_CRIT = 2.0                      # Ashman's D criterion for clean separation
GMM_MAX_N = 250_000               # pixels used to fit the 2-component GMM (random subsample)
WINDOW_MM = 6.0                   # physical size of the zoom windows (gallery and grids)
GALLERY_SAMPLES = ("DS23_CF5", "AG24_VN04", "AG24_VN02", "BC23_TQ06")
GALLERY_ROLE = {"DS23_CF5": "granular pack", "AG24_VN04": "heterogeneous core",
                "AG24_VN02": "unimodal: collapse", "BC23_TQ06": "unimodal: saturation"}
# Optional manual zoom-window centres (mm from the image's top-left corner,
# (x, y)); leave empty to use the automatic, reproducible choice.
WINDOW_OVERRIDE_MM: dict[str, tuple[float, float]] = {}

ROI_LINE = dict(color="white", lw=0.8)
ROI_HALO = [patheffects.withStroke(linewidth=1.9, foreground="black")]
LS_DISK, LS_SQUARE, LS_WINDOW = "-", (0, (4, 2)), (0, (1, 1.2))


# ============================================================================
# pure functions (unit-tested in tests/test_descriptive.py)
# ============================================================================
def sarle_bc(x: np.ndarray) -> dict:
    """Sarle's finite-sample bimodality coefficient.

    BC = (g1^2 + 1) / (g2 + 3 (n-1)^2 / ((n-2)(n-3)))

    where g1 is the adjusted Fisher-Pearson skewness G1 and g2 the adjusted
    excess kurtosis G2 (``scipy.stats.skew/kurtosis(bias=False)``). For a
    normal distribution BC -> 1/3, for a uniform BC -> 5/9; BC > 5/9 suggests
    a bi- or multimodal distribution (SAS Institute 1990; Pfister et al. 2013).
    BC is affine invariant, so it does not depend on the grey-level scaling.
    Caveat: strong skewness of a unimodal distribution can also raise BC.

    Returns dict(bc, skewness, excess_kurtosis, n).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        raise ValueError("BC needs n >= 4")
    g1 = float(sps.skew(x, bias=False))
    g2 = float(sps.kurtosis(x, fisher=True, bias=False))
    bc = (g1 ** 2 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return dict(bc=float(bc), skewness=g1, excess_kurtosis=g2, n=int(n))


def ashman_d(mu1: float, sd1: float, mu2: float, sd2: float) -> float:
    """Ashman's D = sqrt(2) |mu1 - mu2| / sqrt(sd1^2 + sd2^2).

    Separation of two Gaussian components in units of their pooled spread;
    D > 2 is required for a clean separation (Ashman et al. 1994).
    """
    return float(np.sqrt(2.0) * abs(mu1 - mu2) / np.sqrt(sd1 ** 2 + sd2 ** 2))


def fit_gmm2(x: np.ndarray, random_state: int, rng: np.random.Generator | None = None,
             max_n: int = GMM_MAX_N, n_init: int = 3) -> dict:
    """Two-component 1-D Gaussian mixture (EM, ``sklearn.mixture.GaussianMixture``).

    At most ``max_n`` values (random subsample drawn with ``rng``) are used;
    with >= 2.5e5 pixels the sampling error of the component parameters is
    negligible (< 1e-3 grey levels). Components are returned sorted by mean
    (component 1 = darker = pore-like mode) together with Ashman's D.
    """
    from sklearn.mixture import GaussianMixture

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size > max_n:
        rng = rng if rng is not None else np.random.default_rng(random_state)
        x = x[rng.choice(x.size, max_n, replace=False)]
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=int(random_state),
                         n_init=n_init, max_iter=1000, tol=1e-6, reg_covar=1e-8)
    gm.fit(x.reshape(-1, 1))
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    w = gm.weights_.ravel()
    o = np.argsort(mu)
    mu, sd, w = mu[o], sd[o], w[o]
    return dict(gmm_mu1=float(mu[0]), gmm_sd1=float(sd[0]), gmm_w1=float(w[0]),
                gmm_mu2=float(mu[1]), gmm_sd2=float(sd[1]), gmm_w2=float(w[1]),
                ashman_d=ashman_d(mu[0], sd[0], mu[1], sd[1]),
                gmm_converged=bool(gm.converged_), gmm_n_fit=int(x.size))


def gauss_bin_fraction(centres: np.ndarray, width: float, mu: float, sd: float, w: float) -> np.ndarray:
    """Expected fraction of pixels per histogram bin for one mixture component."""
    lo, hi = centres - width / 2, centres + width / 2
    return w * (sps.norm.cdf(hi, mu, sd) - sps.norm.cdf(lo, mu, sd))


def local_threshold_band(tmap: np.ndarray, mask: np.ndarray, q=(5.0, 95.0)) -> tuple[float, float, float]:
    """Mean and (q_lo, q_hi) percentiles of a local threshold map inside ``mask``."""
    v = np.asarray(tmap)[mask].astype(np.float64)
    lo, hi = np.percentile(v, q)
    return float(v.mean()), float(lo), float(hi)


def failure_flags(phi, lo: float = FAIL_LO, hi: float = FAIL_HI) -> np.ndarray:
    """True where a segmentation is degenerate (porosity < lo or > hi)."""
    phi = np.asarray(phi, dtype=float)
    return (phi < lo) | (phi > hi)


def pairwise_overlap(pores: dict[str, np.ndarray], eval_mask: np.ndarray) -> pd.DataFrame:
    """Dice and Jaccard for every unordered pair of binary pore images.

    Dice = 2|A n B| / (|A| + |B|);  Jaccard = |A n B| / |A u B| = Dice / (2 - Dice).
    Computed only on ``eval_mask``; both are 1 by convention when A and B are empty.
    """
    vec = {m: np.asarray(p)[eval_mask] for m, p in pores.items()}
    cnt = {m: int(np.count_nonzero(v)) for m, v in vec.items()}
    rows = []
    for a, b in itertools.combinations(list(pores), 2):
        inter = int(np.count_nonzero(vec[a] & vec[b]))
        s = cnt[a] + cnt[b]
        union = s - inter
        rows.append(dict(method_a=a, method_b=b, n_a=cnt[a], n_b=cnt[b], n_inter=inter,
                         dice=2.0 * inter / s if s else 1.0, jaccard=inter / union if union else 1.0))
    return pd.DataFrame(rows)


def overlap_matrix(pairs: pd.DataFrame, methods: list[str], col: str = "dice") -> np.ndarray:
    """Symmetric matrix (diag = 1) of the mean of ``col`` over all rows of each pair."""
    k = len(methods)
    M = np.eye(k)
    g = pairs.groupby(["method_a", "method_b"])[col].mean()
    idx = {m: i for i, m in enumerate(methods)}
    for (a, b), v in g.items():
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = v
    return M


def _rank_rows(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, axis=-1)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / den, np.nan)


def spearman_rho(x, y) -> float:
    """Spearman's rho = Pearson correlation of the (average) ranks."""
    return float(_pearson_rows(_rank_rows(np.asarray(x, float)), _rank_rows(np.asarray(y, float))))


@lru_cache(maxsize=4)
def _all_permutations(n: int) -> np.ndarray:
    it = itertools.chain.from_iterable(itertools.permutations(range(n)))
    return np.fromiter(it, dtype=np.int8, count=math.factorial(n) * n).reshape(-1, n)


def spearman_permutation(x, y, rng: np.random.Generator | None = None, max_exact_n: int = 10,
                         n_perm: int = 200_000) -> dict:
    """Spearman's rho with a permutation p-value (exact for n <= ``max_exact_n``).

    Under H0 (exchangeability of the pairing) every one of the n! re-pairings
    of the y-ranks with the x-ranks is equally likely; the exact p-value is the
    proportion of re-pairings whose statistic is at least as extreme as the
    observed one (the identity pairing is included, so p >= 1/n!). Ties are
    handled with average ranks (the test is then conditional on the tie
    pattern). Returns rho, the two-sided p (|rho*| >= |rho|) and both one-sided
    p-values, and whether enumeration was exact.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    rx, ry = _rank_rows(x), _rank_rows(y)
    rho = float(_pearson_rows(rx, ry))
    if n <= max_exact_n:
        perms = _all_permutations(n)
        exact = True
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        perms = np.vstack([np.arange(n), rng.permuted(np.tile(np.arange(n), (n_perm - 1, 1)), axis=1)])
        exact = False
    rxc = rx - rx.mean()
    ryp = ry[perms] - ry.mean()
    den = np.sqrt((rxc ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    null = (ryp @ rxc) / den if den > 0 else np.full(len(perms), np.nan)
    eps = 1e-12
    return dict(rho=rho, p_two_sided=float(np.mean(np.abs(null) >= abs(rho) - eps)),
                p_greater=float(np.mean(null >= rho - eps)), p_less=float(np.mean(null <= rho + eps)),
                n=int(n), n_perm=int(len(perms)), exact=exact)


def spearman_boot_ci(x, y, n_boot: int, rng: np.random.Generator, level: float = 0.95) -> dict:
    """Percentile bootstrap CI of Spearman's rho, resampling *images* (pairs).

    Resamples in which x or y is constant (rho undefined) are discarded and
    counted. With n < 10 the bootstrap distribution is coarse and the interval
    should be read as indicative only.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    r = _pearson_rows(_rank_rows(x[idx]), _rank_rows(y[idx]))
    ok = np.isfinite(r)
    a = (1 - level) / 2
    lo, hi = np.quantile(r[ok], [a, 1 - a]) if ok.any() else (np.nan, np.nan)
    return dict(ci_lo=float(lo), ci_hi=float(hi), n_valid=int(ok.sum()), n_boot=int(n_boot))


def bonett_wright_ci(rho: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher-z CI for Spearman's rho with the Bonett & Wright (2000) variance
    var(z) = (1 + rho^2 / 2) / (n - 3)."""
    if n <= 3 or not np.isfinite(rho):
        return (np.nan, np.nan)
    r = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    q = sps.norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def holm(p) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(p, float)
    m = p.size
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def bootstrap_ci(values, stat=np.median, n_boot: int = 10000, rng: np.random.Generator | None = None,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI of ``stat`` (resampling the values)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    b = stat(v[idx], axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(b, [a, 1 - a])
    return float(stat(v)), float(lo), float(hi)


def _window_means(a: np.ndarray, w: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Mean of ``a`` over w x w windows with top-left corners (ys, xs) (summed-area table)."""
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(a, 0, dtype=np.float64), 1)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    tot = S[Y + w, X + w] - S[Y, X + w] - S[Y + w, X] + S[Y, X]
    return tot / (w * w)


def choose_window(score_maps: list[np.ndarray], valid_mask: np.ndarray, w: int,
                  stride: int | None = None) -> dict:
    """Choose a w x w window lying entirely inside ``valid_mask``.

    Each score map is averaged over every candidate window (grid with the
    given stride), standardised (z-score) across the valid candidates, and the
    window maximising the sum of the z-scores is returned. Deterministic.
    """
    h, wd = valid_mask.shape
    if h < w or wd < w:
        raise ValueError("window larger than the image")
    stride = stride or max(1, w // 6)
    ys = np.arange(0, h - w + 1, stride)
    xs = np.arange(0, wd - w + 1, stride)
    inside = _window_means(valid_mask.astype(np.float64), w, ys, xs) >= 1.0 - 1e-12
    if not inside.any():
        raise ValueError("no window fits inside the valid mask")
    total = np.zeros(inside.shape)
    for s in score_maps:
        m = _window_means(np.asarray(s, np.float64), w, ys, xs)
        v = m[inside]
        sd = v.std()
        total += (m - v.mean()) / sd if sd > 0 else 0.0
    total[~inside] = -np.inf
    i, j = np.unravel_index(np.argmax(total), total.shape)
    return dict(y0=int(ys[i]), x0=int(xs[j]), size=int(w), score=float(total[i, j]))


def overlay_rgb(grey: np.ndarray, pore: np.ndarray | None, vmin: float, vmax: float,
                color: str = style.PORE_OVERLAY, alpha: float = 0.6) -> np.ndarray:
    """Grey image (linear contrast vmin..vmax) as RGB with pores alpha-blended in ``color``."""
    g = np.clip((np.asarray(grey, np.float32) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = np.repeat(g[..., None], 3, axis=2)
    if pore is not None:
        c = np.asarray(to_rgb(color), np.float32)
        p = np.asarray(pore, bool)
        rgb[p] = (1 - alpha) * rgb[p] + alpha * c
    return rgb.astype(np.float32)


def robust_limits(values: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    a, b = np.percentile(np.asarray(values, np.float64), [lo, hi])
    return float(a), float(b) if b > a else float(a + 1e-6)


# ============================================================================
# helpers bound to the run context
# ============================================================================
def _release(ctx, sid: str) -> None:
    """Drop the per-sample caches of the context (keeps the process < ~800 MB)."""
    for name in ("_pore", "_thr", "_norm"):
        d = getattr(ctx, name, None)
        if isinstance(d, dict):
            for k in [k for k in d if isinstance(k, tuple) and k and k[0] == sid]:
                d.pop(k, None)
    order = getattr(ctx, "_pore_order", None)
    if isinstance(order, list):
        order[:] = [k for k in order if not (isinstance(k, tuple) and k and k[0] == sid)]
    raw = getattr(ctx, "_raw", None)
    if isinstance(raw, dict):
        raw.pop(sid, None)


def _seg_value(ctx, sid, method, post, roi=None) -> float:
    s = ctx.seg
    roi = roi or ctx.primary_roi
    v = s.loc[(s["sample_id"] == sid) & (s["roi"] == roi) & (s["method"] == method) & (s["post"] == post), "value"]
    return float(v.iloc[0]) if len(v) else np.nan


def _group_label(ctx, sid) -> str:
    return style.GROUP_LABELS.get(ctx.group(sid), ctx.group(sid))


def _fmt_pct(v: float, d: int = 1) -> str:
    return "" if not np.isfinite(v) else f"{100 * v:.{d}f}"


def _process_sample(ctx, sid: str) -> dict:
    """Everything per sample that needs the images; returns small arrays/records only."""
    roi = ctx.primary_roi
    R = ctx.roi(sid, roi)
    N = ctx.norm(sid, roi)
    px = ctx.pixel_size_um(sid)
    ev = R.eval_mask
    v = N.img[ev].astype(np.float64)

    # --- histogram and bimodality -------------------------------------------------
    counts, edges = np.histogram(v, bins=NBINS, range=(0.0, 1.0))
    bc = sarle_bc(v)
    rs = int(ctx.rng("descriptive", "gmm_seed", sid).integers(2 ** 31 - 1))
    gm = fit_gmm2(v, rs, rng=ctx.rng("descriptive", "gmm_subsample", sid))
    rec = dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), n_eval_px=int(v.size),
               mean=float(v.mean()), sd=float(v.std(ddof=1)), median=float(np.median(v)),
               frac_clipped_low=float(np.mean(v <= 0.0)), frac_clipped_high=float(np.mean(v >= 1.0)),
               skewness=bc["skewness"], excess_kurtosis=bc["excess_kurtosis"], bc=bc["bc"], **gm,
               bc_bimodal=bool(bc["bc"] > BC_CRIT), d_separated=bool(gm["ashman_d"] > D_CRIT))
    del v

    # --- thresholds (global value; local mean and 5-95 % band) ----------------
    marks = []
    for m in ctx.methods:
        T = ctx.threshold(sid, m, roi)
        if T.is_map:
            mean, lo, hi = local_threshold_band(T.value, ev)
        else:
            mean = lo = hi = float(T.value)
        marks.append(dict(sample_id=sid, label=ctx.label(sid), method=m, family=T.family,
                          t_mean=mean, t_p05=lo, t_p95=hi,
                          porosity=_seg_value(ctx, sid, m, "none")))
    _release_thr = getattr(ctx, "_thr", None)
    if isinstance(_release_thr, dict):     # local maps are large: drop them now
        for k in [k for k in _release_thr if k[0] == sid]:
            _release_thr.pop(k, None)

    # --- pairwise overlap (area-preserving levels) ----------------------------
    dice_rows = []
    for post in postprocess.AREA_PRESERVING:
        pores = {m: ctx.pore(sid, m, post, roi) for m in ctx.methods}
        d = pairwise_overlap(pores, ev)
        d.insert(0, "post", post)
        d.insert(0, "label", ctx.label(sid))
        d.insert(0, "sample_id", sid)
        dice_rows.append(d)
    dice = pd.concat(dice_rows, ignore_index=True)

    # --- zoom window -------------------------------------------------------------
    w = int(round(WINDOW_MM * 1000.0 / px))
    votes = np.zeros(R.mask.shape, np.float32)
    for m in ctx.methods:
        votes += ctx.pore(sid, m, "none", roi)
    p = votes / len(ctx.methods)
    disagreement = p * (1 - p)
    grad = ndi.gaussian_gradient_magnitude(N.img, 1.0).astype(np.float32)
    valid = ndi.distance_transform_edt(R.mask) > 8
    if sid in WINDOW_OVERRIDE_MM:
        cxm, cym = WINDOW_OVERRIDE_MM[sid]
        x0 = int(round(cxm * 1000 / px - w / 2)) - R.x0
        y0 = int(round(cym * 1000 / px - w / 2)) - R.y0
        win = dict(y0=y0, x0=x0, size=w, score=np.nan, source="manual")
    else:
        win = choose_window([disagreement, grad], valid, w)
        win["source"] = "auto"
    del votes, p, disagreement, grad, valid
    sl = (slice(win["y0"], win["y0"] + w), slice(win["x0"], win["x0"] + w))
    grey_lim = robust_limits(N.img[ev])
    crops = {"grey": N.img[sl].copy()}
    for m in ctx.methods:
        for post in ctx.posts:
            crops[(m, post)] = ctx.pore(sid, m, post, roi)[sl].copy()
    win_rec = dict(sample_id=sid, label=ctx.label(sid), source=win["source"], size_px=w,
                   size_mm=w * px / 1000.0, x0_px=R.x0 + win["x0"], y0_px=R.y0 + win["y0"],
                   x0_mm=(R.x0 + win["x0"]) * px / 1000.0, y0_mm=(R.y0 + win["y0"]) * px / 1000.0,
                   centre_x_mm=(R.x0 + win["x0"] + w / 2) * px / 1000.0,
                   centre_y_mm=(R.y0 + win["y0"] + w / 2) * px / 1000.0,
                   mean_disagreement_score=win["score"],
                   window_otsu_porosity=float(crops[("Otsu", "none")].mean()))

    # --- full-field display image (robust contrast from the ROI) ---------------
    raw = ctx.raw(sid).data
    lo, hi = robust_limits(raw[R.slices][ev])
    disp = (np.clip((raw - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    g = ctx.geometry(sid)
    C = ctx.roi(sid, "center500") if "center500" in ctx.cfg["roi"]["designs"] else None
    return dict(rec=rec, hist=(counts, edges), marks=marks, dice=dice, window=win_rec,
                crops=crops, grey_lim=grey_lim, disp=disp,
                geom=dict(cx=g.cx, cy=g.cy, r=g.r_usable),
                square=None if C is None else (C.x0, C.y0, C.x1 - C.x0, C.y1 - C.y0))


# ============================================================================
# figures
# ============================================================================
def _mm_axes(fig, W, H, x, y, w, h, **kw):
    """Axes placed in millimetres (x, y = bottom-left corner)."""
    return fig.add_axes((x / W, y / H, w / W, h / H), **kw)


def _image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _fig_overview(ctx, per, sids):
    W = style.figsize("double")[0] / style.MM
    ncol = 3
    nrow = int(np.ceil(len(sids) / ncol))
    gap_x, gap_y, top, bottom = 3.0, 6.5, 1.0, 9.0
    pw = (W - (ncol - 1) * gap_x) / ncol
    H = top + nrow * pw + (nrow - 1) * gap_y + 5.5 + bottom
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    letters = "abcdefghijklmnopqrstuvwxyz"
    for k, sid in enumerate(sids):
        r, c = divmod(k, ncol)
        x = c * (pw + gap_x)
        y = bottom + (nrow - 1 - r) * (pw + gap_y)
        ax = _mm_axes(fig, W, H, x, y, pw, pw)
        d = per[sid]
        img = d["disp"]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
        g = d["geom"]
        ax.add_patch(Circle((g["cx"], g["cy"]), g["r"], fill=False, ls=LS_DISK, path_effects=ROI_HALO, **ROI_LINE))
        if d["square"] is not None:
            x0, y0, ww, hh = d["square"]
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), ww, hh, fill=False, ls=LS_SQUARE,
                                   path_effects=ROI_HALO, **ROI_LINE))
        wr = d["window"]
        ax.add_patch(Rectangle((wr["x0_px"] - 0.5, wr["y0_px"] - 0.5), wr["size_px"], wr["size_px"],
                               fill=False, ls=LS_WINDOW, path_effects=ROI_HALO, **ROI_LINE))
        h, wd = img.shape
        side = max(h, wd)
        ax.set_xlim(wd / 2 - side / 2 - 0.5, wd / 2 + side / 2 - 0.5)
        ax.set_ylim(h / 2 + side / 2 - 0.5, h / 2 - side / 2 - 0.5)
        ax.set_facecolor("black")
        _image_axis(ax)
        style.scalebar(ax, ctx.pixel_size_um(sid), color="white")
        ax.set_title(f"{ctx.label(sid)}  ·  {_group_label(ctx, sid).lower()}  ·  {ctx.pixel_size_um(sid):g} µm pixel",
                     pad=2.5, loc="left", x=0.06)
        style.panel_label(ax, letters[k], x=0.05, y=1.005)
    hs = [Line2D([], [], ls=LS_DISK, path_effects=ROI_HALO, label="Largest usable disc (primary ROI)", **ROI_LINE),
          Line2D([], [], ls=LS_SQUARE, path_effects=ROI_HALO, label="Central 500 × 500 px (original design)", **ROI_LINE),
          Line2D([], [], ls=LS_WINDOW, path_effects=ROI_HALO,
                 label=f"{WINDOW_MM:g} × {WINDOW_MM:g} mm zoom window", **ROI_LINE)]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=3.2,
               columnspacing=2.5, frameon=True, facecolor="#9a9994", edgecolor="none", framealpha=1.0)
    return fig


def _box(ax, x, y, w, h, title=None, lines=(), fc="#f3f2ef", ec=style.INK2, lw=0.6, title_size=7.5,
         size=6.8, pad_top=2.6, line_gap=2.75, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    xt = x + w / 2 if align == "center" else x + 2.0
    yy = y + h - pad_top
    if title:
        ax.text(xt, yy, title, ha=align, va="center", fontsize=title_size, fontweight="bold", color=style.INK)
        yy -= line_gap + 0.4
    for ln in lines:
        ax.text(xt, yy, ln, ha=align, va="center", fontsize=size, color=style.INK2)
        yy -= line_gap


def _chip(ax, x, y, w, h, text, color, marker, size=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=color, lw=0.9))
    ax.plot([x + 2.2], [y + h / 2], ls="none", marker=marker, ms=3.6, color=color, mec=color)
    ax.text(x + 4.2, y + h / 2, text, ha="left", va="center", fontsize=size, color=style.INK)


def _arrow(ax, p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color=style.INK2, shrinkA=0, shrinkB=0, **kw))


def _fig_study_design(ctx, n_samples, n_seg):
    W, H = style.figsize("double")[0] / style.MM, 74.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ytop, ybot = 63.5, 9.0
    cols = dict(A=(1.0, 33.0), B=(39.0, 35.0), C=(80.0, 59.0), D=(145.0, 44.0))
    heads = dict(A="Inputs", B="Regions of interest", C="Factorial segmentation", D="Analysis modules")
    for k, (x, w) in cols.items():
        ax.text(x + w / 2, 70.0, heads[k], ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=style.INK)
    # --- A: inputs
    xa, wa = cols["A"]
    _box(ax, xa, 38.0, wa, ytop - 38.0, "Micro-CT slices",
         [f"{n_samples} × 2-D, 16-bit", "4 granular packs", "5 rock cores", "25 / 40 µm pixel"])
    _box(ax, xa, ybot, wa, 25.0, "Laboratory porosity",
         ["4 physical methods", "+ laboratory µCT", "(7 of 9 samples)"])
    # --- B: ROI + normalisation
    xb, wb = cols["B"]
    _box(ax, xb, 38.0, wb, ytop - 38.0, "2 ROI designs",
         ["largest usable disc", "(primary)", "central 500 × 500 px", "(original design)"])
    _box(ax, xb, ybot + 9.5, wb, 15.5, "Normalisation",
         ["ROI percentiles", "0.1–99.9 % → [0, 1]"])
    # --- C: factorial segmentation
    xc, wc = cols["C"]
    ax.add_patch(FancyBboxPatch((xc, ybot), wc, ytop - ybot, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#f3f2ef", ec=style.INK2, lw=0.6))
    ax.text(xc + wc / 2, ytop - 2.6, f"{len(ctx.methods)} thresholding algorithms", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=style.INK)
    chip_w, chip_h = 17.5, 4.2
    fam_rows = [("Clustering", ["Otsu", "Isodata"]), ("Entropy", ["Li", "Yen"]),
                ("Attribute / shape", ["Mean", "Triangle"]), ("Local adaptive", ["Sauvola", "Niblack"])]
    y = ytop - 9.2
    for i, (fam, ms) in enumerate(fam_rows):
        if i == 3:
            y -= 1.4
            ax.plot([xc + 2, xc + wc - 2], [y + chip_h + 1.0, y + chip_h + 1.0], color=style.MUTED, lw=0.4)
        ax.text(xc + 2.0, y + chip_h / 2, fam, ha="left", va="center", fontsize=6.8, color=style.INK2,
                style="italic")
        for j, m in enumerate(ms):
            _chip(ax, xc + 21.5 + j * (chip_w + 1.5), y, chip_w, chip_h, m, style.METHOD_COLORS[m],
                  style.METHOD_MARKERS[m])
        y -= chip_h + 1.4
    ax.text(xc + 2.0, y + 4.2, "", fontsize=1)
    y -= 0.6
    ax.text(xc + wc / 2, y + 1.6, "×", ha="center", va="center", fontsize=9, color=style.INK)
    y -= 2.4
    ax.text(xc + wc / 2, y, "4 post-processing levels", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=style.INK)
    y -= 6.6
    pw_ = (wc - 4 - 3 * 1.2) / 4
    for j, p in enumerate(ctx.posts):
        lab = {"none": "None", "median": "Median", "open_close": "Open–close", "skeleton": "Skeleton"}[p]
        _chip(ax, xc + 2 + j * (pw_ + 1.2), y, pw_, chip_h, lab, style.POST_COLORS[p], style.POST_MARKERS[p],
              size=5.9)
    yb = y - 1.3
    x1 = xc + 2 + 3 * (pw_ + 1.2) - 1.2
    ax.plot([xc + 2, xc + 2, x1, x1], [yb + 0.7, yb, yb, yb + 0.7], color=style.INK2, lw=0.5)
    ax.text((xc + 2 + x1) / 2, yb - 1.7, "area-preserving → porosity", ha="center", va="center", fontsize=6.2,
            color=style.INK2)
    xs0 = xc + 2 + 3 * (pw_ + 1.2)
    ax.plot([xs0, xs0, xs0 + pw_, xs0 + pw_], [yb + 0.7, yb, yb, yb + 0.7], color=style.INK2, lw=0.5)
    ax.text(xs0 + pw_ / 2, yb - 1.7, "topology only", ha="center", va="center", fontsize=6.2, color=style.INK2)
    ax.text(xc + wc / 2, ybot + 2.6,
            f"{n_samples} × 2 × {len(ctx.methods)} × {len(ctx.posts)} = {n_seg} segmentations",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=style.INK)
    # --- D: analysis modules
    xd, wd = cols["D"]
    mods = ["Variance decomposition", "Spatial statistics / REV", "Threshold sensitivity (TSM, PSI)",
            "Artefact perturbation", "Morphology and topology", "Synthetic ground truth",
            "ML baseline (RF, GMM)", "Laboratory agreement"]
    gap = 1.25
    bh = (ytop - ybot - gap * (len(mods) - 1)) / len(mods)
    centres = []
    for i, mname in enumerate(mods):
        yy = ytop - (i + 1) * bh - i * gap
        fc = "#e9eef6" if mname != "Laboratory agreement" else "#f3f2ef"
        ax.add_patch(FancyBboxPatch((xd, yy), wd, bh, boxstyle="round,pad=0,rounding_size=1.0", fc=fc,
                                    ec=style.INK2, lw=0.6))
        ax.text(xd + wd / 2, yy + bh / 2, mname, ha="center", va="center", fontsize=6.8, color=style.INK)
        centres.append(yy + bh / 2)
    # --- arrows
    ym = (38.0 + ytop) / 2
    _arrow(ax, (xa + wa + 0.3, ym), (xb - 0.4, ym))
    _arrow(ax, (xb + wb / 2, 38.0 - 0.2), (xb + wb / 2, ybot + 9.5 + 15.5 + 0.3))
    _arrow(ax, (xb + wb + 0.3, (ybot + 9.5 + ybot + 25.0) / 2), (xc - 0.4, (ybot + 9.5 + ybot + 25.0) / 2))
    # bracket into the analysis column
    xbr = xd - 2.2
    ax.plot([xbr, xbr], [centres[0], centres[-2]], color=style.INK2, lw=0.6)
    for yc in centres[:-1]:
        ax.plot([xbr, xd - 0.3], [yc, yc], color=style.INK2, lw=0.6)
    _arrow(ax, (xc + wc + 0.3, (centres[0] + centres[-2]) / 2), (xbr, (centres[0] + centres[-2]) / 2))
    _arrow(ax, (xc + wc + 0.3, centres[-1]), (xd - 0.4, centres[-1]))
    # laboratory porosity -> laboratory agreement (routed below the columns)
    ylab = 4.0
    ax.plot([xa + wa / 2, xa + wa / 2], [ybot - 0.2, ylab], color=style.INK2, lw=0.8)
    ax.plot([xa + wa / 2, xbr - 1.2], [ylab, ylab], color=style.INK2, lw=0.8)
    ax.plot([xbr - 1.2, xbr - 1.2], [ylab, centres[-1] - 1.6], color=style.INK2, lw=0.8)
    _arrow(ax, (xbr - 1.2, centres[-1] - 1.6), (xd - 0.4, centres[-1] - 1.6))
    ax.text((xc + xc + wc) / 2, ylab + 1.5, "reference values (n = 7 images)", ha="center", va="center",
            fontsize=6.2, color=style.MUTED)
    return fig


def _fig_histograms(ctx, per, sids, rec_by):
    W = style.figsize("double")[0] / style.MM
    ncol = 3
    nrow = int(np.ceil(len(sids) / ncol))
    left, right, gap_x = 15.0, 1.5, 4.0
    pw = (W - left - right - (ncol - 1) * gap_x) / ncol
    strip_h, hist_h, title_h, gap_y, bottom = 15.0, 25.0, 4.0, 7.0, 25.0
    cell = title_h + strip_h + hist_h
    H = bottom + nrow * cell + (nrow - 1) * gap_y + 1.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    nm = len(ctx.methods)
    ymin, ymax = 1e-3, 30.0
    for k, sid in enumerate(sids):
        r, c = divmod(k, ncol)
        x = left + c * (pw + gap_x)
        yb = bottom + (nrow - 1 - r) * (cell + gap_y)
        axh = _mm_axes(fig, W, H, x, yb, pw, hist_h)
        axs = _mm_axes(fig, W, H, x, yb + hist_h, pw, strip_h, sharex=axh)
        d = per[sid]
        counts, edges = d["hist"]
        cen = (edges[:-1] + edges[1:]) / 2
        frac = 100.0 * counts / counts.sum()
        fpos = np.where(frac > 0, frac, np.nan)
        axh.fill_between(cen, ymin, np.nan_to_num(fpos, nan=ymin), step="mid", color=style.REF_BAND, lw=0)
        axh.step(cen, fpos, where="mid", color=style.INK2, lw=0.6)
        rec = rec_by[sid]
        bw = edges[1] - edges[0]
        for j in (1, 2):
            gy = 100 * gauss_bin_fraction(cen, bw, rec[f"gmm_mu{j}"], rec[f"gmm_sd{j}"], rec[f"gmm_w{j}"])
            gy = np.where(gy > ymin, gy, np.nan)
            axh.plot(cen, gy, ls=":", color=style.INK, lw=0.8)
        for i, mk in enumerate(d["marks"]):
            m = mk["method"]
            col = style.METHOD_COLORS[m]
            if mk["family"] == "global":
                axh.axvline(mk["t_mean"], color=col, lw=0.75, ls="-", zorder=3)
                axs.plot([mk["t_mean"]], [i], ls="none", marker=style.METHOD_MARKERS[m], color=col, ms=3.4,
                         zorder=3)
            else:
                axh.axvline(mk["t_mean"], color=col, lw=0.75, ls=(0, (2.5, 1.5)), zorder=3)
                axs.plot([mk["t_p05"], mk["t_p95"]], [i, i], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
                for xe in (mk["t_p05"], mk["t_p95"]):
                    axs.plot([xe, xe], [i - 0.28, i + 0.28], color=col, lw=0.8, zorder=2)
                axs.plot([mk["t_mean"]], [i], ls="none", marker=style.METHOD_MARKERS[m], color=col, ms=3.4,
                         zorder=3)
        axh.set_yscale("log")
        axh.set_ylim(ymin, ymax)
        axh.set_xlim(0, 1)
        axh.set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
        axh.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        if c == 0:
            axh.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
            axh.set_ylabel("Pixels per bin (%)")
        else:
            axh.set_yticklabels([])
        axh.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        if r == nrow - 1 or k + ncol >= len(sids):
            axh.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
            axh.set_xlabel("Normalised grey level")
        else:
            axh.set_xticklabels([])
        axs.set_ylim(nm - 0.4, -0.6)
        for i in range(nm):
            axs.axhline(i, color=style.GRID, lw=0.4, zorder=0)
        axs.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for s in ("top", "right", "bottom"):
            axs.spines[s].set_visible(False)
        axs.set_yticks(range(nm))
        if c == 0:
            axs.set_yticklabels(ctx.methods, fontsize=6.0)
            axs.tick_params(axis="y", length=0, pad=1.5)
        else:
            axs.set_yticklabels([])
            axs.tick_params(axis="y", length=0)
        axs.spines["left"].set_visible(False)
        flag = "*" if rec["bc"] > BC_CRIT else ""
        flagd = "*" if rec["ashman_d"] > D_CRIT else ""
        axs.set_title(ctx.label(sid), loc="left", pad=2.0, fontsize=7.5, fontweight="bold")
        axs.set_title(f"BC = {rec['bc']:.2f}{flag}   D = {rec['ashman_d']:.1f}{flagd}", loc="right", pad=2.0,
                      fontsize=6.8, color=style.INK2)
    hs = style.method_handles(ctx.methods, ms=3.6)
    hs += [Line2D([], [], color=style.INK2, lw=0.8, label="Global threshold"),
           Line2D([], [], color=style.INK2, lw=0.8, ls=(0, (2.5, 1.5)), label="Local threshold (mean)"),
           Line2D([], [], color=style.INK2, lw=1.0, marker="|", ms=4, label="Local 5–95 % range"),
           Line2D([], [], color=style.INK, lw=0.8, ls=":", label="GMM component")]
    fig.legend(handles=hs, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2,
               handletextpad=0.4)
    return fig


def _method_header(ax, m, size=7.0):
    ax.set_title(m, fontsize=size, pad=2.0)
    ax.plot([0.5], [1.0], transform=ax.transAxes, ls="none")  # keeps layout stable


def _put_marker_title(ax, text, color, marker, size=7.0, y=1.035):
    t = ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=size, color=style.INK)
    ax.plot([0.5], [y], transform=ax.transAxes, ls="none")
    return t


def _fig_gallery(ctx, per, sids):
    W = style.figsize("double")[0] / style.MM
    ncols = 1 + len(ctx.methods)
    left, gap = 11.0, 0.8
    pw = (W - left - (ncols - 1) * gap) / ncols
    head, gap_y, bottom = 6.0, 1.2, 0.5
    H = head + len(sids) * pw + (len(sids) - 1) * gap_y + bottom
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    for r, sid in enumerate(sids):
        d = per[sid]
        vmin, vmax = d["grey_lim"]
        y = bottom + (len(sids) - 1 - r) * (pw + gap_y)
        for c in range(ncols):
            ax = _mm_axes(fig, W, H, left + c * (pw + gap), y, pw, pw)
            if c == 0:
                ax.imshow(overlay_rgb(d["crops"]["grey"], None, vmin, vmax), interpolation="antialiased")
                _image_axis(ax)
                style.scalebar(ax, ctx.pixel_size_um(sid), color="white")
                ax.text(-0.30, 0.5, ctx.label(sid), transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=7.5, fontweight="bold")
                ax.text(-0.13, 0.5, GALLERY_ROLE.get(sid, _group_label(ctx, sid).lower()),
                        transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.0,
                        color=style.INK2)
                if r == 0:
                    ax.set_title("Grey (raw)", fontsize=7.0, pad=2.5)
                continue
            m = ctx.methods[c - 1]
            ax.imshow(overlay_rgb(d["crops"]["grey"], d["crops"][(m, "none")], vmin, vmax),
                      interpolation="antialiased")
            _image_axis(ax)
            phi = _seg_value(ctx, sid, m, "none")
            ax.text(0.04, 0.96, f"{100 * phi:.1f} %", transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
                    color=style.INK, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(m, fontsize=7.0, pad=2.5)
                ax.plot([0.5 - 0.11 - 0.028 * len(m)], [1.075], transform=ax.transAxes, ls="none",
                        marker=style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m], ms=3.4, clip_on=False)
    return fig


def _fig_grid(ctx, sid, d):
    W = style.figsize("single")[0] / style.MM
    nr, nc = len(ctx.methods), len(ctx.posts)
    left, gap, head, bottom = 13.0, 0.7, 5.0, 0.5
    pw = (W - left - (nc - 1) * gap) / nc
    H = head + nr * pw + (nr - 1) * gap + bottom
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    vmin, vmax = d["grey_lim"]
    for i, m in enumerate(ctx.methods):
        for j, p in enumerate(ctx.posts):
            ax = _mm_axes(fig, W, H, left + j * (pw + gap), bottom + (nr - 1 - i) * (pw + gap), pw, pw)
            pore = d["crops"][(m, p)]
            if p == "skeleton":   # 1-px skeleton dilated to 3 px for visibility (display only)
                pore = ndi.binary_dilation(pore, np.ones((3, 3), bool))
            ax.imshow(overlay_rgb(d["crops"]["grey"], pore, vmin, vmax, alpha=0.65 if p != "skeleton" else 0.85),
                      interpolation="antialiased")
            _image_axis(ax)
            val = _seg_value(ctx, sid, m, p)
            txt = f"{100 * val:.1f} %" if p != "skeleton" else f"sk {100 * val:.2f} %"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=style.INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(style.POST_LABELS[p], fontsize=6.8, pad=2.0)
            if j == 0:
                ax.text(-0.07, 0.5, m, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=6.8)
                ax.plot([-0.2], [0.5], transform=ax.transAxes, ls="none", marker=style.METHOD_MARKERS[m],
                        color=style.METHOD_COLORS[m], ms=3.4, clip_on=False)
            if i == nr - 1 and j == 0:
                style.scalebar(ax, ctx.pixel_size_um(sid), color="white", loc="lower left")
    return fig


def _fig_similarity(ctx, M, methods):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    k = len(methods)
    dist = squareform(np.clip(1.0 - M, 0, None), checks=False)
    Z = linkage(dist, method="average")
    W = style.figsize("onehalf")[0] / style.MM
    H = 66.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    cell = 5.6
    hm_w = hm_h = k * cell
    x_d, w_d = 4.0, 26.0
    x_h = x_d + w_d + 1.0
    y_h = 13.0
    axd = _mm_axes(fig, W, H, x_d, y_h, w_d, hm_h)
    dn = dendrogram(Z, orientation="left", ax=axd, no_labels=True, color_threshold=0,
                    above_threshold_color=style.INK2, link_color_func=lambda _: style.INK2)
    order = dn["leaves"]                      # bottom -> top
    axd.set_ylim(0, 10 * k)
    for s in ("top", "right", "left"):
        axd.spines[s].set_visible(False)
    axd.set_yticks([])
    axd.set_xlabel("1 − mean Dice", labelpad=2)
    axd.invert_xaxis() if axd.get_xlim()[0] < axd.get_xlim()[1] else None
    axd.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    style.panel_label(axd, "a", x=0.0, y=1.0 + 2.0 / hm_h)
    axh = _mm_axes(fig, W, H, x_h, y_h, hm_w, hm_h)
    Mo = M[np.ix_(order, order)]
    show = Mo.copy()
    np.fill_diagonal(show, np.nan)
    off = Mo[~np.eye(k, dtype=bool)]
    vmin = max(0.0, np.floor(off.min() * 10) / 10)
    im = axh.imshow(show, cmap=style.CMAP_SEQ, vmin=vmin, vmax=1.0, origin="lower", aspect="equal")
    for i in range(k):
        for j in range(k):
            if i == j:
                axh.text(j, i, "–", ha="center", va="center", fontsize=6, color=style.MUTED)
                continue
            v = Mo[i, j]
            frac = (v - vmin) / (1 - vmin)
            axh.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.00", ha="center", va="center", fontsize=5.8,
                     color="white" if frac > 0.55 else style.INK)
    names = [methods[o] for o in order]
    axh.set_xticks(range(k))
    axh.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.8)
    axh.set_yticks(range(k))
    axh.set_yticklabels(names, fontsize=6.8)
    axh.yaxis.tick_right()
    axh.tick_params(length=0, pad=6.0)
    for s in axh.spines.values():
        s.set_visible(False)
    for t, nmth in enumerate(names):   # family markers next to the labels (circle = global, square = local)
        axh.plot([k - 0.5 + 0.35], [t], ls="none", marker=style.METHOD_MARKERS[nmth],
                 color=style.METHOD_COLORS[nmth], ms=3.2, clip_on=False)
        axh.plot([t], [-0.5 - 0.35], ls="none", marker=style.METHOD_MARKERS[nmth],
                 color=style.METHOD_COLORS[nmth], ms=3.2, clip_on=False)
    axh.set_xlim(-0.5, k - 0.5)
    axh.set_ylim(-0.5, k - 0.5)
    style.panel_label(axh, "b", x=0.0, y=1.0 + 2.0 / hm_h)
    cax = _mm_axes(fig, W, H, x_h + hm_w + 19.0, y_h, 2.5, hm_h)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean Dice coefficient", labelpad=3)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.5, length=2)
    hs = [Line2D([], [], ls="none", marker="o", color=style.INK2, ms=3.2, label="Global"),
          Line2D([], [], ls="none", marker="s", color=style.INK2, ms=3.2, label="Local adaptive")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=((x_h + hm_w + 27.0) / W, y_h / H), ncol=1,
               handletextpad=0.3)
    return fig, order


def _fig_roi_effect(ctx, roi_df, sids, summary_by_sample):
    W = style.figsize("double")[0] / style.MM
    ncol = 3
    nrow = int(np.ceil(len(sids) / ncol))
    left, right, gap_x = 13.0, 2.0, 5.0
    pw = (W - left - right - (ncol - 1) * gap_x) / ncol
    ph, gap_y, bottom, top = 27.0, 9.0, 16.0, 4.0
    H = bottom + nrow * ph + (nrow - 1) * gap_y + top
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    nm = len(ctx.methods)
    for k, sid in enumerate(sids):
        r, c = divmod(k, ncol)
        ax = _mm_axes(fig, W, H, left + c * (pw + gap_x), bottom + (nrow - 1 - r) * (ph + gap_y), pw, ph)
        sub = roi_df[roi_df["sample_id"] == sid].set_index("method")
        for i, m in enumerate(ctx.methods):
            if m not in sub.index:
                continue
            a, b = 100 * sub.at[m, "phi_disk"], 100 * sub.at[m, "phi_center500"]
            col = style.METHOD_COLORS[m]
            ax.plot([a, b], [i, i], color=style.MUTED, lw=0.8, zorder=1)
            ax.plot([a], [i], ls="none", marker=style.METHOD_MARKERS[m], color=col, ms=3.8, zorder=3)
            ax.plot([b], [i], ls="none", marker=style.METHOD_MARKERS[m], mfc="white", mec=col, mew=0.9, ms=3.8,
                    zorder=3)
        ax.set_ylim(nm - 0.4, -0.6)
        ax.set_yticks(range(nm))
        ax.set_yticklabels(ctx.methods if c == 0 else [], fontsize=6.3)
        ax.tick_params(axis="y", length=0)
        lo = 100 * min(sub["phi_disk"].min(), sub["phi_center500"].min())
        hi = 100 * max(sub["phi_disk"].max(), sub["phi_center500"].max())
        pad = max(1.0, 0.06 * (hi - lo))
        ax.set_xlim(max(0, lo - pad), min(100, hi + pad))
        style.hgrid(ax, "x")
        ax.set_xlabel("Porosity (%)", labelpad=1.5)
        s = summary_by_sample[sid]
        ax.set_title(ctx.label(sid), loc="left", fontsize=7.5, fontweight="bold", pad=2.0)
        ax.set_title(f"median |Δ| = {s:.1f} pp", loc="right", fontsize=6.8, color=style.INK2, pad=2.0)
    hs = [Line2D([], [], ls="none", marker="o", color=style.INK2, ms=3.8, label="Largest usable disc (primary)"),
          Line2D([], [], ls="none", marker="o", mfc="white", mec=style.INK2, ms=3.8,
                 label="Central 500 × 500 px")]
    hs += style.method_handles(ctx.methods, ms=3.6)
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
               handletextpad=0.4)
    return fig


def _fig_bimod_disp(ctx, bim, assoc):
    W = style.figsize("onehalf")[0] / style.MM
    H = 58.0
    fig = plt.figure(figsize=(W * style.MM, H * style.MM))
    pw, ph, left, gap, bottom = 54.0, 40.0, 13.0, 15.0, 13.0
    specs = [("bc", "Sarle's bimodality coefficient BC", BC_CRIT, "BC = 5/9"),
             ("ashman_d", "Ashman's D (2-component GMM)", D_CRIT, "D = 2")]
    for k, (col, xl, crit, critlab) in enumerate(specs):
        ax = _mm_axes(fig, W, H, left + k * (pw + gap), bottom, pw, ph)
        for _, r in bim.iterrows():
            ax.plot(r[col], 100 * r["intermethod_sd"], ls="none", marker=style.GROUP_MARKERS[r["group"]],
                    mfc="white" if r["group"] == "consolidated" else style.INK2, mec=style.INK2, ms=4.2)
            ax.annotate(r["label"], (r[col], 100 * r["intermethod_sd"]), xytext=(3, 1.5),
                        textcoords="offset points", fontsize=6.0, color=style.INK2)
        ax.axvline(crit, color=style.MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.text(crit, 1.0, f" {critlab}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=6.0,
                color=style.MUTED)
        a = assoc[(assoc["predictor"] == col) & (assoc["response"] == "intermethod_sd") &
                  (assoc["analysis"] == "all_methods")]
        if len(a):
            a = a.iloc[0]
            ax.text(0.98, 0.97, f"ρ = {a['rho']:.2f} [{a['boot_ci_lo']:.2f}, {a['boot_ci_hi']:.2f}]\n"
                                f"exact p = {a['p_exact']:.3f} (Holm {a['p_holm']:.3f})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=style.INK)
        ax.set_xlabel(xl)
        if k == 0:
            ax.set_ylabel("Inter-method SD of porosity (pp)")
        style.hgrid(ax, "y")
        style.panel_label(ax, "ab"[k])
        ax.margins(x=0.12, y=0.12)
    hs = [Line2D([], [], ls="none", marker=style.GROUP_MARKERS[g], mfc=f, mec=style.INK2, ms=4.2,
                 label=style.GROUP_LABELS[g]) for g, f in (("granular_pack", style.INK2), ("consolidated", "white"))]
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    return fig


# ============================================================================
# the stage
# ============================================================================
def run(ctx) -> None:
    sids = ctx.sample_ids
    roi = ctx.primary_roi
    level = float(ctx.cfg["statistics"]["ci_level"])
    per: dict[str, dict] = {}
    for sid in sids:
        per[sid] = _process_sample(ctx, sid)
        _release(ctx, sid)
        ctx.log("%s: BC=%.3f D=%.2f window=(%d,%d) px", sid, per[sid]["rec"]["bc"], per[sid]["rec"]["ashman_d"],
                per[sid]["window"]["x0_px"], per[sid]["window"]["y0_px"])

    # ------------------------------------------------------------- tidy data
    seg = ctx.seg
    base = seg[(seg["roi"] == roi) & (seg["post"] == "none")]
    disp_rows = []
    for sid in sids:
        b = base[base["sample_id"] == sid].set_index("method")["porosity"].reindex(ctx.methods)
        phi = b.to_numpy(float)
        strict = failure_flags(phi)
        lenient = failure_flags(phi, lo=COLLAPSE_LO)
        keep_s, keep_l = phi[~strict], phi[~lenient]
        disp_rows.append(dict(
            sample_id=sid, label=ctx.label(sid), group=ctx.group(sid),
            intermethod_mean=float(np.mean(phi)), intermethod_median=float(np.median(phi)),
            intermethod_sd=float(np.std(phi, ddof=1)), intermethod_range=float(np.ptp(phi)),
            intermethod_iqr=float(np.subtract(*np.percentile(phi, [75, 25]))),
            phi_min=float(phi.min()), phi_max=float(phi.max()),
            method_min=ctx.methods[int(np.argmin(phi))], method_max=ctx.methods[int(np.argmax(phi))],
            n_failed_strict=int(strict.sum()), failed_strict=",".join(np.array(ctx.methods)[strict]),
            n_failed_lenient=int(lenient.sum()), failed_lenient=",".join(np.array(ctx.methods)[lenient]),
            intermethod_sd_excl_strict=float(np.std(keep_s, ddof=1)) if keep_s.size > 1 else np.nan,
            intermethod_range_excl_strict=float(np.ptp(keep_s)) if keep_s.size else np.nan,
            intermethod_sd_excl_lenient=float(np.std(keep_l, ddof=1)) if keep_l.size > 1 else np.nan,
            intermethod_range_excl_lenient=float(np.ptp(keep_l)) if keep_l.size else np.nan,
            otsu_isodata_same=bool(abs(b["Otsu"] - b["Isodata"]) < 1e-4)))
    disp = pd.DataFrame(disp_rows)
    ctx.save_data(disp, "intermethod_dispersion")
    bim = pd.DataFrame([per[s]["rec"] for s in sids]).merge(
        disp[["sample_id", "intermethod_sd", "intermethod_range", "intermethod_sd_excl_strict",
              "intermethod_sd_excl_lenient", "intermethod_range_excl_lenient", "n_failed_strict",
              "n_failed_lenient"]], on="sample_id")
    ctx.save_data(bim, "bimodality")
    hist_rows = []
    for sid in sids:
        counts, edges = per[sid]["hist"]
        hist_rows.append(pd.DataFrame(dict(sample_id=sid, label=ctx.label(sid), bin_lo=edges[:-1], bin_hi=edges[1:],
                                           count=counts, fraction=counts / counts.sum())))
    ctx.save_data(pd.concat(hist_rows, ignore_index=True), "histograms")
    marks = pd.DataFrame([mk for s in sids for mk in per[s]["marks"]])
    ctx.save_data(marks, "threshold_marks")
    windows = pd.DataFrame([per[s]["window"] for s in sids])
    ctx.save_data(windows, "zoom_windows")

    # ------------------------------------------- bimodality vs dispersion tests
    assoc_rows = []
    n = len(sids)
    if n >= 5:
        for analysis, resp_cols in (("all_methods", ("intermethod_sd", "intermethod_range")),
                                    ("excl_lenient_failures", ("intermethod_sd_excl_lenient",
                                                               "intermethod_range_excl_lenient"))):
            fam = []
            for pred in ("bc", "ashman_d"):
                for resp in resp_cols:
                    x, y = bim[pred].to_numpy(float), bim[resp].to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    pr = spearman_permutation(x[ok], y[ok], rng=ctx.rng("descriptive", "perm", pred, resp))
                    bt = spearman_boot_ci(x[ok], y[ok], ctx.n_boot, ctx.rng("descriptive", "boot", pred, resp), level)
                    bw = bonett_wright_ci(pr["rho"], int(ok.sum()), level)
                    fam.append(dict(analysis=analysis, predictor=pred,
                                    response=resp.replace("_excl_lenient", ""), n=int(ok.sum()), rho=pr["rho"],
                                    p_exact=pr["p_two_sided"], p_less=pr["p_less"], exact=pr["exact"],
                                    n_perm=pr["n_perm"], boot_ci_lo=bt["ci_lo"], boot_ci_hi=bt["ci_hi"],
                                    boot_n_valid=bt["n_valid"], bw_ci_lo=bw[0], bw_ci_hi=bw[1]))
            ph = holm([f["p_exact"] for f in fam])
            for f, p in zip(fam, ph):
                f["p_holm"] = float(p)
                f["holm_family"] = f"{analysis}: (BC, D) x (SD, range), m = {len(fam)}"
            assoc_rows += fam
    assoc = pd.DataFrame(assoc_rows)
    if len(assoc):
        ctx.save_data(assoc, "bimodality_association")

    # --------------------------------------------------------------- overlap
    dice = pd.concat([per[s]["dice"] for s in sids], ignore_index=True)
    ctx.save_data(dice[["sample_id", "label", "post", "method_a", "method_b", "dice", "jaccard",
                        "n_a", "n_b", "n_inter"]], "dice_pairs")
    d0 = dice[dice["post"] == "none"]
    M = overlap_matrix(d0, ctx.methods, "dice")
    MJ = overlap_matrix(d0, ctx.methods, "jaccard")
    mm = [dict(method_a=a, method_b=b, mean_dice=M[i, j], mean_jaccard=MJ[i, j],
               min_dice=float(d0[((d0.method_a == a) & (d0.method_b == b)) | ((d0.method_a == b) & (d0.method_b == a))]
                              ["dice"].min()) if a != b else 1.0,
               n_samples=n)
          for i, a in enumerate(ctx.methods) for j, b in enumerate(ctx.methods)]
    ctx.save_data(pd.DataFrame(mm), "dice_mean")
    fam_of = {m: METHOD_INFO[m]["family"] for m in ctx.methods}
    d0 = d0.assign(pair_type=[("local" if fam_of[a] == "local" else "global") + "-" +
                              ("local" if fam_of[b] == "local" else "global")
                              for a, b in zip(d0["method_a"], d0["method_b"])])
    d0 = d0.assign(pair_type=d0["pair_type"].replace({"local-global": "global-local"}))
    per_sample_type = d0.groupby(["sample_id", "pair_type"])["dice"].mean().unstack()
    fam_rows = []
    for t in ("global-global", "global-local", "local-local"):
        if t not in per_sample_type:
            continue
        est, lo, hi = bootstrap_ci(per_sample_type[t].to_numpy(), np.mean, ctx.n_boot,
                                   ctx.rng("descriptive", "dice_type", t), level)
        fam_rows.append(dict(pair_type=t, n_pairs=int((d0["pair_type"] == t).sum() / max(n, 1)), n_samples=n,
                             mean_dice=est, ci_lo=lo, ci_hi=hi,
                             min_sample=float(per_sample_type[t].min()), max_sample=float(per_sample_type[t].max())))
    fam_df = pd.DataFrame(fam_rows)
    ctx.save_data(fam_df, "dice_family_summary")

    # ------------------------------------------------------------ ROI design
    have_c = "center500" in ctx.cfg["roi"]["designs"]
    roi_rows = []
    if have_c:
        c5 = seg[(seg["roi"] == "center500") & (seg["post"] == "none")]
        for sid in sids:
            for m in ctx.methods:
                a = base.loc[(base.sample_id == sid) & (base.method == m), "porosity"]
                b = c5.loc[(c5.sample_id == sid) & (c5.method == m), "porosity"]
                if len(a) and len(b):
                    roi_rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), method=m,
                                         family=fam_of[m], phi_disk=float(a.iloc[0]),
                                         phi_center500=float(b.iloc[0]),
                                         delta_pp=100 * float(b.iloc[0] - a.iloc[0])))
    roi_df = pd.DataFrame(roi_rows)
    roi_sum_rows = []
    s_by = {}
    if len(roi_df):
        roi_df = roi_df.assign(abs_delta_pp=roi_df["delta_pp"].abs())
        ctx.save_data(roi_df, "roi_design_effect")
        ps = roi_df.groupby(["sample_id", "label", "group"], sort=False).agg(
            median_abs_delta_pp=("abs_delta_pp", "median"), mean_delta_pp=("delta_pp", "mean"),
            max_abs_delta_pp=("abs_delta_pp", "max")).reset_index()
        ps = ps.merge(roi_df[roi_df.method == "Otsu"][["sample_id", "delta_pp"]].rename(
            columns={"delta_pp": "otsu_delta_pp"}), on="sample_id", how="left")
        s_by = dict(zip(ps["sample_id"], ps["median_abs_delta_pp"]))
        for grp, sub in [("all", ps)] + [(g, ps[ps.group == g]) for g in ("granular_pack", "consolidated")]:
            if not len(sub):
                continue
            est, lo, hi = bootstrap_ci(sub["median_abs_delta_pp"], np.median, ctx.n_boot,
                                       ctx.rng("descriptive", "roi", grp), level)
            e2, lo2, hi2 = bootstrap_ci(sub["mean_delta_pp"], np.median, ctx.n_boot,
                                        ctx.rng("descriptive", "roi_signed", grp), level)
            e3, lo3, hi3 = bootstrap_ci(sub["otsu_delta_pp"].abs(), np.median, ctx.n_boot,
                                        ctx.rng("descriptive", "roi_otsu", grp), level)
            roi_sum_rows.append(dict(group=grp, n_samples=len(sub), median_abs_delta_pp=est, ci_lo=lo, ci_hi=hi,
                                     min_sample=float(sub["median_abs_delta_pp"].min()),
                                     max_sample=float(sub["median_abs_delta_pp"].max()),
                                     median_signed_delta_pp=e2, signed_ci_lo=lo2, signed_ci_hi=hi2,
                                     otsu_median_abs_delta_pp=e3, otsu_ci_lo=lo3, otsu_ci_hi=hi3))
        roi_sum = pd.DataFrame(roi_sum_rows)
        ctx.save_data(roi_sum, "roi_design_summary")
        ctx.save_data(ps, "roi_design_by_sample")

    # =============================================================== figures
    W = style.figsize("double")[0]
    fig = _fig_overview(ctx, per, sids)
    ctx.save_fig(fig, "samples_overview",
                 f"The {n} micro-CT slices analysed (full field of view; grey levels linearly stretched between the "
                 "0.5 and 99.5 percentiles of the primary ROI, so air and the holder may saturate). Rows 1–2: four "
                 "packs of spheroidal 3–4 mm grains in a cylindrical holder (CF4, CF5, CF7, CF10) and five "
                 "consolidated rock cores (VN01, VN02, VN04, TQ06, GT01). Solid outline: largest usable disc "
                 "inside the holder wall or the core (primary ROI); dashed: the central 500 × 500 px square of "
                 "the original workflow (secondary ROI, centred on the image, not on the sample); dotted: the "
                 f"{WINDOW_MM:g} × {WINDOW_MM:g} mm zoom window used in the segmentation gallery and the "
                 "supplementary segmentation grids (position chosen automatically, see Methods). Pixel sizes "
                 "(25 or 40 µm) were inferred from TIFF tags / file names and are pending confirmation. Scale "
                 "bars in physical units.", kind="main")
    fig = _fig_study_design(ctx, n, int(len(seg)))
    ctx.save_fig(fig, "study_design",
                 f"Study design. Each of the {n} 2-D micro-CT slices is analysed in two region-of-interest designs "
                 "and normalised with robust ROI percentiles; eight thresholding algorithms (six global, grouped "
                 "by family following Sezgin & Sankur 2004, and two local adaptive) are crossed with four "
                 "binary post-processing levels applied to the pore phase, giving "
                 f"{n} × 2 × {len(ctx.methods)} × {len(ctx.posts)} = {len(seg)} segmentations. Three "
                 "post-processing levels preserve area and yield porosity; the medial-axis skeleton is used only "
                 "as a topology descriptor. Laboratory porosity (mean of four physical methods; single value of "
                 "unstated method for VN01, VN02, VN04; the laboratory's own µCT value analysed separately) "
                 "enters only the agreement analysis. Method colours and markers (circle = global, square = "
                 "local) are used consistently in all figures.", kind="main")

    rec_by = {r["sample_id"]: r for _, r in bim.iterrows()}
    fig = _fig_histograms(ctx, per, sids, rec_by)
    ctx.save_fig(fig, "histograms_thresholds",
                 f"Grey-level histograms of the primary ROI (largest usable disc, 2-px rim excluded; {NBINS} bins "
                 "of the normalised grey level, 0.1–99.9 % ROI percentiles mapped to 0–1) for the "
                 f"{n} samples, with the threshold of every algorithm. Counts are shown on a logarithmic axis "
                 "(percentage of ROI pixels per bin) so that the sparse pore tails of the consolidated cores remain "
                 "visible. Solid vertical lines: global thresholds; dashed: mean of the local threshold map "
                 "(Sauvola, Niblack). Upper strip of each panel: the same thresholds, one row per method "
                 "(markers), with the 5–95 % range of the local threshold maps (bars); coincident global "
                 "thresholds (e.g., Otsu and Isodata) are separated here. Dotted curves: the two components of a "
                 "Gaussian mixture fitted to the ROI pixels. BC: Sarle's finite-sample bimodality coefficient "
                 "(* BC > 5/9); D: Ashman's separation of the mixture components (* D > 2). Pores are dark "
                 "(I ≤ T). No post-processing involved.", kind="main")

    gal = [s for s in GALLERY_SAMPLES if s in per] or sids[:min(3, n)]
    fig = _fig_gallery(ctx, per, gal)
    wtxt = "; ".join(f"{ctx.label(s)}: x = {per[s]['window']['x0_mm']:.1f}–"
                     f"{per[s]['window']['x0_mm'] + per[s]['window']['size_mm']:.1f} mm, y = "
                     f"{per[s]['window']['y0_mm']:.1f}–{per[s]['window']['y0_mm'] + per[s]['window']['size_mm']:.1f} mm"
                     for s in gal)
    ctx.save_fig(fig, "segmentation_gallery",
                 f"Segmentations of the eight algorithms on identical {WINDOW_MM:g} × {WINDOW_MM:g} mm windows "
                 "of four representative samples: a granular pack with a bimodal histogram (CF5), the "
                 "heterogeneous core VN04, and two unimodal cores in which global methods fail in opposite "
                 "directions (VN02: entropy/shape methods collapse to a few % while clustering methods split "
                 "the noise; TQ06: Yen and Triangle label almost the whole slice as pore). Pores (I ≤ T, no "
                 "post-processing) painted in orange over the normalised grey image; left column: grey image "
                 "only (same contrast within a row). Numbers: porosity of the WHOLE primary ROI (largest usable "
                 "disc), not of the window. Windows (from the top-left image corner): " + wtxt + "; chosen "
                 "automatically to maximise inter-method disagreement and grey-level structure while lying "
                 "inside the ROI.", kind="main")
    for sid in sids:
        fig = _fig_grid(ctx, sid, per[sid])
        w_ = per[sid]["window"]
        ctx.save_fig(fig, f"segmentation_grid_{ctx.label(sid).lower()}",
                     f"Sample {ctx.label(sid)} ({_group_label(ctx, sid).lower()}): segmentation of the "
                     f"{WINDOW_MM:g} × {WINDOW_MM:g} mm window (x = {w_['x0_mm']:.1f}–{w_['x0_mm'] + w_['size_mm']:.1f} "
                     f"mm, y = {w_['y0_mm']:.1f}–{w_['y0_mm'] + w_['size_mm']:.1f} mm from the top-left image "
                     "corner) by the eight thresholding algorithms (rows) and the four post-processing levels "
                     "(columns; all applied to the pore phase). Pores in orange over the normalised grey image. "
                     "Numbers: porosity of the whole primary ROI (largest usable disc); for the skeleton column, "
                     "skeleton (medial-axis) density 'sk' — a topology descriptor, not a porosity; the 1-px "
                     "skeleton is drawn dilated to 3 px for visibility.", kind="supp")

    fig, order = _fig_similarity(ctx, M, ctx.methods)
    ctx.save_fig(fig, "method_similarity",
                 f"Pixel-wise agreement between thresholding algorithms. (a) Average-linkage (UPGMA) dendrogram on the "
                 f"distance 1 − mean Dice. (b) Dice coefficient for every pair of methods, averaged over the {n} samples "
                 "(primary ROI, no post-processing); rows and columns are ordered by the dendrogram. "
                 "Markers next to the labels: circle = global, square = local adaptive method (method colours as "
                 "in all figures). Per-sample values (also Jaccard, and median / open–close post-processing) are "
                 "provided as supplementary data (dice_pairs.csv). Means over samples hide sample-specific "
                 "failures; see the per-sample minima in dice_mean.csv.", kind="main")

    if len(roi_df):
        fig = _fig_roi_effect(ctx, roi_df, sids, s_by)
        ctx.save_fig(fig, "roi_design_effect",
                     "Effect of the region-of-interest design on porosity, for every sample and method (no "
                     "post-processing). Filled markers: largest usable disc (primary ROI); open markers: central "
                     "500 × 500 px square of the original workflow (12.5 × 12.5 mm at 25 µm, 20 × 20 mm at "
                     "40 µm). Panel annotation: median over the eight methods of the absolute difference "
                     "|φ(square) − φ(disc)| in percentage points (pp). Note the different x-axis ranges.",
                     kind="supp")
    if len(assoc):
        fig = _fig_bimod_disp(ctx, bim, assoc)
        ctx.save_fig(fig, "bimodality_vs_dispersion",
                     f"Histogram bimodality versus disagreement between thresholding algorithms (n = {n} images; "
                     "primary ROI, no post-processing). y: standard deviation of porosity across the eight methods "
                     "(pp). (a) Sarle's bimodality coefficient; (b) Ashman's D of a two-component Gaussian "
                     "mixture. Dashed lines: conventional bimodality criteria (BC = 5/9; D = 2). ρ: Spearman "
                     f"correlation with {int(100 * level)} % percentile-bootstrap CI (resampling images; "
                     f"{ctx.n_boot} replicates) and exact permutation p-value (all {math.factorial(n)} "
                     "re-pairings), Holm-adjusted over the four tests (BC, D) × (SD, range).", kind="supp")

    # ================================================================ tables
    ref = ctx.lab_ref.set_index("sample_id")
    trows = []
    for sid in sids:
        d = {"Sample": ctx.label(sid), "Group": _group_label(ctx, sid)}
        for m in ctx.methods:
            d[m] = 100 * _seg_value(ctx, sid, m, "none")
        dd = disp.set_index("sample_id").loc[sid]
        d["Range (pp)"] = 100 * dd["intermethod_range"]
        d["SD (pp)"] = 100 * dd["intermethod_sd"]
        if sid in ref.index:
            rr = ref.loc[sid]
            if rr["ref_kind"] == "multi-method":
                d["Lab φ (%)"] = f"{100 * rr['ref']:.1f} ± {100 * rr['ref_sd']:.1f}"
            else:
                d["Lab φ (%)"] = f"{100 * rr['ref']:.1f}†"
            d["Lab µCT (%)"] = f"{100 * rr['uct']:.1f}" if np.isfinite(rr["uct"]) else "–"
        else:
            d["Lab φ (%)"] = "–"
            d["Lab µCT (%)"] = "–"
        trows.append(d)
    ctx.save_table(pd.DataFrame(trows), "porosity_by_method",
                   "Image porosity (%) of every sample by thresholding algorithm (primary ROI = largest usable disc, "
                   "2-px rim excluded; no post-processing), inter-method range and standard deviation (percentage "
                   "points), and laboratory reference values. Lab φ: mean ± SD of four physical methods (direct, "
                   "gravimetric, indirect, gas porosimetry); Lab µCT: the laboratory's own tomographic estimate "
                   "(not used as reference).",
                   float_format="{:.1f}",
                   notes="† single laboratory value, method not stated in the source spreadsheet. – no laboratory "
                         "value. Values < 1 % or > 90 % are degenerate segmentations (all pixels or none). "
                         "Pixel size pending confirmation.")
    frows = []
    for sid in sids:
        for m in ctx.methods:
            frows.append({"Sample": ctx.label(sid), "Method": m, "Family": fam_of[m],
                          **{f"{style.POST_LABELS[p]} (%)": 100 * _seg_value(ctx, sid, m, p)
                             for p in postprocess.AREA_PRESERVING},
                          "Skeleton density (%)": 100 * _seg_value(ctx, sid, m, "skeleton")})
    ctx.save_table(pd.DataFrame(frows), "porosity_factorial",
                   "Full factorial results on the primary ROI (largest usable disc): porosity (%) for the three "
                   "area-preserving post-processing levels (none, 3×3 median, 3×3 opening followed by closing, all "
                   "applied to the pore phase) and skeleton (medial-axis) density (% of ROI pixels), which is a "
                   "topology descriptor and not a porosity.", kind="supp", float_format="{:.2f}")
    brows = []
    for _, r in bim.iterrows():
        brows.append({"Sample": r["label"], "Group": style.GROUP_LABELS.get(r["group"], r["group"]),
                      "Skewness": r["skewness"], "Excess kurtosis": r["excess_kurtosis"], "BC": r["bc"],
                      "Ashman D": r["ashman_d"], "μ1": r["gmm_mu1"], "σ1": r["gmm_sd1"], "w1": r["gmm_w1"],
                      "μ2": r["gmm_mu2"], "σ2": r["gmm_sd2"],
                      "Inter-method SD (pp)": 100 * r["intermethod_sd"],
                      "Inter-method range (pp)": 100 * r["intermethod_range"],
                      "Failed (strict)": str(int(r["n_failed_strict"])),
                      "Failed (lenient)": str(int(r["n_failed_lenient"]))})
    ctx.save_table(pd.DataFrame(brows), "histogram_bimodality",
                   "Histogram shape and inter-method dispersion per sample (primary ROI, normalised grey level, "
                   "values winsorised at the 0.1 and 99.9 % ROI percentiles as seen by the algorithms). Skewness "
                   "and excess kurtosis: adjusted sample estimates; BC: Sarle's finite-sample bimodality "
                   "coefficient (BC > 5/9 suggests bimodality); μ1, σ1, w1, μ2, σ2: means, SDs and weight of the "
                   "darker component of a two-component Gaussian mixture (grey-level units 0–1); Ashman D = "
                   "√2|μ1 − μ2|/√(σ1² + σ2²) (D > 2: clean separation). Inter-method SD and range of porosity "
                   "over the eight algorithms (no post-processing).", kind="supp", float_format="{:.2f}",
                   notes="Failed (strict): number of algorithms with degenerate porosity (< 1 % or > 90 %); "
                         "Failed (lenient): < 5 % or > 90 % (adds collapsed segmentations). GMM fitted by EM on a "
                         f"random subsample of {GMM_MAX_N:,} ROI pixels (3 initialisations, fixed seed).")
    if len(roi_df):
        ctx.save_table(ps.assign(group=ps["group"].map(style.GROUP_LABELS))
                       .rename(columns={"label": "Sample", "group": "Group",
                                        "median_abs_delta_pp": "Median |Δ| (pp)",
                                        "mean_delta_pp": "Mean Δ (pp)", "max_abs_delta_pp": "Max |Δ| (pp)",
                                        "otsu_delta_pp": "Δ Otsu (pp)"})
                       .drop(columns=["sample_id"]),
                       "roi_design_effect",
                       "Porosity difference between ROI designs, Δ = φ(central 500 × 500 px) − φ(largest usable "
                       "disc), in percentage points, over the eight thresholding algorithms (no post-processing).",
                       kind="supp", float_format="{:.1f}")

    # ============================================================ key numbers
    P = "descriptive_"
    ctx.add_key(P + "bc_by_sample", {r["label"]: round(r["bc"], 3) for _, r in bim.iterrows()},
                "Sarle's finite-sample bimodality coefficient of the primary-ROI grey-level histogram (normalised, "
                "winsorised 0.1/99.9 %), per sample; BC > 5/9 = 0.556 suggests bimodality")
    ctx.add_key(P + "ashman_d_by_sample", {r["label"]: round(r["ashman_d"], 2) for _, r in bim.iterrows()},
                "Ashman's D of a 2-component GMM fitted to the primary-ROI grey levels, per sample; D > 2 = clean "
                "separation")
    ctx.add_key(P + "n_bimodal_bc", int(bim["bc_bimodal"].sum()),
                f"number of the {n} samples with BC > 5/9 (primary ROI); samples: "
                + ", ".join(bim.loc[bim["bc_bimodal"], "label"]))
    ctx.add_key(P + "n_separated_d", int(bim["d_separated"].sum()),
                f"number of the {n} samples with Ashman's D > 2 (primary ROI); samples: "
                + ", ".join(bim.loc[bim["d_separated"], "label"]))
    ctx.add_key(P + "intermethod_sd_pp_by_sample", {r["label"]: round(100 * r["intermethod_sd"], 2)
                                                    for _, r in disp.iterrows()},
                "SD of porosity across the 8 thresholding algorithms per sample (primary ROI, no post-processing)",
                "pp")
    ctx.add_key(P + "intermethod_range_pp_by_sample", {r["label"]: round(100 * r["intermethod_range"], 2)
                                                       for _, r in disp.iterrows()},
                "max - min porosity across the 8 algorithms per sample (primary ROI, no post-processing)", "pp")
    for g in ("granular_pack", "consolidated"):
        sub = disp[disp.group == g]
        if len(sub):
            ctx.add_key(P + f"intermethod_range_pp_{g}", [round(100 * sub["intermethod_range"].min(), 2),
                                                          round(100 * sub["intermethod_range"].max(), 2)],
                        f"min and max (over the n = {len(sub)} {g} samples) of the inter-method porosity range "
                        "(8 algorithms, primary ROI, no post-processing)", "pp")
            ctx.add_key(P + f"intermethod_sd_pp_median_{g}", round(100 * sub["intermethod_sd"].median(), 2),
                        f"median over the n = {len(sub)} {g} samples of the inter-method porosity SD "
                        "(8 algorithms, primary ROI, no post-processing)", "pp")
    ctx.add_key(P + "failures_strict", {r["label"]: r["failed_strict"] for _, r in disp.iterrows()
                                        if r["n_failed_strict"]},
                "degenerate segmentations (porosity < 1 % or > 90 %; primary ROI, no post-processing) by sample")
    ctx.add_key(P + "failures_lenient", {r["label"]: r["failed_lenient"] for _, r in disp.iterrows()
                                         if r["n_failed_lenient"]},
                "degenerate or collapsed segmentations (porosity < 5 % or > 90 %; primary ROI, no post-processing)")
    ctx.add_key(P + "n_failures_strict", int(disp["n_failed_strict"].sum()),
                f"number of the {n * len(ctx.methods)} sample x method segmentations with porosity < 1 % or > 90 % "
                "(primary ROI, no post-processing)")
    ctx.add_key(P + "n_failures_lenient", int(disp["n_failed_lenient"].sum()),
                f"number of the {n * len(ctx.methods)} sample x method segmentations with porosity < 5 % or > 90 % "
                "(primary ROI, no post-processing)")
    ctx.add_key(P + "otsu_equals_isodata_n", int(disp["otsu_isodata_same"].sum()),
                f"number of the {n} samples in which Otsu and Isodata give the same porosity (|diff| < 0.01 pp; "
                "primary ROI)")
    for _, a in assoc.iterrows():
        nm = f"{P}spearman_{a['predictor']}_vs_{a['response'].replace('intermethod_', '')}" + \
             ("" if a["analysis"] == "all_methods" else "_excl_failures")
        ctx.add_key(nm, dict(rho=round(a["rho"], 3), boot_ci=[round(a["boot_ci_lo"], 3), round(a["boot_ci_hi"], 3)],
                             bw_ci=[round(a["bw_ci_lo"], 3), round(a["bw_ci_hi"], 3)], p_exact=round(a["p_exact"], 4),
                             p_holm=round(a["p_holm"], 4), n=int(a["n"])),
                    f"Spearman rho between {'Sarle BC' if a['predictor'] == 'bc' else 'Ashman D'} and inter-method "
                    f"porosity {a['response'].replace('intermethod_', '')} across n = {int(a['n'])} images "
                    f"(primary ROI, no post-processing"
                    + ("" if a["analysis"] == "all_methods" else
                       "; methods with porosity < 5 % or > 90 % excluded per image") +
                    f"); boot_ci = {int(100 * level)} % percentile bootstrap over images ({ctx.n_boot} replicates); "
                    "bw_ci = Bonett-Wright Fisher-z interval; p_exact = two-sided exact permutation p "
                    f"(all n! re-pairings); p_holm = Holm-adjusted within {a['holm_family']}")
    ctx.add_key(P + "dice_mean_matrix_min", dict(value=round(float(M[~np.eye(len(M), dtype=bool)].min()), 3),
                                                 pair=_argpair(M, ctx.methods, "min")),
                f"lowest mean (over n = {n} samples) pixel-wise Dice between two methods (primary ROI, no "
                "post-processing)")
    ctx.add_key(P + "dice_mean_matrix_max", dict(value=round(float(M[~np.eye(len(M), dtype=bool)].max()), 4),
                                                 pair=_argpair(M, ctx.methods, "max")),
                f"highest mean (over n = {n} samples) pixel-wise Dice between two methods (primary ROI, no "
                "post-processing)")
    for _, r in fam_df.iterrows():
        ctx.add_key(P + f"dice_{r['pair_type'].replace('-', '_')}", dict(
            mean=round(r["mean_dice"], 3), ci=[round(r["ci_lo"], 3), round(r["ci_hi"], 3)],
            range_samples=[round(r["min_sample"], 3), round(r["max_sample"], 3)], n=int(r["n_samples"])),
            f"mean pixel-wise Dice of {r['pair_type']} method pairs ({r['n_pairs']} pairs): per-sample mean over "
            f"pairs, then mean over n = {int(r['n_samples'])} samples with {int(100 * level)} % percentile "
            "bootstrap CI over samples (primary ROI, no post-processing)")
    for _, r in pd.DataFrame(roi_sum_rows).iterrows():
        ctx.add_key(P + f"roi_median_abs_delta_pp_{r['group']}", dict(
            median=round(r["median_abs_delta_pp"], 2), ci=[round(r["ci_lo"], 2), round(r["ci_hi"], 2)],
            range_samples=[round(r["min_sample"], 2), round(r["max_sample"], 2)], n=int(r["n_samples"])),
            f"ROI design effect, group '{r['group']}': per sample, median over the 8 methods of "
            "|phi(central 500x500 px) - phi(largest usable disc)| (no post-processing); then median over "
            f"n = {int(r['n_samples'])} samples with {int(100 * level)} % percentile bootstrap CI over samples "
            "(coarse for n <= 5)", "pp")
        ctx.add_key(P + f"roi_otsu_abs_delta_pp_{r['group']}", dict(
            median=round(r["otsu_median_abs_delta_pp"], 2), ci=[round(r["otsu_ci_lo"], 2), round(r["otsu_ci_hi"], 2)],
            n=int(r["n_samples"])),
            f"ROI design effect for Otsu only, group '{r['group']}': median over n = {int(r['n_samples'])} "
            f"samples of |phi(central 500 px) - phi(disc)| with {int(100 * level)} % bootstrap CI", "pp")
    if len(roi_df):
        ctx.add_key(P + "roi_delta_pp_by_sample_otsu", {r["label"]: round(r["otsu_delta_pp"], 2)
                                                        for _, r in ps.iterrows()},
                    "phi(central 500x500 px) - phi(largest usable disc), Otsu, no post-processing, per sample", "pp")
    ctx.add_key(P + "zoom_windows_mm", {r["label"]: [round(r["x0_mm"], 2), round(r["y0_mm"], 2), round(r["size_mm"], 2)]
                                        for _, r in windows.iterrows()},
                "zoom windows of the gallery/grids: [x0, y0, size] in mm from the top-left image corner")
    ctx.add_key(P + "otsu_porosity_pct_by_sample", {ctx.label(s): round(100 * _seg_value(ctx, s, "Otsu", "none"), 2)
                                                    for s in sids},
                "Otsu porosity, primary ROI (largest usable disc), no post-processing", "%")
    ctx.add_key(P + "niblack_porosity_pct_range", [round(100 * base[base.method == "Niblack"]["porosity"].min(), 2),
                                                   round(100 * base[base.method == "Niblack"]["porosity"].max(), 2)],
                f"min and max Niblack porosity over the {n} samples (w = 25 px, k = 0.8; primary ROI, no "
                "post-processing); compare Phi(-0.8) = 21.2 % expected inside any homogeneous region", "%")
    ctx.log("descriptive done: %d samples, %d dice rows", n, len(dice))


def _argpair(M, methods, which):
    k = len(methods)
    A = M.copy()
    np.fill_diagonal(A, np.nan if which == "max" else np.nan)
    i, j = np.unravel_index(np.nanargmin(A) if which == "min" else np.nanargmax(A), A.shape)
    return f"{methods[i]}-{methods[j]}"
