"""Stage 'perturbation': robustness of the eight thresholding algorithms to
controlled acquisition artefacts applied to the REAL micro-CT images.

Scientific question
-------------------
How much does the porosity delivered by each thresholding algorithm move when
the image is degraded by artefacts that are common in laboratory micro-CT
(Ketcham & Carlson 2001; Wildenschild & Sheppard 2013; Schlueter et al. 2014),
and is the claim of the original manuscript - that the local adaptive methods
(Sauvola, Niblack) are more robust to beam hardening than global methods -
supported by the data?

Design (one factor at a time)
-----------------------------
For every image (n = 9, the experimental unit), the normalised primary-ROI
image ``I`` (``ctx.norm``; grey levels in [0, 1], 0 outside the ROI) is
degraded by ONE artefact at a time, at the levels listed in ``config.yaml``
(section ``perturbation``), and the eight algorithms are RE-RUN on the
degraded image with their default parameters
(:func:`poroseg.thresholds.default_params`), through
:func:`poroseg.thresholds.compute_threshold` and
:func:`poroseg.thresholds.segment` with the ROI mask (global histograms use
ROI pixels only; local statistics are masked). No post-processing is applied
(level 'none') - the question is the response of the thresholding step itself.

1. **Additive Gaussian noise** (reconstructed CT noise is approximately
   Gaussian; Hsieh 2009, ch. 5)::

       I'(x) = clip(I(x) + sigma_n * Z(x), 0, 1),   Z ~ N(0, 1) i.i.d.

   ``sigma_n`` is in normalised grey-level units, i.e. a fraction of the ROI
   dynamic range between the 0.1 and 99.9 percentiles. ``n_rep`` independent
   noise fields per level (``ctx.rng('perturbation', 'noise', sid, level, rep)``);
   all eight methods see the SAME noisy image within a replicate (paired
   design). The native pixel noise of each image is estimated for reference by
   :func:`native_noise_sd`. White (uncorrelated) noise is a simplification:
   real reconstruction noise is spatially correlated by the reconstruction
   filter.

2. **Radial cupping** (beam-hardening-like gain; Brooks & Di Chiro 1976;
   Ketcham & Carlson 2001): a parabolic loss of apparent attenuation towards
   the centre of the sample::

       I'(x) = I(x) * g(r),   g(r) = 1 - c * (1 - (r / R)^2),   r <= R

   with ``r`` the distance (px) from the fitted sample/holder centre
   (``ctx.geometry``) and ``R`` the radius of the usable disc (``r_usable``),
   so the gain is ``1 - c`` at the centre and 1 at the ROI edge (``r/R`` is
   clipped to 1, so ``g <= 1`` everywhere). The gain is applied to the
   *normalised* grey level, whose zero is the darkest (pore/air) phase, so it
   reduces the contrast of every phase relative to the pore phase in
   proportion to its grey level; this is an approximation of the physical
   effect, which acts on the linear attenuation coefficient. Deterministic:
   one replicate.

3. **Gaussian blur** (extra partial-volume effect / poorer spatial resolution;
   Ketcham & Carlson 2001; Iassonov et al. 2009)::

       I'(x) = [G_s * (I M)](x) / [G_s * M](x)   for x in the ROI mask M

   i.e. a *normalised convolution* (Knutsson & Westin 1993) with a Gaussian
   kernel of standard deviation ``s`` px (truncated at 4 s), so that the zeros
   outside the ROI never darken the ROI rim. Deterministic: one replicate.

The perturbed image is **not re-normalised** (otherwise a global gain or
contrast loss would be undone before the algorithm sees it); it is only
clipped to [0, 1] (which only noise can violate; the fraction of clipped ROI
pixels is recorded) and set to 0 outside the ROI.

Outcomes (computed on ``ROI.eval_mask``, i.e. the ROI minus a 2 px rim)
----------------------------------------------------------------------
* ``porosity``  phi' = pore area fraction of the perturbed segmentation;
* ``delta_pp``  100 (phi' - phi0), percentage points, phi0 = unperturbed
  porosity of the same method on the same image (identical to the factorial
  cache ``ctx.pore(sid, method, 'none')``; checked at run time);
* ``dice``      Dice (1945) overlap 2|A n B| / (|A| + |B|) between the perturbed
  and the unperturbed pore sets (1 = identical pixels).

Level 0 rows (unperturbed) are stored with delta_pp = 0 and dice = 1.

Statistics (the image is the unit; pixels are never replicates)
---------------------------------------------------------------
For each sample x method x artefact x level the replicate noise fields are
averaged first (mean |delta| over replicates = expected absolute change for a
single acquisition). Summaries across samples are medians with inter-quartile
ranges (n = 9).

*Family contrast* (are local methods more robust than global methods?):
for each sample i, ``G_i`` = mean |delta| over the 6 global methods and
``L_i`` = mean |delta| over the 2 local methods; ``d_i = G_i - L_i`` (pp;
positive = local methods move less). Test of H0: median(d) = 0 by the exact
Wilcoxon signed-rank test (Wilcoxon 1945; exact null distribution of T+ by
enumeration of the 2^n sign vectors, mid-ranks for ties, zeros dropped);
effect size = Hodges-Lehmann (1963) estimate (median of the n(n+1)/2 Walsh
averages) with its exact distribution-free confidence interval obtained by
inverting the signed-rank test (Hollander, Wolfe & Chicken 2014, sec. 3.3),
and the matched-pairs rank-biserial correlation r = (T+ - T-)/(T+ + T-)
(Kerby 2014). Multiplicity: Holm (1979) within each artefact across its
non-zero levels. The pre-specified primary comparison is cupping at the
reference level (the middle configured level, c = 0.10). Sensitivity
analyses: (i) family *median* instead of mean; (ii) excluding clear method
failures (unperturbed phi0 < 5 % or > 90 %; Yen/Triangle on TQ06, VN01, VN02),
reported separately. With n = 9 the smallest attainable two-sided exact p is
2/2^9 = 0.0039, so power is limited and the direction/size of the effect
(HL estimate, CI, number of samples) is emphasised over p.

Robustness is NOT accuracy: an algorithm can be insensitive to an artefact
because its output is anchored to local statistics rather than to the pore
phase (Niblack labels ~Phi(-k) = 21 % of any homogeneous region as pore).
Accuracy is assessed against laboratory porosity and synthetic ground truth in
other stages.

Outputs
-------
data/perturbation.csv               one row per sample x method x artefact x level x replicate
                                    (sample_id, label, group, method, family, perturbation, level,
                                    rep, porosity, delta_pp, dice, porosity_unperturbed, t_summary)
data/perturbation_by_sample.csv     replicate-averaged per sample x method x artefact x level
data/perturbation_curves.csv        across-sample medians/IQR per method x artefact x level
data/perturbation_image_stats.csv   per perturbed image: clipped fractions, mean, s.d., native noise
data/perturbation_family_test.csv   family contrast tests (all levels + sensitivity analyses)
figures/perturbation_robustness     (main) median |delta phi| vs level, 3 artefacts, 8 methods
figures/supp_perturbation_dice      Dice vs level
figures/supp_perturbation_by_sample signed delta phi per sample x method at the reference levels
figures/supp_perturbation_family    paired per-sample global vs local family means
figures/supp_perturbation_examples  what the artefacts look like on two real images
tables/perturbation_summary         (main) method x artefact: median |delta phi| [IQR] + Dice
tables/perturbation_family_test     (supp) Wilcoxon / Hodges-Lehmann family contrasts

References
----------
Brooks RA, Di Chiro G (1976) Beam hardening in X-ray reconstructive tomography. Phys Med Biol 21:390-398.
Dice LR (1945) Measures of the amount of ecologic association between species. Ecology 26:297-302.
Hodges JL, Lehmann EL (1963) Estimates of location based on rank tests. Ann Math Stat 34:598-611.
Hollander M, Wolfe DA, Chicken E (2014) Nonparametric Statistical Methods, 3rd edn. Wiley.
Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65-70.
Hsieh J (2009) Computed Tomography: Principles, Design, Artifacts, and Recent Advances, 2nd edn. SPIE Press.
Iassonov P, Gebrenegus T, Tuller M (2009) Segmentation of X-ray computed tomography images of porous
    materials: a crucial step for characterization and quantitative analysis of pore structures.
    Water Resour Res 45:W09415.
Ketcham RA, Carlson WD (2001) Acquisition, optimization and interpretation of X-ray computed tomographic
    imagery: applications to the geosciences. Comput Geosci 27:381-400.
Kerby DS (2014) The simple difference formula: an approach to teaching nonparametric correlation.
    Compr Psychol 3:11.IT.3.1.
Knutsson H, Westin C-F (1993) Normalized and differential convolution. Proc IEEE CVPR, 515-523.
Rousseeuw PJ, Croux C (1993) Alternatives to the median absolute deviation. J Am Stat Assoc 88:1273-1283.
Schlueter S, Sheppard A, Brown K, Wildenschild D (2014) Image processing of multiphase images obtained via
    X-ray microtomography: a review. Water Resour Res 50:3615-3639.
Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bull 1:80-83.
Wildenschild D, Sheppard AP (2013) X-ray imaging and analysis techniques for quantifying pore-scale
    structure and processes in subsurface porous medium systems. Adv Water Resour 51:217-246.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats as sst

from .. import style, thresholds
from ..metrics import dice as _dice
from ..metrics import porosity as _porosity
from ..roi import radial_distance

PERTURBATIONS = ("noise", "cupping", "blur")
CFG_KEYS = {"noise": "noise_sd", "cupping": "cupping", "blur": "blur_sigma"}
ART_TITLES = {"noise": "Noise", "cupping": "Cupping", "blur": "Blur"}
LEVEL_LABELS = {"noise": "Added noise σ (normalised grey level)",
                "cupping": "Cupping amplitude c (–)",
                "blur": "Blur σ (px)"}
LEVEL_SYMBOL = {"noise": "σ", "cupping": "c", "blur": "σ"}
LEVEL_UNIT = {"noise": "", "cupping": "", "blur": " px"}
FAILURE_LO, FAILURE_HI = 0.05, 0.90     # clear method failures (unperturbed porosity)
MAX_ENUM_N = 16                         # exact enumeration of 2^n sign vectors up to this n


# =============================================================================
# 1. Artefact models (pure functions)
# =============================================================================
def cupping_gain(r: np.ndarray, R: float, c: float) -> np.ndarray:
    """Parabolic beam-hardening-like gain ``g(r) = 1 - c (1 - (r/R)^2)``.

    Parameters
    ----------
    r : array of distances (px) from the sample centre.
    R : radius (px) at which the gain returns to 1 (usable-disc radius).
    c : cupping amplitude; the gain at the centre is ``1 - c``.

    Returns
    -------
    float32 array of the same shape; ``r/R`` is clipped to [0, 1] so that
    ``1 - c <= g <= 1`` everywhere (pixels outside the disc are never brightened).
    """
    if R <= 0:
        raise ValueError("R must be positive")
    rho = np.clip(np.asarray(r, dtype=np.float64) / float(R), 0.0, 1.0)
    return (1.0 - float(c) * (1.0 - rho ** 2)).astype(np.float32)


def masked_gaussian_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur restricted to a mask by normalised convolution.

    ``out = G_s * (I M) / G_s * M`` inside ``M`` (Knutsson & Westin 1993); 0
    outside. Pixels outside the ROI therefore never leak into it (a plain
    Gaussian filter would darken the ROI rim because the image is 0 outside).
    Far from the mask border (more than 4 sigma) the result equals a plain
    ``scipy.ndimage.gaussian_filter``; a constant image stays constant up to
    the border. ``sigma <= 0`` returns a copy.
    """
    img = np.asarray(img, dtype=np.float32)
    if sigma <= 0:
        out = img.copy()
        out[~mask] = 0.0
        return out
    m = mask.astype(np.float32)
    num = ndi.gaussian_filter(img * m, float(sigma), mode="constant", cval=0.0, output=np.float32)
    den = ndi.gaussian_filter(m, float(sigma), mode="constant", cval=0.0, output=np.float32)
    out = np.zeros_like(img)
    ok = mask & (den > 1e-6)
    out[ok] = num[ok] / den[ok]
    return out


def perturb_image(img: np.ndarray, mask: np.ndarray, kind: str, level: float, *,
                  radius: np.ndarray | None = None, R: float | None = None,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """Apply ONE artefact to a normalised ROI image (no re-normalisation).

    Parameters
    ----------
    img : bbox-shaped float image in [0, 1] (0 outside the ROI).
    mask : bool ROI mask (same shape).
    kind : 'noise' | 'cupping' | 'blur'.
    level : noise s.d. (normalised units) | cupping amplitude c | blur s.d. (px).
    radius, R : distance map (px) from the sample centre and the usable-disc
        radius (required for 'cupping').
    rng : numpy Generator (required for 'noise' with level > 0).

    Returns
    -------
    out : float32 image, clipped to [0, 1] and 0 outside the mask.
    info : dict with ``frac_clip_low`` / ``frac_clip_high`` = fraction of ROI
        pixels that fell below 0 / above 1 before clipping.
    """
    img = np.asarray(img, dtype=np.float32)
    if kind == "noise":
        if level > 0:
            if rng is None:
                raise ValueError("noise needs an rng")
            out = img + np.float32(level) * rng.standard_normal(img.shape, dtype=np.float32)
        else:
            out = img.copy()
    elif kind == "cupping":
        if radius is None or R is None:
            raise ValueError("cupping needs radius and R")
        out = img * cupping_gain(radius, R, level)
    elif kind == "blur":
        out = masked_gaussian_blur(img, mask, level)
    else:
        raise ValueError(f"unknown perturbation '{kind}'")
    n = max(int(mask.sum()), 1)
    info = dict(frac_clip_low=float(((out < 0) & mask).sum() / n),
                frac_clip_high=float(((out > 1) & mask).sum() / n))
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    out[~mask] = 0.0
    return out, info


def native_noise_sd(img: np.ndarray, mask: np.ndarray) -> float:
    """Robust estimate of the native pixel-noise s.d. of an image.

    Uses first differences between horizontally and vertically adjacent pixel
    pairs that both lie inside the mask: for white noise of s.d. sigma on a
    locally constant signal, ``Var(I(x+1) - I(x)) = 2 sigma^2``. Edges between
    phases are a minority of pairs and are down-weighted by the MAD
    (consistency constant 1.4826 for Gaussian data; Rousseeuw & Croux 1993)::

        sigma_hat = 1.4826 * MAD(differences) / sqrt(2)

    Spatially correlated reconstruction noise makes this an underestimate of
    the marginal noise s.d.; it is reported only as a reference scale for the
    added-noise levels.
    """
    img = np.asarray(img, dtype=np.float64)
    dx = (img[:, 1:] - img[:, :-1])[mask[:, 1:] & mask[:, :-1]]
    dy = (img[1:, :] - img[:-1, :])[mask[1:, :] & mask[:-1, :]]
    d = np.concatenate([dx, dy])
    if d.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def segment_methods(img: np.ndarray, mask: np.ndarray, methods, cfg: dict) -> dict:
    """Re-run each thresholding algorithm (default parameters) on ``img``.

    Returns ``{method: (pore_bool_array, threshold_summary)}``; the threshold
    summary is the scalar threshold (global) or the ROI mean of the map (local).
    """
    out = {}
    for m in methods:
        T = thresholds.compute_threshold(m, img, mask, thresholds.default_params(m, cfg))
        out[m] = (thresholds.segment(img, T, mask), float(T.summary))
    return out


# =============================================================================
# 2. Small-sample statistics (pure functions; validated against scipy/pingouin)
# =============================================================================
def signed_rank_null_counts(n: int) -> np.ndarray:
    """Exact null distribution of the Wilcoxon statistic T+ for n untied ranks.

    ``counts[t]`` = number of the 2^n equally likely sign vectors with
    ``T+ = t`` (t = 0..n(n+1)/2), by the standard recursion
    c_k(t) = c_{k-1}(t) + c_{k-1}(t - k).
    """
    M = n * (n + 1) // 2
    c = np.zeros(M + 1, dtype=np.float64)
    c[0] = 1.0
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k]          # right-hand side is evaluated on the old c (c_{k-1})
    return c


def wilcoxon_exact(d) -> dict:
    """Exact two-sided Wilcoxon signed-rank test of H0: d symmetric about 0.

    Zeros are dropped (Wilcoxon's convention); ties among |d| receive
    mid-ranks; the exact p-value is obtained by enumerating all 2^n sign
    vectors of the observed (mid-)ranks (n <= 16; otherwise scipy's normal
    approximation is used and flagged). Two-sided p = P(|T+ - E T+| >=
    |t_obs - E T+|) with E T+ = sum(ranks)/2.

    Returns
    -------
    dict(n, t_plus, t_minus, p, rbc, exact) where rbc is the matched-pairs
    rank-biserial correlation (T+ - T-)/(T+ + T-) (Kerby 2014).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = d.size
    if n == 0:
        return dict(n=0, t_plus=np.nan, t_minus=np.nan, p=np.nan, rbc=np.nan, exact=True)
    ranks = sst.rankdata(np.abs(d))
    tp = float(ranks[d > 0].sum())
    tm = float(ranks[d < 0].sum())
    tot = tp + tm
    rbc = (tp - tm) / tot if tot > 0 else np.nan
    if n <= MAX_ENUM_N:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        tdist = signs @ ranks
        centre = tot / 2.0
        p = float(np.mean(np.abs(tdist - centre) >= abs(tp - centre) - 1e-9))
        exact = True
    else:
        p = float(sst.wilcoxon(d, method="approx").pvalue)
        exact = False
    return dict(n=int(n), t_plus=tp, t_minus=tm, p=min(p, 1.0), rbc=float(rbc), exact=exact)


def hodges_lehmann(d, ci: float = 0.95) -> dict:
    """One-sample (paired) Hodges-Lehmann estimate with exact CI.

    The estimate is the median of the M = n(n+1)/2 Walsh averages
    (d_i + d_j)/2, i <= j (Hodges & Lehmann 1963). The distribution-free CI
    inverts the two-sided exact signed-rank test (Hollander, Wolfe & Chicken
    2014, sec. 3.3): with ``C`` the smallest t such that P(T+ <= t) > alpha/2
    under H0, the interval is [W_(C), W_(M+1-C)] (order statistics of the
    sorted Walsh averages), with achieved coverage 1 - 2 P(T+ <= C - 1)
    (>= nominal). Assumes no ties/zeros (continuous data). If the nominal
    level is not attainable (n too small, e.g. n <= 5 for 95 %), the bounds
    are NaN.

    Returns dict(n, hl, lo, hi, coverage).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return dict(n=0, hl=np.nan, lo=np.nan, hi=np.nan, coverage=np.nan)
    i, j = np.triu_indices(n)
    W = np.sort((d[i] + d[j]) / 2.0)
    M = W.size
    hl = float(np.median(W))
    alpha = 1.0 - ci
    cdf = np.cumsum(signed_rank_null_counts(n)) / 2.0 ** n
    C = int(np.argmax(cdf > alpha / 2.0))       # smallest t with P(T+ <= t) > alpha/2
    if C < 1:
        return dict(n=int(n), hl=hl, lo=np.nan, hi=np.nan, coverage=np.nan)
    return dict(n=int(n), hl=hl, lo=float(W[C - 1]), hi=float(W[M - C]),
                coverage=float(1.0 - 2.0 * cdf[C - 1]))


def holm(p) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (NaNs are left untouched)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def family_means(by_sample: pd.DataFrame, value: str = "absdelta_pp", agg: str = "mean",
                 exclude: pd.Series | None = None) -> pd.DataFrame:
    """Per-sample family summaries of a robustness outcome.

    Parameters
    ----------
    by_sample : rows = sample x method (one artefact level), with columns
        sample_id, family ('global'|'local') and ``value``.
    agg : 'mean' or 'median' across the methods of a family.
    exclude : optional bool Series aligned with ``by_sample`` (True = drop row).

    Returns
    -------
    DataFrame indexed by sample_id with columns global, local, diff = global - local,
    n_global, n_local.
    """
    d = by_sample if exclude is None else by_sample[~exclude.to_numpy()]
    g = d.groupby(["sample_id", "family"])[value].agg(agg).unstack("family")
    n = d.groupby(["sample_id", "family"])[value].size().unstack("family")
    out = pd.DataFrame({"global": g.get("global"), "local": g.get("local")})
    out["diff"] = out["global"] - out["local"]
    out["n_global"] = n.get("global")
    out["n_local"] = n.get("local")
    return out


def contrast_test(diff, ci: float = 0.95) -> dict:
    """Exact Wilcoxon + Hodges-Lehmann summary of paired differences."""
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    w = wilcoxon_exact(diff)
    h = hodges_lehmann(diff, ci)
    return dict(n=int(diff.size), hl=h["hl"], ci_lo=h["lo"], ci_hi=h["hi"], ci_coverage=h["coverage"],
                t_plus=w["t_plus"], p=w["p"], rbc=w["rbc"], exact=w["exact"],
                n_pos=int((diff > 0).sum()), n_neg=int((diff < 0).sum()))


# =============================================================================
# 3. helpers for the stage
# =============================================================================
def reference_level(levels) -> float:
    """Middle configured level (same convention as the report scorecard)."""
    lv = sorted(float(x) for x in levels)
    return lv[len(lv) // 2] if len(lv) > 1 else lv[0]


def _lvl_tag(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _fmt_iqr(med, q1, q3, nd=2) -> str:
    if not np.isfinite(med):
        return ""
    return f"{med:.{nd}f} [{q1:.{nd}f}–{q3:.{nd}f}]"


def _q(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan


# =============================================================================
# 4. stage
# =============================================================================
def _compute(ctx, levels: dict, n_rep: int):
    """Run every perturbation on every sample; return tidy rows + image stats."""
    cfg = ctx.cfg
    rows, irows = [], []
    mism = {}
    t_all = time.time()
    for sid in ctx.sample_ids:
        t0 = time.time()
        R = ctx.roi(sid)
        N = ctx.norm(sid)
        g = ctx.geometry(sid)
        img0, mask, ev = N.img, R.mask, R.eval_mask
        radius = radial_distance(R, g).astype(np.float32)
        base = segment_methods(img0, mask, ctx.methods, cfg)
        # consistency with the factorial cache (should be bit-identical)
        mism[sid] = {m: int(np.count_nonzero(base[m][0] ^ ctx.pore(sid, m, "none"))) for m in ctx.methods}
        if any(mism[sid].values()):
            ctx.log("WARNING %s: baseline differs from factorial cache: %s", sid, mism[sid])
        phi0 = {m: _porosity(base[m][0], ev) for m in ctx.methods}
        nsd = native_noise_sd(img0, mask)
        common = dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid))
        for kind in PERTURBATIONS:
            for lv in levels[kind]:
                if lv == 0:
                    for m in ctx.methods:
                        rows.append(dict(**common, method=m, family=thresholds.METHOD_INFO[m]["family"],
                                         perturbation=kind, level=0.0, rep=0, porosity=phi0[m],
                                         delta_pp=0.0, dice=1.0, porosity_unperturbed=phi0[m],
                                         t_summary=base[m][1]))
                    continue
                reps = n_rep if kind == "noise" else 1
                for rep in range(reps):
                    rng = ctx.rng("perturbation", "noise", sid, f"{lv:g}", rep) if kind == "noise" else None
                    img, info = perturb_image(img0, mask, kind, lv, radius=radius, R=g.r_usable, rng=rng)
                    vals = img[ev]
                    irows.append(dict(**common, perturbation=kind, level=float(lv), rep=rep,
                                      native_noise_sd=nsd, mean=float(vals.mean()), sd=float(vals.std()),
                                      mean_shift=float(vals.mean() - img0[ev].mean()), **info))
                    seg = segment_methods(img, mask, ctx.methods, cfg)
                    for m in ctx.methods:
                        pore, ts = seg[m]
                        phi = _porosity(pore, ev)
                        rows.append(dict(**common, method=m, family=thresholds.METHOD_INFO[m]["family"],
                                         perturbation=kind, level=float(lv), rep=rep, porosity=phi,
                                         delta_pp=100.0 * (phi - phi0[m]),
                                         dice=_dice(pore, base[m][0], ev),
                                         porosity_unperturbed=phi0[m], t_summary=ts))
                    del img, seg
        del base, radius
        ctx.log("%s done in %.1f s (native noise sd = %.4f)", sid, time.time() - t0, nsd)
    return pd.DataFrame(rows), pd.DataFrame(irows), mism, time.time() - t_all


def _aggregate(df: pd.DataFrame):
    keys = ["sample_id", "label", "group", "method", "family", "perturbation", "level"]
    d = df.assign(absdelta_pp=df["delta_pp"].abs())
    bs = d.groupby(keys, sort=False).agg(
        porosity_unperturbed=("porosity_unperturbed", "first"),
        porosity_mean=("porosity", "mean"),
        delta_pp_mean=("delta_pp", "mean"),
        delta_pp_sd=("delta_pp", "std"),
        absdelta_pp=("absdelta_pp", "mean"),
        dice=("dice", "mean"),
        n_rep=("rep", "size")).reset_index()
    bs["failure_unperturbed"] = (bs["porosity_unperturbed"] < FAILURE_LO) | (bs["porosity_unperturbed"] > FAILURE_HI)
    cr = []
    for (art, lv, m, fam), g in bs.groupby(["perturbation", "level", "method", "family"], sort=False):
        a, dc, sd = g["absdelta_pp"].to_numpy(), g["dice"].to_numpy(), g["delta_pp_mean"].to_numpy()
        cr.append(dict(perturbation=art, level=lv, method=m, family=fam, n_samples=len(g),
                       absdelta_median=_q(a, .5), absdelta_q25=_q(a, .25), absdelta_q75=_q(a, .75),
                       absdelta_min=float(np.min(a)), absdelta_max=float(np.max(a)),
                       delta_median=_q(sd, .5), delta_q25=_q(sd, .25), delta_q75=_q(sd, .75),
                       dice_median=_q(dc, .5), dice_q25=_q(dc, .25), dice_q75=_q(dc, .75),
                       dice_min=float(np.min(dc))))
    return bs, pd.DataFrame(cr)


def _family_tests(bs: pd.DataFrame, levels: dict, ref: dict, ci: float) -> pd.DataFrame:
    out = []
    for art in PERTURBATIONS:
        nz = [lv for lv in levels[art] if lv > 0]
        block = []
        for lv in nz:
            sub = bs[(bs["perturbation"] == art) & (bs["level"] == lv)].reset_index(drop=True)
            analyses = [("primary: family mean", "mean", None)]
            if lv == ref[art]:
                analyses += [("sensitivity: family median", "median", None),
                             ("sensitivity: failures excluded", "mean", sub["failure_unperturbed"])]
            for name, agg, excl in analyses:
                fm = family_means(sub, "absdelta_pp", agg, excl)
                fm = fm.dropna(subset=["diff"])
                r = contrast_test(fm["diff"].to_numpy(), ci)
                rec = dict(perturbation=art, level=lv, is_reference=lv == ref[art], analysis=name,
                           global_median=float(fm["global"].median()), local_median=float(fm["local"].median()),
                           ratio_median=float(np.median(fm["global"] / fm["local"])), **r,
                           n_excluded_rows=int(excl.sum()) if excl is not None else 0)
                (block if name.startswith("primary") else out).append(rec)
        ph = holm([b["p"] for b in block])
        for b, p in zip(block, ph):
            b["p_holm"] = float(p)
        out = block + out
    df = pd.DataFrame(out)
    order ={a: i for i, a in enumerate(PERTURBATIONS)}
    df["_o"] = df["perturbation"].map(order) * 10 + df["analysis"].str.startswith("sensitivity").astype(int)
    return df.sort_values(["_o", "level", "analysis"]).drop(columns="_o").reset_index(drop=True)


# ----------------------------------------------------------------------------- figures
def _log_fmt(v, _pos=None):
    if v <= 0:
        return ""
    return f"{v:g}"


def _fig_robustness(ctx, curves, levels, ref, ycol="absdelta", ylog=True, ylabel="", floor=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, axs = plt.subplots(1, 3, figsize=style.figsize("double", 72), sharey=True)
    n_m = len(ctx.methods)
    offs = np.linspace(-0.33, 0.33, n_m)
    for k, (ax, art) in enumerate(zip(axs, PERTURBATIONS)):
        lv = [x for x in levels[art] if (x > 0 or not ylog)]
        xs = np.arange(len(lv))
        for j, m in enumerate(ctx.methods):
            c = curves[(curves["perturbation"] == art) & (curves["method"] == m)].set_index("level")
            c = c.reindex(lv)
            med = c[f"{ycol}_median"].to_numpy()
            lo = c[f"{ycol}_q25"].to_numpy()
            hi = c[f"{ycol}_q75"].to_numpy()
            if floor is not None:
                med, lo, hi = (np.maximum(a, floor) for a in (med, lo, hi))
            x = xs + offs[j]
            col = style.METHOD_COLORS[m]
            ax.plot(x, med, "-", color=col, lw=0.6, alpha=0.55, zorder=2)
            ax.vlines(x, lo, hi, color=col, lw=0.8, zorder=3)
            ax.plot(x, med, ls="none", marker=style.METHOD_MARKERS[m], ms=3.2, mfc=col, mec="white",
                    mew=0.3, color=col, zorder=4)
        if ycol == "absdelta":
            ax.axhline(1.0, color=style.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        for xx in xs[:-1]:
            ax.axvline(xx + 0.5, color=style.GRID, lw=0.5, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in lv])
        ax.set_xlim(-0.55, len(lv) - 0.45)
        for t, x in zip(ax.get_xticklabels(), lv):
            if np.isclose(x, ref[art]):
                t.set_fontweight("bold")
        ax.set_xlabel(LEVEL_LABELS[art])
        ax.set_title(ART_TITLES[art], pad=3)
        style.panel_label(ax, "abc"[k])
        style.hgrid(ax)
        if ylog:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.yaxis.set_major_formatter(FuncFormatter(_log_fmt))
            ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel(ylabel)
    fig.legend(handles=style.method_handles(ctx.methods), loc="upper center", ncol=len(ctx.methods),
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    return fig


def _fig_by_sample(ctx, bs, ref):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axs = plt.subplots(1, 3, figsize=style.figsize("double", 80))
    labels = [ctx.label(s) for s in ctx.sample_ids]
    for k, (ax, art) in enumerate(zip(axs, PERTURBATIONS)):
        sub = bs[(bs["perturbation"] == art) & (bs["level"] == ref[art])]
        M = sub.pivot(index="label", columns="method", values="delta_pp_mean").reindex(index=labels,
                                                                                      columns=ctx.methods)
        v = np.nanmax(np.abs(M.to_numpy()))
        v = max(v, 0.05)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)
        im = ax.imshow(M.to_numpy(), cmap=style.CMAP_DIV, norm=norm, aspect="auto")
        for (i, j), val in np.ndenumerate(M.to_numpy()):
            if not np.isfinite(val):
                continue
            txt = f"{val:.0f}" if abs(val) >= 9.95 else f"{val:.1f}"
            if txt in ("-0.0", "0.0"):
                txt = "0"
            col = "white" if abs(norm(val) - 0.5) > 0.33 else style.INK
            ax.text(j, i, txt.replace("-", "−"), ha="center", va="center", fontsize=5.5, color=col)
        ax.set_xticks(range(len(ctx.methods)))
        ax.set_xticklabels(ctx.methods, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if k == 0 else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{ART_TITLES[art]} ({LEVEL_SYMBOL[art]} = {ref[art]:g}{LEVEL_UNIT[art]})", pad=3)
        style.panel_label(ax, "abc"[k])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2, aspect=30)
        cb.set_label("Δφ (percentage points)")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(length=2)
    fig.tight_layout(w_pad=0.6)
    return fig


def _fig_family(ctx, bs, ref):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, axs = plt.subplots(1, 3, figsize=style.figsize("double", 62))
    for k, (ax, art) in enumerate(zip(axs, PERTURBATIONS)):
        sub = bs[(bs["perturbation"] == art) & (bs["level"] == ref[art])].reset_index(drop=True)
        fm = family_means(sub, "absdelta_pp", "mean")
        lab = {s: ctx.label(s) for s in fm.index}
        floor = 1e-3
        for sid, r in fm.iterrows():
            gm = style.GROUP_MARKERS[ctx.group(sid)]
            y = [max(r["global"], floor), max(r["local"], floor)]
            col = style.INK2 if r["diff"] > 0 else style.PALETTE[7]
            ax.plot([0, 1], y, "-", color=col, lw=0.6, alpha=0.8)
            ax.plot([0, 1], y, ls="none", marker=gm, ms=3, color=col, mfc="white", mew=0.6)
            ax.text(1.06, y[1], lab[sid], fontsize=5.5, va="center", ha="left", color=style.INK2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_log_fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Global (6)", "Local (2)"])
        ax.set_xlim(-0.3, 1.45)
        style.hgrid(ax)
        ax.set_title(f"{ART_TITLES[art]} ({LEVEL_SYMBOL[art]} = {ref[art]:g}{LEVEL_UNIT[art]})", pad=3)
        style.panel_label(ax, "abc"[k])
        r = contrast_test(fm["diff"].to_numpy())
        ax.text(0.02, 0.02, f"HL(G−L) = {r['hl']:.2f} pp\n95% CI [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]\n"
                            f"exact p = {r['p']:.3f}; {r['n_pos']}/{r['n']} G > L",
                transform=ax.transAxes, fontsize=6, va="bottom", ha="left", color=style.INK)
    axs[0].set_ylabel("Family mean |Δφ| (percentage points)")
    hs = [Line2D([], [], color=style.INK2, lw=0.6, label="global > local"),
          Line2D([], [], color=style.PALETTE[7], lw=0.6, label="global < local"),
          Line2D([], [], ls="none", marker="o", mfc="white", color=style.INK2, ms=3, label="Granular pack"),
          Line2D([], [], ls="none", marker="D", mfc="white", color=style.INK2, ms=3, label="Consolidated")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
    return fig


def _fig_examples(ctx, levels, sids):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sids = [s for s in sids if s in ctx.sample_ids]
    if not sids:
        return None
    nmax = max(levels["noise"])
    cmax = max(levels["cupping"])
    bmax = max(levels["blur"])
    cols = ["Unperturbed", f"Cupping c = {cmax:g}", "Unperturbed (zoom)", f"Noise σ = {nmax:g}",
            f"Blur σ = {bmax:g} px"]
    fig, axs = plt.subplots(len(sids), 5, figsize=style.figsize("double", 40 * len(sids) + 6))
    axs = np.atleast_2d(axs)
    for i, sid in enumerate(sids):
        R = ctx.roi(sid)
        N = ctx.norm(sid)
        g = ctx.geometry(sid)
        mask = R.mask
        radius = radial_distance(R, g).astype(np.float32)
        cup, _ = perturb_image(N.img, mask, "cupping", cmax, radius=radius, R=g.r_usable)
        noi, _ = perturb_image(N.img, mask, "noise", nmax, rng=ctx.rng("perturbation", "noise", sid, f"{nmax:g}", 0))
        blu, _ = perturb_image(N.img, mask, "blur", bmax)
        h, w = mask.shape
        zs = 160
        cy = int(round(g.cy - R.y0))
        cx = int(round(g.cx - R.x0 + 0.35 * g.r_usable))
        y0, x0 = max(cy - zs // 2, 0), max(min(cx - zs // 2, w - zs), 0)
        zsl = (slice(y0, y0 + zs), slice(x0, x0 + zs))
        panels = [(N.img, False), (cup, False), (N.img[zsl], True), (noi[zsl], True), (blu[zsl], True)]
        px = ctx.pixel_size_um(sid)
        for j, (im, zoom) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest" if zoom else "antialiased")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            style.scalebar(ax, px, color="white")
            if j == 0:
                ax.add_patch(Rectangle((x0, y0), zs, zs, fill=False, ec=style.PALETTE[3], lw=0.7))
                ax.set_ylabel(ctx.label(sid))
            if i == 0:
                ax.set_title(cols[j], pad=3)
        style.panel_label(axs[i, 0], "ab"[i] if i < 2 else str(i))
    fig.tight_layout(w_pad=0.3, h_pad=0.5)
    return fig


# ----------------------------------------------------------------------------- tables
def _summary_table(curves, ref, methods):
    rows = []
    for m in methods:
        d = dict(Method=m, Family=thresholds.METHOD_INFO[m]["family"])
        for art in PERTURBATIONS:
            c = curves[(curves["perturbation"] == art) & (curves["method"] == m)
                       & np.isclose(curves["level"], ref[art])].iloc[0]
            tag = f"{ART_TITLES[art]} {LEVEL_SYMBOL[art]} = {ref[art]:g}{LEVEL_UNIT[art]}"
            d[f"{tag}: |Δφ| (pp)"] = _fmt_iqr(c["absdelta_median"], c["absdelta_q25"], c["absdelta_q75"])
            d[f"{tag}: Dice"] = _fmt_iqr(c["dice_median"], c["dice_q25"], c["dice_q75"], 3)
        rows.append(d)
    return pd.DataFrame(rows)


def _family_table(ft):
    rows = []
    for _, r in ft.iterrows():
        rows.append({
            "Artefact": ART_TITLES[r["perturbation"]],
            "Level": f"{LEVEL_SYMBOL[r['perturbation']]} = {r['level']:g}{LEVEL_UNIT[r['perturbation']]}"
                     + (" *" if r["is_reference"] else ""),
            "Analysis": r["analysis"].split(": ")[1],
            "n": int(r["n"]),
            "Global |Δφ| (pp)": r["global_median"],
            "Local |Δφ| (pp)": r["local_median"],
            "HL G−L (pp) [95% CI]": (f"{r['hl']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
                                     if np.isfinite(r["ci_lo"]) else f"{r['hl']:.2f} [n/a]"),
            "G > L (samples)": f"{int(r['n_pos'])}/{int(r['n'])}",
            "r_rb": r["rbc"],
            "p exact": r["p"],
            "p Holm": r["p_holm"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(ctx) -> None:
    pc = ctx.cfg["perturbation"]
    levels = {k: sorted(float(x) for x in pc[CFG_KEYS[k]]) for k in PERTURBATIONS}
    for k in PERTURBATIONS:
        if 0.0 not in levels[k]:
            levels[k] = [0.0] + levels[k]
    n_rep = int(pc["n_rep"])
    if ctx.quick:
        n_rep = min(n_rep, 2)
    ref = {k: reference_level(levels[k]) for k in PERTURBATIONS}
    ci = float(ctx.cfg["statistics"]["ci_level"])
    ctx.log("levels %s; n_rep(noise) = %d; reference levels %s", levels, n_rep, ref)

    df, imgstats, mism, runtime = _compute(ctx, levels, n_rep)
    cols = ["sample_id", "label", "group", "method", "family", "perturbation", "level", "rep",
            "porosity", "delta_pp", "dice", "porosity_unperturbed", "t_summary"]
    df = df[cols]
    ctx.save_data(df, "perturbation")
    ctx.save_data(imgstats, "perturbation_image_stats")
    bs, curves = _aggregate(df)
    ctx.save_data(bs, "perturbation_by_sample")
    ctx.save_data(curves, "perturbation_curves")
    ft = _family_tests(bs, levels, ref, ci)
    ctx.save_data(ft, "perturbation_family_test")

    n = len(ctx.sample_ids)
    n_img = len(imgstats)
    ref_txt = ", ".join(f"{ART_TITLES[a].lower()} {LEVEL_SYMBOL[a]} = {ref[a]:g}{LEVEL_UNIT[a]}" for a in PERTURBATIONS)
    nsd = imgstats.groupby("sample_id")["native_noise_sd"].first()
    clip_max = imgstats[(imgstats["perturbation"] == "noise") & (imgstats["level"] == max(levels["noise"]))]
    clip_max = float((clip_max["frac_clip_low"] + clip_max["frac_clip_high"]).max()) if len(clip_max) else 0.0
    design = (f"Each of the n = {n} images (primary ROI = largest usable disc; no post-processing) was degraded by "
              f"one artefact at a time and all eight algorithms were re-run with their default parameters on the "
              f"degraded image, without re-normalisation (values clipped to [0, 1]). Noise: additive white Gaussian "
              f"noise, σ in normalised grey-level units (fraction of the ROI 0.1–99.9 percentile range; native "
              f"pixel-noise σ̂ of the images {nsd.min():.3f}–{nsd.max():.3f}), {n_rep} independent noise fields "
              f"per level, averaged per image. Cupping: radial gain 1 − c(1 − (r/R)²) centred on the sample "
              f"(R = usable-disc radius; gain 1 − c at the centre). Blur: Gaussian, σ in pixels, applied by "
              f"normalised convolution inside the ROI. |Δφ|: absolute change in porosity relative to the "
              f"unperturbed segmentation of the same method and image (percentage points).")

    # ---------------------------------------------------------- main figure
    pos = curves[curves["level"] > 0]
    floor = 1e-3
    fig = _fig_robustness(ctx, pos, levels, ref, "absdelta", True,
                          "Median |Δφ| across images (percentage points)", floor=floor)
    ctx.save_fig(fig, "perturbation_robustness",
                 "Robustness of the eight thresholding algorithms to controlled acquisition artefacts applied "
                 f"to the real images. {design} Points: median over the {n} images of |Δφ|; vertical bars: "
                 "inter-quartile range across images (not a confidence interval); thin lines join the levels of "
                 "one method; symbols are dodged horizontally within each level (circles: global, squares: "
                 "local adaptive methods). Logarithmic y-axis; the unperturbed level (|Δφ| = 0 by definition) is "
                 f"not shown and values below {floor:g} pp are drawn at {floor:g} pp. Dotted line: 1 percentage "
                 f"point. Bold tick: reference level used in the summary table ({ref_txt}).", kind="main")

    # ---------------------------------------------------------- supp figures
    fig = _fig_robustness(ctx, curves, levels, ref, "dice", False, "Median Dice vs unperturbed segmentation")
    for ax in fig.axes:
        if ax.get_ylabel() or ax.get_title():
            ax.set_ylim(top=1.01)
    ctx.save_fig(fig, "supp_perturbation_dice",
                 "Pixel-level agreement between the segmentation of the perturbed and of the unperturbed image "
                 f"(Dice coefficient, evaluated on the ROI minus a 2 px rim). {design} Points: median over the "
                 f"{n} images; vertical bars: inter-quartile range across images; symbols dodged within each level "
                 "(circles: global, squares: local adaptive). Level 0 is the unperturbed image (Dice = 1). "
                 "A method can keep its porosity nearly constant (small |Δφ|) while relocating many pore "
                 "pixels (low Dice).", kind="supp")

    fig = _fig_by_sample(ctx, bs, ref)
    ctx.save_fig(fig, "supp_perturbation_by_sample",
                 f"Signed porosity change Δφ = φ(perturbed) − φ(unperturbed), in percentage points, for every "
                 f"image (rows) and algorithm (columns) at the reference level of each artefact ({ref_txt}). "
                 f"{design} Noise cells are the mean over {n_rep} noise fields. Diverging colour scale centred at "
                 "0 (blue: porosity decreases; red: porosity increases), with its own range in each panel. "
                 "Primary ROI, no post-processing.", kind="supp")

    fig = _fig_family(ctx, bs, ref)
    ctx.save_fig(fig, "supp_perturbation_family",
                 "Paired comparison of the robustness of the global (Otsu, Li, Yen, Mean, Triangle, Isodata) and "
                 "local adaptive (Sauvola, Niblack) families at the reference level of each artefact. For each "
                 f"image (n = {n}; circles: granular packs, diamonds: consolidated samples) the family mean of "
                 "|Δφ| over its methods is shown, joined by a line (grey: global family changed more; red: local "
                 "family changed more). Inset: Hodges–Lehmann estimate of the paired difference global − local "
                 "with its exact 95 % confidence interval, and the exact two-sided Wilcoxon signed-rank p-value "
                 f"(n = {n} images as the unit). Logarithmic y-axis. {design}", kind="supp")

    fig = _fig_examples(ctx, levels, ["DS23_CF5", "AG24_VN04"])
    if fig is not None:
        ctx.save_fig(fig, "supp_perturbation_examples",
                     "Appearance of the strongest artefact levels on two real images (a: granular pack CF5; "
                     "b: consolidated sample VN04), all shown on the same normalised grey scale [0, 1] "
                     "(black = 0). Columns 1–2: whole usable disc, unperturbed and with radial cupping; columns "
                     "3–5: 160 × 160 px zoom (yellow square in column 1), unperturbed, with additive Gaussian "
                     "noise (one of the noise fields used in the analysis) and with Gaussian blur. Scale bars in "
                     "physical units.", kind="supp")

    # ---------------------------------------------------------- tables
    st = _summary_table(curves, ref, ctx.methods)
    ctx.save_table(st, "perturbation_summary",
                   "Robustness of each thresholding algorithm to the three artefacts at the reference levels "
                   f"({ref_txt}). Entries are the median [inter-quartile range] over the n = {n} images of the "
                   "absolute porosity change |Δφ| (percentage points) and of the Dice overlap between perturbed "
                   "and unperturbed segmentations. Primary ROI (usable disc), no post-processing; noise entries "
                   f"are first averaged over {n_rep} independent noise fields per image.",
                   kind="main",
                   notes="Robustness is not accuracy: a method can be insensitive to an artefact while being biased "
                         "(e.g. Niblack labels about Φ(−k) = 21 % of any homogeneous region as pore). The perturbed "
                         "image is not re-normalised. Clear method failures (unperturbed φ < 5 % or > 90 %) are "
                         "included.")
    ftab = _family_table(ft)
    ctx.save_table(ftab, "perturbation_family_test",
                   "Are local adaptive thresholds more robust than global thresholds? For each image, the family "
                   "mean of |Δφ| over the 6 global and the 2 local methods was computed (image = experimental "
                   f"unit, n = {n}); 'Global/Local |Δφ|' are the medians of these per-image values. HL G−L: "
                   "Hodges–Lehmann estimate of the paired difference global − local with exact distribution-free "
                   "95 % confidence interval (positive = local methods change less). G > L: number of images in "
                   "which the global family changed more. r_rb: matched-pairs rank-biserial correlation. p exact: "
                   "two-sided exact Wilcoxon signed-rank test; p Holm: Holm-adjusted across the levels of the same "
                   "artefact (primary analyses only). Primary ROI, no post-processing.",
                   kind="supp", float_format="{:.3f}",
                   notes="* reference level; pre-specified primary comparison: cupping at the reference level. "
                         "Sensitivity analyses at the reference level: family median instead of mean; clear "
                         f"failures (unperturbed φ < {100 * FAILURE_LO:.0f} % or > {100 * FAILURE_HI:.0f} %) "
                         f"excluded. With n = {n} the smallest attainable two-sided exact p is "
                         f"{2 / 2 ** n:.4f}.")

    # ---------------------------------------------------------- key numbers
    ctx.add_key("perturbation_n_images", n, "number of images (experimental units) perturbed")
    ctx.add_key("perturbation_n_rep_noise", n_rep, "independent noise fields per image and noise level")
    ctx.add_key("perturbation_n_perturbed_images", n_img,
                "number of perturbed images segmented (each by all 8 methods), excluding the unperturbed level")
    ctx.add_key("perturbation_reference_levels", ref,
                "reference (middle) level of each artefact used in the summary table and the family tests "
                "(noise: s.d. in normalised grey-level units; cupping: amplitude c; blur: Gaussian s.d. in px)")
    ctx.add_key("perturbation_levels", levels, "configured levels of each artefact (0 = unperturbed)")
    ctx.add_key("perturbation_native_noise_sd_range", [float(nsd.min()), float(nsd.max())],
                "range over images of the native pixel-noise s.d. (MAD of first differences / sqrt 2) of the "
                "normalised primary-ROI images, normalised grey-level units")
    ctx.add_key("perturbation_native_noise_sd_by_sample", {ctx.label(s): float(v) for s, v in nsd.items()},
                "native pixel-noise s.d. per image (normalised grey-level units)")
    ctx.add_key("perturbation_max_clipped_fraction_noise_max", clip_max,
                f"largest fraction of ROI pixels clipped to [0, 1] at the strongest noise level "
                f"(σ = {max(levels['noise']):g}) over images and replicates")
    ctx.add_key("perturbation_baseline_mismatch_px", int(sum(sum(v.values()) for v in mism.values())),
                "pixels differing between the stage's unperturbed re-segmentation and the factorial cache "
                "(should be 0)", "px")
    ctx.add_key("perturbation_runtime_s", float(runtime), "wall time of the perturbation computations", "s")
    for art in PERTURBATIONS:
        c = curves[np.isclose(curves["level"], ref[art]) & (curves["perturbation"] == art)].set_index("method")
        lt = f"{art} level {ref[art]:g}"
        for m in ctx.methods:
            r = c.loc[m]
            ctx.add_key(f"perturbation_{art}_ref_{m.lower()}_absdelta_pp",
                        [r["absdelta_median"], r["absdelta_q25"], r["absdelta_q75"]],
                        f"{m}: median [Q1, Q3] over n = {n} images of |Δφ| at {lt} (primary ROI, post none; "
                        f"noise averaged over {n_rep} fields); IQR across images, not a CI", "pp")
            ctx.add_key(f"perturbation_{art}_ref_{m.lower()}_delta_pp_median", r["delta_median"],
                        f"{m}: median over n = {n} images of the SIGNED porosity change at {lt}", "pp")
            ctx.add_key(f"perturbation_{art}_ref_{m.lower()}_dice",
                        [r["dice_median"], r["dice_q25"], r["dice_q75"]],
                        f"{m}: median [Q1, Q3] over n = {n} images of Dice(perturbed, unperturbed) at {lt}")
        # strongest level, per method
        top = max(levels[art])
        ct = curves[np.isclose(curves["level"], top) & (curves["perturbation"] == art)].set_index("method")
        ctx.add_key(f"perturbation_{art}_max_level_absdelta_pp_by_method",
                    {m: float(ct.loc[m, "absdelta_median"]) for m in ctx.methods},
                    f"median over n = {n} images of |Δφ| at the strongest {art} level ({top:g}), per method", "pp")
        ctx.add_key(f"perturbation_{art}_max_level_dice_by_method",
                    {m: float(ct.loc[m, "dice_median"]) for m in ctx.methods},
                    f"median over n = {n} images of Dice at the strongest {art} level ({top:g}), per method")
    for _, r in ft.iterrows():
        name = r["analysis"].split(": ")[1].replace(" ", "_")
        k = f"perturbation_family_{r['perturbation']}_{_lvl_tag(r['level'])}_{name}"
        ctx.add_key(k, dict(n=int(r["n"]), global_median_pp=r["global_median"], local_median_pp=r["local_median"],
                            hl_global_minus_local_pp=r["hl"], ci95_lo=r["ci_lo"], ci95_hi=r["ci_hi"],
                            ci_coverage=r["ci_coverage"], p_exact=r["p"],
                            p_holm=r["p_holm"] if np.isfinite(r.get("p_holm", np.nan)) else None,
                            rank_biserial=r["rbc"], n_global_gt_local=int(r["n_pos"])),
                    f"family contrast global − local of the per-image family {('median' if 'median' in name else 'mean')} "
                    f"|Δφ| ({r['analysis']}) at {r['perturbation']} level {r['level']:g}: Hodges–Lehmann estimate "
                    f"with exact distribution-free 95 % CI, exact two-sided Wilcoxon signed-rank p (n images as unit)"
                    f"{', Holm-adjusted across levels of this artefact' if r['analysis'].startswith('primary') else ''}; "
                    "positive = local methods change less; primary ROI, post none", "pp")
    ctx.log("done: %d rows, %d perturbed images, %.0f s", len(df), n_img, runtime)
