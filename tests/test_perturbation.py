"""Unit tests of the 'perturbation' stage (artefact models and small-sample statistics).

Run:  python -m pytest -q tests/test_perturbation.py   (from the repository root)
"""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from scipy import stats as sst

from poroseg import thresholds
from poroseg.metrics import porosity
from poroseg.stages import perturbation as P

CFG = {"thresholds": {"nbins": 256, "sauvola": {"window": 25, "k": 0.2, "r": 0.5},
                      "niblack": {"window": 25, "k": 0.8}}}


def _disc(n=201, r=90):
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    rad = np.hypot(yy - c, xx - c)
    return rad <= r, rad.astype(np.float32), float(r)


def _phantom(seed=0, n=256, phi=0.3, noise=0.03):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    truth = np.zeros((n, n), bool)
    while truth.mean() < phi:
        cy, cx, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(4, 12)
        truth |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img = np.where(truth, 0.2, 0.8) + rng.normal(0, noise, truth.shape)
    return np.clip(img, 0, 1).astype(np.float32), truth


# ----------------------------------------------------------------- artefact models
def test_cupping_gain_profile():
    R, c = 100.0, 0.2
    r = np.array([0.0, 50.0, 100.0, 150.0])
    g = P.cupping_gain(r, R, c)
    assert g[0] == pytest.approx(1 - c)
    assert g[1] == pytest.approx(1 - c * (1 - 0.25))
    assert g[2] == pytest.approx(1.0)
    assert g[3] == pytest.approx(1.0)            # clipped: never brightens outside the disc
    assert np.all(np.diff(g) >= 0)
    assert P.cupping_gain(r, R, 0.0) == pytest.approx(np.ones(4))
    with pytest.raises(ValueError):
        P.cupping_gain(r, 0.0, 0.1)


def test_cupping_applied_to_constant_image():
    mask, rad, R = _disc()
    img = np.where(mask, 0.5, 0).astype(np.float32)
    out, info = P.perturb_image(img, mask, "cupping", 0.3, radius=rad, R=R)
    c = 100
    assert out[c, c] == pytest.approx(0.5 * 0.7, abs=1e-6)
    assert out[c, c + 90] == pytest.approx(0.5, abs=1e-6)
    assert np.all(out[~mask] == 0)
    assert info["frac_clip_low"] == 0 and info["frac_clip_high"] == 0


def test_masked_blur_keeps_constant_and_matches_scipy_in_interior():
    mask, _, _ = _disc()
    img = np.where(mask, 0.6, 0).astype(np.float32)
    out = P.masked_gaussian_blur(img, mask, 3.0)
    assert np.allclose(out[mask], 0.6, atol=1e-5)          # no dark rim (normalised convolution)
    naive = ndi.gaussian_filter(img, 3.0, mode="constant")
    assert naive[mask].min() < 0.45                         # what a plain filter would do at the rim
    rng = np.random.default_rng(3)
    x = rng.random((120, 120)).astype(np.float32)
    full = np.ones_like(x, bool)
    a = P.masked_gaussian_blur(x, full, 2.0)
    b = ndi.gaussian_filter(x, 2.0, mode="constant")
    inner = (slice(12, -12), slice(12, -12))
    assert np.allclose(a[inner], b[inner], atol=1e-5)
    c0 = P.masked_gaussian_blur(x, full, 0.0)
    assert np.array_equal(c0, x)


def test_noise_is_reproducible_has_right_sd_and_is_clipped():
    mask = np.ones((300, 300), bool)
    img = np.full(mask.shape, 0.5, np.float32)
    a, ia = P.perturb_image(img, mask, "noise", 0.05, rng=np.random.default_rng(7))
    b, _ = P.perturb_image(img, mask, "noise", 0.05, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.std(a - img) == pytest.approx(0.05, rel=0.02)
    assert ia["frac_clip_low"] == 0
    dark = np.zeros(mask.shape, np.float32)
    d, idk = P.perturb_image(dark, mask, "noise", 0.05, rng=np.random.default_rng(1))
    assert idk["frac_clip_low"] == pytest.approx(0.5, abs=0.01)   # half of N(0, s) is < 0
    assert d.min() >= 0 and d.max() <= 1
    z, _ = P.perturb_image(img, mask, "noise", 0.0)
    assert np.array_equal(z, img)
    with pytest.raises(ValueError):
        P.perturb_image(img, mask, "noise", 0.1)
    with pytest.raises(ValueError):
        P.perturb_image(img, mask, "ring", 0.1)


def test_native_noise_estimate():
    img, _ = _phantom(seed=4, noise=0.03)
    mask = np.ones_like(img, bool)
    assert P.native_noise_sd(img, mask) == pytest.approx(0.03, rel=0.08)


def _cupping_deltas(img, c):
    mask = np.ones_like(img, bool)
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2).astype(np.float32)
    base = P.segment_methods(img, mask, ["Otsu", "Niblack"], CFG)
    cup, _ = P.perturb_image(img, mask, "cupping", c, radius=rad, R=float(rad.max()))
    pert = P.segment_methods(cup, mask, ["Otsu", "Niblack"], CFG)
    return {m: abs(porosity(pert[m][0], mask) - porosity(base[m][0], mask)) for m in base}


def test_niblack_gain_equivariance_and_cupping_depends_on_contrast():
    """T_Niblack = m - k s is equivariant to a multiplicative gain (m and s
    scale with it): a uniform gain leaves the segmentation unchanged. Under a
    radial (cupping) gain a global threshold fails only when the centre loss
    c * I_matrix exceeds the matrix-threshold margin: harmless for a strongly
    bimodal image, large for a low-contrast one (where Niblack moves much less)."""
    img, _ = _phantom(seed=5, n=256)
    mask = np.ones_like(img, bool)
    base = P.segment_methods(img, mask, ["Niblack"], CFG)
    scaled = P.segment_methods(img * np.float32(0.8), mask, ["Niblack"], CFG)
    assert np.mean(scaled["Niblack"][0] != base["Niblack"][0]) < 1e-3
    # high contrast (0.2 vs 0.8): Otsu robust to c = 0.3
    d_hi = _cupping_deltas(img, 0.3)
    assert d_hi["Otsu"] < 0.005
    # low contrast (0.45 vs 0.60): the darkened centre is labelled pore by Otsu
    rng = np.random.default_rng(9)
    _, truth = _phantom(seed=5, n=256)
    low = np.clip(np.where(truth, 0.45, 0.60) + rng.normal(0, 0.02, truth.shape), 0, 1).astype(np.float32)
    d_lo = _cupping_deltas(low, 0.3)
    assert d_lo["Otsu"] > 0.02                 # > 2 percentage points
    assert d_lo["Otsu"] > 5 * d_lo["Niblack"]
    assert d_lo["Otsu"] > 5 * d_hi["Otsu"] + 0.01


def test_segment_methods_matches_core_path():
    img, _ = _phantom(seed=6)
    mask = np.zeros_like(img, bool)
    mask[20:230, 30:220] = True
    out = P.segment_methods(img, mask, thresholds.METHOD_INFO, CFG)
    for m in thresholds.METHOD_INFO:
        T = thresholds.compute_threshold(m, img, mask, thresholds.default_params(m, CFG))
        assert np.array_equal(out[m][0], thresholds.segment(img, T, mask))
        assert out[m][1] == pytest.approx(T.summary)
        assert not out[m][0][~mask].any()


# ----------------------------------------------------------------- statistics
def _brute_counts(n):
    c = np.zeros(n * (n + 1) // 2 + 1)
    for s in itertools.product([0, 1], repeat=n):
        c[int(np.dot(s, np.arange(1, n + 1)))] += 1
    return c


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_signed_rank_null_distribution(n):
    c = P.signed_rank_null_counts(n)
    assert np.array_equal(c, _brute_counts(n))
    assert c.sum() == 2 ** n and np.array_equal(c, c[::-1])
    if n == 9:   # classical table value: P(T+ <= 5) = 10/512
        assert np.cumsum(c)[5] / 512 == pytest.approx(10 / 512)


@pytest.mark.parametrize("seed,n", [(0, 6), (1, 9), (2, 9), (3, 12), (4, 15)])
def test_wilcoxon_exact_matches_scipy_and_pingouin(seed, n):
    import pingouin as pg
    d = np.random.default_rng(seed).normal(0.4, 1.0, n)
    r = P.wilcoxon_exact(d)
    ref = sst.wilcoxon(d, method="exact")
    assert r["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert min(r["t_plus"], r["t_minus"]) == pytest.approx(ref.statistic)
    pgr = pg.wilcoxon(d)
    assert r["rbc"] == pytest.approx(float(pgr["RBC"].iloc[0]), abs=1e-9)
    assert r["exact"]


def test_wilcoxon_exact_all_positive_and_ties():
    d = np.arange(1, 10, dtype=float)
    r = P.wilcoxon_exact(d)
    assert r["p"] == pytest.approx(2 / 512)
    assert r["rbc"] == pytest.approx(1.0)
    # ties and zeros: compare with a brute-force sign-flip permutation on mid-ranks
    d = np.array([0.0, 1.0, 1.0, -2.0, 3.0, 3.0, 3.0, 4.0])
    r = P.wilcoxon_exact(d)
    dz = d[d != 0]
    rk = sst.rankdata(np.abs(dz))
    tobs = rk[dz > 0].sum()
    ts = np.array([np.dot(s, rk) for s in itertools.product([0, 1], repeat=dz.size)])
    pbf = np.mean(np.abs(ts - rk.sum() / 2) >= abs(tobs - rk.sum() / 2) - 1e-9)
    assert r["n"] == 7 and r["p"] == pytest.approx(pbf)


def test_hodges_lehmann_estimate_and_exact_ci_by_test_inversion():
    rng = np.random.default_rng(11)
    d = rng.normal(1.0, 1.0, 9)
    h = P.hodges_lehmann(d, 0.95)
    walsh = [(d[i] + d[j]) / 2 for i in range(9) for j in range(i, 9)]
    assert h["hl"] == pytest.approx(np.median(walsh))
    assert h["coverage"] == pytest.approx(1 - 2 * 10 / 512)          # 0.961 for n = 9
    w = np.sort(walsh)
    assert h["lo"] == pytest.approx(w[5]) and h["hi"] == pytest.approx(w[39])
    # inversion: shifts inside the CI are not rejected by the exact test, shifts outside are
    eps = 1e-6
    for delta in (h["lo"] + eps, h["hl"], h["hi"] - eps):
        assert sst.wilcoxon(d - delta, method="exact").pvalue > 0.05
    for delta in (h["lo"] - eps, h["hi"] + eps):
        assert sst.wilcoxon(d - delta, method="exact").pvalue <= 0.05


def test_hodges_lehmann_small_n_not_attainable():
    h = P.hodges_lehmann([1.0, 2.0, 3.0, 4.0], 0.95)
    assert np.isnan(h["lo"]) and np.isnan(h["hi"]) and h["hl"] == pytest.approx(2.5)


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    assert np.allclose(P.holm(p), multipletests(p, method="holm")[1])
    q = P.holm([0.02, np.nan, 0.01])
    assert np.isnan(q[1]) and q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.02)


def test_family_means_and_contrast():
    rows = []
    for s, (g, l) in enumerate([(2.0, 0.5), (3.0, 1.0), (1.0, 2.0)]):
        for m in ("A", "B", "C"):
            rows.append(dict(sample_id=f"s{s}", method=m, family="global", absdelta_pp=g + (m == "C")))
        for m in ("D", "E"):
            rows.append(dict(sample_id=f"s{s}", method=m, family="local", absdelta_pp=l))
    df = pd.DataFrame(rows)
    fm = P.family_means(df)
    assert fm.loc["s0", "global"] == pytest.approx(2 + 1 / 3)
    assert fm.loc["s0", "local"] == pytest.approx(0.5)
    assert fm.loc["s2", "diff"] == pytest.approx(4 / 3 - 2)
    fmed = P.family_means(df, agg="median")
    assert fmed.loc["s0", "global"] == pytest.approx(2.0)
    excl = df["method"] == "C"
    fx = P.family_means(df, exclude=excl)
    assert fx.loc["s1", "global"] == pytest.approx(3.0) and fx.loc["s1", "n_global"] == 2
    r = P.contrast_test(fm["diff"].to_numpy())
    assert r["n"] == 3 and r["n_pos"] == 2 and r["n_neg"] == 1


def test_reference_level():
    assert P.reference_level([0.0, 0.01, 0.02, 0.04, 0.08]) == pytest.approx(0.02)
    assert P.reference_level([0.0, 0.05, 0.10, 0.20, 0.30]) == pytest.approx(0.10)
    assert P.reference_level([0.0, 0.5, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
