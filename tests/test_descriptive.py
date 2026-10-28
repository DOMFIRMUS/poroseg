"""Unit tests of the pure functions of stage 'descriptive'.

Every statistic is checked on synthetic data with a known answer and, where an
independent implementation exists, cross-validated against scipy / statsmodels
/ pingouin / scikit-learn.

Run:  python -m pytest -q tests/test_descriptive.py   (from the repository root)
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from poroseg.metrics import dice as core_dice
from poroseg.metrics import jaccard as core_jaccard
from poroseg.stages import descriptive as D


# ---------------------------------------------------------------- bimodality
def test_sarle_bc_matches_formula_and_scipy():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, 500)
    out = D.sarle_bc(x)
    n = x.size
    g1 = sps.skew(x, bias=False)
    g2 = sps.kurtosis(x, fisher=True, bias=False)
    bc = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    assert out["bc"] == pytest.approx(bc, rel=1e-12)
    assert out["skewness"] == pytest.approx(g1)
    assert out["excess_kurtosis"] == pytest.approx(g2)
    assert out["n"] == n


def test_sarle_bc_reference_distributions():
    rng = np.random.default_rng(2)
    n = 200_000
    # normal -> 1/3 ; uniform -> 5/9 ; well separated mixture -> > 5/9
    assert D.sarle_bc(rng.normal(size=n))["bc"] == pytest.approx(1 / 3, abs=0.01)
    assert D.sarle_bc(rng.uniform(size=n))["bc"] == pytest.approx(5 / 9, abs=0.01)
    mix = np.r_[rng.normal(0.2, 0.05, n // 2), rng.normal(0.8, 0.05, n // 2)]
    assert D.sarle_bc(mix)["bc"] > D.BC_CRIT
    # affine invariance
    x = rng.normal(size=1000) ** 2
    assert D.sarle_bc(3.0 * x - 7.0)["bc"] == pytest.approx(D.sarle_bc(x)["bc"], rel=1e-10)


def test_sarle_bc_needs_four_values():
    with pytest.raises(ValueError):
        D.sarle_bc([1.0, 2.0, 3.0])


def test_ashman_d_known_values():
    assert D.ashman_d(0.0, 1.0, 2.0, 1.0) == pytest.approx(2.0)          # sqrt2*2/sqrt2
    assert D.ashman_d(0.3, 0.05, 0.7, 0.05) == pytest.approx(np.sqrt(2) * 0.4 / np.sqrt(0.005))
    assert D.ashman_d(1, 0.1, 0, 0.1) == D.ashman_d(0, 0.1, 1, 0.1)       # symmetric


def test_fit_gmm2_recovers_known_mixture():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0.25, 0.05, 30_000), rng.normal(0.70, 0.08, 70_000)]
    g = D.fit_gmm2(x, random_state=0, rng=np.random.default_rng(0), max_n=50_000)
    assert g["gmm_mu1"] == pytest.approx(0.25, abs=0.005)
    assert g["gmm_mu2"] == pytest.approx(0.70, abs=0.005)
    assert g["gmm_sd1"] == pytest.approx(0.05, abs=0.005)
    assert g["gmm_sd2"] == pytest.approx(0.08, abs=0.005)
    assert g["gmm_w1"] == pytest.approx(0.30, abs=0.02)
    assert g["gmm_w1"] + g["gmm_w2"] == pytest.approx(1.0)
    assert g["gmm_n_fit"] == 50_000
    assert g["ashman_d"] == pytest.approx(D.ashman_d(0.25, 0.05, 0.70, 0.08), rel=0.05)
    # reproducible with a fixed seed
    g2 = D.fit_gmm2(x, random_state=0, rng=np.random.default_rng(0), max_n=50_000)
    assert g2["gmm_mu1"] == g["gmm_mu1"]


def test_gauss_bin_fraction_integrates_to_weight():
    edges = np.linspace(0, 1, 257)
    cen = (edges[:-1] + edges[1:]) / 2
    f = D.gauss_bin_fraction(cen, edges[1] - edges[0], 0.5, 0.05, 0.4)
    assert f.sum() == pytest.approx(0.4, rel=1e-6)
    # matches the binned histogram of a large sample
    rng = np.random.default_rng(4)
    x = rng.normal(0.5, 0.05, 400_000)
    h = np.histogram(x, bins=edges)[0] / x.size
    assert np.abs(h - f / 0.4).max() < 2e-3


# ----------------------------------------------------- thresholds / failures
def test_local_threshold_band():
    tmap = np.arange(100, dtype=float).reshape(10, 10)
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    mean, lo, hi = D.local_threshold_band(tmap, mask)
    v = tmap[mask]
    assert mean == pytest.approx(v.mean())
    assert (lo, hi) == pytest.approx(tuple(np.percentile(v, [5, 95])))


def test_failure_flags():
    phi = np.array([0.005, 0.01, 0.3, 0.9, 0.95, np.nan])
    assert D.failure_flags(phi).tolist() == [True, False, False, False, True, False]
    assert D.failure_flags(phi, lo=0.05).tolist() == [True, True, False, False, True, False]


# ---------------------------------------------------------------- overlap
def test_pairwise_overlap_matches_core_metrics_and_identity():
    rng = np.random.default_rng(5)
    ev = np.ones((60, 60), bool)
    ev[:5] = False
    pores = {"A": rng.random((60, 60)) < 0.3, "B": rng.random((60, 60)) < 0.5,
             "C": np.zeros((60, 60), bool)}
    pores["A2"] = pores["A"].copy()
    df = D.pairwise_overlap(pores, ev)
    assert len(df) == math.comb(4, 2)
    for _, r in df.iterrows():
        a, b = pores[r.method_a], pores[r.method_b]
        assert r.dice == pytest.approx(core_dice(a, b, ev))
        assert r.jaccard == pytest.approx(core_jaccard(a, b, ev))
        if r.dice > 0:
            assert r.jaccard == pytest.approx(r.dice / (2 - r.dice))     # J = D / (2 - D)
    same = df[(df.method_a == "A") & (df.method_b == "A2")].iloc[0]
    assert same.dice == 1.0 and same.jaccard == 1.0
    # empty vs non-empty -> 0 ; empty vs empty -> 1 by convention
    ac = df[(df.method_a == "A") & (df.method_b == "C")].iloc[0]
    assert ac.dice == 0.0
    e = D.pairwise_overlap({"x": np.zeros((4, 4), bool), "y": np.zeros((4, 4), bool)}, np.ones((4, 4), bool))
    assert e.dice.iloc[0] == 1.0 and e.jaccard.iloc[0] == 1.0


def test_overlap_matrix_symmetric_mean():
    pairs = pd.DataFrame(dict(method_a=["A", "A", "A", "B"], method_b=["B", "B", "C", "C"],
                              dice=[0.5, 0.7, 0.2, 0.9]))
    M = D.overlap_matrix(pairs, ["A", "B", "C"])
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)
    assert M[0, 1] == pytest.approx(0.6)
    assert M[0, 2] == pytest.approx(0.2)
    assert M[1, 2] == pytest.approx(0.9)


# ---------------------------------------------------------------- Spearman
def test_spearman_rho_matches_scipy_with_ties():
    rng = np.random.default_rng(6)
    for _ in range(5):
        x = rng.integers(0, 4, 12).astype(float)
        y = x + rng.normal(0, 1, 12)
        assert D.spearman_rho(x, y) == pytest.approx(sps.spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_rho_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=9), rng.normal(size=9)
    r = pg.corr(x, y, method="spearman")["r"].iloc[0]
    assert D.spearman_rho(x, y) == pytest.approx(r, abs=1e-12)


def test_spearman_exact_permutation_known_answer():
    # perfectly monotone, n = 5: only the identity (rho = 1) and the reversal
    # (rho = -1) reach |rho| = 1 -> two-sided p = 2/120, one-sided p = 1/120
    x = np.arange(5.0)
    out = D.spearman_permutation(x, x ** 3)
    assert out["exact"] and out["n_perm"] == 120
    assert out["rho"] == pytest.approx(1.0)
    assert out["p_two_sided"] == pytest.approx(2 / 120)
    assert out["p_greater"] == pytest.approx(1 / 120)
    assert out["p_less"] == pytest.approx(1.0)


def test_spearman_exact_permutation_matches_scipy_permutation_test():
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=7), rng.normal(size=7)
    y[:3] += 1.5 * x[:3]
    out = D.spearman_permutation(x, y)

    def stat(yy):
        return sps.spearmanr(x, yy).statistic

    ref = sps.permutation_test((y,), stat, permutation_type="pairings", n_resamples=np.inf,
                               alternative="two-sided")
    assert out["rho"] == pytest.approx(ref.statistic)
    assert out["p_two_sided"] == pytest.approx(ref.pvalue, abs=1e-9)
    ref_g = sps.permutation_test((y,), stat, permutation_type="pairings", n_resamples=np.inf,
                                 alternative="greater")
    assert out["p_greater"] == pytest.approx(ref_g.pvalue, abs=1e-9)


def test_spearman_monte_carlo_branch_close_to_exact():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=8), rng.normal(size=8)
    ex = D.spearman_permutation(x, y)
    mc = D.spearman_permutation(x, y, rng=np.random.default_rng(1), max_exact_n=5, n_perm=60_000)
    assert not mc["exact"]
    assert mc["p_two_sided"] == pytest.approx(ex["p_two_sided"], abs=0.01)


def test_spearman_boot_ci_brackets_estimate():
    rng = np.random.default_rng(10)
    x = rng.normal(size=9)
    y = x + rng.normal(0, 0.5, 9)
    ci = D.spearman_boot_ci(x, y, 2000, np.random.default_rng(0))
    rho = D.spearman_rho(x, y)
    assert -1 <= ci["ci_lo"] <= rho <= ci["ci_hi"] <= 1
    assert ci["n_valid"] <= 2000


def test_bonett_wright_ci_formula():
    lo, hi = D.bonett_wright_ci(0.5, 20, 0.95)
    z, se = np.arctanh(0.5), np.sqrt((1 + 0.5 ** 2 / 2) / 17)
    assert lo == pytest.approx(np.tanh(z - 1.959964 * se), abs=1e-6)
    assert hi == pytest.approx(np.tanh(z + 1.959964 * se), abs=1e-6)
    assert all(np.isnan(D.bonett_wright_ci(0.5, 3)))


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    assert np.allclose(D.holm(p), multipletests(p, method="holm")[1])


def test_bootstrap_ci_mean_of_normal():
    rng = np.random.default_rng(11)
    v = rng.normal(10, 2, 400)
    est, lo, hi = D.bootstrap_ci(v, np.mean, 4000, np.random.default_rng(0))
    se = v.std(ddof=1) / np.sqrt(v.size)
    assert est == pytest.approx(v.mean())
    assert lo == pytest.approx(v.mean() - 1.96 * se, abs=0.06)
    assert hi == pytest.approx(v.mean() + 1.96 * se, abs=0.06)
    assert all(np.isnan(D.bootstrap_ci([np.nan, np.nan], np.median, 10)))


# ---------------------------------------------------------------- windows
def test_window_means_matches_brute_force():
    rng = np.random.default_rng(12)
    a = rng.random((40, 50))
    ys, xs = np.array([0, 3, 17]), np.array([0, 9, 30])
    m = D._window_means(a, 10, ys, xs)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            assert m[i, j] == pytest.approx(a[y:y + 10, x:x + 10].mean())


def test_choose_window_finds_hot_spot_inside_mask():
    score = np.zeros((200, 200))
    score[120:150, 40:70] = 1.0                  # informative structure here
    score[0:30, 0:30] = 5.0                      # stronger, but outside the valid mask
    valid = np.zeros((200, 200), bool)
    valid[20:190, 20:190] = True
    w = D.choose_window([score], valid, 30, stride=5)
    assert valid[w["y0"]:w["y0"] + 30, w["x0"]:w["x0"] + 30].all()
    assert (w["y0"], w["x0"]) == (120, 40)
    with pytest.raises(ValueError):
        D.choose_window([score], np.zeros((200, 200), bool), 30)


def test_overlay_rgb_paints_pores_only():
    grey = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    pore = np.zeros((4, 4), bool)
    pore[0, 0] = pore[3, 3] = True
    rgb = D.overlay_rgb(grey, pore, 0.0, 1.0, color="#ff0000", alpha=1.0)
    assert rgb.shape == (4, 4, 3)
    assert np.allclose(rgb[0, 0], [1, 0, 0]) and np.allclose(rgb[3, 3], [1, 0, 0])
    assert np.allclose(rgb[1, 2], grey[1, 2])      # untouched pixels stay grey
    assert np.allclose(D.overlay_rgb(grey, None, 0, 1)[..., 0], grey)


def test_robust_limits():
    v = np.r_[np.zeros(5), np.linspace(0, 1, 990), np.ones(5) * 50]
    lo, hi = D.robust_limits(v)
    assert lo == pytest.approx(np.percentile(v, 0.5))
    assert hi == pytest.approx(np.percentile(v, 99.5))
    lo, hi = D.robust_limits(np.ones(10))
    assert hi > lo


def test_niblack_homogeneous_fraction():
    # sanity check of the interpretation quoted in the paper: Niblack (T = m - k s)
    # labels about Phi(-k) of any homogeneous Gaussian region as pore
    from poroseg.thresholds import niblack_map

    rng = np.random.default_rng(13)
    img = rng.normal(0.5, 0.05, (300, 300)).astype(np.float32)
    mask = np.ones_like(img, bool)
    t = niblack_map(img, mask, 25, 0.8)
    inner = (slice(25, -25), slice(25, -25))
    frac = float((img[inner] <= t[inner]).mean())
    assert frac == pytest.approx(sps.norm.cdf(-0.8), abs=0.01)
