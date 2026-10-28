"""Unit tests of poroseg.spatial_tools (synthetic data with known answers).

Run:  python -m pytest -q tests/test_spatial.py
"""
import numpy as np
import pytest
from scipy import stats
from scipy.special import erf  # noqa: F401  (scipy import sanity)

from poroseg import spatial_tools as st


def _disc_mask(n, r=None, cy=None, cx=None):
    r = n / 2 - 1 if r is None else r
    cy = (n - 1) / 2 if cy is None else cy
    cx = (n - 1) / 2 if cx is None else cx
    yy, xx = np.mgrid[:n, :n]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _disc_covariogram(h, R):
    """Geometric covariogram of a disc of radius R (area of K intersect K+h)."""
    h = np.asarray(h, float)
    out = np.zeros_like(h)
    m = h < 2 * R
    x = h[m] / (2 * R)
    out[m] = 2 * R * R * (np.arccos(x) - x * np.sqrt(1 - x * x))
    return out


def _boolean_discs(n, lam, R, rng):
    """Boolean model of discs (germs: Poisson, intensity lam per px^2); pore = union."""
    pad = int(np.ceil(R)) + 1
    area = (n + 2 * pad) ** 2
    k = rng.poisson(lam * area)
    cy = rng.uniform(-pad, n + pad, k)
    cx = rng.uniform(-pad, n + pad, k)
    img = np.zeros((n, n), bool)
    Ri = int(np.ceil(R))
    yy, xx = np.mgrid[-Ri:Ri + 1, -Ri:Ri + 1]
    for y, x in zip(cy, cx):
        iy, ix = int(round(y)), int(round(x))
        y0, y1 = max(iy - Ri, 0), min(iy + Ri + 1, n)
        x0, x1 = max(ix - Ri, 0), min(ix + Ri + 1, n)
        if y0 >= y1 or x0 >= x1:
            continue
        sub = ((yy + iy - y) ** 2 + (xx + ix - x) ** 2 <= R * R)[y0 - iy + Ri:y1 - iy + Ri, x0 - ix + Ri:x1 - ix + Ri]
        img[y0:y1, x0:x1] |= sub
    return img


# --------------------------------------------------------------------------- two-point
def test_fft_two_point_equals_bruteforce_on_irregular_mask():
    rng = np.random.default_rng(1)
    ind = rng.random((40, 37)) < 0.3
    mask = _disc_mask(40, r=17)[:, :37] | (rng.random((40, 37)) < 0.05)
    mask[:3, :] = False
    mc = st.MaskedCorrelator(mask, max_lag=6, workers=1, min_pairs_frac=0.0)
    a = mc.two_point(ind)
    b = st.two_point_bruteforce(ind, mask, 6)
    ok = b["npairs"] > 0
    assert np.allclose(mc.npairs[ok], b["npairs"][ok])
    assert np.allclose(a["s2"][ok], b["s2"][ok], atol=1e-10)
    assert np.allclose(a["cov"][ok], b["cov"][ok], atol=1e-10)
    # value at zero lag: S2(0) = phi, rho(0) = 1
    c = mc.max_lag
    assert a["s2"][c, c] == pytest.approx(a["phi"])
    assert a["rho"][c, c] == pytest.approx(1.0)


def test_mask_normalisation_removes_window_bias():
    """S2 of the same stationary field seen through a disc and through the full
    square must agree (circular FFT without normalisation would not)."""
    rng = np.random.default_rng(2)
    img = _boolean_discs(256, lam=0.002, R=5.0, rng=rng)
    full = np.ones_like(img)
    disc = _disc_mask(256, r=120)
    a = st.MaskedCorrelator(full, 30, workers=1).two_point(img, phi=img.mean())
    b = st.MaskedCorrelator(disc, 30, workers=1).two_point(img, phi=img.mean())
    ra = st.radial_average(a["s2"], a["npairs"], a["lag_r"], 30)[1]
    rb = st.radial_average(b["s2"], b["npairs"], b["lag_r"], 30)[1]
    assert np.max(np.abs(ra - rb)) < 0.02


def test_boolean_model_two_point_matches_theory():
    """Boolean discs: P(both points outside the union) = q^(2 - K(h)/K(0)),
    q = exp(-lam pi R^2). Pore = union, so S2_pore(h) = 1 - 2q + q^(2-r(h))."""
    rng = np.random.default_rng(3)
    lam, R = 0.003, 6.0
    img = _boolean_discs(512, lam, R, rng)
    mc = st.MaskedCorrelator(np.ones_like(img), 30, workers=1)
    tp = mc.two_point(img)
    r, s2, _ = st.radial_average(tp["s2"], tp["npairs"], tp["lag_r"], 30)
    q = np.exp(-lam * np.pi * R * R)
    rr = _disc_covariogram(r, R) / (np.pi * R * R)
    s2_th = 1 - 2 * q + q ** (2 - rr)
    assert tp["phi"] == pytest.approx(1 - q, abs=0.02)
    assert np.max(np.abs(s2[1:] - s2_th[1:])) < 0.02
    # correlation length (1/e of the normalised covariance): compare with theory
    rho_th = (q ** (2 - rr) - q * q) / (q * (1 - q))
    lc_th = st.correlation_length(r, rho_th)
    cs = st.summarize_covariance(tp)
    assert cs.lc_px == pytest.approx(lc_th, abs=1.0)
    # integral range: A = int (q^(2-r(h)) - q^2) dh / (q (1-q)) (positive everywhere)
    hh = np.linspace(0, 2 * R, 4001)
    A_th = np.trapezoid(2 * np.pi * hh * (q ** (2 - _disc_covariogram(hh, R) / (np.pi * R * R)) - q * q),
                        hh) / (q * (1 - q))
    assert cs.A_px2 == pytest.approx(A_th, rel=0.15)


def test_iid_field_has_unit_integral_range_and_short_lc():
    rng = np.random.default_rng(4)
    img = rng.random((300, 300)) < 0.35
    tp = st.MaskedCorrelator(np.ones_like(img), 40, workers=1).two_point(img)
    cs = st.summarize_covariance(tp)
    assert cs.A_px2 == pytest.approx(1.0, abs=0.3)
    assert cs.lc_px < 1.0
    assert cs.converged


def test_first_crossing_and_integral_scale_on_exponential():
    r = np.arange(0, 200)
    a = 12.0
    rho = np.exp(-r / a)
    assert st.correlation_length(r, rho) == pytest.approx(a, rel=0.01)
    assert st.integral_scale(r, rho, 199) == pytest.approx(a, rel=0.02)
    assert np.isnan(st.first_crossing(r, rho, 0.0))
    assert st.first_crossing(np.array([0, 1, 2.0]), np.array([1, 0.5, -0.5]), 0.0) == pytest.approx(1.5)


def test_trend_is_flagged_as_not_converged():
    """A field with a large-scale trend never decorrelates inside the window."""
    n = 200
    yy, xx = np.mgrid[:n, :n]
    rng = np.random.default_rng(5)
    prob = 0.2 + 0.6 * xx / n
    img = rng.random((n, n)) < prob
    cs = st.summarize_covariance(st.MaskedCorrelator(np.ones_like(img), 40, workers=1).two_point(img))
    assert not cs.converged
    assert cs.A_px2 > 50


# --------------------------------------------------------------------------- tiling
def test_tile_grid_only_complete_tiles_and_bruteforce_fractions():
    rng = np.random.default_rng(6)
    mask = _disc_mask(101, r=48)
    pore = (rng.random(mask.shape) < 0.4) & mask
    for L in (8, 16, 33):
        tg = st.tile_grid(mask, L)
        v = st.tile_fractions(pore, tg)
        for y0, x0, f in zip(tg.y0, tg.x0, v):
            assert mask[y0:y0 + L, x0:x0 + L].all()
            assert f == pytest.approx(pore[y0:y0 + L, x0:x0 + L].mean())
        # tiles do not overlap
        cover = np.zeros(mask.shape, int)
        for y0, x0 in zip(tg.y0, tg.x0):
            cover[y0:y0 + L, x0:x0 + L] += 1
        assert cover.max() <= 1


def test_tile_count_full_square():
    mask = np.ones((96, 96), bool)
    assert st.tile_grid(mask, 32).n == 9
    assert st.tile_grid(mask, 40).n == 4
    assert st.tile_grid(mask, 97).n == 0


def test_tile_variance_of_iid_field_follows_binomial():
    rng = np.random.default_rng(7)
    p = 0.3
    img = rng.random((512, 512)) < p
    mask = np.ones_like(img)
    for L in (8, 16, 32):
        v = st.tile_fractions(img, st.tile_grid(mask, L))
        assert v.var(ddof=1) == pytest.approx(p * (1 - p) / L ** 2, rel=0.25)


def test_variance_scaling_fit_recovers_integral_range():
    L = np.array([16, 32, 64, 128], float)
    phi, A = 0.3, 400.0
    sd = np.sqrt(phi * (1 - phi) * A) / L
    out = st.fit_variance_scaling(L, sd, np.full(4, 50), phi)
    assert out["A_tile_px2"] == pytest.approx(A)
    assert out["slope"] == pytest.approx(-1.0)


def test_morans_i_signs():
    gi, gj = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    gi, gj = gi.ravel(), gj.ravel()
    checker = ((gi + gj) % 2).astype(float)
    assert st.morans_i(checker, gi, gj) == pytest.approx(-1.0)
    grad = gj.astype(float)
    assert st.morans_i(grad, gi, gj) > 0.7
    rng = np.random.default_rng(8)
    vals = [st.morans_i(rng.normal(size=64), gi, gj) for _ in range(300)]
    assert np.mean(vals) == pytest.approx(-1 / 63, abs=0.02)


# --------------------------------------------------------------------------- REV
def test_l_rev_theory_and_se_ir():
    phi, A, e = 0.25, 2500.0, 0.01
    z = st.z_value(0.95)
    assert z == pytest.approx(1.959964, abs=1e-5)
    L = st.l_rev_theory(phi, A, e, z)
    assert L == pytest.approx(z * np.sqrt(phi * (1 - phi) * A) / e)
    # a square of side L_REV has a half-width of exactly e
    assert z * st.se_integral_range(phi, A, L * L) == pytest.approx(e)
    # iid pixels: SE_IR equals the binomial SE
    assert st.se_integral_range(phi, 1.0, 1e6) == pytest.approx(np.sqrt(phi * (1 - phi) / 1e6))


def test_l_rev_empirical_interpolation_and_extrapolation():
    L = np.array([16, 32, 64, 128], float)
    sd = 0.32 / L          # SD = 0.32/L  ->  z SD = e at L = z*0.32/e
    n = np.full(4, 100)
    z = 1.959964
    Lr, s = st.l_rev_empirical(L, sd, n, e=0.01, z=z)
    assert s == "interpolated" and Lr == pytest.approx(z * 0.32 / 0.01, rel=1e-6)
    Lr, s = st.l_rev_empirical(L, sd * 10, n, e=0.01, z=z)
    assert s == "extrapolated" and Lr == pytest.approx(z * 3.2 / 0.01, rel=1e-6)
    Lr, s = st.l_rev_empirical(L, sd / 100, n, e=0.01, z=z)
    assert s == "<=" and Lr == 16
    # sizes with too few tiles are not trusted
    Lr, s = st.l_rev_empirical(L, sd, np.array([100, 100, 5, 5]), e=0.01, z=z)
    assert s == "extrapolated" and Lr == pytest.approx(32 * z * 0.01 / 0.01)


def test_choose_block_size_rule():
    sizes = [16, 32, 64, 96, 128, 192]
    nt = {16: 900, 32: 220, 64: 50, 96: 22, 128: 12, 192: 4}
    assert st.choose_block_size(sizes, nt, lc_px=10, b_min=64) == (64, "ok")
    assert st.choose_block_size(sizes, nt, lc_px=30, b_min=64) == (96, "ok")
    assert st.choose_block_size(sizes, nt, lc_px=40, b_min=64) == (96, "b<3lc")
    assert st.choose_block_size(sizes, nt, lc_px=np.nan, b_min=64) == (96, "lc_undefined")
    assert st.choose_block_size(sizes, {s: 3 for s in sizes}, lc_px=5, b_min=64)[1] == "too_few_blocks"


# --------------------------------------------------------------------------- resampling
def test_block_bootstrap_se_matches_sd_over_sqrt_n():
    rng = np.random.default_rng(9)
    v = rng.normal(0.3, 0.05, 50)
    out = st.block_bootstrap_mean(v, 20000, np.random.default_rng(10))
    se_plugin = v.std(ddof=0) / np.sqrt(len(v))      # bootstrap SE -> plug-in SE
    assert out["se"] == pytest.approx(se_plugin, rel=0.03)
    assert out["lo"] < v.mean() < out["hi"]
    assert (out["hi"] - out["lo"]) == pytest.approx(2 * 1.96 * se_plugin, rel=0.06)


def test_block_bootstrap_reproducible_with_same_generator_seed():
    v = np.linspace(0, 1, 30)
    a = st.block_bootstrap_mean(v, 1000, np.random.default_rng(11))
    b = st.block_bootstrap_mean(v, 1000, np.random.default_rng(11))
    assert a == b


def test_binomial_shortcut_equals_explicit_pixel_bootstrap():
    rng = np.random.default_rng(12)
    pix = rng.random(20000) < 0.27
    phi, N = pix.mean(), pix.size
    a = st.iid_bootstrap_binomial(phi, N, 4000, np.random.default_rng(13))
    b = st.iid_bootstrap_explicit(pix, 1500, np.random.default_rng(14))
    assert a["se_analytic"] == pytest.approx(np.sqrt(phi * (1 - phi) / N))
    assert a["se_boot"] == pytest.approx(a["se_analytic"], rel=0.05)
    assert b["se_boot"] == pytest.approx(a["se_analytic"], rel=0.07)
    # binomial quantiles (independent implementation: scipy)
    lo, hi = stats.binom.ppf([0.025, 0.975], N, phi) / N
    tol = 0.1 * a["se_analytic"]        # Monte-Carlo error of a 2.5 % quantile with 4000 draws
    assert a["lo"] == pytest.approx(lo, abs=tol)
    assert a["hi"] == pytest.approx(hi, abs=tol)


def test_pixel_iid_underestimates_correlated_field():
    """On a correlated field the pixel-iid SE is ~ sqrt(A) times too small."""
    rng = np.random.default_rng(15)
    reps, ses = [], []
    for _ in range(25):
        img = _boolean_discs(160, 0.004, 5.0, rng)
        reps.append(img.mean())
        ses.append(np.sqrt(img.mean() * (1 - img.mean()) / img.size))
    true_sd = np.std(reps, ddof=1)
    assert true_sd / np.mean(ses) > 3.0


# --------------------------------------------------------------------------- radial
def test_radial_profile_recovers_ring_porosity():
    n = 201
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - 100, xx - 100)
    mask = r <= 95
    pore = (r < 50) & mask                       # pore disc of radius 50
    prof = st.radial_profile(pore, mask, r, bin_px=10)
    assert np.allclose(prof["porosity"][prof["hi"] <= 50], 1.0)
    assert np.allclose(prof["porosity"][prof["lo"] >= 50], 0.0)
    assert prof["n"].sum() == mask.sum()
    # pixel-weighted mean of the rings equals the global porosity
    assert np.sum(prof["porosity"] * prof["n"]) / prof["n"].sum() == pytest.approx(pore[mask].mean())
