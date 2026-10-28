"""Unit tests of the core modules (run:  python -m pytest -q  from the repository root)."""
import numpy as np
import pytest
from skimage import filters

from poroseg import postprocess, thresholds
from poroseg.io import lab_reference
from poroseg.metrics import dice, jaccard, porosity

CFG = {"postprocessing": {"median_size": 3, "morph_se": "square", "morph_size": 3},
       "thresholds": {"nbins": 256, "sauvola": {"window": 25, "k": 0.2, "r": 0.5},
                      "niblack": {"window": 25, "k": 0.8}}}


def _phantom(seed=0, n=256, phi=0.3):
    """Random disks of pore (dark) in bright matrix + mild noise; returns (img, truth)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    truth = np.zeros((n, n), bool)
    while truth.mean() < phi:
        cy, cx, r = rng.uniform(0, n), rng.uniform(0, n), rng.uniform(4, 12)
        truth |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img = np.where(truth, 0.2, 0.8) + rng.normal(0, 0.03, truth.shape)
    return np.clip(img, 0, 1).astype(np.float32), truth


def test_masked_local_stats_equal_skimage_in_interior():
    img, _ = _phantom()
    mask = np.ones_like(img, bool)
    w = 25
    ts = thresholds.sauvola_map(img, mask, w, 0.2, 0.5)
    tn = thresholds.niblack_map(img, mask, w, 0.8)
    ref_s = filters.threshold_sauvola(img.astype(np.float64), window_size=w, k=0.2, r=0.5)
    ref_n = filters.threshold_niblack(img.astype(np.float64), window_size=w, k=0.8)
    inner = (slice(w, -w), slice(w, -w))
    assert np.allclose(ts[inner], ref_s[inner], atol=1e-6)
    assert np.allclose(tn[inner], ref_n[inner], atol=1e-6)


def test_masked_stats_ignore_outside_pixels():
    img, _ = _phantom()
    mask = np.zeros_like(img, bool)
    mask[:, :128] = True
    corrupted = img.copy()
    corrupted[~mask] = 1e6            # anything outside the mask must not matter
    a = thresholds.sauvola_map(img, mask, 25)
    b = thresholds.sauvola_map(corrupted, mask, 25)
    assert np.allclose(a[mask], b[mask])


@pytest.mark.parametrize("method", list(thresholds.METHOD_INFO))
def test_every_method_recovers_clean_bimodal_phantom(method):
    img, truth = _phantom(seed=1)
    mask = np.ones_like(img, bool)
    p = thresholds.default_params(method, CFG)
    T = thresholds.compute_threshold(method, img, mask, p)
    pore = thresholds.segment(img, T, mask)
    if method == "Niblack":
        # Pathology documented in the paper: inside any region wider than the window
        # (matrix OR pore) Niblack labels a fixed fraction Phi(-k) of the noise as
        # pore, so the estimate ~ Phi(-0.8) = 0.21 whatever the true porosity.
        from scipy.stats import norm
        assert abs(porosity(pore, mask) - norm.cdf(-0.8)) < 0.04
    else:
        assert abs(porosity(pore, mask) - truth.mean()) < 0.05


def test_global_threshold_uses_only_roi_histogram():
    img, _ = _phantom(seed=2)
    mask = np.zeros_like(img, bool)
    mask[64:192, 64:192] = True
    polluted = img.copy()
    polluted[~mask] = 0.0          # "air" outside the ROI
    t1 = thresholds.compute_threshold("Otsu", img, mask, {"nbins": 256}).value
    t2 = thresholds.compute_threshold("Otsu", polluted, mask, {"nbins": 256}).value
    assert t1 == pytest.approx(t2)


def test_postprocessing_acts_on_pore_phase():
    pore = np.zeros((50, 50), bool)
    pore[10:30, 10:30] = True      # a 20x20 pore survives opening
    pore[40, 40] = True            # an isolated 1-px pore is removed by opening
    out = postprocess.apply(pore, "open_close", CFG)
    assert out[20, 20] and not out[40, 40]
    med = postprocess.apply(pore, "median", CFG)
    assert not med[40, 40] and med[20, 20]


def test_skeleton_is_a_thin_subset_of_pores_not_inverted():
    pore = np.zeros((60, 60), bool)
    pore[20:40, 5:55] = True
    sk = postprocess.apply(pore, "skeleton", CFG)
    assert sk.sum() > 0
    assert np.all(pore[sk])                     # skeleton lies inside the pore phase
    assert sk.mean() < 0.1 * pore.mean()        # and is thin (legacy code returned ~1 - skeleton)


def test_overlap_metrics():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[:5, :5] = True
    m = np.ones_like(a)
    assert jaccard(a, b, m) == pytest.approx(0.5)
    assert dice(a, b, m) == pytest.approx(2 / 3)


def test_lab_reference_excludes_uct():
    import pandas as pd
    lab = pd.DataFrame(dict(sample_id=["S"] * 5, specimen=["S"] * 5, has_image=[True] * 5,
                            method=["direct", "gravimetric", "indirect", "gas", "uct"],
                            porosity_pct=[40.0, 40.0, 38.0, 38.0, 30.0]))
    r = lab_reference(lab, ["direct", "gravimetric", "indirect", "gas"]).iloc[0]
    assert r["ref"] == pytest.approx(0.39)
    assert r["ref_n"] == 4 and r["uct"] == pytest.approx(0.30)
    assert r["mean_all5"] == pytest.approx(0.372)
