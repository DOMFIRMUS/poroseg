"""Unit tests of the pure functions of the synthetic, morphology, sensitivity,
inference, ml_baseline and legacy stages (known answers on small synthetic data)."""
import math
from pathlib import Path

import numpy as np
import pytest

from poroseg import morpho_tools as mt
from poroseg import synthetic as syn
from poroseg.config import load_config
from poroseg.stages import inference, legacy, ml_baseline, sensitivity

CFG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")


# ----------------------------------------------------------------------------- synthetic
@pytest.mark.parametrize("kind,lo,hi", [("grain_pack", 0.25, 0.50), ("boolean_pores", 0.05, 0.30),
                                         ("multiscale", 0.05, 0.45)])
def test_phantom_porosity_in_design_range(kind, lo, hi):
    ph = syn.make_phantom(kind, 256, np.random.default_rng(1), supersample=2)
    assert lo <= ph["phi_total"] <= hi
    assert ph["truth"].dtype == bool and ph["truth_frac"].min() >= 0 and ph["truth_frac"].max() <= 1


def test_generate_case_is_deterministic_and_normalised():
    a = syn.generate_case("boolean_pores", 1.5, 0.05, 0.15, 0, 123, size=128, supersample=2)
    b = syn.generate_case("boolean_pores", 1.5, 0.05, 0.15, 0, 123, size=128, supersample=2)
    assert np.array_equal(a["grey"], b["grey"])
    g = a["grey"][a["mask"]]
    assert g.min() >= 0 and g.max() <= 1


def test_perfect_segmentation_metrics():
    case = syn.generate_case("grain_pack", 0.5, 0.02, 0.0, 0, 7, size=128, supersample=2)
    m = syn.segmentation_metrics(case["truth"], case["truth"], case["eval_mask"], case["phi_pixel"])
    assert m["dice"] == pytest.approx(1.0) and m["abs_error_pp"] == pytest.approx(0.0, abs=1e-9)
    assert m["euler_rel_error"] == 0


def test_multiscale_total_exceeds_pixel_porosity():
    case = syn.generate_case("multiscale", 1.5, 0.05, 0.0, 0, 11, size=128, supersample=4)
    assert case["phi_total"] > case["phi_pixel"]


# ----------------------------------------------------------------------------- morphology
def _disc(n=201, r=60):
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(yy - n // 2, xx - n // 2) <= r


def test_euler_of_disc_and_annulus():
    d = _disc()
    assert mt.euler_quads(d, None, 8)[0] == pytest.approx(1.0)
    ring = d & ~_disc(r=30)
    assert mt.euler_quads(ring, None, 8)[0] == pytest.approx(0.0)


def test_crofton_boundary_density_of_disc():
    n, r = 401, 100
    d = _disc(n, r)
    W = np.ones_like(d)
    LA = mt.crofton_boundary_density(d, W)["L_A"]
    assert LA * n * n == pytest.approx(2 * math.pi * r, rel=0.03)


def test_granulometry_of_single_disc():
    d = _disc(301, 40)
    W = np.ones_like(d)
    q = mt.psd_quantiles(mt.DEFAULT_RADII, mt.granulometry(d, W))
    assert q["d50"] == pytest.approx(80, rel=0.15)


def test_phi_pv_linear_model():
    assert float(mt.phi_pv(0.7 * 100 + 0.3 * 10, 100, 10)) == pytest.approx(0.3)


# ----------------------------------------------------------------------------- sensitivity
def test_psi_bounds_and_values():
    assert sensitivity.psi_index(0.2, 0.2, 0.2) == pytest.approx(1.0)
    assert sensitivity.psi_index(0.2, 0.3, 0.1) == pytest.approx(0.5)
    assert sensitivity.psi_index(0.1, 0.9, 0.0) == pytest.approx(0.0)


def test_niblack_false_porosity_matches_normal_quantile():
    rng = np.random.default_rng(3)
    img = sensitivity.homogeneous_field(512, 0.7, 0.03, rng)
    phi, _ = sensitivity.field_porosity(img, "Niblack", CFG)
    k = CFG["thresholds"]["niblack"]["k"]
    assert phi == pytest.approx(sensitivity.niblack_false_porosity_analytic(k), abs=0.02)


# ----------------------------------------------------------------------------- inference
def test_agreement_of_identical_measurements():
    x = np.array([10., 20, 30, 40, 25, 35, 15])
    a = inference._agreement_one(x, x + 0.0, np.full(7, 0.5), np.full(7, 0.5), np.random.default_rng(0), 200)
    assert a["bias"] == pytest.approx(0) and a["mae"] == pytest.approx(0)
    assert a["ccc"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- ml baseline
def test_feature_stack_and_balanced_sample():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    F = ml_baseline.features(img)
    assert F.shape[:2] == (64, 64) and F.dtype == np.float32 and F.shape[2] >= 15
    lab = np.zeros(64 * 64, bool)
    lab[:100] = True
    idx = ml_baseline._balanced_sample(lab, np.ones_like(lab), 50, np.random.default_rng(1))
    assert lab[idx].sum() == 50 and (~lab[idx]).sum() == 50


# ----------------------------------------------------------------------------- legacy
def test_legacy_crop_and_inverted_skeleton():
    rng = np.random.default_rng(2)
    raw = rng.integers(1000, 3000, (620, 600)).astype(np.float32)
    raw[200:420, 150:450] -= 900                       # a dark 'pore'
    crop = legacy.legacy_crop(raw, 500)
    assert crop.shape == (500, 500) and crop[0, 0] == raw[60, 50]
    L = legacy.legacy_process(crop)
    b = L[("Otsu", "_bin")]
    from skimage.morphology import skeletonize
    density = skeletonize(b == 0).mean()
    assert L[("Otsu", "Skeletonization")] == pytest.approx(1 - density)
