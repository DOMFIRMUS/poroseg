"""Spatial uncertainty of a slice porosity: block bootstrap, integral range and the naive pixel bootstrap.

Run from the repository root::

    python examples/03_spatial_uncertainty.py                                   # DS23_CF5, Otsu
    python examples/03_spatial_uncertainty.py --sample AG24_VN04 --method Sauvola --post median

For one segmentation of one configured sample (data/samples.csv) the script computes, as the ``spatial``
stage of the pipeline does (Section 2.3 of the article):

* the two-point statistics of the pore phase inside the disc ROI, the correlation length l_c (1/e) and the
  integral range A (Eq. 2);
* the side L_REV of a single square window whose porosity has a 95 % half-width of +/-1 percentage point,
  from the integral range (Eq. 3) and from the standard deviation of tile porosity;
* the 95 % interval of the ROI porosity from a block bootstrap of tiles at least three correlation lengths
  wide, from the integral-range standard error, and from the pixel bootstrap that treats every pixel as an
  independent replicate (which understates the uncertainty).

It uses ``poroseg.context.Context``, the object that every pipeline stage uses, so ROI detection,
normalisation, thresholds and random-number streams are exactly those of the article. Outputs go to
``examples/outputs/`` (a CSV with the tile statistics).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # run from a source checkout without installing the package

import numpy as np
import pandas as pd

from poroseg import spatial_tools as st
from poroseg.config import load_config
from poroseg.context import Context

MIN_BLOCKS, MIN_TILES = 20, 10        # as in poroseg/stages/spatial.py


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default="DS23_CF5")
    ap.add_argument("--method", default="Otsu")
    ap.add_argument("--post", default="none", choices=("none", "median", "open_close"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = ROOT / "examples" / "outputs"
    cfg["paths"]["output_dir"] = str(out / "context")          # keep the article's outputs/ untouched
    ctx = Context(cfg, samples=[a.sample], stage="example")
    sid, roi = a.sample, ctx.primary_roi
    R = ctx.roi(sid, roi)
    E = R.eval_mask
    N = int(E.sum())
    px_mm = ctx.pixel_size_um(sid) / 1000.0
    pore = ctx.pore(sid, a.method, a.post, roi) & E
    phi = pore.sum() / N

    # two-point statistics (lags up to 1/3 of the ROI side, at most 256 px)
    ys, xs = np.nonzero(E.any(axis=1))[0], np.nonzero(E.any(axis=0))[0]
    max_lag = int(min(256, np.floor(min(ys[-1] - ys[0] + 1, xs[-1] - xs[0] + 1) / 3)))
    cs = st.summarize_covariance(st.MaskedCorrelator(E, max_lag).two_point(pore, phi))

    # tiles, representative window and block bootstrap
    sp = cfg["spatial"]
    sizes = sorted(int(s) for s in sp["tile_sizes_px"])
    e = float(sp["target_abs_error_pp"]) / 100.0
    z = st.z_value(float(cfg["statistics"]["ci_level"]))
    tiles = {L: st.tile_grid(E, L) for L in sizes}
    vals = {L: st.tile_fractions(pore, tiles[L]) for L in sizes}
    sd = np.array([v.std(ddof=1) if len(v) > 1 else np.nan for v in vals.values()])
    n = np.array([len(v) for v in vals.values()])
    L_rev_ir = st.l_rev_theory(phi, cs.A_px2, e, z) if np.isfinite(cs.A_px2) else np.nan
    L_rev_tiles, status = st.l_rev_empirical(np.array(sizes), sd, n, e, z, MIN_TILES)
    b, rule = st.choose_block_size(sizes, {L: tiles[L].n for L in sizes}, cs.lc_px, int(sp["block_tile_px"]),
                                   MIN_BLOCKS)
    bb = st.block_bootstrap_mean(vals[b], ctx.n_boot, ctx.rng("spatial", "block", sid, roi, a.method, a.post))
    se_ir = st.se_integral_range(phi, cs.A_px2, N) if np.isfinite(cs.A_px2) else np.nan
    iid = st.iid_bootstrap_binomial(phi, N, int(sp["n_boot_pixels"]),
                                    ctx.rng("spatial", "iid", sid, roi, a.method, a.post))
    hw_block = 100 * (bb["hi"] - bb["lo"]) / 2
    hw_iid = 100 * (iid["hi"] - iid["lo"]) / 2

    print(f"{ctx.label(sid)}, {a.method}, post-processing '{a.post}': porosity {100 * phi:.2f} % "
          f"over {N} pixels ({N * px_mm ** 2:.0f} mm2)")
    print(f"correlation length l_c = {cs.lc_px * px_mm * 1000:.0f} um; integral range A = {cs.A_px2 * px_mm ** 2:.3f} mm2"
          + ("" if cs.converged else " (not converged: large-scale trend, a lower bound)"))
    print(f"window for +/-{100 * e:.0f} pp: L_REV = {L_rev_ir * px_mm:.0f} mm (integral range), "
          f"{L_rev_tiles * px_mm:.0f} mm (tiles, {status}); ROI diameter {2 * ctx.geometry(sid).r_usable * px_mm:.1f} mm")
    print(f"95 % half-width of the ROI porosity: block bootstrap {hw_block:.2f} pp ({bb['n']} tiles of {b} px, "
          f"rule '{rule}'), integral range {100 * z * se_ir:.2f} pp, pixel bootstrap {hw_iid:.3f} pp")
    print(f"the pixel bootstrap interval is {hw_block / hw_iid:.0f} times narrower than the block bootstrap interval")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(L_px=sizes, L_mm=np.array(sizes) * px_mm, n_tiles=n, sd_tile_porosity=sd)).to_csv(
        out / f"spatial_{sid}_{a.method}_{a.post}.csv", index=False)
    print("wrote", out / f"spatial_{sid}_{a.method}_{a.post}.csv")


if __name__ == "__main__":
    main()
