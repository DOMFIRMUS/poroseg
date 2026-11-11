"""Segment one micro-CT slice with the factorial design of the article and measure threshold stability.

Run from the repository root::

    python examples/02_segment_one_slice.py                          # sample DS23_CF5 (a granular pack)
    python examples/02_segment_one_slice.py --sample AG24_VN04       # any sample_id of data/samples.csv
    python examples/02_segment_one_slice.py --image path/to/slice.tif --holder free    # your own slice

For one 2-D slice (TIFF, any bit depth) the script detects the usable disc region of interest (ROI),
normalises the grey levels inside it, applies the eight thresholding algorithms and the three
area-preserving post-processing levels, and prints

* the porosity of every combination (disc ROI, 2-px rim excluded), and
* the dimensionless Pore Stability Index of every algorithm, PSI = 1 - min(1, |phi(T+d) - phi(T-d)| / (2 phi(T)))
  with d = 0.05 of the normalised grey range (Eq. 5 of the article; 1 = insensitive to threshold shifts).

It writes ``examples/outputs/segment_<name>.csv`` and a histogram with the eight thresholds
(``segment_<name>_histogram.png``). For one of the article's samples the porosities equal the
``none``/``median``/``open_close`` rows of the article's factorial table for that sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # run from a source checkout without installing the package

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poroseg import io as pio
from poroseg import metrics, postprocess, roi as proi, thresholds
from poroseg.config import load_config


def stability_index(img, mask, eval_mask, T, delta=0.05):
    """Pore Stability Index of a global threshold (float) or of a local threshold map (array)."""
    phi = lambda t: metrics.porosity((img <= t) & mask, eval_mask)
    p0 = phi(T)
    if not p0 > 0:
        return float("nan")
    return 1.0 - min(1.0, abs(phi(T + delta) - phi(T - delta)) / (2.0 * p0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", default="DS23_CF5", help="sample_id in data/samples.csv")
    ap.add_argument("--image", default=None, help="any 2-D TIFF instead of a configured sample")
    ap.add_argument("--holder", default="free", choices=("free", "container"),
                    help="with --image: 'free' (free-standing core in air) or 'container' (loose material in a holder)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    override = None
    if a.image:
        path, holder, name = Path(a.image), a.holder, Path(a.image).stem
    else:
        samples = pio.load_samples(cfg["paths"]["samples_csv"])
        row = samples.loc[a.sample]
        path, holder, name = Path(cfg["paths"]["images_dir"]) / row["file"], row["holder"], a.sample
        override = {k: float(row[k]) if pd.notna(row.get(k)) else np.nan for k in ("roi_cx", "roi_cy", "roi_r")}

    raw = pio.read_tiff(path, name)
    geom = proi.detect_geometry(raw.data, holder, cfg, override, seed=int(cfg["project"]["seed"]))
    R = proi.disk_roi(raw.shape, geom, int(cfg["roi"]["eval_border_px"]))
    nrm = cfg["normalization"]
    img, lo, hi = thresholds.normalize(raw.data[R.slices], R.mask, nrm["p_low"], nrm["p_high"])
    img[~R.mask] = 0.0
    print(f"{path.name}: {raw.shape[1]} x {raw.shape[0]} px, {raw.dtype}; disc ROI centre "
          f"({geom.cx:.0f}, {geom.cy:.0f}) px, radius {geom.r_usable:.0f} px ({geom.source})")

    rows, marks = [], {}
    for method in cfg["thresholds"]["methods"]:
        T = thresholds.compute_threshold(method, img, R.mask, thresholds.default_params(method, cfg))
        marks[method] = T.summary
        pore = thresholds.segment(img, T, R.mask)
        rec = dict(method=method, family=T.family, threshold=T.summary,
                   psi=stability_index(img, R.mask, R.eval_mask, T.value))
        for post in ("none", "median", "open_close"):
            p = pore if post == "none" else postprocess.apply(pore, post, cfg, R.mask)
            rec[f"porosity_{post}_pct"] = 100.0 * metrics.porosity(p, R.eval_mask)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    rng = df["porosity_none_pct"].max() - df["porosity_none_pct"].min()
    print(f"range across the eight algorithms (no post-processing): {rng:.1f} percentage points")

    out = ROOT / "examples" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"segment_{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(img[R.eval_mask], bins=256, range=(0, 1), color="0.6", log=True)
    for k, (method, t) in enumerate(marks.items()):
        ax.axvline(t, color=plt.cm.tab10(k), lw=1.2, label=method)
    ax.set_xlabel("normalised grey level inside the disc ROI")
    ax.set_ylabel("pixels (log)")
    ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"segment_{name}_histogram.png", dpi=150)
    print("wrote", out / f"segment_{name}.csv", "and", out / f"segment_{name}_histogram.png")


if __name__ == "__main__":
    main()
