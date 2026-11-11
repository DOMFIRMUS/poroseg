"""Quick start: segment a synthetic micro-CT slice with known truth using the eight thresholds.

Run from the repository root (the folder that contains ``config.yaml``)::

    python examples/01_quick_start_synthetic.py
    python examples/01_quick_start_synthetic.py --kind boolean_pores --blur 3 --noise 0.10 --cupping 0.15

No input data are needed. The script draws a phantom with known structure (``grain_pack``, the analogue of the
granular packs of the article; ``boolean_pores``; or ``multiscale``, which has sub-resolution pores), degrades it
with blur, noise and beam-hardening cupping, normalises it exactly as the pipeline normalises a real slice,
segments it with the eight algorithms and the three area-preserving post-processing levels, and scores every
segmentation against the truth. It takes a few seconds and writes
``examples/outputs/quick_start_synthetic.csv`` and ``.png``.
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
import pandas as pd

from poroseg import synthetic
from poroseg.config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", default="grain_pack", choices=synthetic.KINDS)
    ap.add_argument("--blur", type=float, default=1.5, help="s.d. of the Gaussian point-spread function (px)")
    ap.add_argument("--noise", type=float, default=0.05, help="noise s.d. relative to the phase contrast")
    ap.add_argument("--cupping", type=float, default=0.15, help="relative darkening of the centre")
    ap.add_argument("--size", type=int, default=512, help="image side (px)")
    a = ap.parse_args(argv)

    cfg = load_config(ROOT / "config.yaml")
    case = synthetic.generate_case(a.kind, a.blur, a.noise, a.cupping, rep=0, seed=cfg["project"]["seed"],
                                   size=a.size)
    segs = synthetic.segment_all(case, cfg)          # {(method, post): bool pore image}

    rows = []
    for method in cfg["thresholds"]["methods"]:
        for post in ("none", "median", "open_close"):
            m = synthetic.segmentation_metrics(segs[(method, post)], case["truth"], case["eval_mask"],
                                               case["phi_pixel"], case["phi_total"])
            rows.append(dict(method=method, post=post, porosity_pct=100 * m["porosity"],
                             error_pp=m["error_pp"], dice=m["dice"]))
    df = pd.DataFrame(rows)

    print(f"phantom: {a.kind}; blur {a.blur} px, noise {a.noise}, cupping {a.cupping}")
    print(f"true porosity: {100 * case['phi_pixel']:.1f} % on the pixel grid, {100 * case['phi_total']:.1f} % in total")
    print(df[df.post == "none"].drop(columns="post").round(2).to_string(index=False))
    best = df[df.post == "none"].assign(abs_err=lambda d: d.error_pp.abs()).sort_values("abs_err").iloc[0]
    print(f"smallest absolute error without post-processing: {best.method} ({best.error_pp:+.2f} pp)")

    out = ROOT / "examples" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "quick_start_synthetic.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    panels = [(case["grey"], "degraded image"), (case["truth"], "truth (pore = white)"),
              (segs[("Otsu", "none")], "Otsu"), (segs[("Niblack", "none")], "Niblack")]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "quick_start_synthetic.png", dpi=150)
    print("wrote", out / "quick_start_synthetic.csv", "and", out / "quick_start_synthetic.png")


if __name__ == "__main__":
    main()
