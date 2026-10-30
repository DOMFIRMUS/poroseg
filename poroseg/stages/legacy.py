"""Stage 'legacy': quantified audit of the original analysis notebook.

The original notebook ("codigo final interpore", 2025; not distributed, available
from the authors) produced the numbers of the conference abstracts. This stage re-implements its processing
faithfully (500 x 500 px central crop of the raw uint16 image, scikit-image
thresholds on the raw crop, Sauvola with the default dynamic range r,
Niblack k = 0.8, ``binarised = 255 where img > T``, porosity = mean(binarised
== 0); post-processing on the 0/255 image: grey opening then closing with a
2 x 2 footprint, ``skeletonize(binary == 0)`` written back as 255, 3 x 3
median filter), checks that it reproduces the stored legacy results, and
quantifies every defect against the corrected pipeline. Outputs:
tables legacy_audit and legacy_vs_corrected, figure legacy_vs_corrected and the
report outputs/LEGACY_AUDIT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import (threshold_isodata, threshold_li, threshold_mean, threshold_niblack, threshold_otsu,
                             threshold_sauvola, threshold_triangle, threshold_yen)
from skimage.morphology import closing, footprint_rectangle, opening, skeletonize

from .. import style

LEGACY_T = {
    "Otsu": threshold_otsu, "Li": threshold_li, "Yen": threshold_yen, "Mean": threshold_mean,
    "Triangle": threshold_triangle, "Isodata": threshold_isodata,
    "Sauvola": lambda img: threshold_sauvola(img, window_size=25),
    "Niblack": lambda img: threshold_niblack(img, window_size=25, k=0.8),
}
LEGACY_POST = {
    "Binarized": lambda b: b,
    "Enhanced (open-close 2x2 on solid)": lambda b: closing(opening(b, footprint_rectangle((2, 2))), footprint_rectangle((2, 2))),
    "Skeletonization": lambda b: np.where(skeletonize(b == 0), 255, 0).astype(np.uint8),
    "Median Filter": lambda b: ndi.median_filter(b, size=3),
}


def legacy_crop(raw: np.ndarray, size: int = 500) -> np.ndarray:
    """PIL ``img.crop((left, top, left+size, top+size))`` with left = (W-size)//2, top = (H-size)//2."""
    h, w = raw.shape
    left, top = (w - size) // 2, (h - size) // 2
    return raw[top:top + size, left:left + size].astype(np.uint16)


def legacy_process(crop: np.ndarray) -> dict:
    """Porosity of every legacy method x refinement exactly as in the notebook."""
    out = {}
    for m, f in LEGACY_T.items():
        T = f(crop)
        b = np.where(crop > T, 255, 0).astype(np.uint8)
        for r, g in LEGACY_POST.items():
            out[(m, r)] = float(np.mean(g(b) == 0))
        out[(m, "threshold")] = float(np.mean(T))
        out[(m, "_bin")] = b
    return out


def run(ctx) -> None:
    root = Path(ctx.cfg["_root"])
    legacy_dir = Path(ctx.cfg["paths"]["legacy_results_dir"])
    rng = ctx.rng("legacy")
    rows, audit = [], []
    stored = pd.read_csv(legacy_dir / "global_analysis" / "pivoted_data.csv").set_index("Image")
    seg = ctx.seg
    res = {}
    for sid in ctx.sample_ids:
        crop = legacy_crop(ctx.raw(sid).data)
        L = legacy_process(crop)
        res[sid] = (crop, L)
        fname = ctx.samples.at[sid, "file"]
        for m in ctx.methods:
            sk_density = 1.0 - L[(m, "Skeletonization")]
            cc = seg[(seg["sample_id"] == sid) & (seg["method"] == m) & (seg["roi"] == "center500")].set_index("post")["porosity"]
            dk = seg[(seg["sample_id"] == sid) & (seg["method"] == m) & (seg["roi"] == ctx.primary_roi)].set_index("post")["porosity"]
            rows.append(dict(sample_id=sid, label=ctx.label(sid), method=m,
                             legacy_stored=float(stored.at[fname, m]) if fname in stored.index else np.nan,
                             legacy_reproduced=L[(m, "Binarized")],
                             legacy_open_close=L[(m, "Enhanced (open-close 2x2 on solid)")],
                             legacy_median=L[(m, "Median Filter")],
                             legacy_skeleton_porosity=L[(m, "Skeletonization")], skeleton_density=sk_density,
                             corrected_center500=float(cc["none"]), corrected_center500_open_close=float(cc["open_close"]),
                             corrected_disk=float(dk["none"]), legacy_threshold=L[(m, "threshold")]))
    lv = pd.DataFrame(rows)
    ctx.save_data(lv, "legacy_vs_corrected")
    lv["repro_diff"] = (lv["legacy_reproduced"] - lv["legacy_stored"]).abs()
    max_diff = float(lv["repro_diff"].max())

    # ------------------------------------------------------------------ defect quantification
    K = ctx.add_key
    K("legacy_reproduction_max_abs_diff", max_diff, "largest |reproduced - stored| legacy porosity over 9 images x 8 methods")
    K("legacy_reproduction_n_exact", int((lv["repro_diff"] < 1e-6).sum()), "legacy porosities reproduced to 1e-6 (of 72)")
    # 1 skeleton inversion
    gain = 100 * (lv["legacy_skeleton_porosity"] - lv["legacy_reproduced"]) / lv["legacy_reproduced"].where(lv["legacy_reproduced"] > 0)
    audit.append(dict(issue="Skeleton written as 255 and porosity computed as mean(img == 0)",
                      location="apply_skeleton / process_image",
                      consequence="the 'skeletonised porosity' is 1 − skeleton density, i.e. close to 100 %, not a porosity",
                      impact=f"legacy 'skeleton porosity' {100 * lv['legacy_skeleton_porosity'].min():.1f}–"
                             f"{100 * lv['legacy_skeleton_porosity'].max():.1f} %; true skeleton density "
                             f"{100 * lv['skeleton_density'].min():.2f}–{100 * lv['skeleton_density'].max():.2f} %",
                      fix="skeleton reported only as medial-axis length density (morphology stage), never as porosity"))
    K("legacy_skeleton_porosity_range", [float(lv["legacy_skeleton_porosity"].min()), float(lv["legacy_skeleton_porosity"].max())],
      "range of the legacy 'skeletonised porosity' (= 1 - skeleton density)", "fraction")
    K("legacy_skeleton_relative_gain_pct_median", float(np.nanmedian(gain)),
      "median relative 'porosity increase by skeletonisation' that the legacy code would report", "%")
    # 2 morphology on the solid phase
    d_leg = 100 * (lv["legacy_open_close"] - lv["legacy_reproduced"])
    d_cor = 100 * (lv["corrected_center500_open_close"] - lv["corrected_center500"])
    audit.append(dict(issue="Opening/closing applied to the 255-valued (solid) phase with a 2 × 2 element",
                      location="apply_enhancement",
                      consequence="the operation smooths the matrix instead of the pores (opening of the solid creates "
                                  "porosity, closing of the solid removes it) and the 2 × 2 element is not centred",
                      impact=f"porosity change by the legacy operation: median {np.median(d_leg):+.2f} pp (range "
                             f"{d_leg.min():+.1f} to {d_leg.max():+.1f}); by the corrected pore-phase 3 × 3 open-close on "
                             f"the same window: median {np.median(d_cor):+.2f} pp (range {d_cor.min():+.1f} to {d_cor.max():+.1f})",
                      fix="pore-phase opening then closing with a centred 3 × 3 element (postprocess.apply)"))
    K("legacy_open_close_effect_pp", dict(legacy_median=float(np.median(d_leg)), legacy_min=float(d_leg.min()),
                                          legacy_max=float(d_leg.max()), corrected_median=float(np.median(d_cor)),
                                          corrected_min=float(d_cor.min()), corrected_max=float(d_cor.max())),
      "porosity change produced by open-close: legacy (solid phase, 2x2) vs corrected (pore phase, 3x3), central window", "pp")
    # 3 bootstrap on the raw image and pixel-iid bootstrap
    raw_zero = [float(np.mean(res[s][0] == 0)) for s in ctx.sample_ids]
    audit.append(dict(issue="bootstrap_uncertainty() resamples the RAW grey image and counts pixels equal to 0",
                      location="main loop: bootstrap_uncertainty(img_array)",
                      consequence="the reported 95 % CI is the frequency of grey level 0 in the raw image (≈ 0), "
                                  "unrelated to the segmented porosity",
                      impact=f"fraction of raw pixels equal to 0 in the 9 crops: max {max(raw_zero):.2e}; the CI is [0, 0]",
                      fix="spatial uncertainty from a block bootstrap of tiles and the integral range (spatial stage)"))
    try:
        sp = ctx.read_data("spatial_ci")
        r = sp[(sp["roi"] == "center500") & (sp["post"] == "none")]["ci_width_ratio"]
        rr = f"even on the binary image the pixel-iid bootstrap understates the 95 % CI width by a factor {r.min():.0f}–{r.max():.0f}"
    except FileNotFoundError:
        rr = "see spatial stage"
    audit.append(dict(issue="method_uncertainty(): pixel-iid bootstrap of the binary image",
                      location="method_uncertainty", consequence="pseudo-replication: ~250 000 spatially correlated pixels "
                      "treated as independent replicates", impact=rr, fix="block bootstrap; image as experimental unit"))
    audit.append(dict(issue="Block porosity map computed on the raw image (blocks == 0)",
                      location="Figure 4: view_as_blocks(img_array)", consequence="the map is ≈ 0 everywhere",
                      impact=f"max block 'porosity' = {max(raw_zero):.2e}", fix="tile porosity of the binary image"))
    # 4 Sauvola r
    sv = lv[lv["method"] == "Sauvola"]
    audit.append(dict(issue="Sauvola with the default r (half the uint16 range, 32 767.5) on raw 16-bit data",
                      location="BINARIZATION_METHODS['Sauvola']",
                      consequence="s/r ≈ 0, so T ≈ m(1 − k): only the darkest pixels become pore",
                      impact=f"legacy Sauvola porosity {100 * sv['legacy_reproduced'].min():.1f}–{100 * sv['legacy_reproduced'].max():.1f} % "
                             f"vs corrected (r = 0.5 of the normalised range) {100 * sv['corrected_center500'].min():.1f}–"
                             f"{100 * sv['corrected_center500'].max():.1f} % on the same window",
                      fix="images normalised to [0, 1] inside the ROI; r = 0.5"))
    # 5 statistics
    audit.append(dict(issue="Paired t-tests between methods per image on the 4 refinement values",
                      location="statistical_comparison (ttest_rel)",
                      consequence="n = 4 non-independent values (one of them the inverted skeleton) per test; pseudo-replication",
                      impact="all such p-values are invalid", fix="Friedman / RM-ANOVA / exact pairwise tests with the image as unit"))
    audit.append(dict(issue="Tukey HSD across methods on pooled image values",
                      location="tukey_hsd_test", consequence="ignores the repeated-measures structure (same images under every method)",
                      impact="between-image variance inflates the error term", fix="RM-ANOVA with GG correction; Friedman"))
    icc_path = legacy_dir / "global_analysis" / "icc_results.csv"
    icc_v = float(pd.read_csv(icc_path).iloc[0, -1]) if icc_path.exists() else np.nan
    audit.append(dict(issue="ICC(3,k) (consistency) reported as agreement between methods",
                      location="intraclass_correlation", consequence="ICC(3,k) ignores systematic offsets between methods; "
                      "the stored value is negative", impact=f"stored ICC(3,k) = {icc_v:.4f}",
                      fix="variance decomposition; absolute-agreement ICC(2,1) when needed"))
    audit.append(dict(issue="Bland-Altman limits with np.std (ddof = 0) and mean of the two methods on the x axis",
                      location="bland_altman_analysis", consequence="limits of agreement too narrow by sqrt((n−1)/n); x should "
                      "be the reference (Krouwer 2008)", impact="with n = 7 the SD is underestimated by 7.4 %",
                      fix="ddof = 1, exact CIs of the limits (Carkeet 2015)"))
    try:
        lr = ctx.lab_ref
        d5 = 100 * (lr["mean_all5"] - lr["ref"]).dropna()
        imp = f"legacy reference lower than the physical mean by {d5.min():.2f} to {d5.max():.2f} pp"
    except Exception:  # noqa: BLE001
        imp = ""
    audit.append(dict(issue="Laboratory reference = mean of five columns including the laboratory µCT value",
                      location="validation section", consequence="partly circular comparison (µCT against a reference that "
                      "contains µCT)", impact=imp, fix="reference = mean of the four physical methods; µCT analysed separately"))
    cons = []
    for sid in ctx.sample_ids:
        L = res[sid][1]
        for m in ctx.methods:
            vals = [L[(m, r)] for r in LEGACY_POST]
            b0 = L[(m, "Binarized")]
            cons.append((1 - np.std(vals) / b0) * 100 if b0 > 0 else np.nan)
    cons = np.array(cons, float)
    audit.append(dict(issue="'Consistency (%)' = (1 − SD(refinement porosities)/binarised porosity) × 100",
                      location="stat_table", consequence="dominated by the inverted skeleton (≈ 100 %) and unbounded below "
                      "when the binarised porosity is small", impact=f"range over 9 images × 8 methods: {np.nanmin(cons):.0f} to "
                      f"{np.nanmax(cons):.0f} %", fix="removed; 1 − CV of tile porosity reported only as a heterogeneity index"))
    K("legacy_consistency_pct_range", [float(np.nanmin(cons)), float(np.nanmax(cons))],
      "range of the legacy 'Consistency (%)' metric over 9 images x 8 methods")
    audit.append(dict(issue="Image order from os.listdir() and no random seeds", location="setup / bootstrap",
                      consequence="results not reproducible bit-for-bit", impact="order and resamples change between runs",
                      fix="sorted sample table, master seed and per-purpose random streams"))
    audit.append(dict(issue="500 × 500 px central crop as the only region of interest", location="process_image",
                      consequence="window smaller than the representative area; excludes most of the sample",
                      impact="see the ROI-design and REV analyses (centre-vs-disc differences, L_REV)",
                      fix="largest usable disc as primary ROI; 500 × 500 kept as secondary design"))
    ad = pd.DataFrame(audit)
    ctx.save_table(ad.rename(columns={"issue": "Issue", "location": "Location", "consequence": "Consequence",
                                      "impact": "Quantified impact (real data)", "fix": "Fix in this work"}),
                   "legacy_audit", "Defects of the original analysis code, their consequence, their quantified impact on "
                   "the nine images and the correction adopted.", kind="supp")
    t = lv[["label", "method", "legacy_stored", "legacy_reproduced", "corrected_center500", "corrected_disk",
            "legacy_skeleton_porosity", "skeleton_density"]].copy()
    for c in t.columns[2:]:
        t[c] = 100 * t[c]
    ctx.save_table(t.rename(columns={"label": "Image", "method": "Method", "legacy_stored": "Legacy, stored (%)",
                                     "legacy_reproduced": "Legacy, reproduced (%)",
                                     "corrected_center500": "Corrected, 500 px (%)", "corrected_disk": "Corrected, disc (%)",
                                     "legacy_skeleton_porosity": "Legacy 'skeleton porosity' (%)",
                                     "skeleton_density": "Skeleton density (%)"}),
                   "legacy_vs_corrected", "Porosity from the original code (stored results and faithful re-implementation) "
                   "against the corrected pipeline on the same central 500 × 500 px window and on the disc ROI (no "
                   "post-processing), and the legacy 'skeletonised porosity' against the true skeleton density.",
                   kind="supp", float_format="{:.2f}")
    _figure(ctx, lv)
    _report_md(ctx, ad, lv, max_diff)


def _figure(ctx, lv) -> None:
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", 70))
    for m in ctx.methods:
        d = lv[lv["method"] == m]
        axes[0].scatter(100 * d["legacy_reproduced"], 100 * d["corrected_center500"], s=14, marker=style.METHOD_MARKERS[m],
                        color=style.METHOD_COLORS[m], lw=0)
        axes[1].scatter(100 * d["corrected_center500"], 100 * d["corrected_disk"], s=14, marker=style.METHOD_MARKERS[m],
                        color=style.METHOD_COLORS[m], lw=0)
    for ax, (xl, yl) in zip(axes, [("Legacy code, central 500 px (%)", "Corrected, central 500 px (%)"),
                                   ("Corrected, central 500 px (%)", "Corrected, disc ROI (%)")]):
        ax.plot([0, 100], [0, 100], color=style.INK, lw=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        style.hgrid(ax)
    for k, ax in enumerate(axes):
        style.panel_label(ax, "ab"[k])
    fig.legend(handles=style.method_handles(), loc="lower center", ncol=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    ctx.save_fig(fig, "legacy_vs_corrected",
                 "(a) Porosity from the original code against the corrected pipeline on the same central 500 × 500 px "
                 "window (no post-processing); differences come from thresholding the raw 16-bit crop including "
                 "out-of-sample pixels, Sauvola's dynamic range and normalisation. (b) Effect of the ROI design in the "
                 "corrected pipeline (central window vs largest disc). Identity lines in black; 9 images × 8 methods.",
                 kind="supp")
    plt.close(fig)


def _report_md(ctx, ad, lv, max_diff) -> None:
    lines = ["# Audit of the original analysis code (\"codigo final interpore\")", "",
             "Written automatically by the `legacy` stage of the pipeline. Each item gives the defect, where it is in "
             "the original notebook, its consequence, the number that demonstrates it on the 9 slices and the fix "
             "adopted.", "",
             f"**Reproduction:** the faithful re-implementation of the original code reproduces the porosities stored "
             f"in `results/analysis_20250723_145220` with a maximum difference of {100 * max_diff:.4f} pp "
             f"({int((lv['repro_diff'] < 1e-6).sum())} of {len(lv)} values identical). Residual differences, if any, "
             "come from the scikit-image version.", ""]
    for i, r in enumerate(ad.itertuples(), 1):
        lines += [f"## {i}. {r.issue}", "", f"- **Where:** `{r.location}`", f"- **Consequence:** {r.consequence}",
                  f"- **Measured impact:** {r.impact}", f"- **Fix:** {r.fix}", ""]
    sk = lv["legacy_skeleton_porosity"]
    lines += ["## On the claims of the 2025 conference abstracts", "",
              f"- \"Skeletonization increased porosity by up to +94.7 %\": the original code measures the skeleton as "
              f"1 − skeleton density, which is {100 * sk.min():.1f}–{100 * sk.max():.1f} % on these slices. Any "
              "'porosity increase by skeletonisation' is an artefact of this inversion.",
              "- \"Yen CI = 98.7 %, Sauvola CI = −7527 %\": the 'Consistency Index' is 1 − SD/mean computed over the "
              "four refinements (one of them the inverted skeleton); with a binarised porosity close to zero "
              "(Sauvola with the default r) the ratio explodes and the index becomes strongly negative. It does not "
              "measure stability.",
              "- The values of Tables 1 and 2 of manuscript draft v9 (for example Otsu 17.8 ± 0.9 %, reference "
              "21.4 %) are not produced by any version of the code and should not be used.", ""]
    (ctx.out / "LEGACY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
