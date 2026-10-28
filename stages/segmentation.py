"""Stage 'segmentation': the full factorial design.

For every sample x ROI design x thresholding method x post-processing level:
threshold, binary pore image and porosity (or skeleton density for the
'skeleton' level). Also writes sample geometry, QA overlays of the detected
ROI, the dataset table and the methods/parameters table.

Outputs
-------
data/segmentation_factorial.csv   one row per (sample, roi, method, post)
data/thresholds.csv               one row per (sample, roi, method)
data/geometry.csv                 one row per sample
cache/pores_<sid>_<roi>.npz       packed binary images (read by later stages)
figures/supp_roi_detection        detected usable ROI for every sample
tables/dataset                    samples, pixel size, ROI size, lab reference
tables/methods_parameters         every parameter of the workflow
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import postprocess, style
from ..metrics import porosity
from ..thresholds import METHOD_INFO, to_raw


def _roi_overlay(ax, ctx, sid, show_center=True):
    raw = ctx.raw(sid).data
    g = ctx.geometry(sid)
    lo, hi = np.percentile(raw, [0.5, 99.5])
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(g.cx + g.r_boundary * np.cos(th), g.cy + g.r_boundary * np.sin(th), color=style.PALETTE[7], lw=0.6)
    ax.plot(g.cx + g.r_usable * np.cos(th), g.cy + g.r_usable * np.sin(th), color=style.PALETTE[0], lw=0.8)
    if show_center and "center500" in ctx.cfg["roi"]["designs"]:
        R = ctx.roi(sid, "center500")
        ax.add_patch(plt.Rectangle((R.x0, R.y0), R.x1 - R.x0, R.y1 - R.y0, fill=False,
                                   ec=style.PALETTE[3], lw=0.8, ls=(0, (3, 2))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, raw.shape[1])
    ax.set_ylim(raw.shape[0], 0)
    style.scalebar(ax, ctx.pixel_size_um(sid), color="white")


def run(ctx) -> None:
    cfg = ctx.cfg
    rows, trows, grows = [], [], []
    for sid in ctx.sample_ids:
        raw = ctx.raw(sid)
        g = ctx.geometry(sid)
        px = ctx.pixel_size_um(sid)
        if raw.pixel_size_um_tag and abs(raw.pixel_size_um_tag - px) > 0.05 * px:
            ctx.log("WARNING %s: pixel size in samples.csv (%.2f um) differs from TIFF tag (%.2f um)",
                    sid, px, raw.pixel_size_um_tag)
        grows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), holder=g.holder,
                          height_px=raw.shape[0], width_px=raw.shape[1], dtype=raw.dtype,
                          pixel_size_um=px, pixel_size_um_tag=raw.pixel_size_um_tag,
                          cx=g.cx, cy=g.cy, r_boundary_px=g.r_boundary, r_usable_px=g.r_usable,
                          fit_rms_px=g.fit_rms_px, inlier_frac=g.inlier_frac, geometry_source=g.source))
        for roi_name in cfg["roi"]["designs"]:
            R = ctx.roi(sid, roi_name)
            N = ctx.norm(sid, roi_name)
            area_mm2 = R.n_eval * (px / 1000.0) ** 2
            grows[-1][f"{roi_name}_n_eval_px"] = R.n_eval
            grows[-1][f"{roi_name}_area_mm2"] = area_mm2
            grows[-1][f"{roi_name}_norm_lo"] = N.lo
            grows[-1][f"{roi_name}_norm_hi"] = N.hi
            for m in ctx.methods:
                T = ctx.threshold(sid, m, roi_name)
                trows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), roi=roi_name,
                                  method=m, family=T.family, t_norm=T.summary,
                                  t_raw=to_raw(T.summary, N.lo, N.hi),
                                  t_map_sd=float(np.std(T.value[R.mask])) if T.is_map else 0.0))
                for p in ctx.posts:
                    pore = ctx.pore(sid, m, p, roi_name)
                    val = porosity(pore, R.eval_mask)
                    area_ok = p in postprocess.AREA_PRESERVING
                    rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), roi=roi_name,
                                     method=m, family=T.family, post=p,
                                     metric="porosity" if area_ok else "skeleton_density",
                                     value=val, porosity=val if area_ok else np.nan,
                                     n_eval_px=R.n_eval, area_mm2=area_mm2))
            ctx.write_pore_cache(sid, roi_name)
        ctx.log("%s done (r_usable=%.0f px, %s)", sid, g.r_usable, g.source)

    seg = pd.DataFrame(rows)
    ctx.save_data(seg, "segmentation_factorial")
    ctx.save_data(pd.DataFrame(trows), "thresholds")
    geo = pd.DataFrame(grows)
    ctx.save_data(geo, "geometry")
    ctx._seg = seg

    # ------------------------------------------------ QA / supplementary figure
    n = len(ctx.sample_ids)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=style.figsize("double", 40 * nrow + 6))
    axs = np.atleast_1d(axs).ravel()
    for ax, sid in zip(axs, ctx.sample_ids):
        _roi_overlay(ax, ctx, sid)
        ax.set_title(ctx.label(sid), pad=2)
    for ax in axs[n:]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=style.PALETTE[7], lw=0.8, label="Detected sample / holder inner wall"),
                        Line2D([], [], color=style.PALETTE[0], lw=0.8, label="Usable disc (primary ROI)"),
                        Line2D([], [], color=style.PALETTE[3], lw=0.8, ls=(0, (3, 2)), label="Central 500×500 px (original design)")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ctx.save_fig(fig, "supp_roi_detection",
                 "Automatically detected regions of interest. Red: fitted outer boundary of free-standing cores or "
                 "inner wall of the grain holder (RANSAC circle fit); blue: usable disc used as primary ROI "
                 f"(free cores: radius reduced by {100 * cfg['roi']['free_margin_frac']:.0f} %; holders: "
                 f"{cfg['roi']['container_margin_px']} px from the wall); yellow dashed: central 500×500 px "
                 "window of the original workflow. Scale bars in physical units.", kind="supp")
    for sid in ctx.sample_ids:   # one QA image per sample, full resolution overlay
        f, ax = plt.subplots(figsize=(6, 6))
        _roi_overlay(ax, ctx, sid)
        g = ctx.geometry(sid)
        if "edge_points" in g.info:
            ep = g.info["edge_points"]
            ax.plot(ep[:, 1], ep[:, 0], ".", ms=1, color=style.PALETTE[4])
        ax.set_title(f"{sid}: r_boundary={g.r_boundary:.1f}px, r_usable={g.r_usable:.1f}px, "
                     f"rms={g.fit_rms_px:.2f}px", fontsize=7)
        ctx.save_qa(f, f"roi_{sid}")

    # ------------------------------------------------------------ dataset table
    ref = ctx.lab_ref.set_index("sample_id")
    trows = []
    for _, r in geo.iterrows():
        sid = r["sample_id"]
        d = dict(Sample=r["label"], Group=style.GROUP_LABELS.get(r["group"], r["group"]),
                 **{"Image (px)": f"{r['width_px']}×{r['height_px']}",
                    "Pixel (µm)": r["pixel_size_um"],
                    "ROI diameter (mm)": 2 * r["r_usable_px"] * r["pixel_size_um"] / 1000,
                    "ROI area (mm²)": r[f"{ctx.primary_roi}_area_mm2"],
                    "ROI pixels (10⁶)": r[f"{ctx.primary_roi}_n_eval_px"] / 1e6})
        if sid in ref.index:
            rr = ref.loc[sid]
            d["Lab φ (%)"] = 100 * rr["ref"]
            d["Lab SD (%)"] = 100 * rr["ref_sd"] if np.isfinite(rr["ref_sd"]) else np.nan
            d["Lab n"] = f"{int(rr['ref_n'])}" if rr["ref_kind"] == "multi-method" else "1*"
            d["Lab µCT φ (%)"] = 100 * rr["uct"]
        else:
            d.update({"Lab φ (%)": np.nan, "Lab SD (%)": np.nan, "Lab n": "–", "Lab µCT φ (%)": np.nan})
        trows.append(d)
    ctx.save_table(pd.DataFrame(trows), "dataset",
                   "Samples, image geometry and laboratory reference porosity. Lab φ is the mean of the "
                   "independent physical methods (direct porosimetry, gravimetry, indirect porosimetry, gas "
                   "porosimetry); Lab µCT is the laboratory's own tomographic estimate, analysed separately "
                   "and never used as reference.",
                   notes="* single value, laboratory method not stated in the source spreadsheet. "
                         "Pixel sizes inferred from TIFF resolution tags / file names - confirm with acquisition logs.")

    # ---------------------------------------------------- methods & parameters
    t = cfg["thresholds"]
    pp = cfg["postprocessing"]
    prm = [("Input", "Image", "single 2-D slice per sample, 16-bit TIFF, original grey values"),
           ("ROI", "Primary ROI", f"usable disc; free cores: {100 * (1 - cfg['roi']['free_margin_frac']):.0f} % of "
                                  f"fitted radius; holders: inner wall − {cfg['roi']['container_margin_px']} px"),
           ("ROI", "Secondary ROI", f"{cfg['roi']['center_size_px']}×{cfg['roi']['center_size_px']} px centred on image"),
           ("ROI", "Evaluation rim", f"{cfg['roi']['eval_border_px']} px excluded from all metrics"),
           ("Normalisation", "Scaling", f"clip to ROI percentiles [{cfg['normalization']['p_low']}, "
                                        f"{cfg['normalization']['p_high']}], rescale to [0, 1]"),
           ("Thresholding", "Histogram bins", f"{t['nbins']} (global methods)"),
           ("Thresholding", "Pore phase", "I ≤ T (dark, low attenuation)")]
    for m in ctx.methods:
        info = METHOD_INFO[m]
        extra = ""
        if m == "Sauvola":
            extra = f"; w = {t['sauvola']['window']} px, k = {t['sauvola']['k']}, R = {t['sauvola']['r']}"
        elif m == "Niblack":
            extra = f"; w = {t['niblack']['window']} px, k = {t['niblack']['k']}"
        prm.append(("Thresholding", m, f"{info['category']}{extra}; {info['reference']}"))
    prm += [("Post-processing", "Median", f"{pp['median_size']}×{pp['median_size']} majority filter on pore phase"),
            ("Post-processing", "Open–close", f"opening then closing of pore phase, {pp['morph_se']} "
                                              f"{pp['morph_size']}×{pp['morph_size']}"),
            ("Post-processing", "Skeleton", "medial axis of pore phase (Zhang–Suen, scikit-image); reported as "
                                            "skeleton density, not porosity")]
    s = cfg["statistics"]
    prm += [("Statistics", "Bootstrap", f"{s['n_boot']} replicates, percentile / BCa intervals, "
                                         f"{int(100 * s['ci_level'])} % level"),
            ("Statistics", "Multiplicity", f"{s['p_adjust']} adjustment"),
            ("Statistics", "Seed", str(cfg["project"]["seed"]))]
    ctx.save_table(pd.DataFrame(prm, columns=["Step", "Setting", "Value"]), "methods_parameters",
                   "Complete parameterisation of the segmentation workflow (generated from config.yaml).")

    # ------------------------------------------------------------- key numbers
    prim = seg[(seg["roi"] == ctx.primary_roi) & (seg["post"] == "none")]
    ctx.add_key("n_samples", len(ctx.sample_ids), "number of images analysed")
    ctx.add_key("n_segmentations", int(len(seg)), "segmentations computed (samples × ROIs × methods × post-processing)")
    ctx.add_key("porosity_range_primary_raw_pct",
                [float(100 * prim["porosity"].min()), float(100 * prim["porosity"].max())],
                "min and max porosity over all samples and methods (primary ROI, no post-processing)", "%")
