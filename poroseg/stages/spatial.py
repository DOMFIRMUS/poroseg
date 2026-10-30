"""Stage 'spatial': spatial uncertainty of image porosity, representativeness
(REV) and the pseudo-replication of the legacy pixel-iid bootstrap.

Scientific question
-------------------
A porosity measured on one 2-D slice is an *areal sample* of a spatially
correlated medium. Its sampling uncertainty depends on the size of the window
relative to the correlation range of the pore phase, not on the number of
pixels. This stage quantifies, for every segmentation of the factorial design,

1. the two-point statistics of the pore phase inside the ROI (mask-normalised
   FFT estimator of ``S2(h)`` and of the normalised covariance ``rho(h)``),
   the correlation length ``l_c`` (``rho = 1/e``), and the 2-D integral range
   ``A = int rho(h) dh`` (Matheron 1971; Lantuejoul 1991, Math. Geol.
   23:149-160; Kanit et al. 2003, Int. J. Solids Struct. 40:3647-3679), with
   the truncation rule documented in :mod:`poroseg.spatial_tools`;
2. the variability of the porosity of square tiles of side ``L`` (REV
   analysis; Bear 1972; Kanit et al. 2003): ``Var(phi_L) ~ phi(1-phi)A/L^2``
   for ``L >> l_c``, and the side ``L_REV = z sqrt(phi(1-phi)A)/e`` of a single
   window whose 95 % half-width is ``e`` (= ``spatial.target_abs_error_pp``);
3. the uncertainty of the ROI porosity by (a) a block bootstrap of tiles
   (Kunsch 1989; Lahiri 2003), (b) the integral-range standard error
   ``SE_IR = sqrt(phi(1-phi)A/S)`` and (c) the naive pixel-iid bootstrap of the
   legacy code, which treats ~10^6 correlated pixels as independent replicates
   (pseudo-replication, Hurlbert 1984) - its interval is too narrow by about
   ``sqrt(A)``;
4. radial porosity profiles (distance from the holder wall for granular packs,
   from the fitted core surface for free-standing cores), which expose the
   wall effect of packed beds (Benenati & Brosilow 1962, AIChE J. 8:359-361;
   Mueller 1992, Powder Technol. 72:269-275; de Klerk 2003, AIChE J.
   49:2022-2029) and edge (cupping / beam-hardening) artefacts of cores, and
   the difference between the central 500x500 px design and the whole disc,
   decomposed exactly into a *location* part (different pixels, same
   segmentation) and a *re-segmentation* part (threshold re-estimated on the
   500x500 histogram).

What the uncertainties mean
---------------------------
The spatial SEs/CIs describe how much the porosity of *this slice and ROI*
would vary if an equally large window had been taken elsewhere in a
statistically homogeneous medium of the same structure. They do **not**
include the segmentation (analyst-choice) uncertainty, which is analysed in the
other stages, nor the slice-to-slice variability along the core axis (2-D data
only). With large-scale trends (cupping, wall effect) the medium is not
stationary; the block bootstrap then treats the trend as random variation
(conservative), and the covariance-based ``A`` does not converge (flagged).

Block-size rule
---------------
For every segmentation, ``b`` = the smallest configured tile side
``>= max(spatial.block_tile_px, 3 l_c)`` that still gives ``>= 20`` complete
tiles inside the evaluation mask (``choose_block_size``). Blocks of ``>= 3 l_c``
are nearly uncorrelated (``rho(3 l_c) ~ e^-3`` for an exponential covariance);
Moran's I of the block values is reported as a diagnostic. The percentile CI of
the mean tile porosity is translated to the whole-ROI porosity
(``ci = phi_ROI + (q_alpha - mean_tiles)``); because complete tiles cover only
part of the ROI (column ``tile_coverage``) this interval is slightly
conservative for the whole ROI.

Outputs
-------
data/spatial_ci.csv          contract file: one row per (sample, roi, method, post)
data/spatial_summary.csv     same rows + covariance/REV/diagnostic columns
data/rev_curves.csv          tile statistics per tile side
data/two_point.csv           radially averaged S2(r), rho(r), running A(R) (post 'none')
data/radial_profiles.csv     porosity in rings vs distance from wall / edge (disc ROI)
data/center_vs_disk.csv      500x500 vs whole-disc difference, location / re-segmentation parts
data/iid_bootstrap_check.csv explicit pixel bootstrap vs binomial shortcut
figures: rev_tile_variability, uncertainty_block_vs_iid, radial_porosity_profiles,
         two_point_correlation
tables:  rev_integral_range, spatial_uncertainty, center_vs_disk
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from .. import spatial_tools as st
from .. import style
from ..postprocess import AREA_PRESERVING

MAX_LAG_PX = 256          # largest lag of the two-point statistics (px) ...
MAX_LAG_FRAC = 1 / 3      # ... and at most 1/3 of the smaller ROI side (edge-pair support)
MIN_BLOCKS = 20           # minimum number of blocks for the block bootstrap
MIN_TILES_REV = 10        # tile sizes with fewer tiles are not used for SD(L) / L_REV
WALL_ZONE_MM = 1.5        # outer annulus used to quantify wall / edge effects (half the smallest grain, 3 mm)


# ============================================================================
# helpers
# ============================================================================
def _pct(x):
    return 100.0 * np.asarray(x, float)


def _max_lag(E: np.ndarray, ctx) -> int:
    ys, xs = np.nonzero(np.any(E, axis=1))[0], np.nonzero(np.any(E, axis=0))[0]
    side = min(ys[-1] - ys[0] + 1, xs[-1] - xs[0] + 1)
    cap = int(ctx.cfg["spatial"].get("max_lag_px", MAX_LAG_PX))
    return int(min(cap, np.floor(side * MAX_LAG_FRAC)))


def _window_mask_in(ctx, sid: str, roi_from: str, roi_to: str) -> np.ndarray:
    """Evaluation mask of ``roi_from`` expressed in the bbox coordinates of ``roi_to``."""
    shape = ctx.raw(sid).shape
    full = np.zeros(shape, bool)
    Rf = ctx.roi(sid, roi_from)
    full[Rf.slices] = Rf.eval_mask
    Rt = ctx.roi(sid, roi_to)
    return full[Rt.slices]


# ============================================================================
# computation
# ============================================================================
def _compute(ctx):
    sp = ctx.cfg["spatial"]
    sizes = sorted(int(s) for s in sp["tile_sizes_px"])
    b_min = int(sp["block_tile_px"])
    e = float(sp["target_abs_error_pp"]) / 100.0
    ci_level = float(ctx.cfg["statistics"]["ci_level"])
    z = st.z_value(ci_level)
    n_boot = ctx.n_boot
    n_boot_px = int(sp["n_boot_pixels"])
    rois = [r for r in ctx.cfg["roi"]["designs"]]
    rois = [ctx.primary_roi] + [r for r in rois if r != ctx.primary_roi]
    posts = list(AREA_PRESERVING)
    bin_px = float(sp["radial_bin_px"])

    rows, rev_rows, tp_rows, rad_rows, cvd_rows, iid_rows = [], [], [], [], [], []
    for sid in ctx.sample_ids:
        t0 = time.time()
        px_mm = ctx.pixel_size_um(sid) / 1000.0
        lab, grp = ctx.label(sid), ctx.group(sid)
        g = ctx.geometry(sid)
        for roi in rois:
            R = ctx.roi(sid, roi)
            E = R.eval_mask
            N = int(E.sum())
            mlag = _max_lag(E, ctx)
            mc = st.MaskedCorrelator(E, mlag, workers=2)
            satE = st.summed_area_table(E)
            tilings = {L: st.tile_grid(E, L, satE) for L in sizes}
            n_tiles = {L: tilings[L].n for L in sizes}
            if roi == ctx.primary_roi:
                dist_edge = g.r_boundary - st_radial_distance(R, g)
                win = _window_mask_in(ctx, sid, "center500", roi) if "center500" in ctx.cfg["roi"]["designs"] else None
            for m in ctx.methods:
                for p in posts:
                    pore = ctx.pore(sid, m, p, roi) & E
                    k = int(pore.sum())
                    phi = k / N
                    tp = mc.two_point(pore, phi)
                    cs = st.summarize_covariance(tp)
                    lc = cs.lc_px
                    satP = st.summed_area_table(pore)
                    # ---------------------------------------------------------------- tiles / REV
                    tile_vals = {}
                    sdL, nL = [], []
                    for L in sizes:
                        v = st.tile_fractions(pore, tilings[L], satP)
                        tile_vals[L] = v
                        sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
                        mu = float(v.mean()) if len(v) else np.nan
                        sdL.append(sd)
                        nL.append(len(v))
                        rev_rows.append(dict(sample_id=sid, label=lab, group=grp, roi=roi, method=m, post=p,
                                             L_px=L, L_mm=L * px_mm, n_tiles=len(v), mean=mu, sd=sd,
                                             cv=sd / mu if mu and np.isfinite(sd) else np.nan,
                                             sd_theory_ir=float(st.sd_window_theory(phi, cs.A_px2, L))
                                             if np.isfinite(cs.A_px2) else np.nan,
                                             trusted=len(v) >= MIN_TILES_REV))
                    sdL, nL = np.array(sdL), np.array(nL)
                    Lmin_fit = max(3 * lc, sizes[0]) if np.isfinite(lc) else sizes[0]
                    fit = st.fit_variance_scaling(np.array(sizes), sdL, nL, phi, L_min=Lmin_fit,
                                                  min_tiles=MIN_TILES_REV)
                    L_rev_ir = st.l_rev_theory(phi, cs.A_px2, e, z) if np.isfinite(cs.A_px2) else np.nan
                    L_rev_emp, rev_status = st.l_rev_empirical(np.array(sizes), sdL, nL, e, z, MIN_TILES_REV)
                    # ------------------------------------------------------------ block bootstrap
                    b, b_rule = st.choose_block_size(sizes, n_tiles, lc, b_min, MIN_BLOCKS)
                    vb = tile_vals[b]
                    bb = st.block_bootstrap_mean(vb, n_boot, ctx.rng("spatial", "block", sid, roi, m, p), ci_level)
                    shift = phi - bb["mean"]
                    ci_lo = float(np.clip(bb["lo"] + shift, 0, 1)) if np.isfinite(bb["lo"]) else np.nan
                    ci_hi = float(np.clip(bb["hi"] + shift, 0, 1)) if np.isfinite(bb["hi"]) else np.nan
                    mi = st.morans_i(vb, tilings[b].gi, tilings[b].gj)
                    # -------------------------------------------------------------- IR and iid SE
                    se_ir = st.se_integral_range(phi, cs.A_px2, N) if np.isfinite(cs.A_px2) else np.nan
                    iid = st.iid_bootstrap_binomial(phi, N, n_boot_px, ctx.rng("spatial", "iid", sid, roi, m, p),
                                                    ci_level)
                    w_iid = iid["hi"] - iid["lo"]
                    rows.append(dict(
                        sample_id=sid, label=lab, group=grp, roi=roi, method=m, post=p, porosity=phi,
                        n_tiles=bb["n"], tile_px=b, cv_tiles=float(vb.std(ddof=1) / vb.mean())
                        if len(vb) > 1 and vb.mean() > 0 else np.nan,
                        se_block=bb["se"], ci_lo=ci_lo, ci_hi=ci_hi, se_ir=se_ir,
                        se_iid=iid["se_analytic"], ci_iid_lo=iid["lo"], ci_iid_hi=iid["hi"],
                        ci_width_ratio=(ci_hi - ci_lo) / w_iid if w_iid > 0 else np.nan,
                        # ---- extra (non-contract) columns
                        block_rule=b_rule, tile_coverage=bb["n"] * b * b / N, mean_tiles=bb["mean"],
                        morans_i_blocks=mi, n_eval_px=N, area_mm2=N * px_mm ** 2, px_mm=px_mm,
                        max_lag_px=mlag, lc_px=lc, lc_um=lc * px_mm * 1000, r0_px=cs.r0_px,
                        ir_converged=cs.converged, scale_px=cs.scale_px, A_px2=cs.A_px2,
                        A_mm2=cs.A_px2 * px_mm ** 2, A_tile_px2=fit["A_tile_px2"],
                        sd_slope=fit["slope"], n_fit_sizes=fit["n_sizes"],
                        n_eff=N / cs.A_px2 if np.isfinite(cs.A_px2) and cs.A_px2 > 0 else np.nan,
                        se_iid_boot=iid["se_boot"],
                        L_rev_ir_px=L_rev_ir, L_rev_ir_mm=L_rev_ir * px_mm,
                        L_rev_tiles_px=L_rev_emp, L_rev_tiles_mm=L_rev_emp * px_mm, L_rev_tiles_status=rev_status,
                        hw_block_pp=100 * (ci_hi - ci_lo) / 2, hw_ir_pp=100 * z * se_ir,
                        hw_iid_pp=100 * w_iid / 2))
                    # ------------------------------------------------------- two-point curves
                    if p == "none":
                        Rr, Acum = st.running_integral_range(tp, mlag)
                        for i in range(len(cs.r)):
                            tp_rows.append(dict(sample_id=sid, label=lab, group=grp, roi=roi, method=m, post=p,
                                                r_px=int(cs.r[i]), r_mm=cs.r[i] * px_mm, s2=cs.s2_r[i],
                                                rho=cs.rho_r[i], npairs=cs.npairs_r[i], A_cum_px2=Acum[i],
                                                A_cum_mm2=Acum[i] * px_mm ** 2))
                    # ---------------------------------------------- radial profile, 500 vs disc
                    if roi == ctx.primary_roi:
                        prof = st.radial_profile(pore, E, dist_edge, bin_px)
                        for i in range(len(prof["k"])):
                            rad_rows.append(dict(sample_id=sid, label=lab, group=grp, roi=roi, method=m, post=p,
                                                 d_lo_px=prof["lo"][i], d_hi_px=prof["hi"][i],
                                                 d_mean_px=prof["d_mean"][i], d_mean_mm=prof["d_mean"][i] * px_mm,
                                                 r_center_mm=(g.r_boundary - prof["d_mean"][i]) * px_mm,
                                                 porosity=prof["porosity"][i], n_px=int(prof["n"][i])))
                        zone = dist_edge < WALL_ZONE_MM / px_mm
                        phi_wall = float(pore[E & zone].mean())
                        phi_core = float(pore[E & ~zone].mean())
                        rows[-1].update(phi_wall_zone=phi_wall, phi_interior=phi_core)
                        if win is not None:
                            W = win & E
                            phi_win = float(pore[W].mean())
                            cvd_rows.append(dict(sample_id=sid, label=lab, group=grp, method=m, post=p,
                                                 phi_disk=phi, phi_disk_seg_in_window=phi_win,
                                                 window_px_in_disk=int(W.sum()),
                                                 window_px_total=int(win.sum()),
                                                 r_window_mean_mm=float((g.r_boundary - dist_edge[W]).mean()) * px_mm,
                                                 r_disk_mean_mm=float((g.r_boundary - dist_edge[E]).mean()) * px_mm))
                    # explicit pixel-bootstrap check (literal resampling of the pixels)
                    if roi == ctx.primary_roi and m == "Otsu" and p == "none":
                        nb_exp = 50 if ctx.quick else 200
                        ex = st.iid_bootstrap_explicit(pore[E], nb_exp, ctx.rng("spatial", "iid_explicit", sid), ci_level)
                        iid_rows.append(dict(sample_id=sid, label=lab, porosity=phi, n_px=N, n_boot_explicit=nb_exp,
                                             se_analytic=iid["se_analytic"], se_binomial_boot=iid["se_boot"],
                                             se_explicit_boot=ex["se_boot"],
                                             rel_diff_explicit=ex["se_boot"] / iid["se_analytic"] - 1))
            del mc
        ctx.log("%s done (%.1fs)", sid, time.time() - t0)

    out = dict(ci=pd.DataFrame(rows), rev=pd.DataFrame(rev_rows), tp=pd.DataFrame(tp_rows),
               rad=pd.DataFrame(rad_rows), cvd=pd.DataFrame(cvd_rows), iid=pd.DataFrame(iid_rows))
    # ------------------------------------------------ 500x500 vs disc decomposition
    if len(out["cvd"]):
        c5 = out["ci"][out["ci"]["roi"] == "center500"][["sample_id", "method", "post", "porosity"]]
        c5 = c5.rename(columns={"porosity": "phi_center500"})
        cvd = out["cvd"].merge(c5, on=["sample_id", "method", "post"], how="left")
        cvd["d_total"] = cvd["phi_center500"] - cvd["phi_disk"]
        cvd["d_location"] = cvd["phi_disk_seg_in_window"] - cvd["phi_disk"]
        cvd["d_resegmentation"] = cvd["phi_center500"] - cvd["phi_disk_seg_in_window"]
        dk = out["ci"][out["ci"]["roi"] == ctx.primary_roi][["sample_id", "method", "post", "hw_block_pp"]]
        cvd = cvd.merge(dk, on=["sample_id", "method", "post"], how="left")
        cvd["exceeds_block_hw"] = np.abs(100 * cvd["d_total"]) > cvd["hw_block_pp"]
        out["cvd"] = cvd
    return out


def st_radial_distance(R, g) -> np.ndarray:
    """Distance (px) of every bbox pixel of ROI ``R`` from the fitted sample centre."""
    yy, xx = np.mgrid[R.y0:R.y1, R.x0:R.x1]
    return np.hypot(yy - g.cy, xx - g.cx).astype(np.float32)


CONTRACT_COLS = ["sample_id", "label", "group", "roi", "method", "post", "porosity", "n_tiles", "tile_px",
                 "cv_tiles", "se_block", "ci_lo", "ci_hi", "se_ir", "se_iid", "ci_iid_lo", "ci_iid_hi",
                 "ci_width_ratio"]


def run(ctx) -> None:
    t0 = time.time()
    res = _compute(ctx)
    ci = res["ci"]
    extra = ["block_rule", "tile_coverage", "lc_px", "A_px2", "ir_converged", "morans_i_blocks"]
    ctx.save_data(ci[CONTRACT_COLS + extra], "spatial_ci")
    ctx.save_data(ci, "spatial_summary")
    ctx.save_data(res["rev"], "rev_curves")
    ctx.save_data(res["tp"], "two_point")
    ctx.save_data(res["rad"], "radial_profiles")
    ctx.save_data(res["cvd"], "center_vs_disk")
    ctx.save_data(res["iid"], "iid_bootstrap_check")
    ctx.log("computation done in %.1fs", time.time() - t0)
    _outputs(ctx, res)


def _outputs(ctx, res) -> None:
    """Figures, tables and key numbers of the spatial stage."""
    ci, rev, tp, rad, cvd, iid = res["ci"], res["rev"], res["tp"], res["rad"], res["cvd"], res["iid"]
    P = ctx.primary_roi
    order = [ctx.label(s) for s in ctx.sample_ids]
    e_pp = float(ctx.cfg["spatial"]["target_abs_error_pp"])
    z = st.z_value(float(ctx.cfg["statistics"]["ci_level"]))
    sd_crit = e_pp / z                                   # SD of a single window whose 95 % half-width is e
    base = ci[(ci["roi"] == P) & (ci["post"] == "none")].copy()
    otsu = base[base["method"] == "Otsu"].set_index("label").reindex(order)

    # ------------------------------------------------------------------ figure: REV tile variability
    rv = rev[(rev["roi"] == P) & (rev["post"] == "none") & rev["trusted"]]
    fig, axes = plt.subplots(3, 3, figsize=style.figsize("double", 150), sharex=False, sharey=False)
    for ax, sid in zip(axes.ravel(), ctx.sample_ids):
        d = rv[rv["sample_id"] == sid]
        for m in ctx.methods:
            dm = d[d["method"] == m].sort_values("L_mm")
            if dm.empty:
                continue
            hl = m == "Otsu"
            ax.plot(dm["L_mm"], 100 * dm["sd"], marker=style.METHOD_MARKERS[m], ms=2.6 if hl else 1.8,
                    lw=1.2 if hl else 0.6, color=style.METHOD_COLORS[m], alpha=1.0 if hl else 0.55,
                    zorder=3 if hl else 2)
        do = d[d["method"] == "Otsu"].sort_values("L_mm")
        if len(do):
            L0, s0 = do["L_mm"].iloc[0], 100 * do["sd"].iloc[0]
            Lx = np.array([do["L_mm"].min(), do["L_mm"].max() * 1.5])
            ax.plot(Lx, s0 * L0 / Lx, ls=":", color=style.MUTED, lw=0.8)
        ax.axhline(sd_crit, color=style.INK, lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.3, 20)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20, 50]))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        g = ctx.group(sid)
        ax.set_title(f"{ctx.label(sid)} ({style.GROUP_LABELS[g].lower()})", fontsize=7.5)
        style.hgrid(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Tile side L (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SD of tile porosity (pp)")
    for k, ax in enumerate(axes.ravel()):
        style.panel_label(ax, "abcdefghi"[k])
    handles = style.method_handles() + [
        Line2D([], [], ls=":", color=style.MUTED, lw=0.8, label="Slope −1 (uncorrelated tiles)"),
        Line2D([], [], ls="--", color=style.INK, lw=0.6, label=f"SD for ±{e_pp:g} pp (95 %)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ctx.save_fig(fig, "rev_tile_variability",
                 "Representative elementary area analysis. Standard deviation (percentage points, pp) of the "
                 "porosity of non-overlapping square tiles fully inside the disc ROI as a function of the tile side "
                 f"L, for every thresholding algorithm (post-processing none; Otsu highlighted; tile sides with "
                 f"fewer than {MIN_TILES_REV} tiles omitted). The dotted line has slope −1, the decay expected for "
                 "L much larger than the correlation length (Var ∝ A/L²); the dashed line is the SD of a single "
                 f"window whose 95 % half-width is ±{e_pp:g} pp. Panels a-d: granular packs; e-i: consolidated cores.",
                 kind="main")
    plt.close(fig)

    # ------------------------------------------------------- figure: block vs IR vs iid half-widths
    fig, ax = plt.subplots(figsize=style.figsize("single", 82))
    y = np.arange(len(order))
    for m in ctx.methods:
        dm = base[base["method"] == m].set_index("label").reindex(order)
        ax.scatter(dm["hw_block_pp"], y, s=6, color=style.METHOD_COLORS[m], marker=style.METHOD_MARKERS[m],
                   alpha=0.35, lw=0, zorder=2)
    ax.scatter(otsu["hw_iid_pp"], y, s=16, marker="x", color=style.INK, lw=0.8, zorder=4, label="Pixel-iid bootstrap")
    ax.scatter(otsu["hw_ir_pp"], y, s=16, marker="^", facecolor="none", edgecolor=style.INK, lw=0.8, zorder=4,
               label="Integral range")
    ax.scatter(otsu["hw_block_pp"], y, s=16, marker="o", color=style.METHOD_COLORS["Otsu"], zorder=5,
               label="Block bootstrap (Otsu)")
    ax.scatter([], [], s=6, color=style.MUTED, alpha=0.5, label="Block bootstrap (other methods)")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("95 % CI half-width of ROI porosity (pp)")
    style.hgrid(ax, "x")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    ctx.save_fig(fig, "uncertainty_block_vs_iid",
                 "Spatial sampling uncertainty of the porosity of the disc ROI (post-processing none): half-width of "
                 "the 95 % interval from the naive pixel-iid bootstrap of the original code (crosses), from the "
                 "integral range of the pore-phase covariance (triangles) and from the block bootstrap of tiles "
                 "(filled circles: Otsu; faint symbols: the other seven algorithms). The pixel bootstrap treats "
                 "spatially correlated pixels as independent replicates and understates the uncertainty by one to "
                 "two orders of magnitude. Log scale; n = 9 images.", kind="main")
    plt.close(fig)

    # ---------------------------------------------------------------- figure: radial profiles
    if len(rad):
        rr = rad[(rad["method"] == "Otsu") & (rad["post"] == "none")]
        fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", 60), sharey=True)
        grp_order = ["granular_pack", "consolidated"]
        pal = style.PALETTE
        for ax, grp in zip(axes, grp_order):
            sids = [s for s in ctx.sample_ids if ctx.group(s) == grp]
            for k, sid in enumerate(sids):
                d = rr[rr["sample_id"] == sid].sort_values("d_mean_mm")
                ax.plot(d["d_mean_mm"], 100 * d["porosity"], lw=0.9, color=pal[k % len(pal)], label=ctx.label(sid))
            ax.axvline(WALL_ZONE_MM, color=style.MUTED, lw=0.6, ls="--")
            ax.set_xlabel("Distance from holder wall (mm)" if grp == "granular_pack" else "Distance from core surface (mm)")
            ax.set_title(style.GROUP_LABELS[grp], fontsize=7.5)
            style.hgrid(ax)
            ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="upper right" if grp == "granular_pack" else "upper left")
        axes[0].set_ylabel("Ring porosity, Otsu (%)")
        for k, ax in enumerate(axes):
            style.panel_label(ax, "ab"[k])
        fig.tight_layout()
        ctx.save_fig(fig, "radial_porosity_profiles",
                     f"Porosity in {ctx.cfg['spatial']['radial_bin_px']}-px rings as a function of the distance from "
                     "the holder wall (granular packs, a) or from the fitted core surface (consolidated cores, b); Otsu, "
                     f"no post-processing, disc ROI. The dashed line marks the {WALL_ZONE_MM:g} mm outer zone used to "
                     "quantify wall and edge effects. Packed beds show the classical oscillating near-wall porosity.",
                     kind="supp")
        plt.close(fig)

    # ---------------------------------------------------------------- figure: two-point correlation
    if len(tp):
        t1 = tp[(tp["roi"] == P) & (tp["method"] == "Otsu") & (tp["post"] == "none")]
        fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", 60), sharey=True)
        for ax, grp in zip(axes, ["granular_pack", "consolidated"]):
            sids = [s for s in ctx.sample_ids if ctx.group(s) == grp]
            for k, sid in enumerate(sids):
                d = t1[t1["sample_id"] == sid].sort_values("r_mm")
                ax.plot(d["r_mm"], d["rho"], lw=0.9, color=style.PALETTE[k % len(style.PALETTE)], label=ctx.label(sid))
            ax.axhline(0, color=style.INK, lw=0.5)
            ax.axhline(np.exp(-1), color=style.MUTED, lw=0.6, ls="--")
            ax.set_xlabel("Lag r (mm)")
            ax.set_title(style.GROUP_LABELS[grp], fontsize=7.5)
            style.hgrid(ax)
            ax.legend(frameon=False, fontsize=6.5)
        axes[0].set_ylabel("Normalised covariance ρ(r)")
        for k, ax in enumerate(axes):
            style.panel_label(ax, "ab"[k])
        fig.tight_layout()
        ctx.save_fig(fig, "two_point_correlation",
                     "Radially averaged normalised covariance ρ(r) = (S2(r) − φ²)/(φ − φ²) of the pore phase (Otsu, no "
                     "post-processing, disc ROI; mask-corrected FFT estimator). The dashed line (1/e) defines the "
                     "correlation length l_c; the integral of ρ over the plane is the integral range A.", kind="supp")
        plt.close(fig)

    # ---------------------------------------------------------------- tables
    side_500 = {sid: 500 * ctx.pixel_size_um(sid) / 1000 for sid in ctx.sample_ids}
    trows = []
    for sid in ctx.sample_ids:
        r = otsu.loc[ctx.label(sid)]
        area = r["area_mm2"]
        L = r["L_rev_tiles_mm"] if np.isfinite(r["L_rev_tiles_mm"]) else r["L_rev_ir_mm"]
        n_win = area / L ** 2 if np.isfinite(L) and L > 0 else np.nan
        verdict = ("ROI smaller than one REV window" if np.isfinite(n_win) and n_win < 1 else
                   "ROI holds {:.1f} REV windows".format(n_win) if np.isfinite(n_win) else "undetermined")
        trows.append({"Sample": ctx.label(sid), "Group": style.GROUP_LABELS[ctx.group(sid)],
                      "φ (%)": 100 * r["porosity"], "l_c (µm)": r["lc_um"], "A (mm²)": r["A_mm2"],
                      "A converged": "yes" if bool(r["ir_converged"]) else "no",
                      "L_REV, covariance (mm)": r["L_rev_ir_mm"], "L_REV, tiles (mm)": r["L_rev_tiles_mm"],
                      "ROI area (mm²)": area, "√ROI area (mm)": float(np.sqrt(area)),
                      "500 px side (mm)": side_500[sid], "Verdict": verdict})
    tdf = pd.DataFrame(trows)
    ctx.save_table(tdf, "rev_integral_range",
                   f"Correlation length l_c, integral range A and representative window side L_REV for a 95 % "
                   f"half-width of ±{e_pp:g} pp on a single square window (Otsu, no post-processing, disc ROI). "
                   "L_REV from the covariance uses Var(φ_L) = φ(1−φ)A/L²; L_REV from tiles extrapolates the observed "
                   "SD(L). 'A converged' = no means the running integral of the covariance had not stabilised within "
                   "the largest lag (trend or large-scale heterogeneity), so the covariance-based value is a lower bound.",
                   kind="main", float_format="{:.3g}")
    su = base.copy()
    su["φ (%)"] = 100 * su["porosity"]
    su["Block CI (%)"] = su.apply(lambda r: f"{100 * r.ci_lo:.1f}–{100 * r.ci_hi:.1f}", axis=1)
    sut = su[["label", "method", "φ (%)", "Block CI (%)", "hw_block_pp", "hw_ir_pp", "hw_iid_pp", "ci_width_ratio",
              "tile_px"]].rename(columns={"label": "Sample", "method": "Method", "hw_block_pp": "Half-width block (pp)",
                                          "hw_ir_pp": "Half-width IR (pp)", "hw_iid_pp": "Half-width iid (pp)",
                                          "ci_width_ratio": "Block/iid width ratio", "tile_px": "Block side (px)"})
    ctx.save_table(sut, "spatial_uncertainty",
                   "Spatial uncertainty of the disc-ROI porosity for every image and algorithm (post-processing none): "
                   "95 % block-bootstrap interval, half-widths from the block bootstrap, the integral range (IR) and the "
                   f"naive pixel-iid bootstrap, their ratio and the block side used (n_boot = {ctx.n_boot}).",
                   kind="supp", float_format="{:.3g}")
    if len(cvd):
        cv = cvd[cvd["post"] == "none"].copy()
        cvt = pd.DataFrame({"Sample": cv["label"], "Method": cv["method"], "Disc (%)": 100 * cv["phi_disk"],
                            "Central 500 px (%)": 100 * cv["phi_center500"], "Difference (pp)": 100 * cv["d_total"],
                            "Location part (pp)": 100 * cv["d_location"],
                            "Re-segmentation part (pp)": 100 * cv["d_resegmentation"],
                            "Exceeds block half-width": np.where(cv["exceeds_block_hw"], "yes", "no")})
        ctx.save_table(cvt, "center_vs_disk",
                       "Porosity of the central 500×500 px window (the original design) minus the porosity of the whole "
                       "disc ROI, split exactly into a location part (same disc segmentation, evaluated in the window) "
                       "and a re-segmentation part (threshold re-estimated on the window histogram). Post-processing none.",
                       kind="supp", float_format="{:.2f}")

    # ---------------------------------------------------------------- key numbers
    r_otsu = otsu["ci_width_ratio"].to_numpy(float)
    r_all = base["ci_width_ratio"].to_numpy(float)
    ctx.add_key("spatial_ratio_block_iid_otsu_median", float(np.nanmedian(r_otsu)),
                "median over the 9 images of the ratio of 95 % CI widths, block bootstrap / pixel-iid bootstrap "
                "(Otsu, post none, disc ROI)")
    ctx.add_key("spatial_ratio_block_iid_otsu_range", [float(np.nanmin(r_otsu)), float(np.nanmax(r_otsu))],
                "min and max over the 9 images of the block/iid CI-width ratio (Otsu, post none, disc ROI)")
    ctx.add_key("spatial_ratio_block_iid_all_range", [float(np.nanmin(r_all)), float(np.nanmax(r_all))],
                "min and max of the block/iid CI-width ratio over 9 images x 8 methods (post none, disc ROI)")
    ctx.add_key("spatial_hw_iid_otsu_range_pp", [float(otsu["hw_iid_pp"].min()), float(otsu["hw_iid_pp"].max())],
                "range of the 95 % half-width of the naive pixel-iid bootstrap (Otsu, post none, disc ROI)", "pp")
    for grp in ("granular_pack", "consolidated"):
        sids = [ctx.label(s) for s in ctx.sample_ids if ctx.group(s) == grp]
        o = otsu.loc[sids]
        ctx.add_key(f"spatial_hw_block_otsu_{grp}_range_pp", [float(o["hw_block_pp"].min()), float(o["hw_block_pp"].max())],
                    f"range of the 95 % block-bootstrap half-width of ROI porosity, {grp} (Otsu, post none, disc ROI)", "pp")
        Lt = o["L_rev_tiles_mm"].to_numpy(float)
        ctx.add_key(f"spatial_L_rev_tiles_{grp}_range_mm", [float(np.nanmin(Lt)), float(np.nanmax(Lt))],
                    f"range of the window side for a ±{e_pp:g} pp (95 %) single-window porosity, tile-based, {grp} "
                    "(Otsu, post none, disc ROI)", "mm")
        ctx.add_key(f"spatial_lc_{grp}_range_um", [float(o["lc_um"].min()), float(o["lc_um"].max())],
                    f"range of the correlation length of the pore phase, {grp} (Otsu, post none)", "um")
    reach = []
    for sid in ctx.sample_ids:
        L = otsu.loc[ctx.label(sid), "L_rev_tiles_mm"]
        reach.append(bool(np.isfinite(L) and side_500[sid] >= L))
    ctx.add_key("spatial_n_center500_reaches_rev", int(sum(reach)),
                f"number of the 9 images whose 500x500 px window side is at least the tile-based L_REV for ±{e_pp:g} pp "
                "(Otsu, post none)")
    packs = [ctx.label(s) for s in ctx.sample_ids if ctx.group(s) == "granular_pack"]
    if "phi_wall_zone" in otsu.columns:
        dw = 100 * (otsu.loc[packs, "phi_wall_zone"] - otsu.loc[packs, "phi_interior"])
        ctx.add_key("spatial_wall_effect_packs_pp", [float(dw.min()), float(dw.median()), float(dw.max())],
                    f"granular packs: porosity in the outer {WALL_ZONE_MM:g} mm wall zone minus interior porosity "
                    "(min, median, max; Otsu, post none)", "pp")
    if len(cvd):
        cv = cvd[cvd["post"] == "none"]
        ctx.add_key("spatial_center_vs_disk_abs_median_pp", float(np.median(np.abs(100 * cv["d_total"]))),
                    "median over 9 images x 8 methods of |porosity(central 500x500) - porosity(disc)| (post none)", "pp")
        ctx.add_key("spatial_center_vs_disk_frac_exceeding_block_hw", float(np.mean(cv["exceeds_block_hw"])),
                    "fraction of image x method cases in which the 500x500-vs-disc difference exceeds the block-bootstrap "
                    "95 % half-width of the disc porosity (post none)")
    if len(iid):
        ctx.add_key("spatial_iid_explicit_vs_analytic_max_reldiff", float(np.max(np.abs(iid["rel_diff_explicit"]))),
                    "largest relative difference between the SE of an explicit pixel bootstrap and the binomial formula "
                    "sqrt(phi(1-phi)/N) (Otsu, post none, disc ROI)")
    ctx.add_key("spatial_n_boot", int(ctx.n_boot), "block-bootstrap replicates")
