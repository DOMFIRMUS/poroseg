"""Stage 'morphology': what post-processing does to pore-space morphology and
topology, grey-scale (partial-volume) porosity and unresolved porosity.

A. Morphology / topology (disc ROI, every sample x method x post level; all
   estimators restricted to the evaluation mask and edge-corrected, see
   :mod:`poroseg.morpho_tools`):

   * porosity A_A; boundary length density L_A (Crofton intercepts, mm^-1;
     S_V = 4 L_A / pi for isotropic 3-D structures);
   * Euler number density chi_A (bit quads, pore 8-connected / solid
     4-connected, mm^-2) - the 2-D topological descriptor;
   * number density of pore objects N_A (Miles-Lantuejoul corrected, mm^-2)
     and area-weighted equivalent-diameter quantiles of interior objects;
   * granulometric (opening) pore-size distribution and its d10/d50/d90 (um),
     meaningful also for connected pore networks;
   * fraction of the pore area in objects that a 3x3 opening deletes;
   * for 'skeleton': skeleton length density (mm/mm^2) and branch / end
     points per mm^2 (the skeleton is a topology descriptor, not a porosity).

   Post-processing effects are paired differences against 'none' for every
   sample x method, summarised per method across the n = 9 images (median,
   bootstrap CI, exact Wilcoxon signed-rank p with Holm adjustment over the
   methods of each descriptor x level family).

B. Grey-scale porosity phi_PV = (mu_s - mean I)/(mu_s - mu_p) on RAW grey
   levels (two-phase linear attenuation model; unbiased under zero-mean noise
   because it is linear - no per-pixel clipping). Endmembers: mu_s = centre of
   the dominant (brightest major) solid peak of a binned Gaussian KDE of the
   ROI grey levels; mu_p = air: the dark peak inside the ROI for the granular
   packs (large air-filled pores), the median of the air annulus around the
   core in the same slice for free-standing cores. The interval spans the
   endmember variants (KDE bandwidth x0.5/x1/x2, peak-centre definitions,
   air quartiles). Dense inclusions (> mu_s) bias phi_PV low; cupping violates
   the constant-attenuation assumption (flagged).

C. Unresolved porosity for the 7 images with a laboratory value:
   phi_unres = phi_lab - phi_seg for every method, with u = sqrt(lab_sd^2 +
   se_block^2) (spatial SE from the 'spatial' stage when available), and the
   partial-volume part phi_PV - phi_Otsu.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import morpho_tools as mt
from .. import style
from ..thresholds import LOCAL_METHODS

AIR_MARGIN_PX = 15        # air annulus starts this far outside the fitted core boundary
AIR_WIDTH_PX = 60         # ... and is this wide (clipped to the reconstructed field of view)
DESCRIPTORS = [("porosity", "Δφ (pp)", 100.0, "diff"),
               ("N_A_mm2", "Δ log N_A", 1.0, "logratio"),
               ("chi_A_mm2", "Δχ_A (mm$^{-2}$)", 1.0, "diff"),
               ("L_A_mm", "ΔL_A (%)", 100.0, "reldiff"),
               ("psd_d50_um", "Δ log d50", 1.0, "logratio")]


# ============================================================================ A. descriptors
def _descriptors(pore, E, se3, px_mm, post, with_psd=True) -> dict:
    n = int(E.sum())
    out = dict(porosity=float((pore & E).sum() / n))
    if post == "skeleton":
        sk = mt.skeleton_metrics(pore, E)
        out.update(skel_length_density_mm=sk["length"] / (n * px_mm), skel_branch_mm2=sk["n_branch"] / (n * px_mm ** 2),
                   skel_end_mm2=sk["n_end"] / (n * px_mm ** 2), skel_isolated_mm2=sk["n_isolated"] / (n * px_mm ** 2))
        return out
    cr = mt.crofton_boundary_density(pore, E)
    chi, nq = mt.euler_quads(pore, E, 8)
    ot = mt.object_table(pore, E)
    os_ = mt.object_size_stats(ot)
    sub = mt.sub_se_fraction(pore, E, se3)
    out.update(L_A_mm=cr["L_A"] / px_mm, chi_A_mm2=(chi / nq) / px_mm ** 2 if nq else np.nan,
               N_A_mm2=os_["N_A"] / px_mm ** 2, n_objects=os_["n_objects"], n_border=os_["n_border"],
               deq_mean_um=os_["deq_mean"] * px_mm * 1000, deq_d50_um=os_["deq_d50"] * px_mm * 1000,
               frac_area_interior=os_["frac_area_interior"], frac_area_sub_se=sub["frac_area_sub_se"],
               frac_area_removed_by_opening=sub.get("frac_area_removed_by_opening", np.nan))
    if with_psd:
        G = mt.granulometry(pore, E)
        q = mt.psd_quantiles(mt.DEFAULT_RADII, G)
        out.update(psd_d10_um=q["d10"] * px_mm * 1000, psd_d50_um=q["d50"] * px_mm * 1000,
                   psd_d90_um=q["d90"] * px_mm * 1000, _G=G)
    return out


def _morphology(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    se3 = np.ones((3, 3), bool)
    rows, psd_rows = [], []
    for sid in ctx.sample_ids:
        t0 = time.time()
        R = ctx.roi(sid)
        E = R.eval_mask
        px_mm = ctx.pixel_size_um(sid) / 1000.0
        for m in ctx.methods:
            for p in ctx.posts:
                d = _descriptors(ctx.pore(sid, m, p), E, se3, px_mm, p, with_psd=(p != "skeleton"))
                G = d.pop("_G", None)
                rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), method=m,
                                 family="local" if m in LOCAL_METHODS else "global", post=p, **d))
                if G is not None and m == "Otsu":
                    for r, g in zip(mt.DEFAULT_RADII, G):
                        D_um = mt.disc_equivalent_diameter(r) * px_mm * 1000
                        psd_rows.append(dict(sample_id=sid, label=ctx.label(sid), method=m, post=p, radius_px=r,
                                             D_um=D_um, G=g, F=1 - g))
        ctx.log("%s morphology done (%.1fs)", sid, time.time() - t0)
    return pd.DataFrame(rows), pd.DataFrame(psd_rows)


def _effects(ctx, mo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired differences vs 'none' and per-method summaries (sample = unit)."""
    base = mo[mo["post"] == "none"].set_index(["sample_id", "method"])
    eff = []
    for p in ("median", "open_close"):
        cur = mo[mo["post"] == p].set_index(["sample_id", "method"])
        for (sid, m), r in cur.iterrows():
            b = base.loc[(sid, m)]
            row = dict(sample_id=sid, label=r["label"], group=r["group"], method=m, post=p)
            for col, _, scale, kind in DESCRIPTORS:
                x0, x1 = float(b[col]), float(r[col])
                if kind == "diff":
                    v = scale * (x1 - x0)
                elif kind == "reldiff":
                    v = scale * (x1 - x0) / x0 if x0 else np.nan
                else:
                    v = np.log(x1 / x0) if x0 > 0 and x1 > 0 and np.isfinite(x0) and np.isfinite(x1) else np.nan
                row[f"d_{col}"] = v
            eff.append(row)
    eff = pd.DataFrame(eff)
    summ = []
    for p in ("median", "open_close"):
        for col, lab, _, _ in DESCRIPTORS:
            ps, rows = [], []
            for m in ctx.methods:
                x = eff[(eff["post"] == p) & (eff["method"] == m)][f"d_{col}"].to_numpy(float)
                x = x[np.isfinite(x)]
                med, lo, hi = mt.boot_median_ci(x, ctx.rng("morphology", "boot", p, col, m), ctx.n_boot)
                w = mt.signed_rank_exact(x)
                rows.append(dict(post=p, descriptor=col, label=lab, method=m, n=len(x), median=med, lo=lo, hi=hi,
                                 p=w["p"], n_pos=w["n_pos"], n_neg=w["n_neg"]))
                ps.append(w["p"])
            padj = mt.holm(np.array(ps, float))
            for r, pa in zip(rows, padj):
                r["p_holm"] = float(pa)
            summ.extend(rows)
            # pooled over methods: per-sample mean of the differences (n = 9 independent units)
            pooled = eff[eff["post"] == p].groupby("sample_id")[f"d_{col}"].mean().to_numpy(float)
            pooled = pooled[np.isfinite(pooled)]
            med, lo, hi = mt.boot_median_ci(pooled, ctx.rng("morphology", "boot", p, col, "pooled"), ctx.n_boot)
            w = mt.signed_rank_exact(pooled)
            summ.append(dict(post=p, descriptor=col, label=lab, method="all (sample means)", n=len(pooled),
                             median=med, lo=lo, hi=hi, p=w["p"], n_pos=w["n_pos"], n_neg=w["n_neg"], p_holm=np.nan))
    return eff, pd.DataFrame(summ)


# ============================================================================ B. grey-scale porosity
def _kde(v, grid, bw_factor=1.0):
    counts = np.bincount(grid.index(v), minlength=grid.nbins)
    return mt.smooth_hist(counts, grid, bw_factor * mt.silverman_bandwidth(v[:: max(1, v.size // 200000)]))


def _greyscale(ctx) -> pd.DataFrame:
    rows = []
    for sid in ctx.sample_ids:
        raw = ctx.raw(sid).data
        R = ctx.roi(sid)
        g = ctx.geometry(sid)
        v = raw[R.slices][R.eval_mask].astype(np.float64)
        grid = mt.make_grid(v, integer=True)
        mean_I = float(v.mean())
        thr = ctx.threshold(sid, "Otsu")
        nm = ctx.norm(sid)
        t_raw = nm.lo + float(thr.summary) * (nm.hi - nm.lo)
        mus, mups = [], []
        for bwf in (0.5, 1.0, 2.0):
            dens = _kde(v, grid, bwf)
            m_s = mt.mode_in_range(dens, grid, lo=t_raw)
            mus += [m_s, mt.peak_top_centre(dens, grid, m_s, 0.5), mt.peak_top_centre(dens, grid, m_s, 0.3)]
            if ctx.group(sid) == "granular_pack":
                m_p = mt.mode_in_range(dens, grid, hi=t_raw)
                mups += [m_p, mt.peak_top_centre(dens, grid, m_p, 0.5)]
        if ctx.group(sid) == "granular_pack":
            src = "dark (air) peak inside the ROI"
            mu_p = float(np.median(mups))
        else:
            H, Wd = raw.shape
            yy, xx = np.mgrid[0:H, 0:Wd]
            r = np.hypot(yy - g.cy, xx - g.cx)
            field = min(g.cx, g.cy, Wd - 1 - g.cx, H - 1 - g.cy)
            r0 = g.r_boundary + AIR_MARGIN_PX
            r1 = min(r0 + AIR_WIDTH_PX, field - 2)
            air = raw[(r > r0) & (r < r1)].astype(np.float64) if r1 > r0 + 5 else np.array([])
            if air.size > 100:
                q25, q50, q75 = np.percentile(air, [25, 50, 75])
                mu_p, mups = float(q50), [q25, q50, q75]
                src = f"median of the air annulus {r0:.0f}-{r1:.0f} px from the core centre ({air.size} px)"
            else:
                mu_p, mups, src = np.nan, [np.nan], "no air annulus inside the field of view"
        mu_s = float(np.median(mus))
        phis = [mt.phi_pv(mean_I, s, p) for s in mus for p in mups if np.isfinite(s) and np.isfinite(p) and s != p]
        phi = float(mt.phi_pv(mean_I, mu_s, mu_p)) if np.isfinite(mu_p) else np.nan
        frac_bright = float(np.mean(v > mu_s + 0.5 * (mu_s - mu_p))) if np.isfinite(mu_p) else np.nan
        rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), mean_I=mean_I, mu_s=mu_s,
                         mu_p=mu_p, source_mu_p=src, phi_pv=phi,
                         phi_pv_lo=float(np.nanmin(phis)) if phis else np.nan,
                         phi_pv_hi=float(np.nanmax(phis)) if phis else np.nan,
                         frac_bright_inclusions=frac_bright, otsu_raw_threshold=t_raw))
    return pd.DataFrame(rows)


# ============================================================================ C. unresolved porosity
def _unresolved(ctx, gs: pd.DataFrame) -> pd.DataFrame:
    ref = ctx.lab_ref.set_index("sample_id")
    seg = ctx.seg_primary()
    seg = seg[seg["post"] == "none"]
    try:
        sp = ctx.read_data("spatial_ci")
        sp = sp[(sp["roi"] == ctx.primary_roi) & (sp["post"] == "none")].set_index(["sample_id", "method"])["se_block"]
    except FileNotFoundError:
        sp = None
    rows = []
    for sid in ctx.sample_ids:
        if sid not in ref.index or not np.isfinite(ref.at[sid, "ref"]):
            continue
        lab, lab_sd = float(ref.at[sid, "ref"]), float(ref.at[sid, "ref_sd"]) if np.isfinite(ref.at[sid, "ref_sd"]) else np.nan
        g = gs.set_index("sample_id").loc[sid]
        for m in ctx.methods:
            phi = float(seg[(seg["sample_id"] == sid) & (seg["method"] == m)]["porosity"].iloc[0])
            se_b = float(sp.loc[(sid, m)]) if sp is not None else np.nan
            u = np.sqrt(np.nansum([lab_sd ** 2 if np.isfinite(lab_sd) else 0.0, se_b ** 2 if np.isfinite(se_b) else 0.0]))
            rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), method=m, phi_lab=lab,
                             phi_lab_sd=lab_sd, ref_kind=ref.at[sid, "ref_kind"], phi_uct_lab=ref.at[sid, "uct"],
                             phi_seg=phi, se_block=se_b, phi_unres=lab - phi, u_unres=u,
                             phi_pv=g["phi_pv"], phi_pv_minus_seg=g["phi_pv"] - phi))
    return pd.DataFrame(rows)


# ============================================================================ run
def run(ctx) -> None:
    mo, psd = _morphology(ctx)
    ctx.save_data(mo, "morphology")
    ctx.save_data(psd, "psd_granulometry")
    eff, summ = _effects(ctx, mo)
    ctx.save_data(eff, "postprocessing_effects")
    ctx.save_data(summ, "postprocessing_effects_summary")
    gs = _greyscale(ctx)
    ctx.save_data(gs, "greyscale_porosity")
    un = _unresolved(ctx, gs)
    ctx.save_data(un, "unresolved_porosity")
    _fig_effects(ctx, eff)
    _fig_psd(ctx, psd)
    _fig_unresolved(ctx, gs)
    _tables(ctx, summ, gs, un, mo)
    _keys(ctx, summ, gs, un, mo, eff)


def _fig_effects(ctx, eff) -> None:
    fig, axes = plt.subplots(2, 2, figsize=style.figsize("double", 115), sharex=True)
    panels = [("d_porosity", "Δ porosity (pp)"), ("d_N_A_mm2", "Δ log pore number density"),
              ("d_chi_A_mm2", r"Δ Euler density $\chi_A$ (mm$^{-2}$)"), ("d_L_A_mm", "Δ boundary length density (%)")]
    x = np.arange(len(ctx.methods))
    for k, (ax, (col, lab)) in enumerate(zip(axes.ravel(), panels)):
        for j, p in enumerate(("median", "open_close")):
            off = -0.17 if j == 0 else 0.17
            for i, m in enumerate(ctx.methods):
                v = eff[(eff["post"] == p) & (eff["method"] == m)][col].to_numpy(float)
                v = v[np.isfinite(v)]
                jit = np.linspace(-0.07, 0.07, len(v)) if len(v) > 1 else np.zeros(len(v))
                ax.scatter(i + off + jit, v, s=7, marker=style.POST_MARKERS[p], color=style.POST_COLORS[p],
                           alpha=0.75, lw=0, zorder=3)
                if len(v):
                    ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=style.INK, lw=1.0, zorder=4)
        ax.axhline(0, color=style.INK, lw=0.5)
        if col == "d_chi_A_mm2":
            ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel(lab)
        style.hgrid(ax)
        style.panel_label(ax, "abcd"[k])
    for ax in axes[-1]:
        ax.set_xticks(x, ctx.methods, rotation=45, ha="right")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker=style.POST_MARKERS[p], color=style.POST_COLORS[p], label=style.POST_LABELS[p])
          for p in ("median", "open_close")] + [Line2D([], [], color=style.INK, lw=1.0, label="Median over images")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ctx.save_fig(fig, "postprocessing_effects",
                 "Effect of post-processing on porosity and pore-space morphology, measured as paired differences "
                 "against the unprocessed segmentation of the same image and algorithm (disc ROI): (a) porosity, (b) "
                 "log-ratio of the number density of pore objects (Miles-Lantuéjoul corrected), (c) Euler number "
                 "density (pores 8-connected; symmetric-log axis), (d) relative change of the boundary length density "
                 "(Crofton estimator). Points: the 9 images; bars: medians. Median 3×3 and opening-closing 3×3 act on "
                 "the pore phase.", kind="main")
    plt.close(fig)


def _fig_psd(ctx, psd) -> None:
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", 62), sharey=True)
    for ax, grp in zip(axes, ("granular_pack", "consolidated")):
        sids = [s for s in ctx.sample_ids if ctx.group(s) == grp]
        for k, sid in enumerate(sids):
            col = style.PALETTE[k % len(style.PALETTE)]
            for p, ls in (("none", "-"), ("open_close", "--")):
                d = psd[(psd["sample_id"] == sid) & (psd["post"] == p)].sort_values("D_um")
                ax.plot(d["D_um"], d["F"], ls=ls, color=col, lw=0.9, label=ctx.label(sid) if p == "none" else None)
        ax.set_xscale("log")
        ax.set_xlabel("Opening diameter D (µm)")
        ax.set_title(style.GROUP_LABELS[grp], fontsize=7.5)
        style.hgrid(ax)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("Cumulative pore-area fraction F(D)")
    for k, ax in enumerate(axes):
        style.panel_label(ax, "ab"[k])
    fig.tight_layout()
    ctx.save_fig(fig, "pore_size_distributions",
                 "Granulometric (opening-based) pore-size distributions of the Otsu segmentations: fraction of the "
                 "pore area that cannot hold a disc of diameter D, before (solid) and after (dashed) the 3×3 "
                 "opening-closing; disc ROI. The opening removes the finest pores first.", kind="supp")
    plt.close(fig)


def _fig_unresolved(ctx, gs) -> None:
    ref = ctx.lab_ref.set_index("sample_id")
    seg = ctx.seg_primary()
    seg = seg[seg["post"] == "none"]
    sids = ctx.sample_ids
    fig, ax = plt.subplots(figsize=style.figsize("double", 72))
    x = np.arange(len(sids))
    for i, sid in enumerate(sids):
        for j, m in enumerate(ctx.methods):
            v = 100 * float(seg[(seg["sample_id"] == sid) & (seg["method"] == m)]["porosity"].iloc[0])
            ax.scatter(i - 0.28 + 0.08 * j, v, s=10, color=style.METHOD_COLORS[m], marker=style.METHOD_MARKERS[m],
                       lw=0, zorder=3)
        g = gs.set_index("sample_id").loc[sid]
        if np.isfinite(g["phi_pv"]):
            ax.add_patch(plt.Rectangle((i + 0.38 - 0.05, 100 * g["phi_pv_lo"]), 0.1,
                                       100 * (g["phi_pv_hi"] - g["phi_pv_lo"]), color=style.MUTED, alpha=0.5, lw=0))
            ax.plot([i + 0.33, i + 0.43], [100 * g["phi_pv"]] * 2, color=style.INK, lw=1.0)
        if sid in ref.index and np.isfinite(ref.at[sid, "ref"]):
            r, sd = 100 * ref.at[sid, "ref"], ref.at[sid, "ref_sd"]
            ax.errorbar(i + 0.52, r, yerr=100 * sd if np.isfinite(sd) else None, fmt="D", ms=4, color=style.INK,
                        mfc=style.INK, capsize=2, lw=0.8, zorder=4)
            if np.isfinite(ref.at[sid, "uct"]):
                ax.scatter(i + 0.52, 100 * ref.at[sid, "uct"], s=16, marker="s", facecolor="white", edgecolor=style.INK,
                           lw=0.8, zorder=4)
    ax.set_xticks(x, [ctx.label(s) for s in sids])
    ax.set_ylabel("Porosity (%)")
    ax.axvline(3.5, color=style.MUTED, lw=0.5, ls=":")
    style.hgrid(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = style.method_handles() + [
        Patch(color=style.MUTED, alpha=0.5, label=r"Grey-scale $\phi_\mathrm{PV}$ (range)"),
        Line2D([], [], ls="none", marker="D", color=style.INK, label="Laboratory reference (± SD)"),
        Line2D([], [], ls="none", marker="s", mfc="white", mec=style.INK, label="Laboratory µCT")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    ctx.save_fig(fig, "unresolved_porosity",
                 "Porosity of every image from the eight binary segmentations (coloured symbols; disc ROI, no "
                 "post-processing), the grey-scale partial-volume estimate φ_PV (grey bar: range over endmember "
                 "definitions; black line: central value) and, where available, the laboratory reference (black "
                 "diamond: mean ± SD of four physical methods for the granular packs CF4-CF10, single value of "
                 "unstated method for VN01-VN04) and the laboratory's own µCT porosity (open square). The vertical "
                 "spread of the coloured symbols is the analyst-choice uncertainty; the gap between them and the "
                 "laboratory value is porosity below the image resolution (plus partial-volume misclassification).",
                 kind="main")
    plt.close(fig)


def _tables(ctx, summ, gs, un, mo) -> None:
    s = summ.copy()
    s["CI"] = s.apply(lambda r: f"[{r['lo']:.2f}, {r['hi']:.2f}]" if np.isfinite(r["lo"]) else "", axis=1)
    piv = []
    for (p, m), g in s.groupby(["post", "method"], sort=False):
        row = {"Post-processing": style.POST_LABELS[p], "Method": m}
        for _, r in g.iterrows():
            star = "*" if (np.isfinite(r["p_holm"]) and r["p_holm"] < 0.05) else ""
            row[r["label"]] = f"{r['median']:.2f} {r['CI']}{star}"
        piv.append(row)
    ctx.save_table(pd.DataFrame(piv), "postprocessing_effects",
                   "Median paired change (95 % bootstrap CI over the 9 images) of porosity, pore number density "
                   "(log-ratio), Euler number density, boundary length density and granulometric median pore "
                   "diameter (log-ratio) produced by each post-processing level relative to the unprocessed "
                   "segmentation (disc ROI). * Holm-adjusted exact Wilcoxon signed-rank p < 0.05 across the eight "
                   "algorithms; with n = 9 the smallest attainable two-sided p is 0.0039. 'all (sample means)': the "
                   "per-image mean over algorithms.", kind="main")
    g = gs.copy()
    gt = pd.DataFrame({"Sample": g["label"], "Group": g["group"].map(style.GROUP_LABELS),
                       "μ_s (grey level)": g["mu_s"], "μ_p (grey level)": g["mu_p"], "Source of μ_p": g["source_mu_p"],
                       "φ_PV (%)": 100 * g["phi_pv"], "φ_PV range (%)": [f"{100 * a:.1f}–{100 * b:.1f}" for a, b in
                                                                       zip(g["phi_pv_lo"], g["phi_pv_hi"])],
                       "Bright-inclusion fraction (%)": 100 * g["frac_bright_inclusions"]})
    ctx.save_table(gt, "greyscale_porosity",
                   "Grey-scale (partial-volume) porosity φ_PV = (μ_s − Ī)/(μ_s − μ_p) on raw grey levels of the disc "
                   "ROI, with the solid and pore (air) endmembers and the range obtained over endmember definitions. "
                   "Bright inclusions (grey level above μ_s + (μ_s − μ_p)/2) bias φ_PV low; radial cupping violates "
                   "the constant-attenuation assumption.", kind="supp", float_format="{:.1f}")
    if len(un):
        u = un[un["method"] == "Otsu"]
        rng_ = un.groupby("sample_id")["phi_seg"].agg(["min", "max"])
        ut = pd.DataFrame({"Sample": u["label"], "Reference": u["ref_kind"],
                           "φ_lab (%)": 100 * u["phi_lab"], "SD (pp)": 100 * u["phi_lab_sd"],
                           "Lab µCT (%)": 100 * u["phi_uct_lab"], "φ_PV (%)": 100 * u["phi_pv"],
                           "φ Otsu (%)": 100 * u["phi_seg"],
                           "φ range, 8 methods (%)": [f"{100 * rng_.at[s, 'min']:.1f}–{100 * rng_.at[s, 'max']:.1f}"
                                                      for s in u["sample_id"]],
                           "φ_lab − φ_Otsu (pp)": 100 * u["phi_unres"], "u (pp)": 100 * u["u_unres"],
                           "φ_PV − φ_Otsu (pp)": 100 * u["phi_pv_minus_seg"]})
        ctx.save_table(ut, "unresolved_porosity",
                       "Laboratory porosity against image porosity for the seven images with a reference: laboratory "
                       "mean (SD over four physical methods for the packs; single value of unstated method for the "
                       "cores), laboratory µCT, grey-scale φ_PV, Otsu and the range over eight algorithms (no "
                       "post-processing, disc ROI), the apparent unresolved porosity φ_lab − φ_Otsu with its standard "
                       "uncertainty u = sqrt(SD_lab² + SE_block²) and the partial-volume part φ_PV − φ_Otsu.",
                       kind="main", float_format="{:.1f}")


def _keys(ctx, summ, gs, un, mo, eff) -> None:
    K = ctx.add_key
    for p in ("median", "open_close"):
        for col, lab, _, _ in DESCRIPTORS:
            r = summ[(summ["post"] == p) & (summ["descriptor"] == col) & (summ["method"] == "all (sample means)")]
            if len(r):
                r = r.iloc[0]
                K(f"morphology_{p}_{col}_pooled", [float(r["median"]), float(r["lo"]), float(r["hi"])],
                  f"{style.POST_LABELS[p]}: median over 9 images of the per-image mean (over 8 methods) paired change of "
                  f"{lab} vs no post-processing, with 95 % bootstrap CI", "")
                K(f"morphology_{p}_{col}_pooled_p", float(r["p"]), f"exact Wilcoxon p of the pooled change ({lab}), n = 9")
        d = eff[eff["post"] == p]["d_porosity"]
        K(f"morphology_{p}_dporosity_range_pp", [float(d.min()), float(d.max())],
          f"range over 9 images x 8 methods of the porosity change produced by {style.POST_LABELS[p]}", "pp")
    oc = mo[(mo["post"] == "none")].groupby("label")["frac_area_sub_se"].median()
    K("morphology_frac_area_sub_se_median_by_sample", {k: round(float(v), 4) for k, v in oc.items()},
      "median over methods of the fraction of pore area in objects smaller than the 3x3 structuring element "
      "(deleted by opening), per image (no post-processing)")
    for _, r in gs.iterrows():
        K(f"morphology_phi_pv_{r['label'].lower()}", [float(r["phi_pv"]), float(r["phi_pv_lo"]), float(r["phi_pv_hi"])],
          f"grey-scale porosity phi_PV of {r['label']} (central, min, max over endmember definitions)", "fraction")
    if len(un):
        u = un[un["method"] == "Otsu"]
        for grp in ("granular_pack", "consolidated"):
            ug = u[u["group"] == grp]
            if len(ug):
                K(f"morphology_unresolved_otsu_{grp}_pp", [float(100 * ug["phi_unres"].min()),
                                                           float(100 * ug["phi_unres"].median()),
                                                           float(100 * ug["phi_unres"].max())],
                  f"phi_lab - phi_Otsu (min, median, max) for {grp} images with a laboratory reference", "pp")
        pk = un[(un["group"] == "granular_pack")]
        K("morphology_unresolved_packs_all_methods_range_pp", [float(100 * pk["phi_unres"].min()), float(100 * pk["phi_unres"].max())],
          "range over 4 packs x 8 methods of phi_lab - phi_seg (no post-processing)", "pp")
        pv = u[u["group"] == "granular_pack"]
        K("morphology_pv_minus_otsu_packs_pp", [float(100 * pv["phi_pv_minus_seg"].min()), float(100 * pv["phi_pv_minus_seg"].max())],
          "granular packs: phi_PV - phi_Otsu (min, max): porosity hidden in partial-volume pixels", "pp")
