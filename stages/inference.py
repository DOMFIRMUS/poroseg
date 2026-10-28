"""Stage 'inference': the inferential statistics of the paper (subject = image).

1. Method effect (disc ROI, no post-processing, n = 9 images): Friedman test with
   Kendall's W (cluster-bootstrap CI), one-way repeated-measures ANOVA with the
   Greenhouse-Geisser correction on porosity (pp) and on logit porosity, and all
   28 pairwise comparisons (Hodges-Lehmann estimate with exact CI, exact
   Wilcoxon signed-rank p, rank-biserial r, Holm adjustment). With n = 9 the
   smallest attainable two-sided exact p is 2/2^9 = 0.0039, so after Holm over
   28 pairs no comparison can reach 0.05 (28 x 0.0039 = 0.11): the pairwise
   table is reported for effect sizes, not for significance.
2. Factorial design image x method x post-processing (3 area-preserving
   levels): two-way RM-ANOVA (GG-corrected, generalised eta-squared) and the
   exact decomposition of the total sum of squares into material (image) and
   analyst-choice terms, with 95 % cluster-bootstrap CIs over images, on the
   porosity scale (pp) and on the logit scale; a four-factor version adding the
   ROI design; leave-one-image-out influence; a sensitivity analysis without the
   two algorithms that fail on unimodal images (Yen, Triangle); and, per image,
   the inter-method dispersion compared with the spatial (block-bootstrap) SE.
3. Agreement with the laboratory (n = 7 images; packs n = 4, cores n = 3):
   bias (t and exact Hodges-Lehmann CIs), limits of agreement (exact Carkeet
   CIs), MAE/RMSE (bootstrap CIs over images), Lin's CCC, Deming and
   Passing-Bablok regressions (n = 7 is far too small for reliable regression;
   reported for completeness), exact Wilcoxon p of the bias, probability of
   being the best method, and per-image standardised differences.
4. Laboratory-internal comparison over the nine granular specimens: the
   laboratory's own µCT porosity against the mean of the four physical methods,
   the between-method spread of the physical methods, and the effect of the
   legacy reference definition (mean of all five values, including µCT).
5. The old 'Consistency Index' recomputed honestly as 1 - CV of tile porosity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .. import stats as S
from .. import style
from ..postprocess import AREA_PRESERVING

FAIL_LO, FAIL_HI = 0.01, 0.90


def _pp(df):
    d = df.copy()
    d["pp"] = 100 * d["porosity"]
    d["logit"] = S.logit(d["porosity"].to_numpy(), eps=1e-4)
    return d


# ============================================================================ 1. method effect
def _method_effect(ctx, base):
    K = ctx.add_key
    fr = S.friedman(base, "pp", "method", "sample_id", n_boot=ctx.n_boot, rng=ctx.rng("inference", "friedmanW"))
    a_pp = S.rm_anova_oneway(base, "pp", "method", "sample_id")
    a_lg = S.rm_anova_oneway(base, "logit", "method", "sample_id")
    pw = S.pairwise_paired(base, "pp", "method", "sample_id")
    K("inference_friedman_Q", fr["Q"], "Friedman chi-square across 8 methods (disc ROI, post none, n = 9 images)")
    K("inference_friedman_p", fr["p"], "Friedman p (asymptotic)")
    K("inference_friedman_p_iman_davenport", fr["p_iman_davenport"], "Iman-Davenport F-test p for the Friedman design")
    K("inference_kendall_W", [fr["W"], fr["W_lo"], fr["W_hi"]],
      "Kendall's W of the method ranks over images with 95 % cluster-bootstrap CI")
    for tag, a in (("pp", a_pp), ("logit", a_lg)):
        K(f"inference_rmanova_method_{tag}", dict(F=a["F"], df1=a["df1"], df2=a["df2"], p=a["p"], eps_gg=a["eps_gg"],
                                                  p_gg=a["p_gg"], eta2_g=a["eta2_generalized"]),
          f"one-way RM-ANOVA of porosity ({tag}) on method, GG-corrected, generalised eta-squared")
    K("inference_pairwise_min_p_exact", float(pw["p_min"].min()), "smallest attainable exact two-sided Wilcoxon p, n = 9")
    K("inference_pairwise_n_sig_holm", int((pw["p_adj"] < 0.05).sum()), "pairwise comparisons with Holm p < 0.05 (of 28)")
    K("inference_pairwise_n_sig_raw", int((pw["p_wilcoxon"] < 0.05).sum()), "pairwise comparisons with unadjusted exact p < 0.05")
    big = pw.reindex(pw["hl"].abs().sort_values(ascending=False).index).head(3)
    K("inference_pairwise_largest_hl", [f"{r.b}-{r.a}: {r.hl:.1f} [{r.hl_lo:.1f}, {r.hl_hi:.1f}] pp" for r in big.itertuples()],
      "three largest Hodges-Lehmann median differences between methods with exact CIs")
    t = pw.rename(columns={"a": "Method A", "b": "Method B", "n": "n", "mean_diff": "Mean B−A (pp)",
                           "hl": "HL B−A (pp)", "hl_lo": "HL CI low", "hl_hi": "HL CI high",
                           "hl_conf": "Achieved confidence", "p_wilcoxon": "Exact Wilcoxon p", "p_adj": "Holm p",
                           "rank_biserial": "Rank-biserial r"})
    ctx.save_table(t[["Method A", "Method B", "n", "Mean B−A (pp)", "HL B−A (pp)", "HL CI low", "HL CI high",
                      "Achieved confidence", "Exact Wilcoxon p", "Holm p", "Rank-biserial r"]], "pairwise_methods",
                   "All pairwise comparisons of the eight algorithms (porosity in pp, disc ROI, no post-processing; "
                   "subject = image, n = 9): mean and Hodges-Lehmann (HL) median difference with the exact "
                   "distribution-free CI (achieved confidence ≥ 95 %), exact Wilcoxon signed-rank p, Holm-adjusted p "
                   "over the 28 pairs and matched-pairs rank-biserial correlation. With n = 9 the smallest attainable "
                   "exact p is 0.0039, so no Holm-adjusted p can fall below 0.05 (28 × 0.0039 = 0.11).",
                   kind="supp", float_format="{:.3g}")
    # matrix figure of HL differences
    ms = ctx.methods
    M = np.full((len(ms), len(ms)), np.nan)
    star = np.zeros_like(M, dtype=object)
    for r in pw.itertuples():
        i, j = ms.index(r.a), ms.index(r.b)
        M[j, i], M[i, j] = r.hl, -r.hl
        mk = "**" if r.p_adj < 0.05 else ("*" if r.p_wilcoxon < 0.05 else "")
        star[j, i] = star[i, j] = mk
    fig, ax = plt.subplots(figsize=style.figsize("single", 80))
    lim = np.nanmax(np.abs(M))
    from matplotlib.colors import TwoSlopeNorm
    im = ax.imshow(M, cmap=style.CMAP_DIV, norm=TwoSlopeNorm(0, -lim, lim))
    for i in range(len(ms)):
        for j in range(len(ms)):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}{star[i, j]}", ha="center", va="center", fontsize=5.6,
                        color="white" if abs(M[i, j]) > 0.6 * lim else style.INK)
    ax.set_xticks(range(len(ms)), ms, rotation=45, ha="right")
    ax.set_yticks(range(len(ms)), ms)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("HL difference, row − column (pp)")
    fig.tight_layout()
    ctx.save_fig(fig, "pairwise_methods",
                 "Hodges-Lehmann median difference in porosity (pp) between every pair of algorithms (row minus "
                 "column; disc ROI, no post-processing, n = 9 images). * unadjusted exact Wilcoxon p < 0.05; "
                 "** Holm-adjusted p < 0.05 (unattainable with n = 9 and 28 comparisons).", kind="supp")
    plt.close(fig)
    return fr, a_pp, pw


# ============================================================================ 2. factorial / variance decomposition
def _factorial(ctx, seg_all, sp):
    K = ctx.add_key
    P = ctx.primary_roi
    ap = _pp(seg_all[(seg_all["roi"] == P) & (seg_all["post"].isin(AREA_PRESERVING))])
    a2 = S.rm_anova_twoway(ap, "pp", "sample_id", "method", "post")
    for r in a2.itertuples():
        K(f"inference_rmanova2_{r.term.replace(':', 'x')}", dict(F=r.F, df1=r.df1, df2=r.df2, p=r.p, eps_gg=r.eps_gg,
                                                                  p_gg=r.p_gg, eta2_g=r.eta2_generalized),
          f"two-way RM-ANOVA (method x post-processing), term {r.term}, porosity in pp, GG-corrected")
    out = {}
    factors = ["sample_id", "method", "post"]
    for scale in ("pp", "logit"):
        vd = S.variance_decomposition(ap, scale, factors, n_boot=ctx.n_boot // 5 if not ctx.quick else 100,
                                      rng=ctx.rng("inference", "vd", scale))
        out[scale] = vd
        v = vd.set_index("term")
        K(f"inference_vd_{scale}_analyst_pct", [float(v.at["analyst (total)", "pct"]), float(v.at["analyst (total)", "lo"]),
                                                float(v.at["analyst (total)", "hi"])],
          f"share of the total sum of squares of porosity ({scale}) due to analyst choices (method, post-processing "
          "and all their interactions incl. with the image) with 95 % cluster-bootstrap CI; disc ROI, n = 9", "%")
        for term in ("sample_id", "method", "post", "sample_id:method"):
            K(f"inference_vd_{scale}_{term.replace(':', 'x')}_pct", [float(v.at[term, "pct"]), float(v.at[term, "lo"]),
                                                                     float(v.at[term, "hi"])],
              f"share of total SS ({scale}) of the term {term} with 95 % CI", "%")
    # four factors with ROI design
    a4 = _pp(seg_all[seg_all["post"].isin(AREA_PRESERVING)])
    vd4 = S.variance_decomposition(a4, "pp", ["sample_id", "method", "post", "roi"], n_boot=200 if not ctx.quick else 50,
                                   rng=ctx.rng("inference", "vd4"))
    out["roi"] = vd4
    v4 = vd4.set_index("term")
    K("inference_vd4_analyst_pct", [float(v4.at["analyst (total)", "pct"]), float(v4.at["analyst (total)", "lo"]),
                                    float(v4.at["analyst (total)", "hi"])],
      "share of total SS (pp) due to analyst choices when the ROI design is added as a fourth factor", "%")
    roi_terms = v4[[("roi" in t) for t in v4.index]]["pct"].sum()
    K("inference_vd4_roi_terms_pct", float(roi_terms), "summed share of all terms involving the ROI design (pp scale)", "%")
    # leave-one-image-out
    loo = []
    for sid in ctx.sample_ids:
        vv = S.variance_decomposition(ap[ap["sample_id"] != sid], "pp", factors).set_index("term")
        loo.append(dict(left_out=ctx.label(sid), analyst_pct=vv.at["analyst (total)", "pct"],
                        method_pct=vv.at["method", "pct"], interaction_pct=vv.at["sample_id:method", "pct"]))
    loo = pd.DataFrame(loo)
    out["loo"] = loo
    K("inference_vd_loo_analyst_range_pct", [float(loo["analyst_pct"].min()), float(loo["analyst_pct"].max())],
      "range of the analyst-choice share (pp scale) when each image is left out in turn", "%")
    # without the two algorithms that fail on unimodal images
    nf = ap[~ap["method"].isin(["Yen", "Triangle"])]
    vnf = S.variance_decomposition(nf, "pp", factors, n_boot=ctx.n_boot // 5 if not ctx.quick else 100,
                                   rng=ctx.rng("inference", "vd_nofail")).set_index("term")
    out["nofail"] = vnf
    K("inference_vd_nofail_analyst_pct", [float(vnf.at["analyst (total)", "pct"]), float(vnf.at["analyst (total)", "lo"]),
                                          float(vnf.at["analyst (total)", "hi"])],
      "analyst-choice share (pp) excluding Yen and Triangle (the algorithms that collapse on unimodal histograms)", "%")
    # per-image dispersion vs spatial SE
    b = ap[ap["post"] == "none"]
    disp = b.groupby(["sample_id", "label"])["pp"].agg(["std", "min", "max"]).reset_index()
    disp["range"] = disp["max"] - disp["min"]
    se = sp[(sp["roi"] == P) & (sp["post"] == "none")].groupby("sample_id")["se_block"].median() * 100
    disp["se_block_median_pp"] = disp["sample_id"].map(se)
    disp["ratio_sd_to_se"] = disp["std"] / disp["se_block_median_pp"]
    out["disp"] = disp
    K("inference_intermethod_range_pp", {r.label: round(float(r.range), 1) for r in disp.itertuples()},
      "range of porosity over the 8 methods per image (disc ROI, post none)", "pp")
    K("inference_intermethod_sd_over_spatial_se", [float(disp["ratio_sd_to_se"].min()), float(disp["ratio_sd_to_se"].median()),
                                                   float(disp["ratio_sd_to_se"].max())],
      "ratio of the inter-method SD of porosity to the median block-bootstrap spatial SE, per image (min, median, max)")
    # tables
    def fmt(vd):
        t = vd[["term", "df", "pct", "lo", "hi", "source"]].copy()
        t["term"] = t["term"].str.replace("sample_id", "image").str.replace(":", " × ")
        return t.rename(columns={"term": "Term", "df": "df", "pct": "Share of total SS (%)", "lo": "95 % CI low",
                                 "hi": "95 % CI high", "source": "Source"})
    t = fmt(out["pp"]).merge(fmt(out["logit"])[["Term", "Share of total SS (%)", "95 % CI low", "95 % CI high"]],
                             on="Term", suffixes=(" pp", " logit"))
    ctx.save_table(t, "variance_components",
                   "Decomposition of the total variability of porosity over the full factorial design (9 images × 8 "
                   "algorithms × 3 area-preserving post-processing levels, disc ROI) into the material (image) term and "
                   "the analyst-choice terms (algorithm, post-processing and all interactions, including those with the "
                   "image), on the porosity scale (pp) and on the logit scale. Shares of the total sum of squares; 95 % "
                   "percentile CIs from a cluster bootstrap over images. The three-way term is the residual of the "
                   "saturated model.", kind="main", float_format="{:.1f}")
    t4 = fmt(vd4)
    ctx.save_table(t4, "variance_components_roi",
                   "As the variance-component table, adding the ROI design (largest disc vs central 500 × 500 px) as a "
                   "fourth factor (porosity in pp).", kind="supp", float_format="{:.1f}")
    lt = loo.rename(columns={"left_out": "Image left out", "analyst_pct": "Analyst share (%)",
                             "method_pct": "Algorithm share (%)", "interaction_pct": "Image × algorithm share (%)"})
    ctx.save_table(lt, "variance_components_loo",
                   "Leave-one-image-out influence on the variance decomposition (porosity in pp).", kind="supp",
                   float_format="{:.1f}")
    dt = disp.rename(columns={"label": "Image", "std": "Inter-method SD (pp)", "min": "Min (%)", "max": "Max (%)",
                              "range": "Range (pp)", "se_block_median_pp": "Spatial SE (pp)",
                              "ratio_sd_to_se": "SD / spatial SE"}).drop(columns="sample_id")
    ctx.save_table(dt, "intermethod_dispersion",
                   "Inter-method dispersion of porosity per image (8 algorithms, disc ROI, no post-processing) against "
                   "the spatial sampling uncertainty (median over algorithms of the block-bootstrap SE).", kind="supp",
                   float_format="{:.2f}")
    _fig_variance(ctx, out)
    return out


def _fig_variance(ctx, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", 62), sharey=True)
    for ax, scale, ttl in zip(axes, ("pp", "logit"), ("Porosity (pp)", "Logit porosity")):
        vd = out[scale]
        vd = vd[vd["order"] < 99].copy()
        vd["lab"] = vd["term"].str.replace("sample_id", "Image").str.replace("method", "Algorithm") \
            .str.replace("post", "Post-processing").str.replace(":", " × ")
        y = np.arange(len(vd))[::-1]
        col = [style.MUTED if s == "material" else style.METHOD_COLORS["Otsu"] for s in vd["source"]]
        ax.barh(y, vd["pct"], color=col, height=0.6)
        ax.errorbar(vd["pct"], y, xerr=[vd["pct"] - vd["lo"], vd["hi"] - vd["pct"]], fmt="none", ecolor=style.INK,
                    lw=0.7, capsize=1.5)
        ax.set_yticks(y, vd["lab"])
        ax.set_xlabel("Share of total sum of squares (%)")
        ax.set_title(ttl, fontsize=7.5)
        style.hgrid(ax, "x")
        tot = out[scale].set_index("term")
        ax.text(0.98, 0.04, f"Analyst choices: {tot.at['analyst (total)', 'pct']:.0f} % "
                            f"[{tot.at['analyst (total)', 'lo']:.0f}–{tot.at['analyst (total)', 'hi']:.0f}]\n"
                            f"Material: {tot.at['material (total)', 'pct']:.0f} % "
                            f"[{tot.at['material (total)', 'lo']:.0f}–{tot.at['material (total)', 'hi']:.0f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=style.GRID, lw=0.5))
    for k, ax in enumerate(axes):
        style.panel_label(ax, "ab"[k])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=style.MUTED, label="Material (image)"),
                        Patch(color=style.METHOD_COLORS["Otsu"], label="Analyst choices")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    ctx.save_fig(fig, "variance_decomposition",
                 "Share of the total variability of porosity attributable to each term of the factorial design "
                 "(9 images × 8 algorithms × 3 area-preserving post-processing levels, disc ROI), on the porosity scale "
                 "(a) and on the logit scale (b). Bars: share of the total sum of squares; error bars: 95 % "
                 "cluster-bootstrap CIs over images. Grey: material (image main effect); blue: analyst choices.",
                 kind="main")
    plt.close(fig)


# ============================================================================ 3. central figure
def _fig_porosity_by_method(ctx, seg, sp):
    P = ctx.primary_roi
    base = seg[(seg["roi"] == P) & (seg["post"] == "none")]
    spc = sp[(sp["roi"] == P) & (sp["post"] == "none")].set_index(["sample_id", "method"])
    ref = ctx.lab_ref.set_index("sample_id")
    fig, axes = plt.subplots(3, 3, figsize=style.figsize("double", 150))
    for ax, sid in zip(axes.ravel(), ctx.sample_ids):
        x = np.arange(len(ctx.methods))
        for i, m in enumerate(ctx.methods):
            v = 100 * float(base[(base["sample_id"] == sid) & (base["method"] == m)]["porosity"].iloc[0])
            lo, hi = 100 * spc.at[(sid, m), "ci_lo"], 100 * spc.at[(sid, m), "ci_hi"]
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt=style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m],
                        ms=4, lw=0.8, capsize=1.5, zorder=3)
        if sid in ref.index:
            r = 100 * ref.at[sid, "ref"]
            sd = ref.at[sid, "ref_sd"]
            if np.isfinite(sd):
                ax.axhspan(r - 100 * sd, r + 100 * sd, color=style.REF_BAND, zorder=1, lw=0)
            ax.axhline(r, color=style.INK, lw=0.8, zorder=2)
            if np.isfinite(ref.at[sid, "uct"]):
                ax.axhline(100 * ref.at[sid, "uct"], color=style.INK, lw=0.8, ls="--", zorder=2)
        ax.set_xticks(x, ctx.methods, rotation=60, ha="right")
        ax.set_title(f"{ctx.label(sid)} ({style.GROUP_LABELS[ctx.group(sid)].lower()})", fontsize=7.5)
        style.hgrid(ax)
        ymax = max(ax.get_ylim()[1], 5)
        ax.set_ylim(0, min(100, ymax * 1.05))
    for ax in axes[:, 0]:
        ax.set_ylabel("Porosity (%)")
    for k, ax in enumerate(axes.ravel()):
        style.panel_label(ax, "abcdefghi"[k])
    from matplotlib.patches import Patch
    hs = [Line2D([], [], color=style.INK, lw=0.8, label="Laboratory reference"),
          Patch(color=style.REF_BAND, label="± SD of 4 physical methods"),
          Line2D([], [], color=style.INK, lw=0.8, ls="--", label="Laboratory µCT")]
    fig.legend(handles=style.method_handles() + hs, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ctx.save_fig(fig, "porosity_by_method",
                 "Porosity of each image from the eight thresholding algorithms (disc ROI, no post-processing). Error "
                 "bars: 95 % block-bootstrap intervals of the spatial sampling uncertainty (often smaller than the "
                 "symbol). Solid line: laboratory reference (granular packs a-d: mean of four physical methods, grey "
                 "band ± SD; cores e-g: single value of unstated method); dashed line: laboratory µCT porosity. TQ06 "
                 "and GT01 have no laboratory value. Each panel has its own y-range starting at zero.", kind="main")
    plt.close(fig)


# ============================================================================ 4. laboratory agreement
def _agreement_one(x, y, se_y, sd_x, rng, n_boot):
    ba = S.bland_altman(x, y)
    hl = S.hodges_lehmann(y - x)
    em = S.error_metrics(x, y, n_boot=n_boot, rng=rng)
    cc = S.lin_ccc(x, y)
    w = S.wilcoxon_signed_rank(y - x)
    var_y = np.nanmean(np.asarray(se_y) ** 2)
    var_x = np.nanmean(np.asarray(sd_x) ** 2)
    lam = var_y / var_x if np.isfinite(var_x) and var_x > 0 and np.isfinite(var_y) and var_y > 0 else 1.0
    try:
        dm = S.deming(x, y, lam)
    except Exception:  # noqa: BLE001
        dm = dict(slope=np.nan, slope_lo=np.nan, slope_hi=np.nan, intercept=np.nan)
    pb = S.passing_bablok(x, y)
    return dict(n=len(x), bias=ba["bias"], bias_lo=ba["bias_lo"], bias_hi=ba["bias_hi"], hl=hl["estimate"],
                hl_lo=hl["lo"], hl_hi=hl["hi"], sd_diff=ba["sd"], loa_lo=ba["loa_lo"], loa_hi=ba["loa_hi"],
                loa_lo_ci=ba["loa_lo_ci_exact"], loa_hi_ci=ba["loa_hi_ci_exact"], mae=em["mae"],
                mae_lo=em.get("mae_lo"), mae_hi=em.get("mae_hi"), rmse=em["rmse"], rmse_lo=em.get("rmse_lo"),
                rmse_hi=em.get("rmse_hi"), ccc=cc["ccc"], ccc_lo=cc["lo"], ccc_hi=cc["hi"], r=cc["r"],
                p_wilcoxon=w["p"], deming_slope=dm["slope"], deming_lo=dm["slope_lo"], deming_hi=dm["slope_hi"],
                deming_lambda=lam, pb_slope=pb["slope"], pb_lo=pb.get("slope_lo"), pb_hi=pb.get("slope_hi"))


def _lab(ctx, seg, sp):
    K = ctx.add_key
    P = ctx.primary_roi
    ref = ctx.lab_ref.set_index("sample_id")
    with_ref = [s for s in ctx.sample_ids if s in ref.index and np.isfinite(ref.at[s, "ref"])]
    spc = sp[(sp["roi"] == P)].set_index(["sample_id", "method", "post"])["se_block"]
    pack_sd = float(np.sqrt(np.nanmean(ref.loc[[s for s in with_ref if ctx.group(s) == "granular_pack"], "ref_sd"] ** 2)))
    rows, err_cols = [], {}
    for post in AREA_PRESERVING:
        b = seg[(seg["roi"] == P) & (seg["post"] == post)]
        for m in ctx.methods:
            for grp in ("all", "granular_pack", "consolidated"):
                sids = [s for s in with_ref if grp == "all" or ctx.group(s) == grp]
                x = 100 * np.array([ref.at[s, "ref"] for s in sids])
                y = 100 * np.array([float(b[(b["sample_id"] == s) & (b["method"] == m)]["porosity"].iloc[0]) for s in sids])
                se_y = 100 * np.array([spc.get((s, m, post), np.nan) for s in sids])
                sd_x = 100 * np.array([ref.at[s, "ref_sd"] if np.isfinite(ref.at[s, "ref_sd"]) else pack_sd for s in sids])
                a = _agreement_one(x, y, se_y, sd_x, ctx.rng("inference", "lab", post, m, grp), ctx.n_boot)
                rows.append(dict(post=post, method=m, group=grp, **a))
                if post == "none" and grp == "all":
                    err_cols[m] = y - x
    ag = pd.DataFrame(rows)
    pbst = S.prob_best(pd.DataFrame(err_cols), rng=ctx.rng("inference", "probbest"), n_boot=ctx.n_boot)
    ag["prob_best"] = np.where((ag["post"] == "none") & (ag["group"] == "all"), ag["method"].map(pbst), np.nan)
    ctx.save_data(ag.assign(loa_lo_ci=ag["loa_lo_ci"].astype(str), loa_hi_ci=ag["loa_hi_ci"].astype(str)), "lab_agreement")
    # per-image standardised differences
    z = []
    b = seg[(seg["roi"] == P) & (seg["post"] == "none")]
    for s in with_ref:
        for m in ctx.methods:
            phi = float(b[(b["sample_id"] == s) & (b["method"] == m)]["porosity"].iloc[0])
            seb = spc.get((s, m, "none"), np.nan)
            rse = ref.at[s, "ref_se"]
            zz = (phi - ref.at[s, "ref"]) / np.sqrt(seb ** 2 + rse ** 2) if np.isfinite(rse) else np.nan
            z.append(dict(sample_id=s, label=ctx.label(s), method=m, diff_pp=100 * (phi - ref.at[s, "ref"]), z=zz))
    ctx.save_data(pd.DataFrame(z), "lab_standardised_differences")
    # main table (post none, all 7)
    a0 = ag[(ag["post"] == "none") & (ag["group"] == "all")].copy()

    def ci(l, h, d=1):
        return f"[{l:.{d}f}, {h:.{d}f}]"
    tab = pd.DataFrame({"Method": a0["method"],
                        "Bias (pp)": a0["bias"].round(1).astype(str) + " " + [ci(l, h) for l, h in zip(a0["bias_lo"], a0["bias_hi"])],
                        "HL bias (pp)": a0["hl"].round(1).astype(str) + " " + [ci(l, h) for l, h in zip(a0["hl_lo"], a0["hl_hi"])],
                        "95 % LoA (pp)": [f"{l:.1f} to {h:.1f}" for l, h in zip(a0["loa_lo"], a0["loa_hi"])],
                        "MAE (pp)": a0["mae"].round(1).astype(str) + " " + [ci(l, h) for l, h in zip(a0["mae_lo"], a0["mae_hi"])],
                        "RMSE (pp)": a0["rmse"].round(1),
                        "CCC": [f"{c:.2f} " + (ci(l, h, 2) if np.isfinite(l) else "") for c, l, h in zip(a0["ccc"], a0["ccc_lo"], a0["ccc_hi"])],
                        "Exact p (bias)": a0["p_wilcoxon"].round(3),
                        "P(best MAE)": a0["prob_best"].round(2)})
    ctx.save_table(tab, "lab_agreement",
                   f"Agreement between image porosity and the laboratory reference for the {len(with_ref)} images with a "
                   "reference (4 granular packs with four physical methods; 3 cores with a single value of unstated "
                   "method), disc ROI, no post-processing. Bias = image − laboratory with t-based 95 % CI; HL = "
                   "Hodges-Lehmann estimate with exact CI; LoA = Bland-Altman limits of agreement; MAE with "
                   "bootstrap CI over images; CCC = Lin's concordance correlation with Fisher-z CI; exact Wilcoxon "
                   "signed-rank p of the bias; P(best MAE) = bootstrap probability of having the smallest MAE. With "
                   "n = 7 images of two very different materials these are descriptive.", kind="main")
    t2 = ag[ag["post"] != "none"][["post", "method", "group", "n", "bias", "bias_lo", "bias_hi", "mae", "rmse", "ccc"]]
    ctx.save_table(t2, "lab_agreement_postprocessing",
                   "Agreement with the laboratory reference after median and opening-closing post-processing, overall "
                   "(n = 7) and per material group (packs n = 4, cores n = 3).", kind="supp", float_format="{:.2f}")
    t3 = ag[(ag["post"] == "none")][["method", "group", "n", "bias", "mae", "ccc", "deming_slope", "deming_lo",
                                     "deming_hi", "deming_lambda", "pb_slope", "pb_lo", "pb_hi"]]
    ctx.save_table(t3, "lab_agreement_regression",
                   "Bias, MAE and CCC per material group and Deming / Passing-Bablok regression slopes (image on "
                   "laboratory; Deming error-variance ratio λ from the block-bootstrap SE of the image and the SD of the "
                   "laboratory methods). n = 7 (or 3-4 per group) is far below what these regressions need; they are "
                   "reported for completeness only.", kind="supp", float_format="{:.2f}")
    best = a0.sort_values("mae").iloc[0]
    K("inference_lab_best_mae_method", str(best["method"]), "algorithm with the smallest MAE vs laboratory (n = 7, post none)")
    for r in a0.itertuples():
        K(f"inference_lab_{r.method.lower()}", dict(bias=r.bias, bias_lo=r.bias_lo, bias_hi=r.bias_hi, mae=r.mae,
                                                    mae_lo=r.mae_lo, mae_hi=r.mae_hi, ccc=r.ccc, p=r.p_wilcoxon,
                                                    prob_best=r.prob_best),
          f"agreement of {r.method} with the laboratory reference (pp; n = 7; disc ROI, post none)")
    for grp in ("granular_pack", "consolidated"):
        g = ag[(ag["post"] == "none") & (ag["group"] == grp)]
        K(f"inference_lab_bias_range_{grp}_pp", [float(g["bias"].min()), float(g["bias"].max())],
          f"range over 8 methods of the mean bias (image − lab) for {grp} images", "pp")
    _fig_lab(ctx, seg, ref, with_ref, a0)
    return ag


def _fig_lab(ctx, seg, ref, with_ref, a0) -> None:
    P = ctx.primary_roi
    b = seg[(seg["roi"] == P) & (seg["post"] == "none")]
    fig = plt.figure(figsize=style.figsize("double", 75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(gs[0])
    for m in ctx.methods:
        for s in with_ref:
            x = 100 * ref.at[s, "ref"]
            y = 100 * float(b[(b["sample_id"] == s) & (b["method"] == m)]["porosity"].iloc[0])
            filled = ctx.group(s) == "granular_pack"
            ax.scatter(x, y, s=12, marker=style.METHOD_MARKERS[m], facecolor=style.METHOD_COLORS[m] if filled else "white",
                       edgecolor=style.METHOD_COLORS[m], lw=0.7, zorder=3)
    lim = [0, 100]
    ax.plot(lim, lim, color=style.INK, lw=0.6)
    ax.set_xlim(10, 45)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Laboratory porosity (%)")
    ax.set_ylabel("Image porosity (%)")
    style.hgrid(ax)
    ax.text(0.03, 0.97, "filled: granular packs\nopen: consolidated cores", transform=ax.transAxes, va="top", fontsize=6.3)
    y = np.arange(len(ctx.methods))[::-1]
    a = a0.set_index("method").reindex(ctx.methods)
    ax2 = fig.add_subplot(gs[1])
    for yy, m in zip(y, ctx.methods):
        ax2.errorbar(a.at[m, "bias"], yy, xerr=[[a.at[m, "bias"] - a.at[m, "bias_lo"]], [a.at[m, "bias_hi"] - a.at[m, "bias"]]],
                     fmt=style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m], ms=4, lw=0.8, capsize=1.5)
    ax2.axvline(0, color=style.INK, lw=0.6)
    ax2.set_yticks(y, ctx.methods)
    ax2.set_xlabel("Bias, image − laboratory (pp)")
    style.hgrid(ax2, "x")
    ax3 = fig.add_subplot(gs[2], sharey=ax2)
    for yy, m in zip(y, ctx.methods):
        c, lo, hi = a.at[m, "ccc"], a.at[m, "ccc_lo"], a.at[m, "ccc_hi"]
        if np.isfinite(lo):
            ax3.errorbar(c, yy, xerr=[[c - lo], [hi - c]], fmt=style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m],
                         ms=4, lw=0.8, capsize=1.5)
        else:
            ax3.plot(c, yy, style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m], ms=4)
    ax3.axvline(0, color=style.INK, lw=0.6)
    ax3.set_xlabel("Lin's concordance (CCC)")
    plt.setp(ax3.get_yticklabels(), visible=False)
    style.hgrid(ax3, "x")
    for k, axx in enumerate((ax, ax2, ax3)):
        style.panel_label(axx, "abc"[k])
    fig.legend(handles=style.method_handles(), loc="lower center", ncol=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    ctx.save_fig(fig, "lab_agreement",
                 f"Agreement with the laboratory reference ({len(with_ref)} images; disc ROI, no post-processing). (a) "
                 "Image versus laboratory porosity for every algorithm (identity line; filled symbols: granular packs, "
                 "open: consolidated cores). (b) Mean bias with t-based 95 % CI. (c) Lin's concordance correlation "
                 "coefficient with Fisher-z 95 % CI.", kind="main")
    plt.close(fig)
    # Bland-Altman small multiples
    fig, axes = plt.subplots(2, 4, figsize=style.figsize("double", 95), sharey=True)
    for axx, m in zip(axes.ravel(), ctx.methods):
        xs, ds = [], []
        for s in with_ref:
            x = 100 * ref.at[s, "ref"]
            yv = 100 * float(b[(b["sample_id"] == s) & (b["method"] == m)]["porosity"].iloc[0])
            filled = ctx.group(s) == "granular_pack"
            axx.scatter(x, yv - x, s=12, marker=style.METHOD_MARKERS[m], lw=0.7,
                        facecolor=style.METHOD_COLORS[m] if filled else "white", edgecolor=style.METHOD_COLORS[m])
        r = a.loc[m]
        axx.axhline(r["bias"], color=style.INK, lw=0.8)
        axx.axhline(r["loa_lo"], color=style.INK, lw=0.6, ls="--")
        axx.axhline(r["loa_hi"], color=style.INK, lw=0.6, ls="--")
        axx.axhline(0, color=style.MUTED, lw=0.5)
        axx.set_title(m, fontsize=7.5)
        style.hgrid(axx)
    for axx in axes[-1]:
        axx.set_xlabel("Laboratory porosity (%)")
    for axx in axes[:, 0]:
        axx.set_ylabel("Image − laboratory (pp)")
    fig.tight_layout()
    ctx.save_fig(fig, "bland_altman",
                 "Bland-Altman plots against the laboratory reference (x = reference, following Krouwer 2008): "
                 "difference image − laboratory for the 7 images with a reference (filled: granular packs; open: "
                 "cores), mean bias (solid) and 95 % limits of agreement (dashed). Disc ROI, no post-processing.",
                 kind="supp")
    plt.close(fig)


# ============================================================================ 5. laboratory-internal
def _lab_internal(ctx, seg):
    K = ctx.add_key
    r = ctx.lab_ref[ctx.lab_ref["ref_kind"] == "multi-method"].copy()
    x, y = 100 * r["ref"].to_numpy(), 100 * r["uct"].to_numpy()
    ba = S.bland_altman(x, y)
    hl = S.hodges_lehmann(y - x)
    w = S.wilcoxon_signed_rank(y - x)
    lab = ctx.lab[ctx.lab["sample_id"].isin(r["sample_id"]) & ctx.lab["method"].isin(ctx.cfg["lab"]["reference_methods"])]
    within = lab.groupby("sample_id")["porosity_pct"].var(ddof=1)
    pooled_sd = float(np.sqrt(within.mean()))
    meth_means = lab.groupby("method")["porosity_pct"].mean()
    t = pd.DataFrame({"Specimen": r["specimen"], "Image available": np.where(r["has_image"], "yes", "no"),
                      "Physical mean (%)": 100 * r["ref"], "SD of 4 methods (pp)": 100 * r["ref_sd"],
                      "Laboratory µCT (%)": 100 * r["uct"], "µCT − physical (pp)": y - x,
                      "Legacy reference, mean of 5 (%)": 100 * r["mean_all5"]})
    ctx.save_table(t, "lab_internal",
                   f"Laboratory-internal comparison for the {len(r)} granular specimens: mean of the four physical "
                   "methods (direct, gravimetric, indirect, gas), their SD, the laboratory's own µCT porosity and the "
                   "legacy reference used by the original code (mean of all five values, µCT included). µCT minus "
                   f"physical: Hodges-Lehmann {hl['estimate']:.1f} pp (exact 95 % CI {hl['lo']:.1f} to {hl['hi']:.1f}; "
                   f"exact Wilcoxon p = {w['p']:.4f}); pooled within-specimen SD of the physical methods "
                   f"{pooled_sd:.2f} pp.", kind="supp", float_format="{:.2f}")
    K("inference_lab_uct_minus_physical_hl_pp", [hl["estimate"], hl["lo"], hl["hi"]],
      f"laboratory µCT minus mean of physical methods, Hodges-Lehmann with exact CI (n = {len(r)} granular specimens)", "pp")
    K("inference_lab_uct_minus_physical_p", w["p"], "exact Wilcoxon p of the laboratory µCT bias")
    K("inference_lab_uct_bias_mean_sd", [ba["bias"], ba["sd"]], "mean and SD of laboratory µCT − physical mean", "pp")
    K("inference_lab_physical_pooled_sd_pp", pooled_sd, "pooled within-specimen SD of the four physical methods", "pp")
    K("inference_lab_physical_method_means", {k: round(float(v), 2) for k, v in meth_means.items()},
      "mean porosity of each physical method over the granular specimens", "%")
    # effect of the reference definition on agreement
    P = ctx.primary_roi
    b = seg[(seg["roi"] == P) & (seg["post"] == "none")]
    rr = ctx.lab_ref.set_index("sample_id")
    packs = [s for s in ctx.sample_ids if ctx.group(s) == "granular_pack" and s in rr.index]
    rows = []
    for m in ctx.methods:
        yi = 100 * np.array([float(b[(b["sample_id"] == s) & (b["method"] == m)]["porosity"].iloc[0]) for s in packs])
        for name, col in (("physical mean (corrected)", "ref"), ("mean of 5 incl. µCT (legacy)", "mean_all5")):
            xx = 100 * rr.loc[packs, col].to_numpy()
            rows.append({"Method": m, "Reference": name, "Bias (pp)": float(np.mean(yi - xx)),
                         "MAE (pp)": float(np.mean(np.abs(yi - xx)))})
    ctx.save_table(pd.DataFrame(rows), "reference_definition_effect",
                   "Effect of the definition of the laboratory reference on the agreement of the four granular packs: "
                   "corrected reference (mean of the four physical methods) versus the legacy reference that also "
                   "averaged the laboratory µCT value (disc ROI, no post-processing).", kind="supp", float_format="{:.2f}")
    fig, ax = plt.subplots(figsize=style.figsize("single", 70))
    ax.scatter(x, y, s=16, color=style.INK, zorder=3)
    for xi, yi, lab_ in zip(x, y, r["specimen"]):
        ax.annotate(lab_.split("_")[0], (xi, yi), fontsize=5.8, xytext=(3, 2), textcoords="offset points")
    lim = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lim, lim, color=style.MUTED, lw=0.6)
    ax.set_xlabel("Mean of four physical methods (%)")
    ax.set_ylabel("Laboratory µCT porosity (%)")
    style.hgrid(ax)
    fig.tight_layout()
    ctx.save_fig(fig, "lab_internal",
                 f"Laboratory µCT porosity against the mean of the four physical methods for the {len(r)} granular "
                 "specimens (identity line in grey).", kind="supp")
    plt.close(fig)


# ============================================================================ 6. consistency index
def _consistency(ctx, sp):
    P = ctx.primary_roi
    c = sp[(sp["roi"] == P) & (sp["post"] == "none")].copy()
    c["CI_spatial"] = 1 - c["cv_tiles"]
    piv = c.pivot(index="label", columns="method", values="CI_spatial").reindex(
        index=[ctx.label(s) for s in ctx.sample_ids], columns=ctx.methods)
    ctx.save_table(piv.reset_index().rename(columns={"label": "Image"}), "consistency_index",
                   "The 'Consistency Index' of the original manuscript recomputed as CI = 1 − CV of the porosity of the "
                   "block-bootstrap tiles (disc ROI, no post-processing). It is one minus a coefficient of variation: "
                   "it measures the spatial heterogeneity of the segmented image, not accuracy, and it becomes "
                   "negative or undefined when the mean porosity approaches zero.", kind="supp", float_format="{:.2f}")


def run(ctx) -> None:
    seg = ctx.seg
    sp = ctx.read_data("spatial_ci")
    base = _pp(seg[(seg["roi"] == ctx.primary_roi) & (seg["post"] == "none")])
    _method_effect(ctx, base)
    _factorial(ctx, seg, sp)
    _fig_porosity_by_method(ctx, seg, sp)
    _lab(ctx, seg, sp)
    _lab_internal(ctx, seg)
    _consistency(ctx, sp)
    fails = base[(base["porosity"] < FAIL_LO) | (base["porosity"] > FAIL_HI)]
    ctx.add_key("inference_failures", [f"{r.label}/{r.method}: {100 * r.porosity:.1f} %" for r in fails.itertuples()],
                f"image/method combinations with porosity < {100 * FAIL_LO:g} % or > {100 * FAIL_HI:g} % (disc ROI, post none)")
