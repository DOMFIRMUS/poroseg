"""Stage 'synthetic': benchmark of the eight algorithms on phantoms with known ground truth.

The real images have no pixel-level truth, so accuracy claims (for example
that local adaptive thresholds are more robust to partial volume and beam
hardening) cannot be tested on them. Here the three phantom families of
:mod:`poroseg.synthetic` ('grain_pack', 'boolean_pores', 'multiscale') are
degraded with every combination of PSF blur, noise and cupping listed in
``config.yaml`` (``synthetic.*``), normalised exactly like the real images and
segmented by the same eight algorithms and three area-preserving
post-processing levels. Each condition is replicated with ``n_rep`` independent
phantoms (common random numbers: the same phantoms are reused at every
degradation level, which sharpens the comparisons between levels).

Accuracy metrics (inside the evaluation disc): porosity error against the
pixel-grid truth (and, for 'multiscale', against the total porosity that
includes the sub-resolution micropores), Dice and Jaccard overlap with the
truth and the relative error of the Euler number (topology).

Statistics
----------
* Per method: median absolute error and Dice with IQR over all conditions and
  replicates; mean Friedman rank over the conditions (subject = condition,
  replicate-averaged |error|), Kendall's W.
* A linear model of |error| (post-processing none) on method, phantom, blur,
  noise and cupping with all two-way interactions; partial eta-squared of each
  term with a percentile CI from a cluster bootstrap that resamples whole
  replicate phantoms (Davison & Hinkley 1997). With replicated synthetic data
  the model has real residual degrees of freedom, unlike the real-image design.
* Global versus local families: per condition, mean |error| of the six global
  and of the two local methods, compared with the exact Wilcoxon signed-rank
  test and the Hodges-Lehmann estimate (subject = condition), overall and
  separately for the cupped and uncupped conditions.

Outputs: data/synthetic_benchmark.csv (tidy, one row per image x method x post),
figures synthetic_examples (supp), synthetic_benchmark (main), synthetic_dice
(supp); tables synthetic_benchmark_summary (main), synthetic_anova (supp),
synthetic_family (supp); key numbers prefixed ``synthetic_``.
"""
from __future__ import annotations

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import stats as S
from .. import style, synthetic as syn
from ..postprocess import AREA_PRESERVING
from ..thresholds import GLOBAL_METHODS, LOCAL_METHODS

PHANTOM_LABELS = {"grain_pack": "Grain pack", "boolean_pores": "Boolean pores", "multiscale": "Multiscale"}


def _conditions(cfg):
    sc = cfg["synthetic"]
    return [dict(phantom=k, blur=float(b), noise=float(n), cupping=float(c))
            for k in sc["phantoms"] for b in sc["blur_sigma"] for n in sc["noise_sd"] for c in sc["cupping"]]


def _simulate(ctx) -> pd.DataFrame:
    sc = ctx.cfg["synthetic"]
    size = int(sc["size_px"])
    n_rep = 2 if ctx.quick else int(sc["n_rep"])
    nrm = ctx.cfg["normalization"]
    eb = int(ctx.cfg["roi"]["eval_border_px"])
    rows = []
    t0 = time.time()
    for cond in _conditions(ctx.cfg):
        for rep in range(n_rep):
            case = syn.generate_case(cond["phantom"], cond["blur"], cond["noise"], cond["cupping"], rep, ctx.seed,
                                     size, p_low=float(nrm["p_low"]), p_high=float(nrm["p_high"]), eval_border=eb)
            ev = case["eval_mask"]
            chi_true = syn.euler2d(case["truth"], ev)
            segs = syn.segment_all(case, ctx.cfg, ctx.methods, AREA_PRESERVING)
            for m in ctx.methods:
                fam = "local" if m in LOCAL_METHODS else "global"
                for p in AREA_PRESERVING:
                    met = syn.segmentation_metrics(segs[(m, p)], case["truth"], ev, case["phi_pixel"],
                                                   case["phi_total"], chi_true)
                    rows.append(dict(**cond, rep=rep, method=m, family=fam, post=p,
                                     true_porosity=case["phi_pixel"], true_porosity_total=case["phi_total"],
                                     **met))
    ctx.log("simulated %d images in %.1fs", len(rows) // (len(ctx.methods) * len(AREA_PRESERVING)), time.time() - t0)
    return pd.DataFrame(rows)


def _partial_eta2(d: pd.DataFrame) -> pd.Series:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    f = "abs_error_pp ~ (C(method) + C(phantom) + C(blur) + C(noise) + C(cupping))**2"
    a = anova_lm(smf.ols(f, data=d).fit(), typ=2)
    ss_res = a.loc["Residual", "sum_sq"]
    terms = a.drop(index="Residual")
    return (terms["sum_sq"] / (terms["sum_sq"] + ss_res)).rename(lambda t: t.replace("C(", "").replace(")", ""))


def _anova(ctx, d0: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    f = "abs_error_pp ~ (C(method) + C(phantom) + C(blur) + C(noise) + C(cupping))**2"
    a = anova_lm(smf.ols(f, data=d0).fit(), typ=2)
    ss_res = a.loc["Residual", "sum_sq"]
    tab = a.drop(index="Residual").copy()
    tab.index = [t.replace("C(", "").replace(")", "") for t in tab.index]
    tab["eta2_partial"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    tab["share_total_pct"] = 100 * tab["sum_sq"] / a["sum_sq"].sum()
    # cluster bootstrap over replicate phantoms (rep is shared by all conditions of a phantom kind)
    reps = sorted(d0["rep"].unique())
    nb = 40 if ctx.quick else 300
    rng = ctx.rng("synthetic", "anova_boot")
    boots = []
    groups = {r: d0[d0["rep"] == r] for r in reps}
    for _ in range(nb):
        pick = rng.choice(reps, len(reps), replace=True)
        parts = []
        for k, r in enumerate(pick):
            g = groups[r].copy()
            g["rep"] = k
            parts.append(g)
        try:
            boots.append(_partial_eta2(pd.concat(parts, ignore_index=True)))
        except Exception:  # noqa: BLE001
            continue
    B = pd.DataFrame(boots)
    tab["lo"] = B.quantile(0.025).reindex(tab.index)
    tab["hi"] = B.quantile(0.975).reindex(tab.index)
    tab.attrs.update(df_resid=float(a.loc["Residual", "df"]), n_boot=nb)
    return tab.reset_index(names="term")


def _family(d0: pd.DataFrame) -> pd.DataFrame:
    keys = ["phantom", "blur", "noise", "cupping"]
    g = d0.groupby(keys + ["family"])["abs_error_pp"].mean().unstack("family").reset_index()
    rows = []
    for label, sub in (("all conditions", g), ("no cupping", g[g["cupping"] == 0]), ("with cupping", g[g["cupping"] > 0])):
        dlt = (sub["local"] - sub["global"]).to_numpy()
        hl = S.hodges_lehmann(dlt)
        w = S.wilcoxon_signed_rank(dlt)
        rows.append({"Conditions": label, "n": len(dlt), "Mean |error| global (pp)": float(sub["global"].mean()),
                     "Mean |error| local (pp)": float(sub["local"].mean()),
                     "HL local − global (pp)": hl["estimate"], "CI low": hl["lo"], "CI high": hl["hi"],
                     "Wilcoxon p": w["p"], "Local better (%)": 100 * float(np.mean(dlt < 0))})
    return pd.DataFrame(rows)


def run(ctx) -> None:
    d = _simulate(ctx)
    ctx.save_data(d, "synthetic_benchmark")
    d0 = d[d["post"] == "none"].copy()
    keys = ["phantom", "blur", "noise", "cupping"]

    # ------------------------------------------------------------------ per-method summary
    cond = d0.groupby(keys + ["method"])["abs_error_pp"].mean().unstack("method")[ctx.methods]
    ranks = cond.rank(axis=1)
    best = cond.eq(cond.min(axis=1), axis=0)
    long_c = cond.reset_index().melt(id_vars=keys, var_name="method", value_name="mae")
    long_c["condition"] = long_c[keys].astype(str).agg("|".join, axis=1)
    fr = S.friedman(long_c, "mae", "method", "condition")
    rows = []
    for m in ctx.methods:
        dm = d0[d0["method"] == m]
        q = dm["abs_error_pp"].quantile([0.25, 0.5, 0.75])
        qd = dm["dice"].quantile([0.25, 0.5, 0.75])
        r = {"Method": m, "Family": "local" if m in LOCAL_METHODS else "global",
             "Median |error| (pp)": q[0.5], "IQR |error| (pp)": f"{q[0.25]:.1f}–{q[0.75]:.1f}",
             "Median Dice": qd[0.5], "IQR Dice": f"{qd[0.25]:.2f}–{qd[0.75]:.2f}",
             "Mean rank": float(ranks[m].mean()), "Best in conditions (%)": 100 * float(best[m].mean())}
        for p in ("median", "open_close"):
            r[f"Median |error|, {p} (pp)"] = float(d[(d["method"] == m) & (d["post"] == p)]["abs_error_pp"].median())
        ms = d0[(d0["method"] == m) & (d0["phantom"] == "multiscale")]
        r["Median error vs total φ, multiscale (pp)"] = float(ms["error_total_pp"].median())
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values("Mean rank").reset_index(drop=True)
    n_rep = int(d["rep"].nunique())
    n_cond = len(cond)
    ctx.save_table(summ, "synthetic_benchmark_summary",
                   f"Accuracy of the eight algorithms on synthetic images with known truth ({n_cond} degradation "
                   f"conditions = 3 phantom families × {len(ctx.cfg['synthetic']['blur_sigma'])} blur × "
                   f"{len(ctx.cfg['synthetic']['noise_sd'])} noise × {len(ctx.cfg['synthetic']['cupping'])} cupping levels, "
                   f"{n_rep} replicate phantoms each; post-processing none unless stated). |error| = absolute porosity "
                   "error against the pixel-grid truth; Dice = overlap with the truth; mean rank = mean Friedman rank "
                   f"over conditions (1 = best; Kendall's W = {fr['W']:.2f}); last column: signed error against the total "
                   "porosity of the multiscale phantoms, which includes sub-resolution pores.",
                   kind="main", float_format="{:.2f}")

    an = _anova(ctx, d0)
    at = an[["term", "df", "F", "PR(>F)", "eta2_partial", "lo", "hi", "share_total_pct"]].rename(columns={
        "term": "Term", "df": "df", "PR(>F)": "p", "eta2_partial": "Partial η²", "lo": "CI low", "hi": "CI high",
        "share_total_pct": "Share of total SS (%)"})
    at = at.sort_values("Partial η²", ascending=False)
    ctx.save_table(at, "synthetic_anova",
                   "Linear model of the absolute porosity error (pp) on the synthetic benchmark (post-processing none): "
                   "main effects and all two-way interactions of algorithm, phantom family, blur, noise and cupping "
                   f"(type II sums of squares; residual df = {an.attrs['df_resid']:.0f}). Partial η² with 95 % percentile "
                   f"CIs from a cluster bootstrap over replicate phantoms ({an.attrs['n_boot']} replicates).",
                   kind="supp", float_format="{:.3g}")
    fam = _family(d0)
    ctx.save_table(fam, "synthetic_family",
                   "Local adaptive (Sauvola, Niblack) versus global algorithms on the synthetic benchmark: mean absolute "
                   "porosity error per condition, Hodges-Lehmann estimate of the paired difference local − global with "
                   "exact 95 % CI and exact Wilcoxon signed-rank p (subject = degradation condition, replicate-averaged).",
                   kind="supp", float_format="{:.3g}")

    _fig_examples(ctx)
    _fig_benchmark(ctx, d0)
    _fig_dice(ctx, d0)

    # ------------------------------------------------------------------ key numbers
    K = ctx.add_key
    K("synthetic_n_conditions", n_cond, "number of degradation conditions in the synthetic benchmark")
    K("synthetic_n_rep", n_rep, "replicate phantoms per condition")
    K("synthetic_n_images", int(len(d0) // len(ctx.methods)), "synthetic images segmented")
    K("synthetic_friedman_W", fr["W"], "Kendall's W of the method ranks over conditions (|error|, post none)")
    K("synthetic_friedman_p", fr["p"], "Friedman p over conditions (|error|, post none)")
    for _, r in summ.iterrows():
        m = r["Method"].lower()
        K(f"synthetic_median_abs_error_{m}", float(r["Median |error| (pp)"]),
          f"median absolute porosity error of {r['Method']} over all synthetic images (post none)", "pp")
        K(f"synthetic_median_dice_{m}", float(r["Median Dice"]), f"median Dice of {r['Method']} (post none)")
        K(f"synthetic_best_frac_{m}", float(r["Best in conditions (%)"]),
          f"% of conditions in which {r['Method']} has the lowest mean |error| (post none)", "%")
        K(f"synthetic_multiscale_error_total_{m}", float(r["Median error vs total φ, multiscale (pp)"]),
          f"median signed error of {r['Method']} against the TOTAL porosity (incl. sub-resolution pores) of the "
          "multiscale phantoms (post none)", "pp")
    K("synthetic_best_method_by_rank", str(summ.iloc[0]["Method"]), "algorithm with the best mean rank (post none)")
    for _, r in fam.iterrows():
        tag = r["Conditions"].replace(" ", "_")
        K(f"synthetic_local_minus_global_{tag}", [float(r["HL local − global (pp)"]), float(r["CI low"]), float(r["CI high"])],
          f"Hodges-Lehmann local − global mean |error| with exact 95 % CI, {r['Conditions']} (n = {int(r['n'])} conditions)",
          "pp")
        K(f"synthetic_local_minus_global_{tag}_p", float(r["Wilcoxon p"]), f"exact Wilcoxon p, {r['Conditions']}")
    top = an.sort_values("eta2_partial", ascending=False).head(5)
    K("synthetic_anova_top_terms", {t: [round(float(e), 3), round(float(l), 3), round(float(h), 3)]
                                    for t, e, l, h in zip(top["term"], top["eta2_partial"], top["lo"], top["hi"])},
      "five largest partial eta-squared terms of the |error| model with bootstrap 95 % CI")
    mp = d0[d0["phantom"] == "multiscale"]
    K("synthetic_multiscale_unresolved_pp", float(100 * (mp["true_porosity_total"] - mp["true_porosity"]).median()),
      "median total minus pixel-grid porosity of the multiscale phantoms (sub-resolution porosity that no binary "
      "segmentation can recover)", "pp")


def _fig_examples(ctx) -> None:
    sc = ctx.cfg["synthetic"]
    nrm = ctx.cfg["normalization"]
    b, n, c = float(sorted(sc["blur_sigma"])[1]), float(sorted(sc["noise_sd"])[1]), float(max(sc["cupping"]))
    show = ["Otsu", "Yen", "Sauvola", "Niblack"]
    fig, axes = plt.subplots(3, 2 + len(show), figsize=style.figsize("double", 100))
    for i, k in enumerate(sc["phantoms"]):
        case = syn.generate_case(k, b, n, c, 0, ctx.seed, int(sc["size_px"]), p_low=float(nrm["p_low"]),
                                 p_high=float(nrm["p_high"]), eval_border=int(ctx.cfg["roi"]["eval_border_px"]))
        segs = syn.segment_all(case, ctx.cfg, show, ("none",))
        sl = slice(128, 384)
        panels = [("Truth", None, case["truth"]), ("Degraded", case["grey"], None)] + \
                 [(m, case["grey"], segs[(m, "none")]) for m in show]
        for j, (title, grey, pore) in enumerate(panels):
            ax = axes[i, j]
            if grey is None:
                ax.imshow(~pore[sl, sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(grey[sl, sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                if pore is not None:
                    ov = np.zeros(pore[sl, sl].shape + (4,))
                    rgb = np.array([int(style.PORE_OVERLAY[q:q + 2], 16) / 255 for q in (1, 3, 5)])
                    ov[pore[sl, sl]] = np.r_[rgb, 0.6]
                    ax.imshow(ov, interpolation="nearest")
                    met = syn.segmentation_metrics(pore, case["truth"], case["eval_mask"], case["phi_pixel"])
                    ax.text(0.03, 0.04, f"err {met['error_pp']:+.1f} pp\nDice {met['dice']:.2f}", transform=ax.transAxes,
                            fontsize=6, color="white", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(PHANTOM_LABELS[k], fontsize=7.5)
                ax.text(0.03, 0.96, f"φ = {100 * case['phi_pixel']:.1f} %", transform=ax.transAxes, fontsize=6,
                        va="top", color=style.INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))
    fig.tight_layout(w_pad=0.3, h_pad=0.4)
    ctx.save_fig(fig, "synthetic_examples",
                 f"Synthetic phantoms (256×256 px crops of 512×512 px images): pixel-grid truth (pores black), the "
                 f"degraded and normalised image (blur σ = {b:g} px, noise SD = {n:g} of the phase contrast, cupping "
                 f"c = {c:g}) and four segmentations (pores in orange) with their porosity error and Dice over the "
                 "whole evaluation disc.", kind="supp")
    plt.close(fig)


def _fig_benchmark(ctx, d0: pd.DataFrame) -> None:
    sc = ctx.cfg["synthetic"]
    blur, noise, cup = sorted(sc["blur_sigma"]), sorted(sc["noise_sd"]), sorted(sc["cupping"])
    mid_b, mid_n = blur[len(blur) // 2], noise[len(noise) // 2]
    phs = list(sc["phantoms"])
    fig, axes = plt.subplots(3, len(phs), figsize=style.figsize("double", 140), sharey="row")
    rowspec = [("blur", blur, dict(noise=mid_n, cupping=0.0), "PSF blur σ (px)"),
               ("noise", noise, dict(blur=mid_b, cupping=0.0), "Noise SD (fraction of contrast)"),
               ("cupping", cup, dict(blur=mid_b, noise=mid_n), "Cupping amplitude c")]
    for i, (fac, levels, fixed, xl) in enumerate(rowspec):
        for j, ph in enumerate(phs):
            ax = axes[i, j]
            sub = d0[(d0["phantom"] == ph)]
            for k, v in fixed.items():
                sub = sub[np.isclose(sub[k], v)]
            for m in ctx.methods:
                g = sub[sub["method"] == m].groupby(fac)["abs_error_pp"].median().reindex(levels)
                ax.plot(levels, g.values, marker=style.METHOD_MARKERS[m], color=style.METHOD_COLORS[m], ms=3, lw=0.9)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{v:g}" for v in levels])
            ax.set_xlabel(xl)
            style.hgrid(ax)
            if i == 0:
                ax.set_title(PHANTOM_LABELS[ph], fontsize=7.5)
            if j == 0:
                ax.set_ylabel("Median |error| (pp)")
            style.panel_label(ax, "abcdefghi"[i * len(phs) + j])
    fig.legend(handles=style.method_handles(), loc="lower center", ncol=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ctx.save_fig(fig, "synthetic_benchmark",
                 "Absolute porosity error (pp; median over the replicate phantoms; symmetric-log axis) of the eight "
                 "algorithms on synthetic images with known truth, as a function of PSF blur (a-c; noise "
                 f"{mid_n:g}, no cupping), noise (d-f; blur {mid_b:g} px, no cupping) and cupping (g-i; blur {mid_b:g} px, "
                 f"noise {mid_n:g}), for the three phantom families. Post-processing none; error against the pixel-grid "
                 "truth.", kind="main")
    plt.close(fig)


def _fig_dice(ctx, d0: pd.DataFrame) -> None:
    phs = list(ctx.cfg["synthetic"]["phantoms"])
    fig, axes = plt.subplots(1, len(phs), figsize=style.figsize("double", 60), sharey=True)
    for j, ph in enumerate(phs):
        ax = axes[j]
        sub = d0[d0["phantom"] == ph]
        data = [sub[sub["method"] == m]["dice"].to_numpy() for m in ctx.methods]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=style.INK, lw=0.8))
        for patch, m in zip(bp["boxes"], ctx.methods):
            patch.set_facecolor(style.METHOD_COLORS[m])
            patch.set_alpha(0.6)
            patch.set_edgecolor(style.INK)
            patch.set_linewidth(0.5)
        ax.set_xticks(range(1, len(ctx.methods) + 1), ctx.methods, rotation=60, ha="right")
        ax.set_title(PHANTOM_LABELS[ph], fontsize=7.5)
        style.hgrid(ax)
        style.panel_label(ax, "abc"[j])
    axes[0].set_ylabel("Dice with pixel truth")
    fig.tight_layout()
    ctx.save_fig(fig, "synthetic_dice",
                 "Dice overlap between each segmentation and the pixel-grid truth over all degradation conditions and "
                 "replicate phantoms (boxes: interquartile range; line: median; whiskers: 1.5 IQR; post-processing none).",
                 kind="supp")
    plt.close(fig)
