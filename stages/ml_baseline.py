"""Stage 'ml_baseline': a trainable pixel-classifier baseline (Weka / ilastik type).

Reviewers of segmentation studies ask how classical thresholds compare with
machine learning. This stage adds three transparent, CPU-cheap baselines
evaluated under exactly the same protocol as the eight thresholds (disc ROI,
evaluation mask, no post-processing). They are baselines, not new methods.

Features: ``skimage.feature.multiscale_basic_features`` (Gaussian-smoothed
intensity, gradient magnitude ('edges') and Hessian eigenvalues ('texture') at
sigma = 1, 2, 4, 8, 16 px: 20 features), the feature set of Trainable Weka
Segmentation (Arganda-Carreras et al. 2017) and ilastik (Berg et al. 2019),
computed on the normalised image (``ctx.norm``) or on identically normalised
synthetic images.

1. 'RF-synthetic' - random forest (Breiman 2001) trained with the GROUND TRUTH
   of synthetic phantoms (``poroseg.synthetic``; all phantom kinds, blur, noise
   and cupping levels of ``config.yaml``; replicates disjoint from those of the
   'synthetic' benchmark), validated on held-out phantoms and then applied to
   the real images. Its weakness is the domain shift between phantoms and real
   rocks (texture, artefacts, inclusions).
2. 'RF-consensus' - one forest per image trained on sparse labels taken where
   all eight classical thresholds agree (the automatic analogue of an analyst's
   scribbles in ilastik); it inherits the classical bias where they agree and
   arbitrates where they disagree.
3. 'GMM-2' - unsupervised two-component Gaussian mixture on the standardised
   features (pore = component with the lower mean smoothed intensity).
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from .. import stats as S
from .. import style, synthetic as syn

ML_METHODS = ["RF-synthetic", "RF-consensus", "GMM-2"]
ML_STYLE = {"RF-synthetic": dict(marker="P", color="#3d3d3d"), "RF-consensus": dict(marker="X", color="#7a7a7a"),
            "GMM-2": dict(marker="v", color="#a8a8a8")}
TRAIN_REPS = (100, 101)          # disjoint from the benchmark replicates 0..n_rep-1
VALID_REP = 102
SIGMA = dict(sigma_min=1, sigma_max=16)


def features(img: np.ndarray) -> np.ndarray:
    """(H, W, 20) float32 multiscale features of a normalised image."""
    f = multiscale_basic_features(img.astype(np.float32), intensity=True, edges=True, texture=True, **SIGMA)
    return f.astype(np.float32, copy=False)


def _balanced_sample(labels: np.ndarray, valid: np.ndarray, n_per_class: int, rng) -> np.ndarray:
    idx = []
    for c in (False, True):
        cand = np.flatnonzero(valid & (labels == c))
        if cand.size:
            idx.append(rng.choice(cand, min(n_per_class, cand.size), replace=False))
    return np.concatenate(idx) if idx else np.array([], int)


def _predict(clf, F: np.ndarray, sel: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    flat = F.reshape(-1, F.shape[-1])
    out = np.zeros(flat.shape[0], bool)
    ids = np.flatnonzero(sel.ravel())
    for i in range(0, ids.size, chunk):
        j = ids[i:i + chunk]
        out[j] = clf.predict(flat[j]).astype(bool)
    return out.reshape(sel.shape)


def _train_synthetic(ctx):
    sc = ctx.cfg["synthetic"]
    nrm = ctx.cfg["normalization"]
    size = int(sc["size_px"])
    eb = int(ctx.cfg["roi"]["eval_border_px"])
    rng = ctx.rng("ml", "synthetic_train")
    X, y = [], []
    n_pc = 400 if ctx.quick else 1200
    conds = [(k, b, n, c) for k in sc["phantoms"] for b in sc["blur_sigma"] for n in sc["noise_sd"] for c in sc["cupping"]]
    reps = TRAIN_REPS[:1] if ctx.quick else TRAIN_REPS
    for (k, b, n, c) in conds:
        for rep in reps:
            case = syn.generate_case(k, b, n, c, rep, ctx.seed, size, p_low=float(nrm["p_low"]),
                                     p_high=float(nrm["p_high"]), eval_border=eb)
            F = features(case["grey"])
            pick = _balanced_sample(case["truth"].ravel(), case["eval_mask"].ravel(), n_pc, rng)
            X.append(F.reshape(-1, F.shape[-1])[pick])
            y.append(case["truth"].ravel()[pick])
    X, y = np.concatenate(X), np.concatenate(y)
    clf = RandomForestClassifier(n_estimators=60 if ctx.quick else 100, max_depth=20, min_samples_leaf=5, n_jobs=2,
                                 random_state=int(rng.integers(2 ** 31 - 1)))
    clf.fit(X, y)
    # validation on held-out phantoms
    rows = []
    for (k, b, n, c) in conds:
        case = syn.generate_case(k, b, n, c, VALID_REP, ctx.seed, size, p_low=float(nrm["p_low"]),
                                 p_high=float(nrm["p_high"]), eval_border=eb)
        F = features(case["grey"])
        pred = _predict(clf, F, case["mask"])
        met = syn.segmentation_metrics(pred, case["truth"], case["eval_mask"], case["phi_pixel"], case["phi_total"])
        T = syn.segment_all(case, ctx.cfg, ["Otsu"], ("none",))
        mo = syn.segmentation_metrics(T[("Otsu", "none")], case["truth"], case["eval_mask"], case["phi_pixel"])
        rows.append(dict(phantom=k, blur=b, noise=n, cupping=c, rep=VALID_REP, n_train=len(y), **met,
                         otsu_abs_error_pp=mo["abs_error_pp"], otsu_dice=mo["dice"]))
    return clf, pd.DataFrame(rows), len(y)


def _consensus(ctx, sid, F, E, rng):
    pores = np.stack([ctx.pore(sid, m, "none") for m in ctx.methods])
    allp, alls = pores.all(axis=0) & E, (~pores).all(axis=0) & E
    labels = np.zeros(E.shape, bool)
    labels[allp] = True
    valid = (allp | alls).ravel()
    n_pc = 1500 if ctx.quick else 4000
    pick = _balanced_sample(labels.ravel(), valid, n_pc, rng)
    yl = labels.ravel()[pick]
    note = f"{int(allp.sum())} consensus pore px, {int(alls.sum())} consensus solid px"
    if yl.min() == yl.max():
        return None, note + " (one class only: model not trainable)"
    clf = RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_leaf=5, n_jobs=2,
                                 random_state=int(rng.integers(2 ** 31 - 1)))
    clf.fit(F.reshape(-1, F.shape[-1])[pick], yl)
    return clf, note


def _gmm(F, M, E, rng):
    flat = F.reshape(-1, F.shape[-1])
    ids = np.flatnonzero(M.ravel())
    sub = rng.choice(ids, min(60000, ids.size), replace=False)
    mu, sd = flat[sub].mean(0), flat[sub].std(0) + 1e-6
    g = GaussianMixture(2, covariance_type="full", random_state=int(rng.integers(2 ** 31 - 1)), max_iter=300)
    g.fit((flat[sub] - mu) / sd)
    # pore = component with the lower mean of the sigma = 1 smoothed intensity (feature 0)
    pore_comp = int(np.argmin(g.means_[:, 0]))
    out = np.zeros(M.size, bool)
    for i in range(0, ids.size, 400_000):
        j = ids[i:i + 400_000]
        out[j] = g.predict((flat[j] - mu) / sd) == pore_comp
    return out.reshape(M.shape)


def run(ctx) -> None:
    t0 = time.time()
    clf_syn, val, n_train = _train_synthetic(ctx)
    ctx.save_data(val, "ml_synthetic_validation")
    ctx.log("RF-synthetic trained on %d pixels (%.0fs)", n_train, time.time() - t0)
    rows, notes, examples = [], {}, {}
    for sid in ctx.sample_ids:
        R = ctx.roi(sid)
        img = ctx.norm(sid).img
        F = features(img)
        E, M = R.eval_mask, R.mask
        rng = ctx.rng("ml", "real", sid)
        preds = {"RF-synthetic": _predict(clf_syn, F, M)}
        clf_c, note = _consensus(ctx, sid, F, E, rng)
        notes[ctx.label(sid)] = note
        preds["RF-consensus"] = _predict(clf_c, F, M) if clf_c is not None else np.zeros_like(M)
        preds["GMM-2"] = _gmm(F, M, E, rng)
        for m, p in preds.items():
            rows.append(dict(sample_id=sid, label=ctx.label(sid), group=ctx.group(sid), roi=ctx.primary_roi, method=m,
                             post="none", porosity=float((p & E).sum() / E.sum()), n_eval_px=int(E.sum()),
                             note=note if m == "RF-consensus" else ""))
        if ctx.label(sid) in ("CF5", "VN04", "VN02"):
            examples[sid] = {m: preds[m] for m in ("RF-synthetic", "RF-consensus")}
        del F
        ctx.log("%s done (%.0fs)", sid, time.time() - t0)
    ml = pd.DataFrame(rows)
    ctx.save_data(ml, "ml_porosity")
    _report(ctx, ml, val, notes, examples)


def _report(ctx, ml, val, notes, examples) -> None:
    K = ctx.add_key
    seg = ctx.seg_primary()
    base = seg[seg["post"] == "none"]
    ref = ctx.lab_ref.set_index("sample_id")
    rows = []
    for sid in ctx.sample_ids:
        b = base[base["sample_id"] == sid]
        r = {"Image": ctx.label(sid), "Otsu (%)": 100 * float(b[b["method"] == "Otsu"]["porosity"].iloc[0]),
             "8 classical, range (%)": f"{100 * b['porosity'].min():.1f}–{100 * b['porosity'].max():.1f}"}
        for m in ML_METHODS:
            r[f"{m} (%)"] = 100 * float(ml[(ml["sample_id"] == sid) & (ml["method"] == m)]["porosity"].iloc[0])
        r["Laboratory (%)"] = 100 * ref.at[sid, "ref"] if sid in ref.index else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    with_ref = [s for s in ctx.sample_ids if s in ref.index]
    x = 100 * np.array([ref.at[s, "ref"] for s in with_ref])
    acc = []
    for m in ML_METHODS + ["Otsu"]:
        if m == "Otsu":
            y = 100 * np.array([float(base[(base["sample_id"] == s) & (base["method"] == "Otsu")]["porosity"].iloc[0]) for s in with_ref])
        else:
            y = 100 * np.array([float(ml[(ml["sample_id"] == s) & (ml["method"] == m)]["porosity"].iloc[0]) for s in with_ref])
        em = S.error_metrics(x, y, n_boot=ctx.n_boot, rng=ctx.rng("ml", "err", m))
        vv = val if m == "RF-synthetic" else None
        acc.append({"Method": m, "Bias vs lab (pp)": em["bias"], "Bias CI": f"[{em['bias_lo']:.1f}, {em['bias_hi']:.1f}]",
                    "MAE vs lab (pp)": em["mae"], "MAE CI": f"[{em['mae_lo']:.1f}, {em['mae_hi']:.1f}]",
                    "RMSE (pp)": em["rmse"],
                    "Synthetic |error| (pp)": float(vv["abs_error_pp"].median()) if vv is not None else np.nan,
                    "Synthetic Dice": float(vv["dice"].median()) if vv is not None else np.nan})
        K(f"ml_baseline_lab_{m.lower().replace('-', '_')}", dict(bias=em["bias"], bias_lo=em["bias_lo"], bias_hi=em["bias_hi"],
                                                                 mae=em["mae"], mae_lo=em["mae_lo"], mae_hi=em["mae_hi"]),
          f"{m}: bias and MAE vs laboratory over the {len(with_ref)} images with a reference (pp, bootstrap CIs)")
    at = pd.DataFrame(acc)
    ctx.save_table(t, "ml_baseline",
                   "Porosity (%) from three machine-learning baselines (random forest trained on synthetic ground truth, "
                   "random forest trained on consensus labels of the eight thresholds, unsupervised two-component "
                   "Gaussian mixture), next to Otsu, the range of the eight classical thresholds and the laboratory "
                   "reference (disc ROI, no post-processing).", kind="supp", float_format="{:.1f}")
    ctx.save_table(at, "ml_baseline_accuracy",
                   f"Accuracy of the machine-learning baselines against the laboratory reference ({len(with_ref)} images; "
                   "bias and MAE with bootstrap 95 % CIs over images) and, for RF-synthetic, on held-out synthetic "
                   "phantoms (median absolute porosity error and Dice over all degradation conditions).", kind="supp",
                   float_format="{:.2f}")
    K("ml_baseline_rf_synthetic_validation", dict(median_abs_error_pp=float(val["abs_error_pp"].median()),
                                                  median_dice=float(val["dice"].median()),
                                                  otsu_median_abs_error_pp=float(val["otsu_abs_error_pp"].median()),
                                                  otsu_median_dice=float(val["otsu_dice"].median())),
      "RF-synthetic on held-out synthetic phantoms (all degradation conditions) compared with Otsu on the same images")
    K("ml_baseline_consensus_notes", notes, "consensus labels available per image for RF-consensus")
    for m in ML_METHODS:
        v = ml[ml["method"] == m].set_index("label")["porosity"]
        K(f"ml_baseline_porosity_{m.lower().replace('-', '_')}", {k: round(100 * float(val_), 1) for k, val_ in v.items()},
          f"{m} porosity per image (disc ROI)", "%")
    # inside or outside the classical range
    inside = []
    for sid in ctx.sample_ids:
        b = base[base["sample_id"] == sid]["porosity"]
        for m in ML_METHODS:
            p = float(ml[(ml["sample_id"] == sid) & (ml["method"] == m)]["porosity"].iloc[0])
            inside.append(b.min() <= p <= b.max())
    K("ml_baseline_frac_inside_classical_range", float(np.mean(inside)),
      "fraction of image x ML-baseline cases whose porosity falls inside the range of the eight classical thresholds")
    _figure(ctx, ml, base, ref, examples)


def _figure(ctx, ml, base, ref, examples) -> None:
    fig = plt.figure(figsize=style.figsize("double", 110))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(ctx.sample_ids))
    for i, sid in enumerate(ctx.sample_ids):
        b = 100 * base[base["sample_id"] == sid]["porosity"]
        ax.plot([i, i], [b.min(), b.max()], color=style.REF_BAND, lw=6, solid_capstyle="butt", zorder=1)
        ax.scatter(i - 0.18, 100 * float(base[(base["sample_id"] == sid) & (base["method"] == "Otsu")]["porosity"].iloc[0]),
                   color=style.METHOD_COLORS["Otsu"], marker="o", s=14, zorder=3)
        for k, m in enumerate(ML_METHODS):
            p = 100 * float(ml[(ml["sample_id"] == sid) & (ml["method"] == m)]["porosity"].iloc[0])
            ax.scatter(i + 0.06 + 0.12 * k, p, s=18, zorder=4, **ML_STYLE[m])
        if sid in ref.index:
            ax.scatter(i - 0.32, 100 * ref.at[sid, "ref"], marker="D", s=14, color=style.INK, zorder=4)
    ax.set_xticks(x, [ctx.label(s) for s in ctx.sample_ids])
    ax.set_ylabel("Porosity (%)")
    style.hgrid(ax)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=style.REF_BAND, lw=6, label="Range of 8 thresholds"),
          Line2D([], [], ls="none", marker="o", color=style.METHOD_COLORS["Otsu"], label="Otsu"),
          Line2D([], [], ls="none", marker="D", color=style.INK, label="Laboratory")] + \
         [Line2D([], [], ls="none", label=m, **ML_STYLE[m]) for m in ML_METHODS]
    ax.legend(handles=hs, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    style.panel_label(ax, "a")
    sid = next((s for s in ctx.sample_ids if ctx.label(s) == "VN04"), ctx.sample_ids[0])
    img = ctx.norm(sid).img
    h, w = img.shape
    c0, r0, s0 = h // 2 - 120, w // 2 - 120, 240
    sl = (slice(c0, c0 + s0), slice(r0, r0 + s0))
    panels = [("Otsu", ctx.pore(sid, "Otsu", "none")), ("RF-synthetic", examples.get(sid, {}).get("RF-synthetic")),
              ("RF-consensus", examples.get(sid, {}).get("RF-consensus"))]
    rgb = np.array([int(style.PORE_OVERLAY[q:q + 2], 16) / 255 for q in (1, 3, 5)])
    for k, (title, pore) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        a.imshow(img[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if pore is not None:
            ov = np.zeros((s0, s0, 4))
            ov[pore[sl]] = np.r_[rgb, 0.6]
            a.imshow(ov, interpolation="nearest")
        a.set_title(f"{ctx.label(sid)} · {title}", fontsize=7.5)
        a.set_xticks([])
        a.set_yticks([])
        if k == 0:
            style.scalebar(a, ctx.pixel_size_um(sid))
        style.panel_label(a, "bcd"[k])
    fig.tight_layout()
    ctx.save_fig(fig, "ml_baseline",
                 "Machine-learning baselines. (a) Porosity of every image from a random forest trained on synthetic "
                 "ground truth (RF-synthetic), a random forest trained per image on consensus labels of the eight "
                 "thresholds (RF-consensus) and an unsupervised two-component Gaussian mixture (GMM-2), against the "
                 "range of the eight classical thresholds (grey bar), Otsu and the laboratory reference (disc ROI, no "
                 "post-processing). (b-d) The same 6 × 6 mm window of VN04 segmented by Otsu, RF-synthetic and "
                 "RF-consensus (pores in orange).", kind="supp")
    plt.close(fig)
