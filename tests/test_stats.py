"""Validation of poroseg.stats against independent implementations and known answers."""
import itertools
import warnings

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
import scipy.stats as sps

from poroseg import stats as S

warnings.filterwarnings("ignore")


def _long(n=9, k=5, seed=0, b=None):
    rng = np.random.default_rng(seed)
    subj = rng.normal(0, 3, n)
    rows = []
    levels_b = b or [None]
    for i in range(n):
        for j in range(k):
            for lb, vb in zip(levels_b, range(len(levels_b))):
                v = 20 + subj[i] + 1.5 * j + 0.7 * vb + rng.normal(0, 1 + 0.4 * j)
                r = dict(subject=f"s{i}", a=f"a{j}", y=v)
                if lb is not None:
                    r["b"] = lb
                rows.append(r)
    return pd.DataFrame(rows)


def test_friedman_matches_pingouin():
    df = _long()
    f = S.friedman(df, "y", "a", "subject")
    p = pg.friedman(data=df, dv="y", within="a", subject="subject")
    assert f["Q"] == pytest.approx(float(p["Q"].iloc[0]), rel=1e-10)
    assert f["W"] == pytest.approx(float(p["W"].iloc[0]), rel=1e-10)


def test_friedman_ties_and_bootstrap_ci():
    df = _long()
    df.loc[df.a == "a1", "y"] = df.loc[df.a == "a0", "y"].to_numpy()   # exact ties
    f = S.friedman(df, "y", "a", "subject", n_boot=200, rng=1)
    p = pg.friedman(data=df, dv="y", within="a", subject="subject")
    assert f["W"] == pytest.approx(float(p["W"].iloc[0]), rel=1e-10)
    assert f["W_lo"] <= f["W"] <= f["W_hi"]


def test_rm_anova_oneway_matches_pingouin():
    df = _long()
    a = S.rm_anova_oneway(df, "y", "a", "subject")
    p = pg.rm_anova(data=df, dv="y", within="a", subject="subject", correction=True, effsize="ng2").iloc[0]
    assert a["F"] == pytest.approx(p["F"], rel=1e-10)
    assert a["eps_gg"] == pytest.approx(p["eps"], rel=1e-8)
    assert a["eta2_generalized"] == pytest.approx(p["ng2"], rel=1e-8)


def test_rm_anova_twoway_main_effects_match_pingouin_and_interaction_epsilon():
    df = _long(b=["u", "v", "w"])
    a = S.rm_anova_twoway(df, "y", "subject", "a", "b").set_index("term")
    p = pg.rm_anova(data=df, dv="y", within=["a", "b"], subject="subject", effsize="ng2").set_index("Source")
    for t in ("a", "b"):
        assert a.loc[t, "F"] == pytest.approx(p.loc[t, "F"], rel=1e-8)
        assert a.loc[t, "eta2_generalized"] == pytest.approx(p.loc[t, "ng2"], rel=1e-8)
    assert a.loc["a:b", "F"] == pytest.approx(p.loc["a * b", "F"], rel=1e-8)
    # interaction epsilon in a k x 2 design equals the one-way epsilon of the differences
    d2 = df[df.b.isin(["u", "v"])]
    i2 = S.rm_anova_twoway(d2, "y", "subject", "a", "b").set_index("term")
    w = S.to_wide(d2, "y", ["a", "b"], "subject")
    diff = pd.DataFrame({m: w[(m, "v")] - w[(m, "u")] for m in w.columns.get_level_values(0).unique()})
    long = diff.reset_index().melt(id_vars="subject", var_name="a", value_name="d")
    o = S.rm_anova_oneway(long, "d", "a", "subject")
    assert i2.loc["a:b", "eps_gg"] == pytest.approx(o["eps_gg"], rel=1e-10)
    assert i2.loc["a:b", "F"] == pytest.approx(o["F"], rel=1e-10)


def test_variance_decomposition_adds_up_and_bootstraps():
    df = _long(b=["u", "v"])
    vd = S.variance_decomposition(df, "y", ["subject", "a", "b"], n_boot=100, rng=3)
    terms = vd[vd.order < 99]
    assert terms["pct"].sum() == pytest.approx(100.0, abs=1e-9)
    y = df["y"].to_numpy()
    assert vd.attrs["SS_total"] == pytest.approx(np.sum((y - y.mean()) ** 2), rel=1e-12)
    mat = vd.set_index("term").loc["material (total)"]
    assert mat["lo"] <= mat["pct"] <= mat["hi"]


def test_wilcoxon_exact_matches_scipy_without_ties():
    rng = np.random.default_rng(4)
    for _ in range(5):
        d = rng.normal(0.3, 1, 9)
        assert S.wilcoxon_signed_rank(d)["p"] == pytest.approx(sps.wilcoxon(d, method="exact").pvalue, rel=1e-10)
    assert S.wilcoxon_signed_rank(np.ones(9))["p"] == pytest.approx(2 / 2 ** 9)


def test_hodges_lehmann_ci_brute_force_inversion_and_coverage():
    d = np.arange(1, 10) - 3.0
    hl = S.hodges_lehmann(d)
    assert hl["estimate"] == pytest.approx(2.0)
    # the exact CI is the set of shifts not rejected by the exact signed-rank test (closed interval ends)
    grid = np.linspace(-4, 8, 2401)
    keep = [g for g in grid if S.wilcoxon_signed_rank(d - g)["p"] > 1 - hl["conf_achieved"] + 1e-9]
    assert min(keep) >= hl["lo"] - 0.01 and max(keep) <= hl["hi"] + 0.01
    rng = np.random.default_rng(5)
    cover = np.mean([(lambda h: h["lo"] <= 0 <= h["hi"])(S.hodges_lehmann(rng.standard_t(3, 9))) for _ in range(400)])
    assert cover > 0.93


def test_bland_altman_and_ccc_hand_values():
    x = np.array([1., 2, 3, 4, 5, 6, 7])
    e = np.array([.1, -.2, .3, .05, -.1, .2, 0])
    y = x + e
    ba = S.bland_altman(x, y)
    assert ba["bias"] == pytest.approx(e.mean())
    assert ba["sd"] == pytest.approx(e.std(ddof=1))
    lo, hi = ba["loa_hi_ci_exact"]
    assert lo < ba["loa_hi"] < hi
    cc = S.lin_ccc(x, y)
    num = 2 * np.mean((x - x.mean()) * (y - y.mean()))
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    assert cc["ccc"] == pytest.approx(num / den)
    assert cc["lo"] < cc["ccc"] < cc["hi"]


def test_deming_orthogonal_and_passing_bablok():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 10, 30)
    y = 1.5 * x + 2 + rng.normal(0, 0.3, 30)
    dm = S.deming(x, y, 1.0)
    # closed-form orthogonal (total least squares) slope
    C = np.cov(x, y)
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    assert dm["slope"] == pytest.approx(v[1] / v[0], rel=1e-10)
    pb = S.passing_bablok(np.arange(1., 11), 2 * np.arange(1., 11) + 1)
    assert pb["slope"] == pytest.approx(2.0) and pb["intercept"] == pytest.approx(1.0)
    pb_swap = S.passing_bablok(2 * np.arange(1., 11) + 1, np.arange(1., 11))
    assert pb_swap["slope"] == pytest.approx(0.5)


def test_pairwise_and_prob_best():
    df = _long(k=4)
    pw = S.pairwise_paired(df, "y", "a", "subject")
    assert len(pw) == 6 and (pw["p_adj"] >= pw["p_wilcoxon"] - 1e-12).all()
    err = pd.DataFrame({"good": np.r_[0.1, -0.1, 0.2, 0.0, 0.1], "bad": np.r_[2.0, -3.0, 2.5, 1.0, 2.0]})
    pbst = S.prob_best(err, rng=7, n_boot=500)
    assert pbst["good"] > 0.99 and pbst.sum() == pytest.approx(1.0)


def test_error_metrics_point_values():
    x = np.array([10., 20, 30])
    y = np.array([11., 18, 33])
    m = S.error_metrics(x, y, n_boot=200, rng=8)
    assert m["bias"] == pytest.approx(2 / 3)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
