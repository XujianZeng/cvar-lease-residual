#!/usr/bin/env python3
"""Pooled and stress analyses using the rules fitted in `buffer_study.py`.

1. Evidence tiers: development (GM), replication (other issuers' holdout 1) and
   confirmatory holdouts (other issuers' holdout 2, collected after every rule,
   including the base-anchored variant, was fixed). Pooled equal-weighted CVaR
   changes with stratified bootstrap CIs, and Holm-adjusted bootstrap p-values.
2. Pool-level stress: a common multiplicative shock S to disposal proceeds. In a
   large pool the idiosyncratic part diversifies, so the pool loss rate is a
   decreasing function of S and its alpha-quantile is the loss at the
   alpha-quantile of S (one-factor, asymptotic-portfolio argument).
3. Gradient-boosting benchmark: the same bucket LP driven by a gradient-boosted
   return / severity score instead of the logistic / linear score.

Writes `revision_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from . import buffer_study as bs
from .analyze import cvar

HERE = Path(__file__).parent.parent
ALPHA = 0.99
BOOT = 1000
SHOCKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
STRESS = 0.8
RNG = np.random.default_rng(20260925)
TIER = {("gm", "validation"): "development", ("gm", "later"): "development",
        **{(i, "validation"): "replication" for i in ("nissan", "vw", "ford")},
        **{(i, "later"): "confirmatory" for i in ("nissan", "vw", "ford")}}
CA, BA = "Return-aware buckets (K=4)", bs.BASE_ANCHORED
GBM_CA, GBM_BA = "GBM buckets on contract residual (K=4)", "GBM buckets on base residual (K=4)"
# (rule, benchmark) pairs compared at equal mean valued residual
COMPARISONS = [(CA, "uniform"), (CA, "scaled_base"), (BA, "uniform"), (BA, "scaled_base"),
               ("Brand-specific ratios", "uniform"), ("Brand-specific ratios", "scaled_base"),
               (GBM_CA, "uniform"), (GBM_BA, "scaled_base")]


class GBMScore:
    """Gradient-boosted analogue of `bs.RiskScore`: return probability times the
    expected shortfall of proceeds/contract below c, using cross-fitted
    training residuals for the severity distribution."""

    def __init__(self, train, c, seed=0):
        X, r = train["X"], train["r"]
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.2, random_state=seed)
        self.ret = HistGradientBoostingClassifier(**kw).fit(X, r)
        Xr, y = X[r], train["p"][r] / train["q"][r]
        self.sev = HistGradientBoostingRegressor(**kw).fit(Xr, y)
        resid = np.empty(len(y))
        for fit_idx, out_idx in KFold(5, shuffle=True, random_state=seed).split(Xr):
            m = HistGradientBoostingRegressor(**kw).fit(Xr[fit_idx], y[fit_idx])
            resid[out_idx] = y[out_idx] - m.predict(Xr[out_idx])
        self.e = np.sort(resid)
        self.csum = np.cumsum(self.e)
        self.c = c

    def prob(self, d):
        return self.ret.predict_proba(d["X"])[:, 1]

    def __call__(self, d, severity=True):
        pr = self.prob(d)
        if not severity:
            return pr
        t = self.c - self.sev.predict(d["X"])
        k = np.searchsorted(self.e, t)
        below = np.where(k > 0, self.csum[np.maximum(k - 1, 0)], 0.0)
        return pr * (k * t - below) / len(self.e)


def scaled_base_equal(b, q, target):
    """Exact t with mean(min(b t, q)) == target (piecewise linear in t)."""
    k = q / b
    o = np.argsort(k)
    ks, qs, bs_ = k[o], np.cumsum(q[o]), np.cumsum(b[o])
    n, btot = len(q), b.sum()
    at_break = (qs + ks * (btot - bs_)) / n  # mean value at t = ks[j]
    j = int(np.searchsorted(at_break, target))
    if j >= n:
        return q.copy()
    q_below, b_above = (qs[j - 1], btot - bs_[j - 1]) if j else (0.0, btot)
    t = (target * n - q_below) / b_above
    return np.minimum(b * t, q)


def benchmark(kind, v, d):
    if kind == "uniform":
        return d["q"] * v.mean() / d["q"].mean()
    return scaled_base_equal(d["b"], d["q"], v.mean())


def pct_change(v, kind, d):
    ref = benchmark(kind, v, d)
    return 100 * (cvar(bs.shortfall(v, d), ALPHA) / cvar(bs.shortfall(ref, d), ALPHA) - 1)


def pool_loss_rate(v, d, s):
    """Pool residual loss as % of valued residual when all proceeds are scaled by s."""
    return 100 * np.where(d["r"], np.maximum(v - s * d["p"], 0), 0).sum() / v.sum()


def equal_stress_release(v, kind, d, s):
    """Extra mean valued residual (USD/lease) of v over the benchmark family
    member with the same pool loss (USD) under shock s."""
    loss = np.where(d["r"], np.maximum(v - s * d["p"], 0), 0).sum()

    def bench_loss(t):
        w = d["q"] * t if kind == "uniform" else bs.ScaledBase.values(d, t)
        return np.where(d["r"], np.maximum(w - s * d["p"], 0), 0).sum()
    t = bs.bisect(bench_loss, 0, 1 if kind == "uniform" else 1.5, loss)
    w = d["q"] * t if kind == "uniform" else bs.ScaledBase.values(d, t)
    return float(v.mean() - w.mean())


def fit_issuer(key):
    cohorts = bs.ISSUERS[key]
    data = {k: bs.load(*v) for k, v in cohorts.items()}
    train = data["train"]
    brands = bs.brand_groups(train)
    for d in data.values():
        bs.add_features(d, brands)
    target = cvar(bs.shortfall(train["b"], train), ALPHA)
    c = bs.Uniform().fit(train, target, ALPHA).t
    score = bs.RiskScore(train, range(train["X"].shape[1]), c)
    gbm = GBMScore(train, c)
    rules = {CA: bs.RiskBucket(score, 4), BA: bs.RiskBucket(score, 4, anchor="b"),
             "Brand-specific ratios": bs.Brand(len(brands)),
             GBM_CA: bs.RiskBucket(gbm, 4), GBM_BA: bs.RiskBucket(gbm, 4, anchor="b")}
    for r in rules.values():
        r.fit(train, target, ALPHA)
    auc = {k: {"logit": bs.auc(score.prob(d), d["r"]), "gbm": bs.auc(gbm.prob(d), d["r"])}
           for k, d in data.items()}
    return data, rules, auc


def boot_samples(values, d):
    """Bootstrap draws of the % CVaR change for every comparison (common resamples)."""
    out = {f"{r}|{k}": np.empty(BOOT) for r, k in COMPARISONS}
    for b in range(BOOT):
        idx = RNG.integers(0, d["n"], d["n"])
        s = {k: d[k][idx] for k in ("q", "b", "p", "r")}
        for r, k in COMPARISONS:
            out[f"{r}|{k}"][b] = pct_change(values[r][idx], k, s)
    return out


def summarize(point, draws):
    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    if not len(valid):
        return {"pct": float(point), "ci95": [float("nan"), float("nan")],
                "p_boot": 1.0, "undefined_draws": len(draws)}
    # Finite-Monte-Carlo correction: do not report a literal zero probability.
    p = 2 * (1 + min(np.sum(valid >= 0), np.sum(valid <= 0))) / (len(valid) + 1)
    return {"pct": float(point), "ci95": np.percentile(valid, [2.5, 97.5]).tolist(),
            "p_boot": float(min(p, 1.0)), "undefined_draws": int(len(draws) - len(valid))}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


LABEL = {"gm": "GM", "nissan": "Nissan", "vw": "VW", "ford": "Ford"}
COHORT = {"validation": "Holdout 1", "later": "Holdout 2"}


def forest_rows(results):
    """Rows in the format of `summarize.collect()` for the forest plot."""
    names = {"Contract-anchored": CA, "Base-anchored": BA, "Brand ratios": "Brand-specific ratios"}
    rows = []
    for h in results["holdouts"]:
        for label, key in names.items():
            e = {"issuer": LABEL[h["issuer"]], "cohort": h["cohort"], "rule": label,
                 "auc": h["auc"]["logit"], "n": h["n"], "returned": h["returned"]}
            for bench in ("uniform", "scaled_base"):
                c = h["comparisons"][f"{key}|{bench}"]
                e[bench] = {"pct": c["pct"], "ci_pct": c["ci95"]}
            rows.append(e)
    return rows


def pool_figure(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 600})
    panels = [(f"{CA}|uniform", "Contract-anchored vs uniform ratio"),
              (f"{BA}|scaled_base", "Base-anchored vs scaled base residual")]
    colors = {"gm": "tab:blue", "nissan": "tab:red", "vw": "tab:green", "ford": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for h in results["holdouts"]:
            curve = h["pool"][key]["curve"]
            s = sorted(map(float, curve), reverse=True)
            gap = [curve[str(x)]["rule"] - curve[str(x)]["benchmark"] for x in s]
            ax.plot([100 * (1 - x) for x in s], gap, color=colors[h["issuer"]],
                    ls="-" if h["cohort"] == "validation" else "--", marker="o", ms=3,
                    label=f"{LABEL[h['issuer']]} {COHORT[h['cohort']]}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Common decline in disposal proceeds (%)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Pool loss rate: rule minus benchmark (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "pool_stress.png")
    fig.savefig(out / "pool_stress.pdf")
    plt.close(fig)


def main():
    holdouts, draws = [], {}
    for key in bs.ISSUERS:
        print(key, "...", flush=True)
        data, rules, auc = fit_issuer(key)
        for cohort in ("validation", "later"):
            d = data[cohort]
            values = {n: r.value(d) for n, r in rules.items()}
            smp = boot_samples(values, d)
            entry = {"issuer": key, "cohort": cohort, "tier": TIER[(key, cohort)], "n": d["n"],
                     "returned": int(d["r"].sum()), "auc": auc[cohort], "comparisons": {}, "pool": {}}
            for r, k in COMPARISONS:
                name = f"{r}|{k}"
                entry["comparisons"][name] = summarize(pct_change(values[r], k, d), smp[name])
                draws[(key, cohort, name)] = smp[name]
            # Pool level: proposed rules against equal-value benchmarks.
            for r, k in ((CA, "uniform"), (BA, "scaled_base"), (CA, "scaled_base")):
                v, ref = values[r], benchmark(k, values[r], d)
                curve = {str(s): {"rule": pool_loss_rate(v, d, s), "benchmark": pool_loss_rate(ref, d, s)}
                         for s in SHOCKS}
                diffs = np.empty(BOOT)
                for b in range(BOOT):
                    idx = RNG.integers(0, d["n"], d["n"])
                    s = {x: d[x][idx] for x in ("q", "b", "p", "r")}
                    vb = v[idx]
                    diffs[b] = pool_loss_rate(vb, s, STRESS) - pool_loss_rate(benchmark(k, vb, s), s, STRESS)
                entry["pool"][f"{r}|{k}"] = {
                    "curve": curve,
                    "ce_gap_pp_at_stress": curve[str(STRESS)]["rule"] - curve[str(STRESS)]["benchmark"],
                    "ce_gap_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                    "release_usd_per_lease_at_stress": equal_stress_release(v, k, d, STRESS),
                    "mean_valued_residual": float(v.mean())}
            holdouts.append(entry)
    # Multiple testing across the eight holdouts, per comparison.
    for r, k in COMPARISONS:
        name = f"{r}|{k}"
        adj = holm(np.array([h["comparisons"][name]["p_boot"] for h in holdouts]))
        for h, a in zip(holdouts, adj):
            h["comparisons"][name]["p_holm"] = float(a)
    # Conditional stratified bootstrap; this does not model common market shocks.
    pooled = {}
    tiers = {"all": [h for h in holdouts],
             "replication+confirmatory": [h for h in holdouts if h["tier"] != "development"],
             "confirmatory": [h for h in holdouts if h["tier"] == "confirmatory"]}
    for tname, hs in tiers.items():
        pooled[tname] = {}
        for r, k in COMPARISONS:
            name = f"{r}|{k}"
            point = np.mean([h["comparisons"][name]["pct"] for h in hs])
            mix = np.mean([draws[(h["issuer"], h["cohort"], name)] for h in hs], axis=0)
            pooled[tname][name] = summarize(point, mix)
    out = HERE / "revision_results.json"
    out.write_text(json.dumps({"alpha": ALPHA, "bootstrap": BOOT, "stress_shock": STRESS,
                               "holdouts": holdouts, "pooled": pooled}, indent=2))
    pool_figure(json.loads(out.read_text()), HERE / "figures")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
