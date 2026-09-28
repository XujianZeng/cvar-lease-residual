#!/usr/bin/env python3
"""Out-of-time extension across the 2019–2024 used-car cycle (GM Financial).

Fits every rule on one year's terminations and evaluates it unchanged on the
next year's (five tests, 2020–2024).
Rules and evaluation reuse `buffer_study.py` and `revision_study.py`.
Writes `cycle_results.json` and `figures/cycle.{png,pdf}`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import buffer_study as bs
from . import revision_study as rs
from .analyze import cvar

HERE = Path(__file__).parent.parent
ALPHA = 0.99
BOOT = 1000
MIN_LEASES, MIN_RETURNS = 1000, 100
RNG = np.random.default_rng(20260925)
COHORTS = {2019: "data/gm_2017_3_2019.jsonl", 2020: "data/gm_2018_3_2020.jsonl",
           2021: "data/gm_2019_3_2021.jsonl", 2022: "data/gm_2020_3_2022.jsonl",
           2023: "data/gm_2021_3_2023.jsonl", 2024: "data/gm_2022_3_2024.jsonl"}
TRUST = {2019: "GMALT 2017-3", 2020: "GMALT 2018-3", 2021: "GMALT 2019-3", 2022: "GMALT 2020-3",
         2023: "GMALT 2021-3", 2024: "GMALT 2022-3"}
COMPARISONS = [(rs.CA, "uniform"), (rs.BA, "scaled_base"), (rs.CA, "scaled_base"),
               ("Brand-specific ratios", "scaled_base")]


def pct_change(v, kind, d):
    """% CVaR change; undefined (NaN) when the equal-value benchmark has zero CVaR.
    Deviation from the plan, reported in the paper: in near-loss-free years a
    bootstrap resample can contain no benchmark shortfall at all."""
    ref = rs.benchmark(kind, v, d)
    base = cvar(bs.shortfall(ref, d), ALPHA)
    return float("nan") if base == 0 else 100 * (cvar(bs.shortfall(v, d), ALPHA) / base - 1)


def summarize(point, draws):
    ok = draws[~np.isnan(draws)]
    p = 2 * min(np.mean(ok >= 0), np.mean(ok <= 0))
    return {"pct": float(point), "ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))],
            "p_boot": float(min(p, 1.0)), "undefined_draws": int(np.isnan(draws).sum())}


def describe(d):
    r = d["r"]
    return {"n": int(d["n"]), "returned": int(r.sum()), "return_rate": float(r.mean()),
            "mean_proceeds_to_contract": float((d["p"][r] / d["q"][r]).mean()),
            "mean_contract_to_msrp": float((d["q"] / d["msrp"]).mean()),
            "mean_base_to_contract": float((d["b"] / d["q"]).mean())}


def fit(train):
    brands = bs.brand_groups(train)
    bs.add_features(train, brands)
    target = cvar(bs.shortfall(train["b"], train), ALPHA)
    c = bs.Uniform().fit(train, target, ALPHA).t
    score = bs.RiskScore(train, range(train["X"].shape[1]), c)
    rules = {rs.CA: bs.RiskBucket(score, 4), rs.BA: bs.RiskBucket(score, 4, anchor="b"),
             "Brand-specific ratios": bs.Brand(len(brands))}
    for r in rules.values():
        r.fit(train, target, ALPHA)
    return brands, score, rules, target


def evaluate(train_year, test_year, data):
    train = dict(data[train_year])
    test = dict(data[test_year])
    brands, score, rules, target = fit(train)
    bs.add_features(test, brands)
    values = {n: r.value(test) for n, r in rules.items()}
    draws = {f"{r}|{k}": np.empty(BOOT) for r, k in COMPARISONS}
    pool = {f"{r}|{k}": np.empty(BOOT) for r, k in ((rs.CA, "uniform"), (rs.BA, "scaled_base"))}
    for b in range(BOOT):
        idx = RNG.integers(0, test["n"], test["n"])
        s = {k: test[k][idx] for k in ("q", "b", "p", "r")}
        for r, k in COMPARISONS:
            draws[f"{r}|{k}"][b] = pct_change(values[r][idx], k, s)
        for key in pool:
            r, k = key.split("|")
            v = values[r][idx]
            pool[key][b] = rs.pool_loss_rate(v, s, 0.8) - rs.pool_loss_rate(rs.benchmark(k, v, s), s, 0.8)
    out = {"train_year": train_year, "test_year": test_year, "training_target_cvar": target,
           "auc_train": bs.auc(score.prob(train), train["r"]), "auc_test": bs.auc(score.prob(test), test["r"]),
           "bucket_ratios": {"contract": list(map(float, rules[rs.CA].ratios)),
                             "base": list(map(float, rules[rs.BA].ratios))},
           "comparisons": {}, "pool": {}}
    for r, k in COMPARISONS:
        name = f"{r}|{k}"
        out["comparisons"][name] = summarize(pct_change(values[r], k, test), draws[name])
    for key, x in pool.items():
        r, k = key.split("|")
        v = values[r]
        gap = rs.pool_loss_rate(v, test, 0.8) - rs.pool_loss_rate(rs.benchmark(k, v, test), test, 0.8)
        out["pool"][key] = {"gap_pp": float(gap), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                            "release_usd_per_lease": rs.equal_stress_release(v, k, test, 0.8)}
    return out, draws


def figure(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    tests = res["tests"]
    years = [t["test_year"] for t in tests]
    for off, (key, label, color, marker) in zip((-0.1, 0.1), (
            (f"{rs.CA}|uniform", "Contract-anchored vs uniform ratio", "tab:red", "D"),
            (f"{rs.BA}|scaled_base", "Base-anchored vs scaled base residual", "tab:purple", "v"))):
        pts = [t["comparisons"][key] for t in tests]
        x = [y + off for y in years]
        axes[0].errorbar(x, [p["pct"] for p in pts],
                         yerr=[[p["pct"] - p["ci95"][0] for p in pts], [p["ci95"][1] - p["pct"] for p in pts]],
                         fmt=marker, color=color, ms=5, capsize=2, label=label)
    axes[0].axhline(0, color="k", lw=0.8)
    ax2 = axes[0].twinx()
    ax2.plot(years, [100 * res["cohorts"][str(y)]["return_rate"] for y in years], color="0.5", ls=":", marker=".")
    ax2.set_ylabel("Return rate of test cohort (%, dotted)", color="0.4")
    axes[0].set_xticks(years)
    axes[0].set_xlabel("Test year (rule fitted on the previous year)")
    axes[0].set_ylabel("CVaR$_{99}$ change at equal valued residual (%)")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    axes[0].grid(alpha=.3)
    s = res["spearman_all_holdouts"]["points"]
    for p in s:
        axes[1].plot(p["auc"], p["pct"], "o" if p["source"] == "main" else "s",
                     color="tab:blue" if p["source"] == "main" else "tab:green", ms=5)
    axes[1].plot([], [], "o", color="tab:blue", label="Main study holdouts (8)")
    axes[1].plot([], [], "s", color="tab:green", label="Cycle extension, GM 2020–2024 (5)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("Holdout AUC of return model")
    axes[1].set_ylabel("Base-anchored vs scaled base (%)")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "cycle.png")
    fig.savefig(out / "cycle.pdf")
    plt.close(fig)


def main():
    data = {y: bs.load(path, f"{y}-01", f"{y}-09") for y, path in COHORTS.items()}
    cohorts = {str(y): {"trust": TRUST[y], **describe(d)} for y, d in data.items()}
    usable = {y for y, d in data.items() if d["n"] >= MIN_LEASES and d["r"].sum() >= MIN_RETURNS}
    tests, dropped, all_draws = [], [], {}
    for y in range(2019, 2024):
        if y not in usable or y + 1 not in usable:
            dropped.append({"train_year": y, "test_year": y + 1, "reason": "cohort below minimum size"})
            continue
        print(f"train {y} -> test {y + 1}", flush=True)
        out, draws = evaluate(y, y + 1, data)
        tests.append(out)
        all_draws[y + 1] = draws
    for r, k in COMPARISONS:
        name = f"{r}|{k}"
        adj = rs.holm(np.array([t["comparisons"][name]["p_boot"] for t in tests]))
        for t, a in zip(tests, adj):
            t["comparisons"][name]["p_holm"] = float(a)
    pooled = {}
    for r, k in COMPARISONS:
        name = f"{r}|{k}"
        point = np.mean([t["comparisons"][name]["pct"] for t in tests])
        mix = np.mean([all_draws[t["test_year"]][name] for t in tests], axis=0)  # NaN if any year undefined
        pooled[name] = summarize(point, mix)
    main_rev = json.loads((HERE / "revision_results.json").read_text())
    pts = [{"source": "main", "label": f"{h['issuer']} {h['cohort']}", "auc": h["auc"]["logit"],
            "pct": h["comparisons"][f"{rs.BA}|scaled_base"]["pct"]} for h in main_rev["holdouts"]]
    pts += [{"source": "cycle", "label": f"GM {t['test_year']}", "auc": t["auc_test"],
             "pct": t["comparisons"][f"{rs.BA}|scaled_base"]["pct"]} for t in tests]
    rho, p = spearmanr([x["auc"] for x in pts], [x["pct"] for x in pts])
    res = {"plan": "analysis_plan_extension.md", "alpha": ALPHA, "bootstrap": BOOT, "cohorts": cohorts,
           "tests": tests, "dropped": dropped, "pooled": pooled,
           "spearman_all_holdouts": {"rho": float(rho), "p": float(p), "n": len(pts), "points": pts}}
    out = HERE / "cycle_results.json"
    out.write_text(json.dumps(res, indent=2))
    figure(res, HERE / "figures")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
