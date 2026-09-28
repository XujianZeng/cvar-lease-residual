#!/usr/bin/env python3
"""Return-aware residual risk buffers for securitized auto leases.

All rules are fitted on the training cohort only and evaluated unchanged on the
validation and later cohorts. Outputs `study_results.json` and figures in
`figures/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

from .analyze import cvar

HERE = Path(__file__).resolve().parents[1]
# Termination windows are inclusive (YYYY-MM, YYYY-MM). Proceeds are summed over
# the termination month and the two following monthly reports, so each window
# ends at least two months before the last collected report.
ISSUERS = {
    "gm": {"train": ("data/gm_2022_3_2024.jsonl", "2024-01", "2024-09"),
           "validation": ("data/gm_2023_1_2025.jsonl", "2025-01", "2025-04"),
           "later": ("data/gm_2023_3_2025_26.jsonl", "2025-07", "2025-10")},
    "nissan": {"train": ("data/nissan_2022a_2024.jsonl", "2023-12", "2024-06"),
               "validation": ("data/nissan_2023b_2025.jsonl", "2025-01", "2025-08"),
               "later": ("data/nissan_2024a_2025_26.jsonl", "2025-10", "2026-04")},
    "vw": {"train": ("data/vw_2022a_2024.jsonl", "2023-12", "2024-06"),
           "validation": ("data/vw_2023a_2025.jsonl", "2025-01", "2025-10"),
           "later": ("data/vw_2024a_2025_26.jsonl", "2025-11", "2026-06")},
    "ford": {"train": ("data/ford_2022a_2024.jsonl", "2023-12", "2024-09"),
             "validation": ("data/ford_2023b_2025.jsonl", "2025-01", "2025-09"),
             "later": ("data/ford_2024a_2025_26.jsonl", "2025-10", "2026-05")},
}
# Collected but excluded: Mercedes-Benz trusts never report code 2 (return), so
# returns cannot be separated from payoffs; BMW trusts do not report proceeds.
BASE_FEATURES = ("intercept", "contract_residual/msrp", "base_residual/contract_residual",
                 "net_cap_cost/msrp", "term/36", "type_suv", "type_truck")
MIN_BRAND_SHARE = 0.01
BASE_ANCHORED = "Return-aware buckets on base residual (K=4)"
COHORT_LABEL = {"train": "Training", "validation": "Holdout 1", "later": "Holdout 2"}
RNG = np.random.default_rng(20260924)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def month_index(text, sep):
    parts = (text or "").split(sep)
    try:
        if sep == "/" and len(parts) == 2:
            return int(parts[1]) * 12 + int(parts[0]) - 1
        if sep == "-" and len(parts) == 3:
            return int(parts[2]) * 12 + int(parts[0]) - 1
    except ValueError:
        pass
    return None


def load(path, start, end):
    """Nondefault terminations (payoff=1, return=2; SEC ABS-EE code list) with
    effective date in [start, end], one record per lease."""
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    lo, hi = y0 * 12 + m0 - 1, y1 * 12 + m1 - 1
    by_asset = {}
    with open(HERE / path) as handle:
        for line in handle:
            row = json.loads(line)
            by_asset.setdefault(row["assetNumber"], []).append(row)
    counts, kept = {}, []

    def count(key):
        counts[key] = counts.get(key, 0) + 1

    for rows in by_asset.values():
        eligible = [r for r in rows
                    if r.get("zeroBalanceCode") == "1" and r.get("terminationIndicator") in ("1", "2")
                    and (e := month_index(r.get("zeroBalanceEffectiveDate"), "/")) is not None
                    and lo <= e <= hi]
        if not eligible:
            continue
        if len({r["terminationIndicator"] for r in eligible}) != 1:
            count("conflicting_termination_code")
            continue
        row = eligible[0]
        returned = row["terminationIndicator"] == "2"
        event = month_index(row["zeroBalanceEffectiveDate"], "/")
        proceeds = sum(float(r.get("liquidationProceedsAmount") or 0) for r in rows
                       if (m := month_index(r.get("reportingPeriodEndDate"), "-")) is not None
                       and 0 <= m - event <= 2)
        attrs = next((r for r in rows if float(r.get("contractResidualValue") or 0) > 0), row)
        q = float(attrs.get("contractResidualValue") or 0)
        b = float(attrs.get("baseResidualValue") or 0)
        msrp = float(attrs.get("vehicleValueAmount") or 0)
        acq = float(attrs.get("acquisitionCost") or 0)
        term = float(attrs.get("originalLeaseTermNumber") or 0)
        if returned and proceeds <= 0:
            count("returned_without_proceeds")
            continue
        if q <= 0 or b <= 0 or b > q * 1.001:
            count("invalid_residual")
            continue
        if msrp <= 0 or acq <= 0 or term <= 0:
            count("missing_msrp_cost_or_term")
            continue
        count("returned" if returned else "retained")
        kept.append({"q": q, "b": b, "msrp": msrp, "acq": acq, "term": term,
                     "type": attrs.get("vehicleTypeCode"), "p": proceeds, "r": returned,
                     "brand": (attrs.get("vehicleManufacturerName") or "").strip().upper()})
    d = {k: np.array([x[k] for x in kept]) for k in kept[0]}
    d["counts"], d["n"] = counts, len(kept)
    return d


def add_features(d, brands):
    fallback = brands.index("OTHER") if "OTHER" in brands else 0
    brand = np.array([brands.index(b) if b in brands else fallback for b in d["brand"]])
    d["brand_idx"] = brand
    d["X"] = np.column_stack([
        np.ones(d["n"]), d["q"] / d["msrp"], d["b"] / d["q"], d["acq"] / d["msrp"],
        d["term"] / 36, d["type"] == "3", d["type"] == "2",
        *(brand == j for j in range(1, len(brands))),
    ]).astype(float)
    return d


def brand_groups(train):
    names, freq = np.unique(train["brand"], return_counts=True)
    order = np.argsort(-freq)
    major = [names[i] for i in order if freq[i] / train["n"] >= MIN_BRAND_SHARE]
    if len(major) < len(names):
        major.append("OTHER")
    return major


def resample(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ----------------------------------------------------------------------------
# Risk models (training cohort only)
# ----------------------------------------------------------------------------

class Logit:
    def __init__(self, X, y, l2=1e-2):
        self.mu, self.sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-12
        Z = self._z(X)

        def f(w):
            z = Z @ w
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * l2 * w[1:] @ w[1:]

        def g(w):
            grad = Z.T @ (1 / (1 + np.exp(-(Z @ w))) - y)
            grad[1:] += l2 * w[1:]
            return grad
        self.w = minimize(f, np.zeros(Z.shape[1]), jac=g, method="L-BFGS-B").x

    def _z(self, X):
        return np.column_stack([X[:, 0], (X[:, 1:] - self.mu) / self.sd])

    def __call__(self, X):
        return 1 / (1 + np.exp(-(self._z(X) @ self.w)))


class Severity:
    """Linear model for proceeds/contract residual on returned leases, with the
    empirical training residual distribution used for the expected shortfall."""

    def __init__(self, X, ratio):
        self.w, *_ = np.linalg.lstsq(X, ratio, rcond=None)
        self.e = np.sort(ratio - X @ self.w)
        self.csum = np.cumsum(self.e)

    def expected_shortfall(self, X, c):
        t = c - X @ self.w
        k = np.searchsorted(self.e, t)
        below = np.where(k > 0, self.csum[np.maximum(k - 1, 0)], 0.0)
        return (k * t - below) / len(self.e)


class RiskScore:
    def __init__(self, train, columns, c):
        self.cols = list(columns)
        X = train["X"][:, self.cols]
        self.ret = Logit(X, train["r"].astype(float))
        rr = train["r"]
        self.sev = Severity(X[rr], train["p"][rr] / train["q"][rr])
        self.c = c

    def prob(self, d):
        return self.ret(d["X"][:, self.cols])

    def __call__(self, d, severity=True):
        pr = self.prob(d)
        return pr * self.sev.expected_shortfall(d["X"][:, self.cols], self.c) if severity else pr


# ----------------------------------------------------------------------------
# Valuation rules
# ----------------------------------------------------------------------------

def shortfall(values, d):
    return np.where(d["r"], np.maximum(values - d["p"], 0), 0)


def bisect(fn, lo, hi, target, iters=60):
    """Largest x in [lo, hi] with fn(x) <= target for nondecreasing fn."""
    if fn(lo) > target:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) <= target else (lo, mid)
    return lo


class Uniform:
    name = "Uniform ratio on contract residual"

    def fit(self, d, target, alpha):
        self.t = bisect(lambda t: cvar(shortfall(d["q"] * t, d), alpha), 0, 1, target)
        return self

    def value(self, d):
        return d["q"] * self.t


class ScaledBase:
    name = "Scaled issuer base residual"

    @staticmethod
    def values(d, t):
        return np.minimum(d["b"] * t, d["q"])

    def fit(self, d, target, alpha):
        self.t = bisect(lambda t: cvar(shortfall(self.values(d, t), d), alpha), 0, 1.5, target)
        return self

    def value(self, d):
        return self.values(d, self.t)


def group_lp(d, group, G, target, alpha, monotone, anchor="q", upper=1.0):
    """Rockafellar-Uryasev LP: maximize mean valued residual subject to the
    empirical CVaR of per-lease shortfall not exceeding target. Valued residual
    is theta[group] * anchor."""
    q, p, r = d[anchor], d["p"], d["r"]
    n = len(q)
    idx = np.flatnonzero(r)
    m = len(idx)
    obj = np.zeros(G + 1 + m)
    obj[:G] = -np.bincount(group, weights=q, minlength=G) / n
    k = np.arange(m)
    rows = [np.repeat(k, 3), np.full(m, m), [m]]
    cols = [np.column_stack([group[idx], np.full(m, G), G + 1 + k]).ravel(), G + 1 + k, [G]]
    vals = [np.column_stack([q[idx], -np.ones(m), -np.ones(m)]).ravel(),
            np.full(m, 1 / (n * (1 - alpha))), [1.0]]
    rhs = [p[idx], [target]]
    nrow = m + 1
    if monotone:
        j = np.arange(G - 1)
        rows.append(np.repeat(nrow + j, 2))
        cols.append(np.column_stack([j + 1, j]).ravel())
        vals.append(np.tile([1.0, -1.0], G - 1))
        rhs.append(np.zeros(G - 1))
        nrow += G - 1
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nrow, G + 1 + m)).tocsr()
    res = linprog(obj, A_ub=A, b_ub=np.concatenate(rhs),
                  bounds=[(0, upper)] * G + [(0, None)] * (m + 1), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:G]


class Brand:
    name = "Brand-specific ratios"

    def __init__(self, groups):
        self.groups = groups

    def fit(self, d, target, alpha):
        self.ratios = group_lp(d, d["brand_idx"], self.groups, target, alpha, monotone=False)
        return self

    def value(self, d):
        return d["q"] * self.ratios[d["brand_idx"]]


class RiskBucket:
    """Bucket ratios applied to the contract residual (anchor="q") or to the
    issuer base residual (anchor="b"). With the base anchor, ratios may exceed
    one, but valued residual is capped at the contract residual; the LP ignores
    the cap, which only lowers value and shortfall, so the training CVaR limit
    still holds."""

    def __init__(self, score, K=4, severity=True, monotone=True, anchor="q"):
        self.score, self.K, self.severity, self.monotone = score, K, severity, monotone
        self.anchor = anchor
        self.upper = 1.0 if anchor == "q" else 1.5

    def bucket(self, d):
        return np.searchsorted(self.edges, self.score(d, self.severity))

    def fit(self, d, target, alpha):
        s = self.score(d, self.severity)
        self.edges = np.quantile(s, np.linspace(0, 1, self.K + 1)[1:-1])
        self.ratios = group_lp(d, self.bucket(d), self.K, target, alpha, self.monotone,
                               self.anchor, self.upper)
        return self

    def value(self, d):
        return np.minimum(d[self.anchor] * self.ratios[self.bucket(d)], d["q"])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def metrics(values, d, alpha):
    loss = shortfall(values, d)
    return {"mean_valued_residual": float(values.mean()),
            "mean_shortfall": float(loss.mean()),
            "loss_rate": float(np.mean(loss > 0)),
            "cvar": cvar(loss, alpha)}


def equal_value_benchmarks(values, d):
    """Uniform and scaled-base benchmarks with the same mean valued residual."""
    u = d["q"] * values.mean() / d["q"].mean()
    t = bisect(lambda t: ScaledBase.values(d, t).mean(), 0, 1.5, values.mean())
    return u, ScaledBase.values(d, t)


def equal_risk_release(values, d, alpha):
    """Extra mean valued residual of `values` over each benchmark tuned in-sample
    on the evaluation cohort to exactly the same CVaR (favours the benchmark)."""
    c = cvar(shortfall(values, d), alpha)
    tu = bisect(lambda t: cvar(shortfall(d["q"] * t, d), alpha), 0, 1, c)
    tb = bisect(lambda t: cvar(shortfall(ScaledBase.values(d, t), d), alpha), 0, 1.5, c)
    return {"vs_uniform": float(values.mean() - (d["q"] * tu).mean()),
            "vs_scaled_base": float(values.mean() - ScaledBase.values(d, tb).mean())}


def bootstrap_gaps(rule, d, alpha, reps):
    out = {"cvar_gap_vs_uniform": [], "cvar_gap_vs_scaled_base": [],
           "mean_shortfall_gap_vs_uniform": [], "mean_shortfall_gap_vs_scaled_base": []}
    for _ in range(reps):
        s = resample(d, RNG.integers(0, d["n"], d["n"]))
        v = rule.value(s)
        u, sb = equal_value_benchmarks(v, s)
        lv, lu, lb = shortfall(v, s), shortfall(u, s), shortfall(sb, s)
        out["cvar_gap_vs_uniform"].append(cvar(lv, alpha) - cvar(lu, alpha))
        out["cvar_gap_vs_scaled_base"].append(cvar(lv, alpha) - cvar(lb, alpha))
        out["mean_shortfall_gap_vs_uniform"].append(lv.mean() - lu.mean())
        out["mean_shortfall_gap_vs_scaled_base"].append(lv.mean() - lb.mean())
    return {k: {"mean": float(np.mean(x)), "ci95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                "share_rule_better": float(np.mean(np.array(x) < 0))} for k, x in out.items()}


def refit_bootstrap(train, evals, alpha, reps, anchor="q"):
    """Resample the training cohort, refit the whole pipeline, and evaluate on
    the fixed later cohorts: captures estimation risk in the rule itself."""
    out = {k: [] for k in evals}
    cols = range(train["X"].shape[1])
    for _ in range(reps):
        t = resample(train, RNG.integers(0, train["n"], train["n"]))
        target = cvar(shortfall(t["b"], t), alpha)
        c = Uniform().fit(t, target, alpha).t
        rule = RiskBucket(RiskScore(t, cols, c), 4, anchor=anchor).fit(t, target, alpha)
        for k, d in evals.items():
            g = point_gaps(rule, d, alpha)
            out[k].append((g["cvar_rule"] - g["cvar_uniform"], g["cvar_rule"] - g["cvar_scaled_base"]))
    summary = {}
    for k, x in out.items():
        x = np.array(x)
        summary[k] = {name: {"mean": float(x[:, i].mean()),
                             "ci95": [float(np.percentile(x[:, i], 2.5)), float(np.percentile(x[:, i], 97.5))],
                             "share_rule_better": float(np.mean(x[:, i] < 0))}
                      for i, name in enumerate(("cvar_gap_vs_uniform", "cvar_gap_vs_scaled_base"))}
    return summary


def point_gaps(rule, d, alpha, proceeds_scale=1.0):
    d = dict(d, p=d["p"] * proceeds_scale)
    v = rule.value(d)
    u, sb = equal_value_benchmarks(v, d)
    return {"cvar_rule": cvar(shortfall(v, d), alpha),
            "cvar_uniform": cvar(shortfall(u, d), alpha),
            "cvar_scaled_base": cvar(shortfall(sb, d), alpha),
            "mean_shortfall_rule": float(shortfall(v, d).mean()),
            "mean_shortfall_uniform": float(shortfall(u, d).mean()),
            "mean_shortfall_scaled_base": float(shortfall(sb, d).mean())}


def bucket_profile(rule, data):
    rows = []
    for name, d in data.items():
        g = rule.bucket(d)
        for j in range(rule.K):
            m = g == j
            mr = m & d["r"]
            ratio = d["p"][mr] / d["q"][mr]
            rows.append({"cohort": name, "bucket": j + 1, "n": int(m.sum()),
                         "return_rate": float(d["r"][m].mean()),
                         "mean_proceeds_to_contract": float(ratio.mean()),
                         "p05_proceeds_to_contract": float(np.percentile(ratio, 5)),
                         "mean_shortfall_at_contract": float(shortfall(d["q"][m], resample(d, m)).mean()),
                         "valuation_ratio": float(rule.ratios[j])})
    return rows


def auc(score, y):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    return float((rank[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def figures(frontier, profile, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 600})

    fig, axes = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.6), sharey=False)
    styles = {"Uniform ratio on contract residual": ("tab:gray", "o", "Uniform ratio"),
              "Scaled issuer base residual": ("tab:blue", "s", "Scaled base residual"),
              "Brand-specific ratios": ("tab:orange", "^", "Brand ratios"),
              "Return-aware buckets (K=4)": ("tab:red", "D", "Contract-anchored (proposed)"),
              BASE_ANCHORED: ("tab:purple", "v", "Base-anchored (proposed)")}
    for ax, cohort in zip(axes, data):
        for name, (color, marker, label) in styles.items():
            pts = [(r[cohort]["cvar"], r[cohort]["mean_valued_residual"]) for r in frontier[name]]
            ax.plot(*zip(*pts), color=color, marker=marker, ms=3, lw=1, label=label)
        rb = frontier["reported_base"][cohort]
        ax.plot(rb["cvar"], rb["mean_valued_residual"], "k*", ms=9, label="Reported base residual")
        ax.set_title(COHORT_LABEL[cohort])
        ax.set_xlabel("CVaR$_{99}$ of per-lease shortfall (USD)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Mean valued residual (USD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "frontier.png")
    fig.savefig(out / "frontier.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3))
    for cohort, marker in zip(data, "osD"):
        rows = [r for r in profile if r["cohort"] == cohort]
        x = [r["bucket"] for r in rows]
        axes[0].plot(x, [r["return_rate"] for r in rows], marker=marker, label=COHORT_LABEL[cohort])
        axes[1].plot(x, [r["mean_shortfall_at_contract"] for r in rows], marker=marker, label=COHORT_LABEL[cohort])
    axes[0].set_ylabel("Return rate")
    axes[1].set_ylabel("Mean shortfall per lease at\ncontract residual (USD, log)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Risk bucket (training quantiles)")
        ax.set_xticks(x)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "buckets.png")
    fig.savefig(out / "buckets.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--refit-bootstrap", type=int, default=200)
    parser.add_argument("--issuer", choices=sorted(ISSUERS), default="gm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alpha = args.alpha
    suffix = "" if args.issuer == "gm" else f"_{args.issuer}"
    output = args.output or HERE / f"study_results{suffix}.json"
    figure_dir = HERE / "figures" / ("" if args.issuer == "gm" else args.issuer)

    cohorts = ISSUERS[args.issuer]
    data = {k: load(*v) for k, v in cohorts.items()}
    train = data["train"]
    brands = brand_groups(train)
    for d in data.values():
        add_features(d, brands)
    features = list(BASE_FEATURES) + [f"brand_{b.lower()}" for b in brands[1:]]
    holdouts = [k for k in data if k != "train"]
    target = cvar(shortfall(train["b"], train), alpha)
    c = Uniform().fit(train, target, alpha).t
    all_cols = range(len(features))
    score = RiskScore(train, all_cols, c)
    score_no_base = RiskScore(train, [i for i in all_cols if i != 2], c)
    score_parsimonious = RiskScore(train, [0, 1, 2], c)

    def main_rules():
        return {"Uniform ratio on contract residual": Uniform(),
                "Scaled issuer base residual": ScaledBase(),
                "Brand-specific ratios": Brand(len(brands)),
                "Return-aware buckets (K=4)": RiskBucket(score, 4),
                BASE_ANCHORED: RiskBucket(score, 4, anchor="b")}

    rules = main_rules()
    for r in rules.values():
        r.fit(train, target, alpha)
    ablations = {
        "K=2": RiskBucket(score, 2), "K=3": RiskBucket(score, 3),
        "K=5": RiskBucket(score, 5), "K=8": RiskBucket(score, 8),
        "Return probability only": RiskBucket(score, 4, severity=False),
        "Without base/contract feature": RiskBucket(score_no_base, 4),
        "Two ratios only (residual/MSRP, base/contract)": RiskBucket(score_parsimonious, 4),
        "No monotonicity constraint": RiskBucket(score, 4, monotone=False),
    }
    for r in ablations.values():
        r.fit(train, target, alpha)
    proposed = rules["Return-aware buckets (K=4)"]

    report = {
        "alpha": alpha, "training_target_cvar": target,
        "issuer": args.issuer,
        "design": {k: {"file": v[0], "terminations": [v[1], v[2]]} for k, v in cohorts.items()},
        "sample": {k: {"n": d["n"], "returned": int(d["r"].sum()), "counts": d["counts"]} for k, d in data.items()},
        "return_model": {"standardized_coefficients": dict(zip(features, map(float, score.ret.w))),
                         "auc": {k: auc(score.prob(d), d["r"]) for k, d in data.items()}},
        "severity_model_coefficients": dict(zip(features, map(float, score.sev.w))),
        "fitted": {"uniform_ratio": c,
                   "scaled_base_ratio": rules["Scaled issuer base residual"].t,
                   "brand_ratios": dict(zip(brands, map(float, rules["Brand-specific ratios"].ratios))),
                   "bucket_ratios": list(map(float, proposed.ratios)),
                   "bucket_edges_score": list(map(float, proposed.edges))},
        "at_target": {k: {"reported_base": metrics(d["b"], d, alpha),
                          **{n: metrics(r.value(d), d, alpha) for n, r in rules.items()}}
                      for k, d in data.items()},
        "equal_value_gaps": {k: {n: point_gaps(r, d, alpha) for n, r in {**rules, **ablations}.items()
                                 if not n.startswith("Uniform")}
                             for k, d in data.items()},
        "equal_risk_release": {k: {n: equal_risk_release(r.value(d), d, alpha)
                                   for n, r in {**rules, **ablations}.items()}
                               for k, d in data.items()},
        "stress_proceeds": {k: {str(s): point_gaps(proposed, d, alpha, s) for s in (1.0, 0.95, 0.90, 0.85)}
                            for k, d in data.items() if k != "train"},
        "bucket_profile": bucket_profile(proposed, data),
    }

    print("bootstrap ...", flush=True)
    report["bootstrap"] = {k: {n: bootstrap_gaps(rules[n], data[k], alpha, args.bootstrap)
                               for n in ("Return-aware buckets (K=4)", BASE_ANCHORED, "Brand-specific ratios")}
                           for k in holdouts}

    print("refit bootstrap ...", flush=True)
    report["refit_bootstrap"] = {
        name: refit_bootstrap(train, {k: data[k] for k in holdouts}, alpha, args.refit_bootstrap, anchor)
        for name, anchor in (("Return-aware buckets (K=4)", "q"), (BASE_ANCHORED, "b"))}

    print("frontier ...", flush=True)
    frontier = {n: [] for n in main_rules()}
    for scale in np.linspace(0.7, 1.4, 8):
        for n, r in main_rules().items():
            r.fit(train, target * scale, alpha)
            frontier[n].append({"target": target * scale,
                                **{k: metrics(r.value(d), d, alpha) for k, d in data.items()}})
    frontier["reported_base"] = {k: metrics(d["b"], d, alpha) for k, d in data.items()}
    report["frontier"] = frontier

    output.write_text(json.dumps(report, indent=2))
    figures(frontier, report["bucket_profile"], data, figure_dir)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
