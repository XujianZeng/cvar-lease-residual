#!/usr/bin/env python3
"""Cross-issuer summary of `study_results*.json`: forest plot and Markdown table."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNS = [("GM", "study_results.json"), ("Nissan", "study_results_nissan.json"),
        ("VW", "study_results_vw.json"), ("Ford", "study_results_ford.json")]
RULES = [("Contract-anchored", "Return-aware buckets (K=4)", "tab:red", "D"),
         ("Base-anchored", "Return-aware buckets on base residual (K=4)", "tab:purple", "v"),
         ("Brand ratios", "Brand-specific ratios", "tab:orange", "^")]
COHORT_LABEL = {"validation": "Holdout 1", "later": "Holdout 2"}
BENCHMARKS = [("uniform", "Uniform ratio"), ("scaled_base", "Scaled issuer base residual")]


def collect():
    rows = []
    for issuer, file in RUNS:
        r = json.loads((HERE / file).read_text())
        for cohort, boots in r["bootstrap"].items():
            for label, key, *_ in RULES:
                g = r["equal_value_gaps"][cohort][key]
                entry = {"issuer": issuer, "cohort": cohort, "rule": label,
                         "n": r["sample"][cohort]["n"], "returned": r["sample"][cohort]["returned"],
                         "auc": r["return_model"]["auc"][cohort]}
                for bench, _ in BENCHMARKS:
                    ref = g[f"cvar_{bench}"]
                    ci = boots[key][f"cvar_gap_vs_{bench}"]["ci95"]
                    entry[bench] = {"gap": g["cvar_rule"] - ref, "pct": 100 * (g["cvar_rule"] / ref - 1),
                                    "ci": ci, "ci_pct": [100 * c / ref for c in ci]}
                rows.append(entry)
    return rows


def table(rows):
    lines = ["| Issuer | Holdout | Leases (returned) | Rule | CVaR99 gap vs uniform, % [95% CI] | "
             "CVaR99 gap vs scaled base, % [95% CI] |", "| --- | --- | --- | --- | ---: | ---: |"]
    for e in rows:
        cells = []
        for bench, _ in BENCHMARKS:
            x = e[bench]
            cells.append(f"{x['pct']:+.0f} [{x['ci_pct'][0]:+.0f}, {x['ci_pct'][1]:+.0f}]")
        lines.append(f"| {e['issuer']} | {e['cohort']} | {e['n']:,} ({e['returned']:,}) | {e['rule']} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def forest(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 600})
    holdouts = list(dict.fromkeys((e["issuer"], e["cohort"]) for e in rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(holdouts) + 1.4), sharey=True)
    for ax, (bench, title) in zip(axes, BENCHMARKS):
        for j, (label, _, color, marker) in enumerate(RULES):
            for i, h in enumerate(holdouts):
                e = next(x for x in rows if (x["issuer"], x["cohort"]) == h and x["rule"] == label)
                y = i + (j - 1) * 0.22
                lo, hi = e[bench]["ci_pct"]
                ax.plot([lo, hi], [y, y], color=color, lw=1.2)
                ax.plot(e[bench]["pct"], y, marker=marker, color=color, ms=5,
                        label=label if i == 0 else None)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"vs {title}")
        ax.set_xlabel("CVaR$_{99}$ change at equal valued residual (%)")
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(range(len(holdouts)))
    axes[0].set_yticklabels([f"{i} {COHORT_LABEL[c]}" for i, c in holdouts])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(RULES), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "forest.png")
    fig.savefig(out / "forest.pdf")
    plt.close(fig)


def discrimination(rows, out):
    """Gain vs the issuer's scaled base residual against holdout return-model AUC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for label, _, color, marker in RULES[:2]:
        pts = [e for e in rows if e["rule"] == label]
        x = np.array([e["auc"] for e in pts])
        y = np.array([e["scaled_base"]["pct"] for e in pts])
        err = np.array([[e["scaled_base"]["pct"] - e["scaled_base"]["ci_pct"][0],
                         e["scaled_base"]["ci_pct"][1] - e["scaled_base"]["pct"]] for e in pts]).T
        ax.errorbar(x, y, yerr=err, fmt=marker, color=color, ms=5, lw=1, capsize=2, label=label)
        for e in pts if label == RULES[0][0] else []:
            ax.annotate(f"{e['issuer']} H{1 if e['cohort'] == 'validation' else 2}", (e["auc"], e["scaled_base"]["pct"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Holdout AUC of return model")
    ax.set_ylabel("CVaR$_{99}$ change vs scaled\nissuer base residual (%)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "gain_vs_auc.png")
    fig.savefig(out / "gain_vs_auc.pdf")
    plt.close(fig)


def main():
    rows = collect()
    (HERE / "cross_issuer_summary.json").write_text(json.dumps(rows, indent=2))
    (HERE / "cross_issuer_table.md").write_text(table(rows) + "\n")
    forest(rows, HERE / "figures")
    discrimination(rows, HERE / "figures")
    for label, *_ in RULES[:2]:
        pts = [e for e in rows if e["rule"] == label]
        rho = np.corrcoef([e["auc"] for e in pts], [e["scaled_base"]["pct"] for e in pts])[0, 1]
        print(f"corr(AUC, gain vs scaled base) {label}: {rho:.2f}")
    print(table(rows))


if __name__ == "__main__":
    main()
