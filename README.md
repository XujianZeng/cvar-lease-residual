# CVaR lease residual analysis

[简体中文](README.zh-CN.md) | English

This project studies how to set **valued residuals for securitized auto leases** when a vehicle may be returned and sold for less than its assigned residual value. A higher valued residual preserves more value in a lease pool, but can increase losses in adverse outcomes. The research question is whether a rule that accounts for return risk can retain more value while controlling those losses.

For each eligible terminated lease, the code measures shortfall as `max(valued residual − net liquidation proceeds, 0)` if the vehicle was returned, and zero otherwise. It uses **CVaR at 99%**—the average shortfall among the worst 1% of leases—as the tail-risk measure. On a training cohort, it fits valuation ratios under a CVaR constraint based on the issuer's reported base residual. It then applies the fitted rules unchanged to later cohorts. Rules are also compared at the same average valued residual so their tail risk can be assessed on equal value terms.

## What the code does

1. **Collect and prepare observations.** Extract relevant lease termination fields from public SEC ABS-EE asset-level filings, join monthly records by asset, and associate returned vehicles with subsequent liquidation proceeds.
2. **Fit residual valuation rules.** Compare a uniform ratio, issuer base residuals, brand-specific ratios, and return-aware risk buckets. The bucket approach uses origination attributes to estimate return probability and potential disposal shortfall, then optimizes bucket ratios under the CVaR constraint.
3. **Check generalization.** Evaluate fixed rules on later GM, Nissan, Volkswagen, and Ford Credit cohorts; add pooled comparisons, stress scenarios, a gradient-boosting benchmark, and a GM cross-year study covering 2019–2024.

This is research code for historical lease cohorts. It is not a production valuation service or a complete portfolio loss model.

## Research record

- [`analysis_plan_extension.md`](analysis_plan_extension.md) specifies the GM 2019–2024 cross-year evaluation. The source file was last modified on 2026-09-25 at 10:51:37 China Standard Time; its SHA-256 is `74266ffe5d0c36986e78ad13118e101294fb0c36b972936bfdff382ffba55e97`.
- [`filings_used.csv`](filings_used.csv) lists 185 SEC XML filings used across 17 dataset labels, with filing dates, accession numbers, source URLs, file sizes, and extracted event-row counts. It contains filing metadata, not lease-level records.

## Repository map

| Path | Role |
| --- | --- |
| `collection/collect_sec.py` | Collect selected lease records from SEC filings. |
| `analysis/analyze.py` | Original GM brand-based pilot. |
| `analysis/buffer_study.py` | Main return-aware buffer study across issuers. |
| `analysis/revision_study.py` | Pooled evidence, stress analysis, and gradient-boosting benchmark. |
| `analysis/cycle_study.py` | GM year-to-next-year evaluation. |
| `analysis/summarize.py` | Cross-issuer tables and charts from study results. |

## Running the code

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Input data is **not included**. Analysis modules expect JSONL cohorts in `data/`; required filenames and windows are defined in `analysis/buffer_study.py` and `analysis/cycle_study.py`. You can supply the cohort files locally or collect the relevant SEC filings. Collection requires `SEC_USER_AGENT` with an identifying name and contact address:

```bash
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.collect_sec \
  --cik 0001935132 --start 2024-01-01 --end 2024-12-31 \
  --output data/gm_2022_3_2024.jsonl
```

With all files for an issuer present, run its study from the repository root:

```bash
.venv/bin/python -m analysis.buffer_study --issuer gm
```

Other issuer choices are `nissan`, `vw`, and `ford`. Run `analysis.revision_study` for additional analyses, `analysis.cycle_study` for the cross-year study, and `analysis.summarize` after generating results for all four issuers. `cycle_study` also requires `revision_results.json`. The original GM pilot can be run with `python -m analysis.analyze --train ... --test ... --final ... --output results.json`.

Lease-level input data, generated results and figures, and manuscript drafts and build scripts are excluded. Results are written to the repository root and charts to `figures/`.
