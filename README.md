# CVaR lease residual analysis

[简体中文](README.zh-CN.md) | English

This project studies how to set **valued residuals for securitized auto leases** when a vehicle may be returned and sold for less than its assigned residual value. A higher valued residual preserves more value in a lease pool, but can increase losses in adverse outcomes. The research question is whether a rule that accounts for return risk can retain more value while controlling those losses.

For each eligible terminated lease, the code measures shortfall as `max(valued residual − net liquidation proceeds, 0)` if the vehicle was returned, and zero otherwise. It uses **CVaR at 99%**—the average shortfall among the worst 1% of leases—as the tail-risk measure. On a training cohort, it fits valuation ratios under a CVaR constraint based on the issuer's reported base residual. It then applies the fitted rules unchanged to later cohorts. Rules are also compared at the same average valued residual so their tail risk can be assessed on equal value terms.

## 2026-09-29 revision

The base-anchored implementation is a feasible capped heuristic, not an optimizer of the capped objective. The original primary samples and rules are preserved. AUC now uses average ranks for ties; bootstrap sign-tail measures use a finite-simulation plus-one correction. Both percentage-change figures use the same ratio-bootstrap intervals as the manuscript tables.

`analysis.robustness_study` refits the entire pipeline under four explicit proceeds settings and reports sample flows, dollar gaps, sparse-tail counts, undefined ratio draws, leave-one-year-out effects, month-cluster intervals and paired GB/LR comparisons. These are **post-hoc** analyses. Retaining nonpositive recorded three-month proceeds reverses the base-anchored confirmatory pooled estimate from -2.2% to +13.7%; unknown final recoveries cannot be inferred from that scenario. Do not describe the rule as non-inferior or universally robust.

The original extension plan follows a June 2019 pilot and precedes bulk collection. Its hash authenticates the saved text, not an independently certified historical date. The dated revision protocol documents these limitations and additions.

After the four main issuer runs, execute `python -m analysis.revision_study`, `python -m analysis.cycle_study`, `python -m analysis.robustness_study`, then `python -m analysis.summarize`. Run scientific checks with `python -m unittest analysis.test_invariants -v`. Setting `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` makes performance more predictable. The manuscript submission includes a separate reproducibility snapshot with the saved numerical results.

## What the code does

1. **Collect and prepare observations.** Extract relevant lease termination fields from public SEC ABS-EE asset-level filings, join monthly records by asset, and associate returned vehicles with subsequent liquidation proceeds.
2. **Fit residual valuation rules.** Compare a uniform ratio, issuer base residuals, brand-specific ratios, and return-aware risk buckets. The bucket approach uses reported lease attributes to estimate return probability and potential disposal shortfall, then fits bucket ratios under the CVaR constraint. Availability of those attributes at issuance requires separate validation.
3. **Check generalization.** Evaluate fixed rules on later GM, Nissan, Volkswagen, and Ford Credit cohorts; add pooled comparisons, stress scenarios, a gradient-boosting benchmark, and a GM cross-year study covering 2019–2024.

This is research code for historical lease cohorts. It is not a production valuation service or a complete portfolio loss model.

## Research record

- [`analysis_plan_extension.md`](analysis_plan_extension.md) specifies the GM 2019–2024 cross-year evaluation. The source file was last modified on 2026-09-25 at 10:51:37 China Standard Time; its SHA-256 is `74266ffe5d0c36986e78ad13118e101294fb0c36b972936bfdff382ffba55e97`.
- [`filings_used.csv`](filings_used.csv) lists 185 SEC XML filings used across 17 dataset labels, with filing dates, accession numbers, source URLs, file sizes, and extracted event-row counts. It contains filing metadata, not lease-level records.
- [`revision_protocol_2026-09-29.md`](revision_protocol_2026-09-29.md) records the post-hoc corrections and sensitivity analyses. The analysis and collection implementations match the revised manuscript's Supplementary File S3, apart from package imports and repository-relative paths.

## Correspondence with the revised manuscript

| Method or result | Implementation |
| --- | --- |
| AUC with average ranks for tied predictions | `analysis.buffer_study.auc` uses `rankdata(..., method="average")`. |
| Finite bootstrap sign-tail measure | `analysis.revision_study.summarize` uses `min(1, 2 * (1 + min(N_plus, N_minus)) / (B_valid + 1))`; `analysis.cycle_study.summarize` delegates to it. Counts include zero in both tails and exclude non-finite draws, whose number is reported. |
| Table 4 and Figures 3–4 percentage intervals | `analysis.revision_study.forest_rows` and `analysis.summarize.collect` use the same percentage-bootstrap intervals from `revision_results.json`. |
| Equal-value issuer benchmark | `analysis.revision_study.scaled_base_equal` rescales the issuer base residual to the rule's mean valued residual, subject to the contract cap. This differs from the unscaled reported base residual. |
| Table 7 stressed-value release | `analysis.revision_study.equal_stress_release` matches total stressed loss in USD, not the loss rate. |
| Base-anchored rule and proceeds sensitivities | `analysis.buffer_study.RiskBucket` reports cap/feasibility diagnostics; `analysis.robustness_study` refits all four dated proceeds scenarios. |

## Repository map

| Path | Role |
| --- | --- |
| `collection/collect_sec.py` | Collect selected lease records from SEC filings. |
| `collection/rebuild_from_manifest.py` | Rebuild the exact filing selection, verifying event-row counts. |
| `analysis/analyze.py` | Original GM brand-based pilot. |
| `analysis/buffer_study.py` | Main return-aware buffer study across issuers. |
| `analysis/revision_study.py` | Pooled evidence, stress analysis, and gradient-boosting benchmark. |
| `analysis/cycle_study.py` | GM year-to-next-year evaluation. |
| `analysis/robustness_study.py` | Post-hoc proceeds, sparse-tail, cluster, and paired-model checks. |
| `analysis/summarize.py` | Cross-issuer tables and charts from study results. |
| `analysis/test_invariants.py` | Scientific regression checks, including AUC ties and finite bootstrap tails. |

## Running the code

The manuscript calculations were verified with Python 3.13.13 and the pinned packages in `requirements-reproducible.txt`. Other environments have not been verified. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproducible.txt
.venv/bin/python -m unittest analysis.test_invariants -v
```

Input data is **not included**. Analysis modules expect JSONL cohorts in `data/`; required filenames and windows are defined in `analysis/buffer_study.py` and `analysis/cycle_study.py`. To reproduce the paper, inspect the exact 185-filing selection, then download it into a fresh data directory. Collection requires `SEC_USER_AGENT` with your identifying name and contact address:

```bash
.venv/bin/python -m collection.rebuild_from_manifest
export SEC_USER_AGENT='Your Name your.email@example.com'
.venv/bin/python -m collection.rebuild_from_manifest --download
```

The first command makes no network requests. The download command reads the saved filing URLs and refuses to overwrite existing dataset files. `collection.collect_sec` remains available for new filing discovery; its current SEC listing need not return the historical selection used in the paper. Supplementary File S3 records the original input hashes and saved numerical results.

After all 17 input datasets are present, reproduce the analyses in this order:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m analysis.buffer_study --alpha 0.95 --bootstrap 200 --refit-bootstrap 50 --output study_results_alpha0.95.json
.venv/bin/python -m analysis.buffer_study --alpha 0.975 --bootstrap 200 --refit-bootstrap 50 --output study_results_alpha0.975.json
.venv/bin/python -m analysis.buffer_study --issuer gm
.venv/bin/python -m analysis.buffer_study --issuer nissan
.venv/bin/python -m analysis.buffer_study --issuer vw
.venv/bin/python -m analysis.buffer_study --issuer ford
.venv/bin/python -m analysis.revision_study
.venv/bin/python -m analysis.cycle_study
.venv/bin/python -m analysis.robustness_study
.venv/bin/python -m analysis.summarize
```

Main runs use 1,000 conditional holdout bootstrap draws and 200 training-refit draws. The two alpha sensitivities use 200/50 draws and run before the GM main analysis so that the final GM figures use alpha 0.99. Proceeds sensitivities refit each scenario, then use 1,000 conditional draws; these are not 1,000 training refits. The original GM pilot is separate and can be run with `python -m analysis.analyze --train ... --test ... --final ... --output results.json`.

Lease-level input data, generated results and figures, and manuscript drafts and build scripts are excluded. Results are written to the repository root and charts to `figures/`.

Undefined robustness percentages are saved as JSON `null`, with undefined bootstrap draws counted explicitly. Refer to Supplementary File S2 and the dated protocol when interpreting sparse-tail intervals and the base-anchor sensitivity reversal.
