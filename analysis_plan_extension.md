# Analysis plan: out-of-time extension across the 2019–2024 used-car cycle

Written 2026-09-25, **before** any of the data below was downloaded. The file's
modification time and the SHA-256 recorded in `README.md` document this. No rule,
feature, parameter, metric or test below may be changed after the data are seen;
any deviation will be reported as such in the paper.

## Purpose

The main study evaluates rules on terminations from 2024–2026 only. This
extension asks whether the conclusions hold across an earlier and very
different used-car market (high return rates before 2021, a price boom in
2021–2022, normalization afterwards).

## Data (GM Financial only)

One trust per termination year; leases terminating January–September of year
*Y*; ABS-EE filings dated January 1 to December 31 of year *Y* (the same
construction as the existing GM training cohort). Trusts were chosen only by
their filing date ranges, before any outcome was inspected.

| Termination year *Y* | Trust | CIK |
| --- | --- | --- |
| 2019 | GMALT 2017-3 | 0001712799 |
| 2020 | GMALT 2018-3 | 0001747909 |
| 2021 | GMALT 2019-3 | 0001781258 |
| 2022 | GMALT 2020-3 | 0001822360 |
| 2023 | GMALT 2021-3 | 0001870475 |
| 2024 | GMALT 2022-3 | 0001935132 (already collected: `data/gm_2022_3_2024.jsonl`) |

Loading, exclusions, the proceeds window (termination month plus two reports)
and features are exactly those of `buffer_study.load` / `add_features`.
A cohort is used only if it has at least 1,000 leases and 100 returned leases
with proceeds; otherwise the pair is dropped and the drop is reported.

(A probe of one June 2019 filing of GMALT 2017-3 was downloaded on 2026-09-25 to
check field availability. It showed that all required fields are present and
that roughly 83% of May 2019 terminations were returns. No rule was fitted or
evaluated on it.)

## Design: rolling one-year-ahead evaluation

For each *Y* in 2019–2023: fit on the year-*Y* cohort, evaluate unchanged on the
year-(*Y*+1) cohort. This gives five out-of-time tests (2020, 2021, 2022, 2023,
2024). Rules, fitted exactly as in `buffer_study.py`: uniform ratio, scaled
issuer base residual, brand ratios, contract-anchored return-aware buckets
(K = 4) and base-anchored return-aware buckets (K = 4); alpha = 0.99; risk limit
= CVaR99 of the issuer base residual on the training cohort.

## Metrics and tests

1. Equal-value % change in CVaR99: contract-anchored vs uniform ratio (primary
   for that rule) and base-anchored vs scaled issuer base residual (primary for
   that rule); also contract-anchored vs scaled base and brand vs scaled base.
2. Uncertainty: 1,000 lease-level bootstrap resamples of each test cohort;
   two-sided bootstrap p-values; Holm adjustment across the five tests.
3. Pooled equal-weighted mean change across the five tests with a stratified
   bootstrap 95% CI.
4. Pool-level stress: loss-rate gap at a common 20% proceeds decline (S = 0.8)
   against the equal-value benchmark, as in `revision_study.py`.
5. Test-cohort AUC of the return model and return rate, reported per year.

## Hypotheses stated in advance

- H1: The base-anchored rule is not significantly worse (Holm-adjusted) than
  the scaled issuer base residual in any of the five tests.
- H2: The pooled change for contract-anchored vs uniform ratio is negative with
  a 95% CI excluding zero.
- H3: Across all 13 holdouts (8 existing + 5 new), the base-anchored rule's %
  change vs scaled base is negatively related to the test-cohort AUC
  (Spearman correlation < 0). Reported descriptively with its bootstrap-free
  p-value; not used for any decision.
- Expectation (not a test): gains are small in years with very high return
  rates (2019–2020) and larger when returns are selective.

## Reporting

All five tests are reported in the paper whatever their direction, together
with this plan's date and any deviation from it.
