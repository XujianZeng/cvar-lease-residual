# Post-hoc revision protocol 29 September 2026

This document records revisions made after inspection of the primary and
cross-cycle results and the 28 September audit. It is not a preregistration.
The original `analysis_plan_extension.md` and its hash are preserved unchanged.

The primary three-month, positive-proceeds sample and original valuation rules
are retained for comparability. The contract-anchored problem is an exact LP.
The base-anchored implementation is a feasible heuristic obtained by solving
an uncapped surrogate LP and capping afterwards; no claim is made that it
maximizes the capped objective. Training feasibility and clipping diagnostics
are saved explicitly. No ratios are selected using holdout outcomes.

Four proceeds specifications are reported without choosing among them based
on performance: (1) primary three-month positive proceeds; (2) three-month
recorded nonpositive proceeds retained as a stress scenario; (3) three-month
positive proceeds, restricted to terminations with six calendar months of
dataset follow-up; (4) six-month positive proceeds on the same maturity-eligible
population. Positive-proceeds filters may retain different assets in (3) and
(4); this difference is reported. Maturity is determined from the last report
month of the entire dataset, not an individual asset's disappearance. All LR,
severity and bucket models are refitted on the training specification before
fixed-rule holdout evaluation. Scenario (2) does not assume that zero recorded
cash is a verified zero final disposal price.

Each scenario reports cohort construction counts, positive-loss counts,
absolute CVaR levels and gaps, relative gaps, and 1,000 lease-level conditional
bootstrap intervals. Five-year means and leave-one-year-out means are
reported, with undefined relative changes explicitly counted. Additional
checks use termination-month cluster resampling of the primary fitted rules
and directly paired GB/LR CVaR comparisons after scaling both down to a common
mean valued residual. These checks do not establish noninferiority or account
for all joint market and training uncertainty.

AUC uses average ranks for ties. Bootstrap tail probabilities use a plus-one
Monte Carlo correction and never claim a probability of exactly zero. Holm
families retain their stated scope. Figure 4 uses the same percentage-change
draws as Table 4. Both Pearson and Spearman correlations are reported for the
eight- and thirteen-holdout sets. New sensitivities are exploratory.

The papers distinguish residual-component stress loss rates from transaction
credit-enhancement requirements, retrospective termination-cohort evaluation
from issuance-time deployment, and failure to detect deterioration from a
positive noninferiority claim. The original plan's field-availability pilot
and the limits of its local timestamp evidence are disclosed.
