#!/usr/bin/env python3
"""Pilot of return-aware residual haircut calibration on SEC auto leases."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


GROUPS = ("BUICK", "CADILLAC", "CHEVROLET", "GMC")
ALPHA = 0.99


def number(value: str | None) -> float:
    return float(value or 0)


def load_cohort(path: Path, year: int, first_event_month: int, last_event_month: int):
    by_asset = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            by_asset[row["assetNumber"]].append(row)

    records = []
    counts = Counter()
    for asset, rows in by_asset.items():
        eligible = []
        for row in rows:
            date = row.get("zeroBalanceEffectiveDate") or ""
            if row.get("zeroBalanceCode") != "1" or row.get("terminationIndicator") not in ("1", "2"):
                continue
            if len(date.split("/")) != 2:
                continue
            month, event_year = map(int, date.split("/"))
            if event_year == year and first_event_month <= month <= last_event_month:
                eligible.append(row)
        if not eligible:
            continue
        codes = {row["terminationIndicator"] for row in eligible}
        if len(codes) != 1:
            counts["conflicting_termination_code"] += 1
            continue
        row = eligible[0]
        returned = row["terminationIndicator"] == "2"
        counts["returned" if returned else "retained"] += 1
        q = number(row.get("contractResidualValue"))
        b = number(row.get("baseResidualValue"))
        event_month_index = year * 12 + int(row["zeroBalanceEffectiveDate"].split("/")[0])
        proceeds = 0.0
        for item in rows:
            report_date = item.get("reportingPeriodEndDate") or ""
            parts = report_date.split("-")
            if len(parts) != 3:
                continue
            report_month_index = int(parts[2]) * 12 + int(parts[0])
            if 0 <= report_month_index - event_month_index <= 2:
                proceeds += number(item.get("liquidationProceedsAmount"))
        if returned and proceeds <= 0:
            counts["returned_without_proceeds"] += 1
            continue
        if q <= 0 or b <= 0 or b > q * 1.001:
            counts["invalid_residual"] += 1
            continue
        brand = row.get("vehicleManufacturerName")
        if brand not in GROUPS:
            counts["other_brand"] += 1
            continue
        records.append({"asset": asset, "brand": brand, "returned": returned,
                        "q": q, "b": b, "proceeds": proceeds,
                        "event_month": int((row["zeroBalanceEffectiveDate"] or "0/").split("/")[0])})
    return records, dict(counts)


def cvar(loss: np.ndarray, alpha: float = ALPHA) -> float:
    if not len(loss):
        return float("nan")
    ordered = np.sort(loss)
    tail = len(loss) * (1 - alpha)
    k = int(math.floor(tail))
    fractional = tail - k
    tail_sum = ordered[-k:].sum() if k else 0.0
    if fractional > 1e-9:
        tail_sum += fractional * ordered[-k - 1]
    return float(tail_sum / tail)


def arrays(records):
    return (
        np.array([x["q"] for x in records]),
        np.array([x["b"] for x in records]),
        np.array([x["proceeds"] for x in records]),
        np.array([x["returned"] for x in records], dtype=bool),
        np.array([GROUPS.index(x["brand"]) for x in records]),
    )


def loss_for(values, proceeds, returned):
    return np.where(returned, np.maximum(values - proceeds, 0), 0)


def uniform_haircut(q, proceeds, returned, target):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if cvar(loss_for(q * mid, proceeds, returned)) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def optimized_haircuts(q, proceeds, returned, group, target,
                       center: float | None = None, max_deviation: float | None = None):
    n = len(q)
    idx = np.flatnonzero(returned)
    m = len(idx)
    g = len(GROUPS)
    # Variables: group multipliers, CVaR threshold, and one excess-loss
    # variable for each observed return. See Rockafellar-Uryasev CVaR LP.
    objective = np.zeros(g + 1 + m)
    for j in range(g):
        objective[j] = -q[group == j].sum() / n
    rr, cc, vv = [], [], []
    for k, i in enumerate(idx):
        rr.extend((k, k, k))
        cc.extend((group[i], g, g + 1 + k))
        vv.extend((q[i], -1.0, -1.0))
    for k in range(m):
        rr.append(m)
        cc.append(g + 1 + k)
        vv.append(1.0 / (n * (1 - ALPHA)))
    rr.append(m)
    cc.append(g)
    vv.append(1.0)
    matrix = coo_matrix((vv, (rr, cc)), shape=(m + 1, g + 1 + m)).tocsr()
    rhs = np.r_[proceeds[idx], target]
    if center is None or max_deviation is None:
        ratio_bounds = [(0, 1)] * g
    else:
        ratio_bounds = [(max(0, center - max_deviation),
                         min(1, center + max_deviation))] * g
    bounds = ratio_bounds + [(0, None)] * (m + 1)
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds,
                     method="highs")
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x[:g]


def metrics(values, proceeds, returned):
    loss = loss_for(values, proceeds, returned)
    return {"mean_residual": round(float(values.mean()), 2),
            "mean_haircut_from_contract": None,
            "mean_shortfall": round(float(loss.mean()), 2),
            "loss_rate": round(float(np.mean(loss > 0)), 5),
            "cvar99": round(cvar(loss), 2)}


def evaluate(records, uniform_ratio, group_ratios):
    q, b, proceeds, returned, group = arrays(records)
    brand_values = q * group_ratios[group]
    equal_value_ratio = float(brand_values.mean() / q.mean())
    variants = {"contract_no_haircut": q, "reported_base_residual": b,
                "uniform_calibrated": q * uniform_ratio,
                "brand_calibrated": brand_values,
                "uniform_equal_value": q * equal_value_ratio}
    report = {}
    for name, values in variants.items():
        result = metrics(values, proceeds, returned)
        result["mean_haircut_from_contract"] = round(float((q - values).mean()), 2)
        report[name] = result
    return report


def equal_value_cvar_gap(records, group_ratios):
    q, _, proceeds, returned, group = arrays(records)
    group_values = q * group_ratios[group]
    uniform_values = q * (group_values.mean() / q.mean())
    return round(cvar(loss_for(group_values, proceeds, returned)) -
                 cvar(loss_for(uniform_values, proceeds, returned)), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train, train_counts = load_cohort(args.train, 2024, 1, 9)
    test, test_counts = load_cohort(args.test, 2025, 1, 4)
    final, final_counts = load_cohort(args.final, 2025, 7, 10)
    q, b, proceeds, returned, group = arrays(train)
    target = cvar(loss_for(b, proceeds, returned))
    uniform_ratio = uniform_haircut(q, proceeds, returned, target)
    group_ratios = optimized_haircuts(q, proceeds, returned, group, target)
    sensitivity = []
    for cap in (0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
        capped = optimized_haircuts(q, proceeds, returned, group, target,
                                    center=uniform_ratio, max_deviation=cap)
        sensitivity.append({
            "max_deviation": cap,
            "train_mean_residual_gain_over_uniform": round(
                float((q * capped[group]).mean() - q.mean() * uniform_ratio), 2),
            "validation_equal_value_cvar99_gap": equal_value_cvar_gap(test, capped),
            "final_equal_value_cvar99_gap": equal_value_cvar_gap(final, capped),
        })
    report = {
        "design": "GM 2022-3 Jan-Sep 2024 train; GM 2023-1 Jan-Apr 2025 validation; GM 2023-3 Jul-Oct 2025 exploratory final check; two months of sale follow-up",
        "alpha": ALPHA,
        "training_target_cvar99": round(target, 2),
        "training_records": len(train), "test_records": len(test),
        "final_records": len(final),
        "training_counts": train_counts, "test_counts": test_counts,
        "final_counts": final_counts,
        "uniform_ratio": round(float(uniform_ratio), 6),
        "brand_ratios": {brand: round(float(group_ratios[i]), 6) for i, brand in enumerate(GROUPS)},
        "train": evaluate(train, uniform_ratio, group_ratios),
        "test": evaluate(test, uniform_ratio, group_ratios),
        "final": evaluate(final, uniform_ratio, group_ratios),
        "exploratory_deviation_caps": sensitivity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
