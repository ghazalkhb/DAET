#!/usr/bin/env python3
"""Validate the completed detector-specific beta/theta experiment."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Experiments" / "results" / "gate_tuning"
errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


validation = pd.read_csv(OUT / "validation_grid.csv")
test_grid = pd.read_csv(OUT / "test_grid_summary.csv")
detail = pd.read_csv(OUT / "selected_test_detail.csv")
comparison = pd.read_csv(OUT / "ranking_comparison.csv")
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

check(len(validation) == 350, f"validation rows: {len(validation)} != 350")
check(len(test_grid) == 350, f"test-grid rows: {len(test_grid)} != 350")
check(len(detail) == 16371, f"detail rows: {len(detail)} != 16371")
check(len(comparison) == 10, f"comparison rows: {len(comparison)} != 10")

expected_signatures = {"firefox-android": 342, "mozilla-beta": 1477}
for dataset, n_signatures in expected_signatures.items():
    subset = detail[detail["dataset"] == dataset]
    check(
        subset["signature_id"].nunique() == n_signatures,
        f"{dataset}: signature count mismatch",
    )
    counts = subset.groupby("method").size().to_dict()
    for method, count in counts.items():
        expected = n_signatures * (5 if method.startswith("Transformer") else 1)
        check(count == expected, f"{dataset}/{method}: {count} != {expected}")

for (dataset, method), group in validation.groupby(["dataset", "method"]):
    selected = group[group["selected_on_validation"]]
    check(len(selected) == 1, f"{dataset}/{method}: expected one selected row")
    if len(selected) == 1:
        check(
            np.isclose(selected.iloc[0]["mean_f1"], group["mean_f1"].max()),
            f"{dataset}/{method}: selected row is not maximum validation F1",
        )

baseline_delta = (
    comparison["recomputed_shared_f1"] - comparison["canonical_shared_f1"]
).abs()
check(
    baseline_delta.max() <= 5.1e-5,
    f"shared-default reproduction delta too large: {baseline_delta.max()}",
)

expected_orders = {
    "firefox-android": [
        "SMA(w=40)",
        "EWMA(a=0.05)",
        "Transformer(w=20,d=16)",
        "ARIMA(1,1,1)",
        "LAST",
    ],
    "mozilla-beta": [
        "SMA(w=20)",
        "EWMA(a=0.05)",
        "Transformer(w=10,d=32)",
        "LAST",
        "ARIMA(1,1,1)",
    ],
}
for dataset, expected in expected_orders.items():
    observed = summary["ranking_stability"][dataset][
        "detector_specific_tuned_order"
    ]
    check(observed == expected, f"{dataset}: tuned order mismatch")

if errors:
    print("GATE-TUNING CHECK: FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("GATE-TUNING CHECK: PASSED")
print("  validation grid rows: 350")
print("  test grid rows: 350")
print("  selected test detail rows: 16,371")
print(f"  max canonical reproduction delta: {baseline_delta.max():.8f}")
