#!/usr/bin/env python3
"""Validate outputs of transformer_statistical_comparison.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Experiments" / "results" / "transformer_statistics"
errors = []


def check(condition, message):
    if not condition:
        errors.append(message)


comparisons = pd.read_csv(OUT / "paired_comparisons.csv")
methods = pd.read_csv(OUT / "method_bootstrap_ci.csv")
seeds = pd.read_csv(OUT / "transformer_seed_sensitivity.csv")
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))

check(len(comparisons) == 32, f"paired comparison rows: {len(comparisons)} != 32")
check(len(methods) == 48, f"method CI rows: {len(methods)} != 48")
check(len(seeds) == 40, f"seed sensitivity rows: {len(seeds)} != 40")
check(len(summary.get("f1_comparisons", [])) == 8, "summary must contain 8 F1 rows")
check(set(comparisons["n_pairs"]) == {342, 1477}, "unexpected paired sample sizes")
check(
    comparisons["p_value_holm"].between(0, 1).all(),
    "adjusted p-values outside [0,1]",
)
check(
    (comparisons["bootstrap_ci95_lower"] <= comparisons["mean_paired_difference"]).all()
    and (
        comparisons["mean_paired_difference"]
        <= comparisons["bootstrap_ci95_upper"]
    ).all(),
    "a paired mean lies outside its confidence interval",
)

# The canonical aggregate means must reproduce the frozen main result file.
canonical = json.loads(
    (ROOT / "results" / "combined_summary.json").read_text(encoding="utf-8")
)
for dataset, dataset_summary in canonical["datasets"].items():
    for method, record in dataset_summary["methods"].items():
        if not (method.startswith("Transformer") or method.startswith("SMA") or method.startswith("EWMA")):
            continue
        for metric, expected in {
            "precision": record["mean_precision"],
            "recall": record["mean_recall"],
            "f1": record["mean_f1"],
            "far": record["mean_far"],
        }.items():
            row = methods[
                (methods["protocol"] == "canonical_shared_gate")
                & (methods["dataset"] == dataset)
                & (methods["method"] == method)
                & (methods["metric"] == metric)
            ]
            check(len(row) == 1, f"missing method summary: {dataset}/{method}/{metric}")
            if len(row) == 1:
                check(
                    np.isclose(row.iloc[0]["mean"], expected, atol=5.1e-5),
                    f"canonical mean mismatch: {dataset}/{method}/{metric}",
                )

if errors:
    print("TRANSFORMER-STATISTICS CHECK: FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("TRANSFORMER-STATISTICS CHECK: PASSED")
print("  paired comparisons: 32")
print("  F1 planned contrasts: 8")
print("  method bootstrap summaries: 48")
print("  seed-sensitivity rows: 40")
