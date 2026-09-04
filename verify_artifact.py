#!/usr/bin/env python3
"""Verify that frozen outputs match the current manuscript's canonical run.

This check uses only the Python standard library. It validates artifact
identity and internal consistency; it does not rerun the detectors.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

EXPECTED = {
    "firefox-android": {
        "signatures": 342,
        "physical_rows": 3078,
        "telemetry_points": 214063,
        "mapped_alerts": 796,
        "test_alerts": 144,
        "conceptual_runs": 1710,
        "params": {
            "sma_window": 40,
            "ewma_alpha": 0.05,
            "transformer_window": 20,
            "transformer_dmodel": 16,
        },
        "methods": {
            "ARIMA(1,1,1)": (0.0560, 0.2573, 0.0893, 0.056253),
            "LAST": (0.0455, 0.2091, 0.0718, 0.054508),
            "SMA(w=40)": (0.0661, 0.2895, 0.1039, 0.054695),
            "EWMA(a=0.05)": (0.0676, 0.2632, 0.1033, 0.052155),
            "Transformer(w=20,d=16)": (0.0688, 0.2924, 0.1051, 0.065694),
        },
    },
    "mozilla-beta": {
        "signatures": 1477,
        "physical_rows": 13293,
        "telemetry_points": 1183738,
        "mapped_alerts": 3597,
        "test_alerts": 482,
        "conceptual_runs": 7385,
        "params": {
            "sma_window": 20,
            "ewma_alpha": 0.05,
            "transformer_window": 10,
            "transformer_dmodel": 32,
        },
        "methods": {
            "ARIMA(1,1,1)": (0.0503, 0.2317, 0.0761, 0.060171),
            "LAST": (0.0454, 0.2115, 0.0685, 0.053469),
            "SMA(w=20)": (0.0574, 0.2408, 0.0847, 0.059089),
            "EWMA(a=0.05)": (0.0570, 0.2363, 0.0830, 0.059159),
            "Transformer(w=10,d=32)": (0.0539, 0.2348, 0.0789, 0.064139),
        },
    },
}

EXPECTED_SEEDS = {"42", "123", "456", "789", "1024"}
REQUIRED_COLUMNS = {
    "dataset",
    "method",
    "signature_id",
    "T_total",
    "T_train",
    "T_val",
    "T_test",
    "seed",
    "test_n_alerts",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_far",
    "test_storage_reduc",
}

errors: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def load_json(relative: str):
    path = ROOT / relative
    if not path.is_file():
        fail(f"missing required file: {relative}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {relative}: {exc}")
        return {}


def normalized_seed(value: str) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return value.strip()


summary = load_json("results/combined_summary.json")
scale = load_json("Experiments/results/data_scale_audit.json")

detail_path = RESULTS / "combined_detail.csv"
rows: list[dict[str, str]] = []
if not detail_path.is_file():
    fail("missing required file: results/combined_detail.csv")
else:
    with detail_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            fail(f"combined_detail.csv missing columns: {sorted(missing_columns)}")
        rows = list(reader)

rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    rows_by_dataset[row.get("dataset", "")].append(row)

for dataset, expected in EXPECTED.items():
    dataset_rows = rows_by_dataset.get(dataset, [])
    if len(dataset_rows) != expected["physical_rows"]:
        fail(
            f"{dataset}: expected {expected['physical_rows']} detail rows, "
            f"found {len(dataset_rows)}"
        )

    signatures = {row.get("signature_id", "") for row in dataset_rows}
    if len(signatures) != expected["signatures"]:
        fail(
            f"{dataset}: expected {expected['signatures']} signatures, "
            f"found {len(signatures)}"
        )

    method_counts = Counter(row.get("method", "") for row in dataset_rows)
    expected_methods = set(expected["methods"])
    if set(method_counts) != expected_methods:
        fail(
            f"{dataset}: method set mismatch; expected {sorted(expected_methods)}, "
            f"found {sorted(method_counts)}"
        )

    transformer_name = next(
        method for method in expected_methods if method.startswith("Transformer")
    )
    for method in expected_methods:
        multiplier = 5 if method == transformer_name else 1
        expected_count = expected["signatures"] * multiplier
        if method_counts[method] != expected_count:
            fail(
                f"{dataset}/{method}: expected {expected_count} rows, "
                f"found {method_counts[method]}"
            )

    transformer_rows = [
        row for row in dataset_rows if row.get("method") == transformer_name
    ]
    seeds_by_signature: dict[str, set[str]] = defaultdict(set)
    for row in transformer_rows:
        seeds_by_signature[row.get("signature_id", "")].add(
            normalized_seed(row.get("seed", ""))
        )
    bad_seed_signatures = [
        signature
        for signature, seeds in seeds_by_signature.items()
        if seeds != EXPECTED_SEEDS
    ]
    if bad_seed_signatures:
        fail(
            f"{dataset}: {len(bad_seed_signatures)} signatures do not have all "
            "five expected Transformer seeds"
        )

    deterministic_rows = [
        row for row in dataset_rows if not row.get("method", "").startswith("Transformer")
    ]
    # Each alert count is repeated once for each of the four deterministic methods.
    observed_test_alerts = sum(
        int(float(row.get("test_n_alerts", 0) or 0)) for row in deterministic_rows
    ) // 4
    if observed_test_alerts != expected["test_alerts"]:
        fail(
            f"{dataset}: expected {expected['test_alerts']} test alerts, "
            f"found {observed_test_alerts}"
        )

    dataset_summary = summary.get("datasets", {}).get(dataset, {})
    if dataset_summary.get("tuned_params") != expected["params"]:
        fail(
            f"{dataset}: tuned parameters differ from the manuscript-aligned set"
        )

    summary_methods = dataset_summary.get("methods", {})
    if set(summary_methods) != expected_methods:
        fail(f"{dataset}: summary method set does not match detail method set")
    for method, metric_values in expected["methods"].items():
        record = summary_methods.get(method, {})
        actual = tuple(
            record.get(key)
            for key in ("mean_precision", "mean_recall", "mean_f1", "mean_far")
        )
        if actual != metric_values:
            fail(
                f"{dataset}/{method}: expected P/R/F1/FAR {metric_values}, "
                f"found {actual}"
            )

    dataset_detail_path = RESULTS / dataset / "detail.csv"
    if not dataset_detail_path.is_file():
        fail(f"missing dataset-specific detail: results/{dataset}/detail.csv")
    else:
        with dataset_detail_path.open(newline="", encoding="utf-8-sig") as handle:
            dataset_detail_rows = sum(1 for _ in csv.DictReader(handle))
        if dataset_detail_rows != expected["physical_rows"]:
            fail(
                f"results/{dataset}/detail.csv: expected "
                f"{expected['physical_rows']} rows, found {dataset_detail_rows}"
            )

scale_by_dataset = {
    item.get("dataset"): item for item in scale.get("datasets", [])
}
for dataset, expected in EXPECTED.items():
    item = scale_by_dataset.get(dataset, {})
    checks = {
        "n_valid_split": expected["signatures"],
        "total_telemetry_points": expected["telemetry_points"],
    }
    for key, wanted in checks.items():
        if item.get(key) != wanted:
            fail(f"scale audit {dataset}/{key}: expected {wanted}, found {item.get(key)}")
    if item.get("alerts", {}).get("total_mapped_alerts") != expected["mapped_alerts"]:
        fail(f"scale audit {dataset}: mapped-alert count mismatch")
    if (
        item.get("detector_sig_pairs", {}).get("total_pairs_canonical")
        != expected["conceptual_runs"]
    ):
        fail(f"scale audit {dataset}: conceptual run count mismatch")

combined = scale.get("combined", {})
combined_expected = {
    "total_valid_sigs": 1819,
    "total_telemetry_points": 1397801,
    "total_alerts": 4393,
    "total_det_sig_pairs": 9095,
    "n_datasets": 2,
}
for key, wanted in combined_expected.items():
    if combined.get(key) != wanted:
        fail(f"combined scale/{key}: expected {wanted}, found {combined.get(key)}")

required_files = [
    "results/firefox-android/summary.json",
    "results/mozilla-beta/summary.json",
    "Experiments/results/transformer_efficiency_summary.json",
    "results/replay/replay_summary.json",
    "Data/alerts_data.csv",
    "Data/bugs_data.csv",
    "requirements.txt",
    "ARTIFACT_MANIFEST.md",
    "results/README.md",
]
for relative in required_files:
    if not (ROOT / relative).is_file():
        fail(f"missing required artifact file: {relative}")

requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
for pin in (
    "numpy==1.24.4",
    "pandas==2.0.3",
    "scipy==1.11.4",
    "statsmodels==0.13.5",
    "torch==2.4.1+cpu",
):
    if pin not in requirements:
        fail(f"requirements.txt missing manuscript dependency pin: {pin}")

unexpected_datasets = set(rows_by_dataset) - set(EXPECTED)
if unexpected_datasets:
    fail(f"unexpected datasets in combined_detail.csv: {sorted(unexpected_datasets)}")

if errors:
    print("ARTIFACT CHECK: FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("ARTIFACT CHECK: PASSED")
print("  canonical datasets: 2")
print("  signatures: 1,819")
print("  telemetry points: 1,397,801")
print("  conceptual detector-signature runs: 9,095")
print("  physical rows including five Transformer seeds: 16,371")
if not (ROOT / "Data" / "timeseries-data").is_dir():
    print("  info: raw Data/timeseries-data is not included; frozen outputs verified")
