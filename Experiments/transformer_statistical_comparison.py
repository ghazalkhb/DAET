#!/usr/bin/env python3
"""Paired Transformer comparisons against the strongest classical methods.

The signature is the unit of analysis. Transformer results are averaged over
the five seeds within each signature before any test, preventing seed-level
pseudoreplication. Tests are two-sided paired Wilcoxon signed-rank tests.
Holm correction is applied within each protocol/dataset family across the two
planned comparators (SMA and EWMA) and four metrics (P/R/F1/FAR).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Experiments" / "results" / "transformer_statistics"
BOOTSTRAP_REPS = 10_000
BASE_SEED = 42
ALPHA = 0.05

PROTOCOLS = {
    "canonical_shared_gate": ROOT / "results" / "combined_detail.csv",
    "detector_specific_gate": (
        ROOT
        / "Experiments"
        / "results"
        / "gate_tuning"
        / "selected_test_detail.csv"
    ),
}

DATASET_METHODS = {
    "firefox-android": {
        "Transformer": "Transformer(w=20,d=16)",
        "SMA": "SMA(w=40)",
        "EWMA": "EWMA(a=0.05)",
    },
    "mozilla-beta": {
        "Transformer": "Transformer(w=10,d=32)",
        "SMA": "SMA(w=20)",
        "EWMA": "EWMA(a=0.05)",
    },
}

METRICS = {
    "precision": "test_precision",
    "recall": "test_recall",
    "f1": "test_f1",
    "far": "test_far",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_seed(key: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def bootstrap_mean_ci(values: np.ndarray, key: str):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(comparison_seed(key))
    means = []
    batch_size = 500
    remaining = BOOTSTRAP_REPS
    while remaining:
        current = min(batch_size, remaining)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= current
    distribution = np.concatenate(means)
    return (
        float(values.mean()),
        float(np.percentile(distribution, 2.5)),
        float(np.percentile(distribution, 97.5)),
    )


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_wilcoxon(differences: np.ndarray):
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0
    result = wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue), int(len(nonzero))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_maximum = 0.0
    m = len(p_values)
    for position, original_index in enumerate(order):
        adjusted = min(1.0, (m - position) * p_values[original_index])
        running_maximum = max(running_maximum, adjusted)
        adjusted_sorted[position] = running_maximum
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def prepare_protocol(path: Path):
    frame = pd.read_csv(path)
    required = {"dataset", "method", "signature_id", *METRICS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    frame["signature_id"] = frame["signature_id"].astype(str)
    return frame


def method_signature_table(frame, dataset, method_label):
    subset = frame[
        (frame["dataset"] == dataset) & (frame["method"] == method_label)
    ].copy()
    if subset.empty:
        raise RuntimeError(f"missing {dataset}/{method_label}")
    aggregation = {column: "mean" for column in METRICS.values()}
    return subset.groupby("signature_id", as_index=False).agg(aggregation)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    method_rows = []
    seed_rows = []

    for protocol, input_path in PROTOCOLS.items():
        if not input_path.is_file():
            raise RuntimeError(f"missing protocol input: {input_path}")
        frame = prepare_protocol(input_path)
        for dataset, labels in DATASET_METHODS.items():
            tables = {
                name: method_signature_table(frame, dataset, label)
                for name, label in labels.items()
            }
            expected_n = 342 if dataset == "firefox-android" else 1477
            for name, table in tables.items():
                if len(table) != expected_n:
                    raise RuntimeError(
                        f"{protocol}/{dataset}/{name}: expected {expected_n} "
                        f"signatures, found {len(table)}"
                    )
                for metric, column in METRICS.items():
                    mean, lower, upper = bootstrap_mean_ci(
                        table[column].to_numpy(float),
                        f"method:{protocol}:{dataset}:{name}:{metric}",
                    )
                    method_rows.append(
                        {
                            "protocol": protocol,
                            "dataset": dataset,
                            "method": labels[name],
                            "method_family": name,
                            "metric": metric,
                            "n_signatures": len(table),
                            "mean": mean,
                            "bootstrap_ci95_lower": lower,
                            "bootstrap_ci95_upper": upper,
                            "bootstrap_reps": BOOTSTRAP_REPS,
                        }
                    )

            transformer = tables["Transformer"].set_index("signature_id")
            family_start = len(comparison_rows)
            for comparator_name in ("SMA", "EWMA"):
                comparator = tables[comparator_name].set_index("signature_id")
                common = transformer.index.intersection(comparator.index)
                if len(common) != expected_n:
                    raise RuntimeError(
                        f"{protocol}/{dataset}/{comparator_name}: paired n "
                        f"{len(common)} != {expected_n}"
                    )
                for metric, column in METRICS.items():
                    transformer_values = transformer.loc[common, column].to_numpy(float)
                    comparator_values = comparator.loc[common, column].to_numpy(float)
                    differences = transformer_values - comparator_values
                    mean_diff, lower, upper = bootstrap_mean_ci(
                        differences,
                        f"difference:{protocol}:{dataset}:{comparator_name}:{metric}",
                    )
                    statistic, p_value, n_nonzero = paired_wilcoxon(differences)
                    comparison_rows.append(
                        {
                            "protocol": protocol,
                            "dataset": dataset,
                            "contrast": f"Transformer - {comparator_name}",
                            "comparator": labels[comparator_name],
                            "metric": metric,
                            "n_pairs": len(common),
                            "n_nonzero_pairs": n_nonzero,
                            "transformer_mean": float(transformer_values.mean()),
                            "classical_mean": float(comparator_values.mean()),
                            "mean_paired_difference": mean_diff,
                            "median_paired_difference": float(np.median(differences)),
                            "bootstrap_ci95_lower": lower,
                            "bootstrap_ci95_upper": upper,
                            "bootstrap_reps": BOOTSTRAP_REPS,
                            "wilcoxon_statistic": statistic,
                            "p_value_raw": p_value,
                            "rank_biserial": rank_biserial(differences),
                            "positive_difference_pct": float(
                                np.mean(differences > 0) * 100.0
                            ),
                            "zero_difference_pct": float(
                                np.mean(differences == 0) * 100.0
                            ),
                            "difference_definition": "Transformer minus classical",
                            "better_direction": (
                                "negative" if metric == "far" else "positive"
                            ),
                        }
                    )

            family_end = len(comparison_rows)
            adjusted = holm_adjust(
                [
                    row["p_value_raw"]
                    for row in comparison_rows[family_start:family_end]
                ]
            )
            for row, adjusted_p in zip(
                comparison_rows[family_start:family_end], adjusted
            ):
                row["p_value_holm"] = float(adjusted_p)
                row["significant_holm_0_05"] = bool(adjusted_p < ALPHA)

            # Seed-level F1 sensitivity is descriptive and is not used as an
            # independent-sample significance analysis.
            transformer_raw = frame[
                (frame["dataset"] == dataset)
                & (frame["method"] == labels["Transformer"])
            ].copy()
            if "seed" in transformer_raw.columns:
                for seed, seed_frame in transformer_raw.groupby("seed"):
                    seed_table = seed_frame.groupby("signature_id", as_index=False)[
                        "test_f1"
                    ].mean()
                    seed_values = seed_table.set_index("signature_id").loc[
                        transformer.index, "test_f1"
                    ].to_numpy(float)
                    for comparator_name in ("SMA", "EWMA"):
                        comparator_values = tables[comparator_name].set_index(
                            "signature_id"
                        ).loc[transformer.index, "test_f1"].to_numpy(float)
                        differences = seed_values - comparator_values
                        seed_rows.append(
                            {
                                "protocol": protocol,
                                "dataset": dataset,
                                "seed": int(float(seed)),
                                "comparator": labels[comparator_name],
                                "mean_transformer_f1": float(seed_values.mean()),
                                "mean_classical_f1": float(comparator_values.mean()),
                                "mean_paired_difference": float(differences.mean()),
                                "fraction_transformer_higher": float(
                                    np.mean(differences > 0)
                                ),
                            }
                        )

    comparisons = pd.DataFrame(comparison_rows)
    methods = pd.DataFrame(method_rows)
    seeds = pd.DataFrame(seed_rows)
    comparisons.to_csv(OUT / "paired_comparisons.csv", index=False)
    methods.to_csv(OUT / "method_bootstrap_ci.csv", index=False)
    seeds.to_csv(OUT / "transformer_seed_sensitivity.csv", index=False)

    f1_rows = comparisons[comparisons["metric"] == "f1"].copy()
    summary = {
        "analysis": "paired Transformer comparisons with SMA and EWMA",
        "unit_of_analysis": "signature",
        "transformer_handling": (
            "five seeds averaged within signature before inferential testing"
        ),
        "test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity": (
            "Holm correction within each protocol/dataset across 8 planned "
            "tests (2 comparators x 4 metrics)"
        ),
        "bootstrap": {
            "type": "paired signature resampling of Transformer-classical difference",
            "repetitions": BOOTSTRAP_REPS,
            "base_seed": BASE_SEED,
        },
        "f1_comparisons": f1_rows.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in PROTOCOLS.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_version,
        },
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    plot_f1_differences(f1_rows)
    print_f1_summary(f1_rows)


def plot_f1_differences(f1_rows):
    rows = f1_rows.reset_index(drop=True)
    y = np.arange(len(rows))
    means = rows["mean_paired_difference"].to_numpy(float)
    lower = rows["bootstrap_ci95_lower"].to_numpy(float)
    upper = rows["bootstrap_ci95_upper"].to_numpy(float)
    labels = [
        f"{row.protocol} | {row.dataset} | {row.contrast}"
        for row in rows.itertuples()
    ]
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        capsize=4,
        color="#355C7D",
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Mean paired F1 difference (Transformer − classical)")
    axis.set_title("Transformer F1 comparisons with 95% paired-bootstrap CIs")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "f1_paired_difference_forest.png", dpi=180)
    plt.close(fig)


def print_f1_summary(f1_rows):
    print("TRANSFORMER STATISTICAL COMPARISON: COMPLETE")
    for row in f1_rows.itertuples():
        print(
            f"  {row.protocol}/{row.dataset}/{row.contrast}: "
            f"delta={row.mean_paired_difference:.4f}, "
            f"CI=[{row.bootstrap_ci95_lower:.4f}, "
            f"{row.bootstrap_ci95_upper:.4f}], "
            f"p_holm={row.p_value_holm:.6g}"
        )


if __name__ == "__main__":
    main()
