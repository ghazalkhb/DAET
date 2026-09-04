#!/usr/bin/env python3
"""Detector-specific beta/theta tuning under the canonical 60/20/20 protocol.

This experiment addresses comparison fairness without touching the canonical
outputs. Detector/model hyperparameters remain fixed at their validation-
selected manuscript values. For each detector and dataset, beta and theta are
selected only by mean validation F1; the selected pair is then evaluated once
on the held-out test segment. Transformer selection averages validation results
over the same five seeds used by the canonical evaluation.

The detector residual stream and dynamic-threshold anomaly indicators are
independent of beta/theta, so they are generated once. The gate grid is then
evaluated as inexpensive post-processing. Checkpoints make the long Transformer
stage resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import kendalltau, spearmanr
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_full_evaluation as canonical  # noqa: E402


BETA_GRID = [0.10, 0.20, 0.30, 0.50, 0.70]
THETA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
GRID = [(beta, theta) for beta in BETA_GRID for theta in THETA_GRID]
SHARED_BETA = 0.30
SHARED_THETA = 0.30

DATASETS = {
    "firefox-android": {
        "repository": "firefox-android",
        "n_files": 342,
    },
    "mozilla-beta": {
        "repository": "mozilla-beta",
        "n_files": 1477,
    },
}
METHOD_ORDER = ["ARIMA", "LAST", "SMA", "EWMA", "Transformer"]
SEEDS_DEFAULT = [42, 123, 456, 789, 1024]
METRICS = [
    "n_alerts",
    "has_alert",
    "detected",
    "precision",
    "recall",
    "f1",
    "far",
    "storage_reduc",
    "n_on",
]
MIDX = {name: index for index, name in enumerate(METRICS)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=METHOD_ORDER,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Experiments" / "results" / "gate_tuning",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only signature limit; never use limited output in the paper.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints.")
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def detector_parameters(dataset: str, canonical_summary: dict) -> dict:
    tuned = canonical_summary["datasets"][dataset]["tuned_params"]
    return {
        "ARIMA": {"order": [1, 1, 1]},
        "LAST": {},
        "SMA": {"window": int(tuned["sma_window"])},
        "EWMA": {"alpha": float(tuned["ewma_alpha"])},
        "Transformer": {
            "window": int(tuned["transformer_window"]),
            "d_model": int(tuned["transformer_dmodel"]),
        },
    }


def method_label(method: str, params: dict) -> str:
    if method == "ARIMA":
        return "ARIMA(1,1,1)"
    if method == "SMA":
        return f"SMA(w={params['window']})"
    if method == "EWMA":
        return f"EWMA(a={params['alpha']})"
    if method == "Transformer":
        return f"Transformer(w={params['window']},d={params['d_model']})"
    return method


def split_data(dataset: str, alerts: pd.DataFrame, limit: int | None):
    spec = DATASETS[dataset]
    ts_dir = ROOT / "Data" / "timeseries-data" / dataset
    paths = sorted(ts_dir.glob("*_timeseries_data.csv"))
    if len(paths) != spec["n_files"]:
        raise RuntimeError(
            f"{dataset}: expected {spec['n_files']} source CSVs, found {len(paths)}"
        )
    loaded = canonical.load_timeseries_for_dataset(
        str(ts_dir), alerts, spec["repository"]
    )
    if len(loaded) != spec["n_files"]:
        raise RuntimeError(
            f"{dataset}: expected {spec['n_files']} usable signatures, "
            f"loaded {len(loaded)}"
        )
    if limit is not None:
        loaded = loaded[:limit]

    rows = []
    for signature, values, alert_indices in loaded:
        total = len(values)
        n_train = max(
            canonical.ARIMA_MIN_HISTORY + 2,
            int(round(canonical.TRAIN_RATIO * total)),
        )
        n_val = max(2, int(round(canonical.VAL_RATIO * total)))
        n_test = total - n_train - n_val
        if n_test < 2:
            continue
        rows.append(
            {
                "signature_id": signature,
                "values": values,
                "alerts": alert_indices,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
            }
        )
    return rows


def training_flags_classical(values, method, parameter, arima_fit=None):
    flags = np.zeros(len(values), dtype=np.uint8)
    delta_history = []
    ewma = float(values[0]) if len(values) else 0.0
    sma_window = parameter if method == "sma" else 20
    alpha = parameter if method == "ewma" else 0.30
    minimum_history = (
        sma_window
        if method == "sma"
        else (canonical.ARIMA_MIN_HISTORY if method == "arima" else 2)
    )

    for index in range(len(values)):
        actual = canonical.safe_float(values[index])
        if index < minimum_history or actual is None:
            if method == "ewma" and index > 0 and actual is not None:
                ewma = alpha * actual + (1.0 - alpha) * ewma
            continue
        if method == "arima":
            if arima_fit is not None and index < len(arima_fit.fittedvalues):
                fitted = arima_fit.fittedvalues
                predicted = canonical.safe_float(
                    fitted.iloc[index] if hasattr(fitted, "iloc") else fitted[index]
                )
            else:
                predicted = canonical.safe_float(values[index - 1])
        elif method == "last":
            predicted = canonical.safe_float(values[index - 1])
        elif method == "sma":
            predicted = canonical.safe_float(
                np.mean(values[max(0, index - sma_window):index])
            )
        elif method == "ewma":
            predicted = ewma
            ewma = alpha * actual + (1.0 - alpha) * ewma
        else:
            raise ValueError(method)
        if predicted is None:
            continue
        delta = abs(actual - predicted)
        flags[index] = int(delta > canonical.dyn_thr(delta_history))
        delta_history.append(delta)
    return flags, delta_history, ewma


def segment_flags_classical(
    history,
    segment,
    method,
    parameter,
    initial_delta_history,
    initial_ewma,
    arima_predictions=None,
):
    flags = np.zeros(len(segment), dtype=np.uint8)
    delta_history = list(initial_delta_history)
    ewma = initial_ewma
    sma_window = parameter if method == "sma" else 20
    alpha = parameter if method == "ewma" else 0.30
    previous = float(history[-1]) if len(history) else float(segment[0])

    for index, value in enumerate(segment):
        actual = canonical.safe_float(value)
        if actual is None:
            continue
        if method == "arima":
            predicted = (
                canonical.safe_float(arima_predictions[index])
                if arima_predictions is not None
                and index < len(arima_predictions)
                else canonical.safe_float(segment[index - 1] if index else previous)
            )
        elif method == "last":
            predicted = canonical.safe_float(segment[index - 1] if index else previous)
        elif method == "sma":
            buffer = np.concatenate([history, segment[:index]])[-sma_window:]
            predicted = canonical.safe_float(np.mean(buffer)) if len(buffer) else None
        elif method == "ewma":
            predicted = ewma
            ewma = alpha * actual + (1.0 - alpha) * ewma
        else:
            raise ValueError(method)
        if predicted is None:
            continue
        delta = abs(actual - predicted)
        flags[index] = int(delta > canonical.dyn_thr(delta_history))
        delta_history.append(delta)
    return flags, delta_history, ewma


def training_flags_transformer(values, model_tuple):
    flags = np.zeros(len(values), dtype=np.uint8)
    delta_history = []
    if model_tuple is None:
        return flags, delta_history
    model, mean, std, window = model_tuple
    normalized = (np.asarray(values, dtype=np.float32) - mean) / max(std, 1e-8)
    model.eval()
    with canonical.torch.no_grad():
        for index in range(window, len(values)):
            buffer = canonical.torch.tensor(
                normalized[index - window:index], dtype=canonical.torch.float32
            ).unsqueeze(0).unsqueeze(-1)
            predicted = model(buffer).item() * max(std, 1e-8) + mean
            actual = canonical.safe_float(values[index])
            if actual is None:
                continue
            delta = abs(actual - predicted)
            flags[index] = int(delta > canonical.dyn_thr(delta_history))
            delta_history.append(delta)
    return flags, delta_history


def segment_flags_transformer(
    history, segment, model_tuple, initial_delta_history
):
    flags = np.zeros(len(segment), dtype=np.uint8)
    delta_history = list(initial_delta_history)
    if model_tuple is None:
        return flags, delta_history
    model, mean, std, window = model_tuple
    combined = np.concatenate([history, segment])
    offset = len(history)
    normalized = (combined.astype(np.float32) - mean) / max(std, 1e-8)
    model.eval()
    with canonical.torch.no_grad():
        for index, value in enumerate(segment):
            absolute_index = offset + index
            actual = canonical.safe_float(value)
            if actual is None or absolute_index < window:
                continue
            buffer = canonical.torch.tensor(
                normalized[absolute_index - window:absolute_index],
                dtype=canonical.torch.float32,
            ).unsqueeze(0).unsqueeze(-1)
            predicted = model(buffer).item() * max(std, 1e-8) + mean
            delta = abs(actual - predicted)
            flags[index] = int(delta > canonical.dyn_thr(delta_history))
            delta_history.append(delta)
    return flags, delta_history


def metric_vector(decisions, offset: int, alert_indices) -> np.ndarray:
    result = canonical.seg_metrics(decisions.tolist(), offset, alert_indices)
    return np.asarray(
        [
            result["n_alerts"],
            float(result["has_alert"]),
            float(result["detected"]),
            result["precision"],
            result["recall"],
            result["f1"],
            result["far"],
            result["storage_reduc"],
            result["n_on"],
        ],
        dtype=np.float64,
    )


def grid_metrics(train_flags, val_flags, test_flags, n_train, alerts):
    full_flags = np.concatenate([train_flags, val_flags, test_flags]).astype(float)
    n_val = len(val_flags)
    val_results = np.zeros((len(GRID), len(METRICS)), dtype=np.float64)
    test_results = np.zeros_like(val_results)
    grid_index = 0
    for beta in BETA_GRID:
        scores = lfilter([beta], [1.0, -(1.0 - beta)], full_flags)
        val_scores = scores[n_train:n_train + n_val]
        test_scores = scores[n_train + n_val:]
        for theta in THETA_GRID:
            val_results[grid_index] = metric_vector(
                val_scores >= theta, n_train, alerts
            )
            test_results[grid_index] = metric_vector(
                test_scores >= theta, n_train + n_val, alerts
            )
            grid_index += 1
    return val_results, test_results


def checkpoint_path(cache_dir: Path, dataset: str, method: str, seed) -> Path:
    suffix = f"seed{seed}" if seed is not None else "deterministic"
    return cache_dir / f"{dataset}__{method.lower()}__{suffix}.npz"


def save_checkpoint(path: Path, metadata: list[dict], validation, test):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature_id=np.asarray([item["signature_id"] for item in metadata]),
        seed=np.asarray([item["seed"] for item in metadata], dtype=np.int64),
        total=np.asarray([item["total"] for item in metadata], dtype=np.int64),
        n_train=np.asarray([item["n_train"] for item in metadata], dtype=np.int64),
        n_val=np.asarray([item["n_val"] for item in metadata], dtype=np.int64),
        n_test=np.asarray([item["n_test"] for item in metadata], dtype=np.int64),
        validation=validation,
        test=test,
        grid=np.asarray(GRID, dtype=np.float64),
    )


def load_checkpoint(path: Path):
    loaded = np.load(path, allow_pickle=False)
    if not np.array_equal(loaded["grid"], np.asarray(GRID, dtype=np.float64)):
        raise RuntimeError(f"checkpoint grid mismatch: {path}")
    metadata = []
    for index, signature in enumerate(loaded["signature_id"]):
        metadata.append(
            {
                "signature_id": str(signature),
                "seed": int(loaded["seed"][index]),
                "total": int(loaded["total"][index]),
                "n_train": int(loaded["n_train"][index]),
                "n_val": int(loaded["n_val"][index]),
                "n_test": int(loaded["n_test"][index]),
            }
        )
    return metadata, loaded["validation"], loaded["test"]


def process_classical(dataset, method, params, series, cache_dir, force):
    path = checkpoint_path(cache_dir, dataset, method, None)
    if path.exists() and not force:
        print(f"  checkpoint: {path.name}", flush=True)
        return load_checkpoint(path)

    canonical_name = {
        "ARIMA": "arima",
        "LAST": "last",
        "SMA": "sma",
        "EWMA": "ewma",
    }[method]
    parameter = (
        params.get("window")
        if method == "SMA"
        else params.get("alpha", 0.30)
    )
    metadata = []
    validation_rows = []
    test_rows = []
    started = time.time()

    for position, item in enumerate(series, start=1):
        values = item["values"]
        n_train, n_val = item["n_train"], item["n_val"]
        train = values[:n_train]
        validation = values[n_train:n_train + n_val]
        test = values[n_train + n_val:]
        arima_fit = None
        if method == "ARIMA":
            try:
                arima_fit = StatsARIMA(
                    train,
                    order=canonical.ARIMA_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            except Exception:
                arima_fit = None

        train_flags, delta0, ewma0 = training_flags_classical(
            train, canonical_name, parameter, arima_fit
        )
        predictions_val = None
        predictions_test = None
        if method == "ARIMA" and arima_fit is not None:
            predictions_val = canonical.arima_rolling_preds(
                arima_fit, train, validation
            )
            predictions_test = canonical.arima_rolling_preds(
                arima_fit, np.concatenate([train, validation]), test
            )
        val_flags, delta1, ewma1 = segment_flags_classical(
            train,
            validation,
            canonical_name,
            parameter,
            delta0,
            ewma0,
            predictions_val,
        )
        test_flags, _, _ = segment_flags_classical(
            np.concatenate([train, validation]),
            test,
            canonical_name,
            parameter,
            delta1,
            ewma1,
            predictions_test,
        )
        val_metrics, test_metrics = grid_metrics(
            train_flags,
            val_flags,
            test_flags,
            n_train,
            item["alerts"],
        )
        validation_rows.append(val_metrics)
        test_rows.append(test_metrics)
        metadata.append(
            {
                "signature_id": item["signature_id"],
                "seed": -1,
                "total": len(values),
                "n_train": n_train,
                "n_val": n_val,
                "n_test": item["n_test"],
            }
        )
        if position % 50 == 0 or position == len(series):
            elapsed = time.time() - started
            print(
                f"  {method}/{dataset}: {position}/{len(series)} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    validation_array = np.stack(validation_rows)
    test_array = np.stack(test_rows)
    save_checkpoint(path, metadata, validation_array, test_array)
    return metadata, validation_array, test_array


def process_transformer_seed(
    dataset, params, series, seed, cache_dir, force
):
    path = checkpoint_path(cache_dir, dataset, "Transformer", seed)
    if path.exists() and not force:
        print(f"  checkpoint: {path.name}", flush=True)
        return load_checkpoint(path)

    metadata = []
    validation_rows = []
    test_rows = []
    started = time.time()
    window = params["window"]
    d_model = params["d_model"]

    for position, item in enumerate(series, start=1):
        values = item["values"]
        n_train, n_val = item["n_train"], item["n_val"]
        train = values[:n_train]
        validation = values[n_train:n_train + n_val]
        test = values[n_train + n_val:]
        model_tuple = canonical.train_transformer(train, window, d_model, seed)
        train_flags, delta0 = training_flags_transformer(train, model_tuple)
        val_flags, delta1 = segment_flags_transformer(
            train, validation, model_tuple, delta0
        )
        test_flags, _ = segment_flags_transformer(
            np.concatenate([train, validation]), test, model_tuple, delta1
        )
        val_metrics, test_metrics = grid_metrics(
            train_flags,
            val_flags,
            test_flags,
            n_train,
            item["alerts"],
        )
        validation_rows.append(val_metrics)
        test_rows.append(test_metrics)
        metadata.append(
            {
                "signature_id": item["signature_id"],
                "seed": seed,
                "total": len(values),
                "n_train": n_train,
                "n_val": n_val,
                "n_test": item["n_test"],
            }
        )
        if position % 25 == 0 or position == len(series):
            elapsed = time.time() - started
            print(
                f"  Transformer/{dataset}/seed={seed}: "
                f"{position}/{len(series)} ({elapsed:.0f}s)",
                flush=True,
            )

    validation_array = np.stack(validation_rows)
    test_array = np.stack(test_rows)
    save_checkpoint(path, metadata, validation_array, test_array)
    return metadata, validation_array, test_array


def aggregate_metric_array(array, metadata, transformer: bool):
    """Aggregate [units, grid, metrics], averaging seeds within signature."""
    if not transformer:
        return array
    signatures = np.asarray([item["signature_id"] for item in metadata])
    unique_signatures = np.unique(signatures)
    averaged = np.zeros((len(unique_signatures), array.shape[1], array.shape[2]))
    for index, signature in enumerate(unique_signatures):
        selected = array[signatures == signature]
        averaged[index] = selected.mean(axis=0)
        # Alert counts and alert-presence do not vary by seed.
        averaged[index, :, MIDX["n_alerts"]] = selected[0, :, MIDX["n_alerts"]]
        averaged[index, :, MIDX["has_alert"]] = selected[0, :, MIDX["has_alert"]]
    return averaged


def aggregate_one_grid(array, grid_index: int) -> dict:
    selected = array[:, grid_index, :]
    alerted = selected[:, MIDX["has_alert"]] > 0.5
    return {
        "n_signatures": int(len(selected)),
        "n_with_alerts": int(alerted.sum()),
        "mean_precision": float(selected[:, MIDX["precision"]].mean()),
        "mean_recall": float(selected[:, MIDX["recall"]].mean()),
        "mean_f1": float(selected[:, MIDX["f1"]].mean()),
        "mean_far": float(selected[:, MIDX["far"]].mean()),
        "mean_storage_reduc": float(selected[:, MIDX["storage_reduc"]].mean()),
        "detection_rate_pct": (
            float(selected[alerted, MIDX["detected"]].mean() * 100.0)
            if alerted.any()
            else 0.0
        ),
    }


def select_grid(validation_array) -> int:
    means = validation_array[:, :, MIDX["f1"]].mean(axis=0)
    maximum = means.max()
    candidates = np.flatnonzero(np.isclose(means, maximum, atol=1e-12))
    return int(
        min(
            candidates,
            key=lambda index: (
                abs(GRID[index][0] - SHARED_BETA)
                + abs(GRID[index][1] - SHARED_THETA),
                GRID[index][0],
                GRID[index][1],
            ),
        )
    )


def plot_validation_heatmap(rows, dataset, method, output_dir):
    subset = pd.DataFrame(rows)
    matrix = subset.pivot(index="beta", columns="theta", values="mean_f1")
    fig, axis = plt.subplots(figsize=(7, 4.8))
    image = axis.imshow(matrix.values, aspect="auto", cmap="YlGn")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels([f"{value:.2f}" for value in matrix.columns])
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([f"{value:.2f}" for value in matrix.index])
    axis.set_xlabel("Anomaly-score threshold θ")
    axis.set_ylabel("Smoothing β")
    axis.set_title(f"Validation F1: {method} — {dataset}")
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            axis.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=axis, label="Mean validation F1")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"validation_heatmap_{dataset}_{method.lower()}.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if "Transformer" in args.methods and not canonical.TORCH_OK:
        raise SystemExit("PyTorch is required for the Transformer fairness run")
    if canonical.TORCH_OK:
        canonical.torch.set_num_threads(max(1, args.torch_threads))

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    alerts_path = ROOT / "Data" / "alerts_data.csv"
    summary_path = ROOT / "results" / "combined_summary.json"
    alerts = pd.read_csv(alerts_path)
    canonical_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print("Detector-specific beta/theta tuning", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Grid: {len(BETA_GRID)} x {len(THETA_GRID)} = {len(GRID)}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    if args.limit is not None:
        print(f"WARNING: development-only signature limit = {args.limit}", flush=True)

    validation_grid_rows = []
    test_grid_rows = []
    selected_detail_rows = []
    selection_records = {}
    started_all = time.time()

    for dataset in args.datasets:
        print(f"\nDATASET: {dataset}", flush=True)
        series = split_data(dataset, alerts, args.limit)
        params_by_method = detector_parameters(dataset, canonical_summary)
        selection_records[dataset] = {}

        for method in args.methods:
            params = params_by_method[method]
            label = method_label(method, params)
            print(f"\nMETHOD: {label}", flush=True)
            chunks = []
            if method == "Transformer":
                for seed in args.seeds:
                    chunks.append(
                        process_transformer_seed(
                            dataset,
                            params,
                            series,
                            seed,
                            cache_dir,
                            args.force,
                        )
                    )
            else:
                chunks.append(
                    process_classical(
                        dataset, method, params, series, cache_dir, args.force
                    )
                )

            metadata = [item for chunk in chunks for item in chunk[0]]
            validation_raw = np.concatenate([chunk[1] for chunk in chunks], axis=0)
            test_raw = np.concatenate([chunk[2] for chunk in chunks], axis=0)
            is_transformer = method == "Transformer"
            validation_aggregated = aggregate_metric_array(
                validation_raw, metadata, is_transformer
            )
            test_aggregated = aggregate_metric_array(test_raw, metadata, is_transformer)
            selected_index = select_grid(validation_aggregated)
            selected_beta, selected_theta = GRID[selected_index]
            shared_index = GRID.index((SHARED_BETA, SHARED_THETA))

            heatmap_rows = []
            for grid_index, (beta, theta) in enumerate(GRID):
                val_summary = aggregate_one_grid(validation_aggregated, grid_index)
                test_summary = aggregate_one_grid(test_aggregated, grid_index)
                val_row = {
                    "dataset": dataset,
                    "method": label,
                    "beta": beta,
                    "theta": theta,
                    "selected_on_validation": grid_index == selected_index,
                    **val_summary,
                }
                validation_grid_rows.append(val_row)
                heatmap_rows.append(val_row)
                test_grid_rows.append(
                    {
                        "dataset": dataset,
                        "method": label,
                        "beta": beta,
                        "theta": theta,
                        "selected_on_validation": grid_index == selected_index,
                        **test_summary,
                    }
                )
            plot_validation_heatmap(
                heatmap_rows, dataset, method, output_dir
            )

            selected_summary = aggregate_one_grid(test_aggregated, selected_index)
            shared_summary = aggregate_one_grid(test_aggregated, shared_index)
            selection_records[dataset][label] = {
                "detector_parameters_held_fixed": params,
                "selected_beta": selected_beta,
                "selected_theta": selected_theta,
                "validation": aggregate_one_grid(
                    validation_aggregated, selected_index
                ),
                "test_selected": selected_summary,
                "test_shared_default_recomputed": shared_summary,
            }
            print(
                f"  selected beta={selected_beta:.2f}, theta={selected_theta:.2f}; "
                f"val F1={selection_records[dataset][label]['validation']['mean_f1']:.4f}; "
                f"test F1={selected_summary['mean_f1']:.4f}",
                flush=True,
            )

            selected_values = test_raw[:, selected_index, :]
            for meta, values in zip(metadata, selected_values):
                row = {
                    "dataset": dataset,
                    "method": label,
                    "signature_id": meta["signature_id"],
                    "seed": None if meta["seed"] < 0 else meta["seed"],
                    "T_total": meta["total"],
                    "T_train": meta["n_train"],
                    "T_val": meta["n_val"],
                    "T_test": meta["n_test"],
                    "selected_beta": selected_beta,
                    "selected_theta": selected_theta,
                }
                row.update(
                    {
                        f"test_{metric}": values[index]
                        for index, metric in enumerate(METRICS)
                    }
                )
                selected_detail_rows.append(row)

    validation_df = pd.DataFrame(validation_grid_rows)
    test_grid_df = pd.DataFrame(test_grid_rows)
    detail_df = pd.DataFrame(selected_detail_rows)
    validation_df.to_csv(output_dir / "validation_grid.csv", index=False)
    test_grid_df.to_csv(output_dir / "test_grid_summary.csv", index=False)
    detail_df.to_csv(output_dir / "selected_test_detail.csv", index=False)

    comparison_rows = []
    stability = {}
    for dataset in args.datasets:
        canonical_methods = canonical_summary["datasets"][dataset]["methods"]
        labels = [
            method_label(method, detector_parameters(dataset, canonical_summary)[method])
            for method in args.methods
        ]
        shared_f1 = {
            label: canonical_methods[label]["mean_f1"] for label in labels
        }
        tuned_f1 = {
            label: selection_records[dataset][label]["test_selected"]["mean_f1"]
            for label in labels
        }
        shared_order = sorted(labels, key=lambda label: (-shared_f1[label], label))
        tuned_order = sorted(labels, key=lambda label: (-tuned_f1[label], label))
        shared_rank = {label: index + 1 for index, label in enumerate(shared_order)}
        tuned_rank = {label: index + 1 for index, label in enumerate(tuned_order)}
        rank_a = [shared_rank[label] for label in labels]
        rank_b = [tuned_rank[label] for label in labels]
        stability[dataset] = {
            "shared_default_order": shared_order,
            "detector_specific_tuned_order": tuned_order,
            "spearman_rho": float(spearmanr(rank_a, rank_b).statistic),
            "kendall_tau": float(kendalltau(rank_a, rank_b).statistic),
            "ranking_identical": shared_order == tuned_order,
        }
        for label in labels:
            record = selection_records[dataset][label]
            selected = record["test_selected"]
            canonical_record = canonical_methods[label]
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "method": label,
                    "shared_beta": SHARED_BETA,
                    "shared_theta": SHARED_THETA,
                    "canonical_shared_f1": canonical_record["mean_f1"],
                    "recomputed_shared_f1": record[
                        "test_shared_default_recomputed"
                    ]["mean_f1"],
                    "selected_beta": record["selected_beta"],
                    "selected_theta": record["selected_theta"],
                    "tuned_test_precision": selected["mean_precision"],
                    "tuned_test_recall": selected["mean_recall"],
                    "tuned_test_f1": selected["mean_f1"],
                    "tuned_test_far": selected["mean_far"],
                    "tuned_test_storage_reduc": selected["mean_storage_reduc"],
                    "tuned_test_detection_rate_pct": selected[
                        "detection_rate_pct"
                    ],
                    "delta_f1_vs_canonical_shared": (
                        selected["mean_f1"] - canonical_record["mean_f1"]
                    ),
                    "shared_rank": shared_rank[label],
                    "tuned_rank": tuned_rank[label],
                    "rank_change": shared_rank[label] - tuned_rank[label],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "ranking_comparison.csv", index=False)

    run_metadata = {
        "experiment": "detector-specific beta/theta fairness tuning",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "record": "https://zenodo.org/records/14927532",
            "doi": "10.5281/zenodo.14927532",
            "archive": "data.zip",
            "archive_md5": "a6f6d41f262dbb678cdb469d3d94aa13",
            "firefox_android_files": 342,
            "mozilla_beta_files": 1477,
        },
        "selection_rule": (
            "maximum mean per-signature validation F1; ties resolved by "
            "minimum L1 distance to beta=0.30, theta=0.30"
        ),
        "test_policy": "selected validation pair evaluated once on held-out test",
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "beta_grid": BETA_GRID,
        "theta_grid": THETA_GRID,
        "datasets": args.datasets,
        "methods": args.methods,
        "transformer_seeds": args.seeds,
        "signature_limit": args.limit,
        "alerts_sha256": sha256(alerts_path),
        "canonical_summary_sha256": sha256(summary_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "torch": canonical.torch.__version__ if canonical.TORCH_OK else None,
            "torch_threads": args.torch_threads,
        },
        "elapsed_seconds": time.time() - started_all,
        "elapsed_scope": (
            "current invocation; completed checkpoints may have been reused"
        ),
    }
    final_summary = {
        "protocol": run_metadata,
        "selected_results": selection_records,
        "ranking_stability": stability,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(final_summary), indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(json_safe(run_metadata), indent=2), encoding="utf-8"
    )
    print(f"\nSaved results to {output_dir}", flush=True)
    print(json.dumps(json_safe(stability), indent=2), flush=True)


if __name__ == "__main__":
    main()
