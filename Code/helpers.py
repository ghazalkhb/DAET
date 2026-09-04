# ==========================
# helpers.py
# Shared utility functions used across the ARIMA pipeline scripts.
# ==========================

import math
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Signature ID normalization ----------

def normalize_sig(x) -> str:
    """Normalize signature IDs so '4754836.0' and '4754836' compare equal."""
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if math.isfinite(f):
            i = int(f)
            if abs(f - i) < 1e-9:
                return str(i)
    except Exception:
        pass
    return s


# ---------- Safe numeric conversion ----------

def safe_float(x):
    """Convert to float; return None for NaN, Inf, or unconvertible values."""
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ---------- Algorithm 3 helpers ----------

def update_anomaly_score(prev_score: float, is_anomaly: int, beta: float) -> float:
    """Exponential moving average anomaly score update (Algorithm 3)."""
    return (beta * float(is_anomaly)) + ((1.0 - beta) * float(prev_score))


def threshold_dynamic(delta_hist, k_sigma, window, min_thr):
    """Compute dynamic threshold from the recent delta history."""
    if len(delta_hist) == 0:
        return max(min_thr, 0.0)
    w = delta_hist[-window:] if len(delta_hist) > window else delta_hist
    s = float(np.std(w)) if len(w) >= 2 else float(np.std(delta_hist))
    return max(min_thr, k_sigma * s)


# ---------- Plotting ----------

def plot_signature(
    df_sig: pd.DataFrame,
    out_path: str,
    title: str,
    pred_label: str = "Predicted",
):
    """Plot actual vs predicted values together with the anomaly decision/score."""
    t = df_sig["t"].to_numpy()
    actual = df_sig["actual"].to_numpy(dtype=float)
    predicted = df_sig["predicted"].to_numpy(dtype=float)
    decision = df_sig["decision"].to_numpy(dtype=float)
    anomaly_score = df_sig["anomalyScore"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, actual, label="Actual")
    axes[0].plot(t, predicted, label=pred_label)
    axes[0].set_ylabel("Value")
    axes[0].set_title(title)
    axes[0].legend(loc="best")

    axes[1].step(t, decision, where="post", label="Decision (Enable=1)")
    axes[1].plot(t, anomaly_score, label="anomalyScore")
    axes[1].set_ylabel("Decision / anomalyScore")
    axes[1].set_xlabel("t")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best")
    axes[1].set_title("Adaptive Decision")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------- Evaluation helpers ----------

def extract_intervals(ts):
    """Convert a sorted list of t-indices (where decision==1) into contiguous intervals."""
    if not ts:
        return []
    out = []
    s = ts[0]
    p = ts[0]
    for t in ts[1:]:
        if t == p + 1:
            p = t
        else:
            out.append((s, p))
            s = t
            p = t
    out.append((s, p))
    return out


def load_meta_timestamps(sig, meta_dir):
    """Load push timestamps from a signature's metadata JSON file."""
    path = os.path.join(meta_dir, f"{sig}_meta.json")
    with open(path, "r") as f:
        meta = json.load(f)
    ts = pd.to_datetime(
        [r.get("push_timestamp", None) for r in meta["rows"]], errors="coerce"
    )
    return pd.Series(ts)


def closest_t_index(ts_series, target_ts):
    """Return the index in ts_series whose timestamp is closest to target_ts."""
    valid = ts_series.dropna()
    if len(valid) == 0:
        return None
    diffs = (valid - target_ts).abs()
    return int(diffs.idxmin())


# ---------- JSON serialization ----------

def json_sanitize(obj):
    """Recursively convert pandas/numpy types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    return obj
