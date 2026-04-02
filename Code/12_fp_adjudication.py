# ==========================
# 12_fp_adjudication.py
# Stratified random sample of false-positive detection intervals for manual
# adjudication.
#
# For each sampled FP interval the script records:
#   - Exact test-period step boundaries (iv_start, iv_end)
#   - Duration (steps), peak/mean anomaly score, peak prediction error
#   - Distance to the nearest regression alert (–1 if none in test period)
#   - Signature metadata (suite, test, platform)
#   - Blank "adjudication" and "annotation_notes" columns for human review
#
# An optional per-interval PNG plot is written to results/plots/fp_adjudication/
#
# Usage:
#   python Code/12_fp_adjudication.py [--n-sample 40] [--no-plots]
#
# Outputs:
#   results/fp_adjudication_sample.csv
#   results/plots/fp_adjudication/<method>_<sig>_iv<start>.png  (unless --no-plots)
# ==========================

import argparse, glob, math, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
TS_BASE     = os.path.join(DATA_DIR, "timeseries-data", "firefox-android")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots", "fp_adjudication")

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TAU     = 5        # alert-match tolerance (same as baseline evaluation)
GENEROUS_TAU    = 10       # wider window for "nearest alert" metadata reporting
REPO_NAME       = "firefox-android"
METHODS = {
    "ARIMA": os.path.join(RESULTS_DIR, "mozilla_results.csv"),
    "LAST":  os.path.join(RESULTS_DIR, "mozilla_results_LAST.csv"),
    "SMA":   os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"),
    "EWMA":  os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def normalize_sig(x) -> str:
    if x is None: return ""
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return ""
    try:
        f = float(s)
        if math.isfinite(f):
            i = int(f)
            if abs(f - i) < 1e-9:
                return str(i)
    except Exception:
        pass
    return s


def safe_float(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def extract_intervals(ts_sorted):
    """Convert a sorted list of t-indices to (start, end) run pairs."""
    if not ts_sorted:
        return []
    out = []
    s = p = ts_sorted[0]
    for t in ts_sorted[1:]:
        if t == p + 1:
            p = t
        else:
            out.append((s, p))
            s = p = t
    out.append((s, p))
    return out


def closest_t(ts_series: pd.Series, target_ts):
    """Return the row-index in ts_series closest to target_ts (or None)."""
    valid = ts_series.dropna()
    if valid.empty:
        return None
    try:
        return int((valid - target_ts).abs().idxmin())
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pre-load timeseries metadata for all evaluation signatures at once.
# Reading only 5 columns (vs 73) and only T_total rows keeps I/O fast.
# ──────────────────────────────────────────────────────────────────────────────
print("Loading alerts ...")
alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]       = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"] = pd.to_datetime(alerts_df["push_timestamp"], errors="coerce")
alerts_df = alerts_df.dropna(subset=["push_timestamp"])
alerts_reg = alerts_df[
    (alerts_df["alert_summary_repository"] == REPO_NAME) &
    (alerts_df["single_alert_is_regression"] == True)
].copy()

# Collect all alert timestamps per signature
alert_ts_map: dict[str, list] = {}
for sig, grp in alerts_reg.groupby("sig_norm"):
    alert_ts_map[sig] = sorted(grp["push_timestamp"].dropna().tolist())

# Collect T_total per sig from ARIMA results (same value across methods)
print("Pre-loading timeseries metadata for all evaluation signatures ...")
df_arima_meta = pd.read_csv(METHODS["ARIMA"])
t_total_map: dict[str, int] = {}
for sig_raw, grp in df_arima_meta.groupby("signature_id"):
    t_total_map[normalize_sig(sig_raw)] = int(grp["T_total"].iloc[0])

TS_USECOLS = ["push_timestamp", "value", "suite", "test", "machine_platform"]

# ts_cache[sig] = {'ts': pd.Series of timestamps (first T_total rows sorted),
#                  'suite': str, 'test': str, 'platform': str}
ts_cache: dict[str, dict] = {}
for sig in sorted(t_total_map):
    path = os.path.join(TS_BASE, f"{sig}_timeseries_data.csv")
    if not os.path.exists(path):
        ts_cache[sig] = {"ts": pd.Series(dtype="datetime64[ns]"),
                         "suite": "", "test": "", "platform": ""}
        continue
    try:
        try:
            df_ts = pd.read_csv(path, usecols=TS_USECOLS, encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            try:
                df_ts = pd.read_csv(path, usecols=TS_USECOLS, encoding="latin-1")
            except ValueError:
                # Some CSVs may lack certain columns; fall back to what's available
                df_ts = pd.read_csv(path, encoding="latin-1",
                                    usecols=lambda c: c in TS_USECOLS)
        if "push_timestamp" in df_ts.columns:
            df_ts["push_timestamp"] = pd.to_datetime(df_ts["push_timestamp"], errors="coerce")
            df_ts = df_ts.dropna(subset=["push_timestamp"]).sort_values("push_timestamp").reset_index(drop=True)
        T_total = t_total_map.get(sig, len(df_ts))
        df_ts   = df_ts.iloc[:T_total]  # keep only the rows used in the evaluation
        ts_series = df_ts["push_timestamp"] if "push_timestamp" in df_ts.columns \
                    else pd.Series(dtype="datetime64[ns]")
        suite    = str(df_ts["suite"].iloc[0])            if "suite"            in df_ts.columns and len(df_ts) > 0 else ""
        test_col = str(df_ts["test"].iloc[0])             if "test"             in df_ts.columns and len(df_ts) > 0 else ""
        platform = str(df_ts["machine_platform"].iloc[0]) if "machine_platform" in df_ts.columns and len(df_ts) > 0 else ""
        ts_cache[sig] = {"ts": ts_series.reset_index(drop=True),
                         "suite": suite, "test": test_col, "platform": platform}
    except Exception as e:
        ts_cache[sig] = {"ts": pd.Series(dtype="datetime64[ns]"),
                         "suite": "", "test": "", "platform": ""}

print(f"  Cached {len(ts_cache)} signatures.")

# ──────────────────────────────────────────────────────────────────────────────
# Main collection loop: find all FP intervals across all methods
# ──────────────────────────────────────────────────────────────────────────────
print("Collecting FP intervals from per-method result CSVs ...")
all_fp_intervals: list[dict] = []

for method, csv_path in METHODS.items():
    if not os.path.exists(csv_path):
        print(f"  [WARN] Missing: {csv_path} — skipping {method}.")
        continue
    df_res = pd.read_csv(csv_path)
    print(f"  {method}: {df_res['signature_id'].nunique()} signatures loaded.")

    for sig_raw, grp in df_res.groupby("signature_id"):
        sig = normalize_sig(sig_raw)
        grp = grp.sort_values("t").reset_index(drop=True)
        T_train = int(grp["T_train"].iloc[0])
        T_total = int(grp["T_total"].iloc[0])

        # Use pre-cached timeseries data
        meta       = ts_cache.get(sig, {"ts": pd.Series(dtype="datetime64[ns]"),
                                        "suite": "", "test": "", "platform": ""})
        ts_series  = meta["ts"]
        suite      = meta["suite"]
        test_col   = meta["test"]
        platform   = meta["platform"]

        # Map alert timestamps -> t-indices for this sig
        raw_alert_ts = alert_ts_map.get(sig, [])
        alert_t_all: list[int] = []
        if ts_series is not None and not ts_series.empty:
            for ats in raw_alert_ts:
                ti = closest_t(ts_series, ats)
                if ti is not None:
                    alert_t_all.append(ti)
        alert_t_all = sorted(set(alert_t_all))

        # Restrict to test period
        test_alerts = [ta for ta in alert_t_all if ta >= T_train]

        # Find detection intervals in test period
        det_ts = sorted(grp.loc[grp["decision"] == 1, "t"].tolist())
        det_ts = [t for t in det_ts if t >= T_train]
        if not det_ts:
            continue
        intervals = extract_intervals(det_ts)

        # Classify each interval as TP or FP
        for (iv_start, iv_end) in intervals:
            is_tp = any(
                (iv_start - DEFAULT_TAU) <= ta <= (iv_end + DEFAULT_TAU)
                for ta in test_alerts
            )
            if is_tp:
                continue  # TP — skip

            # FP interval: gather metadata
            sub = grp[(grp["t"] >= iv_start) & (grp["t"] <= iv_end)]
            peak_score  = float(sub["anomalyScore"].max()) if "anomalyScore" in sub.columns else float("nan")
            mean_score  = float(sub["anomalyScore"].mean()) if "anomalyScore" in sub.columns else float("nan")
            peak_delta  = float(sub["delta"].max())         if "delta"        in sub.columns else float("nan")

            # Nearest alert distance (generous window)
            if test_alerts:
                nearest_dist = min(abs(ta - iv_start) for ta in test_alerts)
            else:
                nearest_dist = -1

            duration = iv_end - iv_start + 1
            is_sustained = duration >= 5

            all_fp_intervals.append({
                "method"             : method,
                "signature_id"       : sig,
                "suite"              : suite,
                "test_name"          : test_col,
                "platform"           : platform,
                "T_train"            : T_train,
                "T_total"            : T_total,
                "iv_start"           : int(iv_start),
                "iv_end"             : int(iv_end),
                "iv_duration_steps"  : int(duration),
                "is_sustained"       : bool(is_sustained),
                "peak_anomaly_score" : round(peak_score, 4),
                "mean_anomaly_score" : round(mean_score, 4),
                "peak_abs_error"     : round(peak_delta, 4) if not math.isnan(peak_delta) else None,
                "nearest_alert_dist" : int(nearest_dist),
                "adjudication"       : "",   # human fills: True FP / Borderline / Possible Regression
                "annotation_notes"   : "",   # human fills
                # keep raw result-CSV ref for optional plot
                "_csv_path"          : csv_path,
            })

print(f"\nTotal FP intervals collected: {len(all_fp_intervals)}")
fp_df = pd.DataFrame(all_fp_intervals)
print(fp_df.groupby("method")["iv_start"].count().rename("fp_count").to_string())


# ──────────────────────────────────────────────────────────────────────────────
# Stratified random sample
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=40,
                   help="Total number of FP intervals to sample (default 40)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip per-interval PNG plots")
    return p.parse_args()

args = parse_args()
N_SAMPLE = args.n_sample
SEED     = args.seed

rng      = np.random.default_rng(SEED)
methods  = fp_df["method"].unique().tolist()
per_meth = max(1, N_SAMPLE // len(methods))

sampled_rows: list[dict] = []
for meth in methods:
    sub = fp_df[fp_df["method"] == meth]
    n   = min(per_meth, len(sub))
    idx = rng.choice(len(sub), size=n, replace=False)
    sampled_rows.extend(sub.iloc[list(idx)].to_dict("records"))

sample_df = pd.DataFrame(sampled_rows).reset_index(drop=True)
sample_df.insert(0, "sample_id", range(1, len(sample_df) + 1))

# Drop internal path column for final CSV
out_df = sample_df.drop(columns=["_csv_path", "T_train", "T_total"])
out_path = os.path.join(RESULTS_DIR, "fp_adjudication_sample.csv")
out_df.to_csv(out_path, index=False)
print(f"\nSaved {len(out_df)} sampled FP intervals → {out_path}")
print("\nAdjudication column instructions:")
print("  True FP         — detector fired on noise; no real performance issue")
print("  Possible Regression — anomaly looks real but no Treeherder alert filed")
print("  Borderline      — ambiguous; requires further evidence")


# ──────────────────────────────────────────────────────────────────────────────
# Optional: per-interval plots
# ──────────────────────────────────────────────────────────────────────────────
if not args.no_plots:
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\nGenerating {len(sample_df)} plots → {PLOT_DIR}")

    # Pre-load result CSVs once per method
    csv_cache: dict[str, pd.DataFrame] = {}
    for meth, csv_path in METHODS.items():
        if os.path.exists(csv_path):
            csv_cache[meth] = pd.read_csv(csv_path)

    for _, row in sample_df.iterrows():
        meth   = row["method"]
        sig    = row["signature_id"]
        i_start = int(row["iv_start"])
        i_end   = int(row["iv_end"])
        sid    = int(row["sample_id"])

        if meth not in csv_cache:
            continue
        df_m = csv_cache[meth]
        df_s = df_m[df_m["signature_id"].astype(str) == str(sig)].sort_values("t")
        if df_s.empty:
            continue

        t_arr      = df_s["t"].to_numpy()
        actual_arr  = df_s["actual"].to_numpy(dtype=float)
        pred_arr    = df_s["predicted"].to_numpy(dtype=float) if "predicted" in df_s.columns else np.full_like(actual_arr, np.nan)
        score_arr   = df_s["anomalyScore"].to_numpy(dtype=float) if "anomalyScore" in df_s.columns else np.zeros_like(actual_arr)
        dec_arr     = df_s["decision"].to_numpy(dtype=float)

        # Zoom: show ±30 steps around the FP interval
        ctx = 30
        mask = (t_arr >= max(t_arr[0], i_start - ctx)) & (t_arr <= min(t_arr[-1], i_end + ctx))
        t_z = t_arr[mask]; act_z = actual_arr[mask]; pred_z = pred_arr[mask]
        score_z = score_arr[mask]; dec_z = dec_arr[mask]

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax0, ax1 = axes

        ax0.plot(t_z, act_z,  label="Actual",    color="steelblue", lw=1.2)
        ax0.plot(t_z, pred_z, label="Predicted", color="orange",    lw=1.0, linestyle="--")
        ax0.axvspan(i_start, i_end, alpha=0.15, color="red", label="FP interval")
        ax0.set_ylabel("Value"); ax0.legend(fontsize=8); ax0.grid(alpha=0.3)
        ax0.set_title(f"sample_id={sid}  method={meth}  sig={sig}  "
                      f"iv=[{i_start},{i_end}]  dur={int(row['iv_duration_steps'])} steps",
                      fontsize=9)

        ax1.plot(t_z, score_z, color="purple", lw=1.2, label="Anomaly score")
        ax1.axhline(y=0.30, color="gray", lw=0.8, linestyle="--", label="θ=0.30")
        ax1.axvspan(i_start, i_end, alpha=0.15, color="red")
        ax1.fill_between(t_z, 0, dec_z * float(score_z.max() or 1),
                         alpha=0.12, color="green", label="Decision=1")
        ax1.set_ylabel("Anomaly score"); ax1.set_xlabel("t (step)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"sample{sid:03d}_{meth}_{sig}_iv{i_start}.png"
        plt.savefig(os.path.join(PLOT_DIR, fname), dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {PLOT_DIR}")

print("\nDone. Fill in 'adjudication' and 'annotation_notes' columns in:")
print(f"  {out_path}")
