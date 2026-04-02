# ==========================
# 17_fp_label_audit.py
# Targeted false-positive audit for SMA and EWMA detectors.
#
# Collects all FP detection intervals for SMA and EWMA from the
# calibration-baseline test set (70/30 split, firefox-android),
# draws a stratified random sample of 30 per method, applies
# heuristic labeling, and produces a summary table.
#
# Heuristic labeling criteria (documented for reproducibility):
#   "likely genuine anomaly": iv_duration_steps >= 3 AND
#                             peak_anomaly_score >= 0.50
#       Rationale: persistence (≥3 consecutive detection steps) combined
#       with a strong signal (score ≥ 50% above the 0.30 threshold)
#       suggests a real performance shift rather than a transient spike.
#   "likely noise": iv_duration_steps == 1 AND peak_anomaly_score < 0.50
#       Rationale: a single-step, moderate-strength detection is
#       characteristic of a random spike or measurement artefact.
#   "unclear": all other cases (e.g., 2-step intervals, score in [0.30,0.50))
#       Rationale: insufficient evidence to distinguish genuine from noise.
#
# Outputs:
#   results/fp_adjudication_sample.csv    (updated: SMA+EWMA only, labeled)
#   results/fp_audit_summary.csv          (per-method label counts + %)
#
# Run from workspace root:
#   python Code/17_fp_label_audit.py [--n-sample 30] [--seed 42]
# ==========================

import argparse, math, os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths and parameters
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
TS_BASE     = os.path.join(DATA_DIR, "timeseries-data", "firefox-android")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

DEFAULT_TAU  = 5

METHODS_AUDIT = {
    "SMA" : os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"),
    "EWMA": os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"),
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
    except Exception: pass
    return s

def safe_float(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception: return None

def extract_intervals(ts_sorted):
    if not ts_sorted: return []
    out = []; s = p = ts_sorted[0]
    for t in ts_sorted[1:]:
        if t == p + 1: p = t
        else:
            out.append((s, p)); s = p = t
    out.append((s, p))
    return out

def closest_t(ts_series: pd.Series, target_ts):
    valid = ts_series.dropna()
    if valid.empty: return None
    try: return int((valid - target_ts).abs().idxmin())
    except Exception: return None

def heuristic_label(duration: int, peak_score: float) -> str:
    """
    Apply documented heuristic labeling. Returns one of:
      'likely genuine anomaly', 'likely noise', 'unclear'
    """
    if duration >= 3 and peak_score >= 0.50:
        return "likely genuine anomaly"
    if duration == 1 and peak_score < 0.50:
        return "likely noise"
    return "unclear"

# ──────────────────────────────────────────────────────────────────────────────
# Load alerts
# ──────────────────────────────────────────────────────────────────────────────
print("Loading alerts ...")
alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]       = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"] = pd.to_datetime(alerts_df["push_timestamp"], errors="coerce")
alerts_df = alerts_df.dropna(subset=["push_timestamp"])
alerts_reg = alerts_df[
    (alerts_df["alert_summary_repository"] == "firefox-android") &
    (alerts_df["single_alert_is_regression"] == True)
].copy()

alert_ts_map: dict[str, list] = {}
for sig, grp in alerts_reg.groupby("sig_norm"):
    alert_ts_map[sig] = sorted(grp["push_timestamp"].dropna().tolist())

# ──────────────────────────────────────────────────────────────────────────────
# Pre-cache timeseries metadata
# ──────────────────────────────────────────────────────────────────────────────
print("Pre-loading timeseries metadata ...")
df_arima_meta = pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results.csv"))
t_total_map: dict[str, int] = {}
for sig_raw, grp in df_arima_meta.groupby("signature_id"):
    t_total_map[normalize_sig(sig_raw)] = int(grp["T_total"].iloc[0])

TS_USECOLS = ["push_timestamp", "suite", "test", "machine_platform"]
ts_cache: dict[str, dict] = {}
for sig in sorted(t_total_map):
    path = os.path.join(TS_BASE, f"{sig}_timeseries_data.csv")
    if not os.path.exists(path):
        ts_cache[sig] = {"ts": pd.Series(dtype="datetime64[ns]"),
                         "suite": "", "test": "", "platform": ""}
        continue
    try:
        try:
            df_ts = pd.read_csv(path, usecols=lambda c: c in TS_USECOLS, encoding="utf-8")
        except Exception:
            df_ts = pd.read_csv(path, encoding="latin-1",
                                usecols=lambda c: c in TS_USECOLS)
        if "push_timestamp" in df_ts.columns:
            df_ts["push_timestamp"] = pd.to_datetime(df_ts["push_timestamp"], errors="coerce")
            df_ts = df_ts.dropna(subset=["push_timestamp"]).sort_values("push_timestamp").reset_index(drop=True)
        T_total = t_total_map.get(sig, len(df_ts))
        df_ts = df_ts.iloc[:T_total]
        ts_series = df_ts["push_timestamp"] if "push_timestamp" in df_ts.columns \
                    else pd.Series(dtype="datetime64[ns]")
        suite    = str(df_ts["suite"].iloc[0])            if "suite"            in df_ts.columns and len(df_ts) > 0 else ""
        test_col = str(df_ts["test"].iloc[0])             if "test"             in df_ts.columns and len(df_ts) > 0 else ""
        platform = str(df_ts["machine_platform"].iloc[0]) if "machine_platform" in df_ts.columns and len(df_ts) > 0 else ""
        ts_cache[sig] = {"ts": ts_series.reset_index(drop=True),
                         "suite": suite, "test": test_col, "platform": platform}
    except Exception:
        ts_cache[sig] = {"ts": pd.Series(dtype="datetime64[ns]"),
                         "suite": "", "test": "", "platform": ""}

print(f"  Cached {len(ts_cache)} signatures.")

# ──────────────────────────────────────────────────────────────────────────────
# Collect all FP intervals for SMA and EWMA
# ──────────────────────────────────────────────────────────────────────────────
print("\nCollecting FP intervals ...")
all_fp: list[dict] = []

for method, csv_path in METHODS_AUDIT.items():
    if not os.path.exists(csv_path):
        print(f"  [WARN] Missing: {csv_path}")
        continue
    df_res = pd.read_csv(csv_path)
    sigs   = df_res["signature_id"].nunique()
    print(f"  {method}: {sigs} signatures loaded.")

    for sig_raw, grp in df_res.groupby("signature_id"):
        sig = normalize_sig(sig_raw)
        grp = grp.sort_values("t").reset_index(drop=True)
        T_train = int(grp["T_train"].iloc[0])
        T_total = int(grp["T_total"].iloc[0])

        meta      = ts_cache.get(sig, {"ts": pd.Series(dtype="datetime64[ns]"),
                                       "suite": "", "test": "", "platform": ""})
        ts_series = meta["ts"]
        suite     = meta["suite"]
        test_col  = meta["test"]
        platform  = meta["platform"]

        raw_alert_ts     = alert_ts_map.get(sig, [])
        alert_t_all: list[int] = []
        if ts_series is not None and not ts_series.empty:
            for ats in raw_alert_ts:
                ti = closest_t(ts_series, ats)
                if ti is not None:
                    alert_t_all.append(ti)
        alert_t_all = sorted(set(alert_t_all))
        test_alerts = [ta for ta in alert_t_all if ta >= T_train]

        det_ts = sorted(grp.loc[grp["decision"] == 1, "t"].tolist())
        det_ts = [t for t in det_ts if t >= T_train]
        if not det_ts:
            continue
        intervals = extract_intervals(det_ts)

        for (iv_start, iv_end) in intervals:
            is_tp = any(
                (iv_start - DEFAULT_TAU) <= ta <= (iv_end + DEFAULT_TAU)
                for ta in test_alerts
            )
            if is_tp:
                continue  # True positive — skip

            sub = grp[(grp["t"] >= iv_start) & (grp["t"] <= iv_end)]
            peak_score = float(sub["anomalyScore"].max()) if "anomalyScore" in sub.columns else float("nan")
            mean_score = float(sub["anomalyScore"].mean()) if "anomalyScore" in sub.columns else float("nan")
            peak_delta = float(sub["delta"].max())         if "delta"        in sub.columns else float("nan")

            nearest_dist = min(abs(ta - iv_start) for ta in test_alerts) if test_alerts else -1
            duration     = iv_end - iv_start + 1

            # Apply heuristic label
            ps = safe_float(peak_score) or 0.0
            label = heuristic_label(duration, ps)

            all_fp.append({
                "method"             : method,
                "signature_id"       : sig,
                "suite"              : suite,
                "test_name"          : test_col,
                "platform"           : platform,
                "iv_start"           : int(iv_start),
                "iv_end"             : int(iv_end),
                "iv_duration_steps"  : int(duration),
                "is_sustained"       : bool(duration >= 5),
                "peak_anomaly_score" : round(peak_score, 4) if not math.isnan(peak_score) else None,
                "mean_anomaly_score" : round(mean_score, 4) if not math.isnan(mean_score) else None,
                "peak_abs_error"     : round(peak_delta, 4) if not math.isnan(peak_delta) else None,
                "nearest_alert_dist" : int(nearest_dist),
                "adjudication"       : label,
            })

fp_df = pd.DataFrame(all_fp)
print(f"\nTotal FP intervals collected:")
print(fp_df.groupby("method")["iv_start"].count().rename("fp_count").to_string())

# ──────────────────────────────────────────────────────────────────────────────
# Stratified random sample
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, default=30,
                   help="FP intervals to sample per method (default 30)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()

args        = parse_args()
N_PER_METH  = args.n_sample
SEED        = args.seed
rng         = np.random.default_rng(SEED)

sampled: list[dict] = []
for meth in ["SMA", "EWMA"]:
    sub = fp_df[fp_df["method"] == meth]
    n   = min(N_PER_METH, len(sub))
    if n == 0:
        print(f"  [WARN] No FP intervals found for {meth}")
        continue
    idx = rng.choice(len(sub), size=n, replace=False)
    sampled.extend(sub.iloc[list(idx)].to_dict("records"))

sample_df = pd.DataFrame(sampled).reset_index(drop=True)
sample_df.insert(0, "sample_id", range(1, len(sample_df) + 1))

# Save labeled sample
out_path = os.path.join(RESULTS_DIR, "fp_adjudication_sample.csv")
sample_df.to_csv(out_path, index=False)
print(f"\nSaved {len(sample_df)} labeled FP intervals → {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FP AUDIT SUMMARY")
print("=" * 60)

label_order = ["likely genuine anomaly", "likely noise", "unclear"]
summary_rows = []

for meth in ["SMA", "EWMA"]:
    sub = sample_df[sample_df["method"] == meth]
    n   = len(sub)
    if n == 0:
        continue
    counts = sub["adjudication"].value_counts()
    row = {"method": meth, "n_sampled": n}
    for lbl in label_order:
        c = int(counts.get(lbl, 0))
        row[lbl] = c
        row[f"{lbl.replace(' ', '_')}_pct"] = round(100.0 * c / n, 1) if n > 0 else 0.0
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

out_summary = os.path.join(RESULTS_DIR, "fp_audit_summary.csv")
summary_df.to_csv(out_summary, index=False)
print(f"\nSaved summary → {out_summary}")

# Pretty-print for paper use
print("\n--- Table for paper ---")
print(f"{'Method':<8} {'n':<5} {'Likely Genuine':>16} {'Likely Noise':>14} {'Unclear':>9}")
print("-" * 58)
for _, r in summary_df.iterrows():
    ga  = int(r.get("likely genuine anomaly", 0))
    gp  = r.get("likely_genuine_anomaly_pct", 0.0)
    n_  = int(r.get("likely noise", 0))
    np_ = r.get("likely_noise_pct", 0.0)
    u   = int(r.get("unclear", 0))
    up  = r.get("unclear_pct", 0.0)
    tot = int(r["n_sampled"])
    print(f"{r['method']:<8} {tot:<5} {ga:>5} ({gp:.0f}%)      {n_:>4} ({np_:.0f}%)    {u:>3} ({up:.0f}%)")

print("\nLabeling criteria:")
print("  'likely genuine anomaly': duration >= 3 steps AND peak score >= 0.50")
print("  'likely noise':           duration == 1 step AND peak score <  0.50")
print("  'unclear':                all other cases")
print("\nDone.")
