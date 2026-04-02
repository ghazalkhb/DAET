# ==========================
# 16_arima_order_expansion.py
# Expanded ARIMA-order ablation on a 50-signature stratified subset.
#
# Replaces the 20-signature subset in 10_ablation.py with a larger
# 50-signature sample to provide more representative order-selection evidence.
# Uses the same 70/30 chronological split, same pipeline parameters, and
# same five ARIMA order configurations as the original ablation.
#
# Reads timeseries data directly from:
#   Data/timeseries-data/firefox-android/<sig>_timeseries_data.csv
#
# Outputs:
#   results/ablation_arima_order.csv   (updated n_sigs=50)
#
# Run from workspace root:
#   python Code/16_arima_order_expansion.py
# ==========================

import math, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ──────────────────────────────────────────────────────────────────────────────
# Paths and parameters (must match 10_ablation.py defaults)
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
TS_DIR      = os.path.join(DATA_DIR, "timeseries-data", "firefox-android")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

DEFAULT_BETA        = 0.30
DEFAULT_THRESHOLD   = 0.30
DEFAULT_TAU         = 5
DEFAULT_K_SIGMA     = 3.0
DEFAULT_THR_WINDOW  = 30
DEFAULT_MIN_THR     = 1e-6
DEFAULT_ARIMA_WIN   = 60
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_MIN_HISTORY = 20

ARIMA_ORD_GRID = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 2)]
ARIMA_ABL_N    = 50          # expanded from original 20
MAX_SIG_LEN    = 800         # skip very long series to keep runtime manageable

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def normalize_sig(x):
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

def update_score(prev, is_anom, beta):
    return beta * float(is_anom) + (1.0 - beta) * float(prev)

def dyn_thr(hist, k=DEFAULT_K_SIGMA, window=DEFAULT_THR_WINDOW, min_thr=DEFAULT_MIN_THR):
    if not hist: return max(min_thr, 0.0)
    w = hist[-window:] if len(hist) > window else hist
    s = float(np.std(w)) if len(w) >= 2 else float(np.std(hist))
    return max(min_thr, k * s)

def extract_intervals(ts):
    if not ts: return []
    out = []; s = ts[0]; p = ts[0]
    for t in ts[1:]:
        if t == p + 1: p = t
        else:
            out.append((s, p)); s = t; p = t
    out.append((s, p))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Load alerts (regression-classified only, as in 10_ablation.py)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading alerts ...")
alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]       = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"] = pd.to_datetime(alerts_df["push_timestamp"], errors="coerce")
alerts_df = alerts_df.dropna(subset=["push_timestamp"])
alerts_reg = alerts_df[alerts_df["single_alert_is_regression"] == True].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Load all available timeseries files, extract (sig_id, y) pairs
# ──────────────────────────────────────────────────────────────────────────────
print(f"Scanning timeseries directory: {TS_DIR}")
sigs_y = []   # list of (sig_str, y_numpy, ts_series_for_alerts)
for fn in sorted(os.listdir(TS_DIR)):
    if not fn.endswith("_timeseries_data.csv"):
        continue
    sig = fn.replace("_timeseries_data.csv", "")
    sig = normalize_sig(sig)
    path = os.path.join(TS_DIR, fn)
    try:
        df_ts = pd.read_csv(path, usecols=lambda c: c in ["value", "push_timestamp"],
                             encoding="utf-8", low_memory=False)
    except Exception:
        try:
            df_ts = pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            continue

    if "value" not in df_ts.columns:
        continue
    if "push_timestamp" in df_ts.columns:
        df_ts["push_timestamp"] = pd.to_datetime(df_ts["push_timestamp"], errors="coerce")
        df_ts = df_ts.dropna(subset=["push_timestamp"]).sort_values("push_timestamp").reset_index(drop=True)

    y = df_ts["value"].to_numpy(dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < DEFAULT_MIN_HISTORY * 3:
        continue
    if len(y) > MAX_SIG_LEN:
        continue   # skip very long series

    ts_series = df_ts["push_timestamp"].reset_index(drop=True) if "push_timestamp" in df_ts.columns else None
    sigs_y.append((sig, y, ts_series))

# Sort by descending series length for stratified sampling
sigs_y.sort(key=lambda x: -len(x[1]))
print(f"  Loaded {len(sigs_y)} eligible signatures (len <= {MAX_SIG_LEN}, >= {DEFAULT_MIN_HISTORY*3} steps).")

# Stratified subsample of ARIMA_ABL_N signatures evenly spaced across length distribution
N    = min(ARIMA_ABL_N, len(sigs_y))
step = max(1, len(sigs_y) // N)
arima_subset = sigs_y[::step][:N]
print(f"  Selected {len(arima_subset)} signatures (every {step}th) for ablation.")

# ──────────────────────────────────────────────────────────────────────────────
# Map alert timestamps to t-indices for the subset signatures
# ──────────────────────────────────────────────────────────────────────────────
def closest_t(ts_series, target_ts):
    valid = ts_series.dropna()
    if valid.empty: return None
    try: return int((valid - target_ts).abs().idxmin())
    except Exception: return None

alert_cache = {}
for sig, y, ts_series in arima_subset:
    sig_alerts = alerts_reg[alerts_reg["sig_norm"] == sig]
    if ts_series is None or ts_series.empty:
        alert_cache[sig] = []
        continue
    ts_list = [closest_t(ts_series, row["push_timestamp"])
               for _, row in sig_alerts.iterrows()]
    alert_cache[sig] = [x for x in ts_list if x is not None]

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_rows(all_rows, tau=DEFAULT_TAU):
    """Compute per-signature metrics from a list of row dicts, then average."""
    if not all_rows: return None
    df = pd.DataFrame(all_rows)
    rows_out = []
    for sig_raw, sub in df.groupby("signature_id"):
        sig  = str(sig_raw)
        sub  = sub.sort_values("t")
        T_train = int(sub["T_train"].iloc[0])
        T_total = int(sub["T_total"].iloc[0])
        T_test  = max(1, T_total - T_train)

        test_sub  = sub[sub["t"] >= T_train]
        det_ts    = sorted(test_sub.loc[test_sub["decision"] == 1, "t"].tolist())
        intervals = extract_intervals(det_ts)
        n_ivs     = len(intervals)

        alert_t_all = alert_cache.get(sig, [])
        alert_t     = [ta for ta in alert_t_all if ta >= T_train]

        tp_iv = sum(1 for (s, e) in intervals
                    if any((s - tau) <= ta <= (e + tau) for ta in alert_t))
        fp_iv     = n_ivs - tp_iv
        precision = tp_iv / n_ivs if n_ivs > 0 else 0.0
        tp_al     = sum(1 for ta in alert_t
                        if any((s - tau) <= ta <= (e + tau) for (s, e) in intervals))
        recall    = tp_al / len(alert_t) if len(alert_t) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        far       = fp_iv / T_test
        rows_out.append({"precision": precision, "recall": recall, "f1": f1, "far": far})

    ev = pd.DataFrame(rows_out)
    return {
        "mean_precision": round(float(ev["precision"].mean()), 4),
        "mean_recall"   : round(float(ev["recall"].mean()),    4),
        "mean_f1"       : round(float(ev["f1"].mean()),        4),
        "mean_far"      : round(float(ev["far"].mean()),       6),
    }

# ──────────────────────────────────────────────────────────────────────────────
# ARIMA re-run with given order on the subset
# ──────────────────────────────────────────────────────────────────────────────
def run_arima_order(sigs_subset, order):
    min_h    = DEFAULT_MIN_HISTORY
    all_rows = []
    for idx_s, (sig, y, _ts) in enumerate(sigs_subset):
        T       = len(y)
        T_train = max(min_h + 2, int(DEFAULT_TRAIN_RATIO * T))
        if T - T_train < 2:
            continue
        y_train = y[:T_train]
        y_test  = y[T_train:]

        # Fit ARIMA on training data
        arima_fit = None
        try:
            arima_fit = ARIMA(
                y_train, order=order,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit()
        except Exception:
            pass

        # Calibrate delta_hist on training data
        dh = []; score = 0.0
        for t in range(T_train):
            if t < min_h or arima_fit is None: continue
            act = safe_float(y_train[t])
            if act is None: continue
            fv = arima_fit.fittedvalues
            pr = (safe_float(fv.iloc[t] if hasattr(fv, "iloc") else fv[t])
                  if t < len(fv) else safe_float(y_train[t - 1]))
            if pr is None: continue
            delta = abs(act - pr)
            thr = dyn_thr(dh); is_a = 1 if delta > thr else 0
            score = update_score(score, is_a, DEFAULT_BETA)
            dh.append(delta)

        # Fixed-parameter rolling predictions on test set
        params = arima_fit.params if arima_fit is not None else None
        buf    = list(y_train[-DEFAULT_ARIMA_WIN:])
        preds  = []
        for val in y_test:
            pr = None
            if params is not None:
                try:
                    m  = ARIMA(np.array(buf), order=order,
                               enforce_stationarity=False, enforce_invertibility=False)
                    pr = safe_float(m.filter(params).forecast(1)[0])
                except Exception:
                    pass
            if pr is None:
                pr = safe_float(buf[-1])
            preds.append(pr)
            buf.append(val)
            if len(buf) > DEFAULT_ARIMA_WIN: buf.pop(0)

        for i_t, val in enumerate(y_test):
            t_abs = T_train + i_t
            act   = safe_float(val)
            if act is None: continue
            pr    = preds[i_t] if i_t < len(preds) else safe_float(y_train[-1])
            if pr is None: continue
            delta = abs(act - pr)
            thr   = dyn_thr(dh); is_a = 1 if delta > thr else 0
            score = update_score(score, is_a, DEFAULT_BETA)
            dec   = 1 if score >= DEFAULT_THRESHOLD else 0
            dh.append(delta)
            all_rows.append({
                "signature_id": sig, "t": t_abs, "actual": act,
                "predicted": pr, "delta": delta, "isAnomaly": is_a,
                "anomalyScore": score, "decision": dec,
                "T_train": T_train, "T_total": T
            })

        if (idx_s + 1) % 10 == 0:
            print(f"      ... {idx_s + 1}/{len(sigs_subset)} sigs", flush=True)

    return all_rows

# ──────────────────────────────────────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nRunning ARIMA order ablation ({len(arima_subset)} signatures, {len(ARIMA_ORD_GRID)} orders) ...")
arima_rows = []
t_run_start = time.time()

for order in ARIMA_ORD_GRID:
    t0 = time.time()
    print(f"  Order {order} ...", flush=True)
    rows = run_arima_order(arima_subset, order=order)
    ag = evaluate_rows(rows)
    elapsed = time.time() - t0
    if ag:
        arima_rows.append({
            "arima_order"    : str(order),
            "n_sigs"         : len(arima_subset),
            "mean_precision" : ag["mean_precision"],
            "mean_recall"    : ag["mean_recall"],
            "mean_f1"        : ag["mean_f1"],
            "mean_far"       : ag["mean_far"],
        })
        print(f"  {order}: P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
              f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}  ({elapsed:.1f}s)")
    else:
        print(f"  {order}: no results ({elapsed:.1f}s)")

total = time.time() - t_run_start
print(f"\nTotal elapsed: {total:.1f}s")

abl_arima = pd.DataFrame(arima_rows)
out_path = os.path.join(RESULTS_DIR, "ablation_arima_order.csv")
abl_arima.to_csv(out_path, index=False)
print(f"\nSaved {len(abl_arima)} rows -> {out_path}")
print(abl_arima.to_string(index=False))
