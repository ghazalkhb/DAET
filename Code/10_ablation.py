# ==========================
# 10_ablation.py
# Robustness / ablation study + statistical significance tests.
#
# MUST be run AFTER run_pipeline.py (requires results/mozilla_results*.csv,
# results/evaluation_all.csv, results/vectors/).
#
# Three POST-HOC ablations (fast — no detector re-runs):
#   beta        – anomaly-score smoothing factor  in {0.10, 0.20, 0.30, 0.50, 0.70}
#   threshold   – decision threshold θ            in {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
#   tau         – alert-matching tolerance (τ)    in {1, 2, 3, 5, 7, 10}
#
# Two RE-RUN ablations (slower — detector is re-run on a signature subset):
#   SMA window   – w ∈ {5, 10, 15, 20, 30, 50}     (all 214 signatures, ~3 min)
#   ARIMA order  – (p,d,q) ∈ 5 specifications       (stratified 20-sig sample, ~10 min)
#
# Statistical significance:
#   Percentile bootstrap 95 % CI (2 000 resamples) for mean P/R/F1/FAR per method.
#   Paired Wilcoxon signed-rank test for all 6 method pairs × 4 metrics.
#
# Outputs (all in results/):
#   ablation_beta.csv
#   ablation_threshold.csv
#   ablation_tau.csv
#   ablation_sma_window.csv
#   ablation_arima_order.csv
#   stats_significance.json
#
# Run from workspace root:
#   python Code/10_ablation.py
# ==========================

import os, json, math, warnings, time, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed — Wilcoxon tests will be skipped.")
    print("       Install with:  pip install scipy")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
VECTORS_DIR = os.path.join(RESULTS_DIR, "vectors")

# ──────────────────────────────────────────────────────────────────────────────
# Baseline parameters (must match run_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_BETA        = 0.30
DEFAULT_THRESHOLD   = 0.30
DEFAULT_TAU         = 5
DEFAULT_K_SIGMA     = 3.0
DEFAULT_THR_WINDOW  = 30
DEFAULT_MIN_THR     = 1e-6
DEFAULT_SMA_WINDOW  = 20
DEFAULT_ARIMA_ORDER = (1, 1, 1)
DEFAULT_ARIMA_WIN   = 60
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_EWMA_ALPHA  = 0.30
DEFAULT_MIN_HISTORY = 20   # ARIMA minimum history

METHODS = ["ARIMA", "LAST", "SMA", "EWMA"]

# ── Ablation grids ─────────────────────────────────────────────────────────────
BETA_GRID      = [0.10, 0.20, 0.30, 0.50, 0.70]
THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TAU_GRID       = [1, 2, 3, 5, 7, 10]
SMA_WIN_GRID   = [5, 10, 15, 20, 30, 50]
ARIMA_ORD_GRID = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 2)]
ARIMA_ABL_N    = 20   # signatures for ARIMA order ablation

# ──────────────────────────────────────────────────────────────────────────────
# Micro-utilities
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
        if x is None: return None
        v = float(x)
        if math.isnan(v) or math.isinf(v): return None
        return v
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

def json_sanitize(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):  return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [json_sanitize(v) for v in obj]
    return obj

# ──────────────────────────────────────────────────────────────────────────────
# Load shared artefacts
# ──────────────────────────────────────────────────────────────────────────────
print("Loading alerts and evaluation data ...")
alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]        = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"]  = pd.to_datetime(alerts_df["push_timestamp"], errors="coerce")
alerts_df = alerts_df.dropna(subset=["push_timestamp"])
alerts_reg = alerts_df[alerts_df["single_alert_is_regression"] == True].copy()

eval_df = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation_all.csv"))

# Load saved per-step detector results
df_arima = pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results.csv"))
df_last  = pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results_LAST.csv"))
df_sma   = pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"))
df_ewma  = pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"))
results_map = {"ARIMA": df_arima, "LAST": df_last, "SMA": df_sma, "EWMA": df_ewma}

def load_meta_ts(sig):
    path = os.path.join(VECTORS_DIR, f"{sig}_meta.json")
    if not os.path.exists(path):
        return pd.Series([], dtype="datetime64[ns]")
    with open(path) as f:
        meta = json.load(f)
    # Return a proper pd.Series so that .dropna() and arithmetic work correctly
    return pd.Series(
        pd.to_datetime([r.get("push_timestamp") for r in meta["rows"]], errors="coerce")
    )

def closest_t(ts_series, target_ts):
    # Ensure we have a Series (not a DatetimeIndex) before calling .abs()
    valid = pd.Series(ts_series).dropna()
    if valid.empty: return None
    return int((valid - target_ts).abs().idxmin())

def load_vectors():
    avail = []
    for fn in os.listdir(VECTORS_DIR):
        if fn.endswith(".npy"):
            sig = fn[:-4]
            y = np.load(os.path.join(VECTORS_DIR, fn)).astype(float)
            avail.append((sig, y))
    avail.sort(key=lambda x: -len(x[1]))
    return avail

# Pre-cache alert t-indices (read each meta JSON exactly once)
print("Pre-caching alert timestamps ...")
all_sigs = sorted(set(df_arima["signature_id"].astype(str).tolist()))
alert_cache  = {}   # sig -> list of absolute t-indices
T_train_cache = {}  # sig -> T_train

for sig in all_sigs:
    ts_series  = load_meta_ts(sig)
    sig_alerts = alerts_reg[alerts_reg["sig_norm"] == sig]
    ts_list    = [closest_t(ts_series, row["push_timestamp"])
                  for _, row in sig_alerts.iterrows()]
    alert_cache[sig] = [x for x in ts_list if x is not None]

for sig, grp in df_arima.groupby("signature_id"):
    if "T_train" in grp.columns:
        T_train_cache[str(sig)] = int(grp["T_train"].iloc[0])

print(f"  Cached {len(alert_cache)} signatures.")


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation (uses alert_cache for speed; parameterised by tau)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_df(df_res, tau):
    """
    Evaluate a result DataFrame with given alert-matching tolerance `tau`.
    Returns per-signature DataFrame: signature_id, precision, recall, f1, far.
    """
    rows = []
    for sig_raw, sub in df_res.groupby("signature_id"):
        sig  = str(sig_raw)
        sub  = sub.copy()
        sub["t"] = sub["t"].astype(int)

        T_train = int(sub["T_train"].iloc[0]) if "T_train" in sub.columns else T_train_cache.get(sig, 0)
        T_total = int(sub["T_total"].iloc[0]) if "T_total" in sub.columns else sub["t"].max() + 1
        T_test  = max(1, T_total - T_train)

        test_sub  = sub[sub["t"] >= T_train]
        det_ts    = sorted(test_sub.loc[test_sub["decision"] == 1, "t"].tolist())
        intervals = extract_intervals(det_ts)
        n_ivs     = len(intervals)

        alert_t_all = alert_cache.get(sig, [])
        alert_t     = [ta for ta in alert_t_all if ta >= T_train]

        tp_iv = sum(
            1 for (s, e) in intervals
            if any((s - tau) <= ta <= (e + tau) for ta in alert_t)
        )
        fp_iv     = n_ivs - tp_iv
        precision = tp_iv / n_ivs if n_ivs > 0 else 0.0

        tp_al  = sum(
            1 for ta in alert_t
            if any((s - tau) <= ta <= (e + tau) for (s, e) in intervals)
        )
        recall = tp_al / len(alert_t) if len(alert_t) > 0 else 0.0
        f1     = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
        far    = fp_iv / T_test

        rows.append({
            "signature_id"  : sig,
            "precision"     : precision,
            "recall"        : recall,
            "f1"            : f1,
            "far"           : far,
            "n_alerts_test" : len(alert_t),
        })
    return pd.DataFrame(rows)


def agg_eval(ev):
    """Aggregate per-signature evaluation to mean metrics."""
    if ev.empty:
        return {"mean_precision": None, "mean_recall": None,
                "mean_f1": None, "mean_far": None}
    return {
        "mean_precision": round(float(ev["precision"].mean()), 4),
        "mean_recall"   : round(float(ev["recall"].mean()),    4),
        "mean_f1"       : round(float(ev["f1"].mean()),        4),
        "mean_far"      : round(float(ev["far"].mean()),       6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    """Percentile bootstrap CI for the mean of `values`."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None, None
    means = np.array([
        rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo    = float(np.percentile(means, alpha * 100))
    hi    = float(np.percentile(means, (1 - alpha) * 100))
    return round(lo, 4), round(hi, 4)


# ══════════════════════════════════════════════════════════════════════════════
# POST-HOC ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("POST-HOC ABLATIONS  (re-use saved detector outputs)")
print("=" * 60)

# ── 1. Threshold sweep ────────────────────────────────────────────────────────
print("\n[1/3] Threshold ablation ...")
thr_rows = []
for thr in THRESHOLD_GRID:
    for method, df in results_map.items():
        df2 = df.copy()
        df2["decision"] = (df2["anomalyScore"] >= thr).astype(int)
        ev  = evaluate_df(df2, tau=DEFAULT_TAU)
        ag  = agg_eval(ev)
        thr_rows.append({"threshold": thr, "method": method, **ag})
        print(f"  thr={thr:.2f} {method:5s}: "
              f"P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
              f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}")

abl_threshold = pd.DataFrame(thr_rows)
abl_threshold.to_csv(os.path.join(RESULTS_DIR, "ablation_threshold.csv"), index=False)
print(f"  -> Saved ablation_threshold.csv")

# ── 2. Beta sweep ─────────────────────────────────────────────────────────────
print("\n[2/3] Beta ablation ...")
beta_rows = []
for beta in BETA_GRID:
    for method, df in results_map.items():
        df2 = df.copy()
        # Recompute anomalyScore per-signature from saved isAnomaly (score reset to 0).
        # isAnomaly is independent of beta; only the score accumulation changes.
        new_scores = {}
        for sig_raw, grp in df2.groupby("signature_id"):
            grp_s = grp.sort_values("t")
            score = 0.0
            for idx, row in grp_s.iterrows():
                score = update_score(score, row.get("isAnomaly", 0), beta)
                new_scores[idx] = score
        df2["anomalyScore"] = df2.index.map(new_scores)
        df2["decision"]     = (df2["anomalyScore"] >= DEFAULT_THRESHOLD).astype(int)
        ev  = evaluate_df(df2, tau=DEFAULT_TAU)
        ag  = agg_eval(ev)
        beta_rows.append({"beta": beta, "method": method, **ag})
        print(f"  beta={beta:.2f} {method:5s}: "
              f"P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
              f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}")

abl_beta = pd.DataFrame(beta_rows)
abl_beta.to_csv(os.path.join(RESULTS_DIR, "ablation_beta.csv"), index=False)
print(f"  -> Saved ablation_beta.csv")

# ── 3. Tau sweep ──────────────────────────────────────────────────────────────
print("\n[3/3] Tau ablation ...")
tau_rows = []
for tau in TAU_GRID:
    for method, df in results_map.items():
        ev = evaluate_df(df, tau=tau)
        ag = agg_eval(ev)
        tau_rows.append({"tau": tau, "method": method, **ag})
        print(f"  tau={tau:2d} {method:5s}: "
              f"P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
              f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}")

abl_tau = pd.DataFrame(tau_rows)
abl_tau.to_csv(os.path.join(RESULTS_DIR, "ablation_tau.csv"), index=False)
print(f"  -> Saved ablation_tau.csv")


# ══════════════════════════════════════════════════════════════════════════════
# RE-RUN ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RE-RUN ABLATIONS  (detector re-run on signature subset)")
print("=" * 60)

sigs_y = load_vectors()
print(f"Loaded {len(sigs_y)} vectors (sorted by descending length).")


# ── 4. SMA window sweep ───────────────────────────────────────────────────────
def run_sma_rerun(sigs_y, window):
    """Run SMA detector with given window on all signatures."""
    min_h    = window
    all_rows = []
    for sig, y in sigs_y:
        T       = len(y)
        T_train = max(min_h + 2, int(DEFAULT_TRAIN_RATIO * T))
        if T - T_train < 2:
            continue
        y_train = y[:T_train]
        y_test  = y[T_train:]

        dh = []; score = 0.0
        for t in range(T_train):
            if t < min_h: continue
            act = safe_float(y_train[t])
            if act is None: continue
            pr = safe_float(np.mean(y_train[max(0, t - window):t]))
            if pr is None: continue
            delta = abs(act - pr)
            thr   = dyn_thr(dh); is_a = 1 if delta > thr else 0
            score = update_score(score, is_a, DEFAULT_BETA)
            dh.append(delta)

        for i_t, val in enumerate(y_test):
            t_abs = T_train + i_t
            act   = safe_float(val)
            if act is None:
                all_rows.append({
                    "signature_id": sig, "t": t_abs, "actual": None,
                    "predicted": np.nan, "delta": np.nan,
                    "isAnomaly": 0, "anomalyScore": score, "decision": 0,
                    "T_train": T_train, "T_total": T
                })
                continue
            win_d = np.concatenate([y_train, y_test[:i_t]])[-window:]
            pr    = safe_float(np.mean(win_d)) if len(win_d) > 0 else None
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
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


print("\n[4/5] SMA window ablation (all signatures) ...")
sma_rows = []
for w in SMA_WIN_GRID:
    t0   = time.time()
    df_w = run_sma_rerun(sigs_y, window=w)
    if df_w.empty:
        print(f"  w={w}: no results"); continue
    ev  = evaluate_df(df_w, tau=DEFAULT_TAU)
    ag  = agg_eval(ev)
    elapsed = time.time() - t0
    sma_rows.append({"sma_window": w, **ag})
    print(f"  w={w:2d}: P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
          f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}  ({elapsed:.1f}s)")

abl_sma = pd.DataFrame(sma_rows)
abl_sma.to_csv(os.path.join(RESULTS_DIR, "ablation_sma_window.csv"), index=False)
print(f"  -> Saved ablation_sma_window.csv")


# ── 5. ARIMA order sweep ──────────────────────────────────────────────────────
def run_arima_rerun(sigs_subset, order):
    """Re-run ARIMA with a given (p,d,q) order on the provided signature list."""
    min_h    = DEFAULT_MIN_HISTORY
    all_rows = []
    for idx_s, (sig, y) in enumerate(sigs_subset):
        T       = len(y)
        T_train = max(min_h + 2, int(DEFAULT_TRAIN_RATIO * T))
        if T - T_train < 2: continue
        y_train = y[:T_train]
        y_test  = y[T_train:]

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
            thr   = dyn_thr(dh); is_a = 1 if delta > thr else 0
            score = update_score(score, is_a, DEFAULT_BETA)
            dh.append(delta)

        # Fixed-parameter rolling predictions on the test set
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

        if (idx_s + 1) % 5 == 0:
            print(f"    ... {idx_s + 1}/{len(sigs_subset)} sigs", flush=True)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# Stratified sample: take every Nth signature from the length-sorted list
# so the subset spans short, medium, and long series.
N            = ARIMA_ABL_N
step         = max(1, len(sigs_y) // N)
arima_subset = sigs_y[::step][:N]

print(f"\n[5/5] ARIMA order ablation ({len(arima_subset)} signatures) ...")
arima_rows = []
for order in ARIMA_ORD_GRID:
    t0 = time.time()
    print(f"  Order {order} ...", flush=True)
    df_o = run_arima_rerun(arima_subset, order=order)
    if df_o.empty:
        print(f"    No results"); continue
    ev  = evaluate_df(df_o, tau=DEFAULT_TAU)
    ag  = agg_eval(ev)
    elapsed = time.time() - t0
    arima_rows.append({"arima_order": str(order), "n_sigs": len(arima_subset), **ag})
    print(f"  {order}: P={ag['mean_precision']:.3f} R={ag['mean_recall']:.3f} "
          f"F1={ag['mean_f1']:.3f} FAR={ag['mean_far']:.5f}  ({elapsed:.1f}s)")

abl_arima = pd.DataFrame(arima_rows)
abl_arima.to_csv(os.path.join(RESULTS_DIR, "ablation_arima_order.csv"), index=False)
print(f"  -> Saved ablation_arima_order.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

METRIC_COLS = {
    "Precision"       : "Precision",
    "Recall"          : "Recall",
    "F1"              : "F1",
    "False_Alarm_Rate": "False_Alarm_Rate",
}

result = {"bootstrap_ci": {}, "wilcoxon": {}}

# ── Bootstrap 95 % CIs ────────────────────────────────────────────────────────
print("\nComputing bootstrap 95% CIs (2 000 resamples) ...")
for m in METHODS:
    result["bootstrap_ci"][m] = {}
    ev_m = eval_df[eval_df["method"] == m]
    for metric, col in METRIC_COLS.items():
        if col not in ev_m.columns:
            continue
        vals   = ev_m[col].dropna().tolist()
        mean_v = round(float(np.mean(vals)), 4) if vals else None
        lo, hi = bootstrap_ci(vals)
        result["bootstrap_ci"][m][metric] = {
            "mean"    : mean_v,
            "ci95_lo" : lo,
            "ci95_hi" : hi,
            "n"       : len(vals),
        }
        ci_str = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "[N/A]"
        print(f"  {m:5s} {metric:20s}: {mean_v:.4f}  {ci_str}")

# ── Pairwise Wilcoxon signed-rank tests ───────────────────────────────────────
print("\nPairwise Wilcoxon signed-rank tests ...")
method_ev = {
    m: eval_df[eval_df["method"] == m].set_index("signature_id")
    for m in METHODS
}

for m1, m2 in itertools.combinations(METHODS, 2):
    pair_key = f"{m1}_vs_{m2}"
    result["wilcoxon"][pair_key] = {}
    ev1 = method_ev[m1]
    ev2 = method_ev[m2]
    common = ev1.index.intersection(ev2.index)

    for metric, col in METRIC_COLS.items():
        if col not in ev1.columns or col not in ev2.columns:
            result["wilcoxon"][pair_key][metric] = None
            continue

        x1 = ev1.loc[common, col].dropna()
        x2 = ev2.loc[common, col].dropna()
        # Only keep indices present in both after dropna
        both = x1.index.intersection(x2.index)
        x1   = x1.loc[both].to_numpy(dtype=float)
        x2   = x2.loc[both].to_numpy(dtype=float)

        if len(x1) < 5:
            result["wilcoxon"][pair_key][metric] = {"n": int(len(x1)), "p_value": None}
            continue

        if not HAS_SCIPY:
            result["wilcoxon"][pair_key][metric] = {
                "n": int(len(x1)), "p_value": None,
                "note": "scipy not available"
            }
            continue

        diffs = x1 - x2
        if np.all(diffs == 0):
            result["wilcoxon"][pair_key][metric] = {
                "n": int(len(x1)), "statistic": 0.0, "p_value": 1.0,
                "significant_at_0.05": False, "note": "identical distributions"
            }
            continue

        try:
            stat, p = _scipy_wilcoxon(x1, x2)
            sig_flag = bool(p < 0.05)
            result["wilcoxon"][pair_key][metric] = {
                "n"                   : int(len(x1)),
                "statistic"           : round(float(stat), 4),
                "p_value"             : round(float(p), 6),
                "significant_at_0.05" : sig_flag,
            }
            print(f"  {m1:5s} vs {m2:5s} | {metric:20s}: "
                  f"p={p:.4f}{'*' if sig_flag else ' '} (n={len(x1)})")
        except Exception as err:
            result["wilcoxon"][pair_key][metric] = {"error": str(err)}

# ── Save results ──────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "stats_significance.json")
with open(out_path, "w") as f:
    json.dump(json_sanitize(result), f, indent=2)
print(f"\n-> Significance results saved: {out_path}")

print("\n" + "=" * 60)
print("Ablation study complete!")
print("  results/ablation_beta.csv")
print("  results/ablation_threshold.csv")
print("  results/ablation_tau.csv")
print("  results/ablation_sma_window.csv")
print("  results/ablation_arima_order.csv")
print("  results/stats_significance.json")
print("=" * 60)
print("\nNext step:  python Code/generate_report.py")
