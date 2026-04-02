# ==========================
# 11_replay_experiment.py
# End-to-End Replay Experiment -- Mozilla Telemetry
#
# Simulates adaptive "trace-on" decisions by replaying the detector outputs
# already produced by run_pipeline.py.  For each signature × predictor:
#
#   decision_t = 1  ->  trace ON  (data recorded)
#   decision_t = 0  ->  trace OFF (data suppressed)
#
# Three core metrics are measured consistently across the four classical
# predictors (ARIMA, LAST, SMA, EWMA) on the TEST portion of each time series:
#
#   1. Storage Reduction  -- fraction of test-set points NOT recorded
#                           storage_red = 1 - (sum(decision==1) / T_test)
#
#   2. Trigger Frequency  -- number of trace-on activations (0->1 transitions)
#                           per test time step, indicating overhead of
#                           switching the sensor on/off
#                           trigger_freq = n_activations / T_test
#
#   3. Alert Coverage     -- fraction of ground-truth regression alerts that
#                           were covered by at least one trace-on interval
#                           (= recall on the test set, tolerance +/-τ pushes)
#
# Relies on pre-computed outputs from run_pipeline.py:
#   results/evaluation_all.csv   -- per-signature evaluation rows
#   results/mozilla_results*.csv -- raw per-step detector outputs
#   results/vectors/<sig>_meta.json -- push timestamps for delay computation
#   Data/alerts_data.csv         -- ground-truth regression alerts
#
# Saves to results/replay/:
#   replay_results.csv           -- per-signature × method replay metrics
#   replay_summary.json          -- aggregate stats for report
#   plots/replay_*.png           -- six publication-ready plots
#   replay_report.pdf            -- PDF report (generated via fpdf2)
#
# Run from workspace root:
#   python Code/11_replay_experiment.py
# ==========================

import os, sys, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
RESULTS_DIR = os.path.join(ROOT, "results")
VECTORS_DIR = os.path.join(RESULTS_DIR, "vectors")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")

REPLAY_DIR   = os.path.join(RESULTS_DIR, "replay")
REPLAY_PLOTS = os.path.join(REPLAY_DIR, "plots")
for d in [REPLAY_DIR, REPLAY_PLOTS]:
    os.makedirs(d, exist_ok=True)

# ── Settings (must match run_pipeline.py) ────────────────────────────────────
TOL          = 5       # +/-TOL pushes for alert-interval matching
TRAIN_RATIO  = 0.70
METHODS      = ["ARIMA", "LAST", "SMA", "EWMA"]
METHOD_COLORS = {
    "ARIMA": "#1f77b4",
    "LAST" : "#ff7f0e",
    "SMA"  : "#2ca02c",
    "EWMA" : "#d62728",
}
METHOD_LABELS = {
    "ARIMA": "ARIMA(1,1,1)",
    "LAST" : "LAST (naive)",
    "SMA"  : "SMA(w=20)",
    "EWMA" : "EWMA(a=0.30)",
}

# ── Raw result CSVs (produced by run_pipeline.py / individual detectors) ─────
RAW_CSVS = {
    "ARIMA": os.path.join(RESULTS_DIR, "mozilla_results.csv"),
    "LAST" : os.path.join(RESULTS_DIR, "mozilla_results_LAST.csv"),
    "SMA"  : os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"),
    "EWMA" : os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"),
}

printf = lambda *a, **k: print(*a, **k, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def load_meta_ts(sig):
    """Load per-row push_timestamps from <sig>_meta.json."""
    path = os.path.join(VECTORS_DIR, f"{sig}_meta.json")
    if not os.path.exists(path):
        return pd.Series([], dtype="datetime64[ns]")
    with open(path) as f:
        meta = json.load(f)
    ts = pd.to_datetime(
        [r.get("push_timestamp") for r in meta.get("rows", [])],
        errors="coerce",
    )
    return pd.Series(ts)


def closest_t(ts_series, target_ts):
    valid = ts_series.dropna()
    if valid.empty:
        return None
    return int((valid - target_ts).abs().idxmin())


def extract_intervals(ts_list):
    """Convert sorted list of t-indices (decision==1) into contiguous intervals."""
    if not ts_list:
        return []
    out = []; s = ts_list[0]; p = ts_list[0]
    for t in ts_list[1:]:
        if t == p + 1: p = t
        else:
            out.append((s, p)); s = t; p = t
    out.append((s, p))
    return out


def count_activations(decision_array):
    """Count 0->1 transitions in the decision sequence."""
    d = np.asarray(decision_array, dtype=int)
    if len(d) == 0:
        return 0
    transitions = np.where((d[1:] == 1) & (d[:-1] == 0))[0]
    # Also count if first element is 1 (initial activation)
    first = 1 if d[0] == 1 else 0
    return first + len(transitions)


def json_sanitize(obj):
    if obj is None: return None
    try:
        if pd.isna(obj): return None
    except Exception: pass
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):  return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [json_sanitize(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 -- Load pre-computed detector outputs
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 -- Loading pre-computed detector outputs")
print("=" * 65)

raw = {}
for method, csv_path in RAW_CSVS.items():
    if not os.path.exists(csv_path):
        printf(f"  [WARN] Missing: {csv_path} -- {method} will be skipped")
        continue
    df = pd.read_csv(csv_path)
    df["signature_id"] = df["signature_id"].apply(normalize_sig)
    raw[method] = df
    printf(f"  Loaded {method:5s}: {len(df):>7,} rows, "
           f"{df['signature_id'].nunique()} signatures")

missing_methods = [m for m in METHODS if m not in raw]
if missing_methods:
    printf(f"\n[WARN] Methods with no raw CSV: {missing_methods}")
    printf("       Run run_pipeline.py first to generate them.\n")
    if not raw:
        printf("[ERROR] No detector outputs found. Exiting.")
        sys.exit(1)

METHODS_AVAIL = [m for m in METHODS if m in raw]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 -- Load alerts and build per-signature alert-timestamp lookup
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 -- Loading ground-truth regression alerts")
print("=" * 65)

alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]       = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"] = pd.to_datetime(
    alerts_df["push_timestamp"], errors="coerce")
alerts_reg = alerts_df[
    alerts_df["single_alert_is_regression"] == True
].dropna(subset=["push_timestamp"])

printf(f"  Total regression-alert rows : {len(alerts_reg):,}")
printf(f"  Unique alert signatures     : {alerts_reg['sig_norm'].nunique():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 -- Per-signature replay simulation
#
# For each (signature, method) we:
#   a) Extract the TEST portion of the decision time series
#   b) Compute storage_reduction, trigger_frequency
#   c) Map alert timestamps -> t-indices; compute alert_coverage (recall)
#   d) Compute on_interval statistics (median duration, burst ratio)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 -- Replay simulation (trace-on decisions)")
print("=" * 65)

print("  Pre-caching meta timestamps and alert indices ...", flush=True)
# Pre-load all meta timestamps and alert t-indices once -- reused across methods
all_sigs = sorted(set(
    sig for m in METHODS_AVAIL for sig in raw[m]["signature_id"].unique()
))
_ts_cache    = {}    # sig -> pd.Series of push_timestamps
_alert_cache = {}    # sig -> list of absolute t-indices for all alerts

alerts_reg_indexed = alerts_reg.set_index("sig_norm", drop=False)

for sig in all_sigs:
    ts_series = load_meta_ts(sig)
    _ts_cache[sig] = ts_series

    if sig in alerts_reg_indexed.index:
        sig_alts = alerts_reg_indexed.loc[[sig]]
        alert_ts = [closest_t(ts_series, a["push_timestamp"]) for _, a in sig_alts.iterrows()]
        _alert_cache[sig] = [ta for ta in alert_ts if ta is not None]
    else:
        _alert_cache[sig] = []

printf(f"  Cached timestamps for {len(_ts_cache)} signatures, "
       f"{sum(len(v) for v in _alert_cache.values())} total alert t-indices")

replay_rows = []

for method in METHODS_AVAIL:
    df_all = raw[method]

    # Determine T_train column name (may differ between individual scripts
    # and run_pipeline.py outputs)
    has_Ttrain = "T_train" in df_all.columns

    for sig in sorted(df_all["signature_id"].unique()):
        sub = df_all[df_all["signature_id"] == sig].copy()
        sub = sub.sort_values("t").reset_index(drop=True)

        T_total = int(sub["t"].max()) + 1

        # ── Identify train/test boundary ──────────────────────────────────
        if has_Ttrain and sub["T_train"].notna().any():
            T_train = int(sub["T_train"].dropna().iloc[0])
        else:
            T_train = max(2, int(TRAIN_RATIO * T_total))

        # Keep only test-set rows
        test_sub = sub[sub["t"] >= T_train].copy()
        T_test   = len(test_sub)

        if T_test < 2:
            continue

        dec        = test_sub["decision"].fillna(0).astype(int).to_numpy()
        t_indices  = test_sub["t"].to_numpy()

        # ── Storage reduction ─────────────────────────────────────────────
        on_count        = int(np.sum(dec == 1))
        storage_red     = 1.0 - (on_count / T_test)
        enabled_frac    = on_count / T_test

        # ── Trigger frequency ─────────────────────────────────────────────
        n_activations   = count_activations(dec)
        trigger_freq    = n_activations / T_test   # activations per time step

        # ── On-interval statistics ────────────────────────────────────────
        on_ts       = sorted(t_indices[dec == 1].tolist())
        intervals   = extract_intervals(on_ts)
        n_intervals = len(intervals)
        durations   = [e - s + 1 for s, e in intervals]
        med_dur     = float(np.median(durations)) if durations else 0.0
        # Burst ratio: fraction of on-time that is in bursts longer than 1
        burst_dur   = sum(d for d in durations if d > 1)
        burst_ratio = burst_dur / on_count if on_count > 0 else 0.0

        # ── Alert coverage (recall on test set) ──────────────────────────
        # Use pre-cached alert t-indices
        alert_t_all  = _alert_cache.get(sig, [])
        alert_t_test = [ta for ta in alert_t_all if ta >= T_train]
        n_alerts_test = len(alert_t_test)

        tp_alerts = 0
        for ta in alert_t_test:
            for (s, e) in intervals:
                if (s - TOL) <= ta <= (e + TOL):
                    tp_alerts += 1
                    break

        alert_coverage = tp_alerts / n_alerts_test if n_alerts_test > 0 else None

        # ── Precision (interval-level) ────────────────────────────────────
        tp_ivs = sum(
            1 for (s, e) in intervals
            if any((s - TOL) <= ta <= (e + TOL) for ta in alert_t_test)
        )
        fp_ivs    = n_intervals - tp_ivs
        precision = tp_ivs / n_intervals if n_intervals > 0 else 0.0

        replay_rows.append({
            "method"          : method,
            "signature_id"    : sig,
            "T_total"         : T_total,
            "T_train"         : T_train,
            "T_test"          : T_test,
            "On_Count"        : on_count,
            "Enabled_Frac"    : round(enabled_frac,  4),
            "Storage_Red"     : round(storage_red,   4),
            "N_Activations"   : n_activations,
            "Trigger_Freq"    : round(trigger_freq,  6),
            "N_Intervals"     : n_intervals,
            "Median_Interval_Dur": round(med_dur, 2),
            "Burst_Ratio"     : round(burst_ratio, 4),
            "N_Alerts_Test"   : n_alerts_test,
            "TP_Alerts"       : tp_alerts,
            "Alert_Coverage"  : round(alert_coverage, 4) if alert_coverage is not None else None,
            "TP_Intervals"    : tp_ivs,
            "FP_Intervals"    : fp_ivs,
            "Precision"       : round(precision, 4),
        })

    printf(f"  {method:5s}: processed {sum(1 for r in replay_rows if r['method']==method)} signatures")

replay_df = pd.DataFrame(replay_rows)

# ── Save per-signature replay results ────────────────────────────────────────
replay_csv = os.path.join(REPLAY_DIR, "replay_results.csv")
replay_df.to_csv(replay_csv, index=False)
printf(f"\nSaved per-signature results -> {replay_csv}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 -- Aggregate statistics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 -- Aggregate statistics per method")
print("=" * 65)

def agg_replay(df_m):
    n      = len(df_m)
    alerted = df_m[df_m["N_Alerts_Test"] > 0]
    n_alerted = len(alerted)

    cov_vals = alerted["Alert_Coverage"].dropna()
    prec_vals = df_m["Precision"].dropna()

    return {
        "n_signatures"       : n,
        "n_alerted_sigs"     : n_alerted,
        # Storage reduction
        "mean_storage_red"   : round(float(df_m["Storage_Red"].mean()), 4),
        "std_storage_red"    : round(float(df_m["Storage_Red"].std()),  4),
        "median_storage_red" : round(float(df_m["Storage_Red"].median()), 4),
        # Trigger frequency
        "mean_trigger_freq"  : round(float(df_m["Trigger_Freq"].mean()), 6),
        "std_trigger_freq"   : round(float(df_m["Trigger_Freq"].std()),  6),
        "median_trigger_freq": round(float(df_m["Trigger_Freq"].median()), 6),
        "mean_activations"   : round(float(df_m["N_Activations"].mean()), 2),
        # Alert coverage (recall) -- only for signatures that have alerts
        "n_alerted_with_cov" : int(cov_vals.shape[0]),
        "mean_alert_cov"     : round(float(cov_vals.mean()), 4) if len(cov_vals) > 0 else None,
        "std_alert_cov"      : round(float(cov_vals.std()),  4) if len(cov_vals) > 0 else None,
        # Precision
        "mean_precision"     : round(float(prec_vals.mean()), 4) if len(prec_vals) > 0 else None,
        # Interval characteristics
        "mean_interval_dur"  : round(float(df_m["Median_Interval_Dur"].mean()), 2),
        "mean_burst_ratio"   : round(float(df_m["Burst_Ratio"].mean()), 4),
    }

summary = {}
for m in METHODS_AVAIL:
    df_m = replay_df[replay_df["method"] == m]
    summary[m] = agg_replay(df_m)
    a = summary[m]
    printf(f"\n  [{m}]")
    printf(f"    Signatures          : {a['n_signatures']}")
    printf(f"    Storage Reduction   : {a['mean_storage_red']*100:.1f}% +/- {a['std_storage_red']*100:.1f}%")
    printf(f"    Trigger Frequency   : {a['mean_trigger_freq']:.4f} +/- {a['std_trigger_freq']:.4f} (per step)")
    printf(f"    Mean Activations    : {a['mean_activations']:.1f}")
    printf(f"    Alert Coverage      : {a.get('mean_alert_cov', 'N/A')} (n={a['n_alerted_with_cov']})")
    printf(f"    Precision           : {a.get('mean_precision', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 -- Plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 -- Generating publication-ready plots")
print("=" * 65)

meth_labels = [METHOD_LABELS[m] for m in METHODS_AVAIL]
meth_colors = [METHOD_COLORS[m] for m in METHODS_AVAIL]

# ── Plot 1: Storage Reduction (bar + error bars) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
means = [summary[m]["mean_storage_red"] * 100       for m in METHODS_AVAIL]
stds  = [summary[m]["std_storage_red"]  * 100       for m in METHODS_AVAIL]
x     = np.arange(len(METHODS_AVAIL))
bars  = ax.bar(x, means, yerr=stds, capsize=5, color=meth_colors, alpha=0.82,
               edgecolor="black", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(meth_labels, fontsize=11)
ax.set_ylabel("Storage Reduction (%)", fontsize=12)
ax.set_ylim(0, 110)
ax.set_title("Mean Storage Reduction per Predictor\n(test set, Mozilla telemetry)",
             fontsize=12)
ax.axhline(100, color="gray", ls="--", lw=0.8)
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 1.5,
            f"{mean:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
p1 = os.path.join(REPLAY_PLOTS, "replay_storage_reduction.png")
fig.savefig(p1, dpi=180)
plt.close(fig)
printf(f"  Saved: {p1}")

# ── Plot 2: Trigger Frequency (bar) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
tf_m  = [summary[m]["mean_trigger_freq"] * 1000 for m in METHODS_AVAIL]  # per 1000 steps
tf_s  = [summary[m]["std_trigger_freq"]  * 1000 for m in METHODS_AVAIL]
bars  = ax.bar(x, tf_m, yerr=tf_s, capsize=5, color=meth_colors, alpha=0.82,
               edgecolor="black", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(meth_labels, fontsize=11)
ax.set_ylabel("Trigger Frequency (activations per 1000 steps)", fontsize=11)
ax.set_title("Mean Trace-On Trigger Frequency per Predictor\n(test set, Mozilla telemetry)",
             fontsize=12)
for bar, mean, std in zip(bars, tf_m, tf_s):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
p2 = os.path.join(REPLAY_PLOTS, "replay_trigger_frequency.png")
fig.savefig(p2, dpi=180)
plt.close(fig)
printf(f"  Saved: {p2}")

# ── Plot 3: Alert Coverage (bar) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
cov_m = [summary[m].get("mean_alert_cov", 0) or 0 for m in METHODS_AVAIL]
cov_s = [summary[m].get("std_alert_cov",  0) or 0 for m in METHODS_AVAIL]
bars  = ax.bar(x, [c * 100 for c in cov_m],
               yerr=[s * 100 for s in cov_s],
               capsize=5, color=meth_colors, alpha=0.82,
               edgecolor="black", linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels(meth_labels, fontsize=11)
ax.set_ylabel("Alert Coverage / Recall (%)", fontsize=12)
ax.set_ylim(0, 115)
ax.set_title("Mean Alert Coverage (Recall) per Predictor\n"
             "(regression alerts in test set, Mozilla telemetry)", fontsize=12)
for bar, mean, std in zip(bars, cov_m, cov_s):
    if mean is not None:
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean * 100 + std * 100 + 1.5,
                f"{mean*100:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
p3 = os.path.join(REPLAY_PLOTS, "replay_alert_coverage.png")
fig.savefig(p3, dpi=180)
plt.close(fig)
printf(f"  Saved: {p3}")

# ── Plot 4: Efficiency Frontier (storage_red vs alert_coverage scatter) ───────
fig, ax = plt.subplots(figsize=(7, 5))
for m in METHODS_AVAIL:
    df_m   = replay_df[(replay_df["method"] == m) & replay_df["Alert_Coverage"].notna()]
    ax.scatter(df_m["Storage_Red"] * 100,
               df_m["Alert_Coverage"] * 100,
               label=METHOD_LABELS[m],
               color=METHOD_COLORS[m],
               alpha=0.55, s=28, edgecolors="none")
# Method centroids
for m in METHODS_AVAIL:
    a = summary[m]
    cx = a["mean_storage_red"] * 100
    cy = (a.get("mean_alert_cov") or 0) * 100
    if cy > 0:
        ax.scatter(cx, cy, color=METHOD_COLORS[m],
                   s=120, marker="D", edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(m, (cx, cy), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)
ax.set_xlabel("Storage Reduction (%)", fontsize=12)
ax.set_ylabel("Alert Coverage / Recall (%)", fontsize=12)
ax.set_title("Efficiency Frontier: Storage Reduction vs Alert Coverage\n"
             "(Mozilla telemetry, test set)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 115)
# Ideal corner annotation
ax.annotate("Ideal", xy=(100, 100), fontsize=8, color="gray", style="italic")
plt.tight_layout()
p4 = os.path.join(REPLAY_PLOTS, "replay_efficiency_frontier.png")
fig.savefig(p4, dpi=180)
plt.close(fig)
printf(f"  Saved: {p4}")

# ── Plot 5: CDF of Storage Reduction per method ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in METHODS_AVAIL:
    vals = np.sort(replay_df[replay_df["method"] == m]["Storage_Red"].dropna().values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals * 100, cdf * 100,
            label=METHOD_LABELS[m],
            color=METHOD_COLORS[m], lw=2.0)
ax.set_xlabel("Storage Reduction (%)", fontsize=12)
ax.set_ylabel("Cumulative % of Signatures", fontsize=12)
ax.set_title("CDF of Per-Signature Storage Reduction\n(Mozilla telemetry, test set)",
             fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 103)
ax.axvline(50, color="gray", ls=":", lw=0.8, label="50%")
plt.tight_layout()
p5 = os.path.join(REPLAY_PLOTS, "replay_cdf_storage.png")
fig.savefig(p5, dpi=180)
plt.close(fig)
printf(f"  Saved: {p5}")

# ── Plot 6: Three-metric grouped bar chart (summary panel) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metrics = [
    ("mean_storage_red",  "std_storage_red",  "Storage Reduction (%)",           100),
    ("mean_trigger_freq", "std_trigger_freq",  "Trigger Freq (x1e-3 per step)", 1000),
    ("mean_alert_cov",    "std_alert_cov",     "Alert Coverage / Recall (%)",     100),
]
x = np.arange(len(METHODS_AVAIL))

for ax, (mk, sk, ylabel, scale) in zip(axes, metrics):
    m_vals = [(summary[m].get(mk) or 0) * scale for m in METHODS_AVAIL]
    s_vals = [(summary[m].get(sk) or 0) * scale for m in METHODS_AVAIL]
    ax.bar(x, m_vals, yerr=s_vals, capsize=4,
           color=meth_colors, alpha=0.82, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in METHODS_AVAIL], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel.split("(")[0].strip(), fontsize=11)
    ax.set_ylim(0, max(m_vals) * 1.30 + 1)

axes[0].set_ylim(0, 110)
axes[2].set_ylim(0, 110)
fig.suptitle("Replay Experiment Summary -- Mozilla Telemetry (Test Set)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
p6 = os.path.join(REPLAY_PLOTS, "replay_summary_panel.png")
fig.savefig(p6, dpi=180)
plt.close(fig)
printf(f"  Saved: {p6}")

# ── Plot 7: Sample timeline -- best signature for one method ───────────────────
# Pick signature with alerts and all 4 methods available
timeline_plot_path = None
timeline_sig = None

sigs_with_alerts = replay_df[replay_df["N_Alerts_Test"] > 0]["signature_id"].unique()
# Find a sig covered by all available methods and with reasonable T_test
cand_sigs = None
for sig in sigs_with_alerts:
    check = all(
        sig in raw[m]["signature_id"].values
        for m in METHODS_AVAIL
    )
    if check:
        cand_sigs = sig
        break

if cand_sigs is not None:
    sig = cand_sigs
    timeline_sig = sig
    fig, axes = plt.subplots(len(METHODS_AVAIL), 1,
                              figsize=(13, 3.0 * len(METHODS_AVAIL)),
                              sharex=True)
    if len(METHODS_AVAIL) == 1:
        axes = [axes]

    # Load alert t-indices for this sig (use cache)
    alert_t_all = _alert_cache.get(sig, [])

    for ax, m in zip(axes, METHODS_AVAIL):
        df_m = raw[m]
        sub  = df_m[df_m["signature_id"] == sig].sort_values("t")
        if sub.empty:
            ax.set_visible(False)
            continue

        # Infer T_train
        if "T_train" in sub.columns and sub["T_train"].notna().any():
            T_train = int(sub["T_train"].dropna().iloc[0])
        else:
            T_train = max(2, int(TRAIN_RATIO * (sub["t"].max() + 1)))

        test_sub = sub[sub["t"] >= T_train]
        t_arr    = test_sub["t"].to_numpy()
        dec_arr  = test_sub["decision"].fillna(0).astype(int).to_numpy()
        act_arr  = test_sub["actual"].to_numpy(dtype=float)

        # Shade trace-on windows
        ax.fill_between(t_arr, 0, 1, where=(dec_arr == 1),
                        alpha=0.20, color=METHOD_COLORS[m],
                        transform=ax.get_xaxis_transform(),
                        label="Trace ON")
        # Plot actual value
        ax.plot(t_arr, act_arr, lw=1.0, color="black", alpha=0.8, label="Actual")
        # Mark alert events
        alert_t_test = [ta for ta in alert_t_all if ta >= T_train]
        for ta in alert_t_test:
            ax.axvline(ta, color="red", lw=1.5, ls="--", alpha=0.85)
        ax.set_ylabel(m, fontsize=10)
        if m == METHODS_AVAIL[0]:
            ax.set_title(
                f"Signature {sig} -- Adaptive Trace-On Windows (4 Predictors)\n"
                f"Red dashes = regression alerts; shaded = trace-on",
                fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.5)

    axes[-1].set_xlabel("Time step (test set)", fontsize=11)
    plt.tight_layout()
    timeline_plot_path = os.path.join(REPLAY_PLOTS, "replay_timeline_sample.png")
    fig.savefig(timeline_plot_path, dpi=150)
    plt.close(fig)
    printf(f"  Saved timeline: {timeline_plot_path}")
else:
    printf("  [SKIP] No suitable candidate signature for timeline plot")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 -- Save JSON summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 -- Saving JSON summary")
print("=" * 65)

full_summary = {
    "experiment"  : "end_to_end_replay",
    "description" : (
        "Adaptive trace-on replay experiment on Mozilla Firefox-Android "
        "telemetry. Simulates which data points would be recorded if the "
        "detector's decision output gated the sensor. Measures storage "
        "reduction, trigger frequency, and alert coverage across the four "
        "predictors on the chronological test set."
    ),
    "settings"    : {
        "dataset"          : "Mozilla Firefox-Android Treeherder telemetry",
        "methods"          : METHODS_AVAIL,
        "tolerance_tau"    : TOL,
        "train_ratio"      : TRAIN_RATIO,
        "total_signatures" : int(replay_df["signature_id"].nunique()),
    },
    "aggregate_metrics" : summary,
    "plots_saved"       : {
        "storage_reduction"   : p1,
        "trigger_frequency"   : p2,
        "alert_coverage"      : p3,
        "efficiency_frontier" : p4,
        "cdf_storage"         : p5,
        "summary_panel"       : p6,
        "timeline_sample"     : timeline_plot_path,
    },
}

summary_path = os.path.join(REPLAY_DIR, "replay_summary.json")
with open(summary_path, "w") as f:
    json.dump(json_sanitize(full_summary), f, indent=2)
printf(f"Saved -> {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 -- Generate PDF report (fpdf2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 -- Generating PDF report (fpdf2)")
print("=" * 65)

def fmt(v, dec=2):
    if v is None or (isinstance(v, float) and (v != v)):
        return "---"
    return f"{v:.{dec}f}"

def pct(v, dec=1):
    if v is None or (isinstance(v, float) and (v != v)):
        return "---"
    return f"{v * 100:.{dec}f}%"


_PDF_REPLS = {
    '\u2014': '--',   '\u2013': '-',    '\u2012': '-',    '\u2011': '-',
    '\u00b1': '+/-',  '\u00d7': 'x',    '\u2212': '-',    '\u2022': '*',
    '\u00e9': 'e',    '\u00ef': 'i',    '\u00e0': 'a',    '\u00e8': 'e',
    '\u00ea': 'e',    '\u00f9': 'u',    '\u00fb': 'u',    '\u00e2': 'a',
    '\u2019': "'",    '\u2018': "'",    '\u201c': '"',    '\u201d': '"',
    '\u03b1': 'alpha','\u03b2': 'beta', '\u03c4': 'tau',  '\u03c3': 'sigma',
    '\u03b8': 'theta','\u2264': '<=',   '\u2265': '>=',   '\u00b2': '^2',
    '\u00b3': '^3',   '\u2070': '^0',   '\u00b9': '^1',   '\u207b': '^-',
    '\u03b4': 'delta','\u03b8': 'theta','\u2208': 'in',   '\u03c9': 'omega',
    '\u00e4': 'a',    '\u00f6': 'o',    '\u00fc': 'u',    '\u00df': 'ss',
    '\u00e6': 'ae',   '\u00f8': 'o',    '\u00e5': 'a',    '\u00e7': 'c',
    '\u00e3': 'a',    '\u00f5': 'o',    '\u00f1': 'n',    '\u00ed': 'i',
    '\u00e1': 'a',    '\u00f3': 'o',    '\u00fa': 'u',    '\u00fe': 'th',
    '\u00c6': 'AE',   '\u00c5': 'A',    '\u00d8': 'O',    '\u00c9': 'E',
    '\u00c0': 'A',    '\u00c1': 'A',    '\u00c4': 'A',    '\u00c7': 'C',
    '\u00d1': 'N',    '\u00d3': 'O',    '\u00d6': 'O',    '\u00da': 'U',
    '\u00dc': 'U',    '\u00c8': 'E',    '\u00ca': 'E',    '\u00cb': 'E',
    '\u00cc': 'I',    '\u00cd': 'I',    '\u00ce': 'I',    '\u00cf': 'I',
    '\u2026': '...',  '\u2030': 'o/oo', '\u00ac': 'not',  '\u00b7': '.',
}

def _sanitize(s: str) -> str:
    """Replace characters outside Helvetica/CP-1252 range for fpdf2."""
    for ch, repl in _PDF_REPLS.items():
        s = s.replace(ch, repl)
    # Strip any remaining non-latin-1 characters
    return s.encode('latin-1', errors='replace').decode('latin-1')


class ReplayReport(FPDF):
    TITLE_COLOR  = (31, 73, 125)
    HEAD_COLOR   = (68, 114, 196)
    ALT_ROW      = (242, 247, 255)
    LINE_COLOR   = (180, 190, 210)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, _sanitize("Adaptive Trace-On Replay Experiment -- Mozilla Telemetry"),
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(*self.TITLE_COLOR)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, _sanitize(title), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(*self.HEAD_COLOR)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, _sanitize(f"  {title}"), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(2)

    def kv_line(self, key, value, size=10):
        """Print a key-value pair on one line."""
        self.set_font("Helvetica", "B", size)
        self.cell(75, 6, _sanitize(key), new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", size)
        self.cell(0,  6, _sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.HEAD_COLOR)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 6, _sanitize(col), border=1, align="C", fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, cells, widths, alt=False):
        self.set_font("Helvetica", "", 9)
        if alt:
            self.set_fill_color(*self.ALT_ROW)
        else:
            self.set_fill_color(255, 255, 255)
        for cell, w in zip(cells, widths):
            self.cell(w, 5.5, _sanitize(str(cell)), border=1, align="C", fill=True)
        self.ln()

    def insert_image(self, path, w=170, caption=None):
        if path and os.path.exists(path):
            # Check if enough space, else new page
            if self.get_y() + w * 0.6 + 20 > self.h - 20:
                self.add_page()
            x = (self.w - w) / 2
            self.image(path, x=x, w=w)
            if caption:
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(80, 80, 80)
                self.cell(0, 5, _sanitize(caption), align="C",
                          new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)
            self.ln(3)


# ── Build the PDF ─────────────────────────────────────────────────────────────
pdf = ReplayReport(orientation="P", unit="mm", format="A4")
pdf.set_margins(15, 15, 15)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── Cover block ───────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 17)
pdf.set_text_color(*ReplayReport.TITLE_COLOR)
pdf.ln(4)
pdf.multi_cell(0, 9, "Adaptive Trace-On Replay Experiment\nMozilla Firefox-Android Telemetry",
               align="C")
pdf.ln(3)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "End-to-End Reproducible Experiment -- Journal Paper Supplement",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(6)

# ── 1. Overview ───────────────────────────────────────────────────────────────
pdf.chapter_title("1.  Experiment Overview")
pdf.body_text(
    "This report documents a reproducible end-to-end replay experiment that "
    "evaluates four classical adaptive anomaly detectors as gatekeepers for a "
    "performance-telemetry collection system. In each simulated scenario the "
    "detector's binary decision output (decision = 1 => trace ON) determines "
    "which data points are recorded. Three metrics are measured consistently "
    "across all predictors on the chronological test portion of each time "
    "series:\n\n"
    "  (1) Storage Reduction  -- fraction of test-set points NOT recorded.\n"
    "  (2) Trigger Frequency  -- number of trace-on activations per test\n"
    "                            time step (sensor switching overhead).\n"
    "  (3) Alert Coverage     -- fraction of ground-truth regression alerts\n"
    "                            covered by a trace-on interval (+/-tau pushes)."
)

pdf.section_title("Dataset and Experimental Settings")
settings = full_summary["settings"]
pdf.kv_line("Dataset:",         settings["dataset"])
pdf.kv_line("Predictors:",      ", ".join(settings["methods"]))
pdf.kv_line("Total signatures:", str(settings["total_signatures"]))
pdf.kv_line("Alert tolerance tau:", f"+/-{settings['tolerance_tau']} pushes")
pdf.kv_line("Train / Test split:", f"{int(settings['train_ratio']*100)}% / {int((1-settings['train_ratio'])*100)}%")
pdf.ln(2)

pdf.section_title("Algorithm 3 Trace-On Logic")
pdf.body_text(
    "At each test-set time step t:\n"
    "  1. Compute forecast y_hat_t using the chosen predictor.\n"
    "  2. Compute residual delta_t = |y_t - y_hat_t|.\n"
    "  3. Compute dynamic threshold Theta_t = max(Theta_min, k_sigma * std(delta_{t-w:t})).\n"
    "  4. Update anomaly score: s_t = beta * 1[delta_t > Theta_t] + (1-beta)*s_{t-1}.\n"
    "  5. Emit decision_t = 1 (trace ON) if s_t >= theta_anomaly, else 0 (trace OFF).\n\n"
    "Storage overhead is determined by the density of decision=1 steps. "
    "Trigger frequency counts how often the system switches from OFF to ON, "
    "which represents the cost of activating the sensor. Alert coverage "
    "measures whether real regression events were captured."
)

# ── 2. Aggregate Results ──────────────────────────────────────────────────────
pdf.chapter_title("2.  Aggregate Results")

# Table -- three-metric summary
col_w = [42, 32, 32, 32, 32]
headers = ["Method", "Storage Red.", "Trigger Freq.", "Alert Cov.", "Precision"]
pdf.table_header(headers, col_w)
for i, m in enumerate(METHODS_AVAIL):
    a = summary[m]
    cells = [
        METHOD_LABELS[m],
        pct(a["mean_storage_red"]),
        fmt(a["mean_trigger_freq"] * 1000, 3) + "e-3",
        pct(a.get("mean_alert_cov")),
        fmt(a.get("mean_precision"), 3),
    ]
    pdf.table_row(cells, col_w, alt=(i % 2 == 0))
pdf.ln(3)

pdf.body_text(
    "Storage Reduction: fraction of test-set points suppressed (higher is better "
    "for storage savings). Trigger Frequency (x10^-3): activations per step "
    "(lower means fewer sensor switches, less overhead). Alert Coverage: recall "
    "of regression alerts on the test set (higher means more regressions caught). "
    "Precision: fraction of detected intervals that coincide with a real alert."
)

# Extended table -- std and median
pdf.section_title("Storage Reduction -- Detailed Statistics")
col_w2 = [42, 25, 25, 25, 25, 28]
headers2 = ["Method", "Mean", "Std", "Median", "n_activate", "Burst Ratio"]
pdf.table_header(headers2, col_w2)
for i, m in enumerate(METHODS_AVAIL):
    a = summary[m]
    cells = [
        METHOD_LABELS[m],
        pct(a["mean_storage_red"]),
        pct(a["std_storage_red"]),
        pct(a.get("median_storage_red")),
        fmt(a["mean_activations"], 1),
        fmt(a["mean_burst_ratio"], 3),
    ]
    pdf.table_row(cells, col_w2, alt=(i % 2 == 0))
pdf.ln(3)

pdf.body_text(
    "Burst Ratio: fraction of trace-on time that occurs in contiguous runs "
    "longer than one step. A high burst ratio means the detector tends to "
    "stay ON for extended periods rather than flickering, which is operationally "
    "preferable as it reduces the number of start/stop events."
)

# ── 3. Figures ────────────────────────────────────────────────────────────────
pdf.chapter_title("3.  Figures")

pdf.insert_image(p1, w=155,
    caption="Figure 1 – Mean storage reduction per predictor (mean +/- std, test set).")
pdf.insert_image(p2, w=155,
    caption="Figure 2 – Mean trigger frequency per predictor (activations per 1000 test steps).")
pdf.insert_image(p3, w=155,
    caption="Figure 3 – Mean alert coverage (recall) per predictor on the test set.")
pdf.insert_image(p4, w=155,
    caption="Figure 4 – Efficiency frontier: each dot is one signature. "
            "Diamond markers show per-method centroids. "
            "Top-right corner is ideal (high coverage, high storage savings).")
pdf.insert_image(p5, w=155,
    caption="Figure 5 – CDF of per-signature storage reduction. "
            "A curve shifted to the right indicates higher storage savings.")
pdf.insert_image(p6, w=170,
    caption="Figure 6 – Summary panel: all three metrics side by side.")
if timeline_plot_path and os.path.exists(timeline_plot_path):
    pdf.insert_image(timeline_plot_path, w=170,
        caption=f"Figure 7 – Sample timeline for signature {timeline_sig}. "
                "Shaded regions = trace ON. Red dashes = regression alert events.")

# ── 4. Per-Method Signature-Level Tables ──────────────────────────────────────
pdf.chapter_title("4.  Per-Signature Results (Top 15 by Storage Reduction)")
col_w3 = [35, 20, 20, 20, 25, 25, 23]
headers3 = ["Signature", "T_test", "Stor.Red.", "Trg.Freq.", "Activations", "Cov.", "Precision"]
for m in METHODS_AVAIL:
    pdf.section_title(f"Method: {METHOD_LABELS[m]}")
    df_m = replay_df[replay_df["method"] == m].nlargest(15, "Storage_Red")
    pdf.table_header(headers3, col_w3)
    for i, (_, r) in enumerate(df_m.iterrows()):
        cov_str = pct(r["Alert_Coverage"]) if r["Alert_Coverage"] is not None else "N/A"
        cells = [
            str(r["signature_id"])[:14],
            str(int(r["T_test"])),
            pct(r["Storage_Red"]),
            fmt(r["Trigger_Freq"] * 1000, 3),
            str(int(r["N_Activations"])),
            cov_str,
            fmt(r["Precision"], 3),
        ]
        pdf.table_row(cells, col_w3, alt=(i % 2 == 0))
    pdf.ln(4)

# ── 5. Discussion ─────────────────────────────────────────────────────────────
pdf.chapter_title("5.  Discussion and Reproducibility")
pdf.body_text(
    "Reproducibility: This experiment is fully deterministic given the "
    "pre-computed detector output CSVs (mozilla_results*.csv) produced by "
    "run_pipeline.py. All random seeds are fixed by the deterministic nature "
    "of the Algorithm 3 controller. Re-running 11_replay_experiment.py "
    "from the workspace root will reproduce all tables, plots, and this PDF "
    "identically.\n\n"
    "Storage vs Coverage Trade-off: A high storage reduction with high alert "
    "coverage is the ideal operating point. Detectors that achieve both "
    "demonstrate that the adaptive gating strategy correctly identifies "
    "anomalous windows without recording excessive normal operation data.\n\n"
    "Trigger Frequency: Low trigger frequency is desirable in practice because "
    "each ON activation involves system overhead (e.g., starting a logger, "
    "allocating buffers). Methods that activate infrequently but cover all "
    "alerts are particularly suitable for resource-constrained deployments.\n\n"
    "Limitations: Alert coverage is limited to the test-set portion of each "
    "time series (first 70% is training). Signatures without ground-truth "
    "regression alerts in the test set contribute to storage/trigger metrics "
    "but are excluded from alert-coverage aggregation (reported separately "
    "as 'n_alerted_with_cov' in replay_summary.json)."
)

# ── 6. File Index ─────────────────────────────────────────────────────────────
pdf.chapter_title("6.  Output File Index")
files = [
    ("results/replay/replay_results.csv",       "Per-signature x method replay metrics"),
    ("results/replay/replay_summary.json",      "Aggregate statistics (JSON)"),
    ("results/replay/plots/replay_storage_reduction.png", "Figure 1 -- bar chart"),
    ("results/replay/plots/replay_trigger_frequency.png", "Figure 2 -- bar chart"),
    ("results/replay/plots/replay_alert_coverage.png",    "Figure 3 -- bar chart"),
    ("results/replay/plots/replay_efficiency_frontier.png", "Figure 4 -- scatter"),
    ("results/replay/plots/replay_cdf_storage.png",       "Figure 5 -- CDF"),
    ("results/replay/plots/replay_summary_panel.png",     "Figure 6 -- summary panel"),
    ("results/replay/plots/replay_timeline_sample.png",   "Figure 7 -- timeline"),
    ("results/replay/replay_report.pdf",        "This PDF report"),
]

col_wf = [95, 80]
pdf.table_header(["File", "Description"], col_wf)
for i, (fp, desc) in enumerate(files):
    pdf.table_row([fp, desc], col_wf, alt=(i % 2 == 0))

pdf.ln(5)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 5, "Generated automatically by Code/11_replay_experiment.py -- March 2026",
         align="C", new_x="LMARGIN", new_y="NEXT")

# ── Save PDF ──────────────────────────────────────────────────────────────────
pdf_path = os.path.join(REPLAY_DIR, "replay_report.pdf")
pdf.output(pdf_path)
printf(f"\nPDF report saved -> {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("REPLAY EXPERIMENT COMPLETE")
print("=" * 65)
print(f"  results directory   : {REPLAY_DIR}")
print(f"  Per-sig CSV         : {replay_csv}")
print(f"  Summary JSON        : {summary_path}")
print(f"  PDF report          : {pdf_path}")
print(f"  Plots               : {REPLAY_PLOTS}")
print("=" * 65)

