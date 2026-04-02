# ==========================
# 13_bugzilla_validation.py
# Second validation source: Bugzilla-linked regression ground truth.
#
# Motivation
# ----------
# The primary evaluation uses Treeherder's performance alert system:
#   GT_treeherder = alerts where single_alert_is_regression == True
# This script introduces two alternative ground-truth definitions based on
# Bugzilla bug metadata and compares detector performance under all three GTs:
#
#   GT_treeherder  — baseline: any alert Treeherder classifies as regression
#   GT_bz_filed    — alerts where a Bugzilla bug was filed (any resolution)
#                    AND the bug has "regression" in its keywords
#   GT_bz_fixed    — restrictive: same as GT_bz_filed but only FIXED bugs
#                    (a human triager confirmed and resolved the regression)
#
# GT_bz_filed may include alerts not flagged by Treeherder but confirmed by
# a developer filing a Bugzilla report; GT_bz_fixed is the highest-confidence
# label set.
#
# The script evaluates all four classical methods on the 214 firefox-android
# signatures using each GT definition and reports precision / recall / F1 / FAR
# side by side.
#
# Usage:
#   python Code/13_bugzilla_validation.py
#
# Outputs:
#   results/bugzilla_validation.csv
#   results/bugzilla_validation_summary.json
#   results/plots/bugzilla_gt_comparison.png
# ==========================

import json, math, os
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
BUGS_CSV    = os.path.join(DATA_DIR, "bugs_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")

METHODS = {
    "ARIMA": os.path.join(RESULTS_DIR, "mozilla_results.csv"),
    "LAST":  os.path.join(RESULTS_DIR, "mozilla_results_LAST.csv"),
    "SMA":   os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"),
    "EWMA":  os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"),
}

REPO_NAME   = "firefox-android"
DEFAULT_TAU = 5   # ±5-step alert-match tolerance

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
    valid = ts_series.dropna()
    if valid.empty:
        return None
    try:
        return int((valid - target_ts).abs().idxmin())
    except Exception:
        return None


def load_ts_timestamps(sig: str, t_total: int) -> pd.Series:
    """Return the first t_total push_timestamps from the sorted, filtered timeseries."""
    path = os.path.join(TS_BASE, f"{sig}_timeseries_data.csv")
    if not os.path.exists(path):
        return pd.Series(dtype="datetime64[ns]")
    try:
        try:
            df = pd.read_csv(path, usecols=["push_timestamp", "value"],
                             encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, usecols=["push_timestamp", "value"],
                             encoding="latin-1")
        if "push_timestamp" in df.columns:
            df["push_timestamp"] = pd.to_datetime(df["push_timestamp"], errors="coerce")
            df = df.dropna(subset=["push_timestamp"]).sort_values("push_timestamp")
        df = df.iloc[:t_total]   # only the rows the evaluation used
        return df["push_timestamp"].reset_index(drop=True) if "push_timestamp" in df.columns \
            else pd.Series(dtype="datetime64[ns]")
    except Exception:
        return pd.Series(dtype="datetime64[ns]")


def evaluate_with_gt(df_res: pd.DataFrame, alert_cache: dict, tau: int) -> pd.DataFrame:
    """
    Compute per-signature precision/recall/F1/FAR using the provided alert_cache
    (dict: sig_str -> sorted list of t-indices in test period).
    Returns a DataFrame with columns: signature_id, precision, recall, f1, far,
    n_alerts, n_detections.
    """
    rows = []
    for sig_raw, grp in df_res.groupby("signature_id"):
        sig   = normalize_sig(sig_raw)
        grp   = grp.sort_values("t")
        t_train = int(grp["T_train"].iloc[0])
        t_total = int(grp["T_total"].iloc[0])
        T_test  = max(1, t_total - t_train)

        # Test-period decisions
        test_sub = grp[grp["t"] >= t_train]
        det_ts   = sorted(test_sub.loc[test_sub["decision"] == 1, "t"].tolist())
        intervals = extract_intervals(det_ts)
        n_ivs     = len(intervals)

        alert_t = alert_cache.get(sig, [])

        tp_iv = sum(
            1 for (s, e) in intervals
            if any((s - tau) <= ta <= (e + tau) for ta in alert_t)
        )
        fp_iv     = n_ivs - tp_iv
        precision = tp_iv / n_ivs if n_ivs > 0 else 0.0

        tp_al = sum(
            1 for ta in alert_t
            if any((s - tau) <= ta <= (e + tau) for (s, e) in intervals)
        )
        recall = tp_al / len(alert_t) if alert_t else 0.0
        f1     = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
        far    = fp_iv / T_test

        rows.append({
            "signature_id": sig,
            "precision"   : precision,
            "recall"      : recall,
            "f1"          : f1,
            "far"         : far,
            "n_alerts"    : len(alert_t),
            "n_detections": n_ivs,
        })
    return pd.DataFrame(rows)


def agg_metrics(ev_df: pd.DataFrame) -> dict:
    """Aggregate metrics over all signatures (mean ± std)."""
    m = {}
    for metric in ("precision", "recall", "f1", "far"):
        vals = ev_df[metric].dropna().tolist()
        m[f"mean_{metric}"]  = round(float(np.mean(vals)), 4) if vals else None
        m[f"std_{metric}"]   = round(float(np.std(vals)),  4) if vals else None
    m["n_signatures"]         = len(ev_df)
    m["n_sigs_with_alerts"]   = int((ev_df["n_alerts"] > 0).sum())
    m["n_detections_total"]   = int(ev_df["n_detections"].sum())
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load and enrich bugs data
# ──────────────────────────────────────────────────────────────────────────────
print("Loading bugs data ...")
bugs_df = pd.read_csv(BUGS_CSV, usecols=["id", "keywords", "resolution"])
bugs_df["has_reg_kw"] = bugs_df["keywords"].fillna("").str.contains(
    r"\bregression\b", case=False, regex=True
)
bz_filed_ids = set(bugs_df.loc[bugs_df["has_reg_kw"], "id"].astype(int).tolist())
bz_fixed_ids = set(bugs_df.loc[
    bugs_df["has_reg_kw"] & (bugs_df["resolution"] == "FIXED"), "id"
].astype(int).tolist())
print(f"  Bugzilla bugs with 'regression' keyword:         {len(bz_filed_ids)}")
print(f"  Bugzilla bugs with 'regression' + FIXED:         {len(bz_fixed_ids)}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Load alerts and classify under each GT definition
# ──────────────────────────────────────────────────────────────────────────────
print("\nLoading alerts and building ground-truth sets ...")
alerts_df = pd.read_csv(ALERTS_CSV)
alerts_df["sig_norm"]       = alerts_df["signature_id"].apply(normalize_sig)
alerts_df["push_timestamp"] = pd.to_datetime(alerts_df["push_timestamp"], errors="coerce")
alerts_df = alerts_df.dropna(subset=["push_timestamp"])

# Restrict to evaluation repository
alerts_repo = alerts_df[alerts_df["alert_summary_repository"] == REPO_NAME].copy()
alerts_repo["bug_id"] = pd.to_numeric(
    alerts_repo["alert_summary_bug_number"], errors="coerce"
).astype("Int64")

# GT masks
alerts_repo["in_gt_treeherder"] = alerts_repo["single_alert_is_regression"] == True
alerts_repo["in_gt_bz_filed"]   = alerts_repo["bug_id"].notna() & alerts_repo["bug_id"].isin(bz_filed_ids)
alerts_repo["in_gt_bz_fixed"]   = alerts_repo["bug_id"].notna() & alerts_repo["bug_id"].isin(bz_fixed_ids)

print(f"  Alerts in GT_treeherder: {alerts_repo['in_gt_treeherder'].sum()}")
print(f"  Alerts in GT_bz_filed:   {alerts_repo['in_gt_bz_filed'].sum()}")
print(f"  Alerts in GT_bz_fixed:   {alerts_repo['in_gt_bz_fixed'].sum()}")

# Signatures covered by each GT (within the 214 evaluation signatures)
eval_sigs = set(
    normalize_sig(s)
    for s in pd.read_csv(os.path.join(RESULTS_DIR, "mozilla_results.csv"))["signature_id"]
    .unique()
)
print(f"\n  Evaluation signatures: {len(eval_sigs)}")
for gt_col, label in [
    ("in_gt_treeherder", "GT_treeherder"),
    ("in_gt_bz_filed",   "GT_bz_filed"),
    ("in_gt_bz_fixed",   "GT_bz_fixed"),
]:
    covered = alerts_repo.loc[alerts_repo[gt_col] & alerts_repo["sig_norm"].isin(eval_sigs),
                               "sig_norm"].nunique()
    print(f"  {label:<20s}: {covered} eval-sigs with ≥1 GT alert")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Map alert timestamps -> t-indices per signature, per GT
# ──────────────────────────────────────────────────────────────────────────────
print("\nBuilding per-GT alert caches (mapping timestamps → t-indices) ...")

GT_COLS = {
    "GT_treeherder": "in_gt_treeherder",
    "GT_bz_filed"  : "in_gt_bz_filed",
    "GT_bz_fixed"  : "in_gt_bz_fixed",
}

# Collect {gt_name: {sig: {ts_list}}}
gt_ts_map: dict[str, dict[str, list]] = {gt: {} for gt in GT_COLS}
for gt_name, col in GT_COLS.items():
    for sig, grp in alerts_repo[alerts_repo[col]].groupby("sig_norm"):
        gt_ts_map[gt_name][sig] = sorted(grp["push_timestamp"].dropna().tolist())

# Map timestamps -> t-indices using per-signature timeseries CSVs (cached)
def build_alert_t_cache(ts_alert_map: dict, t_train_map: dict) -> dict:
    """Convert timestamp-based alerts to t-index lists, restricted to test period."""
    cache = {}
    for sig, ts_list in ts_alert_map.items():
        ts_series = _ts_cache.get(sig, pd.Series(dtype="datetime64[ns]"))
        t_train   = t_train_map.get(sig, 0)
        indices   = []
        for ats in ts_list:
            ti = closest_t(ts_series, ats)
            if ti is not None:
                indices.append(ti)
        # Keep only test-period alerts
        cache[sig] = sorted(ta for ta in set(indices) if ta >= t_train)
    return cache


# Build T_train map from ARIMA results (same for all methods)
df_arima       = pd.read_csv(METHODS["ARIMA"])
t_train_map: dict = {}
t_total_map: dict = {}
for sig_raw, grp in df_arima.groupby("signature_id"):
    sig_s = normalize_sig(sig_raw)
    t_train_map[sig_s] = int(grp["T_train"].iloc[0])
    t_total_map[sig_s] = int(grp["T_total"].iloc[0])

# Pre-load all timestamps ONCE (only push_timestamp+value columns, first T_total rows)
print("  Pre-loading timeseries timestamps for all signatures ...")
_ts_cache: dict = {}
for sig in sorted(t_train_map):
    _ts_cache[sig] = load_ts_timestamps(sig, t_total_map.get(sig, 9999))
print(f"  Loaded {len(_ts_cache)} timeseries.")

print("  Mapping alert timestamps to t-indices ...")
gt_caches: dict = {}
for gt_name, ts_map in gt_ts_map.items():
    gt_caches[gt_name] = build_alert_t_cache(ts_map, t_train_map)
    n_sigs  = len(gt_caches[gt_name])
    n_total = sum(len(v) for v in gt_caches[gt_name].values())
    print(f"  {gt_name:<20s}: {n_sigs} sigs, {n_total} test-period alert indices")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate each method under each GT
# ──────────────────────────────────────────────────────────────────────────────
print("\nEvaluating all method × GT combinations ...")
records = []

for method, csv_path in METHODS.items():
    if not os.path.exists(csv_path):
        print(f"  [WARN] {method}: file not found — skipping.")
        continue
    df_res = pd.read_csv(csv_path)
    print(f"  {method} ...")
    for gt_name, alert_cache in gt_caches.items():
        ev   = evaluate_with_gt(df_res, alert_cache, tau=DEFAULT_TAU)
        agg  = agg_metrics(ev)
        records.append({
            "method"   : method,
            "gt_source": gt_name,
            **agg,
        })
        print(f"    {gt_name:<20s}: P={agg['mean_precision']:.3f} "
              f"R={agg['mean_recall']:.3f} F1={agg['mean_f1']:.3f} "
              f"FAR={agg['mean_far']:.4f} "
              f"(sigs_with_alerts={agg['n_sigs_with_alerts']})")

results_df = pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Save outputs
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)

csv_out  = os.path.join(RESULTS_DIR, "bugzilla_validation.csv")
json_out = os.path.join(RESULTS_DIR, "bugzilla_validation_summary.json")

results_df.to_csv(csv_out, index=False)
print(f"\nSaved table → {csv_out}")

# JSON summary: nested by method → GT
summary: dict = {}
for _, row in results_df.iterrows():
    m  = row["method"]
    gt = row["gt_source"]
    if m not in summary:
        summary[m] = {}
    summary[m][gt] = {k: v for k, v in row.items() if k not in ("method", "gt_source")}

with open(json_out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary → {json_out}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Comparison bar-chart
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating comparison plot ...")

gt_labels = list(GT_COLS.keys())
method_list = results_df["method"].unique().tolist()
metrics_to_plot = ["mean_precision", "mean_recall", "mean_f1", "mean_far"]
metric_titles   = ["Precision", "Recall", "F1", "FAR"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4), sharey=False)
gt_colors = {"GT_treeherder": "#4C72B0", "GT_bz_filed": "#DD8452", "GT_bz_fixed": "#55A868"}
bar_w = 0.25
x = np.arange(len(method_list))

for ax, metric, title in zip(axes, metrics_to_plot, metric_titles):
    for i, gt in enumerate(gt_labels):
        vals = [
            results_df.loc[(results_df["method"] == m) & (results_df["gt_source"] == gt),
                            metric].values[0]
            if len(results_df.loc[(results_df["method"] == m) & (results_df["gt_source"] == gt)]) > 0
            else 0.0
            for m in method_list
        ]
        ax.bar(x + i * bar_w, vals, width=bar_w, label=gt,
               color=gt_colors.get(gt, f"C{i}"), alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(method_list, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, max(0.01, ax.get_ylim()[1]) * 1.15)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Score")
axes[0].legend(title="Ground truth", fontsize=8, title_fontsize=8)
fig.suptitle("Detector performance under three ground-truth definitions\n"
             "(firefox-android, τ=5)", fontsize=11, y=1.01)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "bugzilla_gt_comparison.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Plot saved → {plot_path}")

print("\n=== Summary (mean metrics across all signatures) ===")
pivot = results_df.pivot_table(
    index="method", columns="gt_source",
    values=["mean_precision", "mean_recall", "mean_f1", "mean_far"]
)
print(pivot.round(4).to_string())
print("\nDone.")
