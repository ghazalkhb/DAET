# ==========================
# 14_dual_operating_point.py
# Joint (β, θ) grid search to identify the F1-maximising operating point.
# Reports two distinct operating modes:
#
#   high-recall mode  — default (β=0.30, θ=0.30): maximises regression coverage
#   balanced/F1 mode  — (β*, θ*): argmax mean F1 across all methods × signatures
#
# Method
# ------
# The saved per-step results in mozilla_results*.csv store the raw `isAnomaly`
# binary flag (from the dynamic threshold) independently of β and θ.  Anomaly
# scores and binary decisions can therefore be recomputed for any (β, θ) pair
# without re-running the detectors:
#
#   score_t = β · isAnomaly_t + (1−β) · score_{t−1}   (score reset to 0 at T_train)
#   decision_t = 1  if score_t ≥ θ  else 0
#
# Grid:  β ∈ {0.10, 0.20, 0.30, 0.50, 0.70}
#        θ ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
#
# Alert matching uses the same τ = 5 tolerance as the primary evaluation.
#
# Usage:
#   python Code/14_dual_operating_point.py
#
# Outputs:
#   results/dual_operating_point.csv        — full grid (method, β, θ, P/R/F1/FAR)
#   results/dual_operating_point_summary.json — two-operating-point comparison
#   results/plots/beta_theta_f1_heatmap.png  — 4-panel F1 heatmap (one per method)
# ==========================

import json, math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "Data")
TS_BASE     = os.path.join(DATA_DIR, "timeseries-data", "firefox-android")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")

METHODS = {
    "ARIMA": os.path.join(RESULTS_DIR, "mozilla_results.csv"),
    "LAST":  os.path.join(RESULTS_DIR, "mozilla_results_LAST.csv"),
    "SMA":   os.path.join(RESULTS_DIR, "mozilla_results_SMA.csv"),
    "EWMA":  os.path.join(RESULTS_DIR, "mozilla_results_EWMA.csv"),
}

REPO_NAME = "firefox-android"

# Search grid
BETA_GRID      = [0.10, 0.20, 0.30, 0.50, 0.70]
THETA_GRID     = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Baseline operating point (high-recall mode)
BASELINE_BETA  = 0.30
BASELINE_THETA = 0.30

DEFAULT_TAU    = 5   # ±5-step alert-match tolerance

# ──────────────────────────────────────────────────────────────────────────────
# Helpers (identical to 10_ablation.py for consistency)
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


def closest_t(ts_series, target_ts):
    valid = ts_series.dropna()
    if valid.empty:
        return None
    try:
            return int((valid - target_ts).abs().idxmin())
    except Exception:
        return None


def load_ts_timestamps(sig: str, t_total: int) -> pd.Series:
    """Return the first t_total push_timestamps from the sorted timeseries CSV."""
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
        return (df["push_timestamp"].reset_index(drop=True)
                if "push_timestamp" in df.columns
                else pd.Series(dtype="datetime64[ns]"))
    except Exception:
        return pd.Series(dtype="datetime64[ns]")


# ──────────────────────────────────────────────────────────────────────────────
# Build alert cache (test-period t-indices per signature)
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

# Collect alert timestamps per sig
alert_ts_map: dict = {}
for sig, grp in alerts_reg.groupby("sig_norm"):
    alert_ts_map[sig] = sorted(grp["push_timestamp"].dropna().tolist())

# Load ARIMA results to get T_train and T_total per sig
df_arima    = pd.read_csv(METHODS["ARIMA"])
t_train_map: dict = {}
t_total_map: dict = {}
for sig_raw, grp in df_arima.groupby("signature_id"):
    sig_s = normalize_sig(sig_raw)
    t_train_map[sig_s] = int(grp["T_train"].iloc[0])
    t_total_map[sig_s] = int(grp["T_total"].iloc[0])

# Pre-load all timestamps ONCE using bounded, minimal-column reads
print("Pre-loading timeseries timestamps for all signatures ...")
_ts_cache: dict = {}
for sig in sorted(t_train_map):
    _ts_cache[sig] = load_ts_timestamps(sig, t_total_map.get(sig, 9999))
print(f"  Loaded {len(_ts_cache)} timeseries.")

print("Building alert t-index cache ...")
alert_cache: dict = {}
for sig in t_train_map:
    ts         = _ts_cache.get(sig, pd.Series(dtype="datetime64[ns]"))
    t_train    = t_train_map[sig]
    raw_ts     = alert_ts_map.get(sig, [])
    indices    = []
    for ats in raw_ts:
        ti = closest_t(ts, ats)
        if ti is not None:
            indices.append(ti)
    alert_cache[sig] = sorted(ta for ta in set(indices) if ta >= t_train)

n_sigs_with_alerts = sum(1 for v in alert_cache.values() if v)
print(f"  {len(alert_cache)} signatures cached, "
      f"{n_sigs_with_alerts} with ≥1 test-period alert.")


# ──────────────────────────────────────────────────────────────────────────────
# Efficient per-method evaluation across the full (β, θ) grid
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_grid_for_method(df_res: pd.DataFrame, tau: int) -> list[dict]:
    """
    For each (beta, theta) pair, recompute decisions and evaluate metrics
    over all signatures in df_res.
    Returns a list of dicts: {beta, theta, mean_P, mean_R, mean_F1, mean_FAR}.
    """
    # Pre-group by signature_id once
    groups: list[tuple[str, pd.DataFrame]] = []
    for sig_raw, grp in df_res.groupby("signature_id"):
        sig     = normalize_sig(sig_raw)
        grp     = grp.sort_values("t").reset_index(drop=True)
        t_train = int(grp["T_train"].iloc[0])
        t_total = int(grp["T_total"].iloc[0])
        is_anom = grp["isAnomaly"].to_numpy(dtype=float)   # fixed; independent of β
        T_test  = max(1, t_total - t_train)
        n_rows  = len(is_anom)
        alert_t = alert_cache.get(sig, [])
        groups.append((sig, t_train, t_total, T_test, is_anom, n_rows, alert_t))

    grid_rows = []
    for beta in BETA_GRID:
        # Compute anomaly scores for this β (reset to 0 at start of test period)
        # Store as list of score arrays, one per sig
        score_arrays: list[np.ndarray] = []
        for (sig, t_train, t_total, T_test, is_anom, n_rows, alert_t) in groups:
            scores = np.empty(n_rows, dtype=float)
            sc = 0.0
            for i in range(n_rows):
                sc = beta * is_anom[i] + (1.0 - beta) * sc
                scores[i] = sc
            score_arrays.append(scores)

        for theta in THETA_GRID:
            prec_list, rec_list, f1_list, far_list = [], [], [], []

            for idx, (sig, t_train, t_total, T_test, is_anom, n_rows, alert_t) in enumerate(groups):
                scores = score_arrays[idx]
                # decisions (test period only — all rows here are test-period rows)
                decisions  = (scores >= theta).astype(int)
                det_ts     = [i for i, d in enumerate(decisions) if d == 1]
                # Convert local index i to absolute t-index
                t_offset   = t_train   # first row in this array is t_train
                det_abs    = [t_offset + i for i in det_ts]
                intervals  = extract_intervals(det_abs)
                n_ivs      = len(intervals)

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

                prec_list.append(precision)
                rec_list.append(recall)
                f1_list.append(f1)
                far_list.append(far)

            grid_rows.append({
                "beta"          : beta,
                "theta"         : theta,
                "mean_precision": round(float(np.mean(prec_list)), 4),
                "mean_recall"   : round(float(np.mean(rec_list)),  4),
                "mean_f1"       : round(float(np.mean(f1_list)),   4),
                "mean_far"      : round(float(np.mean(far_list)),  6),
            })

    return grid_rows


# ──────────────────────────────────────────────────────────────────────────────
# Run grid for all methods
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nRunning {len(BETA_GRID)}×{len(THETA_GRID)} grid for {len(METHODS)} methods ...")
all_rows = []

for method, csv_path in METHODS.items():
    if not os.path.exists(csv_path):
        print(f"  [WARN] {method}: file not found — skipping.")
        continue
    df_res = pd.read_csv(csv_path)
    print(f"  {method} ...", end=" ", flush=True)
    rows = evaluate_grid_for_method(df_res, tau=DEFAULT_TAU)
    for r in rows:
        r["method"] = method
    all_rows.extend(rows)
    print("done.")

grid_df = pd.DataFrame(all_rows)

# ──────────────────────────────────────────────────────────────────────────────
# Identify operating points
# ──────────────────────────────────────────────────────────────────────────────
# Macro-average across all methods for each (beta, theta)
macro = grid_df.groupby(["beta", "theta"])[
    ["mean_precision", "mean_recall", "mean_f1", "mean_far"]
].mean().reset_index()

# High-recall point: baseline
hr_row = macro[(macro["beta"] == BASELINE_BETA) & (macro["theta"] == BASELINE_THETA)]
if hr_row.empty:
    # Closest available
    dist   = (macro["beta"] - BASELINE_BETA).abs() + (macro["theta"] - BASELINE_THETA).abs()
    hr_row = macro.loc[[dist.idxmin()]]

hr_beta  = float(hr_row["beta"].values[0])
hr_theta = float(hr_row["theta"].values[0])
hr_prec  = float(hr_row["mean_precision"].values[0])
hr_rec   = float(hr_row["mean_recall"].values[0])
hr_f1    = float(hr_row["mean_f1"].values[0])
hr_far   = float(hr_row["mean_far"].values[0])

# F1-optimal point
best_idx    = macro["mean_f1"].idxmax()
best_row    = macro.loc[best_idx]
f1_beta     = float(best_row["beta"])
f1_theta    = float(best_row["theta"])
f1_prec     = float(best_row["mean_precision"])
f1_rec      = float(best_row["mean_recall"])
f1_f1       = float(best_row["mean_f1"])
f1_far      = float(best_row["mean_far"])

print("\n" + "═" * 60)
print("OPERATING POINT COMPARISON")
print("═" * 60)
print(f"{'Mode':<22} {'β':>5} {'θ':>6} {'Precision':>10} {'Recall':>8} {'F1':>7} {'FAR':>9}")
print("-" * 60)
print(f"{'High-recall':<22} {hr_beta:>5.2f} {hr_theta:>6.2f} "
      f"{hr_prec:>10.4f} {hr_rec:>8.4f} {hr_f1:>7.4f} {hr_far:>9.5f}")
print(f"{'Balanced / F1-optimal':<22} {f1_beta:>5.2f} {f1_theta:>6.2f} "
      f"{f1_prec:>10.4f} {f1_rec:>8.4f} {f1_f1:>7.4f} {f1_far:>9.5f}")
print("═" * 60)

# Per-method metrics at each operating point
print("\nPer-method breakdown:")
for mode_label, b, t in [("High-recall", hr_beta, hr_theta),
                          ("F1-optimal",  f1_beta, f1_theta)]:
    sub = grid_df[(grid_df["beta"] == b) & (grid_df["theta"] == t)]
    print(f"\n  {mode_label} (β={b}, θ={t}):")
    print(f"  {'Method':<10} {'P':>8} {'R':>8} {'F1':>8} {'FAR':>9}")
    for _, row in sub.iterrows():
        print(f"  {row['method']:<10} {row['mean_precision']:>8.4f} "
              f"{row['mean_recall']:>8.4f} {row['mean_f1']:>8.4f} "
              f"{row['mean_far']:>9.5f}")

# ──────────────────────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)

csv_out  = os.path.join(RESULTS_DIR, "dual_operating_point.csv")
json_out = os.path.join(RESULTS_DIR, "dual_operating_point_summary.json")

grid_df.to_csv(csv_out, index=False)
print(f"\nGrid saved → {csv_out}")

# Build per-method summary at both operating points
summary: dict = {
    "high_recall_mode": {
        "beta": hr_beta, "theta": hr_theta,
        "label": "High-recall (default)",
        "macro": {"precision": hr_prec, "recall": hr_rec, "f1": hr_f1, "far": hr_far},
        "per_method": {}
    },
    "f1_optimal_mode": {
        "beta": f1_beta, "theta": f1_theta,
        "label": "Balanced / F1-optimal",
        "macro": {"precision": f1_prec, "recall": f1_rec, "f1": f1_f1, "far": f1_far},
        "per_method": {}
    },
}
for mode_key, b, t in [("high_recall_mode", hr_beta, hr_theta),
                        ("f1_optimal_mode",  f1_beta, f1_theta)]:
    sub = grid_df[(grid_df["beta"] == b) & (grid_df["theta"] == t)]
    for _, row in sub.iterrows():
        summary[mode_key]["per_method"][row["method"]] = {
            "precision": row["mean_precision"],
            "recall"   : row["mean_recall"],
            "f1"       : row["mean_f1"],
            "far"      : row["mean_far"],
        }

with open(json_out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {json_out}")

# ──────────────────────────────────────────────────────────────────────────────
# Heatmap: one panel per method showing F1 vs (β, θ)
# ──────────────────────────────────────────────────────────────────────────────
print("Generating F1 heatmap ...")
method_list = list(METHODS.keys())
n_methods   = len(method_list)

fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4),
                          gridspec_kw={"width_ratios": [4] * n_methods + [0.3]})

vmin = float(grid_df["mean_f1"].min())
vmax = float(grid_df["mean_f1"].max())
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.get_cmap("YlOrRd")

for ax, method in zip(axes[:-1], method_list):
    sub = grid_df[grid_df["method"] == method]
    pivot = sub.pivot(index="theta", columns="beta", values="mean_f1")
    # theta on y-axis (ascending from top to bottom), beta on x-axis
    pivot = pivot.sort_index(ascending=False)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_title(method, fontsize=11)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{b:.2f}" for b in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{t:.2f}" for t in pivot.index], fontsize=8)
    ax.set_xlabel("β", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("θ", fontsize=9)

    # Annotate each cell with F1 value
    for i, row_idx in enumerate(pivot.index):
        for j, col_idx in enumerate(pivot.columns):
            val = pivot.loc[row_idx, col_idx]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5,
                    color="black" if val < (vmin + vmax) / 2 + (vmax - vmin) * 0.15 else "white")

    # Mark the two operating points
    betas  = list(pivot.columns)
    thetas = list(pivot.index)   # reversed
    for b_op, t_op, marker, color in [(hr_beta, hr_theta, "o", "blue"),
                                       (f1_beta, f1_theta, "*", "lime")]:
        if b_op in betas and t_op in thetas:
            xi = betas.index(b_op)
            yi = thetas.index(t_op)
            ax.plot(xi, yi, marker=marker, markersize=12 if marker == "*" else 8,
                    color=color, markeredgecolor="black", markeredgewidth=0.8,
                    zorder=5)

plt.colorbar(im, cax=axes[-1], label="mean F1")
fig.suptitle("F1 score heatmap — β × θ joint grid\n"
             "● = high-recall (β=0.30, θ=0.30)   ★ = F1-optimal",
             fontsize=10, y=1.02)
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "beta_theta_f1_heatmap.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Heatmap saved → {plot_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Also generate precision–recall trade-off curves along θ for each β
# ──────────────────────────────────────────────────────────────────────────────
print("Generating precision–recall operating-point curve ...")
fig2, ax2 = plt.subplots(figsize=(6, 5))
colors = plt.get_cmap("tab10")
for bi, beta in enumerate(BETA_GRID):
    sub    = macro[macro["beta"] == beta].sort_values("theta")
    precs  = sub["mean_precision"].tolist()
    recs   = sub["mean_recall"].tolist()
    f1s    = sub["mean_f1"].tolist()
    ax2.plot(recs, precs, "o-", color=colors(bi), label=f"β={beta:.2f}", linewidth=1.5)
    # Annotate the θ values at each point
    for rec, prec, theta, f1 in zip(recs, precs, sub["theta"].tolist(), f1s):
        ax2.annotate(f"θ={theta:.2f}", (rec, prec),
                     textcoords="offset points", xytext=(3, 3), fontsize=6,
                     color=colors(bi))

# Mark operating points
ax2.plot(hr_rec, hr_prec, "ob", markersize=10, zorder=6,
         label=f"High-recall (β={hr_beta:.2f},θ={hr_theta:.2f})")
ax2.plot(f1_rec, f1_prec, "g*", markersize=14, zorder=6,
         label=f"F1-optimal (β={f1_beta:.2f},θ={f1_theta:.2f})")

# Iso-F1 curves
for f1_iso in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    prec_iso = np.linspace(f1_iso, 1.0, 200)
    denom    = 2 * prec_iso - f1_iso
    mask     = denom > 0
    rec_iso  = f1_iso * prec_iso[mask] / denom[mask]
    valid    = rec_iso <= 1.0
    ax2.plot(rec_iso[valid], prec_iso[mask][valid], "--", color="lightgray",
             linewidth=0.8, zorder=0)
    if len(rec_iso[valid]) > 0:
        ax2.text(rec_iso[valid][-1], prec_iso[mask][valid][-1],
                 f"F1={f1_iso:.2f}", fontsize=7, color="gray", va="bottom")

ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision–recall trade-off across (β, θ) grid\n"
              "(macro-avg across 4 methods, firefox-android)", fontsize=9)
ax2.legend(fontsize=8, loc="upper right")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(PLOT_DIR, "operating_point_pr_curve.png")
plt.savefig(pr_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"PR curve saved → {pr_path}")

print("\nDone.")
