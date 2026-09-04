# ==============================================================================
# data_scale_audit.py
# Data-Scale Summary Audit.
#
# This script computes and reports exact "large-scale" quantities to support
# the paper's "large-scale" wording:
#
#   - Number of signatures per dataset and combined
#   - Total telemetry data points (sum of series lengths)
#   - Number of alerts mapped per dataset
#   - Detector–signature pairs evaluated (N_sigs × N_methods)
#   - Train / val / test instances under the 60/20/20 protocol
#   - Distribution of series lengths
#   - Coverage of datasets (date ranges)
#
# Both datasets are processed:
#   - Firefox-Android  (342 signatures used in canonical evaluation)
#   - mozilla-beta     (1477 signatures used in canonical evaluation)
#
# Outputs (under Experiments/results/):
#   data_scale_audit.json        — complete audit results
#   data_scale_audit_summary.txt — human-readable report
#   plots/series_length_dist.png — distribution of series lengths
#   plots/alert_coverage.png     — alert coverage per dataset
#
# Usage:
#   python "Experiments/data_scale_audit.py"
# ==============================================================================

import os, sys, re, glob, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
#  Paths
# ==============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(ROOT, "Data")
TS_BASE     = os.path.join(DATA_DIR, "timeseries-data")
ALERTS_CSV  = os.path.join(DATA_DIR, "alerts_data.csv")
BUGS_CSV    = os.path.join(DATA_DIR, "bugs_data.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================================
#  Protocol parameters  (mirrors run_full_evaluation.py)
# ==============================================================================
TRAIN_RATIO  = 0.60
VAL_RATIO    = 0.20
MIN_LEN      = 30        # minimum series length to be usable
ARIMA_MIN_H  = 20        # minimum history for ARIMA

# Number of detectors evaluated in the canonical experiment
N_DETECTORS  = 5         # ARIMA, LAST, SMA, EWMA, Transformer

# Methods evaluated in the calibration baseline (§5.1)
N_DETECTORS_CALIB = 4    # ARIMA, LAST, SMA, EWMA (Transformer separately)

DATASETS = [
    ("firefox-android", "firefox-android", "firefox-android"),
    ("mozilla-beta",    "mozilla-beta",    "mozilla-beta"),
]

# ==============================================================================
#  Utility
# ==============================================================================

def normalize_sig(x):
    if x is None: return ""
    s = str(x).strip()
    if not s or s.lower() == "nan": return ""
    try:
        f = float(s)
        if math.isfinite(f):
            i = int(f)
            if abs(f - i) < 1e-9: return str(i)
    except Exception:
        pass
    return s


# ==============================================================================
#  Data loading for audit
# ==============================================================================

def load_all_sigs(ts_dir, alerts_df, repo_name):
    """
    Load all time-series for a repository; return detailed per-signature info.
    Returns list of dicts with signature_id, T, alerts, date_range, etc.
    """
    repo_alerts = alerts_df[alerts_df["alert_summary_repository"] == repo_name].copy()
    repo_alerts["sig_norm"] = repo_alerts["signature_id"].apply(normalize_sig)
    repo_alerts["push_timestamp"] = pd.to_datetime(
        repo_alerts["push_timestamp"], errors="coerce")
    repo_alerts = repo_alerts.dropna(subset=["push_timestamp"])

    alert_map = {}
    for sig, grp in repo_alerts.groupby("sig_norm"):
        alert_map[sig] = grp["push_timestamp"].tolist()

    csv_files = glob.glob(os.path.join(ts_dir, "*_timeseries_data.csv"))
    sig_to_path = {}
    for path in csv_files:
        m = re.match(r"^(.+)_timeseries_data\.csv$", os.path.basename(path))
        if m:
            sig_to_path[normalize_sig(m.group(1))] = path

    sig_infos = []
    for sig, path in sorted(sig_to_path.items()):
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            if "value" not in df.columns:
                sig_infos.append({"sig": sig, "usable": False, "reason": "no_value_col"})
                continue

            if "push_timestamp" in df.columns:
                df["push_timestamp"] = pd.to_datetime(
                    df["push_timestamp"], errors="coerce")
                df = df.dropna(subset=["push_timestamp"]).sort_values("push_timestamp")
                ts_start = df["push_timestamp"].min()
                ts_end   = df["push_timestamp"].max()
            else:
                ts_start = ts_end = None

            y     = df["value"].values.astype(float)
            T_raw = len(y)
            y     = y[np.isfinite(y)]
            T_fin = len(y)

            usable = T_fin >= MIN_LEN

            # Map alert count
            n_alerts = len(alert_map.get(sig, []))

            # Three-way split sizes
            T_train = T_val = T_test = None
            if usable:
                T_train = max(ARIMA_MIN_H + 2, int(round(TRAIN_RATIO * T_fin)))
                T_val   = max(2, int(round(VAL_RATIO   * T_fin)))
                T_test  = T_fin - T_train - T_val

            sig_infos.append({
                "sig":        sig,
                "usable":     usable,
                "T_raw":      T_raw,
                "T_finite":   T_fin,
                "n_alerts":   n_alerts,
                "T_train":    T_train,
                "T_val":      T_val,
                "T_test":     T_test,
                "ts_start":   ts_start.isoformat() if ts_start is not None and not pd.isna(ts_start) else None,
                "ts_end":     ts_end.isoformat() if ts_end is not None and not pd.isna(ts_end) else None,
            })
        except Exception as e:
            sig_infos.append({"sig": sig, "usable": False, "reason": str(e)})

    return sig_infos, alert_map


# ==============================================================================
#  Compute audit statistics per dataset
# ==============================================================================

def audit_dataset(dataset_name, repo_name, ts_dir, alerts_df):
    print(f"\n  Auditing: {dataset_name} ...")

    sig_infos, alert_map = load_all_sigs(ts_dir, alerts_df, repo_name)
    usable = [s for s in sig_infos if s.get("usable", False)]
    unusable = [s for s in sig_infos if not s.get("usable", False)]

    T_all      = [s["T_finite"]  for s in usable]
    T_train_all= [s["T_train"]   for s in usable if s["T_train"] is not None]
    T_val_all  = [s["T_val"]     for s in usable if s["T_val"]   is not None]
    T_test_all = [s["T_test"]    for s in usable
                  if s["T_test"] is not None and s["T_test"] >= 2]

    valid_split = [s for s in usable if s.get("T_test") is not None and s["T_test"] >= 2]

    total_points   = sum(T_all)
    total_train    = sum(T_train_all)
    total_val      = sum(T_val_all)
    total_test     = sum(T_test_all)
    n_with_alerts  = sum(1 for s in valid_split if s["n_alerts"] > 0)
    total_alerts   = sum(s["n_alerts"] for s in valid_split)
    n_in_valid     = len(valid_split)

    # Date range
    ts_starts = [s["ts_start"] for s in usable if s.get("ts_start")]
    ts_ends   = [s["ts_end"]   for s in usable if s.get("ts_end")]
    date_start = min(ts_starts) if ts_starts else None
    date_end   = max(ts_ends)   if ts_ends   else None

    # Detector-signature pairs
    det_sig_pairs_canonical = n_in_valid * N_DETECTORS

    # Lengths
    lengths_arr = np.array(T_all)

    return {
        "dataset":                   dataset_name,
        "n_csv_files":               len(sig_infos),
        "n_usable_sigs":             len(usable),
        "n_unusable_sigs":           len(unusable),
        "n_valid_split":             n_in_valid,
        "total_telemetry_points":    total_points,
        "split_totals": {
            "total_train_points":    total_train,
            "total_val_points":      total_val,
            "total_test_points":     total_test,
            "sum_all":               total_train + total_val + total_test,
        },
        "series_length": {
            "min":    int(lengths_arr.min()) if len(lengths_arr) > 0 else None,
            "max":    int(lengths_arr.max()) if len(lengths_arr) > 0 else None,
            "mean":   round(float(lengths_arr.mean()), 1) if len(lengths_arr) > 0 else None,
            "median": round(float(np.median(lengths_arr)), 1) if len(lengths_arr) > 0 else None,
            "std":    round(float(lengths_arr.std()), 1) if len(lengths_arr) > 0 else None,
            "p25":    round(float(np.percentile(lengths_arr, 25)), 1) if len(lengths_arr) > 0 else None,
            "p75":    round(float(np.percentile(lengths_arr, 75)), 1) if len(lengths_arr) > 0 else None,
        },
        "alerts": {
            "sigs_with_alerts":      n_with_alerts,
            "sigs_without_alerts":   n_in_valid - n_with_alerts,
            "pct_with_alerts":       round(n_with_alerts / n_in_valid * 100, 1) if n_in_valid > 0 else 0,
            "total_mapped_alerts":   total_alerts,
        },
        "detector_sig_pairs": {
            "n_methods_canonical":   N_DETECTORS,
            "total_pairs_canonical": det_sig_pairs_canonical,
        },
        "date_range": {
            "start": date_start,
            "end":   date_end,
        },
        "usable_sig_list": [s["sig"] for s in valid_split],
        "_length_array":   T_all,
    }


# ==============================================================================
#  Calibration-baseline audit (214 sigs, 70/30, firefox-android)
# ==============================================================================

def audit_calibration_baseline(alerts_df):
    """
    Reproduce the 214-signature calibration baseline audit.
    Counts sigs with T_test >= 2 under a 70/30 split.
    """
    ts_dir    = os.path.join(TS_BASE, "firefox-android")
    sig_infos, _ = load_all_sigs(ts_dir, alerts_df, "firefox-android")
    usable = [s for s in sig_infos if s.get("usable", False)]

    results = []
    for s in usable:
        T = s["T_finite"]
        T_train = max(ARIMA_MIN_H + 2, int(round(0.70 * T)))
        T_test  = T - T_train
        if T_test < 2: continue
        n_alerts = s["n_alerts"]
        results.append({
            "sig": s["sig"],
            "T": T, "T_train": T_train, "T_test": T_test,
            "n_alerts": n_alerts,
        })

    n_sigs        = len(results)
    n_with_alerts = sum(1 for r in results if r["n_alerts"] > 0)
    total_points  = sum(r["T"] for r in results)
    total_train   = sum(r["T_train"] for r in results)
    total_test    = sum(r["T_test"]  for r in results)
    total_alerts  = sum(r["n_alerts"] for r in results)

    return {
        "n_signatures":              n_sigs,
        "n_with_alerts":             n_with_alerts,
        "total_telemetry_points":    total_points,
        "total_train_points":        total_train,
        "total_test_points":         total_test,
        "total_alerts":              total_alerts,
        "det_sig_pairs_4methods":    n_sigs * N_DETECTORS_CALIB,
    }


# ==============================================================================
#  Plotting
# ==============================================================================

def plot_series_length_distribution(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        lengths = r["_length_array"]
        ax.hist(lengths, bins=40, color="#1f77b4", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(lengths), color="red", ls="--", lw=1.5,
                   label=f"Mean = {np.mean(lengths):.0f}")
        ax.axvline(np.median(lengths), color="orange", ls=":", lw=1.5,
                   label=f"Median = {np.median(lengths):.0f}")
        ax.set_xlabel("Series length (T, telemetry points)")
        ax.set_ylabel("Number of signatures")
        ax.set_title(f"Series Length Distribution — {r['dataset']}\n"
                     f"N={r['n_usable_sigs']} sigs, "
                     f"Total={r['total_telemetry_points']:,} pts")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "series_length_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_alert_coverage(all_results):
    datasets  = [r["dataset"]  for r in all_results]
    with_al   = [r["alerts"]["sigs_with_alerts"]    for r in all_results]
    without_al= [r["alerts"]["sigs_without_alerts"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stacked bar: with vs without alerts
    ax = axes[0]
    x  = np.arange(len(datasets))
    b1 = ax.bar(x, with_al,    color="#2ca02c", label="With alerts", alpha=0.85)
    b2 = ax.bar(x, without_al, bottom=with_al, color="#d62728",
                label="Without alerts", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel("Number of signatures")
    ax.set_title("Signature Alert Coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, (wa, tot) in enumerate(zip(with_al, [wa + woa for wa, woa in zip(with_al, without_al)])):
        ax.text(xi, tot + 2, f"{wa}/{tot}\n({wa/tot*100:.0f}%)",
                ha="center", va="bottom", fontsize=9)

    # Detector-signature pairs
    ax = axes[1]
    pairs = [r["detector_sig_pairs"]["total_pairs_canonical"] for r in all_results]
    bars  = ax.bar(datasets, pairs, color=["#1f77b4", "#ff7f0e"][:len(datasets)],
                   alpha=0.85)
    ax.set_ylabel("Detector × Signature pairs")
    ax.set_title(f"Evaluated Detector–Signature Pairs\n({N_DETECTORS} detectors × N signatures)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, pairs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "alert_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_split_breakdown(all_results):
    """Pie/bar showing train/val/test data breakdown per dataset."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        sp = r["split_totals"]
        labels = ["Train (60%)", "Val (20%)", "Test (20%)"]
        sizes  = [sp["total_train_points"], sp["total_val_points"], sp["total_test_points"]]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75)
        ax.set_title(f"Train/Val/Test Split — {r['dataset']}\n"
                     f"Total: {sum(sizes):,} telemetry points\n"
                     f"({r['n_valid_split']} signatures)")
        for autotext in autotexts:
            autotext.set_fontsize(9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "data_split_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ==============================================================================
#  Human-readable report
# ==============================================================================

def write_text_report(all_results, calib_baseline, combined):
    lines = []
    lines.append("=" * 70)
    lines.append("  DATA-SCALE AUDIT REPORT")
    lines.append("  DAET Journal Paper Artifact (May 2026)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("This audit supports the 'large-scale' wording in the paper by")
    lines.append("providing exact counts of signatures, telemetry points,")
    lines.append("detector-signature pairs, and train/val/test instances.")
    lines.append("")

    # ── Per-dataset ────────────────────────────────────────────────────────
    for r in all_results:
        lines.append(f"Dataset: {r['dataset']}")
        lines.append("-" * 50)
        lines.append(f"  CSV files found           : {r['n_csv_files']}")
        lines.append(f"  Usable signatures (T≥30)  : {r['n_usable_sigs']}")
        lines.append(f"  Valid for eval (T_test≥2) : {r['n_valid_split']}")
        lines.append(f"  Unusable                  : {r['n_unusable_sigs']}")
        lines.append("")
        lines.append(f"  Total telemetry points    : {r['total_telemetry_points']:,}")
        lines.append(f"  Series length (min/max)   : {r['series_length']['min']} / {r['series_length']['max']}")
        lines.append(f"  Series length (mean±std)  : {r['series_length']['mean']} ± {r['series_length']['std']}")
        lines.append(f"  Series length (median)    : {r['series_length']['median']}")
        lines.append(f"  Series length (P25/P75)   : {r['series_length']['p25']} / {r['series_length']['p75']}")
        lines.append("")
        lines.append(f"  Sigs with ≥1 alert        : {r['alerts']['sigs_with_alerts']} / "
                     f"{r['n_valid_split']} ({r['alerts']['pct_with_alerts']:.1f}%)")
        lines.append(f"  Total mapped alerts       : {r['alerts']['total_mapped_alerts']}")
        lines.append("")
        sp = r["split_totals"]
        lines.append(f"  60/20/20 split (telemetry points):")
        lines.append(f"    Train (60%)             : {sp['total_train_points']:,}")
        lines.append(f"    Val   (20%)             : {sp['total_val_points']:,}")
        lines.append(f"    Test  (20%)             : {sp['total_test_points']:,}")
        lines.append("")
        lines.append(f"  Detector–sig pairs        : {r['n_valid_split']} sigs × "
                     f"{N_DETECTORS} methods = {r['detector_sig_pairs']['total_pairs_canonical']:,}")
        lines.append(f"  Date range                : {r['date_range']['start'][:10] if r['date_range']['start'] else 'N/A'}"
                     f" – {r['date_range']['end'][:10] if r['date_range']['end'] else 'N/A'}")
        lines.append("")

    # ── Combined ────────────────────────────────────────────────────────────
    lines.append("COMBINED (all datasets, canonical 60/20/20 protocol)")
    lines.append("-" * 50)
    lines.append(f"  Total signatures (eval)   : {combined['total_valid_sigs']}")
    lines.append(f"  Total telemetry points    : {combined['total_telemetry_points']:,}")
    lines.append(f"  Total train points        : {combined['total_train_points']:,}")
    lines.append(f"  Total val points          : {combined['total_val_points']:,}")
    lines.append(f"  Total test points         : {combined['total_test_points']:,}")
    lines.append(f"  Total mapped alerts       : {combined['total_alerts']}")
    lines.append(f"  Total detector-sig pairs  : {combined['total_det_sig_pairs']:,}")
    lines.append("")

    # ── Calibration baseline ────────────────────────────────────────────────
    lines.append("CALIBRATION BASELINE (§5.1: 70/30, Firefox-Android)")
    lines.append("-" * 50)
    cb = calib_baseline
    lines.append(f"  Signatures                : {cb['n_signatures']}")
    lines.append(f"  Sigs with ≥1 alert        : {cb['n_with_alerts']}")
    lines.append(f"  Total telemetry points    : {cb['total_telemetry_points']:,}")
    lines.append(f"  Train points              : {cb['total_train_points']:,}")
    lines.append(f"  Test points               : {cb['total_test_points']:,}")
    lines.append(f"  Total mapped alerts       : {cb['total_alerts']}")
    lines.append(f"  Det–sig pairs (4 methods) : {cb['det_sig_pairs_4methods']}")
    lines.append("")

    # ── LaTeX helper ────────────────────────────────────────────────────────
    lines.append("LATEX TABLE EXCERPT (for paper §4.2 / Table 2 or new table)")
    lines.append("-" * 50)
    lines.append("")
    lines.append(r"\begin{table}[tbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Sigs} & \textbf{Points} & "
                 r"\textbf{Alerts} & \textbf{Det--Sig Pairs} \\")
    lines.append(r"\midrule")
    for r in all_results:
        sigs   = r["n_valid_split"]
        pts    = r["total_telemetry_points"]
        alts   = r["alerts"]["total_mapped_alerts"]
        pairs  = r["detector_sig_pairs"]["total_pairs_canonical"]
        name   = r["dataset"].replace("-", "\\textendash{}")
        lines.append(f"{name} & {sigs:,} & {pts:,} & {alts} & {pairs:,} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Combined & {combined['total_valid_sigs']:,} & "
                 f"{combined['total_telemetry_points']:,} & "
                 f"{combined['total_alerts']} & "
                 f"{combined['total_det_sig_pairs']:,} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Large-scale evaluation dataset summary. "
                 r"Pairs = $N_{\text{sigs}} \times 5$ detectors; "
                 r"alerts are Treeherder-registered performance regressions mapped "
                 r"to the closest telemetry timestamp.}")
    lines.append(r"\label{tab:data_scale}")
    lines.append(r"\end{table}")
    lines.append("")

    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "data_scale_audit_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Text report saved: {path}")
    print(text)
    return text


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("  R3: Data-Scale Audit")
    print("="*70)

    alerts_df = pd.read_csv(ALERTS_CSV)
    print(f"  Loaded alerts_data.csv: {len(alerts_df)} rows")

    # Per-dataset audit
    all_results = []
    for dataset_name, repo_name, ts_subfolder in DATASETS:
        ts_dir = os.path.join(TS_BASE, ts_subfolder)
        result = audit_dataset(dataset_name, repo_name, ts_dir, alerts_df)
        all_results.append(result)

    # Calibration baseline
    calib = audit_calibration_baseline(alerts_df)

    # Combined stats
    combined = {
        "total_valid_sigs":       sum(r["n_valid_split"]  for r in all_results),
        "total_telemetry_points": sum(r["total_telemetry_points"] for r in all_results),
        "total_train_points":     sum(r["split_totals"]["total_train_points"] for r in all_results),
        "total_val_points":       sum(r["split_totals"]["total_val_points"]   for r in all_results),
        "total_test_points":      sum(r["split_totals"]["total_test_points"]  for r in all_results),
        "total_alerts":           sum(r["alerts"]["total_mapped_alerts"]      for r in all_results),
        "total_det_sig_pairs":    sum(r["detector_sig_pairs"]["total_pairs_canonical"] for r in all_results),
        "total_usable_sigs":      sum(r["n_usable_sigs"] for r in all_results),
        "n_datasets":             len(all_results),
    }

    # Plots
    plot_series_length_distribution(all_results)
    plot_alert_coverage(all_results)
    plot_split_breakdown(all_results)

    # Text report
    write_text_report(all_results, calib, combined)

    # Save JSON (strip internal arrays to keep file small)
    json_results = []
    for r in all_results:
        r2 = {k: v for k, v in r.items()
              if k not in ("usable_sig_list", "_length_array")}
        json_results.append(r2)

    output = {
        "datasets":           json_results,
        "calibration_baseline": calib,
        "combined":           combined,
    }
    json_path = os.path.join(RESULTS_DIR, "data_scale_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON saved: {json_path}")

    print("\n  Done — R3 Data-Scale Audit.")


if __name__ == "__main__":
    main()
