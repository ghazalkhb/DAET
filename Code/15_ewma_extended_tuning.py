# ==============================================================================
# 15_ewma_extended_tuning.py
#
# Extend the EWMA tuning grid below alpha=0.05 (adds 0.01, 0.02, 0.03).
# Uses the SAME canonical 60/20/20 chronological split and the SAME pipeline
# logic as run_full_evaluation.py. Only EWMA is re-evaluated; all other method
# results remain unchanged.
#
# REQUIRED procedure (per plan.md §6):
#   1. Tune alpha on the validation set only.
#   2. Evaluate exactly once on the test set using the selected alpha.
#   3. Both datasets (firefox-android, mozilla-beta).
#
# Outputs:
#   results/ewma_extended_tuning.json  — full results for both datasets
#   results/<dataset>/plots/val_tuning_ewma_extended.png  — updated tuning curve
#
# Usage:
#   python Code/15_ewma_extended_tuning.py
# ==============================================================================

import os, re, glob, json, math, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
#  Paths (mirroring run_full_evaluation.py)
# ==============================================================================
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "Data")
TS_BASE    = os.path.join(DATA_DIR, "timeseries-data")
ALERTS_CSV = os.path.join(DATA_DIR, "alerts_data.csv")
RESULTS    = os.path.join(ROOT, "results")

# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = [
    ("firefox-android", "firefox-android", "firefox-android"),
    ("mozilla-beta",    "mozilla-beta",    "mozilla-beta"),
]

# ==============================================================================
#  Shared parameters — MUST match run_full_evaluation.py exactly
# ==============================================================================
BETA              = 0.30
THRESHOLD_ANOMALY = 0.30
K_SIGMA           = 3.0
THR_WINDOW        = 30
MIN_THRESHOLD     = 1e-6
TOL               = 5

ARIMA_MIN_HISTORY = 20
TRAIN_RATIO       = 0.60
VAL_RATIO         = 0.20

# Extended EWMA grid (adds 0.01, 0.02, 0.03 below the previous 0.05 lower bound)
EWMA_ALPHAS_EXTENDED = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# ==============================================================================
#  Utility helpers (identical to run_full_evaluation.py)
# ==============================================================================

def safe_float(x):
    try:
        if x is None: return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


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


def update_score(prev, is_anom):
    return BETA * float(is_anom) + (1.0 - BETA) * float(prev)


def dyn_thr(hist):
    if not hist: return MIN_THRESHOLD
    w = hist[-THR_WINDOW:] if len(hist) > THR_WINDOW else hist
    s = float(np.std(w)) if len(w) >= 2 else 0.0
    return max(MIN_THRESHOLD, K_SIGMA * s)


def extract_intervals(ts):
    if not ts: return []
    out = []; s = ts[0]; p = ts[0]
    for t in ts[1:]:
        if t == p + 1: p = t
        else: out.append((s, p)); s = t; p = t
    out.append((s, p))
    return out


# ==============================================================================
#  Evaluation metrics (identical to run_full_evaluation.py)
# ==============================================================================

def seg_metrics(decisions, T_offset, alert_t_abs):
    T_seg   = len(decisions)
    alert_t = sorted(ta for ta in alert_t_abs
                     if T_offset <= ta < T_offset + T_seg)
    det_abs = [T_offset + i for i, d in enumerate(decisions) if d == 1]
    ivs     = extract_intervals(det_abs)
    n_ivs   = len(ivs)

    tp_iv = sum(1 for (s, e) in ivs
                if any((s - TOL) <= ta <= (e + TOL) for ta in alert_t))
    fp_iv = n_ivs - tp_iv
    precision = tp_iv / n_ivs if n_ivs > 0 else 0.0

    tp_al, delays = 0, []
    for ta in alert_t:
        for (s, e) in ivs:
            if (s - TOL) <= ta <= (e + TOL):
                tp_al += 1
                delays.append(s - ta)
                break
    recall = tp_al / len(alert_t) if alert_t else 0.0
    f1     = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    far    = fp_iv / T_seg if T_seg > 0 else 0.0
    stor   = 1.0 - sum(decisions) / T_seg if T_seg > 0 else 0.0
    m_del  = float(np.mean(delays)) if delays else None

    return dict(
        n_alerts=len(alert_t), has_alert=len(alert_t) > 0, detected=tp_al > 0,
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
        far=round(far, 6), storage_reduc=round(stor, 4),
        tp_iv=tp_iv, fp_iv=fp_iv, tp_al=tp_al, fn_al=len(alert_t) - tp_al,
        mean_delay=round(m_del, 2) if m_del is not None else None,
        n_on=sum(decisions),
    )


# ==============================================================================
#  EWMA warmup (identical logic to run_full_evaluation.py)
# ==============================================================================

def ewma_warmup(y_train, alpha):
    """Warm up EWMA state on training set. Returns (delta_hist, score, ewma_val)."""
    dh, sc = [], 0.0
    ewma = float(y_train[0]) if len(y_train) > 0 else 0.0
    min_h = 2  # EWMA needs at least 2 points

    for t in range(len(y_train)):
        act = safe_float(y_train[t])
        if t < min_h or act is None:
            if t > 0 and act is not None:
                ewma = alpha * act + (1 - alpha) * ewma
            continue
        pr   = ewma
        ewma = alpha * act + (1 - alpha) * ewma
        delta = abs(act - pr)
        is_a  = 1 if delta > dyn_thr(dh) else 0
        sc    = update_score(sc, is_a)
        dh.append(delta)

    return dh, sc, ewma


def ewma_run_segment(y_history, y_segment, alpha, init_dh, init_sc, init_ewma):
    """Run EWMA on a segment given initial state. Returns (decisions, dh, sc, ewma)."""
    dh, sc, ewma = list(init_dh), init_sc, init_ewma
    prev = float(y_history[-1]) if len(y_history) > 0 else float(y_segment[0])
    decisions = []
    for i_t, val in enumerate(y_segment):
        act = safe_float(val)
        if act is None:
            decisions.append(0)
            continue
        pr   = ewma
        ewma = alpha * act + (1 - alpha) * ewma
        delta = abs(act - pr)
        is_a  = 1 if delta > dyn_thr(dh) else 0
        sc    = update_score(sc, is_a)
        dh.append(delta)
        decisions.append(1 if sc >= THRESHOLD_ANOMALY else 0)
    return decisions, dh, sc, ewma


# ==============================================================================
#  Data loading (identical to run_full_evaluation.py)
# ==============================================================================

def load_timeseries_for_dataset(ts_dir, alerts_df, repo_name):
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

    results = []
    for sig, path in sorted(sig_to_path.items()):
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            if "value" not in df.columns: continue
            if "push_timestamp" in df.columns:
                df["push_timestamp"] = pd.to_datetime(
                    df["push_timestamp"], errors="coerce")
                df = df.dropna(subset=["push_timestamp"]).sort_values("push_timestamp")
            elif "push_id" in df.columns:
                df = df.sort_values("push_id")
            y = df["value"].values.astype(float)
            y = y[np.isfinite(y)]
            if len(y) < 30: continue

            at_list = []
            if sig in alert_map:
                ts_col = df["push_timestamp"] if "push_timestamp" in df.columns else None
                if ts_col is not None:
                    ts_s = pd.Series(ts_col.values)
                    valid = ts_s.dropna()
                    for at in alert_map[sig]:
                        try:
                            if pd.isna(at): continue
                            idx = int((valid - at).abs().idxmin())
                            at_list.append(idx)
                        except Exception:
                            pass
                    at_list = sorted(set(at_list))

            results.append((sig, y, at_list))
        except Exception:
            pass

    results.sort(key=lambda x: -len(x[1]))
    return results


# ==============================================================================
#  Per-dataset EWMA-only evaluation
# ==============================================================================

def evaluate_ewma_extended(dataset_name, repo_name, ts_dir):
    print(f"\n{'='*70}")
    print(f"  EWMA Extended Tuning: {dataset_name}  (repo={repo_name})")
    print(f"{'='*70}")

    alerts_df = pd.read_csv(ALERTS_CSV)
    sigs_data = load_timeseries_for_dataset(ts_dir, alerts_df, repo_name)
    print(f"  Loaded {len(sigs_data)} valid signatures (len >= 30)")

    if not sigs_data:
        print("  ERROR: No data. Skipping.")
        return None

    # Compute splits (identical logic to run_full_evaluation.py)
    splits = {}
    for sig, y, at_list in sigs_data:
        T       = len(y)
        T_train = max(ARIMA_MIN_HISTORY + 2, int(round(TRAIN_RATIO * T)))
        T_val   = max(2, int(round(VAL_RATIO * T)))
        T_test  = T - T_train - T_val
        if T_test < 2: continue
        splits[sig] = (y[:T_train],
                       y[T_train: T_train + T_val],
                       y[T_train + T_val:],
                       T_train, T_val, T_test)

    alert_t_map = {sig: at for sig, _, at in sigs_data}
    valid_sigs  = [(sig, y) for sig, y, _ in sigs_data if sig in splits]
    print(f"  Valid signatures (T_test >= 2): {len(splits)}")

    # --- Validation tuning for extended EWMA grid ---
    print(f"\n  Tuning EWMA on validation set (extended grid) ...")
    ewma_vf1 = {}
    for alpha in EWMA_ALPHAS_EXTENDED:
        f1s = []
        for sig, y in valid_sigs:
            y_train, y_val, _, T_train, _, _ = splits[sig]
            dh0, sc0, ew0 = ewma_warmup(y_train, alpha)
            dec, _, _, _  = ewma_run_segment(y_train, y_val, alpha, dh0, sc0, ew0)
            f1s.append(seg_metrics(dec, T_train, alert_t_map.get(sig, []))["f1"])
        ewma_vf1[alpha] = float(np.mean(f1s)) if f1s else 0.0
        print(f"    alpha={alpha:.2f}: val F1 = {ewma_vf1[alpha]:.4f}")

    best_alpha = max(ewma_vf1, key=ewma_vf1.get)
    print(f"\n  >>> Best EWMA alpha = {best_alpha}  "
          f"(val F1 = {ewma_vf1[best_alpha]:.4f})")

    # --- Test-set evaluation with the selected alpha ---
    print(f"\n  Evaluating on test set with alpha = {best_alpha} ...")
    test_rows = []
    for sig, y in valid_sigs:
        y_train, y_val, y_test, T_train, T_val, T_test = splits[sig]
        # 1. Warm up on train
        dh0, sc0, ew0 = ewma_warmup(y_train, best_alpha)
        # 2. Advance state through val (don't tune, just propagate state)
        _, dh1, sc1, ew1 = ewma_run_segment(y_train, y_val, best_alpha, dh0, sc0, ew0)
        # 3. Evaluate on test
        dec_test, _, _, _ = ewma_run_segment(
            np.concatenate([y_train, y_val]), y_test,
            best_alpha, dh1, sc1, ew1)
        m = seg_metrics(dec_test, T_train + T_val, alert_t_map.get(sig, []))
        test_rows.append(m)

    precision_vals = [r["precision"] for r in test_rows]
    recall_vals    = [r["recall"]    for r in test_rows]
    f1_vals        = [r["f1"]        for r in test_rows]
    far_vals       = [r["far"]       for r in test_rows]
    stor_vals      = [r["storage_reduc"] for r in test_rows]

    has_alert_mask = [r["has_alert"] for r in test_rows]
    detected_mask  = [r["detected"]  for r in test_rows]
    alerted_detected = [d for h, d in zip(has_alert_mask, detected_mask) if h]
    det_rate = np.mean(alerted_detected) * 100 if alerted_detected else 0.0

    test_summary = {
        "n_signatures":       len(test_rows),
        "n_with_alerts":      sum(has_alert_mask),
        "mean_precision":     round(np.mean(precision_vals), 4),
        "mean_recall":        round(np.mean(recall_vals),    4),
        "mean_f1":            round(np.mean(f1_vals),        4),
        "mean_far":           round(np.mean(far_vals),       6),
        "mean_storage_reduc": round(np.mean(stor_vals),      4),
        "detection_rate_pct": round(det_rate,                2),
    }

    print(f"\n  Test results (alpha={best_alpha}):")
    for k, v in test_summary.items():
        print(f"    {k}: {v}")

    # --- Save tuning plot ---
    ds_plots = os.path.join(RESULTS, dataset_name, "plots")
    os.makedirs(ds_plots, exist_ok=True)
    alphas_list = list(ewma_vf1.keys())
    f1s_list    = list(ewma_vf1.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas_list, f1s_list, "o-", color="#1f77b4", lw=2, markersize=7)
    ax.axvline(best_alpha, color="#d62728", ls="--", lw=1.5,
               label=f"Best = {best_alpha}  (F1={ewma_vf1[best_alpha]:.4f})")
    ax.fill_between(alphas_list, f1s_list,
                    min(f1s_list) * 0.98, alpha=0.12, color="#1f77b4")
    ax.set_xlabel("EWMA alpha", fontsize=11)
    ax.set_ylabel("Mean Validation F1", fontsize=11)
    ax.set_title(f"EWMA Alpha Tuning (Extended Grid) — {dataset_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(ds_plots, "val_tuning_ewma_extended.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Tuning plot saved: {plot_path}")

    return {
        "dataset":         dataset_name,
        "best_alpha":      best_alpha,
        "val_tuning_f1":   {str(k): round(v, 4) for k, v in ewma_vf1.items()},
        "test_results":    test_summary,
    }


# ==============================================================================
#  Main
# ==============================================================================

if __name__ == "__main__":
    t_start = time.time()
    all_results = {}

    for ds_name, repo_name, ts_sub in DATASETS:
        ts_dir = os.path.join(TS_BASE, ts_sub)
        res = evaluate_ewma_extended(ds_name, repo_name, ts_dir)
        if res:
            all_results[ds_name] = res

    # Save full output
    out_path = os.path.join(RESULTS, "ewma_extended_tuning.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Print comparison summary
    print(f"\n{'='*70}")
    print("  SUMMARY: Extended EWMA Tuning Results")
    print(f"{'='*70}")

    ORIGINAL = {
        "firefox-android": {"alpha": 0.05, "val_f1": 0.0978,
                             "P": 0.0676, "R": 0.2632, "F1": 0.1033,
                             "FAR": 0.052155, "det_pct": 67.65},
        "mozilla-beta":    {"alpha": 0.05, "val_f1": 0.1453,
                             "P": 0.057,  "R": 0.2363, "F1": 0.083,
                             "FAR": 0.059159, "det_pct": 90.63},
    }

    for ds_name, res in all_results.items():
        orig = ORIGINAL.get(ds_name, {})
        new  = res["test_results"]
        old_a = orig.get("alpha", "?")
        new_a = res["best_alpha"]
        changed = (new_a != old_a)
        print(f"\n  {ds_name}:")
        print(f"    Previous best alpha : {old_a}  "
              f"(val F1 = {orig.get('val_f1','?')})")
        print(f"    New best alpha      : {new_a}  "
              f"(val F1 = {res['val_tuning_f1'].get(str(new_a),'?')})")
        print(f"    Alpha changed?      : {'YES — UPDATE PAPER' if changed else 'NO — alpha=0.05 confirmed'}")
        print(f"    Test P/R/F1/FAR/Det : "
              f"{new['mean_precision']}/{new['mean_recall']}/"
              f"{new['mean_f1']}/{new['mean_far']}/{new['detection_rate_pct']}")
        if changed:
            print(f"    [!] Old test results: P={orig.get('P')}, R={orig.get('R')}, "
                  f"F1={orig.get('F1')}, FAR={orig.get('FAR')}, "
                  f"Det={orig.get('det_pct')}")

    print(f"\n  Total time: {time.time()-t_start:.0f}s")
